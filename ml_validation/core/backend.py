"""Dual-mode Checkpoint Loader, Forward Inference Engine, and Grad-CAM Explainer."""
from __future__ import annotations

import hashlib
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .preprocessing import eval_crop_box, preprocess_pil, preprocessing_from_checkpoint

SUPPORTED_MODELS = ("resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2")
REQUIRED_CHECKPOINT_KEYS = ("model_state", "model_name", "class_names")
COMMITTED_CHECKPOINT_SHA256 = "f456fe4cd6042b2c031172bcdab00a40c845475a3f71fa052264d28311ed0cc9"

try:
    import torch
    import torchvision  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pretty_class_name(name: str) -> str:
    """Convert 'Corn_(maize)___Common_rust_' -> 'Corn (maize) — Common rust'."""
    crop, _, condition = name.partition("___")

    def tidy(text: str) -> str:
        text = " ".join(text.replace("_", " ").split())
        return text[:1].upper() + text[1:]

    return f"{tidy(crop)} — {tidy(condition)}" if condition else tidy(crop)


def parse_crop_and_disease(name: str) -> Tuple[str, str, bool]:
    """Return (crop_species, condition_name, is_healthy) from a raw class name."""
    crop, _, condition = name.partition("___")
    crop_clean = " ".join(crop.replace("_", " ").split()).strip()
    cond_clean = " ".join(condition.replace("_", " ").split()).strip() if condition else "Unknown"
    is_healthy = cond_clean.lower() == "healthy"
    return crop_clean, cond_clean, is_healthy


class _TorchStorageRef:
    def __init__(self, storage_type: str, key: str, location: str, size: int, data: bytes):
        self.storage_type = storage_type
        self.key = key
        self.location = location
        self.size = size
        self.data = data


def _rebuild_tensor_v2(storage: _TorchStorageRef, storage_offset: int, size: tuple,
                       stride: tuple, requires_grad: bool, backward_hooks: object,
                       metadata: object = None) -> np.ndarray:
    dtype_map = {
        "FloatStorage": np.float32,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "DoubleStorage": np.float64,
        "HalfStorage": np.float16,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
    }
    if storage.storage_type not in dtype_map:
        raise ValueError(f"Unsupported tensor storage type: {storage.storage_type}")
    dt = dtype_map[storage.storage_type]
    arr = np.frombuffer(storage.data, dtype=dt)
    if len(size) == 0:
        return np.array(arr[storage_offset], dtype=dt)
    total = int(np.prod(size))
    sub = arr[storage_offset:storage_offset + total]
    return sub.reshape(size).copy()


class _SafeTorchZipUnpickler(pickle.Unpickler):
    """Restricted Unpickler that refuses arbitrary classes, matching torch.load(weights_only=True)."""

    ALLOWED_GLOBALS = {
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
        ("collections", "OrderedDict"): OrderedDict,
    }
    ALLOWED_TORCH_STORAGES = {
        "FloatStorage", "LongStorage", "IntStorage", "DoubleStorage",
        "HalfStorage", "ByteStorage", "BoolStorage",
    }

    def __init__(self, file, zip_file: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.zip_file = zip_file
        self.prefix = prefix
        self.storages: Dict[str, _TorchStorageRef] = {}

    def find_class(self, module: str, name: str):
        if (module, name) in self.ALLOWED_GLOBALS:
            return self.ALLOWED_GLOBALS[(module, name)]
        if module == "torch" and name in self.ALLOWED_TORCH_STORAGES:
            return name
        raise ValueError(
            f"Weights-only safe unpickler refused unsupported global: {module}.{name}"
        )

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            _, storage_type, key, location, size = pid[:5]
            if key not in self.storages:
                data_path = f"{self.prefix}/data/{key}" if self.prefix else f"data/{key}"
                raw = self.zip_file.read(data_path)
                self.storages[key] = _TorchStorageRef(storage_type, str(key), str(location), int(size), raw)
            return self.storages[key]
        raise ValueError(f"Unsupported persistent_load ID in checkpoint: {pid!r}")


def load_checkpoint_safe(path: str | Path) -> dict:
    """Load and validate a checkpoint dict safely (via PyTorch weights_only=True or SafeZipUnpickler)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    if HAS_TORCH:
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"Could not safely load checkpoint {path} with weights_only=True: {exc}") from exc
    else:
        try:
            with zipfile.ZipFile(path, "r") as z:
                names = z.namelist()
                data_pkls = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
                if not data_pkls:
                    raise ValueError(f"{path} is not a valid PyTorch zip archive (missing data.pkl)")
                prefix = data_pkls[0].rsplit("/data.pkl", 1)[0] if "/" in data_pkls[0] else ""
                with z.open(data_pkls[0]) as f:
                    ckpt = _SafeTorchZipUnpickler(f, z, prefix=prefix).load()
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Could not load checkpoint {path}: not a valid zip archive") from exc

    if not isinstance(ckpt, dict):
        raise ValueError(f"{path} is not a checkpoint dict (got {type(ckpt).__name__})")
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path} is missing checkpoint keys: {missing}")
    if ckpt["model_name"] not in SUPPORTED_MODELS:
        raise ValueError(f"{path} has unsupported model_name {ckpt['model_name']!r}")
    if not isinstance(ckpt["class_names"], list) or len(ckpt["class_names"]) == 0:
        raise ValueError(f"{path} has empty or invalid class_names")

    np_state = OrderedDict()
    for k, v in ckpt["model_state"].items():
        if HAS_TORCH and isinstance(v, torch.Tensor):
            np_state[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            np_state[k] = v
        else:
            np_state[k] = np.asarray(v)
    ckpt["numpy_state"] = np_state

    num_classes = len(ckpt["class_names"])
    head_key = "fc.weight" if ckpt["model_name"] in ("resnet18", "resnet50") else "classifier.1.weight"
    if head_key in np_state:
        head_out = np_state[head_key].shape[0]
        if head_out != num_classes:
            raise RuntimeError(
                f"Error(s) in loading state_dict: size mismatch for {head_key}: "
                f"checkpoint head has {head_out} outputs but class_names has {num_classes} classes."
            )
    return ckpt


def _varint(val: int) -> bytes:
    out = bytearray()
    while val > 0x7F:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _field_varint(field_num: int, val: int) -> bytes:
    return _varint((field_num << 3) | 0) + _varint(val)


def _field_bytes(field_num: int, data: bytes | str) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _varint((field_num << 3) | 2) + _varint(len(data)) + data


def _make_tensor_proto(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    dt = 1 if arr.dtype == np.float32 else 7
    buf = bytearray()
    for d in arr.shape:
        buf += _field_varint(1, int(d))
    buf += _field_varint(2, dt)
    buf += _field_bytes(8, name)
    buf += _field_bytes(9, arr.tobytes())
    return bytes(buf)


def _make_value_info(name: str, elem_type: int, shape) -> bytes:
    dims_buf = bytearray()
    for d in shape:
        dim_msg = _field_bytes(2, d) if isinstance(d, str) else _field_varint(1, int(d))
        dims_buf += _field_bytes(1, bytes(dim_msg))
    tensor_type_msg = _field_varint(1, elem_type) + _field_bytes(2, bytes(dims_buf))
    type_msg = _field_bytes(1, tensor_type_msg)
    return _field_bytes(1, name) + _field_bytes(2, type_msg)


def _make_attr_ints(name: str, vals) -> bytes:
    buf = bytearray(_field_bytes(1, name))
    for v in vals:
        buf += _field_varint(8, int(v))
    buf += _field_varint(20, 7)
    return bytes(buf)


def _make_attr_int(name: str, val: int) -> bytes:
    return _field_bytes(1, name) + _field_varint(3, int(val)) + _field_varint(20, 2)


def _make_node(op_type: str, inputs, outputs, name: str = "", attrs=()) -> bytes:
    buf = bytearray()
    for i in inputs:
        buf += _field_bytes(1, i)
    for o in outputs:
        buf += _field_bytes(2, o)
    if name:
        buf += _field_bytes(3, name)
    buf += _field_bytes(4, op_type)
    for a in attrs:
        buf += _field_bytes(5, a)
    return bytes(buf)


def build_resnet18_onnx_bytes(np_state: Dict[str, np.ndarray], img_size: int = 224) -> bytes:
    """Compile a ResNet18 state_dict into an ONNX ModelProto byte stream (opset 13)."""
    initializers = []
    nodes = []

    def add_folded_conv(conv_prefix: str, bn_prefix: str, inp: str, out_name: str,
                        kernel: int, stride: int, pad: int, eps: float = 1e-5):
        w = np_state[f"{conv_prefix}.weight"].astype(np.float32)
        bn_w = np_state[f"{bn_prefix}.weight"].astype(np.float32)
        bn_b = np_state[f"{bn_prefix}.bias"].astype(np.float32)
        rm = np_state[f"{bn_prefix}.running_mean"].astype(np.float32)
        rv = np_state[f"{bn_prefix}.running_var"].astype(np.float32)
        scale = bn_w / np.sqrt(rv + eps)
        w_f = (w * scale[:, None, None, None]).astype(np.float32)
        b_f = (bn_b - rm * scale).astype(np.float32)
        w_name, b_name = f"{conv_prefix}_w", f"{conv_prefix}_b"
        initializers.append(_make_tensor_proto(w_name, w_f))
        initializers.append(_make_tensor_proto(b_name, b_f))
        attrs = [
            _make_attr_ints("kernel_shape", [kernel, kernel]),
            _make_attr_ints("strides", [stride, stride]),
            _make_attr_ints("pads", [pad, pad, pad, pad]),
        ]
        nodes.append(_make_node("Conv", [inp, w_name, b_name], [out_name], name=conv_prefix, attrs=attrs))

    add_folded_conv("conv1", "bn1", "input", "stem_conv", kernel=7, stride=2, pad=3)
    nodes.append(_make_node("Relu", ["stem_conv"], ["stem_relu"]))
    nodes.append(_make_node("MaxPool", ["stem_relu"], ["stem_pool"], attrs=[
        _make_attr_ints("kernel_shape", [3, 3]),
        _make_attr_ints("strides", [2, 2]),
        _make_attr_ints("pads", [1, 1, 1, 1]),
    ]))

    curr = "stem_pool"
    for layer_idx in range(1, 5):
        for block_idx in range(2):
            p = f"layer{layer_idx}.{block_idx}"
            stride = 2 if (layer_idx > 1 and block_idx == 0) else 1
            has_ds = f"{p}.downsample.0.weight" in np_state
            c1_out, r1_out, c2_out = f"{p}_c1", f"{p}_r1", f"{p}_c2"
            add_folded_conv(f"{p}.conv1", f"{p}.bn1", curr, c1_out, kernel=3, stride=stride, pad=1)
            nodes.append(_make_node("Relu", [c1_out], [r1_out]))
            add_folded_conv(f"{p}.conv2", f"{p}.bn2", r1_out, c2_out, kernel=3, stride=1, pad=1)
            if has_ds:
                ds_out = f"{p}_ds"
                add_folded_conv(f"{p}.downsample.0", f"{p}.downsample.1", curr, ds_out, kernel=1, stride=stride, pad=0)
                residual = ds_out
            else:
                residual = curr
            add_out, blk_out = f"{p}_add", f"{p}_out"
            nodes.append(_make_node("Add", [c2_out, residual], [add_out]))
            nodes.append(_make_node("Relu", [add_out], [blk_out]))
            curr = blk_out

    nodes.append(_make_node("GlobalAveragePool", [curr], ["gap_out"]))
    nodes.append(_make_node("Flatten", ["gap_out"], ["flat_out"], attrs=[_make_attr_int("axis", 1)]))
    fc_w = np_state["fc.weight"].astype(np.float32)
    fc_b = np_state["fc.bias"].astype(np.float32)
    num_classes = fc_w.shape[0]
    initializers.append(_make_tensor_proto("fc_w", fc_w))
    initializers.append(_make_tensor_proto("fc_b", fc_b))
    nodes.append(_make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["logits"], attrs=[_make_attr_int("transB", 1)]))

    feat_spatial = max(1, img_size // 32)
    graph_buf = bytearray()
    for n in nodes:
        graph_buf += _field_bytes(1, n)
    graph_buf += _field_bytes(2, "resnet18_plant_disease")
    for init in initializers:
        graph_buf += _field_bytes(5, init)
    graph_buf += _field_bytes(11, _make_value_info("input", 1, ["B", 3, img_size, img_size]))
    graph_buf += _field_bytes(12, _make_value_info("logits", 1, ["B", num_classes]))
    graph_buf += _field_bytes(12, _make_value_info("layer4.1_out", 1, ["B", 512, feat_spatial, feat_spatial]))

    opset_msg = _field_bytes(1, "") + _field_varint(2, 13)
    model_buf = _field_varint(1, 8) + _field_bytes(7, bytes(graph_buf)) + _field_bytes(8, opset_msg)
    return bytes(model_buf)


class CheckpointModelAdapter:
    """Unified wrapper around a loaded Plant Disease Classifier checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", backend: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        self.sha256 = file_sha256(self.checkpoint_path)
        self.file_size_bytes = self.checkpoint_path.stat().st_size
        self.ckpt = load_checkpoint_safe(self.checkpoint_path)
        self.model_name: str = self.ckpt["model_name"]
        self.class_names: List[str] = list(self.ckpt["class_names"])
        self.num_classes: int = len(self.class_names)
        self.preprocessing: Dict[str, object] = preprocessing_from_checkpoint(self.ckpt)
        self.np_state: Dict[str, np.ndarray] = self.ckpt["numpy_state"]
        self.device = device

        if backend == "auto":
            self.backend = "torch" if HAS_TORCH else "onnx"
        else:
            self.backend = backend

        self.torch_model = None
        self.ort_session = None
        self.onnx_bytes: Optional[bytes] = None

        if self.backend == "torch":
            if not HAS_TORCH:
                raise RuntimeError("Requested backend='torch', but PyTorch is not installed.")
            self._init_torch_model()
        elif self.backend == "onnx":
            self._init_onnx_model()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _init_torch_model(self):
        from torchvision import models
        import torch.nn as nn

        name = self.model_name.lower()
        if name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        elif name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Unknown model_name: {name}")

        model.load_state_dict(self.ckpt["model_state"], strict=True)
        dev = torch.device(self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.torch_model = model.to(dev).eval()
        self.device = str(dev)

    def _init_onnx_model(self):
        if self.model_name != "resnet18":
            raise ValueError(f"ONNX Runtime fallback supports resnet18; got {self.model_name}")
        img_size = int(self.preprocessing["img_size"])
        self.onnx_bytes = build_resnet18_onnx_bytes(self.np_state, img_size=img_size)
        if not HAS_ORT:
            raise RuntimeError("Neither torch nor onnxruntime is installed.")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(
            self.onnx_bytes, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

    def parameter_summary(self) -> Dict[str, object]:
        """Compute exact parameter and buffer counts from the checkpoint state_dict."""
        learnable_params = 0
        backbone_params = 0
        head_params = 0
        buffer_elements = 0
        bn_layers = 0
        bn_batches_tracked = []

        for k, arr in self.np_state.items():
            if k.endswith("num_batches_tracked"):
                bn_layers += 1
                bn_batches_tracked.append(int(arr.item()))
                buffer_elements += 1
            elif k.endswith("running_mean") or k.endswith("running_var"):
                buffer_elements += int(arr.size)
            else:
                n = int(arr.size)
                learnable_params += n
                if k.startswith("fc.") or k.startswith("classifier."):
                    head_params += n
                else:
                    backbone_params += n

        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "state_dict_keys": len(self.np_state),
            "total_learnable_params": learnable_params,
            "backbone_learnable_params": backbone_params,
            "head_learnable_params": head_params,
            "total_buffer_elements": buffer_elements,
            "num_batchnorm_layers": bn_layers,
            "bn_num_batches_tracked_unique": sorted(set(bn_batches_tracked)),
            "checkpoint_file_size_bytes": self.file_size_bytes,
            "checkpoint_file_size_mb": round(self.file_size_bytes / (1024 * 1024), 4),
            "checkpoint_sha256": self.sha256,
            "has_v2_metadata": "format_version" in self.ckpt,
            "preprocessing": self.preprocessing,
        }

    def forward_logits(self, batch_nchw: np.ndarray) -> np.ndarray:
        """Run forward pass on a normalized float32 batch (B, 3, H, W) and return logits (B, C)."""
        batch_nchw = np.ascontiguousarray(batch_nchw, dtype=np.float32)
        if self.backend == "torch":
            with torch.no_grad():
                t_in = torch.from_numpy(batch_nchw).to(self.device)
                out = self.torch_model(t_in)
                return out.detach().cpu().numpy()
        else:
            logits = self.ort_session.run(["logits"], {"input": batch_nchw})[0]
            return logits

    def forward_logits_and_cam(
        self, single_nchw: np.ndarray, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Compute (logits [1, C], cam [h, w] in [0, 1], target_class) for a single image (1, 3, H, W)."""
        if single_nchw.ndim != 4 or single_nchw.shape[0] != 1:
            raise ValueError(f"Expected a single image batch (1, 3, H, W), got {single_nchw.shape}")
        single_nchw = np.ascontiguousarray(single_nchw, dtype=np.float32)

        if self.backend == "torch":
            target_layer = (
                self.torch_model.layer4[-1]
                if self.model_name in ("resnet18", "resnet50")
                else self.torch_model.features[-1]
            )
            activations = []
            handle = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
            try:
                t_in = torch.from_numpy(single_nchw).to(self.device).requires_grad_(True)
                with torch.enable_grad():
                    out = self.torch_model(t_in)
                    if target_class is None:
                        target_class = int(out.argmax(dim=1).item())
                    (grads,) = torch.autograd.grad(out[0, target_class], activations[0])
                act = activations[0].detach()
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam_t = torch.relu((weights * act).sum(dim=1, keepdim=True))[0, 0]
                cam = cam_t.cpu().numpy()
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                return out.detach().cpu().numpy(), cam, int(target_class)
            finally:
                handle.remove()
        else:
            logits, feat = self.ort_session.run(["logits", "layer4.1_out"], {"input": single_nchw})
            if target_class is None:
                target_class = int(logits[0].argmax())
            weights = self.np_state["fc.weight"][target_class]
            cam = np.tensordot(weights, feat[0], axes=([0], [0])).astype(np.float32)
            cam = np.maximum(cam, 0.0)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            return logits, cam, int(target_class)

    @staticmethod
    def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute numerically stable softmax probabilities along axis=-1."""
        scaled = logits / max(float(temperature), 1e-8)
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict_pil(
        self, image: Image.Image, top_k: int = 5, with_gradcam: bool = True, apply_exif: bool = True
    ) -> Dict[str, object]:
        """Run full evaluation preprocessing, inference, and optional Grad-CAM on a PIL image."""
        img_rgb = (ImageOps.exif_transpose(image) if apply_exif else image).convert("RGB")
        x = preprocess_pil(img_rgb, self.preprocessing, apply_exif=False)[None, ...]

        if with_gradcam:
            logits, cam, target_idx = self.forward_logits_and_cam(x)
        else:
            logits = self.forward_logits(x)
            cam = None
            target_idx = int(logits[0].argmax())

        probs = self.softmax(logits)[0]
        order = np.argsort(-probs)[: min(top_k, self.num_classes)]
        top_preds = [(self.class_names[int(i)], float(probs[int(i)]), int(i)) for i in order]

        pp = self.preprocessing
        box = tuple(
            round(v)
            for v in eval_crop_box(img_rgb.width, img_rgb.height, int(pp["resize_size"]), int(pp["img_size"]))
        )
        model_view = img_rgb.crop(box).resize((int(pp["img_size"]), int(pp["img_size"])), Image.BILINEAR)
        overlay = overlay_heatmap_pil(model_view, cam) if cam is not None else None

        return {
            "logits": logits[0],
            "probs": probs,
            "pred_idx": int(order[0]),
            "pred_class": self.class_names[int(order[0])],
            "pred_prob": float(probs[int(order[0])]),
            "margin": float(probs[int(order[0])] - (probs[int(order[1])] if len(order) > 1 else 0.0)),
            "top_k": top_preds,
            "cam": cam,
            "target_idx": target_idx,
            "crop_box": box,
            "model_view": model_view,
            "overlay": overlay,
        }


def overlay_heatmap_pil(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Resize a [0, 1] Grad-CAM heatmap to `pil_image.size` and blend using COLORMAP_JET."""
    img = np.asarray(pil_image.convert("RGB"))
    cam_resized = cv2.resize(cam.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(cam_resized, 0.0, 1.0)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1.0 - alpha) * img).astype(np.uint8)
    return Image.fromarray(overlay)
