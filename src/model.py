"""Builds a pretrained torchvision backbone with a fresh classification head.

Supports resnet18, resnet50, efficientnet_b0, mobilenet_v2 — chosen for a spread
of accuracy/speed/size trade-offs worth comparing in the README's "extensions" section.
"""
import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ("resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2")
REQUIRED_CHECKPOINT_KEYS = ("model_state", "model_name", "class_names")


def get_model(model_name: str = "resnet18", num_classes: int = 38, pretrained: bool = True):
    """`pretrained=True` downloads ImageNet weights; use False when a checkpoint will
    overwrite every weight anyway (inference/evaluation)."""
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model, backbone_params


def set_backbone_trainable(backbone_params, trainable: bool) -> None:
    """Phase 1 of training freezes these; phase 2 unfreezes them for fine-tuning."""
    for p in backbone_params:
        p.requires_grad = trainable


def set_batchnorm_eval(model: nn.Module) -> None:
    """Put every BatchNorm layer in eval mode so its running mean/var stop updating.

    `requires_grad=False` only freezes BN's affine weights; in `model.train()` the
    running statistics still move. Call this after `model.train()` to truly freeze them.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def get_last_conv_layer(model, model_name: str):
    """Returns the last conv layer — used as the Grad-CAM target layer."""
    model_name = model_name.lower()
    if model_name in ("resnet18", "resnet50"):
        return model.layer4[-1]
    if model_name in ("efficientnet_b0", "mobilenet_v2"):
        return model.features[-1]
    raise ValueError(f"No known last-conv-layer mapping for: {model_name}")


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Load a training checkpoint without executing arbitrary pickled code.

    `weights_only=True` restricts unpickling to tensors and plain containers, which is
    all our checkpoints contain. Never load untrusted checkpoints with weights_only=False.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path} is not a checkpoint dict (got {type(ckpt).__name__})")
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path} is missing checkpoint keys: {missing}")
    if ckpt["model_name"] not in SUPPORTED_MODELS:
        raise ValueError(f"{path} has unsupported model_name {ckpt['model_name']!r}")
    return ckpt


def build_model_from_checkpoint(ckpt: dict, device="cpu"):
    """Instantiate the architecture named in `ckpt` and load its weights (strictly).

    No ImageNet weights are downloaded: every parameter comes from the checkpoint.
    """
    model, _ = get_model(ckpt["model_name"], num_classes=len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval()
