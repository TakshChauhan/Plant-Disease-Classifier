"""Minimal, dependency-free Grad-CAM implementation (Selvaraju et al., 2017).

Registers a forward hook to capture activations on a target conv layer, takes the
gradient of the class score with respect to those activations, and combines them into
a class-discriminative heatmap: weight each channel by its mean gradient, sum, ReLU.

The heatmap lives in the coordinates of the *model input* (after resize + center crop),
so it must be overlaid on that same region — see `dataset.eval_crop_box`.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.logits = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output

    def remove(self) -> None:
        """Detach the hook from the model (hooks otherwise accumulate per GradCAM instance)."""
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, input_tensor: torch.Tensor, target_class: int = None):
        """input_tensor: (1, C, H, W), already normalized.

        Returns (cam, target_class): a (h, w) heatmap in [0, 1] at the target layer's
        resolution (7x7 for 224px ResNet18), and the explained class index. The raw
        logits from the same forward pass are kept in `self.logits`.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"Expected a single image batch (1, C, H, W), got {tuple(input_tensor.shape)}")
        self.model.eval()
        # requires_grad on the input guarantees a graph to the hooked layer even if the
        # model's parameters are frozen; enable_grad guards against a caller's no_grad().
        input_tensor = input_tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            # Gradient w.r.t. the activations only: no parameter .grad buffers are touched
            (self.gradients,) = torch.autograd.grad(output[0, target_class], self.activations)

        self.logits = output.detach()
        activations = self.activations.detach()
        self.activations = None  # don't keep the autograd graph alive between calls

        # Global-average-pool the gradients per channel -> per-channel importance weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, target_class


def overlay_heatmap(pil_image, cam: np.ndarray, alpha: float = 0.45):
    """Resizes `cam` to the image size and overlays it as a color heatmap. Returns a PIL Image.

    `pil_image` must show the same region the model saw (the eval center crop), otherwise
    the heatmap is spatially misaligned.
    """
    img = np.array(pil_image.convert("RGB"))
    cam_resized = cv2.resize(cam.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(cam_resized, 0, 1)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(overlay)
