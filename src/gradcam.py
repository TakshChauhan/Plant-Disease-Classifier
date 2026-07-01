"""Minimal, dependency-free Grad-CAM implementation (Selvaraju et al., 2017).

Registers a forward hook to capture activations and a backward hook to capture
gradients on a target conv layer, then combines them into a class-discriminative
heatmap: weight each channel by the mean gradient flowing into it, sum, ReLU.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_class: int = None):
        """input_tensor: (1, C, H, W), already normalized. Returns a (H, W) heatmap in [0, 1]."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Global-average-pool the gradients per channel -> per-channel importance weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, target_class


def overlay_heatmap(pil_image, cam: np.ndarray, alpha: float = 0.45):
    """Resizes `cam` to the image size and overlays it as a color heatmap. Returns a PIL Image."""
    img = np.array(pil_image.convert("RGB"))
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)

    from PIL import Image
    return Image.fromarray(overlay)
