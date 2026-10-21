import numpy as np
import pytest
import torch
from PIL import Image

from gradcam import GradCAM, overlay_heatmap
from model import SUPPORTED_MODELS, get_last_conv_layer, get_model


def _model(name, num_classes=4):
    torch.manual_seed(0)
    model, _ = get_model(name, num_classes=num_classes, pretrained=False)
    return model.eval()


def _reference_cam(model, layer, x, target):
    """Textbook Grad-CAM with forward + full backward hooks, as in the original implementation."""
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o.detach()))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0].detach()))
    try:
        out = model(x)
        model.zero_grad()
        out[0, target].backward()
    finally:
        h1.remove()
        h2.remove()
    cam = torch.relu((store["grad"].mean(dim=(2, 3), keepdim=True) * store["act"]).sum(1))[0].numpy()
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_cam_shape_range_and_target(name):
    model = _model(name)
    with GradCAM(model, get_last_conv_layer(model, name)) as gc:
        cam, target = gc(torch.randn(1, 3, 224, 224))
        assert cam.shape == (7, 7)
        assert cam.min() >= 0 and cam.max() <= 1 + 1e-6
        assert target == int(gc.logits.argmax())


@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v2"])
def test_cam_matches_reference_implementation(name):
    model = _model(name)
    layer = get_last_conv_layer(model, name)
    x = torch.randn(1, 3, 224, 224)
    with GradCAM(model, layer) as gc:
        cam, _ = gc(x, target_class=2)
    np.testing.assert_allclose(cam, _reference_cam(model, layer, x, 2), atol=1e-5)


def test_logits_match_plain_forward_and_no_param_grads():
    model = _model("resnet18")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = model(x)
    with GradCAM(model, model.layer4[-1]) as gc:
        gc(x)
        assert torch.allclose(gc.logits, expected, atol=1e-5)
    assert all(p.grad is None for p in model.parameters())


def test_works_inside_no_grad_and_with_frozen_parameters():
    model = _model("resnet18")
    for p in model.parameters():
        p.requires_grad = False
    with GradCAM(model, model.layer4[-1]) as gc, torch.no_grad():
        cam, _ = gc(torch.randn(1, 3, 224, 224))
    assert cam.shape == (7, 7)


def test_explicit_target_class_is_respected():
    model = _model("resnet18")
    with GradCAM(model, model.layer4[-1]) as gc:
        _, target = gc(torch.randn(1, 3, 224, 224), target_class=3)
    assert target == 3


def test_rejects_batches():
    model = _model("resnet18")
    with GradCAM(model, model.layer4[-1]) as gc, pytest.raises(ValueError):
        gc(torch.randn(2, 3, 224, 224))


def test_hooks_are_removed():
    model = _model("resnet18")
    layer = model.layer4[-1]
    before = len(layer._forward_hooks)
    gc = GradCAM(model, layer)
    assert len(layer._forward_hooks) == before + 1
    gc.remove()
    assert len(layer._forward_hooks) == before


def test_overlay_heatmap_keeps_image_size():
    img = Image.new("RGB", (123, 77), (0, 128, 0))
    cam = np.zeros((7, 7), dtype=np.float32)
    cam[3, 3] = 1.0
    out = overlay_heatmap(img, cam)
    assert out.size == (123, 77) and out.mode == "RGB"
    arr = np.asarray(out).astype(int)
    # Hottest region (red in JET) sits at the centre, not the corner
    assert arr[38, 61, 0] > arr[2, 2, 0]
