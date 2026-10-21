import pytest
import torch
import torch.nn as nn

from model import (
    SUPPORTED_MODELS,
    build_model_from_checkpoint,
    get_last_conv_layer,
    get_model,
    load_checkpoint,
    set_backbone_trainable,
    set_batchnorm_eval,
)

HEAD_PREFIX = {"resnet18": "fc", "resnet50": "fc", "efficientnet_b0": "classifier", "mobilenet_v2": "classifier"}


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_get_model_head_and_backbone_split(name):
    model, backbone = get_model(name, num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5)

    head = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX[name])]
    backbone_ids = {id(p) for p in backbone}
    assert head and not any(id(p) in backbone_ids for p in head)
    assert len(backbone) + len(head) == len(list(model.parameters()))


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_last_conv_layer_is_a_submodule(name):
    model, _ = get_model(name, num_classes=3, pretrained=False)
    layer = get_last_conv_layer(model, name)
    assert any(layer is m for m in model.modules())


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_model("vgg16", pretrained=False)


def test_set_backbone_trainable_toggles_only_backbone():
    model, backbone = get_model("resnet18", num_classes=3, pretrained=False)
    set_backbone_trainable(backbone, trainable=False)
    assert not any(p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in model.fc.parameters())
    set_backbone_trainable(backbone, trainable=True)
    assert all(p.requires_grad for p in model.parameters())


def test_set_batchnorm_eval_freezes_running_stats():
    model, _ = get_model("resnet18", num_classes=3, pretrained=False)
    model.train()
    set_batchnorm_eval(model)
    before = model.bn1.running_mean.clone()
    model(torch.randn(4, 3, 64, 64))
    assert torch.equal(before, model.bn1.running_mean)
    assert model.fc.training  # non-BN modules stay in train mode


def _save_tiny_checkpoint(path, **overrides):
    model, _ = get_model("resnet18", num_classes=3, pretrained=False)
    ckpt = {"model_state": model.state_dict(), "model_name": "resnet18", "class_names": ["a", "b", "c"]}
    ckpt.update(overrides)
    torch.save(ckpt, path)
    return model


def test_checkpoint_roundtrip_without_downloads(tmp_path):
    original = _save_tiny_checkpoint(tmp_path / "ck.pt")
    restored = build_model_from_checkpoint(load_checkpoint(tmp_path / "ck.pt"))
    assert not restored.training
    x = torch.randn(1, 3, 64, 64)
    original.eval()
    with torch.no_grad():
        assert torch.allclose(original(x), restored(x))


def test_load_checkpoint_rejects_missing_keys(tmp_path):
    torch.save({"model_state": {}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="missing checkpoint keys"):
        load_checkpoint(tmp_path / "bad.pt")


def test_load_checkpoint_rejects_unknown_architecture(tmp_path):
    _save_tiny_checkpoint(tmp_path / "ck.pt", model_name="vgg16")
    with pytest.raises(ValueError, match="unsupported model_name"):
        load_checkpoint(tmp_path / "ck.pt")


class _NotATensor:
    pass


def test_load_checkpoint_refuses_arbitrary_pickled_objects(tmp_path):
    _save_tiny_checkpoint(tmp_path / "ck.pt", extra=_NotATensor())
    with pytest.raises(Exception, match="(?i)weights.only|unsupported"):
        load_checkpoint(tmp_path / "ck.pt")


def test_strict_loading_catches_class_count_mismatch(tmp_path):
    _save_tiny_checkpoint(tmp_path / "ck.pt", class_names=["a", "b"])  # head has 3 outputs
    with pytest.raises(RuntimeError, match="size mismatch"):
        build_model_from_checkpoint(load_checkpoint(tmp_path / "ck.pt"))


def test_batchnorm_layers_exist_in_every_supported_backbone():
    # set_batchnorm_eval is only meaningful if the backbones actually contain BN layers
    for name in SUPPORTED_MODELS:
        model, _ = get_model(name, num_classes=3, pretrained=False)
        assert any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
