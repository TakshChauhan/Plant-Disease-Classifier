"""Training-loop behaviour on a tiny model and synthetic tensors (no dataset download)."""
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import train
from model import get_model, load_checkpoint


def _loaders(n=8, classes=3, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.randn(n, 3, size, size, generator=g), torch.randint(0, classes, (n,), generator=g))
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def _tiny_model():
    torch.manual_seed(0)
    return get_model("resnet18", num_classes=3, pretrained=False)


def _bn_free_model():
    """With no BatchNorm, lr=0 leaves the model bit-identical, so val accuracy is exactly constant."""
    torch.manual_seed(0)
    backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 16), torch.nn.ReLU())
    model = torch.nn.Sequential(backbone, torch.nn.Linear(16, 3))
    return model, list(backbone.parameters())


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _run(model, backbone, **kw):
    train_loader, val_loader = _loaders()
    defaults = dict(freeze_epochs=1, finetune_epochs=1, head_lr=1e-2, finetune_lr=1e-2, patience=5,
                    log=lambda *_: None)
    defaults.update(kw)
    return train.train_two_phase(model, backbone, train_loader, val_loader, torch.device("cpu"), **defaults)


def test_phase1_trains_only_head_and_phase2_trains_backbone():
    model, backbone = _tiny_model()
    start = _snapshot(model)
    after_phase1 = {}

    def capture(info):
        if info["phase"] == "phase1-head":
            after_phase1.update(_snapshot(model))

    # val metric is constant-ish on random data; patience is high so both phases run fully
    history, _ = _run(model, backbone, on_best=capture)
    assert history["phase"] == ["phase1-head", "phase2-finetune"]
    assert after_phase1, "phase 1 epoch should be the first 'best' epoch"
    assert torch.equal(after_phase1["conv1.weight"], start["conv1.weight"])  # backbone frozen
    assert not torch.equal(after_phase1["fc.weight"], start["fc.weight"])  # head trained
    assert not torch.equal(model.conv1.weight, start["conv1.weight"])  # backbone fine-tuned in phase 2
    assert all(p.requires_grad for p in model.parameters())


def test_early_stop_in_phase1_does_not_skip_finetuning():
    model, backbone = _bn_free_model()
    # head_lr=0 -> validation accuracy never improves -> early stop fires inside phase 1
    history, es = _run(model, backbone, freeze_epochs=5, finetune_epochs=2, head_lr=0.0, patience=1)
    phases = history["phase"]
    assert phases.count("phase1-head") == 2  # epoch 1 sets best, epoch 2 exhausts patience
    assert phases.count("phase2-finetune") >= 1


def test_phase2_gets_fresh_patience_but_keeps_best():
    model, backbone = _bn_free_model()
    history, es = _run(model, backbone, freeze_epochs=3, finetune_epochs=3, head_lr=0.0, finetune_lr=0.0,
                       patience=1)
    # Neither phase can improve: each stops after exhausting its own budget of 1 epoch
    assert history["phase"] == ["phase1-head", "phase1-head", "phase2-finetune"]
    assert es.best == history["val_acc"][0]


def test_bn_running_stats_move_in_phase1_by_default():
    """Documents the original recipe: requires_grad=False does not freeze BN running stats."""
    model, backbone = _tiny_model()
    start_stats, start_weight = model.bn1.running_mean.clone(), model.bn1.weight.detach().clone()
    history, _ = _run(model, backbone, finetune_epochs=0)
    assert history["phase"] == ["phase1-head"]
    assert not torch.equal(model.bn1.running_mean, start_stats)  # running stats drifted...
    assert torch.equal(model.bn1.weight, start_weight)  # ...although the affine params were frozen


def test_freeze_bn_stats_keeps_backbone_identical_in_phase1():
    model, backbone = _tiny_model()
    start = _snapshot(model)
    _run(model, backbone, finetune_epochs=0, freeze_bn_stats=True)
    for k, v in model.state_dict().items():
        if not k.startswith("fc."):
            assert torch.equal(v, start[k]), k


def test_on_best_called_only_on_improvement():
    model, backbone = _bn_free_model()
    calls = []
    history, _ = _run(model, backbone, freeze_epochs=3, finetune_epochs=0, head_lr=0.0, patience=10,
                      on_best=calls.append)
    assert len(history["val_acc"]) == 3 and len(calls) == 1


def test_main_refuses_to_overwrite_existing_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "best_model.pt").write_bytes(b"precious")
    monkeypatch.setattr(sys, "argv", ["train.py", "--output-dir", str(tmp_path), "--data-dir", "nowhere"])
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        train.main()
    assert (tmp_path / "best_model.pt").read_bytes() == b"precious"


def test_main_end_to_end_writes_loadable_v2_checkpoint(image_folder, tmp_path, monkeypatch):
    data_dir, classes = image_folder
    real_get_model = train.get_model
    monkeypatch.setattr(train, "get_model", lambda name, num_classes: real_get_model(name, num_classes, False))
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--data-dir", str(data_dir), "--output-dir", str(out), "--img-size", "32",
        "--batch-size", "4", "--freeze-epochs", "1", "--finetune-epochs", "1", "--num-workers", "0",
        "--device", "cpu",
    ])
    train.main()

    ckpt = load_checkpoint(out / "best_model.pt")  # weights_only=True must accept all metadata
    assert ckpt["class_names"] == classes and ckpt["format_version"] == 2
    assert ckpt["preprocessing"]["img_size"] == 32
    assert isinstance(ckpt["versions"]["torch"], str)
    for name in ("class_names.json", "training_curves.png", "training_history.json"):
        assert (out / name).is_file()
