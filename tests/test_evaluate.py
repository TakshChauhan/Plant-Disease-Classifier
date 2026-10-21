import json
import sys

import pytest
import torch

import evaluate
from model import get_model


def test_compute_metrics_known_values():
    labels = [0, 0, 1, 1, 2, 2]
    preds = [0, 0, 1, 0, 2, 2]
    m = evaluate.compute_metrics(labels, preds, ["a", "b", "c"])
    assert m["n_samples"] == 6
    assert m["accuracy"] == pytest.approx(5 / 6)
    # class a: P=2/3 R=1; b: P=1 R=1/2; c: P=1 R=1
    assert m["macro_precision"] == pytest.approx((2 / 3 + 1 + 1) / 3)
    assert m["macro_recall"] == pytest.approx((1 + 0.5 + 1) / 3)


def test_compute_metrics_handles_absent_class():
    m = evaluate.compute_metrics([0, 0], [0, 0], ["a", "b"])
    assert m["accuracy"] == 1.0 and m["macro_recall"] == pytest.approx(0.5)


def test_main_end_to_end(image_folder, tmp_path, monkeypatch):
    data_dir, classes = image_folder
    model, _ = get_model("resnet18", num_classes=len(classes), pretrained=False)
    ckpt_path = tmp_path / "ck.pt"
    torch.save({"model_state": model.state_dict(), "model_name": "resnet18", "class_names": classes,
                "preprocessing": {"img_size": 32, "resize_size": 36,
                                  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}, ckpt_path)
    out = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--data-dir", str(data_dir), "--checkpoint", str(ckpt_path),
                                      "--output-dir", str(out), "--num-workers", "0", "--device", "cpu"])
    evaluate.main()

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_samples"] == 6 and metrics["split"] == "valid"
    assert 0.0 <= metrics["accuracy"] <= 1.0 and len(metrics["checkpoint_sha256"]) == 64
    assert (out / "confusion_matrix.png").is_file()
    assert "Apple___healthy" in (out / "classification_report.txt").read_text()


def test_main_rejects_class_order_mismatch(image_folder, tmp_path, monkeypatch):
    data_dir, classes = image_folder
    model, _ = get_model("resnet18", num_classes=len(classes), pretrained=False)
    torch.save({"model_state": model.state_dict(), "model_name": "resnet18",
                "class_names": list(reversed(classes))}, tmp_path / "ck.pt")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--data-dir", str(data_dir), "--checkpoint",
                                      str(tmp_path / "ck.pt"), "--output-dir", str(tmp_path), "--num-workers", "0"])
    with pytest.raises(SystemExit, match="class order"):
        evaluate.main()
