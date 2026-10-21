"""Gradio app logic, tested without launching a server."""
import importlib
import sys

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from conftest import CHECKPOINT, random_image


def test_import_has_no_side_effects(monkeypatch):
    """Importing the module must not parse CLI args or load a model (it used to do both)."""
    monkeypatch.setattr(sys, "argv", ["pytest", "--unknown-flag"])
    import model

    def _fail(*a, **k):
        raise AssertionError("checkpoint loaded at import time")

    monkeypatch.setattr(model, "load_checkpoint", _fail)
    sys.modules.pop("demo", None)
    try:
        importlib.import_module("demo")
    finally:
        sys.modules.pop("demo", None)  # don't leak a module bound to the patched loader


class TestParseArgs:
    def test_local_defaults_to_localhost(self, monkeypatch):
        import demo
        monkeypatch.delenv("PORT", raising=False)
        monkeypatch.delenv("GRADIO_SERVER_NAME", raising=False)
        monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
        args = demo.parse_args([])
        assert args.host == "127.0.0.1" and args.port == 7860 and not args.share
        assert args.checkpoint == str(demo.DEFAULT_CHECKPOINT)  # absolute, independent of CWD

    def test_hosting_platform_port(self, monkeypatch):
        import demo
        monkeypatch.setenv("PORT", "10000")
        monkeypatch.delenv("GRADIO_SERVER_NAME", raising=False)
        args = demo.parse_args([])
        assert args.host == "0.0.0.0" and args.port == 10000

    def test_share_flag_is_parsed(self):
        import demo
        assert demo.parse_args(["--share"]).share is True


def test_main_exits_cleanly_on_missing_checkpoint(tmp_path):
    import demo
    with pytest.raises(SystemExit, match="Checkpoint not found"):
        demo.main(["--checkpoint", str(tmp_path / "nope.pt")])


def test_main_exits_cleanly_on_corrupt_checkpoint(tmp_path):
    import demo
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(SystemExit, match="Could not load checkpoint"):
        demo.main(["--checkpoint", str(bad)])


def test_summary_flags_inconclusive_predictions():
    import demo
    confident = demo.summarize([("Tomato___healthy", 0.97), ("Tomato___Late_blight", 0.02)])
    split = demo.summarize([("Tomato___healthy", 0.45), ("Tomato___Late_blight", 0.40)])
    assert "Inconclusive" not in confident and "Inconclusive" in split
    assert "Tomato — Healthy" in confident
    for text in (confident, split):
        assert "not calibrated" in text


@pytest.fixture(scope="module")
def predictor():
    import demo
    from utils import get_device
    return demo.Predictor(CHECKPOINT, get_device("cpu"))


@pytest.mark.checkpoint
@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="results/best_model.pt not present")
class TestWithCommittedCheckpoint:
    @pytest.mark.parametrize("size", [(256, 256), (800, 600), (600, 800), (3000, 1000)])
    def test_predict_outputs(self, predictor, rng, size):
        result = predictor.predict(random_image(rng, *size))
        top = result["top"]
        assert len(top) == 5 and all(name in predictor.class_names for name, _ in top)
        probs = [p for _, p in top]
        assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1 + 1e-6
        assert result["cam"].shape == (7, 7)

        # Overlay covers only the analysed (center-crop) region and keeps its square shape
        left, top_, right, bottom = result["crop_box"]
        assert result["overlay"].width == pytest.approx(result["overlay"].height, abs=1)
        assert (right - left) == pytest.approx(bottom - top_, abs=2)
        assert max(result["context"].size) <= 512

    def test_exif_orientation_is_applied(self, predictor):
        img = Image.new("RGB", (400, 200), (30, 120, 30))
        exif = img.getexif()
        exif[0x0112] = 6  # rotate 90° when displayed
        path_img = Image.open(_save_with_exif(img, exif))
        result = predictor.predict(path_img)
        assert result["context"].size[1] > result["context"].size[0]  # now portrait

    def test_analyze_image_validation(self, predictor):
        import demo
        with pytest.raises(gr.Error):
            demo.analyze_image(predictor, None)
        with pytest.raises(gr.Error, match="too small"):
            demo.analyze_image(predictor, Image.new("RGB", (20, 400)))

    def test_analyze_image_hides_internal_errors(self, predictor, monkeypatch, rng):
        import demo
        monkeypatch.setattr(predictor, "predict", lambda image: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(gr.Error) as exc:
            demo.analyze_image(predictor, random_image(rng, 300, 300))
        assert "boom" not in str(exc.value)

    def test_analyze_image_returns_human_readable_labels(self, predictor, rng):
        import demo
        summary, label, context, overlay = demo.analyze_image(predictor, random_image(rng, 300, 300))
        assert all("___" not in k and "_" not in k for k in label)
        assert isinstance(context, Image.Image) and isinstance(overlay, Image.Image)
        assert np.isfinite(list(label.values())).all() and 0 < sum(label.values()) <= 1 + 1e-6
        assert "not calibrated" in summary

    def test_build_demo(self, predictor):
        import demo
        assert isinstance(demo.build_demo(predictor), gr.Blocks)


def _save_with_exif(img, exif):
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return buf
