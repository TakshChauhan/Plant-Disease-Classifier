"""Tests for checkpoint forensics, safe loading, preprocessing geometry, and Grad-CAM."""
from __future__ import annotations

import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ml_validation.core.backend import (
    COMMITTED_CHECKPOINT_SHA256,
    CheckpointModelAdapter,
    file_sha256,
    load_checkpoint_safe,
    overlay_heatmap_pil,
    parse_crop_and_disease,
    pretty_class_name,
)
from ml_validation.core.preprocessing import (
    IMAGENET_MEAN,
    default_preprocessing,
    eval_crop_box,
    preprocess_pil,
    preprocessing_from_checkpoint,
)

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "results" / "best_model.pt"


class _UnsafeCustomClass:
    pass


class TestBackendAndCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_ckpt = CHECKPOINT_PATH.is_file()
        if cls.has_ckpt:
            cls.adapter = CheckpointModelAdapter(CHECKPOINT_PATH)

    def test_committed_checkpoint_sha256_and_parameters(self):
        if not self.has_ckpt:
            self.skipTest("best_model.pt not present")
        self.assertEqual(file_sha256(CHECKPOINT_PATH), COMMITTED_CHECKPOINT_SHA256)
        summary = self.adapter.parameter_summary()
        self.assertEqual(summary["model_name"], "resnet18")
        self.assertEqual(summary["num_classes"], 38)
        self.assertEqual(summary["state_dict_keys"], 122)
        self.assertEqual(summary["total_learnable_params"], 11196006)
        self.assertEqual(summary["backbone_learnable_params"], 11176512)
        self.assertEqual(summary["head_learnable_params"], 19494)
        self.assertEqual(summary["total_buffer_elements"], 9620)
        self.assertEqual(summary["bn_num_batches_tracked_unique"], [21970])

    def test_preprocessing_and_crop_box_geometry(self):
        pp = default_preprocessing()
        self.assertEqual(pp["img_size"], 224)
        self.assertEqual(pp["resize_size"], 255)
        self.assertEqual(preprocessing_from_checkpoint({"model_state": {}}), pp)

        left, top, right, bottom = eval_crop_box(640, 480, 255, 224)
        self.assertAlmostEqual(right - left, bottom - top, delta=2.0)
        self.assertAlmostEqual((left + right) / 2.0, 320.0, delta=2.0)
        self.assertAlmostEqual((top + bottom) / 2.0, 240.0, delta=2.0)

        # Uniform image at ImageNet mean normalizes to ~0
        mean_rgb = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)
        img = Image.new("RGB", (300, 240), mean_rgb)
        arr = preprocess_pil(img, pp)
        self.assertEqual(arr.shape, (3, 224, 224))
        self.assertLess(float(np.max(np.abs(arr))), 0.02)

    def test_safe_loader_rejects_corrupt_and_unsafe_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "corrupt.pt"
            bad_path.write_bytes(b"not a zip archive")
            with self.assertRaises(ValueError):
                load_checkpoint_safe(bad_path)

            # Zip archive containing an arbitrary pickled class must be refused
            unsafe_zip = Path(tmp) / "unsafe.pt"
            with zipfile.ZipFile(unsafe_zip, "w") as z:
                z.writestr("archive/data.pkl", pickle.dumps({"evil": _UnsafeCustomClass()}))
            with self.assertRaises(ValueError):
                load_checkpoint_safe(unsafe_zip)

            # Missing required keys must be refused
            missing_zip = Path(tmp) / "missing.pt"
            with zipfile.ZipFile(missing_zip, "w") as z:
                z.writestr("archive/data.pkl", pickle.dumps({"model_state": {}}))
            with self.assertRaises(ValueError):
                load_checkpoint_safe(missing_zip)

    def test_forward_determinism_and_gradcam_properties(self):
        if not self.has_ckpt:
            self.skipTest("best_model.pt not present")
        rng = np.random.default_rng(123)
        img = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
        x = preprocess_pil(img, self.adapter.preprocessing)[None, ...]

        logits1 = self.adapter.forward_logits(x)
        logits2 = self.adapter.forward_logits(x)
        np.testing.assert_allclose(logits1, logits2, atol=1e-6)

        logits_cam, cam, target_cls = self.adapter.forward_logits_and_cam(x)
        np.testing.assert_allclose(logits1, logits_cam, atol=1e-5)
        self.assertEqual(cam.shape, (7, 7))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0 + 1e-6)
        self.assertEqual(target_cls, int(logits1[0].argmax()))

        # Reject batch > 1 in Grad-CAM
        with self.assertRaises(ValueError):
            self.adapter.forward_logits_and_cam(np.concatenate([x, x], axis=0))

        overlay = overlay_heatmap_pil(img, cam)
        self.assertEqual(overlay.size, (256, 256))

    def test_pretty_class_name_and_parser(self):
        self.assertEqual(
            pretty_class_name("Corn_(maize)___Common_rust_"),
            "Corn (maize) — Common rust",
        )
        crop, cond, is_h = parse_crop_and_disease("Tomato___healthy")
        self.assertEqual(crop, "Tomato")
        self.assertEqual(cond, "healthy")
        self.assertTrue(is_h)
