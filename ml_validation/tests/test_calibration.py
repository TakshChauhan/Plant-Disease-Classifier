"""Tests for Phase 5 calibration statistics and held-out temperature scaling."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_validation.calibration import (
    evaluate_calibration_and_temperature_scaling,
    generate_synthetic_overconfident_logits,
)
from ml_validation.core.metrics import (
    compute_calibration_stats,
    fit_temperature_scaling,
    softmax_np,
)


class TestCalibration(unittest.TestCase):
    def test_temperature_scaling_preserves_accuracy_and_reduces_nll_and_ece(self):
        logits, labels = generate_synthetic_overconfident_logits(n_samples=400, num_classes=38, seed=42)
        fit_res = fit_temperature_scaling(logits[:200], labels[:200])
        opt_T = fit_res["optimal_temperature"]
        self.assertGreater(opt_T, 1.1)
        self.assertLess(fit_res["fit_nll_after"], fit_res["fit_nll_before"])

        # Evaluate on held-out 200 samples
        probs_before = softmax_np(logits[200:], temperature=1.0)
        probs_after = softmax_np(logits[200:], temperature=opt_T)
        st_before = compute_calibration_stats(labels[200:], probs_before)
        st_after = compute_calibration_stats(labels[200:], probs_after)

        # Temperature scaling is a monotonic positive rescaling and MUST NOT change Top-1 accuracy
        self.assertAlmostEqual(st_before["accuracy"], st_after["accuracy"], places=7)
        self.assertLess(st_after["nll"], st_before["nll"])
        self.assertLess(st_after["ece"], st_before["ece"])
        self.assertLess(st_after["brier_score"], st_before["brier_score"])

    def test_end_to_end_calibration_pipeline(self):
        logits, labels = generate_synthetic_overconfident_logits(n_samples=100, num_classes=10, seed=9)
        ood_logits = np.random.default_rng(1).normal(0, 3.0, size=(30, 10))
        with tempfile.TemporaryDirectory() as tmp:
            res = evaluate_calibration_and_temperature_scaling(logits, labels, ood_logits, Path(tmp))
            self.assertEqual(res["n_calibration_fit_split"], 50)
            self.assertEqual(res["n_calibration_eval_split"], 50)
            self.assertTrue((Path(tmp) / "reliability_diagram.png").is_file())
            self.assertTrue((Path(tmp) / "rejection_curve_analysis.png").is_file())
