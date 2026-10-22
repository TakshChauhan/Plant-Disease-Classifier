"""Tests for Phase 1 classification metrics and baseline evaluation."""
from __future__ import annotations

import unittest
import numpy as np

from ml_validation.core.metrics import (
    compute_classification_metrics,
    format_classification_report,
    top_k_accuracy,
)


class TestEvaluateBaseline(unittest.TestCase):
    def test_compute_classification_metrics_exact_values(self):
        labels = [0, 0, 1, 1, 2, 2]
        preds = [0, 0, 1, 0, 2, 2]
        probs = np.array([
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],  # misclassified top-1, but class 1 is in top-2
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ])
        m = compute_classification_metrics(labels, preds, ["c0", "c1", "c2"], probs=probs)
        self.assertEqual(m["n_samples"], 6)
        self.assertAlmostEqual(m["top1_accuracy"], 5.0 / 6.0)
        self.assertAlmostEqual(m["top3_accuracy"], 1.0)
        self.assertAlmostEqual(m["macro_precision"], (2.0 / 3.0 + 1.0 + 1.0) / 3.0)
        self.assertAlmostEqual(m["macro_recall"], (1.0 + 0.5 + 1.0) / 3.0)
        report_txt = format_classification_report(m, digits=4)
        self.assertIn("macro avg", report_txt)
        self.assertIn("weighted avg", report_txt)

    def test_handles_absent_class_without_division_by_zero(self):
        m = compute_classification_metrics([0, 0], [0, 0], ["present_cls", "absent_cls"])
        self.assertAlmostEqual(m["top1_accuracy"], 1.0)
        self.assertAlmostEqual(m["macro_recall"], 0.5)
        self.assertAlmostEqual(m["active_classes_macro_recall"], 1.0)

    def test_top_k_accuracy(self):
        scores = np.array([
            [0.1, 0.5, 0.4],
            [0.7, 0.2, 0.1],
        ])
        labels = [2, 0]
        self.assertAlmostEqual(top_k_accuracy(labels, scores, k=1), 0.5)
        self.assertAlmostEqual(top_k_accuracy(labels, scores, k=2), 1.0)
