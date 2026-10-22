"""Tests for Phase 4 foreground leaf segmentation and perturbation suite."""
from __future__ import annotations

import unittest
import numpy as np

from ml_validation.robustness import (
    build_perturbation_suite,
    jensen_shannon_divergence,
    segment_leaf_foreground_mask,
)


class TestRobustness(unittest.TestCase):
    def test_foreground_segmentation_preserves_interior_lesions(self):
        # Neutral studio grey background (128,128,128) with green leaf and dark interior lesion spot
        canvas = np.full((224, 224, 3), 128, dtype=np.uint8)
        yy, xx = np.mgrid[:224, :224]
        leaf = ((xx - 112) / 65.0) ** 2 + ((yy - 112) / 80.0) ** 2 <= 1.0
        lesion = ((xx - 112) ** 2 + (yy - 112) ** 2) <= 16 ** 2
        canvas[leaf] = np.array([50, 140, 45], dtype=np.uint8)
        canvas[lesion] = np.array([125, 120, 115], dtype=np.uint8)  # necrotic spot inside leaf

        mask = segment_leaf_foreground_mask(canvas)
        self.assertEqual(mask.shape, (224, 224))
        # Center lesion pixel MUST be kept inside the foreground leaf mask
        self.assertTrue(bool(mask[112, 112]))
        # Outer border corner MUST be classified as background
        self.assertFalse(bool(mask[5, 5]))

    def test_perturbation_operators_preserve_shape_and_dtype(self):
        rng = np.random.default_rng(11)
        rgb = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for key, _name, fn in build_perturbation_suite():
            out = fn(rgb)
            self.assertEqual(out.shape, (224, 224, 3), msg=f"Failed shape for {key}")
            self.assertEqual(out.dtype, np.uint8, msg=f"Failed dtype for {key}")

    def test_jensen_shannon_divergence_properties(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.2, 0.7])
        self.assertAlmostEqual(jensen_shannon_divergence(p, p), 0.0, places=7)
        self.assertAlmostEqual(jensen_shannon_divergence(p, q), jensen_shannon_divergence(q, p), places=7)
        self.assertGreater(jensen_shannon_divergence(p, q), 0.0)
        self.assertLessEqual(jensen_shannon_divergence(p, q), 1.0)
