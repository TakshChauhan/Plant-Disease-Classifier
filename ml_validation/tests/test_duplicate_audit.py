"""Tests for Phase 2 D4 orientation-invariant perceptual hashing and BK-tree duplicate detection."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_validation.duplicate_audit import (
    D4_ORIENTATIONS,
    MultiIndexHashSearch,
    apply_d4_transform,
    classify_candidate_pair,
    compute_ahash_64,
    compute_dhash_64,
    compute_phash_64,
    generate_synthetic_leakage_benchmark,
    run_duplicate_audit,
)


class TestDuplicateAudit(unittest.TestCase):
    def test_d4_symmetries_are_indexed_and_retrieved_with_zero_distance(self):
        rng = np.random.default_rng(7)
        base_gray = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        index = MultiIndexHashSearch()

        for orient in D4_ORIENTATIONS:
            g_t = np.ascontiguousarray(apply_d4_transform(base_gray, orient))
            index.add(
                0,
                orient,
                compute_dhash_64(g_t),
                compute_phash_64(g_t),
                compute_ahash_64(g_t),
            )

        # Query with a 90-degree rotated + flipped variant
        for test_orient in ("rot90", "flip_h", "rot180", "transpose"):
            q_img = np.ascontiguousarray(apply_d4_transform(base_gray, test_orient))
            q_d = compute_dhash_64(q_img)
            q_p = compute_phash_64(q_img)
            hits = index.query(q_d, q_p, max_dhash_hamming=2)
            self.assertTrue(any(d_ham == 0 and p_ham == 0 for _, _, d_ham, p_ham, _ in hits))

    def test_candidate_pair_tier_classification(self):
        t1, _ = classify_candidate_pair(True, True, "orig", 0, 0, 0, 1.0)
        self.assertEqual(t1, "CONFIRMED_EXACT_DUPLICATE")

        t2, _ = classify_candidate_pair(False, False, "rot90", 3, 4, 3, 0.98)
        self.assertEqual(t2, "HIGH_CONFIDENCE_AUGMENTATION_LEAK")

        t3, _ = classify_candidate_pair(False, False, "flip_h", 9, 9, 8, 0.93)
        self.assertEqual(t3, "SUSPICIOUS_NEAR_DUPLICATE")

        t4, _ = classify_candidate_pair(False, False, "orig", 15, 14, 13, 0.88)
        self.assertEqual(t4, "AMBIGUOUS_REQUIRES_MANUAL_INSPECTION")

    def test_end_to_end_synthetic_leakage_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t_dir, v_dir, expected = generate_synthetic_leakage_benchmark(root / "split", seed=42)
            out_dir = root / "out"
            summary = run_duplicate_audit(t_dir, v_dir, out_dir)
            bd = summary["validation_split_breakdown"]
            self.assertEqual(bd["confirmed_exact_duplicates"], expected["injected_exact_or_d4"])
            self.assertEqual(bd["no_suspicious_match_found"], expected["injected_independent_clean"])
            self.assertTrue((out_dir / "duplicate_candidates.csv").is_file())
            self.assertTrue((out_dir / "duplicate_gallery.png").is_file())
