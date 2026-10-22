"""Tests for Phase 3 PlantDoc taxonomic mapping and exclusion accounting."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml_validation.evaluate_field import load_mapping_spec

MAPPING_PATH = Path(__file__).resolve().parent.parent / "class_mapping.json"
CLASS_NAMES_PATH = Path(__file__).resolve().parents[2] / "results" / "class_names.json"


class TestEvaluateField(unittest.TestCase):
    def test_mapping_table_integrity_and_strict_vs_permissive_modes(self):
        raw = json.loads(MAPPING_PATH.read_text())
        self.assertEqual(raw["metadata"]["target_checkpoint"]["num_classes"], 38)
        p2v = raw["plantdoc_to_plantvillage"]
        self.assertGreaterEqual(len(p2v), 28)

        # Verify Bell_pepper leaf spot is flagged as AMBIGUOUS_BROAD_DISEASE_DEFINITION
        self.assertEqual(
            p2v["Bell_pepper leaf spot"]["status"],
            "AMBIGUOUS_BROAD_DISEASE_DEFINITION",
        )
        # Verify 9 uncovered PlantVillage classes are explicitly documented
        self.assertEqual(len(raw["unmapped_plantvillage_classes"]), 9)

        if CLASS_NAMES_PATH.is_file():
            ckpt_classes = json.loads(CLASS_NAMES_PATH.read_text())
            strict_spec = load_mapping_spec(MAPPING_PATH, ckpt_classes, strict_taxonomy=True)
            perm_spec = load_mapping_spec(MAPPING_PATH, ckpt_classes, strict_taxonomy=False)
            self.assertNotIn("Bell_pepper leaf spot", strict_spec["accepted_map"])
            self.assertIn("Bell_pepper leaf spot", strict_spec["excluded_policy_map"])
            self.assertIn("Bell_pepper leaf spot", perm_spec["accepted_map"])
