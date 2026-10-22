"""Tests for Phase 6 failure-case analysis helpers."""
from __future__ import annotations

import unittest
from PIL import Image

from ml_validation.failure_analysis import categorize_disease_etiology, draw_crop_box_pil


class TestFailureAnalysis(unittest.TestCase):
    def test_categorize_disease_etiology(self):
        self.assertEqual(categorize_disease_etiology("Tomato___healthy"), "Healthy")
        self.assertEqual(categorize_disease_etiology("Tomato___Tomato_Yellow_Leaf_Curl_Virus"), "Viral")
        self.assertEqual(categorize_disease_etiology("Peach___Bacterial_spot"), "Bacterial")
        self.assertEqual(
            categorize_disease_etiology("Tomato___Spider_mites Two-spotted_spider_mite"),
            "Arachnid / Pest",
        )
        self.assertEqual(categorize_disease_etiology("Apple___Apple_scab"), "Fungal / Oomycete")

    def test_draw_crop_box_pil(self):
        img = Image.new("RGB", (300, 200), (20, 100, 20))
        out = draw_crop_box_pil(img, (50, 10, 250, 190))
        self.assertEqual(out.size, (300, 200))
