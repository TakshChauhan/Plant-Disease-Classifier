"""Tests for Phase 7 theoretical FLOPs and environment telemetry."""
from __future__ import annotations

import unittest
from ml_validation.benchmark import compute_resnet18_theoretical_flops, get_environment_metadata


class TestBenchmark(unittest.TestCase):
    def test_resnet18_theoretical_flops(self):
        flops = compute_resnet18_theoretical_flops(num_classes=38)
        self.assertEqual(flops["input_resolution"], "3x224x224")
        self.assertEqual(flops["total_macs"], 1813580800)
        self.assertAlmostEqual(flops["total_gmacs"], 1.8136, places=4)

    def test_environment_metadata_fields(self):
        env = get_environment_metadata()
        self.assertIn("python_version", env)
        self.assertIn("cpu_count_logical", env)
        self.assertIn("cuda_gpu_available", env)
