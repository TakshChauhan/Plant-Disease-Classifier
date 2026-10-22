"""Unified test runner for ml_validation.

Executes all tests in `ml_validation/tests/` using Python's standard `unittest`
runner (or `pytest` when invoked via pytest CLI) and writes a machine-readable
`ml_validation/reports/test_results.json` summary.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))

if "ml_validation" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "ml_validation", _PKG_DIR / "__init__.py", submodule_search_locations=[str(_PKG_DIR)]
    )
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["ml_validation"] = _mod
        _spec.loader.exec_module(_mod)

TEST_MODULES = [
    "ml_validation.tests.test_backend_and_checkpoint",
    "ml_validation.tests.test_evaluate_baseline",
    "ml_validation.tests.test_duplicate_audit",
    "ml_validation.tests.test_evaluate_field",
    "ml_validation.tests.test_robustness",
    "ml_validation.tests.test_calibration",
    "ml_validation.tests.test_failure_analysis",
    "ml_validation.tests.test_benchmark",
]


def main() -> int:
    t0 = time.perf_counter()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for mod_name in TEST_MODULES:
        mod = importlib.import_module(mod_name)
        suite.addTests(loader.loadTestsFromModule(mod))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed_s = time.perf_counter() - t0

    out_dir = _PKG_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "test_suite": "ml_validation/tests",
        "test_modules": TEST_MODULES,
        "tests_discovered_and_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "all_passed": result.wasSuccessful(),
        "elapsed_seconds": round(elapsed_s, 3),
    }
    (out_dir / "test_results.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\n[Test Summary] Ran {result.testsRun} tests in {elapsed_s:.2f}s — "
        f"Passed: {summary['passed']}, Failed: {summary['failures']}, Errors: {summary['errors']}"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
