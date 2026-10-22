# Integration Guide — Adding `ml_validation` to `Plant-Disease-Classifier`

This guide describes how to integrate the standalone `ml_validation/` research and validation suite into [TakshChauhan/Plant-Disease-Classifier](https://github.com/TakshChauhan/Plant-Disease-Classifier) **without modifying any of Claude Code's repaired files** (`src/`, `app/`, `tests/`, `requirements*.txt`, or `results/best_model.pt`).

## 1. Non-Invasive Directory Placement

Copy the `ml_validation/` folder directly into the root of your repository alongside `src/`, `app/`, and `tests/`:

```text
Plant-Disease-Classifier/
├── app/
│   └── demo.py                 # Unchanged (Claude Code's repaired Gradio app)
├── src/
│   ├── dataset.py              # Unchanged
│   ├── evaluate.py             # Unchanged
│   ├── gradcam.py              # Unchanged
│   ├── model.py                # Unchanged
│   ├── train.py                # Unchanged
│   └── utils.py                # Unchanged
├── tests/                      # Unchanged (Claude Code's 98 unit tests)
├── results/
│   ├── best_model.pt           # Unchanged (SHA-256: f456fe4cd6042b2c...)
│   ├── class_names.json        # Unchanged
│   └── training_curves.png     # Unchanged
└── ml_validation/              # <--- Drop-in standalone validation & research suite
    ├── README.md
    ├── MODEL_CARD.md
    ├── INTEGRATION_GUIDE.md
    ├── VALIDATION_REPORT.md
    ├── requirements.txt
    ├── class_mapping.json
    ├── run_tests.py
    ├── evaluate_baseline.py
    ├── duplicate_audit.py
    ├── evaluate_field.py
    ├── robustness.py
    ├── calibration.py
    ├── failure_analysis.py
    ├── benchmark.py
    ├── core/
    ├── tests/
    └── reports/
```

## 2. Running Both Test Suites Together

In your Python 3.11 virtual environment (where `requirements-dev.txt` is installed):

```bash
# 1. Run Claude Code's existing 98 tests:
pytest tests/

# 2. Run the new 21 ML validation tests (via pytest or unittest):
pytest ml_validation/tests/
# OR:
python3 ml_validation/run_tests.py

# 3. Verify ruff linting passes cleanly across both suites:
ruff check src/ app/ tests/ ml_validation/
```

Notice that `pyproject.toml` currently sets `testpaths = ["tests"]`. Leaving `pyproject.toml` untouched keeps `pytest` running Claude Code's 98 tests by default, while `pytest ml_validation/tests` or `python3 ml_validation/run_tests.py` runs the validation suite on demand. If you later want `pytest` (with no arguments) to run both suites together, simply update `testpaths = ["tests", "ml_validation/tests"]` in `pyproject.toml`.

## 3. Executing the Full Validation Workflow on Local / Colab Datasets

Once you download the Kaggle **New Plant Diseases Dataset (Augmented)** (`data/train` and `data/valid`) and the **PlantDoc Dataset** (`data/PlantDoc-Dataset/test`):

```bash
# Phase 1: Reproduce baseline metrics & save validation logits (.npz)
python3 ml_validation/evaluate_baseline.py \
    --checkpoint results/best_model.pt \
    --class-names-json results/class_names.json \
    --data-dir data \
    --split valid \
    --output-dir ml_validation/reports/phase1_plantvillage_valid

# Phase 2: Audit exact & D4 near-duplicate leakage across train/ and valid/
python3 ml_validation/duplicate_audit.py \
    --train-dir data/train \
    --val-dir data/valid \
    --output-dir ml_validation/reports/phase2_plantvillage_leakage

# Phase 3: Evaluate real-world field generalization on PlantDoc (strict taxonomy)
python3 ml_validation/evaluate_field.py \
    --checkpoint results/best_model.pt \
    --field-dir data/PlantDoc-Dataset/test \
    --mapping-json ml_validation/class_mapping.json \
    --output-dir ml_validation/reports/phase3_plantdoc_field

# Phase 4: Measure perturbation & background-replacement sensitivity on natural leaves
python3 ml_validation/robustness.py \
    --checkpoint results/best_model.pt \
    --data-dir data/valid \
    --max-per-class 5 \
    --output-dir ml_validation/reports/phase4_robustness_valid

# Phase 5: Fit held-out Temperature Scaling & compare ID vs. OOD rejection curves
python3 ml_validation/calibration.py \
    --checkpoint results/best_model.pt \
    --logits-npz ml_validation/reports/phase1_plantvillage_valid/predictions_and_logits.npz \
    --output-dir ml_validation/reports/phase5_calibration_valid

# Phase 6: Generate Grad-CAM failure gallery and error taxonomy on misclassified images
python3 ml_validation/failure_analysis.py \
    --checkpoint results/best_model.pt \
    --data-dir data/valid \
    --output-dir ml_validation/reports/phase6_failures_valid

# Phase 7: Benchmark latency, throughput, memory & ONNX export on your target hardware
python3 ml_validation/benchmark.py \
    --checkpoint results/best_model.pt \
    --output-dir ml_validation/reports/phase7_benchmark
```

## 4. Optional GitHub Actions CI Integration

To run the `ml_validation` test suite automatically in CI without altering the existing test job, append a step to `.github/workflows/ci.yml`:

```yaml
      - name: Run ML Validation Suite Tests
        run: pytest ml_validation/tests
```
