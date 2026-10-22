# Independent ML Validation and Research Suite (`ml_validation`)

A standalone, reproducible machine learning validation and research package for auditing the trained 38-class **Plant Disease Classifier** ([TakshChauhan/Plant-Disease-Classifier](https://github.com/TakshChauhan/Plant-Disease-Classifier), checkpoint `results/best_model.pt`, SHA-256 `f456fe4cd6042b2c031172bcdab00a40c845475a3f71fa052264d28311ed0cc9`).

## Design Principles & Non-Negotiable Safeguards

1. **Zero Modification to Main Project**: Lives entirely in `ml_validation/` and never modifies `src/`, `app/`, `tests/`, or `results/best_model.pt`.
2. **Zero Unsolicited Weight Downloads**: Never downloads pretrained ImageNet weights from the internet (`pretrained=False` / `weights=None` in PyTorch mode, and safe tensor-only zip unpickling in ONNX mode).
3. **Dual-Backend Execution (`torch` + `onnx`)**: Automatically uses native `torch` + `torchvision` when installed in the environment, and seamlessly falls back to a safe tensor-only `.pt` zip unpickler + bit-accurate ONNX Runtime graph compiler (`atol < 1e-5`) when run in lightweight validation environments.
4. **Scientific Integrity**: Strictly separates verified checkpoint measurements from unverified README claims (`99.35%` validation accuracy), separates confirmed exact/D4 duplicates from ambiguous perceptual hash matches, enforces strict taxonomic mapping on external field datasets, and never introduces arbitrary unvalidated confidence rejection thresholds.

## Package Structure

```text
ml_validation/
├── README.md                     # CLI usage & methodology overview
├── MODEL_CARD.md                 # Research-grade Model Card (Phase 8)
├── INTEGRATION_GUIDE.md          # Non-invasive integration guide with Claude Code's repo
├── VALIDATION_REPORT.md          # Complete empirical & forensic validation report (Phases 1-9)
├── requirements.txt              # Validation suite dependencies
├── class_mapping.json            # Auditable 38-class PlantVillage <-> PlantDoc mapping table
├── __init__.py                   # VFS-resilient package initializer
├── run_tests.py                  # Unified automated test runner (unittest & pytest compatible)
├── evaluate_baseline.py          # Phase 1: Baseline evaluation & checkpoint forensic audit
├── duplicate_audit.py            # Phase 2: Scalable SHA-256 + 8-orientation D4 perceptual hash leakage audit
├── evaluate_field.py             # Phase 3: External field generalization evaluator (PlantDoc)
├── robustness.py                 # Phase 4: Controlled perturbations, leaf segmentation & background replacement
├── calibration.py                # Phase 5: ECE/Brier/NLL, split-validation Temperature Scaling & OOD audit
├── failure_analysis.py           # Phase 6: Misclassified sample gallery & Grad-CAM error taxonomy
├── benchmark.py                  # Phase 7: Parameter, latency (p50/p90/p95/p99), throughput, memory & ONNX benchmark
├── core/
│   ├── __init__.py
│   ├── backend.py                # Dual-mode CheckpointModelAdapter & analytical/hook Grad-CAM
│   ├── metrics.py                # Top-1/Top-3, Macro/Weighted P/R/F1, Confusion Matrix, ECE/MCE/Brier/NLL
│   └── preprocessing.py          # Exact Resize(255) -> CenterCrop(224) -> ImageNet normalization & crop box
├── tests/                        # 21 automated unit & end-to-end integration tests
└── reports/                      # Pre-populated JSON/CSV reports and PNG charts from executed experiments
```

## Command-Line Interface Reference

### 1. Run the Automated Test Suite (Phase 8 & 9)

```bash
# Via standard library unittest (zero external test runner required):
python3 ml_validation/run_tests.py

# Or via pytest (in the main project's dev environment):
pytest ml_validation/tests
```

### 2. Phase 1 — Baseline Evaluation & Checkpoint Forensics

```bash
python3 ml_validation/evaluate_baseline.py \
    --checkpoint results/best_model.pt \
    --class-names-json results/class_names.json \
    --data-dir data \
    --split valid \
    --output-dir ml_validation/reports/phase1_checkpoint_audit
```

### 3. Phase 2 — Train vs. Validation Duplicate & Augmentation Leakage Audit

```bash
# Run on the extracted Kaggle PlantVillage dataset:
python3 ml_validation/duplicate_audit.py \
    --train-dir data/train \
    --val-dir data/valid \
    --max-dhash-hamming 16 \
    --output-dir ml_validation/reports/phase2_leakage_audit

# Or run with --run-synthetic-demo-if-missing to verify the D4 + BK-tree pipeline on synthetic splits:
python3 ml_validation/duplicate_audit.py \
    --train-dir data/train \
    --val-dir data/valid \
    --run-synthetic-demo-if-missing \
    --output-dir ml_validation/reports/phase2_leakage_audit_demo
```

### 4. Phase 3 — Real-World Field Generalization Evaluation (PlantDoc)

```bash
# Strict taxonomic mapping (excludes ambiguous 'Bell_pepper leaf spot' and unsupported crops):
python3 ml_validation/evaluate_field.py \
    --checkpoint results/best_model.pt \
    --field-dir data/PlantDoc-Dataset/test \
    --mapping-json ml_validation/class_mapping.json \
    --output-dir ml_validation/reports/phase3_field_generalization
```

### 5. Phase 4 — Robustness & Background Bias Suite

```bash
# Run on natural labelled validation images (or procedural leaf specimens if --data-dir is omitted):
python3 ml_validation/robustness.py \
    --checkpoint results/best_model.pt \
    --data-dir data/valid \
    --max-per-class 3 \
    --output-dir ml_validation/reports/phase4_robustness_and_bg_bias
```

### 6. Phase 5 — Confidence, Calibration, Temperature Scaling & OOD Audit

```bash
# Pass saved logits from Phase 1 (or --data-dir) plus automatic 150-image non-leaf OOD probe:
python3 ml_validation/calibration.py \
    --checkpoint results/best_model.pt \
    --logits-npz ml_validation/reports/phase1_checkpoint_audit/predictions_and_logits.npz \
    --output-dir ml_validation/reports/phase5_calibration_and_ood
```

### 7. Phase 6 — Failure Case Analysis & Grad-CAM Gallery

```bash
python3 ml_validation/failure_analysis.py \
    --checkpoint results/best_model.pt \
    --data-dir data/valid \
    --max-gallery-items 10 \
    --output-dir ml_validation/reports/phase6_failure_analysis
```

### 8. Phase 7 — Model Performance & Efficiency Benchmark

```bash
python3 ml_validation/benchmark.py \
    --checkpoint results/best_model.pt \
    --warmup-runs 10 \
    --timed-runs 50 \
    --batch-sizes 1 4 8 16 \
    --output-dir ml_validation/reports/phase7_efficiency_benchmark
```
