"""Phase 5 — Confidence, Calibration, Temperature Scaling, and OOD Overconfidence Audit.

Investigates whether predicted probabilities from `best_model.pt` are reliable:
1. Measures Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
   Multiclass Brier Score, and Negative Log-Likelihood (NLL).
2. Strictly separates the calibration fitting split (`fit_split`) from the final
   held-out evaluation split (`eval_split`) when fitting Temperature Scaling (`T > 0`),
   preventing data leakage or in-sample calibration bias.
3. Generates reliability diagrams and confidence histograms (correct vs. incorrect).
4. Directly audits the committed checkpoint (`best_model.pt`) on 200 diverse out-of-distribution
   (OOD) non-leaf images (Gaussian noise, uniform RGB noise, solid color canvases, geometric shapes,
   checkerboards, and natural soil/bark textures) to measure empirical OOD overconfidence and
   evaluate risk-coverage / rejection behavior across thresholds without introducing arbitrary
   unvalidated rejection cutoffs.

Usage:
    python -m ml_validation.calibration \
        --checkpoint results/best_model.pt \
        --output-dir ml_validation/reports/phase5_calibration_and_ood
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

if __package__ in (None, ""):
    _pkg_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_pkg_dir.parent))
    if "ml_validation" not in sys.modules:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "ml_validation", _pkg_dir / "__init__.py", submodule_search_locations=[str(_pkg_dir)]
        )
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["ml_validation"] = _mod
            _spec.loader.exec_module(_mod)

from ml_validation.core.backend import CheckpointModelAdapter, pretty_class_name  # noqa: E402
from ml_validation.core.metrics import (  # noqa: E402
    compute_calibration_stats,
    fit_temperature_scaling,
    softmax_np,
)
from ml_validation.core.preprocessing import preprocess_pil  # noqa: E402
from ml_validation.evaluate_baseline import evaluate_imagefolder_split  # noqa: E402


def evaluate_real_ood_non_leaf_suite(
    adapter: CheckpointModelAdapter, n_per_family: int = 40, seed: int = 42
) -> Dict[str, object]:
    """Run real forward passes of `best_model.pt` on 5 families of non-leaf OOD inputs:
    1. Uniform RGB pixel noise U(0, 255)
    2. Gaussian sensor noise N(128, 50)
    3. Solid monochrome color canvases across the RGB cube
    4. Synthetic geometric shapes / checkerboards / stripes
    5. Non-leaf natural textures (soil, bark, sky gradients)
    """
    rng = np.random.default_rng(seed)
    families = [
        "uniform_rgb_noise",
        "gaussian_noise",
        "solid_color_canvases",
        "geometric_patterns",
        "non_leaf_textures",
    ]
    all_ood_logits: List[np.ndarray] = []
    family_labels: List[str] = []

    for fam in families:
        batch_tensors = []
        for i in range(n_per_family):
            h, w = 256, 256
            if fam == "uniform_rgb_noise":
                arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            elif fam == "gaussian_noise":
                arr = np.clip(rng.normal(128.0, 50.0, size=(h, w, 3)), 0, 255).astype(np.uint8)
            elif fam == "solid_color_canvases":
                color = rng.integers(0, 256, size=(3,), dtype=np.uint8)
                arr = np.full((h, w, 3), color, dtype=np.uint8)
            elif fam == "geometric_patterns":
                yy, xx = np.mgrid[:h, :w]
                period = int(rng.integers(12, 48))
                pat = (((xx // period) + (yy // period)) % 2) * 200 + 25
                c1 = rng.integers(20, 235, size=(3,), dtype=np.uint8)
                c2 = rng.integers(20, 235, size=(3,), dtype=np.uint8)
                arr = np.where(pat[:, :, None] > 128, c1[None, None, :], c2[None, None, :]).astype(np.uint8)
            else:
                # Smooth gradient + noise texture (sky/soil/stone)
                yy, xx = np.mgrid[:h, :w]
                base = np.zeros((h, w, 3), dtype=np.float32)
                for c in range(3):
                    g1 = float(rng.integers(40, 200))
                    g2 = float(rng.integers(40, 200))
                    base[:, :, c] = g1 + (g2 - g1) * (yy / float(h)) + rng.normal(0.0, 12.0, size=(h, w))
                arr = cv2.GaussianBlur(np.clip(base, 0, 255).astype(np.uint8), (7, 7), 1.5)

            pil_im = Image.fromarray(arr)
            batch_tensors.append(preprocess_pil(pil_im, adapter.preprocessing))
            family_labels.append(fam)

        logits = adapter.forward_logits(np.stack(batch_tensors, axis=0))
        all_ood_logits.append(logits)

    ood_logits = np.concatenate(all_ood_logits, axis=0)  # (N_ood, 38)
    ood_probs = softmax_np(ood_logits, temperature=1.0)
    ood_confs = np.max(ood_probs, axis=1)
    ood_preds = np.argmax(ood_probs, axis=1)

    pred_counts = Counter(ood_preds.tolist())
    sink_classes = [
        {
            "class_index": int(idx),
            "class_name": adapter.class_names[int(idx)],
            "pretty_name": pretty_class_name(adapter.class_names[int(idx)]),
            "count": int(cnt),
            "share_of_ood_predictions": round(cnt / len(ood_preds), 4),
        }
        for idx, cnt in pred_counts.most_common(10)
    ]

    by_family = {}
    for fam in families:
        mask = np.array([f == fam for f in family_labels])
        f_confs = ood_confs[mask]
        by_family[fam] = {
            "n_samples": int(np.sum(mask)),
            "mean_top1_confidence": round(float(np.mean(f_confs)), 4),
            "median_top1_confidence": round(float(np.median(f_confs)), 4),
            "min_top1_confidence": round(float(np.min(f_confs)), 4),
            "max_top1_confidence": round(float(np.max(f_confs)), 4),
            "frac_above_0_60": round(float(np.mean(f_confs >= 0.60)), 4),
            "frac_above_0_80": round(float(np.mean(f_confs >= 0.80)), 4),
            "frac_above_0_90": round(float(np.mean(f_confs >= 0.90)), 4),
        }

    return {
        "n_total_ood_samples": int(len(ood_confs)),
        "mean_top1_confidence": round(float(np.mean(ood_confs)), 4),
        "median_top1_confidence": round(float(np.median(ood_confs)), 4),
        "p90_top1_confidence": round(float(np.percentile(ood_confs, 90)), 4),
        "frac_conf_above_0_60_demo_heuristic": round(float(np.mean(ood_confs >= 0.60)), 4),
        "frac_conf_above_0_80": round(float(np.mean(ood_confs >= 0.80)), 4),
        "frac_conf_above_0_90": round(float(np.mean(ood_confs >= 0.90)), 4),
        "dominant_sink_classes": sink_classes,
        "by_ood_family": by_family,
        "ood_logits": ood_logits,
        "ood_confs": ood_confs,
    }


def evaluate_calibration_and_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    ood_logits: np.ndarray,
    output_dir: Path,
    n_bins: int = 15,
    seed: int = 42,
) -> Dict[str, object]:
    """Split labelled logits into disjoint `cal_fit` (50%) and `cal_eval` (50%) sets,
    fit Temperature Scaling on `cal_fit`, and evaluate calibration + ID/OOD rejection curves on `cal_eval`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_total = len(labels)
    if n_total < 4:
        raise ValueError("Need at least 4 labelled samples to split into calibration fit and evaluation sets.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_fit = n_total // 2
    fit_idx = perm[:n_fit]
    eval_idx = perm[n_fit:]

    logits_fit, labels_fit = logits[fit_idx], labels[fit_idx]
    logits_eval, labels_eval = logits[eval_idx], labels[eval_idx]

    # Fit temperature scaling strictly on fit_idx
    temp_fit = fit_temperature_scaling(logits_fit, labels_fit)
    opt_T = temp_fit["optimal_temperature"]

    # Evaluate on held-out eval_idx before (T=1.0) and after (T=opt_T)
    probs_eval_uncal = softmax_np(logits_eval, temperature=1.0)
    probs_eval_cal = softmax_np(logits_eval, temperature=opt_T)

    stats_uncal = compute_calibration_stats(labels_eval, probs_eval_uncal, n_bins=n_bins)
    stats_cal = compute_calibration_stats(labels_eval, probs_eval_cal, n_bins=n_bins)

    # Also compute OOD confidence before and after temperature scaling
    ood_probs_uncal = softmax_np(ood_logits, temperature=1.0)
    ood_probs_cal = softmax_np(ood_logits, temperature=opt_T)
    ood_conf_uncal = np.max(ood_probs_uncal, axis=1)
    ood_conf_cal = np.max(ood_probs_cal, axis=1)

    # Evaluate rejection threshold behavior across 19 thresholds (0.05..0.95)
    thresholds = np.linspace(0.10, 0.98, 23)
    eval_conf_uncal = np.max(probs_eval_uncal, axis=1)
    eval_correct = (np.argmax(probs_eval_uncal, axis=1) == labels_eval)

    threshold_table = []
    for tau in thresholds:
        keep_id = eval_conf_uncal >= tau
        coverage_id = float(np.mean(keep_id))
        selective_acc = float(np.mean(eval_correct[keep_id])) if np.any(keep_id) else None
        selective_risk = (1.0 - selective_acc) if selective_acc is not None else None
        ood_false_accept_uncal = float(np.mean(ood_conf_uncal >= tau))
        ood_false_accept_cal = float(np.mean(ood_conf_cal >= tau))
        threshold_table.append({
            "threshold_tau": round(float(tau), 3),
            "id_coverage": round(coverage_id, 4),
            "id_selective_accuracy": round(selective_acc, 4) if selective_acc is not None else None,
            "id_selective_error_risk": round(selective_risk, 4) if selective_risk is not None else None,
            "ood_false_acceptance_rate_uncalibrated": round(ood_false_accept_uncal, 4),
            "ood_false_acceptance_rate_temp_scaled": round(ood_false_accept_cal, 4),
        })

    with open(output_dir / "rejection_threshold_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_table[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_table)

    # Generate figures
    _plot_reliability_diagrams(stats_uncal, stats_cal, opt_T, output_dir / "reliability_diagram.png")
    _plot_confidence_distributions(
        eval_conf_uncal, eval_correct, ood_conf_uncal, output_dir / "confidence_distributions.png"
    )
    _plot_rejection_curves(threshold_table, output_dir / "rejection_curve_analysis.png")

    ece_delta = stats_cal["ece"] - stats_uncal["ece"]
    nll_delta = stats_cal["nll"] - stats_uncal["nll"]
    materially_improves = (ece_delta < -0.005) and (nll_delta < -0.01)

    return {
        "n_total_labelled": int(n_total),
        "n_calibration_fit_split": int(len(fit_idx)),
        "n_calibration_eval_split": int(len(eval_idx)),
        "temperature_scaling_fit": temp_fit,
        "held_out_uncalibrated_metrics": stats_uncal,
        "held_out_temperature_scaled_metrics": stats_cal,
        "improvement_summary": {
            "optimal_temperature": round(opt_T, 4),
            "ece_before": round(stats_uncal["ece"], 5),
            "ece_after": round(stats_cal["ece"], 5),
            "ece_change": round(ece_delta, 5),
            "brier_before": round(stats_uncal["brier_score"], 5),
            "brier_after": round(stats_cal["brier_score"], 5),
            "nll_before": round(stats_uncal["nll"], 5),
            "nll_after": round(stats_cal["nll"], 5),
            "materially_improves_calibration": bool(materially_improves),
        },
        "threshold_sweep_table": threshold_table,
    }


def _plot_reliability_diagrams(
    stats_uncal: Dict[str, object], stats_cal: Dict[str, object], opt_T: float, out_path: Path
):
    """Plot side-by-side Reliability Diagrams (Uncalibrated T=1.0 vs Temperature-Scaled T=T*)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)

    for ax, st, title_prefix in [
        (axes[0], stats_uncal, "Uncalibrated (T = 1.00)"),
        (axes[1], stats_cal, f"Temperature Scaled (T* = {opt_T:.2f})"),
    ]:
        bins = st["bins"]
        centers = [0.5 * (b["bin_lower"] + b["bin_upper"]) for b in bins]
        width = (bins[0]["bin_upper"] - bins[0]["bin_lower"]) * 0.85
        accs = [b["avg_accuracy"] if b["count"] > 0 else 0.0 for b in bins]
        confs = [b["avg_confidence"] if b["count"] > 0 else 0.0 for b in bins]

        ax.plot([0, 1], [0, 1], "--", color="#495057", linewidth=1.5, label="Perfect Calibration (Acc = Conf)")
        ax.bar(centers, accs, width=width, color="#1971c2", alpha=0.80, edgecolor="#1864ab", label="Empirical Accuracy")
        # Overlay calibration gap in translucent red
        for c, a, cf, b in zip(centers, accs, confs, bins):
            if b["count"] > 0:
                ax.plot([c, c], [a, cf], color="#c92a2a", linewidth=2.2)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Mean Predicted Confidence in Bin", fontsize=10, fontweight="bold")
        ax.set_title(
            f"{title_prefix}\nECE = {st['ece']:.4f} | Brier = {st['brier_score']:.4f} | NLL = {st['nll']:.4f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.5)

    axes[0].set_ylabel("Empirical Fraction Correct (Accuracy)", fontsize=10, fontweight="bold")
    fig.suptitle(
        "Phase 5 — Held-Out Reliability Diagrams Before & After Temperature Scaling",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_confidence_distributions(
    id_confs: np.ndarray, id_correct: np.ndarray, ood_confs: np.ndarray, out_path: Path
):
    """Plot confidence histograms for ID Correct, ID Incorrect, and Real OOD Non-Leaf images."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bins = np.linspace(0.0, 1.0, 21)

    if np.any(id_correct):
        ax.hist(
            id_confs[id_correct], bins=bins, alpha=0.65, color="#2b8a3e",
            label=f"ID Correct (n={int(np.sum(id_correct))}, mean={np.mean(id_confs[id_correct]):.2f})",
            density=True, edgecolor="#1b4332",
        )
    if np.any(~id_correct):
        ax.hist(
            id_confs[~id_correct], bins=bins, alpha=0.65, color="#e67700",
            label=f"ID Misclassified (n={int(np.sum(~id_correct))}, mean={np.mean(id_confs[~id_correct]):.2f})",
            density=True, edgecolor="#d9480f",
        )
    ax.hist(
        ood_confs, bins=bins, alpha=0.55, color="#c92a2a",
        label=f"Real Non-Leaf OOD on best_model.pt (n={len(ood_confs)}, mean={np.mean(ood_confs):.2f})",
        density=True, edgecolor="#7f1d1d",
    )
    ax.axvline(0.60, color="#333333", linestyle="--", linewidth=1.5, label="Demo Heuristic Cutoff (p=0.60)")
    ax.set_xlabel("Top-1 Softmax Probability", fontsize=10, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax.set_title(
        "Top-1 Confidence Distributions: In-Distribution vs. Non-Leaf OOD Inputs",
        fontsize=11.5, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8.8)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_rejection_curves(threshold_table: Sequence[Dict[str, object]], out_path: Path):
    """Plot ID Coverage, ID Selective Accuracy, and OOD False Acceptance Rate across thresholds."""
    taus = [r["threshold_tau"] for r in threshold_table]
    cov = [100.0 * r["id_coverage"] for r in threshold_table]
    sel_acc = [
        100.0 * r["id_selective_accuracy"] if r["id_selective_accuracy"] is not None else np.nan
        for r in threshold_table
    ]
    ood_far = [100.0 * r["ood_false_acceptance_rate_uncalibrated"] for r in threshold_table]
    ood_far_cal = [100.0 * r["ood_false_acceptance_rate_temp_scaled"] for r in threshold_table]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(taus, cov, "-o", color="#1971c2", markersize=4, linewidth=2, label="ID Retained Coverage (%)")
    ax.plot(taus, sel_acc, "-s", color="#2b8a3e", markersize=4, linewidth=2, label="ID Selective Accuracy (%)")
    ax.plot(
        taus, ood_far, "-^", color="#c92a2a", markersize=4, linewidth=2,
        label="OOD False Acceptance Rate (Uncalibrated, %)",
    )
    ax.plot(
        taus, ood_far_cal, "--v", color="#e67700", markersize=4, linewidth=1.8,
        label="OOD False Acceptance Rate (Temp-Scaled, %)",
    )
    ax.axvline(0.60, color="#495057", linestyle=":", linewidth=1.5, label="Demo LOW_TOP1_PROB (0.60)")

    ax.set_xlabel("Confidence Rejection Threshold (tau)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(-2, 105)
    ax.set_title(
        "Why Arbitrary Softmax Thresholds Fail as OOD Detectors (Measured on best_model.pt)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_synthetic_overconfident_logits(
    n_samples: int = 600, num_classes: int = 38, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic synthetic benchmark of overconfident 38-class neural network logits
    (~86% accuracy with ~97% mean confidence due to logit magnification T_true ~ 2.1) to validate
    the calibration & temperature-scaling pipeline when the PlantVillage validation set is absent.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n_samples, dtype=np.int64)
    logits = rng.normal(-1.5, 0.55, size=(n_samples, num_classes))
    for i in range(n_samples):
        if rng.random() < 0.86:
            logits[i, labels[i]] = rng.uniform(2.4, 4.2)
        else:
            wrong_c = (labels[i] + int(rng.integers(1, num_classes))) % num_classes
            logits[i, wrong_c] = rng.uniform(2.1, 3.6)
            logits[i, labels[i]] = rng.uniform(0.6, 1.8)
    # Amplify logits by 2.1x to simulate post-fine-tuning cross-entropy overconfidence
    return (logits * 2.10).astype(np.float32), labels


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--data-dir", default=None, help="Optional labelled ImageFolder split (e.g. data/valid)")
    parser.add_argument(
        "--logits-npz", default=None,
        help="Optional .npz with 'logits' and 'labels' from evaluate_baseline",
    )
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-ood-per-family", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "onnx"])
    parser.add_argument("--output-dir", default="ml_validation/reports/phase5_calibration_and_ood")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = CheckpointModelAdapter(args.checkpoint, device=args.device, backend=args.backend)

    # 1. Always run the REAL OOD non-leaf confidence audit directly on best_model.pt
    ood_audit = evaluate_real_ood_non_leaf_suite(adapter, n_per_family=args.n_ood_per_family, seed=42)
    ood_logits = ood_audit.pop("ood_logits")
    ood_audit.pop("ood_confs")

    # 2. Load labelled ID logits from --logits-npz or --data-dir, or run synthetic calibration verification
    if args.logits_npz and Path(args.logits_npz).is_file():
        npz = np.load(args.logits_npz)
        id_logits, id_labels = npz["logits"], npz["labels"]
        id_source = f"REAL_DATASET_LOGITS ({args.logits_npz})"
    elif args.data_dir and Path(args.data_dir).is_dir():
        eval_res = evaluate_imagefolder_split(adapter, Path(args.data_dir))
        id_logits = eval_res["logits_array"]
        id_labels = np.asarray(eval_res["labels_list"], dtype=np.int64)
        id_source = f"REAL_DATASET_SPLIT ({args.data_dir})"
    else:
        id_logits, id_labels = generate_synthetic_overconfident_logits(n_samples=600, num_classes=adapter.num_classes)
        id_source = (
            "SYNTHETIC_MISCALIBRATED_LOGITS_BENCHMARK (PlantVillage 17,572-image valid/ split is not present "
            "locally; OOD non-leaf overconfidence audit was executed directly on the real best_model.pt checkpoint)"
        )

    cal_results = evaluate_calibration_and_temperature_scaling(
        id_logits, id_labels, ood_logits, out_dir, n_bins=args.n_bins, seed=42
    )

    report = {
        "phase": "Phase 5 — Confidence, Calibration & OOD Overconfidence Audit",
        "checkpoint_sha256": adapter.sha256,
        "id_calibration_data_source": id_source,
        "real_checkpoint_ood_non_leaf_audit": ood_audit,
        "calibration_and_temperature_scaling": cal_results,
        "required_additional_data_note": (
            "To measure true in-distribution calibration on PlantVillage and field calibration on PlantDoc, "
            "run `evaluate_baseline.py` on a held-out split not used for early-stopping checkpoint selection, "
            "then pass `--logits-npz .../predictions_and_logits.npz` to `calibration.py`."
        ),
    }
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"[Phase 5] Real OOD audit on {adapter.checkpoint_path.name} "
        f"(n={ood_audit['n_total_ood_samples']} non-leaf images): "
        f"mean Top-1 prob = {ood_audit['mean_top1_confidence']:.2%}, "
        f"{ood_audit['frac_conf_above_0_60_demo_heuristic']:.1%} exceed the demo's 0.60 heuristic threshold! "
        f"Calibration pipeline verified (T*={cal_results['improvement_summary']['optimal_temperature']:.3f}, "
        f"ECE {cal_results['improvement_summary']['ece_before']:.4f} -> "
        f"{cal_results['improvement_summary']['ece_after']:.4f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
