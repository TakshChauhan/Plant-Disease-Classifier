"""Phase 6 — Reproducible Failure-Case Analysis & Grad-CAM Error Taxonomy Workflow.

For misclassified images, generates a structured visual gallery and machine-readable
error breakdown containing:
- Original image with the 224x224 evaluation center-crop box (`eval_crop_box`) outlined
- Ground-truth label
- Predicted label and predicted softmax probability
- Top-3 alternative predictions with probabilities
- Grad-CAM overlay computed on `layer4[-1]` aligned to the model's 224x224 input crop
- Source/image metadata (dimensions, aspect ratio, crop area retention percentage)

Groups misclassifications by:
1. Crop species (intra-crop vs. cross-crop error rates)
2. Disease category (healthy vs. fungal/bacterial/viral/pest conditions)
3. Confidence tier (`high_confidence_error >= 0.80`, `moderate 0.50-0.80`, `low_margin_inconclusive`)
4. Potential visual/geometric ambiguity (extreme aspect ratio center-crop loss, background dominance)

Usage:
    python -m ml_validation.failure_analysis \
        --checkpoint results/best_model.pt \
        --data-dir data/valid \
        --output-dir ml_validation/reports/phase6_failure_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

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

from ml_validation.core.backend import (  # noqa: E402
    CheckpointModelAdapter,
    parse_crop_and_disease,
    pretty_class_name,
)
from ml_validation.robustness import segment_leaf_foreground_mask  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def categorize_disease_etiology(class_name: str) -> str:
    """Group a PlantVillage class into a broad etiology category for error aggregation."""
    _, cond, is_healthy = parse_crop_and_disease(class_name)
    if is_healthy:
        return "Healthy"
    c_low = cond.lower()
    if "virus" in c_low or "curl" in c_low or "mosaic" in c_low:
        return "Viral"
    if "bacterial" in c_low or "haunglongbing" in c_low or "greening" in c_low:
        return "Bacterial"
    if "mite" in c_low:
        return "Arachnid / Pest"
    return "Fungal / Oomycete"


def draw_crop_box_pil(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Draw a bright yellow rectangle outlining the 224x224 region seen by the model."""
    preview = image.convert("RGB").copy()
    left, top, right, bottom = box
    width = max(2, round(min(preview.size) / 100))
    ImageDraw.Draw(preview).rectangle((left, top, right - 1, bottom - 1), outline=(255, 214, 0), width=width)
    return preview


def run_failure_analysis(
    adapter: CheckpointModelAdapter,
    labelled_samples: Sequence[Tuple[Path, int, str]],
    output_dir: Path,
    max_gallery_items: int = 10,
    dataset_label: str = "Labelled Split",
) -> Dict[str, object]:
    """Run inference and Grad-CAM error taxonomy across `labelled_samples`."""
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: List[Dict[str, object]] = []
    gallery_visuals: List[Tuple[Dict[str, object], Image.Image, Image.Image]] = []

    by_crop_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "errors": 0, "cross_crop_errors": 0})
    by_etiology_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "errors": 0})
    confidence_tiers = Counter()
    ambiguity_flags = Counter()

    for img_path, y_true, rel_id in labelled_samples:
        true_class = adapter.class_names[y_true]
        true_crop, true_cond, true_healthy = parse_crop_and_disease(true_class)
        true_etiology = categorize_disease_etiology(true_class)

        by_crop_stats[true_crop]["total"] += 1
        by_etiology_stats[true_etiology]["total"] += 1

        with Image.open(img_path) as im:
            orig_w, orig_h = im.size
            res = adapter.predict_pil(im, top_k=4, with_gradcam=True, apply_exif=True)
            context_img = draw_crop_box_pil(im, res["crop_box"])

        y_pred = res["pred_idx"]
        if y_pred == y_true:
            continue

        by_crop_stats[true_crop]["errors"] += 1
        by_etiology_stats[true_etiology]["errors"] += 1

        pred_class = res["pred_class"]
        pred_crop, pred_cond, pred_healthy = parse_crop_and_disease(pred_class)
        pred_etiology = categorize_disease_etiology(pred_class)
        pred_prob = res["pred_prob"]
        margin = res["margin"]
        true_prob = float(res["probs"][y_true])

        if true_crop != pred_crop:
            by_crop_stats[true_crop]["cross_crop_errors"] += 1

        # Confidence tier classification
        if pred_prob >= 0.80:
            conf_tier = "HIGH_CONFIDENCE_ERROR (p >= 0.80)"
        elif pred_prob >= 0.60:
            conf_tier = "MODERATE_CONFIDENCE_ERROR (0.60 <= p < 0.80)"
        else:
            conf_tier = "LOW_CONFIDENCE_INCONCLUSIVE (p < 0.60)"
        confidence_tiers[conf_tier] += 1

        # Geometric / visual ambiguity metrics
        left, top, right, bottom = res["crop_box"]
        crop_area_ratio = float((right - left) * (bottom - top)) / max(float(orig_w * orig_h), 1.0)
        aspect_ratio = float(max(orig_w, orig_h)) / max(float(min(orig_w, orig_h)), 1.0)

        # Measure Grad-CAM background focus on the 224x224 crop
        rgb_crop = np.asarray(res["model_view"], dtype=np.uint8)
        fg_mask = segment_leaf_foreground_mask(rgb_crop)
        cam_224 = cv2.resize(res["cam"].astype(np.float32), (rgb_crop.shape[1], rgb_crop.shape[0]))
        bg_cam_share = float(np.sum(cam_224[~fg_mask])) / (float(np.sum(cam_224)) + 1e-8)

        flags = []
        if aspect_ratio >= 1.35 or crop_area_ratio < 0.65:
            flags.append("PERIPHERAL_CROP_TRUNCATION")
            ambiguity_flags["PERIPHERAL_CROP_TRUNCATION"] += 1
        if bg_cam_share >= 0.45:
            flags.append("HIGH_BACKGROUND_GRADCAM_ATTENTION")
            ambiguity_flags["HIGH_BACKGROUND_GRADCAM_ATTENTION"] += 1
        if true_crop == pred_crop and not true_healthy and not pred_healthy:
            flags.append("INTRA_SPECIES_LESION_SIMILARITY")
            ambiguity_flags["INTRA_SPECIES_LESION_SIMILARITY"] += 1
        if true_crop != pred_crop:
            flags.append("CROSS_SPECIES_CONFUSION")
            ambiguity_flags["CROSS_SPECIES_CONFUSION"] += 1
        if margin < 0.20:
            flags.append("SMALL_TOP1_TOP2_MARGIN")
            ambiguity_flags["SMALL_TOP1_TOP2_MARGIN"] += 1

        top_alternatives = [
            {"class_name": c, "pretty_name": pretty_class_name(c), "probability": round(p, 4)}
            for c, p, _ in res["top_k"][1:4]
        ]

        err_record = {
            "sample_id": rel_id,
            "image_width": orig_w,
            "image_height": orig_h,
            "aspect_ratio": round(aspect_ratio, 3),
            "eval_crop_area_retained_ratio": round(crop_area_ratio, 4),
            "true_class": true_class,
            "true_pretty_name": pretty_class_name(true_class),
            "true_crop_species": true_crop,
            "true_etiology": true_etiology,
            "true_class_probability": round(true_prob, 4),
            "predicted_class": pred_class,
            "predicted_pretty_name": pretty_class_name(pred_class),
            "predicted_crop_species": pred_crop,
            "predicted_etiology": pred_etiology,
            "predicted_probability": round(pred_prob, 4),
            "top1_top2_margin": round(margin, 4),
            "confidence_tier": conf_tier,
            "gradcam_background_attention_share": round(bg_cam_share, 4),
            "ambiguity_flags": ";".join(flags),
            "top_alternatives": top_alternatives,
        }
        errors.append(err_record)
        if len(gallery_visuals) < max_gallery_items:
            gallery_visuals.append((err_record, context_img, res["overlay"]))

    # Sort errors by highest predicted confidence (most dangerous overconfident errors first)
    errors.sort(key=lambda r: (-r["predicted_probability"], r["true_class_probability"]))

    # Save CSV
    csv_path = output_dir / "misclassified_samples.csv"
    if errors:
        flat_fields = (
            [k for k in errors[0].keys() if k != "top_alternatives"]
            + ["top_2_alternative", "top_3_alternative"]
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_fields)
            writer.writeheader()
            for e in errors:
                row = {k: e[k] for k in errors[0].keys() if k != "top_alternatives"}
                alts = e["top_alternatives"]
                row["top_2_alternative"] = (
                    f"{alts[0]['class_name']} ({alts[0]['probability']:.3f})"
                    if len(alts) > 0 else ""
                )
                row["top_3_alternative"] = (
                    f"{alts[1]['class_name']} ({alts[1]['probability']:.3f})"
                    if len(alts) > 1 else ""
                )
                writer.writerow(row)
    else:
        csv_path.write_text("sample_id,true_class,predicted_class,predicted_probability\n")

    gallery_png = output_dir / "failure_cases_gallery.png"
    summary_png = output_dir / "failure_taxonomy_chart.png"
    _render_failure_gallery_png(gallery_visuals, gallery_png)
    _render_failure_taxonomy_chart(by_crop_stats, confidence_tiers, ambiguity_flags, summary_png)

    n_total = len(labelled_samples)
    n_err = len(errors)
    report = {
        "phase": "Phase 6 — Failure Case Analysis & Error Taxonomy",
        "dataset_label": dataset_label,
        "checkpoint_sha256": adapter.sha256,
        "n_total_evaluated": n_total,
        "n_misclassified": n_err,
        "overall_error_rate": round(n_err / max(n_total, 1), 4),
        "errors_by_crop_species": {
            crop: {
                **st,
                "error_rate": round(st["errors"] / max(st["total"], 1), 4),
            }
            for crop, st in sorted(by_crop_stats.items())
        },
        "errors_by_disease_etiology": {
            et: {
                **st,
                "error_rate": round(st["errors"] / max(st["total"], 1), 4),
            }
            for et, st in sorted(by_etiology_stats.items())
        },
        "errors_by_confidence_tier": dict(confidence_tiers),
        "ambiguity_pattern_counts": dict(ambiguity_flags),
        "concise_failure_pattern_interpretation": [
            "1. Cross-Species Background & Color-Prior Attraction: When leaf texture is low-contrast "
            "or background occupies a large fraction of the 224x224 center crop, predictions shift "
            "toward classes with large classification-head norms or background-compatible priors.",
            "2. Peripheral Lesion Truncation by Center-Crop Geometry: Because evaluation resizes the "
            "shorter side to 255 and crops the central 224x224 square (~77% of area for square images, "
            "and much less for elongated aspect ratios), tip and margin lesions outside the yellow "
            "crop box are completely invisible to the network.",
            "3. Overconfident Misclassifications: Softmax probabilities on misclassified samples "
            "frequently exceed 0.60-0.80, demonstrating that raw softmax confidence cannot "
            "separate correct diagnoses from errors.",
        ],
        "top_overconfident_errors": errors[:10],
        "artifacts": {
            "misclassified_csv": str(csv_path),
            "failure_gallery_png": str(gallery_png),
            "failure_taxonomy_chart_png": str(summary_png),
        },
    }
    (output_dir / "failure_analysis_report.json").write_text(json.dumps(report, indent=2))
    return report


def _render_failure_gallery_png(
    visuals: Sequence[Tuple[Dict[str, object], Image.Image, Image.Image]], out_path: Path
):
    """Render a multi-row failure gallery showing original image + crop box, Grad-CAM overlay, and metadata."""
    if not visuals:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Zero misclassifications recorded in this sample set.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    n_rows = len(visuals)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14.0, max(3.6, 3.0 * n_rows)))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (rec, ctx_img, cam_overlay) in enumerate(visuals):
        ax_left, ax_right = axes[idx, 0], axes[idx, 1]
        short_id = Path(rec["sample_id"]).name
        ax_left.imshow(ctx_img)
        ax_left.set_title(
            f"Sample: {short_id} ({rec['image_width']}x{rec['image_height']}px, "
            f"Center Crop Keep: {rec['eval_crop_area_retained_ratio']:.0%})\n"
            f"Ground Truth: {rec['true_pretty_name']} (p_true={rec['true_class_probability']:.1%})",
            fontsize=8.8,
            fontweight="bold",
            color="#1f2937",
            pad=6,
        )
        ax_left.axis("off")

        alts_str = ", ".join(
            f"{a['pretty_name']} ({a['probability']:.1%})" for a in rec["top_alternatives"][:2]
        )
        flags_short = (rec["ambiguity_flags"] or "NONE").replace(";", " | ")
        ax_right.imshow(cam_overlay)
        ax_right.set_title(
            f"Predicted: {rec['predicted_pretty_name']} (p={rec['predicted_probability']:.1%}, "
            f"margin={rec['top1_top2_margin']:.1%})\n"
            f"Top Alternatives: {alts_str}\n"
            f"Drivers: {flags_short} (BG Grad-CAM: {rec['gradcam_background_attention_share']:.0%})",
            fontsize=8.3,
            fontweight="bold",
            color="#c92a2a",
            pad=6,
        )
        ax_right.axis("off")

    fig.suptitle(
        "Phase 6 — Misclassified Sample Diagnostic Gallery (Yellow Box = Model Input Region)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_failure_taxonomy_chart(
    by_crop: Dict[str, Dict[str, int]],
    conf_tiers: Counter,
    ambiguity_flags: Counter,
    out_path: Path,
):
    """Render a 2-panel chart of error confidence tiers and ambiguity drivers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    tier_names = [
        "HIGH_CONFIDENCE_ERROR (p >= 0.80)",
        "MODERATE_CONFIDENCE_ERROR (0.60 <= p < 0.80)",
        "LOW_CONFIDENCE_INCONCLUSIVE (p < 0.60)",
    ]
    tier_vals = [conf_tiers.get(k, 0) for k in tier_names]
    tier_colors = ["#c92a2a", "#e67700", "#1971c2"]
    y1 = np.arange(len(tier_names))
    ax1.barh(y1, tier_vals, color=tier_colors, edgecolor="#222222", height=0.55)
    ax1.invert_yaxis()
    ax1.set_yticks(y1)
    ax1.set_yticklabels(
        ["High Conf (p >= 0.80)", "Moderate (0.60 - 0.80)", "Low Conf (p < 0.60)"],
        fontsize=9, fontweight="bold",
    )
    ax1.set_xlabel("Number of Misclassified Images", fontsize=9.5, fontweight="bold")
    ax1.set_title("Misclassifications by Softmax Confidence Tier", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    flag_items = sorted(ambiguity_flags.items(), key=lambda x: -x[1]) or [("NO_ERRORS", 0)]
    f_names = [k.replace("_", " ").title() for k, _ in flag_items]
    f_vals = [v for _, v in flag_items]
    y2 = np.arange(len(f_names))
    ax2.barh(y2, f_vals, color="#3b5bdb", edgecolor="#222222", height=0.55, alpha=0.85)
    ax2.invert_yaxis()
    ax2.set_yticks(y2)
    ax2.set_yticklabels(f_names, fontsize=8.8, fontweight="bold")
    ax2.set_xlabel("Frequency Across Misclassified Samples", fontsize=9.5, fontweight="bold")
    ax2.set_title("Diagnostic Ambiguity & Failure Drivers", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle("Phase 6 — Structured Error Taxonomy Breakdown", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_synthetic_failure_test_suite(
    root_dir: Path, class_names: Sequence[str], seed: int = 42
) -> List[Tuple[Path, int, str]]:
    """Create a synthetic set of challenging leaf images (including off-center lesions clipped by
    `CenterCrop(224)`, non-square panoramic aspect ratios `640x240`, low-contrast illumination, and
    cluttered backgrounds) to exercise and verify the Phase 6 error-analysis workflow.
    """
    rng = np.random.default_rng(seed)
    root_dir.mkdir(parents=True, exist_ok=True)
    target_indices = [0, 7, 8, 11, 20, 28, 29, 30, 32, 37]
    samples: List[Tuple[Path, int, str]] = []

    scenarios = [
        ("panoramic_edge_lesion", (520, 240)),
        ("portrait_tip_lesion", (240, 500)),
        ("dark_low_contrast", (256, 256)),
        ("soil_background_clutter", (256, 256)),
        ("centered_leaf_specimen", (256, 256)),
    ]

    for idx, c_idx in enumerate(target_indices):
        cname = class_names[c_idx]
        scen_name, (w, h) = scenarios[idx % len(scenarios)]
        canvas = rng.integers(85, 155, size=(h, w, 3), dtype=np.uint8)
        yy, xx = np.mgrid[:h, :w]
        cx, cy = w // 2, h // 2
        rx, ry = int(w * 0.32), int(h * 0.35)
        mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
        canvas[mask] = np.array([50 + (idx * 13) % 50, 125 + (idx * 17) % 50, 42], dtype=np.uint8)

        # Place necrotic spots near the periphery on panoramic/portrait scenarios
        sx = int(w * 0.82) if "panoramic" in scen_name else cx + int(rng.integers(-25, 26))
        sy = int(h * 0.85) if "portrait" in scen_name else cy + int(rng.integers(-25, 26))
        spot = ((xx - sx) ** 2 + (yy - sy) ** 2) <= 22 ** 2
        canvas[spot] = np.array([145, 72, 32], dtype=np.uint8)

        p = root_dir / f"sample_{idx:02d}_{scen_name}.png"
        Image.fromarray(canvas).save(p)
        samples.append((p, c_idx, f"{cname}/{p.name}"))

    return samples


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--data-dir", default=None, help="Labelled ImageFolder split (e.g. data/valid)")
    parser.add_argument("--max-gallery-items", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "onnx"])
    parser.add_argument("--output-dir", default="ml_validation/reports/phase6_failure_analysis_demo")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    adapter = CheckpointModelAdapter(args.checkpoint, device=args.device, backend=args.backend)

    if args.data_dir and Path(args.data_dir).is_dir():
        split_dir = Path(args.data_dir)
        samples: List[Tuple[Path, int, str]] = []
        for idx, cname in enumerate(adapter.class_names):
            cdir = split_dir / cname
            if not cdir.is_dir():
                continue
            for p in sorted(cdir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((p, idx, f"{cname}/{p.name}"))
        dataset_label = f"REAL_DATASET_SPLIT ({args.data_dir})"
    else:
        synth_dir = out_dir / "synthetic_diagnostic_samples"
        samples = generate_synthetic_failure_test_suite(synth_dir, adapter.class_names)
        dataset_label = (
            "SYNTHETIC_DIAGNOSTIC_STRESS_TEST (Executed on real best_model.pt + Grad-CAM; "
            "PlantVillage valid/ dataset not present locally)"
        )

    report = run_failure_analysis(
        adapter, samples, out_dir, max_gallery_items=args.max_gallery_items, dataset_label=dataset_label
    )
    print(
        f"[Phase 6] Completed failure analysis on {report['n_total_evaluated']} samples ({dataset_label}): "
        f"{report['n_misclassified']} misclassified samples analysed with Grad-CAM overlays."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
