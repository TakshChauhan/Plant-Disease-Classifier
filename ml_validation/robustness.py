"""Phase 4 — Robustness, Perturbation Sensitivity, and Background Bias Investigation.

Designs and executes controlled experiments to test whether the trained ResNet18
checkpoint (`best_model.pt`) relies on studio background shortcuts or is sensitive to
photometric and geometric acquisition shifts:
1. Lighting variation (brightness scaling, gamma correction)
2. Blur (Gaussian blur, directional motion blur)
3. Image noise (Gaussian sensor noise, salt-and-pepper impulse noise)
4. Color shifts (HSV hue rotation, saturation scaling, warm/cool white balance)
5. Background removal (Excess-Green + Otsu/morphological foreground leaf segmentation,
   replacing background with black or ImageNet mean while preserving leaf lesions)
6. Background replacement (compositing the foreground leaf onto soil brown, wood/bark texture,
   foliage clutter, and high-frequency random studio noise)
7. Direct Background-Only Shortcut Probe (evaluating the checkpoint on pure studio grey,
   black, white, soil, foliage, and noise canvases with NO leaf present).

Explicitly distinguishes natural field images from synthetic transformations, and notes that
coarse 7x7 Grad-CAM heatmaps alone do not prove biological feature learning.

Usage:
    python -m ml_validation.robustness \
        --checkpoint results/best_model.pt \
        --output-dir ml_validation/reports/phase4_robustness_and_bg_bias
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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

from ml_validation.core.backend import (  # noqa: E402
    CheckpointModelAdapter,
    overlay_heatmap_pil,
    pretty_class_name,
)
from ml_validation.core.preprocessing import eval_resize_and_crop_pil, preprocess_pil  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Foreground Leaf Segmentation & Controlled Perturbation Operators
# ---------------------------------------------------------------------------

def segment_leaf_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Segment the foreground leaf blade (including interior brown/yellow lesions) from a studio background.

    Uses a combination of:
    1. Chromaticity saturation + Excess Green/Red-Green index (`2G - R - B` and HSV saturation)
       to separate plant tissue from neutral grey/white/dark studio backgrounds.
    2. Morphological closing + flood-fill of interior holes so necrotic spots/lesions inside
       the leaf boundary are strictly preserved as foreground (`mask == True`).
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b

    # Estimate border background color from 6px outer rim
    rim_pixels = np.concatenate([
        rgb[:6, :, :].reshape(-1, 3),
        rgb[-6:, :, :].reshape(-1, 3),
        rgb[:, :6, :].reshape(-1, 3),
        rgb[:, -6:, :].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    bg_median = np.median(rim_pixels, axis=0)
    color_dist = np.linalg.norm(rgb.astype(np.float32) - bg_median[None, None, :], axis=2)

    raw_fg = ((sat > 42.0) & (val > 25.0)) | (exg > 24.0) | (color_dist > 48.0)
    mask_u8 = (raw_fg.astype(np.uint8)) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Strictly keep the single largest connected component and fill internal holes
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.03 * (h * w):
            cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)

    # Fallback to central ellipse if image is uniform
    if np.mean(filled > 0) < 0.05 or np.mean(filled > 0) > 0.96:
        yy, xx = np.ogrid[:h, :w]
        ell = (((xx - w / 2.0) / (0.35 * w)) ** 2 + ((yy - h / 2.0) / (0.40 * h)) ** 2) <= 1.0
        return ell

    return filled > 127


def apply_lighting_brightness(rgb: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    norm = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.clip((norm ** gamma) * 255.0, 0, 255).astype(np.uint8)


def apply_gaussian_blur(rgb: np.ndarray, ksize: int = 5, sigma: float = 2.0) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_motion_blur(rgb: np.ndarray, ksize: int = 9) -> np.ndarray:
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(rgb, -1, kernel)


def apply_gaussian_noise(rgb: np.ndarray, sigma: float = 20.0, seed: int = 101) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=rgb.shape).astype(np.float32)
    return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_salt_and_pepper(rgb: np.ndarray, amount: float = 0.03, seed: int = 102) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    u = rng.random(rgb.shape[:2])
    out[u < (amount / 2.0)] = 0
    out[u > 1.0 - (amount / 2.0)] = 255
    return out


def apply_hsv_shift(rgb: np.ndarray, dh: int = 12, sat_scale: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + dh) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_background_removal(rgb: np.ndarray, fill_rgb: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    mask = segment_leaf_foreground_mask(rgb)
    out = np.full_like(rgb, fill_rgb, dtype=np.uint8)
    out[mask] = rgb[mask]
    return out


def apply_background_replacement(rgb: np.ndarray, bg_type: str = "soil", seed: int = 202) -> np.ndarray:
    mask = segment_leaf_foreground_mask(rgb)
    h, w = rgb.shape[:2]
    rng = np.random.default_rng(seed)

    if bg_type == "soil":
        # Naturalistic mottled brown soil background
        base = np.array([102, 72, 44], dtype=np.float32)
        tex = rng.normal(0.0, 18.0, size=(h, w, 3)).astype(np.float32)
        tex = cv2.GaussianBlur(tex, (5, 5), 1.2)
        bg = np.clip(base[None, None, :] + tex, 0, 255).astype(np.uint8)
    elif bg_type == "wood_bark":
        # Streaked warm wood/bark background
        yy, xx = np.mgrid[:h, :w]
        stripes = 18.0 * np.sin(xx / 6.0 + 0.3 * np.sin(yy / 15.0))
        base = np.array([135, 95, 58], dtype=np.float32)
        noise = rng.normal(0.0, 8.0, size=(h, w, 3)).astype(np.float32)
        bg = np.clip(base[None, None, :] + stripes[:, :, None] + noise, 0, 255).astype(np.uint8)
    elif bg_type == "clutter_foliage":
        # Out-of-focus background canopy clutter
        raw = rng.integers(30, 140, size=(h, w, 3), dtype=np.uint8)
        raw[:, :, 1] = np.clip(raw[:, :, 1].astype(np.int32) + 40, 0, 255).astype(np.uint8)
        bg = cv2.GaussianBlur(raw, (15, 15), 5.0)
    elif bg_type == "random_noise":
        bg = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown background type: {bg_type}")

    bg[mask] = rgb[mask]
    return bg


def build_perturbation_suite() -> List[Tuple[str, str, Callable[[np.ndarray], np.ndarray]]]:
    """Return `(key, display_name, transform_fn)` for all controlled perturbations."""
    return [
        ("clean", "Original (Unperturbed)", lambda x: x),
        ("bright_low", "Lighting: Darkened (x0.65)", lambda x: apply_lighting_brightness(x, 0.65)),
        ("bright_high", "Lighting: Overexposed (x1.35)", lambda x: apply_lighting_brightness(x, 1.35)),
        ("gamma_dark", "Lighting: Gamma 1.6 (Shadowed)", lambda x: apply_gamma(x, 1.6)),
        ("blur_gauss", "Blur: Gaussian (5x5, sigma=2.0)", lambda x: apply_gaussian_blur(x, 5, 2.0)),
        ("blur_motion", "Blur: Horizontal Motion (9x9)", lambda x: apply_motion_blur(x, 9)),
        ("noise_gauss", "Noise: Gaussian (sigma=20)", lambda x: apply_gaussian_noise(x, 20.0)),
        ("noise_sp", "Noise: Salt & Pepper (3%)", lambda x: apply_salt_and_pepper(x, 0.03)),
        ("color_hue", "Color: Hue Shift (+12 deg)", lambda x: apply_hsv_shift(x, dh=12, sat_scale=1.0)),
        ("color_desat", "Color: Desaturated (x0.60)", lambda x: apply_hsv_shift(x, dh=0, sat_scale=0.60)),
        ("bg_removed_black", "Background Removed (Black Fill)", lambda x: apply_background_removal(x, (0, 0, 0))),
        (
            "bg_removed_mean", "Background Removed (ImageNet Mean)",
            lambda x: apply_background_removal(x, (124, 116, 104)),
        ),
        ("bg_soil", "Background Replaced: Soil Texture", lambda x: apply_background_replacement(x, "soil")),
        ("bg_bark", "Background Replaced: Wood/Bark", lambda x: apply_background_replacement(x, "wood_bark")),
        (
            "bg_foliage", "Background Replaced: Canopy Clutter",
            lambda x: apply_background_replacement(x, "clutter_foliage"),
        ),
        ("bg_noise", "Background Replaced: Random Noise", lambda x: apply_background_replacement(x, "random_noise")),
    ]


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute base-2 Jensen-Shannon divergence in [0, 1] between two probability vectors."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))


def probe_background_only_priors(adapter: CheckpointModelAdapter) -> List[Dict[str, object]]:
    """Evaluate `best_model.pt` on pure background canvases with NO leaf present.

    Reveals which classes act as background-prior attractors ('sink classes') when background
    textures dominate the 224x224 center crop.
    """
    h, w = 256, 256
    canvases: List[Tuple[str, Image.Image]] = [
        ("Zero Normalized Tensor (Exact ImageNet Mean)", Image.new("RGB", (w, h), (124, 116, 104))),
        ("Pure Black Studio Backdrop (RGB 0,0,0)", Image.new("RGB", (w, h), (0, 0, 0))),
        ("Dark Charcoal Backdrop (RGB 45,45,48)", Image.new("RGB", (w, h), (45, 45, 48))),
        ("PlantVillage Neutral Studio Grey (RGB 128,128,128)", Image.new("RGB", (w, h), (128, 128, 128))),
        ("Warm Studio Paper Grey (RGB 165,160,152)", Image.new("RGB", (w, h), (165, 160, 152))),
        ("Pure White Backdrop (RGB 255,255,255)", Image.new("RGB", (w, h), (255, 255, 255))),
        (
            "Field Soil Brown Texture",
            Image.fromarray(apply_background_replacement(np.zeros((h, w, 3), dtype=np.uint8), "soil")),
        ),
        (
            "Tree Bark / Wood Texture",
            Image.fromarray(apply_background_replacement(np.zeros((h, w, 3), dtype=np.uint8), "wood_bark")),
        ),
        (
            "Blurred Green Canopy Background",
            Image.fromarray(apply_background_replacement(np.zeros((h, w, 3), dtype=np.uint8), "clutter_foliage")),
        ),
        (
            "Uniform RGB Random Noise (seed=0)",
            Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)),
        ),
    ]

    results = []
    for label, pil_im in canvases:
        res = adapter.predict_pil(pil_im, top_k=3, with_gradcam=False)
        top3 = [
            {"class_name": c, "pretty_name": pretty_class_name(c), "probability": round(p, 4)}
            for c, p, _ in res["top_k"]
        ]
        results.append({
            "background_stimulus": label,
            "top1_predicted_class": res["pred_class"],
            "top1_pretty_name": pretty_class_name(res["pred_class"]),
            "top1_probability": round(res["pred_prob"], 4),
            "top2_predicted_class": top3[1]["class_name"],
            "top2_probability": top3[1]["probability"],
            "top3": top3,
        })
    return results


def generate_procedural_leaf_test_set(
    n_samples: int = 24, seed: int = 42
) -> List[Tuple[str, Image.Image, Optional[int]]]:
    """Generate a diverse set of procedural leaf-on-background specimens spanning multiple leaf
    geometries, venation patterns, lesion morphologies (necrotic spots, chlorotic halos, powdery
    white patches, rust pustules), and PlantVillage-like grey/paper studio backdrops.
    """
    rng = np.random.default_rng(seed)
    specimens: List[Tuple[str, Image.Image, Optional[int]]] = []
    lesion_styles = [
        ("healthy_green", (55, 145, 50), None),
        ("dark_necrotic_scab", (60, 130, 48), (45, 32, 24)),
        ("orange_rust_pustules", (58, 135, 45), (185, 95, 28)),
        ("yellow_chlorotic_blight", (70, 140, 52), (210, 190, 55)),
        ("powdery_white_mildew", (62, 128, 55), (215, 220, 215)),
        ("red_brown_leaf_spot", (50, 132, 46), (135, 52, 38)),
    ]

    for i in range(n_samples):
        style_name, leaf_rgb, spot_rgb = lesion_styles[i % len(lesion_styles)]
        h, w = 256, 256
        # Studio paper background with subtle vignette/grain
        bg_val = int(rng.integers(110, 165))
        bg = np.clip(
            rng.normal(bg_val, 6.0, size=(h, w, 3)), 0, 255
        ).astype(np.uint8)

        yy, xx = np.mgrid[:h, :w]
        cx = 128 + int(rng.integers(-12, 13))
        cy = 128 + int(rng.integers(-12, 13))
        rx = int(rng.integers(58, 82))
        ry = int(rng.integers(75, 98))

        # Leaf contour with natural serration modulation
        angle = np.arctan2(yy - cy, xx - cx)
        radius_mod = 1.0 + 0.06 * np.cos(12.0 * angle) + 0.08 * np.sin(angle)
        dist_norm = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) / radius_mod
        leaf_mask = dist_norm <= 1.0

        # Leaf blade texture + midrib vein
        leaf_canvas = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            leaf_canvas[:, :, c] = leaf_rgb[c] + rng.normal(0.0, 7.0, size=(h, w))
        vein_dist = np.abs(xx - cx)
        midrib = (vein_dist <= 2) & (np.abs(yy - cy) < ry * 0.85)
        leaf_canvas[midrib] = np.array([130, 175, 95], dtype=np.float32)

        if spot_rgb is not None:
            n_spots = int(rng.integers(8, 20))
            for _ in range(n_spots):
                sx = int(rng.integers(cx - int(rx * 0.65), cx + int(rx * 0.65)))
                sy = int(rng.integers(cy - int(ry * 0.65), cy + int(ry * 0.65)))
                sr = int(rng.integers(5, 15))
                d2 = (xx - sx) ** 2 + (yy - sy) ** 2
                halo = (d2 <= (sr + 3) ** 2) & leaf_mask
                core = (d2 <= sr ** 2) & leaf_mask
                leaf_canvas[halo] = np.array([185, 165, 60], dtype=np.float32)
                leaf_canvas[core] = np.array(spot_rgb, dtype=np.float32)

        bg[leaf_mask] = np.clip(leaf_canvas[leaf_mask], 0, 255).astype(np.uint8)
        specimens.append((f"specimen_{i:02d}_{style_name}", Image.fromarray(bg), None))
    return specimens


def run_robustness_experiments(
    adapter: CheckpointModelAdapter,
    samples: Sequence[Tuple[str, Image.Image, Optional[int]]],
    output_dir: Path,
    source_type: str = "SYNTHETIC_PROCEDURAL_SPECIMENS",
) -> Dict[str, object]:
    """Run all perturbation and background ablation experiments across `samples`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = build_perturbation_suite()
    pp = adapter.preprocessing
    img_size = int(pp["img_size"])
    resize_size = int(pp["resize_size"])

    # First compute baseline (clean) predictions, probabilities, foreground masks, and Grad-CAMs
    clean_preds: List[int] = []
    clean_probs: List[np.ndarray] = []
    clean_confs: List[float] = []
    clean_crops_rgb: List[np.ndarray] = []
    fg_masks_224: List[np.ndarray] = []
    clean_bg_cam_ratios: List[float] = []

    for name, pil_im, _lbl in samples:
        crop_pil = eval_resize_and_crop_pil(pil_im, resize_size=resize_size, img_size=img_size, apply_exif=True)
        rgb_224 = np.asarray(crop_pil, dtype=np.uint8)
        clean_crops_rgb.append(rgb_224)
        mask_224 = segment_leaf_foreground_mask(rgb_224)
        fg_masks_224.append(mask_224)

        x = preprocess_pil(crop_pil, {**pp, "resize_size": img_size, "img_size": img_size})[None, ...]
        logits, cam, pred_idx = adapter.forward_logits_and_cam(x)
        probs = adapter.softmax(logits)[0]
        clean_preds.append(pred_idx)
        clean_probs.append(probs)
        clean_confs.append(float(probs[pred_idx]))

        cam_224 = cv2.resize(cam.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        total_cam = float(np.sum(cam_224)) + 1e-8
        bg_cam = float(np.sum(cam_224[~mask_224]))
        clean_bg_cam_ratios.append(bg_cam / total_cam)

    # Evaluate every perturbation condition
    condition_rows: List[Dict[str, object]] = []
    for key, disp_name, fn in suite:
        flips = 0
        conf_deltas = []
        js_divs = []
        top1_confs = []
        bg_cam_ratios = []
        correct_count = 0
        has_labels = all(lbl is not None for _, _, lbl in samples)

        for idx, (name, _pil_im, lbl) in enumerate(samples):
            rgb_orig = clean_crops_rgb[idx]
            rgb_pert = fn(rgb_orig)
            pil_pert = Image.fromarray(rgb_pert)
            x_pert = preprocess_pil(pil_pert, {**pp, "resize_size": img_size, "img_size": img_size})[None, ...]
            logits_p, cam_p, pred_p = adapter.forward_logits_and_cam(x_pert)
            probs_p = adapter.softmax(logits_p)[0]

            if pred_p != clean_preds[idx]:
                flips += 1
            conf_p = float(probs_p[pred_p])
            top1_confs.append(conf_p)
            conf_deltas.append(float(probs_p[clean_preds[idx]] - clean_confs[idx]))
            js_divs.append(jensen_shannon_divergence(clean_probs[idx], probs_p))

            cam_224 = cv2.resize(cam_p.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            bg_ratio = float(np.sum(cam_224[~fg_masks_224[idx]])) / (float(np.sum(cam_224)) + 1e-8)
            bg_cam_ratios.append(bg_ratio)

            if has_labels and pred_p == int(lbl):
                correct_count += 1

        n = len(samples)
        row = {
            "condition_key": key,
            "condition_name": disp_name,
            "n_samples": n,
            "prediction_flip_rate": round(flips / max(n, 1), 4),
            "prediction_stability_rate": round(1.0 - (flips / max(n, 1)), 4),
            "mean_top1_confidence": round(float(np.mean(top1_confs)), 4),
            "mean_orig_class_prob_delta": round(float(np.mean(conf_deltas)), 4),
            "mean_js_divergence": round(float(np.mean(js_divs)), 4),
            "mean_gradcam_background_energy_ratio": round(float(np.mean(bg_cam_ratios)), 4),
            "labelled_top1_accuracy": round(correct_count / max(n, 1), 4) if has_labels else None,
        }
        condition_rows.append(row)

    # Save CSV table
    csv_path = output_dir / "robustness_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(condition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(condition_rows)

    # Probe background-only priors on best_model.pt
    bg_priors = probe_background_only_priors(adapter)
    with open(output_dir / "background_only_priors.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["background_stimulus", "top1_predicted_class", "top1_pretty_name",
                        "top1_probability", "top2_predicted_class", "top2_probability"],
        )
        writer.writeheader()
        for r in bg_priors:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # Render charts and visual galleries
    chart_path = output_dir / "robustness_comparison_chart.png"
    gallery_path = output_dir / "perturbation_and_gradcam_gallery.png"
    _plot_robustness_chart(condition_rows, chart_path, source_type)
    _plot_perturbation_gallery(adapter, clean_crops_rgb[0], suite, gallery_path)

    report = {
        "phase": "Phase 4 — Robustness and Background Bias Evaluation",
        "checkpoint_sha256": adapter.sha256,
        "sample_source_type": source_type,
        "n_samples_evaluated": len(samples),
        "gradcam_interpretation_caveat": (
            "IMPORTANT SCIENTIFIC CAVEAT: Grad-CAM on ResNet18 layer4[-1] produces a coarse 7x7 spatial "
            "attribution map upsampled to 224x224. High Grad-CAM activation over a leaf interior does NOT "
            "prove the classifier learned true biological disease morphology; global feature pooling still "
            "integrates background color/texture statistics across the receptive field. Conversely, substantial "
            "prediction flip rates under background removal/replacement directly demonstrate background reliance."
        ),
        "perturbation_results": condition_rows,
        "background_only_shortcut_probe": bg_priors,
        "artifacts": {
            "metrics_csv": str(csv_path),
            "background_priors_csv": str(output_dir / "background_only_priors.csv"),
            "comparison_chart_png": str(chart_path),
            "visual_gallery_png": str(gallery_path),
        },
    }
    (output_dir / "robustness_report.json").write_text(json.dumps(report, indent=2))
    return report


def _plot_robustness_chart(rows: Sequence[Dict[str, object]], out_path: Path, source_type: str):
    """Plot Prediction Flip Rate and Probability JS Divergence across all perturbations."""
    non_clean = [r for r in rows if r["condition_key"] != "clean"]
    names = [r["condition_name"] for r in non_clean]
    flip_rates = [100.0 * float(r["prediction_flip_rate"]) for r in non_clean]
    js_divs = [float(r["mean_js_divergence"]) for r in non_clean]

    palette = []
    for r in non_clean:
        k = r["condition_key"]
        if k.startswith("bg_"):
            palette.append("#c92a2a")  # Red for background manipulations
        elif k.startswith("noise_") or k.startswith("blur_"):
            palette.append("#e67700")  # Orange for sensor corruptions
        else:
            palette.append("#1971c2")  # Blue for lighting/color shifts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True)
    y = np.arange(len(names))

    ax1.barh(y, flip_rates, color=palette, edgecolor="#222222", height=0.65, alpha=0.88)
    ax1.invert_yaxis()
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9, fontweight="bold")
    ax1.set_xlabel("Prediction Flip Rate (% changed vs. unperturbed)", fontsize=10, fontweight="bold")
    ax1.set_title("Top-1 Prediction Instability by Perturbation", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 105)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for idx, val in enumerate(flip_rates):
        ax1.text(min(val + 1.5, 98), idx, f"{val:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    ax2.barh(y, js_divs, color=palette, edgecolor="#222222", height=0.65, alpha=0.88)
    ax2.set_xlabel("Mean Jensen-Shannon Divergence (bits, [0, 1])", fontsize=10, fontweight="bold")
    ax2.set_title("Output Distribution Shift (JS Divergence)", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, max(js_divs + [0.2]) * 1.2)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for idx, val in enumerate(js_divs):
        ax2.text(val + 0.01, idx, f"{val:.3f}", va="center", fontsize=8.5, fontweight="bold")

    fig.suptitle(
        f"Phase 4 — Perturbation & Background-Replacement Sensitivity ({source_type})",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_perturbation_gallery(
    adapter: CheckpointModelAdapter,
    base_rgb_224: np.ndarray,
    suite: Sequence[Tuple[str, str, Callable[[np.ndarray], np.ndarray]]],
    out_path: Path,
):
    """Render an 8-condition visual gallery showing perturbed image + Grad-CAM overlay + top prediction."""
    selected_keys = [
        "clean", "bright_low", "blur_gauss", "noise_gauss",
        "color_hue", "bg_removed_black", "bg_soil", "bg_bark",
    ]
    selected = [item for item in suite if item[0] in selected_keys]
    pp = {**adapter.preprocessing, "resize_size": 224, "img_size": 224}

    fig, axes = plt.subplots(len(selected), 2, figsize=(9.5, 2.2 * len(selected)))
    for row_idx, (key, disp_name, fn) in enumerate(selected):
        rgb_p = fn(base_rgb_224)
        pil_p = Image.fromarray(rgb_p)
        x_p = preprocess_pil(pil_p, pp)[None, ...]
        logits, cam, pred_idx = adapter.forward_logits_and_cam(x_p)
        probs = adapter.softmax(logits)[0]
        overlay = overlay_heatmap_pil(pil_p, cam)

        ax_img, ax_cam = axes[row_idx, 0], axes[row_idx, 1]
        ax_img.imshow(pil_p)
        ax_img.set_title(f"{disp_name}", fontsize=9, fontweight="bold")
        ax_img.axis("off")

        ax_cam.imshow(overlay)
        ax_cam.set_title(
            f"Pred: {pretty_class_name(adapter.class_names[pred_idx])} ({probs[pred_idx]:.1%})",
            fontsize=8.5,
        )
        ax_cam.axis("off")

    fig.suptitle(
        "Phase 4 — Visual Perturbation & Background-Ablation Gallery (with 7x7 Grad-CAM Overlay)",
        fontsize=11.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_labelled_samples_from_dir(
    data_dir: Path, class_names: Sequence[str], max_per_class: int = 3
) -> List[Tuple[str, Image.Image, Optional[int]]]:
    """Load up to `max_per_class` images per class from a labelled ImageFolder directory."""
    samples: List[Tuple[str, Image.Image, Optional[int]]] = []
    for idx, cname in enumerate(class_names):
        cdir = data_dir / cname
        if not cdir.is_dir():
            continue
        files = sorted(
            p for p in cdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )[:max_per_class]
        for p in files:
            with Image.open(p) as im:
                samples.append((f"{cname}/{p.name}", im.convert("RGB").copy(), idx))
    return samples


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--data-dir", default=None, help="Optional labelled ImageFolder split (e.g. data/valid)")
    parser.add_argument("--max-per-class", type=int, default=3)
    parser.add_argument("--n-synthetic", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "onnx"])
    parser.add_argument("--output-dir", default="ml_validation/reports/phase4_robustness_and_bg_bias")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    adapter = CheckpointModelAdapter(args.checkpoint, device=args.device, backend=args.backend)

    if args.data_dir and Path(args.data_dir).is_dir():
        samples = load_labelled_samples_from_dir(Path(args.data_dir), adapter.class_names, args.max_per_class)
        source_type = f"NATURAL_DATASET_SPLIT ({args.data_dir})"
    else:
        samples = generate_procedural_leaf_test_set(n_samples=args.n_synthetic, seed=42)
        source_type = "SYNTHETIC_PROCEDURAL_LEAF_SPECIMENS (Natural dataset not provided locally)"

    report = run_robustness_experiments(adapter, samples, out_dir, source_type=source_type)
    bg_soil_flip = next(
        r["prediction_flip_rate"] for r in report["perturbation_results"] if r["condition_key"] == "bg_soil"
    )
    print(
        f"[Phase 4] Completed robustness & background bias suite on {report['n_samples_evaluated']} samples "
        f"({source_type}). Soil-background replacement flip rate: {bg_soil_flip:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
