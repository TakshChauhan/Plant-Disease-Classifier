"""Comprehensive classification and calibration metrics for ML validation."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax probabilities from logits `(N, C)` with optional temperature scaling."""
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-8)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(np.float64)


def top_k_accuracy(labels: Sequence[int], scores: np.ndarray, k: int = 1) -> float:
    """Compute Top-K accuracy given ground-truth integer labels `(N,)` and class scores `(N, C)`."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.size == 0:
        return 0.0
    scores_arr = np.asarray(scores)
    k_eff = min(max(1, int(k)), scores_arr.shape[1])
    top_k_indices = np.argsort(-scores_arr, axis=1, kind="mergesort")[:, :k_eff]
    matches = np.any(top_k_indices == labels_arr[:, None], axis=1)
    return float(np.mean(matches))


def compute_confusion_matrix(labels: Sequence[int], preds: Sequence[int], num_classes: int) -> np.ndarray:
    """Compute a `(num_classes, num_classes)` integer confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        if 0 <= int(y_true) < num_classes and 0 <= int(y_pred) < num_classes:
            cm[int(y_true), int(y_pred)] += 1
    return cm


def compute_classification_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    class_names: Sequence[str],
    probs: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute complete overall and per-class metrics matching scikit-learn semantics."""
    num_classes = len(class_names)
    labels_arr = np.asarray(labels, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    n_samples = int(labels_arr.size)

    cm = compute_confusion_matrix(labels_arr, preds_arr, num_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.int64)
    pred_count = cm.sum(axis=0).astype(np.int64)

    precisions = np.where(pred_count > 0, tp / np.maximum(pred_count, 1), 0.0)
    recalls = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    denom = precisions + recalls
    f1s = np.where(denom > 0, 2.0 * precisions * recalls / np.maximum(denom, 1e-12), 0.0)

    top1_acc = float(np.mean(labels_arr == preds_arr)) if n_samples > 0 else 0.0
    top3_acc = top_k_accuracy(labels_arr, probs, k=3) if (probs is not None and n_samples > 0) else None

    macro_p = float(np.mean(precisions)) if num_classes > 0 else 0.0
    macro_r = float(np.mean(recalls)) if num_classes > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if num_classes > 0 else 0.0

    present_mask = support > 0
    active_macro_p = float(np.mean(precisions[present_mask])) if np.any(present_mask) else 0.0
    active_macro_r = float(np.mean(recalls[present_mask])) if np.any(present_mask) else 0.0
    active_macro_f1 = float(np.mean(f1s[present_mask])) if np.any(present_mask) else 0.0

    if n_samples > 0:
        weights = support.astype(np.float64) / n_samples
        weighted_p = float(np.sum(precisions * weights))
        weighted_r = float(np.sum(recalls * weights))
        weighted_f1 = float(np.sum(f1s * weights))
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0

    per_class = []
    for idx, cname in enumerate(class_names):
        per_class.append({
            "class_index": int(idx),
            "class_name": str(cname),
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx]),
            "support": int(support[idx]),
            "predicted_count": int(pred_count[idx]),
            "correct_count": int(tp[idx]),
        })

    return {
        "n_samples": n_samples,
        "num_classes": num_classes,
        "num_active_classes": int(np.sum(present_mask)),
        "accuracy": top1_acc,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "active_classes_macro_precision": active_macro_p,
        "active_classes_macro_recall": active_macro_r,
        "active_classes_macro_f1": active_macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_classification_report(metrics: Dict[str, object], digits: int = 4) -> str:
    """Format a human-readable classification report matching sklearn's layout."""
    per_class = metrics["per_class"]
    max_name = max([len(c["class_name"]) for c in per_class] + [len("weighted avg"), 12])
    header = f"{'':>{max_name}}  {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    lines = [header, ""]
    for c in per_class:
        lines.append(
            f"{c['class_name']:>{max_name}}  "
            f"{c['precision']:10.{digits}f} {c['recall']:10.{digits}f} "
            f"{c['f1']:10.{digits}f} {c['support']:10d}"
        )
    lines.append("")
    n = metrics["n_samples"]
    lines.append(f"{'accuracy':>{max_name}}  {'':>10} {'':>10} {metrics['accuracy']:10.{digits}f} {n:10d}")
    lines.append(
        f"{'macro avg':>{max_name}}  "
        f"{metrics['macro_precision']:10.{digits}f} {metrics['macro_recall']:10.{digits}f} "
        f"{metrics['macro_f1']:10.{digits}f} {n:10d}"
    )
    lines.append(
        f"{'weighted avg':>{max_name}}  "
        f"{metrics['weighted_precision']:10.{digits}f} {metrics['weighted_recall']:10.{digits}f} "
        f"{metrics['weighted_f1']:10.{digits}f} {n:10d}"
    )
    return "\n".join(lines) + "\n"


def compute_calibration_stats(
    labels: Sequence[int], probs: np.ndarray, n_bins: int = 15
) -> Dict[str, object]:
    """Compute ECE, MCE, Brier score, NLL, and bin-level reliability statistics."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    n_samples, num_classes = probs_arr.shape
    if n_samples == 0:
        raise ValueError("Cannot compute calibration statistics on empty sample array.")

    confidences = np.max(probs_arr, axis=1)
    predictions = np.argmax(probs_arr, axis=1)
    accuracies = (predictions == labels_arr).astype(np.float64)

    y_onehot = np.zeros_like(probs_arr)
    y_onehot[np.arange(n_samples), labels_arr] = 1.0
    brier = float(np.mean(np.sum((probs_arr - y_onehot) ** 2, axis=1)))

    true_class_probs = np.clip(probs_arr[np.arange(n_samples), labels_arr], 1e-12, 1.0)
    nll = float(-np.mean(np.log(true_class_probs)))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins_data = []

    for b_idx in range(n_bins):
        low, high = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
        if b_idx == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(confidences[in_bin]))
            avg_acc = float(np.mean(accuracies[in_bin]))
            gap = abs(avg_acc - avg_conf)
            prop = count / n_samples
            ece += prop * gap
            mce = max(mce, gap)
        else:
            avg_conf = float(0.5 * (low + high))
            avg_acc = 0.0
            gap = 0.0
            prop = 0.0

        bins_data.append({
            "bin_index": b_idx,
            "bin_lower": float(low),
            "bin_upper": float(high),
            "count": count,
            "proportion": float(prop),
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc,
            "calibration_gap": float(gap),
        })

    correct_mask = accuracies == 1.0
    incorrect_mask = ~correct_mask

    return {
        "n_samples": int(n_samples),
        "num_classes": int(num_classes),
        "n_bins": int(n_bins),
        "accuracy": float(np.mean(accuracies)),
        "mean_confidence": float(np.mean(confidences)),
        "overconfidence_signed": float(np.mean(confidences) - np.mean(accuracies)),
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier,
        "nll": nll,
        "correct_confidence_mean": float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else None,
        "correct_confidence_median": float(np.median(confidences[correct_mask])) if np.any(correct_mask) else None,
        "incorrect_confidence_mean": float(np.mean(confidences[incorrect_mask])) if np.any(incorrect_mask) else None,
        "incorrect_confidence_median": (
            float(np.median(confidences[incorrect_mask])) if np.any(incorrect_mask) else None
        ),
        "bins": bins_data,
    }


def fit_temperature_scaling(
    logits_fit: np.ndarray, labels_fit: Sequence[int], bounds: Tuple[float, float] = (0.05, 10.0)
) -> Dict[str, float]:
    """Fit a single post-hoc temperature parameter T > 0 by minimizing NLL on a calibration split."""
    logits_arr = np.asarray(logits_fit, dtype=np.float64)
    labels_arr = np.asarray(labels_fit, dtype=np.int64)
    n = labels_arr.size
    if n == 0:
        raise ValueError("Cannot fit temperature scaling on an empty split.")

    def nll_objective(temp: float) -> float:
        probs = softmax_np(logits_arr, temperature=temp)
        p_true = np.clip(probs[np.arange(n), labels_arr], 1e-12, 1.0)
        return float(-np.mean(np.log(p_true)))

    nll_before = nll_objective(1.0)
    res = minimize_scalar(nll_objective, bounds=bounds, method="bounded")
    best_t = float(res.x)
    nll_after = nll_objective(best_t)
    return {
        "optimal_temperature": best_t,
        "fit_nll_before": nll_before,
        "fit_nll_after": nll_after,
        "converged": bool(res.success),
    }
