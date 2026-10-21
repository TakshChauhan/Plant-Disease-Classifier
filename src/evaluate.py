"""Evaluate a trained checkpoint: accuracy, per-class report, confusion matrix.

Usage:
    python src/evaluate.py --data-dir data --checkpoint results/best_model.pt

Note: `valid/` is also the split used for checkpoint selection during training, so
numbers computed on it are optimistic. Evaluate on a held-out split when one exists.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever save figures to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm  # noqa: E402

from dataset import get_eval_loader, preprocessing_from_checkpoint  # noqa: E402
from model import build_model_from_checkpoint, load_checkpoint  # noqa: E402
from utils import file_sha256, get_device  # noqa: E402


def predict_loader(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def compute_metrics(labels, preds, class_names) -> dict:
    label_ids = list(range(len(class_names)))
    metrics = {"n_samples": len(labels), "accuracy": accuracy_score(labels, preds)}
    for avg in ("macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=label_ids,
                                                      average=avg, zero_division=0)
        metrics[f"{avg}_precision"], metrics[f"{avg}_recall"], metrics[f"{avg}_f1"] = p, r, f1
    return {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, labels=class_names, rotation=90, fontsize=6)
    ax.set_yticks(ticks, labels=class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="valid", help="labelled ImageFolder split under --data-dir")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output-dir", default="results/evaluation")
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    class_names = ckpt["class_names"]
    model = build_model_from_checkpoint(ckpt, device)

    loader, loader_classes = get_eval_loader(
        args.data_dir, args.split, args.batch_size, preprocessing_from_checkpoint(ckpt), args.num_workers
    )
    if loader_classes != class_names:
        raise SystemExit("Checkpoint class order doesn't match this dataset's class folders.")

    labels, preds = predict_loader(model, loader, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(labels, preds, labels=list(range(len(class_names))),
                                   target_names=class_names, digits=4, zero_division=0)
    print("\nClassification report:\n")
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    metrics = compute_metrics(labels, preds, class_names)
    metrics.update({
        "split": args.split,
        "data_dir": str(args.data_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "note": "If split == 'valid', this split was also used to select the checkpoint.",
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    print(f"Accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['macro_f1']:.4f} "
          f"| n={metrics['n_samples']} ({args.split})")
    print(f"Outputs saved to {out_dir}/ (classification_report.txt, metrics.json, confusion_matrix.png)")


if __name__ == "__main__":
    main()
