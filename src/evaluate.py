"""Evaluate a trained checkpoint: accuracy, per-class report, confusion matrix.

Usage:
    python src/evaluate.py --data-dir data --checkpoint results/best_model.pt
"""
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model
from utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]

    model, _ = get_model(ckpt["model_name"], num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    _, val_loader, loader_classes = get_dataloaders(args.data_dir, args.batch_size)
    assert loader_classes == class_names, "Checkpoint class order doesn't match this dataset's class order"

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\nClassification report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0))

    with open(f"{args.output_dir}/classification_report.txt", "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True, square=True,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/confusion_matrix.png", dpi=150)
    print(f"\nConfusion matrix saved to {args.output_dir}/confusion_matrix.png")
    print(f"Classification report saved to {args.output_dir}/classification_report.txt")


if __name__ == "__main__":
    main()
