"""Two-phase transfer-learning training loop.

Phase 1: backbone frozen, only the new head trains (fast, stabilizes new weights).
Phase 2: whole network unfrozen, fine-tuned end-to-end at a lower learning rate.

Usage:
    python src/train.py --data-dir data --model resnet18 \
        --freeze-epochs 3 --finetune-epochs 7 --batch-size 32
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model, set_backbone_trainable
from utils import AverageMeter, EarlyStopping, get_device, save_class_names, set_seed


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


def plot_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "acc"]):
        ax.plot(history[f"train_{metric}"], label="train")
        ax.plot(history[f"val_{metric}"], label="val")
        ax.set_title(metric.upper())
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default="resnet18",
                         choices=["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--freeze-epochs", type=int, default=3, help="phase 1: head-only training")
    parser.add_argument("--finetune-epochs", type=int, default=7, help="phase 2: full fine-tuning")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, args.batch_size, args.img_size
    )
    print(f"Classes: {len(class_names)} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model, backbone_params = get_model(args.model, num_classes=len(class_names))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_class_names(class_names, f"{args.output_dir}/class_names.json")

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    early_stop = EarlyStopping(patience=args.patience, mode="max")
    best_path = f"{args.output_dir}/best_model.pt"

    def train_phase(num_epochs, lr, phase_name):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            is_best = early_stop.step(val_acc)
            if is_best:
                torch.save(
                    {"model_state": model.state_dict(), "model_name": args.model, "class_names": class_names},
                    best_path,
                )

            print(
                f"[{phase_name}] epoch {epoch + 1}/{num_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"({time.time() - t0:.1f}s){' *best*' if is_best else ''}"
            )

            if early_stop.should_stop:
                print(f"Early stopping ({phase_name}): no improvement for {args.patience} epochs.")
                return True
        return False

    # Phase 1: freeze backbone, train head only
    set_backbone_trainable(backbone_params, trainable=False)
    stopped = train_phase(args.freeze_epochs, args.head_lr, "phase1-head")

    # Phase 2: unfreeze everything, fine-tune at a lower LR
    if not stopped:
        set_backbone_trainable(backbone_params, trainable=True)
        early_stop.counter = 0  # give fine-tuning phase a fresh patience budget
        train_phase(args.finetune_epochs, args.finetune_lr, "phase2-finetune")

    plot_curves(history, f"{args.output_dir}/training_curves.png")
    print(f"\nBest val accuracy: {early_stop.best:.4f}")
    print(f"Best checkpoint saved to: {best_path}")
    print(f"Training curves saved to: {args.output_dir}/training_curves.png")


if __name__ == "__main__":
    main()
