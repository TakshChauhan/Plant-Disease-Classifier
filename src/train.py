"""Two-phase transfer-learning training loop.

Phase 1: backbone frozen, only the new head trains (fast, stabilizes new weights).
Phase 2: whole network unfrozen, fine-tuned end-to-end at a lower learning rate.

Early stopping has its own patience budget per phase, so a plateau during head
training ends phase 1 early but never skips fine-tuning. The best checkpoint is
selected on validation accuracy across both phases.

Usage:
    python src/train.py --data-dir data --model resnet18 \
        --freeze-epochs 3 --finetune-epochs 7 --batch-size 32 --output-dir results/my_run
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever save figures to disk
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataset import default_preprocessing, get_dataloaders  # noqa: E402
from model import SUPPORTED_MODELS, get_model, set_backbone_trainable, set_batchnorm_eval  # noqa: E402
from utils import AverageMeter, EarlyStopping, get_device, save_class_names, set_seed  # noqa: E402

CHECKPOINT_FORMAT_VERSION = 2


def run_epoch(model, loader, criterion, optimizer, device, train: bool, freeze_bn_stats: bool = False):
    model.train() if train else model.eval()
    if train and freeze_bn_stats:
        set_batchnorm_eval(model)
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


def train_two_phase(model, backbone_params, train_loader, val_loader, device, *,
                    freeze_epochs: int, finetune_epochs: int, head_lr: float, finetune_lr: float,
                    patience: int, freeze_bn_stats: bool = False, on_best=None, log=print):
    """Runs phase 1 (head only) then phase 2 (full fine-tune). Returns (history, early_stop).

    `on_best(epoch_info)` is called whenever validation accuracy improves (across both phases).
    """
    criterion = nn.CrossEntropyLoss()
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "phase"]}
    early_stop = EarlyStopping(patience=patience, mode="max")

    def train_phase(num_epochs, lr, phase_name, bn_frozen):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                              train=True, freeze_bn_stats=bn_frozen)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["phase"].append(phase_name)

            is_best = early_stop.step(val_acc)
            if is_best and on_best is not None:
                on_best({"epoch": len(history["val_acc"]), "phase": phase_name,
                         "val_acc": val_acc, "val_loss": val_loss})

            log(
                f"[{phase_name}] epoch {epoch + 1}/{num_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"({time.time() - t0:.1f}s){' *best*' if is_best else ''}"
            )

            if early_stop.should_stop:
                log(f"Early stopping ({phase_name}): no improvement for {patience} epochs.")
                return

    # Phase 1: freeze backbone, train head only
    set_backbone_trainable(backbone_params, trainable=False)
    train_phase(freeze_epochs, head_lr, "phase1-head", bn_frozen=freeze_bn_stats)

    # Phase 2: unfreeze everything, fine-tune at a lower LR with a fresh patience budget
    set_backbone_trainable(backbone_params, trainable=True)
    early_stop.reset_patience()
    train_phase(finetune_epochs, finetune_lr, "phase2-finetune", bn_frozen=False)

    return history, early_stop


def plot_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    n_phase1 = sum(p == "phase1-head" for p in history.get("phase", []))
    for ax, metric in zip(axes, ["loss", "acc"]):
        ax.plot(epochs, history[f"train_{metric}"], label="train (augmented)")
        ax.plot(epochs, history[f"val_{metric}"], label="val")
        if 0 < n_phase1 < len(history["train_loss"]):
            ax.axvline(n_phase1 + 0.5, color="gray", linestyle="--", linewidth=1, label="start fine-tuning")
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
    parser.add_argument("--model", default="resnet18", choices=list(SUPPORTED_MODELS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--freeze-epochs", type=int, default=3, help="phase 1: head-only training")
    parser.add_argument("--finetune-epochs", type=int, default=7, help="phase 2: full fine-tuning")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4, help="early-stopping patience, per phase")
    parser.add_argument("--freeze-bn-stats", action="store_true",
                        help="keep BatchNorm running stats fixed during phase 1. Off by default "
                             "to match the recipe that produced the committed checkpoint.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output-dir", default="results/run")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing best_model.pt in --output-dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    best_path = out_dir / "best_model.pt"
    if best_path.exists() and not args.overwrite:
        sys.exit(f"Refusing to overwrite existing checkpoint {best_path}. "
                 "Use a different --output-dir or pass --overwrite.")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, args.batch_size, args.img_size, num_workers=args.num_workers, seed=args.seed
    )
    print(f"Classes: {len(class_names)} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model, backbone_params = get_model(args.model, num_classes=len(class_names))
    model.to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_class_names(class_names, out_dir / "class_names.json")

    def save_best(info):
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": args.model,
                "class_names": class_names,
                # Everything below is metadata added in format v2; loaders treat it as optional.
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "preprocessing": default_preprocessing(args.img_size),
                "best_epoch": info,
                "train_args": vars(args),
                # str(): torch.__version__ is a TorchVersion, which weights_only loading rejects
                "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                             "python": platform.python_version()},
            },
            best_path,
        )

    history, early_stop = train_two_phase(
        model, backbone_params, train_loader, val_loader, device,
        freeze_epochs=args.freeze_epochs, finetune_epochs=args.finetune_epochs,
        head_lr=args.head_lr, finetune_lr=args.finetune_lr, patience=args.patience,
        freeze_bn_stats=args.freeze_bn_stats, on_best=save_best,
    )

    plot_curves(history, out_dir / "training_curves.png")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val accuracy: {early_stop.best:.4f} (selected on the validation split, so "
          "it is an optimistic estimate; see README > Limitations)")
    print(f"Best checkpoint saved to: {best_path}")
    print(f"Training curves saved to: {out_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
