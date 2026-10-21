"""Small shared helpers used across train/evaluate/demo scripts."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed random, numpy, and torch.

    This makes runs repeatable in practice but not bit-exact: cuDNN/MPS kernels and
    DataLoader worker scheduling can still introduce small nondeterminism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Pick CUDA, then Apple Silicon MPS, then CPU. `preferred` forces a specific device."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_class_names(class_names, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)


def load_class_names(path: str):
    with open(path) as f:
        return json.load(f)


def pretty_class_name(name: str) -> str:
    """'Corn_(maize)___Common_rust_' -> 'Corn (maize) — Common rust'."""
    crop, _, condition = name.partition("___")

    def tidy(text: str) -> str:
        text = " ".join(text.replace("_", " ").split())
        return text[:1].upper() + text[1:]

    return f"{tidy(crop)} — {tidy(condition)}" if condition else tidy(crop)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AverageMeter:
    """Tracks a running average of a scalar (loss, accuracy, ...)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class EarlyStopping:
    """Stops training when validation metric hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Returns True if this is the best metric seen so far."""
        is_best = self.best is None or (
            metric > self.best if self.mode == "max" else metric < self.best
        )
        if is_best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return is_best

    def reset_patience(self) -> None:
        """Start a fresh patience budget (e.g. for a new training phase) but keep `best`,
        so checkpoint selection still compares against the best epoch of any phase."""
        self.counter = 0
        self.should_stop = False
