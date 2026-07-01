"""Small shared helpers used across train/evaluate/demo scripts."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_class_names(class_names, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)


def load_class_names(path: str):
    with open(path) as f:
        return json.load(f)


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
