"""ImageFolder-based dataloaders and the shared preprocessing for the plant disease dataset."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization stats — required since we use ImageNet-pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Eval resizes the shorter side to int(img_size * RESIZE_RATIO) and center-crops img_size.
# The committed checkpoint was trained/evaluated with exactly this (224 -> 255 -> 224).
RESIZE_RATIO = 1.14
DEFAULT_IMG_SIZE = 224


def default_preprocessing(img_size: int = DEFAULT_IMG_SIZE) -> dict:
    return {
        "img_size": img_size,
        "resize_size": int(img_size * RESIZE_RATIO),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }


def preprocessing_from_checkpoint(ckpt: dict) -> dict:
    """Checkpoints written before preprocessing metadata existed fall back to the
    defaults above, which are what those checkpoints were trained with."""
    return ckpt.get("preprocessing") or default_preprocessing()


def build_transforms(img_size: int = DEFAULT_IMG_SIZE, resize_size: int = None,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    resize_size = resize_size or int(img_size * RESIZE_RATIO)
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfms, eval_tfms


def eval_crop_box(width: int, height: int, resize_size: int, img_size: int):
    """The (left, top, right, bottom) box of the *original* image that the eval transform
    (Resize(resize_size) on the shorter side, then CenterCrop(img_size)) actually keeps.

    Used to draw Grad-CAM on the same region the model saw, instead of stretching a
    heatmap of the center crop over the full uploaded photo.
    """
    # Mirror torchvision's Resize(int) output size for a PIL image
    if width <= height:
        new_w, new_h = resize_size, int(resize_size * height / width)
    else:
        new_w, new_h = int(resize_size * width / height), resize_size
    # Mirror torchvision's CenterCrop offsets
    top = int(round((new_h - img_size) / 2.0))
    left = int(round((new_w - img_size) / 2.0))
    sx, sy = width / new_w, height / new_h
    return (left * sx, top * sy, (left + img_size) * sx, (top + img_size) * sy)


def _image_folder(path: Path, transform):
    if not path.is_dir():
        raise FileNotFoundError(
            f"Expected an ImageFolder directory at {path}. The Kaggle zip is nested — point "
            "--data-dir at the folder that directly contains train/ and valid/ (see data/README.md)."
        )
    return datasets.ImageFolder(str(path), transform=transform)


def _loader_kwargs(num_workers: int):
    # pin_memory only helps (and is only supported without warnings) for CUDA transfers
    return {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}


def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = DEFAULT_IMG_SIZE,
                    num_workers: int = 2, seed: int = 42):
    """Expects data_dir/train/<class>/*.jpg and data_dir/valid/<class>/*.jpg."""
    train_tfms, eval_tfms = build_transforms(img_size)

    train_ds = _image_folder(Path(data_dir) / "train", train_tfms)
    val_ds = _image_folder(Path(data_dir) / "valid", eval_tfms)

    # Guard against the two splits somehow disagreeing on class order
    assert train_ds.classes == val_ds.classes, "train/valid class folders don't match"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), **_loader_kwargs(num_workers),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(num_workers))
    return train_loader, val_loader, train_ds.classes


def get_eval_loader(data_dir: str, split: str = "valid", batch_size: int = 32,
                    preprocessing: dict = None, num_workers: int = 2):
    """Loader for a single labelled split, using the checkpoint's eval preprocessing."""
    pp = preprocessing or default_preprocessing()
    _, eval_tfms = build_transforms(pp["img_size"], pp["resize_size"], pp["mean"], pp["std"])
    ds = _image_folder(Path(data_dir) / split, eval_tfms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(num_workers))
    return loader, ds.classes
