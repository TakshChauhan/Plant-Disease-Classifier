"""Exact evaluation preprocessing and crop-box geometry matching src/dataset.py.

Mirrors the committed checkpoint's evaluation transform pipeline:
1. EXIF transpose (when loading raw user/field photos) and RGB conversion
2. Resize shorter side to `resize_size = int(img_size * 1.14)` (255 for 224px input) with bilinear interpolation
3. CenterCrop to `(img_size, img_size)` (224x224)
4. Scale uint8 [0, 255] -> float32 [0.0, 1.0]
5. Channel-wise ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE_RATIO = 1.14
DEFAULT_IMG_SIZE = 224


def default_preprocessing(img_size: int = DEFAULT_IMG_SIZE) -> Dict[str, object]:
    """Return the default evaluation preprocessing dictionary."""
    return {
        "img_size": int(img_size),
        "resize_size": int(img_size * RESIZE_RATIO),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }


def preprocessing_from_checkpoint(ckpt: dict) -> Dict[str, object]:
    """Extract preprocessing config from checkpoint or fall back to legacy defaults."""
    return ckpt.get("preprocessing") or default_preprocessing()


def eval_crop_box(
    width: int, height: int, resize_size: int = 255, img_size: int = DEFAULT_IMG_SIZE
) -> Tuple[float, float, float, float]:
    """Compute the (left, top, right, bottom) crop box in original image coordinates
    that corresponds to `Resize(resize_size)` followed by `CenterCrop(img_size)`.
    """
    if width <= height:
        new_w, new_h = resize_size, int(resize_size * height / width)
    else:
        new_w, new_h = int(resize_size * width / height), resize_size
    top = int(round((new_h - img_size) / 2.0))
    left = int(round((new_w - img_size) / 2.0))
    sx, sy = width / new_w, height / new_h
    return (left * sx, top * sy, (left + img_size) * sx, (top + img_size) * sy)


def eval_resize_and_crop_pil(
    image: Image.Image, resize_size: int = 255, img_size: int = DEFAULT_IMG_SIZE, apply_exif: bool = False
) -> Image.Image:
    """Return the 224x224 RGB PIL image after Resize(resize_size) + CenterCrop(img_size)."""
    if apply_exif:
        image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    w, h = image.size
    if w <= h:
        new_w, new_h = resize_size, int(resize_size * h / w)
    else:
        new_w, new_h = int(resize_size * w / h), resize_size
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    top = int(round((new_h - img_size) / 2.0))
    left = int(round((new_w - img_size) / 2.0))
    return resized.crop((left, top, left + img_size, top + img_size))


def preprocess_pil(
    image: Image.Image, preprocessing: Dict[str, object] | None = None, apply_exif: bool = False
) -> np.ndarray:
    """Preprocess a single PIL image into a float32 tensor array of shape (3, H, W)."""
    pp = preprocessing or default_preprocessing()
    img_size = int(pp["img_size"])
    resize_size = int(pp["resize_size"])
    mean = np.asarray(pp["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(pp["std"], dtype=np.float32).reshape(3, 1, 1)

    cropped = eval_resize_and_crop_pil(image, resize_size=resize_size, img_size=img_size, apply_exif=apply_exif)
    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return (chw - mean) / std
