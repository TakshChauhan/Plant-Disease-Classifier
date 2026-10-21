import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from conftest import random_image
from dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    default_preprocessing,
    eval_crop_box,
    get_dataloaders,
    get_eval_loader,
    preprocessing_from_checkpoint,
)


def test_default_preprocessing_matches_original_checkpoint_recipe():
    # The committed checkpoint was produced with Resize(int(224 * 1.14)) = 255 then CenterCrop(224)
    assert default_preprocessing() == {"img_size": 224, "resize_size": 255,
                                       "mean": IMAGENET_MEAN, "std": IMAGENET_STD}


def test_legacy_checkpoint_falls_back_to_defaults():
    assert preprocessing_from_checkpoint({"model_state": {}}) == default_preprocessing()
    custom = {**default_preprocessing(), "img_size": 192}
    assert preprocessing_from_checkpoint({"preprocessing": custom}) == custom


@pytest.mark.parametrize("size", [(256, 256), (640, 480), (300, 900), (37, 1000)])
def test_eval_transform_output_shape_and_normalisation(rng, size):
    _, eval_tfms = build_transforms()
    x = eval_tfms(random_image(rng, *size))
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    # A uniform image at the ImageNet mean normalises to ~0
    mean_img = Image.new("RGB", size, tuple(int(round(m * 255)) for m in IMAGENET_MEAN))
    assert eval_tfms(mean_img).abs().max() < 0.02


def test_train_transform_output_shape(rng):
    train_tfms, _ = build_transforms(img_size=64)
    assert train_tfms(random_image(rng, 100, 80)).shape == (3, 64, 64)


@pytest.mark.parametrize("size", [(256, 256), (640, 480), (480, 640), (4032, 3024), (1000, 333)])
def test_eval_crop_box_matches_what_the_model_sees(rng, size):
    """Cropping the original with eval_crop_box and resizing must reproduce the eval input."""
    img = random_image(rng, *size).resize((size[0] // 4 or 1, size[1] // 4 or 1)).resize(size)  # smooth-ish
    resize_size, img_size = 255, 224
    model_view = TF.center_crop(TF.resize(img, resize_size), img_size)
    left, top, right, bottom = eval_crop_box(*size, resize_size, img_size)
    via_box = img.resize((img_size, img_size), Image.BILINEAR, box=(left, top, right, bottom))
    diff = np.abs(np.asarray(model_view, dtype=np.float32) - np.asarray(via_box, dtype=np.float32))
    assert diff.mean() < 6.0  # only resampling differences remain


def test_eval_crop_box_marker_lands_in_same_place():
    """A bright square at a known spot must appear at the same relative spot in both views."""
    w, h = 1200, 800
    img = Image.new("RGB", (w, h))
    img.paste((255, 255, 255), (700, 300, 760, 360))
    model_view = np.asarray(TF.center_crop(TF.resize(img, 255), 224)).sum(-1)
    left, top, right, bottom = eval_crop_box(w, h, 255, 224)
    ys, xs = np.nonzero(model_view > 300)
    # marker centre in model-view pixels -> original pixels via the box
    cx = left + (xs.mean() + 0.5) / 224 * (right - left)
    cy = top + (ys.mean() + 0.5) / 224 * (bottom - top)
    assert abs(cx - 730) < 6 and abs(cy - 330) < 6


def test_eval_crop_box_is_central_square_in_original_pixels():
    left, top, right, bottom = eval_crop_box(640, 480, 255, 224)
    assert right - left == pytest.approx(bottom - top, rel=0.01)
    assert (left + right) / 2 == pytest.approx(320, abs=2) and (top + bottom) / 2 == pytest.approx(240, abs=2)
    assert bottom - top == pytest.approx(480 * 224 / 255, abs=2)  # ~88% of the shorter side


def test_get_dataloaders_on_synthetic_folder(image_folder):
    data_dir, classes = image_folder
    train_loader, val_loader, names = get_dataloaders(str(data_dir), batch_size=4, img_size=32, num_workers=0)
    assert names == classes
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 32, 32)
    assert len(val_loader.dataset) == 6


def test_get_dataloaders_detects_class_mismatch(image_folder):
    data_dir, _ = image_folder
    (data_dir / "valid" / "Extra___class").mkdir()
    random_image(np.random.default_rng(1), 40, 40).save(data_dir / "valid" / "Extra___class" / "x.png")
    with pytest.raises(AssertionError, match="don't match"):
        get_dataloaders(str(data_dir), batch_size=2, img_size=32, num_workers=0)


def test_missing_split_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--data-dir"):
        get_eval_loader(str(tmp_path), "valid", num_workers=0)


def test_get_eval_loader_uses_given_preprocessing(image_folder):
    data_dir, classes = image_folder
    pp = {**default_preprocessing(32), "resize_size": 36}
    loader, names = get_eval_loader(str(data_dir), "valid", batch_size=8, preprocessing=pp, num_workers=0)
    images, _ = next(iter(loader))
    assert names == classes and images.shape[-2:] == (32, 32)
