"""Compatibility checks against the committed ResNet18 checkpoint.

These verify that the artifact loads safely and produces well-formed outputs. They say
nothing about accuracy: no labelled leaf images are used here.
"""
import json

import numpy as np
import pytest
import torch

from conftest import CHECKPOINT, CLASS_NAMES_JSON, random_image
from dataset import build_transforms, preprocessing_from_checkpoint
from model import build_model_from_checkpoint, load_checkpoint
from utils import file_sha256

pytestmark = [
    pytest.mark.checkpoint,
    pytest.mark.skipif(not CHECKPOINT.is_file(), reason="results/best_model.pt not present"),
]

# SHA-256 of results/best_model.pt as committed in 761bf06. Guards against accidental overwrites.
COMMITTED_SHA256 = "f456fe4cd6042b2c031172bcdab00a40c845475a3f71fa052264d28311ed0cc9"


@pytest.fixture(scope="module")
def ckpt():
    return load_checkpoint(CHECKPOINT)


@pytest.fixture(scope="module")
def model(ckpt):
    return build_model_from_checkpoint(ckpt)


def test_checkpoint_file_unchanged():
    assert file_sha256(CHECKPOINT) == COMMITTED_SHA256


def test_checkpoint_metadata(ckpt):
    assert ckpt["model_name"] == "resnet18"
    assert len(ckpt["class_names"]) == 38
    assert ckpt["class_names"] == sorted(ckpt["class_names"])  # ImageFolder order
    assert ckpt["model_state"]["fc.weight"].shape == (38, 512)


def test_class_names_json_matches_checkpoint(ckpt):
    assert json.loads(CLASS_NAMES_JSON.read_text()) == ckpt["class_names"]


def test_legacy_checkpoint_uses_original_preprocessing(ckpt):
    assert "preprocessing" not in ckpt
    assert preprocessing_from_checkpoint(ckpt)["resize_size"] == 255


def test_inference_outputs_are_well_formed(model, ckpt, rng):
    _, tfm = build_transforms()
    batch = torch.stack([tfm(random_image(rng, w, h)) for w, h in [(256, 256), (640, 480), (300, 900)]])
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)
    assert probs.shape == (3, 38)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.isfinite(probs).all()


def test_inference_is_deterministic(model, rng):
    _, tfm = build_transforms()
    x = tfm(random_image(rng, 300, 300)).unsqueeze(0)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_bn_stats_were_updated_every_epoch(ckpt):
    """10 epochs x ceil(70,295 / 32) = 21,970 BN updates: BN stats moved during the frozen phase too,
    and early stopping never fired. Documents how the shipped checkpoint was trained."""
    assert int(ckpt["model_state"]["bn1.num_batches_tracked"]) == 10 * int(np.ceil(70295 / 32))
