"""Shared fixtures. Tests use small synthetic data and never touch the network."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.models._api
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "results" / "best_model.pt"
CLASS_NAMES_JSON = REPO_ROOT / "results" / "class_names.json"


@pytest.fixture(autouse=True, scope="session")
def no_weight_downloads(tmp_path_factory):
    """Any attempt to fetch pretrained weights fails the test instead of hitting the network.

    Session-scoped so it is active before module-scoped fixtures build models.
    """
    def _blocked(*args, **kwargs):
        raise AssertionError(f"Unexpected pretrained-weight download: {args[:1]}")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.hub, "load_state_dict_from_url", _blocked)
        # torchvision binds its own reference at import time, so patch that one too
        mp.setattr(torchvision.models._api, "load_state_dict_from_url", _blocked)
        # An empty cache dir, so a locally cached download can't mask the problem either
        mp.setenv("TORCH_HOME", str(tmp_path_factory.mktemp("torch_home")))
        yield


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_image(rng, width, height):
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.fixture
def image_folder(tmp_path, rng):
    """data/{train,valid}/{class_a,class_b,class_c}/*.png with a few tiny random images each."""
    classes = ["Apple___healthy", "Tomato___Late_blight", "Corn_(maize)___Common_rust_"]
    for split, n in (("train", 4), ("valid", 2)):
        for cls in classes:
            d = tmp_path / "data" / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                random_image(rng, 48, 40).save(d / f"{i}.png")
    return tmp_path / "data", sorted(classes)
