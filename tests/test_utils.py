import json

import pytest
import torch

import utils
from conftest import CLASS_NAMES_JSON
from utils import AverageMeter, EarlyStopping, get_device, pretty_class_name


class TestGetDevice:
    def test_prefers_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert get_device().type == "cuda"

    def test_falls_back_to_mps(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert get_device().type == "mps"

    def test_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert get_device().type == "cpu"

    def test_explicit_override(self):
        assert get_device("cpu").type == "cpu"


class TestEarlyStopping:
    def test_stops_after_patience_without_improvement(self):
        es = EarlyStopping(patience=2, mode="max")
        assert es.step(0.5) is True
        assert es.step(0.4) is False and not es.should_stop
        assert es.step(0.5) is False and es.should_stop  # equal is not an improvement

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=2)
        es.step(0.5)
        es.step(0.4)
        assert es.step(0.6) is True
        assert es.counter == 0 and not es.should_stop

    def test_min_mode(self):
        es = EarlyStopping(patience=1, mode="min")
        assert es.step(1.0) and es.step(0.5)
        assert not es.step(0.7) and es.should_stop

    def test_reset_patience_keeps_best(self):
        es = EarlyStopping(patience=1)
        es.step(0.9)
        es.step(0.1)
        assert es.should_stop
        es.reset_patience()
        assert not es.should_stop and es.counter == 0 and es.best == 0.9
        assert es.step(0.8) is False  # still compared against the best from before the reset

    def test_rejects_bad_mode(self):
        with pytest.raises(ValueError):
            EarlyStopping(mode="maximum")


def test_average_meter_weights_by_count():
    m = AverageMeter()
    assert m.avg == 0.0
    m.update(1.0, n=3)
    m.update(0.0, n=1)
    assert m.avg == pytest.approx(0.75)


@pytest.mark.parametrize("raw, expected", [
    ("Corn_(maize)___Common_rust_", "Corn (maize) — Common rust"),
    ("Pepper,_bell___healthy", "Pepper, bell — Healthy"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato — Tomato Yellow Leaf Curl Virus"),
    ("no_separator", "No separator"),
])
def test_pretty_class_name(raw, expected):
    assert pretty_class_name(raw) == expected


def test_pretty_names_are_unique_for_committed_classes():
    names = json.loads(CLASS_NAMES_JSON.read_text())
    assert len({pretty_class_name(n) for n in names}) == len(names)


def test_class_names_roundtrip(tmp_path):
    names = ["b", "a"]
    utils.save_class_names(names, tmp_path / "nested" / "class_names.json")
    assert utils.load_class_names(tmp_path / "nested" / "class_names.json") == names
