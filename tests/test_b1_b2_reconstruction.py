"""Input routing and safe destinations for v10 local candidate reconstruction."""
import numpy as np
import pytest

from src.v10_first_principles import rebuild_b1_b2 as rebuild


def test_fold_prediction_routing(tmp_path):
    path = tmp_path / "fold.npz"
    np.savez(path, v3_proba=[1], v6_proba=[2], gbm_v3=[3], gbm_v6=[4])
    assert list(rebuild._fold_predictions(path, "v3")) == ["proba"]
    assert list(rebuild._fold_predictions(path, "v6", suffix=True)) == ["gbm"]


def test_rejects_existing_output_before_loading_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, "__file__", str(tmp_path / "src/v10_first_principles/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    target = local / "existing.pkl"
    target.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        rebuild.rebuild_b1_b2(tmp_path, target, "b1")
    assert target.read_bytes() == b"preserve"


def test_rejects_output_outside_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(rebuild, "__file__", str(root / "src/v10_first_principles/module.py"))
    with pytest.raises(ValueError, match="inside"):
        rebuild.rebuild_b1_b2(root, tmp_path / "cache.pkl", "b2")
