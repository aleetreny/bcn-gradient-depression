"""Held-out and output safety checks for trajectory fold reconstruction."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.v10_first_principles import rebuild_trajectory_fold as reconstruction


def test_fold_excludes_held_out_and_uses_recorded_weights(tmp_path, monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class Config:
        static_hide_prob: float = 0.0

    monkeypatch.setattr(reconstruction, "MATCH_IDS", ("a", "held", "c"))
    monkeypatch.setattr(reconstruction, "_load_degrade_configs", lambda root: {"v3": Config(), "v6": Config()})
    monkeypatch.setattr(reconstruction, "load", lambda mid, **kwargs: SimpleNamespace(
        name=mid, labels=mid, half=np.ones(3), meta={"ball_z": None}))
    monkeypatch.setattr(reconstruction, "_load_tensors", lambda cache, mid, tag, *args: (mid, tag))
    seeds = []

    class Model:
        def __init__(self, seed):
            seeds.append(seed)
        def fit(self, tensors, labels, halves, ball_zs, sample_weights):
            assert set(labels) == {"a", "c"}
            assert sample_weights == [1.0, 1.0, 0.95, 0.85] * 2
            return self
        def predict_all_heads(self, test, half):
            assert test[0] == "held"
            return test

    monkeypatch.setattr(reconstruction, "_new_model", Model)
    monkeypatch.setattr(reconstruction, "_pack_predictions", lambda heads, half: {
        key: np.zeros(3) for key in reconstruction.OUTPUT_HEADS})
    result = reconstruction._fit_fold(tmp_path, "held")
    assert seeds == [43, 143]
    assert set(result) == {f"{tag}_{key}" for tag in ("v3", "v6") for key in reconstruction.OUTPUT_HEADS}


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v10_first_principles/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / f"traj_v10_{reconstruction.MATCH_IDS[0]}.npz"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_trajectory_fold(tmp_path, output)
    assert output.read_bytes() == b"keep"
