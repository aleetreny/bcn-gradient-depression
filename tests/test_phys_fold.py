"""Held-out isolation and output protection for PhysConformer folds."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.v11_big_leap import rebuild_phys_fold as reconstruction


def test_fold_excludes_held_out_and_preserves_weights(tmp_path, monkeypatch):
    @dataclass
    class Config:
        static_hide_prob: float = 0.0

    monkeypatch.setattr(reconstruction, "MATCH_IDS", ("a", "held", "c"))
    monkeypatch.setattr(reconstruction, "_load_degrade_configs", lambda root: {"v3": Config(), "v6": Config()})
    monkeypatch.setattr(reconstruction, "load", lambda mid, **kwargs: SimpleNamespace(
        name=mid, labels=mid, half=np.ones(3), meta={"ball_z": None}))
    monkeypatch.setattr(reconstruction, "_features", lambda root, mid, match, tag, config, seed:
                        (np.zeros((3, 1)), np.zeros((3, 1)), ["feature"]))
    seeds = []

    class Helper:
        def __init__(self, **kwargs):
            self.sym_mean, self.sym_std = 0, 1
        def prepare_segments(self, xorig, xmirror, labels, halves, ball_zs, sample_weights):
            assert set(labels) == {"a", "c"}
            assert sample_weights == [1.0, 1.0, 0.95, 0.85] * 2
            return "segments"

    class Model:
        def __init__(self, columns, seed):
            seeds.append(seed)
        def fit(self, xorig, xmirror, labels, halves, prepared_segments, sym_stats):
            assert prepared_segments == "segments"
            assert sym_stats == (0, 1)
            return self
        def predict_all_heads(self, xo, xm, half):
            return (xo, xm)

    monkeypatch.setattr(reconstruction, "SymPhysConformerV11", Helper)
    monkeypatch.setattr(reconstruction, "_model", Model)
    monkeypatch.setattr(reconstruction, "_pack_predictions", lambda heads: {
        "proba": np.zeros((3, 3)), "p_duel_to": np.zeros(3), "p_aerial": np.zeros(3)})
    result = reconstruction._fit_fold(tmp_path, "held")
    assert seeds == [43, 143]
    assert set(result) == {f"{tag}_{head}" for tag in ("v3", "v6")
                           for head in ("proba", "duel_to", "aerial")}


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v11_big_leap/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / f"pc11_{reconstruction.MATCH_IDS[0]}.npz"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_phys_fold(tmp_path, output)
    assert output.read_bytes() == b"keep"
