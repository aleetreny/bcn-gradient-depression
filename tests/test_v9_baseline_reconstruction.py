"""Safety and copy semantics for the local v9 baseline reconstruction."""
import numpy as np
import pytest

from src.v9_swarm import rebuild_v9_baseline as reconstruction


def test_clone_bundle_does_not_mutate_source():
    original = {key: {"match": np.array([0, 1])} for key in ("preds_v3", "preds_v6_std", "preds_v6_gen", "preds_sample")}
    original.update({"pred_atd_rep": np.array([0, 1]), "pred_atd_raw": np.array([1, 0])})
    copied = reconstruction._clone_bundle(original)
    copied["preds_v3"]["match"][0] = 2
    copied["pred_atd_rep"][0] = 2
    assert original["preds_v3"]["match"][0] == 0
    assert original["pred_atd_rep"][0] == 0


def test_existing_cache_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "baseline.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_v9_baseline(tmp_path, output)
    assert output.read_bytes() == b"keep"
