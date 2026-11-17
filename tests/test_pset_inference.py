"""Packing and output checks for local PSet inference."""
import numpy as np
import pytest

from src.v9_swarm import rebuild_pset_inference as reconstruction


def test_prediction_packing():
    proba = np.tile(np.array([0.2, 0.3, 0.5], dtype=np.float32), (4, 1))
    one = np.full(4, 0.2, dtype=np.float32)
    p4d = np.full((4, 4), 0.3, dtype=np.float32)
    heads = (proba, one, p4d, one, one)
    result = reconstruction._pack_predictions([heads, heads])
    np.testing.assert_allclose(result["pset_proba"], proba)
    np.testing.assert_array_equal(result["pset_p_aerial"], one)
    with pytest.raises(ValueError, match="two models"):
        reconstruction._pack_predictions([heads])


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "pset.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_pset_inference(tmp_path, output)
    assert output.read_bytes() == b"keep"
