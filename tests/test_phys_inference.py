"""Packing and destination checks for local PhysConformer inference."""
import numpy as np
import pytest

from src.v11_big_leap import rebuild_phys_inference as reconstruction


def test_pack_preserves_auxiliary_heads_and_normalizes():
    proba = np.tile(np.array([0.2, 0.3, 0.5], dtype=np.float32), (4, 1))
    one = np.full(4, 0.2, dtype=np.float32)
    p4d = np.full((4, 4), 0.3, dtype=np.float32)
    heads = (proba, one, p4d, one, one, one)
    result = reconstruction._pack_predictions([heads, heads])
    np.testing.assert_allclose(result["proba"], proba)
    np.testing.assert_array_equal(result["p_duel_to"], one)
    with pytest.raises(ValueError, match="two models"):
        reconstruction._pack_predictions([heads])


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v11_big_leap/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "phys.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_phys_inference(tmp_path, output)
    assert output.read_bytes() == b"keep"
