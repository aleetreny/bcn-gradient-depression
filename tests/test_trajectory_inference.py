"""Checks for local-only trajectory inference inputs and packing."""
import hashlib

import numpy as np
import pytest

from src.v10_first_principles.rebuild_trajectory_inference import _pack_predictions, _verified_file


def test_verified_input_rejects_changed_file(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"initial")
    item = {"path": "input.bin", "sha256": hashlib.sha256(b"initial").hexdigest()}
    assert _verified_file(tmp_path, item) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _verified_file(tmp_path, item)


def test_prediction_packing_preserves_raw_and_aliases():
    half = np.ones(12, dtype=np.int8)
    proba = np.tile(np.array([0.2, 0.3, 0.5], dtype=np.float32), (12, 1))
    one = np.full(12, 0.2, dtype=np.float32)
    p4d = np.full((12, 4), 0.3, dtype=np.float32)
    heads = (proba, one, p4d, one, one, one)
    result = _pack_predictions([heads, heads], half)
    np.testing.assert_array_equal(result["proba_raw"], proba)
    np.testing.assert_array_equal(result["pset_proba"], result["proba"])
    np.testing.assert_array_equal(result["pset_p_duel_to"], result["p_duel_to"])
    with pytest.raises(ValueError, match="two models"):
        _pack_predictions([heads], half)
