import numpy as np
import pytest

from src.v11_big_leap.export_official_kickoff import apply_second_half_kickoff


def test_kickoff_rule_changes_only_second_half_before_first_play():
    pred = np.array([0, 1, 2, 1, 0, 1, 0], dtype=np.int8)
    half = np.array([1, 1, 0, 2, 2, 2, 2], dtype=np.int8)
    original = pred.copy()
    result = apply_second_half_kickoff(pred, half, 5)
    np.testing.assert_array_equal(result, [0, 1, 2, 2, 2, 1, 0])
    np.testing.assert_array_equal(pred, original)


def test_kickoff_rule_rejects_invalid_boundary():
    with pytest.raises(ValueError):
        apply_second_half_kickoff(np.array([0, 0, 0]), np.array([1, 2, 0]), 3)
