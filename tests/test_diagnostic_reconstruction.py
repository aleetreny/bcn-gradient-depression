"""Safety contract for the historical local diagnostic reconstruction."""
from pathlib import Path

import pytest

from harness import registry
from src.v9_swarm.rebuild_diagnostic_streams import rebuild_diagnostic_streams


def test_diagnostics_are_registered_as_reconstruction():
    assert callable(registry.get("reconstruction", "v9_diagnostic_streams"))
    assert "v9_diagnostic_streams" not in registry.names("predictor")


def test_diagnostics_reject_other_checkout(tmp_path):
    with pytest.raises(ValueError, match="checkout"):
        rebuild_diagnostic_streams(tmp_path, tmp_path / "_local" / "cache.pkl")


def test_diagnostics_reject_outside_path(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="inside"):
        rebuild_diagnostic_streams(root, tmp_path / "cache.pkl")
