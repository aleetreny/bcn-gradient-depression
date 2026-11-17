"""Safety checks for the local strict v8.5 baseline reconstruction."""
import pytest

from src.v9_swarm import rebuild_v85_baseline as reconstruction


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "baseline.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_v85_baseline(tmp_path, output)
    assert output.read_bytes() == b"keep"


def test_rejects_output_outside_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(reconstruction, "__file__", str(root / "src/v9_swarm/module.py"))
    with pytest.raises(ValueError, match="inside"):
        reconstruction.rebuild_v85_baseline(root, tmp_path / "baseline.pkl")
