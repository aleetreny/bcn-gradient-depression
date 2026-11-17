"""Output safety checks for the local v8.5 candidate reconstruction."""
import pytest

from src.v9_swarm import rebuild_v85_candidates as reconstruction


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "candidates.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_v85_candidates(tmp_path, output)
    assert output.read_bytes() == b"keep"
