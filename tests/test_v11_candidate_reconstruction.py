"""Safe output checks for the local v11 candidate reconstruction."""
import pytest

from src.v11_big_leap import rebuild_v11_candidates as reconstruction


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v11_big_leap/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "candidates.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_v11_candidates(tmp_path, output)
    assert output.read_bytes() == b"keep"
