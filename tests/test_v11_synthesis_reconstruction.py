"""Safety checks for the local v11 synthesis reconstruction."""
import pytest

from src.v11_big_leap import rebuild_v11_synthesis as reconstruction


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v11_big_leap/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "synthesis.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_v11_synthesis(tmp_path, output)
    assert output.read_bytes() == b"keep"


def test_rejects_outside_output(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(reconstruction, "__file__", str(root / "src/v11_big_leap/module.py"))
    with pytest.raises(ValueError, match="inside"):
        reconstruction.rebuild_v11_synthesis(root, tmp_path / "outside.pkl")
