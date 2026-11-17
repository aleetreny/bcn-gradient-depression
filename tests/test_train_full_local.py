"""Safety checks for locally saved full-model weights."""
import pytest

from src.v9_swarm import train_full_local as training


def test_rejects_nonlocal_model_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    with pytest.raises(ValueError, match="under _local"):
        training.train_full_local(tmp_path, tmp_path / "experiments/outputs/pset_full_42.pkl")


def test_never_replaces_model(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "pset_full_42.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        training.train_full_local(tmp_path, output)
    assert output.read_bytes() == b"keep"
