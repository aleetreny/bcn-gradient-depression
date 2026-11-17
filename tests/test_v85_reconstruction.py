from pathlib import Path

import pickle
import pytest

from src.v9_swarm import rebuild_v85_streams as reconstruction


@pytest.fixture
def project(tmp_path, monkeypatch):
    module_dir = tmp_path / "src" / "v9_swarm"
    monkeypatch.setattr(reconstruction, "__file__", str(module_dir / "rebuild_v85_streams.py"))
    return tmp_path


def test_v85_output_is_local_and_exclusive(project, monkeypatch):
    payload = {"v3": {"synthetic": {"pred": [0, 1]}}}
    monkeypatch.setattr(reconstruction, "_assemble_streams", lambda root: payload)
    output = Path("_local") / "nested" / "reconstructed.pkl"

    result = reconstruction.rebuild_v85_streams(project, output)

    assert result == project / output
    with result.open("rb") as handle:
        assert pickle.load(handle) == payload

    with pytest.raises(FileExistsError):
        reconstruction.rebuild_v85_streams(project, output)


@pytest.mark.parametrize("output", ["../outside.pkl", "submissions/cache.pkl", "_local/cache.csv"])
def test_v85_rejects_invalid_destination_before_assembly(project, monkeypatch, output):
    def unexpected_assembly(root):
        pytest.fail("attempted assembly with invalid destination")

    monkeypatch.setattr(reconstruction, "_assemble_streams", unexpected_assembly)
    with pytest.raises(ValueError):
        reconstruction.rebuild_v85_streams(project, Path(output))


def test_v85_rejects_symlinked_destination(project, monkeypatch):
    output = project / "_local"
    output.mkdir()
    (output / "escape").symlink_to(project.parent, target_is_directory=True)

    def unexpected_assembly(root):
        pytest.fail("attempted assembly with symlinked destination")

    monkeypatch.setattr(reconstruction, "_assemble_streams", unexpected_assembly)
    with pytest.raises(ValueError):
        reconstruction.rebuild_v85_streams(project, output / "escape" / "cache.pkl")


def test_v85_removes_partial_output_on_serialization_failure(project, monkeypatch):
    monkeypatch.setattr(reconstruction, "_assemble_streams", lambda root: object())

    def fail_after_write(cache, handle):
        handle.write(b"incomplete")
        raise OSError("synthetic write failure")

    monkeypatch.setattr(reconstruction.pickle, "dump", fail_after_write)
    output = project / "_local" / "new.pkl"
    with pytest.raises(OSError, match="synthetic write failure"):
        reconstruction.rebuild_v85_streams(project, output)
    assert not output.exists()
