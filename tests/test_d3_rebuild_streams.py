"""D3 reconstruction adapter routes explicit local inputs and publishes on success."""

import json
import pickle

import pytest

from src.v10_first_principles import rebuild_verified_streams as rebuild


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path.resolve()
    local = root / "_local"
    local.mkdir()
    (local / "d3_reconstruction.json").write_text(json.dumps({
        "variant": "count",
        "manifest_path": "_local/manifest.json",
        "feature_root": "_local/features",
        "stream_root": "_local/outputs",
        "staging_root": "_local",
    }))
    return root


def test_routes_config_to_verified_extraction(checkout, monkeypatch):
    tracking = object()

    def load(**kwargs):
        assert kwargs == {"data_dir": checkout / "data", "use_cache": False}
        return tracking

    def extract(variant, atd, **kwargs):
        assert variant == "count"
        assert atd is tracking
        assert kwargs == {
            "manifest_path": checkout / "_local/manifest.json",
            "feature_root": checkout / "_local/features",
            "stream_root": checkout / "_local/outputs",
            "staging_root": checkout / "_local",
            "project_root": checkout,
        }
        return {"synthetic": [1]}

    monkeypatch.setattr(rebuild, "load_atd", load)
    monkeypatch.setattr(rebuild, "extract_verified_d3_streams", extract)
    target = checkout / "_local/streams.pkl"

    assert rebuild.rebuild_verified_streams(checkout, target) == target
    with target.open("rb") as stream:
        assert pickle.load(stream) == {"synthetic": [1]}


def test_failed_extraction_creates_no_output(checkout, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic extraction failure")

    monkeypatch.setattr(rebuild, "load_atd", lambda **kwargs: object())
    monkeypatch.setattr(rebuild, "extract_verified_d3_streams", fail)
    target = checkout / "_local/streams.pkl"

    with pytest.raises(ValueError, match="synthetic extraction failure"):
        rebuild.rebuild_verified_streams(checkout, target)
    assert not target.exists()


def test_refuses_to_replace_a_file_created_during_extraction(checkout, monkeypatch):
    target = checkout / "_local/streams.pkl"

    def extract(*args, **kwargs):
        target.write_bytes(b"existing")
        return {}

    monkeypatch.setattr(rebuild, "load_atd", lambda **kwargs: object())
    monkeypatch.setattr(rebuild, "extract_verified_d3_streams", extract)

    with pytest.raises(FileExistsError):
        rebuild.rebuild_verified_streams(checkout, target)
    assert target.read_bytes() == b"existing"


def test_rejects_output_outside_checkout(checkout):
    link = checkout / "_local/outside"
    link.symlink_to(checkout.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="outside|inside the checkout"):
        rebuild.rebuild_verified_streams(checkout, link / "streams.pkl")
