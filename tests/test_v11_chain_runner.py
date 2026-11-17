"""Synthetic checks for the v11 chain runner's configuration guards."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_v11_chain", ROOT / "scripts/v11_big_leap/run_v11_chain.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


def _config(tmp_path, **overrides):
    config = {"inputs": {}, "d3_manifest": {"path": "m", "sha256": "0"}, "count_stream_root": "c",
              "output_dir": f"_local/test-chain-{tmp_path.name}"}
    config.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_rejects_missing_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"inputs": {}}))
    with pytest.raises(ValueError):
        chain.load_config(ROOT, path)


def test_rejects_output_outside_local(tmp_path):
    with pytest.raises(ValueError):
        chain.load_config(ROOT, _config(tmp_path, output_dir="experiments/outputs/x"))


def test_rejects_input_target_outside_outputs(tmp_path):
    bad = {"data/x.npz": {"path": "p", "sha256": "0"}}
    with pytest.raises(ValueError):
        chain.load_config(ROOT, _config(tmp_path, inputs=bad))


def test_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError):
        chain.load_config(ROOT, _config(tmp_path, j3=True))


def test_hash_mismatch_is_refused(tmp_path):
    source = tmp_path / "a.bin"
    source.write_bytes(b"abc")
    with pytest.raises(ValueError):
        chain.verified({"path": str(source), "sha256": "0" * 64})
    assert chain.verified({"path": str(source), "sha256": chain.sha256(source)}) == source


def test_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError):
        chain.load_config(ROOT, _config(tmp_path, variant="video"))


def test_accepts_grand_variant(tmp_path):
    assert chain.load_config(ROOT, _config(tmp_path, variant="grand"))["variant"] == "grand"
