"""Synthetic checks for local algorithm-generated CSV export."""
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.possession_metrics import read_label_csv, validate_submission
from src.v11_big_leap import export_local_submission as export


def test_exports_exact_algorithm_output_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "__file__", str(tmp_path / "src/v11_big_leap/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    data = pickle.dumps({"d3": {"pred_atd_rep": np.array([2, 0, 1, 2], dtype=np.int8)}})
    (local / "synthesis.pkl").write_bytes(data)
    (local / "v11_export_inputs.json").write_text(json.dumps({
        "variant": "d3", "synthesis": {"path": "_local/synthesis.pkl", "sha256": hashlib.sha256(data).hexdigest()}}))
    monkeypatch.setattr(export, "load_atd", lambda: SimpleNamespace(
        half=np.array([0, 1, 1, 0]), graded=np.array([False, True, True, False])))
    target = local / "output.csv"
    assert export.export_local_submission(tmp_path, target) == target
    assert validate_submission(read_label_csv(target), 4) is None
    assert read_label_csv(target)["label"].tolist() == ["D", "H", "A", "D"]
    with pytest.raises(FileExistsError):
        export.export_local_submission(tmp_path, target)


def test_rejects_ungraded_live_label(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "__file__", str(tmp_path / "src/v11_big_leap/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    data = pickle.dumps({"d3": {"pred_atd_rep": np.array([0], dtype=np.int8)}})
    (local / "synthesis.pkl").write_bytes(data)
    (local / "v11_export_inputs.json").write_text(json.dumps({
        "variant": "d3", "synthesis": {"path": "_local/synthesis.pkl", "sha256": hashlib.sha256(data).hexdigest()}}))
    monkeypatch.setattr(export, "load_atd", lambda: SimpleNamespace(half=np.array([0]), graded=np.array([False])))
    with pytest.raises(ValueError, match="ungraded"):
        export.export_local_submission(tmp_path, local / "output.csv")
    assert not (local / "output.csv").exists()
