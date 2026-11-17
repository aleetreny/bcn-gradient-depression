"""Safety and registration checks for local cache reconstruction."""
from pathlib import Path

import pytest

from harness import registry
from src.v9_swarm.rebuild_enriched_candidates import rebuild_enriched_candidates


def test_reconstruction_is_registered_separately():
    assert "v9_enriched_candidates" in registry.names("reconstruction")
    assert callable(registry.get("reconstruction", "v9_enriched_candidates"))
    assert "v9_enriched_candidates" not in registry.names("predictor")
    assert {"v9_duel_candidates", "v9_mbr_candidates"} <= set(registry.names("reconstruction"))


@pytest.mark.parametrize("destination", ["data/new.pkl", "submissions/new.pkl", "_local/labels.csv"])
def test_reconstruction_rejects_wrong_destinations(tmp_path: Path, destination: str):
    with pytest.raises(ValueError):
        rebuild_enriched_candidates(tmp_path, tmp_path / destination)


def test_reconstruction_does_not_overwrite(tmp_path: Path):
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "existing.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        rebuild_enriched_candidates(tmp_path, output)
    assert output.read_bytes() == b"keep"


def test_reconstruction_rejects_path_outside_root(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(ValueError):
        rebuild_enriched_candidates(root, tmp_path / "outside.pkl")


def test_reconstruction_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "_local").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        rebuild_enriched_candidates(root, root / "_local" / "outside.pkl")


def test_candidate_rebuilders_reject_foreign_root(tmp_path: Path):
    from src.v9_swarm.rebuild_candidates import rebuild_candidates
    with pytest.raises(ValueError, match="checkout"):
        rebuild_candidates(tmp_path, tmp_path / "_local" / "candidate.pkl", "duel")


def test_candidate_extractors_handle_empty_runs():
    import numpy as np
    import pandas as pd
    from src.v9_swarm.rebuild_candidates import extract_duel_cands_v9, extract_mbr_xyx_cands

    size = 20
    labels = np.full(size, 2, dtype=np.int8)
    proba = np.full((size, 3), 1 / 3)
    change = np.zeros(size)
    boundary = np.zeros((size, 2))
    family = [proba] * 9
    columns = ("dist_ball_H", "dist_ball_A", "ball_visible", "chord_in_flight",
               "chord_turnover_prob", "impulse_mag_d2", "impulse_mag_d4", "impulse_cos_d2",
               "impulse_cos_d4", "ball_angle_change", "ball_speed", "touch_team_r1", "touch_team_r2")
    features = pd.DataFrame({key: np.zeros(size) for key in columns})
    half = np.ones(size, dtype=np.int8)
    for extract in (extract_duel_cands_v9, extract_mbr_xyx_cands):
        assert extract(labels, proba, change, boundary, family, half, features) == []
