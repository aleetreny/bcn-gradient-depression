"""Tests for src.data.skillcorner on a tiny synthetic match folder."""
from __future__ import annotations

import json

import numpy as np

from src.data.common import AWAY, DEAD, HOME
from src.data.skillcorner import build


def _match(tmp_path):
    mid = 1
    folder = tmp_path / str(mid)
    folder.mkdir()
    players = [{"id": 10, "team_id": 7, "player_role": {"acronym": "GK"}},
               {"id": 20, "team_id": 8, "player_role": {"acronym": "CF"}}]
    (folder / "1_match.json").write_text(json.dumps({
        "pitch_length": 100, "pitch_width": 60, "home_team": {"id": 7, "short_name": "H"},
        "away_team": {"id": 8, "short_name": "A"}, "players": players, "home_team_side": ["left_to_right", "right_to_left"]}))
    lines = []
    for f in range(10):                     # period 1: frames 0..9 at 10 fps
        group = {2: "home team", 6: "away team"}.get(f)
        lines.append({"frame": f, "period": 1, "ball_data": {"x": float(f), "y": 0.0, "z": 0.0, "is_detected": f != 5},
                      "possession": {"player_id": None, "group": group},
                      "player_data": [{"player_id": 10, "x": -40.0 + f, "y": 0.0, "is_detected": True},
                                      {"player_id": 20, "x": 10.0, "y": 5.0, "is_detected": f < 8}]})
    (folder / "1_tracking_extrapolated.jsonl").write_text("\n".join(json.dumps(r) for r in lines))
    (folder / "1_phases_of_play.csv").write_text("frame_start,frame_end,team_in_possession_id\n1,7,8\n")
    return build(mid, tmp_path)


def test_upsampling_coordinates_and_detection(tmp_path):
    d = _match(tmp_path)
    assert len(d["half"]) == 23                                 # 9 intervals of 0.1 s -> 22.5 -> 23 frames at 25 fps
    slot_gk = int(np.flatnonzero(d["slot_is_gk"])[0])
    assert np.isclose(d["P"][0, slot_gk, 0], 10.0)              # -40 + 50 (origin moved to the corner)
    assert np.isclose(d["P"][1, slot_gk, 0], 10.4)              # 0.04 s later at 10 m/s
    slot_cf = 1 - slot_gk
    assert np.isnan(d["P"][-1, slot_cf, 0])                     # not detected after frame 7
    assert np.isnan(d["B"][int(5 * 2.5), 0])                    # undetected ball
    assert (d["attack_sign"] == 1).all()


def test_labels_phase_alive_and_carried_possession(tmp_path):
    d = _match(tmp_path)
    lab10 = d["labels"][np.rint(np.arange(10) * 2.5).astype(int)]
    assert lab10[0] == DEAD and lab10[8] == DEAD and lab10[9] == DEAD   # outside the phase
    assert lab10[1] == AWAY                                     # before the first possessor: phase team
    assert (lab10[2:6] == HOME).all()                           # home possession carried forward
    assert (lab10[6:8] == AWAY).all()
