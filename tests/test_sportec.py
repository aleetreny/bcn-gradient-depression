"""Tests for src.data.sportec on a tiny synthetic DFL pair (no downloaded data needed)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data import sportec
from src.data.common import AWAY, DEAD, HOME

INFO = """<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest><MatchInformation><Teams>
<Team TeamId="CLU-H" TeamName="Home FC" Role="home"><Players>
  <Player PersonId="P-GK" PlayingPosition="TW"/><Player PersonId="P-H1" PlayingPosition="STZ"/></Players></Team>
<Team TeamId="CLU-A" TeamName="Away FC" Role="guest"><Players>
  <Player PersonId="P-A1" PlayingPosition="STZ"/></Players></Team>
</Teams></MatchInformation></PutDataRequest>
"""


def frame(n, x, y, ball=None):
    base = '<Frame N="{}" T="t" X="{:.2f}" Y="{:.2f}" '.format(n, x, y)
    if ball is None:
        return base + 'D="0.00" S="3.60" A="0.00" M="1"/>\n'
    poss, status = ball
    return base + 'Z="0.10" D="0.00" S="0.00" A="0.00" M="1" BallPossession="{}" BallStatus="{}"/>\n'.format(poss, status)


def frameset(half, team, person, frames):
    return ('<FrameSet GameSection="{}" MatchId="DFL-MAT-TEST01" TeamId="{}" PersonId="{}">\n'.format(half, team, person)
            + "".join(frames) + "</FrameSet>\n")


def positions():
    out = ['<?xml version="1.0"?>\n<PutDataRequest><Positions>\n<MetaData><PitchSize X="100.00" Y="60.00" /></MetaData>\n']
    # first half N=10..13: home on the left (attacks +x), ball labels H, H, A, dead
    out.append(frameset("firstHalf", "CLU-H", "P-GK", [frame(n, -45, 0) for n in range(10, 14)]))
    out.append(frameset("firstHalf", "CLU-H", "P-H1", [frame(n, -10 + 0.04 * (n - 10), 0) for n in range(10, 14)]))
    out.append(frameset("firstHalf", "CLU-A", "P-A1", [frame(n, 20, 5) for n in range(10, 14)]))
    out.append(frameset("firstHalf", "referee", "REF", [frame(n, 0, 0) for n in range(10, 14)]))
    out.append(frameset("firstHalf", "BALL", "B", [frame(10, 0, 0, (1, 1)), frame(11, 0, 0, (1, 1)),
                                                   frame(12, 0, 0, (2, 1)), frame(13, 0, 0, (2, 0))]))
    # second half N=50..52: sides swapped; P-H1 plays only frames 51-52
    out.append(frameset("secondHalf", "CLU-H", "P-GK", [frame(n, 45, 0) for n in range(50, 53)]))
    out.append(frameset("secondHalf", "CLU-H", "P-H1", [frame(n, 10, 0) for n in range(51, 53)]))
    out.append(frameset("secondHalf", "CLU-A", "P-A1", [frame(n, -20, 5) for n in range(50, 53)]))
    out.append(frameset("secondHalf", "BALL", "B", [frame(n, 1, 1, (1, 1)) for n in range(50, 53)]))
    out.append("</Positions></PutDataRequest>\n")
    return "".join(out)


@pytest.fixture
def match(tmp_path):
    (tmp_path / "DFL_02_01_matchinformation_DFL-COM-1_DFL-MAT-TEST01.xml").write_text(INFO)
    (tmp_path / "DFL_04_03_positions_raw_observed_DFL-COM-1_DFL-MAT-TEST01.xml").write_text(positions())
    return sportec.load("TEST01", data_dir=tmp_path, use_cache=False)


def test_shapes_halves_and_labels(match):
    assert match.P.shape == (7, 3, 3)                       # 4 + 3 frames, 3 players (referee dropped)
    assert match.half.tolist() == [1, 1, 1, 1, 2, 2, 2]
    assert match.labels.tolist() == [HOME, HOME, AWAY, DEAD, HOME, HOME, HOME]
    assert match.meta["frame_n"].tolist() == [10, 11, 12, 13, 50, 51, 52]
    assert match.home_name == "Home FC" and match.pitch == (100.0, 60.0)


def test_coordinates_shifted_and_substitute_nan(match):
    gk = int(np.flatnonzero(match.slot_is_gk)[0])
    assert match.slot_team[gk] == HOME
    np.testing.assert_allclose(match.P[0, gk, :2], [5.0, 30.0])     # -45 + 50, 0 + 30
    h1 = [s for s in range(3) if match.slot_team[s] == HOME and s != gk][0]
    assert np.isnan(match.P[4, h1, 0]) and np.isfinite(match.P[5, h1, 0])
    assert sorted(match.slot_team.tolist()) == [HOME, HOME, AWAY]


def test_speed_from_positions_in_ms(match):
    h1 = [s for s in range(3) if match.slot_team[s] == HOME and not match.slot_is_gk[s]][0]
    np.testing.assert_allclose(match.P[1, h1, 2], 0.04 * 25, rtol=1e-4)   # 4 cm per frame = 1 m/s


def test_attack_sign_per_half(match):
    assert match.attack_sign.tolist() == [1, 1, 1, 1, -1, -1, -1]
    n = match.normalized()
    gk = int(np.flatnonzero(match.slot_is_gk)[0])
    assert n.P[0, gk, 0] < 50 and n.P[4, gk, 0] < 50        # H goalkeeper always on the low-x side
