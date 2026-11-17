"""Tests for src.data.metrica_atd on a tiny synthetic EPTS pair (no competition data needed)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data import metrica_atd
from src.data.metrica_atd import Match, parse_meta, parse_raw

META = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
<Teams>
  <Team id="T_home"><Name>Home FC</Name><ProviderTeamsParameters><ProviderParameter>
    <Name>attack_direction_first_half</Name><Value>left_to_right</Value></ProviderParameter>
  </ProviderTeamsParameters></Team>
  <Team id="T_away"><Name>Away FC</Name><ProviderTeamsParameters><ProviderParameter>
    <Name>attack_direction_first_half</Name><Value>right_to_left</Value></ProviderParameter>
  </ProviderTeamsParameters></Team>
</Teams>
<Players>
  <Player id="2" teamId="T_away"><Name>Track 2</Name><ProviderPlayerParameters><ProviderParameter>
    <Name>position_type</Name><Value>Goalkeeper</Value></ProviderParameter></ProviderPlayerParameters></Player>
  <Player id="1" teamId="T_home"><Name>Track 1</Name><ProviderPlayerParameters><ProviderParameter>
    <Name>position_type</Name><Value>Unknown</Value></ProviderParameter></ProviderPlayerParameters></Player>
</Players>
</Metadata></main>
"""

RAW = ("1:0.10000,0.20000,1.5;NaN,NaN,NaN:0.50000,0.50000,NaN\n"
       "2:NaN,NaN,NaN;0.90000,0.40000,2.0:NaN,NaN,0.7\n"
       "3:0.11000,0.21000,1.6;0.89000,0.41000,2.1:NaN,NaN,NaN\n")


@pytest.fixture
def files(tmp_path):
    meta, raw = tmp_path / "m.xml", tmp_path / "r.txt"
    meta.write_text(META)
    raw.write_text(RAW)
    return meta, raw


def test_parse_meta_orders_slots_by_id_and_reads_teams(files):
    meta = parse_meta(files[0])
    assert meta["home_name"] == "Home FC" and meta["away_name"] == "Away FC"
    assert meta["slot_team"].tolist() == [0, 1]          # slot 1 home, slot 2 away (file order is 2, 1)
    assert meta["slot_is_gk"].tolist() == [False, True]
    assert meta["home_attack_first_half"] == "left_to_right"


def test_parse_raw_layout_and_nans(files):
    frame_ids, raw = parse_raw(files[1], n_slots=2)
    assert frame_ids.tolist() == [1, 2, 3]
    assert raw.shape == (3, 3, 3) and raw.dtype == np.float32
    np.testing.assert_allclose(raw[0, 0], [0.1, 0.2, 1.5], rtol=1e-6)
    assert np.isnan(raw[0, 1]).all()
    np.testing.assert_allclose(raw[0, 2, :2], [0.5, 0.5])   # ball x, y without third channel
    assert np.isnan(raw[1, 2, :2]).all() and raw[1, 2, 2] == pytest.approx(0.7)


def _match(attack_sign):
    P = np.array([[[10.0, 20.0, 1.0]], [[np.nan, np.nan, np.nan]]], np.float32)
    B = np.array([[30.0, 5.0, np.nan], [np.nan, np.nan, 0.4]], np.float32)
    return Match(P=P, B=B, slot_team=np.array([0], np.int8), slot_is_gk=np.array([False]),
                 half=np.array([1, 2], np.int8), graded=np.array([True, True]),
                 attack_sign=np.array(attack_sign, np.int8), pitch=(100.0, 60.0),
                 home_name="H", away_name="A")


def test_masks():
    m = _match([1, -1])
    assert m.slot_visible.tolist() == [[True], [False]]
    assert m.ball_xy_visible.tolist() == [True, False]
    assert m.ball_speed_only.tolist() == [False, True]


def test_normalized_rotates_only_flipped_frames_and_is_idempotent():
    m = _match([-1, 1])
    n = m.normalized()
    np.testing.assert_allclose(n.P[0, 0], [90.0, 40.0, 1.0])   # rotated 180 deg, speed untouched
    np.testing.assert_allclose(n.B[0, :2], [70.0, 55.0])
    assert np.isnan(n.P[1]).all()
    assert n.normalized() is n
    np.testing.assert_allclose(m.P[0, 0], [10.0, 20.0, 1.0])   # original unchanged


def test_frame_structure_directions():
    half, graded, sign = metrica_atd._frame_structure("BAR-ATH", 199319, "left_to_right")
    assert sign[11185] == 1 and sign[90651] == 1          # 1st half: H attacks +x
    assert sign[110808] == -1 and sign[190197] == -1      # 2nd half: H attacks -x
    assert graded.sum() == 158857 and half[100000] == 0
