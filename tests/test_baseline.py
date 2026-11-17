"""Tests for src.models.baseline on a synthetic match."""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, DEAD, HOME, Match
from src.models.baseline import nearest_team, smooth


def _match(P, B, slot_team):
    T = len(B)
    return Match(P=np.asarray(P, np.float32), B=np.asarray(B, np.float32), slot_team=np.asarray(slot_team, np.int8),
                 slot_is_gk=np.zeros(len(slot_team), bool), half=np.ones(T, np.int8), graded=np.ones(T, bool),
                 attack_sign=np.ones(T, np.int8), pitch=(100.0, 60.0), home_name="H", away_name="A")


def test_nearest_team_handles_missing_ball_missing_slots_and_ties():
    nan = np.nan
    P = [[[10, 10, 0], [20, 10, 0]],          # ball nearer slot 0 (H)
         [[10, 10, 0], [20, 10, 0]],          # ball missing -> nobody
         [[nan, nan, 0], [20, 10, 0]],        # slot 0 unseen -> slot 1 (A)
         [[10, 10, 0], [20, 10, 0]],          # exact tie -> first slot (H)
         [[nan, nan, 0], [nan, nan, 0]]]      # nobody visible -> nobody
    B = [[11, 10, nan], [nan, nan, nan], [11, 10, nan], [15, 10, nan], [15, 10, nan]]
    assert nearest_team(_match(P, B, [HOME, AWAY])).tolist() == [HOME, -1, AWAY, HOME, -1]


def test_smooth_majority_ties_to_home_and_dead_when_nobody():
    codes = np.array([HOME, AWAY, -1, -1, -1, -1, AWAY], np.int8)
    assert smooth(codes, 1).tolist() == [HOME, HOME, AWAY, DEAD, DEAD, AWAY, AWAY]
