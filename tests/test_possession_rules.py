"""Tests for src.features.possession_rules on synthetic sequences."""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, HOME, Match
from src.features.possession_rules import Kinematics, carry_forward, rule_control, rule_last_touch, rule_nearest


def test_carry_forward():
    assert carry_forward(np.array([-1, 0, -1, 1, -1], np.int8)).tolist() == [-1, 0, 0, 1, 1]


def _pass_match():
    """H player at x=10 passes to A player at x=30; ball travels 10 -> 30 between frames 10 and 30."""
    T = 50
    P = np.full((T, 2, 3), np.nan, np.float32)
    P[:, 0, :2] = (10, 30)
    P[:, 1, :2] = (30, 30)
    bx = np.r_[np.full(10, 10.3), np.linspace(10.3, 29.7, 20), np.full(20, 29.7)]
    B = np.stack([bx, np.full(T, 30.0), np.full(T, np.nan)], axis=1).astype(np.float32)
    return Match(P=P, B=B, slot_team=np.array([HOME, AWAY], np.int8), slot_is_gk=np.zeros(2, bool),
                 half=np.ones(T, np.int8), graded=np.ones(T, bool), attack_sign=np.ones(T, np.int8),
                 pitch=(100.0, 60.0), home_name="H", away_name="A")


def test_rules_follow_a_pass():
    kin = Kinematics.from_match(_pass_match())
    touch = rule_last_touch(kin, r=1.0, theta=20, delta=3.0)
    assert touch[12] == HOME and touch[-1] == AWAY
    assert 28 <= np.flatnonzero(touch == AWAY)[0] <= 31          # receiver's touch, not the midpoint
    near = rule_nearest(kin, k=0)
    assert near[19] == HOME and near[21] == AWAY                 # nearest flips at the midpoint
    assert rule_nearest(kin, k=5)[22] == HOME                    # hold delays the switch
    ctrl = rule_control(kin, r=1.0, m=3)
    assert ctrl[5] == HOME and ctrl[-1] == AWAY


def _two_team_match(dist_h, dist_a, half=None):
    """Ball fixed at the origin; one H and one A player at the given distances per frame."""
    T = len(dist_h)
    P = np.full((T, 2, 3), np.nan, np.float32)
    P[:, 0, 0], P[:, 1, 0] = dist_h, -np.asarray(dist_a)
    P[:, :, 1] = 0.0
    B = np.zeros((T, 3), np.float32)
    B[:, 2] = np.nan
    half = np.ones(T, np.int8) if half is None else np.asarray(half, np.int8)
    return Match(P=P, B=B, slot_team=np.array([HOME, AWAY], np.int8), slot_is_gk=np.zeros(2, bool),
                 half=half, graded=half > 0, attack_sign=np.ones(T, np.int8), pitch=(100.0, 60.0),
                 home_name="H", away_name="A")


def test_non_exclusive_control_is_symmetric():
    near, far = [0.1] * 4, [1.9] * 4
    a = rule_control(Kinematics.from_match(_two_team_match(near, far)), r=2.0, m=3, exclusive=False)
    b = rule_control(Kinematics.from_match(_two_team_match(far, near)), r=2.0, m=3, exclusive=False)
    assert a.tolist() == [-1, -1, HOME, HOME]
    assert b.tolist() == [-1, -1, AWAY, AWAY]


def test_kinematics_and_state_do_not_cross_halves():
    T = 12
    P = np.full((T, 1, 3), np.nan, np.float32)
    P[:, 0, :2] = (0.5, 0.0)
    B = np.zeros((T, 3), np.float32)
    B[6:, 0] = 20.0                                  # ball "jumps" only because halves are concatenated
    B[:, 2] = np.nan
    half = np.r_[np.ones(6), np.full(6, 2)].astype(np.int8)
    m = Match(P=P, B=B, slot_team=np.array([HOME], np.int8), slot_is_gk=np.zeros(1, bool), half=half,
              graded=np.ones(T, bool), attack_sign=np.ones(T, np.int8), pitch=(100.0, 60.0), home_name="H", away_name="A")
    kin = Kinematics.from_match(m)
    assert np.all(np.isnan(kin.dspeed) | (kin.dspeed < 1e-6))
    team = carry_forward(np.array([HOME, -1, -1, -1, -1, -1, -1, AWAY, -1, -1, -1, -1], np.int8), half)
    assert team[6] == -1 and team[5] == HOME and team[-1] == AWAY
