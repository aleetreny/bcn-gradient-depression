"""Tests for src.data.degrade on a synthetic clean match (no downloaded data needed)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.data.common import AWAY, HOME, Match
from src.data.degrade import DegradeConfig, degrade

PARAMS = {
    "gap_start_rate_by_ball_speed": {"edges_ms": [0, 2, 5, 10, 15, 25, 60], "rate": [4e-3, 2e-3, 8e-4, 3e-4, 1e-4, 1e-3]},
    "gap_start_rate_by_players_within_3m": {"edges": [0, 1, 2, 3, 5, 30], "rate": [1e-3, 1e-3, 2e-3, 4e-3, 0.0]},
    "gap_length_frames_hist": {"edges": [0, 100, 250, 500, 1000], "all": [0, 5, 5, 2], "moving_play": [0, 4, 2, 0]},
}


def _clean_match(T=6000, seed=0):
    rng = np.random.default_rng(seed)
    S = 22
    P = np.zeros((T, S, 3), np.float32)
    walk = np.cumsum(rng.normal(0, 0.05, (T, S, 2)), axis=0) + rng.uniform(10, 60, (1, S, 2))
    P[..., :2] = walk
    P[..., 2] = 2.0
    B = np.zeros((T, 3), np.float32)
    B[:, :2] = np.cumsum(rng.normal(0, 0.2, (T, 2)), axis=0) + 50
    labels = rng.integers(0, 3, T).astype(np.int8)
    half = np.r_[np.ones(T // 2), np.full(T - T // 2, 2)].astype(np.int8)
    return Match(P=P, B=B, slot_team=np.r_[np.full(11, HOME), np.full(11, AWAY)].astype(np.int8),
                 slot_is_gk=np.r_[True, np.zeros(10, bool), True, np.zeros(10, bool)], half=half, graded=half > 0,
                 attack_sign=np.ones(T, np.int8), pitch=(105.0, 68.0), home_name="H", away_name="A", labels=labels)


CFG = DegradeConfig(params=PARAMS, absence_rate=1e-3, episode_rate=5e-4, cut_rate=1e-3, dup_prob=0.5,
                    gap_rate_moving=2e-3, gap_rate_slow=5e-3)


def test_deterministic_and_labels_untouched():
    m = _clean_match()
    a, b = degrade(m, CFG, seed=3), degrade(m, CFG, seed=3)
    np.testing.assert_array_equal(np.isnan(a.P), np.isnan(b.P))
    np.testing.assert_allclose(np.nan_to_num(a.P), np.nan_to_num(b.P))
    assert np.array_equal(a.labels, m.labels)
    assert not np.array_equal(np.isnan(degrade(m, CFG, seed=4).B), np.isnan(a.B))


def test_degradation_never_reads_labels():
    m = _clean_match()
    shuffled = replace(m, labels=np.random.default_rng(9).permutation(m.labels))
    a, b = degrade(m, CFG, seed=1), degrade(shuffled, CFG, seed=1)
    np.testing.assert_array_equal(np.isnan(a.B), np.isnan(b.B))
    np.testing.assert_allclose(np.nan_to_num(a.P), np.nan_to_num(b.P))


def test_team_fixed_slots_and_one_goalkeeper_per_team():
    d = degrade(_clean_match(), CFG, seed=0)
    assert (d.slot_team[:24] == HOME).all() and (d.slot_team[24:46] == AWAY).all()
    assert d.slot_is_gk[d.slot_team == HOME].sum() == 1 and d.slot_is_gk[d.slot_team == AWAY].sum() == 1
    vis = d.slot_visible
    assert vis[:, d.slot_team == HOME].sum(1).max() <= 24
    assert np.isnan(d.B[:, 0]).any()                      # some ball gaps were created


def test_goalkeeper_keeps_a_dedicated_slot():
    m = _clean_match()
    d = degrade(m, CFG, seed=2)
    for team in (HOME, AWAY):
        gk_src = np.flatnonzero(m.slot_is_gk & (m.slot_team == team))[0]
        gk_dst = np.flatnonzero(d.slot_is_gk & (d.slot_team == team))[0]
        both = np.isfinite(d.P[:, gk_dst, 0])
        # wherever the keeper slot is visible it carries the keeper's (smoothed) track, never another player
        assert both.any()
        assert np.nanmax(np.abs(d.P[both, gk_dst, 0] - m.P[both, gk_src, 0])) < 1.0


def test_no_frozen_ball_survives_degradation():
    m = _clean_match()
    B = m.B.copy()
    B[1000:1400, :2] = B[1000, :2]                       # a frozen dead ball, as in Sportec
    d = degrade(replace(m, B=B), replace(CFG, static_hide_prob=0.0), seed=0)
    seg = d.B[1005:1395, :2]
    vis = np.isfinite(seg[:, 0])
    speed = np.linalg.norm(np.diff(seg[vis], axis=0), axis=1)
    assert vis.sum() > 100 and np.mean(speed == 0) < 0.01


def test_ball_interpolation_draws_straight_runs_and_keeps_gaps():
    from src.data.ball_interp import interpolate, linear_mask
    rng = np.random.default_rng(0)
    T = 600
    t = np.arange(T) / 25
    B = np.stack([10 + 5 * np.sin(t), 30 + 3 * np.cos(1.3 * t), np.zeros(T)], 1).astype(np.float32)
    B[:, :2] += rng.normal(0, 0.05, (T, 2)).astype(np.float32)                                       # noisy real ball
    B[200:260] = np.nan                                                                               # a gap
    half = np.ones(T, np.int8)
    params = {"edges": [1, 8, 32], "linear_counts": [0, 1], "curved_counts": [1, 0]}
    out = interpolate(B, half, rng, params)
    assert np.isnan(out[200:260, :2]).all()
    assert linear_mask(out, half).mean() > 0.5 > linear_mask(B, half).mean()
    capped = interpolate(B, half, np.random.default_rng(0), params, dev_max=0.01)
    assert np.max(np.linalg.norm(np.nan_to_num(capped[:, :2] - B[:, :2]), axis=1)) <= 0.01 + 1e-6
