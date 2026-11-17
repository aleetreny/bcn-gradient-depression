"""Tests for src.decode.viterbi_hsmm."""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, DEAD, HOME
from src.decode.viterbi_hsmm import DecoderParams, decode


def _proba(labels, conf=0.9):
    p = np.full((len(labels), 3), (1 - conf) / 2)
    p[np.arange(len(labels)), labels] = conf
    return p


def test_clean_input_is_reproduced():
    y = np.array([HOME] * 60 + [DEAD] * 40 + [AWAY] * 80, np.int8)
    out = decode(_proba(y), np.ones(len(y), np.int8), DecoderParams(lam_ha=3, lam_ld=3, lam_dl=3, dmin_team=10, dmin_dead=10))
    assert out.tolist() == y.tolist()


def test_flicker_shorter_than_min_duration_is_removed():
    y = np.array([HOME] * 60 + [AWAY] * 4 + [HOME] * 60, np.int8)
    out = decode(_proba(y), np.ones(len(y), np.int8), DecoderParams(lam_ha=1, lam_ld=1, lam_dl=1, dmin_team=10, dmin_dead=10))
    assert (out == HOME).all()


def test_switch_cost_suppresses_weak_changes_but_keeps_strong_ones():
    y = np.array([HOME] * 50 + [AWAY] * 30 + [HOME] * 50, np.int8)
    weak = _proba(y, conf=0.4)
    strong = _proba(y, conf=0.95)
    half = np.ones(len(y), np.int8)
    p = DecoderParams(lam_ha=20, lam_ld=20, lam_dl=20, dmin_team=1, dmin_dead=1)
    assert (decode(weak, half, p) == HOME).all()
    assert decode(strong, half, p).tolist() == y.tolist()


def test_halves_are_independent_and_ungraded_frames_are_dead():
    y = np.array([HOME] * 30 + [AWAY] * 30, np.int8)
    half = np.array([1] * 30 + [2] * 30, np.int8)
    out = decode(_proba(y), half, DecoderParams(lam_ha=100, lam_ld=100, lam_dl=100, dmin_team=1, dmin_dead=1))
    assert out.tolist() == y.tolist()                    # no switch cost across halftime
    half0 = half.copy(); half0[:5] = 0
    assert (decode(_proba(y), half0, DecoderParams())[:5] == DEAD).all()


def test_time_dependent_switch_cost_places_the_change_where_it_is_likely():
    # ambiguous emissions around a change; the change probability points to frame 60
    y = np.array([HOME] * 50 + [AWAY] * 50, np.int8)
    proba = _proba(y, conf=0.9)
    proba[40:70] = [0.45, 0.45, 0.10]                     # no information in 40..69
    pc = np.full(100, 0.01)
    pc[60] = 0.99
    half = np.ones(100, np.int8)
    p = DecoderParams(lam_ha=6, lam_ld=50, lam_dl=50, dmin_team=1, dmin_dead=1, beta_ha=2.0)
    out = decode(proba, half, p, p_change=pc)
    assert np.flatnonzero(np.diff(out.astype(int)) != 0).tolist() == [59]
    # beta 0 reproduces the plain decoder
    p0 = DecoderParams(lam_ha=6, lam_ld=50, lam_dl=50, dmin_team=1, dmin_dead=1)
    assert decode(proba, half, p0, p_change=pc).tolist() == decode(proba, half, p0).tolist()
