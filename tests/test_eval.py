"""Tests for src.eval on synthetic sequences."""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, DEAD, HOME
from src.eval import errors, metrics


def test_perfect_prediction_scores_one():
    gt = np.array([HOME] * 50 + [DEAD] * 30 + [AWAY] * 60, np.int8)
    half = np.ones(len(gt), np.int8)
    r = metrics.score(gt, gt.copy(), half)
    assert r["spell_f1_mean"] == 1.0 and r["macro_f1"] == 1.0


def test_half_separator_splits_a_spell_crossing_halves():
    gt = np.array([HOME] * 100, np.int8)
    half = np.array([1] * 50 + [2] * 50, np.int8)
    assert metrics.score(gt, gt, half)["n_gt_spells"] == 2
    assert metrics.score(gt, gt, half, insert_gaps=False)["n_gt_spells"] == 1


def test_unlabelled_truth_is_not_scored():
    gt = np.array([-1] * 20 + [HOME] * 40, np.int8)
    pred = np.array([AWAY] * 20 + [HOME] * 40, np.int8)
    r = metrics.score(gt, pred, np.ones(60, np.int8))
    assert r["n_graded"] == 40 and r["spell_f1_mean"] == 1.0


def test_decompose_measures_lateness():
    gt = np.array([HOME] * 200 + [AWAY] * 200 + [HOME] * 200, np.int8)
    pred = np.r_[np.full(210, HOME), np.full(200, AWAY), np.full(190, HOME)].astype(np.int8)
    d = errors.decompose(gt, pred, np.ones(600, np.int8))
    assert d["team_change_offsets"].tolist() == [10, 10]


def test_decompose_counts_missed_short_spell_and_invented_dead():
    gt = np.array([HOME] * 100 + [AWAY] * 30 + [HOME] * 100, np.int8)        # 1.2 s steal by A
    pred = np.array([HOME] * 100 + [DEAD] * 30 + [HOME] * 100, np.int8)      # steal predicted as D
    t = errors.decompose(gt, pred, np.ones(230, np.int8))["table"].set_index("item")["count"]
    assert t["  short (< 2 s), missed"] == 1
    assert t["  mostly on GT alive (D invented)"] == 1


def test_oracles_fix_one_error_type_each():
    from src.eval import oracles
    half = np.ones(300, np.int8)
    gt = np.array([HOME] * 100 + [DEAD] * 50 + [AWAY] * 150, np.int8)
    late = np.array([HOME] * 120 + [DEAD] * 50 + [AWAY] * 130, np.int8)          # every edge 20 frames late
    assert oracles.perfect_timing(gt, late, half).tolist() == gt.tolist()
    no_dead = np.array([HOME] * 100 + [AWAY] * 200, np.int8)
    assert oracles.perfect_dead(gt, no_dead, half).tolist() == gt.tolist()
    wrong_team = np.array([HOME] * 100 + [DEAD] * 50 + [HOME] * 150, np.int8)
    assert oracles.perfect_team(gt, wrong_team, half).tolist() == gt.tolist()
    flicker = gt.copy(); flicker[200:210] = HOME                                   # 0.4 s spurious H
    assert oracles.no_spurious(gt, flicker, half).tolist() == gt.tolist()
    gt2 = gt.copy(); gt2[200:230] = HOME                                           # 1.2 s true steal
    assert oracles.short_spells(gt2, gt, half).tolist() == gt2.tolist()


def test_change_targets_mark_frames_around_team_changes():
    from src.models.transition import change_targets
    labels = np.array([HOME] * 50 + [AWAY] * 50 + [DEAD] * 50 + [HOME] * 50, np.int8)
    y = change_targets(labels, np.ones(200, np.int8), radius=3)
    assert y[47:54].tolist() == [1] * 7 and y[46] == 0 and y[54] == 0
    assert y[150] == 0                                   # through a 2 s dead stretch: not a change of team


def test_store_manifest_refuses_other_inputs(tmp_path):
    import pytest
    from src.eval.store import check_store
    ident = {"config": {"a": 1}, "features": "v1", "data": {"m": "x"}, "source": "s", "extra": {}}
    check_store(tmp_path / "st", ident)
    check_store(tmp_path / "st", dict(ident))                 # same inputs: reused
    with pytest.raises(RuntimeError, match="config"):
        check_store(tmp_path / "st", {**ident, "config": {"a": 2}})
