"""Tests for src.features.v1: slot-order invariance, no label use, correct masks."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.features.v1 import build_features
from tests.test_degrade import _clean_match


def _with_gaps():
    m = _clean_match(T=1500)
    B = m.B.copy()
    B[200:400] = np.nan
    P = m.P.copy()
    P[100:160, 3] = np.nan
    return replace(m, B=B, P=P)


def test_invariant_to_slot_order():
    m = _with_gaps()
    perm = np.random.default_rng(0).permutation(m.P.shape[1])
    shuffled = replace(m, P=m.P[:, perm], slot_team=m.slot_team[perm], slot_is_gk=m.slot_is_gk[perm])
    a, b = build_features(m), build_features(shuffled)
    assert list(a.columns) == list(b.columns)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-5, atol=1e-5, equal_nan=True)


def test_labels_are_not_read():
    m = _with_gaps()
    other = replace(m, labels=np.zeros_like(m.labels))
    np.testing.assert_array_equal(build_features(m).to_numpy(), build_features(other).to_numpy())


def test_ball_masks():
    X = build_features(_with_gaps())
    assert (X.loc[200:399, "ball_visible"] == 0).all() and X.loc[200:399, "ball_x"].isna().all()
    assert X.loc[250, "ball_since_seen"] == 51 and X.loc[250, "ball_until_seen"] == 150
    assert np.isfinite(X.loc[250, "ball_last_x"]) and np.isfinite(X.loc[250, "ball_next_x"])
    assert (X.loc[0:199, "ball_visible"] == 1).all()


def test_cache_key_follows_content(tmp_path):
    from src.features.v1 import cached_features
    m = _with_gaps()
    a = cached_features(m, "x", cache_dir=tmp_path)
    B = m.B.copy()
    B[:, 0] += 10.0                                   # same key name, different data
    b = cached_features(replace(m, B=B), "x", cache_dir=tmp_path)
    assert not np.allclose(a["ball_x"].fillna(0), b["ball_x"].fillna(0))
    assert len(list(tmp_path.glob("x_*.parquet"))) == 2
