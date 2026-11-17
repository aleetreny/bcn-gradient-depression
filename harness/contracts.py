"""Runtime contracts between pipeline stages.

Every stage output passes one of these checks, so a bug surfaces at the boundary where it happens
instead of three stages later as a slightly lower spellF1.

    features(match)              -> DataFrame, one row per frame               check_features
    model.predict_proba(X)       -> (T, 3) float, columns H, A, D, rows sum 1  check_proba
    decoder(proba, half, params) -> (T,) int8 in {0 H, 1 A, 2 D}               check_labels
    predictor(train, test)       -> (T,) int8 in {0, 1, 2}                     check_labels

The swap helpers express the H<->A symmetry every component is expected to respect: relabelling
the teams must relabel the output and change nothing else (tests/test_harness.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOME, AWAY, DEAD = 0, 1, 2     # the codes of src/data/common.py, repeated so this layer runs on every branch


class ContractError(ValueError):
    pass


def _fail(where: str, msg: str):
    raise ContractError("{}: {}".format(where or "stage output", msg))


def check_labels(y, n: int, where: str = "") -> np.ndarray:
    y = np.asarray(y)
    if y.shape != (n,):
        _fail(where, "labels have shape {}, expected ({},)".format(y.shape, n))
    if y.dtype.kind not in "iu":
        _fail(where, "labels must be integer codes 0 H, 1 A, 2 D, got dtype {}".format(y.dtype))
    bad = ~np.isin(y, (HOME, AWAY, DEAD))
    if bad.any():
        _fail(where, "{} labels outside {{0, 1, 2}} (first at frame {})".format(int(bad.sum()), int(np.argmax(bad))))
    return y.astype(np.int8, copy=False)


def check_proba(p, n: int, where: str = "", atol: float = 1e-3) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (n, 3):
        _fail(where, "proba has shape {}, expected ({}, 3)".format(p.shape, n))
    if not np.isfinite(p).all():
        _fail(where, "proba has {} non-finite values".format(int((~np.isfinite(p)).sum())))
    if (p < -atol).any() or (p > 1 + atol).any():
        _fail(where, "proba outside [0, 1]")
    err = np.abs(p.sum(1) - 1)
    if (err > atol).any():
        _fail(where, "proba rows do not sum to 1 (worst {:.4f} at frame {})".format(err.max(), int(err.argmax())))
    return p


def check_features(X, n: int, where: str = "") -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        _fail(where, "features must be a DataFrame, got {}".format(type(X).__name__))
    if len(X) != n:
        _fail(where, "features have {} rows, expected {}".format(len(X), n))
    if X.columns.duplicated().any():
        _fail(where, "duplicated feature columns: {}".format(sorted(X.columns[X.columns.duplicated()])[:5]))
    if np.isinf(X.select_dtypes("number").to_numpy()).any():
        _fail(where, "features contain +-inf (use NaN for missing)")
    return X


def swap_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).copy()
    h, a = y == HOME, y == AWAY
    y[h], y[a] = AWAY, HOME
    return y


def swap_proba(p: np.ndarray) -> np.ndarray:
    return np.asarray(p)[:, [AWAY, HOME, DEAD]]
