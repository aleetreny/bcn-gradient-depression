"""Wave 5 (`EXP-06`): 4th & 5th Model Families (`TwoHeadCatBoost` Oblivious Trees & `TwoHeadXGB`).

Implements:
1. `TwoHeadCatBoost`:
   - Two-head probability factorization identical to `TwoHeadGBM`:
     - `head_alive`: binary `DEAD` vs `LIVE` (`y != DEAD`) -> `p_alive = P(alive | x)`
     - `head_ha`: binary `HOME` vs `AWAY` on live frames (`y != DEAD`) -> `p_h_given_live = P(H | alive, x)`
     - Combined probabilities:
       `P(H) = p_alive * p_h_given_live`
       `P(A) = p_alive * (1 - p_h_given_live)`
       `P(D) = 1 - p_alive`
   - Uses `catboost.CatBoostClassifier` with Oblivious Trees (`SymmetricTree`, `depth=6`,
     `iterations=900`, `learning_rate=0.045`, `l2_leaf_reg=5.0`, `thread_count=4`,
     `verbose=False`, `allow_writing_files=False`).
   - Oblivious Trees enforce the exact same feature and split threshold across every node at depth `d`,
     acting as a regularized decision table that cannot memorize match-specific coordinate combinations
     and is resilient to tracking noise and ghost balls.

2. `TwoHeadXGB`:
   - Same two-head factorization (`head_alive` and `head_ha`) using `xgboost.XGBClassifier`
     (`max_depth=6`, `n_estimators=600`, `learning_rate=0.04`, `subsample=0.8`,
     `colsample_bytree=0.6`, `colsample_bylevel=0.7`, `reg_lambda=3.0`, `reg_alpha=0.5`,
     `min_child_weight=40`, `tree_method="hist"`, `nthread=4`, `verbosity=0`).

3. Exact `H <-> A` + 180-deg Pitch Rotation TTA Helper:
   - `predict_model_physv2_tta(model, X, half, pitch, X_mir=None)` using `mirror_v1ctl_physv2_df`
     and `swap_proba`.
"""
from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "py_pkgs") not in sys.path:
    sys.path.insert(0, str(ROOT / "py_pkgs"))
import torch  # loads libomp.dylib
ctypes.CDLL(str(Path(torch.__file__).resolve().parent / "lib" / "libomp.dylib"))
import catboost
import xgboost

import numpy as np
import pandas as pd

from src.data.common import DEAD, HOME
from src.features.symmetry import swap_proba
from src.features.v1ctl_physics_v2 import mirror_v1ctl_physv2_df

CATBOOST_DEFAULT_PARAMS = dict(
    depth=6,
    iterations=900,
    learning_rate=0.045,
    l2_leaf_reg=5.0,
    rsm=0.65,
    border_count=128,
    thread_count=4,
    verbose=False,
    allow_writing_files=False,
)

XGB_DEFAULT_PARAMS = dict(
    max_depth=6,
    n_estimators=600,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.6,
    colsample_bylevel=0.7,
    reg_lambda=3.0,
    reg_alpha=0.5,
    min_child_weight=40,
    max_bin=128,
    tree_method="hist",
    nthread=4,
    verbosity=0,
)


def _fit_catboost(X: np.ndarray, y: np.ndarray, params: dict, seed: int) -> catboost.CatBoostClassifier:
    p = dict(params)
    p.setdefault("random_seed", seed)
    clf = catboost.CatBoostClassifier(**p)
    clf.fit(X, y)
    return clf


def _fit_xgb(X: np.ndarray, y: np.ndarray, params: dict, seed: int) -> xgboost.XGBClassifier:
    p = dict(params)
    p.setdefault("random_state", seed)
    clf = xgboost.XGBClassifier(**p)
    clf.fit(X, y)
    return clf


def _stack_arrays(
    Xs: list[pd.DataFrame | np.ndarray],
    ys: list[np.ndarray],
    groups: list[int] | None,
    team_only: list[bool] | None,
    subsample_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    if groups is None:
        groups = list(range(len(Xs)))
    if team_only is None:
        team_only = [False] * len(Xs)

    features: list[str] | None = None
    rows_X, rows_y, rows_g, rows_t = [], [], [], []
    step = max(int(subsample_every), 1)
    for X, y, g, t in zip(Xs, ys, groups, team_only):
        keep = np.flatnonzero(y >= 0)[::step]
        if isinstance(X, pd.DataFrame):
            if features is None:
                features = list(X.columns)
            x_arr = X.iloc[keep].to_numpy(dtype=np.float32, copy=False)
        else:
            x_arr = np.asarray(X[keep], dtype=np.float32)
        rows_X.append(x_arr)
        rows_y.append(np.asarray(y[keep], dtype=np.int8))
        rows_g.append(np.full(len(keep), g, dtype=np.int32))
        rows_t.append(np.full(len(keep), bool(t), dtype=bool))

    return (
        np.concatenate(rows_X, axis=0),
        np.concatenate(rows_y, axis=0),
        np.concatenate(rows_g, axis=0),
        np.concatenate(rows_t, axis=0),
        features,
    )


def _to_float32_matrix(X: pd.DataFrame | np.ndarray, features: list[str] | None) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        if features is not None:
            X = X[features]
        return X.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


@dataclass
class TwoHeadCatBoost:
    """Two-head CatBoost Oblivious Tree classifier (`head_alive` & `head_ha`)."""

    params: dict = field(default_factory=lambda: dict(CATBOOST_DEFAULT_PARAMS))
    calibrate: bool = False
    subsample_every: int = 20
    seed: int = 0
    features: list[str] | None = None

    def fit(self, Xs, ys, groups=None, team_only=None):
        X, y, g, t, feats = _stack_arrays(Xs, ys, groups, team_only, self.subsample_every)
        if feats is not None:
            self.features = feats

        alive_rows = np.flatnonzero(~t)
        team_rows = np.flatnonzero(y != DEAD)
        Xa_ = X[alive_rows]
        ya_ = (y[alive_rows] != DEAD).astype(np.int32)
        ga_ = g[alive_rows]
        Xh_ = X[team_rows]
        yh_ = (y[team_rows] == HOME).astype(np.int32)

        self.head_alive = _fit_catboost(Xa_, ya_, self.params, self.seed)
        self.head_ha = _fit_catboost(Xh_, yh_, self.params, self.seed + 100)
        # Compatibility aliases with TwoHeadGBM
        self.alive_ = self.head_alive
        self.team_ = self.head_ha

        self.iso_alive_ = self.iso_team_ = None
        if self.calibrate and len(np.unique(ga_)) >= 3:
            from sklearn.isotonic import IsotonicRegression

            oof_a = np.zeros(len(ya_), dtype=np.float64)
            oof_t = np.zeros(len(team_rows), dtype=np.float64)
            ug = np.unique(ga_)
            folds = np.array_split(np.random.default_rng(self.seed).permutation(ug), 3)
            extra = [x for x in np.unique(g) if x not in ug]
            folds_t = [np.r_[f, extra[i::3]] for i, f in enumerate(folds)]
            for fold, fold_t in zip(folds, folds_t):
                te = np.isin(ga_, fold)
                m_a = _fit_catboost(Xa_[~te], ya_[~te], self.params, self.seed)
                oof_a[te] = m_a.predict_proba(Xa_[te])[:, 1]
                te_t = np.isin(g[team_rows], fold_t)
                m_t = _fit_catboost(Xh_[~te_t], yh_[~te_t], self.params, self.seed + 100)
                oof_t[te_t] = m_t.predict_proba(Xh_[te_t])[:, 1]
            self.iso_alive_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof_a, ya_)
            self.iso_team_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof_t, yh_)
        return self

    def heads(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xm = _to_float32_matrix(X, self.features)
        pa = self.head_alive.predict_proba(Xm)[:, 1]
        ph = self.head_ha.predict_proba(Xm)[:, 1]
        if self.iso_alive_ is not None:
            pa = self.iso_alive_.predict(pa)
            ph = self.iso_team_.predict(ph)
        return pa, ph

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        pa, ph = self.heads(X)
        return np.stack([pa * ph, pa * (1.0 - ph), 1.0 - pa], axis=1).astype(np.float32)

    def importance(self, top: int = 20) -> dict[str, pd.Series]:
        names = self.features or [f"f{i}" for i in range(int(self.head_alive.tree_count_))]
        out = {}
        for name, m in (("alive", self.head_alive), ("ha", self.head_ha)):
            vals = np.asarray(m.get_feature_importance(), dtype=np.float64)
            imp = pd.Series(vals, index=names[: len(vals)])
            s = float(imp.sum())
            out[name] = (imp / max(s, 1e-12)).sort_values(ascending=False).head(top)
        return out


@dataclass
class TwoHeadXGB:
    """Two-head XGBoost histogram tree classifier (`head_alive` & `head_ha`)."""

    params: dict = field(default_factory=lambda: dict(XGB_DEFAULT_PARAMS))
    calibrate: bool = False
    subsample_every: int = 20
    seed: int = 0
    features: list[str] | None = None

    def fit(self, Xs, ys, groups=None, team_only=None):
        X, y, g, t, feats = _stack_arrays(Xs, ys, groups, team_only, self.subsample_every)
        if feats is not None:
            self.features = feats

        alive_rows = np.flatnonzero(~t)
        team_rows = np.flatnonzero(y != DEAD)
        Xa_ = X[alive_rows]
        ya_ = (y[alive_rows] != DEAD).astype(np.int32)
        ga_ = g[alive_rows]
        Xh_ = X[team_rows]
        yh_ = (y[team_rows] == HOME).astype(np.int32)

        self.head_alive = _fit_xgb(Xa_, ya_, self.params, self.seed)
        self.head_ha = _fit_xgb(Xh_, yh_, self.params, self.seed + 100)
        # Compatibility aliases with TwoHeadGBM
        self.alive_ = self.head_alive
        self.team_ = self.head_ha

        self.iso_alive_ = self.iso_team_ = None
        if self.calibrate and len(np.unique(ga_)) >= 3:
            from sklearn.isotonic import IsotonicRegression

            oof_a = np.zeros(len(ya_), dtype=np.float64)
            oof_t = np.zeros(len(team_rows), dtype=np.float64)
            ug = np.unique(ga_)
            folds = np.array_split(np.random.default_rng(self.seed).permutation(ug), 3)
            extra = [x for x in np.unique(g) if x not in ug]
            folds_t = [np.r_[f, extra[i::3]] for i, f in enumerate(folds)]
            for fold, fold_t in zip(folds, folds_t):
                te = np.isin(ga_, fold)
                m_a = _fit_xgb(Xa_[~te], ya_[~te], self.params, self.seed)
                oof_a[te] = m_a.predict_proba(Xa_[te])[:, 1]
                te_t = np.isin(g[team_rows], fold_t)
                m_t = _fit_xgb(Xh_[~te_t], yh_[~te_t], self.params, self.seed + 100)
                oof_t[te_t] = m_t.predict_proba(Xh_[te_t])[:, 1]
            self.iso_alive_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof_a, ya_)
            self.iso_team_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof_t, yh_)
        return self

    def heads(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xm = _to_float32_matrix(X, self.features)
        pa = self.head_alive.predict_proba(Xm)[:, 1]
        ph = self.head_ha.predict_proba(Xm)[:, 1]
        if self.iso_alive_ is not None:
            pa = self.iso_alive_.predict(pa)
            ph = self.iso_team_.predict(ph)
        return pa, ph

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        pa, ph = self.heads(X)
        return np.stack([pa * ph, pa * (1.0 - ph), 1.0 - pa], axis=1).astype(np.float32)

    def importance(self, top: int = 20) -> dict[str, pd.Series]:
        names = self.features or [f"f{i}" for i in range(len(self.head_alive.feature_importances_))]
        out = {}
        for name, m in (("alive", self.head_alive), ("ha", self.head_ha)):
            vals = np.asarray(m.feature_importances_, dtype=np.float64)
            imp = pd.Series(vals, index=names[: len(vals)])
            s = float(imp.sum())
            out[name] = (imp / max(s, 1e-12)).sort_values(ascending=False).head(top)
        return out


def predict_model_physv2_tta(
    model: TwoHeadCatBoost | TwoHeadXGB,
    X: pd.DataFrame,
    half: np.ndarray,
    pitch: tuple[float, float] = (105.0, 68.0),
    X_mir: pd.DataFrame | None = None,
) -> np.ndarray:
    """Predict exact H <-> A + 180-deg pitch rotation TTA probabilities on `v1ctl_physics_v2` features."""
    p_orig = model.predict_proba(X)
    if X_mir is None:
        X_mir = mirror_v1ctl_physv2_df(X, half, pitch)
    p_mir = swap_proba(model.predict_proba(X_mir))
    out = 0.5 * p_orig + 0.5 * p_mir
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


predict_catboost_physv2_tta = predict_model_physv2_tta
predict_xgb_physv2_tta = predict_model_physv2_tta

__all__ = [
    "CATBOOST_DEFAULT_PARAMS",
    "XGB_DEFAULT_PARAMS",
    "TwoHeadCatBoost",
    "TwoHeadXGB",
    "predict_model_physv2_tta",
    "predict_catboost_physv2_tta",
    "predict_xgb_physv2_tta",
]
