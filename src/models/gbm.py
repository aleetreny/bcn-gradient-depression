"""Two-head LightGBM: P(alive) and P(H | alive), combined into P(H), P(A), P(D) per frame.

    from src.models.gbm import TwoHeadGBM
    model = TwoHeadGBM().fit([X1, X2, ...], [labels1, labels2, ...], groups=[0, 1, ...])
    proba = model.predict_proba(X)            # (T, 3): H, A, D

P(H) = P(alive) * P(H|alive), P(A) = P(alive) * (1 - P(H|alive)), P(D) = 1 - P(alive).
Optional isotonic calibration of each head with out-of-fold predictions grouped by match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.common import AWAY, DEAD, HOME

DEFAULT_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=16)


def _fit_lgb(X, y, params, seed):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(**params, random_state=seed)
    clf.fit(X, y)
    return clf


@dataclass
class TwoHeadGBM:
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    calibrate: bool = True
    subsample_every: int = 10
    seed: int = 0
    features: list = None

    def _stack(self, Xs, ys, groups, team_only):
        rows_X, rows_y, rows_g, rows_t = [], [], [], []
        for X, y, g, t in zip(Xs, ys, groups, team_only):
            keep = np.flatnonzero(y >= 0)[:: self.subsample_every]
            rows_X.append(X.iloc[keep])
            rows_y.append(y[keep])
            rows_g.append(np.full(len(keep), g))
            rows_t.append(np.full(len(keep), bool(t)))
        return (pd.concat(rows_X, ignore_index=True), np.concatenate(rows_y), np.concatenate(rows_g),
                np.concatenate(rows_t))

    def fit(self, Xs, ys, groups, team_only=None):
        """team_only[i] = True: source i trains only the team head (e.g. Metrica Sample, whose missing
        ball marks dead time and would teach the alive head a provider artifact)."""
        team_only = team_only or [False] * len(Xs)
        X, y, g, t = self._stack(Xs, ys, groups, team_only)
        self.features = list(X.columns)
        alive_rows = np.flatnonzero(~t)
        team_rows = np.flatnonzero(y != DEAD)
        Xa_, ya_, ga_ = X.iloc[alive_rows], (y[alive_rows] != DEAD).astype(int), g[alive_rows]
        alive = ya_
        team = (y[team_rows] == HOME).astype(int)
        self.alive_ = _fit_lgb(Xa_, alive, self.params, self.seed)
        self.team_ = _fit_lgb(X.iloc[team_rows], team, self.params, self.seed)
        self.iso_alive_ = self.iso_team_ = None
        if self.calibrate and len(np.unique(ga_)) >= 3:
            from sklearn.isotonic import IsotonicRegression
            oof_a = np.zeros(len(ya_))
            oof_t = np.zeros(len(team_rows))
            ug = np.unique(ga_)
            folds = np.array_split(np.random.default_rng(self.seed).permutation(ug), 3)
            extra = [x for x in np.unique(g) if x not in ug]  # team-only groups join the team-head folds
            folds_t = [np.r_[f, extra[i::3]] for i, f in enumerate(folds)]
            for fold, fold_t in zip(folds, folds_t):
                te = np.isin(ga_, fold)
                m_a = _fit_lgb(Xa_[~te], alive[~te], self.params, self.seed)
                oof_a[te] = m_a.predict_proba(Xa_[te])[:, 1]
                te_t = np.isin(g[team_rows], fold_t)
                m_t = _fit_lgb(X.iloc[team_rows[~te_t]], team[~te_t], self.params, self.seed)
                oof_t[te_t] = m_t.predict_proba(X.iloc[team_rows[te_t]])[:, 1]
            self.iso_alive_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof_a, alive)
            self.iso_team_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof_t, team)
        return self

    def heads(self, X):
        X = X[self.features]
        pa = self.alive_.predict_proba(X)[:, 1]
        ph = self.team_.predict_proba(X)[:, 1]
        if self.iso_alive_ is not None:
            pa, ph = self.iso_alive_.predict(pa), self.iso_team_.predict(ph)
        return pa, ph

    def predict_proba(self, X):
        pa, ph = self.heads(X)
        return np.stack([pa * ph, pa * (1 - ph), 1 - pa], axis=1)

    def importance(self, top=20):
        out = {}
        for name, m in (("alive", self.alive_), ("team", self.team_)):
            imp = pd.Series(m.booster_.feature_importance("gain"), index=self.features)
            out[name] = (imp / imp.sum()).sort_values(ascending=False).head(top)
        return out


def smooth_argmax(proba: np.ndarray, half: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average of the class probabilities (inside each half), then argmax -> 0 H, 1 A, 2 D."""
    if window > 0:
        out = np.empty_like(proba)
        k = np.ones(2 * window + 1) / (2 * window + 1)
        for h in np.unique(half):
            idx = np.flatnonzero(half == h)
            for c in range(proba.shape[1]):
                out[idx, c] = np.convolve(np.pad(proba[idx, c], window, mode="edge"), k, "valid")
        proba = out
    return np.argmax(proba, axis=1).astype(np.int8)


__all__ = ["TwoHeadGBM", "smooth_argmax", "HOME", "AWAY", "DEAD"]
