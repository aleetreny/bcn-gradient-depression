"""M3: transition head, P(a change of team within +/- radius frames of t).

    from src.models.transition import change_targets, change_features, TransitionGBM
    y = change_targets(labels, half)                 # 0/1 per frame (-1 where unlabelled)
    Xc = change_features(X, half)                    # v1 features + forward-minus-backward deltas
    p = TransitionGBM().fit([Xc1, ...], [y1, ...]).predict(Xc)

A change of team is a change between H and A spells inside a half, also through a dead stretch
shorter than 1 s (same definition as src/eval/errors.team_changes). The decoder uses p as a
time-dependent H<->A switch cost (src/decode/viterbi_hsmm.py, beta_ha).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.common import FPS
from src.eval.errors import team_changes

RADIUS = 6
DELTA_KEYS = ("touch_team_r1", "touch_team_r2", "touch_team_r3", "control_team", "nearest_team", "nearest_team_hold25",
              "dist_ball_diff", "dist_lastball_diff", "centroid_gap_x", "centroid_x_H")
DELTA_WINDOWS_S = (0.4, 1.0, 2.0)
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=16)


def change_targets(labels: np.ndarray, half: np.ndarray, radius: int = RADIUS) -> np.ndarray:
    y = np.where(labels >= 0, 0, -1).astype(np.int8)
    frames, _ = team_changes(labels, half)
    for f in frames:
        y[max(0, f - radius):f + radius + 1] = 1
    return y


def change_features(X: pd.DataFrame, half: np.ndarray) -> pd.DataFrame:
    """Add, for team-valued signals, mean over the next w frames minus mean over the previous w."""
    hs = pd.Series(half)
    extra = {}
    for key in DELTA_KEYS:
        s = X[key]
        for w_s in DELTA_WINDOWS_S:
            w = int(w_s * FPS)
            fwd = s.groupby(hs).transform(lambda x: x[::-1].rolling(w, min_periods=1).mean()[::-1].shift(-1))
            bwd = s.groupby(hs).transform(lambda x: x.rolling(w, min_periods=1).mean())
            d = (fwd - bwd).astype(np.float32)
            extra["chg_{}_{}s".format(key, w_s)] = d
            extra["chg_abs_{}_{}s".format(key, w_s)] = d.abs()
    return pd.concat([X, pd.DataFrame(extra)], axis=1)


@dataclass
class TransitionGBM:
    params: dict = field(default_factory=lambda: dict(PARAMS))
    subsample_every: int = 5
    seed: int = 0

    def fit(self, Xs, ys):
        import lightgbm as lgb
        rows_X, rows_y = [], []
        for X, y in zip(Xs, ys):
            keep = np.flatnonzero(y >= 0)[:: self.subsample_every]
            rows_X.append(X.iloc[keep])
            rows_y.append(y[keep])
        X, y = pd.concat(rows_X, ignore_index=True), np.concatenate(rows_y)
        self.features = list(X.columns)
        self.model_ = lgb.LGBMClassifier(**self.params, random_state=self.seed).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict_proba(X[self.features])[:, 1].astype(np.float32)

    def importance(self, top=20):
        imp = pd.Series(self.model_.booster_.feature_importance("gain"), index=self.features)
        return (imp / imp.sum()).sort_values(ascending=False).head(top)
