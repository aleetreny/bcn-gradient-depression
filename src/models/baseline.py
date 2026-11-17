"""The official baseline (starter/baseline.py) re-implemented on the common Match format.

    from src.models.baseline import predict
    labels = predict(match, window=50)          # common codes: 0 H, 1 A, 2 D

1. Each frame with a visible ball gets the team of the nearest visible slot (first slot wins ties,
   as in the official loop); frames without a ball, or without any visible slot, get nobody.
2. Centred majority vote over +/- window frames; ties go to H; D where nobody is seen in the window.

Distances are compared in float64 on positions scaled to metres, like the official code. The
vote runs over the whole array, ignoring halves, exactly as the official script does.
"""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, DEAD, HOME, Match

NOBODY = -1


def nearest_team(match: Match) -> np.ndarray:
    P = match.P[..., :2].astype(np.float64)
    B = match.B[:, :2].astype(np.float64)
    out = np.full(match.n_frames, NOBODY, np.int8)
    rows = np.flatnonzero(match.ball_xy_visible)
    d2 = ((P[rows] - B[rows, None, :]) ** 2).sum(-1)
    d2 = np.where(np.isnan(d2), np.inf, d2)
    best = np.argmin(d2, axis=1)
    seen = np.isfinite(d2[np.arange(len(rows)), best])
    out[rows[seen]] = match.slot_team[best[seen]]
    return out


def smooth(codes: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.where(codes == HOME, HOME, np.where(codes == AWAY, AWAY, DEAD)).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    out = np.full(len(codes), DEAD, np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], HOME, AWAY)
    return out


def predict(match: Match, window: int = 50) -> np.ndarray:
    return smooth(nearest_team(match), window)
