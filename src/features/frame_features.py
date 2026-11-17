"""Slot-invariant per-frame features, identical for every source (BAR-ATH ATD, Sportec, degraded Sportec).

    from src.features.frame_features import basic_features
    X = basic_features(match)            # DataFrame, one row per frame, NaN where undefined

Computed on the direction-normalised match (H attacks +x). Nothing here reads labels or slot
identities; the ball's third channel is not used (it is not simulated for Sportec, S2.1).
This is the v0 set used by the S2.1 adversarial check; S2.2 extends it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.common import AWAY, FPS, HOME, Match

CAP = 30 * FPS


def _since(mask):
    """Frames since mask was last True (CAP if never / beyond)."""
    out = np.full(len(mask), CAP, np.int32)
    last = -CAP
    for i, v in enumerate(mask):
        if v:
            last = i
        out[i] = min(i - last, CAP)
    return out


def _moving_mean(x, w):
    k = np.ones(w) / w
    valid = np.isfinite(x)
    num = np.convolve(np.where(valid, x, 0), k, "same")
    den = np.convolve(valid.astype(float), k, "same")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def basic_features(m: Match) -> pd.DataFrame:
    n = m.normalized()
    P, B = n.P, n.B
    vis = n.slot_visible
    bvis = n.ball_xy_visible
    T = n.n_frames
    f = {}
    f["ball_visible"] = bvis.astype(np.int8)
    f["ball_frames_since_seen"] = _since(bvis)
    f["ball_frames_until_seen"] = _since(bvis[::-1])[::-1]
    f["ball_x"] = np.where(bvis, B[:, 0], np.nan)
    f["ball_y"] = np.where(bvis, B[:, 1], np.nan)
    bsp = np.full(T, np.nan)
    ok = bvis[2:] & bvis[:-2]
    bsp[1:-1][ok] = np.linalg.norm(B[2:, :2][ok] - B[:-2, :2][ok], axis=1) * FPS / 2
    f["ball_speed"] = bsp

    d = np.linalg.norm(P[..., :2] - B[:, None, :2], axis=-1)
    speed = np.where(vis, P[..., 2], np.nan)
    with np.errstate(all="ignore"):
        for team, k in ((HOME, "H"), (AWAY, "A")):
            sel = n.slot_team == team
            dt = d[:, sel]
            f["dist_ball_" + k] = np.where(np.isfinite(dt).any(1), np.nanmin(np.where(np.isfinite(dt), dt, np.inf), 1), np.nan)
            f["n_within_3m_" + k] = np.where(bvis, np.nansum(dt < 3, 1), np.nan)
            f["n_within_10m_" + k] = np.where(bvis, np.nansum(dt < 10, 1), np.nan)
            x, y = np.where(vis[:, sel], P[:, sel, 0], np.nan), np.where(vis[:, sel], P[:, sel, 1], np.nan)
            f["n_visible_" + k] = vis[:, sel].sum(1)
            f["centroid_x_" + k] = np.nanmean(x, 1)
            f["centroid_y_" + k] = np.nanmean(y, 1)
            f["spread_x_" + k] = np.nanstd(x, 1)
            f["spread_y_" + k] = np.nanstd(y, 1)
            sp = speed[:, sel]
            f["speed_mean_" + k] = np.nanmean(sp, 1)
            f["speed_p90_" + k] = np.nanpercentile(sp, 90, axis=1)
            f["frac_slow_" + k] = np.nanmean(np.where(np.isfinite(sp), sp < 1.5, np.nan), 1)
        all_speed = np.nanmean(speed, 1)
    f["speed_mean_all_2s"] = _moving_mean(all_speed, 2 * FPS)
    f["n_visible_all"] = vis.sum(1)
    f["dist_ball_diff"] = f["dist_ball_H"] - f["dist_ball_A"]
    return pd.DataFrame(f)


def jitter_feature(m: Match) -> np.ndarray:
    """Median second difference of visible player positions per frame (smoothness; adversarial only)."""
    P = m.P[..., :2]
    d2 = np.full(P.shape[:2], np.nan)
    d2[1:-1] = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1)
    with np.errstate(all="ignore"):
        return np.nanmedian(d2, axis=1)
