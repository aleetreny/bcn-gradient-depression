"""Knot-Anchored Physics & Ghost-Slot Filtering (EXP-C).

Solves the two core ATD tracking artifacts diagnosed in BAR-ATH and Metrica Sample:
1. Frozen duplicate ghost slots (lingering up to 45s at exact same (x, y) after tracklet death)
   and identity-switch velocity spikes (> 2.0 m in a single 40ms frame).
2. Piecewise-linear interpolated ball trajectories (74% of BAR-ATH in-play ball frames, 90% of Metrica Sample,
   and degrade_v5):
   - Identifies Real Ball Observations (curved bursts + true linear endpoints)
   - Identifies Verified Physical Touch Knots (filtering out the 28% mid-air 'phantom kicks' where no player is within 3.5m)
   - Computes forward/backward knot possession bridges (`knot_prev_team`, `knot_next_team`, `knot_bridge_team`,
     `knot_pass_same_team`, `knot_dist_diff_interp`) that are immune to opponents crossed by a synthetic straight line.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ball_interp import linear_mask
from src.data.common import AWAY, FPS, HOME, Match, runs
from src.features.v1 import _rolling, match_fingerprint

VERSION = 1


def _linear_runs(B: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vis = np.isfinite(B[:, :2]).all(1)
    lin = linear_mask(B, half)
    s, e, v = runs(np.where(vis, lin.astype(int) + 1, 0) * 10 + half)
    sel = (v // 10) == 2
    return s[sel], e[sel]


def filter_ghost_and_switch_slots(m: Match, min_frozen_frames: int = 20, max_step_m: float = 2.2) -> Match:
    """Mask out frozen duplicate ghost slots (zero motion for >= min_frozen_frames while >= 11 teammates visible)
    and recompute slot speeds safely without identity-switch spikes."""
    P64 = m.P.astype(np.float64)
    B64 = m.B.astype(np.float64)
    if not m.direction_normalized:
        L, W = float(m.pitch[0]), float(m.pitch[1])
        flip = m.attack_sign < 0
        P64[flip, :, 0] = L - P64[flip, :, 0]
        P64[flip, :, 1] = W - P64[flip, :, 1]
        B64[flip, 0] = L - B64[flip, 0]
        B64[flip, 1] = W - B64[flip, 1]
    T, S, _ = P64.shape
    half = np.asarray(m.half)

    # 1. Detect frozen runs per slot: |P[t] - P[t-1]| < 1e-4
    dx = np.abs(np.diff(P64[..., 0], axis=0, prepend=np.nan))
    dy = np.abs(np.diff(P64[..., 1], axis=0, prepend=np.nan))
    step = np.hypot(dx, dy)
    still_exact = (step < 1e-4) & np.isfinite(P64[..., 0])

    for s in range(S):
        if m.slot_is_gk[s]:
            continue  # Goalkeepers can stand still legitimately
        col = still_exact[:, s]
        if not np.any(col):
            continue
        edges = np.flatnonzero(col[1:] != col[:-1]) + 1
        starts = np.r_[0, edges]
        stops = np.r_[edges, T]
        vals = col[starts]
        for st, en, v in zip(starts, stops, vals):
            if v and (en - st) >= min_frozen_frames:
                # Mask frozen ghost segment (keep first 3 frames where tracklet stopped)
                P64[st + 3:en, s, :] = np.nan

    # 2. Safe velocity: mask single-frame jumps > max_step_m (55 m/s) caused by slot reassignment
    jump = step > max_step_m
    # Recompute clean 2-frame centered speed (channel 2)
    clean_spd = np.full((T, S), np.nan, dtype=np.float64)
    same_h = (half[2:] == half[:-2])[:, None]
    d2 = np.linalg.norm(P64[2:, :, :2] - P64[:-2, :, :2], axis=2)
    no_jump = (~jump[1:-1]) & (~jump[2:]) & same_h
    clean_spd[1:-1] = np.where(no_jump, d2 * (FPS / 2.0), np.nan)
    P64[..., 2] = np.clip(clean_spd, 0.0, 13.0)

    return replace(m, P=P64, B=B64, direction_normalized=True)


def _ffill_1d(vals: np.ndarray, mask: np.ndarray, half: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full(len(vals), default, dtype=np.float32)
    cur = default
    for i in range(len(vals)):
        if i == 0 or half[i] != half[i - 1]:
            cur = default
        if mask[i]:
            cur = vals[i]
        out[i] = cur
    return out


def _since_1d(mask: np.ndarray, half: np.ndarray, cap: int = 750) -> np.ndarray:
    out = np.full(len(mask), cap, dtype=np.float32)
    last = None
    for i in range(len(mask)):
        if i == 0 or half[i] != half[i - 1]:
            last = None
        if mask[i]:
            last = i
        if last is not None:
            out[i] = min(i - last, cap)
    return out


def knot_physics_features(m: Match) -> pd.DataFrame:
    """Compute knot-anchored possession features immune to linear ball interpolation and ghost slots."""
    nc = filter_ghost_and_switch_slots(m)
    P = nc.P[..., :2].astype(np.float64)
    B = nc.B[:, :2].astype(np.float64)
    half = np.asarray(nc.half)
    T = len(B)
    bvis = nc.ball_xy_visible
    blin = linear_mask(nc.B, half)

    # Identify linear segment endpoints (vertices)
    l_starts, l_ends = _linear_runs(nc.B, half)
    is_vertex = np.zeros(T, dtype=bool)
    if len(l_starts):
        is_vertex[l_starts] = True
        is_vertex[np.minimum(l_ends - 1, T - 1)] = True

    # Real ball evidence: visible AND (not inside interior of a linear run)
    real_ball = bvis & (~blin | is_vertex)

    # Clean team distances to ball at every frame
    d_all = np.linalg.norm(P - B[:, None, :], axis=2)
    sel_h = nc.slot_team == HOME
    sel_a = nc.slot_team == AWAY
    with np.errstate(all="ignore"):
        d_h = np.where(np.isfinite(d_all[:, sel_h]).any(1), np.nanmin(np.where(np.isfinite(d_all[:, sel_h]), d_all[:, sel_h], np.inf), 1), np.nan)
        d_a = np.where(np.isfinite(d_all[:, sel_a]).any(1), np.nanmin(np.where(np.isfinite(d_all[:, sel_a]), d_all[:, sel_a], np.inf), 1), np.nan)
    d_min = np.minimum(np.nan_to_num(d_h, nan=999.0), np.nan_to_num(d_a, nan=999.0))
    d_diff = (d_h - d_a).astype(np.float32)

    # Ball velocity & direction change
    vb_prev = np.full((T, 2), np.nan, np.float64)
    vb_next = np.full((T, 2), np.nan, np.float64)
    vb_prev[1:] = (B[1:] - B[:-1]) * FPS
    vb_next[:-1] = (B[1:] - B[:-1]) * FPS
    sp_prev = np.linalg.norm(vb_prev, axis=1)
    sp_next = np.linalg.norm(vb_next, axis=1)
    with np.errstate(all="ignore"):
        cos_ang = np.einsum("tk,tk->t", vb_prev, vb_next) / (sp_prev * sp_next + 1e-6)
        ang_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        dspeed = np.abs(sp_next - sp_prev)

    # Verified physical knot:
    # A real ball frame (curved or linear vertex) where a player is actually within 3.2m
    # (filters out mid-air phantom vertices where d_min > 3.2m!)
    inflection = (np.nan_to_num(ang_deg, nan=0.0) >= 12.0) | (np.nan_to_num(dspeed, nan=0.0) >= 2.0) | is_vertex | (~blin & (d_min < 2.0))
    verified_knot = bvis & inflection & (d_min < 3.2)
    strict_knot = bvis & inflection & (d_min < 2.0)

    # Signed team at verified knots: continuous soft confidence tanh((d_a - d_h) / 1.5) and hard sign
    knot_soft_team = np.tanh((np.nan_to_num(d_a, nan=50.0) - np.nan_to_num(d_h, nan=50.0)) / 1.5).astype(np.float32)
    knot_hard_team = np.where(d_h < d_a, 1.0, np.where(d_a < d_h, -1.0, 0.0)).astype(np.float32)

    prev_soft = _ffill_1d(knot_soft_team, verified_knot, half, default=0.0)
    next_soft = _ffill_1d(knot_soft_team[::-1], verified_knot[::-1], half[::-1], default=0.0)[::-1]
    prev_hard = _ffill_1d(knot_hard_team, verified_knot, half, default=0.0)
    next_hard = _ffill_1d(knot_hard_team[::-1], verified_knot[::-1], half[::-1], default=0.0)[::-1]

    prev_strict = _ffill_1d(knot_hard_team, strict_knot, half, default=0.0)
    next_strict = _ffill_1d(knot_hard_team[::-1], strict_knot[::-1], half[::-1], default=0.0)[::-1]

    since_knot = _since_1d(verified_knot, half, cap=750)
    until_knot = _since_1d(verified_knot[::-1], half[::-1], cap=750)[::-1]
    span = np.maximum(since_knot + until_knot, 1.0)
    alpha = until_knot / span  # 1 at prev_knot, 0 at next_knot

    # Knot-interpolated possession bridge:
    # If prev_hard == next_hard (pass completed within same team), bridge stays 100% on that team!
    # Otherwise, interpolates linearly by time between the two knots.
    knot_bridge_soft = alpha * prev_soft + (1.0 - alpha) * next_soft
    knot_bridge_hard = np.where(prev_hard == next_hard, prev_hard, alpha * prev_hard + (1.0 - alpha) * next_hard)
    knot_pass_same = np.where((prev_hard != 0) & (prev_hard == next_hard), prev_hard, 0.0).astype(np.float32)

    # Also anchor distance diff at real ball observations (ignoring interior linear points)
    real_d_diff = _ffill_1d(np.nan_to_num(d_diff, nan=0.0), real_ball & (d_min < 15.0), half, default=0.0)
    next_real_d_diff = _ffill_1d(np.nan_to_num(d_diff, nan=0.0)[::-1], (real_ball & (d_min < 15.0))[::-1], half[::-1], default=0.0)[::-1]

    # Phantom vertex indicator: linear vertex where NO player is within 4.0m
    phantom_vertex = (is_vertex & (d_min > 4.0)).astype(np.float32)

    f = {
        "knot_prev_soft": prev_soft,
        "knot_next_soft": next_soft,
        "knot_bridge_soft": knot_bridge_soft.astype(np.float32),
        "knot_bridge_hard": knot_bridge_hard.astype(np.float32),
        "knot_pass_same": knot_pass_same,
        "knot_prev_strict": prev_strict,
        "knot_next_strict": next_strict,
        "knot_since": since_knot,
        "knot_until": until_knot,
        "knot_span": np.minimum(span, 750.0).astype(np.float32),
        "real_d_diff_prev": real_d_diff,
        "real_d_diff_next": next_real_d_diff,
        "clean_d_diff": d_diff,
        "phantom_vertex": phantom_vertex,
    }
    X = pd.DataFrame(f)
    hs = pd.Series(half)
    extra = {}
    for col in ("knot_bridge_soft", "knot_bridge_hard", "knot_pass_same", "real_d_diff_prev"):
        for w_s in (0.8, 2.0, 4.0):
            extra[f"{col}_m{w_s}s"] = _rolling(X[col], hs, max(1, int(w_s * FPS))).astype(np.float32)
    return pd.concat([X, pd.DataFrame(extra)], axis=1)


compute_knot_physics_features = knot_physics_features


def cached_knot_physics(m: Match, key: str, cache_dir=None) -> pd.DataFrame:
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "local_data" / "cache" / f"knot_v{VERSION}"
    path = root / f"{key}_{match_fingerprint(m)}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    K = knot_physics_features(m)
    root.mkdir(parents=True, exist_ok=True)
    K.to_parquet(path)
    return K
