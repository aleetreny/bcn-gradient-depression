"""EXP-C+: Unified Physics & Continuity Feature Block (`v1ctl_plus`).

Builds an additional equivariant feature block on top of cached `v1ctl` combining:
1. 26 Knot-Anchored Ball Physics features (`compute_knot_physics_features(m)`) +
   9 `ghost_incoherence` features (slow ball `<2 m/s` with no player `<3m`, or
   kick/acceleration spike with no player `<4m`, smoothed over 0.5s and 1.5s).
2. Bernardo's Track-Break-Free Multi-Scale Least-Squares Velocities (`lsq_velocity`
   at half-widths `h in (2, 6, 12)` frames = `0.16s, 0.48s, 0.96s`, masking out any
   window with missing frames or single-frame step `> 1.5m` = `37.5 m/s` identity switch):
   per scale `h`, `home_speed_h`, `away_speed_h`, `diff_speed_h`, `home_vx_h`, `away_vx_h`,
   `diff_vx_h`, `home_slow_h`, `away_slow_h`, `diff_slow_h`, `all_speed_h`, `all_slow_h`,
   and 1s/3s rolling means.
3. Bernardo's Gap-Edge Continuity features: `d_home2`, `d_away2`, `margin` (`d_A1 - d_H1`),
   `margin2`, `margin_last`, `margin_next`, `margin_interp` (linear interpolation of `margin`
   between `last` and `next` across ball gaps), `margin_cont`, and clipped rolling means.
4. Exact `mirror_plus_df(df_plus, half, pitch)` and `mirror_v1ctl_plus_df(df, half, pitch)`
   with `< 1e-5` (bit-for-bit 0.0) discrepancy against `compute_plus_features(swap_match(m))`.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.data.common import AWAY, FPS, HOME, Match
from src.features.control import cached_control
from src.features.knot_physics import compute_knot_physics_features, filter_ghost_and_switch_slots
from src.features.symmetry import mirror_v1ctl_df
from src.features.v1 import _rolling, cached_features, match_fingerprint
from src.models.transition import change_features

VERSION = 1
SPEED_CAP = 12.0
BREAK_STEP_M = 1.5
LSQ_HALF_WIDTHS = (2, 6, 12)
TEAM_ROLL_S = (1.0, 3.0)
MARGIN_ROLL_S = (0.5, 1.0, 2.0, 4.0)


def _roll_nanmean_1d(x: np.ndarray, half_width: int) -> np.ndarray:
    """Centred mean over +/- half_width frames ignoring NaN; NaN where the window has no data."""
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half_width)
    hi = np.minimum(len(x), idx + half_width + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def roll_nanmean_by_half(x: np.ndarray, half: np.ndarray, half_width: int) -> np.ndarray:
    """Apply Bernardo's centred roll_nanmean strictly within each half segment."""
    out = np.full(len(x), np.nan, dtype=np.float32)
    edges = np.flatnonzero(half[1:] != half[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(x)]
    for a, b in zip(starts, stops):
        out[a:b] = _roll_nanmean_1d(x[a:b], half_width).astype(np.float32)
    return out


def lsq_velocity(block: np.ndarray, half: np.ndarray, h: int, cap: float = SPEED_CAP, break_m: float = BREAK_STEP_M) -> np.ndarray:
    """(T, N, 2) per-slot velocity (m/s): least-squares slope of position over frames t-h..t+h.

    NaN unless every frame of the window is visible, inside the same half, and no single-frame
    step inside it exceeds break_m (1.5 m = 37.5 m/s).
    """
    T, N = block.shape[:2]
    v = np.full((T, N, 2), np.nan, dtype=np.float64)
    if T <= 2 * h:
        return v
    vis = np.isfinite(block[..., 0]) & np.isfinite(block[..., 1])
    step = np.hypot(block[1:, :, 0] - block[:-1, :, 0], block[1:, :, 1] - block[:-1, :, 1])
    same_step_half = (half[1:] == half[:-1])[:, None]
    ok_step = (np.nan_to_num(step, nan=0.0) <= break_m) & same_step_half

    cv = np.concatenate((np.zeros((1, N), np.int64), np.cumsum(vis, axis=0)))
    full = (cv[2 * h + 1:] - cv[:T - 2 * h]) == (2 * h + 1)

    cb = np.concatenate((np.zeros((1, N), np.int64), np.cumsum(~ok_step, axis=0)))
    clean = (cb[2 * h:] - cb[:T - 2 * h]) == 0

    k = np.arange(-h, h + 1, dtype=np.float64)
    acc = np.zeros((T - 2 * h, N, 2), dtype=np.float64)
    filled = np.nan_to_num(block, nan=0.0)
    for j, kk in enumerate(k):
        if kk != 0.0:
            acc += kk * filled[j:T - 2 * h + j]
    slope = acc / np.sum(k * k) * FPS
    slope[~(full & clean)] = np.nan
    speed = np.hypot(slope[..., 0], slope[..., 1])
    slope[speed > cap] = np.nan
    v[h:T - h] = slope
    return v


def _gap_edges_by_half(visible: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (last, nxt) visible frame indices within each half (-1 if none before, T if none after)."""
    T = len(visible)
    last = np.full(T, -1, dtype=np.int64)
    nxt = np.full(T, T, dtype=np.int64)
    edges = np.flatnonzero(half[1:] != half[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, T]
    for a, b in zip(starts, stops):
        vis_seg = visible[a:b]
        idx_seg = np.arange(a, b, dtype=np.int64)
        last[a:b] = np.maximum.accumulate(np.where(vis_seg, idx_seg, -1))
        nxt[a:b] = np.minimum.accumulate(np.where(vis_seg, idx_seg, T)[::-1])[::-1]
    return last, nxt


def compute_plus_features(m: Match) -> pd.DataFrame:
    """Compute the complete EXP-C+ feature block (Knot Physics + Ghost Incoherence + LSQ Kinematics + Gap-Edge Continuity)."""
    # 1. Start with the 26 Knot-Anchored Physics features (uses float64 normalization & ghost-slot filtering)
    K = compute_knot_physics_features(m)

    # Obtain float64 normalized & ghost-filtered Match
    nc = filter_ghost_and_switch_slots(m)
    P = nc.P[..., :2].astype(np.float64)
    B = nc.B[:, :2].astype(np.float64)
    half = np.asarray(nc.half)
    hs = pd.Series(half)
    T = len(B)
    bvis = nc.ball_xy_visible

    # 2. Ghost-Ball Incoherence feature block
    d_all = np.linalg.norm(P - B[:, None, :], axis=2)
    sel_h = nc.slot_team == HOME
    sel_a = nc.slot_team == AWAY
    with np.errstate(all="ignore"):
        d_h = np.where(np.isfinite(d_all[:, sel_h]).any(1), np.nanmin(np.where(np.isfinite(d_all[:, sel_h]), d_all[:, sel_h], np.inf), 1), np.nan)
        d_a = np.where(np.isfinite(d_all[:, sel_a]).any(1), np.nanmin(np.where(np.isfinite(d_all[:, sel_a]), d_all[:, sel_a], np.inf), 1), np.nan)
    d_min = np.minimum(np.nan_to_num(d_h, nan=999.0), np.nan_to_num(d_a, nan=999.0))

    vb_prev = np.full((T, 2), np.nan, np.float64)
    vb_next = np.full((T, 2), np.nan, np.float64)
    same_prev = np.r_[False, half[1:] == half[:-1]]
    same_next = np.r_[half[1:] == half[:-1], False]
    vb_prev[1:] = np.where(same_prev[1:, None], (B[1:] - B[:-1]) * FPS, np.nan)
    vb_next[:-1] = np.where(same_next[:-1, None], (B[1:] - B[:-1]) * FPS, np.nan)
    sp_prev = np.linalg.norm(vb_prev, axis=1)
    sp_next = np.linalg.norm(vb_next, axis=1)
    sp_mid = 0.5 * (np.nan_to_num(sp_prev, nan=0.0) + np.nan_to_num(sp_next, nan=0.0))
    with np.errstate(all="ignore"):
        cos_ang = np.einsum("tk,tk->t", vb_prev, vb_next) / (sp_prev * sp_next + 1e-6)
        ang_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        dspeed = np.abs(sp_next - sp_prev)

    # (a) Slow ball (< 2 m/s) with no player within 3.0m
    ghost_slow_alone = (bvis & (sp_mid < 2.0) & (d_min > 3.0)).astype(np.float32)
    # (b) Kick / acceleration / direction spike with no player within 4.0m
    has_spike = (np.nan_to_num(dspeed, nan=0.0) > 3.0) | ((np.nan_to_num(ang_deg, nan=0.0) > 20.0) & (sp_mid > 1.0))
    ghost_accel_alone = (bvis & has_spike & (d_min > 4.0)).astype(np.float32)
    ghost_incoherence = np.maximum(ghost_slow_alone, ghost_accel_alone).astype(np.float32)

    f = {
        "ghost_slow_alone": ghost_slow_alone,
        "ghost_accel_alone": ghost_accel_alone,
        "ghost_incoherence": ghost_incoherence,
    }
    for gcol in ("ghost_slow_alone", "ghost_accel_alone", "ghost_incoherence"):
        s_g = pd.Series(f[gcol])
        for w_s in (0.5, 1.5):
            f[f"{gcol}_m{w_s}s"] = _rolling(s_g, hs, max(1, int(round(w_s * FPS)))).to_numpy(np.float32)

    # 3. Bernardo's Track-Break-Free Multi-Scale Least-Squares Velocities
    # Use raw normalized float64 positions so lsq_velocity applies its exact break_m=1.5m & full-window visibility rules
    P_raw = m.P[..., :2].astype(np.float64)
    if not m.direction_normalized:
        L, W = float(m.pitch[0]), float(m.pitch[1])
        flip = m.attack_sign < 0
        P_raw[flip, :, 0] = L - P_raw[flip, :, 0]
        P_raw[flip, :, 1] = W - P_raw[flip, :, 1]

    block_h = P_raw[:, sel_h, :]
    block_a = P_raw[:, sel_a, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for h in LSQ_HALF_WIDTHS:
            vel_h = lsq_velocity(block_h, half, h)
            vel_a = lsq_velocity(block_a, half, h)
            sp_h = np.hypot(vel_h[..., 0], vel_h[..., 1])
            sp_a = np.hypot(vel_a[..., 0], vel_a[..., 1])
            # Own-attacking-goal x-velocity (+vx for HOME, -vx for AWAY in normalized coordinates)
            vx_h = vel_h[..., 0]
            vx_a = -vel_a[..., 0]
            slow_h = np.where(np.isnan(sp_h), np.nan, (sp_h < 1.0).astype(np.float64))
            slow_a = np.where(np.isnan(sp_a), np.nan, (sp_a < 1.0).astype(np.float64))

            hm_sp = np.nanmean(sp_h, axis=1)
            aw_sp = np.nanmean(sp_a, axis=1)
            hm_vx = np.nanmean(vx_h, axis=1)
            aw_vx = np.nanmean(vx_a, axis=1)
            hm_sl = np.nanmean(slow_h, axis=1)
            aw_sl = np.nanmean(slow_a, axis=1)

            df_sp = hm_sp - aw_sp
            df_vx = hm_vx - aw_vx
            df_sl = hm_sl - aw_sl
            al_sp = 0.5 * (hm_sp + aw_sp)
            al_sl = 0.5 * (hm_sl + aw_sl)

            f[f"home_speed_h{h}"] = hm_sp.astype(np.float32)
            f[f"away_speed_h{h}"] = aw_sp.astype(np.float32)
            f[f"diff_speed_h{h}"] = df_sp.astype(np.float32)
            f[f"home_vx_h{h}"] = hm_vx.astype(np.float32)
            f[f"away_vx_h{h}"] = aw_vx.astype(np.float32)
            f[f"diff_vx_h{h}"] = df_vx.astype(np.float32)
            f[f"home_slow_h{h}"] = hm_sl.astype(np.float32)
            f[f"away_slow_h{h}"] = aw_sl.astype(np.float32)
            f[f"diff_slow_h{h}"] = df_sl.astype(np.float32)
            f[f"all_speed_h{h}"] = al_sp.astype(np.float32)
            f[f"all_slow_h{h}"] = al_sl.astype(np.float32)

            for s in TEAM_ROLL_S:
                w = int(round(s * FPS))
                f[f"diff_speed_h{h}_r{s:g}"] = roll_nanmean_by_half(df_sp, half, w)
                f[f"diff_vx_h{h}_r{s:g}"] = roll_nanmean_by_half(df_vx, half, w)
                f[f"all_speed_h{h}_r{s:g}"] = roll_nanmean_by_half(al_sp, half, w)
                f[f"all_slow_h{h}_r{s:g}"] = roll_nanmean_by_half(al_sl, half, w)

        # 4. Bernardo's Gap-Edge Continuity & 2nd-Defender Margin features
        d_sort_h = np.sort(d_all[:, sel_h], axis=1)
        d_sort_a = np.sort(d_all[:, sel_a], axis=1)
        dh1, dh2 = d_sort_h[:, 0], d_sort_h[:, 1]
        da1, da2 = d_sort_a[:, 0], d_sort_a[:, 1]
        margin = da1 - dh1
        margin2 = (da1 + da2) - (dh1 + dh2)

        last_idx, nxt_idx = _gap_edges_by_half(bvis, half)
        has_last = last_idx >= 0
        has_nxt = nxt_idx < T
        before = np.where(has_last, margin[np.clip(last_idx, 0, T - 1)], np.nan)
        after = np.where(has_nxt, margin[np.clip(nxt_idx, 0, T - 1)], np.nan)

        idx = np.arange(T, dtype=np.float64)
        span = np.maximum((nxt_idx - last_idx).astype(np.float64), 1.0)
        w_next = np.clip((idx - last_idx) / span, 0.0, 1.0)
        interp_both = (1.0 - w_next) * before + w_next * after
        interp_val = np.where(has_last & has_nxt, interp_both, np.where(has_last, before, after))

        f["d_home2"] = dh2.astype(np.float32)
        f["d_away2"] = da2.astype(np.float32)
        f["margin"] = margin.astype(np.float32)
        f["margin2"] = margin2.astype(np.float32)
        f["margin_last"] = np.where(bvis, np.nan, before).astype(np.float32)
        f["margin_next"] = np.where(bvis, np.nan, after).astype(np.float32)
        f["margin_interp"] = np.where(bvis, np.nan, interp_val).astype(np.float32)
        f["margin_cont"] = np.where(bvis, margin, interp_val).astype(np.float32)

        sgn = np.sign(margin)
        clip_m = np.clip(margin, -15.0, 15.0)
        for s in MARGIN_ROLL_S:
            w = int(round(s * FPS))
            f[f"margin_r{s:g}"] = roll_nanmean_by_half(clip_m, half, w)
            f[f"margin_sign_r{s:g}"] = roll_nanmean_by_half(sgn, half, w)

    return pd.concat([K, pd.DataFrame(f)], axis=1)


def mirror_plus_df(df_plus: pd.DataFrame, half: np.ndarray | None = None, pitch: tuple[float, float] = (105.0, 68.0)) -> pd.DataFrame:
    """Exact algebraic H <-> A + 180-deg pitch rotation transformation of `compute_plus_features`."""
    M = df_plus.copy()

    # 1. Direct H <-> A swap pairs
    swap_pairs = [("d_home2", "d_away2")]
    for h in LSQ_HALF_WIDTHS:
        swap_pairs.extend([
            (f"home_speed_h{h}", f"away_speed_h{h}"),
            (f"home_vx_h{h}", f"away_vx_h{h}"),
            (f"home_slow_h{h}", f"away_slow_h{h}"),
        ])
    for ch, ca in swap_pairs:
        if ch in M.columns and ca in M.columns:
            M[ch] = df_plus[ca]
            M[ca] = df_plus[ch]

    # 2. Pure sign negations (team-signed features)
    neg_cols = [
        "knot_prev_soft",
        "knot_next_soft",
        "knot_bridge_soft",
        "knot_bridge_hard",
        "knot_pass_same",
        "knot_prev_strict",
        "knot_next_strict",
        "real_d_diff_prev",
        "real_d_diff_next",
        "clean_d_diff",
        "margin",
        "margin2",
        "margin_last",
        "margin_next",
        "margin_interp",
        "margin_cont",
    ]
    for col in ("knot_bridge_soft", "knot_bridge_hard", "knot_pass_same", "real_d_diff_prev"):
        for w_s in (0.8, 2.0, 4.0):
            neg_cols.append(f"{col}_m{w_s}s")
    for h in LSQ_HALF_WIDTHS:
        neg_cols.extend([
            f"diff_speed_h{h}",
            f"diff_vx_h{h}",
            f"diff_slow_h{h}",
        ])
        for s in TEAM_ROLL_S:
            neg_cols.extend([
                f"diff_speed_h{h}_r{s:g}",
                f"diff_vx_h{h}_r{s:g}",
            ])
    for s in MARGIN_ROLL_S:
        neg_cols.extend([
            f"margin_r{s:g}",
            f"margin_sign_r{s:g}",
        ])

    for col in neg_cols:
        if col in M.columns:
            M[col] = -df_plus[col]

    return M


def mirror_v1ctl_plus_df(X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float] = (105.0, 68.0)) -> pd.DataFrame:
    """Algebraically transform a combined `v1ctl_plus` DataFrame under H <-> A + 180-deg pitch rotation."""
    M = mirror_v1ctl_df(X, half, pitch)
    return mirror_plus_df(M, half, pitch)


def cached_plus(m: Match, key: str, cache_dir=None) -> pd.DataFrame:
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "local_data" / "cache" / f"plus_v{VERSION}"
    path = root / f"{key}_{match_fingerprint(m)}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    P = compute_plus_features(m)
    root.mkdir(parents=True, exist_ok=True)
    P.to_parquet(path)
    return P


def build_v1ctl_plus(m: Match, key: str) -> pd.DataFrame:
    """Build full `v1ctl_plus` feature DataFrame (`v1ctl` + `plus`) using disk caches."""
    X_base = change_features(cached_features(m, key), m.half)
    X_ctl = cached_control(m, key)
    X_plus = cached_plus(m, key)
    return pd.concat([X_base, X_ctl, X_plus], axis=1)
