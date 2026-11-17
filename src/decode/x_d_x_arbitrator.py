"""EXP-03: Surgical `X -> D -> X` Arbitrator + Causal Football Restart/Out-of-Bounds Grammar.

Implements `apply_xdx_arbitrator(pred, proba, half, X, p_change=None, p_trans_4d=None, cfg=XDXConfig())`,
a strictly H <-> A symmetric post-Viterbi causal arbitrator with 4 targeted operations:

1. Interior False-D Absorption (`X -> D -> X` -> `X`):
   For every `D` segment `[a, b)` bounded on both sides within the same half by the same live team `X in {H, A}`:
   if `b - a <= cfg.max_absorb_len`, the ball stays strictly inside the pitch interior (`|bx_c| < cfg.interior_max_abs_x`
   and `|by_c| < cfg.interior_max_abs_y`), and `mean(p_D[a:b]) < cfg.tau_absorb` (with no strong `Live->D` + `D->Live`
   4-directional boundary pair or high player stillness), absorb `[a, b)` into `X`.

2. Touchline / Goal-Line Same-Team Split (`X -> X` -> `X -> D -> X`):
   For long live spells `[a, b)` of team `X`: if an internal contiguous window `[u, v)` of length `>= cfg.min_split_dead`
   has out-of-bounds geometry (`|by_c| > cfg.split_oob_y` or `|bx_c| > cfg.split_oob_x` or `ball_outside_pitch > 0`)
   and elevated `p_D` (assisted by `p_Live_to_D` at `u` and `p_D_to_Live` at `v`) exceeding `cfg.tau_split_d`,
   insert `D` on `[u, v)`.

3. Out-of-Bounds Tail Trimming (`Live <-> D` boundary refinement):
   At every `Live -> D` and `D -> Live` boundary, if the adjacent `Live` frames have `ball_visible == 1` and
   `|by_c| > cfg.trim_oob_y` or `|bx_c| > cfg.trim_oob_x` (ball clearly out of bounds) with `p_D > cfg.tau_oob`,
   shift the `D` boundary to absorb those out-of-bounds frames into `D`.

4. Restart Taker Attribution (`D -> H/A`):
   At `D -> Live` transition frame `b` (start of a live spell after `D` of length `>= cfg.min_dead_for_restart`),
   check the restart window `[b - 10, b + 5]`: if one team `T_restart` is uniquely within `cfg.restart_near_m` (`1.8m`)
   of the stationary ball while the other team is `> cfg.restart_far_m` (`3.5m`) away, and the immediately following
   live spell is short/ambiguous (`mean(|p_H - p_A|) < cfg.tau_restart_ambig`), assign the restart spell to `T_restart`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings
import numpy as np
import pandas as pd

from src.data.common import AWAY, DEAD, HOME, runs


def rle_segments(labels: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-length encode (labels, half) so no segment crosses a half boundary."""
    key = np.asarray(labels, dtype=np.int64) * 16 + np.asarray(half, dtype=np.int64)
    s, e, _ = runs(key)
    return s, e, np.asarray(labels)[s]


@dataclass
class XDXConfig:
    # 1. Interior False-D Absorption (X -> D -> X -> X)
    enable_absorb: bool = True
    max_absorb_len: int = 240
    interior_max_abs_x: float = 50.0
    interior_max_abs_y: float = 33.5
    tau_absorb: float = 0.85
    max_absorb_b4d: float = 0.25
    max_absorb_still: float = 1.0
    min_same_team_ratio: float = 0.85

    # 2. Touchline / Goal-Line Same-Team Split (X -> X -> X -> D -> X)
    enable_split: bool = True
    min_split_dead: int = 40
    min_live_after_split: int = 15
    split_oob_x: float = 52.6
    split_oob_y: float = 34.1
    w_oob_split: float = 0.15
    tau_split_frame: float = 0.40
    tau_split_d: float = 0.55
    min_split_b4d: float = 0.25
    require_oob_for_split: bool = True

    # 3. Out-of-Bounds Tail Trimming (Live <-> D boundary refinement)
    enable_oob_trim: bool = False
    max_oob_trim: int = 15
    min_live_len: int = 15
    trim_oob_x: float = 53.0
    trim_oob_y: float = 34.5
    tau_oob: float = 0.45

    # 4. Restart Taker Attribution (D -> H/A)
    enable_restart_attr: bool = True
    min_dead_for_restart: int = 50
    restart_near_m: float = 1.8
    restart_far_m: float = 3.5
    tau_restart_ambig: float = 0.25
    max_restart_flip_len: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_centered_ball_coords(
    X: pd.DataFrame,
    pitch: tuple[float, float] = (105.0, 68.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H<->A invariant `max_abs_x`, `max_abs_y`, and `cur_abs_x`, `cur_abs_y` arrays in meters from center.

    Handles both raw `[0, L] x [0, W]` coordinates and pre-centered `[-L/2, L/2] x [-W/2, W/2]` coordinates.
    Rounds to 1e-3m so `L - x` vs `x` has 0.000e+00 IEEE-754 floating-point drift.
    """
    T = len(X)
    L, W = float(pitch[0]), float(pitch[1])

    # Detect whether ball_x / ball_last_x are in [0, L] or already centered around 0
    ref_col = "ball_last_x" if "ball_last_x" in X.columns else ("ball_x" if "ball_x" in X.columns else None)
    is_centered = False
    if ref_col is not None:
        vals = X[ref_col].to_numpy(np.float64)
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0 and np.min(finite) < -10.0 and np.max(finite) < 60.0:
            is_centered = True

    offset_x = 0.0 if is_centered else 0.5 * L
    offset_y = 0.0 if is_centered else 0.5 * W

    abs_x_list = []
    for cx in ("ball_x", "ball_last_x"):
        if cx in X.columns:
            arr = np.round(np.abs(X[cx].to_numpy(np.float64) - offset_x), 3)
            abs_x_list.append(arr)
    abs_y_list = []
    for cy in ("ball_y", "ball_last_y"):
        if cy in X.columns:
            arr = np.round(np.abs(X[cy].to_numpy(np.float64) - offset_y), 3)
            abs_y_list.append(arr)

    if abs_x_list:
        stack_x = np.vstack(abs_x_list)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_abs_x = np.nanmax(stack_x, axis=0)
        cur_abs_x = abs_x_list[0]
    else:
        max_abs_x = np.zeros(T, dtype=np.float64)
        cur_abs_x = np.zeros(T, dtype=np.float64)

    if abs_y_list:
        stack_y = np.vstack(abs_y_list)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_abs_y = np.nanmax(stack_y, axis=0)
        cur_abs_y = abs_y_list[0]
    else:
        max_abs_y = np.zeros(T, dtype=np.float64)
        cur_abs_y = np.zeros(T, dtype=np.float64)

    return max_abs_x, max_abs_y, cur_abs_x, cur_abs_y


def apply_xdx_arbitrator(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X: pd.DataFrame,
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
    cfg: XDXConfig = XDXConfig(),
    pitch: tuple[float, float] = (105.0, 68.0),
) -> np.ndarray:
    """Apply the 4-stage symmetric causal `X -> D -> X` & restart/OOB arbitrator to decoded labels `pred`."""
    out = np.asarray(pred, dtype=np.int8).copy()
    T = len(out)
    if T == 0:
        return out

    half_arr = np.asarray(half)
    p_H = np.asarray(proba[:, 0], dtype=np.float64)
    p_A = np.asarray(proba[:, 1], dtype=np.float64)
    p_D = np.asarray(proba[:, 2], dtype=np.float64)

    max_abs_x, max_abs_y, cur_abs_x, cur_abs_y = _extract_centered_ball_coords(X, pitch=pitch)
    vis = np.nan_to_num(X["ball_visible"].to_numpy(np.float64), nan=0.0) if "ball_visible" in X.columns else np.zeros(T)
    outside = (
        np.nan_to_num(X["ball_outside_pitch"].to_numpy(np.float64), nan=0.0)
        if "ball_outside_pitch" in X.columns
        else np.zeros(T)
    )
    still = (
        np.nan_to_num(X["frac_still_all"].to_numpy(np.float64), nan=0.0)
        if "frac_still_all" in X.columns
        else np.zeros(T)
    )
    dH = np.nan_to_num(X["dist_ball_H"].to_numpy(np.float64), nan=99.0) if "dist_ball_H" in X.columns else np.full(T, 99.0)
    dA = np.nan_to_num(X["dist_ball_A"].to_numpy(np.float64), nan=99.0) if "dist_ball_A" in X.columns else np.full(T, 99.0)

    p_l2d = np.asarray(p_trans_4d[:, 2], dtype=np.float64) if p_trans_4d is not None else None
    p_d2l = np.asarray(p_trans_4d[:, 3], dtype=np.float64) if p_trans_4d is not None else None

    # -------------------------------------------------------------------------
    # STAGE 1: Interior False-D Absorption (X -> D -> X -> X)
    # -------------------------------------------------------------------------
    if cfg.enable_absorb and cfg.max_absorb_len > 0:
        s_arr, e_arr, v_arr = rle_segments(out, half_arr)
        n_seg = len(s_arr)
        for i in range(1, n_seg - 1):
            if v_arr[i] != DEAD:
                continue
            team_left = int(v_arr[i - 1])
            team_right = int(v_arr[i + 1])
            if team_left not in (HOME, AWAY) or team_left != team_right:
                continue
            a, b = int(s_arr[i]), int(e_arr[i])
            if half_arr[a] <= 0 or half_arr[s_arr[i - 1]] != half_arr[a] or half_arr[e_arr[i + 1] - 1] != half_arr[a]:
                continue
            d_len = b - a
            if d_len > cfg.max_absorb_len:
                continue

            # Check pitch interior condition over [a, b)
            seg_max_x = np.nanmax(max_abs_x[a:b]) if np.any(np.isfinite(max_abs_x[a:b])) else 0.0
            seg_max_y = np.nanmax(max_abs_y[a:b]) if np.any(np.isfinite(max_abs_y[a:b])) else 0.0
            if seg_max_x >= cfg.interior_max_abs_x or seg_max_y >= cfg.interior_max_abs_y:
                continue
            if np.max(outside[a:b]) > 0.5:
                continue

            # Check mean p_D and stillness
            if float(np.mean(p_D[a:b])) >= cfg.tau_absorb:
                continue
            if float(np.mean(still[a:b])) >= cfg.max_absorb_still:
                continue

            # Check 4-directional Live->D and D->Live boundary pair if available
            if p_l2d is not None and p_d2l is not None and cfg.max_absorb_b4d < 1.0:
                lo_a, hi_a = max(0, a - 3), min(T, a + 4)
                lo_b, hi_b = max(0, b - 3), min(T, b + 4)
                b4d_pair = 0.5 * (float(np.max(p_l2d[lo_a:hi_a])) + float(np.max(p_d2l[lo_b:hi_b])))
                if b4d_pair >= cfg.max_absorb_b4d:
                    continue

            # Ensure team X is not contradicted by the opponent inside [a, b)
            p_team = float(np.mean(proba[a:b, team_left]))
            p_opp = float(np.mean(proba[a:b, 1 - team_left]))
            if p_team + 1e-6 < p_opp * cfg.min_same_team_ratio:
                continue

            out[a:b] = team_left

    # -------------------------------------------------------------------------
    # STAGE 2: Touchline / Goal-Line Same-Team Split (X -> X -> X -> D -> X)
    # -------------------------------------------------------------------------
    if cfg.enable_split and cfg.min_split_dead > 0:
        s_arr, e_arr, v_arr = rle_segments(out, half_arr)
        is_oob = (
            (max_abs_x > cfg.split_oob_x)
            | (max_abs_y > cfg.split_oob_y)
            | (outside > 0.5)
        )
        score_d = p_D + cfg.w_oob_split * is_oob.astype(np.float64)

        for i in range(len(s_arr)):
            team_x = int(v_arr[i])
            if team_x not in (HOME, AWAY):
                continue
            a, b = int(s_arr[i]), int(e_arr[i])
            if half_arr[a] <= 0:
                continue
            min_total = 2 * cfg.min_live_after_split + cfg.min_split_dead
            if (b - a) < min_total:
                continue

            lo = a + cfg.min_live_after_split
            hi = b - cfg.min_live_after_split
            mask = score_d[lo:hi] >= cfg.tau_split_frame
            if not np.any(mask):
                continue

            # Find contiguous candidate runs in mask
            padded = np.r_[False, mask, False]
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            runs_u = lo + edges[0::2]
            runs_v = lo + edges[1::2]

            for u, v in zip(runs_u, runs_v):
                if (v - u) < cfg.min_split_dead:
                    continue
                if float(np.mean(score_d[u:v])) < cfg.tau_split_d:
                    continue
                w_lo, w_hi = max(a, int(u) - 6), min(b, int(v) + 6)
                if cfg.require_oob_for_split and not np.any(is_oob[w_lo:w_hi]):
                    continue
                if p_l2d is not None and p_d2l is not None and cfg.min_split_b4d > 0.0:
                    u_lo, u_hi = max(a, int(u) - 6), min(b, int(u) + 7)
                    v_lo, v_hi = max(a, int(v) - 6), min(b, int(v) + 7)
                    b4d_score = 0.5 * (float(np.max(p_l2d[u_lo:u_hi])) + float(np.max(p_d2l[v_lo:v_hi])))
                    if b4d_score < cfg.min_split_b4d:
                        continue
                out[u:v] = DEAD

    # -------------------------------------------------------------------------
    # STAGE 3: Out-of-Bounds Tail Trimming (Live <-> D boundary refinement)
    # -------------------------------------------------------------------------
    if cfg.enable_oob_trim and cfg.max_oob_trim > 0:
        s_arr, e_arr, v_arr = rle_segments(out, half_arr)
        vis_oob = (vis > 0.5) & ((cur_abs_x > cfg.trim_oob_x) | (cur_abs_y > cfg.trim_oob_y)) & (p_D > cfg.tau_oob)
        n_seg = len(s_arr)
        for i in range(n_seg):
            if v_arr[i] != DEAD:
                continue
            a, b = int(s_arr[i]), int(e_arr[i])
            if half_arr[a] <= 0:
                continue

            # Trim tail of preceding Live segment [s_prev, a) into DEAD
            if i > 0 and v_arr[i - 1] in (HOME, AWAY) and half_arr[s_arr[i - 1]] == half_arr[a]:
                s_prev = int(s_arr[i - 1])
                max_steps = min(cfg.max_oob_trim, (a - s_prev) - cfg.min_live_len)
                t = a - 1
                steps = 0
                while steps < max_steps and vis_oob[t]:
                    out[t] = DEAD
                    t -= 1
                    steps += 1

            # Trim head of following Live segment [b, e_next) into DEAD
            if i + 1 < n_seg and v_arr[i + 1] in (HOME, AWAY) and half_arr[b] == half_arr[a]:
                e_next = int(e_arr[i + 1])
                max_steps = min(cfg.max_oob_trim, (e_next - b) - cfg.min_live_len)
                t = b
                steps = 0
                while steps < max_steps and vis_oob[t]:
                    out[t] = DEAD
                    t += 1
                    steps += 1
                s_arr[i + 1] = t

    # -------------------------------------------------------------------------
    # STAGE 4: Restart Taker Attribution (D -> H/A)
    # -------------------------------------------------------------------------
    if cfg.enable_restart_attr and cfg.max_restart_flip_len > 0:
        s_arr, e_arr, v_arr = rle_segments(out, half_arr)
        n_seg = len(s_arr)
        for i in range(n_seg - 1):
            if v_arr[i] != DEAD:
                continue
            a, b = int(s_arr[i]), int(e_arr[i])
            if (b - a) < cfg.min_dead_for_restart:
                continue
            team_next = int(v_arr[i + 1])
            if team_next not in (HOME, AWAY):
                continue
            c = int(e_arr[i + 1])
            if half_arr[a] <= 0 or half_arr[b] != half_arr[a]:
                continue
            live_len = c - b
            if live_len > cfg.max_restart_flip_len:
                continue

            # Check uniqueness of ball proximity in restart window [b - 10, b + 5]
            w_lo = max(a, b - 10)
            w_hi = min(c, b + 6)
            if w_hi <= w_lo:
                continue
            min_dH = float(np.min(dH[w_lo:w_hi]))
            min_dA = float(np.min(dA[w_lo:w_hi]))

            t_restart = None
            if min_dH < cfg.restart_near_m and min_dA > cfg.restart_far_m:
                t_restart = HOME
            elif min_dA < cfg.restart_near_m and min_dH > cfg.restart_far_m:
                t_restart = AWAY

            if t_restart is not None and t_restart != team_next:
                ambig = float(np.mean(np.abs(p_H[b:c] - p_A[b:c])))
                if ambig < cfg.tau_restart_ambig:
                    out[b:c] = t_restart

    return out


def count_interior_xdx(
    pred: np.ndarray,
    half: np.ndarray,
    X: pd.DataFrame,
    max_len: int = 240,
    pitch: tuple[float, float] = (105.0, 68.0),
) -> int:
    """Count how many `X -> D -> X` segments (`len <= max_len`) remain inside the pitch interior."""
    s_arr, e_arr, v_arr = rle_segments(pred, half)
    max_abs_x, max_abs_y, _, _ = _extract_centered_ball_coords(X, pitch=pitch)
    n_int = 0
    for i in range(1, len(s_arr) - 1):
        if v_arr[i] != DEAD:
            continue
        if v_arr[i - 1] in (HOME, AWAY) and v_arr[i - 1] == v_arr[i + 1]:
            a, b = int(s_arr[i]), int(e_arr[i])
            if half[a] <= 0 or half[s_arr[i - 1]] != half[a] or half[e_arr[i + 1] - 1] != half[a]:
                continue
            if (b - a) <= max_len:
                mx = float(np.nanmax(max_abs_x[a:b])) if np.any(np.isfinite(max_abs_x[a:b])) else 0.0
                my = float(np.nanmax(max_abs_y[a:b])) if np.any(np.isfinite(max_abs_y[a:b])) else 0.0
                if mx < 50.0 and my < 33.5:
                    n_int += 1
    return n_int

