"""Exact Home <-> Away (H <-> A) Equivariance / Symmetry Transformations.

Provides:
1. swap_match(m): returns a Match with HOME and AWAY swapped and attack_sign negated (so new HOME attacks +x).
2. mirror_v1ctl_df(X, half, pitch): fast O(1) algebraic transformation of a pre-computed v1ctl DataFrame
   that produces the exact feature DataFrame of swap_match(m) in ~15ms without re-running raw tracking loops.
3. swap_labels(y): swaps HOME (0) <-> AWAY (1), keeping DEAD (2) and UNLABELLED (-1) unchanged.
4. swap_proba(p): swaps columns 0 (H) and 1 (A) of a (T, 3) probability array.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd

from src.data.common import AWAY, FPS, HOME, Match


def swap_match(m: Match) -> Match:
    """Return a Match with HOME and AWAY swapped and attack_sign negated."""
    new_team = np.where(m.slot_team == HOME, AWAY, np.where(m.slot_team == AWAY, HOME, m.slot_team)).astype(np.int8)
    new_labels = swap_labels(m.labels) if m.labels is not None else None
    # If already direction_normalized, rotate 180 deg so new HOME attacks +x
    if m.direction_normalized:
        L, W = m.pitch
        P, B = m.P.copy(), m.B.copy()
        P[..., 0] = L - P[..., 0]
        P[..., 1] = W - P[..., 1]
        B[:, 0] = L - B[:, 0]
        B[:, 1] = W - B[:, 1]
        return replace(
            m, P=P, B=B, slot_team=new_team, attack_sign=-m.attack_sign,
            home_name=m.away_name, away_name=m.home_name, labels=new_labels
        )
    return replace(
        m, slot_team=new_team, attack_sign=-m.attack_sign,
        home_name=m.away_name, away_name=m.home_name, labels=new_labels
    )


def swap_labels(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[y == HOME] = AWAY
    out[y == AWAY] = HOME
    return out


def swap_proba(p: np.ndarray) -> np.ndarray:
    out = p.copy()
    out[:, HOME] = p[:, AWAY]
    out[:, AWAY] = p[:, HOME]
    return out


def mirror_v1ctl_df(X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float] = (105.0, 68.0)) -> pd.DataFrame:
    """Algebraically transform a v1 or v1ctl DataFrame under H <-> A swap + 180-deg pitch rotation."""
    L, W = float(pitch[0]), float(pitch[1])
    M = X.copy()

    # 1. Pitch coordinate reflections (x -> L - x, y -> W - y)
    for col in ("ball_x", "ball_last_x", "ball_next_x"):
        if col in M.columns:
            M[col] = L - X[col]
    for col in ("ball_y", "ball_last_y", "ball_next_y"):
        if col in M.columns:
            M[col] = W - X[col]

    # 2. Centroids: centroid_x_H' = L - centroid_x_A, centroid_y_H' = W - centroid_y_A
    if "centroid_x_H" in M.columns and "centroid_x_A" in M.columns:
        M["centroid_x_H"] = L - X["centroid_x_A"]
        M["centroid_x_A"] = L - X["centroid_x_H"]
    if "centroid_y_H" in M.columns and "centroid_y_A" in M.columns:
        M["centroid_y_H"] = W - X["centroid_y_A"]
        M["centroid_y_A"] = W - X["centroid_y_H"]

    # 3. Direct H <-> A swaps
    swap_pairs = [
        ("dist_ball_H", "dist_ball_A"),
        ("dist_lastball_H", "dist_lastball_A"),
        ("n_within_2m_H", "n_within_2m_A"),
        ("n_within_5m_H", "n_within_5m_A"),
        ("n_within_10m_H", "n_within_10m_A"),
        ("since_touch_H", "since_touch_A"),
        ("until_touch_H", "until_touch_A"),
        ("n_visible_H", "n_visible_A"),
        ("spread_x_H", "spread_x_A"),
        ("spread_y_H", "spread_y_A"),
        ("def_line_H", "def_line_A"),
        ("fwd_line_H", "fwd_line_A"),
        ("width_H", "width_A"),
        ("speed_mean_H", "speed_mean_A"),
        ("speed_p90_H", "speed_p90_A"),
        ("frac_still_H", "frac_still_A"),
        ("in_own_box_H", "in_own_box_A"),
        ("in_opp_box_H", "in_opp_box_A"),
        ("gk_x_H", "gk_x_A"),
        ("gk_dist_ball_H", "gk_dist_ball_A"),
        ("carry_H", "carry_A"),
        ("contest_n2m_H", "contest_n2m_A"),
    ]
    for w in (0.4, 0.8, 2, 4, 8):
        swap_pairs.append((f"speed_mean_H_m{w}s", f"speed_mean_A_m{w}s"))
        swap_pairs.append((f"n_within_5m_H_m{w}s", f"n_within_5m_A_m{w}s"))
    for w in (20, 40, 80):
        swap_pairs.append((f"spread_x_H_m{w}s", f"spread_x_A_m{w}s"))

    for ch, ca in swap_pairs:
        if ch in M.columns and ca in M.columns:
            M[ch] = X[ca]
            M[ca] = X[ch]

    # 4. Swap with sign negation (because x -> L - x AND H <-> A)
    neg_swap_pairs = [
        ("ball_to_centroid_H", "ball_to_centroid_A"),
        ("team_vx_H", "team_vx_A"),
        ("team_vx_H_m1.0s", "team_vx_A_m1.0s"),
        ("team_vx_H_m3.0s", "team_vx_A_m3.0s"),
    ]
    for ch, ca in neg_swap_pairs:
        if ch in M.columns and ca in M.columns:
            M[ch] = -X[ca]
            M[ca] = -X[ch]

    # 5. Pure sign negations (team-signed features and x-velocities)
    neg_cols = [
        "dist_ball_diff",
        "dist_lastball_diff",
        "touch_team_r1",
        "touch_team_r2",
        "touch_team_r3",
        "control_team",
        "nearest_team",
        "nearest_team_hold25",
        "touch_event",
        "carry_diff",
        "ball_vx",
        "receiver_team",
        "dist_lastball_diff_fwd_minus_bwd_2s",
        "touch_team_r2_fwd_minus_bwd_2s",
        "carry_diff_m0.4s",
        "carry_diff_m1.0s",
        "receiver_team_m0.4s",
        "receiver_team_m1.0s",
        "ball_vx_m1.0s",
        "ball_vx_m3.0s",
    ]
    for key in ("dist_ball_diff", "dist_lastball_diff", "touch_team_r2", "control_team", "nearest_team"):
        for w in (0.4, 0.8, 2, 4, 8):
            neg_cols.append(f"{key}_m{w}s")
    for key in ("touch_team_r1", "touch_team_r2", "touch_team_r3", "control_team", "nearest_team", "nearest_team_hold25",
                "dist_ball_diff", "dist_lastball_diff"):
        for w_s in (0.4, 1.0, 2.0):
            neg_cols.append(f"chg_{key}_{w_s}s")
            # Note: chg_abs_* is invariant because |-d| = |d|!

    for col in neg_cols:
        if col in M.columns:
            M[col] = -X[col]

    # 6. Recompute the 4 temporal delta columns that depend on centroid_x_H' = L - centroid_x_A
    hs = pd.Series(half)
    if "centroid_x_H" in M.columns:
        s_new = M["centroid_x_H"]
        if "centroid_x_H_fwd_minus_bwd_2s" in M.columns:
            fwd = s_new.groupby(hs).transform(lambda x: x[::-1].rolling(2 * FPS, min_periods=1).mean()[::-1])
            bwd = s_new.groupby(hs).transform(lambda x: x.rolling(2 * FPS, min_periods=1).mean())
            M["centroid_x_H_fwd_minus_bwd_2s"] = (fwd - bwd).astype(np.float32)
        for w_s in (0.4, 1.0, 2.0):
            c_col = f"chg_centroid_x_H_{w_s}s"
            a_col = f"chg_abs_centroid_x_H_{w_s}s"
            if c_col in M.columns:
                w = int(w_s * FPS)
                fwd = s_new.groupby(hs).transform(lambda x: x[::-1].rolling(w, min_periods=1).mean()[::-1].shift(-1))
                bwd = s_new.groupby(hs).transform(lambda x: x.rolling(w, min_periods=1).mean())
                d = (fwd - bwd).astype(np.float32)
                M[c_col] = d
                M[a_col] = d.abs()

    return M
