"""EXP-15: Controlled Touch vs Deflection / Interception Arbitrator (`SymTouchTypeArbitrator`).

Addresses two persistent live-play error modes on short spells (0.5 s - 2.5 s, L in [5, 85] frames):
1. Mode 1 (Involuntary Rebound / Deflection Absorption):
   Short spells `Y` inside `X -> Y -> X` (`XYX`), `X -> Y -> D` (`XYD`), or `D -> Y -> X` (`DYX`)
   where the ball ricochets off a defender/blocker `Y` without controlled possession.
   Combines a 61-D H<->A-invariant contact-kinematics + post-impulse fate + 9-model consensus
   feature vector with a 6-fold inner bagging ridge ensemble, physical touch invalidity score,
   and a physical-contact veto (`is_real_physical_touch`).
2. Mode 2 (Real Interception / Clearance Rescue):
   Candidate short clearances/interceptions `Y` inside long `X` spells where tree/impulse families,
   `SymBoundary4Dir`, and physical contact (`d1_Y <= 1.25m` + impulse) detect a genuine touch
   by `Y` that temporal smoothing suppressed.

Guarantees exact 100% H <-> A equivariance (`0.000e+00` discrepancy under team swap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

import evaluation.possession_metrics as pm
from src.data.common import AWAY, DEAD, HOME
from src.eval.metrics import to_official

M1_SUB: list[int] = [
    0, 1, 2, 3, 4, 5, 9, 13, 15, 17, 19, 24, 26, 28, 31, 38, 42, 45, 46, 51, 55, 57,
]
M2_SUB: list[int] = [
    0, 3, 4, 9, 12, 14, 16, 21, 24, 26, 28, 29, 30, 33, 41, 42, 45, 46, 48, 49,
]


@dataclass
class TouchTypeArbitratorConfig:
    """Configuration for `SymTouchTypeArbitrator` (`EXP-15`)."""

    min_len_m1: int = 5
    max_len_m1: int = 85
    min_len_m2: int = 8
    max_len_m2: int = 65
    l2_reg: float = 180.0
    min_th_m1: float = 0.01
    min_th_m2: float = 0.15
    min_gate_xyx: float = 0.00025
    min_gate_xyd_dyx: float = 0.00060
    min_gate_m2: float = 0.00050
    enable_mode1: bool = True
    enable_mode2: bool = True
    # Fitted pattern parameters: pat -> dict with 'active', 'w_phys', 'th', 'use_veto', 'models'
    mode1_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    mode2_params: dict[str, Any] = field(default_factory=dict)


def _get_col(xv: pd.DataFrame | dict[str, np.ndarray], name: str, T: int, default: float = 0.0) -> np.ndarray:
    if isinstance(xv, pd.DataFrame):
        if name in xv.columns:
            return xv[name].to_numpy(dtype=np.float64, copy=False)
        return np.full(T, default, dtype=np.float64)
    if name in xv:
        return np.asarray(xv[name], dtype=np.float64)
    return np.full(T, default, dtype=np.float64)


def preclean_tracking_arrays(xv: pd.DataFrame | dict[str, np.ndarray], T: int | None = None) -> dict[str, np.ndarray]:
    """Extract and sanitize tracking/kinematic columns needed by `EXP-15` and `EXP-16`."""
    if isinstance(xv, dict) and "d1_h_vis" in xv and "imp_norm_d4" in xv:
        return xv
    if T is None:
        if isinstance(xv, pd.DataFrame):
            T = len(xv)
        else:
            first_v = next(iter(xv.values()))
            T = len(first_v)

    c: dict[str, np.ndarray] = {}
    vis = np.nan_to_num(_get_col(xv, "ball_visible", T, 0.0), nan=0.0)
    c["vis"] = vis
    d1_ig_h = np.clip(np.nan_to_num(_get_col(xv, "d1_h_impulse_gated", T, 15.0), nan=15.0), 0.0, 30.0)
    d1_ig_a = np.clip(np.nan_to_num(_get_col(xv, "d1_a_impulse_gated", T, 15.0), nan=15.0), 0.0, 30.0)
    c["d1_ig_h"] = d1_ig_h
    c["d1_ig_a"] = d1_ig_a

    d1_h_raw = _get_col(xv, "dist_ball_H", T, np.nan)
    d1_a_raw = _get_col(xv, "dist_ball_A", T, np.nan)
    c["d1_h"] = np.clip(np.where(np.isfinite(d1_h_raw), d1_h_raw, d1_ig_h), 0.0, 30.0)
    c["d1_a"] = np.clip(np.where(np.isfinite(d1_a_raw), d1_a_raw, d1_ig_a), 0.0, 30.0)
    c["d1_h_vis"] = np.where(np.isfinite(d1_h_raw) & (vis > 0.5), d1_h_raw, 30.0)
    c["d1_a_vis"] = np.where(np.isfinite(d1_a_raw) & (vis > 0.5), d1_a_raw, 30.0)

    c["bspd"] = np.clip(np.nan_to_num(_get_col(xv, "ball_speed", T, 5.0), nan=5.0), 0.0, 35.0)
    c["bacc"] = np.clip(np.nan_to_num(_get_col(xv, "ball_accel", T, 0.0), nan=0.0), -50.0, 50.0)
    c["bang"] = np.clip(np.nan_to_num(np.abs(_get_col(xv, "ball_angle_change", T, 0.0)), nan=0.0), 0.0, 3.1416)
    c["imp_d2"] = np.clip(np.nan_to_num(_get_col(xv, "impulse_mag_d2", T, 0.0), nan=0.0), 0.0, 35.0)
    c["imp_d4"] = np.clip(np.nan_to_num(_get_col(xv, "impulse_mag_d4", T, 0.0), nan=0.0), 0.0, 35.0)
    c["imp_norm_d4"] = np.clip(np.nan_to_num(_get_col(xv, "impulse_norm_d4", T, 0.0), nan=0.0), 0.0, 35.0)
    c["cos_d2"] = np.clip(np.nan_to_num(_get_col(xv, "impulse_cos_d2", T, 1.0), nan=1.0), -1.0, 1.0)
    c["cos_d4"] = np.clip(np.nan_to_num(_get_col(xv, "impulse_cos_d4", T, 1.0), nan=1.0), -1.0, 1.0)
    c["chord_flight"] = np.clip(np.nan_to_num(_get_col(xv, "chord_in_flight", T, 0.0), nan=0.0), 0.0, 1.0)
    c["nk_h"] = np.clip(np.nan_to_num(_get_col(xv, "next_knot_d1_h", T, 10.0), nan=10.0), 0.0, 30.0)
    c["nk_a"] = np.clip(np.nan_to_num(_get_col(xv, "next_knot_d1_a", T, 10.0), nan=10.0), 0.0, 30.0)
    c["pk_h"] = np.clip(np.nan_to_num(_get_col(xv, "prev_knot_d1_h", T, 10.0), nan=10.0), 0.0, 30.0)
    c["pk_a"] = np.clip(np.nan_to_num(_get_col(xv, "prev_knot_d1_a", T, 10.0), nan=10.0), 0.0, 30.0)
    c["cs_h"] = np.clip(np.nan_to_num(_get_col(xv, "chord_same_team_prob_h", T, 0.5), nan=0.5), 0.0, 1.0)
    c["cs_a"] = np.clip(np.nan_to_num(_get_col(xv, "chord_same_team_prob_a", T, 0.5), nan=0.5), 0.0, 1.0)
    c["cturn"] = np.clip(np.nan_to_num(_get_col(xv, "chord_turnover_prob", T, 0.5), nan=0.5), 0.0, 1.0)
    c["out_pitch"] = np.clip(np.nan_to_num(_get_col(xv, "ball_outside_pitch", T, 0.0), nan=0.0), 0.0, 1.0)
    c["gdisc"] = np.clip(np.nan_to_num(_get_col(xv, "ball_focus_discrepancy", T, 0.0), nan=0.0), 0.0, 35.0)
    c["fstill"] = np.clip(np.nan_to_num(_get_col(xv, "frac_still_all", T, 0.0), nan=0.0), 0.0, 1.0)
    return c


def rle_runs_half(y: np.ndarray, half: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Run-length encode label array `y` respecting half boundaries."""
    T = len(y)
    if T == 0:
        return []
    ch = np.flatnonzero((y[1:] != y[:-1]) | (half[1:] != half[:-1])) + 1
    starts = np.r_[0, ch]
    ends = np.r_[ch, T]
    return [(int(a), int(b), int(y[a]), int(half[a])) for a, b in zip(starts, ends)]


def build_episode_features(
    a: int,
    b: int,
    y_lbl: int,
    x_lbl: int,
    a_prev: int,
    b_next: int,
    P: np.ndarray,
    p_ch: np.ndarray,
    p4d: np.ndarray,
    cxv: dict[str, np.ndarray],
    fams: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Build 61-D 100% H<->A-invariant feature vector for candidate episode `[a, b)` of team `y_lbl` vs `x_lbl`."""
    T = len(P)
    L = float(b - a)
    sl = slice(a, b)

    # 1. Probabilities & family consensus inside [a, b)
    pY = P[sl, y_lbl]
    pX = P[sl, x_lbl]
    pD = P[sl, DEAD]
    mean_pY = float(np.mean(pY))
    max_pY = float(np.max(pY))
    q75_pY = float(np.percentile(pY, 75))
    mean_pX = float(np.mean(pX))
    mean_pD = float(np.mean(pD))
    margin_YX = float(np.mean(pY - pX))
    max_margin_YX = float(np.max(pY - pX))
    logodds_YX = float(np.mean(np.log((pY + 1e-4) / (pX + 1e-4))))

    if fams is not None and len(fams) >= 8:
        tree_idx = [k for k in (0, 1, 2) if k < len(fams)]
        seq_end = 8 if len(fams) >= 9 else 7
        seq_idx = [k for k in range(3, seq_end) if k < len(fams)]
        pset_idx = 8 if len(fams) >= 9 else 7
        tree_pY = float(np.mean([np.mean(fams[k][sl, y_lbl]) for k in tree_idx]))
        tree_max_pY = float(np.mean([np.max(fams[k][sl, y_lbl]) for k in tree_idx]))
        tree_margin_YX = float(np.mean([np.mean(fams[k][sl, y_lbl] - fams[k][sl, x_lbl]) for k in tree_idx]))
        seq_pY = float(np.mean([np.mean(fams[k][sl, y_lbl]) for k in seq_idx]))
        seq_margin_YX = float(np.mean([np.mean(fams[k][sl, y_lbl] - fams[k][sl, x_lbl]) for k in seq_idx]))
        pset_pY = float(np.mean(fams[pset_idx][sl, y_lbl]))
        pset_margin_YX = float(np.mean(fams[pset_idx][sl, y_lbl] - fams[pset_idx][sl, x_lbl]))
        fam_margins = [float(np.mean(fp[sl, y_lbl] - fp[sl, x_lbl])) for fp in fams]
        fam_vote_Y = float(np.mean([1.0 if m > 0 else 0.0 for m in fam_margins]))
        fam_min_margin = float(np.min(fam_margins))
        tree_minus_seq = tree_margin_YX - seq_margin_YX
    else:
        tree_pY, tree_max_pY, tree_margin_YX = mean_pY, max_pY, margin_YX
        seq_pY, seq_margin_YX = mean_pY, margin_YX
        pset_pY, pset_margin_YX = mean_pY, margin_YX
        fam_vote_Y = 1.0 if margin_YX > 0 else 0.0
        fam_min_margin = margin_YX
        tree_minus_seq = 0.0

    # Flanking context
    sl_pre = slice(max(a_prev, a - 20), a)
    sl_post = slice(b, min(b_next, b + 20))
    pre_pX = float(np.mean(P[sl_pre, x_lbl])) if sl_pre.stop > sl_pre.start else mean_pX
    post_pX = float(np.mean(P[sl_post, x_lbl])) if sl_post.stop > sl_post.start else mean_pX
    post_pD = float(np.mean(P[sl_post, DEAD])) if sl_post.stop > sl_post.start else mean_pD
    l_flank = min(float(a - a_prev) / 100.0, 2.0)
    r_flank = min(float(b_next - b) / 100.0, 2.0)

    # 2. Directional boundaries (SymBoundary4Dir: col 0 is ->H, col 1 is ->A, col 2 is ->D, col 3 is D->Live)
    col_to_Y = 0 if y_lbl == HOME else 1
    col_to_X = 0 if x_lbl == HOME else 1
    wa0, wa1 = max(0, a - 4), min(T, a + 5)
    wb0, wb1 = max(0, b - 4), min(T, b + 5)
    b4d_in_Y = float(np.max(p4d[wa0:wa1, col_to_Y])) if wa1 > wa0 else 0.0
    b4d_in_fromD = float(np.max(p4d[wa0:wa1, 3])) if wa1 > wa0 else 0.0
    b4d_out_X = float(np.max(p4d[wb0:wb1, col_to_X])) if wb1 > wb0 else 0.0
    b4d_out_D = float(np.max(p4d[wb0:wb1, 2])) if wb1 > wb0 else 0.0
    pc_in = float(np.max(p_ch[wa0:wa1])) if wa1 > wa0 else 0.0
    pc_out = float(np.max(p_ch[wb0:wb1])) if wb1 > wb0 else 0.0

    # 3. Contact Kinematics Features (t_in, t_out)
    w_imp = slice(max(0, a - 3), min(T, b + 4))
    imp_d2_max = float(np.max(cxv["imp_d2"][w_imp])) / 30.0
    imp_d4_max = float(np.max(cxv["imp_d4"][w_imp])) / 30.0
    imp_norm_d4_max = float(np.max(cxv["imp_norm_d4"][w_imp])) / 30.0
    imp_in = float(np.max(cxv["imp_d2"][wa0:wa1])) / 30.0 if wa1 > wa0 else 0.0
    imp_out = float(np.max(cxv["imp_d2"][wb0:wb1])) / 30.0 if wb1 > wb0 else 0.0

    # Ball velocity deflection & deceleration/control ratio
    ang_max = float(np.max(cxv["bang"][w_imp])) / 3.1416
    cos_d2_min = float(np.min(cxv["cos_d2"][w_imp]))
    cos_d4_min = float(np.min(cxv["cos_d4"][w_imp]))
    v_seg = cxv["bspd"][sl]
    v_pre_idx = max(0, a - 2)
    v_in_ref = max(float(cxv["bspd"][v_pre_idx]), 1.0)
    decel_ratio = min(float(np.min(v_seg)) / v_in_ref, 3.0)
    mean_v_seg = float(np.mean(v_seg)) / 30.0
    min_v_seg = float(np.min(v_seg)) / 30.0

    # Effective contact frame count of Y (d1_Y < 1.3m with ball_visible == 1) vs min distance of X
    vis = cxv["vis"][sl]
    frac_vis = float(np.mean(vis))
    d1_Y_vis = (cxv["d1_h_vis"] if y_lbl == HOME else cxv["d1_a_vis"])[sl]
    d1_X_vis = (cxv["d1_h_vis"] if x_lbl == HOME else cxv["d1_a_vis"])[sl]
    d1_Y = (cxv["d1_h"] if y_lbl == HOME else cxv["d1_a"])[sl]
    d1_X = (cxv["d1_h"] if x_lbl == HOME else cxv["d1_a"])[sl]
    d1_ig_Y = (cxv["d1_ig_h"] if y_lbl == HOME else cxv["d1_ig_a"])[sl]
    d1_ig_X = (cxv["d1_ig_h"] if x_lbl == HOME else cxv["d1_ig_a"])[sl]

    eff_contact_cnt_Y = float(np.sum(d1_Y_vis < 1.3)) / 25.0
    eff_contact_frac_Y = float(np.mean(d1_Y_vis < 1.3))
    eff_contact_cnt_X = float(np.sum(d1_X_vis < 1.3)) / 25.0
    min_d1_Y = float(np.min(d1_Y)) / 30.0
    min_d1_X = float(np.min(d1_X)) / 30.0
    mean_d1_diff_XY = float(np.mean(np.clip(d1_X - d1_Y, -20.0, 20.0))) / 20.0
    min_d1_ig_Y = float(np.min(d1_ig_Y)) / 30.0
    min_d1_ig_X = float(np.min(d1_ig_X)) / 30.0
    chord_flight = float(np.mean(cxv["chord_flight"][sl]))

    # 4. Post-Impulse Fate Features over [t_out, t_out + 15]
    sl_fate = slice(b, min(T, b + 15))
    if sl_fate.stop > sl_fate.start:
        d1_Y_post = (cxv["d1_h"] if y_lbl == HOME else cxv["d1_a"])[sl_fate]
        d1_X_post = (cxv["d1_h"] if x_lbl == HOME else cxv["d1_a"])[sl_fate]
        post_d1_diff_XY = float(np.mean(np.clip(d1_X_post - d1_Y_post, -20.0, 20.0))) / 20.0
        post_min_d1_Y = float(np.min(d1_Y_post)) / 30.0
        post_min_d1_X = float(np.min(d1_X_post)) / 30.0
        post_out_pitch = float(np.mean(cxv["out_pitch"][sl_fate]))
    else:
        post_d1_diff_XY, post_min_d1_Y, post_min_d1_X, post_out_pitch = 0.0, min_d1_Y, min_d1_X, 0.0

    nk_Y = (cxv["nk_h"] if y_lbl == HOME else cxv["nk_a"])[sl]
    nk_X = (cxv["nk_h"] if x_lbl == HOME else cxv["nk_a"])[sl]
    pk_Y = (cxv["pk_h"] if y_lbl == HOME else cxv["pk_a"])[sl]
    pk_X = (cxv["pk_h"] if x_lbl == HOME else cxv["pk_a"])[sl]
    cs_Y = (cxv["cs_h"] if y_lbl == HOME else cxv["cs_a"])[sl]
    cs_X = (cxv["cs_h"] if x_lbl == HOME else cxv["cs_a"])[sl]

    next_knot_diff_XY = float(np.mean(np.clip(nk_X - nk_Y, -20.0, 20.0))) / 20.0
    prev_knot_diff_XY = float(np.mean(np.clip(pk_X - pk_Y, -20.0, 20.0))) / 20.0
    chord_same_Y = float(np.mean(cs_Y))
    chord_same_X = float(np.mean(cs_X))
    chord_turn = float(np.mean(cxv["cturn"][sl]))
    gdisc = float(np.mean(cxv["gdisc"][sl])) / 30.0

    return np.round(
        np.array(
            [
                L / 80.0, l_flank, r_flank,
                mean_pY, max_pY, q75_pY, mean_pX, mean_pD, margin_YX, max_margin_YX, logodds_YX,
                tree_pY, tree_max_pY, tree_margin_YX, seq_pY, seq_margin_YX, pset_pY, pset_margin_YX,
                fam_vote_Y, fam_min_margin, tree_minus_seq,
                pre_pX, post_pX, post_pD,
                b4d_in_Y, b4d_in_fromD, b4d_out_X, b4d_out_D, pc_in, pc_out,
                imp_d2_max, imp_d4_max, imp_norm_d4_max, imp_in, imp_out,
                ang_max, cos_d2_min, cos_d4_min, decel_ratio, mean_v_seg, min_v_seg,
                frac_vis, eff_contact_cnt_Y, eff_contact_frac_Y, eff_contact_cnt_X,
                min_d1_Y, min_d1_X, mean_d1_diff_XY, min_d1_ig_Y, min_d1_ig_X, chord_flight,
                post_d1_diff_XY, post_min_d1_Y, post_min_d1_X, post_out_pitch,
                next_knot_diff_XY, prev_knot_diff_XY, chord_same_Y, chord_same_X, chord_turn, gdisc,
            ],
            dtype=np.float64,
        ),
        10,
    )


def is_real_physical_touch(f: np.ndarray) -> np.ndarray:
    """Boolean veto against absorbing genuine physical touches in Mode 1.

    Vetoes absorption if `Y` has >= 4 visible contact frames (`d1_Y < 1.3m`) and is closer
    than `X` by >= 0.15m, OR `max_pY >= 0.68`, OR spell duration `L >= 68` frames (`f[:, 0] >= 0.85`).
    """
    frac_vis = f[:, 41]
    cntY = f[:, 42] * 25.0
    d1Y = f[:, 45] * 30.0
    d1X = f[:, 46] * 30.0
    max_pY = f[:, 4]
    return ((frac_vis >= 0.35) & (cntY >= 4.0) & (d1Y < d1X - 0.15)) | (max_pY >= 0.68) | (f[:, 0] >= 0.85)


def touch_invalidity_score(f: np.ndarray, pat: str = "XYX") -> np.ndarray:
    """100% H<->A-invariant physical + consensus invalidity score for Mode 1 absorption."""
    s_prob = (
        (1.0 - f[:, 4]) * 0.30
        + (1.0 - f[:, 5]) * 0.25
        + (-f[:, 9]) * 0.20
        + (-f[:, 15]) * 0.15
        + (-f[:, 17]) * 0.10
    )
    s_kin = (
        0.25 * np.clip(f[:, 38] - 0.5, 0.0, 1.5)
        + 0.25 * (1.0 - np.clip(f[:, 42] * 2.5, 0.0, 1.0))
        + 0.25 * np.clip(f[:, 45] * 5.0 - f[:, 46] * 5.0, -0.5, 0.5)
        + 0.25 * (1.0 - np.clip(f[:, 31] * 2.5, 0.0, 1.0))
    )
    if pat == "XYX":
        s_fate = (
            0.35 * (1.0 - 0.5 * (f[:, 24] + f[:, 26]))
            + 0.35 * np.clip(-f[:, 51], -0.5, 0.5)
            + 0.30 * (f[:, 58] - f[:, 57])
        )
    elif pat == "XYD":
        s_fate = (
            0.40 * (1.0 - f[:, 24])
            + 0.30 * (1.0 - f[:, 28])
            + 0.30 * np.clip(-f[:, 55], -0.5, 0.5)
        )
    else:
        s_fate = (
            0.40 * (1.0 - f[:, 26])
            + 0.30 * (f[:, 22] - f[:, 3])
            + 0.30 * np.clip(-f[:, 51], -0.5, 0.5)
        )
    return 0.65 * s_prob + 0.20 * s_kin + 0.15 * s_fate


def rescue_validity_score(f: np.ndarray) -> np.ndarray:
    """100% H<->A-invariant physical + directional boundary rescue validity score for Mode 2."""
    frac_vis = f[:, 41]
    pc_in, pc_out = f[:, 28], f[:, 29]
    b4d_in_Y, b4d_out_X = f[:, 24], f[:, 26]
    min_d1_Y, min_d1_X = f[:, 45] * 30.0, f[:, 46] * 30.0
    min_d1_ig_Y = f[:, 48] * 30.0
    imp_in, cntY = f[:, 33], f[:, 42] * 25.0
    max_pY, tree_max_pY, seq_pY, pset_pY, L_s = f[:, 4], f[:, 12], f[:, 14], f[:, 16], f[:, 0]
    s_bdy = 0.35 * pc_in + 0.20 * pc_out + 0.25 * b4d_in_Y + 0.20 * b4d_out_X
    s_phys = (
        0.35 * np.clip(1.5 - min_d1_ig_Y, -0.5, 1.5)
        + 0.25 * np.clip((min_d1_X - min_d1_Y) / 3.0, -0.5, 0.8)
        + 0.20 * np.clip(imp_in * 2.0, 0.0, 1.0)
        + 0.20 * np.clip(cntY / 6.0, 0.0, 1.0)
    )
    s_prob = 0.35 * max_pY + 0.35 * tree_max_pY + 0.15 * seq_pY + 0.15 * pset_pY + 0.20 * np.clip(L_s, 0.0, 0.5)
    vis_ok = (min_d1_Y <= 1.25) & (pc_in >= 0.20) & (max_pY >= 0.40) & (tree_max_pY >= 0.44)
    blk_ok = (frac_vis < 0.3) & (pc_in >= 0.28) & (max_pY >= 0.48) & (seq_pY >= 0.42)
    gate = np.where(vis_ok | blk_ok, 0.0, -1.0)
    return 0.40 * s_bdy + 0.35 * s_phys + 0.25 * s_prob + gate


def fit_ridge_sub(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    sub_idx: Sequence[int],
    l2: float = 180.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a weighted ridge regression model on feature subset `sub_idx`."""
    Xs = X_tr[:, sub_idx]
    mu = np.average(Xs, axis=0, weights=w_tr)
    sd = np.sqrt(np.maximum(np.average((Xs - mu) ** 2, axis=0, weights=w_tr), 1e-6))
    Z = np.clip((Xs - mu) / sd, -6.0, 6.0)
    ym = float(np.average(y_tr, weights=w_tr))
    yc = y_tr - ym
    sw = np.sqrt(w_tr)
    Zw = Z * sw[:, None]
    yw = yc * sw
    coef = np.linalg.solve(Zw.T @ Zw + l2 * np.eye(Xs.shape[1]), Zw.T @ yw)
    return mu, sd, coef, ym


def score_ridge_raw(
    X: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    sub_idx: Sequence[int],
) -> np.ndarray:
    """Score linear ridge model on feature subset `sub_idx` without non-linear squashing."""
    mu, sd, coef, ym = model
    Xs = X[:, sub_idx]
    Z = np.clip((Xs - mu) / sd, -6.0, 6.0)
    return Z @ coef + ym


def _f1_from_spells(gt_spells: list, pred_spells: list) -> float:
    ov = pm.spell_overlaps(gt_spells, pred_spells)
    n_gt, n_pr = len(gt_spells), len(pred_spells)
    f1s = []
    for th in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(ov, n_gt, n_pr, th)
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return float(np.mean(f1s))


def extract_mode1_candidates(
    pred: np.ndarray,
    P: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame | dict[str, np.ndarray],
    p_change: np.ndarray,
    p_trans_4d: np.ndarray,
    fams: Sequence[np.ndarray] | None = None,
    gt: np.ndarray | None = None,
    max_len: int = 85,
    min_len: int = 5,
    match_id: str = "",
) -> list[dict[str, Any]]:
    """Extract Mode 1 (`XYX`, `XYD`, `DYX`) short episode candidates and optional GT F1 deltas."""
    cxv = preclean_tracking_arrays(X_df, len(pred))
    if gt is not None:
        gt_off = to_official(gt, half, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pred, half, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_f1 = _f1_from_spells(gt_spells, base_spells)
    else:
        gt_spells, base_spells, base_f1 = [], [], 0.0

    r = rle_runs_half(pred, half)
    out: list[dict[str, Any]] = []
    for k in range(1, len(r) - 1):
        a_prev, b_prev, l_lbl, h_prev = r[k - 1]
        a, b, y_lbl, h_cur = r[k]
        a_next, b_next, r_lbl, h_next = r[k + 1]
        if y_lbl not in (HOME, AWAY):
            continue
        L = b - a
        if L < min_len or L > max_len:
            continue
        if h_cur <= 0 or h_prev != h_cur or h_next != h_cur:
            continue
        x_lbl = 1 - y_lbl
        if l_lbl == x_lbl and r_lbl == x_lbl:
            pat = "XYX"
        elif l_lbl == x_lbl and r_lbl == DEAD:
            pat = "XYD"
        elif l_lbl == DEAD and r_lbl == x_lbl:
            pat = "DYX"
        else:
            continue

        feat = build_episode_features(a, b, y_lbl, x_lbl, a_prev, b_next, P, p_change, p_trans_4d, cxv, fams)
        delta_absorb = 0.0
        frac_x, frac_y, frac_d = 0.0, 0.0, 0.0
        if gt is not None:
            shift = 1 if h_cur == 2 else 0
            a_s, b_s = a + shift, b + shift
            mod_spells = []
            i_sp = 0
            while i_sp < len(base_spells):
                sa, sb, slbl = base_spells[i_sp]
                if sa == a_s and sb == b_s and slbl == y_lbl:
                    if pat == "XYX" and mod_spells and mod_spells[-1][1] == a_s and mod_spells[-1][2] == x_lbl:
                        if i_sp + 1 < len(base_spells) and base_spells[i_sp + 1][0] == b_s and base_spells[i_sp + 1][2] == x_lbl:
                            mod_spells[-1] = (mod_spells[-1][0], base_spells[i_sp + 1][1], x_lbl)
                            i_sp += 2
                            continue
                    elif pat == "XYD" and mod_spells and mod_spells[-1][1] == a_s and mod_spells[-1][2] == x_lbl:
                        mod_spells[-1] = (mod_spells[-1][0], b_s, x_lbl)
                        i_sp += 1
                        continue
                    elif pat == "DYX" and i_sp + 1 < len(base_spells) and base_spells[i_sp + 1][0] == b_s and base_spells[i_sp + 1][2] == x_lbl:
                        mod_spells.append((a_s, base_spells[i_sp + 1][1], x_lbl))
                        i_sp += 2
                        continue
                mod_spells.append((sa, sb, slbl))
                i_sp += 1
            delta_absorb = _f1_from_spells(gt_spells, mod_spells) - base_f1
            frac_x = float(np.mean(gt[a:b] == x_lbl))
            frac_y = float(np.mean(gt[a:b] == y_lbl))
            frac_d = float(np.mean(gt[a:b] == DEAD))

        out.append(
            {
                "m": match_id,
                "a": a,
                "b": b,
                "L": L,
                "y_lbl": y_lbl,
                "x_lbl": x_lbl,
                "a_prev": a_prev,
                "b_next": b_next,
                "pat": pat,
                "feat": feat,
                "delta_absorb": delta_absorb,
                "frac_x": frac_x,
                "frac_y": frac_y,
                "frac_d": frac_d,
            }
        )
    return out


def extract_mode2_candidates(
    pred: np.ndarray,
    P: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame | dict[str, np.ndarray],
    p_change: np.ndarray,
    p_trans_4d: np.ndarray,
    fams: Sequence[np.ndarray] | None = None,
    gt: np.ndarray | None = None,
    min_len: int = 8,
    max_len: int = 65,
    match_id: str = "",
) -> list[dict[str, Any]]:
    """Extract Mode 2 (short interception/clearance rescue inside long `X` spells) candidates."""
    cxv = preclean_tracking_arrays(X_df, len(pred))
    if gt is not None:
        gt_off = to_official(gt, half, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pred, half, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_f1 = _f1_from_spells(gt_spells, base_spells)
    else:
        gt_spells, base_spells, base_f1 = [], [], 0.0

    if fams is not None and len(fams) >= 3:
        tree_P = (fams[0] + fams[1] + fams[2]) / 3.0
    else:
        tree_P = P

    r = rle_runs_half(pred, half)
    out: list[dict[str, Any]] = []
    for sa, sb, x_lbl, h_cur in r:
        if x_lbl not in (HOME, AWAY) or h_cur <= 0:
            continue
        if sb - sa < 50:
            continue
        y_lbl = 1 - x_lbl
        col_to_Y = 0 if y_lbl == HOME else 1
        d1_ig_y = cxv["d1_ig_h"] if y_lbl == HOME else cxv["d1_ig_a"]
        sig = np.maximum(P[sa:sb, y_lbl], tree_P[sa:sb, y_lbl])
        bdy_sig = np.maximum(p_change[sa:sb], p_trans_4d[sa:sb, col_to_Y])
        active = (sig > 0.24) | ((sig > 0.18) & (d1_ig_y[sa:sb] < 1.6) & (bdy_sig > 0.12))
        if not np.any(active):
            continue

        idx = np.flatnonzero(active)
        splits = np.where(np.diff(idx) > 4)[0] + 1
        groups = np.split(idx, splits)
        for grp in groups:
            ca = int(sa + grp[0])
            cb = int(sa + grp[-1] + 1)
            if cb - ca < min_len:
                mid = (ca + cb) // 2
                ca = max(sa + 12, mid - 5)
                cb = min(sb - 12, mid + 5)
            if ca - sa < 12 or sb - cb < 12:
                continue
            L = cb - ca
            if L < min_len or L > max_len:
                continue
            if float(np.max(sig[ca - sa : cb - sa])) < 0.28:
                continue

            feat = build_episode_features(ca, cb, y_lbl, x_lbl, sa, sb, P, p_change, p_trans_4d, cxv, fams)
            delta_insert, frac_y, frac_x = 0.0, 0.0, 0.0
            if gt is not None:
                shift = 1 if h_cur == 2 else 0
                sa_s, sb_s = sa + shift, sb + shift
                ca_s, cb_s = ca + shift, cb + shift
                mod_spells = []
                for spa, spb, splbl in base_spells:
                    if spa == sa_s and spb == sb_s and splbl == x_lbl:
                        mod_spells.append((sa_s, ca_s, x_lbl))
                        mod_spells.append((ca_s, cb_s, y_lbl))
                        mod_spells.append((cb_s, sb_s, x_lbl))
                    else:
                        mod_spells.append((spa, spb, splbl))
                delta_insert = _f1_from_spells(gt_spells, mod_spells) - base_f1
                frac_y = float(np.mean(gt[ca:cb] == y_lbl))
                frac_x = float(np.mean(gt[ca:cb] == x_lbl))

            out.append(
                {
                    "m": match_id,
                    "a": ca,
                    "b": cb,
                    "L": L,
                    "y_lbl": y_lbl,
                    "x_lbl": x_lbl,
                    "a_prev": sa,
                    "b_next": sb,
                    "pat": "INS_Y",
                    "feat": feat,
                    "delta_insert": delta_insert,
                    "frac_y": frac_y,
                    "frac_x": frac_x,
                }
            )
    return out


class SymTouchTypeArbitrator:
    """100% H<->A-equivariant Controlled Touch vs Deflection/Interception Arbitrator (`EXP-15`)."""

    def __init__(self, cfg: TouchTypeArbitratorConfig | None = None):
        self.cfg = cfg if cfg is not None else TouchTypeArbitratorConfig()

    def fit_from_candidates(
        self,
        m1_by_match: dict[str, list[dict[str, Any]]],
        m2_by_match: dict[str, list[dict[str, Any]]] | None = None,
        train_matches: Sequence[str] | None = None,
    ) -> "SymTouchTypeArbitrator":
        """Fit and calibrate Mode 1 and Mode 2 thresholds using inner leave-one-match-out CV."""
        if train_matches is None:
            train_matches = sorted(m1_by_match.keys())
        inner_ms = list(train_matches)

        # 1. Calibrate Mode 1 per pattern ('XYX', 'XYD', 'DYX')
        self.cfg.mode1_params = {}
        if self.cfg.enable_mode1:
            for pat in ("XYX", "XYD", "DYX"):
                in_phy, in_rdg, in_dt, in_m, in_veto, mods_in = [], [], [], [], [], []
                for val_m in inner_ms:
                    tr_in = [r for m in inner_ms if m != val_m for r in m1_by_match[m] if r["pat"] == pat]
                    va_in = [r for r in m1_by_match[val_m] if r["pat"] == pat]
                    if len(tr_in) < 5 or len(va_in) == 0:
                        continue
                    X_tr = np.vstack([r["feat"] for r in tr_in])
                    y_tr = np.array(
                        [
                            0.65 * (1.0 if r["delta_absorb"] > 1e-5 else (-1.0 if r["delta_absorb"] < -1e-5 else 0.0))
                            + 0.35 * (r["frac_x"] - r["frac_y"])
                            for r in tr_in
                        ],
                        dtype=np.float64,
                    )
                    w_tr = np.array([1.0 + 300.0 * abs(r["delta_absorb"]) for r in tr_in], dtype=np.float64)
                    mod_in = fit_ridge_sub(X_tr, y_tr, w_tr, M1_SUB, l2=self.cfg.l2_reg)
                    mods_in.append(mod_in)
                    X_va = np.vstack([r["feat"] for r in va_in])
                    in_phy.extend(touch_invalidity_score(X_va, pat))
                    in_rdg.extend(score_ridge_raw(X_va, mod_in, M1_SUB))
                    in_dt.extend([r["delta_absorb"] for r in va_in])
                    in_m.extend([val_m for _ in va_in])
                    in_veto.extend(is_real_physical_touch(X_va))

                if not in_phy or not mods_in:
                    self.cfg.mode1_params[pat] = {"active": False}
                    continue

                in_phy_arr = np.asarray(in_phy, dtype=np.float64)
                in_rdg_arr = np.asarray(in_rdg, dtype=np.float64)
                in_dt_arr = np.asarray(in_dt, dtype=np.float64)
                in_m_arr = np.asarray(in_m)
                in_veto_arr = np.asarray(in_veto, dtype=bool)

                best_obj, best_w, best_th, best_uv = 0.0, 1.0, 999.0, True
                q_grid = np.linspace(70.0, 99.5, 60)
                n_folds = max(len(inner_ms), 1)
                for uv in (True, False):
                    v_mask = (~in_veto_arr) if uv else np.ones(len(in_veto_arr), dtype=bool)
                    for w_p in (0.0, 0.3, 0.5, 0.7, 1.0):
                        sc_in = w_p * in_phy_arr + (1.0 - w_p) * in_rdg_arr
                        gains_q, ths_q = [], []
                        for q in q_grid:
                            th = max(float(np.percentile(sc_in, q)), self.cfg.min_th_m1)
                            ths_q.append(th)
                            mask = (sc_in >= th) & v_mask
                            g = float(np.sum(in_dt_arr[mask])) / float(n_folds)
                            pm_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) > 1e-5)
                            neg_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) < -1e-5)
                            if pat == "XYX":
                                ok = pm_cnt >= neg_cnt + 1
                            else:
                                ok = (neg_cnt == 0) and (pm_cnt >= 3)
                            gains_q.append(g if ok else -0.01)
                        gains_smooth = np.convolve(gains_q, [0.25, 0.5, 0.25], mode="same")
                        idx_b = int(np.argmax(gains_smooth))
                        if gains_smooth[idx_b] > best_obj + 1e-4:
                            best_obj = float(gains_smooth[idx_b])
                            best_w = float(w_p)
                            best_th = float(ths_q[idx_b])
                            best_uv = bool(uv)

                min_gate = self.cfg.min_gate_xyx if pat == "XYX" else self.cfg.min_gate_xyd_dyx
                self.cfg.mode1_params[pat] = {
                    "active": bool(best_obj > min_gate),
                    "best_obj": best_obj,
                    "w_phys": best_w,
                    "th": best_th,
                    "use_veto": best_uv,
                    "models": mods_in,
                }

        # 2. Calibrate Mode 2 ('INS_Y')
        self.cfg.mode2_params = {"active": False}
        if self.cfg.enable_mode2 and m2_by_match is not None:
            in_phy, in_rdg, in_dt, in_m, mods_in = [], [], [], [], []
            for val_m in inner_ms:
                tr_in = [r for m in inner_ms if m != val_m for r in m2_by_match[m]]
                va_in = [r for r in m2_by_match[val_m]]
                if len(tr_in) < 5 or len(va_in) == 0:
                    continue
                X_tr = np.vstack([r["feat"] for r in tr_in])
                y_tr = np.array(
                    [
                        0.65 * (1.0 if r["delta_insert"] > 1e-5 else (-1.0 if r["delta_insert"] < -1e-5 else 0.0))
                        + 0.35 * (r["frac_y"] - r["frac_x"])
                        for r in tr_in
                    ],
                    dtype=np.float64,
                )
                w_tr = np.array([1.0 + 300.0 * abs(r["delta_insert"]) for r in tr_in], dtype=np.float64)
                mod_in = fit_ridge_sub(X_tr, y_tr, w_tr, M2_SUB, l2=self.cfg.l2_reg)
                mods_in.append(mod_in)
                X_va = np.vstack([r["feat"] for r in va_in])
                in_phy.extend(rescue_validity_score(X_va))
                in_rdg.extend(score_ridge_raw(X_va, mod_in, M2_SUB))
                in_dt.extend([r["delta_insert"] for r in va_in])
                in_m.extend([val_m for _ in va_in])

            if in_phy and mods_in:
                in_phy_arr = np.asarray(in_phy, dtype=np.float64)
                in_rdg_arr = np.asarray(in_rdg, dtype=np.float64)
                in_dt_arr = np.asarray(in_dt, dtype=np.float64)
                in_m_arr = np.asarray(in_m)
                best_obj, best_w, best_th = 0.0, 1.0, 999.0
                q_grid = np.linspace(85.0, 99.8, 50)
                n_folds = max(len(inner_ms), 1)
                for w_p in (0.3, 0.5, 0.7, 1.0):
                    sc_in = w_p * in_phy_arr + (1.0 - w_p) * in_rdg_arr
                    gains_q, ths_q = [], []
                    for q in q_grid:
                        th = max(float(np.percentile(sc_in, q)), self.cfg.min_th_m2)
                        ths_q.append(th)
                        mask = (sc_in >= th) & (in_phy_arr > 0.0)
                        g = float(np.sum(in_dt_arr[mask])) / float(n_folds)
                        pm_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) > 1e-5)
                        neg_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) < -1e-5)
                        ok = (neg_cnt == 0) and (pm_cnt >= 3)
                        gains_q.append(g if ok else -0.01)
                    gains_smooth = np.convolve(gains_q, [0.25, 0.5, 0.25], mode="same")
                    idx_b = int(np.argmax(gains_smooth))
                    if gains_smooth[idx_b] > best_obj + 1e-4:
                        best_obj = float(gains_smooth[idx_b])
                        best_w = float(w_p)
                        best_th = float(ths_q[idx_b])
                self.cfg.mode2_params = {
                    "active": bool(best_obj > self.cfg.min_gate_m2),
                    "best_obj": best_obj,
                    "w_phys": best_w,
                    "th": best_th,
                    "models": mods_in,
                }
        return self

    def apply(
        self,
        pred: np.ndarray,
        P: np.ndarray,
        half: np.ndarray,
        X_df: pd.DataFrame | dict[str, np.ndarray],
        p_change: np.ndarray,
        p_trans_4d: np.ndarray,
        fams: Sequence[np.ndarray] | None = None,
        m1_cands: list[dict[str, Any]] | None = None,
        m2_cands: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Apply calibrated Mode 1 and Mode 2 arbitration to a predicted label sequence."""
        out = np.asarray(pred, dtype=np.int64).copy()

        # Mode 1: Involuntary Rebound / Deflection Absorption
        if self.cfg.enable_mode1 and any(v.get("active", False) for v in self.cfg.mode1_params.values()):
            if m1_cands is None:
                m1_cands = extract_mode1_candidates(
                    out, P, half, X_df, p_change, p_trans_4d, fams,
                    max_len=self.cfg.max_len_m1, min_len=self.cfg.min_len_m1,
                )
            for pat in ("XYX", "XYD", "DYX"):
                p_cfg = self.cfg.mode1_params.get(pat, {})
                if not p_cfg.get("active", False):
                    continue
                pat_cands = [r for r in m1_cands if r["pat"] == pat]
                if not pat_cands:
                    continue
                X_te = np.vstack([r["feat"] for r in pat_cands])
                w_p = float(p_cfg["w_phys"])
                th = float(p_cfg["th"])
                use_veto = bool(p_cfg["use_veto"])
                mods = p_cfg["models"]
                sc_rdg = np.mean([score_ridge_raw(X_te, md, M1_SUB) for md in mods], axis=0)
                sc_te = w_p * touch_invalidity_score(X_te, pat) + (1.0 - w_p) * sc_rdg
                veto_te = is_real_physical_touch(X_te) if use_veto else np.zeros(len(pat_cands), dtype=bool)
                cands_fire = sorted(
                    [(float(s), r) for r, s, vt in zip(pat_cands, sc_te, veto_te) if (s >= th) and (not vt)],
                    key=lambda x: (-x[0], x[1]["a"]),
                )
                for _, r in cands_fire:
                    a, b, x_lbl, y_lbl = r["a"], r["b"], r["x_lbl"], r["y_lbl"]
                    if np.all(out[a:b] == y_lbl):
                        out[a:b] = x_lbl

        # Mode 2: Real Interception / Clearance Rescue
        m2_cfg = self.cfg.mode2_params
        if self.cfg.enable_mode2 and m2_cfg.get("active", False):
            if m2_cands is None:
                m2_cands = extract_mode2_candidates(
                    pred, P, half, X_df, p_change, p_trans_4d, fams,
                    min_len=self.cfg.min_len_m2, max_len=self.cfg.max_len_m2,
                )
            if m2_cands:
                X_te = np.vstack([r["feat"] for r in m2_cands])
                w_p = float(m2_cfg["w_phys"])
                th = float(m2_cfg["th"])
                mods = m2_cfg["models"]
                phy_te = rescue_validity_score(X_te)
                rdg_te = np.mean([score_ridge_raw(X_te, md, M2_SUB) for md in mods], axis=0)
                sc_te = w_p * phy_te + (1.0 - w_p) * rdg_te
                cands_fire = sorted(
                    [(float(s), r) for r, s, p in zip(m2_cands, sc_te, phy_te) if (s >= th) and (p > 0.0)],
                    key=lambda x: (-x[0], x[1]["a"]),
                )
                for _, r in cands_fire:
                    a, b, x_lbl, y_lbl = r["a"], r["b"], r["x_lbl"], r["y_lbl"]
                    if np.all(out[a:b] == x_lbl):
                        out[a:b] = y_lbl

        return out


def apply_touch_type_arbitrator(
    pred: np.ndarray,
    P: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame | dict[str, np.ndarray],
    p_change: np.ndarray,
    p_trans_4d: np.ndarray,
    fams: Sequence[np.ndarray] | None = None,
    arbitrator: SymTouchTypeArbitrator | None = None,
) -> np.ndarray:
    """Convenience wrapper to apply `SymTouchTypeArbitrator`."""
    if arbitrator is None:
        return np.asarray(pred, dtype=np.int64).copy()
    return arbitrator.apply(pred, P, half, X_df, p_change, p_trans_4d, fams)


def swap_tracking_dict(cxv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Swap HOME <-> AWAY channels inside a precleaned tracking dictionary."""
    sw = dict(cxv)
    pairs = [
        ("d1_ig_h", "d1_ig_a"),
        ("d1_h", "d1_a"),
        ("d1_h_vis", "d1_a_vis"),
        ("nk_h", "nk_a"),
        ("pk_h", "pk_a"),
        ("cs_h", "cs_a"),
    ]
    for kh, ka in pairs:
        if kh in sw and ka in sw:
            sw[kh], sw[ka] = cxv[ka].copy(), cxv[kh].copy()
    return sw


def verify_touch_arbitrator_equivariance(
    pred: np.ndarray,
    P: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame | dict[str, np.ndarray],
    p_change: np.ndarray,
    p_trans_4d: np.ndarray,
    fams: Sequence[np.ndarray],
    arbitrator: SymTouchTypeArbitrator,
) -> tuple[float, int]:
    """Verify exact 100% H <-> A equivariance (`0.000e+00`) of features, scores, and predictions."""
    cxv = preclean_tracking_arrays(X_df, len(pred))
    cxv_sw = swap_tracking_dict(cxv)

    pred_sw = np.where(pred == HOME, AWAY, np.where(pred == AWAY, HOME, DEAD))
    P_sw = P[:, [1, 0, 2]].copy()
    p4d_sw = p_trans_4d[:, [1, 0, 2, 3]].copy()
    fams_sw = [f[:, [1, 0, 2]].copy() for f in fams]

    cands_orig = extract_mode1_candidates(pred, P, half, cxv, p_change, p_trans_4d, fams)
    cands_swap = extract_mode1_candidates(pred_sw, P_sw, half, cxv_sw, p_change, p4d_sw, fams_sw)

    max_feat_diff = 0.0
    if cands_orig:
        F_orig = np.vstack([c["feat"] for c in cands_orig])
        F_swap = np.vstack([c["feat"] for c in cands_swap])
        max_feat_diff = float(np.max(np.abs(F_orig - F_swap)))
        for pat in ("XYX", "XYD", "DYX"):
            s1 = touch_invalidity_score(F_orig, pat)
            s2 = touch_invalidity_score(F_swap, pat)
            max_feat_diff = max(max_feat_diff, float(np.max(np.abs(s1 - s2))))

    out_orig = arbitrator.apply(pred, P, half, cxv, p_change, p_trans_4d, fams)
    out_swap = arbitrator.apply(pred_sw, P_sw, half, cxv_sw, p_change, p4d_sw, fams_sw)
    out_swap_back = np.where(out_swap == HOME, AWAY, np.where(out_swap == AWAY, HOME, DEAD))
    n_mismatch = int(np.sum(out_orig != out_swap_back))
    return max_feat_diff, n_mismatch
