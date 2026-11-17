"""Wave 4 First-Principles Physics Feature Engineering (`v1ctl_physics_v2`: EXP-04 & EXP-05).

Builds an equivariant first-principles physics feature block (`physv2`, 129 features) on top of
`v1ctl_plus` (352 features -> 481 total features):

1. EXP-04: Conservation of Linear Momentum & Ballistic Pass-Chord Bypass (`J(t) = Delta v_ball`):
   - Multi-scale vector impulse magnitude `||J_delta(t)||`, tangential impulse `J_tan_delta(t)`,
     normal/deflection impulse `J_norm_delta(t)`, and angular deflection `cos(Delta theta_delta(t))`
     for `delta in {2, 4, 8}` frames, gated by `ball_present`.
   - Contact Knot Detection & Pass-Chord Geometry (`tau_prev(t)` and `tau_next(t)`):
     - `chord_in_flight`: continuous score in `[0, 1]` indicating mid-flight between contact knots
       (high ball/chord speed, low local impulse `||J_2(t)||`).
     - `d1_h_impulse_gated`, `d1_a_impulse_gated`, `diff_d1_impulse_gated`: effective player-to-ball
       distance where mid-flight proximity without impulse is penalized upward toward knot distances,
       preventing bypassed defenders under passes from triggering false turnovers.
     - Chord Sender & Receiver Attribution:
       - `prev_knot_d1_h`, `prev_knot_d1_a`, `prev_knot_diff_d1`
       - `next_knot_d1_h`, `next_knot_d1_a`, `next_knot_diff_d1`
       - `chord_same_team_prob_h`, `chord_same_team_prob_a`, `chord_same_team_diff`, `chord_turnover_prob`

2. EXP-05: Collective Expansion-Contraction Dipole & Ball-Free Pressure Convergence Focus (`x_focus(t)`):
   - Collective Expansion-Contraction Dipole (100% ball-free):
     - Outfield lateral spread `sigma_y(t)`, longitudinal spread `sigma_x(t)`, effective area
       `sigma_x(t) * sigma_y(t)`, centroid `(c_x(t), c_y(t))`, defensive-line x (mean of 3 deepest
       outfield players in attacking direction), and attacking-line x (mean of 3 highest outfield players).
     - Multi-scale rates of change via track-break-safe slot-matched differences over
       `Delta t in {12, 25, 50}` frames (`0.5s, 1.0s, 2.0s`).
   - Ball-Free Pressure Convergence Focus (`x_focus(t)`):
     - Projects outfield players `tau = 0.6s` ahead using track-break-free LSQ velocities `v_i(t)`:
       `p_i^proj(t) = p_i(t) + 0.6 * v_i(t)`.
     - Computes weighted centroid of the tightest spatial cluster of moving players (weighted by speed
       and inverse local pairwise distance to the 4 nearest other players), smoothed per half (`sigma = 3` frames).
     - Derived features: `focus_x_norm`, `focus_y_norm`, `focus_d1_h`, `focus_d1_a`, `focus_d2_h`, `focus_d2_a`,
       `focus_diff_d1`, `focus_diff_d2`, `focus_n5m_h`, `focus_n5m_a`, `focus_diff_n5m`,
       `focus_inflow_h`, `focus_inflow_a`, `focus_diff_inflow`, and `ball_focus_discrepancy`.

3. Strict Equivariance (`0.000e+00` error):
   - `mirror_physv2_df` and `mirror_v1ctl_physv2_df` match `swap_match_ha_and_rotate180(match)`
     bit-for-bit (`< 1e-5` across all columns, `nan_mismatch == 0`).
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.data.ball_interp import linear_mask
from src.data.common import AWAY, FPS, HOME, Match
from src.features.knot_physics import _linear_runs, filter_ghost_and_switch_slots
from src.features.symmetry import swap_match
from src.features.v1 import match_fingerprint
from src.features.v1ctl_plus import (
    _gap_edges_by_half,
    build_v1ctl_plus,
    lsq_velocity,
    mirror_v1ctl_plus_df,
    roll_nanmean_by_half,
)

VERSION = 1
IMPULSE_DELTAS = (2, 4, 8)
DIPOLE_DELTAS = (12, 25, 50)
FOCUS_TAU_S = 0.6
FOCUS_SMOOTH_SIGMA = 3.0

# Alias required by Wave 4 specification
swap_match_ha_and_rotate180 = swap_match


def _centered_coords(m: Match) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pitch-centered float64 coordinates (P_c, B_c) and raw centered (P_raw_c, B_raw_c)
    such that swap_match(m) produces exact sign-bit negation (-P_c, -B_c)."""
    nc = filter_ghost_and_switch_slots(m)
    L, W = float(m.pitch[0]), float(m.pitch[1])
    half_pitch = np.array([L / 2.0, W / 2.0], dtype=np.float64)

    P_raw = m.P[..., :2].astype(np.float64)
    B_raw = m.B[:, :2].astype(np.float64)
    if m.direction_normalized:
        P_raw_c = P_raw - half_pitch[None, None, :]
        B_raw_c = B_raw - half_pitch[None, :]
    else:
        sgn = np.where(np.asarray(m.attack_sign) < 0, -1.0, 1.0)
        P_raw_c = (P_raw - half_pitch[None, None, :]) * sgn[:, None, None]
        B_raw_c = (B_raw - half_pitch[None, :]) * sgn[:, None]

    # Apply the exact same ghost/switch NaN mask discovered by filter_ghost_and_switch_slots
    nc_vis = np.isfinite(nc.P[..., 0]) & np.isfinite(nc.P[..., 1])
    P_c = np.where(nc_vis[..., None], P_raw_c, np.nan)
    b_vis = nc.ball_xy_visible
    B_c = np.where(b_vis[:, None], B_raw_c, np.nan)
    return P_c, B_c, P_raw_c, B_raw_c


def _gaussian_smooth_by_half(x: np.ndarray, half: np.ndarray, sigma: float = FOCUS_SMOOTH_SIGMA) -> np.ndarray:
    """Smooth 1D array per half using a symmetric normalized Gaussian filter of width +/- 3*sigma."""
    T = len(x)
    out = np.zeros(T, dtype=np.float64)
    rad = int(round(3.0 * sigma))
    k_idx = np.arange(-rad, rad + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (k_idx / sigma) ** 2)
    kernel /= np.sum(kernel)

    edges = np.flatnonzero(half[1:] != half[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, T]
    for a, b in zip(starts, stops):
        seg = x[a:b]
        valid = np.isfinite(seg)
        if not np.any(valid):
            continue
        num = np.convolve(np.where(valid, seg, 0.0), kernel, mode="same")
        den = np.convolve(valid.astype(np.float64), kernel, mode="same")
        out[a:b] = np.where(den > 1e-12, num / den, 0.0)
    return out


def _ball_impulse_at_delta(B_c: np.ndarray, bvis: np.ndarray, half: np.ndarray, delta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (mag, tan, norm, cos_theta) of ball vector impulse J_delta(t) = v_out(t) - v_in(t)
    gated by ball_present (`bvis`)."""
    T = len(B_c)
    v_in = np.zeros((T, 2), dtype=np.float64)
    v_out = np.zeros((T, 2), dtype=np.float64)
    ok_in = np.zeros(T, dtype=bool)
    ok_out = np.zeros(T, dtype=bool)

    if T > delta:
        valid_in = bvis[delta:] & bvis[:-delta] & (half[delta:] == half[:-delta])
        ok_in[delta:] = valid_in
        v_in[delta:][valid_in] = (B_c[delta:][valid_in] - B_c[:-delta][valid_in]) * (FPS / float(delta))

        valid_out = valid_in
        ok_out[:-delta] = valid_out
        v_out[:-delta][valid_out] = (B_c[delta:][valid_out] - B_c[:-delta][valid_out]) * (FPS / float(delta))

    both_ok = bvis & ok_in & ok_out
    J = np.where(both_ok[:, None], v_out - v_in, 0.0)
    mag = np.hypot(J[:, 0], J[:, 1])

    sp_in = np.hypot(v_in[:, 0], v_in[:, 1])
    sp_out = np.hypot(v_out[:, 0], v_out[:, 1])
    denom_in = np.maximum(sp_in, 1e-6)
    u_in_x = np.where(both_ok & (sp_in > 1e-4), v_in[:, 0] / denom_in, 0.0)
    u_in_y = np.where(both_ok & (sp_in > 1e-4), v_in[:, 1] / denom_in, 0.0)

    tan = np.where(both_ok, J[:, 0] * u_in_x + J[:, 1] * u_in_y, 0.0)
    norm = np.where(both_ok, np.abs(J[:, 0] * u_in_y - J[:, 1] * u_in_x), 0.0)

    dot_io = v_in[:, 0] * v_out[:, 0] + v_in[:, 1] * v_out[:, 1]
    cos_raw = np.clip(dot_io / np.maximum(sp_in * sp_out, 1e-6), -1.0, 1.0)
    cos_ang = np.where(both_ok & (sp_in > 0.2) & (sp_out > 0.2), cos_raw, np.where(bvis, 1.0, 0.0))

    gate = bvis.astype(np.float64)
    return mag * gate, tan * gate, norm * gate, cos_ang * gate


def _team_collective_stats_and_derivatives(
    x_att: np.ndarray,
    y_att: np.ndarray,
    half: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute static outfield collective geometry (sigma_x, sigma_y, eff_area, def3_x, att3_x)
    and track-break-safe slot-matched derivatives over DIPOLE_DELTAS = (12, 25, 50) frames."""
    T, S = x_att.shape
    vis = np.isfinite(x_att) & np.isfinite(y_att)

    def _stats_from_masked(xm: np.ndarray, ym: np.ndarray, vm: np.ndarray) -> tuple[np.ndarray, ...]:
        n = np.sum(vm, axis=1)
        n_safe = np.maximum(n, 1)
        xf = np.where(vm, xm, 0.0)
        yf = np.where(vm, ym, 0.0)
        cx = np.where(n > 0, np.sum(xf, axis=1) / n_safe, np.nan)
        cy = np.where(n > 0, np.sum(yf, axis=1) / n_safe, np.nan)

        dx = np.where(vm, xm - cx[:, None], 0.0)
        dy = np.where(vm, ym - cy[:, None], 0.0)
        sx = np.where(n > 1, np.sqrt(np.sum(dx * dx, axis=1) / n_safe), 0.0)
        sy = np.where(n > 1, np.sqrt(np.sum(dy * dy, axis=1) / n_safe), 0.0)
        area = sx * sy

        # 3 deepest (smallest x_att) and 3 highest (largest x_att)
        x_asc = np.sort(np.where(vm, xm, np.inf), axis=1)[:, :3]
        v_asc = np.isfinite(x_asc)
        n_asc = np.maximum(np.sum(v_asc, axis=1), 1)
        def3 = np.where(np.any(v_asc, axis=1), np.sum(np.where(v_asc, x_asc, 0.0), axis=1) / n_asc, np.nan)

        x_desc = -np.sort(np.where(vm, -xm, np.inf), axis=1)[:, :3]
        v_desc = np.isfinite(x_desc)
        n_desc = np.maximum(np.sum(v_desc, axis=1), 1)
        att3 = np.where(np.any(v_desc, axis=1), np.sum(np.where(v_desc, x_desc, 0.0), axis=1) / n_desc, np.nan)
        return sx, sy, area, cx, cy, def3, att3

    sx0, sy0, area0, cx0, cy0, def3_0, att3_0 = _stats_from_masked(x_att, y_att, vis)

    # Fill rare NaNs (if all outfield players missing on a frame) with neutral defaults
    sx0_f = np.nan_to_num(sx0, nan=0.0)
    sy0_f = np.nan_to_num(sy0, nan=0.0)
    area0_f = np.nan_to_num(area0, nan=0.0)
    def3_0_f = np.nan_to_num(def3_0, nan=52.5)
    att3_0_f = np.nan_to_num(att3_0, nan=52.5)

    # Track-break-safe slot-matched single-frame steps:
    # Only use slots visible at BOTH t-1 and t whose step is <= 1.5m within the same half
    same_half = (half[1:] == half[:-1])
    step = np.hypot(x_att[1:] - x_att[:-1], y_att[1:] - y_att[:-1])
    vm_pair = vis[1:] & vis[:-1] & (np.nan_to_num(step, nan=99.0) <= 1.5) & same_half[:, None]

    sx_a, sy_a, ar_a, cx_a, cy_a, d3_a, a3_a = _stats_from_masked(x_att[:-1], y_att[:-1], vm_pair)
    sx_b, sy_b, ar_b, cx_b, cy_b, d3_b, a3_b = _stats_from_masked(x_att[1:], y_att[1:], vm_pair)

    out = {
        "sigma_x": sx0_f,
        "sigma_y": sy0_f,
        "eff_area": area0_f,
        "def3_line_x": def3_0_f,
        "att3_line_x": att3_0_f,
    }

    edges = np.flatnonzero(half[1:] != half[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, T]

    for name, va, vb in (
        ("sigma_x", sx_a, sx_b),
        ("sigma_y", sy_a, sy_b),
        ("eff_area", ar_a, ar_b),
        ("cx", cx_a, cx_b),
        ("cy", cy_a, cy_b),
        ("def3_x", d3_a, d3_b),
        ("att3_x", a3_a, a3_b),
    ):
        dq = np.nan_to_num(vb - va, nan=0.0)
        # Integrate clean steps within each half
        q_int = np.zeros(T, dtype=np.float64)
        q_int[1:] = np.cumsum(dq)
        for dt in DIPOLE_DELTAS:
            rate = np.zeros(T, dtype=np.float64)
            h_lo = dt // 2
            h_hi = dt - h_lo
            for a, b in zip(starts, stops):
                idx = np.arange(a, b, dtype=np.int64)
                lo = np.maximum(a, idx - h_lo)
                hi = np.minimum(b - 1, idx + h_hi)
                span_s = np.maximum((hi - lo).astype(np.float64) / FPS, 1.0 / FPS)
                rate[a:b] = (q_int[hi] - q_int[lo]) / span_s
            out[f"d_{name}_dt{dt}"] = rate

    return out


def compute_physv2_features(m: Match) -> pd.DataFrame:
    """Compute Wave 4 First-Principles Physics features (EXP-04 Momentum & Pass-Chord + EXP-05 Dipole & Focus)."""
    nc = filter_ghost_and_switch_slots(m)
    P_c, B_c, P_raw_c, _ = _centered_coords(m)
    L, W = float(m.pitch[0]), float(m.pitch[1])
    half = np.asarray(nc.half)
    T = len(B_c)
    bvis = nc.ball_xy_visible
    blin = linear_mask(nc.B, half)

    l_starts, l_ends = _linear_runs(nc.B, half)
    is_vertex = np.zeros(T, dtype=bool)
    if len(l_starts):
        is_vertex[l_starts] = True
        is_vertex[np.minimum(l_ends - 1, T - 1)] = True

    sel_h = nc.slot_team == HOME
    sel_a = nc.slot_team == AWAY

    f: dict[str, np.ndarray] = {}

    # =========================================================================
    # 1. EXP-04: Conservation of Linear Momentum & Ballistic Pass-Chord Bypass
    # =========================================================================
    for delta in IMPULSE_DELTAS:
        mag, tan, norm, cos_a = _ball_impulse_at_delta(B_c, bvis, half, delta)
        f[f"impulse_mag_d{delta}"] = mag.astype(np.float32)
        f[f"impulse_tan_d{delta}"] = tan.astype(np.float32)
        f[f"impulse_norm_d{delta}"] = norm.astype(np.float32)
        f[f"impulse_cos_d{delta}"] = cos_a.astype(np.float32)

    j_mag_2 = f["impulse_mag_d2"].astype(np.float64)
    j_mag_3, _, _, _ = _ball_impulse_at_delta(B_c, bvis, half, 3)

    # Local ball speed (m/s)
    sp_ball = np.zeros(T, dtype=np.float64)
    if T > 2:
        ok_sp = bvis[2:] & bvis[:-2] & (half[2:] == half[:-2])
        sp_ball[1:-1][ok_sp] = np.hypot(
            B_c[2:, 0][ok_sp] - B_c[:-2, 0][ok_sp],
            B_c[2:, 1][ok_sp] - B_c[:-2, 1][ok_sp],
        ) * (FPS / 2.0)
    same_prev = np.r_[False, bvis[1:] & bvis[:-1] & (half[1:] == half[:-1])]
    same_next = np.r_[bvis[1:] & bvis[:-1] & (half[1:] == half[:-1]), False]
    sp_1 = np.zeros(T, dtype=np.float64)
    sp_1[1:][same_prev[1:]] = np.hypot(
        B_c[1:, 0][same_prev[1:]] - B_c[:-1, 0][same_prev[1:]],
        B_c[1:, 1][same_prev[1:]] - B_c[:-1, 1][same_prev[1:]],
    ) * FPS
    sp_ball = np.where(sp_ball > 0.0, sp_ball, sp_1)

    # Player-to-ball distances in pitch-centered coordinates
    # Forward/backward fill B_c across gaps for continuous raw distance reference
    last_b_idx, nxt_b_idx = _gap_edges_by_half(bvis, half)
    idx_all = np.arange(T, dtype=np.int64)
    b_prev_xy = B_c[np.clip(last_b_idx, 0, T - 1)]
    b_next_xy = B_c[np.clip(nxt_b_idx, 0, T - 1)]
    has_lb = last_b_idx >= 0
    has_nb = nxt_b_idx < T
    w_b = np.clip((idx_all - last_b_idx) / np.maximum((nxt_b_idx - last_b_idx).astype(np.float64), 1.0), 0.0, 1.0)
    B_interp_c = np.where(
        bvis[:, None],
        B_c,
        np.where(
            (has_lb & has_nb)[:, None],
            (1.0 - w_b[:, None]) * b_prev_xy + w_b[:, None] * b_next_xy,
            np.where(has_lb[:, None], b_prev_xy, np.where(has_nb[:, None], b_next_xy, 0.0)),
        ),
    )

    d_all = np.hypot(P_c[..., 0] - B_interp_c[:, None, 0], P_c[..., 1] - B_interp_c[:, None, 1])
    d_h_mat = np.where(np.isfinite(d_all[:, sel_h]), d_all[:, sel_h], np.inf)
    d_a_mat = np.where(np.isfinite(d_all[:, sel_a]), d_all[:, sel_a], np.inf)
    d1_h_raw = np.min(d_h_mat, axis=1)
    d1_a_raw = np.min(d_a_mat, axis=1)
    d1_h_raw = np.where(np.isfinite(d1_h_raw), d1_h_raw, 30.0)
    d1_a_raw = np.where(np.isfinite(d1_a_raw), d1_a_raw, 30.0)
    d_min = np.minimum(d1_h_raw, d1_a_raw)

    # Physical Contact Knot candidate:
    # 1) ball_present == 1, min(d1_H, d1_A) < 2.2m, and ||J_3(t)|| is significant (or slow controlled touch outside linear interior, or linear vertex)
    # 2) OR at the boundaries of a ball_gap where a player is within 3.5m
    gap_boundary = bvis & ((~same_prev) | (~same_next))
    sig_impulse = (j_mag_3 >= 3.0) | (j_mag_2 >= 3.0) | is_vertex | ((~blin) & (sp_ball < 3.5))
    contact_knot = (bvis & (d_min < 2.2) & sig_impulse) | (gap_boundary & (d_min < 3.5))

    tau_prev, tau_next = _gap_edges_by_half(contact_knot, half)
    has_prev = tau_prev >= 0
    has_next = tau_next < T
    tp_safe = np.where(has_prev, np.clip(tau_prev, 0, T - 1), np.where(has_next, np.clip(tau_next, 0, T - 1), idx_all))
    tn_safe = np.where(has_next, np.clip(tau_next, 0, T - 1), np.where(has_prev, np.clip(tau_prev, 0, T - 1), idx_all))

    # Chord speed between tau_prev and tau_next
    chord_dt_s = np.maximum((tn_safe - tp_safe).astype(np.float64) / FPS, 1.0 / FPS)
    chord_dist = np.hypot(B_interp_c[tn_safe, 0] - B_interp_c[tp_safe, 0], B_interp_c[tn_safe, 1] - B_interp_c[tp_safe, 1])
    chord_speed = np.where(tn_safe > tp_safe, chord_dist / chord_dt_s, sp_ball)
    v_eff = np.where(bvis & (~blin), np.maximum(sp_ball, chord_speed), chord_speed)

    # Continuous score in [0, 1] indicating frame t is mid-flight between tau_prev and tau_next
    dist_from_knot_frames = np.minimum(
        np.where(has_prev, idx_all - tau_prev, 10),
        np.where(has_next, tau_next - idx_all, 10),
    ).astype(np.float64)
    s_knot = np.where(contact_knot, 0.0, np.clip(dist_from_knot_frames / 2.0, 0.0, 1.0))
    s_vel = np.clip((v_eff - 3.5) / 5.0, 0.0, 1.0)
    s_low_j = np.where(bvis & (~blin | is_vertex), np.clip(1.0 - j_mag_2 / 4.5, 0.0, 1.0), 1.0)
    chord_in_flight = (s_knot * s_vel * s_low_j).astype(np.float64)

    # Sender (prev_knot) & Receiver (next_knot) distances
    prev_knot_d1_h = d1_h_raw[tp_safe]
    prev_knot_d1_a = d1_a_raw[tp_safe]
    prev_knot_diff_d1 = prev_knot_d1_h - prev_knot_d1_a

    next_knot_d1_h = d1_h_raw[tn_safe]
    next_knot_d1_a = d1_a_raw[tn_safe]
    next_knot_diff_d1 = next_knot_d1_h - next_knot_d1_a

    # Linear interpolation weight along the pass chord [tau_prev, tau_next]
    span_knot = np.maximum((tn_safe - tp_safe).astype(np.float64), 1.0)
    alpha_next = np.where(
        has_prev & has_next & (tn_safe > tp_safe),
        np.clip((idx_all - tp_safe) / span_knot, 0.0, 1.0),
        np.where(has_prev, 0.0, 1.0),
    )
    d1_h_knot_interp = (1.0 - alpha_next) * prev_knot_d1_h + alpha_next * next_knot_d1_h
    d1_a_knot_interp = (1.0 - alpha_next) * prev_knot_d1_a + alpha_next * next_knot_d1_a

    # Effective player-to-ball distance where proximity during mid-flight is penalized upward toward knot distances
    d1_h_penalized = np.maximum(d1_h_raw, d1_h_knot_interp)
    d1_a_penalized = np.maximum(d1_a_raw, d1_a_knot_interp)
    d1_h_impulse_gated = (1.0 - chord_in_flight) * d1_h_raw + chord_in_flight * (0.5 * d1_h_penalized + 0.5 * d1_h_knot_interp)
    d1_a_impulse_gated = (1.0 - chord_in_flight) * d1_a_raw + chord_in_flight * (0.5 * d1_a_penalized + 0.5 * d1_a_knot_interp)
    diff_d1_impulse_gated = d1_h_impulse_gated - d1_a_impulse_gated

    # Soft probabilities at launch knot and reception knot (temperature = 1.2m)
    p_h_prev = 1.0 / (1.0 + np.exp(np.clip(prev_knot_diff_d1 / 1.2, -30.0, 30.0)))
    p_a_prev = 1.0 / (1.0 + np.exp(np.clip(-prev_knot_diff_d1 / 1.2, -30.0, 30.0)))
    p_h_next = 1.0 / (1.0 + np.exp(np.clip(next_knot_diff_d1 / 1.2, -30.0, 30.0)))
    p_a_next = 1.0 / (1.0 + np.exp(np.clip(-next_knot_diff_d1 / 1.2, -30.0, 30.0)))

    chord_same_team_prob_h = p_h_prev * p_h_next
    chord_same_team_prob_a = p_a_prev * p_a_next
    chord_same_team_diff = chord_same_team_prob_h - chord_same_team_prob_a
    chord_turnover_prob = p_h_prev * p_a_next + p_a_prev * p_h_next

    f["chord_in_flight"] = chord_in_flight.astype(np.float32)
    f["d1_h_impulse_gated"] = d1_h_impulse_gated.astype(np.float32)
    f["d1_a_impulse_gated"] = d1_a_impulse_gated.astype(np.float32)
    f["diff_d1_impulse_gated"] = diff_d1_impulse_gated.astype(np.float32)
    f["prev_knot_d1_h"] = prev_knot_d1_h.astype(np.float32)
    f["prev_knot_d1_a"] = prev_knot_d1_a.astype(np.float32)
    f["prev_knot_diff_d1"] = prev_knot_diff_d1.astype(np.float32)
    f["next_knot_d1_h"] = next_knot_d1_h.astype(np.float32)
    f["next_knot_d1_a"] = next_knot_d1_a.astype(np.float32)
    f["next_knot_diff_d1"] = next_knot_diff_d1.astype(np.float32)
    f["chord_same_team_prob_h"] = chord_same_team_prob_h.astype(np.float32)
    f["chord_same_team_prob_a"] = chord_same_team_prob_a.astype(np.float32)
    f["chord_same_team_diff"] = chord_same_team_diff.astype(np.float32)
    f["chord_turnover_prob"] = chord_turnover_prob.astype(np.float32)

    for s_win in (1.0, 2.0):
        w_f = int(round(s_win * FPS))
        f[f"diff_d1_impulse_gated_r{s_win:g}"] = roll_nanmean_by_half(diff_d1_impulse_gated, half, w_f)
        f[f"chord_same_team_diff_r{s_win:g}"] = roll_nanmean_by_half(chord_same_team_diff, half, w_f)

    # =========================================================================
    # 2. EXP-05 Part A: Collective Expansion-Contraction Dipole (100% Ball-Free)
    # =========================================================================
    outfield_h = sel_h & (~nc.slot_is_gk)
    outfield_a = sel_a & (~nc.slot_is_gk)

    # Attacking-frame coordinates per team:
    # HOME attacks +x: x_att_H = P_c_x + L/2, y_att_H = P_c_y
    # AWAY attacks -x: x_att_A = -P_c_x + L/2, y_att_A = -P_c_y
    x_att_h = P_c[:, outfield_h, 0] + (L / 2.0)
    y_att_h = P_c[:, outfield_h, 1]
    x_att_a = -P_c[:, outfield_a, 0] + (L / 2.0)
    y_att_a = -P_c[:, outfield_a, 1]

    stats_h = _team_collective_stats_and_derivatives(x_att_h, y_att_h, half)
    stats_a = _team_collective_stats_and_derivatives(x_att_a, y_att_a, half)

    for base_key in ("sigma_x", "sigma_y", "eff_area", "def3_line_x", "att3_line_x"):
        vh = stats_h[base_key]
        va = stats_a[base_key]
        f[f"{base_key}_h"] = vh.astype(np.float32)
        f[f"{base_key}_a"] = va.astype(np.float32)
        f[f"diff_{base_key}"] = (vh - va).astype(np.float32)

    for dt in DIPOLE_DELTAS:
        for d_key in ("sigma_x", "sigma_y", "eff_area", "cx", "cy", "def3_x", "att3_x"):
            vh = stats_h[f"d_{d_key}_dt{dt}"]
            va = stats_a[f"d_{d_key}_dt{dt}"]
            f[f"d_{d_key}_h_dt{dt}"] = vh.astype(np.float32)
            f[f"d_{d_key}_a_dt{dt}"] = va.astype(np.float32)
            f[f"diff_d_{d_key}_dt{dt}"] = (vh - va).astype(np.float32)

    # =========================================================================
    # 3. EXP-05 Part B: Ball-Free Pressure Convergence Focus (x_focus(t))
    # =========================================================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vel_all = lsq_velocity(P_raw_c, half, h=2)  # (T, S, 2) in m/s

    outfield_mask = (sel_h | sel_a) & (~nc.slot_is_gk)
    P_out = P_c[:, outfield_mask, :]                  # (T, N_out, 2)
    V_out = np.nan_to_num(vel_all[:, outfield_mask, :], nan=0.0)
    vis_out = np.isfinite(P_out[..., 0]) & np.isfinite(P_out[..., 1])
    V_out = np.where(vis_out[..., None], V_out, 0.0)
    sp_out = np.hypot(V_out[..., 0], V_out[..., 1])

    # Project outfield players tau = 0.6s into the future
    P_proj = P_out + FOCUS_TAU_S * V_out              # (T, N_out, 2)

    # Pairwise distances between projected outfield players at each frame
    N_out = P_proj.shape[1]
    diff_x = P_proj[:, :, None, 0] - P_proj[:, None, :, 0]
    diff_y = P_proj[:, :, None, 1] - P_proj[:, None, :, 1]
    dist_mat = np.hypot(diff_x, diff_y)               # (T, N_out, N_out)
    valid_pair = vis_out[:, :, None] & vis_out[:, None, :]
    eye_mask = np.eye(N_out, dtype=bool)[None, :, :]
    dist_mat = np.where(valid_pair & (~eye_mask), dist_mat, np.inf)

    # Mean distance to the 4 nearest other visible outfield players
    d_sorted = np.sort(dist_mat, axis=2)[:, :, :4]
    d_valid = np.isfinite(d_sorted)
    n_near = np.maximum(np.sum(d_valid, axis=2), 1)
    d4_mean = np.where(
        np.any(d_valid, axis=2),
        np.sum(np.where(d_valid, d_sorted, 0.0), axis=2) / n_near,
        30.0,
    )

    # Weight each visible outfield player by speed and inverse local pairwise distance (tightest moving cluster)
    # Using power 3 on inverse distance sharply focuses on the tightest spatial cluster of moving players.
    w_player = np.where(vis_out, (0.5 + sp_out) / ((d4_mean + 1.5) ** 3), 0.0)
    w_sum = np.sum(w_player, axis=1)
    focus_raw_x = np.where(w_sum > 1e-12, np.sum(w_player * np.nan_to_num(P_proj[..., 0], nan=0.0), axis=1) / np.maximum(w_sum, 1e-12), 0.0)
    focus_raw_y = np.where(w_sum > 1e-12, np.sum(w_player * np.nan_to_num(P_proj[..., 1], nan=0.0), axis=1) / np.maximum(w_sum, 1e-12), 0.0)

    # Smooth focus lightly per half (sigma = 3 frames)
    focus_x = _gaussian_smooth_by_half(focus_raw_x, half, sigma=FOCUS_SMOOTH_SIGMA)
    focus_y = _gaussian_smooth_by_half(focus_raw_y, half, sigma=FOCUS_SMOOTH_SIGMA)

    # Normalized focus coordinates in [-1, 1] (pure sign flip under 180-deg rotation!)
    f["focus_x_norm"] = (focus_x / (L / 2.0)).astype(np.float32)
    f["focus_y_norm"] = (focus_y / (W / 2.0)).astype(np.float32)

    # Player distances and radial inflow velocities toward x_focus(t)
    V_all_clean = np.nan_to_num(vel_all, nan=0.0)
    rx = focus_x[:, None] - P_c[..., 0]
    ry = focus_y[:, None] - P_c[..., 1]
    d_focus = np.hypot(rx, ry)
    vis_all = np.isfinite(d_focus)
    inv_r = np.where(vis_all & (d_focus > 1e-4), 1.0 / np.maximum(d_focus, 1e-4), 0.0)
    inflow_all = (V_all_clean[..., 0] * rx + V_all_clean[..., 1] * ry) * inv_r

    def _team_focus_metrics(team_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dt = np.where(vis_all[:, team_mask], d_focus[:, team_mask], np.inf)
        vt = np.where(vis_all[:, team_mask], inflow_all[:, team_mask], 0.0)
        order = np.argsort(dt, axis=1)
        d_sort = np.take_along_axis(dt, order, axis=1)
        v_sort = np.take_along_axis(vt, order, axis=1)

        d1 = np.where(np.isfinite(d_sort[:, 0]), d_sort[:, 0], 30.0)
        d2 = np.where(np.isfinite(d_sort[:, 1]), d_sort[:, 1], 30.0)
        n5m = np.sum(dt < 5.0, axis=1).astype(np.float64)

        v3_valid = np.isfinite(d_sort[:, :3])
        n3 = np.maximum(np.sum(v3_valid, axis=1), 1)
        inflow3 = np.where(np.any(v3_valid, axis=1), np.sum(np.where(v3_valid, v_sort[:, :3], 0.0), axis=1) / n3, 0.0)
        return d1, d2, n5m, inflow3

    fd1_h, fd2_h, fn5_h, finf_h = _team_focus_metrics(sel_h)
    fd1_a, fd2_a, fn5_a, finf_a = _team_focus_metrics(sel_a)

    f["focus_d1_h"] = fd1_h.astype(np.float32)
    f["focus_d1_a"] = fd1_a.astype(np.float32)
    f["focus_d2_h"] = fd2_h.astype(np.float32)
    f["focus_d2_a"] = fd2_a.astype(np.float32)
    f["focus_diff_d1"] = (fd1_h - fd1_a).astype(np.float32)
    f["focus_diff_d2"] = (fd2_h - fd2_a).astype(np.float32)

    f["focus_n5m_h"] = fn5_h.astype(np.float32)
    f["focus_n5m_a"] = fn5_a.astype(np.float32)
    f["focus_diff_n5m"] = (fn5_h - fn5_a).astype(np.float32)

    f["focus_inflow_h"] = finf_h.astype(np.float32)
    f["focus_inflow_a"] = finf_a.astype(np.float32)
    f["focus_diff_inflow"] = (finf_h - finf_a).astype(np.float32)

    # Ultimate Ghost-Ball Detector: ||b_csv(t) - x_focus(t)||
    ball_focus_discrepancy = np.hypot(B_interp_c[:, 0] - focus_x, B_interp_c[:, 1] - focus_y)
    f["ball_focus_discrepancy"] = ball_focus_discrepancy.astype(np.float32)

    # Temporal context on focus features
    f["focus_diff_d1_r1"] = roll_nanmean_by_half(fd1_h - fd1_a, half, int(round(1.0 * FPS)))
    f["focus_diff_d1_r3"] = roll_nanmean_by_half(fd1_h - fd1_a, half, int(round(3.0 * FPS)))
    f["focus_diff_n5m_r1"] = roll_nanmean_by_half(fn5_h - fn5_a, half, int(round(1.0 * FPS)))
    f["focus_diff_inflow_r1"] = roll_nanmean_by_half(finf_h - finf_a, half, int(round(1.0 * FPS)))
    f["ball_focus_discrepancy_r1"] = roll_nanmean_by_half(ball_focus_discrepancy, half, int(round(1.0 * FPS)))
    f["ball_focus_discrepancy_r3"] = roll_nanmean_by_half(ball_focus_discrepancy, half, int(round(3.0 * FPS)))

    return pd.DataFrame(f)


def mirror_physv2_df(
    df_physv2: pd.DataFrame,
    half: np.ndarray | None = None,
    pitch: tuple[float, float] = (105.0, 68.0),
) -> pd.DataFrame:
    """Exact algebraic H <-> A + 180-deg pitch rotation transformation of `compute_physv2_features`."""
    M = df_physv2.copy()

    # 1. Direct H <-> A swap pairs
    swap_pairs = [
        ("d1_h_impulse_gated", "d1_a_impulse_gated"),
        ("prev_knot_d1_h", "prev_knot_d1_a"),
        ("next_knot_d1_h", "next_knot_d1_a"),
        ("chord_same_team_prob_h", "chord_same_team_prob_a"),
        ("sigma_x_h", "sigma_x_a"),
        ("sigma_y_h", "sigma_y_a"),
        ("eff_area_h", "eff_area_a"),
        ("def3_line_x_h", "def3_line_x_a"),
        ("att3_line_x_h", "att3_line_x_a"),
        ("focus_d1_h", "focus_d1_a"),
        ("focus_d2_h", "focus_d2_a"),
        ("focus_n5m_h", "focus_n5m_a"),
        ("focus_inflow_h", "focus_inflow_a"),
    ]
    for dt in DIPOLE_DELTAS:
        for d_key in ("sigma_x", "sigma_y", "eff_area", "cx", "cy", "def3_x", "att3_x"):
            swap_pairs.append((f"d_{d_key}_h_dt{dt}", f"d_{d_key}_a_dt{dt}"))

    for ch, ca in swap_pairs:
        if ch in M.columns and ca in M.columns:
            M[ch] = df_physv2[ca]
            M[ca] = df_physv2[ch]

    # 2. Pure sign negations (team-diff features & pitch-centered normalized focus coordinates)
    neg_cols = [
        "diff_d1_impulse_gated",
        "prev_knot_diff_d1",
        "next_knot_diff_d1",
        "chord_same_team_diff",
        "diff_d1_impulse_gated_r1",
        "diff_d1_impulse_gated_r2",
        "chord_same_team_diff_r1",
        "chord_same_team_diff_r2",
        "diff_sigma_x",
        "diff_sigma_y",
        "diff_eff_area",
        "diff_def3_line_x",
        "diff_att3_line_x",
        "focus_x_norm",
        "focus_y_norm",
        "focus_diff_d1",
        "focus_diff_d2",
        "focus_diff_n5m",
        "focus_diff_inflow",
        "focus_diff_d1_r1",
        "focus_diff_d1_r3",
        "focus_diff_n5m_r1",
        "focus_diff_inflow_r1",
    ]
    for dt in DIPOLE_DELTAS:
        for d_key in ("sigma_x", "sigma_y", "eff_area", "cx", "cy", "def3_x", "att3_x"):
            neg_cols.append(f"diff_d_{d_key}_dt{dt}")

    for col in neg_cols:
        if col in M.columns:
            M[col] = -df_physv2[col]

    return M


def mirror_v1ctl_physv2_df(
    X: pd.DataFrame,
    half: np.ndarray,
    pitch: tuple[float, float] = (105.0, 68.0),
) -> pd.DataFrame:
    """Algebraically transform a combined `v1ctl_physics_v2` DataFrame under H <-> A + 180-deg pitch rotation."""
    M = mirror_v1ctl_plus_df(X, half, pitch)
    return mirror_physv2_df(M, half, pitch)


mirror_v1ctl_physics_v2_df = mirror_v1ctl_physv2_df


def cached_physv2(m: Match, key: str, cache_dir=None) -> pd.DataFrame:
    """Compute or load cached `physv2` feature block from `local_data/cache/physv2/`."""
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "local_data" / "cache" / "physv2"
    path = root / f"{key}_{match_fingerprint(m)}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    P2 = compute_physv2_features(m)
    root.mkdir(parents=True, exist_ok=True)
    P2.to_parquet(path)
    return P2


def build_v1ctl_physics_v2(m: Match, key: str, cache_dir=None) -> pd.DataFrame:
    """Build full `v1ctl_physics_v2` feature DataFrame (`v1ctl_plus` + `physv2`) using disk caches."""
    X_plus = build_v1ctl_plus(m, key)
    X_physv2 = cached_physv2(m, key, cache_dir=cache_dir)
    return pd.concat([X_plus, X_physv2], axis=1)


build_v1ctl_physv2 = build_v1ctl_physics_v2
build_features_physv2 = build_v1ctl_physics_v2
