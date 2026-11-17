"""EXP-11: Permutation-Invariant Player-Set Temporal Network (`SymPlayerSetNet`).

6th Model Family (DeepSets / PointNet + Dilated Temporal Conv + Strided BiGRU) operating
directly over the 22 players' raw kinematics and ball/focus geometry per frame:

1. Vectorized Per-Frame Player-Set Tensor Extraction (`extract_player_set_tensors`):
   - Extracts canonical own-team attacking-frame player tensors `P_H (T, 11, 18)` and
     `P_A (T, 11, 18)` with visibility masks `mask_H (T, 11)` and `mask_A (T, 11)`, plus
     strictly antisymmetric global context `G_anti (T, 44)` and strictly symmetric global
     context `G_sym (T, 50)`.
   - First 12 player channels (`C_bf = 12`) are 100% BALL-FREE (pitch coordinates, track-break-safe
     velocities, speed, relative vector/distance/proximity/closing-speed to the ball-free
     pressure convergence focus `x_focus(t)`, and nearest-opponent marking distance/proximity).
   - Remaining 6 player channels (`C_ball = 6`) are BALL-AWARE (relative vector, distance,
     3m/10m exponential proximity, and closing speed to the interpolated ball).
   - Satisfies exact bit-for-bit equivariance under `swap_match(m)` (Home <-> Away + 180-deg
     pitch rotation: `P_H(swap(m)) == P_A(m)`, `P_A(swap(m)) == P_H(m)`, `G_anti(swap(m)) == -G_anti(m)`,
     `G_sym(swap(m)) == +G_sym(m)` with `0.000e+00` discrepancy) and 100% invariance (`0.000e+00`)
     to player-slot column permutations (`track_breaks`).

2. Split Ball-Free / Ball-Aware Siamese PointNet + Ghost-Gated Equivariant Backbone (`SymPlayerSetNet`):
   - Shared per-player PointNet `phi_theta` splits into a 100% ball-free branch (`phi_bf` +
     ball-free focus attention head) and a ball-aware branch (`phi_ball` + ball-proximity
     attention head), aggregated via masked `mean` + masked `max` + 2-head masked `softmax-attention`
     pooling over the 11 players per team.
   - Symmetric Ghost-Ball Reliability Gate computed from `G_sym` dynamically modulates the
     pooled player-set representations during ghost-ball spikes (`degrade_v6`) and ball blackouts.
   - Twin antisymmetric/symmetric input projection into a dilated temporal ResNet + strided BiGRU
     guarantees `max |p(m) - swap(p(swap_match(m)))| == 0.000e+00` in a single forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.ball_interp import linear_mask
from src.data.common import AWAY, FPS, HOME, Match
from src.features.knot_physics import _linear_runs, filter_ghost_and_switch_slots
from src.features.v1 import match_fingerprint
from src.features.v1ctl_physics_v2 import (
    FOCUS_SMOOTH_SIGMA,
    FOCUS_TAU_S,
    _ball_impulse_at_delta,
    _gaussian_smooth_by_half,
)
from src.features.v1ctl_plus import (
    BREAK_STEP_M,
    SPEED_CAP,
    _gap_edges_by_half,
    lsq_velocity,
    roll_nanmean_by_half,
)
from src.models.transition import change_targets

VERSION = 1
K_PLAYERS = 11
C_PLAYER_BF = 12
C_PLAYER_BALL = 6
C_PLAYER_TOTAL = C_PLAYER_BF + C_PLAYER_BALL


class ChannelNorm(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class _ResBlock(nn.Module):
    def __init__(self, c: int, d: int, dropout: float):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=d, dilation=d),
            ChannelNorm(c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c, c, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


@dataclass
class PlayerSetTensors:
    p_h: np.ndarray       # (T, 11, 18) float32
    p_a: np.ndarray       # (T, 11, 18) float32
    mask_h: np.ndarray    # (T, 11) bool
    mask_a: np.ndarray    # (T, 11) bool
    g_anti: np.ndarray    # (T, C_ga) float32
    g_sym: np.ndarray     # (T, C_gs) float32


def _safe_cascaded_velocity(P_raw_c: np.ndarray, vis: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Compute track-break-free velocity (T, S, 2) in m/s by cascading:
    lsq_velocity(h=2) -> centered 2-frame velocity (h=1) -> safe 1-frame backward/forward step (<= 1.5m).
    """
    T, S, _ = P_raw_c.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = lsq_velocity(P_raw_c, half, h=2, cap=SPEED_CAP, break_m=BREAK_STEP_M)

    if T > 1:
        dx = P_raw_c[1:, :, 0] - P_raw_c[:-1, :, 0]
        dy = P_raw_c[1:, :, 1] - P_raw_c[:-1, :, 1]
        step = np.hypot(dx, dy)
        ok_step = (
            vis[1:]
            & vis[:-1]
            & (half[1:] == half[:-1])[:, None]
            & (step <= BREAK_STEP_M)
        )
        step_vx = np.where(ok_step, dx * FPS, np.nan)
        step_vy = np.where(ok_step, dy * FPS, np.nan)

        v_bwd = np.full((T, S, 2), np.nan, dtype=np.float64)
        v_bwd[1:, :, 0] = step_vx
        v_bwd[1:, :, 1] = step_vy

        v_fwd = np.full((T, S, 2), np.nan, dtype=np.float64)
        v_fwd[:-1, :, 0] = step_vx
        v_fwd[:-1, :, 1] = step_vy

        # Centered 2-frame velocity (h=1) where both backward and forward steps are clean
        v_1h = 0.5 * (v_bwd + v_fwd)
        v_1f = np.where(np.isfinite(v_1h), v_1h, np.where(np.isfinite(v_bwd), v_bwd, v_fwd))
        sp_1f = np.hypot(v_1f[..., 0], v_1f[..., 1])
        scale = np.where(sp_1f > SPEED_CAP, SPEED_CAP / np.maximum(sp_1f, 1e-9), 1.0)
        v_1f = v_1f * scale[..., None]

        v = np.where(np.isfinite(v), v, v_1f)

    v = np.where(vis[..., None] & np.isfinite(v), v, 0.0)
    return v


def _canonical_sort_team_slots(
    P_c: np.ndarray,
    V_c: np.ndarray,
    vis: np.ndarray,
    is_gk: np.ndarray,
    team_mask: np.ndarray,
    sgn: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract team slots and sort them at every frame t by canonical own-frame coordinates
    `(x_own, y_own)` so all subsequent reductions are 100% invariant to slot column order."""
    P_t = P_c[:, team_mask, :]
    V_t = V_c[:, team_mask, :]
    vis_t = vis[:, team_mask]
    gk_t = np.broadcast_to(is_gk[None, team_mask], vis_t.shape)

    x_own = np.where(vis_t, sgn * P_t[..., 0], np.inf)
    y_own = np.where(vis_t, sgn * P_t[..., 1], np.inf)

    # Lexicographical sort by (vis, x_own, y_own) via composite float64 key
    key = np.where(vis_t, x_own * 1000.0 + y_own * 1e-2, 1e12)
    order = np.argsort(key, axis=1, kind="mergesort")
    if order.shape[1] > K_PLAYERS:
        order = order[:, :K_PLAYERS]

    P_sorted = np.take_along_axis(P_t, order[..., None], axis=1)
    V_sorted = np.take_along_axis(V_t, order[..., None], axis=1)
    vis_sorted = np.take_along_axis(vis_t, order, axis=1)
    gk_sorted = np.take_along_axis(gk_t, order, axis=1)
    return P_sorted, V_sorted, vis_sorted, gk_sorted


def _compute_symmetric_focus(
    P_h_s: np.ndarray,
    V_h_s: np.ndarray,
    vis_h_s: np.ndarray,
    gk_h_s: np.ndarray,
    P_a_s: np.ndarray,
    V_a_s: np.ndarray,
    vis_a_s: np.ndarray,
    gk_a_s: np.ndarray,
    half: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ball-free pressure convergence focus `(focus_x, focus_y)` and velocity
    `(focus_vx, focus_vy)` with exact `0.000e+00` equivariance under `swap_match` and
    `0.000e+00` invariance under slot permutation."""
    out_h = vis_h_s & (~gk_h_s)
    out_a = vis_a_s & (~gk_a_s)

    P_proj_h = np.where(out_h[..., None], P_h_s + FOCUS_TAU_S * V_h_s, np.nan)
    P_proj_a = np.where(out_a[..., None], P_a_s + FOCUS_TAU_S * V_a_s, np.nan)

    sp_h = np.where(out_h, np.hypot(V_h_s[..., 0], V_h_s[..., 1]), 0.0)
    sp_a = np.where(out_a, np.hypot(V_a_s[..., 0], V_a_s[..., 1]), 0.0)

    # Concatenate for pairwise distance computation; sorting the 4 nearest makes d4_mean order-independent
    P_proj_all = np.concatenate([P_proj_h, P_proj_a], axis=1).astype(np.float32)
    out_all = np.concatenate([out_h, out_a], axis=1)
    N_all = P_proj_all.shape[1]
    S_h = P_proj_h.shape[1]

    diff_x = P_proj_all[:, :, None, 0] - P_proj_all[:, None, :, 0]
    diff_y = P_proj_all[:, :, None, 1] - P_proj_all[:, None, :, 1]
    d2_mat = diff_x * diff_x + diff_y * diff_y
    valid_pair = out_all[:, :, None] & out_all[:, None, :]
    eye_mask = np.eye(N_all, dtype=bool)[None, :, :]
    d2_mat = np.where(valid_pair & (~eye_mask), d2_mat, np.float32(np.inf))

    k_near = min(4, max(N_all - 1, 1))
    d2_part = np.partition(d2_mat, k_near - 1, axis=2)[:, :, :k_near]
    d_sorted = np.sqrt(np.sort(d2_part, axis=2).astype(np.float64))
    d_valid = np.isfinite(d_sorted)
    n_near = np.maximum(np.sum(d_valid, axis=2), 1)
    d4_mean = np.where(
        np.any(d_valid, axis=2),
        np.sum(np.where(d_valid, d_sorted, 0.0), axis=2) / n_near,
        30.0,
    )

    d4_h = d4_mean[:, :S_h]
    d4_a = d4_mean[:, S_h:]

    w_h = np.where(out_h, (0.5 + sp_h) / ((d4_h + 1.5) ** 3), 0.0)
    w_a = np.where(out_a, (0.5 + sp_a) / ((d4_a + 1.5) ** 3), 0.0)

    # Sum within sorted Home and sorted Away first, then add (commutative in IEEE-754!)
    sw_h = np.sum(w_h, axis=1)
    sw_a = np.sum(w_a, axis=1)
    w_sum = sw_h + sw_a

    sx_h = np.sum(w_h * np.nan_to_num(P_proj_h[..., 0], nan=0.0), axis=1)
    sx_a = np.sum(w_a * np.nan_to_num(P_proj_a[..., 0], nan=0.0), axis=1)
    sy_h = np.sum(w_h * np.nan_to_num(P_proj_h[..., 1], nan=0.0), axis=1)
    sy_a = np.sum(w_a * np.nan_to_num(P_proj_a[..., 1], nan=0.0), axis=1)

    focus_raw_x = np.where(w_sum > 1e-12, (sx_h + sx_a) / np.maximum(w_sum, 1e-12), 0.0)
    focus_raw_y = np.where(w_sum > 1e-12, (sy_h + sy_a) / np.maximum(w_sum, 1e-12), 0.0)

    focus_x = _gaussian_smooth_by_half(focus_raw_x, half, sigma=FOCUS_SMOOTH_SIGMA)
    focus_y = _gaussian_smooth_by_half(focus_raw_y, half, sigma=FOCUS_SMOOTH_SIGMA)

    # Focus velocity (m/s) within half
    T = len(focus_x)
    focus_vx = np.zeros(T, dtype=np.float64)
    focus_vy = np.zeros(T, dtype=np.float64)
    if T > 2:
        ok = half[2:] == half[:-2]
        focus_vx[1:-1][ok] = (focus_x[2:][ok] - focus_x[:-2][ok]) * (FPS / 2.0)
        focus_vy[1:-1][ok] = (focus_y[2:][ok] - focus_y[:-2][ok]) * (FPS / 2.0)

    return focus_x, focus_y, focus_vx, focus_vy


def _pack_active_k_players(
    P_s: np.ndarray,
    V_s: np.ndarray,
    vis_s: np.ndarray,
    focus_x: np.ndarray,
    focus_y: np.ndarray,
    sgn: float,
    k_max: int = K_PLAYERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select and order up to K=11 active players per team at frame t by distance to `x_focus`
    with canonical own-frame tie-breaker `(x_own, y_own)`."""
    T, S, _ = P_s.shape
    x_own = np.where(vis_s, sgn * P_s[..., 0], 0.0)
    y_own = np.where(vis_s, sgn * P_s[..., 1], 0.0)
    vx_own = np.where(vis_s, sgn * V_s[..., 0], 0.0)
    vy_own = np.where(vis_s, sgn * V_s[..., 1], 0.0)

    fx_own = sgn * focus_x[:, None]
    fy_own = sgn * focus_y[:, None]
    d_focus = np.hypot(fx_own - x_own, fy_own - y_own)

    sort_key = np.where(vis_s, d_focus + 1e-4 * x_own + 1e-7 * y_own, 1e12)
    order = np.argsort(sort_key, axis=1, kind="mergesort")

    if S >= k_max:
        order_k = order[:, :k_max]
        x_k = np.take_along_axis(x_own, order_k, axis=1)
        y_k = np.take_along_axis(y_own, order_k, axis=1)
        vx_k = np.take_along_axis(vx_own, order_k, axis=1)
        vy_k = np.take_along_axis(vy_own, order_k, axis=1)
        mask_k = np.take_along_axis(vis_s, order_k, axis=1)
    else:
        pad = k_max - S
        x_k = np.pad( np.take_along_axis(x_own, order, axis=1), ((0, 0), (0, pad)), constant_values=0.0)
        y_k = np.pad( np.take_along_axis(y_own, order, axis=1), ((0, 0), (0, pad)), constant_values=0.0)
        vx_k = np.pad(np.take_along_axis(vx_own, order, axis=1), ((0, 0), (0, pad)), constant_values=0.0)
        vy_k = np.pad(np.take_along_axis(vy_own, order, axis=1), ((0, 0), (0, pad)), constant_values=0.0)
        mask_k = np.pad(np.take_along_axis(vis_s, order, axis=1), ((0, 0), (0, pad)), constant_values=False)

    P_own_k = np.stack([np.where(mask_k, x_k, 0.0), np.where(mask_k, y_k, 0.0)], axis=-1)
    V_own_k = np.stack([np.where(mask_k, vx_k, 0.0), np.where(mask_k, vy_k, 0.0)], axis=-1)
    # Also return pitch-centered coordinates for cross-team opponent distance matrix
    P_c_k = sgn * P_own_k
    return P_own_k, V_own_k, P_c_k, mask_k


def _build_team_player_features(
    P_own: np.ndarray,
    V_own: np.ndarray,
    mask: np.ndarray,
    d_opp1: np.ndarray,
    focus_x_own: np.ndarray,
    focus_y_own: np.ndarray,
    ball_x_own: np.ndarray,
    ball_y_own: np.ndarray,
    L: float,
    W: float,
) -> np.ndarray:
    """Build `(T, 11, 18)` player feature tensor in the team's own attacking coordinate frame.
    Channels `0..11` (12 channels) are 100% BALL-FREE; channels `12..17` (6 channels) are BALL-AWARE.
    """
    T, K, _ = P_own.shape
    x = P_own[..., 0]
    y = P_own[..., 1]
    vx = V_own[..., 0]
    vy = V_own[..., 1]
    sp = np.hypot(vx, vy)

    # Focus relative geometry (100% ball-free)
    rx_f = focus_x_own[:, None] - x
    ry_f = focus_y_own[:, None] - y
    d_f = np.hypot(rx_f, ry_f)
    inv_df = np.where(mask & (d_f > 1e-4), 1.0 / np.maximum(d_f, 1e-4), 0.0)
    v_close_f = (vx * rx_f + vy * ry_f) * inv_df

    # Ball relative geometry (ball-aware)
    rx_b = ball_x_own[:, None] - x
    ry_b = ball_y_own[:, None] - y
    d_b = np.hypot(rx_b, ry_b)
    inv_db = np.where(mask & (d_b > 1e-4), 1.0 / np.maximum(d_b, 1e-4), 0.0)
    v_close_b = (vx * rx_b + vy * ry_b) * inv_db

    out = np.zeros((T, K, C_PLAYER_TOTAL), dtype=np.float32)

    # --- 12 Ball-Free Channels (0..11) ---
    out[..., 0] = np.where(mask, np.clip(x / (L / 2.0), -1.5, 1.5), 0.0)
    out[..., 1] = np.where(mask, np.clip(y / (W / 2.0), -1.5, 1.5), 0.0)
    out[..., 2] = np.where(mask, np.clip(vx / 10.0, -1.5, 1.5), 0.0)
    out[..., 3] = np.where(mask, np.clip(vy / 10.0, -1.5, 1.5), 0.0)
    out[..., 4] = np.where(mask, np.clip(sp / 10.0, 0.0, 1.5), 0.0)
    out[..., 5] = np.where(mask, np.clip(rx_f / 30.0, -2.0, 2.0), 0.0)
    out[..., 6] = np.where(mask, np.clip(ry_f / 20.0, -2.0, 2.0), 0.0)
    out[..., 7] = np.where(mask, np.clip(d_f / 30.0, 0.0, 2.0), 0.0)
    out[..., 8] = np.where(mask, np.exp(-np.clip(d_f, 0.0, 50.0) / 5.0), 0.0)
    out[..., 9] = np.where(mask, np.clip(v_close_f / 10.0, -1.5, 1.5), 0.0)
    out[..., 10] = np.where(mask, np.clip(d_opp1 / 15.0, 0.0, 2.0), 0.0)
    out[..., 11] = np.where(mask, np.exp(-np.clip(d_opp1, 0.0, 50.0) / 3.0), 0.0)

    # --- 6 Ball-Aware Channels (12..17) ---
    out[..., 12] = np.where(mask, np.clip(rx_b / 30.0, -2.0, 2.0), 0.0)
    out[..., 13] = np.where(mask, np.clip(ry_b / 20.0, -2.0, 2.0), 0.0)
    out[..., 14] = np.where(mask, np.clip(d_b / 30.0, 0.0, 2.0), 0.0)
    out[..., 15] = np.where(mask, np.exp(-np.clip(d_b, 0.0, 50.0) / 3.0), 0.0)
    out[..., 16] = np.where(mask, np.exp(-np.clip(d_b, 0.0, 50.0) / 10.0), 0.0)
    out[..., 17] = np.where(mask, np.clip(v_close_b / 10.0, -1.5, 1.5), 0.0)

    return out


def _team_summary_scalars(
    P_team: np.ndarray,
    mask_team: np.ndarray,
    half: np.ndarray,
    last_b_idx: np.ndarray,
    nxt_b_idx: np.ndarray,
    w_b: np.ndarray,
    bvis: np.ndarray,
    tp_safe: np.ndarray,
    tn_safe: np.ndarray,
    alpha_next: np.ndarray,
    chord_in_flight: np.ndarray,
) -> np.ndarray:
    """Compute per-team summary scalar curves `(T, M)` identically for Home and Away from
    their canonical own-frame player tensors `P_team (T, 11, 18)` and `mask_team (T, 11)`.
    Because `P_H(swap(m)) == P_A(m)` bit-for-bit, `Q_H - Q_A` is strictly antisymmetric
    and `Q_H + Q_A` is strictly symmetric (`0.000e+00` error).
    """
    T = P_team.shape[0]
    m_f = mask_team.astype(np.float64)
    n_vis = np.maximum(np.sum(m_f, axis=1), 1.0)

    # 1. Ball-Free Focus & Team Collective Summaries
    x_norm = P_team[..., 0].astype(np.float64)
    y_norm = P_team[..., 1].astype(np.float64)
    vx_norm = P_team[..., 2].astype(np.float64)
    vy_norm = P_team[..., 3].astype(np.float64)
    sp_norm = P_team[..., 4].astype(np.float64)
    d_focus = np.where(mask_team, P_team[..., 7].astype(np.float64), 2.0)
    prox_focus = P_team[..., 8].astype(np.float64)
    v_close_f = P_team[..., 9].astype(np.float64)
    prox_opp = P_team[..., 11].astype(np.float64)

    # P_team is already ordered by d_focus
    d1_focus = d_focus[:, 0]
    d2_focus = d_focus[:, 1]
    prox_focus_sum = np.sum(prox_focus * m_f, axis=1) / 3.0
    v_close_f_top3 = np.sum(v_close_f[:, :3] * m_f[:, :3], axis=1) / np.maximum(np.sum(m_f[:, :3], axis=1), 1.0)
    prox_opp_mean = np.sum(prox_opp * m_f, axis=1) / n_vis

    cx = np.sum(x_norm * m_f, axis=1) / n_vis
    cy = np.sum(y_norm * m_f, axis=1) / n_vis
    vx_mean = np.sum(vx_norm * m_f, axis=1) / n_vis
    vy_mean = np.sum(vy_norm * m_f, axis=1) / n_vis
    sp_mean = np.sum(sp_norm * m_f, axis=1) / n_vis

    # Dead-ball / tempo discriminators: fraction of slow (< 1.5 m/s) and sprinting (> 5.0 m/s) players
    slow_frac = np.sum(((sp_norm < 0.15) & mask_team).astype(np.float64), axis=1) / n_vis
    sprint_frac = np.sum(((sp_norm > 0.50) & mask_team).astype(np.float64), axis=1) / n_vis

    # Defensive and attacking lines (sorted x_norm among visible teammates)
    x_for_min = np.where(mask_team, x_norm, np.inf)
    x_for_max = np.where(mask_team, x_norm, -np.inf)
    x_asc = np.sort(x_for_min, axis=1)
    x_desc = -np.sort(-x_for_max, axis=1)
    def3_slice = x_asc[:, 1:4]
    def3_valid = np.isfinite(def3_slice)
    def3_x = np.where(
        np.any(def3_valid, axis=1),
        np.sum(np.where(def3_valid, def3_slice, 0.0), axis=1) / np.maximum(np.sum(def3_valid, axis=1), 1),
        cx,
    )
    att3_slice = x_desc[:, :3]
    att3_valid = np.isfinite(att3_slice)
    att3_x = np.where(
        np.any(att3_valid, axis=1),
        np.sum(np.where(att3_valid, att3_slice, 0.0), axis=1) / np.maximum(np.sum(att3_valid, axis=1), 1),
        cx,
    )

    var_x = np.sum(m_f * ((x_norm - cx[:, None]) ** 2), axis=1) / n_vis
    var_y = np.sum(m_f * ((y_norm - cy[:, None]) ** 2), axis=1) / n_vis
    sigma_x = np.sqrt(np.maximum(var_x, 0.0))
    sigma_y = np.sqrt(np.maximum(var_y, 0.0))

    # Collective dipole temporal derivatives over 0.5s (12f) and 1.0s (25f)
    def _half_diff(arr: np.ndarray, dt: int) -> np.ndarray:
        out = np.zeros(T, dtype=np.float64)
        if T > dt:
            ok = half[dt:] == half[:-dt]
            out[dt:][ok] = (arr[dt:][ok] - arr[:-dt][ok]) * (FPS / float(dt))
        return out

    d_cx_12 = _half_diff(cx, 12)
    d_cx_25 = _half_diff(cx, 25)
    d_sx_25 = _half_diff(sigma_x, 25)
    d_sy_25 = _half_diff(sigma_y, 25)
    d_def3_25 = _half_diff(def3_x, 25)
    d_att3_25 = _half_diff(att3_x, 25)

    # 2. Ball-Aware Proximity, Gap-Edge Continuity & Knot-Chord Attribution
    d_ball_mat = np.where(mask_team, P_team[..., 14].astype(np.float64), 2.0)  # in units of 30m
    prox_b3 = P_team[..., 15].astype(np.float64)
    prox_b10 = P_team[..., 16].astype(np.float64)
    v_close_b = P_team[..., 17].astype(np.float64)

    order_b = np.argsort(d_ball_mat, axis=1, kind="mergesort")
    d_b_sorted = np.take_along_axis(d_ball_mat, order_b, axis=1)
    vc_b_sorted = np.take_along_axis(v_close_b, order_b, axis=1)

    d1_ball = d_b_sorted[:, 0]
    d2_ball = d_b_sorted[:, 1]
    vc1_ball = vc_b_sorted[:, 0]
    prox_b3_sum = np.sum(prox_b3 * m_f, axis=1) / 2.0
    prox_b10_sum = np.sum(prox_b10 * m_f, axis=1) / 4.0

    # Gap-edge continuity of d1_ball across ball blackouts
    has_lb = last_b_idx >= 0
    has_nb = nxt_b_idx < T
    d1_last = d1_ball[np.clip(last_b_idx, 0, T - 1)]
    d1_next = d1_ball[np.clip(nxt_b_idx, 0, T - 1)]
    d1_gap_interp = np.where(
        bvis,
        d1_ball,
        np.where(
            has_lb & has_nb,
            (1.0 - w_b) * d1_last + w_b * d1_next,
            np.where(has_lb, d1_last, np.where(has_nb, d1_next, d1_ball)),
        ),
    )

    # Pass-chord knot distances & impulse-gated distance
    d1_prev_knot = d1_ball[tp_safe]
    d1_next_knot = d1_ball[tn_safe]
    d1_knot_interp = (1.0 - alpha_next) * d1_prev_knot + alpha_next * d1_next_knot
    d1_pen = np.maximum(d1_ball, d1_knot_interp)
    d1_impulse_gated = (1.0 - chord_in_flight) * d1_ball + chord_in_flight * (0.5 * d1_pen + 0.5 * d1_knot_interp)

    # Rolling multi-scale summaries (0.5s = 6f, 1.0s = 12f, 2.5s = 31f, 3.0s = 37f)
    d1_ball_r12 = np.nan_to_num(roll_nanmean_by_half(d1_gap_interp, half, 12), nan=1.0)
    d1_ball_r31 = np.nan_to_num(roll_nanmean_by_half(d1_gap_interp, half, 31), nan=1.0)
    d1_imp_r12 = np.nan_to_num(roll_nanmean_by_half(d1_impulse_gated, half, 12), nan=1.0)
    prox_b3_r12 = np.nan_to_num(roll_nanmean_by_half(prox_b3_sum, half, 12), nan=0.0)
    d1_focus_r12 = np.nan_to_num(roll_nanmean_by_half(d1_focus, half, 12), nan=1.0)
    d1_focus_r31 = np.nan_to_num(roll_nanmean_by_half(d1_focus, half, 31), nan=1.0)
    prox_focus_r12 = np.nan_to_num(roll_nanmean_by_half(prox_focus_sum, half, 12), nan=0.0)
    slow_frac_r12 = np.nan_to_num(roll_nanmean_by_half(slow_frac, half, 12), nan=0.3)
    slow_frac_r37 = np.nan_to_num(roll_nanmean_by_half(slow_frac, half, 37), nan=0.3)
    sp_mean_r12 = np.nan_to_num(roll_nanmean_by_half(sp_mean, half, 12), nan=0.25)
    sp_mean_r37 = np.nan_to_num(roll_nanmean_by_half(sp_mean, half, 37), nan=0.25)
    vx_mean_r12 = np.nan_to_num(roll_nanmean_by_half(vx_mean, half, 12), nan=0.0)
    vx_mean_r37 = np.nan_to_num(roll_nanmean_by_half(vx_mean, half, 37), nan=0.0)

    cols = [
        # Ball-free focus & collective (29 cols)
        d1_focus,
        d2_focus,
        prox_focus_sum,
        v_close_f_top3,
        prox_opp_mean,
        cx,
        cy,
        vx_mean,
        vy_mean,
        sp_mean,
        slow_frac,
        sprint_frac,
        def3_x,
        att3_x,
        sigma_x,
        sigma_y,
        d_cx_12,
        d_cx_25,
        d_sx_25,
        d_sy_25,
        d_def3_25,
        d_att3_25,
        d1_focus_r12,
        d1_focus_r31,
        prox_focus_r12,
        slow_frac_r12,
        slow_frac_r37,
        sp_mean_r12,
        sp_mean_r37,
        vx_mean_r12,
        vx_mean_r37,
        # Ball-aware proximity, gap & knot continuity (14 cols)
        d1_ball,
        d2_ball,
        vc1_ball,
        prox_b3_sum,
        prox_b10_sum,
        d1_last,
        d1_next,
        d1_gap_interp,
        d1_prev_knot,
        d1_next_knot,
        d1_impulse_gated,
        d1_ball_r12,
        d1_ball_r31,
        d1_imp_r12,
        prox_b3_r12,
    ]
    return np.column_stack(cols).astype(np.float32)


def extract_player_set_tensors(m: Match) -> PlayerSetTensors:
    """Extract canonical per-frame player-set tensors `(P_H, P_A, mask_H, mask_A, G_anti, G_sym)`
    from a `Match` with exact `0.000e+00` H<->A equivariance and slot-permutation invariance."""
    nc = filter_ghost_and_switch_slots(m)
    L, W = float(m.pitch[0]), float(m.pitch[1])
    half_pitch = np.array([L / 2.0, W / 2.0], dtype=np.float64)
    P_raw = m.P[..., :2].astype(np.float64)
    B_raw = m.B[:, :2].astype(np.float64)
    if m.direction_normalized:
        P_raw_c = P_raw - half_pitch[None, None, :]
        B_raw_c = B_raw - half_pitch[None, :]
    else:
        sgn_att = np.where(np.asarray(m.attack_sign) < 0, -1.0, 1.0)
        P_raw_c = (P_raw - half_pitch[None, None, :]) * sgn_att[:, None, None]
        B_raw_c = (B_raw - half_pitch[None, :]) * sgn_att[:, None]
    nc_vis = np.isfinite(nc.P[..., 0]) & np.isfinite(nc.P[..., 1])
    P_c = np.where(nc_vis[..., None], P_raw_c, np.nan)
    bvis = nc.ball_xy_visible
    B_c = np.where(bvis[:, None], B_raw_c, np.nan)

    half = np.asarray(nc.half)
    T = len(B_c)

    vis = nc_vis
    V_c = _safe_cascaded_velocity(P_raw_c, vis, half)

    sel_h = nc.slot_team == HOME
    sel_a = nc.slot_team == AWAY
    is_gk = np.asarray(nc.slot_is_gk, dtype=bool)

    # 1. Canonical sort of Home and Away slots by own-frame coordinates (100% slot-order invariant)
    P_h_s, V_h_s, vis_h_s, gk_h_s = _canonical_sort_team_slots(P_c, V_c, vis, is_gk, sel_h, +1.0)
    P_a_s, V_a_s, vis_a_s, gk_a_s = _canonical_sort_team_slots(P_c, V_c, vis, is_gk, sel_a, -1.0)

    # 2. Ball-free pressure convergence focus x_focus(t)
    focus_x, focus_y, focus_vx, focus_vy = _compute_symmetric_focus(
        P_h_s, V_h_s, vis_h_s, gk_h_s,
        P_a_s, V_a_s, vis_a_s, gk_a_s,
        half,
    )

    # 3. Pack up to K=11 active players per team ordered by distance to x_focus(t)
    P_own_h, V_own_h, P_c_h, mask_h = _pack_active_k_players(P_h_s, V_h_s, vis_h_s, focus_x, focus_y, +1.0)
    P_own_a, V_own_a, P_c_a, mask_a = _pack_active_k_players(P_a_s, V_a_s, vis_a_s, focus_x, focus_y, -1.0)

    # Cross-team nearest opponent distance (T, 11, 11)
    dx_ha = P_c_h[:, :, None, 0] - P_c_a[:, None, :, 0]
    dy_ha = P_c_h[:, :, None, 1] - P_c_a[:, None, :, 1]
    dist_ha = np.hypot(dx_ha, dy_ha)
    valid_ha = mask_h[:, :, None] & mask_a[:, None, :]
    dist_ha = np.where(valid_ha, dist_ha, np.inf)

    d_opp1_h = np.min(dist_ha, axis=2)
    d_opp1_a = np.min(dist_ha, axis=1)
    d_opp1_h = np.where(np.isfinite(d_opp1_h), d_opp1_h, 30.0)
    d_opp1_a = np.where(np.isfinite(d_opp1_a), d_opp1_a, 30.0)

    # 4. Ball trajectory interpolation and contact-knot physics
    bvis = nc.ball_xy_visible
    blin = linear_mask(nc.B, half)
    l_starts, l_ends = _linear_runs(nc.B, half)
    is_vertex = np.zeros(T, dtype=bool)
    if len(l_starts):
        is_vertex[l_starts] = True
        is_vertex[np.minimum(l_ends - 1, T - 1)] = True

    last_b_idx, nxt_b_idx = _gap_edges_by_half(bvis, half)
    idx_all = np.arange(T, dtype=np.int64)
    b_prev_xy = B_c[np.clip(last_b_idx, 0, T - 1)]
    b_next_xy = B_c[np.clip(nxt_b_idx, 0, T - 1)]
    has_lb = last_b_idx >= 0
    has_nb = nxt_b_idx < T
    gap_span = np.maximum((nxt_b_idx - last_b_idx).astype(np.float64), 1.0)
    w_b = np.clip((idx_all - last_b_idx) / gap_span, 0.0, 1.0)
    B_interp_c = np.where(
        bvis[:, None],
        B_c,
        np.where(
            (has_lb & has_nb)[:, None],
            (1.0 - w_b[:, None]) * b_prev_xy + w_b[:, None] * b_next_xy,
            np.where(has_lb[:, None], b_prev_xy, np.where(has_nb[:, None], b_next_xy, 0.0)),
        ),
    )

    # Build 18-channel per-player tensors for Home (+1) and Away (-1)
    p_h = _build_team_player_features(
        P_own_h, V_own_h, mask_h, d_opp1_h,
        +focus_x, +focus_y, +B_interp_c[:, 0], +B_interp_c[:, 1], L, W,
    )
    p_a = _build_team_player_features(
        P_own_a, V_own_a, mask_a, d_opp1_a,
        -focus_x, -focus_y, -B_interp_c[:, 0], -B_interp_c[:, 1], L, W,
    )

    # 5. Ball speed, impulses, and contact knots
    j_mag_2, _, _, _ = _ball_impulse_at_delta(B_c, bvis, half, 2)
    j_mag_3, _, _, _ = _ball_impulse_at_delta(B_c, bvis, half, 3)
    j_mag_4, _, _, _ = _ball_impulse_at_delta(B_c, bvis, half, 4)
    j_mag_8, _, _, _ = _ball_impulse_at_delta(B_c, bvis, half, 8)

    sp_ball = np.zeros(T, dtype=np.float64)
    ball_vx = np.zeros(T, dtype=np.float64)
    ball_vy = np.zeros(T, dtype=np.float64)
    if T > 2:
        ok_sp = bvis[2:] & bvis[:-2] & (half[2:] == half[:-2])
        ball_vx[1:-1][ok_sp] = (B_c[2:, 0][ok_sp] - B_c[:-2, 0][ok_sp]) * (FPS / 2.0)
        ball_vy[1:-1][ok_sp] = (B_c[2:, 1][ok_sp] - B_c[:-2, 1][ok_sp]) * (FPS / 2.0)
        sp_ball[1:-1][ok_sp] = np.hypot(ball_vx[1:-1][ok_sp], ball_vy[1:-1][ok_sp])
    same_prev = np.r_[False, bvis[1:] & bvis[:-1] & (half[1:] == half[:-1])]
    same_next = np.r_[bvis[1:] & bvis[:-1] & (half[1:] == half[:-1]), False]
    sp_1 = np.zeros(T, dtype=np.float64)
    sp_1[1:][same_prev[1:]] = np.hypot(
        B_c[1:, 0][same_prev[1:]] - B_c[:-1, 0][same_prev[1:]],
        B_c[1:, 1][same_prev[1:]] - B_c[:-1, 1][same_prev[1:]],
    ) * FPS
    sp_ball = np.where(sp_ball > 0.0, sp_ball, sp_1)

    d1_h_raw = np.min(np.where(mask_h, p_h[..., 14].astype(np.float64) * 30.0, 30.0), axis=1)
    d1_a_raw = np.min(np.where(mask_a, p_a[..., 14].astype(np.float64) * 30.0, 30.0), axis=1)
    d_min = np.minimum(d1_h_raw, d1_a_raw)

    gap_boundary = bvis & ((~same_prev) | (~same_next))
    sig_impulse = (j_mag_3 >= 3.0) | (j_mag_2 >= 3.0) | is_vertex | ((~blin) & (sp_ball < 3.5))
    contact_knot = (bvis & (d_min < 2.2) & sig_impulse) | (gap_boundary & (d_min < 3.5))

    tau_prev, tau_next = _gap_edges_by_half(contact_knot, half)
    has_prev = tau_prev >= 0
    has_next = tau_next < T
    tp_safe = np.where(has_prev, np.clip(tau_prev, 0, T - 1), np.where(has_next, np.clip(tau_next, 0, T - 1), idx_all))
    tn_safe = np.where(has_next, np.clip(tau_next, 0, T - 1), np.where(has_prev, np.clip(tau_prev, 0, T - 1), idx_all))

    chord_dt_s = np.maximum((tn_safe - tp_safe).astype(np.float64) / FPS, 1.0 / FPS)
    chord_dist = np.hypot(B_interp_c[tn_safe, 0] - B_interp_c[tp_safe, 0], B_interp_c[tn_safe, 1] - B_interp_c[tp_safe, 1])
    chord_speed = np.where(tn_safe > tp_safe, chord_dist / chord_dt_s, sp_ball)
    v_eff = np.where(bvis & (~blin), np.maximum(sp_ball, chord_speed), chord_speed)

    dist_from_knot_frames = np.minimum(
        np.where(has_prev, idx_all - tau_prev, 10),
        np.where(has_next, tau_next - idx_all, 10),
    ).astype(np.float64)
    s_knot = np.where(contact_knot, 0.0, np.clip(dist_from_knot_frames / 2.0, 0.0, 1.0))
    s_vel = np.clip((v_eff - 3.5) / 5.0, 0.0, 1.0)
    s_low_j = np.where(bvis & (~blin | is_vertex), np.clip(1.0 - j_mag_2 / 4.5, 0.0, 1.0), 1.0)
    chord_in_flight = (s_knot * s_vel * s_low_j).astype(np.float64)

    span_knot = np.maximum((tn_safe - tp_safe).astype(np.float64), 1.0)
    alpha_next = np.where(
        has_prev & has_next & (tn_safe > tp_safe),
        np.clip((idx_all - tp_safe) / span_knot, 0.0, 1.0),
        np.where(has_prev, 0.0, 1.0),
    )

    # 6. Team summary scalars Q_H and Q_A -> Q_H - Q_A (antisymmetric) and Q_H + Q_A (symmetric)
    Q_h = _team_summary_scalars(
        p_h, mask_h, half, last_b_idx, nxt_b_idx, w_b, bvis, tp_safe, tn_safe, alpha_next, chord_in_flight,
    )
    Q_a = _team_summary_scalars(
        p_a, mask_a, half, last_b_idx, nxt_b_idx, w_b, bvis, tp_safe, tn_safe, alpha_next, chord_in_flight,
    )
    Q_diff = Q_h - Q_a
    Q_sum = Q_h + Q_a

    # 7. Pure global antisymmetric and symmetric context features
    prev_diff = d1_h_raw[tp_safe] - d1_a_raw[tp_safe]
    next_diff = d1_h_raw[tn_safe] - d1_a_raw[tn_safe]
    p_h_prev = 1.0 / (1.0 + np.exp(np.clip(prev_diff / 1.2, -30.0, 30.0)))
    p_a_prev = 1.0 / (1.0 + np.exp(np.clip(-prev_diff / 1.2, -30.0, 30.0)))
    p_h_next = 1.0 / (1.0 + np.exp(np.clip(next_diff / 1.2, -30.0, 30.0)))
    p_a_next = 1.0 / (1.0 + np.exp(np.clip(-next_diff / 1.2, -30.0, 30.0)))
    chord_same_diff = p_h_prev * p_h_next - p_a_prev * p_a_next
    chord_turnover = p_h_prev * p_a_next + p_a_prev * p_h_next
    chord_same_diff_r25 = np.nan_to_num(roll_nanmean_by_half(chord_same_diff, half, 25), nan=0.0)

    g_anti_extra = np.column_stack([
        np.clip(focus_x / (L / 2.0), -1.5, 1.5),
        np.clip(focus_y / (W / 2.0), -1.5, 1.5),
        np.clip(focus_vx / 10.0, -1.5, 1.5),
        np.clip(focus_vy / 10.0, -1.5, 1.5),
        np.clip(B_interp_c[:, 0] / (L / 2.0), -1.5, 1.5),
        np.clip(B_interp_c[:, 1] / (W / 2.0), -1.5, 1.5),
        np.clip(ball_vx / 20.0, -1.5, 1.5),
        np.clip(ball_vy / 20.0, -1.5, 1.5),
        chord_same_diff,
        chord_same_diff_r25,
    ]).astype(np.float32)

    # Ghost-ball & blackout reliability features (strictly invariant under 180-deg rotation)
    ball_focus_dist = np.hypot(B_interp_c[:, 0] - focus_x, B_interp_c[:, 1] - focus_y) / 25.0
    ball_focus_dist = np.clip(ball_focus_dist, 0.0, 2.5)
    ghost_slow_far = np.where(
        bvis,
        np.exp(-sp_ball / 2.0) * np.clip((d_min - 2.5) / 5.0, 0.0, 2.0),
        0.0,
    )
    ghost_unassisted_j = np.where(
        bvis,
        np.clip((j_mag_2 - 3.0) / 10.0, 0.0, 2.0) * np.clip((d_min - 2.5) / 5.0, 0.0, 2.0),
        0.0,
    )
    bf_dist_r12 = np.nan_to_num(roll_nanmean_by_half(ball_focus_dist, half, 12), nan=0.5)
    bf_dist_r37 = np.nan_to_num(roll_nanmean_by_half(ball_focus_dist, half, 37), nan=0.5)
    ghost_sf_r12 = np.nan_to_num(roll_nanmean_by_half(ghost_slow_far, half, 12), nan=0.0)
    ghost_sf_r37 = np.nan_to_num(roll_nanmean_by_half(ghost_slow_far, half, 37), nan=0.0)
    ghost_uj_r12 = np.nan_to_num(roll_nanmean_by_half(ghost_unassisted_j, half, 12), nan=0.0)

    gap_len = np.where(bvis, 0.0, np.clip(gap_span / 125.0, 0.0, 2.0))
    dist_gap_edge = np.where(
        bvis,
        0.0,
        np.clip(np.minimum(idx_all - last_b_idx, nxt_b_idx - idx_all) / 50.0, 0.0, 2.0),
    )

    g_sym_extra = np.column_stack([
        bvis.astype(np.float32),
        blin.astype(np.float32),
        chord_in_flight.astype(np.float32),
        chord_turnover.astype(np.float32),
        np.clip(sp_ball / 20.0, 0.0, 2.0),
        np.clip(j_mag_2 / 15.0, 0.0, 2.0),
        np.clip(j_mag_4 / 15.0, 0.0, 2.0),
        np.clip(j_mag_8 / 15.0, 0.0, 2.0),
        np.clip(d_min / 15.0, 0.0, 2.0),
        ball_focus_dist,
        bf_dist_r12,
        bf_dist_r37,
        ghost_slow_far,
        ghost_sf_r12,
        ghost_sf_r37,
        ghost_unassisted_j,
        ghost_uj_r12,
        gap_len,
        dist_gap_edge,
        np.clip(np.abs(focus_x) / (L / 2.0), 0.0, 1.5),
        np.clip(np.abs(focus_y) / (W / 2.0), 0.0, 1.5),
        np.clip(np.abs(B_interp_c[:, 0]) / (L / 2.0), 0.0, 1.5),
        np.clip(np.abs(B_interp_c[:, 1]) / (W / 2.0), 0.0, 1.5),
    ]).astype(np.float32)

    g_anti = np.concatenate([Q_diff, g_anti_extra], axis=1).astype(np.float32)
    g_sym = np.concatenate([Q_sum, g_sym_extra], axis=1).astype(np.float32)

    return PlayerSetTensors(
        p_h=np.nan_to_num(p_h, nan=0.0, posinf=0.0, neginf=0.0),
        p_a=np.nan_to_num(p_a, nan=0.0, posinf=0.0, neginf=0.0),
        mask_h=mask_h,
        mask_a=mask_a,
        g_anti=np.nan_to_num(g_anti, nan=0.0, posinf=0.0, neginf=0.0),
        g_sym=np.nan_to_num(g_sym, nan=0.0, posinf=0.0, neginf=0.0),
    )


def cached_player_set_tensors(
    m: Match,
    cache_dir: str | Path | None,
    tag: str = "",
) -> PlayerSetTensors:
    """Load or compute and cache `PlayerSetTensors` for match `m`."""
    if cache_dir is None:
        return extract_player_set_tensors(m)
    prefix = tag or m.source or "match"
    p = Path(cache_dir) / f"pset_v{VERSION}_{prefix}_{match_fingerprint(m)}.npz"
    if p.exists():
        data = np.load(p)
        return PlayerSetTensors(
            p_h=data["p_h"],
            p_a=data["p_a"],
            mask_h=data["mask_h"],
            mask_a=data["mask_a"],
            g_anti=data["g_anti"],
            g_sym=data["g_sym"],
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    pst = extract_player_set_tensors(m)
    np.savez(
        p,
        p_h=pst.p_h,
        p_a=pst.p_a,
        mask_h=pst.mask_h,
        mask_a=pst.mask_a,
        g_anti=pst.g_anti,
        g_sym=pst.g_sym,
    )
    return pst


class _SplitPlayerPointNet(nn.Module):
    """Siamese PointNet / DeepSets player encoder split into:
    1) A 100% ball-free branch (`phi_bf`, reading channels `0..11`) + ball-free attention head.
    2) A ball-aware branch (`phi_ball`, reading all 18 channels) + ball-proximity attention head.
    Aggregates over the K=11 players via masked `mean` + masked `max` + 2-head masked `softmax-attention`.
    """

    def __init__(self, d_bf: int = 24, d_ball: int = 24, hidden: int = 48):
        super().__init__()
        self.d_bf = d_bf
        self.d_ball = d_ball
        self.d_player = d_bf + d_ball
        self.out_dim = 3 * self.d_player  # mean (48) + max (48) + [att_bf (24), att_ball (24)] = 144

        self.net_bf = nn.Sequential(
            nn.Conv2d(C_PLAYER_BF, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, d_bf + 1, kernel_size=1),
        )
        self.net_ball = nn.Sequential(
            nn.Conv2d(C_PLAYER_TOTAL, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, d_ball + 1, kernel_size=1),
        )

    def forward(self, p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        p:    (B, C_p=18, K=11, L)
        mask: (B, K=11, L) bool
        Returns z: (B, 144, L)
        """
        m_bool = mask.unsqueeze(1)               # (B, 1, K, L)
        m_float = m_bool.to(p.dtype)             # (B, 1, K, L)
        n_vis = m_float.sum(dim=2).clamp(min=1.0)  # (B, 1, L)
        any_vis = (m_float.sum(dim=2) > 0.5).to(p.dtype)  # (B, 1, L)

        out_bf = self.net_bf(p[:, :C_PLAYER_BF, :, :])   # (B, d_bf + 1, K, L)
        e_bf = out_bf[:, :self.d_bf, :, :] * m_float
        logit_bf = out_bf[:, self.d_bf:, :, :]

        out_ball = self.net_ball(p)                      # (B, d_ball + 1, K, L)
        e_ball = out_ball[:, :self.d_ball, :, :] * m_float
        logit_ball = out_ball[:, self.d_ball:, :, :]

        e_all = torch.cat([e_bf, e_ball], dim=1)         # (B, 48, K, L)

        # 1. Masked Mean Pool over K players
        z_mean = e_all.sum(dim=2) / n_vis                # (B, 48, L)

        # 2. Masked Max Pool over K players
        e_masked_max = e_all.masked_fill(~m_bool, -1e4)
        z_max = e_masked_max.max(dim=2).values * any_vis  # (B, 48, L)

        # 3. Masked Ball-Free Attention Pool over K players
        w_bf = F.softmax(logit_bf.masked_fill(~m_bool, -1e4), dim=2) * m_float
        w_bf = w_bf / w_bf.sum(dim=2, keepdim=True).clamp(min=1e-6)
        z_att_bf = (e_bf * w_bf).sum(dim=2)              # (B, 24, L)

        # 4. Masked Ball-Aware Attention Pool over K players
        w_ball = F.softmax(logit_ball.masked_fill(~m_bool, -1e4), dim=2) * m_float
        w_ball = w_ball / w_ball.sum(dim=2, keepdim=True).clamp(min=1e-6)
        z_att_ball = (e_ball * w_ball).sum(dim=2)        # (B, 24, L)

        return torch.cat([z_mean, z_max, z_att_bf, z_att_ball], dim=1)


class _PlayerSetBackbone(nn.Module):
    """Intrinsically H<->A + 180-deg Equivariant PointNet + Ghost-Gated Dilated Temporal Conv + Strided BiGRU."""

    def __init__(
        self,
        c_ga: int,
        c_gs: int,
        d_bf: int = 24,
        d_ball: int = 24,
        channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        gru_stride: int = 10,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.pointnet = _SplitPlayerPointNet(d_bf=d_bf, d_ball=d_ball, hidden=48)
        d_pool = self.pointnet.out_dim  # 144

        # Symmetric ghost-ball / blackout reliability gate modulating pooled player features
        self.ghost_gate = nn.Sequential(
            nn.Conv1d(c_gs, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, d_pool, kernel_size=1),
            nn.Sigmoid(),
        )

        # Equivariant twin projection (bias=False on antisymmetric stream!)
        self.inp_anti = nn.Conv1d(d_pool + c_ga, channels, kernel_size=1, bias=False)
        self.inp_sym = nn.Conv1d(d_pool + c_gs, channels, kernel_size=1, bias=True)
        self.in_norm = ChannelNorm(channels)
        self.in_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[_ResBlock(channels, d, dropout) for d in dilations])

        self.gru_stride = gru_stride
        gru_hidden = channels // 2
        self.bigru = nn.GRU(
            input_size=channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            ChannelNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_poss = nn.Conv1d(channels, 3, kernel_size=1)
        self.head_chg = nn.Conv1d(channels, 1, kernel_size=1)

    def _forward_stream(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.in_drop(F.gelu(self.in_norm(x))))

        L = h.shape[-1]
        s = self.gru_stride
        h_ds = F.avg_pool1d(h, kernel_size=s, stride=s, ceil_mode=True)
        gru_in = h_ds.transpose(1, 2)
        gru_out, _ = self.bigru(gru_in)
        gru_feat = gru_out.transpose(1, 2)
        gru_up = F.interpolate(gru_feat, size=L, mode="linear", align_corners=False)

        h = h + self.gru_proj(gru_up)
        return self.head_poss(h), self.head_chg(h).squeeze(1)

    def forward(
        self,
        p_h: torch.Tensor,
        p_a: torch.Tensor,
        m_h: torch.Tensor,
        m_a: torch.Tensor,
        g_anti: torch.Tensor,
        g_sym: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_both = torch.cat([p_h, p_a], dim=0)
        m_both = torch.cat([m_h, m_a], dim=0)
        z_h, z_a = self.pointnet(p_both, m_both).chunk(2, dim=0)

        gate = self.ghost_gate(g_sym)
        z_h = z_h * gate
        z_a = z_a * gate

        u_anti = torch.cat([z_h - z_a, g_anti], dim=1)
        u_sym = torch.cat([z_h + z_a, g_sym], dim=1)

        h_anti = self.inp_anti(u_anti)
        h_sym = self.inp_sym(u_sym)

        x_both = torch.cat([h_sym + h_anti, h_sym - h_anti], dim=0)
        lp_both, lc_both = self._forward_stream(x_both)
        lp1, lp2 = lp_both.chunk(2, dim=0)
        lc1, lc2 = lc_both.chunk(2, dim=0)

        logits_poss = 0.5 * (lp1 + lp2[:, [1, 0, 2], :])
        logits_chg = 0.5 * (lc1 + lc2)
        return logits_poss, logits_chg


class SymPlayerSetNet:
    """End-to-end wrapper for training and evaluating `SymPlayerSetNet` on `PlayerSetTensors`."""

    def __init__(
        self,
        channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        gru_stride: int = 10,
        dropout: float = 0.15,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-4,
        epochs: int = 10,
        crop: int = 1500,
        batch_size: int = 24,
        change_loss_weight: float = 0.35,
        label_smoothing: float = 0.02,
        seed: int = 42,
    ):
        self.channels = channels
        self.dilations = dilations
        self.gru_stride = gru_stride
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.crop = crop
        self.batch_size = batch_size
        self.change_loss_weight = change_loss_weight
        self.label_smoothing = label_smoothing
        self.seed = seed

        self.c_ga: int = 0
        self.c_gs: int = 0
        self.mean_ga_: np.ndarray | None = None
        self.std_ga_: np.ndarray | None = None
        self.mean_gs_: np.ndarray | None = None
        self.std_gs_: np.ndarray | None = None
        self.net_: _PlayerSetBackbone | None = None

    def _device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _fit_scaler(self, psts: list[PlayerSetTensors]) -> None:
        n_total = sum(len(x.g_anti) for x in psts)
        c_ga = psts[0].g_anti.shape[1]
        c_gs = psts[0].g_sym.shape[1]
        self.c_ga = c_ga
        self.c_gs = c_gs

        # Antisymmetric features have exact theoretical mean 0.0; scale by RMS
        sq_ga = np.zeros(c_ga, dtype=np.float64)
        sum_gs = np.zeros(c_gs, dtype=np.float64)
        sq_gs = np.zeros(c_gs, dtype=np.float64)

        for x in psts:
            ga64 = x.g_anti.astype(np.float64)
            gs64 = x.g_sym.astype(np.float64)
            sq_ga += np.sum(ga64 * ga64, axis=0)
            sum_gs += np.sum(gs64, axis=0)
            sq_gs += np.sum(gs64 * gs64, axis=0)

        self.mean_ga_ = np.zeros(c_ga, dtype=np.float32)
        rms_ga = np.sqrt(sq_ga / max(n_total, 1))
        self.std_ga_ = np.where(rms_ga > 1e-6, rms_ga, 1.0).astype(np.float32)

        mean_gs = sum_gs / max(n_total, 1)
        var_gs = np.maximum(sq_gs / max(n_total, 1) - mean_gs * mean_gs, 0.0)
        std_gs = np.sqrt(var_gs)
        self.mean_gs_ = mean_gs.astype(np.float32)
        self.std_gs_ = np.where(std_gs > 1e-6, std_gs, 1.0).astype(np.float32)

    def _prep_match_arrays(
        self, pst: PlayerSetTensors
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return time-first C-contiguous arrays for zero-copy time slicing:
        p_h: (T, 11, 18), p_a: (T, 11, 18), m_h: (T, 11), m_a: (T, 11), g_anti: (T, C_ga), g_sym: (T, C_gs).
        """
        assert self.mean_gs_ is not None and self.std_gs_ is not None and self.std_ga_ is not None
        p_h = np.ascontiguousarray(np.clip(pst.p_h, -5.0, 5.0), dtype=np.float32)
        p_a = np.ascontiguousarray(np.clip(pst.p_a, -5.0, 5.0), dtype=np.float32)
        m_h = np.ascontiguousarray(pst.mask_h, dtype=bool)
        m_a = np.ascontiguousarray(pst.mask_a, dtype=bool)

        ga = np.ascontiguousarray(np.clip(pst.g_anti / self.std_ga_[None, :], -8.0, 8.0), dtype=np.float32)
        gs = np.ascontiguousarray(
            np.clip((pst.g_sym - self.mean_gs_[None, :]) / self.std_gs_[None, :], -8.0, 8.0),
            dtype=np.float32,
        )
        return p_h, p_a, m_h, m_a, ga, gs

    def fit(
        self,
        psts: list[PlayerSetTensors],
        ys: list[np.ndarray],
        halves: list[np.ndarray],
        sample_weights: list[float] | None = None,
    ) -> "SymPlayerSetNet":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._fit_scaler(psts)

        m_weights = sample_weights if sample_weights is not None else [1.0] * len(psts)
        seqs = []
        weights = []
        for pst, y, h, w_m in zip(psts, ys, halves, m_weights):
            ph, pa, mh, ma, ga, gs = self._prep_match_arrays(pst)
            yc = change_targets(y, h).astype(np.float32)
            yi = y.astype(np.int64)
            for hv in np.unique(h[h > 0]):
                idx = np.flatnonzero(h == hv)
                if len(idx) == 0:
                    continue
                s, e = int(idx[0]), int(idx[-1]) + 1
                if e - s < 32:
                    continue
                seqs.append((ph[s:e], pa[s:e], mh[s:e], ma[s:e], ga[s:e], gs[s:e], yi[s:e], yc[s:e]))
                weights.append(float(e - s) * float(w_m))

        dev = self._device()
        net = _PlayerSetBackbone(
            c_ga=self.c_ga,
            c_gs=self.c_gs,
            channels=self.channels,
            dilations=self.dilations,
            gru_stride=self.gru_stride,
            dropout=self.dropout,
        ).to(dev)

        total_frames = sum(len(s[0]) for s in seqs)
        steps_per_epoch = max(1, total_frames // (self.crop * self.batch_size))
        total_steps = self.epochs * steps_per_epoch

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=total_steps)

        probs = np.array(weights, dtype=np.float64)
        probs = probs / probs.sum()
        pos_w = torch.tensor(10.0, device=dev)
        L = self.crop

        net.train()
        for _ in range(total_steps):
            chosen = rng.choice(len(seqs), self.batch_size, p=probs)
            l_ph, l_pa, l_mh, l_ma, l_ga, l_gs, l_y, l_c = [], [], [], [], [], [], [], []
            for si in chosen:
                ph, pa, mh, ma, ga, gs, yi, yc = seqs[si]
                n = len(ph)
                if n >= L:
                    a = int(rng.integers(0, max(1, n - L + 1)))
                    l_ph.append(ph[a:a + L])
                    l_pa.append(pa[a:a + L])
                    l_mh.append(mh[a:a + L])
                    l_ma.append(ma[a:a + L])
                    l_ga.append(ga[a:a + L])
                    l_gs.append(gs[a:a + L])
                    l_y.append(yi[a:a + L])
                    l_c.append(yc[a:a + L])
                else:
                    pad = L - n
                    l_ph.append(np.pad(ph, ((0, pad), (0, 0), (0, 0))))
                    l_pa.append(np.pad(pa, ((0, pad), (0, 0), (0, 0))))
                    l_mh.append(np.pad(mh, ((0, pad), (0, 0))))
                    l_ma.append(np.pad(ma, ((0, pad), (0, 0))))
                    l_ga.append(np.pad(ga, ((0, pad), (0, 0))))
                    l_gs.append(np.pad(gs, ((0, pad), (0, 0))))
                    l_y.append(np.pad(yi, (0, pad), constant_values=-1))
                    l_c.append(np.pad(yc, (0, pad), constant_values=-1.0))

            # Stack contiguous time-first arrays (B, L, ...) and transpose via zero-copy view on GPU
            th_ph = torch.from_numpy(np.stack(l_ph)).to(dev).permute(0, 3, 2, 1)  # (B, 18, 11, L)
            th_pa = torch.from_numpy(np.stack(l_pa)).to(dev).permute(0, 3, 2, 1)  # (B, 18, 11, L)
            th_mh = torch.from_numpy(np.stack(l_mh)).to(dev).permute(0, 2, 1)     # (B, 11, L)
            th_ma = torch.from_numpy(np.stack(l_ma)).to(dev).permute(0, 2, 1)     # (B, 11, L)
            th_ga = torch.from_numpy(np.stack(l_ga)).to(dev).permute(0, 2, 1)     # (B, C_ga, L)
            th_gs = torch.from_numpy(np.stack(l_gs)).to(dev).permute(0, 2, 1)     # (B, C_gs, L)
            ty = torch.from_numpy(np.stack(l_y)).to(dev)
            tc = torch.from_numpy(np.stack(l_c)).to(dev)

            logits_p, logits_c = net(th_ph, th_pa, th_mh, th_ma, th_ga, th_gs)

            loss_p = F.cross_entropy(
                logits_p,
                ty.masked_fill(ty < 0, -100),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )
            c_mask = (tc >= 0).to(torch.float32)
            bce = F.binary_cross_entropy_with_logits(
                logits_c, tc.clamp(min=0.0), pos_weight=pos_w, reduction="none"
            )
            loss_c = (bce * c_mask).sum() / c_mask.sum().clamp(min=1.0)

            loss = loss_p + self.change_loss_weight * loss_c
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()

        self.net_ = net.to("cpu").eval()
        return self

    def predict_proba_and_change(
        self, pst: PlayerSetTensors, half: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.net_ is not None
        ph, pa, mh, ma, ga, gs = self._prep_match_arrays(pst)
        T = len(ph)
        proba = np.zeros((T, 3), dtype=np.float32)
        chg = np.zeros(T, dtype=np.float32)

        dev = self._device()
        net = self.net_.to(dev).eval()
        with torch.no_grad():
            for hid in np.unique(half):
                idx = np.flatnonzero(half == hid)
                if len(idx) == 0:
                    continue
                s, e = int(idx[0]), int(idx[-1]) + 1
                th_ph = torch.from_numpy(ph[s:e][None]).to(dev).permute(0, 3, 2, 1)
                th_pa = torch.from_numpy(pa[s:e][None]).to(dev).permute(0, 3, 2, 1)
                th_mh = torch.from_numpy(mh[s:e][None]).to(dev).permute(0, 2, 1)
                th_ma = torch.from_numpy(ma[s:e][None]).to(dev).permute(0, 2, 1)
                th_ga = torch.from_numpy(ga[s:e][None]).to(dev).permute(0, 2, 1)
                th_gs = torch.from_numpy(gs[s:e][None]).to(dev).permute(0, 2, 1)

                lp, lc = net(th_ph, th_pa, th_mh, th_ma, th_ga, th_gs)
                p = F.softmax(lp, dim=1).squeeze(0).T.cpu().numpy()
                c = torch.sigmoid(lc).squeeze(0).cpu().numpy()
                proba[s:e] = p
                chg[s:e] = c

        self.net_ = net.to("cpu").eval()
        return proba, chg
