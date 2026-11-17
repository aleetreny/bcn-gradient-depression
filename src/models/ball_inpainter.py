"""Wave 8A (EXP-14): `SymBallInpainterNet` — Clean-Ball Teacher Distillation in `ball_gap` (`src/models/ball_inpainter.py`).

Implements the two-stage physically equivariant ball-inpainting and gap-distillation architecture:

1. Part A — `SymBallPositionInpainter`:
   - Predicts the 2D spatial residual vector `(dx_hat(t), dy_hat(t)) = x_ball^clean(t) - x_ball^interp(t)`
     during missing-ball gaps (`~ball_visible` / `ball_gap > 0`) and ghost-ball episodes (`w_ghost > 0.15`).
   - Exact `0.000e+00` Equivariance Construction under 180-deg pitch rotation + `H <-> A` swap:
     Expands `(dx_hat(t), dy_hat(t)) = sum_{m=1}^{11} w_m(s_inv(t)) * v_m(t)`, where:
       * `v_m(t) in R^2` are 11 physically equivariant 2D basis vectors anchored at `x_ball^interp(t)`
         (offsets to player-convergence focus `x_focus`, sharp/medium/smoothed focus-player centroids,
         sharp/medium chord-player centroids, 22-player global centroid, player-focus curvature relative
         to the gap-knot chord, gap midpoint vector, and cubic Hermite velocity tangent vector), all
         computed via symmetric `(H + A)` reductions so `v_m(swap(M)) == -v_m(M)` with `0.000e+00` error.
       * `s_inv(t) in R^26` are 26 strictly `H <-> A` and 180-deg rotation-invariant scalar features,
         satisfying `s_inv(swap(M)) == s_inv(M)` with `0.000e+00` error.
       * `InpaintBasisMLP` maps `s_inv(t) -> w(t) in R^11` and minimizes a smooth Charbonnier + L2 spatial
         loss against the true clean ball `x_ball^clean(t)` in strict 7-fold LOMO on `Sportec`.

2. Part B — `SymGapDistilledExpert`:
   - Reconstructs the inpainted ball trajectory:
     `x_ball^inpaint(t) = x_ball^interp(t) + g_apply(t) * (dx_hat(t), dy_hat(t))`
   - Recomputes a 78-column strictly equivariant table (`build_inpainted_feature_df`) of player-to-inpainted-ball
     distances (`d1..d3`), exponential pitch control shares (`r = 1.5m, 3.0m, 6.0m, 10.0m`), closing velocities,
     inpainted ball kinematics, inpainting distance gains (`d1_interp - d1_inpaint`), gap-knot bridges, and
     multi-scale rolling means (`r15, r45, r125`).
   - Trains a Clean-Ball Teacher on true clean ball coordinates `x_ball^clean(t)` and distills its calibrated
     soft probabilities `p_teacher^clean(t)` + ground-truth labels `y(t)` into `SymGapDistilledExpert`
     on `ball_gap > 0` and `w_ghost > 0.15` frames in strict 7-fold LOMO, with exact `0.000e+00` `H <-> A` equivariance.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

from src.data.common import AWAY, DEAD, FPS, HOME, Match
from src.features.knot_physics import filter_ghost_and_switch_slots
from src.features.v1ctl_physics_v2 import _gaussian_smooth_by_half
from src.features.v1ctl_plus import _gap_edges_by_half, roll_nanmean_by_half
from src.models.player_set_net import (
    _canonical_sort_team_slots,
    _compute_symmetric_focus,
    _pack_active_k_players,
    _safe_cascaded_velocity,
)


def _half_time_diff_2d(arr: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Compute centered 2-frame velocity (m/s) within each half with 1-frame fallback at boundaries."""
    T = len(half)
    out = np.zeros_like(arr, dtype=np.float64)
    if T > 2:
        ok2 = half[2:] == half[:-2]
        out[1:-1][ok2] = (arr[2:][ok2] - arr[:-2][ok2]) * (FPS / 2.0)
    if T > 1:
        ok1 = half[1:] == half[:-1]
        need_f = np.zeros(T, dtype=bool)
        need_f[0] = True
        need_f[1:-1] = ~(half[2:] == half[:-2])
        mask_f = need_f[:-1] & ok1
        out[:-1][mask_f] = (arr[1:][mask_f] - arr[:-1][mask_f]) * float(FPS)
        need_b = np.zeros(T, dtype=bool)
        need_b[-1] = True
        need_b[1:-1] = ~(half[2:] == half[:-2])
        mask_b = need_b[1:] & ok1
        out[1:][mask_b] = (arr[1:][mask_b] - arr[:-1][mask_b]) * float(FPS)
    return out


def extract_centered_kinematics(m: Match) -> dict[str, np.ndarray]:
    """Extract canonical pitch-centered 2D coordinates, velocities, and gap-knot boundary states from `Match`.

    Under `swap_match(m)`:
    - `H_c -> -A_c`, `A_c -> -H_c`, `VH -> -VA`, `VA -> -VH`, `mask_h -> mask_a`, `mask_a -> mask_h`
    - `B_raw_c -> -B_raw_c`, `B_int_c -> -B_int_c`, `VB_int -> -VB_int`
    - `F_c -> -F_c`
    All with exact `0.000e+00` IEEE-754 accuracy and `0.000e+00` slot-permutation invariance.
    """
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
    bvis = np.asarray(nc.ball_xy_visible, dtype=bool)
    B_c = np.where(bvis[:, None], B_raw_c, np.nan)

    half = np.asarray(nc.half)
    T = len(B_c)
    V_c = _safe_cascaded_velocity(P_raw_c, nc_vis, half)

    sel_h = nc.slot_team == HOME
    sel_a = nc.slot_team == AWAY
    is_gk = np.asarray(nc.slot_is_gk, dtype=bool)

    P_h_s, V_h_s, vis_h_s, gk_h_s = _canonical_sort_team_slots(P_c, V_c, nc_vis, is_gk, sel_h, +1.0)
    P_a_s, V_a_s, vis_a_s, gk_a_s = _canonical_sort_team_slots(P_c, V_c, nc_vis, is_gk, sel_a, -1.0)

    focus_x, focus_y, focus_vx, focus_vy = _compute_symmetric_focus(
        P_h_s, V_h_s, vis_h_s, gk_h_s,
        P_a_s, V_a_s, vis_a_s, gk_a_s,
        half,
    )
    F_c = np.column_stack([focus_x, focus_y])

    P_own_h, V_own_h, H_c, mask_h = _pack_active_k_players(P_h_s, V_h_s, vis_h_s, focus_x, focus_y, +1.0)
    P_own_a, V_own_a, A_c, mask_a = _pack_active_k_players(P_a_s, V_a_s, vis_a_s, focus_x, focus_y, -1.0)
    VH = +1.0 * V_own_h
    VA = -1.0 * V_own_a

    # Linear ball interpolation within each half
    last_b_idx, nxt_b_idx = _gap_edges_by_half(bvis, half)
    idx_all = np.arange(T, dtype=np.int64)
    idx_prev = np.clip(last_b_idx, 0, T - 1)
    idx_next = np.clip(nxt_b_idx, 0, T - 1)
    has_lb = last_b_idx >= 0
    has_nb = nxt_b_idx < T
    gap_span = np.maximum((nxt_b_idx - last_b_idx).astype(np.float64), 1.0)
    tau = np.where(bvis, 0.0, np.clip((idx_all - last_b_idx) / gap_span, 0.0, 1.0))
    gap_len = np.where(bvis, 0.0, np.where(has_lb & has_nb, gap_span, 0.0))

    B_zero = np.nan_to_num(B_c, nan=0.0)
    b_prev_xy = B_zero[idx_prev]
    b_next_xy = B_zero[idx_next]
    B_int_c = np.where(
        bvis[:, None],
        B_zero,
        np.where(
            (has_lb & has_nb)[:, None],
            (1.0 - tau[:, None]) * b_prev_xy + tau[:, None] * b_next_xy,
            np.where(has_lb[:, None], b_prev_xy, np.where(has_nb[:, None], b_next_xy, 0.0)),
        ),
    )
    VB_int = _half_time_diff_2d(B_int_c, half)

    # For interior gap frames, ensure idx_prev and idx_next point to valid boundary frames
    idx_prev_safe = np.where(has_lb, idx_prev, idx_next)
    idx_next_safe = np.where(has_nb, idx_next, idx_prev)

    return {
        "H_c": H_c,
        "A_c": A_c,
        "VH": VH,
        "VA": VA,
        "mask_h": mask_h,
        "mask_a": mask_a,
        "B_raw_c": B_c,
        "B_int_c": B_int_c,
        "VB_int": VB_int,
        "F_c": F_c,
        "bvis": bvis,
        "idx_prev": idx_prev_safe,
        "idx_next": idx_next_safe,
        "gap_len": gap_len,
        "tau": tau,
        "half": half,
        "L": L,
        "W": W,
    }


def _soft_22_centroid(
    H_c: np.ndarray,
    A_c: np.ndarray,
    mask_h: np.ndarray,
    mask_a: np.ndarray,
    d_h: np.ndarray,
    d_a: np.ndarray,
    temp: float,
) -> np.ndarray:
    """Compute permutation-invariant and strictly H<->A equivariant soft centroid over all 22 players.

    Sums within sorted Home and sorted Away first, then adds `(sum_H + sum_A) / (w_H + w_A)`
    so IEEE-754 addition commutes bit-for-bit (`0.000e+00`) under `H <-> A` swap.
    """
    min_h = np.min(np.where(mask_h, d_h, 1e6), axis=1, keepdims=True)
    min_a = np.min(np.where(mask_a, d_a, 1e6), axis=1, keepdims=True)
    d_min = np.minimum(min_h, min_a)

    w_h = np.where(mask_h, np.exp(-np.clip(d_h - d_min, 0.0, 60.0) / temp), 0.0)
    w_a = np.where(mask_a, np.exp(-np.clip(d_a - d_min, 0.0, 60.0) / temp), 0.0)

    sw_h = np.sum(w_h, axis=1, keepdims=True)
    sw_a = np.sum(w_a, axis=1, keepdims=True)
    w_tot = np.maximum(sw_h + sw_a, 1e-12)

    sx_h = np.sum(w_h[:, :, None] * H_c, axis=1)
    sx_a = np.sum(w_a[:, :, None] * A_c, axis=1)
    return (sx_h + sx_a) / w_tot


def build_inpaint_basis_and_scalars(
    m: Match,
    w_ghost: np.ndarray | None = None,
    kin: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Construct the 11 equivariant 2D basis vectors `V_basis (T, 11, 2)` and 26 invariant scalars `S_inv (T, 26)`.

    Mathematical Equivariance Guarantee:
    - Under `swap_match(m)`:
      * `V_basis(swap(m)) == -V_basis(m)` (`0.000e+00` error)
      * `S_inv(swap(m))   == +S_inv(m)`   (`0.000e+00` error)
    - Therefore, for ANY scalar weight network `w = MLP(S_inv)`:
      `delta_xy = sum_m w_m(S_inv) * V_basis[:, m, :]` satisfies
      `delta_xy(swap(m)) == -delta_xy(m)` with exact `0.000e+00` error.
    """
    if kin is None:
        kin = extract_centered_kinematics(m)
    H_c = kin["H_c"]
    A_c = kin["A_c"]
    VH = kin["VH"]
    VA = kin["VA"]
    mask_h = kin["mask_h"]
    mask_a = kin["mask_a"]
    B_int = kin["B_int_c"]
    VB_int = kin["VB_int"]
    F_c = kin["F_c"]
    bvis = kin["bvis"]
    idx_prev = kin["idx_prev"]
    idx_next = kin["idx_next"]
    gap_len = kin["gap_len"]
    tau = kin["tau"]
    half = kin["half"]
    T = len(half)

    d_foc_h = np.where(mask_h, np.hypot(H_c[:, :, 0] - F_c[:, None, 0], H_c[:, :, 1] - F_c[:, None, 1]), 1e6)
    d_foc_a = np.where(mask_a, np.hypot(A_c[:, :, 0] - F_c[:, None, 0], A_c[:, :, 1] - F_c[:, None, 1]), 1e6)
    d_int_h = np.where(mask_h, np.hypot(H_c[:, :, 0] - B_int[:, None, 0], H_c[:, :, 1] - B_int[:, None, 1]), 1e6)
    d_int_a = np.where(mask_a, np.hypot(A_c[:, :, 0] - B_int[:, None, 0], A_c[:, :, 1] - B_int[:, None, 1]), 1e6)

    c_foc_sharp = _soft_22_centroid(H_c, A_c, mask_h, mask_a, d_foc_h, d_foc_a, temp=1.5)
    c_foc_med = _soft_22_centroid(H_c, A_c, mask_h, mask_a, d_foc_h, d_foc_a, temp=4.0)
    c_int_sharp = _soft_22_centroid(H_c, A_c, mask_h, mask_a, d_int_h, d_int_a, temp=1.5)
    c_int_med = _soft_22_centroid(H_c, A_c, mask_h, mask_a, d_int_h, d_int_a, temp=4.0)
    c_all = _soft_22_centroid(H_c, A_c, mask_h, mask_a, np.zeros_like(d_foc_h), np.zeros_like(d_foc_a), temp=10.0)

    c_foc_s25 = np.column_stack([
        _gaussian_smooth_by_half(c_foc_sharp[:, 0], half, sigma=8.0),
        _gaussian_smooth_by_half(c_foc_sharp[:, 1], half, sigma=8.0),
    ])
    F_s25 = np.column_stack([
        _gaussian_smooth_by_half(F_c[:, 0], half, sigma=8.0),
        _gaussian_smooth_by_half(F_c[:, 1], half, sigma=8.0),
    ])

    B_prev = B_int[idx_prev]
    B_next = B_int[idx_next]
    V_prev = VB_int[idx_prev]
    V_next = VB_int[idx_next]
    c_prev_foc = c_foc_sharp[idx_prev]
    c_next_foc = c_foc_sharp[idx_next]

    tau_col = tau[:, None]
    c_foc_chord = (1.0 - tau_col) * c_prev_foc + tau_col * c_next_foc
    dt_sec = np.clip(gap_len / float(FPS), 0.0, 12.0)[:, None]
    v_hermite = tau_col * (1.0 - tau_col) * dt_sec * ((1.0 - tau_col) * V_prev - tau_col * V_next)

    # 11 Equivariant 2D basis vectors `v_m(t)`
    v1 = F_c - B_int
    v2 = F_s25 - B_int
    v3 = c_foc_sharp - B_int
    v4 = c_foc_med - B_int
    v5 = c_foc_s25 - B_int
    v6 = c_int_sharp - B_int
    v7 = c_int_med - B_int
    v8 = c_all - B_int
    v9 = c_foc_sharp - c_foc_chord
    v10 = 0.5 * (B_prev + B_next) - B_int
    v11 = v_hermite

    V_basis = np.stack([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11], axis=1).astype(np.float32)

    # 26 Strictly Invariant Scalar Features `S_inv`
    wg_1d = (
        (w_ghost[:, 0] if w_ghost.ndim == 2 else w_ghost).astype(np.float64)
        if w_ghost is not None
        else np.zeros(T, dtype=np.float64)
    )
    gap_s = gap_len / float(FPS)
    bridge_arch = 4.0 * tau * (1.0 - tau)
    tau_sym = np.abs(2.0 * tau - 1.0)
    chord_jump = np.hypot(B_next[:, 0] - B_prev[:, 0], B_next[:, 1] - B_prev[:, 1])
    chord_speed = np.hypot(VB_int[:, 0], VB_int[:, 1])

    sp_h = np.where(mask_h, np.hypot(VH[:, :, 0], VH[:, :, 1]), 0.0)
    sp_a = np.where(mask_a, np.hypot(VA[:, :, 0], VA[:, :, 1]), 0.0)
    n_vis_tot = np.maximum(np.sum(mask_h, axis=1) + np.sum(mask_a, axis=1), 1).astype(np.float64)
    mean_p_speed = (np.sum(sp_h, axis=1) + np.sum(sp_a, axis=1)) / n_vis_tot
    max_p_speed = np.maximum(np.max(sp_h, axis=1), np.max(sp_a, axis=1))
    still_h = np.sum(((sp_h < 1.0) & mask_h).astype(np.float64), axis=1)
    still_a = np.sum(((sp_a < 1.0) & mask_a).astype(np.float64), axis=1)
    frac_still = (still_h + still_a) / n_vis_tot

    d1_foc_h = np.min(d_foc_h, axis=1)
    d1_foc_a = np.min(d_foc_a, axis=1)
    min1_d_foc = np.minimum(d1_foc_h, d1_foc_a)
    min2_d_foc = np.maximum(d1_foc_h, d1_foc_a)

    d1_int_h = np.min(d_int_h, axis=1)
    d1_int_a = np.min(d_int_a, axis=1)
    min1_d_int = np.minimum(d1_int_h, d1_int_a)
    min2_d_int = np.maximum(d1_int_h, d1_int_a)

    v_norms = np.hypot(V_basis[:, :, 0], V_basis[:, :, 1])
    pitch_r_int = np.hypot(B_int[:, 0], B_int[:, 1])
    pitch_r_foc = np.hypot(F_c[:, 0], F_c[:, 1])

    S_list = [
        (~bvis).astype(np.float64),
        np.log1p(gap_s),
        np.clip(gap_s / 10.0, 0.0, 3.0),
        bridge_arch,
        tau_sym,
        np.log1p(chord_jump),
        np.clip(chord_speed / 15.0, 0.0, 3.0),
        wg_1d,
        mean_p_speed / 5.0,
        max_p_speed / 10.0,
        frac_still,
        np.log1p(np.clip(min1_d_foc, 0.0, 60.0)),
        np.log1p(np.clip(min2_d_foc, 0.0, 60.0)),
        np.log1p(np.clip(min1_d_int, 0.0, 60.0)),
        np.log1p(np.clip(min2_d_int, 0.0, 60.0)),
        pitch_r_int / 50.0,
        pitch_r_foc / 50.0,
    ]
    for m_idx in range(9):
        S_list.append(np.log1p(v_norms[:, m_idx]))

    S_inv = np.column_stack(S_list).astype(np.float32)
    S_inv = np.nan_to_num(S_inv, nan=0.0, posinf=0.0, neginf=0.0)
    return V_basis, S_inv, kin


class InpaintBasisMLP(nn.Module):
    """Equivariant 11-Basis Weight Network `w = MLP(S_inv)`."""

    def __init__(self, in_dim: int = 26, n_basis: int = 11, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_basis),
        )

    def forward(self, s_inv: torch.Tensor, v_basis: torch.Tensor) -> torch.Tensor:
        w = self.net(s_inv)
        return torch.sum(w.unsqueeze(-1) * v_basis, dim=1)


@dataclass
class InpainterConfig:
    hidden_dim: int = 64
    lr: float = 2.5e-3
    weight_decay: float = 1e-4
    epochs: int = 35
    batch_size: int = 2048
    charbonnier_eps: float = 1.5
    l2_weight: float = 0.08
    seed: int = 42


class SymBallPositionInpainter:
    """Part A: Strictly Equivariant 2D Ball Position Inpainter (`SymBallPositionInpainter`)."""

    def __init__(self, cfg: InpainterConfig | None = None):
        self.cfg = cfg or InpainterConfig()
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.model_: InpaintBasisMLP | None = None

    def fit(
        self,
        V_list: list[np.ndarray],
        S_list: list[np.ndarray],
        target_delta_list: list[np.ndarray],
        mask_list: list[np.ndarray],
        sample_weight_list: list[np.ndarray] | None = None,
    ) -> "SymBallPositionInpainter":
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        V_cat = np.concatenate([v[m] for v, m in zip(V_list, mask_list)], axis=0).astype(np.float32)
        S_cat = np.concatenate([s[m] for s, m in zip(S_list, mask_list)], axis=0).astype(np.float32)
        Y_cat = np.concatenate([y[m] for y, m in zip(target_delta_list, mask_list)], axis=0).astype(np.float32)
        if sample_weight_list is not None:
            W_cat = np.concatenate([w[m] for w, m in zip(sample_weight_list, mask_list)], axis=0).astype(np.float32)
        else:
            W_cat = np.ones(len(S_cat), dtype=np.float32)

        self.mean_ = S_cat.mean(axis=0, keepdims=True)
        self.std_ = S_cat.std(axis=0, keepdims=True) + 1e-5
        S_norm = (S_cat - self.mean_) / self.std_

        model = InpaintBasisMLP(in_dim=S_cat.shape[1], n_basis=V_cat.shape[1], hidden=self.cfg.hidden_dim)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)

        S_t = torch.from_numpy(S_norm)
        V_t = torch.from_numpy(V_cat)
        Y_t = torch.from_numpy(Y_cat)
        W_t = torch.from_numpy(W_cat)

        N = len(S_t)
        bs = self.cfg.batch_size
        eps2 = float(self.cfg.charbonnier_eps ** 2)

        model.train()
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(N)
            for i in range(0, N, bs):
                idx = perm[i : i + bs]
                pred = model(S_t[idx], V_t[idx])
                diff = pred - Y_t[idx]
                sq_err = torch.sum(diff * diff, dim=-1)
                charb = torch.sqrt(sq_err + eps2)
                loss = torch.mean(W_t[idx] * (charb + self.cfg.l2_weight * sq_err))
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()

        model.eval()
        self.model_ = model
        return self

    def predict_delta(
        self,
        m: Match,
        w_ghost: np.ndarray | None = None,
        apply_gate: bool = True,
        kin: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict `(delta_xy_hat, B_inpaint_centered)` for match `m`.

        Under `swap_match(m)`:
        - `delta_xy_hat(swap(m)) == -delta_xy_hat(m)` (`0.000e+00` error)
        - `B_inpaint_centered(swap(m)) == -B_inpaint_centered(m)` (`0.000e+00` error)
        """
        assert self.model_ is not None and self.mean_ is not None and self.std_ is not None
        V_basis, S_inv, kin_out = build_inpaint_basis_and_scalars(m, w_ghost=w_ghost, kin=kin)
        S_norm = (S_inv - self.mean_) / self.std_

        with torch.no_grad():
            raw_delta = self.model_(torch.from_numpy(S_norm), torch.from_numpy(V_basis)).numpy().astype(np.float64)

        if apply_gate:
            bvis = kin_out["bvis"]
            wg_1d = (
                (w_ghost[:, 0] if w_ghost.ndim == 2 else w_ghost).astype(np.float64)
                if w_ghost is not None
                else np.zeros(len(kin_out["half"]), dtype=np.float64)
            )
            g_gap = (~bvis).astype(np.float64)
            g_ghost = np.clip((wg_1d - 0.12) / 0.18, 0.0, 1.0)
            g_apply = np.maximum(g_gap, 0.75 * g_ghost)
            raw_delta = raw_delta * g_apply[:, None]

        mag = np.hypot(raw_delta[:, 0], raw_delta[:, 1])
        scale = np.where(mag > 45.0, 45.0 / np.maximum(mag, 1e-8), 1.0)
        delta_xy = np.round(raw_delta * scale[:, None], 10)
        B_inpaint_c = np.round(np.nan_to_num(kin_out["B_int_c"], nan=0.0) + delta_xy, 10)
        return delta_xy, B_inpaint_c


def build_inpainted_feature_df(
    m: Match,
    B_ball_c: np.ndarray,
    delta_xy: np.ndarray | None = None,
    w_ghost: np.ndarray | None = None,
    kin: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, str]], list[str]]:
    """Build a 78-column strictly H<->A equivariant feature table from centered ball coordinates `B_ball_c (T, 2)`.

    Returns:
      - `df`: DataFrame of shape `(T, 78)`
      - `ha_pairs`: list of `(col_H, col_A)` pairs that swap under `H <-> A`
      - `antisym_cols`: list of columns that negate (`col -> -col`) under `H <-> A` + 180-deg rotation
    Every remaining column is strictly invariant (`col -> +col`) under `H <-> A` + 180-deg rotation.
    """
    if kin is None:
        kin = extract_centered_kinematics(m)
    H_c = kin["H_c"]
    A_c = kin["A_c"]
    VH = kin["VH"]
    VA = kin["VA"]
    mask_h = kin["mask_h"]
    mask_a = kin["mask_a"]
    B_int_c = kin["B_int_c"]
    F_c = kin["F_c"]
    bvis = kin["bvis"].astype(np.float64)
    gap_s = kin["gap_len"] / float(FPS)
    tau = kin["tau"]
    idx_prev = kin["idx_prev"]
    idx_next = kin["idx_next"]
    half = kin["half"]
    L, W = kin["L"], kin["W"]
    T = len(half)

    B_c = np.nan_to_num(B_ball_c.astype(np.float64), nan=0.0)
    VB_c = _half_time_diff_2d(B_c, half)
    AB_c = _half_time_diff_2d(VB_c, half)

    if delta_xy is None:
        delta_xy = B_c - B_int_c
    delta_mag = np.hypot(delta_xy[:, 0], delta_xy[:, 1])

    vec_H = B_c[:, None, :] - H_c
    vec_A = B_c[:, None, :] - A_c
    d_H = np.where(mask_h, np.hypot(vec_H[:, :, 0], vec_H[:, :, 1]), 50.0)
    d_A = np.where(mask_a, np.hypot(vec_A[:, :, 0], vec_A[:, :, 1]), 50.0)

    d_H_int = np.where(mask_h, np.hypot(B_int_c[:, None, 0] - H_c[:, :, 0], B_int_c[:, None, 1] - H_c[:, :, 1]), 50.0)
    d_A_int = np.where(mask_a, np.hypot(B_int_c[:, None, 0] - A_c[:, :, 0], B_int_c[:, None, 1] - A_c[:, :, 1]), 50.0)

    ord_H = np.argsort(d_H, axis=1, kind="mergesort")
    ord_A = np.argsort(d_A, axis=1, kind="mergesort")
    rows = np.arange(T)[:, None]
    d_H_sort = d_H[rows, ord_H]
    d_A_sort = d_A[rows, ord_A]
    d_H_int_min = np.min(d_H_int, axis=1)
    d_A_int_min = np.min(d_A_int, axis=1)

    d1_H, d2_H, d3_H = d_H_sort[:, 0], d_H_sort[:, 1], d_H_sort[:, 2]
    d1_A, d2_A, d3_A = d_A_sort[:, 0], d_A_sort[:, 1], d_A_sort[:, 2]

    p1_H = ord_H[:, 0]
    p1_A = ord_A[:, 0]
    u_H1 = vec_H[np.arange(T), p1_H] / np.maximum(d1_H[:, None], 0.1)
    u_A1 = vec_A[np.arange(T), p1_A] / np.maximum(d1_A[:, None], 0.1)
    v1_H = VH[np.arange(T), p1_H]
    v1_A = VA[np.arange(T), p1_A]
    vclose_H = np.sum(v1_H * u_H1, axis=1)
    vclose_A = np.sum(v1_A * u_A1, axis=1)
    speed1_H = np.hypot(v1_H[:, 0], v1_H[:, 1])
    speed1_A = np.hypot(v1_A[:, 0], v1_A[:, 1])

    def _ctrl_pair(r_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        wH = np.sum(np.where(mask_h, np.exp(-d_H / r_scale), 0.0), axis=1)
        wA = np.sum(np.where(mask_a, np.exp(-d_A / r_scale), 0.0), axis=1)
        tot = wH + wA + 1e-6
        shH = wH / tot
        shA = wA / tot
        return shH, shA, shH - shA, np.log1p(wH + wA)

    shH_15, shA_15, cdiff_15, cdens_15 = _ctrl_pair(1.5)
    shH_30, shA_30, cdiff_30, cdens_30 = _ctrl_pair(3.0)
    shH_60, shA_60, cdiff_60, cdens_60 = _ctrl_pair(6.0)
    shH_100, shA_100, cdiff_100, cdens_100 = _ctrl_pair(10.0)

    d_foc_H = np.sort(np.where(mask_h, np.hypot(H_c[:, :, 0] - F_c[:, None, 0], H_c[:, :, 1] - F_c[:, None, 1]), 50.0), axis=1)
    d_foc_A = np.sort(np.where(mask_a, np.hypot(A_c[:, :, 0] - F_c[:, None, 0], A_c[:, :, 1] - F_c[:, None, 1]), 50.0), axis=1)
    foc_d1_H, foc_d1_A = d_foc_H[:, 0], d_foc_A[:, 0]
    foc_shH = np.sum(np.exp(-d_foc_H / 3.5), axis=1)
    foc_shA = np.sum(np.exp(-d_foc_A / 3.5), axis=1)
    foc_tot = foc_shH + foc_shA + 1e-6
    foc_ctrl_H = foc_shH / foc_tot
    foc_ctrl_A = foc_shA / foc_tot

    sp_H = np.where(mask_h, np.hypot(VH[:, :, 0], VH[:, :, 1]), 0.0)
    sp_A = np.where(mask_a, np.hypot(VA[:, :, 0], VA[:, :, 1]), 0.0)
    n_vis_H = np.maximum(np.sum(mask_h, axis=1), 1).astype(np.float64)
    n_vis_A = np.maximum(np.sum(mask_a, axis=1), 1).astype(np.float64)
    mean_sp_H = np.sum(sp_H, axis=1) / n_vis_H
    mean_sp_A = np.sum(sp_A, axis=1) / n_vis_A
    still_H = np.sum(((sp_H < 1.0) & mask_h).astype(np.float64), axis=1) / n_vis_H
    still_A = np.sum(((sp_A < 1.0) & mask_a).astype(np.float64), axis=1) / n_vis_A
    all_sp_mean = 0.5 * (mean_sp_H + mean_sp_A)
    all_still_mean = 0.5 * (still_H + still_A)

    diff_d1 = d1_A - d1_H
    diff_d2 = d2_A - d2_H
    diff_d3 = d3_A - d3_H
    log_ratio_d1 = np.log((d1_A + 0.5) / (d1_H + 0.5))
    log_ratio_d2 = np.log((d2_A + 0.5) / (d2_H + 0.5))
    foc_diff_d1 = foc_d1_A - foc_d1_H
    foc_ctrl_diff = foc_ctrl_H - foc_ctrl_A
    vclose_diff = vclose_H - vclose_A

    gain_H = d_H_int_min - d1_H
    gain_A = d_A_int_min - d1_A
    gain_diff = gain_H - gain_A

    cdiff30_prev = cdiff_30[idx_prev]
    cdiff30_next = cdiff_30[idx_next]
    cdiff30_bridge = (1.0 - tau) * cdiff30_prev + tau * cdiff30_next
    cdiff30_same_sign = (cdiff30_prev * cdiff30_next > 0.0).astype(np.float64) * np.minimum(
        np.abs(cdiff30_prev), np.abs(cdiff30_next)
    )

    min_d1 = np.minimum(d1_H, d1_A)
    max_d1 = np.maximum(d1_H, d1_A)
    sum_d1 = d1_H + d1_A
    ball_speed = np.hypot(VB_c[:, 0], VB_c[:, 1])
    ball_accel = np.hypot(AB_c[:, 0], AB_c[:, 1])
    ball_pitch_r = np.hypot(B_c[:, 0], B_c[:, 1])
    sideline_dist = (W / 2.0) - np.abs(B_c[:, 1])
    goalline_dist = (L / 2.0) - np.abs(B_c[:, 0])
    bfd_inp = np.hypot(B_c[:, 0] - F_c[:, 0], B_c[:, 1] - F_c[:, 1])
    wg_1d = (
        (w_ghost[:, 0] if w_ghost.ndim == 2 else w_ghost).astype(np.float64)
        if w_ghost is not None
        else np.zeros(T, dtype=np.float64)
    )

    data: dict[str, np.ndarray] = {
        "inp_d1_H": d1_H, "inp_d1_A": d1_A,
        "inp_d2_H": d2_H, "inp_d2_A": d2_A,
        "inp_d3_H": d3_H, "inp_d3_A": d3_A,
        "inp_vclose_H": vclose_H, "inp_vclose_A": vclose_A,
        "inp_speed1_H": speed1_H, "inp_speed1_A": speed1_A,
        "inp_sh15_H": shH_15, "inp_sh15_A": shA_15,
        "inp_sh30_H": shH_30, "inp_sh30_A": shA_30,
        "inp_sh60_H": shH_60, "inp_sh60_A": shA_60,
        "inp_sh100_H": shH_100, "inp_sh100_A": shA_100,
        "inp_foc_d1_H": foc_d1_H, "inp_foc_d1_A": foc_d1_A,
        "inp_foc_ctrl_H": foc_ctrl_H, "inp_foc_ctrl_A": foc_ctrl_A,
        "inp_mean_sp_H": mean_sp_H, "inp_mean_sp_A": mean_sp_A,
        "inp_still_H": still_H, "inp_still_A": still_A,
        "inp_gain_H": gain_H, "inp_gain_A": gain_A,
        "inp_diff_d1": diff_d1,
        "inp_diff_d2": diff_d2,
        "inp_diff_d3": diff_d3,
        "inp_log_ratio_d1": log_ratio_d1,
        "inp_log_ratio_d2": log_ratio_d2,
        "inp_cdiff_15": cdiff_15,
        "inp_cdiff_30": cdiff_30,
        "inp_cdiff_60": cdiff_60,
        "inp_cdiff_100": cdiff_100,
        "inp_foc_diff_d1": foc_diff_d1,
        "inp_foc_ctrl_diff": foc_ctrl_diff,
        "inp_vclose_diff": vclose_diff,
        "inp_gain_diff": gain_diff,
        "inp_cdiff30_prev": cdiff30_prev,
        "inp_cdiff30_next": cdiff30_next,
        "inp_cdiff30_bridge": cdiff30_bridge,
        "inp_ball_x": B_c[:, 0],
        "inp_ball_y": B_c[:, 1],
        "inp_ball_vx": VB_c[:, 0],
        "inp_ball_vy": VB_c[:, 1],
        "inp_min_d1": min_d1,
        "inp_max_d1": max_d1,
        "inp_sum_d1": sum_d1,
        "inp_cdens_15": cdens_15,
        "inp_cdens_30": cdens_30,
        "inp_cdens_60": cdens_60,
        "inp_cdens_100": cdens_100,
        "inp_all_sp_mean": all_sp_mean,
        "inp_all_still_mean": all_still_mean,
        "inp_ball_speed": ball_speed,
        "inp_ball_accel": ball_accel,
        "inp_ball_pitch_r": ball_pitch_r,
        "inp_sideline_dist": sideline_dist,
        "inp_goalline_dist": goalline_dist,
        "inp_bfd": bfd_inp,
        "inp_delta_mag": delta_mag,
        "inp_bvis": bvis,
        "inp_gap_s": gap_s,
        "inp_bridge_arch": 4.0 * tau * (1.0 - tau),
        "inp_cdiff30_same_sign": cdiff30_same_sign,
        "inp_w_ghost": wg_1d,
    }

    ha_pairs = [
        ("inp_d1_H", "inp_d1_A"),
        ("inp_d2_H", "inp_d2_A"),
        ("inp_d3_H", "inp_d3_A"),
        ("inp_vclose_H", "inp_vclose_A"),
        ("inp_speed1_H", "inp_speed1_A"),
        ("inp_sh15_H", "inp_sh15_A"),
        ("inp_sh30_H", "inp_sh30_A"),
        ("inp_sh60_H", "inp_sh60_A"),
        ("inp_sh100_H", "inp_sh100_A"),
        ("inp_foc_d1_H", "inp_foc_d1_A"),
        ("inp_foc_ctrl_H", "inp_foc_ctrl_A"),
        ("inp_mean_sp_H", "inp_mean_sp_A"),
        ("inp_still_H", "inp_still_A"),
        ("inp_gain_H", "inp_gain_A"),
    ]
    antisym_cols = [
        "inp_diff_d1",
        "inp_diff_d2",
        "inp_diff_d3",
        "inp_log_ratio_d1",
        "inp_log_ratio_d2",
        "inp_cdiff_15",
        "inp_cdiff_30",
        "inp_cdiff_60",
        "inp_cdiff_100",
        "inp_foc_diff_d1",
        "inp_foc_ctrl_diff",
        "inp_vclose_diff",
        "inp_gain_diff",
        "inp_cdiff30_prev",
        "inp_cdiff30_next",
        "inp_cdiff30_bridge",
        "inp_ball_x",
        "inp_ball_y",
        "inp_ball_vx",
        "inp_ball_vy",
    ]

    for w in (15, 45, 125):
        for bname, src_arr in [
            ("diff_d1", diff_d1),
            ("log_ratio_d1", log_ratio_d1),
            ("cdiff_30", cdiff_30),
            ("foc_ctrl_diff", foc_ctrl_diff),
        ]:
            cname = f"inp_{bname}_r{w}"
            data[cname] = np.nan_to_num(roll_nanmean_by_half(src_arr, half, w), nan=0.0)
            antisym_cols.append(cname)
        for bname, src_arr in [
            ("min_d1", min_d1),
            ("all_sp_mean", all_sp_mean),
            ("all_still_mean", all_still_mean),
            ("ball_speed", ball_speed),
            ("bfd", bfd_inp),
        ]:
            cname = f"inp_{bname}_r{w}"
            data[cname] = np.nan_to_num(roll_nanmean_by_half(src_arr, half, w), nan=0.0)

    df = pd.DataFrame({k: np.round(v, 10).astype(np.float32) for k, v in data.items()})
    return df, ha_pairs, antisym_cols


def mirror_inpainted_df(
    df: pd.DataFrame,
    ha_pairs: list[tuple[str, str]],
    antisym_cols: list[str],
) -> pd.DataFrame:
    """Apply exact H <-> A + 180-deg rotation involution on `df` (`0.000e+00` error)."""
    out = df.copy()
    for cH, cA in ha_pairs:
        out[cH] = df[cA].to_numpy().copy()
        out[cA] = df[cH].to_numpy().copy()
    for c in antisym_cols:
        out[c] = -df[c].to_numpy().copy()
    return out


class SymGapDistilledExpert:
    """Part B: Symmetric Gap-Distilled Possession & Transition Expert (`SymGapDistilledExpert`).

    Architecture:
    - Head 1 (`head_dead`): Predicts `P(DEAD)` using LightGBM trained on original + mirrored rows.
    - Head 2 (`head_ha`): Predicts `P(HOME | alive)` using LightGBM trained on original + mirrored alive rows
      plus soft distillation targets from the Clean-Ball Teacher on gap/ghost frames.
    - Head 3 (`head_change`): Predicts `p_change` (boundary transition probability within +/-2 frames).
    - Exact `0.000e+00` H <-> A Equivariance via TTA over `(df, mirror_inpainted_df(df))`.
    """

    def __init__(self, n_estimators: int = 320, learning_rate: float = 0.045, num_leaves: int = 31, seed: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.seed = seed
        self.ha_pairs_: list[tuple[str, str]] = []
        self.antisym_cols_: list[str] = []
        self.feature_cols_: list[str] = []
        self.model_dead_: lgb.Booster | None = None
        self.model_ha_: lgb.Booster | None = None
        self.model_change_: lgb.Booster | None = None

    def fit(
        self,
        df_list: list[pd.DataFrame],
        y_list: list[np.ndarray],
        half_list: list[np.ndarray],
        ha_pairs: list[tuple[str, str]],
        antisym_cols: list[str],
        teacher_proba_list: list[np.ndarray] | None = None,
        gap_mask_list: list[np.ndarray] | None = None,
        distill_alpha: float = 0.40,
        gap_oversample_weight: float = 3.5,
    ) -> "SymGapDistilledExpert":
        self.ha_pairs_ = list(ha_pairs)
        self.antisym_cols_ = list(antisym_cols)
        self.feature_cols_ = list(df_list[0].columns)

        X_orig_parts, X_mir_parts = [], []
        y_orig_parts, y_mir_parts = [], []
        w_parts = []
        t_orig_parts, t_mir_parts = [], []
        c_parts = []

        for idx_m, (df_m, y_m, h_m) in enumerate(zip(df_list, y_list, half_list)):
            valid = (h_m > 0) & (y_m >= 0)
            df_v = df_m.loc[valid].reset_index(drop=True)
            df_v_mir = mirror_inpainted_df(df_v, self.ha_pairs_, self.antisym_cols_)
            yv = y_m[valid].astype(np.int64)
            yv_mir = np.where(yv == HOME, AWAY, np.where(yv == AWAY, HOME, DEAD))

            ch = np.zeros(len(y_m), dtype=np.float32)
            diff_pos = np.flatnonzero((y_m[1:] != y_m[:-1]) & (h_m[1:] == h_m[:-1]) & (h_m[1:] > 0)) + 1
            for dp in diff_pos:
                lo = max(0, dp - 2)
                hi = min(len(y_m), dp + 3)
                ch[lo:hi] = 1.0
            ch_v = ch[valid]

            if gap_mask_list is not None:
                gm = gap_mask_list[idx_m][valid]
                wv = np.where(gm, gap_oversample_weight, 1.0).astype(np.float32)
            else:
                wv = np.ones(len(yv), dtype=np.float32)

            if teacher_proba_list is not None:
                tp = teacher_proba_list[idx_m][valid].astype(np.float32)
                tp_mir = tp[:, [AWAY, HOME, DEAD]]
            else:
                tp = np.zeros((len(yv), 3), dtype=np.float32)
                tp[np.arange(len(yv)), yv] = 1.0
                tp_mir = tp[:, [AWAY, HOME, DEAD]]

            X_orig_parts.append(df_v.to_numpy(dtype=np.float32)[::2])
            X_mir_parts.append(df_v_mir.to_numpy(dtype=np.float32)[::2])
            y_orig_parts.append(yv[::2])
            y_mir_parts.append(yv_mir[::2])
            w_parts.append(wv[::2])
            t_orig_parts.append(tp[::2])
            t_mir_parts.append(tp_mir[::2])
            c_parts.append(ch_v[::2])

        X_all = np.concatenate(X_orig_parts + X_mir_parts, axis=0)
        y_all = np.concatenate(y_orig_parts + y_mir_parts, axis=0)
        w_all = np.concatenate(w_parts + w_parts, axis=0)
        t_all = np.concatenate(t_orig_parts + t_mir_parts, axis=0)
        c_all = np.concatenate(c_parts + c_parts, axis=0)

        params_bin = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "n_jobs": 2,
            "verbose": -1,
            "seed": self.seed,
        }
        params_xent = dict(params_bin)
        params_xent["objective"] = "cross_entropy"

        y_dead_hard = (y_all == DEAD).astype(np.float32)
        y_dead_soft = (1.0 - distill_alpha) * y_dead_hard + distill_alpha * np.clip(t_all[:, DEAD], 0.0, 1.0)
        dtrain_dead = lgb.Dataset(X_all, label=y_dead_soft, weight=w_all)
        self.model_dead_ = lgb.train(params_xent, dtrain_dead, num_boost_round=self.n_estimators)

        alive_mask = (y_all != DEAD) | (t_all[:, DEAD] < 0.65)
        X_ha = X_all[alive_mask]
        w_ha = w_all[alive_mask]
        t_ha = t_all[alive_mask]
        y_ha_raw = y_all[alive_mask]
        q_teacher_h = np.clip(t_ha[:, HOME] / np.maximum(t_ha[:, HOME] + t_ha[:, AWAY], 1e-6), 0.0, 1.0)
        y_ha_hard = np.where(y_ha_raw == HOME, 1.0, np.where(y_ha_raw == AWAY, 0.0, q_teacher_h)).astype(np.float32)
        y_ha_soft = (1.0 - distill_alpha) * y_ha_hard + distill_alpha * q_teacher_h
        dtrain_ha = lgb.Dataset(X_ha, label=y_ha_soft, weight=w_ha)
        self.model_ha_ = lgb.train(params_xent, dtrain_ha, num_boost_round=self.n_estimators)

        dtrain_ch = lgb.Dataset(X_all, label=c_all, weight=w_all)
        self.model_change_ = lgb.train(params_bin, dtrain_ch, num_boost_round=max(180, self.n_estimators // 2))
        return self

    def _predict_raw(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self.model_dead_ is not None and self.model_ha_ is not None and self.model_change_ is not None
        X_mat = df[self.feature_cols_].to_numpy(dtype=np.float32)
        p_d = np.clip(self.model_dead_.predict(X_mat), 1e-6, 1.0 - 1e-6)
        q_h = np.clip(self.model_ha_.predict(X_mat), 1e-6, 1.0 - 1e-6)
        p_h = (1.0 - p_d) * q_h
        p_a = (1.0 - p_d) * (1.0 - q_h)
        proba = np.column_stack([p_h, p_a, p_d]).astype(np.float64)
        p_ch = np.clip(self.model_change_.predict(X_mat), 1e-6, 1.0 - 1e-6).astype(np.float64)
        return proba, p_ch

    def predict_tta(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict strictly H <-> A equivariant `(proba, p_change)` with `0.000e+00` symmetry error."""
        p_orig, c_orig = self._predict_raw(df)
        df_mir = mirror_inpainted_df(df, self.ha_pairs_, self.antisym_cols_)
        p_mir, c_mir = self._predict_raw(df_mir)
        p_mir_swapped = p_mir[:, [AWAY, HOME, DEAD]]
        p_sym = np.round(0.5 * (p_orig + p_mir_swapped), 10).astype(np.float32)
        c_sym = np.round(0.5 * (c_orig + c_mir), 10).astype(np.float32)
        return p_sym, c_sym
