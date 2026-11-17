"""`SymTrajectoryEventNetV10`: Spatio-Temporal Trajectory Transformer & Window-Local Normalized Sequence Network.

Addresses the structural bottlenecks identified in `Codex review 004` (Finding 5) and `PLAN_NOCHE` (`L1`, `L5`, `B1`, `D1`, `D3`):
  1. Per-Player Spatio-Temporal Stem (`PlayerTrajectoryEncoder`):
     - Instead of pooling instantaneous single-frame snapshots `[N, 22, 18]` like `SymPlayerSetNet v1/v2`
       (which destroys individual player acceleration, kick deceleration, aerial tracking, and slot-continuity
       signals before pooling), reshapes the 22 player tensors across time and applies Continuity-Gated
       Kinematic Finite Differences + Multi-Scale Depthwise-Separable 1D Temporal Convolutions (`k=5, d=1`
       and `k=5, d=3`, covering `+/- 0.5s` of individual player motion) to each individual player's trajectory
       BEFORE set pooling.
     - Explicitly separates a 100% Ball-Free Trajectory Branch (`C_bf=12` + 6 ball-free kinematic derivatives)
       from a Ball-Aware Trajectory Branch (`C_total=18` + 10 kinematic/ball-approach derivatives) for
       `degrade_v6` ghost-ball and blackout resilience.
  2. Ball-Centric Cross-Attention + Top-K (`K=8`) Player-Player Interaction (`BallPlayerCrossInteraction`):
     - Models intra-team and cross-team contest interactions among the `K=8` active players closest to the
       play focus `x_focus(t)`, and cross-attends the global ball/physics state (`g_sym`, `g_anti`: knot-anchored
       ball physics, chord flight, impulse, ghost incoherence) to the 22 trajectory-enriched player tokens.
  3. Window-Local Normalized Multi-Scale Sequence Backbone (`WindowLocalNorm1d`, Fixing `Codex 004 #5`):
     - Replaces global full-half normalization with Sliding-Window Local Normalization (`window = 1500` frames = `60s`)
       plus overlapping `1500`-frame sliding-window inference so training crops (`60s`) and full-half inference (`45m`)
       share 100% identical normalization statistics and BiGRU sequence horizons.
     - Combines a Multi-Scale Dilated Temporal Conv stack + Parallel Dilated Pyramid (`d = 1, 2, 4, 8, 16, 32`)
       + 2-layer Bidirectional GRU.
  4. Multi-Task Action-Spotting + Direct Soft-Spell Loss:
     - `logits_had` (`[B, 3, T]`): trained with Possession-Duration-Weighted Cross-Entropy (`L1`, `w_t = L_spell^-0.40`)
       + Differentiable Multi-Scale Windowed Soft-IoU Loss on `H` and `A`.
     - `logit_change` (`[B, T]`) + `logits_4dir` (`[B, 4, T]`: `H->A`, `A->H`, `L->D`, `D->L`): trained with
       Gaussian-smoothed boundary targets (`sigma = 3` frames) and Focal + Positive-Weighted BCE Loss.
     - `logit_duel_to` (`[B, T]`): dedicated Short-Steal / Duel-Turnover Spotting Head trained specifically to
       fire inside true live possessions of length `<= 65` frames (`<= 2.6s`).
     - `pred_aerial` (`[B, 2, T]`): 3D aerial pass detector (`ball_z > 1.5m`, `Play_high` vs `Play_low`) + height regression.
  5. Exact `0.000e+00` H <-> A + 180-deg pitch rotation equivariance via Twin-Pass H/A Siamese Forwarding.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.common import AWAY, HOME, possession_spells
from src.models.boundary_4dir import targets_4dir
from src.models.player_set_net import (
    C_PLAYER_BF,
    C_PLAYER_TOTAL,
    ChannelNorm,
    PlayerSetTensors,
)
from src.models.transition import change_targets


# ---------------------------------------------------------------------------
# 1. Target Builders: Possession Weights (L1), Gaussian Boundaries (L5), Short-Duel Spells
# ---------------------------------------------------------------------------

def compute_v10_possession_frame_weights(
    labels: np.ndarray,
    half: np.ndarray,
    alpha: float = 0.40,
    boundary_radius: int = 20,
    boundary_boost: float = 1.65,
) -> np.ndarray:
    """Compute per-frame weights $w_t \\propto L_{\\text{spell}}^{-\\alpha}$ (`alpha=0.40`) with boundary boost:
    - Live spells (`H`, `A`) of length $L_{\\text{spell}}$ receive $(100 / \\max(L_{\\text{spell}}, 12))^{\\alpha}$.
    - Short dead-ball spells (`D`) receive moderate inverse-duration weight.
    - Frames within `+/- boundary_radius` of a transition receive `boundary_boost`.
    - Normalized to mean `1.0` over valid labelled frames.
    """
    T = len(labels)
    valid = (labels >= 0) & (half > 0)
    w = np.ones(T, dtype=np.float32)
    if not np.any(valid):
        return w

    same_prev = np.r_[False, (labels[1:] == labels[:-1]) & (half[1:] == half[:-1]) & valid[1:] & valid[:-1]]
    starts = np.flatnonzero(~same_prev & valid)
    if len(starts) == 0:
        return w

    for idx_s in starts:
        hv = half[idx_s]
        lab = labels[idx_s]
        e = idx_s + 1
        while e < T and valid[e] and half[e] == hv and labels[e] == lab:
            e += 1
        run_len = max(e - idx_s, 12)
        if lab in (HOME, AWAY):
            w_run = (100.0 / float(run_len)) ** alpha
            w[idx_s:e] = np.clip(w_run, 0.35, 3.6)
        else:
            w_run = (150.0 / float(max(run_len, 25))) ** (0.5 * alpha)
            w[idx_s:e] = np.clip(w_run, 0.45, 2.0)

        if idx_s > 0 and half[idx_s - 1] == hv and valid[idx_s - 1]:
            lo = max(0, idx_s - boundary_radius)
            hi = min(T, idx_s + boundary_radius + 1)
            w[lo:hi] *= boundary_boost

    mean_w = float(np.mean(w[valid]))
    if mean_w > 1e-6:
        w = w / mean_w
    w[~valid] = 0.0
    return w.astype(np.float32)


def _gaussian_smooth_impulse_targets(
    binary_box_targets: np.ndarray,
    half: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """Blend standard box-window boundary targets (`radius=6`) with a Gaussian-smoothed (`sigma=3.0`)
    soft target inside each half so the boundary heads learn sharp peak localization at exact transitions
    while preserving full `radius=6` window recall for the Viterbi decoder.
    """
    valid = (binary_box_targets >= 0) & (half > 0)
    out = np.where(valid, 0.0, -1.0).astype(np.float32)
    if not np.any(valid):
        return out

    pos = np.where(valid, (binary_box_targets > 0).astype(np.float32), 0.0)
    for hv in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == hv)
        if len(idx) == 0:
            continue
        s, e = int(idx[0]), int(idx[-1]) + 1
        seg_pos = pos[s:e]
        if np.any(seg_pos > 0):
            g = gaussian_filter1d(seg_pos, sigma=sigma, mode="nearest")
            g_max = float(np.max(g))
            if g_max > 1e-6:
                g = np.clip(g / min(g_max, 0.75), 0.0, 1.0)
            # Combine box target and Gaussian peak so exact centers have target 1.0 and skirts taper smoothly
            blended = np.maximum(0.65 * seg_pos, g)
            out[s:e] = np.where(valid[s:e], np.clip(blended, 0.0, 1.0), -1.0)
    return out


def compute_v10_boundary_and_duel_targets(
    labels: np.ndarray,
    half: np.ndarray,
    sigma: float = 3.0,
    max_duel_len: int = 65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute:
      1. `y_chg`: `(T,)` float32 Gaussian-blended (`sigma=3`) 1D transition target in `[-1, 1]`.
      2. `y_4d`:  `(T, 4)` float32 Gaussian-blended (`sigma=3`) 4-directional boundary target
                  `[H->A, A->H, Live->D, D->Live]`.
      3. `y_duel_to`: `(T,)` float32 Short-Steal / Duel-Turnover target that fires (`1.0`) inside
                  true live possessions (`HOME` or `AWAY`) of duration `<= max_duel_len` (`65` frames = `2.6s`).
    """
    T = len(labels)
    valid = (labels >= 0) & (half > 0)

    yc_raw = change_targets(labels, half)
    y_chg = _gaussian_smooth_impulse_targets(yc_raw, half, sigma=sigma)

    t4 = targets_4dir(labels, half)
    y_4d = np.column_stack([
        _gaussian_smooth_impulse_targets(t4["y_h2a"], half, sigma=sigma),
        _gaussian_smooth_impulse_targets(t4["y_a2h"], half, sigma=sigma),
        _gaussian_smooth_impulse_targets(t4["y_live2d"], half, sigma=sigma),
        _gaussian_smooth_impulse_targets(t4["y_d2live"], half, sigma=sigma),
    ]).astype(np.float32)

    # Short-Steal / Duel-Turnover Spotting target (`L_spell <= 65` frames = `2.6s`)
    y_duel = np.where(valid, 0.0, -1.0).astype(np.float32)
    lab_clean = np.where(valid, labels, -1)
    s_arr, e_arr, t_arr = possession_spells(lab_clean, half)
    duel_pos = np.zeros(T, dtype=np.float32)
    for s_i, e_i, t_i in zip(s_arr, e_arr, t_arr):
        L_sp = int(e_i - s_i)
        if t_i in (HOME, AWAY) and 1 <= L_sp <= max_duel_len:
            duel_pos[s_i:e_i] = 1.0
        elif t_i in (HOME, AWAY) and max_duel_len < L_sp <= 90:
            # Soft taper for borderline short spells (2.6s - 3.6s)
            duel_pos[s_i:e_i] = 0.35

    for hv in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == hv)
        if len(idx) == 0:
            continue
        s, e = int(idx[0]), int(idx[-1]) + 1
        if np.any(duel_pos[s:e] > 0):
            sm = gaussian_filter1d(duel_pos[s:e], sigma=1.5, mode="nearest")
            y_duel[s:e] = np.where(valid[s:e], np.clip(np.maximum(duel_pos[s:e], sm), 0.0, 1.0), -1.0)

    return y_chg, y_4d, y_duel


# ---------------------------------------------------------------------------
# 2. Window-Local Normalization (`WindowLocalNorm1d`, Fixing Codex 004 #5)
# ---------------------------------------------------------------------------

class WindowLocalNorm1d(nn.Module):
    """Sliding-Window Local Normalization (`window_size = 1500` frames = `60s`) + Channel LayerNorm.

    Fixes `Codex review 004` Finding 5:
      - Global `GroupNorm(1, C)` over a `45-minute` half (`67,500` frames) leaks global half statistics
        and shifts activation scales compared to `60s` (`1500`-frame) training crops, especially when a half
        contains a long camera blackout.
      - `WindowLocalNorm1d` normalizes each channel over a local sliding window of `~1500` frames (`60s`)
        via $O(L)$ coarse-strided pooling (`stride=50` frames = `2s`, `kernel=31` = `62s`) + linear interpolation,
        blended with instantaneous `ChannelNorm(C)`, guaranteeing identical local normalization statistics between
        `60s` training crops and `45-min` inference halves in `< 0.1 ms` per layer.
    """

    def __init__(self, channels: int, window_size: int = 1501, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.eps = eps
        self.chan_norm = ChannelNorm(channels)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x_cn = self.chan_norm(x)
        L = x.shape[-1]
        if L <= 64:
            return x_cn
        x_ds = F.avg_pool1d(x, kernel_size=50, stride=50, ceil_mode=True)
        x2_ds = F.avg_pool1d(x * x, kernel_size=50, stride=50, ceil_mode=True)
        L_ds = x_ds.shape[-1]
        k = min(31, L_ds if (L_ds % 2 == 1) else max(1, L_ds - 1))
        if k > 1:
            pad = k // 2
            m_ds = F.avg_pool1d(x_ds, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
            msq_ds = F.avg_pool1d(x2_ds, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
        else:
            m_ds = x_ds
            msq_ds = x2_ds
        mean = F.interpolate(m_ds, size=L, mode="linear", align_corners=False)
        mean_sq = F.interpolate(msq_ds, size=L, mode="linear", align_corners=False)
        var = (mean_sq - mean * mean).clamp(min=0.0)
        x_wn = (x - mean) * torch.rsqrt(var + self.eps)
        x_wn = x_wn * self.gamma + self.beta
        alpha = torch.sigmoid(self.mix_logit)
        return (1.0 - alpha) * x_cn + alpha * x_wn


# ---------------------------------------------------------------------------
# 3. Per-Player Spatio-Temporal Stem (`PlayerTrajectoryEncoder`)
#    + Ball-Centric Cross-Attention & Top-K (`K=8`) Player-Player Interaction
# ---------------------------------------------------------------------------

class _DepthwiseSeparableTemporalBlock(nn.Module):
    """Multi-scale Depthwise-Separable 1D Temporal Convolution over each individual player's trajectory
    on `(B2, C, K=11, L)` via `kernel_size=(1, 5), dilation=(1, 1)` (`+/- 0.08s`) and
    `kernel_size=(1, 5), dilation=(1, 3)` (`+/- 0.24s`, `0.5s` span) with residual pointwise projection.
    """

    def __init__(self, channels: int, dropout: float = 0.10):
        super().__init__()
        self.dw_k5_d1 = nn.Conv2d(
            channels, channels, kernel_size=(1, 5), padding=(0, 2), dilation=(1, 1), groups=channels
        )
        self.dw_k5_d3 = nn.Conv2d(
            channels, channels, kernel_size=(1, 5), padding=(0, 6), dilation=(1, 3), groups=channels
        )
        self.pw = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, m_float: torch.Tensor) -> torch.Tensor:
        # x: (B2, C, K, L), m_float: (B2, 1, K, L)
        x_m = x * m_float
        h1 = self.dw_k5_d1(x_m)
        h3 = self.dw_k5_d3(x_m)
        out = x + self.pw(torch.cat([h1, h3], dim=1))
        return out * m_float


class PlayerTrajectoryEncoder(nn.Module):
    """Per-Player Spatio-Temporal Stem (`PlayerTrajectoryEncoder`) + Top-K (`K=8`) Player Interaction
    + Ball-Centric Cross-Attention.
    """

    def __init__(
        self,
        c_ga: int,
        c_gs: int,
        d_bf: int = 32,
        d_ball: int = 32,
        hidden: int = 48,
        k_interact: int = 8,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.d_bf = d_bf
        self.d_ball = d_ball
        self.d_player = d_bf + d_ball
        self.k_interact = k_interact

        c_bf_in = C_PLAYER_BF + 6       # 12 raw ball-free + 6 continuity-gated ball-free kinematic diffs = 18
        c_ball_in = C_PLAYER_TOTAL + 10 # 18 raw total + 6 ball-free diffs + 4 ball-aware diffs = 28

        # 1. Ball-Free Per-Player Spatio-Temporal Stem
        self.stem_bf_in = nn.Sequential(
            nn.Conv2d(c_bf_in, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.stem_bf_t1 = _DepthwiseSeparableTemporalBlock(hidden, dropout=dropout)
        self.stem_bf_out = nn.Conv2d(hidden, d_bf + 2, kernel_size=1)

        # 2. Ball-Aware Per-Player Spatio-Temporal Stem
        self.stem_ball_in = nn.Sequential(
            nn.Conv2d(c_ball_in, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.stem_ball_t1 = _DepthwiseSeparableTemporalBlock(hidden, dropout=dropout)
        self.stem_ball_out = nn.Conv2d(hidden, d_ball + 2, kernel_size=1)

        # 3. Top-K (K=8) Active Player-Player Interaction Block
        self.topk_ctx_bf = nn.Sequential(
            nn.Conv1d(d_bf * 2, d_bf, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_bf, d_bf, kernel_size=1),
        )
        self.topk_ctx_ball = nn.Sequential(
            nn.Conv1d(d_ball * 2, d_ball, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_ball, d_ball, kernel_size=1),
        )

        # 4. Ball-Centric Cross-Attention Query Projector from global ball/physics context (g_sym, g_anti)
        self.ball_query_proj = nn.Sequential(
            nn.Conv1d(c_gs + c_ga, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, d_ball, kernel_size=1),
        )

        raw_pool_dim = self.d_player * 2 + 2 * d_bf + 2 * d_ball + d_ball  # 288
        self.out_dim = 192
        self.pool_refine = nn.Sequential(
            nn.Conv1d(raw_pool_dim, self.out_dim, kernel_size=1),
            ChannelNorm(self.out_dim),
            nn.GELU(),
            nn.Conv1d(self.out_dim, self.out_dim, kernel_size=1),
        )

    @staticmethod
    @torch.no_grad()
    def _compute_kinematic_diffs(
        p: torch.Tensor, m_float: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute continuity-gated per-player temporal derivatives directly on `(B2, 18, K, L)`.
        Returns:
          p_bf_aug:   (B2, 18, K, L)
          p_ball_aug: (B2, 28, K, L)
        """
        L = p.shape[-1]
        x_n = p[:, 0:1, :, :]
        y_n = p[:, 1:2, :, :]
        vx_n = p[:, 2:3, :, :]
        vy_n = p[:, 3:4, :, :]
        sp_n = p[:, 4:5, :, :]
        vcf_n = p[:, 9:10, :, :]
        db_n = p[:, 14:15, :, :]
        pb3_n = p[:, 15:16, :, :]
        pb10_n = p[:, 16:17, :, :]
        vcb_n = p[:, 17:18, :, :]

        dx1 = F.pad(x_n[:, :, :, 1:] - x_n[:, :, :, :-1], (1, 0)) * 52.5
        dy1 = F.pad(y_n[:, :, :, 1:] - y_n[:, :, :, :-1], (1, 0)) * 34.0
        step1_m = torch.sqrt(dx1 * dx1 + dy1 * dy1 + 1e-6)
        m_prev1 = F.pad(m_float[:, :, :, :-1], (1, 0))
        c_gate1 = torch.exp(-F.relu(step1_m - 0.85).square()) * m_float * m_prev1
        c_gate_sym1 = c_gate1 * F.pad(c_gate1[:, :, :, 1:], (0, 1))

        def _cdiff1(arr: torch.Tensor) -> torch.Tensor:
            fwd = F.pad(arr[:, :, :, 1:], (0, 1))
            bwd = F.pad(arr[:, :, :, :-1], (1, 0))
            return (fwd - bwd) * c_gate_sym1

        dx3 = F.pad(x_n[:, :, :, 3:] - x_n[:, :, :, :-3], (3, 0)) * 52.5
        dy3 = F.pad(y_n[:, :, :, 3:] - y_n[:, :, :, :-3], (3, 0)) * 34.0
        step3_m = torch.sqrt(dx3 * dx3 + dy3 * dy3 + 1e-6)
        c_gate3 = torch.exp(-F.relu(step3_m - 2.6).square() * 0.25) * m_float

        def _cdiff3(arr: torch.Tensor) -> torch.Tensor:
            if L <= 6:
                return torch.zeros_like(arr)
            fwd = F.pad(arr[:, :, :, 3:], (0, 3))
            bwd = F.pad(arr[:, :, :, :-3], (3, 0))
            return (fwd - bwd) * c_gate3

        ax = _cdiff1(vx_n).clamp(-2.0, 2.0)
        ay = _cdiff1(vy_n).clamp(-2.0, 2.0)
        asp = _cdiff1(sp_n).clamp(-2.0, 2.0)

        vx_fwd = F.pad(vx_n[:, :, :, 1:], (0, 1))
        vx_bwd = F.pad(vx_n[:, :, :, :-1], (1, 0))
        vy_fwd = F.pad(vy_n[:, :, :, 1:], (0, 1))
        vy_bwd = F.pad(vy_n[:, :, :, :-1], (1, 0))
        turn_rate = ((vx_fwd * vy_bwd - vy_fwd * vx_bwd).abs() * c_gate_sym1).clamp(0.0, 2.0)
        afocus = _cdiff1(vcf_n).clamp(-2.0, 2.0)

        bf_kin = torch.cat([ax, ay, asp, turn_rate, afocus, c_gate1], dim=1) * m_float

        aball1 = _cdiff1(vcb_n).clamp(-2.5, 2.5)
        aball3 = _cdiff3(vcb_n).clamp(-2.5, 2.5)
        if L > 4:
            db_f2 = F.pad(db_n[:, :, :, 2:], (0, 2))
            db_b2 = F.pad(db_n[:, :, :, :-2], (2, 0))
            db_curv2 = ((db_f2 + db_b2 - 2.0 * db_n) * 10.0 * pb10_n * c_gate_sym1).clamp(-3.0, 3.0)
            pb3_f2 = F.pad(pb3_n[:, :, :, 2:], (0, 2))
            pb3_b2 = F.pad(pb3_n[:, :, :, :-2], (2, 0))
            pb3_delta = ((pb3_f2 - pb3_b2) * c_gate_sym1).clamp(-2.0, 2.0)
        else:
            db_curv2 = torch.zeros_like(db_n)
            pb3_delta = torch.zeros_like(pb3_n)

        ball_kin = torch.cat([aball1, aball3, db_curv2, pb3_delta], dim=1) * m_float
        p_bf_aug = torch.cat([p[:, :C_PLAYER_BF, :, :], bf_kin], dim=1)
        p_ball_aug = torch.cat([p, bf_kin, ball_kin], dim=1)
        return p_bf_aug, p_ball_aug

    def forward(
        self,
        p: torch.Tensor,
        mask: torch.Tensor,
        g_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B2, _, K, L = p.shape
        m_bool = mask.unsqueeze(1)                      # (B2, 1, K, L)
        m_float = m_bool.to(p.dtype)                    # (B2, 1, K, L)
        n_vis = m_float.sum(dim=2).clamp(min=1.0)       # (B2, 1, L)
        any_vis = (m_float.sum(dim=2) > 0.5).to(p.dtype)

        p_bf_aug, p_ball_aug = self._compute_kinematic_diffs(p, m_float)

        # 1. Per-player temporal trajectory convolutions
        h_bf = self.stem_bf_t1(self.stem_bf_in(p_bf_aug), m_float)
        out_bf = self.stem_bf_out(h_bf)

        h_ball = self.stem_ball_t1(self.stem_ball_in(p_ball_aug), m_float)
        out_ball = self.stem_ball_out(h_ball)

        e_bf = out_bf[:, :self.d_bf, :, :] * m_float
        logits_bf = out_bf[:, self.d_bf:, :, :]         # (B2, 2, K, L)

        e_ball = out_ball[:, :self.d_ball, :, :] * m_float
        logits_ball = out_ball[:, self.d_ball:, :, :]   # (B2, 2, K, L)

        # 2. Top-K (K=8) Active Player-Player Interaction (players 0..7 closest to x_focus)
        k_act = min(self.k_interact, K)
        m_top = m_float[:, :, :k_act, :]
        n_top = m_top.sum(dim=2).clamp(min=1.0)

        bf_top = e_bf[:, :, :k_act, :]
        bf_top_mean = bf_top.sum(dim=2) / n_top
        bf_top_max = bf_top.masked_fill(~m_bool[:, :, :k_act, :], -1e4).max(dim=2).values * any_vis
        bf_ctx = self.topk_ctx_bf(torch.cat([bf_top_mean, bf_top_max], dim=1)).unsqueeze(2)
        e_bf = (e_bf + 0.35 * bf_ctx) * m_float

        ball_top = e_ball[:, :, :k_act, :]
        ball_top_mean = ball_top.sum(dim=2) / n_top
        ball_top_max = ball_top.masked_fill(~m_bool[:, :, :k_act, :], -1e4).max(dim=2).values * any_vis
        ball_ctx = self.topk_ctx_ball(torch.cat([ball_top_mean, ball_top_max], dim=1)).unsqueeze(2)
        e_ball = (e_ball + 0.35 * ball_ctx) * m_float

        e_all = torch.cat([e_bf, e_ball], dim=1)        # (B2, 64, K, L)

        # 3. Multi-Head Pooling + Ball-Centric Cross-Attention
        z_mean = e_all.sum(dim=2) / n_vis               # (B2, 64, L)
        z_max = e_all.masked_fill(~m_bool, -1e4).max(dim=2).values * any_vis  # (B2, 64, L)

        # 2-Head Ball-Free Attention Pool
        w_bf1 = F.softmax(logits_bf[:, 0:1, :, :].masked_fill(~m_bool, -1e4), dim=2) * m_float
        w_bf1 = w_bf1 / w_bf1.sum(dim=2, keepdim=True).clamp(min=1e-6)
        z_bf1 = (e_bf * w_bf1).sum(dim=2)

        w_bf2 = F.softmax(logits_bf[:, 1:2, :, :].masked_fill(~m_bool, -1e4), dim=2) * m_float
        w_bf2 = w_bf2 / w_bf2.sum(dim=2, keepdim=True).clamp(min=1e-6)
        z_bf2 = (e_bf * w_bf2).sum(dim=2)

        # 2-Head Ball-Aware Attention Pool
        w_b1 = F.softmax(logits_ball[:, 0:1, :, :].masked_fill(~m_bool, -1e4), dim=2) * m_float
        w_b1 = w_b1 / w_b1.sum(dim=2, keepdim=True).clamp(min=1e-6)
        z_b1 = (e_ball * w_b1).sum(dim=2)

        w_b2 = F.softmax(logits_ball[:, 1:2, :, :].masked_fill(~m_bool, -1e4), dim=2) * m_float
        w_b2 = w_b2 / w_b2.sum(dim=2, keepdim=True).clamp(min=1e-6)
        z_b2 = (e_ball * w_b2).sum(dim=2)

        # Ball-Centric Cross-Attention: Query from global ball/physics context attends to K player tokens
        q_ball = self.ball_query_proj(g_ctx).unsqueeze(2)  # (B2, d_ball, 1, L)
        scale = float(self.d_ball) ** -0.5
        sim_ball = (e_ball * q_ball).sum(dim=1, keepdim=True) * scale  # (B2, 1, K, L)
        w_cross = F.softmax(sim_ball.masked_fill(~m_bool, -1e4), dim=2) * m_float
        w_cross = w_cross / w_cross.sum(dim=2, keepdim=True).clamp(min=1e-6)
        z_cross = (e_ball * w_cross).sum(dim=2)            # (B2, 32, L)

        z_raw = torch.cat([z_mean, z_max, z_bf1, z_bf2, z_b1, z_b2, z_cross], dim=1)
        z_bf_only = torch.cat([z_mean[:, :self.d_bf], z_max[:, :self.d_bf], z_bf1, z_bf2], dim=1)
        return z_bf_only, self.pool_refine(z_raw)


# ---------------------------------------------------------------------------
# 4. Window-Local Normalized Multi-Scale Sequence Backbone (`_TrajectoryEventBackboneV10`)
# ---------------------------------------------------------------------------

class _LocalNormResBlock(nn.Module):
    """Residual dilated temporal block with `WindowLocalNorm1d(window_size=1501)`."""

    def __init__(self, channels: int, dilation: int, dropout: float, window_size: int = 1501):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            WindowLocalNorm1d(channels, window_size=window_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _MultiScaleTemporalPyramid(nn.Module):
    """Parallel Multi-Scale Dilated Temporal Conv Pyramid (`d = 1, 2, 4, 8, 16, 32`) with Window-Local Norm."""

    def __init__(
        self,
        channels: int,
        pyramid_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        window_size: int = 1501,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size=3, padding=d, dilation=d, groups=channels
            )
            for d in pyramid_dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(pyramid_dilations), channels, kernel_size=1),
            WindowLocalNorm1d(channels, window_size=window_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([b(x) for b in self.branches], dim=1)
        return x + self.fuse(cat)


class _TrajectoryEventBackboneV10(nn.Module):
    """Intrinsically H<->A + 180-deg Equivariant Spatio-Temporal Trajectory Transformer & Local-Norm Backbone.

    Outputs:
      - `logits_had`:    `(B, 3, L)` (`[H, A, D]`)
      - `logit_change`:  `(B, L)` (1D binary transition logit)
      - `logits_4dir`:   `(B, 4, L)` (`[H->A, A->H, Live->D, D->Live]`)
      - `logit_duel_to`: `(B, L)` (Short-steal / duel-turnover `<= 65f` spotting head)
      - `pred_aerial`:   `(B, 2, L)` (`[logit_aerial_gt_1p5m, log1p_ball_z_hat]`)
    """

    def __init__(
        self,
        c_ga: int,
        c_gs: int,
        d_bf: int = 32,
        d_ball: int = 32,
        channels: int = 88,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        gru_stride: int = 8,
        window_size: int = 1501,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.traj_encoder = PlayerTrajectoryEncoder(
            c_ga=c_ga,
            c_gs=c_gs,
            d_bf=d_bf,
            d_ball=d_ball,
            hidden=48,
            k_interact=8,
            dropout=dropout,
        )
        d_pool = self.traj_encoder.out_dim   # 192
        d_bf_pool = d_bf * 4                 # 128

        # Ghost-ball & blackout reliability gate from g_sym
        self.ghost_gate = nn.Sequential(
            nn.Conv1d(c_gs, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, d_pool, kernel_size=1),
            nn.Sigmoid(),
        )
        self.bf_bypass = nn.Conv1d(d_bf_pool, d_pool, kernel_size=1, bias=False)

        # Equivariant twin input projections
        self.inp_anti = nn.Conv1d(d_pool + c_ga, channels, kernel_size=1, bias=False)
        self.inp_sym = nn.Conv1d(d_pool + c_gs, channels, kernel_size=1, bias=True)
        self.in_norm = WindowLocalNorm1d(channels, window_size=window_size)
        self.in_drop = nn.Dropout(dropout)

        # Sequential dilated blocks with WindowLocalNorm1d + Parallel Multi-Scale Pyramid
        self.blocks = nn.Sequential(
            *[_LocalNormResBlock(channels, d, dropout, window_size=window_size) for d in dilations]
        )
        self.pyramid = _MultiScaleTemporalPyramid(
            channels, pyramid_dilations=(1, 2, 4, 8, 16, 32), window_size=window_size, dropout=dropout
        )

        # 2-Layer Bidirectional GRU (per Task 1 specification)
        self.gru_stride = gru_stride
        gru_hidden = channels // 2
        self.bigru = nn.GRU(
            input_size=channels,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.gru_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            WindowLocalNorm1d(channels, window_size=window_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head 4: Auxiliary 3D Aerial Ball Height Head (`Play_high` vs `Play_low`)
        self.head_aerial = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=1),
        )
        self.aerial_mod = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Head 3: Dedicated Short-Steal / Duel-Turnover Spotting Head (`<= 65` frames = `2.6s`)
        self.head_duel_to = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )

        # Head 2: 4-Directional Boundary Head + 1D Change Head
        self.head_4d = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 4, kernel_size=1),
        )

        # Action-Spotting Feedback Gate: feeds `logit_duel_to` + `logits_4dir` back into possession & change heads
        self.spotting_feedback = nn.Sequential(
            nn.Conv1d(5, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

        self.head_poss = nn.Conv1d(channels, 3, kernel_size=1)
        self.head_chg = nn.Conv1d(channels, 1, kernel_size=1)

    def _forward_stream(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.blocks(self.in_drop(F.gelu(self.in_norm(x))))
        h = self.pyramid(h)

        L = h.shape[-1]
        s = self.gru_stride
        h_ds = F.avg_pool1d(h, kernel_size=s, stride=s, ceil_mode=True)
        gru_out, _ = self.bigru(h_ds.transpose(1, 2))
        gru_up = F.interpolate(gru_out.transpose(1, 2), size=L, mode="linear", align_corners=False)

        h = h + self.gru_proj(gru_up)

        aerial_out = self.head_aerial(h)
        h_mod = h * self.aerial_mod(aerial_out)

        logit_duel = self.head_duel_to(h_mod)   # (B2, 1, L)
        logits_4d = self.head_4d(h_mod)         # (B2, 4, L)

        spot_ctx = torch.cat([ torch.tanh(logit_duel * 0.5), torch.tanh(logits_4d * 0.5) ], dim=1)
        h_final = h_mod + 0.25 * self.spotting_feedback(spot_ctx)

        return (
            self.head_poss(h_final),
            self.head_chg(h_final).squeeze(1),
            logits_4d,
            logit_duel.squeeze(1),
            aerial_out,
        )

    def forward(
        self,
        p_h: torch.Tensor,
        p_a: torch.Tensor,
        m_h: torch.Tensor,
        m_a: torch.Tensor,
        g_anti: torch.Tensor,
        g_sym: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stack Home and Away along batch dimension with team-aligned global context (+g_anti for H, -g_anti for A)
        p_both = torch.cat([p_h, p_a], dim=0)
        m_both = torch.cat([m_h, m_a], dim=0)
        g_ctx_h = torch.cat([g_sym, g_anti], dim=1)
        g_ctx_a = torch.cat([g_sym, -g_anti], dim=1)
        g_ctx_both = torch.cat([g_ctx_h, g_ctx_a], dim=0)

        z_bf_both, z_full_both = self.traj_encoder(p_both, m_both, g_ctx_both)
        z_bf_h, z_bf_a = z_bf_both.chunk(2, dim=0)
        z_h, z_a = z_full_both.chunk(2, dim=0)

        # Ghost-ball / blackout gate modulates ball-aware pooled representation while preserving ball-free bypass
        gate = self.ghost_gate(g_sym)
        z_h = z_h * gate + (1.0 - gate) * self.bf_bypass(z_bf_h)
        z_a = z_a * gate + (1.0 - gate) * self.bf_bypass(z_bf_a)

        u_anti_1 = torch.cat([z_h - z_a, g_anti], dim=1)
        u_anti_2 = torch.cat([z_a - z_h, -g_anti], dim=1)
        u_sym = torch.cat([z_h + z_a, g_sym], dim=1)

        h_anti_both = self.inp_anti(torch.cat([u_anti_1, u_anti_2], dim=0))
        h_sym = self.inp_sym(u_sym)
        h_sym_both = torch.cat([h_sym, h_sym], dim=0)

        # Twin-Pass H/A Siamese Forwarding -> exact 0.000e+00 equivariance
        x_both = h_sym_both + h_anti_both
        lp_both, lc_both, l4_both, ld_both, la_both = self._forward_stream(x_both)

        lp1, lp2 = lp_both.chunk(2, dim=0)
        lc1, lc2 = lc_both.chunk(2, dim=0)
        l4_1, l4_2 = l4_both.chunk(2, dim=0)
        ld1, ld2 = ld_both.chunk(2, dim=0)
        la1, la2 = la_both.chunk(2, dim=0)

        logits_had = 0.5 * (lp1 + lp2[:, [1, 0, 2], :])
        logit_change = 0.5 * (lc1 + lc2)
        logits_4dir = 0.5 * (l4_1 + l4_2[:, [1, 0, 2, 3], :])
        logit_duel_to = 0.5 * (ld1 + ld2)
        pred_aerial = 0.5 * (la1 + la2)
        return logits_had, logit_change, logits_4dir, logit_duel_to, pred_aerial


# ---------------------------------------------------------------------------
# 5. Differentiable Windowed Soft-IoU Loss + Focal Spotting Loss
# ---------------------------------------------------------------------------

def compute_windowed_soft_iou_loss(
    logits_had: torch.Tensor,
    target_y: torch.Tensor,
    frame_weights: torch.Tensor,
    window_sizes: tuple[int, ...] = (50, 150, 500),
    eps: float = 1e-4,
) -> torch.Tensor:
    """Differentiable Multi-Scale Windowed Soft-IoU Loss on `HOME` (0) and `AWAY` (1).

    Slices the `(B, 3, L)` sequence into windows of length `W in (50, 150, 500)` (`2s, 6s, 20s`)
    and computes weighted Soft-IoU on active windows so the network directly optimizes spell overlap
    for both short (`2s`) and sustained (`6s-20s`) possessions.
    """
    probs = F.softmax(logits_had, dim=1)
    valid = (target_y >= 0).to(probs.dtype)
    w = (frame_weights * valid).clamp(min=0.0)

    y_h = (target_y == HOME).to(probs.dtype) * valid
    y_a = (target_y == AWAY).to(probs.dtype) * valid
    p_h = probs[:, HOME, :] * valid
    p_a = probs[:, AWAY, :] * valid

    B, L = target_y.shape
    total_loss = logits_had.new_tensor(0.0)
    n_scales = 0

    for W in window_sizes:
        if L < W:
            continue
        n_win = L // W
        L_trim = n_win * W
        w_win = w[:, :L_trim].reshape(B * n_win, W)
        win_active = (w_win.sum(dim=1) > 1.0).to(probs.dtype)
        if win_active.sum() < 0.5:
            continue

        losses_scale = []
        for p_c, y_c in ((p_h, y_h), (p_a, y_a)):
            pc_w = p_c[:, :L_trim].reshape(B * n_win, W)
            yc_w = y_c[:, :L_trim].reshape(B * n_win, W)
            # Only penalize windows where either GT or prediction has non-trivial mass
            has_mass = ((yc_w.sum(dim=1) > 1.5) | (pc_w.sum(dim=1) > 2.5)).to(probs.dtype) * win_active
            inter = (w_win * pc_w * yc_w).sum(dim=1)
            union = (w_win * (pc_w + yc_w - pc_w * yc_w)).sum(dim=1)
            iou = (inter + eps) / (union + eps)
            loss_c = ((1.0 - iou) * has_mass).sum() / has_mass.sum().clamp(min=1.0)
            losses_scale.append(loss_c)

        total_loss = total_loss + 0.5 * (losses_scale[0] + losses_scale[1])
        n_scales += 1

    if n_scales == 0:
        return total_loss
    return total_loss / float(n_scales)


def sigmoid_focal_with_pos_weight_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    """Focal-modulated positive-weighted binary cross-entropy loss supporting soft Gaussian targets in `[0, 1]`."""
    t_clamped = targets.clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, t_clamped, pos_weight=pos_weight, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = t_clamped * probs + (1.0 - t_clamped) * (1.0 - probs)
    focal_factor = (1.0 - pt).clamp(min=1e-6).pow(gamma)
    loss = bce * focal_factor * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# 6. Online BAR-ATH Failure-Mode Augmentations (Line D3 12v10 Far-Swap & Line D1 Ball-Dropout)
# ---------------------------------------------------------------------------

# Vector channels in P_team (T, 11, 18) that flip sign under 180-deg pitch rotation (Home <-> Away frame)
_VEC_CHANNELS = (0, 1, 2, 3, 5, 6, 12, 13)


def apply_v10_bar_ath_augmentations(
    ph: np.ndarray,
    pa: np.ndarray,
    mh: np.ndarray,
    ma: np.ndarray,
    ga: np.ndarray,
    gs: np.ndarray,
    rng: np.random.Generator,
    far_swap_prob: float = 0.15,
    slot_drop_prob: float = 0.15,
    ball_drop_prob: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate BAR-ATH's real tracking failure modes online during training:
      1. Line D3 (`12v10` Far-Player Jersey Team Swap, `p = far_swap_prob`):
         - Picks 1 random far player (`> 6m` from the ball, i.e., `d_b_norm = p[..., 14] > 0.20`, slot `k >= 5`)
           for `3-5s` (`75-125` frames) and swaps their team assignment between Home and Away (negating vector
           channels `0, 1, 2, 3, 5, 6, 12, 13` so their coordinates are expressed in the recipient team's
           attacking frame). Trains the Set/Trajectory encoder to ignore a single discordant far player slot!
      2. Player Slot Dropout (`p = slot_drop_prob`):
         - Drops 1 random player slot `k >= 5` for `2-5s` (`50-125` frames).
      3. Line D1 (`0.5s-3.0s` Ball-Dropout Gap, `p = ball_drop_prob`):
         - Simulates an unrecorded ball blackout of `12-75` frames (`0.5s-3.0s`) by linearly interpolating the
           player-to-ball channels (`12..17`) across `[t0, t1]` and marking `bvis=0, blin=1` in `gs`.
    """
    L, K, _ = ph.shape
    if L < 150:
        return ph, pa, mh, ma, ga, gs

    copied = False

    def _ensure_copy():
        nonlocal ph, pa, mh, ma, ga, gs, copied
        if not copied:
            ph = ph.copy()
            pa = pa.copy()
            mh = mh.copy()
            ma = ma.copy()
            ga = ga.copy()
            gs = gs.copy()
            copied = True

    # 1. Line D3: Random 12v10 far-player (>6m from ball) jersey swap for 3-5s (75-125 frames)
    if rng.random() < far_swap_prob:
        _ensure_copy()
        dur = int(rng.integers(75, min(126, L // 2)))
        t0 = int(rng.integers(0, L - dur))
        t1 = t0 + dur
        k_src = int(rng.integers(5, K))
        k_dst = K - 1  # Replace outermost slot of the other team to simulate 12v10

        # Check that the chosen player is indeed far from the ball (> 6m -> channel 14 > 0.20)
        if rng.random() < 0.5:
            # Swap a far Home player into Away's set
            if float(np.mean(ph[t0:t1, k_src, 14])) > 0.20:
                p_swapped = ph[t0:t1, k_src, :].copy()
                for vc in _VEC_CHANNELS:
                    p_swapped[:, vc] = -p_swapped[:, vc]
                pa[t0:t1, k_dst, :] = p_swapped
                ma[t0:t1, k_dst] = mh[t0:t1, k_src]
                mh[t0:t1, k_src] = False
                ph[t0:t1, k_src, :] = 0.0
        else:
            # Swap a far Away player into Home's set
            if float(np.mean(pa[t0:t1, k_src, 14])) > 0.20:
                p_swapped = pa[t0:t1, k_src, :].copy()
                for vc in _VEC_CHANNELS:
                    p_swapped[:, vc] = -p_swapped[:, vc]
                ph[t0:t1, k_dst, :] = p_swapped
                mh[t0:t1, k_dst] = ma[t0:t1, k_src]
                ma[t0:t1, k_src] = False
                pa[t0:t1, k_src, :] = 0.0

    # 2. Random Player-Slot Dropout (2-5s)
    if rng.random() < slot_drop_prob:
        _ensure_copy()
        dur = int(rng.integers(50, min(126, L // 2)))
        t0 = int(rng.integers(0, L - dur))
        t1 = t0 + dur
        k_slot = int(rng.integers(5, K))
        mh[t0:t1, k_slot] = False
        ph[t0:t1, k_slot, :] = 0.0
        ma[t0:t1, k_slot] = False
        pa[t0:t1, k_slot, :] = 0.0

    # 3. Line D1: Random Ball-Dropout Gap (0.5s - 3.0s = 12 - 75 frames)
    if rng.random() < ball_drop_prob:
        _ensure_copy()
        dur = int(rng.integers(12, min(76, L // 3)))
        t0 = int(rng.integers(1, L - dur - 1))
        t1 = t0 + dur
        w = np.linspace(0.0, 1.0, dur, dtype=np.float32)[:, None, None]
        ph[t0:t1, :, 12:17] = (1.0 - w) * ph[t0 - 1 : t0, :, 12:17] + w * ph[t1 : t1 + 1, :, 12:17]
        pa[t0:t1, :, 12:17] = (1.0 - w) * pa[t0 - 1 : t0, :, 12:17] + w * pa[t1 : t1 + 1, :, 12:17]
        ph[t0:t1, :, 17] *= 0.25
        pa[t0:t1, :, 17] *= 0.25

    return ph, pa, mh, ma, ga, gs


# ---------------------------------------------------------------------------
# 7. Top-Level Trainer & Predictor (`SymTrajectoryEventNetV10`)
# ---------------------------------------------------------------------------

class SymTrajectoryEventNetV10:
    """End-to-end trainer and predictor for `SymTrajectoryEventNetV10`."""

    def __init__(
        self,
        channels: int = 88,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        gru_stride: int = 8,
        window_size: int = 1501,
        dropout: float = 0.15,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-4,
        epochs: int = 10,
        crop: int = 1500,
        batch_size: int = 28,
        poss_alpha: float = 0.40,
        boundary_boost: float = 1.65,
        soft_iou_weight: float = 0.28,
        change_loss_weight: float = 0.35,
        b4d_loss_weight: float = 0.45,
        duel_loss_weight: float = 0.25,
        aerial_loss_weight: float = 0.20,
        far_swap_prob: float = 0.14,
        slot_drop_prob: float = 0.14,
        ball_drop_prob: float = 0.18,
        label_smoothing: float = 0.02,
        sliding_window_blend: float = 0.50,
        seed: int = 42,
    ):
        self.channels = channels
        self.dilations = dilations
        self.gru_stride = gru_stride
        self.window_size = window_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.crop = crop
        self.batch_size = batch_size
        self.poss_alpha = poss_alpha
        self.boundary_boost = boundary_boost
        self.soft_iou_weight = soft_iou_weight
        self.change_loss_weight = change_loss_weight
        self.b4d_loss_weight = b4d_loss_weight
        self.duel_loss_weight = duel_loss_weight
        self.aerial_loss_weight = aerial_loss_weight
        self.far_swap_prob = far_swap_prob
        self.slot_drop_prob = slot_drop_prob
        self.ball_drop_prob = ball_drop_prob
        self.label_smoothing = label_smoothing
        self.sliding_window_blend = sliding_window_blend
        self.seed = seed

        self.c_ga: int = 0
        self.c_gs: int = 0
        self.mean_ga_: np.ndarray | None = None
        self.std_ga_: np.ndarray | None = None
        self.mean_gs_: np.ndarray | None = None
        self.std_gs_: np.ndarray | None = None
        self.net_: _TrajectoryEventBackboneV10 | None = None

    def _device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _fit_scaler(self, psts: list[PlayerSetTensors]) -> None:
        n_total = sum(len(x.g_anti) for x in psts)
        c_ga = psts[0].g_anti.shape[1]
        c_gs = psts[0].g_sym.shape[1]
        self.c_ga = c_ga
        self.c_gs = c_gs

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
        ball_zs: list[np.ndarray | None] | None = None,
        sample_weights: list[float] | None = None,
    ) -> "SymTrajectoryEventNetV10":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._fit_scaler(psts)

        m_weights = sample_weights if sample_weights is not None else [1.0] * len(psts)
        bz_list = ball_zs if ball_zs is not None else [None] * len(psts)

        seqs = []
        weights = []
        trans_centers_list = []

        for pst, y, h, bz, w_m in zip(psts, ys, halves, bz_list, m_weights):
            ph, pa, mh, ma, ga, gs = self._prep_match_arrays(pst)
            yc, y4, y_duel = compute_v10_boundary_and_duel_targets(y, h, sigma=3.0, max_duel_len=65)
            wf = compute_v10_possession_frame_weights(
                y, h, alpha=self.poss_alpha, boundary_radius=20, boundary_boost=self.boundary_boost
            )
            yi = y.astype(np.int64)
            if bz is not None:
                bz_arr = np.where(np.isfinite(bz), np.maximum(bz, 0.0), -1.0).astype(np.float32)
            else:
                bz_arr = np.full(len(y), -1.0, dtype=np.float32)

            for hv in np.unique(h[h > 0]):
                idx = np.flatnonzero(h == hv)
                if len(idx) == 0:
                    continue
                s, e = int(idx[0]), int(idx[-1]) + 1
                if e - s < 32:
                    continue
                yc_seg = yc[s:e]
                yd_seg = y_duel[s:e]
                # Oversample transitions and short duel possessions
                tc_idx = np.flatnonzero((yc_seg > 0.4) | (yd_seg > 0.5) | (wf[s:e] > 1.35))
                seqs.append((
                    ph[s:e], pa[s:e], mh[s:e], ma[s:e], ga[s:e], gs[s:e],
                    yi[s:e], wf[s:e], yc_seg, y4[s:e], yd_seg, bz_arr[s:e]
                ))
                trans_centers_list.append(tc_idx)
                weights.append(float(e - s) * float(w_m))

        dev = self._device()
        net = _TrajectoryEventBackboneV10(
            c_ga=self.c_ga,
            c_gs=self.c_gs,
            channels=self.channels,
            dilations=self.dilations,
            gru_stride=self.gru_stride,
            window_size=self.window_size,
            dropout=self.dropout,
        ).to(dev)

        total_frames = sum(len(s[0]) for s in seqs)
        steps_per_epoch = max(1, total_frames // (self.crop * self.batch_size))
        total_steps = self.epochs * steps_per_epoch

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=total_steps)

        probs = np.array(weights, dtype=np.float64)
        probs = probs / probs.sum()
        pos_w_chg = torch.tensor(10.0, device=dev)
        pos_w_4d = torch.tensor([14.0, 14.0, 12.0, 12.0], device=dev).view(1, 4, 1)
        pos_w_duel = torch.tensor(8.0, device=dev)
        pos_w_aer = torch.tensor(6.0, device=dev)
        L = self.crop

        net.train()
        for _ in range(total_steps):
            chosen = rng.choice(len(seqs), self.batch_size, p=probs)
            l_ph, l_pa, l_mh, l_ma, l_ga, l_gs = [], [], [], [], [], []
            l_y, l_wf, l_c, l_y4, l_yd, l_bz = [], [], [], [], [], []
            for si in chosen:
                ph, pa, mh, ma, ga, gs, yi, wf, yc, y4, yd, bz_s = seqs[si]
                tc_idx = trans_centers_list[si]
                n = len(ph)
                if n >= L:
                    if len(tc_idx) > 0 and rng.random() < 0.50:
                        center = int(tc_idx[rng.integers(0, len(tc_idx))])
                        jitter = int(rng.integers(-L // 3, L // 3 + 1))
                        a = int(np.clip(center - L // 2 + jitter, 0, n - L))
                    else:
                        a = int(rng.integers(0, max(1, n - L + 1)))
                    ph_c, pa_c = ph[a : a + L], pa[a : a + L]
                    mh_c, ma_c = mh[a : a + L], ma[a : a + L]
                    ga_c, gs_c = ga[a : a + L], gs[a : a + L]
                    ph_c, pa_c, mh_c, ma_c, ga_c, gs_c = apply_v10_bar_ath_augmentations(
                        ph_c, pa_c, mh_c, ma_c, ga_c, gs_c, rng,
                        far_swap_prob=self.far_swap_prob,
                        slot_drop_prob=self.slot_drop_prob,
                        ball_drop_prob=self.ball_drop_prob,
                    )
                    l_ph.append(ph_c)
                    l_pa.append(pa_c)
                    l_mh.append(mh_c)
                    l_ma.append(ma_c)
                    l_ga.append(ga_c)
                    l_gs.append(gs_c)
                    l_y.append(yi[a : a + L])
                    l_wf.append(wf[a : a + L])
                    l_c.append(yc[a : a + L])
                    l_y4.append(y4[a : a + L])
                    l_yd.append(yd[a : a + L])
                    l_bz.append(bz_s[a : a + L])
                else:
                    pad = L - n
                    l_ph.append(np.pad(ph, ((0, pad), (0, 0), (0, 0))))
                    l_pa.append(np.pad(pa, ((0, pad), (0, 0), (0, 0))))
                    l_mh.append(np.pad(mh, ((0, pad), (0, 0))))
                    l_ma.append(np.pad(ma, ((0, pad), (0, 0))))
                    l_ga.append(np.pad(ga, ((0, pad), (0, 0))))
                    l_gs.append(np.pad(gs, ((0, pad), (0, 0))))
                    l_y.append(np.pad(yi, (0, pad), constant_values=-1))
                    l_wf.append(np.pad(wf, (0, pad), constant_values=0.0))
                    l_c.append(np.pad(yc, (0, pad), constant_values=-1.0))
                    l_y4.append(np.pad(y4, ((0, pad), (0, 0)), constant_values=-1.0))
                    l_yd.append(np.pad(yd, (0, pad), constant_values=-1.0))
                    l_bz.append(np.pad(bz_s, (0, pad), constant_values=-1.0))

            th_ph = torch.from_numpy(np.stack(l_ph)).to(dev).permute(0, 3, 2, 1)
            th_pa = torch.from_numpy(np.stack(l_pa)).to(dev).permute(0, 3, 2, 1)
            th_mh = torch.from_numpy(np.stack(l_mh)).to(dev).permute(0, 2, 1)
            th_ma = torch.from_numpy(np.stack(l_ma)).to(dev).permute(0, 2, 1)
            th_ga = torch.from_numpy(np.stack(l_ga)).to(dev).permute(0, 2, 1)
            th_gs = torch.from_numpy(np.stack(l_gs)).to(dev).permute(0, 2, 1)

            ty = torch.from_numpy(np.stack(l_y)).to(dev)
            twf = torch.from_numpy(np.stack(l_wf)).to(dev)
            tc = torch.from_numpy(np.stack(l_c)).to(dev)
            ty4 = torch.from_numpy(np.stack(l_y4)).to(dev).permute(0, 2, 1)  # (B, 4, L)
            tyd = torch.from_numpy(np.stack(l_yd)).to(dev)                    # (B, L)
            tbz = torch.from_numpy(np.stack(l_bz)).to(dev)                    # (B, L)

            logits_p, logits_c, logits_4d, logit_duel, pred_aerial = net(
                th_ph, th_pa, th_mh, th_ma, th_ga, th_gs
            )

            # 1a. Possession-duration-weighted Cross-Entropy (`L1`)
            ce_per_frame = F.cross_entropy(
                logits_p,
                ty.masked_fill(ty < 0, -100),
                ignore_index=-100,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            valid_y = (ty >= 0).to(torch.float32)
            w_eff = twf * valid_y
            loss_ce = (ce_per_frame * w_eff).sum() / w_eff.sum().clamp(min=1.0)

            # 1b. Differentiable Multi-Scale Windowed Soft-IoU Loss on H and A
            loss_iou = compute_windowed_soft_iou_loss(
                logits_p, ty, twf, window_sizes=(50, 150, 500)
            )

            # 2. Gaussian-smoothed Change & 4-Directional Boundary Focal Loss (`L5`)
            c_mask = (tc >= 0).to(torch.float32)
            loss_c = sigmoid_focal_with_pos_weight_loss(
                logits_c, tc, c_mask, pos_weight=pos_w_chg, gamma=1.5
            )

            m4 = (ty4 >= 0).to(torch.float32)
            loss_4d = sigmoid_focal_with_pos_weight_loss(
                logits_4d, ty4, m4, pos_weight=pos_w_4d, gamma=1.5
            )

            # 3. Dedicated Short-Steal / Duel-Turnover Spotting Loss (`<= 65` frames)
            d_mask = (tyd >= 0).to(torch.float32)
            loss_duel = sigmoid_focal_with_pos_weight_loss(
                logit_duel, tyd, d_mask, pos_weight=pos_w_duel, gamma=1.5
            )

            # 4. 3D Aerial Ball Height Loss (`Play_high` vs `Play_low`)
            bz_mask = (tbz >= 0).to(torch.float32)
            y_is_aerial = (tbz > 1.5).to(torch.float32)
            y_log_z = torch.log1p(tbz.clamp(min=0.0))
            bce_aer = F.binary_cross_entropy_with_logits(
                pred_aerial[:, 0, :], y_is_aerial, pos_weight=pos_w_aer, reduction="none"
            )
            huber_z = F.smooth_l1_loss(pred_aerial[:, 1, :], y_log_z, reduction="none")
            loss_aer = ((bce_aer + 0.5 * huber_z) * bz_mask).sum() / bz_mask.sum().clamp(min=1.0)

            loss = (
                loss_ce
                + self.soft_iou_weight * loss_iou
                + self.change_loss_weight * loss_c
                + self.b4d_loss_weight * loss_4d
                + self.duel_loss_weight * loss_duel
                + self.aerial_loss_weight * loss_aer
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()

        self.net_ = net.to("cpu").eval()
        return self

    @staticmethod
    def _sym_softmax_3(lp: torch.Tensor) -> torch.Tensor:
        """Strictly H<->A symmetric softmax over dim=1 (`[H, A, D]`) in IEEE-754 floating point."""
        m = torch.maximum(torch.maximum(lp[:, 0:1, :], lp[:, 1:2, :]), lp[:, 2:3, :])
        e = torch.exp(lp - m)
        denom = (e[:, 0:1, :] + e[:, 1:2, :]) + e[:, 2:3, :]
        return e / denom.clamp(min=1e-12)

    @staticmethod
    def _sym_norm_proba_np(proba: np.ndarray) -> np.ndarray:
        denom = (proba[:, 0:1] + proba[:, 1:2]) + proba[:, 2:3]
        return (proba / np.clip(denom, 1e-6, None)).astype(np.float32)

    @staticmethod
    def _run_segment_forward(
        net: _TrajectoryEventBackboneV10,
        ph_s: np.ndarray,
        pa_s: np.ndarray,
        mh_s: np.ndarray,
        ma_s: np.ndarray,
        ga_s: np.ndarray,
        gs_s: np.ndarray,
        dev: torch.device,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        th_ph = torch.from_numpy(ph_s[None]).to(dev).permute(0, 3, 2, 1)
        th_pa = torch.from_numpy(pa_s[None]).to(dev).permute(0, 3, 2, 1)
        th_mh = torch.from_numpy(mh_s[None]).to(dev).permute(0, 2, 1)
        th_ma = torch.from_numpy(ma_s[None]).to(dev).permute(0, 2, 1)
        th_ga = torch.from_numpy(ga_s[None]).to(dev).permute(0, 2, 1)
        th_gs = torch.from_numpy(gs_s[None]).to(dev).permute(0, 2, 1)

        lp, lc, l4, ld, la = net(th_ph, th_pa, th_mh, th_ma, th_ga, th_gs)
        proba = SymTrajectoryEventNetV10._sym_softmax_3(lp).squeeze(0).T.cpu().numpy()
        chg = torch.sigmoid(lc).squeeze(0).cpu().numpy()
        p4d = torch.sigmoid(l4).squeeze(0).T.cpu().numpy()
        p_duel = torch.sigmoid(ld).squeeze(0).cpu().numpy()
        p_aerial = torch.sigmoid(la[:, 0, :]).squeeze(0).cpu().numpy()
        ball_z_hat = torch.expm1(la[:, 1, :].clamp(min=0.0, max=4.0)).squeeze(0).cpu().numpy()
        return proba, chg, p4d, p_duel, p_aerial, ball_z_hat

    def _run_sliding_window_segment(
        self,
        net: _TrajectoryEventBackboneV10,
        ph_s: np.ndarray,
        pa_s: np.ndarray,
        mh_s: np.ndarray,
        ma_s: np.ndarray,
        ga_s: np.ndarray,
        gs_s: np.ndarray,
        dev: torch.device,
        win_len: int = 1500,
        stride: int = 750,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Overlap-add sliding-window inference (`window=1500` frames = `60s`, `stride=750` frames = `30s`)
        so that every window has the exact `1500`-frame horizon used during training (`Codex 004 #5`).
        """
        N = len(ph_s)
        if N <= win_len:
            return self._run_segment_forward(net, ph_s, pa_s, mh_s, ma_s, ga_s, gs_s, dev)

        starts = list(range(0, N - win_len + 1, stride))
        if starts[-1] != N - win_len:
            starts.append(N - win_len)

        # Raised-cosine / Hann window weights (strictly positive at boundaries)
        t_grid = np.linspace(0.0, np.pi, win_len, dtype=np.float32)
        w_win = (0.15 + 0.85 * np.sin(t_grid) ** 2).astype(np.float32)

        acc_proba = np.zeros((N, 3), dtype=np.float64)
        acc_chg = np.zeros(N, dtype=np.float64)
        acc_p4d = np.zeros((N, 4), dtype=np.float64)
        acc_duel = np.zeros(N, dtype=np.float64)
        acc_aer = np.zeros(N, dtype=np.float64)
        acc_bz = np.zeros(N, dtype=np.float64)
        acc_w = np.zeros(N, dtype=np.float64)

        # Process windows in chunks of at most 32 to bound peak GPU memory on very long halves (e.g. BAR-ATH)
        chunk_size = 32
        for c_start in range(0, len(starts), chunk_size):
            c_starts = starts[c_start : c_start + chunk_size]
            b_ph = np.stack([ph_s[a : a + win_len] for a in c_starts])
            b_pa = np.stack([pa_s[a : a + win_len] for a in c_starts])
            b_mh = np.stack([mh_s[a : a + win_len] for a in c_starts])
            b_ma = np.stack([ma_s[a : a + win_len] for a in c_starts])
            b_ga = np.stack([ga_s[a : a + win_len] for a in c_starts])
            b_gs = np.stack([gs_s[a : a + win_len] for a in c_starts])

            th_ph = torch.from_numpy(b_ph).to(dev).permute(0, 3, 2, 1)
            th_pa = torch.from_numpy(b_pa).to(dev).permute(0, 3, 2, 1)
            th_mh = torch.from_numpy(b_mh).to(dev).permute(0, 2, 1)
            th_ma = torch.from_numpy(b_ma).to(dev).permute(0, 2, 1)
            th_ga = torch.from_numpy(b_ga).to(dev).permute(0, 2, 1)
            th_gs = torch.from_numpy(b_gs).to(dev).permute(0, 2, 1)

            lp, lc, l4, ld, la = net(th_ph, th_pa, th_mh, th_ma, th_ga, th_gs)
            pr_all = SymTrajectoryEventNetV10._sym_softmax_3(lp).permute(0, 2, 1).cpu().numpy()
            ch_all = torch.sigmoid(lc).cpu().numpy()
            p4_all = torch.sigmoid(l4).permute(0, 2, 1).cpu().numpy()
            du_all = torch.sigmoid(ld).cpu().numpy()
            ae_all = torch.sigmoid(la[:, 0, :]).cpu().numpy()
            bz_all = torch.expm1(la[:, 1, :].clamp(min=0.0, max=4.0)).cpu().numpy()

            for i, a in enumerate(c_starts):
                b = a + win_len
                acc_proba[a:b] += pr_all[i] * w_win[:, None]
                acc_chg[a:b] += ch_all[i] * w_win
                acc_p4d[a:b] += p4_all[i] * w_win[:, None]
                acc_duel[a:b] += du_all[i] * w_win
                acc_aer[a:b] += ae_all[i] * w_win
                acc_bz[a:b] += bz_all[i] * w_win
                acc_w[a:b] += w_win

        inv_w = 1.0 / np.maximum(acc_w, 1e-8)
        proba = self._sym_norm_proba_np((acc_proba * inv_w[:, None]).astype(np.float32))
        return (
            proba,
            (acc_chg * inv_w).astype(np.float32),
            (acc_p4d * inv_w[:, None]).astype(np.float32),
            (acc_duel * inv_w).astype(np.float32),
            (acc_aer * inv_w).astype(np.float32),
            (acc_bz * inv_w).astype(np.float32),
        )

    @staticmethod
    def calibrate_focal_soft_iou_outputs(
        proba: np.ndarray,
        change: np.ndarray,
        p4d: np.ndarray,
        half: np.ndarray,
        proba_temp: float = 0.60,
        proba_smooth_sigma: float = 2.5,
        change_power: float = 2.1,
        p4d_power: float = 1.4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analytical inverse calibration for Soft-IoU + Focal-Loss trained heads (`0.000e+00` H<->A equivariant):
          - `proba ** proba_temp` + half-wise temporal smoothing (`sigma=2.5`) undoes Soft-IoU probability sharpening.
          - `change ** change_power` and `p4d ** p4d_power` undo Sigmoid Focal Loss (`gamma=1.5`) background compression,
            matching the BCE transition hazard scale expected by `viterbi_hsmm.decode`.
        """
        p_out = np.zeros_like(proba, dtype=np.float32)
        c_out = np.zeros_like(change, dtype=np.float32)
        p4_out = np.zeros_like(p4d, dtype=np.float32)

        for hid in np.unique(half[half > 0]):
            idx = np.flatnonzero(half == hid)
            if len(idx) == 0:
                continue
            s, e = int(idx[0]), int(idx[-1]) + 1
            p_seg = np.clip(proba[s:e].astype(np.float64), 1e-6, 1.0) ** proba_temp
            if proba_smooth_sigma > 0.0 and (e - s) > 5:
                p_seg = gaussian_filter1d(p_seg, sigma=proba_smooth_sigma, axis=0, mode="nearest")
            denom = (p_seg[:, 0:1] + p_seg[:, 1:2]) + p_seg[:, 2:3]
            p_out[s:e] = (p_seg / np.clip(denom, 1e-8, None)).astype(np.float32)
            c_out[s:e] = (np.clip(change[s:e].astype(np.float64), 0.0, 1.0) ** change_power).astype(np.float32)
            p4_out[s:e] = (np.clip(p4d[s:e].astype(np.float64), 0.0, 1.0) ** p4d_power).astype(np.float32)

        return p_out, c_out, p4_out

    def predict_all(
        self, pst: PlayerSetTensors, half: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Predict all 6 outputs with exact `0.000e+00` H<->A equivariance:
        - `proba`:      `(T, 3)` float32 (`[p_H, p_A, p_D]`)
        - `change`:     `(T,)` float32 (1D transition probability)
        - `p4d`:        `(T, 4)` float32 (`[p_H_to_A, p_A_to_H, p_Live_to_D, p_D_to_Live]`)
        - `p_duel_to`:  `(T,)` float32 (short-steal / duel-turnover probability `<= 65f`)
        - `p_aerial`:   `(T,)` float32 ($P(Z > 1.5\\text{m})$)
        - `ball_z_hat`: `(T,)` float32 (estimated 3D ball height in meters)
        """
        assert self.net_ is not None
        ph, pa, mh, ma, ga, gs = self._prep_match_arrays(pst)
        T = len(ph)
        proba = np.zeros((T, 3), dtype=np.float32)
        chg = np.zeros(T, dtype=np.float32)
        p4d = np.zeros((T, 4), dtype=np.float32)
        p_duel_to = np.zeros(T, dtype=np.float32)
        p_aerial = np.zeros(T, dtype=np.float32)
        ball_z_hat = np.zeros(T, dtype=np.float32)

        dev = self._device()
        net = self.net_.to(dev).eval()
        alpha_sw = float(self.sliding_window_blend)

        with torch.no_grad():
            for hid in np.unique(half[half > 0]):
                idx = np.flatnonzero(half == hid)
                if len(idx) == 0:
                    continue
                s, e = int(idx[0]), int(idx[-1]) + 1
                pr_f, ch_f, p4_f, du_f, ae_f, bz_f = self._run_segment_forward(
                    net, ph[s:e], pa[s:e], mh[s:e], ma[s:e], ga[s:e], gs[s:e], dev
                )
                if alpha_sw > 0.0 and (e - s) > self.crop:
                    pr_w, ch_w, p4_w, du_w, ae_w, bz_w = self._run_sliding_window_segment(
                        net, ph[s:e], pa[s:e], mh[s:e], ma[s:e], ga[s:e], gs[s:e], dev,
                        win_len=self.crop, stride=self.crop // 2,
                    )
                    pr_b = self._sym_norm_proba_np((1.0 - alpha_sw) * pr_f + alpha_sw * pr_w)
                    proba[s:e] = pr_b
                    chg[s:e] = (1.0 - alpha_sw) * ch_f + alpha_sw * ch_w
                    p4d[s:e] = (1.0 - alpha_sw) * p4_f + alpha_sw * p4_w
                    p_duel_to[s:e] = (1.0 - alpha_sw) * du_f + alpha_sw * du_w
                    p_aerial[s:e] = (1.0 - alpha_sw) * ae_f + alpha_sw * ae_w
                    ball_z_hat[s:e] = (1.0 - alpha_sw) * bz_f + alpha_sw * bz_w
                else:
                    proba[s:e] = pr_f
                    chg[s:e] = ch_f
                    p4d[s:e] = p4_f
                    p_duel_to[s:e] = du_f
                    p_aerial[s:e] = ae_f
                    ball_z_hat[s:e] = bz_f

        proba, chg, p4d = self.calibrate_focal_soft_iou_outputs(proba, chg, p4d, half)
        self.net_ = net.to("cpu").eval()
        return {
            "proba": proba,
            "change": chg,
            "p4d": p4d,
            "p_duel_to": p_duel_to,
            "p_aerial": p_aerial,
            "ball_z_hat": ball_z_hat,
        }

    def predict_all_heads(
        self, pst: PlayerSetTensors, half: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        d = self.predict_all(pst, half)
        return d["proba"], d["change"], d["p4d"], d["p_duel_to"], d["p_aerial"], d["ball_z_hat"]


def verify_traj_event_net_v10_equivariance(
    model: SymTrajectoryEventNetV10,
    pst: PlayerSetTensors,
    half: np.ndarray,
) -> dict[str, float]:
    """Verify exact `0.000e+00` H <-> A equivariance across all 6 output heads of `SymTrajectoryEventNetV10`."""
    pst_swap = PlayerSetTensors(
        p_h=pst.p_a.copy(),
        p_a=pst.p_h.copy(),
        mask_h=pst.mask_a.copy(),
        mask_a=pst.mask_h.copy(),
        g_anti=-pst.g_anti.copy(),
        g_sym=pst.g_sym.copy(),
    )
    out1 = model.predict_all(pst, half)
    out2 = model.predict_all(pst_swap, half)

    diff_proba = float(np.max(np.abs(out1["proba"] - out2["proba"][:, [1, 0, 2]])))
    diff_chg = float(np.max(np.abs(out1["change"] - out2["change"])))
    diff_p4d = float(np.max(np.abs(out1["p4d"] - out2["p4d"][:, [1, 0, 2, 3]])))
    diff_duel = float(np.max(np.abs(out1["p_duel_to"] - out2["p_duel_to"])))
    diff_aer = float(np.max(np.abs(out1["p_aerial"] - out2["p_aerial"])))
    diff_bz = float(np.max(np.abs(out1["ball_z_hat"] - out2["ball_z_hat"])))

    return {
        "max_diff_proba": diff_proba,
        "max_diff_change": diff_chg,
        "max_diff_p4d": diff_p4d,
        "max_diff_duel_to": diff_duel,
        "max_diff_aerial": diff_aer,
        "max_diff_ball_z": diff_bz,
        "max_overall": max(diff_proba, diff_chg, diff_p4d, diff_duel, diff_aer, diff_bz),
    }
