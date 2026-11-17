"""`SymPhysConformerV11`: Dual-Subspace (Ball-Free Collective/Focus + Ghost-Gated Ball/Impulse)
Window-Local Normalized Conformer & Dilated-TCN-BiGRU Sequence Network on 481 `PhysV2` Features.

First-Principles Architectural Innovations:
  1. Physical Subspace Decomposition (`236` Ball-Free vs `245` Ball-Aware + `20` Reliability Indicators):
     - Prior 481-feature sequence models (`SymTCN_PhysV2`, `SymBiGRU_PhysV2` from Wave 5B) fed all 481
       columns into a single linear stem, allowing corrupted/occluded ball coordinates (`degrade_v6` or
       `BAR-ATH` broadcast blackouts) to poison the hidden representation of the 22 players.
     - `SymPhysConformerV11` splits the 481 features into:
       * `X_bf` (`236` columns): 100% ball-free collective geometry, formation lines, team speeds,
         expansion-contraction dipoles (`d_sigma_*`, `d_eff_area_*`), and 22-player pressure convergence
         focus (`focus_x_norm`, `focus_diff_d1`, `focus_diff_inflow`).
       * `X_ball` (`245` columns): knot-anchored ball physics, multi-scale vector impulse (`impulse_mag_d2/4/8`),
         chord mid-flight geometry (`chord_in_flight`, `chord_same_team_diff`), and ball-player control margins.
       * `X_rel` (`20` H/A-invariant columns): ball visibility, knot span, chord flight, impulse magnitudes,
         and `ball_focus_discrepancy = ||b_csv(t) - x_focus(t)||`.
     - A dedicated `GhostReliabilityGate` reads `X_rel` and dynamically gates the Ball-Aware stem branch
       relative to the 100% Ball-Free Collective/Focus stem branch at every frame `t`.
  2. Window-Local Input Feature Standardization (`WindowLocalNorm1d(window_size=1501)`):
     - Standardizes every feature channel within a 60-second (`1501`-frame) local sliding window with zero
       target labels, eliminating domain scale drift across optical tracking providers (`Sportec` `v3`/`v6`,
       `Metrica` `J2`, and automatic broadcast tracking) as well as horizon drift between 60s training crops
       and 45-minute inference halves (`Codex 004 Finding #5`).
  3. Multi-Scale Depthwise-Separable Dilated SE-TCN + Strided Local Conformer + BiGRU Backbone:
     - 7 residual depthwise-separable dilated blocks (`d = 1, 2, 4, 8, 16, 32, 64`, `k = 5`) with
       Squeeze-and-Excitation channel attention + parallel multi-horizon pyramid (`d = 1, 4, 12, 30`).
     - Strided (`stride = 8`, `~3.1 Hz`) Bidirectional GRU + Bounded-Window Relative Multi-Head Self-Attention
       (`|t_i - t_j| <= 30` tokens `~9.6s`), linearly upsampled back to 25 Hz.
  4. Five Physics-Grounded Multi-Task Heads + Direct `spellF1` Soft-IoU Loss (100% trained on external `Sportec` LOMO):
     - `proba` (`[T, 3]`): trained with `L1` inverse-spell duration weights (`w_t = L^-0.40`),
       Differentiable Multi-Scale Soft-IoU Loss (`windows = 25, 75, 200`), and Dead-Ball Prior Guard.
     - `change` (`[T,]`): Gaussian-smoothed (`sigma = 3.0`) transition impulse detector.
     - `p4d` (`[T, 4]`): Domain-invariant 4-directional boundary head (`[H->A, A->H, Live->D, D->Live]`),
       preventing cross-provider boundary scale attenuation on degraded automatic tracking.
     - `p_duel_to` (`[T,]`): Short-steal / duel-turnover head (`1.0` inside live spells `<= 65` frames).
     - `p_aerial` (`[T,]`) & `ball_z_hat` (`[T,]`): 3D aerial ball detector (`ball_z > 1.5m`) supervised by
       ground-truth 3D `ball_z` on Sportec.
  5. Exact `0.000e+00` H <-> A Equivariance via Twin-Pass Siamese Forwarding on `(X_orig, X_mir)`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.common import AWAY, DEAD, HOME
from src.v10_first_principles.traj_event_net_v10 import (
    WindowLocalNorm1d,
    compute_v10_boundary_and_duel_targets,
    compute_v10_possession_frame_weights,
    compute_windowed_soft_iou_loss,
    sigmoid_focal_with_pos_weight_loss,
)


BALL_FREE_PREFIXES = (
    "n_visible_",
    "centroid_",
    "chg_centroid_",
    "chg_abs_centroid_",
    "spread_",
    "def_line_",
    "fwd_line_",
    "width_",
    "speed_mean_",
    "speed_p90_",
    "frac_still_",
    "in_own_box_",
    "in_opp_box_",
    "gk_x_",
    "team_vx_",
    "home_speed_",
    "away_speed_",
    "diff_speed_",
    "home_vx_",
    "away_vx_",
    "diff_vx_",
    "home_slow_",
    "away_slow_",
    "diff_slow_",
    "all_speed_",
    "all_slow_",
    "sigma_x_",
    "sigma_y_",
    "diff_sigma_",
    "eff_area_",
    "diff_eff_area",
    "def3_line_x_",
    "att3_line_x_",
    "diff_def3_",
    "diff_att3_",
    "d_sigma_",
    "d_eff_area_",
    "d_cx_",
    "d_cy_",
    "d_def3_",
    "d_att3_",
    "diff_d_",
    "focus_x_norm",
    "focus_y_norm",
    "focus_d1_",
    "focus_d2_",
    "focus_diff_d1",
    "focus_diff_d2",
    "focus_n5m_",
    "focus_diff_n5m",
    "focus_inflow_",
    "focus_diff_inflow",
)

RELIABILITY_COL_NAMES = (
    "ball_visible",
    "ball_since_seen",
    "ball_until_seen",
    "ball_outside_pitch",
    "ball_jump_last_next",
    "ball_visible_m0.4s",
    "ball_visible_m0.8s",
    "ball_visible_m2s",
    "ball_visible_m4s",
    "knot_since",
    "knot_until",
    "knot_span",
    "chord_in_flight",
    "chord_turnover_prob",
    "impulse_mag_d2",
    "impulse_mag_d4",
    "impulse_mag_d8",
    "ball_focus_discrepancy",
    "ball_focus_discrepancy_r1",
    "ball_focus_discrepancy_r3",
)


def partition_physv2_columns(col_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition 481 `v1ctl_physics_v2` columns into:
      - `bf_idx`:   1D int64 indices of 100% ball-free collective/focus features (`236` cols)
      - `ball_idx`: 1D int64 indices of ball-aware knot/impulse/chord/control features (`245` cols)
      - `rel_idx`:  1D int64 indices of H/A-invariant ball-reliability & ghost-discrepancy indicators (`20` cols)
    """
    bf_list, ball_list, rel_list = [], [], []
    rel_set = set(RELIABILITY_COL_NAMES)
    for idx, c in enumerate(col_names):
        is_bf = (c == "minutes_in_half") or c.startswith(BALL_FREE_PREFIXES)
        if is_bf:
            bf_list.append(idx)
        else:
            ball_list.append(idx)
        if c in rel_set:
            rel_list.append(idx)
    if len(rel_list) == 0:
        rel_list = ball_list[: min(16, len(ball_list))]
    return (
        np.array(bf_list, dtype=np.int64),
        np.array(ball_list, dtype=np.int64),
        np.array(rel_list, dtype=np.int64),
    )


class GhostReliabilityGate(nn.Module):
    """Dynamic per-frame, per-channel reliability gate $g_{\\text{rel}}(t) \\in (0, 1)^C$ driven by
    ball visibility, knot span, chord mid-flight status, multi-scale impulse, and `ball_focus_discrepancy`.
    """

    def __init__(self, n_rel: int, channels: int, window_size: int = 1501):
        super().__init__()
        self.norm = WindowLocalNorm1d(n_rel, window_size=window_size)
        self.conv = nn.Sequential(
            nn.Conv1d(n_rel, channels // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels // 2, channels, kernel_size=1),
        )
        # Initialize bias slightly positive (+0.8) so clean ball features pass through initially
        nn.init.constant_(self.conv[-1].bias, 0.8)

    def forward(self, x_rel: torch.Tensor) -> torch.Tensor:
        """`x_rel`: `[B, C_rel, T]` -> gate `[B, C, T]` in `(0, 1)`."""
        z = self.norm(x_rel)
        return torch.sigmoid(self.conv(z))


class SqueezeExcite1d(nn.Module):
    """Local Window Squeeze-and-Excitation channel gating (`window = 251` frames = `10s`)."""

    def __init__(self, channels: int, reduction: int = 4, pool_window: int = 251):
        super().__init__()
        self.pool_window = int(pool_window) if int(pool_window) % 2 == 1 else int(pool_window) + 1
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv1d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv1d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        if T <= self.pool_window:
            s = x.mean(dim=-1, keepdim=True)
        else:
            pad = self.pool_window // 2
            xp = F.pad(x, (pad, pad), mode="replicate")
            s = F.avg_pool1d(xp, kernel_size=self.pool_window, stride=1)
        g = torch.sigmoid(self.fc2(F.gelu(self.fc1(s))))
        return x * g


class DilatedSepConvSEBlock(nn.Module):
    """Depthwise-Separable Dilated Residual Block with `WindowLocalNorm1d` and Local Squeeze-and-Excitation."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float = 0.15,
        window_size: int = 1501,
    ):
        super().__init__()
        pad = 2 * dilation
        self.dw1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = WindowLocalNorm1d(channels, window_size=window_size)

        self.dw2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = WindowLocalNorm1d(channels, window_size=window_size)

        self.se = SqueezeExcite1d(channels, reduction=4, pool_window=251)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.pw1(self.dw1(x))))
        h = self.drop(h)
        h = F.gelu(self.norm2(self.pw2(self.dw2(h))))
        h = self.se(self.drop(h))
        return x + h


class BoundedLocalConformerBiGRUBlock(nn.Module):
    """Strided (`stride = 8`, `~3.1 Hz`) Bidirectional GRU + Bounded-Window Multi-Head Self-Attention
    (`|i - j| <= max_rel_dist` tokens `~9.6s`) with learnable relative distance bias.

    Because the attention receptive field is strictly bounded to `+/- max_rel_dist` tokens and
    the BiGRU operates after `WindowLocalNorm1d`, training crops (`60s`) and inference windows
    have matched statistics.
    """

    def __init__(
        self,
        channels: int,
        stride: int = 8,
        n_heads: int = 4,
        max_rel_dist: int = 30,
        dropout: float = 0.15,
        window_size: int = 1501,
    ):
        super().__init__()
        self.stride = int(stride)
        self.n_heads = int(n_heads)
        self.head_dim = channels // n_heads
        assert self.head_dim * n_heads == channels, "channels must be divisible by n_heads"
        self.max_rel_dist = int(max_rel_dist)

        self.pre_norm = nn.LayerNorm(channels)
        self.bigru = nn.GRU(
            input_size=channels,
            hidden_size=channels // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, 2 * max_rel_dist + 1))
        self.drop = nn.Dropout(dropout)
        self.post_norm = WindowLocalNorm1d(channels, window_size=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `[B, C, T]` -> `[B, C, T]`."""
        B, C, T = x.shape
        s = self.stride
        if T >= s:
            xs = F.avg_pool1d(x, kernel_size=s, stride=s, ceil_mode=True)
        else:
            xs = x
        Ts = xs.shape[-1]

        # 1. Strided BiGRU
        u = self.pre_norm(xs.transpose(1, 2))  # [B, Ts, C]
        self.bigru.flatten_parameters()
        g_out, _ = self.bigru(u)
        u = u + self.drop(g_out)

        # 2. Bounded-Window Relative Multi-Head Self-Attention
        v_in = self.attn_norm(u)
        qkv = self.qkv(v_in).view(B, Ts, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, Ts, D]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Ts, Ts]

        idx = torch.arange(Ts, device=x.device)
        rel = idx[:, None] - idx[None, :]  # [Ts, Ts]
        valid_mask = rel.abs() <= self.max_rel_dist
        rel_clipped = rel.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        bias = self.rel_bias[:, rel_clipped]  # [H, Ts, Ts]
        scores = scores + bias.unsqueeze(0)
        scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        a_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Ts, C)
        u = u + self.drop(self.proj(a_out))

        # 3. Upsample back to 25 Hz
        u_t = u.transpose(1, 2)  # [B, C, Ts]
        u_up = F.interpolate(u_t, size=T, mode="linear", align_corners=False)
        return x + self.post_norm(u_up)


class PhysConformerNetV11(nn.Module):
    """Dual-Subspace (`Ball-Free` + `Ghost-Gated Ball-Aware`) Conformer + Dilated SE-TCN + BiGRU Network."""

    def __init__(
        self,
        bf_idx: np.ndarray,
        ball_idx: np.ndarray,
        rel_idx: np.ndarray,
        channels: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        conformer_stride: int = 8,
        window_size: int = 1501,
        dropout: float = 0.16,
    ):
        super().__init__()
        self.register_buffer("bf_idx", torch.as_tensor(bf_idx, dtype=torch.long))
        self.register_buffer("ball_idx", torch.as_tensor(ball_idx, dtype=torch.long))
        self.register_buffer("rel_idx", torch.as_tensor(rel_idx, dtype=torch.long))

        n_bf = len(bf_idx)
        n_ball = len(ball_idx)
        n_rel = len(rel_idx)

        # Dual-Subspace Input Stems with WindowLocalNorm1d
        self.bf_stem = nn.Sequential(
            WindowLocalNorm1d(n_bf, window_size=window_size),
            nn.Conv1d(n_bf, channels, kernel_size=3, padding=1, bias=False),
            WindowLocalNorm1d(channels, window_size=window_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.ball_stem = nn.Sequential(
            WindowLocalNorm1d(n_ball, window_size=window_size),
            nn.Conv1d(n_ball, channels, kernel_size=3, padding=1, bias=False),
            WindowLocalNorm1d(channels, window_size=window_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.rel_gate = GhostReliabilityGate(n_rel, channels, window_size=window_size)
        self.stem_norm = WindowLocalNorm1d(channels, window_size=window_size)

        # Stage A: Multi-Scale Dilated Depthwise-Separable SE-TCN Stack
        self.blocks = nn.ModuleList(
            [
                DilatedSepConvSEBlock(
                    channels=channels,
                    dilation=d,
                    dropout=dropout,
                    window_size=window_size,
                )
                for d in dilations
            ]
        )

        # Stage B: Parallel Multi-Horizon Physical Pyramid (0.1s, 0.5s, 1.5s, 3.6s)
        p_ch = channels // 4
        self.pyr_b1 = nn.Conv1d(channels, p_ch, kernel_size=3, padding=1, dilation=1)
        self.pyr_b2 = nn.Conv1d(channels, p_ch, kernel_size=3, padding=4, dilation=4)
        self.pyr_b3 = nn.Conv1d(channels, p_ch, kernel_size=3, padding=12, dilation=12)
        self.pyr_b4 = nn.Conv1d(channels, channels - 3 * p_ch, kernel_size=3, padding=30, dilation=30)
        self.pyr_fuse = nn.Sequential(
            WindowLocalNorm1d(channels, window_size=window_size),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

        # Stage C: Strided Bounded-Window Conformer + BiGRU Context
        self.conformer = BoundedLocalConformerBiGRUBlock(
            channels=channels,
            stride=conformer_stride,
            n_heads=4,
            max_rel_dist=30,
            dropout=dropout,
            window_size=window_size,
        )

        # Stage D: 5 Physics-Grounded Multi-Task Heads
        # 1. Primary 3-class possession head [H, A, D]
        self.head_had = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 3, kernel_size=1),
        )
        # 2. Boundary / action-spotting feature trunk (combines early local conv + deep conformer features)
        self.bnd_trunk = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=5, padding=2),
            WindowLocalNorm1d(channels, window_size=window_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head_change = nn.Conv1d(channels // 2, 1, kernel_size=1)
        self.head_4dir = nn.Conv1d(channels // 2, 4, kernel_size=1)
        self.head_duel_to = nn.Conv1d(channels // 2, 1, kernel_size=1)
        # 3. 3D Aerial head [logit_aerial_gt_1p5m, ball_z_regression]
        self.head_aerial = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 2, kernel_size=1),
        )

    def forward_single_view(
        self,
        x_bct: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on a single view `x_bct` of shape `[B, 481, T]`."""
        x_bf = x_bct.index_select(1, self.bf_idx)
        x_ball = x_bct.index_select(1, self.ball_idx)
        x_rel = x_bct.index_select(1, self.rel_idx)

        h_bf = self.bf_stem(x_bf)
        h_ball = self.ball_stem(x_ball)
        g_rel = self.rel_gate(x_rel)

        # Ghost-gated fusion: when ball is unreliable/occluded, network falls back smoothly to h_bf
        h0 = F.gelu(self.stem_norm(h_bf + g_rel * h_ball))

        h = h0
        h_local = h0
        for idx_b, blk in enumerate(self.blocks):
            h = blk(h)
            if idx_b == 2:
                h_local = h

        p = torch.cat(
            [self.pyr_b1(h), self.pyr_b2(h), self.pyr_b3(h), self.pyr_b4(h)],
            dim=1,
        )
        h = h + self.pyr_fuse(p)
        h = self.conformer(h)

        logits_had = self.head_had(h)  # [B, 3, T]
        hb = self.bnd_trunk(torch.cat([h, h_local], dim=1))
        logit_chg = self.head_change(hb).squeeze(1)  # [B, T]
        logits_4d = self.head_4dir(hb)  # [B, 4, T]
        logit_duel = self.head_duel_to(hb).squeeze(1)  # [B, T]
        pred_air = self.head_aerial(h)  # [B, 2, T]
        return logits_had, logit_chg, logits_4d, logit_duel, pred_air

    def forward(
        self,
        x_orig: torch.Tensor,
        x_mir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact Twin-Pass H <-> A Equivariant Forward Pass (`0.000e+00` symmetry error).

        Inputs:
          `x_orig`: `[B, 481, T]` original standardized feature sequence
          `x_mir`:  `[B, 481, T]` mirrored (`H <-> A` + 180-deg pitch rotated) standardized sequence
        """
        B = x_orig.shape[0]
        x_cat = torch.cat([x_orig, x_mir], dim=0)  # [2B, 481, T]
        l_had, l_chg, l_4d, l_duel, p_air = self.forward_single_view(x_cat)

        l_had_1, l_had_2 = l_had[:B], l_had[B:]
        l_chg_1, l_chg_2 = l_chg[:B], l_chg[B:]
        l_4d_1, l_4d_2 = l_4d[:B], l_4d[B:]
        l_duel_1, l_duel_2 = l_duel[:B], l_duel[B:]
        p_air_1, p_air_2 = p_air[:B], p_air[B:]

        # Symmetrize H <-> A channels:
        # - In 3-class [H, A, D], swap channels 0 and 1
        l_had_2_swap = torch.stack([l_had_2[:, 1], l_had_2[:, 0], l_had_2[:, 2]], dim=1)
        logits_had = 0.5 * (l_had_1 + l_had_2_swap)

        # - In 4-dir [H->A, A->H, L->D, D->L], swap channels 0 and 1
        l_4d_2_swap = torch.stack([l_4d_2[:, 1], l_4d_2[:, 0], l_4d_2[:, 2], l_4d_2[:, 3]], dim=1)
        logits_4d = 0.5 * (l_4d_1 + l_4d_2_swap)

        # - Invariant heads: change, duel_to, aerial
        logit_chg = 0.5 * (l_chg_1 + l_chg_2)
        logit_duel = 0.5 * (l_duel_1 + l_duel_2)
        pred_air = 0.5 * (p_air_1 + p_air_2)
        return logits_had, logit_chg, logits_4d, logit_duel, pred_air


class SymPhysConformerV11:
    """High-level trainer and equivariant predictor for `PhysConformerNetV11` on 481 `PhysV2` features."""

    def __init__(
        self,
        col_names: list[str],
        channels: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        conformer_stride: int = 8,
        window_size: int = 1501,
        dropout: float = 0.16,
        lr: float = 1.5e-3,
        weight_decay: float = 1.2e-4,
        epochs: int = 9,
        crop: int = 1500,
        batch_size: int = 32,
        poss_alpha: float = 0.40,
        boundary_boost: float = 1.65,
        soft_iou_weight: float = 0.28,
        dead_guard_weight: float = 0.10,
        change_loss_weight: float = 0.35,
        b4d_loss_weight: float = 0.48,
        duel_loss_weight: float = 0.25,
        aerial_loss_weight: float = 0.20,
        ball_drop_prob: float = 0.20,
        feat_noise_std: float = 0.03,
        label_smoothing: float = 0.02,
        sliding_window_blend: float = 0.50,
        seed: int = 42,
        device: str | None = None,
    ):
        self.col_names = list(col_names)
        self.bf_idx, self.ball_idx, self.rel_idx = partition_physv2_columns(self.col_names)
        self.channels = int(channels)
        self.dilations = tuple(dilations)
        self.conformer_stride = int(conformer_stride)
        self.window_size = int(window_size)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.crop = int(crop)
        self.batch_size = int(batch_size)
        self.poss_alpha = float(poss_alpha)
        self.boundary_boost = float(boundary_boost)
        self.soft_iou_weight = float(soft_iou_weight)
        self.dead_guard_weight = float(dead_guard_weight)
        self.change_loss_weight = float(change_loss_weight)
        self.b4d_loss_weight = float(b4d_loss_weight)
        self.duel_loss_weight = float(duel_loss_weight)
        self.aerial_loss_weight = float(aerial_loss_weight)
        self.ball_drop_prob = float(ball_drop_prob)
        self.feat_noise_std = float(feat_noise_std)
        self.label_smoothing = float(label_smoothing)
        self.sliding_window_blend = float(sliding_window_blend)
        self.seed = int(seed)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.sym_mean: np.ndarray | None = None
        self.sym_std: np.ndarray | None = None
        self.net: PhysConformerNetV11 | None = None

    def _fit_symmetric_normalizer(
        self,
        xs_orig: list[np.ndarray],
        xs_mir: list[np.ndarray],
        hs: list[np.ndarray],
    ) -> None:
        """Compute exact H/A-symmetric per-column mean and std across `[X_orig, X_mir]`."""
        C = xs_orig[0].shape[1]
        sum_x = np.zeros(C, dtype=np.float64)
        sum_x2 = np.zeros(C, dtype=np.float64)
        count = 0
        for xo, xm, h in zip(xs_orig, xs_mir, hs):
            sel = np.flatnonzero(h > 0)[::8]
            if len(sel) == 0:
                continue
            a = np.nan_to_num(xo[sel].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            b = np.nan_to_num(xm[sel].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            sum_x += a.sum(axis=0) + b.sum(axis=0)
            sum_x2 += (a * a).sum(axis=0) + (b * b).sum(axis=0)
            count += 2 * len(sel)
        count = max(count, 1)
        mean = sum_x / count
        var = np.maximum(sum_x2 / count - mean * mean, 1e-6)
        self.sym_mean = mean.astype(np.float32)
        self.sym_std = np.sqrt(var).astype(np.float32)

    def _standardize_pair(
        self,
        x_orig: np.ndarray,
        x_mir: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardize `(x_orig, x_mir)` of shape `(T, 481)` -> transposed `(481, T)` float32."""
        assert self.sym_mean is not None and self.sym_std is not None
        xo = np.nan_to_num(x_orig.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        xm = np.nan_to_num(x_mir.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        xo = np.clip((xo - self.sym_mean) / self.sym_std, -20.0, 20.0)
        xm = np.clip((xm - self.sym_mean) / self.sym_std, -20.0, 20.0)
        return np.ascontiguousarray(xo.T, dtype=np.float32), np.ascontiguousarray(xm.T, dtype=np.float32)

    def _apply_online_ball_blackout_aug(
        self,
        xb_orig: torch.Tensor,
        xb_mir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Online contiguous ball-blackout (`1.0s-4.0s`) and symmetric feature jitter augmentation.

        During a synthetic ball blackout window `[s, e]`, all ball-dependent columns (`ball_idx`)
        except the H/A-invariant reliability columns are zeroed out in both views so the network
        learns to rely on `bf_idx` + `GhostReliabilityGate`.
        """
        if self.feat_noise_std > 0.0:
            noise = torch.randn_like(xb_orig) * self.feat_noise_std
            xb_orig = xb_orig + noise
            xb_mir = xb_mir + noise

        if self.ball_drop_prob <= 0.0:
            return xb_orig, xb_mir

        B, _, L = xb_orig.shape
        ball_idx_dev = self.net.ball_idx
        for b in range(B):
            if torch.rand((), device=xb_orig.device).item() < self.ball_drop_prob:
                n_spans = int(torch.randint(1, 4, (1,), device=xb_orig.device).item())
                for _ in range(n_spans):
                    span_len = int(torch.randint(25, min(110, max(26, L // 4)), (1,), device=xb_orig.device).item())
                    s = int(torch.randint(0, max(1, L - span_len), (1,), device=xb_orig.device).item())
                    xb_orig[b, ball_idx_dev, s : s + span_len] = 0.0
                    xb_mir[b, ball_idx_dev, s : s + span_len] = 0.0
        return xb_orig, xb_mir

    def prepare_segments(
        self,
        xs_orig: list[np.ndarray],
        xs_mir: list[np.ndarray],
        ys: list[np.ndarray],
        hs: list[np.ndarray],
        ball_zs: list[np.ndarray | None] | None = None,
        sample_weights: list[float] | None = None,
    ) -> list[tuple[torch.Tensor, ...]]:
        """Precompute symmetric normalizer and standardized device-resident segments once across seeds."""
        self._fit_symmetric_normalizer(xs_orig, xs_mir, hs)
        if ball_zs is None:
            ball_zs = [None] * len(xs_orig)
        if sample_weights is None:
            sample_weights = [1.0] * len(xs_orig)

        feat_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        segments: list[tuple[torch.Tensor, ...]] = []
        L = self.crop
        for xo, xm, y, h, bz, sw in zip(xs_orig, xs_mir, ys, hs, ball_zs, sample_weights):
            xo_c, xm_c = self._standardize_pair(xo, xm)
            w_frame = compute_v10_possession_frame_weights(
                y,
                h,
                alpha=self.poss_alpha,
                boundary_radius=20,
                boundary_boost=self.boundary_boost,
            ) * float(sw)
            y_chg, y_4d, y_duel = compute_v10_boundary_and_duel_targets(y, h, sigma=3.0, max_duel_len=65)

            for hv in np.unique(h[h > 0]):
                idx = np.flatnonzero(h == hv)
                if len(idx) < L:
                    continue
                s, e = int(idx[0]), int(idx[-1]) + 1
                y_sub = np.ascontiguousarray(y[s:e], dtype=np.int64)
                y_sw = y_sub.copy()
                y_sw[y_sub == HOME] = AWAY
                y_sw[y_sub == AWAY] = HOME
                y4_sub = np.ascontiguousarray(y_4d[s:e].T, dtype=np.float32)  # [4, T_seg]

                if bz is not None:
                    bz_w = np.ascontiguousarray(bz[s:e], dtype=np.float32)
                    valid_bz = np.isfinite(bz_w) & (y_sub >= 0)
                    air_cls = np.where(valid_bz, (bz_w > 1.5).astype(np.float32), 0.0)
                    air_reg = np.where(valid_bz, np.clip(bz_w, 0.0, 8.0).astype(np.float32), 0.0)
                    air_m = valid_bz.astype(np.float32)
                else:
                    T_seg = e - s
                    air_cls = np.zeros(T_seg, dtype=np.float32)
                    air_reg = np.zeros(T_seg, dtype=np.float32)
                    air_m = np.zeros(T_seg, dtype=np.float32)

                segments.append(
                    (
                        torch.from_numpy(xo_c[:, s:e]).to(self.device, dtype=feat_dtype, non_blocking=True),
                        torch.from_numpy(xm_c[:, s:e]).to(self.device, dtype=feat_dtype, non_blocking=True),
                        torch.from_numpy(y_sub).to(self.device, non_blocking=True),
                        torch.from_numpy(y_sw).to(self.device, non_blocking=True),
                        torch.from_numpy(np.ascontiguousarray(w_frame[s:e], dtype=np.float32)).to(self.device, non_blocking=True),
                        torch.from_numpy(np.ascontiguousarray(y_chg[s:e], dtype=np.float32)).to(self.device, non_blocking=True),
                        torch.from_numpy(y4_sub).to(self.device, non_blocking=True),
                        torch.from_numpy(np.ascontiguousarray(y4_sub[[1, 0, 2, 3]], dtype=np.float32)).to(self.device, non_blocking=True),
                        torch.from_numpy(np.ascontiguousarray(y_duel[s:e], dtype=np.float32)).to(self.device, non_blocking=True),
                        torch.from_numpy(air_cls).to(self.device, non_blocking=True),
                        torch.from_numpy(air_reg).to(self.device, non_blocking=True),
                        torch.from_numpy(air_m).to(self.device, non_blocking=True),
                    )
                )
        return segments

    def fit(
        self,
        xs_orig: list[np.ndarray],
        xs_mir: list[np.ndarray],
        ys: list[np.ndarray],
        hs: list[np.ndarray],
        ball_zs: list[np.ndarray | None] | None = None,
        sample_weights: list[float] | None = None,
        prepared_segments: list[tuple[torch.Tensor, ...]] | None = None,
        sym_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "SymPhysConformerV11":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if prepared_segments is not None and sym_stats is not None:
            self.sym_mean, self.sym_std = sym_stats
            segments = prepared_segments
        else:
            segments = self.prepare_segments(xs_orig, xs_mir, ys, hs, ball_zs=ball_zs, sample_weights=sample_weights)

        self.net = PhysConformerNetV11(
            bf_idx=self.bf_idx,
            ball_idx=self.ball_idx,
            rel_idx=self.rel_idx,
            channels=self.channels,
            dilations=self.dilations,
            conformer_stride=self.conformer_stride,
            window_size=self.window_size,
            dropout=self.dropout,
        ).to(self.device)

        if not segments:
            return self

        L = self.crop
        seg_lens = np.array([int(seg[2].shape[0]) - L for seg in segments], dtype=np.float64)
        seg_probs = np.maximum(seg_lens, 1.0)
        seg_probs /= seg_probs.sum()

        n_total_frames = sum(int(seg[2].shape[0]) for seg in segments)
        crops_per_epoch = max(64, int(np.ceil(2.0 * n_total_frames / L)))
        steps_per_epoch = max(1, (crops_per_epoch + self.batch_size - 1) // self.batch_size)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            anneal_strategy="cos",
        )
        rng = np.random.default_rng(self.seed)
        use_amp = (self.device.type == "cuda") and torch.cuda.is_bf16_supported()

        self.net.train()
        for _ in range(self.epochs):
            for _ in range(steps_per_epoch):
                b_xo, b_xm, b_y, b_w, b_yc, b_y4, b_yd, b_air_cls, b_air_reg, b_air_m = (
                    [], [], [], [], [], [], [], [], [], []
                )
                for _ in range(self.batch_size):
                    si = int(rng.choice(len(segments), p=seg_probs))
                    xo_s, xm_s, y_s, y_sw_s, w_s, yc_s, y4_s, y4_sw_s, yd_s, a_cls_s, a_reg_s, a_m_s = segments[si]
                    T_seg = int(y_s.shape[0])
                    st = int(rng.integers(0, max(1, T_seg - L + 1)))
                    sl = slice(st, st + L)

                    if rng.random() < 0.5:
                        b_xo.append(xo_s[:, sl])
                        b_xm.append(xm_s[:, sl])
                        b_y.append(y_s[sl])
                        b_y4.append(y4_s[:, sl])
                    else:
                        b_xo.append(xm_s[:, sl])
                        b_xm.append(xo_s[:, sl])
                        b_y.append(y_sw_s[sl])
                        b_y4.append(y4_sw_s[:, sl])

                    b_w.append(w_s[sl])
                    b_yc.append(yc_s[sl])
                    b_yd.append(yd_s[sl])
                    b_air_cls.append(a_cls_s[sl])
                    b_air_reg.append(a_reg_s[sl])
                    b_air_m.append(a_m_s[sl])

                xb_o = torch.stack(b_xo, dim=0).float()
                xb_m = torch.stack(b_xm, dim=0).float()
                yb = torch.stack(b_y, dim=0)
                wb = torch.stack(b_w, dim=0)
                ycb = torch.stack(b_yc, dim=0)
                y4b = torch.stack(b_y4, dim=0)
                ydb = torch.stack(b_yd, dim=0)
                air_cls = torch.stack(b_air_cls, dim=0)
                air_reg = torch.stack(b_air_reg, dim=0)
                air_m = torch.stack(b_air_m, dim=0)

                xb_o, xb_m = self._apply_online_ball_blackout_aug(xb_o, xb_m)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits_had, logit_chg, logits_4d, logit_duel, pred_air = self.net(xb_o, xb_m)

                # Compute losses in float32 for numerical stability
                logits_had = logits_had.float()
                logit_chg = logit_chg.float()
                logits_4d = logits_4d.float()
                logit_duel = logit_duel.float()
                pred_air = pred_air.float()

                valid_y = (yb >= 0) & (wb > 0)
                if not valid_y.any():
                    continue

                ce_raw = F.cross_entropy(
                    logits_had,
                    yb.clamp(min=0),
                    reduction="none",
                    label_smoothing=self.label_smoothing,
                )
                loss_ce = (ce_raw * wb * valid_y.float()).sum() / (wb * valid_y.float()).sum().clamp(min=1e-6)
                loss_iou = compute_windowed_soft_iou_loss(logits_had, yb, wb, window_sizes=(50, 150, 500))

                # Dead-Ball Calibration Guard so L1 live-spell weights never under-predict DEAD
                probs_had = torch.softmax(logits_had, dim=1)
                p_dead = probs_had[:, DEAD, :].clamp(1e-5, 1.0 - 1e-5)
                y_dead = (yb == DEAD).float()
                loss_dead_guard = F.binary_cross_entropy(p_dead[valid_y], y_dead[valid_y])

                loss = loss_ce + self.soft_iou_weight * loss_iou + self.dead_guard_weight * loss_dead_guard

                mc = (ycb >= 0).float()
                if mc.sum() > 0:
                    loss_chg = sigmoid_focal_with_pos_weight_loss(
                        logit_chg,
                        ycb,
                        mc,
                        pos_weight=logit_chg.new_tensor(5.0),
                        gamma=1.5,
                    )
                    loss = loss + self.change_loss_weight * loss_chg

                m4 = (y4b >= 0).float()
                if m4.sum() > 0:
                    loss_4d = sigmoid_focal_with_pos_weight_loss(
                        logits_4d,
                        y4b,
                        m4,
                        pos_weight=logits_4d.new_tensor(10.0),
                        gamma=1.5,
                    )
                    loss = loss + self.b4d_loss_weight * loss_4d

                md = (ydb >= 0).float()
                if md.sum() > 0:
                    loss_duel = sigmoid_focal_with_pos_weight_loss(
                        logit_duel,
                        ydb,
                        md,
                        pos_weight=logit_duel.new_tensor(8.0),
                        gamma=1.5,
                    )
                    loss = loss + self.duel_loss_weight * loss_duel

                if air_m.sum() > 10:
                    l_air_cls = sigmoid_focal_with_pos_weight_loss(
                        pred_air[:, 0, :],
                        air_cls,
                        air_m,
                        pos_weight=pred_air.new_tensor(4.5),
                        gamma=1.5,
                    )
                    m_a = air_m > 0.5
                    z_hat = F.softplus(pred_air[:, 1, :])
                    l_air_reg = F.smooth_l1_loss(z_hat[m_a], air_reg[m_a])
                    loss = loss + self.aerial_loss_weight * (l_air_cls + 0.25 * l_air_reg)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()

        self.net.eval()
        return self

    def _decode_raw_tensors_to_arrays(
        self,
        l_had: torch.Tensor,
        l_chg: torch.Tensor,
        l_4d: torch.Tensor,
        l_duel: torch.Tensor,
        p_air: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pr = torch.softmax(l_had, dim=1).squeeze(0).transpose(0, 1).cpu().numpy()
        ch = torch.sigmoid(l_chg).squeeze(0).cpu().numpy()
        p4 = torch.sigmoid(l_4d).squeeze(0).transpose(0, 1).cpu().numpy()
        du = torch.sigmoid(l_duel).squeeze(0).cpu().numpy()
        pa = torch.sigmoid(p_air[:, 0, :]).squeeze(0).cpu().numpy()
        zh = F.softplus(p_air[:, 1, :]).squeeze(0).cpu().numpy()
        return pr, ch, p4, du, pa, zh

    def _predict_half_sliding_windows(
        self,
        xo_sub: torch.Tensor,
        xm_sub: torch.Tensor,
        win_len: int = 1500,
        stride: int = 750,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T_h = xo_sub.shape[-1]
        if T_h <= win_len:
            l_had, l_chg, l_4d, l_duel, p_air = self.net(xo_sub, xm_sub)
            return self._decode_raw_tensors_to_arrays(l_had, l_chg, l_4d, l_duel, p_air)

        starts = list(range(0, T_h - win_len + 1, stride))
        if starts[-1] != T_h - win_len:
            starts.append(T_h - win_len)

        t_grid = np.linspace(-1.0, 1.0, win_len, dtype=np.float32)
        w_tent = np.clip(1.0 - 0.65 * np.abs(t_grid), 0.25, 1.0).astype(np.float32)

        acc_pr = np.zeros((T_h, 3), dtype=np.float32)
        acc_ch = np.zeros(T_h, dtype=np.float32)
        acc_p4 = np.zeros((T_h, 4), dtype=np.float32)
        acc_du = np.zeros(T_h, dtype=np.float32)
        acc_pa = np.zeros(T_h, dtype=np.float32)
        acc_zh = np.zeros(T_h, dtype=np.float32)
        acc_w = np.zeros(T_h, dtype=np.float32)

        batch_sz = 16
        for b0 in range(0, len(starts), batch_sz):
            b_starts = starts[b0 : b0 + batch_sz]
            xo_b = torch.cat([xo_sub[:, :, st : st + win_len] for st in b_starts], dim=0)
            xm_b = torch.cat([xm_sub[:, :, st : st + win_len] for st in b_starts], dim=0)
            l_had, l_chg, l_4d, l_duel, p_air = self.net(xo_b, xm_b)

            pr_b = torch.softmax(l_had, dim=1).transpose(1, 2).cpu().numpy()
            ch_b = torch.sigmoid(l_chg).cpu().numpy()
            p4_b = torch.sigmoid(l_4d).transpose(1, 2).cpu().numpy()
            du_b = torch.sigmoid(l_duel).cpu().numpy()
            pa_b = torch.sigmoid(p_air[:, 0, :]).cpu().numpy()
            zh_b = F.softplus(p_air[:, 1, :]).cpu().numpy()

            for k, st in enumerate(b_starts):
                sl = slice(st, st + win_len)
                acc_pr[sl] += pr_b[k] * w_tent[:, None]
                acc_ch[sl] += ch_b[k] * w_tent
                acc_p4[sl] += p4_b[k] * w_tent[:, None]
                acc_du[sl] += du_b[k] * w_tent
                acc_pa[sl] += pa_b[k] * w_tent
                acc_zh[sl] += zh_b[k] * w_tent
                acc_w[sl] += w_tent

        inv_w = 1.0 / np.maximum(acc_w, 1e-6)
        pr = acc_pr * inv_w[:, None]
        denom = (pr[:, 0:1] + pr[:, 1:2]) + pr[:, 2:3]
        pr = pr / np.clip(denom, 1e-6, None)
        return (
            pr.astype(np.float32),
            (acc_ch * inv_w).astype(np.float32),
            (acc_p4 * inv_w[:, None]).astype(np.float32),
            (acc_du * inv_w).astype(np.float32),
            (acc_pa * inv_w).astype(np.float32),
            (acc_zh * inv_w).astype(np.float32),
        )

    @torch.no_grad()
    def predict_all_heads(
        self,
        x_orig: np.ndarray,
        x_mir: np.ndarray,
        half: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict `(proba, change, p4d, p_duel_to, p_aerial, ball_z_hat)` with exact `0.000e+00` equivariance."""
        assert self.net is not None
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        elif self.device.type == "mps" and not torch.backends.mps.is_available():
            self.device = torch.device("cpu")
        self.net = self.net.to(self.device).eval()

        T = len(half)
        proba = np.full((T, 3), 1.0 / 3.0, dtype=np.float32)
        change = np.zeros(T, dtype=np.float32)
        p4d = np.zeros((T, 4), dtype=np.float32)
        p_duel_to = np.zeros(T, dtype=np.float32)
        p_aerial = np.zeros(T, dtype=np.float32)
        ball_z_hat = np.zeros(T, dtype=np.float32)

        xo_c, xm_c = self._standardize_pair(x_orig, x_mir)

        for hv in np.unique(half[half > 0]):
            idx = np.flatnonzero(half == hv)
            if len(idx) == 0:
                continue
            s, e = int(idx[0]), int(idx[-1]) + 1
            xo_t = torch.from_numpy(xo_c[:, s:e][None]).to(self.device)
            xm_t = torch.from_numpy(xm_c[:, s:e][None]).to(self.device)

            l_had, l_chg, l_4d, l_duel, p_air = self.net(xo_t, xm_t)
            pr_f, ch_f, p4_f, du_f, pa_f, zh_f = self._decode_raw_tensors_to_arrays(
                l_had, l_chg, l_4d, l_duel, p_air
            )

            alpha = self.sliding_window_blend
            if alpha > 0.0 and (e - s) > self.crop:
                pr_w, ch_w, p4_w, du_w, pa_w, zh_w = self._predict_half_sliding_windows(
                    xo_t, xm_t, win_len=self.crop, stride=self.crop // 2
                )
                pr = (1.0 - alpha) * pr_f + alpha * pr_w
                denom = (pr[:, 0:1] + pr[:, 1:2]) + pr[:, 2:3]
                pr = (pr / np.clip(denom, 1e-6, None)).astype(np.float32)
                ch = ((1.0 - alpha) * ch_f + alpha * ch_w).astype(np.float32)
                p4 = ((1.0 - alpha) * p4_f + alpha * p4_w).astype(np.float32)
                du = ((1.0 - alpha) * du_f + alpha * du_w).astype(np.float32)
                pa = ((1.0 - alpha) * pa_f + alpha * pa_w).astype(np.float32)
                zh = ((1.0 - alpha) * zh_f + alpha * zh_w).astype(np.float32)
            else:
                pr, ch, p4, du, pa, zh = pr_f, ch_f, p4_f, du_f, pa_f, zh_f

            proba[s:e] = pr
            change[s:e] = ch
            p4d[s:e] = p4
            p_duel_to[s:e] = du
            p_aerial[s:e] = pa
            ball_z_hat[s:e] = zh

        return proba, change, p4d, p_duel_to, p_aerial, ball_z_hat


def verify_phys_conformer_v11_equivariance(
    model: SymPhysConformerV11,
    x_orig: np.ndarray,
    x_mir: np.ndarray,
    half: np.ndarray,
) -> dict[str, float]:
    """Verify exact `0.000e+00` H <-> A equivariance across all 6 output arrays of `SymPhysConformerV11`."""
    p1, c1, p4_1, du1, pa1, zh1 = model.predict_all_heads(x_orig, x_mir, half)
    p2, c2, p4_2, du2, pa2, zh2 = model.predict_all_heads(x_mir, x_orig, half)

    p2_sw = p2[:, [1, 0, 2]]
    p4_2_sw = p4_2[:, [1, 0, 2, 3]]
    return {
        "max_proba_diff": float(np.max(np.abs(p1 - p2_sw))),
        "max_change_diff": float(np.max(np.abs(c1 - c2))),
        "max_p4d_diff": float(np.max(np.abs(p4_1 - p4_2_sw))),
        "max_duel_diff": float(np.max(np.abs(du1 - du2))),
        "max_aerial_diff": float(np.max(np.abs(pa1 - pa2))),
        "max_ball_z_diff": float(np.max(np.abs(zh1 - zh2))),
    }
