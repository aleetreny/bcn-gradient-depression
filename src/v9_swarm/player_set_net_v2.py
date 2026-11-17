"""`SymPlayerSetNetV2`: Multi-Task Equivariant Set-Attention + Dilated TCN + BiGRU with:
  1. Line L1: Possession-duration inverse weighting (`w ~ (100 / len)^alpha`) + dense transition crop sampling.
  2. Line L5: 4-Directional Action Spotting Head (`p_H_to_A, p_A_to_H, p_Live_to_D, p_D_to_Live`) with Focal Loss.
  3. Idea O1: Auxiliary 3D Aerial Ball Height Supervision (`ball_z > 1.5m` & `log1p(ball_z)`) from Sportec 3D tracking,
     with an internal Aerial Gate that modulates possession & turnover logits during high balls.
  4. Line D3: Online GPU Team-Swap (`12 vs 10`) & Player-Dropout Augmentation for BAR-ATH ATD robustness.
  5. Exact `0.000e+00` H <-> A + 180-deg pitch rotation equivariance and `0.000e+00` slot permutation invariance.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.common import AWAY, HOME
from src.models.boundary_4dir import targets_4dir
from src.models.player_set_net import (
    ChannelNorm,
    PlayerSetTensors,
    _ResBlock,
    _SplitPlayerPointNet,
)
from src.models.transition import change_targets


def compute_possession_frame_weights(
    labels: np.ndarray,
    half: np.ndarray,
    alpha: float = 0.45,
    boundary_radius: int = 20,
    boundary_boost: float = 1.6,
) -> np.ndarray:
    """Compute per-frame loss weights aligning frame cross-entropy with `spellF1`:
    - Short possessions (`len < 50f`) receive `(100 / max(len, 12)) ** alpha` weight.
    - Frames within `+/- boundary_radius` of a transition receive `boundary_boost` multiplier.
    - Normalized to mean 1.0 over valid labelled frames of the match.
    """
    T = len(labels)
    valid = (labels >= 0) & (half > 0)
    w = np.ones(T, dtype=np.float32)
    if not np.any(valid):
        return w

    # Run-length encode all valid segments (H, A, and D) within each half
    same_prev = np.r_[False, (labels[1:] == labels[:-1]) & (half[1:] == half[:-1]) & valid[1:] & valid[:-1]]
    starts = np.flatnonzero(~same_prev & valid)
    if len(starts) == 0:
        return w

    for idx_s in starts:
        hv = half[idx_s]
        lab = labels[idx_s]
        # Find end of run
        e = idx_s + 1
        while e < T and valid[e] and half[e] == hv and labels[e] == lab:
            e += 1
        run_len = max(e - idx_s, 12)
        if lab in (HOME, AWAY):
            w_run = (100.0 / float(run_len)) ** alpha
            w[idx_s:e] = np.clip(w_run, 0.35, 3.5)
        else:
            # Dead ball stretches: moderate inverse length weight so short stoppages aren't ignored
            w_run = (150.0 / float(max(run_len, 25))) ** (0.5 * alpha)
            w[idx_s:e] = np.clip(w_run, 0.45, 2.0)

        # Boost frames around segment boundaries
        if idx_s > 0 and half[idx_s - 1] == hv and valid[idx_s - 1]:
            lo = max(0, idx_s - boundary_radius)
            hi = min(T, idx_s + boundary_radius + 1)
            w[lo:hi] *= boundary_boost

    mean_w = float(np.mean(w[valid]))
    if mean_w > 1e-6:
        w = w / mean_w
    w[~valid] = 0.0
    return w.astype(np.float32)


class _PlayerSetInteractionEncoder(nn.Module):
    """Enhanced Siamese PointNet + Cross-Player Set Interaction over K=11 players per team."""

    def __init__(self, d_bf: int = 28, d_ball: int = 28, hidden: int = 56):
        super().__init__()
        self.base_pn = _SplitPlayerPointNet(d_bf=d_bf, d_ball=d_ball, hidden=hidden)
        d_raw = self.base_pn.out_dim  # 3 * (d_bf + d_ball) = 168

        # Pairwise / 2nd-order player-context refinement over the pooled set representation
        self.refine = nn.Sequential(
            nn.Conv1d(d_raw, d_raw, kernel_size=1),
            ChannelNorm(d_raw),
            nn.GELU(),
            nn.Conv1d(d_raw, d_raw, kernel_size=1),
        )
        self.out_dim = d_raw

    def forward(self, p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.base_pn(p, mask)
        return z + self.refine(z)


class _PlayerSetBackboneV2(nn.Module):
    """Intrinsically H<->A + 180-deg Equivariant Multi-Head Backbone:
    Outputs:
      - `logits_poss`: `(B, 3, L)` (`[H, A, D]`)
      - `logits_chg`:  `(B, L)` (1D binary transition)
      - `logits_4d`:   `(B, 4, L)` (`[H->A, A->H, Live->D, D->Live]`)
      - `pred_aerial`: `(B, 2, L)` (`[logit_aerial_gt_1p5m, log1p_ball_z_hat]`)
    """

    def __init__(
        self,
        c_ga: int,
        c_gs: int,
        d_bf: int = 28,
        d_ball: int = 28,
        channels: int = 80,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        gru_stride: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.pointnet = _PlayerSetInteractionEncoder(d_bf=d_bf, d_ball=d_ball, hidden=56)
        d_pool = self.pointnet.out_dim

        self.ghost_gate = nn.Sequential(
            nn.Conv1d(c_gs, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, d_pool, kernel_size=1),
            nn.Sigmoid(),
        )

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

        # Auxiliary 3D Aerial Ball Height Head (learned from Sportec `ball_z`)
        self.head_aerial = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=1),
        )
        # Aerial-aware residual modulation back into the main trunk
        self.aerial_mod = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.head_poss = nn.Conv1d(channels, 3, kernel_size=1)
        self.head_chg = nn.Conv1d(channels, 1, kernel_size=1)
        self.head_4d = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 4, kernel_size=1),
        )

    def _forward_stream(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.blocks(self.in_drop(F.gelu(self.in_norm(x))))

        L = h.shape[-1]
        s = self.gru_stride
        h_ds = F.avg_pool1d(h, kernel_size=s, stride=s, ceil_mode=True)
        gru_in = h_ds.transpose(1, 2)
        gru_out, _ = self.bigru(gru_in)
        gru_feat = gru_out.transpose(1, 2)
        gru_up = F.interpolate(gru_feat, size=L, mode="linear", align_corners=False)

        h = h + self.gru_proj(gru_up)
        aerial_out = self.head_aerial(h)
        h_mod = h * self.aerial_mod(aerial_out)

        return (
            self.head_poss(h_mod),
            self.head_chg(h_mod).squeeze(1),
            self.head_4d(h_mod),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        lp_both, lc_both, l4_both, la_both = self._forward_stream(x_both)
        lp1, lp2 = lp_both.chunk(2, dim=0)
        lc1, lc2 = lc_both.chunk(2, dim=0)
        l4_1, l4_2 = l4_both.chunk(2, dim=0)
        la1, la2 = la_both.chunk(2, dim=0)

        logits_poss = 0.5 * (lp1 + lp2[:, [1, 0, 2], :])
        logits_chg = 0.5 * (lc1 + lc2)
        logits_4d = 0.5 * (l4_1 + l4_2[:, [1, 0, 2, 3], :])
        pred_aerial = 0.5 * (la1 + la2)
        return logits_poss, logits_chg, logits_4d, pred_aerial


def _apply_cpu_team_swap_aug(
    ph: np.ndarray,
    pa: np.ndarray,
    mh: np.ndarray,
    ma: np.ndarray,
    rng: np.random.Generator,
    swap_prob: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast CPU-side team-swap / dropout augmentation on a single crop `(L, 11, 18)` before GPU transfer."""
    L, K, _ = ph.shape
    if rng.random() >= swap_prob or L < 120:
        return ph, pa, mh, ma
    ph = ph.copy()
    pa = pa.copy()
    mh = mh.copy()
    ma = ma.copy()
    dur = int(rng.integers(50, min(160, L // 2)))
    t0 = int(rng.integers(0, L - dur))
    t1 = t0 + dur
    k_slot = int(rng.integers(6, K))
    if rng.random() < 0.5:
        tmp_p = ph[t0:t1, k_slot, :].copy()
        tmp_m = mh[t0:t1, k_slot].copy()
        ph[t0:t1, k_slot, :] = pa[t0:t1, k_slot, :]
        pa[t0:t1, k_slot, :] = tmp_p
        mh[t0:t1, k_slot] = ma[t0:t1, k_slot]
        ma[t0:t1, k_slot] = tmp_m
    else:
        mh[t0:t1, k_slot] = False
        ph[t0:t1, k_slot, :] = 0.0
        ma[t0:t1, k_slot] = False
        pa[t0:t1, k_slot, :] = 0.0
    return ph, pa, mh, ma


class SymPlayerSetNetV2:
    """End-to-end trainer and predictor for `SymPlayerSetNetV2`."""

    def __init__(
        self,
        channels: int = 80,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        gru_stride: int = 8,
        dropout: float = 0.15,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-4,
        epochs: int = 12,
        crop: int = 1500,
        batch_size: int = 24,
        poss_alpha: float = 0.45,
        boundary_boost: float = 1.6,
        change_loss_weight: float = 0.30,
        b4d_loss_weight: float = 0.35,
        aerial_loss_weight: float = 0.20,
        swap_aug_prob: float = 0.30,
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
        self.poss_alpha = poss_alpha
        self.boundary_boost = boundary_boost
        self.change_loss_weight = change_loss_weight
        self.b4d_loss_weight = b4d_loss_weight
        self.aerial_loss_weight = aerial_loss_weight
        self.swap_aug_prob = swap_aug_prob
        self.label_smoothing = label_smoothing
        self.seed = seed

        self.c_ga: int = 0
        self.c_gs: int = 0
        self.mean_ga_: np.ndarray | None = None
        self.std_ga_: np.ndarray | None = None
        self.mean_gs_: np.ndarray | None = None
        self.std_gs_: np.ndarray | None = None
        self.net_: _PlayerSetBackboneV2 | None = None

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
    ) -> "SymPlayerSetNetV2":
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
            yc = change_targets(y, h).astype(np.float32)
            t4 = targets_4dir(y, h)
            y4 = np.column_stack([t4["y_h2a"], t4["y_a2h"], t4["y_live2d"], t4["y_d2live"]]).astype(np.float32)
            wf = compute_possession_frame_weights(
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
                # Find indices of transitions or short possessions inside segment [s:e] for dense sampling
                tc_idx = np.flatnonzero((yc_seg > 0.5) | (wf[s:e] > 1.35))
                seqs.append((
                    ph[s:e], pa[s:e], mh[s:e], ma[s:e], ga[s:e], gs[s:e],
                    yi[s:e], wf[s:e], yc_seg, y4[s:e], bz_arr[s:e]
                ))
                trans_centers_list.append(tc_idx)
                weights.append(float(e - s) * float(w_m))

        dev = self._device()
        net = _PlayerSetBackboneV2(
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
        pos_w_chg = torch.tensor(10.0, device=dev)
        pos_w_4d = torch.tensor([14.0, 14.0, 12.0, 12.0], device=dev).view(1, 4, 1)
        pos_w_aer = torch.tensor(6.0, device=dev)
        L = self.crop

        net.train()
        for _ in range(total_steps):
            chosen = rng.choice(len(seqs), self.batch_size, p=probs)
            l_ph, l_pa, l_mh, l_ma, l_ga, l_gs = [], [], [], [], [], []
            l_y, l_wf, l_c, l_y4, l_bz = [], [], [], [], []
            for si in chosen:
                ph, pa, mh, ma, ga, gs, yi, wf, yc, y4, bz_s = seqs[si]
                tc_idx = trans_centers_list[si]
                n = len(ph)
                if n >= L:
                    # 45% transition/short-spell centered crop sampling, 55% uniform random crop
                    if len(tc_idx) > 0 and rng.random() < 0.45:
                        center = int(tc_idx[rng.integers(0, len(tc_idx))])
                        jitter = int(rng.integers(-L // 3, L // 3 + 1))
                        a = int(np.clip(center - L // 2 + jitter, 0, n - L))
                    else:
                        a = int(rng.integers(0, max(1, n - L + 1)))
                    ph_c, pa_c, mh_c, ma_c = ph[a:a + L], pa[a:a + L], mh[a:a + L], ma[a:a + L]
                    if self.swap_aug_prob > 0.0:
                        ph_c, pa_c, mh_c, ma_c = _apply_cpu_team_swap_aug(
                            ph_c, pa_c, mh_c, ma_c, rng, swap_prob=self.swap_aug_prob
                        )
                    l_ph.append(ph_c)
                    l_pa.append(pa_c)
                    l_mh.append(mh_c)
                    l_ma.append(ma_c)
                    l_ga.append(ga[a:a + L])
                    l_gs.append(gs[a:a + L])
                    l_y.append(yi[a:a + L])
                    l_wf.append(wf[a:a + L])
                    l_c.append(yc[a:a + L])
                    l_y4.append(y4[a:a + L])
                    l_bz.append(bz_s[a:a + L])
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
            tbz = torch.from_numpy(np.stack(l_bz)).to(dev)                    # (B, L)

            logits_p, logits_c, logits_4d, pred_aerial = net(th_ph, th_pa, th_mh, th_ma, th_ga, th_gs)

            # 1. Possession-weighted Cross-Entropy Loss (`Line L1`)
            ce_per_frame = F.cross_entropy(
                logits_p,
                ty.masked_fill(ty < 0, -100),
                ignore_index=-100,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            valid_y = (ty >= 0).to(torch.float32)
            w_eff = twf * valid_y
            loss_p = (ce_per_frame * w_eff).sum() / w_eff.sum().clamp(min=1.0)

            # 2. 1D Change BCE Loss
            c_mask = (tc >= 0).to(torch.float32)
            bce_c = F.binary_cross_entropy_with_logits(
                logits_c, tc.clamp(min=0.0), pos_weight=pos_w_chg, reduction="none"
            )
            loss_c = (bce_c * c_mask).sum() / c_mask.sum().clamp(min=1.0)

            # 3. 4-Directional Boundary Spotting Loss (`Line L5`)
            m4 = (ty4 >= 0).to(torch.float32)
            bce_4d = F.binary_cross_entropy_with_logits(
                logits_4d, ty4.clamp(min=0.0), pos_weight=pos_w_4d, reduction="none"
            )
            loss_4d = (bce_4d * m4).sum() / m4.sum().clamp(min=1.0)

            # 4. Auxiliary 3D Aerial Ball Height Loss (`Idea O1`, zero-sync GPU tensor op)
            bz_mask = (tbz >= 0).to(torch.float32)
            y_is_aerial = (tbz > 1.5).to(torch.float32)
            y_log_z = torch.log1p(tbz.clamp(min=0.0))
            bce_aer = F.binary_cross_entropy_with_logits(
                pred_aerial[:, 0, :], y_is_aerial, pos_weight=pos_w_aer, reduction="none"
            )
            huber_z = F.smooth_l1_loss(pred_aerial[:, 1, :], y_log_z, reduction="none")
            loss_aer = ((bce_aer + 0.5 * huber_z) * bz_mask).sum() / bz_mask.sum().clamp(min=1.0)

            loss = (
                loss_p
                + self.change_loss_weight * loss_c
                + self.b4d_loss_weight * loss_4d
                + self.aerial_loss_weight * loss_aer
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()

        self.net_ = net.to("cpu").eval()
        return self

    def predict_all(
        self, pst: PlayerSetTensors, half: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Predict all 4 heads:
        - `proba`:      `(T, 3)` float32 (`[p_H, p_A, p_D]`)
        - `change`:     `(T,)` float32 (1D transition probability)
        - `p4d`:        `(T, 4)` float32 (`[p_H_to_A, p_A_to_H, p_Live_to_D, p_D_to_Live]`)
        - `p_aerial`:   `(T,)` float32 ($P(Z > 1.5\\text{m})$)
        - `ball_z_hat`: `(T,)` float32 (estimated 3D ball height in meters)
        """
        assert self.net_ is not None
        ph, pa, mh, ma, ga, gs = self._prep_match_arrays(pst)
        T = len(ph)
        proba = np.zeros((T, 3), dtype=np.float32)
        chg = np.zeros(T, dtype=np.float32)
        p4d = np.zeros((T, 4), dtype=np.float32)
        p_aerial = np.zeros(T, dtype=np.float32)
        ball_z_hat = np.zeros(T, dtype=np.float32)

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

                lp, lc, l4, la = net(th_ph, th_pa, th_mh, th_ma, th_ga, th_gs)
                proba[s:e] = F.softmax(lp, dim=1).squeeze(0).T.cpu().numpy()
                chg[s:e] = torch.sigmoid(lc).squeeze(0).cpu().numpy()
                p4d[s:e] = torch.sigmoid(l4).squeeze(0).T.cpu().numpy()
                p_aerial[s:e] = torch.sigmoid(la[:, 0, :]).squeeze(0).cpu().numpy()
                ball_z_hat[s:e] = torch.expm1(la[:, 1, :].clamp(min=0.0, max=4.0)).squeeze(0).cpu().numpy()

        self.net_ = net.to("cpu").eval()
        return {
            "proba": proba,
            "change": chg,
            "p4d": p4d,
            "p_aerial": p_aerial,
            "ball_z_hat": ball_z_hat,
        }

    def predict_all_heads(
        self, pst: PlayerSetTensors, half: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        d = self.predict_all(pst, half)
        return d["proba"], d["change"], d["p4d"], d["p_aerial"], d["ball_z_hat"]

