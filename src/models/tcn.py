"""M9: dilated temporal convolutional network over the per-frame features (MPS / CPU).

    from src.models.tcn import TCNModel
    model = TCNModel(seed=0).fit([X1, X2, ...], [labels1, ...], [half1, ...])
    proba, p_change = model.predict(X, half)        # (T, 3) H/A/D probabilities and (T,) change probability

Input: the v1 (+ change) features, robustly standardised with training statistics (median / IQR),
clipped to [-5, 5], NaN -> 0 plus a missing indicator per feature group. Non-causal residual blocks
with dilations 1..128 (receptive field 511 frames, ~20 s), normalisation and GELU.
norm="group" (v2/v3): GroupNorm over channels AND time, so every output depends on the whole input
(60 s crops in training, whole halves at inference; review 004). norm="channel": per-frame LayerNorm
over the channels only, so the network is local (511 frames) and training and inference agree. Two outputs per frame:
H/A/D (cross-entropy, unlabelled frames ignored) and change of team within +/- 6 frames (binary,
weighted). Trained on random 60 s crops; inference on whole halves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.transition import change_targets

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


def _device():
    import torch
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _net(cin, channels, dropout, dilations=DILATIONS, norm="group"):
    import torch.nn as nn

    class ChannelNorm(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.ln = nn.LayerNorm(c)

        def forward(self, x):
            return self.ln(x.transpose(1, 2)).transpose(1, 2)

    class Block(nn.Module):
        def __init__(self, c, d):
            super().__init__()
            self.f = nn.Sequential(nn.Conv1d(c, c, 3, padding=d, dilation=d),
                                   nn.GroupNorm(8, c) if norm == "group" else ChannelNorm(c), nn.GELU(),
                                   nn.Dropout(dropout), nn.Conv1d(c, c, 1))

        def forward(self, x):
            return x + self.f(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(cin, channels, 1)
            self.blocks = nn.Sequential(*[Block(channels, d) for d in dilations])
            self.out = nn.Conv1d(channels, 4, 1)

        def forward(self, x):
            return self.out(self.blocks(self.inp(x)))

    return Net()


@dataclass
class TCNModel:
    channels: int = 128
    dropout: float = 0.1
    crop: int = 1500
    batch: int = 16
    epochs: int = 25
    lr: float = 2e-3
    change_weight: float = 0.5
    dilations: tuple = DILATIONS
    label_smoothing: float = 0.0
    norm: str = "group"
    seed: int = 0
    stats_: dict = field(default_factory=dict)

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.features_].to_numpy(np.float32)
        miss = np.isnan(A)
        A = (np.nan_to_num(A, nan=self.med_) - self.med_) / self.iqr_
        A = np.clip(A, -5, 5)
        A[miss] = 0.0
        groups = np.stack([miss[:, idx].any(1) for idx in self.groups_], 1).astype(np.float32)
        return np.concatenate([A, groups], 1)

    def fit(self, Xs, ys, halves):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.features_ = list(Xs[0].columns)
        big = pd.concat([X.iloc[::20] for X in Xs], ignore_index=True)[self.features_].to_numpy(np.float32)
        self.med_ = np.nanmedian(big, 0)
        q75, q25 = np.nanpercentile(big, 75, 0), np.nanpercentile(big, 25, 0)
        self.iqr_ = np.where((q75 - q25) > 1e-6, q75 - q25, np.nanstd(big, 0) + 1e-6).astype(np.float32)
        self.med_ = np.nan_to_num(self.med_).astype(np.float32)
        prefixes = sorted({c.split("_")[0] for c in self.features_})
        self.groups_ = [np.array([c.split("_")[0] == p for c in self.features_]) for p in prefixes]
        seqs = []
        for X, y, h in zip(Xs, ys, halves):
            A = self._prep(X)
            yc = change_targets(y, h)
            for hv in np.unique(h[h > 0]):
                idx = np.flatnonzero(h == hv)
                seqs.append((A[idx], y[idx].astype(np.int64), yc[idx].astype(np.float32)))
        dev = _device()
        self.net_ = _net(seqs[0][0].shape[1], self.channels, self.dropout, tuple(self.dilations), self.norm).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        total = sum(len(s[0]) for s in seqs)
        steps_per_epoch = max(1, total // (self.crop * self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * steps_per_epoch)
        lengths = np.array([len(s[0]) for s in seqs], float)
        pos_w = torch.tensor(10.0, device=dev)
        self.net_.train()
        for _ in range(self.epochs * steps_per_epoch):
            xb, yb, cb = [], [], []
            for si in rng.choice(len(seqs), self.batch, p=lengths / lengths.sum()):
                A, y, yc = seqs[si]
                a = int(rng.integers(0, max(1, len(A) - self.crop)))
                xb.append(A[a:a + self.crop].T)
                yb.append(y[a:a + self.crop])
                cb.append(yc[a:a + self.crop])
            x = torch.from_numpy(np.stack(xb)).to(dev)
            y = torch.from_numpy(np.stack(yb)).to(dev)
            c = torch.from_numpy(np.stack(cb)).to(dev)
            out = self.net_(x)
            loss = F.cross_entropy(out[:, :3], y.masked_fill(y < 0, -100), ignore_index=-100,
                                   label_smoothing=self.label_smoothing)
            mask = (c >= 0).float()
            bce = F.binary_cross_entropy_with_logits(out[:, 3], c.clamp(min=0), pos_weight=pos_w, reduction="none")
            loss = loss + self.change_weight * (bce * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        return self

    def predict(self, X: pd.DataFrame, half: np.ndarray):
        import torch
        dev = _device()
        A = self._prep(X)
        proba = np.zeros((len(A), 3), np.float32)
        pc = np.zeros(len(A), np.float32)
        self.net_.eval()
        with torch.no_grad():
            for hv in np.unique(half):
                idx = np.flatnonzero(half == hv)
                out = self.net_(torch.from_numpy(A[idx].T[None]).to(dev))[0].cpu().numpy()
                e = np.exp(out[:3] - out[:3].max(0))
                proba[idx] = (e / e.sum(0)).T
                pc[idx] = 1 / (1 + np.exp(-out[3]))
        return proba, pc

    # pickling: the network class is built inside _net, so store its weights and rebuild on load
    def __getstate__(self):
        state = dict(self.__dict__)
        net = state.pop("net_", None)
        if net is not None:
            state["_net_state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            state["_net_cin"] = net.inp.in_channels
        return state

    def __setstate__(self, state):
        net_state = state.pop("_net_state", None)
        cin = state.pop("_net_cin", None)
        self.__dict__.update(state)
        self.__dict__.setdefault("norm", "group")
        if net_state is not None:
            self.net_ = _net(cin, self.channels, self.dropout, tuple(self.dilations), self.norm).to(_device())
            self.net_.load_state_dict(net_state)
