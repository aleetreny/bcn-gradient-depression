"""Player-level possession model (ball-light, log 042): is THIS player the one with the ball?

    from src.features.players import slot_features, owner_labels, PlayerOwnerModel
    F, keep = slot_features(match)                 # rows = (frame, visible slot), slot-invariant columns
    model = PlayerOwnerModel().fit([F1, ...], [y1, ...])
    team = model.team_proba(match)                 # (T,) P(home has the ball), from the slots' owner scores

Idea from the ball-free literature (Ball Radar, PathCRF): the holder is visible in how everybody moves --
opponents run towards him, he slows down and is shielded. Per visible slot, on the direction-normalised
match: own speed and acceleration, how strongly opponents and team-mates move towards it ("attention"),
crowding, position, and the ball (distance, plus whether the ball is an interpolated straight line, so
the model can learn when to ignore it). Labels (Sportec only): the possessing team's player nearest to
the clean ball, mapped to the degraded slot at the same place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.ball_interp import linear_mask
from src.data.common import AWAY, FPS, HOME

CHUNK = 8000


def _vel(xy, half, lag=3):
    v = np.full(xy.shape, np.nan, np.float32)
    same = (half[2 * lag:] == half[:-2 * lag])[:, None, None]
    v[lag:-lag] = np.where(same, (xy[2 * lag:] - xy[:-2 * lag]) * FPS / (2 * lag), np.nan)
    return v


def slot_features(m, every: int = 1):
    """(DataFrame of per-(frame, slot) features, (frame, slot) index arrays) for visible slots on frames ::every."""
    n = m.normalized()
    half = np.asarray(n.half)
    P = n.P[..., :2].astype(np.float32)
    vis = n.slot_visible
    V = _vel(P, half)
    spd = np.linalg.norm(V, axis=2)
    acc = np.full(spd.shape, np.nan, np.float32)
    acc[5:-5] = (spd[10:] - spd[:-10]) * FPS / 10
    B = n.B[:, :2].astype(np.float32)
    bvis = n.ball_xy_visible
    blin = linear_mask(n.B, half)
    team = n.slot_team
    attack = np.where(team == HOME, 1.0, -1.0).astype(np.float32)      # H attacks +x after normalisation
    frames = np.arange(0, len(P), every)
    out = []
    for c0 in range(0, len(frames), CHUNK):
        fr = frames[c0:c0 + CHUNK]
        p, v, s, ok = P[fr], V[fr], spd[fr], vis[fr]
        rel = p[:, None, :, :] - p[:, :, None, :]                       # [t, i, j] = p_j - p_i ... (from i to j)
        dist = np.linalg.norm(rel, axis=3)
        toward = -np.einsum("tijk,tjk->tij", rel, np.nan_to_num(v)) / (dist * np.maximum(s[:, None, :], 1e-3) + 1e-6)
        # toward[t, i, j]: cosine between j's velocity and the direction from j to i (j running at i)
        w = np.clip(toward, 0, 1) * np.nan_to_num(s)[:, None, :] / (1 + dist / 5)
        valid = ok[:, None, :] & ok[:, :, None] & (dist > 0.1) & (dist < 25)
        w = np.where(valid, w, 0.0)
        same = team[None, :, None] == team[None, None, :]
        att_opp = np.where(~same, w, 0).sum(2)
        att_mate = np.where(same, w, 0).sum(2)
        d_ok = np.where(valid, dist, np.inf)
        near_opp = np.where(~same, d_ok, np.inf).min(2)
        near_mate = np.where(same, d_ok, np.inf).min(2)
        n_opp5 = (np.where(~same, d_ok, np.inf) < 5).sum(2)
        n_opp10 = (np.where(~same, d_ok, np.inf) < 10).sum(2)
        bd = np.linalg.norm(p - B[fr][:, None, :], axis=2)
        bd = np.where(bvis[fr][:, None], bd, np.nan)
        rank_b = np.argsort(np.argsort(np.where(ok, np.nan_to_num(bd, nan=999), 999), axis=1), axis=1)
        att_max = np.max(np.where(ok, att_opp, -1), axis=1, keepdims=True)
        tt, ss = np.nonzero(ok)
        out.append(pd.DataFrame({
            "frame": fr[tt], "slot": ss,
            "speed": s[tt, ss], "accel": acc[fr][tt, ss],
            "att_opp": att_opp[tt, ss], "att_mate": att_mate[tt, ss], "att_opp_rel": att_opp[tt, ss] / np.maximum(att_max[tt, 0], 1e-3),
            "near_opp": np.minimum(near_opp[tt, ss], 50), "near_mate": np.minimum(near_mate[tt, ss], 50),
            "n_opp5": n_opp5[tt, ss], "n_opp10": n_opp10[tt, ss],
            "x_att": (p[tt, ss, 0] - n.pitch[0] / 2) * attack[ss], "y_c": np.abs(p[tt, ss, 1] - n.pitch[1] / 2),
            "ball_d": bd[tt, ss], "ball_rank": np.where(bvis[fr][tt], rank_b[tt, ss], np.nan).astype(np.float32),
            "ball_linear": np.where(bvis[fr][tt], blin[fr][tt], np.nan).astype(np.float32),
            "is_home": (team[ss] == HOME).astype(np.float32)}))
    F = pd.concat(out, ignore_index=True)
    # short temporal context per slot track (slots are tracklets; a centred 0.5 s mean of the attention)
    F = F.sort_values(["slot", "frame"], kind="stable")
    g = F.groupby("slot", sort=False)
    for col in ("att_opp", "speed", "ball_d"):
        F[col + "_m"] = g[col].transform(lambda x: x.rolling(13 // max(1, every) or 1, center=True, min_periods=1).mean())
    return F.sort_values(["frame", "slot"], kind="stable").reset_index(drop=True)


def owner_labels(clean, degraded, F) -> np.ndarray:
    """1 for the row of the degraded slot holding the ball (possessing team's player nearest to the clean
    ball, matched to the degraded slot within 1.5 m), 0 for other rows, -1 on frames without a visible owner."""
    lab = clean.labels
    n = clean.normalized()
    nd = degraded.normalized()
    P, B = n.P[..., :2], n.B[:, :2]
    y = np.full(len(F), -1, np.int8)
    fr = F["frame"].to_numpy()
    sl = F["slot"].to_numpy()
    owner_xy = np.full((len(P), 2), np.nan, np.float32)
    for team in (HOME, AWAY):
        rows = np.flatnonzero(lab == team)
        sel = clean.slot_team == team
        d = np.linalg.norm(P[rows][:, sel] - B[rows][:, None], axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
        j = d.argmin(1)
        owner_xy[rows] = P[rows][:, sel][np.arange(len(rows)), j]
    dd = np.linalg.norm(nd.P[fr, sl, :2] - owner_xy[fr], axis=1)
    team_ok = degraded.slot_team[sl] == np.where(np.isin(lab[fr], (HOME, AWAY)), lab[fr], -9)
    cand = np.where(team_ok & (dd < 1.5), dd, np.inf)
    s = pd.Series(cand).groupby(fr).transform("min").to_numpy()
    has = np.isfinite(s)
    y[has] = 0
    y[has & (cand == s)] = 1
    return y


FEATURES = ["speed", "accel", "att_opp", "att_mate", "att_opp_rel", "near_opp", "near_mate", "n_opp5", "n_opp10",
            "x_att", "y_c", "ball_d", "ball_rank", "ball_linear", "att_opp_m", "speed_m", "ball_d_m"]


@dataclass
class PlayerOwnerModel:
    params: dict = field(default_factory=lambda: dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=16))
    seed: int = 0

    def fit(self, Fs, ys):
        import lightgbm as lgb
        X = pd.concat([F.loc[y >= 0, FEATURES] for F, y in zip(Fs, ys)], ignore_index=True)
        y = np.concatenate([y[y >= 0] for y in ys])
        self.model_ = lgb.LGBMClassifier(**self.params, random_state=self.seed).fit(X, y)
        return self

    def team_proba(self, F, n_frames):
        """(P(home), max owner score) per frame; NaN where no slot is visible."""
        p = self.model_.predict_proba(F[FEATURES])[:, 1]
        fr = F["frame"].to_numpy()
        home = F["is_home"].to_numpy() > 0
        tot = np.bincount(fr, p, minlength=n_frames)
        h = np.bincount(fr, np.where(home, p, 0), minlength=n_frames)
        mx = pd.Series(p).groupby(fr).max().reindex(range(n_frames)).to_numpy()
        with np.errstate(all="ignore"):
            return np.where(tot > 0, h / tot, np.nan), mx
