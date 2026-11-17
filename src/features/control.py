"""Control features (F3.6): who carries the ball, who is it travelling to, is it contested, which way is play going.

    from src.features.control import control_features, cached_control
    C = control_features(match)          # DataFrame aligned with build_features(match)

Slot-invariant and label-free, computed on the direction-normalised match (H attacks +x):
- carrier: for each visible player, closeness to the ball x alignment of velocities x similarity of speeds
  (a player moving with the ball); the best score per team, its difference, and short rolling means.
- receiver: when the ball moves faster than 4 m/s, the player ahead of the ball closest to its ray
  (within 35 m): signed team, perpendicular distance, time to reach it.
- contest: players of each team within 2 m of the ball, and whether both teams are there.
- direction: ball velocity along x (towards A's goal is +), and both teams' mean velocity along x,
  with rolling means over 1 and 3 s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.common import AWAY, FPS, HOME, Match

VERSION = 1
VERSION2 = 1  # control_features_v2 (F3.10)


def _velocity(xy, half, lag=2):
    """Centred difference over +/- lag frames (m/s), NaN where not both ends are known or across halves."""
    v = np.full(xy.shape, np.nan, np.float32)
    a, b = xy[2 * lag:], xy[:-2 * lag]
    same = (half[2 * lag:] == half[:-2 * lag])
    d = (a - b) * FPS / (2 * lag)
    if xy.ndim == 3:
        same = same[:, None, None]
    else:
        same = same[:, None]
    v[lag:-lag] = np.where(same, d, np.nan)
    return v


def _roll(x, half, w):
    return pd.Series(x).groupby(pd.Series(half)).transform(lambda s: s.rolling(w, center=True, min_periods=1).mean()).to_numpy(np.float32)


def control_features(m: Match) -> pd.DataFrame:
    n = m.normalized()
    half = np.asarray(n.half)
    P = n.P[..., :2].astype(np.float32)
    B = n.B[:, :2].astype(np.float32)
    bvis = n.ball_xy_visible
    vp = _velocity(P, half)                       # (T, S, 2)
    vb = _velocity(B, half)                       # (T, 2)
    sb = np.linalg.norm(vb, axis=1)
    sp = np.linalg.norm(vp, axis=2)
    d = np.linalg.norm(P - B[:, None, :], axis=2)
    f = {}
    with np.errstate(all="ignore"):
        cos = np.einsum("tsk,tk->ts", vp, vb) / (sp * sb[:, None] + 1e-6)
        moving = (sb[:, None] > 1.0) & (sp > 1.0)
        align = np.where(moving, np.clip(cos, 0, 1), 0.0)
        sim = 1.0 / (1.0 + np.abs(sp - sb[:, None]))
        carry = np.exp(-d / 1.5) * align * sim
        carry = np.where(np.isfinite(carry), carry, np.nan)
        for team, k in ((HOME, "H"), (AWAY, "A")):
            sel = n.slot_team == team
            c = carry[:, sel]
            f["carry_" + k] = np.where(bvis, np.nan_to_num(np.nanmax(np.where(np.isfinite(c), c, -1), 1), nan=0).clip(0), np.nan)
            f["contest_n2m_" + k] = np.where(bvis, np.nansum(d[:, sel] < 2.0, 1), np.nan)
            vx = np.where(n.slot_visible[:, sel], vp[:, sel, 0], np.nan)
            f["team_vx_" + k] = np.nanmean(vx, 1)
        f["carry_diff"] = f["carry_H"] - f["carry_A"]
        f["contested"] = ((f["contest_n2m_H"] > 0) & (f["contest_n2m_A"] > 0)).astype(np.float32)
        f["contested"][~bvis] = np.nan
        f["ball_vx"] = vb[:, 0]
        f["ball_vy_abs"] = np.abs(vb[:, 1])

        # receiver: player ahead of the ball nearest to its ray
        T = len(B)
        rec_team = np.full(T, np.nan, np.float32)
        rec_perp = np.full(T, np.nan, np.float32)
        rec_eta = np.full(T, np.nan, np.float32)
        fast = np.flatnonzero(bvis & (sb > 4.0) & np.isfinite(sb))
        if len(fast):
            u = vb[fast] / sb[fast, None]
            rel = P[fast] - B[fast, None, :]
            along = np.einsum("tsk,tk->ts", rel, u)
            perp = np.abs(rel[..., 0] * u[:, None, 1] - rel[..., 1] * u[:, None, 0])
            ok = np.isfinite(along) & (along > 0.5) & (along < 35)
            score = np.where(ok, perp + 0.1 * along, np.inf)
            best = np.argmin(score, 1)
            has = np.isfinite(score[np.arange(len(fast)), best])
            idx = fast[has]
            bb = best[has]
            rec_team[idx] = np.where(n.slot_team[bb] == HOME, 1.0, -1.0)
            rec_perp[idx] = perp[has, bb]
            rec_eta[idx] = along[has, bb] / sb[idx]
        f["receiver_team"] = rec_team
        f["receiver_perp"] = rec_perp
        f["receiver_eta"] = rec_eta

    X = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
    extra = {}
    for key, ws in (("carry_diff", (0.4, 1.0)), ("receiver_team", (0.4, 1.0)), ("contested", (1.0,)),
                    ("ball_vx", (1.0, 3.0)), ("team_vx_H", (1.0, 3.0)), ("team_vx_A", (1.0, 3.0))):
        for w in ws:
            extra["{}_m{}s".format(key, w)] = _roll(np.nan_to_num(X[key].to_numpy(), nan=0.0) if key == "receiver_team" else X[key].to_numpy(),
                                                     half, max(1, int(w * FPS)))
    return pd.concat([X, pd.DataFrame(extra)], axis=1)


def _since(mask, half, cap=750):
    out = np.full(len(mask), cap, np.float32)
    last = None
    for i in range(len(mask)):
        if i == 0 or half[i] != half[i - 1]:
            last = None
        if mask[i]:
            last = i
        if last is not None:
            out[i] = min(i - last, cap)
    return out


def control_features_v2(m: Match, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extra control features (F3.10): kicks, ball in flight, persistence of carrying, long context."""
    n = m.normalized()
    half = np.asarray(n.half)
    C = base if base is not None else control_features(m)
    B = n.B[:, :2].astype(np.float32)
    bvis = n.ball_xy_visible
    vb = _velocity(B, half, lag=1)
    sb = np.linalg.norm(vb, axis=1)
    acc = np.full(len(B), np.nan, np.float32)
    acc[1:-1] = (sb[2:] - sb[:-2]) * FPS / 2
    d = np.linalg.norm(n.P[..., :2] - B[:, None, :], axis=2)
    with np.errstate(all="ignore"):
        dmin = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1)
        near = np.argmin(np.where(np.isfinite(d), d, np.inf), 1)
    near_team = np.where(np.isfinite(dmin), np.where(n.slot_team[near] == HOME, 1.0, -1.0), 0.0)
    kick = bvis & np.isfinite(acc) & (acc > 15.0) & (dmin < 3.0)
    kick_team = np.where(kick, near_team, 0.0)
    f = {}
    # team of the last kick, carried forward inside each half
    last = np.zeros(len(B), np.float32)
    cur = 0.0
    for i in range(len(B)):
        if i == 0 or half[i] != half[i - 1]:
            cur = 0.0
        if kick_team[i] != 0:
            cur = kick_team[i]
        last[i] = cur
    f["last_kick_team"] = last
    f["since_kick"] = _since(kick, half)
    f["in_flight"] = np.where(bvis, ((sb > 6.0) & (dmin > 2.0)).astype(np.float32), np.nan)
    carry_h = np.nan_to_num(C["carry_H"].to_numpy(), nan=0.0) > 0.3
    carry_a = np.nan_to_num(C["carry_A"].to_numpy(), nan=0.0) > 0.3
    f["since_carry_H"] = _since(carry_h, half)
    f["since_carry_A"] = _since(carry_a, half)
    f["since_carry_diff"] = f["since_carry_A"] - f["since_carry_H"]
    X = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
    extra = {}
    for key, src, ws in (("carry_diff", C["carry_diff"].to_numpy(), (4.0, 8.0)), ("ball_vx", C["ball_vx"].to_numpy(), (6.0,)),
                         ("last_kick_team", X["last_kick_team"].to_numpy(), (1.0, 3.0)), ("in_flight", X["in_flight"].to_numpy(), (1.0,))):
        for w in ws:
            extra["{}_m{}s".format(key, w)] = _roll(src, half, int(w * FPS))
    return pd.concat([X, pd.DataFrame(extra)], axis=1)


def cached_control2(m: Match, key: str, cache_dir=None) -> pd.DataFrame:
    from pathlib import Path
    from src.features.v1 import match_fingerprint
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "local_data" / "cache" / "control2_v{}".format(VERSION2)
    path = root / "{}_{}.parquet".format(key, match_fingerprint(m))
    if path.is_file():
        return pd.read_parquet(path)
    C2 = control_features_v2(m, cached_control(m, key))
    root.mkdir(parents=True, exist_ok=True)
    C2.to_parquet(path)
    return C2


def cached_control(m: Match, key: str, cache_dir=None) -> pd.DataFrame:
    from pathlib import Path
    from src.features.v1 import match_fingerprint
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "local_data" / "cache" / "control_v{}".format(VERSION)
    path = root / "{}_{}.parquet".format(key, match_fingerprint(m))
    if path.is_file():
        return pd.read_parquet(path)
    C = control_features(m)
    root.mkdir(parents=True, exist_ok=True)
    C.to_parquet(path)
    return C
