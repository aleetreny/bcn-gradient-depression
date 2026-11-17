"""Simple possession rules (team in possession per frame) from tracking alone, <= 3 parameters each.

    from src.features.possession_rules import Kinematics, rule_nearest, rule_last_touch, rule_control
    kin = Kinematics.from_match(match)                 # precomputed once per match
    team = rule_last_touch(kin, r=1.5, theta=35, delta=3.0)

Each rule returns int8 per frame: 0 = H, 1 = A, -1 = unknown (before the first decision). The team is
carried forward through frames without a ball, never across halftime (state restarts at each half).
No rule says anything about dead ball (D).

- rule_nearest(k):          team of the nearest player, switching only after the new team has been
                            nearest on k consecutive ball-visible frames (k = 0: raw nearest).
- rule_last_touch(r, theta, delta): a touch is a frame where the ball changes direction by more than
                            theta degrees or speed by more than delta m/s (over +/-3 frames) with the
                            nearest player within r m; the team is that of the last touch.
- rule_control(r, m, exclusive): a team takes control when one of its players stays within r m of the
                            ball for m consecutive frames (and, if exclusive, no opponent is within r m).

Inspired by Vidal-Codina et al. 2022 (touch detection) and Link & Hoernig 2017 (control zones).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.common import AWAY, FPS, HOME, Match

TOUCH_K = 3


def _segments(n: int, half: np.ndarray | None):
    """[(start, stop)] of constant-half stretches (the whole array if half is None)."""
    if half is None:
        return [(0, n)]
    edges = np.flatnonzero(half[1:] != half[:-1]) + 1
    bounds = np.r_[0, edges, n]
    return list(zip(bounds[:-1], bounds[1:]))


def carry_forward(decisions: np.ndarray, half: np.ndarray | None = None) -> np.ndarray:
    """Forward-fill -1 entries with the last decision (-1 before the first), restarting at each
    change of half so that no decision crosses halftime."""
    out = np.full(len(decisions), -1, np.int8)
    for a, b in _segments(len(decisions), half):
        seg = decisions[a:b]
        idx = np.maximum.accumulate(np.where(seg >= 0, np.arange(len(seg)), -1))
        out[a:b] = np.where(idx >= 0, seg[np.maximum(idx, 0)], -1)
    return out


def run_length_true(mask: np.ndarray, half: np.ndarray | None = None) -> np.ndarray:
    """For each frame, how many consecutive True frames end there (0 where False); runs restart
    at each change of half."""
    out = np.zeros(len(mask), np.int32)
    for a, b in _segments(len(mask), half):
        count = 0
        for i in range(a, b):  # simple loop; ~0.1 s per 200k frames
            count = count + 1 if mask[i] else 0
            out[i] = count
    return out


@dataclass
class Kinematics:
    vis: np.ndarray          # ball x,y visible
    near_team: np.ndarray    # team of nearest visible player (-1 none)
    near_dist: np.ndarray    # distance to nearest player (m)
    dist_team: np.ndarray    # (T, 2) distance to nearest H / A player
    angle: np.ndarray        # direction change at t over +/-TOUCH_K frames (deg), NaN if unknown
    dspeed: np.ndarray       # |speed_out - speed_in| at t (m/s), NaN if unknown
    half: np.ndarray         # half of each frame; rules restart their state at each change

    @classmethod
    def from_match(cls, m: Match) -> "Kinematics":
        T = m.n_frames
        vis = m.ball_xy_visible
        B = m.B[:, :2].astype(np.float64)
        near_team = np.full(T, -1, np.int8)
        near_dist = np.full(T, np.nan)
        dist_team = np.full((T, 2), np.nan)
        rows = np.flatnonzero(vis)
        d = np.linalg.norm(m.P[rows, :, :2].astype(np.float64) - B[rows, None, :], axis=-1)
        d = np.where(np.isnan(d), np.inf, d)
        best = np.argmin(d, axis=1)
        bd = d[np.arange(len(rows)), best]
        ok = np.isfinite(bd)
        near_team[rows[ok]] = m.slot_team[best[ok]]
        near_dist[rows[ok]] = bd[ok]
        for j, team in enumerate((HOME, AWAY)):
            dt = d[:, m.slot_team == team]
            mn = dt.min(axis=1) if dt.shape[1] else np.full(len(rows), np.inf)
            dist_team[rows, j] = np.where(np.isfinite(mn), mn, np.nan)

        k = TOUCH_K
        angle = np.full(T, np.nan)
        dspeed = np.full(T, np.nan)
        t = np.arange(k, T - k)
        same_half = (m.half[t - k] == m.half[t]) & (m.half[t + k] == m.half[t])
        ok = vis[t] & vis[t - k] & vis[t + k] & same_half
        t = t[ok]
        v_in = (B[t] - B[t - k]) * FPS / k
        v_out = (B[t + k] - B[t]) * FPS / k
        s_in = np.linalg.norm(v_in, axis=1)
        s_out = np.linalg.norm(v_out, axis=1)
        cos = np.einsum("ij,ij->i", v_in, v_out) / np.maximum(s_in * s_out, 1e-9)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        ang[(s_in < 0.5) & (s_out < 0.5)] = 0.0   # a ball at rest has no direction
        angle[t] = ang
        dspeed[t] = np.abs(s_out - s_in)
        return cls(vis, near_team, near_dist, dist_team, angle, dspeed, np.asarray(m.half))


def rule_nearest(kin: Kinematics, k: int = 0) -> np.ndarray:
    nt = kin.near_team
    if k <= 0:
        return carry_forward(nt, kin.half)
    out = np.full(len(nt), -1, np.int8)
    current, candidate, count = -1, -1, 0
    for i in range(len(nt)):
        if i > 0 and kin.half[i] != kin.half[i - 1]:
            current, candidate, count = -1, -1, 0
        x = nt[i]
        if x >= 0:
            if x == current:
                candidate, count = -1, 0
            elif x == candidate:
                count += 1
            else:
                candidate, count = x, 1
            if current < 0 or count >= k:
                current, candidate, count = x, -1, 0
        out[i] = current
    return out


def touches(kin: Kinematics, r: float, theta: float, delta: float) -> np.ndarray:
    """Team of the touching player on touch frames, -1 elsewhere."""
    with np.errstate(invalid="ignore"):
        hit = ((kin.angle > theta) | (kin.dspeed > delta)) & (kin.near_dist <= r)
    return np.where(hit, kin.near_team, -1).astype(np.int8)


def rule_last_touch(kin: Kinematics, r: float = 1.5, theta: float = 35.0, delta: float = 3.0) -> np.ndarray:
    return carry_forward(touches(kin, r, theta, delta), kin.half)


def rule_control(kin: Kinematics, r: float = 1.5, m: int = 6, exclusive: bool = True) -> np.ndarray:
    """When both teams qualify on the same frame (only possible with exclusive=False) the closer
    team wins, and an exact tie decides nothing, so swapping H and A swaps the output."""
    with np.errstate(invalid="ignore"):
        close = [kin.dist_team[:, j] <= r for j in (0, 1)]
    held = []
    for j in (0, 1):
        own = close[j] & ~close[1 - j] if exclusive else close[j]
        held.append(run_length_true(own, kin.half) >= m)
    dh, da = kin.dist_team[:, 0], kin.dist_team[:, 1]
    decisions = np.full(len(kin.vis), -1, np.int8)
    decisions[held[0] & ~held[1]] = HOME
    decisions[held[1] & ~held[0]] = AWAY
    both = held[0] & held[1]
    with np.errstate(invalid="ignore"):
        decisions[both & (dh < da)] = HOME
        decisions[both & (da < dh)] = AWAY
    return carry_forward(decisions, kin.half)
