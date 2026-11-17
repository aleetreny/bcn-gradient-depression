"""Features v1: slot-invariant, label-free per-frame features, same code for every source (Annex A of plan 05).

    from src.features.v1 import build_features
    X = build_features(match)          # pandas DataFrame (float32), one row per frame

Groups
- ball:       visibility, frames since/until seen, position (normalised so H attacks +x), last/next known
              position, distance of the last known position to the touchlines/goal lines, outside the pitch,
              speed, acceleration.
- ball-teams: distance of each team's nearest player to the ball (and to the last known ball position),
              their difference, players of each team within 2/5/10 m.
- rules (M1): team of the last touch with radii 1/2/3 m (S1.2), of the control zone (r 2 m exclusive,
              m 3) and of the nearest player; frames since the last touch of each team; touch events.
              Team-valued features are encoded +1 = H, -1 = A, 0 = unknown.
- collective: per team centroid, spread, defensive and forward lines, width, mean/p90 speed, fraction
              almost still, players in each penalty area, visible slots, goalkeeper position and distance.
- context:    centred rolling means of key features over +/-{0.4, 0.8, 2, 4, 8} s and, for the dead-ball
              signals, +/-{20, 40, 80} s; forward-minus-backward differences over 2 s.
- structure:  minutes since the start of the half.
Not used: the ball's third channel (not simulated for Sportec), position jitter (domain giveaway, S2.1),
Smart Tagging (absent in Sportec; possible second stage, M5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.common import AWAY, FPS, HOME, Match
from src.features.possession_rules import Kinematics, rule_control, rule_last_touch, rule_nearest, touches

CAP = 30 * FPS
SHORT_WINDOWS_S = (0.4, 0.8, 2, 4, 8)
LONG_WINDOWS_S = (20, 40, 80)
PENALTY_DEPTH, PENALTY_HALF_WIDTH = 16.5, 20.16


def _since(mask, half):
    out = np.full(len(mask), CAP, np.int32)
    last = None
    for i in range(len(mask)):
        if i == 0 or half[i] != half[i - 1]:
            last = None
        if mask[i]:
            last = i
        out[i] = CAP if last is None else min(i - last, CAP)
    return out


def _ffill(values, mask, half):
    """Last value where mask was True (NaN before), restarting at each half."""
    out = np.full(values.shape, np.nan, np.float32)
    last = None
    for i in range(len(mask)):
        if i == 0 or half[i] != half[i - 1]:
            last = None
        if mask[i]:
            last = values[i]
        if last is not None:
            out[i] = last
    return out


def _signed(team):
    return np.where(team == HOME, 1.0, np.where(team == AWAY, -1.0, 0.0)).astype(np.float32)


def _rolling(s: pd.Series, half, frames):
    return s.groupby(half).transform(lambda x: x.rolling(frames, center=True, min_periods=1).mean())


def build_features(m: Match) -> pd.DataFrame:
    n = m.normalized()
    P, B, half = n.P, n.B, np.asarray(n.half)
    L, W = n.pitch
    vis = n.slot_visible
    bvis = n.ball_xy_visible
    T = n.n_frames
    f = {}

    # ---------------- ball
    f["ball_visible"] = bvis.astype(np.float32)
    f["ball_since_seen"] = _since(bvis, half)
    f["ball_until_seen"] = _since(bvis[::-1], half[::-1])[::-1]
    bx = np.where(bvis, B[:, 0], np.nan)
    by = np.where(bvis, B[:, 1], np.nan)
    f["ball_x"], f["ball_y"] = bx, by
    last_xy = _ffill(B[:, :2], bvis, half)
    next_xy = _ffill(B[::-1, :2], bvis[::-1], half[::-1])[::-1]
    f["ball_last_x"], f["ball_last_y"] = last_xy[:, 0], last_xy[:, 1]
    f["ball_next_x"], f["ball_next_y"] = next_xy[:, 0], next_xy[:, 1]
    f["ball_last_to_touchline"] = np.minimum(last_xy[:, 1], W - last_xy[:, 1])
    f["ball_last_to_goalline"] = np.minimum(last_xy[:, 0], L - last_xy[:, 0])
    f["ball_outside_pitch"] = np.where(bvis, ((bx < 0) | (bx > L) | (by < 0) | (by > W)).astype(np.float32), np.nan)
    f["ball_jump_last_next"] = np.linalg.norm(next_xy - last_xy, axis=1)
    sp = np.full(T, np.nan, np.float32)
    ok = bvis[2:] & bvis[:-2] & (half[2:] == half[:-2])
    sp[1:-1][ok] = np.linalg.norm(B[2:, :2][ok] - B[:-2, :2][ok], axis=1) * FPS / 2
    f["ball_speed"] = sp
    acc = np.full(T, np.nan, np.float32)
    acc[1:-1] = np.abs(sp[2:] - sp[:-2]) * FPS / 2
    f["ball_accel"] = acc

    # ---------------- ball <-> teams
    d_ball = np.linalg.norm(P[..., :2] - B[:, None, :2], axis=-1)
    d_last = np.linalg.norm(P[..., :2] - last_xy[:, None, :], axis=-1)
    speed = np.where(vis, P[..., 2], np.nan)
    with np.errstate(all="ignore"):
        for team, k in ((HOME, "H"), (AWAY, "A")):
            sel = n.slot_team == team
            for name, d in (("ball", d_ball), ("lastball", d_last)):
                dt = d[:, sel]
                f["dist_{}_{}".format(name, k)] = np.where(np.isfinite(dt).any(1), np.nanmin(np.where(np.isfinite(dt), dt, np.inf), 1), np.nan)
            for r in (2, 5, 10):
                f["n_within_{}m_{}".format(r, k)] = np.where(bvis, np.nansum(d_ball[:, sel] < r, 1), np.nan)
        f["dist_ball_diff"] = f["dist_ball_H"] - f["dist_ball_A"]
        f["dist_lastball_diff"] = f["dist_lastball_H"] - f["dist_lastball_A"]

    # ---------------- possession rules (M1)
    kin = Kinematics.from_match(n)
    for r in (1.0, 2.0, 3.0):
        f["touch_team_r{}".format(int(r))] = _signed(rule_last_touch(kin, r, 20, 3.0))
    f["control_team"] = _signed(rule_control(kin, 2.0, 3, True))
    f["nearest_team"] = _signed(rule_nearest(kin, 0))
    f["nearest_team_hold25"] = _signed(rule_nearest(kin, 25))
    ev = touches(kin, 3.0, 20, 3.0)
    f["touch_event"] = _signed(ev)
    for team, k in ((HOME, "H"), (AWAY, "A")):
        f["since_touch_" + k] = _since(ev == team, half)
        f["until_touch_" + k] = _since((ev == team)[::-1], half[::-1])[::-1]
    f["ball_angle_change"] = kin.angle.astype(np.float32)
    f["ball_speed_change"] = kin.dspeed.astype(np.float32)

    # ---------------- collective
    with np.errstate(all="ignore"):
        for team, k in ((HOME, "H"), (AWAY, "A")):
            sel = (n.slot_team == team)
            outfield = sel & ~n.slot_is_gk
            x = np.where(vis[:, outfield], P[:, outfield, 0], np.nan)
            y = np.where(vis[:, outfield], P[:, outfield, 1], np.nan)
            f["n_visible_" + k] = vis[:, sel].sum(1).astype(np.float32)
            f["centroid_x_" + k] = np.nanmean(x, 1)
            f["centroid_y_" + k] = np.nanmean(y, 1)
            f["spread_x_" + k] = np.nanstd(x, 1)
            f["spread_y_" + k] = np.nanstd(y, 1)
            # H attacks +x: its defensive line is the deepest (min x) outfield player; A's is the max x
            f["def_line_" + k] = np.nanmin(x, 1) if team == HOME else L - np.nanmax(x, 1)
            f["fwd_line_" + k] = np.nanmax(x, 1) if team == HOME else L - np.nanmin(x, 1)
            f["width_" + k] = np.nanmax(y, 1) - np.nanmin(y, 1)
            s = speed[:, sel]
            f["speed_mean_" + k] = np.nanmean(s, 1)
            f["speed_p90_" + k] = np.nanpercentile(s, 90, axis=1)
            f["frac_still_" + k] = np.nanmean(np.where(np.isfinite(s), s < 1.5, np.nan), 1)
            in_box_own = (x < PENALTY_DEPTH) if team == HOME else (x > L - PENALTY_DEPTH)
            in_box_opp = (x > L - PENALTY_DEPTH) if team == HOME else (x < PENALTY_DEPTH)
            wide_ok = np.abs(y - W / 2) < PENALTY_HALF_WIDTH
            f["in_own_box_" + k] = np.nansum(in_box_own & wide_ok, 1).astype(np.float32)
            f["in_opp_box_" + k] = np.nansum(in_box_opp & wide_ok, 1).astype(np.float32)
            gk = np.flatnonzero(sel & n.slot_is_gk)
            if len(gk):
                gx = P[:, gk[0], 0]
                f["gk_x_" + k] = gx if team == HOME else L - gx
                f["gk_dist_ball_" + k] = d_ball[:, gk[0]]
            else:
                f["gk_x_" + k] = np.full(T, np.nan)
                f["gk_dist_ball_" + k] = np.full(T, np.nan)
        f["speed_mean_all"] = np.nanmean(speed, 1)
        f["frac_still_all"] = np.nanmean(np.where(np.isfinite(speed), speed < 1.5, np.nan), 1)
    f["n_visible_all"] = vis.sum(1).astype(np.float32)
    f["ball_to_centroid_H"] = f["ball_last_x"] - f["centroid_x_H"]
    f["ball_to_centroid_A"] = f["ball_last_x"] - f["centroid_x_A"]
    f["centroid_gap_x"] = f["centroid_x_H"] - f["centroid_x_A"]

    # ---------------- structure
    start = np.zeros(T)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        start[idx] = idx[0]
    f["minutes_in_half"] = ((np.arange(T) - start) / FPS / 60).astype(np.float32)

    X = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})

    # ---------------- temporal context
    short_keys = ["ball_visible", "ball_speed", "dist_ball_diff", "dist_lastball_diff", "touch_team_r2", "control_team",
                  "nearest_team", "speed_mean_H", "speed_mean_A", "centroid_gap_x", "n_within_5m_H", "n_within_5m_A"]
    long_keys = ["speed_mean_all", "frac_still_all", "ball_visible", "spread_x_H", "spread_x_A", "n_visible_all"]
    hs = pd.Series(half)
    extra = {}
    for key in short_keys:
        for w in SHORT_WINDOWS_S:
            extra["{}_m{}s".format(key, w)] = _rolling(X[key], hs, max(1, int(w * FPS)))
    for key in long_keys:
        for w in LONG_WINDOWS_S:
            extra["{}_m{}s".format(key, w)] = _rolling(X[key], hs, int(w * FPS))
    for key in ("speed_mean_all", "ball_speed", "dist_lastball_diff", "touch_team_r2", "centroid_x_H"):
        fwd = X[key].groupby(hs).transform(lambda x: x[::-1].rolling(2 * FPS, min_periods=1).mean()[::-1])
        bwd = X[key].groupby(hs).transform(lambda x: x.rolling(2 * FPS, min_periods=1).mean())
        extra["{}_fwd_minus_bwd_2s".format(key)] = fwd - bwd
    X = pd.concat([X, pd.DataFrame(extra).astype(np.float32)], axis=1)
    return X


GROUPS = {
    "ball": ("ball_",),
    "ball-teams": ("dist_", "n_within_"),
    "rules (M1)": ("touch_", "control_team", "nearest_team", "since_touch", "until_touch"),
    "collective": ("n_visible", "centroid", "spread", "def_line", "fwd_line", "width", "speed_", "frac_still", "in_own_box",
                   "in_opp_box", "gk_"),
    "structure": ("minutes_in_half",),
}


def feature_group(name: str) -> str:
    if "_m" in name and name.rsplit("_m", 1)[-1].endswith("s") or name.endswith("_fwd_minus_bwd_2s"):
        return "context"
    for group, prefixes in GROUPS.items():
        if name.startswith(prefixes):
            return group
    return "other"


FEATURES_VERSION = 1  # bump whenever build_features changes


def match_fingerprint(m: Match) -> str:
    """Hash of everything build_features reads from a Match (positions, ball, slots, halves, direction)."""
    import hashlib
    h = hashlib.blake2b(digest_size=12)
    for arr in (m.P, m.B, m.slot_team, m.slot_is_gk, m.half, m.attack_sign):
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    h.update(repr(tuple(m.pitch)).encode() + b"features_v%d" % FEATURES_VERSION)
    return h.hexdigest()


def cached_features(m: Match, key: str, cache_dir=None, use_cache: bool = True) -> pd.DataFrame:
    """build_features with a parquet cache keyed by `key` AND the content of the match.

    The file name carries match_fingerprint(m), so any change in the data, the degradation config,
    its empirical parameters or the degradation code produces a different file; a stale cache can
    never be returned for different inputs. use_cache=False always rebuilds (and does not write).
    """
    from pathlib import Path
    if not use_cache:
        return build_features(m)
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "local_data" / "cache" / "features_v1"
    path = root / "{}_{}.parquet".format(key, match_fingerprint(m))
    if path.is_file():
        return pd.read_parquet(path)
    X = build_features(m)
    root.mkdir(parents=True, exist_ok=True)
    X.to_parquet(path)
    return X
