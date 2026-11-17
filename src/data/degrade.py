"""Degrade a clean Match (Sportec) so that it looks like the BAR-ATH ATD. Labels are never used.

    from src.data.degrade import DegradeConfig, degrade
    d = degrade(match, DegradeConfig(), seed=0)

Components (each can be switched off in DegradeConfig), calibrated in S2.1 against the ATD
statistics of S1.3/S1.4 (local_data/degradation_params.json):
- smooth:      centred moving average of player positions (the ATD is ~7x smoother than Sportec);
               speeds are recomputed from the smoothed positions.
- absences:    players disappear, independently (short absences) and in team episodes where several
               players of one team vanish together (partial occlusion).
- fragments:   identities break (extra cuts on top of absences) and fragments are packed into a
               reusable, team-fixed pool of slots, so slots are tracklets as in the ATD; each team's
               goalkeeper keeps a dedicated slot (v2; in v1 keeper fragments leaked into outfield slots).
- duplicates:  at a cut, the old slot can linger for a few seconds next to the player.
- ball gaps:   a hazard process that starts long gaps more often when the ball is slow, when several
               players are near it, and when the players are slow (stoppage proxy, from tracking
               only); gap lengths are drawn from the ATD's empirical distribution per context, and
               a minimum visible stretch separates gaps.
- static ball: a ball at rest for more than static_min_s is hidden until it moves (with a probability):
               only ~5 % of visible ATD ball frames are slower than 0.5 m/s, against ~40 % in Sportec,
               whose dead balls sit still.
- handled ball (v3): Sportec freezes the ball during stoppages (a perfect, Sportec-only label cue
               found in S2.3); the visible part of every rest now follows the nearest player.
- dead ball (v4, dead_ball_model="atd"): v3 left the ball glued to one player (review 003); rests
               become held / rolling segments calibrated to the ATD's dead-ball distance and speed.
- blackouts:   a few ~1-minute stretches with almost no players and no ball.
- interp (v5, ball_interp=True): the ATD ball is drawn piecewise-linearly (log 042): inside visible
               stretches, linear runs replace the real ball by the straight line between their ends,
               alternating with kept (curved) runs; run lengths from src/data/ball_interp.py (measured
               on the full BAR-ATH ATD, no labels). Uses its own random stream, so configs without it
               are unchanged.
- ghosts (v6, ghost_occlusion_rate / ghost_offset_rate > 0): the ATD ball and players often disagree
               (a kick with nobody within 4 m: 25-28 % in the ATD, 5 % here before, log 042-044):
               episodes hide the player nearest to the ball (occlusion at the ball) or displace the ball
               by a few metres (misplaced ball). Rates calibrated without labels; own random stream.
The ball's third channel is not simulated (B[:, 2] keeps the speed from positions); v1 features
must not use it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from src.data.common import FPS, HOME, AWAY, Match

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "local_data" / "degradation_params.json"
SLOW = 1.2
VERSION = 5  # bump when the algorithm changes, so cached features of degraded matches are rebuilt


@dataclass
class DegradeConfig:
    smooth_window: int = 9                 # frames (odd); 0 = off
    absence_rate: float = 1 / (25 * 120)   # per player per frame (one absence every ~2 min)
    absence_median_s: float = 4.0
    episode_rate: float = 1 / (25 * 60)    # per team per frame
    episode_players: tuple = (2, 5)
    episode_len_s: tuple = (2.0, 15.0)
    cut_rate: float = 1 / (25 * 90)        # extra identity cuts per player per frame
    dup_prob: float = 0.35                 # probability that a cut leaves a lingering duplicate
    dup_len_s: tuple = (1.0, 6.0)
    dup_offset_m: tuple = (1.0, 4.0)
    gap_rate_moving: float = 0.0008        # ball gap starts per visible frame in moving play (before speed/crowd factors)
    gap_rate_slow: float = 0.004           # ... when the players are slow (stoppage proxy)
    gap_len_scale: float = 1.0             # multiplies lengths drawn from the ATD gap distribution
    min_visible_s: float = 1.0             # visible stretch kept after each gap (ATD gaps do not chain)
    static_speed: float = 0.5              # m/s; the ATD rarely shows a ball at rest
    static_min_s: float = 1.0              # a ball at rest longer than this ...
    static_hide_prob: float = 0.8          # ... is hidden from then on, with this probability per rest
    handled_ball: bool = True              # v3: a resting ball follows the nearest player (no frozen ball)
    handled_offset_m: float = 0.6
    handled_noise_m: float = 0.15
    dead_ball_model: str = "handled"       # "handled" (v3, keeps degrade_v2.json reproducible) or "atd" (v4)
    dead_roll_prob: float = 0.5            # v4: share of dead-ball segments where the ball rolls / flies
    dead_speed_median: float = 2.5         # v4: m/s, median speed of a rolling dead ball (lognormal)
    dead_hold_max_m: float = 2.5           # v4: a held dead ball sits 0.3..this many metres from the holder
    dead_segment_s: tuple = (1.0, 4.0)     # v4: duration of each held / rolling segment
    ball_interp: bool = False              # v5: piecewise-linear ATD-like ball (log 042)
    interp_dev_max: float = 1.0            # v5: m, a straight run may stray this far from the real ball
    ghost_occlusion_rate: float = 0.0      # v6: episodes per visible-ball frame hiding the player nearest the ball
    ghost_occlusion_s: tuple = (0.5, 3.0)
    ghost_offset_rate: float = 0.0         # v6: episodes per visible-ball frame displacing the ball
    ghost_offset_s: tuple = (0.4, 2.0)
    ghost_offset_m: tuple = (2.0, 6.0)
    blackouts_per_90: float = 3.0
    blackout_len_s: tuple = (35.0, 65.0)
    pool_slots: tuple = (24, 22)           # team-fixed slot pools (H, A), as in the ATD
    components: tuple = ("smooth", "absences", "fragments", "duplicates", "ball", "blackouts")
    params: dict = field(default_factory=dict)


def load_params() -> dict:
    return json.loads(PARAMS.read_text())


def _moving_average(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    valid = np.isfinite(x)
    num = np.convolve(np.where(valid, x, 0.0), k, "same")
    den = np.convolve(valid.astype(float), k, "same")
    return np.where(valid & (den > 0), num / np.maximum(den, 1e-9), np.nan)


def _speed(xy, half):
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        if len(idx) > 2:
            v[idx] = np.linalg.norm(np.gradient(xy[idx], axis=0), axis=-1) * FPS
    return v


def _runs_mask(T, rng, rate, length_sampler):
    """Boolean mask True inside events started by a Bernoulli(rate) process with sampled lengths."""
    mask = np.zeros(T, bool)
    starts = np.flatnonzero(rng.random(T) < rate)
    for s in starts:
        mask[s:s + max(1, int(length_sampler()))] = True
    return mask


def _hist_sampler(rng, edges, counts):
    edges = np.asarray(edges, float)
    p = np.asarray(counts, float)
    p = p / p.sum()

    def sample():  # log-uniform inside the chosen bin: the bins span orders of magnitude
        i = rng.choice(len(p), p=p)
        lo, hi = max(edges[i], 1.0), edges[i + 1]
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return sample


def _handled_ball(B, Q, rest, rng, cfg):
    """Replace frozen-ball stretches by a ball carried by the nearest visible player (offset + noise)."""
    B = B.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, rest.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 3 or not np.isfinite(B[a, 0]):
            continue
        d = np.linalg.norm(Q[a, :, :2] - B[a, :2], axis=1)
        if not np.isfinite(d).any():
            continue
        k = int(np.nanargmin(np.where(np.isfinite(d), d, np.inf)))
        track = Q[a:b, k, :2].astype(np.float64)
        ok = np.isfinite(track[:, 0])
        if ok.sum() < 2:
            continue
        idx = np.arange(b - a)
        track = np.stack([np.interp(idx, idx[ok], track[ok, j]) for j in (0, 1)], axis=1)
        ang = rng.uniform(0, 2 * np.pi)
        offset = cfg.handled_offset_m * np.array([np.cos(ang), np.sin(ang)])
        noise = np.cumsum(rng.normal(0, cfg.handled_noise_m / 5, (b - a, 2)), axis=0)
        noise = np.clip(noise, -cfg.handled_noise_m * 3, cfg.handled_noise_m * 3)
        B[a:b, :2] = (track + offset + noise).astype(np.float32)
    return B


def _dead_ball_motion(B, Q, rest, rng, cfg, pitch):
    """v4: ATD-like dead ball. Each rest of the original (frozen) ball becomes a sequence of segments:
    held by one of the three players nearest to it (at 0.3..dead_hold_max_m, following the holder) or
    rolling / thrown in a straight line to a point 3-15 m away at a lognormal speed. The last second
    brings the ball back to the original spot (where the restart is taken). Calibrated in S2.1b so
    that the distance to the nearest player and the speed of visible dead balls match the ATD."""
    B = B.copy()
    L, W = pitch
    edges = np.flatnonzero(np.diff(np.r_[0, rest.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 3 or not np.isfinite(B[a, 0]):
            continue
        spot = B[a, :2].astype(np.float64).copy()
        pos = spot.copy()
        path = np.empty((b - a, 2))
        t = 0
        end_back = max(0, (b - a) - FPS)
        while t < end_back:
            n = min(end_back - t, max(2, int(rng.uniform(*cfg.dead_segment_s) * FPS)))
            frame = a + t
            if rng.random() < cfg.dead_roll_prob:
                ang = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(3.0, 15.0)
                target = np.clip(pos + dist * np.array([np.cos(ang), np.sin(ang)]), [-2.0, -2.0], [L + 2.0, W + 2.0])
                speed = rng.lognormal(np.log(cfg.dead_speed_median), 0.6)
                step = np.linalg.norm(target - pos)
                k = np.minimum(np.arange(1, n + 1) * speed / FPS / max(step, 1e-6), 1.0)
                seg = pos + k[:, None] * (target - pos)
            else:
                d = np.linalg.norm(Q[frame, :, :2] - pos, axis=1)
                ok = np.flatnonzero(np.isfinite(d))
                if len(ok) == 0:
                    seg = np.repeat(pos[None], n, axis=0)
                else:
                    k = int(rng.choice(ok[np.argsort(d[ok])[:3]]))
                    track = Q[frame:frame + n, k, :2].astype(np.float64)
                    good = np.isfinite(track[:, 0])
                    if good.sum() < 2:
                        track = np.repeat(Q[frame, k, :2][None].astype(np.float64), n, axis=0)
                    else:
                        idx = np.arange(n)
                        track = np.stack([np.interp(idx, idx[good], track[good, j]) for j in (0, 1)], axis=1)
                    ang = rng.uniform(0, 2 * np.pi)
                    off = rng.uniform(0.3, cfg.dead_hold_max_m) * np.array([np.cos(ang), np.sin(ang)])
                    seg = track + off
            path[t:t + n] = seg
            pos = seg[-1]
            t += n
        if end_back < b - a:  # back to the restart spot over the last second
            n = (b - a) - end_back
            k = np.linspace(0, 1, n)[:, None]
            path[end_back:] = pos + k * (spot - pos)
        jitter = np.cumsum(rng.normal(0, 0.03, (b - a, 2)), axis=0)
        B[a:b, :2] = (path + np.clip(jitter, -0.3, 0.3)).astype(np.float32)
    return B


def _ghosts(Q, B, half, rng, cfg):
    """Hide the slot nearest to the ball for short episodes, and displace the ball for others."""
    Q, B = Q.copy(), B.copy()
    T = len(B)
    bvis = np.isfinite(B[:, 0]) & np.isfinite(B[:, 1])
    d = np.linalg.norm(Q[..., :2] - B[:, None, :2], axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    near = d.argmin(1)
    for t in np.flatnonzero(bvis & (rng.random(T) < cfg.ghost_occlusion_rate)):
        if not np.isfinite(d[t, near[t]]) or d[t, near[t]] > 3:
            continue
        L = int(rng.uniform(*cfg.ghost_occlusion_s) * FPS)
        Q[t:t + L, near[t]] = np.nan
    for t in np.flatnonzero(bvis & (rng.random(T) < cfg.ghost_offset_rate)):
        L = int(rng.uniform(*cfg.ghost_offset_s) * FPS)
        ang = rng.uniform(0, 2 * np.pi)
        off = rng.uniform(*cfg.ghost_offset_m) * np.array([np.cos(ang), np.sin(ang)], np.float32)
        ramp = np.sin(np.linspace(0, np.pi, max(L, 2)))[:, None].astype(np.float32)   # in and out smoothly
        seg = slice(t, min(t + L, T))
        B[seg, :2] += off * ramp[: seg.stop - seg.start]
    B[:, 2] = _speed(B[:, :2], half)
    return Q, B


def stoppage_proxy(P, visible, half):
    """Players' mean speed smoothed over 2 s below SLOW m/s (tracking only, as in S1.3)."""
    sp = np.where(visible, P[..., 2], np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(sp, axis=1)
    return _moving_average(mean, 2 * FPS) < SLOW


def degrade(m: Match, cfg: DegradeConfig | None = None, seed: int = 0) -> Match:
    cfg = cfg or DegradeConfig()
    params = cfg.params or load_params()
    rng = np.random.default_rng(seed)
    T, S = m.P.shape[:2]
    comps = set(cfg.components)
    P = m.P.copy()
    half = m.half

    # 1. smoothing
    if "smooth" in comps and cfg.smooth_window > 1:
        for s in range(S):
            for j in (0, 1):
                P[:, s, j] = _moving_average(P[:, s, j].astype(np.float64), cfg.smooth_window)
        P[..., 2] = _speed(P[..., :2], half)

    # 2. absences (per player) and team episodes
    vis = np.isfinite(P[..., 0])
    if "absences" in comps:
        med = cfg.absence_median_s * FPS
        for s in range(S):
            vis[:, s] &= ~_runs_mask(T, rng, cfg.absence_rate, lambda: rng.lognormal(np.log(med), 0.8))
        for team in (HOME, AWAY):
            idx = np.flatnonzero(m.slot_team == team)
            for s0 in np.flatnonzero(rng.random(T) < cfg.episode_rate):
                L = int(rng.uniform(*cfg.episode_len_s) * FPS)
                on = idx[vis[s0, idx]]
                if len(on) == 0:
                    continue
                k = min(len(on), rng.integers(cfg.episode_players[0], cfg.episode_players[1] + 1))
                vis[s0:s0 + L, rng.choice(on, k, replace=False)] = False

    # 3. fragments: visible runs per player, extra cuts, optional lingering duplicates
    frags = []  # (start, stop, source player, offset vector or None)
    for s in range(S):
        v = vis[:, s]
        edges = np.flatnonzero(np.diff(np.r_[0, v.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            cuts = [a]
            if "fragments" in comps:
                n_cut = rng.poisson(cfg.cut_rate * (b - a))
                cuts += sorted(rng.integers(a + 1, b, n_cut).tolist()) if b - a > 2 and n_cut else []
            cuts.append(b)
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                frags.append((c0, c1, s, None))
                if "duplicates" in comps and c1 < b and rng.random() < cfg.dup_prob:
                    L = int(rng.uniform(*cfg.dup_len_s) * FPS)
                    ang = rng.uniform(0, 2 * np.pi)
                    off = rng.uniform(*cfg.dup_offset_m) * np.array([np.cos(ang), np.sin(ang)])
                    frags.append((c1, min(c1 + L, b), s, off))
    frags.sort(key=lambda f: f[0])
    # team-fixed pools: slots 0..nH-1 are H, the rest A; a pool grows if it runs out. The first slot of
    # each pool is reserved for that team's goalkeeper(s), as in the ATD (stable keeper slot, S1.4):
    # keeper fragments always go there and no outfield fragment ever uses it. A keeper's lingering
    # duplicate goes to the normal pool (it is a duplicate, not the keeper).
    gk_slot = {HOME: 0, AWAY: cfg.pool_slots[0]}
    pools = {HOME: list(range(1, cfg.pool_slots[0])), AWAY: list(range(cfg.pool_slots[0] + 1, sum(cfg.pool_slots)))}
    free_at = np.zeros(sum(cfg.pool_slots), int)
    slot_team = np.r_[np.full(cfg.pool_slots[0], HOME), np.full(cfg.pool_slots[1], AWAY)].astype(np.int8)
    assign = []
    for a, b, s, off in frags:
        team = int(m.slot_team[s])
        if m.slot_is_gk[s] and off is None and free_at[gk_slot[team]] <= a:
            k = gk_slot[team]
        else:
            cand = [k for k in pools[team] if free_at[k] <= a]
            if not cand:
                free_at = np.r_[free_at, 0]
                slot_team = np.r_[slot_team, team].astype(np.int8)
                pools[team].append(len(free_at) - 1)
                cand = [len(free_at) - 1]
            k = cand[0]
        free_at[k] = b
        assign.append((a, b, s, off, k))
    Q = np.full((T, len(free_at), 3), np.nan, np.float32)
    for a, b, s, off, k in assign:
        Q[a:b, k] = P[a:b, s]
        if off is not None:
            Q[a:b, k, :2] += off.astype(np.float32)
    slot_is_gk = np.zeros(len(free_at), bool)
    for team in (HOME, AWAY):
        if np.any(m.slot_is_gk & (m.slot_team == team)):
            slot_is_gk[gk_slot[team]] = True

    # 4. ball gaps (context hazard, tracking-only context)
    B = m.B.copy()
    if "ball" in comps:
        qvis = np.isfinite(Q[..., 0])
        slow = stoppage_proxy(Q, qvis, half)
        bxy = B[:, :2]
        bspeed = np.r_[np.nan, np.linalg.norm(np.diff(bxy, axis=0), axis=1) * FPS]
        d = np.linalg.norm(Q[..., :2] - bxy[:, None, :], axis=-1)
        crowd = np.nansum(d < 3, axis=1)
        sp_edges = params["gap_start_rate_by_ball_speed"]["edges_ms"]
        sp_rate = np.asarray(params["gap_start_rate_by_ball_speed"]["rate"])
        cr_edges = params["gap_start_rate_by_players_within_3m"]["edges"]
        cr_rate = np.asarray(params["gap_start_rate_by_players_within_3m"]["rate"])
        f_speed = sp_rate[np.clip(np.digitize(np.nan_to_num(bspeed), sp_edges) - 1, 0, len(sp_rate) - 1)] / sp_rate[:5].mean()
        f_crowd = cr_rate[np.clip(np.digitize(crowd, cr_edges) - 1, 0, len(cr_rate) - 1)] / cr_rate[:4].mean()
        hazard = f_speed * f_crowd * np.where(slow, cfg.gap_rate_slow, cfg.gap_rate_moving)
        hist = params["gap_length_frames_hist"]
        moving_counts = np.asarray(hist["moving_play"], float)
        slow_counts = np.maximum(np.asarray(hist["all"], float) - moving_counts, 0)
        sample_moving = _hist_sampler(rng, hist["edges"], moving_counts)
        sample_slow = _hist_sampler(rng, hist["edges"], slow_counts)
        gap = np.zeros(T, bool)
        t = 0
        u = rng.random(T)
        min_vis = int(cfg.min_visible_s * FPS)
        while t < T:
            if u[t] < hazard[t]:
                L = max(1, int(cfg.gap_len_scale * (sample_slow() if slow[t] else sample_moving())))
                gap[t:t + L] = True
                t += L + min_vis
            t += 1
        # static ball: hide rests longer than static_min_s (from that point until the ball moves)
        sm = np.convolve(np.nan_to_num(bspeed, nan=99.0), np.ones(5) / 5, "same")
        rest = sm < cfg.static_speed
        edges = np.flatnonzero(np.diff(np.r_[0, rest.astype(int), 0]))
        lag = int(cfg.static_min_s * FPS)
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > lag and rng.random() < cfg.static_hide_prob:
                gap[a + lag:b] = True
        # handled ball: Sportec freezes the ball during stoppages (speed exactly 0 on ~99.8 % of visible
        # dead-ball frames) while the ATD ball keeps moving with the players placing it (median 2.7 m/s
        # on public D frames, never 0). On every rest of the ORIGINAL ball the visible ball now follows
        # the nearest player at its start, with a small offset and noise; rests are hidden as above.
        if cfg.handled_ball and cfg.dead_ball_model == "atd":
            B = _dead_ball_motion(B, Q, rest, rng, cfg, m.pitch)
        elif cfg.handled_ball:
            B = _handled_ball(B, Q, rest, rng, cfg)
        B[gap] = np.nan

    # 5. blackouts
    if "blackouts" in comps:
        n = rng.poisson(cfg.blackouts_per_90 * T / (90 * 60 * FPS))
        for _ in range(n):
            L = int(rng.uniform(*cfg.blackout_len_s) * FPS)
            s0 = int(rng.integers(0, max(1, T - L)))
            keep = rng.random(Q.shape[1]) < 0.12
            Q[s0:s0 + L, ~keep] = np.nan
            B[s0:s0 + L] = np.nan

    # 6. ATD-like piecewise-linear ball (v5); own random stream so earlier configs are reproduced exactly
    if cfg.ball_interp:
        from src.data.ball_interp import interp_params, interpolate
        B = interpolate(B, half, np.random.default_rng([seed, 5]), interp_params(), cfg.interp_dev_max)
        B[:, 2] = _speed(B[:, :2], half)

    # 7. ball-player incoherence (v6); own random stream so earlier configs are reproduced exactly
    if cfg.ghost_occlusion_rate > 0 or cfg.ghost_offset_rate > 0:
        Q, B = _ghosts(Q, B, half, np.random.default_rng([seed, 7]), cfg)

    meta = dict(m.meta)
    meta.update({"degraded": True, "degrade_seed": seed, "degrade_config": {k: v for k, v in cfg.__dict__.items() if k != "params"}})
    return replace(m, P=Q, B=B, slot_team=slot_team, slot_is_gk=slot_is_gk, meta=meta)
