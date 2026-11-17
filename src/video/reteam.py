"""Automatic Tracking Slot-Count Input Check (`D3`):
Detect and repair cross-team slot-swap stretches (`12 vs 10` visible players instead of `11 vs 11`)
in automatic tracking data using only the provided input files.

    from src.video.reteam import wrong_team_segments, corroborated, reteam

The automatic tracking (`ATD`) occasionally carries a player under the opposite team slot for a few seconds
(an identity switch across teams), which manifests as impossible `12 vs 10` visible player counts.
`degrade_v3` never simulates this, and team-shape features (lines, width, centroid) suffer: on external `J1`
such stretches at `BAR-ATH`'s rate cost ~0.014 `spellF1` (`scripts/noche/oracle_swaps.py`).
`wrong_team_segments` and `corroborated` identify candidate stretches corroborated by the `12 vs 10`
team-count check (`team_counts`), and `reteam` swaps the slot back to restore `11 vs 11` balance.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np


def team_counts(P, team):
    """(T, 2) visible players per team."""
    team = np.asarray(team)
    return np.stack([np.isfinite(P[:, team == t, 0]).sum(1) for t in (0, 1)], 1)


def wrong_team_segments(fr, slot, vote, P, team, win=15, min_votes=8, skip=None):
    """[(slot, a, b)]: frame stretches [a, b) of one visible run where a centred rolling majority of `win` sure
    shirt votes contradicts the slot's team, with at least `min_votes` votes in the stretch.

    fr, slot, vote: one entry per vote (frame, ATD slot, colour class 0/1). skip: slots never moved (the
    goalkeepers: the ATD keeps them in a fixed slot, and a keeper's kit can resemble the other team's shirt)."""
    team = np.asarray(team)
    out = []
    for s in range(P.shape[1]):
        if skip is not None and skip[s]:
            continue
        v = np.isfinite(P[:, s, 0])
        e = np.flatnonzero(np.diff(np.r_[0, v.astype(np.int8), 0]))
        k = np.flatnonzero(slot == s)
        k = k[np.argsort(fr[k], kind="stable")]
        for a, b in zip(e[::2], e[1::2]):
            kk = k[(fr[k] >= a) & (fr[k] < b)]
            if len(kk) < min_votes:
                continue
            dis = (vote[kk] != team[s]).astype(float)
            c = np.r_[0, np.cumsum(dis)]
            i = np.arange(len(dis))
            lo_i, hi_i = np.maximum(i - win // 2, 0), np.minimum(i + win // 2 + 1, len(dis))
            bad = (c[hi_i] - c[lo_i]) / (hi_i - lo_i) > 0.5      # centred rolling majority, clipped at the ends
            ee = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]))
            t = fr[kk]
            for i0, i1 in zip(ee[::2], ee[1::2]):
                if i1 - i0 < min_votes:
                    continue
                lo = a if i0 == 0 else (t[i0 - 1] + t[i0] + 1) // 2
                hi = b if i1 == len(kk) else (t[i1 - 1] + t[i1] + 1) // 2
                out.append((s, int(lo), int(hi)))
    return out


def corroborated(segs, P, team, min_share=0.5):
    """Keep the stretches the team counts agree with: on most of their frames the slot's team shows at least two
    players more than the other team, so moving the slot turns 12-v-10 into 11-v-11."""
    team = np.asarray(team)
    cnt = team_counts(P, team)
    return [(s, a, b) for s, a, b in segs if np.mean(cnt[a:b, team[s]] - cnt[a:b, 1 - team[s]] >= 2) > min_share]


def paired(segs, P, team, min_overlap=0.5, max_start_m=3.0):
    """Keep the stretches with a partner on the other team: an overlapping stretch (at least `min_overlap` of the
    shorter one) whose player stood within `max_start_m` when the overlap began. Two players who cross and swap
    tracks leave the counts balanced, so the count check cannot see them; two independent shirt stretches that
    start together can."""
    team = np.asarray(team)
    keep = []
    for i, (s, a, b) in enumerate(segs):
        for j, (s2, a2, b2) in enumerate(segs):
            if i == j or team[s2] == team[s]:
                continue
            if min(b, b2) - max(a, a2) < min_overlap * min(b - a, b2 - a2):
                continue
            t0 = max(a, a2)
            d = np.linalg.norm(P[t0, s, :2] - P[t0, s2, :2])
            if np.isfinite(d) and d <= max_start_m:
                keep.append((s, a, b))
                break
    return keep


def reteam(m, segs, gap=None):
    """Match with every stretch moved to a slot of the other team (the old slot is emptied there).

    gap=None gives each stretch a new slot of its own. With gap (frames), stretches of the same new team share a
    slot when they are at least `gap` frames apart, so a match with hundreds of stretches keeps a few dozen
    slots: features that compare every pair of slots grow with the square of their number."""
    P, team = m.P.copy(), np.asarray(m.slot_team).copy()
    gk = np.asarray(m.slot_is_gk).copy()
    extra, extra_team, free_at = [], [], []
    for s, a, b in (segs if gap is None else sorted(segs, key=lambda x: x[1])):
        new = 1 - team[s]
        k = next((i for i in range(len(extra)) if gap is not None and extra_team[i] == new
                  and free_at[i] + gap <= a), None)
        if k is None:
            extra.append(np.full((len(P), P.shape[2]), np.nan, P.dtype))
            extra_team.append(new)
            free_at.append(0)
            k = len(extra) - 1
        extra[k][a:b] = P[a:b, s]
        free_at[k] = b
        P[a:b, s] = np.nan
    if extra:
        P = np.concatenate([P, np.stack(extra, 1)], 1)
        team = np.r_[team, np.array(extra_team)].astype(np.int8)
        gk = np.r_[gk, np.zeros(len(extra), bool)]
    return replace(m, P=P, slot_team=team, slot_is_gk=gk)


__all__ = ["team_counts", "wrong_team_segments", "corroborated", "paired", "reteam"]
