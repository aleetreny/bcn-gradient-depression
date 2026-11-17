"""Oracle repairs: fix ONE error type of a prediction with the truth, to measure what it is worth.

All functions take common codes (0 H, 1 A, 2 D, -1 ungraded truth) and return a repaired copy of
`pred`. They never touch frames where the truth is ungraded.

- perfect_dead:    D exactly where the truth has D; alive frames keep the prediction's team
                   (frames the prediction called D get its last team, carried forward).
- perfect_team:    on frames the prediction calls alive and the truth is alive, the true team.
- perfect_timing:  every predicted label change that matches a true change to the same label
                   within +/- MAX_LAG frames is moved onto the true frame (order preserved).
- no_spurious:     predicted possessions with no same-team true spell at IoU >= 0.3 are merged
                   into the preceding segment.
- short_spells:    true possessions shorter than 2 s that the prediction misses (IoU < 0.3) are
                   painted in -- including over predicted D, so it also repairs false stoppages.
- short_spells_alive_only: the same, but predicted D frames are left untouched.
Each repair is an intervention on ONE prediction; the gains overlap and are not ceilings.
"""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, DEAD, FPS, HOME, possession_spells
from src.eval.errors import IOU, _iou_best

MAX_LAG = 3 * FPS
SHORT = 2 * FPS


def _carry_team(pred):
    """pred with D replaced by the last team (first team back-filled)."""
    team = np.where(np.isin(pred, (HOME, AWAY)), pred, -1)
    idx = np.maximum.accumulate(np.where(team >= 0, np.arange(len(team)), -1))
    out = np.where(idx >= 0, team[np.maximum(idx, 0)], -1)
    first = np.flatnonzero(out >= 0)
    if len(first):
        out[:first[0]] = out[first[0]]
    else:
        out[:] = HOME
    return out.astype(np.int8)


def perfect_dead(gt, pred, half):
    out = np.where(gt == DEAD, DEAD, _carry_team(pred)).astype(np.int8)
    return np.where(gt >= 0, out, pred).astype(np.int8)


def perfect_team(gt, pred, half):
    fix = np.isin(pred, (HOME, AWAY)) & np.isin(gt, (HOME, AWAY))
    return np.where(fix, gt, pred).astype(np.int8)


def _changes(codes, half):
    idx = np.flatnonzero((codes[1:] != codes[:-1]) & (half[1:] == half[:-1]) & (half[1:] > 0)) + 1
    return idx, codes[idx]


def perfect_timing(gt, pred, half):
    gf, gk = _changes(np.where(gt >= 0, gt, -9), half)
    pf, pk = _changes(pred, half)
    new = pf.copy()
    pairs = []
    for i, (f, k) in enumerate(zip(pf, pk)):
        cand = np.flatnonzero((gk == k) & (np.abs(gf - f) <= MAX_LAG))
        pairs += [(abs(int(gf[j]) - int(f)), i, j) for j in cand]
    moved = np.zeros(len(pf), bool)
    used = np.zeros(len(gf), bool)
    for _, i, j in sorted(pairs):  # nearest pairs first, one-to-one
        if not (moved[i] or used[j]):
            new[i], moved[i], used[j] = gf[j], True, True
    out = pred.copy()
    for h in np.unique(half[half > 0]):
        rows = np.flatnonzero(half == h)
        sel = np.flatnonzero(half[pf] == h)
        pos = new[sel].copy()
        changed = True
        while changed:  # revert moves that break the order of boundaries
            changed = False
            for n in range(len(pos)):
                lo = pos[n - 1] if n > 0 else rows[0]
                hi = pos[n + 1] if n + 1 < len(pos) else rows[-1] + 1
                if not lo < pos[n] < hi and pos[n] != pf[sel[n]]:
                    pos[n] = pf[sel[n]]
                    changed = True
        starts = np.r_[rows[0], pos]
        labels = np.r_[pred[rows[0]], pk[sel]]
        stops = np.r_[pos, rows[-1] + 1]
        for a_, b_, k in zip(starts, stops, labels):
            out[a_:b_] = k
    return np.where(gt >= 0, out, pred).astype(np.int8)


def no_spurious(gt, pred, half):
    p_sp = possession_spells(pred, half)
    g_sp = possession_spells(np.where(gt >= 0, gt, -1), half)
    best = _iou_best(p_sp, g_sp)
    out = pred.copy()
    for s, e, b in zip(p_sp[0], p_sp[1], best):
        if b < IOU:
            fill = out[s - 1] if s > 0 and half[s - 1] == half[s] else (pred[e] if e < len(pred) else DEAD)
            out[s:e] = fill
    return np.where(gt >= 0, out, pred).astype(np.int8)


def short_spells(gt, pred, half, keep_dead=False):
    """Paint missed short true possessions. With keep_dead, only frames the prediction calls alive
    are repainted, so predicted D (and its boundaries) is left untouched."""
    g_sp = possession_spells(np.where(gt >= 0, gt, -1), half)
    p_sp = possession_spells(pred, half)
    best = _iou_best(g_sp, p_sp)
    out = pred.copy()
    for s, e, t, b in zip(*g_sp, best):
        if e - s < SHORT and b < IOU:
            if keep_dead:
                seg = out[s:e]
                seg[seg != DEAD] = t
            else:
                out[s:e] = t
    return out.astype(np.int8)


def short_spells_alive_only(gt, pred, half):
    return short_spells(gt, pred, half, keep_dead=True)


def short_spell_recovery(gt, pred, repaired, half):
    """(missed short true possessions before, still unmatched after the repair)."""
    g_sp = possession_spells(np.where(gt >= 0, gt, -1), half)
    short = (g_sp[1] - g_sp[0]) < SHORT
    before = _iou_best(g_sp, possession_spells(pred, half)) < IOU
    after = _iou_best(g_sp, possession_spells(repaired, half)) < IOU
    return int((short & before).sum()), int((short & before & after).sum())


ORACLES = {"perfect D": perfect_dead, "perfect team (alive frames)": perfect_team,
           "perfect change timing (±3 s)": perfect_timing, "no spurious possessions": no_spurious,
           "short possessions recovered": short_spells,
           "short possessions recovered (alive frames only)": short_spells_alive_only}
