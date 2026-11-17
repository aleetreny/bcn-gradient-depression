"""Post-processing of decoded labels.

M6, short D as a separator: where the change-of-team probability peaks above `threshold` but the
decoded labels keep the same team for at least `guard` frames on both sides, insert a D of `length`
frames centred on the peak. A long predicted possession that truly was two (with a short possession
of the opponent in between, or a contested moment) is split into two spells, which can each match
the truth in spellF1 even if the short possession itself is missed; a wrong insertion costs little,
because D frames inside a spell only shorten it.
"""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, DEAD, HOME


def short_dead_separator(pred: np.ndarray, p_change: np.ndarray, half: np.ndarray, threshold: float,
                         length: int, guard: int = 12, min_distance: int = 25) -> np.ndarray:
    out = pred.copy()
    T = len(pred)
    if threshold >= 1 or length <= 0:
        return out
    pc = p_change
    # local maxima above threshold
    peak = (pc >= threshold) & (pc >= np.r_[pc[1:], 0]) & (pc > np.r_[0, pc[:-1]])
    last = -10 ** 9
    for t in np.flatnonzero(peak):
        if t - last < min_distance:
            continue
        a, b = max(0, t - guard), min(T, t + guard + 1)
        seg = pred[a:b]
        team = pred[t]
        if team not in (HOME, AWAY) or not np.all(seg == team) or not np.all(half[a:b] == half[t]):
            continue
        s = max(0, t - length // 2)
        out[s:s + length] = DEAD
        last = t
    return out
