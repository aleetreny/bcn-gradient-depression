"""Semi-Markov Viterbi decoder for H/A/D with switch costs and minimum durations (Annex A [C] of plan 05).

    from src.decode.viterbi_hsmm import DecoderParams, decode
    labels = decode(proba, half, DecoderParams(lam_ha=8, lam_ld=6, lam_dl=6, dmin_team=10, dmin_dead=25))

score(path) = sum_t log P(s_t | x_t) - sum_t [s_t != s_{t-1}] * cost(s_{t-1} -> s_t)
cost(H<->A) = lam_ha, cost(H/A -> D) = lam_ld, cost(D -> H/A) = lam_dl. With a per-frame change
probability (M3) and beta_ha > 0: cost(H<->A, t) = max(0, lam_ha - beta_ha * logit P_change(t)).
Minimum durations are enforced with chains of sub-states (a class can only be left from the last
sub-state of its chain), so a team spell lasts >= dmin_team frames and a dead spell >= dmin_dead.
Each half is decoded independently. Frames with half == 0 are returned as D.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numba import njit

from src.data.common import DEAD


@dataclass
class DecoderParams:
    lam_ha: float = 8.0
    lam_ld: float = 6.0
    lam_dl: float = 6.0
    dmin_team: int = 10
    dmin_dead: int = 25
    beta_ha: float = 0.0    # M3: H<->A cost at t is max(0, lam_ha - beta_ha * logit P_change(t))

    def to_dict(self):
        return asdict(self)


@njit(cache=True)
def _viterbi_chain(logp, dmins, trans, ha_bonus):
    T = logp.shape[0]
    starts = np.zeros(3, np.int64)
    for c in range(1, 3):
        starts[c] = starts[c - 1] + dmins[c - 1]
    S = starts[2] + dmins[2]
    cls = np.empty(S, np.int64)
    last = np.empty(3, np.int64)
    for c in range(3):
        for k in range(dmins[c]):
            cls[starts[c] + k] = c
        last[c] = starts[c] + dmins[c] - 1
    NEG = -1e18
    score = np.full(S, NEG)
    for c in range(3):
        score[starts[c]] = logp[0, c]
    back = np.empty((T, S), np.int32)
    back[0, :] = -1
    new = np.empty(S)
    for t in range(1, T):
        for s in range(S):
            c = cls[s]
            best, arg = NEG, -1
            if s == starts[c]:
                # enter class c from the last sub-state of another class
                for c2 in range(3):
                    if c2 != c:
                        cost = trans[c2, c]
                        if c2 != 2 and c != 2:
                            cost = max(0.0, cost - ha_bonus[t])
                        v = score[last[c2]] - cost
                        if v > best:
                            best, arg = v, last[c2]
                if dmins[c] == 1:  # chain of length one: also stay
                    if score[s] > best:
                        best, arg = score[s], s
            else:
                best, arg = score[s - 1], s - 1
                if s == last[c] and score[s] > best:
                    best, arg = score[s], s
            new[s] = best + logp[t, c]
            back[t, s] = arg
        for s in range(S):
            score[s] = new[s]
    # backtrack from the best last sub-state (a spell may not end unfinished)
    s = last[0]
    for c in range(1, 3):
        if score[last[c]] > score[s]:
            s = last[c]
    if score[s] <= NEG / 2:  # half shorter than the minimum durations: take any state
        s = int(np.argmax(score))
    path = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = cls[s]
        s = back[t, s]
        if s < 0 and t > 0:
            s = 0
    return path


def decode(proba: np.ndarray, half: np.ndarray, p: DecoderParams, eps: float = 1e-6,
           p_change: np.ndarray | None = None) -> np.ndarray:
    """proba: (T, 3) columns H, A, D. Returns int8 labels (0 H, 1 A, 2 D).

    p_change (optional, M3): per-frame probability of a change of team; with beta_ha > 0 the H<->A
    switch cost becomes time dependent, cheaper where a change is likely and dearer where it is not.
    """
    logp = np.log(np.clip(proba.astype(np.float64), eps, 1.0))
    trans = np.array([[0.0, p.lam_ha, p.lam_ld],
                      [p.lam_ha, 0.0, p.lam_ld],
                      [p.lam_dl, p.lam_dl, 0.0]])
    dmins = np.array([max(1, int(p.dmin_team)), max(1, int(p.dmin_team)), max(1, int(p.dmin_dead))], np.int64)
    if p_change is not None and p.beta_ha != 0:
        q = np.clip(p_change.astype(np.float64), 1e-4, 1 - 1e-4)
        bonus = p.beta_ha * np.log(q / (1 - q))
    else:
        bonus = np.zeros(len(proba))
    out = np.full(len(proba), DEAD, np.int8)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        out[idx] = _viterbi_chain(logp[idx], dmins, trans, bonus[idx]).astype(np.int8)
    return out
