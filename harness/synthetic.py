"""Synthetic matches in the common Match format: labelled, tiny, deterministic, and free to commit.

They are not realistic football. They exist so judges, contracts, decoders and the runner can be
tested end-to-end in CI, where no competition or external data is ever available.

    from harness.synthetic import make_match
    m = make_match(seed=0, minutes=4)
"""
from __future__ import annotations

import numpy as np

from src.data.common import AWAY, DEAD, FPS, HOME, Match

PITCH = (105.0, 68.0)


def _spells(rng, n: int) -> np.ndarray:
    labels = np.empty(n, np.int8)
    t, state = 0, HOME
    while t < n:
        mean = 6.0 if state == DEAD else 9.0
        d = int(FPS * (0.6 + rng.exponential(mean)))
        labels[t:t + d] = state
        t += d
        if state == DEAD:
            state = int(rng.integers(0, 2))
        else:
            state = DEAD if rng.random() < 0.3 else 1 - state
    return labels


def make_match(seed: int = 0, minutes: float = 4.0, slots_per_team: int = 11) -> Match:
    rng = np.random.default_rng(seed)
    T = int(minutes * 60 * FPS)
    half = np.where(np.arange(T) < T // 2, 1, 2).astype(np.int8)
    labels = _spells(rng, T)
    L, W = PITCH
    S = 2 * slots_per_team
    slot_team = np.repeat([HOME, AWAY], slots_per_team).astype(np.int8)
    base = np.c_[rng.uniform(5, L - 5, S), rng.uniform(5, W - 5, S)]
    walk = np.cumsum(rng.normal(0, 0.08, (T, S, 2)), axis=0)
    xy = np.clip(base + walk, 0, [L, W])
    P = np.empty((T, S, 3), np.float32)
    P[..., :2] = xy
    P[1:, :, 2] = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * FPS
    P[0, :, 2] = 0
    P[rng.random((T, S)) < 0.03] = np.nan            # dropped slots
    # the ball sits on a carrier of the possessing team, who changes every ~2 s
    B = np.full((T, 3), np.nan, np.float32)
    carrier = -1
    for t in range(T):
        if labels[t] == DEAD:
            carrier = -1
            continue
        team_slots = np.flatnonzero(slot_team == labels[t])
        if carrier not in team_slots or rng.random() < 1 / (2 * FPS):
            carrier = int(rng.choice(team_slots))
        B[t, :2] = xy[t, carrier] + rng.normal(0, 0.6, 2)
    B[:, 2] = 0
    hide = (rng.random(T) < 0.15) | ((labels == DEAD) & (rng.random(T) < 0.9))
    B[hide] = np.nan
    return Match(P=P, B=B, slot_team=slot_team, slot_is_gk=np.zeros(S, bool), half=half,
                 graded=half > 0, attack_sign=np.where(half == 1, 1, -1).astype(np.int8), pitch=PITCH,
                 home_name="SynHome", away_name="SynAway", source="synthetic", labels=labels,
                 meta={"seed": seed}, direction_normalized=False)
