"""EXP-01: Semi-Markov Viterbi Decoder v2 (`viterbi_hsmm_v2.py`).

Fixes and upgrades over `src/decode/viterbi_hsmm.py`:
1. Base-Rate Calibrated Log-Odds Surprise `s(p, pi0)`:
   In `viterbi_hsmm.py`, `cost = max(0, lam_ha - beta_ha * logit(p_change[t]))`. Because `p_change < 0.5`
   on ~98% of frames (base rate `pi0 ~ 0.035`), `logit(p_change)` is negative even during genuine turnovers
   (`p = 0.25 -> logit = -1.10`), so `-beta_ha * logit(p_change)` added penalty instead of discounting it,
   and random fluctuations in tiny background probabilities (`0.001` vs `0.01`) caused massive 15-nat swings
   in switch cost.
   Here we define the positive log-odds surprise above base rate `pi0`:
       s(p, pi0) = max(0.0, logit(p) - logit(pi0))
   so background frames (`p <= pi0`) have exact constant cost `lam`, and elevated boundary probabilities
   (`p > pi0`) strictly discount the transition cost.

2. 4-Directional Edge Discounts (`p_trans_4d` of shape `(T, 4)`: `[p_H_to_A, p_A_to_H, p_Live_to_D, p_D_to_Live]`):
   - `H -> A` cost at frame `t`: `max(0.0, lam_ha - beta_ha * s(p_H_to_A[t], pi0_ha))`
   - `A -> H` cost at frame `t`: `max(0.0, lam_ha - beta_ha * s(p_A_to_H[t], pi0_ha))`
   - `H/A -> D` cost at frame `t`: `max(0.0, lam_ld - beta_to_d * s(p_Live_to_D[t], pi0_d))`
   - `D -> H/A` cost at frame `t`: `max(0.0, lam_dl - beta_from_d * s(p_D_to_Live[t], pi0_d))`
   Supports 1D `p_change`, 4D `p_trans_4d`, or a blend of both when both are supplied.

3. 2-Stage Piecewise Duration Potential:
   In addition to hard minimum duration floors `dmin_team` and `dmin_dead`, supports a second piecewise
   stage `[dmin_team, short_dur_cap]` with linear exit penalty/bonus `gamma_short` for short live spells,
   allowing genuine 1.0s-2.5s interceptions to survive when bounded by strong transition spikes without
   permitting sub-second random flicker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numba import njit

from src.data.common import DEAD
from src.decode.viterbi_hsmm import DecoderParams


def _logit(p: np.ndarray | float, eps: float = 1e-4) -> np.ndarray | float:
    q = np.clip(p, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def surprise_logodds(p: np.ndarray, pi0: float, eps: float = 1e-4) -> np.ndarray:
    """Positive log-odds surprise above base rate pi0: s(p, pi0) = max(0.0, logit(p) - logit(pi0))."""
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    pi0_c = float(np.clip(pi0, eps, 1.0 - eps))
    lo_0 = np.log(pi0_c / (1.0 - pi0_c))
    return np.maximum(0.0, np.log(q / (1.0 - q)) - lo_0)


@dataclass
class DecoderParamsV2:
    lam_ha: float = 23.70
    lam_ld: float = 10.12
    lam_dl: float = 32.40
    dmin_team: int = 11
    dmin_dead: int = 75
    beta_ha: float = 5.54
    beta_to_d: float = 0.0
    beta_from_d: float = 0.0
    pi0_ha: float = 0.001
    pi0_d: float = 0.001
    short_dur_cap: int = 11
    gamma_short: float = 0.0
    w_b4d_1d: float = 0.25
    w_dir_ha: float = 0.35
    tau_dir: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_legacy(
        cls,
        dp: DecoderParams,
        pi0_ha: float = 0.001,
        pi0_d: float = 0.001,
        beta_to_d: float = 0.0,
        beta_from_d: float = 0.0,
        short_dur_cap: int | None = None,
        gamma_short: float = 0.0,
        w_b4d_1d: float = 0.25,
        w_dir_ha: float = 0.35,
        tau_dir: float = 0.02,
    ) -> "DecoderParamsV2":
        """Convert a legacy DecoderParams (calibrated against unshifted logit) to DecoderParamsV2.

        At p = pi0_ha, legacy cost was `dp.lam_ha - dp.beta_ha * logit(pi0_ha)`.
        Setting `lam_ha = dp.lam_ha - dp.beta_ha * logit(pi0_ha)` matches the legacy cost for all
        `p >= pi0_ha` while clamping the artificial penalty explosion when `p < pi0_ha`.
        """
        lo_0 = float(_logit(pi0_ha))
        lam_ha_v2 = float(dp.lam_ha - dp.beta_ha * lo_0)
        return cls(
            lam_ha=lam_ha_v2,
            lam_ld=float(dp.lam_ld),
            lam_dl=float(dp.lam_dl),
            dmin_team=int(dp.dmin_team),
            dmin_dead=int(dp.dmin_dead),
            beta_ha=float(dp.beta_ha),
            beta_to_d=float(beta_to_d),
            beta_from_d=float(beta_from_d),
            pi0_ha=float(pi0_ha),
            pi0_d=float(pi0_d),
            short_dur_cap=int(dp.dmin_team if short_dur_cap is None else short_dur_cap),
            gamma_short=float(gamma_short),
            w_b4d_1d=float(w_b4d_1d),
            w_dir_ha=float(w_dir_ha),
            tau_dir=float(tau_dir),
        )


@njit(cache=True)
def _viterbi_chain_v2(
    logp: np.ndarray,
    dmins: np.ndarray,
    chain_lens: np.ndarray,
    exit_pen: np.ndarray,
    trans: np.ndarray,
    disc_h2a: np.ndarray,
    disc_a2h: np.ndarray,
    disc_l2d: np.ndarray,
    disc_d2l: np.ndarray,
) -> np.ndarray:
    """Semi-Markov Viterbi with 4-directional edge discounts and 2-stage piecewise duration potential."""
    T = logp.shape[0]
    starts = np.zeros(3, np.int64)
    for c in range(1, 3):
        starts[c] = starts[c - 1] + chain_lens[c - 1]
    S = starts[2] + chain_lens[2]

    cls = np.empty(S, np.int64)
    last = np.empty(3, np.int64)
    min_exit = np.empty(3, np.int64)
    for c in range(3):
        for k in range(chain_lens[c]):
            cls[starts[c] + k] = c
        last[c] = starts[c] + chain_lens[c] - 1
        min_exit[c] = starts[c] + dmins[c] - 1

    NEG = -1e18
    score = np.full(S, NEG)
    for c in range(3):
        score[starts[c]] = logp[0, c]

    back = np.empty((T, S), np.int32)
    back[0, :] = -1
    new = np.empty(S)

    best_exit_val = np.empty(3, np.float64)
    best_exit_arg = np.empty(3, np.int64)

    for t in range(1, T):
        # Precompute best exit sub-state for each class c2 in {0, 1, 2}
        for c2 in range(3):
            bv = NEG
            ba = last[c2]
            for s2 in range(min_exit[c2], last[c2] + 1):
                v2 = score[s2] - exit_pen[s2]
                if v2 > bv:
                    bv = v2
                    ba = s2
            best_exit_val[c2] = bv
            best_exit_arg[c2] = ba

        for s in range(S):
            c = cls[s]
            best, arg = NEG, -1
            if s == starts[c]:
                # Enter class c from any valid exit sub-state of another class c2 != c
                for c2 in range(3):
                    if c2 != c:
                        cost = trans[c2, c]
                        if c2 == 0 and c == 1:
                            cost = max(0.0, cost - disc_h2a[t])
                        elif c2 == 1 and c == 0:
                            cost = max(0.0, cost - disc_a2h[t])
                        elif c2 != 2 and c == 2:
                            cost = max(0.0, cost - disc_l2d[t])
                        elif c2 == 2 and c != 2:
                            cost = max(0.0, cost - disc_d2l[t])
                        v = best_exit_val[c2] - cost
                        if v > best:
                            best, arg = v, best_exit_arg[c2]
                if chain_lens[c] == 1:
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

    # Backtrack from the best valid exit sub-state at T - 1
    best_final = NEG
    s_best = last[0]
    for c in range(3):
        for s2 in range(min_exit[c], last[c] + 1):
            v2 = score[s2] - exit_pen[s2]
            if v2 > best_final:
                best_final = v2
                s_best = s2
    if best_final <= NEG / 2:
        s_best = int(np.argmax(score))

    path = np.empty(T, np.int64)
    s = s_best
    for t in range(T - 1, -1, -1):
        path[t] = cls[s]
        s = back[t, s]
        if s < 0 and t > 0:
            s = 0
    return path


def decode_v2(
    proba: np.ndarray,
    half: np.ndarray,
    p: DecoderParamsV2 | DecoderParams,
    eps: float = 1e-6,
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
) -> np.ndarray:
    """Decode (T, 3) probabilities into int8 labels (0=H, 1=A, 2=D) using viterbi_hsmm_v2.

    Accepts either `DecoderParamsV2` or legacy `DecoderParams` (automatically converted via `from_legacy`).
    Supports 1D `p_change`, 4-column `p_trans_4d` (`[p_H_to_A, p_A_to_H, p_Live_to_D, p_D_to_Live]`), or both.
    """
    if isinstance(p, DecoderParams):
        p = DecoderParamsV2.from_legacy(p)

    T = len(proba)
    logp = np.log(np.clip(proba.astype(np.float64), eps, 1.0))
    trans = np.array(
        [
            [0.0, p.lam_ha, p.lam_ld],
            [p.lam_ha, 0.0, p.lam_ld],
            [p.lam_dl, p.lam_dl, 0.0],
        ],
        dtype=np.float64,
    )

    dmin_team = max(1, int(p.dmin_team))
    dmin_dead = max(1, int(p.dmin_dead))
    short_cap = max(dmin_team, int(p.short_dur_cap)) if p.gamma_short != 0.0 else dmin_team

    dmins = np.array([dmin_team, dmin_team, dmin_dead], dtype=np.int64)
    chain_lens = np.array([short_cap, short_cap, dmin_dead], dtype=np.int64)

    S = int(chain_lens.sum())
    exit_pen = np.zeros(S, dtype=np.float64)
    if short_cap > dmin_team and p.gamma_short != 0.0:
        span = float(short_cap - dmin_team)
        starts_c = (0, short_cap)
        for st in starts_c:
            for k in range(dmin_team - 1, short_cap - 1):
                exit_pen[st + k] = p.gamma_short * float((short_cap - 1) - k) / span

    # Compute 4-directional discounts
    disc_h2a = np.zeros(T, dtype=np.float64)
    disc_a2h = np.zeros(T, dtype=np.float64)
    disc_l2d = np.zeros(T, dtype=np.float64)
    disc_d2l = np.zeros(T, dtype=np.float64)

    if p_trans_4d is not None and p_change is not None:
        # Combine 1D p_change with 4-directional p_H_to_A and p_A_to_H via scale-free directional share ratio
        w_1d = float(p.w_b4d_1d)
        w_dir = float(p.w_dir_ha)
        tau_dir = float(max(1e-4, p.tau_dir))
        p_tot_4d = np.clip(p_trans_4d[:, 0] + p_trans_4d[:, 1], 0.0, 1.0)
        p_base = (1.0 - w_1d) * np.asarray(p_change, dtype=np.float64) + w_1d * p_tot_4d
        denom = 0.5 * p_tot_4d + tau_dir
        r_h2a = (p_trans_4d[:, 0] + tau_dir) / denom
        r_a2h = (p_trans_4d[:, 1] + tau_dir) / denom
        p_eff_h2a = np.clip(p_base * ((1.0 - w_dir) + w_dir * r_h2a), 1e-4, 1.0 - 1e-4)
        p_eff_a2h = np.clip(p_base * ((1.0 - w_dir) + w_dir * r_a2h), 1e-4, 1.0 - 1e-4)
        if p.beta_ha != 0.0:
            disc_h2a = p.beta_ha * surprise_logodds(p_eff_h2a, p.pi0_ha)
            disc_a2h = p.beta_ha * surprise_logodds(p_eff_a2h, p.pi0_ha)
        if p.beta_to_d != 0.0:
            disc_l2d = p.beta_to_d * surprise_logodds(p_trans_4d[:, 2], p.pi0_d)
        if p.beta_from_d != 0.0:
            disc_d2l = p.beta_from_d * surprise_logodds(p_trans_4d[:, 3], p.pi0_d)
    elif p_trans_4d is not None:
        pi0_dir = 0.5 * p.pi0_ha
        if p.beta_ha != 0.0:
            disc_h2a = p.beta_ha * surprise_logodds(p_trans_4d[:, 0], pi0_dir)
            disc_a2h = p.beta_ha * surprise_logodds(p_trans_4d[:, 1], pi0_dir)
        if p.beta_to_d != 0.0:
            disc_l2d = p.beta_to_d * surprise_logodds(p_trans_4d[:, 2], p.pi0_d)
        if p.beta_from_d != 0.0:
            disc_d2l = p.beta_from_d * surprise_logodds(p_trans_4d[:, 3], p.pi0_d)
    elif p_change is not None and p.beta_ha != 0.0:
        s_1d = surprise_logodds(p_change, p.pi0_ha)
        disc_h2a = p.beta_ha * s_1d
        disc_a2h = p.beta_ha * s_1d

    out = np.full(T, DEAD, dtype=np.int8)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        out[idx] = _viterbi_chain_v2(
            logp[idx],
            dmins,
            chain_lens,
            exit_pen,
            trans,
            disc_h2a[idx],
            disc_a2h[idx],
            disc_l2d[idx],
            disc_d2l[idx],
        ).astype(np.int8)
    return out
