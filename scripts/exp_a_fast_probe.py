"""EXP-A Fast Probe: Dissecting decoder & ensemble mathematical improvements on v4 OOFs, Sample, and BAR-ATH."""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import maximum_filter1d, uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_3_frame_models import plausibility
from s3_7_nested_full import build_features
from s3_11_tcn_local import without_dead_margin
from src.data.common import DEAD, HOME, AWAY
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics


@njit(cache=True)
def _viterbi_ext(logp, dmin_team, dmin_dead, d_short, lam_ha, lam_ld, lam_dl, ha_bonus, pair_gamma):
    """Extended Semi-Markov chain with short-spell boundary-pair synergy."""
    T = logp.shape[0]
    L_team = max(dmin_team, d_short)
    L_dead = max(1, dmin_dead)
    starts = np.array([0, L_team, 2 * L_team], np.int64)
    lengths = np.array([L_team, L_team, L_dead], np.int64)
    dmins = np.array([dmin_team, dmin_team, L_dead], np.int64)
    S = 2 * L_team + L_dead

    cls = np.empty(S, np.int64)
    pos = np.empty(S, np.int64)
    last = np.empty(3, np.int64)
    for c in range(3):
        for k in range(lengths[c]):
            cls[starts[c] + k] = c
            pos[starts[c] + k] = k
        last[c] = starts[c] + lengths[c] - 1

    trans = np.array([
        [0.0, lam_ha, lam_ld],
        [lam_ha, 0.0, lam_ld],
        [lam_dl, lam_dl, 0.0],
    ])

    NEG = -1e18
    score = np.full(S, NEG)
    entry_b = np.zeros(S)
    for c in range(3):
        score[starts[c]] = logp[0, c]

    back = np.empty((T, S), np.int32)
    back[0, :] = -1
    new_score = np.empty(S)
    new_entry_b = np.empty(S)

    for t in range(1, T):
        for c in range(3):
            s0 = starts[c]
            best, arg, eb = NEG, -1, 0.0
            for c2 in range(3):
                if c2 == c:
                    continue
                base_cost = trans[c2, c]
                if c2 != 2 and c != 2:
                    cost_now = max(0.0, base_cost - ha_bonus[t])
                    # Can exit c2 from any k >= dmin_team - 1 up to L_team - 1
                    for k in range(dmins[c2] - 1, lengths[c2]):
                        prev_s = starts[c2] + k
                        sc_prev = score[prev_s]
                        if sc_prev <= NEG / 2:
                            continue
                        # If k < lengths[c2] - 1, it is a short spell of duration k+1:
                        # apply boundary-pair bonus if both entry and exit have positive turnover signal
                        syn = 0.0
                        if pair_gamma > 0.0 and k < lengths[c2] - 1:
                            b_in = max(0.0, entry_b[prev_s])
                            b_out = max(0.0, ha_bonus[t])
                            syn = pair_gamma * min(b_in, b_out)
                        eff_cost = max(0.0, cost_now - syn)
                        v = sc_prev - eff_cost
                        if v > best:
                            best, arg, eb = v, prev_s, ha_bonus[t]
                else:
                    # Transition involving D: can exit team state from any k >= dmin_team - 1
                    for k in range(dmins[c2] - 1, lengths[c2]):
                        prev_s = starts[c2] + k
                        v = score[prev_s] - base_cost
                        if v > best:
                            best, arg, eb = v, prev_s, 0.0
            if lengths[c] == 1 and score[s0] > best:
                best, arg, eb = score[s0], s0, entry_b[s0]
            new_score[s0] = best + logp[t, c]
            new_entry_b[s0] = eb
            back[t, s0] = arg

            # Advance along chain
            for k in range(1, lengths[c]):
                s = starts[c] + k
                best, arg, eb = score[s - 1], s - 1, entry_b[s - 1]
                if k == lengths[c] - 1 and score[s] > best:
                    best, arg, eb = score[s], s, 0.0
                new_score[s] = best + logp[t, c]
                new_entry_b[s] = eb
                back[t, s] = arg

        for s in range(S):
            score[s] = new_score[s]
            entry_b[s] = new_entry_b[s]

    # Backtrack from any valid ending sub-state (k >= dmin - 1)
    best_end_s = last[0]
    best_end_v = NEG
    for c in range(3):
        for k in range(dmins[c] - 1, lengths[c]):
            s = starts[c] + k
            if score[s] > best_end_v:
                best_end_v = score[s]
                best_end_s = s
    if best_end_v <= NEG / 2:
        best_end_s = int(np.argmax(score))

    path = np.empty(T, np.int64)
    s = best_end_s
    for t in range(T - 1, -1, -1):
        path[t] = cls[s]
        s = back[t, s]
        if s < 0 and t > 0:
            s = 0
    return path


def decode_ext(
    proba: np.ndarray,
    half: np.ndarray,
    p: DecoderParams,
    p_change: np.ndarray | None = None,
    delta_d: float = 0.0,
    change_smooth: int = 1,
    change_maxpool: int = 1,
    q0: float = 0.5,
    pair_gamma: float = 0.0,
    d_short: int = 50,
    eps: float = 1e-6,
) -> np.ndarray:
    logp = np.log(np.clip(proba.astype(np.float64), eps, 1.0))
    if delta_d != 0.0:
        logp[:, DEAD] += delta_d
    dmin_team = max(1, int(p.dmin_team))
    dmin_dead = max(1, int(p.dmin_dead))
    if p_change is not None and p.beta_ha != 0:
        q = np.clip(p_change.astype(np.float64), 1e-4, 1 - 1e-4)
        if change_maxpool > 1:
            q = maximum_filter1d(q, size=change_maxpool, mode="nearest")
        if change_smooth > 1:
            q = uniform_filter1d(q, size=change_smooth, mode="nearest")
        logit_q0 = np.log(q0 / (1.0 - q0))
        bonus = p.beta_ha * (np.log(q / (1.0 - q)) - logit_q0)
    else:
        bonus = np.zeros(len(proba), dtype=np.float64)

    out = np.full(len(proba), DEAD, np.int8)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        if pair_gamma == 0.0 and d_short <= dmin_team:
            out[idx] = _viterbi_ext(
                logp[idx], dmin_team, dmin_dead, dmin_team,
                p.lam_ha, p.lam_ld, p.lam_dl, bonus[idx], 0.0
            ).astype(np.int8)
        else:
            out[idx] = _viterbi_ext(
                logp[idx], dmin_team, dmin_dead, max(dmin_team, d_short),
                p.lam_ha, p.lam_ld, p.lam_dl, bonus[idx], pair_gamma
            ).astype(np.int8)
    return out


def blend_proba(p_gbm: np.ndarray, p_tcn: np.ndarray, w_gbm: float = 0.5, mode: str = "linear", eps: float = 1e-6) -> np.ndarray:
    if mode == "linear":
        return w_gbm * p_gbm + (1.0 - w_gbm) * p_tcn
    elif mode == "geom":
        lp = w_gbm * np.log(np.clip(p_gbm, eps, 1.0)) + (1.0 - w_gbm) * np.log(np.clip(p_tcn, eps, 1.0))
        lp -= lp.max(axis=1, keepdims=True)
        e = np.exp(lp)
        return e / e.sum(axis=1, keepdims=True)
    elif mode == "hybrid":
        p_lin = w_gbm * p_gbm + (1.0 - w_gbm) * p_tcn
        p_geo = blend_proba(p_gbm, p_tcn, w_gbm, mode="geom", eps=eps)
        return 0.5 * p_lin + 0.5 * p_geo
    raise ValueError(mode)


def main():
    store = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    sportec = load_all()

    # Load outer predictions and stored per-fold params for v4 (ens_mean)
    outer_data = {}
    for T in MATCH_IDS:
        g = np.load(store / T / f"gbm_outer_{T}.npy")
        tr = np.load(store / T / f"trans_outer_{T}.npy")
        z = np.load(store / T / f"tcn_outer_{T}.npz")
        params_v4 = DecoderParams(**json.loads((store / T / "params_ens_mean.json").read_text()))
        outer_data[T] = (g, tr, z["proba"], z["change"], sportec[T].labels, sportec[T].half, params_v4)

    # 1. Verify exact v4 baseline on LOMO
    scores_v4 = []
    for T in MATCH_IDS:
        g, tr, tp, tc, y, h, params_v4 = outer_data[T]
        p = 0.5 * g + 0.5 * tp
        c = 0.5 * tr + 0.5 * tc
        pred = decode_v4(p, h, params_v4, p_change=c)
        pred_ext = decode_ext(p, h, params_v4, p_change=c)
        assert np.array_equal(pred, pred_ext), f"Mismatch on {T}"
        scores_v4.append(metrics.score(y, pred, h)["spell_f1_mean"])
    print(f"[Baseline v4 exact verification] LOMO mean = {np.mean(scores_v4):.4f} (folds: {[round(x, 4) for x in scores_v4]})")

    # 2. Quick sensitivity sweep on outer folds (holding v4 fold params fixed first, to see directional signal)
    for mode in ("linear", "geom", "hybrid"):
        for w_gbm in (0.4, 0.5, 0.6, 0.65):
            sc = []
            for T in MATCH_IDS:
                g, tr, tp, tc, y, h, params_v4 = outer_data[T]
                p = blend_proba(g, tp, w_gbm=w_gbm, mode=mode)
                c = 0.5 * tr + 0.5 * tc
                pred = decode_ext(p, h, params_v4, p_change=c)
                sc.append(metrics.score(y, pred, h)["spell_f1_mean"])
            print(f"  mode={mode:6s} w_gbm={w_gbm:.2f} -> LOMO={np.mean(sc):.4f} ({sum(a > b for a, b in zip(sc, scores_v4))}/7 folds > v4)")

    for cm, cs in ((1, 1), (3, 1), (5, 3), (7, 5), (1, 5)):
        sc = []
        for T in MATCH_IDS:
            g, tr, tp, tc, y, h, params_v4 = outer_data[T]
            p = 0.5 * g + 0.5 * tp
            c = 0.5 * tr + 0.5 * tc
            pred = decode_ext(p, h, params_v4, p_change=c, change_maxpool=cm, change_smooth=cs)
            sc.append(metrics.score(y, pred, h)["spell_f1_mean"])
        print(f"  change_maxpool={cm} smooth={cs} -> LOMO={np.mean(sc):.4f} ({sum(a > b for a, b in zip(sc, scores_v4))}/7 folds > v4)")

    for delta_d in (-0.4, -0.25, -0.15, 0.0, 0.15, 0.3):
        sc = []
        for T in MATCH_IDS:
            g, tr, tp, tc, y, h, params_v4 = outer_data[T]
            p = 0.5 * g + 0.5 * tp
            c = 0.5 * tr + 0.5 * tc
            pred = decode_ext(p, h, params_v4, p_change=c, delta_d=delta_d)
            sc.append(metrics.score(y, pred, h)["spell_f1_mean"])
        print(f"  delta_d={delta_d:+.2f} -> LOMO={np.mean(sc):.4f}")

    for pg in (0.0, 0.25, 0.5, 1.0, 1.5):
        for ds in (35, 50, 75):
            if pg == 0.0 and ds != 35:
                continue
            sc = []
            for T in MATCH_IDS:
                g, tr, tp, tc, y, h, params_v4 = outer_data[T]
                p = 0.5 * g + 0.5 * tp
                c = 0.5 * tr + 0.5 * tc
                # Shift q0=0.15 so bonus is positive when p_change > 0.15
                pred = decode_ext(p, h, params_v4, p_change=c, pair_gamma=pg, d_short=ds, q0=0.5)
                sc.append(metrics.score(y, pred, h)["spell_f1_mean"])
            print(f"  pair_gamma={pg:.2f} d_short={ds} -> LOMO={np.mean(sc):.4f}")


if __name__ == "__main__":
    main()
