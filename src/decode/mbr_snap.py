"""EXP-10: Active 4-Dir Viterbi, Physical Boundary Impulse Snapping & Direct `spell_scores` MBR Consensus (`src/decode/mbr_snap.py`).

Implements three post-frame-model decoding innovations with strict `0.000e+00` Home <-> Away (`H <-> A`) equivariance:
1. `decode_active_4dir`:
   - Active 4-Directional (`H -> A`, `A -> H`, `Live -> D`, `D -> Live`) + 2-Stage Duration HSMM Viterbi wrapper
     around `src.decode.viterbi_hsmm_v2.decode_v2`.
2. `snap_boundaries_to_impulse`:
   - Local physical contact snapping (`[t* - w, t* + w]`, `w <= 6` frames = ±0.24s) for `H <-> A` live turnovers.
   - Shifts a Viterbi turnover boundary `t*` to a local peak of ball acceleration impulse (`max(impulse_mag_d2, impulse_mag_d4)`)
     + directional boundary probability (`p_H2A` or `p_A2H`) when `ball_visible == 1` and adjacent spells remain `>= min_spell_len`.
3. `mbr_decode_sequences`:
   - Direct `spell_scores` Minimum Bayes Risk (MBR) Consensus Decoding across `M` diverse candidate label sequences.
   - Evaluates the exact competition metric `spell_f1_mean` (`IoU in {0.3, 0.4, 0.5, 0.6, 0.7}`) between all candidate
     pairs `(y^{(m)}, y^{(k)})` (plus optional vote-consensus sequences) and selects `y^* = argmax_y sum_k w_k * spell_f1_mean(y^{(k)}, y)`.
   - Because `spell_f1_mean = 0.5 * (spell_f1_H + spell_f1_A)` is strictly symmetric under `H <-> A`,
     exact `0.000e+00` H <-> A equivariance is guaranteed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data.common import AWAY, DEAD, HOME
from src.decode.viterbi_hsmm_v2 import DecoderParamsV2, decode_v2
from src.decode.spell_reranker import rle_runs


@dataclass
class ImpulseSnapConfig:
    """Configuration for Physical Boundary Impulse Snapping (`EXP-10`)."""
    window: int = 5              # Search radius w in frames (w <= 6, ±0.20s at 25Hz)
    min_spell_len: int = 30      # Minimum duration of both adjacent spells after snapping
    w_impulse: float = 0.45      # Weight on normalized ball impulse max(impulse_mag_d2, impulse_mag_d4)
    w_b4d: float = 0.40          # Weight on directional boundary probability (p_H2A or p_A2H)
    w_pchange: float = 0.15      # Weight on 1D boundary change probability
    dist_penalty: float = 0.04   # Penalty per frame of shift |t - t*|
    min_gain: float = 0.08       # Minimum score improvement over t* required to snap
    require_visible: bool = True # Only snap to frames where ball_visible == 1


def decode_active_4dir(
    proba: np.ndarray,
    half: np.ndarray,
    base_params,
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
    beta_dir_ha: float = 0.30,
    beta_to_d: float = 0.0,
    beta_from_d: float = 0.0,
    gamma_short: float = 0.0,
    dmin_team_hard: int | None = None,
    d_soft_team: int | None = None,
    lambda_step_scale: float = 1.0,
) -> np.ndarray:
    """Decode using `decode_v2` with active 4-directional boundary modulation and optional 2-stage duration hazard."""
    p2 = DecoderParamsV2.from_legacy(
        base_params,
        beta_to_d=beta_to_d,
        beta_from_d=beta_from_d,
        short_dur_cap=d_soft_team,
        gamma_short=gamma_short,
        w_dir_ha=beta_dir_ha,
    )
    if dmin_team_hard is not None:
        p2.dmin_team = int(dmin_team_hard)
    if abs(lambda_step_scale - 1.0) > 1e-6:
        p2.lam_ha = float(p2.lam_ha * lambda_step_scale)
        p2.lam_ld = float(p2.lam_ld * lambda_step_scale)
        p2.lam_dl = float(p2.lam_dl * lambda_step_scale)
    return decode_v2(proba, half, p2, p_change=p_change, p_trans_4d=p_trans_4d)


def snap_boundaries_to_impulse(
    pred: np.ndarray,
    half: np.ndarray,
    X_physv2: pd.DataFrame | None,
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
    cfg: ImpulseSnapConfig | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Snap `H <-> A` live turnover boundaries `t*` within `[t* - w, t* + w]` to local physical impulse peaks.

    Guarantees exact `0.000e+00` H <-> A equivariance.
    """
    if cfg is None:
        cfg = ImpulseSnapConfig()
    out = np.asarray(pred, dtype=np.int8).copy()
    T = len(out)
    if cfg.window <= 0 or X_physv2 is None:
        return (out, {"n_ha_boundaries": 0, "n_snapped": 0, "mean_shift": 0.0}) if return_diagnostics else out

    imp2 = np.nan_to_num(X_physv2["impulse_mag_d2"].to_numpy(dtype=np.float64), nan=0.0) if "impulse_mag_d2" in X_physv2.columns else np.zeros(T)
    imp4 = np.nan_to_num(X_physv2["impulse_mag_d4"].to_numpy(dtype=np.float64), nan=0.0) if "impulse_mag_d4" in X_physv2.columns else np.zeros(T)
    imp_norm = np.clip(np.maximum(imp2, imp4) / 15.0, 0.0, 1.5)

    vis = np.nan_to_num(X_physv2["ball_visible"].to_numpy(dtype=np.float64), nan=0.0) if "ball_visible" in X_physv2.columns else np.ones(T)
    pc = np.zeros(T, dtype=np.float64) if p_change is None else np.nan_to_num(np.asarray(p_change, dtype=np.float64), nan=0.0)
    p4d = np.nan_to_num(np.asarray(p_trans_4d, dtype=np.float64), nan=0.0) if p_trans_4d is not None else np.column_stack([0.5 * pc, 0.5 * pc])

    runs = [list(r) for r in rle_runs(out)]
    w = min(int(cfg.window), 6)
    n_ha = 0
    shifts = []

    for k in range(len(runs) - 1):
        a_left, b_left, lbl_left = runs[k]
        a_right, b_right, lbl_right = runs[k + 1]
        t_star = b_left  # == a_right

        # Only snap direct H <-> A live turnovers inside the same active half
        if lbl_left not in (HOME, AWAY) or lbl_right not in (HOME, AWAY) or lbl_left == lbl_right:
            continue
        if half[t_star - 1] <= 0 or half[t_star] != half[t_star - 1]:
            continue

        n_ha += 1
        # Directional channel: col 0 is H -> A, col 1 is A -> H
        col_dir = 0 if (lbl_left == HOME and lbl_right == AWAY) else 1

        t_min = max(a_left + cfg.min_spell_len, t_star - w)
        t_max = min(b_right - cfg.min_spell_len, t_star + w)
        if t_min > t_max or not (t_min <= t_star <= t_max):
            continue

        score_star = (
            cfg.w_impulse * imp_norm[t_star]
            + cfg.w_b4d * p4d[t_star, col_dir]
            + cfg.w_pchange * pc[t_star]
        )
        best_t = t_star
        best_score = score_star + cfg.min_gain

        for t_cand in range(t_min, t_max + 1):
            if t_cand == t_star:
                continue
            if cfg.require_visible and vis[t_cand] < 0.5:
                continue
            sc = (
                cfg.w_impulse * imp_norm[t_cand]
                + cfg.w_b4d * p4d[t_cand, col_dir]
                + cfg.w_pchange * pc[t_cand]
                - cfg.dist_penalty * abs(t_cand - t_star)
            )
            # Round to 1e-12 for deterministic symmetry
            sc = round(float(sc), 12)
            if sc > best_score:
                best_score = sc
                best_t = t_cand

        if best_t != t_star:
            runs[k][1] = best_t
            runs[k + 1][0] = best_t
            shifts.append(abs(best_t - t_star))

    if shifts:
        for a, b, lbl in runs:
            if b > a:
                out[a:b] = lbl

    if return_diagnostics:
        return out, {
            "n_ha_boundaries": n_ha,
            "n_snapped": len(shifts),
            "mean_shift": float(np.mean(shifts)) if shifts else 0.0,
        }
    return out


def build_vote_consensus_sequence(
    candidate_seqs: list[np.ndarray],
    half: np.ndarray,
    weights: list[float] | np.ndarray | None = None,
    base_proba: np.ndarray | None = None,
    base_p_change: np.ndarray | None = None,
    dec_params=None,
    vote_sharpness: float = 3.5,
    proba_blend: float = 0.50,
) -> np.ndarray:
    """Construct a smooth, H<->A equivariant consensus sequence from `M` candidate label sequences.

    Converts the weighted candidate label votes at each frame into soft probabilities, blends with `base_proba`,
    and decodes with `dec_params` (or takes majority vote when `dec_params is None`).
    """
    M = len(candidate_seqs)
    T = len(candidate_seqs[0])
    if weights is None:
        w = np.ones(M, dtype=np.float64) / M
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.maximum(np.sum(w), 1e-8)

    vote_mat = np.zeros((T, 3), dtype=np.float64)
    vote_trans = np.zeros(T, dtype=np.float64)
    for seq, wm in zip(candidate_seqs, w):
        s = np.asarray(seq, dtype=np.int8)
        for c in (HOME, AWAY, DEAD):
            vote_mat[s == c, c] += wm
        if T > 1:
            changed = (s[1:] != s[:-1]) & (half[1:] == half[:-1]) & (half[1:] > 0)
            vote_trans[1:][changed] += wm

    # Sharpen vote distribution symmetrically
    logits = vote_sharpness * (vote_mat - 1.0 / 3.0)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(logits)
    p_vote = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    if base_proba is not None and dec_params is not None:
        p_comb = np.round((1.0 - proba_blend) * p_vote + proba_blend * np.asarray(base_proba, dtype=np.float64), 12)
        if base_p_change is not None:
            # Smooth vote transitions slightly (±2 frames)
            kernel = np.array([0.15, 0.25, 0.40, 0.25, 0.15], dtype=np.float64)
            vt_smooth = np.convolve(vote_trans, kernel, mode="same")
            c_comb = np.round(0.75 * np.asarray(base_p_change, dtype=np.float64) + 0.25 * np.clip(vt_smooth, 0.0, 1.0), 12)
        else:
            c_comb = None
        from src.decode.viterbi_hsmm import decode as decode_v1
        return decode_v1(p_comb, half, dec_params, p_change=c_comb)

    # Pure symmetric majority vote fallback (tie-break H vs A using base_proba if available, else DEAD)
    out = np.argmax(vote_mat, axis=1).astype(np.int8)
    tie_ha = np.abs(vote_mat[:, HOME] - vote_mat[:, AWAY]) < 1e-9
    tie_top = tie_ha & (vote_mat[:, HOME] >= vote_mat[:, DEAD])
    if np.any(tie_top):
        if base_proba is not None:
            out[tie_top] = np.where(base_proba[tie_top, HOME] >= base_proba[tie_top, AWAY], HOME, AWAY).astype(np.int8)
        else:
            out[tie_top] = DEAD
    return out


def mbr_decode_sequences(
    candidate_seqs: list[np.ndarray],
    half: np.ndarray,
    weights: list[float] | np.ndarray | None = None,
    include_vote_consensus: bool = True,
    base_proba: np.ndarray | None = None,
    base_p_change: np.ndarray | None = None,
    dec_params=None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Select the Minimum Bayes Risk (MBR) sequence maximizing expected `spell_f1_mean` across `M` candidates.

    Let `y^{(1)}, ..., y^{(M)}` be the reference hypotheses with weights `w_1, ..., w_M`.
    Optionally constructs additional smooth vote-consensus candidate(s) `y^{(M+1)}` and evaluates:
        `R(y) = sum_{k=1}^M w_k * spell_f1_mean(gt=y^{(k)}, pred=y)`
    Selects `y^* = argmax_y R(y)`.

    Because `spell_f1_mean(gt, pred)` is the exact unweighted average of HOME and AWAY spell F1,
    `R(y)` is 100% invariant (`0.000e+00`) under `H <-> A` swap.
    """
    from src.eval import metrics

    M = len(candidate_seqs)
    assert M >= 1, "At least 1 candidate sequence required for MBR decoding"
    if M == 1:
        seq0 = np.asarray(candidate_seqs[0], dtype=np.int8).copy()
        return (seq0, {"best_idx": 0, "best_mbr_score": 1.0, "mbr_scores": [1.0]}) if return_diagnostics else seq0

    if weights is None:
        w = np.ones(M, dtype=np.float64) / M
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.maximum(np.sum(w), 1e-8)

    pool = [np.asarray(s, dtype=np.int8) for s in candidate_seqs]
    if include_vote_consensus and base_proba is not None and dec_params is not None:
        for pb in (0.35, 0.60):
            seq_cons = build_vote_consensus_sequence(
                candidate_seqs, half, weights=w,
                base_proba=base_proba, base_p_change=base_p_change,
                dec_params=dec_params, proba_blend=pb,
            )
            pool.append(seq_cons)

    n_pool = len(pool)
    mbr_scores = np.zeros(n_pool, dtype=np.float64)

    # Pairwise cache to avoid redundant spell_f1 evaluations when two candidates are identical
    for i in range(n_pool):
        pred_i = pool[i]
        expected_f1 = 0.0
        for k in range(M):
            ref_k = pool[k]
            if i == k or np.array_equal(pred_i, ref_k):
                f1_ik = 1.0
            else:
                f1_ik = float(metrics.score(ref_k, pred_i, half)["spell_f1_mean"])
            expected_f1 += w[k] * f1_ik
        mbr_scores[i] = round(expected_f1, 12)

    best_idx = int(np.argmax(mbr_scores))
    best_seq = pool[best_idx].copy()

    if return_diagnostics:
        return best_seq, {
            "best_idx": best_idx,
            "n_ref": M,
            "n_pool": n_pool,
            "best_mbr_score": float(mbr_scores[best_idx]),
            "mbr_scores": [float(x) for x in mbr_scores],
        }
    return best_seq
