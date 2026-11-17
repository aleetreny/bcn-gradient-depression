"""EXP-08: Spell-Level Physics Re-Ranker & Surgical Episode Merger (`src/decode/spell_reranker.py`).

Targets the primary structural error mode of frame-level HSMM/Viterbi decoding under `spell_scores`:
1. False short `Y -> X -> Y` live-possession islands (`X in {HOME, AWAY}`, `Y = 1 - X`, length `L <= max_len`),
   typically triggered when a ball in pass-chord flight or tracking dropout passes near an opponent `X`
   and the hard minimum duration (`dmin_team ~ 34`) forces a ~34-100 frame phantom spell.
   Each false `Y -> X -> Y` island inflicts a triple penalty in `spell_scores`:
   - 1 False Positive spell for team `X`
   - Splits the true flanking `Y` spell into two fragments (1 False Positive + reduced IoU for team `Y`).
2. Optional false short `D -> X -> D` islands (`X in {HOME, AWAY}` flanked by `DEAD`), triggered when a player
   retrieves or places a dead ball during a stoppage (`frac_still_all` high, low ball speed / impulse).

Strict Guarantees:
- Exact `0.000e+00` Home <-> Away (`H <-> A`) equivariance: all episode features are formulated strictly
  relative to `(team_X, team_Y)` so swapping `H <-> A` in `pred`, `proba`, `p_trans_4d`, and mirroring `X_physv2`
  produces identical feature vectors (`0.000e+00` max diff) and identical merge decisions.
- Strict 7-fold Nested LOMO training & threshold selection on Sportec: L2-regularized logistic/ridge weights
  and absorption thresholds are calibrated via inner leave-one-match-out CV on the 6 training matches.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data.common import AWAY, DEAD, HOME, runs


def rle_runs(y: np.ndarray) -> list[tuple[int, int, int]]:
    s, e, v = runs(np.asarray(y, dtype=np.int64))
    return [(int(a), int(b), int(c)) for a, b, c in zip(s, e, v)]


FEATURE_NAMES_YXY = [
    "len_norm",
    "flank_min_len",
    "mean_p_X",
    "max_p_X",
    "mean_p_Y",
    "mean_p_D",
    "mean_margin_XY",
    "mean_logodds_XY",
    "flank_p_Y",
    "b4d_entry",
    "b4d_exit",
    "b4d_pair",
    "b4d_min",
    "p_change_pair",
    "max_impulse",
    "boundary_impulse",
    "frac_vis",
    "frac_chord_flight",
    "mean_ball_speed",
    "min_d1_X",
    "mean_d1_diff_YX",
    "min_d1_imp_X",
    "chord_same_Y",
    "chord_turnover",
    "ghost_disc",
    "fam_vote_X",
    "fam_min_margin_XY",
    "fam_std_margin_XY",
]


@dataclass
class SpellRerankerConfig:
    """Configuration and learned parameters for the `Y -> X -> Y` (and optional `D -> X -> D`) spell re-ranker."""
    mode: str = "hybrid"          # "physics", "linear", or "hybrid"
    max_len_yxy: int = 85         # Max candidate duration in frames (85 frames = 3.4s at 25Hz)
    min_flank_len: int = 25       # Minimum flanking Y spell duration to absorb into
    tau_absorb: float = 0.42      # Absorb candidate into Y when validity score < tau_absorb
    hard_keep_p_x: float = 0.62   # Never absorb if mean_p_X >= hard_keep_p_x
    hard_keep_b4d: float = 0.28   # Never absorb if b4d_min >= hard_keep_b4d and mean_p_X >= 0.52
    l2_reg: float = 10.0          # Ridge / L2 regularization strength for learned linear scorer
    w_phys: float = 0.50          # Blend weight between physics rule validity and learned model validity
    enable_dxd: bool = False      # Optional D -> X -> D false live blip absorption into D
    max_len_dxd: int = 65         # Max length for D -> X -> D candidate
    tau_dxd: float = 0.32         # Absorb D -> X -> D into D if live validity < tau_dxd
    # Learned parameters (populated by fit_spell_reranker)
    feat_mean: np.ndarray | None = None
    feat_std: np.ndarray | None = None
    coef: np.ndarray | None = None
    intercept: float = 0.0


def _safe_col(X: pd.DataFrame | None, col: str, T: int, default: float = 0.0) -> np.ndarray:
    if X is not None and col in X.columns:
        arr = X[col].to_numpy(dtype=np.float64)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
    return np.full(T, default, dtype=np.float64)


def _precompute_match_arrays(
    proba: np.ndarray,
    X_physv2: pd.DataFrame | None,
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
    family_probas: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Pre-extract match-level arrays once for fast O(1) candidate feature slicing."""
    T = len(proba)
    p = np.asarray(proba, dtype=np.float64)
    pc = np.zeros(T, dtype=np.float64) if p_change is None else np.nan_to_num(np.asarray(p_change, dtype=np.float64), nan=0.0)

    if p_trans_4d is not None:
        p4d = np.nan_to_num(np.asarray(p_trans_4d, dtype=np.float64), nan=0.0)
    else:
        p4d = np.column_stack([0.5 * pc, 0.5 * pc, 0.3 * pc, 0.3 * pc])

    imp2 = _safe_col(X_physv2, "impulse_mag_d2", T, 0.0)
    imp4 = _safe_col(X_physv2, "impulse_mag_d4", T, 0.0)
    imp = np.maximum(imp2, imp4)

    vis = _safe_col(X_physv2, "ball_visible", T, 1.0)
    chord_flight = _safe_col(X_physv2, "chord_in_flight", T, 1.0 - vis)
    bspd = np.clip(_safe_col(X_physv2, "ball_speed", T, 5.0), 0.0, 30.0)

    d1_h = np.clip(_safe_col(X_physv2, "dist_ball_H", T, 5.0), 0.0, 30.0)
    d1_a = np.clip(_safe_col(X_physv2, "dist_ball_A", T, 5.0), 0.0, 30.0)

    d1_imp_h = np.clip(_safe_col(X_physv2, "d1_h_impulse_gated", T, 5.0), 0.0, 30.0)
    d1_imp_a = np.clip(_safe_col(X_physv2, "d1_a_impulse_gated", T, 5.0), 0.0, 30.0)

    if X_physv2 is not None and "chord_same_team_prob_h" in X_physv2.columns:
        cs_h = _safe_col(X_physv2, "chord_same_team_prob_h", T, 0.5)
        cs_a = _safe_col(X_physv2, "chord_same_team_prob_a", T, 0.5)
    else:
        kps = _safe_col(X_physv2, "knot_pass_same", T, 0.5)
        cs_h = kps
        cs_a = kps

    cturn = _safe_col(X_physv2, "chord_turnover_prob", T, 0.5)
    gdisc = np.clip(_safe_col(X_physv2, "ball_focus_discrepancy", T, 0.0), 0.0, 30.0)
    fstill = _safe_col(X_physv2, "frac_still_all", T, 0.0)

    fams = [np.asarray(fp, dtype=np.float64) for fp in family_probas] if family_probas else [p]

    return {
        "p": p,
        "pc": pc,
        "p4d": p4d,
        "imp": imp,
        "vis": vis,
        "chord_flight": chord_flight,
        "bspd": bspd,
        "d1_h": d1_h,
        "d1_a": d1_a,
        "d1_imp_h": d1_imp_h,
        "d1_imp_a": d1_imp_a,
        "cs_h": cs_h,
        "cs_a": cs_a,
        "cturn": cturn,
        "gdisc": gdisc,
        "fstill": fstill,
        "fams": fams,
    }


def extract_yxy_candidates(
    pred: np.ndarray,
    half: np.ndarray,
    max_len_yxy: int = 125,
    min_flank_len: int = 15,
) -> list[tuple[int, int, int, int, int, int]]:
    """Find all `Y -> X -> Y` live possession candidate runs within the same half.

    Returns list of tuples `(k_idx, a, b, X, a_prev, b_next)` where:
    - `[a, b)` has label `X in (HOME, AWAY)` and length `1 <= b - a <= max_len_yxy`
    - `[a_prev, a)` and `[b, b_next)` both have label `Y = 1 - X` within the same active half (`half > 0`)
    - `max(a - a_prev, b_next - b) >= min_flank_len`
    """
    runs = rle_runs(pred)
    candidates = []
    for k in range(1, len(runs) - 1):
        a_prev, b_prev, lbl_prev = runs[k - 1]
        a, b, lbl_cur = runs[k]
        a_next, b_next, lbl_next = runs[k + 1]

        if lbl_cur not in (HOME, AWAY):
            continue
        Y = 1 - lbl_cur
        if lbl_prev != Y or lbl_next != Y:
            continue
        L = b - a
        if L <= 0 or L > max_len_yxy:
            continue
        # Ensure all three runs lie in the same graded half
        h_cur = half[a]
        if h_cur <= 0 or half[b - 1] != h_cur or half[a_prev] != h_cur or half[b_next - 1] != h_cur:
            continue
        if max(a - a_prev, b_next - b) < min_flank_len:
            continue
        candidates.append((k, a, b, int(lbl_cur), a_prev, b_next))
    return candidates


def compute_yxy_feature_vector(
    a: int,
    b: int,
    X_team: int,
    a_prev: int,
    b_next: int,
    arrays: dict,
    win: int = 4,
) -> np.ndarray:
    """Compute a 28-D strictly H<->A invariant feature vector for candidate `[a, b, X_team]` flanked by `Y = 1 - X_team`."""
    Y_team = 1 - X_team
    p = arrays["p"]
    pc = arrays["pc"]
    p4d = arrays["p4d"]
    imp = arrays["imp"]
    T = len(p)

    L = float(b - a)
    len_norm = L / 125.0
    flank_min_len = min(float(a - a_prev), float(b_next - b)) / 125.0
    flank_min_len = min(flank_min_len, 2.0)

    sl = slice(a, b)
    pX = p[sl, X_team]
    pY = p[sl, Y_team]
    pD = p[sl, DEAD]

    mean_p_X = float(np.mean(pX))
    max_p_X = float(np.max(pX))
    mean_p_Y = float(np.mean(pY))
    mean_p_D = float(np.mean(pD))
    mean_margin_XY = float(np.mean(pX - pY))
    mean_logodds_XY = float(np.mean(np.log((pX + 1e-4) / (pY + 1e-4))))

    sl_left = slice(max(a_prev, a - 25), a)
    sl_right = slice(b, min(b_next, b + 25))
    flank_l = float(np.mean(p[sl_left, Y_team])) if sl_left.stop > sl_left.start else mean_p_Y
    flank_r = float(np.mean(p[sl_right, Y_team])) if sl_right.stop > sl_right.start else mean_p_Y
    flank_p_Y = 0.5 * (flank_l + flank_r)

    # 4-Directional boundary probabilities:
    # p4d[:, 0] is H -> A, p4d[:, 1] is A -> H
    # If X_team == HOME (0) and Y_team == AWAY (1):
    #   entry Y -> X is A -> H (col 1), exit X -> Y is H -> A (col 0)
    # If X_team == AWAY (1) and Y_team == HOME (0):
    #   entry Y -> X is H -> A (col 0), exit X -> Y is A -> H (col 1)
    col_y2x = 1 if X_team == HOME else 0
    col_x2y = 0 if X_team == HOME else 1

    wa0, wa1 = max(0, a - win), min(T, a + win + 1)
    wb0, wb1 = max(0, b - win), min(T, b + win + 1)

    b4d_entry = float(np.max(p4d[wa0:wa1, col_y2x])) if wa1 > wa0 else 0.0
    b4d_exit = float(np.max(p4d[wb0:wb1, col_x2y])) if wb1 > wb0 else 0.0
    b4d_pair = 0.5 * (b4d_entry + b4d_exit)
    b4d_min = min(b4d_entry, b4d_exit)

    pc_entry = float(np.max(pc[wa0:wa1])) if wa1 > wa0 else 0.0
    pc_exit = float(np.max(pc[wb0:wb1])) if wb1 > wb0 else 0.0
    p_change_pair = 0.5 * (pc_entry + pc_exit)

    w_span0, w_span1 = max(0, a - win), min(T, b + win + 1)
    max_impulse = float(np.clip(np.max(imp[w_span0:w_span1]), 0.0, 30.0) / 30.0) if w_span1 > w_span0 else 0.0
    imp_entry = float(np.clip(np.max(imp[wa0:wa1]), 0.0, 30.0) / 30.0) if wa1 > wa0 else 0.0
    imp_exit = float(np.clip(np.max(imp[wb0:wb1]), 0.0, 30.0) / 30.0) if wb1 > wb0 else 0.0
    boundary_impulse = 0.5 * (imp_entry + imp_exit)

    frac_vis = float(np.mean(arrays["vis"][sl]))
    frac_chord_flight = float(np.mean(arrays["chord_flight"][sl]))
    mean_ball_speed = float(np.mean(arrays["bspd"][sl]) / 30.0)

    d1_X = arrays["d1_h"][sl] if X_team == HOME else arrays["d1_a"][sl]
    d1_Y = arrays["d1_a"][sl] if X_team == HOME else arrays["d1_h"][sl]
    d1_imp_X = arrays["d1_imp_h"][sl] if X_team == HOME else arrays["d1_imp_a"][sl]

    min_d1_X = float(np.min(d1_X) / 30.0)
    mean_d1_diff_YX = float(np.mean(np.clip(d1_Y - d1_X, -20.0, 20.0)) / 20.0)
    min_d1_imp_X = float(np.min(d1_imp_X) / 30.0)

    cs_Y = arrays["cs_a"][sl] if X_team == HOME else arrays["cs_h"][sl]
    chord_same_Y = float(np.mean(cs_Y))
    chord_turnover = float(np.mean(arrays["cturn"][sl]))
    ghost_disc = float(np.mean(arrays["gdisc"][sl]) / 30.0)

    # Multi-family agreement
    fams = arrays["fams"]
    fam_margins = [float(np.mean(fp[sl, X_team] - fp[sl, Y_team])) for fp in fams]
    fam_vote_X = float(np.mean([1.0 if m > 0.0 else (0.5 if m == 0.0 else 0.0) for m in fam_margins]))
    fam_min_margin_XY = float(np.min(fam_margins))
    fam_std_margin_XY = float(np.std(fam_margins)) if len(fam_margins) > 1 else 0.0

    vec = np.array([
        len_norm,
        flank_min_len,
        mean_p_X,
        max_p_X,
        mean_p_Y,
        mean_p_D,
        mean_margin_XY,
        mean_logodds_XY,
        flank_p_Y,
        b4d_entry,
        b4d_exit,
        b4d_pair,
        b4d_min,
        p_change_pair,
        max_impulse,
        boundary_impulse,
        frac_vis,
        frac_chord_flight,
        mean_ball_speed,
        min_d1_X,
        mean_d1_diff_YX,
        min_d1_imp_X,
        chord_same_Y,
        chord_turnover,
        ghost_disc,
        fam_vote_X,
        fam_min_margin_XY,
        fam_std_margin_XY,
    ], dtype=np.float64)
    # Round to 1e-12 to eliminate floating-point instruction-order jitter
    return np.round(vec, 12)


def compute_physics_validity_score(feat: np.ndarray) -> float:
    """Compute a physically grounded, deterministic [0, 1] validity score for a `Y -> X -> Y` candidate.

    Higher score => genuinely valid `X` possession spell.
    Lower score  => false `X` blip during `Y`'s pass flight / tracking gap / model flicker.
    """
    len_norm = feat[0]         # L / 125.0
    mean_p_X = feat[2]
    max_p_X = feat[3]
    mean_margin_XY = feat[6]
    flank_p_Y = feat[8]
    b4d_pair = feat[11]
    b4d_min = feat[12]
    boundary_impulse = feat[15]
    frac_vis = feat[16]
    frac_chord_flight = feat[17]
    min_d1_X = feat[19]        # min_d1_X / 30.0 (e.g. 1.5m -> 0.05, 6m -> 0.20)
    mean_d1_diff_YX = feat[20]
    min_d1_imp_X = feat[21]    # imputed min distance during blackouts / 30.0
    chord_same_Y = feat[22]
    fam_vote_X = feat[25]
    fam_min_margin_XY = feat[26]

    # Positive evidence that X truly touched/controlled the ball:
    # 1. Probability & multi-family margin support
    prob_support = 0.45 * mean_p_X + 0.15 * max_p_X + 0.20 * np.clip(0.5 + mean_margin_XY, 0.0, 1.0) + 0.20 * fam_vote_X
    # 2. 4-directional boundary confirmation + physical impulse at boundaries
    bound_support = 0.60 * b4d_pair + 0.25 * b4d_min + 0.15 * np.clip(boundary_impulse * 2.5, 0.0, 1.0)
    # 3. Physical proximity of X to ball (visible ball distance when visible; imputed/neutral when invisible so v6/blackouts are not penalized)
    vis_prox = np.clip(1.0 - (min_d1_X * 30.0) / 4.5, 0.0, 1.0)
    imp_prox = np.clip(1.0 - (min_d1_imp_X * 30.0) / 6.0, 0.45, 0.75)
    prox_support = frac_vis * vis_prox + (1.0 - frac_vis) * imp_prox
    # 4. Duration support (longer spells up to 100 frames are more likely real than 34-frame minimum-duration spikes)
    dur_support = np.clip((len_norm * 125.0 - 30.0) / 70.0, 0.0, 1.0)

    # Penalties for pass-chord flight / same-team Y pass / strong flanking Y / single-family disagreement:
    flight_penalty = 0.14 * frac_chord_flight * np.clip(chord_same_Y, 0.3, 1.0)
    flank_penalty = 0.10 * np.clip(flank_p_Y - mean_p_X, 0.0, 0.6)
    disagree_penalty = 0.10 * np.clip(-fam_min_margin_XY, 0.0, 0.5)
    far_penalty = 0.10 * frac_vis * np.clip(-mean_d1_diff_YX, 0.0, 0.5)

    score = (
        0.48 * prob_support
        + 0.24 * bound_support
        + 0.18 * prox_support
        + 0.10 * dur_support
        - flight_penalty
        - flank_penalty
        - disagree_penalty
        - far_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


def score_yxy_candidate(feat: np.ndarray, cfg: SpellRerankerConfig) -> float:
    """Compute combined validity score in [0, 1] under `cfg`."""
    phys_score = compute_physics_validity_score(feat)
    if cfg.mode == "physics" or cfg.coef is None or cfg.feat_mean is None or cfg.feat_std is None:
        return phys_score

    z = (feat - cfg.feat_mean) / np.maximum(cfg.feat_std, 1e-6)
    z = np.clip(z, -6.0, 6.0)
    lin_logit = float(np.dot(z, cfg.coef) + cfg.intercept)
    lin_score = float(1.0 / (1.0 + np.exp(-np.clip(lin_logit, -20.0, 20.0))))

    if cfg.mode == "linear":
        return lin_score
    return float(cfg.w_phys * phys_score + (1.0 - cfg.w_phys) * lin_score)


def apply_spell_reranker(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X_physv2: pd.DataFrame | None,
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
    family_probas: list[np.ndarray] | None = None,
    cfg: SpellRerankerConfig | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Apply the Spell-Level Physics Re-Ranker & Surgical `Y -> X -> Y` Merger.

    Guarantees exact `0.000e+00` H <-> A equivariance.
    """
    if cfg is None:
        cfg = SpellRerankerConfig()

    out = np.asarray(pred, dtype=np.int8).copy()
    arrays = _precompute_match_arrays(proba, X_physv2, p_change, p_trans_4d, family_probas)

    n_yxy_cand = len(extract_yxy_candidates(out, half, max_len_yxy=max(cfg.max_len_yxy, 1), min_flank_len=cfg.min_flank_len))
    n_yxy_absorbed = 0
    n_dxd_absorbed = 0

    if cfg.tau_absorb > 0.0 and cfg.max_len_yxy > 0:
        # Up to 2 passes so adjacent fragments merged in pass 1 can expose nested micro-blips
        for _pass in range(2):
            cands = extract_yxy_candidates(out, half, max_len_yxy=cfg.max_len_yxy, min_flank_len=cfg.min_flank_len)
            if not cands:
                break

            absorbed_any = False
            for _k, a, b, X_team, a_prev, b_next in cands:
                feat = compute_yxy_feature_vector(a, b, X_team, a_prev, b_next, arrays)
                mean_p_X = feat[2]
                b4d_min = feat[12]
                frac_vis = feat[16]
                L = b - a

                # Hard safety guards: never absorb high-confidence or strongly boundary-confirmed spells,
                # nor medium/long spells during ball blackouts (v6 / broadcast camera cuts) unless p_X < 0.48
                if mean_p_X >= cfg.hard_keep_p_x:
                    continue
                if b4d_min >= cfg.hard_keep_b4d and mean_p_X >= 0.52:
                    continue
                if frac_vis < 0.50 and L > 55 and mean_p_X >= 0.48:
                    continue

                val_score = score_yxy_candidate(feat, cfg)
                if val_score < cfg.tau_absorb:
                    Y_team = 1 - X_team
                    out[a:b] = Y_team
                    n_yxy_absorbed += 1
                    absorbed_any = True
            if not absorbed_any:
                break

    if cfg.enable_dxd and cfg.tau_dxd > 0.0 and cfg.max_len_dxd > 0:
        runs = rle_runs(out)
        p = arrays["p"]
        vis = arrays["vis"]
        bspd = arrays["bspd"]
        fstill = arrays["fstill"]
        for k in range(1, len(runs) - 1):
            a_prev, b_prev, lbl_prev = runs[k - 1]
            a, b, lbl_cur = runs[k]
            a_next, b_next, lbl_next = runs[k + 1]
            if lbl_cur not in (HOME, AWAY):
                continue
            if lbl_prev != DEAD or lbl_next != DEAD:
                continue
            L = b - a
            if L <= 0 or L > cfg.max_len_dxd:
                continue
            h_cur = half[a]
            if h_cur <= 0 or half[b - 1] != h_cur:
                continue
            # Symmetric live-vs-dead validity inside D -> X -> D
            mean_p_D = float(np.mean(p[a:b, DEAD]))
            mean_p_X = float(np.mean(p[a:b, lbl_cur]))
            mean_still = float(np.mean(fstill[a:b]))
            mean_spd = float(np.mean(bspd[a:b]) / 30.0)
            mean_vis = float(np.mean(vis[a:b]))
            live_val = 0.50 * (1.0 - mean_p_D) + 0.25 * mean_p_X + 0.15 * (1.0 - mean_still) + 0.10 * mean_spd * mean_vis
            if live_val < cfg.tau_dxd and (a - a_prev) >= 25 and (b_next - b) >= 25:
                out[a:b] = DEAD
                n_dxd_absorbed += 1

    if return_diagnostics:
        return out, {
            "n_yxy_candidates": n_yxy_cand,
            "n_yxy_absorbed": n_yxy_absorbed,
            "n_dxd_absorbed": n_dxd_absorbed,
        }
    return out


def _fit_ridge_logit_scorer(
    X_mat: np.ndarray,
    y_target: np.ndarray,
    sample_weight: np.ndarray,
    l2_reg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit an L2-regularized linear scorer (weighted ridge regression onto clipped log-odds target).

    Because every feature in `X_mat` is already strictly H<->A invariant (`team_X` vs `team_Y`),
    the learned weights are 100% H<->A equivariant (`0.000e+00` symmetry error).
    """
    n, d = X_mat.shape
    if n < 5:
        return np.zeros(d), np.ones(d), np.zeros(d), 0.0

    w = sample_weight / np.maximum(np.sum(sample_weight), 1e-8) * n
    feat_mean = np.average(X_mat, axis=0, weights=w)
    feat_var = np.average((X_mat - feat_mean) ** 2, axis=0, weights=w)
    feat_std = np.sqrt(np.maximum(feat_var, 1e-6))

    Z = np.clip((X_mat - feat_mean) / feat_std, -6.0, 6.0)
    # Target in logit space: y_target in [0, 1] mapped to [-2.5, +2.5]
    y_clip = np.clip(y_target, 0.05, 0.95)
    y_logit = np.log(y_clip / (1.0 - y_clip))

    y_mean = float(np.average(y_logit, weights=w))
    y_centered = y_logit - y_mean

    sqrt_w = np.sqrt(w)[:, None]
    Zw = Z * sqrt_w
    yw = y_centered * np.sqrt(w)

    A = Zw.T @ Zw + l2_reg * np.eye(d, dtype=np.float64)
    b = Zw.T @ yw
    coef = np.linalg.solve(A, b)
    return np.round(feat_mean, 10), np.round(feat_std, 10), np.round(coef, 10), round(y_mean, 10)


def fit_spell_reranker_lomo(
    train_match_ids: list[str],
    preds_by_match: dict[str, np.ndarray],
    probas_by_match: dict[str, np.ndarray],
    halves_by_match: dict[str, np.ndarray],
    gt_by_match: dict[str, np.ndarray],
    physv2_by_match: dict[str, pd.DataFrame],
    p_change_by_match: dict[str, np.ndarray | None],
    p_trans_4d_by_match: dict[str, np.ndarray | None],
    family_probas_by_match: dict[str, list[np.ndarray] | None] | None = None,
    max_len_grid: tuple[int, ...] = (60, 85, 110, 125),
    tau_grid: tuple[float, ...] = (0.0, 0.30, 0.34, 0.38, 0.42, 0.45, 0.48),
    modes: tuple[str, ...] = ("physics", "hybrid"),
    l2_reg: float = 15.0,
) -> tuple[SpellRerankerConfig, dict]:
    """Fit and select `SpellRerankerConfig` using strict inner leave-one-match-out CV on `train_match_ids`.

    For each candidate `[a, b, X]` on training match `M`, we compute:
    1. Its 28-D H<->A invariant feature vector
    2. The exact oracle `spell_f1_mean` delta when absorbing `[a, b, X]` into `Y = 1 - X`,
       plus the ground-truth fraction of `X` vs `Y` inside `[a, b)`.
    Then we run inner LOMO across `train_match_ids` (fitting the Ridge scorer on 5 matches, predicting on the 6th)
    and select `(mode, max_len_yxy, tau_absorb)` that maximizes inner-OOF `spell_f1_mean`.
    """
    from src.eval import metrics

    arrays_map = {}
    cand_records_by_match = {}
    base_scores = {}

    max_L = max(max_len_grid)
    for mid in train_match_ids:
        pred_m = preds_by_match[mid]
        half_m = halves_by_match[mid]
        gt_m = gt_by_match[mid]
        fams_m = family_probas_by_match[mid] if family_probas_by_match else None
        arr_m = _precompute_match_arrays(
            probas_by_match[mid],
            physv2_by_match[mid],
            p_change_by_match.get(mid),
            p_trans_4d_by_match.get(mid),
            fams_m,
        )
        arrays_map[mid] = arr_m
        base_f1 = float(metrics.score(gt_m, pred_m, half_m)["spell_f1_mean"])
        base_scores[mid] = base_f1

        cands = extract_yxy_candidates(pred_m, half_m, max_len_yxy=max_L, min_flank_len=15)
        recs = []
        for k_idx, a, b, X_team, a_prev, b_next in cands:
            feat = compute_yxy_feature_vector(a, b, X_team, a_prev, b_next, arr_m)
            Y_team = 1 - X_team
            # Oracle spell_f1_mean change if this single candidate is absorbed into Y_team
            pred_mod = pred_m.copy()
            pred_mod[a:b] = Y_team
            mod_f1 = float(metrics.score(gt_m, pred_mod, half_m)["spell_f1_mean"])
            delta_keep = base_f1 - mod_f1  # > 0 means keeping X is better; < 0 means absorbing into Y is better

            gt_slice = gt_m[a:b]
            valid_gt = gt_slice[(gt_slice == HOME) | (gt_slice == AWAY)]
            frac_gt_X = float(np.mean(valid_gt == X_team)) if len(valid_gt) > 0 else 0.5

            # Continuous validity target combining ground-truth fraction and oracle spell_f1 benefit
            oracle_sign = 1.0 if delta_keep > 1e-5 else (0.0 if delta_keep < -1e-5 else frac_gt_X)
            y_target = 0.65 * oracle_sign + 0.35 * frac_gt_X
            weight = 1.0 + 250.0 * abs(delta_keep)
            recs.append((a, b, X_team, a_prev, b_next, feat, y_target, weight, delta_keep))
        cand_records_by_match[mid] = recs

    # Fit inner-LOMO linear models (for each M_val in train_match_ids, fit on train_match_ids \ {M_val})
    inner_models = {}
    for m_val in train_match_ids:
        X_list, y_list, w_list = [], [], []
        for m_tr in train_match_ids:
            if m_tr == m_val:
                continue
            for r in cand_records_by_match[m_tr]:
                X_list.append(r[5])
                y_list.append(r[6])
                w_list.append(r[7])
        if len(X_list) >= 5:
            f_mean, f_std, coef, inter = _fit_ridge_logit_scorer(
                np.vstack(X_list), np.array(y_list), np.array(w_list), l2_reg=l2_reg
            )
        else:
            d = len(FEATURE_NAMES_YXY)
            f_mean, f_std, coef, inter = np.zeros(d), np.ones(d), np.zeros(d), 0.0
        inner_models[m_val] = (f_mean, f_std, coef, inter)

    # Grid search (mode, max_len_yxy, tau_absorb) on inner-OOF matches
    base_mean = float(np.mean([base_scores[m] for m in train_match_ids]))
    best_score = base_mean
    best_tuple = ("physics", 100, 0.0)

    for mode in modes:
        for mlen in max_len_grid:
            for tau in tau_grid:
                if tau == 0.0:
                    continue
                scores = []
                for m_val in train_match_ids:
                    f_mean, f_std, coef, inter = inner_models[m_val]
                    cfg_try = SpellRerankerConfig(
                        mode=mode,
                        max_len_yxy=mlen,
                        tau_absorb=tau,
                        l2_reg=l2_reg,
                        feat_mean=f_mean,
                        feat_std=f_std,
                        coef=coef,
                        intercept=inter,
                    )
                    fams_m = family_probas_by_match[m_val] if family_probas_by_match else None
                    pred_post = apply_spell_reranker(
                        preds_by_match[m_val],
                        probas_by_match[m_val],
                        halves_by_match[m_val],
                        physv2_by_match[m_val],
                        p_change_by_match.get(m_val),
                        p_trans_4d_by_match.get(m_val),
                        fams_m,
                        cfg=cfg_try,
                    )
                    scores.append(float(metrics.score(gt_by_match[m_val], pred_post, halves_by_match[m_val])["spell_f1_mean"]))
                mean_sc = float(np.mean(scores))
                deltas = [sc - base_scores[m_val] for sc, m_val in zip(scores, train_match_ids)]
                n_pos = sum(1 for d in deltas if d > 1e-6)
                n_neg = sum(1 for d in deltas if d < -1e-6)
                if mean_sc > best_score + 1e-5 and n_pos >= n_neg and min(deltas) >= -1.8e-3:
                    best_score = mean_sc
                    best_tuple = (mode, mlen, tau)

    # Fit final linear scorer on all training matches in `train_match_ids`
    X_all, y_all, w_all = [], [], []
    for m_tr in train_match_ids:
        for r in cand_records_by_match[m_tr]:
            X_all.append(r[5])
            y_all.append(r[6])
            w_all.append(r[7])
    if len(X_all) >= 5:
        f_mean, f_std, coef, inter = _fit_ridge_logit_scorer(
            np.vstack(X_all), np.array(y_all), np.array(w_all), l2_reg=l2_reg
        )
    else:
        d = len(FEATURE_NAMES_YXY)
        f_mean, f_std, coef, inter = np.zeros(d), np.ones(d), np.zeros(d), 0.0

    best_cfg = SpellRerankerConfig(
        mode=best_tuple[0],
        max_len_yxy=best_tuple[1],
        tau_absorb=best_tuple[2],
        l2_reg=l2_reg,
        feat_mean=f_mean,
        feat_std=f_std,
        coef=coef,
        intercept=inter,
    )
    info = {
        "base_inner_f1": base_mean,
        "best_inner_f1": best_score,
        "inner_gain": best_score - base_mean,
        "best_mode": best_tuple[0],
        "best_max_len": best_tuple[1],
        "best_tau": best_tuple[2],
        "n_train_candidates": len(X_all),
    }
    return best_cfg, info
