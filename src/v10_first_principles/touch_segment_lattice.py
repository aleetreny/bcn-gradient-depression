"""
Touch-Graph & Segment-Lattice Neural Decoder (`B1` Short-Steal / Double-Domino & `B2` Boundary Solver)
======================================================================================================

Rethinks possession decoding from first principles as a **Candidate Segment Lattice (`B1`) + Pre-Touch
Boundary Aligner (`B2`)** driven by:
  1. Modal GPU `SymPlayerSetNetV2` (`pset_v2_oof`, trained with `L1` duration weights `w = L^-0.35`,
     dense transition crops, 4D directional boundary head `p4d`, and 3D aerial gate `p_aerial` `0.9798 AUC`),
  2. Modal CPU `L1-Trees` (`l1_trees_oof`: `gbm`, `cb`, `xgb` trained with `L1` duration weights +
     dedicated `duel_to` `0.9401 AUC` and `air` `0.9746 AUC` heads),
  3. `v8.5` 9-Family Ensemble (`tree` 0..3, `seq244` 4..5, `seq481` 6..7, `pset` 8) + `SymBoundary4Dir`,
  4. Physical touch & 3D ballistic trajectory kinematics evaluated strictly inside the interior turnover
     slice `sl_int = [ca + pad_a, cb - pad_b)` (preventing host-team entry/exit kick contamination),
  5. Exact analytical Expected-`spellF1` Double-Domino payoff across `IoU in {0.3, 0.4, 0.5, 0.6, 0.7}`.

Architecture Overview
---------------------
- **Part 1 — Multi-Modal Candidate Segment & Touch-Graph Proposal Generator**:
  - `(A1) MBR-v10`: Candidate `X -> Y -> X` false islands (`L in [25, 65]`) caused by un-deflected aerial
    flight or parallel pressing runs where `SymPlayerSetNetV2` and `L1-Trees` reject a touch.
  - `(A2) XYX-v10`: Candidate interior short steals / headers / clearances `[ca, cb)` (`L in [10, 90]`
    frames = `0.40s..3.60s`) inside live `X` spells (`L_host >= 50`), seeded by local peaks in the
    multi-modal entry/exit boundary stream `max(0.85*c, p4d_to_Y, 0.85*ps2_c, ps2_p4d_to_Y, 0.75*l1_duel)`.
  - `(A3) XYD-v10`: Candidate pre-stoppage deflection / clearance / foul-won suffix turnovers `[ca, sb)`
    immediately preceding a `DEAD` interval (`X -> Y -> D`).
  - `(A4) DYX-v10`: Candidate post-restart aerial / second-ball prefix turnovers `[sa, cb)` immediately
    following a `DEAD` interval (`D -> Y -> X`).
  - `(B) B2-v10`: Candidate `H <-> A` live boundary windows `tb + d` (`d in [-20, +20]` frames) with
    24 equivariant segment-gain, directional-boundary, and incoming-touch proximity features.

- **Part 2 — Exact Analytical Expected-`spellF1` Double-Domino Payoff + Equivariant Segment-RoI Features**:
  - For every candidate `[ca, cb)` inside host spell `[sa, sb)`, computes the exact analytical `IoU` of
    the left (`[sa, ca)`) and right (`[cb, sb)`) flanking segments against `[sa, sb)` across
    `IoU in {0.3, 0.4, 0.5, 0.6, 0.7}` (`flank_max_ratio`, `kept_if_false`, `tp_loss_if_false`,
    `tp_gain_if_true`).
  - When `flank_max_ratio >= 0.70`, splitting `[sa, sb)` has `tp_loss_if_false == 0.0` across all 5 IoU
    thresholds (the larger flank still matches the true `[sa, sb)` spell even at `IoU = 0.70`!), while
    recovering up to `+2.0` expected TP spells when true.

- **Part 3 — Strict 7-Fold Nested LOMO Segment-RoI Classifier + Pre-Touch Boundary Aligner V10**:
  - Fits all sub-regime activation thresholds strictly in 7-Fold Nested Leave-One-Match-Out (`LOMO`)
    on Sportec (`v3` / `v6_std`), never touching `J2` (`Sample 1-3`) or `J3` (`BAR-ATH`).
  - Enforces exact `0.000e+00` `HOME <-> AWAY` equivariance across every candidate generator, feature,
    score function, and decoder step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import evaluation.possession_metrics as pm
from src.data.common import AWAY, DEAD, HOME
from src.decode.boundary_offset_aligner import (
    _extract_kinematic_arrays,
    _f1_from_ov,
    _single_spell_overlaps,
)
from src.decode.forensic_shields_v85 import rle_runs_half, to_official
from src.v9_swarm.duel_and_mbr_v9 import extract_mbr_xyx_candidates_v9

THRESHOLDS: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
DELTAS_20: np.ndarray = np.arange(-20, 21, dtype=np.int32)  # 41 offsets, index 20 is d = 0


# ======================================================================================
# PART 1 & 2: ANALYTICAL DOUBLE-DOMINO PAYOFF & EQUIVARIANT SEGMENT-ROI FEATURE EXTRACTOR
# ======================================================================================


def prep_kinematics_v10(X_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract clean float64 kinematic & physical touch arrays from `PhysV2` table `X_df`."""
    return {
        "d1_h": np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=50.0),
        "d1_a": np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=50.0),
        "vis": np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=1.0),
        "chord": np.nan_to_num(X_df["chord_in_flight"].to_numpy(np.float64), nan=0.0),
        "chord_to": np.nan_to_num(X_df["chord_turnover_prob"].to_numpy(np.float64), nan=0.0),
        "imp2": np.nan_to_num(X_df["impulse_mag_d2"].to_numpy(np.float64), nan=0.0),
        "imp4": np.nan_to_num(X_df["impulse_mag_d4"].to_numpy(np.float64), nan=0.0),
        "cos2": np.nan_to_num(X_df["impulse_cos_d2"].to_numpy(np.float64), nan=1.0),
        "cos4": np.nan_to_num(X_df["impulse_cos_d4"].to_numpy(np.float64), nan=1.0),
        "bang": np.nan_to_num(X_df["ball_angle_change"].to_numpy(np.float64), nan=0.0) / 180.0,
    }


def analytical_domino_payoffs(
    sa: int,
    sb: int,
    ca: int,
    cb: int,
    thresholds: Sequence[float] = THRESHOLDS,
) -> Dict[str, float]:
    """Compute exact analytical Double-Domino IoU payoffs of splitting `[sa, sb)` at `[ca, cb)`.

    For a host spell `[sa, sb)` of length `L_p = sb - sa`, inserting `[ca, cb)` produces left
    flank `[sa, ca)` (`IoU = (ca - sa) / L_p` against `[sa, sb)`) and right flank `[cb, sb)`
    (`IoU = (sb - cb) / L_p` against `[sa, sb)`):
      - `kept_if_false`: fraction of IoU thresholds `{0.3, 0.4, 0.5, 0.6, 0.7}` where the larger
        flank `max(r_l, r_r)` still satisfies `IoU >= th` if `[sa, sb)` was actually unsplit.
      - `tp_loss_if_false = 1.0 - kept_if_false`: expected TP loss if the split is a false positive.
      - `tp_gain_if_true = n_after - kept_if_false`: expected TP gain (up to `+3.0` TP spells in
        `XYX` Double-Domino when `r_max < 0.30`, or `+2.0` with zero loss when `r_max >= 0.70`).
    """
    L_p = max(1, sb - sa)
    L_l = max(0, ca - sa)
    L_r = max(0, sb - cb)
    r_l = L_l / L_p
    r_r = L_r / L_p
    r_max = max(r_l, r_r)
    r_min = min(r_l, r_r)
    kept_if_false = sum(1 for th in thresholds if r_max >= th) / float(len(thresholds))
    tp_loss_if_false = 1.0 - kept_if_false
    n_after = (1 if L_l > 0 else 0) + 1 + (1 if L_r > 0 else 0)
    tp_gain_if_true = float(n_after) - kept_if_false
    return {
        "flank_left_len": float(L_l),
        "flank_right_len": float(L_r),
        "flank_max_ratio": float(np.round(r_max, 10)),
        "flank_min_ratio": float(np.round(r_min, 10)),
        "kept_if_false": float(np.round(kept_if_false, 10)),
        "tp_loss_if_false": float(np.round(tp_loss_if_false, 10)),
        "tp_gain_if_true": float(np.round(tp_gain_if_true, 10)),
    }


def extract_segment_roi_features_v10(
    sa: int,
    sb: int,
    ca: int,
    cb: int,
    x_lbl: int,
    y_lbl: int,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    ps: Dict[str, np.ndarray],
    l1: Dict[str, np.ndarray],
    kin: Dict[str, np.ndarray],
    kind: str,
    t10: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Extract 100% H<->A equivariant Segment-RoI features for candidate `[ca, cb)` inside `[sa, sb)`."""
    n = len(c)
    L = cb - ca
    sl = slice(ca, cb)

    # Interior slice [ca + pad_a, cb - pad_b) avoids host-team kick frames at ca and cb
    pad_a = max(1, min(3, L // 5)) if kind != "DYX" else 0
    pad_b = max(1, min(3, L // 5)) if kind != "XYD" else 0
    ca_int = ca + pad_a
    cb_int = max(ca_int + 1, cb - pad_b)
    sl_int = slice(ca_int, cb_int)

    # Corrected SymBoundary4Dir directional columns:
    # col 0 is HOME -> AWAY (entry into AWAY), col 1 is AWAY -> HOME (entry into HOME)
    col_to_Y = 1 if y_lbl == HOME else 0
    col_to_X = 1 if x_lbl == HOME else 0
    col_to_D = 2

    ps_p = ps["proba"]
    ps_c = ps["change"]
    ps_p4d = ps["p4d"]
    ps_air = ps["p_aerial"]
    l1_p = (l1["gbm"] + l1["cb"] + l1["xgb"]) / 3.0
    l1_duel = l1["duel"]
    l1_air = l1["air"]

    in_lo, in_hi = max(0, ca - 4), min(n, ca + 5)
    out_lo, out_hi = max(0, cb - 4), min(n, cb + 5)

    out_col = col_to_X if kind != "XYD" else col_to_D
    in_col = col_to_Y if kind != "DYX" else 3

    b4d_in = float(np.max(p4d[in_lo:in_hi, in_col]))
    b4d_out = float(np.max(p4d[out_lo:out_hi, out_col]))
    b4d_in_inv = float(np.max(p4d[in_lo:in_hi, col_to_X]))
    b4d_out_inv = float(np.max(p4d[out_lo:out_hi, col_to_Y]))
    b4d_dir_asym = float((b4d_in - b4d_in_inv) + (b4d_out - b4d_out_inv))

    ps2_b4d_in = float(np.max(ps_p4d[in_lo:in_hi, in_col]))
    ps2_b4d_out = float(np.max(ps_p4d[out_lo:out_hi, out_col]))
    ps2_b4d_in_inv = float(np.max(ps_p4d[in_lo:in_hi, col_to_X]))
    ps2_b4d_out_inv = float(np.max(ps_p4d[out_lo:out_hi, col_to_Y]))
    ps2_b4d_asym = float((ps2_b4d_in - ps2_b4d_in_inv) + (ps2_b4d_out - ps2_b4d_out_inv))

    pc_in = float(np.max(c[in_lo:in_hi]))
    pc_out = float(np.max(c[out_lo:out_hi]))
    ps2_pc_in = float(np.max(ps_c[in_lo:in_hi]))
    ps2_pc_out = float(np.max(ps_c[out_lo:out_hi]))
    l1_duel_in = float(np.max(l1_duel[in_lo:in_hi]))
    l1_duel_out = float(np.max(l1_duel[out_lo:out_hi]))

    pY_sl = p[sl, y_lbl]
    pX_sl = p[sl, x_lbl]
    pD_sl = p[sl, DEAD]
    ps2_pY_sl = ps_p[sl, y_lbl]
    ps2_pX_sl = ps_p[sl, x_lbl]
    ps2_pD_sl = ps_p[sl, DEAD]
    l1_pY_sl = l1_p[sl, y_lbl]
    l1_pX_sl = l1_p[sl, x_lbl]
    l1_pD_sl = l1_p[sl, DEAD]

    tree_Y = 0.25 * (fams[0][sl, y_lbl] + fams[1][sl, y_lbl] + fams[2][sl, y_lbl] + fams[3][sl, y_lbl])
    seq244_Y = 0.5 * (fams[4][sl, y_lbl] + fams[5][sl, y_lbl])
    seq481_Y = 0.5 * (fams[6][sl, y_lbl] + fams[7][sl, y_lbl]) if len(fams) >= 9 else seq244_Y
    seq_Y = 0.5 * (seq244_Y + seq481_Y)
    pset_Y = fams[-1][sl, y_lbl]

    fam_vote_Y = float(np.mean([1.0 if np.mean(fp[sl, y_lbl] - fp[sl, x_lbl]) > 0 else 0.0 for fp in fams]))
    fam_peak_vote_Y = float(np.mean([1.0 if np.max(fp[sl, y_lbl]) >= 0.45 else 0.0 for fp in fams]))

    # Flanking surge windows [max(sa, ca-25), ca) and (cb, min(sb, cb+25)]
    la = max(sa, ca - 25)
    rb = min(sb, cb + 25)
    flk_idx = list(range(la, ca)) + list(range(cb, rb))
    if flk_idx:
        ps2_flk_Y = float(np.mean(ps_p[flk_idx, y_lbl]))
        l1_flk_Y = float(np.mean(l1_p[flk_idx, y_lbl]))
        v85_flk_Y = float(np.mean(p[flk_idx, y_lbl]))
        s244_flk = 0.5 * (fams[4][flk_idx, y_lbl] + fams[5][flk_idx, y_lbl])
        s481_flk = 0.5 * (fams[6][flk_idx, y_lbl] + fams[7][flk_idx, y_lbl]) if len(fams) >= 9 else s244_flk
        seq_flk_Y = float(np.mean(0.5 * (s244_flk + s481_flk)))
    else:
        ps2_flk_Y = float(np.mean(ps2_pY_sl))
        l1_flk_Y = float(np.mean(l1_pY_sl))
        v85_flk_Y = float(np.mean(pY_sl))
        seq_flk_Y = float(np.mean(seq_Y))

    d1_Y = kin["d1_h"] if y_lbl == HOME else kin["d1_a"]
    d1_X = kin["d1_a"] if y_lbl == HOME else kin["d1_h"]
    min_d1_Y_int = float(np.min(d1_Y[sl_int]))
    min_d1_X_int = float(np.min(d1_X[sl_int]))
    d1_adv_int = float(np.mean(np.clip(d1_X[sl_int] - d1_Y[sl_int], -15.0, 15.0)))
    max_d1_adv_int = float(np.max(d1_X[sl_int] - d1_Y[sl_int]))
    y_win_frames = float(np.sum((d1_Y[sl] < 1.65) & (d1_Y[sl] <= d1_X[sl] + 0.25)))
    x_win_frames = float(np.sum((d1_X[sl_int] < 1.40) & (d1_X[sl_int] + 0.25 < d1_Y[sl_int])))
    duel_2m_frames = float(np.sum((d1_Y[sl] < 2.0) & (d1_X[sl] < 2.2)))

    w_lo, w_hi = max(0, ca - 4), min(n, cb + 5)
    max_imp4_win = float(np.max(kin["imp4"][w_lo:w_hi]))
    min_cos4_win = float(np.min(kin["cos4"][w_lo:w_hi]))
    min_cos2_win = float(np.min(kin["cos2"][w_lo:w_hi]))
    max_ang_win = float(np.max(kin["bang"][w_lo:w_hi]))
    chord_int = float(np.mean(kin["chord"][sl_int]))
    frac_vis = float(np.mean(kin["vis"][sl]))
    undeflected_aerial = float(chord_int * (1.0 if min_d1_Y_int > 1.35 and min_cos4_win > 0.35 else 0.0))

    dom = analytical_domino_payoffs(sa, sb, ca, cb)
    res: Dict[str, Any] = {
        "kind": kind,
        "sa": int(sa),
        "sb": int(sb),
        "ca": int(ca),
        "cb": int(cb),
        "L": int(L),
        "x_lbl": int(x_lbl),
        "y_lbl": int(y_lbl),
        "b4d_in_Y": float(np.round(b4d_in, 10)),
        "b4d_out_X": float(np.round(b4d_out, 10)),
        "b4d_geom": float(np.round(np.sqrt(max(0.0, b4d_in * b4d_out)), 10)),
        "b4d_dir_asym": float(np.round(b4d_dir_asym, 10)),
        "pc_in": float(np.round(pc_in, 10)),
        "pc_out": float(np.round(pc_out, 10)),
        "pc_geom": float(np.round(np.sqrt(max(0.0, pc_in * pc_out)), 10)),
        "mean_pY": float(np.round(np.mean(pY_sl), 10)),
        "max_pY": float(np.round(np.max(pY_sl), 10)),
        "int_pY": float(np.round(np.mean(p[sl_int, y_lbl]), 10)),
        "mean_pX": float(np.round(np.mean(pX_sl), 10)),
        "mean_pD": float(np.round(np.mean(pD_sl), 10)),
        "max_pD": float(np.round(np.max(pD_sl), 10)),
        "tree_pY": float(np.round(np.mean(tree_Y), 10)),
        "tree_max_pY": float(np.round(np.max(tree_Y), 10)),
        "seq_pY": float(np.round(np.mean(seq_Y), 10)),
        "seq_max_pY": float(np.round(np.max(seq_Y), 10)),
        "pset_pY": float(np.round(np.mean(pset_Y), 10)),
        "pset_max_pY": float(np.round(np.max(pset_Y), 10)),
        "seq_minus_tree": float(np.round(max(float(np.mean(seq_Y)), float(np.mean(pset_Y))) - float(np.mean(tree_Y)), 10)),
        "fam_vote_Y": float(np.round(fam_vote_Y, 10)),
        "fam_peak_vote_Y": float(np.round(fam_peak_vote_Y, 10)),
        "ps2_pY": float(np.round(np.mean(ps2_pY_sl), 10)),
        "ps2_max_pY": float(np.round(np.max(ps2_pY_sl), 10)),
        "ps2_int_pY": float(np.round(np.mean(ps_p[sl_int, y_lbl]), 10)),
        "ps2_pX": float(np.round(np.mean(ps2_pX_sl), 10)),
        "ps2_pD": float(np.round(np.mean(ps2_pD_sl), 10)),
        "ps2_b4d_in": float(np.round(ps2_b4d_in, 10)),
        "ps2_b4d_out": float(np.round(ps2_b4d_out, 10)),
        "ps2_b4d_geom": float(np.round(np.sqrt(max(0.0, ps2_b4d_in * ps2_b4d_out)), 10)),
        "ps2_b4d_asym": float(np.round(ps2_b4d_asym, 10)),
        "ps2_pc_in": float(np.round(ps2_pc_in, 10)),
        "ps2_pc_out": float(np.round(ps2_pc_out, 10)),
        "ps2_pc_geom": float(np.round(np.sqrt(max(0.0, ps2_pc_in * ps2_pc_out)), 10)),
        "ps2_air_int": float(np.round(np.mean(ps_air[sl_int]), 10)),
        "l1_tree_pY": float(np.round(np.mean(l1_pY_sl), 10)),
        "l1_tree_max_pY": float(np.round(np.max(l1_pY_sl), 10)),
        "l1_tree_int_pY": float(np.round(np.mean(l1_p[sl_int, y_lbl]), 10)),
        "l1_tree_pX": float(np.round(np.mean(l1_pX_sl), 10)),
        "l1_tree_pD": float(np.round(np.mean(l1_pD_sl), 10)),
        "l1_duel_in": float(np.round(l1_duel_in, 10)),
        "l1_duel_out": float(np.round(l1_duel_out, 10)),
        "l1_duel_geom": float(np.round(np.sqrt(max(0.0, l1_duel_in * l1_duel_out)), 10)),
        "l1_duel_int": float(np.round(np.mean(l1_duel[sl_int]), 10)),
        "l1_air_int": float(np.round(np.mean(l1_air[sl_int]), 10)),
        "v9_air_int": float(np.round(0.5 * (float(np.mean(ps_air[sl_int])) + float(np.mean(l1_air[sl_int]))), 10)),
        "ps2_surge_Y": float(np.round(float(np.mean(ps2_pY_sl)) - ps2_flk_Y, 10)),
        "l1_surge_Y": float(np.round(float(np.mean(l1_pY_sl)) - l1_flk_Y, 10)),
        "v85_surge_Y": float(np.round(float(np.mean(pY_sl)) - v85_flk_Y, 10)),
        "seq_surge_Y": float(np.round(float(np.mean(seq_Y)) - seq_flk_Y, 10)),
        "ps2_diff_YX": float(np.round(float(np.mean(ps2_pY_sl) - np.mean(ps2_pX_sl)), 10)),
        "l1_diff_YX": float(np.round(float(np.mean(l1_pY_sl) - np.mean(l1_pX_sl)), 10)),
        "min_d1_Y_int": float(np.round(min_d1_Y_int, 10)),
        "min_d1_X_int": float(np.round(min_d1_X_int, 10)),
        "d1_adv_int": float(np.round(d1_adv_int, 10)),
        "max_d1_adv_int": float(np.round(max_d1_adv_int, 10)),
        "y_win_frames": float(y_win_frames),
        "x_win_frames": float(x_win_frames),
        "duel_2m_frames": float(duel_2m_frames),
        "max_imp4_win": float(np.round(max_imp4_win, 10)),
        "min_cos4_win": float(np.round(min_cos4_win, 10)),
        "min_cos2_win": float(np.round(min_cos2_win, 10)),
        "max_ang_win": float(np.round(max_ang_win, 10)),
        "chord_int": float(np.round(chord_int, 10)),
        "frac_vis": float(np.round(frac_vis, 10)),
        "undeflected_aerial": float(np.round(undeflected_aerial, 10)),
        "t10_pY": 0.0,
        "t10_duel": 0.0,
        "t10_b4d_in": 0.0,
        "t10_b4d_out": 0.0,
        "t10_b4d_geom": 0.0,
    }
    res.update(dom)
    if t10 is not None:
        enrich_candidate_with_t10(res, t10)
    return res


def enrich_candidate_with_t10(c: Dict[str, Any], t10: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Enrich a candidate dict in-place with 100% H<->A equivariant `SymTrajectoryEventNetV10` features."""
    ca, cb = int(c["ca"]), int(c["cb"])
    y_lbl, x_lbl = int(c["y_lbl"]), int(c["x_lbl"])
    col_to_Y = 1 if y_lbl == HOME else 0
    col_to_X = 1 if x_lbl == HOME else 0
    kind = str(c.get("kind", "XYX"))
    out_col = col_to_X if kind != "XYD" else 2
    in_col = col_to_Y if kind != "DYX" else 3
    p_raw = t10["proba_raw"] if "proba_raw" in t10 else t10["proba"]
    p4d_t10 = t10["p4d"]
    n = len(p_raw)
    in_lo, in_hi = max(0, ca - 4), min(n, ca + 5)
    out_lo, out_hi = max(0, cb - 4), min(n, cb + 5)
    t10_in = float(np.max(p4d_t10[in_lo:in_hi, in_col]))
    t10_out = float(np.max(p4d_t10[out_lo:out_hi, out_col]))
    c["t10_pY"] = float(np.round(np.mean(p_raw[ca:cb, y_lbl]), 10))
    c["t10_duel"] = float(np.round(np.mean(t10["p_duel_to"][ca:cb]), 10))
    c["t10_b4d_in"] = float(np.round(t10_in, 10))
    c["t10_b4d_out"] = float(np.round(t10_out, 10))
    c["t10_b4d_geom"] = float(np.round(np.sqrt(max(0.0, t10_in * t10_out)), 10))
    return c


def enrich_candidates_with_t10(
    cands: Sequence[Dict[str, Any]],
    t10: Optional[Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Enrich a list of candidate dicts with `SymTrajectoryEventNetV10` features."""
    if t10 is None:
        return list(cands)
    return [enrich_candidate_with_t10(c, t10) for c in cands]


def extract_mbr_xyx_candidates_v10(
    pred: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    hl: np.ndarray,
    X_df: pd.DataFrame,
    ps: Dict[str, np.ndarray],
    l1: Dict[str, np.ndarray],
    gt: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Extract `X -> Y -> X` false-island absorption candidates enriched with `SymPlayerSetNetV2` & `L1-Trees`."""
    ps_p = ps["proba"]
    ps_p4d = ps["p4d"]
    ps_air = ps["p_aerial"]
    l1_p = (l1["gbm"] + l1["cb"] + l1["xgb"]) / 3.0
    l1_air = l1["air"]
    l1_duel = l1["duel"]
    v9_air = 0.5 * (ps_air + l1_air)
    v9_duel = l1_duel

    cands = extract_mbr_xyx_candidates_v9(
        pred, p, c, p4d, fams, hl, X_df, gt=gt, v9_air=v9_air, v9_duel=v9_duel
    )
    n = len(pred)
    out: List[Dict[str, Any]] = []
    for cand in cands:
        d = dict(cand)
        sa, sb = int(d["sa"]), int(d["sb"])
        x_lbl, y_lbl = int(d["x_lbl"]), int(d["y_lbl"])
        col_to_Y = 1 if y_lbl == HOME else 0
        col_to_X = 1 if x_lbl == HOME else 0
        in_lo, in_hi = max(0, sa - 4), min(n, sa + 5)
        out_lo, out_hi = max(0, sb - 4), min(n, sb + 5)
        ps2_in = float(np.max(ps_p4d[in_lo:in_hi, col_to_Y]))
        ps2_out = float(np.max(ps_p4d[out_lo:out_hi, col_to_X]))
        d["ps2_pY"] = float(np.round(np.mean(ps_p[sa:sb, y_lbl]), 10))
        d["ps2_pX"] = float(np.round(np.mean(ps_p[sa:sb, x_lbl]), 10))
        d["ps2_b4d_geom"] = float(np.round(np.sqrt(max(0.0, ps2_in * ps2_out)), 10))
        d["l1_tree_pY"] = float(np.round(np.mean(l1_p[sa:sb, y_lbl]), 10))
        d["l1_tree_pX"] = float(np.round(np.mean(l1_p[sa:sb, x_lbl]), 10))
        out.append(d)
    return out


def extract_touch_segment_proposals_v10(
    pred: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    hl: np.ndarray,
    X_df: pd.DataFrame,
    ps: Dict[str, np.ndarray],
    l1: Dict[str, np.ndarray],
    gt: Optional[np.ndarray] = None,
    t10: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Propose multi-modal `XYX`, `XYD`, and `DYX` short-possession candidates on top of `pred`."""
    kin = prep_kinematics_v10(X_df)
    ps_p, ps_c, ps_p4d = ps["proba"], ps["change"], ps["p4d"]
    l1_p = (l1["gbm"] + l1["cb"] + l1["xgb"]) / 3.0
    l1_duel = l1["duel"]

    if gt is not None:
        gt_off = to_official(gt, hl, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pred, hl, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_ov = pm.spell_overlaps(gt_spells, base_spells)
        n_gt, n_pr = len(gt_spells), len(base_spells)
        base_f1 = _f1_from_ov(base_ov, n_gt, n_pr)
        start_map = {sp[0]: idx for idx, sp in enumerate(base_spells)}
    else:
        gt_spells, base_spells, base_ov = [], [], []
        n_gt = n_pr = 0
        base_f1 = 0.0
        start_map = {}

    xyx_list: List[Dict[str, Any]] = []
    dyx_list: List[Dict[str, Any]] = []
    xyd_list: List[Dict[str, Any]] = []

    runs = rle_runs_half(pred, hl)
    for idx_r, (sa, sb, x_lbl, h_cur) in enumerate(runs):
        if x_lbl not in (HOME, AWAY) or h_cur <= 0:
            continue
        L_host = sb - sa
        y_lbl = 1 - x_lbl
        col_to_Y = 1 if y_lbl == HOME else 0
        col_to_X = 1 if x_lbl == HOME else 0
        shift = 1 if h_cur == 2 else 0
        sa_s, sb_s = sa + shift, sb + shift
        idx_host = start_map.get(sa_s, None) if gt is not None else None
        x_off = pm.H if x_lbl == HOME else pm.A
        y_off = pm.H if y_lbl == HOME else pm.A
        ov_without = [item for item in base_ov if item[2] != idx_host] if idx_host is not None else base_ov

        bdy_in = np.maximum.reduce([
            0.85 * c[sa:sb],
            p4d[sa:sb, col_to_Y],
            0.85 * ps_c[sa:sb],
            ps_p4d[sa:sb, col_to_Y],
            0.75 * l1_duel[sa:sb],
        ])
        bdy_out = np.maximum.reduce([
            0.85 * c[sa:sb],
            p4d[sa:sb, col_to_X],
            0.85 * ps_c[sa:sb],
            ps_p4d[sa:sb, col_to_X],
            0.75 * l1_duel[sa:sb],
        ])
        sig_y = np.maximum.reduce([
            p[sa:sb, y_lbl],
            ps_p[sa:sb, y_lbl],
            l1_p[sa:sb, y_lbl],
        ])

        host_ok = (gt is None) or (idx_host is not None and base_spells[idx_host][1] == sb_s)

        # 1. X -> Y -> X interior Double-Domino candidates
        if L_host >= 50 and host_ok:
            min_m = 14
            c_wins: List[Tuple[int, int]] = []
            c_wins_strict: List[Tuple[int, int]] = []
            for rel_a in range(min_m, L_host - min_m - 10):
                if bdy_in[rel_a] < 0.12:
                    continue
                if rel_a > 0 and bdy_in[rel_a] < bdy_in[rel_a - 1]:
                    continue
                if rel_a + 1 < L_host and bdy_in[rel_a] < bdy_in[rel_a + 1]:
                    continue
                b_start = rel_a + 10
                b_stop = min(L_host - min_m, rel_a + 90 + 1)
                if b_stop <= b_start:
                    continue
                rel_b = b_start + int(np.argmax(bdy_out[b_start:b_stop]))
                if bdy_out[rel_b] < 0.12:
                    continue
                if float(np.max(sig_y[rel_a:rel_b])) < 0.18:
                    continue
                ca, cb = int(sa + rel_a), int(sa + rel_b)
                if c_wins and ca < c_wins[-1][0] + 6:
                    continue
                c_wins.append((ca, cb))
                is_shadow = bool(c_wins_strict and ca < c_wins_strict[-1][1] + 8)
                if not is_shadow:
                    c_wins_strict.append((ca, cb))
                feat = extract_segment_roi_features_v10(
                    sa, sb, ca, cb, x_lbl, y_lbl, p, c, p4d, fams, ps, l1, kin, "XYX", t10=t10
                )
                feat["is_shadow_ca"] = int(is_shadow)
                if gt is not None and idx_host is not None:
                    ca_s, cb_s = ca + shift, cb + shift
                    new_ov = (
                        ov_without
                        + _single_spell_overlaps(gt_spells, (sa_s, ca_s, x_off), idx_host)
                        + _single_spell_overlaps(gt_spells, (ca_s, cb_s, y_off), n_pr)
                        + _single_spell_overlaps(gt_spells, (cb_s, sb_s, x_off), n_pr + 1)
                    )
                    new_ov.sort(key=lambda item: (-item[0], item[1], item[2]))
                    feat["dy"] = float(_f1_from_ov(new_ov, n_gt, n_pr + 2) - base_f1)
                xyx_list.append(feat)

        # 2. D -> Y -> X post-restart prefix candidates
        if (
            idx_r > 0
            and runs[idx_r - 1][2] == DEAD
            and runs[idx_r - 1][3] == h_cur
            and L_host >= 45
            and host_ok
        ):
            b_start, b_stop = 12, min(L_host - 20, 80)
            if b_stop > b_start:
                rel_b = b_start + int(np.argmax(bdy_out[b_start:b_stop]))
                if bdy_out[rel_b] >= 0.18 and float(np.max(sig_y[:rel_b])) >= 0.25:
                    cb = int(sa + rel_b)
                    feat = extract_segment_roi_features_v10(
                        sa, sb, sa, cb, x_lbl, y_lbl, p, c, p4d, fams, ps, l1, kin, "DYX", t10=t10
                    )
                    if gt is not None and idx_host is not None:
                        cb_s = cb + shift
                        new_ov = (
                            ov_without
                            + _single_spell_overlaps(gt_spells, (sa_s, cb_s, y_off), idx_host)
                            + _single_spell_overlaps(gt_spells, (cb_s, sb_s, x_off), n_pr)
                        )
                        new_ov.sort(key=lambda item: (-item[0], item[1], item[2]))
                        feat["dy"] = float(_f1_from_ov(new_ov, n_gt, n_pr + 1) - base_f1)
                    dyx_list.append(feat)

        # 3. X -> Y -> D pre-stoppage suffix candidates
        if (
            idx_r + 1 < len(runs)
            and runs[idx_r + 1][2] == DEAD
            and runs[idx_r + 1][3] == h_cur
            and L_host >= 45
            and host_ok
        ):
            a_start, a_stop = max(20, L_host - 80), L_host - 12
            if a_stop > a_start:
                rel_a0 = a_start + int(np.argmax(bdy_in[a_start:a_stop]))
                rel_a_cands = [rel_a0]
                if rel_a0 - a_start <= 8 and rel_a0 + 12 < a_stop:
                    rel_a2 = (rel_a0 + 12) + int(np.argmax(bdy_in[rel_a0 + 12 : a_stop]))
                    rel_a_cands.append(rel_a2)
                for rel_a in rel_a_cands:
                    if bdy_in[rel_a] >= 0.18 and float(np.max(sig_y[rel_a:])) >= 0.25:
                        ca = int(sa + rel_a)
                        feat = extract_segment_roi_features_v10(
                            sa, sb, ca, sb, x_lbl, y_lbl, p, c, p4d, fams, ps, l1, kin, "XYD", t10=t10
                        )
                        if gt is not None and idx_host is not None:
                            ca_s = ca + shift
                            new_ov = (
                                ov_without
                                + _single_spell_overlaps(gt_spells, (sa_s, ca_s, x_off), idx_host)
                                + _single_spell_overlaps(gt_spells, (ca_s, sb_s, y_off), n_pr)
                            )
                            new_ov.sort(key=lambda item: (-item[0], item[1], item[2]))
                            feat["dy"] = float(_f1_from_ov(new_ov, n_gt, n_pr + 1) - base_f1)
                        xyd_list.append(feat)

    return {"XYX": xyx_list, "DYX": dyx_list, "XYD": xyd_list}


# ======================================================================================
# PART 3A: SEGMENT-LATTICE NEURAL ARBITRATOR (`SymTouchSegmentLatticeV10`)
# ======================================================================================


def score_mbr_v10(c: Dict[str, Any]) -> float:
    """Equivariant score for `MBR-v10` (`X -> Y -> X` false-island absorption sub-regimes `A4-A6`)."""
    # Sub-regime A4: Driven ballistic pass-through rejected by SymPlayerSetNetV2 & SymPlayerSetNetV1
    is_a4 = (
        c["ps2_pY"] <= 0.16
        and c["ps2_pX"] >= 0.80
        and c["ps2_b4d_geom"] <= 0.16
        and c["chord_mean"] >= 0.35
        and c["pset_pY"] <= 0.25
    )
    # Sub-regime A5: Shadow-pressing parallel run where both Player-Set networks side with host X
    is_a5 = (
        c["eff_contact_Y"] >= 24.0
        and c["ps2_pY"] <= 0.30
        and c["ps2_pX"] >= 0.68
        and c["pset_pY"] <= 0.48
        and c["min_d1_X"] >= 0.12
    )
    # Sub-regime A6: High-altitude 3D overhead flight with zero duel turnover
    is_a6 = (
        48 <= c["L"] <= 60
        and 1.0 <= c["eff_contact_Y"] <= 4.0
        and c["chord_mean"] >= 0.68
        and c["v9_air_mean"] >= 0.68
        and c["v9_duel_mean"] <= 0.105
        and c["ps2_pY"] <= 0.90
        and c["pset_pY"] <= 0.70
    )
    if not (is_a4 or is_a5 or is_a6):
        return -999.0
    return float(np.round(0.50 * c["ps2_pX"] + 0.30 * (1.0 - c["ps2_pY"]) + 0.20 * c["chord_mean"], 6))


def score_xyx_v10(c: Dict[str, Any]) -> float:
    """Equivariant score for `XYX-v10` (`X -> Y -> X` interior Double-Domino short steals, 5 sub-regimes)."""
    if int(c.get("is_shadow_ca", 0)) == 1:
        return -999.0
    # Regime XYX-1: Analytical Zero-Loss Neural Consensus (flank_max_ratio >= 0.70 => tp_loss_if_false == 0)
    r1 = (
        c["flank_max_ratio"] >= 0.70
        and 20 <= c["L"] <= 88
        and c["ps2_pY"] >= 0.78
        and c["ps2_b4d_geom"] >= 0.925
        and c["l1_tree_pY"] >= 0.45
        and c["pset_pY"] >= 0.55
        and c["mean_pD"] <= 0.10
    )
    # Regime XYX-2: 3D Aerial Header / Clearance Steal
    r2 = (
        c["x_win_frames"] == 0
        and c["y_win_frames"] >= 5
        and c["min_d1_Y_int"] <= 0.75
        and c["min_d1_X_int"] >= 1.20
        and c["d1_adv_int"] >= 0.80
        and c["ps2_pY"] >= 0.55
        and c["ps2_b4d_geom"] >= 0.65
        and c["b4d_geom"] >= 0.12
        and c["l1_tree_pY"] >= 0.40
        and c["v9_air_int"] >= 0.35
        and c["mean_pD"] <= 0.10
        and c["undeflected_aerial"] <= 0.15
    )
    # Regime XYX-3: Interior Double-Domino with Multi-Family Consensus & Distance Advantage
    r3 = (
        c["flank_max_ratio"] >= 0.42
        and 30 <= c["L"] <= 75
        and c["ps2_pY"] >= 0.82
        and c["ps2_b4d_geom"] >= 0.92
        and c["l1_tree_pY"] >= 0.43
        and c["seq_pY"] >= 0.50
        and c["pset_pY"] >= 0.55
        and c["d1_adv_int"] >= -0.25
        and c["x_win_frames"] <= 3
        and c["mean_pD"] <= 0.10
    )
    # Regime XYX-4: Fast 1-Touch / Short Duel Steal (0.40s..1.00s)
    r4 = (
        c["flank_max_ratio"] >= 0.70
        and 10 <= c["L"] <= 25
        and c["min_d1_Y_int"] <= 0.25
        and c["min_d1_X_int"] >= 1.10
        and c["d1_adv_int"] >= 0.50
        and c["y_win_frames"] >= 7
        and c["x_win_frames"] <= 1
        and c["l1_tree_pY"] >= 0.60
        and c["l1_duel_geom"] >= 0.55
        and c["ps2_b4d_geom"] >= 0.75
        and c["b4d_geom"] >= 0.30
        and c["mean_pD"] <= 0.10
    )
    # Regime XYX-5: Ball-Dropout 22-Player Set Turnover (when ball tracking is missing d1_Y >= 49m)
    r5 = (
        c["min_d1_Y_int"] >= 49.0
        and 45 <= c["L"] <= 75
        and c["ps2_pY"] >= 0.72
        and c["ps2_b4d_geom"] >= 0.70
        and c["l1_tree_pY"] >= 0.55
        and c["mean_pD"] <= 0.10
    )
    # Regime XYX-6: SymTrajectoryEventNetV10 Spatio-Temporal Duel & Broadcast-Dropout Consensus
    t10_pY = float(c.get("t10_pY", 0.0))
    t10_duel = float(c.get("t10_duel", 0.0))
    t10_b4d_geom = float(c.get("t10_b4d_geom", 0.0))
    r6a = (
        c["flank_max_ratio"] >= 0.45
        and c["x_win_frames"] == 0
        and c["y_win_frames"] >= 10
        and c["min_d1_Y_int"] <= 0.40
        and c["d1_adv_int"] >= 0.80
        and t10_pY >= 0.70
        and t10_duel >= 0.65
        and t10_b4d_geom >= 0.78
        and c["ps2_pY"] >= 0.50
        and c["l1_tree_pY"] >= 0.32
        and c["mean_pD"] <= 0.10
    )
    r6b = (
        c["flank_max_ratio"] >= 0.45
        and 30 <= c["L"] <= 75
        and c["x_win_frames"] == 0
        and c["d1_adv_int"] >= 1.00
        and c["ps2_pY"] >= 0.82
        and c["ps2_b4d_geom"] >= 0.92
        and c["l1_tree_pY"] >= 0.50
        and t10_pY >= 0.88
        and t10_duel >= 0.80
        and t10_b4d_geom >= 0.85
        and c["mean_pD"] <= 0.10
    )
    if not (r1 or r2 or r3 or r4 or r5 or r6a or r6b):
        return -999.0
    return float(
        np.round(
            0.35 * c["ps2_pY"]
            + 0.30 * c["ps2_b4d_geom"]
            + 0.20 * c["l1_tree_pY"]
            + 0.15 * c["b4d_geom"],
            6,
        )
    )


def score_xyd_v10(c: Dict[str, Any]) -> float:
    """Equivariant score for `XYD-v10` (`X -> Y -> D` pre-stoppage clearance/foul-won suffix turnovers)."""
    t10_pY = float(c.get("t10_pY", 0.0))
    t10_duel = float(c.get("t10_duel", 0.0))
    t10_b4d_in = float(c.get("t10_b4d_in", 0.0))
    r4 = (
        c["flank_max_ratio"] >= 0.55
        and 18 <= c["L"] <= 45
        and c["x_win_frames"] == 0
        and c["y_win_frames"] >= 18
        and c["min_d1_Y_int"] <= 0.25
        and c["d1_adv_int"] >= 0.80
        and t10_pY >= 0.75
        and t10_duel >= 0.80
        and t10_b4d_in >= 0.35
        and c["l1_tree_pY"] >= 0.55
        and c["mean_pD"] <= 0.20
        and c["max_imp4_win"] <= 35.0
    )
    base_ok = (
        c["flank_max_ratio"] >= 0.65
        and 14 <= c["L"] <= 76
        and c["x_win_frames"] == 0
        and c["y_win_frames"] >= 4
        and c["min_d1_Y_int"] <= 0.75
        and c["d1_adv_int"] >= 0.05
        and c["pset_pY"] >= 0.20
        and c["mean_pD"] <= 0.30
        and c["max_imp4_win"] <= 35.0
    )
    if not (base_ok or r4):
        return -999.0
    r1 = base_ok and (
        c["L"] <= 56
        and c["ps2_pY"] >= 0.55
        and c["ps2_diff_YX"] >= 0.10
        and c["b4d_geom"] >= 0.65
        and c["mean_pD"] <= 0.15
    )
    r2 = base_ok and (
        c["L"] <= 56
        and c["ps2_b4d_geom"] >= 0.75
        and c["ps2_pY"] >= 0.60
        and c["ps2_diff_YX"] >= 0.25
        and c["seq_pY"] >= 0.30
        and c["pset_pY"] >= 0.25
        and c["y_win_frames"] >= 8
        and c["min_d1_Y_int"] <= 0.65
    )
    r3 = base_ok and (
        c["min_d1_X_int"] >= 4.5
        and c["d1_adv_int"] >= 5.0
        and c["y_win_frames"] >= 14
        and c["l1_tree_pY"] >= 0.55
        and c["ps2_b4d_geom"] >= 0.70
        and c["pset_pY"] >= 0.25
    )
    if not (r1 or r2 or r3 or r4):
        return -999.0
    return float(
        np.round(
            0.35 * c["ps2_pY"]
            + 0.30 * c["ps2_b4d_geom"]
            + 0.20 * c["l1_tree_pY"]
            + 0.15 * c["b4d_geom"],
            6,
        )
    )


def score_dyx_v10(c: Dict[str, Any]) -> float:
    """Equivariant score for `DYX-v10` (`D -> Y -> X` post-restart aerial/second-ball prefix turnovers)."""
    r1 = (
        c["flank_max_ratio"] >= 0.78
        and c["L"] >= 60
        and c["ps2_pY"] >= 0.55
        and c["pset_pY"] >= 0.40
        and c["ps2_b4d_out"] >= 0.75
        and c["seq_pY"] >= 0.24
        and c["mean_pD"] <= 0.25
    )
    r2 = (
        c["v9_air_int"] >= 0.88
        and c["l1_duel_out"] >= 0.75
        and c["b4d_out_X"] >= 0.35
        and c["ps2_b4d_out"] >= 0.60
        and c["y_win_frames"] >= 12
        and c["x_win_frames"] <= 1
        and c["mean_pD"] <= 0.20
    )
    if not (r1 or r2):
        return -999.0
    return float(np.round(0.40 * c["ps2_pY"] + 0.35 * c["ps2_b4d_out"] + 0.25 * c["l1_duel_out"], 6))


def fit_threshold_lomo(
    train_lists: Sequence[Sequence[Dict[str, Any]]],
    score_fn: Callable[[Dict[str, Any]], float],
    grid: np.ndarray = np.linspace(0.20, 0.75, 23),
) -> Tuple[bool, float]:
    """Fit activation threshold on training folds maximizing exact `spellF1` gain (`dy`)."""
    valid: List[Tuple[float, float]] = []
    for clist in train_lists:
        by_p: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for c in clist:
            s = score_fn(c)
            if s <= -100.0:
                continue
            k = (int(c.get("sa", c.get("ca", 0))), int(c.get("sb", c.get("cb", 0))))
            if k not in by_p or s > by_p[k][0]:
                by_p[k] = (s, float(c.get("dy", 0.0)))
        valid.extend(by_p.values())
    if not valid:
        return False, 999.0
    best_th = float(grid[0])
    best_g = -1e9
    for th in grid:
        fires = [dy for s, dy in valid if s >= th]
        if not fires:
            continue
        g = float(np.sum(fires))
        if g > best_g:
            best_g = g
            best_th = float(th)
    return (best_g > 0.0), (best_th if best_g > 0.0 else 999.0)


def apply_scored_candidates(
    pred: np.ndarray,
    clist: Sequence[Dict[str, Any]],
    score_fn: Callable[[Dict[str, Any]], float],
    active: bool,
    th: float,
    use_x_lbl: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Apply highest-scoring candidate per host spell `(sa, sb)`."""
    out = np.asarray(pred, dtype=np.int64).copy()
    if not active:
        return out, []
    by_p: Dict[Tuple[int, int], List[Tuple[float, Dict[str, Any]]]] = {}
    for c in clist:
        s = score_fn(c)
        if s < th:
            continue
        k = (int(c.get("sa", c.get("ca", 0))), int(c.get("sb", c.get("cb", 0))))
        by_p.setdefault(k, []).append((s, c))

    selected: List[Tuple[float, Dict[str, Any]]] = []
    for (sa, sb), items in by_p.items():
        items.sort(key=lambda x: (-x[0], int(x[1].get("ca", x[1].get("sa", 0)))))
        selected.append(items[0])

    fired: List[Dict[str, Any]] = []
    for s, c in sorted(selected, key=lambda x: (-x[0], int(x[1].get("ca", x[1].get("sa", 0))))):
        a = int(c["sa"] if use_x_lbl else c["ca"])
        b = int(c["sb"] if use_x_lbl else c["cb"])
        lbl = int(c["x_lbl"] if use_x_lbl else c["y_lbl"])
        out[a:b] = lbl
        rec = dict(c)
        rec["score"] = float(s)
        fired.append(rec)
    return out, fired


@dataclass
class SymTouchSegmentLatticeConfigV10:
    """Learned LOMO parameters for `SymTouchSegmentLatticeV10`."""
    active_mbr: bool = True
    th_mbr: float = 0.20
    active_xyx: bool = True
    th_xyx: float = 0.20
    active_xyd: bool = True
    th_xyd: float = 0.20
    active_dyx: bool = True
    th_dyx: float = 0.20


class SymTouchSegmentLatticeV10:
    """First-Principles Multi-Modal Segment-Lattice Arbitrator (`MBR-v10`, `XYX-v10`, `XYD-v10`, `DYX-v10`)."""

    def __init__(self, cfg: Optional[SymTouchSegmentLatticeConfigV10] = None) -> None:
        self.cfg = cfg or SymTouchSegmentLatticeConfigV10()

    def fit_lomo(
        self,
        train_mbr_lists: Sequence[Sequence[Dict[str, Any]]],
        train_xyx_lists: Sequence[Sequence[Dict[str, Any]]],
        train_xyd_lists: Sequence[Sequence[Dict[str, Any]]],
        train_dyx_lists: Sequence[Sequence[Dict[str, Any]]],
    ) -> "SymTouchSegmentLatticeV10":
        self.cfg.active_mbr, self.cfg.th_mbr = fit_threshold_lomo(train_mbr_lists, score_mbr_v10)
        self.cfg.active_xyx, self.cfg.th_xyx = fit_threshold_lomo(train_xyx_lists, score_xyx_v10)
        self.cfg.active_xyd, self.cfg.th_xyd = fit_threshold_lomo(train_xyd_lists, score_xyd_v10)
        self.cfg.active_dyx, self.cfg.th_dyx = fit_threshold_lomo(train_dyx_lists, score_dyx_v10)
        return self

    def apply_from_candidates(
        self,
        pred: np.ndarray,
        mbr_cands: Sequence[Dict[str, Any]],
        xyx_cands: Sequence[Dict[str, Any]],
        xyd_cands: Sequence[Dict[str, Any]],
        dyx_cands: Sequence[Dict[str, Any]],
        enable_mbr: bool = True,
        enable_xyx: bool = True,
        enable_xyd: bool = True,
        enable_dyx: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, List[Dict[str, Any]]]]:
        out = np.asarray(pred, dtype=np.int64).copy()
        fired: Dict[str, List[Dict[str, Any]]] = {"MBR": [], "XYX": [], "XYD": [], "DYX": []}
        if enable_mbr:
            out, fired["MBR"] = apply_scored_candidates(
                out, mbr_cands, score_mbr_v10, self.cfg.active_mbr, self.cfg.th_mbr, use_x_lbl=True
            )
        if enable_xyx:
            out, fired["XYX"] = apply_scored_candidates(
                out, xyx_cands, score_xyx_v10, self.cfg.active_xyx, self.cfg.th_xyx, use_x_lbl=False
            )
        if enable_xyd:
            out, fired["XYD"] = apply_scored_candidates(
                out, xyd_cands, score_xyd_v10, self.cfg.active_xyd, self.cfg.th_xyd, use_x_lbl=False
            )
        if enable_dyx:
            out, fired["DYX"] = apply_scored_candidates(
                out, dyx_cands, score_dyx_v10, self.cfg.active_dyx, self.cfg.th_dyx, use_x_lbl=False
            )
        return out, fired

    def apply(
        self,
        pred: np.ndarray,
        p: np.ndarray,
        c: np.ndarray,
        p4d: np.ndarray,
        fams: Sequence[np.ndarray],
        hl: np.ndarray,
        X_df: pd.DataFrame,
        ps: Dict[str, np.ndarray],
        l1: Dict[str, np.ndarray],
        t10: Optional[Dict[str, np.ndarray]] = None,
        enable_mbr: bool = True,
        enable_xyx: bool = True,
        enable_xyd: bool = True,
        enable_dyx: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, List[Dict[str, Any]]]]:
        mbr_cands = extract_mbr_xyx_candidates_v10(pred, p, c, p4d, fams, hl, X_df, ps, l1, gt=None)
        props = extract_touch_segment_proposals_v10(pred, p, c, p4d, fams, hl, X_df, ps, l1, gt=None, t10=t10)
        return self.apply_from_candidates(
            pred,
            mbr_cands,
            props["XYX"],
            props["XYD"],
            props["DYX"],
            enable_mbr=enable_mbr,
            enable_xyx=enable_xyx,
            enable_xyd=enable_xyd,
            enable_dyx=enable_dyx,
        )


# ======================================================================================
# PART 3B: PRE-TOUCH LIVE BOUNDARY ALIGNER (`SymTouchBoundaryAlignerV10`, `B2`)
# ======================================================================================


def build_b2_v10_features(
    tb: int,
    btype: str,
    s_old: int,
    s_new: int,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    ps_p: np.ndarray,
    ps_c: np.ndarray,
    ps_p4d: np.ndarray,
    l1_p: np.ndarray,
    l1_duel: np.ndarray,
    cxv: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute `(41, 24)` 100% H<->A-invariant feature matrix across `DELTAS_20 = [-20..+20]`."""
    T = len(p)
    feats = np.zeros((len(DELTAS_20), 24), dtype=np.float64)

    p_tree = (fams[0] + fams[1] + fams[2] + fams[3]) / 4.0
    p_seq244 = (fams[4] + fams[5]) / 2.0
    p_seq481 = (fams[6] + fams[7]) / 2.0 if len(fams) >= 9 else p_seq244
    p_seq = (p_seq244 + p_seq481) / 2.0
    p_pset1 = fams[-1]

    p_diff = p[:, s_new] - p[:, s_old]
    pt_diff = p_tree[:, s_new] - p_tree[:, s_old]
    psq_diff = p_seq[:, s_new] - p_seq[:, s_old]
    pp1_diff = p_pset1[:, s_new] - p_pset1[:, s_old]
    ps2_diff = ps_p[:, s_new] - ps_p[:, s_old]
    l1_diff = l1_p[:, s_new] - l1_p[:, s_old]

    col_dir = 1 if s_new == HOME else 0
    col_rev = 0 if s_new == HOME else 1
    b4d_dir = p4d[:, col_dir]
    b4d_rev = p4d[:, col_rev]
    ps2_b4d_dir = ps_p4d[:, col_dir]
    ps2_b4d_rev = ps_p4d[:, col_rev]
    d1_new = cxv["dH1"] if s_new == HOME else cxv["dA1"]
    d1_ig_new = cxv["dH1_ig"] if s_new == HOME else cxv["dA1_ig"]
    d1_ig_old = cxv["dH1_ig"] if s_old == HOME else cxv["dA1_ig"]

    for idx_d, d in enumerate(DELTAS_20):
        tc = int(np.clip(tb + int(d), 6, T - 7))
        if d == 0:
            seg_p_gain = seg_tree_gain = seg_seq_gain = seg_pset1_gain = seg_ps2_gain = seg_l1_gain = 0.0
        elif d < 0:
            sl = slice(max(0, tb + int(d)), tb)
            seg_p_gain = float(np.sum(p_diff[sl])) / 15.0
            seg_tree_gain = float(np.sum(pt_diff[sl])) / 15.0
            seg_seq_gain = float(np.sum(psq_diff[sl])) / 15.0
            seg_pset1_gain = float(np.sum(pp1_diff[sl])) / 15.0
            seg_ps2_gain = float(np.sum(ps2_diff[sl])) / 15.0
            seg_l1_gain = float(np.sum(l1_diff[sl])) / 15.0
        else:
            sl = slice(tb, min(T, tb + int(d)))
            seg_p_gain = float(np.sum(-p_diff[sl])) / 15.0
            seg_tree_gain = float(np.sum(-pt_diff[sl])) / 15.0
            seg_seq_gain = float(np.sum(-psq_diff[sl])) / 15.0
            seg_pset1_gain = float(np.sum(-pp1_diff[sl])) / 15.0
            seg_ps2_gain = float(np.sum(-ps2_diff[sl])) / 15.0
            seg_l1_gain = float(np.sum(-l1_diff[sl])) / 15.0

        dp_step6 = float(np.mean(p_diff[tc : tc + 6]) - np.mean(p_diff[tc - 6 : tc]))
        ps2_step6 = float(np.mean(ps2_diff[tc : tc + 6]) - np.mean(ps2_diff[tc - 6 : tc]))
        l1_step6 = float(np.mean(l1_diff[tc : tc + 6]) - np.mean(l1_diff[tc - 6 : tc]))

        b4d_dir_tc = float(np.max(b4d_dir[tc - 1 : tc + 2]))
        b4d_rev_tc = float(b4d_rev[tc])
        ps2_b4d_dir_tc = float(np.max(ps2_b4d_dir[tc - 1 : tc + 2]))
        ps2_b4d_rev_tc = float(ps2_b4d_rev[tc])
        pch_tc = float(np.max(c[tc - 1 : tc + 2]))
        ps2_pch_tc = float(np.max(ps_c[tc - 1 : tc + 2]))
        l1_duel_tc = float(np.max(l1_duel[tc - 1 : tc + 2]))

        imp_d2_tc = float(cxv["imp_d2"][tc]) / 30.0
        imp_d4_tc = float(cxv["imp_d4"][tc]) / 30.0
        ang_tc = float(cxv["bang"][tc]) / 180.0
        vis_tc = float(cxv["vis"][tc])

        d1_new_tc = float(d1_ig_new[tc]) / 30.0
        d1_old_tc = float(d1_ig_old[tc]) / 30.0
        contact_new_tc = 1.0 if (vis_tc > 0.5 and float(d1_new[tc]) < 1.40) else 0.0
        offset_pen = abs(float(d)) / 20.0

        feats[idx_d] = np.round([
            seg_p_gain, seg_tree_gain, seg_seq_gain, seg_pset1_gain, seg_ps2_gain, seg_l1_gain,
            dp_step6, ps2_step6, l1_step6,
            b4d_dir_tc, b4d_rev_tc, ps2_b4d_dir_tc, ps2_b4d_rev_tc,
            pch_tc, ps2_pch_tc, l1_duel_tc,
            imp_d2_tc, imp_d4_tc, ang_tc, vis_tc,
            d1_new_tc, d1_old_tc, contact_new_tc, offset_pen,
        ], 10)
    return feats


def extract_b2_v10_boundary_candidates(
    pred: np.ndarray,
    half: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    ps: Dict[str, np.ndarray],
    l1: Dict[str, np.ndarray],
    X_df: pd.DataFrame,
    labels: Optional[np.ndarray] = None,
    min_adj_len: int = 24,
) -> List[Dict[str, Any]]:
    """Extract live `H <-> A` boundary candidates (`DELTAS_20 = [-20..+20]`) with 24 equivariant features."""
    T = len(pred)
    cxv = _extract_kinematic_arrays(X_df, T)
    ps_p, ps_c, ps_p4d = ps["proba"], ps["change"], ps["p4d"]
    l1_p = (l1["gbm"] + l1["cb"] + l1["xgb"]) / 3.0
    l1_duel = l1["duel"]

    if labels is not None:
        gt_off = to_official(labels, half, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pred, half, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_ov = pm.spell_overlaps(gt_spells, base_spells)
        n_gt, n_pr = len(gt_spells), len(base_spells)
        base_f1 = _f1_from_ov(base_ov, n_gt, n_pr)
        start_map = {sp[0]: idx for idx, sp in enumerate(base_spells)}
        end_map = {sp[1]: idx for idx, sp in enumerate(base_spells)}
    else:
        gt_spells, base_spells, base_ov = [], [], []
        n_gt = n_pr = 0
        base_f1 = 0.0
        start_map, end_map = {}, {}

    runs = rle_runs_half(pred, half)
    out: List[Dict[str, Any]] = []
    for k in range(len(runs) - 1):
        a1, b1, s_old, h1 = runs[k]
        a2, b2, s_new, h2 = runs[k + 1]
        if h1 <= 0 or h1 != h2 or b1 != a2:
            continue
        tb = b1
        L_left = b1 - a1
        L_right = b2 - a2
        if L_left < min_adj_len or L_right < min_adj_len:
            continue
        if not (s_old in (HOME, AWAY) and s_new in (HOME, AWAY)):
            continue
        btype = "HA"
        feats = build_b2_v10_features(
            tb, btype, s_old, s_new, p, c, p4d, fams, ps_p, ps_c, ps_p4d, l1_p, l1_duel, cxv
        )
        valid_mask = np.ones(len(DELTAS_20), dtype=bool)
        vis_tb = float(cxv["vis"][tb]) > 0.5
        gd_tb = float(cxv["gdisc"][tb]) <= 22.0
        for idx_d, d in enumerate(DELTAS_20):
            tc = tb + int(d)
            if tc < 6 or tc >= T - 7:
                valid_mask[idx_d] = False
                continue
            if L_left + int(d) < min_adj_len or L_right - int(d) < min_adj_len:
                valid_mask[idx_d] = False
                continue
            vis_tc = float(cxv["vis"][tc]) > 0.5
            gd_tc = float(cxv["gdisc"][tc]) <= 22.0
            if ((not vis_tb) or (not vis_tc) or (not gd_tb) or (not gd_tc)) and d != 0:
                valid_mask[idx_d] = False

        cand: Dict[str, Any] = {
            "tb": int(tb),
            "btype": btype,
            "s_old": int(s_old),
            "s_new": int(s_new),
            "L_left": int(L_left),
            "L_right": int(L_right),
            "feats": feats,
            "valid_mask": valid_mask,
        }
        if labels is not None:
            f1_deltas = np.zeros(len(DELTAS_20), dtype=np.float64)
            iou_deltas = np.zeros(len(DELTAS_20), dtype=np.float64)
            shift = 1 if h1 == 2 else 0
            tb_s = tb + shift
            idx_left = end_map.get(tb_s, None)
            idx_right = start_map.get(tb_s, None)
            rem_set = {x for x in (idx_left, idx_right) if x is not None}
            ov_without = [item for item in base_ov if item[2] not in rem_set]
            ov_orig_lr = [item for item in base_ov if item[2] in rem_set]
            orig_iou_sum = sum(item[0] for item in ov_orig_lr)

            for idx_d, d in enumerate(DELTAS_20):
                if not valid_mask[idx_d] or d == 0:
                    continue
                if idx_left is not None and idx_right is not None:
                    sp_l = (base_spells[idx_left][0], tb_s + int(d), base_spells[idx_left][2])
                    sp_r = (tb_s + int(d), base_spells[idx_right][1], base_spells[idx_right][2])
                    new_lr = _single_spell_overlaps(gt_spells, sp_l, idx_left) + _single_spell_overlaps(
                        gt_spells, sp_r, idx_right
                    )
                else:
                    new_lr = []
                iou_deltas[idx_d] = sum(item[0] for item in new_lr) - orig_iou_sum
                if abs(iou_deltas[idx_d]) > 1e-9:
                    ov_mod = ov_without + new_lr
                    ov_mod.sort(key=lambda item: (-item[0], item[1], item[2]))
                    f1_deltas[idx_d] = _f1_from_ov(ov_mod, n_gt, n_pr) - base_f1
            cand["f1_deltas"] = f1_deltas
            cand["iou_deltas"] = iou_deltas
        out.append(cand)
    return out


class SymTouchBoundaryAlignerV10:
    """First-Principles Pre-Touch Live `H <-> A` Boundary Aligner (`B2-v10`).

    Advances delayed `s_old -> s_new` boundaries (`d in [-20, -1]`) to the incoming player's
    physical touch (`d1_new <= 0.60m`) when our `L1`-weighted `SymPlayerSetNetV2` (`seg_ps2_gain`)
    and `L1-Trees` (`seg_l1_gain`) detect the pre-boundary turnover that unweighted `v8.5` trees
    missed (`seg_l1_gain > seg_tree_gain + 0.05` and `seg_tree_gain <= 0.25`, or full multi-family
    pre-touch consensus with `d1_new <= 0.45m`).
    """

    def __init__(self, active: bool = True) -> None:
        self.active = active

    @staticmethod
    def score_offsets(c: Dict[str, Any]) -> np.ndarray:
        F = c["feats"]
        r1 = (
            (DELTAS_20 < 0)
            & (F[:, 20] <= 0.60 / 30.0)
            & (F[:, 4] >= 0.25)
            & (F[:, 5] >= 0.14)
            & (F[:, 5] > F[:, 1] + 0.05)
            & (F[:, 1] <= 0.25)
        )
        r2 = (
            (DELTAS_20 < 0)
            & (F[:, 20] <= 0.45 / 30.0)
            & (F[:, 0] >= 0.0)
            & (F[:, 3] >= 0.40)
            & (F[:, 4] >= 0.30)
            & (F[:, 5] >= -0.01)
            & (F[:, 1] <= 0.25)
        )
        valid_mask = c["valid_mask"] & (r1 | r2)
        scores = np.round(F[:, 4] + 0.5 * F[:, 5] + 0.3 * F[:, 3], 6)
        scores[~valid_mask] = -999.0
        return scores

    def fit_lomo(self, train_b2_lists: Sequence[Sequence[Dict[str, Any]]]) -> "SymTouchBoundaryAlignerV10":
        tot_dy = 0.0
        n_fires = 0
        for cl in train_b2_lists:
            for c in cl:
                sc = self.score_offsets(c)
                bi = int(np.argmax(sc))
                if sc[bi] > -100.0:
                    tot_dy += float(c.get("f1_deltas", np.zeros(len(DELTAS_20)))[bi])
                    n_fires += 1
        self.active = (n_fires > 0) and (tot_dy >= 0.0)
        return self

    def apply_from_candidates(
        self,
        pred: np.ndarray,
        b2_cands: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        out = np.asarray(pred, dtype=np.int64).copy()
        if not self.active:
            return out, []
        fired: List[Dict[str, Any]] = []
        for c in b2_cands:
            tc = int(c["tb"])
            s_old, s_new = int(c["s_old"]), int(c["s_new"])
            if out[tc - 1] != s_old or out[tc] != s_new:
                continue
            scores = self.score_offsets(c)
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > -100.0:
                d_star = int(DELTAS_20[best_idx])
                if d_star < 0 and np.all(out[tc + d_star : tc] == s_old):
                    out[tc + d_star : tc] = s_new
                    fired.append({"tb": tc, "d_star": d_star, "s_old": s_old, "s_new": s_new, "score": float(scores[best_idx])})
                elif d_star > 0 and np.all(out[tc : tc + d_star] == s_new):
                    out[tc : tc + d_star] = s_old
                    fired.append({"tb": tc, "d_star": d_star, "s_old": s_old, "s_new": s_new, "score": float(scores[best_idx])})
        return out, fired

    def apply(
        self,
        pred: np.ndarray,
        half: np.ndarray,
        p: np.ndarray,
        c: np.ndarray,
        p4d: np.ndarray,
        fams: Sequence[np.ndarray],
        ps: Dict[str, np.ndarray],
        l1: Dict[str, np.ndarray],
        X_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        b2_cands = extract_b2_v10_boundary_candidates(pred, half, p, c, p4d, fams, ps, l1, X_df, labels=None)
        return self.apply_from_candidates(pred, b2_cands)


# ======================================================================================
# EXACT 0.000e+00 H <-> A EQUIVARIANCE VERIFICATION
# ======================================================================================


def _swap_labels(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.int64).copy()
    h_mask = out == HOME
    a_mask = out == AWAY
    out[h_mask] = AWAY
    out[a_mask] = HOME
    return out


def _swap_proba_3(p: np.ndarray) -> np.ndarray:
    out = np.asarray(p, dtype=np.float64).copy()
    out[:, [HOME, AWAY]] = out[:, [AWAY, HOME]]
    return out


def _swap_p4d(p4d: np.ndarray) -> np.ndarray:
    out = np.asarray(p4d, dtype=np.float64).copy()
    out[:, [0, 1]] = out[:, [1, 0]]
    return out


def _swap_physv2_df(X_df: pd.DataFrame) -> pd.DataFrame:
    df_sw = X_df.copy()
    for col in list(df_sw.columns):
        for pat_h, pat_a in (("_H", "_A"), ("_h_", "_a_"), ("home", "away")):
            if pat_h in col:
                other = col.replace(pat_h, pat_a)
                if other in df_sw.columns:
                    df_sw[col] = X_df[other].to_numpy().copy()
                    df_sw[other] = X_df[col].to_numpy().copy()
        if col.startswith("touch_team_r") or "touch_team_r" in col:
            if "abs_" not in col:
                df_sw[col] = -X_df[col].to_numpy().copy()
    return df_sw


def verify_touch_segment_lattice_equivariance(
    pred: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    hl: np.ndarray,
    X_df: pd.DataFrame,
    ps: Dict[str, np.ndarray],
    l1: Dict[str, np.ndarray],
    t10: Optional[Dict[str, np.ndarray]] = None,
    lattice: Optional[SymTouchSegmentLatticeV10] = None,
    aligner: Optional[SymTouchBoundaryAlignerV10] = None,
) -> Dict[str, Any]:
    """Verify exact `0.000e+00` H<->A equivariance across all proposals, features, scores, and predictions."""
    lattice = lattice or SymTouchSegmentLatticeV10()
    aligner = aligner or SymTouchBoundaryAlignerV10()

    pred_sw = _swap_labels(pred)
    p_sw = _swap_proba_3(p)
    p4d_sw = _swap_p4d(p4d)
    fams_sw = [_swap_proba_3(fp) for fp in fams]
    X_sw = _swap_physv2_df(X_df)
    ps_sw = {
        "proba": _swap_proba_3(ps["proba"]),
        "change": ps["change"].copy(),
        "p4d": _swap_p4d(ps["p4d"]),
        "p_aerial": ps["p_aerial"].copy(),
    }
    l1_sw = {
        "gbm": _swap_proba_3(l1["gbm"]),
        "cb": _swap_proba_3(l1["cb"]),
        "xgb": _swap_proba_3(l1["xgb"]),
        "air": l1["air"].copy(),
        "duel": l1["duel"].copy(),
    }
    t10_sw: Optional[Dict[str, np.ndarray]] = None
    if t10 is not None:
        t10_sw = {
            "proba": _swap_proba_3(t10["proba"]),
            "proba_raw": _swap_proba_3(t10["proba_raw"]) if "proba_raw" in t10 else _swap_proba_3(t10["proba"]),
            "change": t10["change"].copy(),
            "p4d": _swap_p4d(t10["p4d"]),
            "p_duel_to": t10["p_duel_to"].copy(),
            "p_aerial": t10["p_aerial"].copy(),
        }

    out1, f1 = lattice.apply(pred, p, c, p4d, fams, hl, X_df, ps, l1, t10=t10)
    out2, f2 = aligner.apply(out1, hl, p, c, p4d, fams, ps, l1, X_df)

    out1_sw, f1_sw = lattice.apply(pred_sw, p_sw, c, p4d_sw, fams_sw, hl, X_sw, ps_sw, l1_sw, t10=t10_sw)
    out2_sw, f2_sw = aligner.apply(out1_sw, hl, p_sw, c, p4d_sw, fams_sw, ps_sw, l1_sw, X_sw)

    mismatches_b1 = int(np.sum(_swap_labels(out1) != out1_sw))
    mismatches_b2 = int(np.sum(_swap_labels(out2) != out2_sw))

    max_score_diff = 0.0
    for k in ("MBR", "XYX", "XYD", "DYX"):
        s_orig = [float(x["score"]) for x in f1[k]]
        s_swap = [float(x["score"]) for x in f1_sw[k]]
        if len(s_orig) == len(s_swap) and s_orig:
            max_score_diff = max(max_score_diff, float(np.max(np.abs(np.array(s_orig) - np.array(s_swap)))))
        elif len(s_orig) != len(s_swap):
            max_score_diff = 1.0

    return {
        "equivariant": (mismatches_b1 == 0 and mismatches_b2 == 0 and max_score_diff == 0.0),
        "mismatches_b1": mismatches_b1,
        "mismatches_b2": mismatches_b2,
        "max_score_diff": float(max_score_diff),
        "fires_orig": {k: len(v) for k, v in f1.items()} | {"B2": len(f2)},
        "fires_swap": {k: len(v) for k, v in f1_sw.items()} | {"B2": len(f2_sw)},
    }
