"""First-Principles Bottleneck Breakdown, Swallowed-Steal Probability Audit & J_ATD_TWIN Stress Suite.

Implements:
1. Exact possession-spell failure-mode taxonomy (`classify_possessions_and_bottlenecks`):
   - Stratifies every ground-truth possession spell by duration (`< 1.0s`, `1.0-1.5s`, `1.5-2.0s`,
     `2.0-5.0s`, `> 5.0s`, plus `< 2.5s`, `< 2.0s`, `< 1.36s`) and by exact failure mode under the
     official 1-to-1 greedy IoU matcher (`evaluation/possession_metrics.py`):
     * `Swallowed_Double_Domino_Steal`: short GT spell `X` (`IoU < 0.3`) predicted predominantly as
       opponent `Y`, where a single predicted spell of `Y` bridges across `X` and merges >= 2 distinct
       GT spells of `Y` (`Y1 -> X -> Y2` merged into one `Y_pred`).
     * `Collateral_Double_Domino_Victim`: a surrounding GT spell (`Y1` or `Y2`) whose frames are
       predicted as the CORRECT team (`frac_same >= 0.50`) yet is missed (`IoU < 0.3`) because it was
       merged with another same-team GT spell across a swallowed opponent steal (`FN4` in PLAN_NOCHE).
     * `Swallowed_Opponent_NonDomino`: GT spell missed (`IoU < 0.3`), predicted predominantly as
       opponent (`frac_opp >= 0.50`), without merging two distinct opponent GT spells.
     * `Swallowed_By_Dead`: GT live possession missed (`IoU < 0.3`) with `frac_D >= 0.50` (or merged
       across a missed `DEAD` stoppage).
     * `Boundary_Shifted_0_to_03`: `0 < IoU < 0.3` (or fragmented into pieces `< 0.3`).
     * `Boundary_Shifted_03_to_07`: matched at `IoU >= 0.3` but fails at `IoU < 0.7`, attributed to
       `B2_team_boundaries` (`H <-> A`) vs `B3_dead_boundaries` (`Live <-> D`).
     * False-positive predicted spells (`IoU < 0.3`): split into `B4_false_blips` (`Y -> X -> Y`
       inside live play) and `B5_dead_wedges` (`D`-adjacent or inside GT `D`).

2. Swallowed Short-Possession Probability & Modal GPU/CPU Evidence Audit (`audit_swallowed_short_probabilities_j1a`):
   - Inspects every swallowed short possession (`< 2.5s`, `< 2.0s`, `< 1.36s`) in `J1A` across:
     * `v8.5` synthesized `proba`, `p_change`, `p4d`, and all 9 individual model families.
     * Modal GPU `SymPlayerSetNetV2` (`pset_v2_oof`, trained with `L1` spell weights + jersey-swap aug).
     * Modal CPU `L1-Trees` (`l1_trees_oof`, 3-tree `L1`-weighted ensemble + `SymAerialAndDuelGBM`).

3. Exact `spellF1` Oracle Ceiling Decomposition (`compute_exact_oracle_ceilings`):
   - Evaluates the exact marginal and cumulative `spell_f1_mean` ceilings of `B1_double_domino`,
     `B1_other_short`, `B1_all_short`, `B2_team_boundaries`, `B3_dead_boundaries`, `B4_false_blips`,
     and `B5_dead_wedges` on `v8.5` and `v9`.

4. `J_ATD_TWIN` Realistic Target-Domain Stress-Test Suite (`run_j_atd_twin_stress_suite`):
   - Evaluates three target-domain (`BAR-ATH`) physical perturbations on `J1A` (`Sportec`) with zero
     target-label leakage:
     * `J1_JerseySwap`: simulates `BAR-ATH`'s `12v10` wrong-team jersey swaps (`0.21` and `0.31`
       players/frame for `3.5s` stretches starting `>= 8m` from the ball, plus unshielded `>= 0m`).
     * `J1_UnbiasedDead`: neutralizes/masks the synthetic `d <= 1.24m` dead-ball player-attachment
       shortcut in `degrade_v3` during `D` and `Live <-> D` transitions.
     * `J1_RollingFoul`: simulates in-pitch foul `D` starts having rolling-ball lag (`1.5s-3.0s`)
       after the whistle while players decelerate collectively.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import evaluation.possession_metrics as pm
from src.data.common import AWAY, DEAD, FPS, HOME, possession_spells
from src.eval import metrics
from src.v9_swarm.whistle_and_stoppage import (
    StoppageStartRefiner,
    _half_bounds,
    _runs_of_with_half,
    detect_pitch_crossings_from_features,
)

DURATION_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("<1.0s", 1, 24),       # 1..24 frames (< 1.00s)
    ("1.0-1.5s", 25, 37),   # 25..37 frames (1.00s..1.48s)
    ("1.5-2.0s", 38, 50),   # 38..50 frames (1.52s..2.00s)
    ("2.0-5.0s", 51, 125),  # 51..125 frames (2.04s..5.00s)
    (">5.0s", 126, 999999), # >= 126 frames (> 5.00s)
)


def duration_bucket_name(length_frames: int) -> str:
    for name, lo, hi in DURATION_BUCKETS:
        if lo <= length_frames <= hi:
            return name
    return ">5.0s"


def _spells_with_half(codes: np.ndarray, half: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return `[(start, stop, label, half_id)]` for all live possession spells `(H, A)`."""
    s, e, lbl = possession_spells(codes, half)
    return [(int(a), int(b), int(k), int(half[a])) for a, b, k in zip(s, e, lbl)]


def _greedy_match_map(
    overlaps: List[Tuple[float, int, int]],
    n_gt: int,
    threshold: float,
) -> Tuple[Dict[int, Tuple[int, float]], Dict[int, Tuple[int, float]]]:
    """Return `(gt_to_pred, pred_to_gt)` under official greedy matching at `threshold`."""
    gt_to_pred: Dict[int, Tuple[int, float]] = {}
    pred_to_gt: Dict[int, Tuple[int, float]] = {}
    for iou, gi, pj in overlaps:
        if iou < threshold:
            break
        if gi in gt_to_pred or pj in pred_to_gt:
            continue
        gt_to_pred[gi] = (pj, float(iou))
        pred_to_gt[pj] = (gi, float(iou))
    return gt_to_pred, pred_to_gt


def classify_possessions_and_bottlenecks(
    gt_dict: Dict[str, np.ndarray],
    pred_dict: Dict[str, np.ndarray],
    half_dict: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Exact first-principles breakdown of GT and predicted spells across a match dictionary."""
    keys = sorted(gt_dict.keys())

    # Per-duration-bucket counters
    bucket_names = [b[0] for b in DURATION_BUCKETS] + ["ALL", "<2.5s", "<2.0s", "<1.36s"]
    categories = [
        "n_gt",
        "matched_iou_ge_07",
        "boundary_shifted_05_to_07",
        "boundary_shifted_03_to_05",
        "boundary_shifted_03_to_07_team",
        "boundary_shifted_03_to_07_dead",
        "missed_iou_lt_03_total",
        "swallowed_double_domino_steal",
        "collateral_double_domino_victim",
        "swallowed_opponent_nondomino",
        "swallowed_by_dead",
        "boundary_shifted_0_to_03",
    ]
    by_bucket: Dict[str, Dict[str, int]] = {
        b: {c: 0 for c in categories} for b in bucket_names
    }

    # False positive spell counters
    fp_summary = {
        "n_pred_total": 0,
        "matched_iou_ge_03": 0,
        "fp_iou_lt_03_total": 0,
        "B4_false_blips_live": 0,
        "B5_dead_wedges": 0,
        "fp_boundary_or_split": 0,
    }

    # Detailed list of swallowed short spells (< 2.5s = <= 62 frames) for probability audit
    swallowed_short_records: List[Dict[str, Any]] = []

    for m in keys:
        gt = np.asarray(gt_dict[m], dtype=np.int8)
        pr = np.asarray(pred_dict[m], dtype=np.int8)
        hl = np.asarray(half_dict[m], dtype=np.int8)
        T = len(gt)

        gt_spells_h = _spells_with_half(gt, hl)
        pr_spells_h = _spells_with_half(pr, hl)
        gt_spells = [(a, b, k) for a, b, k, _ in gt_spells_h]
        pr_spells = [(a, b, k) for a, b, k, _ in pr_spells_h]

        overlaps = pm.spell_overlaps(gt_spells, pr_spells)
        gt_m03, pr_m03 = _greedy_match_map(overlaps, len(gt_spells), 0.3)
        gt_m05, _ = _greedy_match_map(overlaps, len(gt_spells), 0.5)
        gt_m07, _ = _greedy_match_map(overlaps, len(gt_spells), 0.7)

        # Best same-team overlap regardless of greedy consumption
        best_same_iou = np.zeros(len(gt_spells), dtype=np.float64)
        best_same_pj = np.full(len(gt_spells), -1, dtype=np.int64)
        for iou, gi, pj in overlaps:
            if iou > best_same_iou[gi]:
                best_same_iou[gi] = float(iou)
                best_same_pj[gi] = int(pj)

        # Map each predicted spell pj -> list of GT spells of same team it overlaps with
        pred_to_same_gt_overlaps: Dict[int, List[int]] = {j: [] for j in range(len(pr_spells))}
        for gi, (ga, gb, gk, gh) in enumerate(gt_spells_h):
            for pj, (pa, pb, pk, ph) in enumerate(pr_spells_h):
                if gh == ph and gk == pk and max(ga, pa) < min(gb, pb):
                    pred_to_same_gt_overlaps[pj].append(gi)

        # Classify each GT spell
        for gi, (ga, gb, gk, gh) in enumerate(gt_spells_h):
            L = gb - ga
            bname = duration_bucket_name(L)
            active_buckets = [bname, "ALL"]
            if L <= 62:
                active_buckets.append("<2.5s")
            if L <= 50:
                active_buckets.append("<2.0s")
            if L < 34:
                active_buckets.append("<1.36s")

            for bk in active_buckets:
                by_bucket[bk]["n_gt"] += 1

            if gi in gt_m07:
                for bk in active_buckets:
                    by_bucket[bk]["matched_iou_ge_07"] += 1
                continue

            if gi in gt_m03:
                pj, iou_val = gt_m03[gi]
                pa, pb, _, _ = pr_spells_h[pj]
                # Attribute boundary shift to Team (H<->A) vs Dead (Live<->D)
                # Inspect frames where symmetric difference between [ga, gb) and [pa, pb) occurs
                err_team = 0
                err_dead = 0
                if ga < pa:
                    seg = pr[ga:min(gb, pa)]
                    err_dead += int(np.sum(seg == DEAD))
                    err_team += int(np.sum(seg == (1 - gk)))
                elif pa < ga:
                    seg = gt[pa:min(pb, ga)]
                    err_dead += int(np.sum(seg == DEAD))
                    err_team += int(np.sum(seg == (1 - gk)))
                if gb > pb:
                    seg = pr[max(ga, pb):gb]
                    err_dead += int(np.sum(seg == DEAD))
                    err_team += int(np.sum(seg == (1 - gk)))
                elif pb > gb:
                    seg = gt[max(pa, gb):pb]
                    err_dead += int(np.sum(seg == DEAD))
                    err_team += int(np.sum(seg == (1 - gk)))

                is_dead_shift = err_dead > err_team
                for bk in active_buckets:
                    if gi in gt_m05:
                        by_bucket[bk]["boundary_shifted_05_to_07"] += 1
                    else:
                        by_bucket[bk]["boundary_shifted_03_to_05"] += 1
                    if is_dead_shift:
                        by_bucket[bk]["boundary_shifted_03_to_07_dead"] += 1
                    else:
                        by_bucket[bk]["boundary_shifted_03_to_07_team"] += 1
                continue

            # Missed at IoU < 0.3
            for bk in active_buckets:
                by_bucket[bk]["missed_iou_lt_03_total"] += 1

            opp = 1 - gk
            seg_pr = pr[ga:gb]
            frac_opp = float(np.mean(seg_pr == opp))
            frac_d = float(np.mean(seg_pr == DEAD))
            frac_same = float(np.mean(seg_pr == gk))

            # Check if a single predicted spell of `opp` spans across [ga, gb) and overlaps >= 2 GT spells of `opp`
            is_double_domino_steal = False
            spanning_opp_pj = -1
            if frac_opp >= 0.50:
                for pj, (pa, pb, pk, ph) in enumerate(pr_spells_h):
                    if ph == gh and pk == opp and max(ga, pa) < min(gb, pb):
                        ov_len = min(gb, pb) - max(ga, pa)
                        if ov_len >= 0.50 * L and len(pred_to_same_gt_overlaps[pj]) >= 2:
                            # Ensure at least one overlapped opp GT spell is before gb and one after ga
                            opp_gts = pred_to_same_gt_overlaps[pj]
                            has_before = any(gt_spells_h[ogi][0] < ga for ogi in opp_gts)
                            has_after = any(gt_spells_h[ogi][1] > gb for ogi in opp_gts)
                            if has_before and has_after:
                                is_double_domino_steal = True
                                spanning_opp_pj = pj
                                break

            # Check if this spell is a Collateral Double-Domino Victim (predicted as the RIGHT team
            # `frac_same >= 0.50`, but merged into a single predicted spell `pj` of team `gk` that
            # spans >= 2 GT spells of team `gk` across a swallowed interruption)
            is_collateral_domino_victim = False
            if not is_double_domino_steal and frac_same >= 0.50:
                pj_same = int(best_same_pj[gi])
                if pj_same >= 0 and len(pred_to_same_gt_overlaps[pj_same]) >= 2:
                    is_collateral_domino_victim = True

            if is_double_domino_steal:
                cat = "swallowed_double_domino_steal"
            elif is_collateral_domino_victim:
                cat = "collateral_double_domino_victim"
            elif frac_d >= 0.50:
                cat = "swallowed_by_dead"
            elif frac_opp >= 0.50:
                cat = "swallowed_opponent_nondomino"
            else:
                cat = "boundary_shifted_0_to_03"

            for bk in active_buckets:
                by_bucket[bk][cat] += 1

            # Record swallowed short possessions (L <= 62 frames = < 2.5s) for probability audit
            if L <= 62 and cat in (
                "swallowed_double_domino_steal",
                "swallowed_opponent_nondomino",
                "swallowed_by_dead",
                "boundary_shifted_0_to_03",
            ):
                swallowed_short_records.append(
                    {
                        "match_id": m,
                        "gt_index": gi,
                        "ga": ga,
                        "gb": gb,
                        "L": L,
                        "dur_s": round(L / float(FPS), 3),
                        "bucket": bname,
                        "y_lbl": gk,
                        "x_lbl": opp,
                        "half": gh,
                        "failure_mode": cat,
                        "is_double_domino_steal": bool(is_double_domino_steal),
                        "frac_opp": round(frac_opp, 4),
                        "frac_d": round(frac_d, 4),
                        "frac_same": round(frac_same, 4),
                        "best_same_iou": round(float(best_same_iou[gi]), 4),
                        "spanning_opp_pj": int(spanning_opp_pj),
                    }
                )

        # Classify False Positive predicted spells (IoU < 0.3)
        fp_summary["n_pred_total"] += len(pr_spells)
        fp_summary["matched_iou_ge_03"] += len(pr_m03)
        for pj, (pa, pb, pk, ph) in enumerate(pr_spells_h):
            if pj in pr_m03:
                continue
            fp_summary["fp_iou_lt_03_total"] += 1
            seg_gt = gt[pa:pb]
            frac_gt_d = float(np.mean(seg_gt == DEAD))
            left_d = (pa > 0 and hl[pa - 1] == ph and pr[pa - 1] == DEAD) or (
                pa > 0 and hl[pa - 1] == ph and gt[pa - 1] == DEAD
            )
            right_d = (pb < T and hl[pb] == ph and pr[pb] == DEAD) or (
                pb < T and hl[pb] == ph and gt[pb] == DEAD
            )
            if frac_gt_d >= 0.35 or left_d or right_d:
                fp_summary["B5_dead_wedges"] += 1
            elif float(np.mean(seg_gt == (1 - pk))) >= 0.50:
                fp_summary["B4_false_blips_live"] += 1
            else:
                fp_summary["fp_boundary_or_split"] += 1

    return {
        "by_duration_bucket": by_bucket,
        "false_positives": fp_summary,
        "swallowed_short_records": swallowed_short_records,
    }


def audit_swallowed_short_probabilities_j1a(
    swallowed_records: List[Dict[str, Any]],
    streams_v3: Dict[str, Any],
    pset_v2_dir: Any,
    l1_trees_dir: Any,
) -> Dict[str, Any]:
    """Inspect v8.5 9-family probabilities, Modal GPU SymPlayerSetNetV2, and Modal CPU L1-Trees
    on every swallowed short possession (< 2.5s) in J1A.
    """
    mids = sorted({r["match_id"] for r in swallowed_records})
    pset_cache = {m: np.load(pset_v2_dir / f"pset_v2_{m}.npz") for m in mids}
    l1_cache = {m: np.load(l1_trees_dir / f"l1_trees_{m}.npz") for m in mids}

    fam_names = [
        "0_SymGBM_v5",
        "1_SymCatBoost",
        "2_SymXGBoost",
        "3_SymGBM_v6",
        "4_SymTCN_244",
        "5_SymTrans_244",
        "6_SymTCN_PhysV2_481",
        "7_SymTrans_PhysV2_481",
        "8_SymPlayerSetGRU_v1",
    ]

    enriched_records: List[Dict[str, Any]] = []
    for rec in swallowed_records:
        m = rec["match_id"]
        ga, gb = int(rec["ga"]), int(rec["gb"])
        y_lbl = int(rec["y_lbl"])
        x_lbl = int(rec["x_lbl"])

        p_v85, c_v85, p4d_v85, fams_v85 = streams_v3[m]
        T = len(p_v85)
        w_lo = max(0, ga - 2)
        w_hi = min(T, gb + 2)
        b_lo = max(0, ga - 4)
        b_hi = min(T, gb + 4)

        # v8.5 synthesized probabilities
        v85_max_pY = float(np.max(p_v85[ga:gb, y_lbl]))
        v85_mean_pY = float(np.mean(p_v85[ga:gb, y_lbl]))
        v85_max_pY_pad2 = float(np.max(p_v85[w_lo:w_hi, y_lbl]))
        v85_max_pchange = float(np.max(c_v85[b_lo:b_hi]))
        # Corrected directional transition column into Y: col 1 if Y==HOME else 0
        col_to_Y = 1 if y_lbl == HOME else 0
        col_to_X = 1 if x_lbl == HOME else 0
        ent_lo, ent_hi = max(0, ga - 4), min(T, ga + 5)
        ext_lo, ext_hi = max(0, gb - 4), min(T, gb + 5)
        v85_p4d_to_Y = float(np.max(p4d_v85[ent_lo:ent_hi, col_to_Y]))
        v85_p4d_to_X = float(np.max(p4d_v85[ext_lo:ext_hi, col_to_X]))
        v85_p4d_geom = float(np.sqrt(max(0.0, v85_p4d_to_Y * v85_p4d_to_X)))

        # Individual 9 families in v8.5
        fam_max_pY = {
            fam_names[idx]: float(np.max(fams_v85[idx][ga:gb, y_lbl]))
            for idx in range(len(fams_v85))
        }
        fam_mean_pY = {
            fam_names[idx]: float(np.mean(fams_v85[idx][ga:gb, y_lbl]))
            for idx in range(len(fams_v85))
        }
        v85_tree4_P = (fams_v85[0] + fams_v85[1] + fams_v85[2] + fams_v85[3]) / 4.0
        v85_tree4_max_pY = float(np.max(v85_tree4_P[ga:gb, y_lbl]))
        v85_tree4_mean_pY = float(np.mean(v85_tree4_P[ga:gb, y_lbl]))

        # Modal GPU SymPlayerSetNetV2 (trained with L1 inverse-sqrt spell weights + jersey swap aug)
        ps = pset_cache[m]
        ps_proba = np.asarray(ps["v3_proba"], dtype=np.float64)
        ps_change = np.asarray(ps["v3_change"], dtype=np.float64)
        ps_p4d = np.asarray(ps["v3_p4d"], dtype=np.float64)
        ps_air = np.asarray(ps["v3_p_aerial"], dtype=np.float64)

        pset_v2_max_pY = float(np.max(ps_proba[ga:gb, y_lbl]))
        pset_v2_mean_pY = float(np.mean(ps_proba[ga:gb, y_lbl]))
        pset_v2_max_pchange = float(np.max(ps_change[b_lo:b_hi]))
        pset_v2_p4d_to_Y = float(np.max(ps_p4d[ent_lo:ent_hi, col_to_Y]))
        pset_v2_p4d_to_X = float(np.max(ps_p4d[ext_lo:ext_hi, col_to_X]))
        pset_v2_p4d_geom = float(np.sqrt(max(0.0, pset_v2_p4d_to_Y * pset_v2_p4d_to_X)))
        pset_v2_mean_air = float(np.mean(ps_air[ga:gb]))

        # Modal CPU L1-Trees (0.45*GBM + 0.30*CatBoost + 0.25*XGBoost trained with L1 spell weights)
        l1 = l1_cache[m]
        l1_ens = (
            0.45 * np.asarray(l1["gbm_v3"], dtype=np.float64)
            + 0.30 * np.asarray(l1["cb_v3"], dtype=np.float64)
            + 0.25 * np.asarray(l1["xgb_v3"], dtype=np.float64)
        )
        l1_duel = np.asarray(l1["duel_v3"], dtype=np.float64)
        l1_air = np.asarray(l1["air_v3"], dtype=np.float64)

        l1_tree_max_pY = float(np.max(l1_ens[ga:gb, y_lbl]))
        l1_tree_mean_pY = float(np.mean(l1_ens[ga:gb, y_lbl]))
        l1_gbm_max_pY = float(np.max(l1["gbm_v3"][ga:gb, y_lbl]))
        l1_cb_max_pY = float(np.max(l1["cb_v3"][ga:gb, y_lbl]))
        l1_xgb_max_pY = float(np.max(l1["xgb_v3"][ga:gb, y_lbl]))
        l1_duel_max = float(np.max(l1_duel[w_lo:w_hi]))
        l1_duel_mean = float(np.mean(l1_duel[ga:gb]))
        l1_air_mean = float(np.mean(l1_air[ga:gb]))

        r_out = dict(rec)
        r_out.update(
            {
                "v85_max_pY": round(v85_max_pY, 4),
                "v85_mean_pY": round(v85_mean_pY, 4),
                "v85_max_pY_pad2": round(v85_max_pY_pad2, 4),
                "v85_max_pchange": round(v85_max_pchange, 4),
                "v85_p4d_to_Y": round(v85_p4d_to_Y, 4),
                "v85_p4d_geom": round(v85_p4d_geom, 4),
                "v85_tree4_max_pY": round(v85_tree4_max_pY, 4),
                "v85_tree4_mean_pY": round(v85_tree4_mean_pY, 4),
                "pset_v2_max_pY": round(pset_v2_max_pY, 4),
                "pset_v2_mean_pY": round(pset_v2_mean_pY, 4),
                "pset_v2_max_pchange": round(pset_v2_max_pchange, 4),
                "pset_v2_p4d_to_Y": round(pset_v2_p4d_to_Y, 4),
                "pset_v2_p4d_geom": round(pset_v2_p4d_geom, 4),
                "pset_v2_mean_air": round(pset_v2_mean_air, 4),
                "l1_tree_max_pY": round(l1_tree_max_pY, 4),
                "l1_tree_mean_pY": round(l1_tree_mean_pY, 4),
                "l1_gbm_max_pY": round(l1_gbm_max_pY, 4),
                "l1_cb_max_pY": round(l1_cb_max_pY, 4),
                "l1_xgb_max_pY": round(l1_xgb_max_pY, 4),
                "l1_duel_max": round(l1_duel_max, 4),
                "l1_duel_mean": round(l1_duel_mean, 4),
                "l1_air_mean": round(l1_air_mean, 4),
                "fam_max_pY": {k: round(v, 4) for k, v in fam_max_pY.items()},
                "fam_mean_pY": {k: round(v, 4) for k, v in fam_mean_pY.items()},
            }
        )
        enriched_records.append(r_out)

    def summarize_subset(sub: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {"subset": label, "n": 0}
        v85_mx = np.array([r["v85_max_pY"] for r in sub])
        v85_mn = np.array([r["v85_mean_pY"] for r in sub])
        v85_t4_mx = np.array([r["v85_tree4_max_pY"] for r in sub])
        v85_t4_mn = np.array([r["v85_tree4_mean_pY"] for r in sub])
        v85_pc = np.array([r["v85_max_pchange"] for r in sub])
        v85_p4d = np.array([r["v85_p4d_to_Y"] for r in sub])

        ps_mx = np.array([r["pset_v2_max_pY"] for r in sub])
        ps_mn = np.array([r["pset_v2_mean_pY"] for r in sub])
        ps_pc = np.array([r["pset_v2_max_pchange"] for r in sub])
        ps_p4d = np.array([r["pset_v2_p4d_to_Y"] for r in sub])

        l1_mx = np.array([r["l1_tree_max_pY"] for r in sub])
        l1_mn = np.array([r["l1_tree_mean_pY"] for r in sub])
        l1_duel = np.array([r["l1_duel_max"] for r in sub])

        any_modal_mx = np.maximum(ps_mx, l1_mx)
        any_all_mx = np.maximum(v85_mx, any_modal_mx)

        zero_ev_v85 = v85_mx < 0.25
        n_zero_v85 = int(np.sum(zero_ev_v85))

        fam_ge_025 = {}
        fam_mean_val = {}
        for fn in fam_names:
            arr = np.array([r["fam_max_pY"][fn] for r in sub])
            fam_ge_025[fn] = int(np.sum(arr >= 0.25))
            fam_mean_val[fn] = round(float(np.mean(arr)), 4)

        return {
            "subset": label,
            "n": n,
            # v8.5 max P(Y) >= 0.25 vs < 0.25
            "v85_max_pY_ge_025": int(np.sum(v85_mx >= 0.25)),
            "v85_max_pY_lt_025": n_zero_v85,
            "v85_max_pY_ge_025_pct": round(100.0 * float(np.mean(v85_mx >= 0.25)), 2),
            "v85_max_pY_lt_025_pct": round(100.0 * float(np.mean(v85_mx < 0.25)), 2),
            "v85_max_pY_ge_035": int(np.sum(v85_mx >= 0.35)),
            "v85_max_pY_ge_050": int(np.sum(v85_mx >= 0.50)),
            # SymPlayerSetNetV2 (Modal GPU) and L1-Trees (Modal CPU) >= 0.25
            "pset_v2_max_pY_ge_025": int(np.sum(ps_mx >= 0.25)),
            "pset_v2_max_pY_ge_035": int(np.sum(ps_mx >= 0.35)),
            "l1_trees_max_pY_ge_025": int(np.sum(l1_mx >= 0.25)),
            "l1_trees_max_pY_ge_035": int(np.sum(l1_mx >= 0.35)),
            "l1_duel_max_ge_025": int(np.sum(l1_duel >= 0.25)),
            "l1_duel_max_ge_035": int(np.sum(l1_duel >= 0.35)),
            "union_v85_or_modal_max_pY_ge_025": int(np.sum(any_all_mx >= 0.25)),
            "union_v85_or_modal_or_duel_ge_025": int(
                np.sum(np.maximum(any_all_mx, l1_duel) >= 0.25)
            ),
            # Head-to-head comparisons vs v8.5
            "pset_v2_higher_max_pY_than_v85": int(np.sum(ps_mx > v85_mx)),
            "pset_v2_higher_mean_pY_than_v85": int(np.sum(ps_mn > v85_mn)),
            "pset_v2_higher_pchange_than_v85": int(np.sum(ps_pc > v85_pc)),
            "pset_v2_higher_p4d_to_Y_than_v85": int(np.sum(ps_p4d > v85_p4d)),
            "l1_trees_higher_max_pY_than_v85": int(np.sum(l1_mx > v85_mx)),
            "l1_trees_higher_mean_pY_than_v85": int(np.sum(l1_mn > v85_mn)),
            "l1_trees_higher_max_pY_than_v85_tree4": int(np.sum(l1_mx > v85_t4_mx)),
            "l1_trees_higher_mean_pY_than_v85_tree4": int(np.sum(l1_mn > v85_t4_mn)),
            "either_modal_higher_max_pY_than_v85": int(np.sum(any_modal_mx > v85_mx)),
            "either_modal_or_duel_higher_than_v85_max_pY": int(
                np.sum(np.maximum(any_modal_mx, l1_duel) > v85_mx)
            ),
            # Rescue of v8.5 "zero-evidence" (< 0.25) spells
            "rescued_from_v85_lt_025_by_pset_v2_pY_ge_025": int(
                np.sum(zero_ev_v85 & (ps_mx >= 0.25))
            ),
            "rescued_from_v85_lt_025_by_l1_trees_pY_ge_025": int(
                np.sum(zero_ev_v85 & (l1_mx >= 0.25))
            ),
            "rescued_from_v85_lt_025_by_either_modal_pY_ge_025": int(
                np.sum(zero_ev_v85 & (any_modal_mx >= 0.25))
            ),
            "rescued_from_v85_lt_025_by_modal_pY_or_duel_ge_025": int(
                np.sum(zero_ev_v85 & (np.maximum(any_modal_mx, l1_duel) >= 0.25))
            ),
            # Mean levels across subset
            "mean_of_v85_max_pY": round(float(np.mean(v85_mx)), 4),
            "mean_of_pset_v2_max_pY": round(float(np.mean(ps_mx)), 4),
            "mean_of_l1_trees_max_pY": round(float(np.mean(l1_mx)), 4),
            "mean_of_v85_tree4_max_pY": round(float(np.mean(v85_t4_mx)), 4),
            "mean_of_l1_duel_max": round(float(np.mean(l1_duel)), 4),
            "mean_of_v85_p4d_to_Y": round(float(np.mean(v85_p4d)), 4),
            "mean_of_pset_v2_p4d_to_Y": round(float(np.mean(ps_p4d)), 4),
            "per_family_ge_025_counts": fam_ge_025,
            "per_family_mean_max_pY": fam_mean_val,
        }

    sub_all_25 = enriched_records
    sub_dom_25 = [r for r in enriched_records if r["is_double_domino_steal"]]
    sub_all_20 = [r for r in enriched_records if r["L"] <= 50]
    sub_dom_20 = [r for r in enriched_records if r["is_double_domino_steal"] and r["L"] <= 50]
    sub_all_136 = [r for r in enriched_records if r["L"] < 34]
    sub_dom_136 = [r for r in enriched_records if r["is_double_domino_steal"] and r["L"] < 34]

    return {
        "all_swallowed_lt_2p5s": summarize_subset(sub_all_25, "all_swallowed_lt_2.5s (<=62f)"),
        "double_domino_lt_2p5s": summarize_subset(sub_dom_25, "double_domino_steals_lt_2.5s (<=62f)"),
        "all_swallowed_lt_2p0s": summarize_subset(sub_all_20, "all_swallowed_lt_2.0s (<=50f)"),
        "double_domino_lt_2p0s": summarize_subset(sub_dom_20, "double_domino_steals_lt_2.0s (<=50f)"),
        "all_swallowed_lt_1p36s": summarize_subset(sub_all_136, "all_swallowed_lt_1.36s (<34f)"),
        "double_domino_lt_1p36s": summarize_subset(sub_dom_136, "double_domino_steals_lt_1.36s (<34f)"),
    }


# ======================================================================================
# PART C: EXACT SPELL-F1 ORACLE CEILINGS OF EACH BOTTLENECK (B1..B5)
# ======================================================================================


def _oracle_b1_short_possessions(
    gt: np.ndarray,
    pr: np.ndarray,
    hl: np.ndarray,
    max_len: int = 75,
    mode: str = "all",  # "double_domino", "other_short", "all"
) -> np.ndarray:
    """Surgically restore missed short GT possessions (`L <= max_len`, `IoU < 0.3`)."""
    out = pr.copy()
    gt_spells_h = _spells_with_half(gt, hl)
    pr_spells_h = _spells_with_half(pr, hl)
    gt_spells = [(a, b, k) for a, b, k, _ in gt_spells_h]
    pr_spells = [(a, b, k) for a, b, k, _ in pr_spells_h]

    overlaps = pm.spell_overlaps(gt_spells, pr_spells)
    gt_m03, _ = _greedy_match_map(overlaps, len(gt_spells), 0.3)

    pred_to_same_gt: Dict[int, List[int]] = {j: [] for j in range(len(pr_spells))}
    for gi, (ga, gb, gk, gh) in enumerate(gt_spells_h):
        for pj, (pa, pb, pk, ph) in enumerate(pr_spells_h):
            if gh == ph and gk == pk and max(ga, pa) < min(gb, pb):
                pred_to_same_gt[pj].append(gi)

    for gi, (ga, gb, gk, gh) in enumerate(gt_spells_h):
        L = gb - ga
        if L > max_len or gi in gt_m03:
            continue
        opp = 1 - gk
        frac_opp = float(np.mean(pr[ga:gb] == opp))
        is_domino = False
        if frac_opp >= 0.50:
            for pj, (pa, pb, pk, ph) in enumerate(pr_spells_h):
                if ph == gh and pk == opp and max(ga, pa) < min(gb, pb):
                    ov_len = min(gb, pb) - max(ga, pa)
                    if ov_len >= 0.50 * L and len(pred_to_same_gt[pj]) >= 2:
                        opp_gts = pred_to_same_gt[pj]
                        if any(gt_spells_h[ogi][0] < ga for ogi in opp_gts) and any(
                            gt_spells_h[ogi][1] > gb for ogi in opp_gts
                        ):
                            is_domino = True
                            break
        if mode == "double_domino" and not is_domino:
            continue
        if mode == "other_short" and is_domino:
            continue
        out[ga:gb] = gk
    return out


def _oracle_b2_team_boundaries(
    gt: np.ndarray,
    pr: np.ndarray,
    hl: np.ndarray,
    max_shift: int = 35,
) -> np.ndarray:
    """Snap H <-> A live transition boundaries (within +/- max_shift frames) to GT without
    wiping out an entire predicted spell."""
    out = pr.copy()
    for ha, hb in _half_bounds(hl):
        # Find live H<->A transitions in GT and in out[ha:hb]
        gt_chg = [
            t
            for t in range(ha + 1, hb)
            if gt[t] in (HOME, AWAY) and gt[t - 1] in (HOME, AWAY) and gt[t] != gt[t - 1]
        ]
        pr_chg = [
            t
            for t in range(ha + 1, hb)
            if out[t] in (HOME, AWAY) and out[t - 1] in (HOME, AWAY) and out[t] != out[t - 1]
        ]
        used_pr = set()
        for tg in gt_chg:
            pre_lbl, post_lbl = int(gt[tg - 1]), int(gt[tg])
            # Match nearest predicted transition with the same (pre_lbl -> post_lbl)
            cands = [
                tp
                for tp in pr_chg
                if tp not in used_pr
                and abs(tp - tg) <= max_shift
                and int(out[tp - 1]) == pre_lbl
                and int(out[tp]) == post_lbl
            ]
            if not cands:
                continue
            tp = min(cands, key=lambda x: abs(x - tg))
            used_pr.add(tp)
            if tp < tg:
                # Predicted transition was early: fill [tp, tg) with pre_lbl if that doesn't erase post spell
                if tg < hb and int(out[tg]) == post_lbl:
                    out[tp:tg] = pre_lbl
            elif tg < tp:
                # Predicted transition was late: fill [tg, tp) with post_lbl if that doesn't erase pre spell
                if tg > ha and int(out[tg - 1]) == pre_lbl:
                    out[tg:tp] = post_lbl
    return out


def _oracle_b3_dead_boundaries(
    gt: np.ndarray,
    pr: np.ndarray,
    hl: np.ndarray,
    max_shift: int = 50,
) -> np.ndarray:
    """Snap Live <-> D stoppage/restart boundaries (within +/- max_shift frames) to GT."""
    out = pr.copy()
    for ha, hb in _half_bounds(hl):
        # Match Live->D and D->Live transitions
        for to_dead in (True, False):
            gt_chg = [
                t
                for t in range(ha + 1, hb)
                if (
                    (gt[t - 1] in (HOME, AWAY) and gt[t] == DEAD)
                    if to_dead
                    else (gt[t - 1] == DEAD and gt[t] in (HOME, AWAY))
                )
            ]
            pr_chg = [
                t
                for t in range(ha + 1, hb)
                if (
                    (out[t - 1] in (HOME, AWAY) and out[t] == DEAD)
                    if to_dead
                    else (out[t - 1] == DEAD and out[t] in (HOME, AWAY))
                )
            ]
            used_pr = set()
            for tg in gt_chg:
                cands = [tp for tp in pr_chg if tp not in used_pr and abs(tp - tg) <= max_shift]
                if not cands:
                    continue
                tp = min(cands, key=lambda x: abs(x - tg))
                used_pr.add(tp)
                if to_dead:
                    live_lbl = int(out[tp - 1]) if out[tp - 1] in (HOME, AWAY) else int(gt[tg - 1])
                    if tp < tg:
                        out[tp:tg] = live_lbl
                    elif tg < tp:
                        out[tg:tp] = DEAD
                else:
                    live_lbl = int(out[tp]) if out[tp] in (HOME, AWAY) else int(gt[tg])
                    if tp < tg:
                        out[tp:tg] = DEAD
                    elif tg < tp:
                        out[tg:tp] = live_lbl
    return out


def _oracle_b4_false_blips(gt: np.ndarray, pr: np.ndarray, hl: np.ndarray) -> np.ndarray:
    """Absorb false-positive interior live blips (`IoU < 0.3`, surrounded by live opponent or inside GT opponent)."""
    out = pr.copy()
    T = len(pr)
    gt_spells_h = _spells_with_half(gt, hl)
    pr_spells_h = _spells_with_half(pr, hl)
    gt_spells = [(a, b, k) for a, b, k, _ in gt_spells_h]
    pr_spells = [(a, b, k) for a, b, k, _ in pr_spells_h]
    overlaps = pm.spell_overlaps(gt_spells, pr_spells)
    _, pr_m03 = _greedy_match_map(overlaps, len(gt_spells), 0.3)

    for pj, (pa, pb, pk, ph) in enumerate(pr_spells_h):
        if pj in pr_m03:
            continue
        opp = 1 - pk
        seg_gt = gt[pa:pb]
        frac_gt_d = float(np.mean(seg_gt == DEAD))
        left_d = (pa > 0 and hl[pa - 1] == ph and pr[pa - 1] == DEAD) or (
            pa > 0 and hl[pa - 1] == ph and gt[pa - 1] == DEAD
        )
        right_d = (pb < T and hl[pb] == ph and pr[pb] == DEAD) or (
            pb < T and hl[pb] == ph and gt[pb] == DEAD
        )
        if not (frac_gt_d >= 0.35 or left_d or right_d) and float(np.mean(seg_gt == opp)) >= 0.50:
            out[pa:pb] = opp
    return out


def _oracle_b5_dead_wedges(gt: np.ndarray, pr: np.ndarray, hl: np.ndarray) -> np.ndarray:
    """Absorb false-positive D-adjacent wedges (`IoU < 0.3`, adjacent to or inside DEAD)."""
    out = pr.copy()
    T = len(pr)
    gt_spells_h = _spells_with_half(gt, hl)
    pr_spells_h = _spells_with_half(pr, hl)
    gt_spells = [(a, b, k) for a, b, k, _ in gt_spells_h]
    pr_spells = [(a, b, k) for a, b, k, _ in pr_spells_h]
    overlaps = pm.spell_overlaps(gt_spells, pr_spells)
    _, pr_m03 = _greedy_match_map(overlaps, len(gt_spells), 0.3)

    for pj, (pa, pb, pk, ph) in enumerate(pr_spells_h):
        if pj in pr_m03:
            continue
        seg_gt = gt[pa:pb]
        frac_gt_d = float(np.mean(seg_gt == DEAD))
        left_d = (pa > 0 and hl[pa - 1] == ph and pr[pa - 1] == DEAD) or (
            pa > 0 and hl[pa - 1] == ph and gt[pa - 1] == DEAD
        )
        right_d = (pb < T and hl[pb] == ph and pr[pb] == DEAD) or (
            pb < T and hl[pb] == ph and gt[pb] == DEAD
        )
        if frac_gt_d >= 0.35 or left_d or right_d:
            # Replace with GT label where GT is scored (0,1,2), else DEAD
            rep = np.where(np.isin(seg_gt, (HOME, AWAY, DEAD)), seg_gt, DEAD)
            out[pa:pb] = rep
    return out


def compute_exact_oracle_ceilings(
    gt_dict: Dict[str, np.ndarray],
    pred_dict: Dict[str, np.ndarray],
    half_dict: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute exact marginal and cumulative spell_f1_mean oracle ceilings for B1..B5."""
    keys = sorted(gt_dict.keys())

    def eval_mean(p_map: Dict[str, np.ndarray]) -> float:
        return float(
            np.mean([metrics.score(gt_dict[k], p_map[k], half_dict[k])["spell_f1_mean"] for k in keys])
        )

    base = eval_mean(pred_dict)

    # Marginal (single-bottleneck) ceilings
    m_b1_dom_62 = eval_mean(
        {k: _oracle_b1_short_possessions(gt_dict[k], pred_dict[k], half_dict[k], 62, "double_domino") for k in keys}
    )
    m_b1_dom_75 = eval_mean(
        {k: _oracle_b1_short_possessions(gt_dict[k], pred_dict[k], half_dict[k], 75, "double_domino") for k in keys}
    )
    m_b1_oth_62 = eval_mean(
        {k: _oracle_b1_short_possessions(gt_dict[k], pred_dict[k], half_dict[k], 62, "other_short") for k in keys}
    )
    m_b1_oth_75 = eval_mean(
        {k: _oracle_b1_short_possessions(gt_dict[k], pred_dict[k], half_dict[k], 75, "other_short") for k in keys}
    )
    m_b1_all_62 = eval_mean(
        {k: _oracle_b1_short_possessions(gt_dict[k], pred_dict[k], half_dict[k], 62, "all") for k in keys}
    )
    m_b1_all_75 = eval_mean(
        {k: _oracle_b1_short_possessions(gt_dict[k], pred_dict[k], half_dict[k], 75, "all") for k in keys}
    )
    m_b2_team = eval_mean(
        {k: _oracle_b2_team_boundaries(gt_dict[k], pred_dict[k], half_dict[k], 35) for k in keys}
    )
    m_b3_dead = eval_mean(
        {k: _oracle_b3_dead_boundaries(gt_dict[k], pred_dict[k], half_dict[k], 50) for k in keys}
    )
    m_b4_blips = eval_mean(
        {k: _oracle_b4_false_blips(gt_dict[k], pred_dict[k], half_dict[k]) for k in keys}
    )
    m_b5_wedges = eval_mean(
        {k: _oracle_b5_dead_wedges(gt_dict[k], pred_dict[k], half_dict[k]) for k in keys}
    )

    # Progressive cumulative ladder
    c1 = {k: _oracle_b1_short_possessions(gt_dict[k], pred_dict[k], half_dict[k], 75, "double_domino") for k in keys}
    c2 = {k: _oracle_b1_short_possessions(gt_dict[k], c1[k], half_dict[k], 75, "other_short") for k in keys}
    c3 = {k: _oracle_b2_team_boundaries(gt_dict[k], c2[k], half_dict[k], 35) for k in keys}
    c4 = {k: _oracle_b3_dead_boundaries(gt_dict[k], c3[k], half_dict[k], 50) for k in keys}
    c5 = {k: _oracle_b4_false_blips(gt_dict[k], c4[k], half_dict[k]) for k in keys}
    c6 = {k: _oracle_b5_dead_wedges(gt_dict[k], c5[k], half_dict[k]) for k in keys}

    return {
        "base_spell_f1": round(base, 4),
        "marginal_ceilings": {
            "B1_double_domino_le_2p5s": {"spell_f1": round(m_b1_dom_62, 4), "delta": round(m_b1_dom_62 - base, 4)},
            "B1_double_domino_le_3p0s": {"spell_f1": round(m_b1_dom_75, 4), "delta": round(m_b1_dom_75 - base, 4)},
            "B1_other_short_le_2p5s": {"spell_f1": round(m_b1_oth_62, 4), "delta": round(m_b1_oth_62 - base, 4)},
            "B1_other_short_le_3p0s": {"spell_f1": round(m_b1_oth_75, 4), "delta": round(m_b1_oth_75 - base, 4)},
            "B1_all_short_le_2p5s": {"spell_f1": round(m_b1_all_62, 4), "delta": round(m_b1_all_62 - base, 4)},
            "B1_all_short_le_3p0s": {"spell_f1": round(m_b1_all_75, 4), "delta": round(m_b1_all_75 - base, 4)},
            "B2_team_boundaries_pm35f": {"spell_f1": round(m_b2_team, 4), "delta": round(m_b2_team - base, 4)},
            "B3_dead_boundaries_pm50f": {"spell_f1": round(m_b3_dead, 4), "delta": round(m_b3_dead - base, 4)},
            "B4_false_blips_live": {"spell_f1": round(m_b4_blips, 4), "delta": round(m_b4_blips - base, 4)},
            "B5_dead_wedges": {"spell_f1": round(m_b5_wedges, 4), "delta": round(m_b5_wedges - base, 4)},
        },
        "cumulative_ladder": {
            "0_base": round(base, 4),
            "1_plus_B1_double_domino": round(eval_mean(c1), 4),
            "2_plus_B1_other_short": round(eval_mean(c2), 4),
            "3_plus_B2_team_boundaries": round(eval_mean(c3), 4),
            "4_plus_B3_dead_boundaries": round(eval_mean(c4), 4),
            "5_plus_B4_false_blips": round(eval_mean(c5), 4),
            "6_plus_B5_dead_wedges": round(eval_mean(c6), 4),
        },
    }


# ======================================================================================
# PART D: J_ATD_TWIN REALISTIC TARGET-DOMAIN STRESS-TEST SUITE ON J1A (SPORTEC)
# ======================================================================================


def simulate_j1_jersey_swap_table(
    X_df: pd.DataFrame,
    half: np.ndarray,
    rate: float = 0.21,
    dur_s: float = 3.5,
    min_start_dist_m: float = 8.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Simulate BAR-ATH's `12v10` jersey team swaps on the feature table `X_df` without target leakage.

    In BAR-ATH (PLAN_NOCHE Sec 13 D3), `0.21` players/frame (count+color corroborated, `0.31` total color)
    are assigned to the wrong team for median `3.5s` (`88f`) stretches starting `>= 8m` from the ball
    (producing `12 vs 10` team counts in `10.8%` of frames, and occasionally drifting near the ball).
    """
    rng = np.random.default_rng(seed)
    X_out = X_df.copy()
    T = len(X_df)
    L_seg = max(10, int(round(dur_s * FPS)))
    n_segs = int(round(rate * T / L_seg))

    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=15.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=15.0)
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=0.0) > 0.5

    # Active swap count delta: +1 means 1 Away player moved to Home (12v10), -1 means 1 Home moved to Away (10v12)
    delta_h = np.zeros(T, dtype=np.float64)
    near_ball_corrupted = np.zeros(T, dtype=bool)
    placed = 0

    for _ in range(n_segs * 4):
        if placed >= n_segs:
            break
        t0 = int(rng.integers(0, max(1, T - L_seg)))
        t1 = min(T, t0 + L_seg)
        if half[t0] <= 0 or half[t0] != half[t1 - 1]:
            continue
        direction = 1.0 if rng.random() < 0.5 else -1.0  # +1: A->H, -1: H->A
        # If min_start_dist_m >= 8.0, starts away from ball (>8m), so only 0.7% of frames drift nearest to ball
        # If min_start_dist_m == 0.0, starts anywhere on pitch (~1/11 = 9.1% chance it is the nearest player!)
        p_drift_nearest = 0.035 if min_start_dist_m >= 8.0 else 0.14
        delta_h[t0:t1] += direction
        if rng.random() < p_drift_nearest:
            # During the final third of the stretch, the swapped player comes within [0.8m, 2.2m] of the ball
            sub_a = t0 + (t1 - t0) // 2
            near_ball_corrupted[sub_a:t1] = True
        placed += 1

    # Perturb team counts (12 vs 10) and team geometry centroids/widths/lines
    if "n_visible_H" in X_out.columns and "n_visible_A" in X_out.columns:
        nh = X_out["n_visible_H"].to_numpy(np.float64)
        na = X_out["n_visible_A"].to_numpy(np.float64)
        X_out["n_visible_H"] = np.clip(nh + delta_h, 0.0, 15.0)
        X_out["n_visible_A"] = np.clip(na - delta_h, 0.0, 15.0)

    for col_h, col_a, weight in [
        ("centroid_x_H", "centroid_x_A", 1.0 / 11.0),
        ("centroid_y_H", "centroid_y_A", 1.0 / 11.0),
        ("spread_x_H", "spread_x_A", 0.12),
        ("spread_y_H", "spread_y_A", 0.12),
        ("width_H", "width_A", 0.14),
        ("def_line_H", "def_line_A", 0.15),
        ("fwd_line_H", "fwd_line_A", 0.15),
    ]:
        if col_h in X_out.columns and col_a in X_out.columns:
            vh = X_out[col_h].to_numpy(np.float64)
            va = X_out[col_a].to_numpy(np.float64)
            diff = va - vh
            X_out[col_h] = vh + np.maximum(0.0, delta_h) * weight * diff
            X_out[col_a] = va - np.maximum(0.0, -delta_h) * weight * diff

    # When a swapped player drifts near the ball, the wrong team appears closer to the ball
    idx_corrupt = np.flatnonzero(near_ball_corrupted & vis & (delta_h != 0))
    if len(idx_corrupt) > 0:
        for t in idx_corrupt:
            if delta_h[t] > 0:  # An Away player is labelled Home near the ball
                X_out.iat[t, X_out.columns.get_loc("dist_ball_H")] = min(d1h[t], d1a[t])
                X_out.iat[t, X_out.columns.get_loc("dist_ball_A")] = max(d1h[t], d1a[t], 2.8)
            else:  # A Home player is labelled Away near the ball
                X_out.iat[t, X_out.columns.get_loc("dist_ball_A")] = min(d1h[t], d1a[t])
                X_out.iat[t, X_out.columns.get_loc("dist_ball_H")] = max(d1h[t], d1a[t], 2.8)
        if "dist_ball_diff" in X_out.columns:
            X_out["dist_ball_diff"] = X_out["dist_ball_H"] - X_out["dist_ball_A"]

    frac_12v10 = float(np.mean(np.abs(delta_h) >= 1.0))
    mean_swapped = float(np.mean(np.abs(delta_h)))
    return X_out, {
        "n_segments_placed": placed,
        "mean_swapped_players_per_frame": round(mean_swapped, 4),
        "frac_frames_with_swap": round(frac_12v10, 4),
        "frac_frames_near_ball_corrupted": round(float(len(idx_corrupt) / max(1, T)), 4),
    }


def simulate_j1_unbiased_dead(
    gt: np.ndarray,
    pr_v85: np.ndarray,
    proba: np.ndarray,
    p4d: np.ndarray,
    X_df: pd.DataFrame,
    half: np.ndarray,
    sp_tags: Any,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate `J1_UnbiasedDead` on a match:
    In `Sportec degrade_v3` (`Codex 003 #2`), `91.5-97.0%` of visible `D` frames have a player
    glued at `0.6m +/- 0.45m` (`<= 1.24m`), vs `32.0%` in `BAR-ATH` (median `2.89m`).
    When this synthetic `d <= 1.24m` dead-ball attachment shortcut is neutralized during `D`
    transitions (shifting held-ball `min_d1` to `BAR-ATH`'s `2.89m` median on `65%` of `D` spells
    and attenuating the `p_D` shortcut logit near `Live <-> D` transitions):
      - How much does a proximity-shortcut detector degrade?
      - How much does `v8.5` degrade when `p_D` loses the synthetic `d <= 1.24m` boost at `D` edges?
      - How much does `v9` (`Line L4 StoppageStartRefiner` + `Line M5-v9 SmartTaggingArbitrator`) recover?
    """
    rng = np.random.default_rng(seed)
    X_unb = X_df.copy()
    proba_unb = proba.copy()
    pr_v85_unb = pr_v85.copy()

    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=15.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=15.0)
    min_d1 = np.minimum(d1h, d1a)
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=0.0) > 0.5
    bspd = np.nan_to_num(X_df["ball_speed"].to_numpy(np.float64), nan=0.0)

    # Measure raw Sportec degrade_v3 vs unbiased BAR-ATH stats on visible D
    vis_d_mask = (gt == DEAD) & vis
    orig_frac_le_124 = float(np.mean(min_d1[vis_d_mask] <= 1.24)) if np.any(vis_d_mask) else 0.0
    orig_median_d1 = float(np.median(min_d1[vis_d_mask])) if np.any(vis_d_mask) else 0.0

    # Neutralize the held-ball attachment on ~65% of visible D segments so frac(min_d1 <= 1.24m)
    # drops from ~94% to ~32% (matching BAR-ATH's 32.0% and median ~2.89m)
    s_r, e_r, v_r = _runs_of_with_half(gt, half)
    n_d_unbiased = 0
    for r_idx in range(len(v_r)):
        if v_r[r_idx] != DEAD:
            continue
        sa, sb = int(s_r[r_idx]), int(e_r[r_idx])
        L = sb - sa
        if L < 25:
            continue
        # For frames inside [sa, sb) where ball is visible and artificially glued <= 1.24m
        seg_vis_glued = np.flatnonzero(vis[sa:sb] & (min_d1[sa:sb] <= 1.24)) + sa
        if len(seg_vis_glued) == 0:
            continue
        # Keep only ~32% glued (matching BAR-ATH), un-glue the rest to ~N(3.1m, 0.9m)
        if rng.random() < 0.68:
            n_d_unbiased += 1
            shift_d = rng.normal(2.55, 0.65, size=len(seg_vis_glued)).clip(1.45, 5.0)
            X_unb.iloc[seg_vis_glued, X_unb.columns.get_loc("dist_ball_H")] = d1h[seg_vis_glued] + shift_d
            X_unb.iloc[seg_vis_glued, X_unb.columns.get_loc("dist_ball_A")] = d1a[seg_vis_glued] + shift_d
            if "n_within_2m_H" in X_unb.columns:
                X_unb.iloc[seg_vis_glued, X_unb.columns.get_loc("n_within_2m_H")] = 0.0
                X_unb.iloc[seg_vis_glued, X_unb.columns.get_loc("n_within_2m_A")] = 0.0
            if "ball_speed" in X_unb.columns:
                X_unb.iloc[seg_vis_glued, X_unb.columns.get_loc("ball_speed")] = np.maximum(
                    bspd[seg_vis_glued], rng.uniform(2.0, 3.8, size=len(seg_vis_glued))
                )
            # Without the synthetic d <= 1.24m held-ball shortcut at the start/end of visible D,
            # the tree/seq ensemble's p_D lags by ~18-28 frames at visible-ball D starts/ends
            edge_w = min(24, L // 3)
            edge_idx = seg_vis_glued[(seg_vis_glued < sa + edge_w) | (seg_vis_glued >= sb - edge_w)]
            if len(edge_idx) > 0:
                # Attenuate p_D on those un-glued visible edge frames
                p_d_old = proba_unb[edge_idx, DEAD].copy()
                p_d_new = p_d_old * 0.42
                rem = 1.0 - p_d_new
                live_sum = np.maximum(1e-6, proba_unb[edge_idx, HOME] + proba_unb[edge_idx, AWAY])
                proba_unb[edge_idx, HOME] = proba_unb[edge_idx, HOME] / live_sum * rem
                proba_unb[edge_idx, AWAY] = proba_unb[edge_idx, AWAY] / live_sum * rem
                proba_unb[edge_idx, DEAD] = p_d_new
                # Corresponding v8.5 decoded prediction leaks live team into the un-glued visible D start
                live_team = (
                    int(pr_v85[sa - 1])
                    if (sa > 0 and pr_v85[sa - 1] in (HOME, AWAY))
                    else int( np.argmax(proba_unb[edge_idx[0], :2]) )
                )
                start_leak = seg_vis_glued[seg_vis_glued < sa + min(22, L // 3)]
                if len(start_leak) > 0:
                    pr_v85_unb[start_leak] = live_team

    min_d1_unb = np.minimum(
        np.nan_to_num(X_unb["dist_ball_H"].to_numpy(np.float64), nan=15.0),
        np.nan_to_num(X_unb["dist_ball_A"].to_numpy(np.float64), nan=15.0),
    )
    unb_frac_le_124 = float(np.mean(min_d1_unb[vis_d_mask] <= 1.24)) if np.any(vis_d_mask) else 0.0
    unb_median_d1 = float(np.median(min_d1_unb[vis_d_mask])) if np.any(vis_d_mask) else 0.0

    # Apply v9 Line L4 (StoppageStartRefiner) + Line M5-v9 (SmartTaggingArbitrator) on the unbiased stream
    from src.v9_swarm.smart_tagging_arbitrator import SmartTaggingArbitrator

    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()
    pr_v9_unb = refiner.apply(pr_v85_unb, proba_unb, p4d, X_unb, half, enable_sliver_and_nocontact=True)
    pr_v9_unb = arbitrator.apply(pr_v9_unb, proba_unb, X_unb, half, sp_tags)

    return {
        "pr_v85_unbiased_dead": pr_v85_unb,
        "pr_v9_unbiased_dead": pr_v9_unb,
        "orig_vis_d_frac_le_124m": round(orig_frac_le_124, 4),
        "unbiased_vis_d_frac_le_124m": round(unb_frac_le_124, 4),
        "orig_vis_d_median_d1_m": round(orig_median_d1, 3),
        "unbiased_vis_d_median_d1_m": round(unb_median_d1, 3),
        "n_d_spells_unbiased": n_d_unbiased,
    }


def simulate_j1_rolling_foul(
    gt: np.ndarray,
    pr_v85: np.ndarray,
    proba: np.ndarray,
    p4d: np.ndarray,
    X_df: pd.DataFrame,
    half: np.ndarray,
    sp_tags: Any,
    lag_range_frames: Tuple[int, int] = (38, 75),  # 1.5s to 3.0s at 25 fps
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate `J1_RollingFoul` on a match:
    In `BAR-ATH` (`PLAN_NOCHE D6 & D9`), in-pitch foul stoppages (`85` mid-pitch `D` runs) have the ball
    still rolling after the referee whistle while players decelerate collectively (`speed_mean_all` drops),
    causing `v8.5`'s `D` start prediction to lag by `1.5s-3.0s` (up to `4.4s` on `15889` and `17380`).
    In `Sportec J1A`, `v8.5` has `0f` median lag because `degrade_v3` snaps the ball at `D` start.
    Here we inject a realistic `1.5s-3.0s` (`38f-75f`) rolling-ball lag at the start of in-pitch `D`
    stoppages, together with the realistic collective player deceleration signature after the whistle,
    and measure `v8.5` degradation vs `v9` (`Line L4` `CollectiveFoulDeceleration` + `StoppageStartRefiner`
    + `M5-v9`) recovery!
    """
    rng = np.random.default_rng(seed)
    X_foul = X_df.copy()
    proba_foul = proba.copy()
    pr_v85_foul = pr_v85.copy()

    bx = np.nan_to_num(X_df["ball_x"].to_numpy(np.float64), nan=52.5)
    by = np.nan_to_num(X_df["ball_y"].to_numpy(np.float64), nan=34.0)
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=0.0) > 0.5
    cross_frames = detect_pitch_crossings_from_features(bx, by, vis, half)

    s_r, e_r, v_r = _runs_of_with_half(gt, half)
    n_in_pitch_fouls = 0
    lags_injected: List[int] = []

    for r_idx in range(1, len(v_r)):
        if v_r[r_idx] != DEAD or v_r[r_idx - 1] not in (HOME, AWAY):
            continue
        sa, sb = int(s_r[r_idx]), int(e_r[r_idx])
        prev_a, prev_b = int(s_r[r_idx - 1]), int(e_r[r_idx - 1])
        L_dead = sb - sa
        L_prev = prev_b - prev_a
        if L_dead < 65 or L_prev < 25 or half[sa] != half[prev_a]:
            continue
        # Exclude out-of-bounds stoppages (pitch crossing within +/- 30 frames of sa)
        if len(cross_frames) > 0 and np.min(np.abs(cross_frames - sa)) <= 30:
            continue
        # Exclude near-touchline / near-goalline stoppages
        if bx[sa] < 4.0 or bx[sa] > 101.0 or by[sa] < 4.0 or by[sa] > 64.0:
            continue

        lag = int(rng.integers(lag_range_frames[0], lag_range_frames[1] + 1))
        lag = min(lag, L_dead - 30)
        if lag < 20:
            continue

        n_in_pitch_fouls += 1
        lags_injected.append(lag)
        live_lbl = int(v_r[r_idx - 1])

        # In BAR-ATH rolling fouls:
        # 1. During [sa, sa + lag), the ball is still visible & rolling so v8.5 continues predicting `live_lbl`
        pr_v85_foul[sa : sa + lag] = live_lbl
        X_foul.iloc[sa : sa + lag, X_foul.columns.get_loc("ball_visible")] = 1.0
        # 2. Players decelerate collectively after the true whistle at `sa` (`speed_mean_H` and `speed_mean_A` drop)
        pre_w = max(prev_a, sa - 35)
        pre_spd_h = float(np.mean(X_df["speed_mean_H"].to_numpy(np.float64)[pre_w:sa]))
        pre_spd_a = float(np.mean(X_df["speed_mean_A"].to_numpy(np.float64)[pre_w:sa]))
        base_spd = max(2.2, 0.5 * (pre_spd_h + pre_spd_a))
        X_foul.iloc[pre_w:sa, X_foul.columns.get_loc("speed_mean_H")] = base_spd
        X_foul.iloc[pre_w:sa, X_foul.columns.get_loc("speed_mean_A")] = base_spd
        X_foul.iloc[sa : sa + lag, X_foul.columns.get_loc("speed_mean_H")] = max(0.8, base_spd - 0.85)
        X_foul.iloc[sa : sa + lag, X_foul.columns.get_loc("speed_mean_A")] = max(0.8, base_spd - 0.85)
        # Ensure ball stays in-bounds away from touchlines during the rolling lag
        X_foul.iloc[sa : sa + lag, X_foul.columns.get_loc("ball_x")] = np.clip(bx[sa : sa + lag], 12.0, 93.0)
        X_foul.iloc[sa : sa + lag, X_foul.columns.get_loc("ball_y")] = np.clip(by[sa : sa + lag], 10.0, 58.0)
        # Moderate dead-ball posterior suspicion (`p_D ~ 0.22-0.30`) during the rolling lag
        proba_foul[sa : sa + lag, DEAD] = np.maximum(proba_foul[sa : sa + lag, DEAD], 0.24)
        rem = 1.0 - proba_foul[sa : sa + lag, DEAD]
        proba_foul[sa : sa + lag, live_lbl] = rem * 0.82
        proba_foul[sa : sa + lag, 1 - live_lbl] = rem * 0.18

    from src.v9_swarm.smart_tagging_arbitrator import SmartTaggingArbitrator

    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()
    pr_v9_foul = refiner.apply(pr_v85_foul, proba_foul, p4d, X_foul, half, enable_sliver_and_nocontact=True)
    pr_v9_foul = arbitrator.apply(pr_v9_foul, proba_foul, X_foul, half, sp_tags)

    return {
        "pr_v85_rolling_foul": pr_v85_foul,
        "pr_v9_rolling_foul": pr_v9_foul,
        "n_in_pitch_fouls_perturbed": n_in_pitch_fouls,
        "mean_injected_lag_frames": round(float(np.mean(lags_injected)), 2) if lags_injected else 0.0,
        "mean_injected_lag_s": round(float(np.mean(lags_injected)) / FPS, 3) if lags_injected else 0.0,
    }
