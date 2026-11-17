"""Comprehensive Forensic Failure-Mode Autopsy of Super-Champion v7 (`autopsy_v7.py`).

Strictly READ-ONLY over `exp_lab` cached OOF and evaluation predictions.
Writes all outputs to `experiments/outputs/forensic_v7/` (gitignored).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore")

EXP_LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_LAB / "py_pkgs"))
sys.path.insert(0, str(EXP_LAB))
sys.path.insert(0, str(EXP_LAB / "scripts"))
sys.path.insert(0, str(EXP_LAB / "evaluation"))

import numpy as np
import pandas as pd

import possession_metrics as pm
from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, HOME
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.mbr_snap import (
    ImpulseSnapConfig,
    decode_active_4dir,
    mbr_decode_sequences,
    snap_boundaries_to_impulse,
)
from src.decode.spell_reranker import SpellRerankerConfig, apply_spell_reranker, rle_runs
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    compute_seq481_gate,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator, count_interior_xdx
from src.eval import metrics
from exp_v6_super_champion import compute_physical_gates
from exp_w7a_player_set_net import load_cached_physv2_table
from exp_w7b_super_champion_v7 import (
    _blend_6fam_heads_v7,
    compute_local_newton_gate,
    compute_w_clean_v7,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "outputs" / "forensic_v7"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_spell_autopsy(
    gt_raw: np.ndarray,
    pred_raw: np.ndarray,
    half: np.ndarray | None,
    proba: np.ndarray | None = None,
    p_change: np.ndarray | None = None,
    p4d: np.ndarray | None = None,
    X_df: pd.DataFrame | None = None,
    insert_gaps: bool = True,
) -> dict:
    """Perform a complete spell-by-spell, boundary-by-boundary, and oracle-bucket forensic autopsy."""
    g = metrics.to_official(gt_raw, half, insert_gaps)
    p = metrics.to_official(pred_raw, half, insert_gaps)
    p = np.where(p == pm.X, pm.D, p).astype(np.int8)
    if half is not None and insert_gaps:
        p[g == pm.X] = pm.X

    runs = pm.valid_runs(g)
    p_filled = pm.fill_abstentions(p, runs)

    # Also build index mapping from official (gap-inserted) array back to raw frame array
    if half is not None and insert_gaps:
        boundaries = np.flatnonzero((half[1:] != half[:-1]) & (half[1:] > 0) & (half[:-1] > 0)) + 1
        # Map official index -> raw index
        raw_idx = np.arange(len(gt_raw), dtype=np.int64)
        off_to_raw = np.insert(raw_idx, boundaries, np.maximum(0, boundaries - 1))
    else:
        off_to_raw = np.arange(len(gt_raw), dtype=np.int64)

    gt_spells = pm.possession_spells(g, runs)
    pred_spells = pm.possession_spells(p_filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    n_gt = len(gt_spells)
    n_pred = len(pred_spells)

    # 1. Per-IoU metrics & exact matching pairs at each threshold
    per_iou = {}
    matched_gt_by_iou = {}
    matched_pred_by_iou = {}
    pairs_by_iou = {}

    for tau in pm.SPELL_IOUS:
        g_flags = np.zeros(n_gt, dtype=bool)
        p_flags = np.zeros(n_pred, dtype=bool)
        pairs = []
        for iou, gi, pj in overlaps:
            if iou < tau:
                break
            if g_flags[gi] or p_flags[pj]:
                continue
            g_flags[gi] = True
            p_flags[pj] = True
            pairs.append((gi, pj, float(iou)))
        tp = int(g_flags.sum())
        fp = n_pred - tp
        fn = n_gt - tp
        prec = tp / float(tp + fp) if (tp + fp) else 0.0
        rec = tp / float(tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * tp / float(n_gt + n_pred) if (n_gt + n_pred) else 0.0
        per_iou[tau] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        }
        matched_gt_by_iou[tau] = g_flags
        matched_pred_by_iou[tau] = p_flags
        pairs_by_iou[tau] = pairs

    spell_f1_mean = float(np.mean([per_iou[tau]["f1"] for tau in pm.SPELL_IOUS]))

    # 2. Best IoU for each GT spell under greedy 0.3 matching (and unconstrained max overlap)
    gt_matched_03 = {}
    pred_matched_03 = {}
    for gi, pj, iou in pairs_by_iou[0.3]:
        gt_matched_03[gi] = (pj, iou)
        pred_matched_03[pj] = (gi, iou)

    max_iou_for_gt = np.zeros(n_gt, dtype=np.float64)
    best_pred_for_gt = np.full(n_gt, -1, dtype=np.int64)
    for iou, gi, pj in overlaps:
        if iou > max_iou_for_gt[gi]:
            max_iou_for_gt[gi] = iou
            best_pred_for_gt[gi] = pj

    max_iou_for_pred = np.zeros(n_pred, dtype=np.float64)
    best_gt_for_pred = np.full(n_pred, -1, dtype=np.int64)
    for iou, gi, pj in overlaps:
        if iou > max_iou_for_pred[pj]:
            max_iou_for_pred[pj] = iou
            best_gt_for_pred[pj] = gi

    # 3. Classify every GT spell that fails at IoU=0.3 (FN@0.3)
    fn03_buckets = {
        "FN1a_opp_swallow_ultrashort_lt25": 0,   # <1.0s (impossible for dmin_team=34)
        "FN1b_opp_swallow_short_25_50": 0,       # 1.0-2.0s quick interception/clearance
        "FN1c_opp_swallow_med_51_125": 0,        # 2.0-5.0s possession swallowed by opponent
        "FN1d_opp_swallow_long_gt125": 0,        # >5.0s possession called wrong team
        "FN2_dead_swallow": 0,                   # >50% predicted as DEAD
        "FN3_merged_across_missed_D": 0,         # Two GT spells of same team X separated by GT D merged into 1 pred X spell
        "FN4_merged_across_missed_opp": 0,       # Two GT spells of same team X separated by GT opp Y merged into 1 pred X spell
        "FN5_low_iou_fragment_or_shift": 0,      # Overlapped same team but IoU < 0.3 (and not FN3/FN4)
    }
    fn03_details = []

    for gi, (ga, gb, g_team) in enumerate(gt_spells):
        if matched_gt_by_iou[0.3][gi]:
            continue
        L = gb - ga
        opp_team = 1 - g_team
        pred_slice = p_filled[ga:gb]
        frac_same = float(np.mean(pred_slice == g_team))
        frac_opp = float(np.mean(pred_slice == opp_team))
        frac_dead = float(np.mean(pred_slice == pm.D))

        # Check if this GT spell is covered mostly by a predicted spell of the SAME team `pj`
        # that was already consumed by an adjacent GT spell `gi_other` of the same team!
        pj_cov = best_pred_for_gt[gi]
        bucket = None
        if frac_same >= 0.50 and pj_cov >= 0 and pj_cov in pred_matched_03:
            gi_other, _ = pred_matched_03[pj_cov]
            ga_o, gb_o, _ = gt_spells[gi_other]
            gap_a, gap_b = (gb_o, ga) if gb_o <= ga else (gb, ga_o)
            if gap_b > gap_a:
                gap_gt = g[gap_a:gap_b]
                if np.mean(gap_gt == pm.D) >= 0.50:
                    bucket = "FN3_merged_across_missed_D"
                else:
                    bucket = "FN4_merged_across_missed_opp"
            else:
                bucket = "FN5_low_iou_fragment_or_shift"
        elif frac_dead >= 0.50:
            bucket = "FN2_dead_swallow"
        elif frac_opp >= 0.50:
            if L < 25:
                bucket = "FN1a_opp_swallow_ultrashort_lt25"
            elif L <= 50:
                bucket = "FN1b_opp_swallow_short_25_50"
            elif L <= 125:
                bucket = "FN1c_opp_swallow_med_51_125"
            else:
                bucket = "FN1d_opp_swallow_long_gt125"
        else:
            bucket = "FN5_low_iou_fragment_or_shift"

        fn03_buckets[bucket] += 1
        r_a, r_b = int(off_to_raw[ga]), int(off_to_raw[gb - 1]) + 1
        mean_p_team = float(np.mean(proba[r_a:r_b, g_team])) if proba is not None and r_b > r_a else -1.0
        max_p_team = float(np.max(proba[r_a:r_b, g_team])) if proba is not None and r_b > r_a else -1.0
        fn03_details.append({
            "gi": gi, "ga": r_a, "gb": r_b, "len": L, "team": int(g_team),
            "bucket": bucket, "frac_same": frac_same, "frac_opp": frac_opp, "frac_dead": frac_dead,
            "mean_p_team": mean_p_team, "max_p_team": max_p_team,
        })

    # 4. Classify every Predicted spell that fails at IoU=0.3 (FP@0.3)
    fp03_buckets = {
        "FP1_hallucinated_opp_blip_YXY": 0,      # Predicted X inside true Y spell
        "FP2a_split_fragment_by_false_D": 0,     # Extra fragment of true X split by false D (X -> D_pred -> X)
        "FP2b_split_fragment_by_false_opp": 0,   # Extra fragment of true X split by false opp Y (X -> Y_pred -> X)
        "FP3_false_live_inside_DEAD": 0,         # Predicted live X when GT is DEAD
        "FP4_wrong_team_on_live": 0,             # Predicted X on a standalone GT Y spell (e.g., after D restart)
        "FP5_other_boundary_spill": 0,
    }

    for pj, (pa, pb, p_team) in enumerate(pred_spells):
        if matched_pred_by_iou[0.3][pj]:
            continue
        L = pb - pa
        opp_team = 1 - p_team
        gt_slice = g[pa:pb]
        frac_same = float(np.mean(gt_slice == p_team))
        frac_opp = float(np.mean(gt_slice == opp_team))
        frac_dead = float(np.mean(gt_slice == pm.D))

        # Check if this predicted spell is mostly inside a GT spell of the SAME team `gi_cov`
        # that was already matched to another predicted spell `pj_other`!
        gi_cov = best_gt_for_pred[pj]
        bucket = None
        if frac_same >= 0.50 and gi_cov >= 0 and gi_cov in gt_matched_03:
            pj_other, _ = gt_matched_03[gi_cov]
            pa_o, pb_o, _ = pred_spells[pj_other]
            gap_a, gap_b = (pb_o, pa) if pb_o <= pa else (pb, pa_o)
            if gap_b > gap_a:
                gap_pred = p_filled[gap_a:gap_b]
                if np.mean(gap_pred == pm.D) >= 0.50:
                    bucket = "FP2a_split_fragment_by_false_D"
                else:
                    bucket = "FP2b_split_fragment_by_false_opp"
            else:
                bucket = "FP5_other_boundary_spill"
        elif frac_dead >= 0.50:
            bucket = "FP3_false_live_inside_DEAD"
        elif frac_opp >= 0.50:
            # Was it flanked by opp_team in pred (Y -> X -> Y) or preceded by D?
            prev_lbl = p_filled[pa - 1] if pa > 0 else pm.X
            next_lbl = p_filled[pb] if pb < len(p_filled) else pm.X
            if prev_lbl == opp_team and next_lbl == opp_team:
                bucket = "FP1_hallucinated_opp_blip_YXY"
            else:
                bucket = "FP4_wrong_team_on_live"
        else:
            bucket = "FP5_other_boundary_spill"

        fp03_buckets[bucket] += 1

    # 5. Boundary Error Anatomy for Matched Spells (Matched at IoU >= 0.3, failing at 0.4, 0.5, 0.6, or 0.7)
    boundary_stats = {
        "n_matched_03": len(pairs_by_iou[0.3]),
        "n_fail_04": 0,
        "n_fail_05": 0,
        "n_fail_06": 0,
        "n_fail_07": 0,
        "start_err_from_D_abs": [],
        "start_err_from_D_signed": [],
        "start_err_from_opp_abs": [],
        "start_err_from_opp_signed": [],
        "end_err_to_D_abs": [],
        "end_err_to_D_signed": [],
        "end_err_to_opp_abs": [],
        "end_err_to_opp_signed": [],
        # Attribution of why IoU < 0.7 on matched spells:
        "fail07_due_to_D_entry": 0,        # |da| > |db| and GT preceded by D
        "fail07_due_to_opp_entry": 0,      # |da| > |db| and GT preceded by Opp
        "fail07_due_to_D_exit": 0,         # |db| >= |da| and GT followed by D
        "fail07_due_to_opp_exit": 0,       # |db| >= |da| and GT followed by Opp
    }

    for gi, pj, iou in pairs_by_iou[0.3]:
        ga, gb, g_team = gt_spells[gi]
        pa, pb, p_team = pred_spells[pj]
        da = pa - ga   # >0 means predicted spell started LATE; <0 means started EARLY
        db = pb - gb   # >0 means predicted spell ended LATE; <0 means ended EARLY

        prev_gt = g[ga - 1] if ga > 0 else pm.X
        next_gt = g[gb] if gb < len(g) else pm.X

        if prev_gt == pm.D:
            boundary_stats["start_err_from_D_abs"].append(abs(da))
            boundary_stats["start_err_from_D_signed"].append(da)
        elif prev_gt in (pm.H, pm.A):
            boundary_stats["start_err_from_opp_abs"].append(abs(da))
            boundary_stats["start_err_from_opp_signed"].append(da)

        if next_gt == pm.D:
            boundary_stats["end_err_to_D_abs"].append(abs(db))
            boundary_stats["end_err_to_D_signed"].append(db)
        elif next_gt in (pm.H, pm.A):
            boundary_stats["end_err_to_opp_abs"].append(abs(db))
            boundary_stats["end_err_to_opp_signed"].append(db)

        if iou < 0.4:
            boundary_stats["n_fail_04"] += 1
        if iou < 0.5:
            boundary_stats["n_fail_05"] += 1
        if iou < 0.6:
            boundary_stats["n_fail_06"] += 1
        if iou < 0.7:
            boundary_stats["n_fail_07"] += 1
            if abs(da) >= abs(db):
                if prev_gt == pm.D:
                    boundary_stats["fail07_due_to_D_entry"] += 1
                else:
                    boundary_stats["fail07_due_to_opp_entry"] += 1
            else:
                if next_gt == pm.D:
                    boundary_stats["fail07_due_to_D_exit"] += 1
                else:
                    boundary_stats["fail07_due_to_opp_exit"] += 1

    # 6. Exact Oracle Marginal Gain Simulation (`+Delta spell_f1_mean`) for each surgical repair type!
    # We test on the official arrays `g` and `p_filled`:
    def _eval_mod(p_mod: np.ndarray) -> float:
        return float(pm.spell_scores(g, p_mod, runs)["mean_f1"]) - spell_f1_mean

    # Oracle A: Snap boundaries of already-matched spells (`IoU >= 0.3`) within +/- 15 frames to GT boundaries
    p_orc_snap15 = p_filled.copy()
    p_orc_snap35 = p_filled.copy()
    p_orc_snap_d = p_filled.copy()
    p_orc_snap_ha = p_filled.copy()
    for gi, pj, iou in pairs_by_iou[0.3]:
        ga, gb, g_team = gt_spells[gi]
        pa, pb, p_team = pred_spells[pj]
        prev_gt = g[ga - 1] if ga > 0 else pm.X
        next_gt = g[gb] if gb < len(g) else pm.X

        # +/- 15 frame boundary snap
        if 0 < abs(pa - ga) <= 15:
            lo, hi = min(pa, ga), max(pa, ga)
            p_orc_snap15[lo:hi] = g[lo:hi]
        if 0 < abs(pb - gb) <= 15:
            lo, hi = min(pb, gb), max(pb, gb)
            p_orc_snap15[lo:hi] = g[lo:hi]

        # +/- 35 frame boundary snap
        if 0 < abs(pa - ga) <= 35:
            lo, hi = min(pa, ga), max(pa, ga)
            p_orc_snap35[lo:hi] = g[lo:hi]
            if prev_gt == pm.D:
                p_orc_snap_d[lo:hi] = g[lo:hi]
            elif prev_gt in (pm.H, pm.A):
                p_orc_snap_ha[lo:hi] = g[lo:hi]
        if 0 < abs(pb - gb) <= 35:
            lo, hi = min(pb, gb), max(pb, gb)
            p_orc_snap35[lo:hi] = g[lo:hi]
            if next_gt == pm.D:
                p_orc_snap_d[lo:hi] = g[lo:hi]
            elif next_gt in (pm.H, pm.A):
                p_orc_snap_ha[lo:hi] = g[lo:hi]

    # Oracle B: Fix False Positives (`Y -> X -> Y` hallucinated blips & false splits)
    p_orc_kill_fp_yxy = p_filled.copy()
    p_orc_kill_fp_d = p_filled.copy()
    for pj, (pa, pb, p_team) in enumerate(pred_spells):
        if matched_pred_by_iou[0.3][pj]:
            continue
        opp_team = 1 - p_team
        gt_slice = g[pa:pb]
        if np.mean(gt_slice == opp_team) >= 0.50:
            p_orc_kill_fp_yxy[pa:pb] = g[pa:pb]
        elif np.mean(gt_slice == pm.D) >= 0.50:
            p_orc_kill_fp_d[pa:pb] = pm.D

    # Oracle C: Recover Missed Short/Med Opponent Turnovers (`FN1b` + `FN1c`: 25..125 frames)
    p_orc_rec_short_turn = p_filled.copy()
    p_orc_rec_ultrashort = p_filled.copy()
    p_orc_split_missed_d = p_filled.copy()
    for gi, (ga, gb, g_team) in enumerate(gt_spells):
        if matched_gt_by_iou[0.3][gi]:
            continue
        L = gb - ga
        opp_team = 1 - g_team
        pred_slice = p_filled[ga:gb]
        if np.mean(pred_slice == opp_team) >= 0.50:
            if L < 25:
                p_orc_rec_ultrashort[ga:gb] = g_team
            elif L <= 125:
                p_orc_rec_short_turn[ga:gb] = g_team

    # Oracle D: Insert missed GT `D` stoppages that merged two same-team spells (`X -> D_gt -> X`)
    gt_segs = pm.segments(g, runs)
    for k in range(1, len(gt_segs) - 1):
        sa, sb, scode = gt_segs[k]
        if scode == pm.D and gt_segs[k - 1][2] in (pm.H, pm.A) and gt_segs[k - 1][2] == gt_segs[k + 1][2]:
            team_x = gt_segs[k - 1][2]
            if np.mean(p_filled[sa:sb] == team_x) >= 0.50:
                p_orc_split_missed_d[sa:sb] = pm.D

    # Oracle E: Perfect `Alive vs Dead` mask (replace D vs Live with GT D, keeping H vs A preference from model/proba)
    p_orc_perfect_d = p_filled.copy()
    is_gt_d = (g == pm.D)
    p_orc_perfect_d[is_gt_d] = pm.D
    # Where GT is alive (H or A) but pred was D, fill with nearest/proba H vs A
    gt_alive_pred_d = ((g == pm.H) | (g == pm.A)) & (p_filled == pm.D)
    if np.any(gt_alive_pred_d):
        if proba is not None:
            raw_i = off_to_raw[gt_alive_pred_d]
            p_orc_perfect_d[gt_alive_pred_d] = np.where(
                proba[raw_i, HOME] >= proba[raw_i, AWAY], pm.H, pm.A
            ).astype(np.int8)
        else:
            p_orc_perfect_d[gt_alive_pred_d] = g[gt_alive_pred_d]

    oracle_gains = {
        "orc_snap_15f": _eval_mod(p_orc_snap15),
        "orc_snap_35f": _eval_mod(p_orc_snap35),
        "orc_snap_D_bounds_35f": _eval_mod(p_orc_snap_d),
        "orc_snap_HA_bounds_35f": _eval_mod(p_orc_snap_ha),
        "orc_kill_fp_opp_yxy": _eval_mod(p_orc_kill_fp_yxy),
        "orc_kill_fp_dead": _eval_mod(p_orc_kill_fp_d),
        "orc_recover_short_turnovers_25_125f": _eval_mod(p_orc_rec_short_turn),
        "orc_recover_ultrashort_lt25f": _eval_mod(p_orc_rec_ultrashort),
        "orc_insert_missed_D_in_XDX": _eval_mod(p_orc_split_missed_d),
        "orc_perfect_alive_dead_mask": _eval_mod(p_orc_perfect_d),
    }

    return {
        "spell_f1_mean": spell_f1_mean,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "per_iou": per_iou,
        "fn03_buckets": fn03_buckets,
        "fp03_buckets": fp03_buckets,
        "boundary_stats": boundary_stats,
        "oracle_gains": oracle_gains,
        "fn03_details": fn03_details,
    }


def aggregate_autopsies(autopsies: list[dict], name: str) -> dict:
    """Aggregate forensic dictionaries across folds/matches."""
    n = len(autopsies)
    mean_f1 = float(np.mean([a["spell_f1_mean"] for a in autopsies]))
    tot_gt = int(sum(a["n_gt"] for a in autopsies))
    tot_pred = int(sum(a["n_pred"] for a in autopsies))

    per_iou_agg = {}
    for tau in pm.SPELL_IOUS:
        per_iou_agg[tau] = {
            "f1": float(np.mean([a["per_iou"][tau]["f1"] for a in autopsies])),
            "precision": float(np.mean([a["per_iou"][tau]["precision"] for a in autopsies])),
            "recall": float(np.mean([a["per_iou"][tau]["recall"] for a in autopsies])),
            "tp": int(sum(a["per_iou"][tau]["tp"] for a in autopsies)),
            "fp": int(sum(a["per_iou"][tau]["fp"] for a in autopsies)),
            "fn": int(sum(a["per_iou"][tau]["fn"] for a in autopsies)),
        }

    fn_keys = list(autopsies[0]["fn03_buckets"].keys())
    fn_agg = {k: int(sum(a["fn03_buckets"][k] for a in autopsies)) for k in fn_keys}

    fp_keys = list(autopsies[0]["fp03_buckets"].keys())
    fp_agg = {k: int(sum(a["fp03_buckets"][k] for a in autopsies)) for k in fp_keys}

    orc_keys = list(autopsies[0]["oracle_gains"].keys())
    orc_agg = {k: float(np.mean([a["oracle_gains"][k] for a in autopsies])) for k in orc_keys}

    b_agg = {
        "n_matched_03": int(sum(a["boundary_stats"]["n_matched_03"] for a in autopsies)),
        "n_fail_04": int(sum(a["boundary_stats"]["n_fail_04"] for a in autopsies)),
        "n_fail_05": int(sum(a["boundary_stats"]["n_fail_05"] for a in autopsies)),
        "n_fail_06": int(sum(a["boundary_stats"]["n_fail_06"] for a in autopsies)),
        "n_fail_07": int(sum(a["boundary_stats"]["n_fail_07"] for a in autopsies)),
        "fail07_due_to_D_entry": int(sum(a["boundary_stats"]["fail07_due_to_D_entry"] for a in autopsies)),
        "fail07_due_to_opp_entry": int(sum(a["boundary_stats"]["fail07_due_to_opp_entry"] for a in autopsies)),
        "fail07_due_to_D_exit": int(sum(a["boundary_stats"]["fail07_due_to_D_exit"] for a in autopsies)),
        "fail07_due_to_opp_exit": int(sum(a["boundary_stats"]["fail07_due_to_opp_exit"] for a in autopsies)),
    }
    for k in ("start_err_from_D_abs", "start_err_from_D_signed", "start_err_from_opp_abs", "start_err_from_opp_signed",
              "end_err_to_D_abs", "end_err_to_D_signed", "end_err_to_opp_abs", "end_err_to_opp_signed"):
        vals = [x for a in autopsies for x in a["boundary_stats"][k]]
        b_agg[k + "_mean"] = float(np.mean(vals)) if vals else 0.0
        b_agg[k + "_median"] = float(np.median(vals)) if vals else 0.0
        b_agg[k + "_p90"] = float(np.percentile(np.abs(vals), 90)) if vals else 0.0

    print(f"\n=====================================================================================")
    print(f"FORENSIC AUTOPSY REPORT: {name}")
    print(f"=====================================================================================")
    print(f"  Headline spell_f1_mean = {mean_f1:.4f} | Total GT Spells = {tot_gt} | Total Pred Spells = {tot_pred} (Diff = {tot_pred - tot_gt:+d})")
    print(f"\n  [1] Per-IoU Threshold Breakdown:")
    print(f"      {'IoU':>5s} | {'F1':>7s} | {'Prec':>7s} | {'Recall':>7s} | {'TP':>5s} | {'FP':>5s} | {'FN':>5s} | {'Drop vs Prev IoU':>16s}")
    prev_f1 = None
    for tau in pm.SPELL_IOUS:
        r = per_iou_agg[tau]
        drop_str = f"{r['f1'] - prev_f1:+.4f}" if prev_f1 is not None else "   ---"
        prev_f1 = r["f1"]
        print(f"      {tau:5.1f} | {r['f1']:7.4f} | {r['precision']:7.4f} | {r['recall']:7.4f} | {r['tp']:5d} | {r['fp']:5d} | {r['fn']:5d} | {drop_str:>16s}")

    tot_fn03 = sum(fn_agg.values())
    print(f"\n  [2] False Negative Taxonomy at IoU=0.3 (Total FN@0.3 = {tot_fn03} / {tot_gt} GT spells = {100*tot_fn03/max(tot_gt,1):.1f}%):")
    for k, v in fn_agg.items():
        print(f"      {k:40s}: {v:4d} ({100*v/max(tot_fn03,1):5.1f}% of FN@0.3 | {100*v/max(tot_gt,1):4.1f}% of all GT)")

    tot_fp03 = sum(fp_agg.values())
    print(f"\n  [3] False Positive Taxonomy at IoU=0.3 (Total FP@0.3 = {tot_fp03} / {tot_pred} Pred spells = {100*tot_fp03/max(tot_pred,1):.1f}%):")
    for k, v in fp_agg.items():
        print(f"      {k:40s}: {v:4d} ({100*v/max(tot_fp03,1):5.1f}% of FP@0.3 | {100*v/max(tot_pred,1):4.1f}% of all Pred)")

    m03 = max(b_agg["n_matched_03"], 1)
    print(f"\n  [4] Boundary & Cliff-Drop Anatomy on Matched Spells (Matched@0.3 = {b_agg['n_matched_03']}):")
    print(f"      Fail@0.4 = {b_agg['n_fail_04']:3d} | Fail@0.5 = {b_agg['n_fail_05']:3d} | Fail@0.6 = {b_agg['n_fail_06']:3d} | Fail@0.7 = {b_agg['n_fail_07']:3d} ({100*b_agg['n_fail_07']/m03:.1f}% of Matched@0.3)")
    print(f"      Why Matched@0.3 fail @0.7: D_entry={b_agg['fail07_due_to_D_entry']}, Opp_entry={b_agg['fail07_due_to_opp_entry']}, D_exit={b_agg['fail07_due_to_D_exit']}, Opp_exit={b_agg['fail07_due_to_opp_exit']}")
    print(f"      Start Error after DEAD (D->Live)  : mean|err|={b_agg['start_err_from_D_abs_mean']:5.1f}f, signed={b_agg['start_err_from_D_signed_mean']:+5.1f}f, p90={b_agg['start_err_from_D_abs_p90']:5.1f}f")
    print(f"      Start Error after OPP  (Y->X)     : mean|err|={b_agg['start_err_from_opp_abs_mean']:5.1f}f, signed={b_agg['start_err_from_opp_signed_mean']:+5.1f}f, p90={b_agg['start_err_from_opp_abs_p90']:5.1f}f")
    print(f"      End Error before DEAD  (Live->D)  : mean|err|={b_agg['end_err_to_D_abs_mean']:5.1f}f, signed={b_agg['end_err_to_D_signed_mean']:+5.1f}f, p90={b_agg['end_err_to_D_abs_p90']:5.1f}f")
    print(f"      End Error before OPP   (X->Y)     : mean|err|={b_agg['end_err_to_opp_abs_mean']:5.1f}f, signed={b_agg['end_err_to_opp_signed_mean']:+5.1f}f, p90={b_agg['end_err_to_opp_abs_p90']:5.1f}f")

    print(f"\n  [5] Exact Oracle Marginal Gain (+Delta spell_f1_mean if bucket is surgically fixed):")
    for k, v in sorted(orc_agg.items(), key=lambda kv: -kv[1]):
        print(f"      {k:40s}: {v:+.4f} -> {mean_f1 + v:.4f}")

    return {
        "name": name,
        "spell_f1_mean": mean_f1,
        "n_gt": tot_gt,
        "n_pred": tot_pred,
        "per_iou": per_iou_agg,
        "fn03_buckets": fn_agg,
        "fp03_buckets": fp_agg,
        "boundary_stats": b_agg,
        "oracle_gains": orc_agg,
    }


def main():
    t0 = time.time()
    print("=== LOADING CACHED V7 PREDICTIONS & DATA FROM EXP_LAB (READ-ONLY) ===", flush=True)

    sportec = load_sportec()
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    X_v3, X_v6, gates_v3, gates_v6 = {}, {}, {}, {}
    for mid, m in sportec.items():
        X_v3[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", m.half)
        X_v6[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", m.half)
        gates_v3[mid] = compute_physical_gates(X_v3[mid], m.half)
        gates_v6[mid] = compute_physical_gates(X_v6[mid], m.half)

    X_sample, gates_sample = {}, {}
    for gid in GAMES:
        X_sample[gid] = load_cached_physv2_table(f"sample_{gid}_raw", sample_matches[gid].half)
        gates_sample[gid] = compute_physical_gates(X_sample[gid], sample_matches[gid].half)

    X_atd_raw = load_cached_physv2_table("atd_0f0834", atd.half)
    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)
    gate_atd_raw = compute_physical_gates(X_atd_raw, atd.half)
    gate_atd_rep = compute_physical_gates(X_atd_rep, atd.half)

    w7b_dir = EXP_LAB / "experiments" / "outputs" / "exp_w7b_oof"
    w6_dir = EXP_LAB / "experiments" / "outputs" / "exp_w6_oof"
    b_dir = EXP_LAB / "experiments" / "outputs" / "exp_b_sym_oof"
    plus_dir = EXP_LAB / "experiments" / "outputs" / "exp_c_plus_oof"
    g_dir = EXP_LAB / "experiments" / "outputs" / "exp_g_oof"
    h_dir = EXP_LAB / "experiments" / "outputs" / "exp_h_oof"
    w3_dir = EXP_LAB / "experiments" / "outputs" / "exp_w3_oof"
    w4_dir = EXP_LAB / "experiments" / "outputs" / "exp_w4_oof"
    w5_dir = EXP_LAB / "experiments" / "outputs" / "exp_w5_tree_oof"
    w5b_dir = EXP_LAB / "experiments" / "outputs" / "exp_w5b_oof"
    w7a_dir = EXP_LAB / "experiments" / "outputs" / "exp_w7a_oof"

    with open(w6_dir / "w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)
    with open(b_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)

    best_calib_cfg: TransductiveCalibConfig = w6_sum["best_calib_cfg"]
    lomo_rerank_cfgs = w6_sum["lomo_rerank_cfgs"]
    full_rerank_cfg = w6_sum["full_rerank_cfg"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)

    # Load saved w7b OOF predictions for Sportec
    w7b_oof = {T: np.load(w7b_dir / f"w7b_{T}.npz") for T in MATCH_IDS}
    w3_oof = {T: np.load(w3_dir / f"b4d_{T}.npz") for T in MATCH_IDS}

    # 1. Judge 1A (Sportec v3) Autopsy on v7 Rerank (Candidate 8) & v7 MBR (Candidate 11)
    aut_j1a_rerank = []
    aut_j1a_mbr = []
    aut_j1b_std_rerank = []
    aut_j1b_gen_rerank = []

    for T in MATCH_IDS:
        h = sportec[T].half
        y = sportec[T].labels
        z = w7b_oof[T]
        p4d_v3 = w3_oof[T]["p4d_v3"]
        p4d_v6 = w3_oof[T]["p4d_v6"]

        aut_j1a_rerank.append(analyze_spell_autopsy(
            y, z["pred_v3_rerank"], h,
            proba=z["proba_v3"], p_change=z["pchange_v3"], p4d=p4d_v3, X_df=X_v3[T],
        ))
        aut_j1a_mbr.append(analyze_spell_autopsy(
            y, z["pred_v3_mbr"], h,
            proba=z["proba_v3"], p_change=z["pchange_v3"], p4d=p4d_v3, X_df=X_v3[T],
        ))
        aut_j1b_std_rerank.append(analyze_spell_autopsy(
            y, z["pred_v6_std_rerank"], h,
            proba=z["proba_v6_std"], p_change=z["pchange_v6_std"], p4d=p4d_v6, X_df=X_v6[T],
        ))
        aut_j1b_gen_rerank.append(analyze_spell_autopsy(
            y, z["pred_v6_gen_rerank"], h,
            proba=z["proba_v6_gen"], p_change=z["pchange_v6_gen"], p4d=p4d_v6, X_df=X_v6[T],
        ))

    rep_j1a_rr = aggregate_autopsies(aut_j1a_rerank, "JUDGE 1A: Sportec v3 (7-Fold LOMO) — v7 6-Fam Local-Newton + XDX + 9-Model SpellReranker")
    rep_j1a_mbr = aggregate_autopsies(aut_j1a_mbr, "JUDGE 1A: Sportec v3 (7-Fold LOMO) — v7 Super-Champion Grand MBR")
    rep_j1b_std = aggregate_autopsies(aut_j1b_std_rerank, "JUDGE 1B-Std: Sportec v6 Ghost-Ball (7-Fold LOMO) — v7 Rerank")
    rep_j1b_gen = aggregate_autopsies(aut_j1b_gen_rerank, "JUDGE 1B-Gen: Sportec v6 100% Genuine Native-v6 (7-Fold LOMO) — v7 Rerank")

    # 2. Judge 2 (Metrica Sample 3 games) & Judge 3 (BAR-ATH)
    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        pe_plus = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f:
        sample_v6, atd_rep_v6 = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(w3_dir / "b4d_j23_preds.pkl", "rb") as f:
        b4d_j23 = pickle.load(f)
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f:
        pe_p2 = pickle.load(f)
    with open(w5_dir / "eval_preds_cb_xgb.pkl", "rb") as f:
        pe_w5 = pickle.load(f)
    with open(w5b_dir / "seq_physv2_j23_preds.pkl", "rb") as f:
        pe_w5b = pickle.load(f)
    with open(w6_dir / "atd_raw_v6_bigru.pkl", "rb") as f:
        atd_raw_extra = pickle.load(f)
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f:
        pset_j23 = pickle.load(f)

    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )

    aut_j2_raw = []
    aut_j2_nod = []
    for gid in GAMES:
        ms = sample_matches[gid]
        dh = sample_h[gid]
        dpl = pe_plus["sample"][gid]
        dv6 = sample_v6[gid]
        dp2 = pe_p2["sample"][gid]
        dw5 = pe_w5["sample"][gid]
        dw5b = pe_w5b["sample"][gid]
        dpset = pset_j23["sample"][gid]
        a_s = compute_local_newton_gate(X_sample[gid], ms.half, gates_sample[gid][0])
        wc_s = compute_w_clean_v7(X_sample[gid], ms.half, gates_sample[gid][0])

        ps, cs = synthesize_head_decoupled_proba(
            dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
            dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
            ms.half, dv6["g_v6"], dv6["tr_v6"],
            w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main,
            tcn_physv2_proba=dw5b["tcn_physv2_sym"],
            bigru_physv2_proba=dw5b["bigru_physv2_sym"],
            tcn_physv2_change=dw5b["tcnc_physv2_sym"],
            bigru_physv2_change=dw5b["bigruc_physv2_sym"],
            alpha_p2=a_s,
        )
        ps, cs = _blend_6fam_heads_v7(
            ps, cs, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"],
            dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"],
            a_s, wc_s, 0.05, 0.30, 0.18, 0.08,
        )
        ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], best_calib_cfg)
        fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"]]
        p4ds = b4d_j23["sample"][gid]

        prs = apply_spell_reranker(
            apply_xdx_arbitrator(decode_v4(ps, ms.half, dec_v5_hyb, p_change=cs), ps, ms.half, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1),
            ps, ms.half, X_sample[gid], cs, p4ds, fs, cfg=full_rerank_cfg,
        )
        aut_j2_raw.append(analyze_spell_autopsy(ms.labels, prs, ms.half, proba=ps, p_change=cs, p4d=p4ds, X_df=X_sample[gid]))
        aut_j2_nod.append(analyze_spell_autopsy(without_dead_margin(ms.labels, ms.half), prs, ms.half, proba=ps, p_change=cs, p4d=p4ds, X_df=X_sample[gid]))

    rep_j2_raw = aggregate_autopsies(aut_j2_raw, "JUDGE 2 (Raw): Metrica Sample 3 Games — v7 Rerank")
    rep_j2_nod = aggregate_autopsies(aut_j2_nod, "JUDGE 2 (without_dead_margin ±3s): Metrica Sample 3 Games — v7 Rerank")

    # 3. Judge 3 (BAR-ATH Public Window: Raw & Broadcast-Repaired)
    sub_dir = EXP_LAB / "submissions"
    df_sub = pd.read_csv(sub_dir / "submission_v7_6family_local_newton_xdx.csv")
    pred_rep = pm.label_codes(df_sub)

    # Also build BAR-ATH Raw & Repaired probabilities for oracle inspection
    dpl_ap = pe_plus["atd_rep"]
    dp2_ap = pe_p2["atd_rep"]
    dw5_ap = pe_w5["atd_rep"]
    dw5b_ap = pe_w5b["atd_rep"]
    dpset_ap = pset_j23["atd_rep"]
    a_ap = compute_local_newton_gate(X_atd_rep, atd.half, gate_atd_rep[0])
    wc_ap = compute_w_clean_v7(X_atd_rep, atd.half, gate_atd_rep[0])
    pap, cap = synthesize_head_decoupled_proba(
        dpl_ap["g_plus_sym"], dp2_ap["g_physv2_sym"], dw5_ap["cb_physv2_sym"], dw5_ap["xgb_physv2_sym"],
        atd_rep_h["t_sym"], atd_rep_h["b_sym"], dpl_ap["tr_plus_sym"], dp2_ap["tr_physv2_sym"],
        atd_rep_h["tc_sym"], atd_rep_h["bc_sym"],
        atd.half, atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"],
        w_ghost=gate_atd_rep[0], w_gap=gate_atd_rep[1], cfg=scfg_main,
        tcn_physv2_proba=dw5b_ap["tcn_physv2_sym"],
        bigru_physv2_proba=dw5b_ap["bigru_physv2_sym"],
        tcn_physv2_change=dw5b_ap["tcnc_physv2_sym"],
        bigru_physv2_change=dw5b_ap["bigruc_physv2_sym"],
        alpha_p2=a_ap,
    )
    pap, cap = _blend_6fam_heads_v7(
        pap, cap, atd_rep_h["t_sym"], atd_rep_h["b_sym"], dw5b_ap["bigru_physv2_sym"], dpset_ap["pset_proba"],
        atd_rep_h["tc_sym"], atd_rep_h["bc_sym"], dw5b_ap["bigruc_physv2_sym"], dpset_ap["pset_change"],
        a_ap, wc_ap, 0.05, 0.30, 0.18, 0.08,
    )
    pap = calibrate_match_transductive(pap, half_atd_dec, X_atd_rep, best_calib_cfg)

    pub_gt = metrics.public_gt()
    aut_j3_rep = analyze_spell_autopsy(
        pub_gt, pred_rep, None, proba=pap, p_change=cap, p4d=b4d_j23["atd_rep"], X_df=X_atd_rep, insert_gaps=False
    )
    rep_j3_rep = aggregate_autopsies([aut_j3_rep], "JUDGE 3 (BAR-ATH Public Window Repaired): v7 6-Fam Local-Newton + XDX + Rerank")

    # Print exact spell-by-spell list on BAR-ATH public window!
    print("\n--- BAR-ATH Public Window (11,185..18,684) Exact Spell-by-Spell Comparison ---", flush=True)
    runs_pub = pm.valid_runs(pub_gt)
    gt_sp_pub = pm.possession_spells(pub_gt, runs_pub)
    pr_sp_pub = pm.possession_spells(pred_rep, runs_pub)
    ov_pub = pm.spell_overlaps(gt_sp_pub, pr_sp_pub)
    print(f"  GT Spells ({len(gt_sp_pub)}):")
    for i, (ga, gb, gc) in enumerate(gt_sp_pub):
        best_m = [(iou, pj) for iou, gi, pj in ov_pub if gi == i]
        best_iou, best_pj = best_m[0] if best_m else (0.0, -1)
        pa, pb, pc = pr_sp_pub[best_pj] if best_pj >= 0 else (-1, -1, -1)
        print(f"    GT#{i:02d} [{'H' if gc==0 else 'A'}] [{ga:5d}..{gb:5d}] (len={gb-ga:3d}f) <-> Pred#{best_pj:02d} [{pa:5d}..{pb:5d}] IoU={best_iou:.3f} (da={pa-ga:+d}, db={pb-gb:+d})")
    print(f"  Unmatched Pred Spells at IoU=0.3:")
    for j, (pa, pb, pc) in enumerate(pr_sp_pub):
        if not aut_j3_rep["per_iou"][0.3] or j not in [pj for iou, gi, pj in ov_pub if iou >= 0.3]:
            print(f"    Pred#{j:02d} [{'H' if pc==0 else 'A'}] [{pa:5d}..{pb:5d}] (len={pb-pa:3d}f) | GT slice counts: H={np.sum(pub_gt[pa:pb]==0)}, A={np.sum(pub_gt[pa:pb]==1)}, D={np.sum(pub_gt[pa:pb]==2)}")

    # Save JSON summary
    with open(OUT_DIR / "autopsy_v7_summary.json", "w") as f:
        json.dump({
            "j1a_rerank": rep_j1a_rr,
            "j1a_mbr": rep_j1a_mbr,
            "j1b_std": rep_j1b_std,
            "j1b_gen": rep_j1b_gen,
            "j2_raw": rep_j2_raw,
            "j2_nod": rep_j2_nod,
            "j3_rep": rep_j3_rep,
        }, f, indent=2)

    print(f"\nCompleted full v7 forensic autopsy in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
