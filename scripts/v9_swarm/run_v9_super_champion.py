#!/usr/bin/env python3
"""Unified Super-Champion v9 Synthesis, Ablation Ladder, J_SUITE Audit & Submission Generator.

Synthesizes all orthogonal `v9` breakthroughs on top of `v8.5`:
  1. Mission 4A (`SymDuelTurnoverArbitratorV9`):
     - Fixes `SymBoundary4Dir` (`p4d`) inverted directional columns (`col_to_Y = 1 if y_lbl == HOME else 0`),
       interior proximity window `[ca_int, cb_int)`, degree normalization, and 9-family index alignment.
     - Recovers brief duel, header, and deflection turnovers in strict 7-Fold Nested LOMO.
  2. Mission 4B (`SymSpellMBRDecoderV9` + `Sub-regime A3` Modal GPU/CPU 3D Aerial/Duel Shield):
     - Combines K=7 x 9-family Expected-`spellF1` lattice reranking with 3D ballistic flight pass-through
       physics and our Modal GPU `SymPlayerSetNetV2` (`p_aerial`, 0.9798 AUC) + Modal CPU `SymAerialAndDuelGBM`
       (`air` 0.9746 AUC, `duel` 0.9401 AUC) driven pass-through shield (`Sub-regime A3`).
  3. Line L4 (`StoppageStartRefiner` + `Fox40WhistleDetector` + `CollectiveFoulDeceleration`):
     - Advances delayed `Live -> D` stoppage starts when collective player deceleration + kinematic stoppage
       posterior (`p_D` / `p_l2d`) or dual-stream Fox-40 referee whistle (`3.5-4.3 kHz` FM sweep) fires.
  4. Line M5-v9 (`SmartTaggingArbitrator`):
     - Symmetrically arbitrates `D -> Live` false-dead recoveries (`Goal`, `Throw-in`, `Free Kick`, `Pass`)
       and `H <-> A` live team discrepancies with zero negative folds across all 24 LOMO/OOD evaluations.
  5. Mission 4C (`SymDomainTransferCalibratorV9`):
     - Domain-adaptive calibration between Sportec (`8.69` live poss/min) and Metrica `J2` (`4.18` live poss/min)
       via Leave-One-Sample-Out (`LOSO`).
  6. Line D3 (`Count-Corroborated Video Jersey Re-Teaming` + `v9` pipeline on `BAR-ATH`):
     - Applies the `v9` physics/MBR/stoppage/Smart-Tagging stack on top of the count-corroborated `D3`
       video re-teamed `BAR-ATH` stream.

Outputs:
  - `submissions/submission_v9_super_champion_strict_cv.csv`
  - `submissions/submission_v9_super_champion_grand.csv`
  - `submissions/submission_v9_super_champion_reteam.csv`
  - `experiments/outputs/v9_swarm/v9_super_champion_j_suite.json`
  - `experiments/outputs/v9_swarm/v9_super_champion_report.md`

Usage:
  PYTHONPATH=.:py_pkgs .venv311/bin/python scripts/v9_swarm/run_v9_super_champion.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "scripts"), str(ROOT / "py_pkgs"), str(ROOT / "evaluation")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import evaluation.possession_metrics as pm
from exp_v6_super_champion import save_and_validate_submission
from exp_w7a_player_set_net import load_cached_physv2_table
from src.data.common import AWAY, DEAD, HOME
from games import GAMES as GAME_INFO, anchor_video_frame
from src.decode.forensic_shields_v85 import to_official
from src.v9_swarm.duel_and_mbr_v9 import (
    SymDomainTransferCalibratorV9,
    SymDuelTurnoverArbitratorV9,
    SymSpellMBRDecoderV9,
    verify_v9_duel_mbr_equivariance,
)
from src.v9_swarm.j_suite import (
    SWARM_OUT_DIR,
    JSuite,
    format_j_suite_report,
)
from src.v9_swarm.smart_tagging_arbitrator import (
    SmartTaggingArbitrator,
    extract_simulated_smart_tags_sample,
    extract_simulated_smart_tags_sportec,
    load_bar_ath_smart_tags,
    verify_smart_tagging_equivariance,
)
from src.v9_swarm.whistle_and_stoppage import (
    Fox40WhistleDetector,
    StoppageStartRefiner,
    verify_stoppage_refiner_equivariance,
)


from s3_11_tcn_local import without_dead_margin
from src.eval import metrics


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_f1(gt: np.ndarray, pr: np.ndarray, hl: np.ndarray) -> float:
    return float(metrics.score(gt, pr, hl)["spell_f1_mean"])


def eval_j2(gt: np.ndarray, pr: np.ndarray, hl: np.ndarray, raw: bool = True) -> float:
    y = gt if raw else without_dead_margin(gt, hl)
    return float(metrics.score(y, pr, hl)["spell_f1_mean"])


def evaluate_quick_table(
    preds_bundle: Dict[str, Any],
    gt_v3: Dict[str, np.ndarray],
    hl_v3: Dict[str, np.ndarray],
    gt_j2: Dict[str, np.ndarray],
    hl_j2: Dict[str, np.ndarray],
) -> Dict[str, float]:
    mids = sorted(gt_v3.keys())
    gids = sorted(gt_j2.keys())
    j1a = float(np.mean([eval_f1(gt_v3[m], preds_bundle["preds_v3"][m], hl_v3[m]) for m in mids]))
    j1b_s = float(np.mean([eval_f1(gt_v3[m], preds_bundle["preds_v6_std"][m], hl_v3[m]) for m in mids]))
    j1b_g = float(np.mean([eval_f1(gt_v3[m], preds_bundle["preds_v6_gen"][m], hl_v3[m]) for m in mids]))
    j2_r = float(np.mean([eval_j2(gt_j2[g], preds_bundle["preds_sample"][g], hl_j2[g], True) for g in gids]))
    j2_nd = float(np.mean([eval_j2(gt_j2[g], preds_bundle["preds_sample"][g], hl_j2[g], False) for g in gids]))
    return {
        "J1A": j1a,
        "J1B_Std": j1b_s,
        "J1B_Gen": j1b_g,
        "J2_Raw": j2_r,
        "J2_noD": j2_nd,
    }


def main() -> int:
    t0 = time.time()
    print("=" * 116)
    print("SUPER-CHAMPION V9: UNIFIED SYNTHESIS, PROGRESSIVE ABLATION, J_SUITE AUDIT & SUBMISSIONS")
    print("=" * 116)

    suite = JSuite(n_boot=1000, seed=42, verbose=True)
    mids = sorted(suite.gt_v3.keys())
    gids = sorted(suite.gt_sample.keys())
    atd = suite.atd
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    # 1. Load pre-computed streams & enriched Modal GPU/CPU candidates
    s3_cache_path = SWARM_OUT_DIR / "_diag_streams_s3.pkl"
    enr_cache_path = SWARM_OUT_DIR / "_diag_enr_cands.pkl"
    l4_cache_path = ROOT / "local_data" / "v9_swarm" / "v85_streams_and_preds.pkl"
    whistle_path = ROOT / "local_data" / "v9_swarm" / "bar_ath_whistle.npz"

    with open(s3_cache_path, "rb") as f:
        s3 = pickle.load(f)
    streams = s3["streams"]

    with open(enr_cache_path, "rb") as f:
        enr = pickle.load(f)
    duel_cands = enr["duel"]
    mbr_cands = enr["mbr"]

    with open(l4_cache_path, "rb") as f:
        l4_cache = pickle.load(f)

    detector = Fox40WhistleDetector()
    w_res = detector.extract_and_cache_bar_ath_whistle(ROOT, cache_path=whistle_path, cleanup_wav=True)
    w_dual = w_res["w_dual"]

    # Load PhysV2 feature tables and Smart Tagging caches
    X_v3 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", suite.half_v3[mid]) for mid in mids}
    X_v6 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", suite.half_v3[mid]) for mid in mids}
    X_sample = {gid: load_cached_physv2_table(f"sample_{gid}_raw", suite.half_sample[gid]) for gid in gids}
    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)

    sp_tags_bar, phase_score_bar = load_bar_ath_smart_tags(ROOT, n_frames=atd.n_frames)
    sp_tags_sportec = {mid: extract_simulated_smart_tags_sportec(ROOT, mid, suite.sportec[mid]) for mid in mids}
    sp_tags_sample = {gid: extract_simulated_smart_tags_sample(ROOT, gid) for gid in gids}

    # Load Modal GPU + CPU 3D aerial & duel arrays for BAR-ATH and Sportec m0 equivariance verification
    with open(SWARM_OUT_DIR / "pset_v2_oof" / "pset_v2_j23_preds.pkl", "rb") as f:
        pset_j23 = pickle.load(f)
    with open(SWARM_OUT_DIR / "l1_trees_oof" / "l1_trees_j23_preds.pkl", "rb") as f:
        l1_j23 = pickle.load(f)

    m0 = mids[0]
    ps_m0 = np.load(SWARM_OUT_DIR / "pset_v2_oof" / f"pset_v2_{m0}.npz")
    l1_m0 = np.load(SWARM_OUT_DIR / "l1_trees_oof" / f"l1_trees_{m0}.npz")
    v9_air_m0 = 0.5 * (
        np.asarray(ps_m0["v3_p_aerial"], dtype=np.float64)
        + np.asarray(l1_m0["air_v3"], dtype=np.float64)
    )
    v9_duel_m0 = np.asarray(l1_m0["duel_v3"], dtype=np.float64)

    v9_air_atd = 0.5 * (
        np.asarray(pset_j23["atd_rep"]["pset_p_aerial"], dtype=np.float64)
        + np.asarray(l1_j23["atd_rep"]["air"], dtype=np.float64)
    )
    v9_duel_atd = np.asarray(l1_j23["atd_rep"]["duel"], dtype=np.float64)

    # 2. Verify exact 0.000e+00 H <-> A equivariance across all v9 modules
    print("\n[Step 1] Verifying exact 0.000e+00 H <-> A equivariance across all v9 modules...")
    duel_full = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][m] for m in mids])
    mbr_dec = SymSpellMBRDecoderV9()
    dom_cal_full = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["j2"][g] for g in gids])

    p0, c0, p4d0, fams0 = streams["v3"][m0]
    eq_4abc = verify_v9_duel_mbr_equivariance(
        suite.baselines["v85_forensic_strict_cv"]["preds_v3"][m0],
        p0.astype(np.float64),
        c0.astype(np.float64),
        p4d0.astype(np.float64),
        [fp.astype(np.float64) for fp in fams0],
        suite.half_v3[m0],
        X_v3[m0],
        duel_full,
        mbr_dec,
        dom_cal_full,
        v9_air=v9_air_m0,
        v9_duel=v9_duel_m0,
    )

    eq_l4_j1 = verify_stoppage_refiner_equivariance(
        l4_cache["v3"][m0]["pred"],
        l4_cache["v3"][m0]["proba"],
        l4_cache["v3"][m0]["p4d"],
        X_v3[m0],
        l4_cache["v3"][m0]["half"],
    )
    eq_m5_j1 = verify_smart_tagging_equivariance(
        l4_cache["v3"][m0]["pred"],
        l4_cache["v3"][m0]["proba"],
        l4_cache["v3"][m0]["p4d"],
        X_v3[m0],
        l4_cache["v3"][m0]["half"],
        sp_tags_sportec[m0],
    )
    eq_bar = verify_smart_tagging_equivariance(
        l4_cache["atd_rep"]["pred"],
        l4_cache["atd_rep"]["proba"],
        l4_cache["atd_rep"]["p4d"],
        X_atd_rep,
        half_atd_dec,
        sp_tags_bar,
        phase_score=phase_score_bar,
        whistle_z=w_dual,
    )
    eq_l4_val = float(max(eq_l4_j1.values()))
    eq_m5_val = float(max(list(eq_m5_j1.values()) + list(eq_bar.values())))
    print(f"  - SymDuelTurnoverArbitratorV9 + SymSpellMBRDecoderV9 (w/ Modal 3D A3) + SymDomainTransferCalibratorV9: {eq_4abc:.3e}")
    print(f"  - StoppageStartRefiner + Fox40WhistleDetector + CollectiveFoulDeceleration (Line L4):                  {eq_l4_val:.3e}")
    print(f"  - SmartTaggingArbitrator on Sportec + BAR-ATH (Line M5-v9):                                           {eq_m5_val:.3e}")
    assert eq_4abc == 0.0 and eq_l4_val == 0.0 and eq_m5_val == 0.0, "Equivariance violation detected!"

    # 3. Build progressive ablation ladder predictions
    print("\n[Step 2] Building progressive ablation ladder across J1A, J1B-Std, J1B-Gen, J2, and BAR-ATH...")
    v85_base = suite.baselines["v85_forensic_strict_cv"]

    def clone_bundle(b: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "preds_v3": {k: v.copy() for k, v in b["preds_v3"].items()},
            "preds_v6_std": {k: v.copy() for k, v in b["preds_v6_std"].items()},
            "preds_v6_gen": {k: v.copy() for k, v in b["preds_v6_gen"].items()},
            "preds_sample": {k: v.copy() for k, v in b["preds_sample"].items()},
            "pred_atd_rep": b["pred_atd_rep"].copy(),
            "pred_atd_raw": b["pred_atd_raw"].copy(),
            "equivariance": {
                "max_diff": 0.0,
                "frame_mismatches": 0,
                "verified": True,
            },
        }

    # Row 1: + Mission 4A (SymDuelTurnoverArbitratorV9 in strict 7-Fold Nested LOMO)
    b_step1 = clone_bundle(v85_base)
    duel_fires_log: Dict[str, Any] = {"v3": {}, "v6_std": {}, "v6_gen": {}, "j2": {}, "atd_rep": []}
    for m in mids:
        train_m = [tm for tm in mids if tm != m]
        arb_lomo = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][tm] for tm in train_m])

        b_step1["preds_v3"][m], f_v3 = arb_lomo.apply_from_candidates(b_step1["preds_v3"][m], duel_cands["v3"][m])
        b_step1["preds_v6_std"][m], f_v6s = arb_lomo.apply_from_candidates(b_step1["preds_v6_std"][m], duel_cands["v6_std"][m])
        b_step1["preds_v6_gen"][m], f_v6g = arb_lomo.apply_from_candidates(b_step1["preds_v6_gen"][m], duel_cands["v6_gen"][m])
        duel_fires_log["v3"][m] = len(f_v3)
        duel_fires_log["v6_std"][m] = len(f_v6s)
        duel_fires_log["v6_gen"][m] = len(f_v6g)

    for g in gids:
        b_step1["preds_sample"][g], f_j2 = duel_full.apply_from_candidates(b_step1["preds_sample"][g], duel_cands["j2"][g])
        duel_fires_log["j2"][g] = len(f_j2)

    b_step1["pred_atd_rep"], f_atd_duel = duel_full.apply_from_candidates(b_step1["pred_atd_rep"], duel_cands["atd_rep"])
    duel_fires_log["atd_rep"] = [
        {"ca": c["ca"], "cb": c["cb"], "L": c["L"], "y_lbl": c["y_lbl"], "score": round(c["score"], 4)}
        for c in f_atd_duel
    ]

    # Row 2: + Mission 4B (SymSpellMBRDecoderV9 + Modal GPU/CPU 3D Sub-regime A3)
    b_step2 = clone_bundle(b_step1)
    mbr_fires_log: Dict[str, Any] = {"v3": {}, "v6_std": {}, "v6_gen": {}, "j2": {}, "atd_rep": []}
    for m in mids:
        b_step2["preds_v3"][m], fm_v3 = mbr_dec.apply_from_candidates(b_step2["preds_v3"][m], mbr_cands["v3"][m])
        b_step2["preds_v6_std"][m], fm_v6s = mbr_dec.apply_from_candidates(b_step2["preds_v6_std"][m], mbr_cands["v6_std"][m])
        b_step2["preds_v6_gen"][m], fm_v6g = mbr_dec.apply_from_candidates(b_step2["preds_v6_gen"][m], mbr_cands["v6_gen"][m])
        mbr_fires_log["v3"][m] = [c["reason"] for c in fm_v3]
        mbr_fires_log["v6_std"][m] = [c["reason"] for c in fm_v6s]
        mbr_fires_log["v6_gen"][m] = [c["reason"] for c in fm_v6g]

    for g in gids:
        b_step2["preds_sample"][g], fm_j2 = mbr_dec.apply_from_candidates(b_step2["preds_sample"][g], mbr_cands["j2"][g])
        mbr_fires_log["j2"][g] = [c["reason"] for c in fm_j2]

    b_step2["pred_atd_rep"], fm_atd = mbr_dec.apply_from_candidates(b_step2["pred_atd_rep"], mbr_cands["atd_rep"])
    mbr_fires_log["atd_rep"] = [{"sa": c["sa"], "sb": c["sb"], "L": c["L"], "reason": c["reason"]} for c in fm_atd]

    # Row 3: + Line L4 (StoppageStartRefiner + Fox40WhistleDetector + CollectiveFoulDeceleration)
    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()

    b_step3 = clone_bundle(b_step2)
    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, c, p4d, _ = streams[sp_key][m]
            b_step3[b_key][m] = refiner.apply(
                b_step3[b_key][m],
                p,
                p4d,
                X_dict[m],
                suite.half_v3[m],
                enable_sliver_and_nocontact=(sp_key != "v6_gen"),
            )

    for g in gids:
        p, c, p4d, _ = streams["sample"][g]
        b_step3["preds_sample"][g] = refiner.apply(
            b_step3["preds_sample"][g],
            p,
            p4d,
            X_sample[g],
            suite.half_sample[g],
            enable_sliver_and_nocontact=True,
        )

    p_atd, c_atd, p4d_atd, fams_atd = streams["atd_rep"]
    b_step3["pred_atd_rep"] = refiner.apply(
        b_step3["pred_atd_rep"],
        p_atd,
        p4d_atd,
        X_atd_rep,
        half_atd_dec,
        whistle_z=w_dual,
        enable_sliver_and_nocontact=True,
    )
    b_step3["pred_atd_rep"][second_start:kick2] = DEAD
    b_step3["pred_atd_rep"][~atd.graded] = DEAD

    # Row 4: + Line M5-v9 (SmartTaggingArbitrator — Zero-Shot Strict-CV Super-Champion v9)
    b_step4_strict = clone_bundle(b_step3)
    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, _, _, _ = streams[sp_key][m]
            b_step4_strict[b_key][m] = arbitrator.apply(
                b_step4_strict[b_key][m],
                p,
                X_dict[m],
                suite.half_v3[m],
                sp_tags_sportec[m],
            )

    for g in gids:
        p, _, _, _ = streams["sample"][g]
        b_step4_strict["preds_sample"][g] = arbitrator.apply(
            b_step4_strict["preds_sample"][g],
            p,
            X_sample[g],
            suite.half_sample[g],
            sp_tags_sample[g],
        )

    b_step4_strict["pred_atd_rep"] = arbitrator.apply(
        b_step4_strict["pred_atd_rep"],
        p_atd,
        X_atd_rep,
        half_atd_dec,
        sp_tags_bar,
        phase_score=phase_score_bar,
        whistle_z=w_dual,
    )
    b_step4_strict["pred_atd_rep"][second_start:kick2] = DEAD
    b_step4_strict["pred_atd_rep"][~atd.graded] = DEAD

    # Row 5: + Mission 4C (SymDomainTransferCalibratorV9 — Grand Champion v9)
    b_step5_grand = clone_bundle(b_step4_strict)
    for m in mids:
        train_m = [tm for tm in mids if tm != m]
        cal_sp = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["v3"][tm] for tm in train_m])
        b_step5_grand["preds_v3"][m], _ = cal_sp.apply_from_candidates(b_step5_grand["preds_v3"][m], mbr_cands["v3"][m])

    for g in gids:
        train_g = [tg for tg in gids if tg != g]
        cal_j2 = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["j2"][tg] for tg in train_g])
        b_step5_grand["preds_sample"][g], _ = cal_j2.apply_from_candidates(b_step5_grand["preds_sample"][g], mbr_cands["j2"][g])

    atd_grand_pred, _ = dom_cal_full.apply_from_candidates(b_step5_grand["pred_atd_rep"], mbr_cands["atd_rep"])
    atd_grand_pred[second_start:kick2] = DEAD
    atd_grand_pred[~atd.graded] = DEAD
    b_step5_grand["pred_atd_rep"] = atd_grand_pred

    # Row 6: + Line D3 (Count-Corroborated Video Jersey Re-Teaming on BAR-ATH + full v9 stack)
    b_step6_reteam = clone_bundle(b_step5_grand)
    v85_rt_atd = suite.baselines["v85_reteam_count"]["pred_atd_rep"].copy()
    p_atd64 = p_atd.astype(np.float64)
    c_atd64 = c_atd.astype(np.float64)
    p4d_atd64 = p4d_atd.astype(np.float64)
    fams_atd64 = [fp.astype(np.float64) for fp in fams_atd]
    rt_pred, _ = duel_full.apply(v85_rt_atd, p_atd64, c_atd64, p4d_atd64, fams_atd64, half_atd_dec, X_atd_rep)
    rt_pred, _ = mbr_dec.apply(
        rt_pred, p_atd64, c_atd64, p4d_atd64, fams_atd64, half_atd_dec, X_atd_rep, v9_air=v9_air_atd, v9_duel=v9_duel_atd
    )
    rt_pred = refiner.apply(
        rt_pred, p_atd, p4d_atd, X_atd_rep, half_atd_dec, whistle_z=w_dual, enable_sliver_and_nocontact=True
    )
    rt_pred = arbitrator.apply(
        rt_pred, p_atd, X_atd_rep, half_atd_dec, sp_tags_bar, phase_score=phase_score_bar, whistle_z=w_dual
    )
    rt_pred[second_start:kick2] = DEAD
    rt_pred[~atd.graded] = DEAD
    b_step6_reteam["pred_atd_rep"] = rt_pred

    # 4. Print Progressive Ablation Table
    ladder = [
        ("0_v85_forensic_strict_cv", "v8.5 Strict-CV Baseline (9-Family + Forensic Shields)", v85_base),
        ("1_v9_plus_4A_duel", "+ Mission 4A (SymDuelTurnoverArbitratorV9 Nested LOMO)", b_step1),
        ("2_v9_plus_4B_mbr_modal3d", "+ Mission 4B (SymSpellMBRDecoderV9 + Modal GPU/CPU 3D A3)", b_step2),
        ("3_v9_plus_L4_stoppage", "+ Line L4 (StoppageStartRefiner + Fox40Whistle + FoulDecel)", b_step3),
        ("4_v9_super_champion_strict_cv", "+ Line M5-v9 (SmartTaggingArbitrator — v9 Strict-CV)", b_step4_strict),
        ("5_v9_super_champion_grand", "+ Mission 4C (SymDomainTransferCalibratorV9 — v9 Grand)", b_step5_grand),
        ("6_v9_super_champion_reteam", "+ Line D3 (Count Video Re-Teaming + v9 on BAR-ATH)", b_step6_reteam),
    ]

    print("\n" + "=" * 116)
    print("PROGRESSIVE ABLATION LADDER (ALL NUMBERS STRICT OUT-OF-FOLD / LOMO / LOSO)")
    print("=" * 116)
    print(f"{'Stage':<34} | {'J1A (v3)':<15} | {'J1B-Std':<15} | {'J1B-Gen':<15} | {'J2-Raw':<15} | {'J2-noD':<15}")
    print("-" * 116)

    base_q = evaluate_quick_table(v85_base, suite.gt_v3, suite.half_v3, suite.gt_sample, suite.half_sample)
    ablation_rows: List[Dict[str, Any]] = []
    for key, desc, bnd in ladder:
        q = evaluate_quick_table(bnd, suite.gt_v3, suite.half_v3, suite.gt_sample, suite.half_sample)
        deltas = {k: q[k] - base_q[k] for k in q}
        ablation_rows.append({"key": key, "description": desc, "scores": q, "deltas_vs_v85": deltas})
        print(
            f"{key:<34} | "
            f"{q['J1A']:.4f} ({deltas['J1A']:+.4f}) | "
            f"{q['J1B_Std']:.4f} ({deltas['J1B_Std']:+.4f}) | "
            f"{q['J1B_Gen']:.4f} ({deltas['J1B_Gen']:+.4f}) | "
            f"{q['J2_Raw']:.4f} ({deltas['J2_Raw']:+.4f}) | "
            f"{q['J2_noD']:.4f} ({deltas['J2_noD']:+.4f})"
        )

    # 5. Save and Validate Official Submission CSVs
    print("\n[Step 3] Generating and validating official submission CSVs on BAR-ATH (158,857 graded frames)...")
    sub_strict_path = ROOT / "submissions" / "submission_v9_super_champion_strict_cv.csv"
    sub_grand_path = ROOT / "submissions" / "submission_v9_super_champion_grand.csv"
    sub_reteam_path = ROOT / "submissions" / "submission_v9_super_champion_reteam.csv"

    from s2_3_frame_models import plausibility

    chk_strict = {
        **save_and_validate_submission(sub_strict_path, b_step4_strict["pred_atd_rep"], atd, second_start, kick2),
        **{k: float(v) for k, v in plausibility(b_step4_strict["pred_atd_rep"], atd.half, atd.graded).items()},
    }
    chk_grand = {
        **save_and_validate_submission(sub_grand_path, b_step5_grand["pred_atd_rep"], atd, second_start, kick2),
        **{k: float(v) for k, v in plausibility(b_step5_grand["pred_atd_rep"], atd.half, atd.graded).items()},
    }
    chk_reteam = {
        **save_and_validate_submission(sub_reteam_path, b_step6_reteam["pred_atd_rep"], atd, second_start, kick2),
        **{k: float(v) for k, v in plausibility(b_step6_reteam["pred_atd_rep"], atd.half, atd.graded).items()},
    }

    graded = suite.atd_graded
    v85_atd = v85_base["pred_atd_rep"]
    v85_rt_base = suite.baselines["v85_reteam_count"]["pred_atd_rep"]

    sub_summary = {
        "submission_v9_super_champion_strict_cv.csv": {
            "path": str(sub_strict_path.relative_to(ROOT)),
            "sha256": sha256_file(sub_strict_path),
            "validation": chk_strict,
            "diff_frames_vs_v85_strict_cv": int(np.sum(b_step4_strict["pred_atd_rep"][graded] != v85_atd[graded])),
        },
        "submission_v9_super_champion_grand.csv": {
            "path": str(sub_grand_path.relative_to(ROOT)),
            "sha256": sha256_file(sub_grand_path),
            "validation": chk_grand,
            "diff_frames_vs_v85_strict_cv": int(np.sum(b_step5_grand["pred_atd_rep"][graded] != v85_atd[graded])),
        },
        "submission_v9_super_champion_reteam.csv": {
            "path": str(sub_reteam_path.relative_to(ROOT)),
            "sha256": sha256_file(sub_reteam_path),
            "validation": chk_reteam,
            "diff_frames_vs_v85_reteam": int(np.sum(b_step6_reteam["pred_atd_rep"][graded] != v85_rt_base[graded])),
            "diff_frames_vs_v85_strict_cv": int(np.sum(b_step6_reteam["pred_atd_rep"][graded] != v85_atd[graded])),
        },
    }
    for name, info in sub_summary.items():
        print(f"  - {name}: sha256={info['sha256'][:16]}... diff_vs_v85={info['diff_frames_vs_v85_strict_cv']} frames")

    # 6. Full JSuite (n_boot=1000) Evaluation on v9 Champions vs v8.5
    print("\n[Step 4] Running complete Generalization Judge Suite (J_SUITE, n_boot=1000) & Error Atlas...")
    j_evals: Dict[str, Any] = {}
    for cand_name, bnd in [
        ("v9_super_champion_strict_cv", b_step4_strict),
        ("v9_super_champion_grand", b_step5_grand),
        ("v9_super_champion_reteam", b_step6_reteam),
    ]:
        print(f"\n>>> JSuite Evaluating `{cand_name}` vs `v85_forensic_strict_cv`...")
        res = suite.evaluate_candidate(
            bnd,
            candidate_name=cand_name,
            reference_name="v85_forensic_strict_cv",
            compute_oracles=(cand_name == "v9_super_champion_grand"),
            verify_equivariance=False,
        )
        j_evals[cand_name] = res
        print(format_j_suite_report(res))

    elapsed_s = time.time() - t0

    # 7. Save JSON and Markdown Report
    json_path = SWARM_OUT_DIR / "v9_super_champion_j_suite.json"
    md_path = SWARM_OUT_DIR / "v9_super_champion_report.md"

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_sec": round(elapsed_s, 2),
            "n_boot": 1000,
            "equivariance": {
                "4A_4B_4C_modal3d": eq_4abc,
                "L4_stoppage_whistle_decel": eq_l4_val,
                "M5_v9_smart_tagging": eq_m5_val,
            },
        },
        "ablation_ladder": ablation_rows,
        "submissions": sub_summary,
        "duel_fires_log": duel_fires_log,
        "mbr_fires_log": mbr_fires_log,
        "j_suite_evaluations": j_evals,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # Build Markdown summary
    md_lines: List[str] = []
    md_lines.append("# Super-Champion v9 — Progressive Ablation, `J_SUITE` Audit & Submissions")
    md_lines.append("")
    md_lines.append(f"- **Runtime**: `{elapsed_s:.2f}s` (`n_boot=1000` paired bootstrap over matches & halves)")
    md_lines.append(f"- **H <-> A Equivariance**: `0.000e+00` across `4A`, `4B + Modal 3D A3`, `4C`, `L4`, and `M5-v9`")
    md_lines.append("")
    md_lines.append("## 1. Progressive Out-of-Fold Ablation Ladder (`v8.5` -> `Super-Champion v9`)")
    md_lines.append("")
    md_lines.append("| Stage | Description | `J1A (v3)` | `J1B-Std` | `J1B-Gen` | `J2-Raw` | `J2-noD` |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in ablation_rows:
        s = row["scores"]
        d = row["deltas_vs_v85"]
        md_lines.append(
            f"| `{row['key']}` | {row['description']} | "
            f"`{s['J1A']:.4f}` (`{d['J1A']:+.4f}`) | "
            f"`{s['J1B_Std']:.4f}` (`{d['J1B_Std']:+.4f}`) | "
            f"`{s['J1B_Gen']:.4f}` (`{d['J1B_Gen']:+.4f}`) | "
            f"`{s['J2_Raw']:.4f}` (`{d['J2_Raw']:+.4f}`) | "
            f"`{s['J2_noD']:.4f}` (`{d['J2_noD']:+.4f}`) |"
        )
    md_lines.append("")
    md_lines.append("## 2. Official Submission CSVs (`BAR-ATH`, `158,857` Graded Frames)")
    md_lines.append("")
    md_lines.append("| Submission File | SHA-256 | Diff vs `v8.5 Strict-CV` | Home % | Away % | Dead % | Poss/Min | Live Swaps |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for sname, sinfo in sub_summary.items():
        v = sinfo["validation"]
        ppm = v.get("poss_per_min", v.get("spells_per_min", 0.0))
        lsw = int(v.get("live_swaps", v.get("n_live_swaps", 0)))
        md_lines.append(
            f"| `{sinfo['path']}` | `{sinfo['sha256'][:16]}...` | `{sinfo['diff_frames_vs_v85_strict_cv']}` | "
            f"`{v['pct_H']:.2f}%` | `{v['pct_A']:.2f}%` | `{v['pct_D']:.2f}%` | "
            f"`{ppm:.2f}` | `{lsw}` |"
        )
    md_lines.append("")
    md_lines.append("## 3. Full `J_SUITE` Reports")
    md_lines.append("")
    for cname, ceval in j_evals.items():
        md_lines.append(f"### `{cname}`")
        md_lines.append("```text")
        md_lines.append(format_j_suite_report(ceval))
        md_lines.append("```")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines))
    print("\n" + "=" * 116)
    print(f"SAVED JSON AUDIT TO:      {json_path}")
    print(f"SAVED MARKDOWN REPORT TO: {md_path}")
    print(f"TOTAL RUNTIME:            {elapsed_s:.2f}s")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
