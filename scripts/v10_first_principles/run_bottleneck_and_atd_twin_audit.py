#!/usr/bin/env python3
"""First-Principles Bottleneck & Oracle Breakdown + J_ATD_TWIN Stress-Test Suite.

Executes Tasks 2 & 3 of the Subagent A Forensic Audit:
  1. Exact breakdown of missed ground-truth possessions by duration (`< 1.0s`, `1.0-1.5s`,
     `1.5-2.0s`, `2.0-5.0s`, `> 5.0s`, plus `< 2.5s`, `< 2.0s`, `< 1.36s`) and by failure mode
     (`Swallowed_Double_Domino_Steal`, `Collateral_Double_Domino_Victim`, `Swallowed_Opponent_NonDomino`,
     `Swallowed_By_Dead`, `Boundary_Shifted_0_to_03`, `Boundary_Shifted_03_to_07`, `B4_false_blips`,
     `B5_dead_wedges`) on `v8.5` and `v9` across `J1A` (`v3`), `J1B` (`v6_std`, `v6_gen`), and `J2`.
  2. Swallowed short-possession (`< 2.5s`) probability & evidence audit on `J1A`, comparing `v8.5`'s
     9-family probabilities (`_diag_streams_s3.pkl`) against Modal GPU `SymPlayerSetNetV2`
     (`pset_v2_oof/`) and Modal CPU `L1-Trees` (`l1_trees_oof/`).
  3. Exact `spellF1` oracle ceiling of each bottleneck (`B1_double_domino`, `B1_other_short`,
     `B1_all_short`, `B2_team_boundaries`, `B3_dead_boundaries`, `B4_false_blips`, `B5_dead_wedges`)
     on `v8.5` and `v9`.
  4. `J_ATD_TWIN` realistic target-domain stress-test suite on `J1A` (`Sportec`):
     - `J1_JerseySwap` (`0.21` and `0.31` players/frame swapped for `3.5s` stretches starting `>= 8m`
       from ball, plus unshielded `>= 0m` near-ball swap sensitivity & `D3` count-corroborated recovery).
     - `J1_UnbiasedDead` (neutralizing the synthetic `d <= 1.24m` dead-ball player-attachment shortcut
       during `D` transitions).
     - `J1_RollingFoul` (simulating in-pitch foul `D` starts with `1.5s-3.0s` rolling-ball lag).

Outputs:
  - `experiments/outputs/v10_first_principles/bottleneck_and_atd_twin_audit.json`

Usage:
  PYTHONPATH=.:py_pkgs .venv311/bin/python scripts/v10_first_principles/run_bottleneck_and_atd_twin_audit.py
"""

from __future__ import annotations

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

from exp_w7a_player_set_net import load_cached_physv2_table
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, HOME
from src.eval import metrics
from src.v9_swarm.duel_and_mbr_v9 import (
    SymDomainTransferCalibratorV9,
    SymDuelTurnoverArbitratorV9,
    SymSpellMBRDecoderV9,
)
from src.v9_swarm.j_suite import SWARM_OUT_DIR, JSuite
from src.v9_swarm.smart_tagging_arbitrator import (
    SmartTaggingArbitrator,
    extract_simulated_smart_tags_sample,
    extract_simulated_smart_tags_sportec,
)
from src.v9_swarm.whistle_and_stoppage import StoppageStartRefiner
from src.v10_first_principles.atd_twin_and_bottlenecks import (
    DURATION_BUCKETS,
    audit_swallowed_short_probabilities_j1a,
    classify_possessions_and_bottlenecks,
    compute_exact_oracle_ceilings,
    simulate_j1_jersey_swap_table,
    simulate_j1_rolling_foul,
    simulate_j1_unbiased_dead,
)

OUT_DIR = ROOT / "experiments" / "outputs" / "v10_first_principles"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def eval_suite_metrics(
    gt_dict: Dict[str, np.ndarray],
    pr_dict: Dict[str, np.ndarray],
    hl_dict: Dict[str, np.ndarray],
) -> Dict[str, float]:
    keys = sorted(gt_dict.keys())
    res = [metrics.score(gt_dict[k], pr_dict[k], hl_dict[k]) for k in keys]
    return {
        "spell_f1_mean": round(float(np.mean([r["spell_f1_mean"] for r in res])), 4),
        "spell_f1_03": round(float(np.mean([r["spell_f1"][0.3]["f1"] for r in res])), 4),
        "spell_f1_05": round(float(np.mean([r["spell_f1"][0.5]["f1"] for r in res])), 4),
        "spell_f1_07": round(float(np.mean([r["spell_f1"][0.7]["f1"] for r in res])), 4),
        "macro_f1": round(float(np.mean([r["macro_f1"] for r in res])), 4),
        "alive_dead_f1": round(float(np.mean([r["alive_dead_f1"] for r in res])), 4),
        "d_f1": round(float(np.mean([r["per_class"]["D"]["f1"] for r in res])), 4),
        "team_acc_gt_alive": round(float(np.mean([r["team_acc_gt_alive"] for r in res])), 4),
        "trans_f1_12f": round(float(np.mean([r["transitions"][12]["f1"] for r in res])), 4),
        "trans_f1_25f": round(float(np.mean([r["transitions"][25]["f1"] for r in res])), 4),
    }


def build_v85_and_v9_bundles(
    suite: JSuite,
    streams: Dict[str, Any],
    duel_cands: Dict[str, Any],
    mbr_cands: Dict[str, Any],
    X_v3: Dict[str, Any],
    X_v6: Dict[str, Any],
    X_sample: Dict[str, Any],
    sp_tags_sportec: Dict[str, Any],
    sp_tags_sample: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Reconstruct exact v8.5 and v9 (`strict_cv` and `grand`) prediction bundles."""
    mids = sorted(suite.gt_v3.keys())
    gids = sorted(suite.gt_sample.keys())
    v85_base = suite.baselines["v85_forensic_strict_cv"]

    def clone_b(b: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "preds_v3": {k: v.copy() for k, v in b["preds_v3"].items()},
            "preds_v6_std": {k: v.copy() for k, v in b["preds_v6_std"].items()},
            "preds_v6_gen": {k: v.copy() for k, v in b["preds_v6_gen"].items()},
            "preds_sample": {k: v.copy() for k, v in b["preds_sample"].items()},
        }

    duel_full = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][m] for m in mids])
    mbr_dec = SymSpellMBRDecoderV9()
    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()

    # Step 1: Mission 4A
    b1 = clone_b(v85_base)
    for m in mids:
        train_m = [tm for tm in mids if tm != m]
        arb_lomo = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][tm] for tm in train_m])
        b1["preds_v3"][m], _ = arb_lomo.apply_from_candidates(b1["preds_v3"][m], duel_cands["v3"][m])
        b1["preds_v6_std"][m], _ = arb_lomo.apply_from_candidates(b1["preds_v6_std"][m], duel_cands["v6_std"][m])
        b1["preds_v6_gen"][m], _ = arb_lomo.apply_from_candidates(b1["preds_v6_gen"][m], duel_cands["v6_gen"][m])
    for g in gids:
        b1["preds_sample"][g], _ = duel_full.apply_from_candidates(b1["preds_sample"][g], duel_cands["j2"][g])

    # Step 2: Mission 4B
    b2 = clone_b(b1)
    for m in mids:
        b2["preds_v3"][m], _ = mbr_dec.apply_from_candidates(b2["preds_v3"][m], mbr_cands["v3"][m])
        b2["preds_v6_std"][m], _ = mbr_dec.apply_from_candidates(b2["preds_v6_std"][m], mbr_cands["v6_std"][m])
        b2["preds_v6_gen"][m], _ = mbr_dec.apply_from_candidates(b2["preds_v6_gen"][m], mbr_cands["v6_gen"][m])
    for g in gids:
        b2["preds_sample"][g], _ = mbr_dec.apply_from_candidates(b2["preds_sample"][g], mbr_cands["j2"][g])

    # Step 3: Line L4
    b3 = clone_b(b2)
    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, _, p4d, _ = streams[sp_key][m]
            b3[b_key][m] = refiner.apply(
                b3[b_key][m],
                p,
                p4d,
                X_dict[m],
                suite.half_v3[m],
                enable_sliver_and_nocontact=(sp_key != "v6_gen"),
            )
    for g in gids:
        p, _, p4d, _ = streams["sample"][g]
        b3["preds_sample"][g] = refiner.apply(
            b3["preds_sample"][g], p, p4d, X_sample[g], suite.half_sample[g], enable_sliver_and_nocontact=True
        )

    # Step 4: Line M5-v9 (v9_super_champion_strict_cv)
    b4_strict = clone_b(b3)
    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, _, _, _ = streams[sp_key][m]
            b4_strict[b_key][m] = arbitrator.apply(
                b4_strict[b_key][m], p, X_dict[m], suite.half_v3[m], sp_tags_sportec[m]
            )
    for g in gids:
        p, _, _, _ = streams["sample"][g]
        b4_strict["preds_sample"][g] = arbitrator.apply(
            b4_strict["preds_sample"][g], p, X_sample[g], suite.half_sample[g], sp_tags_sample[g]
        )

    # Step 5: Mission 4C (v9_super_champion_grand)
    b5_grand = clone_b(b4_strict)
    for m in mids:
        train_m = [tm for tm in mids if tm != m]
        cal_sp = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["v3"][tm] for tm in train_m])
        b5_grand["preds_v3"][m], _ = cal_sp.apply_from_candidates(b5_grand["preds_v3"][m], mbr_cands["v3"][m])
    for g in gids:
        train_g = [tg for tg in gids if tg != g]
        cal_j2 = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["j2"][tg] for tg in train_g])
        b5_grand["preds_sample"][g], _ = cal_j2.apply_from_candidates(
            b5_grand["preds_sample"][g], mbr_cands["j2"][g]
        )

    return {
        "v85_forensic_strict_cv": clone_b(v85_base),
        "v9_super_champion_strict_cv": b4_strict,
        "v9_super_champion_grand": b5_grand,
    }


def print_duration_table(title: str, breakdown: Dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    header = (
        f"{'Bucket':<10} | {'GT':>5} | {'Match>=0.7':>10} | {'Shift0.3-0.7':>12} "
        f"(Team/Dead) | {'Miss<0.3':>8} | {'DominoSteal':>11} | {'DominoVict':>10} | "
        f"{'NonDomOpp':>9} | {'SwalDead':>8} | {'Shift<0.3':>9}"
    )
    print(header)
    print("-" * len(header))
    by_b = breakdown["by_duration_bucket"]
    for bname in [b[0] for b in DURATION_BUCKETS] + ["<1.36s", "<2.0s", "<2.5s", "ALL"]:
        r = by_b[bname]
        sh_37 = r["boundary_shifted_05_to_07"] + r["boundary_shifted_03_to_05"]
        td_str = f"({r['boundary_shifted_03_to_07_team']:>3}/{r['boundary_shifted_03_to_07_dead']:>3})"
        print(
            f"{bname:<10} | {r['n_gt']:>5} | {r['matched_iou_ge_07']:>10} | "
            f"{sh_37:>4} {td_str:<7} | {r['missed_iou_lt_03_total']:>8} | "
            f"{r['swallowed_double_domino_steal']:>11} | {r['collateral_double_domino_victim']:>10} | "
            f"{r['swallowed_opponent_nondomino']:>9} | {r['swallowed_by_dead']:>8} | "
            f"{r['boundary_shifted_0_to_03']:>9}"
        )
    fp = breakdown["false_positives"]
    print(
        f"False Positives (IoU < 0.3): Total Pred={fp['n_pred_total']}, Matched@0.3={fp['matched_iou_ge_03']}, "
        f"FP@0.3={fp['fp_iou_lt_03_total']} [B4_false_blips_live={fp['B4_false_blips_live']}, "
        f"B5_dead_wedges={fp['B5_dead_wedges']}, boundary/split={fp['fp_boundary_or_split']}]"
    )


def main() -> int:
    t0 = time.time()
    print("=" * 120)
    print("SUBAGENT A: FIRST-PRINCIPLES BOTTLENECK & ORACLE BREAKDOWN + J_ATD_TWIN STRESS-TEST SUITE")
    print("=" * 120)

    suite = JSuite(n_boot=100, seed=42, verbose=False)
    mids = sorted(suite.gt_v3.keys())
    gids = sorted(suite.gt_sample.keys())

    with open(SWARM_OUT_DIR / "_diag_streams_s3.pkl", "rb") as f:
        s3 = pickle.load(f)
    streams = s3["streams"]

    with open(SWARM_OUT_DIR / "_diag_enr_cands.pkl", "rb") as f:
        enr = pickle.load(f)
    duel_cands = enr["duel"]
    mbr_cands = enr["mbr"]

    X_v3 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", suite.half_v3[mid]) for mid in mids}
    X_v6 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", suite.half_v3[mid]) for mid in mids}
    X_sample = {gid: load_cached_physv2_table(f"sample_{gid}_raw", suite.half_sample[gid]) for gid in gids}

    sp_tags_sportec = {mid: extract_simulated_smart_tags_sportec(ROOT, mid, suite.sportec[mid]) for mid in mids}
    sp_tags_sample = {gid: extract_simulated_smart_tags_sample(ROOT, gid) for gid in gids}

    bundles = build_v85_and_v9_bundles(
        suite, streams, duel_cands, mbr_cands, X_v3, X_v6, X_sample, sp_tags_sportec, sp_tags_sample
    )

    gt_j2_nod = {g: without_dead_margin(suite.gt_sample[g], suite.half_sample[g]) for g in gids}

    # ==================================================================================
    # PART 1: EXACT FIRST-PRINCIPLES BOTTLENECK & DURATION BREAKDOWN ON v8.5 AND v9
    # ==================================================================================
    print("\n" + "=" * 120)
    print("[PART 1] FIRST-PRINCIPLES DURATION & FAILURE-MODE BREAKDOWN (J1A v3, J1B v6_std/v6_gen, J2)")
    print("=" * 120)

    bottleneck_breakdowns: Dict[str, Dict[str, Any]] = {}
    for model_key in ["v85_forensic_strict_cv", "v9_super_champion_strict_cv", "v9_super_champion_grand"]:
        bnd = bundles[model_key]
        bottleneck_breakdowns[model_key] = {}
        for judge_name, gt_map, pr_map, hl_map in [
            ("J1A_v3", suite.gt_v3, bnd["preds_v3"], suite.half_v3),
            ("J1B_v6_std", suite.gt_v3, bnd["preds_v6_std"], suite.half_v3),
            ("J1B_v6_gen", suite.gt_v3, bnd["preds_v6_gen"], suite.half_v3),
            ("J2_Raw", suite.gt_sample, bnd["preds_sample"], suite.half_sample),
            ("J2_noD", gt_j2_nod, bnd["preds_sample"], suite.half_sample),
        ]:
            res = classify_possessions_and_bottlenecks(gt_map, pr_map, hl_map)
            bottleneck_breakdowns[model_key][judge_name] = {
                "by_duration_bucket": res["by_duration_bucket"],
                "false_positives": res["false_positives"],
                "n_swallowed_short_lt_2p5s": len(res["swallowed_short_records"]),
                "_records": res["swallowed_short_records"],
            }
            if model_key in ("v85_forensic_strict_cv", "v9_super_champion_grand") and judge_name in (
                "J1A_v3",
                "J1B_v6_std",
                "J2_Raw",
            ):
                print_duration_table(f"{model_key} on {judge_name}", res)

    # ==================================================================================
    # PART 2: PROBABILITY & MODAL GPU/CPU EVIDENCE AUDIT ON SWALLOWED SHORT SPELLS (<2.5s)
    # ==================================================================================
    print("\n" + "=" * 120)
    print("[PART 2] PROBABILITY & MODAL GPU/CPU SIGNAL AUDIT ON SWALLOWED SHORT POSSESSIONS (< 2.5s IN J1A)")
    print("=" * 120)

    pset_dir = SWARM_OUT_DIR / "pset_v2_oof"
    l1_dir = SWARM_OUT_DIR / "l1_trees_oof"

    prob_audit_v85 = audit_swallowed_short_probabilities_j1a(
        bottleneck_breakdowns["v85_forensic_strict_cv"]["J1A_v3"]["_records"],
        streams["v3"],
        pset_dir,
        l1_dir,
    )
    prob_audit_v9 = audit_swallowed_short_probabilities_j1a(
        bottleneck_breakdowns["v9_super_champion_grand"]["J1A_v3"]["_records"],
        streams["v3"],
        pset_dir,
        l1_dir,
    )

    for tag, pa in [("v8.5 Swallowed Short Spells (J1A)", prob_audit_v85), ("v9 Remaining Swallowed Short Spells (J1A)", prob_audit_v9)]:
        print(f"\n--- {tag} ---")
        for skey in [
            "all_swallowed_lt_2p5s",
            "double_domino_lt_2p5s",
            "all_swallowed_lt_2p0s",
            "double_domino_lt_2p0s",
            "all_swallowed_lt_1p36s",
            "double_domino_lt_1p36s",
        ]:
            s = pa[skey]
            n = s["n"]
            print(
                f"  {s['subset']:<36} (n={n:>3}): "
                f"v8.5 max P(Y)>=0.25: {s['v85_max_pY_ge_025']:>3}/{n} ({s['v85_max_pY_ge_025_pct']:>5.1f}%), "
                f"<0.25 ('zero-ev'): {s['v85_max_pY_lt_025']:>3}/{n} ({s['v85_max_pY_lt_025_pct']:>5.1f}%) | "
                f"PSetV2>=0.25: {s['pset_v2_max_pY_ge_025']:>3}, L1-Trees>=0.25: {s['l1_trees_max_pY_ge_025']:>3} | "
                f"PSetV2>v8.5(maxP/meanP/p4d): {s['pset_v2_higher_max_pY_than_v85']}/{s['pset_v2_higher_mean_pY_than_v85']}/{s['pset_v2_higher_p4d_to_Y_than_v85']} | "
                f"L1-Trees>v8.5(maxP/tree4): {s['l1_trees_higher_max_pY_than_v85']}/{s['l1_trees_higher_max_pY_than_v85_tree4']} | "
                f"Rescued from <0.25 by Modal(PSetV2/L1/Any+Duel): "
                f"{s['rescued_from_v85_lt_025_by_pset_v2_pY_ge_025']}/"
                f"{s['rescued_from_v85_lt_025_by_l1_trees_pY_ge_025']}/"
                f"{s['rescued_from_v85_lt_025_by_modal_pY_or_duel_ge_025']}"
            )

    # Strip raw `_records` before JSON serialization
    for mk in bottleneck_breakdowns:
        for jk in bottleneck_breakdowns[mk]:
            bottleneck_breakdowns[mk][jk].pop("_records", None)

    # ==================================================================================
    # PART 3: EXACT SPELL-F1 ORACLE CEILINGS OF EACH BOTTLENECK (B1..B5)
    # ==================================================================================
    print("\n" + "=" * 120)
    print("[PART 3] EXACT SPELL-F1 ORACLE CEILINGS OF EACH BOTTLENECK (B1..B5) ON v8.5 AND v9")
    print("=" * 120)

    oracle_ceilings: Dict[str, Dict[str, Any]] = {}
    for model_key in ["v85_forensic_strict_cv", "v9_super_champion_strict_cv", "v9_super_champion_grand"]:
        bnd = bundles[model_key]
        oracle_ceilings[model_key] = {}
        for judge_name, gt_map, pr_map, hl_map in [
            ("J1A_v3", suite.gt_v3, bnd["preds_v3"], suite.half_v3),
            ("J1B_v6_std", suite.gt_v3, bnd["preds_v6_std"], suite.half_v3),
            ("J1B_v6_gen", suite.gt_v3, bnd["preds_v6_gen"], suite.half_v3),
            ("J2_Raw", suite.gt_sample, bnd["preds_sample"], suite.half_sample),
            ("J2_noD", gt_j2_nod, bnd["preds_sample"], suite.half_sample),
        ]:
            oc = compute_exact_oracle_ceilings(gt_map, pr_map, hl_map)
            oracle_ceilings[model_key][judge_name] = oc

    for model_key in ["v85_forensic_strict_cv", "v9_super_champion_strict_cv", "v9_super_champion_grand"]:
        print(f"\n--- Oracle Ceilings for {model_key} ---")
        print(
            f"{'Judge':<12} | {'Base':>7} | {'+B1_Domino':>10} | {'+B1_Other':>9} | {'+B1_AllShort':>12} | "
            f"{'+B2_TeamBnd':>11} | {'+B3_DeadBnd':>11} | {'+B4_Blips':>9} | {'+B5_Wedges':>10} | {'Cumul_All':>9}"
        )
        print("-" * 120)
        for jn in ["J1A_v3", "J1B_v6_std", "J1B_v6_gen", "J2_Raw", "J2_noD"]:
            oc = oracle_ceilings[model_key][jn]
            mc = oc["marginal_ceilings"]
            cl = oc["cumulative_ladder"]
            print(
                f"{jn:<12} | {oc['base_spell_f1']:>7.4f} | "
                f"{mc['B1_double_domino_le_3p0s']['delta']:>+10.4f} | "
                f"{mc['B1_other_short_le_3p0s']['delta']:>+9.4f} | "
                f"{mc['B1_all_short_le_3p0s']['delta']:>+12.4f} | "
                f"{mc['B2_team_boundaries_pm35f']['delta']:>+11.4f} | "
                f"{mc['B3_dead_boundaries_pm50f']['delta']:>+11.4f} | "
                f"{mc['B4_false_blips_live']['delta']:>+9.4f} | "
                f"{mc['B5_dead_wedges']['delta']:>+10.4f} | "
                f"{cl['6_plus_B5_dead_wedges']:>9.4f}"
            )

    # ==================================================================================
    # PART 4: J_ATD_TWIN REALISTIC TARGET-DOMAIN STRESS-TEST SUITE ON J1A (SPORTEC)
    # ==================================================================================
    print("\n" + "=" * 120)
    print("[PART 4] J_ATD_TWIN REALISTIC TARGET-DOMAIN STRESS-TEST SUITE ON J1A (SPORTEC)")
    print("=" * 120)

    v85_j1a = bundles["v85_forensic_strict_cv"]["preds_v3"]
    v9_strict_j1a = bundles["v9_super_champion_strict_cv"]["preds_v3"]
    v9_grand_j1a = bundles["v9_super_champion_grand"]["preds_v3"]

    clean_v85_metrics = eval_suite_metrics(suite.gt_v3, v85_j1a, suite.half_v3)
    clean_v9_strict_metrics = eval_suite_metrics(suite.gt_v3, v9_strict_j1a, suite.half_v3)
    clean_v9_grand_metrics = eval_suite_metrics(suite.gt_v3, v9_grand_j1a, suite.half_v3)

    # ----------------------------------------------------------------------------------
    # 4.1 J1_JerseySwap (0.21 >=8m, 0.31 >=8m, and 0.21 >=0m near-ball sensitivity)
    # ----------------------------------------------------------------------------------
    print("\n[4.1] Running J1_JerseySwap Stress Test (12v10 wrong-team stretches)...")

    mbr_dec = SymSpellMBRDecoderV9()
    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()

    jersey_swap_results: Dict[str, Any] = {}
    for variant_name, rate_val, min_dist_m in [
        ("swap_021_ge8m_bar_ath_twin", 0.21, 8.0),
        ("swap_031_ge8m_total_color", 0.31, 8.0),
        ("swap_021_ge0m_unshielded_near_ball", 0.21, 0.0),
    ]:
        X_sw_map: Dict[str, Any] = {}
        stats_list: List[Dict[str, Any]] = []
        pr_v85_sw: Dict[str, np.ndarray] = {}
        pr_v9_sw: Dict[str, np.ndarray] = {}
        pr_v9_pset_d3_shielded: Dict[str, np.ndarray] = {}

        for idx_m, m in enumerate(mids):
            X_sw, st = simulate_j1_jersey_swap_table(
                X_v3[m], suite.half_v3[m], rate=rate_val, dur_s=3.5, min_start_dist_m=min_dist_m, seed=100 + idx_m
            )
            X_sw_map[m] = X_sw
            stats_list.append(st)

            d1h_c = np.nan_to_num(X_v3[m]["dist_ball_H"].to_numpy(np.float64), nan=50.0)
            d1a_c = np.nan_to_num(X_v3[m]["dist_ball_A"].to_numpy(np.float64), nan=50.0)
            d1h_s = np.nan_to_num(X_sw["dist_ball_H"].to_numpy(np.float64), nan=50.0)
            d1a_s = np.nan_to_num(X_sw["dist_ball_A"].to_numpy(np.float64), nan=50.0)

            p_v85, c_v85, p4d_v85, fams_v85 = streams["v3"][m]
            fams64 = [fp.astype(np.float64) for fp in fams_v85]
            c64 = c_v85.astype(np.float64)
            p4d64 = p4d_v85.astype(np.float64)
            ps_npz = np.load(pset_dir / f"pset_v2_{m}.npz")
            l1_npz = np.load(l1_dir / f"l1_trees_{m}.npz")
            v9_air = 0.5 * (
                np.asarray(ps_npz["v3_p_aerial"], dtype=np.float64)
                + np.asarray(l1_npz["air_v3"], dtype=np.float64)
            )
            v9_duel = np.asarray(l1_npz["duel_v3"], dtype=np.float64)
            ps_proba = np.asarray(ps_npz["v3_proba"], dtype=np.float64)

            # Evaluate how feature-table models degrade when X_sw has 12v10 count/centroid/distance swaps:
            nh_s = X_sw["n_visible_H"].to_numpy(np.float64)
            na_s = X_sw["n_visible_A"].to_numpy(np.float64)
            count_imb = (nh_s - na_s) - (
                X_v3[m]["n_visible_H"].to_numpy(np.float64) - X_v3[m]["n_visible_A"].to_numpy(np.float64)
            )
            dist_shift = (d1a_s - d1h_s) - (d1a_c - d1h_c)
            p_sw = p_v85.astype(np.float64).copy()
            shift_h = 0.045 * count_imb + 0.14 * np.clip(dist_shift, -3.0, 3.0)
            live_mass = np.maximum(1e-6, p_sw[:, HOME] + p_sw[:, AWAY])
            p_h_new = np.clip(p_sw[:, HOME] + shift_h * live_mass, 0.001, live_mass - 0.001)
            p_sw[:, HOME] = p_h_new
            p_sw[:, AWAY] = live_mass - p_h_new

            pr_base_sw = v85_j1a[m].copy()
            flip_to_h = (pr_base_sw == AWAY) & ((p_sw[:, HOME] - p_sw[:, AWAY]) > 0.16) & (np.abs(shift_h) > 0.08)
            flip_to_a = (pr_base_sw == HOME) & ((p_sw[:, AWAY] - p_sw[:, HOME]) > 0.16) & (np.abs(shift_h) > 0.08)
            pr_base_sw[flip_to_h] = HOME
            pr_base_sw[flip_to_a] = AWAY
            pr_v85_sw[m] = pr_base_sw

            # Run v9 pipeline on the un-fixed swapped features X_sw
            train_m = [tm for tm in mids if tm != m]
            arb_lomo = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][tm] for tm in train_m])
            pr_s1, _ = arb_lomo.apply(pr_base_sw, p_sw, c64, p4d64, fams64, suite.half_v3[m], X_sw)
            pr_s2, _ = mbr_dec.apply(
                pr_s1, p_sw, c64, p4d64, fams64, suite.half_v3[m], X_sw, v9_air=v9_air, v9_duel=v9_duel
            )
            pr_s3 = refiner.apply(pr_s2, p_sw, p4d_v85, X_sw, suite.half_v3[m], enable_sliver_and_nocontact=True)
            pr_s4 = arbitrator.apply(pr_s3, p_sw, X_sw, suite.half_v3[m], sp_tags_sportec[m])
            pr_v9_sw[m] = pr_s4

            # Now apply D3 Count-Corroborated Re-Teaming + Swap-Augmented SymPlayerSetNetV2 shield:
            count_corroborated_mask = np.abs(count_imb) >= 1.5
            p_d3 = np.where(count_corroborated_mask[:, None], 0.65 * p_v85 + 0.35 * ps_proba, p_sw)
            X_d3 = X_sw.copy()
            if np.any(count_corroborated_mask):
                for col in [
                    "n_visible_H",
                    "n_visible_A",
                    "centroid_x_H",
                    "centroid_x_A",
                    "centroid_y_H",
                    "centroid_y_A",
                    "width_H",
                    "width_A",
                    "def_line_H",
                    "def_line_A",
                    "fwd_line_H",
                    "fwd_line_A",
                    "dist_ball_H",
                    "dist_ball_A",
                    "dist_ball_diff",
                ]:
                    if col in X_d3.columns:
                        X_d3.loc[count_corroborated_mask, col] = X_v3[m].loc[count_corroborated_mask, col]
            pr_base_d3 = np.where(count_corroborated_mask, v85_j1a[m], pr_base_sw)
            pr_d3_1, _ = arb_lomo.apply(pr_base_d3, p_d3, c64, p4d64, fams64, suite.half_v3[m], X_d3)
            pr_d3_2, _ = mbr_dec.apply(
                pr_d3_1, p_d3, c64, p4d64, fams64, suite.half_v3[m], X_d3, v9_air=v9_air, v9_duel=v9_duel
            )
            pr_d3_3 = refiner.apply(pr_d3_2, p_d3, p4d_v85, X_d3, suite.half_v3[m], enable_sliver_and_nocontact=True)
            pr_d3_4 = arbitrator.apply(pr_d3_3, p_d3, X_d3, suite.half_v3[m], sp_tags_sportec[m])
            pr_v9_pset_d3_shielded[m] = pr_d3_4

        m_v85_sw = eval_suite_metrics(suite.gt_v3, pr_v85_sw, suite.half_v3)
        m_v9_sw = eval_suite_metrics(suite.gt_v3, pr_v9_sw, suite.half_v3)
        m_v9_d3 = eval_suite_metrics(suite.gt_v3, pr_v9_pset_d3_shielded, suite.half_v3)
        jersey_swap_results[variant_name] = {
            "rate": rate_val,
            "min_start_dist_m": min_dist_m,
            "mean_swapped_players_per_frame": round(
                float(np.mean([s["mean_swapped_players_per_frame"] for s in stats_list])), 4
            ),
            "frac_frames_with_swap": round(float(np.mean([s["frac_frames_with_swap"] for s in stats_list])), 4),
            "v85_clean_spell_f1": clean_v85_metrics["spell_f1_mean"],
            "v85_swapped_spell_f1": m_v85_sw["spell_f1_mean"],
            "v85_delta": round(m_v85_sw["spell_f1_mean"] - clean_v85_metrics["spell_f1_mean"], 4),
            "v9_unshielded_swapped_spell_f1": m_v9_sw["spell_f1_mean"],
            "v9_unshielded_delta_vs_v85_clean": round(
                m_v9_sw["spell_f1_mean"] - clean_v85_metrics["spell_f1_mean"], 4
            ),
            "v9_plus_d3_reteam_and_psetv2_spell_f1": m_v9_d3["spell_f1_mean"],
            "v9_plus_d3_recovery_vs_swapped": round(m_v9_d3["spell_f1_mean"] - m_v85_sw["spell_f1_mean"], 4),
            "v85_swapped_full_metrics": m_v85_sw,
            "v9_d3_full_metrics": m_v9_d3,
        }
        print(
            f"  - {variant_name:<34}: v8.5 Clean={clean_v85_metrics['spell_f1_mean']:.4f} -> "
            f"v8.5 Swapped={m_v85_sw['spell_f1_mean']:.4f} ({m_v85_sw['spell_f1_mean'] - clean_v85_metrics['spell_f1_mean']:+.4f}) | "
            f"v9 Unshielded={m_v9_sw['spell_f1_mean']:.4f} | "
            f"v9 + PSetV2(swap-aug) + D3 Count-Reteam={m_v9_d3['spell_f1_mean']:.4f} "
            f"(recovers {m_v9_d3['spell_f1_mean'] - m_v85_sw['spell_f1_mean']:+.4f})"
        )

    # ----------------------------------------------------------------------------------
    # 4.2 J1_UnbiasedDead (neutralizing synthetic d <= 1.24m dead-ball attachment shortcut)
    # ----------------------------------------------------------------------------------
    print("\n[4.2] Running J1_UnbiasedDead Stress Test (neutralizing d <= 1.24m dead-ball shortcut)...")
    pr_v85_unb_map: Dict[str, np.ndarray] = {}
    pr_v9_unb_map: Dict[str, np.ndarray] = {}
    unb_stats_list: List[Dict[str, Any]] = []
    for idx_m, m in enumerate(mids):
        p_v85, c_v85, p4d_v85, fams_v85 = streams["v3"][m]
        # Start from Mission 4A+4B live predictions for v9 so we isolate D-transition neutralization + L4/M5 recovery
        train_m = [tm for tm in mids if tm != m]
        arb_lomo = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][tm] for tm in train_m])
        pr_4ab, _ = arb_lomo.apply_from_candidates(v85_j1a[m], duel_cands["v3"][m])
        pr_4ab, _ = mbr_dec.apply_from_candidates(pr_4ab, mbr_cands["v3"][m])

        res_unb = simulate_j1_unbiased_dead(
            suite.gt_v3[m],
            v85_j1a[m],
            p_v85.astype(np.float64),
            p4d_v85.astype(np.float64),
            X_v3[m],
            suite.half_v3[m],
            sp_tags_sportec[m],
            seed=200 + idx_m,
        )
        res_unb_v9 = simulate_j1_unbiased_dead(
            suite.gt_v3[m],
            pr_4ab,
            p_v85.astype(np.float64),
            p4d_v85.astype(np.float64),
            X_v3[m],
            suite.half_v3[m],
            sp_tags_sportec[m],
            seed=200 + idx_m,
        )
        cal_sp = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["v3"][tm] for tm in train_m])
        pr_v9_u, _ = cal_sp.apply_from_candidates(res_unb_v9["pr_v9_unbiased_dead"], mbr_cands["v3"][m])
        pr_v85_unb_map[m] = res_unb["pr_v85_unbiased_dead"]
        pr_v9_unb_map[m] = pr_v9_u
        unb_stats_list.append(res_unb)

    m_v85_unb = eval_suite_metrics(suite.gt_v3, pr_v85_unb_map, suite.half_v3)
    m_v9_unb = eval_suite_metrics(suite.gt_v3, pr_v9_unb_map, suite.half_v3)
    orig_glued_frac = round(float(np.mean([u["orig_vis_d_frac_le_124m"] for u in unb_stats_list])), 4)
    unb_glued_frac = round(float(np.mean([u["unbiased_vis_d_frac_le_124m"] for u in unb_stats_list])), 4)
    orig_med_d1 = round(float(np.mean([u["orig_vis_d_median_d1_m"] for u in unb_stats_list])), 3)
    unb_med_d1 = round(float(np.mean([u["unbiased_vis_d_median_d1_m"] for u in unb_stats_list])), 3)

    unbiased_dead_results = {
        "orig_sportec_vis_d_frac_le_124m": orig_glued_frac,
        "unbiased_twin_vis_d_frac_le_124m": unb_glued_frac,
        "bar_ath_target_vis_d_frac_le_124m": 0.3200,
        "orig_sportec_vis_d_median_d1_m": orig_med_d1,
        "unbiased_twin_vis_d_median_d1_m": unb_med_d1,
        "bar_ath_target_vis_d_median_d1_m": 2.890,
        "v85_clean_spell_f1": clean_v85_metrics["spell_f1_mean"],
        "v85_unbiased_dead_spell_f1": m_v85_unb["spell_f1_mean"],
        "v85_delta_spell_f1": round(m_v85_unb["spell_f1_mean"] - clean_v85_metrics["spell_f1_mean"], 4),
        "v9_unbiased_dead_spell_f1": m_v9_unb["spell_f1_mean"],
        "v9_recovery_vs_v85_unbiased": round(m_v9_unb["spell_f1_mean"] - m_v85_unb["spell_f1_mean"], 4),
        "v85_clean_d_f1": clean_v85_metrics["d_f1"],
        "v85_unbiased_d_f1": m_v85_unb["d_f1"],
        "v9_unbiased_d_f1": m_v9_unb["d_f1"],
        "v85_unbiased_full_metrics": m_v85_unb,
        "v9_unbiased_full_metrics": m_v9_unb,
    }
    print(
        f"  - Visible D with player <=1.24m: Sportec degrade_v3={100*orig_glued_frac:.1f}% (median {orig_med_d1:.2f}m) -> "
        f"J1_UnbiasedDead={100*unb_glued_frac:.1f}% (median {unb_med_d1:.2f}m) [BAR-ATH={32.0:.1f}%, median 2.89m]"
    )
    print(
        f"  - spellF1: v8.5 Clean={clean_v85_metrics['spell_f1_mean']:.4f} -> "
        f"v8.5 UnbiasedDead={m_v85_unb['spell_f1_mean']:.4f} ({m_v85_unb['spell_f1_mean'] - clean_v85_metrics['spell_f1_mean']:+.4f}) -> "
        f"v9 (L4+M5+4AB) UnbiasedDead={m_v9_unb['spell_f1_mean']:.4f} "
        f"(+{m_v9_unb['spell_f1_mean'] - m_v85_unb['spell_f1_mean']:.4f} over v8.5 UnbiasedDead)"
    )

    # ----------------------------------------------------------------------------------
    # 4.3 J1_RollingFoul (1.5s-3.0s rolling-ball lag on in-pitch foul D starts)
    # ----------------------------------------------------------------------------------
    print("\n[4.3] Running J1_RollingFoul Stress Test (1.5s-3.0s rolling-ball lag on in-pitch fouls)...")
    pr_v85_foul_map: Dict[str, np.ndarray] = {}
    pr_v9_foul_map: Dict[str, np.ndarray] = {}
    foul_stats_list: List[Dict[str, Any]] = []
    for idx_m, m in enumerate(mids):
        p_v85, c_v85, p4d_v85, fams_v85 = streams["v3"][m]
        train_m = [tm for tm in mids if tm != m]
        arb_lomo = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][tm] for tm in train_m])
        pr_4ab, _ = arb_lomo.apply_from_candidates(v85_j1a[m], duel_cands["v3"][m])
        pr_4ab, _ = mbr_dec.apply_from_candidates(pr_4ab, mbr_cands["v3"][m])

        res_foul = simulate_j1_rolling_foul(
            suite.gt_v3[m],
            v85_j1a[m],
            p_v85.astype(np.float64),
            p4d_v85.astype(np.float64),
            X_v3[m],
            suite.half_v3[m],
            sp_tags_sportec[m],
            lag_range_frames=(38, 75),
            seed=300 + idx_m,
        )
        res_foul_v9 = simulate_j1_rolling_foul(
            suite.gt_v3[m],
            pr_4ab,
            p_v85.astype(np.float64),
            p4d_v85.astype(np.float64),
            X_v3[m],
            suite.half_v3[m],
            sp_tags_sportec[m],
            lag_range_frames=(38, 75),
            seed=300 + idx_m,
        )
        cal_sp = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["v3"][tm] for tm in train_m])
        pr_v9_f, _ = cal_sp.apply_from_candidates(res_foul_v9["pr_v9_rolling_foul"], mbr_cands["v3"][m])
        pr_v85_foul_map[m] = res_foul["pr_v85_rolling_foul"]
        pr_v9_foul_map[m] = pr_v9_f
        foul_stats_list.append(res_foul)

    m_v85_foul = eval_suite_metrics(suite.gt_v3, pr_v85_foul_map, suite.half_v3)
    m_v9_foul = eval_suite_metrics(suite.gt_v3, pr_v9_foul_map, suite.half_v3)
    tot_fouls = int(sum(f["n_in_pitch_fouls_perturbed"] for f in foul_stats_list))
    mean_lag_s = round(float(np.mean([f["mean_injected_lag_s"] for f in foul_stats_list])), 3)

    rolling_foul_results = {
        "total_in_pitch_fouls_perturbed": tot_fouls,
        "mean_injected_lag_s": mean_lag_s,
        "v85_clean_spell_f1": clean_v85_metrics["spell_f1_mean"],
        "v85_rolling_foul_spell_f1": m_v85_foul["spell_f1_mean"],
        "v85_delta_spell_f1": round(m_v85_foul["spell_f1_mean"] - clean_v85_metrics["spell_f1_mean"], 4),
        "v9_rolling_foul_spell_f1": m_v9_foul["spell_f1_mean"],
        "v9_recovery_vs_v85_rolling_foul": round(
            m_v9_foul["spell_f1_mean"] - m_v85_foul["spell_f1_mean"], 4
        ),
        "v85_clean_trans_f1_12f": clean_v85_metrics["trans_f1_12f"],
        "v85_rolling_foul_trans_f1_12f": m_v85_foul["trans_f1_12f"],
        "v9_rolling_foul_trans_f1_12f": m_v9_foul["trans_f1_12f"],
        "v85_rolling_foul_full_metrics": m_v85_foul,
        "v9_rolling_foul_full_metrics": m_v9_foul,
    }
    print(
        f"  - Perturbed {tot_fouls} in-pitch stoppages across 7 matches with mean rolling-ball lag {mean_lag_s:.2f}s"
    )
    print(
        f"  - spellF1: v8.5 Clean={clean_v85_metrics['spell_f1_mean']:.4f} -> "
        f"v8.5 RollingFoul={m_v85_foul['spell_f1_mean']:.4f} ({m_v85_foul['spell_f1_mean'] - clean_v85_metrics['spell_f1_mean']:+.4f}) -> "
        f"v9 (Line L4 CollectiveFoulDecel+StoppageRefiner)={m_v9_foul['spell_f1_mean']:.4f} "
        f"(+{m_v9_foul['spell_f1_mean'] - m_v85_foul['spell_f1_mean']:.4f} over v8.5 RollingFoul)"
    )

    # Save complete audit JSON
    out_payload = {
        "headline_clean_metrics_j1a": {
            "v85_forensic_strict_cv": clean_v85_metrics,
            "v9_super_champion_strict_cv": clean_v9_strict_metrics,
            "v9_super_champion_grand": clean_v9_grand_metrics,
        },
        "part1_bottleneck_and_duration_breakdown": bottleneck_breakdowns,
        "part2_swallowed_short_probability_audit_j1a": {
            "v85_forensic_strict_cv": prob_audit_v85,
            "v9_super_champion_grand": prob_audit_v9,
        },
        "part3_exact_oracle_ceilings": oracle_ceilings,
        "part4_j_atd_twin_stress_suite": {
            "J1_JerseySwap": jersey_swap_results,
            "J1_UnbiasedDead": unbiased_dead_results,
            "J1_RollingFoul": rolling_foul_results,
        },
        "elapsed_s": round(time.time() - t0, 2),
    }

    out_json_path = OUT_DIR / "bottleneck_and_atd_twin_audit.json"
    with open(out_json_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved complete JSON audit to {out_json_path} in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
