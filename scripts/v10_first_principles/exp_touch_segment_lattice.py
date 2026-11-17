#!/usr/bin/env python3
"""
Experiment `B1` & `B2` — Touch-Graph & Segment-Lattice Neural Benchmark (`v10`)
===============================================================================

Evaluates our First-Principles Touch-Graph & Segment-Lattice Neural Decoder (`B1` Short-Steal /
Double-Domino & `B2` Pre-Touch Boundary Solver) in strict **7-Fold Nested LOMO** on top of
`v9_super_champion_grand` (and `v8.5 Strict-CV`):
  1. Verifies exact `0.000e+00` `HOME <-> AWAY` equivariance across `MBR-v10`, `XYX-v10`, `XYD-v10`,
     `DYX-v10`, and `B2-v10` on both Sportec (`J03WMX`) and `BAR-ATH` (`199,319` frames).
  2. Evaluates progressive out-of-fold ablation ladder across `J1A (v3)`, `J1B-Std (v6_std)`,
     `J1B-Gen (v6_gen)`, `J2-Raw (sample)`, `J2-noD (sample)`, and `BAR-ATH` telemetry (`F1@0.3`,
     `F1@0.5`, `F1@0.6`, `F1@0.7`, `spell_f1_mean`, and exact per-fold breakdowns).
  3. Saves full benchmark audit JSON to `experiments/outputs/v10_first_principles/touch_segment_lattice_results.json`.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "scripts"), str(ROOT / "py_pkgs"), str(ROOT / "evaluation")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_w7a_player_set_net import load_cached_physv2_table
from games import GAMES as GAME_INFO, anchor_video_frame
from s3_11_tcn_local import without_dead_margin
from src.data.common import DEAD
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
    load_bar_ath_smart_tags,
)
from src.v9_swarm.whistle_and_stoppage import Fox40WhistleDetector, StoppageStartRefiner
from src.v10_first_principles.touch_segment_lattice import (
    SymTouchBoundaryAlignerV10,
    SymTouchSegmentLatticeV10,
    enrich_candidates_with_t10,
    extract_b2_v10_boundary_candidates,
    extract_mbr_xyx_candidates_v10,
    extract_touch_segment_proposals_v10,
    verify_touch_segment_lattice_equivariance,
)

OUT_DIR = ROOT / "experiments" / "outputs" / "v10_first_principles"


def _clone_bundle(b: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "preds_v3": {k: v.copy() for k, v in b["preds_v3"].items()},
        "preds_v6_std": {k: v.copy() for k, v in b["preds_v6_std"].items()},
        "preds_v6_gen": {k: v.copy() for k, v in b["preds_v6_gen"].items()},
        "preds_sample": {k: v.copy() for k, v in b["preds_sample"].items()},
        "pred_atd_rep": b["pred_atd_rep"].copy(),
        "pred_atd_raw": b["pred_atd_raw"].copy(),
    }


def _score_split(
    preds_dict: Dict[Any, np.ndarray],
    gt_dict: Dict[Any, np.ndarray],
    hl_dict: Dict[Any, np.ndarray],
    no_dead_margin: bool = False,
) -> Dict[str, Any]:
    keys = sorted(gt_dict.keys())
    per_fold: Dict[str, float] = {}
    f03, f05, f06, f07, fmean = [], [], [], [], []
    for k in keys:
        gt = without_dead_margin(gt_dict[k], hl_dict[k]) if no_dead_margin else gt_dict[k]
        r = metrics.score(gt, preds_dict[k], hl_dict[k])
        sf = r["spell_f1"]
        m_val = float(r["spell_f1_mean"])
        per_fold[str(k)] = round(m_val, 6)
        fmean.append(m_val)
        f03.append(float(sf[0.3]["f1"]))
        f05.append(float(sf[0.5]["f1"]))
        f06.append(float(sf[0.6]["f1"]))
        f07.append(float(sf[0.7]["f1"]))
    return {
        "spell_f1_mean": round(float(np.mean(fmean)), 6),
        "f1_03": round(float(np.mean(f03)), 6),
        "f1_05": round(float(np.mean(f05)), 6),
        "f1_06": round(float(np.mean(f06)), 6),
        "f1_07": round(float(np.mean(f07)), 6),
        "per_fold": per_fold,
    }


def _evaluate_bundle(
    bundle: Dict[str, Any],
    base_bundle: Dict[str, Any],
    suite: JSuite,
) -> Dict[str, Any]:
    splits = {}
    for name, b_key, gt_d, hl_d, no_d in [
        ("J1A", "preds_v3", suite.gt_v3, suite.half_v3, False),
        ("J1B_Std", "preds_v6_std", suite.gt_v3, suite.half_v3, False),
        ("J1B_Gen", "preds_v6_gen", suite.gt_v3, suite.half_v3, False),
        ("J2_Raw", "preds_sample", suite.gt_sample, suite.half_sample, False),
        ("J2_noD", "preds_sample", suite.gt_sample, suite.half_sample, True),
    ]:
        cur = _score_split(bundle[b_key], gt_d, hl_d, no_dead_margin=no_d)
        ref = _score_split(base_bundle[b_key], gt_d, hl_d, no_dead_margin=no_d)
        deltas = {k: round(cur["per_fold"][k] - ref["per_fold"][k], 6) for k in cur["per_fold"]}
        cur["delta_vs_ref"] = round(cur["spell_f1_mean"] - ref["spell_f1_mean"], 6)
        cur["per_fold_delta_vs_ref"] = deltas
        cur["n_pos_folds"] = sum(1 for d in deltas.values() if d > 1e-6)
        cur["n_neg_folds"] = sum(1 for d in deltas.values() if d < -1e-6)
        splits[name] = cur

    # BAR-ATH telemetry + report-only J3
    pr_atd = bundle["pred_atd_rep"]
    r_j3 = metrics.score_public(pr_atd, bootstrap_n=0)
    graded = suite.atd.graded
    pr_g = pr_atd[graded]
    n_g = len(pr_g)
    changes = int(np.sum(pr_g[1:] != pr_g[:-1]))
    poss_per_min = float((changes + 1) / (n_g / 1500.0))
    splits["BAR_ATH"] = {
        "j3_report_only_spell_f1": round(float(r_j3["spell_f1_mean"]), 6),
        "j3_f1_03": round(float(r_j3["spell_f1"][0.3]["f1"]), 6),
        "j3_f1_05": round(float(r_j3["spell_f1"][0.5]["f1"]), 6),
        "j3_f1_06": round(float(r_j3["spell_f1"][0.6]["f1"]), 6),
        "j3_f1_07": round(float(r_j3["spell_f1"][0.7]["f1"]), 6),
        "poss_per_min_graded": round(poss_per_min, 4),
        "pct_D_graded": round(100.0 * float(np.mean(pr_g == DEAD)), 2),
        "diff_frames_vs_ref_graded": int(np.sum(pr_g != base_bundle["pred_atd_rep"][graded])),
    }
    return splits


def build_v9_grand_baseline(
    suite: JSuite,
    streams: Dict[str, Any],
    X_v3: Dict[str, Any],
    X_v6: Dict[str, Any],
    X_sample: Dict[int, Any],
    X_atd_rep: Any,
) -> Dict[str, Any]:
    """Reconstruct exact `v9_super_champion_grand` predictions across all splits."""
    mids = sorted(suite.gt_v3.keys())
    gids = sorted(suite.gt_sample.keys())
    atd = suite.atd
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    with open(SWARM_OUT_DIR / "_diag_enr_cands.pkl", "rb") as f:
        enr = pickle.load(f)
    duel_cands = enr["duel"]
    mbr_cands = enr["mbr"]

    whistle_path = ROOT / "local_data" / "v9_swarm" / "bar_ath_whistle.npz"
    w_dual = Fox40WhistleDetector().extract_and_cache_bar_ath_whistle(
        ROOT, cache_path=whistle_path, cleanup_wav=True
    )["w_dual"]
    sp_tags_bar, phase_score_bar = load_bar_ath_smart_tags(ROOT, n_frames=atd.n_frames)
    sp_tags_sportec = {mid: extract_simulated_smart_tags_sportec(ROOT, mid, suite.sportec[mid]) for mid in mids}
    sp_tags_sample = {gid: extract_simulated_smart_tags_sample(ROOT, gid) for gid in gids}

    v85_base = suite.baselines["v85_forensic_strict_cv"]
    duel_full = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][m] for m in mids])
    mbr_dec = SymSpellMBRDecoderV9()
    dom_cal_full = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["j2"][g] for g in gids])
    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()

    b = _clone_bundle(v85_base)
    for m in mids:
        train_m = [tm for tm in mids if tm != m]
        arb_lomo = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][tm] for tm in train_m])
        b["preds_v3"][m], _ = arb_lomo.apply_from_candidates(b["preds_v3"][m], duel_cands["v3"][m])
        b["preds_v6_std"][m], _ = arb_lomo.apply_from_candidates(b["preds_v6_std"][m], duel_cands["v6_std"][m])
        b["preds_v6_gen"][m], _ = arb_lomo.apply_from_candidates(b["preds_v6_gen"][m], duel_cands["v6_gen"][m])
    for g in gids:
        b["preds_sample"][g], _ = duel_full.apply_from_candidates(b["preds_sample"][g], duel_cands["j2"][g])
    b["pred_atd_rep"], _ = duel_full.apply_from_candidates(b["pred_atd_rep"], duel_cands["atd_rep"])

    for m in mids:
        b["preds_v3"][m], _ = mbr_dec.apply_from_candidates(b["preds_v3"][m], mbr_cands["v3"][m])
        b["preds_v6_std"][m], _ = mbr_dec.apply_from_candidates(b["preds_v6_std"][m], mbr_cands["v6_std"][m])
        b["preds_v6_gen"][m], _ = mbr_dec.apply_from_candidates(b["preds_v6_gen"][m], mbr_cands["v6_gen"][m])
    for g in gids:
        b["preds_sample"][g], _ = mbr_dec.apply_from_candidates(b["preds_sample"][g], mbr_cands["j2"][g])
    b["pred_atd_rep"], _ = mbr_dec.apply_from_candidates(b["pred_atd_rep"], mbr_cands["atd_rep"])

    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, _, p4d, _ = streams[sp_key][m]
            b[b_key][m] = refiner.apply(
                b[b_key][m], p, p4d, X_dict[m], suite.half_v3[m], enable_sliver_and_nocontact=(sp_key != "v6_gen")
            )
    for g in gids:
        p, _, p4d, _ = streams["sample"][g]
        b["preds_sample"][g] = refiner.apply(
            b["preds_sample"][g], p, p4d, X_sample[g], suite.half_sample[g], enable_sliver_and_nocontact=True
        )
    p_atd, c_atd, p4d_atd, fams_atd = streams["atd_rep"]
    b["pred_atd_rep"] = refiner.apply(
        b["pred_atd_rep"], p_atd, p4d_atd, X_atd_rep, half_atd_dec, whistle_z=w_dual, enable_sliver_and_nocontact=True
    )
    b["pred_atd_rep"][second_start:kick2] = DEAD
    b["pred_atd_rep"][~atd.graded] = DEAD

    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, _, _, _ = streams[sp_key][m]
            b[b_key][m] = arbitrator.apply(b[b_key][m], p, X_dict[m], suite.half_v3[m], sp_tags_sportec[m])
    for g in gids:
        p, _, _, _ = streams["sample"][g]
        b["preds_sample"][g] = arbitrator.apply(b["preds_sample"][g], p, X_sample[g], suite.half_sample[g], sp_tags_sample[g])
    b["pred_atd_rep"] = arbitrator.apply(
        b["pred_atd_rep"], p_atd, X_atd_rep, half_atd_dec, sp_tags_bar, phase_score=phase_score_bar, whistle_z=w_dual
    )
    b["pred_atd_rep"][second_start:kick2] = DEAD
    b["pred_atd_rep"][~atd.graded] = DEAD

    b_strict = _clone_bundle(b)
    for g in gids:
        train_g = [tg for tg in gids if tg != g]
        cal_j2 = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["j2"][tg] for tg in train_g])
        b["preds_sample"][g], _ = cal_j2.apply_from_candidates(b["preds_sample"][g], mbr_cands["j2"][g])
    atd_grand_pred, _ = dom_cal_full.apply_from_candidates(b["pred_atd_rep"], mbr_cands["atd_rep"])
    atd_grand_pred[second_start:kick2] = DEAD
    atd_grand_pred[~atd.graded] = DEAD
    b["pred_atd_rep"] = atd_grand_pred
    with open(OUT_DIR / "v9_preds_cache.pkl", "wb") as f:
        pickle.dump({"strict": b_strict, "grand": b}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return b


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suite = JSuite(n_boot=10, seed=42, verbose=False)
    mids = sorted(suite.gt_v3.keys())
    gids = sorted(suite.gt_sample.keys())
    atd = suite.atd
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    print("=" * 116)
    print("STEP 1: Loading v8.5 9-Family Streams + Modal GPU SymPlayerSetNetV2 + Modal CPU L1-Trees + PhysV2")
    print("=" * 116)
    with open(SWARM_OUT_DIR / "_diag_streams_s3.pkl", "rb") as f:
        s3 = pickle.load(f)
    streams = s3["streams"]

    X_v3 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", suite.half_v3[mid]) for mid in mids}
    X_v6 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", suite.half_v3[mid]) for mid in mids}
    X_sample = {gid: load_cached_physv2_table(f"sample_{gid}_raw", suite.half_sample[gid]) for gid in gids}
    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)

    ps_v3, ps_v6, l1_v3, l1_v6, t10_v3, t10_v6 = {}, {}, {}, {}, {}, {}
    for m in mids:
        z_ps = np.load(SWARM_OUT_DIR / "pset_v2_oof" / f"pset_v2_{m}.npz")
        z_l1 = np.load(SWARM_OUT_DIR / "l1_trees_oof" / f"l1_trees_{m}.npz")
        z_t10 = np.load(OUT_DIR / "traj_v10_oof" / f"traj_v10_{m}.npz")
        ps_v3[m] = {k[3:]: z_ps[k] for k in z_ps.files if k.startswith("v3_")}
        ps_v6[m] = {k[3:]: z_ps[k] for k in z_ps.files if k.startswith("v6_")}
        l1_v3[m] = {k[:-3]: z_l1[k] for k in z_l1.files if k.endswith("_v3")}
        l1_v6[m] = {k[:-3]: z_l1[k] for k in z_l1.files if k.endswith("_v6")}
        t10_v3[m] = {k[3:]: z_t10[k] for k in z_t10.files if k.startswith("v3_")}
        t10_v6[m] = {k[3:]: z_t10[k] for k in z_t10.files if k.startswith("v6_")}

    with open(SWARM_OUT_DIR / "pset_v2_oof" / "pset_v2_j23_preds.pkl", "rb") as f:
        ps_j23 = pickle.load(f)
    with open(SWARM_OUT_DIR / "l1_trees_oof" / "l1_trees_j23_preds.pkl", "rb") as f:
        l1_j23 = pickle.load(f)
    with open(OUT_DIR / "traj_v10_oof" / "traj_v10_j23_preds.pkl", "rb") as f:
        t10_j23 = pickle.load(f)

    ps_sample = {g: {kk[5:]: vv for kk, vv in ps_j23["sample"][g].items()} for g in gids}
    l1_sample = {g: l1_j23["sample"][g] for g in gids}
    t10_sample = {g: t10_j23["sample"][g] for g in gids}
    ps_atd = {kk[5:]: vv for kk, vv in ps_j23["atd_rep"].items()}
    l1_atd = l1_j23["atd_rep"]
    t10_atd = t10_j23["atd_rep"]

    v85_base = suite.baselines["v85_forensic_strict_cv"]
    scratch_v9 = OUT_DIR / "v9_preds_cache.pkl"
    if scratch_v9.exists():
        with open(scratch_v9, "rb") as f:
            v9_grand = pickle.load(f)["grand"]
    else:
        v9_grand = build_v9_grand_baseline(suite, streams, X_v3, X_v6, X_sample, X_atd_rep)

    print("\n" + "=" * 116)
    print("STEP 2: Verifying Exact 0.000e+00 H <-> A Equivariance on Sportec (J03WMX) & BAR-ATH (199,319 frames)")
    print("=" * 116)
    m0 = mids[0]
    p0, c0, p4d0, fams0 = streams["v3"][m0]
    eq_sportec = verify_touch_segment_lattice_equivariance(
        v9_grand["preds_v3"][m0], p0, c0, p4d0, fams0, suite.half_v3[m0], X_v3[m0], ps_v3[m0], l1_v3[m0], t10=t10_v3[m0]
    )
    p_atd, c_atd, p4d_atd, fams_atd = streams["atd_rep"]
    eq_bar_ath = verify_touch_segment_lattice_equivariance(
        v9_grand["pred_atd_rep"], p_atd, c_atd, p4d_atd, fams_atd, half_atd_dec, X_atd_rep, ps_atd, l1_atd, t10=t10_atd
    )
    print(f"  Sportec ({m0}) Equivariance: {eq_sportec}")
    print(f"  BAR-ATH          Equivariance: {eq_bar_ath}")
    assert eq_sportec["equivariant"], f"Equivariance failed on Sportec: {eq_sportec}"
    assert eq_bar_ath["equivariant"], f"Equivariance failed on BAR-ATH: {eq_bar_ath}"

    print("\n" + "=" * 116)
    print("STEP 3: Extracting Multi-Modal Segment-Lattice Proposals (MBR, XYX, XYD, DYX) & B2 Boundary Windows")
    print("=" * 116)
    with open(SWARM_OUT_DIR / "_diag_enr_cands.pkl", "rb") as f:
        enr_mbr = pickle.load(f)["mbr"]

    scratch_b1 = OUT_DIR / "b1_v10_cands.pkl"
    scratch_b2 = OUT_DIR / "b2_v10_cands.pkl"
    if scratch_b1.exists() and scratch_b2.exists():
        with open(scratch_b1, "rb") as f:
            b1_cands = pickle.load(f)
        with open(scratch_b2, "rb") as f:
            b2_cands = pickle.load(f)
    else:
        b1_cands = {"XYX": {}, "XYD": {}, "DYX": {}}
        b2_cands = {}
        for sp_key, b_key, X_dict in [
            ("v3", "preds_v3", X_v3),
            ("v6_std", "preds_v6_std", X_v6),
            ("v6_gen", "preds_v6_gen", X_v6),
        ]:
            b1_cands["XYX"][sp_key] = {}
            b1_cands["XYD"][sp_key] = {}
            b1_cands["DYX"][sp_key] = {}
            b2_cands[sp_key] = {}
            for m in mids:
                p, c, p4d, fams = streams[sp_key][m]
                ps = ps_v3[m] if sp_key == "v3" else ps_v6[m]
                l1 = l1_v3[m] if sp_key == "v3" else l1_v6[m]
                t10 = t10_v3[m] if sp_key == "v3" else t10_v6[m]
                props = extract_touch_segment_proposals_v10(
                    v9_grand[b_key][m], p, c, p4d, fams, suite.half_v3[m], X_dict[m], ps, l1, gt=suite.gt_v3[m], t10=t10
                )
                b1_cands["XYX"][sp_key][m] = props["XYX"]
                b1_cands["XYD"][sp_key][m] = props["XYD"]
                b1_cands["DYX"][sp_key][m] = props["DYX"]
                b2_cands[sp_key][m] = extract_b2_v10_boundary_candidates(
                    v9_grand[b_key][m], suite.half_v3[m], p, c, p4d, fams, ps, l1, X_dict[m], labels=suite.gt_v3[m]
                )

        b1_cands["XYX"]["sample"] = {}
        b1_cands["XYD"]["sample"] = {}
        b1_cands["DYX"]["sample"] = {}
        b2_cands["sample"] = {}
        for g in gids:
            p, c, p4d, fams = streams["sample"][g]
            props = extract_touch_segment_proposals_v10(
                v9_grand["preds_sample"][g],
                p,
                c,
                p4d,
                fams,
                suite.half_sample[g],
                X_sample[g],
                ps_sample[g],
                l1_sample[g],
                gt=suite.gt_sample[g],
                t10=t10_sample[g],
            )
            b1_cands["XYX"]["sample"][g] = props["XYX"]
            b1_cands["XYD"]["sample"][g] = props["XYD"]
            b1_cands["DYX"]["sample"][g] = props["DYX"]
            b2_cands["sample"][g] = extract_b2_v10_boundary_candidates(
                v9_grand["preds_sample"][g],
                suite.half_sample[g],
                p,
                c,
                p4d,
                fams,
                ps_sample[g],
                l1_sample[g],
                X_sample[g],
                labels=suite.gt_sample[g],
            )
        with open(scratch_b1, "wb") as f:
            pickle.dump(b1_cands, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(scratch_b2, "wb") as f:
            pickle.dump(b2_cands, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Always enrich cached b1_cands with SymTrajectoryEventNetV10 (`t10`) features
    for sp_key, t_dict, keys in [
        ("v3", t10_v3, mids),
        ("v6_std", t10_v6, mids),
        ("v6_gen", t10_v6, mids),
        ("sample", t10_sample, gids),
    ]:
        for k in keys:
            for kind in ("XYX", "XYD", "DYX"):
                enrich_candidates_with_t10(b1_cands[kind][sp_key][k], t_dict[k])

    props_atd = extract_touch_segment_proposals_v10(
        v9_grand["pred_atd_rep"], p_atd, c_atd, p4d_atd, fams_atd, half_atd_dec, X_atd_rep, ps_atd, l1_atd, gt=None, t10=t10_atd
    )
    b2_atd = extract_b2_v10_boundary_candidates(
        v9_grand["pred_atd_rep"], half_atd_dec, p_atd, c_atd, p4d_atd, fams_atd, ps_atd, l1_atd, X_atd_rep, labels=None
    )

    print("\n" + "=" * 116)
    print("STEP 4: Evaluating Progressive Ablation Ladder in Strict 7-Fold Nested LOMO")
    print("=" * 116)

    def run_stage(
        enable_mbr: bool,
        enable_xyx: bool,
        enable_xyd: bool,
        enable_dyx: bool,
        enable_b2: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        b = _clone_bundle(v9_grand)
        fire_counts = {"v3": 0, "v6_std": 0, "v6_gen": 0, "sample": 0, "atd_rep": 0}

        # 7-Fold Nested LOMO on Sportec
        for m in mids:
            tr_m = [tm for tm in mids if tm != m]
            lat = SymTouchSegmentLatticeV10().fit_lomo(
                [enr_mbr["v3"][tm] for tm in tr_m],
                [b1_cands["XYX"]["v3"][tm] for tm in tr_m],
                [b1_cands["XYD"]["v3"][tm] for tm in tr_m],
                [b1_cands["DYX"]["v3"][tm] + b1_cands["DYX"]["v6_std"][tm] for tm in tr_m],
            )
            al = SymTouchBoundaryAlignerV10().fit_lomo([b2_cands["v3"][tm] for tm in tr_m])
            for sp_key, b_key in [("v3", "preds_v3"), ("v6_std", "preds_v6_std"), ("v6_gen", "preds_v6_gen")]:
                p_cur, f_dict = lat.apply_from_candidates(
                    b[b_key][m],
                    enr_mbr[sp_key][m],
                    b1_cands["XYX"][sp_key][m],
                    b1_cands["XYD"][sp_key][m],
                    b1_cands["DYX"][sp_key][m],
                    enable_mbr=enable_mbr,
                    enable_xyx=enable_xyx,
                    enable_xyd=enable_xyd,
                    enable_dyx=enable_dyx,
                )
                f_b2 = []
                if enable_b2:
                    p_cur, f_b2 = al.apply_from_candidates(p_cur, b2_cands[sp_key][m])
                b[b_key][m] = p_cur
                fire_counts[sp_key] += sum(len(v) for v in f_dict.values()) + len(f_b2)

        # Fit on all 7 Sportec matches for OOD J2 (Sample 1-3) & BAR-ATH
        lat_full = SymTouchSegmentLatticeV10().fit_lomo(
            [enr_mbr["v3"][tm] for tm in mids],
            [b1_cands["XYX"]["v3"][tm] for tm in mids],
            [b1_cands["XYD"]["v3"][tm] for tm in mids],
            [b1_cands["DYX"]["v3"][tm] + b1_cands["DYX"]["v6_std"][tm] for tm in mids],
        )
        al_full = SymTouchBoundaryAlignerV10().fit_lomo([b2_cands["v3"][tm] for tm in mids])

        for g in gids:
            p_cur, f_dict = lat_full.apply_from_candidates(
                b["preds_sample"][g],
                enr_mbr["j2"][g],
                b1_cands["XYX"]["sample"][g],
                b1_cands["XYD"]["sample"][g],
                b1_cands["DYX"]["sample"][g],
                enable_mbr=enable_mbr,
                enable_xyx=enable_xyx,
                enable_xyd=enable_xyd,
                enable_dyx=enable_dyx,
            )
            f_b2 = []
            if enable_b2:
                p_cur, f_b2 = al_full.apply_from_candidates(p_cur, b2_cands["sample"][g])
            b["preds_sample"][g] = p_cur
            fire_counts["sample"] += sum(len(v) for v in f_dict.values()) + len(f_b2)

        p_atd_cur, f_atd = lat_full.apply_from_candidates(
            b["pred_atd_rep"],
            enr_mbr["atd_rep"],
            props_atd["XYX"],
            props_atd["XYD"],
            props_atd["DYX"],
            enable_mbr=enable_mbr,
            enable_xyx=enable_xyx,
            enable_xyd=enable_xyd,
            enable_dyx=enable_dyx,
        )
        f_b2_atd = []
        if enable_b2:
            p_atd_cur, f_b2_atd = al_full.apply_from_candidates(p_atd_cur, b2_atd)
        p_atd_cur[second_start:kick2] = DEAD
        p_atd_cur[~atd.graded] = DEAD
        b["pred_atd_rep"] = p_atd_cur
        fire_counts["atd_rep"] = sum(len(v) for v in f_atd.values()) + len(f_b2_atd)
        return b, fire_counts

    stages = [
        ("0_v85_forensic_strict_cv", "v8.5 Strict-CV Baseline (9-Family + Forensic Shields)", v85_base, {}),
        ("1_v9_super_champion_grand", "v9 Super-Champion Grand (4A + 4B + L4 + M5-v9 + 4C)", v9_grand, {}),
    ]
    for st_key, st_desc, flags in [
        ("2_v10_plus_MBR_A4_A6", "+ B1-MBR-v10 (3D Pass-Through & Parallel-Run Merger A4-A6)", (True, False, False, False, False)),
        ("3_v10_plus_XYX_domino", "+ B1-XYX-v10 (Analytical Double-Domino Short-Steal & Header Arbitrator)", (True, True, False, False, False)),
        ("4_v10_plus_XYD_pre_stoppage", "+ B1-XYD-v10 (Pre-Stoppage Deflection / Clearance / Foul-Won Arbitrator)", (True, True, True, False, False)),
        ("5_v10_plus_DYX_post_restart", "+ B1-DYX-v10 (Post-Restart Aerial / Second-Ball Arbitrator)", (True, True, True, True, False)),
        ("6_v10_full_touch_segment_lattice", "+ B2-v10 (SymTouchBoundaryAlignerV10 Pre-Touch Boundary Snapper)", (True, True, True, True, True)),
    ]:
        b_st, fc_st = run_stage(*flags)
        stages.append((st_key, st_desc, b_st, fc_st))

    results_ladder: Dict[str, Any] = {}
    print(
        f"{'Stage':36s} | {'J1A (v3)':18s} | {'J1B-Std':18s} | {'J1B-Gen':18s} | "
        f"{'J2-Raw':18s} | {'J2-noD':18s} | {'J3[Rep]'}"
    )
    print("-" * 145)
    for st_key, st_desc, b_st, fc_st in stages:
        ev_vs_v9 = _evaluate_bundle(b_st, v9_grand, suite)
        ev_vs_v85 = _evaluate_bundle(b_st, v85_base, suite)
        results_ladder[st_key] = {
            "description": st_desc,
            "fires": fc_st,
            "metrics_vs_v9_grand": ev_vs_v9,
            "metrics_vs_v85_strict_cv": {
                k: v["delta_vs_ref"] for k, v in ev_vs_v85.items() if k != "BAR_ATH"
            },
        }
        j1a = ev_vs_v9["J1A"]
        j1bs = ev_vs_v9["J1B_Std"]
        j1bg = ev_vs_v9["J1B_Gen"]
        j2r = ev_vs_v9["J2_Raw"]
        j2n = ev_vs_v9["J2_noD"]
        j3r = ev_vs_v9["BAR_ATH"]["j3_report_only_spell_f1"]
        print(
            f"{st_key:36s} | "
            f"{j1a['spell_f1_mean']:.4f} ({j1a['delta_vs_ref']:+.4f} +{j1a['n_pos_folds']}/-{j1a['n_neg_folds']}) | "
            f"{j1bs['spell_f1_mean']:.4f} ({j1bs['delta_vs_ref']:+.4f} +{j1bs['n_pos_folds']}/-{j1bs['n_neg_folds']}) | "
            f"{j1bg['spell_f1_mean']:.4f} ({j1bg['delta_vs_ref']:+.4f} +{j1bg['n_pos_folds']}/-{j1bg['n_neg_folds']}) | "
            f"{j2r['spell_f1_mean']:.4f} ({j2r['delta_vs_ref']:+.4f} +{j2r['n_pos_folds']}/-{j2r['n_neg_folds']}) | "
            f"{j2n['spell_f1_mean']:.4f} ({j2n['delta_vs_ref']:+.4f} +{j2n['n_pos_folds']}/-{j2n['n_neg_folds']}) | "
            f"{j3r:.4f}"
        )

    final_ev = results_ladder["6_v10_full_touch_segment_lattice"]["metrics_vs_v9_grand"]
    print("\n" + "=" * 116)
    print("STEP 5: Detailed Threshold Breakdown (F1@0.3, F1@0.5, F1@0.6, F1@0.7) — v9_grand vs v10_full_lattice")
    print("=" * 116)
    base_ev = results_ladder["1_v9_super_champion_grand"]["metrics_vs_v9_grand"]
    for sp_name in ["J1A", "J1B_Std", "J1B_Gen", "J2_Raw", "J2_noD"]:
        b_m = base_ev[sp_name]
        f_m = final_ev[sp_name]
        print(
            f"  {sp_name:8s}: spellF1 {b_m['spell_f1_mean']:.4f} -> {f_m['spell_f1_mean']:.4f} ({f_m['delta_vs_ref']:+.4f}) | "
            f"F1@0.3: {b_m['f1_03']:.4f}->{f_m['f1_03']:.4f} | "
            f"F1@0.5: {b_m['f1_05']:.4f}->{f_m['f1_05']:.4f} | "
            f"F1@0.6: {b_m['f1_06']:.4f}->{f_m['f1_06']:.4f} | "
            f"F1@0.7: {b_m['f1_07']:.4f}->{f_m['f1_07']:.4f}"
        )
        print(f"            per-fold deltas vs v9_grand: {f_m['per_fold_delta_vs_ref']}")

    out_payload = {
        "runtime_sec": round(time.time() - t0, 2),
        "equivariance": {
            "sportec_m0": eq_sportec,
            "bar_ath": eq_bar_ath,
        },
        "ablation_ladder": results_ladder,
    }
    json_path = OUT_DIR / "touch_segment_lattice_results.json"
    json_path.write_text(json.dumps(out_payload, indent=2))
    print(f"\nSaved full JSON results to {json_path} in {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
