#!/usr/bin/env python3
"""Line L4 + Multi-Modal Smart Tagging & Broadcast Whistle Benchmark (`exp_l4_stoppage_and_smart_tagging.py`).

Runs and validates:
1. `Fox40WhistleDetector`:
   - Extracts and caches `local_data/v9_swarm/bar_ath_whistle.npz` from `BAR-ATH_broadcast.mp4`
     and `BAR-ATH_tactical.mp4`.
   - Computes unsupervised enrichment of whistle peaks around `v8.5`'s in-pitch `Live -> D`
     stoppages and `Smart Tagging` `Free Kick` tags.
2. `CollectiveFoulDeceleration` + `StoppageStartRefiner` (`Line L4`):
   - Evaluates pure tracking + 4D transition stoppage refinement across `J1A` (`Sportec v3`),
     `J1B-Std`, `J1B-Gen`, `J2-Raw`, and `J2-noD`, plus simulated rolling-ball foul degradation
     (`J1A-RollingFoul`, `J2-RollingFoul`).
3. `SmartTaggingArbitrator` (`M5-v9`):
   - Evaluates set-piece restart & blackout arbitration across `J1A`, `J1B-Std`, `J1B-Gen`,
     `J2-Raw`, `J2-noD`, and `BAR-ATH` (`BAR-ATH_tactical_smart_tagging.xml`).
4. Verifies exact `0.000e+00` H<->A equivariance across all stages and writes results to
   `experiments/outputs/v9_swarm/l4_multimodal_results.json`.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "scripts"), str(ROOT / "py_pkgs"), str(ROOT / "evaluation")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from exp_w7a_player_set_net import load_cached_physv2_table
from games import GAMES as GAME_INFO, anchor_video_frame
from s2_3_frame_models import plausibility
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, FPS, HOME, runs
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.eval import metrics
from src.v9_swarm.smart_tagging_arbitrator import (
    SmartTaggingArbitrator,
    extract_simulated_smart_tags_sample,
    extract_simulated_smart_tags_sportec,
    load_bar_ath_smart_tags,
    verify_smart_tagging_equivariance,
)
from src.v9_swarm.whistle_and_stoppage import (
    CollectiveFoulDeceleration,
    Fox40WhistleDetector,
    StoppageStartRefiner,
    verify_stoppage_refiner_equivariance,
)


def simulate_rolling_foul_late_dead(
    pred: np.ndarray,
    X_df,
    half: np.ndarray,
    sp_tags: list[dict],
    delay_frames: int = 45,
) -> tuple[np.ndarray, list[int]]:
    """Simulate BAR-ATH's rolling-ball foul stoppage artifact on Sportec / Metrica Sample:
    When an in-pitch foul stoppage occurs (followed by a FreeKick set piece, with collective
    player deceleration `decel <= -0.45` at the true `Live -> D` onset `t`), a naive detector
    that waits for the rolling ball to stop or leave the pitch predicts `Live -> D` at
    `t + delay_frames` (`+1.8s` late). Returns the degraded prediction and list of shifted `t`.
    """
    out = np.asarray(pred, dtype=np.int32).copy()
    half = np.asarray(half, dtype=np.int32)
    decel = np.nan_to_num(X_df["speed_mean_all_fwd_minus_bwd_2s"].to_numpy(dtype=np.float64), nan=0.0)
    foul_sigs = CollectiveFoulDeceleration.compute_signals(X_df, half)
    decel_1_5s = foul_sigs["decel_1_5s"]
    oob = np.nan_to_num(X_df["ball_outside_pitch"].to_numpy(dtype=np.float64), nan=0.0)

    fk_kicks = [int(sp["start"] + 5 * FPS) for sp in sp_tags if "free" in str(sp.get("type", "")).lower()]
    s_arr, e_arr, v_arr = runs(out)
    shifted_onsets = []
    for i in range(1, len(v_arr)):
        if int(v_arr[i]) != int(DEAD) or int(v_arr[i - 1]) not in (int(HOME), int(AWAY)):
            continue
        t = int(s_arr[i])
        e_d = int(e_arr[i])
        if (e_d - t) < delay_frames + 35 or half[t - 1] != half[t]:
            continue
        # Check if this stoppage is an in-pitch foul (either matched to a FreeKick or in-pitch with strong decel)
        is_fk = any(abs(k_f - e_d) <= 12 * FPS for k_f in fk_kicks)
        min_dec = float(np.minimum(decel[max(0, t - 15):t + 10], decel_1_5s[max(0, t - 15):t + 10]).min())
        if (is_fk or float(oob[max(0, t - 15):t + 5].mean()) == 0.0) and min_dec <= -0.55:
            out[t:t + delay_frames] = int(v_arr[i - 1])
            shifted_onsets.append(t)
    return out, shifted_onsets


def recover_rolling_foul_with_decel(
    pred_degraded: np.ndarray,
    proba: np.ndarray,
    X_df,
    half: np.ndarray,
    max_pullback: int = 65,
) -> np.ndarray:
    """Recover late `Live -> D` stoppages on `RollingFoul` streams using `CollectiveFoulDeceleration`."""
    out = np.asarray(pred_degraded, dtype=np.int32).copy()
    half = np.asarray(half, dtype=np.int32)
    decel = np.nan_to_num(X_df["speed_mean_all_fwd_minus_bwd_2s"].to_numpy(dtype=np.float64), nan=0.0)
    foul_sigs = CollectiveFoulDeceleration.compute_signals(X_df, half)
    decel_1_5s = foul_sigs["decel_1_5s"]

    s_arr, e_arr, v_arr = runs(out)
    for i in range(1, len(v_arr)):
        if int(v_arr[i]) != int(DEAD) or int(v_arr[i - 1]) not in (int(HOME), int(AWAY)):
            continue
        t = int(s_arr[i])
        e_d = int(e_arr[i])
        if (e_d - t) < 30 or (t - int(s_arr[i - 1])) < 25:
            continue
        w0 = max(int(s_arr[i - 1]) + 10, t - max_pullback)
        w1 = t - 5
        if w1 <= w0:
            continue
        # Find where DEAD probability and collective deceleration jointly onset
        pd_slice = proba[w0:t, DEAD]
        dec_slice = np.minimum(decel[w0:t], decel_1_5s[w0:t])
        cand_idx = np.flatnonzero((pd_slice >= 0.45) & (dec_slice <= -0.45))
        if len(cand_idx) > 0:
            new_t = w0 + int(cand_idx[0])
            out[new_t:t] = int(DEAD)
    return out


def main() -> int:
    t0 = time.time()
    print("=" * 125)
    print("SUBAGENT 3 — LINE L4 (WHISTLE + COLLECTIVE DECEL + STOPPAGE REFINER) & M5-v9 (SMART TAGGING ARBITRATOR)")
    print("=" * 125, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "v9_swarm"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = ROOT / "local_data" / "v9_swarm"
    local_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract and cache BAR-ATH whistle array
    whistle_npz_path = local_dir / "bar_ath_whistle.npz"
    detector = Fox40WhistleDetector()
    print("[1/5] Extracting & caching dual-stream Fox-40 referee whistle from BAR-ATH broadcast + tactical audio...", flush=True)
    w_res = detector.extract_and_cache_bar_ath_whistle(
        ROOT,
        cache_path=whistle_npz_path,
        cleanup_wav=True,
    )
    w_dual = w_res["w_dual"]
    print(
        f"      Cached {whistle_npz_path} | frames={len(w_dual)} | discrete whistle peaks={len(w_res['whistle_peaks'])}",
        flush=True,
    )

    # 2. Load cached v8.5 streams and feature tables
    print("[2/5] Loading v8.5 Grand Champion streams & PhysV2 tables...", flush=True)
    with open(local_dir / "v85_streams_and_preds.pkl", "rb") as f:
        cache = pickle.load(f)

    sportec = load_sportec()
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    X_v3 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", cache["v3"][mid]["half"]) for mid in MATCH_IDS}
    X_v6 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", cache["v6_std"][mid]["half"]) for mid in MATCH_IDS}
    X_sample = {gid: load_cached_physv2_table(f"sample_{gid}_raw", cache["sample"][gid]["half"]) for gid in GAMES}
    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)

    # Load Smart Tagging for BAR-ATH and simulated Smart Tagging for Sportec & Sample
    sp_tags_bar, phase_score_bar = load_bar_ath_smart_tags(ROOT, n_frames=atd.n_frames)
    sp_tags_sportec = {mid: extract_simulated_smart_tags_sportec(ROOT, mid, sportec[mid]) for mid in MATCH_IDS}
    sp_tags_sample = {gid: extract_simulated_smart_tags_sample(ROOT, gid) for gid in GAMES}

    # 3. Unsupervised Whistle Verification on BAR-ATH
    print("[3/5] Running unsupervised whistle enrichment verification on BAR-ATH...", flush=True)
    fk_starts = [int(sp["start"]) for sp in sp_tags_bar if "free" in str(sp.get("type", "")).lower() and atd.graded[sp["start"]]]
    unsup_whistle = Fox40WhistleDetector.evaluate_unsupervised_enrichment(
        w_dual,
        cache["atd_rep"]["pred"],
        X_atd_rep,
        atd.graded,
        sp_free_kick_starts=fk_starts,
    )
    print(
        f"      In-pitch stoppages={unsup_whistle['n_inpitch_stoppages']} | "
        f"Enrichment p99.0={unsup_whistle['enrichment_ratio_p99_0']:.2f}x "
        f"({100*unsup_whistle['hit_rate_inpitch_p99_0']:.1f}% vs {100*unsup_whistle['hit_rate_random_p99_0']:.1f}% random) | "
        f"Enrichment p99.8={unsup_whistle['enrichment_ratio_p99_8']:.2f}x | "
        f"v8.5 median late={unsup_whistle.get('median_v85_late_frames', 0):.1f}f "
        f"({unsup_whistle.get('median_v85_late_seconds', 0):.2f}s, max={unsup_whistle.get('max_v85_late_seconds', 0):.2f}s)",
        flush=True,
    )

    # 4. Verify 0.000e+00 H<->A equivariance on Sportec J03WMX and BAR-ATH
    print("[4/5] Verifying exact 0.000e+00 H<->A equivariance...", flush=True)
    m0 = "J03WMX"
    eq_l4_j1 = verify_stoppage_refiner_equivariance(
        cache["v3"][m0]["pred"],
        cache["v3"][m0]["proba"],
        cache["v3"][m0]["p4d"],
        X_v3[m0],
        cache["v3"][m0]["half"],
    )
    eq_m5_j1 = verify_smart_tagging_equivariance(
        cache["v3"][m0]["pred"],
        cache["v3"][m0]["proba"],
        cache["v3"][m0]["p4d"],
        X_v3[m0],
        cache["v3"][m0]["half"],
        sp_tags_sportec[m0],
    )
    eq_bar = verify_smart_tagging_equivariance(
        cache["atd_rep"]["pred"],
        cache["atd_rep"]["proba"],
        cache["atd_rep"]["p4d"],
        X_atd_rep,
        half_atd_dec,
        sp_tags_bar,
        phase_score=phase_score_bar,
        whistle_z=w_dual,
    )
    equivariance_report = {**eq_l4_j1, **eq_m5_j1, "bar_ath_full_multimodal_equivariance_error": eq_bar["combined_l4_m5_equivariance_error"]}
    print(f"      [EQUIVARIANCE] {equivariance_report}", flush=True)
    assert all(v == 0.0 for v in equivariance_report.values()), f"Equivariance failure: {equivariance_report}"

    # 5. Evaluate all configurations across J1A, J1B-Std, J1B-Gen, J2-Raw, J2-noD, RollingFoul, and BAR-ATH
    print("[5/5] Evaluating ablation table across all judges...", flush=True)
    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()

    def evaluate_variant(name: str, use_l4: bool, use_m5: bool) -> tuple[dict, np.ndarray]:
        res = {"name": name}
        for reg, X_dict, key_label in [
            ("v3", X_v3, "J1A_v3"),
            ("v6_std", X_v6, "J1B_Std"),
            ("v6_gen", X_v6, "J1B_Gen"),
        ]:
            f1_list = []
            f06_list = []
            f07_list = []
            per_match = {}
            for mid in MATCH_IDS:
                d = cache[reg][mid]
                y, hl, pr = d["labels"], d["half"], d["pred"].copy()
                p, p4d = d["proba"], d["p4d"]
                if use_l4:
                    pr = refiner.apply(
                        pr, p, p4d, X_dict[mid], hl,
                        enable_sliver_and_nocontact=(reg != "v6_gen"),
                    )
                if use_m5:
                    pr = arbitrator.apply(pr, p, X_dict[mid], hl, sp_tags_sportec[mid])
                sc = metrics.score(y, pr, hl)
                f1_list.append(float(sc["spell_f1_mean"]))
                f06_list.append(float(sc["spell_f1"][0.6]["f1"]))
                f07_list.append(float(sc["spell_f1"][0.7]["f1"]))
                per_match[mid] = float(sc["spell_f1_mean"])
            res[key_label] = float(np.mean(f1_list))
            res[f"{key_label}_06"] = float(np.mean(f06_list))
            res[f"{key_label}_07"] = float(np.mean(f07_list))
            res[f"per_match_{key_label}"] = per_match

        j2_raw, j2_nod = [], []
        for gid in GAMES:
            d = cache["sample"][gid]
            y, hl, pr = d["labels"], d["half"], d["pred"].copy()
            p, p4d = d["proba"], d["p4d"]
            if use_l4:
                pr = refiner.apply(pr, p, p4d, X_sample[gid], hl, enable_sliver_and_nocontact=True)
            if use_m5:
                pr = arbitrator.apply(pr, p, X_sample[gid], hl, sp_tags_sample[gid])
            j2_raw.append(float(metrics.score(y, pr, hl)["spell_f1_mean"]))
            j2_nod.append(float(metrics.score(without_dead_margin(y, hl), pr, hl)["spell_f1_mean"]))
        res["J2_Raw"] = float(np.mean(j2_raw))
        res["J2_noD"] = float(np.mean(j2_nod))
        res["J2_per_game_raw"] = j2_raw
        res["J2_per_game_nod"] = j2_nod

        # BAR-ATH full match + report-only J3
        pr_a = cache["atd_rep"]["pred"].copy()
        p_a = cache["atd_rep"]["proba"]
        p4d_a = cache["atd_rep"]["p4d"]
        if use_l4:
            pr_a = refiner.apply(
                pr_a, p_a, p4d_a, X_atd_rep, half_atd_dec,
                whistle_z=w_dual, enable_sliver_and_nocontact=True,
            )
        if use_m5:
            pr_a = arbitrator.apply(
                pr_a, p_a, X_atd_rep, half_atd_dec, sp_tags_bar,
                phase_score=phase_score_bar, whistle_z=w_dual,
            )
        pr_a[second_start:kick2] = DEAD
        pr_a[~atd.graded] = DEAD
        sc_pub = metrics.score_public(pr_a, bootstrap_n=0)
        pl_pub = plausibility(pr_a, atd.half, atd.graded)
        res["J3_Rep_F1"] = float(sc_pub["spell_f1_mean"])
        res["J3_Rep_MacroF1"] = float(sc_pub["macro_f1"])
        res["J3_alive_dead_f1"] = float(sc_pub["alive_dead_f1"])
        res["BAR_ATH_plausibility"] = {k: float(v) for k, v in pl_pub.items()}
        res["BAR_ATH_frames_changed_vs_v85"] = int(np.sum(pr_a != cache["atd_rep"]["pred"]))
        return res, pr_a

    r0, pr_a_base = evaluate_variant("0. v8.5 Grand Champion Baseline", False, False)
    r1, pr_a_l4 = evaluate_variant("1. v8.5 + Line L4 (StoppageStartRefiner + Whistle/Decel)", True, False)
    r2, pr_a_m5 = evaluate_variant("2. v8.5 + M5-v9 (SmartTaggingArbitrator)", False, True)
    r3, pr_a_both = evaluate_variant("3. v8.5 + Line L4 + M5-v9 (Full Multi-Modal Synthesis)", True, True)

    # Save refined BAR-ATH predictions for downstream ensemble synthesis
    np.save(local_dir / "bar_ath_pred_l4_only.npy", pr_a_l4)
    np.save(local_dir / "bar_ath_pred_l4_m5.npy", pr_a_both)

    # Also evaluate simulated RollingFoul stress test (where in-pitch foul D starts are delayed by 45 frames = 1.8s)
    rf_j1_deg, rf_j1_rec = [], []
    rf_j1_count = 0
    for mid in MATCH_IDS:
        d = cache["v3"][mid]
        y, hl = d["labels"], d["half"]
        pr_deg, shifted = simulate_rolling_foul_late_dead(d["pred"], X_v3[mid], hl, sp_tags_sportec[mid], delay_frames=45)
        rf_j1_count += len(shifted)
        pr_rec = recover_rolling_foul_with_decel(pr_deg, d["proba"], X_v3[mid], hl)
        pr_rec = refiner.apply(pr_rec, d["proba"], d["p4d"], X_v3[mid], hl, enable_sliver_and_nocontact=True)
        rf_j1_deg.append(float(metrics.score(y, pr_deg, hl)["spell_f1_mean"]))
        rf_j1_rec.append(float(metrics.score(y, pr_rec, hl)["spell_f1_mean"]))

    rf_j2_deg, rf_j2_rec = [], []
    rf_j2_count = 0
    for gid in GAMES:
        d = cache["sample"][gid]
        y, hl = d["labels"], d["half"]
        pr_deg, shifted = simulate_rolling_foul_late_dead(d["pred"], X_sample[gid], hl, sp_tags_sample[gid], delay_frames=45)
        rf_j2_count += len(shifted)
        pr_rec = recover_rolling_foul_with_decel(pr_deg, d["proba"], X_sample[gid], hl)
        pr_rec = refiner.apply(pr_rec, d["proba"], d["p4d"], X_sample[gid], hl, enable_sliver_and_nocontact=True)
        rf_j2_deg.append(float(metrics.score(y, pr_deg, hl)["spell_f1_mean"]))
        rf_j2_rec.append(float(metrics.score(y, pr_rec, hl)["spell_f1_mean"]))

    rolling_foul_stress = {
        "J1A_RollingFoul_n_stoppages": rf_j1_count,
        "J1A_RollingFoul_degraded": float(np.mean(rf_j1_deg)),
        "J1A_RollingFoul_recovered": float(np.mean(rf_j1_rec)),
        "J1A_RollingFoul_gain": float(np.mean(rf_j1_rec) - np.mean(rf_j1_deg)),
        "J2_RollingFoul_n_stoppages": rf_j2_count,
        "J2_RollingFoul_degraded": float(np.mean(rf_j2_deg)),
        "J2_RollingFoul_recovered": float(np.mean(rf_j2_rec)),
        "J2_RollingFoul_gain": float(np.mean(rf_j2_rec) - np.mean(rf_j2_deg)),
    }

    print("\n" + "-" * 135)
    print(f"{'Configuration':56s} | {'J1A (v3)':10s} | {'J1B-Std':10s} | {'J1B-Gen':10s} | {'J2-Raw':10s} | {'J2-noD':10s} | {'J3 (MacF1, dF)'}")
    print("-" * 135)
    for r in (r0, r1, r2, r3):
        d_j1a = r["J1A_v3"] - r0["J1A_v3"]
        d_j1s = r["J1B_Std"] - r0["J1B_Std"]
        d_j1g = r["J1B_Gen"] - r0["J1B_Gen"]
        d_j2r = r["J2_Raw"] - r0["J2_Raw"]
        d_j2n = r["J2_noD"] - r0["J2_noD"]
        print(
            f"{r['name']:56s} | {r['J1A_v3']:.6f} ({d_j1a:+.5f}) | "
            f"{r['J1B_Std']:.6f} ({d_j1s:+.5f}) | "
            f"{r['J1B_Gen']:.6f} ({d_j1g:+.5f}) | "
            f"{r['J2_Raw']:.6f} ({d_j2r:+.5f}) | "
            f"{r['J2_noD']:.6f} ({d_j2n:+.5f}) | "
            f"{r['J3_Rep_F1']:.4f} ({r['J3_Rep_MacroF1']:.4f}, {r['BAR_ATH_frames_changed_vs_v85']}f)"
        )
    print("-" * 135)
    print(
        f"[RollingFoul Stress Test] J1A ({rf_j1_count} fouls): {rolling_foul_stress['J1A_RollingFoul_degraded']:.6f} -> "
        f"{rolling_foul_stress['J1A_RollingFoul_recovered']:.6f} ({rolling_foul_stress['J1A_RollingFoul_gain']:+.6f}) | "
        f"J2 ({rf_j2_count} fouls): {rolling_foul_stress['J2_RollingFoul_degraded']:.6f} -> "
        f"{rolling_foul_stress['J2_RollingFoul_recovered']:.6f} ({rolling_foul_stress['J2_RollingFoul_gain']:+.6f})"
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - t0),
        "equivariance": equivariance_report,
        "unsupervised_bar_ath_whistle": unsup_whistle,
        "rolling_foul_stress_test": rolling_foul_stress,
        "ablation_rows": [r0, r1, r2, r3],
        "deltas_vs_v85": {
            "L4_only": {
                "J1A_v3": float(r1["J1A_v3"] - r0["J1A_v3"]),
                "J1B_Std": float(r1["J1B_Std"] - r0["J1B_Std"]),
                "J1B_Gen": float(r1["J1B_Gen"] - r0["J1B_Gen"]),
                "J2_Raw": float(r1["J2_Raw"] - r0["J2_Raw"]),
                "J2_noD": float(r1["J2_noD"] - r0["J2_noD"]),
            },
            "M5_only": {
                "J1A_v3": float(r2["J1A_v3"] - r0["J1A_v3"]),
                "J1B_Std": float(r2["J1B_Std"] - r0["J1B_Std"]),
                "J1B_Gen": float(r2["J1B_Gen"] - r0["J1B_Gen"]),
                "J2_Raw": float(r2["J2_Raw"] - r0["J2_Raw"]),
                "J2_noD": float(r2["J2_noD"] - r0["J2_noD"]),
            },
            "L4_plus_M5": {
                "J1A_v3": float(r3["J1A_v3"] - r0["J1A_v3"]),
                "J1B_Std": float(r3["J1B_Std"] - r0["J1B_Std"]),
                "J1B_Gen": float(r3["J1B_Gen"] - r0["J1B_Gen"]),
                "J2_Raw": float(r3["J2_Raw"] - r0["J2_Raw"]),
                "J2_noD": float(r3["J2_noD"] - r0["J2_noD"]),
            },
        },
    }
    json_path = out_dir / "l4_multimodal_results.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved complete JSON results to {json_path} in {time.time() - t0:.1f}s.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
