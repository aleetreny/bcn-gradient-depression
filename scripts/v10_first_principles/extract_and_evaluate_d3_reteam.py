#!/usr/bin/env python3
"""Subagent D (`v10_first_principles`): Extract & Verify True `D3` Re-Teamed Streams
(`_local/v85sc-count` and `_local/v85sc-all`) + Full `v9`/`v10` Stack Evaluation.

1. Extracts the true re-teamed `BAR-ATH` (`atd_rep` and `atd_raw`) probability streams
   `(p, c, p4d, fams)` and `X_atd_rep` from `_local/v85sc-control/`, `_local/v85sc-count/`,
   and `_local/v85sc-all/`, along with the Modal 3D aerial/duel heads (`atd_reteam_rep`),
   and caches them in `experiments/outputs/v10_first_principles/d3_reteamed_bar_ath_streams.pkl`.
2. Verifies 100% byte-for-byte SHA-256 reproduction of all 6 `v8.5` submissions
   (`strict_cv` and `grand_champion` across `control`, `count`, and `all`) AND both existing
   `v9` submissions (`submission_v9_super_champion_strict_cv.csv` and
   `submission_v9_super_champion_reteam.csv`).
3. Applies the full `v9` upgrade stack (`4A + 4B + L4 + M5-v9`) and `v10` Set-Piece Post-Kick
   Micro-Sliver Bridge (`apply_d3_post_kick_sliver_bridge`) to the TRUE re-teamed streams
   and compares `12v10` swap stretches fixed, `SmartTagging` agreement (`in_phase_team_agree`,
   `pre_kick_D`, `post_kick_team_alive`), `poss/min`, `% < 2s`, and episode-level physical coherence.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "py_pkgs"), str(ROOT), str(ROOT / "scripts"), str(ROOT / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from games import GAMES as GAME_INFO, anchor_video_frame  # noqa: E402
from scripts.exp_v6_super_champion import save_and_validate_submission  # noqa: E402
from src.data.common import DEAD  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all as load_sportec  # noqa: E402
from src.v9_swarm.duel_and_mbr_v9 import (  # noqa: E402
    SymDomainTransferCalibratorV9,
    SymDuelTurnoverArbitratorV9,
    SymSpellMBRDecoderV9,
    verify_v9_duel_mbr_equivariance,
)
from src.v9_swarm.j_suite import (  # noqa: E402
    SWARM_OUT_DIR,
    _load_or_build_smart_tagging_cache,
    evaluate_bar_ath_unsupervised,
)
from src.v9_swarm.smart_tagging_arbitrator import (  # noqa: E402
    SmartTaggingArbitrator,
    load_bar_ath_smart_tags,
    verify_smart_tagging_equivariance,
)
from src.v9_swarm.whistle_and_stoppage import (  # noqa: E402
    Fox40WhistleDetector,
    StoppageStartRefiner,
    verify_stoppage_refiner_equivariance,
)
from src.v10_first_principles.d3_reteam_and_repro import (  # noqa: E402
    apply_d3_post_kick_sliver_bridge,
    compute_d3_reteam_anatomy,
    compute_episode_physical_coherence,
    decode_v85_bar_ath,
    extract_bar_ath_streams_from_worktree,
    load_v85_fitted_shield_models,
    sha256_file,
    verify_d3_sliver_bridge_equivariance,
)


EXPECTED_SHA256 = {
    "v85_control_strict_cv": "2c834b92408ead731ff18be2713b94b4710df082bb9712796ca7668c2effc329",
    "v85_control_grand_champion": "f923326ac28f3ac4590aa1de98e8a1211b3db281d6bf79ec5251b9deb6595309",
    "v85_count_strict_cv": "b6c7974d4dd6cd5576b7d18098291d1e6b08ea25b2d54199c8bae95fe5bbb506",
    "v85_count_grand_champion": "b61c6ef03da3eb8101aa0895f8899c576b5a7d1f20f1b3e7aa57fa91e1eaf8b1",
    "v85_all_strict_cv": "74d6087cdfee739bde4985cecc060364010f77a3090951c043e46e605a49d45f",
    "v85_all_grand_champion": "d4c63f5958e39481a63868a8d8f92ad0724b1e14fbec3723eb0e3c287c1edc2c",
    "v9_control_strict_cv": "ecb410c2b9bfc3f8557d09c789aa114ef183a0a97c2dfe61c10490d051db8fdc",
    "v9_reteam_unreteamed_bug": "1fca00d3d0dbdc01b4305a89948556b16990223a44175f7a3dc23d6ac09a4372",
}


def main() -> None:
    t0 = time.time()
    print("=" * 130)
    print("SUBAGENT D (`v10_first_principles`): TRUE D3 VIDEO RE-TEAMED STREAMS + BYTE-FOR-BYTE v8.5 & v9/v10 SYNTHESIS")
    print("=" * 130, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "v10_first_principles"
    sub_dir = out_dir / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    # 1. Extract streams from repo-local local_data/d3_reteam_cache
    print("\n[STEP 1] Extracting true BAR-ATH (`atd_rep` & `atd_raw`) 9-family streams & PhysV2 tables...", flush=True)
    worktrees = {
        "control": ROOT,
        "count": ROOT / "local_data" / "d3_reteam_cache" / "count",
        "all": ROOT / "local_data" / "d3_reteam_cache" / "all",
    }
    extracted: dict[str, dict] = {}
    for wt_name, wt_path in worktrees.items():
        t_w = time.time()
        extracted[wt_name] = extract_bar_ath_streams_from_worktree(wt_path, ROOT, atd)
        p_rep = extracted[wt_name]["hyb"]["atd_rep"][0]
        print(
            f"  [{wt_name:7s}] Extracted from {wt_path.relative_to(ROOT) if wt_path != ROOT else '.'} "
            f"in {time.time() - t_w:.2f}s | p_rep shape={p_rep.shape}, mean={np.round(p_rep.mean(axis=0), 5).tolist()}",
            flush=True,
        )

    # Also load Modal 3D v9 heads (`atd_rep` AND `atd_reteam_rep`)
    with open(SWARM_OUT_DIR / "pset_v2_oof" / "pset_v2_j23_preds.pkl", "rb") as f:
        pset_v2_j23 = pickle.load(f)
    with open(SWARM_OUT_DIR / "l1_trees_oof" / "l1_trees_j23_preds.pkl", "rb") as f:
        l1_j23 = pickle.load(f)

    v9_heads = {
        "control": {
            "air": 0.5 * (
                np.asarray(pset_v2_j23["atd_rep"]["pset_p_aerial"], dtype=np.float64)
                + np.asarray(l1_j23["atd_rep"]["air"], dtype=np.float64)
            ),
            "duel": np.asarray(l1_j23["atd_rep"]["duel"], dtype=np.float64),
        },
        "reteam_count": {
            "air": 0.5 * (
                np.asarray(pset_v2_j23["atd_reteam_rep"]["pset_p_aerial"], dtype=np.float64)
                + np.asarray(l1_j23["atd_reteam_rep"]["air"], dtype=np.float64)
            ),
            "duel": np.asarray(l1_j23["atd_reteam_rep"]["duel"], dtype=np.float64),
        },
    }
    print(
        f"  Loaded Modal 3D v9 heads (`atd_rep` vs `atd_reteam_rep`): "
        f"max |air_diff|={np.max(np.abs(v9_heads['control']['air'] - v9_heads['reteam_count']['air'])):.5f}, "
        f"max |duel_diff|={np.max(np.abs(v9_heads['control']['duel'] - v9_heads['reteam_count']['duel'])):.5f}",
        flush=True,
    )

    # Save cached stream bundle to experiments/outputs/v10_first_principles/d3_reteamed_bar_ath_streams.pkl
    streams_pkl_path = out_dir / "d3_reteamed_bar_ath_streams.pkl"
    cache_payload = {
        "control": {
            "atd_rep": extracted["control"]["hyb"]["atd_rep"],
            "atd_raw": extracted["control"]["hyb"]["atd_raw"],
            "atd_rep_inp": extracted["control"]["inp"]["atd_rep"],
            "v9_air": v9_heads["control"]["air"],
            "v9_duel": v9_heads["control"]["duel"],
            "worktree_path": str(worktrees["control"]),
        },
        "count": {
            "atd_rep": extracted["count"]["hyb"]["atd_rep"],
            "atd_raw": extracted["count"]["hyb"]["atd_raw"],
            "atd_rep_inp": extracted["count"]["inp"]["atd_rep"],
            "v9_air": v9_heads["reteam_count"]["air"],
            "v9_duel": v9_heads["reteam_count"]["duel"],
            "worktree_path": str(worktrees["count"]),
        },
        "all": {
            "atd_rep": extracted["all"]["hyb"]["atd_rep"],
            "atd_raw": extracted["all"]["hyb"]["atd_raw"],
            "atd_rep_inp": extracted["all"]["inp"]["atd_rep"],
            "v9_air": v9_heads["reteam_count"]["air"],
            "v9_duel": v9_heads["reteam_count"]["duel"],
            "worktree_path": str(worktrees["all"]),
        },
    }
    with open(streams_pkl_path, "wb") as f:
        pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved cached D3 streams bundle to {streams_pkl_path.relative_to(ROOT)} ({streams_pkl_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # 2. Verify 100% byte-for-byte SHA-256 reproduction of all 6 v8.5 CSVs
    print("\n[STEP 2] Verifying 100% Byte-for-Byte SHA-256 Reproduction of v8.5 Submissions (`control`, `count`, `all`)...", flush=True)
    v85_bundle = load_v85_fitted_shield_models(ROOT)
    v85_preds: dict[str, np.ndarray] = {}
    sha_verification: dict[str, dict] = {}

    for wt_name in ("control", "count", "all"):
        st_rep = extracted[wt_name]["hyb"]["atd_rep"]
        X_rep = extracted[wt_name]["X_atd_rep"]
        for mode_suffix, use_pit in [("strict_cv", False), ("grand_champion", True)]:
            key = f"v85_{wt_name}_{mode_suffix}"
            pred_arr, _ = decode_v85_bar_ath(st_rep, X_rep, atd, v85_bundle, s_tag="hyb", use_pit_domino=use_pit)
            v85_preds[key] = pred_arr
            csv_path = sub_dir / f"submission_{key}.csv"
            save_and_validate_submission(csv_path, pred_arr, atd, second_start, kick2)
            actual_sha = sha256_file(csv_path)
            expected_sha = EXPECTED_SHA256[key]
            exact_match = actual_sha == expected_sha
            sha_verification[key] = {
                "csv_path": str(csv_path.relative_to(ROOT)),
                "actual_sha256": actual_sha,
                "expected_sha256": expected_sha,
                "exact_byte_match": exact_match,
            }
            print(f"  [{key:28s}] SHA-256={actual_sha[:16]}... | exact_byte_match={exact_match}", flush=True)
            assert exact_match, f"SHA-256 mismatch on {key}: {actual_sha} != {expected_sha}"

    # 3. Fit v9 modules and verify 0.000e+00 H<->A equivariance
    print("\n[STEP 3] Fitting 7-match `v9` modules (`4A + 4B + L4 + M5-v9 + 4C`) & verifying `0.000e+00` H<->A equivariance...", flush=True)
    with open(SWARM_OUT_DIR / "_diag_enr_cands.pkl", "rb") as f:
        enr = pickle.load(f)
    duel_cands = enr["duel"]
    mbr_cands = enr["mbr"]
    mids = sorted(MATCH_IDS)
    gids = sorted(mbr_cands["j2"].keys())

    duel_full = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][m] for m in mids])
    mbr_dec = SymSpellMBRDecoderV9()
    dom_cal_full = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["j2"][g] for g in gids])
    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()

    whistle_path = ROOT / "local_data" / "v9_swarm" / "bar_ath_whistle.npz"
    detector = Fox40WhistleDetector()
    w_res = detector.extract_and_cache_bar_ath_whistle(ROOT, cache_path=whistle_path, cleanup_wav=True)
    w_dual = w_res["w_dual"]
    sp_tags_bar, phase_score_bar = load_bar_ath_smart_tags(ROOT, n_frames=atd.n_frames)

    p_rt_c, c_rt_c, p4d_rt_c, fams_rt_c = extracted["count"]["hyb"]["atd_rep"]
    X_rt_c = extracted["count"]["X_atd_rep"]
    eq_4abc = verify_v9_duel_mbr_equivariance(
        v85_preds["v85_count_strict_cv"],
        p_rt_c.astype(np.float64),
        c_rt_c.astype(np.float64),
        p4d_rt_c.astype(np.float64),
        [fp.astype(np.float64) for fp in fams_rt_c],
        half_atd_dec,
        X_rt_c,
        duel_full,
        mbr_dec,
        dom_cal_full,
        v9_air=v9_heads["reteam_count"]["air"],
        v9_duel=v9_heads["reteam_count"]["duel"],
    )
    eq_l4 = verify_stoppage_refiner_equivariance(
        v85_preds["v85_count_strict_cv"],
        p_rt_c,
        p4d_rt_c,
        X_rt_c,
        half_atd_dec,
    )
    eq_m5 = verify_smart_tagging_equivariance(
        v85_preds["v85_count_strict_cv"],
        p_rt_c,
        p4d_rt_c,
        X_rt_c,
        half_atd_dec,
        sp_tags_bar,
        phase_score=phase_score_bar,
        whistle_z=w_dual,
    )
    eq_bridge = verify_d3_sliver_bridge_equivariance(
        v85_preds["v85_count_strict_cv"],
        p_rt_c,
        X_rt_c,
        half_atd_dec,
        sp_tags_bar,
        phase_score_bar,
    )
    equivariance_summary = {
        "v9_4abc_equivariance_error": float(eq_4abc),
        **{k: float(v) for k, v in eq_l4.items()},
        **{k: float(v) for k, v in eq_m5.items()},
        "d3_post_kick_sliver_bridge_equivariance_error": float(eq_bridge),
    }
    print(f"  [EQUIVARIANCE CHECK] {equivariance_summary}", flush=True)
    assert all(v == 0.0 for v in equivariance_summary.values()), f"Equivariance failure: {equivariance_summary}"

    def _run_v9_stack(
        base_pred: np.ndarray,
        stream_tuple: tuple,
        X_df,
        v9_air: np.ndarray,
        v9_duel: np.ndarray,
        use_sliver_bridge: bool = False,
        precomputed_duel_cands=None,
        precomputed_mbr_cands=None,
    ) -> dict[str, np.ndarray]:
        p_s, c_s, p4d_s, fams_s = stream_tuple
        p64 = p_s.astype(np.float64)
        c64 = c_s.astype(np.float64)
        p4d64 = p4d_s.astype(np.float64)
        fams64 = [fp.astype(np.float64) for fp in fams_s]

        if precomputed_duel_cands is not None:
            pr_4a, f_duel = duel_full.apply_from_candidates(base_pred.copy(), precomputed_duel_cands)
        else:
            pr_4a, f_duel = duel_full.apply(base_pred.copy(), p64, c64, p4d64, fams64, half_atd_dec, X_df)

        if precomputed_mbr_cands is not None:
            pr_4ab, f_mbr = mbr_dec.apply_from_candidates(pr_4a.copy(), precomputed_mbr_cands)
        else:
            pr_4ab, f_mbr = mbr_dec.apply(
                pr_4a.copy(), p64, c64, p4d64, fams64, half_atd_dec, X_df, v9_air=v9_air, v9_duel=v9_duel
            )

        pr_l4 = refiner.apply(
            pr_4ab.copy(), p_s, p4d_s, X_df, half_atd_dec, whistle_z=w_dual, enable_sliver_and_nocontact=True
        )
        pr_m5 = arbitrator.apply(
            pr_l4.copy(), p_s, X_df, half_atd_dec, sp_tags_bar, phase_score=phase_score_bar, whistle_z=w_dual
        )
        if use_sliver_bridge:
            pr_final = apply_d3_post_kick_sliver_bridge(
                pr_m5.copy(), p_s, X_df, half_atd_dec, sp_tags_bar, phase_score=phase_score_bar
            )
        else:
            pr_final = pr_m5.copy()

        for arr in (pr_4a, pr_4ab, pr_l4, pr_m5, pr_final):
            arr[second_start:kick2] = DEAD
            arr[~atd.graded] = DEAD
        return {
            "after_4a": pr_4a,
            "after_4ab": pr_4ab,
            "after_l4": pr_l4,
            "after_m5": pr_m5,
            "final": pr_final,
            "n_duel_fires": len(f_duel),
            "n_mbr_fires": len(f_mbr),
        }

    print("\n[STEP 4] Running `v9`/`v10` stacks on Control, Un-Reteamed Bug, and TRUE `D3` Re-Teamed Streams...", flush=True)
    st_ctrl = extracted["control"]["hyb"]["atd_rep"]
    X_ctrl = extracted["control"]["X_atd_rep"]
    st_count = extracted["count"]["hyb"]["atd_rep"]
    X_count = extracted["count"]["X_atd_rep"]
    st_all = extracted["all"]["hyb"]["atd_rep"]
    X_all = extracted["all"]["X_atd_rep"]

    # Variant A: v9 on Control (reproduces submission_v9_super_champion_strict_cv.csv)
    res_v9_ctrl = _run_v9_stack(
        v85_preds["v85_control_strict_cv"],
        st_ctrl,
        X_ctrl,
        v9_heads["control"]["air"],
        v9_heads["control"]["duel"],
        precomputed_duel_cands=duel_cands["atd_rep"],
        precomputed_mbr_cands=mbr_cands["atd_rep"],
    )
    p_v9_ctrl_csv = sub_dir / "submission_v9_control_strict_cv.csv"
    save_and_validate_submission(p_v9_ctrl_csv, res_v9_ctrl["final"], atd, second_start, kick2)
    sha_v9_ctrl = sha256_file(p_v9_ctrl_csv)
    sha_verification["v9_control_strict_cv"] = {
        "csv_path": str(p_v9_ctrl_csv.relative_to(ROOT)),
        "actual_sha256": sha_v9_ctrl,
        "expected_sha256": EXPECTED_SHA256["v9_control_strict_cv"],
        "exact_byte_match": sha_v9_ctrl == EXPECTED_SHA256["v9_control_strict_cv"],
    }
    print(
        f"  [v9_control_strict_cv        ] SHA-256={sha_v9_ctrl[:16]}... | "
        f"exact_byte_match={sha_verification['v9_control_strict_cv']['exact_byte_match']}",
        flush=True,
    )
    assert sha_verification["v9_control_strict_cv"]["exact_byte_match"]

    # Variant B: v9 on v85_reteam_count with UN-RETEAMED streams (reproduces submission_v9_super_champion_reteam.csv)
    res_v9_unrt_bug = _run_v9_stack(
        v85_preds["v85_count_strict_cv"],
        st_ctrl,
        X_ctrl,
        v9_heads["control"]["air"],
        v9_heads["control"]["duel"],
    )
    p_v9_unrt_csv = sub_dir / "submission_v9_reteam_unreteamed_bug.csv"
    save_and_validate_submission(p_v9_unrt_csv, res_v9_unrt_bug["final"], atd, second_start, kick2)
    sha_v9_unrt = sha256_file(p_v9_unrt_csv)
    sha_verification["v9_reteam_unreteamed_bug"] = {
        "csv_path": str(p_v9_unrt_csv.relative_to(ROOT)),
        "actual_sha256": sha_v9_unrt,
        "expected_sha256": EXPECTED_SHA256["v9_reteam_unreteamed_bug"],
        "exact_byte_match": sha_v9_unrt == EXPECTED_SHA256["v9_reteam_unreteamed_bug"],
    }
    print(
        f"  [v9_reteam_unreteamed_bug    ] SHA-256={sha_v9_unrt[:16]}... | "
        f"exact_byte_match={sha_verification['v9_reteam_unreteamed_bug']['exact_byte_match']}",
        flush=True,
    )
    assert sha_verification["v9_reteam_unreteamed_bug"]["exact_byte_match"]

    # Variant C: v9 on v85_reteam_count with TRUE re-teamed streams (`v85sc-count` + `atd_reteam_rep` Modal 3D heads)
    res_v9_count_true = _run_v9_stack(
        v85_preds["v85_count_strict_cv"],
        st_count,
        X_count,
        v9_heads["reteam_count"]["air"],
        v9_heads["reteam_count"]["duel"],
    )
    p_v9_count_true_csv = sub_dir / "submission_v9_d3_reteam_count_true.csv"
    save_and_validate_submission(p_v9_count_true_csv, res_v9_count_true["final"], atd, second_start, kick2)

    # Variant D: v9 on v85_reteam_count Grand Champion (w/ PIT Domino) + TRUE re-teamed streams
    res_v9_count_grand_true = _run_v9_stack(
        v85_preds["v85_count_grand_champion"],
        st_count,
        X_count,
        v9_heads["reteam_count"]["air"],
        v9_heads["reteam_count"]["duel"],
    )
    p_v9_count_grand_csv = sub_dir / "submission_v9_d3_reteam_count_grand_true.csv"
    save_and_validate_submission(p_v9_count_grand_csv, res_v9_count_grand_true["final"], atd, second_start, kick2)

    # Variant E: v9 on v85_reteam_all with TRUE re-teamed streams (`v85sc-all`)
    res_v9_all_true = _run_v9_stack(
        v85_preds["v85_all_strict_cv"],
        st_all,
        X_all,
        v9_heads["reteam_count"]["air"],
        v9_heads["reteam_count"]["duel"],
    )
    p_v9_all_true_csv = sub_dir / "submission_v9_d3_reteam_all_true.csv"
    save_and_validate_submission(p_v9_all_true_csv, res_v9_all_true["final"], atd, second_start, kick2)

    # Variant F: v10 on v85_reteam_count with TRUE re-teamed streams + Set-Piece Post-Kick Sliver Bridge
    res_v10_count_bridged = _run_v9_stack(
        v85_preds["v85_count_strict_cv"],
        st_count,
        X_count,
        v9_heads["reteam_count"]["air"],
        v9_heads["reteam_count"]["duel"],
        use_sliver_bridge=True,
    )
    p_v10_count_bridged_csv = sub_dir / "submission_v10_d3_reteam_count_true_bridged.csv"
    save_and_validate_submission(p_v10_count_bridged_csv, res_v10_count_bridged["final"], atd, second_start, kick2)

    # Variant G: v10 on v85_reteam_all with TRUE re-teamed streams + Set-Piece Post-Kick Sliver Bridge
    res_v10_all_bridged = _run_v9_stack(
        v85_preds["v85_all_strict_cv"],
        st_all,
        X_all,
        v9_heads["reteam_count"]["air"],
        v9_heads["reteam_count"]["duel"],
        use_sliver_bridge=True,
    )
    p_v10_all_bridged_csv = sub_dir / "submission_v10_d3_reteam_all_true_bridged.csv"
    save_and_validate_submission(p_v10_all_bridged_csv, res_v10_all_bridged["final"], atd, second_start, kick2)

    # 4. Compute D3 anatomy (`12v10` stretches fixed) and evaluate all variants on BAR-ATH
    print("\n[STEP 5] Computing `D3` `12v10+` swap anatomy & full-match BAR-ATH physical/tactical coherence...", flush=True)
    d3_anatomy = compute_d3_reteam_anatomy(atd, ROOT)
    st_cache = _load_or_build_smart_tagging_cache(atd)
    reteam_mask = st_cache["reteam_near_mask"]

    variants_to_eval = [
        ("1. v8.5 Control (`strict_cv`)", v85_preds["v85_control_strict_cv"], X_ctrl, sub_dir / "submission_v85_control_strict_cv.csv"),
        ("2. v8.5 D3 Reteam Count (`v85sc-count`)", v85_preds["v85_count_strict_cv"], X_count, sub_dir / "submission_v85_count_strict_cv.csv"),
        ("3. v8.5 D3 Reteam All (`v85sc-all`)", v85_preds["v85_all_strict_cv"], X_all, sub_dir / "submission_v85_all_strict_cv.csv"),
        ("4. v9 Control (`4A+4B+L4+M5`)", res_v9_ctrl["final"], X_ctrl, p_v9_ctrl_csv),
        ("5. v9 Reteam Count (Un-Reteamed Streams Bug)", res_v9_unrt_bug["final"], X_count, p_v9_unrt_csv),
        ("6. v9 Reteam Count (TRUE Reteamed Streams)", res_v9_count_true["final"], X_count, p_v9_count_true_csv),
        ("7. v9 Reteam Count Grand (TRUE + PIT Domino)", res_v9_count_grand_true["final"], X_count, p_v9_count_grand_csv),
        ("8. v9 Reteam All (TRUE Reteamed Streams)", res_v9_all_true["final"], X_all, p_v9_all_true_csv),
        ("9. v10 Reteam Count (TRUE + Post-Kick Bridge)", res_v10_count_bridged["final"], X_count, p_v10_count_bridged_csv),
        ("10. v10 Reteam All (TRUE + Post-Kick Bridge)", res_v10_all_bridged["final"], X_all, p_v10_all_bridged_csv),
    ]

    eval_table = {}
    print("\n" + "-" * 142)
    print(
        f"{'Variant':45s} | {'Graded p/m':10s} | {'% <2s':6s} | {'PhaseAg%':8s} | "
        f"{'PreKickD%':9s} | {'PostKick%':9s} | {'<34f Spells':11s} | {'ClearProx%':10s} | {'Diff vs v85_ctrl'}"
    )
    print("-" * 142, flush=True)

    for label, pr_arr, X_df_use, csv_p in variants_to_eval:
        unsup = evaluate_bar_ath_unsupervised(
            pr_arr,
            atd=atd,
            v85_strict_pred=v85_preds["v85_control_strict_cv"],
            v85_reteam_pred=v85_preds["v85_count_strict_cv"],
        )
        ep_coh = compute_episode_physical_coherence(pr_arr, X_df_use, atd, reteam_mask=reteam_mask)
        diff_vs_ctrl = int(np.sum((pr_arr != v85_preds["v85_control_strict_cv"]) & atd.graded))
        diff_vs_rt_v85 = int(np.sum((pr_arr != v85_preds["v85_count_strict_cv"]) & atd.graded))
        diff_vs_unrt_bug = int(np.sum((pr_arr != res_v9_unrt_bug["final"]) & atd.graded))

        fg = unsup["plausibility"]["full_graded_158857f"]
        st_m = unsup["smart_tagging_consistency"]["full_graded_158857f"]
        st_lag = unsup["smart_tagging_consistency"]["physical_kick_restart_alignment"]
        rec = {
            "csv_path": str(csv_p.relative_to(ROOT)),
            "sha256": sha256_file(csv_p),
            "full_graded_poss_per_min": fg["poss_per_min"],
            "full_graded_pct_lt_2s": fg["pct_lt_2s"],
            "full_graded_pct_D": fg["pct_D"],
            "unseen_private_poss_per_min": unsup["plausibility"]["unseen_private_151357f"]["poss_per_min"],
            "unseen_private_pct_lt_2s": unsup["plausibility"]["unseen_private_151357f"]["pct_lt_2s"],
            "smart_tagging_in_phase_team_agree_pct": st_m["in_phase_live_team_agree_pct"],
            "smart_tagging_pre_kick_dead_agree_pct": st_m["set_piece_pre_kick_D_agree_pct"],
            "smart_tagging_post_kick_team_alive_pct": st_m["set_piece_post_kick_team_when_alive_pct"],
            "smart_tagging_post_kick_team_strict_pct": st_m["set_piece_post_kick_team_strict_pct"],
            "smart_tagging_restart_lag_median_f": st_lag["median_restart_lag_f"],
            "episode_coherence": ep_coh,
            "reference_agreements": unsup["reference_agreements"],
            "diff_frames_vs_v85_ctrl": diff_vs_ctrl,
            "diff_frames_vs_v85_reteam_count": diff_vs_rt_v85,
            "diff_frames_vs_v9_unreteamed_bug": diff_vs_unrt_bug,
            "j3_public_report_only": unsup["J3_public_5min_REPORT_ONLY"],
        }
        eval_table[label] = rec
        print(
            f"{label:45s} | {fg['poss_per_min']:10.2f} | {fg['pct_lt_2s']:5.2f}% | "
            f"{st_m['in_phase_live_team_agree_pct']:7.2f}% | {st_m['set_piece_pre_kick_D_agree_pct']:8.2f}% | "
            f"{st_m['set_piece_post_kick_team_when_alive_pct']:8.2f}% | "
            f"{ep_coh['n_sub_34f_micro_spells']:11d} | {ep_coh['clear_proximity_team_agree_pct']:9.2f}% | "
            f"{diff_vs_ctrl:5d}f (vs rt:{diff_vs_rt_v85:4d}f)",
            flush=True,
        )
    print("-" * 142, flush=True)

    # Compare stage-by-stage frame diffs between Un-Reteamed Streams Bug and TRUE Reteamed Streams on v85_reteam_count
    stage_diffs = {}
    for stg in ("after_4a", "after_4ab", "after_l4", "after_m5", "final"):
        d_unrt_vs_base = int(np.sum((res_v9_unrt_bug[stg] != v85_preds["v85_count_strict_cv"]) & atd.graded))
        d_true_vs_base = int(np.sum((res_v9_count_true[stg] != v85_preds["v85_count_strict_cv"]) & atd.graded))
        d_bridged_vs_base = int(np.sum((res_v10_count_bridged[stg] != v85_preds["v85_count_strict_cv"]) & atd.graded))
        d_true_vs_unrt = int(np.sum((res_v9_count_true[stg] != res_v9_unrt_bug[stg]) & atd.graded))
        stage_diffs[stg] = {
            "unreteamed_bug_diff_vs_v85_reteam_count": d_unrt_vs_base,
            "true_streams_diff_vs_v85_reteam_count": d_true_vs_base,
            "true_bridged_diff_vs_v85_reteam_count": d_bridged_vs_base,
            "true_vs_unreteamed_bug_diff_frames": d_true_vs_unrt,
        }
    print("\n[STEP 6] Stage-by-Stage Frame Diffs on `v85_reteam_count` (Un-Reteamed Bug vs TRUE Reteamed Streams):", flush=True)
    for stg, srec in stage_diffs.items():
        print(f"  {stg:10s}: {srec}", flush=True)

    full_report = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "streams_cache_path": str(streams_pkl_path.relative_to(ROOT)),
        "sha256_verification": sha_verification,
        "equivariance_verification": equivariance_summary,
        "d3_reteam_anatomy": d3_anatomy,
        "stage_by_stage_unreteamed_vs_true": stage_diffs,
        "variants_evaluation": eval_table,
    }
    json_path = out_dir / "d3_reteam_v9_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved complete D3 + v9/v10 evaluation JSON to {json_path.relative_to(ROOT)} in {time.time() - t0:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
