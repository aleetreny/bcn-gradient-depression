"""Deep Physics & Pipeline Stage Autopsy of Super-Champion v7 (`deep_physics_autopsy.py`).

Answers 5 critical forensic questions across J1A, J1B-Std, J1B-Gen, J2, and J3:
1. Stage-by-stage progression (`Raw Argmax -> SymDec Viterbi -> XDX -> SpellReranker -> MBR`):
   Which stage creates or destroys True Positives vs False Positives at each IoU threshold?
   Specifically: how many TRUE short turnovers were predicted by Viterbi (`SymDec`) and then
   MISTAKENLY ABSORBED by `SpellReranker` vs how many were never emitted by `SymDec`?
2. Anatomy of Missed Short/Medium Turnovers (`FN1a <25f`, `FN1b 25-50f`, `FN1c 51-125f`) and `FN4` merged spells:
   Do the frame probabilities (`max_p_X`, `mean_p_X`, `b4d_pair`, `min_d1_X`, `impulse`) actually detect them?
   Can a targeted `Y -> X -> Y` Surgical Turnover Injector (the exact inverse of `SpellReranker`!) recover
   high-confidence short interceptions (`15..65f`) that Viterbi smoothed over due to `dmin_team=34 / lam_ha=10.5`?
3. Anatomy of Surviving False Positives (`FP1` & `FP2b` in J1A and especially J1B-Gen):
   Why did `82` (`J1A`) and `211` (`J1B-Gen`) false `Y -> X -> Y` opponent blips survive `SpellReranker`?
   Which hard guard (`hard_keep_p_x=0.62`, `hard_keep_b4d=0.28`, `max_len_yxy`, or `tau_absorb`) blocked their absorption?
4. Anatomy of `10-35` Frame `H <-> A` Boundary Offsets on Matched Spells (`IoU 0.3..0.6` failing `0.7`):
   Where is the true physical contact (`min(d1_X)`, `impulse`, `b4d`) relative to `t_pred` and `t_gt`?
5. Exact frame-by-frame physical trace of the errors in `BAR-ATH` (`GT#08..GT#12` and `Pred#11`).
"""
from __future__ import annotations

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
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, HOME
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.spell_reranker import (
    FEATURE_NAMES_YXY,
    SpellRerankerConfig,
    _precompute_match_arrays,
    apply_spell_reranker,
    compute_yxy_feature_vector,
    extract_yxy_candidates,
    rle_runs,
    score_yxy_candidate,
)
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
from src.eval import metrics
from exp_v6_super_champion import compute_physical_gates
from exp_w7a_player_set_net import load_cached_physv2_table
from exp_w7b_super_champion_v7 import (
    _blend_6fam_heads_v7,
    compute_local_newton_gate,
    compute_w_clean_v7,
)


def main():
    t0 = time.time()
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

    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)
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

    w7b_oof = {T: np.load(w7b_dir / f"w7b_{T}.npz") for T in MATCH_IDS}
    w3_oof = {T: np.load(w3_dir / f"b4d_{T}.npz") for T in MATCH_IDS}

    # Load all 9 family probas for Sportec
    oof_fams_v3 = {}
    oof_fams_v6_std = {}
    oof_fams_v6_gen = {}
    for T in MATCH_IDS:
        sym_z = np.load(b_dir / f"sym_{T}.npz")
        tcn_z = np.load(b_dir / f"tcn_sym_{T}.npz")
        pl_z = np.load(plus_dir / f"plus_{T}.npz")
        gz = np.load(g_dir / f"ghost_sym_{T}.npz")
        hz = np.load(h_dir / f"bigru_sym_{T}.npz")
        p2z = np.load(w4_dir / f"physv2_{T}.npz")
        w5z = np.load(w5_dir / f"cb_xgb_{T}.npz")
        t2z = np.load(w5b_dir / f"tcn_physv2_{T}.npz")
        g2z = np.load(w5b_dir / f"bigru_physv2_{T}.npz")
        psz = np.load(w7a_dir / f"pset_{T}.npz")
        v6gz = np.load(w7b_dir / f"v6_genuine_{T}.npz")

        oof_fams_v3[T] = [
            pl_z["symgbm_plus_tta"], p2z["symgbm_physv2_tta"], w5z["symcatboost_physv2_tta"], w5z["symxgb_physv2_tta"],
            tcn_z["symtcn_tta_proba"], hz["symbigru_tta_proba"], t2z["v3_proba"], g2z["v3_proba"],
            psz["v3_proba"],
        ]
        oof_fams_v6_std[T] = [
            v6gz["symgbm_plus_tta_on_v6"], p2z["symgbm_physv2_tta_on_v6"], w5z["symcatboost_physv2_tta_on_v6"], w5z["symxgb_physv2_tta_on_v6"],
            tcn_z["symtcn_tta_proba"], hz["symbigru_tta_proba"], t2z["v6_proba"], g2z["v6_proba"],
            psz["v6_proba"],
        ]
        oof_fams_v6_gen[T] = [
            v6gz["symgbm_plus_tta_on_v6"], p2z["symgbm_physv2_tta_on_v6"], w5z["symcatboost_physv2_tta_on_v6"], w5z["symxgb_physv2_tta_on_v6"],
            t2z["v6_proba"], g2z["v6_proba"], gz["g_v6_on_v6"], psz["v6_proba"],
        ]

    # =========================================================================
    # QUESTION 1: STAGE-BY-STAGE PROGRESSION & SPELL RERANKER AUDIT
    # =========================================================================
    print("=====================================================================================")
    print("Q1: STAGE-BY-STAGE PROGRESSION & SPELL RERANKER AUDIT (J1A v3, J1B-Std, J1B-Gen)")
    print("=====================================================================================")
    for mode_name, p_key, c_key, p4d_key, X_map, fams_map in [
        ("J1A (v3)", "proba_v3", "pchange_v3", "p4d_v3", X_v3, oof_fams_v3),
        ("J1B-Std (v6)", "proba_v6_std", "pchange_v6_std", "p4d_v6", X_v6, oof_fams_v6_std),
        ("J1B-Gen (v6)", "proba_v6_gen", "pchange_v6_gen", "p4d_v6", X_v6, oof_fams_v6_gen),
    ]:
        f1_symdec, f1_xdx, f1_rerank = [], [], []
        tp03_sym, fp03_sym, fn03_sym = 0, 0, 0
        tp03_xdx, fp03_xdx, fn03_xdx = 0, 0, 0
        tp03_rr, fp03_rr, fn03_rr = 0, 0, 0

        # Track what SpellReranker absorbs: how many absorbed candidates helped vs hurt spell_f1_mean!
        n_cand_tot, n_abs_tot, n_abs_helped, n_abs_hurt, n_abs_neutral = 0, 0, 0, 0, 0
        n_kept_tot, n_kept_should_abs, n_kept_blocked_by_p, n_kept_blocked_by_b4d, n_kept_blocked_by_vis, n_kept_blocked_by_tau = 0, 0, 0, 0, 0, 0
        sum_delta_abs_helped, sum_delta_abs_hurt, sum_delta_missed_abs = 0.0, 0.0, 0.0

        for T in MATCH_IDS:
            h = sportec[T].half
            y = sportec[T].labels
            p = w7b_oof[T][p_key]
            c = w7b_oof[T][c_key]
            p4d = w3_oof[T][p4d_key]
            X_df = X_map[T]
            fams = fams_map[T]
            rr_cfg = lomo_rerank_cfgs[T]

            pr_sym = decode_v4(p, h, loo_dec_hyb[T], p_change=c)
            pr_xdx = apply_xdx_arbitrator(pr_sym, p, h, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1)
            pr_rr = apply_spell_reranker(pr_xdx, p, h, X_df, c, p4d, fams, cfg=rr_cfg)

            sc_sym = metrics.score(y, pr_sym, h)
            sc_xdx = metrics.score(y, pr_xdx, h)
            sc_rr = metrics.score(y, pr_rr, h)

            f1_symdec.append(sc_sym["spell_f1_mean"])
            f1_xdx.append(sc_xdx["spell_f1_mean"])
            f1_rerank.append(sc_rr["spell_f1_mean"])

            tp03_sym += sc_sym["spell_f1"][0.3]["n_matched"]
            fp03_sym += sc_sym["spell_f1"][0.3]["n_spurious"]
            fn03_sym += sc_sym["spell_f1"][0.3]["n_missed"]

            tp03_xdx += sc_xdx["spell_f1"][0.3]["n_matched"]
            fp03_xdx += sc_xdx["spell_f1"][0.3]["n_spurious"]
            fn03_xdx += sc_xdx["spell_f1"][0.3]["n_missed"]

            tp03_rr += sc_rr["spell_f1"][0.3]["n_matched"]
            fp03_rr += sc_rr["spell_f1"][0.3]["n_spurious"]
            fn03_rr += sc_rr["spell_f1"][0.3]["n_missed"]

            # Inspect every Y -> X -> Y candidate up to length 150 on pr_xdx
            arr = _precompute_match_arrays(p, X_df, c, p4d, fams)
            base_f1 = float(sc_xdx["spell_f1_mean"])
            cands = extract_yxy_candidates(pr_xdx, h, max_len_yxy=150, min_flank_len=rr_cfg.min_flank_len)
            for _k, a, b, X_team, a_prev, b_next in cands:
                n_cand_tot += 1
                L = b - a
                feat = compute_yxy_feature_vector(a, b, X_team, a_prev, b_next, arr)
                mean_p_X = feat[2]
                b4d_min = feat[12]
                frac_vis = feat[16]
                val_score = score_yxy_candidate(feat, rr_cfg)

                # Test oracle effect of absorbing [a, b) into Y = 1 - X_team
                pr_test = pr_xdx.copy()
                pr_test[a:b] = 1 - X_team
                mod_f1 = float(metrics.score(y, pr_test, h)["spell_f1_mean"])
                gain_if_abs = mod_f1 - base_f1  # >0 means absorbing HELPS; <0 means absorbing HURTS!

                # Did rr_cfg absorb it?
                blocked_len = L > rr_cfg.max_len_yxy
                blocked_p = mean_p_X >= rr_cfg.hard_keep_p_x
                blocked_b4d = (b4d_min >= rr_cfg.hard_keep_b4d and mean_p_X >= 0.52)
                blocked_vis = (frac_vis < 0.50 and L > 55 and mean_p_X >= 0.48)
                blocked_tau = val_score >= rr_cfg.tau_absorb

                was_absorbed = not (blocked_len or blocked_p or blocked_b4d or blocked_vis or blocked_tau)
                if was_absorbed:
                    n_abs_tot += 1
                    if gain_if_abs > 1e-5:
                        n_abs_helped += 1
                        sum_delta_abs_helped += gain_if_abs / 7.0
                    elif gain_if_abs < -1e-5:
                        n_abs_hurt += 1
                        sum_delta_abs_hurt += gain_if_abs / 7.0
                    else:
                        n_abs_neutral += 1
                else:
                    n_kept_tot += 1
                    if gain_if_abs > 1e-4:
                        n_kept_should_abs += 1
                        sum_delta_missed_abs += gain_if_abs / 7.0
                        if blocked_p:
                            n_kept_blocked_by_p += 1
                        elif blocked_b4d:
                            n_kept_blocked_by_b4d += 1
                        elif blocked_vis:
                            n_kept_blocked_by_vis += 1
                        elif blocked_len:
                            pass
                        elif blocked_tau:
                            n_kept_blocked_by_tau += 1

        print(f"\n  --- {mode_name} ---")
        print(f"    SymDec Viterbi : F1={np.mean(f1_symdec):.4f} | TP@0.3={tp03_sym:4d}, FP@0.3={fp03_sym:4d}, FN@0.3={fn03_sym:4d}")
        print(f"    + XDX(FixA)    : F1={np.mean(f1_xdx):.4f} | TP@0.3={tp03_xdx:4d}, FP@0.3={fp03_xdx:4d}, FN@0.3={fn03_xdx:4d}")
        print(f"    + SpellRerank  : F1={np.mean(f1_rerank):.4f} | TP@0.3={tp03_rr:4d}, FP@0.3={fp03_rr:4d}, FN@0.3={fn03_rr:4d}")
        print(f"    Y->X->Y Candidates (L<=150): {n_cand_tot}")
        print(f"      Absorbed by Reranker: {n_abs_tot} -> Helped={n_abs_helped} ({sum_delta_abs_helped:+.4f}), Hurt(True Turnovers Killed!)={n_abs_hurt} ({sum_delta_abs_hurt:+.4f}), Neutral={n_abs_neutral}")
        print(f"      Kept by Reranker    : {n_kept_tot} -> Should have been absorbed={n_kept_should_abs} (Oracle gain left on table={sum_delta_missed_abs:+.4f})")
        print(f"        Blocked by: hard_keep_p_x={n_kept_blocked_by_p}, hard_keep_b4d={n_kept_blocked_by_b4d}, vis_guard={n_kept_blocked_by_vis}, tau_absorb={n_kept_blocked_by_tau}")

    # =========================================================================
    # QUESTION 2: CAN WE RECOVER MISSED SHORT TURNOVERS (`FN1a + FN1b + FN1c + FN4`)?
    # =========================================================================
    print("\n=====================================================================================")
    print("Q2: ANATOMY OF MISSED SHORT TURNOVERS INSIDE PREDICTED SPELLS (J1A Sportec v3)")
    print("=====================================================================================")
    # For every GT spell in J1A that is swallowed by opponent Y (FN1a/b/c):
    # What do `proba[:, X]`, `p_change`, `p4d`, `dist_ball_X`, `impulse` look like inside `[ga, gb)`?
    fn_short_stats = []
    for T in MATCH_IDS:
        h = sportec[T].half
        y = sportec[T].labels
        p = w7b_oof[T]["proba_v3"]
        c = w7b_oof[T]["pchange_v3"]
        p4d = w3_oof[T]["p4d_v3"]
        pr = w7b_oof[T]["pred_v3_rerank"]
        X_df = X_v3[T]

        d1_h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=99.0)
        d1_a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=99.0)
        imp = np.maximum(
            np.nan_to_num(X_df["impulse_mag_d2"].to_numpy(np.float64), nan=0.0),
            np.nan_to_num(X_df["impulse_mag_d4"].to_numpy(np.float64), nan=0.0),
        )
        vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=0.0)

        g_runs = rle_runs(y)
        for k in range(1, len(g_runs) - 1):
            ga, gb, g_team = g_runs[k]
            if g_team not in (HOME, AWAY):
                continue
            L = gb - ga
            if L > 100 or h[ga] <= 0:
                continue
            opp = 1 - g_team
            # Check if completely swallowed by opp in pr
            if np.mean(pr[ga:gb] == opp) < 0.80:
                continue
            pX = p[ga:gb, g_team]
            pY = p[ga:gb, opp]
            d1_X = d1_h[ga:gb] if g_team == HOME else d1_a[ga:gb]
            d1_Y = d1_a[ga:gb] if g_team == HOME else d1_h[ga:gb]
            col_entry = 1 if g_team == HOME else 0
            col_exit = 0 if g_team == HOME else 1
            b4d_in = float(np.max(p4d[max(0, ga - 5):min(len(p), ga + 6), col_entry]))
            b4d_out = float(np.max(p4d[max(0, gb - 5):min(len(p), gb + 6), col_exit]))
            fn_short_stats.append({
                "len": L,
                "max_pX": float(np.max(pX)),
                "mean_pX": float(np.mean(pX)),
                "max_margin_XY": float(np.max(pX - pY)),
                "frames_pX_gt_pY": int(np.sum(pX > pY)),
                "frames_pX_gt_040": int(np.sum(pX > 0.40)),
                "min_d1_X": float(np.min(d1_X)),
                "min_d1_Y_at_minX": float(d1_Y[np.argmin(d1_X)]),
                "max_imp": float(np.max(imp[ga:gb])),
                "b4d_pair": 0.5 * (b4d_in + b4d_out),
                "mean_vis": float(np.mean(vis[ga:gb])),
            })

    df_fn = pd.DataFrame(fn_short_stats)
    print(f"  Total Swallowed Short/Med GT Spells (L <= 100f) in J1A: {len(df_fn)}")
    print(f"    With max_pX > pY (frames_pX_gt_pY >= 1) : {int((df_fn['frames_pX_gt_pY'] >= 1).sum())} ({100*(df_fn['frames_pX_gt_pY'] >= 1).mean():.1f}%)")
    print(f"    With frames_pX_gt_pY >= 8               : {int((df_fn['frames_pX_gt_pY'] >= 8).sum())} ({100*(df_fn['frames_pX_gt_pY'] >= 8).mean():.1f}%)")
    print(f"    With frames_pX_gt_040 >= 8              : {int((df_fn['frames_pX_gt_040'] >= 8).sum())} ({100*(df_fn['frames_pX_gt_040'] >= 8).mean():.1f}%)")
    print(f"    With max_pX >= 0.55 & min_d1_X < 2.0m   : {int(((df_fn['max_pX'] >= 0.55) & (df_fn['min_d1_X'] < 2.0)).sum())}")
    print(f"    With max_pX >= 0.65 & min_d1_X < 1.8m   : {int(((df_fn['max_pX'] >= 0.65) & (df_fn['min_d1_X'] < 1.8)).sum())}")
    print("    Quantiles of Swallowed Short GT Spells:")
    print(df_fn[["len", "max_pX", "mean_pX", "frames_pX_gt_pY", "min_d1_X", "max_imp", "b4d_pair"]].describe().to_string())

    # =========================================================================
    # QUESTION 3: WHY DO MATCHED H<->A BOUNDARIES HAVE 10-35 FRAME OFFSETS?
    # =========================================================================
    print("\n=====================================================================================")
    print("Q3: PHYSICAL ANATOMY OF H <-> A TURNOVER BOUNDARY OFFSETS (J1A & J2)")
    print("=====================================================================================")
    # At every matched H <-> A transition where 5 <= |t_pred - t_gt| <= 40:
    # Let's compare `t_gt` vs `t_pred`: which one is closer to:
    # (a) Probability crossover `p_X == p_Y`
    # (b) Peak `p_change` / `p4d`
    # (c) Receiver physical ball contact `d1_new < 1.8m` + peak `impulse_mag_d2`
    for ds_name, matches_dict, X_dict, is_sportec in [
        ("J1A (Sportec v3)", sportec, X_v3, True),
    ]:
        errs_raw = []
        errs_prob_cross = []
        errs_p4d_peak = []
        errs_phys_touch = []

        for T in MATCH_IDS:
            h = sportec[T].half
            y = sportec[T].labels
            p = w7b_oof[T]["proba_v3"]
            c = w7b_oof[T]["pchange_v3"]
            p4d = w3_oof[T]["p4d_v3"]
            pr = w7b_oof[T]["pred_v3_rerank"]
            X_df = X_v3[T]

            d1_h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=99.0)
            d1_a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=99.0)
            imp2 = np.nan_to_num(X_df["impulse_mag_d2"].to_numpy(np.float64), nan=0.0)
            vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=0.0)

            # Find all direct H <-> A transitions in GT and Pred within +/- 40 frames
            g_runs = rle_runs(y)
            p_runs = rle_runs(pr)
            p_ha_trans = []
            for k in range(len(p_runs) - 1):
                if p_runs[k][2] in (HOME, AWAY) and p_runs[k + 1][2] in (HOME, AWAY) and p_runs[k][2] != p_runs[k + 1][2]:
                    t_p = p_runs[k][1]
                    if h[t_p - 1] > 0 and h[t_p] == h[t_p - 1]:
                        p_ha_trans.append((t_p, p_runs[k][2], p_runs[k + 1][2], p_runs[k][0], p_runs[k + 1][1]))

            for k in range(len(g_runs) - 1):
                if g_runs[k][2] in (HOME, AWAY) and g_runs[k + 1][2] in (HOME, AWAY) and g_runs[k][2] != g_runs[k + 1][2]:
                    t_g = g_runs[k][1]
                    old_t, new_t = g_runs[k][2], g_runs[k + 1][2]
                    if h[t_g - 1] <= 0 or h[t_g] != h[t_g - 1]:
                        continue
                    # Find closest matching pred transition with same (old_t, new_t) within +/- 45 frames
                    cands = [pt for pt in p_ha_trans if pt[1] == old_t and pt[2] == new_t and abs(pt[0] - t_g) <= 45]
                    if not cands:
                        continue
                    t_p, _, _, pa_left, pb_right = min(cands, key=lambda pt: abs(pt[0] - t_g))
                    errs_raw.append(t_p - t_g)

                    # Search window [t_p - 30, t_p + 30] clipped to [pa_left + 15, pb_right - 15]
                    lo = max(pa_left + 15, t_p - 30, 1)
                    hi = min(pb_right - 15, t_p + 31, len(p) - 1)
                    if hi <= lo + 2:
                        continue

                    col_dir = 0 if (old_t == HOME and new_t == AWAY) else 1
                    # 1. Peak of p4d[:, col_dir] + 0.5 * c in [lo, hi]
                    b_sig = p4d[lo:hi, col_dir] + 0.5 * c[lo:hi]
                    t_b4d = lo + int(np.argmax(b_sig))
                    errs_p4d_peak.append(t_b4d - t_g)

                    # 2. First frame where new_t has physical contact (d1_new < 2.2m & d1_new < d1_old) AND p_new > 0.35
                    d1_new = d1_h[lo:hi] if new_t == HOME else d1_a[lo:hi]
                    d1_old = d1_a[lo:hi] if new_t == HOME else d1_h[lo:hi]
                    p_new = p[lo:hi, new_t]
                    p_old = p[lo:hi, old_t]
                    # Score for exact turnover frame t: before t old_t dominates, from t onward new_t dominates + contact
                    csum_diff = np.cumsum(p_old - p_new)
                    t_prob = lo + int(np.argmax(csum_diff))
                    errs_prob_cross.append(t_prob - t_g)

                    contact_score = csum_diff + 1.5 * b_sig + 0.15 * np.clip(imp2[lo:hi], 0, 15) * (d1_new < 2.2)
                    t_phys = lo + int(np.argmax(contact_score))
                    errs_phys_touch.append(t_phys - t_g)

        print(f"  {ds_name} Matched H<->A Transitions (n={len(errs_raw)}):")
        for lbl, arr in [
            ("Current Viterbi t_pred", errs_raw),
            ("Log-Odds Crossover t_prob", errs_prob_cross),
            ("Peak 4D-Boundary t_b4d", errs_p4d_peak),
            ("Hybrid Contact+Prob t_phys", errs_phys_touch),
        ]:
            a = np.asarray(arr)
            print(f"    {lbl:28s}: mean|err|={np.mean(np.abs(a)):5.2f}f | median|err|={np.median(np.abs(a)):4.1f}f | signed={np.mean(a):+5.2f}f | |err|<=5f: {100*np.mean(np.abs(a)<=5):4.1f}% | |err|<=15f: {100*np.mean(np.abs(a)<=15):4.1f}%")

    # =========================================================================
    # QUESTION 4: BAR-ATH PUBLIC WINDOW FRAME-BY-FRAME TRACE OF THE 2 ERROR ZONES
    # =========================================================================
    print("\n=====================================================================================")
    print("Q4: BAR-ATH PUBLIC WINDOW EXACT TRACE OF THE 2 ERROR ZONES")
    print("=====================================================================================")
    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        pe_plus = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f:
        _, atd_rep_v6 = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        _, atd_rep_h = pickle.load(f)
    with open(w3_dir / "b4d_j23_preds.pkl", "rb") as f:
        b4d_j23 = pickle.load(f)
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f:
        pe_p2 = pickle.load(f)
    with open(w5_dir / "eval_preds_cb_xgb.pkl", "rb") as f:
        pe_w5 = pickle.load(f)
    with open(w5b_dir / "seq_physv2_j23_preds.pkl", "rb") as f:
        pe_w5b = pickle.load(f)
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f:
        pset_j23 = pickle.load(f)

    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )
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
    p4d_ap = b4d_j23["atd_rep"]

    for z_name, a_z, b_z in [
        ("Zone 1: GT#09 [H] [15346..15365] (19f)", 15335, 15380),
        ("Zone 2: GT#11 [H] [15766..15799] (33f)", 15755, 15815),
        ("Zone 3: Pred#11 [A] [17385..17452] (67f in GT D)", 17370, 17460),
    ]:
        print(f"\n  --- {z_name} ---")
        sl = slice(a_z, b_z)
        print(f"    mean(p_H, p_A, p_D) = ({np.mean(pap[sl,0]):.3f}, {np.mean(pap[sl,1]):.3f}, {np.mean(pap[sl,2]):.3f}) | max(p_H)={np.max(pap[sl,0]):.3f}, max(p_A)={np.max(pap[sl,1]):.3f}, max(p_D)={np.max(pap[sl,2]):.3f}")
        print(f"    max(p_change)={np.max(cap[sl]):.3f} | max(p4d H->A, A->H, L->D, D->L)=({np.max(p4d_ap[sl,0]):.3f}, {np.max(p4d_ap[sl,1]):.3f}, {np.max(p4d_ap[sl,2]):.3f}, {np.max(p4d_ap[sl,3]):.3f})")
        dH = X_atd_rep["dist_ball_H"].to_numpy()[sl]
        dA = X_atd_rep["dist_ball_A"].to_numpy()[sl]
        vis = X_atd_rep["ball_visible"].to_numpy()[sl]
        imp = X_atd_rep["impulse_mag_d4"].to_numpy()[sl]
        print(f"    min(d1_H)={np.nanmin(dH):.2f}m, min(d1_A)={np.nanmin(dA):.2f}m | mean(vis)={np.nanmean(vis):.2f} | max(imp4)={np.nanmax(imp):.2f}")
        # Print 5-frame buckets
        for f0 in range(a_z, b_z, 5):
            f1 = min(f0 + 5, b_z)
            s5 = slice(f0, f1)
            print(f"      [{f0:5d}..{f1:5d}] p=({np.mean(pap[s5,0]):.2f},{np.mean(pap[s5,1]):.2f},{np.mean(pap[s5,2]):.2f}) c={np.max(cap[s5]):.2f} d1=({np.nanmin(dH[f0-a_z:f1-a_z]):4.1f}m,{np.nanmin(dA[f0-a_z:f1-a_z]):4.1f}m) vis={np.mean(vis[f0-a_z:f1-a_z]):.1f} imp={np.nanmax(imp[f0-a_z:f1-a_z]):4.1f}")

    print(f"\nCompleted deep physics autopsy in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
