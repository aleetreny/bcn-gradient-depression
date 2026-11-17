"""Test zero-leakage surgical post-decoding & native-v6 fixes derived from the v7 Forensic Autopsy:
1. Fix 1 (J1B-Gen Native-v6 Local-Newton Gate): Gate 481-col sequence models by `a_v6` on `v6_gen` instead of `np.ones`, falling back to ghost-robust v6 models during ghost-ball gaps.
2. Fix 2 (Asymmetric Dead-Ball Wedge Arbitrator `Y->X->D`, `D->X->Y`, `D->X->D`): Absorb/trim false live wedges adjacent to DEAD where `p_X` is weak and `p_D + p_Y` dominates.
3. Fix 3 (Recalibrated 9-Family SpellReranker): Relax the overly aggressive `hard_keep_p_x` guard so genuine false `Y->X->Y` blips can be evaluated and absorbed in 7-fold LOMO.
4. Fix 4 (Double-Domino Short-Turnover Recovery): Surgically split merged opponent spells where a clear short turnover (`18..48f`, high `p_X` or `p4d[Y->X]` + physical ball contact `d1_X` + impulse) was swallowed by `dmin_team=34`.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore")

EXP_LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_LAB / "py_pkgs"))
sys.path.insert(0, str(EXP_LAB))
sys.path.insert(0, str(EXP_LAB / "scripts"))
sys.path.insert(0, str(EXP_LAB / "evaluation"))

import possession_metrics as pm
from games import GAMES as GAME_INFO, anchor_video_frame
from s2_3_frame_models import plausibility
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, HOME
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.spell_reranker import (
    SpellRerankerConfig,
    apply_spell_reranker,
    rle_runs,
)
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
from src.eval import metrics
from exp_v6_super_champion import compute_physical_gates
from exp_w7a_player_set_net import load_cached_physv2_table
from exp_w7b_super_champion_v7 import (
    _blend_6fam_heads_v7,
    compute_local_newton_gate,
    compute_w_clean_v7,
)


def inspect_yxd_dxy_candidates(
    pred: np.ndarray,
    gt: np.ndarray,
    half: np.ndarray,
    proba: np.ndarray,
    p_change: np.ndarray,
    p4d: np.ndarray,
    X_df: pd.DataFrame,
):
    """Collect feature statistics of all Y->X->D, D->X->Y, and D->X->D spells (True vs False)."""
    valid = (half == 1) | (half == 2)
    arr = np.where(valid, pred, -1)
    runs = rle_runs(arr)
    rows = []

    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=50.0)
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=1.0) if "ball_visible" in X_df else np.ones(len(pred))
    imp = np.nan_to_num(X_df["impulse_mag_d4"].to_numpy(np.float64), nan=0.0) if "impulse_mag_d4" in X_df else np.zeros(len(pred))
    oop = np.nan_to_num(X_df["out_of_pitch"].to_numpy(np.float64), nan=0.0) if "out_of_pitch" in X_df else np.zeros(len(pred))

    for idx in range(1, len(runs) - 1):
        s, e, x = runs[idx]
        if x not in (HOME, AWAY):
            continue
        ps, pe, prev_y = runs[idx - 1]
        ns, ne, next_y = runs[idx + 1]
        if half[s] != half[ps] or half[s] != half[ns]:
            continue
        L = e - s
        if L > 120:
            continue
        opp = AWAY if x == HOME else HOME
        topo = None
        if prev_y == opp and next_y == DEAD:
            topo = "Y_X_D"
        elif prev_y == DEAD and next_y == opp:
            topo = "D_X_Y"
        elif prev_y == DEAD and next_y == DEAD:
            topo = "D_X_D"
        else:
            continue

        # Check if absorbing X improves official spell_f1_mean on this match
        p_test = pred.copy()
        if topo == "Y_X_D":
            # Split between Y and D at crossover or absorb into D/Y
            # Find frame k in [s, e] maximizing sum_{t<k} p_Y(t) + sum_{t>=k} p_D(t)
            p_y_seg = proba[s:e, opp]
            p_d_seg = proba[s:e, DEAD]
            diff = p_y_seg - p_d_seg
            c_sum = np.concatenate([[0.0], np.cumsum(diff)])
            k_best = int(np.argmax(c_sum))
            p_test[s : s + k_best] = opp
            p_test[s + k_best : e] = DEAD
        elif topo == "D_X_Y":
            p_d_seg = proba[s:e, DEAD]
            p_y_seg = proba[s:e, opp]
            diff = p_d_seg - p_y_seg
            c_sum = np.concatenate([[0.0], np.cumsum(diff)])
            k_best = int(np.argmax(c_sum))
            p_test[s : s + k_best] = DEAD
            p_test[s + k_best : e] = opp
        else:
            p_test[s:e] = DEAD

        f1_base = metrics.score(gt, pred, half)["spell_f1_mean"]
        f1_new = metrics.score(gt, p_test, half)["spell_f1_mean"]
        delta_f1 = f1_new - f1_base

        d1_x = d1h[s:e] if x == HOME else d1a[s:e]
        d1_y = d1a[s:e] if x == HOME else d1h[s:e]
        dir_yx = 0 if (opp == HOME and x == AWAY) else 1
        sw = max(0, s - 6)
        ew = min(len(pred), s + 7)
        b4d_entry = float(np.max(p4d[sw:ew, dir_yx])) if ew > sw else 0.0

        rows.append({
            "topo": topo,
            "L": L,
            "delta_f1": delta_f1,
            "should_absorb": delta_f1 > 1e-6,
            "mean_pX": float(np.mean(proba[s:e, x])),
            "max_pX": float(np.max(proba[s:e, x])),
            "mean_pD": float(np.mean(proba[s:e, DEAD])),
            "max_pD": float(np.max(proba[s:e, DEAD])),
            "mean_pY": float(np.mean(proba[s:e, opp])),
            "frac_pX_dom": float(np.mean(proba[s:e, x] > np.maximum(proba[s:e, opp], proba[s:e, DEAD]))),
            "b4d_entry": b4d_entry,
            "min_d1_x": float(np.min(d1_x)),
            "mean_d1_x": float(np.mean(d1_x)),
            "mean_vis": float(np.mean(vis[s:e])),
            "max_oop": float(np.max(oop[s:e])),
        })
    return rows


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
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)

    w7b_oof = {T: np.load(w7b_dir / f"w7b_{T}.npz") for T in MATCH_IDS}
    w3_oof = {T: np.load(w3_dir / f"b4d_{T}.npz") for T in MATCH_IDS}

    # =========================================================================
    # TEST 1: FIXING J1B-GEN (`100% Native-v6`) BY GATING 481-SEQ WITH `a_v6` & GHOST-TREE FALLBACK
    # =========================================================================
    print("=====================================================================================")
    print("TEST 1: J1B-GEN (100% NATIVE-V6) LOCAL-NEWTON GATING & GHOST-ROBUST FALLBACK")
    print("=====================================================================================")
    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )

    oof_raw = {}
    for T in MATCH_IDS:
        zp = np.load(plus_dir / f"plus_{T}.npz")
        zg = np.load(g_dir / f"ghost_sym_{T}.npz")
        zw3 = np.load(w3_dir / f"b4d_{T}.npz")
        zw4 = np.load(w4_dir / f"physv2_{T}.npz")
        zw5 = np.load(w5_dir / f"cb_xgb_{T}.npz")
        zw5b_tcn = np.load(w5b_dir / f"tcn_physv2_{T}.npz")
        zw5b_gru = np.load(w5b_dir / f"bigru_physv2_{T}.npz")
        zw7a = np.load(w7a_dir / f"pset_{T}.npz")
        zw7b_v6 = np.load(w7b_dir / f"v6_genuine_{T}.npz")
        oof_raw[T] = {
            "g_pl_on_v6": zw7b_v6["symgbm_plus_tta_on_v6"],
            "tr_pl_on_v6": zw7b_v6["symtrans_plus_tta_on_v6"],
            "g_v6_on_v6": zg["g_v6_on_v6"],
            "tr_v6_on_v6": zg["tr_v6_on_v6"],
            "p4d_v6": zw3["p4d_v6"],
            "g_p2_v6": zw4["symgbm_physv2_tta_on_v6"],
            "tr_p2_v6": zw4["symtrans_physv2_tta_on_v6"],
            "cb_p2_v6": zw5["symcatboost_physv2_tta_on_v6"],
            "xgb_p2_v6": zw5["symxgb_physv2_tta_on_v6"],
            "tcn_p2_v6": zw5b_tcn["v6_proba"],
            "tcnc_p2_v6": zw5b_tcn["v6_change"],
            "gru_p2_v6": zw5b_gru["v6_proba"],
            "gruc_p2_v6": zw5b_gru["v6_change"],
            "pset_v6": zw7a["v6_proba"],
            "psetc_v6": zw7a["v6_change"],
        }

    print("  [Standalone Native-v6 Model Audit on Sportec v6 (7-Fold LOMO)]:")
    for m_name, p_k, c_k in [
        ("g_pl_on_v6   (SymGBM_Plus_352)", "g_pl_on_v6", "tr_pl_on_v6"),
        ("g_p2_v6      (SymGBM_PhysV2_481)", "g_p2_v6", "tr_p2_v6"),
        ("cb_p2_v6     (SymCatBoost_481)", "cb_p2_v6", "tr_p2_v6"),
        ("xgb_p2_v6    (SymXGB_481)", "xgb_p2_v6", "tr_p2_v6"),
        ("g_v6_on_v6   (GhostGBM_244)", "g_v6_on_v6", "tr_v6_on_v6"),
        ("tcn_p2_v6    (SymTCN_481)", "tcn_p2_v6", "tcnc_p2_v6"),
        ("gru_p2_v6    (SymBiGRU_481)", "gru_p2_v6", "gruc_p2_v6"),
        ("pset_v6      (SymPlayerSetNet)", "pset_v6", "psetc_v6"),
    ]:
        sc_list = []
        fp_list = []
        for T in MATCH_IDS:
            d = oof_raw[T]
            h = sportec[T].half
            y = sportec[T].labels
            pr = decode_v4(d[p_k], h, loo_dec_hyb[T], p_change=d[c_k])
            sc = metrics.score(y, pr, h)
            sc_list.append(sc["spell_f1_mean"])
        print(f"    {m_name:36s}: spell_f1_mean = {np.mean(sc_list):.4f}")

    for mode_label, use_newton_fallback in [
        ("v7 Baseline J1B-Gen (alpha_p2=1.0 hardcoded)", False),
        ("v7 Fixed J1B-Gen (Local-Newton gated -> pset_v6 + g_v6_on_v6 fallback)", True),
    ]:
        f1s = []
        fps = []
        tps = []
        for T in MATCH_IDS:
            d = oof_raw[T]
            h = sportec[T].half
            y = sportec[T].labels
            a_v6 = compute_local_newton_gate(X_v6[T], h, gates_v6[T][0])
            wc_v6 = compute_w_clean_v7(X_v6[T], h, gates_v6[T][0])

            if not use_newton_fallback:
                p6g = w7b_oof[T]["proba_v6_gen"]
                c6g = w7b_oof[T]["pchange_v6_gen"]
            else:
                # When 244-col sequence models are absent, use ghost-robust 100%-native-v6 fallback
                # (50% SymPlayerSetNet_v6 + 50% Ghost-SymGBM_v6) in the 244-col fallback slots!
                fb_p = 0.50 * d["pset_v6"] + 0.50 * d["g_v6_on_v6"]
                fb_c = 0.50 * d["psetc_v6"] + 0.50 * d["tr_v6_on_v6"]
                p6g, c6g = synthesize_head_decoupled_proba(
                    d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                    fb_p, fb_p, d["tr_pl_on_v6"], d["tr_p2_v6"], fb_c, fb_c,
                    h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                    w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                    tcn_physv2_proba=d["tcn_p2_v6"],
                    bigru_physv2_proba=d["gru_p2_v6"],
                    tcn_physv2_change=d["tcnc_p2_v6"],
                    bigru_physv2_change=d["gruc_p2_v6"],
                    alpha_p2=a_v6,
                )
                p6g, c6g = _blend_6fam_heads_v7(
                    p6g, c6g, fb_p, fb_p, d["gru_p2_v6"], d["pset_v6"],
                    fb_c, fb_c, d["gruc_p2_v6"], d["psetc_v6"],
                    a_v6, wc_v6, 0.05, 0.30, 0.18, 0.08,
                )
                p6g = calibrate_match_transductive(p6g, h, X_v6[T], best_calib_cfg)

            pr = decode_v4(p6g, h, loo_dec_hyb[T], p_change=c6g)
            pr = apply_xdx_arbitrator(pr, p6g, h, X_v6[T], p_change=c6g, p_trans_4d=d["p4d_v6"], cfg=xdx_cfg_st1)
            sc = metrics.score(y, pr, h)
            f1s.append(sc["spell_f1_mean"])
        print(f"  {mode_label:72s}: spell_f1_mean = {np.mean(f1s):.4f}")

    # =========================================================================
    # TEST 2: INSPECT ASYMMETRIC DEAD-BALL WEDGES (Y->X->D, D->X->Y, D->X->D) ON J1A
    # =========================================================================
    print("\n=====================================================================================")
    print("TEST 2: ANATOMY OF Y->X->D, D->X->Y, D->X->D CANDIDATES IN J1A (Sportec v3)")
    print("=====================================================================================")
    all_wedges = []
    for T in MATCH_IDS:
        z = w7b_oof[T]
        rows = inspect_yxd_dxy_candidates(
            z["pred_v3_rerank"], sportec[T].labels, sportec[T].half,
            z["proba_v3"], z["pchange_v3"], w3_oof[T]["p4d_v3"], X_v3[T],
        )
        all_wedges.extend(rows)

    df_w = pd.DataFrame(all_wedges)
    for topo in ["Y_X_D", "D_X_Y", "D_X_D"]:
        sub = df_w[df_w["topo"] == topo]
        pos = sub[sub["should_absorb"]]
        neg = sub[~sub["should_absorb"]]
        print(f"\n  Topology {topo}: Total Candidates (L<=120) = {len(sub)} | Should Absorb = {len(pos)} (Sum +Delta F1/7 = {pos['delta_f1'].sum()/7:+.4f}) | Should Keep = {len(neg)}")
        if len(pos) > 0:
            print(f"    Should Absorb (False X): mean_L={pos['L'].mean():.1f}, mean_pX={pos['mean_pX'].mean():.3f}, max_pX={pos['max_pX'].mean():.3f}, mean_pD={pos['mean_pD'].mean():.3f}, b4d_in={pos['b4d_entry'].mean():.3f}, min_d1_x={pos['min_d1_x'].mean():.2f}")
        if len(neg) > 0:
            print(f"    Should Keep   (True  X): mean_L={neg['L'].mean():.1f}, mean_pX={neg['mean_pX'].mean():.3f}, max_pX={neg['max_pX'].mean():.3f}, mean_pD={neg['mean_pD'].mean():.3f}, b4d_in={neg['b4d_entry'].mean():.3f}, min_d1_x={neg['min_d1_x'].mean():.2f}")


if __name__ == "__main__":
    main()
