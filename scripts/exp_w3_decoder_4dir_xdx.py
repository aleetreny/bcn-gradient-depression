"""Wave 3 Comprehensive Evaluation: Semi-Markov Decoder v2 (EXP-01),
Symmetric 4-Directional Boundary Detector (EXP-02), and Surgical X->D->X Causal Arbitrator (EXP-03).

Evaluates all 3 orthogonal Wave 3 innovations across all 4 official benchmarks:
  - Judge 1A: 7-Fold Nested LOMO on Sportec `degrade_v3`
  - Judge 1B: 7-Fold Nested LOMO on Sportec `degrade_v6` (Ghost-Ball Incoherence)
  - Judge 2:  Metrica Sample (3 games, standard and `without_dead_margin`)
  - Judge 3:  BAR-ATH Broadcast-Repaired (Full-Match Plausibility + Public 5-Min Check)
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_3_frame_models import plausibility
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, HOME
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.decode.viterbi_hsmm_v2 import DecoderParamsV2, decode_v2
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator, rle_segments
from src.eval import metrics
from src.features.v1ctl_plus import mirror_v1ctl_plus_df
from src.models.transition import change_features
from src.models.boundary_4dir import targets_4dir, swap_trans_4d
from exp_b_symmetry_eval import blend_proba
from exp_v6_super_champion import compute_physical_gates, moe_combine, save_and_validate_submission


def build_4dir_targets(labels: np.ndarray, half: np.ndarray) -> np.ndarray:
    d = targets_4dir(labels, half)
    return np.column_stack([d["y_h2a"], d["y_a2h"], d["y_live2d"], d["y_d2live"]])


def load_cached_plus(prefix: str, half: np.ndarray | None = None, with_change: bool = False) -> pd.DataFrame:
    """Fast load of cached (features_v1 + control_v1 + plus_v1) parquet tables."""
    f1 = sorted(glob.glob(str(ROOT / f"local_data/cache/features_v1/{prefix}*.parquet")))[0]
    f2 = sorted(glob.glob(str(ROOT / f"local_data/cache/control_v1/{prefix}*.parquet")))[0]
    f3 = sorted(glob.glob(str(ROOT / f"local_data/cache/plus_v1/{prefix}*.parquet")))[0]
    df1 = pd.read_parquet(f1)
    if with_change and half is not None:
        df1 = change_features(df1, half)
    return pd.concat([df1, pd.read_parquet(f2), pd.read_parquet(f3)], axis=1)


def count_interior_xdx_islands(pred: np.ndarray, half: np.ndarray, X: pd.DataFrame, max_len: int = 240) -> tuple[int, int]:
    """Count total X->D->X segments (len <= max_len) and how many stay inside the pitch interior."""
    s_arr, e_arr, v_arr = rle_segments(pred, half)
    n_tot, n_int = 0, 0
    bx_c = np.abs(X["ball_last_x"].to_numpy(np.float64) - 52.5) if "ball_last_x" in X.columns else np.zeros(len(pred))
    by_c = np.abs(X["ball_last_y"].to_numpy(np.float64) - 34.0) if "ball_last_y" in X.columns else np.zeros(len(pred))
    if np.nanmin(X["ball_last_x"].to_numpy(np.float64)) < -10.0:
        bx_c = np.abs(X["ball_last_x"].to_numpy(np.float64))
        by_c = np.abs(X["ball_last_y"].to_numpy(np.float64))
    for i in range(1, len(s_arr) - 1):
        if v_arr[i] != DEAD:
            continue
        if v_arr[i - 1] in (HOME, AWAY) and v_arr[i - 1] == v_arr[i + 1]:
            a, b = int(s_arr[i]), int(e_arr[i])
            if half[a] <= 0 or half[s_arr[i - 1]] != half[a] or half[e_arr[i + 1] - 1] != half[a]:
                continue
            if (b - a) <= max_len:
                n_tot += 1
                mx = float(np.nanmax(bx_c[a:b])) if np.any(np.isfinite(bx_c[a:b])) else 0.0
                my = float(np.nanmax(by_c[a:b])) if np.any(np.isfinite(by_c[a:b])) else 0.0
                if mx < 50.0 and my < 33.5:
                    n_int += 1
    return n_tot, n_int


def main():
    t0 = time.time()
    print("=== WAVE 3: DECODER V2 (EXP-01) + 4-DIR BOUNDARIES (EXP-02) + X->D->X ARBITRATOR (EXP-03) ===", flush=True)

    sportec = load_all()
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)

    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    plus_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    h_dir = ROOT / "experiments" / "outputs" / "exp_h_oof"
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    w3_dir = ROOT / "experiments" / "outputs" / "exp_w3_oof"
    store_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std = pickle.load(f)[0]
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        loo_dec_lin, loo_dec_hyb, dec_v5_lin, dec_v5_hyb = pickle.load(f)
    dec_v4 = DecoderParams(**{
        k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items()
        if not k.startswith("_")
    })

    # Load cached feature tables for physical gates & X->D->X arbitrator
    X_v3, X_v6, gates_v3, gates_v6 = {}, {}, {}, {}
    for mid, m in sportec.items():
        X_v3[mid] = load_cached_plus(f"sportec_{mid}_degrade_v3_s0")
        X_v6[mid] = load_cached_plus(f"sportec_{mid}_degrade_v6_s0")
        gates_v3[mid] = compute_physical_gates(X_v3[mid], m.half)
        gates_v6[mid] = compute_physical_gates(X_v6[mid], m.half)

    X_sample, gates_sample = {}, {}
    for gid in GAMES:
        ms = sample_matches[gid]
        X_sample[gid] = load_cached_plus(f"sample_{gid}_raw")
        gates_sample[gid] = compute_physical_gates(X_sample[gid], ms.half)

    Xa_rep = load_cached_plus("atd_rep_True_True_None_75a82cda201463f46ad07ae9")
    Xa_raw = load_cached_plus("atd_0f083428befb19d09710022b")
    gate_atd = compute_physical_gates(Xa_rep, atd.half)

    # Load all OOF predictions including EXP-02 4-directional boundary predictions
    oof = {}
    b4d_v3, b4d_v6 = {}, {}
    for T in MATCH_IDS:
        zb = np.load(sym_dir / f"sym_{T}.npz")
        zt = np.load(sym_dir / f"tcn_sym_{T}.npz")
        zp = np.load(plus_dir / f"plus_{T}.npz")
        zh = np.load(h_dir / f"bigru_sym_{T}.npz")
        zg = np.load(g_dir / f"ghost_sym_{T}.npz")
        zw3 = np.load(w3_dir / f"b4d_{T}.npz")
        z_orig_tcn = np.load(store_v3 / T / f"tcn_outer_{T}.npz")
        oof[T] = {
            "g_orig": np.load(store_v3 / T / f"gbm_outer_{T}.npy"),
            "tr_orig": np.load(store_v3 / T / f"trans_outer_{T}.npy"),
            "t_orig": z_orig_tcn["proba"],
            "tc_orig": z_orig_tcn["change"],
            "g_v1": zb["symgbm_tta"],
            "tr_v1": zb["symtrans_tta"],
            "g_pl": zp["symgbm_plus_tta"],
            "tr_pl": zp["symtrans_plus_tta"],
            "g_v6_on_v3": zg["g_v6_on_v3"],
            "tr_v6_on_v3": zg["tr_v6_on_v3"],
            "g_v3_on_v6": zg["g_v3_on_v6"],
            "tr_v3_on_v6": zg["tr_v3_on_v6"],
            "g_v6_on_v6": zg["g_v6_on_v6"],
            "tr_v6_on_v6": zg["tr_v6_on_v6"],
            "tcn": zt["symtcn_tta_proba"],
            "tcnc": zt["symtcn_tta_change"],
            "gru": zh["symbigru_tta_proba"],
            "gruc": zh["symbigru_tta_change"],
        }
        b4d_v3[T] = zw3["p4d_v3"]
        b4d_v6[T] = zw3["p4d_v6"]

    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        pe = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f:
        sample_v6, atd_rep_v6 = pickle.load(f)
    with open(w3_dir / "b4d_j23_preds.pkl", "rb") as f:
        b4d_j23 = pickle.load(f)
    with open(w3_dir / "symb4d_full.pkl", "rb") as f:
        symb4d_full = pickle.load(f)

    # =========================================================================
    # PART 1: EXACT H <-> A SYMMETRY VERIFICATION (0.000e+00)
    # =========================================================================
    print("\n--- PART 1: EXACT H <-> A EQUIVARIANCE VERIFICATION ---", flush=True)
    sub_X_base = X_v3["J03WMX"].iloc[1000:4000].reset_index(drop=True)
    sub_half = sportec["J03WMX"].half[1000:4000]
    f1_wmx = sorted(glob.glob(str(ROOT / "local_data/cache/features_v1/sportec_J03WMX_degrade_v3_s0*.parquet")))[0]
    f2_wmx = sorted(glob.glob(str(ROOT / "local_data/cache/control_v1/sportec_J03WMX_degrade_v3_s0*.parquet")))[0]
    f3_wmx = sorted(glob.glob(str(ROOT / "local_data/cache/plus_v1/sportec_J03WMX_degrade_v3_s0*.parquet")))[0]
    sub_X = pd.concat([
        change_features(pd.read_parquet(f1_wmx).iloc[1000:4000].reset_index(drop=True), sub_half),
        pd.read_parquet(f2_wmx).iloc[1000:4000].reset_index(drop=True),
        pd.read_parquet(f3_wmx).iloc[1000:4000].reset_index(drop=True),
    ], axis=1)
    sub_X_mir = mirror_v1ctl_plus_df(sub_X, sub_half)

    p4d_orig = symb4d_full.predict(sub_X, sub_half)
    p4d_mir = symb4d_full.predict(sub_X_mir, sub_half)
    err_b4d = float(np.max(np.abs(p4d_orig - swap_trans_4d(p4d_mir))))

    p_sub = oof["J03WMX"]["g_pl"][1000:4000]
    p_sub_mir = p_sub[:, [1, 0, 2]]
    c_sub = oof["J03WMX"]["tr_pl"][1000:4000]

    dp2_test = DecoderParamsV2.from_legacy(dec_v4, beta_to_d=1.0, beta_from_d=1.0, short_dur_cap=18, gamma_short=0.5)
    pred_v2_orig = decode_v2(p_sub, sub_half, dp2_test, p_change=c_sub, p_trans_4d=p4d_orig)
    pred_v2_mir = decode_v2(p_sub_mir, sub_half, dp2_test, p_change=c_sub, p_trans_4d=p4d_mir)
    pred_v2_mir_swap = np.where(pred_v2_mir == 0, 1, np.where(pred_v2_mir == 1, 0, 2))
    err_v2_frames = int(np.sum(pred_v2_orig != pred_v2_mir_swap))

    cfg_test = XDXConfig(enable_oob_trim=True)
    pred_xdx_orig = apply_xdx_arbitrator(pred_v2_orig, p_sub, sub_half, sub_X, p_change=c_sub, p_trans_4d=p4d_orig, cfg=cfg_test)
    pred_xdx_mir = apply_xdx_arbitrator(pred_v2_mir, p_sub_mir, sub_half, sub_X_mir, p_change=c_sub, p_trans_4d=p4d_mir, cfg=cfg_test)
    pred_xdx_mir_swap = np.where(pred_xdx_mir == 0, 1, np.where(pred_xdx_mir == 1, 0, 2))
    err_xdx_frames = int(np.sum(pred_xdx_orig != pred_xdx_mir_swap))

    print(f"  [EXP-02] SymBoundary4DirGBM max |p4d(X) - swap(p4d(X_mir))| : {err_b4d:.3e}", flush=True)
    print(f"  [EXP-01] decode_v2 H<->A symmetry mismatch frames            : {err_v2_frames} / {len(sub_X)}", flush=True)
    print(f"  [EXP-03] apply_xdx_arbitrator H<->A symmetry mismatch frames : {err_xdx_frames} / {len(sub_X)}", flush=True)
    assert err_b4d == 0.0 and err_v2_frames == 0 and err_xdx_frames == 0

    # =========================================================================
    # PART 2: EXP-02 4-DIRECTIONAL BOUNDARY DETECTOR STANDALONE BENCHMARK
    # =========================================================================
    print("\n--- PART 2: EXP-02 4-DIRECTIONAL BOUNDARY DETECTOR PR-AUC & ROC-AUC ---", flush=True)
    ch_names = ["H -> A", "A -> H", "Live -> D", "D -> Live"]
    for deg_tag, b4d_dict in [("Sportec 7-Fold LOMO (degrade_v3)", b4d_v3), ("Sportec 7-Fold LOMO (degrade_v6)", b4d_v6)]:
        pr_mat, roc_mat = [], []
        for T in MATCH_IDS:
            m = sportec[T]
            y4 = build_4dir_targets(m.labels, m.half)
            valid = (m.labels >= 0) & (m.half > 0)
            p4 = b4d_dict[T]
            pr_mat.append([average_precision_score(y4[valid, c], p4[valid, c]) for c in range(4)])
            roc_mat.append([roc_auc_score(y4[valid, c], p4[valid, c]) for c in range(4)])
        pr_arr = np.array(pr_mat)
        roc_arr = np.array(roc_mat)
        print(f"  {deg_tag}:", flush=True)
        for c in range(4):
            print(
                f"    Channel {c} ({ch_names[c]:9s}): PR-AUC = {pr_arr[:, c].mean():.4f} (std={pr_arr[:, c].std():.4f}) | "
                f"ROC-AUC = {roc_arr[:, c].mean():.4f}",
                flush=True,
            )

    print("\n  [1D Turnover (H <-> A) Boundary PR-AUC Comparison]", flush=True)
    b_models = [
        ("SymTrans_v1ctl_TTA (282 feat)", lambda T, d: d["tr_v1"], lambda g: sample_h[g]["tr_sym"]),
        ("SymTrans_Plus_TTA (352 feat)", lambda T, d: d["tr_pl"], lambda g: pe["sample"][g]["tr_plus_sym"]),
        ("SymTCN_TTA_change", lambda T, d: d["tcnc"], lambda g: sample_h[g]["tc_sym"]),
        ("SymBoundary4DirGBM (p_H2A + p_A2H, 601 feat)", lambda T, d: np.clip(b4d_v3[T][:, 0] + b4d_v3[T][:, 1], 0, 1),
         lambda g: np.clip(b4d_j23["sample"][g][:, 0] + b4d_j23["sample"][g][:, 1], 0, 1)),
        ("Wave 2 Hybrid (0.50*SymTrans_Plus + 0.50*SymTCN)", lambda T, d: 0.50 * d["tr_pl"] + 0.50 * d["tcnc"],
         lambda g: 0.50 * pe["sample"][g]["tr_plus_sym"] + 0.50 * sample_h[g]["tc_sym"]),
        ("Wave 3 Hybrid (0.25*SymTrans_Plus + 0.50*SymTCN + 0.25*SymB4D)",
         lambda T, d: 0.25 * d["tr_pl"] + 0.50 * d["tcnc"] + 0.25 * np.clip(b4d_v3[T][:, 0] + b4d_v3[T][:, 1], 0, 1),
         lambda g: 0.25 * pe["sample"][g]["tr_plus_sym"] + 0.50 * sample_h[g]["tc_sym"] + 0.25 * np.clip(b4d_j23["sample"][g][:, 0] + b4d_j23["sample"][g][:, 1], 0, 1)),
    ]
    for bname, fn_sp, fn_sm in b_models:
        sp_pr, sp_roc, sm_pr, sm_roc = [], [], [], []
        for T in MATCH_IDS:
            m = sportec[T]
            y4 = build_4dir_targets(m.labels, m.half)
            y_turn = np.maximum(y4[:, 0], y4[:, 1])
            valid = (m.labels >= 0) & (m.half > 0)
            pred_b = fn_sp(T, oof[T])
            sp_pr.append(average_precision_score(y_turn[valid], pred_b[valid]))
            sp_roc.append(roc_auc_score(y_turn[valid], pred_b[valid]))
        for gid in GAMES:
            ms = sample_matches[gid]
            y4s = build_4dir_targets(ms.labels, ms.half)
            y_turns = np.maximum(y4s[:, 0], y4s[:, 1])
            valids = (ms.labels >= 0) & (ms.half > 0)
            pred_bs = fn_sm(gid)
            sm_pr.append(average_precision_score(y_turns[valids], pred_bs[valids]))
            sm_roc.append(roc_auc_score(y_turns[valids], pred_bs[valids]))
        print(
            f"    {bname:62s} | Sportec LOMO PR-AUC={np.mean(sp_pr):.4f} (ROC={np.mean(sp_roc):.4f}) "
            f"| Sample PR-AUC={np.mean(sm_pr):.4f} (ROC={np.mean(sm_roc):.4f})",
            flush=True,
        )

    # =========================================================================
    # PART 3: STRICT 7-FOLD NESTED LOMO SELECTION OF DECODER V2 & XDX CONFIG
    # =========================================================================
    print("\n--- PART 3: STRICT 7-FOLD NESTED LOMO SELECTION ON SPORTEC ---", flush=True)
    # Precompute MoE probabilities & p_change for all 7 Sportec folds on v3 and v6
    moe_p3, moe_c3, moe_p6, moe_c6 = {}, {}, {}, {}
    for T in MATCH_IDS:
        d = oof[T]
        moe_p3[T], moe_c3[T] = moe_combine(
            d["g_v1"], d["tr_v1"], d["g_pl"], d["tr_pl"],
            d["g_v6_on_v3"], d["tr_v6_on_v3"], d["tcn"], d["tcnc"], d["gru"], d["gruc"],
            w_pl=0.5, w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], w_gbm=0.55, w_c=0.50, mode="linear",
        )
        moe_p6[T], moe_c6[T] = moe_combine(
            d["g_v3_on_v6"], d["tr_v3_on_v6"], d["g_v3_on_v6"], d["tr_v3_on_v6"],
            d["g_v6_on_v6"], d["tr_v6_on_v6"], d["tcn"], d["tcnc"], d["gru"], d["gruc"],
            w_pl=0.0, w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], w_gbm=0.55, w_c=0.50, mode="linear",
        )

    # Candidate grid for strict 7-fold nested LOMO selection:
    # (w_b4d_1d, w_dir_ha, max_absorb_len, tau_absorb, max_absorb_b4d, enable_split, enable_restart_attr)
    cand_grid = [
        (0.20, 0.25, 220, 0.85, 0.25, True, True),
        (0.25, 0.35, 240, 0.85, 0.25, True, True),
        (0.25, 0.35, 240, 0.82, 0.20, True, True),
        (0.15, 0.25, 240, 0.85, 0.25, True, False),
    ]
    grid_scores_v3 = np.zeros((len(cand_grid), len(MATCH_IDS)))
    grid_scores_v6 = np.zeros((len(cand_grid), len(MATCH_IDS)))

    for g_idx, (wb4d, wdir, max_abs, tau_abs, max_b4d, en_split, en_rest) in enumerate(cand_grid):
        cfg_g = XDXConfig(
            enable_absorb=True,
            max_absorb_len=max_abs,
            interior_max_abs_x=50.0,
            interior_max_abs_y=33.5,
            tau_absorb=tau_abs,
            max_absorb_b4d=max_b4d,
            enable_split=en_split,
            enable_oob_trim=False,
            enable_restart_attr=en_rest,
        )
        for t_idx, T in enumerate(MATCH_IDS):
            dp2 = DecoderParamsV2.from_legacy(fold_dec_std[T], pi0_ha=0.001, pi0_d=0.001, w_b4d_1d=wb4d, w_dir_ha=wdir)
            pr3 = decode_v2(moe_p3[T], sportec[T].half, dp2, p_change=moe_c3[T], p_trans_4d=b4d_v3[T])
            pr3 = apply_xdx_arbitrator(pr3, moe_p3[T], sportec[T].half, X_v3[T], p_change=moe_c3[T], p_trans_4d=b4d_v3[T], cfg=cfg_g)
            pr6 = decode_v2(moe_p6[T], sportec[T].half, dp2, p_change=moe_c6[T], p_trans_4d=b4d_v6[T])
            pr6 = apply_xdx_arbitrator(pr6, moe_p6[T], sportec[T].half, X_v6[T], p_change=moe_c6[T], p_trans_4d=b4d_v6[T], cfg=cfg_g)
            grid_scores_v3[g_idx, t_idx] = metrics.score(sportec[T].labels, pr3, sportec[T].half)["spell_f1_mean"]
            grid_scores_v6[g_idx, t_idx] = metrics.score(sportec[T].labels, pr6, sportec[T].half)["spell_f1_mean"]

    nested_sel_idx = {}
    for t_idx, T in enumerate(MATCH_IDS):
        inner_mask = np.arange(len(MATCH_IDS)) != t_idx
        inner_mean = 0.5 * (grid_scores_v3[:, inner_mask].mean(axis=1) + grid_scores_v6[:, inner_mask].mean(axis=1))
        best_g = int(np.argmax(inner_mean))
        nested_sel_idx[T] = best_g
        print(
            f"  Outer fold {T}: inner-selected config #{best_g} {cand_grid[best_g]} "
            f"(inner mean={inner_mean[best_g]:.5f} -> outer v3={grid_scores_v3[best_g, t_idx]:.4f}, v6={grid_scores_v6[best_g, t_idx]:.4f})",
            flush=True,
        )

    consensus_idx = int(np.argmax(0.5 * (grid_scores_v3.mean(axis=1) + grid_scores_v6.mean(axis=1))))
    wb4d_c, wdir_c, max_abs_c, tau_abs_c, max_b4d_c, en_split_c, en_rest_c = cand_grid[consensus_idx]
    consensus_xdx_cfg = XDXConfig(
        enable_absorb=True,
        max_absorb_len=max_abs_c,
        interior_max_abs_x=50.0,
        interior_max_abs_y=33.5,
        tau_absorb=tau_abs_c,
        max_absorb_b4d=max_b4d_c,
        enable_split=en_split_c,
        enable_oob_trim=False,
        enable_restart_attr=en_rest_c,
    )
    print(f"  Consensus 7-fold config for J2/J3: #{consensus_idx} {cand_grid[consensus_idx]}", flush=True)

    # =========================================================================
    # PART 4: COMPREHENSIVE 4-BENCHMARK ABLATION MATRIX
    # =========================================================================
    # Base ensembles:
    # (ens_key, label, w_pl, gmode, gapmode, bmode, wg, wc, dmode)
    base_ensembles = [
        ("v4_base", "v4_ens_mean", 0.0, "v4", "none", "linear", 0.50, 0.50, "v4"),
        ("v5_hyb", "v5_sym_hybrid", 0.0, "none", "none", "hybrid", 0.55, 0.50, "hyb"),
        ("v6_lin", "v6_sym_plus_linear", 1.0, "none", "none", "linear", 0.55, 0.50, "v4"),
        ("v6_hyb", "v6_sym_plus_hybrid", 0.5, "none", "none", "hybrid", 0.55, 0.50, "hyb"),
        ("v6_pure", "v6_pure_plus_hybrid", 1.0, "none", "none", "hybrid", 0.50, 0.50, "hyb"),
        ("v6_moe_hyb", "v6_moe_ghost_hyb_symdec", 0.5, "phys", "phys", "hybrid", 0.55, 0.50, "hyb"),
        ("v6_moe", "v6_moe_super_champion", 0.5, "phys", "phys", "linear", 0.55, 0.50, "v4"),
    ]

    # Decoding / Arbitration stages:
    # - S0_legacy: decode_v4
    # - S1_exp01: decode_v2 (1D p_change only, calibrated positive log-odds surprise s(p, pi0=0.001))
    # - S2_exp01_02: decode_v2 (1D p_change + 4-dir SymBoundary4DirGBM)
    # - S3_exp03_s1: decode_v4 + EXP-03 Stage 1 (Interior False-D Absorption X->D->X)
    # - S4_exp03_all: decode_v4 + EXP-03 All Stages (Absorb + Split + Restart Attr)
    # - S5_w3_full: decode_v2 (1D + 4-dir SymBoundary4DirGBM) + EXP-03 All Stages (Nested LOMO)
    stages = [
        ("S0_legacy", "Legacy HSMM (`decode_v4`)"),
        ("S1_exp01", "+ EXP-01 (`decode_v2` Calibrated Surprise)"),
        ("S2_exp01_02", "+ EXP-01 + EXP-02 (`decode_v2` + 4-Dir Boundaries)"),
        ("S3_exp03_s1", "+ EXP-03 Stage 1 (Interior `X->D->X` Absorption)"),
        ("S4_exp03_all", "+ EXP-03 Full Arbitrator (Absorb + Split + Restart)"),
        ("S5_w3_full", "+ EXP-01 + EXP-02 + EXP-03 (Full Wave 3 Nested)"),
    ]

    print("\n--- PART 4: COMPREHENSIVE 4-BENCHMARK ABLATION TABLE ---", flush=True)
    header = (
        f"  {'Ensemble & Wave 3 Stage':74s} | {'J1A: LOMO v3':18s} | {'J1B: LOMO v6':14s} "
        f"| {'J2: Sample (noD)':18s} | {'J3: BAR-ATH Pub (D_f1, pos/m, %D, int_XDX)':42s}"
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    saved_subs = {}
    detailed_results = {}

    def apply_stage(stage_key: str, p: np.ndarray, c: np.ndarray, p4d: np.ndarray, half: np.ndarray, X: pd.DataFrame, dp_leg: DecoderParams, fold_T: str | None = None):
        if stage_key == "S0_legacy":
            return decode_v4(p, half, dp_leg, p_change=c)
        if stage_key == "S1_exp01":
            dp2 = DecoderParamsV2.from_legacy(dp_leg, pi0_ha=0.001, pi0_d=0.001)
            return decode_v2(p, half, dp2, p_change=c, p_trans_4d=None)
        if stage_key == "S2_exp01_02":
            wb4d, wdir = (cand_grid[nested_sel_idx[fold_T]][0], cand_grid[nested_sel_idx[fold_T]][1]) if fold_T else (wb4d_c, wdir_c)
            dp2 = DecoderParamsV2.from_legacy(dp_leg, pi0_ha=0.001, pi0_d=0.001, w_b4d_1d=wb4d, w_dir_ha=wdir)
            return decode_v2(p, half, dp2, p_change=c, p_trans_4d=p4d)
        if stage_key == "S3_exp03_s1":
            pr = decode_v4(p, half, dp_leg, p_change=c)
            cfg1 = replace(consensus_xdx_cfg, enable_split=False, enable_oob_trim=False, enable_restart_attr=False)
            return apply_xdx_arbitrator(pr, p, half, X, p_change=c, p_trans_4d=p4d, cfg=cfg1)
        if stage_key == "S4_exp03_all":
            pr = decode_v4(p, half, dp_leg, p_change=c)
            return apply_xdx_arbitrator(pr, p, half, X, p_change=c, p_trans_4d=p4d, cfg=consensus_xdx_cfg)
        if stage_key == "S5_w3_full":
            if fold_T is not None:
                wb4d, wdir, max_abs, tau_abs, max_b4d, en_split, en_rest = cand_grid[nested_sel_idx[fold_T]]
                cfg_u = XDXConfig(
                    enable_absorb=True, max_absorb_len=max_abs, interior_max_abs_x=50.0, interior_max_abs_y=33.5,
                    tau_absorb=tau_abs, max_absorb_b4d=max_b4d, enable_split=en_split, enable_oob_trim=False, enable_restart_attr=en_rest,
                )
            else:
                wb4d, wdir = wb4d_c, wdir_c
                cfg_u = consensus_xdx_cfg
            dp2 = DecoderParamsV2.from_legacy(dp_leg, pi0_ha=0.001, pi0_d=0.001, w_b4d_1d=wb4d, w_dir_ha=wdir)
            pr = decode_v2(p, half, dp2, p_change=c, p_trans_4d=p4d)
            return apply_xdx_arbitrator(pr, p, half, X, p_change=c, p_trans_4d=p4d, cfg=cfg_u)
        raise ValueError(stage_key)

    for ens_key, elabel, w_pl, gmode, gapmode, bmode, wg, wc, dmode in base_ensembles:
        stages_to_run = [("S0_legacy", "Baseline (`decode_v4`)")] if ens_key == "v4_base" else stages
        for skey, slabel in stages_to_run:
            row_name = f"{elabel} | {slabel}"
            j1_v3, j1_v6 = [], []
            per_fold_v3, per_fold_v6 = {}, {}
            xdx_int_v3 = 0

            for T in MATCH_IDS:
                d = oof[T]
                dp = fold_dec_std[T] if dmode == "v4" else (loo_dec_lin[T] if dmode == "lin" else loo_dec_hyb[T])
                if gmode == "v4":
                    p3 = blend_proba(d["g_orig"], d["t_orig"], w_gbm=0.50, mode="linear")
                    c3 = 0.50 * d["tr_orig"] + 0.50 * d["tc_orig"]
                    p6 = blend_proba(d["g_v3_on_v6"], d["t_orig"], w_gbm=0.50, mode="linear")
                    c6 = 0.50 * d["tr_v3_on_v6"] + 0.50 * d["tc_orig"]
                else:
                    wg_v3 = gates_v3[T][0] if gmode == "phys" else 0.0
                    wgap_v3 = gates_v3[T][1] if gapmode == "phys" else 0.0
                    p3, c3 = moe_combine(
                        d["g_v1"], d["tr_v1"], d["g_pl"], d["tr_pl"],
                        d["g_v6_on_v3"], d["tr_v6_on_v3"], d["tcn"], d["tcnc"], d["gru"], d["gruc"],
                        w_pl=w_pl, w_ghost=wg_v3, w_gap=wgap_v3, w_gbm=wg, w_c=wc, mode=bmode,
                    )
                    wg_v6 = gates_v6[T][0] if gmode == "phys" else 0.0
                    wgap_v6 = gates_v6[T][1] if gapmode == "phys" else 0.0
                    p6, c6 = moe_combine(
                        d["g_v3_on_v6"], d["tr_v3_on_v6"], d["g_v3_on_v6"], d["tr_v3_on_v6"],
                        d["g_v6_on_v6"], d["tr_v6_on_v6"], d["tcn"], d["tcnc"], d["gru"], d["gruc"],
                        w_pl=0.0, w_ghost=wg_v6, w_gap=wgap_v6, w_gbm=wg, w_c=wc, mode=bmode,
                    )

                pred3 = apply_stage(skey, p3, c3, b4d_v3[T], sportec[T].half, X_v3[T], dp, fold_T=T)
                pred6 = apply_stage(skey, p6, c6, b4d_v6[T], sportec[T].half, X_v6[T], dp, fold_T=T)
                f1_3 = metrics.score(sportec[T].labels, pred3, sportec[T].half)["spell_f1_mean"]
                f1_6 = metrics.score(sportec[T].labels, pred6, sportec[T].half)["spell_f1_mean"]
                j1_v3.append(f1_3)
                j1_v6.append(f1_6)
                per_fold_v3[T] = f1_3
                per_fold_v6[T] = f1_6
                _, n_int = count_interior_xdx_islands(pred3, sportec[T].half, X_v3[T])
                xdx_int_v3 += n_int

            # Judge 2: Metrica Sample (3 games)
            s_raw, s_nod = [], []
            per_game_raw, per_game_nod = {}, {}
            d_use = dec_v4 if dmode == "v4" else (dec_v5_lin if dmode == "lin" else dec_v5_hyb)
            for gid in GAMES:
                ms = sample_matches[gid]
                dh = sample_h[gid]
                dp_s = pe["sample"][gid]
                dv6 = sample_v6[gid]
                if gmode == "v4":
                    ps = blend_proba(dh["g_orig"], dh["t_orig"], w_gbm=0.50, mode="linear")
                    cs = 0.50 * dh["tr_orig"] + 0.50 * dh["tc_orig"]
                else:
                    wg_s = gates_sample[gid][0] if gmode == "phys" else 0.0
                    wgap_s = gates_sample[gid][1] if gapmode == "phys" else 0.0
                    ps, cs = moe_combine(
                        dh["g_sym"], dh["tr_sym"], dp_s["g_plus_sym"], dp_s["tr_plus_sym"],
                        dv6["g_v6"], dv6["tr_v6"], dh["t_sym"], dh["tc_sym"], dh["b_sym"], dh["bc_sym"],
                        w_pl=w_pl, w_ghost=wg_s, w_gap=wgap_s, w_gbm=wg, w_c=wc, mode=bmode,
                    )
                pred_s = apply_stage(skey, ps, cs, b4d_j23["sample"][gid], ms.half, X_sample[gid], d_use, fold_T=None)
                f_r = metrics.score(ms.labels, pred_s, ms.half)["spell_f1_mean"]
                f_n = metrics.score(without_dead_margin(ms.labels, ms.half), pred_s, ms.half)["spell_f1_mean"]
                s_raw.append(f_r)
                s_nod.append(f_n)
                per_game_raw[gid] = f_r
                per_game_nod[gid] = f_n

            # Judge 3: BAR-ATH Broadcast-Repaired
            if gmode == "v4":
                pa = blend_proba(atd_rep_h["g_orig"], atd_rep_h["t_orig"], w_gbm=0.50, mode="linear")
                ca = 0.50 * atd_rep_h["tr_orig"] + 0.50 * atd_rep_h["tc_orig"]
            else:
                wg_a = gate_atd[0] if gmode == "phys" else 0.0
                wgap_a = gate_atd[1] if gapmode == "phys" else 0.0
                pa, ca = moe_combine(
                    atd_rep_h["g_sym"], atd_rep_h["tr_sym"], pe["atd_rep"]["g_plus_sym"], pe["atd_rep"]["tr_plus_sym"],
                    atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"], atd_rep_h["t_sym"], atd_rep_h["tc_sym"],
                    atd_rep_h["b_sym"], atd_rep_h["bc_sym"],
                    w_pl=w_pl, w_ghost=wg_a, w_gap=wgap_a, w_gbm=wg, w_c=wc, mode=bmode,
                )
            pred_a = apply_stage(skey, pa, ca, b4d_j23["atd_rep"], half_dec, Xa_rep, d_use, fold_T=None)
            pred_a[second_start:kick2] = DEAD
            pred_a[~atd.graded] = DEAD

            pub = metrics.score_public(pred_a, bootstrap_n=100)
            pl = plausibility(pred_a, atd.half, atd.graded)
            _, n_int_atd = count_interior_xdx_islands(pred_a, half_dec, Xa_rep)
            sp = pub["spell_f1_mean"]
            df1 = pub["per_class"]["D"]["f1"]
            pm = pl["poss_per_min"]
            pd_val = pl["pct_D"]

            print(
                f"  {row_name:74s} | {np.mean(j1_v3):.4f} (s={np.std(j1_v3):.4f}) | {np.mean(j1_v6):.4f}         "
                f"| {np.mean(s_raw):.4f} ({np.mean(s_nod):.4f}) | {sp:.4f} (D={df1:.3f}, {pm:.2f}/m, {pd_val:.1f}%D, int={n_int_atd})",
                flush=True,
            )

            detailed_results[(ens_key, skey)] = {
                "j1_v3_mean": float(np.mean(j1_v3)),
                "j1_v3_std": float(np.std(j1_v3)),
                "j1_v6_mean": float(np.mean(j1_v6)),
                "j1_v6_std": float(np.std(j1_v6)),
                "j2_raw": float(np.mean(s_raw)),
                "j2_nod": float(np.mean(s_nod)),
                "j3_pub": float(sp),
                "j3_df1": float(df1),
                "j3_pm": float(pm),
                "j3_pct_d": float(pd_val),
                "xdx_int_v3": xdx_int_v3,
                "xdx_int_atd": n_int_atd,
                "per_fold_v3": per_fold_v3,
                "per_fold_v6": per_fold_v6,
                "per_game_raw": per_game_raw,
                "per_game_nod": per_game_nod,
                "pred_atd": pred_a.copy(),
            }

            # Save validated Wave 3 submissions for top ensembles
            sub_map = {
                ("v6_moe", "S5_w3_full"): "submission_v7_w3_moe_super_xdx.csv",
                ("v6_moe_hyb", "S5_w3_full"): "submission_v7_w3_moe_ghost_hyb_xdx.csv",
                ("v6_hyb", "S5_w3_full"): "submission_v7_w3_sym_plus_hybrid_xdx.csv",
                ("v6_pure", "S5_w3_full"): "submission_v7_w3_pure_plus_hybrid_xdx.csv",
                ("v6_lin", "S5_w3_full"): "submission_v7_w3_sym_plus_linear_xdx.csv",
                ("v5_hyb", "S5_w3_full"): "submission_v7_w3_v5_sym_hybrid_xdx.csv",
            }
            if (ens_key, skey) in sub_map:
                sname = sub_map[(ens_key, skey)]
                info = save_and_validate_submission(sub_dir / sname, pred_a, atd, second_start, kick2)
                saved_subs[sname] = (pred_a.copy(), info)

        if ens_key != "v4_base":
            print("  " + "-" * (len(header) - 2), flush=True)

    # =========================================================================
    # PART 5: PER-MATCH BREAKDOWN (SPORTEC 7 FOLDS & METRICA SAMPLE 3 GAMES)
    # =========================================================================
    print("\n--- PART 5: PER-MATCH BREAKDOWN (v6_moe_super_champion: Legacy S0 vs Wave 3 Full S5) ---", flush=True)
    r_s0 = detailed_results[("v6_moe", "S0_legacy")]
    r_s5 = detailed_results[("v6_moe", "S5_w3_full")]
    for T in MATCH_IDS:
        d3 = r_s5["per_fold_v3"][T] - r_s0["per_fold_v3"][T]
        d6 = r_s5["per_fold_v6"][T] - r_s0["per_fold_v6"][T]
        print(
            f"  Sportec {T}: v3 {r_s0['per_fold_v3'][T]:.4f} -> {r_s5['per_fold_v3'][T]:.4f} ({d3:+.4f}) | "
            f"v6 {r_s0['per_fold_v6'][T]:.4f} -> {r_s5['per_fold_v6'][T]:.4f} ({d6:+.4f})",
            flush=True,
        )
    for gid in GAMES:
        dr = r_s5["per_game_raw"][gid] - r_s0["per_game_raw"][gid]
        dn = r_s5["per_game_nod"][gid] - r_s0["per_game_nod"][gid]
        print(
            f"  Metrica Sample game_{gid}: raw {r_s0['per_game_raw'][gid]:.4f} -> {r_s5['per_game_raw'][gid]:.4f} ({dr:+.4f}) | "
            f"noD {r_s0['per_game_nod'][gid]:.4f} -> {r_s5['per_game_nod'][gid]:.4f} ({dn:+.4f})",
            flush=True,
        )

    # =========================================================================
    # PART 6: SAVED & VALIDATED WAVE 3 SUBMISSIONS
    # =========================================================================
    print("\n--- PART 6: SAVED & VALIDATED WAVE 3 SUBMISSIONS ---", flush=True)
    for sname, (pred_vec, info) in saved_subs.items():
        print(
            f"  [VALIDATED] {sname:44s}: rows={info['rows']} (graded={info['graded']}) | "
            f"%H={info['pct_H']:.2f}% %A={info['pct_A']:.2f}% %D={info['pct_D']:.2f}%",
            flush=True,
        )

    print(f"\nWave 3 evaluation completed in {(time.time() - t0):.1f}s.", flush=True)


if __name__ == "__main__":
    main()
