"""Red-Team Forensic Audit — PRUEBA 4: Perfect Synthesis & Sub-Rule Ablation Investigation.
Tests:
1. Exact sub-rule ablation of `apply_dead_wedge_arbitrator` (Case 1 `Y->X->D`, Case 2 `D->X->Y`, Case 3 `D->X->D`,
   and visibility-shielded Case 2 / Case 3) across all 4 Judges (J1A, J1B-Std, J1B-Gen, J2-Raw, J2-noD, J3)
   and the 85 unseen minutes of BAR-ATH.
2. Open-Play-Shielded Inpainter (`EXP-14-OpenPlay`): Fixes the 85.7% Dead-Ball contamination discovered in Prueba 1.2
   by gating `w_inp` strictly to true mid-field open-play tracking blackouts (rolling `p_D < 0.08`, never altering `p_D` or `c`).
3. Determines mathematically whether `v85_forensic_grand_champion` can be improved without any tradeoff,
   or if `v85_forensic_grand_champion` is already the Pareto-optimal global maximum across all 4 Judges.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "py_pkgs"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "forensic_v7"))
sys.path.insert(0, str(ROOT / "evaluation"))

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_3_frame_models import plausibility
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, HOME
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.forensic_shields_v85 import (
    PostImpactTurnoverClassifier,
    PostImpactTurnoverConfig,
    ShieldedBoundaryOffsetAligner,
    ShieldedTouchTypeArbitrator,
)
from src.decode.hybrid_gate_v8 import (
    blend_6fam_heads_v8,
    compute_hybrid_gate_v8,
    compute_inpaint_gate_v8,
    compute_w_clean_v8,
)
from src.decode.spell_reranker import apply_spell_reranker, rle_runs
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


def apply_dead_wedge_custom(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame,
    p_change: np.ndarray,
    p4d: np.ndarray,
    use_case1_yxd: bool = True,
    use_case2_dxy: bool = True,
    use_case3_dxd: bool = True,
    require_vis_case2: bool = False,
    require_vis_case3: bool = False,
) -> np.ndarray:
    if "ball_visible" in X_df.columns:
        vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=1.0)
    elif "d1_h_vis" in X_df.columns:
        vis = (np.nan_to_num(X_df["d1_h_vis"].to_numpy(np.float64), nan=99.0) < 90.0).astype(np.float64)
    else:
        vis = np.ones(len(pred), dtype=np.float64)

    live_m = ((half == 1) | (half == 2)) & ((pred == HOME) | (pred == AWAY))
    if np.any(live_m) and float(np.mean(vis[live_m])) < 0.80:
        return pred.copy()

    out = np.asarray(pred, dtype=np.int8).copy()
    valid = (half == 1) | (half == 2)
    arr = np.where(valid, out, -1)
    runs = rle_runs(arr)

    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=50.0)

    for idx in range(1, len(runs) - 1):
        s, e, x = runs[idx]
        if x not in (HOME, AWAY):
            continue
        ps, pe, prev_y = runs[idx - 1]
        ns, ne, next_y = runs[idx + 1]
        if half[s] != half[ps] or half[s] != half[ns]:
            continue
        L = e - s
        opp = AWAY if x == HOME else HOME

        mean_pX = float(np.mean(proba[s:e, x]))
        max_pX = float(np.max(proba[s:e, x]))
        mean_pD = float(np.mean(proba[s:e, DEAD]))
        max_pD = float(np.max(proba[s:e, DEAD]))
        d1_x = d1h[s:e] if x == HOME else d1a[s:e]
        min_d1_x = float(np.min(d1_x))
        mean_vis = float(np.mean(vis[s:e]))

        if use_case1_yxd and prev_y == opp and next_y == DEAD and L <= 90:
            dir_yx = 0 if (opp == HOME and x == AWAY) else 1
            b4d_in = float(np.max(p4d[max(0, s - 6) : min(len(out), s + 7), dir_yx]))
            cond_d_spill = (mean_pX < 0.62 and (mean_pD > 0.20 or max_pD > 0.50) and b4d_in < 0.25)
            cond_no_ctl = (min_d1_x > 2.8 and mean_vis > 0.6 and mean_pX < 0.68 and b4d_in < 0.45)
            cond_weak_x = (mean_pX < 0.54 and max_pX < 0.72 and b4d_in < 0.35)

            if cond_d_spill or cond_no_ctl or cond_weak_x:
                p_y_seg = proba[s:e, opp]
                p_d_seg = proba[s:e, DEAD]
                if float(np.max(p_d_seg)) > 0.45 and mean_pD > 0.20:
                    out[s:e] = DEAD
                else:
                    diff = np.log(np.clip(p_y_seg, 1e-5, 1.0)) - np.log(np.clip(p_d_seg, 1e-5, 1.0))
                    c_sum = np.concatenate([[0.0], np.cumsum(diff)])
                    k_best = int(np.argmax(c_sum))
                    out[s : s + k_best] = opp
                    out[s + k_best : e] = DEAD

        elif use_case2_dxy and prev_y == DEAD and next_y == opp and L <= 85:
            if require_vis_case2 and mean_vis < 0.5:
                continue
            cond_no_ctl_dxy = (min_d1_x > 3.2 and mean_vis > 0.6 and mean_pX < 0.74)
            cond_weak_dxy = (mean_pX < 0.56 and max_pX < 0.72)
            if cond_no_ctl_dxy or cond_weak_dxy:
                p_d_seg = proba[s:e, DEAD]
                p_y_seg = proba[s:e, opp]
                diff = np.log(np.clip(p_d_seg, 1e-5, 1.0)) - np.log(np.clip(p_y_seg, 1e-5, 1.0))
                c_sum = np.concatenate([[0.0], np.cumsum(diff)])
                k_best = int(np.argmax(c_sum))
                out[s : s + k_best] = DEAD
                out[s + k_best : e] = opp

        elif use_case3_dxd and prev_y == DEAD and next_y == DEAD and L <= 95:
            if require_vis_case3 and mean_vis < 0.5:
                continue
            if (mean_pX < 0.62 and mean_pD > 0.12) or (min_d1_x > 4.0 and mean_pX < 0.72):
                out[s:e] = DEAD

    return out


def compute_openplay_inpaint_gate(
    w_inp_raw: np.ndarray,
    p_base: np.ndarray,
    X_df: pd.DataFrame,
    half: np.ndarray,
    max_p_dead_roll: float = 0.06,
    win_roll: int = 125,
) -> np.ndarray:
    """Restrict `w_inp` strictly to genuine open-play mid-field blackouts where the 5-second rolling
    dead-ball probability is `< max_p_dead_roll` and the ball is inside the pitch.
    """
    p_d = p_base[:, 2].astype(np.float64)
    k = np.ones(win_roll, dtype=np.float64) / float(win_roll)
    p_d_roll = np.ones_like(p_d)
    for hv in (1, 2):
        idx = np.flatnonzero(half == hv)
        if len(idx) > 0:
            p_d_roll[idx] = np.convolve(p_d[idx], k, mode="same")
    out_pitch = (
        np.nan_to_num(X_df["ball_outside_pitch"].to_numpy(np.float64), nan=0.0)
        if "ball_outside_pitch" in X_df.columns
        else np.zeros(len(half))
    )
    open_play_mask = np.clip(1.0 - p_d_roll / max_p_dead_roll, 0.0, 1.0) * (out_pitch < 0.5).astype(np.float64)
    return (w_inp_raw * open_play_mask).astype(np.float32)


def blend_openplay_inpaint(
    p_6f: np.ndarray,
    p_inp: np.ndarray,
    w_inp_op: np.ndarray,
) -> np.ndarray:
    """Blend ONLY `H vs A` conditional ratio `q_h` during open-play blackouts, NEVER altering `p_D`."""
    wi = np.asarray(w_inp_op, dtype=np.float32)
    if not np.any(wi > 0.0):
        return p_6f
    p_d = p_6f[:, 2]
    q_out = p_6f[:, 0] / np.maximum(p_6f[:, 0] + p_6f[:, 1], 1e-6)
    q_inp = p_inp[:, 0] / np.maximum(p_inp[:, 0] + p_inp[:, 1], 1e-6)
    q_h = np.clip((1.0 - wi) * q_out + wi * q_inp, 1e-6, 1.0 - 1e-6)
    return np.column_stack([(1.0 - p_d) * q_h, (1.0 - p_d) * (1.0 - q_h), p_d]).astype(np.float32)


def main() -> None:
    print("=" * 130)
    print("RED-TEAM FORENSIC AUDIT — PRUEBA 4: PERFECT SYNTHESIS & SUB-RULE ABLATION ACROSS ALL 4 JUDGES")
    print("=" * 130, flush=True)

    OUT_DIR = ROOT / "experiments" / "outputs"
    W8A_DIR = OUT_DIR / "exp_w8a_oof"
    FOR_DIR = OUT_DIR / "forensic_v7"

    sportec = load_sportec()
    matches = sorted(sportec.keys())
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

    sym_dir = OUT_DIR / "exp_b_sym_oof"
    plus_dir = OUT_DIR / "exp_c_plus_oof"
    g_dir = OUT_DIR / "exp_g_oof"
    h_dir = OUT_DIR / "exp_h_oof"
    w3_dir = OUT_DIR / "exp_w3_oof"
    w4_dir = OUT_DIR / "exp_w4_oof"
    w5_dir = OUT_DIR / "exp_w5_tree_oof"
    w5b_dir = OUT_DIR / "exp_w5b_oof"
    w6_dir = OUT_DIR / "exp_w6_oof"
    w7a_dir = OUT_DIR / "exp_w7a_oof"
    w7b_dir = OUT_DIR / "exp_w7b_oof"

    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)

    oof, inp_oof = {}, {}
    for T in MATCH_IDS:
        zb = np.load(sym_dir / f"sym_{T}.npz")
        zt = np.load(sym_dir / f"tcn_sym_{T}.npz")
        zp = np.load(plus_dir / f"plus_{T}.npz")
        zg = np.load(g_dir / f"ghost_sym_{T}.npz")
        zh = np.load(h_dir / f"bigru_sym_{T}.npz")
        zw3 = np.load(w3_dir / f"b4d_{T}.npz")
        zw4 = np.load(w4_dir / f"physv2_{T}.npz")
        zw5 = np.load(w5_dir / f"cb_xgb_{T}.npz")
        zw5b_tcn = np.load(w5b_dir / f"tcn_physv2_{T}.npz")
        zw5b_gru = np.load(w5b_dir / f"bigru_physv2_{T}.npz")
        zw7a = np.load(w7a_dir / f"pset_{T}.npz")
        zw7b_v6 = np.load(w7b_dir / f"v6_genuine_{T}.npz")
        zw8a = np.load(W8A_DIR / f"w8a_inpaint_{T}.npz")
        oof[T] = {
            "g_v1": zb["symgbm_tta"], "tr_v1": zb["symtrans_tta"],
            "g_pl": zp["symgbm_plus_tta"], "tr_pl": zp["symtrans_plus_tta"],
            "g_pl_on_v6": zw7b_v6["symgbm_plus_tta_on_v6"], "tr_pl_on_v6": zw7b_v6["symtrans_plus_tta_on_v6"],
            "g_v6_on_v3": zg["g_v6_on_v3"], "tr_v6_on_v3": zg["tr_v6_on_v3"],
            "g_v3_on_v6": zg["g_v3_on_v6"], "tr_v3_on_v6": zg["tr_v3_on_v6"],
            "g_v6_on_v6": zg["g_v6_on_v6"], "tr_v6_on_v6": zg["tr_v6_on_v6"],
            "tcn": zt["symtcn_tta_proba"], "tcnc": zt["symtcn_tta_change"],
            "gru": zh["symbigru_tta_proba"], "gruc": zh["symbigru_tta_change"],
            "p4d_v3": zw3["p4d_v3"], "p4d_v6": zw3["p4d_v6"],
            "g_p2_v3": zw4["symgbm_physv2_tta"], "tr_p2_v3": zw4["symtrans_physv2_tta"],
            "g_p2_v6": zw4["symgbm_physv2_tta_on_v6"], "tr_p2_v6": zw4["symtrans_physv2_tta_on_v6"],
            "cb_p2_v3": zw5["symcatboost_physv2_tta"], "xgb_p2_v3": zw5["symxgb_physv2_tta"],
            "cb_p2_v6": zw5["symcatboost_physv2_tta_on_v6"], "xgb_p2_v6": zw5["symxgb_physv2_tta_on_v6"],
            "tcn_p2_v3": zw5b_tcn["v3_proba"], "tcnc_p2_v3": zw5b_tcn["v3_change"],
            "tcn_p2_v6": zw5b_tcn["v6_proba"], "tcnc_p2_v6": zw5b_tcn["v6_change"],
            "gru_p2_v3": zw5b_gru["v3_proba"], "gruc_p2_v3": zw5b_gru["v3_change"],
            "gru_p2_v6": zw5b_gru["v6_proba"], "gruc_p2_v6": zw5b_gru["v6_change"],
            "pset_v3": zw7a["v3_proba"], "psetc_v3": zw7a["v3_change"],
            "pset_v6": zw7a["v6_proba"], "psetc_v6": zw7a["v6_change"],
        }
        inp_oof[T] = {
            "p_inp_v3": zw8a["p_inp_v3"],
            "c_inp_v3": zw8a["c_inp_v3"],
            "w_inp_v3": compute_inpaint_gate_v8(X_v3[T], sportec[T].half, gates_v3[T][0], gates_v3[T][1]),
            "p_inp_v6": zw8a["p_inp_v6"],
            "c_inp_v6": zw8a["c_inp_v6"],
            "w_inp_v6": compute_inpaint_gate_v8(X_v6[T], sportec[T].half, gates_v6[T][0], gates_v6[T][1]),
        }

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
    with open(w6_dir / "w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)
    with open(W8A_DIR / "w8a_j23_preds.pkl", "rb") as f:
        j23_inp = pickle.load(f)

    for gid in GAMES:
        j23_inp["sample"][gid]["w_inp"] = compute_inpaint_gate_v8(
            X_sample[gid], sample_matches[gid].half, gates_sample[gid][0], gates_sample[gid][1]
        )
    j23_inp["atd_raw"]["w_inp"] = compute_inpaint_gate_v8(X_atd_raw, atd.half, gate_atd_raw[0], gate_atd_raw[1])
    j23_inp["atd_rep"]["w_inp"] = compute_inpaint_gate_v8(X_atd_rep, atd.half, gate_atd_rep[0], gate_atd_rep[1])

    best_calib_cfg: TransductiveCalibConfig = w6_sum["best_calib_cfg"]
    lomo_rerank_cfgs = w6_sum["lomo_rerank_cfgs"]
    full_rerank_cfg = w6_sum["full_rerank_cfg"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)

    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )

    def build_streams(inp_mode: str = "off", inp_scale: float = 0.0) -> dict:
        s3_d, s3_ha, s3_c, ps_ha = 0.05, 0.30, 0.18, 0.08
        streams = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}, "atd_raw": None, "atd_rep": None}

        for T in MATCH_IDS:
            d = oof[T]
            di = inp_oof[T]
            h = sportec[T].half

            a_v3 = compute_hybrid_gate_v8(X_v3[T], h, gates_v3[T][0])
            wc_v3 = compute_w_clean_v8(X_v3[T], h, gates_v3[T][0])
            p3, c3 = synthesize_head_decoupled_proba(
                d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"],
                d["tcn"], d["gru"], d["tr_pl"], d["tr_p2_v3"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v3"], d["tr_v6_on_v3"],
                w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v3"], bigru_physv2_proba=d["gru_p2_v3"],
                tcn_physv2_change=d["tcnc_p2_v3"], bigru_physv2_change=d["gruc_p2_v3"],
                alpha_p2=a_v3,
            )
            p3, c3 = blend_6fam_heads_v8(
                p3, c3, d["tcn"], d["gru"], d["gru_p2_v3"], d["pset_v3"],
                d["tcnc"], d["gruc"], d["gruc_p2_v3"], d["psetc_v3"],
                a_v3, wc_v3, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True,
            )
            if inp_mode == "openplay" and inp_scale > 0:
                wi_op = inp_scale * compute_openplay_inpaint_gate(di["w_inp_v3"], p3, X_v3[T], h)
                p3 = blend_openplay_inpaint(p3, di["p_inp_v3"], wi_op)
            p3 = calibrate_match_transductive(p3, h, X_v3[T], best_calib_cfg)
            f3 = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"], d["pset_v3"]]
            streams["v3"][T] = (p3, c3, d["p4d_v3"], f3)

            a_v6 = compute_hybrid_gate_v8(X_v6[T], h, gates_v6[T][0])
            wc_v6 = compute_w_clean_v8(X_v6[T], h, gates_v6[T][0])
            p6s, c6s = synthesize_head_decoupled_proba(
                d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn"], d["gru"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=a_v6,
            )
            p6s, c6s = blend_6fam_heads_v8(
                p6s, c6s, d["tcn"], d["gru"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc"], d["gruc"], d["gruc_p2_v6"], d["psetc_v6"],
                a_v6, wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True,
            )
            if inp_mode == "openplay" and inp_scale > 0:
                wi_op = inp_scale * compute_openplay_inpaint_gate(di["w_inp_v6"], p6s, X_v6[T], h)
                p6s = blend_openplay_inpaint(p6s, di["p_inp_v6"], wi_op)
            p6s = calibrate_match_transductive(p6s, h, X_v6[T], best_calib_cfg)
            f6s = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tcn_p2_v6"], d["gru_p2_v6"], d["pset_v6"]]
            streams["v6_std"][T] = (p6s, c6s, d["p4d_v6"], f6s)

            p6g, c6g = synthesize_head_decoupled_proba(
                d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn_p2_v6"], d["gru_p2_v6"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc_p2_v6"], d["gruc_p2_v6"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=np.ones(len(h), dtype=np.float32),
            )
            p6g, c6g = blend_6fam_heads_v8(
                p6g, c6g, d["tcn_p2_v6"], d["gru_p2_v6"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc_p2_v6"], d["gruc_p2_v6"], d["gruc_p2_v6"], d["psetc_v6"],
                np.ones(len(h), dtype=np.float32), wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=False,
            )
            if inp_mode == "openplay" and inp_scale > 0:
                wi_op = inp_scale * compute_openplay_inpaint_gate(di["w_inp_v6"], p6g, X_v6[T], h)
                p6g = blend_openplay_inpaint(p6g, di["p_inp_v6"], wi_op)
            p6g = calibrate_match_transductive(p6g, h, X_v6[T], best_calib_cfg)
            f6g = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn_p2_v6"], d["gru_p2_v6"], d["g_v6_on_v6"], d["pset_v6"]]
            streams["v6_gen"][T] = (p6g, c6g, d["p4d_v6"], f6g)

        for gid in GAMES:
            ms = sample_matches[gid]
            dh = sample_h[gid]
            dpl = pe_plus["sample"][gid]
            dv6 = sample_v6[gid]
            dp2 = pe_p2["sample"][gid]
            dw5 = pe_w5["sample"][gid]
            dw5b = pe_w5b["sample"][gid]
            dpset = pset_j23["sample"][gid]
            dinp = j23_inp["sample"][gid]
            a_s = compute_hybrid_gate_v8(X_sample[gid], ms.half, gates_sample[gid][0])
            wc_s = compute_w_clean_v8(X_sample[gid], ms.half, gates_sample[gid][0])

            ps, cs = synthesize_head_decoupled_proba(
                dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
                dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
                ms.half, dv6["g_v6"], dv6["tr_v6"],
                w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main,
                tcn_physv2_proba=dw5b["tcn_physv2_sym"], bigru_physv2_proba=dw5b["bigru_physv2_sym"],
                tcn_physv2_change=dw5b["tcnc_physv2_sym"], bigru_physv2_change=dw5b["bigruc_physv2_sym"],
                alpha_p2=a_s,
            )
            ps, cs = blend_6fam_heads_v8(
                ps, cs, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"],
                dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"],
                a_s, wc_s, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True,
            )
            if inp_mode == "openplay" and inp_scale > 0:
                wi_op = inp_scale * compute_openplay_inpaint_gate(dinp["w_inp"], ps, X_sample[gid], ms.half)
                ps = blend_openplay_inpaint(ps, dinp["p_inp"], wi_op)
            ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], best_calib_cfg)
            fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"]]
            streams["sample"][gid] = (ps, cs, b4d_j23["sample"][gid], fs)

        for vkey, X_df, gate_t, t2_p, g2_p, t2_c, g2_c, gv6_p, gv6_c in [
            ("atd_raw", X_atd_raw, gate_atd_raw, pe_plus["atd_raw"]["t_sym"], atd_raw_extra["b_sym"], pe_plus["atd_raw"]["tc_sym"], atd_raw_extra["bc_sym"], atd_raw_extra["g_v6"], atd_raw_extra["tr_v6"]),
            ("atd_rep", X_atd_rep, gate_atd_rep, atd_rep_h["t_sym"], atd_rep_h["b_sym"], atd_rep_h["tc_sym"], atd_rep_h["bc_sym"], atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"]),
        ]:
            dpl_a = pe_plus[vkey]
            dp2_a = pe_p2[vkey]
            dw5_a = pe_w5[vkey]
            dw5b_a = pe_w5b[vkey]
            dpset_a = pset_j23[vkey]
            dinp_a = j23_inp[vkey]
            a_a = compute_hybrid_gate_v8(X_df, atd.half, gate_t[0])
            wc_a = compute_w_clean_v8(X_df, atd.half, gate_t[0])
            pa, ca = synthesize_head_decoupled_proba(
                dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"],
                t2_p, g2_p, dpl_a["tr_plus_sym"], dp2_a["tr_physv2_sym"], t2_c, g2_c,
                atd.half, gv6_p, gv6_c,
                w_ghost=gate_t[0], w_gap=gate_t[1], cfg=scfg_main,
                tcn_physv2_proba=dw5b_a["tcn_physv2_sym"], bigru_physv2_proba=dw5b_a["bigru_physv2_sym"],
                tcn_physv2_change=dw5b_a["tcnc_physv2_sym"], bigru_physv2_change=dw5b_a["bigruc_physv2_sym"],
                alpha_p2=a_a,
            )
            pa, ca = blend_6fam_heads_v8(
                pa, ca, t2_p, g2_p, dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"],
                t2_c, g2_c, dw5b_a["bigruc_physv2_sym"], dpset_a["pset_change"],
                a_a, wc_a, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True,
            )
            if inp_mode == "openplay" and inp_scale > 0:
                wi_op = inp_scale * compute_openplay_inpaint_gate(dinp_a["w_inp"], pa, X_df, atd.half)
                pa = blend_openplay_inpaint(pa, dinp_a["p_inp"], wi_op)
            pa = calibrate_match_transductive(pa, half_atd_dec, X_df, best_calib_cfg)
            fa = [dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"], t2_p, g2_p, dw5b_a["tcn_physv2_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"]]
            streams[vkey] = (pa, ca, b4d_j23[vkey], fa)
        return streams

    print("Building streams (`st_v8_hyb` and `st_v8_openplay_inp`)...", flush=True)
    st_v8_hyb = build_streams("off", 0.0)
    st_v8_op_inp = build_streams("openplay", 0.35)

    with open(FOR_DIR / "v85_cands_cache.pkl", "rb") as f:
        v8_cands_bundle = pickle.load(f)

    _, m1_d_full, m2_d_full, bdy_d_full, pit_d_full = v8_cands_bundle["hyb"]["v3"]
    arb_shielded_full = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d_full, m2_d_full, matches)
    aln_shielded_full = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_d_full)
    pit_shielded_full = PostImpactTurnoverClassifier().fit_from_candidates(pit_d_full, matches)

    def eval_variant(
        name: str,
        st: dict,
        use_exp15: bool = True,
        use_exp16: bool = True,
        dw_c1: bool = True,
        dw_c2: bool = True,
        dw_c3: bool = True,
        dw_vis2: bool = False,
        dw_vis3: bool = False,
        use_pit: bool = True,
        redecode_j1: bool = False,
    ) -> dict:
        v3_f1, v6s_f1, v6g_f1 = [], [], []
        for m_te in matches:
            tr_ms = [m for m in matches if m != m_te]
            hl = sportec[m_te].half
            gt = sportec[m_te].labels
            for tr, X_df, f1_l in [
                ("v3", X_v3[m_te], v3_f1),
                ("v6_std", X_v6[m_te], v6s_f1),
                ("v6_gen", X_v6[m_te], v6g_f1),
            ]:
                preds_rr, m1_d, m2_d, bdy_d, pit_d = v8_cands_bundle["hyb"][tr]
                p_raw, c_raw, p4d_raw, fams_raw = st[tr][m_te]
                p = p_raw.astype(np.float64)
                c = c_raw.astype(np.float64)
                p4d = p4d_raw.astype(np.float64)
                f64 = [f.astype(np.float64) for f in fams_raw]

                if redecode_j1:
                    p_xdx = apply_xdx_arbitrator(
                        decode_v4(p_raw, hl, loo_dec_hyb[m_te], p_change=c_raw),
                        p_raw, hl, X_df, p_change=c_raw, p_trans_4d=p4d_raw, cfg=xdx_cfg_st1,
                    )
                    pr_base = apply_spell_reranker(p_xdx, p_raw, hl, X_df, c_raw, p4d_raw, fams_raw, cfg=lomo_rerank_cfgs[m_te])
                else:
                    pr_base = preds_rr[m_te].copy()

                pr = pr_base.copy()
                if use_exp15:
                    arb = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, tr_ms)
                    pr = arb.apply(pr, p, hl, X_df, c, p4d, fams=f64, m1_cands=m1_d[m_te], m2_cands=m2_d[m_te])

                if (dw_c1 or dw_c2 or dw_c3) and tr != "v6_gen":
                    pr = apply_dead_wedge_custom(pr, p, hl, X_df, c, p4d, dw_c1, dw_c2, dw_c3, dw_vis2, dw_vis3)

                if use_exp16:
                    aln = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=False).fit({m: bdy_d[m] for m in tr_ms})
                    pr = aln.apply(
                        pr, p, hl, X_df, p_change=c, p_trans_4d=p4d, family_probas=f64,
                        base_pred=pr_base, precomputed_cands=bdy_d[m_te],
                    )

                if use_pit and tr == "v3":
                    pit = PostImpactTurnoverClassifier(PostImpactTurnoverConfig(enable_missing_player_domino=False)).fit_from_candidates(
                        pit_d, tr_ms
                    )
                    pr = pit.apply(pr, p, hl, X_df, c, p4d, fams=f64, cands=pit_d[m_te])

                f1_l.append(float(metrics.score(gt, pr, hl)["spell_f1_mean"]))

        j2_raw, j2_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            p_raw, c_raw, p4d_raw, fams_raw = st["sample"][gid]
            p = p_raw.astype(np.float64)
            c = c_raw.astype(np.float64)
            p4d = p4d_raw.astype(np.float64)
            f64 = [f.astype(np.float64) for f in fams_raw]
            pr_xdx = apply_xdx_arbitrator(
                decode_v4(p_raw, ms.half, dec_v5_hyb, p_change=c_raw),
                p_raw, ms.half, X_sample[gid], p_change=c_raw, p_trans_4d=p4d_raw, cfg=xdx_cfg_st1,
            )
            pr_base = apply_spell_reranker(pr_xdx, p_raw, ms.half, X_sample[gid], c_raw, p4d_raw, fams_raw, cfg=full_rerank_cfg)
            pr = pr_base.copy()
            if use_exp15:
                pr = arb_shielded_full.apply(pr, p, ms.half, X_sample[gid], c, p4d, fams=f64)
            if dw_c1 or dw_c2 or dw_c3:
                pr = apply_dead_wedge_custom(pr, p, ms.half, X_sample[gid], c, p4d, dw_c1, dw_c2, dw_c3, dw_vis2, dw_vis3)
            if use_exp16:
                pr = aln_shielded_full.apply(pr, p, ms.half, X_sample[gid], p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base)
            j2_raw.append(float(metrics.score(ms.labels, pr, ms.half)["spell_f1_mean"]))
            j2_nod.append(float(metrics.score(without_dead_margin(ms.labels, ms.half), pr, ms.half)["spell_f1_mean"]))

        p_raw, c_raw, p4d_raw, fams_raw = st["atd_rep"]
        p = p_raw.astype(np.float64)
        c = c_raw.astype(np.float64)
        p4d = p4d_raw.astype(np.float64)
        f64 = [f.astype(np.float64) for f in fams_raw]
        pr_xdx_a = apply_xdx_arbitrator(
            decode_v4(p_raw, half_atd_dec, dec_v5_hyb, p_change=c_raw),
            p_raw, half_atd_dec, X_atd_rep, p_change=c_raw, p_trans_4d=p4d_raw, cfg=xdx_cfg_st1,
        )
        pr_base_a = apply_spell_reranker(pr_xdx_a, p_raw, half_atd_dec, X_atd_rep, c_raw, p4d_raw, fams_raw, cfg=full_rerank_cfg)
        pr_a = pr_base_a.copy()
        if use_exp15:
            pr_a = arb_shielded_full.apply(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)
        if dw_c1 or dw_c2 or dw_c3:
            pr_a = apply_dead_wedge_custom(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, dw_c1, dw_c2, dw_c3, dw_vis2, dw_vis3)
        if use_exp16:
            pr_a = aln_shielded_full.apply(pr_a, p, half_atd_dec, X_atd_rep, p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base_a)
        if use_pit:
            pr_a = pit_shielded_full.apply(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)
        pr_a[second_start:kick2] = DEAD
        pr_a[~atd.graded] = DEAD
        sc_rep = metrics.score_public(pr_a, bootstrap_n=0)

        return {
            "j1a": float(np.mean(v3_f1)),
            "j1b_std": float(np.mean(v6s_f1)),
            "j1b_gen": float(np.mean(v6g_f1)),
            "j2_raw": float(np.mean(j2_raw)),
            "j2_nod": float(np.mean(j2_nod)),
            "j3_spell": float(sc_rep["spell_f1_mean"]),
            "j3_60": float(sc_rep["spell_f1"][0.6]["f1"]),
            "j3_70": float(sc_rep["spell_f1"][0.7]["f1"]),
            "pred_atd": pr_a,
        }

    variants = [
        ("1. v85_grand_champion (Baseline Reference)", st_v8_hyb, True, True, True, True, True, False, False, True, False),
        ("2. v85 minus DeadWedge Case1 (Y->X->D off)", st_v8_hyb, True, True, False, True, True, False, False, True, False),
        ("3. v85 minus DeadWedge Case2 (D->X->Y off)", st_v8_hyb, True, True, True, False, True, False, False, True, False),
        ("4. v85 minus DeadWedge Case3 (D->X->D off)", st_v8_hyb, True, True, True, True, False, False, False, True, False),
        ("5. v85 with DeadWedge Case2 require_vis>=0.5", st_v8_hyb, True, True, True, True, True, True, False, True, False),
        ("6. v85 with DeadWedge Case3 require_vis>=0.5", st_v8_hyb, True, True, True, True, True, False, True, True, False),
        ("7. v85 + OpenPlay-Shielded EXP-14 Inpainter", st_v8_op_inp, True, True, True, True, True, False, False, True, True),
    ]

    print("-" * 130)
    print(f"{'Variant Name':44s} | {'J1A (v3)':8s} | {'J1B-Std':8s} | {'J1B-Gen':8s} | {'J2-Raw':8s} | {'J2-noD':8s} | {'J3 spell (60/70)':18s} | {'Diff vs v85 (85m)':17s}")
    print("-" * 130, flush=True)

    unseen_mask = atd.graded & ((np.arange(len(atd.half)) < 11185) | (np.arange(len(atd.half)) > 18685))
    ref_atd = None
    out_summary = {}

    for vname, sdict, e15, e16, c1, c2, c3, rv2, rv3, upit, redec in variants:
        r = eval_variant(vname, sdict, e15, e16, c1, c2, c3, rv2, rv3, upit, redec)
        if ref_atd is None:
            ref_atd = r["pred_atd"]
        diff_85m = int(np.sum(r["pred_atd"][unseen_mask] != ref_atd[unseen_mask]))
        print(
            f"{vname:44s} | {r['j1a']:8.4f} | {r['j1b_std']:8.4f} | {r['j1b_gen']:8.4f} | "
            f"{r['j2_raw']:8.4f} | {r['j2_nod']:8.4f} | "
            f"{r['j3_spell']:6.4f} ({r['j3_60']:.3f}/{r['j3_70']:.3f}) | {diff_85m:10d} frames",
            flush=True,
        )
        out_summary[vname] = {k: v for k, v in r.items() if k != "pred_atd"}
        out_summary[vname]["diff_85m_frames_vs_v85"] = diff_85m

    print("-" * 130)
    out_json = FOR_DIR / "redteam_prueba4_synthesis_ablation.json"
    with open(out_json, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"Saved Prueba 4 synthesis ablation results to {out_json}", flush=True)


if __name__ == "__main__":
    main()
