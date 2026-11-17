#!/usr/bin/env python3
"""Red-Team Forensic Audit — Prueba 3:
Torture-Test of Robustness across:
1. 5-Minute Windows (7,500 frames, matching the BAR-ATH public evaluation window size)
   across all 10 Ground-Truth Matches (7 Sportec v3, 7 Sportec v6_std, 7 Sportec v6_gen, 3 Metrica Sample).
2. Full IoU Threshold Curve: IoU in {0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
import warnings

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "py_pkgs"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation"))

import numpy as np
import pandas as pd

import evaluation.possession_metrics as pm
from exp_v6_super_champion import compute_physical_gates
from exp_w7a_player_set_net import load_cached_physv2_table
from exp_w7b_super_champion_v7 import compute_local_newton_gate
from s3_11_tcn_local import without_dead_margin
from scripts.forensic_v7.run_unified_v8_forensic_champion import (
    apply_vis_gated_dead_wedge,
)
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
    compute_w_clean_v8,
)
from src.decode.spell_reranker import apply_spell_reranker
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
from src.eval.metrics import to_official

EXTENDED_IOUS = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def score_ious(gt: np.ndarray, pred: np.ndarray, half: np.ndarray, ious=EXTENDED_IOUS) -> dict[float, float]:
    g = to_official(gt, half, True)
    p = to_official(pred, half, True)
    p = np.where(p == pm.X, pm.D, p).astype(np.int8)
    runs = pm.valid_runs(g)
    filled = pm.fill_abstentions(p, runs)
    res = pm.spell_scores(g, filled, runs, ious=ious)["per_iou"]
    return {float(tau): float(res[tau]["f1"]) for tau in ious}


def main() -> None:
    print("=" * 120)
    print("RED-TEAM FORENSIC AUDIT — PRUEBA 3: 5-MINUTE WINDOW TORTURE TEST & EXTENDED IoU CURVE (0.30 .. 0.80)")
    print("=" * 120, flush=True)

    OUT_DIR = ROOT / "experiments" / "outputs"
    OOF_DIR = OUT_DIR / "exp_w8b_oof"
    FOR_DIR = OUT_DIR / "forensic_v7"

    sportec = load_sportec()
    matches = sorted(sportec.keys())
    sample_matches = {gid: load_sample(gid) for gid in GAMES}

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

    oof = {}
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
        oof[T] = {
            "g_pl": zp["symgbm_plus_tta"], "tr_pl": zp["symtrans_plus_tta"],
            "g_pl_on_v6": zw7b_v6["symgbm_plus_tta_on_v6"], "tr_pl_on_v6": zw7b_v6["symtrans_plus_tta_on_v6"],
            "g_v6_on_v3": zg["g_v6_on_v3"], "tr_v6_on_v3": zg["tr_v6_on_v3"],
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

    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        pe_plus = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f:
        sample_v6, _ = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, _ = pickle.load(f)
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
    with open(w6_dir / "w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)

    best_calib_cfg: TransductiveCalibConfig = w6_sum["best_calib_cfg"]
    full_rerank_cfg = w6_sum["full_rerank_cfg"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)

    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )

    with open(OOF_DIR / "sportec_fast_cache.pkl", "rb") as f:
        w7b_cache = pickle.load(f)
    with open(FOR_DIR / "v85_cands_cache.pkl", "rb") as f:
        v8_cands_bundle = pickle.load(f)

    # Reconstruct J1A, J1B-Std, J1B-Gen, J2 predictions for v7 and v8.5 Grand Champion
    preds_v7 = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}}
    preds_v85 = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}}

    for m_te in matches:
        tr_ms = [m for m in matches if m != m_te]
        hl = sportec[m_te].half
        d = oof[m_te]
        for tr, X_df, g_t in [("v3", X_v3[m_te], gates_v3[m_te]), ("v6_std", X_v6[m_te], gates_v6[m_te]), ("v6_gen", X_v6[m_te], gates_v6[m_te])]:
            preds_v7[tr][m_te] = w7b_cache[m_te]["w7b"][f"pred_{tr}_rerank"].copy()

            preds_rr, m1_d, m2_d, bdy_d, pit_d = v8_cands_bundle["hyb"][tr]
            pr_base = preds_rr[m_te].copy()
            if tr == "v3":
                a_v = compute_hybrid_gate_v8(X_df, hl, g_t[0])
                wc_v = compute_w_clean_v8(X_df, hl, g_t[0])
                p, c = synthesize_head_decoupled_proba(
                    d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"],
                    d["tcn"], d["gru"], d["tr_pl"], d["tr_p2_v3"], d["tcnc"], d["gruc"],
                    hl, d["g_v6_on_v3"], d["tr_v6_on_v3"],
                    w_ghost=g_t[0], w_gap=g_t[1], cfg=scfg_main,
                    tcn_physv2_proba=d["tcn_p2_v3"], bigru_physv2_proba=d["gru_p2_v3"],
                    tcn_physv2_change=d["tcnc_p2_v3"], bigru_physv2_change=d["gruc_p2_v3"],
                    alpha_p2=a_v,
                )
                p, c = blend_6fam_heads_v8(
                    p, c, d["tcn"], d["gru"], d["gru_p2_v3"], d["pset_v3"],
                    d["tcnc"], d["gruc"], d["gruc_p2_v3"], d["psetc_v3"],
                    a_v, wc_v, 0.05, 0.30, 0.18, 0.08, adaptive_regime=True,
                )
                p = calibrate_match_transductive(p, hl, X_df, best_calib_cfg)
                p4d = d["p4d_v3"]
                fams = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"], d["pset_v3"]]
            elif tr == "v6_std":
                a_v = compute_hybrid_gate_v8(X_df, hl, g_t[0])
                wc_v = compute_w_clean_v8(X_df, hl, g_t[0])
                p, c = synthesize_head_decoupled_proba(
                    d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                    d["tcn"], d["gru"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc"], d["gruc"],
                    hl, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                    w_ghost=g_t[0], w_gap=g_t[1], cfg=scfg_main,
                    tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                    tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                    alpha_p2=a_v,
                )
                p, c = blend_6fam_heads_v8(
                    p, c, d["tcn"], d["gru"], d["gru_p2_v6"], d["pset_v6"],
                    d["tcnc"], d["gruc"], d["gruc_p2_v6"], d["psetc_v6"],
                    a_v, wc_v, 0.05, 0.30, 0.18, 0.08, adaptive_regime=True,
                )
                p = calibrate_match_transductive(p, hl, X_df, best_calib_cfg)
                p4d = d["p4d_v6"]
                fams = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tcn_p2_v6"], d["gru_p2_v6"], d["pset_v6"]]
            else:
                wc_v = compute_w_clean_v8(X_df, hl, g_t[0])
                p, c = synthesize_head_decoupled_proba(
                    d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                    d["tcn_p2_v6"], d["gru_p2_v6"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc_p2_v6"], d["gruc_p2_v6"],
                    hl, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                    w_ghost=g_t[0], w_gap=g_t[1], cfg=scfg_main,
                    tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                    tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                    alpha_p2=np.ones(len(hl), dtype=np.float32),
                )
                p, c = blend_6fam_heads_v8(
                    p, c, d["tcn_p2_v6"], d["gru_p2_v6"], d["gru_p2_v6"], d["pset_v6"],
                    d["tcnc_p2_v6"], d["gruc_p2_v6"], d["gruc_p2_v6"], d["psetc_v6"],
                    np.ones(len(hl), dtype=np.float32), wc_v, 0.05, 0.30, 0.18, 0.08, adaptive_regime=False,
                )
                p = calibrate_match_transductive(p, hl, X_df, best_calib_cfg)
                p4d = d["p4d_v6"]
                fams = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn_p2_v6"], d["gru_p2_v6"], d["g_v6_on_v6"], d["pset_v6"]]

            p64 = p.astype(np.float64)
            c64 = c.astype(np.float64)
            p4d64 = p4d.astype(np.float64)
            f64 = [f.astype(np.float64) for f in fams]

            pr = pr_base.copy()
            arb = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, tr_ms)
            pr = arb.apply(pr, p64, hl, X_df, c64, p4d64, fams=f64, m1_cands=m1_d[m_te], m2_cands=m2_d[m_te])
            if tr != "v6_gen":
                pr = apply_vis_gated_dead_wedge(pr, p64, hl, X_df, c64, p4d64, vis_thresh=0.85)
            aln = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=False).fit({m: bdy_d[m] for m in tr_ms})
            pr = aln.apply(pr, p64, hl, X_df, p_change=c64, p_trans_4d=p4d64, family_probas=f64, base_pred=pr_base, precomputed_cands=bdy_d[m_te])
            preds_v85[tr][m_te] = pr

    # Fit full 7-match models for Judge 2 (Metrica Sample)
    _, m1_v3, m2_v3, bdy_v3, _ = v8_cands_bundle["hyb"]["v3"]
    arb_full = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_v3, m2_v3, matches)
    aln_full = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_v3)

    for gid in GAMES:
        ms = sample_matches[gid]
        dh = sample_h[gid]
        dpl = pe_plus["sample"][gid]
        dv6 = sample_v6[gid]
        dp2 = pe_p2["sample"][gid]
        dw5 = pe_w5["sample"][gid]
        dw5b = pe_w5b["sample"][gid]
        dpset = pset_j23["sample"][gid]
        p4d_s = b4d_j23["sample"][gid]
        fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"]]

        # v7 on sample
        a_s7 = compute_local_newton_gate(X_sample[gid], ms.half, gates_sample[gid][0])
        wc_s = compute_w_clean_v8(X_sample[gid], ms.half, gates_sample[gid][0])
        ps7, cs7 = synthesize_head_decoupled_proba(
            dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
            dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
            ms.half, dv6["g_v6"], dv6["tr_v6"],
            w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main,
            tcn_physv2_proba=dw5b["tcn_physv2_sym"], bigru_physv2_proba=dw5b["bigru_physv2_sym"],
            tcn_physv2_change=dw5b["tcnc_physv2_sym"], bigru_physv2_change=dw5b["bigruc_physv2_sym"],
            alpha_p2=a_s7,
        )
        ps7, cs7 = blend_6fam_heads_v8(
            ps7, cs7, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"],
            dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"],
            a_s7, wc_s, 0.05, 0.30, 0.18, 0.08, adaptive_regime=False,
        )
        ps7 = calibrate_match_transductive(ps7, ms.half, X_sample[gid], best_calib_cfg)
        pr7_xdx = apply_xdx_arbitrator(
            decode_v4(ps7, ms.half, dec_v5_hyb, p_change=cs7),
            ps7, ms.half, X_sample[gid], p_change=cs7, p_trans_4d=p4d_s, cfg=xdx_cfg_st1,
        )
        preds_v7["sample"][gid] = apply_spell_reranker(pr7_xdx, ps7, ms.half, X_sample[gid], cs7, p4d_s, fs, cfg=full_rerank_cfg)

        # v85 on sample
        a_s8 = compute_hybrid_gate_v8(X_sample[gid], ms.half, gates_sample[gid][0])
        ps8, cs8 = synthesize_head_decoupled_proba(
            dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
            dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
            ms.half, dv6["g_v6"], dv6["tr_v6"],
            w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main,
            tcn_physv2_proba=dw5b["tcn_physv2_sym"], bigru_physv2_proba=dw5b["bigru_physv2_sym"],
            tcn_physv2_change=dw5b["tcnc_physv2_sym"], bigru_physv2_change=dw5b["bigruc_physv2_sym"],
            alpha_p2=a_s8,
        )
        ps8, cs8 = blend_6fam_heads_v8(
            ps8, cs8, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"],
            dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"],
            a_s8, wc_s, 0.05, 0.30, 0.18, 0.08, adaptive_regime=True,
        )
        ps8 = calibrate_match_transductive(ps8, ms.half, X_sample[gid], best_calib_cfg)
        pr8_xdx = apply_xdx_arbitrator(
            decode_v4(ps8, ms.half, dec_v5_hyb, p_change=cs8),
            ps8, ms.half, X_sample[gid], p_change=cs8, p_trans_4d=p4d_s, cfg=xdx_cfg_st1,
        )
        pr8_base = apply_spell_reranker(pr8_xdx, ps8, ms.half, X_sample[gid], cs8, p4d_s, fs, cfg=full_rerank_cfg)
        f64_s = [f.astype(np.float64) for f in fs]
        pr8 = arb_full.apply(pr8_base.copy(), ps8.astype(np.float64), ms.half, X_sample[gid], cs8.astype(np.float64), p4d_s.astype(np.float64), fams=f64_s)
        pr8 = apply_vis_gated_dead_wedge(pr8, ps8.astype(np.float64), ms.half, X_sample[gid], cs8.astype(np.float64), p4d_s.astype(np.float64), vis_thresh=0.85)
        pr8 = aln_full.apply(pr8, ps8.astype(np.float64), ms.half, X_sample[gid], p_change=cs8.astype(np.float64), p_trans_4d=p4d_s.astype(np.float64), family_probas=f64_s, base_pred=pr8_base)
        preds_v85["sample"][gid] = pr8

    # ------------------------------------------------------------------
    # 1. Extended IoU Curve {0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}
    # ------------------------------------------------------------------
    print("\n[PRUEBA 3.A] Extended IoU Curve (IoU in {0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}):")
    header = "Benchmark         | Model  | " + " | ".join(f"IoU@{tau:.2f}" for tau in EXTENDED_IOUS) + " | Mean(0.3..0.7) | Mean(0.5..0.8)"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    iou_summary = {}
    for b_name, key, keys_list, gt_fn, hl_fn in [
        ("J1A (Sportec v3)", "v3", matches, lambda k: sportec[k].labels, lambda k: sportec[k].half),
        ("J1B-Std (v6)    ", "v6_std", matches, lambda k: sportec[k].labels, lambda k: sportec[k].half),
        ("J1B-Gen (v6)    ", "v6_gen", matches, lambda k: sportec[k].labels, lambda k: sportec[k].half),
        ("J2-Raw (Metrica)", "sample", GAMES, lambda k: sample_matches[k].labels, lambda k: sample_matches[k].half),
        ("J2-noD (Metrica)", "sample", GAMES, lambda k: without_dead_margin(sample_matches[k].labels, sample_matches[k].half), lambda k: sample_matches[k].half),
    ]:
        c_v7 = {tau: [] for tau in EXTENDED_IOUS}
        c_v85 = {tau: [] for tau in EXTENDED_IOUS}
        for k in keys_list:
            s7 = score_ious(gt_fn(k), preds_v7[key][k], hl_fn(k))
            s8 = score_ious(gt_fn(k), preds_v85[key][k], hl_fn(k))
            for tau in EXTENDED_IOUS:
                c_v7[tau].append(s7[tau])
                c_v85[tau].append(s8[tau])
        m7 = {tau: float(np.mean(c_v7[tau])) for tau in EXTENDED_IOUS}
        m8 = {tau: float(np.mean(c_v85[tau])) for tau in EXTENDED_IOUS}
        d8 = {tau: m8[tau] - m7[tau] for tau in EXTENDED_IOUS}
        mean7_off = float(np.mean([m7[t] for t in (0.30, 0.40, 0.50, 0.60, 0.70)]))
        mean8_off = float(np.mean([m8[t] for t in (0.30, 0.40, 0.50, 0.60, 0.70)]))
        mean7_hi = float(np.mean([m7[t] for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)]))
        mean8_hi = float(np.mean([m8[t] for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)]))
        iou_summary[b_name.strip()] = {"v7": m7, "v85": m8, "delta": d8, "mean_off_v7": mean7_off, "mean_off_v85": mean8_off, "mean_hi_v7": mean7_hi, "mean_hi_v85": mean8_hi}

        str7 = " | ".join(f"{m7[t]:8.4f}" for t in EXTENDED_IOUS)
        str8 = " | ".join(f"{m8[t]:8.4f}" for t in EXTENDED_IOUS)
        strd = " | ".join(f"{d8[t]:+8.4f}" for t in EXTENDED_IOUS)
        print(f"{b_name:17s} | v7     | {str7} | {mean7_off:14.4f} | {mean7_hi:14.4f}")
        print(f"{b_name:17s} | v8.5   | {str8} | {mean8_off:14.4f} | {mean8_hi:14.4f}")
        print(f"{'':17s} | Delta  | {strd} | {mean8_off - mean7_off:+14.4f} | {mean8_hi - mean7_hi:+14.4f}")
        print("-" * len(header))

    # ------------------------------------------------------------------
    # 2. 5-Minute Rolling Windows (7,500 frames) Torture Test
    # ------------------------------------------------------------------
    print("\n[PRUEBA 3.B] 5-Minute Window (7,500 frames) Torture Test across all 10 Full GT Matches:")
    WIN_LEN = 7500
    STRIDE = 3750  # 50% overlap (2.5-min stride) + also report non-overlapping (7,500 stride)

    win_results = {}
    for stride_tag, stride_val in [("Non-Overlapping 5-Min Windows (stride=7,500f)", 7500), ("Sliding 5-Min Windows (stride=3,750f)", 3750)]:
        print(f"\n  >>> {stride_tag}:")
        win_results[stride_tag] = {}
        for b_name, key, keys_list, gt_fn, hl_fn in [
            ("10 Matches Clean/OOD (7 v3 + 3 Metrica)", "clean_10", [("v3", m) for m in matches] + [("sample", g) for g in GAMES], None, None),
            ("7 Sportec v3 (J1A)", "v3", [("v3", m) for m in matches], None, None),
            ("7 Sportec v6-Std (J1B-Std)", "v6_std", [("v6_std", m) for m in matches], None, None),
            ("7 Sportec v6-Gen (J1B-Gen)", "v6_gen", [("v6_gen", m) for m in matches], None, None),
            ("3 Metrica Sample Raw (J2)", "sample", [("sample", g) for g in GAMES], None, None),
        ]:
            deltas_off = []
            deltas_hi = []
            wins, ties, losses = 0, 0, 0
            wins_hi, ties_hi, losses_hi = 0, 0, 0
            for stream_k, match_id in keys_list:
                if stream_k == "sample":
                    gt = sample_matches[match_id].labels
                    hl = sample_matches[match_id].half
                else:
                    gt = sportec[match_id].labels
                    hl = sportec[match_id].half
                p7 = preds_v7[stream_k][match_id]
                p8 = preds_v85[stream_k][match_id]

                for h_val in (1, 2):
                    idx_h = np.flatnonzero(hl == h_val)
                    if len(idx_h) < WIN_LEN:
                        continue
                    s_h, e_h = int(idx_h[0]), int(idx_h[-1]) + 1
                    for w_s in range(s_h, e_h - WIN_LEN + 1, stride_val):
                        w_e = w_s + WIN_LEN
                        gt_w = gt[w_s:w_e]
                        hl_w = hl[w_s:w_e]
                        sc7 = score_ious(gt_w, p7[w_s:w_e], hl_w)
                        sc8 = score_ious(gt_w, p8[w_s:w_e], hl_w)
                        f1_7 = float(np.mean([sc7[t] for t in (0.30, 0.40, 0.50, 0.60, 0.70)]))
                        f1_8 = float(np.mean([sc8[t] for t in (0.30, 0.40, 0.50, 0.60, 0.70)]))
                        hi_7 = float(np.mean([sc7[t] for t in (0.60, 0.70)]))
                        hi_8 = float(np.mean([sc8[t] for t in (0.60, 0.70)]))
                        d_off = f1_8 - f1_7
                        d_hi = hi_8 - hi_7
                        deltas_off.append(d_off)
                        deltas_hi.append(d_hi)
                        if d_off > 1e-6:
                            wins += 1
                        elif d_off < -1e-6:
                            losses += 1
                        else:
                            ties += 1
                        if d_hi > 1e-6:
                            wins_hi += 1
                        elif d_hi < -1e-6:
                            losses_hi += 1
                        else:
                            ties_hi += 1

            n_w = len(deltas_off)
            win_or_tie_pct = 100.0 * (wins + ties) / max(n_w, 1)
            win_among_changed = 100.0 * wins / max(wins + losses, 1)
            win_or_tie_hi_pct = 100.0 * (wins_hi + ties_hi) / max(n_w, 1)
            win_among_changed_hi = 100.0 * wins_hi / max(wins_hi + losses_hi, 1)
            rec_w = {
                "n_windows": n_w,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_or_tie_pct": round(win_or_tie_pct, 2),
                "win_rate_when_different_pct": round(win_among_changed, 2),
                "mean_delta_spellF1": round(float(np.mean(deltas_off)), 5),
                "mean_delta_F1_06_07": round(float(np.mean(deltas_hi)), 5),
                "wins_06_07": wins_hi,
                "ties_06_07": ties_hi,
                "losses_06_07": losses_hi,
                "win_or_tie_06_07_pct": round(win_or_tie_hi_pct, 2),
                "win_rate_when_different_06_07_pct": round(win_among_changed_hi, 2),
            }
            win_results[stride_tag][b_name] = rec_w
            print(
                f"    {b_name:40s} | Windows={n_w:3d} | "
                f"spellF1: Win={wins:2d}, Tie={ties:3d}, Loss={losses:2d} (Win/Tie={win_or_tie_pct:5.1f}%, Win|Diff={win_among_changed:5.1f}%, MeanDelta={np.mean(deltas_off):+.5f}) | "
                f"F1@[0.6,0.7]: Win={wins_hi:2d}, Tie={ties_hi:3d}, Loss={losses_hi:2d} (Win/Tie={win_or_tie_hi_pct:5.1f}%, Win|Diff={win_among_changed_hi:5.1f}%, MeanDelta={np.mean(deltas_hi):+.5f})"
            )

    out_path = FOR_DIR / "redteam_torture_test_windows_iou.json"
    with open(out_path, "w") as f:
        json.dump({"extended_iou_curve": iou_summary, "rolling_5min_windows": win_results}, f, indent=2)
    print(f"\nSaved complete Prueba 3 Torture-Test results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
