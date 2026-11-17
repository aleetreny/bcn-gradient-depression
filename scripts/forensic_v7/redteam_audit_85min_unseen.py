#!/usr/bin/env python3
"""Red-Team Forensic Audit — Prueba 1 & Prueba 2:
1. Code & Gate Audit (`compute_inpaint_gate_v8`, `DeadWedge`, `PIT/Domino`, `Shielded 8B`).
2. Complete Episode-by-Episode Physical Inspection in the 85 Unseen Minutes of BAR-ATH
   (frames 0..11184 and 18686..199318; 151,357 active unseen frames).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any
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

from exp_v6_super_champion import compute_physical_gates
from exp_w7a_player_set_net import load_cached_physv2_table
from exp_w7b_super_champion_v7 import compute_local_newton_gate
from games import GAMES as GAME_INFO, anchor_video_frame
from scripts.forensic_v7.prototype_v7_forensic_upgrades import (
    apply_dead_wedge_arbitrator,
    apply_short_turnover_recovery,
)
from src.decode.touch_type_arbitrator import rle_runs_half
from scripts.forensic_v7.run_unified_v8_forensic_champion import (
    apply_vis_gated_dead_wedge,
)
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
from src.decode.spell_reranker import apply_spell_reranker
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    compute_seq481_gate,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator

LBL_MAP = {0: "H", 1: "A", 2: "D", -1: "U"}


def get_contiguous_diff_regions(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    diff = (a != b).astype(np.int8)
    if not np.any(diff):
        return []
    padded = np.concatenate([[0], diff, [0]])
    starts = np.flatnonzero((padded[1:] == 1) & (padded[:-1] == 0))
    ends = np.flatnonzero((padded[1:] == 0) & (padded[:-1] == 1))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def describe_spell_context(pred: np.ndarray, start: int, end: int, half: np.ndarray) -> str:
    n = len(pred)
    p_before = "START"
    if start > 0 and half[start - 1] == half[start]:
        b_lbl = int(pred[start - 1])
        b_s = start - 1
        while b_s > 0 and pred[b_s - 1] == b_lbl and half[b_s - 1] == half[start]:
            b_s -= 1
        p_before = f"{LBL_MAP[b_lbl]}[{b_s}:{start}]({start - b_s}f)"
    p_after = "END"
    if end < n and half[end] == half[end - 1]:
        a_lbl = int(pred[end])
        a_e = end
        while a_e < n and pred[a_e] == a_lbl and half[a_e] == half[end - 1]:
            a_e += 1
        p_after = f"{LBL_MAP[a_lbl]}[{end}:{a_e}]({a_e - end}f)"
    mid_spells = rle_runs_half(pred[start:end], half[start:end])
    mid_str = "+".join(f"{LBL_MAP[int(lbl)]}({int(e - s)}f)" for s, e, lbl, *_ in mid_spells)
    return f"{p_before} -> [{mid_str}] -> {p_after}"


def main() -> None:
    print("=" * 115)
    print("RED-TEAM FORENSIC AUDIT — PRUEBA 1 (CODE & GATE AUDIT) & PRUEBA 2 (85 UNSEEN MINUTES EPISODE INSPECTION)")
    print("=" * 115, flush=True)

    OUT_DIR = ROOT / "experiments" / "outputs"
    OOF_DIR = OUT_DIR / "exp_w8b_oof"
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

    # ------------------------------------------------------------------
    # PART 1: Audit `compute_inpaint_gate_v8` (Why EXP-14 Inpainter drops J1A & J1B-Std)
    # ------------------------------------------------------------------
    print("\n[PRUEBA 1.A] Auditing `compute_inpaint_gate_v8` across Sportec v3, v6, Metrica, and BAR-ATH:")
    for tag, X_dict, gt_dict, h_dict, g_dict in [
        ("Sportec v3", X_v3, {m: sportec[m].labels for m in matches}, {m: sportec[m].half for m in matches}, gates_v3),
        ("Sportec v6", X_v6, {m: sportec[m].labels for m in matches}, {m: sportec[m].half for m in matches}, gates_v6),
        ("Metrica Sample", X_sample, {g: sample_matches[g].labels for g in GAMES}, {g: sample_matches[g].half for g in GAMES}, gates_sample),
    ]:
        scales = []
        tot_active = 0
        tot_dead_gt = 0
        tot_out_pitch = 0
        tot_open_play_blackout = 0
        for k, X_df in X_dict.items():
            since = np.nan_to_num(X_df["ball_since_seen"].to_numpy(np.float64), nan=0.0)
            max_since = float(np.nanmax(since))
            scale = 25.0 if max_since > 250.0 else 1.0
            scales.append((k, round(max_since, 1), scale))
            w_inp = compute_inpaint_gate_v8(X_df, h_dict[k], g_dict[k][0], g_dict[k][1])
            act = w_inp > 0.05
            gt = gt_dict[k]
            out_p = np.nan_to_num(X_df["ball_outside_pitch"].to_numpy(np.float64), nan=0.0) > 0.5
            tot_active += int(np.sum(act))
            tot_dead_gt += int(np.sum(act & (gt == 2)))
            tot_out_pitch += int(np.sum(act & out_p))
            tot_open_play_blackout += int(np.sum(act & (gt != 2) & (~out_p)))
        pct_dead = 100.0 * tot_dead_gt / max(tot_active, 1)
        pct_open = 100.0 * tot_open_play_blackout / max(tot_active, 1)
        print(f"  {tag:16s} | max(ball_since_seen) & scale: {scales}")
        print(
            f"                   | w_inp>0.05 frames: {tot_active:6d} | In GT Dead (D): {tot_dead_gt:6d} ({pct_dead:5.1f}%) | "
            f"Outside Pitch: {tot_out_pitch:6d} | True Open-Play Blackout (H/A in-pitch): {tot_open_play_blackout:5d} ({pct_open:4.1f}%)"
        )

    w_inp_atd = compute_inpaint_gate_v8(X_atd_rep, atd.half, gate_atd_rep[0], gate_atd_rep[1])
    since_atd = np.nan_to_num(X_atd_rep["ball_since_seen"].to_numpy(np.float64), nan=0.0)
    out_atd = np.nan_to_num(X_atd_rep["ball_outside_pitch"].to_numpy(np.float64), nan=0.0) > 0.5
    act_atd = (w_inp_atd > 0.05) & atd.graded
    print(
        f"  {'BAR-ATH Rep':16s} | max(ball_since_seen)={np.nanmax(since_atd):.1f} | "
        f"w_inp>0.05 active frames={int(np.sum(act_atd))} (Outside pitch={int(np.sum(act_atd & out_atd))})"
    )

    # ------------------------------------------------------------------
    # PART 2: Audit PostImpactTurnoverClassifier & Double-Domino Gate
    # ------------------------------------------------------------------
    with open(FOR_DIR / "v85_cands_cache.pkl", "rb") as f:
        v8_cands_bundle = pickle.load(f)

    _, m1_d, m2_d, bdy_d, pit_d = v8_cands_bundle["hyb"]["v3"]
    arb_v85 = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, matches)
    aln_v85 = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_d)
    pit_v85 = PostImpactTurnoverClassifier().fit_from_candidates(pit_d, matches)
    print(f"\n[PRUEBA 1.B] PostImpactTurnoverClassifier LOMO inner-CV status on 7 v3 matches:")
    print(f"  active_lomo={pit_v85.cfg.active_lomo}, best_obj={pit_v85.cfg.best_obj:.6f}, th_lomo={pit_v85.cfg.th_lomo:.4f}")

    for tag, X_dict in [("Sportec v3", X_v3), ("Sportec v6", X_v6), ("Metrica Sample", X_sample)]:
        max_d1h = max(float(np.nanmax(df["dist_ball_H"].to_numpy(np.float64))) for df in X_dict.values())
        max_d1a = max(float(np.nanmax(df["dist_ball_A"].to_numpy(np.float64))) for df in X_dict.values())
        n_over_50 = sum(
            int(
                np.sum(
                    (np.nan_to_num(df["dist_ball_H"].to_numpy(np.float64), nan=100.0) > 50.0)
                    | (np.nan_to_num(df["dist_ball_A"].to_numpy(np.float64), nan=100.0) > 50.0)
                )
            )
            for df in X_dict.values()
        )
        print(f"  {tag:16s} | max(dist_ball_H)={max_d1h:.1f}m, max(dist_ball_A)={max_d1a:.1f}m, frames with dist>50m/NaN={n_over_50}")

    d1h_atd = np.nan_to_num(X_atd_rep["dist_ball_H"].to_numpy(np.float64), nan=100.0)
    d1a_atd = np.nan_to_num(X_atd_rep["dist_ball_A"].to_numpy(np.float64), nan=100.0)
    pub_mask = np.zeros(len(X_atd_rep), dtype=bool)
    pub_mask[11185:18686] = True
    unseen_mask = atd.graded & (~pub_mask)
    print(
        f"  {'BAR-ATH Rep':16s} | Public (7,501f) dist>50m/NaN: H={int(np.sum(d1h_atd[pub_mask] > 50))}, A={int(np.sum(d1a_atd[pub_mask] > 50))} | "
        f"Unseen (151,357f) dist>50m/NaN: H={int(np.sum(d1h_atd[unseen_mask] > 50))}, A={int(np.sum(d1a_atd[unseen_mask] > 50))}"
    )

    # ------------------------------------------------------------------
    # PART 3: Reconstruct BAR-ATH streams & inspect Stage-by-Stage predictions
    # ------------------------------------------------------------------
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

    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)
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
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f:
        pset_j23 = pickle.load(f)
    with open(w6_dir / "w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)
    with open(W8A_DIR / "w8a_j23_preds.pkl", "rb") as f:
        j23_inp = pickle.load(f)
    j23_inp["atd_rep"]["w_inp"] = w_inp_atd

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

    def get_atd_stream(gate_mode: str, adaptive_regime: bool, inp_scale: float):
        vkey = "atd_rep"
        dpl_a = pe_plus[vkey]
        dp2_a = pe_p2[vkey]
        dw5_a = pe_w5[vkey]
        dw5b_a = pe_w5b[vkey]
        dpset_a = pset_j23[vkey]
        dinp_a = j23_inp[vkey]
        t2_p, g2_p, t2_c, g2_c = atd_rep_h["t_sym"], atd_rep_h["b_sym"], atd_rep_h["tc_sym"], atd_rep_h["bc_sym"]
        gv6_p, gv6_c = atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"]
        if gate_mode == "hybrid_v8":
            a_a = compute_hybrid_gate_v8(X_atd_rep, atd.half, gate_atd_rep[0])
        elif gate_mode == "local_newton":
            a_a = compute_local_newton_gate(X_atd_rep, atd.half, gate_atd_rep[0])
        else:
            a_a = compute_seq481_gate(X_atd_rep, atd.half, gate_atd_rep[0])
        wc_a = compute_w_clean_v8(X_atd_rep, atd.half, gate_atd_rep[0])
        pa, ca = synthesize_head_decoupled_proba(
            dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"],
            t2_p, g2_p, dpl_a["tr_plus_sym"], dp2_a["tr_physv2_sym"], t2_c, g2_c,
            atd.half, gv6_p, gv6_c,
            w_ghost=gate_atd_rep[0], w_gap=gate_atd_rep[1], cfg=scfg_main,
            tcn_physv2_proba=dw5b_a["tcn_physv2_sym"], bigru_physv2_proba=dw5b_a["bigru_physv2_sym"],
            tcn_physv2_change=dw5b_a["tcnc_physv2_sym"], bigru_physv2_change=dw5b_a["bigruc_physv2_sym"],
            alpha_p2=a_a,
        )
        pa, ca = blend_6fam_heads_v8(
            pa, ca, t2_p, g2_p, dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"],
            t2_c, g2_c, dw5b_a["bigruc_physv2_sym"], dpset_a["pset_change"],
            a_a, wc_a, 0.05, 0.30, 0.18, 0.08, adaptive_regime=adaptive_regime,
            p_inp=dinp_a["p_inp"] if inp_scale > 0 else None,
            c_inp=dinp_a["c_inp"] if inp_scale > 0 else None,
            w_inp=(inp_scale * dinp_a["w_inp"]) if inp_scale > 0 else None,
        )
        pa = calibrate_match_transductive(pa, half_atd_dec, X_atd_rep, best_calib_cfg)
        fa = [
            dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"],
            t2_p, g2_p, dw5b_a["tcn_physv2_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"]
        ]
        return pa, ca, b4d_j23[vkey], fa

    p_v7, c_v7, p4d_v7, fams_v7 = get_atd_stream("local_newton", False, 0.0)
    p_v8, c_v8, p4d_v8, fams_v8 = get_atd_stream("hybrid_v8", True, 0.0)
    p_v9, c_v9, p4d_v9, fams_v9 = get_atd_stream("hybrid_v8", True, 0.35)

    def finalize_atd(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[second_start:kick2] = DEAD
        out[~atd.graded] = DEAD
        return out

    # Compute exact intermediate stages on BAR-ATH
    pr_v7_xdx = apply_xdx_arbitrator(
        decode_v4(p_v7, half_atd_dec, dec_v5_hyb, p_change=c_v7),
        p_v7, half_atd_dec, X_atd_rep, p_change=c_v7, p_trans_4d=p4d_v7, cfg=xdx_cfg_st1,
    )
    stage0_v7 = finalize_atd(
        apply_spell_reranker(pr_v7_xdx, p_v7, half_atd_dec, X_atd_rep, c_v7, p4d_v7, fams_v7, cfg=full_rerank_cfg)
    )

    pr_v8_xdx = apply_xdx_arbitrator(
        decode_v4(p_v8, half_atd_dec, dec_v5_hyb, p_change=c_v8),
        p_v8, half_atd_dec, X_atd_rep, p_change=c_v8, p_trans_4d=p4d_v8, cfg=xdx_cfg_st1,
    )
    pr_v8_rr = apply_spell_reranker(pr_v8_xdx, p_v8, half_atd_dec, X_atd_rep, c_v8, p4d_v8, fams_v8, cfg=full_rerank_cfg)
    stage1_w8a = finalize_atd(pr_v8_rr)

    f64_v8 = [f.astype(np.float64) for f in fams_v8]
    pr_s2 = arb_v85.apply(pr_v8_rr.copy(), p_v8.astype(np.float64), half_atd_dec, X_atd_rep, c_v8.astype(np.float64), p4d_v8.astype(np.float64), fams=f64_v8)
    stage2_exp15 = finalize_atd(pr_s2)

    pr_s3 = apply_vis_gated_dead_wedge(pr_s2.copy(), p_v8.astype(np.float64), half_atd_dec, X_atd_rep, c_v8.astype(np.float64), p4d_v8.astype(np.float64), vis_thresh=0.85)
    stage3_dw = finalize_atd(pr_s3)

    pr_s4 = aln_v85.apply(pr_s3.copy(), p_v8.astype(np.float64), half_atd_dec, X_atd_rep, p_change=c_v8.astype(np.float64), p_trans_4d=p4d_v8.astype(np.float64), family_probas=f64_v8, base_pred=pr_v8_rr)
    stage4_exp16_strict_cv = finalize_atd(pr_s4)

    pr_s5 = pit_v85.apply(pr_s4.copy(), p_v8.astype(np.float64), half_atd_dec, X_atd_rep, c_v8.astype(np.float64), p4d_v8.astype(np.float64), fams=f64_v8)
    stage5_grand_champion = finalize_atd(pr_s5)

    # Also build Stage 6 (v9_all6)
    _, m1_d9, m2_d9, bdy_d9, pit_d9 = v8_cands_bundle["inp"]["v3"]
    arb_v9 = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d9, m2_d9, matches)
    aln_v9 = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_d9)
    pit_v9 = PostImpactTurnoverClassifier().fit_from_candidates(pit_d9, matches)
    f64_v9 = [f.astype(np.float64) for f in fams_v9]
    pr_v9_xdx = apply_xdx_arbitrator(
        decode_v4(p_v9, half_atd_dec, dec_v5_hyb, p_change=c_v9),
        p_v9, half_atd_dec, X_atd_rep, p_change=c_v9, p_trans_4d=p4d_v9, cfg=xdx_cfg_st1,
    )
    pr_v9_rr = apply_spell_reranker(pr_v9_xdx, p_v9, half_atd_dec, X_atd_rep, c_v9, p4d_v9, fams_v9, cfg=full_rerank_cfg)
    pr_v9_cur = arb_v9.apply(pr_v9_rr.copy(), p_v9.astype(np.float64), half_atd_dec, X_atd_rep, c_v9.astype(np.float64), p4d_v9.astype(np.float64), fams=f64_v9)
    pr_v9_cur = apply_vis_gated_dead_wedge(pr_v9_cur, p_v9.astype(np.float64), half_atd_dec, X_atd_rep, c_v9.astype(np.float64), p4d_v9.astype(np.float64), vis_thresh=0.85)
    pr_v9_cur = aln_v9.apply(pr_v9_cur, p_v9.astype(np.float64), half_atd_dec, X_atd_rep, p_change=c_v9.astype(np.float64), p_trans_4d=p4d_v9.astype(np.float64), family_probas=f64_v9, base_pred=pr_v9_rr)
    pr_v9_cur = pit_v9.apply(pr_v9_cur, p_v9.astype(np.float64), half_atd_dec, X_atd_rep, c_v9.astype(np.float64), p4d_v9.astype(np.float64), fams=f64_v9)
    stage6_v9_all6 = finalize_atd(pr_v9_cur)

    # Extract tracking arrays for physical telemetry
    vis = np.nan_to_num(X_atd_rep["ball_visible"].to_numpy(np.float64), nan=1.0)
    out_p = np.nan_to_num(X_atd_rep["ball_outside_pitch"].to_numpy(np.float64), nan=0.0)
    d1_h = np.nan_to_num(X_atd_rep["dist_ball_H"].to_numpy(np.float64), nan=100.0)
    d1_a = np.nan_to_num(X_atd_rep["dist_ball_A"].to_numpy(np.float64), nan=100.0)
    bs = np.nan_to_num(X_atd_rep["ball_speed"].to_numpy(np.float64), nan=0.0)
    imp = np.nan_to_num(X_atd_rep["impulse_mag_d4"].to_numpy(np.float64), nan=0.0)
    fams_stack = np.stack([np.argmax(f, axis=1) for f in fams_v8], axis=0)  # (9, N)

    def print_diff_telemetry(name_from: str, name_to: str, arr_from: np.ndarray, arr_to: np.ndarray) -> list[dict[str, Any]]:
        regions = get_contiguous_diff_regions(arr_from, arr_to)
        pub_regions = [r for r in regions if not (r[1] <= 11185 or r[0] >= 18686)]
        unseen_regions = [r for r in regions if (r[1] <= 11185 or r[0] >= 18686)]
        tot_f = int(np.sum(arr_from != arr_to))
        unseen_f = int(np.sum((arr_from != arr_to) & unseen_mask))
        pub_f = int(np.sum((arr_from != arr_to) & pub_mask))
        print("\n" + "-" * 115)
        print(
            f"[STAGE DIFF] {name_from}  --->  {name_to}\n"
            f"  Total changed frames: {tot_f} ({len(regions)} episodes) | "
            f"Public 5-min (7,501f): {pub_f}f ({len(pub_regions)} eps) | "
            f"Private 85-min (151,357f): {unseen_f}f ({len(unseen_regions)} eps)"
        )
        print("-" * 115)
        records = []
        for idx, (s, e) in enumerate(regions, 1):
            is_pub = not (e <= 11185 or s >= 18686)
            zone = "PUBLIC_5MIN" if is_pub else "UNSEEN_85MIN"
            dur = e - s
            hl_val = int(atd.half[s])
            ctx_from = describe_spell_context(arr_from, s, e, half_atd_dec)
            ctx_to = describe_spell_context(arr_to, s, e, half_atd_dec)
            m_vis = float(np.mean(vis[s:e]))
            m_out = float(np.mean(out_p[s:e]))
            mh = float(np.mean(d1_h[s:e]))
            min_h = float(np.min(d1_h[s:e]))
            ma = float(np.mean(d1_a[s:e]))
            min_a = float(np.min(d1_a[s:e]))
            m_bs = float(np.mean(bs[s:e]))
            max_bs = float(np.max(bs[s:e]))
            max_imp_in = float(np.max(imp[s:e]))
            imp_s = float(np.max(imp[max(0, s - 2) : min(len(imp), s + 3)]))
            imp_e = float(np.max(imp[max(0, e - 2) : min(len(imp), e + 3)]))
            mp = np.mean(p_v8[s:e], axis=0)
            votes_h = float(np.mean(fams_stack[:, s:e] == 0))
            votes_a = float(np.mean(fams_stack[:, s:e] == 1))
            votes_d = float(np.mean(fams_stack[:, s:e] == 2))
            rec = {
                "ep_idx": idx,
                "zone": zone,
                "half": hl_val,
                "start": s,
                "end": e,
                "dur_frames": dur,
                "dur_sec": round(dur / 25.0, 2),
                "ctx_from": ctx_from,
                "ctx_to": ctx_to,
                "mean_vis": round(m_vis, 3),
                "mean_out": round(m_out, 3),
                "mean_d1_h": round(mh, 2),
                "min_d1_h": round(min_h, 2),
                "mean_d1_a": round(ma, 2),
                "min_d1_a": round(min_a, 2),
                "mean_bs": round(m_bs, 2),
                "max_bs": round(max_bs, 2),
                "max_imp_in": round(max_imp_in, 2),
                "imp_at_start": round(imp_s, 2),
                "imp_at_end": round(imp_e, 2),
                "mean_p_H": round(float(mp[0]), 3),
                "mean_p_A": round(float(mp[1]), 3),
                "mean_p_D": round(float(mp[2]), 3),
                "fam_votes_HAD": (round(votes_h, 2), round(votes_a, 2), round(votes_d, 2)),
            }
            records.append(rec)
            print(
                f"  Ep#{idx:02d} [{zone:12s}] H{hl_val} [{s:6d}:{e:6d}] ({dur:3d}f = {dur/25.0:4.2f}s)\n"
                f"    BEFORE: {ctx_from}\n"
                f"    AFTER : {ctx_to}\n"
                f"    PHYSICS: vis={m_vis:.2f}, out_pitch={m_out:.2f} | d_H={mh:.2f}m(min {min_h:.2f}m), d_A={ma:.2f}m(min {min_a:.2f}m) | "
                f"speed={m_bs:.1f}m/s(max {max_bs:.1f}) | imp_in={max_imp_in:.1f}, imp@start={imp_s:.1f}, imp@end={imp_e:.1f}\n"
                f"    PROBAS : p(H,A,D)=({mp[0]:.3f}, {mp[1]:.3f}, {mp[2]:.3f}) | 9-Stream Votes(H,A,D)=({votes_h:.2f}, {votes_a:.2f}, {votes_d:.2f})"
            )
        return records

    all_stage_diffs = {}
    all_stage_diffs["Stage0_v7_to_Stage1_Wave8A"] = print_diff_telemetry(
        "Stage 0 (v7 Baseline)", "Stage 1 (Wave 8A: EXP-13 Hybrid Gate)", stage0_v7, stage1_w8a
    )
    all_stage_diffs["Stage1_Wave8A_to_Stage2_ShieldedEXP15"] = print_diff_telemetry(
        "Stage 1 (Wave 8A)", "Stage 2 (+ Shielded EXP-15 TouchArbitrator)", stage1_w8a, stage2_exp15
    )
    all_stage_diffs["Stage2_EXP15_to_Stage3_DeadWedge"] = print_diff_telemetry(
        "Stage 2 (+ Shielded EXP-15)", "Stage 3 (+ DeadWedge Arbitrator)", stage2_exp15, stage3_dw
    )
    all_stage_diffs["Stage3_DeadWedge_to_Stage4_ShieldedEXP16"] = print_diff_telemetry(
        "Stage 3 (+ DeadWedge)", "Stage 4 (+ Shielded EXP-16 BoundaryAligner = v85_strict_cv)", stage3_dw, stage4_exp16_strict_cv
    )
    all_stage_diffs["Stage4_StrictCV_to_Stage5_PIT_GrandChamp"] = print_diff_telemetry(
        "Stage 4 (v85_strict_cv)", "Stage 5 (+ PIT/Domino = v85_grand_champion)", stage4_exp16_strict_cv, stage5_grand_champion
    )
    all_stage_diffs["Stage5_GrandChamp_to_Stage6_v9_All6"] = print_diff_telemetry(
        "Stage 5 (v85_grand_champion)", "Stage 6 (+ EXP-14 Inpainter = v9_all6)", stage5_grand_champion, stage6_v9_all6
    )
    all_stage_diffs["Total_v7_to_v85_GrandChamp"] = print_diff_telemetry(
        "Stage 0 (v7 Baseline)", "Stage 5 (v85_forensic_grand_champion)", stage0_v7, stage5_grand_champion
    )

    out_path = FOR_DIR / "redteam_85min_episode_audit.json"
    with open(out_path, "w") as f:
        json.dump(all_stage_diffs, f, indent=2)
    print(f"\nSaved complete episode-by-episode 85-minute telemetry to {out_path}", flush=True)


if __name__ == "__main__":
    main()
