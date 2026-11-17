#!/usr/bin/env python3
"""Wave 8B & 8C Master Runner:
- `EXP-15`: Controlled Touch vs Deflection/Interception Arbitrator (`SymTouchTypeArbitrator`)
- `EXP-16`: Supervised Sub-Second Boundary Offset Aligner (`SymBoundaryOffsetAligner`)
- `Super-Champion v8` Grand Synthesis combining:
  * `EXP-13`: `compute_hybrid_gate_v8` + `compute_w_clean_v8` + `blend_6fam_heads_v8(adaptive_regime=True)`
  * `EXP-14`: `SymBallInpainterNet` (`SymBallPositionInpainter` + `SymGapDistilledExpert`)
  * `EXP-15`: `SymTouchTypeArbitrator` (Mode 1 Deflection Absorption + Mode 2 Clearance Rescue)
  * `EXP-16`: `SymBoundaryOffsetAligner` (Sub-Second Boundary Alignment for IoU 0.6 & 0.7)

Evaluates across all 4 Judges:
  - Judge 1A (`v3` Clean 7-fold Nested LOMO)
  - Judge 1B-Std (`v6_std` Standard-Noisy 7-fold Nested LOMO)
  - Judge 1B-Gen (`v6_gen` Genuine-Noisy 7-fold Nested LOMO)
  - Judge 2 (`Metrica Sample` 3 games OOD, Raw & without dead margin)
  - Judge 3 (`BAR-ATH` Raw & Repaired OOD)
And generates + officially validates (`validate_submission(table, 199319)`) all `submission_v8_*.csv` files.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, HOME
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.boundary_offset_aligner import (
    BoundaryOffsetAlignerConfig,
    SymBoundaryOffsetAligner,
    extract_boundary_candidates,
    verify_boundary_aligner_equivariance,
)
from src.decode.hybrid_gate_v8 import (
    blend_6fam_heads_v8,
    compute_hybrid_gate_v8,
    compute_inpaint_gate_v8,
    compute_w_clean_v8,
)
from src.decode.mbr_snap import (
    ImpulseSnapConfig,
    decode_active_4dir,
    mbr_decode_sequences,
    snap_boundaries_to_impulse,
)
from src.decode.spell_reranker import (
    SpellRerankerConfig,
    apply_spell_reranker,
)
from src.decode.touch_type_arbitrator import (
    SymTouchTypeArbitrator,
    TouchTypeArbitratorConfig,
    extract_mode1_candidates,
    extract_mode2_candidates,
    preclean_tracking_arrays,
    verify_touch_arbitrator_equivariance,
)
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    compute_seq481_gate,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
from src.eval import metrics
from exp_v6_super_champion import compute_physical_gates, moe_combine, save_and_validate_submission
from exp_w7a_player_set_net import load_cached_physv2_table
from exp_w7b_super_champion_v7 import compute_local_newton_gate

OUT_DIR = ROOT / "experiments" / "outputs"
OOF_DIR = OUT_DIR / "exp_w8b_oof"
W8A_DIR = OUT_DIR / "exp_w8a_oof"
LOG_PATH = OUT_DIR / "exp_w8b_touch_and_boundary_align.log"
SUB_DIR = ROOT / "submissions"


class TeeLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg: str):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def get_cached_track_arrays(cache_m: dict, track: str):
    """Extract base prediction, probabilities, boundary signals, and kinematic dict from Sportec cache."""
    w7b = cache_m["w7b"]
    pred_base = w7b[f"pred_{track}_rerank"]
    P = w7b[f"proba_{track}"]
    p_ch = w7b[f"pchange_{track}"]
    p4d = cache_m["p4d_v3"] if track == "v3" else cache_m["p4d_v6"]
    xv = cache_m["xv3"] if track == "v3" else cache_m["xv6"]
    fams = [f.astype(np.float64) for f in cache_m[f"fams_{track}"]]
    return pred_base, P, p_ch, p4d, xv, fams


def eval_sportec_oof(matches: list[str], cache: dict, preds_by_m: dict[str, np.ndarray]) -> dict:
    """Compute official 7-match mean metrics including spell_f1[0.6], spell_f1[0.7], spell_f1_mean, macro_f1."""
    rows = []
    for m in matches:
        ev = metrics.score(cache[m]["gt"], preds_by_m[m], cache[m]["hl"])
        ev["spell_f1_0.5"] = float(ev["spell_f1"][0.5]["f1"])
        ev["spell_f1_0.6"] = float(ev["spell_f1"][0.6]["f1"])
        ev["spell_f1_0.7"] = float(ev["spell_f1"][0.7]["f1"])
        rows.append(ev)
    keys = [
        "spell_f1_mean",
        "spell_f1_0.5",
        "spell_f1_0.6",
        "spell_f1_0.7",
        "macro_f1",
        "accuracy",
        "team_acc_gt_alive",
        "alive_dead_f1",
    ]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys if k in rows[0]}
    out["per_match"] = {m: float(r["spell_f1_mean"]) for m, r in zip(matches, rows)}
    return out


def extract_cands_for_stream(
    matches: list[str],
    st_track: dict,
    sportec: dict,
    X_dict: dict[str, pd.DataFrame],
    loo_dec_hyb: dict,
    xdx_cfg: XDXConfig,
    lomo_rerank_cfgs: dict,
) -> tuple[dict, dict, dict, dict]:
    """Compute base Rerank predictions and extract Mode 1, Mode 2, and Boundary Offset candidates for a stream."""
    preds_rr = {}
    m1_dict = {}
    m2_dict = {}
    bdy_dict = {}
    for m in matches:
        p, c, p4d, fams = st_track[m]
        hl = sportec[m].half
        gt = sportec[m].labels
        X_df = X_dict[m]
        p_xdx = apply_xdx_arbitrator(
            decode_v4(p, hl, loo_dec_hyb[m], p_change=c),
            p, hl, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg,
        )
        p_rr = apply_spell_reranker(p_xdx, p, hl, X_df, c, p4d, fams, cfg=lomo_rerank_cfgs[m])
        preds_rr[m] = p_rr
        cxv = preclean_tracking_arrays(X_df, len(p_rr))
        f64 = [f.astype(np.float64) for f in fams]
        m1_dict[m] = extract_mode1_candidates(
            p_rr, p.astype(np.float64), hl, cxv, c.astype(np.float64), p4d.astype(np.float64), f64, gt=gt, match_id=m
        )
        m2_dict[m] = extract_mode2_candidates(
            p_rr, p.astype(np.float64), hl, cxv, c.astype(np.float64), p4d.astype(np.float64), f64, gt=gt, match_id=m
        )
        bdy_dict[m] = extract_boundary_candidates(
            p_rr, p.astype(np.float64), hl, X_df,
            p_change=c.astype(np.float64), p_trans_4d=p4d.astype(np.float64),
            family_probas=f64, labels=gt,
        )
    return preds_rr, m1_dict, m2_dict, bdy_dict


def main():
    t0 = time.time()
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = TeeLogger(LOG_PATH)

    print("=" * 105)
    print("WAVE 8B & 8C: TOUCH-TYPE ARBITRATOR (EXP-15) + BOUNDARY ALIGNER (EXP-16) + SUPER-CHAMPION V8 SYNTHESIS")
    print("=" * 105, flush=True)

    # ------------------------------------------------------------------
    # 1. Load Sportec 7-fold cache & pre-extracted candidate structures
    # ------------------------------------------------------------------
    with open(OOF_DIR / "sportec_fast_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    matches = sorted(cache.keys())

    with open(OOF_DIR / "mode1_cands_clean.pkl", "rb") as f:
        m1_all = pickle.load(f)
    with open(OOF_DIR / "mode2_cands_clean.pkl", "rb") as f:
        m2_all = pickle.load(f)
    with open(OOF_DIR / "bdy_cands_clean.pkl", "rb") as f:
        bdy_clean = pickle.load(f)

    # ------------------------------------------------------------------
    # 2. Verify Exact 0.000e+00 H <-> A Equivariance on Sportec Match 0
    # ------------------------------------------------------------------
    print("\n[STEP 1] Verifying Exact 0.000e+00 H <-> A Equivariance of EXP-15 and EXP-16...", flush=True)
    m0 = matches[0]
    arb_eq = SymTouchTypeArbitrator().fit_from_candidates(m1_all["v3"], m2_all["v3"], matches)
    aln_eq = SymBoundaryOffsetAligner().fit(bdy_clean["v3"])
    pred_b0, P_b0, pch_b0, p4d_b0, xv_b0, fams_b0 = get_cached_track_arrays(cache[m0], "v3")

    max_fdiff_15, mismatch_15 = verify_touch_arbitrator_equivariance(
        pred_b0, P_b0, cache[m0]["hl"], xv_b0, pch_b0, p4d_b0, fams_b0, arb_eq
    )
    eq_16 = verify_boundary_aligner_equivariance(
        pred_b0, P_b0, cache[m0]["hl"], xv_b0, pch_b0, p4d_b0, fams_b0, aln_eq
    )
    print(
        f"  [EQUIVARIANCE EXP-15] SymTouchTypeArbitrator : max_feat_diff={max_fdiff_15:.3e}, mismatch_frames={mismatch_15}",
        flush=True,
    )
    print(
        f"  [EQUIVARIANCE EXP-16] SymBoundaryOffsetAligner: max_feat_diff={eq_16['max_feat_diff']:.3e}, mismatch_frames={eq_16['mismatch_frames']}",
        flush=True,
    )
    assert max_fdiff_15 == 0.0 and mismatch_15 == 0, "EXP-15 equivariance check failed!"
    assert eq_16["max_feat_diff"] == 0.0 and eq_16["mismatch_frames"] == 0, "EXP-16 equivariance check failed!"

    # ------------------------------------------------------------------
    # 3. 7-Fold Nested LOMO Evaluation of EXP-15 & EXP-16 on w7b Base
    # ------------------------------------------------------------------
    print("\n[STEP 2] 7-Fold Nested LOMO on Sportec w7b Base (Judge 1A, 1B-Std, 1B-Gen)...", flush=True)
    track_labels = {
        "v3": "Judge 1A (Clean v3)",
        "v6_std": "Judge 1B-Std (Standard-Noisy v6_std)",
        "v6_gen": "Judge 1B-Gen (Genuine-Noisy v6_gen)",
    }

    lomo_summary: dict[str, dict[str, dict]] = {}
    full_arbs_w7b: dict[str, SymTouchTypeArbitrator] = {}
    full_alns_w7b: dict[str, SymBoundaryOffsetAligner] = {}

    for tr, judge_label in track_labels.items():
        preds_base = {}
        preds_mbr_base = {}
        preds_15 = {}
        preds_16 = {}
        preds_1516 = {}
        preds_mbr_w8b = {}

        for m_te in matches:
            tr_ms = [m for m in matches if m != m_te]
            arb = SymTouchTypeArbitrator().fit_from_candidates(m1_all[tr], m2_all[tr], tr_ms)
            aln = SymBoundaryOffsetAligner().fit({m: bdy_clean[tr][m] for m in tr_ms})

            pred_b, P_b, pch_b, p4d_b, xv_b, fams_b = get_cached_track_arrays(cache[m_te], tr)
            hl_b = cache[m_te]["hl"]
            p_mbr_b = cache[m_te]["w7b"][f"pred_{tr}_mbr"]

            p15 = arb.apply(
                pred_b, P_b, hl_b, xv_b, pch_b, p4d_b, fams=fams_b,
                m1_cands=m1_all[tr][m_te], m2_cands=m2_all[tr][m_te],
            )
            p16 = aln.apply(
                pred_b, P_b, hl_b, xv_b, pch_b, p4d_b, family_probas=fams_b,
                precomputed_cands=bdy_clean[tr][m_te],
            )
            p1516 = aln.apply(
                p15, P_b, hl_b, xv_b, pch_b, p4d_b, family_probas=fams_b,
                base_pred=pred_b, precomputed_cands=bdy_clean[tr][m_te],
            )
            p_mbr_w8b = mbr_decode_sequences(
                [p1516, p15, p16, pred_b, p_mbr_b],
                hl_b,
                weights=[0.42, 0.22, 0.14, 0.12, 0.10],
            )

            preds_base[m_te] = pred_b
            preds_mbr_base[m_te] = p_mbr_b
            preds_15[m_te] = p15
            preds_16[m_te] = p16
            preds_1516[m_te] = p1516
            preds_mbr_w8b[m_te] = p_mbr_w8b

        full_arbs_w7b[tr] = SymTouchTypeArbitrator().fit_from_candidates(m1_all[tr], m2_all[tr], matches)
        full_alns_w7b[tr] = SymBoundaryOffsetAligner().fit(bdy_clean[tr])

        lomo_summary[tr] = {
            "w7b_rerank": eval_sportec_oof(matches, cache, preds_base),
            "w7b_mbr": eval_sportec_oof(matches, cache, preds_mbr_base),
            "w8b_exp15": eval_sportec_oof(matches, cache, preds_15),
            "w8b_exp16": eval_sportec_oof(matches, cache, preds_16),
            "w8b_exp15_16": eval_sportec_oof(matches, cache, preds_1516),
            "w8b_mbr": eval_sportec_oof(matches, cache, preds_mbr_w8b),
        }

        npz_dict = {}
        for m in matches:
            npz_dict[f"{m}_gt"] = cache[m]["gt"]
            npz_dict[f"{m}_half"] = cache[m]["hl"]
            npz_dict[f"{m}_proba"] = cache[m]["w7b"][f"proba_{tr}"]
            npz_dict[f"{m}_pchange"] = cache[m]["w7b"][f"pchange_{tr}"]
            npz_dict[f"{m}_pred_w7b_rerank"] = preds_base[m]
            npz_dict[f"{m}_pred_w7b_mbr"] = preds_mbr_base[m]
            npz_dict[f"{m}_pred_exp15"] = preds_15[m]
            npz_dict[f"{m}_pred_exp16"] = preds_16[m]
            npz_dict[f"{m}_pred_exp15_16"] = preds_1516[m]
            npz_dict[f"{m}_pred_w8b_mbr"] = preds_mbr_w8b[m]
        np.savez_compressed(OOF_DIR / f"w8b_{tr}.npz", **npz_dict)

        print(f"\n--- {judge_label} ---", flush=True)
        b_f1 = lomo_summary[tr]["w7b_rerank"]["spell_f1_mean"]
        for k_name, label in [
            ("w7b_rerank", "1. w7b Base (LN+XDX+Rerank)"),
            ("w7b_mbr", "2. w7b Grand MBR"),
            ("w8b_exp15", "3. w7b + EXP-15 (TouchArbitrator)"),
            ("w8b_exp16", "4. w7b + EXP-16 (BoundaryAligner)"),
            ("w8b_exp15_16", "5. w7b + EXP-15 + EXP-16 (Stacked)"),
            ("w8b_mbr", "6. w8b Grand MBR (EXP-15+16 Pool)"),
        ]:
            ev = lomo_summary[tr][k_name]
            df1 = ev["spell_f1_mean"] - b_f1
            print(
                f"  {label:<36s}: spell_f1_mean={ev['spell_f1_mean']:.6f} ({df1:+.6f}) | "
                f"F1@0.6={ev['spell_f1_0.6']:.4f} | F1@0.7={ev['spell_f1_0.7']:.4f} | "
                f"macro_f1={ev['macro_f1']:.4f} | team_acc={ev['team_acc_gt_alive']:.4f} | D_F1={ev['alive_dead_f1']:.4f}",
                flush=True,
            )
        print("  Per-match spell_f1_mean (w7b Base -> +EXP-15 -> +EXP-16 -> +EXP-15+16):", flush=True)
        for m in matches:
            v0 = lomo_summary[tr]["w7b_rerank"]["per_match"][m]
            v15 = lomo_summary[tr]["w8b_exp15"]["per_match"][m]
            v16 = lomo_summary[tr]["w8b_exp16"]["per_match"][m]
            v1516 = lomo_summary[tr]["w8b_exp15_16"]["per_match"][m]
            print(
                f"    {m}: {v0:.4f} -> +15:{v15:.4f} ({v15-v0:+.4f}) | +16:{v16:.4f} ({v16-v0:+.4f}) | +15+16:{v1516:.4f} ({v1516-v0:+.4f})",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 4. GRAND SYNTHESIS `SUPER-CHAMPION V8` (EXP-13 + EXP-14 + EXP-15 + EXP-16)
    # ------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("[STEP 3] GRAND SYNTHESIS: SUPER-CHAMPION V8 (EXP-13 + EXP-14 + EXP-15 + EXP-16) ACROSS ALL 4 JUDGES")
    print("=" * 105, flush=True)

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
    inp_oof = {}
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
        # Recompute strict (1 - vis) interior gap gate for EXP-14 per Red-Team Auditor
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

    # Recompute strict (1 - vis) interior gap gate on J2 & J3
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

    def build_v8_streams(
        s3_d: float = 0.05,
        s3_ha: float = 0.30,
        s3_c: float = 0.18,
        ps_ha: float = 0.08,
        gate_mode: str = "hybrid_v8",
        adaptive_regime: bool = True,
        inp_scale: float = 0.0,
    ) -> dict:
        streams = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}, "atd_raw": None, "atd_rep": None}

        def _get_alpha(X_df, h_arr, wg_arr):
            if gate_mode == "hybrid_v8":
                return compute_hybrid_gate_v8(X_df, h_arr, wg_arr)
            if gate_mode == "local_newton":
                return compute_local_newton_gate(X_df, h_arr, wg_arr)
            return compute_seq481_gate(X_df, h_arr, wg_arr)

        for T in MATCH_IDS:
            d = oof[T]
            di = inp_oof[T]
            h = sportec[T].half

            a_v3 = _get_alpha(X_v3[T], h, gates_v3[T][0])
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
                a_v3, wc_v3, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=di["p_inp_v3"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v3"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v3"]) if inp_scale > 0 else None,
            )
            p3 = calibrate_match_transductive(p3, h, X_v3[T], best_calib_cfg)
            f3 = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"], d["pset_v3"]]
            streams["v3"][T] = (p3, c3, d["p4d_v3"], f3)

            a_v6 = _get_alpha(X_v6[T], h, gates_v6[T][0])
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
                a_v6, wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=di["p_inp_v6"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v6"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v6"]) if inp_scale > 0 else None,
            )
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
                p_inp=di["p_inp_v6"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v6"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v6"]) if inp_scale > 0 else None,
            )
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
            a_s = _get_alpha(X_sample[gid], ms.half, gates_sample[gid][0])
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
                a_s, wc_s, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=dinp["p_inp"] if inp_scale > 0 else None,
                c_inp=dinp["c_inp"] if inp_scale > 0 else None,
                w_inp=(inp_scale * dinp["w_inp"]) if inp_scale > 0 else None,
            )
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
            a_a = _get_alpha(X_df, atd.half, gate_t[0])
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
                a_a, wc_a, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=dinp_a["p_inp"] if inp_scale > 0 else None,
                c_inp=dinp_a["c_inp"] if inp_scale > 0 else None,
                w_inp=(inp_scale * dinp_a["w_inp"]) if inp_scale > 0 else None,
            )
            pa = calibrate_match_transductive(pa, half_atd_dec, X_df, best_calib_cfg)
            fa = [dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"], t2_p, g2_p, dw5b_a["tcn_physv2_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"]]
            streams[vkey] = (pa, ca, b4d_j23[vkey], fa)
        return streams

    st_v7_bal = build_v8_streams(0.00, 0.25, 0.18, 0.10, gate_mode="seq481_global", adaptive_regime=False, inp_scale=0.0)
    st_v7_newt = build_v8_streams(0.05, 0.30, 0.18, 0.08, gate_mode="local_newton", adaptive_regime=False, inp_scale=0.0)
    st_v8_hyb = build_v8_streams(0.05, 0.30, 0.18, 0.08, gate_mode="hybrid_v8", adaptive_regime=True, inp_scale=0.0)
    st_v8_inp = build_v8_streams(0.05, 0.30, 0.18, 0.08, gate_mode="hybrid_v8", adaptive_regime=True, inp_scale=0.35)

    # Extract nested-LOMO candidates on `st_v8_hyb` and `st_v8_inp` so EXP-15 & EXP-16 fit natively on v8 streams!
    print("\n  Extracting nested-LOMO candidates on `v8_hybrid` & `v8_inp` streams...", flush=True)
    v8_cands = {}
    v8_arbs_full = {}
    v8_alns_full = {}
    for tr, X_dict in [("v3", X_v3), ("v6_std", X_v6), ("v6_gen", X_v6)]:
        pr_rr, m1_d, m2_d, bdy_d = extract_cands_for_stream(
            matches, st_v8_hyb[tr], sportec, X_dict, loo_dec_hyb, xdx_cfg_st1, lomo_rerank_cfgs
        )
        v8_cands[tr] = (pr_rr, m1_d, m2_d, bdy_d)
        v8_arbs_full[tr] = SymTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, matches)
        v8_alns_full[tr] = SymBoundaryOffsetAligner().fit(bdy_d)

    def eval_v8_pipeline(label: str, st: dict, mode: str = "rerank") -> tuple[dict, np.ndarray, np.ndarray]:
        v3_f1, v6s_f1, v6g_f1 = [], [], []
        v3_f1_06, v3_f1_07 = [], []
        v6s_f1_06, v6s_f1_07 = [], []
        v6g_f1_06, v6g_f1_07 = [], []

        for T in MATCH_IDS:
            h = sportec[T].half
            y = sportec[T].labels
            tr_ms = [m for m in matches if m != T]
            for key_s, X_df, f1_list, f06_list, f07_list in [
                ("v3", X_v3[T], v3_f1, v3_f1_06, v3_f1_07),
                ("v6_std", X_v6[T], v6s_f1, v6s_f1_06, v6s_f1_07),
                ("v6_gen", X_v6[T], v6g_f1, v6g_f1_06, v6g_f1_07),
            ]:
                p, c, p4d, fams = st[key_s][T]
                f64 = [f.astype(np.float64) for f in fams]
                pr_xdx = apply_xdx_arbitrator(
                    decode_v4(p, h, loo_dec_hyb[T], p_change=c),
                    p, h, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1,
                )
                if mode == "xdx":
                    pr = pr_xdx
                else:
                    pr_rr = apply_spell_reranker(pr_xdx, p, h, X_df, c, p4d, fams, cfg=lomo_rerank_cfgs[T])
                    if mode == "rerank":
                        pr = pr_rr
                    else:
                        _, m1_d, m2_d, bdy_d = v8_cands[key_s]
                        arb = SymTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, tr_ms)
                        aln = SymBoundaryOffsetAligner().fit({m: bdy_d[m] for m in tr_ms})
                        p15 = arb.apply(pr_rr, p.astype(np.float64), h, X_df, c.astype(np.float64), p4d.astype(np.float64), fams=f64)
                        p16 = aln.apply(pr_rr, p.astype(np.float64), h, X_df, c.astype(np.float64), p4d.astype(np.float64), family_probas=f64)
                        p1516 = aln.apply(p15, p.astype(np.float64), h, X_df, c.astype(np.float64), p4d.astype(np.float64), family_probas=f64, base_pred=pr_rr)
                        if mode == "exp15":
                            pr = p15
                        elif mode == "exp16":
                            pr = p16
                        elif mode == "exp15_16":
                            pr = p1516
                        elif mode == "mbr_v8":
                            pr = mbr_decode_sequences([p1516, p15, p16, pr_rr, pr_xdx], h, weights=[0.42, 0.22, 0.14, 0.12, 0.10])
                        else:
                            pr = p1516
                sc_m = metrics.score(y, pr, h)
                f1_list.append(float(sc_m["spell_f1_mean"]))
                f06_list.append(float(sc_m["spell_f1"][0.6]["f1"]))
                f07_list.append(float(sc_m["spell_f1"][0.7]["f1"]))

        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            p, c, p4d, fams = st["sample"][gid]
            f64 = [f.astype(np.float64) for f in fams]
            pr_xdx = apply_xdx_arbitrator(
                decode_v4(p, ms.half, dec_v5_hyb, p_change=c),
                p, ms.half, X_sample[gid], p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1,
            )
            if mode == "xdx":
                pr = pr_xdx
            else:
                pr_rr = apply_spell_reranker(pr_xdx, p, ms.half, X_sample[gid], c, p4d, fams, cfg=full_rerank_cfg)
                if mode == "rerank":
                    pr = pr_rr
                else:
                    arb = v8_arbs_full["v3"]
                    aln = v8_alns_full["v3"]
                    p15 = arb.apply(pr_rr, p.astype(np.float64), ms.half, X_sample[gid], c.astype(np.float64), p4d.astype(np.float64), fams=f64)
                    p16 = aln.apply(pr_rr, p.astype(np.float64), ms.half, X_sample[gid], c.astype(np.float64), p4d.astype(np.float64), family_probas=f64)
                    p1516 = aln.apply(p15, p.astype(np.float64), ms.half, X_sample[gid], c.astype(np.float64), p4d.astype(np.float64), family_probas=f64, base_pred=pr_rr)
                    if mode == "exp15":
                        pr = p15
                    elif mode == "exp16":
                        pr = p16
                    elif mode == "exp15_16":
                        pr = p1516
                    elif mode == "mbr_v8":
                        pr = mbr_decode_sequences([p1516, p15, p16, pr_rr, pr_xdx], ms.half, weights=[0.42, 0.22, 0.14, 0.12, 0.10])
                    else:
                        pr = p1516
            s_raw.append(float(metrics.score(ms.labels, pr, ms.half)["spell_f1_mean"]))
            s_nod.append(float(metrics.score(without_dead_margin(ms.labels, ms.half), pr, ms.half)["spell_f1_mean"]))

        def _dec_a(vkey, X_df, tr_key="v3"):
            p, c, p4d, fams = st[vkey]
            f64 = [f.astype(np.float64) for f in fams]
            pr_xdx = apply_xdx_arbitrator(
                decode_v4(p, half_atd_dec, dec_v5_hyb, p_change=c),
                p, half_atd_dec, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1,
            )
            if mode == "xdx":
                pr = pr_xdx
            else:
                pr_rr = apply_spell_reranker(pr_xdx, p, half_atd_dec, X_df, c, p4d, fams, cfg=full_rerank_cfg)
                if mode == "rerank":
                    pr = pr_rr
                else:
                    arb = v8_arbs_full[tr_key]
                    aln = v8_alns_full[tr_key]
                    p15 = arb.apply(pr_rr, p.astype(np.float64), half_atd_dec, X_df, c.astype(np.float64), p4d.astype(np.float64), fams=f64)
                    p16 = aln.apply(pr_rr, p.astype(np.float64), half_atd_dec, X_df, c.astype(np.float64), p4d.astype(np.float64), family_probas=f64)
                    p1516 = aln.apply(p15, p.astype(np.float64), half_atd_dec, X_df, c.astype(np.float64), p4d.astype(np.float64), family_probas=f64, base_pred=pr_rr)
                    if mode == "exp15":
                        pr = p15
                    elif mode == "exp16":
                        pr = p16
                    elif mode == "exp15_16":
                        pr = p1516
                    elif mode == "mbr_v8":
                        pr = mbr_decode_sequences([p1516, p15, p16, pr_rr, pr_xdx], half_atd_dec, weights=[0.42, 0.22, 0.14, 0.12, 0.10])
                    else:
                        pr = p1516
            pr[second_start:kick2] = DEAD
            pr[~atd.graded] = DEAD
            return pr

        pr_raw = _dec_a("atd_raw", X_atd_raw, tr_key="v3")
        pr_rep = _dec_a("atd_rep", X_atd_rep, tr_key="v3")
        sc_raw = metrics.score_public(pr_raw, bootstrap_n=0)
        sc_rep = metrics.score_public(pr_rep, bootstrap_n=0)
        pl_rep = plausibility(pr_rep, atd.half, atd.graded)
        res = {
            "label": label,
            "j1a": float(np.mean(v3_f1)),
            "j1a_06": float(np.mean(v3_f1_06)),
            "j1a_07": float(np.mean(v3_f1_07)),
            "j1b_std": float(np.mean(v6s_f1)),
            "j1b_std_06": float(np.mean(v6s_f1_06)),
            "j1b_std_07": float(np.mean(v6s_f1_07)),
            "j1b_gen": float(np.mean(v6g_f1)),
            "j1b_gen_06": float(np.mean(v6g_f1_06)),
            "j1b_gen_07": float(np.mean(v6g_f1_07)),
            "j2_raw": float(np.mean(s_raw)),
            "j2_nod": float(np.mean(s_nod)),
            "j3_raw": float(sc_raw["spell_f1_mean"]),
            "j3_rep": float(sc_rep["spell_f1_mean"]),
            "j3_rep_06": float(sc_rep["spell_f1"][0.6]["f1"]),
            "j3_rep_07": float(sc_rep["spell_f1"][0.7]["f1"]),
            "j3_df1": float(sc_rep["alive_dead_f1"]),
            "j3_pm": float(pl_rep["poss_per_min"]),
            "j3_pctd": float(pl_rep["pct_D"]),
        }
        print(
            f"  {label:64s} | J1A={res['j1a']:.4f} (0.6:{res['j1a_06']:.4f}/0.7:{res['j1a_07']:.4f}) | "
            f"J1B={res['j1b_std']:.4f}/{res['j1b_gen']:.4f} | "
            f"J2={res['j2_raw']:.4f} ({res['j2_nod']:.4f}) | J3 Raw={res['j3_raw']:.4f} | "
            f"J3 Rep={res['j3_rep']:.4f} (D={res['j3_df1']:.3f}, p/m={res['j3_pm']:.2f})",
            flush=True,
        )
        return res, pr_raw, pr_rep

    print("\n--- Multi-Judge Ablation Table: Wave 7B vs Wave 8 (`EXP-13` + `EXP-14` + `EXP-15` + `EXP-16`) ---", flush=True)
    r_v7_bal_xdx, _, _ = eval_v8_pipeline("1. v7_bal (global gate, 0.00/0.25/0.10) + XDX", st_v7_bal, "xdx")
    r_v7_newt_xdx, _, _ = eval_v8_pipeline("2. v7_newt (local_newton gate, 0.05/0.30/0.08) + XDX", st_v7_newt, "xdx")
    r_v7_bal_rr, _, _ = eval_v8_pipeline("3. v7_bal (global gate, 0.00/0.25/0.10) + Rerank", st_v7_bal, "rerank")
    r_v8_hyb_xdx, _, sub_hyb_xdx = eval_v8_pipeline("4. v8_hybrid (EXP-13 Hybrid Gate + Adaptive Regime) + XDX", st_v8_hyb, "xdx")
    r_v8_hyb_rr, _, sub_hyb_rr = eval_v8_pipeline("5. v8_hybrid (EXP-13 Hybrid Gate + Adaptive Regime) + Rerank", st_v8_hyb, "rerank")
    r_v8_inp_rr, _, sub_inp_rr = eval_v8_pipeline("6. v8_hybrid + EXP-14 Inpainter (scale=0.35) + Rerank", st_v8_inp, "rerank")
    r_v8_hyb_15, _, sub_hyb_15 = eval_v8_pipeline("7. v8_hybrid + EXP-15 (SymTouchTypeArbitrator)", st_v8_hyb, "exp15")
    r_v8_hyb_16, _, sub_hyb_16 = eval_v8_pipeline("8. v8_hybrid + EXP-16 (SymBoundaryOffsetAligner)", st_v8_hyb, "exp16")
    r_v8_hyb_1516, raw_1516, sub_hyb_1516 = eval_v8_pipeline("9. v8_hybrid + EXP-15 + EXP-16 (Super-Champion v8)", st_v8_hyb, "exp15_16")
    r_v8_inp_1516, _, sub_inp_1516 = eval_v8_pipeline("10. v8_hybrid + EXP-14 + EXP-15 + EXP-16 (Full v8 All-4)", st_v8_inp, "exp15_16")
    r_v8_mbr, _, sub_v8_mbr = eval_v8_pipeline("11. Super-Champion v8 Grand MBR (EXP-13+14+15+16 Pool)", st_v8_hyb, "mbr_v8")

    # ------------------------------------------------------------------
    # 5. Generate and Officially Validate Final `submission_v8_*.csv` Portfolio
    # ------------------------------------------------------------------
    print("\n[STEP 4] Generating and Officially Validating `submission_v8_*.csv` Portfolio (199,319 rows)...", flush=True)
    submissions_to_save = [
        ("v8_super_champion", sub_hyb_1516),
        ("v8_super_champion_inp_all4", sub_inp_1516),
        ("v8_super_champion_mbr", sub_v8_mbr),
        ("v8_hybrid_touch_arbitrator", sub_hyb_15),
        ("v8_hybrid_boundary_aligner", sub_hyb_16),
        ("v8_hybrid_inp_rerank", sub_inp_rr),
        ("v8_hybrid_gate_xdx", sub_hyb_xdx),
        ("v8_hybrid_gate_rerank", sub_hyb_rr),
    ]
    for sub_key, pred_arr in submissions_to_save:
        fname = f"submission_{sub_key}.csv"
        out_path = SUB_DIR / fname
        save_and_validate_submission(out_path, pred_arr, atd, second_start, kick2)
        sc = metrics.score_public(pred_arr, bootstrap_n=0)
        pl = plausibility(pred_arr, atd.half, atd.graded)
        print(
            f"  [VALIDATED] {fname:44s} | rows=199,319 | J3 spell_f1_mean={sc['spell_f1_mean']:.4f} | "
            f"F1@0.6={sc['spell_f1'][0.6]['f1']:.4f} | F1@0.7={sc['spell_f1'][0.7]['f1']:.4f} | "
            f"D_F1={sc['alive_dead_f1']:.3f} | poss/min={pl['poss_per_min']:.2f}",
            flush=True,
        )

    summary_payload = {
        "equivariance": {
            "exp15_max_feat_diff": max_fdiff_15,
            "exp15_mismatch_frames": mismatch_15,
            "exp16_max_feat_diff": eq_16["max_feat_diff"],
            "exp16_mismatch_frames": eq_16["mismatch_frames"],
        },
        "w8b_lomo_on_w7b": lomo_summary,
        "v8_ablation_rows": [
            r_v7_bal_xdx,
            r_v7_newt_xdx,
            r_v7_bal_rr,
            r_v8_hyb_xdx,
            r_v8_hyb_rr,
            r_v8_inp_rr,
            r_v8_hyb_15,
            r_v8_hyb_16,
            r_v8_hyb_1516,
            r_v8_inp_1516,
            r_v8_mbr,
        ],
    }
    with open(OOF_DIR / "w8b_results_summary.pkl", "wb") as f:
        pickle.dump(summary_payload, f)

    print("\n" + "=" * 105)
    print(f"WAVE 8B & 8C MASTER EVALUATION COMPLETE IN {time.time() - t0:.1f}s")
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
