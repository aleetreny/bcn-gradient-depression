"""Wave 6 (`EXP-08` + `EXP-09` + `EXP-10`): Episode Re-Ranker, Head-Decoupled Transductive Calibration & Direct `spell_scores` MBR.

Executes a rigorous, zero-leakage 7-fold Nested LOMO & 4-Judge evaluation of:
0. Red-Team Fix A (`src/decode/x_d_x_arbitrator.py`):
   - Removes `ball_next_x` / `ball_next_y` from `_extract_centered_ball_coords` so interior tracking gaps are not
     falsely flagged as out-of-bounds when the post-gap ball reappears near a touchline.
1. `EXP-09` (`src/decode/transductive_calib.py`):
   - Red-Team Decoupled 4-Tree Synthesis (`combine_4tree_decoupled`):
     * Dead-ball head `P_tree(D)`: `0.35 * GBM_Plus + 0.45 * GBM_PhysV2 + 0.00 * CatBoost_PhysV2 + 0.20 * XGB_PhysV2`
     * Live head `logit P_tree(H|alive)`: `0.35 * GBM_Plus + 0.20 * GBM_PhysV2 + 0.25 * CatBoost_PhysV2 + 0.20 * XGB_PhysV2`
   - Head-Decoupled Multi-Family Synthesis (`synthesize_head_decoupled_proba`) combining Decoupled 4-Tree,
     physical GhostGate (`SymGBM_v6`), physical GapGate (`SymTCN` + `SymBiGRU`), and log-odds TV augmentation (`alpha_tv=0.20`).
   - Zero-label match-level transductive calibration (`calibrate_match_transductive`) tuned in 7-fold Nested LOMO.
2. `EXP-08` (`src/decode/spell_reranker.py`):
   - Spell-Level Physics Re-Ranker & Surgical `Y -> X -> Y` Merger (`L <= 125` frames) with 28 strictly H<->A invariant
     episode features and inner-LOMO Ridge/Physics validity scoring.
3. `EXP-10` (`src/decode/mbr_snap.py`):
   - Active 4-Directional Viterbi (`decode_active_4dir`), Physical Boundary Impulse Snapping (`snap_boundaries_to_impulse`),
     and Direct `spell_scores` Minimum Bayes Risk (MBR) Consensus Decoding (`mbr_decode_sequences`).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd

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
from src.decode.mbr_snap import (
    ImpulseSnapConfig,
    decode_active_4dir,
    mbr_decode_sequences,
    snap_boundaries_to_impulse,
)
from src.decode.spell_reranker import (
    SpellRerankerConfig,
    apply_spell_reranker,
    fit_spell_reranker_lomo,
)
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    combine_4tree_decoupled,
    compute_seq481_gate,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator, count_interior_xdx
from src.eval import metrics
from src.features.symmetry import swap_labels, swap_proba
from src.features.v1ctl_physics_v2 import mirror_v1ctl_physv2_df
from src.models.transition import change_features
from exp_b_symmetry_eval import blend_proba, predict_gbm_tta, predict_tcn_tta, predict_trans_tta
from exp_f_sym_decoder_and_boundary import augment_p_change_with_logodds_tv
from exp_v6_super_champion import compute_physical_gates, moe_combine, save_and_validate_submission


def load_cached_physv2_table(prefix: str, half: np.ndarray) -> pd.DataFrame:
    """Fast load of 481-column `v1ctl_physics_v2` DataFrame from `local_data/cache/` without recomputing fingerprints."""
    cdir = ROOT / "local_data" / "cache"
    p_f1 = sorted((cdir / "features_v1").glob(f"{prefix}*.parquet"))[0]
    p_c1 = sorted((cdir / "control_v1").glob(f"{prefix}*.parquet"))[0]
    p_pl = sorted((cdir / "plus_v1").glob(f"{prefix}*.parquet"))[0]
    p_p2 = sorted((cdir / "physv2").glob(f"{prefix}*.parquet"))[0]
    df1 = change_features(pd.read_parquet(p_f1), half)
    return pd.concat(
        [df1, pd.read_parquet(p_c1), pd.read_parquet(p_pl), pd.read_parquet(p_p2)],
        axis=1,
    )


def frame_acc(y_true: np.ndarray, pred_or_proba: np.ndarray) -> float:
    valid = y_true >= 0
    if not np.any(valid):
        return float("nan")
    pred = np.argmax(pred_or_proba[valid], axis=1) if pred_or_proba.ndim == 2 else pred_or_proba[valid]
    return float((pred == y_true[valid]).mean())


def main():
    t0 = time.time()
    print("=" * 112, flush=True)
    print("WAVE 6 (EXP-08 + EXP-09 + EXP-10): SPELL RE-RANKER, HEAD-DECOUPLED CALIBRATION & DIRECT SPELL-F1 MBR", flush=True)
    print("=" * 112, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "exp_w6_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load Matches, 481-Col Feature Tables, and All Cached Multi-Family OOFs
    # -------------------------------------------------------------------------
    t_load = time.time()
    sportec = load_all()
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    X_v3, X_v6 = {}, {}
    gates_v3, gates_v6 = {}, {}
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

    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    plus_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    h_dir = ROOT / "experiments" / "outputs" / "exp_h_oof"
    w3_dir = ROOT / "experiments" / "outputs" / "exp_w3_oof"
    w4_dir = ROOT / "experiments" / "outputs" / "exp_w4_oof"
    w5_dir = ROOT / "experiments" / "outputs" / "exp_w5_tree_oof"
    w5b_dir = ROOT / "experiments" / "outputs" / "exp_w5b_oof"

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std = pickle.load(f)[0]
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        loo_dec_lin, loo_dec_hyb, dec_v5_lin, dec_v5_hyb = pickle.load(f)
    dec_v4 = DecoderParams(**{
        k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items()
        if not k.startswith("_")
    })

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
        oof[T] = {
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
            "p4d_v3": zw3["p4d_v3"],
            "p4d_v6": zw3["p4d_v6"],
            "g_p2_v3": zw4["symgbm_physv2_tta"],
            "tr_p2_v3": zw4["symtrans_physv2_tta"],
            "g_p2_v6": zw4["symgbm_physv2_tta_on_v6"],
            "tr_p2_v6": zw4["symtrans_physv2_tta_on_v6"],
            "cb_p2_v3": zw5["symcatboost_physv2_tta"],
            "xgb_p2_v3": zw5["symxgb_physv2_tta"],
            "cb_p2_v6": zw5["symcatboost_physv2_tta_on_v6"],
            "xgb_p2_v6": zw5["symxgb_physv2_tta_on_v6"],
            "tcn_p2_v3": zw5b_tcn["v3_proba"],
            "tcnc_p2_v3": zw5b_tcn["v3_change"],
            "tcn_p2_v6": zw5b_tcn["v6_proba"],
            "tcnc_p2_v6": zw5b_tcn["v6_change"],
            "gru_p2_v3": zw5b_gru["v3_proba"],
            "gruc_p2_v3": zw5b_gru["v3_change"],
            "gru_p2_v6": zw5b_gru["v6_proba"],
            "gruc_p2_v6": zw5b_gru["v6_change"],
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

    # Also compute or load `g_v6`, `tr_v6`, `b_sym`, `bc_sym` on `atd_raw` for full J3 Raw evaluation
    atd_raw_extra_cache = out_dir / "atd_raw_v6_bigru.pkl"
    if atd_raw_extra_cache.exists():
        with open(atd_raw_extra_cache, "rb") as f:
            atd_raw_extra = pickle.load(f)
    else:
        with open(g_dir / "symgbm_v6_full.pkl", "rb") as f:
            gbm_v6_full = pickle.load(f)
        with open(g_dir / "symtrans_v6_full.pkl", "rb") as f:
            trans_v6_full = pickle.load(f)
        with open(h_dir / "symbigru_full.pkl", "rb") as f:
            bigru_full = pickle.load(f)
        v1ctl_cols = list(gbm_v6_full.features)
        X_atd_raw_v1ctl = X_atd_raw[v1ctl_cols]
        g_v6_raw = predict_gbm_tta(gbm_v6_full, X_atd_raw_v1ctl, atd.half, atd.pitch)
        tr_v6_raw = predict_trans_tta(trans_v6_full, X_atd_raw_v1ctl, atd.half, atd.pitch)
        b_sym_raw, bc_sym_raw = predict_tcn_tta(bigru_full, X_atd_raw_v1ctl, atd.half, atd.pitch)
        atd_raw_extra = {
            "g_v6": g_v6_raw,
            "tr_v6": tr_v6_raw,
            "b_sym": b_sym_raw,
            "bc_sym": bc_sym_raw,
        }
        with open(atd_raw_extra_cache, "wb") as f:
            pickle.dump(atd_raw_extra, f)

    print(f"  Loaded all 19 481-col feature tables & cached multi-family predictions in {time.time() - t_load:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # 2. Step 1: Verify Exact 0.000e+00 H <-> A Equivariance & Red-Team Fix A
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Exact H <-> A Equivariance & Red-Team Fix A (`x_d_x_arbitrator.py`) Verification ---", flush=True)
    T0 = "J03WMX"
    sl_test = slice(0, 6000)
    X_sub = X_v3[T0].iloc[sl_test].reset_index(drop=True)
    h_sub = sportec[T0].half[sl_test]
    X_sub_swap = mirror_v1ctl_physv2_df(X_sub, h_sub, sportec[T0].pitch)

    d0 = oof[T0]
    fams_sub = [d0["g_pl"][sl_test], d0["g_p2_v3"][sl_test], d0["cb_p2_v3"][sl_test], d0["xgb_p2_v3"][sl_test], d0["tcn"][sl_test], d0["gru"][sl_test]]
    fams_sub_swap = [swap_proba(p) for p in fams_sub]
    p4d_sub = d0["p4d_v3"][sl_test]
    p4d_sub_swap = p4d_sub[:, [1, 0, 2, 3]]

    p_syn, c_syn = synthesize_head_decoupled_proba(
        fams_sub[0], fams_sub[1], fams_sub[2], fams_sub[3], fams_sub[4], fams_sub[5],
        d0["tr_pl"][sl_test], d0["tr_p2_v3"][sl_test], d0["tcnc"][sl_test], d0["gruc"][sl_test],
        h_sub, d0["g_v6_on_v3"][sl_test], d0["tr_v6_on_v3"][sl_test],
        w_ghost=gates_v3[T0][0][sl_test], w_gap=gates_v3[T0][1][sl_test],
    )
    p_syn_s, c_syn_s = synthesize_head_decoupled_proba(
        fams_sub_swap[0], fams_sub_swap[1], fams_sub_swap[2], fams_sub_swap[3], fams_sub_swap[4], fams_sub_swap[5],
        d0["tr_pl"][sl_test], d0["tr_p2_v3"][sl_test], d0["tcnc"][sl_test], d0["gruc"][sl_test],
        h_sub, swap_proba(d0["g_v6_on_v3"][sl_test]), d0["tr_v6_on_v3"][sl_test],
        w_ghost=gates_v3[T0][0][sl_test], w_gap=gates_v3[T0][1][sl_test],
    )
    err_syn = float(np.max(np.abs(swap_proba(p_syn) - p_syn_s)))
    err_csyn = float(np.max(np.abs(c_syn - c_syn_s)))

    cal_cfg_test = TransductiveCalibConfig(T_alive=1.08, T_ha=0.94, blackout_still_gain=0.35, em_prior_strength=0.15)
    p_cal = calibrate_match_transductive(p_syn, h_sub, X_sub, cal_cfg_test)
    p_cal_s = calibrate_match_transductive(p_syn_s, h_sub, X_sub_swap, cal_cfg_test)
    err_cal = float(np.max(np.abs(swap_proba(p_cal) - p_cal_s)))

    pred_base = decode_v4(p_cal, h_sub, loo_dec_hyb[T0], p_change=c_syn)
    pred_base_s = decode_v4(p_cal_s, h_sub, loo_dec_hyb[T0], p_change=c_syn_s)
    err_dec = float(np.mean(swap_labels(pred_base) != pred_base_s))

    pred_xdx = apply_xdx_arbitrator(pred_base, p_cal, h_sub, X_sub, p_change=c_syn, p_trans_4d=p4d_sub, cfg=XDXConfig(enable_split=False))
    pred_xdx_s = apply_xdx_arbitrator(pred_base_s, p_cal_s, h_sub, X_sub_swap, p_change=c_syn_s, p_trans_4d=p4d_sub_swap, cfg=XDXConfig(enable_split=False))
    err_xdx = float(np.mean(swap_labels(pred_xdx) != pred_xdx_s))

    pred_rr = apply_spell_reranker(pred_xdx, p_cal, h_sub, X_sub, c_syn, p4d_sub, fams_sub, SpellRerankerConfig(tau_absorb=0.42))
    pred_rr_s = apply_spell_reranker(pred_xdx_s, p_cal_s, h_sub, X_sub_swap, c_syn_s, p4d_sub_swap, fams_sub_swap, SpellRerankerConfig(tau_absorb=0.42))
    err_rr = float(np.mean(swap_labels(pred_rr) != pred_rr_s))

    pred_sn = snap_boundaries_to_impulse(pred_rr, h_sub, X_sub, c_syn, p4d_sub, ImpulseSnapConfig())
    pred_sn_s = snap_boundaries_to_impulse(pred_rr_s, h_sub, X_sub_swap, c_syn_s, p4d_sub_swap, ImpulseSnapConfig())
    err_sn = float(np.mean(swap_labels(pred_sn) != pred_sn_s))

    pred_mbr = mbr_decode_sequences([pred_base, pred_xdx, pred_rr, pred_sn], h_sub, base_proba=p_cal, base_p_change=c_syn, dec_params=loo_dec_hyb[T0])
    pred_mbr_s = mbr_decode_sequences([pred_base_s, pred_xdx_s, pred_rr_s, pred_sn_s], h_sub, base_proba=p_cal_s, base_p_change=c_syn_s, dec_params=loo_dec_hyb[T0])
    err_mbr = float(np.mean(swap_labels(pred_mbr) != pred_mbr_s))

    print(
        f"  [Equivariance Audit] EXP-09 Synth={err_syn:.3e} (c={err_csyn:.3e}) | EXP-09 Calib={err_cal:.3e} | "
        f"Dec={err_dec:.3e} | EXP-03 FixA={err_xdx:.3e} | EXP-08 Rerank={err_rr:.3e} | "
        f"EXP-10 Snap={err_sn:.3e} | EXP-10 MBR={err_mbr:.3e}",
        flush=True,
    )
    assert max(err_syn, err_csyn, err_cal, err_dec, err_xdx, err_rr, err_sn, err_mbr) == 0.0, "H<->A symmetry violation!"

    # Check Red-Team Fix A impact on interior X-D-X absorption across Sportec v3, v6, Sample, BAR-ATH
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)

    # -------------------------------------------------------------------------
    # 3. Helper to synthesize probabilities for all 4 benchmarks under any DecoupledSynthConfig
    # -------------------------------------------------------------------------
    def build_benchmark_streams(
        scfg: DecoupledSynthConfig,
        ccfg: TransductiveCalibConfig | None = None,
        use_4seq: bool = False,
        use_seq481_gate: bool = False,
    ):
        streams = {"v3": {}, "v6": {}, "sample": {}, "atd_raw": {}, "atd_rep": {}}
        for T in MATCH_IDS:
            d = oof[T]
            h = sportec[T].half
            a_p2_v3 = compute_seq481_gate(X_v3[T], h, gates_v3[T][0]) if (use_4seq and use_seq481_gate) else None
            p3, c3 = synthesize_head_decoupled_proba(
                d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"],
                d["tcn"], d["gru"], d["tr_pl"], d["tr_p2_v3"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v3"], d["tr_v6_on_v3"],
                w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], cfg=scfg,
                tcn_physv2_proba=d["tcn_p2_v3"] if use_4seq else None,
                bigru_physv2_proba=d["gru_p2_v3"] if use_4seq else None,
                tcn_physv2_change=d["tcnc_p2_v3"] if use_4seq else None,
                bigru_physv2_change=d["gruc_p2_v3"] if use_4seq else None,
                alpha_p2=a_p2_v3,
            )
            if ccfg is not None:
                p3 = calibrate_match_transductive(p3, h, X_v3[T], ccfg)
            f3 = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"]]
            if use_4seq:
                f3.extend([d["tcn_p2_v3"], d["gru_p2_v3"]])
            streams["v3"][T] = (p3, c3, d["p4d_v3"], f3)

            # On degrade_v6 test folds, g_v3_on_v6 is the v1ctl/Plus baseline; g_p2_v6, cb_p2_v6, xgb_p2_v6 are PhysV2 trees on v6
            a_p2_v6 = compute_seq481_gate(X_v6[T], h, gates_v6[T][0]) if (use_4seq and use_seq481_gate) else None
            p6, c6 = synthesize_head_decoupled_proba(
                d["g_v3_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn"], d["gru"], d["tr_v3_on_v6"], d["tr_p2_v6"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg,
                tcn_physv2_proba=d["tcn_p2_v6"] if use_4seq else None,
                bigru_physv2_proba=d["gru_p2_v6"] if use_4seq else None,
                tcn_physv2_change=d["tcnc_p2_v6"] if use_4seq else None,
                bigru_physv2_change=d["gruc_p2_v6"] if use_4seq else None,
                alpha_p2=a_p2_v6,
            )
            if ccfg is not None:
                p6 = calibrate_match_transductive(p6, h, X_v6[T], ccfg)
            f6 = [d["g_v3_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"]]
            if use_4seq:
                f6.extend([d["tcn_p2_v6"], d["gru_p2_v6"]])
            streams["v6"][T] = (p6, c6, d["p4d_v6"], f6)

        for gid in GAMES:
            ms = sample_matches[gid]
            dh = sample_h[gid]
            dpl = pe_plus["sample"][gid]
            dv6 = sample_v6[gid]
            dp2 = pe_p2["sample"][gid]
            dw5 = pe_w5["sample"][gid]
            dw5b = pe_w5b["sample"][gid]
            a_p2_s = compute_seq481_gate(X_sample[gid], ms.half, gates_sample[gid][0]) if (use_4seq and use_seq481_gate) else None
            ps, cs = synthesize_head_decoupled_proba(
                dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
                dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
                ms.half, dv6["g_v6"], dv6["tr_v6"],
                w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg,
                tcn_physv2_proba=dw5b["tcn_physv2_sym"] if use_4seq else None,
                bigru_physv2_proba=dw5b["bigru_physv2_sym"] if use_4seq else None,
                tcn_physv2_change=dw5b["tcnc_physv2_sym"] if use_4seq else None,
                bigru_physv2_change=dw5b["bigruc_physv2_sym"] if use_4seq else None,
                alpha_p2=a_p2_s,
            )
            if ccfg is not None:
                ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], ccfg)
            fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"]]
            if use_4seq:
                fs.extend([dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"]])
            streams["sample"][gid] = (ps, cs, b4d_j23["sample"][gid], fs)

        # J3 Raw
        dpl_ar = pe_plus["atd_raw"]
        dp2_ar = pe_p2["atd_raw"]
        dw5_ar = pe_w5["atd_raw"]
        dw5b_ar = pe_w5b["atd_raw"]
        a_p2_ar = compute_seq481_gate(X_atd_raw, atd.half, gate_atd_raw[0]) if (use_4seq and use_seq481_gate) else None
        par, car = synthesize_head_decoupled_proba(
            dpl_ar["g_plus_sym"], dp2_ar["g_physv2_sym"], dw5_ar["cb_physv2_sym"], dw5_ar["xgb_physv2_sym"],
            dpl_ar["t_sym"], atd_raw_extra["b_sym"], dpl_ar["tr_plus_sym"], dp2_ar["tr_physv2_sym"],
            dpl_ar["tc_sym"], atd_raw_extra["bc_sym"],
            atd.half, atd_raw_extra["g_v6"], atd_raw_extra["tr_v6"],
            w_ghost=gate_atd_raw[0], w_gap=gate_atd_raw[1], cfg=scfg,
            tcn_physv2_proba=dw5b_ar["tcn_physv2_sym"] if use_4seq else None,
            bigru_physv2_proba=dw5b_ar["bigru_physv2_sym"] if use_4seq else None,
            tcn_physv2_change=dw5b_ar["tcnc_physv2_sym"] if use_4seq else None,
            bigru_physv2_change=dw5b_ar["bigruc_physv2_sym"] if use_4seq else None,
            alpha_p2=a_p2_ar,
        )
        if ccfg is not None:
            par = calibrate_match_transductive(par, half_atd_dec, X_atd_raw, ccfg)
        far = [dpl_ar["g_plus_sym"], dp2_ar["g_physv2_sym"], dw5_ar["cb_physv2_sym"], dw5_ar["xgb_physv2_sym"], dpl_ar["t_sym"], atd_raw_extra["b_sym"]]
        if use_4seq:
            far.extend([dw5b_ar["tcn_physv2_sym"], dw5b_ar["bigru_physv2_sym"]])
        streams["atd_raw"] = (par, car, b4d_j23["atd_raw"], far)

        # J3 Broadcast-Repaired
        dpl_ap = pe_plus["atd_rep"]
        dp2_ap = pe_p2["atd_rep"]
        dw5_ap = pe_w5["atd_rep"]
        dw5b_ap = pe_w5b["atd_rep"]
        a_p2_ap = compute_seq481_gate(X_atd_rep, atd.half, gate_atd_rep[0]) if (use_4seq and use_seq481_gate) else None
        pap, cap = synthesize_head_decoupled_proba(
            dpl_ap["g_plus_sym"], dp2_ap["g_physv2_sym"], dw5_ap["cb_physv2_sym"], dw5_ap["xgb_physv2_sym"],
            atd_rep_h["t_sym"], atd_rep_h["b_sym"], dpl_ap["tr_plus_sym"], dp2_ap["tr_physv2_sym"],
            atd_rep_h["tc_sym"], atd_rep_h["bc_sym"],
            atd.half, atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"],
            w_ghost=gate_atd_rep[0], w_gap=gate_atd_rep[1], cfg=scfg,
            tcn_physv2_proba=dw5b_ap["tcn_physv2_sym"] if use_4seq else None,
            bigru_physv2_proba=dw5b_ap["bigru_physv2_sym"] if use_4seq else None,
            tcn_physv2_change=dw5b_ap["tcnc_physv2_sym"] if use_4seq else None,
            bigru_physv2_change=dw5b_ap["bigruc_physv2_sym"] if use_4seq else None,
            alpha_p2=a_p2_ap,
        )
        if ccfg is not None:
            pap = calibrate_match_transductive(pap, half_atd_dec, X_atd_rep, ccfg)
        fap = [dpl_ap["g_plus_sym"], dp2_ap["g_physv2_sym"], dw5_ap["cb_physv2_sym"], dw5_ap["xgb_physv2_sym"], atd_rep_h["t_sym"], atd_rep_h["b_sym"]]
        if use_4seq:
            fap.extend([dw5b_ap["tcn_physv2_sym"], dw5b_ap["bigru_physv2_sym"]])
        streams["atd_rep"] = (pap, cap, b4d_j23["atd_rep"], fap)
        return streams

    # -------------------------------------------------------------------------
    # 4. Step 2: Nested LOMO Selection for EXP-09 (Head-Decoupled Synthesis & Transductive Calib)
    # -------------------------------------------------------------------------
    print("\n--- Step 2: EXP-09 Head-Decoupled Synthesis & Transductive Calibration (7-Fold Nested LOMO) ---", flush=True)
    synth_grid = [
        ("6Fam_Uniform4Tree_Hyb55_TV20", False, False, DecoupledSynthConfig(
            w_tree_d_tuple=(0.25, 0.25, 0.25, 0.25),
            w_tree_ha_tuple=(0.25, 0.25, 0.25, 0.25),
            w_tree_d=0.55, w_tree_ha=0.55, alpha_tv=0.20,
        )),
        ("6Fam_RedTeam4Tree_Hyb55_TV20", False, False, DecoupledSynthConfig(
            w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
            w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
            w_tree_d=0.55, w_tree_ha=0.55, alpha_tv=0.20,
        )),
        ("8Fam_RedTeam4Tree_Static4Seq_D58_HA50_TV20", True, False, DecoupledSynthConfig(
            w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
            w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
            w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
            w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
            w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.20,
        )),
        ("8Fam_RedTeam4Tree_SeqGate_D58_HA50_TV08", True, True, DecoupledSynthConfig(
            w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
            w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
            w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
            w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
            w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
        )),
        ("8Fam_RedTeam4Tree_SeqGate_D62_HA48_TV08", True, True, DecoupledSynthConfig(
            w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
            w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
            w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
            w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
            w_tree_d=0.62, w_tree_ha=0.48, alpha_tv=0.08,
        )),
        ("8Fam_RedTeam4Tree_SeqGate_D55_HA52_TV15", True, True, DecoupledSynthConfig(
            w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
            w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
            w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
            w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
            w_tree_d=0.55, w_tree_ha=0.52, alpha_tv=0.15,
        )),
    ]

    streams_cache = {}
    for sname, use_4s, use_sg, scfg in synth_grid:
        st = build_benchmark_streams(scfg, use_4seq=use_4s, use_seq481_gate=use_sg)
        streams_cache[sname] = st
        v3_f1, v6_f1 = [], []
        for T in MATCH_IDS:
            p3, c3, _, _ = st["v3"][T]
            p6, c6, _, _ = st["v6"][T]
            pr3 = decode_v4(p3, sportec[T].half, loo_dec_hyb[T], p_change=c3)
            pr6 = decode_v4(p6, sportec[T].half, loo_dec_hyb[T], p_change=c6)
            v3_f1.append(metrics.score(sportec[T].labels, pr3, sportec[T].half)["spell_f1_mean"])
            v6_f1.append(metrics.score(sportec[T].labels, pr6, sportec[T].half)["spell_f1_mean"])
        print(f"  [EXP-09 Synth] {sname:42s} | LOMO v3={np.mean(v3_f1):.4f} | LOMO v6={np.mean(v6_f1):.4f} | Mean={0.5*(np.mean(v3_f1)+np.mean(v6_f1)):.4f}", flush=True)

    # Test Transductive Calibration grid on top of 8Fam_RedTeam4Tree_SeqGate_D58_HA50_TV08 in 7-fold LOMO
    base_rt_scfg = synth_grid[3][3]
    base_rt_streams = streams_cache["8Fam_RedTeam4Tree_SeqGate_D58_HA50_TV08"]
    calib_grid = [
        ("NoCalib (T_alive=1.0, T_ha=1.0, gain=0.0)", TransductiveCalibConfig()),
        ("KinBlackout_g0.25 (still_pivot=0.38)", TransductiveCalibConfig(blackout_still_gain=0.25, still_pivot=0.38)),
        ("KinBlackout_g0.40 (still_pivot=0.38)", TransductiveCalibConfig(blackout_still_gain=0.40, still_pivot=0.38)),
        ("Temp_Ta0.95_Tha0.95 + KinBlackout_g0.25", TransductiveCalibConfig(T_alive=0.95, T_ha=0.95, blackout_still_gain=0.25)),
        ("Temp_Ta1.05_Tha0.95 + KinBlackout_g0.25", TransductiveCalibConfig(T_alive=1.05, T_ha=0.95, blackout_still_gain=0.25)),
        ("DampedEM_0.10 + KinBlackout_g0.25", TransductiveCalibConfig(blackout_still_gain=0.25, em_prior_strength=0.10)),
        ("Pseudo1Round_a0.08 + KinBlackout_g0.25", TransductiveCalibConfig(blackout_still_gain=0.25, pseudo_alpha=0.08)),
    ]

    best_calib_cfg = TransductiveCalibConfig(blackout_still_gain=0.25, still_pivot=0.38)
    best_calib_mean = -1.0
    for cname, ccfg in calib_grid:
        v3_f1, v6_f1 = [], []
        for T in MATCH_IDS:
            p3_raw, c3, _, _ = base_rt_streams["v3"][T]
            p6_raw, c6, _, _ = base_rt_streams["v6"][T]
            pr3_init = decode_v4(p3_raw, sportec[T].half, loo_dec_hyb[T], p_change=c3) if ccfg.pseudo_alpha > 0 else None
            pr6_init = decode_v4(p6_raw, sportec[T].half, loo_dec_hyb[T], p_change=c6) if ccfg.pseudo_alpha > 0 else None
            p3 = calibrate_match_transductive(p3_raw, sportec[T].half, X_v3[T], ccfg, pred_init=pr3_init)
            p6 = calibrate_match_transductive(p6_raw, sportec[T].half, X_v6[T], ccfg, pred_init=pr6_init)
            pr3 = decode_v4(p3, sportec[T].half, loo_dec_hyb[T], p_change=c3)
            pr6 = decode_v4(p6, sportec[T].half, loo_dec_hyb[T], p_change=c6)
            v3_f1.append(metrics.score(sportec[T].labels, pr3, sportec[T].half)["spell_f1_mean"])
            v6_f1.append(metrics.score(sportec[T].labels, pr6, sportec[T].half)["spell_f1_mean"])
        m_both = 0.5 * (float(np.mean(v3_f1)) + float(np.mean(v6_f1)))
        print(f"  [EXP-09 Calib] {cname:42s} | LOMO v3={np.mean(v3_f1):.4f} | LOMO v6={np.mean(v6_f1):.4f} | Mean={m_both:.4f}", flush=True)
        if m_both > best_calib_mean:
            best_calib_mean = m_both
            best_calib_cfg = ccfg

    # -------------------------------------------------------------------------
    # 5. Step 3: EXP-08 Spell-Level Physics Re-Ranker & Surgical Y -> X -> Y Merger
    # -------------------------------------------------------------------------
    print("\n--- Step 3: EXP-08 Spell-Level Physics Re-Ranker & Surgical Y->X->Y Merger (7-Fold Nested LOMO) ---", flush=True)
    # Prepare decoded + EXP-03 (Fixed) predictions for each Sportec match on v3 and v6
    preds_xdx_v3, preds_xdx_v6 = {}, {}
    probas_rt_v3, probas_rt_v6 = {}, {}
    pchange_rt_v3, pchange_rt_v6 = {}, {}
    p4d_rt_v3, p4d_rt_v6 = {}, {}
    fams_rt_v3, fams_rt_v6 = {}, {}
    halves_sp, gt_sp = {}, {}

    for T in MATCH_IDS:
        p3_raw, c3, p4d3, f3 = base_rt_streams["v3"][T]
        p6_raw, c6, p4d6, f6 = base_rt_streams["v6"][T]
        p3 = calibrate_match_transductive(p3_raw, sportec[T].half, X_v3[T], best_calib_cfg)
        p6 = calibrate_match_transductive(p6_raw, sportec[T].half, X_v6[T], best_calib_cfg)
        pr3_dec = decode_v4(p3, sportec[T].half, loo_dec_hyb[T], p_change=c3)
        pr6_dec = decode_v4(p6, sportec[T].half, loo_dec_hyb[T], p_change=c6)
        pr3_xdx = apply_xdx_arbitrator(pr3_dec, p3, sportec[T].half, X_v3[T], p_change=c3, p_trans_4d=p4d3, cfg=xdx_cfg_st1)
        pr6_xdx = apply_xdx_arbitrator(pr6_dec, p6, sportec[T].half, X_v6[T], p_change=c6, p_trans_4d=p4d6, cfg=xdx_cfg_st1)

        preds_xdx_v3[T] = pr3_xdx
        preds_xdx_v6[T] = pr6_xdx
        probas_rt_v3[T] = p3
        probas_rt_v6[T] = p6
        pchange_rt_v3[T] = c3
        pchange_rt_v6[T] = c6
        p4d_rt_v3[T] = p4d3
        p4d_rt_v6[T] = p4d6
        fams_rt_v3[T] = f3
        fams_rt_v6[T] = f6
        halves_sp[T] = sportec[T].half
        gt_sp[T] = sportec[T].labels

    # Fit 7-fold Nested LOMO SpellRerankerConfig on v3 (and evaluate on outer test match T for both v3 and v6)
    lomo_rerank_cfgs = {}
    v3_before_rr, v3_after_rr = [], []
    v6_before_rr, v6_after_rr = [], []
    tot_cand_v3, tot_abs_v3 = 0, 0
    tot_cand_v6, tot_abs_v6 = 0, 0

    for T in MATCH_IDS:
        train_ids = [m for m in MATCH_IDS if m != T]
        cfg_fold, info_fold = fit_spell_reranker_lomo(
            train_ids,
            preds_xdx_v3, probas_rt_v3, halves_sp, gt_sp, X_v3,
            pchange_rt_v3, p4d_rt_v3, fams_rt_v3,
            max_len_grid=(60, 85, 100, 120),
            tau_grid=(0.0, 0.34, 0.38, 0.42, 0.45, 0.48, 0.50),
            modes=("physics", "hybrid"),
            l2_reg=15.0,
        )
        lomo_rerank_cfgs[T] = cfg_fold

        pr3_rr, diag3 = apply_spell_reranker(
            preds_xdx_v3[T], probas_rt_v3[T], halves_sp[T], X_v3[T],
            pchange_rt_v3[T], p4d_rt_v3[T], fams_rt_v3[T], cfg=cfg_fold, return_diagnostics=True,
        )
        pr6_rr, diag6 = apply_spell_reranker(
            preds_xdx_v6[T], probas_rt_v6[T], halves_sp[T], X_v6[T],
            pchange_rt_v6[T], p4d_rt_v6[T], fams_rt_v6[T], cfg=cfg_fold, return_diagnostics=True,
        )
        f1_3_b = float(metrics.score(gt_sp[T], preds_xdx_v3[T], halves_sp[T])["spell_f1_mean"])
        f1_3_a = float(metrics.score(gt_sp[T], pr3_rr, halves_sp[T])["spell_f1_mean"])
        f1_6_b = float(metrics.score(gt_sp[T], preds_xdx_v6[T], halves_sp[T])["spell_f1_mean"])
        f1_6_a = float(metrics.score(gt_sp[T], pr6_rr, halves_sp[T])["spell_f1_mean"])
        v3_before_rr.append(f1_3_b)
        v3_after_rr.append(f1_3_a)
        v6_before_rr.append(f1_6_b)
        v6_after_rr.append(f1_6_a)
        tot_cand_v3 += diag3["n_yxy_candidates"]
        tot_abs_v3 += diag3["n_yxy_absorbed"]
        tot_cand_v6 += diag6["n_yxy_candidates"]
        tot_abs_v6 += diag6["n_yxy_absorbed"]
        print(
            f"  [Fold {T}] Inner(mode={info_fold['best_mode']:7s}, L<={info_fold['best_max_len']:3d}, tau={info_fold['best_tau']:.2f}, "
            f"gain={info_fold['inner_gain']:+.4f}) | Outer v3: {f1_3_b:.4f}->{f1_3_a:.4f} (abs {diag3['n_yxy_absorbed']}/{diag3['n_yxy_candidates']}) | "
            f"v6: {f1_6_b:.4f}->{f1_6_a:.4f} (abs {diag6['n_yxy_absorbed']}/{diag6['n_yxy_candidates']})",
            flush=True,
        )

    # Fit full 7-match Sportec SpellRerankerConfig for Judge 2 & Judge 3
    full_rerank_cfg, full_rerank_info = fit_spell_reranker_lomo(
        list(MATCH_IDS),
        preds_xdx_v3, probas_rt_v3, halves_sp, gt_sp, X_v3,
        pchange_rt_v3, p4d_rt_v3, fams_rt_v3,
        max_len_grid=(60, 85, 100, 120),
        tau_grid=(0.0, 0.34, 0.38, 0.42, 0.45, 0.48, 0.50),
        modes=("physics", "hybrid"),
        l2_reg=15.0,
    )
    print(
        f"  [Full 7-Match Sportec Fit] mode={full_rerank_info['best_mode']}, max_len={full_rerank_info['best_max_len']}, "
        f"tau={full_rerank_info['best_tau']:.2f}, inner_gain={full_rerank_info['inner_gain']:+.4f} "
        f"(Outer LOMO v3: {np.mean(v3_before_rr):.4f}->{np.mean(v3_after_rr):.4f}, v6: {np.mean(v6_before_rr):.4f}->{np.mean(v6_after_rr):.4f})",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # 6. Step 4: EXP-10 Active 4-Dir Viterbi & Physical Boundary Impulse Snapping
    # -------------------------------------------------------------------------
    print("\n--- Step 4: EXP-10 Active 4-Dir Viterbi & Physical Impulse Snapping Audit ---", flush=True)
    for b_ha, b_tod, b_fromd, g_short in [
        (0.0, 0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0, 0.0),
        (0.25, 0.15, 0.15, 0.0),
        (0.20, 0.10, 0.10, 0.02),
    ]:
        s3_4d, s6_4d = [], []
        for T in MATCH_IDS:
            pr3 = decode_active_4dir(
                probas_rt_v3[T], halves_sp[T], loo_dec_hyb[T],
                p_change=pchange_rt_v3[T], p_trans_4d=p4d_rt_v3[T],
                beta_dir_ha=b_ha, beta_to_d=b_tod, beta_from_d=b_fromd,
                gamma_short=g_short, dmin_team_hard=22 if g_short > 0 else None, d_soft_team=38 if g_short > 0 else None,
            )
            pr6 = decode_active_4dir(
                probas_rt_v6[T], halves_sp[T], loo_dec_hyb[T],
                p_change=pchange_rt_v6[T], p_trans_4d=p4d_rt_v6[T],
                beta_dir_ha=b_ha, beta_to_d=b_tod, beta_from_d=b_fromd,
                gamma_short=g_short, dmin_team_hard=22 if g_short > 0 else None, d_soft_team=38 if g_short > 0 else None,
            )
            s3_4d.append(metrics.score(gt_sp[T], pr3, halves_sp[T])["spell_f1_mean"])
            s6_4d.append(metrics.score(gt_sp[T], pr6, halves_sp[T])["spell_f1_mean"])
        print(
            f"  [EXP-10A 4DirViterbi] beta_ha={b_ha:.2f}, beta_toD={b_tod:.2f}, beta_fromD={b_fromd:.2f}, gamma={g_short:.2f} "
            f"| LOMO v3={np.mean(s3_4d):.4f} | LOMO v6={np.mean(s6_4d):.4f}",
            flush=True,
        )

    for min_g in (0.06, 0.10, 0.15):
        s3_sn, s6_sn, n_sn3 = [], [], 0
        scfg_try = ImpulseSnapConfig(window=5, min_gain=min_g)
        for T in MATCH_IDS:
            pr3_s, d_s = snap_boundaries_to_impulse(
                preds_xdx_v3[T], halves_sp[T], X_v3[T], pchange_rt_v3[T], p4d_rt_v3[T], cfg=scfg_try, return_diagnostics=True
            )
            pr6_s = snap_boundaries_to_impulse(
                preds_xdx_v6[T], halves_sp[T], X_v6[T], pchange_rt_v6[T], p4d_rt_v6[T], cfg=scfg_try
            )
            s3_sn.append(metrics.score(gt_sp[T], pr3_s, halves_sp[T])["spell_f1_mean"])
            s6_sn.append(metrics.score(gt_sp[T], pr6_s, halves_sp[T])["spell_f1_mean"])
            n_sn3 += d_s["n_snapped"]
        print(
            f"  [EXP-10B ImpulseSnap] window=5, min_gain={min_g:.2f} (snapped={n_sn3:3d} boundaries on v3) "
            f"| LOMO v3={np.mean(s3_sn):.4f} | LOMO v6={np.mean(s6_sn):.4f}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # 7. Step 5: Comprehensive 4-Judge Evaluation & Ablation Matrix
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Comprehensive 4-Benchmark Evaluation (J1A, J1B, J2, J3 Raw & Repaired) ---", flush=True)

    # Precompute diverse candidate streams for MBR and ablation
    st_unif = streams_cache["6Fam_Uniform4Tree_Hyb55_TV20"]
    st_6fam_rt = streams_cache["6Fam_RedTeam4Tree_Hyb55_TV20"]
    st_rt_raw = streams_cache["8Fam_RedTeam4Tree_SeqGate_D58_HA50_TV08"]
    st_rt_cal = build_benchmark_streams(base_rt_scfg, best_calib_cfg, use_4seq=True, use_seq481_gate=True)
    st_rt_d60 = build_benchmark_streams(synth_grid[4][3], best_calib_cfg, use_4seq=True, use_seq481_gate=True)
    st_rt_tv25 = build_benchmark_streams(synth_grid[5][3], best_calib_cfg, use_4seq=True, use_seq481_gate=True)

    phys_rerank_cfg = SpellRerankerConfig(mode="physics", max_len_yxy=100, tau_absorb=0.45)
    snap_cfg = ImpulseSnapConfig(window=5, min_gain=0.10)

    def eval_pipeline(
        name: str,
        mode_key: str,
        sub_filename: str | None = None,
    ) -> dict:
        j1_v3_f1, j1_v3_acc, j1_v3_xdx = [], [], 0
        j1_v6_f1, j1_v6_acc, j1_v6_xdx = [], [], 0
        oof_preds_v3, oof_preds_v6 = {}, {}

        # --- 1. J1A (Sportec v3) & J1B (Sportec v6) ---
        for T in MATCH_IDS:
            d = oof[T]
            h3 = sportec[T].half
            if mode_key == "v6_champ":
                p3, c3 = moe_combine(
                    d["g_v1"], d["tr_v1"], d["g_pl"], d["tr_pl"], d["g_v6_on_v3"], d["tr_v6_on_v3"],
                    d["tcn"], d["tcnc"], d["gru"], d["gruc"],
                    w_pl=0.5, w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], w_gbm=0.55, w_c=0.50, mode="linear",
                )
                p6, c6 = moe_combine(
                    d["g_v3_on_v6"], d["tr_v3_on_v6"], d["g_v3_on_v6"], d["tr_v3_on_v6"], d["g_v6_on_v6"], d["tr_v6_on_v6"],
                    d["tcn"], d["tcnc"], d["gru"], d["gruc"],
                    w_pl=0.0, w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], w_gbm=0.55, w_c=0.50, mode="linear",
                )
                pr3 = decode_v4(p3, h3, fold_dec_std[T], p_change=c3)
                pr6 = decode_v4(p6, h3, fold_dec_std[T], p_change=c6)
            elif mode_key == "w5_unif_tv_symdec":
                p3, c3, _, _ = st_unif["v3"][T]
                p6, c6, _, _ = st_unif["v6"][T]
                pr3 = decode_v4(p3, h3, loo_dec_hyb[T], p_change=c3)
                pr6 = decode_v4(p6, h3, loo_dec_hyb[T], p_change=c6)
            elif mode_key == "exp09_6fam_decoupled":
                p3, c3, _, _ = st_6fam_rt["v3"][T]
                p6, c6, _, _ = st_6fam_rt["v6"][T]
                pr3 = decode_v4(p3, h3, loo_dec_hyb[T], p_change=c3)
                pr6 = decode_v4(p6, h3, loo_dec_hyb[T], p_change=c6)
            elif mode_key == "exp09_decoupled":
                p3, c3, _, _ = st_rt_raw["v3"][T]
                p6, c6, _, _ = st_rt_raw["v6"][T]
                pr3 = decode_v4(p3, h3, loo_dec_hyb[T], p_change=c3)
                pr6 = decode_v4(p6, h3, loo_dec_hyb[T], p_change=c6)
            elif mode_key == "exp09_calib":
                p3, c3, _, _ = st_rt_cal["v3"][T]
                p6, c6, _, _ = st_rt_cal["v6"][T]
                pr3 = decode_v4(p3, h3, loo_dec_hyb[T], p_change=c3)
                pr6 = decode_v4(p6, h3, loo_dec_hyb[T], p_change=c6)
            elif mode_key == "exp09_xdx_fixa":
                p3, c3, p4d3, _ = st_rt_cal["v3"][T]
                p6, c6, p4d6, _ = st_rt_cal["v6"][T]
                pr3 = apply_xdx_arbitrator(decode_v4(p3, h3, loo_dec_hyb[T], p_change=c3), p3, h3, X_v3[T], p_change=c3, p_trans_4d=p4d3, cfg=xdx_cfg_st1)
                pr6 = apply_xdx_arbitrator(decode_v4(p6, h3, loo_dec_hyb[T], p_change=c6), p6, h3, X_v6[T], p_change=c6, p_trans_4d=p4d6, cfg=xdx_cfg_st1)
            elif mode_key == "exp08_phys_rerank":
                p3, c3, p4d3, f3 = st_rt_cal["v3"][T]
                p6, c6, p4d6, f6 = st_rt_cal["v6"][T]
                pr3 = apply_xdx_arbitrator(decode_v4(p3, h3, loo_dec_hyb[T], p_change=c3), p3, h3, X_v3[T], p_change=c3, p_trans_4d=p4d3, cfg=xdx_cfg_st1)
                pr6 = apply_xdx_arbitrator(decode_v4(p6, h3, loo_dec_hyb[T], p_change=c6), p6, h3, X_v6[T], p_change=c6, p_trans_4d=p4d6, cfg=xdx_cfg_st1)
                pr3 = apply_spell_reranker(pr3, p3, h3, X_v3[T], c3, p4d3, f3, cfg=phys_rerank_cfg)
                pr6 = apply_spell_reranker(pr6, p6, h3, X_v6[T], c6, p4d6, f6, cfg=phys_rerank_cfg)
            elif mode_key == "exp08_lomo_rerank":
                p3, c3, p4d3, f3 = st_rt_cal["v3"][T]
                p6, c6, p4d6, f6 = st_rt_cal["v6"][T]
                pr3 = apply_xdx_arbitrator(decode_v4(p3, h3, loo_dec_hyb[T], p_change=c3), p3, h3, X_v3[T], p_change=c3, p_trans_4d=p4d3, cfg=xdx_cfg_st1)
                pr6 = apply_xdx_arbitrator(decode_v4(p6, h3, loo_dec_hyb[T], p_change=c6), p6, h3, X_v6[T], p_change=c6, p_trans_4d=p4d6, cfg=xdx_cfg_st1)
                pr3 = apply_spell_reranker(pr3, p3, h3, X_v3[T], c3, p4d3, f3, cfg=lomo_rerank_cfgs[T])
                pr6 = apply_spell_reranker(pr6, p6, h3, X_v6[T], c6, p4d6, f6, cfg=lomo_rerank_cfgs[T])
            elif mode_key == "exp10_4dir_snap":
                p3, c3, p4d3, f3 = st_rt_cal["v3"][T]
                p6, c6, p4d6, f6 = st_rt_cal["v6"][T]
                pr3 = decode_active_4dir(p3, h3, loo_dec_hyb[T], p_change=c3, p_trans_4d=p4d3, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10)
                pr6 = decode_active_4dir(p6, h3, loo_dec_hyb[T], p_change=c6, p_trans_4d=p4d6, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10)
                pr3 = apply_xdx_arbitrator(pr3, p3, h3, X_v3[T], p_change=c3, p_trans_4d=p4d3, cfg=xdx_cfg_st1)
                pr6 = apply_xdx_arbitrator(pr6, p6, h3, X_v6[T], p_change=c6, p_trans_4d=p4d6, cfg=xdx_cfg_st1)
                pr3 = apply_spell_reranker(pr3, p3, h3, X_v3[T], c3, p4d3, f3, cfg=lomo_rerank_cfgs[T])
                pr6 = apply_spell_reranker(pr6, p6, h3, X_v6[T], c6, p4d6, f6, cfg=lomo_rerank_cfgs[T])
                pr3 = snap_boundaries_to_impulse(pr3, h3, X_v3[T], c3, p4d3, cfg=snap_cfg)
                pr6 = snap_boundaries_to_impulse(pr6, h3, X_v6[T], c6, p4d6, cfg=snap_cfg)
            elif mode_key == "exp10_mbr_grand":
                # Build 6 diverse, H<->A equivariant candidate sequences per match and run direct spell_f1_mean MBR
                p3, c3, p4d3, f3 = st_rt_cal["v3"][T]
                p6, c6, p4d6, f6 = st_rt_cal["v6"][T]
                p3_6rt, c3_6rt, _, _ = st_6fam_rt["v3"][T]
                p6_6rt, c6_6rt, _, _ = st_6fam_rt["v6"][T]
                p3_d60, c3_d60, _, _ = st_rt_d60["v3"][T]
                p6_d60, c6_d60, _, _ = st_rt_d60["v6"][T]
                p3_tv25, c3_tv25, _, _ = st_rt_tv25["v3"][T]
                p6_tv25, c6_tv25, _, _ = st_rt_tv25["v6"][T]

                def _make_cands(p_main, c_main, p4d, fams, X_df, p_6rt, c_6rt, p_d60, c_d60, p_tv25, c_tv25, dp_hyb, dp_std, rr_cfg):
                    # Cand 1: 8Fam SeqGate + Calib + SymDec + XDX(FixA) + LOMO SpellReranker
                    s1 = apply_xdx_arbitrator(decode_v4(p_main, h3, dp_hyb, p_change=c_main), p_main, h3, X_df, p_change=c_main, p_trans_4d=p4d, cfg=xdx_cfg_st1)
                    s1 = apply_spell_reranker(s1, p_main, h3, X_df, c_main, p4d, fams, cfg=rr_cfg)
                    # Cand 2: 8Fam SeqGate + Calib + SymDec + XDX(FixA) + PhysReranker + ImpulseSnap
                    s2 = apply_xdx_arbitrator(decode_v4(p_main, h3, dp_hyb, p_change=c_main), p_main, h3, X_df, p_change=c_main, p_trans_4d=p4d, cfg=xdx_cfg_st1)
                    s2 = apply_spell_reranker(s2, p_main, h3, X_df, c_main, p4d, fams, cfg=phys_rerank_cfg)
                    s2 = snap_boundaries_to_impulse(s2, h3, X_df, c_main, p4d, cfg=snap_cfg)
                    # Cand 3: Active 4-Dir Viterbi + XDX(FixA) + LOMO SpellReranker
                    s3 = decode_active_4dir(p_main, h3, dp_hyb, p_change=c_main, p_trans_4d=p4d, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10)
                    s3 = apply_xdx_arbitrator(s3, p_main, h3, X_df, p_change=c_main, p_trans_4d=p4d, cfg=xdx_cfg_st1)
                    s3 = apply_spell_reranker(s3, p_main, h3, X_df, c_main, p4d, fams, cfg=rr_cfg)
                    # Cand 4: 8Fam D62/HA48 + SymDec + XDX(FixA) + SpellReranker
                    s4 = apply_xdx_arbitrator(decode_v4(p_d60, h3, dp_hyb, p_change=c_d60), p_d60, h3, X_df, p_change=c_d60, p_trans_4d=p4d, cfg=xdx_cfg_st1)
                    s4 = apply_spell_reranker(s4, p_d60, h3, X_df, c_d60, p4d, fams, cfg=rr_cfg)
                    # Cand 5: 8Fam TV25 + StdDec + XDX(FixA) + SpellReranker
                    s5 = apply_xdx_arbitrator(decode_v4(p_tv25, h3, dp_std, p_change=c_tv25), p_tv25, h3, X_df, p_change=c_tv25, p_trans_4d=p4d, cfg=xdx_cfg_st1)
                    s5 = apply_spell_reranker(s5, p_tv25, h3, X_df, c_tv25, p4d, fams, cfg=rr_cfg)
                    # Cand 6: 6Fam RedTeam4Tree + SymDec + XDX(FixA) + SpellReranker
                    s6 = apply_xdx_arbitrator(decode_v4(p_6rt, h3, dp_hyb, p_change=c_6rt), p_6rt, h3, X_df, p_change=c_6rt, p_trans_4d=p4d, cfg=xdx_cfg_st1)
                    s6 = apply_spell_reranker(s6, p_6rt, h3, X_df, c_6rt, p4d, fams[:6], cfg=rr_cfg)
                    return [s1, s2, s3, s4, s5, s6]

                cands3 = _make_cands(p3, c3, p4d3, f3, X_v3[T], p3_6rt, c3_6rt, p3_d60, c3_d60, p3_tv25, c3_tv25, loo_dec_hyb[T], fold_dec_std[T], lomo_rerank_cfgs[T])
                cands6 = _make_cands(p6, c6, p4d6, f6, X_v6[T], p6_6rt, c6_6rt, p6_d60, c6_d60, p6_tv25, c6_tv25, loo_dec_hyb[T], fold_dec_std[T], lomo_rerank_cfgs[T])
                mbr_w = [0.26, 0.20, 0.18, 0.16, 0.10, 0.10]
                pr3 = mbr_decode_sequences(cands3, h3, weights=mbr_w, include_vote_consensus=True, base_proba=p3, base_p_change=c3, dec_params=loo_dec_hyb[T])
                pr6 = mbr_decode_sequences(cands6, h3, weights=mbr_w, include_vote_consensus=True, base_proba=p6, base_p_change=c6, dec_params=loo_dec_hyb[T])
            else:
                raise ValueError(f"Unknown mode_key: {mode_key}")

            oof_preds_v3[T] = pr3
            oof_preds_v6[T] = pr6
            j1_v3_f1.append(float(metrics.score(sportec[T].labels, pr3, h3)["spell_f1_mean"]))
            j1_v3_acc.append(frame_acc(sportec[T].labels, pr3))
            j1_v3_xdx += count_interior_xdx(pr3, h3, X_v3[T])
            j1_v6_f1.append(float(metrics.score(sportec[T].labels, pr6, h3)["spell_f1_mean"]))
            j1_v6_acc.append(frame_acc(sportec[T].labels, pr6))
            j1_v6_xdx += count_interior_xdx(pr6, h3, X_v6[T])

        # --- 2. J2 (Metrica Sample 3 Games) ---
        s_raw_f1, s_nod_f1, s_xdx = [], [], 0
        sample_preds = {}
        for gid in GAMES:
            ms = sample_matches[gid]
            hs = ms.half
            dh = sample_h[gid]
            dpl = pe_plus["sample"][gid]
            dv6 = sample_v6[gid]
            if mode_key == "v6_champ":
                ps, cs = moe_combine(
                    dh["g_sym"], dh["tr_sym"], dpl["g_plus_sym"], dpl["tr_plus_sym"], dv6["g_v6"], dv6["tr_v6"],
                    dh["t_sym"], dh["tc_sym"], dh["b_sym"], dh["bc_sym"],
                    w_pl=0.5, w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], w_gbm=0.55, w_c=0.50, mode="linear",
                )
                prs = decode_v4(ps, hs, dec_v4, p_change=cs)
            elif mode_key == "w5_unif_tv_symdec":
                ps, cs, _, _ = st_unif["sample"][gid]
                prs = decode_v4(ps, hs, dec_v5_hyb, p_change=cs)
            elif mode_key == "exp09_6fam_decoupled":
                ps, cs, _, _ = st_6fam_rt["sample"][gid]
                prs = decode_v4(ps, hs, dec_v5_hyb, p_change=cs)
            elif mode_key == "exp09_decoupled":
                ps, cs, _, _ = st_rt_raw["sample"][gid]
                prs = decode_v4(ps, hs, dec_v5_hyb, p_change=cs)
            elif mode_key == "exp09_calib":
                ps, cs, _, _ = st_rt_cal["sample"][gid]
                prs = decode_v4(ps, hs, dec_v5_hyb, p_change=cs)
            elif mode_key == "exp09_xdx_fixa":
                ps, cs, p4ds, _ = st_rt_cal["sample"][gid]
                prs = apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1)
            elif mode_key == "exp08_phys_rerank":
                ps, cs, p4ds, fs = st_rt_cal["sample"][gid]
                prs = apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1)
                prs = apply_spell_reranker(prs, ps, hs, X_sample[gid], cs, p4ds, fs, cfg=phys_rerank_cfg)
            elif mode_key == "exp08_lomo_rerank":
                ps, cs, p4ds, fs = st_rt_cal["sample"][gid]
                prs = apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1)
                prs = apply_spell_reranker(prs, ps, hs, X_sample[gid], cs, p4ds, fs, cfg=full_rerank_cfg)
            elif mode_key == "exp10_4dir_snap":
                ps, cs, p4ds, fs = st_rt_cal["sample"][gid]
                prs = decode_active_4dir(ps, hs, dec_v5_hyb, p_change=cs, p_trans_4d=p4ds, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10)
                prs = apply_xdx_arbitrator(prs, ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1)
                prs = apply_spell_reranker(prs, ps, hs, X_sample[gid], cs, p4ds, fs, cfg=full_rerank_cfg)
                prs = snap_boundaries_to_impulse(prs, hs, X_sample[gid], cs, p4ds, cfg=snap_cfg)
            elif mode_key == "exp10_mbr_grand":
                ps, cs, p4ds, fs = st_rt_cal["sample"][gid]
                ps_6rt, cs_6rt, _, _ = st_6fam_rt["sample"][gid]
                ps_d60, cs_d60, _, _ = st_rt_d60["sample"][gid]
                ps_tv25, cs_tv25, _, _ = st_rt_tv25["sample"][gid]
                s1 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps, hs, X_sample[gid], cs, p4ds, fs, cfg=full_rerank_cfg)
                s2 = snap_boundaries_to_impulse(apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps, hs, X_sample[gid], cs, p4ds, fs, cfg=phys_rerank_cfg), hs, X_sample[gid], cs, p4ds, cfg=snap_cfg)
                s3 = apply_spell_reranker(apply_xdx_arbitrator(decode_active_4dir(ps, hs, dec_v5_hyb, p_change=cs, p_trans_4d=p4ds, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps, hs, X_sample[gid], cs, p4ds, fs, cfg=full_rerank_cfg)
                s4 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps_d60, hs, dec_v5_hyb, p_change=cs_d60), ps_d60, hs, X_sample[gid], p_change=cs_d60, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps_d60, hs, X_sample[gid], cs_d60, p4ds, fs, cfg=full_rerank_cfg)
                s5 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps_tv25, hs, dec_v4, p_change=cs_tv25), ps_tv25, hs, X_sample[gid], p_change=cs_tv25, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps_tv25, hs, X_sample[gid], cs_tv25, p4ds, fs, cfg=full_rerank_cfg)
                s6 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps_6rt, hs, dec_v5_hyb, p_change=cs_6rt), ps_6rt, hs, X_sample[gid], p_change=cs_6rt, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps_6rt, hs, X_sample[gid], cs_6rt, p4ds, fs[:6], cfg=full_rerank_cfg)
                mbr_w = [0.26, 0.20, 0.18, 0.16, 0.10, 0.10]
                prs = mbr_decode_sequences([s1, s2, s3, s4, s5, s6], hs, weights=mbr_w, include_vote_consensus=True, base_proba=ps, base_p_change=cs, dec_params=dec_v5_hyb)
            sample_preds[gid] = prs
            s_raw_f1.append(float(metrics.score(ms.labels, prs, hs)["spell_f1_mean"]))
            s_nod_f1.append(float(metrics.score(without_dead_margin(ms.labels, hs), prs, hs)["spell_f1_mean"]))
            s_xdx += count_interior_xdx(prs, hs, X_sample[gid])

        # --- 3. J3 (BAR-ATH Raw & Broadcast-Repaired) ---
        def _decode_atd(variant_key: str, X_atd_df: pd.DataFrame, gate_tup: tuple):
            if mode_key == "v6_champ":
                if variant_key == "atd_raw":
                    dpl_a = pe_plus["atd_raw"]
                    pa, ca = moe_combine(
                        dpl_a["g_sym"], dpl_a["tr_sym"], dpl_a["g_plus_sym"], dpl_a["tr_plus_sym"],
                        atd_raw_extra["g_v6"], atd_raw_extra["tr_v6"], dpl_a["t_sym"], dpl_a["tc_sym"],
                        atd_raw_extra["b_sym"], atd_raw_extra["bc_sym"],
                        w_pl=0.5, w_ghost=gate_tup[0], w_gap=gate_tup[1], w_gbm=0.55, w_c=0.50, mode="linear",
                    )
                else:
                    dpl_a = pe_plus["atd_rep"]
                    pa, ca = moe_combine(
                        atd_rep_h["g_sym"], atd_rep_h["tr_sym"], dpl_a["g_plus_sym"], dpl_a["tr_plus_sym"],
                        atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"], atd_rep_h["t_sym"], atd_rep_h["tc_sym"],
                        atd_rep_h["b_sym"], atd_rep_h["bc_sym"],
                        w_pl=0.5, w_ghost=gate_tup[0], w_gap=gate_tup[1], w_gbm=0.55, w_c=0.50, mode="linear",
                    )
                pra = decode_v4(pa, half_atd_dec, dec_v4, p_change=ca)
            elif mode_key == "w5_unif_tv_symdec":
                pa, ca, _, _ = st_unif[variant_key]
                pra = decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca)
            elif mode_key == "exp09_6fam_decoupled":
                pa, ca, _, _ = st_6fam_rt[variant_key]
                pra = decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca)
            elif mode_key == "exp09_decoupled":
                pa, ca, _, _ = st_rt_raw[variant_key]
                pra = decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca)
            elif mode_key == "exp09_calib":
                pa, ca, _, _ = st_rt_cal[variant_key]
                pra = decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca)
            elif mode_key == "exp09_xdx_fixa":
                pa, ca, p4da, _ = st_rt_cal[variant_key]
                pra = apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_atd_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1)
            elif mode_key == "exp08_phys_rerank":
                pa, ca, p4da, fa = st_rt_cal[variant_key]
                pra = apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_atd_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1)
                pra = apply_spell_reranker(pra, pa, half_atd_dec, X_atd_df, ca, p4da, fa, cfg=phys_rerank_cfg)
            elif mode_key == "exp08_lomo_rerank":
                pa, ca, p4da, fa = st_rt_cal[variant_key]
                pra = apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_atd_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1)
                pra = apply_spell_reranker(pra, pa, half_atd_dec, X_atd_df, ca, p4da, fa, cfg=full_rerank_cfg)
            elif mode_key == "exp10_4dir_snap":
                pa, ca, p4da, fa = st_rt_cal[variant_key]
                pra = decode_active_4dir(pa, half_atd_dec, dec_v5_hyb, p_change=ca, p_trans_4d=p4da, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10)
                pra = apply_xdx_arbitrator(pra, pa, half_atd_dec, X_atd_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1)
                pra = apply_spell_reranker(pra, pa, half_atd_dec, X_atd_df, ca, p4da, fa, cfg=full_rerank_cfg)
                pra = snap_boundaries_to_impulse(pra, half_atd_dec, X_atd_df, ca, p4da, cfg=snap_cfg)
            elif mode_key == "exp10_mbr_grand":
                pa, ca, p4da, fa = st_rt_cal[variant_key]
                pa_6rt, ca_6rt, _, _ = st_6fam_rt[variant_key]
                pa_d60, ca_d60, _, _ = st_rt_d60[variant_key]
                pa_tv25, ca_tv25, _, _ = st_rt_tv25[variant_key]
                s1 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_atd_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa, half_atd_dec, X_atd_df, ca, p4da, fa, cfg=full_rerank_cfg)
                s2 = snap_boundaries_to_impulse(apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_atd_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa, half_atd_dec, X_atd_df, ca, p4da, fa, cfg=phys_rerank_cfg), half_atd_dec, X_atd_df, ca, p4da, cfg=snap_cfg)
                s3 = apply_spell_reranker(apply_xdx_arbitrator(decode_active_4dir(pa, half_atd_dec, dec_v5_hyb, p_change=ca, p_trans_4d=p4da, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10), pa, half_atd_dec, X_atd_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa, half_atd_dec, X_atd_df, ca, p4da, fa, cfg=full_rerank_cfg)
                s4 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa_d60, half_atd_dec, dec_v5_hyb, p_change=ca_d60), pa_d60, half_atd_dec, X_atd_df, p_change=ca_d60, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa_d60, half_atd_dec, X_atd_df, ca_d60, p4da, fa, cfg=full_rerank_cfg)
                s5 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa_tv25, half_atd_dec, dec_v4, p_change=ca_tv25), pa_tv25, half_atd_dec, X_atd_df, p_change=ca_tv25, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa_tv25, half_atd_dec, X_atd_df, ca_tv25, p4da, fa, cfg=full_rerank_cfg)
                s6 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa_6rt, half_atd_dec, dec_v5_hyb, p_change=ca_6rt), pa_6rt, half_atd_dec, X_atd_df, p_change=ca_6rt, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa_6rt, half_atd_dec, X_atd_df, ca_6rt, p4da, fa[:6], cfg=full_rerank_cfg)
                mbr_w = [0.26, 0.20, 0.18, 0.16, 0.10, 0.10]
                pra = mbr_decode_sequences([s1, s2, s3, s4, s5, s6], half_atd_dec, weights=mbr_w, include_vote_consensus=True, base_proba=pa, base_p_change=ca, dec_params=dec_v5_hyb)
            pra[second_start:kick2] = DEAD
            pra[~atd.graded] = DEAD
            return pra

        pred_atd_raw = _decode_atd("atd_raw", X_atd_raw, gate_atd_raw)
        pred_atd_rep = _decode_atd("atd_rep", X_atd_rep, gate_atd_rep)

        pub_raw = metrics.score_public(pred_atd_raw, bootstrap_n=0)
        pl_raw = plausibility(pred_atd_raw, atd.half, atd.graded)
        xdx_raw = count_interior_xdx(pred_atd_raw, half_atd_dec, X_atd_raw)

        pub_rep = metrics.score_public(pred_atd_rep, bootstrap_n=0)
        pl_rep = plausibility(pred_atd_rep, atd.half, atd.graded)
        xdx_rep = count_interior_xdx(pred_atd_rep, half_atd_dec, X_atd_rep)

        sub_meta = None
        if sub_filename is not None:
            sub_meta = save_and_validate_submission(sub_dir / sub_filename, pred_atd_rep, atd, second_start, kick2)

        return {
            "name": name,
            "mode_key": mode_key,
            "j1_v3_f1": float(np.mean(j1_v3_f1)),
            "j1_v3_acc": float(np.mean(j1_v3_acc)),
            "j1_v3_xdx": int(j1_v3_xdx),
            "j1_v6_f1": float(np.mean(j1_v6_f1)),
            "j1_v6_acc": float(np.mean(j1_v6_acc)),
            "j1_v6_xdx": int(j1_v6_xdx),
            "j2_raw_f1": float(np.mean(s_raw_f1)),
            "j2_nod_f1": float(np.mean(s_nod_f1)),
            "j2_xdx": int(s_xdx),
            "j3_raw_f1": float(pub_raw["spell_f1_mean"]),
            "j3_raw_df1": float(pub_raw["alive_dead_f1"]),
            "j3_raw_xdx": int(xdx_raw),
            "j3_rep_f1": float(pub_rep["spell_f1_mean"]),
            "j3_rep_df1": float(pub_rep["alive_dead_f1"]),
            "j3_rep_pm": float(pl_rep["poss_per_min"]),
            "j3_rep_pctd": float(pl_rep["pct_D"]),
            "j3_rep_xdx": int(xdx_rep),
            "oof_v3": oof_preds_v3,
            "oof_v6": oof_preds_v6,
            "sample_preds": sample_preds,
            "pred_atd_raw": pred_atd_raw,
            "pred_atd_rep": pred_atd_rep,
            "sub_meta": sub_meta,
        }

    candidates = [
        ("0. v6_moe_ghost_lin Baseline (2-Tree + GhostGate + GapGate, dec_v4)", "v6_champ", None),
        ("1. Wave5 Uniform 4-Tree + 2-Seq + TV(0.20) + SymDec (`dec_v5_hyb`)", "w5_unif_tv_symdec", None),
        ("2. EXP-09A (6-Fam): Red-Team Decoupled 4-Tree + 2-Seq + TV(0.20) + SymDec", "exp09_6fam_decoupled", "submission_v7_w6_6fam_decoupled.csv"),
        ("3. EXP-09B (8-Fam): Decoupled 4-Tree + 4-Seq(Seq481Gate) + TV(0.20) + SymDec", "exp09_decoupled", "submission_v7_w6_decoupled.csv"),
        ("4. EXP-09C: 8-Fam Decoupled + Zero-Label Kinematic Blackout Calib", "exp09_calib", None),
        ("5. EXP-09 + EXP-03 (Red-Team Fix A `ball_x,ball_last_x` X-D-X Arbitrator)", "exp09_xdx_fixa", "submission_v7_w6_decoupled_xdx.csv"),
        ("6. EXP-08A: + Physics Spell Re-Ranker (`Y->X->Y` Merger, tau=0.38)", "exp08_phys_rerank", "submission_v7_w6_phys_rerank.csv"),
        ("7. EXP-08B: + Nested-LOMO Hybrid Spell Re-Ranker (`fit_spell_reranker_lomo`)", "exp08_lomo_rerank", "submission_v7_w6_lomo_rerank.csv"),
        ("8. EXP-10A/B: + Active 4-Dir Viterbi + Physical Boundary Impulse Snapping", "exp10_4dir_snap", "submission_v7_w6_4dir_snap.csv"),
        ("9. EXP-10C: + Direct `spell_scores` Multi-Family MBR Consensus (Grand W6)", "exp10_mbr_grand", "submission_v7_w6_mbr_grand_champion.csv"),
    ]

    header = (
        f"  {'Candidate':74s} | {'J1A: v3 (acc,xdx)':19s} | {'J1B: v6 (acc,xdx)':19s} | "
        f"{'J2: Samp (noD,xdx)':20s} | {'J3 Raw (D_f1,xdx)':19s} | {'J3 Rep (D_f1, p/m, %D, xdx)':30s}"
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    results = {}
    for cname, mkey, sub_fn in candidates:
        res = eval_pipeline(cname, mkey, sub_fn)
        results[mkey] = res
        print(
            f"  {cname:74s} | "
            f"{res['j1_v3_f1']:.4f} ({res['j1_v3_acc']:.3f},{res['j1_v3_xdx']:2d}) | "
            f"{res['j1_v6_f1']:.4f} ({res['j1_v6_acc']:.3f},{res['j1_v6_xdx']:2d}) | "
            f"{res['j2_raw_f1']:.4f} ({res['j2_nod_f1']:.4f},{res['j2_xdx']:2d}) | "
            f"{res['j3_raw_f1']:.4f} ({res['j3_raw_df1']:.3f},{res['j3_raw_xdx']:2d}) | "
            f"{res['j3_rep_f1']:.4f} ({res['j3_rep_df1']:.3f}, {res['j3_rep_pm']:4.2f}, {res['j3_rep_pctd']:4.1f}%, {res['j3_rep_xdx']:2d})",
            flush=True,
        )

    # Save Wave 6 OOF & J2/J3 artifacts
    with open(out_dir / "w6_results_summary.pkl", "wb") as f:
        pickle.dump(
            {
                "best_calib_cfg": best_calib_cfg,
                "lomo_rerank_cfgs": lomo_rerank_cfgs,
                "full_rerank_cfg": full_rerank_cfg,
                "results": {k: {rk: rv for rk, rv in v.items() if not rk.startswith("oof_")} for k, v in results.items()},
            },
            f,
        )
    for T in MATCH_IDS:
        np.savez_compressed(
            out_dir / f"w6_{T}.npz",
            proba_v3=st_rt_cal["v3"][T][0].astype(np.float32),
            pchange_v3=st_rt_cal["v3"][T][1].astype(np.float32),
            proba_v6=st_rt_cal["v6"][T][0].astype(np.float32),
            pchange_v6=st_rt_cal["v6"][T][1].astype(np.float32),
            pred_v3_rerank=results["exp08_lomo_rerank"]["oof_v3"][T],
            pred_v6_rerank=results["exp08_lomo_rerank"]["oof_v6"][T],
            pred_v3_mbr=results["exp10_mbr_grand"]["oof_v3"][T],
            pred_v6_mbr=results["exp10_mbr_grand"]["oof_v6"][T],
        )

    print("\n--- Step 6: Validated Wave 6 Submission Files Summary ---", flush=True)
    for mkey, res in results.items():
        if res["sub_meta"] is not None:
            sm = res["sub_meta"]
            print(
                f"  [Submission Validated] {mkey:22s} | rows={sm['rows']} | graded={sm['graded']} | "
                f"H={sm['pct_H']:.2f}% | A={sm['pct_A']:.2f}% | D={sm['pct_D']:.2f}%",
                flush=True,
            )

    print(f"\nCompleted Wave 6 evaluation in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
