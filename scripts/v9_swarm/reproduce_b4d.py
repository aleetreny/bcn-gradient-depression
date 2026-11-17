#!/usr/bin/env python3
"""Rule 3 Reproducibility of Wave 3 EXP-02 (`SymBoundary4DirGBM`) and Orphan J2/J3 Caches for v8.5.

Reconstructs from scratch:
  1. 7-Fold Outer LOMO `b4d_{T}.npz` (`p4d_v3` and `p4d_v6`) using `SymBoundary4DirGBM(subsample_every=1, seed=k)`
     on pre-subsampled (`[::5]`) 601-feature augmented `v1ctl_plus` matrices across `TRAIN_VARIANTS`
     (`s1`, `s2`, `static05_s1` of `degrade_v3`).
  2. Full 7-match `symb4d_full.pkl` using `SymBoundary4DirGBM(subsample_every=1, seed=42)`.
  3. `b4d_j23_preds.pkl` on Metrica Sample (games 1-3, default pitch 105.0x68.0) and BAR-ATH
     (`atd_raw` and broadcast-repaired `atd_rep`, pitch 105.3x68.0).
  4. Optional `--also-v6-j23` to rebuild `v6_j23_preds.pkl` from `exp_g_oof/{symgbm_v6_full.pkl,symtrans_v6_full.pkl}`.
  5. Bit-for-bit verification against `experiments/outputs/exp_w3_oof/` and full downstream `v8.5`
     unified synthesis verification (recomputing nested LOMO candidates from scratch and checking
     J1A/J1B/J2/J3 scores and final CSV SHA-256 hashes).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import glob
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "py_pkgs"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evaluation"))

import evaluation.possession_metrics as pm
from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_7_nested_full import TRAIN_VARIANTS, build_features
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.common import AWAY, DEAD, HOME
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.features.v1 import match_fingerprint
from src.features.v1ctl_plus import build_v1ctl_plus, mirror_v1ctl_plus_df
from src.models.boundary_4dir import (
    DEFAULT_4DIR_PARAMS,
    SymBoundary4DirGBM,
    augment_boundary_features,
    swap_trans_4d,
    targets_4dir,
)
from src.models.transition import change_features
from exp_b_symmetry_eval import predict_gbm_tta, predict_trans_tta
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match

SUBSAMPLE_B4D = 5
FULL_SEED = 42


def load_or_build_plus(prefix: str, half: np.ndarray, m_fallback=None) -> pd.DataFrame:
    """Load cached (features_v1 with change_features + control_v1 + plus_v1), or build if missing."""
    f1 = sorted(glob.glob(str(ROOT / f"local_data/cache/features_v1/{prefix}*.parquet")))
    f2 = sorted(glob.glob(str(ROOT / f"local_data/cache/control_v1/{prefix}*.parquet")))
    f3 = sorted(glob.glob(str(ROOT / f"local_data/cache/plus_v1/{prefix}*.parquet")))
    if f1 and f2 and f3:
        df1 = change_features(pd.read_parquet(f1[0]), half)
        return pd.concat([df1, pd.read_parquet(f2[0]), pd.read_parquet(f3[0])], axis=1)
    if m_fallback is None:
        raise FileNotFoundError(f"Missing cached v1ctl_plus parquet for prefix={prefix}")
    return build_v1ctl_plus(m_fallback, prefix)


def _fit_worker(task: dict) -> dict:
    """Worker process that trains a single SymBoundary4DirGBM with exact n_jobs=4 OpenMP threads."""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    t0 = time.time()
    kind = task["kind"]
    seed = task["seed"]
    Xo = task["Xo"]
    Xm = task["Xm"]
    yd = task["yd"]

    model = SymBoundary4DirGBM(
        params=dict(DEFAULT_4DIR_PARAMS),
        subsample_every=1,
        seed=seed,
    ).fit(Xo, Xm, yd)

    if kind == "fold":
        T = task["T"]
        p4d_v3 = model.predict_from_augmented(task["Xa_v3_orig"], task["Xa_v3_mir"])
        p4d_v6 = model.predict_from_augmented(task["Xa_v6_orig"], task["Xa_v6_mir"])
        return {
            "kind": "fold",
            "T": T,
            "seed": seed,
            "p4d_v3": p4d_v3,
            "p4d_v6": p4d_v6,
            "elapsed_s": time.time() - t0,
        }
    return {
        "kind": "full",
        "seed": seed,
        "model": model,
        "elapsed_s": time.time() - t0,
    }


def build_4dir_target_matrix(labels: np.ndarray, half: np.ndarray) -> np.ndarray:
    d = targets_4dir(labels, half)
    return np.column_stack([d["y_h2a"], d["y_a2h"], d["y_live2d"], d["y_d2live"]])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild_v6_j23_preds(out_path: Path, samples: dict, atd_rep) -> None:
    """Rebuild exp_g_oof/v6_j23_preds.pkl from saved symgbm_v6_full.pkl and symtrans_v6_full.pkl."""
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    gbm = pickle.loads((g_dir / "symgbm_v6_full.pkl").read_bytes())
    trans = pickle.loads((g_dir / "symtrans_v6_full.pkl").read_bytes())

    def _pred_v6(m, key: str) -> dict[str, np.ndarray]:
        X = build_features(m, key, "v1ctl")
        return {
            "g_v6": predict_gbm_tta(gbm, X, m.half, m.pitch),
            "tr_v6": predict_trans_tta(trans, X, m.half, m.pitch),
        }

    sample_v6 = {gid: _pred_v6(ms, f"sample_{gid}_raw") for gid, ms in samples.items()}
    atd_rep_v6 = _pred_v6(atd_rep, "atd_rep_True_True_None")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump((sample_v6, atd_rep_v6), f)


def verify_downstream_v85(out_dir: Path, b4d_v3_repro: dict, b4d_v6_repro: dict, b4d_j23_repro: dict) -> dict:
    """Run the downstream v8.5 unified synthesis using the reproduced b4d outputs and verify 100% match."""
    from src.decode.boundary_offset_aligner import SymBoundaryOffsetAligner, extract_boundary_candidates
    from src.decode.forensic_shields_v85 import (
        PostImpactTurnoverClassifier,
        PostImpactTurnoverConfig,
        ShieldedBoundaryOffsetAligner,
        ShieldedTouchTypeArbitrator,
        extract_post_impact_turnover_candidates,
    )
    from src.decode.hybrid_gate_v8 import blend_6fam_heads_v8, compute_hybrid_gate_v8, compute_w_clean_v8
    from src.decode.spell_reranker import apply_spell_reranker
    from src.decode.touch_type_arbitrator import (
        SymTouchTypeArbitrator,
        extract_mode1_candidates,
        extract_mode2_candidates,
        preclean_tracking_arrays,
    )
    from src.decode.transductive_calib import (
        DecoupledSynthConfig,
        calibrate_match_transductive,
        synthesize_head_decoupled_proba,
    )
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
    from src.eval import metrics
    from exp_v6_super_champion import compute_physical_gates, save_and_validate_submission
    from exp_w7a_player_set_net import load_cached_physv2_table
    from scripts.forensic_v7.run_unified_v8_forensic_champion import apply_vis_gated_dead_wedge

    t0 = time.time()
    OUT_BASE = ROOT / "experiments" / "outputs"
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

    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)
    gate_atd_rep = compute_physical_gates(X_atd_rep, atd.half)

    sym_dir = OUT_BASE / "exp_b_sym_oof"
    plus_dir = OUT_BASE / "exp_c_plus_oof"
    g_dir = OUT_BASE / "exp_g_oof"
    h_dir = OUT_BASE / "exp_h_oof"
    w4_dir = OUT_BASE / "exp_w4_oof"
    w5_dir = OUT_BASE / "exp_w5_tree_oof"
    w5b_dir = OUT_BASE / "exp_w5b_oof"
    w6_dir = OUT_BASE / "exp_w6_oof"
    w7a_dir = OUT_BASE / "exp_w7a_oof"
    w7b_dir = OUT_BASE / "exp_w7b_oof"

    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)

    oof = {}
    for T in MATCH_IDS:
        zb = np.load(sym_dir / f"sym_{T}.npz")
        zt = np.load(sym_dir / f"tcn_sym_{T}.npz")
        zp = np.load(plus_dir / f"plus_{T}.npz")
        zg = np.load(g_dir / f"ghost_sym_{T}.npz")
        zh = np.load(h_dir / f"bigru_sym_{T}.npz")
        zw4 = np.load(w4_dir / f"physv2_{T}.npz")
        zw5 = np.load(w5_dir / f"cb_xgb_{T}.npz")
        zw5b_tcn = np.load(w5b_dir / f"tcn_physv2_{T}.npz")
        zw5b_gru = np.load(w5b_dir / f"bigru_physv2_{T}.npz")
        zw7a = np.load(w7a_dir / f"pset_{T}.npz")
        zw7b_v6 = np.load(w7b_dir / f"v6_genuine_{T}.npz")
        oof[T] = {
            "g_v1": zb["symgbm_tta"], "tr_v1": zb["symtrans_tta"],
            "g_pl": zp["symgbm_plus_tta"], "tr_pl": zp["symtrans_plus_tta"],
            "g_pl_on_v6": zw7b_v6["symgbm_plus_tta_on_v6"], "tr_pl_on_v6": zw7b_v6["symtrans_plus_tta_on_v6"],
            "g_v6_on_v3": zg["g_v6_on_v3"], "tr_v6_on_v3": zg["tr_v6_on_v3"],
            "g_v3_on_v6": zg["g_v3_on_v6"], "tr_v3_on_v6": zg["tr_v3_on_v6"],
            "g_v6_on_v6": zg["g_v6_on_v6"], "tr_v6_on_v6": zg["tr_v6_on_v6"],
            "tcn": zt["symtcn_tta_proba"], "tcnc": zt["symtcn_tta_change"],
            "gru": zh["symbigru_tta_proba"], "gruc": zh["symbigru_tta_change"],
            "p4d_v3": b4d_v3_repro[T], "p4d_v6": b4d_v6_repro[T],
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
        sample_v6, atd_rep_v6 = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
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

    best_calib_cfg = w6_sum["best_calib_cfg"]
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

    s3_d, s3_ha, s3_c, ps_ha = 0.05, 0.30, 0.18, 0.08
    st = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}, "atd_rep": None}
    for T in MATCH_IDS:
        d = oof[T]
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
        p3 = calibrate_match_transductive(p3, h, X_v3[T], best_calib_cfg)
        f3 = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"], d["pset_v3"]]
        st["v3"][T] = (p3, c3, d["p4d_v3"], f3)

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
        p6s = calibrate_match_transductive(p6s, h, X_v6[T], best_calib_cfg)
        f6s = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tcn_p2_v6"], d["gru_p2_v6"], d["pset_v6"]]
        st["v6_std"][T] = (p6s, c6s, d["p4d_v6"], f6s)

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
        p6g = calibrate_match_transductive(p6g, h, X_v6[T], best_calib_cfg)
        f6g = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn_p2_v6"], d["gru_p2_v6"], d["g_v6_on_v6"], d["pset_v6"]]
        st["v6_gen"][T] = (p6g, c6g, d["p4d_v6"], f6g)

    for gid in GAMES:
        ms = sample_matches[gid]
        dh = sample_h[gid]
        dpl = pe_plus["sample"][gid]
        dv6 = sample_v6[gid]
        dp2 = pe_p2["sample"][gid]
        dw5 = pe_w5["sample"][gid]
        dw5b = pe_w5b["sample"][gid]
        dpset = pset_j23["sample"][gid]
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
        ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], best_calib_cfg)
        fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"]]
        st["sample"][gid] = (ps, cs, b4d_j23_repro["sample"][gid], fs)

    dpl_a = pe_plus["atd_rep"]
    dp2_a = pe_p2["atd_rep"]
    dw5_a = pe_w5["atd_rep"]
    dw5b_a = pe_w5b["atd_rep"]
    dpset_a = pset_j23["atd_rep"]
    a_a = compute_hybrid_gate_v8(X_atd_rep, atd.half, gate_atd_rep[0])
    wc_a = compute_w_clean_v8(X_atd_rep, atd.half, gate_atd_rep[0])
    pa, ca = synthesize_head_decoupled_proba(
        dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"],
        atd_rep_h["t_sym"], atd_rep_h["b_sym"], dpl_a["tr_plus_sym"], dp2_a["tr_physv2_sym"], atd_rep_h["tc_sym"], atd_rep_h["bc_sym"],
        atd.half, atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"],
        w_ghost=gate_atd_rep[0], w_gap=gate_atd_rep[1], cfg=scfg_main,
        tcn_physv2_proba=dw5b_a["tcn_physv2_sym"], bigru_physv2_proba=dw5b_a["bigru_physv2_sym"],
        tcn_physv2_change=dw5b_a["tcnc_physv2_sym"], bigru_physv2_change=dw5b_a["bigruc_physv2_sym"],
        alpha_p2=a_a,
    )
    pa, ca = blend_6fam_heads_v8(
        pa, ca, atd_rep_h["t_sym"], atd_rep_h["b_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"],
        atd_rep_h["tc_sym"], atd_rep_h["bc_sym"], dw5b_a["bigruc_physv2_sym"], dpset_a["pset_change"],
        a_a, wc_a, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True,
    )
    pa = calibrate_match_transductive(pa, half_atd_dec, X_atd_rep, best_calib_cfg)
    fa = [dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"], atd_rep_h["t_sym"], atd_rep_h["b_sym"], dw5b_a["tcn_physv2_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"]]
    st["atd_rep"] = (pa, ca, b4d_j23_repro["atd_rep"], fa)

    # Extract candidates from scratch using reproduced p4d
    cands = {}
    for tr, X_dict in [("v3", X_v3), ("v6_std", X_v6), ("v6_gen", X_v6)]:
        preds_rr, m1_d, m2_d, bdy_d, pit_d = {}, {}, {}, {}, {}
        for m in matches:
            p, c, p4d, fams = st[tr][m]
            hl = sportec[m].half
            gt = sportec[m].labels
            X_df = X_dict[m]
            p_xdx = apply_xdx_arbitrator(
                decode_v4(p, hl, loo_dec_hyb[m], p_change=c),
                p, hl, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1,
            )
            p_rr = apply_spell_reranker(p_xdx, p, hl, X_df, c, p4d, fams, cfg=lomo_rerank_cfgs[m])
            preds_rr[m] = p_rr
            cxv = preclean_tracking_arrays(X_df, len(p_rr))
            f64 = [f.astype(np.float64) for f in fams]
            m1_d[m] = extract_mode1_candidates(
                p_rr, p.astype(np.float64), hl, cxv, c.astype(np.float64), p4d.astype(np.float64), f64, gt=gt, match_id=m
            )
            m2_d[m] = extract_mode2_candidates(
                p_rr, p.astype(np.float64), hl, cxv, c.astype(np.float64), p4d.astype(np.float64), f64, gt=gt, match_id=m
            )
            bdy_d[m] = extract_boundary_candidates(
                p_rr, p.astype(np.float64), hl, X_df,
                p_change=c.astype(np.float64), p_trans_4d=p4d.astype(np.float64),
                family_probas=f64, labels=gt,
            )
            pit_d[m] = extract_post_impact_turnover_candidates(
                p_rr, p.astype(np.float64), hl, cxv, c.astype(np.float64), p4d.astype(np.float64), f64, gt=gt, match_id=m
            )
        cands[tr] = (preds_rr, m1_d, m2_d, bdy_d, pit_d)

    _, m1_v3, m2_v3, bdy_v3, pit_v3 = cands["v3"]
    arb_full = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_v3, m2_v3, matches)
    aln_full = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_v3)
    pit_full = PostImpactTurnoverClassifier().fit_from_candidates(pit_v3, matches)

    def _eval_variant(use_pit_domino: bool) -> tuple[dict, np.ndarray]:
        v3_f1, v6s_f1, v6g_f1 = [], [], []
        for m_te in matches:
            tr_ms = [m for m in matches if m != m_te]
            hl = sportec[m_te].half
            gt = sportec[m_te].labels
            for tr, X_df, f1_l in [("v3", X_v3[m_te], v3_f1), ("v6_std", X_v6[m_te], v6s_f1), ("v6_gen", X_v6[m_te], v6g_f1)]:
                preds_rr, m1_d, m2_d, bdy_d, pit_d = cands[tr]
                pr_base = preds_rr[m_te].copy()
                p_raw, c_raw, p4d_raw, fams_raw = st[tr][m_te]
                p = p_raw.astype(np.float64)
                c = c_raw.astype(np.float64)
                p4d = p4d_raw.astype(np.float64)
                f64 = [f.astype(np.float64) for f in fams_raw]
                pr = pr_base.copy()
                arb = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, tr_ms)
                pr = arb.apply(pr, p, hl, X_df, c, p4d, fams=f64, m1_cands=m1_d[m_te], m2_cands=m2_d[m_te])
                if tr != "v6_gen":
                    pr = apply_vis_gated_dead_wedge(pr, p, hl, X_df, c, p4d, vis_thresh=0.85)
                aln = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=False).fit({m: bdy_d[m] for m in tr_ms})
                pr = aln.apply(
                    pr, p, hl, X_df, p_change=c, p_trans_4d=p4d, family_probas=f64,
                    base_pred=pr_base, precomputed_cands=bdy_d[m_te],
                )
                if use_pit_domino and tr == "v3":
                    pit = PostImpactTurnoverClassifier(PostImpactTurnoverConfig(enable_missing_player_domino=False)).fit_from_candidates(pit_d, tr_ms)
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
            pr = arb_full.apply(pr_base.copy(), p, ms.half, X_sample[gid], c, p4d, fams=f64)
            pr = apply_vis_gated_dead_wedge(pr, p, ms.half, X_sample[gid], c, p4d, vis_thresh=0.85)
            pr = aln_full.apply(pr, p, ms.half, X_sample[gid], p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base)
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
        pr_a = arb_full.apply(pr_base_a.copy(), p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)
        pr_a = apply_vis_gated_dead_wedge(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, vis_thresh=0.85)
        pr_a = aln_full.apply(pr_a, p, half_atd_dec, X_atd_rep, p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base_a)
        if use_pit_domino:
            pr_a = pit_full.apply(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)
        pr_a[second_start:kick2] = DEAD
        pr_a[~atd.graded] = DEAD
        sc_rep = metrics.score_public(pr_a, bootstrap_n=0)
        return {
            "J1A_v3": float(np.mean(v3_f1)),
            "J1B_Std": float(np.mean(v6s_f1)),
            "J1B_Gen": float(np.mean(v6g_f1)),
            "J2_Raw": float(np.mean(j2_raw)),
            "J2_noD": float(np.mean(j2_nod)),
            "J3_Rep_F1": float(sc_rep["spell_f1_mean"]),
        }, pr_a

    rec_strict, pr_strict = _eval_variant(use_pit_domino=False)
    rec_grand, pr_grand = _eval_variant(use_pit_domino=True)

    sub_strict_path = out_dir / "submission_v85_forensic_strict_cv.csv"
    sub_grand_path = out_dir / "submission_v85_forensic_grand_champion.csv"
    save_and_validate_submission(sub_strict_path, pr_strict, atd, second_start, kick2)
    save_and_validate_submission(sub_grand_path, pr_grand, atd, second_start, kick2)

    ref_strict_path = ROOT / "submissions" / "submission_v85_forensic_strict_cv.csv"
    ref_grand_path = ROOT / "submissions" / "submission_v85_forensic_grand_champion.csv"
    sha_strict_repro = sha256_file(sub_strict_path)
    sha_grand_repro = sha256_file(sub_grand_path)
    sha_strict_ref = sha256_file(ref_strict_path)
    sha_grand_ref = sha256_file(ref_grand_path)

    return {
        "elapsed_s": time.time() - t0,
        "strict_cv": {
            "scores": rec_strict,
            "sha256_repro": sha_strict_repro,
            "sha256_ref": sha_strict_ref,
            "exact_byte_match": sha_strict_repro == sha_strict_ref,
        },
        "grand_champion": {
            "scores": rec_grand,
            "sha256_repro": sha_grand_repro,
            "sha256_ref": sha_grand_ref,
            "exact_byte_match": sha_grand_repro == sha_grand_ref,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce SymBoundary4DirGBM LOMO folds, full model, and J2/J3 predictions.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments" / "outputs" / "v9_swarm" / "b4d_repro",
        help="Directory to write reproduced b4d_{T}.npz, symb4d_full.pkl, and b4d_j23_preds.pkl",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (each runs LightGBM with n_jobs=4)",
    )
    parser.add_argument(
        "--also-v6-j23",
        action="store_true",
        help="Also rebuild exp_g_oof/v6_j23_preds.pkl into --v6-j23-path",
    )
    parser.add_argument(
        "--v6-j23-path",
        type=Path,
        default=None,
        help="Output path for v6_j23_preds.pkl when --also-v6-j23 is set",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip comparison against existing exp_w3_oof and downstream v8.5 verification",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "experiments" / "outputs" / "v9_swarm" / "b4d_repro_report.json",
        help="Path to write JSON verification report",
    )
    args = parser.parse_args()

    t_start = time.time()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/4] Loading & Augmenting 601-Feature `v1ctl_plus` Matrices ===", flush=True)
    t_feat = time.time()
    sportec = load_sportec()
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")

    sub_orig, sub_mir, sub_yd = {}, {}, {}
    test_v3_aug, test_v6_aug = {}, {}

    for mid in MATCH_IDS:
        m = sportec[mid]
        yd_full = targets_4dir(m.labels, m.half)
        keep = np.flatnonzero(yd_full["y_h2a"] >= 0)[::SUBSAMPLE_B4D]
        sub_yd[mid] = {k: v[keep] for k, v in yd_full.items()}
        for key, over, seed in TRAIN_VARIANTS:
            c3 = replace(cfg_v3, **over) if over else cfg_v3
            prefix = f"sportec_{mid}_degrade_v3_{key}"
            X = load_or_build_plus(prefix, m.half, m_fallback=degrade(m, c3, seed))
            X_mir = mirror_v1ctl_plus_df(X, m.half, m.pitch)
            sub_orig[(mid, key)] = augment_boundary_features(X, m.half, m.pitch).iloc[keep]
            sub_mir[(mid, key)] = augment_boundary_features(X_mir, m.half, m.pitch).iloc[keep]

        X3_s0 = load_or_build_plus(f"sportec_{mid}_degrade_v3_s0", m.half, m_fallback=degrade(m, cfg_v3, 0))
        X3_s0_mir = mirror_v1ctl_plus_df(X3_s0, m.half, m.pitch)
        test_v3_aug[mid] = (
            augment_boundary_features(X3_s0, m.half, m.pitch),
            augment_boundary_features(X3_s0_mir, m.half, m.pitch),
        )

        X6_s0 = load_or_build_plus(f"sportec_{mid}_degrade_v6_s0", m.half, m_fallback=degrade(m, cfg_v6, 0))
        X6_s0_mir = mirror_v1ctl_plus_df(X6_s0, m.half, m.pitch)
        test_v6_aug[mid] = (
            augment_boundary_features(X6_s0, m.half, m.pitch),
            augment_boundary_features(X6_s0_mir, m.half, m.pitch),
        )

    print(f"  Prepared all 7 matches x 3 train variants + v3/v6 test matrices in {time.time() - t_feat:.1f}s", flush=True)

    print(f"\n=== [2/4] Training 7 LOMO Folds (seeds 0..6) + Full 7-Match Model (seed={FULL_SEED}) with {args.workers} workers ===", flush=True)
    tasks = []
    for k, T in enumerate(MATCH_IDS):
        train_ids = [m for m in MATCH_IDS if m != T]
        tasks.append({
            "kind": "fold",
            "T": T,
            "seed": k,
            "Xo": [sub_orig[(m, key)] for m in train_ids for key, _, _ in TRAIN_VARIANTS],
            "Xm": [sub_mir[(m, key)] for m in train_ids for key, _, _ in TRAIN_VARIANTS],
            "yd": [sub_yd[m] for m in train_ids for _ in TRAIN_VARIANTS],
            "Xa_v3_orig": test_v3_aug[T][0],
            "Xa_v3_mir": test_v3_aug[T][1],
            "Xa_v6_orig": test_v6_aug[T][0],
            "Xa_v6_mir": test_v6_aug[T][1],
        })

    tasks.append({
        "kind": "full",
        "seed": FULL_SEED,
        "Xo": [sub_orig[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS],
        "Xm": [sub_mir[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS],
        "yd": [sub_yd[m] for m in MATCH_IDS for _ in TRAIN_VARIANTS],
    })

    b4d_v3_repro, b4d_v6_repro = {}, {}
    symb4d_full: SymBoundary4DirGBM | None = None

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(_fit_worker, tasks):
            if res["kind"] == "fold":
                T = res["T"]
                b4d_v3_repro[T] = res["p4d_v3"]
                b4d_v6_repro[T] = res["p4d_v6"]
                npz_path = out_dir / f"b4d_{T}.npz"
                np.savez_compressed(npz_path, p4d_v3=res["p4d_v3"], p4d_v6=res["p4d_v6"])
                print(f"  [FOLD {T}] seed={res['seed']} trained & saved to {npz_path.name} in {res['elapsed_s']:.1f}s", flush=True)
            else:
                symb4d_full = res["model"]
                full_path = out_dir / "symb4d_full.pkl"
                with open(full_path, "wb") as f:
                    pickle.dump(symb4d_full, f)
                print(f"  [FULL 7-MATCH] seed={res['seed']} trained & saved to {full_path.name} in {res['elapsed_s']:.1f}s", flush=True)

    assert symb4d_full is not None

    print("\n=== [3/4] Generating Judge 2 (Metrica Sample) & Judge 3 (BAR-ATH) Predictions ===", flush=True)
    t_j23 = time.time()
    samples = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    Pb, Bb, team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    mapping, _ = enhanced_broadcast_alignment(atd, Pb, hom["valid"].to_numpy().astype(bool))
    atd_rep = repair_atd_match(atd, Pb, Bb, team_b, mapping, True, True, None)

    atd_raw_key = "atd_" + match_fingerprint(atd)
    atd_rep_key = "atd_rep_True_True_None_" + match_fingerprint(atd_rep)

    b4d_j23 = {
        "sample": {
            gid: symb4d_full.predict(load_or_build_plus(f"sample_{gid}_raw", ms.half, m_fallback=ms), ms.half)
            for gid, ms in samples.items()
        },
        "atd_raw": symb4d_full.predict(
            load_or_build_plus(atd_raw_key, atd.half, m_fallback=atd), atd.half, atd.pitch
        ),
        "atd_rep": symb4d_full.predict(
            load_or_build_plus(atd_rep_key, atd.half, m_fallback=atd_rep), atd.half, atd.pitch
        ),
    }
    j23_path = out_dir / "b4d_j23_preds.pkl"
    with open(j23_path, "wb") as f:
        pickle.dump(b4d_j23, f)
    print(f"  Saved {j23_path.name} in {time.time() - t_j23:.1f}s", flush=True)

    if args.also_v6_j23:
        v6_path = args.v6_j23_path or (out_dir / "v6_j23_preds.pkl")
        rebuild_v6_j23_preds(v6_path, samples, atd_rep)
        print(f"  Saved {v6_path}", flush=True)

    if args.skip_compare:
        print(f"\nCompleted in {(time.time() - t_start) / 60:.2f} min.", flush=True)
        return

    print("\n=== [4/4] Auditing Bit-Exactness vs `experiments/outputs/exp_w3_oof/` & Downstream v8.5 ===", flush=True)
    ref_dir = ROOT / "experiments" / "outputs" / "exp_w3_oof"
    with open(ref_dir / "symb4d_full.pkl", "rb") as f:
        ref_full = pickle.load(f)
    with open(ref_dir / "b4d_j23_preds.pkl", "rb") as f:
        ref_j23 = pickle.load(f)

    # 1. Check H <-> A equivariance
    X_eq = load_or_build_plus("sportec_J03WMX_degrade_v3_s0", sportec["J03WMX"].half).iloc[1000:4000].reset_index(drop=True)
    h_eq = sportec["J03WMX"].half[1000:4000]
    X_eq_mir = mirror_v1ctl_plus_df(X_eq, h_eq)
    p_eq_o = symb4d_full.predict(X_eq, h_eq)
    p_eq_m = symb4d_full.predict(X_eq_mir, h_eq)
    equiv_err = float(np.max(np.abs(p_eq_o - swap_trans_4d(p_eq_m))))

    # 2. Check fold-by-fold diffs and PR-AUC / ROC-AUC
    fold_diffs = {}
    pr_v3_repro, roc_v3_repro, pr_v6_repro, roc_v6_repro = [], [], [], []
    pr_v3_ref, roc_v3_ref, pr_v6_ref, roc_v6_ref = [], [], [], []
    turn1d_sp_repro, turn1d_sp_ref = [], []

    for T in MATCH_IDS:
        ref_z = np.load(ref_dir / f"b4d_{T}.npz")
        d3_max = float(np.max(np.abs(b4d_v3_repro[T] - ref_z["p4d_v3"])))
        d3_mean = float(np.mean(np.abs(b4d_v3_repro[T] - ref_z["p4d_v3"])))
        d6_max = float(np.max(np.abs(b4d_v6_repro[T] - ref_z["p4d_v6"])))
        d6_mean = float(np.mean(np.abs(b4d_v6_repro[T] - ref_z["p4d_v6"])))
        fold_diffs[T] = {
            "p4d_v3_max_abs_diff": d3_max,
            "p4d_v3_mean_abs_diff": d3_mean,
            "p4d_v6_max_abs_diff": d6_max,
            "p4d_v6_mean_abs_diff": d6_mean,
        }
        print(f"  Fold {T}: max|p4d_v3 - ref|={d3_max:.3e} | max|p4d_v6 - ref|={d6_max:.3e}", flush=True)

        m = sportec[T]
        y4 = build_4dir_target_matrix(m.labels, m.half)
        valid = (m.labels >= 0) & (m.half > 0)
        pr_v3_repro.append([float(average_precision_score(y4[valid, c], b4d_v3_repro[T][valid, c])) for c in range(4)])
        roc_v3_repro.append([float(roc_auc_score(y4[valid, c], b4d_v3_repro[T][valid, c])) for c in range(4)])
        pr_v6_repro.append([float(average_precision_score(y4[valid, c], b4d_v6_repro[T][valid, c])) for c in range(4)])
        roc_v6_repro.append([float(roc_auc_score(y4[valid, c], b4d_v6_repro[T][valid, c])) for c in range(4)])

        pr_v3_ref.append([float(average_precision_score(y4[valid, c], ref_z["p4d_v3"][valid, c])) for c in range(4)])
        roc_v3_ref.append([float(roc_auc_score(y4[valid, c], ref_z["p4d_v3"][valid, c])) for c in range(4)])
        pr_v6_ref.append([float(average_precision_score(y4[valid, c], ref_z["p4d_v6"][valid, c])) for c in range(4)])
        roc_v6_ref.append([float(roc_auc_score(y4[valid, c], ref_z["p4d_v6"][valid, c])) for c in range(4)])

        y_turn = np.maximum(y4[:, 0], y4[:, 1])
        turn_repro = np.clip(b4d_v3_repro[T][:, 0] + b4d_v3_repro[T][:, 1], 0.0, 1.0)
        turn_ref = np.clip(ref_z["p4d_v3"][:, 0] + ref_z["p4d_v3"][:, 1], 0.0, 1.0)
        turn1d_sp_repro.append(float(average_precision_score(y_turn[valid], turn_repro[valid])))
        turn1d_sp_ref.append(float(average_precision_score(y_turn[valid], turn_ref[valid])))

    j23_diffs = {
        "sample_1_max_abs_diff": float(np.max(np.abs(b4d_j23["sample"][1] - ref_j23["sample"][1]))),
        "sample_2_max_abs_diff": float(np.max(np.abs(b4d_j23["sample"][2] - ref_j23["sample"][2]))),
        "sample_3_max_abs_diff": float(np.max(np.abs(b4d_j23["sample"][3] - ref_j23["sample"][3]))),
        "atd_raw_max_abs_diff": float(np.max(np.abs(b4d_j23["atd_raw"] - ref_j23["atd_raw"]))),
        "atd_rep_max_abs_diff": float(np.max(np.abs(b4d_j23["atd_rep"] - ref_j23["atd_rep"]))),
    }
    print(f"  b4d_j23_preds max abs diffs: {j23_diffs}", flush=True)

    print("\n  Running downstream v8.5 unified synthesis from scratch with reproduced b4d outputs...", flush=True)
    v85_verif = verify_downstream_v85(out_dir, b4d_v3_repro, b4d_v6_repro, b4d_j23)
    print(f"  [v8.5 strict_cv]      SHA256 match={v85_verif['strict_cv']['exact_byte_match']} ({v85_verif['strict_cv']['sha256_repro']})", flush=True)
    print(f"  [v8.5 grand_champion] SHA256 match={v85_verif['grand_champion']['exact_byte_match']} ({v85_verif['grand_champion']['sha256_repro']})", flush=True)

    ch_names = ["H_to_A", "A_to_H", "Live_to_D", "D_to_Live"]
    report = {
        "model_architecture": {
            "class": "src.models.boundary_4dir.SymBoundary4DirGBM",
            "params": symb4d_full.params,
            "subsample_every_in_fit": symb4d_full.subsample_every,
            "caller_pre_subsample_step": SUBSAMPLE_B4D,
            "lomo_fold_seeds": {T: k for k, T in enumerate(MATCH_IDS)},
            "full_model_seed": FULL_SEED,
            "num_features": len(symb4d_full.features_),
            "base_v1ctl_plus_features": 352,
            "dropped_asymmetric_centroid_x_H_cols": 7,
            "added_b4d_temporal_diff_and_extrema_cols": 256,
            "train_variants": [
                {"key": key, "overrides": over, "seed": seed, "degrade_config": "configs/degrade_v3.json"}
                for key, over, seed in TRAIN_VARIANTS
            ],
            "n_rows_full_X_sym": 1203192,
            "base_rates_repro": symb4d_full.base_rates_,
            "base_rates_ref": ref_full.base_rates_,
            "features_exact_match": symb4d_full.features_ == ref_full.features_,
            "params_exact_match": symb4d_full.params == ref_full.params,
        },
        "equivariance_check": {
            "max_abs_diff_p4d_orig_vs_swap_mir": equiv_err,
        },
        "lomo_fold_verification": {
            "per_fold_diffs": fold_diffs,
            "max_abs_diff_across_all_folds_v3": float(max(d["p4d_v3_max_abs_diff"] for d in fold_diffs.values())),
            "max_abs_diff_across_all_folds_v6": float(max(d["p4d_v6_max_abs_diff"] for d in fold_diffs.values())),
            "pr_auc_v3_mean_repro": {ch_names[c]: float(np.mean(np.array(pr_v3_repro)[:, c])) for c in range(4)},
            "pr_auc_v3_mean_ref": {ch_names[c]: float(np.mean(np.array(pr_v3_ref)[:, c])) for c in range(4)},
            "roc_auc_v3_mean_repro": {ch_names[c]: float(np.mean(np.array(roc_v3_repro)[:, c])) for c in range(4)},
            "pr_auc_v6_mean_repro": {ch_names[c]: float(np.mean(np.array(pr_v6_repro)[:, c])) for c in range(4)},
            "pr_auc_v6_mean_ref": {ch_names[c]: float(np.mean(np.array(pr_v6_ref)[:, c])) for c in range(4)},
            "roc_auc_v6_mean_repro": {ch_names[c]: float(np.mean(np.array(roc_v6_repro)[:, c])) for c in range(4)},
            "turnover_1d_pr_auc_v3_repro": float(np.mean(turn1d_sp_repro)),
            "turnover_1d_pr_auc_v3_ref": float(np.mean(turn1d_sp_ref)),
        },
        "j23_preds_verification": j23_diffs,
        "downstream_v85_verification": v85_verif,
        "total_elapsed_s": time.time() - t_start,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full verification report to {args.report_json} in {(time.time() - t_start) / 60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
