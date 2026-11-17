"""Wave 5 (`EXP-06`): 4th & 5th Model Families — `SymCatBoost_PhysV2_TTA` (Oblivious Trees) & `SymXGB_PhysV2_TTA`.

Performs:
1. Loads the precomputed 481-column `v1ctl_physics_v2` feature matrices from disk cache via `build_features_physv2`.
2. Runs 7-Fold Nested LOMO for `SymCatBoost_PhysV2_TTA` (`TwoHeadCatBoost` with Oblivious Trees) and
   `SymXGB_PhysV2_TTA` (`TwoHeadXGB` with Depth-Wise Histogram Trees) with exact H <-> A + 180-deg
   symmetric training and test-time augmentation (TTA) on both `degrade_v3` and `degrade_v6`.
3. Trains Full 7-Match `symcatboost_physv2_full.pkl` and `symxgb_physv2_full.pkl` and saves Judge 2
   (Metrica Sample 3 games) and Judge 3 (`BAR-ATH` raw & broadcast-repaired) predictions to
   `experiments/outputs/exp_w5_tree_oof/eval_preds_cb_xgb.pkl`.
4. Evaluates standalone `SymCatBoost_PhysV2_TTA`, standalone `SymXGB_PhysV2_TTA`, the 3-Tree-Family
   Super-Ensemble (`SymGBM + SymCatBoost + SymXGB`), and their 5-Family combination with `SymTCN_TTA`
   and `SymBiGRU_TTA` across all 4 Judges:
   - Judge 1A: 7-fold LOMO on Sportec `degrade_v3`
   - Judge 1B: 7-fold LOMO on Sportec `degrade_v6` (Ghost-Ball)
   - Judge 2:  Metrica Sample Data (3 games, standard and `noD`)
   - Judge 3:  BAR-ATH (`raw` and `repaired` public `spellF1`, `D_F1`, `poss/min`, `%D`)
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py_pkgs"))
import torch  # loads libomp.dylib
ctypes.CDLL(str(Path(torch.__file__).resolve().parent / "lib" / "libomp.dylib"))
import catboost
import xgboost

from dataclasses import replace
import json
import os
import pickle
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_7_nested_full import TRAIN_VARIANTS, build_features
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.common import DEAD
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from src.features.symmetry import swap_labels
from src.features.v1ctl_physics_v2 import mirror_v1ctl_physv2_df
from src.models.catboost_xgb import (
    CATBOOST_DEFAULT_PARAMS,
    XGB_DEFAULT_PARAMS,
    TwoHeadCatBoost,
    TwoHeadXGB,
    predict_model_physv2_tta,
)
from exp_b_symmetry_eval import blend_proba, predict_tcn_tta
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
from exp_w4_physics_v2_build import build_features_physv2

# 4 cached degrade_v3 variants per match (s0, s1, s2, static05_s1)
ALL_V3_TRAIN_VARIANTS = (("s0", None, 0),) + TRAIN_VARIANTS
SUBSAMPLE_EVERY = 22  # ~312k symmetrically paired frames per fold across 4 variants x 6 matches x 2 views


class _TeeLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def combine_tree_hybrid(
    p_gbm: np.ndarray,
    p_cb: np.ndarray,
    p_xgb: np.ndarray,
    weights=(0.45, 0.30, 0.25),
    w_d=None,
) -> np.ndarray:
    """Factorized two-head hybrid blend across the 3 tree families:
    linear blend on P(D) (using w_d if provided, else weights),
    log-odds (geometric) blend on conditional P(H | alive) using weights."""
    w1, w2, w3 = weights
    wd1, wd2, wd3 = w_d if w_d is not None else weights
    eps = 1e-6
    pD = wd1 * p_gbm[:, 2] + wd2 * p_cb[:, 2] + wd3 * p_xgb[:, 2]
    alive = np.clip(1.0 - pD, eps, 1.0)

    def _logit_h(p: np.ndarray) -> np.ndarray:
        r = np.clip(p[:, 0] / np.maximum(p[:, 0] + p[:, 1], eps), eps, 1.0 - eps)
        return np.log(r / (1.0 - r))

    logit_mix = w1 * _logit_h(p_gbm) + w2 * _logit_h(p_cb) + w3 * _logit_h(p_xgb)
    rH = 1.0 / (1.0 + np.exp(-np.clip(logit_mix, -30.0, 30.0)))
    out = np.empty_like(p_gbm, dtype=np.float32)
    out[:, 0] = alive * rH
    out[:, 1] = alive * (1.0 - rH)
    out[:, 2] = pD
    out /= np.maximum(out.sum(axis=1, keepdims=True), eps)
    return out


def combine_seq_tcn_bigru(tcn: np.ndarray, tcnc: np.ndarray, gru: np.ndarray, gruc: np.ndarray, w_gru_ha: float = 0.20):
    """Blend SymBiGRU_TTA into SymTCN_TTA's conditional H/(H+A) while keeping TCN's P(D) anchor."""
    if w_gru_ha <= 0.0:
        return tcn, tcnc
    eps = 1e-6
    pD = tcn[:, 2]
    alive = np.clip(1.0 - pD, eps, 1.0)
    r_tcn = tcn[:, 0] / np.maximum(tcn[:, 0] + tcn[:, 1], eps)
    r_gru = gru[:, 0] / np.maximum(gru[:, 0] + gru[:, 1], eps)
    rH = (1.0 - w_gru_ha) * r_tcn + w_gru_ha * r_gru
    pseq = np.empty_like(tcn, dtype=np.float32)
    pseq[:, 0] = alive * rH
    pseq[:, 1] = alive * (1.0 - rH)
    pseq[:, 2] = pD
    cseq = ((1.0 - 0.5 * w_gru_ha) * tcnc + (0.5 * w_gru_ha) * gruc).astype(np.float32)
    return pseq, cseq


def main() -> None:
    log_path = ROOT / "experiments" / "outputs" / "exp_w5_catboost_xgb.log"
    sys.stdout = _TeeLogger(log_path)

    t0 = time.time()
    print("=== Wave 5 (EXP-06): 4th & 5th Model Families (`SymCatBoost_PhysV2_TTA` & `SymXGB_PhysV2_TTA`) ===", flush=True)
    print(f"  catboost={catboost.__version__} | xgboost={xgboost.__version__} | OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    # 1. Load cached 481-column `v1ctl_physics_v2` matrices & pre-subsample symmetric training pairs
    print("\n--- Step 1: Loading Cached 481-Col `v1ctl_physics_v2` Matrices & Symmetric Pairs ---", flush=True)
    tb = time.time()
    sportec = load_all()
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")

    train_sub_X, train_sub_X_mir, train_sub_y, train_sub_y_mir = {}, {}, {}, {}
    test_X_v3, test_X_v3_mir, test_X_v6, test_X_v6_mir = {}, {}, {}, {}
    feature_names: list[str] | None = None

    for mid, m in sportec.items():
        y = m.labels
        y_sw = swap_labels(y)
        keep = np.flatnonzero(y >= 0)[::SUBSAMPLE_EVERY]
        for key, over, seed in ALL_V3_TRAIN_VARIANTS:
            c3 = replace(cfg_v3, **over) if over else cfg_v3
            m_deg = degrade(m, c3, seed)
            X_tr = build_features_physv2(m_deg, f"sportec_{mid}_degrade_v3_{key}")
            if feature_names is None:
                feature_names = list(X_tr.columns)
            X_mir = mirror_v1ctl_physv2_df(X_tr, m.half, m.pitch)
            train_sub_X[(mid, key)] = X_tr.iloc[keep].to_numpy(dtype=np.float32, copy=True)
            train_sub_X_mir[(mid, key)] = X_mir.iloc[keep].to_numpy(dtype=np.float32, copy=True)
            train_sub_y[(mid, key)] = y[keep]
            train_sub_y_mir[(mid, key)] = y_sw[keep]
            if key == "s0":
                test_X_v3[mid] = X_tr
                test_X_v3_mir[mid] = X_mir

        X6 = build_features_physv2(degrade(m, cfg_v6, 0), f"sportec_{mid}_degrade_v6_s0")
        test_X_v6[mid] = X6
        test_X_v6_mir[mid] = mirror_v1ctl_physv2_df(X6, m.half, m.pitch)

    sample_matches, sample_X, sample_X_mir = {}, {}, {}
    for gid in GAMES:
        ms = load_sample(gid)
        sample_matches[gid] = ms
        Xs = build_features_physv2(ms, f"sample_{gid}_raw")
        sample_X[gid] = Xs
        sample_X_mir[gid] = mirror_v1ctl_physv2_df(Xs, ms.half, ms.pitch)

    atd = load_atd()
    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)

    X_atd_raw = build_features_physv2(atd, "atd")
    X_atd_raw_mir = mirror_v1ctl_physv2_df(X_atd_raw, atd.half, atd.pitch)
    X_atd_rep = build_features_physv2(atd_rep, "atd_rep_True_True_None")
    X_atd_rep_mir = mirror_v1ctl_physv2_df(X_atd_rep, atd_rep.half, atd_rep.pitch)

    n_fold0 = sum(len(train_sub_y[(m, k)]) * 2 for m in MATCH_IDS[1:] for k, _, _ in ALL_V3_TRAIN_VARIANTS)
    print(
        f"  Loaded & prepared all 481-col matrices in {time.time()-tb:.1f}s | "
        f"Symmetrically paired frames per LOMO fold = {n_fold0:,} (subsample_every={SUBSAMPLE_EVERY})",
        flush=True,
    )

    # 2. 7-Fold Outer LOMO (`SymCatBoost_PhysV2_TTA` & `SymXGB_PhysV2_TTA`)
    print("\n--- Step 2: 7-Fold Outer LOMO (`SymCatBoost_PhysV2_TTA` & `SymXGB_PhysV2_TTA`) ---", flush=True)
    out_dir = ROOT / "experiments" / "outputs" / "exp_w5_tree_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_w5 = {}
    for k, T in enumerate(MATCH_IDS):
        p_path = out_dir / f"cb_xgb_{T}.npz"
        if p_path.exists():
            oof_w5[T] = dict(np.load(p_path))
            print(f"  Fold {k} ({T}) loaded from cache {p_path.name}", flush=True)
            continue

        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        Xs_orig = [train_sub_X[(mid, key)] for mid in train_ids for key, _, _ in ALL_V3_TRAIN_VARIANTS]
        ys_orig = [train_sub_y[(mid, key)] for mid in train_ids for key, _, _ in ALL_V3_TRAIN_VARIANTS]
        gs_orig = [i for i, mid in enumerate(train_ids) for _ in ALL_V3_TRAIN_VARIANTS]

        Xs_mir = [train_sub_X_mir[(mid, key)] for mid in train_ids for key, _, _ in ALL_V3_TRAIN_VARIANTS]
        ys_mir = [train_sub_y_mir[(mid, key)] for mid in train_ids for key, _, _ in ALL_V3_TRAIN_VARIANTS]
        gs_mir = [len(train_ids) + i for i, mid in enumerate(train_ids) for _ in ALL_V3_TRAIN_VARIANTS]

        Xs_sym = Xs_orig + Xs_mir
        ys_sym = ys_orig + ys_mir
        gs_sym = gs_orig + gs_mir

        t_cb = time.time()
        symcb = TwoHeadCatBoost(
            params=CATBOOST_DEFAULT_PARAMS,
            calibrate=False,
            subsample_every=1,
            seed=k,
            features=feature_names,
        ).fit(Xs_sym, ys_sym, gs_sym)
        cb_v3 = predict_model_physv2_tta(symcb, test_X_v3[T], sportec[T].half, sportec[T].pitch, X_mir=test_X_v3_mir[T])
        cb_v6 = predict_model_physv2_tta(symcb, test_X_v6[T], sportec[T].half, sportec[T].pitch, X_mir=test_X_v6_mir[T])
        dt_cb = time.time() - t_cb

        t_xgb = time.time()
        symxgb = TwoHeadXGB(
            params=XGB_DEFAULT_PARAMS,
            calibrate=False,
            subsample_every=1,
            seed=k,
            features=feature_names,
        ).fit(Xs_sym, ys_sym, gs_sym)
        xgb_v3 = predict_model_physv2_tta(symxgb, test_X_v3[T], sportec[T].half, sportec[T].pitch, X_mir=test_X_v3_mir[T])
        xgb_v6 = predict_model_physv2_tta(symxgb, test_X_v6[T], sportec[T].half, sportec[T].pitch, X_mir=test_X_v6_mir[T])
        dt_xgb = time.time() - t_xgb

        oof_w5[T] = {
            "symcatboost_physv2_tta": cb_v3,
            "symxgb_physv2_tta": xgb_v3,
            "symcatboost_physv2_tta_on_v6": cb_v6,
            "symxgb_physv2_tta_on_v6": xgb_v6,
        }
        np.savez_compressed(p_path, **oof_w5[T])
        print(
            f"  Fold {k} ({T}) trained & saved in {time.time()-tf:.1f}s (CatBoost={dt_cb:.1f}s, XGB={dt_xgb:.1f}s)",
            flush=True,
        )

    # 3. Train Full 7-Match `symcatboost_physv2_full.pkl` & `symxgb_physv2_full.pkl`
    print("\n--- Step 3: Training Full 7-Match `symcatboost_physv2_full.pkl` & `symxgb_physv2_full.pkl` ---", flush=True)
    cb_full_path = out_dir / "symcatboost_physv2_full.pkl"
    xgb_full_path = out_dir / "symxgb_physv2_full.pkl"
    if cb_full_path.exists() and xgb_full_path.exists():
        with open(cb_full_path, "rb") as f:
            symcb_full = pickle.load(f)
        with open(xgb_full_path, "rb") as f:
            symxgb_full = pickle.load(f)
        print("  Loaded symcatboost_physv2_full.pkl & symxgb_physv2_full.pkl from cache.", flush=True)
    else:
        tf_full = time.time()
        Xs_all = [train_sub_X[(mid, key)] for mid in MATCH_IDS for key, _, _ in ALL_V3_TRAIN_VARIANTS]
        ys_all = [train_sub_y[(mid, key)] for mid in MATCH_IDS for key, _, _ in ALL_V3_TRAIN_VARIANTS]
        gs_all = [i for i, mid in enumerate(MATCH_IDS) for _ in ALL_V3_TRAIN_VARIANTS]
        Xs_mir = [train_sub_X_mir[(mid, key)] for mid in MATCH_IDS for key, _, _ in ALL_V3_TRAIN_VARIANTS]
        ys_mir = [train_sub_y_mir[(mid, key)] for mid in MATCH_IDS for key, _, _ in ALL_V3_TRAIN_VARIANTS]
        gs_mir = [len(MATCH_IDS) + i for i, mid in enumerate(MATCH_IDS) for _ in ALL_V3_TRAIN_VARIANTS]

        symcb_full = TwoHeadCatBoost(
            params=CATBOOST_DEFAULT_PARAMS,
            calibrate=False,
            subsample_every=1,
            seed=0,
            features=feature_names,
        ).fit(Xs_all + Xs_mir, ys_all + ys_mir, gs_all + gs_mir)

        symxgb_full = TwoHeadXGB(
            params=XGB_DEFAULT_PARAMS,
            calibrate=False,
            subsample_every=1,
            seed=0,
            features=feature_names,
        ).fit(Xs_all + Xs_mir, ys_all + ys_mir, gs_all + gs_mir)

        with open(cb_full_path, "wb") as f:
            pickle.dump(symcb_full, f)
        with open(xgb_full_path, "wb") as f:
            pickle.dump(symxgb_full, f)
        print(
            f"  Trained & saved symcatboost_physv2_full.pkl & symxgb_physv2_full.pkl in {time.time()-tf_full:.1f}s",
            flush=True,
        )

    # Feature importance inspection for Oblivious Trees (CatBoost) & Depth-Wise Trees (XGBoost)
    cb_imp = symcb_full.importance(top=12)
    xgb_imp = symxgb_full.importance(top=12)
    print("\n  [Top 10 Features — SymCatBoost_PhysV2 (Oblivious Trees)]:", flush=True)
    for hname in ("alive", "ha"):
        top_str = ", ".join(f"{k}={v*100:.1f}%" for k, v in cb_imp[hname].head(8).items())
        print(f"    head_{hname:5s}: {top_str}", flush=True)
    print("  [Top 10 Features — SymXGB_PhysV2 (Depth-Wise Hist Trees)]:", flush=True)
    for hname in ("alive", "ha"):
        top_str = ", ".join(f"{k}={v*100:.1f}%" for k, v in xgb_imp[hname].head(8).items())
        print(f"    head_{hname:5s}: {top_str}", flush=True)

    # 4. Evaluate on Judge 1A (Sportec degrade_v3) & Judge 1B (Sportec degrade_v6)
    store_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    ref_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl"
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    plus_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    h_dir = ROOT / "experiments" / "outputs" / "exp_h_oof"
    w4_dir = ROOT / "experiments" / "outputs" / "exp_w4_oof"

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std, fold_dec_hyb, _, _ = pickle.load(f)
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        loo_dec_lin, loo_dec_hyb, dec_v5_lin, dec_v5_hyb = pickle.load(f)

    oof_all = {}
    for T in MATCH_IDS:
        g_orig = np.load(store_v3 / T / f"gbm_outer_{T}.npy")
        tr_orig = np.load(store_v3 / T / f"trans_outer_{T}.npy")
        z_orig = np.load(store_v3 / T / f"tcn_outer_{T}.npz")
        sb = np.load(sym_dir / f"sym_{T}.npz")
        st = np.load(sym_dir / f"tcn_sym_{T}.npz")
        sp = np.load(plus_dir / f"plus_{T}.npz")
        sg = np.load(g_dir / f"ghost_sym_{T}.npz")
        sh = np.load(h_dir / f"bigru_sym_{T}.npz")
        sw4 = np.load(w4_dir / f"physv2_{T}.npz")
        sw5 = oof_w5[T]

        oof_all[T] = {
            "g_orig": g_orig,
            "tr_orig": tr_orig,
            "t_orig": z_orig["proba"],
            "tc_orig": z_orig["change"],
            "g_sym": sb["symgbm_tta"],
            "tr_sym": sb["symtrans_tta"],
            "t_sym": st["symtcn_tta_proba"],
            "tc_sym": st["symtcn_tta_change"],
            "b_sym": sh["symbigru_tta_proba"],
            "bc_sym": sh["symbigru_tta_change"],
            "g_plus": sp["symgbm_plus_tta"],
            "tr_plus": sp["symtrans_plus_tta"],
            "g_p2": sw4["symgbm_physv2_tta"],
            "tr_p2": sw4["symtrans_physv2_tta"],
            "cb_p2": sw5["symcatboost_physv2_tta"],
            "xgb_p2": sw5["symxgb_physv2_tta"],
            "g_v3_on_v6": sg["g_v3_on_v6"],
            "tr_v3_on_v6": sg["tr_v3_on_v6"],
            "g_v6_on_v6": sg["g_v6_on_v6"],
            "tr_v6_on_v6": sg["tr_v6_on_v6"],
            "g_p2_on_v6": sw4["symgbm_physv2_tta_on_v6"],
            "tr_p2_on_v6": sw4["symtrans_physv2_tta_on_v6"],
            "cb_p2_on_v6": sw5["symcatboost_physv2_tta_on_v6"],
            "xgb_p2_on_v6": sw5["symxgb_physv2_tta_on_v6"],
        }

    # Diagnostic: Diversity & Pairwise Argmax Agreement / Probability Correlation across the 3 Tree Families
    agr_gbm_cb, agr_gbm_xgb, agr_cb_xgb = [], [], []
    for T in MATCH_IDS:
        valid = sportec[T].labels >= 0
        d = oof_all[T]
        y_gbm = np.argmax(d["g_p2"][valid], axis=1)
        y_cb = np.argmax(d["cb_p2"][valid], axis=1)
        y_xgb = np.argmax(d["xgb_p2"][valid], axis=1)
        agr_gbm_cb.append(float(np.mean(y_gbm == y_cb)))
        agr_gbm_xgb.append(float(np.mean(y_gbm == y_xgb)))
        agr_cb_xgb.append(float(np.mean(y_cb == y_xgb)))
    print(
        f"\n  [Tree Family Diversity on Sportec v3 LOMO (Frame Argmax Agreement)]:\n"
        f"    LightGBM vs CatBoost (Oblivious): {np.mean(agr_gbm_cb)*100:.2f}%\n"
        f"    LightGBM vs XGBoost  (Depth-Wise): {np.mean(agr_gbm_xgb)*100:.2f}%\n"
        f"    CatBoost vs XGBoost              : {np.mean(agr_cb_xgb)*100:.2f}%",
        flush=True,
    )

    print("\n--- JUDGE 1A & 1B: 7-Fold Nested LOMO Results on Sportec (`degrade_v3` & `degrade_v6`) ---", flush=True)
    print("  [A] Standalone Tree Families (no TCN, no M3 change head):", flush=True)
    for name, fn_v3, fn_v6 in [
        ("0a. v4 GBM_orig (v1ctl, 244 cols)",
         lambda d: d["g_orig"], lambda d: d["g_v3_on_v6"]),
        ("1a. SymGBM_PhysV2_TTA (LightGBM Leaf-Wise, 481 cols)",
         lambda d: d["g_p2"], lambda d: d["g_p2_on_v6"]),
        ("2a. SymCatBoost_PhysV2_TTA (Oblivious Trees, 481 cols)",
         lambda d: d["cb_p2"], lambda d: d["cb_p2_on_v6"]),
        ("3a. SymXGB_PhysV2_TTA (Depth-Wise Hist, 481 cols)",
         lambda d: d["xgb_p2"], lambda d: d["xgb_p2_on_v6"]),
        ("4a. 3-Tree Equal (1/3 GBM + 1/3 CatBoost + 1/3 XGB)",
         lambda d: (d["g_p2"] + d["cb_p2"] + d["xgb_p2"]) / 3.0,
         lambda d: (d["g_p2_on_v6"] + d["cb_p2_on_v6"] + d["xgb_p2_on_v6"]) / 3.0),
        ("5a. 3-Tree Weighted (0.45 GBM + 0.30 CatBoost + 0.25 XGB)",
         lambda d: 0.45 * d["g_p2"] + 0.30 * d["cb_p2"] + 0.25 * d["xgb_p2"],
         lambda d: 0.45 * d["g_p2_on_v6"] + 0.30 * d["cb_p2_on_v6"] + 0.25 * d["xgb_p2_on_v6"]),
        ("6a. 3-Tree Hybrid LogOdds (0.45 GBM + 0.30 CB + 0.25 XGB)",
         lambda d: combine_tree_hybrid(d["g_p2"], d["cb_p2"], d["xgb_p2"], (0.45, 0.30, 0.25)),
         lambda d: combine_tree_hybrid(d["g_p2_on_v6"], d["cb_p2_on_v6"], d["xgb_p2_on_v6"], (0.45, 0.30, 0.25))),
        ("7a. 3-Tree Decoupled (wD=0.75/0.10/0.15, wHA=0.40/0.30/0.30)",
         lambda d: combine_tree_hybrid(d["g_p2"], d["cb_p2"], d["xgb_p2"], (0.40, 0.30, 0.30), w_d=(0.75, 0.10, 0.15)),
         lambda d: combine_tree_hybrid(d["g_p2_on_v6"], d["cb_p2_on_v6"], d["xgb_p2_on_v6"], (0.40, 0.30, 0.30), w_d=(0.75, 0.10, 0.15))),
    ]:
        s_v3, s_v6, f_v3, f_v6 = [], [], [], []
        for T in MATCH_IDS:
            params_gbm = DecoderParams(**json.loads((ref_v3 / T / "params_gbm.json").read_text()))
            valid = sportec[T].labels >= 0
            p3 = fn_v3(oof_all[T])
            p6 = fn_v6(oof_all[T])
            pred3 = decode_v4(p3, sportec[T].half, params_gbm)
            pred6 = decode_v4(p6, sportec[T].half, params_gbm)
            s_v3.append(metrics.score(sportec[T].labels, pred3, sportec[T].half)["spell_f1_mean"])
            s_v6.append(metrics.score(sportec[T].labels, pred6, sportec[T].half)["spell_f1_mean"])
            f_v3.append(float((np.argmax(p3[valid], axis=1) == sportec[T].labels[valid]).mean()))
            f_v6.append(float((np.argmax(p6[valid], axis=1) == sportec[T].labels[valid]).mean()))
        print(
            f"    {name:56s}: v3_SpellF1={np.mean(s_v3):.4f} (Acc={np.mean(f_v3):.4f}) | "
            f"v6_SpellF1={np.mean(s_v6):.4f} (Acc={np.mean(f_v6):.4f})",
            flush=True,
        )

    print("\n  [B] Standalone Tree Families + SymTrans_PhysV2_TTA (no TCN):", flush=True)
    for name, fn_v3, fn_v6 in [
        ("1b. SymGBM_PhysV2_TTA + SymTrans_PhysV2_TTA",
         lambda d: d["g_p2"], lambda d: d["g_p2_on_v6"]),
        ("2b. SymCatBoost_PhysV2_TTA + SymTrans_PhysV2_TTA",
         lambda d: d["cb_p2"], lambda d: d["cb_p2_on_v6"]),
        ("3b. SymXGB_PhysV2_TTA + SymTrans_PhysV2_TTA",
         lambda d: d["xgb_p2"], lambda d: d["xgb_p2_on_v6"]),
        ("4b. 3-Tree Equal (GBM+CB+XGB) + SymTrans_PhysV2_TTA",
         lambda d: (d["g_p2"] + d["cb_p2"] + d["xgb_p2"]) / 3.0,
         lambda d: (d["g_p2_on_v6"] + d["cb_p2_on_v6"] + d["xgb_p2_on_v6"]) / 3.0),
        ("5b. 3-Tree Weighted (0.45/0.30/0.25) + SymTrans_PhysV2",
         lambda d: 0.45 * d["g_p2"] + 0.30 * d["cb_p2"] + 0.25 * d["xgb_p2"],
         lambda d: 0.45 * d["g_p2_on_v6"] + 0.30 * d["cb_p2_on_v6"] + 0.25 * d["xgb_p2_on_v6"]),
        ("6b. 3-Tree Hybrid LogOdds + SymTrans_PhysV2_TTA",
         lambda d: combine_tree_hybrid(d["g_p2"], d["cb_p2"], d["xgb_p2"], (0.45, 0.30, 0.25)),
         lambda d: combine_tree_hybrid(d["g_p2_on_v6"], d["cb_p2_on_v6"], d["xgb_p2_on_v6"], (0.45, 0.30, 0.25))),
        ("7b. 3-Tree Decoupled (wD=0.75/0.10/0.15, wHA=0.40/0.30/0.30) + SymTrans",
         lambda d: combine_tree_hybrid(d["g_p2"], d["cb_p2"], d["xgb_p2"], (0.40, 0.30, 0.30), w_d=(0.75, 0.10, 0.15)),
         lambda d: combine_tree_hybrid(d["g_p2_on_v6"], d["cb_p2_on_v6"], d["xgb_p2_on_v6"], (0.40, 0.30, 0.30), w_d=(0.75, 0.10, 0.15))),
    ]:
        s_v3, s_v6 = [], []
        for T in MATCH_IDS:
            d = oof_all[T]
            pred3 = decode_v4(fn_v3(d), sportec[T].half, fold_dec_std[T], p_change=d["tr_p2"])
            pred6 = decode_v4(fn_v6(d), sportec[T].half, fold_dec_std[T], p_change=d["tr_p2_on_v6"])
            s_v3.append(metrics.score(sportec[T].labels, pred3, sportec[T].half)["spell_f1_mean"])
            s_v6.append(metrics.score(sportec[T].labels, pred6, sportec[T].half)["spell_f1_mean"])
        print(
            f"    {name:56s}: v3_LOMO={np.mean(s_v3):.4f} (std={np.std(s_v3):.4f}) | v6_LOMO={np.mean(s_v6):.4f} (std={np.std(s_v6):.4f})",
            flush=True,
        )

    print("\n  [C] Multi-Family Ensembles with SymTCN_TTA & SymBiGRU_TTA (`degrade_v3` & `degrade_v6`):", flush=True)
    for name, fg_v3, ftr_v3, fg_v6, ftr_v6, w_gru, bmode, wg, dmode in [
        ("0c. v4_ens_mean Baseline (orig_GBM + orig_TCN)",
         lambda d: d["g_orig"], lambda d: d["tr_orig"], lambda d: d["g_v3_on_v6"], lambda d: d["tr_v3_on_v6"], 0.0, "linear", 0.50, "v4"),
        ("1c. SymGBM_PhysV2 + SymTCN_TTA (hyb 0.55)",
         lambda d: d["g_p2"], lambda d: d["tr_p2"], lambda d: d["g_p2_on_v6"], lambda d: d["tr_p2_on_v6"], 0.0, "hybrid", 0.55, "v4"),
        ("2c. SymCatBoost_PhysV2 + SymTCN_TTA (hyb 0.55)",
         lambda d: d["cb_p2"], lambda d: d["tr_p2"], lambda d: d["cb_p2_on_v6"], lambda d: d["tr_p2_on_v6"], 0.0, "hybrid", 0.55, "v4"),
        ("3c. SymXGB_PhysV2 + SymTCN_TTA (hyb 0.55)",
         lambda d: d["xgb_p2"], lambda d: d["tr_p2"], lambda d: d["xgb_p2_on_v6"], lambda d: d["tr_p2_on_v6"], 0.0, "hybrid", 0.55, "v4"),
        ("4c. 3-Tree Equal (GBM+CB+XGB) + SymTCN_TTA (lin 0.55)",
         lambda d: (d["g_p2"] + d["cb_p2"] + d["xgb_p2"]) / 3.0, lambda d: d["tr_p2"],
         lambda d: (d["g_p2_on_v6"] + d["cb_p2_on_v6"] + d["xgb_p2_on_v6"]) / 3.0, lambda d: d["tr_p2_on_v6"], 0.0, "linear", 0.55, "v4"),
        ("5c. 3-Tree Equal (GBM+CB+XGB) + SymTCN_TTA (hyb 0.55)",
         lambda d: (d["g_p2"] + d["cb_p2"] + d["xgb_p2"]) / 3.0, lambda d: d["tr_p2"],
         lambda d: (d["g_p2_on_v6"] + d["cb_p2_on_v6"] + d["xgb_p2_on_v6"]) / 3.0, lambda d: d["tr_p2_on_v6"], 0.0, "hybrid", 0.55, "v4"),
        ("6c. 3-Tree Weighted (0.45/0.30/0.25) + SymTCN_TTA (hyb 0.55)",
         lambda d: 0.45 * d["g_p2"] + 0.30 * d["cb_p2"] + 0.25 * d["xgb_p2"], lambda d: d["tr_p2"],
         lambda d: 0.45 * d["g_p2_on_v6"] + 0.30 * d["cb_p2_on_v6"] + 0.25 * d["xgb_p2_on_v6"], lambda d: d["tr_p2_on_v6"], 0.0, "hybrid", 0.55, "v4"),
        ("7c. 3-Tree Hybrid LogOdds + SymTCN_TTA (hyb 0.55)",
         lambda d: combine_tree_hybrid(d["g_p2"], d["cb_p2"], d["xgb_p2"], (0.45, 0.30, 0.25)), lambda d: d["tr_p2"],
         lambda d: combine_tree_hybrid(d["g_p2_on_v6"], d["cb_p2_on_v6"], d["xgb_p2_on_v6"], (0.45, 0.30, 0.25)), lambda d: d["tr_p2_on_v6"], 0.0, "hybrid", 0.55, "v4"),
        ("8c. 5-Family (3-Tree Equal + SymTCN + SymBiGRU_0.20, lin 0.55)",
         lambda d: (d["g_p2"] + d["cb_p2"] + d["xgb_p2"]) / 3.0, lambda d: d["tr_p2"],
         lambda d: (d["g_p2_on_v6"] + d["cb_p2_on_v6"] + d["xgb_p2_on_v6"]) / 3.0, lambda d: d["tr_p2_on_v6"], 0.20, "linear", 0.55, "v4"),
        ("9c. 5-Family (3-Tree Equal + SymTCN + SymBiGRU_0.20, hyb 0.55)",
         lambda d: (d["g_p2"] + d["cb_p2"] + d["xgb_p2"]) / 3.0, lambda d: d["tr_p2"],
         lambda d: (d["g_p2_on_v6"] + d["cb_p2_on_v6"] + d["xgb_p2_on_v6"]) / 3.0, lambda d: d["tr_p2_on_v6"], 0.20, "hybrid", 0.55, "v4"),
        ("10c. 5-Family (3-Tree Weighted + SymTCN + SymBiGRU_0.20, hyb 0.55)",
         lambda d: 0.45 * d["g_p2"] + 0.30 * d["cb_p2"] + 0.25 * d["xgb_p2"], lambda d: d["tr_p2"],
         lambda d: 0.45 * d["g_p2_on_v6"] + 0.30 * d["cb_p2_on_v6"] + 0.25 * d["xgb_p2_on_v6"], lambda d: d["tr_p2_on_v6"], 0.20, "hybrid", 0.55, "v4"),
        ("11c. 5-Family (3-Tree Decoupled + SymTCN + SymBiGRU_0.20, hyb 0.55)",
         lambda d: combine_tree_hybrid(d["g_p2"], d["cb_p2"], d["xgb_p2"], (0.40, 0.30, 0.30), w_d=(0.75, 0.10, 0.15)), lambda d: d["tr_p2"],
         lambda d: combine_tree_hybrid(d["g_p2_on_v6"], d["cb_p2_on_v6"], d["xgb_p2_on_v6"], (0.40, 0.30, 0.30), w_d=(0.75, 0.10, 0.15)), lambda d: d["tr_p2_on_v6"], 0.20, "hybrid", 0.55, "v4"),
        ("12c. 5-Family (0.35*Plus + 0.65*3Tree + SymTCN + SymBiGRU_0.20, dec_v4)",
         lambda d: 0.35 * d["g_plus"] + 0.65 * (0.45 * d["g_p2"] + 0.30 * d["cb_p2"] + 0.25 * d["xgb_p2"]),
         lambda d: 0.35 * d["tr_plus"] + 0.65 * d["tr_p2"],
         lambda d: 0.35 * d["g_v6_on_v6"] + 0.65 * (0.45 * d["g_p2_on_v6"] + 0.30 * d["cb_p2_on_v6"] + 0.25 * d["xgb_p2_on_v6"]),
         lambda d: 0.35 * d["tr_v6_on_v6"] + 0.65 * d["tr_p2_on_v6"], 0.20, "hybrid", 0.55, "v4"),
        ("13c. 5-Family (0.35*Plus + 0.65*3Tree + SymTCN + SymBiGRU_0.20, dec_v5)",
         lambda d: 0.35 * d["g_plus"] + 0.65 * ((d["g_p2"] + d["cb_p2"] + d["xgb_p2"]) / 3.0),
         lambda d: 0.50 * d["tr_plus"] + 0.50 * d["tr_p2"],
         lambda d: 0.35 * d["g_v6_on_v6"] + 0.65 * ((d["g_p2_on_v6"] + d["cb_p2_on_v6"] + d["xgb_p2_on_v6"]) / 3.0),
         lambda d: 0.50 * d["tr_v6_on_v6"] + 0.50 * d["tr_p2_on_v6"], 0.20, "hybrid", 0.55, "hyb"),
    ]:
        s_v3, s_v6 = [], []
        for T in MATCH_IDS:
            d = oof_all[T]
            dp = fold_dec_std[T] if dmode == "v4" else loo_dec_hyb[T]
            if name.startswith("0c."):
                pseq, cseq = d["t_orig"], d["tc_orig"]
            else:
                pseq, cseq = combine_seq_tcn_bigru(d["t_sym"], d["tc_sym"], d["b_sym"], d["bc_sym"], w_gru_ha=w_gru)

            p3 = blend_proba(fg_v3(d), pseq, w_gbm=wg, mode=bmode)
            c3 = 0.5 * ftr_v3(d) + 0.5 * cseq
            pred3 = decode_v4(p3, sportec[T].half, dp, p_change=c3)
            s_v3.append(metrics.score(sportec[T].labels, pred3, sportec[T].half)["spell_f1_mean"])

            p6 = blend_proba(fg_v6(d), pseq, w_gbm=wg, mode=bmode)
            c6 = 0.5 * ftr_v6(d) + 0.5 * cseq
            pred6 = decode_v4(p6, sportec[T].half, dp, p_change=c6)
            s_v6.append(metrics.score(sportec[T].labels, pred6, sportec[T].half)["spell_f1_mean"])
        print(f"    {name:64s}: v3_LOMO={np.mean(s_v3):.4f} | v6_LOMO={np.mean(s_v6):.4f}", flush=True)

    # 5. Evaluate on Judge 2 (Metrica Sample 3 Games) & Judge 3 (BAR-ATH Raw & Broadcast-Repaired)
    print("\n--- Step 4: Judge 2 (Metrica Sample 3 Games) & Judge 3 (BAR-ATH Raw & Repaired) ---", flush=True)
    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        plus_preds = pickle.load(f)
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f:
        w4_preds = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(h_dir / "symbigru_full.pkl", "rb") as f:
        symbigru_full = pickle.load(f)

    # Compute SymBiGRU_TTA on BAR-ATH Raw as well so both Raw and Repaired have all 5 model families
    b_raw_sym, bc_raw_sym = predict_tcn_tta(symbigru_full, X_atd_raw[symbigru_full.features_], atd.half, atd.pitch)

    eval_cache_path = out_dir / "eval_preds_cb_xgb.pkl"
    if eval_cache_path.exists():
        with open(eval_cache_path, "rb") as f:
            cb_xgb_preds = pickle.load(f)
        sample_w5 = cb_xgb_preds["sample"]
        atd_raw_w5 = cb_xgb_preds["atd_raw"]
        atd_rep_w5 = cb_xgb_preds["atd_rep"]
        print("  Loaded eval_preds_cb_xgb.pkl from cache.", flush=True)
    else:
        sample_w5 = {}
        for gid in GAMES:
            ms = sample_matches[gid]
            sample_w5[gid] = {
                "cb_physv2_sym": predict_model_physv2_tta(symcb_full, sample_X[gid], ms.half, ms.pitch, X_mir=sample_X_mir[gid]),
                "xgb_physv2_sym": predict_model_physv2_tta(symxgb_full, sample_X[gid], ms.half, ms.pitch, X_mir=sample_X_mir[gid]),
            }
        atd_raw_w5 = {
            "cb_physv2_sym": predict_model_physv2_tta(symcb_full, X_atd_raw, atd.half, atd.pitch, X_mir=X_atd_raw_mir),
            "xgb_physv2_sym": predict_model_physv2_tta(symxgb_full, X_atd_raw, atd.half, atd.pitch, X_mir=X_atd_raw_mir),
        }
        atd_rep_w5 = {
            "cb_physv2_sym": predict_model_physv2_tta(symcb_full, X_atd_rep, atd_rep.half, atd_rep.pitch, X_mir=X_atd_rep_mir),
            "xgb_physv2_sym": predict_model_physv2_tta(symxgb_full, X_atd_rep, atd_rep.half, atd_rep.pitch, X_mir=X_atd_rep_mir),
        }
        with open(eval_cache_path, "wb") as f:
            pickle.dump({"sample": sample_w5, "atd_raw": atd_raw_w5, "atd_rep": atd_rep_w5}, f)
        print("  Saved eval_preds_cb_xgb.pkl to experiments/outputs/exp_w5_tree_oof/.", flush=True)

    dec_gbm = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not k.startswith("_")})
    dec_ens = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})

    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)

    eval_configs = [
        ("A1. SymGBM_PhysV2_TTA (LightGBM Leaf-Wise) only",
         lambda d, p2, w5: p2["g_physv2_sym"], lambda d, p2, w5: None, 0.0, None, 1.0, dec_gbm),
        ("A2. SymCatBoost_PhysV2_TTA (Oblivious Trees) only",
         lambda d, p2, w5: w5["cb_physv2_sym"], lambda d, p2, w5: None, 0.0, None, 1.0, dec_gbm),
        ("A3. SymXGB_PhysV2_TTA (Depth-Wise Hist) only",
         lambda d, p2, w5: w5["xgb_physv2_sym"], lambda d, p2, w5: None, 0.0, None, 1.0, dec_gbm),
        ("A4. 3-Tree Equal (SymGBM + SymCatBoost + SymXGB) only",
         lambda d, p2, w5: (p2["g_physv2_sym"] + w5["cb_physv2_sym"] + w5["xgb_physv2_sym"]) / 3.0,
         lambda d, p2, w5: None, 0.0, None, 1.0, dec_gbm),
        ("A5. 3-Tree Weighted (0.45 GBM + 0.30 CB + 0.25 XGB) only",
         lambda d, p2, w5: 0.45 * p2["g_physv2_sym"] + 0.30 * w5["cb_physv2_sym"] + 0.25 * w5["xgb_physv2_sym"],
         lambda d, p2, w5: None, 0.0, None, 1.0, dec_gbm),
        ("A6. 3-Tree Equal + SymTrans_PhysV2_TTA (no TCN)",
         lambda d, p2, w5: (p2["g_physv2_sym"] + w5["cb_physv2_sym"] + w5["xgb_physv2_sym"]) / 3.0,
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, None, 1.0, dec_ens),
        ("A7. 3-Tree Decoupled (wD=0.75/0.10/0.15, wHA=0.40/0.30/0.30) + SymTrans",
         lambda d, p2, w5: combine_tree_hybrid(p2["g_physv2_sym"], w5["cb_physv2_sym"], w5["xgb_physv2_sym"], (0.40, 0.30, 0.30), w_d=(0.75, 0.10, 0.15)),
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, None, 1.0, dec_ens),
        ("B1. SymGBM_PhysV2_TTA + SymTCN_TTA (hyb 0.55)",
         lambda d, p2, w5: p2["g_physv2_sym"], lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, "hybrid", 0.55, dec_ens),
        ("B2. SymCatBoost_PhysV2_TTA + SymTCN_TTA (hyb 0.55)",
         lambda d, p2, w5: w5["cb_physv2_sym"], lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, "hybrid", 0.55, dec_ens),
        ("B3. SymXGB_PhysV2_TTA + SymTCN_TTA (hyb 0.55)",
         lambda d, p2, w5: w5["xgb_physv2_sym"], lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, "hybrid", 0.55, dec_ens),
        ("B4. 3-Tree Equal (GBM+CB+XGB) + SymTCN_TTA (lin 0.55)",
         lambda d, p2, w5: (p2["g_physv2_sym"] + w5["cb_physv2_sym"] + w5["xgb_physv2_sym"]) / 3.0,
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, "linear", 0.55, dec_ens),
        ("B5. 3-Tree Equal (GBM+CB+XGB) + SymTCN_TTA (hyb 0.55)",
         lambda d, p2, w5: (p2["g_physv2_sym"] + w5["cb_physv2_sym"] + w5["xgb_physv2_sym"]) / 3.0,
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, "hybrid", 0.55, dec_ens),
        ("B6. 3-Tree Weighted (0.45/0.30/0.25) + SymTCN_TTA (hyb 0.55)",
         lambda d, p2, w5: 0.45 * p2["g_physv2_sym"] + 0.30 * w5["cb_physv2_sym"] + 0.25 * w5["xgb_physv2_sym"],
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, "hybrid", 0.55, dec_ens),
        ("B7. 3-Tree Hybrid LogOdds + SymTCN_TTA (hyb 0.55)",
         lambda d, p2, w5: combine_tree_hybrid(p2["g_physv2_sym"], w5["cb_physv2_sym"], w5["xgb_physv2_sym"], (0.45, 0.30, 0.25)),
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.0, "hybrid", 0.55, dec_ens),
        ("C1. 5-Family (3-Tree Equal + SymTCN + SymBiGRU_0.20, lin 0.55)",
         lambda d, p2, w5: (p2["g_physv2_sym"] + w5["cb_physv2_sym"] + w5["xgb_physv2_sym"]) / 3.0,
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.20, "linear", 0.55, dec_ens),
        ("C2. 5-Family (3-Tree Equal + SymTCN + SymBiGRU_0.20, hyb 0.55)",
         lambda d, p2, w5: (p2["g_physv2_sym"] + w5["cb_physv2_sym"] + w5["xgb_physv2_sym"]) / 3.0,
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.20, "hybrid", 0.55, dec_ens),
        ("C3. 5-Family (3-Tree Weighted + SymTCN + SymBiGRU_0.20, hyb 0.55)",
         lambda d, p2, w5: 0.45 * p2["g_physv2_sym"] + 0.30 * w5["cb_physv2_sym"] + 0.25 * w5["xgb_physv2_sym"],
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.20, "hybrid", 0.55, dec_ens),
        ("C4. 5-Family (3-Tree Decoupled + SymTCN + SymBiGRU_0.20, hyb 0.55)",
         lambda d, p2, w5: combine_tree_hybrid(p2["g_physv2_sym"], w5["cb_physv2_sym"], w5["xgb_physv2_sym"], (0.40, 0.30, 0.30), w_d=(0.75, 0.10, 0.15)),
         lambda d, p2, w5: p2["tr_physv2_sym"], 0.20, "hybrid", 0.55, dec_ens),
        ("C5. 5-Family (0.35*Plus + 0.65*3Tree + SymTCN + SymBiGRU_0.20, dec_v4)",
         lambda d, p2, w5: 0.35 * d["g_plus_sym"] + 0.65 * (0.45 * p2["g_physv2_sym"] + 0.30 * w5["cb_physv2_sym"] + 0.25 * w5["xgb_physv2_sym"]),
         lambda d, p2, w5: 0.35 * d["tr_plus_sym"] + 0.65 * p2["tr_physv2_sym"], 0.20, "hybrid", 0.55, dec_ens),
        ("C6. 5-Family (0.35*Plus + 0.65*3Tree + SymTCN + SymBiGRU_0.20, dec_v5)",
         lambda d, p2, w5: 0.35 * d["g_plus_sym"] + 0.65 * ((p2["g_physv2_sym"] + w5["cb_physv2_sym"] + w5["xgb_physv2_sym"]) / 3.0),
         lambda d, p2, w5: 0.50 * d["tr_plus_sym"] + 0.50 * p2["tr_physv2_sym"], 0.20, "hybrid", 0.55, dec_v5_hyb),
    ]

    for vname, fg, ftr, w_gru, bmode, wg, dec in eval_configs:
        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            d = plus_preds["sample"][gid]
            p2 = w4_preds["sample"][gid]
            w5 = sample_w5[gid]
            dh = sample_h[gid]

            ptree = fg(d, p2, w5)
            tr = ftr(d, p2, w5)
            if wg == 1.0:
                p = ptree
                c = tr
            else:
                pseq, cseq = combine_seq_tcn_bigru(d["t_sym"], d["tc_sym"], dh["b_sym"], dh["bc_sym"], w_gru_ha=w_gru)
                p = blend_proba(ptree, pseq, w_gbm=wg, mode=bmode)
                c = None if tr is None else (0.5 * tr + 0.5 * cseq)
            pred = decode_v4(p, ms.half, dec, p_change=c)
            s_raw.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

        # BAR-ATH Raw
        d_raw = plus_preds["atd_raw"]
        p2_raw = w4_preds["atd_raw"]
        ptree_raw = fg(d_raw, p2_raw, atd_raw_w5)
        tr_r = ftr(d_raw, p2_raw, atd_raw_w5)
        if wg == 1.0:
            p_raw = ptree_raw
            c_raw = tr_r
        else:
            pseq_r, cseq_r = combine_seq_tcn_bigru(d_raw["t_sym"], d_raw["tc_sym"], b_raw_sym, bc_raw_sym, w_gru_ha=w_gru)
            p_raw = blend_proba(ptree_raw, pseq_r, w_gbm=wg, mode=bmode)
            c_raw = None if tr_r is None else (0.5 * tr_r + 0.5 * cseq_r)
        pred_raw = decode_v4(p_raw, half_dec, dec, p_change=c_raw)
        pred_raw[second_start:kick2] = DEAD
        pred_raw[~atd.graded] = DEAD
        pub_raw = metrics.score_public(pred_raw, bootstrap_n=100)
        pl_raw = plausibility(pred_raw, atd.half, atd.graded)
        pct_h_raw = 100.0 * float(np.mean(pred_raw[atd.graded & (pred_raw != DEAD)] == 0))

        # BAR-ATH Repaired
        d_rep = plus_preds["atd_rep"]
        p2_rep = w4_preds["atd_rep"]
        ptree_rep = fg(d_rep, p2_rep, atd_rep_w5)
        tr_p = ftr(d_rep, p2_rep, atd_rep_w5)
        if wg == 1.0:
            p_rep = ptree_rep
            c_rep = tr_p
        else:
            pseq_p, cseq_p = combine_seq_tcn_bigru(d_rep["t_sym"], d_rep["tc_sym"], atd_rep_h["b_sym"], atd_rep_h["bc_sym"], w_gru_ha=w_gru)
            p_rep = blend_proba(ptree_rep, pseq_p, w_gbm=wg, mode=bmode)
            c_rep = None if tr_p is None else (0.5 * tr_p + 0.5 * cseq_p)
        pred_rep = decode_v4(p_rep, half_dec, dec, p_change=c_rep)
        pred_rep[second_start:kick2] = DEAD
        pred_rep[~atd.graded] = DEAD
        pub_rep = metrics.score_public(pred_rep, bootstrap_n=100)
        pl_rep = plausibility(pred_rep, atd.half, atd.graded)
        pct_h_rep = 100.0 * float(np.mean(pred_rep[atd.graded & (pred_rep != DEAD)] == 0))

        print(
            f"  {vname:64s}\n"
            f"    -> Judge 2 (Sample): mean={np.mean(s_raw):.4f} (noD={np.mean(s_nod):.4f}) | games={[round(x, 4) for x in s_raw]}\n"
            f"    -> Judge 3 (BAR-ATH Raw): Pub={pub_raw['spell_f1_mean']:.4f} (D={pub_raw['per_class']['D']['f1']:.3f}) | pos/m={pl_raw['poss_per_min']:.2f} %D={pl_raw['pct_D']:.1f}% %H={pct_h_raw:.1f}%\n"
            f"    -> Judge 3 (BAR-ATH Rep): Pub={pub_rep['spell_f1_mean']:.4f} (D={pub_rep['per_class']['D']['f1']:.3f}) | pos/m={pl_rep['poss_per_min']:.2f} %D={pl_rep['pct_D']:.1f}% %H={pct_h_rep:.1f}%",
            flush=True,
        )

    print(f"\nWave 5 (EXP-06) completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
