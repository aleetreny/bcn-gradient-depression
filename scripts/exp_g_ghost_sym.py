"""EXP-G: Ghost-Incoherence `degrade_v6` Symmetric Models (`SymGBM_v6_TTA`, `SymTrans_v6_TTA`).

Motivation:
In BAR-ATH, 24.3% of slow balls have no player within 3m (`slow_ghost`) and 28.3% of kicks have no
player within 4m (`kick_ghost`). `configs/degrade_v6.json` simulates this exact ball-player
incoherence ("ghosts": `ghost_occlusion_rate=0.005`, `ghost_offset_rate=0.01`).

Evaluates:
1. 7-fold LOMO `SymGBM_v6_TTA` and `SymTrans_v6_TTA` trained on `degrade_v6` (`X_v6 + mirror(X_v6)`),
   plus full 7-match `symgbm_v6_full.pkl` and `symtrans_v6_full.pkl` saved to `experiments/outputs/exp_g_oof/`.
2. Judge 1 (7-fold LOMO on both `v3` test folds and `v6` test folds):
   - `SymGBM_v3_TTA`, `SymGBM_v6_TTA`, and `0.5 * SymGBM_v3_TTA + 0.5 * SymGBM_v6_TTA` (with and without `SymTCN_TTA`).
   - Frame-level accuracy specifically on ghost-incoherent frames vs non-ghost frames on `v6` test folds.
3. Judge 2 (Metrica Sample 3 games) & Judge 3 (BAR-ATH Broadcast-Repaired):
   - Multi-regime blend (`SymGBM_v3_TTA + SymGBM_v6_TTA + SymTCN_TTA`).
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
from src.features.symmetry import mirror_v1ctl_df, swap_labels
from src.models.gbm import DEFAULT_PARAMS as GBM_DEFAULT_PARAMS, TwoHeadGBM
from src.models.transition import PARAMS as TRANS_DEFAULT_PARAMS, TransitionGBM, change_targets
from exp_a_nested_eval import mbr_select
from exp_b_symmetry_eval import blend_proba, predict_gbm_tta, predict_tcn_tta, predict_trans_tta
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match

optuna.logging.set_verbosity(optuna.logging.WARNING)

FAST_GBM_PARAMS = dict(GBM_DEFAULT_PARAMS, n_jobs=6)
FAST_TRANS_PARAMS = dict(TRANS_DEFAULT_PARAMS, n_jobs=6)


def frame_acc(y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = y_true >= 0
    if mask is not None:
        valid = valid & mask
    if not np.any(valid):
        return float("nan")
    return float((np.argmax(proba[valid], axis=1) == y_true[valid]).mean())


def main():
    t0 = time.time()
    print("=== EXP-G: Ghost-Incoherence `degrade_v6` Symmetric Models ===", flush=True)
    sportec = load_all()
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")
    store_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    out_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  Building / loading degrade_v3 and degrade_v6 feature matrices...", flush=True)
    tf_feat = time.time()
    train_X_v3, train_X_v3_mir, test_X_v3 = {}, {}, {}
    train_X_v6, train_X_v6_mir, test_X_v6 = {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c3 = replace(cfg_v3, **over) if over else cfg_v3
            X3 = build_features(degrade(m, c3, seed), f"sportec_{mid}_degrade_v3_{key}", "v1ctl")
            train_X_v3[(mid, key)] = X3
            train_X_v3_mir[(mid, key)] = mirror_v1ctl_df(X3, m.half, m.pitch)

            c6 = replace(cfg_v6, **over) if over else cfg_v6
            X6 = build_features(degrade(m, c6, seed), f"sportec_{mid}_degrade_v6_{key}", "v1ctl")
            train_X_v6[(mid, key)] = X6
            train_X_v6_mir[(mid, key)] = mirror_v1ctl_df(X6, m.half, m.pitch)

        test_X_v3[mid] = build_features(degrade(m, cfg_v3, 0), f"sportec_{mid}_degrade_v3_s0", "v1ctl")
        test_X_v6[mid] = build_features(degrade(m, cfg_v6, 0), f"sportec_{mid}_degrade_v6_s0", "v1ctl")
    print(f"  Feature matrices ready in {time.time()-tf_feat:.1f}s", flush=True)

    # Train 7-fold LOMO SymGBM_v6 & SymTrans_v6 (and evaluate both v3 and v6 models on v3 and v6 test folds)
    oof_g = {}
    for k, T in enumerate(MATCH_IDS):
        f_path = out_dir / f"ghost_sym_{T}.npz"
        if f_path.exists():
            z = np.load(f_path)
            oof_g[T] = dict(z)
            print(f"  Fold {k} ({T}) loaded from cache {f_path.name}", flush=True)
            continue

        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        ys_std = [sportec[mid].labels for mid in train_ids for _ in TRAIN_VARIANTS]
        hs_std = [sportec[mid].half for mid in train_ids for _ in TRAIN_VARIANTS]
        gs_std = [i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]
        ys_sym = ys_std + [swap_labels(sportec[mid].labels) for mid in train_ids for _ in TRAIN_VARIANTS]
        gs_sym = gs_std + [len(train_ids) + i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]
        ct_std = [change_targets(y, h) for y, h in zip(ys_std, hs_std)]
        ct_sym = ct_std + ct_std

        # 1. SymGBM_v6 and SymTrans_v6
        Xs_v6_std = [train_X_v6[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        Xs_v6_sym = Xs_v6_std + [train_X_v6_mir[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        gbm_v6_sym = TwoHeadGBM(params=FAST_GBM_PARAMS, seed=k).fit(Xs_v6_sym, ys_sym, gs_sym)
        tr_v6_sym = TransitionGBM(params=FAST_TRANS_PARAMS, seed=k).fit(Xs_v6_sym, ct_sym)

        g_v6_on_v3 = predict_gbm_tta(gbm_v6_sym, test_X_v3[T], sportec[T].half, sportec[T].pitch)
        tr_v6_on_v3 = predict_trans_tta(tr_v6_sym, test_X_v3[T], sportec[T].half, sportec[T].pitch)
        g_v6_on_v6 = predict_gbm_tta(gbm_v6_sym, test_X_v6[T], sportec[T].half, sportec[T].pitch)
        tr_v6_on_v6 = predict_trans_tta(tr_v6_sym, test_X_v6[T], sportec[T].half, sportec[T].pitch)

        # 2. SymGBM_v3 and SymTrans_v3 evaluated on test_X_v6[T] (test_X_v3[T] already in sym_{T}.npz)
        Xs_v3_std = [train_X_v3[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        Xs_v3_sym = Xs_v3_std + [train_X_v3_mir[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        gbm_v3_sym = TwoHeadGBM(params=FAST_GBM_PARAMS, seed=k).fit(Xs_v3_sym, ys_sym, gs_sym)
        tr_v3_sym = TransitionGBM(params=FAST_TRANS_PARAMS, seed=k).fit(Xs_v3_sym, ct_sym)

        g_v3_on_v6 = predict_gbm_tta(gbm_v3_sym, test_X_v6[T], sportec[T].half, sportec[T].pitch)
        tr_v3_on_v6 = predict_trans_tta(tr_v3_sym, test_X_v6[T], sportec[T].half, sportec[T].pitch)

        sb_v3 = np.load(sym_dir / f"sym_{T}.npz")
        g_v3_on_v3 = sb_v3["symgbm_tta"]
        tr_v3_on_v3 = sb_v3["symtrans_tta"]

        oof_g[T] = {
            "g_v3_on_v3": g_v3_on_v3,
            "tr_v3_on_v3": tr_v3_on_v3,
            "g_v6_on_v3": g_v6_on_v3,
            "tr_v6_on_v3": tr_v6_on_v3,
            "g_v3_on_v6": g_v3_on_v6,
            "tr_v3_on_v6": tr_v3_on_v6,
            "g_v6_on_v6": g_v6_on_v6,
            "tr_v6_on_v6": tr_v6_on_v6,
        }
        np.savez_compressed(f_path, **oof_g[T])
        print(f"  Fold {k} ({T}) SymGBM_v6 & SymGBM_v3 evaluated in {time.time()-tf:.1f}s", flush=True)

    # Load SymTCN_TTA OOF predictions and inner-tuned fold decoders
    for T in MATCH_IDS:
        st = np.load(sym_dir / f"tcn_sym_{T}.npz")
        oof_g[T]["t_sym"] = st["symtcn_tta_proba"]
        oof_g[T]["tc_sym"] = st["symtcn_tta_change"]
        oof_g[T]["g_orig"] = np.load(store_v3 / T / f"gbm_outer_{T}.npy")
        oof_g[T]["tr_orig"] = np.load(store_v3 / T / f"trans_outer_{T}.npy")
        z_orig = np.load(store_v3 / T / f"tcn_outer_{T}.npz")
        oof_g[T]["t_orig"] = z_orig["proba"]
        oof_g[T]["tc_orig"] = z_orig["change"]

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std, fold_dec_hyb, fold_cands, fold_w = pickle.load(f)

    # Diagnostic: Frame accuracy on ghost-affected frames on v6 test folds
    print("\n--- Diagnostic: Frame Accuracy on Ghost-Affected vs Clean Frames (`degrade_v6` Test Folds) ---", flush=True)
    for mname, fk in [
        ("SymGBM_v3_TTA (trained without ghosts)", "g_v3_on_v6"),
        ("SymGBM_v6_TTA (trained with ghosts)", "g_v6_on_v6"),
        ("50/50 Multi-Regime (SymGBM_v3 + SymGBM_v6)", None),
    ]:
        acc_all, acc_ghost, acc_clean = [], [], []
        for T in MATCH_IDS:
            y_t = sportec[T].labels
            d = oof_g[T]
            p_t = 0.5 * (d["g_v3_on_v6"] + d["g_v6_on_v6"]) if fk is None else d[fk]
            diff_feat = np.abs(
                np.nan_to_num(test_X_v6[T].to_numpy(np.float32), nan=0.0)
                - np.nan_to_num(test_X_v3[T].to_numpy(np.float32), nan=0.0)
            ).max(axis=1) > 1e-4
            acc_all.append(frame_acc(y_t, p_t))
            acc_ghost.append(frame_acc(y_t, p_t, diff_feat))
            acc_clean.append(frame_acc(y_t, p_t, ~diff_feat))
        print(
            f"  {mname:44s}: All={np.mean(acc_all):.4f} | GhostFrames={np.mean(acc_ghost):.4f} | NonGhost={np.mean(acc_clean):.4f}",
            flush=True,
        )

    # Judge 1A: 7-Fold LOMO on degrade_v3 Test Folds
    print("\n--- JUDGE 1A: 7-Fold LOMO on `degrade_v3` Test Folds ---", flush=True)
    j1_v3_configs = [
        ("0. v4_ens_mean Baseline",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"],
         "linear", 0.50, 0.50, False),
        ("1. SymGBM_v3_TTA only (no TCN)",
         lambda d: d["g_v3_on_v3"], lambda d: d["g_v3_on_v3"], lambda d: d["tr_v3_on_v3"], lambda d: d["tr_v3_on_v3"],
         "linear", 1.00, 1.00, False),
        ("2. SymGBM_v6_TTA only (no TCN)",
         lambda d: d["g_v6_on_v3"], lambda d: d["g_v6_on_v3"], lambda d: d["tr_v6_on_v3"], lambda d: d["tr_v6_on_v3"],
         "linear", 1.00, 1.00, False),
        ("3. 0.5*SymGBM_v3_TTA + 0.5*SymGBM_v6_TTA only (no TCN)",
         lambda d: 0.5 * (d["g_v3_on_v3"] + d["g_v6_on_v3"]),
         lambda d: 0.5 * (d["g_v3_on_v3"] + d["g_v6_on_v3"]),
         lambda d: 0.5 * (d["tr_v3_on_v3"] + d["tr_v6_on_v3"]),
         lambda d: 0.5 * (d["tr_v3_on_v3"] + d["tr_v6_on_v3"]),
         "linear", 1.00, 1.00, False),
        ("4. SymGBM_v3_TTA + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_v3_on_v3"], lambda d: d["t_sym"], lambda d: d["tr_v3_on_v3"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("5. SymGBM_v6_TTA + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_v6_on_v3"], lambda d: d["t_sym"], lambda d: d["tr_v6_on_v3"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("6. (0.5*SymGBM_v3 + 0.5*SymGBM_v6) + SymTCN_TTA (lin 0.55, 0.50)",
         lambda d: 0.5 * (d["g_v3_on_v3"] + d["g_v6_on_v3"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_v3_on_v3"] + d["tr_v6_on_v3"]),
         lambda d: d["tc_sym"],
         "linear", 0.55, 0.50, False),
        ("7. (0.5*SymGBM_v3 + 0.5*SymGBM_v6) + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: 0.5 * (d["g_v3_on_v3"] + d["g_v6_on_v3"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_v3_on_v3"] + d["tr_v6_on_v3"]),
         lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("8. (0.5*SymGBM_v3 + 0.5*SymGBM_v6) + SymTCN_TTA (hyb 0.55, 0.50) + MBR",
         lambda d: 0.5 * (d["g_v3_on_v3"] + d["g_v6_on_v3"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_v3_on_v3"] + d["tr_v6_on_v3"]),
         lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, True),
    ]

    for label, fg, ft, ftr, ftc, bmode, wg, wc, use_mbr in j1_v3_configs:
        scores = []
        for T in MATCH_IDS:
            d = oof_g[T]
            pg, pt = fg(d), ft(d)
            tr, tc = ftr(d), ftc(d)
            p = pg if wg == 1.0 else blend_proba(pg, pt, w_gbm=wg, mode=bmode)
            c = tr if wc == 1.0 else wc * tr + (1.0 - wc) * tc
            if use_mbr:
                preds_k = [decode_v4(p, sportec[T].half, dp, p_change=c) for dp in fold_cands[T]]
                pred = mbr_select(preds_k, sportec[T].half, fold_w[T])
            else:
                pred = decode_v4(p, sportec[T].half, fold_dec_hyb[T], p_change=c)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(f"  {label:68s}: mean={arr.mean():.4f} std={arr.std():.4f} | folds={[round(x, 4) for x in scores]}", flush=True)

    # Judge 1B: 7-Fold LOMO on degrade_v6 Test Folds
    print("\n--- JUDGE 1B: 7-Fold LOMO on `degrade_v6` (Ghost-Incoherent) Test Folds ---", flush=True)
    j1_v6_configs = [
        ("1. SymGBM_v3_TTA only (on v6 test)",
         lambda d: d["g_v3_on_v6"], lambda d: d["g_v3_on_v6"], lambda d: d["tr_v3_on_v6"], lambda d: d["tr_v3_on_v6"],
         "linear", 1.00, 1.00, False),
        ("2. SymGBM_v6_TTA only (on v6 test)",
         lambda d: d["g_v6_on_v6"], lambda d: d["g_v6_on_v6"], lambda d: d["tr_v6_on_v6"], lambda d: d["tr_v6_on_v6"],
         "linear", 1.00, 1.00, False),
        ("3. 0.5*SymGBM_v3_TTA + 0.5*SymGBM_v6_TTA only (on v6 test)",
         lambda d: 0.5 * (d["g_v3_on_v6"] + d["g_v6_on_v6"]),
         lambda d: 0.5 * (d["g_v3_on_v6"] + d["g_v6_on_v6"]),
         lambda d: 0.5 * (d["tr_v3_on_v6"] + d["tr_v6_on_v6"]),
         lambda d: 0.5 * (d["tr_v3_on_v6"] + d["tr_v6_on_v6"]),
         "linear", 1.00, 1.00, False),
        ("4. SymGBM_v3_TTA (on v6) + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_v3_on_v6"], lambda d: d["t_sym"], lambda d: d["tr_v3_on_v6"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("5. SymGBM_v6_TTA (on v6) + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_v6_on_v6"], lambda d: d["t_sym"], lambda d: d["tr_v6_on_v6"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("6. (0.5*SymGBM_v3 + 0.5*SymGBM_v6) (on v6) + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: 0.5 * (d["g_v3_on_v6"] + d["g_v6_on_v6"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_v3_on_v6"] + d["tr_v6_on_v6"]),
         lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("7. (0.5*SymGBM_v3 + 0.5*SymGBM_v6) (on v6) + SymTCN_TTA (hyb 0.55) + MBR",
         lambda d: 0.5 * (d["g_v3_on_v6"] + d["g_v6_on_v6"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_v3_on_v6"] + d["tr_v6_on_v6"]),
         lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, True),
    ]

    for label, fg, ft, ftr, ftc, bmode, wg, wc, use_mbr in j1_v6_configs:
        scores = []
        for T in MATCH_IDS:
            d = oof_g[T]
            pg, pt = fg(d), ft(d)
            tr, tc = ftr(d), ftc(d)
            p = pg if wg == 1.0 else blend_proba(pg, pt, w_gbm=wg, mode=bmode)
            c = tr if wc == 1.0 else wc * tr + (1.0 - wc) * tc
            if use_mbr:
                preds_k = [decode_v4(p, sportec[T].half, dp, p_change=c) for dp in fold_cands[T]]
                pred = mbr_select(preds_k, sportec[T].half, fold_w[T])
            else:
                pred = decode_v4(p, sportec[T].half, fold_dec_hyb[T], p_change=c)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(f"  {label:68s}: mean={arr.mean():.4f} std={arr.std():.4f} | folds={[round(x, 4) for x in scores]}", flush=True)

    # Full 7-match SymGBM_v6 & SymTrans_v6
    full_gbm_pkl = out_dir / "symgbm_v6_full.pkl"
    full_tr_pkl = out_dir / "symtrans_v6_full.pkl"
    if full_gbm_pkl.exists() and full_tr_pkl.exists():
        with open(full_gbm_pkl, "rb") as f:
            symgbm_v6_full = pickle.load(f)
        with open(full_tr_pkl, "rb") as f:
            symtrans_v6_full = pickle.load(f)
        print("\n--- Loaded Full 7-Match SymGBM_v6 & SymTrans_v6 from disk ---", flush=True)
    else:
        print("\n--- Training Full 7-Match SymGBM_v6 & SymTrans_v6 ---", flush=True)
        tf = time.time()
        Xs_all, ys_all, hs_all, gs_all = [], [], [], []
        Xs_mir, ys_mir, gs_mir = [], [], []
        for i, mid in enumerate(MATCH_IDS):
            m = sportec[mid]
            for key, _, _ in TRAIN_VARIANTS:
                Xs_all.append(train_X_v6[(mid, key)])
                ys_all.append(m.labels)
                hs_all.append(m.half)
                gs_all.append(i)
                Xs_mir.append(train_X_v6_mir[(mid, key)])
                ys_mir.append(swap_labels(m.labels))
                gs_mir.append(len(MATCH_IDS) + i)
        Xs_sym = Xs_all + Xs_mir
        ys_sym = ys_all + ys_mir
        gs_sym = gs_all + gs_mir
        ct_all = [change_targets(y, h) for y, h in zip(ys_all, hs_all)]
        symgbm_v6_full = TwoHeadGBM(params=FAST_GBM_PARAMS, seed=0).fit(Xs_sym, ys_sym, gs_sym)
        symtrans_v6_full = TransitionGBM(params=FAST_TRANS_PARAMS, seed=0).fit(Xs_sym, ct_all + ct_all)
        with open(full_gbm_pkl, "wb") as f:
            pickle.dump(symgbm_v6_full, f)
        with open(full_tr_pkl, "wb") as f:
            pickle.dump(symtrans_v6_full, f)
        print(f"  Full SymGBM_v6 & SymTrans_v6 trained in {time.time()-tf:.1f}s", flush=True)

    # Judge 2 (Metrica Sample) & Judge 3 (BAR-ATH Broadcast-Repaired)
    print("\n--- JUDGE 2 (Metrica Sample) & JUDGE 3 (BAR-ATH Repaired): Multi-Regime Blend ---", flush=True)
    models_v4 = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())
    gbm_v4 = models_v4["gbm"]
    trans_v4 = models_v4["trans"]
    tcn_v4 = models_v4["tcn"]
    dec_v4 = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})

    with open(sym_dir / "symgbm_full.pkl", "rb") as f:
        symgbm_v3_full = pickle.load(f)
    with open(sym_dir / "symtrans_full.pkl", "rb") as f:
        symtrans_v3_full = pickle.load(f)
    with open(sym_dir / "symtcn_full.pkl", "rb") as f:
        symtcn_full = pickle.load(f)

    sample_matches, sample_preds = {}, {}
    for gid in GAMES:
        ms = load_sample(gid)
        Xs = build_features(ms, f"sample_{gid}_raw", "v1ctl")
        sample_matches[gid] = ms
        pt0, ct0 = tcn_v4.predict(Xs, ms.half)
        pt_sym, ct_sym = predict_tcn_tta(symtcn_full, Xs, ms.half, ms.pitch)
        sample_preds[gid] = {
            "g_orig": gbm_v4.predict_proba(Xs),
            "tr_orig": trans_v4.predict(Xs),
            "t_orig": pt0,
            "tc_orig": ct0,
            "g_v3": predict_gbm_tta(symgbm_v3_full, Xs, ms.half, ms.pitch),
            "tr_v3": predict_trans_tta(symtrans_v3_full, Xs, ms.half, ms.pitch),
            "g_v6": predict_gbm_tta(symgbm_v6_full, Xs, ms.half, ms.pitch),
            "tr_v6": predict_trans_tta(symtrans_v6_full, Xs, ms.half, ms.pitch),
            "t_sym": pt_sym,
            "tc_sym": ct_sym,
        }

    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)

    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)
    Xa_rep = build_features(atd_rep, "atd_rep_True_True_None", "v1ctl")

    pt0_a, ct0_a = tcn_v4.predict(Xa_rep, atd.half)
    pt_sym_a, ct_sym_a = predict_tcn_tta(symtcn_full, Xa_rep, atd.half, atd.pitch)
    atd_rep_d = {
        "g_orig": gbm_v4.predict_proba(Xa_rep),
        "tr_orig": trans_v4.predict(Xa_rep),
        "t_orig": pt0_a,
        "tc_orig": ct0_a,
        "g_v3": predict_gbm_tta(symgbm_v3_full, Xa_rep, atd.half, atd.pitch),
        "tr_v3": predict_trans_tta(symtrans_v3_full, Xa_rep, atd.half, atd.pitch),
        "g_v6": predict_gbm_tta(symgbm_v6_full, Xa_rep, atd.half, atd.pitch),
        "tr_v6": predict_trans_tta(symtrans_v6_full, Xa_rep, atd.half, atd.pitch),
        "t_sym": pt_sym_a,
        "tc_sym": ct_sym_a,
    }

    j23_variants = [
        ("0. v4 + Repair Baseline",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"],
         "linear", 0.50, 0.50, False),
        ("1. SymGBM_v3_TTA only + Repair",
         lambda d: d["g_v3"], lambda d: d["g_v3"], lambda d: d["tr_v3"], lambda d: d["tr_v3"],
         "linear", 1.00, 1.00, False),
        ("2. SymGBM_v6_TTA only + Repair",
         lambda d: d["g_v6"], lambda d: d["g_v6"], lambda d: d["tr_v6"], lambda d: d["tr_v6"],
         "linear", 1.00, 1.00, False),
        ("3. 0.5*SymGBM_v3_TTA + 0.5*SymGBM_v6_TTA only + Repair",
         lambda d: 0.5 * (d["g_v3"] + d["g_v6"]),
         lambda d: 0.5 * (d["g_v3"] + d["g_v6"]),
         lambda d: 0.5 * (d["tr_v3"] + d["tr_v6"]),
         lambda d: 0.5 * (d["tr_v3"] + d["tr_v6"]),
         "linear", 1.00, 1.00, False),
        ("4. SymGBM_v3_TTA + SymTCN_TTA (hyb 0.55, 0.50) + Repair",
         lambda d: d["g_v3"], lambda d: d["t_sym"], lambda d: d["tr_v3"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("5. SymGBM_v6_TTA + SymTCN_TTA (hyb 0.55, 0.50) + Repair",
         lambda d: d["g_v6"], lambda d: d["t_sym"], lambda d: d["tr_v6"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("6. (0.5*SymGBM_v3 + 0.5*SymGBM_v6) + SymTCN_TTA (lin 0.55, 0.50) + Repair",
         lambda d: 0.5 * (d["g_v3"] + d["g_v6"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_v3"] + d["tr_v6"]),
         lambda d: d["tc_sym"],
         "linear", 0.55, 0.50, False),
        ("7. (0.5*SymGBM_v3 + 0.5*SymGBM_v6) + SymTCN_TTA (hyb 0.55, 0.50) + Repair",
         lambda d: 0.5 * (d["g_v3"] + d["g_v6"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_v3"] + d["tr_v6"]),
         lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("8. (0.65*SymGBM_v3 + 0.35*SymGBM_v6) + SymTCN_TTA (hyb 0.55, 0.50) + Repair",
         lambda d: 0.65 * d["g_v3"] + 0.35 * d["g_v6"],
         lambda d: d["t_sym"],
         lambda d: 0.65 * d["tr_v3"] + 0.35 * d["tr_v6"],
         lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("9. (0.5*SymGBM_v3 + 0.5*SymGBM_v6) + SymTCN_TTA (hyb 0.55) + MBR + Repair",
         lambda d: 0.5 * (d["g_v3"] + d["g_v6"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_v3"] + d["tr_v6"]),
         lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, True),
    ]

    for vname, fg, ft, ftr, ftc, bmode, wg, wc, use_mbr in j23_variants:
        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            d = sample_preds[gid]
            pg, pt = fg(d), ft(d)
            tr, tc = ftr(d), ftc(d)
            p = pg if wg == 1.0 else blend_proba(pg, pt, w_gbm=wg, mode=bmode)
            c = tr if wc == 1.0 else wc * tr + (1.0 - wc) * tc
            if use_mbr:
                cands_7 = [fold_dec_hyb[T] for T in MATCH_IDS]
                preds_k = [decode_v4(p, ms.half, dp, p_change=c) for dp in cands_7]
                pred = mbr_select(preds_k, ms.half)
            else:
                pred = decode_v4(p, ms.half, dec_v4, p_change=c)
            s_raw.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

        da = atd_rep_d
        pg_a, pt_a = fg(da), ft(da)
        tr_a, tc_a = ftr(da), ftc(da)
        p_a = pg_a if wg == 1.0 else blend_proba(pg_a, pt_a, w_gbm=wg, mode=bmode)
        c_a = tr_a if wc == 1.0 else wc * tr_a + (1.0 - wc) * tc_a
        if use_mbr:
            cands_7 = [fold_dec_hyb[T] for T in MATCH_IDS]
            preds_a_k = [decode_v4(p_a, half_dec, dp, p_change=c_a) for dp in cands_7]
            pred_a = mbr_select(preds_a_k, half_dec)
        else:
            pred_a = decode_v4(p_a, half_dec, dec_v4, p_change=c_a)
        pred_a[second_start:kick2] = DEAD
        pred_a[~atd.graded] = DEAD

        pub = metrics.score_public(pred_a, bootstrap_n=100)
        pl = plausibility(pred_a, atd.half, atd.graded)
        pct_H = 100.0 * float((pred_a[atd.graded] == 0).mean())
        pct_A = 100.0 * float((pred_a[atd.graded] == 1).mean())
        print(
            f"  {vname:76s} | Sample={np.mean(s_raw):.4f} (noD={np.mean(s_nod):.4f}) "
            f"| Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
            f"| pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}% (%H={pct_H:.1f}/%A={pct_A:.1f})",
            flush=True,
        )

    print(f"\nEXP-G completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
