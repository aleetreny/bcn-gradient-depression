"""Wave 4 (`EXP-04` + `EXP-05`): First-Principles Physics Feature Engineering (`v1ctl_physics_v2`).

Performs:
1. Strict Equivariance Verification (`0.000e+00` error) on real match data (`sportec J03WMX` and `BAR-ATH`):
   - Verifies `compute_physv2_features(swap_match_ha_and_rotate180(m))` vs `mirror_physv2_df(compute_physv2_features(m))`
   - Verifies `compute_plus_features` + `compute_physv2_features` equivariance and full 481-column
     `mirror_v1ctl_physv2_df` involution (`< 1e-5` across every single column, `nan_mismatch == 0`).
2. Precomputes & caches all `physv2` feature blocks in `local_data/cache/physv2/` for:
   - All 7 Sportec matches (`degrade_v3` seeds `s0, s1, s2, static05_s1` + `degrade_v6` `s0`)
   - All 3 Metrica Sample games (`raw`)
   - `BAR-ATH` (`raw` and `repaired`)
3. Runs 7-fold LOMO `SymGBM_PhysV2_TTA` and `SymTrans_PhysV2_TTA` probe + full 7-match models,
   saving OOF predictions (`v3` and `v6` test folds), full models, and Sample/BAR-ATH predictions
   to `experiments/outputs/exp_w4_oof/`.
4. Evaluates across Judge 1A (`degrade_v3`), Judge 1B (`degrade_v6`), Judge 2 (Metrica Sample 3 games),
   and Judge 3 (`BAR-ATH` Unrepaired & Broadcast-Repaired).
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_7_nested_full import TRAIN_VARIANTS
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.common import DEAD
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load as load_sportec, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from src.features.symmetry import swap_labels, swap_proba
from src.features.v1ctl_plus import compute_plus_features, mirror_plus_df
from src.features.v1ctl_physics_v2 import (
    build_v1ctl_physics_v2,
    build_v1ctl_physics_v2 as build_features_physv2,
    cached_physv2,
    compute_physv2_features,
    mirror_physv2_df,
    mirror_v1ctl_physv2_df,
    swap_match_ha_and_rotate180,
)
from src.models.gbm import DEFAULT_PARAMS as GBM_DEFAULT_PARAMS, TwoHeadGBM
from src.models.transition import PARAMS as TRANS_DEFAULT_PARAMS, TransitionGBM, change_targets
from exp_b_symmetry_eval import blend_proba

GBM_PARAMS_4T = dict(GBM_DEFAULT_PARAMS, n_jobs=4)
TRANS_PARAMS_4T = dict(TRANS_DEFAULT_PARAMS, n_jobs=4)


def predict_gbm_physv2_tta(gbm: TwoHeadGBM, X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    p_orig = gbm.predict_proba(X)
    X_mir = mirror_v1ctl_physv2_df(X, half, pitch)
    p_mir = swap_proba(gbm.predict_proba(X_mir))
    out = 0.5 * p_orig + 0.5 * p_mir
    out /= out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


def predict_trans_physv2_tta(tr: TransitionGBM, X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    c_orig = tr.predict(X)
    X_mir = mirror_v1ctl_physv2_df(X, half, pitch)
    c_mir = tr.predict(X_mir)
    return (0.5 * c_orig + 0.5 * c_mir).astype(np.float32)


def verify_equivariance() -> None:
    print("\n--- Step 1: Strict Equivariance Verification on Sportec J03WMX & BAR-ATH ---", flush=True)
    m_j03 = load_sportec("J03WMX")
    m_atd = load_atd()

    for name, m in [("sportec J03WMX", m_j03), ("BAR-ATH", m_atd)]:
        t_eq = time.time()
        m_swap = swap_match_ha_and_rotate180(m)

        # 1. Exact equivariance of physv2 (129 cols) + plus (108 cols) = 237 physics cols
        P2_orig = compute_physv2_features(m)
        P2_swap = compute_physv2_features(m_swap)
        P2_mir = mirror_physv2_df(P2_orig, m.half, m.pitch)

        max_err_p2 = 0.0
        worst_col_p2 = "none"
        nan_mis_p2 = 0
        for c in P2_orig.columns:
            a, b = P2_mir[c].to_numpy(), P2_swap[c].to_numpy()
            nan_mis_p2 += int((np.isnan(a) != np.isnan(b)).sum())
            if np.isfinite(a).any():
                err = float(np.nanmax(np.abs(a - b)))
                if err > max_err_p2:
                    max_err_p2 = err
                    worst_col_p2 = c

        P1_orig = compute_plus_features(m)
        P1_swap = compute_plus_features(m_swap)
        P1_mir = mirror_plus_df(P1_orig, m.half, m.pitch)
        max_err_p1 = 0.0
        nan_mis_p1 = 0
        for c in P1_orig.columns:
            a, b = P1_mir[c].to_numpy(), P1_swap[c].to_numpy()
            nan_mis_p1 += int((np.isnan(a) != np.isnan(b)).sum())
            if np.isfinite(a).any():
                err = float(np.nanmax(np.abs(a - b)))
                if err > max_err_p1:
                    max_err_p1 = err

        print(
            f"  [{name}] physv2 ({len(P2_orig.columns)} cols): max_err = {max_err_p2:.3e} (worst={worst_col_p2}), nan_mis = {nan_mis_p2} | "
            f"plus ({len(P1_orig.columns)} cols): max_err = {max_err_p1:.3e}, nan_mis = {nan_mis_p1} ({time.time()-t_eq:.1f}s)",
            flush=True,
        )
        assert max_err_p2 < 1e-5 and nan_mis_p2 == 0, f"physv2 equivariance failed on {name}: {max_err_p2}"
        assert max_err_p1 < 1e-5 and nan_mis_p1 == 0, f"plus equivariance failed on {name}: {max_err_p1}"


def main() -> None:
    t0 = time.time()
    print("=== Wave 4 (EXP-04 + EXP-05): First-Principles Physics v2 (`v1ctl_physics_v2`) ===", flush=True)

    # 1. Strict Equivariance Check on J03WMX and BAR-ATH
    verify_equivariance()

    # 2. Build & Cache All `physv2` Feature Matrices in `local_data/cache/physv2/`
    print("\n--- Step 2: Precomputing & Caching `physv2` Feature Matrices (Sportec v3/v6, Sample, BAR-ATH) ---", flush=True)
    tb = time.time()
    sportec = load_all()
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")

    train_X_v3, train_X_v3_mir, test_X_v3, test_X_v6 = {}, {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c3 = replace(cfg_v3, **over) if over else cfg_v3
            m_deg = degrade(m, c3, seed)
            X_tr = build_v1ctl_physics_v2(m_deg, f"sportec_{mid}_degrade_v3_{key}")
            train_X_v3[(mid, key)] = X_tr
            train_X_v3_mir[(mid, key)] = mirror_v1ctl_physv2_df(X_tr, m.half, m.pitch)
        test_X_v3[mid] = build_v1ctl_physics_v2(degrade(m, cfg_v3, 0), f"sportec_{mid}_degrade_v3_s0")
        test_X_v6[mid] = build_v1ctl_physics_v2(degrade(m, cfg_v6, 0), f"sportec_{mid}_degrade_v6_s0")

    # Verify full 481-column mirror involution on Sportec J03WMX s0
    X0 = test_X_v3["J03WMX"]
    X0_m2 = mirror_v1ctl_physv2_df(mirror_v1ctl_physv2_df(X0, sportec["J03WMX"].half, sportec["J03WMX"].pitch), sportec["J03WMX"].half, sportec["J03WMX"].pitch)
    inv_err = float(np.nanmax(np.abs(X0.to_numpy(np.float32) - X0_m2.to_numpy(np.float32))))
    print(f"  [Full 481-col Involution Check] max |mirror(mirror(X)) - X| = {inv_err:.3e}", flush=True)
    assert inv_err < 1e-5

    sample_matches, sample_X = {}, {}
    for gid in GAMES:
        ms = load_sample(gid)
        sample_matches[gid] = ms
        sample_X[gid] = build_v1ctl_physics_v2(ms, f"sample_{gid}_raw")

    from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
    atd = load_atd()
    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)

    X_atd_raw = build_v1ctl_physics_v2(atd, "atd")
    X_atd_rep = build_v1ctl_physics_v2(atd_rep, "atd_rep_True_True_None")

    physv2_cache_files = list((ROOT / "local_data/cache/physv2").glob("*.parquet"))
    print(
        f"  Cached {len(physv2_cache_files)} physv2 parquet files in local_data/cache/physv2/ | "
        f"Total features = {X0.shape[1]} cols ({time.time()-tb:.1f}s)",
        flush=True,
    )

    # 3. 7-Fold Outer LOMO Probe (`SymGBM_PhysV2_TTA` & `SymTrans_PhysV2_TTA`)
    print("\n--- Step 3: 7-Fold Outer LOMO Probe (`SymGBM_PhysV2_TTA` & `SymTrans_PhysV2_TTA`) ---", flush=True)
    out_dir = ROOT / "experiments" / "outputs" / "exp_w4_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_w4 = {}
    for k, T in enumerate(MATCH_IDS):
        p_path = out_dir / f"physv2_{T}.npz"
        if p_path.exists():
            oof_w4[T] = dict(np.load(p_path))
            print(f"  Fold {k} ({T}) loaded from cache {p_path.name}", flush=True)
            continue

        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        Xs_std = [train_X_v3[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_std = [sportec[mid].labels for mid in train_ids for _ in TRAIN_VARIANTS]
        hs_std = [sportec[mid].half for mid in train_ids for _ in TRAIN_VARIANTS]
        gs_std = [i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]

        Xs_sym = Xs_std + [train_X_v3_mir[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_sym = ys_std + [swap_labels(sportec[mid].labels) for mid in train_ids for _ in TRAIN_VARIANTS]
        gs_sym = gs_std + [len(train_ids) + i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]

        symgbm_p2 = TwoHeadGBM(params=GBM_PARAMS_4T, seed=k).fit(Xs_sym, ys_sym, gs_sym)
        p_v3 = predict_gbm_physv2_tta(symgbm_p2, test_X_v3[T], sportec[T].half, sportec[T].pitch)
        p_v6 = predict_gbm_physv2_tta(symgbm_p2, test_X_v6[T], sportec[T].half, sportec[T].pitch)

        ct_std = [change_targets(y, h) for y, h in zip(ys_std, hs_std)]
        symtrans_p2 = TransitionGBM(params=TRANS_PARAMS_4T, seed=k).fit(Xs_sym, ct_std + ct_std)
        tr_v3 = predict_trans_physv2_tta(symtrans_p2, test_X_v3[T], sportec[T].half, sportec[T].pitch)
        tr_v6 = predict_trans_physv2_tta(symtrans_p2, test_X_v6[T], sportec[T].half, sportec[T].pitch)

        oof_w4[T] = {
            "symgbm_physv2_tta": p_v3,
            "symtrans_physv2_tta": tr_v3,
            "symgbm_physv2_tta_on_v6": p_v6,
            "symtrans_physv2_tta_on_v6": tr_v6,
        }
        np.savez_compressed(p_path, **oof_w4[T])
        print(f"  Fold {k} ({T}) trained & saved in {time.time()-tf:.1f}s", flush=True)

    # 4. Evaluate on Judge 1A (Sportec degrade_v3) and Judge 1B (Sportec degrade_v6)
    store_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    ref_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl"
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    plus_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std, fold_dec_hyb, _, _ = pickle.load(f)

    oof_all = {}
    for T in MATCH_IDS:
        g_orig = np.load(store_v3 / T / f"gbm_outer_{T}.npy")
        tr_orig = np.load(store_v3 / T / f"trans_outer_{T}.npy")
        z_orig = np.load(store_v3 / T / f"tcn_outer_{T}.npz")
        sb = np.load(sym_dir / f"sym_{T}.npz")
        st = np.load(sym_dir / f"tcn_sym_{T}.npz")
        sp = np.load(plus_dir / f"plus_{T}.npz")
        sg = np.load(g_dir / f"ghost_sym_{T}.npz")
        oof_all[T] = {
            "g_orig": g_orig,
            "tr_orig": tr_orig,
            "t_orig": z_orig["proba"],
            "tc_orig": z_orig["change"],
            "g_sym": sb["symgbm_tta"],
            "tr_sym": sb["symtrans_tta"],
            "t_sym": st["symtcn_tta_proba"],
            "tc_sym": st["symtcn_tta_change"],
            "g_plus": sp["symgbm_plus_tta"],
            "tr_plus": sp["symtrans_plus_tta"],
            "g_p2": oof_w4[T]["symgbm_physv2_tta"],
            "tr_p2": oof_w4[T]["symtrans_physv2_tta"],
            "g_v3_on_v6": sg["g_v3_on_v6"],
            "tr_v3_on_v6": sg["tr_v3_on_v6"],
            "g_v6_on_v6": sg["g_v6_on_v6"],
            "tr_v6_on_v6": sg["tr_v6_on_v6"],
            "g_p2_on_v6": oof_w4[T]["symgbm_physv2_tta_on_v6"],
            "tr_p2_on_v6": oof_w4[T]["symtrans_physv2_tta_on_v6"],
        }

    print("\n--- JUDGE 1A: 7-Fold Nested LOMO Results on Sportec (`degrade_v3`) ---", flush=True)
    print("  [A] Standalone GBM-only (no TCN, no M3 change head):", flush=True)
    for name, fn_g in [
        ("0a. v4 GBM_orig (v1ctl, 244 cols)",                 lambda d: d["g_orig"]),
        ("1a. SymGBM_TTA (v1ctl, 244 cols)",                  lambda d: d["g_sym"]),
        ("2a. SymGBM_Plus_TTA (v1ctl_plus, 352 cols)",        lambda d: d["g_plus"]),
        ("3a. SymGBM_PhysV2_TTA (v1ctl_physv2, 481 cols)",    lambda d: d["g_p2"]),
        ("4a. 50/50(SymGBM_Plus, SymGBM_PhysV2)",             lambda d: 0.5 * (d["g_plus"] + d["g_p2"])),
        ("5a. 3-Way Tree(v1ctl + Plus + PhysV2)",             lambda d: (d["g_sym"] + d["g_plus"] + d["g_p2"]) / 3.0),
    ]:
        scores, f_accs = [], []
        for T in MATCH_IDS:
            params_gbm = DecoderParams(**json.loads((ref_v3 / T / "params_gbm.json").read_text()))
            p = fn_g(oof_all[T])
            pred = decode_v4(p, sportec[T].half, params_gbm)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
            valid = sportec[T].labels >= 0
            f_accs.append(float((np.argmax(p[valid], axis=1) == sportec[T].labels[valid]).mean()))
        arr = np.array(scores)
        print(f"    {name:48s}: SpellF1={arr.mean():.4f} (std={arr.std():.4f}) | FrameAcc={np.mean(f_accs):.4f} | folds={[round(x, 4) for x in scores]}", flush=True)

    print("\n  [B] Standalone GBM + TransitionHead (no TCN):", flush=True)
    for name, fn_g, fn_tr in [
        ("0b. v4 GBM_orig + Trans_orig (v1ctl)",              lambda d: d["g_orig"], lambda d: d["tr_orig"]),
        ("1b. SymGBM_TTA + SymTrans_TTA (v1ctl)",             lambda d: d["g_sym"],  lambda d: d["tr_sym"]),
        ("2b. SymGBM_Plus_TTA + SymTrans_Plus_TTA",           lambda d: d["g_plus"], lambda d: d["tr_plus"]),
        ("3b. SymGBM_PhysV2_TTA + SymTrans_PhysV2_TTA",       lambda d: d["g_p2"],   lambda d: d["tr_p2"]),
        ("4b. 50/50(Plus, PhysV2) + SymTrans",                lambda d: 0.5 * (d["g_plus"] + d["g_p2"]), lambda d: 0.5 * (d["tr_plus"] + d["tr_p2"])),
        ("5b. 3-Way Tree(v1ctl+Plus+PhysV2) + SymTrans",      lambda d: (d["g_sym"] + d["g_plus"] + d["g_p2"]) / 3.0, lambda d: (d["tr_sym"] + d["tr_plus"] + d["tr_p2"]) / 3.0),
    ]:
        scores = []
        for T in MATCH_IDS:
            d = oof_all[T]
            pred = decode_v4(fn_g(d), sportec[T].half, fold_dec_std[T], p_change=fn_tr(d))
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(f"    {name:48s}: SpellF1={arr.mean():.4f} (std={arr.std():.4f}) | folds={[round(x, 4) for x in scores]}", flush=True)

    print("\n  [C] Ensembles with SymTCN_TTA (`degrade_v3` & `degrade_v6`):", flush=True)
    for name, fg_v3, ftr_v3, fg_v6, ftr_v6, bmode, wg in [
        ("0c. v4_ens_mean Baseline (orig_GBM + orig_TCN)",
         lambda d: d["g_orig"], lambda d: d["tr_orig"], lambda d: d["g_v3_on_v6"], lambda d: d["tr_v3_on_v6"], "linear", 0.50),
        ("1c. SymGBM_v1ctl + SymTCN_TTA (hyb 0.55)",
         lambda d: d["g_sym"], lambda d: d["tr_sym"], lambda d: d["g_v3_on_v6"], lambda d: d["tr_v3_on_v6"], "hybrid", 0.55),
        ("2c. SymGBM_Plus + SymTCN_TTA (hyb 0.55)",
         lambda d: d["g_plus"], lambda d: d["tr_plus"], lambda d: d["g_p2_on_v6"], lambda d: d["tr_p2_on_v6"], "hybrid", 0.55),
        ("3c. SymGBM_PhysV2 + SymTCN_TTA (hyb 0.55)",
         lambda d: d["g_p2"], lambda d: d["tr_p2"], lambda d: d["g_p2_on_v6"], lambda d: d["tr_p2_on_v6"], "hybrid", 0.55),
        ("4c. 50/50(Plus, PhysV2) + SymTCN_TTA (hyb 0.55)",
         lambda d: 0.5 * (d["g_plus"] + d["g_p2"]), lambda d: 0.5 * (d["tr_plus"] + d["tr_p2"]),
         lambda d: 0.5 * (d["g_p2_on_v6"] + d["g_v6_on_v6"]), lambda d: 0.5 * (d["tr_p2_on_v6"] + d["tr_v6_on_v6"]), "hybrid", 0.55),
        ("5c. 3-Way Tree(v1ctl+Plus+PhysV2) + SymTCN_TTA (hyb 0.55)",
         lambda d: (d["g_sym"] + d["g_plus"] + d["g_p2"]) / 3.0, lambda d: (d["tr_sym"] + d["tr_plus"] + d["tr_p2"]) / 3.0,
         lambda d: (d["g_v3_on_v6"] + d["g_v6_on_v6"] + d["g_p2_on_v6"]) / 3.0, lambda d: (d["tr_v3_on_v6"] + d["tr_v6_on_v6"] + d["tr_p2_on_v6"]) / 3.0, "hybrid", 0.55),
    ]:
        s_v3, s_v6 = [], []
        for T in MATCH_IDS:
            d = oof_all[T]
            p3 = blend_proba(fg_v3(d), d["t_sym"] if "0c" not in name else d["t_orig"], w_gbm=wg, mode=bmode)
            c3 = 0.5 * ftr_v3(d) + 0.5 * (d["tc_sym"] if "0c" not in name else d["tc_orig"])
            pred3 = decode_v4(p3, sportec[T].half, fold_dec_std[T], p_change=c3)
            s_v3.append(metrics.score(sportec[T].labels, pred3, sportec[T].half)["spell_f1_mean"])

            p6 = blend_proba(fg_v6(d), d["t_sym"] if "0c" not in name else d["t_orig"], w_gbm=wg, mode=bmode)
            c6 = 0.5 * ftr_v6(d) + 0.5 * (d["tc_sym"] if "0c" not in name else d["tc_orig"])
            pred6 = decode_v4(p6, sportec[T].half, fold_dec_std[T], p_change=c6)
            s_v6.append(metrics.score(sportec[T].labels, pred6, sportec[T].half)["spell_f1_mean"])
        print(f"    {name:56s}: v3_LOMO={np.mean(s_v3):.4f} | v6_LOMO={np.mean(s_v6):.4f}", flush=True)

    # 5. Train Full 7-Match `symgbm_physv2_full.pkl` & `symtrans_physv2_full.pkl`
    print("\n--- Step 4: Training Full 7-Match `symgbm_physv2_full.pkl` & `symtrans_physv2_full.pkl` ---", flush=True)
    gbm_full_path = out_dir / "symgbm_physv2_full.pkl"
    tr_full_path = out_dir / "symtrans_physv2_full.pkl"
    if gbm_full_path.exists() and tr_full_path.exists():
        with open(gbm_full_path, "rb") as f:
            symgbm_p2_full = pickle.load(f)
        with open(tr_full_path, "rb") as f:
            symtrans_p2_full = pickle.load(f)
        print("  Loaded symgbm_physv2_full.pkl & symtrans_physv2_full.pkl from cache.", flush=True)
    else:
        tf_full = time.time()
        Xs_all, ys_all, hs_all, gs_all = [], [], [], []
        Xs_mir, ys_mir, gs_mir = [], [], []
        for i, mid in enumerate(MATCH_IDS):
            m = sportec[mid]
            for key, _, _ in TRAIN_VARIANTS:
                Xs_all.append(train_X_v3[(mid, key)])
                ys_all.append(m.labels)
                hs_all.append(m.half)
                gs_all.append(i)
                Xs_mir.append(train_X_v3_mir[(mid, key)])
                ys_mir.append(swap_labels(m.labels))
                gs_mir.append(len(MATCH_IDS) + i)
        Xs_sym = Xs_all + Xs_mir
        ys_sym = ys_all + ys_mir
        gs_sym = gs_all + gs_mir
        ct_all = [change_targets(y, h) for y, h in zip(ys_all, hs_all)]
        symgbm_p2_full = TwoHeadGBM(params=GBM_PARAMS_4T, seed=0).fit(Xs_sym, ys_sym, gs_sym)
        symtrans_p2_full = TransitionGBM(params=TRANS_PARAMS_4T, seed=0).fit(Xs_sym, ct_all + ct_all)
        with open(gbm_full_path, "wb") as f:
            pickle.dump(symgbm_p2_full, f)
        with open(tr_full_path, "wb") as f:
            pickle.dump(symtrans_p2_full, f)
        print(f"  Trained & saved symgbm_physv2_full.pkl & symtrans_physv2_full.pkl in {time.time()-tf_full:.1f}s", flush=True)

    # Inspect Top Feature Importances among the new 129 physv2 features
    physv2_cols = set(test_X_v3["J03WMX"].columns[-129:])
    feat_names = list(test_X_v3["J03WMX"].columns)
    imp_alive = symgbm_p2_full.alive_.feature_importances_
    imp_ha = symgbm_p2_full.team_.feature_importances_
    imp_tr = symtrans_p2_full.model_.feature_importances_
    new_ranks = []
    for idx, col in enumerate(feat_names):
        if col in physv2_cols:
            new_ranks.append((int(imp_alive[idx] + imp_ha[idx] + imp_tr[idx]), int(imp_ha[idx]), int(imp_alive[idx]), int(imp_tr[idx]), col))
    new_ranks.sort(reverse=True)
    print("\n  [Top 15 New `physv2` Feature Split Counts (Total = HA + Alive + Trans)]:", flush=True)
    for tot, i_ha, i_al, i_tr, col in new_ranks[:15]:
        print(f"    {col:34s}: total={tot:4d} (HA={i_ha:3d}, Alive={i_al:3d}, Trans={i_tr:3d})", flush=True)

    # 6. Evaluate on Judge 2 (Metrica Sample 3 games) & Judge 3 (BAR-ATH Raw and Repaired)
    print("\n--- Step 5: Judge 2 (Metrica Sample 3 Games) & Judge 3 (BAR-ATH Raw & Repaired) ---", flush=True)
    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        plus_preds = pickle.load(f)

    sample_p2_preds = {}
    for gid in GAMES:
        ms = sample_matches[gid]
        sample_p2_preds[gid] = {
            "g_physv2_sym": predict_gbm_physv2_tta(symgbm_p2_full, sample_X[gid], ms.half, ms.pitch),
            "tr_physv2_sym": predict_trans_physv2_tta(symtrans_p2_full, sample_X[gid], ms.half, ms.pitch),
        }

    atd_raw_p2 = {
        "g_physv2_sym": predict_gbm_physv2_tta(symgbm_p2_full, X_atd_raw, atd.half, atd.pitch),
        "tr_physv2_sym": predict_trans_physv2_tta(symtrans_p2_full, X_atd_raw, atd.half, atd.pitch),
    }
    atd_rep_p2 = {
        "g_physv2_sym": predict_gbm_physv2_tta(symgbm_p2_full, X_atd_rep, atd_rep.half, atd_rep.pitch),
        "tr_physv2_sym": predict_trans_physv2_tta(symtrans_p2_full, X_atd_rep, atd_rep.half, atd_rep.pitch),
    }

    with open(out_dir / "eval_preds_physv2.pkl", "wb") as f:
        pickle.dump({"sample": sample_p2_preds, "atd_raw": atd_raw_p2, "atd_rep": atd_rep_p2}, f)

    dec_gbm = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not k.startswith("_")})
    dec_ens = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})

    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)

    eval_configs = [
        ("A1. SymGBM_TTA (v1ctl) only",
         lambda d, p2: d["g_sym"], lambda d, p2: None, None, 1.0, dec_gbm),
        ("A2. SymGBM_Plus_TTA (v1ctl_plus) only",
         lambda d, p2: d["g_plus_sym"], lambda d, p2: None, None, 1.0, dec_gbm),
        ("A3. SymGBM_PhysV2_TTA (v1ctl_physv2) only",
         lambda d, p2: p2["g_physv2_sym"], lambda d, p2: None, None, 1.0, dec_gbm),
        ("A4. SymGBM_PhysV2_TTA + SymTrans_PhysV2_TTA (no TCN)",
         lambda d, p2: p2["g_physv2_sym"], lambda d, p2: p2["tr_physv2_sym"], None, 1.0, dec_ens),
        ("B1. SymGBM_Plus_TTA + SymTCN_TTA (hyb 0.55)",
         lambda d, p2: d["g_plus_sym"], lambda d, p2: d["tr_plus_sym"], "hybrid", 0.55, dec_ens),
        ("B2. SymGBM_PhysV2_TTA + SymTCN_TTA (hyb 0.55)",
         lambda d, p2: p2["g_physv2_sym"], lambda d, p2: p2["tr_physv2_sym"], "hybrid", 0.55, dec_ens),
        ("B3. 50/50(Plus, PhysV2) + SymTCN_TTA (hyb 0.55)",
         lambda d, p2: 0.5 * (d["g_plus_sym"] + p2["g_physv2_sym"]),
         lambda d, p2: 0.5 * (d["tr_plus_sym"] + p2["tr_physv2_sym"]), "hybrid", 0.55, dec_ens),
        ("B4. 3-Way Tree(v1ctl+Plus+PhysV2) + SymTCN_TTA (hyb 0.55)",
         lambda d, p2: (d["g_sym"] + d["g_plus_sym"] + p2["g_physv2_sym"]) / 3.0,
         lambda d, p2: (d["tr_sym"] + d["tr_plus_sym"] + p2["tr_physv2_sym"]) / 3.0, "hybrid", 0.55, dec_ens),
    ]

    for vname, fg, ftr, bmode, wg, dec in eval_configs:
        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            d = plus_preds["sample"][gid]
            p2 = sample_p2_preds[gid]
            p = fg(d, p2) if wg == 1.0 else blend_proba(fg(d, p2), d["t_sym"], w_gbm=wg, mode=bmode)
            tr = ftr(d, p2)
            c = None if tr is None else (tr if wg == 1.0 else 0.5 * tr + 0.5 * d["tc_sym"])
            pred = decode_v4(p, ms.half, dec, p_change=c)
            s_raw.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

        # BAR-ATH Raw
        d_raw = plus_preds["atd_raw"]
        p_raw = fg(d_raw, atd_raw_p2) if wg == 1.0 else blend_proba(fg(d_raw, atd_raw_p2), d_raw["t_sym"], w_gbm=wg, mode=bmode)
        tr_r = ftr(d_raw, atd_raw_p2)
        c_raw = None if tr_r is None else (tr_r if wg == 1.0 else 0.5 * tr_r + 0.5 * d_raw["tc_sym"])
        pred_raw = decode_v4(p_raw, half_dec, dec, p_change=c_raw)
        pred_raw[second_start:kick2] = DEAD
        pred_raw[~atd.graded] = DEAD
        pub_raw = metrics.score_public(pred_raw, bootstrap_n=100)
        pl_raw = plausibility(pred_raw, atd.half, atd.graded)
        pct_h_raw = 100.0 * float(np.mean(pred_raw[atd.graded & (pred_raw != DEAD)] == 0))

        # BAR-ATH Repaired
        d_rep = plus_preds["atd_rep"]
        p_rep = fg(d_rep, atd_rep_p2) if wg == 1.0 else blend_proba(fg(d_rep, atd_rep_p2), d_rep["t_sym"], w_gbm=wg, mode=bmode)
        tr_p = ftr(d_rep, atd_rep_p2)
        c_rep = None if tr_p is None else (tr_p if wg == 1.0 else 0.5 * tr_p + 0.5 * d_rep["tc_sym"])
        pred_rep = decode_v4(p_rep, half_dec, dec, p_change=c_rep)
        pred_rep[second_start:kick2] = DEAD
        pred_rep[~atd.graded] = DEAD
        pub_rep = metrics.score_public(pred_rep, bootstrap_n=100)
        pl_rep = plausibility(pred_rep, atd.half, atd.graded)
        pct_h_rep = 100.0 * float(np.mean(pred_rep[atd.graded & (pred_rep != DEAD)] == 0))

        print(
            f"  {vname:58s}\n"
            f"    -> Judge 2 (Sample): mean={np.mean(s_raw):.4f} (noD={np.mean(s_nod):.4f}) | games={[round(x, 4) for x in s_raw]}\n"
            f"    -> Judge 3 (BAR-ATH Raw): Pub={pub_raw['spell_f1_mean']:.4f} (D={pub_raw['per_class']['D']['f1']:.3f}) | pos/m={pl_raw['poss_per_min']:.2f} %D={pl_raw['pct_D']:.1f}% %H={pct_h_raw:.1f}%\n"
            f"    -> Judge 3 (BAR-ATH Rep): Pub={pub_rep['spell_f1_mean']:.4f} (D={pub_rep['per_class']['D']['f1']:.3f}) | pos/m={pl_rep['poss_per_min']:.2f} %D={pl_rep['pct_D']:.1f}% %H={pct_h_rep:.1f}%",
            flush=True,
        )

    print(f"\nWave 4 (EXP-04 + EXP-05) completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
