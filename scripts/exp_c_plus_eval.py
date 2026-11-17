"""EXP-C+: Knot-Anchored Physics + Bernardo Track-Break-Free LSQ Velocities + Gap-Edge Continuity + Equivariant Symmetry.

Evaluates `v1ctl_plus` (`v1ctl` + 108 equivariant physics & continuity features from `src/features/v1ctl_plus.py`):
1. Verifies `< 1e-5` (bit-for-bit 0.0) discrepancy between `mirror_plus_df(compute_plus_features(m))`
   and `compute_plus_features(swap_match(m))`.
2. Judge 1: 7-fold LOMO on Sportec (`degrade_v3`) for `GBM_Plus_TTA`, `SymGBM_Plus_TTA`, and `SymTrans_Plus_TTA`:
   - Standalone (GBM-only and GBM + Trans)
   - Blended with `SymTCN_TTA` and `SymGBM_TTA` from `experiments/outputs/exp_b_sym_oof/`
3. Trains full 7-match `symgbm_plus_full.pkl` and `symtrans_plus_full.pkl` and saves them to `experiments/outputs/exp_c_plus_oof/`.
4. Judge 2 (Metrica Sample 3 games) & Judge 3 (BAR-ATH unrepaired and broadcast-repaired) evaluation.
"""
from __future__ import annotations

from dataclasses import replace
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_7_nested_full import TRAIN_VARIANTS, build_features as build_base_features
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.common import DEAD
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from src.features.symmetry import swap_labels, swap_match, swap_proba
from src.features.v1ctl_plus import (
    build_v1ctl_plus,
    compute_plus_features,
    mirror_plus_df,
    mirror_v1ctl_plus_df,
)
from src.models.gbm import TwoHeadGBM
from src.models.transition import TransitionGBM, change_targets
from exp_b_symmetry_eval import blend_proba, predict_gbm_tta, predict_tcn_tta, predict_trans_tta
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match


def predict_gbm_plus_tta(gbm: TwoHeadGBM, X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    p_orig = gbm.predict_proba(X)
    X_mir = mirror_v1ctl_plus_df(X, half, pitch)
    p_mir = swap_proba(gbm.predict_proba(X_mir))
    out = 0.5 * p_orig + 0.5 * p_mir
    out /= out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


def predict_trans_plus_tta(tr: TransitionGBM, X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    c_orig = tr.predict(X)
    X_mir = mirror_v1ctl_plus_df(X, half, pitch)
    c_mir = tr.predict(X_mir)
    return (0.5 * c_orig + 0.5 * c_mir).astype(np.float32)


def main() -> None:
    t0 = time.time()
    import torch
    from src.models import tcn as tcn_mod
    torch.use_deterministic_algorithms(True)
    tcn_mod._device = lambda: torch.device("cpu")

    print("=== EXP-C+: Knot Physics + Bernardo LSQ Velocities + Gap-Edge Continuity + Equivariance ===", flush=True)

    # 1. Exactness check on mirror_plus_df vs compute_plus_features(swap_match(m))
    print("\n--- Step 1: Equivariance Verification (mirror_plus_df vs compute_plus_features(swap_match(m))) ---", flush=True)
    ms1 = load_sample(1)
    ms1_swap = swap_match(ms1)
    P_orig = compute_plus_features(ms1)
    P_swap = compute_plus_features(ms1_swap)
    P_mir = mirror_plus_df(P_orig, ms1.half, ms1.pitch)
    max_err = 0.0
    nan_mis = 0
    for c in P_orig.columns:
        a, b = P_mir[c].to_numpy(), P_swap[c].to_numpy()
        nan_mis += int((np.isnan(a) != np.isnan(b)).sum())
        if np.isfinite(a).any():
            err = float(np.nanmax(np.abs(a - b)))
            if err > max_err:
                max_err = err
    print(f"  [Equivariance Audit] plus_cols={len(P_orig.columns)} | max |mirror_plus_df(P) - P(swap_match)| = {max_err:.3e} | nan_mismatch = {nan_mis}", flush=True)
    assert max_err < 1e-5 and nan_mis == 0, f"Equivariance check failed: max_err={max_err}, nan_mis={nan_mis}"

    # 2. Build & cache v1ctl_plus features for Sportec degrade_v3, Metrica Sample, and BAR-ATH
    print("\n--- Step 2: Building & Caching v1ctl_plus Features ---", flush=True)
    tb = time.time()
    sportec = load_all()
    cfg = load_config(ROOT / "configs/degrade_v3.json")
    tag = "degrade_v3"
    store = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    ref_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl"
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    out_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_X, train_X_mir, test_X = {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            X_tr = build_v1ctl_plus(degrade(m, c, seed), f"sportec_{mid}_{tag}_{key}")
            train_X[(mid, key)] = X_tr
            train_X_mir[(mid, key)] = mirror_v1ctl_plus_df(X_tr, m.half, m.pitch)
        test_X[mid] = build_v1ctl_plus(degrade(m, cfg, 0), f"sportec_{mid}_{tag}_s0")
    print(f"  Sportec v1ctl_plus built ({test_X[MATCH_IDS[0]].shape[1]} cols) in {time.time()-tb:.1f}s", flush=True)

    # 3. 7-Fold Outer LOMO on Sportec degrade_v3
    print("\n--- Step 3: 7-Fold Outer LOMO Training (GBM_Plus_TTA, SymGBM_Plus_TTA, SymTrans_Plus_TTA) ---", flush=True)
    oof_gbm_plus_tta = {}
    oof_symgbm_plus_tta = {}
    oof_symtrans_plus_tta = {}

    for k, T in enumerate(MATCH_IDS):
        p_path = out_dir / f"plus_{T}.npz"
        if p_path.exists():
            z = np.load(p_path)
            oof_gbm_plus_tta[T] = z["gbm_plus_tta"]
            oof_symgbm_plus_tta[T] = z["symgbm_plus_tta"]
            oof_symtrans_plus_tta[T] = z["symtrans_plus_tta"]
            print(f"  Fold {k} ({T}) loaded from cache {p_path.name}", flush=True)
            continue

        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        Xs_std = [train_X[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_std = [sportec[mid].labels for mid in train_ids for _ in TRAIN_VARIANTS]
        gs_std = [i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]

        gbm_plus = TwoHeadGBM(seed=k).fit(Xs_std, ys_std, gs_std)
        oof_gbm_plus_tta[T] = predict_gbm_plus_tta(gbm_plus, test_X[T], sportec[T].half, sportec[T].pitch)

        Xs_sym = Xs_std + [train_X_mir[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_sym = ys_std + [swap_labels(sportec[mid].labels) for mid in train_ids for _ in TRAIN_VARIANTS]
        gs_sym = gs_std + [len(train_ids) + i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]

        symgbm_plus = TwoHeadGBM(seed=k).fit(Xs_sym, ys_sym, gs_sym)
        oof_symgbm_plus_tta[T] = predict_gbm_plus_tta(symgbm_plus, test_X[T], sportec[T].half, sportec[T].pitch)

        hs_std = [sportec[mid].half for mid in train_ids for _ in TRAIN_VARIANTS]
        ct_std = [change_targets(y, h) for y, h in zip(ys_std, hs_std)]
        symtrans_plus = TransitionGBM(seed=k).fit(Xs_sym, ct_std + ct_std)
        oof_symtrans_plus_tta[T] = predict_trans_plus_tta(symtrans_plus, test_X[T], sportec[T].half, sportec[T].pitch)

        np.savez_compressed(
            p_path,
            gbm_plus_tta=oof_gbm_plus_tta[T],
            symgbm_plus_tta=oof_symgbm_plus_tta[T],
            symtrans_plus_tta=oof_symtrans_plus_tta[T],
        )
        print(f"  Fold {k} ({T}) trained & saved in {time.time()-tf:.1f}s", flush=True)

    # Load saved OOFs from v4 store and EXP-B / EXP-B2
    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std, fold_dec_hyb, fold_cands, fold_w = pickle.load(f)

    oof_all = {}
    for T in MATCH_IDS:
        g_orig = np.load(store / T / f"gbm_outer_{T}.npy")
        tr_orig = np.load(store / T / f"trans_outer_{T}.npy")
        z_orig = np.load(store / T / f"tcn_outer_{T}.npz")
        sb = np.load(sym_dir / f"sym_{T}.npz")
        st = np.load(sym_dir / f"tcn_sym_{T}.npz")
        oof_all[T] = {
            "g_orig": g_orig,
            "tr_orig": tr_orig,
            "t_orig": z_orig["proba"],
            "tc_orig": z_orig["change"],
            "g_tta": sb["gbm_tta"],
            "g_sym": sb["symgbm_tta"],
            "tr_sym": sb["symtrans_tta"],
            "t_tta": st["tcn_tta_proba"],
            "tc_tta": st["tcn_tta_change"],
            "t_sym": st["symtcn_tta_proba"],
            "tc_sym": st["symtcn_tta_change"],
            "g_plus_tta": oof_gbm_plus_tta[T],
            "g_plus_sym": oof_symgbm_plus_tta[T],
            "tr_plus_sym": oof_symtrans_plus_tta[T],
        }

    print("\n--- JUDGE 1: 7-Fold Nested LOMO Results on Sportec (degrade_v3) ---", flush=True)
    # A. Standalone GBM-only (using per-fold params_gbm.json)
    print("  [A] Standalone GBM-only (no TCN, no M3 change head):", flush=True)
    for name, fn_g in [
        ("0a. v4 GBM_orig (v1ctl)",                lambda d: d["g_orig"]),
        ("1a. GBM_TTA (v1ctl)",                    lambda d: d["g_tta"]),
        ("2a. SymGBM_TTA (v1ctl)",                 lambda d: d["g_sym"]),
        ("3a. GBM_Plus_TTA (v1ctl_plus)",          lambda d: d["g_plus_tta"]),
        ("4a. SymGBM_Plus_TTA (v1ctl_plus)",       lambda d: d["g_plus_sym"]),
        ("5a. 50/50(SymGBM_v1ctl, SymGBM_Plus)",   lambda d: 0.5 * (d["g_sym"] + d["g_plus_sym"])),
    ]:
        scores = []
        for T in MATCH_IDS:
            params_gbm = DecoderParams(**json.loads((ref_v3 / T / "params_gbm.json").read_text()))
            pred = decode_v4(fn_g(oof_all[T]), sportec[T].half, params_gbm)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(f"    {name:44s}: mean={arr.mean():.4f} std={arr.std():.4f} | folds={[round(x, 4) for x in scores]}", flush=True)

    # B. Standalone GBM + TransitionHead (using fold_dec_std)
    print("\n  [B] Standalone GBM + TransitionHead (no TCN):", flush=True)
    for name, fn_g, fn_tr in [
        ("0b. v4 GBM_orig + Trans_orig (v1ctl)",              lambda d: d["g_orig"],     lambda d: d["tr_orig"]),
        ("1b. SymGBM_TTA + SymTrans_TTA (v1ctl)",             lambda d: d["g_sym"],      lambda d: d["tr_sym"]),
        ("2b. SymGBM_Plus_TTA + SymTrans_Plus_TTA",           lambda d: d["g_plus_sym"], lambda d: d["tr_plus_sym"]),
        ("3b. 50/50(SymGBM_v1ctl, SymGBM_Plus) + SymTrans",   lambda d: 0.5 * (d["g_sym"] + d["g_plus_sym"]), lambda d: 0.5 * (d["tr_sym"] + d["tr_plus_sym"])),
    ]:
        scores = []
        for T in MATCH_IDS:
            d = oof_all[T]
            pred = decode_v4(fn_g(d), sportec[T].half, fold_dec_std[T], p_change=fn_tr(d))
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(f"    {name:44s}: mean={arr.mean():.4f} std={arr.std():.4f} | folds={[round(x, 4) for x in scores]}", flush=True)

    # C. Blended with TCN / SymTCN_TTA
    print("\n  [C] Ensembles with TCN / SymTCN_TTA (inner-tuned fold_dec_std):", flush=True)
    ens_configs = [
        ("0c. v4_ens_mean Baseline (orig_GBM + orig_TCN, lin 0.50)",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"], "linear", 0.50, 0.50),
        ("1c. SymGBM_TTA(v1ctl) + SymTCN_TTA (lin 0.55, 0.50)",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"], "linear", 0.55, 0.50),
        ("2c. SymGBM_TTA(v1ctl) + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"], "hybrid", 0.55, 0.50),
        ("3c. SymGBM_Plus_TTA + SymTCN_TTA (lin 0.50, 0.50)",
         lambda d: d["g_plus_sym"], lambda d: d["t_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tc_sym"], "linear", 0.50, 0.50),
        ("4c. SymGBM_Plus_TTA + SymTCN_TTA (lin 0.55, 0.50)",
         lambda d: d["g_plus_sym"], lambda d: d["t_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tc_sym"], "linear", 0.55, 0.50),
        ("5c. SymGBM_Plus_TTA + SymTCN_TTA (hyb 0.50, 0.50)",
         lambda d: d["g_plus_sym"], lambda d: d["t_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tc_sym"], "hybrid", 0.50, 0.50),
        ("6c. SymGBM_Plus_TTA + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_plus_sym"], lambda d: d["t_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tc_sym"], "hybrid", 0.55, 0.50),
        ("7c. 50/50(SymGBM_v1ctl, SymGBM_Plus) + SymTCN_TTA (lin 0.55)",
         lambda d: 0.5 * (d["g_sym"] + d["g_plus_sym"]), lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_sym"] + d["tr_plus_sym"]), lambda d: d["tc_sym"], "linear", 0.55, 0.50),
        ("8c. 50/50(SymGBM_v1ctl, SymGBM_Plus) + SymTCN_TTA (hyb 0.55)",
         lambda d: 0.5 * (d["g_sym"] + d["g_plus_sym"]), lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_sym"] + d["tr_plus_sym"]), lambda d: d["tc_sym"], "hybrid", 0.55, 0.50),
        ("9c. 3-GBM(v4_TTA + Sym_v1ctl + Sym_Plus) + SymTCN_TTA (hyb 0.55)",
         lambda d: (d["g_tta"] + d["g_sym"] + d["g_plus_sym"]) / 3.0, lambda d: d["t_sym"],
         lambda d: (d["tr_orig"] + d["tr_sym"] + d["tr_plus_sym"]) / 3.0, lambda d: d["tc_sym"], "hybrid", 0.55, 0.50),
    ]
    for name, fg, ft, ftr, ftc, bmode, wg, wc in ens_configs:
        scores = []
        for T in MATCH_IDS:
            d = oof_all[T]
            p = blend_proba(fg(d), ft(d), w_gbm=wg, mode=bmode)
            c = wc * ftr(d) + (1.0 - wc) * ftc(d)
            pred = decode_v4(p, sportec[T].half, fold_dec_std[T], p_change=c)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(f"    {name:62s}: mean={arr.mean():.4f} std={arr.std():.4f} | folds={[round(x, 4) for x in scores]}", flush=True)

    # 4. Train Full 7-Match symgbm_plus_full.pkl and symtrans_plus_full.pkl
    print("\n--- Step 4: Training Full 7-Match symgbm_plus_full.pkl & symtrans_plus_full.pkl ---", flush=True)
    gbm_full_path = out_dir / "symgbm_plus_full.pkl"
    tr_full_path = out_dir / "symtrans_plus_full.pkl"
    if gbm_full_path.exists() and tr_full_path.exists():
        with open(gbm_full_path, "rb") as f:
            symgbm_plus_full = pickle.load(f)
        with open(tr_full_path, "rb") as f:
            symtrans_plus_full = pickle.load(f)
        print("  Loaded symgbm_plus_full.pkl & symtrans_plus_full.pkl from cache.", flush=True)
    else:
        tf_full = time.time()
        Xs_all, ys_all, hs_all, gs_all = [], [], [], []
        Xs_mir, ys_mir, gs_mir = [], [], []
        for i, mid in enumerate(MATCH_IDS):
            m = sportec[mid]
            for key, _, _ in TRAIN_VARIANTS:
                Xs_all.append(train_X[(mid, key)])
                ys_all.append(m.labels)
                hs_all.append(m.half)
                gs_all.append(i)
                Xs_mir.append(train_X_mir[(mid, key)])
                ys_mir.append(swap_labels(m.labels))
                gs_mir.append(len(MATCH_IDS) + i)
        Xs_sym = Xs_all + Xs_mir
        ys_sym = ys_all + ys_mir
        gs_sym = gs_all + gs_mir
        ct_all = [change_targets(y, h) for y, h in zip(ys_all, hs_all)]
        symgbm_plus_full = TwoHeadGBM(seed=0).fit(Xs_sym, ys_sym, gs_sym)
        symtrans_plus_full = TransitionGBM(seed=0).fit(Xs_sym, ct_all + ct_all)
        with open(gbm_full_path, "wb") as f:
            pickle.dump(symgbm_plus_full, f)
        with open(tr_full_path, "wb") as f:
            pickle.dump(symtrans_plus_full, f)
        print(f"  Trained & saved symgbm_plus_full.pkl & symtrans_plus_full.pkl in {time.time()-tf_full:.1f}s", flush=True)

    # 5. Evaluate on Judge 2 (Metrica Sample 3 games) and Judge 3 (BAR-ATH Unrepaired & Broadcast-Repaired)
    print("\n--- Step 5: Judge 2 (Metrica Sample 3 Games) & Judge 3 (BAR-ATH Unrepaired & Repaired) ---", flush=True)
    models_v4 = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())
    gbm_v4 = models_v4["gbm"]
    trans_v4 = models_v4["trans"]
    tcn_v4 = models_v4["tcn"]
    with open(sym_dir / "symgbm_full.pkl", "rb") as f:
        symgbm_v1ctl = pickle.load(f)
    with open(sym_dir / "symtrans_full.pkl", "rb") as f:
        symtrans_v1ctl = pickle.load(f)
    with open(sym_dir / "symtcn_full.pkl", "rb") as f:
        symtcn_full = pickle.load(f)

    dec_gbm = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not k.startswith("_")})
    dec_ens = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})

    sample_matches, sample_preds = {}, {}
    for gid in GAMES:
        ms = load_sample(gid)
        sample_matches[gid] = ms
        Xs_base = build_base_features(ms, f"sample_{gid}_raw", "v1ctl")
        Xs_plus = build_v1ctl_plus(ms, f"sample_{gid}_raw")
        pt0, ct0 = tcn_v4.predict(Xs_base, ms.half)
        pt_tta, ct_tta = predict_tcn_tta(tcn_v4, Xs_base, ms.half, ms.pitch)
        pt_sym, ct_sym = predict_tcn_tta(symtcn_full, Xs_base, ms.half, ms.pitch)
        sample_preds[gid] = {
            "g_orig": gbm_v4.predict_proba(Xs_base),
            "g_tta": predict_gbm_tta(gbm_v4, Xs_base, ms.half, ms.pitch),
            "g_sym": predict_gbm_tta(symgbm_v1ctl, Xs_base, ms.half, ms.pitch),
            "g_plus_orig": symgbm_plus_full.predict_proba(Xs_plus),
            "g_plus_sym": predict_gbm_plus_tta(symgbm_plus_full, Xs_plus, ms.half, ms.pitch),
            "tr_orig": trans_v4.predict(Xs_base),
            "tr_sym": predict_trans_tta(symtrans_v1ctl, Xs_base, ms.half, ms.pitch),
            "tr_plus_sym": predict_trans_plus_tta(symtrans_plus_full, Xs_plus, ms.half, ms.pitch),
            "t_orig": pt0,
            "tc_orig": ct0,
            "t_tta": pt_tta,
            "tc_tta": ct_tta,
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

    def eval_atd_bundle(m_obj, key_str):
        Xa_base = build_base_features(m_obj, key_str, "v1ctl")
        Xa_plus = build_v1ctl_plus(m_obj, key_str)
        pt0, ct0 = tcn_v4.predict(Xa_base, m_obj.half)
        pt_tta, ct_tta = predict_tcn_tta(tcn_v4, Xa_base, m_obj.half, m_obj.pitch)
        pt_sym, ct_sym = predict_tcn_tta(symtcn_full, Xa_base, m_obj.half, m_obj.pitch)
        return {
            "g_orig": gbm_v4.predict_proba(Xa_base),
            "g_tta": predict_gbm_tta(gbm_v4, Xa_base, m_obj.half, m_obj.pitch),
            "g_sym": predict_gbm_tta(symgbm_v1ctl, Xa_base, m_obj.half, m_obj.pitch),
            "g_plus_orig": symgbm_plus_full.predict_proba(Xa_plus),
            "g_plus_sym": predict_gbm_plus_tta(symgbm_plus_full, Xa_plus, m_obj.half, m_obj.pitch),
            "tr_orig": trans_v4.predict(Xa_base),
            "tr_sym": predict_trans_tta(symtrans_v1ctl, Xa_base, m_obj.half, m_obj.pitch),
            "tr_plus_sym": predict_trans_plus_tta(symtrans_plus_full, Xa_plus, m_obj.half, m_obj.pitch),
            "t_orig": pt0,
            "tc_orig": ct0,
            "t_tta": pt_tta,
            "tc_tta": ct_tta,
            "t_sym": pt_sym,
            "tc_sym": ct_sym,
        }

    atd_raw_d = eval_atd_bundle(atd, "atd")
    atd_rep_d = eval_atd_bundle(atd_rep, "atd_rep_True_True_None")

    with open(out_dir / "eval_preds_plus.pkl", "wb") as f:
        pickle.dump({"sample": sample_preds, "atd_raw": atd_raw_d, "atd_rep": atd_rep_d}, f)

    eval_rows = [
        # Standalone GBM-only
        ("A1. v4 GBM_orig only (no TCN)",
         lambda d: d["g_orig"], lambda d: d["g_orig"], lambda d: None, lambda d: None, "linear", 1.0, 1.0, dec_gbm),
        ("A2. SymGBM_TTA (v1ctl) only (no TCN)",
         lambda d: d["g_sym"], lambda d: d["g_sym"], lambda d: None, lambda d: None, "linear", 1.0, 1.0, dec_gbm),
        ("A3. SymGBM_Plus_TTA (v1ctl_plus) only (no TCN)",
         lambda d: d["g_plus_sym"], lambda d: d["g_plus_sym"], lambda d: None, lambda d: None, "linear", 1.0, 1.0, dec_gbm),
        ("A4. SymGBM_Plus_TTA + SymTrans_Plus_TTA (no TCN)",
         lambda d: d["g_plus_sym"], lambda d: d["g_plus_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tr_plus_sym"], "linear", 1.0, 1.0, dec_ens),
        # Ensembles with TCN
        ("B0. v4_ens_mean Baseline (orig_GBM + orig_TCN)",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"], "linear", 0.50, 0.50, dec_ens),
        ("B1. SymGBM_TTA(v1ctl) + SymTCN_TTA (lin 0.55, 0.50)",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"], "linear", 0.55, 0.50, dec_ens),
        ("B2. SymGBM_TTA(v1ctl) + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"], "hybrid", 0.55, 0.50, dec_ens),
        ("B3. SymGBM_Plus_TTA + SymTCN_TTA (lin 0.50, 0.50)",
         lambda d: d["g_plus_sym"], lambda d: d["t_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tc_sym"], "linear", 0.50, 0.50, dec_ens),
        ("B4. SymGBM_Plus_TTA + SymTCN_TTA (lin 0.55, 0.50)",
         lambda d: d["g_plus_sym"], lambda d: d["t_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tc_sym"], "linear", 0.55, 0.50, dec_ens),
        ("B5. SymGBM_Plus_TTA + SymTCN_TTA (hyb 0.50, 0.50)",
         lambda d: d["g_plus_sym"], lambda d: d["t_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tc_sym"], "hybrid", 0.50, 0.50, dec_ens),
        ("B6. SymGBM_Plus_TTA + SymTCN_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_plus_sym"], lambda d: d["t_sym"], lambda d: d["tr_plus_sym"], lambda d: d["tc_sym"], "hybrid", 0.55, 0.50, dec_ens),
        ("B7. 50/50(SymGBM_v1ctl, SymGBM_Plus) + SymTCN_TTA (lin 0.55)",
         lambda d: 0.5 * (d["g_sym"] + d["g_plus_sym"]), lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_sym"] + d["tr_plus_sym"]), lambda d: d["tc_sym"], "linear", 0.55, 0.50, dec_ens),
        ("B8. 50/50(SymGBM_v1ctl, SymGBM_Plus) + SymTCN_TTA (hyb 0.55)",
         lambda d: 0.5 * (d["g_sym"] + d["g_plus_sym"]), lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_sym"] + d["tr_plus_sym"]), lambda d: d["tc_sym"], "hybrid", 0.55, 0.50, dec_ens),
        ("B9. Grand Blend (SymGBM_v1ctl+SymGBM_Plus + v4_TCN_TTA+SymTCN_TTA, hyb 0.55)",
         lambda d: 0.5 * (d["g_sym"] + d["g_plus_sym"]), lambda d: 0.5 * (d["t_tta"] + d["t_sym"]),
         lambda d: 0.5 * (d["tr_sym"] + d["tr_plus_sym"]), lambda d: 0.5 * (d["tc_tta"] + d["tc_sym"]), "hybrid", 0.55, 0.50, dec_ens),
    ]

    for vname, fg, ft, ftr, ftc, bmode, wg, wc, dec in eval_rows:
        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            d = sample_preds[gid]
            p = fg(d) if wg == 1.0 else blend_proba(fg(d), ft(d), w_gbm=wg, mode=bmode)
            tr = ftr(d)
            c = None if tr is None else (tr if wc == 1.0 else wc * tr + (1.0 - wc) * ftc(d))
            pred = decode_v4(p, ms.half, dec, p_change=c)
            s_raw.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

        # BAR-ATH unrepaired
        p_raw = fg(atd_raw_d) if wg == 1.0 else blend_proba(fg(atd_raw_d), ft(atd_raw_d), w_gbm=wg, mode=bmode)
        tr_r = ftr(atd_raw_d)
        c_raw = None if tr_r is None else (tr_r if wc == 1.0 else wc * tr_r + (1.0 - wc) * ftc(atd_raw_d))
        pred_raw = decode_v4(p_raw, half_dec, dec, p_change=c_raw)
        pred_raw[second_start:kick2] = DEAD
        pred_raw[~atd.graded] = DEAD
        pub_raw = metrics.score_public(pred_raw, bootstrap_n=100)
        pl_raw = plausibility(pred_raw, atd.half, atd.graded)
        pct_h_raw = 100.0 * float(np.mean(pred_raw[atd.graded & (pred_raw != DEAD)] == 0))

        # BAR-ATH broadcast-repaired
        p_rep = fg(atd_rep_d) if wg == 1.0 else blend_proba(fg(atd_rep_d), ft(atd_rep_d), w_gbm=wg, mode=bmode)
        tr_p = ftr(atd_rep_d)
        c_rep = None if tr_p is None else (tr_p if wc == 1.0 else wc * tr_p + (1.0 - wc) * ftc(atd_rep_d))
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

    print(f"\nEXP-C+ completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
