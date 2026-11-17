"""EXP-B: H <-> A Symmetry & Pitch-180 Equivariance (TTA + Symmetric Training).

Evaluates:
1. Algebraic mirror involution & exactness check (mirror_v1ctl_df).
2. Zero-shot Inference TTA on existing v4 CPU models (models.pkl) across:
   - Judge 2: Metrica Sample (3 games, raw & without D+-3s)
   - Judge 3: BAR-ATH full match plausibility (%H vs %A symmetry, poss/min, %D) + public 5-min sanity gate
3. 7-Fold Outer LOMO on Sportec (degrade_v3 and degrade_v5):
   - Standard GBM vs Standard GBM + TTA
   - Symmetric GBM (trained on X + mirror(X), doubling effective match diversity from 6 to 12 matches) + TTA
   - Combined with Local TCN + Hybrid Blend (EXP-A + EXP-B synergy!)
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
from s3_1_transition_head import mean_score, plateau, suggest
from s3_7_nested_full import TRAIN_VARIANTS, build_features
from s3_11_tcn_local import without_dead_margin
from src.data.common import DEAD
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from src.features.symmetry import mirror_v1ctl_df, swap_labels, swap_proba
from src.models.gbm import TwoHeadGBM
from src.models.transition import TransitionGBM, change_targets


def blend_proba(p_gbm: np.ndarray, p_tcn: np.ndarray, w_gbm: float = 0.60, mode: str = "hybrid", eps: float = 1e-6) -> np.ndarray:
    if mode == "linear":
        return (w_gbm * p_gbm + (1.0 - w_gbm) * p_tcn).astype(np.float32)
    lp = w_gbm * np.log(np.clip(p_gbm, eps, 1.0)) + (1.0 - w_gbm) * np.log(np.clip(p_tcn, eps, 1.0))
    lp -= lp.max(axis=1, keepdims=True)
    e = np.exp(lp)
    p_geo = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    if mode == "geom":
        return p_geo
    elif mode == "hybrid":
        p_lin = (w_gbm * p_gbm + (1.0 - w_gbm) * p_tcn).astype(np.float32)
        return (0.5 * p_lin + 0.5 * p_geo).astype(np.float32)
    raise ValueError(mode)


def predict_gbm_tta(gbm: TwoHeadGBM, X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    p_orig = gbm.predict_proba(X)
    X_mir = mirror_v1ctl_df(X, half, pitch)
    p_mir = swap_proba(gbm.predict_proba(X_mir))
    out = 0.5 * p_orig + 0.5 * p_mir
    out /= out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


def predict_tcn_tta(tcn, X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    p_orig, c_orig = tcn.predict(X, half)
    X_mir = mirror_v1ctl_df(X, half, pitch)
    p_mir_raw, c_mir = tcn.predict(X_mir, half)
    p_mir = swap_proba(p_mir_raw)
    p_out = (0.5 * p_orig + 0.5 * p_mir).astype(np.float32)
    p_out /= p_out.sum(axis=1, keepdims=True)
    c_out = (0.5 * c_orig + 0.5 * c_mir).astype(np.float32)
    return p_out, c_out


def predict_trans_tta(tr: TransitionGBM, X: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    c_orig = tr.predict(X)
    X_mir = mirror_v1ctl_df(X, half, pitch)
    c_mir = tr.predict(X_mir)
    return (0.5 * c_orig + 0.5 * c_mir).astype(np.float32)


def main() -> None:
    t0 = time.time()
    import torch
    from src.models import tcn as tcn_mod
    torch.use_deterministic_algorithms(True)
    tcn_mod._device = lambda: torch.device("cpu")

    print("=== EXP-B: H <-> A Symmetry & Pitch-180 Equivariance ===", flush=True)

    models = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())
    dec_mean = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})
    dec_m3 = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_m3.json").read_text()).items() if not k.startswith("_")})

    # Step 1: Test Zero-Shot TTA on Judge 2 (Metrica Sample) and Judge 3 (BAR-ATH)
    print("\n--- Step 1: Zero-Shot Inference TTA on Existing v4 CPU Models (Sample & BAR-ATH) ---", flush=True)
    sample_data = {}
    for g in GAMES:
        ms = load_sample(g)
        X = build_features(ms, f"sample_{g}", "v1ctl")
        if g == GAMES[0]:
            X_m = mirror_v1ctl_df(X, ms.half, ms.pitch)
            X_m2 = mirror_v1ctl_df(X_m, ms.half, ms.pitch)
            err = float(np.nanmax(np.abs(X.to_numpy(dtype=np.float32) - X_m2.to_numpy(dtype=np.float32))))
            print(f"  [Sanity] mirror(mirror(X)) involution max error = {err:.2e}", flush=True)
        p_g0 = models["gbm"].predict_proba(X)
        p_g_tta = predict_gbm_tta(models["gbm"], X, ms.half, ms.pitch)
        p_t0, c_t0 = models["tcn"].predict(X, ms.half)
        p_t_tta, c_t_tta = predict_tcn_tta(models["tcn"], X, ms.half, ms.pitch)
        c_m0 = models["trans"].predict(X)
        c_m_tta = predict_trans_tta(models["trans"], X, ms.half, ms.pitch)
        sample_data[g] = (ms, p_g0, p_g_tta, p_t0, p_t_tta, c_t0, c_t_tta, c_m0, c_m_tta)

    atd = load_atd()
    Xa = build_features(atd, "atd", "v1ctl")
    ag0 = models["gbm"].predict_proba(Xa)
    ag_tta = predict_gbm_tta(models["gbm"], Xa, atd.half, atd.pitch)
    at0, act0 = models["tcn"].predict(Xa, atd.half)
    at_tta, act_tta = predict_tcn_tta(models["tcn"], Xa, atd.half, atd.pitch)
    acm0 = models["trans"].predict(Xa)
    acm_tta = predict_trans_tta(models["trans"], Xa, atd.half, atd.pitch)
    half_dec = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])

    # Check H/A symmetry discrepancy before vs after TTA on BAR-ATH!
    diff_g_bar = float(np.mean(np.abs(ag0 - ag_tta)))
    diff_t_bar = float(np.mean(np.abs(at0 - at_tta)))
    print(f"  [Asymmetry Audit on BAR-ATH] Mean |p_orig - p_TTA|: GBM = {diff_g_bar:.4f}, TCN = {diff_t_bar:.4f}", flush=True)

    tta_variants = [
        ("v4_ens_mean (no TTA)",      False, False, "linear", 0.50, 0.50, dec_mean),
        ("v4_ens_m3 (no TTA)",        False, False, "linear", 0.50, 1.00, dec_m3),
        ("GBM_TTA only (ens_mean)",   True,  False, "linear", 0.50, 0.50, dec_mean),
        ("TCN_TTA only (ens_mean)",   False, True,  "linear", 0.50, 0.50, dec_mean),
        ("Both_TTA (ens_mean)",       True,  True,  "linear", 0.50, 0.50, dec_mean),
        ("Both_TTA (ens_m3)",         True,  True,  "linear", 0.50, 1.00, dec_m3),
        ("Both_TTA + Hyb(0.60,mean)", True,  True,  "hybrid", 0.60, 0.50, dec_mean),
        ("Both_TTA + Hyb(0.60,0.75)", True,  True,  "hybrid", 0.60, 0.75, dec_mean),
    ]
    for vname, g_tta, t_tta, bmode, wg, wc, dec in tta_variants:
        s_raw, s_nod = [], []
        for g in GAMES:
            ms, pg0, pg1, pt0, pt1, ct0, ct1, cm0, cm1 = sample_data[g]
            pg = pg1 if g_tta else pg0
            pt = pt1 if t_tta else pt0
            ct = ct1 if t_tta else ct0
            cm = cm1 if g_tta else cm0
            p = blend_proba(pg, pt, w_gbm=wg, mode=bmode)
            c = wc * cm + (1.0 - wc) * ct
            pred = decode_v4(p, ms.half, dec, p_change=c)
            s_raw.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

        pg_a = ag_tta if g_tta else ag0
        pt_a = at_tta if t_tta else at0
        ct_a = act_tta if t_tta else act0
        cm_a = acm_tta if g_tta else acm0
        p_a = blend_proba(pg_a, pt_a, w_gbm=wg, mode=bmode)
        c_a = wc * cm_a + (1.0 - wc) * ct_a
        pred_a = decode_v4(p_a, half_dec, dec, p_change=c_a)
        pred_a[second_start:kick2] = DEAD
        pred_a[~atd.graded] = DEAD
        pub = metrics.score_public(pred_a, bootstrap_n=100)
        pl = plausibility(pred_a, atd.half, atd.graded)
        print(
            f"  {vname:26s} | Sample={np.mean(s_raw):.4f} (noD={np.mean(s_nod):.4f}) "
            f"| Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
            f"| BAR-ATH pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}%",
            flush=True,
        )

    # Step 2: 7-Fold Outer LOMO on Sportec degrade_v3 — Standard GBM vs TTA vs Symmetric-Trained GBM + TTA
    print("\n--- Step 2: 7-Fold Outer LOMO (Sportec degrade_v3): Standard vs TTA vs SymGBM (X + mirror(X)) ---", flush=True)
    sportec = load_all()
    cfg = load_config(ROOT / "configs/degrade_v3.json")
    tag = "degrade_v3"
    store = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"

    train_X, train_X_mir, test_X = {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            X_tr = build_features(degrade(m, c, seed), f"sportec_{mid}_{tag}_{key}", "v1ctl")
            train_X[(mid, key)] = X_tr
            train_X_mir[(mid, key)] = mirror_v1ctl_df(X_tr, m.half, m.pitch)
        test_X[mid] = build_features(degrade(m, cfg, 0), f"sportec_{mid}_{tag}_s0", "v1ctl")

    # Pre-load inner Optuna plateau decoders per fold so we can compare fairly on each outer fold
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fold_decoders = {}
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        inner_data = {}
        for J in others:
            g = np.load(store / T / f"gbm_inner_{J}.npy")
            tr = np.load(store / T / f"trans_inner_{J}.npy")
            z = np.load(store / T / f"tcn_inner_{J}.npz")
            p = blend_proba(g, z["proba"], w_gbm=0.60, mode="hybrid")
            c = 0.75 * tr + 0.25 * z["change"]
            inner_data[J] = (p, sportec[J].labels, sportec[J].half, c)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
        st.optimize(lambda t: mean_score(suggest(t), inner_data, others), n_trials=100)
        fold_decoders[T] = plateau(st)

    oof_gbm_orig = {}
    oof_gbm_tta = {}
    oof_symgbm_tta = {}
    oof_symtrans_tta = {}

    for k, T in enumerate(MATCH_IDS):
        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        # 1. Load exact outer GBM from store, and also fit standard GBM to get its TTA prediction on test_X[T]
        oof_gbm_orig[T] = np.load(store / T / f"gbm_outer_{T}.npy")
        Xs_std = [train_X[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_std = [sportec[mid].labels for mid in train_ids for _ in TRAIN_VARIANTS]
        gs_std = [i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]
        gbm_std = TwoHeadGBM(seed=k).fit(Xs_std, ys_std, gs_std)
        oof_gbm_tta[T] = predict_gbm_tta(gbm_std, test_X[T], sportec[T].half, sportec[T].pitch)

        # 2. Symmetric GBM: trained on 6 original + 6 mirrored matches (36 match-variants total, 12 groups)
        Xs_sym = Xs_std + [train_X_mir[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_sym = ys_std + [swap_labels(sportec[mid].labels) for mid in train_ids for _ in TRAIN_VARIANTS]
        gs_sym = gs_std + [len(train_ids) + i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]
        gbm_sym = TwoHeadGBM(seed=k).fit(Xs_sym, ys_sym, gs_sym)
        oof_symgbm_tta[T] = predict_gbm_tta(gbm_sym, test_X[T], sportec[T].half, sportec[T].pitch)

        # 3. Symmetric TransitionGBM: trained on original + mirrored matches
        hs_std = [sportec[mid].half for mid in train_ids for _ in TRAIN_VARIANTS]
        ct_std = [change_targets(y, h) for y, h in zip(ys_std, hs_std)]
        tr_sym = TransitionGBM(seed=k).fit(Xs_sym, ct_std + ct_std)
        oof_symtrans_tta[T] = predict_trans_tta(tr_sym, test_X[T], sportec[T].half, sportec[T].pitch)

        print(f"  Fold {k} ({T}) trained in {time.time()-tf:.1f}s", flush=True)

    # Save OOF predictions for downstream combination
    out_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    for T in MATCH_IDS:
        np.savez_compressed(
            out_dir / f"sym_{T}.npz",
            gbm_orig=oof_gbm_orig[T],
            gbm_tta=oof_gbm_tta[T],
            symgbm_tta=oof_symgbm_tta[T],
            symtrans_tta=oof_symtrans_tta[T],
        )

    print("\n--- Step 3: 7-Fold LOMO Results Summary (with inner-tuned decoders) ---", flush=True)
    for label, get_pg, get_tr, bmode, wg, wc in [
        ("GBM_orig only (no TCN)",            lambda T: oof_gbm_orig[T],   lambda T: np.load(store / T / f"trans_outer_{T}.npy"), "linear", 1.00, 1.00),
        ("GBM_orig + TTA only (no TCN)",      lambda T: oof_gbm_tta[T],    lambda T: np.load(store / T / f"trans_outer_{T}.npy"), "linear", 1.00, 1.00),
        ("SymGBM + TTA only (no TCN)",        lambda T: oof_symgbm_tta[T], lambda T: oof_symtrans_tta[T],                         "linear", 1.00, 1.00),
        ("GBM_orig + TCN (hyb 0.60, 0.75)",   lambda T: oof_gbm_orig[T],   lambda T: np.load(store / T / f"trans_outer_{T}.npy"), "hybrid", 0.60, 0.75),
        ("GBM_TTA + TCN (hyb 0.60, 0.75)",    lambda T: oof_gbm_tta[T],    lambda T: np.load(store / T / f"trans_outer_{T}.npy"), "hybrid", 0.60, 0.75),
        ("SymGBM_TTA + TCN (hyb 0.60, 0.75)", lambda T: oof_symgbm_tta[T], lambda T: oof_symtrans_tta[T],                         "hybrid", 0.60, 0.75),
        ("50/50(GBM_orig,SymGBM_TTA) + TCN",  lambda T: 0.5*oof_gbm_orig[T]+0.5*oof_symgbm_tta[T], lambda T: 0.5*np.load(store / T / f"trans_outer_{T}.npy")+0.5*oof_symtrans_tta[T], "hybrid", 0.60, 0.75),
    ]:
        f_scores = {}
        for T in MATCH_IDS:
            z = np.load(store / T / f"tcn_outer_{T}.npz")
            pg = get_pg(T)
            tr = get_tr(T)
            p = pg if wg == 1.0 else blend_proba(pg, z["proba"], w_gbm=wg, mode=bmode)
            c = tr if wc == 1.0 else wc * tr + (1.0 - wc) * z["change"]
            pred = decode_v4(p, sportec[T].half, fold_decoders[T], p_change=c)
            f_scores[T] = metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"]
        m_val = float(np.mean(list(f_scores.values())))
        print(f"  {label:36s} -> LOMO = {m_val:.4f} | folds: {[round(f_scores[m], 4) for m in MATCH_IDS]}", flush=True)

    print(f"\nEXP-B completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
