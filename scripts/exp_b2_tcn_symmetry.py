"""EXP-B2: TCN Equivariant TTA & Symmetric TCN Training (MPS GPU) across 7-Fold LOMO, Sample & BAR-ATH.

In EXP-B Step 1, TCN_TTA alone boosted Judge 2 (Metrica Sample) by +1.55 pp (0.8110 -> 0.8265)
and BAR-ATH Public D_F1 from 0.850 -> 0.919 because TCN has 4.05% H<->A asymmetry (2.3x higher than GBM).
Here we:
1. Compute exact 7-fold outer LOMO `tcn_tta` (from outer TCNModel(norm="channel", seed=k)) and
   `symtcn_tta` (from TCNModel trained on X + mirror(X)).
2. Combine `tcn_tta` and `symtcn_tta` with `gbm_outer`, `gbm_tta`, and `symgbm_tta` using our
   EXP-A Hybrid Linear+LogPool blend + inner-tuned Optuna decoders.
3. Train full 7-match `SymTCN` and evaluate on Metrica Sample (3 games) and BAR-ATH.
"""
from __future__ import annotations

from dataclasses import replace
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import optuna

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
from src.features.symmetry import mirror_v1ctl_df, swap_labels
from src.models.tcn import TCNModel
from exp_b_symmetry_eval import blend_proba, predict_tcn_tta

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    t0 = time.time()
    print("=== EXP-B2: TCN Equivariant TTA & SymTCN (MPS GPU) ===", flush=True)
    sportec = load_all()
    cfg = load_config(ROOT / "configs/degrade_v3.json")
    tag = "degrade_v3"
    store = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    out_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_X, train_X_mir, test_X = {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            X_tr = build_features(degrade(m, c, seed), f"sportec_{mid}_{tag}_{key}", "v1ctl")
            train_X[(mid, key)] = X_tr
            train_X_mir[(mid, key)] = mirror_v1ctl_df(X_tr, m.half, m.pitch)
        test_X[mid] = build_features(degrade(m, cfg, 0), f"sportec_{mid}_{tag}_s0", "v1ctl")

    oof_tcn_orig = {}
    oof_tcn_tta = {}
    oof_symtcn_tta = {}

    for k, T in enumerate(MATCH_IDS):
        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        Xs_std = [train_X[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_std = [sportec[mid].labels for mid in train_ids for _ in TRAIN_VARIANTS]
        hs_std = [sportec[mid].half for mid in train_ids for _ in TRAIN_VARIANTS]

        # 1. Standard outer TCN + TTA
        z_cached = np.load(store / T / f"tcn_outer_{T}.npz")
        oof_tcn_orig[T] = (z_cached["proba"], z_cached["change"])

        tcn_std = TCNModel(norm="channel", seed=k).fit(Xs_std, ys_std, hs_std)
        p_tta, c_tta = predict_tcn_tta(tcn_std, test_X[T], sportec[T].half, sportec[T].pitch)
        oof_tcn_tta[T] = (p_tta, c_tta)

        # 2. Symmetric outer TCN (trained on X + mirror(X), 15 epochs = 30 effective epochs of 18 matches) + TTA
        Xs_sym = Xs_std + [train_X_mir[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_sym = ys_std + [swap_labels(sportec[mid].labels) for mid in train_ids for _ in TRAIN_VARIANTS]
        hs_sym = hs_std + hs_std
        tcn_sym = TCNModel(norm="channel", epochs=15, seed=k).fit(Xs_sym, ys_sym, hs_sym)
        ps_tta, cs_tta = predict_tcn_tta(tcn_sym, test_X[T], sportec[T].half, sportec[T].pitch)
        oof_symtcn_tta[T] = (ps_tta, cs_tta)

        np.savez_compressed(
            out_dir / f"tcn_sym_{T}.npz",
            tcn_tta_proba=p_tta,
            tcn_tta_change=c_tta,
            symtcn_tta_proba=ps_tta,
            symtcn_tta_change=cs_tta,
        )
        print(f"  Fold {k} ({T}) TCN_std+TTA & SymTCN+TTA trained in {time.time()-tf:.1f}s", flush=True)

    # Evaluate on 7-fold LOMO using inner-tuned decoders
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

    print("\n--- 7-Fold Outer LOMO Results: TCN Symmetry & TTA ---", flush=True)
    for label, get_pt, get_ct, bmode, wg, wc in [
        ("v4_ens_mean (orig GBM + orig TCN)",       lambda T: oof_tcn_orig[T][0],   lambda T: oof_tcn_orig[T][1],   "linear", 0.50, 0.50),
        ("orig GBM + TCN_TTA (ens_mean)",           lambda T: oof_tcn_tta[T][0],    lambda T: oof_tcn_tta[T][1],    "linear", 0.50, 0.50),
        ("orig GBM + SymTCN_TTA (ens_mean)",        lambda T: oof_symtcn_tta[T][0], lambda T: oof_symtcn_tta[T][1], "linear", 0.50, 0.50),
        ("Hyb(0.60,0.75): orig GBM + orig TCN",     lambda T: oof_tcn_orig[T][0],   lambda T: oof_tcn_orig[T][1],   "hybrid", 0.60, 0.75),
        ("Hyb(0.60,0.75): orig GBM + TCN_TTA",      lambda T: oof_tcn_tta[T][0],    lambda T: oof_tcn_tta[T][1],    "hybrid", 0.60, 0.75),
        ("Hyb(0.60,0.75): orig GBM + SymTCN_TTA",   lambda T: oof_symtcn_tta[T][0], lambda T: oof_symtcn_tta[T][1], "hybrid", 0.60, 0.75),
    ]:
        scores = []
        for T in MATCH_IDS:
            pg = np.load(store / T / f"gbm_outer_{T}.npy")
            tr = np.load(store / T / f"trans_outer_{T}.npy")
            pt = get_pt(T)
            ct = get_ct(T)
            p = blend_proba(pg, pt, w_gbm=wg, mode=bmode)
            c = wc * tr + (1.0 - wc) * ct
            pred = decode_v4(p, sportec[T].half, fold_decoders[T], p_change=c)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(
            f"  {label:38s}: mean={arr.mean():.4f} std={arr.std():.4f} "
            f"| folds={[round(x, 4) for x in scores]}",
            flush=True,
        )

    # Full 7-match SymTCN for Judge 2 (Sample) and Judge 3 (BAR-ATH)
    print("\n--- Training Full 7-Match SymTCN for Sample & BAR-ATH ---", flush=True)
    tf = time.time()
    Xs_all = [train_X[(mid, key)] for mid in MATCH_IDS for key, _, _ in TRAIN_VARIANTS]
    ys_all = [sportec[mid].labels for mid in MATCH_IDS for _ in TRAIN_VARIANTS]
    hs_all = [sportec[mid].half for mid in MATCH_IDS for _ in TRAIN_VARIANTS]
    Xs_all_sym = Xs_all + [train_X_mir[(mid, key)] for mid in MATCH_IDS for key, _, _ in TRAIN_VARIANTS]
    ys_all_sym = ys_all + [swap_labels(sportec[mid].labels) for mid in MATCH_IDS for _ in TRAIN_VARIANTS]
    hs_all_sym = hs_all + hs_all
    symtcn_full = TCNModel(norm="channel", epochs=15, seed=0).fit(Xs_all_sym, ys_all_sym, hs_all_sym)
    with open(out_dir / "symtcn_full.pkl", "wb") as f:
        pickle.dump(symtcn_full, f)
    print(f"  Full SymTCN trained in {time.time()-tf:.1f}s", flush=True)

    print(f"\nEXP-B2 completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
