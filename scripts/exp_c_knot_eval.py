"""EXP-C: Knot-Anchored Physics & Ghost-Slot Filtering Evaluation.

Tests:
1. Inference-time Ghost-Slot & Spike-Velocity Filtering on BAR-ATH & Metrica Sample.
2. Feature Set `v1ctlk` (`v1ctl` + 26 Knot-Anchored Physics features from `src/features/knot_physics.py`):
   - Trained under `degrade_v3` and under a Mixed Knot-Consistent Simulator (`s1=degrade_v3`, `s2=degrade_v3+interp(dev=1.5m)`, `static05_s1=degrade_v3`)
   - Evaluated on:
     a) Judge 1: 7-fold LOMO on `degrade_v3` (and on `degrade_v5` interpolated ball)
     b) Judge 2: Metrica Sample (3 games, 90% piecewise-linear ball!) — both GBM-only and Ensembled with v4 TCN
     c) Judge 3: BAR-ATH full match plausibility + public 5-min sanity gate
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
from src.data.common import DEAD
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from src.features.ballq import cached_ballq
from src.features.knot_physics import cached_knot_physics, filter_ghost_and_switch_slots
from src.models.gbm import TwoHeadGBM
from src.models.transition import TransitionGBM, change_targets


def build_features_ext(m, key: str, fs: str) -> pd.DataFrame:
    X = build_base_features(m, key, "v1ctl")
    if "q" in fs:
        X = pd.concat([X, cached_ballq(m, key)], axis=1)
    if "k" in fs:
        X = pd.concat([X, cached_knot_physics(m, key)], axis=1)
    return X


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


def main() -> None:
    t0 = time.time()
    import torch
    from src.models import tcn as tcn_mod
    torch.use_deterministic_algorithms(True)
    tcn_mod._device = lambda: torch.device("cpu")

    print("=== EXP-C: Knot-Anchored Physics & Ghost-Slot Filtering ===", flush=True)

    sportec = load_all()
    sample = {g: load_sample(g) for g in GAMES}
    atd = load_atd()
    half_dec = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])

    models_v4 = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())
    dec_gbm = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not k.startswith("_")})
    dec_ens = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})
    ref_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl"
    store_tcn = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"

    # Precompute v4 TCN predictions on Sample and BAR-ATH so we can test Ensembles!
    sample_tcn = {}
    for g, ms in sample.items():
        X_s = build_base_features(ms, f"sample_{g}", "v1ctl")
        p_tcn, c_tcn = models_v4["tcn"].predict(X_s, ms.half)
        c_tr = models_v4["trans"].predict(X_s)
        sample_tcn[g] = (p_tcn, c_tcn, c_tr)

    Xa_base = build_base_features(atd, "atd", "v1ctl")
    atd_tcn, atd_ctcn = models_v4["tcn"].predict(Xa_base, atd.half)
    atd_ctr = models_v4["trans"].predict(Xa_base)

    # Build Knot Physics features for Sportec (degrade_v3 and knot-consistent interp), Sample, and BAR-ATH
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v5_real = replace(cfg_v3, ball_interp=True, interp_dev_max=1.5)  # Knot-consistent 1.5m max deviation!

    print("\n--- Step 1: Computing & Caching Knot-Physics Features (v1ctlk & v1ctlqk) ---", flush=True)
    TX_v3, testX_v3 = {}, {}
    TX_mix, testX_v5real = {}, {}

    for mid, m in sportec.items():
        for key, over, s in TRAIN_VARIANTS:
            c3 = replace(cfg_v3, **over) if over else cfg_v3
            dm3 = degrade(m, c3, s)
            TX_v3[(mid, key)] = build_features_ext(dm3, f"sportec_{mid}_degrade_v3_{key}", "v1ctlqk")
            if key == "s2":
                # Replace s2 with knot-consistent piecewise-linear ball (dev_max=1.5m)
                cm = replace(cfg_v5_real, **over) if over else cfg_v5_real
                dmm = degrade(m, cm, s)
                TX_mix[(mid, key)] = build_features_ext(dmm, f"sportec_{mid}_degv5real_{key}", "v1ctlqk")
            else:
                TX_mix[(mid, key)] = TX_v3[(mid, key)]
        dm_test3 = degrade(m, cfg_v3, 0)
        testX_v3[mid] = build_features_ext(dm_test3, f"sportec_{mid}_degrade_v3_s0", "v1ctlqk")
        dm_test5 = degrade(m, cfg_v5_real, 0)
        testX_v5real[mid] = build_features_ext(dm_test5, f"sportec_{mid}_degv5real_s0", "v1ctlqk")

    sample_X = {g: build_features_ext(ms, f"sample_{g}", "v1ctlqk") for g, ms in sample.items()}
    atd_X = build_features_ext(atd, "atd", "v1ctlqk")

    base_cols = list(Xa_base.columns)
    knot_cols = [c for c in atd_X.columns if c.startswith(("knot_", "real_d_", "clean_d_", "phantom_"))]
    ballq_cols = [c for c in atd_X.columns if c not in base_cols and c not in knot_cols]
    print(f"  Columns: base v1ctl={len(base_cols)}, ballq={len(ballq_cols)}, knot_physics={len(knot_cols)}", flush=True)

    variants = [
        ("v3_v1ctl (baseline)",     TX_v3,  base_cols),
        ("v3_v1ctlk (+knot)",       TX_v3,  base_cols + knot_cols),
        ("v3_v1ctlqk (+ballq+knot)", TX_v3, base_cols + ballq_cols + knot_cols),
        ("mix_v1ctlk (s2=interp1.5m + knot)", TX_mix, base_cols + knot_cols),
        ("mix_v1ctlqk (s2=interp1.5m + qk)",  TX_mix, base_cols + ballq_cols + knot_cols),
    ]

    print("\n--- Step 2: 7-Fold LOMO (on degrade_v3 and degrade_v5_real) + Judge 2 (Sample) + Judge 3 (BAR-ATH) ---", flush=True)
    oof_knot_preds = {}
    for vname, TX_dict, cols in variants:
        tv = time.time()
        lomo_v3, lomo_v5r, lomo_ens_v3 = [], [], []
        oof_knot_preds[vname] = {}

        for k, T in enumerate(MATCH_IDS):
            m = sportec[T]
            params_gbm = DecoderParams(**json.loads((ref_v3 / T / "params_gbm.json").read_text()))
            params_ens = DecoderParams(**json.loads((store_tcn / T / "params_ens_mean.json").read_text()))
            tr_ids = [x for x in MATCH_IDS if x != T]

            if vname == "v3_v1ctl (baseline)":
                p_v3 = np.load(ref_v3 / T / f"gbm_outer_{T}.npy")
                gbm_k = None
            else:
                Xs = [TX_dict[(mid, key)][cols] for mid in tr_ids for key, _, _ in TRAIN_VARIANTS]
                ys = [sportec[mid].labels for mid in tr_ids for _ in TRAIN_VARIANTS]
                gs = [i for i, mid in enumerate(tr_ids) for _ in TRAIN_VARIANTS]
                gbm_k = TwoHeadGBM(seed=k).fit(Xs, ys, gs)
                p_v3 = gbm_k.predict_proba(testX_v3[T][cols])

            oof_knot_preds[vname][T] = p_v3
            pred_gbm = decode_v4(p_v3, m.half, params_gbm)
            lomo_v3.append(metrics.score(m.labels, pred_gbm, m.half)["spell_f1_mean"])

            # Also evaluate when blended with v4 OOF Local TCN + Hybrid(0.60, 0.75)!
            z_tcn = np.load(store_tcn / T / f"tcn_outer_{T}.npz")
            tr_m3 = np.load(store_tcn / T / f"trans_outer_{T}.npy")
            p_hyb = blend_proba(p_v3, z_tcn["proba"], w_gbm=0.60, mode="hybrid")
            c_hyb = 0.75 * tr_m3 + 0.25 * z_tcn["change"]
            pred_ens = decode_v4(p_hyb, m.half, params_ens, p_change=c_hyb)
            lomo_ens_v3.append(metrics.score(m.labels, pred_ens, m.half)["spell_f1_mean"])

            # Evaluate on interpolated ball test match (degrade_v5_real)
            if gbm_k is not None:
                p_v5r = gbm_k.predict_proba(testX_v5real[T][cols])
                lomo_v5r.append(metrics.score(m.labels, decode_v4(p_v5r, m.half, params_gbm), m.half)["spell_f1_mean"])
            else:
                # Fit baseline on base_cols to test on v5real
                Xs0 = [TX_v3[(mid, key)][base_cols] for mid in tr_ids for key, _, _ in TRAIN_VARIANTS]
                ys0 = [sportec[mid].labels for mid in tr_ids for _ in TRAIN_VARIANTS]
                gs0 = [i for i, mid in enumerate(tr_ids) for _ in TRAIN_VARIANTS]
                p_v5r = TwoHeadGBM(seed=k).fit(Xs0, ys0, gs0).predict_proba(testX_v5real[T][base_cols])
                lomo_v5r.append(metrics.score(m.labels, decode_v4(p_v5r, m.half, params_gbm), m.half)["spell_f1_mean"])

        # Fit full 7-match model (seed=99, exact protocol of s4_5 and make_submission)
        if vname == "v3_v1ctl (baseline)":
            full_gbm = models_v4["gbm"]
        else:
            Xs_all = [TX_dict[(mid, key)][cols] for mid in MATCH_IDS for key, _, _ in TRAIN_VARIANTS]
            ys_all = [sportec[mid].labels for mid in MATCH_IDS for _ in TRAIN_VARIANTS]
            gs_all = [i for i, mid in enumerate(MATCH_IDS) for _ in TRAIN_VARIANTS]
            full_gbm = TwoHeadGBM(seed=99).fit(Xs_all, ys_all, gs_all)

        # Judge 2: Metrica Sample (both GBM-only and Ensembled with TCN!)
        s_gbm_raw, s_gbm_nod = [], []
        s_ens_raw, s_ens_nod = [], []
        for g, ms in sample.items():
            pg = full_gbm.predict_proba(sample_X[g][cols])
            pred_g = decode_v4(pg, ms.half, dec_gbm)
            s_gbm_raw.append(metrics.score(ms.labels, pred_g, ms.half)["spell_f1_mean"])
            s_gbm_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred_g, ms.half)["spell_f1_mean"])

            p_tcn, c_tcn, c_tr = sample_tcn[g]
            p_ens = blend_proba(pg, p_tcn, w_gbm=0.60, mode="hybrid")
            c_ens = 0.75 * c_tr + 0.25 * c_tcn
            pred_e = decode_v4(p_ens, ms.half, dec_ens, p_change=c_ens)
            s_ens_raw.append(metrics.score(ms.labels, pred_e, ms.half)["spell_f1_mean"])
            s_ens_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred_e, ms.half)["spell_f1_mean"])

        # Judge 3: BAR-ATH
        pg_a = full_gbm.predict_proba(atd_X[cols])
        p_ens_a = blend_proba(pg_a, atd_tcn, w_gbm=0.60, mode="hybrid")
        c_ens_a = 0.75 * atd_ctr + 0.25 * atd_ctcn
        pred_a = decode_v4(p_ens_a, half_dec, dec_ens, p_change=c_ens_a)
        pred_a[second_start:kick2] = DEAD
        pred_a[~atd.graded] = DEAD
        pub = metrics.score_public(pred_a, bootstrap_n=100)
        pl = plausibility(pred_a, atd.half, atd.graded)

        print(
            f"  {vname:34s} ({time.time()-tv:4.1f}s)\n"
            f"    -> LOMO_gbm(v3)={np.mean(lomo_v3):.4f} | LOMO_gbm(v5real)={np.mean(lomo_v5r):.4f} | LOMO_Ens(v3)={np.mean(lomo_ens_v3):.4f}\n"
            f"    -> Sample_GBM={np.mean(s_gbm_raw):.4f} (noD={np.mean(s_gbm_nod):.4f}) | Sample_Ens={np.mean(s_ens_raw):.4f} (noD={np.mean(s_ens_nod):.4f})\n"
            f"    -> BAR-ATH Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) | pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}%",
            flush=True,
        )

    # Save OOF predictions of v1ctlk and mix_v1ctlk for multi-model blending!
    out_dir = ROOT / "experiments" / "outputs" / "exp_c_knot_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    for T in MATCH_IDS:
        np.savez_compressed(
            out_dir / f"knot_{T}.npz",
            **{vname.split()[0]: oof_knot_preds[vname][T] for vname, _, _ in variants},
        )

    print(f"\nEXP-C completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
