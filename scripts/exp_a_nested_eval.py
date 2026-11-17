"""EXP-A Full Nested LOMO + Metrica Sample + BAR-ATH Evaluation (Parallelized across 7 folds):
Tests:
1. Hybrid / Geometric probability blending (GBM + Local TCN) and weighted change head (M3 + TCN change)
2. Orthogonalized HSMM parameter space (lam_d_sum, lam_d_asym, delta_d) to fix the (lam_ld, lam_dl) collinearity in plateau()
3. Exact Minimum Bayes Risk (MBR) Consensus Decoding under the official spellF1 utility over top-K inner Optuna decoders
4. Domain-prior dead-ball logit shift (delta_d) on Judge 2 (Metrica Sample) and Judge 3 (BAR-ATH)
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_3_frame_models import plausibility
from s3_1_transition_head import mean_score, plateau, suggest
from s3_7_nested_full import build_features
from s3_11_tcn_local import without_dead_margin
from src.data.common import DEAD
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics


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


def apply_delta_d(proba: np.ndarray, delta_d: float, eps: float = 1e-6) -> np.ndarray:
    if delta_d == 0.0:
        return proba
    lp = np.log(np.clip(proba.astype(np.float64), eps, 1.0))
    lp[:, DEAD] += delta_d
    lp -= lp.max(axis=1, keepdims=True)
    e = np.exp(lp)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def mbr_select(preds: list[np.ndarray], half: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    K = len(preds)
    if K == 1:
        return preds[0]
    if weights is None:
        weights = np.ones(K, dtype=np.float64) / K
    sim = np.eye(K, dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            if np.array_equal(preds[i], preds[j]):
                s = 1.0
            else:
                s = 0.5 * (
                    metrics.score(preds[i], preds[j], half)["spell_f1_mean"]
                    + metrics.score(preds[j], preds[i], half)["spell_f1_mean"]
                )
            sim[i, j] = s
            sim[j, i] = s
    expected_utility = sim @ weights
    best_idx = int(np.argmax(expected_utility))
    return preds[best_idx]


def top_k_params_from_study(st, top_frac: float = 0.10, k_max: int = 11) -> tuple[DecoderParams, list[DecoderParams], np.ndarray]:
    df = st.trials_dataframe(attrs=("value", "params")).dropna().sort_values("value", ascending=False)
    n_top = max(5, int(len(df) * top_frac))
    df_top = df.head(n_top)
    lam_ld_arr = df_top["params_lam_ld"].to_numpy()
    lam_dl_arr = df_top["params_lam_dl"].to_numpy()
    d_sum_med = float(np.median(lam_ld_arr + lam_dl_arr))
    d_rat_med = float(np.median(lam_ld_arr / np.maximum(lam_ld_arr + lam_dl_arr, 1e-6)))
    ortho_center = DecoderParams(
        lam_ha=float(np.median(df_top["params_lam_ha"])),
        lam_ld=d_sum_med * d_rat_med,
        lam_dl=d_sum_med * (1.0 - d_rat_med),
        dmin_team=int(round(float(np.median(df_top["params_dmin_team"])))),
        dmin_dead=int(round(float(np.median(df_top["params_dmin_dead"])))),
        beta_ha=float(np.median(df_top["params_beta_ha"])),
    )
    cands = [ortho_center]
    vals = [float(df_top["value"].median())]
    for _, row in df_top.head(k_max - 1).iterrows():
        cands.append(DecoderParams(
            lam_ha=float(row["params_lam_ha"]),
            lam_ld=float(row["params_lam_ld"]),
            lam_dl=float(row["params_lam_dl"]),
            dmin_team=int(round(float(row["params_dmin_team"]))),
            dmin_dead=int(round(float(row["params_dmin_dead"]))),
            beta_ha=float(row["params_beta_ha"]),
        ))
        vals.append(float(row["value"]))
    w = np.array(vals, dtype=np.float64)
    w = np.exp((w - w.max()) * 50.0)
    w /= w.sum()
    return ortho_center, cands, w


def _run_outer_fold(args_tuple):
    """Worker for parallel outer-fold Optuna tuning + evaluation of std, ortho, and mbr simultaneously."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    k, T, mode, w_gbm, w_c, delta_d, n_trials = args_tuple
    store = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    sportec = load_all()
    others = [m for m in MATCH_IDS if m != T]
    inner_data = {}
    for J in others:
        g = np.load(store / T / f"gbm_inner_{J}.npy")
        tr = np.load(store / T / f"trans_inner_{J}.npy")
        z = np.load(store / T / f"tcn_inner_{J}.npz")
        p = apply_delta_d(blend_proba(g, z["proba"], w_gbm=w_gbm, mode=mode), delta_d)
        c = w_c * tr + (1.0 - w_c) * z["change"]
        inner_data[J] = (p, sportec[J].labels, sportec[J].half, c)

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
    st.optimize(lambda t: mean_score(suggest(t), inner_data, others), n_trials=n_trials)

    g_out = np.load(store / T / f"gbm_outer_{T}.npy")
    tr_out = np.load(store / T / f"trans_outer_{T}.npy")
    z_out = np.load(store / T / f"tcn_outer_{T}.npz")
    p_out = apply_delta_d(blend_proba(g_out, z_out["proba"], w_gbm=w_gbm, mode=mode), delta_d)
    c_out = w_c * tr_out + (1.0 - w_c) * z_out["change"]

    prm_std = plateau(st)
    prm_ortho, cands, weights = top_k_params_from_study(st, top_frac=0.10, k_max=11)

    pred_std = decode_v4(p_out, sportec[T].half, prm_std, p_change=c_out)
    pred_ortho = decode_v4(p_out, sportec[T].half, prm_ortho, p_change=c_out)
    preds_mbr = [decode_v4(p_out, sportec[T].half, prm, p_change=c_out) for prm in cands]
    pred_mbr = mbr_select(preds_mbr, sportec[T].half, weights)

    y_T = sportec[T].labels
    h_T = sportec[T].half
    return (
        T,
        metrics.score(y_T, pred_std, h_T)["spell_f1_mean"],
        metrics.score(y_T, pred_ortho, h_T)["spell_f1_mean"],
        metrics.score(y_T, pred_mbr, h_T)["spell_f1_mean"],
    )


def main():
    t0 = time.time()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    store = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    sportec = load_all()
    labels = {m: sportec[m].labels for m in MATCH_IDS}
    halves = {m: sportec[m].half for m in MATCH_IDS}

    print("=== Step 1: Parallel Nested LOMO Evaluation (150 trials/fold, 7 workers) ===", flush=True)
    # Distinct (tag, mode, w_gbm, w_c, delta_d) combinations — each produces std, ortho, and mbr!
    base_configs = [
        ("v4_ens_mean", "linear", 0.50, 0.50, 0.0),
        ("v4_ens_m3",   "linear", 0.50, 1.00, 0.0),
        ("hyb_60_50",   "hybrid", 0.60, 0.50, 0.0),
        ("hyb_60_75",   "hybrid", 0.60, 0.75, 0.0),
        ("hyb_60_100",  "hybrid", 0.60, 1.00, 0.0),
        ("geom_65_75",  "geom",   0.65, 0.75, 0.0),
    ]

    lomo_summary = {}
    with ProcessPoolExecutor(max_workers=7) as ex:
        for tag, mode, w_gbm, w_c, delta_d in base_configs:
            tc = time.time()
            jobs = [(k, T, mode, w_gbm, w_c, delta_d, 150) for k, T in enumerate(MATCH_IDS)]
            res_list = list(ex.map(_run_outer_fold, jobs))
            f_std = {T: s1 for T, s1, _, _ in res_list}
            f_ort = {T: s2 for T, _, s2, _ in res_list}
            f_mbr = {T: s3 for T, _, _, s3 in res_list}
            for sub, fd in [("std", f_std), ("ortho", f_ort), ("mbr", f_mbr)]:
                full_name = f"{tag}_{sub}"
                m_val = float(np.mean(list(fd.values())))
                lomo_summary[full_name] = (m_val, fd)
                print(
                    f"  {full_name:18s} ({time.time()-tc:4.1f}s) -> Nested LOMO = {m_val:.4f} | "
                    f"folds: {[round(fd[m], 4) for m in MATCH_IDS]}",
                    flush=True,
                )

    print("\n=== Step 2: Judge 2 (Metrica Sample) & Judge 3 (BAR-ATH) with Full v4 CPU Models ===", flush=True)
    import torch
    from src.models import tcn as tcn_mod
    torch.use_deterministic_algorithms(True)
    tcn_mod._device = lambda: torch.device("cpu")
    models = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())

    sample_preds_raw = {}
    for g in GAMES:
        ms = load_sample(g)
        X = build_features(ms, f"sample_{g}", "v1ctl")
        p_gbm = models["gbm"].predict_proba(X)
        p_tcn, c_tcn = models["tcn"].predict(X, ms.half)
        p_tr = models["trans"].predict(X)
        sample_preds_raw[g] = (ms, p_gbm, p_tr, p_tcn, c_tcn)

    atd = load_atd()
    Xa = build_features(atd, "atd", "v1ctl")
    atd_gbm = models["gbm"].predict_proba(Xa)
    atd_tcn, atd_ctcn = models["tcn"].predict(Xa, atd.half)
    atd_tr = models["trans"].predict(Xa)
    half_dec = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])

    raw_outer = {}
    for T in MATCH_IDS:
        g = np.load(store / T / f"gbm_outer_{T}.npy")
        tr = np.load(store / T / f"trans_outer_{T}.npy")
        z = np.load(store / T / f"tcn_outer_{T}.npz")
        raw_outer[T] = (g, tr, z["proba"], z["change"])

    for tag, mode, w_gbm, w_c, delta_d in base_configs:
        outer_all = {}
        for T in MATCH_IDS:
            g, tr, tp, tc = raw_outer[T]
            p = apply_delta_d(blend_proba(g, tp, w_gbm=w_gbm, mode=mode), delta_d)
            c = w_c * tr + (1.0 - w_c) * tc
            outer_all[T] = (p, labels[T], halves[T], c)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
        st.optimize(lambda t: mean_score(suggest(t), outer_all, list(MATCH_IDS)), n_trials=150)

        ortho_prm, cands, weights = top_k_params_from_study(st, top_frac=0.10, k_max=11)
        std_prm = plateau(st)

        for sub, use_ortho, use_mbr in [("std", False, False), ("ortho", True, False), ("mbr", True, True)]:
            full_name = f"{tag}_{sub}"
            prm = ortho_prm if use_ortho else std_prm
            s_scores, s_nod_scores = [], []
            for g in GAMES:
                ms, p_gbm, p_tr, p_tcn, c_tcn = sample_preds_raw[g]
                p = apply_delta_d(blend_proba(p_gbm, p_tcn, w_gbm=w_gbm, mode=mode), delta_d)
                c = w_c * p_tr + (1.0 - w_c) * c_tcn
                if use_mbr:
                    preds = [decode_v4(p, ms.half, cand, p_change=c) for cand in cands]
                    pred = mbr_select(preds, ms.half, weights)
                else:
                    pred = decode_v4(p, ms.half, prm, p_change=c)
                s_scores.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
                s_nod_scores.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

            p_a = apply_delta_d(blend_proba(atd_gbm, atd_tcn, w_gbm=w_gbm, mode=mode), delta_d)
            c_a = w_c * atd_tr + (1.0 - w_c) * atd_ctcn
            if use_mbr:
                preds_a = [decode_v4(p_a, half_dec, cand, p_change=c_a) for cand in cands]
                pred_a = mbr_select(preds_a, half_dec, weights)
            else:
                pred_a = decode_v4(p_a, half_dec, prm, p_change=c_a)
            pred_a[second_start:kick2] = DEAD
            pred_a[~atd.graded] = DEAD
            pub = metrics.score_public(pred_a, bootstrap_n=200)
            pl = plausibility(pred_a, atd.half, atd.graded)
            lomo_val = lomo_summary[full_name][0]
            print(
                f"  {full_name:18s} | LOMO={lomo_val:.4f} | Sample={np.mean(s_scores):.4f} (noD={np.mean(s_nod_scores):.4f}) "
                f"| Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
                f"| BAR-ATH pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}%",
                flush=True,
            )

    # Also test domain-calibrated delta_d on BAR-ATH and Sample for the best hybrid blend!
    print("\n=== Step 3: Dead-Ball Logit Shift (delta_d) Sensitivity on Sample & BAR-ATH (hyb_60_75_std) ===", flush=True)
    outer_all = {}
    for T in MATCH_IDS:
        g, tr, tp, tc = raw_outer[T]
        outer_all[T] = (blend_proba(g, tp, w_gbm=0.60, mode="hybrid"), labels[T], halves[T], 0.75 * tr + 0.25 * tc)
    st_best = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
    st_best.optimize(lambda t: mean_score(suggest(t), outer_all, list(MATCH_IDS)), n_trials=150)
    prm_best = plateau(st_best)

    for dd in [-0.6, -0.4, -0.25, -0.15, 0.0, 0.15]:
        s_scores, s_nod_scores = [], []
        for g in GAMES:
            ms, p_gbm, p_tr, p_tcn, c_tcn = sample_preds_raw[g]
            p = apply_delta_d(blend_proba(p_gbm, p_tcn, w_gbm=0.60, mode="hybrid"), dd)
            c = 0.75 * p_tr + 0.25 * c_tcn
            pred = decode_v4(p, ms.half, prm_best, p_change=c)
            s_scores.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod_scores.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])
        p_a = apply_delta_d(blend_proba(atd_gbm, atd_tcn, w_gbm=0.60, mode="hybrid"), dd)
        c_a = 0.75 * atd_tr + 0.25 * atd_ctcn
        pred_a = decode_v4(p_a, half_dec, prm_best, p_change=c_a)
        pred_a[second_start:kick2] = DEAD
        pred_a[~atd.graded] = DEAD
        pub = metrics.score_public(pred_a, bootstrap_n=100)
        pl = plausibility(pred_a, atd.half, atd.graded)
        print(
            f"  delta_d={dd:+5.2f}     | Sample={np.mean(s_scores):.4f} (noD={np.mean(s_nod_scores):.4f}) "
            f"| Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
            f"| BAR-ATH pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}%",
            flush=True,
        )

    print(f"\nDone in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
