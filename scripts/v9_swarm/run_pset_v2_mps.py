"""Fast Apple Silicon MPS GPU Trainer & Evaluator for `SymPlayerSetNetV2`.

Leverages the 13 GB of pre-extracted `PlayerSetTensors` already cached in `local_data/cache/pset/`
for zero-overhead tensor loading, plus extracts `atd_reteam_rep` (`D3` jersey re-teamed + broadcast
repaired BAR-ATH) in ~2.5s.

Trains:
- 7-fold outer LOMO `SymPlayerSetNetV2` (`experiments/outputs/v9_swarm/pset_v2_oof/pset_v2_{T}.npz`)
  with:
  1. Line L1 Possession-Duration Inverse Weighting (`poss_weight_alpha=0.45`) + boundary boost (`0.80`)
     + 45% transition-centered crop sampling.
  2. Line L5 4-Directional Action Spotting head (`p4d: [H->A, A->H, Live->D, D->Live]`).
  3. Idea O1 3D Aerial Ball Height head (`p_aerial: P(Z > 1.5m)` and `ball_z_hat`) supervised by
     ground-truth Sportec 3D `m.meta["ball_z"]`, with internal aerial residual gating.
  4. Line D3 online `12 vs 10` cross-team slot swap augmentation (`team_swap_aug_prob=0.50`).
- Full 7-match `SymPlayerSetNetV2` (`symplayerset_v2_full.pkl` & `pset_v2_j23_preds.pkl`) predicting on:
  - Metrica Sample 1, 2, 3
  - `atd_raw`, `atd_rep`, and `atd_reteam_rep` (`D3` jersey re-teamed + broadcast repaired).
"""
from __future__ import annotations

from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from s2_1_degradation import load_config
from s3_7_nested_full import TRAIN_VARIANTS
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.viterbi_hsmm import decode as decode_v4
from src.eval import metrics
from src.models.player_set_net import PlayerSetTensors, cached_player_set_tensors
from src.models.transition import change_targets
from src.v9_swarm.player_set_net_v2 import SymPlayerSetNetV2
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match


def load_atd_reteam():
    atd = load_atd()
    reteam_path = ROOT / "local_data" / "video" / "players_reteam.npz"
    if not reteam_path.exists():
        return atd
    z = np.load(reteam_path)
    P_rt = z["P"].astype(np.float32)
    T = min(len(atd.P), len(P_rt))
    P_new = atd.P.copy()
    P_new[:T] = P_rt[:T]
    return replace(atd, P=P_new)


def load_pst_fast(cache_dir: Path, tag: str, match_fn=None) -> PlayerSetTensors:
    matches = sorted(cache_dir.glob(f"pset_v1_{tag}_*.npz"))
    if matches:
        z = np.load(matches[0])
        return PlayerSetTensors(
            p_h=z["p_h"].astype(np.float32),
            p_a=z["p_a"].astype(np.float32),
            mask_h=z["mask_h"].astype(bool),
            mask_a=z["mask_a"].astype(bool),
            g_anti=z["g_anti"].astype(np.float32),
            g_sym=z["g_sym"].astype(np.float32),
        )
    assert match_fn is not None, f"Missing cache for {tag}"
    return cached_player_set_tensors(match_fn(), cache_dir, tag=tag)


def main():
    t0 = time.time()
    print("=" * 115, flush=True)
    print("WAVE V9 — `SymPlayerSetNetV2` (L1 POSS-WEIGHTED + L5 4-DIR HEAD + O1 3D AERIAL BALL-Z HEAD + D3 RETEAM AUG)", flush=True)
    print("=" * 115, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "v9_swarm" / "pset_v2_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    pset_cache_dir = ROOT / "local_data" / "cache" / "pset"
    pset_cache_dir.mkdir(parents=True, exist_ok=True)

    sportec = load_sportec()
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")

    print("\n--- Step 1: Loading Pre-Cached `PlayerSetTensors` from `local_data/cache/pset/` ---", flush=True)
    t_ext = time.time()
    train_pst_v3: dict[tuple[str, str], PlayerSetTensors] = {}
    train_pst_v6: dict[str, PlayerSetTensors] = {}
    test_pst_v3: dict[str, PlayerSetTensors] = {}
    test_pst_v6: dict[str, PlayerSetTensors] = {}

    need_lomo = any(not (out_dir / f"pset_v2_{T}.npz").exists() for T in MATCH_IDS)
    need_full = not (out_dir / "symplayerset_v2_full.pkl").exists()

    for mid in MATCH_IDS:
        m = sportec[mid]
        if need_lomo or need_full:
            for key, overrides, seed in TRAIN_VARIANTS:
                cfg = replace(cfg_v3, **(overrides or {}))
                train_pst_v3[(mid, key)] = load_pst_fast(
                    pset_cache_dir, f"sportec_{mid}_v3_{key}", lambda m=m, cfg=cfg, seed=seed: degrade(m, cfg, seed)
                )
            train_pst_v6[mid] = load_pst_fast(
                pset_cache_dir, f"sportec_{mid}_v6_s1", lambda m=m: degrade(m, cfg_v6, 11)
            )

        test_pst_v3[mid] = load_pst_fast(
            pset_cache_dir, f"sportec_{mid}_v3_s0", lambda m=m: degrade(m, cfg_v3, 0)
        )
        test_pst_v6[mid] = load_pst_fast(
            pset_cache_dir, f"sportec_{mid}_v6_s0", lambda m=m: degrade(m, cfg_v6, 0)
        )

    print(f"  Loaded all 42 Sportec `PlayerSetTensors` in {time.time() - t_ext:.1f}s", flush=True)

    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    w7a_dir = ROOT / "experiments" / "outputs" / "exp_w7a_oof"
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)

    print("\n--- Step 2A: 7-Fold Outer LOMO Training of `SymPlayerSetNetV2` (Apple Silicon MPS GPU) ---", flush=True)
    for k, T in enumerate(MATCH_IDS):
        f_npz = out_dir / f"pset_v2_{T}.npz"
        if f_npz.exists():
            print(f"  [Fold {k+1}/7 {T}] already cached.", flush=True)
            continue

        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        psts_tr, ys_tr, hs_tr, bzs_tr, ws_tr = [], [], [], [], []
        for mid in train_ids:
            bz_mid = sportec[mid].meta.get("ball_z")
            for key, _, _ in TRAIN_VARIANTS:
                psts_tr.append(train_pst_v3[(mid, key)])
                ys_tr.append(sportec[mid].labels)
                hs_tr.append(sportec[mid].half)
                bzs_tr.append(bz_mid)
                ws_tr.append(1.0)
            psts_tr.append(train_pst_v6[mid])
            ys_tr.append(sportec[mid].labels)
            hs_tr.append(sportec[mid].half)
            bzs_tr.append(bz_mid)
            ws_tr.append(0.80)

        model = SymPlayerSetNetV2(
            channels=64,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            gru_stride=10,
            dropout=0.15,
            lr=1.5e-3,
            weight_decay=1e-4,
            epochs=10,
            crop=1500,
            batch_size=24,
            poss_alpha=0.45,
            boundary_boost=1.6,
            change_loss_weight=0.35,
            b4d_loss_weight=0.45,
            aerial_loss_weight=0.20,
            swap_aug_prob=0.35,
            label_smoothing=0.02,
            seed=42 + k,
        ).fit(psts_tr, ys_tr, hs_tr, ball_zs=bzs_tr, sample_weights=ws_tr)

        p3, c3, p4d_3, air_3, bz_3 = model.predict_all_heads(test_pst_v3[T], sportec[T].half)
        p6, c6, p4d_6, air_6, bz_6 = model.predict_all_heads(test_pst_v6[T], sportec[T].half)

        np.savez_compressed(
            f_npz,
            v3_proba=p3,
            v3_change=c3,
            v3_p4d=p4d_3,
            v3_p_aerial=air_3,
            v3_ball_z_hat=bz_3,
            v6_proba=p6,
            v6_change=c6,
            v6_p4d=p4d_6,
            v6_p_aerial=air_6,
            v6_ball_z_hat=bz_6,
        )

        z1 = np.load(w7a_dir / f"pset_{T}.npz")
        pr1_3 = decode_v4(z1["v3_proba"], sportec[T].half, loo_dec_hyb[T], p_change=z1["v3_change"])
        pr2_3 = decode_v4(p3, sportec[T].half, loo_dec_hyb[T], p_change=c3)
        pr1_6 = decode_v4(z1["v6_proba"], sportec[T].half, loo_dec_hyb[T], p_change=z1["v6_change"])
        pr2_6 = decode_v4(p6, sportec[T].half, loo_dec_hyb[T], p_change=c6)

        f1_1_3 = float(metrics.score(sportec[T].labels, pr1_3, sportec[T].half)["spell_f1_mean"])
        f1_2_3 = float(metrics.score(sportec[T].labels, pr2_3, sportec[T].half)["spell_f1_mean"])
        f1_1_6 = float(metrics.score(sportec[T].labels, pr1_6, sportec[T].half)["spell_f1_mean"])
        f1_2_6 = float(metrics.score(sportec[T].labels, pr2_6, sportec[T].half)["spell_f1_mean"])

        bz_gt = sportec[T].meta["ball_z"]
        vbz = (sportec[T].half > 0) & np.isfinite(bz_gt)
        air_auc = float(roc_auc_score((bz_gt[vbz] > 1.5).astype(int), air_3[vbz]))

        print(
            f"  [Fold {k+1}/7 {T} ({time.time() - tf:.1f}s)] "
            f"v3: v1={f1_1_3:.4f} -> v2={f1_2_3:.4f} ({f1_2_3 - f1_1_3:+.4f}) | "
            f"v6: v1={f1_1_6:.4f} -> v2={f1_2_6:.4f} ({f1_2_6 - f1_1_6:+.4f}) | "
            f"AerialAUC={air_auc:.4f}",
            flush=True,
        )
        del model, psts_tr, ys_tr, hs_tr, bzs_tr, ws_tr
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    print("\n--- Step 2B: Full 7-Match `SymPlayerSetNetV2` Training & J2/J3 Prediction ---", flush=True)
    full_pkl = out_dir / "symplayerset_v2_full.pkl"
    j23_pkl = out_dir / "pset_v2_j23_preds.pkl"

    if full_pkl.exists():
        with open(full_pkl, "rb") as f:
            pset_full = pickle.load(f)
        print("  Loaded symplayerset_v2_full.pkl from cache.", flush=True)
    else:
        tf = time.time()
        psts_all, ys_all, hs_all, bzs_all, ws_all = [], [], [], [], []
        for mid in MATCH_IDS:
            bz_mid = sportec[mid].meta.get("ball_z")
            for key, _, _ in TRAIN_VARIANTS:
                psts_all.append(train_pst_v3[(mid, key)])
                ys_all.append(sportec[mid].labels)
                hs_all.append(sportec[mid].half)
                bzs_all.append(bz_mid)
                ws_all.append(1.0)
            psts_all.append(train_pst_v6[mid])
            ys_all.append(sportec[mid].labels)
            hs_all.append(sportec[mid].half)
            bzs_all.append(bz_mid)
            ws_all.append(0.80)

        pset_full = SymPlayerSetNetV2(
            channels=64,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            gru_stride=10,
            dropout=0.15,
            lr=1.5e-3,
            weight_decay=1e-4,
            epochs=10,
            crop=1500,
            batch_size=24,
            poss_alpha=0.45,
            boundary_boost=1.6,
            change_loss_weight=0.35,
            b4d_loss_weight=0.45,
            aerial_loss_weight=0.20,
            swap_aug_prob=0.35,
            label_smoothing=0.02,
            seed=42,
        ).fit(psts_all, ys_all, hs_all, ball_zs=bzs_all, sample_weights=ws_all)

        with open(full_pkl, "wb") as f:
            pickle.dump(pset_full, f)
        print(f"  Trained & saved symplayerset_v2_full.pkl in {time.time() - tf:.1f}s", flush=True)
        del psts_all, ys_all, hs_all, bzs_all, ws_all
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    train_pst_v3.clear()
    train_pst_v6.clear()
    gc.collect()

    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()

    if not j23_pkl.exists():
        tf = time.time()
        Pb, Bb, slot_team_b = load_broadcast()
        hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
        hom_valid = hom["valid"].to_numpy().astype(bool)
        mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
        atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)

        atd_rt = load_atd_reteam()
        mapping_rt, _ = enhanced_broadcast_alignment(atd_rt, Pb, hom_valid)
        atd_reteam_rep = repair_atd_match(atd_rt, Pb, Bb, slot_team_b, mapping_rt, True, True, None)

        sample_preds = {}
        for gid in GAMES:
            ms = sample_matches[gid]
            pst_s = load_pst_fast(pset_cache_dir, f"sample_game{gid}", lambda ms=ms: ms)
            ps, cs, p4ds, airs, bzs = pset_full.predict_all_heads(pst_s, ms.half)
            sample_preds[gid] = {
                "pset_proba": ps,
                "pset_change": cs,
                "pset_p4d": p4ds,
                "pset_p_aerial": airs,
                "pset_ball_z_hat": bzs,
            }

        pst_atd_raw = load_pst_fast(pset_cache_dir, "atd_raw", lambda: atd)
        par, car, p4dr, airr, bzr = pset_full.predict_all_heads(pst_atd_raw, atd.half)

        pst_atd_rep = load_pst_fast(pset_cache_dir, "atd_rep", lambda: atd_rep)
        pap, cap, p4dp, airp, bzp = pset_full.predict_all_heads(pst_atd_rep, atd.half)

        pst_atd_rt = cached_player_set_tensors(atd_reteam_rep, pset_cache_dir, tag="atd_reteam_rep")
        part, cart, p4drt, airrt, bzrt = pset_full.predict_all_heads(pst_atd_rt, atd.half)

        pset_v2_j23 = {
            "sample": sample_preds,
            "atd_raw": {
                "pset_proba": par,
                "pset_change": car,
                "pset_p4d": p4dr,
                "pset_p_aerial": airr,
                "pset_ball_z_hat": bzr,
            },
            "atd_rep": {
                "pset_proba": pap,
                "pset_change": cap,
                "pset_p4d": p4dp,
                "pset_p_aerial": airp,
                "pset_ball_z_hat": bzp,
            },
            "atd_reteam_rep": {
                "pset_proba": part,
                "pset_change": cart,
                "pset_p4d": p4drt,
                "pset_p_aerial": airrt,
                "pset_ball_z_hat": bzrt,
            },
        }
        with open(j23_pkl, "wb") as f:
            pickle.dump(pset_v2_j23, f)
        print(f"  Computed & saved pset_v2_j23_preds.pkl in {time.time() - tf:.1f}s", flush=True)
    else:
        with open(j23_pkl, "rb") as f:
            pset_v2_j23 = pickle.load(f)

    # Benchmark vs v1
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f:
        pset_v1_j23 = pickle.load(f)

    v1_j1a, v2_j1a, v1_j1b, v2_j1b = [], [], [], []
    v1_chg_auc, v2_chg_auc, v2_air_v3, v2_air_v6 = [], [], [], []

    for mid in MATCH_IDS:
        m = sportec[mid]
        z1 = np.load(w7a_dir / f"pset_{mid}.npz")
        z2 = np.load(out_dir / f"pset_v2_{mid}.npz")

        pr1_v3 = decode_v4(z1["v3_proba"], m.half, loo_dec_hyb[mid], p_change=z1["v3_change"])
        pr2_v3 = decode_v4(z2["v3_proba"], m.half, loo_dec_hyb[mid], p_change=z2["v3_change"])
        pr1_v6 = decode_v4(z1["v6_proba"], m.half, loo_dec_hyb[mid], p_change=z1["v6_change"])
        pr2_v6 = decode_v4(z2["v6_proba"], m.half, loo_dec_hyb[mid], p_change=z2["v6_change"])

        v1_j1a.append(float(metrics.score(m.labels, pr1_v3, m.half)["spell_f1_mean"]))
        v2_j1a.append(float(metrics.score(m.labels, pr2_v3, m.half)["spell_f1_mean"]))
        v1_j1b.append(float(metrics.score(m.labels, pr1_v6, m.half)["spell_f1_mean"]))
        v2_j1b.append(float(metrics.score(m.labels, pr2_v6, m.half)["spell_f1_mean"]))

        yc = change_targets(m.labels, m.half)
        vc = yc >= 0
        yt = (yc[vc] > 0.5).astype(np.int32)
        v1_chg_auc.append(float(roc_auc_score(yt, z1["v3_change"][vc])))
        v2_chg_auc.append(float(roc_auc_score(yt, z2["v3_change"][vc])))

        bz = m.meta["ball_z"]
        vbz = (m.half > 0) & np.isfinite(bz)
        yt_air = (bz[vbz] > 1.5).astype(np.int32)
        v2_air_v3.append(float(roc_auc_score(yt_air, z2["v3_p_aerial"][vbz])))
        v2_air_v6.append(float(roc_auc_score(yt_air, z2["v6_p_aerial"][vbz])))

    v1_j2_raw, v2_j2_raw, v1_j2_nod, v2_j2_nod = [], [], [], []
    for gid in GAMES:
        ms = sample_matches[gid]
        d1 = pset_v1_j23["sample"][gid]
        d2 = pset_v2_j23["sample"][gid]
        pr1 = decode_v4(d1["pset_proba"], ms.half, dec_v5_hyb, p_change=d1["pset_change"])
        pr2 = decode_v4(d2["pset_proba"], ms.half, dec_v5_hyb, p_change=d2["pset_change"])
        v1_j2_raw.append(float(metrics.score(ms.labels, pr1, ms.half)["spell_f1_mean"]))
        v2_j2_raw.append(float(metrics.score(ms.labels, pr2, ms.half)["spell_f1_mean"]))
        y_nd = without_dead_margin(ms.labels, ms.half, 3.0)
        v1_j2_nod.append(float(metrics.score(y_nd, pr1, ms.half)["spell_f1_mean"]))
        v2_j2_nod.append(float(metrics.score(y_nd, pr2, ms.half)["spell_f1_mean"]))

    summary = {
        "v1_J1A_mean": float(np.mean(v1_j1a)),
        "v2_J1A_mean": float(np.mean(v2_j1a)),
        "delta_J1A": float(np.mean(v2_j1a) - np.mean(v1_j1a)),
        "v1_J1B_mean": float(np.mean(v1_j1b)),
        "v2_J1B_mean": float(np.mean(v2_j1b)),
        "delta_J1B": float(np.mean(v2_j1b) - np.mean(v1_j1b)),
        "v1_J2_raw_mean": float(np.mean(v1_j2_raw)),
        "v2_J2_raw_mean": float(np.mean(v2_j2_raw)),
        "delta_J2_raw": float(np.mean(v2_j2_raw) - np.mean(v1_j2_raw)),
        "v1_J2_noD_mean": float(np.mean(v1_j2_nod)),
        "v2_J2_noD_mean": float(np.mean(v2_j2_nod)),
        "delta_J2_noD": float(np.mean(v2_j2_nod) - np.mean(v1_j2_nod)),
        "v1_chg_auc_mean": float(np.mean(v1_chg_auc)),
        "v2_chg_auc_mean": float(np.mean(v2_chg_auc)),
        "v2_aerial_auc_v3_mean": float(np.mean(v2_air_v3)),
        "v2_aerial_auc_v6_mean": float(np.mean(v2_air_v6)),
        "total_elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "pset_v2_benchmark.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 115, flush=True)
    print(
        f"[SUMMARY SymPlayerSetNet v1 -> v2] "
        f"J1A: {summary['v1_J1A_mean']:.4f} -> {summary['v2_J1A_mean']:.4f} ({summary['delta_J1A']:+.4f}) | "
        f"J1B: {summary['v1_J1B_mean']:.4f} -> {summary['v2_J1B_mean']:.4f} ({summary['delta_J1B']:+.4f}) | "
        f"J2-Raw: {summary['v1_J2_raw_mean']:.4f} -> {summary['v2_J2_raw_mean']:.4f} ({summary['delta_J2_raw']:+.4f}) | "
        f"J2-noD: {summary['v1_J2_noD_mean']:.4f} -> {summary['v2_J2_noD_mean']:.4f} ({summary['delta_J2_noD']:+.4f}) | "
        f"AerialAUC: v3={summary['v2_aerial_auc_v3_mean']:.4f}, v6={summary['v2_aerial_auc_v6_mean']:.4f}",
        flush=True,
    )
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()
