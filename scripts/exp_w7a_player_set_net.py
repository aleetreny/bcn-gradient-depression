"""Wave 7A (EXP-11): 6th Model Family — `SymPlayerSetNet` (Permutation-Invariant PointNet / DeepSets + Ghost-Gated Dilated Temporal Conv + Strided BiGRU).

Architecture & Experimental Design:
- Existing 5 model families (`GBM`, `CatBoost`, `XGB`, `TCN`, `BiGRU`) all consume hand-engineered tabular features
  (`v1ctl` 244 cols or `v1ctl_physics_v2` 481 cols).
- `SymPlayerSetNet` (`src/models/player_set_net.py`) operates directly on the 22 players' spatio-temporal trajectories
  `(T, 11, 18)` per team:
  1. Channels `0..11` (12 channels) are 100% ball-free: own-attacking coordinates `(x_own, y_own)`, track-break-safe
     cascaded velocities `(vx_own, vy_own)`, speed, still indicator, relative vector/distance/proximity/closing-speed
     to the ball-free 22-player convergence focus `x_focus(t)`, and nearest-opponent distance/proximity.
  2. Channels `12..17` (6 channels) are ball-aware: relative vector, distance, 3m/10m proximity, and closing speed
     to the interpolated ball.
  3. `_SplitPlayerPointNet` runs a split ball-free branch (`phi_bf`, reading channels `0..11`) and ball-aware branch
     (`phi_ball`, reading all 18 channels) with masked Mean + Max + 2-head Softmax-Attention pooling over the `K=11`
     players per team, achieving 100% permutation invariance to player-slot column ordering and track-breaks.
  4. A symmetric ghost-ball / blackout reliability gate (`ghost_gate(g_sym)`) modulates the pooled player representations
     before an intrinsically H <-> A + 180-deg pitch rotation equivariant twin-stream Dilated TCN (`dilations=1..128`)
     + Strided BiGRU (`gru_stride=10`) backbone.
  5. Guarantees exact `0.000e+00` H <-> A equivariance and `0.000e+00` player-slot permutation invariance in a single
     forward pass without requiring external 2x test-time augmentation.

Evaluation Protocol (All 4 Judges):
- Step 1: Verify exact `0.000e+00` H <-> A + 180-deg equivariance and `0.000e+00` player-slot permutation invariance
  of `extract_player_set_tensors` and `SymPlayerSetNet` on `J03WMX`.
- Step 2: Extract & cache `PlayerSetTensors` for all Sportec training variants (`degrade_v3` `TRAIN_VARIANTS` + `degrade_v6` `s1`)
  and all 4 Judge test sets (`Sportec v3`, `Sportec v6`, `Metrica Sample` 3 games, `BAR-ATH Raw` & `Broadcast-Repaired`).
- Step 3: Train 7-fold outer LOMO `SymPlayerSetNet` (`experiments/outputs/exp_w7a_oof/pset_{T}.npz`) and full 7-match
  `SymPlayerSetNet` (`symplayerset_full.pkl` & `pset_j23_preds.pkl`).
- Step 4: Evaluate standalone `SymPlayerSetNet`, multi-sequence blends (`TCN + BiGRU + SymPlayerSetNet`), and 6-Family
  ensembles (`GBM + CatBoost + XGB + TCN + BiGRU + SymPlayerSetNet`) across all 4 Judges under `decode_v4`,
  `decode_v4 + xdx_fixA`, and the full Wave 6 `EXP-09 + EXP-08 + EXP-10` MBR pipeline.
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

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import torch

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_7_nested_full import TRAIN_VARIANTS
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.common import AWAY, DEAD, HOME
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.mbr_snap import (
    ImpulseSnapConfig,
    decode_active_4dir,
    mbr_decode_sequences,
    snap_boundaries_to_impulse,
)
from src.decode.spell_reranker import (
    SpellRerankerConfig,
    apply_spell_reranker,
    fit_spell_reranker_lomo,
)
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    compute_seq481_gate,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator, count_interior_xdx
from src.eval import metrics
from src.features.symmetry import swap_labels, swap_match, swap_proba
from src.models.player_set_net import (
    PlayerSetTensors,
    SymPlayerSetNet,
    cached_player_set_tensors,
    extract_player_set_tensors,
)
from src.models.transition import change_features, change_targets
from exp_b_symmetry_eval import blend_proba
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
from exp_v6_super_champion import compute_physical_gates, save_and_validate_submission


def load_cached_physv2_table(prefix: str, half: np.ndarray) -> pd.DataFrame:
    """Fast load of 481-column `v1ctl_physics_v2` DataFrame from `local_data/cache/`."""
    cdir = ROOT / "local_data" / "cache"
    p_f1 = sorted((cdir / "features_v1").glob(f"{prefix}*.parquet"))[0]
    p_c1 = sorted((cdir / "control_v1").glob(f"{prefix}*.parquet"))[0]
    p_pl = sorted((cdir / "plus_v1").glob(f"{prefix}*.parquet"))[0]
    p_p2 = sorted((cdir / "physv2").glob(f"{prefix}*.parquet"))[0]
    df1 = change_features(pd.read_parquet(p_f1), half)
    return pd.concat(
        [df1, pd.read_parquet(p_c1), pd.read_parquet(p_pl), pd.read_parquet(p_p2)],
        axis=1,
    )


def frame_acc(y_true: np.ndarray, pred_or_proba: np.ndarray) -> float:
    valid = y_true >= 0
    if not np.any(valid):
        return float("nan")
    pred = np.argmax(pred_or_proba[valid], axis=1) if pred_or_proba.ndim == 2 else pred_or_proba[valid]
    return float((pred == y_true[valid]).mean())


def frame_logloss(y_true: np.ndarray, proba: np.ndarray) -> float:
    valid = y_true >= 0
    if not np.any(valid):
        return float("nan")
    return float(log_loss(y_true[valid], np.clip(proba[valid], 1e-6, 1.0 - 1e-6), labels=[0, 1, 2]))


def change_auc_ap(y_true: np.ndarray, half: np.ndarray, p_change: np.ndarray) -> tuple[float, float]:
    yc = change_targets(y_true, half)
    valid = yc >= 0
    if not np.any(valid) or len(np.unique(yc[valid])) < 2:
        return float("nan"), float("nan")
    yt = (yc[valid] > 0.5).astype(np.int32)
    pc = np.nan_to_num(p_change[valid], nan=0.0)
    return float(roc_auc_score(yt, pc)), float(average_precision_score(yt, pc))


def swap_match_ha_180(m):
    """Exact H <-> A + 180-deg pitch rotation of a Match object."""
    L, W = m.pitch
    P_new = m.P.copy()
    P_new[:, :, 0] = L - P_new[:, :, 0]
    P_new[:, :, 1] = W - P_new[:, :, 1]
    B_new = m.B.copy()
    B_new[:, 0] = L - B_new[:, 0]
    B_new[:, 1] = W - B_new[:, 1]
    st_new = np.where(m.slot_team == HOME, AWAY, np.where(m.slot_team == AWAY, HOME, m.slot_team))
    return replace(
        m,
        P=P_new,
        B=B_new,
        slot_team=st_new,
        labels=swap_labels(m.labels) if m.labels is not None else None,
    )


def permute_match_slots(m, seed: int = 123):
    """Randomly permute player slot columns of a Match object."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m.P.shape[1])
    return replace(
        m,
        P=m.P[:, perm, :].copy(),
        slot_team=m.slot_team[perm].copy(),
        slot_is_gk=m.slot_is_gk[perm].copy(),
    )


def main():
    t0 = time.time()
    print("=" * 112, flush=True)
    print("WAVE 7A (EXP-11): 6TH MODEL FAMILY — `SymPlayerSetNet` (POINTNET + GHOST-GATE + DILATED TCN + STRIDED BIGRU)", flush=True)
    print("=" * 112, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "exp_w7a_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    pset_cache_dir = ROOT / "local_data" / "cache" / "pset"
    pset_cache_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    sportec = load_sportec()
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")

    # =========================================================================
    # STEP 1: MATHEMATICAL VERIFICATION OF EXACT H<->A EQUIVARIANCE & SLOT INVARIANCE
    # =========================================================================
    print("\n--- Step 1: Exact H <-> A + 180-deg Equivariance & Player-Slot Permutation Invariance Audit ---", flush=True)
    t_eq = time.time()
    m0 = degrade(sportec["J03WMX"], cfg_v3, seed=0)
    sl_eq = slice(0, 6000)
    m0_sub = replace(
        m0,
        P=m0.P[sl_eq],
        B=m0.B[sl_eq],
        half=m0.half[sl_eq],
        graded=m0.graded[sl_eq],
        attack_sign=m0.attack_sign[sl_eq],
        labels=m0.labels[sl_eq] if m0.labels is not None else None,
    )
    m0_swap = swap_match(m0_sub)
    m0_perm = permute_match_slots(m0_sub, seed=999)

    pst_orig = extract_player_set_tensors(m0_sub)
    pst_swap = extract_player_set_tensors(m0_swap)
    pst_perm = extract_player_set_tensors(m0_perm)

    err_t_ph = float(np.max(np.abs(pst_orig.p_h - pst_swap.p_a)))
    err_t_pa = float(np.max(np.abs(pst_orig.p_a - pst_swap.p_h)))
    err_t_mh = float(np.max(np.abs(pst_orig.mask_h.astype(int) - pst_swap.mask_a.astype(int))))
    err_t_ga = float(np.max(np.abs(pst_orig.g_anti + pst_swap.g_anti)))
    err_t_gs = float(np.max(np.abs(pst_orig.g_sym - pst_swap.g_sym)))

    err_p_ph = float(np.max(np.abs(pst_orig.p_h - pst_perm.p_h)))
    err_p_pa = float(np.max(np.abs(pst_orig.p_a - pst_perm.p_a)))
    err_p_ga = float(np.max(np.abs(pst_orig.g_anti - pst_perm.g_anti)))
    err_p_gs = float(np.max(np.abs(pst_orig.g_sym - pst_perm.g_sym)))

    toy_net = SymPlayerSetNet(epochs=1, batch_size=8, crop=1000, seed=42)
    toy_net.fit([pst_orig], [m0_sub.labels], [m0_sub.half])
    p_orig, c_orig = toy_net.predict_proba_and_change(pst_orig, m0_sub.half)
    p_swap, c_swap = toy_net.predict_proba_and_change(pst_swap, m0_swap.half)
    p_perm, c_perm = toy_net.predict_proba_and_change(pst_perm, m0_perm.half)

    err_net_ha_p = float(np.max(np.abs(swap_proba(p_orig) - p_swap)))
    err_net_ha_c = float(np.max(np.abs(c_orig - c_swap)))
    err_net_perm_p = float(np.max(np.abs(p_orig - p_perm)))
    err_net_perm_c = float(np.max(np.abs(c_orig - c_perm)))

    print(
        f"  [Tensor Audit] H<->A + 180-deg: p_h={err_t_ph:.3e}, p_a={err_t_pa:.3e}, mask={err_t_mh:.3e}, "
        f"g_anti={err_t_ga:.3e}, g_sym={err_t_gs:.3e}",
        flush=True,
    )
    print(
        f"  [Tensor Audit] Slot Permutation: p_h={err_p_ph:.3e}, p_a={err_p_pa:.3e}, "
        f"g_anti={err_p_ga:.3e}, g_sym={err_p_gs:.3e}",
        flush=True,
    )
    print(
        f"  [Model Audit]  H<->A Equivariance: proba={err_net_ha_p:.3e}, change={err_net_ha_c:.3e} | "
        f"Slot Permutation Invariance: proba={err_net_perm_p:.3e}, change={err_net_perm_c:.3e} ({time.time() - t_eq:.2f}s)",
        flush=True,
    )
    assert max(err_t_ph, err_t_pa, err_t_mh, err_t_ga, err_t_gs) == 0.0, "Tensor H<->A symmetry violation!"
    assert max(err_p_ph, err_p_pa, err_p_ga, err_p_gs) == 0.0, "Tensor slot permutation violation!"
    assert max(err_net_ha_p, err_net_ha_c, err_net_perm_p, err_net_perm_c) == 0.0, "Model symmetry/permutation violation!"
    del toy_net, pst_orig, pst_swap, pst_perm, m0_sub, m0_swap, m0_perm
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    # =========================================================================
    # STEP 2: EXTRACT & CACHE `PlayerSetTensors` FOR SPORTEC (TRAIN + TEST V3/V6)
    # =========================================================================
    print("\n--- Step 2: Extracting & Caching `PlayerSetTensors` for Sportec (7 Matches x Train/Test Variants) ---", flush=True)
    t_ext = time.time()

    need_lomo = any(not (out_dir / f"pset_{T}.npz").exists() for T in MATCH_IDS)
    need_full = not (out_dir / "symplayerset_full.pkl").exists()

    train_pst_v3: dict[tuple[str, str], PlayerSetTensors] = {}
    train_pst_v6: dict[str, PlayerSetTensors] = {}
    test_pst_v3: dict[str, PlayerSetTensors] = {}
    test_pst_v6: dict[str, PlayerSetTensors] = {}

    for mid in MATCH_IDS:
        m = sportec[mid]
        tm0 = time.time()
        if need_lomo or need_full:
            for key, overrides, seed in TRAIN_VARIANTS:
                cfg = replace(cfg_v3, **(overrides or {}))
                mv = degrade(m, cfg, seed)
                train_pst_v3[(mid, key)] = cached_player_set_tensors(mv, pset_cache_dir, tag=f"sportec_{mid}_v3_{key}")
            mv6_tr = degrade(m, cfg_v6, 11)
            train_pst_v6[mid] = cached_player_set_tensors(mv6_tr, pset_cache_dir, tag=f"sportec_{mid}_v6_s1")

        mv3_te = degrade(m, cfg_v3, 0)
        mv6_te = degrade(m, cfg_v6, 0)
        test_pst_v3[mid] = cached_player_set_tensors(mv3_te, pset_cache_dir, tag=f"sportec_{mid}_v3_s0")
        test_pst_v6[mid] = cached_player_set_tensors(mv6_te, pset_cache_dir, tag=f"sportec_{mid}_v6_s0")
        print(f"  [Cache Sportec {mid}] ready in {time.time() - tm0:.2f}s", flush=True)

    print(f"  All Sportec `PlayerSetTensors` ready in {time.time() - t_ext:.1f}s", flush=True)

    # =========================================================================
    # STEP 3: 7-FOLD OUTER LOMO + FULL 7-MATCH TRAINING OF `SymPlayerSetNet`
    # =========================================================================
    print("\n--- Step 3A: 7-Fold Outer LOMO Training of `SymPlayerSetNet` (MPS GPU) ---", flush=True)
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        loo_dec_lin, loo_dec_hyb, dec_v5_lin, dec_v5_hyb = pickle.load(f)

    oof_pset: dict[str, dict[str, np.ndarray]] = {}
    for k, T in enumerate(MATCH_IDS):
        f_npz = out_dir / f"pset_{T}.npz"
        if f_npz.exists():
            z = np.load(f_npz)
            oof_pset[T] = {
                "v3_proba": z["v3_proba"],
                "v3_change": z["v3_change"],
                "v6_proba": z["v6_proba"],
                "v6_change": z["v6_change"],
            }
            p3, c3 = oof_pset[T]["v3_proba"], oof_pset[T]["v3_change"]
            p6, c6 = oof_pset[T]["v6_proba"], oof_pset[T]["v6_change"]
            pr3 = decode_v4(p3, sportec[T].half, loo_dec_hyb[T], p_change=c3)
            pr6 = decode_v4(p6, sportec[T].half, loo_dec_hyb[T], p_change=c6)
            f1_3 = metrics.score(sportec[T].labels, pr3, sportec[T].half)["spell_f1_mean"]
            f1_6 = metrics.score(sportec[T].labels, pr6, sportec[T].half)["spell_f1_mean"]
            print(
                f"  [Fold {k+1}/7 {T} (cached)] v3: SpellF1={f1_3:.4f}, acc={frame_acc(sportec[T].labels, p3):.4f}, "
                f"ll={frame_logloss(sportec[T].labels, p3):.4f} | v6: SpellF1={f1_6:.4f}, acc={frame_acc(sportec[T].labels, p6):.4f}",
                flush=True,
            )
            continue

        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        psts_tr, ys_tr, hs_tr, ws_tr = [], [], [], []
        for mid in train_ids:
            for key, _, _ in TRAIN_VARIANTS:
                psts_tr.append(train_pst_v3[(mid, key)])
                ys_tr.append(sportec[mid].labels)
                hs_tr.append(sportec[mid].half)
                ws_tr.append(1.0)
            psts_tr.append(train_pst_v6[mid])
            ys_tr.append(sportec[mid].labels)
            hs_tr.append(sportec[mid].half)
            ws_tr.append(0.75)

        model = SymPlayerSetNet(
            channels=64,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            gru_stride=10,
            dropout=0.15,
            lr=1.5e-3,
            weight_decay=1e-4,
            epochs=10,
            crop=1500,
            batch_size=24,
            change_loss_weight=0.35,
            label_smoothing=0.02,
            seed=42 + k,
        ).fit(psts_tr, ys_tr, hs_tr, sample_weights=ws_tr)

        p3, c3 = model.predict_proba_and_change(test_pst_v3[T], sportec[T].half)
        p6, c6 = model.predict_proba_and_change(test_pst_v6[T], sportec[T].half)

        np.savez_compressed(
            f_npz,
            v3_proba=p3.astype(np.float32),
            v3_change=c3.astype(np.float32),
            v6_proba=p6.astype(np.float32),
            v6_change=c6.astype(np.float32),
        )
        oof_pset[T] = {"v3_proba": p3, "v3_change": c3, "v6_proba": p6, "v6_change": c6}

        pr3 = decode_v4(p3, sportec[T].half, loo_dec_hyb[T], p_change=c3)
        pr6 = decode_v4(p6, sportec[T].half, loo_dec_hyb[T], p_change=c6)
        f1_3 = metrics.score(sportec[T].labels, pr3, sportec[T].half)["spell_f1_mean"]
        f1_6 = metrics.score(sportec[T].labels, pr6, sportec[T].half)["spell_f1_mean"]
        auc3, ap3 = change_auc_ap(sportec[T].labels, sportec[T].half, c3)
        print(
            f"  [Fold {k+1}/7 {T} ({time.time() - tf:.1f}s)] v3: SpellF1={f1_3:.4f}, acc={frame_acc(sportec[T].labels, p3):.4f}, "
            f"ll={frame_logloss(sportec[T].labels, p3):.4f}, chgAUC={auc3:.4f} | v6: SpellF1={f1_6:.4f}, acc={frame_acc(sportec[T].labels, p6):.4f}",
            flush=True,
        )
        del model, psts_tr, ys_tr, hs_tr, ws_tr
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    # Step 3B: Train Full 7-Match `SymPlayerSetNet` for Judge 2 & Judge 3
    print("\n--- Step 3B: Full 7-Match `SymPlayerSetNet` Training & Judge 2 / Judge 3 Prediction ---", flush=True)
    full_pkl = out_dir / "symplayerset_full.pkl"
    j23_pkl = out_dir / "pset_j23_preds.pkl"

    if full_pkl.exists():
        with open(full_pkl, "rb") as f:
            pset_full = pickle.load(f)
        print("  Loaded symplayerset_full.pkl from cache.", flush=True)
    else:
        tf = time.time()
        psts_all, ys_all, hs_all, ws_all = [], [], [], []
        for mid in MATCH_IDS:
            for key, _, _ in TRAIN_VARIANTS:
                psts_all.append(train_pst_v3[(mid, key)])
                ys_all.append(sportec[mid].labels)
                hs_all.append(sportec[mid].half)
                ws_all.append(1.0)
            psts_all.append(train_pst_v6[mid])
            ys_all.append(sportec[mid].labels)
            hs_all.append(sportec[mid].half)
            ws_all.append(0.75)

        pset_full = SymPlayerSetNet(
            channels=64,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            gru_stride=10,
            dropout=0.15,
            lr=1.5e-3,
            weight_decay=1e-4,
            epochs=10,
            crop=1500,
            batch_size=24,
            change_loss_weight=0.35,
            label_smoothing=0.02,
            seed=42,
        ).fit(psts_all, ys_all, hs_all, sample_weights=ws_all)
        with open(full_pkl, "wb") as f:
            pickle.dump(pset_full, f)
        print(f"  Trained & saved symplayerset_full.pkl in {time.time() - tf:.1f}s", flush=True)
        del psts_all, ys_all, hs_all, ws_all
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    train_pst_v3.clear()
    train_pst_v6.clear()
    gc.collect()

    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    if j23_pkl.exists():
        with open(j23_pkl, "rb") as f:
            pset_j23 = pickle.load(f)
        print("  Loaded pset_j23_preds.pkl from cache.", flush=True)
    else:
        tf = time.time()
        Pb, Bb, slot_team_b = load_broadcast()
        hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
        hom_valid = hom["valid"].to_numpy().astype(bool)
        mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
        atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)

        sample_pset_preds = {}
        for gid in GAMES:
            ms = sample_matches[gid]
            pst_s = cached_player_set_tensors(ms, pset_cache_dir, tag=f"sample_game{gid}")
            ps, cs = pset_full.predict_proba_and_change(pst_s, ms.half)
            sample_pset_preds[gid] = {"pset_proba": ps, "pset_change": cs}

        pst_atd_raw = cached_player_set_tensors(atd, pset_cache_dir, tag="atd_raw")
        par, car = pset_full.predict_proba_and_change(pst_atd_raw, atd.half)

        pst_atd_rep = cached_player_set_tensors(atd_rep, pset_cache_dir, tag="atd_rep")
        pap, cap = pset_full.predict_proba_and_change(pst_atd_rep, atd.half)

        pset_j23 = {
            "sample": sample_pset_preds,
            "atd_raw": {"pset_proba": par, "pset_change": car},
            "atd_rep": {"pset_proba": pap, "pset_change": cap},
        }
        with open(j23_pkl, "wb") as f:
            pickle.dump(pset_j23, f)
        print(f"  Computed & saved pset_j23_preds.pkl in {time.time() - tf:.1f}s", flush=True)

    # =========================================================================
    # STEP 4: COMPREHENSIVE 4-JUDGE EVALUATION (STANDALONE + 6-FAMILY ENSEMBLES)
    # =========================================================================
    print("\n--- Step 4A: Standalone Sequence Family Comparison Across All 4 Judges ---", flush=True)
    plus_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    h_dir = ROOT / "experiments" / "outputs" / "exp_h_oof"
    w3_dir = ROOT / "experiments" / "outputs" / "exp_w3_oof"
    w4_dir = ROOT / "experiments" / "outputs" / "exp_w4_oof"
    w5_dir = ROOT / "experiments" / "outputs" / "exp_w5_tree_oof"
    w5b_dir = ROOT / "experiments" / "outputs" / "exp_w5b_oof"
    w6_dir = ROOT / "experiments" / "outputs" / "exp_w6_oof"

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std = pickle.load(f)[0]
    dec_v4 = DecoderParams(**{
        k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items()
        if not k.startswith("_")
    })

    X_v3, X_v6 = {}, {}
    gates_v3, gates_v6 = {}, {}
    for mid, m in sportec.items():
        X_v3[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", m.half)
        X_v6[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", m.half)
        gates_v3[mid] = compute_physical_gates(X_v3[mid], m.half)
        gates_v6[mid] = compute_physical_gates(X_v6[mid], m.half)

    X_sample, gates_sample = {}, {}
    for gid in GAMES:
        X_sample[gid] = load_cached_physv2_table(f"sample_{gid}_raw", sample_matches[gid].half)
        gates_sample[gid] = compute_physical_gates(X_sample[gid], sample_matches[gid].half)

    X_atd_raw = load_cached_physv2_table("atd_0f0834", atd.half)
    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)
    gate_atd_raw = compute_physical_gates(X_atd_raw, atd.half)
    gate_atd_rep = compute_physical_gates(X_atd_rep, atd.half)

    oof = {}
    for T in MATCH_IDS:
        zb = np.load(sym_dir / f"sym_{T}.npz")
        zt = np.load(sym_dir / f"tcn_sym_{T}.npz")
        zp = np.load(plus_dir / f"plus_{T}.npz")
        zg = np.load(g_dir / f"ghost_sym_{T}.npz")
        zh = np.load(h_dir / f"bigru_sym_{T}.npz")
        zw3 = np.load(w3_dir / f"b4d_{T}.npz")
        zw4 = np.load(w4_dir / f"physv2_{T}.npz")
        zw5 = np.load(w5_dir / f"cb_xgb_{T}.npz")
        zw5b_tcn = np.load(w5b_dir / f"tcn_physv2_{T}.npz")
        zw5b_gru = np.load(w5b_dir / f"bigru_physv2_{T}.npz")
        oof[T] = {
            "g_v1": zb["symgbm_tta"],
            "tr_v1": zb["symtrans_tta"],
            "g_pl": zp["symgbm_plus_tta"],
            "tr_pl": zp["symtrans_plus_tta"],
            "g_v6_on_v3": zg["g_v6_on_v3"],
            "tr_v6_on_v3": zg["tr_v6_on_v3"],
            "g_v3_on_v6": zg["g_v3_on_v6"],
            "tr_v3_on_v6": zg["tr_v3_on_v6"],
            "g_v6_on_v6": zg["g_v6_on_v6"],
            "tr_v6_on_v6": zg["tr_v6_on_v6"],
            "tcn": zt["symtcn_tta_proba"],
            "tcnc": zt["symtcn_tta_change"],
            "gru": zh["symbigru_tta_proba"],
            "gruc": zh["symbigru_tta_change"],
            "p4d_v3": zw3["p4d_v3"],
            "p4d_v6": zw3["p4d_v6"],
            "g_p2_v3": zw4["symgbm_physv2_tta"],
            "tr_p2_v3": zw4["symtrans_physv2_tta"],
            "g_p2_v6": zw4["symgbm_physv2_tta_on_v6"],
            "tr_p2_v6": zw4["symtrans_physv2_tta_on_v6"],
            "cb_p2_v3": zw5["symcatboost_physv2_tta"],
            "xgb_p2_v3": zw5["symxgb_physv2_tta"],
            "cb_p2_v6": zw5["symcatboost_physv2_tta_on_v6"],
            "xgb_p2_v6": zw5["symxgb_physv2_tta_on_v6"],
            "tcn_p2_v3": zw5b_tcn["v3_proba"],
            "tcnc_p2_v3": zw5b_tcn["v3_change"],
            "tcn_p2_v6": zw5b_tcn["v6_proba"],
            "tcnc_p2_v6": zw5b_tcn["v6_change"],
            "gru_p2_v3": zw5b_gru["v3_proba"],
            "gruc_p2_v3": zw5b_gru["v3_change"],
            "gru_p2_v6": zw5b_gru["v6_proba"],
            "gruc_p2_v6": zw5b_gru["v6_change"],
            "pset_v3": oof_pset[T]["v3_proba"],
            "psetc_v3": oof_pset[T]["v3_change"],
            "pset_v6": oof_pset[T]["v6_proba"],
            "psetc_v6": oof_pset[T]["v6_change"],
        }

    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        pe_plus = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f:
        sample_v6, atd_rep_v6 = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(w3_dir / "b4d_j23_preds.pkl", "rb") as f:
        b4d_j23 = pickle.load(f)
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f:
        pe_p2 = pickle.load(f)
    with open(w5_dir / "eval_preds_cb_xgb.pkl", "rb") as f:
        pe_w5 = pickle.load(f)
    with open(w5b_dir / "seq_physv2_j23_preds.pkl", "rb") as f:
        pe_w5b = pickle.load(f)
    with open(w6_dir / "atd_raw_v6_bigru.pkl", "rb") as f:
        atd_raw_extra = pickle.load(f)

    def eval_standalone_seq(
        label: str,
        get_v3,
        get_v6,
        get_sample,
        get_atd_raw,
        get_atd_rep,
    ):
        v3_f1, v3_acc, v3_ll, v3_auc = [], [], [], []
        v6_f1, v6_acc, v6_ll = [], [], []
        for T in MATCH_IDS:
            h = sportec[T].half
            y = sportec[T].labels
            p3, c3 = get_v3(T)
            p6, c6 = get_v6(T)
            pr3 = decode_v4(p3, h, loo_dec_hyb[T], p_change=c3)
            pr6 = decode_v4(p6, h, loo_dec_hyb[T], p_change=c6)
            v3_f1.append(metrics.score(y, pr3, h)["spell_f1_mean"])
            v3_acc.append(frame_acc(y, p3))
            v3_ll.append(frame_logloss(y, p3))
            v3_auc.append(change_auc_ap(y, h, c3)[0])
            v6_f1.append(metrics.score(y, pr6, h)["spell_f1_mean"])
            v6_acc.append(frame_acc(y, p6))
            v6_ll.append(frame_logloss(y, p6))

        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            ps, cs = get_sample(gid)
            prs = decode_v4(ps, ms.half, dec_v5_hyb, p_change=cs)
            s_raw.append(metrics.score(ms.labels, prs, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), prs, ms.half)["spell_f1_mean"])

        par, car = get_atd_raw()
        prar = decode_v4(par, half_atd_dec, dec_v5_hyb, p_change=car)
        prar[second_start:kick2] = DEAD
        prar[~atd.graded] = DEAD
        pub_raw = metrics.score_public(prar, bootstrap_n=0)

        pap, cap = get_atd_rep()
        prap = decode_v4(pap, half_atd_dec, dec_v5_hyb, p_change=cap)
        prap[second_start:kick2] = DEAD
        prap[~atd.graded] = DEAD
        pub_rep = metrics.score_public(prap, bootstrap_n=0)

        print(
            f"  {label:56s} | "
            f"J1A: {np.mean(v3_f1):.4f} (acc={np.mean(v3_acc):.3f}, ll={np.mean(v3_ll):.3f}, auc={np.mean(v3_auc):.3f}) | "
            f"J1B: {np.mean(v6_f1):.4f} (acc={np.mean(v6_acc):.3f}) | "
            f"J2: {np.mean(s_raw):.4f} (noD={np.mean(s_nod):.4f}) | "
            f"J3 Raw: {pub_raw['spell_f1_mean']:.4f} (D={pub_raw['alive_dead_f1']:.3f}) | "
            f"J3 Rep: {pub_rep['spell_f1_mean']:.4f} (D={pub_rep['alive_dead_f1']:.3f})",
            flush=True,
        )

    eval_standalone_seq(
        "1. SymTCN_TTA (244 v1ctl)",
        lambda T: (oof[T]["tcn"], oof[T]["tcnc"]),
        lambda T: (oof[T]["tcn"], oof[T]["tcnc"]),
        lambda gid: (sample_h[gid]["t_sym"], sample_h[gid]["tc_sym"]),
        lambda: (pe_plus["atd_raw"]["t_sym"], pe_plus["atd_raw"]["tc_sym"]),
        lambda: (atd_rep_h["t_sym"], atd_rep_h["tc_sym"]),
    )
    eval_standalone_seq(
        "2. SymBiGRU_TTA (244 v1ctl)",
        lambda T: (oof[T]["gru"], oof[T]["gruc"]),
        lambda T: (oof[T]["gru"], oof[T]["gruc"]),
        lambda gid: (sample_h[gid]["b_sym"], sample_h[gid]["bc_sym"]),
        lambda: (atd_raw_extra["b_sym"], atd_raw_extra["bc_sym"]),
        lambda: (atd_rep_h["b_sym"], atd_rep_h["bc_sym"]),
    )
    eval_standalone_seq(
        "3. SymTCN_PhysV2_TTA (481 physv2)",
        lambda T: (oof[T]["tcn_p2_v3"], oof[T]["tcnc_p2_v3"]),
        lambda T: (oof[T]["tcn_p2_v6"], oof[T]["tcnc_p2_v6"]),
        lambda gid: (pe_w5b["sample"][gid]["tcn_physv2_sym"], pe_w5b["sample"][gid]["tcnc_physv2_sym"]),
        lambda: (pe_w5b["atd_raw"]["tcn_physv2_sym"], pe_w5b["atd_raw"]["tcnc_physv2_sym"]),
        lambda: (pe_w5b["atd_rep"]["tcn_physv2_sym"], pe_w5b["atd_rep"]["tcnc_physv2_sym"]),
    )
    eval_standalone_seq(
        "4. SymBiGRU_PhysV2_TTA (481 physv2)",
        lambda T: (oof[T]["gru_p2_v3"], oof[T]["gruc_p2_v3"]),
        lambda T: (oof[T]["gru_p2_v6"], oof[T]["gruc_p2_v6"]),
        lambda gid: (pe_w5b["sample"][gid]["bigru_physv2_sym"], pe_w5b["sample"][gid]["bigruc_physv2_sym"]),
        lambda: (pe_w5b["atd_raw"]["bigru_physv2_sym"], pe_w5b["atd_raw"]["bigruc_physv2_sym"]),
        lambda: (pe_w5b["atd_rep"]["bigru_physv2_sym"], pe_w5b["atd_rep"]["bigruc_physv2_sym"]),
    )
    eval_standalone_seq(
        "5. SymPlayerSetNet (EXP-11 PointNet+GhostGate+TCN+BiGRU)",
        lambda T: (oof[T]["pset_v3"], oof[T]["psetc_v3"]),
        lambda T: (oof[T]["pset_v6"], oof[T]["psetc_v6"]),
        lambda gid: (pset_j23["sample"][gid]["pset_proba"], pset_j23["sample"][gid]["pset_change"]),
        lambda: (pset_j23["atd_raw"]["pset_proba"], pset_j23["atd_raw"]["pset_change"]),
        lambda: (pset_j23["atd_rep"]["pset_proba"], pset_j23["atd_rep"]["pset_change"]),
    )
    eval_standalone_seq(
        "6. 3-Family Sequence Blend (TCN + BiGRU + SymPlayerSetNet)",
        lambda T: (
            0.35 * oof[T]["tcn"] + 0.15 * oof[T]["gru"] + 0.25 * oof[T]["gru_p2_v3"] + 0.25 * oof[T]["pset_v3"],
            0.35 * oof[T]["tcnc"] + 0.15 * oof[T]["gruc"] + 0.25 * oof[T]["gruc_p2_v3"] + 0.25 * oof[T]["psetc_v3"],
        ),
        lambda T: (
            0.35 * oof[T]["tcn"] + 0.15 * oof[T]["gru"] + 0.25 * oof[T]["gru_p2_v6"] + 0.25 * oof[T]["pset_v6"],
            0.35 * oof[T]["tcnc"] + 0.15 * oof[T]["gruc"] + 0.25 * oof[T]["gruc_p2_v6"] + 0.25 * oof[T]["psetc_v6"],
        ),
        lambda gid: (
            0.35 * sample_h[gid]["t_sym"] + 0.15 * sample_h[gid]["b_sym"] + 0.25 * pe_w5b["sample"][gid]["bigru_physv2_sym"] + 0.25 * pset_j23["sample"][gid]["pset_proba"],
            0.35 * sample_h[gid]["tc_sym"] + 0.15 * sample_h[gid]["bc_sym"] + 0.25 * pe_w5b["sample"][gid]["bigruc_physv2_sym"] + 0.25 * pset_j23["sample"][gid]["pset_change"],
        ),
        lambda: (
            0.35 * pe_plus["atd_raw"]["t_sym"] + 0.15 * atd_raw_extra["b_sym"] + 0.25 * pe_w5b["atd_raw"]["bigru_physv2_sym"] + 0.25 * pset_j23["atd_raw"]["pset_proba"],
            0.35 * pe_plus["atd_raw"]["tc_sym"] + 0.15 * atd_raw_extra["bc_sym"] + 0.25 * pe_w5b["atd_raw"]["bigruc_physv2_sym"] + 0.25 * pset_j23["atd_raw"]["pset_change"],
        ),
        lambda: (
            0.35 * atd_rep_h["t_sym"] + 0.15 * atd_rep_h["b_sym"] + 0.25 * pe_w5b["atd_rep"]["bigru_physv2_sym"] + 0.25 * pset_j23["atd_rep"]["pset_proba"],
            0.35 * atd_rep_h["tc_sym"] + 0.15 * atd_rep_h["bc_sym"] + 0.25 * pe_w5b["atd_rep"]["bigruc_physv2_sym"] + 0.25 * pset_j23["atd_rep"]["pset_change"],
        ),
    )

    # =========================================================================
    # STEP 4B: 6-FAMILY ENSEMBLE EVALUATION (4 TREES + 3 NEURAL FAMILIES: TCN, BIGRU, PSET)
    # =========================================================================
    print("\n--- Step 4B: 6-Family (`GBM + CatBoost + XGB + TCN + BiGRU + SymPlayerSetNet`) Multi-Judge Ablation ---", flush=True)

    with open(w6_dir / "w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)
    best_calib_cfg: TransductiveCalibConfig = w6_sum["best_calib_cfg"]
    lomo_rerank_cfgs = w6_sum["lomo_rerank_cfgs"]
    full_rerank_cfg = w6_sum["full_rerank_cfg"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)
    phys_rerank_cfg = SpellRerankerConfig(mode="physics", max_len_yxy=100, tau_absorb=0.38)
    snap_cfg = ImpulseSnapConfig(window=5, min_gain=0.10)

    def _blend_6fam_heads(
        p_5f: np.ndarray,
        c_5f: np.ndarray,
        tcn_244: np.ndarray,
        gru_244: np.ndarray,
        gru_481: np.ndarray,
        pset_p: np.ndarray,
        tcnc_244: np.ndarray,
        gruc_244: np.ndarray,
        gruc_481: np.ndarray,
        pset_c: np.ndarray,
        alpha_p2: np.ndarray,
        s3_d: float,
        s3_ha: float,
        s3_c: float,
        ps_ha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if s3_d == 0.0 and s3_ha == 0.0 and s3_c == 0.0 and ps_ha == 0.0:
            return p_5f, c_5f
        a3 = alpha_p2[:, None]
        g481 = a3 * gru_481 + (1.0 - a3) * gru_244
        ps3 = 0.35 * tcn_244 + 0.15 * gru_244 + 0.25 * g481 + 0.25 * pset_p
        cs3 = (
            0.35 * tcnc_244
            + 0.15 * gruc_244
            + 0.25 * (alpha_p2 * gruc_481 + (1.0 - alpha_p2) * gruc_244)
            + 0.25 * pset_c
        )
        p_d = np.clip((1.0 - s3_d) * p_5f[:, 2] + s3_d * ps3[:, 2], 1e-6, 1.0 - 1e-6)
        q_5f = p_5f[:, 0] / np.maximum(p_5f[:, 0] + p_5f[:, 1], 1e-6)
        q_s3 = ps3[:, 0] / np.maximum(ps3[:, 0] + ps3[:, 1], 1e-6)
        q_ps = pset_p[:, 0] / np.maximum(pset_p[:, 0] + pset_p[:, 1], 1e-6)
        q_h = np.clip((1.0 - s3_ha - ps_ha) * q_5f + s3_ha * q_s3 + ps_ha * q_ps, 1e-6, 1.0 - 1e-6)
        p_out = np.column_stack([(1.0 - p_d) * q_h, (1.0 - p_d) * (1.0 - q_h), p_d]).astype(np.float32)
        c_out = ((1.0 - s3_c) * c_5f + s3_c * cs3).astype(np.float32)
        return p_out, c_out

    def build_6family_streams(
        scfg: DecoupledSynthConfig,
        ccfg: TransductiveCalibConfig | None = None,
        s3_d: float = 0.00,
        s3_ha: float = 0.25,
        s3_c: float = 0.18,
        ps_ha: float = 0.10,
    ):
        """Synthesize 6-Family (9-Model: 4 Trees + SymGBM_v6 + SymTCN_244 + SymBiGRU_244 + SymTCN_481 + SymBiGRU_481 + SymPlayerSetNet) streams."""
        streams = {"v3": {}, "v6": {}, "sample": {}, "atd_raw": {}, "atd_rep": {}}
        include_pset = (s3_d > 0.0 or s3_ha > 0.0 or s3_c > 0.0 or ps_ha > 0.0)
        for T in MATCH_IDS:
            d = oof[T]
            h = sportec[T].half
            a_p2_v3 = compute_seq481_gate(X_v3[T], h, gates_v3[T][0])
            p3, c3 = synthesize_head_decoupled_proba(
                d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"],
                d["tcn"], d["gru"], d["tr_pl"], d["tr_p2_v3"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v3"], d["tr_v6_on_v3"],
                w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], cfg=scfg,
                tcn_physv2_proba=d["tcn_p2_v3"],
                bigru_physv2_proba=d["gru_p2_v3"],
                tcn_physv2_change=d["tcnc_p2_v3"],
                bigru_physv2_change=d["gruc_p2_v3"],
                alpha_p2=a_p2_v3,
            )
            p3, c3 = _blend_6fam_heads(
                p3, c3, d["tcn"], d["gru"], d["gru_p2_v3"], d["pset_v3"],
                d["tcnc"], d["gruc"], d["gruc_p2_v3"], d["psetc_v3"],
                a_p2_v3, s3_d, s3_ha, s3_c, ps_ha,
            )
            if ccfg is not None:
                p3 = calibrate_match_transductive(p3, h, X_v3[T], ccfg)
            f3 = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"]]
            if include_pset:
                f3.append(d["pset_v3"])
            streams["v3"][T] = (p3, c3, d["p4d_v3"], f3)

            a_p2_v6 = compute_seq481_gate(X_v6[T], h, gates_v6[T][0])
            p6, c6 = synthesize_head_decoupled_proba(
                d["g_v3_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn"], d["gru"], d["tr_v3_on_v6"], d["tr_p2_v6"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg,
                tcn_physv2_proba=d["tcn_p2_v6"],
                bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"],
                bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=a_p2_v6,
            )
            p6, c6 = _blend_6fam_heads(
                p6, c6, d["tcn"], d["gru"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc"], d["gruc"], d["gruc_p2_v6"], d["psetc_v6"],
                a_p2_v6, s3_d, s3_ha, s3_c, ps_ha,
            )
            if ccfg is not None:
                p6 = calibrate_match_transductive(p6, h, X_v6[T], ccfg)
            f6 = [d["g_v3_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tcn_p2_v6"], d["gru_p2_v6"]]
            if include_pset:
                f6.append(d["pset_v6"])
            streams["v6"][T] = (p6, c6, d["p4d_v6"], f6)

        for gid in GAMES:
            ms = sample_matches[gid]
            dh = sample_h[gid]
            dpl = pe_plus["sample"][gid]
            dv6 = sample_v6[gid]
            dp2 = pe_p2["sample"][gid]
            dw5 = pe_w5["sample"][gid]
            dw5b = pe_w5b["sample"][gid]
            dpset = pset_j23["sample"][gid]
            a_p2_s = compute_seq481_gate(X_sample[gid], ms.half, gates_sample[gid][0])

            ps, cs = synthesize_head_decoupled_proba(
                dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
                dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
                ms.half, dv6["g_v6"], dv6["tr_v6"],
                w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg,
                tcn_physv2_proba=dw5b["tcn_physv2_sym"],
                bigru_physv2_proba=dw5b["bigru_physv2_sym"],
                tcn_physv2_change=dw5b["tcnc_physv2_sym"],
                bigru_physv2_change=dw5b["bigruc_physv2_sym"],
                alpha_p2=a_p2_s,
            )
            ps, cs = _blend_6fam_heads(
                ps, cs, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"],
                dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"],
                a_p2_s, s3_d, s3_ha, s3_c, ps_ha,
            )
            if ccfg is not None:
                ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], ccfg)
            fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"]]
            if include_pset:
                fs.append(dpset["pset_proba"])
            streams["sample"][gid] = (ps, cs, b4d_j23["sample"][gid], fs)

        # J3 Raw
        dpl_ar = pe_plus["atd_raw"]
        dp2_ar = pe_p2["atd_raw"]
        dw5_ar = pe_w5["atd_raw"]
        dw5b_ar = pe_w5b["atd_raw"]
        dpset_ar = pset_j23["atd_raw"]
        a_p2_ar = compute_seq481_gate(X_atd_raw, atd.half, gate_atd_raw[0])
        par, car = synthesize_head_decoupled_proba(
            dpl_ar["g_plus_sym"], dp2_ar["g_physv2_sym"], dw5_ar["cb_physv2_sym"], dw5_ar["xgb_physv2_sym"],
            dpl_ar["t_sym"], atd_raw_extra["b_sym"], dpl_ar["tr_plus_sym"], dp2_ar["tr_physv2_sym"],
            dpl_ar["tc_sym"], atd_raw_extra["bc_sym"],
            atd.half, atd_raw_extra["g_v6"], atd_raw_extra["tr_v6"],
            w_ghost=gate_atd_raw[0], w_gap=gate_atd_raw[1], cfg=scfg,
            tcn_physv2_proba=dw5b_ar["tcn_physv2_sym"],
            bigru_physv2_proba=dw5b_ar["bigru_physv2_sym"],
            tcn_physv2_change=dw5b_ar["tcnc_physv2_sym"],
            bigru_physv2_change=dw5b_ar["bigruc_physv2_sym"],
            alpha_p2=a_p2_ar,
        )
        par, car = _blend_6fam_heads(
            par, car, dpl_ar["t_sym"], atd_raw_extra["b_sym"], dw5b_ar["bigru_physv2_sym"], dpset_ar["pset_proba"],
            dpl_ar["tc_sym"], atd_raw_extra["bc_sym"], dw5b_ar["bigruc_physv2_sym"], dpset_ar["pset_change"],
            a_p2_ar, s3_d, s3_ha, s3_c, ps_ha,
        )
        if ccfg is not None:
            par = calibrate_match_transductive(par, half_atd_dec, X_atd_raw, ccfg)
        far = [dpl_ar["g_plus_sym"], dp2_ar["g_physv2_sym"], dw5_ar["cb_physv2_sym"], dw5_ar["xgb_physv2_sym"], dpl_ar["t_sym"], atd_raw_extra["b_sym"], dw5b_ar["tcn_physv2_sym"], dw5b_ar["bigru_physv2_sym"]]
        if include_pset:
            far.append(dpset_ar["pset_proba"])
        streams["atd_raw"] = (par, car, b4d_j23["atd_raw"], far)

        # J3 Broadcast-Repaired
        dpl_ap = pe_plus["atd_rep"]
        dp2_ap = pe_p2["atd_rep"]
        dw5_ap = pe_w5["atd_rep"]
        dw5b_ap = pe_w5b["atd_rep"]
        dpset_ap = pset_j23["atd_rep"]
        a_p2_ap = compute_seq481_gate(X_atd_rep, atd.half, gate_atd_rep[0])
        pap, cap = synthesize_head_decoupled_proba(
            dpl_ap["g_plus_sym"], dp2_ap["g_physv2_sym"], dw5_ap["cb_physv2_sym"], dw5_ap["xgb_physv2_sym"],
            atd_rep_h["t_sym"], atd_rep_h["b_sym"], dpl_ap["tr_plus_sym"], dp2_ap["tr_physv2_sym"],
            atd_rep_h["tc_sym"], atd_rep_h["bc_sym"],
            atd.half, atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"],
            w_ghost=gate_atd_rep[0], w_gap=gate_atd_rep[1], cfg=scfg,
            tcn_physv2_proba=dw5b_ap["tcn_physv2_sym"],
            bigru_physv2_proba=dw5b_ap["bigru_physv2_sym"],
            tcn_physv2_change=dw5b_ap["tcnc_physv2_sym"],
            bigru_physv2_change=dw5b_ap["bigruc_physv2_sym"],
            alpha_p2=a_p2_ap,
        )
        pap, cap = _blend_6fam_heads(
            pap, cap, atd_rep_h["t_sym"], atd_rep_h["b_sym"], dw5b_ap["bigru_physv2_sym"], dpset_ap["pset_proba"],
            atd_rep_h["tc_sym"], atd_rep_h["bc_sym"], dw5b_ap["bigruc_physv2_sym"], dpset_ap["pset_change"],
            a_p2_ap, s3_d, s3_ha, s3_c, ps_ha,
        )
        if ccfg is not None:
            pap = calibrate_match_transductive(pap, half_atd_dec, X_atd_rep, ccfg)
        fap = [dpl_ap["g_plus_sym"], dp2_ap["g_physv2_sym"], dw5_ap["cb_physv2_sym"], dw5_ap["xgb_physv2_sym"], atd_rep_h["t_sym"], atd_rep_h["b_sym"], dw5b_ap["tcn_physv2_sym"], dw5b_ap["bigru_physv2_sym"]]
        if include_pset:
            fap.append(dpset_ap["pset_proba"])
        streams["atd_rep"] = (pap, cap, b4d_j23["atd_rep"], fap)
        return streams

    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )
    scfg_d60 = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.62, w_tree_ha=0.48, alpha_tv=0.10,
    )
    scfg_tv25 = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.55, w_tree_ha=0.52, alpha_tv=0.12,
    )

    st_5fam = build_6family_streams(scfg_main, best_calib_cfg, s3_d=0.0, s3_ha=0.0, s3_c=0.0, ps_ha=0.0)
    st_6fam_bal = build_6family_streams(scfg_main, best_calib_cfg, s3_d=0.00, s3_ha=0.25, s3_c=0.18, ps_ha=0.10)
    st_6fam_hi = build_6family_streams(scfg_main, best_calib_cfg, s3_d=0.05, s3_ha=0.25, s3_c=0.18, ps_ha=0.08)
    st_6fam_j1a = build_6family_streams(scfg_main, best_calib_cfg, s3_d=0.10, s3_ha=0.35, s3_c=0.20, ps_ha=0.00)
    st_6fam_bcast = build_6family_streams(scfg_main, best_calib_cfg, s3_d=0.00, s3_ha=0.00, s3_c=0.10, ps_ha=0.10)
    st_6fam_d60 = build_6family_streams(scfg_d60, best_calib_cfg, s3_d=0.00, s3_ha=0.25, s3_c=0.18, ps_ha=0.10)
    st_6fam_tv25 = build_6family_streams(scfg_tv25, best_calib_cfg, s3_d=0.00, s3_ha=0.25, s3_c=0.18, ps_ha=0.10)
    st_6fam_50 = st_6fam_hi

    def eval_6fam_pipeline(label: str, st_primary: dict, mode: str, sub_fn: str | None = None) -> dict:
        v3_f1, v3_acc, v3_xdx = [], [], 0
        v6_f1, v6_acc, v6_xdx = [], [], 0
        for T in MATCH_IDS:
            h = sportec[T].half
            y = sportec[T].labels
            p3, c3, p4d3, f3 = st_primary["v3"][T]
            p6, c6, p4d6, f6 = st_primary["v6"][T]
            if mode == "symdec":
                pr3 = decode_v4(p3, h, loo_dec_hyb[T], p_change=c3)
                pr6 = decode_v4(p6, h, loo_dec_hyb[T], p_change=c6)
            elif mode == "xdx":
                pr3 = apply_xdx_arbitrator(decode_v4(p3, h, loo_dec_hyb[T], p_change=c3), p3, h, X_v3[T], p_change=c3, p_trans_4d=p4d3, cfg=xdx_cfg_st1)
                pr6 = apply_xdx_arbitrator(decode_v4(p6, h, loo_dec_hyb[T], p_change=c6), p6, h, X_v6[T], p_change=c6, p_trans_4d=p4d6, cfg=xdx_cfg_st1)
            elif mode == "rerank":
                pr3 = apply_xdx_arbitrator(decode_v4(p3, h, loo_dec_hyb[T], p_change=c3), p3, h, X_v3[T], p_change=c3, p_trans_4d=p4d3, cfg=xdx_cfg_st1)
                pr6 = apply_xdx_arbitrator(decode_v4(p6, h, loo_dec_hyb[T], p_change=c6), p6, h, X_v6[T], p_change=c6, p_trans_4d=p4d6, cfg=xdx_cfg_st1)
                pr3 = apply_spell_reranker(pr3, p3, h, X_v3[T], c3, p4d3, f3[:8], cfg=lomo_rerank_cfgs[T])
                pr6 = apply_spell_reranker(pr6, p6, h, X_v6[T], c6, p4d6, f6[:8], cfg=lomo_rerank_cfgs[T])
            elif mode == "mbr_6fam":
                p3_5f, c3_5f, _, f3_5f = st_5fam["v3"][T]
                p6_5f, c6_5f, _, f6_5f = st_5fam["v6"][T]
                p3_d60, c3_d60, _, _ = st_6fam_d60["v3"][T]
                p6_d60, c6_d60, _, _ = st_6fam_d60["v6"][T]
                p3_tv25, c3_tv25, _, _ = st_6fam_tv25["v3"][T]
                p6_tv25, c6_tv25, _, _ = st_6fam_tv25["v6"][T]
                p3_50, c3_50, _, _ = st_6fam_50["v3"][T]
                p6_50, c6_50, _, _ = st_6fam_50["v6"][T]

                def _cands(pm, cm, p4d, fams, X_df, p5f, c5f, pd60, cd60, ptv, ctv, p50, c50, dp_hyb, dp_std, rr_cfg):
                    s1 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pm, h, dp_hyb, p_change=cm), pm, h, X_df, p_change=cm, p_trans_4d=p4d, cfg=xdx_cfg_st1), pm, h, X_df, cm, p4d, fams[:8], cfg=rr_cfg)
                    s2 = snap_boundaries_to_impulse(apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pm, h, dp_hyb, p_change=cm), pm, h, X_df, p_change=cm, p_trans_4d=p4d, cfg=xdx_cfg_st1), pm, h, X_df, cm, p4d, fams[:8], cfg=phys_rerank_cfg), h, X_df, cm, p4d, cfg=snap_cfg)
                    s3 = apply_spell_reranker(apply_xdx_arbitrator(decode_active_4dir(pm, h, dp_hyb, p_change=cm, p_trans_4d=p4d, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10), pm, h, X_df, p_change=cm, p_trans_4d=p4d, cfg=xdx_cfg_st1), pm, h, X_df, cm, p4d, fams[:8], cfg=rr_cfg)
                    s4 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pd60, h, dp_hyb, p_change=cd60), pd60, h, X_df, p_change=cd60, p_trans_4d=p4d, cfg=xdx_cfg_st1), pd60, h, X_df, cd60, p4d, fams[:8], cfg=rr_cfg)
                    s5 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ptv, h, dp_std, p_change=ctv), ptv, h, X_df, p_change=ctv, p_trans_4d=p4d, cfg=xdx_cfg_st1), ptv, h, X_df, ctv, p4d, fams[:8], cfg=rr_cfg)
                    s6 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(p5f, h, dp_hyb, p_change=c5f), p5f, h, X_df, p_change=c5f, p_trans_4d=p4d, cfg=xdx_cfg_st1), p5f, h, X_df, c5f, p4d, fams[:8], cfg=rr_cfg)
                    s7 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(p50, h, dp_hyb, p_change=c50), p50, h, X_df, p_change=c50, p_trans_4d=p4d, cfg=xdx_cfg_st1), p50, h, X_df, c50, p4d, fams[:8], cfg=rr_cfg)
                    return [s1, s2, s3, s4, s5, s6, s7]

                c3_list = _cands(p3, c3, p4d3, f3, X_v3[T], p3_5f, c3_5f, p3_d60, c3_d60, p3_tv25, c3_tv25, p3_50, c3_50, loo_dec_hyb[T], fold_dec_std[T], lomo_rerank_cfgs[T])
                c6_list = _cands(p6, c6, p4d6, f6, X_v6[T], p6_5f, c6_5f, p6_d60, c6_d60, p6_tv25, c6_tv25, p6_50, c6_50, loo_dec_hyb[T], fold_dec_std[T], lomo_rerank_cfgs[T])
                mbr_w = [0.22, 0.16, 0.16, 0.14, 0.10, 0.12, 0.10]
                pr3 = mbr_decode_sequences(c3_list, h, weights=mbr_w, include_vote_consensus=True, base_proba=p3, base_p_change=c3, dec_params=loo_dec_hyb[T])
                pr6 = mbr_decode_sequences(c6_list, h, weights=mbr_w, include_vote_consensus=True, base_proba=p6, base_p_change=c6, dec_params=loo_dec_hyb[T])

            v3_f1.append(float(metrics.score(y, pr3, h)["spell_f1_mean"]))
            v3_acc.append(frame_acc(y, pr3))
            v3_xdx += count_interior_xdx(pr3, h, X_v3[T])
            v6_f1.append(float(metrics.score(y, pr6, h)["spell_f1_mean"]))
            v6_acc.append(frame_acc(y, pr6))
            v6_xdx += count_interior_xdx(pr6, h, X_v6[T])

        s_raw, s_nod, s_xdx = [], [], 0
        for gid in GAMES:
            ms = sample_matches[gid]
            hs = ms.half
            ps, cs, p4ds, fs = st_primary["sample"][gid]
            if mode == "symdec":
                prs = decode_v4(ps, hs, dec_v5_hyb, p_change=cs)
            elif mode == "xdx":
                prs = apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1)
            elif mode == "rerank":
                prs = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps, hs, X_sample[gid], cs, p4ds, fs[:8], cfg=full_rerank_cfg)
            elif mode == "mbr_6fam":
                ps_5f, cs_5f, _, _ = st_5fam["sample"][gid]
                ps_d60, cs_d60, _, _ = st_6fam_d60["sample"][gid]
                ps_tv25, cs_tv25, _, _ = st_6fam_tv25["sample"][gid]
                ps_50, cs_50, _, _ = st_6fam_50["sample"][gid]
                s1 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps, hs, X_sample[gid], cs, p4ds, fs[:8], cfg=full_rerank_cfg)
                s2 = snap_boundaries_to_impulse(apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps, hs, dec_v5_hyb, p_change=cs), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps, hs, X_sample[gid], cs, p4ds, fs[:8], cfg=phys_rerank_cfg), hs, X_sample[gid], cs, p4ds, cfg=snap_cfg)
                s3 = apply_spell_reranker(apply_xdx_arbitrator(decode_active_4dir(ps, hs, dec_v5_hyb, p_change=cs, p_trans_4d=p4ds, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10), ps, hs, X_sample[gid], p_change=cs, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps, hs, X_sample[gid], cs, p4ds, fs[:8], cfg=full_rerank_cfg)
                s4 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps_d60, hs, dec_v5_hyb, p_change=cs_d60), ps_d60, hs, X_sample[gid], p_change=cs_d60, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps_d60, hs, X_sample[gid], cs_d60, p4ds, fs[:8], cfg=full_rerank_cfg)
                s5 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps_tv25, hs, dec_v4, p_change=cs_tv25), ps_tv25, hs, X_sample[gid], p_change=cs_tv25, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps_tv25, hs, X_sample[gid], cs_tv25, p4ds, fs[:8], cfg=full_rerank_cfg)
                s6 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps_5f, hs, dec_v5_hyb, p_change=cs_5f), ps_5f, hs, X_sample[gid], p_change=cs_5f, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps_5f, hs, X_sample[gid], cs_5f, p4ds, fs[:8], cfg=full_rerank_cfg)
                s7 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(ps_50, hs, dec_v5_hyb, p_change=cs_50), ps_50, hs, X_sample[gid], p_change=cs_50, p_trans_4d=p4ds, cfg=xdx_cfg_st1), ps_50, hs, X_sample[gid], cs_50, p4ds, fs[:8], cfg=full_rerank_cfg)
                mbr_w = [0.22, 0.16, 0.16, 0.14, 0.10, 0.12, 0.10]
                prs = mbr_decode_sequences([s1, s2, s3, s4, s5, s6, s7], hs, weights=mbr_w, include_vote_consensus=True, base_proba=ps, base_p_change=cs, dec_params=dec_v5_hyb)
            s_raw.append(float(metrics.score(ms.labels, prs, hs)["spell_f1_mean"]))
            s_nod.append(float(metrics.score(without_dead_margin(ms.labels, hs), prs, hs)["spell_f1_mean"]))
            s_xdx += count_interior_xdx(prs, hs, X_sample[gid])

        def _dec_atd(vkey: str, X_df: pd.DataFrame):
            pa, ca, p4da, fa = st_primary[vkey]
            if mode == "symdec":
                pra = decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca)
            elif mode == "xdx":
                pra = apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1)
            elif mode == "rerank":
                pra = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa, half_atd_dec, X_df, ca, p4da, fa[:8], cfg=full_rerank_cfg)
            elif mode == "mbr_6fam":
                pa_5f, ca_5f, _, _ = st_5fam[vkey]
                pa_d60, ca_d60, _, _ = st_6fam_d60[vkey]
                pa_tv25, ca_tv25, _, _ = st_6fam_tv25[vkey]
                pa_50, ca_50, _, _ = st_6fam_50[vkey]
                s1 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa, half_atd_dec, X_df, ca, p4da, fa[:8], cfg=full_rerank_cfg)
                s2 = snap_boundaries_to_impulse(apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa, half_atd_dec, X_df, ca, p4da, fa[:8], cfg=phys_rerank_cfg), half_atd_dec, X_df, ca, p4da, cfg=snap_cfg)
                s3 = apply_spell_reranker(apply_xdx_arbitrator(decode_active_4dir(pa, half_atd_dec, dec_v5_hyb, p_change=ca, p_trans_4d=p4da, beta_dir_ha=0.25, beta_to_d=0.10, beta_from_d=0.10), pa, half_atd_dec, X_df, p_change=ca, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa, half_atd_dec, X_df, ca, p4da, fa[:8], cfg=full_rerank_cfg)
                s4 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa_d60, half_atd_dec, dec_v5_hyb, p_change=ca_d60), pa_d60, half_atd_dec, X_df, p_change=ca_d60, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa_d60, half_atd_dec, X_df, ca_d60, p4da, fa[:8], cfg=full_rerank_cfg)
                s5 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa_tv25, half_atd_dec, dec_v4, p_change=ca_tv25), pa_tv25, half_atd_dec, X_df, p_change=ca_tv25, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa_tv25, half_atd_dec, X_df, ca_tv25, p4da, fa[:8], cfg=full_rerank_cfg)
                s6 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa_5f, half_atd_dec, dec_v5_hyb, p_change=ca_5f), pa_5f, half_atd_dec, X_df, p_change=ca_5f, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa_5f, half_atd_dec, X_df, ca_5f, p4da, fa[:8], cfg=full_rerank_cfg)
                s7 = apply_spell_reranker(apply_xdx_arbitrator(decode_v4(pa_50, half_atd_dec, dec_v5_hyb, p_change=ca_50), pa_50, half_atd_dec, X_df, p_change=ca_50, p_trans_4d=p4da, cfg=xdx_cfg_st1), pa_50, half_atd_dec, X_df, ca_50, p4da, fa[:8], cfg=full_rerank_cfg)
                mbr_w = [0.22, 0.16, 0.16, 0.14, 0.10, 0.12, 0.10]
                pra = mbr_decode_sequences([s1, s2, s3, s4, s5, s6, s7], half_atd_dec, weights=mbr_w, include_vote_consensus=True, base_proba=pa, base_p_change=ca, dec_params=dec_v5_hyb)
            pra[second_start:kick2] = DEAD
            pra[~atd.graded] = DEAD
            return pra

        pred_raw = _dec_atd("atd_raw", X_atd_raw)
        pred_rep = _dec_atd("atd_rep", X_atd_rep)
        pub_raw = metrics.score_public(pred_raw, bootstrap_n=0)
        pub_rep = metrics.score_public(pred_rep, bootstrap_n=0)
        pl_rep = plausibility(pred_rep, atd.half, atd.graded)
        xdx_raw = count_interior_xdx(pred_raw, half_atd_dec, X_atd_raw)
        xdx_rep = count_interior_xdx(pred_rep, half_atd_dec, X_atd_rep)

        sub_meta = None
        if sub_fn is not None:
            sub_meta = save_and_validate_submission(sub_dir / sub_fn, pred_rep, atd, second_start, kick2)

        res = {
            "label": label,
            "j1_v3_f1": float(np.mean(v3_f1)),
            "j1_v3_acc": float(np.mean(v3_acc)),
            "j1_v3_xdx": int(v3_xdx),
            "j1_v6_f1": float(np.mean(v6_f1)),
            "j1_v6_acc": float(np.mean(v6_acc)),
            "j1_v6_xdx": int(v6_xdx),
            "j2_raw_f1": float(np.mean(s_raw)),
            "j2_nod_f1": float(np.mean(s_nod)),
            "j2_xdx": int(s_xdx),
            "j3_raw_f1": float(pub_raw["spell_f1_mean"]),
            "j3_raw_df1": float(pub_raw["alive_dead_f1"]),
            "j3_raw_xdx": int(xdx_raw),
            "j3_rep_f1": float(pub_rep["spell_f1_mean"]),
            "j3_rep_df1": float(pub_rep["alive_dead_f1"]),
            "j3_rep_pm": float(pl_rep["poss_per_min"]),
            "j3_rep_pctd": float(pl_rep["pct_D"]),
            "j3_rep_xdx": int(xdx_rep),
            "sub_meta": sub_meta,
        }
        print(
            f"  {label:68s} | "
            f"{res['j1_v3_f1']:.4f} ({res['j1_v3_acc']:.3f},{res['j1_v3_xdx']:2d}) | "
            f"{res['j1_v6_f1']:.4f} ({res['j1_v6_acc']:.3f},{res['j1_v6_xdx']:2d}) | "
            f"{res['j2_raw_f1']:.4f} ({res['j2_nod_f1']:.4f},{res['j2_xdx']:2d}) | "
            f"{res['j3_raw_f1']:.4f} ({res['j3_raw_df1']:.3f},{res['j3_raw_xdx']:2d}) | "
            f"{res['j3_rep_f1']:.4f} ({res['j3_rep_df1']:.3f}, {res['j3_rep_pm']:4.2f}, {res['j3_rep_pctd']:4.1f}%, {res['j3_rep_xdx']:2d})",
            flush=True,
        )
        return res

    header = (
        f"  {'Pipeline / Ensemble':68s} | {'J1A: v3 (acc,xdx)':19s} | {'J1B: v6 (acc,xdx)':19s} | "
        f"{'J2: Samp (noD,xdx)':20s} | {'J3 Raw (D_f1,xdx)':19s} | {'J3 Rep (D_f1, p/m, %D, xdx)':30s}"
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    summary_rows = []
    summary_rows.append(eval_6fam_pipeline("0. Wave 6 5-Fam Baseline (s3=0, ps=0) + Calib + SymDec", st_5fam, "symdec"))
    summary_rows.append(eval_6fam_pipeline("1. Wave 7A 6-Fam Balanced (s3=0/25/18, ps_ha=10) + Calib + SymDec", st_6fam_bal, "symdec"))
    summary_rows.append(eval_6fam_pipeline("2. Wave 7A 6-Fam High-J1/J2 (s3=5/25/18, ps_ha=8) + Calib + SymDec", st_6fam_hi, "symdec"))
    summary_rows.append(eval_6fam_pipeline("3. Wave 7A 6-Fam High-J1A (s3=10/35/20, ps_ha=0) + Calib + SymDec", st_6fam_j1a, "symdec"))
    summary_rows.append(eval_6fam_pipeline("4. Wave 7A 6-Fam Broadcast-Spec (s3=0/0/10, ps_ha=10) + Calib + SymDec", st_6fam_bcast, "symdec"))
    summary_rows.append(eval_6fam_pipeline("5. Wave 6 5-Fam Baseline + XDX(FixA) + LOMO Rerank", st_5fam, "rerank"))
    summary_rows.append(eval_6fam_pipeline("6. Wave 7A 6-Fam Balanced + XDX(FixA) + LOMO Rerank", st_6fam_bal, "rerank", "submission_w7a_6fam_rerank.csv"))
    summary_rows.append(eval_6fam_pipeline("7. Wave 7A 6-Fam High-J1/J2 + XDX(FixA) + LOMO Rerank", st_6fam_hi, "rerank", "submission_w7a_6fam_hi_rerank.csv"))
    summary_rows.append(eval_6fam_pipeline("8. Wave 7A 6-Fam High-J1A + XDX(FixA) + LOMO Rerank", st_6fam_j1a, "rerank"))
    summary_rows.append(eval_6fam_pipeline("9. Wave 7A 6-Fam Broadcast-Spec + XDX(FixA) + LOMO Rerank", st_6fam_bcast, "rerank", "submission_w7a_6fam_bcast_spec.csv"))
    summary_rows.append(eval_6fam_pipeline("10. Wave 7A 6-Fam Balanced + Direct `spell_scores` MBR (7-Cand)", st_6fam_bal, "mbr_6fam", "submission_w7a_6fam_mbr_champion.csv"))
    summary_rows.append(eval_6fam_pipeline("11. Wave 7A 6-Fam High-J1/J2 + Direct `spell_scores` MBR (7-Cand)", st_6fam_hi, "mbr_6fam", "submission_w7a_6fam_hi_mbr.csv"))

    with open(out_dir / "w7a_results_summary.pkl", "wb") as f:
        pickle.dump(summary_rows, f)

    print(f"\nCompleted Wave 7A (`EXP-11` SymPlayerSetNet) evaluation in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
