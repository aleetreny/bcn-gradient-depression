"""Wave 5B (EXP-07): Neural Sequence Models (`SymTCN_PhysV2_TTA` & `SymBiGRU_PhysV2_TTA`) on `v1ctl_physics_v2` (481 features).

Architecture & Experimental Design:
- Prior neural sequence models (`SymTCN_TTA` and `SymBiGRU_TTA`) were trained only on the 244-column `v1ctl`
  feature table, while tree models (`SymGBM_PhysV2`, `SymCatBoost_PhysV2`, `SymXGB_PhysV2`) already benefit from
  the 481-column `v1ctl_physics_v2` table (`244 v1ctl + 108 plus + 129 physv2`).
- Giving the dilated causal/bidirectional TCN (`TCNModel`, receptive field ~12s) and recurrent BiGRU (`BiGRUSeqModel`)
  direct access to multi-scale vector impulse (`J_2, J_4, J_8`), pass-chord mid-flight score (`chord_in_flight`),
  impulse-gated distances (`diff_d1_impulse_gated`), collective expansion-contraction dipole (`diff_sigma_x`,
  `diff_eff_area`, `diff_def3_line_x`), and ball-free pressure convergence focus (`focus_diff_d1`, `focus_diff_inflow`,
  `ball_focus_discrepancy`) allows the sequence encoders to learn continuous temporal patterns over physical dynamics.
- Both sequence models are trained with strict 2x H <-> A + 180-deg pitch rotation symmetry augmentation
  (`mirror_v1ctl_physv2_df`, `swap_labels_ha`) and evaluated with exact equivariant test-time augmentation
  (`predict_seq_physv2_tta`).

Evaluation Protocol (All 4 Judges):
- Step 1: Verify exact H <-> A equivariance of `predict_seq_physv2_tta` on `J03WMX` for both `TCNModel` and `BiGRUSeqModel`.
- Step 2: Load the 481-column `v1ctl_physics_v2` feature tables (`train_X_v3`, `train_X_v3_mir`, `test_X_v3`, `test_X_v6`).
- Step 3: Run strict 7-fold LOMO + Full 7-match training with incremental fold checkpointing to `experiments/outputs/exp_w5b_oof/`:
  - `tcn_physv2_{T}.npz` (`v3_proba`, `v3_change`, `v6_proba`, `v6_change`) for `TCNModel(norm="channel", epochs=18, seed=k)`
  - `bigru_physv2_{T}.npz` (`v3_proba`, `v3_change`, `v6_proba`, `v6_change`) for `BiGRUSeqModel(norm="channel", epochs=12, seed=k)`
  - `symtcn_physv2_full.pkl` (`epochs=18, seed=0`) and `symbigru_physv2_full.pkl` (`epochs=12, seed=0`)
  - `seq_physv2_j23_preds.pkl` (Metrica Sample 3 games, `atd_raw`, `atd_rep`)
- Step 4: Comprehensive 4-Judge evaluation across standalone sequence models, multi-representation sequence blends
  (`v1ctl` 244 + `physv2` 481), and full multi-family ensembles (with `SymGBM_PhysV2`, `SymGBM_Plus`, and
  `SymCatBoost_PhysV2` + `SymXGB_PhysV2` if ready) under both `decode_v4` and Wave 3 `decode_v2` + `apply_xdx_arbitrator`.
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
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.decode.viterbi_hsmm_v2 import DecoderParamsV2, decode_v2
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator, rle_segments
from src.eval import metrics
from src.features.symmetry import swap_labels as swap_labels_ha, swap_proba as swap_proba_ha
from src.features.v1ctl_physics_v2 import (
    build_v1ctl_physics_v2,
    mirror_v1ctl_physv2_df as mirror_physv2_df,
)
from src.models.bigru import BiGRUSeqModel
from src.models.boundary_4dir import targets_4dir
from src.models.tcn import TCNModel
from src.models.transition import change_features
from exp_b_symmetry_eval import blend_proba, predict_tcn_tta
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match


def build_4dir_targets(labels: np.ndarray, half: np.ndarray) -> np.ndarray:
    d = targets_4dir(labels, half)
    return np.column_stack([d["y_h2a"], d["y_a2h"], d["y_live2d"], d["y_d2live"]])


def count_interior_xdx_islands(pred: np.ndarray, half: np.ndarray, X: pd.DataFrame, max_len: int = 240) -> tuple[int, int]:
    """Count total X->D->X segments (len <= max_len) and how many stay inside the pitch interior."""
    s_arr, e_arr, v_arr = rle_segments(pred, half)
    n_tot, n_int = 0, 0
    bx_c = np.abs(X["ball_last_x"].to_numpy(np.float64) - 52.5) if "ball_last_x" in X.columns else np.zeros(len(pred))
    by_c = np.abs(X["ball_last_y"].to_numpy(np.float64) - 34.0) if "ball_last_y" in X.columns else np.zeros(len(pred))
    if np.nanmin(X["ball_last_x"].to_numpy(np.float64)) < -10.0:
        bx_c = np.abs(X["ball_last_x"].to_numpy(np.float64))
        by_c = np.abs(X["ball_last_y"].to_numpy(np.float64))
    for i in range(1, len(s_arr) - 1):
        if v_arr[i] != DEAD:
            continue
        if v_arr[i - 1] in (HOME, AWAY) and v_arr[i - 1] == v_arr[i + 1]:
            a, b = int(s_arr[i]), int(e_arr[i])
            if half[a] <= 0 or half[s_arr[i - 1]] != half[a] or half[e_arr[i + 1] - 1] != half[a]:
                continue
            if (b - a) <= max_len:
                n_tot += 1
                mx = float(np.nanmax(bx_c[a:b])) if np.any(np.isfinite(bx_c[a:b])) else 0.0
                my = float(np.nanmax(by_c[a:b])) if np.any(np.isfinite(by_c[a:b])) else 0.0
                if mx < 50.0 and my < 33.5:
                    n_int += 1
    return n_tot, n_int


def predict_seq_physv2_tta(
    model,
    X_df: pd.DataFrame,
    half: np.ndarray,
    pitch=(105.0, 68.0),
    X_mir: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Equivariant test-time augmentation for a sequence model (TCN or BiGRU) on 481-col `v1ctl_physics_v2`."""
    p1, c1 = model.predict(X_df, half)
    if X_mir is None:
        X_mir = mirror_physv2_df(X_df, half, pitch)
    p2_swap, c2 = model.predict(X_mir, half)
    return 0.5 * (p1 + swap_proba_ha(p2_swap)), 0.5 * (c1 + c2)


def load_cached_physv2(
    key_prefix: str,
    half: np.ndarray,
    match_fn=None,
) -> pd.DataFrame:
    """Fast loader for 481-column `v1ctl_physics_v2` directly from the 4 cached parquet blocks (~0.35s/table)."""
    cache_root = ROOT / "local_data" / "cache"

    def _find(sub: str):
        cands = list((cache_root / sub).glob(f"{key_prefix}_*.parquet"))
        if key_prefix == "atd":
            cands = [p for p in cands if not p.name.startswith("atd_rep")]
        return cands

    f1, f2, f3, f4 = _find("features_v1"), _find("control_v1"), _find("plus_v1"), _find("physv2")
    if len(f1) == 1 and len(f2) == 1 and len(f3) == 1 and len(f4) == 1:
        X1 = change_features(pd.read_parquet(f1[0]), half)
        X2 = pd.read_parquet(f2[0])
        X3 = pd.read_parquet(f3[0])
        X4 = pd.read_parquet(f4[0])
        df = pd.concat([X1, X2, X3, X4], axis=1)
        assert df.shape[1] == 481, f"Expected 481 cols for {key_prefix}, got {df.shape[1]}"
        return df
    if match_fn is None:
        raise FileNotFoundError(f"Cache files missing for {key_prefix} and no match_fn provided")
    return build_v1ctl_physics_v2(match_fn(), key_prefix)


def combine_seq_tcn_bigru(
    p_tcn: np.ndarray,
    c_tcn: np.ndarray,
    p_gru: np.ndarray,
    c_gru: np.ndarray,
    w_gru_alive: float = 0.0,
    w_gru_ha: float = 0.20,
    w_gru_change: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decoupled sequence combination: keep TCN's sharp `P_alive` while blending BiGRU into `P(H|alive)`."""
    eps = 1e-6
    p_d_tcn = np.clip(p_tcn[:, 2], eps, 1.0 - eps)
    p_d_gru = np.clip(p_gru[:, 2], eps, 1.0 - eps)
    p_alive = (1.0 - w_gru_alive) * (1.0 - p_d_tcn) + w_gru_alive * (1.0 - p_d_gru)
    p_d = 1.0 - p_alive

    s_tcn = np.maximum(p_tcn[:, 0] + p_tcn[:, 1], eps)
    s_gru = np.maximum(p_gru[:, 0] + p_gru[:, 1], eps)
    h_tcn = p_tcn[:, 0] / s_tcn
    h_gru = p_gru[:, 0] / s_gru
    h_ens = (1.0 - w_gru_ha) * h_tcn + w_gru_ha * h_gru

    out = np.zeros_like(p_tcn, dtype=np.float32)
    out[:, 0] = p_alive * h_ens
    out[:, 1] = p_alive * (1.0 - h_ens)
    out[:, 2] = p_d
    c_ens = (1.0 - w_gru_change) * c_tcn + w_gru_change * c_gru
    return out, c_ens.astype(np.float32)


def main():
    t0 = time.time()
    out_dir = ROOT / "experiments" / "outputs" / "exp_w5b_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 105, flush=True)
    print("WAVE 5B (EXP-07): NEURAL SEQUENCE MODELS (SymTCN_PhysV2_TTA & SymBiGRU_PhysV2_TTA) ON 481 FEATURES", flush=True)
    print("=" * 105, flush=True)

    sportec = load_sportec()
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")

    # =========================================================================
    # STEP 1: VERIFY EXACT H <-> A EQUIVARIANCE OF predict_seq_physv2_tta
    # =========================================================================
    print("\n--- Step 1: Verifying Exact H <-> A Equivariance of `predict_seq_physv2_tta` (481 cols) ---", flush=True)
    m0 = sportec["J03WMX"]
    X0_full = load_cached_physv2(
        "sportec_J03WMX_degrade_v3_s0",
        m0.half,
        lambda: degrade(m0, cfg_v3, 0),
    )
    print(f"  Loaded J03WMX degrade_v3_s0 table: shape={X0_full.shape} (expected 481 columns)", flush=True)

    sl = slice(1000, 4500)
    h_sub = m0.half[sl]
    y_sub = m0.labels[sl]
    # Project slice onto involution manifold so boundary-window rolling features satisfy mirror(mirror(X_sub)) == X_sub
    X_sub_raw = X0_full.iloc[sl].reset_index(drop=True)
    X_sub_mir = mirror_physv2_df(X_sub_raw, h_sub, m0.pitch)
    X_sub = mirror_physv2_df(X_sub_mir, h_sub, m0.pitch)
    X_sub_mir = mirror_physv2_df(X_sub, h_sub, m0.pitch)
    inv_err = float(np.nanmax(np.abs(X_sub.to_numpy(np.float32) - mirror_physv2_df(X_sub_mir, h_sub, m0.pitch).to_numpy(np.float32))))
    print(f"  Involution check max |X - mirror(mirror(X))| on slice: {inv_err:.3e}", flush=True)

    tcn_toy = TCNModel(norm="channel", epochs=1, seed=0).fit(
        [X_sub, X_sub_mir],
        [y_sub, swap_labels_ha(y_sub)],
        [h_sub, h_sub],
    )
    p_tcn_1, c_tcn_1 = predict_seq_physv2_tta(tcn_toy, X_sub, h_sub, m0.pitch)
    p_tcn_2, c_tcn_2 = predict_seq_physv2_tta(tcn_toy, X_sub_mir, h_sub, m0.pitch)
    err_tcn_p = float(np.max(np.abs(p_tcn_1 - swap_proba_ha(p_tcn_2))))
    err_tcn_c = float(np.max(np.abs(c_tcn_1 - c_tcn_2)))
    print(
        f"  [TCNModel Equivariance]      max |p(X) - swap(p(mirror(X)))| = {err_tcn_p:.3e} | "
        f"max |c(X) - c(mirror(X))| = {err_tcn_c:.3e}",
        flush=True,
    )
    assert err_tcn_p < 1e-5 and err_tcn_c < 1e-5, f"TCN equivariance failed: {err_tcn_p}, {err_tcn_c}"

    gru_toy = BiGRUSeqModel(norm="channel", epochs=1, seed=0).fit(
        [X_sub, X_sub_mir],
        [y_sub, swap_labels_ha(y_sub)],
        [h_sub, h_sub],
    )
    p_gru_1, c_gru_1 = predict_seq_physv2_tta(gru_toy, X_sub, h_sub, m0.pitch)
    p_gru_2, c_gru_2 = predict_seq_physv2_tta(gru_toy, X_sub_mir, h_sub, m0.pitch)
    err_gru_p = float(np.max(np.abs(p_gru_1 - swap_proba_ha(p_gru_2))))
    err_gru_c = float(np.max(np.abs(c_gru_1 - c_gru_2)))
    print(
        f"  [BiGRUSeqModel Equivariance] max |p(X) - swap(p(mirror(X)))| = {err_gru_p:.3e} | "
        f"max |c(X) - c(mirror(X))| = {err_gru_c:.3e}",
        flush=True,
    )
    assert err_gru_p < 1e-5 and err_gru_c < 1e-5, f"BiGRU equivariance failed: {err_gru_p}, {err_gru_c}"
    del tcn_toy, gru_toy, X_sub_raw, X_sub, X_sub_mir, X0_full
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    # =========================================================================
    # STEP 2: LOAD 481-COLUMN `v1ctl_physics_v2` FEATURE TABLES FOR SPORTEC
    # =========================================================================
    print("\n--- Step 2: Loading 481-Column `v1ctl_physics_v2` Tables for Sportec (7 Matches) ---", flush=True)
    t_load = time.time()

    need_lomo_train = any(
        not (out_dir / f"tcn_physv2_{T}.npz").exists() or not (out_dir / f"bigru_physv2_{T}.npz").exists()
        for T in MATCH_IDS
    )
    need_full_train = (
        not (out_dir / "symtcn_physv2_full.pkl").exists()
        or not (out_dir / "symbigru_physv2_full.pkl").exists()
    )

    train_X_v3: dict[tuple[str, str], pd.DataFrame] = {}
    train_X_v3_mir: dict[tuple[str, str], pd.DataFrame] = {}
    train_y_v3: dict[tuple[str, str], np.ndarray] = {}
    train_y_v3_mir: dict[tuple[str, str], np.ndarray] = {}
    train_h_v3: dict[tuple[str, str], np.ndarray] = {}

    test_X_v3: dict[str, pd.DataFrame] = {}
    test_X_v6: dict[str, pd.DataFrame] = {}

    for mid in MATCH_IDS:
        m = sportec[mid]
        if need_lomo_train or need_full_train:
            for key, overrides, seed in TRAIN_VARIANTS:
                cfg = replace(cfg_v3, **(overrides or {}))
                X = load_cached_physv2(
                    f"sportec_{mid}_degrade_v3_{key}",
                    m.half,
                    lambda m=m, cfg=cfg, seed=seed: degrade(m, cfg, seed),
                )
                X_mir = mirror_physv2_df(X, m.half, m.pitch)
                train_X_v3[(mid, key)] = X
                train_X_v3_mir[(mid, key)] = X_mir
                train_y_v3[(mid, key)] = m.labels
                train_y_v3_mir[(mid, key)] = swap_labels_ha(m.labels)
                train_h_v3[(mid, key)] = m.half

        # Test v3 (s0) & Test v6 (s0)
        test_X_v3[mid] = load_cached_physv2(
            f"sportec_{mid}_degrade_v3_s0",
            m.half,
            lambda m=m: degrade(m, cfg_v3, 0),
        )
        test_X_v6[mid] = load_cached_physv2(
            f"sportec_{mid}_degrade_v6_s0",
            m.half,
            lambda m=m: degrade(m, cfg_v6, 0),
        )

    gc.collect()
    print(f"  Loaded Sportec 481-col tables + mirrored training views in {time.time() - t_load:.1f}s", flush=True)

    # =========================================================================
    # STEP 3: 7-FOLD LOMO + FULL 7-MATCH TRAINING (TCN & BiGRU)
    # =========================================================================
    print("\n--- Step 3A: 7-Fold LOMO Training of `SymTCN_PhysV2_TTA` (epochs=18) & `SymBiGRU_PhysV2_TTA` (epochs=12) ---", flush=True)
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    h_dir = ROOT / "experiments" / "outputs" / "exp_h_oof"
    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std, fold_dec_hyb, _, _ = pickle.load(f)

    oof_tcn: dict[str, dict[str, np.ndarray]] = {}
    oof_bigru: dict[str, dict[str, np.ndarray]] = {}

    for k, T in enumerate(MATCH_IDS):
        tcn_path = out_dir / f"tcn_physv2_{T}.npz"
        gru_path = out_dir / f"bigru_physv2_{T}.npz"
        train_ids = [m for m in MATCH_IDS if m != T]

        if not tcn_path.exists() or not gru_path.exists():
            Xs = [train_X_v3[(m, key)] for m in train_ids for key, _, _ in TRAIN_VARIANTS] + [
                train_X_v3_mir[(m, key)] for m in train_ids for key, _, _ in TRAIN_VARIANTS
            ]
            ys = [train_y_v3[(m, key)] for m in train_ids for key, _, _ in TRAIN_VARIANTS] + [
                train_y_v3_mir[(m, key)] for m in train_ids for key, _, _ in TRAIN_VARIANTS
            ]
            hs = [train_h_v3[(m, key)] for m in train_ids for key, _, _ in TRAIN_VARIANTS] + [
                train_h_v3[(m, key)] for m in train_ids for key, _, _ in TRAIN_VARIANTS
            ]
            X_v3_mir_T = mirror_physv2_df(test_X_v3[T], sportec[T].half, sportec[T].pitch)
            X_v6_mir_T = mirror_physv2_df(test_X_v6[T], sportec[T].half, sportec[T].pitch)

        # 1) SymTCN_PhysV2_TTA fold k
        if tcn_path.exists():
            z = np.load(tcn_path)
            oof_tcn[T] = {k_: z[k_] for k_ in z.files}
            f1_v3 = metrics.score(
                sportec[T].labels,
                decode_v4(oof_tcn[T]["v3_proba"], sportec[T].half, fold_dec_std[T], p_change=oof_tcn[T]["v3_change"]),
                sportec[T].half,
            )["spell_f1_mean"]
            f1_v6 = metrics.score(
                sportec[T].labels,
                decode_v4(oof_tcn[T]["v6_proba"], sportec[T].half, fold_dec_std[T], p_change=oof_tcn[T]["v6_change"]),
                sportec[T].half,
            )["spell_f1_mean"]
            print(f"  [Fold {k} ({T})] Loaded tcn_physv2_{T}.npz from cache | v3_SpellF1={f1_v3:.4f} | v6_SpellF1={f1_v6:.4f}", flush=True)
        else:
            tf = time.time()
            symtcn = TCNModel(norm="channel", epochs=18, seed=k).fit(Xs, ys, hs)
            v3_p, v3_c = predict_seq_physv2_tta(
                symtcn, test_X_v3[T], sportec[T].half, sportec[T].pitch, X_mir=X_v3_mir_T
            )
            v6_p, v6_c = predict_seq_physv2_tta(
                symtcn, test_X_v6[T], sportec[T].half, sportec[T].pitch, X_mir=X_v6_mir_T
            )
            oof_tcn[T] = {
                "v3_proba": v3_p,
                "v3_change": v3_c,
                "v6_proba": v6_p,
                "v6_change": v6_c,
            }
            np.savez_compressed(tcn_path, **oof_tcn[T])
            del symtcn
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            f1_v3 = metrics.score(
                sportec[T].labels,
                decode_v4(v3_p, sportec[T].half, fold_dec_std[T], p_change=v3_c),
                sportec[T].half,
            )["spell_f1_mean"]
            f1_v6 = metrics.score(
                sportec[T].labels,
                decode_v4(v6_p, sportec[T].half, fold_dec_std[T], p_change=v6_c),
                sportec[T].half,
            )["spell_f1_mean"]
            print(
                f"  [Fold {k} ({T})] SymTCN_PhysV2_TTA trained & saved in {time.time() - tf:.1f}s "
                f"| v3_SpellF1={f1_v3:.4f} | v6_SpellF1={f1_v6:.4f}",
                flush=True,
            )

        # 2) SymBiGRU_PhysV2_TTA fold k
        if gru_path.exists():
            z = np.load(gru_path)
            oof_bigru[T] = {k_: z[k_] for k_ in z.files}
            f1_v3 = metrics.score(
                sportec[T].labels,
                decode_v4(oof_bigru[T]["v3_proba"], sportec[T].half, fold_dec_std[T], p_change=oof_bigru[T]["v3_change"]),
                sportec[T].half,
            )["spell_f1_mean"]
            f1_v6 = metrics.score(
                sportec[T].labels,
                decode_v4(oof_bigru[T]["v6_proba"], sportec[T].half, fold_dec_std[T], p_change=oof_bigru[T]["v6_change"]),
                sportec[T].half,
            )["spell_f1_mean"]
            print(f"  [Fold {k} ({T})] Loaded bigru_physv2_{T}.npz from cache | v3_SpellF1={f1_v3:.4f} | v6_SpellF1={f1_v6:.4f}", flush=True)
        else:
            tf = time.time()
            symbigru = BiGRUSeqModel(norm="channel", epochs=12, seed=k).fit(Xs, ys, hs)
            v3_p, v3_c = predict_seq_physv2_tta(
                symbigru, test_X_v3[T], sportec[T].half, sportec[T].pitch, X_mir=X_v3_mir_T
            )
            v6_p, v6_c = predict_seq_physv2_tta(
                symbigru, test_X_v6[T], sportec[T].half, sportec[T].pitch, X_mir=X_v6_mir_T
            )
            oof_bigru[T] = {
                "v3_proba": v3_p,
                "v3_change": v3_c,
                "v6_proba": v6_p,
                "v6_change": v6_c,
            }
            np.savez_compressed(gru_path, **oof_bigru[T])
            del symbigru
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            f1_v3 = metrics.score(
                sportec[T].labels,
                decode_v4(v3_p, sportec[T].half, fold_dec_std[T], p_change=v3_c),
                sportec[T].half,
            )["spell_f1_mean"]
            f1_v6 = metrics.score(
                sportec[T].labels,
                decode_v4(v6_p, sportec[T].half, fold_dec_std[T], p_change=v6_c),
                sportec[T].half,
            )["spell_f1_mean"]
            print(
                f"  [Fold {k} ({T})] SymBiGRU_PhysV2_TTA trained & saved in {time.time() - tf:.1f}s "
                f"| v3_SpellF1={f1_v3:.4f} | v6_SpellF1={f1_v6:.4f}",
                flush=True,
            )

    # Full 7-Match Models (`symtcn_physv2_full.pkl` & `symbigru_physv2_full.pkl`)
    print("\n--- Step 3B: Training Full 7-Match `symtcn_physv2_full.pkl` & `symbigru_physv2_full.pkl` ---", flush=True)
    tcn_full_path = out_dir / "symtcn_physv2_full.pkl"
    gru_full_path = out_dir / "symbigru_physv2_full.pkl"

    if not tcn_full_path.exists() or not gru_full_path.exists():
        Xs_all = [train_X_v3[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS] + [
            train_X_v3_mir[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS
        ]
        ys_all = [train_y_v3[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS] + [
            train_y_v3_mir[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS
        ]
        hs_all = [train_h_v3[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS] + [
            train_h_v3[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS
        ]

    if tcn_full_path.exists():
        with open(tcn_full_path, "rb") as f:
            symtcn_full = pickle.load(f)
        print("  Loaded symtcn_physv2_full.pkl from cache.", flush=True)
    else:
        tf = time.time()
        symtcn_full = TCNModel(norm="channel", epochs=18, seed=0).fit(Xs_all, ys_all, hs_all)
        with open(tcn_full_path, "wb") as f:
            pickle.dump(symtcn_full, f)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        print(f"  Trained & saved symtcn_physv2_full.pkl in {time.time() - tf:.1f}s", flush=True)

    if gru_full_path.exists():
        with open(gru_full_path, "rb") as f:
            symbigru_full = pickle.load(f)
        print("  Loaded symbigru_physv2_full.pkl from cache.", flush=True)
    else:
        tf = time.time()
        symbigru_full = BiGRUSeqModel(norm="channel", epochs=12, seed=0).fit(Xs_all, ys_all, hs_all)
        with open(gru_full_path, "wb") as f:
            pickle.dump(symbigru_full, f)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        print(f"  Trained & saved symbigru_physv2_full.pkl in {time.time() - tf:.1f}s", flush=True)

    # Free training variants from RAM before loading Metrica Sample & BAR-ATH
    train_X_v3.clear()
    train_X_v3_mir.clear()
    train_y_v3.clear()
    train_y_v3_mir.clear()
    train_h_v3.clear()
    gc.collect()

    # Load Metrica Sample & BAR-ATH (Raw & Repaired) and compute/cache J2/J3 sequence predictions
    print("\n--- Step 3C: Predicting on Metrica Sample (3 Games) & BAR-ATH (Raw & Repaired) ---", flush=True)
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)

    sample_X: dict[int, pd.DataFrame] = {}
    for gid in GAMES:
        ms = sample_matches[gid]
        sample_X[gid] = load_cached_physv2(f"sample_{gid}_raw", ms.half, lambda ms=ms: ms)
    X_atd_raw = load_cached_physv2("atd", atd.half, lambda: atd)
    X_atd_rep = load_cached_physv2("atd_rep_True_True_None", atd_rep.half, lambda: atd_rep)

    j23_cache_path = out_dir / "seq_physv2_j23_preds.pkl"
    if j23_cache_path.exists():
        with open(j23_cache_path, "rb") as f:
            seq_j23 = pickle.load(f)
        print("  Loaded seq_physv2_j23_preds.pkl from cache.", flush=True)
    else:
        tf = time.time()
        sample_seq_p2 = {}
        for gid in GAMES:
            ms = sample_matches[gid]
            X_mir_g = mirror_physv2_df(sample_X[gid], ms.half, ms.pitch)
            tp, tc = predict_seq_physv2_tta(symtcn_full, sample_X[gid], ms.half, ms.pitch, X_mir=X_mir_g)
            bp, bc = predict_seq_physv2_tta(symbigru_full, sample_X[gid], ms.half, ms.pitch, X_mir=X_mir_g)
            sample_seq_p2[gid] = {
                "tcn_physv2_sym": tp,
                "tcnc_physv2_sym": tc,
                "bigru_physv2_sym": bp,
                "bigruc_physv2_sym": bc,
            }

        X_atd_raw_mir = mirror_physv2_df(X_atd_raw, atd.half, atd.pitch)
        tp_raw, tc_raw = predict_seq_physv2_tta(symtcn_full, X_atd_raw, atd.half, atd.pitch, X_mir=X_atd_raw_mir)
        bp_raw, bc_raw = predict_seq_physv2_tta(symbigru_full, X_atd_raw, atd.half, atd.pitch, X_mir=X_atd_raw_mir)
        atd_raw_seq_p2 = {
            "tcn_physv2_sym": tp_raw,
            "tcnc_physv2_sym": tc_raw,
            "bigru_physv2_sym": bp_raw,
            "bigruc_physv2_sym": bc_raw,
        }

        X_atd_rep_mir = mirror_physv2_df(X_atd_rep, atd_rep.half, atd_rep.pitch)
        tp_rep, tc_rep = predict_seq_physv2_tta(symtcn_full, X_atd_rep, atd_rep.half, atd_rep.pitch, X_mir=X_atd_rep_mir)
        bp_rep, bc_rep = predict_seq_physv2_tta(symbigru_full, X_atd_rep, atd_rep.half, atd_rep.pitch, X_mir=X_atd_rep_mir)
        atd_rep_seq_p2 = {
            "tcn_physv2_sym": tp_rep,
            "tcnc_physv2_sym": tc_rep,
            "bigru_physv2_sym": bp_rep,
            "bigruc_physv2_sym": bc_rep,
        }

        seq_j23 = {
            "sample": sample_seq_p2,
            "atd_raw": atd_raw_seq_p2,
            "atd_rep": atd_rep_seq_p2,
        }
        with open(j23_cache_path, "wb") as f:
            pickle.dump(seq_j23, f)
        print(f"  Computed & saved seq_physv2_j23_preds.pkl in {time.time() - tf:.1f}s", flush=True)

    # =========================================================================
    # STEP 4: COMPREHENSIVE 4-JUDGE EVALUATION & ABLATION STUDY
    # =========================================================================
    print("\n" + "=" * 105, flush=True)
    print("STEP 4: COMPREHENSIVE 4-JUDGE EVALUATION (STANDALONE, MULTI-REP SEQUENCE, & MULTI-FAMILY ENSEMBLES)", flush=True)
    print("=" * 105, flush=True)

    store_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    plus_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    w3_dir = ROOT / "experiments" / "outputs" / "exp_w3_oof"
    w4_dir = ROOT / "experiments" / "outputs" / "exp_w4_oof"
    w5_tree_dir = ROOT / "experiments" / "outputs" / "exp_w5_tree_oof"

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std, fold_dec_hyb, _, _ = pickle.load(f)
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        loo_dec_lin, loo_dec_hyb, dec_v5_lin, dec_v5_hyb = pickle.load(f)
    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        plus_preds = pickle.load(f)
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f:
        w4_preds = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(h_dir / "symbigru_full.pkl", "rb") as f:
        symbigru_v1ctl_full = pickle.load(f)
    with open(w3_dir / "b4d_j23_preds.pkl", "rb") as f:
        b4d_j23 = pickle.load(f)

    # Compute SymBiGRU_TTA (244 cols) on BAR-ATH Raw so we have complete raw & repaired baselines
    b_raw_v1ctl, bc_raw_v1ctl = predict_tcn_tta(symbigru_v1ctl_full, X_atd_raw[symbigru_v1ctl_full.features_], atd.half, atd.pitch)

    has_w5_trees = all((w5_tree_dir / f"cb_xgb_{T}.npz").exists() for T in MATCH_IDS) and (
        w5_tree_dir / "eval_preds_cb_xgb.pkl"
    ).exists()
    if has_w5_trees:
        with open(w5_tree_dir / "eval_preds_cb_xgb.pkl", "rb") as f:
            w5_tree_j23 = pickle.load(f)
        print("  Detected completed Wave 5A (`exp_w5_tree_oof`) caches! Including 5-family PhysV2 blends.", flush=True)

    oof_all = {}
    b4d_v3, b4d_v6 = {}, {}
    for T in MATCH_IDS:
        g_orig = np.load(store_v3 / T / f"gbm_outer_{T}.npy")
        tr_orig = np.load(store_v3 / T / f"trans_outer_{T}.npy")
        z_orig = np.load(store_v3 / T / f"tcn_outer_{T}.npz")
        sb = np.load(sym_dir / f"sym_{T}.npz")
        st = np.load(sym_dir / f"tcn_sym_{T}.npz")
        sp = np.load(plus_dir / f"plus_{T}.npz")
        sg = np.load(g_dir / f"ghost_sym_{T}.npz")
        sh = np.load(h_dir / f"bigru_sym_{T}.npz")
        sw3 = np.load(w3_dir / f"b4d_{T}.npz")
        sw4 = np.load(w4_dir / f"physv2_{T}.npz")
        b4d_v3[T] = sw3["p4d_v3"]
        b4d_v6[T] = sw3["p4d_v6"]

        d_fold = {
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
            "g_v3_on_v6": sg["g_v3_on_v6"],
            "tr_v3_on_v6": sg["tr_v3_on_v6"],
            "g_v6_on_v6": sg["g_v6_on_v6"],
            "tr_v6_on_v6": sg["tr_v6_on_v6"],
            "g_p2": sw4["symgbm_physv2_tta"],
            "tr_p2": sw4["symtrans_physv2_tta"],
            "g_p2_on_v6": sw4["symgbm_physv2_tta_on_v6"],
            "tr_p2_on_v6": sw4["symtrans_physv2_tta_on_v6"],
            "t_p2": oof_tcn[T]["v3_proba"],
            "tc_p2": oof_tcn[T]["v3_change"],
            "t_p2_on_v6": oof_tcn[T]["v6_proba"],
            "tc_p2_on_v6": oof_tcn[T]["v6_change"],
            "b_p2": oof_bigru[T]["v3_proba"],
            "bc_p2": oof_bigru[T]["v3_change"],
            "b_p2_on_v6": oof_bigru[T]["v6_proba"],
            "bc_p2_on_v6": oof_bigru[T]["v6_change"],
        }
        if has_w5_trees:
            sw5t = np.load(w5_tree_dir / f"cb_xgb_{T}.npz")
            d_fold["cb_p2"] = sw5t["symcatboost_physv2_tta"]
            d_fold["xgb_p2"] = sw5t["symxgb_physv2_tta"]
            d_fold["cb_p2_on_v6"] = sw5t["symcatboost_physv2_tta_on_v6"]
            d_fold["xgb_p2_on_v6"] = sw5t["symxgb_physv2_tta_on_v6"]
        oof_all[T] = d_fold

    # -------------------------------------------------------------------------
    # PART 4A: FRAME ACCURACY, LOG-LOSS, & BOUNDARY CHANGE-HEAD PR-AUC/ROC-AUC
    # -------------------------------------------------------------------------
    print("\n--- Part 4A: Sequence Model Frame Quality & Boundary Change-Head Discrimination ---", flush=True)
    seq_diag_configs = [
        ("0. orig_TCN (v1ctl, 244 cols, no sym)",
         lambda d: (d["t_orig"], d["tc_orig"]),
         lambda d: (d["t_orig"], d["tc_orig"]),
         lambda g: (sample_h[g]["t_orig"], sample_h[g]["tc_orig"])),
        ("1. SymTCN_TTA (v1ctl, 244 cols)",
         lambda d: (d["t_sym"], d["tc_sym"]),
         lambda d: (d["t_sym"], d["tc_sym"]),
         lambda g: (plus_preds["sample"][g]["t_sym"], plus_preds["sample"][g]["tc_sym"])),
        ("2. SymTCN_PhysV2_TTA (v1ctl_physics_v2, 481 cols)",
         lambda d: (d["t_p2"], d["tc_p2"]),
         lambda d: (d["t_p2_on_v6"], d["tc_p2_on_v6"]),
         lambda g: (seq_j23["sample"][g]["tcn_physv2_sym"], seq_j23["sample"][g]["tcnc_physv2_sym"])),
        ("3. SymBiGRU_TTA (v1ctl, 244 cols)",
         lambda d: (d["b_sym"], d["bc_sym"]),
         lambda d: (d["b_sym"], d["bc_sym"]),
         lambda g: (sample_h[g]["b_sym"], sample_h[g]["bc_sym"])),
        ("4. SymBiGRU_PhysV2_TTA (v1ctl_physics_v2, 481 cols)",
         lambda d: (d["b_p2"], d["bc_p2"]),
         lambda d: (d["b_p2_on_v6"], d["bc_p2_on_v6"]),
         lambda g: (seq_j23["sample"][g]["bigru_physv2_sym"], seq_j23["sample"][g]["bigruc_physv2_sym"])),
        ("5. MultiRep SymTCN (0.5 v1ctl_244 + 0.5 physv2_481)",
         lambda d: (0.5 * d["t_sym"] + 0.5 * d["t_p2"], 0.5 * d["tc_sym"] + 0.5 * d["tc_p2"]),
         lambda d: (0.5 * d["t_sym"] + 0.5 * d["t_p2_on_v6"], 0.5 * d["tc_sym"] + 0.5 * d["tc_p2_on_v6"]),
         lambda g: (
             0.5 * plus_preds["sample"][g]["t_sym"] + 0.5 * seq_j23["sample"][g]["tcn_physv2_sym"],
             0.5 * plus_preds["sample"][g]["tc_sym"] + 0.5 * seq_j23["sample"][g]["tcnc_physv2_sym"],
         )),
        ("6. MultiRep SymBiGRU (0.5 v1ctl_244 + 0.5 physv2_481)",
         lambda d: (0.5 * d["b_sym"] + 0.5 * d["b_p2"], 0.5 * d["bc_sym"] + 0.5 * d["bc_p2"]),
         lambda d: (0.5 * d["b_sym"] + 0.5 * d["b_p2_on_v6"], 0.5 * d["bc_sym"] + 0.5 * d["bc_p2_on_v6"]),
         lambda g: (
             0.5 * sample_h[g]["b_sym"] + 0.5 * seq_j23["sample"][g]["bigru_physv2_sym"],
             0.5 * sample_h[g]["bc_sym"] + 0.5 * seq_j23["sample"][g]["bigruc_physv2_sym"],
         )),
        ("7. PhysV2 TCN+BiGRU (Decoupled w_gru_ha=0.30, w_c=0.30)",
         lambda d: combine_seq_tcn_bigru(d["t_p2"], d["tc_p2"], d["b_p2"], d["bc_p2"], w_gru_ha=0.30, w_gru_change=0.30),
         lambda d: combine_seq_tcn_bigru(d["t_p2_on_v6"], d["tc_p2_on_v6"], d["b_p2_on_v6"], d["bc_p2_on_v6"], w_gru_ha=0.30, w_gru_change=0.30),
         lambda g: combine_seq_tcn_bigru(
             seq_j23["sample"][g]["tcn_physv2_sym"], seq_j23["sample"][g]["tcnc_physv2_sym"],
             seq_j23["sample"][g]["bigru_physv2_sym"], seq_j23["sample"][g]["bigruc_physv2_sym"],
             w_gru_ha=0.30, w_gru_change=0.30,
         )),
        ("8. 4-Seq MultiRep (TCN_244+TCN_481 + BiGRU_244+BiGRU_481)",
         lambda d: combine_seq_tcn_bigru(
             0.5 * d["t_sym"] + 0.5 * d["t_p2"], 0.5 * d["tc_sym"] + 0.5 * d["tc_p2"],
             0.5 * d["b_sym"] + 0.5 * d["b_p2"], 0.5 * d["bc_sym"] + 0.5 * d["bc_p2"],
             w_gru_ha=0.25, w_gru_change=0.20,
         ),
         lambda d: combine_seq_tcn_bigru(
             0.5 * d["t_sym"] + 0.5 * d["t_p2_on_v6"], 0.5 * d["tc_sym"] + 0.5 * d["tc_p2_on_v6"],
             0.5 * d["b_sym"] + 0.5 * d["b_p2_on_v6"], 0.5 * d["bc_sym"] + 0.5 * d["bc_p2_on_v6"],
             w_gru_ha=0.25, w_gru_change=0.20,
         ),
         lambda g: combine_seq_tcn_bigru(
             0.5 * plus_preds["sample"][g]["t_sym"] + 0.5 * seq_j23["sample"][g]["tcn_physv2_sym"],
             0.5 * plus_preds["sample"][g]["tc_sym"] + 0.5 * seq_j23["sample"][g]["tcnc_physv2_sym"],
             0.5 * sample_h[g]["b_sym"] + 0.5 * seq_j23["sample"][g]["bigru_physv2_sym"],
             0.5 * sample_h[g]["bc_sym"] + 0.5 * seq_j23["sample"][g]["bigruc_physv2_sym"],
             w_gru_ha=0.25, w_gru_change=0.20,
         )),
    ]

    for sname, fn_v3, fn_v6, fn_sm in seq_diag_configs:
        acc_v3, ll_v3, pr_v3, acc_v6, pr_v6 = [], [], [], [], []
        acc_sm, pr_sm = [], []
        for T in MATCH_IDS:
            m = sportec[T]
            valid = (m.labels >= 0) & (m.half > 0)
            y4 = build_4dir_targets(m.labels, m.half)
            y_turn = np.maximum(y4[:, 0], y4[:, 1])

            p3, c3 = fn_v3(oof_all[T])
            acc_v3.append(float((np.argmax(p3[valid], axis=1) == m.labels[valid]).mean()))
            ll_v3.append(float(log_loss(m.labels[valid], np.clip(p3[valid], 1e-6, 1 - 1e-6), labels=[0, 1, 2])))
            pr_v3.append(float(average_precision_score(y_turn[valid], c3[valid])))

            p6, c6 = fn_v6(oof_all[T])
            acc_v6.append(float((np.argmax(p6[valid], axis=1) == m.labels[valid]).mean()))
            pr_v6.append(float(average_precision_score(y_turn[valid], c6[valid])))

        for gid in GAMES:
            ms = sample_matches[gid]
            valids = (ms.labels >= 0) & (ms.half > 0)
            y4s = build_4dir_targets(ms.labels, ms.half)
            y_turns = np.maximum(y4s[:, 0], y4s[:, 1])
            ps, cs = fn_sm(gid)
            acc_sm.append(float((np.argmax(ps[valids], axis=1) == ms.labels[valids]).mean()))
            pr_sm.append(float(average_precision_score(y_turns[valids], cs[valids])))

        print(
            f"  {sname:58s} | v3 Acc={np.mean(acc_v3):.4f} (LL={np.mean(ll_v3):.4f}, chgPR={np.mean(pr_v3):.4f}) "
            f"| v6 Acc={np.mean(acc_v6):.4f} (chgPR={np.mean(pr_v6):.4f}) "
            f"| Sample Acc={np.mean(acc_sm):.4f} (chgPR={np.mean(pr_sm):.4f})",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # PART 4B: STANDALONE & MULTI-REP SEQUENCE MODELS ACROSS ALL 4 JUDGES
    # -------------------------------------------------------------------------
    print("\n--- Part 4B: Standalone & Multi-Representation Sequence Models Across All 4 Judges ---", flush=True)
    dec_ens_glob = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})
    dec_tcn_glob = dec_ens_glob

    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)

    seq_only_rows = [
        ("S0. orig_TCN (244 cols, no sym)",
         lambda d: (d["t_orig"], d["tc_orig"]),
         lambda d: (d["t_orig"], d["tc_orig"]),
         lambda g: (sample_h[g]["t_orig"], sample_h[g]["tc_orig"]),
         (plus_preds["atd_raw"]["t_sym"], plus_preds["atd_raw"]["tc_sym"]),
         (atd_rep_h["t_orig"], atd_rep_h["tc_orig"])),
        ("S1. SymTCN_TTA (v1ctl, 244 cols)",
         lambda d: (d["t_sym"], d["tc_sym"]),
         lambda d: (d["t_sym"], d["tc_sym"]),
         lambda g: (plus_preds["sample"][g]["t_sym"], plus_preds["sample"][g]["tc_sym"]),
         (plus_preds["atd_raw"]["t_sym"], plus_preds["atd_raw"]["tc_sym"]),
         (atd_rep_h["t_sym"], atd_rep_h["tc_sym"])),
        ("S2. SymTCN_PhysV2_TTA (v1ctl_physics_v2, 481 cols)",
         lambda d: (d["t_p2"], d["tc_p2"]),
         lambda d: (d["t_p2_on_v6"], d["tc_p2_on_v6"]),
         lambda g: (seq_j23["sample"][g]["tcn_physv2_sym"], seq_j23["sample"][g]["tcnc_physv2_sym"]),
         (seq_j23["atd_raw"]["tcn_physv2_sym"], seq_j23["atd_raw"]["tcnc_physv2_sym"]),
         (seq_j23["atd_rep"]["tcn_physv2_sym"], seq_j23["atd_rep"]["tcnc_physv2_sym"])),
        ("S3. SymBiGRU_TTA (v1ctl, 244 cols)",
         lambda d: (d["b_sym"], d["bc_sym"]),
         lambda d: (d["b_sym"], d["bc_sym"]),
         lambda g: (sample_h[g]["b_sym"], sample_h[g]["bc_sym"]),
         (b_raw_v1ctl, bc_raw_v1ctl),
         (atd_rep_h["b_sym"], atd_rep_h["bc_sym"])),
        ("S4. SymBiGRU_PhysV2_TTA (v1ctl_physics_v2, 481 cols)",
         lambda d: (d["b_p2"], d["bc_p2"]),
         lambda d: (d["b_p2_on_v6"], d["bc_p2_on_v6"]),
         lambda g: (seq_j23["sample"][g]["bigru_physv2_sym"], seq_j23["sample"][g]["bigruc_physv2_sym"]),
         (seq_j23["atd_raw"]["bigru_physv2_sym"], seq_j23["atd_raw"]["bigruc_physv2_sym"]),
         (seq_j23["atd_rep"]["bigru_physv2_sym"], seq_j23["atd_rep"]["bigruc_physv2_sym"])),
        ("S5. MultiRep SymTCN (0.5 v1ctl_244 + 0.5 physv2_481)",
         lambda d: (0.5 * d["t_sym"] + 0.5 * d["t_p2"], 0.5 * d["tc_sym"] + 0.5 * d["tc_p2"]),
         lambda d: (0.5 * d["t_sym"] + 0.5 * d["t_p2_on_v6"], 0.5 * d["tc_sym"] + 0.5 * d["tc_p2_on_v6"]),
         lambda g: (
             0.5 * plus_preds["sample"][g]["t_sym"] + 0.5 * seq_j23["sample"][g]["tcn_physv2_sym"],
             0.5 * plus_preds["sample"][g]["tc_sym"] + 0.5 * seq_j23["sample"][g]["tcnc_physv2_sym"],
         ),
         (
             0.5 * plus_preds["atd_raw"]["t_sym"] + 0.5 * seq_j23["atd_raw"]["tcn_physv2_sym"],
             0.5 * plus_preds["atd_raw"]["tc_sym"] + 0.5 * seq_j23["atd_raw"]["tcnc_physv2_sym"],
         ),
         (
             0.5 * atd_rep_h["t_sym"] + 0.5 * seq_j23["atd_rep"]["tcn_physv2_sym"],
             0.5 * atd_rep_h["tc_sym"] + 0.5 * seq_j23["atd_rep"]["tcnc_physv2_sym"],
         )),
        ("S6. MultiRep SymBiGRU (0.5 v1ctl_244 + 0.5 physv2_481)",
         lambda d: (0.5 * d["b_sym"] + 0.5 * d["b_p2"], 0.5 * d["bc_sym"] + 0.5 * d["bc_p2"]),
         lambda d: (0.5 * d["b_sym"] + 0.5 * d["b_p2_on_v6"], 0.5 * d["bc_sym"] + 0.5 * d["bc_p2_on_v6"]),
         lambda g: (
             0.5 * sample_h[g]["b_sym"] + 0.5 * seq_j23["sample"][g]["bigru_physv2_sym"],
             0.5 * sample_h[g]["bc_sym"] + 0.5 * seq_j23["sample"][g]["bigruc_physv2_sym"],
         ),
         (
             0.5 * b_raw_v1ctl + 0.5 * seq_j23["atd_raw"]["bigru_physv2_sym"],
             0.5 * bc_raw_v1ctl + 0.5 * seq_j23["atd_raw"]["bigruc_physv2_sym"],
         ),
         (
             0.5 * atd_rep_h["b_sym"] + 0.5 * seq_j23["atd_rep"]["bigru_physv2_sym"],
             0.5 * atd_rep_h["bc_sym"] + 0.5 * seq_j23["atd_rep"]["bigruc_physv2_sym"],
         )),
        ("S7. PhysV2 Seq Blend (50% SymTCN_PhysV2 + 50% SymBiGRU_PhysV2)",
         lambda d: (0.5 * d["t_p2"] + 0.5 * d["b_p2"], 0.5 * d["tc_p2"] + 0.5 * d["bc_p2"]),
         lambda d: (0.5 * d["t_p2_on_v6"] + 0.5 * d["b_p2_on_v6"], 0.5 * d["tc_p2_on_v6"] + 0.5 * d["bc_p2_on_v6"]),
         lambda g: (
             0.5 * seq_j23["sample"][g]["tcn_physv2_sym"] + 0.5 * seq_j23["sample"][g]["bigru_physv2_sym"],
             0.5 * seq_j23["sample"][g]["tcnc_physv2_sym"] + 0.5 * seq_j23["sample"][g]["bigruc_physv2_sym"],
         ),
         (
             0.5 * seq_j23["atd_raw"]["tcn_physv2_sym"] + 0.5 * seq_j23["atd_raw"]["bigru_physv2_sym"],
             0.5 * seq_j23["atd_raw"]["tcnc_physv2_sym"] + 0.5 * seq_j23["atd_raw"]["bigruc_physv2_sym"],
         ),
         (
             0.5 * seq_j23["atd_rep"]["tcn_physv2_sym"] + 0.5 * seq_j23["atd_rep"]["bigru_physv2_sym"],
             0.5 * seq_j23["atd_rep"]["tcnc_physv2_sym"] + 0.5 * seq_j23["atd_rep"]["bigruc_physv2_sym"],
         )),
        ("S8. PhysV2 Seq Decoupled (TCN P_alive + 0.30 BiGRU P_H|alive)",
         lambda d: combine_seq_tcn_bigru(d["t_p2"], d["tc_p2"], d["b_p2"], d["bc_p2"], w_gru_ha=0.30, w_gru_change=0.25),
         lambda d: combine_seq_tcn_bigru(d["t_p2_on_v6"], d["tc_p2_on_v6"], d["b_p2_on_v6"], d["bc_p2_on_v6"], w_gru_ha=0.30, w_gru_change=0.25),
         lambda g: combine_seq_tcn_bigru(
             seq_j23["sample"][g]["tcn_physv2_sym"], seq_j23["sample"][g]["tcnc_physv2_sym"],
             seq_j23["sample"][g]["bigru_physv2_sym"], seq_j23["sample"][g]["bigruc_physv2_sym"],
             w_gru_ha=0.30, w_gru_change=0.25,
         ),
         combine_seq_tcn_bigru(
             seq_j23["atd_raw"]["tcn_physv2_sym"], seq_j23["atd_raw"]["tcnc_physv2_sym"],
             seq_j23["atd_raw"]["bigru_physv2_sym"], seq_j23["atd_raw"]["bigruc_physv2_sym"],
             w_gru_ha=0.30, w_gru_change=0.25,
         ),
         combine_seq_tcn_bigru(
             seq_j23["atd_rep"]["tcn_physv2_sym"], seq_j23["atd_rep"]["tcnc_physv2_sym"],
             seq_j23["atd_rep"]["bigru_physv2_sym"], seq_j23["atd_rep"]["bigruc_physv2_sym"],
             w_gru_ha=0.30, w_gru_change=0.25,
         )),
        ("S9. 4-Seq MultiRep (TCN_244+481 + BiGRU_244+481, w_gru=0.25)",
         lambda d: combine_seq_tcn_bigru(
             0.5 * d["t_sym"] + 0.5 * d["t_p2"], 0.5 * d["tc_sym"] + 0.5 * d["tc_p2"],
             0.5 * d["b_sym"] + 0.5 * d["b_p2"], 0.5 * d["bc_sym"] + 0.5 * d["bc_p2"],
             w_gru_ha=0.25, w_gru_change=0.20,
         ),
         lambda d: combine_seq_tcn_bigru(
             0.5 * d["t_sym"] + 0.5 * d["t_p2_on_v6"], 0.5 * d["tc_sym"] + 0.5 * d["tc_p2_on_v6"],
             0.5 * d["b_sym"] + 0.5 * d["b_p2_on_v6"], 0.5 * d["bc_sym"] + 0.5 * d["bc_p2_on_v6"],
             w_gru_ha=0.25, w_gru_change=0.20,
         ),
         lambda g: combine_seq_tcn_bigru(
             0.5 * plus_preds["sample"][g]["t_sym"] + 0.5 * seq_j23["sample"][g]["tcn_physv2_sym"],
             0.5 * plus_preds["sample"][g]["tc_sym"] + 0.5 * seq_j23["sample"][g]["tcnc_physv2_sym"],
             0.5 * sample_h[g]["b_sym"] + 0.5 * seq_j23["sample"][g]["bigru_physv2_sym"],
             0.5 * sample_h[g]["bc_sym"] + 0.5 * seq_j23["sample"][g]["bigruc_physv2_sym"],
             w_gru_ha=0.25, w_gru_change=0.20,
         ),
         combine_seq_tcn_bigru(
             0.5 * plus_preds["atd_raw"]["t_sym"] + 0.5 * seq_j23["atd_raw"]["tcn_physv2_sym"],
             0.5 * plus_preds["atd_raw"]["tc_sym"] + 0.5 * seq_j23["atd_raw"]["tcnc_physv2_sym"],
             0.5 * b_raw_v1ctl + 0.5 * seq_j23["atd_raw"]["bigru_physv2_sym"],
             0.5 * bc_raw_v1ctl + 0.5 * seq_j23["atd_raw"]["bigruc_physv2_sym"],
             w_gru_ha=0.25, w_gru_change=0.20,
         ),
         combine_seq_tcn_bigru(
             0.5 * atd_rep_h["t_sym"] + 0.5 * seq_j23["atd_rep"]["tcn_physv2_sym"],
             0.5 * atd_rep_h["tc_sym"] + 0.5 * seq_j23["atd_rep"]["tcnc_physv2_sym"],
             0.5 * atd_rep_h["b_sym"] + 0.5 * seq_j23["atd_rep"]["bigru_physv2_sym"],
             0.5 * atd_rep_h["bc_sym"] + 0.5 * seq_j23["atd_rep"]["bigruc_physv2_sym"],
             w_gru_ha=0.25, w_gru_change=0.20,
         )),
    ]

    for sname, fn_v3, fn_v6, fn_sm, (pa_raw, ca_raw), (pa_rep, ca_rep) in seq_only_rows:
        s_v3, s_v6 = [], []
        for T in MATCH_IDS:
            params_tcn = fold_dec_std[T]
            p3, c3 = fn_v3(oof_all[T])
            p6, c6 = fn_v6(oof_all[T])
            pr3 = decode_v4(p3, sportec[T].half, params_tcn, p_change=c3)
            pr6 = decode_v4(p6, sportec[T].half, params_tcn, p_change=c6)
            s_v3.append(metrics.score(sportec[T].labels, pr3, sportec[T].half)["spell_f1_mean"])
            s_v6.append(metrics.score(sportec[T].labels, pr6, sportec[T].half)["spell_f1_mean"])

        s_sm_raw, s_sm_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            ps, cs = fn_sm(gid)
            prs = decode_v4(ps, ms.half, dec_tcn_glob, p_change=cs)
            s_sm_raw.append(metrics.score(ms.labels, prs, ms.half)["spell_f1_mean"])
            s_sm_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), prs, ms.half)["spell_f1_mean"])

        pred_raw = decode_v4(pa_raw, half_dec, dec_tcn_glob, p_change=ca_raw)
        pred_raw[second_start:kick2] = DEAD
        pred_raw[~atd.graded] = DEAD
        pub_raw = metrics.score_public(pred_raw, bootstrap_n=100)
        pl_raw = plausibility(pred_raw, atd.half, atd.graded)

        pred_rep = decode_v4(pa_rep, half_dec, dec_tcn_glob, p_change=ca_rep)
        pred_rep[second_start:kick2] = DEAD
        pred_rep[~atd.graded] = DEAD
        pub_rep = metrics.score_public(pred_rep, bootstrap_n=100)
        pl_rep = plausibility(pred_rep, atd.half, atd.graded)

        print(
            f"  {sname:58s}\n"
            f"    -> J1A (v3 LOMO): {np.mean(s_v3):.4f} (s={np.std(s_v3):.4f}) | J1B (v6 LOMO): {np.mean(s_v6):.4f} (s={np.std(s_v6):.4f})\n"
            f"    -> J2  (Sample) : {np.mean(s_sm_raw):.4f} (noD={np.mean(s_sm_nod):.4f}) | games={[round(x, 4) for x in s_sm_raw]}\n"
            f"    -> J3  (ATD Raw): Pub={pub_raw['spell_f1_mean']:.4f} (D={pub_raw['per_class']['D']['f1']:.3f}, {pl_raw['poss_per_min']:.2f}/m, {pl_raw['pct_D']:.1f}%D) "
            f"| ATD Rep: Pub={pub_rep['spell_f1_mean']:.4f} (D={pub_rep['per_class']['D']['f1']:.3f}, {pl_rep['poss_per_min']:.2f}/m, {pl_rep['pct_D']:.1f}%D)",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # PART 4C: FULL MULTI-FAMILY ENSEMBLES (TREES + SEQUENCE) UNDER LEGACY & WAVE 3 DECODERS
    # -------------------------------------------------------------------------
    print("\n--- Part 4C: Full Multi-Family Ensembles (Trees + Sequence) Across All 4 Judges ---", flush=True)
    xdx_cfg = XDXConfig(
        enable_absorb=True,
        max_absorb_len=240,
        interior_max_abs_x=50.0,
        interior_max_abs_y=33.5,
        tau_absorb=0.85,
        max_absorb_b4d=0.25,
        enable_split=True,
        enable_oob_trim=False,
        enable_restart_attr=True,
    )

    def decode_stage(
        stage: str,
        p: np.ndarray,
        c: np.ndarray,
        p4d: np.ndarray | None,
        half: np.ndarray,
        X: pd.DataFrame,
        dp_leg: DecoderParams,
    ) -> np.ndarray:
        if stage == "v4":
            return decode_v4(p, half, dp_leg, p_change=c)
        if stage == "w3":
            dp2 = DecoderParamsV2.from_legacy(dp_leg, pi0_ha=0.001, pi0_d=0.001, w_b4d_1d=0.25, w_dir_ha=0.35)
            pr = decode_v2(p, half, dp2, p_change=c, p_trans_4d=p4d)
            return apply_xdx_arbitrator(pr, p, half, X, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg)
        raise ValueError(stage)

    # Define multi-family ensemble builders returning (p_tree, c_tree, p_seq, c_seq) on v3, v6, sample[gid], atd_raw, atd_rep
    def get_tree_preds(tree_mode: str, d_fold=None, gid=None, is_v6=False, is_raw=False):
        if d_fold is not None:
            if not is_v6:
                if tree_mode == "gbm_p2":
                    return d_fold["g_p2"], d_fold["tr_p2"]
                if tree_mode == "gbm_plus_p2":
                    return 0.5 * d_fold["g_plus"] + 0.5 * d_fold["g_p2"], 0.5 * d_fold["tr_plus"] + 0.5 * d_fold["tr_p2"]
                if tree_mode == "3tree_p2":
                    return (d_fold["g_p2"] + d_fold["cb_p2"] + d_fold["xgb_p2"]) / 3.0, d_fold["tr_p2"]
                if tree_mode == "3tree_plus_p2":
                    t3 = (d_fold["g_p2"] + d_fold["cb_p2"] + d_fold["xgb_p2"]) / 3.0
                    return 0.35 * d_fold["g_plus"] + 0.65 * t3, 0.5 * d_fold["tr_plus"] + 0.5 * d_fold["tr_p2"]
            else:
                if tree_mode == "gbm_p2":
                    return d_fold["g_p2_on_v6"], d_fold["tr_p2_on_v6"]
                if tree_mode == "gbm_plus_p2":
                    return 0.5 * d_fold["g_v6_on_v6"] + 0.5 * d_fold["g_p2_on_v6"], 0.5 * d_fold["tr_v6_on_v6"] + 0.5 * d_fold["tr_p2_on_v6"]
                if tree_mode == "3tree_p2":
                    return (d_fold["g_p2_on_v6"] + d_fold["cb_p2_on_v6"] + d_fold["xgb_p2_on_v6"]) / 3.0, d_fold["tr_p2_on_v6"]
                if tree_mode == "3tree_plus_p2":
                    t3 = (d_fold["g_p2_on_v6"] + d_fold["cb_p2_on_v6"] + d_fold["xgb_p2_on_v6"]) / 3.0
                    return 0.35 * d_fold["g_v6_on_v6"] + 0.65 * t3, 0.5 * d_fold["tr_v6_on_v6"] + 0.5 * d_fold["tr_p2_on_v6"]
        elif gid is not None:
            dp_s = plus_preds["sample"][gid]
            dw4 = w4_preds["sample"][gid]
            if tree_mode == "gbm_p2":
                return dw4["g_physv2_sym"], dw4["tr_physv2_sym"]
            if tree_mode == "gbm_plus_p2":
                return 0.5 * dp_s["g_plus_sym"] + 0.5 * dw4["g_physv2_sym"], 0.5 * dp_s["tr_plus_sym"] + 0.5 * dw4["tr_physv2_sym"]
            if tree_mode == "3tree_p2":
                dw5 = w5_tree_j23["sample"][gid]
                return (dw4["g_physv2_sym"] + dw5["cb_physv2_sym"] + dw5["xgb_physv2_sym"]) / 3.0, dw4["tr_physv2_sym"]
            if tree_mode == "3tree_plus_p2":
                dw5 = w5_tree_j23["sample"][gid]
                t3 = (dw4["g_physv2_sym"] + dw5["cb_physv2_sym"] + dw5["xgb_physv2_sym"]) / 3.0
                return 0.35 * dp_s["g_plus_sym"] + 0.65 * t3, 0.5 * dp_s["tr_plus_sym"] + 0.5 * dw4["tr_physv2_sym"]
        else:
            key = "atd_raw" if is_raw else "atd_rep"
            dp_a = plus_preds[key]
            dw4 = w4_preds[key]
            if tree_mode == "gbm_p2":
                return dw4["g_physv2_sym"], dw4["tr_physv2_sym"]
            if tree_mode == "gbm_plus_p2":
                return 0.5 * dp_a["g_plus_sym"] + 0.5 * dw4["g_physv2_sym"], 0.5 * dp_a["tr_plus_sym"] + 0.5 * dw4["tr_physv2_sym"]
            if tree_mode == "3tree_p2":
                dw5 = w5_tree_j23[key]
                return (dw4["g_physv2_sym"] + dw5["cb_physv2_sym"] + dw5["xgb_physv2_sym"]) / 3.0, dw4["tr_physv2_sym"]
            if tree_mode == "3tree_plus_p2":
                dw5 = w5_tree_j23[key]
                t3 = (dw4["g_physv2_sym"] + dw5["cb_physv2_sym"] + dw5["xgb_physv2_sym"]) / 3.0
                return 0.35 * dp_a["g_plus_sym"] + 0.65 * t3, 0.5 * dp_a["tr_plus_sym"] + 0.5 * dw4["tr_physv2_sym"]
        raise ValueError(tree_mode)

    def get_seq_preds(seq_mode: str, d_fold=None, gid=None, is_v6=False, is_raw=False):
        if d_fold is not None:
            t_p2 = d_fold["t_p2_on_v6"] if is_v6 else d_fold["t_p2"]
            tc_p2 = d_fold["tc_p2_on_v6"] if is_v6 else d_fold["tc_p2"]
            b_p2 = d_fold["b_p2_on_v6"] if is_v6 else d_fold["b_p2"]
            bc_p2 = d_fold["bc_p2_on_v6"] if is_v6 else d_fold["bc_p2"]
            t_v1, tc_v1 = d_fold["t_sym"], d_fold["tc_sym"]
            b_v1, bc_v1 = d_fold["b_sym"], d_fold["bc_sym"]
        elif gid is not None:
            t_p2 = seq_j23["sample"][gid]["tcn_physv2_sym"]
            tc_p2 = seq_j23["sample"][gid]["tcnc_physv2_sym"]
            b_p2 = seq_j23["sample"][gid]["bigru_physv2_sym"]
            bc_p2 = seq_j23["sample"][gid]["bigruc_physv2_sym"]
            t_v1 = plus_preds["sample"][gid]["t_sym"]
            tc_v1 = plus_preds["sample"][gid]["tc_sym"]
            b_v1 = sample_h[gid]["b_sym"]
            bc_v1 = sample_h[gid]["bc_sym"]
        else:
            key = "atd_raw" if is_raw else "atd_rep"
            t_p2 = seq_j23[key]["tcn_physv2_sym"]
            tc_p2 = seq_j23[key]["tcnc_physv2_sym"]
            b_p2 = seq_j23[key]["bigru_physv2_sym"]
            bc_p2 = seq_j23[key]["bigruc_physv2_sym"]
            t_v1 = plus_preds[key]["t_sym"]
            tc_v1 = plus_preds[key]["tc_sym"]
            b_v1 = b_raw_v1ctl if is_raw else atd_rep_h["b_sym"]
            bc_v1 = bc_raw_v1ctl if is_raw else atd_rep_h["bc_sym"]

        if seq_mode == "tcn_244":
            return t_v1, tc_v1
        if seq_mode == "tcn_481":
            return t_p2, tc_p2
        if seq_mode == "tcn_multirep":
            return 0.5 * t_v1 + 0.5 * t_p2, 0.5 * tc_v1 + 0.5 * tc_p2
        if seq_mode == "bigru_244":
            return b_v1, bc_v1
        if seq_mode == "bigru_481":
            return b_p2, bc_p2
        if seq_mode == "tcn_bigru_244":
            return combine_seq_tcn_bigru(t_v1, tc_v1, b_v1, bc_v1, w_gru_ha=0.20, w_gru_change=0.0)
        if seq_mode == "tcn_bigru_481_dec20":
            return combine_seq_tcn_bigru(t_p2, tc_p2, b_p2, bc_p2, w_gru_ha=0.20, w_gru_change=0.15)
        if seq_mode == "tcn_bigru_481_dec30":
            return combine_seq_tcn_bigru(t_p2, tc_p2, b_p2, bc_p2, w_gru_ha=0.30, w_gru_change=0.20)
        if seq_mode == "tcn_bigru_481_lin50":
            return 0.5 * t_p2 + 0.5 * b_p2, 0.5 * tc_p2 + 0.5 * bc_p2
        if seq_mode == "4seq_multirep":
            return combine_seq_tcn_bigru(
                0.5 * t_v1 + 0.5 * t_p2,
                0.5 * tc_v1 + 0.5 * tc_p2,
                0.5 * b_v1 + 0.5 * b_p2,
                0.5 * bc_v1 + 0.5 * bc_p2,
                w_gru_ha=0.25,
                w_gru_change=0.15,
            )
        raise ValueError(seq_mode)

    ens_configs = [
        ("E1. SymGBM_PhysV2 + SymTCN_TTA (244 cols) [hyb 0.55]", "gbm_p2", "tcn_244", "hybrid", 0.55, "v4", "v4"),
        ("E2. SymGBM_PhysV2 + SymTCN_PhysV2_TTA (481 cols) [hyb 0.55]", "gbm_p2", "tcn_481", "hybrid", 0.55, "v4", "v4"),
        ("E3. SymGBM_PhysV2 + MultiRep_TCN (244+481) [hyb 0.55]", "gbm_p2", "tcn_multirep", "hybrid", 0.55, "v4", "v4"),
        ("E4. SymGBM_PhysV2 + SymBiGRU_TTA (244 cols) [hyb 0.55]", "gbm_p2", "bigru_244", "hybrid", 0.55, "v4", "v4"),
        ("E5. SymGBM_PhysV2 + SymBiGRU_PhysV2_TTA (481 cols) [hyb 0.55]", "gbm_p2", "bigru_481", "hybrid", 0.55, "v4", "v4"),
        ("E6. SymGBM_PhysV2 + (SymTCN_244 + SymBiGRU_244) [hyb 0.55]", "gbm_p2", "tcn_bigru_244", "hybrid", 0.55, "v4", "v4"),
        ("E7. SymGBM_PhysV2 + (SymTCN_PhysV2 + SymBiGRU_PhysV2 dec20) [hyb 0.55]", "gbm_p2", "tcn_bigru_481_dec20", "hybrid", 0.55, "v4", "v4"),
        ("E8. SymGBM_PhysV2 + (SymTCN_PhysV2 + SymBiGRU_PhysV2 dec30) [hyb 0.55]", "gbm_p2", "tcn_bigru_481_dec30", "hybrid", 0.55, "v4", "v4"),
        ("E9. SymGBM_PhysV2 + (SymTCN_PhysV2 + SymBiGRU_PhysV2 lin50) [hyb 0.55]", "gbm_p2", "tcn_bigru_481_lin50", "hybrid", 0.55, "v4", "v4"),
        ("E10. SymGBM_PhysV2 + 4-Seq MultiRep (TCN_244+481 + BiGRU_244+481) [hyb 0.55]", "gbm_p2", "4seq_multirep", "hybrid", 0.55, "v4", "v4"),
        ("E11. SymGBM_(Plus+PhysV2) + (SymTCN_PhysV2 + SymBiGRU_PhysV2) [dec_v5]", "gbm_plus_p2", "tcn_bigru_481_dec20", "hybrid", 0.55, "hyb", "v4"),
        ("E12. SymGBM_(Plus+PhysV2) + 4-Seq MultiRep [dec_v5]", "gbm_plus_p2", "4seq_multirep", "hybrid", 0.55, "hyb", "v4"),
        ("W1. E7 (SymGBM_PhysV2 + Seq_PhysV2 dec20) + Wave 3 (decode_v2 + XDX)", "gbm_p2", "tcn_bigru_481_dec20", "hybrid", 0.55, "v4", "w3"),
        ("W2. E10 (SymGBM_PhysV2 + 4-Seq MultiRep) + Wave 3 (decode_v2 + XDX)", "gbm_p2", "4seq_multirep", "hybrid", 0.55, "v4", "w3"),
        ("W3. E12 (SymGBM_Plus+PhysV2 + 4-Seq MultiRep) + Wave 3 (decode_v2 + XDX)", "gbm_plus_p2", "4seq_multirep", "hybrid", 0.55, "hyb", "w3"),
    ]

    if has_w5_trees:
        ens_configs.extend([
            ("E13. 3-Tree_PhysV2 (GBM+CB+XGB) + (SymTCN_244 + SymBiGRU_244) [hyb 0.55]", "3tree_p2", "tcn_bigru_244", "hybrid", 0.55, "v4", "v4"),
            ("E14. 5-Family Pure PhysV2 (3-Tree_481 + SymTCN_481 + SymBiGRU_481) [hyb 0.55]", "3tree_p2", "tcn_bigru_481_dec20", "hybrid", 0.55, "v4", "v4"),
            ("E15. 5-Family Pure PhysV2 (3-Tree_481 + SymTCN_481 + SymBiGRU_481) [lin 0.55]", "3tree_p2", "tcn_bigru_481_dec20", "linear", 0.55, "v4", "v4"),
            ("E16. 5-Family MultiRep (3-Tree_481 + 4-Seq MultiRep) [hyb 0.55]", "3tree_p2", "4seq_multirep", "hybrid", 0.55, "v4", "v4"),
            ("E17. 5-Family (3-Tree_Plus+PhysV2 + SymTCN_481 + SymBiGRU_481) [dec_v5]", "3tree_plus_p2", "tcn_bigru_481_dec20", "hybrid", 0.55, "hyb", "v4"),
            ("E18. 5-Family (3-Tree_Plus+PhysV2 + 4-Seq MultiRep) [dec_v5]", "3tree_plus_p2", "4seq_multirep", "hybrid", 0.55, "hyb", "v4"),
            ("W4. E14 (5-Family Pure PhysV2) + Wave 3 (decode_v2 + XDX)", "3tree_p2", "tcn_bigru_481_dec20", "hybrid", 0.55, "v4", "w3"),
            ("W5. E16 (5-Family MultiRep) + Wave 3 (decode_v2 + XDX)", "3tree_p2", "4seq_multirep", "hybrid", 0.55, "v4", "w3"),
            ("W6. E18 (5-Family Plus+PhysV2 + 4-Seq MultiRep) + Wave 3 (decode_v2 + XDX)", "3tree_plus_p2", "4seq_multirep", "hybrid", 0.55, "hyb", "w3"),
        ])

    for ename, tmode, smode, bmode, wg, dmode, stage in ens_configs:
        s_v3, s_v6 = [], []
        for T in MATCH_IDS:
            d = oof_all[T]
            dp = fold_dec_std[T] if dmode == "v4" else loo_dec_hyb[T]

            pt3, ct3 = get_tree_preds(tmode, d_fold=d, is_v6=False)
            ps3, cs3 = get_seq_preds(smode, d_fold=d, is_v6=False)
            p3 = blend_proba(pt3, ps3, w_gbm=wg, mode=bmode)
            c3 = 0.5 * ct3 + 0.5 * cs3
            pr3 = decode_stage(stage, p3, c3, b4d_v3[T], sportec[T].half, test_X_v3[T], dp)
            s_v3.append(metrics.score(sportec[T].labels, pr3, sportec[T].half)["spell_f1_mean"])

            pt6, ct6 = get_tree_preds(tmode, d_fold=d, is_v6=True)
            ps6, cs6 = get_seq_preds(smode, d_fold=d, is_v6=True)
            p6 = blend_proba(pt6, ps6, w_gbm=wg, mode=bmode)
            c6 = 0.5 * ct6 + 0.5 * cs6
            pr6 = decode_stage(stage, p6, c6, b4d_v6[T], sportec[T].half, test_X_v6[T], dp)
            s_v6.append(metrics.score(sportec[T].labels, pr6, sportec[T].half)["spell_f1_mean"])

        dec_glob = dec_ens_glob if dmode == "v4" else dec_v5_hyb
        s_sm_raw, s_sm_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            pts, cts = get_tree_preds(tmode, gid=gid)
            pss, css = get_seq_preds(smode, gid=gid)
            ps = blend_proba(pts, pss, w_gbm=wg, mode=bmode)
            cs = 0.5 * cts + 0.5 * css
            prs = decode_stage(stage, ps, cs, b4d_j23["sample"][gid], ms.half, sample_X[gid], dec_glob)
            s_sm_raw.append(metrics.score(ms.labels, prs, ms.half)["spell_f1_mean"])
            s_sm_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), prs, ms.half)["spell_f1_mean"])

        # BAR-ATH Raw
        pt_raw, ct_raw = get_tree_preds(tmode, is_raw=True)
        ps_raw, cs_raw = get_seq_preds(smode, is_raw=True)
        p_raw = blend_proba(pt_raw, ps_raw, w_gbm=wg, mode=bmode)
        c_raw = 0.5 * ct_raw + 0.5 * cs_raw
        p4d_raw = b4d_j23.get("atd_raw", b4d_j23["atd_rep"])
        pred_raw = decode_stage(stage, p_raw, c_raw, p4d_raw, half_dec, X_atd_raw, dec_glob)
        pred_raw[second_start:kick2] = DEAD
        pred_raw[~atd.graded] = DEAD
        pub_raw = metrics.score_public(pred_raw, bootstrap_n=100)
        pl_raw = plausibility(pred_raw, atd.half, atd.graded)

        # BAR-ATH Repaired
        pt_rep, ct_rep = get_tree_preds(tmode, is_raw=False)
        ps_rep, cs_rep = get_seq_preds(smode, is_raw=False)
        p_rep = blend_proba(pt_rep, ps_rep, w_gbm=wg, mode=bmode)
        c_rep = 0.5 * ct_rep + 0.5 * cs_rep
        pred_rep = decode_stage(stage, p_rep, c_rep, b4d_j23["atd_rep"], half_dec, X_atd_rep, dec_glob)
        pred_rep[second_start:kick2] = DEAD
        pred_rep[~atd.graded] = DEAD
        pub_rep = metrics.score_public(pred_rep, bootstrap_n=100)
        pl_rep = plausibility(pred_rep, atd.half, atd.graded)
        _, n_int_rep = count_interior_xdx_islands(pred_rep, half_dec, X_atd_rep)

        print(
            f"  {ename:76s}\n"
            f"    -> J1A (v3 LOMO): {np.mean(s_v3):.4f} (s={np.std(s_v3):.4f}) | J1B (v6 LOMO): {np.mean(s_v6):.4f} (s={np.std(s_v6):.4f})\n"
            f"    -> J2  (Sample) : {np.mean(s_sm_raw):.4f} (noD={np.mean(s_sm_nod):.4f}) | games={[round(x, 4) for x in s_sm_raw]}\n"
            f"    -> J3  (ATD Raw): Pub={pub_raw['spell_f1_mean']:.4f} (D={pub_raw['per_class']['D']['f1']:.3f}, {pl_raw['poss_per_min']:.2f}/m, {pl_raw['pct_D']:.1f}%D) "
            f"| ATD Rep: Pub={pub_rep['spell_f1_mean']:.4f} (D={pub_rep['per_class']['D']['f1']:.3f}, {pl_rep['poss_per_min']:.2f}/m, {pl_rep['pct_D']:.1f}%D, int={n_int_rep})",
            flush=True,
        )

    print(f"\nWave 5B (EXP-07) completed in {(time.time() - t0) / 60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
