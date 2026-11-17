"""Modal Cloud Swarm Job #2: 8 Parallel 16-Core Cloud Containers (128 CPU Cores) for:
1. Line L1 + Multi-Seed (`v3 + v6`) Possession-Weighted & Transition-Dense Tree Families on 481 `physv2` features
   (trained ONLY on the 7 public Sportec matches; zero competition data is uploaded to Modal):
   - `SymGBM_PhysV2_L1` (LightGBM Two-Head + Isotonic Calibration with sample weights)
   - `SymCatBoost_PhysV2_L1` (CatBoost Oblivious Trees Two-Head with sample weights + `v6` ghost-ball training)
   - `SymXGB_PhysV2_L1` (XGBoost Hist Two-Head with sample weights + `v6` ghost-ball training)
2. Idea O1 + O3 Auxiliary 3D Aerial & Contested-Duel GBDT (`SymAerialAndDuelGBM_PhysV2`):
   - `head_aerial`: predicts `p_aerial_physv2(t) = P(ball_z > 1.5m | X_physv2(t))` supervised by ground-truth
     Sportec 3D `ball_z` (symmetric across H <-> A).
   - `head_duel_to`: predicts `p_duel_to(t) = P(live H<->A turnover within ±6f | X_physv2(t))` with extra
     weight on rapid steals and contested duels.
3. 100% Local Inference (Stage 3): Downloads trained tree checkpoints (`l1_trees_full_models.pkl`) from Modal
   and runs `BAR-ATH` (`atd_raw`, `atd_rep`, `atd_reteam_rep`) feature extraction & TTA prediction 100% locally.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

import modal
import numpy as np
import pandas as pd

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT / "py_pkgs") not in sys.path:
        sys.path.insert(0, str(ROOT / "py_pkgs"))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))

    from sklearn.metrics import roc_auc_score
    from s3_11_tcn_local import without_dead_margin
    from s4_6_broadcast_align import load_broadcast
    from src.data.common import AWAY, DEAD, HOME
    from src.data.metrica_atd import load as load_atd
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.sportec import load_all as load_sportec
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.eval import metrics
    from src.features.symmetry import swap_labels, swap_proba
    from src.features.v1ctl_physics_v2 import build_v1ctl_physics_v2, mirror_v1ctl_physv2_df
    from src.models.transition import change_features
    from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
else:
    ROOT = Path("/root/project")
    HOME, AWAY, DEAD = 0, 1, 2

APP_NAME = "bcn-v9-l1-trees-swarm"
VOLUME_NAME = "bcn-possession-v9"

SPORTEC_MATCH_IDS = (
    "J03WMX",
    "J03WN1",
    "J03WOH",
    "J03WOY",
    "J03WPY",
    "J03WQQ",
    "J03WR9",
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install(
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.1.0",
        "xgboost>=2.0.0",
        "catboost>=1.2.2",
    )
)

app = modal.App(APP_NAME, image=image)


def load_cached_physv2_table(prefix: str, half: np.ndarray) -> pd.DataFrame:
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


def load_atd_reteam():
    atd = load_atd()
    reteam_path = ROOT / "local_data" / "video" / "players_reteam.npz"
    if not reteam_path.exists():
        return atd
    z = np.load(reteam_path)
    return replace(
        atd,
        P=z["P"].astype(atd.P.dtype),
        slot_team=z["slot_team"].astype(atd.slot_team.dtype),
        slot_is_gk=z["slot_is_gk"].astype(bool),
    )


def compute_l1_keep_indices_and_weights(
    y: np.ndarray,
    half: np.ndarray,
    alpha: float = 0.38,
    ref_len: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Line L1 Transition-Dense + Short-Spell-Dense subsampling and smooth possession weights.

    Returns:
      keep_idx: 1D int array of selected frame indices (`y >= 0`)
      weights:  1D float32 array of sample weights for `keep_idx`
      y_duel_to: 1D int8 array (for all frames `T`) equal to 1 within ±6 frames of a live H<->A turnover
    """
    T = len(y)
    spell_len = np.full(T, ref_len, dtype=np.float32)
    dist_bnd = np.full(T, 999, dtype=np.int32)
    y_duel_to = np.zeros(T, dtype=np.int8)

    for hv in (1, 2):
        idx = np.flatnonzero((half == hv) & (y >= 0))
        if len(idx) == 0:
            continue
        s_h, e_h = int(idx[0]), int(idx[-1]) + 1
        yh = y[s_h:e_h]
        changes = np.flatnonzero(yh[1:] != yh[:-1]) + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, len(yh)]
        for s, e in zip(starts, ends):
            L = max(1, e - s)
            spell_len[s_h + s : s_h + e] = float(L)
        for c in changes:
            t_abs = s_h + int(c)
            a = max(s_h, t_abs - 15)
            b = min(e_h, t_abs + 16)
            d = np.abs(np.arange(a, b) - t_abs)
            dist_bnd[a:b] = np.minimum(dist_bnd[a:b], d)
            # Check if live H <-> A turnover (either direct or across tiny <=5f gap)
            prev_lbl = int(yh[c - 1])
            next_lbl = int(yh[c])
            if prev_lbl in (HOME, AWAY) and next_lbl in (HOME, AWAY) and prev_lbl != next_lbl:
                ya = max(s_h, t_abs - 6)
                yb = min(e_h, t_abs + 7)
                y_duel_to[ya:yb] = 1

    valid = np.flatnonzero(y >= 0)
    # Stratified deterministic subsampling:
    # - Short spells (<= 85f) or boundary neighborhood (<= 8f): stride 6
    # - Medium spells (85 < L <= 225f): stride 14
    # - Long steady interiors (L > 225f and dist_bnd > 8): stride 26
    is_short_or_bnd = (spell_len[valid] <= 85.0) | (dist_bnd[valid] <= 8)
    is_med = (~is_short_or_bnd) & (spell_len[valid] <= 225.0)
    is_long = (~is_short_or_bnd) & (~is_med)

    k_short = valid[is_short_or_bnd][::6]
    k_med = valid[is_med][::14]
    k_long = valid[is_long][::26]

    keep_idx = np.unique(np.concatenate([k_short, k_med, k_long]))
    # Smooth possession-inverse weight + boundary boost on kept samples
    w_raw = np.power(ref_len / np.maximum(spell_len[keep_idx], 15.0), alpha)
    w_bnd = np.where(dist_bnd[keep_idx] <= 6, 1.35, 1.0)
    weights = np.clip(w_raw * w_bnd, 0.55, 2.60).astype(np.float32)
    return keep_idx, weights, y_duel_to


def build_and_upload_l1_bundle(sportec: dict) -> Path:
    bundle_path = ROOT / "local_data" / "v9_swarm" / "l1_tree_train_bundle.npz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        print(f"  [Bundle] Using existing {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return bundle_path

    t0 = time.time()
    print("  [Bundle] Building L1 Transition-Dense + Multi-Seed (`v3 + v6`) training bundle...", flush=True)
    variant_specs = (
        ("degrade_v3_s1", 1.00),
        ("degrade_v3_s2", 1.00),
        ("degrade_v3_static05_s1", 0.95),
        ("degrade_v6_s0", 0.75),
        ("degrade_v6_s1", 0.75),
    )

    arrays = {}
    feat_names = None

    for mid in SPORTEC_MATCH_IDS:
        m = sportec[mid]
        keep_idx, w_l1, y_duel_to = compute_l1_keep_indices_and_weights(m.labels, m.half)
        bz = m.meta.get("ball_z")
        y_air_full = ((bz > 1.5) & np.isfinite(bz)).astype(np.int8) if bz is not None else np.zeros(len(m.labels), dtype=np.int8)

        X_list, y_list, w_list, y_air_list, y_duel_list, is_v6_list = [], [], [], [], [], []
        for vtag, v_w in variant_specs:
            df = load_cached_physv2_table(f"sportec_{mid}_{vtag}", m.half)
            if feat_names is None:
                feat_names = list(df.columns)
            df_sub = df.iloc[keep_idx]
            df_mir = mirror_v1ctl_physv2_df(df_sub, m.half[keep_idx], m.pitch)

            x_orig = df_sub.to_numpy(dtype=np.float32, copy=True)
            x_mir = df_mir.to_numpy(dtype=np.float32, copy=True)
            y_orig = m.labels[keep_idx].astype(np.int8)
            y_mir = swap_labels(y_orig).astype(np.int8)
            w_sub = (w_l1 * float(v_w)).astype(np.float32)
            ya_sub = y_air_full[keep_idx].astype(np.int8)
            yd_sub = y_duel_to[keep_idx].astype(np.int8)
            v6_flag = np.full(len(keep_idx), 1 if "v6" in vtag else 0, dtype=np.int8)

            X_list.extend([x_orig, x_mir])
            y_list.extend([y_orig, y_mir])
            w_list.extend([w_sub, w_sub])
            y_air_list.extend([ya_sub, ya_sub])
            y_duel_list.extend([yd_sub, yd_sub])
            is_v6_list.extend([v6_flag, v6_flag])

        arrays[f"{mid}_X"] = np.vstack(X_list)
        arrays[f"{mid}_y"] = np.concatenate(y_list)
        arrays[f"{mid}_w"] = np.concatenate(w_list)
        arrays[f"{mid}_y_air"] = np.concatenate(y_air_list)
        arrays[f"{mid}_y_duel"] = np.concatenate(y_duel_list)
        arrays[f"{mid}_is_v6"] = np.concatenate(is_v6_list)
        print(f"    [{mid}] packed {len(arrays[f'{mid}_y']):,} symmetric L1 rows", flush=True)

    arrays["feature_names"] = np.array(feat_names, dtype=object)
    np.savez_compressed(bundle_path, **arrays)
    print(
        f"  [Bundle] Saved {bundle_path.name} ({bundle_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s",
        flush=True,
    )

    t_up = time.time()
    subprocess.run(
        [sys.executable, "-m", "modal", "volume", "put", VOLUME_NAME, str(bundle_path), "/l1_tree_train_bundle.npz", "--force"],
        check=True,
    )
    print(f"  [Bundle] Uploaded to Modal Volume `{VOLUME_NAME}` in {time.time() - t_up:.1f}s", flush=True)
    return bundle_path


@app.function(
    cpu=16.0,
    memory=32768,
    timeout=900,
    retries=0,
    volumes={"/vol": vol},
)
def train_l1_trees_modal_worker(worker_idx: int) -> dict:
    """Trains L1 Multi-Seed `SymGBM`, `SymCatBoost`, `SymXGB`, and `SymAerialAndDuelGBM` on 16 cloud CPU cores."""
    t0 = time.time()
    import catboost as cb
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    import xgboost as xgb

    vol.reload()
    z = np.load("/vol/l1_tree_train_bundle.npz", allow_pickle=True)

    if worker_idx < 7:
        held_out_mid = SPORTEC_MATCH_IDS[worker_idx]
        train_mids = [m for m in SPORTEC_MATCH_IDS if m != held_out_mid]
        seed = 100 + worker_idx
    else:
        held_out_mid = None
        train_mids = list(SPORTEC_MATCH_IDS)
        seed = 100

    Xs, ys, ws, y_airs, y_duels, gs = [], [], [], [], [], []
    for g_idx, mid in enumerate(train_mids):
        Xm = z[f"{mid}_X"].astype(np.float32)
        ym = z[f"{mid}_y"].astype(np.int8)
        wm = z[f"{mid}_w"].astype(np.float32)
        yam = z[f"{mid}_y_air"].astype(np.int8)
        ydm = z[f"{mid}_y_duel"].astype(np.int8)
        Xs.append(Xm)
        ys.append(ym)
        ws.append(wm)
        y_airs.append(yam)
        y_duels.append(ydm)
        gs.append(np.full(len(ym), g_idx, dtype=np.int16))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    y_air = np.concatenate(y_airs)
    y_duel = np.concatenate(y_duels)
    g = np.concatenate(gs)
    del Xs, ys, ws, y_airs, y_duels, gs

    y_alive = (y != 2).astype(np.int32)
    live_mask = y != 2
    X_live = X[live_mask]
    y_team = (y[live_mask] == 0).astype(np.int32)
    w_live = w[live_mask]
    g_live = g[live_mask]

    # 1. SymGBM_PhysV2_L1 (Two-Head LightGBM + 3-fold Group Isotonic Calibration)
    lgb_params = dict(
        n_estimators=480,
        learning_rate=0.045,
        num_leaves=63,
        min_child_samples=90,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.68,
        reg_lambda=1.5,
        verbose=-1,
        n_jobs=16,
        random_state=seed,
    )
    gbm_alive = lgb.LGBMClassifier(**lgb_params).fit(X, y_alive, sample_weight=w)
    gbm_team = lgb.LGBMClassifier(**lgb_params).fit(X_live, y_team, sample_weight=w_live)

    ug = np.unique(g)
    folds = np.array_split(np.random.default_rng(seed).permutation(ug), 3)
    oof_a = np.zeros(len(y_alive), dtype=np.float32)
    oof_t = np.zeros(len(y_team), dtype=np.float32)
    for fold in folds:
        te_a = np.isin(g, fold)
        ma = lgb.LGBMClassifier(**lgb_params).fit(X[~te_a], y_alive[~te_a], sample_weight=w[~te_a])
        oof_a[te_a] = ma.predict_proba(X[te_a])[:, 1]

        te_t = np.isin(g_live, fold)
        mt = lgb.LGBMClassifier(**lgb_params).fit(X_live[~te_t], y_team[~te_t], sample_weight=w_live[~te_t])
        oof_t[te_t] = mt.predict_proba(X_live[te_t])[:, 1]

    iso_gbm_a = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1.0 - 1e-4).fit(oof_a, y_alive, sample_weight=w)
    iso_gbm_t = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1.0 - 1e-4).fit(oof_t, y_team, sample_weight=w_live)

    # 2. SymCatBoost_PhysV2_L1 (Oblivious Trees, subsample [::2] of L1 rows for fast fit)
    cb_step = 2
    cb_params = dict(
        depth=6,
        iterations=950,
        learning_rate=0.045,
        l2_leaf_reg=5.0,
        rsm=0.65,
        border_count=128,
        thread_count=16,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    cb_alive = cb.CatBoostClassifier(**cb_params).fit(X[::cb_step], y_alive[::cb_step], sample_weight=w[::cb_step])
    cb_team = cb.CatBoostClassifier(**cb_params).fit(X_live[::cb_step], y_team[::cb_step], sample_weight=w_live[::cb_step])

    # 3. SymXGB_PhysV2_L1 (Depth-wise Histogram Trees)
    xgb_params = dict(
        max_depth=6,
        n_estimators=600,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.6,
        colsample_bylevel=0.7,
        reg_lambda=3.0,
        reg_alpha=0.5,
        min_child_weight=40,
        max_bin=128,
        tree_method="hist",
        nthread=16,
        random_state=seed,
        verbosity=0,
    )
    xgb_alive = xgb.XGBClassifier(**xgb_params).fit(X[::cb_step], y_alive[::cb_step], sample_weight=w[::cb_step])
    xgb_team = xgb.XGBClassifier(**xgb_params).fit(X_live[::cb_step], y_team[::cb_step], sample_weight=w_live[::cb_step])

    # 4. SymAerialAndDuelGBM_PhysV2 (3D ball_z > 1.5m & live duel turnover detector)
    air_params = dict(
        n_estimators=350,
        learning_rate=0.045,
        num_leaves=47,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=2.0,
        verbose=-1,
        n_jobs=16,
        random_state=seed,
    )
    gbm_aerial = lgb.LGBMClassifier(**air_params).fit(X, y_air.astype(np.int32))
    gbm_duel = lgb.LGBMClassifier(**air_params).fit(X_live, y_duel[live_mask].astype(np.int32))

    payload = pickle.dumps(
        {
            "gbm": (gbm_alive, gbm_team, iso_gbm_a, iso_gbm_t),
            "cb": (cb_alive, cb_team),
            "xgb": (xgb_alive, xgb_team),
            "aerial": gbm_aerial,
            "duel": gbm_duel,
        }
    )
    return {
        "worker_idx": worker_idx,
        "mid": held_out_mid,
        "n_rows": int(len(y)),
        "elapsed_s": round(time.time() - t0, 2),
        "payload": payload,
    }


def _predict_twohead_tta(pair_or_quad, X_orig: np.ndarray, X_mir: np.ndarray) -> np.ndarray:
    if len(pair_or_quad) == 4:
        m_a, m_t, iso_a, iso_t = pair_or_quad
    else:
        m_a, m_t = pair_or_quad
        iso_a = iso_t = None

    def _pred_single(X_arr: np.ndarray) -> np.ndarray:
        pa = m_a.predict_proba(X_arr)[:, 1]
        ph = m_t.predict_proba(X_arr)[:, 1]
        if iso_a is not None:
            pa = iso_a.predict(pa)
            ph = iso_t.predict(ph)
        return np.column_stack([pa * ph, pa * (1.0 - ph), 1.0 - pa]).astype(np.float32)

    p1 = _pred_single(X_orig)
    p2 = swap_proba(_pred_single(X_mir))
    out = 0.5 * (p1 + p2)
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-6)
    return out.astype(np.float32)


def _predict_sym_binary_tta(clf, X_orig: np.ndarray, X_mir: np.ndarray) -> np.ndarray:
    p1 = clf.predict_proba(X_orig)[:, 1]
    p2 = clf.predict_proba(X_mir)[:, 1]
    return (0.5 * (p1 + p2)).astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 115, flush=True)
    print("MODAL CLOUD SWARM #2: 8 x 16-CORE CONTAINERS (128 CORES) — L1 MULTI-SEED TREES + 3D AERIAL/DUEL GBDT", flush=True)
    print("=" * 115, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "v9_swarm" / "l1_trees_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    sportec = load_sportec()
    have_cached_oof = (out_dir / "l1_trees_full_models.pkl").exists() and all(
        (out_dir / f"l1_trees_{mid}.npz").exists() for mid in SPORTEC_MATCH_IDS
    )

    if have_cached_oof:
        print("\n[Stage 2] Found cached `l1_trees_full_models.pkl` and all 7 LOMO `.npz` files — skipping Modal re-run!", flush=True)
        with open(out_dir / "l1_trees_full_models.pkl", "rb") as f:
            full_models = pickle.load(f)
        lomo_models = {}
    else:
        build_and_upload_l1_bundle(sportec)

        with app.run():
            print("\n[Stage 2] Launching 8 parallel 16-core Modal containers (128 cloud CPU cores)...", flush=True)
            t_m = time.time()
            results = list(train_l1_trees_modal_worker.map(list(range(8))))
            print(f"[Stage 2 Complete] All 8 x 16-core containers finished in {time.time() - t_m:.1f}s!", flush=True)

        lomo_models = {}
        full_models = None
        for r in results:
            mods = pickle.loads(r["payload"])
            if r["mid"] is not None:
                lomo_models[r["mid"]] = mods
                print(f"  [Worker {r['worker_idx']} LOMO {r['mid']}] trained on {r['n_rows']:,} rows in {r['elapsed_s']:.1f}s", flush=True)
            else:
                full_models = mods
                print(f"  [Worker 7 FULL 7-MATCH] trained on {r['n_rows']:,} rows in {r['elapsed_s']:.1f}s", flush=True)

        with open(out_dir / "l1_trees_full_models.pkl", "wb") as f:
            pickle.dump(full_models, f)

    print("\n[Stage 3] Running Equivariant TTA Inference & Benchmarking vs Wave 4/5 across J1A, J1B, J2, J3...", flush=True)
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    w4_dir = ROOT / "experiments" / "outputs" / "exp_w4_oof"
    w5_dir = ROOT / "experiments" / "outputs" / "exp_w5_tree_oof"
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)

    old_3tree_j1a, new_3tree_j1a = [], []
    old_3tree_j1b, new_3tree_j1b = [], []
    air_auc_v3_list, air_auc_v6_list = [], []
    duel_auc_v3_list = []

    for mid in SPORTEC_MATCH_IDS:
        m = sportec[mid]
        npz_p = out_dir / f"l1_trees_{mid}.npz"
        if npz_p.exists() and mid not in lomo_models:
            zl = np.load(npz_p)
            gbm_v3 = zl["gbm_v3"]
            cb_v3 = zl["cb_v3"]
            xgb_v3 = zl["xgb_v3"]
            air_v3 = zl["air_v3"]
            duel_v3 = zl["duel_v3"]
            gbm_v6 = zl["gbm_v6"]
            cb_v6 = zl["cb_v6"]
            xgb_v6 = zl["xgb_v6"]
            air_v6 = zl["air_v6"]
            duel_v6 = zl["duel_v6"]
        else:
            mods = lomo_models[mid]

            df_v3 = load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", m.half)
            df_v3_mir = mirror_v1ctl_physv2_df(df_v3, m.half, m.pitch)
            X3 = df_v3.to_numpy(dtype=np.float32, copy=False)
            X3_m = df_v3_mir.to_numpy(dtype=np.float32, copy=False)

            df_v6 = load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", m.half)
            df_v6_mir = mirror_v1ctl_physv2_df(df_v6, m.half, m.pitch)
            X6 = df_v6.to_numpy(dtype=np.float32, copy=False)
            X6_m = df_v6_mir.to_numpy(dtype=np.float32, copy=False)

            gbm_v3 = _predict_twohead_tta(mods["gbm"], X3, X3_m)
            cb_v3 = _predict_twohead_tta(mods["cb"], X3, X3_m)
            xgb_v3 = _predict_twohead_tta(mods["xgb"], X3, X3_m)
            air_v3 = _predict_sym_binary_tta(mods["aerial"], X3, X3_m)
            duel_v3 = _predict_sym_binary_tta(mods["duel"], X3, X3_m)

            gbm_v6 = _predict_twohead_tta(mods["gbm"], X6, X6_m)
            cb_v6 = _predict_twohead_tta(mods["cb"], X6, X6_m)
            xgb_v6 = _predict_twohead_tta(mods["xgb"], X6, X6_m)
            air_v6 = _predict_sym_binary_tta(mods["aerial"], X6, X6_m)
            duel_v6 = _predict_sym_binary_tta(mods["duel"], X6, X6_m)

            np.savez_compressed(
                npz_p,
                gbm_v3=gbm_v3,
                cb_v3=cb_v3,
                xgb_v3=xgb_v3,
                air_v3=air_v3,
                duel_v3=duel_v3,
                gbm_v6=gbm_v6,
                cb_v6=cb_v6,
                xgb_v6=xgb_v6,
                air_v6=air_v6,
                duel_v6=duel_v6,
            )

        zw4 = np.load(w4_dir / f"physv2_{mid}.npz")
        zw5 = np.load(w5_dir / f"cb_xgb_{mid}.npz")
        old_mix_v3 = 0.45 * zw4["symgbm_physv2_tta"] + 0.30 * zw5["symcatboost_physv2_tta"] + 0.25 * zw5["symxgb_physv2_tta"]
        new_mix_v3 = 0.45 * gbm_v3 + 0.30 * cb_v3 + 0.25 * xgb_v3
        old_mix_v6 = 0.45 * zw4["symgbm_physv2_tta_on_v6"] + 0.30 * zw5["symcatboost_physv2_tta_on_v6"] + 0.25 * zw5["symxgb_physv2_tta_on_v6"]
        new_mix_v6 = 0.45 * gbm_v6 + 0.30 * cb_v6 + 0.25 * xgb_v6

        c_v3 = zw4["symtrans_physv2_tta"]
        c_v6 = zw4["symtrans_physv2_tta_on_v6"]

        f1_old_3 = float(metrics.score(m.labels, decode_v4(old_mix_v3, m.half, loo_dec_hyb[mid], p_change=c_v3), m.half)["spell_f1_mean"])
        f1_new_3 = float(metrics.score(m.labels, decode_v4(new_mix_v3, m.half, loo_dec_hyb[mid], p_change=c_v3), m.half)["spell_f1_mean"])
        f1_old_6 = float(metrics.score(m.labels, decode_v4(old_mix_v6, m.half, loo_dec_hyb[mid], p_change=c_v6), m.half)["spell_f1_mean"])
        f1_new_6 = float(metrics.score(m.labels, decode_v4(new_mix_v6, m.half, loo_dec_hyb[mid], p_change=c_v6), m.half)["spell_f1_mean"])

        old_3tree_j1a.append(f1_old_3)
        new_3tree_j1a.append(f1_new_3)
        old_3tree_j1b.append(f1_old_6)
        new_3tree_j1b.append(f1_new_6)

        bz = m.meta["ball_z"]
        vbz = (m.half > 0) & np.isfinite(bz)
        yt_air = (bz[vbz] > 1.5).astype(np.int32)
        a_auc3 = float(roc_auc_score(yt_air, air_v3[vbz]))
        a_auc6 = float(roc_auc_score(yt_air, air_v6[vbz]))
        air_auc_v3_list.append(a_auc3)
        air_auc_v6_list.append(a_auc6)

        _, _, y_duel_to = compute_l1_keep_indices_and_weights(m.labels, m.half)
        live_m = (m.half > 0) & (m.labels != DEAD) & (m.labels >= 0)
        d_auc3 = float(roc_auc_score(y_duel_to[live_m], duel_v3[live_m]))
        duel_auc_v3_list.append(d_auc3)

        print(
            f"  [{mid}] 3-Tree J1A(v3): {f1_old_3:.4f} -> {f1_new_3:.4f} ({f1_new_3 - f1_old_3:+.4f}) | "
            f"J1B(v6): {f1_old_6:.4f} -> {f1_new_6:.4f} ({f1_new_6 - f1_old_6:+.4f}) | "
            f"AerialAUC(v3/v6): {a_auc3:.4f}/{a_auc6:.4f} | DuelToAUC: {d_auc3:.4f}",
            flush=True,
        )

    # Predict on Sample 1..3 + atd_raw, atd_rep, atd_reteam_rep
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)

    atd_rt = load_atd_reteam()
    mapping_rt, _ = enhanced_broadcast_alignment(atd_rt, Pb, hom_valid)
    atd_reteam_rep = repair_atd_match(atd_rt, Pb, Bb, slot_team_b, mapping_rt, True, True, None)

    def _eval_full_on_df(df: pd.DataFrame, half: np.ndarray, pitch: tuple[float, float]) -> dict[str, np.ndarray]:
        df_mir = mirror_v1ctl_physv2_df(df, half, pitch)
        X = df.to_numpy(dtype=np.float32, copy=False)
        Xm = df_mir.to_numpy(dtype=np.float32, copy=False)
        return {
            "gbm": _predict_twohead_tta(full_models["gbm"], X, Xm),
            "cb": _predict_twohead_tta(full_models["cb"], X, Xm),
            "xgb": _predict_twohead_tta(full_models["xgb"], X, Xm),
            "air": _predict_sym_binary_tta(full_models["aerial"], X, Xm),
            "duel": _predict_sym_binary_tta(full_models["duel"], X, Xm),
        }

    j23_preds = {"sample": {}}
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f:
        pe_p2 = pickle.load(f)
    with open(w5_dir / "eval_preds_cb_xgb.pkl", "rb") as f:
        pe_w5 = pickle.load(f)

    old_j2_raw, new_j2_raw, old_j2_nod, new_j2_nod = [], [], [], []
    for gid in GAMES:
        ms = sample_matches[gid]
        df_s = load_cached_physv2_table(f"sample_{gid}_raw", ms.half)
        j23_preds["sample"][gid] = _eval_full_on_df(df_s, ms.half, ms.pitch)

        old_p = 0.45 * pe_p2["sample"][gid]["g_physv2_sym"] + 0.30 * pe_w5["sample"][gid]["cb_physv2_sym"] + 0.25 * pe_w5["sample"][gid]["xgb_physv2_sym"]
        new_p = 0.45 * j23_preds["sample"][gid]["gbm"] + 0.30 * j23_preds["sample"][gid]["cb"] + 0.25 * j23_preds["sample"][gid]["xgb"]
        c_s = pe_p2["sample"][gid]["tr_physv2_sym"]

        pr_old = decode_v4(old_p, ms.half, dec_v5_hyb, p_change=c_s)
        pr_new = decode_v4(new_p, ms.half, dec_v5_hyb, p_change=c_s)
        old_j2_raw.append(float(metrics.score(ms.labels, pr_old, ms.half)["spell_f1_mean"]))
        new_j2_raw.append(float(metrics.score(ms.labels, pr_new, ms.half)["spell_f1_mean"]))
        y_nd = without_dead_margin(ms.labels, ms.half, 3.0)
        old_j2_nod.append(float(metrics.score(y_nd, pr_old, ms.half)["spell_f1_mean"]))
        new_j2_nod.append(float(metrics.score(y_nd, pr_new, ms.half)["spell_f1_mean"]))

    try:
        df_atd_raw = load_cached_physv2_table("atd_0f0834", atd.half)
    except IndexError:
        df_atd_raw = build_v1ctl_physics_v2(atd, "atd")
    try:
        df_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)
    except IndexError:
        df_atd_rep = build_v1ctl_physics_v2(atd_rep, "atd_rep_True_True_None")
    print("  Building/loading 481-col `v1ctl_physics_v2` table for `atd_reteam_rep`...", flush=True)
    df_atd_rt = build_v1ctl_physics_v2(atd_reteam_rep, "atd_reteam_rep")

    j23_preds["atd_raw"] = _eval_full_on_df(df_atd_raw, atd.half, atd.pitch)
    j23_preds["atd_rep"] = _eval_full_on_df(df_atd_rep, atd.half, atd.pitch)
    j23_preds["atd_reteam_rep"] = _eval_full_on_df(df_atd_rt, atd.half, atd.pitch)

    with open(out_dir / "l1_trees_j23_preds.pkl", "wb") as f:
        pickle.dump(j23_preds, f)

    summary = {
        "old_3tree_J1A": float(np.mean(old_3tree_j1a)),
        "new_3tree_J1A": float(np.mean(new_3tree_j1a)),
        "delta_J1A": float(np.mean(new_3tree_j1a) - np.mean(old_3tree_j1a)),
        "old_3tree_J1B": float(np.mean(old_3tree_j1b)),
        "new_3tree_J1B": float(np.mean(new_3tree_j1b)),
        "delta_J1B": float(np.mean(new_3tree_j1b) - np.mean(old_3tree_j1b)),
        "old_3tree_J2_raw": float(np.mean(old_j2_raw)),
        "new_3tree_J2_raw": float(np.mean(new_j2_raw)),
        "delta_J2_raw": float(np.mean(new_j2_raw) - np.mean(old_j2_raw)),
        "old_3tree_J2_noD": float(np.mean(old_j2_nod)),
        "new_3tree_J2_noD": float(np.mean(new_j2_nod)),
        "delta_J2_noD": float(np.mean(new_j2_nod) - np.mean(old_j2_nod)),
        "aerial_auc_v3_mean": float(np.mean(air_auc_v3_list)),
        "aerial_auc_v6_mean": float(np.mean(air_auc_v6_list)),
        "duel_to_auc_v3_mean": float(np.mean(duel_auc_v3_list)),
        "total_elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "l1_trees_benchmark.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 115, flush=True)
    print(
        f"[SUMMARY 3-Tree PhysV2 (W4/W5 -> V9 L1+MultiSeed)] "
        f"J1A: {summary['old_3tree_J1A']:.4f} -> {summary['new_3tree_J1A']:.4f} ({summary['delta_J1A']:+.4f}) | "
        f"J1B: {summary['old_3tree_J1B']:.4f} -> {summary['new_3tree_J1B']:.4f} ({summary['delta_J1B']:+.4f}) | "
        f"J2-Raw: {summary['old_3tree_J2_raw']:.4f} -> {summary['new_3tree_J2_raw']:.4f} ({summary['delta_J2_raw']:+.4f}) | "
        f"J2-noD: {summary['old_3tree_J2_noD']:.4f} -> {summary['new_3tree_J2_noD']:.4f} ({summary['delta_J2_noD']:+.4f}) | "
        f"AerialAUC: v3={summary['aerial_auc_v3_mean']:.4f}, v6={summary['aerial_auc_v6_mean']:.4f} | "
        f"DuelToAUC: {summary['duel_to_auc_v3_mean']:.4f}",
        flush=True,
    )
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()
