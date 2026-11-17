"""Modal GPU Swarm Job (`v11_big_leap`): Parallel Extraction + 8-GPU Training of `SymPhysConformerV11`.

1. Stage 1 (`extract_physv2_seq_worker`, `max_containers=8` CPU containers — 100% public data only):
   - Loads public raw tracking matches from Modal Volume `bcn-possession-v9` (`/vol/modal_bundle/`).
   - Computes exact 481-column `v1ctl_physics_v2` tables in memory, projects each table onto the
     exact H <-> A involution manifold (`mirror(mirror(X)) == X` with `0.0` error), and caches to
     `/vol/physv2_seq_cache_v11/`:
     * 7 Sportec matches x 6 realizations (`v3_s0, v6_s0, v3_s1, v3_s2, v3_static05_s1, v6_s1`)
     * 3 Metrica Sample matches (`sample_1, sample_2, sample_3`)
   - Competition-gated `BAR-ATH` data is NEVER uploaded to or processed on Modal.
2. Stage 2 (`train_phys_conformer_v11_gpu_worker`, `max_containers=8` `A10G`/`L4` GPU containers):
   - Workers `0..6`: 7 outer LOMO folds of a 2-seed `SymPhysConformerV11` ensemble trained on
     all 4 non-test realizations (`v3_s1, v3_s2, v3_static05_s1, v6_s1`) with online ball-blackout
     augmentation (`ball_drop_prob=0.20`), `L1` inverse-spell duration weights (`poss_alpha=0.40`),
     Differentiable Multi-Scale Windowed Soft-IoU Loss, and Dead-Ball Calibration Guard.
   - Worker `7`: Full 7-match training of a 2-seed `SymPhysConformerV11` ensemble, inference on
     Metrica Sample 1..3, and return of trained model checkpoints (`pc11_full_models.pkl`) to local.
3. Stage 3 (100% Local CPU/MPS):
   - Extracts `BAR-ATH` (`atd_raw`, `atd_rep`, `atd_reteam_rep`) locally and runs local inference with
     the returned `SymPhysConformerV11` models.
   - Saves OOF predictions to `experiments/outputs/v11_big_leap/phys_conformer_v11_oof/pc11_{mid}.npz`
   - Saves J2/J3 predictions to `experiments/outputs/v11_big_leap/phys_conformer_v11_oof/pc11_j23_preds.pkl`
   - Benchmarks `SymPhysConformerV11` standalone vs Wave 5B (`SymTCN_PhysV2` & `SymBiGRU_PhysV2`) across
     `J1A`, `J1B`, `J2_Raw`, `J2_noD`, Change AUC, 4-Dir AUC, Duel-Turnover AUC, and 3D Aerial AUC.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import sys
import time

import modal
import numpy as np

if modal.is_local():
    ROOT = Path(__file__).resolve().parents[2]
    for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "py_pkgs"), str(ROOT / "evaluation")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
else:
    ROOT = Path("/root/project")

APP_NAME = "bcn-v11-phys-conformer"
VOLUME_NAME = "bcn-possession-v9"
USE_GPU = os.environ.get("MODAL_GPU_TYPE", "A10G")
if "," in USE_GPU:
    USE_GPU = [g.strip() for g in USE_GPU.split(",")]
elif USE_GPU == "A10G":
    USE_GPU = ["A10G", "L4"]

SPORTEC_MATCH_IDS = (
    "J03WMX",
    "J03WN1",
    "J03WOH",
    "J03WOY",
    "J03WPY",
    "J03WQQ",
    "J03WR9",
)

TRAIN_V3_SPECS = (
    ("v3_s1", None, 1),
    ("v3_s2", None, 2),
    ("v3_static05_s1", {"static_hide_prob": 0.5}, 1),
)
TRAIN_V6_SPECS = (
    ("v6_s1", 1),
)
TRAIN_REALIZATIONS = (
    ("v3_s1", 1.00),
    ("v3_s2", 1.00),
    ("v3_static05_s1", 0.95),
    ("v6_s1", 0.85),
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "torch>=2.2.0",
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/root/project/src")
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/project/configs")
)

app = modal.App(APP_NAME, image=image)


def _setup_remote_paths() -> None:
    if "/root/project" not in sys.path:
        sys.path.insert(0, "/root/project")


def _load_match_from_npz(npz_path: Path):
    from src.data.common import Match

    z = np.load(npz_path, allow_pickle=False)
    meta = {}
    if "ball_z" in z.files:
        meta["ball_z"] = z["ball_z"].astype(np.float32)
    labels = z["labels"].astype(np.int8) if ("labels" in z.files and z["labels"].size > 0) else None
    return Match(
        P=z["P"].astype(np.float32),
        B=z["B"].astype(np.float32),
        half=z["half"].astype(np.int8),
        graded=z["graded"].astype(bool),
        attack_sign=z["attack_sign"].astype(np.int8),
        slot_team=z["slot_team"].astype(np.int8),
        slot_is_gk=z["slot_is_gk"].astype(bool),
        pitch=(float(z["pitch"][0]), float(z["pitch"][1])),
        home_name="H",
        away_name="A",
        labels=labels,
        source=str(z["source"]),
        direction_normalized=bool(z["direction_normalized"]),
        meta=meta,
    )


def _load_degrade_cfg(cfg_path: Path):
    from src.data.degrade import DegradeConfig

    raw = {
        k: (tuple(v) if isinstance(v, list) else v)
        for k, v in json.loads(cfg_path.read_text()).items()
        if not k.startswith("_")
    }
    return DegradeConfig(**raw)


def _extract_and_save_physv2_npz(
    m,
    out_p: Path,
    labels: np.ndarray | None = None,
    ball_z: np.ndarray | None = None,
) -> None:
    import pandas as pd
    from src.features.control import control_features
    from src.features.v1 import build_features
    from src.features.v1ctl_physics_v2 import compute_physv2_features, mirror_v1ctl_physv2_df
    from src.features.v1ctl_plus import compute_plus_features
    from src.models.transition import change_features

    X1 = change_features(build_features(m), m.half)
    X2 = control_features(m)
    X3 = compute_plus_features(m)
    X4 = compute_physv2_features(m)
    df_raw = pd.concat([X1, X2, X3, X4], axis=1)

    # Project onto exact H <-> A involution manifold so mirror(mirror(df_orig)) == df_orig (0.0 error)
    df_m1 = mirror_v1ctl_physv2_df(df_raw, m.half, m.pitch)
    df_orig = mirror_v1ctl_physv2_df(df_m1, m.half, m.pitch)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    kw = {
        "x_orig": df_orig.to_numpy(dtype=np.float32, copy=True),
        "col_names": np.array(list(df_orig.columns), dtype="<U48"),
        "half": m.half.astype(np.int8),
        "pitch": np.array(m.pitch, dtype=np.float32),
    }
    if labels is not None:
        kw["labels"] = labels.astype(np.int8)
    if ball_z is not None:
        kw["ball_z"] = ball_z.astype(np.float32)
    np.savez(out_p, **kw)


def _load_physv2_pair_npz(path: Path):
    import pandas as pd
    from src.features.v1ctl_physics_v2 import mirror_v1ctl_physv2_df

    z = np.load(path, allow_pickle=False)
    x_orig = z["x_orig"].astype(np.float32)
    col_names = [str(c) for c in z["col_names"]]
    half = z["half"].astype(np.int8)
    pitch = (float(z["pitch"][0]), float(z["pitch"][1]))
    labels = z["labels"].astype(np.int8) if "labels" in z.files else None
    ball_z = z["ball_z"].astype(np.float32) if "ball_z" in z.files else None

    df_orig = pd.DataFrame(x_orig, columns=col_names, copy=False)
    df_mir = mirror_v1ctl_physv2_df(df_orig, half, pitch)
    x_mir = df_mir.to_numpy(dtype=np.float32, copy=True)
    return x_orig, x_mir, col_names, half, labels, ball_z


@app.function(
    cpu=4.0,
    memory=12288,
    timeout=600,
    max_containers=8,
    retries=0,
    volumes={"/vol": vol},
)
def extract_physv2_seq_worker(task_spec: dict) -> dict:
    """Extracts and caches 481-column `v1ctl_physics_v2` sequence tables on `/vol/physv2_seq_cache_v11/`."""
    _setup_remote_paths()
    t0 = time.time()
    from src.data.degrade import degrade

    vol.reload()
    bundle_dir = Path("/vol/modal_bundle")
    cache_dir = Path("/vol/physv2_seq_cache_v11")
    cache_dir.mkdir(parents=True, exist_ok=True)

    proj_local = Path("/root/project/local_data")
    proj_local.mkdir(parents=True, exist_ok=True)
    for fn in ("degradation_params.json", "ball_interp_params.json"):
        src_p = bundle_dir / fn
        dst_p = proj_local / fn
        if src_p.exists() and not dst_p.exists():
            dst_p.write_text(src_p.read_text())

    kind = task_spec["kind"]
    name = task_spec["name"]
    built = []

    if kind == "sportec":
        mid = name
        m = _load_match_from_npz(bundle_dir / f"sportec_{mid}.npz")
        ball_z = m.meta.get("ball_z")
        cfg_v3 = _load_degrade_cfg(Path("/root/project/configs/degrade_v3.json"))
        cfg_v6 = _load_degrade_cfg(Path("/root/project/configs/degrade_v6.json"))

        for tag, cfg, seed in (("v3_s0", cfg_v3, 0), ("v6_s0", cfg_v6, 0)):
            out_p = cache_dir / f"sportec_{mid}_{tag}.npz"
            if not out_p.exists():
                mv = degrade(m, cfg, seed)
                _extract_and_save_physv2_npz(mv, out_p, m.labels, ball_z)
                built.append(out_p.name)

        for tag, overrides, seed in TRAIN_V3_SPECS:
            out_p = cache_dir / f"sportec_{mid}_{tag}.npz"
            if not out_p.exists():
                cfg = replace(cfg_v3, **(overrides or {}))
                mv = degrade(m, cfg, seed)
                _extract_and_save_physv2_npz(mv, out_p, m.labels, ball_z)
                built.append(out_p.name)

        for tag, seed in TRAIN_V6_SPECS:
            out_p = cache_dir / f"sportec_{mid}_{tag}.npz"
            if not out_p.exists():
                mv = degrade(m, cfg_v6, seed)
                _extract_and_save_physv2_npz(mv, out_p, m.labels, ball_z)
                built.append(out_p.name)
    else:
        out_p = cache_dir / f"{name}.npz"
        if not out_p.exists():
            m = _load_match_from_npz(bundle_dir / f"{name}.npz")
            _extract_and_save_physv2_npz(m, out_p, m.labels, None)
            built.append(out_p.name)

    if built:
        vol.commit()

    return {
        "name": name,
        "built": built,
        "elapsed_s": round(time.time() - t0, 2),
    }


def _sym_average_multi_head_preds(preds_list: list[tuple[np.ndarray, ...]]) -> dict[str, np.ndarray]:
    """Symmetrically average predictions from multiple seeds while preserving exact `0.000e+00` equivariance."""
    n = float(len(preds_list))
    proba = sum(p[0] for p in preds_list) / n
    denom = (proba[:, 0:1] + proba[:, 1:2]) + proba[:, 2:3]
    proba = (proba / np.clip(denom, 1e-6, None)).astype(np.float32)
    change = (sum(p[1] for p in preds_list) / n).astype(np.float32)
    p4d = (sum(p[2] for p in preds_list) / n).astype(np.float32)
    p_duel_to = (sum(p[3] for p in preds_list) / n).astype(np.float32)
    p_aerial = (sum(p[4] for p in preds_list) / n).astype(np.float32)
    ball_z_hat = (sum(p[5] for p in preds_list) / n).astype(np.float32)
    return {
        "proba": proba,
        "change": change,
        "p4d": p4d,
        "p_duel_to": p_duel_to,
        "p_aerial": p_aerial,
        "ball_z_hat": ball_z_hat,
    }


@app.function(
    gpu=USE_GPU,
    cpu=8.0,
    memory=32768,
    timeout=1800,
    max_containers=8,
    retries=0,
    volumes={"/vol": vol},
)
def train_phys_conformer_v11_gpu_worker(worker_idx: int) -> dict:
    """Runs either 1 outer LOMO fold (`worker_idx in 0..6`) or full 7-match training (`worker_idx == 7`) on GPU."""
    _setup_remote_paths()
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from src.v11_big_leap.phys_conformer_v11 import (
        SymPhysConformerV11,
        verify_phys_conformer_v11_equivariance,
    )

    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    vol.reload()
    cache_dir = Path("/vol/physv2_seq_cache_v11")
    res_dir = Path("/vol/phys_conformer_v11_results_v1")
    res_dir.mkdir(parents=True, exist_ok=True)
    cached_res_file = res_dir / f"worker_{worker_idx}.pkl"
    if cached_res_file.exists():
        return pickle.loads(cached_res_file.read_bytes())

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"cpu-{os.cpu_count()}c"

    if worker_idx < 7:
        held_out_mid = SPORTEC_MATCH_IDS[worker_idx]
        train_mids = [m for m in SPORTEC_MATCH_IDS if m != held_out_mid]
        seeds = (42 + worker_idx, 142 + worker_idx)
    else:
        held_out_mid = None
        train_mids = list(SPORTEC_MATCH_IDS)
        seeds = (42, 142)

    load_specs = []
    for mid in train_mids:
        for tag, w_tag in TRAIN_REALIZATIONS:
            p_file = cache_dir / f"sportec_{mid}_{tag}.npz"
            if p_file.exists():
                load_specs.append((p_file, w_tag))

    with ThreadPoolExecutor(max_workers=8) as ex:
        loaded = list(ex.map(lambda sp: (_load_physv2_pair_npz(sp[0]), sp[1]), load_specs))

    xs_o, xs_m, ys_tr, hs_tr, bzs_tr, ws_tr = [], [], [], [], [], []
    col_names = None
    for (xo, xm, cols, h, y, bz), w in loaded:
        if col_names is None:
            col_names = cols
        xs_o.append(xo)
        xs_m.append(xm)
        ys_tr.append(y)
        hs_tr.append(h)
        bzs_tr.append(bz)
        ws_tr.append(w)
    del loaded

    # Precompute symmetric normalizer and GPU-resident float16 segments ONCE across seeds
    prep_helper = SymPhysConformerV11(
        col_names=col_names,
        crop=1500,
        poss_alpha=0.40,
        boundary_boost=1.65,
    )
    shared_segments = prep_helper.prepare_segments(
        xs_o, xs_m, ys_tr, hs_tr, ball_zs=bzs_tr, sample_weights=ws_tr
    )
    shared_sym_stats = (prep_helper.sym_mean, prep_helper.sym_std)
    del xs_o, xs_m

    models = []
    for sd in seeds:
        m_net = SymPhysConformerV11(
            col_names=col_names,
            channels=96,
            dilations=(1, 2, 4, 8, 16, 32, 64),
            conformer_stride=8,
            window_size=1501,
            dropout=0.16,
            lr=1.5e-3,
            weight_decay=1.2e-4,
            epochs=8,
            crop=1500,
            batch_size=32,
            poss_alpha=0.40,
            boundary_boost=1.65,
            soft_iou_weight=0.28,
            dead_guard_weight=0.10,
            change_loss_weight=0.35,
            b4d_loss_weight=0.48,
            duel_loss_weight=0.25,
            aerial_loss_weight=0.20,
            ball_drop_prob=0.20,
            feat_noise_std=0.03,
            label_smoothing=0.02,
            sliding_window_blend=0.50,
            seed=sd,
        ).fit(
            [], [], ys_tr, hs_tr,
            prepared_segments=shared_segments,
            sym_stats=shared_sym_stats,
        )
        models.append(m_net)
    del shared_segments
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if held_out_mid is not None:
        xo_v3, xm_v3, _, h_v3, _, _ = _load_physv2_pair_npz(cache_dir / f"sportec_{held_out_mid}_v3_s0.npz")
        xo_v6, xm_v6, _, h_v6, _, _ = _load_physv2_pair_npz(cache_dir / f"sportec_{held_out_mid}_v6_s0.npz")

        sl = slice(0, min(3000, len(h_v3)))
        eq_chk = verify_phys_conformer_v11_equivariance(models[0], xo_v3[sl], xm_v3[sl], h_v3[sl])

        preds_v3 = _sym_average_multi_head_preds([m.predict_all_heads(xo_v3, xm_v3, h_v3) for m in models])
        preds_v6 = _sym_average_multi_head_preds([m.predict_all_heads(xo_v6, xm_v6, h_v6) for m in models])

        out = {
            "mode": "lomo",
            "mid": held_out_mid,
            "gpu": gpu_name,
            "n_train_seqs": len(ys_tr),
            "equivariance": eq_chk,
            "v3": preds_v3,
            "v6": preds_v6,
            "elapsed_s": round(time.time() - t0, 2),
        }
        cached_res_file.write_bytes(pickle.dumps(out))
        vol.commit()
        return out

    eval_out = {"sample": {}}
    for gid in (1, 2, 3):
        xo_s, xm_s, _, h_s, _, _ = _load_physv2_pair_npz(cache_dir / f"sample_{gid}.npz")
        eval_out["sample"][gid] = _sym_average_multi_head_preds([m.predict_all_heads(xo_s, xm_s, h_s) for m in models])

    for m in models:
        if getattr(m, "net", None) is not None:
            m.net = m.net.to("cpu").eval()
        m.device = torch.device("cpu")

    out = {
        "mode": "full_eval",
        "gpu": gpu_name,
        "n_train_seqs": len(ys_tr),
        "preds": eval_out,
        "models_pkl": pickle.dumps(models),
        "elapsed_s": round(time.time() - t0, 2),
    }
    cached_res_file.write_bytes(pickle.dumps(out))
    vol.commit()
    return out


def _predict_atd_locally_pc11(models: list) -> dict[str, dict[str, np.ndarray]]:
    """Extract 481-column `v1ctl_physics_v2` tables and predict `BAR-ATH` (`atd_raw`, `atd_rep`, `atd_reteam_rep`) 100% locally."""
    import pandas as pd
    from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
    from s4_6_broadcast_align import load_broadcast
    from src.data.metrica_atd import load as load_atd
    from src.features.v1ctl_physics_v2 import build_v1ctl_physics_v2, mirror_v1ctl_physv2_df

    atd = load_atd()
    Pb, Bb, team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)
    mapping, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
    atd_rep = repair_atd_match(atd, Pb, Bb, team_b, mapping, True, True, None)

    reteam_path = ROOT / "local_data" / "video" / "players_reteam.npz"
    if reteam_path.is_file():
        z = np.load(reteam_path)
        atd_rt = replace(
            atd,
            P=z["P"].astype(atd.P.dtype),
            slot_team=z["slot_team"].astype(atd.slot_team.dtype),
            slot_is_gk=z["slot_is_gk"].astype(bool),
        )
        mapping_rt, _ = enhanced_broadcast_alignment(atd_rt, Pb, hom_valid)
        atd_reteam_rep = repair_atd_match(atd_rt, Pb, Bb, team_b, mapping_rt, True, True, None)
    else:
        atd_reteam_rep = atd_rep

    local_atd_out = {}
    for k_name, c_key, m_obj in (
        ("atd_raw", "atd", atd),
        ("atd_rep", "atd_rep_True_True_None", atd_rep),
        ("atd_reteam_rep", "atd_reteam_rep", atd_reteam_rep),
    ):
        df_raw = build_v1ctl_physics_v2(m_obj, c_key)
        df_m1 = mirror_v1ctl_physv2_df(df_raw, m_obj.half, m_obj.pitch)
        df_orig = mirror_v1ctl_physv2_df(df_m1, m_obj.half, m_obj.pitch)
        df_mir = mirror_v1ctl_physv2_df(df_orig, m_obj.half, m_obj.pitch)
        xo_a = df_orig.to_numpy(dtype=np.float32, copy=True)
        xm_a = df_mir.to_numpy(dtype=np.float32, copy=True)
        local_atd_out[k_name] = _sym_average_multi_head_preds(
            [m.predict_all_heads(xo_a, xm_a, m_obj.half) for m in models]
        )
    return local_atd_out


def main() -> None:
    t_start = time.time()
    print("=" * 115, flush=True)
    print("MODAL GPU SWARM (`v11_big_leap`): 8 x A10G GPUs — `SymPhysConformerV11` ON 481 `PhysV2` FEATURES (PUBLIC ONLY)", flush=True)
    print("=" * 115, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "v11_big_leap" / "phys_conformer_v11_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    have_all_local = (
        (out_dir / "pc11_j23_preds.pkl").exists()
        and all((out_dir / f"pc11_{mid}.npz").exists() for mid in SPORTEC_MATCH_IDS)
        and os.environ.get("FORCE_MODAL", "0") != "1"
    )
    if not have_all_local:
        tasks = [{"kind": "sportec", "name": mid} for mid in SPORTEC_MATCH_IDS]
        for gid in (1, 2, 3):
            tasks.append({"kind": "sample", "name": f"sample_{gid}"})

        with app.run():
            print(f"\n[Stage 1] Extracting/verifying public 481-col `PhysV2` tables on Modal Volume ({len(tasks)} public tasks, max 8 containers)...", flush=True)
            t1 = time.time()
            ext_res = list(extract_physv2_seq_worker.map(tasks))
            n_built = sum(len(r["built"]) for r in ext_res)
            print(f"[Stage 1 Complete] Built {n_built} new `.npz` tables in {time.time() - t1:.1f}s", flush=True)

            print("\n[Stage 2] Launching 8 parallel Modal GPU containers (7 LOMO folds + 1 Full 7-Match Model)...", flush=True)
            t2 = time.time()
            gpu_results = list(train_phys_conformer_v11_gpu_worker.map(list(range(8))))
            print(f"[Stage 2 Complete] All 8 GPU workers finished in {time.time() - t2:.1f}s!", flush=True)

        for r in gpu_results:
            if r["mode"] == "lomo":
                mid = r["mid"]
                eq = r["equivariance"]
                max_eq = max(eq.values())
                print(
                    f"  [LOMO {mid}] GPU={r['gpu']} | seqs={r['n_train_seqs']} | "
                    f"max_equiv_err={max_eq:.3e} | time={r['elapsed_s']:.1f}s",
                    flush=True,
                )
                assert max_eq < 1e-5, f"Equivariance violation on {mid}: {eq}"
                v3, v6 = r["v3"], r["v6"]
                np.savez_compressed(
                    out_dir / f"pc11_{mid}.npz",
                    v3_proba=v3["proba"],
                    v3_change=v3["change"],
                    v3_p4d=v3["p4d"],
                    v3_duel_to=v3["p_duel_to"],
                    v3_aerial=v3["p_aerial"],
                    v3_ball_z_hat=v3["ball_z_hat"],
                    v6_proba=v6["proba"],
                    v6_change=v6["change"],
                    v6_p4d=v6["p4d"],
                    v6_duel_to=v6["p_duel_to"],
                    v6_aerial=v6["p_aerial"],
                    v6_ball_z_hat=v6["ball_z_hat"],
                )
            else:
                print(
                    f"  [FULL 7-MATCH] GPU={r['gpu']} | seqs={r['n_train_seqs']} | time={r['elapsed_s']:.1f}s",
                    flush=True,
                )
                preds = dict(r["preds"])
                full_models = pickle.loads(r["models_pkl"])
                with open(out_dir / "pc11_full_models.pkl", "wb") as f:
                    pickle.dump(full_models, f)
                print("  Running 100% local inference on BAR-ATH (`atd_raw`, `atd_rep`, `atd_reteam_rep`)...", flush=True)
                preds.update(_predict_atd_locally_pc11(full_models))
                with open(out_dir / "pc11_j23_preds.pkl", "wb") as f:
                    pickle.dump(preds, f)
    else:
        print("\n[Stage 1-2] Found cached `pc11_*.npz` and `pc11_j23_preds.pkl` locally — skipping Modal launch!", flush=True)

    # =========================================================================
    # STAGE 3: STANDALONE BENCHMARK VS WAVE 5B (`SymTCN_PhysV2` & `SymBiGRU_PhysV2`)
    # =========================================================================
    print("\n[Stage 3] Benchmarking `SymPhysConformerV11` vs Wave 5B (`SymTCN_PhysV2` + `SymBiGRU_PhysV2`)...", flush=True)
    from sklearn.metrics import roc_auc_score
    from s3_11_tcn_local import without_dead_margin
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.sportec import load_all as load_sportec
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.eval import metrics
    from src.models.boundary_4dir import targets_4dir
    from src.models.transition import change_targets

    sportec = load_sportec()
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    w5b_dir = ROOT / "experiments" / "outputs" / "exp_w5b_oof"
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)

    tcn_j1a, gru_j1a, pc11_j1a = [], [], []
    tcn_j1b, gru_j1b, pc11_j1b = [], [], []
    chg_auc_v3, b4d_auc_v3, duel_auc_v3, air_auc_v3 = [], [], [], []
    chg_auc_v6, b4d_auc_v6, duel_auc_v6, air_auc_v6 = [], [], [], []

    for mid in SPORTEC_MATCH_IDS:
        m = sportec[mid]
        zt = np.load(w5b_dir / f"tcn_physv2_{mid}.npz")
        zg = np.load(w5b_dir / f"bigru_physv2_{mid}.npz")
        zp = np.load(out_dir / f"pc11_{mid}.npz")

        f_tcn_3 = float(metrics.score(m.labels, decode_v4(zt["v3_proba"], m.half, loo_dec_hyb[mid], p_change=zt["v3_change"]), m.half)["spell_f1_mean"])
        f_gru_3 = float(metrics.score(m.labels, decode_v4(zg["v3_proba"], m.half, loo_dec_hyb[mid], p_change=zg["v3_change"]), m.half)["spell_f1_mean"])
        f_pc_3 = float(metrics.score(m.labels, decode_v4(zp["v3_proba"], m.half, loo_dec_hyb[mid], p_change=zp["v3_change"]), m.half)["spell_f1_mean"])

        f_tcn_6 = float(metrics.score(m.labels, decode_v4(zt["v6_proba"], m.half, loo_dec_hyb[mid], p_change=zt["v6_change"]), m.half)["spell_f1_mean"])
        f_gru_6 = float(metrics.score(m.labels, decode_v4(zg["v6_proba"], m.half, loo_dec_hyb[mid], p_change=zg["v6_change"]), m.half)["spell_f1_mean"])
        f_pc_6 = float(metrics.score(m.labels, decode_v4(zp["v6_proba"], m.half, loo_dec_hyb[mid], p_change=zp["v6_change"]), m.half)["spell_f1_mean"])

        tcn_j1a.append(f_tcn_3)
        gru_j1a.append(f_gru_3)
        pc11_j1a.append(f_pc_3)
        tcn_j1b.append(f_tcn_6)
        gru_j1b.append(f_gru_6)
        pc11_j1b.append(f_pc_6)

        yc = change_targets(m.labels, m.half)
        vc = yc >= 0
        chg_auc_v3.append(float(roc_auc_score(yc[vc], zp["v3_change"][vc])))
        chg_auc_v6.append(float(roc_auc_score(yc[vc], zp["v6_change"][vc])))

        t4 = targets_4dir(m.labels, m.half)
        y4_stack = np.column_stack([t4["y_h2a"], t4["y_a2h"], t4["y_live2d"], t4["y_d2live"]])
        v4 = y4_stack[:, 0] >= 0
        b4d_auc_v3.append(float(np.mean([roc_auc_score(y4_stack[v4, k], zp["v3_p4d"][v4, k]) for k in range(4)])))
        b4d_auc_v6.append(float(np.mean([roc_auc_score(y4_stack[v4, k], zp["v6_p4d"][v4, k]) for k in range(4)])))

        from src.v10_first_principles.traj_event_net_v10 import compute_v10_boundary_and_duel_targets
        _, _, y_duel = compute_v10_boundary_and_duel_targets(m.labels, m.half)
        vd = y_duel >= 0
        duel_auc_v3.append(float(roc_auc_score((y_duel[vd] > 0.5).astype(int), zp["v3_duel_to"][vd])))
        duel_auc_v6.append(float(roc_auc_score((y_duel[vd] > 0.5).astype(int), zp["v6_duel_to"][vd])))

        bz = m.meta["ball_z"]
        vbz = (m.half > 0) & np.isfinite(bz)
        air_auc_v3.append(float(roc_auc_score((bz[vbz] > 1.5).astype(int), zp["v3_aerial"][vbz])))
        air_auc_v6.append(float(roc_auc_score((bz[vbz] > 1.5).astype(int), zp["v6_aerial"][vbz])))

        print(
            f"  [{mid}] J1A(v3): TCN={f_tcn_3:.4f}, BiGRU={f_gru_3:.4f} -> PC11={f_pc_3:.4f} | "
            f"J1B(v6): TCN={f_tcn_6:.4f}, BiGRU={f_gru_6:.4f} -> PC11={f_pc_6:.4f} | "
            f"4DirAUC(v3/v6)={b4d_auc_v3[-1]:.4f}/{b4d_auc_v6[-1]:.4f} | AirAUC(v3/v6)={air_auc_v3[-1]:.4f}/{air_auc_v6[-1]:.4f}",
            flush=True,
        )

    with open(w5b_dir / "seq_physv2_j23_preds.pkl", "rb") as f:
        w5b_j23 = pickle.load(f)
    with open(out_dir / "pc11_j23_preds.pkl", "rb") as f:
        pc11_j23 = pickle.load(f)

    tcn_j2_r, gru_j2_r, pc11_j2_r = [], [], []
    tcn_j2_nd, gru_j2_nd, pc11_j2_nd = [], [], []
    for gid in GAMES:
        ms = load_sample(gid)
        y_nd = without_dead_margin(ms.labels, ms.half, 3.0)
        pt = decode_v4(w5b_j23["sample"][gid]["tcn_physv2_sym"], ms.half, dec_v5_hyb, p_change=w5b_j23["sample"][gid]["tcnc_physv2_sym"])
        pg = decode_v4(w5b_j23["sample"][gid]["bigru_physv2_sym"], ms.half, dec_v5_hyb, p_change=w5b_j23["sample"][gid]["bigruc_physv2_sym"])
        pp = decode_v4(pc11_j23["sample"][gid]["proba"], ms.half, dec_v5_hyb, p_change=pc11_j23["sample"][gid]["change"])

        tcn_j2_r.append(float(metrics.score(ms.labels, pt, ms.half)["spell_f1_mean"]))
        gru_j2_r.append(float(metrics.score(ms.labels, pg, ms.half)["spell_f1_mean"]))
        pc11_j2_r.append(float(metrics.score(ms.labels, pp, ms.half)["spell_f1_mean"]))

        tcn_j2_nd.append(float(metrics.score(y_nd, pt, ms.half)["spell_f1_mean"]))
        gru_j2_nd.append(float(metrics.score(y_nd, pg, ms.half)["spell_f1_mean"]))
        pc11_j2_nd.append(float(metrics.score(y_nd, pp, ms.half)["spell_f1_mean"]))

    summary = {
        "tcn_physv2_J1A": float(np.mean(tcn_j1a)),
        "bigru_physv2_J1A": float(np.mean(gru_j1a)),
        "pc11_J1A": float(np.mean(pc11_j1a)),
        "tcn_physv2_J1B": float(np.mean(tcn_j1b)),
        "bigru_physv2_J1B": float(np.mean(gru_j1b)),
        "pc11_J1B": float(np.mean(pc11_j1b)),
        "tcn_physv2_J2_raw": float(np.mean(tcn_j2_r)),
        "bigru_physv2_J2_raw": float(np.mean(gru_j2_r)),
        "pc11_J2_raw": float(np.mean(pc11_j2_r)),
        "tcn_physv2_J2_noD": float(np.mean(tcn_j2_nd)),
        "bigru_physv2_J2_noD": float(np.mean(gru_j2_nd)),
        "pc11_J2_noD": float(np.mean(pc11_j2_nd)),
        "chg_auc_v3": float(np.mean(chg_auc_v3)),
        "chg_auc_v6": float(np.mean(chg_auc_v6)),
        "b4d_auc_v3": float(np.mean(b4d_auc_v3)),
        "b4d_auc_v6": float(np.mean(b4d_auc_v6)),
        "duel_auc_v3": float(np.mean(duel_auc_v3)),
        "duel_auc_v6": float(np.mean(duel_auc_v6)),
        "air_auc_v3": float(np.mean(air_auc_v3)),
        "air_auc_v6": float(np.mean(air_auc_v6)),
        "total_elapsed_s": round(time.time() - t_start, 2),
    }
    (out_dir.parent / "phys_conformer_v11_benchmark.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 115, flush=True)
    print(
        f"[STANDALONE SEQUENCE BENCHMARK (481 PhysV2)] "
        f"J1A(v3): TCN={summary['tcn_physv2_J1A']:.4f}, BiGRU={summary['bigru_physv2_J1A']:.4f} -> PC11={summary['pc11_J1A']:.4f} | "
        f"J1B(v6): TCN={summary['tcn_physv2_J1B']:.4f}, BiGRU={summary['bigru_physv2_J1B']:.4f} -> PC11={summary['pc11_J1B']:.4f} | "
        f"J2-Raw: TCN={summary['tcn_physv2_J2_raw']:.4f}, BiGRU={summary['bigru_physv2_J2_raw']:.4f} -> PC11={summary['pc11_J2_raw']:.4f} | "
        f"J2-noD: TCN={summary['tcn_physv2_J2_noD']:.4f}, BiGRU={summary['bigru_physv2_J2_noD']:.4f} -> PC11={summary['pc11_J2_noD']:.4f}",
        flush=True,
    )
    print(
        f"  Aux Heads (v3/v6 AUC): Change={summary['chg_auc_v3']:.4f}/{summary['chg_auc_v6']:.4f} | "
        f"4-Dir={summary['b4d_auc_v3']:.4f}/{summary['b4d_auc_v6']:.4f} | "
        f"DuelTo={summary['duel_auc_v3']:.4f}/{summary['duel_auc_v6']:.4f} | "
        f"Aerial={summary['air_auc_v3']:.4f}/{summary['air_auc_v6']:.4f}",
        flush=True,
    )
    print("=" * 115, flush=True)


if __name__ == "__main__":
    main()
