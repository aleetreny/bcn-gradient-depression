"""Modal GPU Swarm Job #1: Parallel Extraction + 8-GPU Parallel Training of `SymPlayerSetNetV2`.

Runs on Modal cloud infrastructure (100% public datasets only — zero competition data leaves local machine):
- Stage 1 (10 parallel CPU containers): Extracts & caches `PlayerSetTensors` on Modal Volume
  `bcn-possession-v9` (`/vol/pset_v2_cache/`) for:
  - 7 Sportec matches x 6 degradation realizations (`v3_s0, v3_s1, v3_s2, v3_static05_s1, v6_s0, v6_s1`)
  - 3 Metrica Sample matches (`sample_1, sample_2, sample_3`)
- Stage 2 (8 parallel `L4` GPU containers):
  - Containers 0..6: 7-fold outer LOMO training of 2-seed `SymPlayerSetNetV2` ensembles
    (with Line L1 possession-duration weighting `alpha=0.45`, Line L5 4-directional action spotting head,
    Idea O1 3D aerial ball height `ball_z` head & gate, and Line D3 online `12 vs 10` team-swap augmentation).
  - Container 7: Full 7-match training of a 2-seed `SymPlayerSetNetV2` ensemble, evaluation on
    Metrica Sample 1-3, and return of trained model weights (`pset_v2_full_models.pkl`) to local.
- Stage 3 (100% Local CPU/MPS): Extracts `BAR-ATH` (`atd_raw`, `atd_rep`, `atd_reteam_rep`) locally,
  runs local inference with the returned `SymPlayerSetNetV2` models, saves all OOF and J2/J3 predictions to
  `experiments/outputs/v9_swarm/pset_v2_oof/`, and benchmarks `SymPlayerSetNetV2` vs `SymPlayerSetNet (v1)`.
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
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
else:
    ROOT = Path("/root/project")

APP_NAME = "bcn-v9-pset-v2-swarm"
VOLUME_NAME = "bcn-possession-v9"
USE_GPU_ENV = os.environ.get("MODAL_GPU_TYPE", "any")
if USE_GPU_ENV.lower() in ("none", "cpu", "0", "false"):
    USE_GPU = None
elif USE_GPU_ENV.lower() == "any":
    USE_GPU = ["T4", "L4", "A10G"]
else:
    USE_GPU = [g.strip() for g in USE_GPU_ENV.split(",")] if "," in USE_GPU_ENV else USE_GPU_ENV

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
    ("v6_s1", 11),
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


def _setup_remote_paths():
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


def _save_pst_npz(path: Path, pst, half: np.ndarray, labels: np.ndarray | None = None, ball_z: np.ndarray | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {
        "p_h": pst.p_h.astype(np.float32),
        "p_a": pst.p_a.astype(np.float32),
        "mask_h": pst.mask_h.astype(bool),
        "mask_a": pst.mask_a.astype(bool),
        "g_anti": pst.g_anti.astype(np.float32),
        "g_sym": pst.g_sym.astype(np.float32),
        "half": half.astype(np.int8),
    }
    if labels is not None:
        kw["labels"] = labels.astype(np.int8)
    if ball_z is not None:
        kw["ball_z"] = ball_z.astype(np.float32)
    np.savez_compressed(path, **kw)


def _load_pst_npz(path: Path):
    from src.models.player_set_net import PlayerSetTensors

    z = np.load(path)
    pst = PlayerSetTensors(
        p_h=z["p_h"].astype(np.float32),
        p_a=z["p_a"].astype(np.float32),
        mask_h=z["mask_h"].astype(bool),
        mask_a=z["mask_a"].astype(bool),
        g_anti=z["g_anti"].astype(np.float32),
        g_sym=z["g_sym"].astype(np.float32),
    )
    half = z["half"].astype(np.int8)
    labels = z["labels"].astype(np.int8) if "labels" in z.files else None
    ball_z = z["ball_z"].astype(np.float32) if "ball_z" in z.files else None
    return pst, half, labels, ball_z


@app.function(
    cpu=4.0,
    memory=8192,
    timeout=450,
    retries=0,
    volumes={"/vol": vol},
)
def extract_match_tensors_worker(task_spec: dict) -> dict:
    """Extracts and caches `PlayerSetTensors` on the Modal Volume for one match."""
    _setup_remote_paths()
    t0 = time.time()
    from src.data.degrade import degrade
    from src.models.player_set_net import extract_player_set_tensors

    vol.reload()
    bundle_dir = Path("/vol/modal_bundle")
    cache_dir = Path("/vol/pset_v2_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Ensure local_data/degradation_params.json and ball_interp_params.json exist at /root/project/local_data/
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

        # Test realizations (s0)
        for tag, cfg, seed in (("v3_s0", cfg_v3, 0), ("v6_s0", cfg_v6, 0)):
            out_p = cache_dir / f"sportec_{mid}_{tag}.npz"
            if not out_p.exists():
                mv = degrade(m, cfg, seed)
                pst = extract_player_set_tensors(mv)
                _save_pst_npz(out_p, pst, m.half, m.labels, ball_z)
                built.append(out_p.name)

        # Train v3 realizations
        for tag, overrides, seed in TRAIN_V3_SPECS:
            out_p = cache_dir / f"sportec_{mid}_{tag}.npz"
            if not out_p.exists():
                cfg = replace(cfg_v3, **(overrides or {}))
                mv = degrade(m, cfg, seed)
                pst = extract_player_set_tensors(mv)
                _save_pst_npz(out_p, pst, m.half, m.labels, ball_z)
                built.append(out_p.name)

        # Train v6 realizations
        for tag, seed in TRAIN_V6_SPECS:
            out_p = cache_dir / f"sportec_{mid}_{tag}.npz"
            if not out_p.exists():
                mv = degrade(m, cfg_v6, seed)
                pst = extract_player_set_tensors(mv)
                _save_pst_npz(out_p, pst, m.half, m.labels, ball_z)
                built.append(out_p.name)

    else:
        out_p = cache_dir / f"{name}.npz"
        if not out_p.exists():
            m = _load_match_from_npz(bundle_dir / f"{name}.npz")
            pst = extract_player_set_tensors(m)
            _save_pst_npz(out_p, pst, m.half, m.labels, None)
            built.append(out_p.name)

    if built:
        vol.commit()

    return {
        "name": name,
        "built": built,
        "elapsed_s": round(time.time() - t0, 2),
    }


def _average_multi_head_preds(preds_list: list[tuple[np.ndarray, ...]]) -> dict[str, np.ndarray]:
    n = float(len(preds_list))
    proba = sum(p[0] for p in preds_list) / n
    proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-6, None)
    change = sum(p[1] for p in preds_list) / n
    p4d = sum(p[2] for p in preds_list) / n
    p_aerial = sum(p[3] for p in preds_list) / n
    ball_z_hat = sum(p[4] for p in preds_list) / n
    return {
        "proba": proba.astype(np.float32),
        "change": change.astype(np.float32),
        "p4d": p4d.astype(np.float32),
        "p_aerial": p_aerial.astype(np.float32),
        "ball_z_hat": ball_z_hat.astype(np.float32),
    }


@app.function(
    gpu=USE_GPU,
    cpu=4.0 if USE_GPU else 16.0,
    memory=16384,
    timeout=600,
    retries=0,
    volumes={"/vol": vol},
)
def train_pset_v2_gpu_worker(worker_idx: int) -> dict:
    """Runs either 1 outer LOMO fold (`worker_idx in 0..6`) or full 7-match training (`worker_idx == 7`) on GPU."""
    _setup_remote_paths()
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from src.v9_swarm.player_set_net_v2 import SymPlayerSetNetV2

    torch.set_num_threads(8)
    vol.reload()
    cache_dir = Path("/vol/pset_v2_cache")
    res_dir = Path("/vol/pset_v2_results")
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
        for tag, _, _ in TRAIN_V3_SPECS:
            load_specs.append((cache_dir / f"sportec_{mid}_{tag}.npz", 1.0))
        for tag, _ in TRAIN_V6_SPECS:
            load_specs.append((cache_dir / f"sportec_{mid}_{tag}.npz", 0.80))

    with ThreadPoolExecutor(max_workers=8) as ex:
        loaded = list(ex.map(lambda sp: (_load_pst_npz(sp[0]), sp[1]), load_specs))

    psts_tr, ys_tr, hs_tr, bzs_tr, ws_tr = [], [], [], [], []
    for (pst, h, y, bz), w in loaded:
        psts_tr.append(pst)
        ys_tr.append(y)
        hs_tr.append(h)
        bzs_tr.append(bz)
        ws_tr.append(w)

    models = []
    for sd in seeds:
        m_net = SymPlayerSetNetV2(
            channels=64,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            gru_stride=10,
            dropout=0.15,
            lr=1.6e-3,
            weight_decay=1e-4,
            epochs=9,
            crop=1500,
            batch_size=32,
            poss_alpha=0.45,
            boundary_boost=1.6,
            change_loss_weight=0.35,
            b4d_loss_weight=0.45,
            aerial_loss_weight=0.20,
            swap_aug_prob=0.35,
            label_smoothing=0.02,
            seed=sd,
        ).fit(psts_tr, ys_tr, hs_tr, ball_zs=bzs_tr, sample_weights=ws_tr)
        models.append(m_net)

    if held_out_mid is not None:
        pst_v3, h_v3, _, _ = _load_pst_npz(cache_dir / f"sportec_{held_out_mid}_v3_s0.npz")
        pst_v6, h_v6, _, _ = _load_pst_npz(cache_dir / f"sportec_{held_out_mid}_v6_s0.npz")

        preds_v3 = _average_multi_head_preds([m.predict_all_heads(pst_v3, h_v3) for m in models])
        preds_v6 = _average_multi_head_preds([m.predict_all_heads(pst_v6, h_v6) for m in models])

        out = {
            "mode": "lomo",
            "mid": held_out_mid,
            "gpu": gpu_name,
            "elapsed_s": round(time.time() - t0, 2),
            "v3": preds_v3,
            "v6": preds_v6,
        }
    else:
        eval_names = ("sample_1", "sample_2", "sample_3")
        out_eval = {}
        for ename in eval_names:
            pst_e, h_e, _, _ = _load_pst_npz(cache_dir / f"{ename}.npz")
            out_eval[ename] = _average_multi_head_preds([m.predict_all_heads(pst_e, h_e) for m in models])

        for m in models:
            if getattr(m, "net_", None) is not None:
                m.net_ = m.net_.to("cpu").eval()

        out = {
            "mode": "full",
            "gpu": gpu_name,
            "elapsed_s": round(time.time() - t0, 2),
            "eval": out_eval,
            "models_pkl": pickle.dumps(models),
        }

    cached_res_file.write_bytes(pickle.dumps(out))
    vol.commit()
    return out


def _predict_atd_locally_pset_v2(models: list) -> dict[str, dict[str, np.ndarray]]:
    """Extract and predict `BAR-ATH` (`atd_raw`, `atd_rep`, `atd_reteam_rep`) 100% on the local machine."""
    import pandas as pd
    from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
    from s4_6_broadcast_align import load_broadcast
    from src.data.metrica_atd import load as load_atd
    from src.models.player_set_net import cached_player_set_tensors

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

    pset_cdir = ROOT / "local_data" / "cache" / "pset"
    local_atd_out = {}
    for k_name, m_obj in (("atd_raw", atd), ("atd_rep", atd_rep), ("atd_reteam_rep", atd_reteam_rep)):
        pst_local = cached_player_set_tensors(m_obj, pset_cdir, k_name)
        local_atd_out[k_name] = _average_multi_head_preds(
            [m.predict_all_heads(pst_local, m_obj.half) for m in models]
        )
    return local_atd_out


def run_modal_swarm_and_evaluate():
    t_start = time.time()
    print("=" * 115, flush=True)
    print("MODAL GPU SWARM #1: `SymPlayerSetNetV2` (10 PUBLIC CPU EXTRACTORS + 8 PARALLEL L4 GPUs + LOCAL BAR-ATH)", flush=True)
    print("=" * 115, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "v9_swarm" / "pset_v2_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    have_all_local = (
        all((out_dir / f"pset_v2_{mid}.npz").exists() for mid in SPORTEC_MATCH_IDS)
        and (out_dir / "pset_v2_j23_preds.pkl").exists()
        and os.environ.get("FORCE_MODAL", "0") != "1"
    )
    if have_all_local:
        print("\n[Stage 1-2] Found cached `pset_v2_*.npz` and `pset_v2_j23_preds.pkl` locally — skipping Modal launch!", flush=True)
    else:
        extract_specs = [{"kind": "sportec", "name": mid} for mid in SPORTEC_MATCH_IDS] + [
            {"kind": "eval", "name": ename}
            for ename in ("sample_1", "sample_2", "sample_3")
        ]

        with app.run():
            if os.environ.get("SKIP_STAGE1", "0") != "1":
                print("\n[Stage 1] Launching 10 parallel Modal CPU workers to extract public PlayerSetTensors...", flush=True)
                t1 = time.time()
                ext_results = list(extract_match_tensors_worker.map(extract_specs))
                for r in ext_results:
                    print(f"  [Cache {r['name']:15s}] built={len(r['built'])} files in {r['elapsed_s']:.1f}s", flush=True)
                print(f"[Stage 1 Complete] All public tensors cached on Modal Volume in {time.time() - t1:.1f}s", flush=True)
            else:
                print("\n[Stage 1] Skipping Stage 1 (all 10 public match tensor sets already cached on `/vol/pset_v2_cache`)!", flush=True)

            print(f"\n[Stage 2] Launching 8 parallel Modal GPU containers (gpu={USE_GPU})...", flush=True)
            t2 = time.time()
            gpu_results = list(train_pset_v2_gpu_worker.map(list(range(8))))
            print(f"[Stage 2 Complete] All 8 GPU jobs finished in {time.time() - t2:.1f}s wall-clock!", flush=True)

        for res in gpu_results:
            if res["mode"] == "lomo":
                mid = res["mid"]
                v3 = res["v3"]
                v6 = res["v6"]
                np.savez_compressed(
                    out_dir / f"pset_v2_{mid}.npz",
                    v3_proba=v3["proba"],
                    v3_change=v3["change"],
                    v3_p4d=v3["p4d"],
                    v3_p_aerial=v3["p_aerial"],
                    v3_ball_z_hat=v3["ball_z_hat"],
                    v6_proba=v6["proba"],
                    v6_change=v6["change"],
                    v6_p4d=v6["p4d"],
                    v6_p_aerial=v6["p_aerial"],
                    v6_ball_z_hat=v6["ball_z_hat"],
                )
                print(f"  Saved LOMO fold {mid} ({res['gpu']}, {res['elapsed_s']:.1f}s)", flush=True)
            else:
                ev = dict(res["eval"])
                full_models = pickle.loads(res["models_pkl"])
                with open(out_dir / "pset_v2_full_models.pkl", "wb") as f:
                    pickle.dump(full_models, f)
                print("  Running 100% local inference on BAR-ATH (`atd_raw`, `atd_rep`, `atd_reteam_rep`)...", flush=True)
                ev.update(_predict_atd_locally_pset_v2(full_models))
                j23_dict = {
                    "sample": {
                        1: {
                            "pset_proba": ev["sample_1"]["proba"],
                            "pset_change": ev["sample_1"]["change"],
                            "pset_p4d": ev["sample_1"]["p4d"],
                            "pset_p_aerial": ev["sample_1"]["p_aerial"],
                            "pset_ball_z_hat": ev["sample_1"]["ball_z_hat"],
                        },
                        2: {
                            "pset_proba": ev["sample_2"]["proba"],
                            "pset_change": ev["sample_2"]["change"],
                            "pset_p4d": ev["sample_2"]["p4d"],
                            "pset_p_aerial": ev["sample_2"]["p_aerial"],
                            "pset_ball_z_hat": ev["sample_2"]["ball_z_hat"],
                        },
                        3: {
                            "pset_proba": ev["sample_3"]["proba"],
                            "pset_change": ev["sample_3"]["change"],
                            "pset_p4d": ev["sample_3"]["p4d"],
                            "pset_p_aerial": ev["sample_3"]["p_aerial"],
                            "pset_ball_z_hat": ev["sample_3"]["ball_z_hat"],
                        },
                    },
                    "atd_raw": {
                        "pset_proba": ev["atd_raw"]["proba"],
                        "pset_change": ev["atd_raw"]["change"],
                        "pset_p4d": ev["atd_raw"]["p4d"],
                        "pset_p_aerial": ev["atd_raw"]["p_aerial"],
                        "pset_ball_z_hat": ev["atd_raw"]["ball_z_hat"],
                    },
                    "atd_rep": {
                        "pset_proba": ev["atd_rep"]["proba"],
                        "pset_change": ev["atd_rep"]["change"],
                        "pset_p4d": ev["atd_rep"]["p4d"],
                        "pset_p_aerial": ev["atd_rep"]["p_aerial"],
                        "pset_ball_z_hat": ev["atd_rep"]["ball_z_hat"],
                    },
                    "atd_reteam_rep": {
                        "pset_proba": ev["atd_reteam_rep"]["proba"],
                        "pset_change": ev["atd_reteam_rep"]["change"],
                        "pset_p4d": ev["atd_reteam_rep"]["p4d"],
                        "pset_p_aerial": ev["atd_reteam_rep"]["p_aerial"],
                        "pset_ball_z_hat": ev["atd_reteam_rep"]["ball_z_hat"],
                    },
                }
                with open(out_dir / "pset_v2_j23_preds.pkl", "wb") as f:
                    pickle.dump(j23_dict, f)
                print(f"  Saved Full J2/J3 predictions ({res['gpu']}, {res['elapsed_s']:.1f}s)", flush=True)

    # Stage 3: Benchmark SymPlayerSetNetV2 vs SymPlayerSetNet (v1) locally
    print("\n[Stage 3] Benchmarking SymPlayerSetNetV2 vs SymPlayerSetNet (v1)...", flush=True)
    from sklearn.metrics import roc_auc_score
    from s3_11_tcn_local import without_dead_margin
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.sportec import load_all as load_sportec
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.eval import metrics
    from src.models.transition import change_targets

    sportec = load_sportec()
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    w7a_dir = ROOT / "experiments" / "outputs" / "exp_w7a_oof"
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f:
        pset_v1_j23 = pickle.load(f)
    with open(out_dir / "pset_v2_j23_preds.pkl", "rb") as f:
        pset_v2_j23 = pickle.load(f)

    v1_j1a, v2_j1a = [], []
    v1_j1b, v2_j1b = [], []
    v1_chg_auc, v2_chg_auc = [], []
    v2_aerial_auc_v3, v2_aerial_auc_v6 = [], []

    for mid in SPORTEC_MATCH_IDS:
        m = sportec[mid]
        z1 = np.load(w7a_dir / f"pset_{mid}.npz")
        z2 = np.load(out_dir / f"pset_v2_{mid}.npz")

        pr1_v3 = decode_v4(z1["v3_proba"], m.half, loo_dec_hyb[mid], p_change=z1["v3_change"])
        pr2_v3 = decode_v4(z2["v3_proba"], m.half, loo_dec_hyb[mid], p_change=z2["v3_change"])
        pr1_v6 = decode_v4(z1["v6_proba"], m.half, loo_dec_hyb[mid], p_change=z1["v6_change"])
        pr2_v6 = decode_v4(z2["v6_proba"], m.half, loo_dec_hyb[mid], p_change=z2["v6_change"])

        f1_1_v3 = float(metrics.score(m.labels, pr1_v3, m.half)["spell_f1_mean"])
        f1_2_v3 = float(metrics.score(m.labels, pr2_v3, m.half)["spell_f1_mean"])
        f1_1_v6 = float(metrics.score(m.labels, pr1_v6, m.half)["spell_f1_mean"])
        f1_2_v6 = float(metrics.score(m.labels, pr2_v6, m.half)["spell_f1_mean"])

        v1_j1a.append(f1_1_v3)
        v2_j1a.append(f1_2_v3)
        v1_j1b.append(f1_1_v6)
        v2_j1b.append(f1_2_v6)

        yc = change_targets(m.labels, m.half)
        vc = yc >= 0
        yt = (yc[vc] > 0.5).astype(np.int32)
        auc1 = float(roc_auc_score(yt, z1["v3_change"][vc]))
        auc2 = float(roc_auc_score(yt, z2["v3_change"][vc]))
        v1_chg_auc.append(auc1)
        v2_chg_auc.append(auc2)

        bz = m.meta["ball_z"]
        vbz = (m.half > 0) & np.isfinite(bz)
        yt_air = (bz[vbz] > 1.5).astype(np.int32)
        air_auc_3 = float(roc_auc_score(yt_air, z2["v3_p_aerial"][vbz]))
        air_auc_6 = float(roc_auc_score(yt_air, z2["v6_p_aerial"][vbz]))
        v2_aerial_auc_v3.append(air_auc_3)
        v2_aerial_auc_v6.append(air_auc_6)

        print(
            f"  [{mid}] J1A(v3): v1={f1_1_v3:.4f} -> v2={f1_2_v3:.4f} ({f1_2_v3 - f1_1_v3:+.4f}) | "
            f"J1B(v6): v1={f1_1_v6:.4f} -> v2={f1_2_v6:.4f} ({f1_2_v6 - f1_1_v6:+.4f}) | "
            f"chgAUC: {auc1:.4f}->{auc2:.4f} | AerialAUC(v3/v6): {air_auc_3:.4f}/{air_auc_6:.4f}",
            flush=True,
        )

    v1_j2_raw, v2_j2_raw = [], []
    v1_j2_nod, v2_j2_nod = [], []
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
        "v2_aerial_auc_v3_mean": float(np.mean(v2_aerial_auc_v3)),
        "v2_aerial_auc_v6_mean": float(np.mean(v2_aerial_auc_v6)),
        "total_elapsed_s": round(time.time() - t_start, 2),
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
    run_modal_swarm_and_evaluate()
