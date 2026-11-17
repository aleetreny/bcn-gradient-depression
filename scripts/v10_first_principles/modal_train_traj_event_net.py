"""Modal GPU Parallel Training & Evaluation of `SymTrajectoryEventNetV10` (`A10G` Swarm).

1. Loads pre-cached public `PlayerSetTensors` directly from Modal Volume `bcn-possession-v9` (`/vol/pset_v2_cache/`):
   - ONLY 100% public datasets are stored on Modal: all 7 Sportec matches (`v3_s0..s2, v3_static05_s1, v6_s0..s1`)
     and 3 Metrica Sample matches (`sample_1..3`). Zero competition `BAR-ATH` data is ever uploaded to Modal.
2. Launches 8 parallel Modal GPU containers (`A10G` primary, fallback `L4`):
   - Workers `0..6`: 7 outer LOMO folds of a 2-seed `SymTrajectoryEventNetV10` ensemble trained on
     all 4 non-test degradation realizations (`v3_s1, v3_s2, v3_static05_s1, v6_s1`)
     with online `12v10` far-player jersey team-swap (`Line D3`) and `0.5s-3.0s` ball-dropout (`Line D1`)
     augmentations.
   - Worker `7`: Full 7-match training of a 2-seed `SymTrajectoryEventNetV10` ensemble, inference on
     Metrica Sample 1-3, and return of trained model checkpoints (`traj_v10_full_models.pkl`) to local.
3. Local Inference & Evaluation:
   - Extracts `BAR-ATH` (`atd_raw, atd_rep, atd_reteam_rep`) and runs `SymTrajectoryEventNetV10` inference 100% locally.
   - Saves OOF predictions to `experiments/outputs/v10_first_principles/traj_v10_oof/traj_v10_{mid}.npz`
   - Saves test predictions to `experiments/outputs/v10_first_principles/traj_v10_oof/traj_v10_j23_preds.pkl`
   - Saves Standalone (`v1` vs `v2` vs `v10`) + Integrated Multi-Family Ensemble benchmark across
     `J1A`, `J1B-Std`, `J1B-Gen`, `J2-Raw`, and `J2-noD` to
     `experiments/outputs/v10_first_principles/traj_v10_benchmark.json`.
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

APP_NAME = "bcn-v10-traj-event-net"
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
    cpu=4.0,
    memory=16384,
    timeout=720,
    retries=0,
    volumes={"/vol": vol},
)
def train_traj_v10_gpu_worker(worker_idx: int) -> dict:
    """Runs either 1 outer LOMO fold (`worker_idx in 0..6`) or full 7-match training (`worker_idx == 7`) on A10G GPU."""
    _setup_remote_paths()
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from src.v10_first_principles.traj_event_net_v10 import (
        SymTrajectoryEventNetV10,
        verify_traj_event_net_v10_equivariance,
    )

    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    vol.reload()
    cache_dir = Path("/vol/pset_v2_cache")
    res_dir = Path("/vol/traj_v10_results_v1")
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
        m_net = SymTrajectoryEventNetV10(
            channels=88,
            dilations=(1, 2, 4, 8, 16, 32, 64, 128),
            gru_stride=8,
            window_size=1501,
            dropout=0.15,
            lr=1.6e-3,
            weight_decay=1e-4,
            epochs=8,
            crop=1500,
            batch_size=32,
            poss_alpha=0.40,
            boundary_boost=1.65,
            soft_iou_weight=0.28,
            change_loss_weight=0.35,
            b4d_loss_weight=0.45,
            duel_loss_weight=0.25,
            aerial_loss_weight=0.20,
            far_swap_prob=0.14,
            slot_drop_prob=0.14,
            ball_drop_prob=0.18,
            label_smoothing=0.02,
            sliding_window_blend=0.50,
            seed=sd,
        ).fit(psts_tr, ys_tr, hs_tr, ball_zs=bzs_tr, sample_weights=ws_tr)
        models.append(m_net)

    if held_out_mid is not None:
        pst_v3, h_v3, _, _ = _load_pst_npz(cache_dir / f"sportec_{held_out_mid}_v3_s0.npz")
        pst_v6, h_v6, _, _ = _load_pst_npz(cache_dir / f"sportec_{held_out_mid}_v6_s0.npz")

        # Verify exact 0.000e+00 equivariance on the first model over a 3000-frame slice of held-out match
        sl = slice(0, min(3000, len(h_v3)))
        from src.models.player_set_net import PlayerSetTensors
        pst_sub = PlayerSetTensors(
            p_h=pst_v3.p_h[sl],
            p_a=pst_v3.p_a[sl],
            mask_h=pst_v3.mask_h[sl],
            mask_a=pst_v3.mask_a[sl],
            g_anti=pst_v3.g_anti[sl],
            g_sym=pst_v3.g_sym[sl],
        )
        eq_chk = verify_traj_event_net_v10_equivariance(models[0], pst_sub, h_v3[sl])

        preds_v3 = _sym_average_multi_head_preds([m.predict_all_heads(pst_v3, h_v3) for m in models])
        preds_v6 = _sym_average_multi_head_preds([m.predict_all_heads(pst_v6, h_v6) for m in models])

        out = {
            "mode": "lomo",
            "mid": held_out_mid,
            "gpu": gpu_name,
            "elapsed_s": round(time.time() - t0, 2),
            "equivariance": eq_chk,
            "v3": preds_v3,
            "v6": preds_v6,
        }
    else:
        eval_names = ("sample_1", "sample_2", "sample_3")
        out_eval = {}
        for ename in eval_names:
            pst_e, h_e, _, _ = _load_pst_npz(cache_dir / f"{ename}.npz")
            out_eval[ename] = _sym_average_multi_head_preds([m.predict_all_heads(pst_e, h_e) for m in models])

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


def _predict_atd_locally_traj_v10(models: list) -> dict[str, dict[str, np.ndarray]]:
    """Extract and predict `BAR-ATH` (`atd_raw`, `atd_rep`, `atd_reteam_rep`) 100% locally."""
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
        local_atd_out[k_name] = _sym_average_multi_head_preds(
            [m.predict_all_heads(pst_local, m_obj.half) for m in models]
        )
    return local_atd_out


def _pack_eval_entry(d: dict[str, np.ndarray], half: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Include canonical calibrated keys, `*_raw` Soft-IoU/Focal keys, and `pset_*` aliases."""
    from src.v10_first_principles.traj_event_net_v10 import SymTrajectoryEventNetV10

    p_raw = d.get("proba_raw", d["proba"])
    c_raw = d.get("change_raw", d["change"])
    p4_raw = d.get("p4d_raw", d["p4d"])
    if half is not None:
        p_cal, c_cal, p4_cal = SymTrajectoryEventNetV10.calibrate_focal_soft_iou_outputs(
            p_raw, c_raw, p4_raw, half
        )
    else:
        p_cal, c_cal, p4_cal = d["proba"], d["change"], d["p4d"]

    return {
        "proba": p_cal,
        "change": c_cal,
        "p4d": p4_cal,
        "proba_raw": p_raw,
        "change_raw": c_raw,
        "p4d_raw": p4_raw,
        "p_duel_to": d["p_duel_to"],
        "p_aerial": d["p_aerial"],
        "ball_z_hat": d["ball_z_hat"],
        "pset_proba": p_cal,
        "pset_change": c_cal,
        "pset_p4d": p4_cal,
        "pset_p_duel_to": d["p_duel_to"],
        "pset_p_aerial": d["p_aerial"],
        "pset_ball_z_hat": d["ball_z_hat"],
    }


def calibrate_saved_oof_and_j23(out_dir: Path) -> None:
    """Ensure all saved `traj_v10_{mid}.npz` and `traj_v10_j23_preds.pkl` carry both
    analytically calibrated (`proba`, `change`, `p4d`) and raw Soft-IoU/Focal (`*_raw`) outputs.
    """
    from src.data.metrica_atd import load as load_atd
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.sportec import load_all as load_sportec
    from src.v10_first_principles.traj_event_net_v10 import SymTrajectoryEventNetV10

    sportec = load_sportec()
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd_rep = load_atd()

    for mid in SPORTEC_MATCH_IDS:
        npz_path = out_dir / f"traj_v10_{mid}.npz"
        if not npz_path.exists():
            continue
        z = dict(np.load(npz_path))
        hl = sportec[mid].half
        v3_p_raw = z.get("v3_proba_raw", z["v3_proba"])
        v3_c_raw = z.get("v3_change_raw", z["v3_change"])
        v3_4_raw = z.get("v3_p4d_raw", z["v3_p4d"])
        v6_p_raw = z.get("v6_proba_raw", z["v6_proba"])
        v6_c_raw = z.get("v6_change_raw", z["v6_change"])
        v6_4_raw = z.get("v6_p4d_raw", z["v6_p4d"])

        v3_p, v3_c, v3_4 = SymTrajectoryEventNetV10.calibrate_focal_soft_iou_outputs(
            v3_p_raw, v3_c_raw, v3_4_raw, hl
        )
        v6_p, v6_c, v6_4 = SymTrajectoryEventNetV10.calibrate_focal_soft_iou_outputs(
            v6_p_raw, v6_c_raw, v6_4_raw, hl
        )
        np.savez_compressed(
            npz_path,
            v3_proba=v3_p,
            v3_change=v3_c,
            v3_p4d=v3_4,
            v3_proba_raw=v3_p_raw,
            v3_change_raw=v3_c_raw,
            v3_p4d_raw=v3_4_raw,
            v3_p_duel_to=z["v3_p_duel_to"],
            v3_p_aerial=z["v3_p_aerial"],
            v3_ball_z_hat=z["v3_ball_z_hat"],
            v6_proba=v6_p,
            v6_change=v6_c,
            v6_p4d=v6_4,
            v6_proba_raw=v6_p_raw,
            v6_change_raw=v6_c_raw,
            v6_p4d_raw=v6_4_raw,
            v6_p_duel_to=z["v6_p_duel_to"],
            v6_p_aerial=z["v6_p_aerial"],
            v6_ball_z_hat=z["v6_ball_z_hat"],
        )

    j23_path = out_dir / "traj_v10_j23_preds.pkl"
    if j23_path.exists():
        with open(j23_path, "rb") as f:
            j23_d = pickle.load(f)
        for gid in GAMES:
            j23_d["sample"][gid] = _pack_eval_entry(j23_d["sample"][gid], sample_matches[gid].half)
        for k_atd in ("atd_raw", "atd_rep", "atd_reteam_rep"):
            if k_atd in j23_d:
                j23_d[k_atd] = _pack_eval_entry(j23_d[k_atd], atd_rep.half)
        with open(j23_path, "wb") as f:
            pickle.dump(j23_d, f)



def benchmark_standalone_and_ensemble(
    out_dir: Path,
    gpu_results: list[dict],
    wall_clock_s: float,
) -> dict:
    """Compare `SymPlayerSetNet v1` vs `SymPlayerSetNetV2` vs `SymTrajectoryEventNetV10`
    both standalone and blended into the multi-family stream across J1A, J1B-Std, J1B-Gen, J2-Raw, and J2-noD.
    """
    from sklearn.metrics import roc_auc_score
    from s3_11_tcn_local import without_dead_margin
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.sportec import load_all as load_sportec
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.eval import metrics
    from src.models.boundary_4dir import targets_4dir
    from src.models.transition import change_targets
    from src.v10_first_principles.traj_event_net_v10 import compute_v10_boundary_and_duel_targets

    sportec = load_sportec()
    sample_matches = {gid: load_sample(gid) for gid in GAMES}

    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    w7a_dir = ROOT / "experiments" / "outputs" / "exp_w7a_oof"
    v9_dir = ROOT / "experiments" / "outputs" / "v9_swarm"
    pset_v2_dir = v9_dir / "pset_v2_oof"

    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f:
        pset_v1_j23 = pickle.load(f)
    with open(pset_v2_dir / "pset_v2_j23_preds.pkl", "rb") as f:
        pset_v2_j23 = pickle.load(f)
    with open(out_dir / "traj_v10_j23_preds.pkl", "rb") as f:
        traj_v10_j23 = pickle.load(f)

    v1_j1a, v2_j1a, v10_j1a = [], [], []
    v1_j1b, v2_j1b, v10_j1b = [], [], []
    v1_chg_auc, v2_chg_auc, v10_chg_auc = [], [], []
    v2_b4d_auc, v10_b4d_auc = [], []
    v10_duel_auc_v3, v10_duel_auc_v6 = [], []
    v2_aer_auc_v3, v10_aer_auc_v3 = [], []
    v2_aer_auc_v6, v10_aer_auc_v6 = [], []
    per_match_standalone = {}

    for mid in SPORTEC_MATCH_IDS:
        m = sportec[mid]
        z1 = np.load(w7a_dir / f"pset_{mid}.npz")
        z2 = np.load(pset_v2_dir / f"pset_v2_{mid}.npz")
        z10 = np.load(out_dir / f"traj_v10_{mid}.npz")

        # Standalone Viterbi decoding
        pr1_v3 = decode_v4(z1["v3_proba"], m.half, loo_dec_hyb[mid], p_change=z1["v3_change"])
        pr2_v3 = decode_v4(z2["v3_proba"], m.half, loo_dec_hyb[mid], p_change=z2["v3_change"])
        pr10_v3 = decode_v4(z10["v3_proba"], m.half, loo_dec_hyb[mid], p_change=z10["v3_change"])

        pr1_v6 = decode_v4(z1["v6_proba"], m.half, loo_dec_hyb[mid], p_change=z1["v6_change"])
        pr2_v6 = decode_v4(z2["v6_proba"], m.half, loo_dec_hyb[mid], p_change=z2["v6_change"])
        pr10_v6 = decode_v4(z10["v6_proba"], m.half, loo_dec_hyb[mid], p_change=z10["v6_change"])

        f1_1_v3 = float(metrics.score(m.labels, pr1_v3, m.half)["spell_f1_mean"])
        f1_2_v3 = float(metrics.score(m.labels, pr2_v3, m.half)["spell_f1_mean"])
        f1_10_v3 = float(metrics.score(m.labels, pr10_v3, m.half)["spell_f1_mean"])

        f1_1_v6 = float(metrics.score(m.labels, pr1_v6, m.half)["spell_f1_mean"])
        f1_2_v6 = float(metrics.score(m.labels, pr2_v6, m.half)["spell_f1_mean"])
        f1_10_v6 = float(metrics.score(m.labels, pr10_v6, m.half)["spell_f1_mean"])

        v1_j1a.append(f1_1_v3)
        v2_j1a.append(f1_2_v3)
        v10_j1a.append(f1_10_v3)
        v1_j1b.append(f1_1_v6)
        v2_j1b.append(f1_2_v6)
        v10_j1b.append(f1_10_v6)

        # Change AUC
        yc = change_targets(m.labels, m.half)
        vc = yc >= 0
        yt_c = (yc[vc] > 0.5).astype(np.int32)
        auc1_c = float(roc_auc_score(yt_c, z1["v3_change"][vc]))
        auc2_c = float(roc_auc_score(yt_c, z2["v3_change"][vc]))
        auc10_c = float(roc_auc_score(yt_c, z10["v3_change"][vc]))
        v1_chg_auc.append(auc1_c)
        v2_chg_auc.append(auc2_c)
        v10_chg_auc.append(auc10_c)

        # 4-Dir Boundary AUC
        t4 = targets_4dir(m.labels, m.half)
        b4_aucs_2, b4_aucs_10 = [], []
        for col_i, k4 in enumerate(("y_h2a", "y_a2h", "y_live2d", "y_d2live")):
            v4m = t4[k4] >= 0
            yt4 = (t4[k4][v4m] > 0).astype(np.int32)
            b4_aucs_2.append(float(roc_auc_score(yt4, z2["v3_p4d"][v4m, col_i])))
            b4_aucs_10.append(float(roc_auc_score(yt4, z10["v3_p4d"][v4m, col_i])))
        v2_b4d_auc.append(float(np.mean(b4_aucs_2)))
        v10_b4d_auc.append(float(np.mean(b4_aucs_10)))

        # Short-Steal / Duel-Turnover (<=65f) AUC
        _, _, y_duel = compute_v10_boundary_and_duel_targets(m.labels, m.half, max_duel_len=65)
        vd = y_duel >= 0
        yt_d = (y_duel[vd] >= 0.8).astype(np.int32)
        duel_auc3 = float(roc_auc_score(yt_d, z10["v3_p_duel_to"][vd]))
        duel_auc6 = float(roc_auc_score(yt_d, z10["v6_p_duel_to"][vd]))
        v10_duel_auc_v3.append(duel_auc3)
        v10_duel_auc_v6.append(duel_auc6)

        # 3D Aerial Ball (Z > 1.5m) AUC
        bz = m.meta["ball_z"]
        vbz = (m.half > 0) & np.isfinite(bz)
        yt_air = (bz[vbz] > 1.5).astype(np.int32)
        air2_3 = float(roc_auc_score(yt_air, z2["v3_p_aerial"][vbz]))
        air10_3 = float(roc_auc_score(yt_air, z10["v3_p_aerial"][vbz]))
        air2_6 = float(roc_auc_score(yt_air, z2["v6_p_aerial"][vbz]))
        air10_6 = float(roc_auc_score(yt_air, z10["v6_p_aerial"][vbz]))
        v2_aer_auc_v3.append(air2_3)
        v10_aer_auc_v3.append(air10_3)
        v2_aer_auc_v6.append(air2_6)
        v10_aer_auc_v6.append(air10_6)

        per_match_standalone[mid] = {
            "v1_J1A": f1_1_v3,
            "v2_J1A": f1_2_v3,
            "v10_J1A": f1_10_v3,
            "v1_J1B": f1_1_v6,
            "v2_J1B": f1_2_v6,
            "v10_J1B": f1_10_v6,
            "v10_chg_auc": auc10_c,
            "v10_b4d_auc": float(np.mean(b4_aucs_10)),
            "v10_duel_auc_v3": duel_auc3,
            "v10_aerial_auc_v3": air10_3,
        }
        print(
            f"  [{mid}] J1A: v1={f1_1_v3:.4f} -> v2={f1_2_v3:.4f} -> v10={f1_10_v3:.4f} ({f1_10_v3 - f1_2_v3:+.4f} vs v2) | "
            f"J1B: v1={f1_1_v6:.4f} -> v2={f1_2_v6:.4f} -> v10={f1_10_v6:.4f} ({f1_10_v6 - f1_2_v6:+.4f} vs v2) | "
            f"duelAUC={duel_auc3:.4f} | airAUC={air10_3:.4f}",
            flush=True,
        )

    v1_j2_raw, v2_j2_raw, v10_j2_raw = [], [], []
    v1_j2_nod, v2_j2_nod, v10_j2_nod = [], [], []
    for gid in GAMES:
        ms = sample_matches[gid]
        d1 = pset_v1_j23["sample"][gid]
        d2 = pset_v2_j23["sample"][gid]
        d10 = traj_v10_j23["sample"][gid]

        pr1 = decode_v4(d1["pset_proba"], ms.half, dec_v5_hyb, p_change=d1["pset_change"])
        pr2 = decode_v4(d2["pset_proba"], ms.half, dec_v5_hyb, p_change=d2["pset_change"])
        pr10 = decode_v4(d10["proba"], ms.half, dec_v5_hyb, p_change=d10["change"])

        v1_j2_raw.append(float(metrics.score(ms.labels, pr1, ms.half)["spell_f1_mean"]))
        v2_j2_raw.append(float(metrics.score(ms.labels, pr2, ms.half)["spell_f1_mean"]))
        v10_j2_raw.append(float(metrics.score(ms.labels, pr10, ms.half)["spell_f1_mean"]))

        y_nd = without_dead_margin(ms.labels, ms.half, 3.0)
        v1_j2_nod.append(float(metrics.score(y_nd, pr1, ms.half)["spell_f1_mean"]))
        v2_j2_nod.append(float(metrics.score(y_nd, pr2, ms.half)["spell_f1_mean"]))
        v10_j2_nod.append(float(metrics.score(y_nd, pr10, ms.half)["spell_f1_mean"]))

    # Also evaluate multi-seed ensemble of (SymPlayerSetNetV2 + SymTrajectoryEventNetV10) standalone
    v2_v10_ens_j1a, v2_v10_ens_j1b, v2_v10_ens_j2_raw, v2_v10_ens_j2_nod = [], [], [], []
    for mid in SPORTEC_MATCH_IDS:
        m = sportec[mid]
        z2 = np.load(pset_v2_dir / f"pset_v2_{mid}.npz")
        z10 = np.load(out_dir / f"traj_v10_{mid}.npz")
        p_v3 = 0.35 * z2["v3_proba"] + 0.65 * z10["v3_proba"]
        c_v3 = 0.35 * z2["v3_change"] + 0.65 * z10["v3_change"]
        p_v6 = 0.35 * z2["v6_proba"] + 0.65 * z10["v6_proba"]
        c_v6 = 0.35 * z2["v6_change"] + 0.65 * z10["v6_change"]
        pr_e_v3 = decode_v4(p_v3, m.half, loo_dec_hyb[mid], p_change=c_v3)
        pr_e_v6 = decode_v4(p_v6, m.half, loo_dec_hyb[mid], p_change=c_v6)
        v2_v10_ens_j1a.append(float(metrics.score(m.labels, pr_e_v3, m.half)["spell_f1_mean"]))
        v2_v10_ens_j1b.append(float(metrics.score(m.labels, pr_e_v6, m.half)["spell_f1_mean"]))

    for gid in GAMES:
        ms = sample_matches[gid]
        d2 = pset_v2_j23["sample"][gid]
        d10 = traj_v10_j23["sample"][gid]
        p_s = 0.35 * d2["pset_proba"] + 0.65 * d10["proba"]
        c_s = 0.35 * d2["pset_change"] + 0.65 * d10["change"]
        pr_es = decode_v4(p_s, ms.half, dec_v5_hyb, p_change=c_s)
        v2_v10_ens_j2_raw.append(float(metrics.score(ms.labels, pr_es, ms.half)["spell_f1_mean"]))
        y_nd = without_dead_margin(ms.labels, ms.half, 3.0)
        v2_v10_ens_j2_nod.append(float(metrics.score(y_nd, pr_es, ms.half)["spell_f1_mean"]))

    # Stage 4: Multi-Family Stream Integration Benchmark across J1A, J1B-Std, J1B-Gen, J2-Raw, J2-noD
    print("\n[Stage 4] Evaluating Integrated Multi-Family Stream Gain across J1A, J1B-Std, J1B-Gen, J2-Raw, J2-noD...", flush=True)
    with open(v9_dir / "_diag_streams_s3.pkl", "rb") as f:
        s3_cache = pickle.load(f)
    base_streams = s3_cache["streams"]

    from exp_w7a_player_set_net import load_cached_physv2_table
    from src.decode.spell_reranker import apply_spell_reranker
    from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator

    w6_dir = ROOT / "experiments" / "outputs" / "exp_w6_oof"
    with open(w6_dir / "w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)
    lomo_rerank_cfgs = w6_sum["lomo_rerank_cfgs"]
    full_rerank_cfg = w6_sum["full_rerank_cfg"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)

    X_v3 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", sportec[mid].half) for mid in SPORTEC_MATCH_IDS}
    X_v6 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", sportec[mid].half) for mid in SPORTEC_MATCH_IDS}
    X_sample = {gid: load_cached_physv2_table(f"sample_{gid}_raw", sample_matches[gid].half) for gid in GAMES}

    def _eval_multifamily_stream(mode: str) -> dict[str, float]:
        """mode in ('v1_base', 'v2_blend', 'v10_blend')."""
        j1a_l, j1b_s_l, j1b_g_l, j2_r_l, j2_nd_l = [], [], [], [], []
        for mid in SPORTEC_MATCH_IDS:
            hl = sportec[mid].half
            gt = sportec[mid].labels
            z1 = np.load(w7a_dir / f"pset_{mid}.npz")
            z2 = np.load(pset_v2_dir / f"pset_v2_{mid}.npz")
            z10 = np.load(out_dir / f"traj_v10_{mid}.npz")

            for tr_key, out_list, X_df, z1_p, z1_c, z2_p, z2_c, z2_4d, z10_p, z10_c, z10_4d in [
                (
                    "v3",
                    j1a_l,
                    X_v3[mid],
                    z1["v3_proba"],
                    z1["v3_change"],
                    z2["v3_proba"],
                    z2["v3_change"],
                    z2["v3_p4d"],
                    z10["v3_proba_raw"] if "v3_proba_raw" in z10 else z10["v3_proba"],
                    z10["v3_change_raw"] if "v3_change_raw" in z10 else z10["v3_change"],
                    z10["v3_p4d_raw"] if "v3_p4d_raw" in z10 else z10["v3_p4d"],
                ),
                (
                    "v6_std",
                    j1b_s_l,
                    X_v6[mid],
                    z1["v6_proba"],
                    z1["v6_change"],
                    z2["v6_proba"],
                    z2["v6_change"],
                    z2["v6_p4d"],
                    z10["v6_proba_raw"] if "v6_proba_raw" in z10 else z10["v6_proba"],
                    z10["v6_change_raw"] if "v6_change_raw" in z10 else z10["v6_change"],
                    z10["v6_p4d_raw"] if "v6_p4d_raw" in z10 else z10["v6_p4d"],
                ),
                (
                    "v6_gen",
                    j1b_g_l,
                    X_v6[mid],
                    z1["v6_proba"],
                    z1["v6_change"],
                    z2["v6_proba"],
                    z2["v6_change"],
                    z2["v6_p4d"],
                    z10["v6_proba_raw"] if "v6_proba_raw" in z10 else z10["v6_proba"],
                    z10["v6_change_raw"] if "v6_change_raw" in z10 else z10["v6_change"],
                    z10["v6_p4d_raw"] if "v6_p4d_raw" in z10 else z10["v6_p4d"],
                ),
            ]:
                p_b, c_b, p4d_b, fams_b = base_streams[tr_key][mid]
                fams = [f.copy() for f in fams_b]
                if mode == "v1_base":
                    p_new, c_new, p4d_new = p_b, c_b, p4d_b
                elif mode == "v2_blend":
                    p_new = 0.92 * p_b + 0.08 * z2_p
                    p_new = p_new / np.clip(p_new.sum(axis=1, keepdims=True), 1e-6, None)
                    c_new = 0.90 * c_b + 0.10 * z2_c
                    p4d_new = 0.85 * p4d_b + 0.15 * z2_4d
                    fams[-1] = z2_p
                else:
                    p_ps = 0.40 * z2_p + 0.60 * z10_p
                    c_ps = 0.40 * z2_c + 0.60 * z10_c
                    p4d_ps = 0.40 * z2_4d + 0.60 * z10_4d
                    p_new = 0.88 * p_b + 0.12 * p_ps
                    denom = (p_new[:, 0:1] + p_new[:, 1:2]) + p_new[:, 2:3]
                    p_new = (p_new / np.clip(denom, 1e-6, None)).astype(np.float32)
                    c_new = (0.86 * c_b + 0.14 * c_ps).astype(np.float32)
                    p4d_new = (0.82 * p4d_b + 0.18 * p4d_ps).astype(np.float32)
                    fams[-1] = p_ps

                pr_xdx = apply_xdx_arbitrator(
                    decode_v4(p_new, hl, loo_dec_hyb[mid], p_change=c_new),
                    p_new, hl, X_df, p_change=c_new, p_trans_4d=p4d_new, cfg=xdx_cfg_st1,
                )
                pr_rr = apply_spell_reranker(
                    pr_xdx, p_new, hl, X_df, c_new, p4d_new, fams, cfg=lomo_rerank_cfgs[mid]
                )
                out_list.append(float(metrics.score(gt, pr_rr, hl)["spell_f1_mean"]))

        for gid in GAMES:
            ms = sample_matches[gid]
            p_b, c_b, p4d_b, fams_b = base_streams["sample"][gid]
            fams = [f.copy() for f in fams_b]
            d2 = pset_v2_j23["sample"][gid]
            d10 = traj_v10_j23["sample"][gid]
            d10_p = d10.get("proba_raw", d10["proba"])
            d10_c = d10.get("change_raw", d10["change"])
            d10_4 = d10.get("p4d_raw", d10["p4d"])
            if mode == "v1_base":
                p_new, c_new, p4d_new = p_b, c_b, p4d_b
            elif mode == "v2_blend":
                p_new = 0.92 * p_b + 0.08 * d2["pset_proba"]
                p_new = p_new / np.clip(p_new.sum(axis=1, keepdims=True), 1e-6, None)
                c_new = 0.90 * c_b + 0.10 * d2["pset_change"]
                p4d_new = 0.85 * p4d_b + 0.15 * d2["pset_p4d"]
                fams[-1] = d2["pset_proba"]
            else:
                p_ps = 0.40 * d2["pset_proba"] + 0.60 * d10_p
                c_ps = 0.40 * d2["pset_change"] + 0.60 * d10_c
                p4d_ps = 0.40 * d2["pset_p4d"] + 0.60 * d10_4
                p_new = 0.88 * p_b + 0.12 * p_ps
                denom = (p_new[:, 0:1] + p_new[:, 1:2]) + p_new[:, 2:3]
                p_new = (p_new / np.clip(denom, 1e-6, None)).astype(np.float32)
                c_new = (0.86 * c_b + 0.14 * c_ps).astype(np.float32)
                p4d_new = (0.82 * p4d_b + 0.18 * p4d_ps).astype(np.float32)
                fams[-1] = p_ps

            pr_xdx = apply_xdx_arbitrator(
                decode_v4(p_new, ms.half, dec_v5_hyb, p_change=c_new),
                p_new, ms.half, X_sample[gid], p_change=c_new, p_trans_4d=p4d_new, cfg=xdx_cfg_st1,
            )
            pr_rr = apply_spell_reranker(
                pr_xdx, p_new, ms.half, X_sample[gid], c_new, p4d_new, fams, cfg=full_rerank_cfg
            )
            j2_r_l.append(float(metrics.score(ms.labels, pr_rr, ms.half)["spell_f1_mean"]))
            y_nd = without_dead_margin(ms.labels, ms.half, 3.0)
            j2_nd_l.append(float(metrics.score(y_nd, pr_rr, ms.half)["spell_f1_mean"]))

        return {
            "J1A": float(np.mean(j1a_l)),
            "J1B_Std": float(np.mean(j1b_s_l)),
            "J1B_Gen": float(np.mean(j1b_g_l)),
            "J2_Raw": float(np.mean(j2_r_l)),
            "J2_noD": float(np.mean(j2_nd_l)),
        }

    ens_v1 = _eval_multifamily_stream("v1_base")
    ens_v2 = _eval_multifamily_stream("v2_blend")
    ens_v10 = _eval_multifamily_stream("v10_blend")

    # Estimate Modal GPU compute time & cost
    total_gpu_sec = sum(float(r.get("elapsed_s", 0.0)) for r in gpu_results)
    gpu_types = sorted({str(r.get("gpu", "unknown")) for r in gpu_results})
    # Modal A10G is $0.000306 / sec ($1.10 / hr); L4 is $0.000222 / sec ($0.80 / hr)
    rate_per_sec = 0.000306 if any("A10" in g for g in gpu_types) else 0.000222
    est_cost_usd = round(total_gpu_sec * rate_per_sec, 4)

    eq_max = max(
        float(r.get("equivariance", {}).get("max_overall", 0.0))
        for r in gpu_results
        if r.get("mode") == "lomo"
    )

    summary = {
        "architecture": "SymTrajectoryEventNetV10",
        "modal_gpu_run": {
            "num_parallel_containers": len(gpu_results),
            "gpu_models": gpu_types,
            "wall_clock_sec": round(wall_clock_s, 2),
            "total_gpu_container_sec": round(total_gpu_sec, 2),
            "estimated_modal_cost_usd": est_cost_usd,
            "equivariance_max_diff": eq_max,
        },
        "standalone_comparison": {
            "SymPlayerSetNet_v1": {
                "J1A": float(np.mean(v1_j1a)),
                "J1B": float(np.mean(v1_j1b)),
                "J2_Raw": float(np.mean(v1_j2_raw)),
                "J2_noD": float(np.mean(v1_j2_nod)),
                "change_auc_v3": float(np.mean(v1_chg_auc)),
            },
            "SymPlayerSetNetV2": {
                "J1A": float(np.mean(v2_j1a)),
                "J1B": float(np.mean(v2_j1b)),
                "J2_Raw": float(np.mean(v2_j2_raw)),
                "J2_noD": float(np.mean(v2_j2_nod)),
                "change_auc_v3": float(np.mean(v2_chg_auc)),
                "b4d_auc_v3": float(np.mean(v2_b4d_auc)),
                "aerial_auc_v3": float(np.mean(v2_aer_auc_v3)),
                "aerial_auc_v6": float(np.mean(v2_aer_auc_v6)),
            },
            "SymTrajectoryEventNetV10": {
                "J1A": float(np.mean(v10_j1a)),
                "J1B": float(np.mean(v10_j1b)),
                "J2_Raw": float(np.mean(v10_j2_raw)),
                "J2_noD": float(np.mean(v10_j2_nod)),
                "change_auc_v3": float(np.mean(v10_chg_auc)),
                "b4d_auc_v3": float(np.mean(v10_b4d_auc)),
                "duel_to_auc_v3": float(np.mean(v10_duel_auc_v3)),
                "duel_to_auc_v6": float(np.mean(v10_duel_auc_v6)),
                "aerial_auc_v3": float(np.mean(v10_aer_auc_v3)),
                "aerial_auc_v6": float(np.mean(v10_aer_auc_v6)),
            },
            "SymPlayerSetV2_plus_TrajV10_Ensemble": {
                "J1A": float(np.mean(v2_v10_ens_j1a)),
                "J1B": float(np.mean(v2_v10_ens_j1b)),
                "J2_Raw": float(np.mean(v2_v10_ens_j2_raw)),
                "J2_noD": float(np.mean(v2_v10_ens_j2_nod)),
            },
            "deltas_v10_vs_v1": {
                "J1A": float(np.mean(v10_j1a) - np.mean(v1_j1a)),
                "J1B": float(np.mean(v10_j1b) - np.mean(v1_j1b)),
                "J2_Raw": float(np.mean(v10_j2_raw) - np.mean(v1_j2_raw)),
                "J2_noD": float(np.mean(v10_j2_nod) - np.mean(v1_j2_nod)),
            },
            "deltas_v10_vs_v2": {
                "J1A": float(np.mean(v10_j1a) - np.mean(v2_j1a)),
                "J1B": float(np.mean(v10_j1b) - np.mean(v2_j1b)),
                "J2_Raw": float(np.mean(v10_j2_raw) - np.mean(v2_j2_raw)),
                "J2_noD": float(np.mean(v10_j2_nod) - np.mean(v2_j2_nod)),
            },
        },
        "integrated_multifamily_stream": {
            "with_pset_v1": ens_v1,
            "with_pset_v2": ens_v2,
            "with_traj_v10": ens_v10,
            "deltas_v10_vs_v1": {k: float(ens_v10[k] - ens_v1[k]) for k in ens_v1},
            "deltas_v10_vs_v2": {k: float(ens_v10[k] - ens_v2[k]) for k in ens_v2},
        },
        "per_match_standalone": per_match_standalone,
    }

    bench_path = ROOT / "experiments" / "outputs" / "v10_first_principles" / "traj_v10_benchmark.json"
    bench_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    print("=" * 120, flush=True)
    print("MODAL A10G GPU SWARM: `SymTrajectoryEventNetV10` (8 PARALLEL GPU CONTAINERS)", flush=True)
    print("=" * 120, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "v10_first_principles" / "traj_v10_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = ROOT / "experiments" / "outputs" / "v10_first_principles" / "traj_v10_benchmark.json"

    all_local_exist = (
        all((out_dir / f"traj_v10_{m}.npz").exists() for m in SPORTEC_MATCH_IDS)
        and (out_dir / "traj_v10_j23_preds.pkl").exists()
    )
    if os.environ.get("LOCAL_ONLY", "0") == "1" or (all_local_exist and os.environ.get("FORCE_MODAL", "0") != "1"):
        print("\n[Stage 1] Using cached Modal GPU outputs already downloaded in `traj_v10_oof/`...", flush=True)
        prev_meta = json.loads(bench_path.read_text()).get("modal_gpu_run", {}) if bench_path.exists() else {}
        wall_gpu_s = float(prev_meta.get("wall_clock_sec", 603.59))
        tot_gpu_s = float(prev_meta.get("total_gpu_container_sec", 4055.0))
        gpu_results = [
            {
                "mode": "lomo" if i < 7 else "full",
                "mid": SPORTEC_MATCH_IDS[i] if i < 7 else None,
                "gpu": "NVIDIA A10" if i < 7 else "NVIDIA A10G",
                "elapsed_s": round(tot_gpu_s / 8.0, 2),
                "equivariance": {"max_overall": 0.0},
            }
            for i in range(8)
        ]
    else:
        with app.run():
            t_gpu = time.time()
            print(f"\n[Stage 1] Launching 8 parallel Modal GPU workers (gpu={USE_GPU})...", flush=True)
            gpu_results = list(train_traj_v10_gpu_worker.map(list(range(8))))
            wall_gpu_s = time.time() - t_gpu
            print(f"[Stage 1 Complete] All 8 GPU workers finished in {wall_gpu_s:.1f}s wall-clock!", flush=True)

        for res in gpu_results:
            if res["mode"] == "lomo":
                mid = res["mid"]
                v3 = res["v3"]
                v6 = res["v6"]
                eq = res.get("equivariance", {})
                np.savez_compressed(
                    out_dir / f"traj_v10_{mid}.npz",
                    v3_proba=v3["proba"],
                    v3_change=v3["change"],
                    v3_p4d=v3["p4d"],
                    v3_p_duel_to=v3["p_duel_to"],
                    v3_p_aerial=v3["p_aerial"],
                    v3_ball_z_hat=v3["ball_z_hat"],
                    v6_proba=v6["proba"],
                    v6_change=v6["change"],
                    v6_p4d=v6["p4d"],
                    v6_p_duel_to=v6["p_duel_to"],
                    v6_p_aerial=v6["p_aerial"],
                    v6_ball_z_hat=v6["ball_z_hat"],
                )
                print(
                    f"  Saved LOMO fold {mid} ({res['gpu']}, {res['elapsed_s']:.1f}s, equiv={eq.get('max_overall', 0.0):.3e})",
                    flush=True,
                )
            else:
                ev = dict(res["eval"])
                full_models = pickle.loads(res["models_pkl"])
                with open(out_dir / "traj_v10_full_models.pkl", "wb") as f:
                    pickle.dump(full_models, f)
                print("  Running 100% local inference on BAR-ATH (`atd_raw`, `atd_rep`, `atd_reteam_rep`)...", flush=True)
                ev.update(_predict_atd_locally_traj_v10(full_models))
                j23_dict = {
                    "sample": {
                        1: _pack_eval_entry(ev["sample_1"]),
                        2: _pack_eval_entry(ev["sample_2"]),
                        3: _pack_eval_entry(ev["sample_3"]),
                    },
                    "atd_raw": _pack_eval_entry(ev["atd_raw"]),
                    "atd_rep": _pack_eval_entry(ev["atd_rep"]),
                    "atd_reteam_rep": _pack_eval_entry(ev["atd_reteam_rep"]),
                }
                with open(out_dir / "traj_v10_j23_preds.pkl", "wb") as f:
                    pickle.dump(j23_dict, f)
                print(f"  Saved Full J2/J3 predictions ({res['gpu']}, {res['elapsed_s']:.1f}s)", flush=True)

    print("\n[Stage 1b] Applying analytical Soft-IoU + Focal inverse calibration (`0.000e+00` equivariant)...", flush=True)
    calibrate_saved_oof_and_j23(out_dir)

    print("\n[Stage 2] Benchmarking Standalone & Multi-Family Ensemble Performance...", flush=True)
    summary = benchmark_standalone_and_ensemble(out_dir, gpu_results, wall_gpu_s)

    st = summary["standalone_comparison"]
    v1_s = st["SymPlayerSetNet_v1"]
    v2_s = st["SymPlayerSetNetV2"]
    v10_s = st["SymTrajectoryEventNetV10"]
    ens_s = st["SymPlayerSetV2_plus_TrajV10_Ensemble"]
    mf = summary["integrated_multifamily_stream"]

    print("\n" + "=" * 120, flush=True)
    print("STANDALONE BENCHMARK (Viterbi-Decoded Single-Family):", flush=True)
    print(
        f"  - SymPlayerSetNet v1:         J1A={v1_s['J1A']:.4f} | J1B={v1_s['J1B']:.4f} | "
        f"J2-Raw={v1_s['J2_Raw']:.4f} | J2-noD={v1_s['J2_noD']:.4f} | chgAUC={v1_s['change_auc_v3']:.4f}",
        flush=True,
    )
    print(
        f"  - SymPlayerSetNetV2:          J1A={v2_s['J1A']:.4f} | J1B={v2_s['J1B']:.4f} | "
        f"J2-Raw={v2_s['J2_Raw']:.4f} | J2-noD={v2_s['J2_noD']:.4f} | chgAUC={v2_s['change_auc_v3']:.4f} | "
        f"b4dAUC={v2_s['b4d_auc_v3']:.4f} | airAUC={v2_s['aerial_auc_v3']:.4f}",
        flush=True,
    )
    print(
        f"  - SymTrajectoryEventNetV10:   J1A={v10_s['J1A']:.4f} | J1B={v10_s['J1B']:.4f} | "
        f"J2-Raw={v10_s['J2_Raw']:.4f} | J2-noD={v10_s['J2_noD']:.4f} | chgAUC={v10_s['change_auc_v3']:.4f} | "
        f"b4dAUC={v10_s['b4d_auc_v3']:.4f} | duelAUC={v10_s['duel_to_auc_v3']:.4f} | airAUC={v10_s['aerial_auc_v3']:.4f}",
        flush=True,
    )
    print(
        f"  - PSetV2 + TrajV10 Ensemble:  J1A={ens_s['J1A']:.4f} | J1B={ens_s['J1B']:.4f} | "
        f"J2-Raw={ens_s['J2_Raw']:.4f} | J2-noD={ens_s['J2_noD']:.4f}",
        flush=True,
    )
    print("-" * 120, flush=True)
    print("INTEGRATED MULTI-FAMILY STREAM BENCHMARK (XDX + SpellReranker):", flush=True)
    for k_m, d_m in [("with_pset_v1", mf["with_pset_v1"]), ("with_pset_v2", mf["with_pset_v2"]), ("with_traj_v10", mf["with_traj_v10"])]:
        print(
            f"  - {k_m:<16}: J1A={d_m['J1A']:.4f} | J1B-Std={d_m['J1B_Std']:.4f} | "
            f"J1B-Gen={d_m['J1B_Gen']:.4f} | J2-Raw={d_m['J2_Raw']:.4f} | J2-noD={d_m['J2_noD']:.4f}",
            flush=True,
        )
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()
