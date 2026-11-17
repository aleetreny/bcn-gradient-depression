"""EXP-V6: Super-Champion Multi-Family & Physics-Gated Mixture of Experts (`v6`).

Integrates all Wave 1 + Wave 2 orthogonal discoveries:
1. Exact H<->A Pitch & Team Equivariance (Symmetric Training + TTA across all model families).
2. `v1ctl_plus` (352 features): Knot-Anchored Ball Physics + Ghost-Incoherence + Bernardo Multi-Scale
   Track-Break-Free LSQ Player Velocities + Gap-Edge Continuity (`SymGBM_Plus_TTA` & `SymTrans_Plus_TTA`).
3. Multi-Regime Ghost-Ball Expert (`SymGBM_v6_TTA` & `SymTrans_v6_TTA`, trained on `degrade_v6` with
   slow/incoherent visible ghost balls).
4. 3rd Model Family (`SymBiGRU_TTA`, channel-norm Conv1d stem + 2-layer Bidirectional GRU for long ball gaps).
5. Symmetric OOF HSMM Decoders (`dec_v5_lin`, `dec_v5_hyb`) + LogOdds Total-Variation Denoising.
6. Physics-Gated Mixture of Experts (`MoE_v6`):
   - `w_ghost(t)`: smoothly activates `SymGBM_v6_TTA` on frames where `ball_visible == 1`, `min(d_H, d_A) > 3.0m`,
     and `ball_speed < 6.0 m/s` (smoothed over 1.5s).
   - `w_gap(t)`: smoothly activates `SymBiGRU_TTA` conditional H/A probabilities inside deep missing-ball gaps
     (`ball_gap_len > 6.0s` and `> 2.0s` from both gap edges) while keeping `p_D` anchored to `GBM + TCN`.
7. Broadcast Top-K Blackout & Ball Repair (`EXP-E2`) for BAR-ATH.

Evaluates all candidates across ALL 4 BENCHMARKS:
- Judge 1A: 7-Fold Nested LOMO on Sportec `degrade_v3`
- Judge 1B: 7-Fold Nested LOMO on Sportec `degrade_v6` (Ghost-Ball Incoherence)
- Judge 2:  Metrica Sample (3 games, standard and `without_dead_margin`)
- Judge 3:  BAR-ATH Broadcast-Repaired (Full-Match Plausibility + Public 5-Min Gate)
"""
from __future__ import annotations

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
from s3_7_nested_full import build_features
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.common import AWAY, DEAD, HOME

INT_TO_LABEL = {0: "H", 1: "A", 2: "D"}
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from exp_b_symmetry_eval import blend_proba
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
from exp_f_sym_decoder_and_boundary import augment_p_change_with_logodds_tv


def compute_physical_gates(X: pd.DataFrame, half: np.ndarray, max_ghost: float = 0.40, max_gap: float = 0.35):
    """Compute zero-leakage physical gates w_ghost(t) and w_gap(t) from tracking features."""
    vis = np.nan_to_num(X["ball_visible"].to_numpy(np.float32), nan=0.0)
    dH = np.nan_to_num(X["dist_ball_H"].to_numpy(np.float32), nan=0.0)
    dA = np.nan_to_num(X["dist_ball_A"].to_numpy(np.float32), nan=0.0)
    min_d = np.minimum(dH, dA)
    bspd = np.nan_to_num(X["ball_speed"].to_numpy(np.float32), nan=10.0)

    # Raw ghost indicator: ball is visible, > 3.0m from closest player of either team, and speed < 6.0 m/s
    g_dist = np.clip((min_d - 3.0) / 4.0, 0.0, 1.0)
    g_spd = np.clip((6.0 - bspd) / 4.0, 0.0, 1.0)
    g_raw = vis * g_dist * g_spd

    # Smooth ghost score over 1.5s (37 frames) within each half
    g_smooth = np.zeros_like(g_raw)
    kernel = np.ones(37, dtype=np.float32) / 37.0
    for hv in np.unique(half):
        if hv <= 0:
            continue
        idx = np.flatnonzero(half == hv)
        g_smooth[idx] = np.convolve(g_raw[idx], kernel, mode="same")
    w_ghost = max_ghost * np.clip(g_smooth * 1.5, 0.0, 1.0)

    # Gap gate: ball_since_seen and ball_until_seen (in seconds or frames; let's normalize automatically)
    since = np.nan_to_num(X["ball_since_seen"].to_numpy(np.float32), nan=0.0)
    until = np.nan_to_num(X["ball_until_seen"].to_numpy(np.float32), nan=0.0)
    # Check scale: if max > 250, it's in frames (25Hz); convert to seconds
    scale = 25.0 if np.nanmax(since) > 250.0 else 1.0
    since_s = since / scale
    until_s = until / scale
    gap_len_s = since_s + until_s
    edge_dist_s = np.minimum(since_s, until_s)

    # Activate BiGRU inside gaps > 6s and at least > 1.5s from gap edges
    gap_active = (1.0 - vis) * np.clip((gap_len_s - 6.0) / 6.0, 0.0, 1.0) * np.clip((edge_dist_s - 1.5) / 3.5, 0.0, 1.0)
    w_gap = max_gap * gap_active
    return w_ghost[:, None], w_gap


def moe_combine(
    g_v1: np.ndarray,
    tr_v1: np.ndarray,
    g_pl: np.ndarray,
    tr_pl: np.ndarray,
    g_v6: np.ndarray,
    tr_v6: np.ndarray,
    tcn: np.ndarray,
    tcnc: np.ndarray,
    gru: np.ndarray,
    gruc: np.ndarray,
    w_pl: float = 0.50,
    w_ghost: np.ndarray | float = 0.0,
    w_gap: np.ndarray | float = 0.0,
    w_gbm: float = 0.55,
    w_c: float = 0.50,
    mode: str = "hybrid",
):
    """Combine tree experts (v1ctl, v1ctl_plus, v6_ghost) and sequence experts (TCN, BiGRU)."""
    # 1. Clean tree expert = blend of SymGBM_v1ctl_TTA and SymGBM_Plus_TTA
    g_clean = (1.0 - w_pl) * g_v1 + w_pl * g_pl
    tr_clean = (1.0 - w_pl) * tr_v1 + w_pl * tr_pl

    # 2. Ghost-gated tree expert: blend in SymGBM_v6_TTA on ghost frames
    if isinstance(w_ghost, np.ndarray) or w_ghost > 0.0:
        wg_arr = w_ghost if isinstance(w_ghost, np.ndarray) else float(w_ghost)
        pg = (1.0 - wg_arr) * g_clean + wg_arr * g_v6
        wg_1d = wg_arr[:, 0] if isinstance(wg_arr, np.ndarray) else wg_arr
        ptr = (1.0 - wg_1d) * tr_clean + wg_1d * tr_v6
    else:
        pg = g_clean
        ptr = tr_clean

    # 3. Sequence expert: TCN provides p_D and baseline H/A; BiGRU blends into conditional H/(H+A) via w_gap
    if isinstance(w_gap, np.ndarray) or w_gap > 0.0:
        eps = 1e-6
        pD = tcn[:, 2]
        alive = np.clip(1.0 - pD, eps, 1.0)
        r_tcn_H = tcn[:, 0] / np.maximum(tcn[:, 0] + tcn[:, 1], eps)
        r_gru_H = gru[:, 0] / np.maximum(gru[:, 0] + gru[:, 1], eps)
        wgap_1d = w_gap[:, 0] if (isinstance(w_gap, np.ndarray) and w_gap.ndim == 2) else w_gap
        rH = (1.0 - wgap_1d) * r_tcn_H + wgap_1d * r_gru_H
        pseq = np.empty_like(tcn)
        pseq[:, 0] = alive * rH
        pseq[:, 1] = alive * (1.0 - rH)
        pseq[:, 2] = pD
        psc = (1.0 - 0.5 * wgap_1d) * tcnc + (0.5 * wgap_1d) * gruc
    else:
        pseq = tcn
        psc = tcnc

    p = blend_proba(pg, pseq, w_gbm=w_gbm, mode=mode)
    c = w_c * ptr + (1.0 - w_c) * psc
    return p, c


def save_and_validate_submission(path: Path, pred: np.ndarray, atd, second_start: int, kick2: int) -> dict:
    import possession_metrics as pm
    char_map = np.array(["H", "A", "D"])
    df = pd.DataFrame({
        "frame": np.arange(len(pred), dtype=np.int32),
        "label": char_map[pred.astype(np.int8)],
    })
    df.to_csv(path, index=False)

    assert len(pred) == 199319, f"{path.name}: Expected 199319 rows, got {len(pred)}"
    assert (pred[second_start:kick2] == DEAD).all(), f"{path.name}: Halftime break not all DEAD"
    assert (pred[~atd.graded] == DEAD).all(), f"{path.name}: Ungraded frames not all DEAD"
    table = pm.read_label_csv(str(path))
    reason = pm.validate_submission(table, 199319)
    assert not reason, f"{path.name}: Official validator failed: {reason}"

    g = atd.graded
    return {
        "rows": len(pred),
        "graded": int(g.sum()),
        "pct_H": 100.0 * float((pred[g] == HOME).mean()),
        "pct_A": 100.0 * float((pred[g] == AWAY).mean()),
        "pct_D": 100.0 * float((pred[g] == DEAD).mean()),
    }


def main():
    t0 = time.time()
    print("=== EXP-V6: SUPER-CHAMPION MULTI-FAMILY & PHYSICS-GATED EVALUATION ===", flush=True)

    sportec = load_all()
    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)

    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    plus_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    h_dir = ROOT / "experiments" / "outputs" / "exp_h_oof"
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    store_v3 = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std = pickle.load(f)[0]
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        loo_dec_lin, loo_dec_hyb, dec_v5_lin, dec_v5_hyb = pickle.load(f)
    dec_v4 = DecoderParams(**{
        k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items()
        if not k.startswith("_")
    })

    # Load physical feature matrices to compute exact zero-leakage physical gates
    gates_v3, gates_v6 = {}, {}
    for mid, m in sportec.items():
        Xv3 = build_features(degrade(m, cfg_v3, 0), f"sportec_{mid}_degrade_v3_s0", "v1ctl")
        Xv6 = build_features(degrade(m, cfg_v6, 0), f"sportec_{mid}_degrade_v6_s0", "v1ctl")
        gates_v3[mid] = compute_physical_gates(Xv3, m.half)
        gates_v6[mid] = compute_physical_gates(Xv6, m.half)

    gates_sample = {}
    for gid in GAMES:
        ms = sample_matches[gid]
        Xs = build_features(ms, f"sample_{gid}_raw", "v1ctl")
        gates_sample[gid] = compute_physical_gates(Xs, ms.half)

    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom["valid"].to_numpy().astype(bool))
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)
    Xa_rep = build_features(atd_rep, "atd_rep_True_True_None", "v1ctl")
    gate_atd = compute_physical_gates(Xa_rep, atd.half)

    print(
        f"  [Physical Gate Summary] mean w_ghost: Sportec_v3={np.mean([ float(g[0].mean()) for g in gates_v3.values() ]):.4f}, "
        f"Sportec_v6={np.mean([ float(g[0].mean()) for g in gates_v6.values() ]):.4f}, "
        f"Sample={np.mean([ float(g[0].mean()) for g in gates_sample.values() ]):.4f}, "
        f"BAR-ATH={float(gate_atd[0].mean()):.4f}",
        flush=True,
    )
    print(
        f"  [Physical Gate Summary] mean w_gap  : Sportec_v3={np.mean([ float(g[1].mean()) for g in gates_v3.values() ]):.4f}, "
        f"Sample={np.mean([ float(g[1].mean()) for g in gates_sample.values() ]):.4f}, "
        f"BAR-ATH={float(gate_atd[1].mean()):.4f}",
        flush=True,
    )

    # Load all OOF predictions
    oof = {}
    for T in MATCH_IDS:
        zb = np.load(sym_dir / f"sym_{T}.npz")
        zt = np.load(sym_dir / f"tcn_sym_{T}.npz")
        zp = np.load(plus_dir / f"plus_{T}.npz")
        zh = np.load(h_dir / f"bigru_sym_{T}.npz")
        zg = np.load(g_dir / f"ghost_sym_{T}.npz")
        z_orig_tcn = np.load(store_v3 / T / f"tcn_outer_{T}.npz")
        oof[T] = {
            "g_orig": np.load(store_v3 / T / f"gbm_outer_{T}.npy"),
            "tr_orig": np.load(store_v3 / T / f"trans_outer_{T}.npy"),
            "t_orig": z_orig_tcn["proba"],
            "tc_orig": z_orig_tcn["change"],
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
        }

    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        pe = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f:
        sample_v6, atd_rep_v6 = pickle.load(f)

    # Candidate definitions:
    # (name, w_pl, ghost_mode, gap_mode, bmode, wg, wc, dmode, tv_lam, sub_filename)
    candidates = [
        ("0. v4_ens_mean Baseline (with Broadcast Repair on J3)",
         0.0, "v4", "none", "linear", 0.50, 0.50, "v4", 0.0, None),
        ("1. v5_sym_linear (SymGBM_v1ctl + SymTCN, lin 0.55, dec_v4)",
         0.0, "none", "none", "linear", 0.55, 0.50, "v4", 0.0, None),
        ("2. v5_sym_hybrid (SymGBM_v1ctl + SymTCN, hyb 0.55, dec_v4)",
         0.0, "none", "none", "hybrid", 0.55, 0.50, "v4", 0.0, None),
        ("3. v5_sym_hybrid + SymDecoder (`dec_v5_hyb`)",
         0.0, "none", "none", "hybrid", 0.55, 0.50, "hyb", 0.0, None),
        ("4. v6_sym_plus_linear (SymGBM_Plus + SymTCN, lin 0.55, dec_v4)",
         1.0, "none", "none", "linear", 0.55, 0.50, "v4", 0.0, "submission_v6_sym_plus_linear.csv"),
        ("5. v6_sym_plus_hybrid (0.5*v1ctl + 0.5*Plus + SymTCN, hyb 0.55, dec_v5_hyb)",
         0.5, "none", "none", "hybrid", 0.55, 0.50, "hyb", 0.0, "submission_v6_sym_plus_hybrid.csv"),
        ("6. v6_pure_plus_hybrid (SymGBM_Plus + SymTCN, hyb 0.50, dec_v5_hyb)",
         1.0, "none", "none", "hybrid", 0.50, 0.50, "hyb", 0.0, "submission_v6_pure_plus_hybrid.csv"),
        ("7. v6_moe_ghost_lin (0.5*v1ctl + 0.5*Plus + GhostGate(v6) + GapGate(BiGRU), lin 0.55)",
         0.5, "phys", "phys", "linear", 0.55, 0.50, "v4", 0.0, "submission_v6_moe_super_champion.csv"),
        ("8. v6_moe_ghost_hyb (0.5*v1ctl + 0.5*Plus + GhostGate(v6) + GapGate(BiGRU), hyb 0.55, dec_v4)",
         0.5, "phys", "phys", "hybrid", 0.55, 0.50, "v4", 0.0, None),
        ("9. v6_moe_ghost_hyb_symdec (0.5*v1ctl + 0.5*Plus + GhostGate + GapGate, hyb 0.55, dec_v5_hyb)",
         0.5, "phys", "phys", "hybrid", 0.55, 0.50, "hyb", 0.0, None),
        ("10. v6_3family_seq (0.5*v1ctl + 0.5*Plus + 0.85*SymTCN + 0.15*SymBiGRU_HA, lin 0.52, dec_v4)",
         0.5, "phys", "const_0.15", "linear", 0.52, 0.50, "v4", 0.0, "submission_v6_3family_seq.csv"),
    ]

    print("\n--- COMPREHENSIVE 4-BENCHMARK EVALUATION ---", flush=True)
    header = (
        f"  {'Candidate':78s} | {'J1A: LOMO v3':18s} | {'J1B: LOMO v6':14s} "
        f"| {'J2: Sample (noD)':18s} | {'J3: BAR-ATH Pub (D_f1, pos/m, %D)':34s}"
    )
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    saved_subs = {}

    for cname, w_pl, gmode, gapmode, bmode, wg, wc, dmode, tv_lam, sub_name in candidates:
        # 1. Judge 1A (Sportec degrade_v3) & Judge 1B (Sportec degrade_v6)
        j1_v3, j1_v6 = [], []
        for T in MATCH_IDS:
            d = oof[T]
            dp = fold_dec_std[T] if dmode == "v4" else (loo_dec_lin[T] if dmode == "lin" else loo_dec_hyb[T])
            if gmode == "v4":
                p3 = blend_proba(d["g_orig"], d["t_orig"], w_gbm=0.50, mode="linear")
                c3 = 0.50 * d["tr_orig"] + 0.50 * d["tc_orig"]
                p6 = blend_proba(d["g_v3_on_v6"], d["t_orig"], w_gbm=0.50, mode="linear")
                c6 = 0.50 * d["tr_v3_on_v6"] + 0.50 * d["tc_orig"]
            else:
                wg_v3 = gates_v3[T][0] if gmode == "phys" else 0.0
                wgap_v3 = gates_v3[T][1] if gapmode == "phys" else (0.15 if gapmode == "const_0.15" else 0.0)
                p3, c3 = moe_combine(
                    d["g_v1"], d["tr_v1"], d["g_pl"], d["tr_pl"],
                    d["g_v6_on_v3"], d["tr_v6_on_v3"], d["tcn"], d["tcnc"], d["gru"], d["gruc"],
                    w_pl=w_pl, w_ghost=wg_v3, w_gap=wgap_v3, w_gbm=wg, w_c=wc, mode=bmode,
                )
                wg_v6 = gates_v6[T][0] if gmode == "phys" else 0.0
                wgap_v6 = gates_v6[T][1] if gapmode == "phys" else (0.15 if gapmode == "const_0.15" else 0.0)
                # On v6 test folds, use g_v3_on_v6 as the clean tree baseline and g_v6_on_v6 as the ghost tree
                p6, c6 = moe_combine(
                    d["g_v3_on_v6"], d["tr_v3_on_v6"], d["g_v3_on_v6"], d["tr_v3_on_v6"],
                    d["g_v6_on_v6"], d["tr_v6_on_v6"], d["tcn"], d["tcnc"], d["gru"], d["gruc"],
                    w_pl=0.0, w_ghost=wg_v6, w_gap=wgap_v6, w_gbm=wg, w_c=wc, mode=bmode,
                )
            if tv_lam > 0:
                c3 = augment_p_change_with_logodds_tv(p3, c3, sportec[T].half, alpha_grad=tv_lam)
                c6 = augment_p_change_with_logodds_tv(p6, c6, sportec[T].half, alpha_grad=tv_lam)
            pred3 = decode_v4(p3, sportec[T].half, dp, p_change=c3)
            pred6 = decode_v4(p6, sportec[T].half, dp, p_change=c6)
            j1_v3.append(metrics.score(sportec[T].labels, pred3, sportec[T].half)["spell_f1_mean"])
            j1_v6.append(metrics.score(sportec[T].labels, pred6, sportec[T].half)["spell_f1_mean"])

        # 2. Judge 2 (Metrica Sample 3 Games)
        s_raw, s_nod = [], []
        d_use = dec_v4 if dmode == "v4" else (dec_v5_lin if dmode == "lin" else dec_v5_hyb)
        for gid in GAMES:
            ms = sample_matches[gid]
            dh = sample_h[gid]
            dp_s = pe["sample"][gid]
            dv6 = sample_v6[gid]
            if gmode == "v4":
                ps = blend_proba(dh["g_orig"], dh["t_orig"], w_gbm=0.50, mode="linear")
                cs = 0.50 * dh["tr_orig"] + 0.50 * dh["tc_orig"]
            else:
                wg_s = gates_sample[gid][0] if gmode == "phys" else 0.0
                wgap_s = gates_sample[gid][1] if gapmode == "phys" else (0.15 if gapmode == "const_0.15" else 0.0)
                ps, cs = moe_combine(
                    dh["g_sym"], dh["tr_sym"], dp_s["g_plus_sym"], dp_s["tr_plus_sym"],
                    dv6["g_v6"], dv6["tr_v6"], dh["t_sym"], dh["tc_sym"], dh["b_sym"], dh["bc_sym"],
                    w_pl=w_pl, w_ghost=wg_s, w_gap=wgap_s, w_gbm=wg, w_c=wc, mode=bmode,
                )
            if tv_lam > 0:
                cs = augment_p_change_with_logodds_tv(ps, cs, ms.half, alpha_grad=tv_lam)
            pred_s = decode_v4(ps, ms.half, d_use, p_change=cs)
            s_raw.append(metrics.score(ms.labels, pred_s, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred_s, ms.half)["spell_f1_mean"])

        # 3. Judge 3 (BAR-ATH Broadcast-Repaired)
        if gmode == "v4":
            pa = blend_proba(atd_rep_h["g_orig"], atd_rep_h["t_orig"], w_gbm=0.50, mode="linear")
            ca = 0.50 * atd_rep_h["tr_orig"] + 0.50 * atd_rep_h["tc_orig"]
        else:
            wg_a = gate_atd[0] if gmode == "phys" else 0.0
            wgap_a = gate_atd[1] if gapmode == "phys" else (0.15 if gapmode == "const_0.15" else 0.0)
            pa, ca = moe_combine(
                atd_rep_h["g_sym"], atd_rep_h["tr_sym"], pe["atd_rep"]["g_plus_sym"], pe["atd_rep"]["tr_plus_sym"],
                atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"], atd_rep_h["t_sym"], atd_rep_h["tc_sym"],
                atd_rep_h["b_sym"], atd_rep_h["bc_sym"],
                w_pl=w_pl, w_ghost=wg_a, w_gap=wgap_a, w_gbm=wg, w_c=wc, mode=bmode,
            )
        if tv_lam > 0:
            ca = augment_p_change_with_logodds_tv(pa, ca, atd.half, alpha_grad=tv_lam)
        pred_a = decode_v4(pa, half_dec, d_use, p_change=ca)
        pred_a[second_start:kick2] = DEAD
        pred_a[~atd.graded] = DEAD

        pub = metrics.score_public(pred_a, bootstrap_n=100)
        pl = plausibility(pred_a, atd.half, atd.graded)
        sp = pub["spell_f1_mean"]
        df1 = pub["per_class"]["D"]["f1"]
        pm = pl["poss_per_min"]
        pd_val = pl["pct_D"]

        print(
            f"  {cname:78s} | {np.mean(j1_v3):.4f} (s={np.std(j1_v3):.4f}) | {np.mean(j1_v6):.4f}         "
            f"| {np.mean(s_raw):.4f} ({np.mean(s_nod):.4f}) | {sp:.4f} (D={df1:.3f}, {pm:.2f}/m, {pd_val:.1f}%D)",
            flush=True,
        )

        if sub_name is not None:
            sub_path = sub_dir / sub_name
            info = save_and_validate_submission(sub_path, pred_a, atd, second_start, kick2)
            saved_subs[sub_name] = (pred_a.copy(), info)

    print("\n--- SAVED & VALIDATED v6 SUBMISSIONS ---", flush=True)
    for sname, (pred_vec, info) in saved_subs.items():
        print(
            f"  [VALIDATED] {sname:38s}: rows={info['rows']} (graded={info['graded']}) | %H={info['pct_H']:.2f}% %A={info['pct_A']:.2f}% %D={info['pct_D']:.2f}%",
            flush=True,
        )

    # Compare agreement across v4, v5, and v6 submissions
    ref_files = [
        ("v4_ens_mean", ROOT / "submissions" / "submission_v4_ens_mean.csv"),
        ("v5_sym_linear", sub_dir / "submission_v5_sym_linear.csv"),
        ("v5_sym_hybrid", sub_dir / "submission_v5_sym_hybrid.csv"),
        ("v6_sym_plus_linear", sub_dir / "submission_v6_sym_plus_linear.csv"),
        ("v6_sym_plus_hybrid", sub_dir / "submission_v6_sym_plus_hybrid.csv"),
        ("v6_pure_plus_hybrid", sub_dir / "submission_v6_pure_plus_hybrid.csv"),
        ("v6_moe_super_champion", sub_dir / "submission_v6_moe_super_champion.csv"),
        ("v6_3family_seq", sub_dir / "submission_v6_3family_seq.csv"),
    ]
    loaded_vecs = {}
    for tag_n, fp in ref_files:
        if fp.exists():
            df_fp = pd.read_csv(fp)
            col = "label" if "label" in df_fp.columns else "team"
            loaded_vecs[tag_n] = df_fp[col].to_numpy()[atd.graded]

    print("\n--- PAIRWISE AGREEMENT MATRIX ON GRADED BAR-ATH (145,259 FRAMES) ---", flush=True)
    keys = list(loaded_vecs.keys())
    print("  " + f"{'Model':24s}" + "".join(f"{k:>22s}" for k in keys), flush=True)
    for k1 in keys:
        row_str = f"  {k1:24s}"
        for k2 in keys:
            agr = 100.0 * float(np.mean(loaded_vecs[k1] == loaded_vecs[k2]))
            row_str += f"{agr:21.2f}%"
        print(row_str, flush=True)

    print(f"\nEXP-V6 completed in {(time.time() - t0):.1f}s.", flush=True)


if __name__ == "__main__":
    main()
