"""EXP-E2: Broadcast Clock-Locked Blackout Alignment & Direct Tracking Repair on BAR-ATH.

1. Fixes the blackout alignment bug in `s4_6_broadcast_align.py`:
   - During BAR-ATH's 3 tactical blackouts, only 2-5 tactical players remain visible while 12-18 broadcast
     players are visible. Standard median across all broadcast players yields >3m cost even at the exact
     clock offset (L1=4014 in H1, L2=26920 in H2).
   - Using top-K visible tactical-to-broadcast matching recovers the broadcast segments right inside the
     tactical blackouts!
2. Directly repairs the BAR-ATH `Match` object (`atd_repaired`) before `build_features`:
   - Variant A: Repair players + ball ONLY inside the 3 tactical blackouts (`n_tac_vis <= 8`)
   - Variant B: Repair players + ball in blackouts + repair missing tactical ball in long gaps (>= 2s)
   - Variant C: Repair players + ball in blackouts + Smart Tagging live-phase prior in remaining unaligned blackouts
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
from s2_3_frame_models import plausibility
from s3_7_nested_full import build_features
from s4_6_broadcast_align import load_broadcast
from src.data.common import AWAY, DEAD, FPS, HOME
from src.data.degrade import _speed
from src.data.metrica_atd import load as load_atd
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from exp_e_multimodal_bar_ath import apply_multimodal_fusion, build_broadcast_tactical_Arrays, build_smart_tagging_arrays


def enhanced_broadcast_alignment(atd, Pb, hom_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover broadcast alignment including segments that fall inside tactical blackouts."""
    al = np.load(ROOT / "local_data/broadcast_alignment.npz")
    mapping = al["tactical_frame"].copy()
    fit = al["cost"].copy()

    Pt = atd.P[..., :2].astype(np.float32)
    T = len(Pt)
    edges = np.flatnonzero(np.diff(np.r_[0, hom_valid.astype(int), 0]))
    segs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 75]

    # Notice: in Half 1 (k < 85500), live clock offset is L1 ~ 4014 (+/- 2 frames).
    # In Half 2 (k >= 85500), live clock offset is L2 ~ 26920 (+/- 2 frames).
    recovered = 0
    for a, b in segs:
        if (mapping[a:b] >= 0).all():
            continue
        L_clock = 4014 if b < 85500 else 26920
        # Check if [a + L_clock, b + L_clock] overlaps a tactical blackout or low-visibility window,
        # and evaluate top-K matching cost around L_clock +/- 15 frames!
        ks = np.unique(np.linspace(a, b - 1, 25).astype(int))
        best_L, best_c = L_clock, np.inf
        for cand_L in range(L_clock - 15, L_clock + 16):
            costs_k = []
            for k in ks:
                t = k + cand_L
                if t < 0 or t >= T:
                    continue
                pts_b = Pb[k][np.isfinite(Pb[k, :, 0])]
                pts_t = Pt[t][np.isfinite(Pt[t, :, 0])]
                if len(pts_b) < 4 or len(pts_t) < 2:
                    continue
                # Pairwise distance between visible broadcast and visible tactical players
                dmat = np.linalg.norm(pts_t[:, None, :] - pts_b[None, :, :], axis=-1)  # (nt, nb)
                # For each visible tactical player, distance to nearest broadcast player
                d_t2b = dmat.min(axis=1)
                # Take median of best min(len(pts_t), len(pts_b)//2) matches
                k_match = max(2, min(len(pts_t), len(pts_b) - 1))
                costs_k.append(float(np.median(np.sort(d_t2b)[:k_match])))
            if costs_k:
                med_c = float(np.median(costs_k))
                if med_c < best_c:
                    best_c = med_c
                    best_L = cand_L
        if best_c < 1.2:
            mapping[a:b] = np.arange(a, b) + best_L
            fit[a:b] = best_c
            recovered += (b - a)
            print(f"  [Recovered Segment] bcast={a}..{b} ({(b-a)/FPS:.1f}s) -> tac={a+best_L}..{b+best_L} | offset={best_L} | topK_cost={best_c:.3f}m", flush=True)

    print(f"  Total newly recovered broadcast frames: {recovered} ({recovered/FPS:.1f}s)", flush=True)
    return mapping, fit


def repair_atd_match(
    atd,
    Pb: np.ndarray,
    Bb: np.ndarray,
    slot_team_b: np.ndarray,
    mapping: np.ndarray,
    repair_blackout_players: bool = True,
    repair_blackout_ball: bool = True,
    repair_gap_ball_min_s: float | None = None,
):
    """Impute missing tactical players (in blackouts) and missing tactical ball from aligned Broadcast ATD."""
    P = atd.P.copy()
    B = atd.B.copy()
    T, S, _ = P.shape

    # Build inverse mapping: tactical frame t -> broadcast frame k
    t_to_k = np.full(T, -1, dtype=np.int64)
    ok_k = np.flatnonzero((mapping >= 0) & (mapping < T))
    t_to_k[mapping[ok_k]] = ok_k

    n_tac_vis = np.isfinite(P[..., 0]).sum(axis=1)
    is_blackout = atd.graded & (n_tac_vis <= 8)

    # Compute tactical ball gap lengths
    bvis = np.isfinite(B[:, 0])
    gap_len = np.zeros(T, dtype=np.int32)
    cur = 0
    for i in range(T):
        cur = cur + 1 if not bvis[i] else 0
        gap_len[i] = cur
    cur = 0
    for i in range(T - 1, -1, -1):
        cur = max(cur, gap_len[i]) if not bvis[i] else 0
        gap_len[i] = cur

    n_p_rep = 0
    n_b_rep = 0

    for t in range(T):
        k = int(t_to_k[t])
        if k < 0:
            continue
        # 1. Repair players during tactical blackouts
        if repair_blackout_players and is_blackout[t]:
            for tm in (HOME, AWAY):
                b_idx = np.flatnonzero((slot_team_b == tm) & np.isfinite(Pb[k, :, 0]))
                if len(b_idx) == 0:
                    continue
                t_slots = np.flatnonzero((atd.slot_team == tm) & (~atd.slot_is_gk))
                vis_t = t_slots[np.isfinite(P[t, t_slots, 0])]
                free_t = t_slots[~np.isfinite(P[t, t_slots, 0])]
                ptr = 0
                for bi in b_idx:
                    if ptr >= len(free_t):
                        break
                    b_pos = Pb[k, bi, :2]
                    if len(vis_t) > 0:
                        d_existing = np.linalg.norm(P[t, vis_t, :2] - b_pos[None, :], axis=1).min()
                        if d_existing < 1.8:
                            continue
                    P[t, free_t[ptr], :2] = b_pos
                    ptr += 1
            if ptr > 0:
                n_p_rep += 1

        # 2. Repair ball
        if not np.isfinite(B[t, 0]) and np.isfinite(Bb[k, 0]):
            do_ball = False
            if repair_blackout_ball and is_blackout[t]:
                do_ball = True
            if repair_gap_ball_min_s is not None and gap_len[t] >= int(repair_gap_ball_min_s * FPS):
                do_ball = True
            if do_ball:
                B[t, :2] = Bb[k, :2]
                n_b_rep += 1

    # Recompute speeds on repaired P and B
    if n_p_rep > 0:
        P[..., 2] = _speed(P[..., :2], atd.half)
    if n_b_rep > 0:
        B[:, 2] = _speed(B[:, :2], atd.half)

    print(f"  [Repair Summary] Repaired players on {n_p_rep} blackout frames; repaired ball on {n_b_rep} frames.", flush=True)
    return replace(atd, P=P, B=B)


def main() -> None:
    t0 = time.time()
    import torch
    from src.models import tcn as tcn_mod
    torch.use_deterministic_algorithms(True)
    tcn_mod._device = lambda: torch.device("cpu")

    print("=== EXP-E2: Broadcast Clock-Locked Blackout Alignment & Direct Tracking Repair ===", flush=True)
    atd = load_atd()
    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)

    mapping, fit = enhanced_broadcast_alignment(atd, Pb, hom_valid)

    # Check blackout coverage after enhanced alignment!
    n_tac_vis = np.isfinite(atd.P[..., 0]).sum(axis=1)
    is_blackout = atd.graded & (n_tac_vis <= 6)
    t_to_k = np.full(atd.n_frames, -1, dtype=np.int64)
    ok_k = np.flatnonzero((mapping >= 0) & (mapping < atd.n_frames))
    t_to_k[mapping[ok_k]] = ok_k
    bo_covered = is_blackout & (t_to_k >= 0)
    print(f"  Blackout frames covered by Enhanced Broadcast Alignment: {int(bo_covered.sum())} / {int(is_blackout.sum())} ({100*bo_covered.sum()/max(1,is_blackout.sum()):.1f}%)", flush=True)
    # Inspect the 3 blackouts individually!
    edges_bo = np.flatnonzero(np.diff(np.r_[0, is_blackout.astype(int), 0]))
    for a, b in zip(edges_bo[::2], edges_bo[1::2]):
        if b - a >= 25:
            cov = 100.0 * float(np.mean(t_to_k[a:b] >= 0))
            print(f"    Blackout {a}..{b} ({(b-a)/FPS:.1f}s, half={atd.half[a]}): Broadcast aligned = {cov:.1f}%", flush=True)

    models = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())
    dec_ens = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})
    half_dec = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    st_info = build_smart_tagging_arrays(atd)
    bcast_info = build_broadcast_tactical_Arrays(atd)

    # Evaluate:
    # 0. Baseline atd
    # 1. Blackout-Repaired atd (players + ball repaired inside blackouts)
    # 2. Blackout + Long-Gap (>=3s) Ball Repaired atd
    # 3. Blackout-Repaired atd + Smart Tagging Live Prior on remaining unaligned blackouts
    variants = [
        ("0. v4 Baseline (unrepaired)", False, False, None, 0.0),
        ("1. +Broadcast Blackout Repair (P+B in blackouts)", True, True, None, 0.0),
        ("2. +Broadcast Blackout + Gap>=3s Ball Repair", True, True, 3.0, 0.0),
        ("3. +Broadcast Blackout Repair + SmartTagging Blackouts", True, True, None, 2.5),
    ]

    pred_ref = None
    for vname, rep_p, rep_b, gap_s, alpha_bo_st in variants:
        print(f"\n--- Evaluating {vname} ---", flush=True)
        if not rep_p and not rep_b and gap_s is None:
            m_use = atd
            X_use = build_features(m_use, "atd", "v1ctl")
        else:
            m_use = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping, rep_p, rep_b, gap_s)
            X_use = build_features(m_use, f"atd_rep_{rep_p}_{rep_b}_{gap_s}", "v1ctl")

        pg = models["gbm"].predict_proba(X_use)
        pt, ct = models["tcn"].predict(X_use, m_use.half)
        cm = models["trans"].predict(X_use)
        p_ens = 0.5 * pg + 0.5 * pt
        c_ens = 0.5 * cm + 0.5 * ct

        if alpha_bo_st > 0:
            # For any STILL-unrepaired blackout frame (<=6 visible players in m_use), use Smart Tagging active phase!
            n_vis_now = np.isfinite(m_use.P[..., 0]).sum(axis=1)
            still_bo = m_use.graded & (n_vis_now <= 6) & (np.abs(st_info["net_team_score"]) > 0) & (~st_info["in_set_piece"])
            lp = np.log(np.clip(p_ens.astype(np.float64), 1e-6, 1.0))
            sc = np.clip(st_info["net_team_score"][still_bo], -1.5, 1.5)
            lp[still_bo, HOME] += alpha_bo_st * sc
            lp[still_bo, AWAY] -= alpha_bo_st * sc
            lp[still_bo, DEAD] -= 2.5 * alpha_bo_st  # Overcome the false DEAD logit collapse in blackouts!
            lp -= lp.max(axis=1, keepdims=True)
            e = np.exp(lp)
            p_ens = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

        pred = decode_v4(p_ens, half_dec, dec_ens, p_change=c_ens)
        pred[second_start:kick2] = DEAD
        pred[~atd.graded] = DEAD
        if pred_ref is None:
            pred_ref = pred.copy()

        pub = metrics.score_public(pred, bootstrap_n=100)
        pl = plausibility(pred, atd.half, atd.graded)
        bo_h = 100.0 * float(np.mean(pred[is_blackout] == HOME))
        bo_a = 100.0 * float(np.mean(pred[is_blackout] == AWAY))
        bo_d = 100.0 * float(np.mean(pred[is_blackout] == DEAD))
        diff_all = 100.0 * float(np.mean(pred[atd.graded] != pred_ref[atd.graded]))
        print(
            f"  Result -> Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
            f"| pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}% "
            f"| Blackouts(H/A/D)={bo_h:.1f}/{bo_a:.1f}/{bo_d:.1f}% | diff={diff_all:.2f}%",
            flush=True,
        )

    print(f"\nEXP-E2 completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
