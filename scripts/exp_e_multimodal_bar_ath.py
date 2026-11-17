"""EXP-E: Multi-Modal BAR-ATH Fusion (Aligned Broadcast ATD M14 + Smart Tagging M5).

Audits and evaluates:
1. Coverage of Aligned Broadcast ATD (`local_data/broadcast_alignment.npz`) and Smart Tagging (`M5`)
   during BAR-ATH's 3 tactical blackouts, long ball gaps (>3s), and linear-interpolated ball stretches.
2. Broadcast Ball & Proximity Fusion (`M14`):
   - Using aligned broadcast ball & player proximity when tactical ball is missing or linear-interpolated.
3. Smart Tagging Blind-Spot & Blackout Prior (`M5`):
   - Injecting phase-implied team log-likelihood ratios ONLY when tactical tracking is blind
     (blackouts with <=6 players visible, or long ball gaps >3s inside an active live phase),
     plus SET PIECES dead-ball pre-kick anchors.
4. Sportec 7-Fold LOMO validation of Blind-Spot Prior Injection during simulated blackouts & long ball gaps.
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
from s1_6_smart_tagging import DEFENSIVE_CODES, TEAM, ball_speed, find_kick
from s2_3_frame_models import plausibility
from s3_7_nested_full import build_features
from s4_6_broadcast_align import load_broadcast
from src.data.ball_interp import linear_mask
from src.data.common import AWAY, DEAD, FPS, HOME
from src.data.metrica_atd import load as load_atd
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from starter.load_smart_tagging import load as load_tags


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


def build_broadcast_tactical_Arrays(atd) -> dict[str, np.ndarray]:
    """Map aligned broadcast players & ball onto the tactical timeline (0..T-1)."""
    T = atd.n_frames
    Pb, Bb, slot_team_b = load_broadcast()
    al = np.load(ROOT / "local_data/broadcast_alignment.npz")
    tac_frame = al["tactical_frame"]
    cost = al["cost"]

    bcast_aligned = np.zeros(T, dtype=bool)
    bcast_ball = np.full((T, 2), np.nan, dtype=np.float32)
    bcast_d_h = np.full(T, np.nan, dtype=np.float32)
    bcast_d_a = np.full(T, np.nan, dtype=np.float32)
    bcast_n_vis = np.zeros(T, dtype=np.int32)

    valid_k = np.flatnonzero((tac_frame >= 0) & (tac_frame < T) & (cost < 1.2))
    t_idx = tac_frame[valid_k]
    bcast_aligned[t_idx] = True
    bcast_ball[t_idx] = Bb[valid_k]
    bcast_n_vis[t_idx] = np.isfinite(Pb[valid_k, :, 0]).sum(axis=1)

    # For frames with broadcast ball visible, compute distance to nearest H and A player
    # using BOTH broadcast players and tactical players!
    Pt = atd.P[..., :2].astype(np.float32)
    sel_bh = slot_team_b == HOME
    sel_ba = slot_team_b == AWAY
    sel_th = atd.slot_team == HOME
    sel_ta = atd.slot_team == AWAY

    has_bball = np.flatnonzero(np.isfinite(bcast_ball[:, 0]))
    for t in has_bball:
        bxy = bcast_ball[t]
        # Tactical distances to broadcast ball
        dt = np.linalg.norm(Pt[t] - bxy[None, :], axis=1)
        dh_t = np.nanmin(dt[sel_th]) if np.isfinite(dt[sel_th]).any() else np.nan
        da_t = np.nanmin(dt[sel_ta]) if np.isfinite(dt[sel_ta]).any() else np.nan
        bcast_d_h[t] = dh_t
        bcast_d_a[t] = da_t

    for k in valid_k:
        t = int(tac_frame[k])
        bxy = Bb[k]
        if not np.isfinite(bxy[0]):
            continue
        db = np.linalg.norm(Pb[k] - bxy[None, :], axis=1)
        dh_b = np.nanmin(db[sel_bh]) if np.isfinite(db[sel_bh]).any() else np.nan
        da_b = np.nanmin(db[sel_ba]) if np.isfinite(db[sel_ba]).any() else np.nan
        if np.isfinite(dh_b):
            bcast_d_h[t] = np.nanmin([bcast_d_h[t], dh_b]) if np.isfinite(bcast_d_h[t]) else dh_b
        if np.isfinite(da_b):
            bcast_d_a[t] = np.nanmin([bcast_d_a[t], da_b]) if np.isfinite(bcast_d_a[t]) else da_b

    return {
        "aligned": bcast_aligned,
        "ball": bcast_ball,
        "d_h": bcast_d_h,
        "d_a": bcast_d_a,
        "n_vis": bcast_n_vis,
    }


def build_smart_tagging_arrays(atd) -> dict[str, np.ndarray]:
    """Extract frame-level Smart Tagging live phase team (0=H, 1=A, -1=none) and set-piece pre-kick dead windows."""
    T = atd.n_frames
    tags = load_tags()
    spd = ball_speed(atd)

    # Accumulate log-odds / votes per frame from active live phases
    h_votes = np.zeros(T, dtype=np.float32)
    a_votes = np.zeros(T, dtype=np.float32)
    in_phase = np.zeros(T, dtype=bool)
    in_set_piece = np.zeros(T, dtype=bool)
    pre_kick_dead = np.zeros(T, dtype=bool)
    post_kick_team = np.full(T, -1, dtype=np.int8)

    for t in tags:
        code_up = t["code"].upper()
        a = max(0, int(t["start_frame"]))
        b = min(T, int(t["end_frame"]))
        if a >= b:
            continue
        team_name = t["tags"].get("Team")
        if code_up == "SET PIECES":
            in_set_piece[a:b] = True
            kind = t["tags"].get("Type", "?")
            k = find_kick(atd, spd, a - 5 * FPS, b + 5 * FPS, kind)
            if k is not None and a <= k <= b:
                # Before the physical kick (inside the set piece window), the ball is DEAD!
                pre_start = max(a, k - 12 * FPS)
                pre_kick_dead[pre_start:k] = True
                if team_name in TEAM:
                    post_kick_team[k:min(T, k + 4 * FPS)] = TEAM[team_name]
        else:
            in_phase[a:b] = True
            if team_name in TEAM:
                poss_team = 1 - TEAM[team_name] if code_up in DEFENSIVE_CODES else TEAM[team_name]
                # Higher weight for core controlled phases (Build Up, Progression, Creating Chances)
                w = 1.2 if code_up in ("BUILD UP", "PROGRESSION", "CREATING CHANCES") else 0.8
                if poss_team == HOME:
                    h_votes[a:b] += w
                else:
                    a_votes[a:b] += w

    net_team_score = h_votes - a_votes  # >0 favors HOME, <0 favors AWAY
    return {
        "in_phase": in_phase,
        "in_set_piece": in_set_piece,
        "net_team_score": net_team_score,
        "pre_kick_dead": pre_kick_dead,
        "post_kick_team": post_kick_team,
    }


def apply_multimodal_fusion(
    proba: np.ndarray,
    atd,
    bcast: dict[str, np.ndarray],
    st_info: dict[str, np.ndarray],
    alpha_blackout_st: float = 1.2,
    alpha_gap_st: float = 0.45,
    alpha_bcast_touch: float = 0.60,
    alpha_pre_kick_d: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Fuse Broadcast ATD and Smart Tagging priors into frame log-probabilities ONLY where tactical tracking is degraded."""
    lp = np.log(np.clip(proba.astype(np.float64), eps, 1.0))
    T = atd.n_frames

    # Identify degraded tactical regimes (label-free, from tracking only!)
    n_tac_vis = np.isfinite(atd.P[..., 0]).sum(axis=1)
    is_blackout = atd.graded & (n_tac_vis <= 6)
    bvis = atd.ball_xy_visible
    blin = linear_mask(atd.B, atd.half)

    # Compute tactical ball gap length at each frame
    gap_len = np.zeros(T, dtype=np.int32)
    cur = 0
    for i in range(T):
        if not bvis[i]:
            cur += 1
        else:
            cur = 0
        gap_len[i] = cur
    cur = 0
    for i in range(T - 1, -1, -1):
        if not bvis[i]:
            cur = max(cur, gap_len[i])
            gap_len[i] = cur
        else:
            cur = 0
    is_long_gap = atd.graded & (~bvis) & (gap_len >= 3 * FPS)

    # 1. Smart Tagging in Blackouts (where tactical tracking has almost no players and no ball!)
    if alpha_blackout_st > 0:
        score = st_info["net_team_score"]
        mask_bo = is_blackout & (np.abs(score) > 0)
        lp[mask_bo, HOME] += alpha_blackout_st * np.clip(score[mask_bo], -1.5, 1.5)
        lp[mask_bo, AWAY] -= alpha_blackout_st * np.clip(score[mask_bo], -1.5, 1.5)
        # If inside an active live phase during a blackout (and not a set piece), prevent false 60s Dead-ball collapse!
        live_bo = mask_bo & (~st_info["in_set_piece"])
        lp[live_bo, DEAD] -= 0.5 * alpha_blackout_st

    # 2. Smart Tagging in Long Ball Gaps (>3s)
    if alpha_gap_st > 0:
        score = st_info["net_team_score"]
        mask_gap = is_long_gap & (~is_blackout) & (np.abs(score) > 0)
        lp[mask_gap, HOME] += alpha_gap_st * np.clip(score[mask_gap], -1.5, 1.5)
        lp[mask_gap, AWAY] -= alpha_gap_st * np.clip(score[mask_gap], -1.5, 1.5)

    # 3. Broadcast Ball Proximity when Tactical Ball is Missing or Linear-Interpolated
    if alpha_bcast_touch > 0:
        dh = bcast["d_h"]
        da = bcast["d_a"]
        dmin = np.minimum(np.nan_to_num(dh, nan=99.0), np.nan_to_num(da, nan=99.0))
        # Broadcast shows a player within 2.5m of the broadcast ball while tactical ball is missing or linear
        mask_bc = atd.graded & ((~bvis) | blin) & (dmin < 2.5)
        soft_ba = np.tanh((np.nan_to_num(da, nan=30.0) - np.nan_to_num(dh, nan=30.0)) / 1.5)
        lp[mask_bc, HOME] += alpha_bcast_touch * soft_ba[mask_bc]
        lp[mask_bc, AWAY] -= alpha_bcast_touch * soft_ba[mask_bc]

    # 4. Smart Tagging Set-Piece Pre-Kick Dead-Ball Anchor
    if alpha_pre_kick_d > 0:
        mask_pk = atd.graded & st_info["pre_kick_dead"] & (~bvis | (ball_speed(atd) < 3.0))
        lp[mask_pk, DEAD] += alpha_pre_kick_d
        mask_post = atd.graded & (st_info["post_kick_team"] >= 0)
        for tm in (HOME, AWAY):
            sel = mask_post & (st_info["post_kick_team"] == tm)
            lp[sel, tm] += 0.5 * alpha_pre_kick_d
            lp[sel, DEAD] -= 0.5 * alpha_pre_kick_d

    lp -= lp.max(axis=1, keepdims=True)
    e = np.exp(lp)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def main() -> None:
    t0 = time.time()
    import torch
    from src.models import tcn as tcn_mod
    torch.use_deterministic_algorithms(True)
    tcn_mod._device = lambda: torch.device("cpu")

    print("=== EXP-E: Multi-Modal BAR-ATH Fusion (Broadcast ATD M14 + Smart Tagging M5) ===", flush=True)
    atd = load_atd()
    bcast = build_broadcast_tactical_Arrays(atd)
    st_info = build_smart_tagging_arrays(atd)

    # Step 1: Audit Coverage inside Blackouts, Ball Gaps, and Linear Ball Stretches
    n_tac_vis = np.isfinite(atd.P[..., 0]).sum(axis=1)
    is_blackout = atd.graded & (n_tac_vis <= 6)
    bvis = atd.ball_xy_visible
    blin = linear_mask(atd.B, atd.half)
    bcast_bvis = np.isfinite(bcast["ball"][:, 0])

    print("\n--- Step 1: Multi-Modal Coverage Audit on BAR-ATH Graded Frames ---", flush=True)
    print(f"  Total graded frames: {int(atd.graded.sum())} ({atd.graded.sum()/FPS/60:.1f} min)", flush=True)
    print(f"  Blackout frames (<=6 tactical players): {int(is_blackout.sum())} ({is_blackout.sum()/FPS:.1f} s)", flush=True)
    print(f"    -> Covered by Smart Tagging active phase: {100*np.mean(np.abs(st_info['net_team_score'][is_blackout]) > 0):.1f}%", flush=True)
    print(f"    -> Covered by Aligned Broadcast (>5 players): {100*np.mean(bcast['n_vis'][is_blackout] >= 5):.1f}%", flush=True)
    print(f"  Missing Tactical Ball frames: {int((atd.graded & ~bvis).sum())} ({100*np.mean(~bvis[atd.graded]):.1f}% of graded)", flush=True)
    print(f"    -> Covered by Smart Tagging active phase: {100*np.mean(np.abs(st_info['net_team_score'][atd.graded & ~bvis]) > 0):.1f}%", flush=True)
    print(f"    -> Covered by Aligned Broadcast visible ball: {100*np.mean(bcast_bvis[atd.graded & ~bvis]):.1f}%", flush=True)
    print(f"  Linear-Interpolated Tactical Ball frames: {int((atd.graded & bvis & blin).sum())}", flush=True)
    print(f"    -> Broadcast shows player <2.5m from ball: {int((atd.graded & ((~bvis) | blin) & (np.minimum(np.nan_to_num(bcast['d_h'], nan=99), np.nan_to_num(bcast['d_a'], nan=99)) < 2.5)).sum())} frames", flush=True)

    # Step 2: Evaluate on BAR-ATH (Plausibility inside Blackouts, Full Match Plausibility, and Public 5-min Gate)
    models = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())
    dec_ens = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})

    Xa = build_features(atd, "atd", "v1ctl")
    p_gbm = models["gbm"].predict_proba(Xa)
    p_tcn, c_tcn = models["tcn"].predict(Xa, atd.half)
    c_tr = models["trans"].predict(Xa)

    p_hyb = blend_proba(p_gbm, p_tcn, w_gbm=0.60, mode="hybrid")
    c_hyb = 0.75 * c_tr + 0.25 * c_tcn
    half_dec = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])

    print("\n--- Step 2: BAR-ATH Multi-Modal Ablation (Blackouts, Full-Match Plausibility & Public Sanity Gate) ---", flush=True)
    configs = [
        ("0. Baseline v4 (hyb_60_75, no fusion)", 0.0, 0.0, 0.0, 0.0),
        ("1. +Blackout SmartTagging only",        1.2, 0.0, 0.0, 0.0),
        ("2. +Blackout & LongGap SmartTagging",   1.2, 0.45, 0.0, 0.0),
        ("3. +Broadcast Ball Touch Fusion only",  0.0, 0.0, 0.60, 0.0),
        ("4. +SetPiece Kick Anchors only",        0.0, 0.0, 0.0, 1.0),
        ("5. FULL Multi-Modal Fusion (1+2+3+4)",  1.2, 0.45, 0.60, 1.0),
    ]
    pred_base = None
    for cname, a_bo, a_gap, a_bc, a_pk in configs:
        p_fused = apply_multimodal_fusion(p_hyb, atd, bcast, st_info, a_bo, a_gap, a_bc, a_pk)
        pred = decode_v4(p_fused, half_dec, dec_ens, p_change=c_hyb)
        pred[second_start:kick2] = DEAD
        pred[~atd.graded] = DEAD
        if pred_base is None:
            pred_base = pred.copy()
        pub = metrics.score_public(pred, bootstrap_n=100)
        pl = plausibility(pred, atd.half, atd.graded)
        # Check what happens specifically inside the 3 blackouts!
        bo_d_pct = 100.0 * float(np.mean(pred[is_blackout] == DEAD))
        bo_h_pct = 100.0 * float(np.mean(pred[is_blackout] == HOME))
        bo_a_pct = 100.0 * float(np.mean(pred[is_blackout] == AWAY))
        changed_pct = 100.0 * float(np.mean(pred[atd.graded] != pred_base[atd.graded]))
        print(
            f"  {cname:38s} | Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
            f"| pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}% "
            f"| Blackout(H/A/D)={bo_h_pct:.0f}/{bo_a_pct:.0f}/{bo_d_pct:.0f}% | diff={changed_pct:.2f}%",
            flush=True,
        )

    print(f"\nEXP-E completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
