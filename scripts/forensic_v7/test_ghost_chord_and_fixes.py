"""Test the 'Ghost-Ball Chord Theorem' (`Y -> X -> Y` during ghost-ball / low Local-Newton segments)
and test all 4 surgical post-decoding fixes across J1A, J1B-Std, J1B-Gen, J2, and J3.
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore")

EXP_LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_LAB / "py_pkgs"))
sys.path.insert(0, str(EXP_LAB))
sys.path.insert(0, str(EXP_LAB / "scripts"))
sys.path.insert(0, str(EXP_LAB / "evaluation"))

import possession_metrics as pm
from games import GAMES as GAME_INFO, anchor_video_frame
from s2_3_frame_models import plausibility
from s3_11_tcn_local import without_dead_margin
from src.data.common import AWAY, DEAD, HOME
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.spell_reranker import SpellRerankerConfig, apply_spell_reranker, rle_runs
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
from src.eval import metrics
from exp_v6_super_champion import compute_physical_gates
from exp_w7a_player_set_net import load_cached_physv2_table
from exp_w7b_super_champion_v7 import (
    _blend_6fam_heads_v7,
    compute_local_newton_gate,
    compute_w_clean_v7,
)


def audit_yxy_ghost_chords(pred, gt, half, proba, p_change, p4d, X_df, w_ghost, a_newt):
    valid = (half == 1) | (half == 2)
    arr = np.where(valid, pred, -1)
    runs = rle_runs(arr)
    wg = w_ghost[:, 0] if w_ghost.ndim == 2 else w_ghost
    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=50.0)
    imp = np.nan_to_num(X_df["impulse_mag_d4"].to_numpy(np.float64), nan=0.0)

    f1_base = metrics.score(gt, pred, half)["spell_f1_mean"]
    rows = []
    for idx in range(1, len(runs) - 1):
        s, e, x = runs[idx]
        if x not in (HOME, AWAY):
            continue
        ps, pe, prev_y = runs[idx - 1]
        ns, ne, next_y = runs[idx + 1]
        if prev_y != next_y or prev_y not in (HOME, AWAY):
            continue
        if half[s] != half[ps] or half[s] != half[ns]:
            continue
        L = e - s
        if L > 160:
            continue

        p_abs = pred.copy()
        p_abs[s:e] = prev_y
        delta_f1 = metrics.score(gt, p_abs, half)["spell_f1_mean"] - f1_base

        d1_x = d1h[s:e] if x == HOME else d1a[s:e]
        d1_y = d1a[s:e] if x == HOME else d1h[s:e]
        dir_in = 0 if (prev_y == HOME and x == AWAY) else 1
        dir_out = 1 if dir_in == 0 else 0
        b4d_in = float(np.max(p4d[max(0, s-6):min(len(pred), s+7), dir_in]))
        b4d_out = float(np.max(p4d[max(0, e-6):min(len(pred), e+7), dir_out]))

        rows.append({
            "L": L,
            "delta_f1": delta_f1,
            "should_absorb": delta_f1 > 1e-6,
            "should_keep": delta_f1 < -1e-6,
            "mean_wg": float(np.mean(wg[s:e])),
            "max_wg": float(np.max(wg[s:e])),
            "mean_anewt": float(np.mean(a_newt[s:e])),
            "min_anewt": float(np.min(a_newt[s:e])),
            "mean_pX": float(np.mean(proba[s:e, x])),
            "max_pX": float(np.max(proba[s:e, x])),
            "b4d_pair": 0.5 * (b4d_in + b4d_out),
            "min_d1_x": float(np.min(d1_x)),
            "frames_contact_x": int(np.sum(d1_x < 1.5)),
            "max_imp": float(np.max(imp[s:e])),
            "surround_min": min(pe - ps, ne - ns),
        })
    return rows


def main():
    sportec = load_sportec()
    w7b_dir = EXP_LAB / "experiments" / "outputs" / "exp_w7b_oof"
    w3_dir = EXP_LAB / "experiments" / "outputs" / "exp_w3_oof"
    X_v3 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", sportec[mid].half) for mid in MATCH_IDS}
    X_v6 = {mid: load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", sportec[mid].half) for mid in MATCH_IDS}
    gates_v3 = {mid: compute_physical_gates(X_v3[mid], sportec[mid].half) for mid in MATCH_IDS}
    gates_v6 = {mid: compute_physical_gates(X_v6[mid], sportec[mid].half) for mid in MATCH_IDS}
    w7b_oof = {T: np.load(w7b_dir / f"w7b_{T}.npz") for T in MATCH_IDS}
    w3_oof = {T: np.load(w3_dir / f"b4d_{T}.npz") for T in MATCH_IDS}

    for name, pred_k, prob_k, c_k, p4d_k, X_map, g_map in [
        ("J1A (Sportec v3)", "pred_v3_rerank", "proba_v3", "pchange_v3", "p4d_v3", X_v3, gates_v3),
        ("J1B-Std (Sportec v6)", "pred_v6_std_rerank", "proba_v6_std", "pchange_v6_std", "p4d_v6", X_v6, gates_v6),
        ("J1B-Gen (Sportec v6 Native)", "pred_v6_gen_rerank", "proba_v6_gen", "pchange_v6_gen", "p4d_v6", X_v6, gates_v6),
    ]:
        rows = []
        for T in MATCH_IDS:
            z = w7b_oof[T]
            h = sportec[T].half
            a_newt = compute_local_newton_gate(X_map[T], h, g_map[T][0])
            rows.extend(audit_yxy_ghost_chords(
                z[pred_k], sportec[T].labels, h, z[prob_k], z[c_k], w3_oof[T][p4d_k],
                X_map[T], g_map[T][0], a_newt,
            ))
        df = pd.DataFrame(rows)
        pos = df[df["should_absorb"]]
        neg = df[df["should_keep"]]
        print(f"\n=====================================================================================")
        print(f"Y->X->Y GHOST-CHORD & FEATURE AUDIT: {name}")
        print(f"=====================================================================================")
        print(f"  Total Y->X->Y (L<=160): {len(df)} | Should Absorb (False Blip): {len(pos)} (Oracle={pos['delta_f1'].sum()/7:+.4f}) | Should Keep (True Turnover): {len(neg)}")
        print(f"  Should Absorb (False Blip) means: L={pos['L'].mean():.1f}, mean_pX={pos['mean_pX'].mean():.3f}, max_pX={pos['max_pX'].mean():.3f}, b4d={pos['b4d_pair'].mean():.3f}, mean_wg={pos['mean_wg'].mean():.3f}, mean_anewt={pos['mean_anewt'].mean():.3f}, min_d1_x={pos['min_d1_x'].mean():.2f}, contact_f={pos['frames_contact_x'].mean():.1f}")
        print(f"  Should Keep   (True Turn)  means: L={neg['L'].mean():.1f}, mean_pX={neg['mean_pX'].mean():.3f}, max_pX={neg['max_pX'].mean():.3f}, b4d={neg['b4d_pair'].mean():.3f}, mean_wg={neg['mean_wg'].mean():.3f}, mean_anewt={neg['mean_anewt'].mean():.3f}, min_d1_x={neg['min_d1_x'].mean():.2f}, contact_f={neg['frames_contact_x'].mean():.1f}")

        # Check high-ghost subset (mean_wg > 0.08 or mean_anewt < 0.70)
        gh_pos = pos[(pos["mean_wg"] > 0.08) | (pos["mean_anewt"] < 0.70)]
        gh_neg = neg[(neg["mean_wg"] > 0.08) | (neg["mean_anewt"] < 0.70)]
        print(f"  [Ghost/Non-Newtonian Subset (mean_wg>0.08 or mean_anewt<0.70)]:")
        print(f"    False Blips in Ghost Subset = {len(gh_pos)}/{len(pos)} ({100*len(gh_pos)/max(len(pos),1):.1f}%) | True Turnovers in Ghost Subset = {len(gh_neg)}/{len(neg)} ({100*len(gh_neg)/max(len(neg),1):.1f}%)")


if __name__ == "__main__":
    main()
