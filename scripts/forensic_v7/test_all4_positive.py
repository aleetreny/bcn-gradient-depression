"""Tune physical thresholds of Fix A, Fix B, Fix C, Fix D on LOMO so all 4 Judges improve simultaneously."""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "2"
warnings.filterwarnings("ignore")

SCRATCH = Path(__file__).resolve().parents[2] / "experiments" / "outputs" / "forensic_v7"
sys.path.insert(0, str(SCRATCH))

import prototype_v7_forensic_upgrades as proto
from src.data.common import AWAY, DEAD, HOME
from src.decode.spell_reranker import rle_runs


def apply_dead_wedge_v2(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame,
    p_change: np.ndarray,
    p4d: np.ndarray,
) -> np.ndarray:
    """Clean, high-precision Dead-Ball Wedge & Island Arbitrator (`Y->X->D`, `D->X->Y`, `D->X->D`).
    100% H<->A equivariant.
    """
    out = np.asarray(pred, dtype=np.int8).copy()
    valid = (half == 1) | (half == 2)
    arr = np.where(valid, out, -1)
    runs = rle_runs(arr)

    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=50.0)
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=1.0) if "ball_visible" in X_df else np.ones(len(out))

    for idx in range(1, len(runs) - 1):
        s, e, x = runs[idx]
        if x not in (HOME, AWAY):
            continue
        ps, pe, prev_y = runs[idx - 1]
        ns, ne, next_y = runs[idx + 1]
        if half[s] != half[ps] or half[s] != half[ns]:
            continue
        L = e - s
        opp = AWAY if x == HOME else HOME

        mean_pX = float(np.mean(proba[s:e, x]))
        max_pX = float(np.max(proba[s:e, x]))
        mean_pD = float(np.mean(proba[s:e, DEAD]))
        max_pD = float(np.max(proba[s:e, DEAD]))
        d1_x = d1h[s:e] if x == HOME else d1a[s:e]
        min_d1_x = float(np.min(d1_x))
        mean_vis = float(np.mean(vis[s:e]))

        # 1. Y -> X -> D (False live tail before DEAD)
        if prev_y == opp and next_y == DEAD and L <= 90:
            dir_yx = 0 if (opp == HOME and x == AWAY) else 1
            b4d_in = float(np.max(p4d[max(0, s - 6) : min(len(out), s + 7), dir_yx]))
            cond_d_spill = (mean_pX < 0.62 and (mean_pD > 0.20 or max_pD > 0.50) and b4d_in < 0.25)
            cond_no_ctl = (min_d1_x > 2.8 and mean_vis > 0.6 and mean_pX < 0.68 and b4d_in < 0.45)
            cond_weak_x = (mean_pX < 0.54 and max_pX < 0.72 and b4d_in < 0.35)
            if cond_d_spill or cond_no_ctl or cond_weak_x:
                p_y_seg = proba[s:e, opp]
                p_d_seg = proba[s:e, DEAD]
                if float(np.max(p_d_seg)) > 0.45 and mean_pD > 0.20:
                    out[s:e] = DEAD
                else:
                    diff = np.log(np.clip(p_y_seg, 1e-5, 1.0)) - np.log(np.clip(p_d_seg, 1e-5, 1.0))
                    c_sum = np.concatenate([[0.0], np.cumsum(diff)])
                    k_best = int(np.argmax(c_sum))
                    out[s : s + k_best] = opp
                    out[s + k_best : e] = DEAD

        # 2. D -> X -> Y (False live prefix after DEAD)
        elif prev_y == DEAD and next_y == opp and L <= 85:
            cond_no_ctl_dxy = (min_d1_x > 3.2 and mean_vis > 0.6 and mean_pX < 0.74)
            cond_weak_dxy = (mean_pX < 0.56 and max_pX < 0.72)
            if cond_no_ctl_dxy or cond_weak_dxy:
                p_d_seg = proba[s:e, DEAD]
                p_y_seg = proba[s:e, opp]
                diff = np.log(np.clip(p_d_seg, 1e-5, 1.0)) - np.log(np.clip(p_y_seg, 1e-5, 1.0))
                c_sum = np.concatenate([[0.0], np.cumsum(diff)])
                k_best = int(np.argmax(c_sum))
                out[s : s + k_best] = DEAD
                out[s + k_best : e] = opp

        # 3. D -> X -> D
        elif prev_y == DEAD and next_y == DEAD and L <= 95:
            if (mean_pX < 0.62 and mean_pD > 0.12) or (min_d1_x > 4.0 and mean_pX < 0.72):
                out[s:e] = DEAD

    return out


def apply_short_turnover_v2(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame,
    p_change: np.ndarray,
    p4d: np.ndarray,
    min_parent_len: int = 220,
    margin_edges: int = 55,
) -> np.ndarray:
    """High-precision Short-Turnover Recovery: only split when BOTH probability and physical deflection confirm."""
    out = np.asarray(pred, dtype=np.int8).copy()
    valid = (half == 1) | (half == 2)
    arr = np.where(valid, out, -1)
    runs = rle_runs(arr)

    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=50.0)
    imp = np.maximum(
        np.nan_to_num(X_df["impulse_mag_d2"].to_numpy(np.float64), nan=0.0),
        np.nan_to_num(X_df["impulse_mag_d4"].to_numpy(np.float64), nan=0.0),
    )
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=0.0)

    for s, e, y in runs:
        if y not in (HOME, AWAY):
            continue
        L = e - s
        if L < min_parent_len:
            continue
        x = AWAY if y == HOME else HOME
        d1_x = d1h if x == HOME else d1a
        d1_y = d1a if x == HOME else d1h
        dir_yx = 0 if (y == HOME and x == AWAY) else 1
        dir_xy = 1 if dir_yx == 0 else 0

        a_int = s + margin_edges
        b_int = e - margin_edges
        if b_int - a_int < 24:
            continue

        p_x = proba[:, x]
        p_y = proba[:, y]
        p_d = proba[:, DEAD]

        t_cursor = a_int
        while t_cursor < b_int - 18:
            win_end = min(b_int, t_cursor + 40)
            seg_px = p_x[t_cursor:win_end]
            seg_b4d = p4d[t_cursor:win_end, dir_yx]

            # Ultra-strict Criterion A: Sustained probability majority (`max(p_x) >= 0.62`, `n_maj >= 11`, `mean_p_x >= 0.46`)
            # with clear entry and exit change signals and close physical contact
            if float(np.max(seg_px)) >= 0.62:
                pk = t_cursor + int(np.argmax(seg_px))
                ks = pk
                while ks > a_int and p_x[ks - 1] >= 0.36 and (pk - ks) < 25:
                    ks -= 1
                ke = pk + 1
                while ke < b_int and p_x[ke] >= 0.36 and (ke - ks) < 40:
                    ke += 1
                if ke - ks < 24:
                    pad = (24 - (ke - ks)) // 2
                    ks = max(a_int, ks - pad)
                    ke = min(b_int, ks + 24)

                dur = ke - ks
                n_maj = int(np.sum(p_x[ks:ke] > p_y[ks:ke]))
                min_dx = float(np.min(d1_x[ks:ke]))
                min_dy_at_pk = float(d1_y[pk])
                max_im = float(np.max(imp[ks:ke]))
                mean_v = float(np.mean(vis[ks:ke]))
                max_c = float(np.max(p_change[max(0, ks - 4) : min(len(out), ke + 4)]))
                b_in = float(np.max(p4d[max(0, ks - 6) : min(len(out), ks + 10), dir_yx]))
                b_out = float(np.max(p4d[max(0, ke - 10) : min(len(out), ke + 6), dir_xy]))

                if (
                    dur >= 20
                    and n_maj >= 11
                    and float(np.mean(p_x[ks:ke])) >= 0.46
                    and float(np.mean(p_d[ks:ke])) < 0.04
                    and min_dx <= 1.65
                    and min_dy_at_pk >= 2.2
                    and max_im >= 2.1
                    and max_c >= 0.19
                    and (b_in >= 0.14 or b_out >= 0.12)
                    and mean_v >= 0.92
                ):
                    out[ks:ke] = x
                    t_cursor = ke + margin_edges
                    continue

            # Ultra-strict Criterion B: Sharp directional 4D boundary + impulse deflection (`b4d >= 0.58`, `p_change >= 0.40`)
            if float(np.max(seg_b4d)) >= 0.58:
                pk_b = t_cursor + int(np.argmax(seg_b4d))
                ks = max(a_int, pk_b - 8)
                ke = min(b_int, ks + 22)
                min_dx = float(np.min(d1_x[ks:ke]))
                d_y_after = float(np.mean(d1_y[pk_b + 3 : min(len(out), pk_b + 16)]))
                max_im = float(np.max(imp[ks:ke]))
                max_c = float(np.max(p_change[ks:ke]))
                mean_v = float(np.mean(vis[ks:ke]))
                b_out = float(np.max(p4d[pk_b + 5 : min(len(out), ke + 12), dir_xy]))
                if (
                    min_dx <= 0.90
                    and d_y_after >= 3.2
                    and max_im >= 2.6
                    and max_c >= 0.40
                    and b_out >= 0.15
                    and float(np.max(p_x[ks:ke])) >= 0.30
                    and float(np.mean(p_d[ks:ke])) < 0.04
                    and mean_v >= 0.95
                ):
                    out[ks:ke] = x
                    t_cursor = ke + margin_edges
                    continue

            t_cursor += 15

    return out


if __name__ == "__main__":
    proto.apply_dead_wedge_arbitrator = apply_dead_wedge_v2
    proto.apply_short_turnover_recovery = apply_short_turnover_v2
    proto.main()
