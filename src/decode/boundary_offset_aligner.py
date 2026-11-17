"""EXP-16: Supervised Sub-Second Boundary Offset Aligner (`SymBoundaryOffsetAligner`).

Resolves the +/-2 to +/-12 frame (0.08s - 0.48s) lag in decoded possession boundaries caused by
symmetric tree smoothing and causal/bidirectional sequence receptive fields:
- Even when a spell's team identity is right, a +/-6-8 frame boundary offset on a 26-45 frame
  spell pushes its IoU across the 0.60 / 0.70 / 0.80 thresholds of `spell_f1_mean`.
- Unlike heuristic `ImpulseSnapConfig(window=5)` (which greedily snaps to the nearest raw `impulse_mag_d2`
  peak within +/-5 frames and can snap to pass-reception or bounce kinks), `SymBoundaryOffsetAligner`
  evaluates every candidate offset `d in [-12, +12]` around each decoded boundary `t_b` (`HA`, `L2D`, `D2L`)
  using a 26-D 100% `H <-> A`-invariant offset feature vector:
  1. Multi-family probability step & segment probability gain over the shifted interval `[t_b + d, t_b)` or `[t_b, t_b + d)`:
     - `dp_step4`, `dp_step8`, `seg_prob_gain`, `seg_tree_gain`, `seg_seq_gain` (`SymTCN + SymBiGRU`),
       `seg_pset_gain` (`SymPlayerSetNet`), `p_diff_at_tc`, `tree_diff_at_tc`.
  2. Directional transition & change heads at `t_b + d`:
     - `b4d_dir_tc`, `b4d_dir_max1`, `b4d_rev_tc`, `pch_tc`, `pch_max1`.
  3. Anisotropic ball kinematics & visibility at `t_b + d`:
     - `imp_d2_tc`, `imp_d4_tc`, `imp_norm_d4_tc`, `acc_tc`, `ang_tc`, `vis_tc`.
  4. Player-ball proximity & contact geometry at `t_b + d`:
     - `d1_new_tc`, `d1_old_tc`, `d1_cross_step`, `contact_new_tc`.
  5. Pitch geometry & offset regularization:
     - `min_line_tc`, `out_pitch_step`, `offset_pen = |d| / 12.0`.

Mathematical `H <-> A` Equivariance (`0.000e+00`):
- Time `t` flows forward identically under `H <-> A` label/feature swap while `s_old <-> s_new` swap roles.
- Because all 26 features at offset `d` are indexed strictly by `(s_old, s_new)` (outgoing vs incoming
  state at boundary `t_b`), the feature matrix `feats[d]` (`25 x 26`) is 100% invariant (`0.000e+00`)
  under `H <-> A` reflection.
- Consequently, the selected offset `d* in [-12, +12]` is identical under `H <-> A` swap, and shifting
  `t_b -> t_b + d*` commutes with `swap_labels` (`mismatch_frames = 0`).

Strict Zero-Degradation Safety Gate:
1. Never shifts a boundary if doing so would shrink either adjacent spell below `min_spell_len` frames
   (or if either adjacent spell has length `< min_adj_spell_len = 26`, protecting newly inserted short
   single-touch spells from `EXP-15`).
2. Never shifts a boundary into or out of an invisible-ball interpolation gap (`ball_visible(t_b) == 1`
   and `ball_visible(t_b + d) == 1`).
3. Nested inner-LOMO selection chooses between the clean-sequence offset expert (`seq_gain`) and the
   supervised Ridge+Physics offset regressor (`ridge_phys`), calibrated so that zero inner-fold matches
   are degraded (`neg_matches == 0`) before any shift is applied on the held-out match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

import evaluation.possession_metrics as pm
from src.data.common import AWAY, DEAD, HOME
from src.decode.touch_type_arbitrator import (
    _get_col,
    preclean_tracking_arrays,
    rle_runs_half as _rle_runs_half,
)
from src.eval.metrics import to_official
from src.features.symmetry import swap_labels, swap_proba


def _extract_kinematic_arrays(
    xv: pd.DataFrame | dict[str, np.ndarray],
    T: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract and sanitize kinematic & pitch-line arrays for `SymBoundaryOffsetAligner`."""
    if isinstance(xv, dict) and "dH1" in xv and "dH1_ig" in xv and "touchline" in xv:
        return xv
    c = dict(preclean_tracking_arrays(xv, T))
    if T is None:
        T = len(c["vis"])
    c["dH1"] = c["d1_h"]
    c["dA1"] = c["d1_a"]
    c["dH1_ig"] = c["d1_ig_h"]
    c["dA1_ig"] = c["d1_ig_a"]
    tl_raw = _get_col(xv, "ball_last_to_touchline", T, np.nan)
    if np.all(np.isnan(tl_raw)):
        tl_raw = _get_col(xv, "dist_to_touchline", T, 20.0)
    gl_raw = _get_col(xv, "ball_last_to_goalline", T, np.nan)
    if np.all(np.isnan(gl_raw)):
        gl_raw = _get_col(xv, "dist_to_goalline", T, 30.0)
    c["touchline"] = np.clip(np.nan_to_num(tl_raw, nan=20.0), -5.0, 40.0)
    c["goalline"] = np.clip(np.nan_to_num(gl_raw, nan=30.0), -5.0, 60.0)
    return c


def _swap_kinematic_arrays(cxv: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Swap HOME <-> AWAY channels inside a kinematic dictionary."""
    sw = dict(cxv)
    for kh, ka in (
        ("d1_h", "d1_a"),
        ("d1_ig_h", "d1_ig_a"),
        ("d1_h_vis", "d1_a_vis"),
        ("dH1", "dA1"),
        ("dH1_ig", "dA1_ig"),
        ("nk_h", "nk_a"),
        ("pk_h", "pk_a"),
        ("cs_h", "cs_a"),
    ):
        if kh in sw and ka in sw:
            sw[kh], sw[ka] = cxv[ka].copy(), cxv[kh].copy()
    return sw


DELTAS: np.ndarray = np.arange(-12, 13, dtype=np.int32)  # 25 offsets, index 12 is d = 0

BOUNDARY_OFFSET_FEATURE_NAMES: tuple[str, ...] = (
    "dp_step4",
    "dp_step8",
    "seg_prob_gain",
    "seg_tree_gain",
    "seg_seq_gain",
    "seg_pset_gain",
    "p_diff_at_tc",
    "tree_diff_at_tc",
    "b4d_dir_tc",
    "b4d_dir_max1",
    "b4d_rev_tc",
    "pch_tc",
    "pch_max1",
    "imp_d2_tc",
    "imp_d4_tc",
    "imp_norm_d4_tc",
    "acc_tc",
    "ang_tc",
    "vis_tc",
    "d1_new_tc",
    "d1_old_tc",
    "d1_cross_step",
    "contact_new_tc",
    "min_line_tc",
    "out_pitch_step",
    "offset_pen",
)

BDY_RIDGE_SUB_IDX: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25
)


@dataclass(frozen=True)
class BoundaryOffsetAlignerConfig:
    """Configuration for `SymBoundaryOffsetAligner` (`EXP-16`)."""

    max_delta: int = 12
    min_spell_len: int = 20
    min_adj_spell_len: int = 26
    min_sep_frames: int = 25
    ridge_l2: float = 250.0
    offset_pen_seq: float = 0.03
    offset_pen_phys: float = 0.04
    min_th_seq: float = 0.040
    min_th_ridge: float = 0.005
    min_inner_f1_gain_ha: float = 0.00030
    min_inner_f1_gain_ld: float = 0.00035
    enable_ha: bool = True
    enable_l2d: bool = False
    enable_d2l: bool = False


def _split_family_Summaries(
    proba: np.ndarray,
    family_probas: Sequence[np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tree mean, sequence/tabular mean (`fams[3:8]`), and `SymPlayerSetNet` probabilities."""
    P = np.asarray(proba, dtype=np.float64)
    if family_probas is None or len(family_probas) == 0:
        return P, P, P
    fams = [np.asarray(f, dtype=np.float64) for f in family_probas]
    n_f = len(fams)
    if n_f >= 9:
        p_tree = (fams[0] + fams[1] + fams[2]) / 3.0
        p_seq = (fams[3] + fams[4] + fams[5] + fams[6] + fams[7]) / 5.0
        p_pset = fams[8]
    elif n_f == 8:
        p_tree = (fams[0] + fams[1] + fams[2]) / 3.0
        p_seq = (fams[3] + fams[4] + fams[5] + fams[6]) / 4.0
        p_pset = fams[7]
    else:
        p_tree = np.mean(fams[: min(3, n_f)], axis=0)
        p_seq = np.mean(fams[min(3, n_f - 1) :], axis=0)
        p_pset = fams[-1]
    return p_tree, p_seq, p_pset


def build_offset_features(
    tb: int,
    btype: str,
    s_old: int,
    s_new: int,
    proba: np.ndarray,
    p_change: np.ndarray,
    p_trans_4d: np.ndarray | None,
    cxv: dict[str, np.ndarray],
    p_tree: np.ndarray,
    p_seq: np.ndarray,
    p_pset: np.ndarray,
) -> np.ndarray:
    """Compute `(25, 26)` H<->A-invariant feature matrix across `DELTAS = [-12..+12]` for boundary `tb`."""
    T = len(proba)
    feats = np.zeros((len(DELTAS), len(BOUNDARY_OFFSET_FEATURE_NAMES)), dtype=np.float64)

    p_old = proba[:, s_old]
    p_new = proba[:, s_new]
    p_diff = p_new - p_old

    pt_diff = p_tree[:, s_new] - p_tree[:, s_old]
    ps_diff = p_seq[:, s_new] - p_seq[:, s_old]
    pp_diff = p_pset[:, s_new] - p_pset[:, s_old]

    if p_trans_4d is not None:
        if btype == "HA":
            b4d_dir = p_trans_4d[:, 0] if s_new == HOME else p_trans_4d[:, 1]
            b4d_rev = p_trans_4d[:, 1] if s_new == HOME else p_trans_4d[:, 0]
        elif btype == "L2D":
            b4d_dir = p_trans_4d[:, 2]
            b4d_rev = p_trans_4d[:, 3]
        else:  # D2L
            b4d_dir = p_trans_4d[:, 3]
            b4d_rev = p_trans_4d[:, 2]
    else:
        b4d_dir = p_change
        b4d_rev = np.zeros_like(p_change)

    if btype == "HA":
        d1_new = cxv["dH1"] if s_new == HOME else cxv["dA1"]
        d1_ig_new = cxv["dH1_ig"] if s_new == HOME else cxv["dA1_ig"]
        d1_ig_old = cxv["dH1_ig"] if s_old == HOME else cxv["dA1_ig"]
    elif btype == "L2D":
        d1_new = cxv["dA1"] if s_old == HOME else cxv["dH1"]
        d1_ig_old = cxv["dH1_ig"] if s_old == HOME else cxv["dA1_ig"]
        d1_ig_new = cxv["dA1_ig"] if s_old == HOME else cxv["dH1_ig"]
    else:  # D2L
        d1_new = cxv["dH1"] if s_new == HOME else cxv["dA1"]
        d1_ig_new = cxv["dH1_ig"] if s_new == HOME else cxv["dA1_ig"]
        d1_ig_old = cxv["dA1_ig"] if s_new == HOME else cxv["dH1_ig"]

    for idx_d, d in enumerate(DELTAS):
        tc = int(np.clip(tb + int(d), 4, T - 5))
        dp_step4 = float(np.mean(p_diff[tc : tc + 4]) - np.mean(p_diff[tc - 4 : tc]))
        w8_l = max(0, tc - 8)
        w8_r = min(T, tc + 8)
        dp_step8 = float(np.mean(p_diff[tc:w8_r]) - np.mean(p_diff[w8_l:tc]))

        if d == 0:
            seg_prob_gain = 0.0
            seg_tree_gain = 0.0
            seg_seq_gain = 0.0
            seg_pset_gain = 0.0
        elif d < 0:
            sl = slice(max(0, tb + int(d)), tb)
            seg_prob_gain = float(np.sum(p_diff[sl])) / 12.0
            seg_tree_gain = float(np.sum(pt_diff[sl])) / 12.0
            seg_seq_gain = float(np.sum(ps_diff[sl])) / 12.0
            seg_pset_gain = float(np.sum(pp_diff[sl])) / 12.0
        else:
            sl = slice(tb, min(T, tb + int(d)))
            seg_prob_gain = float(np.sum(-p_diff[sl])) / 12.0
            seg_tree_gain = float(np.sum(-pt_diff[sl])) / 12.0
            seg_seq_gain = float(np.sum(-ps_diff[sl])) / 12.0
            seg_pset_gain = float(np.sum(-pp_diff[sl])) / 12.0

        p_diff_at_tc = float(p_diff[tc])
        tree_diff_at_tc = float(pt_diff[tc])

        b4d_dir_tc = float(b4d_dir[tc])
        b4d_dir_max1 = float(np.max(b4d_dir[tc - 1 : tc + 2]))
        b4d_rev_tc = float(b4d_rev[tc])
        pch_tc = float(p_change[tc])
        pch_max1 = float(np.max(p_change[tc - 1 : tc + 2]))

        imp_d2_tc = float(cxv["imp_d2"][tc]) / 30.0
        imp_d4_tc = float(cxv["imp_d4"][tc]) / 30.0
        imp_norm_d4_tc = float(cxv["imp_norm_d4"][tc]) / 30.0
        acc_tc = float(np.abs(cxv["bacc"][tc])) / 40.0
        ang_tc = float(cxv["bang"][tc]) / 3.1416
        vis_tc = float(cxv["vis"][tc])

        d1_new_tc = float(d1_ig_new[tc]) / 30.0
        d1_old_tc = float(d1_ig_old[tc]) / 30.0
        d1_cross_step = (
            float(np.mean(d1_ig_old[tc : tc + 4] - d1_ig_new[tc : tc + 4]))
            - float(np.mean(d1_ig_old[tc - 4 : tc] - d1_ig_new[tc - 4 : tc]))
        ) / 20.0
        contact_new_tc = 1.0 if (vis_tc > 0.5 and float(d1_new[tc]) < 1.35) else 0.0

        min_line_tc = min(float(cxv["touchline"][tc]), float(cxv["goalline"][tc])) / 30.0
        out_pitch_step = float(np.mean(cxv["out_pitch"][tc : tc + 4]) - np.mean(cxv["out_pitch"][tc - 4 : tc]))
        offset_pen = abs(float(d)) / 12.0

        feats[idx_d] = np.round(
            [
                dp_step4, dp_step8, seg_prob_gain, seg_tree_gain, seg_seq_gain, seg_pset_gain,
                p_diff_at_tc, tree_diff_at_tc,
                b4d_dir_tc, b4d_dir_max1, b4d_rev_tc, pch_tc, pch_max1,
                imp_d2_tc, imp_d4_tc, imp_norm_d4_tc, acc_tc, ang_tc, vis_tc,
                d1_new_tc, d1_old_tc, d1_cross_step, contact_new_tc,
                min_line_tc, out_pitch_step, offset_pen,
            ],
            10,
        )
    return feats


def _f1_from_ov(ov: list[tuple[float, int, int]], n_gt: int, n_pr: int) -> float:
    f1s = []
    for th in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(ov, n_gt, n_pr, th)
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return float(np.mean(f1s))


def _single_spell_overlaps(
    gt_spells: list[tuple[int, int, int]],
    sp: tuple[int, int, int],
    pred_idx: int,
) -> list[tuple[float, int, int]]:
    a_p, b_p, c_p = sp
    out = []
    for i, (a_g, b_g, c_g) in enumerate(gt_spells):
        if c_p != c_g or b_p <= a_g or b_g <= a_p:
            continue
        inter = min(b_g, b_p) - max(a_g, a_p)
        uni = max(b_g, b_p) - min(a_g, a_p)
        out.append((inter / float(uni), i, pred_idx))
    return out


def extract_boundary_candidates(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X: pd.DataFrame,
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
    family_probas: Sequence[np.ndarray] | None = None,
    labels: np.ndarray | None = None,
    match_id: str = "",
    cfg: BoundaryOffsetAlignerConfig | None = None,
) -> list[dict]:
    """Extract candidate transition boundaries (`HA`, `L2D`, `D2L`) and 26-D offset feature matrices.

    If `labels` is provided (training time only), also computes exact `f1_deltas`, `iou_deltas`, and
    `frame_deltas` for each offset `d in [-12..+12]` using incremental bipartite spell overlap updates.
    """
    if cfg is None:
        cfg = BoundaryOffsetAlignerConfig()
    T = len(pred)
    P = np.asarray(proba, dtype=np.float64)
    p_ch = (
        np.asarray(p_change, dtype=np.float64)
        if p_change is not None
        else np.zeros(T, dtype=np.float64)
    )
    p4d = np.asarray(p_trans_4d, dtype=np.float64) if p_trans_4d is not None else None
    cxv = _extract_kinematic_arrays(X, T)
    p_tree, p_seq, p_pset = _split_family_Summaries(P, family_probas)

    # Optional training-time ground-truth overlap structures
    if labels is not None:
        gt_off = to_official(labels, half, True)
        runs = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, runs)
        p_off = to_official(pred, half, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, runs)
        base_ov = pm.spell_overlaps(gt_spells, base_spells)
        n_gt, n_pr = len(gt_spells), len(base_spells)
        base_f1 = _f1_from_ov(base_ov, n_gt, n_pr)
        start_map = {sp[0]: idx for idx, sp in enumerate(base_spells)}
        end_map = {sp[1]: idx for idx, sp in enumerate(base_spells)}
    else:
        gt_spells = []
        base_spells = []
        base_ov = []
        n_gt = n_pr = 0
        base_f1 = 0.0
        start_map = {}
        end_map = {}

    r = _rle_runs_half(pred, half)
    out: list[dict] = []
    for k in range(len(r) - 1):
        a1, b1, s_old, h1 = r[k]
        a2, b2, s_new, h2 = r[k + 1]
        if h1 <= 0 or h1 != h2 or b1 != a2:
            continue
        tb = b1
        L_left = b1 - a1
        L_right = b2 - a2
        if L_left < cfg.min_spell_len or L_right < cfg.min_spell_len:
            continue
        if s_old in (HOME, AWAY) and s_new in (HOME, AWAY):
            btype = "HA"
        elif s_old in (HOME, AWAY) and s_new == DEAD:
            btype = "L2D"
        elif s_old == DEAD and s_new in (HOME, AWAY):
            btype = "D2L"
        else:
            continue

        feats = build_offset_features(
            tb, btype, s_old, s_new, P, p_ch, p4d, cxv, p_tree, p_seq, p_pset
        )
        valid_mask = np.ones(len(DELTAS), dtype=bool)
        vis_tb = float(cxv["vis"][tb]) > 0.5
        gd_tb = float(cxv["gdisc"][tb]) <= 22.0

        for idx_d, d in enumerate(DELTAS):
            tc = tb + int(d)
            if tc < 4 or tc >= T - 5:
                valid_mask[idx_d] = False
                continue
            if L_left + int(d) < cfg.min_adj_spell_len or L_right - int(d) < cfg.min_adj_spell_len:
                valid_mask[idx_d] = False
                continue
            vis_tc = float(cxv["vis"][tc]) > 0.5
            gd_tc = float(cxv["gdisc"][tc]) <= 22.0
            if ((not vis_tb) or (not vis_tc) or (not gd_tb) or (not gd_tc)) and d != 0:
                valid_mask[idx_d] = False

        cand = {
            "m": match_id,
            "tb": int(tb),
            "btype": btype,
            "s_old": int(s_old),
            "s_new": int(s_new),
            "L_left": int(L_left),
            "L_right": int(L_right),
            "vis_tb": bool(vis_tb),
            "feats": feats,
            "valid_mask": valid_mask,
        }

        if labels is not None:
            f1_deltas = np.zeros(len(DELTAS), dtype=np.float64)
            iou_deltas = np.zeros(len(DELTAS), dtype=np.float64)
            frame_deltas = np.zeros(len(DELTAS), dtype=np.float64)

            shift = 1 if h1 == 2 else 0
            tb_s = tb + shift
            idx_left = end_map.get(tb_s, None)
            idx_right = start_map.get(tb_s, None)
            rem_set = {x for x in (idx_left, idx_right) if x is not None}
            ov_without = [item for item in base_ov if item[2] not in rem_set]
            ov_orig_lr = [item for item in base_ov if item[2] in rem_set]
            orig_iou_sum = sum(item[0] for item in ov_orig_lr)

            for idx_d, d in enumerate(DELTAS):
                if not valid_mask[idx_d] or d == 0:
                    continue
                if d < 0:
                    sl = slice(tb + int(d), tb)
                    frame_deltas[idx_d] = float(
                        np.sum(labels[sl] == s_new) - np.sum(labels[sl] == s_old)
                    )
                else:
                    sl = slice(tb, tb + int(d))
                    frame_deltas[idx_d] = float(
                        np.sum(labels[sl] == s_old) - np.sum(labels[sl] == s_new)
                    )

                new_lr: list[tuple[float, int, int]] = []
                if btype == "HA" and idx_left is not None and idx_right is not None:
                    sp_l = (base_spells[idx_left][0], tb_s + int(d), base_spells[idx_left][2])
                    sp_r = (tb_s + int(d), base_spells[idx_right][1], base_spells[idx_right][2])
                    new_lr = _single_spell_overlaps(gt_spells, sp_l, idx_left) + _single_spell_overlaps(
                        gt_spells, sp_r, idx_right
                    )
                elif btype == "L2D" and idx_left is not None:
                    sp_l = (base_spells[idx_left][0], tb_s + int(d), base_spells[idx_left][2])
                    new_lr = _single_spell_overlaps(gt_spells, sp_l, idx_left)
                elif btype == "D2L" and idx_right is not None:
                    sp_r = (tb_s + int(d), base_spells[idx_right][1], base_spells[idx_right][2])
                    new_lr = _single_spell_overlaps(gt_spells, sp_r, idx_right)

                iou_deltas[idx_d] = sum(item[0] for item in new_lr) - orig_iou_sum
                if abs(iou_deltas[idx_d]) > 1e-9:
                    ov_mod = ov_without + new_lr
                    ov_mod.sort(key=lambda item: (-item[0], item[1], item[2]))
                    f1_deltas[idx_d] = _f1_from_ov(ov_mod, n_gt, n_pr) - base_f1

            cand["f1_deltas"] = f1_deltas
            cand["iou_deltas"] = iou_deltas
            cand["frame_deltas"] = frame_deltas

        out.append(cand)
    return out


def compute_phys_offset_scores(feats: np.ndarray, btype: str, pen: float = 0.04) -> np.ndarray:
    """Compute 25-D physical + multi-family consensus advantage over `d = 0` (`index 12`)."""
    f0 = feats[12]
    df = feats - f0[None, :]
    d_abs = feats[:, 25]

    if btype == "HA":
        s_cons = (
            0.35 * df[:, 4]
            + 0.25 * df[:, 3]
            + 0.20 * df[:, 5]
            + 0.20 * df[:, 2]
        )
        s_phys = (
            0.30 * df[:, 13]
            + 0.20 * df[:, 14]
            + 0.25 * df[:, 22]
            - 0.25 * df[:, 19]
        )
        s_bdy = 0.50 * df[:, 8] + 0.50 * df[:, 11]
        sc = 0.55 * s_cons + 0.30 * s_phys + 0.15 * s_bdy - pen * d_abs
    elif btype == "L2D":
        s_cons = (
            0.35 * df[:, 3]
            + 0.25 * df[:, 4]
            + 0.20 * df[:, 5]
            + 0.20 * df[:, 2]
        )
        s_phys = (
            0.35 * df[:, 14]
            + 0.25 * df[:, 13]
            + 0.20 * df[:, 24]
            - 0.20 * df[:, 23]
        )
        s_bdy = 0.60 * df[:, 8] + 0.40 * df[:, 11]
        sc = 0.50 * s_cons + 0.30 * s_phys + 0.20 * s_bdy - pen * d_abs
    else:  # D2L
        s_cons = (
            0.35 * df[:, 3]
            + 0.25 * df[:, 4]
            + 0.20 * df[:, 5]
            + 0.20 * df[:, 2]
        )
        s_phys = (
            0.35 * df[:, 22]
            - 0.35 * df[:, 19]
            + 0.30 * df[:, 21]
        )
        s_bdy = 0.60 * df[:, 8] + 0.40 * df[:, 11]
        sc = 0.50 * s_cons + 0.30 * s_phys + 0.20 * s_bdy - pen * d_abs

    sc = np.round(sc, 10)
    sc[12] = 0.0
    return sc


def _fit_bdy_ridge(
    b_list: Sequence[dict],
    sub_idx: Sequence[int] = BDY_RIDGE_SUB_IDX,
    l2: float = 250.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X_rows = []
    y_rows = []
    w_rows = []
    for b in b_list:
        f0 = b["feats"][12]
        for idx_d, d in enumerate(DELTAS):
            if not b["valid_mask"][idx_d] or d == 0:
                continue
            df_d = b["feats"][idx_d] - f0
            f1_d = float(b["f1_deltas"][idx_d])
            iou_d = float(b["iou_deltas"][idx_d])
            frm_d = float(b["frame_deltas"][idx_d]) / 12.0
            target = (
                0.55 * (1.0 if f1_d > 1e-5 else (-1.0 if f1_d < -1e-5 else 0.0))
                + 0.25 * np.clip(iou_d * 10.0, -1.0, 1.0)
                + 0.20 * np.clip(frm_d, -1.0, 1.0)
            )
            wt = 1.0 + 400.0 * abs(f1_d) + 20.0 * abs(iou_d)
            X_rows.append(df_d[list(sub_idx)])
            y_rows.append(target)
            w_rows.append(wt)
    if len(X_rows) < 10:
        d_sub = len(sub_idx)
        return np.zeros(d_sub), np.ones(d_sub), np.zeros(d_sub), 0.0
    X_tr = np.asarray(X_rows, dtype=np.float64)
    y_tr = np.asarray(y_rows, dtype=np.float64)
    w_tr = np.asarray(w_rows, dtype=np.float64)
    mu = np.average(X_tr, axis=0, weights=w_tr)
    sd = np.sqrt(np.maximum(np.average((X_tr - mu) ** 2, axis=0, weights=w_tr), 1e-6))
    Z = np.clip((X_tr - mu) / sd, -6.0, 6.0)
    ym = float(np.average(y_tr, weights=w_tr))
    yc = y_tr - ym
    sw = np.sqrt(w_tr)
    Zw = Z * sw[:, None]
    yw = yc * sw
    coef = np.linalg.solve(Zw.T @ Zw + l2 * np.eye(X_tr.shape[1]), Zw.T @ yw)
    return mu, sd, coef, ym


def _score_bdy_ridge(
    feats: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    sub_idx: Sequence[int] = BDY_RIDGE_SUB_IDX,
    pen: float = 0.04,
) -> np.ndarray:
    mu, sd, coef, ym = model
    f0 = feats[12]
    df = (feats - f0[None, :])[:, list(sub_idx)]
    Z = np.clip((df - mu) / sd, -6.0, 6.0)
    sc = (Z @ coef + ym) - pen * feats[:, 25]
    sc = np.round(sc, 10)
    sc[12] = 0.0
    return sc


def _score_offset_expert(
    feats: np.ndarray,
    btype: str,
    exp_name: str,
    rdg_sc: np.ndarray | None = None,
    cfg: BoundaryOffsetAlignerConfig | None = None,
) -> np.ndarray:
    if cfg is None:
        cfg = BoundaryOffsetAlignerConfig()
    d_abs = feats[:, 25]
    if exp_name == "seq_gain":
        sc = feats[:, 4] - cfg.offset_pen_seq * d_abs
    elif exp_name == "ridge_phys":
        p_sc = compute_phys_offset_scores(feats, btype, pen=cfg.offset_pen_phys)
        sc = 0.50 * p_sc + 0.50 * (rdg_sc if rdg_sc is not None else p_sc)
    else:
        sc = compute_phys_offset_scores(feats, btype, pen=cfg.offset_pen_phys)
    sc = np.round(sc.copy(), 10)
    sc[12] = 0.0
    return sc


def _get_d_mask(d_mode: str) -> np.ndarray:
    if d_mode == "small4":
        return (np.abs(DELTAS) <= 4) & (DELTAS != 0)
    if d_mode == "small6":
        return (np.abs(DELTAS) <= 6) & (DELTAS != 0)
    if d_mode == "small8":
        return (np.abs(DELTAS) <= 8) & (DELTAS != 0)
    if d_mode == "small12":
        return (np.abs(DELTAS) <= 12) & (DELTAS != 0)
    return (np.abs(DELTAS) <= 8) & (DELTAS != 0)


def _candidate_phys_valid_mask(b: dict, cfg: BoundaryOffsetAlignerConfig) -> np.ndarray:
    """Enforce post-shift minimum spell length (`min_adj_spell_len`) and incoming player proximity."""
    post_ok = ((b["L_left"] + DELTAS) >= cfg.min_adj_spell_len) & (
        (b["L_right"] - DELTAS) >= cfg.min_adj_spell_len
    )
    f = b["feats"]
    d1_ok = ((f[:, 19] * 30.0) <= 4.0) | (f[:, 19] <= f[12, 19] + 0.35 / 30.0)
    return b["valid_mask"] & post_ok & d1_ok


def apply_boundary_shifts(
    pred: np.ndarray,
    shifts: Sequence[tuple[float, dict, int]],
    min_sep_frames: int = 25,
) -> np.ndarray:
    """Apply non-overlapping boundary shifts `(score, b, idx_d)` sorted by score descending."""
    out = np.asarray(pred, dtype=np.int64).copy()
    used: list[int] = []
    for sc, b, idx_d in sorted(shifts, key=lambda x: (-x[0], x[1]["tb"])):
        tb = int(b["tb"])
        d = int(DELTAS[idx_d])
        if d == 0:
            continue
        if any(abs(tb - u) < min_sep_frames for u in used):
            continue
        s_old, s_new = int(b["s_old"]), int(b["s_new"])
        if d < 0:
            sl = slice(tb + d, tb)
            if np.all(out[sl] == s_old) and np.all(out[tb : tb + 4] == s_new):
                out[sl] = s_new
                used.append(tb)
        else:
            sl = slice(tb, tb + d)
            if np.all(out[sl] == s_new) and np.all(out[tb - 4 : tb] == s_old):
                out[sl] = s_old
                used.append(tb)
    return out


class SymBoundaryOffsetAligner:
    """100% H<->A-Equivariant Supervised Sub-Second Boundary Offset Aligner (`EXP-16`)."""

    def __init__(self, cfg: BoundaryOffsetAlignerConfig | None = None):
        self.cfg = cfg or BoundaryOffsetAlignerConfig()
        self.models_: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]] = {}
        self.selected_cfgs_: dict[str, tuple[str, str, float, float] | None] = {}
        self.fitted_: bool = False

    def fit(self, bdy_by_match: dict[str, list[dict]]) -> "SymBoundaryOffsetAligner":
        """Fit inner-LOMO boundary offset experts and Zero-Degradation safety gates across training matches."""
        cfg = self.cfg
        match_ids = sorted(bdy_by_match.keys())
        self.models_ = {}
        self.selected_cfgs_ = {}

        btypes = []
        if cfg.enable_ha:
            btypes.append("HA")
        if cfg.enable_l2d:
            btypes.append("L2D")
        if cfg.enable_d2l:
            btypes.append("D2L")

        for bt in btypes:
            mods_in = []
            in_b_list = []
            in_rdg_list = []
            in_m_list = []

            for val_m in match_ids:
                tr_in = [
                    b
                    for m in match_ids
                    if m != val_m
                    for b in bdy_by_match[m]
                    if b["btype"] == bt
                    and b["L_left"] >= cfg.min_adj_spell_len
                    and b["L_right"] >= cfg.min_adj_spell_len
                ]
                va_in = [
                    b
                    for b in bdy_by_match[val_m]
                    if b["btype"] == bt
                    and b["L_left"] >= cfg.min_adj_spell_len
                    and b["L_right"] >= cfg.min_adj_spell_len
                ]
                if len(tr_in) < 10 or len(va_in) == 0:
                    continue
                mod_in = _fit_bdy_ridge(tr_in, l2=cfg.ridge_l2)
                mods_in.append(mod_in)
                for b in va_in:
                    in_b_list.append(b)
                    in_rdg_list.append(_score_bdy_ridge(b["feats"], mod_in, pen=cfg.offset_pen_phys))
                    in_m_list.append(val_m)

            self.models_[bt] = mods_in
            if not in_b_list:
                self.selected_cfgs_[bt] = None
                continue

            in_m_arr = np.asarray(in_m_list)
            in_f1_mat = np.asarray([b["f1_deltas"] for b in in_b_list])
            in_iou_mat = np.asarray([b["iou_deltas"] for b in in_b_list])
            in_val_mat = np.asarray([_candidate_phys_valid_mask(b, cfg) for b in in_b_list])
            n_inner_m = max(1, len(match_ids))

            best_obj = 0.0
            best_cfg: tuple[str, str, float, float] | None = None
            q_grid = np.linspace(86.0, 99.2, 35)
            experts = ["seq_gain", "ridge_phys"] if bt == "HA" else ["ridge_phys"]

            for exp_name in experts:
                sc_mat = np.asarray(
                    [
                        _score_offset_expert(b["feats"], bt, exp_name, rdg, cfg=cfg)
                        for b, rdg in zip(in_b_list, in_rdg_list)
                    ]
                )
                min_th_exp = cfg.min_th_seq if exp_name == "seq_gain" else cfg.min_th_ridge
                d_modes_use = (
                    ["small6", "small8"]
                    if exp_name == "seq_gain"
                    else ["small4", "small8", "small12"]
                )
                for d_mode in d_modes_use:
                    dm = _get_d_mask(d_mode)
                    masked = np.where(in_val_mat & dm[None, :], sc_mat, -999.0)
                    idx_pick = np.argmax(masked, axis=1)
                    val_pick = masked[np.arange(len(masked)), idx_pick]
                    f1_pick = in_f1_mat[np.arange(len(masked)), idx_pick]
                    iou_pick = in_iou_mat[np.arange(len(masked)), idx_pick]
                    valid_v = val_pick[val_pick > -100.0]
                    if len(valid_v) < 10:
                        continue
                    gains_q, ths_q = [], []
                    for q in q_grid:
                        th = max(float(np.percentile(valid_v, q)), min_th_exp)
                        ths_q.append(th)
                        fire = val_pick >= th
                        if np.sum(fire) < 3:
                            gains_q.append(-0.01)
                            continue
                        g_f1 = float(np.sum(f1_pick[fire])) / float(n_inner_m)
                        g_iou = float(np.sum(iou_pick[fire])) / float(n_inner_m)
                        pm_cnt = sum(
                            1 for vm in match_ids if np.sum(f1_pick[fire & (in_m_arr == vm)]) > 1e-5
                        )
                        neg_cnt = sum(
                            1 for vm in match_ids if np.sum(f1_pick[fire & (in_m_arr == vm)]) < -1e-5
                        )
                        if exp_name == "seq_gain":
                            ok = (neg_cnt == 0) and (pm_cnt >= 2)
                            bonus = 1.25
                        else:
                            ok = (
                                (neg_cnt == 0)
                                and (pm_cnt >= 3)
                                and (g_iou >= -0.02)
                                and (np.sum(fire) <= 30 * n_inner_m)
                            )
                            bonus = 1.0
                        gains_q.append((g_f1 * bonus) if ok else -0.01)
                    gains_smooth = np.convolve(gains_q, [0.25, 0.5, 0.25], mode="same")
                    idx_b = int(np.argmax(gains_smooth))
                    if gains_smooth[idx_b] > best_obj + 1e-5:
                        best_obj = float(gains_smooth[idx_b])
                        best_cfg = (exp_name, d_mode, float(ths_q[idx_b]), best_obj)

            min_g = cfg.min_inner_f1_gain_ha if bt == "HA" else cfg.min_inner_f1_gain_ld
            if best_cfg is not None and best_obj >= min_g:
                self.selected_cfgs_[bt] = best_cfg
            else:
                self.selected_cfgs_[bt] = None

        self.fitted_ = True
        return self

    def score_candidates(self, b_list: Sequence[dict]) -> list[tuple[float, dict, int]]:
        """Score candidate boundaries and return selected shifts `(score, b, idx_d)`."""
        if not self.fitted_:
            return []
        cfg = self.cfg
        shifts: list[tuple[float, dict, int]] = []
        for bt, sel_cfg in self.selected_cfgs_.items():
            if sel_cfg is None:
                continue
            exp_name, d_mode, th, _ = sel_cfg
            dm = _get_d_mask(d_mode)
            mods = self.models_.get(bt, [])
            for b in b_list:
                if (
                    b["btype"] != bt
                    or b["L_left"] < cfg.min_adj_spell_len
                    or b["L_right"] < cfg.min_adj_spell_len
                ):
                    continue
                rdg_te = (
                    np.mean(
                        [_score_bdy_ridge(b["feats"], md, pen=cfg.offset_pen_phys) for md in mods],
                        axis=0,
                    )
                    if (exp_name == "ridge_phys" and len(mods) > 0)
                    else None
                )
                sc = _score_offset_expert(b["feats"], bt, exp_name, rdg_te, cfg=cfg)
                sc[~_candidate_phys_valid_mask(b, cfg) | (~dm)] = -999.0
                idx_best = int(np.argmax(sc))
                if sc[idx_best] >= th:
                    shifts.append((float(sc[idx_best]), b, idx_best))
        return shifts

    def apply(
        self,
        pred: np.ndarray,
        proba: np.ndarray,
        half: np.ndarray,
        X: pd.DataFrame,
        p_change: np.ndarray | None = None,
        p_trans_4d: np.ndarray | None = None,
        family_probas: Sequence[np.ndarray] | None = None,
        base_pred: np.ndarray | None = None,
        precomputed_cands: Sequence[dict] | None = None,
    ) -> np.ndarray:
        """Apply learned boundary offset alignment to `pred`.

        If `base_pred` is provided (e.g., when stacking `EXP-16` with `EXP-15`), extracts candidate
        boundaries from `base_pred` so newly inserted short single-touch spells from `EXP-15` are
        never perturbed, and applies valid shifts onto `pred`.
        """
        if not self.fitted_:
            return np.asarray(pred, dtype=np.int64).copy()
        if precomputed_cands is not None:
            cands = list(precomputed_cands)
        else:
            ref_pred = base_pred if base_pred is not None else pred
            cands = extract_boundary_candidates(
                ref_pred,
                proba,
                half,
                X,
                p_change=p_change,
                p_trans_4d=p_trans_4d,
                family_probas=family_probas,
                labels=None,
                cfg=self.cfg,
            )
        shifts = self.score_candidates(cands)
        if not shifts:
            return np.asarray(pred, dtype=np.int64).copy()
        return apply_boundary_shifts(pred, shifts, min_sep_frames=self.cfg.min_sep_frames)


def apply_boundary_offset_aligner(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X: pd.DataFrame,
    aligner: SymBoundaryOffsetAligner,
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
    family_probas: Sequence[np.ndarray] | None = None,
    base_pred: np.ndarray | None = None,
    precomputed_cands: Sequence[dict] | None = None,
) -> np.ndarray:
    """Functional wrapper for `SymBoundaryOffsetAligner.apply`."""
    return aligner.apply(
        pred,
        proba,
        half,
        X,
        p_change=p_change,
        p_trans_4d=p_trans_4d,
        family_probas=family_probas,
        base_pred=base_pred,
        precomputed_cands=precomputed_cands,
    )


def verify_boundary_aligner_equivariance(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X: pd.DataFrame | dict[str, np.ndarray],
    p_change: np.ndarray | None = None,
    p_trans_4d: np.ndarray | None = None,
    family_probas: Sequence[np.ndarray] | None = None,
    aligner: SymBoundaryOffsetAligner | None = None,
) -> dict[str, float | int]:
    """Mathematically verify exact `0.000e+00` H <-> A equivariance of `SymBoundaryOffsetAligner`."""
    pred_sw = swap_labels(pred)
    proba_sw = swap_proba(proba)
    cxv = _extract_kinematic_arrays(X, len(pred))
    cxv_sw = _swap_kinematic_arrays(cxv)
    p4d_sw = p_trans_4d[:, [1, 0, 2, 3]].copy() if p_trans_4d is not None else None
    fams_sw = [swap_proba(f) for f in family_probas] if family_probas is not None else None

    cfg = aligner.cfg if aligner is not None else BoundaryOffsetAlignerConfig()
    c_orig = extract_boundary_candidates(
        pred, proba, half, cxv, p_change=p_change, p_trans_4d=p_trans_4d, family_probas=family_probas, cfg=cfg
    )
    c_swap = extract_boundary_candidates(
        pred_sw, proba_sw, half, cxv_sw, p_change=p_change, p_trans_4d=p4d_sw, family_probas=fams_sw, cfg=cfg
    )

    assert len(c_orig) == len(c_swap), f"Boundary candidate count mismatch: {len(c_orig)} vs {len(c_swap)}"
    max_feat_diff = 0.0
    for b1, b2 in zip(c_orig, c_swap):
        assert b1["tb"] == b2["tb"] and b1["btype"] == b2["btype"]
        diff = float(np.max(np.abs(b1["feats"] - b2["feats"])))
        max_feat_diff = max(max_feat_diff, diff)

    if aligner is not None and aligner.fitted_:
        out_orig = aligner.apply(
            pred, proba, half, cxv, p_change=p_change, p_trans_4d=p_trans_4d, family_probas=family_probas
        )
        out_swap = aligner.apply(
            pred_sw, proba_sw, half, cxv_sw, p_change=p_change, p_trans_4d=p4d_sw, family_probas=fams_sw
        )
        mismatch_frames = int(np.sum(swap_labels(out_orig) != out_swap))
    else:
        # Also test synthetic non-trivial threshold to verify shift application equivariance
        dummy = SymBoundaryOffsetAligner(cfg)
        dummy.fitted_ = True
        dummy.models_ = {"HA": []}
        dummy.selected_cfgs_ = {"HA": ("seq_gain", "small8", 0.02, 0.001)}
        out_orig = dummy.apply(
            pred, proba, half, cxv, p_change=p_change, p_trans_4d=p_trans_4d, family_probas=family_probas
        )
        out_swap = dummy.apply(
            pred_sw, proba_sw, half, cxv_sw, p_change=p_change, p_trans_4d=p4d_sw, family_probas=fams_sw
        )
        mismatch_frames = int(np.sum(swap_labels(out_orig) != out_swap))

    return {
        "n_boundaries": len(c_orig),
        "max_feat_diff": max_feat_diff,
        "mismatch_frames": mismatch_frames,
    }
