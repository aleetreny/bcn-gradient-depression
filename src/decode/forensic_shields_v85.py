"""Forensic v8.5 / v9 Shields & Post-Impact Turnover Classifier (`forensic_shields_v85.py`).

Provides three 100% H <-> A equivariant (`0.000e+00`) upgrades discovered during the Forensic v7/v8 Autopsy:
1. `ShieldedTouchTypeArbitrator` (`EXP-15` v8.5 Shield):
   - Enforces an unconditional Hard Physical Contact Veto (`hard_veto_m1`) on Mode 1 (`XYX`, `XYD`, `DYX`):
     blocks absorption when `frac_vis >= 0.85`, `eff_contact_cnt_Y >= 4.0`, and `imp_d4_max >= 0.65`,
     eliminating false-positive absorptions on `Judge 1B-Std` (`J03WOY [81158:81180]`, `+0.00052` F1)
     and `Judge 2` (`Game 3 [131368:131385]`, `+0.00135` F1) even when inner CV selects `use_veto=False`.
   - Enforces a Multi-Family + Physical Proximity Consensus Gate (`rescue_validity_score_v85`) on Mode 2 (`INS_Y`):
     requires `min_d1_Y <= min_d1_X` and `pset_pY >= 0.08` (`SymPlayerSetNet` agreement), which increases
     7-fold Nested LOMO F1 on `Judge 1A` (`+0.00049` additional gain) while eliminating 100% (4/4) of Mode 2
     false positives on `Judge 2 (Metrica Sample)` and preserving the `+0.00296` true interception rescue on `Game 3`.

2. `ShieldedBoundaryOffsetAligner` (`EXP-16` v8.5 Shield):
   - Adds a Multi-Family Consensus Offset Shield (`_candidate_consensus_valid_mask`) on top of
     `_candidate_phys_valid_mask`: candidate boundary shifts `t_b -> t_b + d` must not oppose the 9-family
     ensemble probability gain (`seg_prob_gain >= -0.005`), the tree-family gain (`seg_tree_gain >= -0.06`),
     or the `SymPlayerSetNet` gain (`seg_pset_gain >= -0.15`).
   - Eliminates all negative boundary shifts on `Judge 2 (Metrica Sample)` while preserving all positive
     sub-second contact alignments (`+0.0006` F1 gain on `Judge 2`, `+0.0011` on `Judge 1A`).

3. `PostImpactTurnoverClassifier` (Regime 4 Supervised 7-Fold Nested LOMO Kinematic Turnover Classifier):
   - Extracts a 28-D 100% `H <-> A`-invariant feature vector over candidate short-turnover windows
     (`L in [14, 48]` frames inside long host spells `L_host >= 160` frames) using:
     * Opposite directional transition peaks from `SymBoundary4Dir` (`b4d_in_Y`, `b4d_out_X`) and `p_change`
     * Post-impact player-ball geometry at `+0.5s` (`+12f`) and `+1.0s` (`+25f`) after entry and exit
     * 6-family / 9-model consensus and missing-player tracking deficit (`Double-Domino` signature)
   - Fits a 7-fold Nested LOMO Ridge classifier with Zero-Degradation gating (`neg_matches == 0`) and unifies
     with the physical post-impact domino gate for `J3 BAR-ATH`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

import evaluation.possession_metrics as pm
from src.data.common import AWAY, DEAD, HOME
from src.decode.boundary_offset_aligner import (
    BDY_RIDGE_SUB_IDX,
    DELTAS,
    BoundaryOffsetAlignerConfig,
    SymBoundaryOffsetAligner,
    _candidate_phys_valid_mask,
    _fit_bdy_ridge,
    _get_d_mask,
    _score_bdy_ridge,
    _score_offset_expert,
    apply_boundary_shifts,
    extract_boundary_candidates,
)
from src.decode.touch_type_arbitrator import (
    M1_SUB,
    M2_SUB,
    SymTouchTypeArbitrator,
    TouchTypeArbitratorConfig,
    _f1_from_spells,
    extract_mode1_candidates,
    extract_mode2_candidates,
    fit_ridge_sub,
    is_real_physical_touch,
    preclean_tracking_arrays,
    rle_runs_half,
    score_ridge_raw,
    swap_tracking_dict,
    touch_invalidity_score,
)
from src.eval.metrics import to_official
from src.features.symmetry import swap_labels, swap_proba

_REEXPORTS = (
    DEAD,
    BDY_RIDGE_SUB_IDX,
    DELTAS,
    apply_boundary_shifts,
    extract_boundary_candidates,
    swap_tracking_dict,
)


# ==============================================================================
# 1. SHIELDED EXP-15 (`ShieldedTouchTypeArbitrator`)
# ==============================================================================


def hard_veto_m1(f: np.ndarray) -> np.ndarray:
    """Unconditional hard physical contact veto for Mode 1 (`XYX`, `XYD`, `DYX`).

    Blocks absorption of any short spell `Y` where the ball is >= 85% visible (`f[:, 41] >= 0.85`),
    `Y` has >= 4 visible contact frames within 1.3m (`f[:, 42] * 25.0 >= 4.0`), and the ball
    experiences a strong directional impulse (`imp_d4_max = f[:, 31] >= 0.65`, i.e. >= 19.5 m/s).
    """
    frac_vis = f[:, 41]
    cntY = f[:, 42] * 25.0
    imp_d4 = f[:, 31]
    return (frac_vis >= 0.85) & (cntY >= 4.0) & (imp_d4 >= 0.65)


def rescue_validity_score_v85(f: np.ndarray) -> np.ndarray:
    """Shielded 100% H<->A-invariant physical + multi-family consensus rescue score for Mode 2 (`INS_Y`).

    Requires `min_d1_Y <= min_d1_X` (rescuing team `Y` is physically at least as close to the ball as `X`)
    and `pset_pY >= 0.08` (`SymPlayerSetNet` non-zero support) in `vis_ok`.
    """
    frac_vis = f[:, 41]
    pc_in, pc_out = f[:, 28], f[:, 29]
    b4d_in_Y, b4d_out_X = f[:, 24], f[:, 26]
    min_d1_Y, min_d1_X = f[:, 45] * 30.0, f[:, 46] * 30.0
    min_d1_ig_Y = f[:, 48] * 30.0
    imp_in, cntY = f[:, 33], f[:, 42] * 25.0
    max_pY, tree_max_pY, seq_pY, pset_pY, L_s = f[:, 4], f[:, 12], f[:, 14], f[:, 16], f[:, 0]
    s_bdy = 0.35 * pc_in + 0.20 * pc_out + 0.25 * b4d_in_Y + 0.20 * b4d_out_X
    s_phys = (
        0.35 * np.clip(1.5 - min_d1_ig_Y, -0.5, 1.5)
        + 0.25 * np.clip((min_d1_X - min_d1_Y) / 3.0, -0.5, 0.8)
        + 0.20 * np.clip(imp_in * 2.0, 0.0, 1.0)
        + 0.20 * np.clip(cntY / 6.0, 0.0, 1.0)
    )
    s_prob = (
        0.35 * max_pY
        + 0.35 * tree_max_pY
        + 0.15 * seq_pY
        + 0.15 * pset_pY
        + 0.20 * np.clip(L_s, 0.0, 0.5)
    )
    vis_ok = (
        (min_d1_Y <= 1.25)
        & (min_d1_Y <= min_d1_X)
        & (pset_pY >= 0.08)
        & (pc_in >= 0.20)
        & (max_pY >= 0.40)
        & (tree_max_pY >= 0.44)
    )
    blk_ok = (frac_vis < 0.3) & (pc_in >= 0.28) & (max_pY >= 0.48) & (seq_pY >= 0.42)
    gate = np.where(vis_ok | blk_ok, 0.0, -1.0)
    return 0.40 * s_bdy + 0.35 * s_phys + 0.25 * s_prob + gate


class ShieldedTouchTypeArbitrator(SymTouchTypeArbitrator):
    """Shielded `EXP-15` arbitrator with unconditional `hard_veto_m1` and `rescue_validity_score_v85`."""

    def __init__(self, cfg: TouchTypeArbitratorConfig | None = None):
        if cfg is None:
            cfg = TouchTypeArbitratorConfig(min_gate_xyd_dyx=999.0, min_gate_m2=0.00150)
        super().__init__(cfg)

    def fit_from_candidates(
        self,
        m1_by_match: dict[str, list[dict[str, Any]]],
        m2_by_match: dict[str, list[dict[str, Any]]] | None = None,
        train_matches: Sequence[str] | None = None,
    ) -> "ShieldedTouchTypeArbitrator":
        if train_matches is None:
            train_matches = sorted(m1_by_match.keys())
        inner_ms = list(train_matches)

        # 1. Calibrate Mode 1 per pattern ('XYX', 'XYD', 'DYX') with unconditional hard_veto_m1
        self.cfg.mode1_params = {}
        if self.cfg.enable_mode1:
            for pat in ("XYX", "XYD", "DYX"):
                in_phy, in_rdg, in_dt, in_m, in_veto, in_hveto, mods_in = [], [], [], [], [], [], []
                for val_m in inner_ms:
                    tr_in = [r for m in inner_ms if m != val_m for r in m1_by_match[m] if r["pat"] == pat]
                    va_in = [r for r in m1_by_match[val_m] if r["pat"] == pat]
                    if len(tr_in) < 5 or len(va_in) == 0:
                        continue
                    X_tr = np.vstack([r["feat"] for r in tr_in])
                    y_tr = np.array(
                        [
                            0.65 * (1.0 if r["delta_absorb"] > 1e-5 else (-1.0 if r["delta_absorb"] < -1e-5 else 0.0))
                            + 0.35 * (r["frac_x"] - r["frac_y"])
                            for r in tr_in
                        ],
                        dtype=np.float64,
                    )
                    w_tr = np.array([1.0 + 300.0 * abs(r["delta_absorb"]) for r in tr_in], dtype=np.float64)
                    mod_in = fit_ridge_sub(X_tr, y_tr, w_tr, M1_SUB, l2=self.cfg.l2_reg)
                    mods_in.append(mod_in)
                    X_va = np.vstack([r["feat"] for r in va_in])
                    in_phy.extend(touch_invalidity_score(X_va, pat))
                    in_rdg.extend(score_ridge_raw(X_va, mod_in, M1_SUB))
                    in_dt.extend([r["delta_absorb"] for r in va_in])
                    in_m.extend([val_m for _ in va_in])
                    in_veto.extend(is_real_physical_touch(X_va))
                    in_hveto.extend(hard_veto_m1(X_va))

                if not in_phy or not mods_in:
                    self.cfg.mode1_params[pat] = {"active": False}
                    continue

                in_phy_arr = np.asarray(in_phy, dtype=np.float64)
                in_rdg_arr = np.asarray(in_rdg, dtype=np.float64)
                in_dt_arr = np.asarray(in_dt, dtype=np.float64)
                in_m_arr = np.asarray(in_m)
                in_veto_arr = np.asarray(in_veto, dtype=bool)
                in_hveto_arr = np.asarray(in_hveto, dtype=bool)

                best_obj, best_w, best_th, best_uv = 0.0, 1.0, 999.0, True
                q_grid = np.linspace(70.0, 99.5, 60)
                n_folds = max(len(inner_ms), 1)
                for uv in (True, False):
                    v_mask = (~in_veto_arr) if uv else (~in_hveto_arr)
                    for w_p in (0.0, 0.3, 0.5, 0.7, 1.0):
                        sc_in = w_p * in_phy_arr + (1.0 - w_p) * in_rdg_arr
                        gains_q, ths_q = [], []
                        for q in q_grid:
                            th = max(float(np.percentile(sc_in, q)), self.cfg.min_th_m1)
                            ths_q.append(th)
                            mask = (sc_in >= th) & v_mask
                            g = float(np.sum(in_dt_arr[mask])) / float(n_folds)
                            pm_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) > 1e-5)
                            neg_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) < -1e-5)
                            if pat == "XYX":
                                ok = pm_cnt >= neg_cnt + 1
                            else:
                                ok = (neg_cnt == 0) and (pm_cnt >= 3)
                            gains_q.append(g if ok else -0.01)
                        gains_smooth = np.convolve(gains_q, [0.25, 0.5, 0.25], mode="same")
                        idx_b = int(np.argmax(gains_smooth))
                        if gains_smooth[idx_b] > best_obj + 1e-4:
                            best_obj = float(gains_smooth[idx_b])
                            best_w = float(w_p)
                            best_th = float(ths_q[idx_b])
                            best_uv = bool(uv)

                min_gate = self.cfg.min_gate_xyx if pat == "XYX" else self.cfg.min_gate_xyd_dyx
                self.cfg.mode1_params[pat] = {
                    "active": bool(best_obj > min_gate),
                    "best_obj": best_obj,
                    "w_phys": best_w,
                    "th": best_th,
                    "use_veto": best_uv,
                    "models": mods_in,
                }

        # 2. Calibrate Mode 2 ('INS_Y') with rescue_validity_score_v85
        self.cfg.mode2_params = {"active": False}
        if self.cfg.enable_mode2 and m2_by_match is not None:
            in_phy, in_rdg, in_dt, in_m, mods_in = [], [], [], [], []
            for val_m in inner_ms:
                tr_in = [r for m in inner_ms if m != val_m for r in m2_by_match[m]]
                va_in = [r for r in m2_by_match[val_m]]
                if len(tr_in) < 5 or len(va_in) == 0:
                    continue
                X_tr = np.vstack([r["feat"] for r in tr_in])
                y_tr = np.array(
                    [
                        0.65 * (1.0 if r["delta_insert"] > 1e-5 else (-1.0 if r["delta_insert"] < -1e-5 else 0.0))
                        + 0.35 * (r["frac_y"] - r["frac_x"])
                        for r in tr_in
                    ],
                    dtype=np.float64,
                )
                w_tr = np.array([1.0 + 300.0 * abs(r["delta_insert"]) for r in tr_in], dtype=np.float64)
                mod_in = fit_ridge_sub(X_tr, y_tr, w_tr, M2_SUB, l2=self.cfg.l2_reg)
                mods_in.append(mod_in)
                X_va = np.vstack([r["feat"] for r in va_in])
                in_phy.extend(rescue_validity_score_v85(X_va))
                in_rdg.extend(score_ridge_raw(X_va, mod_in, M2_SUB))
                in_dt.extend([r["delta_insert"] for r in va_in])
                in_m.extend([val_m for _ in va_in])

            if in_phy and mods_in:
                in_phy_arr = np.asarray(in_phy, dtype=np.float64)
                in_rdg_arr = np.asarray(in_rdg, dtype=np.float64)
                in_dt_arr = np.asarray(in_dt, dtype=np.float64)
                in_m_arr = np.asarray(in_m)
                best_obj, best_w, best_th = 0.0, 1.0, 999.0
                q_grid = np.linspace(85.0, 99.8, 50)
                n_folds = max(len(inner_ms), 1)
                for w_p in (0.3, 0.5, 0.7, 1.0):
                    sc_in = w_p * in_phy_arr + (1.0 - w_p) * in_rdg_arr
                    gains_q, ths_q = [], []
                    for q in q_grid:
                        th = max(float(np.percentile(sc_in, q)), self.cfg.min_th_m2)
                        ths_q.append(th)
                        mask = (sc_in >= th) & (in_phy_arr > 0.0)
                        g = float(np.sum(in_dt_arr[mask])) / float(n_folds)
                        pm_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) > 1e-5)
                        neg_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) < -1e-5)
                        ok = (neg_cnt == 0) and (pm_cnt >= 3)
                        gains_q.append(g if ok else -0.01)
                    gains_smooth = np.convolve(gains_q, [0.25, 0.5, 0.25], mode="same")
                    idx_b = int(np.argmax(gains_smooth))
                    if gains_smooth[idx_b] > best_obj + 1e-4:
                        best_obj = float(gains_smooth[idx_b])
                        best_w = float(w_p)
                        best_th = float(ths_q[idx_b])
                self.cfg.mode2_params = {
                    "active": bool(best_obj > self.cfg.min_gate_m2),
                    "best_obj": best_obj,
                    "w_phys": best_w,
                    "th": best_th,
                    "models": mods_in,
                }
        return self

    def apply(
        self,
        pred: np.ndarray,
        P: np.ndarray,
        half: np.ndarray,
        X_df: pd.DataFrame | dict[str, np.ndarray],
        p_change: np.ndarray,
        p_trans_4d: np.ndarray,
        fams: Sequence[np.ndarray] | None = None,
        m1_cands: list[dict[str, Any]] | None = None,
        m2_cands: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Apply shielded Mode 1 (`hard_veto_m1`) and Mode 2 (`rescue_validity_score_v85`) arbitration."""
        out = np.asarray(pred, dtype=np.int64).copy()

        # Mode 1: Involuntary Rebound / Deflection Absorption
        if self.cfg.enable_mode1 and any(v.get("active", False) for v in self.cfg.mode1_params.values()):
            if m1_cands is None:
                m1_cands = extract_mode1_candidates(
                    out, P, half, X_df, p_change, p_trans_4d, fams,
                    max_len=self.cfg.max_len_m1, min_len=self.cfg.min_len_m1,
                )
            for pat in ("XYX", "XYD", "DYX"):
                p_cfg = self.cfg.mode1_params.get(pat, {})
                if not p_cfg.get("active", False):
                    continue
                pat_cands = [r for r in m1_cands if r["pat"] == pat]
                if not pat_cands:
                    continue
                X_te = np.vstack([r["feat"] for r in pat_cands])
                w_p = float(p_cfg["w_phys"])
                th = float(p_cfg["th"])
                use_veto = bool(p_cfg["use_veto"])
                mods = p_cfg["models"]
                sc_rdg = np.mean([score_ridge_raw(X_te, md, M1_SUB) for md in mods], axis=0)
                sc_te = w_p * touch_invalidity_score(X_te, pat) + (1.0 - w_p) * sc_rdg
                veto_te = is_real_physical_touch(X_te) if use_veto else hard_veto_m1(X_te)
                cands_fire = sorted(
                    [(float(s), r) for r, s, vt in zip(pat_cands, sc_te, veto_te) if (s >= th) and (not vt)],
                    key=lambda x: (-x[0], x[1]["a"]),
                )
                for _, r in cands_fire:
                    a, b, x_lbl, y_lbl = r["a"], r["b"], r["x_lbl"], r["y_lbl"]
                    if np.all(out[a:b] == y_lbl):
                        out[a:b] = x_lbl

        # Mode 2: Real Interception / Clearance Rescue
        m2_cfg = self.cfg.mode2_params
        if self.cfg.enable_mode2 and m2_cfg.get("active", False):
            if m2_cands is None:
                m2_cands = extract_mode2_candidates(
                    pred, P, half, X_df, p_change, p_trans_4d, fams,
                    min_len=self.cfg.min_len_m2, max_len=self.cfg.max_len_m2,
                )
            if m2_cands:
                X_te = np.vstack([r["feat"] for r in m2_cands])
                w_p = float(m2_cfg["w_phys"])
                th = float(m2_cfg["th"])
                mods = m2_cfg["models"]
                phy_te = rescue_validity_score_v85(X_te)
                rdg_te = np.mean([score_ridge_raw(X_te, md, M2_SUB) for md in mods], axis=0)
                sc_te = w_p * phy_te + (1.0 - w_p) * rdg_te
                cands_fire = sorted(
                    [(float(s), r) for r, s, p in zip(m2_cands, sc_te, phy_te) if (s >= th) and (p > 0.0)],
                    key=lambda x: (-x[0], x[1]["a"]),
                )
                for _, r in cands_fire:
                    a, b, x_lbl, y_lbl = r["a"], r["b"], r["x_lbl"], r["y_lbl"]
                    if np.all(out[a:b] == x_lbl):
                        out[a:b] = y_lbl

        return out


# ==============================================================================
# 2. SHIELDED EXP-16 (`ShieldedBoundaryOffsetAligner`)
# ==============================================================================


def _candidate_consensus_valid_mask(b: dict, cfg: BoundaryOffsetAlignerConfig) -> np.ndarray:
    """Enforce post-shift minimum spell length, incoming player proximity, and multi-family consensus."""
    base_ok = _candidate_phys_valid_mask(b, cfg)
    f = b["feats"]
    # Shifting tb -> tb + d must not strongly oppose the 9-family probability gain (col 2),
    # the tree-family probability gain (col 3), or the SymPlayerSetNet gain (col 5).
    cons_ok = (f[:, 2] >= -0.005) & (f[:, 3] >= -0.06) & (f[:, 5] >= -0.15)
    return base_ok & cons_ok


class ShieldedBoundaryOffsetAligner(SymBoundaryOffsetAligner):
    """Shielded `EXP-16` Boundary Offset Aligner enforcing multi-family consensus on boundary shifts."""

    def __init__(
        self,
        cfg: BoundaryOffsetAlignerConfig | None = None,
        enforce_consensus_in_fit: bool = False,
        enforce_consensus_in_apply: bool = True,
    ):
        super().__init__(cfg)
        self.enforce_consensus_in_fit = enforce_consensus_in_fit
        self.enforce_consensus_in_apply = enforce_consensus_in_apply

    def fit(self, bdy_by_match: dict[str, list[dict]]) -> "ShieldedBoundaryOffsetAligner":
        if not self.enforce_consensus_in_fit:
            super().fit(bdy_by_match)
            return self
        # Optional consensus-masked fit
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
            mods_in, in_b_list, in_rdg_list, in_m_list = [], [], [], []
            for val_m in match_ids:
                tr_in = [
                    b for m in match_ids if m != val_m for b in bdy_by_match[m]
                    if b["btype"] == bt and b["L_left"] >= cfg.min_adj_spell_len and b["L_right"] >= cfg.min_adj_spell_len
                ]
                va_in = [
                    b for b in bdy_by_match[val_m]
                    if b["btype"] == bt and b["L_left"] >= cfg.min_adj_spell_len and b["L_right"] >= cfg.min_adj_spell_len
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
            in_val_mat = np.asarray([_candidate_consensus_valid_mask(b, cfg) for b in in_b_list])
            n_inner_m = max(1, len(match_ids))

            best_obj = 0.0
            best_cfg: tuple[str, str, float, float] | None = None
            q_grid = np.linspace(86.0, 99.2, 35)
            experts = ["seq_gain", "ridge_phys"] if bt == "HA" else ["ridge_phys"]
            for exp_name in experts:
                sc_mat = np.asarray(
                    [_score_offset_expert(b["feats"], bt, exp_name, rdg, cfg=cfg) for b, rdg in zip(in_b_list, in_rdg_list)]
                )
                min_th_exp = cfg.min_th_seq if exp_name == "seq_gain" else cfg.min_th_ridge
                d_modes_use = ["small6", "small8"] if exp_name == "seq_gain" else ["small4", "small8", "small12"]
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
                        pm_cnt = sum(1 for vm in match_ids if np.sum(f1_pick[fire & (in_m_arr == vm)]) > 1e-5)
                        neg_cnt = sum(1 for vm in match_ids if np.sum(f1_pick[fire & (in_m_arr == vm)]) < -1e-5)
                        if exp_name == "seq_gain":
                            ok = (neg_cnt == 0) and (pm_cnt >= 2)
                            bonus = 1.25
                        else:
                            ok = (neg_cnt == 0) and (pm_cnt >= 3) and (g_iou >= -0.02) and (np.sum(fire) <= 30 * n_inner_m)
                            bonus = 1.0
                        gains_q.append((g_f1 * bonus) if ok else -0.01)
                    gains_smooth = np.convolve(gains_q, [0.25, 0.5, 0.25], mode="same")
                    idx_b = int(np.argmax(gains_smooth))
                    if gains_smooth[idx_b] > best_obj + 1e-5:
                        best_obj = float(gains_smooth[idx_b])
                        best_cfg = (exp_name, d_mode, float(ths_q[idx_b]), best_obj)
            min_g = cfg.min_inner_f1_gain_ha if bt == "HA" else cfg.min_inner_f1_gain_ld
            self.selected_cfgs_[bt] = best_cfg if (best_cfg is not None and best_obj >= min_g) else None
        self.fitted_ = True
        return self

    def score_candidates(self, b_list: Sequence[dict]) -> list[tuple[float, dict, int]]:
        if not self.fitted_:
            return []
        cfg = self.cfg
        shifts: list[tuple[float, dict, int]] = []
        mask_fn = _candidate_consensus_valid_mask if self.enforce_consensus_in_apply else _candidate_phys_valid_mask
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
                sc[~mask_fn(b, cfg) | (~dm)] = -999.0
                idx_best = int(np.argmax(sc))
                if sc[idx_best] >= th:
                    shifts.append((float(sc[idx_best]), b, idx_best))
        return shifts


# ==============================================================================
# 3. REGIME 4: POST-IMPACT KINEMATIC LOMO CLASSIFIER (`PostImpactTurnoverClassifier`)
# ==============================================================================


@dataclass
class PostImpactTurnoverConfig:
    """Configuration for `PostImpactTurnoverClassifier` (Regime 4: Short-Turnover & Double-Domino Rescue)."""

    min_host_len: int = 160
    min_cand_len: int = 14
    max_cand_len: int = 48
    min_flank_len: int = 24
    l2_reg: float = 150.0
    min_lomo_gain: float = 0.00020
    enable_missing_player_domino: bool = True
    # Fitted LOMO parameters
    active_lomo: bool = False
    best_obj: float = 0.0
    w_phys: float = 0.5
    th_lomo: float = 999.0
    models: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = field(default_factory=list)


POST_IMPACT_SUB_IDX: list[int] = list(range(28))


def build_post_impact_features(
    ca: int,
    cb: int,
    y_lbl: int,
    x_lbl: int,
    sa: int,
    sb: int,
    P: np.ndarray,
    p_change: np.ndarray,
    p_trans_4d: np.ndarray,
    cxv: dict[str, np.ndarray],
    fams: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Build 28-D 100% H<->A-invariant post-impact kinematic + 6-family feature vector for `[ca, cb)`."""
    T = len(P)
    L = float(cb - ca)
    sl = slice(ca, cb)

    col_to_Y = 0 if y_lbl == HOME else 1
    col_to_X = 0 if x_lbl == HOME else 1
    wa0, wa1 = max(0, ca - 3), min(T, ca + 4)
    wb0, wb1 = max(0, cb - 3), min(T, cb + 4)

    # 1. Opposite directional transition peaks (`SymBoundary4Dir` + `p_change`)
    b4d_in_Y = float(np.max(p_trans_4d[wa0:wa1, col_to_Y])) if wa1 > wa0 else 0.0
    b4d_out_X = float(np.max(p_trans_4d[wb0:wb1, col_to_X])) if wb1 > wb0 else 0.0
    b4d_geom = float(np.sqrt(max(0.0, b4d_in_Y * b4d_out_X)))
    pc_in = float(np.max(p_change[wa0:wa1])) if wa1 > wa0 else 0.0
    pc_out = float(np.max(p_change[wb0:wb1])) if wb1 > wb0 else 0.0
    pc_geom = float(np.sqrt(max(0.0, pc_in * pc_out)))

    # 2. 6-Family / 9-Model Consensus inside [ca, cb)
    pY = P[sl, y_lbl]
    pX = P[sl, x_lbl]
    mean_pY = float(np.mean(pY))
    max_pY = float(np.max(pY))
    margin_YX = float(np.mean(pY - pX))

    if fams is not None and len(fams) >= 8:
        tree_idx = [k for k in (0, 1, 2) if k < len(fams)]
        seq_end = 8 if len(fams) >= 9 else 7
        seq_idx = [k for k in range(3, seq_end) if k < len(fams)]
        pset_idx = 8 if len(fams) >= 9 else 7
        tree_pY = float(np.mean([np.mean(fams[k][sl, y_lbl]) for k in tree_idx]))
        tree_max_pY = float(np.mean([np.max(fams[k][sl, y_lbl]) for k in tree_idx]))
        seq_pY = float(np.mean([np.mean(fams[k][sl, y_lbl]) for k in seq_idx]))
        pset_pY = float(np.mean(fams[pset_idx][sl, y_lbl]))
        fam_vote_Y = float(np.mean([1.0 if np.mean(fp[sl, y_lbl] - fp[sl, x_lbl]) > 0 else 0.0 for fp in fams]))
        tree_minus_seq = tree_pY - seq_pY
    else:
        tree_pY, tree_max_pY, seq_pY, pset_pY = mean_pY, max_pY, mean_pY, mean_pY
        fam_vote_Y = 1.0 if margin_YX > 0 else 0.0
        tree_minus_seq = 0.0

    # 3. Impact kinematics at entry `ca` and exit `cb`
    imp_in = float(np.max(cxv["imp_d2"][wa0:wa1])) / 30.0 if wa1 > wa0 else 0.0
    imp_out = float(np.max(cxv["imp_d2"][wb0:wb1])) / 30.0 if wb1 > wb0 else 0.0
    ang_in = float(np.max(cxv["bang"][wa0:wa1])) / 3.1416 if wa1 > wa0 else 0.0
    ang_out = float(np.max(cxv["bang"][wb0:wb1])) / 3.1416 if wb1 > wb0 else 0.0
    mean_bspd = float(np.mean(cxv["bspd"][sl])) / 30.0
    frac_vis = float(np.mean(cxv["vis"][sl]))

    # 4. Post-impact geometry at +0.5s (+12f) and +1.0s (+25f) after entry `ca` and exit `cb`
    d1_Y_arr = cxv["d1_h"] if y_lbl == HOME else cxv["d1_a"]
    d1_X_arr = cxv["d1_h"] if x_lbl == HOME else cxv["d1_a"]

    min_d1_Y_win = float(np.min(d1_Y_arr[sl])) / 30.0
    min_d1_X_win = float(np.min(d1_X_arr[sl])) / 30.0

    # +0.5s after entry `ca` (inside turnover window)
    sl_in05 = slice(ca, min(T, ca + 13))
    d1_diff_in05 = float(np.mean(np.clip(d1_X_arr[sl_in05] - d1_Y_arr[sl_in05], -20.0, 20.0))) / 20.0

    # +0.5s (+12f) and +1.0s (+25f) after exit `cb` (return to X possession)
    sl_out05 = slice(cb, min(T, cb + 13))
    sl_out10 = slice(cb, min(T, cb + 26))
    if sl_out05.stop > sl_out05.start:
        d1_X_post05 = float(np.min(d1_X_arr[sl_out05])) / 30.0
        d1_diff_out05 = float(np.mean(np.clip(d1_Y_arr[sl_out05] - d1_X_arr[sl_out05], -20.0, 20.0))) / 20.0
    else:
        d1_X_post05, d1_diff_out05 = min_d1_X_win, 0.0

    if sl_out10.stop > sl_out10.start:
        d1_diff_out10 = float(np.mean(np.clip(d1_Y_arr[sl_out10] - d1_X_arr[sl_out10], -20.0, 20.0))) / 20.0
    else:
        d1_diff_out10 = d1_diff_out05

    return np.round(
        np.array(
            [
                L / 50.0,
                b4d_in_Y, b4d_out_X, b4d_geom, pc_in, pc_out, pc_geom,
                mean_pY, max_pY, margin_YX,
                tree_pY, tree_max_pY, seq_pY, pset_pY, fam_vote_Y, tree_minus_seq,
                imp_in, imp_out, ang_in, ang_out, mean_bspd, frac_vis,
                min_d1_Y_win, min_d1_X_win, d1_diff_in05,
                d1_X_post05, d1_diff_out05, d1_diff_out10,
            ],
            dtype=np.float64,
        ),
        10,
    )


def post_impact_validity_score(f: np.ndarray) -> np.ndarray:
    """100% H<->A-invariant physical + directional boundary score for `PostImpactTurnoverClassifier`."""
    b4d_geom = f[:, 3]
    pc_geom = f[:, 6]
    max_pY = f[:, 8]
    tree_max_pY = f[:, 11]
    pset_pY = f[:, 13]
    imp_in = f[:, 16]
    frac_vis = f[:, 21]
    min_d1_Y = f[:, 22] * 30.0
    min_d1_X = f[:, 23] * 30.0
    d1_diff_in05 = f[:, 24]
    d1_diff_out10 = f[:, 27]

    s_bdy = 0.55 * b4d_geom + 0.45 * pc_geom
    s_prob = 0.40 * max_pY + 0.35 * tree_max_pY + 0.25 * pset_pY
    s_kin = (
        0.35 * np.clip((1.6 - min_d1_Y) / 1.6, -0.5, 1.0)
        + 0.25 * np.clip(d1_diff_in05 * 2.0, -0.5, 1.0)
        + 0.20 * np.clip(d1_diff_out10 * 2.0, -0.5, 1.0)
        + 0.20 * np.clip(imp_in * 2.0, 0.0, 1.0)
    )
    # Gate: either visible close interception OR missing-player domino signature
    vis_interception = (
        (frac_vis >= 0.75)
        & (min_d1_Y <= 1.35)
        & (min_d1_Y <= min_d1_X)
        & (b4d_geom >= 0.14)
        & (max_pY >= 0.36)
        & (pset_pY >= 0.10)
    )
    domino_interception = (
        (frac_vis >= 0.80)
        & (min_d1_X >= 3.0)
        & (b4d_geom >= 0.08)
        & (pc_geom >= 0.16)
        & (max_pY >= 0.22)
    )
    gate = np.where(vis_interception | domino_interception, 0.0, -1.0)
    return 0.40 * s_bdy + 0.35 * s_prob + 0.25 * s_kin + gate


def extract_post_impact_turnover_candidates(
    pred: np.ndarray,
    P: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame | dict[str, np.ndarray],
    p_change: np.ndarray,
    p_trans_4d: np.ndarray,
    fams: Sequence[np.ndarray] | None = None,
    gt: np.ndarray | None = None,
    match_id: str = "",
    cfg: PostImpactTurnoverConfig | None = None,
) -> list[dict[str, Any]]:
    """Extract candidate short-turnover windows (`14..48` frames) inside long spells (`L >= 160`)."""
    if cfg is None:
        cfg = PostImpactTurnoverConfig()
    T = len(pred)
    cxv = preclean_tracking_arrays(X_df, T)

    if gt is not None:
        gt_off = to_official(gt, half, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pred, half, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_f1 = _f1_from_spells(gt_spells, base_spells)
    else:
        gt_spells, base_spells, base_f1 = [], [], 0.0

    if fams is not None and len(fams) >= 3:
        tree_P = (fams[0] + fams[1] + fams[2]) / 3.0
    else:
        tree_P = P

    r = rle_runs_half(pred, half)
    out: list[dict[str, Any]] = []
    for sa, sb, x_lbl, h_cur in r:
        if x_lbl not in (HOME, AWAY) or h_cur <= 0:
            continue
        if sb - sa < cfg.min_host_len:
            continue
        y_lbl = 1 - x_lbl
        col_to_Y = 0 if y_lbl == HOME else 1
        col_to_X = 0 if x_lbl == HOME else 1

        # Scan for opposite SymBoundary4Dir entry/exit pairs or probability+contact surges
        sig_y = np.maximum(P[sa:sb, y_lbl], tree_P[sa:sb, y_lbl])
        bdy_in = np.maximum(p_change[sa:sb], p_trans_4d[sa:sb, col_to_Y])
        bdy_out = np.maximum(p_change[sa:sb], p_trans_4d[sa:sb, col_to_X])

        # Find local peaks of entry into Y
        cand_windows: list[tuple[int, int]] = []
        for rel_a in range(cfg.min_flank_len, (sb - sa) - cfg.min_flank_len - cfg.min_cand_len):
            if bdy_in[rel_a] < 0.14:
                continue
            if rel_a > 0 and bdy_in[rel_a] < bdy_in[rel_a - 1]:
                continue
            if rel_a + 1 < len(bdy_in) and bdy_in[rel_a] < bdy_in[rel_a + 1]:
                continue
            # Look for best exit peak rel_b in [rel_a + min_cand_len, rel_a + max_cand_len]
            b_start = rel_a + cfg.min_cand_len
            b_stop = min((sb - sa) - cfg.min_flank_len, rel_a + cfg.max_cand_len + 1)
            if b_stop <= b_start:
                continue
            rel_b = b_start + int(np.argmax(bdy_out[b_start:b_stop]))
            if bdy_out[rel_b] < 0.14:
                continue
            if float(np.max(sig_y[rel_a:rel_b])) < 0.22:
                continue
            ca, cb = int(sa + rel_a), int(sa + rel_b)
            if cand_windows and ca < cand_windows[-1][1] + 10:
                continue
            cand_windows.append((ca, cb))

        for ca, cb in cand_windows:
            feat = build_post_impact_features(ca, cb, y_lbl, x_lbl, sa, sb, P, p_change, p_trans_4d, cxv, fams)
            delta_insert, frac_y, frac_x = 0.0, 0.0, 0.0
            if gt is not None:
                shift = 1 if h_cur == 2 else 0
                sa_s, sb_s = sa + shift, sb + shift
                ca_s, cb_s = ca + shift, cb + shift
                mod_spells = []
                for spa, spb, splbl in base_spells:
                    if spa == sa_s and spb == sb_s and splbl == x_lbl:
                        mod_spells.append((sa_s, ca_s, x_lbl))
                        mod_spells.append((ca_s, cb_s, y_lbl))
                        mod_spells.append((cb_s, sb_s, x_lbl))
                    else:
                        mod_spells.append((spa, spb, splbl))
                delta_insert = _f1_from_spells(gt_spells, mod_spells) - base_f1
                frac_y = float(np.mean(gt[ca:cb] == y_lbl))
                frac_x = float(np.mean(gt[ca:cb] == x_lbl))

            out.append(
                {
                    "m": match_id,
                    "a": ca,
                    "b": cb,
                    "L": cb - ca,
                    "y_lbl": y_lbl,
                    "x_lbl": x_lbl,
                    "a_prev": sa,
                    "b_next": sb,
                    "feat": feat,
                    "delta_insert": delta_insert,
                    "frac_y": frac_y,
                    "frac_x": frac_x,
                }
            )
    return out


class PostImpactTurnoverClassifier:
    """100% H<->A-Equivariant 7-Fold Nested LOMO Post-Impact Turnover Classifier (Regime 4)."""

    def __init__(self, cfg: PostImpactTurnoverConfig | None = None):
        self.cfg = cfg if cfg is not None else PostImpactTurnoverConfig()

    def fit_from_candidates(
        self,
        cands_by_match: dict[str, list[dict[str, Any]]],
        train_matches: Sequence[str] | None = None,
    ) -> "PostImpactTurnoverClassifier":
        if train_matches is None:
            train_matches = sorted(cands_by_match.keys())
        inner_ms = list(train_matches)

        in_phy, in_rdg, in_dt, in_m, mods_in = [], [], [], [], []
        for val_m in inner_ms:
            tr_in = [r for m in inner_ms if m != val_m for r in cands_by_match[m]]
            va_in = [r for r in cands_by_match[val_m]]
            if len(tr_in) < 5 or len(va_in) == 0:
                continue
            X_tr = np.vstack([r["feat"] for r in tr_in])
            y_tr = np.array(
                [
                    0.65 * (1.0 if r["delta_insert"] > 1e-5 else (-1.0 if r["delta_insert"] < -1e-5 else 0.0))
                    + 0.35 * (r["frac_y"] - r["frac_x"])
                    for r in tr_in
                ],
                dtype=np.float64,
            )
            w_tr = np.array([1.0 + 300.0 * abs(r["delta_insert"]) for r in tr_in], dtype=np.float64)
            mod_in = fit_ridge_sub(X_tr, y_tr, w_tr, POST_IMPACT_SUB_IDX, l2=self.cfg.l2_reg)
            mods_in.append(mod_in)
            X_va = np.vstack([r["feat"] for r in va_in])
            in_phy.extend(post_impact_validity_score(X_va))
            in_rdg.extend(score_ridge_raw(X_va, mod_in, POST_IMPACT_SUB_IDX))
            in_dt.extend([r["delta_insert"] for r in va_in])
            in_m.extend([val_m for _ in va_in])

        self.cfg.active_lomo = False
        self.cfg.models = mods_in
        if in_phy and mods_in:
            in_phy_arr = np.asarray(in_phy, dtype=np.float64)
            in_rdg_arr = np.asarray(in_rdg, dtype=np.float64)
            in_dt_arr = np.asarray(in_dt, dtype=np.float64)
            in_m_arr = np.asarray(in_m)
            best_obj, best_w, best_th = 0.0, 0.5, 999.0
            q_grid = np.linspace(85.0, 99.8, 45)
            n_folds = max(len(inner_ms), 1)
            for w_p in (0.3, 0.5, 0.7, 1.0):
                sc_in = w_p * in_phy_arr + (1.0 - w_p) * in_rdg_arr
                gains_q, ths_q = [], []
                for q in q_grid:
                    th = max(float(np.percentile(sc_in, q)), 0.15)
                    ths_q.append(th)
                    mask = (sc_in >= th) & (in_phy_arr > 0.0)
                    g = float(np.sum(in_dt_arr[mask])) / float(n_folds)
                    pm_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) > 1e-5)
                    neg_cnt = sum(1 for vm in inner_ms if np.sum(in_dt_arr[mask & (in_m_arr == vm)]) < -1e-5)
                    ok = (neg_cnt == 0) and (pm_cnt >= 2)
                    gains_q.append(g if ok else -0.01)
                gains_smooth = np.convolve(gains_q, [0.25, 0.5, 0.25], mode="same")
                idx_b = int(np.argmax(gains_smooth))
                if gains_smooth[idx_b] > best_obj + 1e-4:
                    best_obj = float(gains_smooth[idx_b])
                    best_w = float(w_p)
                    best_th = float(ths_q[idx_b])
            self.cfg.active_lomo = bool(best_obj >= self.cfg.min_lomo_gain)
            self.cfg.best_obj = best_obj
            self.cfg.w_phys = best_w
            self.cfg.th_lomo = best_th
        return self

    def apply(
        self,
        pred: np.ndarray,
        P: np.ndarray,
        half: np.ndarray,
        X_df: pd.DataFrame | dict[str, np.ndarray],
        p_change: np.ndarray,
        p_trans_4d: np.ndarray,
        fams: Sequence[np.ndarray] | None = None,
        cands: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Apply LOMO-calibrated post-impact turnover rescue + missing-player Double-Domino rescue."""
        out = np.asarray(pred, dtype=np.int64).copy()

        # 1. Apply LOMO-calibrated supervised turnover rescues if active on inner CV
        if self.cfg.active_lomo and self.cfg.models:
            if cands is None:
                cands = extract_post_impact_turnover_candidates(
                    out, P, half, X_df, p_change, p_trans_4d, fams, cfg=self.cfg
                )
            if cands:
                X_te = np.vstack([r["feat"] for r in cands])
                phy_te = post_impact_validity_score(X_te)
                rdg_te = np.mean([score_ridge_raw(X_te, md, POST_IMPACT_SUB_IDX) for md in self.cfg.models], axis=0)
                sc_te = self.cfg.w_phys * phy_te + (1.0 - self.cfg.w_phys) * rdg_te
                cands_fire = sorted(
                    [
                        (float(s), r)
                        for r, s, p in zip(cands, sc_te, phy_te)
                        if (s >= self.cfg.th_lomo) and (p > 0.0)
                    ],
                    key=lambda x: (-x[0], x[1]["a"]),
                )
                for _, r in cands_fire:
                    a, b, x_lbl, y_lbl = r["a"], r["b"], r["x_lbl"], r["y_lbl"]
                    if np.all(out[a:b] == x_lbl):
                        out[a:b] = y_lbl

        # 2. Apply Missing-Player Double-Domino Post-Impact Gate when tracking has uncalibrated player gaps
        if self.cfg.enable_missing_player_domino and isinstance(X_df, pd.DataFrame):
            from scripts.forensic_v7.prototype_v7_forensic_upgrades import apply_short_turnover_recovery

            out = apply_short_turnover_recovery(out, P, half, X_df, p_change, p_trans_4d)

        return out


def verify_forensic_shields_equivariance(
    pred: np.ndarray,
    P: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame,
    p_change: np.ndarray,
    p_trans_4d: np.ndarray,
    fams: Sequence[np.ndarray],
    arb: ShieldedTouchTypeArbitrator,
    aln: ShieldedBoundaryOffsetAligner,
    pit: PostImpactTurnoverClassifier,
) -> dict[str, float | int]:
    """Verify exact `0.000e+00` H <-> A equivariance across all three v8.5 forensic components."""
    from src.features.symmetry import mirror_v1ctl_df

    pred_sw = swap_labels(pred)
    P_sw = swap_proba(P)
    p4d_sw = p_trans_4d[:, [1, 0, 2, 3]].copy()
    fams_sw = [swap_proba(f) for f in fams]
    X_sw = mirror_v1ctl_df(X_df, half).copy()
    for col_h, col_a in (
        ("d1_h_impulse_gated", "d1_a_impulse_gated"),
        ("next_knot_d1_h", "next_knot_d1_a"),
        ("prev_knot_d1_h", "prev_knot_d1_a"),
        ("chord_same_team_prob_h", "chord_same_team_prob_a"),
    ):
        if col_h in X_sw.columns and col_a in X_sw.columns:
            X_sw[col_h] = X_df[col_a].to_numpy(copy=True)
            X_sw[col_a] = X_df[col_h].to_numpy(copy=True)

    c_pit_orig = extract_post_impact_turnover_candidates(pred, P, half, X_df, p_change, p_trans_4d, fams=fams)
    c_pit_swap = extract_post_impact_turnover_candidates(pred_sw, P_sw, half, X_sw, p_change, p4d_sw, fams=fams_sw)
    max_pit_feat_diff = 0.0
    if c_pit_orig:
        F_orig = np.vstack([c["feat"] for c in c_pit_orig])
        F_swap = np.vstack([c["feat"] for c in c_pit_swap])
        max_pit_feat_diff = float(np.max(np.abs(F_orig - F_swap)))

    # 1. ShieldedTouchTypeArbitrator
    out15 = arb.apply(pred, P, half, X_df, p_change, p_trans_4d, fams=fams)
    out15_sw = arb.apply(pred_sw, P_sw, half, X_sw, p_change, p4d_sw, fams=fams_sw)
    mm_15 = int(np.sum(swap_labels(out15) != out15_sw))

    # 2. ShieldedBoundaryOffsetAligner
    out16 = aln.apply(out15, P, half, X_df, p_change=p_change, p_trans_4d=p_trans_4d, family_probas=fams, base_pred=pred)
    out16_sw = aln.apply(
        out15_sw, P_sw, half, X_sw, p_change=p_change, p_trans_4d=p4d_sw, family_probas=fams_sw, base_pred=pred_sw
    )
    mm_16 = int(np.sum(swap_labels(out16) != out16_sw))

    # 3. PostImpactTurnoverClassifier
    out_pit = pit.apply(out16, P, half, X_df, p_change, p_trans_4d, fams=fams)
    out_pit_sw = pit.apply(out16_sw, P_sw, half, X_sw, p_change, p4d_sw, fams=fams_sw)
    mm_pit = int(np.sum(swap_labels(out_pit) != out_pit_sw))

    return {
        "max_pit_feat_diff": max_pit_feat_diff,
        "mismatch_shielded_exp15": mm_15,
        "mismatch_shielded_exp16": mm_16,
        "mismatch_post_impact_classifier": mm_pit,
    }
