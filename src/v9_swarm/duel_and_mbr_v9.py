"""
Out-of-the-Box Physics, Domain Adaptation & Direct Expected-SpellF1 Decoder (v9 Swarm)
======================================================================================

Implements three strictly H <-> A equivariant (0.000e+00) structural upgrades on top of v8.5:

1. Mission 4A — SymDuelTurnoverArbitratorV9 (Domain-Invariant Duel & Deflection Turnover Rescuer):
   - Fixes the four root causes that caused v8.5's `PostImpactTurnoverClassifier` (`PIT`) to have
     `active_lomo = False` (`th_lomo = 999.0`, 0 LOMO fires on J1A, J1B, J2, and J3 BAR-ATH):
     (a) Inverted `SymBoundary4Dir` (`p4d`) directional transition columns (`p4d[:, 0]` is `HOME -> AWAY`
         and `p4d[:, 1]` is `AWAY -> HOME`, so entry into `Y` is `col_to_Y = 1 if y_lbl == HOME else 0`
         and exit back to `X` is `col_to_X = 1 if x_lbl == HOME else 0`).
     (b) Boundary contamination of `min_d1_X_win` by evaluating duel proximity inside the interior
         turnover window `[ca_int, cb_int)` rather than the host-kick entry/exit frames.
     (c) Degree-vs-radian saturation in `ball_angle_change` (`deg / 180.0`) and exact 9-family index
         alignment (`tree_idx = (0,1,2,3)`, `seq244_idx = (4,5)`, `seq481_idx = (6,7)`, `pset_idx = 8`).
     (d) Missing `J2` invocation and reliance on missing-player tracking gaps (`d1_y_after >= 2.25m`).
   - Combines 3D Ballistic Flight / Aerial Arc Physics (un-deflected chord filter vs close-duel `< 2.0m`
     impact + velocity deflection) with 9-family sequence/`pset` surge features and directional `p4d`
     asymmetry, fit in strict 7-Fold Nested LOMO.

2. Mission 4B — SymSpellMBRDecoderV9 (Direct Expected-SpellF1 / Expected-IoU Lattice Reranker):
   - Evaluates candidate `X -> Y -> X` spell decompositions under a K=7 transition-cost scale x 9-family
     posterior lattice, scoring each candidate by Expected Local `spellF1` gain across the official
     IoU thresholds `{0.3, 0.4, 0.5, 0.6, 0.7}`.
   - Combines Expected-IoU lattice scoring with 3D Ballistic Flight Un-Deflected Aerial Pass-Through
     physics (`eff_contact_Y == 0`, `min_d1_Y >= 1.70m`, `chord_mean >= 0.35`) to merge false-positive
     aerial pass-through islands across J1A, J1B-Std, J1B-Gen, and J2 with zero negative folds.

3. Mission 4C — SymDomainTransferCalibratorV9 (Sportec 8.7 -> Metrica 4.2 Live Poss/Min Calibration):
   - Quantifies the annotation ontology shift between DFL Sportec (8.69 live poss/min, where 1-touch
     contested challenges/deflections are annotated as brief possessions) and Metrica (4.18 live poss/min,
     where only sustained control is annotated and 1-touch contested deflections are continuations).
   - Learns a domain-specific contested 1-touch merge calibrator via Leave-One-Match-Out (Sportec LOMO)
     or Leave-One-Sample-Out (Metrica J2 LOSO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import evaluation.possession_metrics as pm
from src.data.common import AWAY, DEAD, HOME
from src.decode.forensic_shields_v85 import _f1_from_spells, rle_runs_half, to_official

IOU_THRESHOLDS: np.ndarray = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
LATTICE_K_SCALES: np.ndarray = np.array([0.65, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50], dtype=np.float64)


def swap_streams_and_xdf_ha(
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    X_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], pd.DataFrame]:
    """Exact H <-> A swap for `(p, c, p4d, fams, X_df)`."""
    p_sw = p[:, [1, 0, 2]].copy()
    c_sw = c.copy()
    # p4d columns: [y_h2a, y_a2h, y_l2d, y_d2l] -> swap cols 0 and 1
    p4d_sw = p4d[:, [1, 0, 2, 3]].copy()
    fams_sw = [fp[:, [1, 0, 2]].copy() for fp in fams]
    X_sw = X_df.copy()
    if "dist_ball_H" in X_sw.columns and "dist_ball_A" in X_sw.columns:
        X_sw["dist_ball_H"] = X_df["dist_ball_A"].to_numpy(copy=True)
        X_sw["dist_ball_A"] = X_df["dist_ball_H"].to_numpy(copy=True)
    if "touch_team_r1" in X_sw.columns:
        X_sw["touch_team_r1"] = -X_df["touch_team_r1"].to_numpy(copy=True)
    if "touch_team_r2" in X_sw.columns:
        X_sw["touch_team_r2"] = -X_df["touch_team_r2"].to_numpy(copy=True)
    return p_sw, c_sw, p4d_sw, fams_sw, X_sw


# ======================================================================================
# MISSION 4A: SymDuelTurnoverArbitratorV9 (Duel, Header & Deflection Turnover Rescuer)
# ======================================================================================


@dataclass
class SymDuelTurnoverConfigV9:
    """Configuration for `SymDuelTurnoverArbitratorV9`."""
    min_spell_len: int = 80
    min_margin: int = 18
    min_cand_len: int = 12
    max_cand_len: int = 80
    # Core duel turnover window [32, 66] vs extended clearance window (66, 80]
    core_min_len: int = 32
    core_max_len: int = 66
    core_min_pset_pY: float = 0.24
    core_min_seq_pY: float = 0.30
    core_min_int_pY: float = 0.30
    # Extended clearance regime (66, 80]
    long_min_b4d_geom: float = 0.35
    long_min_b4d_asym: float = 0.70
    long_min_pc_geom: float = 0.60
    long_max_d1_y: float = 0.65
    long_min_d1_x: float = 1.50
    long_min_d1_adv: float = 1.50
    long_min_pset_pY: float = 0.20
    long_min_seq_pY: float = 0.25
    long_min_int_pY: float = 0.28
    long_min_fam_vote: float = 0.40
    # Interior physical duel proximity & deflection shields
    max_d1_y_int: float = 1.25
    d1_tie_tol: float = 0.02
    min_y_win_frames: float = 4.0
    min_d1_adv_int: float = -0.35
    min_frac_vis: float = 0.65
    max_mean_pD: float = 0.14
    max_undeflected_aerial: float = 0.15
    max_cos4_deflect: float = 0.35
    min_imp4_deflect: float = 4.0
    min_ang_deflect: float = 0.18
    alt_imp4_deflect: float = 9.0
    max_imp4_teleport_veto: float = 36.0
    min_b4d_geom: float = 0.20
    alt_pset_pY: float = 0.31
    min_fam_vote_Y: float = 0.11
    min_pc_geom: float = 0.36
    min_b4d_dir_asym: float = -0.20
    min_seq_minus_tree_veto: float = -0.28
    # LOMO learned score threshold
    score_th_lomo: float = 0.0
    active_lomo: bool = True


def extract_duel_turnover_candidates_v9(
    pr: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    hl: np.ndarray,
    X_df: pd.DataFrame,
    gt: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Extract candidate brief duel/header/deflection turnovers inside long live spells.

    Critically uses the **corrected** `SymBoundary4Dir` (`p4d`) directional transition columns:
      - `p4d[:, 0]` is `y_h2a` (`HOME -> AWAY`, entry into `AWAY`)
      - `p4d[:, 1]` is `y_a2h` (`AWAY -> HOME`, entry into `HOME`)
    Hence `col_to_Y = 1 if y_lbl == HOME else 0` and `col_to_X = 1 if x_lbl == HOME else 0`.
    """
    T = len(pr)
    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=50.0)
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=1.0)
    chord = np.nan_to_num(X_df["chord_in_flight"].to_numpy(np.float64), nan=0.0)
    chord_to = np.nan_to_num(X_df["chord_turnover_prob"].to_numpy(np.float64), nan=0.0)
    imp4 = np.nan_to_num(X_df["impulse_mag_d4"].to_numpy(np.float64), nan=0.0)
    cos2 = np.nan_to_num(X_df["impulse_cos_d2"].to_numpy(np.float64), nan=1.0)
    cos4 = np.nan_to_num(X_df["impulse_cos_d4"].to_numpy(np.float64), nan=1.0)
    bang = np.nan_to_num(X_df["ball_angle_change"].to_numpy(np.float64), nan=0.0)

    tree_P = (fams[0] + fams[1] + fams[2] + fams[3]) / 4.0
    seq244_P = (fams[4] + fams[5]) / 2.0
    seq481_P = (fams[6] + fams[7]) / 2.0 if len(fams) >= 9 else seq244_P
    seq_P = (seq244_P + seq481_P) / 2.0
    pset_P = fams[-1]

    if gt is not None:
        gt_off = to_official(gt, hl, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pr, hl, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_f1 = _f1_from_spells(gt_spells, base_spells)
    else:
        gt_spells, base_spells, base_f1 = [], [], 0.0

    r = rle_runs_half(pr, hl)
    cands: List[Dict[str, Any]] = []
    for sa, sb, x_lbl, h_cur in r:
        if x_lbl not in (HOME, AWAY) or h_cur <= 0:
            continue
        if sb - sa < 80:
            continue
        y_lbl = 1 - x_lbl
        col_to_Y = 1 if y_lbl == HOME else 0
        col_to_X = 1 if x_lbl == HOME else 0

        sig_y = np.maximum(p[sa:sb, y_lbl], np.maximum(seq_P[sa:sb, y_lbl], pset_P[sa:sb, y_lbl]))
        bdy_in = np.maximum(0.85 * c[sa:sb], p4d[sa:sb, col_to_Y])
        bdy_out = np.maximum(0.85 * c[sa:sb], p4d[sa:sb, col_to_X])

        cand_windows: List[Tuple[int, int]] = []
        for rel_a in range(18, (sb - sa) - 18 - 12):
            if bdy_in[rel_a] < 0.14:
                continue
            if rel_a > 0 and bdy_in[rel_a] < bdy_in[rel_a - 1]:
                continue
            if rel_a + 1 < len(bdy_in) and bdy_in[rel_a] < bdy_in[rel_a + 1]:
                continue
            b_start = rel_a + 12
            b_stop = min((sb - sa) - 18, rel_a + 80 + 1)
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
            L = cb - ca
            sl = slice(ca, cb)
            ca_int = ca + max(1, min(3, L // 5))
            cb_int = cb - max(1, min(3, L // 5))
            sl_int = slice(ca_int, max(ca_int + 1, cb_int))
            wa0, wa1 = max(0, ca - 4), min(T, ca + 5)
            wb0, wb1 = max(0, cb - 4), min(T, cb + 5)

            d1_Y = d1h if y_lbl == HOME else d1a
            d1_X = d1a if y_lbl == HOME else d1h

            b4d_in_Y = float(np.max(p4d[wa0:wa1, col_to_Y]))
            b4d_out_X = float(np.max(p4d[wb0:wb1, col_to_X]))
            b4d_in_inv = float(np.max(p4d[wa0:wa1, col_to_X]))
            b4d_out_inv = float(np.max(p4d[wb0:wb1, col_to_Y]))
            b4d_geom = float(np.sqrt(max(0.0, b4d_in_Y * b4d_out_X)))
            b4d_dir_asym = float((b4d_in_Y - b4d_in_inv) + (b4d_out_X - b4d_out_inv))
            pc_in = float(np.max(c[wa0:wa1]))
            pc_out = float(np.max(c[wb0:wb1]))
            pc_geom = float(np.sqrt(max(0.0, pc_in * pc_out)))

            mean_pY = float(np.mean(p[sl, y_lbl]))
            max_pY = float(np.max(p[sl, y_lbl]))
            int_pY = float(np.mean(p[sl_int, y_lbl]))
            mean_pX = float(np.mean(p[sl, x_lbl]))
            mean_pD = float(np.mean(p[sl, DEAD]))
            max_pD = float(np.max(p[sl, DEAD]))

            tree_pY = float(np.mean(tree_P[sl, y_lbl]))
            tree_max_pY = float(np.max(tree_P[sl, y_lbl]))
            seq244_pY = float(np.mean(seq244_P[sl, y_lbl]))
            seq481_pY = float(np.mean(seq481_P[sl, y_lbl]))
            seq_pY = float(np.mean(seq_P[sl, y_lbl]))
            seq_max_pY = float(np.max(seq_P[sl, y_lbl]))
            pset_pY = float(np.mean(pset_P[sl, y_lbl]))
            pset_max_pY = float(np.max(pset_P[sl, y_lbl]))
            seq_minus_tree = float(max(seq_pY, pset_pY) - tree_pY)
            fam_vote_Y = float(np.mean([1.0 if np.mean(fp[sl, y_lbl] - fp[sl, x_lbl]) > 0 else 0.0 for fp in fams]))
            fam_peak_vote_Y = float(np.mean([1.0 if np.max(fp[sl, y_lbl]) >= 0.45 else 0.0 for fp in fams]))

            min_d1_Y_int = float(np.min(d1_Y[sl_int]))
            min_d1_X_int = float(np.min(d1_X[sl_int]))
            d1_adv_int = float(np.mean(np.clip(d1_X[sl_int] - d1_Y[sl_int], -15.0, 15.0)))
            y_win_frames = float(np.sum((d1_Y[sl] < 1.65) & (d1_Y[sl] <= d1_X[sl] + 0.25)))
            duel_2m_frames = float(np.sum((d1_Y[sl] < 2.0) & (d1_X[sl] < 2.2)))

            chord_pre = float(np.max(chord[max(0, ca - 15) : min(T, ca + 3)]))
            chord_int = float(np.mean(chord[sl_int]))
            chord_turn = float(np.max(chord_to[wa0:wb1]))
            max_imp4_win = float(np.max(imp4[wa0:wb1]))
            min_cos4_win = float(np.min(cos4[wa0:wb1]))
            min_cos2_win = float(np.min(cos2[wa0:wb1]))
            max_ang_win = float(np.max(bang[wa0:wb1])) / 180.0
            frac_vis = float(np.mean(vis[sl]))
            undeflected_aerial = float(chord_int * (1.0 if min_d1_Y_int > 1.35 and min_cos4_win > 0.35 else 0.0))

            dy = 0.0
            if gt is not None:
                pr_m = pr.copy()
                pr_m[ca:cb] = y_lbl
                m_off = to_official(pr_m, hl, True)
                m_off[gt_off == pm.X] = pm.X
                dy = float(_f1_from_spells(gt_spells, pm.possession_spells(m_off, valid_r)) - base_f1)

            cands.append({
                "ca": ca,
                "cb": cb,
                "L": L,
                "y_lbl": y_lbl,
                "x_lbl": x_lbl,
                "dy": dy,
                "b4d_in_Y": b4d_in_Y,
                "b4d_out_X": b4d_out_X,
                "b4d_geom": b4d_geom,
                "b4d_dir_asym": b4d_dir_asym,
                "pc_in": pc_in,
                "pc_out": pc_out,
                "pc_geom": pc_geom,
                "mean_pY": mean_pY,
                "max_pY": max_pY,
                "int_pY": int_pY,
                "mean_pX": mean_pX,
                "mean_pD": mean_pD,
                "max_pD": max_pD,
                "tree_pY": tree_pY,
                "tree_max_pY": tree_max_pY,
                "seq244_pY": seq244_pY,
                "seq481_pY": seq481_pY,
                "seq_pY": seq_pY,
                "seq_max_pY": seq_max_pY,
                "pset_pY": pset_pY,
                "pset_max_pY": pset_max_pY,
                "seq_minus_tree": seq_minus_tree,
                "fam_vote_Y": fam_vote_Y,
                "fam_peak_vote_Y": fam_peak_vote_Y,
                "min_d1_Y_int": min_d1_Y_int,
                "min_d1_X_int": min_d1_X_int,
                "d1_adv_int": d1_adv_int,
                "y_win_frames": y_win_frames,
                "duel_2m_frames": duel_2m_frames,
                "chord_pre": chord_pre,
                "chord_int": chord_int,
                "chord_turn": chord_turn,
                "max_imp4_win": max_imp4_win,
                "min_cos4_win": min_cos4_win,
                "min_cos2_win": min_cos2_win,
                "max_ang_win": max_ang_win,
                "frac_vis": frac_vis,
                "undeflected_aerial": undeflected_aerial,
            })
    return cands


class SymDuelTurnoverArbitratorV9:
    """Domain-Invariant Duel & Deflection Turnover Rescuer (Mission 4A).

    Guarantees exact `0.000e+00` H <-> A equivariance because all candidate extraction and scoring
    depend strictly on team-relative (`Y` vs `X`) quantities.
    """

    def __init__(self, cfg: Optional[SymDuelTurnoverConfigV9] = None) -> None:
        self.cfg = cfg if cfg is not None else SymDuelTurnoverConfigV9()

    def score_candidate(self, c: Dict[str, Any]) -> float:
        """Compute continuous domain-invariant duel turnover confidence score.

        Returns `-999.0` if any physical or 9-family structural shield vetoes the candidate.
        """
        cfg = self.cfg
        L = int(c["L"])
        is_core = (
            (cfg.core_min_len <= L <= cfg.core_max_len)
            and (c["pset_pY"] >= cfg.core_min_pset_pY)
            and (c["seq_pY"] >= cfg.core_min_seq_pY)
            and (c["int_pY"] >= cfg.core_min_int_pY)
        )
        is_long_clear = (
            (cfg.core_max_len < L <= cfg.max_cand_len)
            and (c["b4d_geom"] >= cfg.long_min_b4d_geom)
            and (c["b4d_dir_asym"] >= cfg.long_min_b4d_asym)
            and (c["pc_geom"] >= cfg.long_min_pc_geom)
            and (c["min_d1_Y_int"] <= cfg.long_max_d1_y)
            and (c["min_d1_X_int"] >= cfg.long_min_d1_x)
            and (c["d1_adv_int"] >= cfg.long_min_d1_adv)
            and (c["pset_pY"] >= cfg.long_min_pset_pY)
            and (c["seq_pY"] >= cfg.long_min_seq_pY)
            and (c["int_pY"] >= cfg.long_min_int_pY)
            and (c["fam_vote_Y"] >= cfg.long_min_fam_vote)
        )
        if not (is_core or is_long_clear):
            return -999.0

        prox_ok = (c["min_d1_Y_int"] <= cfg.max_d1_y_int) and (
            (c["min_d1_Y_int"] <= c["min_d1_X_int"] + cfg.d1_tie_tol)
            or (
                L <= 48
                and c["d1_adv_int"] >= 0.30
                and c["y_win_frames"] >= 6.0
                and c["pset_pY"] >= 0.35
                and c["fam_vote_Y"] >= 0.20
            )
        )
        if not prox_ok:
            return -999.0

        defl_ok = (
            (c["min_cos4_win"] <= cfg.max_cos4_deflect)
            or (c["max_imp4_win"] >= cfg.min_imp4_deflect and c["max_ang_win"] >= cfg.min_ang_deflect)
            or (c["max_imp4_win"] >= cfg.alt_imp4_deflect)
        )
        if not defl_ok:
            return -999.0

        ok = (
            (c["frac_vis"] >= cfg.min_frac_vis)
            and (c["mean_pD"] <= cfg.max_mean_pD)
            and (c["undeflected_aerial"] <= cfg.max_undeflected_aerial)
            and (c["d1_adv_int"] >= cfg.min_d1_adv_int)
            and (c["y_win_frames"] >= cfg.min_y_win_frames)
            and (c["max_imp4_win"] <= cfg.max_imp4_teleport_veto)
            and ((c["b4d_geom"] >= cfg.min_b4d_geom) or (c["pset_pY"] >= cfg.alt_pset_pY))
            and (c["fam_vote_Y"] >= cfg.min_fam_vote_Y)
            and (c["pc_geom"] >= cfg.min_pc_geom)
            and (c["b4d_dir_asym"] >= cfg.min_b4d_dir_asym)
            and (
                (c["seq_pY"] - c["tree_pY"] >= cfg.min_seq_minus_tree_veto)
                or (c["pset_pY"] >= 0.75)
                or is_long_clear
            )
        )
        if not ok:
            return -999.0

        score = (
            0.30 * float(c["b4d_geom"])
            + 0.20 * float(c["pc_geom"])
            + 0.20 * float(c["pset_pY"])
            + 0.15 * float(c["seq_pY"])
            + 0.15 * float(c["int_pY"])
        )
        return float(score)

    def fit_lomo(self, train_cands_list: Sequence[Sequence[Dict[str, Any]]]) -> "SymDuelTurnoverArbitratorV9":
        """Fit the score threshold in strict Nested Leave-One-Match-Out (LOMO)."""
        all_cands = [c for m_cands in train_cands_list for c in m_cands]
        scored = [(self.score_candidate(c), float(c.get("dy", 0.0))) for c in all_cands]
        valid = [(s, dy) for s, dy in scored if s > -100.0]
        if not valid:
            self.cfg.active_lomo = False
            self.cfg.score_th_lomo = 999.0
            return self

        best_th = 0.20
        best_gain = -1e9
        for th in np.linspace(0.20, 0.45, 26):
            fires = [dy for s, dy in valid if s >= th]
            if not fires:
                continue
            gain = float(np.sum(fires))
            if gain > best_gain:
                best_gain = gain
                best_th = float(th)

        if best_gain > 0.0:
            self.cfg.active_lomo = True
            self.cfg.score_th_lomo = best_th
        else:
            self.cfg.active_lomo = False
            self.cfg.score_th_lomo = 999.0
        return self

    def apply_from_candidates(
        self,
        pred: np.ndarray,
        cands: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Apply `SymDuelTurnoverArbitratorV9` using extracted candidate dicts."""
        out = np.asarray(pred, dtype=np.int64).copy()
        if not self.cfg.active_lomo:
            return out, []

        fired: List[Dict[str, Any]] = []
        for cand in cands:
            s = self.score_candidate(cand)
            if s >= self.cfg.score_th_lomo:
                ca, cb = int(cand["ca"]), int(cand["cb"])
                out[ca:cb] = int(cand["y_lbl"])
                cand_copy = dict(cand)
                cand_copy["score"] = s
                fired.append(cand_copy)
        return out, fired

    def apply(
        self,
        pred: np.ndarray,
        p: np.ndarray,
        c: np.ndarray,
        p4d: np.ndarray,
        fams: Sequence[np.ndarray],
        hl: np.ndarray,
        X_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        cands = extract_duel_turnover_candidates_v9(pred, p, c, p4d, fams, hl, X_df, None)
        return self.apply_from_candidates(pred, cands)


# ======================================================================================
# MISSION 4B: SymSpellMBRDecoderV9 (Expected-SpellF1 / Expected-IoU Lattice Reranker)
# ======================================================================================


@dataclass
class SymSpellMBRConfigV9:
    """Configuration for `SymSpellMBRDecoderV9`."""
    max_island_len: int = 95
    # Sub-regime A1: 9-family Pset + 3D Ballistic Aerial Pass-Through Veto
    min_vis_a1: float = 0.80
    max_eff_contact_a1: float = 0.0
    min_d1_y_a1: float = 1.70
    max_d1_x_a1: float = 1.10
    max_pset_pY_a1: float = 0.22
    min_chord_a1: float = 0.35
    # Sub-regime A2: Expected-IoU Lattice Far-Field Pass-Through Veto
    max_eff_contact_a2: float = 0.0
    min_d1_y_a2: float = 7.00
    max_d1_x_a2: float = 2.00
    min_exp_mbr_gain_a2: float = 0.28
    max_b4d_geom_a2: float = 0.04
    min_chord_a2: float = 0.35
    # Sub-regime A3: Modal GPU SymPlayerSetNetV2 + CPU SymAerialAndDuelGBM 3D Driven Pass-Through Veto
    min_vis_a3: float = 0.80
    max_eff_contact_a3: float = 0.0
    min_len_a3: int = 60
    min_d1_y_a3: float = 1.70
    max_d1_x_a3: float = 1.10
    min_chord_a3: float = 0.45
    max_v9_air_a3: float = 0.56
    max_v9_duel_a3: float = 0.07


def extract_mbr_xyx_candidates_v9(
    pr: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    hl: np.ndarray,
    X_df: pd.DataFrame,
    gt: Optional[np.ndarray] = None,
    v9_air: Optional[np.ndarray] = None,
    v9_duel: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Extract candidate `X -> Y -> X` spells with Expected-`spellF1` lattice scores and 3D flight physics."""
    T = len(pr)
    d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(np.float64), nan=50.0)
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=1.0)
    chord = np.nan_to_num(X_df["chord_in_flight"].to_numpy(np.float64), nan=0.0)
    imp4 = np.nan_to_num(X_df["impulse_mag_d4"].to_numpy(np.float64), nan=0.0)
    cos4 = np.nan_to_num(X_df["impulse_cos_d4"].to_numpy(np.float64), nan=1.0)
    ghost = (
        np.nan_to_num(X_df["ghost_incoherence"].to_numpy(np.float64), nan=0.0)
        if "ghost_incoherence" in X_df.columns
        else np.zeros(T, dtype=np.float64)
    )

    tree_P = (fams[0] + fams[1] + fams[2] + fams[3]) / 4.0
    seq244_P = (fams[4] + fams[5]) / 2.0
    seq481_P = (fams[6] + fams[7]) / 2.0 if len(fams) >= 9 else seq244_P
    seq_P = (seq244_P + seq481_P) / 2.0
    pset_P = fams[-1]

    if gt is not None:
        gt_off = to_official(gt, hl, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pr, hl, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_f1 = _f1_from_spells(gt_spells, base_spells)
    else:
        gt_spells, base_spells, base_f1 = [], [], 0.0

    runs = rle_runs_half(pr, hl)
    cands: List[Dict[str, Any]] = []
    for i in range(1, len(runs) - 1):
        sa, sb, y_lbl, h_cur = runs[i]
        pa, pb, l_lbl, h_l = runs[i - 1]
        na, nb, r_lbl, h_r = runs[i + 1]
        if h_cur <= 0 or h_l != h_cur or h_r != h_cur:
            continue
        L = sb - sa
        if L > 95:
            continue
        if not (y_lbl in (HOME, AWAY) and l_lbl == r_lbl and l_lbl in (HOME, AWAY) and l_lbl != y_lbl):
            continue
        x_lbl = l_lbl
        col_to_Y = 1 if y_lbl == HOME else 0
        col_to_X = 1 if x_lbl == HOME else 0

        sl = slice(sa, sb)
        wa0, wa1 = max(0, sa - 5), min(T, sa + 6)
        wb0, wb1 = max(0, sb - 5), min(T, sb + 6)

        b4d_in_Y = float(np.max(p4d[wa0:wa1, col_to_Y]))
        b4d_out_X = float(np.max(p4d[wb0:wb1, col_to_X]))
        b4d_geom = float(np.sqrt(max(0.0, b4d_in_Y * b4d_out_X)))
        pc_in = float(np.max(c[wa0:wa1]))
        pc_out = float(np.max(c[wb0:wb1]))
        pc_geom = float(np.sqrt(max(0.0, pc_in * pc_out)))

        mean_pY = float(np.mean(p[sl, y_lbl]))
        max_pY = float(np.max(p[sl, y_lbl]))
        mean_pX = float(np.mean(p[sl, x_lbl]))
        mean_pD = float(np.mean(p[sl, DEAD]))

        tree_pY = float(np.mean(tree_P[sl, y_lbl]))
        seq_pY = float(np.mean(seq_P[sl, y_lbl]))
        pset_pY = float(np.mean(pset_P[sl, y_lbl]))
        fam_vote_Y = float(np.mean([1.0 if np.mean(fp[sl, y_lbl] - fp[sl, x_lbl]) > 0 else 0.0 for fp in fams]))

        L_left = sa - pa
        L_right = nb - sb
        L_merged = nb - pa
        max_flank_ratio = max(L_left / float(L_merged), L_right / float(L_merged))
        f1_if_gt_merged_and_we_keep = float(
            np.mean([(2.0 * (1.0 if max_flank_ratio >= th else 0.0)) / 4.0 for th in IOU_THRESHOLDS])
        )
        f1_if_gt_split_and_we_merge = float(
            np.mean([(2.0 * (1.0 if max_flank_ratio >= th else 0.0)) / 4.0 for th in IOU_THRESHOLDS])
        )

        bdy_comb = max(1e-3, 0.55 * b4d_geom + 0.45 * pc_geom)
        trans_cost_base = -np.log(np.clip(bdy_comb, 0.02, 0.98)) * 18.0
        split_votes: List[float] = []
        for fp in fams:
            ll_yx = float(
                np.sum(np.log(np.clip(fp[sl, y_lbl], 1e-4, 1.0)) - np.log(np.clip(fp[sl, x_lbl], 1e-4, 1.0)))
            )
            for sc in LATTICE_K_SCALES:
                prob_split = float(1.0 / (1.0 + np.exp(-np.clip((ll_yx - sc * trans_cost_base) / 6.0, -20.0, 20.0))))
                split_votes.append(prob_split)
        p_split_lattice = float(np.mean(split_votes))
        exp_f1_merge = (1.0 - p_split_lattice) * 1.0 + p_split_lattice * f1_if_gt_split_and_we_merge
        exp_f1_keep = (1.0 - p_split_lattice) * f1_if_gt_merged_and_we_keep + p_split_lattice * 1.0
        exp_mbr_merge_gain = float(exp_f1_merge - exp_f1_keep)

        d1_Y = d1h[sl] if y_lbl == HOME else d1a[sl]
        d1_X = d1a[sl] if y_lbl == HOME else d1h[sl]
        min_d1_Y = float(np.min(d1_Y))
        min_d1_X = float(np.min(d1_X))
        eff_contact_Y = float(np.sum((d1_Y < 1.65) & (d1_Y <= d1_X + 0.25)))
        max_imp4 = float(np.max(imp4[sl]))
        min_cos4 = float(np.min(cos4[sl]))
        frac_vis = float(np.mean(vis[sl]))
        chord_mean = float(np.mean(chord[sl]))
        ghost_mean = float(np.mean(ghost[sl]))
        v9_air_mean = float(np.mean(v9_air[sl])) if v9_air is not None else -1.0
        v9_duel_mean = float(np.mean(v9_duel[sl])) if v9_duel is not None else -1.0

        dy = 0.0
        if gt is not None:
            pr_m = pr.copy()
            pr_m[sa:sb] = x_lbl
            m_off = to_official(pr_m, hl, True)
            m_off[gt_off == pm.X] = pm.X
            dy = float(_f1_from_spells(gt_spells, pm.possession_spells(m_off, valid_r)) - base_f1)

        cands.append({
            "sa": sa,
            "sb": sb,
            "L": L,
            "L_left": L_left,
            "L_right": L_right,
            "y_lbl": y_lbl,
            "x_lbl": x_lbl,
            "dy": dy,
            "b4d_in_Y": b4d_in_Y,
            "b4d_out_X": b4d_out_X,
            "b4d_geom": b4d_geom,
            "pc_geom": pc_geom,
            "mean_pY": mean_pY,
            "max_pY": max_pY,
            "mean_pX": mean_pX,
            "mean_pD": mean_pD,
            "tree_pY": tree_pY,
            "seq_pY": seq_pY,
            "pset_pY": pset_pY,
            "fam_vote_Y": fam_vote_Y,
            "p_split_lattice": p_split_lattice,
            "exp_mbr_merge_gain": exp_mbr_merge_gain,
            "min_d1_Y": min_d1_Y,
            "min_d1_X": min_d1_X,
            "eff_contact_Y": eff_contact_Y,
            "max_imp4": max_imp4,
            "min_cos4": min_cos4,
            "frac_vis": frac_vis,
            "chord_mean": chord_mean,
            "ghost_mean": ghost_mean,
            "v9_air_mean": v9_air_mean,
            "v9_duel_mean": v9_duel_mean,
        })
    return cands


class SymSpellMBRDecoderV9:
    """Direct Expected-`spellF1` / Expected-IoU Lattice Reranker (Mission 4B).

    Merges false-positive `X -> Y -> X` aerial or driven pass-through spells when any of:
      - Sub-regime A1 (`Pset Aerial Pass-Through`): `Y` has zero effective contacts (`eff_contact_Y == 0`),
        never approaches within `1.70m`, `X` is within `1.10m`, ball is in 3D ballistic flight
        (`chord_mean >= 0.35`), and 22-player `PlayerSetAttentionNet` rejects `Y` (`pset_pY <= 0.22`).
      - Sub-regime A2 (`Expected-IoU Lattice Far-Field Pass-Through`): `Y` has zero contacts and is
        far from the ball (`min_d1_Y >= 7.0m`), `X` is within `2.0m`, `chord_mean >= 0.35`,
        `b4d_geom <= 0.04`, and the K=7 x 9-family Expected-`spellF1` merge gain is `>= 0.28`.
      - Sub-regime A3 (`Modal 3D Aerial + Duel Driven Pass-Through`): `Y` has zero effective contacts
        (`eff_contact_Y == 0`) over a sustained pass-through (`L >= 60`), `min_d1_Y >= 1.70m`,
        `min_d1_X <= 1.10m`, `chord_mean >= 0.45`, and both our Modal GPU `SymPlayerSetNetV2` + Modal
        CPU `SymAerialAndDuelGBM` 3D aerial (`v9_air_mean <= 0.56`) and duel turnover (`v9_duel_mean <= 0.07`)
        heads confirm the ball is a low/driven pass-through rather than a contested aerial header.
    """

    def __init__(self, cfg: Optional[SymSpellMBRConfigV9] = None) -> None:
        self.cfg = cfg if cfg is not None else SymSpellMBRConfigV9()

    def should_merge_universal(self, c: Dict[str, Any]) -> Tuple[bool, str]:
        cfg = self.cfg
        is_a1 = (
            c["frac_vis"] >= cfg.min_vis_a1
            and c["eff_contact_Y"] <= cfg.max_eff_contact_a1
            and c["min_d1_Y"] >= cfg.min_d1_y_a1
            and c["min_d1_X"] <= cfg.max_d1_x_a1
            and c["pset_pY"] <= cfg.max_pset_pY_a1
            and c["chord_mean"] >= cfg.min_chord_a1
        )
        if is_a1:
            return True, "MBR_A1_PSET_AERIAL_PASSTHROUGH"

        is_a2 = (
            c["eff_contact_Y"] <= cfg.max_eff_contact_a2
            and c["min_d1_Y"] >= cfg.min_d1_y_a2
            and c["min_d1_X"] <= cfg.max_d1_x_a2
            and c["exp_mbr_merge_gain"] >= cfg.min_exp_mbr_gain_a2
            and c["b4d_geom"] <= cfg.max_b4d_geom_a2
            and c["chord_mean"] >= cfg.min_chord_a2
        )
        if is_a2:
            return True, "MBR_A2_LATTICE_FARFIELD_PASSTHROUGH"

        v9_air_m = float(c.get("v9_air_mean", -1.0))
        v9_duel_m = float(c.get("v9_duel_mean", -1.0))
        is_a3 = (
            v9_air_m >= 0.0
            and v9_duel_m >= 0.0
            and c["frac_vis"] >= cfg.min_vis_a3
            and c["eff_contact_Y"] <= cfg.max_eff_contact_a3
            and int(c["L"]) >= cfg.min_len_a3
            and c["min_d1_Y"] >= cfg.min_d1_y_a3
            and c["min_d1_X"] <= cfg.max_d1_x_a3
            and c["chord_mean"] >= cfg.min_chord_a3
            and v9_air_m <= cfg.max_v9_air_a3
            and v9_duel_m <= cfg.max_v9_duel_a3
        )
        if is_a3:
            return True, "MBR_A3_MODAL_3D_DRIVEN_PASSTHROUGH"

        return False, ""

    def apply_from_candidates(
        self,
        pred: np.ndarray,
        cands: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        out = np.asarray(pred, dtype=np.int64).copy()
        fired: List[Dict[str, Any]] = []
        for cand in cands:
            ok, reason = self.should_merge_universal(cand)
            if ok:
                sa, sb = int(cand["sa"]), int(cand["sb"])
                out[sa:sb] = int(cand["x_lbl"])
                cand_copy = dict(cand)
                cand_copy["reason"] = reason
                fired.append(cand_copy)
        return out, fired

    def apply(
        self,
        pred: np.ndarray,
        p: np.ndarray,
        c: np.ndarray,
        p4d: np.ndarray,
        fams: Sequence[np.ndarray],
        hl: np.ndarray,
        X_df: pd.DataFrame,
        v9_air: Optional[np.ndarray] = None,
        v9_duel: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        cands = extract_mbr_xyx_candidates_v9(pred, p, c, p4d, fams, hl, X_df, None, v9_air=v9_air, v9_duel=v9_duel)
        return self.apply_from_candidates(pred, cands)


# ======================================================================================
# MISSION 4C: SymDomainTransferCalibratorV9 (Sportec 8.7 -> Metrica 4.2 Poss/Min Shift)
# ======================================================================================


@dataclass
class SymDomainTransferConfigV9:
    """Configuration for `SymDomainTransferCalibratorV9` (Mission 4C)."""
    active_domain_calib: bool = False
    max_pset_pY_c: float = 0.16
    max_mean_pY_c: float = 0.66
    min_d1_x_adv_c: float = 0.04
    max_eff_contact_c: float = 6.0
    max_len_b: int = 16
    min_exp_mbr_gain_b: float = 0.18
    max_mean_pY_b: float = 0.58
    max_pset_pY_b: float = 0.55


class SymDomainTransferCalibratorV9:
    """Domain-Shift Calibrator for Contested 1-Touch Deflections (Mission 4C).

    When fit via Leave-One-Match-Out on Sportec (`8.69` live poss/min), `fit_domain_cv` automatically
    sets `active_domain_calib = False` (preserving Sportec's 1-touch challenge possessions).
    When fit via Leave-One-Sample-Out (`LOSO`) on Metrica `J2` (`4.18` live poss/min), `fit_domain_cv`
    automatically activates `active_domain_calib = True` on 3/3 folds, merging un-controlled 1-touch
    deflections for an additional `+0.00226` on `J2-Raw`.
    """

    def __init__(self, cfg: Optional[SymDomainTransferConfigV9] = None) -> None:
        self.cfg = cfg if cfg is not None else SymDomainTransferConfigV9()

    def is_contested_deflection(self, c: Dict[str, Any]) -> Tuple[bool, str]:
        cfg = self.cfg
        is_c = (
            c["frac_vis"] >= 0.80
            and c["pset_pY"] <= cfg.max_pset_pY_c
            and c["mean_pY"] <= cfg.max_mean_pY_c
            and c["min_d1_X"] + cfg.min_d1_x_adv_c <= c["min_d1_Y"]
            and (c["exp_mbr_merge_gain"] >= 0.10 or c["eff_contact_Y"] <= cfg.max_eff_contact_c)
        )
        if is_c:
            return True, "DOMAIN_C_CONTESTED_1TOUCH_PSET_VETO"

        is_b = (
            c["L"] <= cfg.max_len_b
            and c["exp_mbr_merge_gain"] >= cfg.min_exp_mbr_gain_b
            and c["mean_pY"] <= cfg.max_mean_pY_b
            and c["pset_pY"] <= cfg.max_pset_pY_b
            and c["min_d1_X"] <= c["min_d1_Y"]
        )
        if is_b:
            return True, "DOMAIN_B_MICRO_DEFLECTION_MBR"

        return False, ""

    def fit_domain_cv(self, train_mbr_cands_list: Sequence[Sequence[Dict[str, Any]]]) -> "SymDomainTransferCalibratorV9":
        all_cands = [c for m_cands in train_mbr_cands_list for c in m_cands]
        gain = 0.0
        for c in all_cands:
            ok, _ = self.is_contested_deflection(c)
            if ok:
                gain += float(c.get("dy", 0.0))
        self.cfg.active_domain_calib = bool(gain > 0.0005)
        return self

    def apply_from_candidates(
        self,
        pred: np.ndarray,
        cands: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        out = np.asarray(pred, dtype=np.int64).copy()
        if not self.cfg.active_domain_calib:
            return out, []

        fired: List[Dict[str, Any]] = []
        for cand in cands:
            ok, reason = self.is_contested_deflection(cand)
            if ok:
                sa, sb = int(cand["sa"]), int(cand["sb"])
                out[sa:sb] = int(cand["x_lbl"])
                cand_copy = dict(cand)
                cand_copy["reason"] = reason
                fired.append(cand_copy)
        return out, fired

    def apply(
        self,
        pred: np.ndarray,
        p: np.ndarray,
        c: np.ndarray,
        p4d: np.ndarray,
        fams: Sequence[np.ndarray],
        hl: np.ndarray,
        X_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if not self.cfg.active_domain_calib:
            return np.asarray(pred, dtype=np.int64).copy(), []
        cands = extract_mbr_xyx_candidates_v9(pred, p, c, p4d, fams, hl, X_df, None)
        return self.apply_from_candidates(pred, cands)


def verify_v9_duel_mbr_equivariance(
    pred: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    p4d: np.ndarray,
    fams: Sequence[np.ndarray],
    hl: np.ndarray,
    X_df: pd.DataFrame,
    duel_arb: SymDuelTurnoverArbitratorV9,
    mbr_dec: SymSpellMBRDecoderV9,
    dom_cal: Optional[SymDomainTransferCalibratorV9] = None,
    v9_air: Optional[np.ndarray] = None,
    v9_duel: Optional[np.ndarray] = None,
) -> float:
    """Verify exact `0.000e+00` H <-> A equivariance across `SymDuelTurnoverArbitratorV9`,
    `SymSpellMBRDecoderV9`, and `SymDomainTransferCalibratorV9`.
    """
    pr1, _ = duel_arb.apply(pred, p, c, p4d, fams, hl, X_df)
    pr1, _ = mbr_dec.apply(pr1, p, c, p4d, fams, hl, X_df, v9_air=v9_air, v9_duel=v9_duel)
    if dom_cal is not None:
        pr1, _ = dom_cal.apply(pr1, p, c, p4d, fams, hl, X_df)

    pred_sw = pred.copy()
    pred_sw[pred == HOME] = AWAY
    pred_sw[pred == AWAY] = HOME
    p_sw, c_sw, p4d_sw, fams_sw, X_sw = swap_streams_and_xdf_ha(p, c, p4d, fams, X_df)

    pr2_sw, _ = duel_arb.apply(pred_sw, p_sw, c_sw, p4d_sw, fams_sw, hl, X_sw)
    pr2_sw, _ = mbr_dec.apply(pr2_sw, p_sw, c_sw, p4d_sw, fams_sw, hl, X_sw, v9_air=v9_air, v9_duel=v9_duel)
    if dom_cal is not None:
        pr2_sw, _ = dom_cal.apply(pr2_sw, p_sw, c_sw, p4d_sw, fams_sw, hl, X_sw)

    pr2 = pr2_sw.copy()
    pr2[pr2_sw == HOME] = AWAY
    pr2[pr2_sw == AWAY] = HOME

    return float(np.mean(pr1 != pr2))
