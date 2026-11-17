"""EXP-09: Head-Decoupled Multi-Family Synthesis & Zero-Label Transductive Calibration (`src/decode/transductive_calib.py`).

Implements:
1. `combine_4tree_decoupled`:
   - Exact Red-Team Decoupled 4-Tree Synthesis across `SymGBM_Plus`, `SymGBM_PhysV2`, `SymCatBoost_PhysV2`, `SymXGB_PhysV2`.
   - Dead-ball head `P_tree(D)` excludes CatBoost (`w_d = (0.35, 0.45, 0.00, 0.20)`) because CatBoost's `head_alive`
     splits heavily on `ball_speed` (which is `NaN` in BAR-ATH's 629 ball blackouts), whereas `SymXGB_PhysV2` relies on
     22-player collective speed (`all_speed_h12_r3`, `all_speed_h6_r3`) and `SymGBM_PhysV2` leads on Sportec `P_D`.
   - Live possession head `logit P_tree(H | alive)` combines all 4 tree families in log-odds space
     (`w_ha = (0.35, 0.20, 0.25, 0.20)`), leveraging `SymCatBoost_PhysV2`'s #1 standalone live possession accuracy.
2. `synthesize_head_decoupled_proba`:
   - Combines the decoupled 4-tree ensemble, physical ghost-gated `SymGBM_v6`, and gap-gated sequence experts
     (`SymTCN_TTA` + `SymBiGRU_TTA`) with independent weights on the `Alive vs Dead` head (`w_tree_d`) and
     the conditional `H vs A | Alive` head (`w_tree_ha`), plus multi-family `p_change` blending and log-odds TV augmentation.
3. `calibrate_match_transductive`:
   - Zero-label match-level transductive calibration with strict `0.000e+00` H <-> A equivariance:
     (a) Decoupled temperature scaling (`T_alive`, `T_ha`)
     (b) Kinematics-guided blackout dead-ball calibration (using `ball_visible` and symmetric `frac_still_all = 0.5*(frac_still_H + frac_still_A)`)
     (c) Damped prior-shift alignment on `logit(P_D)`
     (d) 1-round symmetric spell-interior pseudo-label smoothing.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data.common import AWAY, DEAD, HOME
from src.decode.spell_reranker import rle_runs


@dataclass
class DecoupledSynthConfig:
    """Configuration for Head-Decoupled Multi-Family Synthesis (`EXP-09`)."""
    # 4-Tree weights: (SymGBM_Plus, SymGBM_PhysV2, SymCatBoost_PhysV2, SymXGB_PhysV2)
    w_tree_d_tuple: tuple[float, float, float, float] = (0.35, 0.45, 0.00, 0.20)
    w_tree_ha_tuple: tuple[float, float, float, float] = (0.35, 0.20, 0.25, 0.20)
    # 4-Seq weights when PhysV2 sequences are provided: (TCN_244, BiGRU_244, TCN_481, BiGRU_481)
    w_seq_d_tuple: tuple[float, float, float, float] = (0.45, 0.05, 0.30, 0.20)
    w_seq_ha_tuple: tuple[float, float, float, float] = (0.30, 0.15, 0.25, 0.30)
    # Tree vs Sequence weights on the two heads
    w_tree_d: float = 0.55        # Weight on tree ensemble for P(D) vs sequence P(D)
    w_tree_ha: float = 0.55       # Weight on tree ensemble for P(H, A | alive) vs sequence P(H, A | alive)
    d_blend_mode: str = "linear"  # "linear", "hybrid", or "logit" for P(D)
    ha_blend_mode: str = "hybrid" # "hybrid" (0.5*lin + 0.5*geom), "linear", or "logit" for P(H, A | alive)
    # Sequence internal blending
    use_phys_ghost_gate: bool = True
    use_phys_gap_gate: bool = True
    const_bigru_ha: float = 0.0   # Minimum constant BiGRU weight outside gaps (when using 2-seq)
    # Boundary change stream blending & TV augmentation
    w_change_tree: float = 0.50
    alpha_tv: float = 0.20


@dataclass
class TransductiveCalibConfig:
    """Configuration for Zero-Label Match-Level Transductive Calibration (`EXP-09`)."""
    T_alive: float = 1.0          # Temperature on logit(P_D) around d_pivot
    T_ha: float = 1.0             # Temperature on antisymmetric logit P(H | alive)
    d_logit_shift: float = 0.0    # Additive shift on logit(P_D)
    blackout_still_gain: float = 0.0  # Kinematic dead-ball adjustment during ball blackouts (ball_visible == 0)
    still_pivot: float = 0.38     # Pivot for frac_still_all during blackouts
    em_prior_strength: float = 0.0    # Damped EM prior shift weight in [0, 0.5]
    target_d_prior: float = 0.275     # Reference dead-ball prior for damped EM anchoring
    pseudo_alpha: float = 0.0     # 1-round symmetric spell-interior smoothing weight in [0, 0.20]
    pseudo_margin: float = 0.20   # Minimum interior probability margin for pseudo-label refinement


def _symmetric_conditional_logit_ha(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute strictly antisymmetric conditional log-odds `log(P(H|alive) / P(A|alive))` = `log(P_H) - log(P_A)`.

    Under H <-> A swap (`p[:, 0] <-> p[:, 1]`), this returns the exact negation `-logit_ha` (`0.000e+00` error).
    """
    pH = np.clip(np.asarray(p[:, HOME], dtype=np.float64), eps, 1.0)
    pA = np.clip(np.asarray(p[:, AWAY], dtype=np.float64), eps, 1.0)
    return np.log(pH) - np.log(pA)


def _symmetric_sigmoid_pair(logit_ha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert antisymmetric `logit_ha` to `(r_H, r_A)` with exact H <-> A symmetry (`r_H(-x) == r_A(x)`)."""
    x = np.clip(np.asarray(logit_ha, dtype=np.float64), -20.0, 20.0)
    eh = np.exp(0.5 * x)
    ea = np.exp(-0.5 * x)
    denom = eh + ea
    return eh / denom, ea / denom


def _logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    xc = np.clip(np.asarray(x, dtype=np.float64), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-xc))


def combine_4tree_decoupled(
    g_plus: np.ndarray,
    g_physv2: np.ndarray,
    cb_physv2: np.ndarray,
    xgb_physv2: np.ndarray,
    w_d: tuple[float, float, float, float] = (0.35, 0.45, 0.00, 0.20),
    w_ha: tuple[float, float, float, float] = (0.35, 0.20, 0.25, 0.20),
) -> np.ndarray:
    """Combine the 4 symmetric tree families with decoupled Dead-Ball (`P_D`) and Live (`H|alive`) weights.

    Guarantees exact `0.000e+00` H <-> A equivariance.
    """
    trees = (
        np.asarray(g_plus, dtype=np.float64),
        np.asarray(g_physv2, dtype=np.float64),
        np.asarray(cb_physv2, dtype=np.float64),
        np.asarray(xgb_physv2, dtype=np.float64),
    )
    wd = np.asarray(w_d, dtype=np.float64)
    wd = wd / np.maximum(np.sum(wd), 1e-8)

    wha = np.asarray(w_ha, dtype=np.float64)
    wha = wha / np.maximum(np.sum(wha), 1e-8)

    pD = np.zeros(len(g_plus), dtype=np.float64)
    logit_ha = np.zeros(len(g_plus), dtype=np.float64)

    for m_p, w_d_i, w_ha_i in zip(trees, wd, wha):
        if w_d_i > 0.0:
            pD += w_d_i * np.clip(m_p[:, DEAD], 0.0, 1.0)
        if w_ha_i > 0.0:
            logit_ha += w_ha_i * _symmetric_conditional_logit_ha(m_p)

    pD = np.clip(pD, 1e-6, 1.0 - 1e-6)
    alive = 1.0 - pD
    rH, rA = _symmetric_sigmoid_pair(logit_ha)

    out = np.empty((len(g_plus), 3), dtype=np.float64)
    out[:, HOME] = alive * rH
    out[:, AWAY] = alive * rA
    out[:, DEAD] = pD
    return np.round(out, 12)


def _conditional_ha_pair(p: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly symmetric conditional `(r_H, r_A)` summing to 1.0."""
    pH = np.clip(np.asarray(p[:, HOME], dtype=np.float64), eps, 1.0)
    pA = np.clip(np.asarray(p[:, AWAY], dtype=np.float64), eps, 1.0)
    s = pH + pA
    return pH / s, pA / s


def compute_seq481_gate(
    X_phys: pd.DataFrame,
    half: np.ndarray,
    w_ghost: np.ndarray | float = 0.0,
    vis_weight: float = 0.95,
    gap_weight: float = 0.85,
    max_ghost: float = 0.45,
) -> np.ndarray:
    """Compute kinematic + visibility gate `alpha_p2(t)` for weighting `Seq_481` (`SymTCN_PhysV2 + SymBiGRU_PhysV2`) vs `Seq_244`.

    Kinematic trajectory jitter detection:
    - Clean ball trajectories (`Sportec v3`, `Metrica Sample`, `BAR-ATH`) have active-half `mean(impulse_mag_d4) in [0.88, 1.58]`.
    - Synthetic-ball-jitter degraded trajectories (`Sportec v6`) have `mean(impulse_mag_d4) in [1.85, 2.28]`.
    - `jitter_gate = clip(1.0 - (mean_imp4 - 1.62) / (1.84 - 1.62), 0.05, 1.0)` cleanly routes ~90-95% weight
      to `Seq_481` on clean trajectories and ~95% weight to `Seq_244` on synthetic-jitter degraded trajectories,
      without creating sharp frame-to-frame step-jumps at `ball_visible` transitions.
    - Guarantees exact `0.000e+00` H <-> A equivariance (`impulse_mag_d4` and `ball_visible` are strictly H<->A invariant).
    """
    T = len(half)
    active = half > 0
    if not np.any(active):
        active = np.ones(T, dtype=bool)

    vis = (
        np.nan_to_num(X_phys["ball_visible"].to_numpy(dtype=np.float64), nan=0.0)
        if "ball_visible" in X_phys.columns
        else np.ones(T, dtype=np.float64)
    )
    if "impulse_mag_d4" in X_phys.columns:
        imp4 = np.nan_to_num(X_phys["impulse_mag_d4"].to_numpy(dtype=np.float64), nan=0.0)
        imp4_mean = float(np.mean(imp4[active]))
    else:
        imp4_mean = 1.25
    jitter_gate = float(np.clip(1.0 - (imp4_mean - 1.62) / (1.84 - 1.62), 0.05, 1.0))

    if isinstance(w_ghost, np.ndarray):
        wg = w_ghost[:, 0] if w_ghost.ndim == 2 else w_ghost
        g_ghost = np.clip(np.asarray(wg, dtype=np.float64) / max(max_ghost, 1e-6), 0.0, 1.0)
    else:
        g_ghost = np.full(T, np.clip(float(w_ghost) / max(max_ghost, 1e-6), 0.0, 1.0), dtype=np.float64)

    base_vis = gap_weight + (vis_weight - gap_weight) * vis
    raw = jitter_gate * base_vis * (1.0 - 0.30 * g_ghost)
    smooth = np.copy(raw)
    kernel = np.ones(25, dtype=np.float64) / 25.0
    for hv in np.unique(half):
        if hv <= 0:
            continue
        idx = np.flatnonzero(half == hv)
        if len(idx) > 0:
            smooth[idx] = np.convolve(raw[idx], kernel, mode="same")
    return np.round(np.clip(smooth, 0.0, 1.0), 10)


def synthesize_head_decoupled_proba(
    g_plus: np.ndarray,
    g_physv2: np.ndarray,
    cb_physv2: np.ndarray,
    xgb_physv2: np.ndarray,
    tcn_proba: np.ndarray,
    bigru_proba: np.ndarray,
    tr_plus: np.ndarray,
    tr_physv2: np.ndarray,
    tcn_change: np.ndarray,
    bigru_change: np.ndarray,
    half: np.ndarray,
    g_v6: np.ndarray | None = None,
    tr_v6: np.ndarray | None = None,
    w_ghost: np.ndarray | float = 0.0,
    w_gap: np.ndarray | float = 0.0,
    cfg: DecoupledSynthConfig | None = None,
    tcn_physv2_proba: np.ndarray | None = None,
    bigru_physv2_proba: np.ndarray | None = None,
    tcn_physv2_change: np.ndarray | None = None,
    bigru_physv2_change: np.ndarray | None = None,
    alpha_p2: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize multi-family probabilities and boundary `p_change` with head-decoupled weights.

    Supports both 6-family (4-Tree + 2-Seq) and 8-family (4-Tree + 4-Seq MultiRep, optionally gated by `alpha_p2(t)`) synthesis.
    Combines Head-Decoupled Linear synthesis (`p_lin`) with 3-Class Head-Decoupled Product-of-Experts (`p_geo`)
    so Tree vs Sequence H/A disagreement naturally elevates `P(DEAD)` while preserving exact `0.000e+00` H <-> A equivariance.
    """
    if cfg is None:
        cfg = DecoupledSynthConfig()

    # 1. Decoupled 4-Tree synthesis
    p_tree = combine_4tree_decoupled(
        g_plus, g_physv2, cb_physv2, xgb_physv2,
        w_d=cfg.w_tree_d_tuple,
        w_ha=cfg.w_tree_ha_tuple,
    )
    c_tree = 0.50 * np.asarray(tr_plus, dtype=np.float64) + 0.50 * np.asarray(tr_physv2, dtype=np.float64)

    # 2. Optional physical ghost gate blending in SymGBM_v6_TTA
    if cfg.use_phys_ghost_gate and g_v6 is not None and (isinstance(w_ghost, np.ndarray) or float(w_ghost) > 0.0):
        wg = np.asarray(w_ghost, dtype=np.float64)
        if wg.ndim == 1:
            wg = wg[:, None]
        p_tree = (1.0 - wg) * p_tree + wg * np.asarray(g_v6, dtype=np.float64)
        if tr_v6 is not None:
            wg_1d = wg[:, 0] if wg.ndim == 2 else wg
            c_tree = (1.0 - wg_1d) * c_tree + wg_1d * np.asarray(tr_v6, dtype=np.float64)

    # 3. Sequence expert: 2-Seq (TCN_244 + BiGRU_244) or 4-Seq MultiRep (+ TCN_481 + BiGRU_481)
    tcn_p = np.asarray(tcn_proba, dtype=np.float64)
    gru_p = np.asarray(bigru_proba, dtype=np.float64)
    tcn_c = np.asarray(tcn_change, dtype=np.float64)
    gru_c = np.asarray(bigru_change, dtype=np.float64)

    if cfg.use_phys_gap_gate and (isinstance(w_gap, np.ndarray) or float(w_gap) > 0.0):
        wgap_1d = np.asarray(w_gap, dtype=np.float64)
        if wgap_1d.ndim == 2:
            wgap_1d = wgap_1d[:, 0]
        w_gru = np.maximum(cfg.const_bigru_ha, wgap_1d)
    else:
        w_gru = np.full(len(tcn_p), cfg.const_bigru_ha, dtype=np.float64)

    # Seq_244 (v1ctl) internal combination
    rH_tcn, rA_tcn = _conditional_ha_pair(tcn_p)
    rH_gru, rA_gru = _conditional_ha_pair(gru_p)
    rH_244 = (1.0 - w_gru) * rH_tcn + w_gru * rH_gru
    rA_244 = (1.0 - w_gru) * rA_tcn + w_gru * rA_gru
    pD_244 = np.clip(tcn_p[:, DEAD], 1e-6, 1.0 - 1e-6)
    c_244 = (1.0 - 0.5 * w_gru) * tcn_c + (0.5 * w_gru) * gru_c

    if tcn_physv2_proba is not None and bigru_physv2_proba is not None:
        tcn2_p = np.asarray(tcn_physv2_proba, dtype=np.float64)
        gru2_p = np.asarray(bigru_physv2_proba, dtype=np.float64)
        tcn2_c = np.asarray(tcn_physv2_change if tcn_physv2_change is not None else tcn_c, dtype=np.float64)
        gru2_c = np.asarray(bigru_physv2_change if bigru_physv2_change is not None else gru_c, dtype=np.float64)

        if alpha_p2 is not None:
            ap2 = np.asarray(alpha_p2, dtype=np.float64)
            if ap2.ndim == 2:
                ap2 = ap2[:, 0]
            # Seq_481 (v1ctl_physics_v2): 100% SymTCN_PhysV2 for P(DEAD), 80/20 SymTCN_PhysV2 + SymBiGRU_PhysV2 for H/A
            w_gru481 = np.maximum(0.20, 0.50 * w_gru)
            rH_t2, rA_t2 = _conditional_ha_pair(tcn2_p)
            rH_g2, rA_g2 = _conditional_ha_pair(gru2_p)
            rH_481 = (1.0 - w_gru481) * rH_t2 + w_gru481 * rH_g2
            rA_481 = (1.0 - w_gru481) * rA_t2 + w_gru481 * rA_g2
            pD_481 = np.clip(tcn2_p[:, DEAD], 1e-6, 1.0 - 1e-6)
            c_481 = 0.85 * tcn2_c + 0.15 * gru2_c

            pD_seq = np.clip((1.0 - ap2) * pD_244 + ap2 * pD_481, 1e-6, 1.0 - 1e-6)
            rH_seq = (1.0 - ap2) * rH_244 + ap2 * rH_481
            rA_seq = (1.0 - ap2) * rA_244 + ap2 * rA_481
            c_seq = (1.0 - ap2) * c_244 + ap2 * c_481
        else:
            wsd = np.asarray(cfg.w_seq_d_tuple, dtype=np.float64)
            wsd = wsd / np.maximum(np.sum(wsd), 1e-8)
            wsha = np.asarray(cfg.w_seq_ha_tuple, dtype=np.float64)
            wsha = wsha / np.maximum(np.sum(wsha), 1e-8)

            seqs = (tcn_p, gru_p, tcn2_p, gru2_p)
            seq_cs = (tcn_c, gru_c, tcn2_c, gru2_c)
            pD_seq = sum(w_i * np.clip(sp[:, DEAD], 0.0, 1.0) for w_i, sp in zip(wsd, seqs))
            pD_seq = np.clip(pD_seq, 1e-6, 1.0 - 1e-6)

            rH_4s = sum(w_i * _conditional_ha_pair(sp)[0] for w_i, sp in zip(wsha, seqs))
            rA_4s = sum(w_i * _conditional_ha_pair(sp)[1] for w_i, sp in zip(wsha, seqs))
            rH_gru_pair = 0.5 * (_conditional_ha_pair(gru_p)[0] + _conditional_ha_pair(gru2_p)[0])
            rA_gru_pair = 0.5 * (_conditional_ha_pair(gru_p)[1] + _conditional_ha_pair(gru2_p)[1])
            rH_seq = (1.0 - 0.5 * w_gru) * rH_4s + (0.5 * w_gru) * rH_gru_pair
            rA_seq = (1.0 - 0.5 * w_gru) * rA_4s + (0.5 * w_gru) * rA_gru_pair
            c_seq = sum(w_i * sc for w_i, sc in zip(wsha, seq_cs))
    else:
        rH_seq, rA_seq = rH_244, rA_244
        pD_seq = pD_244
        c_seq = c_244

    # Construct 3-class Sequence expert probability table p_seq
    alive_seq = 1.0 - pD_seq
    p_seq = np.stack([alive_seq * rH_seq, alive_seq * rA_seq, pD_seq], axis=1)

    # 4. Head-Decoupled Tree + Sequence synthesis (3-Class PoE + Linear Hybrid)
    pt = np.clip(p_tree, 1e-6, 1.0 - 1e-6)
    ps = np.clip(p_seq, 1e-6, 1.0 - 1e-6)

    # 4a. Head-Decoupled Linear combination
    pD_lin = np.clip(cfg.w_tree_d * pt[:, DEAD] + (1.0 - cfg.w_tree_d) * ps[:, DEAD], 1e-6, 1.0 - 1e-6)
    alive_lin = 1.0 - pD_lin
    h_lin = cfg.w_tree_ha * pt[:, HOME] + (1.0 - cfg.w_tree_ha) * ps[:, HOME]
    a_lin = cfg.w_tree_ha * pt[:, AWAY] + (1.0 - cfg.w_tree_ha) * ps[:, AWAY]
    ha_sum = np.maximum(h_lin + a_lin, 1e-8)
    p_lin = np.stack([alive_lin * (h_lin / ha_sum), alive_lin * (a_lin / ha_sum), pD_lin], axis=1)

    # 4b. Head-Decoupled 3-Class Product-of-Experts (PoE) combination
    log_h = cfg.w_tree_ha * np.log(pt[:, HOME]) + (1.0 - cfg.w_tree_ha) * np.log(ps[:, HOME])
    log_a = cfg.w_tree_ha * np.log(pt[:, AWAY]) + (1.0 - cfg.w_tree_ha) * np.log(ps[:, AWAY])
    log_d = cfg.w_tree_d * np.log(pt[:, DEAD]) + (1.0 - cfg.w_tree_d) * np.log(ps[:, DEAD])
    logits = np.stack([log_h, log_a, log_d], axis=1)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(logits)
    p_geo = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    if cfg.ha_blend_mode == "logit":
        proba_out = p_geo
    elif cfg.ha_blend_mode == "linear":
        proba_out = p_lin
    else:
        proba_out = 0.50 * (p_lin + p_geo)
    proba_out = np.round(proba_out, 12)

    p_change_out = cfg.w_change_tree * c_tree + (1.0 - cfg.w_change_tree) * c_seq
    if cfg.alpha_tv > 0.0:
        from scripts.exp_f_sym_decoder_and_boundary import augment_p_change_with_logodds_tv
        p_change_out = augment_p_change_with_logodds_tv(proba_out, p_change_out, half, alpha_grad=cfg.alpha_tv)
    p_change_out = np.round(np.asarray(p_change_out, dtype=np.float64), 12)
    return proba_out, p_change_out


def calibrate_match_transductive(
    proba: np.ndarray,
    half: np.ndarray,
    X_phys: pd.DataFrame | None = None,
    cfg: TransductiveCalibConfig | None = None,
    pred_init: np.ndarray | None = None,
) -> np.ndarray:
    """Apply zero-label match-level transductive calibration with exact `0.000e+00` H <-> A equivariance.

    All operations act either on:
    - `logit_D = logit(P_D)` using strictly team-symmetric kinematics (`frac_still_all`, `ball_visible`), or
    - `logit_HA = log(P_H) - log(P_A)` via strictly odd/antisymmetric scaling (`logit_HA / T_ha`),
    guaranteeing zero H/A bias and `0.000e+00` H <-> A equivariance.
    """
    if cfg is None:
        return np.asarray(proba, dtype=np.float64).copy()

    p = np.asarray(proba, dtype=np.float64)
    T = len(p)
    pD = np.clip(p[:, DEAD], 1e-5, 1.0 - 1e-5)
    logit_d = _logit(pD)
    logit_ha = _symmetric_conditional_logit_ha(p)

    active = half > 0
    if not np.any(active):
        active = np.ones(T, dtype=bool)

    # 1. Temperature scaling on logit_D (centered around active median/mean) and antisymmetric logit_HA
    if abs(cfg.T_alive - 1.0) > 1e-6:
        pivot_d = float(np.mean(logit_d[active]))
        logit_d = pivot_d + (logit_d - pivot_d) / max(cfg.T_alive, 0.25)

    if abs(cfg.T_ha - 1.0) > 1e-6:
        logit_ha = logit_ha / max(cfg.T_ha, 0.25)

    # 2. Additive prior shift on logit_D
    if abs(cfg.d_logit_shift) > 1e-6:
        logit_d = logit_d + cfg.d_logit_shift

    # 3. Kinematics-guided blackout dead-ball calibration (active only when ball_visible == 0)
    if cfg.blackout_still_gain > 0.0 and X_phys is not None:
        vis = np.nan_to_num(X_phys["ball_visible"].to_numpy(dtype=np.float64), nan=1.0) if "ball_visible" in X_phys.columns else np.ones(T)
        if "frac_still_H" in X_phys.columns and "frac_still_A" in X_phys.columns:
            still_sym = 0.5 * (
                np.nan_to_num(X_phys["frac_still_H"].to_numpy(dtype=np.float64), nan=0.2)
                + np.nan_to_num(X_phys["frac_still_A"].to_numpy(dtype=np.float64), nan=0.2)
            )
        elif "frac_still_all" in X_phys.columns:
            still_sym = np.nan_to_num(X_phys["frac_still_all"].to_numpy(dtype=np.float64), nan=0.2)
        else:
            still_sym = np.full(T, cfg.still_pivot, dtype=np.float64)

        blackout_mask = np.clip(1.0 - vis, 0.0, 1.0)
        logit_d = logit_d + cfg.blackout_still_gain * blackout_mask * (still_sym - cfg.still_pivot)

    # 4. Damped EM prior shift on P_D toward reference / kinematic stoppage rate
    if cfg.em_prior_strength > 0.0:
        cur_pD = _sigmoid(logit_d)
        mean_d = float(np.mean(cur_pD[active]))
        # Anchor target between match mean_d and target_d_prior (damped in logit space)
        if 0.05 < mean_d < 0.85:
            raw_shift = np.log(cfg.target_d_prior / (1.0 - cfg.target_d_prior)) - np.log(mean_d / (1.0 - mean_d))
            damped_shift = np.clip(cfg.em_prior_strength * raw_shift, -0.25, 0.25)
            logit_d = logit_d + damped_shift

    # 5. Optional 1-round symmetric spell-interior pseudo-label smoothing on logit_HA
    if cfg.pseudo_alpha > 0.0 and pred_init is not None:
        runs = rle_runs(pred_init)
        pseudo_shift = np.zeros(T, dtype=np.float64)
        rH_cur, rA_cur = _symmetric_sigmoid_pair(logit_ha)
        margin_ha = np.abs(rH_cur - rA_cur)
        for a, b, lbl in runs:
            if lbl not in (HOME, AWAY):
                continue
            L = b - a
            if L < 36 or half[a] <= 0:
                continue
            # Interior slice at least 10 frames away from boundaries
            ia, ib = a + 10, b - 10
            if ib <= ia:
                continue
            mean_marg = float(np.mean(margin_ha[ia:ib]))
            if mean_marg >= cfg.pseudo_margin:
                sign = 1.0 if lbl == HOME else -1.0
                pseudo_shift[ia:ib] = sign * cfg.pseudo_alpha * min(mean_marg, 0.60)
        logit_ha = logit_ha + pseudo_shift

    pD_out = np.clip(_sigmoid(logit_d), 1e-6, 1.0 - 1e-6)
    alive_out = 1.0 - pD_out
    rH_out, rA_out = _symmetric_sigmoid_pair(logit_ha)

    out = np.empty((T, 3), dtype=np.float64)
    out[:, HOME] = alive_out * rH_out
    out[:, AWAY] = alive_out * rA_out
    out[:, DEAD] = pD_out
    return np.round(out, 12)
