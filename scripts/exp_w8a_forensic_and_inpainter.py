"""Wave 8A (EXP-13 & EXP-14): Forensic Gate Audit & `SymBallInpainterNet` Clean-Ball Teacher Distillation.

Executes:
1. EXP-13 (`src/decode/hybrid_gate_v8.py`):
   - Forensic diagnosis of the 0.57% (901-frame) discrepancy on BAR-ATH and the Sportec v6 stoppage/gap dilution spike.
   - Exact `0.000e+00` H <-> A equivariance audit of `compute_hybrid_gate_v8`, `compute_w_clean_v8`,
     `compute_inpaint_gate_v8`, and `blend_6fam_heads_v8(adaptive_regime=True)`.
2. EXP-14 (`src/models/ball_inpainter.py`):
   - Part A: `SymBallPositionInpainter` (11-basis equivariant 2D ball-deviation regressor in 7-fold LOMO on Sportec v3/v6).
     Evaluates spatial MAE (m) and RMSE (m) vs linear chord interpolation across all gaps, active in-play gaps, and dead gaps.
   - Part B: Clean-Ball Teacher distillation into `SymGapDistilledExpert` (78 equivariant inpainted features) in 7-fold LOMO.
   - Evaluates multi-judge `spell_f1_mean` across `J1A`, `J1B-Std`, `J1B-Gen`, `J2`, and `J3 Raw & Rep`, and caches all OOF
     and J2/J3 predictions to `experiments/outputs/exp_w8a_oof/`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.common import AWAY, DEAD, HOME
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec
from src.decode.hybrid_gate_v8 import (
    blend_6fam_heads_v8,
    compute_hybrid_gate_v8,
    compute_inpaint_gate_v8,
    compute_w_clean_v8,
)
from src.decode.mbr_snap import (
    ImpulseSnapConfig,
    decode_active_4dir,
    mbr_decode_sequences,
    snap_boundaries_to_impulse,
)
from src.decode.spell_reranker import SpellRerankerConfig, apply_spell_reranker
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    compute_seq481_gate,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator, count_interior_xdx
from src.eval import metrics
from src.features.symmetry import mirror_v1ctl_df, swap_match, swap_proba
from src.models.ball_inpainter import (
    InpainterConfig,
    SymBallPositionInpainter,
    SymGapDistilledExpert,
    build_inpaint_basis_and_scalars,
    build_inpainted_feature_df,
    extract_centered_kinematics,
    mirror_inpainted_df,
)
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
from exp_v6_super_champion import compute_physical_gates
from exp_w7a_player_set_net import frame_acc, load_cached_physv2_table
from exp_w7b_super_champion_v7 import compute_local_newton_gate, count_spells


def _extract_clean_centered_ball(m_clean) -> np.ndarray:
    """Extract true clean centered 2D ball coordinates `B_clean_c (T, 2)` from clean Sportec Match."""
    L, W = float(m_clean.pitch[0]), float(m_clean.pitch[1])
    half_pitch = np.array([L / 2.0, W / 2.0], dtype=np.float64)
    B_raw = m_clean.B[:, :2].astype(np.float64)
    if m_clean.direction_normalized:
        return B_raw - half_pitch[None, :]
    sgn = np.where(np.asarray(m_clean.attack_sign) < 0, -1.0, 1.0)
    return (B_raw - half_pitch[None, :]) * sgn[:, None]


def main():
    t0 = time.time()
    print("=" * 118, flush=True)
    print("WAVE 8A (EXP-13 & EXP-14): FORENSIC HYBRID GATE & `SymBallInpainterNet` TEACHER DISTILLATION", flush=True)
    print("=" * 118, flush=True)

    out_dir = ROOT / "experiments" / "outputs" / "exp_w8a_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    sportec = load_sportec()
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)

    cfg_v3 = load_config(ROOT / "configs/degrade_v3.json")
    cfg_v6 = load_config(ROOT / "configs/degrade_v6.json")
    deg_v3 = {mid: degrade(m, cfg_v3, seed=0) for mid, m in sportec.items()}
    deg_v6 = {mid: degrade(m, cfg_v6, seed=0) for mid, m in sportec.items()}

    # Load 481-column feature tables and physical gates across all 4 Judges
    X_v3, X_v6, gates_v3, gates_v6 = {}, {}, {}, {}
    for mid, m in sportec.items():
        X_v3[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", m.half)
        X_v6[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", m.half)
        gates_v3[mid] = compute_physical_gates(X_v3[mid], m.half)
        gates_v6[mid] = compute_physical_gates(X_v6[mid], m.half)

    X_sample, gates_sample = {}, {}
    for gid in GAMES:
        X_sample[gid] = load_cached_physv2_table(f"sample_{gid}_raw", sample_matches[gid].half)
        gates_sample[gid] = compute_physical_gates(X_sample[gid], sample_matches[gid].half)

    X_atd_raw = load_cached_physv2_table("atd_0f0834", atd.half)
    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)
    gate_atd_raw = compute_physical_gates(X_atd_raw, atd.half)
    gate_atd_rep = compute_physical_gates(X_atd_rep, atd.half)

    # =========================================================================
    # STEP 1: MATHEMATICAL EQUIVARIANCE AUDIT (`0.000e+00`) OF EXP-13 & EXP-14
    # =========================================================================
    print("\n--- Step 1: Mathematical Audit of Exact `0.000e+00` H <-> A Equivariance (`EXP-13` & `EXP-14`) ---", flush=True)
    T0_id = "J03WMX"
    X0 = X_v3[T0_id].iloc[:5000].copy()
    h0 = sportec[T0_id].half[:5000].copy()
    X0_swap = mirror_v1ctl_df(X0, h0)
    wg0, wgap0 = compute_physical_gates(X0, h0)
    wg0_sw, wgap0_sw = compute_physical_gates(X0_swap, h0)

    a_hyb_o = compute_hybrid_gate_v8(X0, h0, wg0)
    a_hyb_s = compute_hybrid_gate_v8(X0_swap, h0, wg0_sw)
    wc_o = compute_w_clean_v8(X0, h0, wg0)
    wc_s = compute_w_clean_v8(X0_swap, h0, wg0_sw)
    wi_o = compute_inpaint_gate_v8(X0, h0, wg0, wgap0)
    wi_s = compute_inpaint_gate_v8(X0_swap, h0, wg0_sw, wgap0_sw)

    err_hyb = float(np.max(np.abs(a_hyb_o - a_hyb_s)))
    err_wc = float(np.max(np.abs(wc_o - wc_s)))
    err_wi = float(np.max(np.abs(wi_o - wi_s)))

    # Basis & Inpainter equivariance on swapped Match
    m0_deg = deg_v3[T0_id]
    m0_swap = swap_match(m0_deg)
    V0_o, S0_o, kin0_o = build_inpaint_basis_and_scalars(m0_deg, w_ghost=gates_v3[T0_id][0])
    V0_s, S0_s, kin0_s = build_inpaint_basis_and_scalars(m0_swap, w_ghost=gates_v3[T0_id][0])
    err_V = float(np.max(np.abs(V0_o + V0_s)))
    err_S = float(np.max(np.abs(S0_o - S0_s)))

    B_clean_0 = _extract_clean_centered_ball(sportec[T0_id])
    target_d0 = B_clean_0 - kin0_o["B_int_c"]
    mask_g0 = (~kin0_o["bvis"]) & (sportec[T0_id].half > 0)
    inp_test = SymBallPositionInpainter(InpainterConfig(epochs=5, seed=42))
    inp_test.fit([V0_o], [S0_o], [target_d0], [mask_g0])
    dxy_o, Binp_o = inp_test.predict_delta(m0_deg, w_ghost=gates_v3[T0_id][0], kin=kin0_o)
    dxy_s, Binp_s = inp_test.predict_delta(m0_swap, w_ghost=gates_v3[T0_id][0], kin=kin0_s)
    err_dxy = float(np.max(np.abs(dxy_o + dxy_s)))
    err_Binp = float(np.max(np.abs(Binp_o + Binp_s)))

    df_inp_o, ha_p, anti_c = build_inpainted_feature_df(m0_deg, Binp_o, dxy_o, gates_v3[T0_id][0], kin=kin0_o)
    df_inp_s, _, _ = build_inpainted_feature_df(m0_swap, Binp_s, dxy_s, gates_v3[T0_id][0], kin=kin0_s)
    df_inp_o_mir = mirror_inpainted_df(df_inp_o, ha_p, anti_c)
    err_df_inp = float(np.max(np.abs(df_inp_o_mir.to_numpy() - df_inp_s.to_numpy())))

    exp_test = SymGapDistilledExpert(n_estimators=40, seed=42)
    exp_test.fit([df_inp_o.iloc[:8000]], [sportec[T0_id].labels[:8000]], [sportec[T0_id].half[:8000]], ha_p, anti_c)
    p_exp_o, c_exp_o = exp_test.predict_tta(df_inp_o.iloc[:5000])
    p_exp_s, c_exp_s = exp_test.predict_tta(df_inp_s.iloc[:5000])
    err_exp_p = float(np.max(np.abs(swap_proba(p_exp_o) - p_exp_s)))
    err_exp_c = float(np.max(np.abs(c_exp_o - c_exp_s)))

    print(f"  [Equivariance Audit on {T0_id}]", flush=True)
    print(f"    compute_hybrid_gate_v8   max|g(M) - g(swap(M))|           = {err_hyb:.3e}", flush=True)
    print(f"    compute_w_clean_v8       max|wc(M) - wc(swap(M))|         = {err_wc:.3e}", flush=True)
    print(f"    compute_inpaint_gate_v8  max|wi(M) - wi(swap(M))|         = {err_wi:.3e}", flush=True)
    print(f"    Inpaint Basis V_basis    max|V(M) + V(swap(M))|           = {err_V:.3e}", flush=True)
    print(f"    Inpaint Scalars S_inv    max|S(M) - S(swap(M))|           = {err_S:.3e}", flush=True)
    print(f"    SymBallPositionInpainter max|dx(M) + dx(swap(M))|         = {err_dxy:.3e} (B_inp={err_Binp:.3e})", flush=True)
    print(f"    build_inpainted_feat_df  max|mirror(df(M)) - df(swap(M))| = {err_df_inp:.3e}", flush=True)
    print(f"    SymGapDistilledExpert    max|swap(P(M)) - P(swap(M))|     = {err_exp_p:.3e} (change={err_exp_c:.3e})", flush=True)
    assert max(err_hyb, err_wc, err_wi, err_V, err_S, err_dxy, err_Binp, err_df_inp, err_exp_p, err_exp_c) < 1e-6

    # =========================================================================
    # STEP 2: EXP-13 FORENSIC DIAGNOSIS ACROSS ALL 4 JUDGES
    # =========================================================================
    print("\n--- Step 2 (EXP-13): Forensic Diagnosis of `alpha_p2` Gates Across All 4 Judges ---", flush=True)
    for jname, x_dict, h_dict, g_dict in [
        ("Judge 1A (Sportec v3)", X_v3, {m: sportec[m].half for m in MATCH_IDS}, gates_v3),
        ("Judge 1B (Sportec v6)", X_v6, {m: sportec[m].half for m in MATCH_IDS}, gates_v6),
        ("Judge 2  (Sample 3G) ", X_sample, {g: sample_matches[g].half for g in GAMES}, gates_sample),
        ("Judge 3  (BAR-ATH Raw)", {"raw": X_atd_raw}, {"raw": atd.half}, {"raw": gate_atd_raw}),
        ("Judge 3  (BAR-ATH Rep)", {"rep": X_atd_rep}, {"rep": atd.half}, {"rep": gate_atd_rep}),
    ]:
        a_glob = [float(compute_seq481_gate(x_dict[k], h_dict[k], g_dict[k][0]).mean()) for k in x_dict]
        a_loc = [float(compute_local_newton_gate(x_dict[k], h_dict[k], g_dict[k][0]).mean()) for k in x_dict]
        a_hyb = [float(compute_hybrid_gate_v8(x_dict[k], h_dict[k], g_dict[k][0]).mean()) for k in x_dict]
        a_hyb_p90 = [float(np.percentile(compute_hybrid_gate_v8(x_dict[k], h_dict[k], g_dict[k][0]), 90)) for k in x_dict]
        a_loc_p90 = [float(np.percentile(compute_local_newton_gate(x_dict[k], h_dict[k], g_dict[k][0]), 90)) for k in x_dict]
        print(
            f"  {jname:24s} | global={np.mean(a_glob):.4f} | "
            f"local_newton={np.mean(a_loc):.4f} (p90={np.mean(a_loc_p90):.3f}) | "
            f"hybrid_v8={np.mean(a_hyb):.4f} (p90={np.mean(a_hyb_p90):.3f})",
            flush=True,
        )

    # =========================================================================
    # STEP 3: EXP-14 PART A (`SymBallPositionInpainter` 7-FOLD LOMO) & PART B (`SymGapDistilledExpert`)
    # =========================================================================
    print("\n--- Step 3 (EXP-14): 7-Fold LOMO `SymBallPositionInpainter` & `SymGapDistilledExpert` Distillation ---", flush=True)
    j23_inp_pkl = out_dir / "w8a_j23_preds.pkl"
    all_cached = j23_inp_pkl.exists() and all((out_dir / f"w8a_inpaint_{T}.npz").exists() for T in MATCH_IDS)

    B_clean = {mid: _extract_clean_centered_ball(sportec[mid]) for mid in MATCH_IDS}
    kin_v3, V_v3, S_v3, tdelta_v3, mgap_v3 = {}, {}, {}, {}, {}
    kin_v6, V_v6, S_v6, tdelta_v6, mgap_v6 = {}, {}, {}, {}, {}

    for mid in MATCH_IDS:
        V_v3[mid], S_v3[mid], kin_v3[mid] = build_inpaint_basis_and_scalars(deg_v3[mid], w_ghost=gates_v3[mid][0])
        tdelta_v3[mid] = B_clean[mid] - kin_v3[mid]["B_int_c"]
        mgap_v3[mid] = (~kin_v3[mid]["bvis"]) & (sportec[mid].half > 0)

        V_v6[mid], S_v6[mid], kin_v6[mid] = build_inpaint_basis_and_scalars(deg_v6[mid], w_ghost=gates_v6[mid][0])
        tdelta_v6[mid] = B_clean[mid] - kin_v6[mid]["B_int_c"]
        mgap_v6[mid] = ((~kin_v6[mid]["bvis"]) | (gates_v6[mid][0][:, 0] > 0.15)) & (sportec[mid].half > 0)

    inp_oof = {}
    if all_cached:
        print("  Loading cached 7-fold LOMO `w8a_inpaint_{T}.npz` and `w8a_j23_preds.pkl`...", flush=True)
        for T in MATCH_IDS:
            z = np.load(out_dir / f"w8a_inpaint_{T}.npz")
            inp_oof[T] = {k: z[k] for k in z.files}
        with open(j23_inp_pkl, "rb") as f:
            j23_inp = pickle.load(f)
    else:
        # Build Clean-Ball Teacher features once per match
        df_clean, ha_p, anti_c = {}, [], []
        for mid in MATCH_IDS:
            kin_cl = extract_centered_kinematics(sportec[mid])
            df_clean[mid], ha_p, anti_c = build_inpainted_feature_df(
                sportec[mid], B_clean[mid], np.zeros_like(B_clean[mid]),
                np.zeros_like(gates_v3[mid][0]), kin=kin_cl,
            )

        # 7-Fold LOMO for Part A (`SymBallPositionInpainter`)
        B_inp_v3, dxy_v3 = {}, {}
        B_inp_v6, dxy_v6 = {}, {}
        for T in MATCH_IDS:
            tr_ids = [m for m in MATCH_IDS if m != T]
            inp_model = SymBallPositionInpainter(InpainterConfig(epochs=35, seed=42))
            V_tr = [V_v3[m] for m in tr_ids] + [V_v6[m] for m in tr_ids]
            S_tr = [S_v3[m] for m in tr_ids] + [S_v6[m] for m in tr_ids]
            Y_tr = [tdelta_v3[m] for m in tr_ids] + [tdelta_v6[m] for m in tr_ids]
            M_tr = [mgap_v3[m] for m in tr_ids] + [mgap_v6[m] for m in tr_ids]
            inp_model.fit(V_tr, S_tr, Y_tr, M_tr)
            dxy_v3[T], B_inp_v3[T] = inp_model.predict_delta(deg_v3[T], w_ghost=gates_v3[T][0], kin=kin_v3[T])
            dxy_v6[T], B_inp_v6[T] = inp_model.predict_delta(deg_v6[T], w_ghost=gates_v6[T][0], kin=kin_v6[T])

        # Build 78-col inpainted feature tables for all 7 matches on v3 and v6
        df_inp_v3, df_inp_v6 = {}, {}
        for mid in MATCH_IDS:
            df_inp_v3[mid], _, _ = build_inpainted_feature_df(
                deg_v3[mid], B_inp_v3[mid], dxy_v3[mid], gates_v3[mid][0], kin=kin_v3[mid],
            )
            df_inp_v6[mid], _, _ = build_inpainted_feature_df(
                deg_v6[mid], B_inp_v6[mid], dxy_v6[mid], gates_v6[mid][0], kin=kin_v6[mid],
            )

        # 7-Fold LOMO Clean-Ball Teacher + Distilled Expert (`SymGapDistilledExpert`)
        teacher_oof = {}
        for T in MATCH_IDS:
            tr_ids = [m for m in MATCH_IDS if m != T]
            teacher = SymGapDistilledExpert(n_estimators=260, seed=42)
            teacher.fit(
                [df_clean[m] for m in tr_ids],
                [sportec[m].labels for m in tr_ids],
                [sportec[m].half for m in tr_ids],
                ha_p, anti_c, distill_alpha=0.0, gap_oversample_weight=1.0,
            )
            teacher_oof[T], _ = teacher.predict_tta(df_clean[T])

        for T in MATCH_IDS:
            tf_fold = time.time()
            tr_ids = [m for m in MATCH_IDS if m != T]
            expert = SymGapDistilledExpert(n_estimators=300, seed=42)
            df_tr = [df_inp_v3[m] for m in tr_ids] + [df_inp_v6[m] for m in tr_ids]
            y_tr = [sportec[m].labels for m in tr_ids] + [sportec[m].labels for m in tr_ids]
            h_tr = [sportec[m].half for m in tr_ids] + [sportec[m].half for m in tr_ids]
            t_tr = [teacher_oof[m] for m in tr_ids] + [teacher_oof[m] for m in tr_ids]
            g_tr = [mgap_v3[m] for m in tr_ids] + [mgap_v6[m] for m in tr_ids]
            expert.fit(df_tr, y_tr, h_tr, ha_p, anti_c, teacher_proba_list=t_tr, gap_mask_list=g_tr, distill_alpha=0.40)

            p_i3, c_i3 = expert.predict_tta(df_inp_v3[T])
            p_i6, c_i6 = expert.predict_tta(df_inp_v6[T])
            w_i3 = compute_inpaint_gate_v8(X_v3[T], sportec[T].half, gates_v3[T][0], gates_v3[T][1])
            w_i6 = compute_inpaint_gate_v8(X_v6[T], sportec[T].half, gates_v6[T][0], gates_v6[T][1])

            inp_oof[T] = {
                "delta_v3": dxy_v3[T].astype(np.float32),
                "B_inp_v3": B_inp_v3[T].astype(np.float32),
                "p_inp_v3": p_i3,
                "c_inp_v3": c_i3,
                "w_inp_v3": w_i3,
                "delta_v6": dxy_v6[T].astype(np.float32),
                "B_inp_v6": B_inp_v6[T].astype(np.float32),
                "p_inp_v6": p_i6,
                "c_inp_v6": c_i6,
                "w_inp_v6": w_i6,
            }
            np.savez_compressed(out_dir / f"w8a_inpaint_{T}.npz", **inp_oof[T])
            print(f"    Fold {T} finished in {time.time() - tf_fold:.1f}s", flush=True)

        # Train Full 7-Match Inpainter + Distilled Expert for Judge 2 (`Metrica Sample`) & Judge 3 (`BAR-ATH`)
        inp_full = SymBallPositionInpainter(InpainterConfig(epochs=35, seed=42))
        inp_full.fit(
            [V_v3[m] for m in MATCH_IDS] + [V_v6[m] for m in MATCH_IDS],
            [S_v3[m] for m in MATCH_IDS] + [S_v6[m] for m in MATCH_IDS],
            [tdelta_v3[m] for m in MATCH_IDS] + [tdelta_v6[m] for m in MATCH_IDS],
            [mgap_v3[m] for m in MATCH_IDS] + [mgap_v6[m] for m in MATCH_IDS],
        )
        exp_full = SymGapDistilledExpert(n_estimators=300, seed=42)
        exp_full.fit(
            [df_inp_v3[m] for m in MATCH_IDS] + [df_inp_v6[m] for m in MATCH_IDS],
            [sportec[m].labels for m in MATCH_IDS] + [sportec[m].labels for m in MATCH_IDS],
            [sportec[m].half for m in MATCH_IDS] + [sportec[m].half for m in MATCH_IDS],
            ha_p, anti_c,
            teacher_proba_list=[teacher_oof[m] for m in MATCH_IDS] + [teacher_oof[m] for m in MATCH_IDS],
            gap_mask_list=[mgap_v3[m] for m in MATCH_IDS] + [mgap_v6[m] for m in MATCH_IDS],
            distill_alpha=0.40,
        )

        sample_inp = {}
        for gid in GAMES:
            ms = sample_matches[gid]
            dxy_s, Binp_s = inp_full.predict_delta(ms, w_ghost=gates_sample[gid][0])
            df_s, _, _ = build_inpainted_feature_df(ms, Binp_s, dxy_s, gates_sample[gid][0])
            ps_i, cs_i = exp_full.predict_tta(df_s)
            ws_i = compute_inpaint_gate_v8(X_sample[gid], ms.half, gates_sample[gid][0], gates_sample[gid][1])
            sample_inp[gid] = {"p_inp": ps_i, "c_inp": cs_i, "w_inp": ws_i, "delta": dxy_s.astype(np.float32)}

        dxy_ar, Binp_ar = inp_full.predict_delta(atd, w_ghost=gate_atd_raw[0])
        df_ar, _, _ = build_inpainted_feature_df(atd, Binp_ar, dxy_ar, gate_atd_raw[0])
        par_i, car_i = exp_full.predict_tta(df_ar)
        war_i = compute_inpaint_gate_v8(X_atd_raw, atd.half, gate_atd_raw[0], gate_atd_raw[1])

        dxy_ap, Binp_ap = inp_full.predict_delta(atd_rep, w_ghost=gate_atd_rep[0])
        df_ap, _, _ = build_inpainted_feature_df(atd_rep, Binp_ap, dxy_ap, gate_atd_rep[0])
        pap_i, cap_i = exp_full.predict_tta(df_ap)
        wap_i = compute_inpaint_gate_v8(X_atd_rep, atd.half, gate_atd_rep[0], gate_atd_rep[1])

        j23_inp = {
            "sample": sample_inp,
            "atd_raw": {"p_inp": par_i, "c_inp": car_i, "w_inp": war_i, "delta": dxy_ar.astype(np.float32)},
            "atd_rep": {"p_inp": pap_i, "c_inp": cap_i, "w_inp": wap_i, "delta": dxy_ap.astype(np.float32)},
        }
        with open(j23_inp_pkl, "wb") as f:
            pickle.dump(j23_inp, f)

    # Report Spatial Inpainting Accuracy (MAE & RMSE in meters) across all 7 Sportec folds
    for tag_deg, kin_d, b_key in [("Sportec v3 (`degrade_v3`)", kin_v3, "B_inp_v3"), ("Sportec v6 (`degrade_v6`)", kin_v6, "B_inp_v6")]:
        err_int_all, err_inp_all = [], []
        err_int_act, err_inp_act = [], []
        err_int_dead, err_inp_dead = [], []
        for T in MATCH_IDS:
            b_true = B_clean[T]
            b_int = kin_d[T]["B_int_c"]
            b_inp = inp_oof[T][b_key].astype(np.float64)
            valid_b = np.isfinite(b_true[:, 0]) & np.isfinite(b_true[:, 1]) & (sportec[T].half > 0)
            m_gap = (~kin_d[T]["bvis"]) & valid_b
            m_act = m_gap & (sportec[T].labels != DEAD) & (sportec[T].labels >= 0)
            m_dead = m_gap & (sportec[T].labels == DEAD)

            d_int = np.hypot(b_int[:, 0] - b_true[:, 0], b_int[:, 1] - b_true[:, 1])
            d_inp = np.hypot(b_inp[:, 0] - b_true[:, 0], b_inp[:, 1] - b_true[:, 1])
            err_int_all.append(d_int[m_gap])
            err_inp_all.append(d_inp[m_gap])
            err_int_act.append(d_int[m_act])
            err_inp_act.append(d_inp[m_act])
            err_int_dead.append(d_int[m_dead])
            err_inp_dead.append(d_inp[m_dead])

        print(f"\n  [Spatial Ball Reconstruction 7-Fold LOMO — {tag_deg}]", flush=True)
        for rname, e0_l, e1_l in [
            ("All Missing-Ball Gaps (`~bvis`)", err_int_all, err_inp_all),
            ("Active In-Play Gaps (`labels != D`)", err_int_act, err_inp_act),
            ("Dead-Ball Gaps (`labels == D`)", err_int_dead, err_inp_dead),
        ]:
            e0 = np.concatenate(e0_l)
            e1 = np.concatenate(e1_l)
            mae0, rmse0 = float(np.mean(e0)), float(np.sqrt(np.mean(e0 ** 2)))
            mae1, rmse1 = float(np.mean(e1)), float(np.sqrt(np.mean(e1 ** 2)))
            print(
                f"    {rname:38s} (n={len(e0):6d}) | "
                f"Linear Interp: MAE={mae0:6.3f}m, RMSE={rmse0:6.3f}m -> "
                f"SymBallInpainter: MAE={mae1:6.3f}m ({mae1 - mae0:+6.3f}m), RMSE={rmse1:6.3f}m ({rmse1 - rmse0:+6.3f}m)",
                flush=True,
            )

    # =========================================================================
    # STEP 4: MULTI-JUDGE EVALUATION OF `EXP-13` (`compute_hybrid_gate_v8`) & `EXP-14` (`SymGapDistilledExpert`)
    # =========================================================================
    print("\n--- Step 4: Multi-Judge Evaluation (`J1A`, `J1B-Std`, `J1B-Gen`, `J2`, `J3 Raw & Rep`) ---", flush=True)
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    plus_dir = ROOT / "experiments" / "outputs" / "exp_c_plus_oof"
    g_dir = ROOT / "experiments" / "outputs" / "exp_g_oof"
    h_dir = ROOT / "experiments" / "outputs" / "exp_h_oof"
    w3_dir = ROOT / "experiments" / "outputs" / "exp_w3_oof"
    w4_dir = ROOT / "experiments" / "outputs" / "exp_w4_oof"
    w5_dir = ROOT / "experiments" / "outputs" / "exp_w5_tree_oof"
    w5b_dir = ROOT / "experiments" / "outputs" / "exp_w5b_oof"
    w6_dir = ROOT / "experiments" / "outputs" / "exp_w6_oof"
    w7a_dir = ROOT / "experiments" / "outputs" / "exp_w7a_oof"
    w7b_dir = ROOT / "experiments" / "outputs" / "exp_w7b_oof"

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std = pickle.load(f)[0]
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)
    dec_v4 = DecoderParams(**{
        k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items()
        if not k.startswith("_")
    })

    oof = {}
    for T in MATCH_IDS:
        zb = np.load(sym_dir / f"sym_{T}.npz")
        zt = np.load(sym_dir / f"tcn_sym_{T}.npz")
        zp = np.load(plus_dir / f"plus_{T}.npz")
        zg = np.load(g_dir / f"ghost_sym_{T}.npz")
        zh = np.load(h_dir / f"bigru_sym_{T}.npz")
        zw3 = np.load(w3_dir / f"b4d_{T}.npz")
        zw4 = np.load(w4_dir / f"physv2_{T}.npz")
        zw5 = np.load(w5_dir / f"cb_xgb_{T}.npz")
        zw5b_tcn = np.load(w5b_dir / f"tcn_physv2_{T}.npz")
        zw5b_gru = np.load(w5b_dir / f"bigru_physv2_{T}.npz")
        zw7a = np.load(w7a_dir / f"pset_{T}.npz")
        zw7b_v6 = np.load(w7b_dir / f"v6_genuine_{T}.npz")
        oof[T] = {
            "g_v1": zb["symgbm_tta"], "tr_v1": zb["symtrans_tta"],
            "g_pl": zp["symgbm_plus_tta"], "tr_pl": zp["symtrans_plus_tta"],
            "g_pl_on_v6": zw7b_v6["symgbm_plus_tta_on_v6"], "tr_pl_on_v6": zw7b_v6["symtrans_plus_tta_on_v6"],
            "g_v6_on_v3": zg["g_v6_on_v3"], "tr_v6_on_v3": zg["tr_v6_on_v3"],
            "g_v3_on_v6": zg["g_v3_on_v6"], "tr_v3_on_v6": zg["tr_v3_on_v6"],
            "g_v6_on_v6": zg["g_v6_on_v6"], "tr_v6_on_v6": zg["tr_v6_on_v6"],
            "tcn": zt["symtcn_tta_proba"], "tcnc": zt["symtcn_tta_change"],
            "gru": zh["symbigru_tta_proba"], "gruc": zh["symbigru_tta_change"],
            "p4d_v3": zw3["p4d_v3"], "p4d_v6": zw3["p4d_v6"],
            "g_p2_v3": zw4["symgbm_physv2_tta"], "tr_p2_v3": zw4["symtrans_physv2_tta"],
            "g_p2_v6": zw4["symgbm_physv2_tta_on_v6"], "tr_p2_v6": zw4["symtrans_physv2_tta_on_v6"],
            "cb_p2_v3": zw5["symcatboost_physv2_tta"], "xgb_p2_v3": zw5["symxgb_physv2_tta"],
            "cb_p2_v6": zw5["symcatboost_physv2_tta_on_v6"], "xgb_p2_v6": zw5["symxgb_physv2_tta_on_v6"],
            "tcn_p2_v3": zw5b_tcn["v3_proba"], "tcnc_p2_v3": zw5b_tcn["v3_change"],
            "tcn_p2_v6": zw5b_tcn["v6_proba"], "tcnc_p2_v6": zw5b_tcn["v6_change"],
            "gru_p2_v3": zw5b_gru["v3_proba"], "gruc_p2_v3": zw5b_gru["v3_change"],
            "gru_p2_v6": zw5b_gru["v6_proba"], "gruc_p2_v6": zw5b_gru["v6_change"],
            "pset_v3": zw7a["v3_proba"], "psetc_v3": zw7a["v3_change"],
            "pset_v6": zw7a["v6_proba"], "psetc_v6": zw7a["v6_change"],
        }

    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        pe_plus = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f:
        sample_v6, atd_rep_v6 = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(w3_dir / "b4d_j23_preds.pkl", "rb") as f:
        b4d_j23 = pickle.load(f)
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f:
        pe_p2 = pickle.load(f)
    with open(w5_dir / "eval_preds_cb_xgb.pkl", "rb") as f:
        pe_w5 = pickle.load(f)
    with open(w5b_dir / "seq_physv2_j23_preds.pkl", "rb") as f:
        pe_w5b = pickle.load(f)
    with open(w6_dir / "atd_raw_v6_bigru.pkl", "rb") as f:
        atd_raw_extra = pickle.load(f)
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f:
        pset_j23 = pickle.load(f)
    with open(w6_dir / "w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)

    best_calib_cfg: TransductiveCalibConfig = w6_sum["best_calib_cfg"]
    lomo_rerank_cfgs = w6_sum["lomo_rerank_cfgs"]
    full_rerank_cfg = w6_sum["full_rerank_cfg"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)

    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )

    def build_w8a_streams(
        s3_d: float = 0.05,
        s3_ha: float = 0.30,
        s3_c: float = 0.18,
        ps_ha: float = 0.08,
        gate_mode: str = "hybrid_v8",
        adaptive_regime: bool = True,
        inp_scale: float = 0.0,
    ) -> dict:
        streams = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}, "atd_raw": None, "atd_rep": None}

        def _get_alpha(X_df, h_arr, wg_arr):
            if gate_mode == "hybrid_v8":
                return compute_hybrid_gate_v8(X_df, h_arr, wg_arr)
            if gate_mode == "local_newton":
                return compute_local_newton_gate(X_df, h_arr, wg_arr)
            return compute_seq481_gate(X_df, h_arr, wg_arr)

        for T in MATCH_IDS:
            d = oof[T]
            di = inp_oof[T]
            h = sportec[T].half

            a_v3 = _get_alpha(X_v3[T], h, gates_v3[T][0])
            wc_v3 = compute_w_clean_v8(X_v3[T], h, gates_v3[T][0])
            p3, c3 = synthesize_head_decoupled_proba(
                d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"],
                d["tcn"], d["gru"], d["tr_pl"], d["tr_p2_v3"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v3"], d["tr_v6_on_v3"],
                w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v3"], bigru_physv2_proba=d["gru_p2_v3"],
                tcn_physv2_change=d["tcnc_p2_v3"], bigru_physv2_change=d["gruc_p2_v3"],
                alpha_p2=a_v3,
            )
            p3, c3 = blend_6fam_heads_v8(
                p3, c3, d["tcn"], d["gru"], d["gru_p2_v3"], d["pset_v3"],
                d["tcnc"], d["gruc"], d["gruc_p2_v3"], d["psetc_v3"],
                a_v3, wc_v3, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=di["p_inp_v3"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v3"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v3"]) if inp_scale > 0 else None,
            )
            p3 = calibrate_match_transductive(p3, h, X_v3[T], best_calib_cfg)
            f3 = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"], d["pset_v3"]]
            streams["v3"][T] = (p3, c3, d["p4d_v3"], f3)

            a_v6 = _get_alpha(X_v6[T], h, gates_v6[T][0])
            wc_v6 = compute_w_clean_v8(X_v6[T], h, gates_v6[T][0])
            p6s, c6s = synthesize_head_decoupled_proba(
                d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn"], d["gru"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=a_v6,
            )
            p6s, c6s = blend_6fam_heads_v8(
                p6s, c6s, d["tcn"], d["gru"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc"], d["gruc"], d["gruc_p2_v6"], d["psetc_v6"],
                a_v6, wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=di["p_inp_v6"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v6"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v6"]) if inp_scale > 0 else None,
            )
            p6s = calibrate_match_transductive(p6s, h, X_v6[T], best_calib_cfg)
            f6s = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tcn_p2_v6"], d["gru_p2_v6"], d["pset_v6"]]
            streams["v6_std"][T] = (p6s, c6s, d["p4d_v6"], f6s)

            p6g, c6g = synthesize_head_decoupled_proba(
                d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn_p2_v6"], d["gru_p2_v6"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc_p2_v6"], d["gruc_p2_v6"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=np.ones(len(h), dtype=np.float32),
            )
            p6g, c6g = blend_6fam_heads_v8(
                p6g, c6g, d["tcn_p2_v6"], d["gru_p2_v6"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc_p2_v6"], d["gruc_p2_v6"], d["gruc_p2_v6"], d["psetc_v6"],
                np.ones(len(h), dtype=np.float32), wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=False,
                p_inp=di["p_inp_v6"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v6"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v6"]) if inp_scale > 0 else None,
            )
            p6g = calibrate_match_transductive(p6g, h, X_v6[T], best_calib_cfg)
            f6g = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn_p2_v6"], d["gru_p2_v6"], d["g_v6_on_v6"], d["pset_v6"]]
            streams["v6_gen"][T] = (p6g, c6g, d["p4d_v6"], f6g)

        for gid in GAMES:
            ms = sample_matches[gid]
            dh = sample_h[gid]
            dpl = pe_plus["sample"][gid]
            dv6 = sample_v6[gid]
            dp2 = pe_p2["sample"][gid]
            dw5 = pe_w5["sample"][gid]
            dw5b = pe_w5b["sample"][gid]
            dpset = pset_j23["sample"][gid]
            dinp = j23_inp["sample"][gid]
            a_s = _get_alpha(X_sample[gid], ms.half, gates_sample[gid][0])
            wc_s = compute_w_clean_v8(X_sample[gid], ms.half, gates_sample[gid][0])

            ps, cs = synthesize_head_decoupled_proba(
                dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
                dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
                ms.half, dv6["g_v6"], dv6["tr_v6"],
                w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main,
                tcn_physv2_proba=dw5b["tcn_physv2_sym"], bigru_physv2_proba=dw5b["bigru_physv2_sym"],
                tcn_physv2_change=dw5b["tcnc_physv2_sym"], bigru_physv2_change=dw5b["bigruc_physv2_sym"],
                alpha_p2=a_s,
            )
            ps, cs = blend_6fam_heads_v8(
                ps, cs, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"],
                dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"],
                a_s, wc_s, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=dinp["p_inp"] if inp_scale > 0 else None,
                c_inp=dinp["c_inp"] if inp_scale > 0 else None,
                w_inp=(inp_scale * dinp["w_inp"]) if inp_scale > 0 else None,
            )
            ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], best_calib_cfg)
            fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"]]
            streams["sample"][gid] = (ps, cs, b4d_j23["sample"][gid], fs)

        # Judge 3 Raw & Repaired
        for vkey, X_df, gate_t, t2_p, g2_p, t2_c, g2_c, gv6_p, gv6_c in [
            ("atd_raw", X_atd_raw, gate_atd_raw, pe_plus["atd_raw"]["t_sym"], atd_raw_extra["b_sym"], pe_plus["atd_raw"]["tc_sym"], atd_raw_extra["bc_sym"], atd_raw_extra["g_v6"], atd_raw_extra["tr_v6"]),
            ("atd_rep", X_atd_rep, gate_atd_rep, atd_rep_h["t_sym"], atd_rep_h["b_sym"], atd_rep_h["tc_sym"], atd_rep_h["bc_sym"], atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"]),
        ]:
            dpl_a = pe_plus[vkey]
            dp2_a = pe_p2[vkey]
            dw5_a = pe_w5[vkey]
            dw5b_a = pe_w5b[vkey]
            dpset_a = pset_j23[vkey]
            dinp_a = j23_inp[vkey]
            a_a = _get_alpha(X_df, atd.half, gate_t[0])
            wc_a = compute_w_clean_v8(X_df, atd.half, gate_t[0])
            pa, ca = synthesize_head_decoupled_proba(
                dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"],
                t2_p, g2_p, dpl_a["tr_plus_sym"], dp2_a["tr_physv2_sym"], t2_c, g2_c,
                atd.half, gv6_p, gv6_c,
                w_ghost=gate_t[0], w_gap=gate_t[1], cfg=scfg_main,
                tcn_physv2_proba=dw5b_a["tcn_physv2_sym"], bigru_physv2_proba=dw5b_a["bigru_physv2_sym"],
                tcn_physv2_change=dw5b_a["tcnc_physv2_sym"], bigru_physv2_change=dw5b_a["bigruc_physv2_sym"],
                alpha_p2=a_a,
            )
            pa, ca = blend_6fam_heads_v8(
                pa, ca, t2_p, g2_p, dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"],
                t2_c, g2_c, dw5b_a["bigruc_physv2_sym"], dpset_a["pset_change"],
                a_a, wc_a, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=dinp_a["p_inp"] if inp_scale > 0 else None,
                c_inp=dinp_a["c_inp"] if inp_scale > 0 else None,
                w_inp=(inp_scale * dinp_a["w_inp"]) if inp_scale > 0 else None,
            )
            pa = calibrate_match_transductive(pa, half_atd_dec, X_df, best_calib_cfg)
            fa = [dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"], t2_p, g2_p, dw5b_a["tcn_physv2_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"]]
            streams[vkey] = (pa, ca, b4d_j23[vkey], fa)
        return streams

    st_v7_bal = build_w8a_streams(0.00, 0.25, 0.18, 0.10, gate_mode="seq481_global", adaptive_regime=False, inp_scale=0.0)
    st_v7_newt = build_w8a_streams(0.05, 0.30, 0.18, 0.08, gate_mode="local_newton", adaptive_regime=False, inp_scale=0.0)
    st_v8_hyb = build_w8a_streams(0.05, 0.30, 0.18, 0.08, gate_mode="hybrid_v8", adaptive_regime=True, inp_scale=0.0)
    st_v8_inp04 = build_w8a_streams(0.05, 0.30, 0.18, 0.08, gate_mode="hybrid_v8", adaptive_regime=True, inp_scale=0.35)
    st_v8_inp08 = build_w8a_streams(0.05, 0.30, 0.18, 0.08, gate_mode="hybrid_v8", adaptive_regime=True, inp_scale=0.65)

    def eval_stream(label: str, st: dict, mode: str = "rerank") -> dict:
        v3_f1, v6s_f1, v6g_f1 = [], [], []
        for T in MATCH_IDS:
            h = sportec[T].half
            y = sportec[T].labels
            for key_s, X_df, f1_list in [("v3", X_v3[T], v3_f1), ("v6_std", X_v6[T], v6s_f1), ("v6_gen", X_v6[T], v6g_f1)]:
                p, c, p4d, fams = st[key_s][T]
                pr = apply_xdx_arbitrator(decode_v4(p, h, loo_dec_hyb[T], p_change=c), p, h, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1)
                if mode == "rerank":
                    pr = apply_spell_reranker(pr, p, h, X_df, c, p4d, fams, cfg=lomo_rerank_cfgs[T])
                f1_list.append(float(metrics.score(y, pr, h)["spell_f1_mean"]))

        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            p, c, p4d, fams = st["sample"][gid]
            pr = apply_xdx_arbitrator(decode_v4(p, ms.half, dec_v5_hyb, p_change=c), p, ms.half, X_sample[gid], p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1)
            if mode == "rerank":
                pr = apply_spell_reranker(pr, p, ms.half, X_sample[gid], c, p4d, fams, cfg=full_rerank_cfg)
            s_raw.append(float(metrics.score(ms.labels, pr, ms.half)["spell_f1_mean"]))
            s_nod.append(float(metrics.score(without_dead_margin(ms.labels, ms.half), pr, ms.half)["spell_f1_mean"]))

        def _dec_a(vkey, X_df):
            p, c, p4d, fams = st[vkey]
            pr = apply_xdx_arbitrator(decode_v4(p, half_atd_dec, dec_v5_hyb, p_change=c), p, half_atd_dec, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1)
            if mode == "rerank":
                pr = apply_spell_reranker(pr, p, half_atd_dec, X_df, c, p4d, fams, cfg=full_rerank_cfg)
            pr[second_start:kick2] = DEAD
            pr[~atd.graded] = DEAD
            return pr

        pr_raw = _dec_a("atd_raw", X_atd_raw)
        pr_rep = _dec_a("atd_rep", X_atd_rep)
        sc_raw = metrics.score_public(pr_raw, bootstrap_n=0)
        sc_rep = metrics.score_public(pr_rep, bootstrap_n=0)
        pl_rep = plausibility(pr_rep, atd.half, atd.graded)
        res = {
            "label": label,
            "j1a": float(np.mean(v3_f1)),
            "j1b_std": float(np.mean(v6s_f1)),
            "j1b_gen": float(np.mean(v6g_f1)),
            "j2_raw": float(np.mean(s_raw)),
            "j2_nod": float(np.mean(s_nod)),
            "j3_raw": float(sc_raw["spell_f1_mean"]),
            "j3_rep": float(sc_rep["spell_f1_mean"]),
            "j3_df1": float(sc_rep["alive_dead_f1"]),
            "j3_pm": float(pl_rep["poss_per_min"]),
            "j3_pctd": float(pl_rep["pct_D"]),
        }
        print(
            f"  {label:60s} | J1A={res['j1a']:.4f} | J1B={res['j1b_std']:.4f}/{res['j1b_gen']:.4f} | "
            f"J2={res['j2_raw']:.4f} ({res['j2_nod']:.4f}) | J3 Raw={res['j3_raw']:.4f} | "
            f"J3 Rep={res['j3_rep']:.4f} (D={res['j3_df1']:.3f}, p/m={res['j3_pm']:.2f})",
            flush=True,
        )
        return res

    rows = [
        eval_stream("1. v7_bal (global gate, 0.00/0.25/0.10) + XDX", st_v7_bal, "xdx"),
        eval_stream("2. v7_newt (local_newton gate, 0.05/0.30/0.08) + XDX", st_v7_newt, "xdx"),
        eval_stream("3. v8_hybrid (EXP-13 Hybrid Gate + Adaptive Regime) + XDX", st_v8_hyb, "xdx"),
        eval_stream("4. v7_bal (global gate, 0.00/0.25/0.10) + Rerank", st_v7_bal, "rerank"),
        eval_stream("5. v7_newt (local_newton gate, 0.05/0.30/0.08) + Rerank", st_v7_newt, "rerank"),
        eval_stream("6. v8_hybrid (EXP-13 Hybrid Gate + Adaptive Regime) + Rerank", st_v8_hyb, "rerank"),
        eval_stream("7. v8_hybrid + EXP-14 SymGapDistilledExpert (scale=0.35) + Rerank", st_v8_inp04, "rerank"),
        eval_stream("8. v8_hybrid + EXP-14 SymGapDistilledExpert (scale=0.65) + Rerank", st_v8_inp08, "rerank"),
    ]

    with open(out_dir / "w8a_results_summary.pkl", "wb") as f:
        pickle.dump(rows, f)

    print(f"\nCompleted Wave 8A (`EXP-13` & `EXP-14`) in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
