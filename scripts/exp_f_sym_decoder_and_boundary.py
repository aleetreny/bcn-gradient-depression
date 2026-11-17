"""EXP-F: Out-of-the-Box Spell-Level Decoding for SymGBM_TTA + SymTCN_TTA + SymTrans_TTA.

Tests 3 orthogonal decoding breakthroughs on top of our symmetric predictions:
1. Official `s4_3` Plateau-Center Decoder (`dec_v5_sym`) tuned on the Symmetric OOFs (with LOO validation on Judge 1).
2. Log-Odds Total-Variation Boundary Impulse (`g_ha` and `g_d`): augments `p_change` with the local derivative of smoothed emission log-odds.
3. Scale-Free Short-Spell Rescuer (`rescue_short_spells`): resolves the fundamental mismatch between frame-additive Viterbi and spell-level `spellF1`, where missing a single 1-3s interception merges two surrounding spells and incurs a triple penalty in `spellF1`.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match  # noqa: E402
from games import GAMES as GAME_INFO, anchor_video_frame  # noqa: E402
from s2_3_frame_models import plausibility  # noqa: E402
from s3_1_transition_head import mean_score, plateau, suggest  # noqa: E402
from s3_7_nested_full import build_features  # noqa: E402
from s3_11_tcn_local import without_dead_margin  # noqa: E402
from s4_6_broadcast_align import load_broadcast  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from exp_b_symmetry_eval import blend_proba, predict_gbm_tta, predict_tcn_tta, predict_trans_tta  # noqa: E402
from src.data.common import AWAY, DEAD, HOME  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4  # noqa: E402
from src.eval import metrics  # noqa: E402


def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(x.astype(np.float64), k, mode="same")


def augment_p_change_with_logodds_tv(
    proba: np.ndarray,
    p_change: np.ndarray,
    half: np.ndarray,
    w_smooth: int = 7,
    step: int = 6,
    alpha_grad: float = 0.25,
    scale: float = 2.5,
) -> np.ndarray:
    """Compute local log-odds total variation impulse at turnovers and stoppages."""
    p = np.clip(proba.astype(np.float64), 1e-5, 1.0 - 1e-5)
    p_live = np.clip(p[:, HOME] + p[:, AWAY], 1e-5, 1.0)
    lo_ha = np.log((p[:, HOME] + 1e-4) / (p[:, AWAY] + 1e-4))
    lo_ld = np.log(p_live / (p[:, DEAD] + 1e-4))

    out = p_change.astype(np.float64).copy()
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx) <= 2 * step + 2:
            continue
        s_ha = _moving_avg(lo_ha[idx], w_smooth)
        s_ld = _moving_avg(lo_ld[idx], w_smooth)
        d_ha = np.zeros_like(s_ha)
        d_ld = np.zeros_like(s_ld)
        d_ha[step:-step] = np.abs(s_ha[2 * step:] - s_ha[:-2 * step])
        d_ld[step:-step] = np.abs(s_ld[2 * step:] - s_ld[:-2 * step])
        imp = np.tanh(np.maximum(d_ha, 0.8 * d_ld) / scale)
        out[idx] = np.clip((1.0 - alpha_grad) * out[idx] + alpha_grad * imp, 0.0, 1.0)
    return out.astype(np.float32)


def rescue_short_spells(
    pred: np.ndarray,
    proba: np.ndarray,
    p_change: np.ndarray,
    half: np.ndarray,
    min_len: int = 15,
    max_len: int = 85,
    surr_min_len: int = 75,
    margin_thresh: float = 0.15,
    peak_thresh: float = 0.62,
    edge_change_weight: float = 0.35,
) -> np.ndarray:
    """Scale-free rescuer for short (0.6s - 3.4s) possessions swallowed by frame-additive Viterbi.

    When a long predicted spell of team S (H or A) contains a contiguous candidate interval [a, b]
    of length in [min_len, max_len] where the opposite team O = 1 - S has:
      - P(O) > P(S) and P(O) > P(D) throughout [a, b]
      - Peak P(O) >= peak_thresh
      - Scale-free score: mean(log P(O) - log P(S)) + edge_change_weight * (max_c(a) + max_c(b)) >= margin_thresh
      - Both left and right remaining segments of S have length >= surr_min_len (so we don't fragment short spells)
    we split the merged spell and insert O on [a, b].
    """
    out = pred.copy()
    p = np.clip(proba.astype(np.float64), 1e-5, 1.0)
    c = p_change.astype(np.float64)

    for h in (1, 2):
        idx_h = np.flatnonzero(half == h)
        if len(idx_h) == 0:
            continue
        h0, h1 = int(idx_h[0]), int(idx_h[-1]) + 1
        seq = out[h0:h1]
        # Find maximal runs in seq
        changes = np.flatnonzero(np.diff(seq) != 0) + 1
        starts = np.r_[0, changes]
        stops = np.r_[changes, len(seq)]

        for r_start, r_stop in zip(starts, stops):
            state = int(seq[r_start])
            if state not in (HOME, AWAY):
                continue
            if (r_stop - r_start) < (2 * surr_min_len + min_len):
                continue
            opp = AWAY if state == HOME else HOME
            sl = slice(h0 + r_start, h0 + r_stop)
            p_sub = p[sl]
            c_sub = c[sl]

            # Candidate frames where opposite team dominates both current team and Dead
            cand_mask = (p_sub[:, opp] > p_sub[:, state]) & (p_sub[:, opp] > p_sub[:, DEAD])
            if not np.any(cand_mask):
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, cand_mask.astype(np.int8), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                L = b - a
                if L < min_len or L > max_len:
                    continue
                if a < surr_min_len or (len( p_sub ) - b) < surr_min_len:
                    continue
                peak_p = float(np.max(p_sub[a:b, opp]))
                if peak_p < peak_thresh:
                    continue
                mean_log_adv = float(np.mean(np.log(p_sub[a:b, opp]) - np.log(p_sub[a:b, state])))
                c_left = float(np.max(c_sub[max(0, a - 5):min(len(c_sub), a + 6)]))
                c_right = float(np.max(c_sub[max(0, b - 5):min(len(c_sub), b + 6)]))
                score = mean_log_adv + edge_change_weight * (c_left + c_right)
                if score >= margin_thresh:
                    out[h0 + r_start + a : h0 + r_start + b] = opp
    return out


def main() -> None:
    t0 = time.time()
    print("=== EXP-F: SYMMETRIC PLATEAU DECODER + LOG-ODDS TV IMPULSE + SHORT-SPELL RESCUER ===", flush=True)
    sportec = load_all()
    store = ROOT / "local_data/oof/nested_degrade_v3_v1ctl_tcnlocal"
    sym_dir = ROOT / "experiments/outputs/exp_b_sym_oof"

    oof = {}
    for T in MATCH_IDS:
        sb = np.load(sym_dir / f"sym_{T}.npz")
        st = np.load(sym_dir / f"tcn_sym_{T}.npz")
        oof[T] = {
            "g_sym": sb["symgbm_tta"],
            "tr_sym": sb["symtrans_tta"],
            "t_sym": st["symtcn_tta_proba"],
            "tc_sym": st["symtcn_tta_change"],
        }

    # Build blended OOFs for Sym_Linear(0.55, 0.50) and Sym_Hybrid(0.55, 0.50)
    data_lin = {}
    data_hyb = {}
    for T in MATCH_IDS:
        d = oof[T]
        p_lin = blend_proba(d["g_sym"], d["t_sym"], 0.55, "linear")
        p_hyb = blend_proba(d["g_sym"], d["t_sym"], 0.55, "hybrid")
        c = 0.50 * d["tr_sym"] + 0.50 * d["tc_sym"]
        data_lin[T] = (p_lin, sportec[T].labels, sportec[T].half, c)
        data_hyb[T] = (p_hyb, sportec[T].labels, sportec[T].half, c)

    # 1. Strict Leave-One-Match-Out (LOO) decoder tuning on the 6 other Symmetric OOFs
    print("\n--- 1. Tuning LOO Symmetric Decoders (6 folds -> 1 fold) & Full 7-Fold Plateau Decoder ---", flush=True)
    loo_dec_lin, loo_dec_hyb = {}, {}
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        st_l = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=100 + k))
        st_l.optimize(lambda t: mean_score(suggest(t), data_lin, others), n_trials=150)
        loo_dec_lin[T] = plateau(st_l)

        st_h = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=200 + k))
        st_h.optimize(lambda t: mean_score(suggest(t), data_hyb, others), n_trials=150)
        loo_dec_hyb[T] = plateau(st_h)

    # Full 7-fold plateau center (exact s4_3 protocol) for Sample & BAR-ATH
    st_full_lin = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
    st_full_lin.optimize(lambda t: mean_score(suggest(t), data_lin, list(MATCH_IDS)), n_trials=200)
    dec_v5_lin = plateau(st_full_lin)

    st_full_hyb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
    st_full_hyb.optimize(lambda t: mean_score(suggest(t), data_hyb, list(MATCH_IDS)), n_trials=200)
    dec_v5_hyb = plateau(st_full_hyb)

    print(f"  dec_v5_lin: {dec_v5_lin.to_dict()}", flush=True)
    print(f"  dec_v5_hyb: {dec_v5_hyb.to_dict()}", flush=True)
    with open(sym_dir / "dec_v5_sym.pkl", "wb") as f:
        pickle.dump((loo_dec_lin, loo_dec_hyb, dec_v5_lin, dec_v5_hyb), f)

    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std, _, _, _ = pickle.load(f)

    # 2. Evaluate on Judge 1 (7-fold LOMO)
    print("\n--- JUDGE 1: 7-Fold LOMO Evaluation ---", flush=True)
    for label, data_dict, fdecs, tv_alpha, rescue_cfg in [
        ("A0. Sym_Lin(0.55,0.50) + old fold_dec_std", data_lin, fold_dec_std, 0.0, None),
        ("A1. Sym_Lin(0.55,0.50) + LOO Sym Decoder", data_lin, loo_dec_lin, 0.0, None),
        ("A2. Sym_Hyb(0.55,0.50) + old fold_dec_std", data_hyb, fold_dec_std, 0.0, None),
        ("A3. Sym_Hyb(0.55,0.50) + LOO Sym Decoder", data_hyb, loo_dec_hyb, 0.0, None),
        ("B1. Sym_Hyb + old_dec + LogOdds TV (alpha=0.15)", data_hyb, fold_dec_std, 0.15, None),
        ("B2. Sym_Hyb + old_dec + LogOdds TV (alpha=0.25)", data_hyb, fold_dec_std, 0.25, None),
        ("B3. Sym_Hyb + LOO_dec + LogOdds TV (alpha=0.15)", data_hyb, loo_dec_hyb, 0.15, None),
        ("C1. Sym_Hyb + old_dec + ShortSpellRescue(m=0.18, p=0.65)", data_hyb, fold_dec_std, 0.0, (0.18, 0.65)),
        ("C2. Sym_Hyb + old_dec + ShortSpellRescue(m=0.14, p=0.60)", data_hyb, fold_dec_std, 0.0, (0.14, 0.60)),
        ("C3. Sym_Hyb + LOO_dec + ShortSpellRescue(m=0.18, p=0.65)", data_hyb, loo_dec_hyb, 0.0, (0.18, 0.65)),
        ("D1. Sym_Hyb + old_dec + TV(0.15) + Rescue(0.18, 0.65)", data_hyb, fold_dec_std, 0.15, (0.18, 0.65)),
    ]:
        scores = []
        for T in MATCH_IDS:
            p, y, h, c = data_dict[T]
            c_use = augment_p_change_with_logodds_tv(p, c, h, alpha_grad=tv_alpha) if tv_alpha > 0 else c
            pred = decode_v4(p, h, fdecs[T], p_change=c_use)
            if rescue_cfg is not None:
                pred = rescue_short_spells(pred, p, c_use, h, margin_thresh=rescue_cfg[0], peak_thresh=rescue_cfg[1])
            scores.append(metrics.score(y, pred, h)["spell_f1_mean"])
        arr = np.array(scores)
        print(f"  {label:56s}: mean={arr.mean():.4f} std={arr.std():.4f} | folds={[round(x, 4) for x in scores]}", flush=True)

    # 3. Evaluate on Judge 2 (Metrica Sample) and Judge 3 (BAR-ATH)
    print("\n--- JUDGE 2 (Metrica Sample) & JUDGE 3 (BAR-ATH) ---", flush=True)
    with open(sym_dir / "symgbm_full.pkl", "rb") as f:
        symgbm_full = pickle.load(f)
    with open(sym_dir / "symtrans_full.pkl", "rb") as f:
        symtrans_full = pickle.load(f)
    with open(sym_dir / "symtcn_full.pkl", "rb") as f:
        symtcn_full = pickle.load(f)
    dec_v4 = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})

    sample_data = {}
    for gid in GAMES:
        ms = load_sample(gid)
        Xs = build_features(ms, f"sample_{gid}_raw", "v1ctl")
        pg = predict_gbm_tta(symgbm_full, Xs, ms.half, ms.pitch)
        tr = predict_trans_tta(symtrans_full, Xs, ms.half, ms.pitch)
        pt, ct = predict_tcn_tta(symtcn_full, Xs, ms.half, ms.pitch)
        sample_data[gid] = (ms, pg, pt, 0.5 * tr + 0.5 * ct)

    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)
    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom["valid"].to_numpy().astype(bool))
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)
    Xa_rep = build_features(atd_rep, "atd_rep_True_True_None", "v1ctl")
    pg_a = predict_gbm_tta(symgbm_full, Xa_rep, atd.half, atd.pitch)
    tr_a = predict_trans_tta(symtrans_full, Xa_rep, atd.half, atd.pitch)
    pt_a, ct_a = predict_tcn_tta(symtcn_full, Xa_rep, atd.half, atd.pitch)
    c_a_base = 0.5 * tr_a + 0.5 * ct_a

    for vname, bmode, dec_use, tv_alpha, rescue_cfg in [
        ("0. Sym_Lin + dec_v4 (current v5_sym_linear)", "linear", dec_v4, 0.0, None),
        ("1. Sym_Lin + dec_v5_lin (tuned on Sym OOFs)", "linear", dec_v5_lin, 0.0, None),
        ("2. Sym_Hyb + dec_v4 (current v5_sym_hybrid)", "hybrid", dec_v4, 0.0, None),
        ("3. Sym_Hyb + dec_v5_hyb (tuned on Sym OOFs)", "hybrid", dec_v5_hyb, 0.0, None),
        ("4. Sym_Hyb + dec_v4 + LogOdds TV (0.15)", "hybrid", dec_v4, 0.15, None),
        ("5. Sym_Hyb + dec_v5_hyb + LogOdds TV (0.15)", "hybrid", dec_v5_hyb, 0.15, None),
        ("6. Sym_Hyb + dec_v4 + Rescue(0.18, 0.65)", "hybrid", dec_v4, 0.0, (0.18, 0.65)),
        ("7. Sym_Hyb + dec_v4 + Rescue(0.14, 0.60)", "hybrid", dec_v4, 0.0, (0.14, 0.60)),
        ("8. Sym_Hyb + dec_v5_hyb + Rescue(0.18, 0.65)", "hybrid", dec_v5_hyb, 0.0, (0.18, 0.65)),
        ("9. Sym_Lin + dec_v5_lin + TV(0.15) + Rescue(0.18, 0.65)", "linear", dec_v5_lin, 0.15, (0.18, 0.65)),
    ]:
        s_raw, s_nod = [], []
        for gid in GAMES:
            ms, pg, pt, c = sample_data[gid]
            p = blend_proba(pg, pt, 0.55, bmode)
            c_use = augment_p_change_with_logodds_tv(p, c, ms.half, alpha_grad=tv_alpha) if tv_alpha > 0 else c
            pred = decode_v4(p, ms.half, dec_use, p_change=c_use)
            if rescue_cfg is not None:
                pred = rescue_short_spells(pred, p, c_use, ms.half, margin_thresh=rescue_cfg[0], peak_thresh=rescue_cfg[1])
            s_raw.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

        p_a = blend_proba(pg_a, pt_a, 0.55, bmode)
        c_a = augment_p_change_with_logodds_tv(p_a, c_a_base, half_dec, alpha_grad=tv_alpha) if tv_alpha > 0 else c_a_base
        pred_a = decode_v4(p_a, half_dec, dec_use, p_change=c_a)
        if rescue_cfg is not None:
            pred_a = rescue_short_spells(pred_a, p_a, c_a, half_dec, margin_thresh=rescue_cfg[0], peak_thresh=rescue_cfg[1])
        pred_a[second_start:kick2] = DEAD
        pred_a[~atd.graded] = DEAD

        pub = metrics.score_public(pred_a, bootstrap_n=200)
        pl = plausibility(pred_a, atd.half, atd.graded)
        print(
            f"  {vname:56s} | Sample={np.mean(s_raw):.4f} (noD={np.mean(s_nod):.4f}) "
            f"| Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
            f"| pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}%",
            flush=True,
        )

    print(f"\nEXP-F completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
