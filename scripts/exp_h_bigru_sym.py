"""EXP-H: Symmetric Channel-Norm BiGRU Sequence Model (`BiGRUSeqModel`) for Long Ball Gaps.

Why BiGRUSeqModel:
TCNModel has a fixed local receptive field (10.1s - 20.4s), whereas 85% of missing-ball frames in
BAR-ATH lie in gaps longer than 10s (up to 221s). BiGRUSeqModel uses the exact same per-frame
channel normalization (norm="channel") and data loader / loss function as TCNModel, followed by a
3-level dilated Conv1d local stem (dilation=1, 2, 4, channels=64) + a 2-layer Bidirectional GRU
(hidden_size=48 per direction -> 96 dim) + residual projection + two heads (head_p, head_c).

Evaluates:
1. 7-Fold Outer LOMO `SymBiGRU_TTA` (`epochs=12` on `X + mirror(X)` using MPS GPU), saving OOF predictions
   to `experiments/outputs/exp_h_oof/`.
2. Full 7-match `symbigru_full.pkl`.
3. Blending `SymBiGRU_TTA` with `SymTCN_TTA` + `SymGBM_TTA` across:
   - Judge 1: 7-Fold Nested LOMO on Sportec `degrade_v3` (overall + stratified by ball-gap duration >10s)
   - Judge 2: Metrica Sample (3 games, standard & without_dead_margin)
   - Judge 3: BAR-ATH Broadcast-Repaired (Public 5-min gate + Full-Match Plausibility)
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from games import GAMES as GAME_INFO, anchor_video_frame
from s2_1_degradation import load_config
from s2_3_frame_models import plausibility
from s3_1_transition_head import mean_score, plateau, suggest
from s3_7_nested_full import TRAIN_VARIANTS, build_features
from s3_11_tcn_local import without_dead_margin
from s4_6_broadcast_align import load_broadcast
from src.data.common import DEAD
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from src.features.symmetry import mirror_v1ctl_df, swap_labels
from src.models.bigru import BiGRUSeqModel
from exp_a_nested_eval import mbr_select, top_k_params_from_study
from exp_b_symmetry_eval import blend_proba, predict_tcn_tta
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match

optuna.logging.set_verbosity(optuna.logging.WARNING)


def frame_acc_on_mask(y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray) -> float:
    valid = (y_true >= 0) & mask
    if not np.any(valid):
        return float("nan")
    pred = np.argmax(proba[valid], axis=1)
    return float((pred == y_true[valid]).mean())


def main():
    t0 = time.time()
    print("=== EXP-H: Symmetric Channel-Norm BiGRU Sequence Model (`SymBiGRU_TTA`) ===", flush=True)
    sportec = load_all()
    cfg = load_config(ROOT / "configs/degrade_v3.json")
    tag = "degrade_v3"
    store = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"
    out_dir = ROOT / "experiments" / "outputs" / "exp_h_oof"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_X, train_X_mir, test_X = {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            X_tr = build_features(degrade(m, c, seed), f"sportec_{mid}_{tag}_{key}", "v1ctl")
            train_X[(mid, key)] = X_tr
            train_X_mir[(mid, key)] = mirror_v1ctl_df(X_tr, m.half, m.pitch)
        test_X[mid] = build_features(degrade(m, cfg, 0), f"sportec_{mid}_{tag}_s0", "v1ctl")

    oof_symbigru_tta = {}
    for k, T in enumerate(MATCH_IDS):
        f_path = out_dir / f"bigru_sym_{T}.npz"
        if f_path.exists():
            z = np.load(f_path)
            oof_symbigru_tta[T] = (z["symbigru_tta_proba"], z["symbigru_tta_change"])
            print(f"  Fold {k} ({T}) loaded from cache {f_path.name}", flush=True)
            continue

        tf = time.time()
        train_ids = [m for m in MATCH_IDS if m != T]
        Xs_std = [train_X[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_std = [sportec[mid].labels for mid in train_ids for _ in TRAIN_VARIANTS]
        hs_std = [sportec[mid].half for mid in train_ids for _ in TRAIN_VARIANTS]

        Xs_sym = Xs_std + [train_X_mir[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys_sym = ys_std + [swap_labels(sportec[mid].labels) for mid in train_ids for _ in TRAIN_VARIANTS]
        hs_sym = hs_std + hs_std

        bigru_sym = BiGRUSeqModel(norm="channel", epochs=12, seed=k).fit(Xs_sym, ys_sym, hs_sym)
        pb_tta, cb_tta = predict_tcn_tta(bigru_sym, test_X[T], sportec[T].half, sportec[T].pitch)
        oof_symbigru_tta[T] = (pb_tta, cb_tta)

        np.savez_compressed(
            f_path,
            symbigru_tta_proba=pb_tta,
            symbigru_tta_change=cb_tta,
        )
        print(f"  Fold {k} ({T}) SymBiGRU+TTA trained in {time.time()-tf:.1f}s", flush=True)

    # Load inner-tuned decoders and EXP-B / EXP-B2 OOF predictions
    dec_cache = sym_dir / "fold_decs.pkl"
    with open(dec_cache, "rb") as f:
        fold_dec_std, fold_dec_hyb, fold_cands, fold_w = pickle.load(f)

    oof = {}
    for T in MATCH_IDS:
        g_orig = np.load(store / T / f"gbm_outer_{T}.npy")
        tr_orig = np.load(store / T / f"trans_outer_{T}.npy")
        z_orig = np.load(store / T / f"tcn_outer_{T}.npz")
        sb = np.load(sym_dir / f"sym_{T}.npz")
        st = np.load(sym_dir / f"tcn_sym_{T}.npz")
        pb_tta, cb_tta = oof_symbigru_tta[T]
        oof[T] = {
            "g_orig": g_orig,
            "tr_orig": tr_orig,
            "t_orig": z_orig["proba"],
            "tc_orig": z_orig["change"],
            "g_tta": sb["gbm_tta"],
            "g_sym": sb["symgbm_tta"],
            "tr_sym": sb["symtrans_tta"],
            "t_sym": st["symtcn_tta_proba"],
            "tc_sym": st["symtcn_tta_change"],
            "b_sym": pb_tta,
            "bc_sym": cb_tta,
        }

    # Diagnostic 1: Frame accuracy on short/visible vs long missing-ball gaps (>10s = 250 frames)
    print("\n--- Diagnostic: Frame-Level Accuracy by Ball-Gap Length (Sportec degrade_v3 LOMO) ---", flush=True)
    for mname, f_p in [
        ("SymGBM_TTA", lambda d: d["g_sym"]),
        ("SymTCN_TTA", lambda d: d["t_sym"]),
        ("SymBiGRU_TTA", lambda d: d["b_sym"]),
        ("0.65*SymTCN + 0.35*SymBiGRU", lambda d: 0.65 * d["t_sym"] + 0.35 * d["b_sym"]),
        ("0.50*SymTCN + 0.50*SymBiGRU", lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"]),
    ]:
        acc_all, acc_short, acc_long = [], [], []
        for T in MATCH_IDS:
            X_t = test_X[T]
            y_t = sportec[T].labels
            p_t = f_p(oof[T])
            # `ball_gap_len` or `time_since_last_ball` + `time_to_next_ball` in v1 features
            if "ball_gap_s" in X_t.columns:
                gap_s = X_t["ball_gap_s"].to_numpy()
            elif "dt_last_ball" in X_t.columns and "dt_next_ball" in X_t.columns:
                gap_s = np.nan_to_num(X_t["dt_last_ball"].to_numpy(), nan=0.0) + np.nan_to_num(X_t["dt_next_ball"].to_numpy(), nan=0.0)
            else:
                b_miss = X_t["ball_x"].isna().to_numpy()
                gap_s = b_miss.astype(float) * 15.0
            acc_all.append(frame_acc_on_mask(y_t, p_t, np.ones(len(y_t), dtype=bool)))
            acc_short.append(frame_acc_on_mask(y_t, p_t, gap_s <= 10.0))
            acc_long.append(frame_acc_on_mask(y_t, p_t, gap_s > 10.0))
        print(
            f"  {mname:32s}: All={np.nanmean(acc_all):.4f} | Gap<=10s={np.nanmean(acc_short):.4f} | Gap>10s={np.nanmean(acc_long):.4f}",
            flush=True,
        )

    print("\n--- JUDGE 1: 7-Fold Outer LOMO Results (Sportec degrade_v3) ---", flush=True)
    lomo_variants = [
        ("0. v4_ens_mean Baseline (orig GBM + orig TCN)",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"],
         "linear", 0.50, 0.50, False),
        ("1. SymTCN_TTA alone",
         lambda d: d["t_sym"], lambda d: d["t_sym"], lambda d: d["tc_sym"], lambda d: d["tc_sym"],
         "linear", 0.00, 0.00, False),
        ("2. SymBiGRU_TTA alone",
         lambda d: d["b_sym"], lambda d: d["b_sym"], lambda d: d["bc_sym"], lambda d: d["bc_sym"],
         "linear", 0.00, 0.00, False),
        ("3. SeqBlend(0.65*SymTCN + 0.35*SymBiGRU) alone",
         lambda d: 0.65 * d["t_sym"] + 0.35 * d["b_sym"],
         lambda d: 0.65 * d["t_sym"] + 0.35 * d["b_sym"],
         lambda d: 0.65 * d["tc_sym"] + 0.35 * d["bc_sym"],
         lambda d: 0.65 * d["tc_sym"] + 0.35 * d["bc_sym"],
         "linear", 0.00, 0.00, False),
        ("4. SeqBlend(0.50*SymTCN + 0.50*SymBiGRU) alone",
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         "linear", 0.00, 0.00, False),
        ("5. SymGBM_TTA + SymTCN_TTA (hyb 0.55, 0.50) [EXP-B2 Winner]",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("6. SymGBM_TTA + SymBiGRU_TTA (hyb 0.55, 0.50)",
         lambda d: d["g_sym"], lambda d: d["b_sym"], lambda d: d["tr_sym"], lambda d: d["bc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("7. SymGBM_TTA + (0.75*SymTCN + 0.25*SymBiGRU) (hyb 0.55, 0.50)",
         lambda d: d["g_sym"],
         lambda d: 0.75 * d["t_sym"] + 0.25 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.75 * d["tc_sym"] + 0.25 * d["bc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("8. SymGBM_TTA + (0.65*SymTCN + 0.35*SymBiGRU) (hyb 0.55, 0.50)",
         lambda d: d["g_sym"],
         lambda d: 0.65 * d["t_sym"] + 0.35 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.65 * d["tc_sym"] + 0.35 * d["bc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("9. SymGBM_TTA + (0.50*SymTCN + 0.50*SymBiGRU) (hyb 0.55, 0.50)",
         lambda d: d["g_sym"],
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("10. SymGBM_TTA + (0.50*SymTCN + 0.50*SymBiGRU) (hyb 0.50, 0.50)",
         lambda d: d["g_sym"],
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         "hybrid", 0.50, 0.50, False),
        ("11. SymGBM_TTA + (0.65*SymTCN + 0.35*SymBiGRU) (hyb 0.50, 0.50) + MBR",
         lambda d: d["g_sym"],
         lambda d: 0.65 * d["t_sym"] + 0.35 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.65 * d["tc_sym"] + 0.35 * d["bc_sym"],
         "hybrid", 0.50, 0.50, True),
        ("12. Equal 3-Family (0.34*SymGBM + 0.33*SymTCN + 0.33*SymBiGRU) lin",
         lambda d: d["g_sym"],
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         "linear", 0.34, 0.34, False),
        ("13. Equal 3-Family (0.34*SymGBM + 0.33*SymTCN + 0.33*SymBiGRU) hyb",
         lambda d: d["g_sym"],
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         "hybrid", 0.34, 0.34, False),
    ]

    for label, fg, ft, ftr, ftc, bmode, wg, wc, use_mbr in lomo_variants:
        scores = []
        for T in MATCH_IDS:
            d = oof[T]
            pg, pt = fg(d), ft(d)
            tr, tc = ftr(d), ftc(d)
            p = pt if wg == 0.0 else blend_proba(pg, pt, w_gbm=wg, mode=bmode)
            c = tc if wc == 0.0 else wc * tr + (1.0 - wc) * tc
            if use_mbr:
                preds_k = [decode_v4(p, sportec[T].half, dp, p_change=c) for dp in fold_cands[T]]
                pred = mbr_select(preds_k, sportec[T].half, fold_w[T])
            else:
                pred = decode_v4(p, sportec[T].half, fold_dec_std[T], p_change=c)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(
            f"  {label:66s}: mean={arr.mean():.4f} std={arr.std():.4f} | folds={[round(x, 4) for x in scores]}",
            flush=True,
        )

    # Full 7-match SymBiGRU for Judge 2 (Sample) and Judge 3 (BAR-ATH)
    full_pkl = out_dir / "symbigru_full.pkl"
    if full_pkl.exists():
        with open(full_pkl, "rb") as f:
            symbigru_full = pickle.load(f)
        print(f"\n--- Loaded Full 7-Match SymBiGRU from {full_pkl.name} ---", flush=True)
    else:
        print("\n--- Training Full 7-Match SymBiGRU for Sample & BAR-ATH ---", flush=True)
        tf = time.time()
        Xs_all = [train_X[(mid, key)] for mid in MATCH_IDS for key, _, _ in TRAIN_VARIANTS]
        ys_all = [sportec[mid].labels for mid in MATCH_IDS for _ in TRAIN_VARIANTS]
        hs_all = [sportec[mid].half for mid in MATCH_IDS for _ in TRAIN_VARIANTS]
        Xs_all_sym = Xs_all + [train_X_mir[(mid, key)] for mid in MATCH_IDS for key, _, _ in TRAIN_VARIANTS]
        ys_all_sym = ys_all + [swap_labels(sportec[mid].labels) for mid in MATCH_IDS for _ in TRAIN_VARIANTS]
        hs_all_sym = hs_all + hs_all
        symbigru_full = BiGRUSeqModel(norm="channel", epochs=12, seed=0).fit(Xs_all_sym, ys_all_sym, hs_all_sym)
        with open(full_pkl, "wb") as f:
            pickle.dump(symbigru_full, f)
        print(f"  Full SymBiGRU trained in {time.time()-tf:.1f}s", flush=True)

    # Evaluate on Judge 2 (Metrica Sample) and Judge 3 (BAR-ATH Repaired)
    print("\n--- JUDGE 2 (Metrica Sample) & JUDGE 3 (BAR-ATH Repaired) ---", flush=True)
    models_v4 = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())
    gbm_v4 = models_v4["gbm"]
    trans_v4 = models_v4["trans"]
    tcn_v4 = models_v4["tcn"]
    dec_v4 = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})

    with open(sym_dir / "symgbm_full.pkl", "rb") as f:
        symgbm_full = pickle.load(f)
    with open(sym_dir / "symtrans_full.pkl", "rb") as f:
        symtrans_full = pickle.load(f)
    with open(sym_dir / "symtcn_full.pkl", "rb") as f:
        symtcn_full = pickle.load(f)

    from exp_b_symmetry_eval import predict_gbm_tta, predict_trans_tta

    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)

    j23_cache = out_dir / "bigru_j23_preds.pkl"
    if j23_cache.exists():
        with open(j23_cache, "rb") as f:
            sample_preds, atd_rep_d = pickle.load(f)
    else:
        sample_preds = {}
        for gid in GAMES:
            ms = sample_matches[gid]
            Xs = build_features(ms, f"sample_{gid}_raw", "v1ctl")
            pt0, ct0 = tcn_v4.predict(Xs, ms.half)
            pt_sym, ct_sym = predict_tcn_tta(symtcn_full, Xs, ms.half, ms.pitch)
            pb_sym, cb_sym = predict_tcn_tta(symbigru_full, Xs, ms.half, ms.pitch)
            sample_preds[gid] = {
                "g_orig": gbm_v4.predict_proba(Xs),
                "tr_orig": trans_v4.predict(Xs),
                "t_orig": pt0,
                "tc_orig": ct0,
                "g_sym": predict_gbm_tta(symgbm_full, Xs, ms.half, ms.pitch),
                "tr_sym": predict_trans_tta(symtrans_full, Xs, ms.half, ms.pitch),
                "t_sym": pt_sym,
                "tc_sym": ct_sym,
                "b_sym": pb_sym,
                "bc_sym": cb_sym,
            }

        Pb, Bb, slot_team_b = load_broadcast()
        hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
        hom_valid = hom["valid"].to_numpy().astype(bool)
        mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
        atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)
        Xa_rep = build_features(atd_rep, "atd_rep_True_True_None", "v1ctl")

        pt0_a, ct0_a = tcn_v4.predict(Xa_rep, atd.half)
        pt_sym_a, ct_sym_a = predict_tcn_tta(symtcn_full, Xa_rep, atd.half, atd.pitch)
        pb_sym_a, cb_sym_a = predict_tcn_tta(symbigru_full, Xa_rep, atd.half, atd.pitch)
        atd_rep_d = {
            "g_orig": gbm_v4.predict_proba(Xa_rep),
            "tr_orig": trans_v4.predict(Xa_rep),
            "t_orig": pt0_a,
            "tc_orig": ct0_a,
            "g_sym": predict_gbm_tta(symgbm_full, Xa_rep, atd.half, atd.pitch),
            "tr_sym": predict_trans_tta(symtrans_full, Xa_rep, atd.half, atd.pitch),
            "t_sym": pt_sym_a,
            "tc_sym": ct_sym_a,
            "b_sym": pb_sym_a,
            "bc_sym": cb_sym_a,
        }
        with open(j23_cache, "wb") as f:
            pickle.dump((sample_preds, atd_rep_d), f)

    j23_variants = [
        ("0. v4 + Repair Baseline",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"],
         "linear", 0.50, 0.50, False),
        ("1. SymBiGRU_TTA alone + Repair",
         lambda d: d["b_sym"], lambda d: d["b_sym"], lambda d: d["bc_sym"], lambda d: d["bc_sym"],
         "linear", 0.00, 0.00, False),
        ("2. SymTCN_TTA alone + Repair",
         lambda d: d["t_sym"], lambda d: d["t_sym"], lambda d: d["tc_sym"], lambda d: d["tc_sym"],
         "linear", 0.00, 0.00, False),
        ("3. SymGBM_TTA + SymTCN_TTA (hyb 0.55, 0.50) + Repair",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("4. SymGBM_TTA + SymBiGRU_TTA (hyb 0.55, 0.50) + Repair",
         lambda d: d["g_sym"], lambda d: d["b_sym"], lambda d: d["tr_sym"], lambda d: d["bc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("5. SymGBM_TTA + (0.65*SymTCN + 0.35*SymBiGRU) (lin 0.55, 0.50) + Repair",
         lambda d: d["g_sym"],
         lambda d: 0.65 * d["t_sym"] + 0.35 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.65 * d["tc_sym"] + 0.35 * d["bc_sym"],
         "linear", 0.55, 0.50, False),
        ("6. SymGBM_TTA + (0.65*SymTCN + 0.35*SymBiGRU) (hyb 0.55, 0.50) + Repair",
         lambda d: d["g_sym"],
         lambda d: 0.65 * d["t_sym"] + 0.35 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.65 * d["tc_sym"] + 0.35 * d["bc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("7. SymGBM_TTA + (0.50*SymTCN + 0.50*SymBiGRU) (hyb 0.55, 0.50) + Repair",
         lambda d: d["g_sym"],
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         "hybrid", 0.55, 0.50, False),
        ("8. Equal 3-Family (0.34*SymGBM + 0.33*SymTCN + 0.33*SymBiGRU) lin + Repair",
         lambda d: d["g_sym"],
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         "linear", 0.34, 0.34, False),
        ("9. Equal 3-Family (0.34*SymGBM + 0.33*SymTCN + 0.33*SymBiGRU) hyb + Repair",
         lambda d: d["g_sym"],
         lambda d: 0.50 * d["t_sym"] + 0.50 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.50 * d["tc_sym"] + 0.50 * d["bc_sym"],
         "hybrid", 0.34, 0.34, False),
        ("10. SymGBM_TTA + (0.65*SymTCN + 0.35*SymBiGRU) (hyb 0.55, 0.50) + MBR + Repair",
         lambda d: d["g_sym"],
         lambda d: 0.65 * d["t_sym"] + 0.35 * d["b_sym"],
         lambda d: d["tr_sym"],
         lambda d: 0.65 * d["tc_sym"] + 0.35 * d["bc_sym"],
         "hybrid", 0.55, 0.50, True),
    ]

    for vname, fg, ft, ftr, ftc, bmode, wg, wc, use_mbr in j23_variants:
        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            d = sample_preds[gid]
            pg, pt = fg(d), ft(d)
            tr, tc = ftr(d), ftc(d)
            p = pt if wg == 0.0 else blend_proba(pg, pt, w_gbm=wg, mode=bmode)
            c = tc if wc == 0.0 else wc * tr + (1.0 - wc) * tc
            if use_mbr:
                cands_7 = [fold_dec_std[T] for T in MATCH_IDS]
                preds_k = [decode_v4(p, ms.half, dp, p_change=c) for dp in cands_7]
                pred = mbr_select(preds_k, ms.half)
            else:
                pred = decode_v4(p, ms.half, dec_v4, p_change=c)
            s_raw.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

        da = atd_rep_d
        pg_a, pt_a = fg(da), ft(da)
        tr_a, tc_a = ftr(da), ftc(da)
        p_a = pt_a if wg == 0.0 else blend_proba(pg_a, pt_a, w_gbm=wg, mode=bmode)
        c_a = tc_a if wc == 0.0 else wc * tr_a + (1.0 - wc) * tc_a
        if use_mbr:
            cands_7 = [fold_dec_std[T] for T in MATCH_IDS]
            preds_a_k = [decode_v4(p_a, half_dec, dp, p_change=c_a) for dp in cands_7]
            pred_a = mbr_select(preds_a_k, half_dec)
        else:
            pred_a = decode_v4(p_a, half_dec, dec_v4, p_change=c_a)
        pred_a[second_start:kick2] = DEAD
        pred_a[~atd.graded] = DEAD

        pub = metrics.score_public(pred_a, bootstrap_n=100)
        pl = plausibility(pred_a, atd.half, atd.graded)
        pct_H = 100.0 * float((pred_a[atd.graded] == 0).mean())
        pct_A = 100.0 * float((pred_a[atd.graded] == 1).mean())
        print(
            f"  {vname:74s} | Sample={np.mean(s_raw):.4f} (noD={np.mean(s_nod):.4f}) "
            f"| Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
            f"| pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}% (%H={pct_H:.1f}/%A={pct_A:.1f})",
            flush=True,
        )

    print(f"\nEXP-H completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
