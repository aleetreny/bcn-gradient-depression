"""Grand Unified Champion Evaluation & Submission Generator (EXP-A + EXP-B + EXP-B2 + EXP-D + EXP-E2).

Combines all orthogonal, rule-compliant improvements verified in `exp_lab`:
1. EXP-A: Hybrid Linear + Log-Pool (Product-of-Experts) blending + Orthogonalized HSMM.
2. EXP-B & B2: Exact H <-> A + 180-deg Pitch Rotation Equivariance (Inference TTA + Symmetric Training `SymGBM`, `SymTrans`, `SymTCN`).
3. EXP-D: Minimum Bayes Risk (MBR) Consensus Decoding for `spell_f1_mean`.
4. EXP-E2: Broadcast Top-K Clock-Locked Blackout Repair + Smart Tagging Live-Phase Prior during TV-cut blackouts on BAR-ATH.

Evaluates every combination across all 3 official judges:
- Judge 1: 7-Fold Nested LOMO on Sportec `degrade_v3` (inner-tuned decoders per fold, zero outer leakage)
- Judge 2: Metrica Sample Data (3 full games, standard and `without_dead_margin`)
- Judge 3: BAR-ATH full-match label-free plausibility + 5-min public check gate + verified submission CSVs
"""
from __future__ import annotations

from dataclasses import replace
import json
import pickle
import sys
import time
from pathlib import Path

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
from src.data.common import AWAY, DEAD, HOME
from src.data.degrade import degrade
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import DecoderParams, decode as decode_v4
from src.eval import metrics
from src.features.symmetry import mirror_v1ctl_df, swap_labels, swap_proba
from src.models.gbm import TwoHeadGBM
from src.models.transition import TransitionGBM, change_targets
from exp_a_nested_eval import mbr_select, top_k_params_from_study
from exp_b_symmetry_eval import blend_proba, predict_gbm_tta, predict_tcn_tta, predict_trans_tta
from exp_e_multimodal_bar_ath import build_smart_tagging_arrays
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match

optuna.logging.set_verbosity(optuna.logging.WARNING)


def verify_submission_csv(out_path: Path, pred_a: np.ndarray, atd, second_start: int, kick2: int, name: str) -> None:
    import possession_metrics as pm
    assert len(pred_a) == 199319, f"{name}: Expected 199319 rows, got {len(pred_a)}"
    assert (pred_a[second_start:kick2] == DEAD).all(), f"{name}: Halftime break not all DEAD"
    assert (pred_a[~atd.graded] == DEAD).all(), f"{name}: Ungraded frames not all DEAD"
    table = pm.read_label_csv(str(out_path))
    reason = pm.validate_submission(table, 199319)
    assert not reason, f"{name}: Official validator failed: {reason}"


def main():
    t0 = time.time()
    print("=== GRAND UNIFIED CHAMPION EVALUATION (EXP-A + EXP-B + EXP-B2 + EXP-D + EXP-E2) ===", flush=True)

    sportec = load_all()
    cfg = load_config(ROOT / "configs/degrade_v3.json")
    store = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl_tcnlocal"
    sym_dir = ROOT / "experiments" / "outputs" / "exp_b_sym_oof"

    # 1. Load 7-fold OOF arrays from v4 store + EXP-B + EXP-B2
    oof = {}
    for T in MATCH_IDS:
        g_orig = np.load(store / T / f"gbm_outer_{T}.npy")
        tr_orig = np.load(store / T / f"trans_outer_{T}.npy")
        z_orig = np.load(store / T / f"tcn_outer_{T}.npz")
        sb = np.load(sym_dir / f"sym_{T}.npz")
        st = np.load(sym_dir / f"tcn_sym_{T}.npz")
        oof[T] = {
            "g_orig": g_orig,
            "tr_orig": tr_orig,
            "t_orig": z_orig["proba"],
            "tc_orig": z_orig["change"],
            "g_tta": sb["gbm_tta"],
            "g_sym": sb["symgbm_tta"],
            "tr_sym": sb["symtrans_tta"],
            "t_tta": st["tcn_tta_proba"],
            "tc_tta": st["tcn_tta_change"],
            "t_sym": st["symtcn_tta_proba"],
            "tc_sym": st["symtcn_tta_change"],
        }

    # 2. Inner Optuna decoders per fold (strict nested protocol, cached)
    dec_cache = sym_dir / "fold_decs.pkl"
    if dec_cache.exists():
        with open(dec_cache, "rb") as f:
            fold_dec_std, fold_dec_hyb, fold_cands, fold_w = pickle.load(f)
    else:
        fold_dec_std = {}
        fold_dec_hyb = {}
        fold_cands = {}
        fold_w = {}
        for k, T in enumerate(MATCH_IDS):
            others = [m for m in MATCH_IDS if m != T]
            inner_std = {}
            for J in others:
                g = np.load(store / T / f"gbm_inner_{J}.npy")
                tr = np.load(store / T / f"trans_inner_{J}.npy")
                z = np.load(store / T / f"tcn_inner_{J}.npz")
                inner_std[J] = (0.5 * (g + z["proba"]), sportec[J].labels, sportec[J].half, 0.5 * (tr + z["change"]))

            st1 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
            st1.optimize(lambda t: mean_score(suggest(t), inner_std, others), n_trials=150)
            fold_dec_std[T] = plateau(st1)
            fold_dec_hyb[T] = fold_dec_std[T]
            _, cands, w = top_k_params_from_study(st1)
            fold_cands[T], fold_w[T] = cands, w
        with open(dec_cache, "wb") as f:
            pickle.dump((fold_dec_std, fold_dec_hyb, fold_cands, fold_w), f)

    print("\n--- JUDGE 1: 7-Fold Nested LOMO on Degraded Sportec ---", flush=True)
    lomo_configs = [
        ("0. v4_ens_mean Baseline (orig GBM + orig TCN)",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"],
         "linear", 0.50, 0.50, fold_dec_std, False),
        ("1. EXP-A: Hyb(0.60,0.75) on v4",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"],
         "hybrid", 0.60, 0.75, fold_dec_hyb, False),
        ("2. SymTCN_TTA alone (no GBM)",
         lambda d: d["t_sym"], lambda d: d["t_sym"], lambda d: d["tc_sym"], lambda d: d["tc_sym"],
         "linear", 0.00, 0.00, fold_dec_std, False),
        ("3. orig_GBM + SymTCN_TTA (linear 0.50, 0.50)",
         lambda d: d["g_orig"], lambda d: d["t_sym"], lambda d: d["tr_orig"], lambda d: d["tc_sym"],
         "linear", 0.50, 0.50, fold_dec_hyb, False),
        ("4. SymGBM_TTA + SymTCN_TTA + SymTrans_TTA (linear 0.55, 0.50)",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"],
         "linear", 0.55, 0.50, fold_dec_hyb, False),
        ("5. SymGBM_TTA + SymTCN_TTA + SymTrans_TTA (hybrid 0.55, 0.50) [7/7 FOLDS WINNER]",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, fold_dec_hyb, False),
        ("6. SymGBM_TTA + SymTCN_TTA + SymTrans_TTA (hybrid 0.50, 0.50)",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"],
         "hybrid", 0.50, 0.50, fold_dec_hyb, False),
        ("7. SymGBM_TTA + SymTCN_TTA + SymTrans_TTA (hybrid 0.55, 0.50) + MBR",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, fold_dec_hyb, True),
        ("8. GRAND ENSEMBLE (0.5*GBM_TTA+0.5*SymGBM, SymTCN_TTA, hyb 0.55, 0.50)",
         lambda d: 0.5 * (d["g_tta"] + d["g_sym"]),
         lambda d: d["t_sym"],
         lambda d: 0.5 * (d["tr_orig"] + d["tr_sym"]),
         lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, fold_dec_hyb, False),
    ]

    for name, fg, ft, ftr, ftc, bmode, wg, wc, fdec, use_mbr in lomo_configs:
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
                pred = decode_v4(p, sportec[T].half, fdec[T], p_change=c)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        arr = np.array(scores)
        print(f"  {name:68s}: mean={arr.mean():.4f} std={arr.std():.4f} | folds={[round(x, 4) for x in scores]}", flush=True)

    # 3. JUDGE 2 & JUDGE 3: Metrica Sample (3 games) + BAR-ATH (Unrepaired vs Broadcast-Repaired)
    print("\n--- JUDGE 2 (Metrica Sample) & JUDGE 3 (BAR-ATH) ---", flush=True)
    models_v4 = pickle.loads((ROOT / "local_data/models/v4/models.pkl").read_bytes())
    gbm_v4 = models_v4["gbm"]
    trans_v4 = models_v4["trans"]
    tcn_v4 = models_v4["tcn"]
    dec_v4 = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_tcnlocal_ens_mean.json").read_text()).items() if not k.startswith("_")})

    if not (sym_dir / "symgbm_full.pkl").exists():
        print("  Training full 7-match SymGBM & SymTrans for Sample & BAR-ATH...", flush=True)
        tag = "degrade_v3"
        Xs_all, ys_all, hs_all, gs_all = [], [], [], []
        Xs_mir, ys_mir, gs_mir = [], [], []
        for i, (mid, m) in enumerate(sportec.items()):
            for key, over, seed in TRAIN_VARIANTS:
                c = replace(cfg, **over) if over else cfg
                X_tr = build_features(degrade(m, c, seed), f"sportec_{mid}_{tag}_{key}", "v1ctl")
                Xs_all.append(X_tr)
                ys_all.append(m.labels)
                hs_all.append(m.half)
                gs_all.append(i)
                Xs_mir.append(mirror_v1ctl_df(X_tr, m.half, m.pitch))
                ys_mir.append(swap_labels(m.labels))
                gs_mir.append(len(sportec) + i)
        Xs_sym = Xs_all + Xs_mir
        ys_sym = ys_all + ys_mir
        gs_sym = gs_all + gs_mir
        ct_all = [change_targets(y, h) for y, h in zip(ys_all, hs_all)]
        symgbm_full = TwoHeadGBM(seed=0).fit(Xs_sym, ys_sym, gs_sym)
        symtrans_full = TransitionGBM(seed=0).fit(Xs_sym, ct_all + ct_all)
        with open(sym_dir / "symgbm_full.pkl", "wb") as f:
            pickle.dump(symgbm_full, f)
        with open(sym_dir / "symtrans_full.pkl", "wb") as f:
            pickle.dump(symtrans_full, f)
    else:
        with open(sym_dir / "symgbm_full.pkl", "rb") as f:
            symgbm_full = pickle.load(f)
        with open(sym_dir / "symtrans_full.pkl", "rb") as f:
            symtrans_full = pickle.load(f)
    with open(sym_dir / "symtcn_full.pkl", "rb") as f:
        symtcn_full = pickle.load(f)

    # Precompute Sample predictions
    sample_matches, sample_preds = {}, {}
    for gid in GAMES:
        ms = load_sample(gid)
        Xs = build_features(ms, f"sample_{gid}_raw", "v1ctl")
        sample_matches[gid] = ms
        pt0, ct0 = tcn_v4.predict(Xs, ms.half)
        pt_tta, ct_tta = predict_tcn_tta(tcn_v4, Xs, ms.half, ms.pitch)
        pt_sym, ct_sym = predict_tcn_tta(symtcn_full, Xs, ms.half, ms.pitch)
        sample_preds[gid] = {
            "g_orig": gbm_v4.predict_proba(Xs),
            "g_tta": predict_gbm_tta(gbm_v4, Xs, ms.half, ms.pitch),
            "g_sym": predict_gbm_tta(symgbm_full, Xs, ms.half, ms.pitch),
            "tr_orig": trans_v4.predict(Xs),
            "tr_sym": predict_trans_tta(symtrans_full, Xs, ms.half, ms.pitch),
            "t_orig": pt0,
            "tc_orig": ct0,
            "t_tta": pt_tta,
            "tc_tta": ct_tta,
            "t_sym": pt_sym,
            "tc_sym": ct_sym,
        }

    # Precompute BAR-ATH predictions (both unrepaired and Broadcast-Blackout-Repaired)
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_dec = np.where(atd.graded, atd.half, 0)
    st_info = build_smart_tagging_arrays(atd)

    Pb, Bb, slot_team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    hom_valid = hom["valid"].to_numpy().astype(bool)
    mapping_enh, _ = enhanced_broadcast_alignment(atd, Pb, hom_valid)
    atd_rep = repair_atd_match(atd, Pb, Bb, slot_team_b, mapping_enh, True, True, None)

    Xa_raw = build_features(atd, "atd", "v1ctl")
    Xa_rep = build_features(atd_rep, "atd_rep_True_True_None", "v1ctl")

    def get_atd_dict(Xa):
        pt0, ct0 = tcn_v4.predict(Xa, atd.half)
        pt_tta, ct_tta = predict_tcn_tta(tcn_v4, Xa, atd.half, atd.pitch)
        pt_sym, ct_sym = predict_tcn_tta(symtcn_full, Xa, atd.half, atd.pitch)
        return {
            "g_orig": gbm_v4.predict_proba(Xa),
            "g_tta": predict_gbm_tta(gbm_v4, Xa, atd.half, atd.pitch),
            "g_sym": predict_gbm_tta(symgbm_full, Xa, atd.half, atd.pitch),
            "tr_orig": trans_v4.predict(Xa),
            "tr_sym": predict_trans_tta(symtrans_full, Xa, atd.half, atd.pitch),
            "t_orig": pt0,
            "tc_orig": ct0,
            "t_tta": pt_tta,
            "tc_tta": ct_tta,
            "t_sym": pt_sym,
            "tc_sym": ct_sym,
        }

    atd_raw_d = get_atd_dict(Xa_raw)
    atd_rep_d = get_atd_dict(Xa_rep)
    pred_v4_base = None

    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    eval_variants = [
        ("0. v4_ens_mean Baseline (unrepaired)",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"],
         "linear", 0.50, 0.50, False, False, False, None),
        ("1. v4 + Broadcast Blackout Repair (EXP-E2)",
         lambda d: d["g_orig"], lambda d: d["t_orig"], lambda d: d["tr_orig"], lambda d: d["tc_orig"],
         "linear", 0.50, 0.50, True, False, False, None),
        ("2. Both_TTA (v4) + Linear(0.50,1.00) + Repair",
         lambda d: d["g_tta"], lambda d: d["t_tta"], lambda d: d["tr_orig"], lambda d: d["tc_tta"],
         "linear", 0.50, 1.00, True, False, False, "submission_v5_tta_m3.csv"),
        ("3. SymGBM_TTA + TCN_TTA + SymTrans_TTA + Linear(0.50,0.75) + Repair",
         lambda d: d["g_sym"], lambda d: d["t_tta"], lambda d: d["tr_sym"], lambda d: d["tc_tta"],
         "linear", 0.50, 0.75, True, False, False, None),
        ("4. SymGBM_TTA + SymTCN_TTA + SymTrans_TTA + Linear(0.55,0.50) + Repair",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"],
         "linear", 0.55, 0.50, True, False, False, "submission_v5_sym_linear.csv"),
        ("5. SymGBM_TTA + SymTCN_TTA + SymTrans_TTA + Hyb(0.55,0.50) + Repair",
         lambda d: d["g_sym"], lambda d: d["t_sym"], lambda d: d["tr_sym"], lambda d: d["tc_sym"],
         "hybrid", 0.55, 0.50, True, False, False, "submission_v5_sym_hybrid.csv"),
        ("6. GRAND ENSEMBLE (v4_TTA + Sym_TTA) + Hyb(0.55,0.50) + MBR + Repair",
         lambda d: 0.5 * (d["g_tta"] + d["g_sym"]),
         lambda d: 0.5 * (d["t_tta"] + d["t_sym"]),
         lambda d: 0.5 * (d["tr_orig"] + d["tr_sym"]),
         lambda d: 0.5 * (d["tc_tta"] + d["tc_sym"]),
         "hybrid", 0.55, 0.50, True, True, False, "submission_v5_champion.csv"),
        ("7. GRAND ENSEMBLE + Hyb(0.55,0.50) + MBR + Repair + SmartTag Blackouts",
         lambda d: 0.5 * (d["g_tta"] + d["g_sym"]),
         lambda d: 0.5 * (d["t_tta"] + d["t_sym"]),
         lambda d: 0.5 * (d["tr_orig"] + d["tr_sym"]),
         lambda d: 0.5 * (d["tc_tta"] + d["tc_sym"]),
         "hybrid", 0.55, 0.50, True, True, True, "submission_v5_champion_multimodal.csv"),
    ]

    for vname, fg, ft, ftr, ftc, bmode, wg, wc, use_rep, use_mbr, use_st, sub_file in eval_variants:
        s_raw, s_nod = [], []
        for gid in GAMES:
            ms = sample_matches[gid]
            d = sample_preds[gid]
            pg, pt = fg(d), ft(d)
            tr, tc = ftr(d), ftc(d)
            p = blend_proba(pg, pt, w_gbm=wg, mode=bmode)
            c = wc * tr + (1.0 - wc) * tc
            if use_mbr:
                cands_7 = [fold_dec_hyb[T] for T in MATCH_IDS]
                preds_k = [decode_v4(p, ms.half, dp, p_change=c) for dp in cands_7]
                pred = mbr_select(preds_k, ms.half)
            else:
                pred = decode_v4(p, ms.half, dec_v4, p_change=c)
            s_raw.append(metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"])
            s_nod.append(metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])

        da = atd_rep_d if use_rep else atd_raw_d
        m_cur = atd_rep if use_rep else atd
        pg_a, pt_a = fg(da), ft(da)
        tr_a, tc_a = ftr(da), ftc(da)
        p_a = blend_proba(pg_a, pt_a, w_gbm=wg, mode=bmode)
        c_a = wc * tr_a + (1.0 - wc) * tc_a
        if use_st:
            n_vis_now = np.isfinite(m_cur.P[..., 0]).sum(axis=1)
            still_bo = m_cur.graded & (n_vis_now <= 6) & (np.abs(st_info["net_team_score"]) > 0) & (~st_info["in_set_piece"])
            lp = np.log(np.clip(p_a.astype(np.float64), 1e-6, 1.0))
            sc = np.clip(st_info["net_team_score"][still_bo], -1.5, 1.5)
            lp[still_bo, HOME] += 2.5 * sc
            lp[still_bo, AWAY] -= 2.5 * sc
            lp[still_bo, DEAD] -= 2.5 * 2.5
            lp -= lp.max(axis=1, keepdims=True)
            e = np.exp(lp)
            p_a = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

        if use_mbr:
            cands_7 = [fold_dec_hyb[T] for T in MATCH_IDS]
            preds_a_k = [decode_v4(p_a, half_dec, dp, p_change=c_a) for dp in cands_7]
            pred_a = mbr_select(preds_a_k, half_dec)
        else:
            pred_a = decode_v4(p_a, half_dec, dec_v4, p_change=c_a)
        pred_a[second_start:kick2] = DEAD
        pred_a[~atd.graded] = DEAD

        if pred_v4_base is None:
            pred_v4_base = pred_a.copy()

        pub = metrics.score_public(pred_a, bootstrap_n=200)
        pl = plausibility(pred_a, atd.half, atd.graded)
        diff_v4 = 100.0 * float((pred_a[atd.graded] != pred_v4_base[atd.graded]).mean())

        print(
            f"  {vname:68s} | Sample={np.mean(s_raw):.4f} (noD={np.mean(s_nod):.4f}) "
            f"| Pub={pub['spell_f1_mean']:.4f} (D_f1={pub['per_class']['D']['f1']:.3f}) "
            f"| pos/m={pl['poss_per_min']:.2f} %D={pl['pct_D']:.1f}% | diff={diff_v4:.2f}%",
            flush=True,
        )

        if sub_file is not None:
            char_map = np.array(["H", "A", "D"])
            df_sub = pd.DataFrame({"frame": np.arange(len(pred_a), dtype=np.int32), "label": char_map[pred_a.astype(np.int8)]})
            out_path = sub_dir / sub_file
            df_sub.to_csv(out_path, index=False)
            verify_submission_csv(out_path, pred_a, atd, second_start, kick2, sub_file)
            print(f"    -> Saved & officially validated candidate submission: {out_path}", flush=True)

    print(f"\nGRAND UNIFIED CHAMPION EVALUATION completed in {(time.time()-t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
