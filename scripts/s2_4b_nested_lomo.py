"""S2.4b: fully nested LOMO of the whole chain (GBM + calibration + decoder), review 003 point 1.

    .venv311/bin/python scripts/s2_4b_nested_lomo.py [--config configs/degrade_v3.json] [--trials 120]
                                                     [--out experiments/outputs/023]

For each outer match T:
  1. inner OOF: for each other match J, a TwoHeadGBM trained (and calibrated) on the remaining five
     matches -- neither T nor J -- predicts J;
  2. the decoder is tuned on those six inner OOF predictions (Optuna, plateau centre as in S2.4);
  3. a TwoHeadGBM trained on the six matches except T predicts T, decoded with the step-2 parameters,
     and T is scored once.
Training data per match: the degradation config with seeds 1, 2 and a static_hide_prob 0.5 variant;
test data: seed 0 (as in S2.3). Saves per-fold outer probabilities and decoded predictions under
local_data/oof/nested_<tag>/ for S2.6 and the Sportec panels. Also reports, for comparison, the
S2.3 smoothing (w 12) and the S2.4 protocol (decoder tuned on outer-model OOF) on the same folds.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from s2_4_decoder import plateau_center, study  # noqa: E402
from src.data.common import DEAD  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.gbm import TwoHeadGBM, smooth_argmax  # noqa: E402

TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))


def d_metrics(y, pred):
    """Precision / recall / F1 of class D on labelled frames (alive_dead_f1 of the scorer uses alive as positive)."""
    lab = y >= 0
    t, p = (y[lab] == DEAD), (pred[lab] == DEAD)
    tp = float(np.sum(t & p))
    prec = tp / max(p.sum(), 1)
    rec = tp / max(t.sum(), 1)
    return {"D_precision": prec, "D_recall": rec, "D_f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/023")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = args.config.stem
    store = ROOT / "local_data" / "oof" / "nested_{}".format(tag)

    sportec = load_all()
    train_X, test_X, labels, halves = {}, {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            train_X[(mid, key)] = cached_features(degrade(m, c, seed), "sportec_{}_{}_{}".format(mid, tag, key))
        test_X[mid] = cached_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag))
        labels[mid], halves[mid] = m.labels, m.half

    def fit_predict(train_ids, test_id, seed, path):
        if path.is_file():
            return np.load(path)
        Xs = [train_X[(mid, k)] for mid in train_ids for k, _, _ in TRAIN_VARIANTS]
        ys = [labels[mid] for mid in train_ids for _ in TRAIN_VARIANTS]
        gs = [i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]
        proba = TwoHeadGBM(calibrate=True, seed=seed).fit(Xs, ys, gs).predict_proba(test_X[test_id]).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, proba)
        return proba

    rows = []
    outer_proba = {}
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        inner = {}
        for J in others:
            inner[J] = (fit_predict([m for m in others if m != J], J, seed=100 + k, path=store / T / "inner_{}.npy".format(J)),
                        labels[J], halves[J])
        st = study(inner, others, args.trials, seed=k)
        params, _ = plateau_center(st)
        outer_proba[T] = fit_predict(others, T, seed=k, path=store / T / "outer_{}.npy".format(T))
        pred = decode(outer_proba[T], halves[T], params)
        np.save(store / T / "pred_{}.npy".format(T), pred)
        r = metrics.score(labels[T], pred, halves[T])
        row = {"fold": T, "nested_spellf1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
               "alive_dead_f1_official": r["alive_dead_f1"], **d_metrics(labels[T], pred),
               "poss_pred": r["n_pred_spells"], "poss_gt": r["n_gt_spells"],
               "smooth_w12": metrics.score(labels[T], smooth_argmax(outer_proba[T], halves[T], 12), halves[T])["spell_f1_mean"],
               "inner_best": st.best_value, **{"p_" + kk: v for kk, v in params.to_dict().items()}}
        rows.append(row)
        (store / T / "params.json").write_text(json.dumps(params.to_dict()))
        print("[{}] {}".format(T, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items() if kk != "fold"}), flush=True)

    # S2.4 protocol on the same outer probabilities, for comparison (decoder tuned on the other outer OOF)
    outer = {T: (outer_proba[T], labels[T], halves[T]) for T in MATCH_IDS}
    s24 = []
    for k, T in enumerate(MATCH_IDS):
        st = study(outer, [m for m in MATCH_IDS if m != T], args.trials, seed=k)
        params, _ = plateau_center(st)
        s24.append(metrics.score(labels[T], decode(outer_proba[T], halves[T], params), halves[T])["spell_f1_mean"])
    folds = pd.DataFrame(rows).set_index("fold")
    folds["s24_protocol"] = s24

    summary = {"config": str(args.config), "nested_mean": float(folds["nested_spellf1"].mean()),
               "nested_std": float(folds["nested_spellf1"].std(ddof=0)),
               "smooth_w12_mean": float(folds["smooth_w12"].mean()), "s24_protocol_mean": float(np.mean(s24)),
               "folds_nested_better_than_smooth": int((folds["nested_spellf1"] > folds["smooth_w12"]).sum()),
               "D_f1_mean": float(folds["D_f1"].mean()), "D_recall_mean": float(folds["D_recall"].mean()),
               "D_precision_mean": float(folds["D_precision"].mean())}
    folds.to_csv(args.out / "nested_folds.csv", float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(folds)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
