"""F3.3: GBM + TCN ensemble (+ change signal) evaluated with the fully nested LOMO.

    .venv311/bin/python scripts/s3_3_ensemble_nested.py [--config configs/degrade_v3.json] [--trials 150]
                                                        [--out experiments/outputs/029]

Completes the nested scheme for the TCN: for outer T, TCNs trained on the five matches other than T
and J predict each inner J (the outer TCN, trained on the six, comes from F3.2). With the GBM inner /
outer probabilities (S2.4b) and the M3 transition heads (F3.1) already stored, it tunes and scores:
  A) mean(GBM, TCN) probabilities + M3 change probability;
  B) mean(GBM, TCN) probabilities + mean(M3, TCN change head);
each with its own decoder study on the inner predictions (plateau centre), T scored once. Paired with
v1.1 and v1.2 (M3) on the same folds.
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
from s3_1_transition_head import mean_score, plateau, suggest  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402
from src.models.transition import change_features  # noqa: E402

TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))
VARIANTS = {"A_mean_proba_m3_change": "trans", "B_mean_proba_mean_change": "mean"}


def main() -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/029")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = args.config.stem
    store = ROOT / "local_data" / "oof" / "nested_{}".format(tag)
    ref_v11 = pd.read_csv(ROOT / "experiments/outputs/023/nested_folds.csv", index_col=0)["nested_spellf1"]
    ref_m3 = pd.read_csv(ROOT / "experiments/outputs/027/folds.csv", index_col=0)["m3_spellf1"]

    sportec = load_all()
    train_X, test_X, labels, halves = {}, {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            train_X[(mid, key)] = change_features(cached_features(degrade(m, c, seed), "sportec_{}_{}_{}".format(mid, tag, key)), m.half)
        test_X[mid] = change_features(cached_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag)), m.half)
        labels[mid], halves[mid] = m.labels, m.half

    def tcn(train_ids, test_id, seed, path):
        if path.is_file():
            z = np.load(path)
            return z["proba"], z["change"]
        model = TCNModel(seed=seed).fit([train_X[(m, k)] for m in train_ids for k, _, _ in TRAIN_VARIANTS],
                                        [labels[m] for m in train_ids for _ in TRAIN_VARIANTS],
                                        [halves[m] for m in train_ids for _ in TRAIN_VARIANTS])
        p, c = model.predict(test_X[test_id], halves[test_id])
        np.savez(path, proba=p, change=c)
        return p, c

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        parts = {}
        for J in others:
            p_t, c_t = tcn([m for m in others if m != J], J, 300 + k, store / T / "tcn_inner_{}.npz".format(J))
            parts[J] = (np.load(store / T / "inner_{}.npy".format(J)), p_t,
                        np.load(store / T / "trans_inner_{}.npy".format(J)), c_t)
        zo = np.load(store / T / "tcn_outer_{}.npz".format(T))
        outer = (np.load(store / T / "outer_{}.npy".format(T)), zo["proba"], np.load(store / T / "trans_outer_{}.npy".format(T)), zo["change"])
        row = {"fold": T, "v1_1": ref_v11[T], "v1_2_m3": ref_m3[T]}
        for name, change in VARIANTS.items():
            def pack(g, t, tr, tc):
                return (g + t) / 2, (tr if change == "trans" else (tr + tc) / 2)
            inner = {}
            for J in others:
                pj, cj = pack(*parts[J])
                inner[J] = (pj, labels[J], halves[J], cj)
            st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
            st.optimize(lambda t: mean_score(suggest(t), inner, others), n_trials=args.trials)
            params = plateau(st)
            proba, pc = pack(*outer)
            pred = decode(proba, halves[T], params, p_change=pc)
            np.save(store / T / "pred_{}_{}.npy".format(name, T), pred)
            r = metrics.score(labels[T], pred, halves[T])
            row[name] = r["spell_f1_mean"]
            row[name + "_poss"] = r["n_pred_spells"]
            (store / T / "params_{}.json".format(name)).write_text(json.dumps(params.to_dict()))
        row["poss_gt"] = r["n_gt_spells"]
        rows.append(row)
        print("[{}] {}".format(T, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items() if kk != "fold"}), flush=True)

    folds = pd.DataFrame(rows).set_index("fold")
    summary = {"v1_1": float(folds["v1_1"].mean()), "v1_2_m3": float(folds["v1_2_m3"].mean())}
    for name in VARIANTS:
        summary[name] = float(folds[name].mean())
        summary[name + "_std"] = float(folds[name].std(ddof=0))
        summary[name + "_folds_better_than_v1_2"] = int((folds[name] > folds["v1_2_m3"]).sum())
        summary[name + "_delta_vs_v1_2"] = float((folds[name] - folds["v1_2_m3"]).mean())
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(folds)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
