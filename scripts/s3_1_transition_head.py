"""F3.1 (M3): transition head + time-dependent switch cost, evaluated with the fully nested LOMO.

    .venv311/bin/python scripts/s3_1_transition_head.py [--config configs/degrade_v3.json] [--trials 150]
                                                        [--out experiments/outputs/027]

Reuses the nested two-head probabilities of S2.4b (local_data/oof/nested_<tag>/<T>/inner_J.npy and
outer_T.npy) and trains the transition head in the same nested scheme: for outer T, inner heads on the
five matches other than T and J predict J; the outer head on the six matches other than T predicts T.
The decoder (now with beta_ha) is tuned on the inner predictions (plateau centre) and T is scored once.
Paired comparison with v1.1 (same folds, S2.4b nested_folds.csv), change-detection errors, and the
transition head's own AUC / average precision.
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
from src.data.degrade import degrade  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import errors, metrics  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.transition import TransitionGBM, change_features, change_targets  # noqa: E402

TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))
SPACE = {"lam_ha": (0.5, 40.0), "lam_ld": (0.5, 40.0), "lam_dl": (0.5, 40.0), "dmin_team": (1, 50), "dmin_dead": (1, 150),
         "beta_ha": (0.0, 6.0)}
INT_PARAMS = ("dmin_team", "dmin_dead")


def suggest(trial):
    p = {}
    for k, (lo, hi) in SPACE.items():
        if k in INT_PARAMS:
            p[k] = trial.suggest_int(k, lo, hi)
        elif k == "beta_ha":
            p[k] = trial.suggest_float(k, lo, hi)
        else:
            p[k] = trial.suggest_float(k, lo, hi, log=True)
    return DecoderParams(**p)


def mean_score(params, data, ids):
    return float(np.mean([metrics.score(data[i][1], decode(data[i][0], data[i][2], params, p_change=data[i][3]),
                                        data[i][2])["spell_f1_mean"] for i in ids]))


def plateau(st, top=0.10):
    df = st.trials_dataframe(attrs=("value", "params")).dropna().sort_values("value", ascending=False)
    df = df.head(max(5, int(len(df) * top)))
    p = {k: float(np.median(df["params_" + k])) for k in SPACE}
    for k in INT_PARAMS:
        p[k] = int(round(p[k]))
    return DecoderParams(**p)


def main() -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.metrics import average_precision_score, roc_auc_score
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/027")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = args.config.stem
    store = ROOT / "local_data" / "oof" / "nested_{}".format(tag)
    ref = pd.read_csv(ROOT / "experiments/outputs/023/nested_folds.csv", index_col=0)

    sportec = load_all()
    train_X, test_X, ytr, labels, halves = {}, {}, {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            train_X[(mid, key)] = change_features(cached_features(degrade(m, c, seed), "sportec_{}_{}_{}".format(mid, tag, key)), m.half)
        test_X[mid] = change_features(cached_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag)), m.half)
        labels[mid], halves[mid] = m.labels, m.half
        ytr[mid] = change_targets(m.labels, m.half)

    def head(train_ids, test_id, seed, path):
        if path.is_file():
            return np.load(path)
        model = TransitionGBM(seed=seed).fit([train_X[(mid, k)] for mid in train_ids for k, _, _ in TRAIN_VARIANTS],
                                            [ytr[mid] for mid in train_ids for _ in TRAIN_VARIANTS])
        p = model.predict(test_X[test_id])
        np.save(path, p)
        if test_id not in train_ids and path.name.startswith("trans_outer"):
            model.importance(25).to_csv(path.with_suffix(".importance.csv"))
        return p

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        inner = {J: (np.load(store / T / "inner_{}.npy".format(J)), labels[J], halves[J],
                     head([m for m in others if m != J], J, 200 + k, store / T / "trans_inner_{}.npy".format(J))) for J in others}
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
        st.optimize(lambda t: mean_score(suggest(t), inner, others), n_trials=args.trials)
        params = plateau(st)
        p_out = head(others, T, k, store / T / "trans_outer_{}.npy".format(T))
        proba = np.load(store / T / "outer_{}.npy".format(T))
        pred = decode(proba, halves[T], params, p_change=p_out)
        np.save(store / T / "pred_m3_{}.npy".format(T), pred)
        r = metrics.score(labels[T], pred, halves[T])
        y = ytr[T]
        lab = y >= 0
        dec_m3 = errors.decompose(labels[T], pred, halves[T])["table"].set_index("item")["count"]
        v11 = np.load(store / T / "pred_{}.npy".format(T))
        dec_v11 = errors.decompose(labels[T], v11, halves[T])["table"].set_index("item")["count"]
        rows.append({"fold": T, "m3_spellf1": r["spell_f1_mean"], "v1_1_spellf1": ref.loc[T, "nested_spellf1"],
                     "macro_f1": r["macro_f1"], "poss_pred": r["n_pred_spells"], "poss_gt": r["n_gt_spells"],
                     "trans_auc": float(roc_auc_score(y[lab], p_out[lab])), "trans_ap": float(average_precision_score(y[lab], p_out[lab])),
                     "trans_base_rate": float(y[lab].mean()),
                     "missed_changes_m3": int(dec_m3["  missed"].iloc[0]), "missed_changes_v1_1": int(dec_v11["  missed"].iloc[0]),
                     "short_missed_m3": int(dec_m3["  short (< 2 s), missed"]), "short_missed_v1_1": int(dec_v11["  short (< 2 s), missed"]),
                     **{"p_" + kk: v for kk, v in params.to_dict().items()}})
        print("[{}] {}".format(T, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rows[-1].items() if kk != "fold"}), flush=True)

    folds = pd.DataFrame(rows).set_index("fold")
    diff = folds["m3_spellf1"] - folds["v1_1_spellf1"]
    summary = {"m3_mean": float(folds["m3_spellf1"].mean()), "m3_std": float(folds["m3_spellf1"].std(ddof=0)),
               "v1_1_mean": float(folds["v1_1_spellf1"].mean()), "delta_mean": float(diff.mean()),
               "folds_improved": int((diff > 0).sum()), "trans_auc_mean": float(folds["trans_auc"].mean()),
               "trans_ap_mean": float(folds["trans_ap"].mean()), "beta_ha_median": float(folds["p_beta_ha"].median())}
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(folds)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
