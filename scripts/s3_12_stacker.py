"""F3.12: temporal stacker over the GBM, TCN and M3 outputs (instead of their plain mean), fully nested.

    .venv311/bin/python scripts/s3_12_stacker.py [--trials 150] [--out experiments/outputs/037]

Base signals per frame (stored F3.7 v1ctl predictions): P(alive) and P(H|alive) of the GBM and of the
TCN, the M3 change probability and the TCN change head. Stacker features: each signal, its values
shifted by +/- 6, 12, 25 and 50 frames, centred means over 1 and 3 s, and forward-minus-backward means
over 0.5 and 2 s (inside each half). A small two-head GBM (H/A/D) and a change GBM are trained on them.
Nesting, for each outer match T (directory of T, nothing trained on T):
  - inner stacked prediction of J: stacker trained on the inner predictions of the other five K
    (made by base models trained without T and K);
  - outer stacked prediction of T: stacker trained on all six inner predictions;
  - the decoder is tuned on the six inner stacked predictions (plateau centre) and T is scored once.
Compared with ens_mean of F3.7 on the same folds.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from s3_1_transition_head import mean_score, plateau, suggest  # noqa: E402
from s3_7_nested_full import TRAIN_VARIANTS  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.eval.store import check_store, identity  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402
from src.models.transition import TransitionGBM, change_targets  # noqa: E402

SHIFTS = (6, 12, 25, 50)
STACK_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=16)


def _shift(x, half, k):
    """x[t + k] inside each half, edge-padded."""
    out = np.empty_like(x)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        v = x[idx]
        j = np.clip(np.arange(len(v)) + k, 0, len(v) - 1)
        out[idx] = v[j]
    return out


def _mean(x, half, lo, hi):
    """Mean of x[t + lo .. t + hi] inside each half (edge-padded)."""
    out = np.empty_like(x)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        v = np.pad(x[idx], (max(0, -lo), max(0, hi)), mode="edge")
        c = np.r_[0.0, np.cumsum(v, dtype=np.float64)]
        n = len(idx)
        start = np.arange(n) + lo + max(0, -lo)
        out[idx] = ((c[start + (hi - lo + 1)] - c[start]) / (hi - lo + 1)).astype(np.float32)
    return out


def stack_features(g, tcn, tr, half):
    pa_g = 1 - g[:, 2]
    pa_t = 1 - tcn["proba"][:, 2]
    sig = {"g_alive": pa_g, "g_home": g[:, 0] / np.maximum(pa_g, 1e-6),
           "t_alive": pa_t, "t_home": tcn["proba"][:, 0] / np.maximum(pa_t, 1e-6),
           "m3_change": tr, "t_change": tcn["change"]}
    f = {}
    for name, x in sig.items():
        x = np.asarray(x, np.float32)
        f[name] = x
        for k in SHIFTS:
            f["{}_p{}".format(name, k)] = _shift(x, half, k)
            f["{}_m{}".format(name, k)] = _shift(x, half, -k)
        for w in (25, 75):
            f["{}_c{}".format(name, w)] = _mean(x, half, -(w // 2), w // 2)
        for w in (12, 50):
            f["{}_d{}".format(name, w)] = _mean(x, half, 1, w) - _mean(x, half, -w, -1)
    return pd.DataFrame(f)


def main() -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/037")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    store = ROOT / "local_data" / "oof" / "nested_{}_v1ctl".format(args.config.stem)
    sportec = load_all()
    check_store(store, identity(asdict(cfg), "v1ctl", sportec,
                                {"train_variants": TRAIN_VARIANTS, "script": "s3_7 seeds 1000*(1+model)+k inner, k outer"}),
                ignore=("source",))
    labels = {mid: m.labels for mid, m in sportec.items()}
    halves = {mid: m.half for mid, m in sportec.items()}

    def feats(T, who, J):
        return stack_features(np.load(store / T / "gbm_{}_{}.npy".format(who, J)), np.load(store / T / "tcn_{}_{}.npz".format(who, J)),
                              np.load(store / T / "trans_{}_{}.npy".format(who, J)), halves[J])

    def fit_predict(train_ids, F_train, F_test, seed):
        Xs = [F_train[j] for j in train_ids]
        ys = [labels[j] for j in train_ids]
        gbm = TwoHeadGBM(params=dict(STACK_PARAMS), subsample_every=5, seed=seed).fit(Xs, ys, list(range(len(Xs))))
        chg = TransitionGBM(params=dict(STACK_PARAMS), seed=seed).fit(
            Xs, [change_targets(y, halves[j]) for y, j in zip(ys, train_ids)])
        return gbm.predict_proba(F_test).astype(np.float32), chg.predict(F_test)

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        F_inner = {J: feats(T, "inner", J) for J in others}
        inner = {}
        for J in others:
            p, c = fit_predict([K for K in others if K != J], F_inner, F_inner[J], 5000 + k)
            inner[J] = (p, labels[J], halves[J], c)
        p_out, c_out = fit_predict(others, F_inner, feats(T, "outer", T), 6000 + k)
        np.savez(store / T / "stack_outer_{}.npz".format(T), proba=p_out, change=c_out)
        row = {"fold": T}
        for how in ("stack", "stack_nochange"):
            data = {J: (v[0], v[1], v[2], v[3] if how == "stack" else None) for J, v in inner.items()}
            st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
            st.optimize(lambda t: mean_score(suggest(t), data, others), n_trials=args.trials)
            params = plateau(st)
            pred = decode(p_out, halves[T], params, p_change=c_out if how == "stack" else None)
            r = metrics.score(labels[T], pred, halves[T])
            row[how] = r["spell_f1_mean"]
            row[how + "_poss"] = r["n_pred_spells"]
            (store / T / "params_{}.json".format(how)).write_text(json.dumps(params.to_dict()))
        row["poss_gt"] = r["n_gt_spells"]
        rows.append(row)
        print("[{}] {}".format(T, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items() if kk != "fold"}), flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    ref = pd.read_csv(ROOT / "experiments" / "outputs" / "033_v1ctl" / "folds.csv", index_col=0)
    folds["ens_mean"] = ref.loc[folds.index, "ens_mean"]
    folds["ens_mean_poss"] = ref.loc[folds.index, "ens_mean_poss"]
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    summary = {c: float(folds[c].mean()) for c in ("stack", "stack_nochange", "ens_mean")}
    summary["folds_stack_better"] = int((folds["stack"] > folds["ens_mean"]).sum())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(folds)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
