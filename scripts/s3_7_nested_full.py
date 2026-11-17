"""F3.7: general fully nested LOMO for a feature set and a set of models, then decoder variants.

    # train (can run as parallel processes: CPU models and the GPU model)
    .venv311/bin/python scripts/s3_7_nested_full.py --features v1ctl --models gbm,trans --no-decode
    .venv311/bin/python scripts/s3_7_nested_full.py --features v1ctl --models tcn --no-decode
    # decode and score every variant (uses the stored predictions)
    .venv311/bin/python scripts/s3_7_nested_full.py --features v1ctl --decode-only [--trials 150]

Feature sets: v1 (160 + 60 change features) | v1ctl (v1 + change + 24 control features) |
v1ctl2 (v1ctl + 12 control-v2 features: kicks, flight, carry persistence, long context) |
v1ctlq (v1ctl + 7 ball-quality features: is the ball an interpolated straight line, log 042).
Models: gbm (two-head, calibrated), trans (M3 head), tcn (M9). For each outer T: inner predictions of
each J by models trained on the five matches other than T and J; outer prediction of T by models
trained on the six. Decoder variants (each tuned on the inner predictions, plateau centre, T scored once):
  gbm         : GBM probabilities, no change signal
  gbm_m3      : GBM + M3 change
  ens_mean    : mean(GBM, TCN) + mean(M3, TCN change)
  ens_m3      : mean(GBM, TCN) + M3 change
Stored under local_data/oof/nested_<degradation>_<features>[_tcnlocal]/. --tcn-norm channel trains the local TCN
(F3.11); --reuse-from copies the GBM and M3 predictions of another store made from the same inputs and seeds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, replace
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
from src.eval.store import check_store, identity  # noqa: E402
from src.features.ballq import cached_ballq  # noqa: E402
from src.features.control import cached_control, cached_control2  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402
from src.models.transition import TransitionGBM, change_features, change_targets  # noqa: E402

TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))
VARIANTS = {"gbm": ("gbm", "none"), "gbm_m3": ("gbm", "trans"), "ens_mean": ("mean", "mean"), "ens_m3": ("mean", "trans")}


def build_features(m, key, feature_set):
    X = change_features(cached_features(m, key), m.half)
    if feature_set in ("v1ctl", "v1ctl2", "v1ctlq"):
        X = pd.concat([X, cached_control(m, key)], axis=1)
    if feature_set == "v1ctlq":
        X = pd.concat([X, cached_ballq(m, key)], axis=1)
    if feature_set == "v1ctl2":
        X = pd.concat([X, cached_control2(m, key)], axis=1)
    return X


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--features", choices=("v1", "v1ctl", "v1ctl2", "v1ctlq"), default="v1ctl")
    parser.add_argument("--models", default="gbm,trans,tcn")
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tcn-norm", choices=("group", "channel"), default="group")
    parser.add_argument("--reuse-from", type=Path, default=None, help="store whose gbm_*/trans_* predictions to copy")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tag = args.config.stem
    suffix = "" if args.tcn_norm == "group" else "_tcnlocal"
    store = ROOT / "local_data" / "oof" / "nested_{}_{}{}".format(tag, args.features, suffix)
    out = args.out or ROOT / "experiments" / "outputs" / "033_{}{}".format(args.features, suffix)
    out.mkdir(parents=True, exist_ok=True)
    sportec = load_all()
    labels = {mid: m.labels for mid, m in sportec.items()}
    halves = {mid: m.half for mid, m in sportec.items()}
    # stored predictions are reused only if made from these inputs (re-decoding ignores later code edits)
    check_store(store, identity(asdict(cfg), args.features, sportec,
                                dict({"train_variants": TRAIN_VARIANTS, "script": "s3_7 seeds 1000*(1+model)+k inner, k outer"},
                                     **({"tcn_norm": args.tcn_norm} if suffix else {}))),
                ignore=("source",) if args.decode_only else ())

    if args.reuse_from:
        for f in Path(args.reuse_from).glob("*/*.npy"):
            if f.name.startswith(("gbm_", "trans_")) and not (store / f.parent.name / f.name).is_file():
                (store / f.parent.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, store / f.parent.name / f.name)

    if not args.decode_only:
        train_X, test_X = {}, {}
        for mid, m in sportec.items():
            for key, over, seed in TRAIN_VARIANTS:
                c = replace(cfg, **over) if over else cfg
                train_X[(mid, key)] = build_features(degrade(m, c, seed), "sportec_{}_{}_{}".format(mid, tag, key), args.features)
            test_X[mid] = build_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag), args.features)

        def run(model, train_ids, test_id, seed, path):
            if path.is_file():
                return
            Xs = [train_X[(mid, k)] for mid in train_ids for k, _, _ in TRAIN_VARIANTS]
            ys = [labels[mid] for mid in train_ids for _ in TRAIN_VARIANTS]
            hs = [halves[mid] for mid in train_ids for _ in TRAIN_VARIANTS]
            if model == "gbm":
                gs = [i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]
                np.save(path, TwoHeadGBM(seed=seed).fit(Xs, ys, gs).predict_proba(test_X[test_id]).astype(np.float32))
            elif model == "trans":
                np.save(path, TransitionGBM(seed=seed).fit(Xs, [change_targets(y, h) for y, h in zip(ys, hs)]).predict(test_X[test_id]))
            else:
                p, c = TCNModel(seed=seed, norm=args.tcn_norm).fit(Xs, ys, hs).predict(test_X[test_id], halves[test_id])
                np.savez(path, proba=p, change=c)

        ext = {"gbm": "npy", "trans": "npy", "tcn": "npz"}
        for model in args.models.split(","):
            for k, T in enumerate(MATCH_IDS):
                others = [m for m in MATCH_IDS if m != T]
                (store / T).mkdir(parents=True, exist_ok=True)
                for J in others:
                    run(model, [m for m in others if m != J], J, 1000 * (1 + ["gbm", "trans", "tcn"].index(model)) + k,
                        store / T / "{}_inner_{}.{}".format(model, J, ext[model]))
                run(model, others, T, k, store / T / "{}_outer_{}.{}".format(model, T, ext[model]))
                print("[{}] {} done".format(model, T), flush=True)
    if args.no_decode:
        return 0

    def load(T, who, J):
        g = np.load(store / T / "gbm_{}_{}.npy".format(who, J))
        tr = np.load(store / T / "trans_{}_{}.npy".format(who, J))
        zp = store / T / "tcn_{}_{}.npz".format(who, J)
        z = np.load(zp) if zp.is_file() else None
        return g, tr, z

    def pack(g, tr, z, how):
        proba_kind, change_kind = VARIANTS[how]
        proba = g if proba_kind == "gbm" else (g + z["proba"]) / 2
        change = None if change_kind == "none" else (tr if change_kind == "trans" else (tr + z["change"]) / 2)
        return proba, change

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    have_tcn = all((store / T / "tcn_outer_{}.npz".format(T)).is_file() for T in MATCH_IDS)
    variants = [v for v in VARIANTS if have_tcn or VARIANTS[v][0] == "gbm"]
    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        row = {"fold": T}
        for how in variants:
            inner = {}
            for J in others:
                p, c = pack(*load(T, "inner", J), how)
                inner[J] = (p, labels[J], halves[J], c)
            st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
            st.optimize(lambda t: mean_score(suggest(t), inner, others), n_trials=args.trials)
            params = plateau(st)
            p, c = pack(*load(T, "outer", T), how)
            pred = decode(p, halves[T], params, p_change=c)
            np.save(store / T / "pred_{}_{}.npy".format(how, T), pred)
            (store / T / "params_{}.json".format(how)).write_text(json.dumps(params.to_dict()))
            r = metrics.score(labels[T], pred, halves[T])
            row[how] = r["spell_f1_mean"]
            row[how + "_poss"] = r["n_pred_spells"]
        row["poss_gt"] = r["n_gt_spells"]
        rows.append(row)
        print("[{}] {}".format(T, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items() if kk != "fold"}), flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    summary = {how: {"mean": float(folds[how].mean()), "std": float(folds[how].std(ddof=0))} for how in variants}
    folds.to_csv(out / "folds.csv", float_format="%.4f")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(folds)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
