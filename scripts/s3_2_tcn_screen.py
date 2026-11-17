"""F3.2 (M9) screen: TCN per-frame model vs the GBM, outer LOMO only (decoder parameters fixed per fold).

    .venv311/bin/python scripts/s3_2_tcn_screen.py [--config configs/degrade_v3.json] [--out experiments/outputs/028]

For each outer match T, a TCN (src/models/tcn.py) trained on the six other matches (3 degradation
variants each) predicts T (seed 0). Compared on the same fold with the GBM outer probabilities of
S2.4b and with their average, all decoded with v1.1's per-fold decoder parameters
(local_data/oof/nested_<tag>/<T>/params.json, tuned on GBM inner predictions -- a handicap for the
TCN, so this is a screen: a full nested run follows if it is promising). Also frame-level metrics.
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
from src.data.common import DEAD, HOME  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.gbm import smooth_argmax  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402
from src.models.transition import change_features, change_targets  # noqa: E402

TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))


def frame_metrics(proba, y):
    from sklearn.metrics import roc_auc_score
    lab = y >= 0
    alive = lab & (y != DEAD)
    pa = 1 - proba[:, 2]
    ph = proba[:, 0] / np.maximum(pa, 1e-9)
    return {"alive_auc": float(roc_auc_score((y[lab] != DEAD).astype(int), pa[lab])),
            "team_acc": float(np.mean((ph[alive] > 0.5) == (y[alive] == HOME))),
            "logloss": float(-np.mean(np.log(np.clip(proba[lab, y[lab]], 1e-6, 1))))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--name", default="", help="variant name (stored as tcn_outer_<name>_<T>.npz); empty = base TCN")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--max-dilation", type=int, default=128)
    parser.add_argument("--change-weight", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/028")
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
            train_X[(mid, key)] = change_features(cached_features(degrade(m, c, seed), "sportec_{}_{}_{}".format(mid, tag, key)), m.half)
        test_X[mid] = change_features(cached_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag)), m.half)
        labels[mid], halves[mid] = m.labels, m.half

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        path = store / T / ("tcn_outer_{}.npz".format(T) if not args.name else "tcn_outer_{}_{}.npz".format(args.name, T))
        if path.is_file():
            z = np.load(path)
            p_tcn, c_tcn = z["proba"], z["change"]
        else:
            dil = tuple(2 ** i for i in range(int(np.log2(args.max_dilation)) + 1))
            model = TCNModel(seed=k, epochs=args.epochs, channels=args.channels, dilations=dil, change_weight=args.change_weight,
                             label_smoothing=args.label_smoothing).fit([train_X[(m, key)] for m in others for key, _, _ in TRAIN_VARIANTS],
                                                            [labels[m] for m in others for _ in TRAIN_VARIANTS],
                                                            [halves[m] for m in others for _ in TRAIN_VARIANTS])
            p_tcn, c_tcn = model.predict(test_X[T], halves[T])
            np.savez(path, proba=p_tcn, change=c_tcn)
        p_gbm = np.load(store / T / "outer_{}.npy".format(T))
        params = DecoderParams(**json.loads((store / T / "params.json").read_text()))
        pb_path = store / T / "params_B_mean_proba_mean_change.json"
        trans = np.load(store / T / "trans_outer_{}.npy".format(T))
        y, h = labels[T], halves[T]
        row = {"fold": T}
        if pb_path.is_file():  # ensemble as in F3.3 B, decoder parameters of that fold
            pb = DecoderParams(**json.loads(pb_path.read_text()))
            row["ensB_dec"] = metrics.score(y, decode((p_gbm + p_tcn) / 2, h, pb, p_change=(trans + c_tcn) / 2), h)["spell_f1_mean"]
        for name, p in (("gbm", p_gbm), ("tcn", p_tcn), ("mean", (p_gbm + p_tcn) / 2)):
            row["{}_dec".format(name)] = metrics.score(y, decode(p, h, params), h)["spell_f1_mean"]
            row["{}_w12".format(name)] = metrics.score(y, smooth_argmax(p, h, 12), h)["spell_f1_mean"]
            for kk, v in frame_metrics(p, y).items():
                row["{}_{}".format(name, kk)] = v
        from sklearn.metrics import average_precision_score
        yc = change_targets(y, h)
        row["tcn_change_ap"] = float(average_precision_score(yc[yc >= 0], c_tcn[yc >= 0]))
        rows.append(row)
        print("[{}] {}".format(T, {kk: round(v, 3) for kk, v in row.items() if kk != "fold"}), flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    means = folds.mean().round(4).to_dict()
    summary = {"means": means,
               "folds_tcn_better_than_gbm_dec": int((folds["tcn_dec"] > folds["gbm_dec"]).sum()),
               "folds_mean_better_than_gbm_dec": int((folds["mean_dec"] > folds["gbm_dec"]).sum())}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(folds[[c for c in folds.columns if c.endswith(("_dec", "_w12", "team_acc", "alive_auc"))]])
    print("variant:", args.name or "base", vars(args))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
