"""Tune the final decoder (with optional time-dependent switch cost) on the 7 outer nested predictions.

    .venv311/bin/python scripts/s3_1b_decoder_final.py --save configs/decoder_m3.json [--change trans] [--proba gbm|mean]
                                                       [--config configs/degrade_v3.json] [--trials 150]

Reads local_data/oof/nested_<tag>/<T>/outer_<T>.npy (GBM), optional TCN outer probabilities
(tcn_outer_<T>.npz) and optional change probabilities (trans_outer_<T>.npy, or the TCN change head),
tunes the decoder on all 7 matches (Optuna TPE, plateau centre of the top 10 %) and writes the config
used to build a submission for BAR-ATH. This is the 'final fit' step; it is never used to score Sportec.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s3_1_transition_head import mean_score, plateau, suggest  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402


def load_inputs(store, T, proba, change):
    p_gbm = np.load(store / T / "outer_{}.npy".format(T))
    z = np.load(store / T / "tcn_outer_{}.npz".format(T)) if (proba != "gbm" or change in ("tcn", "mean")) else None
    p = {"gbm": p_gbm, "tcn": z["proba"] if z is not None else None,
         "mean": (p_gbm + z["proba"]) / 2 if z is not None else None}[proba]
    c = None
    if change == "trans":
        c = np.load(store / T / "trans_outer_{}.npy".format(T))
    elif change == "tcn":
        c = z["change"]
    elif change == "mean":
        c = (np.load(store / T / "trans_outer_{}.npy".format(T)) + z["change"]) / 2
    return p, c


def main() -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--proba", choices=("gbm", "tcn", "mean"), default="gbm")
    parser.add_argument("--change", choices=("none", "trans", "tcn", "mean"), default="trans")
    parser.add_argument("--trials", type=int, default=150)
    args = parser.parse_args()
    store = ROOT / "local_data" / "oof" / "nested_{}".format(args.config.stem)
    sportec = load_all()
    data = {}
    for T in MATCH_IDS:
        p, c = load_inputs(store, T, args.proba, args.change)
        data[T] = (p, sportec[T].labels, sportec[T].half, c)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
    st.optimize(lambda t: mean_score(suggest(t), data, list(MATCH_IDS)), n_trials=args.trials)
    params = plateau(st)
    value = mean_score(params, data, list(MATCH_IDS))
    out = {**params.to_dict(), "_provenance": "s3_1b: plateau centre on the 7 nested outer predictions; proba={}, change={}, "
           "degradation {}; mean spellF1 on the 7 (in-sample, not a score) {:.4f}".format(args.proba, args.change, args.config.stem, value)}
    args.save.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
