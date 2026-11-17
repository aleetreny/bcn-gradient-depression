"""F3.4 (M6): short D as a separator, post-processing of v2, evaluated with the fully nested LOMO.

    .venv311/bin/python scripts/s3_4_short_dead.py [--config configs/degrade_v3.json] [--out experiments/outputs/030]

For each outer T: the inner predictions (F3.3 variant B inputs: mean of GBM and TCN probabilities,
mean of M3 and TCN change probabilities) are decoded with that fold's decoder parameters; the
post-processing parameters (threshold, length) are chosen by grid search on the six inner matches and
applied once to T's v2 prediction. Paired with v2 (F3.3 B).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.postprocess import short_dead_separator  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402

GRID = {"threshold": (0.3, 0.4, 0.5, 0.6, 0.7, 0.8), "length": (2, 4, 8, 12, 20)}
NAME = "B_mean_proba_mean_change"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/030")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    store = ROOT / "local_data" / "oof" / "nested_{}".format(args.config.stem)
    sp = load_all()
    rows = []
    for T in MATCH_IDS:
        others = [m for m in MATCH_IDS if m != T]
        params = DecoderParams(**json.loads((store / T / "params_{}.json".format(NAME)).read_text()))
        inner = {}
        for J in others:
            z = np.load(store / T / "tcn_inner_{}.npz".format(J))
            proba = (np.load(store / T / "inner_{}.npy".format(J)) + z["proba"]) / 2
            pc = (np.load(store / T / "trans_inner_{}.npy".format(J)) + z["change"]) / 2
            inner[J] = (decode(proba, sp[J].half, params, p_change=pc), pc)
        best, best_v = (1.0, 0), -1
        base_inner = np.mean([metrics.score(sp[J].labels, inner[J][0], sp[J].half)["spell_f1_mean"] for J in others])
        for thr, L in itertools.product(*GRID.values()):
            v = np.mean([metrics.score(sp[J].labels, short_dead_separator(inner[J][0], inner[J][1], sp[J].half, thr, L),
                                       sp[J].half)["spell_f1_mean"] for J in others])
            if v > best_v:
                best, best_v = (thr, L), v
        zo = np.load(store / T / "tcn_outer_{}.npz".format(T))
        pc_out = (np.load(store / T / "trans_outer_{}.npy".format(T)) + zo["change"]) / 2
        pred = np.load(store / T / "pred_{}_{}.npy".format(NAME, T))
        post = short_dead_separator(pred, pc_out, sp[T].half, *best) if best_v > base_inner else pred
        r0 = metrics.score(sp[T].labels, pred, sp[T].half)
        r1 = metrics.score(sp[T].labels, post, sp[T].half)
        rows.append({"fold": T, "v2": r0["spell_f1_mean"], "m6": r1["spell_f1_mean"], "inner_gain": best_v - base_inner,
                     "threshold": best[0], "length": best[1], "poss_v2": r0["n_pred_spells"], "poss_m6": r1["n_pred_spells"],
                     "poss_gt": r0["n_gt_spells"], "inserted_frames": int(np.sum((post == 2) & (pred != 2)))})
        print("[{}] {}".format(T, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "fold"}), flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    d = folds["m6"] - folds["v2"]
    summary = {"v2": float(folds["v2"].mean()), "m6": float(folds["m6"].mean()), "delta": float(d.mean()),
               "folds_improved": int((d > 0).sum()), "folds_worse": int((d < 0).sum())}
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(folds)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
