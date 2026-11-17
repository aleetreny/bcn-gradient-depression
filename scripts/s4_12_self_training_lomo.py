"""F4.8b: transductive self-training judged on Sportec (second judge for F4.8).

    .venv311/bin/python scripts/s4_12_self_training_lomo.py [--out experiments/outputs/048]

For each outer match T (its labels only score): the stored v1ctl GBM outer prediction of T, decoded with
the fold's v1ctl GBM parameters, gives pseudo-labels on T with the setting that helped on Metrica Sample
(F4.8: tau 0.9, 25-frame margin around predicted changes, team and dead ball, one round). The GBM is
retrained on the six other matches + pseudo-labelled T and T is decoded again with the same parameters.
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
from s3_7_nested_full import TRAIN_VARIANTS, build_features  # noqa: E402
from s4_8_self_training import pseudo_labels  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/048")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/degrade_v3.json")
    sportec = load_all()
    ref = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl"
    TX, testX = {}, {}
    for mid, m in sportec.items():
        for key, over, s in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            TX[(mid, key)] = build_features(degrade(m, c, s), "sportec_{}_degrade_v3_{}".format(mid, key), "v1ctl")
        testX[mid] = build_features(degrade(m, cfg, 0), "sportec_{}_degrade_v3_s0".format(mid), "v1ctl")
    rows = []
    for k, T in enumerate(MATCH_IDS):
        m = sportec[T]
        others = [x for x in MATCH_IDS if x != T]
        params = DecoderParams(**json.loads((ref / T / "params_gbm.json").read_text()))
        p0 = np.load(ref / T / "gbm_outer_{}.npy".format(T))
        pred0 = decode(p0, m.half, params)
        pl = pseudo_labels(p0, pred0, m.half, 0.9, 25, True)
        Xs = [TX[(x, key)] for x in others for key, _, _ in TRAIN_VARIANTS] + [testX[T]]
        ys = [sportec[x].labels for x in others for _ in TRAIN_VARIANTS] + [pl]
        gs = [i for i, x in enumerate(others) for _ in TRAIN_VARIANTS] + [100]
        p1 = TwoHeadGBM(seed=k).fit(Xs, ys, gs).predict_proba(testX[T])
        row = {"fold": T, "base": metrics.score(m.labels, pred0, m.half)["spell_f1_mean"],
               "self_trained": metrics.score(m.labels, decode(p1, m.half, params), m.half)["spell_f1_mean"],
               "pseudo_share": float(np.mean(pl[m.half > 0] >= 0)), "pseudo_acc": float(np.mean(pl[pl >= 0] == m.labels[pl >= 0]))}
        rows.append(row)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    summary = {"base": float(folds.base.mean()), "self_trained": float(folds.self_trained.mean()),
               "folds_better": int((folds.self_trained > folds.base).sum())}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
