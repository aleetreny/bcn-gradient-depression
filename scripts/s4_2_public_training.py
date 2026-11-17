"""F4.2: should the 5 public BAR-ATH minutes enter the final training? (GBM v1ctl, Metrica Sample as judge)

    .venv311/bin/python scripts/s4_2_public_training.py [--out experiments/outputs/041]

The full v1ctl GBM (7 Sportec x 3 degradation variants, seed 99) vs the same GBM with the labelled
public BAR-ATH frames added as 1 or 10 extra sources (repetitions = weight). Scored on the 3 Metrica
Sample games (same provider as BAR-ATH) with configs/decoder_v1ctl_gbm.json, also with D +/- 3 s left
out. The public score itself is in-sample once the window is trained on, so it is not a judge here.
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
from s3_11_tcn_local import without_dead_margin  # noqa: E402
from src.data.common import UNLABELLED  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/041")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = args.config.stem
    sportec = load_all()
    Xs, ys, gs = [], [], []
    for i, (mid, m) in enumerate(sportec.items()):
        for key, over, s in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            Xs.append(build_features(degrade(m, c, s), "sportec_{}_{}_{}".format(mid, tag, key), "v1ctl"))
            ys.append(m.labels)
            gs.append(i)
    atd = load_atd()
    gt = metrics.public_gt()
    y_atd = np.where(gt <= 2, gt, UNLABELLED).astype(np.int8)
    XA = build_features(atd, "atd", "v1ctl")
    print("public labelled frames", int((y_atd >= 0).sum()), flush=True)
    dec = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not k.startswith("_")})
    sample = {g: load_sample(g) for g in GAMES}
    QX = {g: build_features(sample[g], "sample_{}".format(g), "v1ctl") for g in GAMES}
    rows = []
    for reps in (0, 1, 10):
        model = TwoHeadGBM(seed=99).fit(Xs + [XA] * reps, ys + [y_atd] * reps, gs + [100] * reps)
        r = {"public_copies": reps}
        for g, ms in sample.items():
            pred = decode(model.predict_proba(QX[g]), ms.half, dec)
            r["sample_{}".format(g)] = metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"]
            r["sampleNoD_{}".format(g)] = metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"]
        r["sample_mean"] = float(np.mean([r["sample_{}".format(g)] for g in GAMES]))
        r["sampleNoD_mean"] = float(np.mean([r["sampleNoD_{}".format(g)] for g in GAMES]))
        rows.append(r)
        print(r, flush=True)
    table = pd.DataFrame(rows).set_index("public_copies")
    table.to_csv(args.out / "sample.csv", float_format="%.4f")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
