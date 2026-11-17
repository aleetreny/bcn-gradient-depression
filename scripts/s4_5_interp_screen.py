"""F4.5 screen: ATD-like piecewise-linear ball in the simulator (degrade_v5) and ball-quality features (v1ctlq).

    .venv311/bin/python scripts/s4_5_interp_screen.py [--out experiments/outputs/043]

GBM only, decoder parameters fixed (a screen). Variants (degradation, feature set):
  v3_v1ctl  : the current GBM (stored F3.7 outer predictions; reference)
  v5_v1ctl  : simulator with the interpolated ball, same features
  v5_v1ctlq : simulator with the interpolated ball + ball-quality features
Checks: Sportec outer LOMO (6 matches -> T, decoded with the fold's v1ctl GBM parameters; the test match
is degraded with the variant's own simulator) and Metrica Sample (full GBM on 7 matches, decoded with
configs/decoder_v1ctl_gbm.json; its ball is also ~90 % straight lines), with and without D +/- 3 s.
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
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402

VARIANTS = {"v5_v1ctl": ("degrade_v5", "v1ctl"), "v5_v1ctlq": ("degrade_v5", "v1ctlq")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/043")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sportec = load_all()
    ref = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl"
    dec = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not k.startswith("_")})
    sample = {g: load_sample(g) for g in GAMES}
    rows, srows = [], []
    for name, (tag, fs) in VARIANTS.items():
        cfg = load_config(ROOT / "configs" / "{}.json".format(tag))
        TX, testX = {}, {}
        for mid, m in sportec.items():
            for key, over, s in TRAIN_VARIANTS:
                c = replace(cfg, **over) if over else cfg
                TX[(mid, key)] = build_features(degrade(m, c, s), "sportec_{}_{}_{}".format(mid, tag, key), fs)
            testX[mid] = build_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag), fs)
        print(name, "features ready", flush=True)

        def fit(ids, seed):
            Xs = [TX[(mid, k)] for mid in ids for k, _, _ in TRAIN_VARIANTS]
            ys = [sportec[mid].labels for mid in ids for _ in TRAIN_VARIANTS]
            gs = [i for i, mid in enumerate(ids) for _ in TRAIN_VARIANTS]
            return TwoHeadGBM(seed=seed).fit(Xs, ys, gs)

        for k, T in enumerate(MATCH_IDS):
            m = sportec[T]
            params = DecoderParams(**json.loads((ref / T / "params_gbm.json").read_text()))
            p = fit([x for x in MATCH_IDS if x != T], k).predict_proba(testX[T])
            r = {"variant": name, "fold": T, "spellF1": metrics.score(m.labels, decode(p, m.half, params), m.half)["spell_f1_mean"]}
            if name == list(VARIANTS)[0]:
                rows.append({"variant": "v3_v1ctl", "fold": T, "spellF1": metrics.score(
                    m.labels, decode(np.load(ref / T / "gbm_outer_{}.npy".format(T)), m.half, params), m.half)["spell_f1_mean"]})
            rows.append(r)
            print(r, flush=True)
        model = fit(list(MATCH_IDS), 99)
        for g, ms in sample.items():
            pred = decode(model.predict_proba(build_features(ms, "sample_{}".format(g), fs)), ms.half, dec)
            srows.append({"variant": name, "game": g, "sample": metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"],
                          "sampleNoD": metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"]})
            print(srows[-1], flush=True)
    folds = pd.DataFrame(rows).pivot(index="fold", columns="variant", values="spellF1")
    samp = pd.DataFrame(srows).pivot(index="game", columns="variant")
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    samp.to_csv(args.out / "sample.csv", float_format="%.4f")
    summary = {"lomo": folds.mean().round(4).to_dict(),
               "folds_better_than_v3": {v: int((folds[v] > folds["v3_v1ctl"]).sum()) for v in VARIANTS},
               "sample": samp["sample"].mean().round(4).to_dict(), "sampleNoD": samp["sampleNoD"].mean().round(4).to_dict(),
               "sample_reference_v3_v1ctl": 0.7995, "sampleNoD_reference_v3_v1ctl": 0.8414}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(folds)
        print(samp)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
