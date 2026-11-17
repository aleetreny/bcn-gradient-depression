"""F3.6 screen: does the GBM gain from the control features? Outer LOMO + Metrica Sample transfer.

    .venv311/bin/python scripts/s3_6_control_screen.py [--config configs/degrade_v3.json] [--out experiments/outputs/032]

For each outer T, a two-head GBM on v1 + control features (src/features/control.py), trained on the six
other matches, predicts T; compared with the stored v1 GBM outer prediction (S2.4b), both decoded with
the fold's v1.1 decoder parameters and, inside the F3.3 B ensemble (with the stored TCN and M3), with the
fold's B parameters. Then a full GBM on the 7 matches with control features is applied to Metrica Sample
and decoded with configs/decoder_v1_1.json (paired with v1.1 there: 0.770). A screen only: decoder
parameters are not re-tuned for the new model.
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
from s3_2_tcn_screen import frame_metrics  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.control import cached_control  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.gbm import TwoHeadGBM, smooth_argmax  # noqa: E402

TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))


def feats(m, key):
    return pd.concat([cached_features(m, key), cached_control(m, key)], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/032")
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
            train_X[(mid, key)] = feats(degrade(m, c, seed), "sportec_{}_{}_{}".format(mid, tag, key))
        test_X[mid] = feats(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag))
        labels[mid], halves[mid] = m.labels, m.half

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        path = store / T / "gbmctl_outer_{}.npy".format(T)
        if path.is_file():
            p_ctl = np.load(path)
        else:
            model = TwoHeadGBM(seed=k).fit([train_X[(m, key)] for m in others for key, _, _ in TRAIN_VARIANTS],
                                           [labels[m] for m in others for _ in TRAIN_VARIANTS],
                                           [i for i, m in enumerate(others) for _ in TRAIN_VARIANTS])
            p_ctl = model.predict_proba(test_X[T])
            np.save(path, p_ctl.astype(np.float32))
            if k == 0:
                imp = model.importance(25)
                pd.DataFrame(imp).to_csv(args.out / "importance_fold0.csv")
        p_v1 = np.load(store / T / "outer_{}.npy".format(T))
        y, h = labels[T], halves[T]
        p11 = DecoderParams(**json.loads((store / T / "params.json").read_text()))
        pb = DecoderParams(**json.loads((store / T / "params_B_mean_proba_mean_change.json").read_text()))
        zt = np.load(store / T / "tcn_outer_{}.npz".format(T))
        pc = (np.load(store / T / "trans_outer_{}.npy".format(T)) + zt["change"]) / 2
        row = {"fold": T}
        for name, p in (("v1", p_v1), ("ctl", p_ctl)):
            row[name + "_dec"] = metrics.score(y, decode(p, h, p11), h)["spell_f1_mean"]
            row[name + "_w12"] = metrics.score(y, smooth_argmax(p, h, 12), h)["spell_f1_mean"]
            row[name + "_ensB"] = metrics.score(y, decode((p + zt["proba"]) / 2, h, pb, p_change=pc), h)["spell_f1_mean"]
            for kk, v in frame_metrics(p, y).items():
                row[name + "_" + kk] = v
        rows.append(row)
        print("[{}] {}".format(T, {kk: round(v, 4) for kk, v in row.items() if kk != "fold"}), flush=True)
    folds = pd.DataFrame(rows).set_index("fold")

    # transfer: full GBM with control features on Metrica Sample
    full = TwoHeadGBM(seed=99).fit([train_X[(m, key)] for m in MATCH_IDS for key, _, _ in TRAIN_VARIANTS],
                                   [labels[m] for m in MATCH_IDS for _ in TRAIN_VARIANTS],
                                   [i for i, m in enumerate(MATCH_IDS) for _ in TRAIN_VARIANTS])
    d11 = DecoderParams(**{kk: v for kk, v in json.loads((ROOT / "configs/decoder_v1_1.json").read_text()).items() if not kk.startswith("_")})
    sample = {}
    for g in GAMES:
        ms = load_sample(g)
        sample[g] = metrics.score(ms.labels, decode(full.predict_proba(feats(ms, "sample_{}".format(g))), ms.half, d11), ms.half)["spell_f1_mean"]

    summary = {"means": folds.mean().round(4).to_dict(),
               "folds_ctl_better_dec": int((folds["ctl_dec"] > folds["v1_dec"]).sum()),
               "folds_ctl_better_ensB": int((folds["ctl_ensB"] > folds["v1_ensB"]).sum()),
               "sample_v1_1_with_control": sample, "sample_v1_1_with_control_mean": float(np.mean(list(sample.values()))),
               "sample_v1_1_reference": 0.770}
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(folds[[c for c in folds.columns if c.endswith(("_dec", "_w12", "_ensB", "team_acc"))]])
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
