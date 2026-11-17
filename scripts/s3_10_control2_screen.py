"""F3.10 screen: control-v2 features (kicks, flight, carry persistence, long context) for the GBM.

    .venv311/bin/python scripts/s3_10_control2_screen.py [--out experiments/outputs/036]

GBM-only, decoder parameters fixed (a screen), paired with the v1ctl GBM of F3.7/F3.8:
  1. Sportec outer LOMO: v1ctl2 GBM on six matches -> T, decoded with the fold's v1ctl GBM parameters,
     vs the stored v1ctl GBM outer prediction (same parameters);
  2. Metrica Sample: full v1ctl2 GBM decoded with configs/decoder_v1ctl_gbm.json (v1ctl: 0.812 / 0.749 / 0.837);
  3. BAR-ATH public (v1ctl GBM: 0.584).
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
from games import GAMES as GAME_INFO, anchor_video_frame  # noqa: E402
from src.data.common import DEAD  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402

REF_SAMPLE = {1: 0.8119, 2: 0.7490, 3: 0.8374}
REF_PUBLIC = 0.5838


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/036")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = args.config.stem
    store = ROOT / "local_data" / "oof" / "nested_{}_v1ctl".format(tag)
    sportec = load_all()
    SX = {}
    for mid, m in sportec.items():
        for key, over, s in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            SX[(mid, key)] = build_features(degrade(m, c, s), "sportec_{}_{}_{}".format(mid, tag, key), "v1ctl2")
    test_X = {mid: build_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag), "v1ctl2") for mid, m in sportec.items()}

    def fit(ids, seed):
        return TwoHeadGBM(seed=seed).fit([SX[(mid, k)] for mid in ids for k, _, _ in TRAIN_VARIANTS],
                                         [sportec[mid].labels for mid in ids for _ in TRAIN_VARIANTS],
                                         [i for i, mid in enumerate(ids) for _ in TRAIN_VARIANTS])

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        model = fit(others, k)
        p_new = model.predict_proba(test_X[T])
        np.save(store / T / "gbmctl2_outer_{}.npy".format(T), p_new.astype(np.float32))
        if k == 0:
            pd.DataFrame(model.importance(25)).to_csv(args.out / "importance_fold0.csv")
        p_old = np.load(store / T / "gbm_outer_{}.npy".format(T))
        params = DecoderParams(**json.loads((store / T / "params_gbm.json").read_text()))
        m = sportec[T]
        rows.append({"fold": T, "v1ctl": metrics.score(m.labels, decode(p_old, m.half, params), m.half)["spell_f1_mean"],
                     "v1ctl2": metrics.score(m.labels, decode(p_new, m.half, params), m.half)["spell_f1_mean"]})
        print(rows[-1], flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    full = fit(list(MATCH_IDS), 99)
    dec = DecoderParams(**{kk: v for kk, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not kk.startswith("_")})
    sample = {}
    for g in GAMES:
        ms = load_sample(g)
        sample[g] = metrics.score(ms.labels, decode(full.predict_proba(build_features(ms, "sample_{}".format(g), "v1ctl2")), ms.half, dec),
                                  ms.half)["spell_f1_mean"]
    atd = load_atd()
    pred = decode(full.predict_proba(build_features(atd, "atd", "v1ctl2")), np.where(atd.graded, atd.half, 0), dec)
    pred[int(np.flatnonzero(atd.half == 2)[0]):anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])] = DEAD
    pred[~atd.graded] = DEAD
    pub = metrics.score_public(pred, bootstrap_n=1000)
    summary = {"v1ctl_mean": float(folds["v1ctl"].mean()), "v1ctl2_mean": float(folds["v1ctl2"].mean()),
               "folds_better": int((folds["v1ctl2"] > folds["v1ctl"]).sum()),
               "sample": sample, "sample_mean": float(np.mean(list(sample.values()))), "sample_ref_mean": float(np.mean(list(REF_SAMPLE.values()))),
               "public": pub["spell_f1_mean"], "public_ci": [pub["spell_bootstrap_low"], pub["spell_bootstrap_high"]], "public_ref": REF_PUBLIC}
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(folds)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
