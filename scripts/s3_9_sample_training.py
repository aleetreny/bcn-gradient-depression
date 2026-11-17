"""F3.9 (M7) screen: Metrica Sample games as extra TEAM-head training data for the GBM (features v1ctl).

    .venv311/bin/python scripts/s3_9_sample_training.py [--config configs/degrade_v3.json] [--out experiments/outputs/035]

The Sample is the same provider as BAR-ATH but its missing ball marks dead time, so it only trains the
team head (TwoHeadGBM team_only); the alive head stays Sportec-only. Three checks for the GBM-only
system (decoder parameters fixed, a screen):
  1. Sportec outer LOMO: 6 Sportec matches (+3 Sample games) -> T, decoded with the fold's v1ctl GBM
     parameters, vs the stored v1ctl GBM outer prediction;
  2. Sample leave-one-game-out: 7 Sportec + the other 2 Sample games -> game g, decoded with
     configs/decoder_v1ctl_gbm.json, vs the full Sportec-only GBM (F3.8: 0.812 / 0.749 / 0.837);
  3. BAR-ATH public: 7 Sportec + 3 Sample -> public score (Sportec-only GBM: 0.584).
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

REF_SAMPLE = {1: 0.8121, 2: 0.7492, 3: 0.8370}  # F3.8, Sportec-only v1ctl GBM with decoder_v1ctl_gbm
REF_PUBLIC = 0.5842


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/035")
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
            SX[(mid, key)] = build_features(degrade(m, c, s), "sportec_{}_{}_{}".format(mid, tag, key), "v1ctl")
    test_X = {mid: build_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag), "v1ctl") for mid, m in sportec.items()}
    sample = {g: load_sample(g) for g in GAMES}
    QX = {g: build_features(sample[g], "sample_{}".format(g), "v1ctl") for g in GAMES}

    def fit(sportec_ids, sample_ids, seed):
        Xs = [SX[(mid, k)] for mid in sportec_ids for k, _, _ in TRAIN_VARIANTS] + [QX[g] for g in sample_ids]
        ys = [sportec[mid].labels for mid in sportec_ids for _ in TRAIN_VARIANTS] + [sample[g].labels for g in sample_ids]
        gs = [i for i, mid in enumerate(sportec_ids) for _ in TRAIN_VARIANTS] + [100 + g for g in sample_ids]
        to = [False] * (len(sportec_ids) * len(TRAIN_VARIANTS)) + [True] * len(sample_ids)
        return TwoHeadGBM(seed=seed).fit(Xs, ys, gs, team_only=to)

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        p_new = fit(others, list(GAMES), k).predict_proba(test_X[T])
        p_old = np.load(store / T / "gbm_outer_{}.npy".format(T))
        params = DecoderParams(**json.loads((store / T / "params_gbm.json").read_text()))
        m = sportec[T]
        rows.append({"fold": T, "sportec_only": metrics.score(m.labels, decode(p_old, m.half, params), m.half)["spell_f1_mean"],
                     "with_sample": metrics.score(m.labels, decode(p_new, m.half, params), m.half)["spell_f1_mean"]})
        print(rows[-1], flush=True)
    folds = pd.DataFrame(rows).set_index("fold")

    dec = DecoderParams(**{kk: v for kk, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not kk.startswith("_")})
    logo = {}
    for g in GAMES:
        model = fit(list(MATCH_IDS), [h for h in GAMES if h != g], 50 + g)
        ms = sample[g]
        logo[g] = metrics.score(ms.labels, decode(model.predict_proba(QX[g]), ms.half, dec), ms.half)["spell_f1_mean"]
        print("sample game", g, logo[g], "ref", REF_SAMPLE[g], flush=True)

    model = fit(list(MATCH_IDS), list(GAMES), 99)
    atd = load_atd()
    pred = decode(model.predict_proba(build_features(atd, "atd", "v1ctl")), np.where(atd.graded, atd.half, 0), dec)
    pred[int(np.flatnonzero(atd.half == 2)[0]):anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])] = DEAD
    pred[~atd.graded] = DEAD
    pub = metrics.score_public(pred, bootstrap_n=1000)

    summary = {"sportec_only_mean": float(folds["sportec_only"].mean()), "with_sample_mean": float(folds["with_sample"].mean()),
               "sportec_folds_better": int((folds["with_sample"] > folds["sportec_only"]).sum()),
               "sample_logo": logo, "sample_logo_mean": float(np.mean(list(logo.values()))),
               "sample_reference_mean": float(np.mean(list(REF_SAMPLE.values()))),
               "public": pub["spell_f1_mean"], "public_ci": [pub["spell_bootstrap_low"], pub["spell_bootstrap_high"]],
               "public_reference": REF_PUBLIC}
    folds.to_csv(args.out / "sportec_folds.csv", float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(folds)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
