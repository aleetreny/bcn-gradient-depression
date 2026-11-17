"""F3.5: transfer check on Metrica Sample (same provider as BAR-ATH, 3 full games, event-derived labels).

    .venv311/bin/python scripts/s3_5_sample_transfer.py [--config configs/degrade_v3.json] [--out experiments/outputs/031]

Trains once the full models on the 7 degraded Sportec matches (two-head GBM, M3 transition head, TCN on
MPS) and saves them under local_data/models/full_<tag>/ for reuse. Then predicts Metrica Sample games
1-3 and scores the candidates with their own decoder configs:
  baseline (smooth50) | v1.1 (GBM, decoder_v1_1) | v1.2 (GBM + M3, decoder_m3) | v2 (mean GBM/TCN + mean change, decoder_v2).
Caveats: Sample labels are DERIVED from events by our rules (S0.4), and in the Sample the ball is missing
when it is dead, which makes D easier than in the ATD. It is a second, larger transfer judge for the
ranking of candidates, not an absolute score.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.baseline import predict as baseline  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402
from src.models.transition import TransitionGBM, change_features, change_targets  # noqa: E402

TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))


def dec(path):
    return DecoderParams(**{k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")})


def full_models(cfg, tag, seed=99):
    root = ROOT / "local_data" / "models" / "full_{}".format(tag)
    path = root / "models.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    sportec = load_all()
    Xs, Xcs, ys, hs, gs = [], [], [], [], []
    for i, (mid, m) in enumerate(sportec.items()):
        for key, over, s in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            X = cached_features(degrade(m, c, s), "sportec_{}_{}_{}".format(mid, tag, key))
            Xs.append(X)
            Xcs.append(change_features(X, m.half))
            ys.append(m.labels)
            hs.append(m.half)
            gs.append(i)
    models = {"gbm": TwoHeadGBM(seed=seed).fit(Xs, ys, gs),
              "trans": TransitionGBM(seed=seed).fit(Xcs, [change_targets(y, h) for y, h in zip(ys, hs)]),
              "tcn": TCNModel(seed=seed).fit(Xcs, ys, hs)}
    root.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(models))
    return models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/031")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    models = full_models(cfg, args.config.stem)
    d11, dm3, dv2 = dec(ROOT / "configs/decoder_v1_1.json"), dec(ROOT / "configs/decoder_m3.json"), dec(ROOT / "configs/decoder_v2.json")

    rows = []
    for g in GAMES:
        m = load_sample(g)
        X = cached_features(m, "sample_{}".format(g))
        Xc = change_features(X, m.half)
        pg = models["gbm"].predict_proba(X)
        pt = models["trans"].predict(Xc)
        pn, cn = models["tcn"].predict(Xc, m.half)
        preds = {"baseline": baseline(m, 50), "v1.1": decode(pg, m.half, d11), "v1.2": decode(pg, m.half, dm3, p_change=pt),
                 "v2": decode((pg + pn) / 2, m.half, dv2, p_change=(pt + cn) / 2)}
        for name, p in preds.items():
            r = metrics.score(m.labels, p, m.half)
            rows.append({"game": g, "system": name, "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                         "team_acc": r["team_acc_gt_alive"], "D_f1": r["per_class"]["D"]["f1"], "poss_pred": r["n_pred_spells"],
                         "poss_gt": r["n_gt_spells"]})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    table = df.groupby("system", sort=False)[["spellF1", "macroF1", "team_acc", "D_f1", "poss_pred", "poss_gt"]].mean()
    df.to_csv(args.out / "per_game.csv", index=False, float_format="%.4f")
    table.to_csv(args.out / "summary.csv", float_format="%.4f")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.pivot(index="system", columns="game", values="spellF1"))
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
