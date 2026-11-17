"""F4.8: self-training on the target provider's own unlabelled tracking (domain adaptation), judged on Metrica Sample.

    .venv311/bin/python scripts/s4_8_self_training.py [--out experiments/outputs/046]

The three Sample games play the role of BAR-ATH: their labels are NOT used to adapt, only to score.
Round 0 = the full v1ctl GBM trained on the 7 degraded Sportec matches (seed 99), decoded with
configs/decoder_v1ctl_gbm.json. Pseudo-labels on each target game: the decoded team where the model's
P(predicted team | alive) >= tau and the frame is more than `margin` frames from any predicted change;
everything else unlabelled. The GBM is retrained on Sportec + the pseudo-labelled games (repeated `w`
times; team head only, or both heads with the decoded D as well) and the targets are decoded again.
Variants: tau, w, heads, rounds.
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
from src.data.common import DEAD, HOME, UNLABELLED  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402


def pseudo_labels(p, pred, half, tau, margin, with_dead):
    pa = 1 - p[:, 2]
    ph = p[:, 0] / np.maximum(pa, 1e-9)
    conf_team = np.where(pred == HOME, ph, 1 - ph)
    change = np.flatnonzero((pred[1:] != pred[:-1]) | (half[1:] != half[:-1])) + 1
    near = np.zeros(len(pred), bool)
    for c in change:
        near[max(0, c - margin):c + margin] = True
    y = np.full(len(pred), UNLABELLED, np.int8)
    team = np.isin(pred, (0, 1)) & (conf_team >= tau) & (pa >= 0.5) & ~near & (half > 0)
    y[team] = pred[team]
    if with_dead:
        dead = (pred == DEAD) & (p[:, 2] >= tau) & ~near & (half > 0)
        y[dead] = DEAD
    return y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/046")
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
    dec = DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not k.startswith("_")})
    games = {g: load_sample(g) for g in GAMES}
    QX = {g: build_features(ms, "sample_{}".format(g), "v1ctl") for g, ms in games.items()}

    def evaluate(model, name):
        r = {"variant": name}
        preds, probas = {}, {}
        for g, ms in games.items():
            p = model.predict_proba(QX[g])
            pred = decode(p, ms.half, dec)
            preds[g], probas[g] = pred, p
            r["s{}".format(g)] = metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"]
            r["noD{}".format(g)] = metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"]
            alive = np.isin(ms.labels, (0, 1))
            ph = p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9)
            r["acc{}".format(g)] = float(np.mean((ph[alive] > 0.5) == (ms.labels[alive] == HOME)))
        r["sample"] = float(np.mean([r["s{}".format(g)] for g in GAMES]))
        r["sampleNoD"] = float(np.mean([r["noD{}".format(g)] for g in GAMES]))
        r["team_acc"] = float(np.mean([r["acc{}".format(g)] for g in GAMES]))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        return r, preds, probas

    base = TwoHeadGBM(seed=99).fit(Xs, ys, gs)
    rows = []
    r0, pred0, prob0 = evaluate(base, "round0")
    rows.append(r0)
    for tau, w, with_dead, margin in ((0.9, 1, False, 25), (0.95, 1, False, 25), (0.9, 3, False, 25), (0.9, 1, True, 25), (0.8, 1, False, 50)):
        preds, probas = pred0, prob0
        for rnd in (1, 2):
            pl = {g: pseudo_labels(probas[g], preds[g], games[g].half, tau, margin, with_dead) for g in GAMES}
            extra_X = [QX[g] for g in GAMES for _ in range(w)]
            extra_y = [pl[g] for g in GAMES for _ in range(w)]
            extra_g = [100 + g for g in GAMES for _ in range(w)]
            model = TwoHeadGBM(seed=99).fit(Xs + extra_X, ys + extra_y, gs + extra_g,
                                            team_only=[False] * len(Xs) + [not with_dead] * len(extra_X))
            name = "tau{}_w{}_{}_m{}_r{}".format(tau, w, "both" if with_dead else "team", margin, rnd)
            r, preds, probas = evaluate(model, name)
            r["pseudo_share"] = float(np.mean([np.mean(pl[g][games[g].half > 0] >= 0) for g in GAMES]))
            r["pseudo_acc"] = float(np.mean([np.mean(pl[g][pl[g] >= 0] == games[g].labels[pl[g] >= 0]) for g in GAMES]))
            rows.append(r)
            if rnd == 1 and tau != 0.9:
                break
    table = pd.DataFrame(rows).set_index("variant")
    table.to_csv(args.out / "self_training.csv", float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(table[["sample", "sampleNoD", "team_acc", "pseudo_share", "pseudo_acc"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
