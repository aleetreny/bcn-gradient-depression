"""F3.13 (M8) screen: SkillCorner Open Data (20 matches, broadcast tracking) as extra GBM training data.

    .venv311/bin/python scripts/s3_13_skillcorner.py [--out experiments/outputs/038]

SkillCorner labels are SkillCorner's own (phases of play + per-frame possession, src/data/skillcorner.py),
and its tracking is a broadcast subset (~12 visible players, ball 59 %). Two ways to use it with the
v1ctl GBM:
  team : SkillCorner trains only the team head (TwoHeadGBM team_only), the alive head stays Sportec;
  both : SkillCorner trains both heads.
Checks (decoder parameters fixed, a screen):
  1. Sportec outer LOMO: 6 Sportec (3 degradation variants) + 20 SkillCorner -> T, decoded with the
     fold's v1ctl GBM parameters, vs the stored Sportec-only v1ctl GBM outer prediction;
  2. Metrica Sample: 7 Sportec + 20 SkillCorner -> each game, configs/decoder_v1ctl_gbm.json (also with
     D +/- 3 s left out), vs the Sportec-only full GBM of F3.8;
  3. BAR-ATH public window and full-match plausibility.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from s2_3_frame_models import plausibility  # noqa: E402
from s3_7_nested_full import TRAIN_VARIANTS, build_features  # noqa: E402
from s3_11_tcn_local import without_dead_margin  # noqa: E402
from games import GAMES as GAME_INFO, anchor_video_frame  # noqa: E402
from src.data import skillcorner  # noqa: E402
from src.data.common import DEAD  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402

MODES = ("team", "both")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--extrapolated", action="store_true", help="keep SkillCorner's extrapolated positions")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/038")
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
    det = not args.extrapolated
    sc_lab, SCX = {}, {}
    for mid in skillcorner.MATCH_IDS:
        m = skillcorner.load(mid, detected_only=det)
        SCX[mid] = build_features(m, "skillcorner_{}_{}".format(mid, "det" if det else "ext"), "v1ctl")
        sc_lab[mid] = m.labels
        print("skillcorner", mid, "features", SCX[mid].shape, flush=True)

    def fit(sportec_ids, mode, seed):
        Xs = [SX[(mid, k)] for mid in sportec_ids for k, _, _ in TRAIN_VARIANTS]
        ys = [sportec[mid].labels for mid in sportec_ids for _ in TRAIN_VARIANTS]
        gs = [i for i, mid in enumerate(sportec_ids) for _ in TRAIN_VARIANTS]
        to = [False] * len(Xs)
        if mode != "none":
            Xs += [SCX[mid] for mid in skillcorner.MATCH_IDS]
            ys += [sc_lab[mid] for mid in skillcorner.MATCH_IDS]
            gs += [100 + i for i in range(len(skillcorner.MATCH_IDS))]
            to += [mode == "team"] * len(skillcorner.MATCH_IDS)
        return TwoHeadGBM(seed=seed).fit(Xs, ys, gs, team_only=to)

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [m for m in MATCH_IDS if m != T]
        params = DecoderParams(**json.loads((store / T / "params_gbm.json").read_text()))
        m = sportec[T]
        row = {"fold": T, "sportec_only": metrics.score(m.labels, decode(np.load(store / T / "gbm_outer_{}.npy".format(T)), m.half, params),
                                                        m.half)["spell_f1_mean"]}
        for mode in MODES:
            p = fit(others, mode, k).predict_proba(test_X[T])
            r = metrics.score(m.labels, decode(p, m.half, params), m.half)
            row[mode], row[mode + "_poss"] = r["spell_f1_mean"], r["n_pred_spells"]
        row["poss_gt"] = r["n_gt_spells"]
        rows.append(row)
        print(row, flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    folds.to_csv(args.out / "sportec_folds.csv", float_format="%.4f")

    dec = DecoderParams(**{kk: v for kk, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not kk.startswith("_")})
    full_ref = pickle.loads((ROOT / "local_data" / "models" / "full_{}_v1ctl".format(tag) / "models.pkl").read_bytes())["gbm"]
    sample = {g: load_sample(g) for g in GAMES}
    QX = {g: build_features(sample[g], "sample_{}".format(g), "v1ctl") for g in GAMES}
    atd = load_atd()
    AX = build_features(atd, "atd", "v1ctl")
    half_dec = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    trows = []
    for mode in ("none",) + MODES:
        model = full_ref if mode == "none" else fit(list(MATCH_IDS), mode, 99)
        r = {"mode": mode}
        for g in GAMES:
            ms = sample[g]
            pred = decode(model.predict_proba(QX[g]), ms.half, dec)
            r["sample_{}".format(g)] = metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"]
            r["sampleNoD_{}".format(g)] = metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"]
        pred = decode(model.predict_proba(AX), half_dec, dec)
        pred[second_start:kick2] = DEAD
        pred[~atd.graded] = DEAD
        pub = metrics.score_public(pred, bootstrap_n=1000)
        pl = plausibility(pred, atd.half, atd.graded)
        r.update({"sample_mean": np.mean([r["sample_{}".format(g)] for g in GAMES]),
                  "sampleNoD_mean": np.mean([r["sampleNoD_{}".format(g)] for g in GAMES]),
                  "public": pub["spell_f1_mean"], "public_D_f1": pub["per_class"]["D"]["f1"], "public_poss": pub["n_pred_spells"],
                  "poss_per_min": pl["poss_per_min"], "pct_D": pl["pct_D"]})
        trows.append(r)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
    table = pd.DataFrame(trows).set_index("mode")
    table.to_csv(args.out / "transfer.csv", float_format="%.4f")
    summary = {"lomo": {c: float(folds[c].mean()) for c in ("sportec_only",) + MODES},
               "folds_better": {mode: int((folds[mode] > folds["sportec_only"]).sum()) for mode in MODES},
               "detected_only": det}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(folds)
        print(table)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
