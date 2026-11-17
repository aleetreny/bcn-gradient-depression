"""S0.6 check: the re-implemented baseline is frame-identical on BAR-ATH and gives the first LOMO row.

    .venv311/bin/python scripts/report_s0_6_baseline.py [--out experiments/outputs/006]

1. BAR-ATH: src.models.baseline for windows 0/25/50/125, written with the official writer, must be
   byte-identical to evaluation/results/baseline_*.csv.
2. Sportec (clean) LOMO with the same rule, per window: the reference rows for every later step.
3. Metrica Sample (derived labels), same rule, as an extra reference.
4. Error decomposition of smooth50 pooled over the 7 Sportec matches.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import LABEL_CHARS  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.eval import errors, metrics  # noqa: E402
from src.models.baseline import nearest_team, predict, smooth  # noqa: E402
from starter.baseline import write_submission  # noqa: E402

WINDOWS = {"raw": 0, "smooth25": 25, "smooth50": 50, "smooth125": 125}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/006")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. byte identity on BAR-ATH
    atd = load_atd()
    codes = nearest_team(atd)
    identity = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, w in WINDOWS.items():
            path = Path(tmp) / "baseline_{}.csv".format(name)
            write_submission(str(path), LABEL_CHARS[smooth(codes, w)])
            official = ROOT / "evaluation" / "results" / "baseline_{}.csv".format(name)
            identity[name] = filecmp.cmp(path, official, shallow=False)
    public = metrics.score_public(LABEL_CHARS[predict(atd, 50)], bootstrap_n=1000)

    # 2. Sportec clean LOMO (the rule has no parameters to fit, so train is ignored)
    sportec = load_all()
    lomo_rows, folds_by_w = [], {}
    for name, w in WINDOWS.items():
        folds = metrics.lomo(sportec, lambda train, test, w=w: predict(test, w), name=name, verbose=False)
        folds_by_w[name] = folds
        s = metrics.lomo_summary(folds)
        lomo_rows.append({"window": name, **s, "pct_pred_D": 0.0,
                          "gt_poss_per_min": float(folds["gt_poss_per_min"].mean()),
                          "team_acc_gt_alive": float(folds["team_acc_gt_alive"].mean()),
                          "alive_dead_f1": float(folds["alive_dead_f1"].mean())})
    lomo_table = pd.DataFrame(lomo_rows).set_index("window")
    per_fold = folds_by_w["smooth50"][["spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells",
                                       "pred_poss_per_min", "gt_poss_per_min", "team_acc_gt_alive"]]

    # 3. Metrica Sample
    sample = {g: load_sample(g) for g in GAMES}
    sample_rows = []
    for name, w in WINDOWS.items():
        for g, m in sample.items():
            pred = predict(m, w)
            r = metrics.score(m.labels, pred, m.half)
            sample_rows.append({"window": name, "game": g, "spell_f1_mean": r["spell_f1_mean"],
                                "macro_f1": r["macro_f1"], "n_pred_spells": r["n_pred_spells"],
                                "n_gt_spells": r["n_gt_spells"], "pct_pred_D": 100 * float(np.mean(pred[m.labels >= 0] == 2))})
    sample_table = pd.DataFrame(sample_rows).groupby("window", sort=False).mean(numeric_only=True).drop(columns="game")

    # 4. error decomposition, smooth50 pooled over Sportec
    tables = []
    for mid, m in sportec.items():
        t = errors.decompose(m.labels, predict(m, 50), m.half)["table"]
        tables.append(t.set_index("item")["count"].rename(mid))
    pooled = pd.concat(tables, axis=1)
    pooled["total"] = pooled.sum(axis=1)

    lomo_table.to_csv(args.out / "lomo_sportec_baseline.csv", float_format="%.4f")
    per_fold.to_csv(args.out / "lomo_sportec_smooth50_per_fold.csv", float_format="%.4f")
    sample_table.to_csv(args.out / "metrica_sample_baseline.csv", float_format="%.4f")
    pooled.to_csv(args.out / "errors_smooth50_sportec.csv")
    (args.out / "summary.json").write_text(json.dumps({
        "bar_ath_byte_identical": identity,
        "public_smooth50": metrics.headline(public)}, indent=2, default=float))

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print("1. BAR-ATH byte-identical to the official CSVs:", identity)
        print("   public smooth50: spellF1 {:.4f} [{:.3f}, {:.3f}]  macroF1 {:.4f}".format(
            public["spell_f1_mean"], public["spell_bootstrap_low"], public["spell_bootstrap_high"], public["macro_f1"]))
        print("\n2. Sportec clean, LOMO (7 folds)")
        print(lomo_table)
        print("\n   smooth50 per fold")
        print(per_fold)
        print("\n3. Metrica Sample (derived labels), mean of 3 games")
        print(sample_table)
        print("\n4. Error decomposition, smooth50, Sportec pooled")
        print(pooled["total"].to_string())
    failed = [name for name, ok in identity.items() if not ok]
    if abs(public["spell_f1_mean"] - 0.325581) > 1e-6:
        failed.append("public smooth50 spellF1")
    print("\nCHECKS:", "OK" if not failed else "FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
