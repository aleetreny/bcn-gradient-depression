"""S0.5 check: the src/eval harness reproduces the official scores and behaves sensibly.

    .venv311/bin/python scripts/report_s0_5_evaluator.py [--out experiments/outputs/005]

1. The four published baselines scored through src.eval.metrics.score_public must equal
   evaluation/scoreboard.csv (written by the official run_evaluation.py) on every column.
2. Bootstrap intervals (1000 draws) for the baselines on the public window.
3. LOMO harness on Sportec with "oracle with lag" predictors (truth delayed by k frames): sanity
   check of the harness and a measure of what lateness costs in spellF1 on full matches.
4. Error decomposition of baseline_smooth50 on the public window and of the 1 s-late oracle.
5. Timeline of the public 5 minutes: truth vs smooth50 vs smooth125, with ball visibility.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.eval import errors, metrics, plots  # noqa: E402

BASELINES = ("baseline_smooth125", "baseline_smooth50", "baseline_smooth25", "baseline_raw")
LAGS = (0, 6, 12, 25, 50)
PUBLIC_WINDOW = (11185, 18685)


def official_to_common(codes):
    return np.where(codes == 3, -1, codes).astype(np.int8)


def lagged(labels, half, k):
    """Truth delayed by k frames inside each half (the first k frames repeat the first label)."""
    out = labels.copy()
    if k == 0:
        return out
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        seg = labels[idx]
        out[idx] = np.r_[np.repeat(seg[0], k), seg[:-k]]
    return out


def compare_with_scoreboard():
    """Rebuild each baseline's scoreboard.csv line from our scores with the runner's own row writer
    and compare the formatted strings: identical text means identical scores on every column."""
    sys.path.insert(0, str(ROOT / "evaluation"))
    import run_evaluation as runner
    lines = (ROOT / "evaluation" / "scoreboard.csv").read_text().splitlines()
    header = lines[0].split(",")
    official = {line.split(",")[1]: line.split(",") for line in lines[1:]}
    rows, all_identical = [], True
    for name in BASELINES:
        r = metrics.score_public(ROOT / "evaluation" / "results" / "{}.csv".format(name), bootstrap_n=0)
        ours = runner.scoreboard_row(official[name][0], {"name": name, "result": r}, metrics.pm.TAUS)
        ours = [runner.format_value(ours.get(col)) for col in header]
        mismatched = [col for col, x, y in zip(header, ours, official[name]) if x != y]
        all_identical &= not mismatched
        rows.append({"baseline": name, "spell_f1_mean": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
                     "n_pred_spells": r["n_pred_spells"], "columns_compared": len(header),
                     "columns_mismatched": len(mismatched)})
    return pd.DataFrame(rows).set_index("baseline"), all_identical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/005")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1-2. official reproduction + bootstrap
    repro, identical = compare_with_scoreboard()
    boot = {}
    for name in BASELINES:
        r = metrics.score_public(ROOT / "evaluation" / "results" / "{}.csv".format(name), bootstrap_n=1000)
        boot[name] = {"spell_f1_mean": r["spell_f1_mean"], "spell_ci_5_95": [r["spell_bootstrap_low"], r["spell_bootstrap_high"]],
                      "macro_f1": r["macro_f1"], "macro_ci_5_95": [r["bootstrap_low"], r["bootstrap_high"]]}
    repro["spell_ci_5"] = [boot[n]["spell_ci_5_95"][0] for n in repro.index]
    repro["spell_ci_95"] = [boot[n]["spell_ci_5_95"][1] for n in repro.index]

    # 3. LOMO with lagged oracles on Sportec
    matches = load_all()
    lag_rows = []
    for k in LAGS:
        folds = metrics.lomo(matches, lambda train, test, k=k: lagged(test.labels, test.half, k),
                             name="lag{}".format(k), verbose=False)
        s = metrics.lomo_summary(folds)
        lag_rows.append({"lag_frames": k, "lag_s": k / 25, "lomo_spellf1": s["lomo_spellf1_media"],
                         "lomo_std": s["lomo_std"], "macro_f1": s["lomo_macrof1_media"],
                         "min_fold": folds["spell_f1_mean"].min(), "max_fold": folds["spell_f1_mean"].max()})
    lag_table = pd.DataFrame(lag_rows).set_index("lag_frames")

    # gap insertion matters: same oracle without the X separator between halves
    m0 = next(iter(matches.values()))
    with_gap = metrics.score(m0.labels, m0.labels, m0.half)["n_gt_spells"]
    no_gap = metrics.score(m0.labels, m0.labels, m0.half, insert_gaps=False)["n_gt_spells"]

    # 4. error decomposition
    gt_public = official_to_common(metrics.public_gt())
    atd = load_atd()
    smooth50 = official_to_common(metrics.pm.label_codes(metrics.pm.read_label_csv(
        str(ROOT / "evaluation/results/baseline_smooth50.csv"))))
    dec_public = errors.decompose(gt_public, smooth50, atd.half)
    lag25 = lagged(m0.labels, m0.half, 25)
    dec_lag = errors.decompose(m0.labels, lag25, m0.half)

    # 5. timeline
    smooth125 = official_to_common(metrics.pm.label_codes(metrics.pm.read_label_csv(
        str(ROOT / "evaluation/results/baseline_smooth125.csv"))))
    a, b = PUBLIC_WINDOW
    plots.timeline(args.out / "timeline_public.png",
                   [("truth (public GT)", gt_public), ("baseline smooth50", smooth50), ("baseline smooth125", smooth125)],
                   a, b, title="BAR-ATH public 5 minutes: truth vs baselines",
                   extra=[("ball visible (ATD)", atd.ball_xy_visible)])

    repro.to_csv(args.out / "baselines_reproduced.csv", float_format="%.6f")
    lag_table.to_csv(args.out / "lomo_lagged_oracle.csv", float_format="%.4f")
    dec_public["table"].to_csv(args.out / "errors_smooth50_public.csv", index=False)
    dec_lag["table"].to_csv(args.out / "errors_lag25_sportec_{}.csv".format(m0.meta["match_id"]), index=False)
    (args.out / "summary.json").write_text(json.dumps(
        {"scoreboard_rows_identical": identical, "bootstrap": boot,
         "gap_insertion_check": {"match": m0.meta["match_id"], "spells_with_gap": with_gap, "spells_without_gap": no_gap}},
        indent=2, default=float))

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print("1-2. Official reproduction: scoreboard.csv lines identical as text: {}".format(identical))
        print(repro)
        print("\n3. LOMO on Sportec, oracle delayed by k frames")
        print(lag_table)
        print("   gap insertion: {} spells with the half separator, {} without".format(with_gap, no_gap))
        print("\n4a. Error decomposition, baseline_smooth50 on the public window")
        print(dec_public["table"].to_string(index=False))
        print("\n4b. Error decomposition, oracle 25 frames late, Sportec {}".format(m0.meta["match_id"]))
        print(dec_lag["table"].to_string(index=False))
    checks = {"official scoreboard reproduced": identical,
              "oracle without lag scores 1": lag_table.loc[0, "lomo_spellf1"] == 1.0,
              "spellF1 falls with lag": bool(np.all(np.diff(lag_table["lomo_spellf1"].to_numpy()) < 0)),
              "lag-25 offsets all +25": bool(np.all(dec_lag["team_change_offsets"] == 25))}
    failed = [name for name, ok in checks.items() if not ok]
    print("\nCHECKS:", "OK" if not failed else "FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
