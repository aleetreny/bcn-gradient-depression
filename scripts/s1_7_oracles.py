"""S1.7: how much spellF1 is behind each error type? Oracle repairs of the baseline, one at a time.

    .venv311/bin/python scripts/s1_7_oracles.py [--out experiments/outputs/013]

Prediction: the baseline (nearest player + majority +/-50). Repairs: src/eval/oracles.py.
Scenarios:
  - BAR-ATH public 5 min (17 possessions: noisy, shown with the bootstrap interval of the baseline);
  - Sportec clean, 7 matches (the ball is always visible, so the baseline never predicts D);
  - Sportec with the BAR-ATH ball-visibility sequence transplanted (graded frames of the ATD, a
    different stretch per match), the closest to our conditions available before S2.1.
Also "ball fully visible" for the transplanted scenario (= Sportec clean) and one combination.
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
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.eval import metrics, oracles  # noqa: E402
from src.models.baseline import predict  # noqa: E402

COMBO = ("perfect D + timing", ("perfect D", "perfect change timing (±3 s)"))


def apply(name, gt, pred, half):
    if name == COMBO[0]:
        p = pred
        for n in COMBO[1]:
            p = oracles.ORACLES[n](gt, p, half)
        return p
    return oracles.ORACLES[name](gt, pred, half)


def transplant(m, atd_vis, offset):
    """Copy of a Sportec match with the ball hidden where the ATD hides it (from `offset`)."""
    T = m.n_frames
    seq = np.resize(np.roll(atd_vis, -offset), T)
    B = m.B.copy()
    B[~seq] = np.nan
    return replace(m, B=B)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/013")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = list(oracles.ORACLES) + [COMBO[0]]

    # public
    atd = load_atd()
    gt_pub = metrics.public_gt()
    gt_pub = np.where(gt_pub == 3, -1, gt_pub).astype(np.int8)
    base_pub = predict(atd, 50)
    pub = {"baseline": metrics.score_public(base_pub, bootstrap_n=1000)}
    for n in names:
        pub[n] = metrics.score_public(apply(n, gt_pub, base_pub, atd.half), bootstrap_n=0)
    ci = (pub["baseline"]["spell_bootstrap_low"], pub["baseline"]["spell_bootstrap_high"])

    # Sportec clean and transplanted
    sportec = load_all()
    atd_vis = atd.ball_xy_visible[atd.graded]
    scen = {"Sportec clean": {}, "Sportec + ATD ball gaps": {}}
    recovery, painted_over_dead = [], []
    for i, (mid, m) in enumerate(sportec.items()):
        variants = {"Sportec clean": m, "Sportec + ATD ball gaps": transplant(m, atd_vis, i * len(atd_vis) // 7)}
        for sname, mm in variants.items():
            base = predict(mm, 50)
            r = {"baseline": metrics.score(m.labels, base, m.half)}
            for n in names:
                r[n] = metrics.score(m.labels, apply(n, m.labels, base, m.half), m.half)
            if sname == "Sportec + ATD ball gaps":
                clean = predict(m, 50)
                r["ball fully visible"] = metrics.score(m.labels, clean, m.half)
                r["perfect team, ball fully visible"] = metrics.score(m.labels, oracles.perfect_team(m.labels, clean, m.half), m.half)
                r["perfect team, ATD ball gaps"] = r["perfect team (alive frames)"]
                before, still = oracles.short_spell_recovery(m.labels, base, oracles.short_spells(m.labels, base, m.half), m.half)
                recovery.append((before, still))
                changed = oracles.short_spells(m.labels, base, m.half) != base
                painted_over_dead.append((int(changed.sum()), int((changed & (base == 2)).sum())))
            scen[sname][mid] = r

    rows = []
    for n in ["baseline"] + names + ["ball fully visible", "perfect team, ball fully visible"]:
        row = {"repair": n}
        if n in pub:
            row["public spellF1"] = pub[n]["spell_f1_mean"]
            row["public Δ"] = pub[n]["spell_f1_mean"] - pub["baseline"]["spell_f1_mean"]
        for sname, per in scen.items():
            vals = [per[mid][n]["spell_f1_mean"] for mid in per if n in per[mid]]
            if not vals:
                continue
            base = [per[mid]["baseline"]["spell_f1_mean"] for mid in per]
            row[sname + " spellF1"] = float(np.mean(vals))
            row[sname + " Δ"] = float(np.mean(vals) - np.mean(base))
            row[sname + " folds improved"] = int(sum(v > b for v, b in zip(vals, base)))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("repair")
    gaps = {sname: {"pct_D_pred_baseline": float(np.mean([np.mean(predict(transplant(m, atd_vis, i * len(atd_vis) // 7), 50) == 2)
                                                          if sname != "Sportec clean" else 0.0 for i, m in enumerate(sportec.values())]))}
            for sname in scen}

    table.to_csv(args.out / "oracles.csv", float_format="%.4f")
    per = scen["Sportec + ATD ball gaps"]
    ball_under_team = float(np.mean([per[k]["perfect team, ball fully visible"]["spell_f1_mean"] - per[k]["perfect team, ATD ball gaps"]["spell_f1_mean"] for k in per]))
    diag = {"public_baseline_ci_5_95": ci, "pred_D_share": gaps,
            "short_possessions_missed_before_and_still_unmatched_after (transplanted)": [int(sum(x[0] for x in recovery)), int(sum(x[1] for x in recovery))],
            "short_repair_frames_changed_and_of_them_predicted_D (transplanted)": [int(sum(x[0] for x in painted_over_dead)), int(sum(x[1] for x in painted_over_dead))],
            "ball visible gain under perfect team (transplanted)": ball_under_team}
    (args.out / "summary.json").write_text(json.dumps(diag, indent=2))
    plot(table, args.out / "oracles.png")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table)
    print(json.dumps(diag, indent=1))
    return 0


def plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid = "#52514e", "#e4e3df"
    cols = [("Sportec + ATD ball gaps Δ", "Sportec + ATD ball gaps", "#2a78d6"), ("Sportec clean Δ", "Sportec clean", "#8f8e88"),
            ("public Δ", "BAR-ATH public 5 min", "#eb6834")]
    t = table.drop(index="baseline")
    order = t["Sportec + ATD ball gaps Δ"].fillna(-1).sort_values().index
    t = t.loc[order]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    y = np.arange(len(t))
    for j, (c, lab, color) in enumerate(cols):
        ax.barh(y + (j - 1) * 0.27, t[c].fillna(0), height=0.25, color=color, label=lab)
    ax.set_yticks(y, t.index)
    ax.axvline(0, color=ink, lw=0.8)
    ax.set_xlabel("spellF1 gained over the baseline (smooth50) when this error type is fixed with the truth")
    ax.set_title("Where is the score? Oracle repairs, one error type at a time", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=grid, lw=0.6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
