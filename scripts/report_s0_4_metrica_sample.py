"""S0.4 check: Metrica Sample games 1-3 with event-derived H/A/D labels, next to Sportec and BAR-ATH.

    .venv311/bin/python scripts/report_s0_4_metrica_sample.py [--out experiments/outputs/004]

Writes per_game.csv, comparison.csv, summary.json, timeline_game1.png and spell_durations.png.
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
sys.path.insert(0, str(ROOT / "scripts"))
from report_s0_3_sportec import public_gt_stats  # noqa: E402
from src.data.common import AWAY, DEAD, FPS, HOME, label_stats  # noqa: E402
from src.data.metrica_sample import GAMES, ON_BALL, load  # noqa: E402
from src.data.sportec import load_all as load_sportec  # noqa: E402

COLORS = {"H": "#2a78d6", "A": "#eb6834", "D": "#8f8e88", "ink": "#52514e", "grid": "#e4e3df",
          "sample": "#1baf7a", "sportec": "#2a78d6", "public": "#eb6834"}
TIMELINE = (3 * 60, 8 * 60)  # seconds of game 1 shown in the timeline (a 5-minute stretch)


def event_alignment(m):
    """Median distance (m) from the ball to the nearest player of the acting team at on-ball event
    start frames, where the ball is visible. Small values mean events and tracking share a clock."""
    ev = m.meta["events"]
    ev = ev[ev["type"].isin(ON_BALL - {"SET PIECE"})]
    rows = ev["start_frame"].to_numpy() - 1
    rows = rows[(rows >= 0) & (rows < m.n_frames)]
    team = np.where(ev["team"].to_numpy()[:len(rows)] == "H", HOME, AWAY)
    d = np.linalg.norm(m.P[rows, :, :2] - m.B[rows, None, :2], axis=-1)
    same = m.slot_team[None, :] == team[:, None]
    d = np.where(same, d, np.nan)
    nearest = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    nearest = nearest[np.isfinite(nearest) & m.ball_xy_visible[rows]]
    out = {"events": int(len(nearest)), "median_m": float(np.median(nearest)),
           "pct_within_2m": 100 * float(np.mean(nearest < 2))}
    for shift in (-25, 25):  # the same with events shifted by +/- 1 s, as a control
        r2 = np.clip(rows + shift, 0, m.n_frames - 1)
        d2 = np.linalg.norm(m.P[r2, :, :2] - m.B[r2, None, :2], axis=-1)
        d2 = np.where(same, d2, np.nan)
        n2 = np.nanmin(np.where(np.isfinite(d2), d2, np.inf), axis=1)
        n2 = n2[np.isfinite(n2) & m.ball_xy_visible[r2]]
        out["median_m_shift_{:+d}".format(shift)] = float(np.median(n2))
    return out


def plot_timeline(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = (int(s * FPS) for s in TIMELINE)
    t = np.arange(a, b) / FPS
    lab = m.labels[a:b]
    ev = m.meta["events"]
    ev = ev[(ev["start_frame"] - 1 >= a) & (ev["start_frame"] - 1 < b)]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["ink"],
                         "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"]})
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    for code, name in ((HOME, "H"), (AWAY, "A"), (DEAD, "D")):
        ax.fill_between(t, 2.6, 3.4, where=lab == code, step="post", color=COLORS[name], lw=0,
                        label="derived label {}".format(name))
    ax.fill_between(t, 1.75, 2.25, where=m.ball_xy_visible[a:b], step="post", color=COLORS["ink"], lw=0, alpha=0.5,
                    label="ball visible in tracking")
    lanes = {"PASS": 1.0, "CARRY": 1.0, "RECOVERY": 0.6, "BALL LOST": 0.6, "CHALLENGE": 0.6,
             "SHOT": 0.2, "BALL OUT": 0.2, "FAULT RECEIVED": 0.2, "SET PIECE": 0.2}
    for e in ev.itertuples(index=False):
        y = lanes.get(e.type, 0.2)
        x = (e.start_frame - 1) / FPS
        color = COLORS["H"] if e.team == "H" else COLORS["A"]
        ax.plot([x, x], [y - 0.15, y + 0.15], color=color, lw=1.2)
        if e.type in ("SET PIECE", "BALL OUT", "FAULT RECEIVED", "SHOT"):
            ax.text(x, y + 0.2, e.type.title() if e.type != "SET PIECE" else e.subtype.title(), rotation=90,
                    fontsize=7, color=COLORS["ink"], va="bottom", ha="center")
    ax.set_yticks([0.2, 0.6, 1.0, 2.0, 3.0],
                  ["stoppage events", "duel / turnover", "pass / carry", "ball visible", "label"])
    ax.set_ylim(-0.1, 3.6)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("game time (s) — event ticks coloured by team (blue H, orange A)")
    ax.set_title("Metrica Sample game 1, minutes 3–8: event-derived labels vs the events they come from", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper right", bbox_to_anchor=(1, 1.12), fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_durations(sample_durs, sportec_durs, public_dur, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)

    def ecdf(d, **kw):
        x = np.sort(d)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)

    for i, d in enumerate(sportec_durs):
        ecdf(d, color=COLORS["sportec"], lw=1, alpha=0.35, label="Sportec (7 matches)" if i == 0 else None)
    for i, d in enumerate(sample_durs):
        ecdf(d, color=COLORS["sample"], lw=1.8, label="Metrica Sample, derived (3 games)" if i == 0 else None)
    ecdf(public_dur, color=COLORS["public"], lw=2.2, label="BAR-ATH public 5 min (17)")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 200)
    ax.axvline(2, color=COLORS["ink"], lw=0.8, ls=":")
    ax.set_xlabel("possession duration (s, log scale)")
    ax.set_ylabel("share of possessions ≤ x")
    ax.set_title("Possession duration, cumulative, by label source", loc="left")
    ax.grid(color=COLORS["grid"], lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/004")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, durs, align = [], [], {}
    for g in GAMES:
        m = load(g)
        st, dur = label_stats(m.labels, m.half, m.ball_xy_visible)
        n = m.normalized()
        gk_x = {("H" if m.slot_team[i] == HOME else "A"): float(np.nanmean(n.P[:, i, 0]))
                for i in np.flatnonzero(m.slot_is_gk)}
        st.update({"game": g, "unlogged_restarts": m.meta["unlogged_restarts"],
                   "events": len(m.meta["events"]), "slots": m.P.shape[1],
                   "pct_unlabelled_in_halves": 100 * float(np.mean(m.labels[m.half > 0] < 0)),
                   "gk_norm_x_H": gk_x.get("H"), "gk_norm_x_A": gk_x.get("A")})
        rows.append(st)
        durs.append(dur)
        align[g] = event_alignment(m)
    df = pd.DataFrame(rows).set_index("game")

    sportec_rows, sportec_durs = [], []
    for mid, m in load_sportec().items():
        st, dur = label_stats(m.labels, m.half, m.ball_xy_visible)
        sportec_rows.append(st)
        sportec_durs.append(dur)
    public, public_dur = public_gt_stats()

    def pooled(rows_, durs_):
        d = np.concatenate(durs_)
        r = pd.DataFrame(rows_)
        return {"pct_D": r["pct_D"].mean(), "possessions_per_min": r["possessions_per_min"].mean(),
                "poss_median_s": float(np.median(d)), "poss_mean_s": float(d.mean()),
                "pct_poss_lt_2s": 100 * float(np.mean(d < 2)), "dead_mean_s": r["dead_mean_s"].mean(),
                "pct_changes_direct": r["pct_changes_direct"].mean(),
                "pct_ball_visible": r["pct_ball_visible"].mean() if "pct_ball_visible" in r else np.nan}

    comparison = pd.DataFrame({
        "Sportec (7)": pooled(sportec_rows, sportec_durs),
        "Metrica Sample (3, derived)": pooled(rows, durs),
        "BAR-ATH public 5 min": {k: public.get(k, np.nan) for k in pooled(rows, durs)},
    }).T

    df.to_csv(args.out / "per_game.csv", float_format="%.3f")
    comparison.to_csv(args.out / "comparison.csv", float_format="%.3f")
    (args.out / "summary.json").write_text(json.dumps({"event_alignment": align}, indent=2))
    plot_timeline(load(1), args.out / "timeline_game1.png")
    plot_durations(durs, sportec_durs, public_dur, args.out / "spell_durations.png")

    cols = ["minutes", "pct_D", "possessions", "possessions_per_min", "poss_mean_s", "poss_median_s",
            "pct_poss_lt_2s", "dead_runs", "dead_mean_s", "pct_changes_direct", "pct_ball_visible",
            "unlogged_restarts", "pct_unlabelled_in_halves", "gk_norm_x_H", "gk_norm_x_A"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(df[cols])
        print()
        print(comparison)
    print(json.dumps(align, indent=1))
    failed = ["game {} possessions/min {:.2f} outside 2.5-6".format(g, v)
              for g, v in df["possessions_per_min"].items() if not 2.5 <= v <= 6]
    failed += ["game {} events not aligned with tracking".format(g) for g, a in align.items() if a["median_m"] > 0.5]
    print("\nCHECKS:", "OK" if not failed else "FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
