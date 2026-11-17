"""S1.1: what does the tracking show around each real label change of the BAR-ATH public window?

    .venv311/bin/python scripts/s1_1_transition_anatomy.py [--out experiments/outputs/007]

For each of the 20 ground-truth transitions in the public 5 minutes, a +/-3 s window:
ball state (x,y visible / third channel only / absent), team of the nearest visible player,
distance from the ball to the nearest player of each team, ball outside the pitch, Smart Tagging
phases active and the nearest SET PIECES start. For team changes, the offset between the change of
the nearest team and the change of the label. Writes transitions.csv, summary.json and
mini_timelines.png (one panel per transition).
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
from src.data.common import AWAY, DEAD, FPS, HOME  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.baseline import nearest_team  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402

W = 3 * FPS
NAMES = {HOME: "H", AWAY: "A", DEAD: "D"}
COLORS = {"H": "#2a78d6", "A": "#eb6834", "D": "#8f8e88", "none": "#e4e3df", "ink": "#52514e", "grid": "#e4e3df"}
KICKOFF_1 = 11185


def team_distances(m):
    """(T, 2) distance from the ball to the nearest visible player of H and of A (NaN without ball)."""
    out = np.full((m.n_frames, 2), np.nan, np.float32)
    rows = np.flatnonzero(m.ball_xy_visible)
    d = np.linalg.norm(m.P[rows, :, :2] - m.B[rows, None, :2], axis=-1)
    for k, team in enumerate((HOME, AWAY)):
        dt = d[:, m.slot_team == team]
        with np.errstate(all="ignore"):
            out[rows, k] = np.where(np.isfinite(dt).any(1), np.nanmin(np.where(np.isfinite(dt), dt, np.inf), 1), np.nan)
    return out


def nearest_change_offset(nearest, f, team):
    """Offset (frames) of the change of the nearest team to `team` closest to f within +/-W.

    A change is a frame c with nearest[c] == team whose previous known nearest (skipping frames with
    nobody) was the other team. Returns ('already', None) if `team` was nearest throughout the
    window before f, ('never', None) if `team` is never nearest in the window.
    """
    lo, hi = max(f - W, 0), f + W + 1
    seg = nearest[lo:hi]
    known = np.flatnonzero(seg >= 0)
    if not np.any(seg[known] == team):
        return "never", None
    changes = [lo + known[i] for i in range(1, len(known)) if seg[known[i]] == team and seg[known[i - 1]] != team]
    if not changes:
        return "already", None
    c = min(changes, key=lambda c: abs(c - f))
    return "change", int(c - f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/007")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = load_atd()
    gt_official = metrics.public_gt()
    gt = np.where(gt_official == 3, -1, gt_official).astype(np.int8)
    trans = metrics.pm.transitions(gt_official, metrics.pm.valid_runs(gt_official))
    nearest = nearest_team(m)
    dist = team_distances(m)
    vis, c3 = m.ball_xy_visible, m.ball_speed_only
    length, width = m.pitch
    outside = vis & ((m.B[:, 0] < 0) | (m.B[:, 0] > length) | (m.B[:, 1] < 0) | (m.B[:, 1] > width))
    tags = load_tags()
    set_piece_starts = np.array([t["start_frame"] for t in tags if t["code"].upper() == "SET PIECES"])

    rows = []
    for i, (f, a, b) in enumerate(trans, start=1):
        kind = ("cambio de equipo" if {a, b} == {HOME, AWAY} else "inicio D" if b == DEAD else "fin D")
        win = slice(f - W, f + W + 1)
        near = slice(f - FPS // 2, f + FPS // 2 + 1)
        state = "x,y" if vis[f] else ("solo c3" if c3[f] else "nada")
        row = {"n": i, "frame": f, "t_partido_s": round((f - KICKOFF_1) / FPS, 1), "tipo": kind,
               "de_a": "{}→{}".format(NAMES[a], NAMES[b]), "balon_en_f": state,
               "pct_balon_±3s": round(100 * vis[win].mean()), "pct_balon_±0.5s": round(100 * vis[near].mean()),
               "mas_cercano_en_f": NAMES.get(int(nearest[f]), "—"),
               "dist_H_m": None if np.isnan(dist[f, 0]) else round(float(dist[f, 0]), 1),
               "dist_A_m": None if np.isnan(dist[f, 1]) else round(float(dist[f, 1]), 1),
               "balon_fuera_±3s": int(outside[win].sum())}
        if b in (HOME, AWAY):
            status, off = nearest_change_offset(nearest, f, b)
            row["cambio_mas_cercano"] = status
            row["desfase_frames"] = off
        else:
            row["cambio_mas_cercano"] = "n/a"
            row["desfase_frames"] = None
        # ball disappearance / reappearance relative to the label change
        if b == DEAD:
            before = np.flatnonzero(vis[f - W:f + W + 1])
            row["ultimo_balon_visible_rel"] = int(before[before <= W].max() - W) if np.any(before <= W) else None
        active = [t for t in tags if t["start_frame"] <= f < t["end_frame"]]
        row["smart_tagging_activo"] = "; ".join("{} ({})".format(t["code"].title(), t["tags"].get("Team", "-")) for t in active) or "—"
        sp = set_piece_starts[np.argmin(np.abs(set_piece_starts - f))] if len(set_piece_starts) else None
        row["set_piece_mas_cercano_rel"] = int(sp - f) if sp is not None and abs(sp - f) <= 10 * FPS else None
        rows.append(row)
    df = pd.DataFrame(rows).set_index("n")

    team = df[df["tipo"] == "cambio de equipo"]
    to_team = df[df["de_a"].str.endswith(("H", "A"))]
    offs = to_team["desfase_frames"].dropna().astype(int)
    summary = {
        "n_transitions": len(df), "by_type": df["tipo"].value_counts().to_dict(),
        "team_changes_ball_visible_at_f": int((team["balon_en_f"] == "x,y").sum()),
        "team_changes_ball_visible_pct_±0.5s_median": float(team["pct_balon_±0.5s"].median()),
        "to_team_nearest_status": to_team["cambio_mas_cercano"].value_counts().to_dict(),
        "to_team_offset_frames": offs.tolist(),
        "to_team_offset_median": float(offs.median()) if len(offs) else None,
        "dead_starts_ball_visible_at_f": int(((df["tipo"] == "inicio D") & (df["balon_en_f"] == "x,y")).sum()),
        "dead_ends_with_set_piece_within_10s": int(df.loc[df["tipo"] == "fin D", "set_piece_mas_cercano_rel"].notna().sum()),
    }

    df.to_csv(args.out / "transitions.csv")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_panels(df, gt, nearest, dist, vis, c3, args.out / "mini_timelines.png")

    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 60):
        print(df.drop(columns=["smart_tagging_activo"]))
        print()
        print(df[["tipo", "de_a", "smart_tagging_activo", "set_piece_mas_cercano_rel"]])
    print(json.dumps(summary, indent=1, default=str))
    return 0


def plot_panels(df, gt, nearest, dist, vis, c3, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["ink"],
                         "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"]})
    n = len(df)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(13, 2.3 * np.ceil(n / cols)),
                             sharex=True, sharey=True, constrained_layout=True)
    t = np.arange(-W, W + 1) / FPS
    for ax, (i, r) in zip(axes.flat, df.iterrows()):
        f = int(r["frame"])
        sl = slice(f - W, f + W + 1)
        for code, key in ((HOME, "H"), (AWAY, "A"), (DEAD, "D")):
            ax.fill_between(t, 26, 29, where=gt[sl] == code, step="mid", color=COLORS[key], lw=0)
        for code, key in ((HOME, "H"), (AWAY, "A")):
            ax.fill_between(t, 22, 25, where=nearest[sl] == code, step="mid", color=COLORS[key], lw=0)
        ax.fill_between(t, 22, 25, where=nearest[sl] < 0, step="mid", color=COLORS["none"], lw=0)
        ax.fill_between(t, 19.5, 20.5, where=vis[sl], step="mid", color=COLORS["ink"], lw=0, alpha=0.5)
        ax.fill_between(t, 19.5, 20.5, where=c3[sl], step="mid", color=COLORS["ink"], lw=0, alpha=0.2)
        ax.plot(t, np.clip(dist[sl, 0], 0, 18), color=COLORS["H"], lw=1.2)
        ax.plot(t, np.clip(dist[sl, 1], 0, 18), color=COLORS["A"], lw=1.2)
        ax.axvline(0, color=COLORS["ink"], lw=0.8, ls=":")
        ax.set_title("#{} {} {} (t={} s)".format(i, r["de_a"], r["tipo"], r["t_partido_s"]), loc="left", fontsize=7.5)
        ax.set_ylim(0, 29.5)
        ax.set_yticks([5, 10, 15, 20, 23.5, 27.5], ["5 m", "10", "15", "ball", "nearest", "truth"])
        ax.grid(axis="y", color=COLORS["grid"], lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("s from the true change")
    fig.suptitle("Public transitions: truth, nearest team, ball visible, and ball-to-nearest-player distance per team "
                 "(blue H, orange A)", x=0.01, ha="left", fontsize=9)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
