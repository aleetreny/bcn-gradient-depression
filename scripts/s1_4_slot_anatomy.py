"""S1.4: how broken is the player tracking of the BAR-ATH ATD? (Sportec shown as the clean reference)

    .venv311/bin/python scripts/s1_4_slot_anatomy.py [--out experiments/outputs/010]

On graded frames:
- fragments: maximal visible runs of one slot; lengths, births per minute, fragments per slot;
- visible slots per team and frame (11 = complete), frames with > 11 (duplicates) or < 11;
- duplicates: two visible slots of the same team closer than 1 m;
- impossible jumps: > 1 m between consecutive frames inside a fragment (> 25 m/s);
- smoothness: median second difference of positions;
- goalkeeper slots (position_type == Goalkeeper): visibility, position, and whether another slot of
  the same team stands in the goal area when the goalkeeper slot is missing;
- where missing players are likely to be: team centroid of visible players vs the ball side.
Aggregates are merged into local_data/degradation_params.json under "players".
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
from src.data.common import AWAY, FPS, HOME, runs  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load as load_sportec  # noqa: E402

COLORS = {"H": "#2a78d6", "A": "#eb6834", "ink": "#52514e", "grid": "#e4e3df", "ref": "#8f8e88"}


def fragments(vis, graded):
    """Lengths (frames) of visible runs per slot, cut at ungraded frames."""
    out = []
    for s in range(vis.shape[1]):
        key = np.where(graded, vis[:, s].astype(int), -1)
        a, e, v = runs(key)
        out.append((e - a)[v == 1])
    return out


def describe(m, graded):
    vis = m.slot_visible & graded[:, None]
    frags = fragments(m.slot_visible, graded)
    lengths = np.concatenate(frags) / FPS
    minutes = graded.sum() / FPS / 60
    per_team = {k: vis[:, m.slot_team == code].sum(1)[graded] for k, code in (("H", HOME), ("A", AWAY))}

    # duplicates and jumps
    P = m.P[..., :2]
    dup_frames = 0
    for code in (HOME, AWAY):
        idx = np.flatnonzero(m.slot_team == code)
        sub = P[graded][:, idx]
        d = np.linalg.norm(sub[:, :, None, :] - sub[:, None, :, :], axis=-1)
        iu = np.triu_indices(len(idx), 1)
        close = (d[:, iu[0], iu[1]] < 1.0)
        dup_frames += int(np.any(close, axis=1).sum())
    step = np.linalg.norm(np.diff(P, axis=0), axis=-1)
    both = m.slot_visible[1:] & m.slot_visible[:-1] & graded[1:, None]
    jumps = int((step[both] > 1.0).sum())
    d2 = P[2:] - 2 * P[1:-1] + P[:-2]
    smooth = float(np.nanmedian(np.linalg.norm(d2, axis=-1)[graded[1:-1]]))
    return {
        "slots": int(m.P.shape[1]),
        "fragments": int(len(lengths)), "fragments_per_min": len(lengths) / minutes,
        "fragment_len_s_p10_p50_p90": np.percentile(lengths, [10, 50, 90]).round(1).tolist(),
        "pct_visible_time_in_fragments_lt_10s": 100 * float(lengths[lengths < 10].sum() / lengths.sum()),
        "pct_visible_time_in_fragments_gt_5min": 100 * float(lengths[lengths > 300].sum() / lengths.sum()),
        "visible_per_team_mean": {k: float(v.mean()) for k, v in per_team.items()},
        "pct_frames_team_lt_11": {k: 100 * float((v < 11).mean()) for k, v in per_team.items()},
        "pct_frames_team_gt_11": {k: 100 * float((v > 11).mean()) for k, v in per_team.items()},
        "pct_frames_team_le_9": {k: 100 * float((v <= 9).mean()) for k, v in per_team.items()},
        "pct_frames_with_same_team_pair_lt_1m": 100 * dup_frames / int(graded.sum()) / 2,
        "jumps_gt_1m_per_frame": jumps, "jumps_per_min": jumps / minutes,
        "second_diff_median_m": smooth,
        "per_team_hist": {k: np.bincount(v, minlength=16)[:16].tolist() for k, v in per_team.items()},
        "fragment_lengths_s": lengths,
    }


def goalkeepers(m, graded):
    n = m.normalized()
    out = {}
    L, W = m.pitch
    for s in np.flatnonzero(m.slot_is_gk):
        team = "H" if m.slot_team[s] == HOME else "A"
        vis = m.slot_visible[:, s] & graded
        x = n.P[:, s, 0]
        own_goal_x = 0.0 if team == "H" else L
        dist_goal = np.abs(x - own_goal_x)
        # when the GK slot is missing: is another same-team slot within 20 m of the own goal line?
        missing = graded & ~m.slot_visible[:, s]
        same = (m.slot_team == m.slot_team[s]) & (np.arange(len(m.slot_team)) != s)
        other_x = n.P[:, same, 0]
        near_goal = np.any(np.abs(other_x - own_goal_x) < 20, axis=1)
        out[team] = {"slot": int(s) + 1, "visible_pct_graded": 100 * float(vis.sum() / graded.sum()),
                     "fragments": int(len(fragments(m.slot_visible[:, [s]], graded)[0])),
                     "median_dist_own_goal_m": float(np.nanmedian(dist_goal[vis])),
                     "pct_within_20m_own_goal": 100 * float(np.mean(dist_goal[vis] < 20)),
                     "when_missing_other_slot_near_goal_pct": 100 * float(near_goal[missing].mean()) if missing.any() else None}
        # do other slots of this team ever behave like a keeper (deepest player, < 20 m from goal most of the time)?
        deep = []
        for t in np.flatnonzero(same):
            v = m.slot_visible[:, t] & graded
            if v.sum() > 25 * 60:
                deep.append((int(t) + 1, 100 * float(np.mean(np.abs(n.P[v, t, 0] - own_goal_x) < 20)), int(v.sum())))
        out[team]["other_slots_mostly_near_goal"] = [d for d in deep if d[1] > 80]
    return out


def missing_side(m, graded):
    """When a team shows < 11 slots, is the ball far from its visible players (missing ones off-camera)?"""
    rows = np.flatnonzero(graded & m.ball_xy_visible)
    res = {}
    for k, code in (("H", HOME), ("A", AWAY)):
        idx = m.slot_team == code
        cnt = m.slot_visible[rows][:, idx].sum(1)
        cx = np.nanmean(np.where(m.slot_visible[rows][:, idx], m.P[rows][:, idx, 0], np.nan), axis=1)
        dist = np.abs(cx - m.B[rows, 0])
        res[k] = {"centroid_to_ball_x_m_when_11plus": float(np.nanmedian(dist[cnt >= 11])),
                  "centroid_to_ball_x_m_when_le_9": float(np.nanmedian(dist[cnt <= 9]))}
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/010")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = load_atd()
    g = m.graded
    blackout = m.slot_visible.sum(1) < 12
    atd = describe(m, g)
    atd_nb = describe(m, g & ~blackout)
    sp_m = load_sportec("J03WMX")
    sp = describe(sp_m, sp_m.graded)
    gk = goalkeepers(m, g)
    side = missing_side(m, g & ~blackout)

    keys = ["slots", "fragments", "fragments_per_min", "fragment_len_s_p10_p50_p90", "pct_visible_time_in_fragments_lt_10s",
            "pct_visible_time_in_fragments_gt_5min", "visible_per_team_mean", "pct_frames_team_lt_11", "pct_frames_team_gt_11",
            "pct_frames_team_le_9", "pct_frames_with_same_team_pair_lt_1m", "jumps_per_min", "second_diff_median_m"]
    table = pd.DataFrame({"BAR-ATH ATD": {k: atd[k] for k in keys}, "ATD without blackouts": {k: atd_nb[k] for k in keys},
                          "Sportec J03WMX": {k: sp[k] for k in keys}})

    params = json.loads((ROOT / "local_data/degradation_params.json").read_text()) if (ROOT / "local_data/degradation_params.json").is_file() else {}
    edges = [0, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 3000, 6000]
    params["players"] = {
        "source": "BAR-ATH ATD graded frames without blackouts, S1.4",
        "fragment_len_s_hist": {"edges": edges, "counts": np.histogram(atd_nb["fragment_lengths_s"], edges)[0].tolist()},
        "fragments_per_min": atd_nb["fragments_per_min"],
        "visible_per_team_hist": atd_nb["per_team_hist"],
        "pct_frames_same_team_pair_lt_1m": atd_nb["pct_frames_with_same_team_pair_lt_1m"],
        "jumps_per_min": atd_nb["jumps_per_min"], "second_diff_median_m": atd_nb["second_diff_median_m"],
        "blackout_share_of_graded": float(blackout[g].mean()),
        "goalkeeper_slots": gk,
    }
    (ROOT / "local_data/degradation_params.json").write_text(json.dumps(params, indent=2))
    (args.out / "degradation_params.json").write_text(json.dumps(params, indent=2))
    table.to_csv(args.out / "slots_summary.csv")
    (args.out / "summary.json").write_text(json.dumps({"goalkeepers": gk, "missing_side": side}, indent=2))
    plot(atd_nb, sp, args.out / "slots.png")

    with pd.option_context("display.width", 220, "display.max_colwidth", 70):
        print(table)
    print(json.dumps({"goalkeepers": gk, "missing_side": side}, indent=1))
    return 0


def plot(atd, sp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["ink"],
                         "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax = axes[0]
    edges = np.logspace(np.log10(0.04), np.log10(6000), 40)
    for name, d, c in (("BAR-ATH ATD (no blackouts)", atd, COLORS["H"]), ("Sportec J03WMX", sp, COLORS["ref"])):
        L = d["fragment_lengths_s"]
        ax.hist(L, bins=edges, weights=L / L.sum() * 100, histtype="step", lw=2, color=c, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("slot fragment length (s, log)")
    ax.set_ylabel("% of visible player-time")
    ax.set_title("How long a slot keeps the same player", loc="left")
    ax.legend(frameon=False)
    ax = axes[1]
    x = np.arange(16)
    for i, (k, c) in enumerate((("H", COLORS["H"]), ("A", COLORS["A"]))):
        h = np.array(atd["per_team_hist"][k], float)
        ax.bar(x + (i - 0.5) * 0.4, 100 * h / h.sum(), width=0.38, color=c, label=k)
    ax.axvline(11, color=COLORS["ink"], ls=":", lw=0.8)
    ax.set_xlabel("visible slots of the team in a frame")
    ax.set_ylabel("% of graded frames (no blackouts)")
    ax.set_title("Players visible per team (dotted = 11)", loc="left")
    ax.legend(frameon=False)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=COLORS["grid"], lw=0.6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
