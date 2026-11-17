"""S1.3: anatomy of the ball gaps in the full BAR-ATH match.

    .venv311/bin/python scripts/s1_3_ball_gaps.py [--out experiments/outputs/009]

A gap is a maximal run of graded frames without ball x,y, inside one half. For every gap: length,
context when it starts (ball speed, players within 3 m, distance to the nearest pitch line, zone),
blackout (< 12 visible slots), third channel present, a stoppage proxy (players' mean speed), Smart
Tagging, and whether the nearest team is the same on both sides of the gap.

There is no full-match truth, so "is the ball alive?" is approximated by the players' collective
speed (stoppage proxy), first validated against the public 5 minutes. Aggregates only; the
degradation parameters for S2.1 are written to local_data/degradation_params.json.
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
from src.data.common import DEAD, FPS, runs  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.baseline import nearest_team  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402

BUCKETS = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)]
BUCKET_NAMES = ["<0.5 s", "0.5–1 s", "1–2 s", "2–5 s", "5–10 s", "10–30 s", ">30 s"]
SLOW = 1.2      # m/s, players' mean speed under which the stoppage proxy fires (smoothed over 2 s)
BLACKOUT = 12   # visible slots
COLORS = {"main": "#2a78d6", "alt": "#eb6834", "ink": "#52514e", "grid": "#e4e3df", "band": "#f1f0ec"}


def smooth(x, w):
    k = np.ones(w) / w
    valid = np.isfinite(x)
    num = np.convolve(np.where(valid, x, 0), k, "same")
    den = np.convolve(valid.astype(float), k, "same")
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def bucket_of(seconds):
    for name, (a, b) in zip(BUCKET_NAMES, BUCKETS):
        if a <= seconds < b:
            return name
    return BUCKET_NAMES[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/009")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = load_atd()
    L, W = m.pitch
    vis, c3, g = m.ball_xy_visible, m.ball_speed_only, m.graded
    slots = m.slot_visible.sum(1)
    blackout = slots < BLACKOUT
    with np.errstate(all="ignore"):
        team_speed = np.nanmean(np.where(m.slot_visible, m.P[..., 2], np.nan), axis=1)
    speed2s = smooth(team_speed, 2 * FPS)
    slow = speed2s < SLOW
    nearest = nearest_team(m)
    tags = load_tags()
    phase_cover = np.zeros(m.n_frames, bool)
    setpiece_cover = np.zeros(m.n_frames, bool)
    for t in tags:
        (setpiece_cover if t["code"].upper() == "SET PIECES" else phase_cover)[t["start_frame"]:t["end_frame"]] = True

    # --- 1. validate the stoppage proxy on the public window
    gt = metrics.public_gt()
    pub = gt != 3
    dead = gt == DEAD
    proxy_valid = {
        "public_frames": int(pub.sum()),
        "P(slow | D)": float(slow[pub & dead].mean()), "P(slow | alive)": float(slow[pub & ~dead].mean()),
        "P(D | slow)": float(dead[pub & slow].mean()) if (pub & slow).any() else None,
        "P(visible | alive)": float(vis[pub & ~dead].mean()), "P(visible | D)": float(vis[pub & dead].mean()),
    }

    # --- 2. per-gap table
    key = np.where(g, (~vis).astype(int) + 10 * m.half, -1)
    s, e, v = runs(key)
    is_gap = (v % 10 == 1) & (v >= 10)
    B = m.B[:, :2]
    rows = []
    for a, b in zip(s[is_gap], e[is_gap]):
        n = b - a
        before = a - 1
        info = {"start": a, "frames": n, "seconds": n / FPS, "half": int(m.half[a])}
        if before >= 0 and vis[before]:
            x, y = B[before]
            info["edge_dist_m"] = float(min(x, L - x, y, W - y))
            info["outside_pitch"] = bool(x < 0 or x > L or y < 0 or y > W)
            in_box = (x < 16.5 or x > L - 16.5) and abs(y - W / 2) < 20.16
            info["zone"] = "box" if in_box else ("wing" if min(y, W - y) < 13.6 else "central")
            d = np.linalg.norm(m.P[before, :, :2] - B[before], axis=1)
            info["players_within_3m"] = int(np.nansum(d < 3))
            info["nearest_dist_m"] = float(np.nanmin(d)) if np.isfinite(d).any() else np.nan
            if before >= 3 and vis[before - 3]:
                info["ball_speed_ms"] = float(np.linalg.norm(B[before] - B[before - 3]) * FPS / 3)
        info["team_before"] = int(nearest[before]) if before >= 0 else -1
        info["team_after"] = int(nearest[b]) if b < m.n_frames else -1
        info["blackout_share"] = float(blackout[a:b].mean())
        info["c3_share"] = float(c3[a:b].mean())
        info["slow_share"] = float(slow[a:b].mean())
        info["phase_share"] = float(phase_cover[a:b].mean())
        info["setpiece_inside"] = bool(setpiece_cover[a:b].any())
        rows.append(info)
    gaps = pd.DataFrame(rows)
    gaps["bucket"] = gaps["seconds"].map(bucket_of)
    gaps["team_same"] = np.where((gaps["team_before"] >= 0) & (gaps["team_after"] >= 0),
                                 (gaps["team_before"] == gaps["team_after"]).astype(float), np.nan)
    gaps["context"] = np.where(gaps["blackout_share"] >= 0.5, "blackout",
                               np.where(gaps["slow_share"] >= 0.5, "slow (stoppage proxy)", "moving play"))
    by_context = gaps.groupby("context").agg(gaps=("frames", "size"), minutes=("frames", lambda x: x.sum() / FPS / 60),
                                             median_s=("seconds", "median"),
                                             team_same_pct=("team_same", lambda x: 100 * np.nanmean(x.astype(float))))
    gap_frames_total = int(gaps["frames"].sum())

    by_bucket = gaps.groupby("bucket", sort=False).agg(
        gaps=("frames", "size"), frames=("frames", "sum"),
        ball_speed_ms=("ball_speed_ms", "median"), players_within_3m=("players_within_3m", "median"),
        edge_dist_m=("edge_dist_m", "median"), near_edge_pct=("edge_dist_m", lambda x: 100 * np.mean(x < 3)),
        outside_pct=("outside_pitch", lambda x: 100 * np.nanmean(x.astype(float))),
        team_same_pct=("team_same", lambda x: 100 * np.nanmean(x)),
        blackout_pct=("blackout_share", lambda x: 100 * np.mean(x)),
        slow_pct=("slow_share", lambda x: 100 * np.mean(x)),
        c3_pct=("c3_share", lambda x: 100 * np.mean(x)),
        setpiece_inside_pct=("setpiece_inside", lambda x: 100 * np.mean(x)),
    ).reindex(BUCKET_NAMES)
    by_bucket["pct_of_gap_frames"] = 100 * by_bucket["frames"] / gap_frames_total

    # --- 3. where do missing-ball frames fall (graded frames)
    missing = g & ~vis
    split = {
        "graded_frames": int(g.sum()), "missing_frames": int(missing.sum()),
        "missing_pct_of_graded": 100 * float(missing.sum() / g.sum()),
        "of_missing: blackout": 100 * float(blackout[missing].mean()),
        "of_missing: slow (stoppage proxy), not blackout": 100 * float((slow & ~blackout)[missing].mean()),
        "of_missing: moving play, not blackout": 100 * float((~slow & ~blackout)[missing].mean()),
        "P(visible | moving, no blackout)": float(vis[g & ~slow & ~blackout].mean()),
        "P(visible | slow, no blackout)": float(vis[g & slow & ~blackout].mean()),
        "P(visible | blackout)": float(vis[g & blackout].mean()),
        "share of graded frames: moving, no blackout": float((g & ~slow & ~blackout).mean() / g.mean()),
        "missing frames in moving play (count)": int((missing & ~slow & ~blackout).sum()),
    }
    c3_frames = g & c3
    c3_split = {"c3_only_frames_graded": int(c3_frames.sum()),
                "of_c3: blackout": 100 * float(blackout[c3_frames].mean()),
                "of_c3: slow": 100 * float(slow[c3_frames].mean()),
                "of_c3: inside gaps >10 s": 100 * float(np.isin(np.arange(m.n_frames)[c3_frames],
                                                                np.concatenate([np.arange(a, a + n) for a, n in zip(gaps.loc[gaps.seconds > 10, "start"], gaps.loc[gaps.seconds > 10, "frames"])]) if (gaps.seconds > 10).any() else []).mean())}

    # --- 4. public transitions inside gaps (from S1.1)
    trans = metrics.pm.transitions(gt, metrics.pm.valid_runs(gt))
    in_gap = [(f, a_, b_) for f, a_, b_ in trans if not vis[f]]

    # --- 5. degradation parameters (aggregates only)
    moving = gaps[gaps["slow_share"] < 0.5]
    hist_edges = [0, 1, 2, 3, 5, 8, 12, 25, 50, 100, 250, 500, 1000, 5000, 100000]
    params = {
        "source": "BAR-ATH ATD, graded frames, S1.3",
        "gap_length_frames_hist": {"edges": hist_edges,
                                   "all": np.histogram(gaps["frames"], hist_edges)[0].tolist(),
                                   "moving_play": np.histogram(moving["frames"], hist_edges)[0].tolist()},
        "gap_start_rate_per_visible_frame": float(len(gaps) / max(int((g & vis).sum()), 1)),
        "gap_start_rate_moving_play": float(len(moving) / max(int((g & vis & ~slow).sum()), 1)),
        "p_visible_moving_play": split["P(visible | moving, no blackout)"],
        "p_visible_slow": split["P(visible | slow, no blackout)"],
        "blackout_share_of_graded": float(blackout[g].mean()),
        "c3_only_share_of_graded": float(c3[g].mean()),
        "context_at_gap_start_median": {
            "ball_speed_ms": float(gaps["ball_speed_ms"].median()),
            "players_within_3m": float(gaps["players_within_3m"].median()),
            "edge_dist_m": float(gaps["edge_dist_m"].median())},
        "visible_reference_median": {},
    }
    # reference: context on visible frames (to see what makes a gap start likelier)
    vrows = np.flatnonzero(g & vis)
    vrows = vrows[vrows >= 3]
    vrows = vrows[vis[vrows - 3]]
    sp_v = np.linalg.norm(B[vrows] - B[vrows - 3], axis=1) * FPS / 3
    d_v = np.linalg.norm(m.P[vrows, :, :2] - B[vrows, None, :], axis=-1)
    params["visible_reference_median"] = {"ball_speed_ms": float(np.median(sp_v)),
                                          "players_within_3m": float(np.median(np.nansum(d_v < 3, axis=1)))}
    speed_bins = [0, 2, 5, 10, 15, 25, 60]
    gap_start_speed = gaps["ball_speed_ms"].dropna().to_numpy()
    params["gap_start_rate_by_ball_speed"] = {
        "edges_ms": speed_bins,
        "rate": (np.histogram(gap_start_speed, speed_bins)[0] / np.maximum(np.histogram(sp_v, speed_bins)[0], 1)).tolist()}
    crowd_bins = [0, 1, 2, 3, 5, 30]
    params["gap_start_rate_by_players_within_3m"] = {
        "edges": crowd_bins,
        "rate": (np.histogram(gaps["players_within_3m"].dropna(), crowd_bins)[0]
                 / np.maximum(np.histogram(np.nansum(d_v < 3, axis=1), crowd_bins)[0], 1)).tolist()}

    (ROOT / "local_data").mkdir(exist_ok=True)
    (ROOT / "local_data" / "degradation_params.json").write_text(json.dumps(params, indent=2))
    (args.out / "degradation_params.json").write_text(json.dumps(params, indent=2))
    by_bucket.to_csv(args.out / "gaps_by_length.csv", float_format="%.2f")
    by_context.to_csv(args.out / "gaps_by_context.csv", float_format="%.2f")
    (args.out / "summary.json").write_text(json.dumps(
        {"proxy_validation_public": proxy_valid, "missing_split": split, "c3": c3_split,
         "n_gaps": len(gaps), "gap_frames": gap_frames_total,
         "gap_seconds_quantiles": np.percentile(gaps["seconds"], [25, 50, 75, 90, 99]).round(2).tolist(),
         "public_transitions_in_gaps": [(int(f), int(a_), int(b_)) for f, a_, b_ in in_gap]}, indent=2))
    plot(gaps, speed2s, vis, slow, blackout, g, m, args.out / "ball_gaps.png")

    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.1f}".format):
        print("Stoppage proxy on the public window:", json.dumps(proxy_valid, indent=1))
        print("\nGaps: {} gaps, {} frames; seconds quantiles 25/50/75/90/99: {}".format(
            len(gaps), gap_frames_total, np.percentile(gaps["seconds"], [25, 50, 75, 90, 99]).round(2).tolist()))
        print(by_bucket)
        print(by_context)
        print("\nWhere missing-ball frames fall:", json.dumps(split, indent=1))
        print("Third channel:", json.dumps(c3_split, indent=1))
        print("Public transitions inside gaps:", in_gap)
        print("Gap start rate by ball speed:", params["gap_start_rate_by_ball_speed"])
        print("Gap start rate by players within 3 m:", params["gap_start_rate_by_players_within_3m"])
        print("Visible reference:", params["visible_reference_median"])
    return 0


def plot(gaps, speed2s, vis, slow, blackout, g, m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["ink"],
                         "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"]})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    ax = axes[0]
    edges = np.logspace(np.log10(0.04), np.log10(600), 30)
    ax.hist(gaps["seconds"], bins=edges, weights=gaps["frames"] / FPS / 60, color=COLORS["main"], rwidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("gap length (s, log)")
    ax.set_ylabel("minutes of missing ball")
    ax.set_title("Missing ball by gap length (minutes)", loc="left")

    ax = axes[1]
    cats = ["moving play", "slow (stoppage proxy)", "blackout (<12 slots)"]
    masks = [g & ~slow & ~blackout, g & slow & ~blackout, g & blackout]
    vis_rate = [100 * vis[mk].mean() for mk in masks]
    minutes = [mk.sum() / FPS / 60 for mk in masks]
    bars = ax.bar(cats, vis_rate, color=[COLORS["main"], COLORS["alt"], COLORS["ink"]], width=0.6)
    for b, r, mi in zip(bars, vis_rate, minutes):
        ax.text(b.get_x() + b.get_width() / 2, r + 1.5, "{:.0f}%\n({:.0f} min)".format(r, mi), ha="center", fontsize=8,
                color=COLORS["ink"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("% frames with ball x,y")
    ax.set_title("Ball visible, by context (graded frames)", loc="left")

    ax = axes[2]
    sp = gaps["ball_speed_ms"].dropna()
    ax.hist(sp.clip(0, 40), bins=np.arange(0, 41, 2), color=COLORS["main"], rwidth=0.9, density=True, label="at gap start")
    ax.set_xlabel("ball speed just before the gap (m/s, clipped at 40)")
    ax.set_ylabel("density")
    ax.set_title("Ball speed when it disappears", loc="left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=COLORS["grid"], lw=0.6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
