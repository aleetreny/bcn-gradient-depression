"""S0.2 check: load BAR-ATH with src.data.metrica_atd and compare against DATA.md.

    .venv311/bin/python scripts/report_s0_2_bar_ath.py [--out experiments/outputs/002]

Writes summary.json and overview.png; prints the comparison table. Only aggregates are written,
never frame-level data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import asset_paths, half_video_blocks  # noqa: E402
from src.data.metrica_atd import HOME, load  # noqa: E402

FPS = 25
KLOPPY_EVERY = 50  # kloppy sample: one frame in 50 across the whole match (~4000 frames)
COLORS = {"H": "#2a78d6", "A": "#eb6834", "ink": "#52514e", "grid": "#e4e3df", "band": "#f1f0ec"}


def finite_diff_speed_ratio(xy, speed, max_step=1.5):
    """Median of reported speed / speed from consecutive positions, both in m/s.

    Only pairs where both frames are visible; steps > max_step m/frame (tracker jumps) dropped.
    """
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * FPS
    reported = speed[1:]
    ok = np.isfinite(step) & np.isfinite(reported) & (step > 0.5) & (step < max_step * FPS)
    return float(np.median(reported[ok] / step[ok])), int(ok.sum())


def kloppy_crosscheck(match, every):
    """Max |ours - kloppy| on one frame in `every` across the match: x, y normalised; speed raw."""
    from kloppy import metrica
    paths = asset_paths()
    ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"],
                                    sample_rate=1 / every, coordinates="metrica")
    df = ds.to_df()
    rows = df["frame_id"].to_numpy(np.int64) - 1
    length, width = match.pitch
    ours_ball = match.B[rows, :2] / np.array([length, width])
    diffs = [np.nanmax(np.abs(ours_ball[:, 0] - df["ball_x"].to_numpy()), initial=0),
             np.nanmax(np.abs(ours_ball[:, 1] - df["ball_y"].to_numpy()), initial=0)]
    speed_diffs = []
    nan_agree = np.array_equal(np.isnan(ours_ball[:, 0]), df["ball_x"].isna().to_numpy())
    player_ids = [p.player_id for team in ds.metadata.teams for p in team.players]
    for pid in player_ids:
        s = int(pid) - 1
        ox, oy = match.P[rows, s, 0] / length, match.P[rows, s, 1] / width
        kx, ky = df["{}_x".format(pid)].to_numpy(), df["{}_y".format(pid)].to_numpy()
        nan_agree &= np.array_equal(np.isnan(ox), np.isnan(kx))
        diffs += [np.nanmax(np.abs(ox - kx), initial=0), np.nanmax(np.abs(oy - ky), initial=0)]
        speed_diffs.append(np.nanmax(np.abs(match.P[rows, s, 2] - df["{}_s".format(pid)].to_numpy()), initial=0))
    return {"frames": len(rows), "graded_frames": int(match.graded[rows].sum()),
            "visible_slot_values": int(np.isfinite(match.P[rows, :, 0]).sum()), "max_abs_diff_normalized": float(max(diffs)),
            "max_abs_diff_speed": float(max(speed_diffs)),
            "nan_pattern_identical": bool(nan_agree), "n_slots_compared": len(player_ids)}


def plot_overview(match, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_min = FPS * 60
    n_min = match.n_frames // per_min
    minute = np.arange(n_min)
    trim = n_min * per_min
    ball = match.ball_xy_visible[:trim].reshape(n_min, per_min).mean(1) * 100
    speed_only = match.ball_speed_only[:trim].reshape(n_min, per_min).mean(1) * 100
    vis = match.slot_visible[:trim]
    team_counts = {k: vis[:, match.slot_team == code].sum(1).reshape(n_min, per_min).mean(1)
                   for k, code in (("H", 0), ("A", 1))}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["ink"],
                         "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"]})
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for ax in axes:
        for _, first, last in half_video_blocks("BAR-ATH"):
            ax.axvspan(first / per_min, last / per_min, color=COLORS["band"], zorder=0, lw=0)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].plot(minute, ball, color=COLORS["H"], lw=2)
    axes[0].set_ylabel("% frames")
    axes[0].set_title("Ball position visible, per minute of video (shaded = graded halves)", loc="left")
    axes[0].set_ylim(0, 100)

    axes[1].plot(minute, speed_only, color=COLORS["ink"], lw=2)
    axes[1].set_ylabel("% frames")
    axes[1].set_title("Ball third channel without x, y ('speed only' in the plan; not a speed), per minute", loc="left")

    for k in ("H", "A"):
        axes[2].plot(minute, team_counts[k], color=COLORS[k], lw=2)
        axes[2].text(minute[-1] + 0.5, team_counts[k][-15:].mean(), k, color=COLORS[k], va="center")
    axes[2].axhline(11, color=COLORS["ink"], lw=0.8, ls=":")
    axes[2].set_ylabel("visible slots")
    axes[2].set_title("Mean visible slots per team, per minute (dotted = 11 players)", loc="left")
    axes[2].set_xlabel("video minute")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/002")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = load()
    n = m.normalized()
    g = m.graded
    bxy, bso = m.ball_xy_visible, m.ball_speed_only
    bspeed = np.isfinite(m.B[:, 2])
    objects = m.slot_visible.sum(1)

    s = {
        "frames": m.n_frames,
        "graded_frames": int(g.sum()),
        "slots": int(m.P.shape[1]),
        "slots_H": int((m.slot_team == HOME).sum()),
        "slots_A": int((m.slot_team != HOME).sum()),
        "gk_slots_1based": (np.flatnonzero(m.slot_is_gk) + 1).tolist(),
        "gk_team": ["H" if m.slot_team[i] == HOME else "A" for i in np.flatnonzero(m.slot_is_gk)],
        "home": m.home_name, "away": m.away_name,
        "ball_xy_frames": int(bxy.sum()),
        "ball_xy_pct": 100 * bxy.mean(),
        "ball_xy_graded_frames": int(bxy[g].sum()),
        "ball_xy_graded_pct": 100 * bxy[g].mean(),
        "objects_mean_all": float(objects.mean()),
        "objects_mean_graded": float(objects[g].mean()),
        "objects_max": int(objects.max()),
        "objects_mean_outside_halves": float(objects[~g].mean()),
        "ball_patterns_pct": {
            "xy_no_speed": 100 * (bxy & ~bspeed).mean(),
            "xy_and_speed": 100 * (bxy & bspeed).mean(),
            "nothing": 100 * (~bxy & ~bspeed).mean(),
            "speed_only": 100 * bso.mean(),
        },
        "ball_speed_only_graded_pct": 100 * bso[g].mean(),
    }

    pr, pn = finite_diff_speed_ratio(m.P[..., :2], m.P[..., 2])
    br, bn = finite_diff_speed_ratio(m.B[:, :2], m.B[:, 2])
    s["speed_check"] = {
        "player_reported_over_fd_median": pr, "player_pairs": pn,
        "ball_reported_over_fd_median": br, "ball_pairs": bn,
        "ball_speed_ms_quantiles_when_xy": np.nanpercentile(m.B[bxy, 2], [5, 50, 95]).tolist(),
        "ball_speed_ms_quantiles_when_speed_only": np.nanpercentile(m.B[bso, 2], [5, 50, 95]).tolist(),
        "player_speed_ms_p50_p99": np.nanpercentile(m.P[g][..., 2], [50, 99]).tolist(),
    }

    direction = {}
    length = m.pitch[0]
    for half, first, last in half_video_blocks("BAR-ATH"):
        for i in np.flatnonzero(m.slot_is_gk):
            team = "H" if m.slot_team[i] == HOME else "A"
            direction["half{}_gk_{}_raw_mean_x".format(half, team)] = float(np.nanmean(m.P[first:last + 1, i, 0]))
            direction["half{}_gk_{}_norm_mean_x".format(half, team)] = float(np.nanmean(n.P[first:last + 1, i, 0]))
        for team, code in (("H", 0), ("A", 1)):
            xs = n.P[first:last + 1][:, m.slot_team == code, 0]
            direction["half{}_{}_norm_centroid_x".format(half, team)] = float(np.nanmean(xs))
    direction["pitch_length"] = length
    s["direction"] = direction

    s["kloppy_crosscheck"] = kloppy_crosscheck(m, KLOPPY_EVERY)
    plot_overview(m, args.out / "overview.png")
    (args.out / "summary.json").write_text(json.dumps(s, indent=2))

    expected = [
        ("frames", s["frames"], 199319),
        ("graded frames", s["graded_frames"], 158857),
        ("slots (H / A)", "{} ({} / {})".format(s["slots"], s["slots_H"], s["slots_A"]), "46 (24 / 22)"),
        ("ball x,y visible, all", "{} ({:.1f}%)".format(s["ball_xy_frames"], s["ball_xy_pct"]), "113488 (56.9%)"),
        ("ball x,y visible, graded", "{} ({:.1f}%)".format(s["ball_xy_graded_frames"], s["ball_xy_graded_pct"]), "94214 (59.3%)"),
        ("mean objects, graded", "{:.2f}".format(s["objects_mean_graded"]), "20.5"),
        ("mean objects, all", "{:.2f}".format(s["objects_mean_all"]), "18.98 (doc 06)"),
    ]
    print("{:<28} {:<22} {}".format("check", "ours", "DATA.md / doc 06"))
    for name, ours, ref in expected:
        print("{:<28} {:<22} {}".format(name, str(ours), ref))
    print(json.dumps({k: s[k] for k in ("ball_patterns_pct", "speed_check", "direction", "kloppy_crosscheck",
                                       "gk_slots_1based", "gk_team", "objects_max")}, indent=1))
    k = s["kloppy_crosscheck"]
    checks = {
        "frames": s["frames"] == 199319, "graded frames": s["graded_frames"] == 158857,
        "slots 24/22": (s["slots_H"], s["slots_A"]) == (24, 22),
        "ball visible all = 113488": s["ball_xy_frames"] == 113488,
        "ball visible graded = 94214": s["ball_xy_graded_frames"] == 94214,
        "mean objects graded ~ 20.5": abs(s["objects_mean_graded"] - 20.5) < 0.05,
        "kloppy identical": k["nan_pattern_identical"] and k["max_abs_diff_normalized"] < 1e-6 and k["max_abs_diff_speed"] == 0,
        "player speed is m/s": abs(s["speed_check"]["player_reported_over_fd_median"] - 1) < 0.05,
        "H goalkeeper on low x after normalising": all(s["direction"]["half{}_gk_H_norm_mean_x".format(h)] < 30 for h in (1, 2)),
    }
    failed = [name for name, ok in checks.items() if not ok]
    print("\nCHECKS:", "OK" if not failed else "FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
