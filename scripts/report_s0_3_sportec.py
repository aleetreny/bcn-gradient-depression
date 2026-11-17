"""S0.3 check: load the 7 Sportec matches and describe their H/A/D labels next to BAR-ATH public GT.

    .venv311/bin/python scripts/report_s0_3_sportec.py [--out experiments/outputs/003] [--no-kloppy]

Writes per_match.csv, summary.json and spell_durations.png.
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
from evaluation.possession_metrics import read_label_csv  # noqa: E402
from games import half_video_blocks  # noqa: E402
from src.data.common import AWAY, DEAD, HOME, label_stats  # noqa: E402
from src.data.sportec import DATA_DIR, _file, load_all  # noqa: E402

PUBLIC_GT = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
COLORS = {"sportec": "#2a78d6", "public": "#eb6834", "ink": "#52514e", "grid": "#e4e3df"}


def public_gt_stats():
    frame = read_label_csv(str(PUBLIC_GT))
    codes = frame["label"].map({"H": HOME, "A": AWAY, "D": DEAD}).fillna(-1).to_numpy(np.int8)
    half = np.zeros(len(codes), np.int8)
    for h, first, last in half_video_blocks("BAR-ATH"):
        half[first:last + 1] = h
    return label_stats(codes, half)


def kloppy_crosscheck(match, every=25):
    """Our arrays vs kloppy's sportec loader on one frame in `every`: H/A/D label, ball x/y and
    every player's x/y (matched by DFL PersonId). Positions compared in kloppy's centred metres."""
    from kloppy import sportec
    mid = match.meta["match_id"]
    ds = sportec.load_tracking(raw_data=str(_file("04_03_positions_raw_observed", mid, DATA_DIR)),
                               meta_data=str(_file("02_01_matchinformation", mid, DATA_DIR)),
                               sample_rate=1 / every, coordinates="sportec", only_alive=False)
    home_id = ds.metadata.teams[0].team_id
    n1 = int(np.sum(match.half == 1))
    slot_of = {p: s for s, p in enumerate(match.meta["slot_person"].tolist())}
    half_l, half_w = match.pitch[0] / 2, match.pitch[1] / 2
    rows, k_lab, ball_d, player_d = [], [], [], []
    nan_mismatch = n_player_values = 0
    for r in ds.records:
        h = r.period.id
        row = (0 if h == 1 else n1) + (r.frame_id - match.meta["frame_n"][0 if h == 1 else n1])
        if not 0 <= row < match.n_frames:
            continue
        rows.append(row)
        k_lab.append(DEAD if r.ball_state.value == "dead" else
                     (HOME if r.ball_owning_team.team_id == home_id else AWAY))
        if r.ball_coordinates is not None:
            ball_d.append(max(abs(match.B[row, 0] - half_l - r.ball_coordinates.x),
                              abs(match.B[row, 1] - half_w - r.ball_coordinates.y)))
        seen = set()
        for player, pdata in r.players_data.items():
            s = slot_of.get(player.player_id)
            if s is None or pdata.coordinates is None:
                continue
            seen.add(s)
            n_player_values += 1
            player_d.append(max(abs(match.P[row, s, 0] - half_l - pdata.coordinates.x),
                                abs(match.P[row, s, 1] - half_w - pdata.coordinates.y)))
        visible = set(np.flatnonzero(match.slot_visible[row]).tolist())
        nan_mismatch += len(visible ^ seen)
    rows = np.array(rows)
    return {"match_id": mid, "frames_compared": len(rows),
            "label_agreement": float(np.mean(match.labels[rows] == np.array(k_lab))),
            "max_abs_ball_diff_m": float(np.nanmax(ball_d)),
            "player_values_compared": n_player_values,
            "max_abs_player_diff_m": float(np.nanmax(player_d)),
            "visibility_mismatches": int(nan_mismatch)}


def plot_durations(durs, public_dur, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["ink"],
                         "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"]})
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for i, (mid, d) in enumerate(durs.items()):
        x = np.sort(d)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COLORS["sportec"], lw=1.2,
                alpha=0.45, label="Sportec, one line per match (7)" if i == 0 else None)
    x = np.sort(public_dur)
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COLORS["public"], lw=2.2,
            label="BAR-ATH public 5 min (17 possessions)")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 200)
    ax.axvline(2, color=COLORS["ink"], lw=0.8, ls=":")
    ax.text(2.1, 0.03, "2 s", color=COLORS["ink"])
    ax.set_xlabel("possession duration (s, log scale)")
    ax.set_ylabel("share of possessions ≤ x")
    ax.set_title("Possession duration, cumulative: Sportec labels vs BAR-ATH public ground truth", loc="left")
    ax.grid(color=COLORS["grid"], lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/003")
    parser.add_argument("--no-kloppy", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    matches = load_all()
    rows, durs, checks = [], {}, {}
    for mid, m in matches.items():
        st, dur = label_stats(m.labels, m.half, m.ball_xy_visible)
        n = m.normalized()
        gk = np.flatnonzero(m.slot_is_gk)
        gk_x = {("H" if m.slot_team[i] == HOME else "A"): float(np.nanmean(n.P[:, i, 0])) for i in gk}
        st.update({"match_id": mid, "home": m.home_name, "away": m.away_name, "slots": m.P.shape[1],
                   "mean_visible_slots": float(m.slot_visible.sum(1).mean()),
                   "unlabelled_frames": int(np.sum(m.labels < 0)),
                   "attack_sign_h1_h2": "{:+d}/{:+d}".format(m.attack_sign[m.half == 1][0], m.attack_sign[m.half == 2][0]),
                   "gk_norm_x_H": gk_x.get("H", np.nan), "gk_norm_x_A": gk_x.get("A", np.nan)})
        rows.append(st)
        durs[mid] = dur
        checks[mid] = m.meta["raw_speed_check"]

    df = pd.DataFrame(rows).set_index("match_id")
    public, public_dur = public_gt_stats()
    pooled = np.concatenate(list(durs.values()))
    summary = {
        "public_gt": public,
        "sportec_pooled": {"possessions": int(len(pooled)), "poss_mean_s": float(pooled.mean()),
                           "poss_median_s": float(np.median(pooled)), "pct_poss_lt_2s": 100 * float(np.mean(pooled < 2)),
                           "possessions_per_min_mean": float(df["possessions_per_min"].mean()),
                           "pct_D_mean": float(df["pct_D"].mean())},
        "dfl_raw_speed_check": checks,
    }
    failures = []
    if not args.no_kloppy:
        summary["kloppy_crosscheck"] = [kloppy_crosscheck(m) for m in matches.values()]
        for c in summary["kloppy_crosscheck"]:
            if (c["label_agreement"] < 1 or c["max_abs_ball_diff_m"] > 0.01 or c["max_abs_player_diff_m"] > 0.01
                    or c["visibility_mismatches"]):
                failures.append("kloppy mismatch in {}".format(c["match_id"]))
    failures += ["{} has unlabelled frames".format(mid) for mid in df.index if df.loc[mid, "unlabelled_frames"]]
    df.to_csv(args.out / "per_match.csv", float_format="%.3f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plot_durations(durs, public_dur, args.out / "spell_durations.png")

    cols = ["minutes", "pct_D", "possessions", "possessions_per_min", "poss_mean_s", "poss_median_s",
            "pct_poss_lt_2s", "dead_runs", "dead_mean_s", "pct_changes_direct", "pct_ball_visible",
            "slots", "unlabelled_frames", "attack_sign_h1_h2", "gk_norm_x_H", "gk_norm_x_A"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(df[cols])
        print("\nBAR-ATH public GT:", {k: round(v, 2) if isinstance(v, float) else v for k, v in public.items()})
    print(json.dumps({k: v for k, v in summary.items() if k != "public_gt"}, indent=1, default=float))
    print("\nCHECKS:", "OK" if not failures else "FAILED: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
