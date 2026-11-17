"""S1.6: is Smart Tagging usable as an anchor (restart timing, team)?

    .venv311/bin/python scripts/s1_6_smart_tagging.py [--out experiments/outputs/012]

1. Restart timing, physically, over the whole match: for each SET PIECES instance, the kick is the
   first frame in [start - 10 s, end + 10 s] where the ball leaves the expected spot (corner arc,
   touchline, goal area, centre spot) faster than 5 m/s for 3 frames. Offsets kick - tag start and
   kick - tag end, by Type. Public window: tag starts vs the four true D ends.
2. Team: share of frames where the team in possession implied by a phase (its Team tag, or the other
   team for "Defending ..." and "Defensive Transition", which carry the defending side) equals (a) the public truth,
   (b) the full-match last-touch rule (r 3 m, theta 20, delta 3; S1.2), by phase code. N/A separately.
3. Coverage: graded time inside any phase / any set piece; ball visibility and the stoppage proxy
   (S1.3) inside vs outside phases.
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
from src.data.common import AWAY, DEAD, FPS, HOME, runs  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.possession_rules import Kinematics, rule_last_touch  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402

TEAM = {"Barcelona": HOME, "Atletic Club": AWAY}
DEFENSIVE_CODES = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD",
                   "DEFENSIVE TRANSITION"}
SLOW = 1.2


def expected_spot(kind, x, y, L, W):
    """Is the ball at the place this restart is taken from? (raw coordinates, either end)"""
    if kind.startswith("Corner"):
        return min(np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)) < 3.0
    if kind.startswith("Throw"):
        return min(y, W - y) < 2.5
    if kind.startswith("Goal Kick"):
        return (min(x, L - x) < 7.0) and abs(y - W / 2) < 11.0
    if kind.startswith("Kick Off"):
        return np.hypot(x - L / 2, y - W / 2) < 3.0
    return True  # free kicks: anywhere


def ball_speed(m):
    B = m.B[:, :2]
    v = m.ball_xy_visible
    s = np.full(m.n_frames, np.nan)
    ok = v[2:] & v[:-2]
    s[1:-1][ok] = np.linalg.norm(B[2:][ok] - B[:-2][ok], axis=1) * FPS / 2
    return s


def find_kick(m, speed, a, b, kind):
    L, W = m.pitch
    for t in range(max(a, 2), min(b, m.n_frames - 3)):
        if not m.ball_xy_visible[t - 1]:
            continue
        if np.all(speed[t:t + 3] > 5) and expected_spot(kind, *m.B[t - 1, :2], L, W):
            return t
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/012")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = load_atd()
    g = m.graded
    tags = load_tags()
    speed = ball_speed(m)

    # 1. restart timing
    sp_rows = []
    for t in tags:
        if t["code"].upper() != "SET PIECES":
            continue
        kind = t["tags"].get("Type", "?")
        k = find_kick(m, speed, t["start_frame"] - 10 * FPS, t["end_frame"] + 10 * FPS, kind)
        sp_rows.append({"type": kind, "team": t["tags"].get("Team", "?"), "start": t["start_frame"], "end": t["end_frame"],
                        "length_s": (t["end_frame"] - t["start_frame"]) / FPS, "graded": bool(g[t["start_frame"]]),
                        "kick": k, "kick_minus_start": None if k is None else k - t["start_frame"],
                        "kick_minus_end": None if k is None else k - t["end_frame"],
                        "ball_visible_at_start": bool(m.ball_xy_visible[t["start_frame"]])})
    sp = pd.DataFrame(sp_rows)
    sp_g = sp[sp["graded"]]
    by_type = sp_g.groupby("type").agg(n=("start", "size"), found=("kick", lambda x: int(x.notna().sum())),
                                       length_s=("length_s", "median"),
                                       kick_minus_start_p25=("kick_minus_start", lambda x: x.dropna().quantile(.25)),
                                       kick_minus_start_med=("kick_minus_start", "median"),
                                       kick_minus_start_p75=("kick_minus_start", lambda x: x.dropna().quantile(.75)),
                                       team_na_pct=("team", lambda x: 100 * np.mean(x == "N/A")))

    gt = metrics.public_gt()
    gt = np.where(gt == 3, -1, gt).astype(np.int8)
    key = np.where(gt >= 0, gt, -1).astype(int)
    s, e, v = runs(key)
    d_ends = [int(s[i]) for i in range(1, len(s)) if v[i - 1] == DEAD and v[i] in (HOME, AWAY)]
    starts = sp["start"].to_numpy()
    public_restarts = [{"true_restart": f, "nearest_set_piece_start_minus_restart": int(starts[np.argmin(np.abs(starts - f))] - f),
                        "type": sp.iloc[int(np.argmin(np.abs(starts - f)))]["type"]} for f in d_ends]

    # 2. team agreement
    kin = Kinematics.from_match(m)
    touch = rule_last_touch(kin, 3.0, 20, 3.0)
    team_rows = []
    for t in tags:
        team = t["tags"].get("Team")
        a, b = t["start_frame"], t["end_frame"]
        sl = slice(a, b)
        alive_pub = np.isin(gt[sl], (HOME, AWAY))
        row = {"code": t["code"].title(), "team_tag": team, "frames": b - a,
               "graded_frames": int(g[sl].sum())}
        if team in TEAM:
            # "Defending ..." and "Defensive Transition" carry the defending team: possession is the other one
            code = 1 - TEAM[team] if t["code"].upper() in DEFENSIVE_CODES else TEAM[team]
            row["pub_alive_frames"] = int(alive_pub.sum())
            row["pub_agree_frames"] = int((gt[sl][alive_pub] == code).sum())
            known = g[sl] & (touch[sl] >= 0)
            row["touch_frames"] = int(known.sum())
            row["touch_agree_frames"] = int((touch[sl][known] == code).sum())
        team_rows.append(row)
    tm = pd.DataFrame(team_rows)
    tagged = tm[tm["team_tag"].isin(TEAM)]
    by_code = tagged.groupby("code").agg(instances=("frames", "size"),
                                         touch_agree=("touch_agree_frames", "sum"), touch_frames=("touch_frames", "sum"),
                                         pub_agree=("pub_agree_frames", "sum"), pub_frames=("pub_alive_frames", "sum"))
    by_code["touch_agreement"] = by_code["touch_agree"] / by_code["touch_frames"]
    by_code["public_agreement"] = by_code["pub_agree"] / by_code["pub_frames"].replace(0, np.nan)
    by_code = by_code.sort_values("touch_frames", ascending=False)
    overall = {"touch_agreement_all_phases": float(tagged["touch_agree_frames"].sum() / tagged["touch_frames"].sum()),
               "public_agreement_all_phases": float(tagged["pub_agree_frames"].sum() / max(tagged["pub_alive_frames"].sum(), 1)),
               "public_alive_frames_covered": int(tagged["pub_alive_frames"].sum()),
               "na_instances": int((tm["team_tag"] == "N/A").sum()),
               "na_codes": tm.loc[tm["team_tag"] == "N/A", "code"].value_counts().to_dict()}
    # reference: the touch rule itself against the public truth (alive frames)
    alive = np.isin(gt, (HOME, AWAY)) & (touch >= 0)
    overall["touch_rule_vs_public_truth"] = float(np.mean(touch[alive] == gt[alive]))

    # 3. coverage
    phase = np.zeros(m.n_frames, bool)
    setp = np.zeros(m.n_frames, bool)
    for t in tags:
        (setp if t["code"].upper() == "SET PIECES" else phase)[t["start_frame"]:t["end_frame"]] = True
    with np.errstate(all="ignore"):
        ts = np.nanmean(np.where(m.slot_visible, m.P[..., 2], np.nan), axis=1)
    k = np.ones(2 * FPS) / (2 * FPS)
    slow = np.convolve(np.nan_to_num(ts), k, "same") < SLOW
    coverage = {"graded_in_any_phase_pct": 100 * float(phase[g].mean()), "graded_in_set_piece_pct": 100 * float(setp[g].mean()),
                "graded_in_nothing_pct": 100 * float((~phase & ~setp)[g].mean()),
                "ball_visible_inside_phases": float(m.ball_xy_visible[g & phase].mean()),
                "ball_visible_outside_tags": float(m.ball_xy_visible[g & ~phase & ~setp].mean()),
                "slow_inside_phases": float(slow[g & phase].mean()), "slow_outside_tags": float(slow[g & ~phase & ~setp].mean()),
                "public_D_inside_phases": float(np.mean(gt[phase & (gt >= 0)] == DEAD)),
                "public_D_outside_tags": float(np.mean(gt[~phase & ~setp & (gt >= 0)] == DEAD))}

    sp.to_csv(args.out / "set_pieces.csv", index=False)
    by_type.to_csv(args.out / "set_pieces_by_type.csv", float_format="%.1f")
    by_code.to_csv(args.out / "team_agreement_by_code.csv", float_format="%.3f")
    (args.out / "summary.json").write_text(json.dumps({"public_restarts": public_restarts, "team": overall, "coverage": coverage},
                                                     indent=2, default=str))
    plot(sp_g, args.out / "set_piece_offsets.png")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(by_type)
        print("\nPublic restarts vs nearest SET PIECES start:", public_restarts)
        print(by_code[["instances", "touch_frames", "touch_agreement", "pub_frames", "public_agreement"]])
        print(json.dumps({"team": overall, "coverage": coverage}, indent=1, default=str))
    return 0


def plot(sp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid = "#52514e", "#e4e3df"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    types = [t for t in sp["type"].value_counts().index]
    for i, t in enumerate(types):
        d = sp[(sp["type"] == t)]["kick_minus_start"].dropna() / FPS
        ax.scatter(d, np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), s=18, color=colors[i % 6])
    ax.set_yticks(range(len(types)), ["{} ({} found / {})".format(t, int(sp[sp.type == t].kick.notna().sum()), int((sp.type == t).sum())) for t in types])
    ax.axvline(0, color=ink, ls=":")
    ax.set_xlabel("physical kick − SET PIECES tag start (s); dotted: tag start (tags last 35 s, kick-offs 5 s)")
    ax.set_title("When is the restart actually taken, relative to the Smart Tagging interval?", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=grid, lw=0.6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
