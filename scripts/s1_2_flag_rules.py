"""S1.2: which simple rule best imitates the possession flag?

    .venv311/bin/python scripts/s1_2_flag_rules.py [--out experiments/outputs/008]

Three rules from src/features/possession_rules.py (<= 3 parameters each), each over a small grid:
nearest with hold, last touch, control zone. Scored on frames where the truth is H or A and the
ball is visible:
  - agreement: share of those frames where the rule's team equals the truth;
  - team-change timing: truth changes of team matched to rule changes of the same destination
    within +/-3 s; median |offset| and median signed offset (frames, > 0 = rule late);
  - extra: rule changes on truth-alive frames that match no truth change.
Primary: BAR-ATH public 5 min (12 team changes; the plan allows calibrating <= 3 parameters here).
Control: the 7 Sportec matches pooled (2,351 team changes), same grid.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import AWAY, HOME  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.eval import errors, metrics, plots  # noqa: E402
from src.features.possession_rules import Kinematics, rule_control, rule_last_touch, rule_nearest  # noqa: E402

MAX_LAG = 75
GRIDS = {
    "nearest+hold": (rule_nearest, {"k": [0, 3, 6, 12, 25]}),
    "last touch": (rule_last_touch, {"r": [1.0, 1.5, 2.0, 3.0], "theta": [20, 35, 50], "delta": [1.5, 3.0, 5.0]}),
    "control zone": (rule_control, {"r": [1.0, 1.5, 2.0, 3.0], "m": [3, 6, 12, 25], "exclusive": [True, False]}),
}
PUBLIC_WINDOW = (11185, 18685)


def rule_changes(team, half):
    prev = np.r_[-1, team[:-1]]
    same_half = np.r_[False, half[1:] == half[:-1]]
    idx = np.flatnonzero((team != prev) & (prev >= 0) & (team >= 0) & same_half)
    return idx, team[idx]


def evaluate(team, gt, half, vis):
    """Per-match counts so that several matches can be pooled."""
    alive = np.isin(gt, (HOME, AWAY)) & (half > 0)
    frames = alive & vis
    gf, gk = errors.team_changes(gt, half)
    rf, rk = rule_changes(team, half)
    keep = alive[rf]
    offs, missed, _ = errors._match_edges(gf, gk, rf[keep], rk[keep], MAX_LAG)
    extra = int(keep.sum()) - len(offs)
    return {"agree_n": int((team[frames] == gt[frames]).sum()), "frames": int(frames.sum()),
            "offsets": offs, "n_gt_changes": len(gf), "missed": missed, "extra": extra}


def pooled(results):
    offs = np.concatenate([r["offsets"] for r in results]) if results else np.array([])
    n_gt = sum(r["n_gt_changes"] for r in results)
    return {"agreement": sum(r["agree_n"] for r in results) / max(sum(r["frames"] for r in results), 1),
            "matched": len(offs), "n_gt_changes": n_gt,
            "median_abs_offset": float(np.median(np.abs(offs))) if len(offs) else np.nan,
            "median_offset": float(np.median(offs)) if len(offs) else np.nan,
            "extra_per_gt_change": sum(r["extra"] for r in results) / max(n_gt, 1)}


def shift_profile(m, shifts=(-20, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12, 20)):
    """Median distance from the ball at t+s to the nearest player at t (graded, ball visible).

    If the ball track lagged the players, the minimum would sit at a different shift than in Sportec.
    A shared minimum rules out a time lag, not a constant spatial bias.
    """
    P, B, T = m.P[..., :2].astype(np.float64), m.B[:, :2].astype(np.float64), m.n_frames
    out = {}
    for s in shifts:
        t = np.arange(max(0, -s), min(T, T - s))
        t = t[m.ball_xy_visible[t + s] & m.graded[t]]
        d = np.linalg.norm(P[t] - B[t + s, None, :], axis=-1)
        d = np.where(np.isnan(d), np.inf, d).min(1)
        out[int(s)] = float(np.median(d[np.isfinite(d)]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/008")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    atd = load_atd()
    gt_pub = metrics.public_gt()
    gt_pub = np.where(gt_pub == 3, -1, gt_pub).astype(np.int8)
    kin_pub = Kinematics.from_match(atd)
    sportec = load_all()
    kin_sp = {mid: Kinematics.from_match(m) for mid, m in sportec.items()}

    rows, best_team = [], {}
    for rule, (fn, grid) in GRIDS.items():
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            team_pub = fn(kin_pub, **params)
            pub = pooled([evaluate(team_pub, gt_pub, atd.half, atd.ball_xy_visible)])
            sp = pooled([evaluate(fn(kin_sp[mid], **params), m.labels, m.half, m.ball_xy_visible)
                         for mid, m in sportec.items()])
            rows.append({"rule": rule, "params": json.dumps(params),
                         **{"pub_" + k: v for k, v in pub.items()}, **{"sp_" + k: v for k, v in sp.items()}})
            key = (rule, json.dumps(params))
            best_team[key] = team_pub
    grid_df = pd.DataFrame(rows)

    def best(df, prefix):
        return df.sort_values([prefix + "agreement", prefix + "median_abs_offset"], ascending=[False, True]).groupby("rule", sort=False).head(1)

    best_pub = best(grid_df, "pub_").set_index("rule")
    best_sp = best(grid_df, "sp_").set_index("rule")
    cols_pub = ["params", "pub_agreement", "pub_matched", "pub_n_gt_changes", "pub_median_abs_offset", "pub_median_offset",
                "pub_extra_per_gt_change", "sp_agreement", "sp_median_abs_offset", "sp_median_offset", "sp_extra_per_gt_change"]
    cols_sp = ["params", "sp_agreement", "sp_matched", "sp_n_gt_changes", "sp_median_abs_offset", "sp_median_offset",
               "sp_extra_per_gt_change", "pub_agreement", "pub_median_abs_offset"]

    grid_df.to_csv(args.out / "grid.csv", index=False, float_format="%.4f")
    best_pub[cols_pub].to_csv(args.out / "best_on_public.csv", float_format="%.4f")
    best_sp[cols_sp].to_csv(args.out / "best_on_sportec.csv", float_format="%.4f")

    a, b = PUBLIC_WINDOW
    lanes = [("truth", gt_pub)] + [("{} {}".format(r, best_pub.loc[r, "params"]),
                                    np.where(gt_pub == 2, 2, best_team[(r, best_pub.loc[r, "params"])]))
                                   for r in best_pub.index]
    plots.timeline(args.out / "timeline_public_rules.png", lanes, a, b,
                   title="Public 5 min: truth vs the best setting of each rule (truth's D copied in, rules say nothing about D)",
                   extra=[("ball visible", atd.ball_xy_visible)])

    # public offsets, one by one, for the best setting of each rule and for unfiltered nearest
    per_change = {}
    for label, team in [("nearest k=0 (no hold)", rule_nearest(kin_pub, 0))] + \
            [("{} {}".format(r, best_pub.loc[r, "params"]), best_team[(r, best_pub.loc[r, "params"])]) for r in best_pub.index]:
        ev = evaluate(team, gt_pub, atd.half, atd.ball_xy_visible)
        per_change[label] = {"offsets": sorted(ev["offsets"].tolist()), "matched": len(ev["offsets"]),
                             "missed": ev["missed"], "extra": ev["extra"],
                             "median": float(np.median(ev["offsets"])) if len(ev["offsets"]) else None}
    # ball vs players time-shift profile: does the ball track lag the players in the ATD?
    profile = {"ATD BAR-ATH": shift_profile(atd), "Sportec J03WMX": shift_profile(sportec["J03WMX"])}
    (args.out / "public_offsets_and_shift_profile.json").write_text(json.dumps(
        {"public_offsets": per_change, "ball_shift_profile": profile}, indent=2))

    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format,
                           "display.max_colwidth", 60):
        print("Best per rule on the PUBLIC window (ranked by agreement, then |offset|)")
        print(best_pub[cols_pub])
        print("\nBest per rule on SPORTEC (7 matches pooled)")
        print(best_sp[cols_sp])
        print("\nTop 8 of the whole grid on Sportec")
        print(grid_df.sort_values("sp_agreement", ascending=False).head(8)[["rule", "params", "sp_agreement", "sp_median_abs_offset",
                                                                           "sp_median_offset", "pub_agreement", "pub_median_abs_offset"]])
    print(json.dumps(per_change, indent=1))
    print("ball shift profile (median m):", json.dumps(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
