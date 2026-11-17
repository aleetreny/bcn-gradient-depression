"""S1.5: does the Sportec flag behave like the Metrica flag (BAR-ATH public truth)?

    .venv311/bin/python scripts/s1_5_flag_semantics.py [--out experiments/outputs/011]

1. Is the public window unusual? Sliding 5-minute windows (every 30 s, inside halves) in the 7
   Sportec matches and the 3 Metrica Sample games: possessions, median possession, % D. Percentile of
   the public values (17 possessions, 7.56 s, 19.6 % D) in those distributions.
2. When does the Sportec flag change relative to the events? DFL events are aligned to the tracking
   (time zones, then the event-to-tracking lag estimated from passer-to-ball distance at Play events).
   For each flag change of team: offset of the old team's last Play, and of the new team's on-ball
   event (Play, OtherBallAction, BallClaiming, ShotAtGoal, TacklingGame winner) NEAREST to the change,
   and its first one inside the +/-3 s window. Neither proves which event belongs to the possession.
3. Dead-ball edges vs the ball: D starts vs the ball leaving the pitch, D ends vs the ball starting to
   move (> 3 m/s), in Sportec and in the BAR-ATH public window.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import AWAY, DEAD, FPS, HOME, label_stats, runs  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import DATA_DIR, _file, load_all, parse_match_info  # noqa: E402
from src.eval import errors, metrics  # noqa: E402

WIN = 5 * 60 * FPS
STEP = 30 * FPS
PUBLIC = {"possessions": 17, "poss_median_s": 7.56, "pct_D": 19.59}
ON_BALL = {"Play", "OtherBallAction", "BallClaiming", "ShotAtGoal"}


# ---------------------------------------------------------------- 1. sliding windows
def windows(m):
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(m.half == h)
        for a in range(idx[0], idx[-1] - WIN + 2, STEP):
            sl = slice(a, a + WIN)
            lab, half = m.labels[sl], m.half[sl]
            if np.mean(lab >= 0) < 0.95:
                continue
            st, _ = label_stats(lab, half)
            out.append({"start_from_kickoff_s": (a - idx[0]) / FPS, "half": h, "possessions": st["possessions"],
                        "poss_median_s": st["poss_median_s"], "pct_D": st["pct_D"]})
    return out


def percentile_of(values, x):
    values = np.asarray(values, float)
    return 100 * float(np.mean(values <= x))


# ---------------------------------------------------------------- 2. events vs flag
def parse_time(s):
    return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)


def half_origins(match_id):
    """{half: (first DFL frame N, UTC time of that frame)} from the BALL FrameSets."""
    data = Path(_file("04_03_positions_raw_observed", match_id, DATA_DIR)).read_bytes()
    out = {}
    for h, sec in ((1, b"firstHalf"), (2, b"secondHalf")):
        i = data.index(b'GameSection="' + sec + b'" MatchId="DFL-MAT-' + match_id.encode() + b'" TeamId="BALL"')
        f = re.search(rb'<Frame N="(\d+)" T="([^"]+)"', data[i:i + 400])
        out[h] = (int(f.group(1)), parse_time(f.group(2).decode()))
    return out


def read_events(match_id, m):
    """DataFrame of on-ball events with row (frame index in m), team code and player id."""
    info = parse_match_info(_file("02_01_matchinformation", match_id, DATA_DIR))
    team_code = {info["home_id"]: HOME, info["away_id"]: AWAY}
    origins = half_origins(match_id)
    n1 = int(np.sum(m.half == 1))
    rows = []
    for e in ET.parse(_file("03_02_events_raw", match_id, DATA_DIR)).getroot().iter("Event"):
        if not len(e):
            continue
        t = parse_time(e.get("EventTime"))
        h = 2 if t >= origins[2][1] else 1
        row = (0 if h == 1 else n1) + int(round((t - origins[h][1]).total_seconds() * FPS))
        for node in e.iter():
            kind, team, player = node.tag, None, None
            if kind in ON_BALL:
                team, player = node.get("Team"), node.get("Player")
            elif kind == "TacklingGame":
                team, player = node.get("WinnerTeam"), node.get("Winner")
            if team in team_code and 0 <= row < m.n_frames:
                rows.append({"row": row, "kind": kind, "team": team_code[team], "player": player,
                             "evaluation": node.get("Evaluation")})
                break
    return pd.DataFrame(rows)


def event_lag(m, ev, shifts=range(-50, 51, 2)):
    """Shift s (frames) minimising the median passer-to-ball distance at Play events (ball at row+s)."""
    slot_of = {p: s for s, p in enumerate(m.meta["slot_person"].tolist())}
    plays = ev[(ev["kind"] == "Play") & ev["player"].isin(slot_of)]
    rows = plays["row"].to_numpy()
    slots = plays["player"].map(slot_of).to_numpy()
    prof = []
    for s in shifts:
        r = np.clip(rows + s, 0, m.n_frames - 1)
        d = np.linalg.norm(m.P[r, slots, :2] - m.B[r, :2], axis=1)
        prof.append((s, float(np.nanmedian(d))))
    best = min(prof, key=lambda x: x[1])
    return best[0], prof


def change_vs_events(m, ev, lag):
    gf, gk = errors.team_changes(m.labels, m.half)
    rows = ev["row"].to_numpy() + lag
    out = []
    for f, new in zip(gf, gk):
        old = 1 - new
        w = (rows >= f - 75) & (rows <= f + 75)
        old_play = rows[w & (ev["kind"].to_numpy() == "Play") & (ev["team"].to_numpy() == old) & (rows <= f + 25)]
        new_act = rows[w & (ev["team"].to_numpy() == new)]
        out.append({"old_last_play": int(old_play.max() - f) if len(old_play) else None,
                    "new_nearest_action": int(new_act[np.argmin(np.abs(new_act - f))] - f) if len(new_act) else None,
                    "new_first_action_in_window": int(new_act.min() - f) if len(new_act) else None})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 3. dead-ball edges vs ball
def dead_edges(m, labels, vis, frames_mask):
    """Offsets of D starts vs ball leaving the pitch, and of D ends vs ball starting to move."""
    L, W = m.pitch
    B = m.B[:, :2]
    out_pitch = vis & ((B[:, 0] < 0) | (B[:, 0] > L) | (B[:, 1] < 0) | (B[:, 1] > W))
    speed = np.full(m.n_frames, np.nan)
    ok = vis[2:] & vis[:-2]
    speed[1:-1][ok] = np.linalg.norm(B[2:][ok] - B[:-2][ok], axis=1) * FPS / 2
    key = np.where(frames_mask & (labels >= 0), labels, -1).astype(int) * 10 + m.half
    s, e, v = runs(key)
    lab = v // 10
    starts, ends = [], []
    for i in range(1, len(s)):
        if m.half[s[i]] != m.half[s[i - 1]] or not frames_mask[s[i]]:
            continue
        if lab[i] == DEAD and lab[i - 1] in (HOME, AWAY):
            f = s[i]
            w = np.flatnonzero(out_pitch[f - 75:f + 76])
            starts.append({"frame": int(f), "ball_out_offset": int(w[0] - 75) if len(w) else None,
                           "ball_visible_at_f": bool(vis[f])})
        if lab[i] in (HOME, AWAY) and lab[i - 1] == DEAD:
            f = s[i]
            seg = speed[f - 75:f + 76]
            moving = np.flatnonzero(seg > 3)
            # first sustained movement: > 3 m/s on 3 consecutive frames
            first = next((j for j in moving if j + 2 < len(seg) and np.all(seg[j:j + 3] > 3)), None)
            ends.append({"frame": int(f), "ball_moves_offset": int(first - 75) if first is not None else None,
                         "ball_visible_at_f": bool(vis[f])})
    return pd.DataFrame(starts), pd.DataFrame(ends)


def q(x):
    x = pd.Series(x).dropna().astype(float)
    return {"n": int(len(x)), "p25": float(x.quantile(.25)) if len(x) else None,
            "median": float(x.median()) if len(x) else None, "p75": float(x.quantile(.75)) if len(x) else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/011")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sportec = load_all()

    # 1
    win_sp = pd.DataFrame([dict(w, source="Sportec", match=mid) for mid, m in sportec.items() for w in windows(m)])
    win_ms = pd.DataFrame([dict(w, source="Metrica Sample", match=str(g)) for g in GAMES for w in windows(load_sample(g))])
    wins = pd.concat([win_sp, win_ms])
    pct = {}
    for src, d in wins.groupby("source"):
        ko = d[d["start_from_kickoff_s"] == 0]
        pct[src] = {"windows": len(d),
                    **{"pct_windows_with_{}_<=_public".format(k): percentile_of(d[k], v) for k, v in PUBLIC.items()},
                    "possessions_p10_p50_p90": np.percentile(d["possessions"], [10, 50, 90]).tolist(),
                    "poss_median_s_p10_p50_p90": np.percentile(d["poss_median_s"], [10, 50, 90]).round(2).tolist(),
                    "pct_D_p10_p50_p90": np.percentile(d["pct_D"], [10, 50, 90]).round(1).tolist(),
                    "low_D_windows (D <= 30 %)": int((d["pct_D"] <= 30).sum()),
                    "low_D: pct_windows_possessions_<=_17": percentile_of(d.loc[d["pct_D"] <= 30, "possessions"], 17),
                    "low_D: pct_windows_median_>=_7.56_s": 100 * float(np.mean(d.loc[d["pct_D"] <= 30, "poss_median_s"] >= 7.56)),
                    "corr_pct_D_vs_poss_median": float(d["pct_D"].corr(d["poss_median_s"])),
                    "kickoff_windows_possessions": ko["possessions"].tolist(),
                    "kickoff_windows_pct_D": ko["pct_D"].round(1).tolist()}

    # 2
    lags, changes = {}, []
    for mid, m in sportec.items():
        ev = read_events(mid, m)
        lag, prof = event_lag(m, ev)
        lags[mid] = {"lag_frames": lag, "median_dist_at_lag_m": dict(prof)[lag], "median_dist_at_0_m": dict(prof)[0]}
        c = change_vs_events(m, ev, lag)
        c["match"] = mid
        changes.append(c)
    changes = pd.concat(changes)
    ev_summary = {"lag_by_match": lags,
                  "old_team_last_play_offset": q(changes["old_last_play"]),
                  "new_team_nearest_action_offset": q(changes["new_nearest_action"]),
                  "new_team_first_action_in_window_offset": q(changes["new_first_action_in_window"]),
                  "share_changes_with_new_action_within_0.5s": float(np.mean(changes["new_nearest_action"].abs() <= 12)),
                  "share_changes_with_old_play_within_0.5s_before": float(np.mean((changes["old_last_play"] >= -12) & (changes["old_last_play"] <= 0)))}

    # 3
    sp_starts, sp_ends = [], []
    for mid, m in sportec.items():
        a, b = dead_edges(m, m.labels, m.ball_xy_visible, m.half > 0)
        sp_starts.append(a)
        sp_ends.append(b)
    sp_starts, sp_ends = pd.concat(sp_starts), pd.concat(sp_ends)
    atd = load_atd()
    gt = metrics.public_gt()
    gt = np.where(gt == 3, -1, gt).astype(np.int8)
    pub_starts, pub_ends = dead_edges(atd, gt, atd.ball_xy_visible, gt >= 0)
    dead_summary = {
        "sportec_D_starts": len(sp_starts),
        "sportec_ball_out_minus_D_start_frames": q(sp_starts["ball_out_offset"]),
        "sportec_D_starts_with_ball_out_within_3s_pct": 100 * float(sp_starts["ball_out_offset"].notna().mean()),
        "sportec_ball_moves_minus_D_end_frames": q(sp_ends["ball_moves_offset"]),
        "public_D_starts": pub_starts.to_dict("records"), "public_D_ends": pub_ends.to_dict("records"),
    }

    wins.to_csv(args.out / "sliding_windows.csv", index=False, float_format="%.3f")
    changes.to_csv(args.out / "sportec_changes_vs_events.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps({"windows": pct, "events": ev_summary, "dead": dead_summary},
                                                     indent=2, default=str))
    plot(wins, changes, sp_ends, args.out / "flag_semantics.png")
    print(json.dumps({"windows": pct, "events": ev_summary, "dead": dead_summary}, indent=1, default=str))
    return 0


def plot(wins, changes, sp_ends, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid = "#52514e", "#e4e3df"
    colors = {"Sportec": "#2a78d6", "Metrica Sample": "#1baf7a"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for src, d in wins.groupby("source"):
        axes[0].hist(d["possessions"], bins=np.arange(5, 50, 1.5), histtype="step", lw=2, density=True, color=colors[src], label=src)
        axes[1].hist(d["pct_D"], bins=np.arange(0, 85, 4), histtype="step", lw=2, density=True, color=colors[src], label=src)
    for ax, v, lab in ((axes[0], PUBLIC["possessions"], "public: 17"), (axes[1], PUBLIC["pct_D"], "public: 19.6 %")):
        ax.axvline(v, color="#eb6834", lw=2)
        ax.text(v, ax.get_ylim()[1] * 0.92, " " + lab, color="#eb6834")
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_xlabel("possessions in a 5-minute window")
    axes[1].set_xlabel("% D in a 5-minute window")
    axes[0].set_title("Is 17 possessions in 5 min unusual?", loc="left")
    axes[1].set_title("Is 19.6 % D in 5 min unusual?", loc="left")
    ax = axes[2]
    bins = np.arange(-75, 76, 5)
    ax.hist(changes["new_nearest_action"].dropna(), bins=bins, histtype="step", lw=2, color="#2a78d6", label="new team's nearest on-ball event")
    ax.hist(changes["old_last_play"].dropna(), bins=bins, histtype="step", lw=2, color="#eb6834", label="old team's last pass (Play)")
    ax.axvline(0, color=ink, ls=":")
    ax.set_xlabel("event frame − flag change frame (lag-corrected)")
    ax.set_title("Sportec: when does the flag change?", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=grid, lw=0.6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
