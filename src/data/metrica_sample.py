"""Read Metrica Sample Data games 1-3 into the common format, with H/A/D labels derived from events.

    from src.data.metrica_sample import GAMES, load
    m = load(1)                        # tracking via kloppy (cached), labels from the event file
    m.labels                           # 0 = H, 1 = A, 2 = D, -1 outside play
    m.meta["events"]                   # unified event table (pandas)

Same provider as BAR-ATH, but this is a manually curated sample: no tracking ground truth for
possession exists, so the labels are DERIVED and only as good as the rules below.

Tracking (kloppy): games 1-2 are the CSV pair (home/away), game 3 is FIFA/EPTS. Rows are tracking
frames (frame_id - 1), both periods back to back as in the files. Coordinates 0-1 are scaled by
105 x 68 m (the files carry no measured pitch); speed is recomputed from positions (m/s), since
the CSV has none. B[:, 2] is ball speed from positions. Slots are the players in metadata order,
shuffled with a fixed seed; the first listed player of each team is the goalkeeper in games 1-2
(Player11 / Player25) and position_type == Goalkeeper in game 3.

Label derivation (events sorted by start frame; frames are tracking frames):
- In possession: the team of the latest on-ball event -- PASS, CARRY, SHOT, RECOVERY, SET PIECE,
  BALL LOST -- from its start frame. CHALLENGE and CARD do not change the team (duels involve both
  teams and are logged once per side).
- BALL LOST hands the ball to the other team from its end frame, unless a later event says otherwise.
- PASS with subtype GOAL KICK is a restart too (goal kicks are not logged as SET PIECE).
- An on-ball event other than RECOVERY while dead, with no restart logged, counts as an unlogged
  restart (alive again). A RECOVERY while dead (keeper collecting) keeps the ball dead.
- Dead (D) from the end frame of BALL OUT, of SHOT ending OUT or GOAL, and of BALL LOST ending
  OFFSIDE / HAND BALL / END HALF, and from the start frame of FAULT RECEIVED, until the start frame
  of the next SET PIECE (the restart is alive).
- Before the first kick-off and after the last event of each period: unlabelled (-1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.common import AWAY, DEAD, HOME, UNLABELLED, Match  # noqa: E402

DATA_DIR = ROOT / "data_external" / "metrica_sample_data" / "data"
CACHE_DIR = ROOT / "local_data" / "cache"
GAMES = (1, 2, 3)
PITCH = (105.0, 68.0)
CACHE_VERSION = 3

ON_BALL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST"}
DEAD_AFTER_END_SUBTYPES = ("OFFSIDE", "HAND BALL", "END HALF")


def _game_dir(game: int, data_dir: Path) -> Path:
    return Path(data_dir) / "Sample_Game_{}".format(game)


def read_events(game: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Unified table: team ('H'/'A'), type, subtype, period, start_frame, end_frame (tracking frames)."""
    folder = _game_dir(game, data_dir)
    if game in (1, 2):
        raw = pd.read_csv(folder / "Sample_Game_{}_RawEventsData.csv".format(game))
        ev = pd.DataFrame({
            "team": raw["Team"].map({"Home": "H", "Away": "A"}),
            "type": raw["Type"].str.upper(), "subtype": raw["Subtype"].fillna("").str.upper(),
            "period": raw["Period"].astype(int),
            "start_frame": raw["Start Frame"].astype(int), "end_frame": raw["End Frame"].astype(int)})
    else:
        data = json.loads((folder / "Sample_Game_3_events.json").read_text())["data"]

        def subtype(e):
            s = e.get("subtypes")
            if not s:
                return ""
            return "-".join(x["name"] for x in s) if isinstance(s, list) else s["name"]

        ev = pd.DataFrame({
            "team": ["H" if e["team"]["name"] == "Team A" else "A" for e in data],  # Team A = idLocalTeam
            "type": [e["type"]["name"].upper() for e in data],
            "subtype": [subtype(e).upper() for e in data],
            "period": [int(e["period"]) for e in data],
            "start_frame": [int(e["start"]["frame"]) for e in data],
            "end_frame": [int(e["end"]["frame"] or 0) for e in data]})
    ev["end_frame"] = np.where(ev["end_frame"] > 0, ev["end_frame"], ev["start_frame"])
    return ev.sort_values(["start_frame", "end_frame"], kind="stable").reset_index(drop=True)


def _starts_dead(e) -> bool:
    if e.type == "BALL OUT":
        return True
    if e.type == "SHOT" and (e.subtype.endswith("OUT") or "GOAL" in e.subtype.split("-")[-1]):
        return True
    return e.type == "BALL LOST" and any(s in e.subtype for s in DEAD_AFTER_END_SUBTYPES)


def derive_labels(events: pd.DataFrame, n_frames: int, period: np.ndarray):
    """(labels per row, diagnostics). Row = frame - 1. See the module docstring for the rules.

    An on-ball event while the ball is dead, with no SET PIECE (or goal-kick PASS) logged in between,
    is taken as an unlogged restart; diagnostics["unlogged_restarts"] counts them. A RECOVERY while
    dead (typically the keeper collecting a ball that went out) does not restart play.
    """
    points = []  # (frame, priority, kind, team); at equal frames: restart, on-ball, dead
    for e in events.itertuples(index=False):
        team = HOME if e.team == "H" else AWAY
        if e.type == "SET PIECE":
            points.append((e.start_frame, 0, "restart", team))
        elif e.type == "PASS" and "GOAL KICK" in e.subtype:
            points.append((e.start_frame, 0, "restart", team))
        elif e.type in ON_BALL:
            points.append((e.start_frame, 1, "on_ball", team))
        if e.type == "BALL LOST":
            points.append((e.end_frame, 1, "handoff", 1 - team))
        if e.type == "FAULT RECEIVED":
            points.append((e.start_frame, 3, "dead", team))
        elif _starts_dead(e):
            points.append((e.end_frame, 3, "dead", team))
    points.sort(key=lambda p: (p[0], p[1]))
    kind_type = {}
    for e in events.itertuples(index=False):
        if e.type in ON_BALL and e.type != "SET PIECE":
            key = (e.start_frame, HOME if e.team == "H" else AWAY)
            kind_type[key] = e.type if key not in kind_type or e.type != "RECOVERY" else kind_type[key]

    labels = np.full(n_frames, UNLABELLED, np.int8)
    team, dead, cursor, unlogged = None, True, 0, 0
    for frame, _, kind, value in points:
        row = min(max(frame - 1, 0), n_frames)
        if team is not None and row > cursor:
            labels[cursor:row] = DEAD if dead else team
        cursor = max(cursor, row)
        if kind == "restart":
            dead, team = False, value
        elif kind == "on_ball":
            if dead and kind_type[(frame, value)] == "RECOVERY":
                team = value            # keeper collecting a ball that went out: still dead
                continue
            if dead and team is not None:
                unlogged += 1
            dead, team = False, value
        elif kind == "handoff":
            team = value
        else:
            dead = True
    if team is not None:
        labels[cursor:] = DEAD if dead else team   # trimmed to the last event of the period below

    for p in np.unique(period[period > 0]):
        pe = events[events["period"] == p]
        kick = pe[pe["type"] == "SET PIECE"]["start_frame"]
        rows = np.flatnonzero(period == p)
        if kick.empty or rows.size == 0:
            labels[rows] = UNLABELLED
            continue
        first, last = int(kick.min()) - 1, int(pe["end_frame"].max()) - 1
        labels[rows[rows < first]] = UNLABELLED
        labels[rows[rows > last]] = UNLABELLED
    labels[period == 0] = UNLABELLED
    return labels, {"unlogged_restarts": unlogged, "events": len(events)}


def _load_tracking(game: int, data_dir: Path):
    from kloppy import metrica
    folder = _game_dir(game, data_dir)
    if game in (1, 2):
        ds = metrica.load_tracking_csv(
            home_data=str(folder / "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game)),
            away_data=str(folder / "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game)),
            coordinates="metrica")
    else:
        ds = metrica.load_tracking_epts(meta_data=str(folder / "Sample_Game_3_metadata.xml"),
                                        raw_data=str(folder / "Sample_Game_3_tracking.txt"),
                                        coordinates="metrica")
    return ds


def build(game: int, data_dir: Path = DATA_DIR, seed: int = 0) -> dict:
    ds = _load_tracking(game, data_dir)
    df = ds.to_df()
    frame_id = df["frame_id"].to_numpy(np.int64)
    if not np.array_equal(frame_id, np.arange(frame_id[0], frame_id[0] + len(frame_id))):
        raise ValueError("Sample game {}: tracking frames not contiguous".format(game))
    if frame_id[0] != 1:
        raise ValueError("Sample game {}: first frame is {}, expected 1".format(game, frame_id[0]))
    period = df["period_id"].to_numpy(np.int8)

    home, away = ds.metadata.teams
    players = [(HOME, p) for p in home.players] + [(AWAY, p) for p in away.players]
    order = np.random.default_rng([seed, game]).permutation(len(players))
    players = [players[i] for i in order]
    length, width = PITCH
    T = len(df)
    P = np.full((T, len(players), 3), np.nan, np.float32)
    gk = np.zeros(len(players), bool)
    for s, (team, p) in enumerate(players):
        P[:, s, 0] = df["{}_x".format(p.player_id)].to_numpy() * length
        P[:, s, 1] = df["{}_y".format(p.player_id)].to_numpy() * width
        position = str(getattr(p, "starting_position", "") or "")
        first_of_team = p is (home if team == HOME else away).players[0]
        gk[s] = "Goalkeeper" in position or (game in (1, 2) and first_of_team)
    B = np.full((T, 3), np.nan, np.float32)
    B[:, 0] = df["ball_x"].to_numpy() * length
    B[:, 1] = df["ball_y"].to_numpy() * width

    from src.data.sportec import _speed
    P[..., 2] = _speed(P[..., :2], period)
    B[:, 2] = _speed(B[:, :2], period)

    events = read_events(game, data_dir)
    labels, diag = derive_labels(events, T, period)
    slot_team = np.array([t for t, _ in players], np.int8)
    attack_sign = np.ones(T, np.int8)
    for h in (1, 2):
        m = period == h
        hx = np.nanmean(P[m][:, slot_team == HOME, 0])
        ax = np.nanmean(P[m][:, slot_team == AWAY, 0])
        attack_sign[m] = 1 if hx < ax else -1
    return {"P": P, "B": B, "slot_team": slot_team, "slot_is_gk": gk, "half": period,
            "attack_sign": attack_sign, "labels": labels, "frame_id": frame_id,
            "home_name": str(home.name), "away_name": str(away.name),
            "unlogged_restarts": np.int64(diag["unlogged_restarts"])}


def load(game: int, data_dir: Path = DATA_DIR, use_cache: bool = True) -> Match:
    cache = CACHE_DIR / "metrica_sample_{}_v{}.npz".format(game, CACHE_VERSION)
    if use_cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    else:
        d = build(game, data_dir)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **d)
    return Match(P=d["P"], B=d["B"], slot_team=d["slot_team"], slot_is_gk=d["slot_is_gk"],
                 half=d["half"], graded=d["half"] > 0, attack_sign=d["attack_sign"], pitch=PITCH,
                 home_name=str(d["home_name"]), away_name=str(d["away_name"]), source="metrica_sample",
                 labels=d["labels"], meta={"game": game, "frame_id": d["frame_id"],
                                           "events": read_events(game, data_dir),
                                           "unlogged_restarts": int(d["unlogged_restarts"])})
