"""Read Metrica's FIFA/EPTS ATD (BAR-ATH) into the common array format used by every source.

    from src.data.metrica_atd import load
    m = load()                       # parses data/tactical/*_atd.{xml,txt}, cached after the first call
    m.P.shape, m.B.shape             # (199319, 46, 3), (199319, 3)
    n = m.normalized()               # same match, H always attacking +x

Returns the common `Match` (src/data/common.py). Row t is video frame t, i.e. submission row t
(ATD frame_id = t + 1). slot_is_gk comes from position_type == Goalkeeper; labels is None.

Units: raw x, y are normalised 0-1 and scaled by games.py's pitch (105.3 x 68.0 m, not the
metadata's 105 x 68). The metadata calls the third channel `speed_kph`, but for players it is
already m/s: it matches the frame-to-frame displacement with median ratio 0.99 (S0.2, experiments
log 002). It is therefore NOT divided by 3.6. For the ball the third channel is not a speed at all
(no correlation with ball displacement; smooth; present in 51 long runs), so it is kept raw and
unscaled until S1.3 explains it. `ball_speed_only` keeps its plan name but means "third channel
without x, y".
"""
from __future__ import annotations

import io
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from src.data.common import AWAY, HOME, Match  # noqa: E402,F401

CACHE_DIR = ROOT / "local_data" / "cache"


def parse_meta(meta_path) -> dict:
    """Team per slot, goalkeeper flag, team names and first-half attack direction from the XML."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")  # FIFA/EPTS: first team is the home side (Barcelona, per DATA.md)

    def provider_params(node):
        return {p.findtext("Name"): p.findtext("Value") for p in node.iter("ProviderParameter")}

    directions = [provider_params(team).get("attack_direction_first_half") for team in teams]
    players = sorted(root.iter("Player"), key=lambda p: int(p.get("id")))
    if [int(p.get("id")) for p in players] != list(range(1, len(players) + 1)):
        raise ValueError("{}: player ids are not 1..N".format(meta_path))
    return {
        "home_name": teams[0].findtext("Name"),
        "away_name": teams[1].findtext("Name"),
        "home_attack_first_half": directions[0],
        "away_attack_first_half": directions[1],
        "slot_team": np.array([HOME if p.get("teamId") == home_id else AWAY for p in players], np.int8),
        "slot_is_gk": np.array([provider_params(p).get("position_type") == "Goalkeeper" for p in players]),
    }


def parse_raw(raw_path, n_slots: int):
    """(frame_ids[int64], raw[float32, T, n_slots + 1, 3]) straight from the text file.

    Each line is `frame:x,y,s;x,y,s;...:x,y,s`. Swapping both separators for commas turns the file
    into a fixed-width CSV that pandas' C reader parses in seconds.
    """
    data = Path(raw_path).read_bytes().replace(b":", b",").replace(b";", b",")
    n_cols = 1 + 3 * (n_slots + 1)
    dtypes = {0: np.int64, **{c: np.float32 for c in range(1, n_cols)}}
    df = pd.read_csv(io.BytesIO(data), header=None, names=range(n_cols), dtype=dtypes, engine="c")
    if df.shape[1] != n_cols:
        raise ValueError("{}: unexpected layout {}".format(raw_path, df.shape))
    values = df.iloc[:, 1:].to_numpy(np.float32).reshape(len(df), n_slots + 1, 3)
    return df[0].to_numpy(np.int64), values


def _frame_structure(game: str, n_frames: int, home_attack_first_half: str):
    """half[T], graded[T] and attack_sign[T] in video frames.

    Outside the halves, frames take the direction of the nearest half (pre-match -> 1st,
    halftime and after -> 2nd); they are ungraded, so this only keeps features continuous.
    """
    half = np.zeros(n_frames, np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h
    (_, _, end_first), (_, start_second, _) = half_video_blocks(game)
    sign_first = {"left_to_right": 1, "right_to_left": -1}[home_attack_first_half]
    attack_sign = np.full(n_frames, -sign_first, np.int8)
    attack_sign[:end_first + 1] = sign_first
    halftime_mid = (end_first + start_second) // 2
    attack_sign[end_first + 1:halftime_mid] = sign_first
    return half, half > 0, attack_sign


def _cache_path(raw_path) -> Path:
    stat = os.stat(raw_path)
    return CACHE_DIR / "{}_{}_{}.npz".format(Path(raw_path).stem, stat.st_size, int(stat.st_mtime))


def load(game: str = GAME, data_dir=None, use_cache: bool = True) -> Match:
    """The tactical ATD of `game` in the common format. Cached as .npz under local_data/cache/."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    meta = parse_meta(meta_path)
    n_slots = len(meta["slot_team"])

    cache = _cache_path(raw_path)
    if use_cache and cache.is_file():
        with np.load(cache) as z:
            frame_ids, raw = z["frame_ids"], z["raw"]
    else:
        frame_ids, raw = parse_raw(raw_path, n_slots)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache, frame_ids=frame_ids, raw=raw)

    n_frames = GAMES[game]["n_video_frames"]
    if not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("ATD frame_ids are not 1..{} (video_frame = frame_id - 1)".format(n_frames))

    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width, 1.0], np.float32)
    raw = raw * scale
    half, graded, attack_sign = _frame_structure(game, n_frames, meta["home_attack_first_half"])
    return Match(P=raw[:, :n_slots], B=raw[:, n_slots], slot_team=meta["slot_team"],
                 slot_is_gk=meta["slot_is_gk"], half=half, graded=graded, attack_sign=attack_sign,
                 pitch=(length, width), home_name=meta["home_name"], away_name=meta["away_name"],
                 source="metrica_atd", meta={"game": game})
