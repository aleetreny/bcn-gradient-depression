"""Read SkillCorner Open Data (20 A-League 2024/25 matches, broadcast tracking) into the common format.

    from src.data.skillcorner import MATCH_IDS, load
    m = load(1927964)                  # 25 fps, labels from phases of play + per-frame possession
    m = load(1927964, detected_only=False)   # keep SkillCorner's extrapolated positions

Tracking (`*_tracking_extrapolated.jsonl`, 10 fps, metres with the origin at the centre spot) is
shifted to 0..L x 0..W and linearly interpolated to 25 fps inside each period. An interpolated value
needs both neighbouring 10 fps samples. detected_only=True (default) drops players and ball that the
camera did not detect (SkillCorner extrapolates them), so the visible set is a broadcast-like subset;
frames without tracking (replays, cut-aways) are all NaN. Speeds are recomputed from positions (m/s);
B[:, 2] is ball speed. Slots are the players in metadata order shuffled with a fixed seed; goalkeeper
= role acronym GK. attack_sign from `home_team_side` per period.

Labels (SkillCorner's own definitions, NOT Metrica's -- use with care):
- alive = inside a phase of play (`*_phases_of_play.csv`, frame_start..frame_end); D elsewhere in a period;
- team = per-frame `possession.group` carried forward inside each alive stretch (the last team that had
  the ball keeps it while it travels); before the first possessor of a stretch, the phase's
  `team_in_possession_id`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.common import AWAY, DEAD, FPS, HOME, UNLABELLED, Match, runs  # noqa: E402

DATA_DIR = ROOT / "data_external" / "skillcorner_opendata" / "data" / "matches"
CACHE_DIR = ROOT / "local_data" / "cache"
SRC_FPS = 10
CACHE_VERSION = 1
MATCH_IDS = tuple(sorted(int(p.name) for p in DATA_DIR.iterdir() if p.is_dir())) if DATA_DIR.is_dir() else ()


def _upsample(v: np.ndarray, n_out: int) -> np.ndarray:
    """10 fps -> 25 fps along axis 0 (linear; NaN unless both neighbours are known)."""
    pos = np.arange(n_out) * SRC_FPS / FPS
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(v) - 1)
    w = (pos - lo).reshape((-1,) + (1,) * (v.ndim - 1)).astype(np.float32)
    out = (1 - w) * v[lo] + w * v[hi]
    exact = (w == 0).reshape(-1)
    out[exact] = v[lo[exact]]
    return out


def build(match_id: int, data_dir: Path = DATA_DIR, detected_only: bool = True) -> dict:
    from src.data.sportec import _speed
    folder = Path(data_dir) / str(match_id)
    meta = json.loads((folder / "{}_match.json".format(match_id)).read_text())
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    players = sorted(meta["players"], key=lambda p: p["id"])
    order = np.random.default_rng(match_id).permutation(len(players))
    players = [players[i] for i in order]
    slot_of = {p["id"]: s for s, p in enumerate(players)}
    slot_team = np.array([HOME if p["team_id"] == home_id else AWAY for p in players], np.int8)
    slot_is_gk = np.array([p["player_role"]["acronym"] == "GK" for p in players], bool)
    team_of_group = {"home team": HOME, "away team": AWAY}

    recs = [json.loads(line) for line in (folder / "{}_tracking_extrapolated.jsonl".format(match_id)).open()]
    frame = np.array([r["frame"] for r in recs])
    period = np.array([r["period"] or 0 for r in recs], np.int8)
    S = len(players)
    P10 = np.full((len(recs), S, 2), np.nan, np.float32)
    B10 = np.full((len(recs), 2), np.nan, np.float32)
    grp = np.full(len(recs), UNLABELLED, np.int8)
    for i, r in enumerate(recs):
        for pd_ in r["player_data"]:
            s = slot_of.get(pd_["player_id"])
            if s is not None and (pd_["is_detected"] or not detected_only):
                P10[i, s] = (pd_["x"] + length / 2, pd_["y"] + width / 2)
        b = r["ball_data"]
        if b["x"] is not None and (b["is_detected"] or not detected_only):
            B10[i] = (b["x"] + length / 2, b["y"] + width / 2)
        g = r["possession"]["group"]
        if g in team_of_group and r["player_data"]:
            grp[i] = team_of_group[g]

    phases = pd.read_csv(folder / "{}_phases_of_play.csv".format(match_id))
    alive10 = np.zeros(len(recs), bool)
    phase_team = np.full(len(recs), UNLABELLED, np.int8)
    pos_of = {f: i for i, f in enumerate(frame)}
    for ph in phases.itertuples():
        a, b = pos_of.get(ph.frame_start), pos_of.get(ph.frame_end)
        if a is None or b is None:
            continue
        alive10[a:b + 1] = True
        phase_team[a:b + 1] = HOME if ph.team_in_possession_id == home_id else AWAY

    lab10 = np.full(len(recs), UNLABELLED, np.int8)
    for h in (1, 2):
        inside = period == h
        lab10[inside] = DEAD
        starts, stops, val = runs(np.where(inside, alive10, False))
        for a, b, v in zip(starts, stops, val):
            if not v:
                continue
            cur = UNLABELLED
            for i in range(a, b):
                if grp[i] != UNLABELLED:
                    cur = grp[i]
                lab10[i] = cur if cur != UNLABELLED else phase_team[i]

    out = {k: [] for k in ("P", "B", "half", "labels", "frame_id")}
    for h in (1, 2):
        idx = np.flatnonzero(period == h)
        if not len(idx):
            continue
        n_out = int((len(idx) - 1) * FPS / SRC_FPS) + 1
        near = np.minimum(np.rint(np.arange(n_out) * SRC_FPS / FPS).astype(int), len(idx) - 1)
        out["P"].append(_upsample(P10[idx], n_out))
        out["B"].append(_upsample(B10[idx], n_out))
        out["half"].append(np.full(n_out, h, np.int8))
        out["labels"].append(lab10[idx][near])
        out["frame_id"].append(frame[idx][near])
    xy = np.concatenate(out["P"])
    bxy = np.concatenate(out["B"])
    half = np.concatenate(out["half"])
    P = np.concatenate([xy, _speed(xy, half)[..., None]], axis=2).astype(np.float32)
    B = np.concatenate([bxy, _speed(bxy, half)[:, None]], axis=1).astype(np.float32)
    attack_sign = np.ones(len(half), np.int8)
    for h, side in zip((1, 2), meta["home_team_side"]):
        attack_sign[half == h] = 1 if side == "left_to_right" else -1
    return {"P": P, "B": B, "slot_team": slot_team, "slot_is_gk": slot_is_gk, "half": half,
            "attack_sign": attack_sign, "labels": np.concatenate(out["labels"]),
            "frame_id": np.concatenate(out["frame_id"]), "pitch": np.array([length, width]),
            "home_name": str(meta["home_team"]["short_name"]), "away_name": str(meta["away_team"]["short_name"])}


def load(match_id: int, data_dir: Path = DATA_DIR, detected_only: bool = True, use_cache: bool = True) -> Match:
    cache = CACHE_DIR / "skillcorner_{}_{}_v{}.npz".format(match_id, "det" if detected_only else "ext", CACHE_VERSION)
    if use_cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    else:
        d = build(match_id, data_dir, detected_only)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **d)
    return Match(P=d["P"], B=d["B"], slot_team=d["slot_team"], slot_is_gk=d["slot_is_gk"],
                 half=d["half"], graded=d["half"] > 0, attack_sign=d["attack_sign"], pitch=tuple(d["pitch"].tolist()),
                 home_name=str(d["home_name"]), away_name=str(d["away_name"]), source="skillcorner",
                 labels=d["labels"], meta={"match_id": match_id, "frame_id": d["frame_id"], "detected_only": detected_only})
