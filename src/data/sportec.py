"""Read the Sportec/DFL open tracking data (Bassek et al. 2025, CC-BY 4.0) into the common format.

    from src.data.sportec import MATCH_IDS, load
    m = load("J03WMX")                 # cached as .npz in local_data/cache after the first parse
    m.labels                           # 0 = H, 1 = A, 2 = D per frame (BallStatus / BallPossession)

Source files (data_external/sportec_dfl/, see scripts/download_public_datasets.sh):
    DFL_02_01_matchinformation_*_DFL-MAT-<id>.xml   teams (Role="home"/"guest"), goalkeepers (TW)
    DFL_04_03_positions_raw_observed_*_<id>.xml     one FrameSet per (half, person), 25 fps

Mapping to the common Match (src/data/common.py):
- Rows are the first half followed by the second, with no gap: half[t] says which.
  meta["frame_n"] keeps DFL's own frame number N; meta["slot_person"] the DFL PersonId of each slot
  (only for cross-checks: models must never see it).
- One slot per person who played (starters and substitutes), NaN while off the pitch. Slot order is
  shuffled with a fixed per-match seed, so the slot index carries no meaning (as in the ATD).
  Referees are dropped.
- x, y: DFL is centred on the pitch; shifted to 0..L, 0..W metres.
- Speed: recomputed from positions (centred differences, m/s), because that is what the Metrica ATD
  channel turned out to be (S0.2). DFL's own S/D attributes are kept only as a check in meta.
- B[:, 2] is ball speed from positions (m/s); the ball height Z is in meta["ball_z"].
- labels: BallStatus 0 -> D; otherwise BallPossession 1 -> H, 2 -> A; anything else -> -1.
- attack_sign: per half, H attacks +x if the home team's mean x is below the away team's.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.common import AWAY, DEAD, FPS, HOME, UNLABELLED, Match  # noqa: E402

DATA_DIR = ROOT / "data_external" / "sportec_dfl"
CACHE_DIR = ROOT / "local_data" / "cache"
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
HALVES = {"firstHalf": 1, "secondHalf": 2}
CACHE_VERSION = 2

FRAMESET = re.compile(rb'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]+)" PersonId="([^"]+)">')
NUM = rb'"(-?[\d.]+)"'
PLAYER_FRAME = re.compile(rb'<Frame N="(\d+)" T="[^"]*" X=' + NUM + rb' Y=' + NUM + rb' D=' + NUM + rb' S=' + NUM)
BALL_FRAME = re.compile(rb'<Frame N="(\d+)" T="[^"]*" X=' + NUM + rb' Y=' + NUM + rb' Z=' + NUM
                        + rb' D=' + NUM + rb' S=' + NUM + rb' A="[^"]*" M="[^"]*"'
                        + rb' BallPossession="(\d)" BallStatus="(\d)"')


def _file(kind: str, match_id: str, data_dir: Path) -> Path:
    found = sorted(Path(data_dir).glob("DFL_{}_*DFL-MAT-{}.xml".format(kind, match_id)))
    if len(found) != 1:
        raise FileNotFoundError("{} file for {} in {}: found {}".format(kind, match_id, data_dir, found))
    return found[0]


def parse_match_info(path) -> dict:
    root = ET.parse(path).getroot()
    teams = {t.get("Role"): t for t in root.iter("Team")}
    gk = {p.get("PersonId") for p in root.iter("Player") if p.get("PlayingPosition") == "TW"}
    return {"home_id": teams["home"].get("TeamId"), "away_id": teams["guest"].get("TeamId"),
            "home_name": teams["home"].get("TeamName"), "away_name": teams["guest"].get("TeamName"),
            "goalkeepers": gk}


def _to_array(rows, n_cols):
    return np.array(rows, dtype="S16").reshape(-1, n_cols)


def parse_positions(path):
    """{(half, team, person): array} with columns N, X, Y, [Z], D, S, [poss, status] as float64."""
    data = Path(path).read_bytes()
    pitch = re.search(rb'<PitchSize X="([\d.]+)" Y="([\d.]+)"', data)
    headers = list(FRAMESET.finditer(data))
    out = {}
    for i, h in enumerate(headers):
        chunk = data[h.end():headers[i + 1].start() if i + 1 < len(headers) else len(data)]
        half, team, person = HALVES[h.group(1).decode()], h.group(2).decode(), h.group(3).decode()
        pattern, n_cols = (BALL_FRAME, 8) if team == "BALL" else (PLAYER_FRAME, 5)
        rows = pattern.findall(chunk)
        if len(rows) != chunk.count(b"<Frame "):
            raise ValueError("{}: {} frames unparsed in {}".format(path, chunk.count(b"<Frame ") - len(rows), h.group(0)))
        out[(half, team, person)] = _to_array(rows, n_cols).astype(np.float64)
    return out, (float(pitch.group(1)), float(pitch.group(2)))


def _speed(xy: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Centred-difference speed in m/s along axis 0, never across a half boundary."""
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx) > 2:
            seg = xy[idx]
            v[idx] = np.linalg.norm(np.gradient(seg, axis=0), axis=-1) * FPS
    return v


def _ratio(raw, fd, scale=1.0):
    ok = np.isfinite(raw) & np.isfinite(fd) & (fd > 0.5)
    return float(np.median(raw[ok] * scale / fd[ok])) if ok.any() else float("nan")


def build(match_id: str, data_dir: Path = DATA_DIR, seed: int = 0) -> dict:
    info = parse_match_info(_file("02_01_matchinformation", match_id, data_dir))
    sets, pitch = parse_positions(_file("04_03_positions_raw_observed", match_id, data_dir))
    length, width = pitch

    ball = {h: sets[(h, "BALL", p)] for (h, t, p) in sets if t == "BALL"}
    first_n = {h: int(ball[h][0, 0]) for h in ball}
    n_rows = {h: len(ball[h]) for h in ball}
    for h in ball:
        if not np.array_equal(ball[h][:, 0], np.arange(first_n[h], first_n[h] + n_rows[h])):
            raise ValueError("{} half {}: ball frame numbers not contiguous".format(match_id, h))
    offset = {1: 0, 2: n_rows[1]}
    T = n_rows[1] + n_rows[2]
    half = np.r_[np.ones(n_rows[1], np.int8), np.full(n_rows[2], 2, np.int8)]

    people = sorted({(t, p) for (h, t, p) in sets if t in (info["home_id"], info["away_id"])})
    order = np.random.default_rng([seed, int.from_bytes(match_id.encode(), "little") % 2**32]).permutation(len(people))
    people = [people[i] for i in order]
    P = np.full((T, len(people), 3), np.nan, np.float32)
    raw_s = np.full((T, len(people)), np.nan, np.float32)
    raw_d = np.full((T, len(people)), np.nan, np.float32)
    for s, (team, person) in enumerate(people):
        for h in (1, 2):
            arr = sets.get((h, team, person))
            if arr is None:
                continue
            rows = offset[h] + arr[:, 0].astype(np.int64) - first_n[h]
            if rows.min() < 0 or rows.max() >= offset[h] + n_rows[h]:
                raise ValueError("{}: {} frames outside the ball's range".format(match_id, person))
            P[rows, s, 0] = arr[:, 1] + length / 2
            P[rows, s, 1] = arr[:, 2] + width / 2
            raw_d[rows, s], raw_s[rows, s] = arr[:, 3], arr[:, 4]
    P[..., 2] = _speed(P[..., :2], half)

    B = np.full((T, 3), np.nan, np.float32)
    ball_z = np.full(T, np.nan, np.float32)
    poss = np.zeros(T, np.int8)
    status = np.zeros(T, np.int8)
    ball_raw_s = np.full(T, np.nan, np.float32)
    for h in (1, 2):
        sl = slice(offset[h], offset[h] + n_rows[h])
        B[sl, 0] = ball[h][:, 1] + length / 2
        B[sl, 1] = ball[h][:, 2] + width / 2
        ball_z[sl] = ball[h][:, 3]
        ball_raw_s[sl] = ball[h][:, 5]
        poss[sl], status[sl] = ball[h][:, 6], ball[h][:, 7]
    B[:, 2] = _speed(B[:, :2], half)

    labels = np.full(T, UNLABELLED, np.int8)
    labels[(status == 1) & (poss == 1)] = HOME
    labels[(status == 1) & (poss == 2)] = AWAY
    labels[status == 0] = DEAD

    slot_team = np.array([HOME if t == info["home_id"] else AWAY for t, _ in people], np.int8)
    attack_sign = np.ones(T, np.int8)
    for h in (1, 2):
        m = half == h
        home_x = np.nanmean(P[m][:, slot_team == HOME, 0])
        away_x = np.nanmean(P[m][:, slot_team == AWAY, 0])
        attack_sign[m] = 1 if home_x < away_x else -1

    return {
        "P": P, "B": B, "slot_team": slot_team,
        "slot_is_gk": np.array([p in info["goalkeepers"] for _, p in people]),
        "slot_person": np.array([p for _, p in people]),
        "half": half, "attack_sign": attack_sign, "labels": labels, "ball_z": ball_z,
        "frame_n": np.r_[ball[1][:, 0], ball[2][:, 0]].astype(np.int64),
        "pitch": np.array(pitch), "home_name": info["home_name"], "away_name": info["away_name"],
        "raw_speed_check": np.array([_ratio(raw_s, P[..., 2]), _ratio(raw_d, P[..., 2], 1 / 3.6),
                                     _ratio(ball_raw_s, B[:, 2]), _ratio(ball_raw_s, B[:, 2], 1 / 3.6)]),
    }


def load(match_id: str, data_dir: Path = DATA_DIR, use_cache: bool = True) -> Match:
    cache = CACHE_DIR / "sportec_{}_v{}.npz".format(match_id, CACHE_VERSION)
    if use_cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    else:
        d = build(match_id, data_dir)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **d)
    return Match(P=d["P"], B=d["B"], slot_team=d["slot_team"], slot_is_gk=d["slot_is_gk"],
                 half=d["half"], graded=d["half"] > 0, attack_sign=d["attack_sign"],
                 pitch=tuple(float(v) for v in d["pitch"]), home_name=str(d["home_name"]),
                 away_name=str(d["away_name"]), source="sportec", labels=d["labels"],
                 meta={"match_id": match_id, "frame_n": d["frame_n"], "ball_z": d["ball_z"],
                       "slot_person": d["slot_person"],
                       "raw_speed_check": dict(zip(("player_S_over_fd", "player_D_kph_over_fd",
                                                    "ball_S_over_fd", "ball_S_kph_over_fd"),
                                                   d["raw_speed_check"].tolist()))})


def load_all(data_dir: Path = DATA_DIR, n_jobs: int = 7) -> dict:
    """All seven matches, parsing uncached ones in parallel."""
    from joblib import Parallel, delayed
    todo = [m for m in MATCH_IDS if not (CACHE_DIR / "sportec_{}_v{}.npz".format(m, CACHE_VERSION)).is_file()]
    if todo:
        Parallel(n_jobs=min(n_jobs, len(todo)))(delayed(load)(m, data_dir) for m in todo)
    return {m: load(m, data_dir) for m in MATCH_IDS}
