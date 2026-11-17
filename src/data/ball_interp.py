"""The ATD ball is drawn piecewise-linearly (log 042): short bursts of real detections joined by straight,
constant-velocity stretches. This module measures that pattern on the BAR-ATH tactical ATD (no labels)
and reproduces it on a clean ball (used by src/data/degrade.py, component "interp").

    from src.data.ball_interp import interp_params, linear_mask, interpolate
    p = interp_params()                 # cached in local_data/ball_interp_params.json
    lin = linear_mask(B, half)          # frames where the ball moves on a straight constant-velocity line

A frame is "linear" when the ball's second difference is below LIN_TOL metres. On the ATD that is
74 % of the visible in-play frames; on Sportec 1-2 % (measurement noise is larger than LIN_TOL).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.common import runs

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "local_data" / "ball_interp_params.json"
LIN_TOL = 0.02
EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
VERSION = 1


def linear_mask(B: np.ndarray, half: np.ndarray, tol: float = LIN_TOL) -> np.ndarray:
    """True where the ball is visible on three consecutive frames of one half with a second difference < tol."""
    xy = B[:, :2]
    vis = np.isfinite(xy).all(1)
    out = np.zeros(len(xy), bool)
    ok = vis[2:] & vis[1:-1] & vis[:-2] & (half[2:] == half[:-2])
    acc = np.full(len(xy) - 2, np.inf)
    acc[ok] = np.linalg.norm(xy[2:][ok] - 2 * xy[1:-1][ok] + xy[:-2][ok], axis=1)
    out[1:-1] = acc < tol
    return out


def _run_lengths(B, half, frames):
    """Lengths of linear and of curved runs inside visible ball stretches restricted to `frames`."""
    vis = np.isfinite(B[:, :2]).all(1) & frames
    lin = linear_mask(B, half)
    key = np.where(vis, lin.astype(int) + 1, 0) * 10 + half      # 0 = hidden; 1 curved; 2 linear
    s, e, v = runs(key)
    lengths = e - s
    return lengths[v // 10 == 2], lengths[v // 10 == 1]


def measure(m) -> dict:
    """Histogram of linear / curved run lengths on the graded frames of a Match (no labels read)."""
    lin_len, cur_len = _run_lengths(m.B, m.half, m.graded)
    vis = np.isfinite(m.B[:, :2]).all(1) & m.graded
    return {"version": VERSION, "edges": EDGES, "tol": LIN_TOL,
            "linear_counts": np.histogram(lin_len, EDGES)[0].tolist(),
            "curved_counts": np.histogram(cur_len, EDGES)[0].tolist(),
            "linear_share_visible": float(linear_mask(m.B, m.half)[vis].mean())}


def interp_params(rebuild: bool = False) -> dict:
    if rebuild or not PARAMS.is_file():
        from src.data.metrica_atd import load
        PARAMS.parent.mkdir(parents=True, exist_ok=True)
        PARAMS.write_text(json.dumps(measure(load()), indent=2))
    return json.loads(PARAMS.read_text())


def interpolate(B: np.ndarray, half: np.ndarray, rng, params: dict, dev_max: float | None = None) -> np.ndarray:
    """Within each visible ball stretch, alternate curved runs (kept) and linear runs (replaced by the
    straight line between the frames just before and just after), with run lengths drawn from the
    measured ATD histograms (log-uniform inside a bin). Hidden frames stay hidden.

    dev_max (m): a linear run is shortened (halved until it fits) while the straight line strays more
    than dev_max from the real ball -- the ATD joins detections that agree with a straight path, so its
    lines stay close to the player dribbling (log 042). None = no limit."""
    from src.data.degrade import _hist_sampler
    out = B.copy()
    draw_lin = _hist_sampler(rng, params["edges"], params["linear_counts"])
    draw_cur = _hist_sampler(rng, params["edges"], params["curved_counts"])
    vis = np.isfinite(B[:, :2]).all(1)
    s, e, v = runs(np.where(vis, half, 0))
    for a, b, h in zip(s, e, v):
        if h == 0 or b - a < 3:
            continue
        t = a + 1 + int(rng.integers(0, max(1, int(draw_cur()))))   # random phase; the run's first frame stays real
        while t < b - 1:
            L = max(1, int(round(draw_lin())))
            stop = min(t + L, b - 1)                            # last frame of the run stays a real point
            while dev_max is not None and stop > t:
                w = (np.arange(t, stop) - (t - 1)) / (stop - (t - 1))
                line = B[t - 1, :2] + w[:, None] * (B[stop, :2] - B[t - 1, :2])
                if np.max(np.linalg.norm(line - B[t:stop, :2], axis=1)) <= dev_max:
                    break
                stop = t + (stop - t) // 2
            if stop > t:
                w = (np.arange(t, stop) - (t - 1)) / (stop - (t - 1))
                p0, p1 = B[t - 1, :2], B[stop, :2]
                out[t:stop, :2] = p0 + w[:, None] * (p1 - p0)
            t = stop + max(1, int(round(draw_cur())))
    return out
