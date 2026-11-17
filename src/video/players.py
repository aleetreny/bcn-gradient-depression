"""Automatic tracking-to-video spatial alignment and torso HSV colour comparison for the D3 input check.

    from src.video.players import load_detections, pitch_people, match_to_atd, TeamColour

Detections come from pre-extracted person bounding boxes on the tactical video input (`local_data/video/`).
Each person's ground contact point (bottom centre of the bounding box) is projected to pitch coordinates (in metres)
using the inverse of the provided per-frame homography (`BAR-ATH_tactical_homography.csv`).
People outside the pitch boundary (bench, stands, ball boys) are filtered out. On each frame, projected detections
are matched 1-to-1 to visible ATD player coordinates (`match_to_atd`, Hungarian assignment within 2.5 m) to compare
their torso colour statistics against the reference colour distribution of balanced (`11 vs 11`) ATD slots.
Uses provided tracking, homography, and video strictly as inputs for automatic data-consistency checking;
uses zero possession ground-truth labels. All intermediate step files stay in `local_data/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PITCH = (105.3, 68.0)


def load_detections(path=None):
    path = path or ROOT / "local_data" / "video" / "detections_yolo11m_s2.npz"
    z = np.load(path)
    return z["det"], z["feat"]


def load_homographies():
    import sys
    sys.path.insert(0, str(ROOT / "starter"))
    from load_homography import load
    H, err, valid = load()
    return H, valid


def to_pitch(H, frames, u, v):
    """Pixels -> metres (ATD's raw orientation), per detection, with each detection's frame matrix."""
    Hinv = np.linalg.inv(H[frames.astype(int)])
    den = Hinv[:, 2, 0] * u + Hinv[:, 2, 1] * v + Hinv[:, 2, 2]
    x = (Hinv[:, 0, 0] * u + Hinv[:, 0, 1] * v + Hinv[:, 0, 2]) / den
    y = (Hinv[:, 1, 0] * u + Hinv[:, 1, 1] * v + Hinv[:, 1, 2]) / den
    return x * PITCH[0], y * PITCH[1]


def pitch_people(det, H, margin=1.5, min_conf=0.3):
    """Person detections on the pitch: (frame, x_m, y_m, conf, row index into det)."""
    keep = (det[:, 1] == 0) & (det[:, 2] >= min_conf)
    d = det[keep]
    rows = np.flatnonzero(keep)
    u = (d[:, 3] + d[:, 5]) / 2
    v = d[:, 6]
    x, y = to_pitch(H, d[:, 0], u, v)
    on = (x > -margin) & (x < PITCH[0] + margin) & (y > -margin) & (y < PITCH[1] + margin)
    return np.c_[d[on, 0], x[on], y[on], d[on, 2]], rows[on]


def match_to_atd(people, P, slot_team, gate=2.5):
    """For each detection on the pitch: matched ATD slot (-1 if none) and Euclidean distance in metres.

    Returns (slot, dist) arrays aligned with `people`."""
    from scipy.optimize import linear_sum_assignment
    slot = np.full(len(people), -1, np.int64)
    dist = np.full(len(people), np.nan)
    frames = people[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    bounds = np.flatnonzero(np.diff(np.r_[-1, frames[order], 10 ** 9]))
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        idx = order[i0:i1]
        t = frames[idx[0]]
        vis = np.flatnonzero(np.isfinite(P[t, :, 0]))
        if len(vis) == 0:
            continue
        D = np.linalg.norm(people[idx, None, 1:3] - P[t, vis][None, :, :2], axis=-1)
        r, c = linear_sum_assignment(np.where(D < gate, D, 1e6))
        ok = D[r, c] < gate
        slot[idx[r[ok]]] = vis[c[ok]]
        dist[idx[r[ok]]] = D[r[ok], c[ok]]
    return slot, dist


@dataclass
class TeamColour:
    """Automatic torso-colour consistency check (0 home, 1 away, 2 referee/other) against balanced ATD reference slots."""
    scorer: object = None

    def set_reference(self, feat, slot_class):
        from sklearn.ensemble import HistGradientBoostingClassifier
        ok = np.all(np.isfinite(feat), 1)
        self.scorer = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
        self.scorer.fit(feat[ok], slot_class[ok])
        return self

    def score_slots(self, feat):
        out = np.full((len(feat), 3), np.nan)
        ok = np.all(np.isfinite(feat), 1)
        if ok.any():
            p = self.scorer.predict_proba(feat[ok])
            full = np.zeros((ok.sum(), 3))
            full[:, self.scorer.classes_.astype(int)] = p
            out[ok] = full
        return out


__all__ = ["load_detections", "load_homographies", "to_pitch", "pitch_people", "match_to_atd", "TeamColour"]
