"""The common match format every reader returns (metrica_atd, sportec, metrica_sample, ...).

Row t is one frame at 25 fps. For BAR-ATH it is the video frame / submission row; for sources
without a video it is the halves concatenated in order (no halftime gap), see `half`.

    P[T, S, 3]     float32  per slot: x, y in metres (0..L, 0..W), speed in m/s. NaN if not seen.
    B[T, 3]        float32  ball: x, y in metres, third channel (source-specific, see each reader).
    slot_team[S]   int8     0 = H (home), 1 = A (away). Slots are tracklets, never identities.
    slot_is_gk[S]  bool     goalkeeper according to the source metadata.
    half[T]        int8     1, 2, or 0 outside both halves.
    graded[T]      bool     frame would be scored (inside a half).
    attack_sign[T] int8     +1 if H attacks towards +x on that frame in raw coordinates, else -1.
    labels[T]      int8     0 = H, 1 = A, 2 = D, -1 = unlabelled. None when the source has none.
    meta           dict     source extras (match id, original frame numbers, ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

HOME, AWAY, DEAD, UNLABELLED = 0, 1, 2, -1
LABEL_CHARS = np.array(["H", "A", "D"])
FPS = 25


@dataclass(frozen=True)
class Match:
    P: np.ndarray
    B: np.ndarray
    slot_team: np.ndarray
    slot_is_gk: np.ndarray
    half: np.ndarray
    graded: np.ndarray
    attack_sign: np.ndarray
    pitch: tuple
    home_name: str
    away_name: str
    source: str = ""
    labels: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)
    direction_normalized: bool = False

    @property
    def n_frames(self) -> int:
        return self.P.shape[0]

    @property
    def slot_visible(self) -> np.ndarray:
        return np.isfinite(self.P[..., 0]) & np.isfinite(self.P[..., 1])

    @property
    def ball_xy_visible(self) -> np.ndarray:
        return np.isfinite(self.B[:, 0]) & np.isfinite(self.B[:, 1])

    @property
    def ball_speed_only(self) -> np.ndarray:
        return ~self.ball_xy_visible & np.isfinite(self.B[:, 2])

    def normalized(self) -> "Match":
        """Copy with the pitch rotated 180 degrees where needed so H always attacks +x.

        Both axes flip (a rotation, not a mirror), so left/right stays consistent for the attacker.
        """
        if self.direction_normalized:
            return self
        length, width = self.pitch
        flip = self.attack_sign < 0
        P, B = self.P.copy(), self.B.copy()
        P[flip, :, 0] = length - P[flip, :, 0]
        P[flip, :, 1] = width - P[flip, :, 1]
        B[flip, 0] = length - B[flip, 0]
        B[flip, 1] = width - B[flip, 1]
        return replace(self, P=P, B=B, direction_normalized=True)


def runs(values: np.ndarray):
    """(starts, stops, values) of maximal constant runs; stop is exclusive."""
    values = np.asarray(values)
    if len(values) == 0:
        return np.array([], int), np.array([], int), values[:0]
    edges = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(values)]
    return starts, stops, values[starts]


def possession_spells(labels: np.ndarray, half: np.ndarray):
    """(starts, stops, team) of possession spells: maximal runs of H or of A within one half.

    Same definition as evaluation/possession_metrics.py: D (and unlabelled) frames end a spell,
    and no spell crosses a half boundary.
    """
    key = np.where(np.isin(labels, (HOME, AWAY)), labels, -9).astype(np.int64) * 10 + half
    starts, stops, _ = runs(key)
    team = labels[starts]
    keep = np.isin(team, (HOME, AWAY)) & (half[starts] > 0)
    return starts[keep], stops[keep], team[keep]


def label_stats(labels, half, ball_visible=None):
    """Label statistics over labelled frames (H/A/D), in the units the plan asks for."""
    lab = np.isin(labels, (HOME, AWAY, DEAD)) & (half > 0)
    minutes = lab.sum() / FPS / 60
    s, e, team = possession_spells(np.where(lab, labels, -1), np.where(lab, half, 0))
    dur = (e - s) / FPS
    key = np.where(lab, labels, -1).astype(int) * 10 + half
    ds, de, dv = runs(key)
    dead_runs = (de - ds)[(dv // 10 == DEAD)] / FPS
    # direct H<->A changes: consecutive spells with no D in between
    direct = int(np.sum((s[1:] == e[:-1]) & (team[1:] != team[:-1])))
    out = {
        "frames": int(lab.sum()), "minutes": minutes,
        "pct_D": 100 * np.mean(labels[lab] == DEAD),
        "pct_H_of_alive": 100 * np.mean(labels[lab & (labels != DEAD)] == HOME),
        "possessions": len(dur), "possessions_per_min": len(dur) / minutes,
        "poss_mean_s": float(dur.mean()), "poss_median_s": float(np.median(dur)),
        "pct_poss_lt_2s": 100 * np.mean(dur < 2), "poss_min_s": float(dur.min()),
        "dead_runs": len(dead_runs), "dead_mean_s": float(dead_runs.mean()) if len(dead_runs) else float("nan"),
        "pct_changes_direct": 100 * direct / max(len(dur) - 1, 1),
    }
    if ball_visible is not None:
        out["pct_ball_visible"] = 100 * ball_visible[lab].mean()
    return out, dur
