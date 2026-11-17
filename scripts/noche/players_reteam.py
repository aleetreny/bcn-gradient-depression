"""Automatic tracking slot-count input consistency check (D3): identify and repair ATD player slots
assigned to the wrong team during `12 vs 10` tracking count anomalies.

    .venv311/bin/python scripts/noche/players_reteam.py [--all]

In automatic optical tracking (ATD), close player crossings occasionally swap a tracklet across teams,
producing impossible `12 vs 10` team counts instead of `11 vs 11`. Per ATD slot, homography-matched video
detections (src/video/players.py) provide an automatic sequence of torso-colour consistency votes compared
against the opposite half's balanced (`11 vs 11`) ATD slots, keeping only high-confidence votes (p > 0.9).
src/video/reteam.py identifies contiguous stretches whose shirt-colour majority contradicts the slot's ATD team
(excluding goalkeepers) and, by default (`--count`), keeps only stretches corroborated by the ATD team counts
themselves (where moving the slot restores `12 vs 10` to `11 vs 11`). Writes intermediate step files to
`local_data/video/players_reteam[_pairs|_all].npz`. Uses provided inputs only; zero ground-truth labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.video.players import TeamColour, load_detections, load_homographies, match_to_atd, pitch_people  # noqa: E402
from src.video.reteam import corroborated, paired, reteam, team_counts, wrong_team_segments  # noqa: E402

OUT = ROOT / "local_data" / "video"
SPLIT = 100000          # between the halves


def colour_votes(m, min_conf=0.4):
    """Per sure matched detection: frame, ATD slot, colour class (0 H, 1 A)."""
    det, feat = load_detections()
    H, _ = load_homographies()
    P, team = m.P, np.asarray(m.slot_team)
    pp, rows = pitch_people(det, H, min_conf=min_conf)
    f = feat[rows]
    slot, _ = match_to_atd(pp, P, team)
    fr = pp[:, 0].astype(int)
    cnt = team_counts(P, team)
    y = np.full(len(pp), -1)
    y[slot >= 0] = team[slot[slot >= 0]]
    full = (cnt[fr, 0] >= 11) & (cnt[fr, 1] >= 11)
    dmin = np.array([np.nanmin(np.linalg.norm(P[t, :, :2] - pp[i, 1:3], axis=1)) if np.isfinite(P[t, :, 0]).any()
                     else np.inf for i, t in enumerate(fr)])
    y[(slot < 0) & full & (dmin > 4)] = 2
    balanced = (cnt[fr, 0] <= 11) & (cnt[fr, 1] <= 11)     # reference frames where ATD counts are balanced (<=11 per team)
    pr = np.full((len(pp), 3), np.nan)
    for part in (fr < SPLIT, fr >= SPLIT):
        ref = ~part & (y >= 0) & (balanced | (y == 2))
        pr[part] = TeamColour().set_reference(f[ref], y[ref]).score_slots(f[part])
    pred = np.argmax(np.nan_to_num(pr, nan=-1), 1)
    conf = np.nanmax(np.nan_to_num(pr, nan=0), 1)
    keep = (slot >= 0) & (conf > 0.9) & (pred < 2)
    return fr[keep], slot[keep], pred[keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true", help="corroborate colour stretches with 12v10 ATD count check (default)")
    ap.add_argument("--all", action="store_true", help="move every colour stretch, without the count check")
    ap.add_argument("--pairs", action="store_true", help="also move the stretches with an other-team partner")
    ap.add_argument("--win", type=int, default=15)
    ap.add_argument("--min_votes", type=int, default=8)
    ap.add_argument("--gap", type=int, default=125,
                    help="frames between stretches that share a new slot (src/video/reteam.py); 0 = one slot each")
    a = ap.parse_args()
    m = load_atd()
    P, team = m.P, np.asarray(m.slot_team)
    fr, slot, vote = colour_votes(m)
    segs_all = wrong_team_segments(fr, slot, vote, P, team, a.win, a.min_votes, skip=np.asarray(m.slot_is_gk))
    segs = segs_all if a.all else corroborated(segs_all, P, team)
    if a.pairs and not a.all:
        segs = sorted(set(segs) | set(paired(segs_all, P, team)))
    r = reteam(m, segs, gap=a.gap or None)
    g = np.asarray(m.half) > 0
    c0, c1 = team_counts(P, team), team_counts(r.P, r.slot_team)
    B = m.B[:, :2]
    d = np.linalg.norm(P[:, :, :2] - B[:, None], axis=-1)
    dd = np.where(np.isfinite(d), d, np.inf)
    seg_mask = np.zeros(P.shape[:2], bool)
    for s, a0, b0 in segs:
        seg_mask[a0:b0, s] = True
    nb = seg_mask[np.arange(len(P)), np.argmin(dd, 1)] & (dd.min(1) < 3)
    L = np.array([b0 - a0 for _, a0, b0 in segs]) / 25 if segs else np.zeros(0)
    rep = {"votes": int(len(fr)), "vote_disagree_share": float(np.mean(vote != team[slot])),
           "colour_segments": len(segs_all), "segments_moved": len(segs), "count_check": not a.all, "pairs": a.pairs,
           "segment_len_s_median": float(np.median(L)) if len(L) else None,
           "moved_slot_frames_per_graded_frame": float(seg_mask[g].sum() / g.sum()),
           "frames_nearest_to_ball_moved": float(nb[g].mean()),
           "frames_12v10_before": float(np.mean((c0[g].max(1) > 11) & (c0[g].min(1) < 11))),
           "frames_12v10_after": float(np.mean((c1[g].max(1) > 11) & (c1[g].min(1) < 11))),
           "frames_over_11_before": float(np.mean((c0[g] > 11).any(1))),
           "frames_over_11_after": float(np.mean((c1[g] > 11).any(1))),
           "frames_11v11_before": float(np.mean((c0[g] == 11).all(1))),
           "frames_11v11_after": float(np.mean((c1[g] == 11).all(1))),
           "win": a.win, "min_votes": a.min_votes, "gap": a.gap, "slots_after": int(r.P.shape[1])}
    name = "players_reteam{}.npz".format("_all" if a.all else "_pairs" if a.pairs else "")
    np.savez(OUT / name, P=r.P, slot_team=r.slot_team, slot_is_gk=r.slot_is_gk,
             segments=np.array(segs), report=json.dumps(rep))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
