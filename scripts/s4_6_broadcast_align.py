"""F4.6: align the broadcast ATD to the tactical timeline (no mapping table is provided).

    .venv311/bin/python scripts/s4_6_broadcast_align.py [--out experiments/outputs/044]

Both trackings are in the same pitch coordinates (normalised 0-1 x 105.3 x 68 m, same attack
directions). The broadcast has cuts and replays, so the mapping is found per continuous stretch of
valid broadcast homography (122 stretches): one constant offset t_tactical = k_broadcast + L per
stretch, chosen to minimise the median distance from each visible broadcast player to the nearest
visible tactical player, on up to 25 sampled frames. The first long stretch searches every offset;
later ones search from 2 min before to 25 min after the previous offset (replays go back in time). The
result (local_data/broadcast_alignment.npz) gives, for every broadcast frame, its tactical frame
(-1 = not aligned) and the fit cost in metres. Tracking only, no labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.metrica_atd import load as load_atd, parse_meta, parse_raw  # noqa: E402

PITCH = np.array([105.3, 68.0], np.float32)


def load_broadcast():
    cache = ROOT / "local_data" / "cache" / "broadcast_raw.npz"
    if cache.is_file():
        z = np.load(cache)
        raw, team = z["raw"], z["slot_team"]
    else:
        meta = parse_meta(ROOT / "data/broadcast/BAR-ATH_broadcast_atd.xml")
        team = meta["slot_team"]
        _, raw = parse_raw(ROOT / "data/broadcast/BAR-ATH_broadcast_atd.txt", len(team))
        np.savez(cache, raw=raw, slot_team=team, slot_is_gk=meta["slot_is_gk"])
    S = len(team)
    return raw[:, :S, :2] * PITCH, raw[:, S, :2] * PITCH, team


def cost_matrix(Pb_frames, Pt, cand):
    """median nearest-neighbour distance for each sampled broadcast frame (rows) and candidate offset (cols)."""
    out = np.full((len(Pb_frames), len(cand)), np.inf, np.float32)
    for i, (k, pts) in enumerate(Pb_frames):
        t = k + cand
        ok = (t >= 0) & (t < len(Pt))
        if not ok.any() or len(pts) < 5:
            continue
        T = Pt[t[ok]]                                          # (n, 46, 2)
        d = np.linalg.norm(T[:, None, :, :] - pts[None, :, None, :], axis=-1)   # (n, nb, 46)
        d = np.where(np.isfinite(d), d, np.inf).min(2)        # nearest tactical player for each broadcast player
        out[i, ok] = np.median(d, axis=1)
    return out


def best_offset(frames, Pt, lo, hi, step):
    cand = np.arange(lo, hi + 1, step)
    c = np.nanmedian(np.where(np.isfinite(cost := cost_matrix(frames, Pt, cand)), cost, np.nan), axis=0)
    c = np.where(np.isfinite(c), c, np.inf)
    j = int(np.argmin(c))
    return int(cand[j]), float(c[j])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/044")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    atd = load_atd()
    Pt = atd.P[..., :2].astype(np.float32)
    Pb, Bb, _ = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    valid = hom["valid"].to_numpy().astype(bool)
    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    segs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 75]
    mapping = np.full(len(Pb), -1, np.int64)
    fit = np.full(len(Pb), np.nan, np.float32)
    rows = []
    prev = None
    for a, b in segs:
        ks = np.unique(np.linspace(a, b - 1, 25).astype(int))
        frames = [(k, Pb[k][np.isfinite(Pb[k, :, 0])]) for k in ks]
        if prev is None:
            L, c = best_offset(frames[::3], Pt, -len(Pb), len(Pt), 25)
            L, c = best_offset(frames, Pt, L - 50, L + 50, 1)
        else:
            L, c = best_offset(frames[::3], Pt, prev - 3000, prev + 37500, 10)
            L, c = best_offset(frames, Pt, L - 20, L + 20, 1)
        replay = prev is not None and L < prev - 25
        rows.append({"start": a, "stop": b, "seconds": (b - a) / 25, "offset": L, "cost_m": c, "replay": replay})
        print(rows[-1], flush=True)
        if c < 1.5 and not replay:
            mapping[a:b] = np.arange(a, b) + L
            fit[a:b] = c
            prev = L
        elif prev is None:
            prev = L
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "segments.csv", index=False, float_format="%.3f")
    np.savez(ROOT / "local_data" / "broadcast_alignment.npz", tactical_frame=mapping, cost=fit)
    ok = mapping >= 0
    summary = {"segments": len(table), "aligned_segments": int(((table.cost_m < 1.5) & ~table.replay).sum()),
               "replays": int(table.replay.sum()), "broadcast_frames_aligned": int(ok.sum()),
               "tactical_graded_covered": float(np.isin(np.flatnonzero(atd.graded), mapping[ok]).mean()),
               "median_cost_m": float(np.nanmedian(fit))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
