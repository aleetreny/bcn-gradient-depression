"""F4.10a: calibrate the ball-player incoherence ("ghosts") of the simulator without labels.

    .venv311/bin/python scripts/s4_10_ghost_calibration.py   # -> configs/degrade_v6.json

Target: on label-free in-play frames (stoppage proxy) of the full BAR-ATH ATD, the share of slow balls
(< 2 m/s) with nobody within 3 m and of kicks (+4 m/s in 0.2 s) with nobody within 4 m. Grid over the
two ghost rates on 4 degraded Sportec matches (degrade_v3 + ghosts), same statistics, same masks.
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from src.data.degrade import degrade, stoppage_proxy  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402

GRID = {"ghost_occlusion_rate": (0.0, 0.002, 0.005, 0.01, 0.02), "ghost_offset_rate": (0.0, 0.002, 0.005, 0.01)}


def stats(m):
    play = (m.half > 0) & ~stoppage_proxy(m.P, m.slot_visible, m.half)
    B = m.B[:, :2]
    vis = m.ball_xy_visible
    d = np.linalg.norm(m.P[..., :2] - B[:, None], axis=2)
    dmin = np.where(np.isfinite(d), d, np.inf).min(1)
    sp = np.full(len(B), np.nan)
    ok = vis[2:] & vis[:-2]
    sp[1:-1][ok] = np.linalg.norm(B[2:][ok] - B[:-2][ok], axis=1) * 12.5
    acc = np.full(len(B), np.nan)
    acc[5:] = sp[5:] - sp[:-5]
    base = play & vis & np.isfinite(sp)
    return {"slow_ghost": float(np.mean(dmin[base & (sp < 2)] > 3)), "kick_ghost": float(np.mean(dmin[base & (acc > 4)] > 4)),
            "ball_visible_play": float(vis[play].mean())}


def main() -> int:
    target = stats(load_atd())
    print("target", target, flush=True)
    sportec = load_all()
    base = load_config(ROOT / "configs/degrade_v3.json")
    rows = []
    for values in itertools.product(*GRID.values()):
        cfg = replace(base, **dict(zip(GRID, values)))
        st = [stats(degrade(sportec[mid], cfg, 0)) for mid in list(sportec)[:4]]
        mean = {k: float(np.mean([s[k] for s in st])) for k in st[0]}
        loss = sum(((mean[k] - target[k]) / target[k]) ** 2 for k in ("slow_ghost", "kick_ghost"))
        rows.append({**dict(zip(GRID, values)), **mean, "loss": loss})
        print(rows[-1], flush=True)
    grid = pd.DataFrame(rows).sort_values("loss")
    best = grid.iloc[0][list(GRID)].to_dict()
    out = {k: v for k, v in asdict(replace(base, **best)).items() if k != "params"}
    out["_provenance"] = "degrade_v3 + v6 ball-player incoherence (ghosts), rates calibrated in s4_10 without labels against the full BAR-ATH ATD"
    (ROOT / "configs" / "degrade_v6.json").write_text(json.dumps(out, indent=2))
    outdir = ROOT / "experiments" / "outputs" / "047"
    outdir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(outdir / "ghost_grid.csv", index=False, float_format="%.4f")
    print(grid.head(6).round(4).to_string())
    print("best", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
