"""Ball-quality features (log 042): is the ball an interpolated straight line here, and how long since and
until a real (curved) detection? Lets a model discount ball-player proximity where the ATD draws the
ball with straight lines.

    from src.features.ballq import ballq_features, cached_ballq
    Q = ballq_features(match)          # DataFrame aligned with build_features(match)

Label-free and slot-invariant: computed from the ball track only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.ball_interp import linear_mask
from src.data.common import FPS, runs

VERSION = 1
CAP = 30 * FPS


def _since(event, half, cap=CAP):
    """Frames since the last True of `event` inside each half (cap if none yet)."""
    out = np.full(len(event), cap, np.float32)
    idx = np.where(event, np.arange(len(event)), -1)
    for h in np.unique(half):
        sel = np.flatnonzero(half == h)
        last = np.maximum.accumulate(idx[sel])
        out[sel] = np.where(last >= 0, np.minimum(sel - last, cap), cap)
    return out


def ballq_features(m) -> pd.DataFrame:
    half = np.asarray(m.half)
    vis = np.isfinite(m.B[:, :2]).all(1)
    lin = linear_mask(m.B, half)
    curved = vis & ~lin
    f = {"ball_linear": np.where(vis, lin, np.nan).astype(np.float32)}
    s, e, v = runs(np.where(vis, lin.astype(int) + 1, 0) * 10 + half)
    run_len = np.zeros(len(lin), np.float32)
    run_pos = np.full(len(lin), np.nan, np.float32)
    for a, b, x in zip(s, e, v):
        if x // 10 == 2:
            run_len[a:b] = min(b - a, CAP)
            run_pos[a:b] = (np.arange(a, b) - a + 0.5) / (b - a)
    f["lin_run_len"] = run_len
    f["lin_run_pos"] = run_pos
    f["since_curved"] = _since(curved, half)
    f["until_curved"] = _since(curved[::-1], half[::-1])[::-1]
    X = pd.DataFrame(f)
    for w in (1.0, 3.0):
        X["ball_linear_m{}s".format(w)] = (pd.Series(np.nan_to_num(f["ball_linear"], nan=0.0)).groupby(pd.Series(half))
                                           .transform(lambda x: x.rolling(int(w * FPS), center=True, min_periods=1).mean())
                                           .to_numpy(np.float32))
    return X


def cached_ballq(m, key: str, cache_dir=None) -> pd.DataFrame:
    from pathlib import Path
    from src.features.v1 import match_fingerprint
    root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "local_data" / "cache" / "ballq_v{}".format(VERSION)
    path = root / "{}_{}.parquet".format(key, match_fingerprint(m))
    if path.is_file():
        return pd.read_parquet(path)
    X = ballq_features(m)
    root.mkdir(parents=True, exist_ok=True)
    X.to_parquet(path)
    return X
