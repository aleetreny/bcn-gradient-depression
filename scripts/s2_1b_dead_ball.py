"""S2.1b: audit and calibrate the dead-ball simulation (review 003, point 2).

    .venv311/bin/python scripts/s2_1b_dead_ball.py [--out experiments/outputs/022]

The calibration reads NO labels (since 2026-10-24: the rules forbid tuning on the public window, and
review 004 found the first version used the public D and the true D of two Sportec matches):
1. Target: visible ball on the full-match BAR-ATH frames that the label-free stoppage proxy marks as slow
   (players slow, >= 12 visible) -- distance to the nearest player, share within 1.24 m, speed median, p75.
2. Grid over the three v4 constants (dead_roll_prob, dead_speed_median, dead_hold_max_m): the same
   statistics on the slow-proxy frames of the 7 degraded Sportec matches; best -> configs/degrade_v3.json
   (= degrade_v2 + dead_ball_model "atd"). It selects the same constants as the first version (0.7 / 3.5 / 1.5).
The labels (public D, Sportec D) appear only in the audit below, which reports and never selects.
3. Audit on all 7 matches, v2 (ball glued to a player) vs v3, against the ATD: per-condition
   statistics, AUC of the ball speed alone (D vs alive) and a 6-ball-feature GBM probe (CV by match
   for Sportec, by contiguous blocks for the public window).
"""
from __future__ import annotations

import argparse
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
from src.data.common import AWAY, DEAD, FPS, HOME  # noqa: E402
from src.data.degrade import degrade, stoppage_proxy  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402

GRID = {"dead_roll_prob": (0.3, 0.5, 0.7), "dead_speed_median": (1.5, 2.5, 3.5), "dead_hold_max_m": (1.5, 2.5, 3.5)}
PROBE = ["ball_speed", "ball_speed_m0.8s", "ball_speed_m2s", "dist_ball_H", "dist_ball_A", "ball_accel"]


def ball_stats(m, mask):
    vis = m.ball_xy_visible
    B = m.B[:, :2]
    sp = np.full(m.n_frames, np.nan)
    ok = vis[2:] & vis[:-2]
    sp[1:-1][ok] = np.linalg.norm(B[2:][ok] - B[:-2][ok], axis=1) * FPS / 2
    d = np.linalg.norm(m.P[..., :2] - B[:, None], axis=-1)
    d = np.where(np.isnan(d), np.inf, d).min(1)
    mm = mask & vis & np.isfinite(d)
    return {"visible_share": float(vis[mask].mean()), "n_visible": int(mm.sum()), "dist_median": float(np.median(d[mm])),
            "share_within_1.24m": float(np.mean(d[mm] <= 1.24)), "speed_median": float(np.nanmedian(sp[mm])),
            "speed_p75": float(np.nanpercentile(sp[mm], 75))}, sp


def slow_mask(m):
    """Label-free stoppage frames: the slow-players proxy with enough players visible, inside the halves."""
    return stoppage_proxy(m.P, m.slot_visible, m.half) & (m.slot_visible.sum(1) >= 12) & (m.half > 0)


def loss(s, t):
    return sum(((s[k] - t[k]) / t[k]) ** 2 for k in ("dist_median", "share_within_1.24m", "speed_median", "speed_p75"))


def speed_auc(sp, y_dead, mask):
    from sklearn.metrics import roc_auc_score
    ok = mask & np.isfinite(sp)
    return float(roc_auc_score(y_dead[ok].astype(int), -sp[ok]))


def probe_auc(frames):
    """frames: list of (X, y_dead, group). LightGBM on PROBE features, CV by group."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    X = pd.concat([f[0][PROBE] for f in frames], ignore_index=True)
    y = np.concatenate([f[1] for f in frames]).astype(int)
    g = np.concatenate([np.full(len(f[1]), f[2]) for f in frames])
    pred = np.zeros(len(y))
    for k in np.unique(g):
        te = g == k
        clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, verbose=-1, random_state=0).fit(X[~te], y[~te])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/022")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    atd = load_atd()
    gt = metrics.public_gt()
    gt = np.where(gt == 3, -1, gt).astype(np.int8)
    pub_dead, pub_alive = gt == DEAD, np.isin(gt, (HOME, AWAY))
    public_D, sp_atd = ball_stats(atd, pub_dead)          # audit only
    target, _ = ball_stats(atd, slow_mask(atd) & atd.graded)

    base = load_config(ROOT / "configs/degrade_v2.json")
    sportec = load_all()
    grid = []
    for values in itertools.product(*GRID.values()):
        over = dict(zip(GRID, values))
        cfg = replace(base, dead_ball_model="atd", **over)
        stats = [ball_stats(d, slow_mask(d))[0] for d in (degrade(m, cfg, 0) for m in sportec.values())]
        mean = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        grid.append({**over, **mean, "loss": loss(mean, target)})
    grid = pd.DataFrame(grid).sort_values("loss")
    best = grid.iloc[0][list(GRID)].to_dict()
    v3 = replace(base, dead_ball_model="atd", **best)
    cfg_dict = {k: v for k, v in asdict(v3).items() if k != "params"}
    cfg_dict["_provenance"] = "degrade_v2 + v4 dead-ball model calibrated in S2.1b without labels (ATD vs degraded Sportec slow-proxy frames)"
    (ROOT / "configs" / "degrade_v3.json").write_text(json.dumps(cfg_dict, indent=2))

    rows = [{"source": "ATD full, slow proxy (target)", **target}, {"source": "ATD public D (audit)", **public_D}]
    aucs = {"ATD public": {"speed_auc": speed_auc(sp_atd, pub_dead, pub_dead | pub_alive)}}
    Xa = cached_features(atd, "atd")
    lab = gt >= 0
    blocks = np.minimum(np.cumsum(lab) * 5 // max(lab.sum(), 1), 4)
    aucs["ATD public"]["probe_auc"] = probe_auc([(Xa[lab & (blocks == b)], pub_dead[lab & (blocks == b)], b) for b in range(5)])
    for name, cfg in (("Sportec degraded v2 (handled ball)", base), ("Sportec degraded v3 (ATD dead ball)", v3)):
        stats, frames, sps = [], [], []
        for i, (mid, m) in enumerate(sportec.items()):
            d = degrade(m, cfg, 0)
            st, sp = ball_stats(d, m.labels == DEAD)
            stats.append(st)
            lm = m.labels >= 0
            sps.append(speed_auc(sp, m.labels == DEAD, lm))
            X = cached_features(d, "audit_{}".format(mid))
            frames.append((X[lm], (m.labels == DEAD)[lm], i))
        rows.append({"source": name, **{k: float(np.mean([s[k] for s in stats])) for k in stats[0]}})
        aucs[name] = {"speed_auc": float(np.mean(sps)), "probe_auc": probe_auc(frames)}
    audit = pd.DataFrame(rows).set_index("source")
    auc_df = pd.DataFrame(aucs).T

    grid.to_csv(args.out / "grid.csv", index=False, float_format="%.4f")
    audit.to_csv(args.out / "dead_ball_audit.csv", float_format="%.3f")
    auc_df.to_csv(args.out / "dead_ball_aucs.csv", float_format="%.3f")
    (args.out / "summary.json").write_text(json.dumps({"best": best, "target_slow_proxy": target}, indent=2))
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(grid.head(5))
        print(audit)
        print(auc_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
