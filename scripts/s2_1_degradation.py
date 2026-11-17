"""S2.1: calibrate the Sportec -> ATD degradation and check it with an adversarial classifier.

    .venv311/bin/python scripts/s2_1_degradation.py [--out experiments/outputs/015] [--trials 60]

1. Calibration (Optuna, 2 calibration matches, fixed seeds): player parameters against the ATD's
   visible-players-per-team distribution and fragment births per minute; then ball parameters
   against P(ball visible | moving), P(ball visible | slow), gaps per 90 min and median gap length.
   Targets are the ATD statistics measured with the same code (S1.3/S1.4).
2. Statistics of the calibrated degradation on all 7 matches vs the ATD.
3. Adversarial validation: LightGBM tries to tell BAR-ATH frames from Sportec frames (clean, then
   degraded) using the slot-invariant features (+ smoothness); grouped 5-fold CV by 1-minute blocks.
   AUC and the 10 most revealing features.
4. New reference row: the baseline (smooth50) LOMO on degraded Sportec (3 degradation seeds).
The calibrated config is saved to configs/degrade_v1.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import AWAY, FPS, HOME, runs  # noqa: E402
from src.data.degrade import DegradeConfig, degrade, stoppage_proxy  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.frame_features import basic_features, jitter_feature  # noqa: E402
from src.models.baseline import predict  # noqa: E402

CALIB = ("J03WMX", "J03WQQ")
SEEDS = (0, 1, 2)


def load_config(path) -> DegradeConfig:
    raw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(Path(path).read_text()).items()
           if not k.startswith("_")}
    return DegradeConfig(**raw)


def stats(m):
    g = m.half > 0
    vis = m.slot_visible
    bl = vis.sum(1) < 12
    slow = stoppage_proxy(m.P, vis, m.half)
    bv = m.ball_xy_visible
    key = np.where(g, (~bv).astype(int) + 10 * m.half, -1)
    a, e, v = runs(key)
    gl = (e - a)[(v % 10 == 1) & (v >= 10)] / FPS
    minutes = g.sum() / FPS / 60
    out = {"ball_visible": float(bv[g].mean()), "p_vis_moving": float(bv[g & ~slow & ~bl].mean()),
           "p_vis_slow": float(bv[g & slow & ~bl].mean()), "slow_share": float(slow[g].mean()),
           "low_slot_share": float(bl[g].mean()), "gaps_per_90": len(gl) / minutes * 90,
           "gap_median_s": float(np.median(gl)) if len(gl) else np.nan,
           "gap_p75_s": float(np.percentile(gl, 75)) if len(gl) else np.nan}
    buckets = []
    for code in (HOME, AWAY):
        c = vis[:, m.slot_team == code].sum(1)[g & ~bl]
        buckets.append([np.mean(c <= 8), np.mean(c == 9), np.mean(c == 10), np.mean(c == 11), np.mean(c >= 12)])
    out["visible_buckets_le8_9_10_11_ge12"] = (100 * np.mean(buckets, axis=0)).round(1).tolist()
    out["births_per_min"] = float((vis[1:] & ~vis[:-1])[g[1:]].sum() / minutes)
    bxy = m.B[:, :2]
    ok = bv[2:] & bv[:-2] & g[1:-1]
    bsp = np.linalg.norm(bxy[2:][ok] - bxy[:-2][ok], axis=1) * FPS / 2
    out["ball_static_share"] = float(np.mean(bsp < 0.5)) if len(bsp) else np.nan
    d2 = m.P[2:, :, :2] - 2 * m.P[1:-1, :, :2] + m.P[:-2, :, :2]
    out["jitter_m"] = float(np.nanmedian(np.linalg.norm(d2, axis=-1)))
    return out


BUCKET_WEIGHTS = np.array([1, 1, 1, 1, 5])  # iteration 2: duplicates (>= 12 visible) were the most revealing count


def player_loss(s, t):
    b = np.array(s["visible_buckets_le8_9_10_11_ge12"]) - np.array(t["visible_buckets_le8_9_10_11_ge12"])
    return float(np.sum(BUCKET_WEIGHTS * b ** 2) / 100 + ((s["births_per_min"] - t["births_per_min"]) / t["births_per_min"]) ** 2 * 4)


def ball_loss(s, t):
    return float(((s["p_vis_moving"] - t["p_vis_moving"]) / 0.05) ** 2 + ((s["p_vis_slow"] - t["p_vis_slow"]) / 0.03) ** 2
                 + ((s["gaps_per_90"] - t["gaps_per_90"]) / 25) ** 2 + ((s["gap_median_s"] - t["gap_median_s"]) / 3) ** 2
                 + ((s["ball_static_share"] - t["ball_static_share"]) / 0.015) ** 2)


def mean_stats(ms, cfg, seeds=(0,)):
    rows = [stats(degrade(m, cfg, seed)) for m in ms for seed in seeds]
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if not isinstance(rows[0][k], list)}
    out["visible_buckets_le8_9_10_11_ge12"] = np.mean([r["visible_buckets_le8_9_10_11_ge12"] for r in rows], axis=0).round(1).tolist()
    return out


def calibrate(ms, target, trials, base=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    keep_players = base is not None
    base = base or DegradeConfig()

    def players(trial):
        cfg = replace(base, components=("smooth", "absences", "fragments", "duplicates"),
                      absence_rate=trial.suggest_float("absence_rate", 1e-5, 2e-3, log=True),
                      episode_rate=trial.suggest_float("episode_rate", 1e-5, 2e-3, log=True),
                      cut_rate=trial.suggest_float("cut_rate", 1e-6, 1e-3, log=True),
                      dup_prob=trial.suggest_float("dup_prob", 0.0, 1.0),
                      dup_len_s=(1.0, trial.suggest_float("dup_len_max_s", 2.0, 15.0)))
        return player_loss(mean_stats(ms, cfg), target)

    if keep_players:
        cfg, player_value = base, None
    else:
        sp = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        sp.optimize(players, n_trials=trials)
        best = dict(sp.best_params)
        dup_max = best.pop("dup_len_max_s")
        cfg, player_value = replace(base, **best, dup_len_s=(1.0, dup_max)), sp.best_value

    def ball(trial):
        c = replace(cfg, components=("smooth", "absences", "fragments", "duplicates", "ball"),
                    gap_rate_moving=trial.suggest_float("gap_rate_moving", 1e-5, 3e-3, log=True),
                    gap_rate_slow=trial.suggest_float("gap_rate_slow", 1e-4, 3e-2, log=True),
                    gap_len_scale=trial.suggest_float("gap_len_scale", 0.3, 1.5),
                    static_hide_prob=trial.suggest_float("static_hide_prob", 0.0, 1.0),
                    static_min_s=trial.suggest_float("static_min_s", 0.2, 3.0))
        return ball_loss(mean_stats(ms, c), target)

    sb = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    sb.optimize(ball, n_trials=trials)
    return replace(cfg, **sb.best_params), {"player_loss": player_value, "ball_loss": sb.best_value}


QUALITY = ["ball_visible", "ball_frames_since_seen", "ball_frames_until_seen", "ball_speed", "n_visible_H", "n_visible_A",
           "n_visible_all", "n_within_3m_H", "n_within_3m_A", "n_within_10m_H", "n_within_10m_A", "dist_ball_H",
           "dist_ball_A", "dist_ball_diff"]
STYLE = ["centroid_x_H", "centroid_x_A", "centroid_y_H", "centroid_y_A", "spread_x_H", "spread_x_A", "spread_y_H",
         "spread_y_A", "speed_mean_H", "speed_mean_A", "speed_p90_H", "speed_p90_A", "frac_slow_H", "frac_slow_A",
         "speed_mean_all_2s", "ball_x", "ball_y"]


def adversarial(atd, others, name, use_jitter=True, columns=None):
    """AUC of LightGBM separating BAR-ATH graded frames from the given matches (every 10th frame)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    def frames(m, domain, gid):
        X = basic_features(m)
        if columns is not None:
            X = X[columns].copy()
        if use_jitter:
            X["jitter"] = jitter_feature(m)
        rows = np.flatnonzero(m.half > 0)[::10]
        X = X.iloc[rows].copy()
        X["_y"] = domain
        X["_g"] = ["{}_{}".format(gid, r // (60 * FPS)) for r in rows]
        return X

    data = pd.concat([frames(atd, 1, "atd")] + [frames(m, 0, mid) for mid, m in others.items()], ignore_index=True)
    y, groups = data.pop("_y").to_numpy(), data.pop("_g").to_numpy()
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    pred = np.zeros(len(y))
    imp = np.zeros(data.shape[1])
    for tr, te in GroupKFold(5).split(data, y, groups):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        clf.fit(data.iloc[tr], y[tr], sample_weight=w[tr])
        pred[te] = clf.predict_proba(data.iloc[te])[:, 1]
        imp += clf.booster_.feature_importance("gain")
    top = pd.Series(imp / imp.sum(), index=data.columns).sort_values(ascending=False).head(10)
    return {"name": name, "auc": float(roc_auc_score(y, pred, sample_weight=w)), "top_features": top.round(3).to_dict()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/015")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--config", type=Path, default=None, help="evaluate this saved config instead of calibrating")
    parser.add_argument("--recalibrate-ball", type=Path, default=None,
                        help="keep the player parameters of this config and recalibrate only the ball ones")
    parser.add_argument("--save", type=Path, default=ROOT / "configs/degrade_v1.json", help="where a calibrated config is written")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    atd = load_atd()
    sportec = load_all()
    target = stats(atd)
    if args.config:
        cfg, losses = load_config(args.config), {"note": "evaluated saved config {}".format(args.config)}
    else:
        base = load_config(args.recalibrate_ball) if args.recalibrate_ball else None
        cfg, losses = calibrate([sportec[m] for m in CALIB], target, args.trials, base=base)
        (ROOT / "configs").mkdir(exist_ok=True)
        args.save.write_text(json.dumps({k: v for k, v in asdict(cfg).items() if k != "params"}, indent=2))
    cfg_dict = {k: v for k, v in asdict(cfg).items() if k != "params"}

    # 2. stats on all matches
    rows = {"ATD BAR-ATH": target, "Sportec clean (mean)": mean_stats(list(sportec.values()), replace(cfg, components=()), (0,))}
    rows["Sportec degraded v1 (mean, 3 seeds)"] = mean_stats(list(sportec.values()), cfg, SEEDS)
    comp = pd.DataFrame(rows)

    # 3. adversarial
    degraded0 = {mid: degrade(m, cfg, 0) for mid, m in sportec.items()}
    adv = []
    for label, ms in (("clean", sportec), ("degraded v1", degraded0)):
        adv.append(adversarial(atd, ms, "Sportec {}: all features + smoothness".format(label)))
        adv.append(adversarial(atd, ms, "Sportec {}: all features, no smoothness".format(label), use_jitter=False))
        adv.append(adversarial(atd, ms, "Sportec {}: tracking-quality features".format(label), use_jitter=False, columns=QUALITY))
        adv.append(adversarial(atd, ms, "Sportec {}: playing-style features".format(label), use_jitter=False, columns=STYLE))

    # 4. baseline LOMO on degraded Sportec
    lomo_rows = []
    for seed in SEEDS:
        dm = {mid: replace(degrade(m, cfg, seed)) for mid, m in sportec.items()}
        folds = metrics.lomo(dm, lambda train, test: predict(test, 50), name="seed{}".format(seed), verbose=False)
        s = metrics.lomo_summary(folds)
        lomo_rows.append({"seed": seed, **s, "pct_pred_D": float(np.mean([np.mean(predict(m, 50) == 2) for m in dm.values()]))})
    lomo = pd.DataFrame(lomo_rows).set_index("seed")

    comp.to_csv(args.out / "degradation_vs_atd.csv")
    lomo.to_csv(args.out / "baseline_lomo_degraded.csv", float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps({"config": cfg_dict, "calibration_loss": losses, "adversarial": adv,
                                                       "baseline_lomo_degraded_mean": float(lomo["lomo_spellf1_media"].mean()),
                                                       "baseline_lomo_degraded_seed_std": float(lomo["lomo_spellf1_media"].std(ddof=0))},
                                                      indent=2, default=float))
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(json.dumps(cfg_dict, indent=1, default=str))
        print(comp)
        for a in adv:
            print("\nAUC {:.3f}  {}".format(a["auc"], a["name"]))
            print("  ", a["top_features"])
        print(lomo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
