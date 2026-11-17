"""S2.3: per-frame models (two-head LightGBM) in LOMO on degraded Sportec, without a decoder yet.

    .venv311/bin/python scripts/s2_3_frame_models.py [--out experiments/outputs/017] [--no-calibration]

Train (per fold, 6 matches): degraded with --config (default configs/degrade_v2.json) seeds 1 and 2, plus a variant with
less static-ball hiding (static_hide_prob 0.5, seed 1) as augmentation. Test: the held-out match
degraded with seed 0 (paired with the baseline on the same degradation), and the clean held-out match.
Decoding here is only a centred moving average of the probabilities + argmax (windows 0..100 frames).
Then the model trained on all 7 matches predicts BAR-ATH: public score and plausibility panel.
Out-of-fold test probabilities are saved to local_data/oof/s2_3_<tag>/<match>.npy for the decoder (S2.4).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from src.data.common import DEAD, FPS, HOME, LABEL_CHARS, possession_spells  # noqa: E402
from src.data.degrade import VERSION as DEG_VERSION, degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.baseline import predict as baseline  # noqa: E402
from src.models.gbm import TwoHeadGBM, smooth_argmax  # noqa: E402
from starter.baseline import write_submission  # noqa: E402

WINDOWS = (0, 12, 25, 50, 100)
TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))


def plausibility(pred, half, graded):
    s, e, _ = possession_spells(np.where(graded, pred, -1), half)
    d = (e - s) / FPS
    return {"poss_per_min": len(s) / (graded.sum() / FPS / 60), "pct_D": 100 * float(np.mean(pred[graded] == DEAD)),
            "median_poss_s": float(np.median(d)), "pct_poss_lt_2s": 100 * float(np.mean(d < 2))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/017")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v2.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = "d{}_{}".format(DEG_VERSION, args.config.stem)
    sportec = load_all()

    feats, labels, halves, test_deg, test_clean = {}, {}, {}, {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            feats[(mid, key)] = cached_features(degrade(m, c, seed), "sportec_{}_{}_{}".format(mid, tag, key))
        d0 = degrade(m, cfg, 0)
        test_deg[mid] = (cached_features(d0, "sportec_{}_{}_s0".format(mid, tag)), d0)
        test_clean[mid] = cached_features(m, "sportec_{}_clean".format(mid))
        labels[mid], halves[mid] = m.labels, m.half

    rows, rel, imps = [], [], []
    for fold, test_id in enumerate(sportec):
        train_ids = [k for k in sportec if k != test_id]
        Xs = [feats[(mid, key)] for mid in train_ids for key, _, _ in TRAIN_VARIANTS]
        ys = [labels[mid] for mid in train_ids for _ in TRAIN_VARIANTS]
        gs = [i for i, mid in enumerate(train_ids) for _ in TRAIN_VARIANTS]
        model = TwoHeadGBM(calibrate=not args.no_calibration, seed=fold).fit(Xs, ys, gs)
        imps.append(model.importance(30))
        Xd, md = test_deg[test_id]
        y, half = labels[test_id], halves[test_id]
        pa, ph = model.heads(Xd)
        proba = np.stack([pa * ph, pa * (1 - ph), 1 - pa], axis=1)
        oof_dir = ROOT / "local_data" / "oof" / "s2_3_{}".format(tag)
        oof_dir.mkdir(parents=True, exist_ok=True)
        np.save(oof_dir / "{}.npy".format(test_id), proba.astype(np.float32))
        alive = (y >= 0) & (y != DEAD)
        lab = y >= 0
        from sklearn.metrics import roc_auc_score
        row = {"fold": test_id, "alive_auc": float(roc_auc_score((y[lab] != DEAD).astype(int), pa[lab])),
               "team_acc_on_true_alive": float(np.mean((ph[alive] > 0.5) == (y[alive] == HOME))),
               "frame_acc_argmax": float(np.mean(np.argmax(proba[lab], 1) == y[lab])),
               "baseline_spellf1": metrics.score(y, baseline(md, 50), half)["spell_f1_mean"]}
        for w in WINDOWS:
            r = metrics.score(y, smooth_argmax(proba, half, w), half)
            row["spellf1_w{}".format(w)] = r["spell_f1_mean"]
            row["macrof1_w{}".format(w)] = r["macro_f1"]
            row["poss_w{}".format(w)] = r["n_pred_spells"]
        row["n_gt_spells"] = r["n_gt_spells"]
        pc = model.predict_proba(test_clean[test_id])
        row["clean_spellf1_w50"] = metrics.score(y, smooth_argmax(pc, half, 50), half)["spell_f1_mean"]
        rows.append(row)
        bins = np.linspace(0, 1, 11)
        for name, p, t in (("alive", pa[lab], (y[lab] != DEAD)), ("team", ph[alive], (y[alive] == HOME))):
            idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
            for b in range(10):
                sel = idx == b
                if sel.any():
                    rel.append({"head": name, "fold": test_id, "bin": b, "pred": float(p[sel].mean()), "obs": float(t[sel].mean()), "n": int(sel.sum())})
        print("[{}] {}".format(test_id, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k != "fold"}), flush=True)

    folds = pd.DataFrame(rows).set_index("fold")
    means = folds.mean(numeric_only=True)
    best_w = max(WINDOWS, key=lambda w: means["spellf1_w{}".format(w)])

    # full model -> BAR-ATH
    Xs = [feats[(mid, key)] for mid in sportec for key, _, _ in TRAIN_VARIANTS]
    ys = [labels[mid] for mid in sportec for _ in TRAIN_VARIANTS]
    gs = [i for i, mid in enumerate(sportec) for _ in TRAIN_VARIANTS]
    full = TwoHeadGBM(calibrate=not args.no_calibration, seed=99).fit(Xs, ys, gs)
    atd = load_atd()
    Xa = cached_features(atd, "atd")
    pa_atd = full.predict_proba(Xa)
    pred_atd = smooth_argmax(pa_atd, atd.half, best_w)
    pred_atd = np.where(atd.graded, pred_atd, DEAD).astype(np.int8)
    pub = metrics.score_public(pred_atd, bootstrap_n=1000)
    plaus = plausibility(pred_atd, atd.half, atd.graded)
    sub_dir = ROOT / "submissions"
    write_submission(str(sub_dir / "s2_3_gbm_w{}.csv".format(best_w)), LABEL_CHARS[pred_atd])
    np.save(ROOT / "local_data" / "s2_3_proba_atd.npy", pa_atd.astype(np.float32))
    np.save(ROOT / "local_data" / "oof" / "s2_3_{}".format(tag) / "atd_full_model.npy", pa_atd.astype(np.float32))

    rel_df = pd.DataFrame(rel)
    folds.to_csv(args.out / "lomo_folds.csv", float_format="%.4f")
    rel_df.to_csv(args.out / "reliability.csv", index=False, float_format="%.4f")
    imp = {h: pd.concat([i[h] for i in imps], axis=1).mean(axis=1).sort_values(ascending=False).head(20).round(4).to_dict()
           for h in ("alive", "team")}
    summary = {"best_window": best_w, "means": means.round(4).to_dict(),
               "lomo_std_best": float(folds["spellf1_w{}".format(best_w)].std(ddof=0)),
               "folds_better_than_baseline_best": int((folds["spellf1_w{}".format(best_w)] > folds["baseline_spellf1"]).sum()),
               "public": metrics.headline(pub), "plausibility_bar_ath": plaus, "importance": imp,
               "calibrated": not args.no_calibration}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plot_reliability(rel_df, args.out / "reliability.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(folds)
    print(json.dumps({k: v for k, v in summary.items() if k != "importance"}, indent=1, default=float))
    print(json.dumps(imp, indent=1))
    return 0


def plot_reliability(rel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid = "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for ax, head, title in ((axes[0], "alive", "P(alive): predicted vs observed"), (axes[1], "team", "P(H | alive): predicted vs observed")):
        d = rel[rel["head"] == head]
        agg = d.groupby("bin").apply(lambda x: pd.Series({"pred": np.average(x["pred"], weights=x["n"]), "obs": np.average(x["obs"], weights=x["n"])}))
        for _, f in d.groupby("fold"):
            ax.plot(f["pred"], f["obs"], color="#2a78d6", alpha=0.25, lw=1)
        ax.plot(agg["pred"], agg["obs"], color="#2a78d6", lw=2.2, marker="o", ms=4, label="pooled (lines: folds)")
        ax.plot([0, 1], [0, 1], color=ink, ls=":", lw=0.8)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=grid, lw=0.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
