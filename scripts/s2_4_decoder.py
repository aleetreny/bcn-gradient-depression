"""S2.4: tune the semi-Markov decoder on the S2.3 out-of-fold probabilities, maximising LOMO spellF1.

    .venv311/bin/python scripts/s2_4_decoder.py [--oof local_data/oof/s2_3_d3_degrade_v2] [--trials 120]
                                                [--out experiments/outputs/018]

Nested LOMO: for each held-out match, Optuna (TPE) maximises mean spellF1 on the other 6 matches;
the decoder parameters are the centre of the plateau (median of the top 10 % of trials), evaluated
on the held-out match (paired with the S2.3 smoothing, w 12). A final study on all 7 matches gives
the parameters for BAR-ATH, a plateau plot and a +/-20 % robustness check. BAR-ATH: the full S2.3
model's probabilities decoded, with the hard constraint that the second half before its kickoff is D.
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
sys.path.insert(0, str(ROOT / "scripts"))
from s2_3_frame_models import plausibility  # noqa: E402
from src.data.common import DEAD, LABEL_CHARS  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.gbm import smooth_argmax  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from games import GAMES, anchor_video_frame  # noqa: E402

SPACE = {"lam_ha": (0.5, 40.0), "lam_ld": (0.5, 40.0), "lam_dl": (0.5, 40.0), "dmin_team": (1, 50), "dmin_dead": (1, 150)}
INT_PARAMS = ("dmin_team", "dmin_dead")


def suggest(trial):
    p = {}
    for k, (lo, hi) in SPACE.items():
        p[k] = trial.suggest_int(k, lo, hi) if k in INT_PARAMS else trial.suggest_float(k, lo, hi, log=True)
    return DecoderParams(**p)


def mean_spellf1(params, data, ids):
    return float(np.mean([metrics.score(data[i][1], decode(data[i][0], data[i][2], params), data[i][2])["spell_f1_mean"] for i in ids]))


def study(data, ids, trials, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(lambda t: mean_spellf1(suggest(t), data, ids), n_trials=trials)
    return st


def plateau_center(st, top=0.10):
    df = st.trials_dataframe(attrs=("value", "params")).dropna()
    df = df.sort_values("value", ascending=False).head(max(5, int(len(df) * top)))
    p = {k: float(np.median(df["params_" + k])) for k in SPACE}
    for k in INT_PARAMS:
        p[k] = int(round(p[k]))
    return DecoderParams(**p), df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=ROOT / "local_data/oof/s2_3_d3_degrade_v2")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/018")
    parser.add_argument("--nested", type=Path, default=None,
                        help="read outer probabilities from local_data/oof/nested_<tag>/<T>/outer_<T>.npy (S2.4b)")
    parser.add_argument("--save-config", type=Path, default=None, help="write the final decoder parameters here")
    parser.add_argument("--atd-proba", type=Path, default=None, help="BAR-ATH probabilities to decode (default: <oof>/atd_full_model.npy)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sportec = load_all()
    def oof_path(mid):
        return args.nested / mid / "outer_{}.npy".format(mid) if args.nested else args.oof / "{}.npy".format(mid)
    data = {mid: (np.load(oof_path(mid)), m.labels, m.half) for mid, m in sportec.items()}

    rows = []
    for k, test in enumerate(MATCH_IDS):
        train = [i for i in MATCH_IDS if i != test]
        st = study(data, train, args.trials, seed=k)
        center, _ = plateau_center(st)
        best = DecoderParams(**st.best_params)
        proba, y, half = data[test]
        r_center = metrics.score(y, decode(proba, half, center), half)
        rows.append({"fold": test, "train_best": st.best_value, "test_center": r_center["spell_f1_mean"],
                     "test_best_trial": metrics.score(y, decode(proba, half, best), half)["spell_f1_mean"],
                     "test_smooth_w12": metrics.score(y, smooth_argmax(proba, half, 12), half)["spell_f1_mean"],
                     "macro_center": r_center["macro_f1"], "poss_pred": r_center["n_pred_spells"], "poss_gt": r_center["n_gt_spells"],
                     **{"p_" + kk: v for kk, v in center.to_dict().items()}})
        print("[{}] {}".format(test, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rows[-1].items() if kk != "fold"}), flush=True)
    folds = pd.DataFrame(rows).set_index("fold")

    # final parameters on all 7, plateau and robustness
    st_all = study(data, list(MATCH_IDS), args.trials, seed=99)
    final, top = plateau_center(st_all)
    robust = []
    for k in SPACE:
        for f in (0.8, 1.2):
            p = final.to_dict()
            p[k] = p[k] * f
            if k in INT_PARAMS:
                p[k] = max(1, int(round(p[k])))
            robust.append({"param": k, "factor": f, "lomo_all_spellf1": mean_spellf1(DecoderParams(**p), data, list(MATCH_IDS))})
    robust = pd.DataFrame(robust)
    final_value = mean_spellf1(final, data, list(MATCH_IDS))

    # BAR-ATH
    atd = load_atd()
    proba_atd = np.load(args.atd_proba or (args.oof / "atd_full_model.npy"))
    if args.save_config:
        args.save_config.write_text(json.dumps({**final.to_dict(), "_provenance": "S2.4 plateau centre on 7 matches, probabilities {}".format(
            args.nested or args.oof)}, indent=2))
    pred = decode(proba_atd, np.where(atd.graded, atd.half, 0), final)
    kickoff2 = anchor_video_frame(GAMES["BAR-ATH"]["kickoff_2nd"])
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    pred[second_start:kickoff2] = DEAD
    pred[~atd.graded] = DEAD
    pub = metrics.score_public(pred, bootstrap_n=1000)
    plaus = plausibility(pred, atd.half, atd.graded)
    write_submission(str(ROOT / "submissions" / "s2_4_gbm_hsmm.csv"), LABEL_CHARS[pred])

    folds.to_csv(args.out / "lomo_folds.csv", float_format="%.4f")
    robust.to_csv(args.out / "robustness.csv", index=False, float_format="%.4f")
    st_all.trials_dataframe().to_csv(args.out / "trials_all.csv", index=False)
    summary = {"lomo_center_mean": float(folds["test_center"].mean()), "lomo_center_std": float(folds["test_center"].std(ddof=0)),
               "lomo_smooth_w12_mean": float(folds["test_smooth_w12"].mean()),
               "folds_better_than_smooth": int((folds["test_center"] > folds["test_smooth_w12"]).sum()),
               "final_params": final.to_dict(), "final_all7_spellf1": final_value,
               "second_half_pre_kickoff_frames_forced_D": [second_start, kickoff2 - 1],
               "public": metrics.headline(pub), "plausibility_bar_ath": plaus}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plot(st_all, final, args.out / "plateau.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(folds)
        print(robust)
    print(json.dumps(summary, indent=1, default=float))
    return 0


def plot(st, final, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grid = "#52514e", "#e4e3df"
    df = st.trials_dataframe(attrs=("value", "params")).dropna()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, axes = plt.subplots(1, len(SPACE), figsize=(15, 3.4), sharey=True, constrained_layout=True)
    for ax, k in zip(axes, SPACE):
        ax.scatter(df["params_" + k], df["value"], s=12, color="#2a78d6", alpha=0.6)
        ax.axvline(final.to_dict()[k], color="#eb6834", lw=1.5)
        if k not in INT_PARAMS:
            ax.set_xscale("log")
        ax.set_xlabel(k)
        ax.grid(color=grid, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean spellF1, 7 matches")
    fig.suptitle("Decoder plateau: every Optuna trial (orange = chosen plateau centre)", x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
