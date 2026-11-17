"""F3.8: final decoders, Metrica Sample transfer and BAR-ATH public check for the F3.7 variants.

    .venv311/bin/python scripts/s3_8_candidates.py --features v1ctl [--trials 150] [--out experiments/outputs/034]

For each decoder variant of F3.7 (gbm, gbm_m3, ens_mean, ens_m3):
  1. final decoder = plateau centre of an Optuna study on the 7 nested outer predictions
     -> configs/decoder_<features>_<variant>.json;
  2. full models (GBM, M3, TCN) trained once on the 7 degraded Sportec matches with the feature set,
     saved in local_data/models/full_<degradation>_<features>/;
  3. Metrica Sample (3 games) and BAR-ATH public scores (TCN on MPS here; the reproducible build
     uses scripts/make_submission.py).
The nested LOMO score of each variant comes from F3.7 (experiments/outputs/033_<features>/).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from s2_3_frame_models import plausibility  # noqa: E402
from s3_1_transition_head import mean_score, plateau, suggest  # noqa: E402
from s3_7_nested_full import TRAIN_VARIANTS, VARIANTS, build_features  # noqa: E402
from games import GAMES as GAME_INFO, anchor_video_frame  # noqa: E402
from src.data.common import DEAD  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.eval.store import check_store, identity  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402
from src.models.transition import TransitionGBM, change_targets  # noqa: E402


def compose(how, g, tr, tp, tc):
    proba_kind, change_kind = VARIANTS[how]
    proba = g if proba_kind == "gbm" else (g + tp) / 2
    change = None if change_kind == "none" else (tr if change_kind == "trans" else (tr + tc) / 2)
    return proba, change


def main() -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--features", choices=("v1", "v1ctl", "v1ctl2"), default="v1ctl")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/034")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = args.config.stem
    store = ROOT / "local_data" / "oof" / "nested_{}_{}".format(tag, args.features)
    sportec = load_all()

    # 1. final decoders
    finals = {}
    for how in VARIANTS:
        data = {}
        for T in MATCH_IDS:
            z = np.load(store / T / "tcn_outer_{}.npz".format(T))
            p, c = compose(how, np.load(store / T / "gbm_outer_{}.npy".format(T)), np.load(store / T / "trans_outer_{}.npy".format(T)),
                           z["proba"], z["change"])
            data[T] = (p, sportec[T].labels, sportec[T].half, c)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
        st.optimize(lambda t: mean_score(suggest(t), data, list(MATCH_IDS)), n_trials=args.trials)
        finals[how] = plateau(st)
        (ROOT / "configs" / "decoder_{}_{}.json".format(args.features, how)).write_text(json.dumps(
            {**finals[how].to_dict(), "_provenance": "s3_8: plateau centre on the 7 nested outer predictions ({}, {}, {})".format(
                tag, args.features, how)}, indent=2))
        print("decoder", how, finals[how], flush=True)

    # 2. full models
    mpath = ROOT / "local_data" / "models" / "full_{}_{}".format(tag, args.features) / "models.pkl"
    check_store(mpath.parent, identity(asdict(cfg), args.features, sportec, {"train_variants": TRAIN_VARIANTS, "seed": 99}))
    if mpath.is_file():
        models = pickle.loads(mpath.read_bytes())
    else:
        Xs, ys, hs, gs = [], [], [], []
        for i, (mid, m) in enumerate(sportec.items()):
            for key, over, s in TRAIN_VARIANTS:
                c = replace(cfg, **over) if over else cfg
                Xs.append(build_features(degrade(m, c, s), "sportec_{}_{}_{}".format(mid, tag, key), args.features))
                ys.append(m.labels)
                hs.append(m.half)
                gs.append(i)
        models = {"gbm": TwoHeadGBM(seed=99).fit(Xs, ys, gs),
                  "trans": TransitionGBM(seed=99).fit(Xs, [change_targets(y, h) for y, h in zip(ys, hs)]),
                  "tcn": TCNModel(seed=99).fit(Xs, ys, hs)}
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_bytes(pickle.dumps(models))

    def predict_all(m, key):
        X = build_features(m, key, args.features)
        tp, tc = models["tcn"].predict(X, m.half)
        return models["gbm"].predict_proba(X), models["trans"].predict(X), tp, tc

    # 3. Metrica Sample and BAR-ATH public
    rows = []
    sample_inputs = {g: (load_sample(g), predict_all(load_sample(g), "sample_{}".format(g))) for g in GAMES}
    atd = load_atd()
    atd_inputs = predict_all(atd, "atd")
    half_dec = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    for how in VARIANTS:
        sc = []
        for g, (ms, inp) in sample_inputs.items():
            p, c = compose(how, *inp)
            sc.append(metrics.score(ms.labels, decode(p, ms.half, finals[how], p_change=c), ms.half)["spell_f1_mean"])
        p, c = compose(how, *atd_inputs)
        pred = decode(p, half_dec, finals[how], p_change=c)
        pred[second_start:kick2] = DEAD
        pred[~atd.graded] = DEAD
        np.save(ROOT / "local_data" / "pred_{}_{}_atd.npy".format(args.features, how), pred)
        pub = metrics.score_public(pred, bootstrap_n=1000)
        pl = plausibility(pred, atd.half, atd.graded)
        rows.append({"variant": how, "sample_mean": float(np.mean(sc)), "sample_1": sc[0], "sample_2": sc[1], "sample_3": sc[2],
                     "public_spellF1": pub["spell_f1_mean"], "public_lo": pub["spell_bootstrap_low"], "public_hi": pub["spell_bootstrap_high"],
                     "public_D_f1": pub["per_class"]["D"]["f1"], "public_poss": pub["n_pred_spells"],
                     "bar_ath_poss_per_min": pl["poss_per_min"], "bar_ath_pct_D": pl["pct_D"]})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows).set_index("variant")
    nested = pd.read_csv(ROOT / "experiments" / "outputs" / "033_{}".format(args.features) / "folds.csv", index_col=0)
    table.insert(0, "nested_lomo", [nested[v].mean() for v in table.index])
    table.to_csv(args.out / "candidates_{}.csv".format(args.features), float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
