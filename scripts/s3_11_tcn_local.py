"""F3.11: TCN with local normalisation (review 004, finding 5) -- outer LOMO screen, Metrica Sample and public.

    .venv311/bin/python scripts/s3_11_tcn_local.py [--norm channel] [--out experiments/outputs/036]

The v2/v3 TCN uses GroupNorm over channels and time, so each prediction depends on the whole input and
the normalisation context differs between training (60 s crops) and inference (whole halves). Here the
same TCN with norm="channel" (per-frame LayerNorm over channels: a local 511-frame network).
1. Outer LOMO (v1ctl features, same seeds as F3.7): one new TCN per outer match; ens_mean and ens_m3
   decoded with the F3.7 per-fold decoder parameters (tuned on the old TCN: a handicap for the new one).
2. Full TCN on the 7 matches (seed 99), combined with the stored full GBM and M3 of F3.8, on Metrica
   Sample (also with D +/- 3 s left out) and the BAR-ATH public window, with the F3.8 final decoders.
Everything on MPS, compared with the old TCN trained the same way.
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
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from s2_3_frame_models import plausibility  # noqa: E402
from s3_2_tcn_screen import frame_metrics  # noqa: E402
from s3_7_nested_full import TRAIN_VARIANTS, build_features  # noqa: E402
from s3_8_candidates import compose  # noqa: E402
from games import GAMES as GAME_INFO, anchor_video_frame  # noqa: E402
from src.data.common import DEAD, FPS  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.eval.store import check_store, identity  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402


def without_dead_margin(y, half, seconds=3):
    """Labels with D and +/- `seconds` around it set to unlabelled (diagnostic of the team signal)."""
    out = y.copy()
    near = binary_dilation(y == DEAD, iterations=int(seconds * FPS))
    out[near] = -1
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--norm", default="channel")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/036")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = args.config.stem
    old = ROOT / "local_data" / "oof" / "nested_{}_v1ctl".format(tag)
    store = ROOT / "local_data" / "oof" / "tcn_{}_{}_v1ctl".format(args.norm, tag)
    sportec = load_all()
    check_store(old, identity(asdict(cfg), "v1ctl", sportec,
                              {"train_variants": TRAIN_VARIANTS, "script": "s3_7 seeds 1000*(1+model)+k inner, k outer"}),
                ignore=("source",))
    check_store(store, identity(asdict(cfg), "v1ctl", sportec, {"train_variants": TRAIN_VARIANTS, "norm": args.norm,
                                                                "seeds": "outer k, full 99"}))
    labels = {mid: m.labels for mid, m in sportec.items()}
    halves = {mid: m.half for mid, m in sportec.items()}
    train_X, test_X = {}, {}
    for mid, m in sportec.items():
        for key, over, seed in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            train_X[(mid, key)] = build_features(degrade(m, c, seed), "sportec_{}_{}_{}".format(mid, tag, key), "v1ctl")
        test_X[mid] = build_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag), "v1ctl")

    def fit(ids, seed):
        return TCNModel(seed=seed, norm=args.norm).fit([train_X[(m, k)] for m in ids for k, _, _ in TRAIN_VARIANTS],
                                                      [labels[m] for m in ids for _ in TRAIN_VARIANTS],
                                                      [halves[m] for m in ids for _ in TRAIN_VARIANTS])

    # 1. outer LOMO screen
    rows = []
    for k, T in enumerate(MATCH_IDS):
        path = store / T / "tcn_outer_{}.npz".format(T)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            p, c = fit([m for m in MATCH_IDS if m != T], k).predict(test_X[T], halves[T])
            np.savez(path, proba=p, change=c)
        new, prev = np.load(path), np.load(old / T / "tcn_outer_{}.npz".format(T))
        g, tr = np.load(old / T / "gbm_outer_{}.npy".format(T)), np.load(old / T / "trans_outer_{}.npy".format(T))
        y, h = labels[T], halves[T]
        row = {"fold": T}
        for how in ("ens_mean", "ens_m3"):
            params = DecoderParams(**json.loads((old / T / "params_{}.json".format(how)).read_text()))
            for name, z in (("old", prev), ("new", new)):
                p, c = compose(how, g, tr, z["proba"], z["change"])
                row["{}_{}".format(how, name)] = metrics.score(y, decode(p, h, params, p_change=c), h)["spell_f1_mean"]
        for name, z in (("old", prev), ("new", new)):
            for kk, v in frame_metrics(z["proba"], y).items():
                row["tcn_{}_{}".format(name, kk)] = v
        rows.append(row)
        print("[{}] {}".format(T, {kk: round(v, 4) for kk, v in row.items() if kk != "fold"}), flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")

    # 2. full models: Metrica Sample and BAR-ATH public
    full = pickle.loads((ROOT / "local_data" / "models" / "full_{}_v1ctl".format(tag) / "models.pkl").read_bytes())
    mpath = store / "full_tcn.pkl"
    if mpath.is_file():
        tcn_new = pickle.loads(mpath.read_bytes())
    else:
        tcn_new = fit(list(MATCH_IDS), 99)
        mpath.write_bytes(pickle.dumps(tcn_new))
    finals = {how: DecoderParams(**{k: v for k, v in json.loads((ROOT / "configs" / "decoder_v1ctl_{}.json".format(how)).read_text()).items()
                                    if not k.startswith("_")}) for how in ("ens_mean", "ens_m3")}

    def inputs(m, key, tcn):
        X = build_features(m, key, "v1ctl")
        tp, tc = tcn.predict(X, m.half)
        return full["gbm"].predict_proba(X), full["trans"].predict(X), tp, tc

    atd = load_atd()
    half_dec = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    samples = {g: load_sample(g) for g in GAMES}
    out_rows = []
    for name, tcn in (("old", full["tcn"]), ("new", tcn_new)):
        s_in = {g: inputs(ms, "sample_{}".format(g), tcn) for g, ms in samples.items()}
        a_in = inputs(atd, "atd", tcn)
        for how in ("ens_mean", "ens_m3"):
            r = {"tcn": name, "variant": how}
            for g, ms in samples.items():
                p, c = compose(how, *s_in[g])
                pred = decode(p, ms.half, finals[how], p_change=c)
                r["sample_{}".format(g)] = metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"]
                r["sampleNoD_{}".format(g)] = metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"]
            p, c = compose(how, *a_in)
            pred = decode(p, half_dec, finals[how], p_change=c)
            pred[second_start:kick2] = DEAD
            pred[~atd.graded] = DEAD
            pub = metrics.score_public(pred, bootstrap_n=1000)
            pl = plausibility(pred, atd.half, atd.graded)
            r.update({"sample_mean": np.mean([r["sample_{}".format(g)] for g in GAMES]),
                      "sampleNoD_mean": np.mean([r["sampleNoD_{}".format(g)] for g in GAMES]),
                      "public": pub["spell_f1_mean"], "public_D_f1": pub["per_class"]["D"]["f1"], "public_poss": pub["n_pred_spells"],
                      "poss_per_min": pl["poss_per_min"], "pct_D": pl["pct_D"]})
            out_rows.append(r)
            print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
    table = pd.DataFrame(out_rows).set_index(["tcn", "variant"])
    table.to_csv(args.out / "transfer.csv", float_format="%.4f")
    summary = {"lomo_means": folds.mean().round(4).to_dict(),
               "folds_new_better": {how: int((folds[how + "_new"] > folds[how + "_old"]).sum()) for how in ("ens_mean", "ens_m3")},
               "norm": args.norm}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format):
        print(folds)
        print(table)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
