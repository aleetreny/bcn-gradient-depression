"""F4.1: decoder robustness -- each parameter moved by -20 %, -10 %, +10 %, +20 %, one at a time.

    .venv311/bin/python scripts/s4_1_robustness.py --variant ens_mean [--store nested_degrade_v3_v1ctl]
        [--atd-inputs local_data/v3_inputs.npz --decoder-config configs/decoder_v1ctl_ens_mean.json]

1. Nested LOMO: for each outer match T, its own nested decoder parameters (params_<variant>.json, tuned
   on the inner predictions) scaled one at a time, applied to the stored outer prediction of T.
2. BAR-ATH: the final decoder scaled the same way, applied to the stored probabilities of the build
   (make_submission.py writes local_data/<name>_inputs.npz) -> possessions per minute, % D, public score.
A candidate on a plateau keeps the LOMO within ~0.01 and the BAR-ATH possessions per minute stable.
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
from s2_3_frame_models import plausibility  # noqa: E402
from s3_7_nested_full import VARIANTS  # noqa: E402
from games import GAMES as GAME_INFO, anchor_video_frame  # noqa: E402
from src.data.common import DEAD  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402

FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
PARAMS = ("lam_ha", "lam_ld", "lam_dl", "dmin_team", "dmin_dead", "beta_ha")


def scaled(p: DecoderParams, name: str, f: float) -> DecoderParams:
    v = getattr(p, name) * f
    return replace(p, **{name: max(1, int(round(v))) if name.startswith("dmin") else float(v)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="ens_mean", choices=tuple(VARIANTS))
    parser.add_argument("--store", default="nested_degrade_v3_v1ctl")
    parser.add_argument("--atd-inputs", type=Path, default=None)
    parser.add_argument("--decoder-config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/040")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    store = ROOT / "local_data" / "oof" / args.store
    proba_kind, change_kind = VARIANTS[args.variant]
    sportec = load_all()
    params = [p for p in PARAMS if change_kind != "none" or p != "beta_ha"]

    outer = {}
    for T in MATCH_IDS:
        g = np.load(store / T / "gbm_outer_{}.npy".format(T))
        tr = np.load(store / T / "trans_outer_{}.npy".format(T))
        z = np.load(store / T / "tcn_outer_{}.npz".format(T)) if proba_kind == "mean" or change_kind == "mean" else None
        proba = g if proba_kind == "gbm" else (g + z["proba"]) / 2
        change = None if change_kind == "none" else (tr if change_kind == "trans" else (tr + z["change"]) / 2)
        outer[T] = (proba, change, DecoderParams(**json.loads((store / T / "params_{}.json".format(args.variant)).read_text())))

    rows = []
    for name in params:
        for f in FACTORS:
            if f == 1.0 and name != params[0]:
                continue
            sc, poss = [], []
            for T, (proba, change, p) in outer.items():
                m = sportec[T]
                r = metrics.score(m.labels, decode(proba, m.half, scaled(p, name, f), p_change=change), m.half)
                sc.append(r["spell_f1_mean"])
                poss.append(r["n_pred_spells"] / r["n_gt_spells"])
            rows.append({"param": "base" if f == 1.0 else name, "factor": f, "lomo": float(np.mean(sc)),
                         "lomo_min_fold": float(np.min(sc)), "pred_over_true_poss": float(np.mean(poss))})
            print(rows[-1], flush=True)
    lomo = pd.DataFrame(rows)
    base = lomo.loc[lomo.param == "base", "lomo"].item()
    lomo["delta"] = lomo["lomo"] - base
    lomo.to_csv(args.out / "lomo_{}_{}.csv".format(args.store, args.variant), float_format="%.4f", index=False)

    if args.atd_inputs is not None:
        z = np.load(args.atd_inputs)
        proba, change = z["proba"], (z["p_change"] if z["p_change"].size > 1 else None)
        dec = DecoderParams(**{k: v for k, v in json.loads(args.decoder_config.read_text()).items() if not k.startswith("_")})
        atd = load_atd()
        half_dec = np.where(atd.graded, atd.half, 0)
        second_start = int(np.flatnonzero(atd.half == 2)[0])
        kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
        arows = []
        for name in params:
            for f in FACTORS:
                if f == 1.0 and name != params[0]:
                    continue
                pred = decode(proba, half_dec, scaled(dec, name, f), p_change=change)
                pred[second_start:kick2] = DEAD
                pred[~atd.graded] = DEAD
                pl = plausibility(pred, atd.half, atd.graded)
                arows.append({"param": "base" if f == 1.0 else name, "factor": f, "poss_per_min": pl["poss_per_min"], "pct_D": pl["pct_D"],
                              "public": metrics.score_public(pred, bootstrap_n=0)["spell_f1_mean"]})
        atdt = pd.DataFrame(arows)
        atdt.to_csv(args.out / "atd_{}.csv".format(args.atd_inputs.stem), float_format="%.4f", index=False)
        lomo = lomo.merge(atdt, on=["param", "factor"])
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(lomo.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
