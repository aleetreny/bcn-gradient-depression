"""F4.10b screen: does training with ball-player incoherence (degrade_v6, "ghosts") make the GBM more robust?

    .venv311/bin/python scripts/s4_11_ghost_screen.py [--out experiments/outputs/047]

GBM v1ctl only, decoder parameters fixed (each fold's v1ctl GBM parameters; a screen). Per outer match T:
  v3 -> v3 : stored F3.7 outer prediction (the current GBM, test degraded with degrade_v3)
  v6 -> v3 : trained on degrade_v6, test degraded with degrade_v3 (does it hurt the ordinary domain?)
  v3 -> v6 : the current recipe on a test degraded with degrade_v6 (the ATD-like incoherent domain)
  v6 -> v6 : trained and tested with ghosts
Metrica Sample with the full models (configs/decoder_v1ctl_gbm.json). Writes summary.json with the
criteria fixed before the run (experiments/log/046): pass = Sample(v6) >= Sample(v3) - 0.003,
LOMO(v6->v3) >= LOMO(v3->v3) - 0.005 and LOMO(v6->v6) >= LOMO(v3->v6) + 0.01.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from s3_7_nested_full import TRAIN_VARIANTS, build_features  # noqa: E402
from s3_11_tcn_local import without_dead_margin  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.models.gbm import TwoHeadGBM  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/047")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sportec = load_all()
    ref = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl"
    cfgs = {t: load_config(ROOT / "configs" / "degrade_{}.json".format(t)) for t in ("v3", "v6")}
    TX, testX = {}, {}
    for t, cfg in cfgs.items():
        for mid, m in sportec.items():
            for key, over, s in TRAIN_VARIANTS:
                c = replace(cfg, **over) if over else cfg
                TX[(t, mid, key)] = build_features(degrade(m, c, s), "sportec_{}_degrade_{}_{}".format(mid, t, key), "v1ctl")
            testX[(t, mid)] = build_features(degrade(m, cfg, 0), "sportec_{}_degrade_{}_s0".format(mid, t), "v1ctl")
        print("features", t, flush=True)

    def fit(t, ids, seed):
        Xs = [TX[(t, mid, k)] for mid in ids for k, _, _ in TRAIN_VARIANTS]
        ys = [sportec[mid].labels for mid in ids for _ in TRAIN_VARIANTS]
        gs = [i for i, mid in enumerate(ids) for _ in TRAIN_VARIANTS]
        return TwoHeadGBM(seed=seed).fit(Xs, ys, gs)

    rows = []
    for k, T in enumerate(MATCH_IDS):
        m = sportec[T]
        others = [x for x in MATCH_IDS if x != T]
        params = DecoderParams(**json.loads((ref / T / "params_gbm.json").read_text()))

        def sc(p):
            return metrics.score(m.labels, decode(p, m.half, params), m.half)["spell_f1_mean"]
        g3, g6 = fit("v3", others, k), fit("v6", others, k)
        row = {"fold": T, "v3->v3": sc(np.load(ref / T / "gbm_outer_{}.npy".format(T))),
               "v6->v3": sc(g6.predict_proba(testX[("v3", T)])), "v3->v6": sc(g3.predict_proba(testX[("v6", T)])),
               "v6->v6": sc(g6.predict_proba(testX[("v6", T)]))}
        rows.append(row)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")

    dec = DecoderParams(**{kk: v for kk, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not kk.startswith("_")})
    full = {"v3": pickle.loads((ROOT / "local_data/models/full_degrade_v3_v1ctl/models.pkl").read_bytes())["gbm"], "v6": fit("v6", list(MATCH_IDS), 99)}
    srows = []
    for g in GAMES:
        ms = load_sample(g)
        X = build_features(ms, "sample_{}".format(g), "v1ctl")
        r = {"game": g}
        for t, model in full.items():
            pred = decode(model.predict_proba(X), ms.half, dec)
            r[t] = metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"]
            r[t + "_noD"] = metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"]
        srows.append(r)
        print(r, flush=True)
    samp = pd.DataFrame(srows).set_index("game")
    samp.to_csv(args.out / "sample.csv", float_format="%.4f")
    L = folds.mean()
    S = samp.mean()
    crit = {"sample_ok": bool(S["v6"] >= S["v3"] - 0.003), "ordinary_ok": bool(L["v6->v3"] >= L["v3->v3"] - 0.005),
            "robust_gain": bool(L["v6->v6"] >= L["v3->v6"] + 0.01)}
    summary = {"lomo": L.round(4).to_dict(), "sample": S.round(4).to_dict(), "criteria": crit, "pass": all(crit.values())}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
