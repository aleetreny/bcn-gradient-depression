"""F4.3: final decoders of a nested store, and Metrica Sample scored with the exact CPU models of a build.

    # 1. final decoder = plateau centre on the 7 nested outer predictions (as F3.8), per variant
    .venv311/bin/python scripts/s4_3_final.py decoders --store nested_degrade_v3_v1ctl_tcnlocal --name v1ctl_tcnlocal
    # 2. after make_submission.py --save-models: the same models (CPU) on the Sample games and the public window
    .venv311/bin/python scripts/s4_3_final.py sample --models local_data/models/v4/models.pkl \
        --decoder-config configs/decoder_v1ctl_tcnlocal_ens_mean.json --proba mean --change mean

Closes review 004 finding 2: the Sample score is computed with the weights that produce the CSV.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s3_1_transition_head import mean_score, plateau, suggest  # noqa: E402
from s3_7_nested_full import VARIANTS, build_features  # noqa: E402
from s3_11_tcn_local import without_dead_margin  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402


def decoders(args) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    store = ROOT / "local_data" / "oof" / args.store
    sportec = load_all()
    for how in args.variants.split(","):
        proba_kind, change_kind = VARIANTS[how]
        data = {}
        for T in MATCH_IDS:
            g = np.load(store / T / "gbm_outer_{}.npy".format(T))
            tr = np.load(store / T / "trans_outer_{}.npy".format(T))
            z = np.load(store / T / "tcn_outer_{}.npz".format(T))
            p = g if proba_kind == "gbm" else (g + z["proba"]) / 2
            c = None if change_kind == "none" else (tr if change_kind == "trans" else (tr + z["change"]) / 2)
            data[T] = (p, sportec[T].labels, sportec[T].half, c)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
        st.optimize(lambda t: mean_score(suggest(t), data, list(MATCH_IDS)), n_trials=args.trials)
        best = plateau(st)
        path = ROOT / "configs" / "decoder_{}_{}.json".format(args.name, how)
        path.write_text(json.dumps({**best.to_dict(), "_provenance": "s4_3: plateau centre on the 7 nested outer predictions ({}, {})".format(
            args.store, how)}, indent=2))
        print(path.name, best, flush=True)


def sample(args) -> None:
    import torch
    from src.models import tcn as tcn_mod
    torch.use_deterministic_algorithms(True)
    tcn_mod._device = lambda: torch.device("cpu")  # same device as the build
    models = pickle.loads(Path(args.models).read_bytes())
    dec = DecoderParams(**{k: v for k, v in json.loads(Path(args.decoder_config).read_text()).items() if not k.startswith("_")})
    feature_set = "v1ctl" if models["args"]["control"] else "v1"
    rows = {}
    for g in GAMES:
        ms = load_sample(g)
        X = build_features(ms, "sample_{}".format(g), feature_set)
        p_gbm = models["gbm"].predict_proba(X)
        p_tcn, c_tcn = models["tcn"].predict(X, ms.half) if models["tcn"] is not None else (None, None)
        p_tr = models["trans"].predict(X) if models["trans"] is not None else None
        proba = {"gbm": p_gbm, "tcn": p_tcn, "mean": None if p_tcn is None else (p_gbm + p_tcn) / 2}[args.proba]
        change = {"none": None, "trans": p_tr, "tcn": c_tcn, "mean": None if c_tcn is None else (p_tr + c_tcn) / 2}[args.change]
        pred = decode(proba, ms.half, dec, p_change=change)
        rows[g] = (metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"],
                   metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"])
        print("sample", g, "spellF1 %.4f  without D+-3s %.4f" % rows[g], flush=True)
    out = {"models": str(args.models), "sample": {g: v[0] for g, v in rows.items()}, "sample_noD": {g: v[1] for g, v in rows.items()},
           "sample_mean": float(np.mean([v[0] for v in rows.values()])), "sample_noD_mean": float(np.mean([v[1] for v in rows.values()]))}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decoders")
    d.add_argument("--store", required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--variants", default="ens_mean,ens_m3")
    d.add_argument("--trials", type=int, default=150)
    s = sub.add_parser("sample")
    s.add_argument("--models", required=True)
    s.add_argument("--decoder-config", required=True)
    s.add_argument("--proba", choices=("gbm", "tcn", "mean"), default="mean")
    s.add_argument("--change", choices=("none", "trans", "tcn", "mean"), default="mean")
    s.add_argument("--out", default=None)
    args = parser.parse_args()
    decoders(args) if args.cmd == "decoders" else sample(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
