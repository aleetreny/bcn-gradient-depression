"""Build a submission end to end from components (deterministic): raw data -> features -> models -> decoder -> CSV.

    .venv311/bin/python scripts/make_submission.py --name v1_2 --decoder-config configs/decoder_m3.json --transition
    .venv311/bin/python scripts/make_submission.py --name v2 --decoder-config ... --transition --tcn --proba mean

Components, all trained on the 7 Sportec matches degraded with --degrade-config (seeds 1, 2 and the
static_hide_prob 0.5 variant):
  - two-head GBM (always; deterministic LightGBM);
  - --transition: M3 transition head (deterministic LightGBM), giving P(change) to the decoder;
  - --tcn: M9 TCN (trained on CPU for reproducibility), whose probabilities can replace or be averaged
    with the GBM's (--proba) and whose change head can feed the decoder (--change).
Decoded with the semi-Markov decoder (--decoder-config), D outside the halves and in the second half
before its kickoff. With --control every model uses the v1ctl feature set (v1 + change + control
features, F3.7). --tcn-norm channel uses the local TCN (F3.11); --save-models keeps the trained models
in local_data/models/<name>/models.pkl (to score them elsewhere, e.g. Metrica Sample, with the same CPU weights).
Writes submissions/<name>.csv and prints its SHA-256. make_submission_v1.py stays
as the frozen builder of v1 / v1.1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_submission_v1 import TRAIN_VARIANTS, ensure_degradation_params  # noqa: E402
from s2_1_degradation import load_config  # noqa: E402
from games import GAMES, anchor_video_frame  # noqa: E402
from src.data.common import DEAD, LABEL_CHARS  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.features.control import cached_control  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.gbm import DEFAULT_PARAMS, TwoHeadGBM  # noqa: E402
from src.models.transition import PARAMS as TRANS_PARAMS, TransitionGBM, change_features, change_targets  # noqa: E402
from starter.baseline import write_submission  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--degrade-config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--decoder-config", type=Path, required=True)
    parser.add_argument("--transition", action="store_true")
    parser.add_argument("--tcn", action="store_true")
    parser.add_argument("--proba", choices=("gbm", "tcn", "mean"), default="gbm")
    parser.add_argument("--change", choices=("none", "trans", "tcn", "mean"), default=None)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--control", action="store_true", help="add the F3.6 control features to every model (feature set v1ctl)")
    parser.add_argument("--tcn-norm", choices=("group", "channel"), default="group")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    change = args.change or ("trans" if args.transition else "none")
    print("degradation_params.json sha256 {}".format(ensure_degradation_params()))

    cfg = load_config(args.degrade_config)

    def feats(m, key):
        """v1 features for the GBM; with --control the control features are appended (as in F3.7)."""
        X = cached_features(m, key)
        if args.control:
            X = pd.concat([change_features(X, m.half), cached_control(m, key)], axis=1)
        return X

    tag = args.degrade_config.stem
    sportec = load_all()
    Xs, ys, hs, gs = [], [], [], []
    for i, (mid, m) in enumerate(sportec.items()):
        for key, over, s in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            dm = degrade(m, c, s)
            key_ = "sportec_{}_{}_{}".format(mid, tag, key)
            Xs.append(feats(dm, key_))
            ys.append(m.labels)
            hs.append(m.half)
            gs.append(i)
    atd = load_atd()
    Xa = feats(atd, "atd")
    half_dec = np.where(atd.graded, atd.half, 0)

    gbm = TwoHeadGBM(params=dict(DEFAULT_PARAMS, deterministic=True, force_row_wise=True), calibrate=True, seed=args.seed).fit(Xs, ys, gs)
    p_gbm = gbm.predict_proba(Xa)
    p_change, p_tcn, c_tcn = None, None, None
    if args.transition or args.tcn:
        if args.control:  # change features already included (feature set v1ctl, F3.7)
            Xcs, Xca = Xs, Xa
        else:
            Xcs = [change_features(X, h) for X, h in zip(Xs, hs)]
            Xca = change_features(Xa, atd.half)
    if args.transition:
        th = TransitionGBM(params=dict(TRANS_PARAMS, deterministic=True, force_row_wise=True), seed=args.seed)
        p_trans = th.fit(Xcs, [change_targets(y, h) for y, h in zip(ys, hs)]).predict(Xca)
    if args.tcn:
        import torch
        from src.models import tcn as tcn_mod
        torch.use_deterministic_algorithms(True)
        tcn_mod._device = lambda: torch.device("cpu")  # CPU for a reproducible final build
        model = tcn_mod.TCNModel(seed=args.seed, norm=args.tcn_norm).fit(Xcs, ys, hs)
        p_tcn, c_tcn = model.predict(Xca, atd.half)
    proba = {"gbm": p_gbm, "tcn": p_tcn, "mean": None if p_tcn is None else (p_gbm + p_tcn) / 2}[args.proba]
    if change == "trans":
        p_change = p_trans
    elif change == "tcn":
        p_change = c_tcn
    elif change == "mean":
        p_change = (p_trans + c_tcn) / 2

    dec = DecoderParams(**{k: v for k, v in json.loads(args.decoder_config.read_text()).items() if not k.startswith("_")})
    pred = decode(proba, half_dec, dec, p_change=p_change)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    pred[second_start:anchor_video_frame(GAMES["BAR-ATH"]["kickoff_2nd"])] = DEAD
    pred[~atd.graded] = DEAD
    out = ROOT / "submissions" / "{}.csv".format(args.name)
    write_submission(str(out), LABEL_CHARS[pred])
    np.savez(ROOT / "local_data" / "{}_inputs.npz".format(args.name), proba=proba.astype(np.float32),
             p_change=np.zeros(1) if p_change is None else p_change.astype(np.float32))
    if args.save_models:
        import pickle
        mdir = ROOT / "local_data" / "models" / args.name
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "models.pkl").write_bytes(pickle.dumps({"gbm": gbm, "trans": th if args.transition else None,
                                                        "tcn": model if args.tcn else None, "args": vars(args)}))
    print("wrote {}  sha256 {}".format(out, hashlib.sha256(out.read_bytes()).hexdigest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
