"""Build submission v1 end to end, deterministically: raw data -> features -> GBM -> decoder -> CSV.

    .venv311/bin/python scripts/make_submission_v1.py [--out submissions/v1.csv]

v1 = two-head LightGBM (S2.3) trained on the 7 Sportec matches degraded with configs/degrade_v2.json
(seeds 1 and 2 + the static_hide_prob 0.5 variant), applied to the BAR-ATH ATD, decoded with the
semi-Markov decoder (configs/decoder_v1.json); D outside the halves and in the second half before
its kickoff. LightGBM runs with deterministic=True / force_row_wise=True so two runs give the same
file. Prints the SHA-256 of the CSV. Needs only the raw data and the versioned configs: the aggregate
ATD statistics (local_data/degradation_params.json) are regenerated from the raw ATD when missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from games import GAMES, anchor_video_frame  # noqa: E402
from src.data.common import DEAD, LABEL_CHARS  # noqa: E402
from src.data.degrade import VERSION as DEG_VERSION, degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.features.v1 import cached_features  # noqa: E402
from src.models.gbm import DEFAULT_PARAMS, TwoHeadGBM  # noqa: E402
from starter.baseline import write_submission  # noqa: E402

TRAIN_VARIANTS = (("s1", None, 1), ("s2", None, 2), ("static05_s1", {"static_hide_prob": 0.5}, 1))


def ensure_degradation_params(rebuild: bool = False) -> str:
    """local_data/degradation_params.json (aggregate ATD statistics used by the degradation) is not in
    git: build it deterministically from the raw ATD with the S1.3 and S1.4 scripts if missing."""
    import subprocess
    path = ROOT / "local_data" / "degradation_params.json"
    if rebuild or not path.is_file():
        for script in ("s1_3_ball_gaps.py", "s1_4_slot_anatomy.py"):  # S1.4 merges into S1.3's file
            subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / script)], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path, degrade_cfg: Path, decoder_cfg: Path, seed: int = 99) -> np.ndarray:
    cfg = load_config(degrade_cfg)
    tag = "d{}_{}".format(DEG_VERSION, degrade_cfg.stem)
    sportec = load_all()
    Xs, ys, gs = [], [], []
    for i, (mid, m) in enumerate(sportec.items()):
        for key, over, s in TRAIN_VARIANTS:
            c = replace(cfg, **over) if over else cfg
            Xs.append(cached_features(degrade(m, c, s), "sportec_{}_{}_{}".format(mid, tag, key)))
            ys.append(m.labels)
            gs.append(i)
    params = dict(DEFAULT_PARAMS, deterministic=True, force_row_wise=True)
    model = TwoHeadGBM(params=params, calibrate=True, seed=seed).fit(Xs, ys, gs)
    atd = load_atd()
    proba = model.predict_proba(cached_features(atd, "atd"))
    dec = {k: v for k, v in json.loads(decoder_cfg.read_text()).items() if not k.startswith("_")}
    pred = decode(proba, np.where(atd.graded, atd.half, 0), DecoderParams(**dec))
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    pred[second_start:anchor_video_frame(GAMES["BAR-ATH"]["kickoff_2nd"])] = DEAD
    pred[~atd.graded] = DEAD
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(out), LABEL_CHARS[pred])
    np.save(ROOT / "local_data" / "{}_proba.npy".format(out.stem), proba.astype(np.float32))
    return pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "submissions/v1.csv")
    parser.add_argument("--degrade-config", type=Path, default=ROOT / "configs/degrade_v2.json")
    parser.add_argument("--decoder-config", type=Path, default=ROOT / "configs/decoder_v1.json")
    parser.add_argument("--rebuild-params", action="store_true", help="regenerate local_data/degradation_params.json")
    args = parser.parse_args()
    print("degradation_params.json sha256 {}".format(ensure_degradation_params(args.rebuild_params)))
    build(args.out, args.degrade_config, args.decoder_config)
    print("wrote {}  sha256 {}".format(args.out, hashlib.sha256(args.out.read_bytes()).hexdigest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
