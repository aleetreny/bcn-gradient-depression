"""Evaluate explicitly fingerprinted v11 prediction bundles on J1 and J2.

This replays existing OOF/external predictions; it does not train a model or
establish the provenance or independence of the caches that produced them.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from src.data.sportec import MATCH_IDS
from src.features.v1 import match_fingerprint

ROOT = Path(__file__).resolve().parents[2]
FILES = {"original": "external_original.pkl", "retrained": "external_retrained.pkl"}


def _verified(path, expected):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"input hash mismatch: {path.name}")
    return path


def external_bundle_predictor(which: str, bundle_sha256: str, fingerprints_sha256: str):
    if which not in FILES:
        raise ValueError("unknown external prediction bundle")
    base = ROOT / "_local"
    with _verified(base / "external_fingerprints.json", fingerprints_sha256).open() as handle:
        fingerprints = json.load(handle)
    with _verified(base / FILES[which], bundle_sha256).open("rb") as handle:
        bundle = pickle.load(handle)["d3"]

    def fit_predict(train, test):
        if test.source == "sportec":
            mid = test.meta.get("match_id")
            if mid not in MATCH_IDS:
                raise ValueError("unexpected Sportec match")
            expected_train = {f"{m}_s1" for m in MATCH_IDS if m != mid}
            if set(train) != expected_train:
                raise ValueError("Sportec training fold does not exclude the held-out match")
            key = f"sportec_{mid}_degrade_v3_s0"
            prediction = bundle["preds_v3"][mid]
        elif test.source == "metrica_sample":
            game = test.meta.get("game")
            if game not in (1, 2, 3) or set(train) != {f"{m}_s1" for m in MATCH_IDS}:
                raise ValueError("unexpected Metrica test or Sportec training fold")
            key = f"sample_{game}_raw"
            prediction = bundle["preds_sample"][game]
        else:
            raise ValueError("cached external predictor accepts only J1 and J2")
        if match_fingerprint(test) != fingerprints.get(key):
            raise ValueError(f"test tracking fingerprint mismatch for {key}")
        prediction = np.asarray(prediction)
        if prediction.shape != (test.n_frames,) or not np.isin(prediction, (0, 1, 2)).all():
            raise ValueError(f"invalid cached predictions for {key}")
        return prediction.astype(np.int8, copy=False)

    return fit_predict
