"""Report-only predictor for an explicitly hashed local BAR-ATH CSV.

It is not a model and must not be used to select or tune a submission. J3 is
the only compatible judge; the existing harness controls scoring and records.
"""
import hashlib
from pathlib import Path

import numpy as np

from evaluation.possession_metrics import label_codes, read_label_csv, validate_submission

ROOT = Path(__file__).resolve().parents[2]
FILES = {"original": "j3_report_original.csv", "retrained": "j3_report_retrained.csv"}


def public_csv_predictor(which: str, sha256: str):
    if which not in FILES or len(sha256) != 64:
        raise ValueError("a named local CSV and its full SHA-256 are required")
    path = ROOT / "_local" / FILES[which]

    def fit_predict(train, test):
        if test.source != "metrica_atd" or test.meta.get("game") != "BAR-ATH":
            raise ValueError("this report-only predictor only accepts BAR-ATH")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != sha256:
            raise ValueError("the local CSV does not match its registered hash")
        frame = read_label_csv(path)
        reason = validate_submission(frame, test.n_frames)
        if reason:
            raise ValueError(reason)
        labels = label_codes(frame)
        if not np.isin(labels, (0, 1, 2)).all():
            raise ValueError("J3 report requires H/A/D labels")
        return labels.astype(np.int8)

    return fit_predict
