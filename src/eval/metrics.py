"""Scoring on top of the OFFICIAL evaluator (evaluation/possession_metrics.py), never a re-implementation.

    from src.eval.metrics import score, score_public, lomo
    r = score(gt_labels, pred_labels, half)             # common codes (0 H, 1 A, 2 D, -1 none)
    r = score_public(pred)                              # BAR-ATH public 5 min, with bootstrap
    folds = lomo(matches, fit_predict)                  # leave-one-match-out over common Matches

Common codes are converted to the official ones (H 0, A 1, D 2, X 3). Sources whose halves are
concatenated back to back (Sportec, Metrica Sample) get one X frame inserted at each half boundary
in both arrays, so the official run detection splits the halves exactly as it splits BAR-ATH's
halftime hole. Unlabelled ground truth (-1) becomes X and is not scored.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402
from src.data.common import FPS, LABEL_CHARS, Match  # noqa: E402

PUBLIC_GT = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
HEADLINE = ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells", "spell_count_error",
            "mean_spell_duration_s", "gt_mean_spell_duration_s", "alive_dead_f1", "team_acc_gt_alive")


def to_official(labels: np.ndarray, half: np.ndarray | None = None, insert_gaps: bool = True) -> np.ndarray:
    """Common codes -> official codes, with an X frame between consecutive halves if asked."""
    codes = np.where(np.isin(labels, (0, 1, 2)), labels, pm.X).astype(np.int8)
    if half is None or not insert_gaps:
        return codes
    boundaries = np.flatnonzero((half[1:] != half[:-1]) & (half[1:] > 0) & (half[:-1] > 0)) + 1
    return np.insert(codes, boundaries, pm.X)


def score(gt: np.ndarray, pred: np.ndarray, half: np.ndarray | None = None, insert_gaps: bool = True,
          bootstrap_n: int = 0) -> dict:
    """Official evaluate() on common-code arrays of equal length."""
    if len(gt) != len(pred):
        raise ValueError("length mismatch {} vs {}".format(len(gt), len(pred)))
    g = to_official(gt, half, insert_gaps)
    p = to_official(pred, half, insert_gaps)
    p = np.where(p == pm.X, pm.D, p).astype(np.int8)  # a prediction never abstains here; X only marks gaps
    if half is not None and insert_gaps:
        p[g == pm.X] = pm.X
    return pm.evaluate(g, p, bootstrap_n=bootstrap_n)


def public_gt() -> np.ndarray:
    """Official codes of the public BAR-ATH ground truth (199319 frames, X outside 5 min)."""
    return pm.label_codes(pm.read_label_csv(str(PUBLIC_GT)))


def score_public(pred, bootstrap_n: int = 1000) -> dict:
    """Score a full-length BAR-ATH prediction on the public window exactly as run_evaluation.py does.

    `pred` may be a path to a CSV, an array of label characters, or an array of common codes.
    """
    if isinstance(pred, (str, Path)):
        table = pm.read_label_csv(str(pred))
        reason = pm.validate_submission(table, len(public_gt()))
        if reason:
            raise ValueError(reason)
        codes = pm.label_codes(table)
    elif np.asarray(pred).dtype.kind in "iu":
        codes = to_official(np.asarray(pred))
        codes[codes == pm.X] = pm.D
    else:
        codes = pm.label_codes(np.asarray(pred))
    return pm.evaluate(public_gt(), codes, bootstrap_n=bootstrap_n)


def headline(result: dict) -> dict:
    out = {k: result[k] for k in HEADLINE}
    out["spell_bootstrap"] = (result.get("spell_bootstrap_low"), result.get("spell_bootstrap_high"))
    return out


def possessions_per_min(pred: np.ndarray, half: np.ndarray) -> float:
    from src.data.common import possession_spells
    s, _, _ = possession_spells(pred, half)
    return len(s) / (np.sum(half > 0) / FPS / 60)


def lomo(matches: Dict[str, Match], fit_predict: Callable, name: str = "", verbose: bool = True) -> pd.DataFrame:
    """Leave-one-match-out. fit_predict(train: dict[str, Match], test: Match) -> common-code labels.

    Scores every fold with the official metrics on the held-out match (both halves graded).
    """
    rows = []
    for test_id, test in matches.items():
        train = {k: v for k, v in matches.items() if k != test_id}
        t0 = time.time()
        pred = np.asarray(fit_predict(train, test), dtype=np.int8)
        if pred.shape != test.labels.shape:
            raise ValueError("fit_predict returned shape {} for {}".format(pred.shape, test.labels.shape))
        r = score(test.labels, pred, test.half)
        row = {"fold": test_id, **{k: r[k] for k in HEADLINE},
               "pred_poss_per_min": possessions_per_min(pred, test.half),
               "gt_poss_per_min": possessions_per_min(test.labels, test.half),
               "seconds": time.time() - t0}
        rows.append(row)
        if verbose:
            print("  [{}] {:<8} spellF1 {:.4f}  macroF1 {:.4f}  spells {}/{}".format(
                name, test_id, row["spell_f1_mean"], row["macro_f1"], row["n_pred_spells"], row["n_gt_spells"]))
    return pd.DataFrame(rows).set_index("fold")


def lomo_summary(folds: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict:
    """Mean/std of the LOMO folds, and how many folds improve on a reference run if given."""
    out = {"lomo_spellf1_media": float(folds["spell_f1_mean"].mean()),
           "lomo_std": float(folds["spell_f1_mean"].std(ddof=0)),
           "lomo_macrof1_media": float(folds["macro_f1"].mean()),
           "posesiones_por_min": float(folds["pred_poss_per_min"].mean())}
    if reference is not None:
        diff = folds["spell_f1_mean"] - reference.loc[folds.index, "spell_f1_mean"]
        out["folds_mejorados"] = int((diff > 0).sum())
        out["delta_media"] = float(diff.mean())
    return out


def labels_to_chars(labels: np.ndarray) -> np.ndarray:
    return np.where(np.isin(labels, (0, 1, 2)), LABEL_CHARS[np.clip(labels, 0, 2)], "X")
