"""The judges: the only places a number that drives a decision may come from.

  J0  synthetic   4 synthetic matches, LOMO. Plumbing only; CI runs it. Never a model-selection signal.
  J1  sportec     LOMO on the 7 Sportec/DFL matches degraded to look like the ATD (configs/degrade_v3.json):
                  test copy seed 0, training copies `train_seeds`. The primary selection judge.
  J2  sample      train on all 7 degraded Sportec, test on Metrica Sample games 1-3 (same provider as
                  BAR-ATH). Transfer check: a candidate may not lose here.
  J3  public      train on all 7 degraded Sportec, predict BAR-ATH, score the public 5 minutes only.
                  17 possessions: REPORT ONLY, it never enters a gate (EVALUATION.md, README).

`role` is "select" (may gate a decision), "guard" (may veto) or "report" (printed, never gates).
A judge whose data is missing says so in `available()` and is skipped, never faked.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np

from harness import ROOT

if TYPE_CHECKING:
    from src.data.common import Match


@dataclass(frozen=True)
class Fold:
    name: str
    train: dict
    test: Match


def fold_metrics(gt, pred, half) -> dict:
    from src.eval import metrics
    r = metrics.score(gt, pred, half)
    return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
            "n_gt_spells": int(r["n_gt_spells"]), "n_pred_spells": int(r["n_pred_spells"]),
            "pred_poss_per_min": float(metrics.possessions_per_min(pred, half)),
            "gt_poss_per_min": float(metrics.possessions_per_min(gt, half))}


class Judge:
    name = ""
    role = "report"
    description = ""
    requires = ("src/data/common.py", "src/eval/metrics.py")    # research code the judge runs on

    def available(self) -> tuple:
        gone = [f for f in self.requires if not (ROOT / f).is_file()]
        return (False, "needs the research code ({}); this branch has only the harness".format(", ".join(gone))) \
            if gone else (True, "")

    def fingerprint(self) -> dict:
        """Hashes of the inputs that make two runs of this judge comparable (beyond its params)."""
        return {}

    def folds(self) -> Iterator[Fold]:
        raise NotImplementedError

    def score(self, fold: Fold, pred: np.ndarray) -> dict:
        return fold_metrics(fold.test.labels, pred, fold.test.half)


class SyntheticJudge(Judge):
    name, role = "J0", "select"
    description = "LOMO on synthetic matches (plumbing; CI)"

    def __init__(self, n_matches: int = 4, minutes: float = 3.0):
        self.n_matches, self.minutes = n_matches, minutes

    def folds(self):
        from harness.synthetic import make_match
        matches = {"syn{}".format(i): make_match(seed=i, minutes=self.minutes) for i in range(self.n_matches)}
        for mid, m in matches.items():
            yield Fold(mid, {k: v for k, v in matches.items() if k != mid}, m)


def _degrade_config(path):
    from src.data.degrade import DegradeConfig
    raw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads((ROOT / path).read_text()).items()
           if not k.startswith("_")}
    return DegradeConfig(**raw)


DEGRADE_PARAMS = ROOT / "local_data" / "degradation_params.json"
BAR_ATH_ATD = ROOT / "data" / "tactical" / "BAR-ATH_tactical_atd.txt"


def ensure_degradation_params() -> None:
    """Aggregate ATD statistics the degradation needs (not in git); built from the raw ATD by S1.3 and
    S1.4 exactly as scripts/make_submission_v1.ensure_degradation_params does."""
    if not DEGRADE_PARAMS.is_file():
        for script in ("s1_3_ball_gaps.py", "s1_4_slot_anatomy.py"):  # S1.4 merges into S1.3's file
            subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / script)], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else ""


class _SportecTrained(Judge):
    def __init__(self, degrade: str = "configs/degrade_v3.json", train_seeds=(1,), test_seed: int = 0):
        self.degrade, self.train_seeds, self.test_seed = degrade, tuple(train_seeds), test_seed

    requires = Judge.requires + ("src/data/sportec.py", "src/data/degrade.py")

    def _sportec_ok(self):
        ok, why = Judge.available(self)
        if not ok:
            return ok, why
        from src.data.sportec import DATA_DIR, MATCH_IDS
        missing = [m for m in MATCH_IDS if not list(DATA_DIR.glob("*{}*".format(m)))]
        if missing:
            return False, "Sportec matches missing in data_external/ ({}): bash scripts/download_public_datasets.sh sportec".format(
                ", ".join(missing))
        if not DEGRADE_PARAMS.is_file() and not BAR_ATH_ATD.is_file():
            return False, "degradation needs local_data/degradation_params.json or the BAR-ATH ATD to build it"
        return True, ""

    def fingerprint(self):
        return {"degrade_config": _sha(ROOT / self.degrade), "degradation_params": _sha(DEGRADE_PARAMS)}

    def _degraded(self):
        from src.data.degrade import degrade
        from src.data.sportec import load_all
        ensure_degradation_params()
        cfg = _degrade_config(self.degrade)
        raw = load_all()
        train = {"{}_s{}".format(mid, s): degrade(m, cfg, s) for mid, m in raw.items() for s in self.train_seeds}
        test = {mid: degrade(m, cfg, self.test_seed) for mid, m in raw.items()}
        return train, test


class SportecLOMO(_SportecTrained):
    name, role = "J1", "select"
    description = "LOMO on 7 degraded Sportec matches (primary selection judge)"

    def available(self):
        return self._sportec_ok()

    def folds(self):
        train, test = self._degraded()
        for mid, m in test.items():
            yield Fold(mid, {k: v for k, v in train.items() if not k.startswith(mid + "_")}, m)


class MetricaSample(_SportecTrained):
    name, role = "J2", "guard"
    description = "train on degraded Sportec, test on Metrica Sample 1-3 (transfer guard)"

    requires = _SportecTrained.requires + ("src/data/metrica_sample.py",)

    def available(self):
        ok, why = self._sportec_ok()
        if not ok:
            return ok, why
        from src.data.metrica_sample import DATA_DIR
        if ok and not DATA_DIR.is_dir():
            return False, "Metrica Sample missing in data_external/: bash scripts/download_public_datasets.sh"
        return ok, why

    def folds(self):
        from src.data.metrica_sample import GAMES, load
        train, _ = self._degraded()
        for g in GAMES:
            yield Fold("sample{}".format(g), train, load(g))


class PublicBarAth(_SportecTrained):
    name, role = "J3", "report"
    description = "train on degraded Sportec, score BAR-ATH on the public 5 min (report only)"
    requires = _SportecTrained.requires + ("src/data/metrica_atd.py",)

    def available(self):
        ok, why = self._sportec_ok()
        if ok and not (ROOT / "data" / "tactical" / "BAR-ATH_tactical_atd.txt").is_file():
            return False, "BAR-ATH ATD missing in data/tactical/ (docs/LOCAL_DATA_SETUP.md)"
        return ok, why

    def folds(self):
        from src.data.metrica_atd import load
        train, _ = self._degraded()
        yield Fold("BAR-ATH_public", train, load())

    def score(self, fold, pred):
        from src.eval import metrics
        r = metrics.score_public(np.asarray(pred, np.int8), bootstrap_n=0)
        return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
                "n_gt_spells": int(r["n_gt_spells"]), "n_pred_spells": int(r["n_pred_spells"]),
                "pred_poss_per_min": float(metrics.possessions_per_min(pred, fold.test.half))}


def build(name: str, params: dict | None = None) -> Judge:
    cls = {"J0": SyntheticJudge, "J1": SportecLOMO, "J2": MetricaSample, "J3": PublicBarAth}.get(name)
    if cls is None:
        raise KeyError("unknown judge {!r}; one of J0, J1, J2, J3".format(name))
    return cls(**(params or {}))


ALL = ("J0", "J1", "J2", "J3")

