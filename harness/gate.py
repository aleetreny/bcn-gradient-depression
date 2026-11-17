"""The decision gate: is a candidate better than the champion, or is it noise?

Paired by fold (same matches, same degradation seeds), so match difficulty cancels out:

    d_i = spellF1(candidate, fold i) - spellF1(champion, fold i)

  ACCEPT  mean(d) >= min_delta, at least min_folds_improved folds with d_i > 0, and the paired
          bootstrap 90 % interval of mean(d) above 0
  REJECT  mean(d) <= 0 and the interval below min_delta (it is not hiding a real gain)
  NOISE   anything else: the change may be real, but these folds cannot tell. Do not stack it.

Defaults are set from the project's own history: the J1 fold std is about 0.03, so with 7 folds
the standard error of a paired mean is about 0.01 (experiments/registry.csv, MODEL_V8_DESIGN).
Criteria are pre-registered in the spec *before* the run; changing them afterwards is visible in git.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ACCEPT, NOISE, REJECT = "ACCEPT", "NOISE", "REJECT"


@dataclass(frozen=True)
class Criteria:
    min_delta: float = 0.01
    min_folds_improved: float = 0.7      # fraction of folds (rounded up) when < 1, else a count
    confidence: float = 0.90
    n_boot: int = 4000
    seed: int = 0

    def folds_needed(self, n: int) -> int:
        k = self.min_folds_improved
        return int(np.ceil(k * n)) if k < 1 else int(k)


@dataclass(frozen=True)
class Verdict:
    decision: str
    n_folds: int
    mean_delta: float
    ci_low: float
    ci_high: float
    folds_improved: int
    folds_needed: int
    deltas: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        return "{:<6} delta {:+.4f}  90% CI [{:+.4f}, {:+.4f}]  folds improved {}/{} (need {})".format(
            self.decision, self.mean_delta, self.ci_low, self.ci_high, self.folds_improved, self.n_folds,
            self.folds_needed)


def compare(candidate: dict, champion: dict, criteria: Criteria | None = None) -> Verdict:
    """candidate / champion: {fold name: spellF1}. Only folds present in both are compared."""
    c = criteria or Criteria()
    folds = sorted(set(candidate) & set(champion))
    if not folds:
        raise ValueError("no common folds between candidate {} and champion {}".format(sorted(candidate), sorted(champion)))
    d = np.array([candidate[f] - champion[f] for f in folds], np.float64)
    rng = np.random.default_rng(c.seed)
    boot = d[rng.integers(0, len(d), (c.n_boot, len(d)))].mean(1)
    alpha = (1 - c.confidence) / 2
    lo, hi = (float(x) for x in np.quantile(boot, [alpha, 1 - alpha]))
    mean, improved, need = float(d.mean()), int((d > 0).sum()), c.folds_needed(len(d))
    if mean >= c.min_delta and improved >= need and lo > 0:
        decision = ACCEPT
    elif mean <= 0 and hi < c.min_delta:
        decision = REJECT
    else:
        decision = NOISE
    return Verdict(decision, len(d), mean, lo, hi, improved, need, dict(zip(folds, d.round(6).tolist())))


def combine(verdicts: dict, roles: dict) -> str:
    """Overall decision: every "select" judge must ACCEPT and no "guard" judge may REJECT."""
    select = [v.decision for j, v in verdicts.items() if roles.get(j) == "select"]
    guard = [v.decision for j, v in verdicts.items() if roles.get(j) == "guard"]
    if not select:
        return NOISE
    if all(s == ACCEPT for s in select) and REJECT not in guard:
        return ACCEPT
    if REJECT in select or REJECT in guard:
        return REJECT
    return NOISE
