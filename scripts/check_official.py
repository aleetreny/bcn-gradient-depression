"""Smoke-test the imported official evaluator using PUBLIC competition files only.

The restricted match data are deliberately not needed and must not be used in CI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_GT = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
BASELINES = tuple((ROOT / "evaluation/results").glob("baseline_*.csv"))


def run(*arguments: str) -> None:
    cmd = [sys.executable, *arguments]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    if not PUBLIC_GT.is_file():
        raise SystemExit("Missing official public ground truth: " + str(PUBLIC_GT))
    if len(BASELINES) != 4:
        raise SystemExit("Expected four official baseline CSVs; found {}".format(len(BASELINES)))
    for file in sorted(BASELINES):
        run("evaluation/validate_submission.py", str(file))
    run("evaluation/run_evaluation.py")
    print("\nPublic reference checks passed. This is NOT a full-match benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
