"""Reproducible version-zero submission: the unmodified official nearest-player baseline.

This is a baseline/fallback only, NOT a claimed improvement. It needs the licensed
tactical ATD metadata (.xml) and tracking (.txt) stored outside Git.

Usage:
    python src/predict.py --window 50 --output submissions/draft_v0.csv

For proposed improvements see research/CHALLENGE_ANALYSIS.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter.baseline import build, smooth, write_submission  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, default=50,
                        help="Half-width of official majority vote, in frames (default: 50).")
    parser.add_argument("--output", type=Path, default=Path("submissions/draft_v0.csv"),
                        help="Final CSV path; default submissions/draft_v0.csv")
    args = parser.parse_args()
    if args.window < 0:
        parser.error("--window must be nonnegative")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    codes = build()  # uses ./data or MTAH_DATA_DIR; will error clearly if assets are missing
    labels = smooth(codes, args.window)
    write_submission(str(output), labels)
    print("Wrote {} using OFFICIAL baseline, window +/-{} frames".format(
        output, args.window))

    # Validate the exact file, not a different in-memory intermediate.
    subprocess.run([sys.executable, str(ROOT / "evaluation/validate_submission.py"),
                    str(output)], check=True, cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
