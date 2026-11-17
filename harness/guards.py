"""Leak guard: nothing gated, private or heavy is tracked or staged. .gitignore is the first line of
defence; this is the second, because `git add -f` and a renamed file both walk straight past it.

    python -m harness guard            # tracked + staged files; exit 1 on any violation
    git config core.hooksPath .githooks  # run it on every commit
"""
from __future__ import annotations

import fnmatch
import subprocess

from harness import ROOT

MAX_BYTES = 5 * 1024 * 1024

# (pattern, why). Matched against the repo-relative path.
FORBIDDEN = (
    ("*_possession_gt.csv", "private full-match ground truth"),
    ("*_possession_gt_meta.json", "private full-match ground truth"),
    ("*_possession_gt_reference.csv", "private full-match ground truth"),
    ("*possession_by_frame.pkl", "private full-match ground truth"),
    ("*_atd.txt", "gated tracking"),
    ("*_atd.xml", "gated tracking"),
    ("*_smart_tagging.xml", "gated Smart Tagging"),
    ("*_homography.csv", "gated homography"),
    ("*.mp4", "video"),
    ("data/*", "gated data folder (only data/README.md is committed)"),
    ("data_external/*", "external datasets are downloaded by script"),
    ("local_data/*", "local caches and predictions"),
    ("submissions/*", "full-match label files"),
    ("ENTREGA_FINAL_HACKATHON/*.csv", "full-match label files"),
    ("experiments/outputs/*", "heavy experiment outputs"),
    ("*.npy", "array dump (predictions or features)"), ("*.npz", "array dump"),
    ("*.parquet", "feature cache"), ("*.pkl", "pickle"), ("*.joblib", "model"),
    ("*.pt", "model"), ("*.pth", "model"), ("*.ckpt", "model"), ("*.onnx", "model"),
    ("*.safetensors", "model"), ("*.pdf", "internal document"), (".env", "secrets"), (".env.*", "secrets"),
)
# Exceptions to FORBIDDEN (the size limit still applies). Text run logs in experiments/outputs/ are
# committed on purpose (waves 3-7); everything else there stays local.
ALLOWED = ("data/README.md", "evaluation/gt/*_possession_gt_public.csv", "evaluation/results/baseline_*.csv",
           "*.env.example", "experiments/outputs/*.log")


def _files(*args) -> list:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\0") if f]


def violations(paths=None) -> list:
    if paths is None:
        paths = sorted(set(_files("ls-files", "-z")) | set(_files("diff", "--cached", "--name-only", "-z",
                                                                   "--diff-filter=AM")))
    found = []
    for p in paths:
        why = None
        if not any(fnmatch.fnmatch(p, a) for a in ALLOWED):
            why = next((w for pat, w in FORBIDDEN if fnmatch.fnmatch(p, pat)), None)
            if why is None and p.startswith("evaluation/results/") and p.endswith(".csv"):
                why = "submission labels in evaluation/results/"
        if why is None:
            f = ROOT / p
            if f.is_file() and f.stat().st_size > MAX_BYTES:
                why = "larger than {} MB".format(MAX_BYTES // 2 ** 20)
        if why:
            found.append((p, why))
    return found
