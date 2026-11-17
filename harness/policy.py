"""What an AI assistant (or anyone) may not touch without a human, in one place. Stdlib only: the
agent hook imports this on every tool call.

  PROTECTED     changing these needs a human sign-off: the official scorer and public truth, the
                competition documents, everything the frozen v4 submission is built from (any edit can
                change its SHA-256), and the harness's own judges, gate and guards.
  RUNNER_ONLY   written only by `python -m harness run` (never by hand or by an agent's editor).
  NO_HAND_LABELS  label files: must come out of an algorithm (RULES.md), never be typed or edited.
"""
from __future__ import annotations

import fnmatch

PROTECTED = (
    # official scorer and public truth
    "evaluation/possession_metrics.py", "evaluation/run_evaluation.py", "evaluation/validate_submission.py",
    "evaluation/gt/*", "evaluation/results/baseline_*.csv",
    # competition documents
    "CHALLENGE.md", "DATA.md", "EVALUATION.md", "RULES.md", "FAQ.md", "SUBMISSION.md", "UPSTREAM_README.md", "LICENSE",
    # frozen v4 submission (scripts/reproduce_final.sh)
    "scripts/reproduce_final.sh", "scripts/make_submission.py", "scripts/make_submission_v1.py",
    "scripts/setup_research_env.sh", "scripts/download_public_datasets.sh",
    "scripts/s1_3_ball_gaps.py", "scripts/s1_4_slot_anatomy.py", "scripts/s2_1_degradation.py",
    "games.py", "starter/baseline.py",
    "src/data/common.py", "src/data/degrade.py", "src/data/metrica_atd.py", "src/data/sportec.py",
    "src/decode/viterbi_hsmm.py", "src/features/v1.py", "src/features/control.py",
    "src/models/gbm.py", "src/models/transition.py", "src/models/tcn.py",
    "configs/degrade_v3.json", "configs/decoder_v1ctl_tcnlocal_ens_m3.json", "configs/decoder_v1ctl_gbm.json",
    "requirements-lock.txt", "requirements-research-lock.txt", "requirements-research.txt",
    # the harness's own safety and judging
    "harness/policy.py", "harness/guards.py", "harness/gate.py", "harness/judges.py", "harness/check.py",
    "harness/agent_hook.py", ".githooks/*", ".github/workflows/*", ".claude/settings.json", "AGENTS.md",
)
RUNNER_ONLY = ("experiments/runs/*/result.json", "experiments/runs/CHAMPION")
NO_HAND_LABELS = ("submissions/*", "evaluation/results/*.csv", "*_possession_gt*")
GATED_DIRS = ("data/", "data_external/", "local_data/")


def matches(path: str, patterns) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def edit_block_reason(path: str) -> str:
    """Why an assistant may not edit `path` directly ('' if it may)."""
    if matches(path, NO_HAND_LABELS):
        return "label files must be produced by an algorithm, never written or edited by hand (RULES.md)"
    if matches(path, RUNNER_ONLY):
        return "only `python -m harness run` / `promote` write run records"
    if path.startswith(GATED_DIRS):
        return "gated and local data is read-only for assistants"
    if matches(path, PROTECTED):
        return "protected file (scorer, competition docs, frozen v4 inputs or harness safety): a human must make or approve this change"
    return ""
