"""`python -m harness check`: the definition of done for every change, run by the assistant before it
says it has finished, by the pre-push hook, and by CI.

  FAIL    must be fixed before the change can land:
          leaks · rewritten registry rows · a run record without a spec committed before it ·
          a champion from a dirty tree · lint · tests
  REVIEW  allowed, but only with a human's sign-off: protected files changed (harness/policy.py).
          An assistant stops here and asks; it never passes --allow-protected itself.

Exit code 0 = all clear, 1 = FAIL, 3 = REVIEW only.
"""
from __future__ import annotations

import json
import subprocess
import sys

from harness import ROOT, RUNS
from harness.policy import PROTECTED, matches

DEFAULT_BASE = "origin/main"
OK, FAIL, REVIEW, SKIP = "ok", "FAIL", "REVIEW", "skip"


def _git(*args, check=False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


def changed_files(base: str) -> list | None:
    """Files changed on this branch since `base` (committed, staged, unstaged and untracked)."""
    mb = _git("merge-base", "HEAD", base)
    if mb.returncode:
        return None
    out = set(_git("diff", "--name-only", mb.stdout.strip()).stdout.split())
    out |= set(_git("ls-files", "--others", "--exclude-standard").stdout.split())
    return sorted(out)


def check_leaks():
    from harness.guards import violations
    found = violations()
    return (FAIL, ["{}  ({})".format(p, w) for p, w in found]) if found else (OK, [])


def check_protected(changed):
    if changed is None:
        return SKIP, ["base branch not found; fetch it or pass --base"]
    hit = [p for p in changed if matches(p, PROTECTED)]
    return (REVIEW, hit) if hit else (OK, [])


def check_registry(base: str):
    """experiments/registry.csv is append-only: the base version must be a prefix of the current one."""
    mb = _git("merge-base", "HEAD", base)
    if mb.returncode:
        return SKIP, ["base branch not found"]
    old = _git("show", "{}:experiments/registry.csv".format(mb.stdout.strip()))
    if old.returncode:
        return OK, []
    new = (ROOT / "experiments" / "registry.csv").read_text()
    if not new.startswith(old.stdout.rstrip("\n")):
        return FAIL, ["existing rows were edited or removed; the registry is append-only"]
    return OK, []


def check_runs(changed):
    """Every changed run record: its spec is committed, unchanged since, and committed no later than the run."""
    from harness.spec import load
    problems = []
    paths = [p for p in (changed or []) if matches(p, ("experiments/runs/*/result.json",))]
    for p in paths:
        f = ROOT / p
        if not f.is_file():
            continue
        res = json.loads(f.read_text())
        spec_path = res.get("spec", "")
        rel = spec_path.split(str(ROOT) + "/")[-1]
        spec_commit = _git("log", "-1", "--format=%h", "--", rel).stdout.strip()
        if not spec_commit:
            problems.append("{}: spec {} was never committed (pre-register it before running)".format(p, rel))
            continue
        if _git("diff", "--quiet", "HEAD", "--", rel).returncode:
            problems.append("{}: spec {} changed after it was committed".format(p, rel))
            continue
        if load(ROOT / rel).digest != res.get("spec_digest"):
            problems.append("{}: result was produced from a different version of {}".format(p, rel))
            continue
        run_commit = res.get("provenance", {}).get("commit", "")
        if run_commit and _git("merge-base", "--is-ancestor", spec_commit, run_commit).returncode:
            problems.append("{}: run at {} does not contain the spec commit {}".format(p, run_commit, spec_commit))
    champ = RUNS / "CHAMPION"
    if champ.is_file():
        cid = champ.read_text().strip()
        cres = RUNS / cid / "result.json"
        if not cres.is_file():
            problems.append("CHAMPION names {} but it has no result.json".format(cid))
        elif json.loads(cres.read_text()).get("provenance", {}).get("dirty"):
            problems.append("champion {} was run from a dirty tree".format(cid))
    return (FAIL, problems) if problems else (OK, ["{} run record(s) checked".format(len(paths))] if paths else [])


def lint_paths() -> list:
    return [d for d in ("src", "tests", "harness") if (ROOT / d).is_dir()]


def check_cmd(name, cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        return OK, []
    if "No module named" in r.stderr:
        return SKIP, ["{} not installed".format(name)]
    tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
    return FAIL, tail


def run(base: str = DEFAULT_BASE, fast: bool = False, allow_protected: bool = False, log=print) -> int:
    changed = changed_files(base)
    checks = [("leak guard", check_leaks), ("protected files", lambda: check_protected(changed)),
              ("registry append-only", lambda: check_registry(base)), ("run records", lambda: check_runs(changed)),
              ("lint", lambda: check_cmd("pyflakes", [sys.executable, "-m", "pyflakes", *lint_paths()]))]
    if not fast:
        checks.append(("tests", lambda: check_cmd("pytest", [sys.executable, "-m", "pytest", "-q", "-x", "tests"])))
    worst = OK
    log("harness check (base {}, {} changed files)".format(base, "?" if changed is None else len(changed)))
    for name, fn in checks:
        status, lines = fn()
        if status == REVIEW and allow_protected:
            status = OK
            lines = ["approved by a human (--allow-protected): " + ", ".join(lines)]
        log("  [{:<6}] {}".format(status, name))
        for line in lines:
            log("           " + line)
        if status == FAIL or (status == REVIEW and worst != FAIL):
            worst = status
    if worst == FAIL:
        log("RESULT: FAIL. Fix the items above before finishing.")
        return 1
    if worst == REVIEW:
        log("RESULT: REVIEW. Protected files changed. Stop and ask a human to approve; do not revert or work around it.")
        return 3
    log("RESULT: OK")
    return 0
