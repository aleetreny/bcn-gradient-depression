"""python -m harness {list, new, run, compare, promote, show, check, guard}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness import SPECS

TEMPLATE = SPECS / "_template.toml"


def cmd_list(a):
    from harness import judges, registry
    for kind, items in registry.describe().items():
        print(kind)
        for name, doc in items.items():
            print("  {:<14} {}".format(name, doc))
    print("judges")
    for name in judges.ALL:
        j = judges.build(name)
        ok, why = j.available()
        print("  {:<3} {:<7} {:<62} {}".format(name, j.role, j.description, "ok" if ok else "unavailable: " + why))


def cmd_reconstruct(a):
    from harness import registry
    gone = registry.missing("reconstruction", a.name)
    if gone:
        sys.exit("reconstruction {!r} needs {}".format(a.name, ", ".join(gone)))
    artifact = registry.get("reconstruction", a.name)(a.root, a.output)
    print("wrote local artifact {}".format(artifact))


def cmd_new(a):
    from harness.spec import ID
    if not ID.match(a.id):
        sys.exit("id must be lowercase letters, digits and _")
    path = SPECS / "{}.toml".format(a.id)
    if path.exists():
        sys.exit("{} already exists".format(path))
    path.write_text(TEMPLATE.read_text().replace("h999_template", a.id))
    print("wrote {}: fill in the hypothesis and criteria, commit it, then run it".format(path.relative_to(SPECS.parents[1])))


def cmd_run(a):
    from harness import runner, spec
    sp = spec.load(a.spec)
    only = a.judges.split(",") if a.judges else None
    res = runner.run(sp, only=only, save_pred=a.save_pred, write=not a.no_write)
    print()
    runner.report(res)
    if a.register:
        print("registry row", runner.register_row(res))
    if a.strict and res.get("decision") not in (None, "ACCEPT", "NO_CHAMPION"):
        sys.exit(1)


def cmd_compare(a):
    from harness import gate, runner
    ra, rb = runner.load_result(a.candidate), runner.load_result(a.champion)
    for name, v in runner.verdicts(ra, rb, gate.Criteria(min_delta=a.min_delta)).items():
        print("{}: {}".format(name, gate.Verdict(**v).line() if "n_folds" in v else v["decision"]))


def cmd_promote(a):
    from harness import runner
    res = runner.load_result(a.run)
    if res["provenance"]["dirty"] and not a.force:
        sys.exit("run {} came from a dirty tree ({}); re-run it from a clean commit".format(
            a.run, ", ".join(res["provenance"]["dirty"][:5])))
    if res.get("decision") not in ("ACCEPT", "NO_CHAMPION", None) and not a.force:
        sys.exit("run {} did not pass the gate (decision {}); --force to override, and say why in the PR".format(
            a.run, res.get("decision")))
    runner.CHAMPION_FILE.parent.mkdir(parents=True, exist_ok=True)
    runner.CHAMPION_FILE.write_text(a.run + "\n")
    print("champion is now {}".format(a.run))


def cmd_guard(a):
    from harness import guards
    found = guards.violations()
    for p, why in found:
        print("LEAK  {:<60} {}".format(p, why))
    if found:
        sys.exit("{} file(s) must not be in git; git rm --cached them and check .gitignore".format(len(found)))
    print("guard: OK")


def cmd_check(a):
    from harness import check
    sys.exit(check.run(base=a.base, fast=a.fast, allow_protected=a.allow_protected))


def cmd_show(a):
    from harness import runner
    res = runner.load_result(a.run)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        runner.report(res)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m harness", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="registered components and judges").set_defaults(fn=cmd_list)
    s = sub.add_parser("reconstruct", help="run a registered local reconstruction")
    s.add_argument("name")
    s.add_argument("--root", type=Path, default=Path.cwd(), help="project root with the required inputs")
    s.add_argument("--output", type=Path, required=True, help="new local output file")
    s.set_defaults(fn=cmd_reconstruct)
    s = sub.add_parser("new", help="scaffold experiments/specs/<id>.toml")
    s.add_argument("id")
    s.set_defaults(fn=cmd_new)
    s = sub.add_parser("run", help="run a spec on its judges and gate it against the champion")
    s.add_argument("spec", type=Path)
    s.add_argument("--judges", help="comma-separated subset, e.g. J1,J3")
    s.add_argument("--save-pred", action="store_true", help="keep predictions in local_data/harness/")
    s.add_argument("--no-write", action="store_true", help="do not write experiments/runs/<id>/result.json")
    s.add_argument("--register", action="store_true", help="append a row to experiments/registry.csv")
    s.add_argument("--strict", action="store_true", help="exit 1 unless the gate says ACCEPT")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("compare", help="gate one stored run against another")
    s.add_argument("candidate")
    s.add_argument("champion")
    s.add_argument("--min-delta", type=float, default=0.01)
    s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("promote", help="make a run the champion (experiments/runs/CHAMPION)")
    s.add_argument("run")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_promote)
    s = sub.add_parser("show", help="print a stored run")
    s.add_argument("run")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_show)
    s = sub.add_parser("check", help="definition of done: leaks, protected files, records, lint, tests")
    s.add_argument("--base", default="origin/main", help="branch the change is compared against")
    s.add_argument("--fast", action="store_true", help="skip the test suite")
    s.add_argument("--allow-protected", action="store_true", help="HUMANS ONLY: approve protected-file changes")
    s.set_defaults(fn=cmd_check)
    sub.add_parser("guard", help="fail if gated/private/heavy files are tracked or staged").set_defaults(fn=cmd_guard)
    a = p.parse_args(argv)
    a.fn(a)
