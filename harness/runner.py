"""Spec -> judges -> result.json -> verdict against the champion.

    from harness import runner, spec
    result = runner.run(spec.load("experiments/specs/h000_baseline_smooth50.toml"))

experiments/runs/<id>/result.json holds metrics only (per fold and summary), the spec, the git commit
and a digest of the source; it is small and committed. Predictions never go there: --save-pred writes
them to local_data/harness/<id>/ (gitignored). A run from a dirty tree is marked `dirty` and cannot
be promoted to champion.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time

import numpy as np

from harness import ROOT, RUNS
from harness import gate, judges as judges_mod, registry
from harness.contracts import check_labels

CHAMPION_FILE = RUNS / "CHAMPION"
PRED_DIR = ROOT / "local_data" / "harness"
WATCHED = ("src", "harness", "configs", "lab")


def _git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def source_digest(dirs=("src", "harness")) -> str:
    """Hash of every .py under `dirs` (same idea as src/eval/store.source_digest, branch-independent)."""
    import hashlib
    h = hashlib.blake2b(digest_size=12)
    for p in sorted(f for d in dirs if (ROOT / d).is_dir() for f in (ROOT / d).rglob("*.py")):
        h.update(str(p.relative_to(ROOT)).encode() + p.read_bytes())
    return h.hexdigest()


def provenance() -> dict:
    dirty = [l[3:] for l in _git("status", "--porcelain", "--", *WATCHED).splitlines()]
    return {"commit": _git("rev-parse", "--short", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty, "source_digest": source_digest(),
            "python": sys.version.split()[0], "platform": platform.platform()}


def champion_id(sp) -> str:
    if sp.champion:
        return sp.champion
    return CHAMPION_FILE.read_text().strip() if CHAMPION_FILE.is_file() else ""


def load_result(run_id: str, runs_dir=RUNS) -> dict:
    path = runs_dir / run_id / "result.json"
    if not path.is_file():
        raise FileNotFoundError("no run {!r} ({} missing); run its spec first".format(run_id, path))
    return json.loads(path.read_text())


def summarise(folds: dict) -> dict:
    s = np.array([f["spell_f1"] for f in folds.values()])
    return {"spell_f1_mean": float(s.mean()), "spell_f1_std": float(s.std(ddof=0)),
            "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds.values()])),
            "pred_poss_per_min": float(np.mean([f["pred_poss_per_min"] for f in folds.values()])),
            "n_folds": len(folds)}


def run_judge(sp, name: str, save_pred: bool = False, log=print) -> dict:
    judge = judges_mod.build(name, sp.judge_params.get(name))
    ok, why = judge.available()
    out = {"role": judge.role, "params": sp.judge_params.get(name, {}), "description": judge.description}
    if not ok:
        log("  [{}] skipped: {}".format(name, why))
        return dict(out, status="unavailable", reason=why)
    gone = registry.missing("predictor", sp.predictor)
    if gone:
        why = "predictor {!r} needs {} (not on this branch)".format(sp.predictor, ", ".join(gone))
        log("  [{}] skipped: {}".format(name, why))
        return dict(out, status="unavailable", reason=why)
    factory = registry.get("predictor", sp.predictor)
    folds = {}
    fingerprint = None
    for fold in judge.folds():
        fingerprint = judge.fingerprint() if fingerprint is None else fingerprint   # after folds() built its inputs
        t0 = time.time()
        fit_predict = factory(**sp.params)            # fresh per fold: no state leaks across folds
        pred = check_labels(fit_predict(fold.train, fold.test), fold.test.n_frames,
                            "predictor {} on {}/{}".format(sp.predictor, name, fold.name))
        m = judge.score(fold, pred)
        m["seconds"] = round(time.time() - t0, 2)
        folds[fold.name] = m
        log("  [{}] {:<16} spellF1 {:.4f}  macroF1 {:.4f}  spells {}/{}  {:.0f}s".format(
            name, fold.name, m["spell_f1"], m["macro_f1"], m["n_pred_spells"], m["n_gt_spells"], m["seconds"]))
        if save_pred:
            d = PRED_DIR / sp.id
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "{}_{}.npy".format(name, fold.name), pred)
    return dict(out, status="ok", fingerprint=fingerprint or {}, folds=folds, summary=summarise(folds))


def verdicts(result: dict, champion: dict, criteria: gate.Criteria) -> dict:
    out = {}
    for name, j in result["judges"].items():
        cj = champion["judges"].get(name, {})
        if j.get("status") != "ok" or cj.get("status") != "ok" or j["role"] == "report":
            continue
        if cj.get("params") != j.get("params") or cj.get("fingerprint") != j.get("fingerprint"):
            out[name] = {"decision": "INCOMPARABLE", "reason": "judge params or input hashes differ from the champion's"}
            continue
        v = gate.compare({f: m["spell_f1"] for f, m in j["folds"].items()},
                         {f: m["spell_f1"] for f, m in cj["folds"].items()}, criteria)
        out[name] = v.to_dict()
    return out


def run(sp, only=None, save_pred: bool = False, write: bool = True, runs_dir=RUNS, log=print) -> dict:
    t0 = time.time()
    names = [j for j in sp.judges if not only or j in only]
    log("{}  ({} on {})".format(sp.id, sp.predictor, ", ".join(names)))
    result = {"id": sp.id, "hypothesis": sp.hypothesis, "owner": sp.owner, "spec": sp.path, "spec_digest": sp.digest,
              "predictor": sp.predictor, "params": sp.params, "provenance": provenance(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "judges": {}}
    for name in names:
        result["judges"][name] = run_judge(sp, name, save_pred, log)
    champ = champion_id(sp)
    result["champion"] = champ
    if champ and champ not in ("none", sp.id):     # "none": a reference run, gated against nothing
        try:
            cres = load_result(champ, runs_dir)
            vs = verdicts(result, cres, sp.criteria)
            roles = {n: result["judges"][n]["role"] for n in vs}
            result["verdicts"] = vs
            result["decision"] = gate.combine(
                {n: gate.Verdict(**v) for n, v in vs.items() if "n_folds" in v}, roles)
        except FileNotFoundError as e:
            result["decision"], result["decision_reason"] = "NO_CHAMPION", str(e)
    result["seconds"] = round(time.time() - t0, 1)
    if write:
        d = runs_dir / sp.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")
    return result


def report(result: dict, log=print):
    for name, j in result["judges"].items():
        if j.get("status") != "ok":
            log("  {:<3} {:<7} {}".format(name, j["role"], j.get("reason", j.get("status"))))
            continue
        s = j["summary"]
        log("  {:<3} {:<7} spellF1 {:.4f} +- {:.4f}  macroF1 {:.4f}  poss/min {:.2f}  ({} folds)".format(
            name, j["role"], s["spell_f1_mean"], s["spell_f1_std"], s["macro_f1_mean"], s["pred_poss_per_min"],
            s["n_folds"]))
    for name, v in result.get("verdicts", {}).items():
        log("  {} vs {}: {}".format(name, result["champion"], gate.Verdict(**v).line() if "n_folds" in v
                                     else "{} ({})".format(v["decision"], v["reason"])))
    if "decision" in result:
        log("  decision: {}{}".format(result["decision"],
                                      "  (dirty tree: provisional)" if result["provenance"]["dirty"] else ""))


def register_row(result: dict) -> str:
    """Append the run to experiments/registry.csv (the team's historical log)."""
    if not (ROOT / "src" / "eval" / "registry.py").is_file():
        raise RuntimeError("--register needs src/eval/registry.py (research branches only)")
    from src.eval import registry as reg
    j1, j3 = result["judges"].get("J1", {}), result["judges"].get("J3", {})
    vals = {"config": result["spec"]}
    if j1.get("status") == "ok":
        vals.update(lomo_spellf1_media=j1["summary"]["spell_f1_mean"], lomo_std=j1["summary"]["spell_f1_std"],
                    posesiones_por_min=j1["summary"]["pred_poss_per_min"])
        v = result.get("verdicts", {}).get("J1", {})
        if "folds_improved" in v:
            vals["folds_mejorados"] = str(v["folds_improved"])
    if j3.get("status") == "ok":
        f = next(iter(j3["folds"].values()))
        vals.update(public_spellf1=f["spell_f1"], public_macrof1=f["macro_f1"])
    note = "harness {}: {} [{}]".format(result["id"], result["hypothesis"][:120], result.get("decision", "no gate"))
    return reg.append(result["id"], note, commit=result["provenance"]["commit"], **vals)
