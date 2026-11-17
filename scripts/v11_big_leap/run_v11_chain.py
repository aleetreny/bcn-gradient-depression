"""Run the registered v11 reconstruction chain end to end in one checkout.

Usage (from the checkout root, research environment):

    python scripts/v11_big_leap/run_v11_chain.py _local/v11_chain_inputs.json

The private configuration (keep it under ``_local/``; it names local,
match-derived files that must never be committed or uploaded) is a JSON object:

``inputs``
    Mapping from a canonical path under ``experiments/outputs/`` to
    ``{"path": ..., "sha256": ...}``. Each source is hash-checked and copied
    into place before the chain runs (model folds, inference caches, etc.).
``d3_manifest``
    ``{"path": ..., "sha256": ...}`` for the D3 input manifest.
``count_stream_root``
    Directory holding the count-variant streams listed in that manifest.
``output_dir``
    New directory under ``_local/`` for every chain output.
``variant`` (optional)
    Exported synthesis variant: ``grand`` (default; tracking, broadcast tracking and
    smart tags), ``strict`` or ``d3``.
``expected`` (optional)
    ``{"synthesis_sha256": ..., "csv_sha256": ...}`` to compare at the end.

Every product of the chain must be absent from ``experiments/outputs/`` before
the run, so nothing previously generated is reused. The script runs, in order:
v85_candidates, v85_streams, v9_diagnostic_streams, v9_duel_candidates,
v9_mbr_candidates, v9_enriched_candidates, d3_verified_streams (main and
count), v85_baseline, v9_baseline, v10_b1_candidates, v10_b2_candidates,
v11_enriched_candidates, v11_local_synthesis and v11_official_kickoff_export.
It never scores the public window (J3).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PRODUCTS = {
    "v85_candidates": "forensic_v7/v85_cands_cache.pkl",
    "v9_diagnostic_streams": "v9_swarm/_diag_streams_s3.pkl",
    "v9_duel_candidates": "v9_swarm/_diag_duel_cands.pkl",
    "v9_mbr_candidates": "v9_swarm/_diag_mbr_cands.pkl",
    "v9_enriched_candidates": "v9_swarm/_diag_enr_cands.pkl",
    "v85_baseline": "v9_swarm/baseline_preds_cache.pkl",
    "v9_baseline": "v10_first_principles/v9_preds_cache.pkl",
    "v10_b1_candidates": "v10_first_principles/b1_v10_cands.pkl",
    "v10_b2_candidates": "v10_first_principles/b2_v10_cands.pkl",
}
REQUIRED = {"inputs", "d3_manifest", "count_stream_root", "output_dir"}


def sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_config(root: Path, config_path: Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not REQUIRED <= set(config) or set(config) - REQUIRED - {"expected", "variant"}:
        raise ValueError(f"configuration must contain exactly {sorted(REQUIRED)} and optional 'expected', 'variant'")
    if config.get("variant", "grand") not in ("grand", "strict", "d3"):
        raise ValueError("variant must be grand, strict or d3")
    out = (root / config["output_dir"]).resolve()
    if out.relative_to(root).parts[:1] != ("_local",):
        raise ValueError("output_dir must be under _local/")
    if out.exists():
        raise FileExistsError(out)
    for target, source in config["inputs"].items():
        if not target.startswith("experiments/outputs/") or ".." in Path(target).parts:
            raise ValueError(f"input target must be under experiments/outputs/: {target}")
        if set(source) != {"path", "sha256"}:
            raise ValueError(f"input {target} needs path and sha256")
    outputs = root / "experiments/outputs"
    present = [rel for rel in PRODUCTS.values() if (outputs / rel).exists()]
    if present:
        raise FileExistsError(f"chain products already present (move them aside first): {present}")
    return config


def verified(entry: dict) -> Path:
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def main(config_path: str) -> int:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root, Path(config_path))
    out = (root / config["output_dir"]).resolve()
    out.mkdir(parents=True)
    log = out / "chain.log"
    record = {"inputs": {}, "steps": [], "variant": config.get("variant", "grand")}

    for target, entry in config["inputs"].items():
        source = verified(entry)
        destination = root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        shutil.copyfile(source, destination)
        record["inputs"][target] = entry["sha256"]

    manifest_path = verified(config["d3_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count_streams = out / "count_streams"
    for rel in manifest["count"]["streams"]:
        (count_streams / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(config["count_stream_root"]) / rel, count_streams / rel)
    for stage in ("d3-stage-main", "d3-stage-count"):
        (out / stage).mkdir()

    def write_private(name: str, payload: dict) -> None:
        path = root / "_local" / name
        if path.exists():
            path.unlink()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def entry(path: Path) -> dict:
        return {"path": str(path), "sha256": sha256(path)}

    def run(component: str, name: str, install: bool = False) -> Path:
        output = out / name
        start = time.time()
        with log.open("a") as handle:
            subprocess.run([sys.executable, "-m", "harness", "reconstruct", component, "--root", str(root),
                            "--output", str(output)], cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)
        if install:
            target = root / "experiments/outputs" / PRODUCTS[component]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output, target)
        record["steps"].append({"component": component, "sha256": sha256(output),
                                "seconds": round(time.time() - start, 1)})
        print(f"{component}: done in {time.time() - start:.1f}s", flush=True)
        return output

    rel_out = out.relative_to(root)
    run("v85_candidates", "v85-candidates.pkl", install=True)
    v85 = run("v85_streams", "v85-streams.pkl")
    diag = run("v9_diagnostic_streams", "diag.pkl", install=True)
    run("v9_duel_candidates", "duel.pkl", install=True)
    run("v9_mbr_candidates", "mbr.pkl", install=True)
    run("v9_enriched_candidates", "enriched.pkl", install=True)
    write_private("d3_reconstruction.json", {
        "variant": "main", "manifest_path": str(manifest_path), "feature_root": "local_data/cache",
        "stream_root": "experiments/outputs", "staging_root": str(rel_out / "d3-stage-main")})
    d3_main = run("d3_verified_streams", "d3-main.pkl")
    write_private("d3_reconstruction.json", {
        "variant": "count", "manifest_path": str(manifest_path), "feature_root": "local_data/d3_reteam_cache/count",
        "stream_root": str(rel_out / "count_streams"), "staging_root": str(rel_out / "d3-stage-count")})
    d3_count = run("d3_verified_streams", "d3-count.pkl")
    write_private("v85_baseline_inputs.json", {
        "v85_streams": entry(v85), "diagnostic_streams": entry(diag), "control_streams": entry(d3_main)})
    run("v85_baseline", "v85-baseline.pkl", install=True)
    v9 = run("v9_baseline", "v9-baseline.pkl", install=True)
    run("v10_b1_candidates", "b1.pkl", install=True)
    run("v10_b2_candidates", "b2.pkl", install=True)
    candidates = run("v11_enriched_candidates", "v11-candidates.pkl")
    write_private("v11_synthesis_inputs.json", {"v9": entry(v9), "candidates": entry(candidates),
                                                "d3_streams": entry(d3_count)})
    synthesis = run("v11_local_synthesis", "v11-synthesis.pkl")
    write_private("v11_export_inputs.json", {"synthesis": entry(synthesis), "variant": config.get("variant", "grand")})
    csv = run("v11_official_kickoff_export", "gradient_depression.csv")

    record["synthesis_sha256"] = sha256(synthesis)
    record["csv_sha256"] = sha256(csv)
    expected = config.get("expected", {})
    ok = all(record[key] == value for key, value in expected.items())
    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"synthesis {record['synthesis_sha256']}\ncsv {record['csv_sha256']}")
    if expected:
        print("MATCHES EXPECTED" if ok else "DIFFERS FROM EXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
