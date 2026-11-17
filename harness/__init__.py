"""The team harness: one way to build, judge, compare and record a possession pipeline.

    python -m harness list                                   # registered components and judges
    python -m harness run experiments/specs/h000_baseline_smooth50.toml
    python -m harness compare <run_a> <run_b>
    python -m harness guard                                  # data-leak check (CI + pre-commit)

Layers (docs/HARNESS.md):
  contracts  shape/dtype/semantics checks at every stage boundary
  registry   named features / models / decoders / predictors, built from TOML specs
  judges     J0 synthetic, J1 LOMO degraded Sportec, J2 Metrica Sample, J3 public BAR-ATH (report only)
  gate       paired, pre-registered decision against the champion: ACCEPT / NOISE / REJECT
  runner     spec -> judges -> experiments/runs/<id>/result.json (+ optional registry.csv row)
  guards     nothing gated or private ever reaches git
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
SPECS = ROOT / "experiments" / "specs"
