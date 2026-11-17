"""Append one row per run to experiments/registry.csv (columns fixed by the plan)."""
from __future__ import annotations

import csv
import datetime as dt
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments" / "registry.csv"
COLUMNS = ("id", "fecha", "commit", "paso", "config", "lomo_spellf1_media", "lomo_std", "folds_mejorados",
           "public_spellf1", "public_macrof1", "posesiones_por_min", "notas")


def head_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def next_id() -> str:
    with REGISTRY.open() as f:
        ids = [int(r["id"]) for r in csv.DictReader(f) if r["id"].isdigit()]
    return "{:03d}".format(max(ids, default=0) + 1)


def append(paso: str, notas: str, commit: str | None = None, run_id: str | None = None, **values) -> str:
    """Write a row; unknown keys raise, missing metric columns stay empty. Returns the id."""
    unknown = set(values) - set(COLUMNS)
    if unknown:
        raise KeyError("not registry columns: {}".format(sorted(unknown)))
    row = {c: "" for c in COLUMNS}
    row.update({k: ("{:.6f}".format(v) if isinstance(v, float) else v) for k, v in values.items()})
    row.update({"id": run_id or next_id(), "fecha": dt.date.today().isoformat(),
                "commit": commit or head_commit(), "paso": paso, "notas": notas})
    with REGISTRY.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
    return row["id"]
