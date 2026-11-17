"""Experiment specs: one TOML file per hypothesis, committed before the run (experiments/specs/).

    id = "h012_soft_hsmm"
    hypothesis = "A soft minimum duration recovers 0.7-1.8 s turnovers without new blips"
    owner = "ale"

    [predictor]
    name = "staged"
    params = { features = "v1", model = "gbm", decoder = "hsmm", decoder_params = { config = "configs/decoder_v1.json" } }

    [judges]
    use = ["J1", "J2", "J3"]

    [criteria]                 # pre-registered; see harness/gate.py
    champion = "h000_baseline_smooth50"
    min_delta = 0.01
    min_folds_improved = 0.7
"""
from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from harness.gate import Criteria

ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class Spec:
    id: str
    hypothesis: str
    predictor: str
    params: dict
    judges: tuple
    judge_params: dict
    champion: str
    criteria: Criteria
    owner: str = ""
    path: str = ""
    digest: str = ""
    extra: dict = field(default_factory=dict)


def load(path) -> Spec:
    path = Path(path)
    text = path.read_text()
    raw = tomllib.loads(text)
    for key in ("id", "hypothesis", "predictor"):
        if key not in raw:
            raise ValueError("{}: missing required key {!r}".format(path, key))
    if not ID.match(raw["id"]):
        raise ValueError("{}: id {!r} must be lowercase letters, digits and _".format(path, raw["id"]))
    if path.stem != raw["id"]:
        raise ValueError("{}: file name must equal the id ({}.toml)".format(path, raw["id"]))
    pred = raw["predictor"]
    judges = raw.get("judges", {})
    crit = dict(raw.get("criteria", {}))
    champion = crit.pop("champion", "")
    return Spec(id=raw["id"], hypothesis=raw["hypothesis"].strip(), predictor=pred["name"],
                params=pred.get("params", {}), judges=tuple(judges.get("use", ["J1", "J3"])),
                judge_params={k: v for k, v in judges.items() if k != "use"}, champion=champion,
                criteria=Criteria(**crit), owner=raw.get("owner", ""), path=str(path),
                digest=hashlib.blake2b(text.encode(), digest_size=8).hexdigest(),
                extra={k: v for k, v in raw.items() if k not in ("id", "hypothesis", "predictor", "judges",
                                                               "criteria", "owner")})
