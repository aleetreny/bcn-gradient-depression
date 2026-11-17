"""Identity manifests for stores of predictions and trained models (review 004, finding 3).

    from src.eval.store import identity, check_store
    ident = identity(config=asdict(cfg), features="v1ctl", matches=sportec, extra={...})
    check_store(store_dir, ident)        # writes manifest.json, or raises if the store was made from other inputs

A store is only reused when its manifest matches the current inputs: degradation config contents,
feature set, the raw data and labels of the matches, the model settings, and the source code of
src/data, src/features and src/models (any edit there invalidates stored predictions). A store that
already has files but no manifest is adopted with a warning (made before manifests existed).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("src/data", "src/features", "src/models")


def _digest(*chunks: bytes) -> str:
    h = hashlib.blake2b(digest_size=12)
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def source_digest(dirs=SOURCE_DIRS) -> str:
    files = sorted(p for d in dirs for p in (ROOT / d).rglob("*.py"))
    return _digest(*(str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in files))


def data_digest(matches: dict) -> dict:
    """Per match: hash of positions, ball, slots, halves, direction and labels."""
    from src.features.v1 import match_fingerprint
    return {mid: _digest(match_fingerprint(m).encode(), np.ascontiguousarray(m.labels).tobytes()) for mid, m in matches.items()}


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else repr(o)))


def identity(config: dict, features: str, matches: dict, extra: dict | None = None) -> dict:
    return _jsonable({"config": config, "features": features, "data": data_digest(matches),
                      "source": source_digest(), "extra": extra or {}})


def check_store(store: Path, ident: dict, ignore=()) -> None:
    """`ignore`: keys not compared (e.g. "source" when only re-decoding stored predictions)."""
    store = Path(store)
    path = store / "manifest.json"
    if path.is_file():
        old = json.loads(path.read_text())
        diff = sorted(k for k in set(old) | set(ident) if k not in ignore and old.get(k) != ident.get(k))
        if diff:
            raise RuntimeError("store {} was built from other inputs (differs in: {}); delete it or use another --out".format(
                store, ", ".join(diff)))
        return
    if store.is_dir() and any(store.iterdir()):
        warnings.warn("adopting store {} made before manifests existed".format(store))
    store.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ident, indent=1, sort_keys=True))
