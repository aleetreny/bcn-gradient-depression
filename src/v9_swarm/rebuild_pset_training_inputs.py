"""Build all PlayerSet training and held-out tensors for one public match.

Tensors are local intermediates; no model, evaluation score, or submission is
written. The manifest is a completion record, not proof of model reproducibility.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src.data.degrade import degrade
from src.data.sportec import MATCH_IDS, load
from src.features.v1 import match_fingerprint
from src.v9_swarm.rebuild_pset_fold import TRAIN_REALIZATIONS, _load_degrade_configs


def rebuild_pset_training_inputs(root: Path, output_path: Path) -> Path:
    from src.models.player_set_net import VERSION, extract_player_set_tensors

    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    if output_path.parent != root / "_local":
        raise ValueError("manifest must be directly under _local/")
    names = {f"pset_inputs_{mid}.json": mid for mid in MATCH_IDS}
    if output_path.name not in names:
        raise ValueError("manifest must be named pset_inputs_<Sportec match ID>.json")
    if output_path.exists():
        raise FileExistsError(output_path)
    mid = names[output_path.name]
    match = load(mid, data_dir=root / "data_external/sportec_dfl", use_cache=False)
    configs = _load_degrade_configs(root)
    specs = [(tag, version, seed, overrides) for tag, version, seed, overrides, _ in TRAIN_REALIZATIONS]
    specs.extend((tag + "_s0", tag, 0, {}) for tag in ("v3", "v6"))
    cache = root / "local_data/cache/pset"
    cache.mkdir(parents=True, exist_ok=True)
    entries = []
    for tag, version, seed, overrides in specs:
        config = replace(configs[version], **overrides) if overrides else configs[version]
        degraded = degrade(match, config, seed)
        name = f"pset_v{VERSION}_sportec_{mid}_{tag}_{match_fingerprint(degraded)}.npz"
        destination = cache / name
        # Do not trust or replace existing caches, even if the filename matches.
        if destination.exists():
            raise FileExistsError(destination)
        tensor = extract_player_set_tensors(degraded)
        fields = ("p_h", "p_a", "mask_h", "mask_a", "g_anti", "g_sym")
        arrays = {key: np.asarray(getattr(tensor, key)) for key in fields}
        if any(len(value) != len(match.half) or not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"invalid tensor values or frame count for {tag}")
        with NamedTemporaryFile(dir=cache, suffix=".npz") as tmp:
            np.savez(tmp, **arrays)
            tmp.flush()
            os.link(tmp.name, destination)
        with destination.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        entries.append({"tag": tag, "path": str(destination.relative_to(root)), "sha256": sha256})
        print(f"Built {mid} {tag}", flush=True)
    with output_path.open("x") as handle:
        json.dump({"match_id": mid, "tensors": entries}, handle, indent=2)
        handle.write("\n")
    return output_path
