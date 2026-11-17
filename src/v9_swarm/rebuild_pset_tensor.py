"""Rebuild one public Sportec PlayerSet tensor cache locally.

The request chooses a match and a recorded degradation realization. The
output contains tracking-derived tensors and must remain on this machine.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from src.data.degrade import degrade
from src.data.sportec import MATCH_IDS, load
from src.features.v1 import match_fingerprint
from src.v9_swarm.rebuild_pset_fold import TRAIN_REALIZATIONS, _load_degrade_configs


def rebuild_pset_tensor(root: Path, output_path: Path) -> Path:
    from src.models.player_set_net import VERSION, extract_player_set_tensors

    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] != ("_local",) or output_path.suffix != ".npz":
        raise ValueError("output must be a local .npz under _local/")
    if output_path.exists():
        raise FileExistsError(output_path)
    request = json.loads((root / "_local/pset_tensor_request.json").read_text())
    if set(request) != {"match_id", "tag"} or request["match_id"] not in MATCH_IDS:
        raise ValueError("request must name a public Sportec match and realization")
    specs = {tag: (version, seed, overrides)
             for tag, version, seed, overrides, _ in TRAIN_REALIZATIONS}
    specs.update(v3_s0=("v3", 0, {}), v6_s0=("v6", 0, {}))
    if request["tag"] not in specs:
        raise ValueError("unknown PlayerSet realization")
    version, seed, overrides = specs[request["tag"]]
    config = _load_degrade_configs(root)[version]
    if overrides:
        config = replace(config, **overrides)
    match = load(request["match_id"], data_dir=root / "data_external/sportec_dfl", use_cache=False)
    degraded = degrade(match, config, seed)
    expected = f"pset_v{VERSION}_sportec_{request['match_id']}_{request['tag']}_{match_fingerprint(degraded)}.npz"
    if output_path.name != expected:
        raise ValueError(f"output must use the degraded-input fingerprint: {expected}")
    tensor = extract_player_set_tensors(degraded)
    fields = ("p_h", "p_a", "mask_h", "mask_a", "g_anti", "g_sym")
    arrays = {key: np.asarray(getattr(tensor, key)) for key in fields}
    if any(len(value) != len(match.half) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("invalid tensor values or frame count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        np.savez(handle, **arrays)
    return output_path
