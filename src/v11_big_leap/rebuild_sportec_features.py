"""Build one public Sportec feature realization required by PhysConformer."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from src.data.degrade import degrade
from src.data.sportec import MATCH_IDS, load
from src.features.v1 import match_fingerprint
from src.v9_swarm.rebuild_pset_fold import _load_degrade_configs
from src.v11_big_leap.rebuild_phys_fold import REALIZATIONS


def rebuild_sportec_features(root: Path, output_path: Path) -> Path:
    from src.features.v1ctl_physics_v2 import build_v1ctl_physics_v2

    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    if output_path.parent != root / "_local":
        raise ValueError("manifest must be directly under _local/")
    specs = {tag: (version, seed, overrides) for tag, version, seed, overrides, _ in REALIZATIONS}
    specs.update(v3_s0=("v3", 0, {}), v6_s0=("v6", 0, {}))
    names = {f"phys_features_{mid}_{tag}.json": (mid, tag) for mid in MATCH_IDS for tag in specs}
    if output_path.name not in names:
        raise ValueError("manifest must name a Sportec match and PhysConformer realization")
    if output_path.exists():
        raise FileExistsError(output_path)
    mid, tag = names[output_path.name]
    version, seed, overrides = specs[tag]
    config = _load_degrade_configs(root)[version]
    config = replace(config, **overrides) if overrides else config
    match = load(mid, data_dir=root / "data_external/sportec_dfl", use_cache=False)
    degraded = degrade(match, config, seed)
    key = f"sportec_{mid}_degrade_{tag}"
    filename = f"{key}_{match_fingerprint(degraded)}.parquet"
    paths = [root / "local_data/cache" / part / filename for part in
             ("features_v1", "control_v1", "plus_v1", "physv2")]
    if any(path.exists() for path in paths):
        raise FileExistsError("a feature cache for this realization already exists")
    frame = build_v1ctl_physics_v2(degraded, key)
    if len(frame) != len(match.half) or not all(path.is_file() for path in paths):
        raise ValueError("feature output is incomplete or has the wrong frame count")
    entries = []
    for path in paths:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        entries.append({"path": str(path.relative_to(root)), "sha256": digest})
    with output_path.open("x") as handle:
        json.dump({"match_id": mid, "tag": tag, "features": entries}, handle, indent=2)
        handle.write("\n")
    return output_path
