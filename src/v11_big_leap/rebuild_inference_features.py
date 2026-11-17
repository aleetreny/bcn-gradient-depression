"""Build raw-match or public-sample PhysConformer inference feature caches."""
import hashlib
import json
from pathlib import Path

from src.features.v1 import match_fingerprint


def rebuild_inference_features(root: Path, output_path: Path) -> Path:
    from src.data.metrica_atd import load as load_atd
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.features.v1ctl_physics_v2 import build_v1ctl_physics_v2

    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    if output_path.parent != root / "_local":
        raise ValueError("manifest must be directly under _local/")
    names = {f"inference_features_sample_{gid}_raw.json": (f"sample_{gid}_raw", gid) for gid in GAMES}
    names["inference_features_atd.json"] = ("atd", None)
    if output_path.name not in names:
        raise ValueError("manifest must identify a raw match or public sample")
    if output_path.exists():
        raise FileExistsError(output_path)
    key, gid = names[output_path.name]
    match = load_atd(use_cache=False) if gid is None else load_sample(gid, use_cache=False)
    filename = f"{key}_{match_fingerprint(match)}.parquet"
    paths = [root / "local_data/cache" / part / filename for part in
             ("features_v1", "control_v1", "plus_v1", "physv2")]
    if any(path.exists() for path in paths):
        raise FileExistsError("a feature cache for this match already exists")
    features = build_v1ctl_physics_v2(match, key)
    if len(features) != len(match.half) or not all(path.is_file() for path in paths):
        raise ValueError("feature output is incomplete or has the wrong frame count")
    records = []
    for path in paths:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        records.append({"path": str(path.relative_to(root)), "sha256": digest})
    with output_path.open("x") as handle:
        json.dump({"key": key, "features": records}, handle, indent=2)
        handle.write("\n")
    return output_path
