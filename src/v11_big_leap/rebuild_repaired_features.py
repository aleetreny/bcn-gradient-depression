"""Build local inference features from broadcast-repaired match tracking."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.features.v1 import match_fingerprint


def rebuild_repaired_features(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    if output_path.parent != root / "_local":
        raise ValueError("manifest must be directly under _local/")
    names = {"repaired_features_atd_rep.json": ("atd_rep_True_True_None", False),
             "repaired_features_atd_reteam_rep.json": ("atd_reteam_rep", True)}
    if output_path.name not in names:
        raise ValueError("manifest must identify a repaired match variant")
    if output_path.exists():
        raise FileExistsError(output_path)
    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
    from s4_6_broadcast_align import load_broadcast
    from src.data.metrica_atd import load as load_atd
    from src.features.v1ctl_physics_v2 import build_v1ctl_physics_v2

    key, use_reteam = names[output_path.name]
    atd = load_atd(use_cache=False)
    if use_reteam:
        with np.load(root / "local_data/video/players_reteam.npz", allow_pickle=False) as data:
            atd = replace(atd, P=data["P"].astype(atd.P.dtype),
                          slot_team=data["slot_team"].astype(atd.slot_team.dtype),
                          slot_is_gk=data["slot_is_gk"].astype(bool))
    players, ball, team = load_broadcast()
    valid = pd.read_csv(root / "data/broadcast/BAR-ATH_broadcast_homography.csv")["valid"].to_numpy().astype(bool)
    mapping, _ = enhanced_broadcast_alignment(atd, players, valid)
    repaired = repair_atd_match(atd, players, ball, team, mapping, True, True, None)
    filename = f"{key}_{match_fingerprint(repaired)}.parquet"
    paths = [root / "local_data/cache" / part / filename for part in
             ("features_v1", "control_v1", "plus_v1", "physv2")]
    if any(path.exists() for path in paths):
        raise FileExistsError("a feature cache for this match already exists")
    features = build_v1ctl_physics_v2(repaired, key)
    if len(features) != len(atd.half) or not all(path.is_file() for path in paths):
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
