"""Rebuild local v10 B1/B2 candidates using Sportec and Metrica inputs.

This extracts candidates only; it does not run the historical public-window
scoring ladder. The input caches and outputs are local-only.
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import load_all as load_sportec
from src.models.transition import change_features
from src.v9_swarm.rebuild_candidates import _feature_cache_paths
from src.v10_first_principles.touch_segment_lattice import (
    extract_b2_v10_boundary_candidates,
    extract_touch_segment_proposals_v10,
)


def _features(root, key, half):
    paths = _feature_cache_paths(root / "local_data/cache", key)
    return pd.concat([change_features(pd.read_parquet(paths[0]), half),
                      *(pd.read_parquet(p) for p in paths[1:])], axis=1)


def _fold_predictions(path, prefix, suffix=False):
    with np.load(path, allow_pickle=False) as archive:
        if suffix:
            return {k[:-3]: archive[k] for k in archive.files if k.endswith("_" + prefix)}
        return {k[3:]: archive[k] for k in archive.files if k.startswith(prefix + "_")}


def rebuild_b1_b2(root: Path, output_path: Path, kind: str) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this reconstruction module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if output_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.exists():
        raise FileExistsError(output_path)
    if kind not in ("b1", "b2"):
        raise ValueError("kind must be b1 or b2")

    outputs = root / "experiments/outputs"
    swarm = outputs / "v9_swarm"
    v10 = outputs / "v10_first_principles"
    with (swarm / "_diag_streams_s3.pkl").open("rb") as handle:
        streams = pickle.load(handle)["streams"]
    with (v10 / "v9_preds_cache.pkl").open("rb") as handle:
        baseline = pickle.load(handle)["grand"]
    sportec = load_sportec()
    samples = {gid: load_sample(gid) for gid in GAMES}
    result = {key: {} for key in ("XYX", "XYD", "DYX")} if kind == "b1" else {}

    def extract(track, key, match, pred, ps, l1, traj, features):
        p, c, p4d, fams = streams[track][key]
        if kind == "b1":
            proposals = extract_touch_segment_proposals_v10(
                pred, p, c, p4d, fams, match.half, features, ps, l1,
                gt=match.labels, t10=traj,
            )
            for name in result:
                result[name].setdefault(track, {})[key] = proposals[name]
        else:
            result.setdefault(track, {})[key] = extract_b2_v10_boundary_candidates(
                pred, match.half, p, c, p4d, fams, ps, l1, features, labels=match.labels,
            )

    for mid in sorted(sportec):
        match = sportec[mid]
        for track, version, baseline_key in (("v3", "v3", "preds_v3"),
                                              ("v6_std", "v6", "preds_v6_std"),
                                              ("v6_gen", "v6", "preds_v6_gen")):
            ps = _fold_predictions(swarm / "pset_v2_oof" / f"pset_v2_{mid}.npz", version)
            l1 = _fold_predictions(swarm / "l1_trees_oof" / f"l1_trees_{mid}.npz", version, suffix=True)
            traj = _fold_predictions(v10 / "traj_v10_oof" / f"traj_v10_{mid}.npz", version)
            X = _features(root, f"sportec_{mid}_degrade_{version}_s0", match.half)
            extract(track, mid, match, baseline[baseline_key][mid], ps, l1, traj, X)

    with (swarm / "pset_v2_oof/pset_v2_j23_preds.pkl").open("rb") as handle:
        sample_ps = pickle.load(handle)["sample"]
    with (swarm / "l1_trees_oof/l1_trees_j23_preds.pkl").open("rb") as handle:
        sample_l1 = pickle.load(handle)["sample"]
    with (v10 / "traj_v10_oof/traj_v10_j23_preds.pkl").open("rb") as handle:
        sample_traj = pickle.load(handle)["sample"]
    for gid in GAMES:
        match = samples[gid]
        ps = {key[5:]: value for key, value in sample_ps[gid].items()}
        X = _features(root, f"sample_{gid}_raw", match.half)
        extract("sample", gid, match, baseline["preds_sample"][gid], ps,
                sample_l1[gid], sample_traj[gid], X)
    with output_path.open("xb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
