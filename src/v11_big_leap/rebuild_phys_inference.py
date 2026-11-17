"""Generate local PhysConformer inference from explicitly verified inputs.

The models and match-derived features remain on this machine. No Modal or
public-window scoring is performed by this reconstruction.
"""
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.features.v1ctl_physics_v2 import mirror_v1ctl_physv2_df
from src.models.transition import change_features
from src.v10_first_principles.rebuild_trajectory_inference import TAGS, HEADS, _verified_file
from src.v11_big_leap.phys_conformer_v11 import SymPhysConformerV11


PARTS = ("features_v1", "control_v1", "plus_v1", "physv2")


def _pack_predictions(predictions):
    if len(predictions) != 2 or any(len(heads) != len(HEADS) for heads in predictions):
        raise ValueError("two models with six heads each are required")
    out = {name: ((predictions[0][i] + predictions[1][i]) / 2).astype(np.float32)
           for i, name in enumerate(HEADS)}
    p = out["proba"]
    out["proba"] = p / np.clip((p[:, 0:1] + p[:, 1:2]) + p[:, 2:3], 1e-6, None)
    return out


def rebuild_phys_inference(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
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
    config = json.loads((root / "_local/phys_inference.json").read_text())
    if set(config) != {"models", "features"} or len(config["models"]) != 2 or set(config["features"]) != set(TAGS):
        raise ValueError("configuration must name two models and all six feature inputs")
    model_paths = [_verified_file(root, item) for item in config["models"]]
    feature_paths = {}
    for tag in TAGS:
        if set(config["features"][tag]) != set(PARTS):
            raise ValueError(f"missing feature blocks for {tag}")
        feature_paths[tag] = [_verified_file(root, config["features"][tag][part]) for part in PARTS]
        if len({path.stem for path in feature_paths[tag]}) != 1:
            raise ValueError(f"feature blocks have different fingerprints for {tag}")
    models = []
    for path in model_paths:
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, SymPhysConformerV11):
            raise ValueError(f"unexpected model type: {path}")
        models.append(model)
    samples = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    result = {"sample": {}}
    for tag in TAGS:
        match = samples[int(tag[-1])] if tag.startswith("sample_game") else atd
        paths = feature_paths[tag]
        table = pd.concat([change_features(pd.read_parquet(paths[0]), match.half),
                           *(pd.read_parquet(path) for path in paths[1:])], axis=1)
        if len(table) != len(match.half):
            raise ValueError(f"feature length mismatch for {tag}")
        mirrored = mirror_v1ctl_physv2_df(table, match.half, match.pitch)
        orig = mirror_v1ctl_physv2_df(mirrored, match.half, match.pitch)
        mirrored = mirror_v1ctl_physv2_df(orig, match.half, match.pitch)
        xo = orig.to_numpy(dtype=np.float32, copy=True)
        xm = mirrored.to_numpy(dtype=np.float32, copy=True)
        packed = _pack_predictions([model.predict_all_heads(xo, xm, match.half) for model in models])
        if tag.startswith("sample_game"):
            result["sample"][int(tag[-1])] = packed
        else:
            result[tag] = packed
    with output_path.open("xb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
