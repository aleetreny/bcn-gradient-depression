"""Infer local PSetV2 heads from explicitly hashed local models and tensors.

This reads private local artifacts, never calls Modal or scores the public
window, and writes a new local prediction cache only.
"""
import json
from pathlib import Path
import pickle

import numpy as np

from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.models.player_set_net import PlayerSetTensors
from src.v9_swarm.player_set_net_v2 import SymPlayerSetNetV2
from src.v10_first_principles.rebuild_trajectory_inference import TAGS, _verified_file


HEADS = ("pset_proba", "pset_change", "pset_p4d", "pset_p_aerial", "pset_ball_z_hat")


def _pack_predictions(predictions):
    if len(predictions) != 2 or any(len(heads) != len(HEADS) for heads in predictions):
        raise ValueError("two models with five heads each are required")
    out = {name: ((predictions[0][i] + predictions[1][i]) / 2).astype(np.float32)
           for i, name in enumerate(HEADS)}
    p = out["pset_proba"]
    out["pset_proba"] = p / np.clip(p.sum(axis=1, keepdims=True), 1e-6, None)
    return out


def rebuild_pset_inference(root: Path, output_path: Path) -> Path:
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
    config = json.loads((root / "_local/pset_inference.json").read_text())
    if set(config) != {"models", "tensors"} or len(config["models"]) != 2 or set(config["tensors"]) != set(TAGS):
        raise ValueError("configuration must name two models and all six tensor inputs")
    model_paths = [_verified_file(root, item) for item in config["models"]]
    tensor_paths = {tag: _verified_file(root, config["tensors"][tag]) for tag in TAGS}
    models = []
    for path in model_paths:
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, SymPlayerSetNetV2):
            raise ValueError(f"unexpected model type: {path}")
        models.append(model)
    samples = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    result = {"sample": {}}
    for tag in TAGS:
        half = samples[int(tag[-1])].half if tag.startswith("sample_game") else atd.half
        with np.load(tensor_paths[tag], allow_pickle=False) as data:
            tensors = PlayerSetTensors(**{key: data[key] for key in ("p_h", "p_a", "mask_h", "mask_a", "g_anti", "g_sym")})
        if any(len(getattr(tensors, key)) != len(half) for key in ("p_h", "p_a", "mask_h", "mask_a", "g_anti", "g_sym")):
            raise ValueError(f"tensor length mismatch for {tag}")
        packed = _pack_predictions([model.predict_all_heads(tensors, half) for model in models])
        if tag.startswith("sample_game"):
            result["sample"][int(tag[-1])] = packed
        else:
            result[tag] = packed
    with output_path.open("xb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
