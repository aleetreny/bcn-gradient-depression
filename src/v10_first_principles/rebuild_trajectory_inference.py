"""Generate local trajectory-model inference from explicitly verified inputs.

The models and tracking-derived tensors are private local artifacts. This
command does not train models, call Modal, or score the public match window.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.models.player_set_net import PlayerSetTensors
from src.v10_first_principles.traj_event_net_v10 import SymTrajectoryEventNetV10


TAGS = ("sample_game1", "sample_game2", "sample_game3", "atd_raw", "atd_rep", "atd_reteam_rep")
HEADS = ("proba", "change", "p4d", "p_duel_to", "p_aerial", "ball_z_hat")


def _verified_file(root, item):
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError("each input requires path and sha256")
    path = (root / item["path"]).resolve(strict=True)
    expected = item["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("invalid SHA-256")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"input SHA-256 mismatch: {path}")
    return path


def _pack_predictions(predictions, half):
    if len(predictions) != 2 or any(len(heads) != len(HEADS) for heads in predictions):
        raise ValueError("two models with six heads each are required")
    out = {name: ((predictions[0][i] + predictions[1][i]) / 2).astype(np.float32)
           for i, name in enumerate(HEADS)}
    p = out["proba"]
    out["proba"] = p / np.clip((p[:, 0:1] + p[:, 1:2]) + p[:, 2:3], 1e-6, None)
    raw_p, raw_c, raw_p4 = out["proba"], out["change"], out["p4d"]
    p, c, p4 = SymTrajectoryEventNetV10.calibrate_focal_soft_iou_outputs(raw_p, raw_c, raw_p4, half)
    return {
        "proba": p, "change": c, "p4d": p4,
        "proba_raw": raw_p, "change_raw": raw_c, "p4d_raw": raw_p4,
        "p_duel_to": out["p_duel_to"], "p_aerial": out["p_aerial"], "ball_z_hat": out["ball_z_hat"],
        "pset_proba": p, "pset_change": c, "pset_p4d": p4,
        "pset_p_duel_to": out["p_duel_to"], "pset_p_aerial": out["p_aerial"],
        "pset_ball_z_hat": out["ball_z_hat"],
    }


def rebuild_trajectory_inference(root: Path, output_path: Path) -> Path:
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
    config_path = root / "_local/trajectory_inference.json"
    config = json.loads(config_path.read_text())
    if set(config) != {"models", "tensors"} or len(config["models"]) != 2 or set(config["tensors"]) != set(TAGS):
        raise ValueError("configuration must name two models and all six tensor inputs")
    model_paths = [_verified_file(root, item) for item in config["models"]]
    tensor_paths = {tag: _verified_file(root, config["tensors"][tag]) for tag in TAGS}
    models = []
    for path in model_paths:
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, SymTrajectoryEventNetV10):
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
        packed = _pack_predictions([model.predict_all_heads(tensors, half) for model in models], half)
        if tag.startswith("sample_game"):
            result["sample"][int(tag[-1])] = packed
        else:
            result[tag] = packed
    with output_path.open("xb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
