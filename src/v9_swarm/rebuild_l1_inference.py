"""Rebuild local L1 inference from hashed models and feature blocks.

This is inference only: it does not upload data, train models or score the
provided game's public window. Outputs and the input manifest remain local.
"""
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.features.symmetry import swap_proba
from src.features.v1ctl_physics_v2 import mirror_v1ctl_physv2_df
from src.models.transition import change_features
from src.v10_first_principles.rebuild_trajectory_inference import TAGS, _verified_file
from src.v11_big_leap.rebuild_phys_inference import PARTS


def _twohead_tta(models, original, mirrored):
    if len(models) not in (2, 4):
        raise ValueError("expected two models and optional two calibrators")
    alive, team = models[:2]
    calibrators = models[2:] if len(models) == 4 else (None, None)

    def predict(x):
        pa = alive.predict_proba(x)[:, 1]
        ph = team.predict_proba(x)[:, 1]
        if calibrators[0] is not None:
            pa = calibrators[0].predict(pa)
            ph = calibrators[1].predict(ph)
        return np.column_stack((pa * ph, pa * (1.0 - ph), 1.0 - pa)).astype(np.float32)

    out = 0.5 * (predict(original) + swap_proba(predict(mirrored)))
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-6)
    return out.astype(np.float32)


def _binary_tta(model, original, mirrored):
    return (0.5 * (model.predict_proba(original)[:, 1]
                   + model.predict_proba(mirrored)[:, 1])).astype(np.float32)


def rebuild_l1_inference(root: Path, output_path: Path) -> Path:
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
    if output_path.suffix != ".pkl" or output_path.exists():
        raise ValueError("output must be a new .pkl cache")
    config = json.loads((root / "_local/l1_inference.json").read_text())
    if set(config) != {"models", "features"} or set(config["features"]) != set(TAGS):
        raise ValueError("configuration must name models and all six feature inputs")
    model_path = _verified_file(root, config["models"])
    features = {}
    for tag in TAGS:
        if set(config["features"][tag]) != set(PARTS):
            raise ValueError(f"missing feature blocks for {tag}")
        features[tag] = [_verified_file(root, config["features"][tag][part]) for part in PARTS]
        if len({p.stem for p in features[tag]}) != 1:
            raise ValueError(f"feature blocks have different fingerprints for {tag}")
    with model_path.open("rb") as handle:
        models = pickle.load(handle)
    if set(models) != {"gbm", "cb", "xgb", "aerial", "duel"}:
        raise ValueError("unexpected L1 model bundle")
    samples = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    result = {"sample": {}}
    for tag in TAGS:
        match = samples[int(tag[-1])] if tag.startswith("sample_game") else atd
        paths = features[tag]
        table = pd.concat([change_features(pd.read_parquet(paths[0]), match.half),
                           *(pd.read_parquet(p) for p in paths[1:])], axis=1)
        if len(table) != len(match.half):
            raise ValueError(f"feature length mismatch for {tag}")
        mirrored = mirror_v1ctl_physv2_df(table, match.half, match.pitch)
        x = table.to_numpy(dtype=np.float32, copy=False)
        xm = mirrored.to_numpy(dtype=np.float32, copy=False)
        packed = {name: _twohead_tta(models[name], x, xm) for name in ("gbm", "cb", "xgb")}
        packed.update(air=_binary_tta(models["aerial"], x, xm),
                      duel=_binary_tta(models["duel"], x, xm))
        if tag.startswith("sample_game"):
            result["sample"][int(tag[-1])] = packed
        else:
            result[tag] = packed
    with output_path.open("xb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
