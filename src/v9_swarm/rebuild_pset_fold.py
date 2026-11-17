"""Reconstruct a PSetV2 leave-one-match-out cache using public Sportec only.

Requires local Sportec raw data and 26 cached PlayerSetTensors NPZs per fold:
four training realizations for each of the six training matches and v3/v6 s0
for the held-out match, plus the calibrated degradation parameters. Runs nine epochs for each of two seeds on the available
PyTorch device. Local weights may differ from the historical CUDA weights.
"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from src.data.degrade import DegradeConfig, degrade
from src.data.sportec import MATCH_IDS, load
from src.features.v1 import match_fingerprint


TRAIN_REALIZATIONS = (
    ("v3_s1", "v3", 1, {}, 1.0),
    ("v3_s2", "v3", 2, {}, 1.0),
    ("v3_static05_s1", "v3", 1, {"static_hide_prob": 0.5}, 1.0),
    ("v6_s1", "v6", 11, {}, 0.8),
)
HEADS = ("proba", "change", "p4d", "p_aerial", "ball_z_hat")


def _load_degrade_configs(root):
    params = json.loads((root / "local_data" / "degradation_params.json").read_text())
    configs = {}
    for version in ("v3", "v6"):
        raw = json.loads((root / "configs" / f"degrade_{version}.json").read_text())
        settings = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in raw.items()
            if not key.startswith("_")
        }
        configs[version] = DegradeConfig(**settings)
        if not configs[version].params:
            configs[version] = replace(configs[version], params=params)
    return configs


def _load_tensors(cache_dir, match_id, tag, match, config, seed):
    from src.models.player_set_net import PlayerSetTensors, VERSION

    fingerprint = match_fingerprint(degrade(match, config, seed))
    path = cache_dir / f"pset_v{VERSION}_sportec_{match_id}_{tag}_{fingerprint}.npz"
    with np.load(path, allow_pickle=False) as archive:
        tensors = {
            key: archive[key].astype(dtype, copy=False)
            for key, dtype in (
                ("p_h", np.float32), ("p_a", np.float32),
                ("mask_h", bool), ("mask_a", bool),
                ("g_anti", np.float32), ("g_sym", np.float32),
            )
        }
    if any(len(values) != len(match.half) for values in tensors.values()):
        raise ValueError(f"tensor frame count differs from Sportec tracking: {path}")
    return PlayerSetTensors(**tensors)


def _new_model(seed):
    from src.v9_swarm.player_set_net_v2 import SymPlayerSetNetV2

    return SymPlayerSetNetV2(
        channels=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        gru_stride=10, dropout=0.15, lr=1.6e-3, weight_decay=1e-4,
        epochs=9, crop=1500, batch_size=32, poss_alpha=0.45,
        boundary_boost=1.6, change_loss_weight=0.35, b4d_loss_weight=0.45,
        aerial_loss_weight=0.20, swap_aug_prob=0.35, label_smoothing=0.02,
        seed=seed,
    )


def _fit_fold(root, held_out):
    cache_dir = root / "local_data" / "cache" / "pset"
    data_dir = root / "data_external" / "sportec_dfl"
    configs = _load_degrade_configs(root)
    test_match = load(held_out, data_dir=data_dir, use_cache=False)
    test_tensors = {
        tag: _load_tensors(cache_dir, held_out, f"{tag}_s0", test_match, configs[tag], 0)
        for tag in ("v3", "v6")
    }

    tensors, labels, halves, ball_zs, weights = [], [], [], [], []
    for mid in MATCH_IDS:
        if mid == held_out:
            continue
        match = load(mid, data_dir=data_dir, use_cache=False)
        for tag, version, seed, overrides, weight in TRAIN_REALIZATIONS:
            config = replace(configs[version], **overrides) if overrides else configs[version]
            tensors.append(_load_tensors(cache_dir, mid, tag, match, config, seed))
            labels.append(match.labels)
            halves.append(match.half)
            ball_zs.append(match.meta.get("ball_z"))
            weights.append(weight)

    index = MATCH_IDS.index(held_out)
    predictions = {tag: [] for tag in test_tensors}
    for seed in (42 + index, 142 + index):
        print(f"Training held-out fold {held_out}, seed {seed}", flush=True)
        model = _new_model(seed).fit(tensors, labels, halves, ball_zs=ball_zs, sample_weights=weights)
        for tag, test in test_tensors.items():
            predictions[tag].append(model.predict_all_heads(test, test_match.half))
        print(f"Finished model for held-out fold {held_out}, seed {seed}", flush=True)
        del model

    result = {}
    for tag, heads in predictions.items():
        for index, (name, pair) in enumerate(zip(HEADS, zip(*heads, strict=True), strict=True)):
            averaged = sum(pair) / len(pair)
            if index == 0:
                averaged = averaged / np.clip(averaged.sum(axis=1, keepdims=True), 1e-6, None)
            result[f"{tag}_{name}"] = averaged.astype(np.float32)
    return result


def rebuild_pset_fold(root: Path, output_path: Path) -> Path:
    """Train the public Sportec fold named by pset_v2_<match ID>.npz."""
    root = Path(root).resolve()
    requested = root / output_path
    output_path = requested.resolve()
    relative = output_path.relative_to(root)
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    matches = {f"pset_v2_{mid}.npz": mid for mid in MATCH_IDS}
    if output_path.name not in matches:
        raise ValueError("output must be named pset_v2_<Sportec match ID>.npz")
    if requested.is_symlink() or output_path.exists():
        raise FileExistsError(requested)

    arrays = _fit_fold(root, matches[output_path.name])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output:
        try:
            np.savez_compressed(output, **arrays)
            output.flush()
        except BaseException:
            output_path.unlink()
            raise
    return output_path
