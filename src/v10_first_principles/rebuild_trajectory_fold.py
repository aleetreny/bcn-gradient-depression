"""Retrain one public Sportec trajectory fold locally and generate its OOF cache.

This uses retained PlayerSet tensor caches and public Sportec labels. New
weights and outputs need not match the historical CUDA run.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.data.sportec import MATCH_IDS, load
from src.v9_swarm.rebuild_pset_fold import _load_degrade_configs, _load_tensors
from src.v10_first_principles.rebuild_trajectory_inference import _pack_predictions
from src.v10_first_principles.traj_event_net_v10 import SymTrajectoryEventNetV10


TRAIN_REALIZATIONS = (
    ("v3_s1", "v3", 1, {}, 1.0),
    ("v3_s2", "v3", 2, {}, 1.0),
    ("v3_static05_s1", "v3", 1, {"static_hide_prob": 0.5}, 0.95),
    ("v6_s1", "v6", 11, {}, 0.85),
)
OUTPUT_HEADS = ("proba", "change", "p4d", "proba_raw", "change_raw", "p4d_raw",
                "p_duel_to", "p_aerial", "ball_z_hat")


def _new_model(seed):
    return SymTrajectoryEventNetV10(
        channels=88, dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        gru_stride=8, window_size=1501, dropout=0.15, lr=1.6e-3,
        weight_decay=1e-4, epochs=8, crop=1500, batch_size=32,
        poss_alpha=0.40, boundary_boost=1.65, soft_iou_weight=0.28,
        change_loss_weight=0.35, b4d_loss_weight=0.45, duel_loss_weight=0.25,
        aerial_loss_weight=0.20, far_swap_prob=0.14, slot_drop_prob=0.14,
        ball_drop_prob=0.18, label_smoothing=0.02, sliding_window_blend=0.50,
        seed=seed,
    )


def _fit_fold(root, held_out):
    cache = root / "local_data/cache/pset"
    data = root / "data_external/sportec_dfl"
    configs = _load_degrade_configs(root)
    test_match = load(held_out, data_dir=data, use_cache=False)
    test_tensors = {tag: _load_tensors(cache, held_out, f"{tag}_s0", test_match, configs[tag], 0)
                    for tag in ("v3", "v6")}
    tensors, labels, halves, ball_zs, weights = [], [], [], [], []
    for mid in MATCH_IDS:
        if mid == held_out:
            continue
        match = load(mid, data_dir=data, use_cache=False)
        for tag, version, seed, overrides, weight in TRAIN_REALIZATIONS:
            config = replace(configs[version], **overrides) if overrides else configs[version]
            tensors.append(_load_tensors(cache, mid, tag, match, config, seed))
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
        packed = _pack_predictions(heads, test_match.half)
        result.update({f"{tag}_{key}": packed[key] for key in OUTPUT_HEADS})
    return result


def rebuild_trajectory_fold(root: Path, output_path: Path) -> Path:
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
    names = {f"traj_v10_{mid}.npz": mid for mid in MATCH_IDS}
    if output_path.name not in names:
        raise ValueError("output must be named traj_v10_<Sportec match ID>.npz")
    if output_path.exists():
        raise FileExistsError(output_path)
    arrays = _fit_fold(root, names[output_path.name])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return output_path
