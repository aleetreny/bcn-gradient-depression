"""Retrain one PhysConformer Sportec fold locally from verified feature caches.

This preserves the recorded fold recipe but produces new weights and outputs,
not the original CUDA model. It never scores the public match window.
"""
from dataclasses import replace
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.degrade import degrade
from src.data.sportec import MATCH_IDS, load
from src.features.v1 import match_fingerprint
from src.features.v1ctl_physics_v2 import mirror_v1ctl_physv2_df
from src.models.transition import change_features
from src.v9_swarm.rebuild_candidates import _feature_cache_paths
from src.v9_swarm.rebuild_pset_fold import _load_degrade_configs
from src.v11_big_leap.phys_conformer_v11 import SymPhysConformerV11
from src.v11_big_leap.rebuild_phys_inference import _pack_predictions


REALIZATIONS = (("v3_s1", "v3", 1, {}, 1.0), ("v3_s2", "v3", 2, {}, 1.0),
                ("v3_static05_s1", "v3", 1, {"static_hide_prob": 0.5}, 0.95),
                ("v6_s1", "v6", 1, {}, 0.85))


def _features(root, match_id, match, tag, config, seed):
    key = f"sportec_{match_id}_degrade_{tag}"
    expected = match_fingerprint(degrade(match, config, seed))
    paths = _feature_cache_paths(root / "local_data/cache", key)
    if any(path.stem != f"{key}_{expected}" for path in paths):
        raise ValueError(f"feature fingerprint mismatch for {key}")
    raw = pd.concat([change_features(pd.read_parquet(paths[0]), match.half),
                     *(pd.read_parquet(path) for path in paths[1:])], axis=1)
    mirrored = mirror_v1ctl_physv2_df(raw, match.half, match.pitch)
    orig = mirror_v1ctl_physv2_df(mirrored, match.half, match.pitch)
    columns = list(orig.columns)
    xo = orig.to_numpy(dtype=np.float32, copy=True)
    xm = mirror_v1ctl_physv2_df(pd.DataFrame(xo, columns=columns), match.half, match.pitch).to_numpy(dtype=np.float32, copy=True)
    return xo, xm, columns


def _model(columns, seed):
    return SymPhysConformerV11(
        col_names=columns, channels=96, dilations=(1, 2, 4, 8, 16, 32, 64),
        conformer_stride=8, window_size=1501, dropout=0.16, lr=1.5e-3,
        weight_decay=1.2e-4, epochs=8, crop=1500, batch_size=32, poss_alpha=0.40,
        boundary_boost=1.65, soft_iou_weight=0.28, dead_guard_weight=0.10,
        change_loss_weight=0.35, b4d_loss_weight=0.48, duel_loss_weight=0.25,
        aerial_loss_weight=0.20, ball_drop_prob=0.20, feat_noise_std=0.03,
        label_smoothing=0.02, sliding_window_blend=0.50, seed=seed,
    )


def _fit_fold(root, held_out):
    data = root / "data_external/sportec_dfl"
    configs = _load_degrade_configs(root)
    xorig, xmirror, labels, halves, ball_zs, weights = [], [], [], [], [], []
    columns = None
    for mid in MATCH_IDS:
        if mid == held_out:
            continue
        match = load(mid, data_dir=data, use_cache=False)
        for tag, version, seed, overrides, weight in REALIZATIONS:
            config = replace(configs[version], **overrides) if overrides else configs[version]
            xo, xm, cols = _features(root, mid, match, tag, config, seed)
            if columns is None:
                columns = cols
            if columns != cols:
                raise ValueError("inconsistent feature columns")
            xorig.append(xo); xmirror.append(xm); labels.append(match.labels)
            halves.append(match.half); ball_zs.append(match.meta.get("ball_z")); weights.append(weight)
    helper = SymPhysConformerV11(col_names=columns, crop=1500, poss_alpha=0.40, boundary_boost=1.65)
    segments = helper.prepare_segments(xorig, xmirror, labels, halves, ball_zs=ball_zs, sample_weights=weights)
    stats = (helper.sym_mean, helper.sym_std)
    del xorig, xmirror, helper
    gc.collect()
    test_match = load(held_out, data_dir=data, use_cache=False)
    tests = {tag: _features(root, held_out, test_match, f"{tag}_s0", configs[tag], 0)
             for tag in ("v3", "v6")}
    if any(cols != columns for _, _, cols in tests.values()):
        raise ValueError("held-out feature columns differ from training")
    index = MATCH_IDS.index(held_out)
    predictions = {tag: [] for tag in tests}
    for seed in (42 + index, 142 + index):
        print(f"Training held-out fold {held_out}, seed {seed}", flush=True)
        model = _model(columns, seed).fit([], [], labels, halves, prepared_segments=segments, sym_stats=stats)
        for tag, (xo, xm, _) in tests.items():
            predictions[tag].append(model.predict_all_heads(xo, xm, test_match.half))
        print(f"Finished model for held-out fold {held_out}, seed {seed}", flush=True)
        del model
        gc.collect()
    result = {}
    for tag, heads in predictions.items():
        # The historical OOF NPZ uses shorter names than the inference cache.
        names = {"p_duel_to": "duel_to", "p_aerial": "aerial"}
        result.update({f"{tag}_{names.get(key, key)}": value
                       for key, value in _pack_predictions(heads).items()})
    return result


def rebuild_phys_fold(root: Path, output_path: Path) -> Path:
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
    names = {f"pc11_{mid}.npz": mid for mid in MATCH_IDS}
    if output_path.name not in names:
        raise ValueError("output must be named pc11_<Sportec match ID>.npz")
    if output_path.exists():
        raise FileExistsError(output_path)
    arrays = _fit_fold(root, names[output_path.name])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return output_path
