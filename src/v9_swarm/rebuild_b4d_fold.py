"""Regenerate a public Sportec boundary-model fold used by the v11 chain."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from src.data.degrade import degrade
from src.data.sportec import MATCH_IDS, load_all
from src.features.v1ctl_plus import build_v1ctl_plus, mirror_v1ctl_plus_df
from src.models.boundary_4dir import DEFAULT_4DIR_PARAMS, SymBoundary4DirGBM, augment_boundary_features, targets_4dir


def rebuild_b4d_fold(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the checkout") from exc
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be local")
    names = {f"b4d_{mid}.npz": mid for mid in MATCH_IDS}
    if output_path.name not in names:
        raise ValueError("output must identify a public Sportec held-out match")
    if output_path.exists():
        raise FileExistsError(output_path)
    for directory in (root / "scripts", root / "evaluation"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from s2_1_degradation import load_config
    from s3_7_nested_full import TRAIN_VARIANTS

    held = names[output_path.name]
    matches = load_all()
    cfg3 = load_config(root / "configs/degrade_v3.json")
    cfg6 = load_config(root / "configs/degrade_v6.json")
    original, mirrored, targets = [], [], []
    for mid in MATCH_IDS:
        if mid == held:
            continue
        match = matches[mid]
        yd = targets_4dir(match.labels, match.half)
        keep = np.flatnonzero(yd["y_h2a"] >= 0)[::5]
        target = {key: value[keep] for key, value in yd.items()}
        for tag, overrides, seed in TRAIN_VARIANTS:
            cfg = replace(cfg3, **overrides) if overrides else cfg3
            frame = build_v1ctl_plus(degrade(match, cfg, seed), f"sportec_{mid}_degrade_v3_{tag}")
            flipped = mirror_v1ctl_plus_df(frame, match.half, match.pitch)
            original.append(augment_boundary_features(frame, match.half, match.pitch).iloc[keep])
            mirrored.append(augment_boundary_features(flipped, match.half, match.pitch).iloc[keep])
            targets.append(target)
    model = SymBoundary4DirGBM(params=dict(DEFAULT_4DIR_PARAMS), subsample_every=1,
                               seed=MATCH_IDS.index(held)).fit(original, mirrored, targets)
    match = matches[held]
    outputs = {}
    for version, cfg in (("v3", cfg3), ("v6", cfg6)):
        frame = build_v1ctl_plus(degrade(match, cfg, 0), f"sportec_{held}_degrade_{version}_s0")
        flipped = mirror_v1ctl_plus_df(frame, match.half, match.pitch)
        outputs[f"p4d_{version}"] = model.predict_from_augmented(
            augment_boundary_features(frame, match.half, match.pitch),
            augment_boundary_features(flipped, match.half, match.pitch))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        try:
            np.savez_compressed(handle, **outputs)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
    return output_path
