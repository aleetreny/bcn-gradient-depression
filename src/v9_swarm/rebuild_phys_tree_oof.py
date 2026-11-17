"""Regenerate one public Sportec physics-tree fold used by the v11 chain."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from src.data.degrade import degrade
from src.data.sportec import MATCH_IDS, load_all
from src.features.symmetry import swap_labels
from src.features.v1ctl_physics_v2 import build_v1ctl_physics_v2, mirror_v1ctl_physv2_df
from src.models.gbm import DEFAULT_PARAMS as GBM_DEFAULT_PARAMS, TwoHeadGBM
from src.models.transition import PARAMS as TRANS_DEFAULT_PARAMS, TransitionGBM, change_targets


def rebuild_phys_tree_oof(root: Path, output_path: Path) -> Path:
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
    names = {f"physv2_{mid}.npz": mid for mid in MATCH_IDS}
    if output_path.name not in names:
        raise ValueError("output must identify a public Sportec held-out match")
    if output_path.exists():
        raise FileExistsError(output_path)
    for directory in (root / "scripts", root / "evaluation"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from exp_w4_physics_v2_build import predict_gbm_physv2_tta, predict_trans_physv2_tta
    from s2_1_degradation import load_config
    from s3_7_nested_full import TRAIN_VARIANTS

    held = names[output_path.name]
    matches = load_all()
    cfg3 = load_config(root / "configs/degrade_v3.json")
    cfg6 = load_config(root / "configs/degrade_v6.json")
    train_ids = [mid for mid in MATCH_IDS if mid != held]
    original, mirrored, labels, groups, transitions = [], [], [], [], []
    for group, mid in enumerate(train_ids):
        match = matches[mid]
        for tag, overrides, seed in TRAIN_VARIANTS:
            cfg = replace(cfg3, **overrides) if overrides else cfg3
            frame = build_v1ctl_physics_v2(degrade(match, cfg, seed), f"sportec_{mid}_degrade_v3_{tag}")
            original.append(frame)
            mirrored.append(mirror_v1ctl_physv2_df(frame, match.half, match.pitch))
            labels.append(match.labels)
            groups.append(group)
            transitions.append(change_targets(match.labels, match.half))
    symmetric_x = original + mirrored
    symmetric_y = labels + [swap_labels(y) for y in labels]
    symmetric_g = groups + [len(train_ids) + group for group in groups]
    seed = MATCH_IDS.index(held)
    gbm = TwoHeadGBM(params=dict(GBM_DEFAULT_PARAMS, n_jobs=4), seed=seed).fit(symmetric_x, symmetric_y, symmetric_g)
    transition = TransitionGBM(params=dict(TRANS_DEFAULT_PARAMS, n_jobs=4), seed=seed).fit(symmetric_x, transitions + transitions)
    match = matches[held]
    outputs = {}
    for version, cfg in (("v3", cfg3), ("v6", cfg6)):
        frame = build_v1ctl_physics_v2(degrade(match, cfg, 0), f"sportec_{held}_degrade_{version}_s0")
        suffix = "" if version == "v3" else "_on_v6"
        outputs[f"symgbm_physv2_tta{suffix}"] = predict_gbm_physv2_tta(gbm, frame, match.half, match.pitch)
        outputs[f"symtrans_physv2_tta{suffix}"] = predict_trans_physv2_tta(transition, frame, match.half, match.pitch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        try:
            np.savez_compressed(handle, **outputs)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
    return output_path
