"""Regenerate one public Sportec plus-model OOF input required by v11."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from src.data.degrade import degrade
from src.data.sportec import MATCH_IDS, load_all
from src.features.symmetry import swap_labels
from src.features.v1ctl_plus import build_v1ctl_plus, mirror_v1ctl_plus_df
from src.models.gbm import TwoHeadGBM
from src.models.transition import TransitionGBM, change_targets


def rebuild_plus_oof(root: Path, output_path: Path) -> Path:
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
    names = {f"plus_{mid}.npz": mid for mid in MATCH_IDS}
    if output_path.name not in names:
        raise ValueError("output must identify a public Sportec held-out match")
    if output_path.exists():
        raise FileExistsError(output_path)
    for directory in (root / "scripts", root / "evaluation"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from exp_c_plus_eval import predict_gbm_plus_tta, predict_trans_plus_tta
    from s2_1_degradation import load_config
    from s3_7_nested_full import TRAIN_VARIANTS

    held = names[output_path.name]
    matches = load_all()
    cfg = load_config(root / "configs/degrade_v3.json")
    train_ids = [mid for mid in MATCH_IDS if mid != held]
    features, mirrored, labels, groups, transitions = [], [], [], [], []
    for group, mid in enumerate(train_ids):
        match = matches[mid]
        for tag, overrides, seed in TRAIN_VARIANTS:
            config = replace(cfg, **overrides) if overrides else cfg
            x = build_v1ctl_plus(degrade(match, config, seed), f"sportec_{mid}_degrade_v3_{tag}")
            features.append(x)
            mirrored.append(mirror_v1ctl_plus_df(x, match.half, match.pitch))
            labels.append(match.labels)
            groups.append(group)
            transitions.append(change_targets(match.labels, match.half))
    test_match = matches[held]
    test = build_v1ctl_plus(degrade(test_match, cfg, 0), f"sportec_{held}_degrade_v3_s0")
    seed = MATCH_IDS.index(held)
    standard = TwoHeadGBM(seed=seed).fit(features, labels, groups)
    standard_p = predict_gbm_plus_tta(standard, test, test_match.half, test_match.pitch)
    symmetric_x = features + mirrored
    symmetric_y = labels + [swap_labels(y) for y in labels]
    symmetric_g = groups + [len(train_ids) + group for group in groups]
    symmetric = TwoHeadGBM(seed=seed).fit(symmetric_x, symmetric_y, symmetric_g)
    symmetric_p = predict_gbm_plus_tta(symmetric, test, test_match.half, test_match.pitch)
    transition = TransitionGBM(seed=seed).fit(symmetric_x, transitions + transitions)
    transition_p = predict_trans_plus_tta(transition, test, test_match.half, test_match.pitch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        try:
            np.savez_compressed(handle, gbm_plus_tta=standard_p, symgbm_plus_tta=symmetric_p,
                                symtrans_plus_tta=transition_p)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
    return output_path
