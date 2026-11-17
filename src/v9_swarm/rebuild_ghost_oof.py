"""Regenerate public Sportec ghost-model fold inputs used by the v11 chain."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from src.data.degrade import degrade
from src.data.sportec import MATCH_IDS, load_all
from src.features.symmetry import mirror_v1ctl_df, swap_labels
from src.models.gbm import TwoHeadGBM
from src.models.transition import TransitionGBM, change_targets


def rebuild_ghost_oof(root: Path, output_path: Path) -> Path:
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
    names = {f"ghost_sym_{mid}.npz": mid for mid in MATCH_IDS}
    if output_path.name not in names:
        raise ValueError("output must identify a public Sportec held-out match")
    if output_path.exists():
        raise FileExistsError(output_path)
    for directory in (root / "scripts", root / "evaluation"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from exp_b_symmetry_eval import predict_gbm_tta, predict_trans_tta
    from exp_g_ghost_sym import FAST_GBM_PARAMS, FAST_TRANS_PARAMS
    from s2_1_degradation import load_config
    from s3_7_nested_full import TRAIN_VARIANTS, build_features

    held = names[output_path.name]
    matches = load_all()
    configs = {key: load_config(root / f"configs/degrade_{key}.json") for key in ("v3", "v6")}
    train_ids = [mid for mid in MATCH_IDS if mid != held]
    labels, groups, transitions = [], [], []
    for group, mid in enumerate(train_ids):
        match = matches[mid]
        for _tag, _overrides, _seed in TRAIN_VARIANTS:
            labels.append(match.labels)
            groups.append(group)
            transitions.append(change_targets(match.labels, match.half))
    symmetric_y = labels + [swap_labels(y) for y in labels]
    symmetric_g = groups + [len(train_ids) + group for group in groups]
    test_match = matches[held]
    tests = {v: build_features(degrade(test_match, cfg, 0), f"sportec_{held}_degrade_{v}_s0", "v1ctl")
             for v, cfg in configs.items()}
    outputs = {}
    for version, cfg in configs.items():
        original, mirrored = [], []
        for mid in train_ids:
            match = matches[mid]
            for tag, overrides, seed in TRAIN_VARIANTS:
                config = replace(cfg, **overrides) if overrides else cfg
                x = build_features(degrade(match, config, seed), f"sportec_{mid}_degrade_{version}_{tag}", "v1ctl")
                original.append(x)
                mirrored.append(mirror_v1ctl_df(x, match.half, match.pitch))
        xs = original + mirrored
        seed = MATCH_IDS.index(held)
        model = TwoHeadGBM(params=FAST_GBM_PARAMS, seed=seed).fit(xs, symmetric_y, symmetric_g)
        transition = TransitionGBM(params=FAST_TRANS_PARAMS, seed=seed).fit(xs, transitions + transitions)
        targets = ("v3", "v6") if version == "v6" else ("v6",)
        for test_version in targets:
            outputs[f"g_{version}_on_{test_version}"] = predict_gbm_tta(model, tests[test_version], test_match.half, test_match.pitch)
            outputs[f"tr_{version}_on_{test_version}"] = predict_trans_tta(transition, tests[test_version], test_match.half, test_match.pitch)
        if version == "v3":
            # The original v3-on-v3 entries came from EXP-B's default-thread models.
            model = TwoHeadGBM(seed=seed).fit(xs, symmetric_y, symmetric_g)
            transition = TransitionGBM(seed=seed).fit(xs, transitions + transitions)
            outputs["g_v3_on_v3"] = predict_gbm_tta(model, tests["v3"], test_match.half, test_match.pitch)
            outputs["tr_v3_on_v3"] = predict_trans_tta(transition, tests["v3"], test_match.half, test_match.pitch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        try:
            np.savez_compressed(handle, **outputs)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
    return output_path
