"""Rebuild an external Sportec out-of-fold v6-genuine prediction cache.

The historical fold recipe is preserved. Retrained GBM probabilities can
differ numerically from the saved historical probabilities.
"""
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


def rebuild_v6_genuine(root: Path, output_path: Path) -> Path:
    """Train and evaluate the fold named by v6_genuine_<Sportec match ID>.npz."""
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this reconstruction module")
    requested = root / output_path
    destination = requested.resolve()
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    matches_by_filename = {f"v6_genuine_{mid}.npz": mid for mid in MATCH_IDS}
    if destination.name not in matches_by_filename:
        raise ValueError("output must be named v6_genuine_<Sportec match ID>.npz")
    if requested.is_symlink() or destination.exists():
        raise FileExistsError(requested)

    target = matches_by_filename[destination.name]
    proba, transition = _fit_fold(root, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        try:
            np.savez_compressed(
                handle,
                symgbm_plus_tta_on_v6=proba,
                symtrans_plus_tta_on_v6=transition,
            )
            handle.flush()
        except BaseException:
            destination.unlink()
            raise
    return destination


def _fit_fold(root: Path, target: str) -> tuple[np.ndarray, np.ndarray]:
    for directory in (root / "scripts", root / "evaluation"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from exp_c_plus_eval import predict_gbm_plus_tta, predict_trans_plus_tta
    from s2_1_degradation import load_config
    from s3_7_nested_full import TRAIN_VARIANTS

    cfg_v3 = load_config(root / "configs" / "degrade_v3.json")
    cfg_v6 = load_config(root / "configs" / "degrade_v6.json")
    sportec = load_all()
    train_ids = [mid for mid in MATCH_IDS if mid != target]
    variants = [
        (key, replace(cfg_v3, **overrides) if overrides else cfg_v3, seed)
        for key, overrides, seed in TRAIN_VARIANTS
    ]
    features, mirrored, labels, groups, transitions = [], [], [], [], []
    for group, mid in enumerate(train_ids):
        match = sportec[mid]
        for key, config, seed in variants:
            table = build_v1ctl_plus(
                degrade(match, config, seed), f"sportec_{mid}_degrade_v3_{key}"
            )
            features.append(table)
            mirrored.append(mirror_v1ctl_plus_df(table, match.half, match.pitch))
            labels.append(match.labels)
            groups.append(group)
            transitions.append(change_targets(match.labels, match.half))

    train = features + mirrored
    train_labels = labels + [swap_labels(y) for y in labels]
    train_groups = groups + [len(train_ids) + group for group in groups]
    fold_seed = MATCH_IDS.index(target)
    gbm = TwoHeadGBM(seed=fold_seed).fit(train, train_labels, train_groups)
    transition_gbm = TransitionGBM(seed=fold_seed).fit(train, transitions + transitions)

    held_out = sportec[target]
    test = build_v1ctl_plus(
        degrade(held_out, cfg_v6, 0), f"sportec_{target}_degrade_v6_s0"
    )
    return (
        predict_gbm_plus_tta(gbm, test, held_out.half, held_out.pitch),
        predict_trans_plus_tta(transition_gbm, test, held_out.half, held_out.pitch),
    )
