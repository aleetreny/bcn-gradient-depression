"""Train one full public-Sportec PhysConformer and save it locally only.

This produces new weights, not the historical model. Feature caches and model
files are private local artifacts and must never be uploaded or committed.
"""
from dataclasses import replace
import gc
from pathlib import Path
import pickle

import torch

from src.data.sportec import MATCH_IDS, load
from src.v9_swarm.rebuild_pset_fold import _load_degrade_configs
from src.v11_big_leap.phys_conformer_v11 import SymPhysConformerV11
from src.v11_big_leap.rebuild_phys_fold import REALIZATIONS, _features, _model


def train_phys_full_local(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] != ("_local",):
        raise ValueError("model output must be under _local/")
    names = {f"phys_full_{seed}.pkl": seed for seed in (42, 142)}
    if output_path.name not in names:
        raise ValueError("output must be named phys_full_42.pkl or phys_full_142.pkl")
    if output_path.exists():
        raise FileExistsError(output_path)
    configs = _load_degrade_configs(root)
    data = root / "data_external/sportec_dfl"
    xorig, xmirror, labels, halves, ball_zs, weights = [], [], [], [], [], []
    columns = None
    for mid in MATCH_IDS:
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
    model = _model(columns, names[output_path.name]).fit(
        [], [], labels, halves, prepared_segments=segments, sym_stats=stats,
    )
    if model.net is not None:
        model.net = model.net.to("cpu").eval()
    model.device = torch.device("cpu")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
