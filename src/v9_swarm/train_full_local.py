"""Train one full public-Sportec PSet or trajectory model and save it locally.

The file is a new local model, not a recovery of historical weights. It must
never be uploaded or committed. Existing feature/tensor caches are required.
"""
from dataclasses import replace
from pathlib import Path
import pickle

from src.data.sportec import MATCH_IDS, load
from src.v9_swarm.rebuild_pset_fold import _load_degrade_configs, _load_tensors


def train_full_local(root: Path, output_path: Path) -> Path:
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
    names = {f"{kind}_full_{seed}.pkl": (kind, seed)
             for kind in ("pset", "trajectory") for seed in (42, 142)}
    if output_path.name not in names:
        raise ValueError("output must be named pset_full_<seed>.pkl or trajectory_full_<seed>.pkl")
    if output_path.exists():
        raise FileExistsError(output_path)
    kind, seed = names[output_path.name]
    if kind == "pset":
        from src.v9_swarm.rebuild_pset_fold import TRAIN_REALIZATIONS, _new_model
    else:
        from src.v10_first_principles.rebuild_trajectory_fold import TRAIN_REALIZATIONS, _new_model
    data = root / "data_external/sportec_dfl"
    cache = root / "local_data/cache/pset"
    configs = _load_degrade_configs(root)
    tensors, labels, halves, ball_zs, weights = [], [], [], [], []
    for mid in MATCH_IDS:
        match = load(mid, data_dir=data, use_cache=False)
        for tag, version, degradation_seed, overrides, weight in TRAIN_REALIZATIONS:
            config = replace(configs[version], **overrides) if overrides else configs[version]
            tensors.append(_load_tensors(cache, mid, tag, match, config, degradation_seed))
            labels.append(match.labels)
            halves.append(match.half)
            ball_zs.append(match.meta.get("ball_z"))
            weights.append(weight)
    model = _new_model(seed).fit(tensors, labels, halves, ball_zs=ball_zs, sample_weights=weights)
    if getattr(model, "net_", None) is not None:
        model.net_ = model.net_.to("cpu").eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
