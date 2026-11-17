"""Rebuild raw local or public-sample PlayerSet inference inputs."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def rebuild_inference_tensor(root: Path, output_path: Path) -> Path:
    from src.data.metrica_atd import load as load_atd
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.models.player_set_net import extract_player_set_tensors

    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    if output_path.parent != root / "_local":
        raise ValueError("output must be directly under _local/")
    samples = {f"pset_inference_sample_game{gid}.npz": gid for gid in GAMES}
    if output_path.name not in samples and output_path.name != "pset_inference_atd_raw.npz":
        raise ValueError("output must identify a public sample or the raw match")
    if output_path.exists():
        raise FileExistsError(output_path)
    if output_path.name in samples:
        match = load_sample(samples[output_path.name], use_cache=False)
    else:
        match = load_atd(use_cache=False)
    tensor = extract_player_set_tensors(match)
    fields = ("p_h", "p_a", "mask_h", "mask_a", "g_anti", "g_sym")
    arrays = {key: np.asarray(getattr(tensor, key)) for key in fields}
    if any(len(value) != len(match.half) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("invalid tensor values or frame count")
    with NamedTemporaryFile(dir=output_path.parent, suffix=".npz") as tmp:
        np.savez(tmp, **arrays)
        tmp.flush()
        os.link(tmp.name, output_path)
    return output_path
