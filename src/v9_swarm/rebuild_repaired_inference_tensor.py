"""Build the broadcast-repaired raw match tensor entirely on this machine."""
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


def rebuild_repaired_inference_tensor(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    if output_path != root / "_local/pset_inference_atd_rep.npz":
        raise ValueError("output must be _local/pset_inference_atd_rep.npz")
    if output_path.exists():
        raise FileExistsError(output_path)
    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match
    from s4_6_broadcast_align import load_broadcast
    from src.data.metrica_atd import load as load_atd
    from src.models.player_set_net import extract_player_set_tensors

    atd = load_atd(use_cache=False)
    players, ball, team = load_broadcast()
    valid = pd.read_csv(root / "data/broadcast/BAR-ATH_broadcast_homography.csv")["valid"].to_numpy().astype(bool)
    mapping, _ = enhanced_broadcast_alignment(atd, players, valid)
    repaired = repair_atd_match(atd, players, ball, team, mapping, True, True, None)
    tensor = extract_player_set_tensors(repaired)
    fields = ("p_h", "p_a", "mask_h", "mask_a", "g_anti", "g_sym")
    arrays = {key: np.asarray(getattr(tensor, key)) for key in fields}
    if any(len(value) != len(atd.half) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("invalid tensor values or frame count")
    with NamedTemporaryFile(dir=output_path.parent, suffix=".npz") as tmp:
        np.savez(tmp, **arrays)
        tmp.flush()
        os.link(tmp.name, output_path)
    return output_path
