"""Reassemble W8B's Sportec cache from public data and historical OOF files."""

from contextlib import ExitStack
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.data.sportec import load_all
from src.models.transition import change_features
from src.v9_swarm.rebuild_candidates import _feature_cache_paths


COLUMNS = (
    "ball_visible", "ball_x", "ball_y", "ball_last_to_touchline",
    "ball_last_to_goalline", "ball_outside_pitch", "ball_speed", "ball_accel",
    "dist_ball_H", "dist_ball_A", "dist_ball_diff", "ball_angle_change",
    "ball_speed_change", "impulse_mag_d2", "impulse_tan_d2", "impulse_norm_d2",
    "impulse_cos_d2", "impulse_mag_d4", "impulse_tan_d4", "impulse_norm_d4",
    "impulse_cos_d4", "impulse_mag_d8", "chord_in_flight", "d1_h_impulse_gated",
    "d1_a_impulse_gated", "prev_knot_d1_h", "prev_knot_d1_a", "next_knot_d1_h",
    "next_knot_d1_a", "chord_same_team_prob_h", "chord_same_team_prob_a",
    "chord_turnover_prob", "ball_focus_discrepancy", "frac_still_all",
)

FOLD_PATHS = {
    "plus": "exp_c_plus_oof/plus",
    "ghost": "exp_g_oof/ghost_sym",
    "gru": "exp_h_oof/bigru_sym",
    "tcn": "exp_b_sym_oof/tcn_sym",
    "b4d": "exp_w3_oof/b4d",
    "phys": "exp_w4_oof/physv2",
    "trees": "exp_w5_tree_oof/cb_xgb",
    "phys_tcn": "exp_w5b_oof/tcn_physv2",
    "phys_gru": "exp_w5b_oof/bigru_physv2",
    "pset": "exp_w7a_oof/pset",
    "v6": "exp_w7b_oof/v6_genuine",
    "w7b": "exp_w7b_oof/w7b",
}

V3_TREES = (
    ("plus", "symgbm_plus_tta"),
    ("phys", "symgbm_physv2_tta"),
    ("trees", "symcatboost_physv2_tta"),
    ("trees", "symxgb_physv2_tta"),
)
V6_TREES = (
    ("v6", "symgbm_plus_tta_on_v6"),
    ("phys", "symgbm_physv2_tta_on_v6"),
    ("trees", "symcatboost_physv2_tta_on_v6"),
    ("trees", "symxgb_physv2_tta_on_v6"),
)
BASE_SEQUENCE = (
    ("tcn", "symtcn_tta_proba"),
    ("gru", "symbigru_tta_proba"),
)
V6_SEQUENCE = (
    ("phys_tcn", "v6_proba"),
    ("phys_gru", "v6_proba"),
)


def _load_features(cache_dir, prefix, half):
    tables = []
    for index, path in enumerate(_feature_cache_paths(cache_dir, prefix)):
        table = pd.read_parquet(path)
        if len(table) != len(half):
            raise ValueError(f"frame count differs from Sportec tracking: {path}")
        tables.append(change_features(table, half) if index == 0 else table)
    return pd.concat(tables, axis=1)[list(COLUMNS)]


def _load_fold(out_dir, match_id, match, xv3, xv6):
    with ExitStack() as stack:
        folds = {
            kind: stack.enter_context(np.load(out_dir / f"{prefix}_{match_id}.npz"))
            for kind, prefix in FOLD_PATHS.items()
        }
        return {
            "gt": match.labels,
            "hl": match.half,
            "xv3": xv3,
            "xv6": xv6,
            "p4d_v3": folds["b4d"]["p4d_v3"],
            "p4d_v6": folds["b4d"]["p4d_v6"],
            "fams_v3": [
                folds[model][key] for model, key in V3_TREES + BASE_SEQUENCE + (
                    ("phys_tcn", "v3_proba"), ("phys_gru", "v3_proba"), ("pset", "v3_proba"),
                )
            ],
            "fams_v6_std": [
                folds[model][key] for model, key in V6_TREES + BASE_SEQUENCE + V6_SEQUENCE + (
                    ("pset", "v6_proba"),
                )
            ],
            "fams_v6_gen": [
                folds[model][key] for model, key in V6_TREES + V6_SEQUENCE + (
                    ("ghost", "g_v6_on_v6"), ("pset", "v6_proba"),
                )
            ],
            "w7b": dict(folds["w7b"]),
        }


def rebuild_sportec_cache(root: Path, output_path: Path) -> Path:
    """Write a new local pickle for EXP-15/16 without running their scoring."""
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this reconstruction module")
    output_path = Path(output_path).resolve()
    relative = output_path.relative_to(root)
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if output_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.exists():
        raise FileExistsError(output_path)

    out_dir = root / "experiments" / "outputs"
    cache_dir = root / "local_data" / "cache"
    cache = {}
    for match_id, match in load_all().items():
        xv3 = _load_features(cache_dir, f"sportec_{match_id}_degrade_v3_s0", match.half)
        xv6 = _load_features(cache_dir, f"sportec_{match_id}_degrade_v6_s0", match.half)
        cache[match_id] = _load_fold(out_dir, match_id, match, xv3, xv6)
    with output_path.open("xb") as output:
        pickle.dump(cache, output)
    return output_path
