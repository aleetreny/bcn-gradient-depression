"""Read-only verification of local D3 inputs against a trusted JSON manifest.

The manifest maps variant (main, count, all) to role (features, streams, heads)
to relative path to lowercase SHA-256. Build it from trusted reference inputs
independently of the candidate directories being checked.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


VARIANTS = ("main", "count", "all")
FEATURE_PATHS = tuple(
    f"{part}/{name}.parquet"
    for part in ("features_v1", "control_v1", "plus_v1", "physv2")
    for name in ("atd_0f083428befb19d09710022b", "atd_rep_True_True_None_75a82cda201463f46ad07ae9")
)
STREAM_PATHS = (
    "exp_c_plus_oof/eval_preds_plus.pkl",
    "exp_g_oof/v6_j23_preds.pkl",
    "exp_h_oof/bigru_j23_preds.pkl",
    "exp_w3_oof/b4d_j23_preds.pkl",
    "exp_w4_oof/eval_preds_physv2.pkl",
    "exp_w5_tree_oof/eval_preds_cb_xgb.pkl",
    "exp_w5b_oof/seq_physv2_j23_preds.pkl",
    "exp_w6_oof/atd_raw_v6_bigru.pkl",
    "exp_w7a_oof/pset_j23_preds.pkl",
    "exp_w6_oof/w6_results_summary.pkl",
    "exp_w8a_oof/w8a_j23_preds.pkl",
)
HEAD_PATHS = (
    "v9_swarm/pset_v2_oof/pset_v2_j23_preds.pkl",
    "v9_swarm/l1_trees_oof/l1_trees_j23_preds.pkl",
    "v10_first_principles/traj_v10_oof/traj_v10_j23_preds.pkl",
    "v11_big_leap/phys_conformer_v11_oof/pc11_j23_preds.pkl",
)


@dataclass(frozen=True)
class D3Inputs:
    variant: str
    feature_paths: tuple[Path, ...]
    stream_paths: tuple[Path, ...]
    head_paths: tuple[Path, ...]
    head_variant: str | None


def _check_paths(manifest, variant, role, root, relative_paths, errors):
    root = Path(root)
    if any(entry.is_symlink() for entry in (root, *root.absolute().parents)):
        errors.append(f"{variant}/{role}: symlink in root path {root}")
        return ()
    if not root.is_dir():
        errors.append(f"{variant}/{role}: missing directory {root}")
        return ()
    base = root.resolve()
    verified = []
    for relative in relative_paths:
        path = base / relative
        label = f"{variant}/{role}/{relative}"
        expected = manifest.get(variant, {}).get(role, {}).get(relative)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"{label}: missing or invalid manifest SHA-256")
            continue
        if any(
            entry.is_symlink()
            for entry in (path, *path.parents)
            if entry.is_relative_to(base)
        ):
            errors.append(f"{label}: symlink in input path {path}")
            continue
        if not path.is_file():
            errors.append(f"{label}: missing regular file {path}")
            continue
        try:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            errors.append(f"{label}: could not read {path} ({exc})")
            continue
        if actual != expected:
            matching = [
                other for other in VARIANTS
                if other != variant and manifest.get(other, {}).get(role, {}).get(relative) == actual
            ]
            detail = f"; matches variant(s) {', '.join(matching)}" if matching else ""
            errors.append(f"{label}: SHA-256 mismatch at {path}{detail}")
            continue
        verified.append(path)
    return tuple(verified)


def preflight_d3_inputs(
    variant: str,
    *,
    manifest_path: Path,
    feature_root: Path,
    stream_root: Path,
    head_root: Path | None = None,
    use_count_heads_for_all: bool = False,
) -> D3Inputs:
    """Verify raw and repaired features and variant streams.

    Roots are the feature cache directory and the stream experiments/outputs
    directory. Omit head_root when verifying only the historical v8.5 inputs.
    For v11 synthesis, head_root is the experiments/outputs directory carrying
    the additional heads. The historical all/count combination requires opt-in.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown D3 variant {variant!r}; expected one of {VARIANTS}")
    if use_count_heads_for_all and (variant != "all" or head_root is None):
        raise ValueError("Count-head reuse requires variant all and an explicit head_root")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or any(
        not isinstance(roles, dict) or any(not isinstance(hashes, dict) for hashes in roles.values())
        for roles in manifest.values()
    ):
        raise ValueError("D3 manifest must map variants to roles to path-to-SHA-256 mappings")

    errors = []
    features = _check_paths(manifest, variant, "features", feature_root, FEATURE_PATHS, errors)
    streams = _check_paths(manifest, variant, "streams", stream_root, STREAM_PATHS, errors)
    heads = ()
    head_variant = None
    if head_root is not None:
        head_variant = "count" if use_count_heads_for_all else variant
        if variant == "all" and not use_count_heads_for_all and "heads" not in manifest.get("all", {}):
            errors.append("all/heads: all-specific heads required; historical count reuse requires explicit opt-in")
        heads = _check_paths(manifest, head_variant, "heads", head_root, HEAD_PATHS, errors)
    if errors:
        raise ValueError("D3 input preflight failed:\n" + "\n".join(errors))
    return D3Inputs(variant, features, streams, heads, head_variant)
