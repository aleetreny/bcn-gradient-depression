"""Rebuild one verified local D3 stream variant for the harness.

This reconstructs historical streams. It does not verify the v11 heads or
score predictions.
"""

import json
import os
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile

from src.data.metrica_atd import load as load_atd
from src.v10_first_principles.d3_verified_extraction import extract_verified_d3_streams


CONFIG_PATH = Path("_local/d3_reconstruction.json")
CONFIG_KEYS = ("variant", "manifest_path", "feature_root", "stream_root", "staging_root")


def rebuild_verified_streams(root: Path, output_path: Path) -> Path:
    """Read private configuration and exclusively publish a local stream cache.

    Relative configuration paths are resolved from the checkout root.
    """
    root = Path(root).resolve(strict=True)
    output_path = Path(output_path)
    if output_path.is_symlink():
        raise ValueError("output must not be a symlink")
    output_path = output_path.resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the checkout") from exc
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if output_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.exists():
        raise FileExistsError(output_path)

    config_path = root / CONFIG_PATH
    if config_path.is_symlink() or config_path.parent.is_symlink():
        raise ValueError("D3 reconstruction config must be a regular local file")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != set(CONFIG_KEYS) or any(
        not isinstance(config[key], str) or not config[key].strip() for key in CONFIG_KEYS
    ):
        raise ValueError(f"D3 reconstruction config must supply {', '.join(CONFIG_KEYS)} as strings")
    atd = load_atd(data_dir=root / "data", use_cache=False)
    streams = extract_verified_d3_streams(
        config["variant"],
        atd,
        manifest_path=root / config["manifest_path"],
        feature_root=root / config["feature_root"],
        stream_root=root / config["stream_root"],
        staging_root=root / config["staging_root"],
        project_root=root,
    )
    with NamedTemporaryFile(dir=output_path.parent) as temporary:
        pickle.dump(streams, temporary, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.flush()
        os.link(temporary.name, output_path)
    return output_path
