"""Run the historical D3 stream extractor against isolated, verified inputs."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Any

from src.v10_first_principles.d3_input_preflight import (
    FEATURE_PATHS,
    STREAM_PATHS,
    preflight_d3_inputs,
)


def extract_verified_d3_streams(
    variant: str,
    atd: Any,
    *,
    manifest_path: Path,
    feature_root: Path,
    stream_root: Path,
    staging_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Extract one variant from a private snapshot of its historical inputs.

    The caller supplies the already loaded tracking object and a local staging
    parent. The legacy extractor receives an empty main root for its fallbacks.
    """
    from src.v10_first_principles.d3_reteam_and_repro import (
        extract_bar_ath_streams_from_worktree,
    )

    if any(entry.is_symlink() for entry in (staging_root, *staging_root.absolute().parents)):
        raise ValueError("D3 staging root cannot contain symlinks")
    staging_root = staging_root.resolve(strict=True)
    relative = staging_root.relative_to(project_root.resolve(strict=True))
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("D3 staging root must be under _local/ or experiments/outputs/")

    manifest = manifest_path.read_bytes()
    verified = preflight_d3_inputs(
        variant,
        manifest_path=manifest_path,
        feature_root=feature_root,
        stream_root=stream_root,
    )
    with TemporaryDirectory(prefix="d3-", dir=staging_root) as staging:
        staged = Path(staging).resolve()
        worktree = staged / f"v85sc-{variant}"
        stage_features = worktree / "local_data" / "cache"
        stage_streams = worktree / "experiments" / "outputs"
        no_fallback_root = staged / "empty-main"
        no_fallback_root.mkdir()
        for sources, relative_paths, destination in (
            (verified.feature_paths, FEATURE_PATHS, stage_features),
            (verified.stream_paths, STREAM_PATHS, stage_streams),
        ):
            for source, relative in zip(sources, relative_paths, strict=True):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                copyfile(source, target)
        stage_manifest = staged / "manifest.json"
        stage_manifest.write_bytes(manifest)
        preflight_d3_inputs(
            variant,
            manifest_path=stage_manifest,
            feature_root=stage_features,
            stream_root=stage_streams,
        )
        return extract_bar_ath_streams_from_worktree(worktree, no_fallback_root, atd)
