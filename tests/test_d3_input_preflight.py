"""D3 preflight checks provenance without interpreting artifact contents."""

import hashlib
import json

import pytest

from src.v10_first_principles.d3_input_preflight import (
    FEATURE_PATHS,
    HEAD_PATHS,
    STREAM_PATHS,
    preflight_d3_inputs,
)


@pytest.fixture
def inputs(tmp_path):
    tmp_path = tmp_path.resolve()
    manifest = {}
    for variant in ("main", "count", "all"):
        roles = {"features": FEATURE_PATHS, "streams": STREAM_PATHS}
        if variant == "count":
            roles["heads"] = HEAD_PATHS
        manifest[variant] = {}
        for role, paths in roles.items():
            manifest[variant][role] = {}
            for path in paths:
                contents = f"{variant}/{role}/{path}".encode()
                artifact = tmp_path / variant / role / path
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(contents)
                manifest[variant][role][path] = hashlib.sha256(contents).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def verify(root, variant, **kwargs):
    return preflight_d3_inputs(
        variant,
        manifest_path=root / "manifest.json",
        feature_root=root / variant / "features",
        stream_root=root / variant / "streams",
        **kwargs,
    )


def test_complete_variant_without_heads(inputs):
    result = verify(inputs, "count")

    assert result.feature_paths == tuple(inputs / "count/features" / p for p in FEATURE_PATHS)
    assert result.stream_paths == tuple(inputs / "count/streams" / p for p in STREAM_PATHS)
    assert result.head_variant is None


def test_collect_missing_and_mixed_files(inputs):
    source = inputs / "main/streams" / STREAM_PATHS[1]
    (inputs / "count/streams" / STREAM_PATHS[1]).write_bytes(source.read_bytes())
    (inputs / "count/features" / FEATURE_PATHS[0]).unlink()

    with pytest.raises(ValueError) as error:
        verify(inputs, "count")

    assert "missing regular file" in str(error.value)
    assert "matches variant(s) main" in str(error.value)


def test_reject_symlink_even_when_target_matches_hash(inputs):
    path = inputs / "count/streams" / STREAM_PATHS[0]
    target = inputs / "outside.pkl"
    path.rename(target)
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symlink in input path"):
        verify(inputs, "count")


def test_all_head_source_must_be_explicit(inputs):
    head_root = inputs / "count/heads"
    with pytest.raises(ValueError, match="all-specific heads required"):
        verify(inputs, "all", head_root=head_root)

    result = verify(inputs, "all", head_root=head_root, use_count_heads_for_all=True)

    assert result.variant == "all"
    assert result.head_variant == "count"
    assert result.head_paths == tuple(head_root / p for p in HEAD_PATHS)


def test_reject_symlink_in_root_ancestors(inputs):
    alias = inputs / "alias"
    alias.symlink_to(inputs / "count", target_is_directory=True)

    with pytest.raises(ValueError, match="symlink in root path"):
        preflight_d3_inputs(
            "count",
            manifest_path=inputs / "manifest.json",
            feature_root=alias / "features",
            stream_root=inputs / "count/streams",
        )
