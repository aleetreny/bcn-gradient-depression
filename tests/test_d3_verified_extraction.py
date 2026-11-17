"""The legacy extractor only sees verified inputs in its staging worktree."""

import hashlib
import json

import pytest

from src.v10_first_principles import d3_reteam_and_repro, d3_verified_extraction
from src.v10_first_principles.d3_input_preflight import FEATURE_PATHS, STREAM_PATHS


@pytest.fixture
def extraction_inputs(tmp_path):
    root = tmp_path.resolve()
    roles = {"features": FEATURE_PATHS, "streams": STREAM_PATHS}
    manifest = {"count": {}}
    for role, paths in roles.items():
        manifest["count"][role] = {}
        for relative in paths:
            path = root / role / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            contents = relative.encode()
            path.write_bytes(contents)
            manifest["count"][role][relative] = hashlib.sha256(contents).hexdigest()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    stage_root = root / "_local"
    stage_root.mkdir()
    return {
        "feature_root": root / "features",
        "stream_root": root / "streams",
        "manifest_path": manifest_path,
        "staging_root": stage_root,
        "project_root": root,
    }


def test_extracts_only_the_verified_snapshot(monkeypatch, extraction_inputs):
    atd = object()
    stages = []

    def extract(worktree, main, tracking):
        stages.append(worktree.parent)
        assert tracking is atd
        assert worktree.name == "v85sc-count"
        assert not list(main.iterdir())
        for prefix, paths in (
            ("local_data/cache", FEATURE_PATHS),
            ("experiments/outputs", STREAM_PATHS),
        ):
            base = worktree / prefix
            assert {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()} == set(paths)
            assert all((base / p).read_bytes() == p.encode() for p in paths)
        return {"example": tracking}

    monkeypatch.setattr(d3_reteam_and_repro, "extract_bar_ath_streams_from_worktree", extract)
    result = d3_verified_extraction.extract_verified_d3_streams("count", atd, **extraction_inputs)

    assert result == {"example": atd}
    assert all(not stage.exists() for stage in stages)


def test_rejects_copy_changed_after_source_verification(monkeypatch, extraction_inputs):
    def changed_copy(source, target):
        target.write_bytes(b"changed")

    def extract(*args):
        pytest.fail("extractor must not run after verification fails")

    monkeypatch.setattr(d3_verified_extraction, "copyfile", changed_copy)
    monkeypatch.setattr(d3_reteam_and_repro, "extract_bar_ath_streams_from_worktree", extract)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        d3_verified_extraction.extract_verified_d3_streams("count", object(), **extraction_inputs)


def test_rejects_staging_outside_allowed_dirs(extraction_inputs):
    extraction_inputs["staging_root"] = extraction_inputs["project_root"]

    with pytest.raises(ValueError, match="must be under _local"):
        d3_verified_extraction.extract_verified_d3_streams("count", object(), **extraction_inputs)


def test_rejects_symlink_staging(extraction_inputs):
    stage = extraction_inputs["staging_root"]
    link = stage / "link"
    link.symlink_to(stage.parent, target_is_directory=True)
    extraction_inputs["staging_root"] = link

    with pytest.raises(ValueError, match="cannot contain symlinks"):
        d3_verified_extraction.extract_verified_d3_streams("count", object(), **extraction_inputs)
