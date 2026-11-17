"""Feature-cache selection uses fingerprints shared by all producers."""

import pytest

from src.v9_swarm.rebuild_candidates import _feature_cache_paths


PARTS = ("features_v1", "control_v1", "plus_v1", "physv2")
KEY = "sample_1_raw"
FINGERPRINT = "b" * 24
OTHER_FINGERPRINT = "a" * 24


def make_cache(root, part, key, fingerprint):
    path = root / part / f"{key}_{fingerprint}.parquet"
    path.parent.mkdir(exist_ok=True)
    path.touch()
    return path


def test_find_one_complete_set(tmp_path):
    expected = tuple(make_cache(tmp_path, part, KEY, FINGERPRINT) for part in PARTS)
    for part in PARTS:
        make_cache(tmp_path, part, f"{KEY}_extra", OTHER_FINGERPRINT)
    make_cache(tmp_path, PARTS[0], KEY, OTHER_FINGERPRINT)

    assert _feature_cache_paths(tmp_path, KEY) == expected


def test_reject_mixed_fingerprints(tmp_path):
    for part in PARTS[:-1]:
        make_cache(tmp_path, part, KEY, FINGERPRINT)
    make_cache(tmp_path, PARTS[-1], KEY, OTHER_FINGERPRINT)

    with pytest.raises(FileNotFoundError, match=f"No common feature cache fingerprint for {KEY}"):
        _feature_cache_paths(tmp_path, KEY)


def test_reject_multiple_complete_sets(tmp_path):
    for part in PARTS:
        make_cache(tmp_path, part, KEY, FINGERPRINT)
        make_cache(tmp_path, part, KEY, OTHER_FINGERPRINT)

    with pytest.raises(ValueError, match=f"Ambiguous feature caches for {KEY}"):
        _feature_cache_paths(tmp_path, KEY)
