from types import SimpleNamespace

import numpy as np
import pytest

from src.v9_swarm import rebuild_pset_fold as pset


def test_pset_fold_excludes_target_and_packs_historical_heads(tmp_path, monkeypatch):
    split = pset.MATCH_IDS[2]
    recorded_seeds = []
    predictions = []

    def load(mid, *, data_dir, use_cache):
        assert data_dir == tmp_path / "data_external" / "sportec_dfl"
        assert not use_cache
        return SimpleNamespace(half=np.array([1, 1]), labels=mid, meta={"ball_z": np.zeros(2)})

    def load_test_tensors(cache_dir, mid, tag, match, config, seed):
        assert cache_dir == tmp_path / "local_data/cache/pset"
        assert len(match.half) == 2
        assert seed == {"v3_s1": 1, "v3_s2": 2, "v3_static05_s1": 1,
                        "v6_s1": 11, "v3_s0": 0, "v6_s0": 0}[tag]
        assert config.smooth_window == (3 if tag.startswith("v3") else 6)
        assert config.static_hide_prob == (0.5 if tag == "v3_static05_s1" else 0.7)
        return mid, tag

    expected_fits = [(mid, tag) for mid in pset.MATCH_IDS if mid != split
                     for tag in ("v3_s1", "v3_s2", "v3_static05_s1", "v6_s1")]

    class FakeModel:
        def __init__(self, seed):
            recorded_seeds.append(seed)
            self.value = 2.0 if seed < 100 else 4.0

        def fit(self, tensors, labels, halves, *, ball_zs, sample_weights):
            assert tensors == expected_fits
            assert labels == [mid for mid, _ in expected_fits]
            assert len(halves) == len(ball_zs) == len(expected_fits)
            assert sample_weights == [1.0, 1.0, 1.0, 0.8] * 6
            return self

        def predict_all_heads(self, tensors, half):
            predictions.append(tensors)
            assert len(half) == 2
            return (np.tile([self.value, 2 * self.value, self.value], (2, 1)),
                    *(np.full(2, self.value) for _ in range(4)))

    monkeypatch.setattr(pset, "_load_degrade_configs", lambda root: {
        f"v{version}": pset.DegradeConfig(smooth_window=version, static_hide_prob=0.7)
        for version in (3, 6)
    })
    monkeypatch.setattr(pset, "load", load)
    monkeypatch.setattr(pset, "_load_tensors", load_test_tensors)
    monkeypatch.setattr(pset, "_new_model", FakeModel)
    output = tmp_path / f"_local/pset_v2_{split}.npz"
    assert pset.rebuild_pset_fold(tmp_path, output) == output
    assert recorded_seeds == [44, 144]
    assert predictions == [(split, "v3_s0"), (split, "v6_s0")] * 2
    with np.load(output) as cache:
        assert set(cache.files) == {f"{tag}_{head}" for tag in ("v3", "v6") for head in pset.HEADS}
        for tag in ("v3", "v6"):
            np.testing.assert_array_equal(cache[f"{tag}_proba"], [[0.25, 0.5, 0.25]] * 2)
            for head in pset.HEADS[1:]:
                assert cache[f"{tag}_{head}"].dtype == np.float32
                np.testing.assert_array_equal(cache[f"{tag}_{head}"], [3.0, 3.0])

    with pytest.raises(FileExistsError):
        pset.rebuild_pset_fold(tmp_path, output)
    with pytest.raises(ValueError, match="Sportec match ID"):
        pset.rebuild_pset_fold(tmp_path, tmp_path / "_local/pset_v2_ATD.npz")
    assert recorded_seeds == [44, 144]



def test_pset_rejects_cache_with_wrong_tracking_fingerprint(tmp_path, monkeypatch):
    from src.models.player_set_net import VERSION

    raw = SimpleNamespace(half=np.array([1, 1]))
    degraded = object()
    monkeypatch.setattr(pset, "degrade", lambda match, cfg, seed: degraded)

    def fingerprint(match):
        assert match is degraded
        return "a" * 24

    monkeypatch.setattr(pset, "match_fingerprint", fingerprint)
    prefix = f"pset_v{VERSION}_sportec_J03WMX_v3_s0_"
    arrays = {key: np.zeros(2) for key in ("p_h", "p_a", "mask_h", "mask_a", "g_anti", "g_sym")}
    np.savez(tmp_path / f"{prefix}{'b' * 24}.npz", **arrays)
    with pytest.raises(FileNotFoundError):
        pset._load_tensors(tmp_path, "J03WMX", "v3_s0", raw, pset.DegradeConfig(), 0)

    arrays["p_h"][:] = 1
    np.savez(tmp_path / f"{prefix}{'a' * 24}.npz", **arrays)
    tensors = pset._load_tensors(tmp_path, "J03WMX", "v3_s0", raw, pset.DegradeConfig(), 0)
    np.testing.assert_array_equal(tensors.p_h, [1, 1])
