from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.v9_swarm import rebuild_v6_genuine as reconstruction


def test_v6_genuine_writes_compatible_npz(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    target = reconstruction.MATCH_IDS[1]
    output = Path("_local") / f"v6_genuine_{target}.npz"
    proba = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)
    change = np.array([0.8], dtype=np.float32)

    def fit(root, held_out):
        assert root == tmp_path
        assert held_out == target
        return proba, change

    monkeypatch.setattr(reconstruction, "_fit_fold", fit)
    result = reconstruction.rebuild_v6_genuine(tmp_path, output)
    with np.load(result) as archive:
        assert set(archive.files) == {"symgbm_plus_tta_on_v6", "symtrans_plus_tta_on_v6"}
        np.testing.assert_array_equal(archive["symgbm_plus_tta_on_v6"], proba)
        np.testing.assert_array_equal(archive["symtrans_plus_tta_on_v6"], change)

    with pytest.raises(FileExistsError):
        reconstruction.rebuild_v6_genuine(tmp_path, output)


@pytest.mark.parametrize(
    "output",
    ("../v6_genuine_J03WN1.npz", "submissions/v6_genuine_J03WN1.npz", "_local/unknown.npz"),
)
def test_v6_genuine_checks_destination_before_training(tmp_path, monkeypatch, output):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    monkeypatch.setattr(reconstruction, "_fit_fold", lambda *args: pytest.fail("trained"))
    with pytest.raises(ValueError):
        reconstruction.rebuild_v6_genuine(tmp_path, Path(output))


def test_v6_genuine_uses_held_out_match_and_historical_fold_seed(tmp_path, monkeypatch):
    matches = ("first", "held_out", "last")
    monkeypatch.setattr(reconstruction, "MATCH_IDS", matches)
    monkeypatch.setattr(
        reconstruction, "load_all",
        lambda: {
            mid: SimpleNamespace(mid=mid, half="half", pitch="pitch", labels=mid)
            for mid in matches
        },
    )

    @dataclass
    class Config:
        tag: str
        static_hide_prob: float = 0.0

    monkeypatch.setitem(sys.modules, "s2_1_degradation", SimpleNamespace(
        load_config=lambda path: Config(path.stem)
    ))
    monkeypatch.setitem(sys.modules, "s3_7_nested_full", SimpleNamespace(
        TRAIN_VARIANTS=(("s1", None, 1), ("static05_s1", {"static_hide_prob": 0.5}, 1))
    ))
    monkeypatch.setitem(sys.modules, "exp_c_plus_eval", SimpleNamespace(
        predict_gbm_plus_tta=lambda model, table, half, pitch: table,
        predict_trans_plus_tta=lambda model, table, half, pitch: table,
    ))
    monkeypatch.setattr(reconstruction, "degrade", lambda m, c, s: (m.mid, c.tag, c.static_hide_prob, s))
    monkeypatch.setattr(reconstruction, "build_v1ctl_plus", lambda data, key: (*data, key))
    monkeypatch.setattr(reconstruction, "mirror_v1ctl_plus_df", lambda table, half, pitch: (*table, "mirror"))
    monkeypatch.setattr(reconstruction, "swap_labels", lambda labels: f"{labels}:mirror")
    monkeypatch.setattr(reconstruction, "change_targets", lambda labels, half: f"{labels}:change")

    class GBM:
        def __init__(self, seed):
            assert seed == 1

        def fit(self, train, labels, groups):
            assert [(r[0], r[1], r[2], r[3]) for r in train[:4]] == [
                ("first", "degrade_v3", 0.0, 1),
                ("first", "degrade_v3", 0.5, 1),
                ("last", "degrade_v3", 0.0, 1),
                ("last", "degrade_v3", 0.5, 1),
            ]
            assert [(*row, "mirror") for row in train[:4]] == train[4:]
            assert labels == [
                "first", "first", "last", "last",
                "first:mirror", "first:mirror", "last:mirror", "last:mirror",
            ]
            assert groups == [0, 0, 1, 1, 2, 2, 3, 3]
            return self

    class Transition:
        def __init__(self, seed):
            assert seed == 1

        def fit(self, train, targets):
            assert len(train) == 8
            assert targets == ["first:change", "first:change", "last:change", "last:change"] * 2
            return self

    monkeypatch.setattr(reconstruction, "TwoHeadGBM", GBM)
    monkeypatch.setattr(reconstruction, "TransitionGBM", Transition)
    proba, change = reconstruction._fit_fold(tmp_path, "held_out")
    assert proba == change == ("held_out", "degrade_v6", 0.0, 0, "sportec_held_out_degrade_v6_s0")
