import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.v9_swarm import rebuild_sportec_cache as sportec


def test_rebuild_sportec_cache_preserves_family_order_and_output(tmp_path, monkeypatch):
    match = SimpleNamespace(labels=np.array([0, 2]), half=np.array([1, 1]))
    monkeypatch.setattr(sportec, "__file__", tmp_path / "src/v9_swarm/rebuild_sportec_cache.py")
    monkeypatch.setattr(sportec, "load_all", lambda: {"SYN": match})
    monkeypatch.setattr(sportec, "_load_features", lambda cache, prefix, half: prefix)

    folds = {
        "exp_c_plus_oof/plus": {"symgbm_plus_tta": 1},
        "exp_w4_oof/physv2": {"symgbm_physv2_tta": 2, "symgbm_physv2_tta_on_v6": 12},
        "exp_w5_tree_oof/cb_xgb": {
            "symcatboost_physv2_tta": 3, "symxgb_physv2_tta": 4,
            "symcatboost_physv2_tta_on_v6": 13, "symxgb_physv2_tta_on_v6": 14,
        },
        "exp_b_sym_oof/tcn_sym": {"symtcn_tta_proba": 5},
        "exp_h_oof/bigru_sym": {"symbigru_tta_proba": 6},
        "exp_w5b_oof/tcn_physv2": {"v3_proba": 7, "v6_proba": 17},
        "exp_w5b_oof/bigru_physv2": {"v3_proba": 8, "v6_proba": 18},
        "exp_w7a_oof/pset": {"v3_proba": 9, "v6_proba": 19},
        "exp_w7b_oof/v6_genuine": {"symgbm_plus_tta_on_v6": 11},
        "exp_g_oof/ghost_sym": {"g_v6_on_v6": 20},
        "exp_w3_oof/b4d": {"p4d_v3": 21, "p4d_v6": 22},
        "exp_w7b_oof/w7b": {"w7b": 23},
    }
    for prefix, arrays in folds.items():
        path = tmp_path / "experiments/outputs" / f"{prefix}_SYN.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{key: np.array([value, value], dtype=np.float32)
                          for key, value in arrays.items()})

    output = tmp_path / "experiments/outputs/sportec_fast_cache.pkl"
    assert sportec.rebuild_sportec_cache(tmp_path, output) == output
    with output.open("rb") as cache_file:
        result = pickle.load(cache_file)["SYN"]
    assert [arr[0] for arr in result["fams_v3"]] == list(range(1, 10))
    assert [arr[0] for arr in result["fams_v6_std"]] == [11, 12, 13, 14, 5, 6, 17, 18, 19]
    assert [arr[0] for arr in result["fams_v6_gen"]] == [11, 12, 13, 14, 17, 18, 20, 19]
    assert result["p4d_v3"].dtype == np.float32
    assert result["p4d_v3"][0] == 21
    assert result["p4d_v6"][0] == 22
    assert result["w7b"]["w7b"][0] == 23
    np.testing.assert_array_equal(result["gt"], match.labels)
    np.testing.assert_array_equal(result["hl"], match.half)

    with pytest.raises(FileExistsError):
        sportec.rebuild_sportec_cache(tmp_path, output)
