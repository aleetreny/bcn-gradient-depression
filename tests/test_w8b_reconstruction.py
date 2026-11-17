import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.common import AWAY, HOME
from src.v9_swarm import rebuild_w8b_candidates as w8b
from src.v9_swarm.rebuild_sportec_cache import COLUMNS


@pytest.mark.parametrize(
    "kind, expected",
    [("mode1", "XYX"), ("mode2", "INS_Y"), ("boundaries", "HA")],
)
def test_w8b_candidates_from_public_fold(tmp_path, monkeypatch, kind, expected):
    pred = np.full(170, HOME)
    pred[30:55] = AWAY
    proba = np.full((len(pred), 3), 0.05, dtype=np.float32)
    proba[np.arange(len(pred)), pred] = 0.9
    proba[85:98, AWAY] = 0.4
    half = np.ones_like(pred)
    tracking = pd.DataFrame({column: np.ones(len(pred), dtype=np.float32) for column in COLUMNS})
    match = {
        "hl": half,
        "gt": pred.copy(),
        "xv3": tracking,
        "xv6": tracking,
        "p4d_v3": np.zeros((len(pred), 4), dtype=np.float32),
        "p4d_v6": np.zeros((len(pred), 4), dtype=np.float32),
        "w7b": {},
    }
    for track in w8b.TRACKS:
        match[f"fams_{track}"] = [proba.copy() for _ in range(9)]
        match["w7b"][f"pred_{track}_rerank"] = pred
        match["w7b"][f"proba_{track}"] = proba
        match["w7b"][f"pchange_{track}"] = np.zeros(len(pred), dtype=np.float32)

    match_id = "SYN"
    monkeypatch.setattr(w8b, "MATCH_IDS", (match_id,))
    outputs = tmp_path / "experiments/outputs/exp_w8b_oof"
    outputs.mkdir(parents=True)
    source = outputs / "sportec_fast_cache.pkl"
    with source.open("wb") as input_file:
        pickle.dump({match_id: match}, input_file)
    output = outputs / f"{kind}.pkl"
    assert w8b.rebuild_w8b_candidates(tmp_path, output, kind) == output
    with output.open("rb") as candidate_file:
        result = pickle.load(candidate_file)

    key = "btype" if kind == "boundaries" else "pat"
    assert tuple(result) == w8b.TRACKS
    for track in w8b.TRACKS:
        assert [entry[key] for entry in result[track][match_id]]
        assert expected in {entry[key] for entry in result[track][match_id]}
    with pytest.raises(FileExistsError):
        w8b.rebuild_w8b_candidates(tmp_path, output, kind)


def test_w8b_rejects_non_sportec_input():
    with pytest.raises(ValueError, match="public Sportec"):
        w8b.extract_sportec_candidates({"unknown": None}, "mode1")
