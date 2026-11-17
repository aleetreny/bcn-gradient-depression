"""Tests for the harness's experiment layer on synthetic data (CI never sees competition or external data).

Needs the research code (src/); on branches that only carry the harness (main) the module is skipped.
The safety layer is tested in tests/test_harness_safety.py, which runs everywhere.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("src.data.common", reason="research code (src/) not on this branch")

from harness import gate, registry, runner, spec  # noqa: E402
from harness.contracts import ContractError, check_labels, check_proba, swap_labels, swap_proba  # noqa: E402
from harness.synthetic import make_match  # noqa: E402

DECODER_PARAMS = {"argmax": {}, "smooth": {"window": 12}, "hsmm": {"config": "configs/decoder_v1.json"}}


def _noisy_proba(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    y = make_match(seed, minutes=1).labels[:n]
    p = rng.dirichlet(np.ones(3), n) * 0.5
    p[np.arange(n), y] += 0.5
    return p / p.sum(1, keepdims=True), np.where(np.arange(n) < n // 2, 1, 2).astype(np.int8)


# -- contracts -------------------------------------------------------------------------------------

def test_contracts_reject_bad_outputs():
    with pytest.raises(ContractError):
        check_labels(np.array([0, 1, 3]), 3)
    with pytest.raises(ContractError):
        check_labels(np.array([0, 1]), 3)
    with pytest.raises(ContractError):
        check_proba(np.full((4, 3), 0.5), 4)
    assert check_labels(np.array([0, 1, 2]), 3).dtype == np.int8


@pytest.mark.parametrize("name", registry.names("decoder"))
def test_every_decoder_honours_the_contract(name):
    """Runs automatically for every registered decoder: shape, codes, determinism, H<->A symmetry."""
    params = DECODER_PARAMS.get(name)
    if params is None:
        pytest.skip("add {!r} to DECODER_PARAMS to test it".format(name))
    dec = registry.get("decoder", name)
    p, half = _noisy_proba()
    a = check_labels(dec(p, half, params), len(p), name)
    assert np.array_equal(a, dec(p, half, params)), "not deterministic"
    assert np.array_equal(swap_labels(a), dec(swap_proba(p), half, params)), "H<->A relabelling changes the output"


def test_registry_rejects_duplicates_and_unknown_names():
    with pytest.raises(KeyError):
        registry.register("decoder", "argmax")(lambda *a: None)
    with pytest.raises(KeyError):
        registry.get("decoder", "does_not_exist")


# -- runner, end to end on J0 ----------------------------------------------------------------------

def _spec(tmp_path, sid, predictor, params, champion="none"):
    path = tmp_path / "{}.toml".format(sid)
    path.write_text('id = "{}"\nhypothesis = "test"\n[predictor]\nname = "{}"\nparams = {}\n'
                    '[judges]\nuse = ["J0"]\nJ0 = {{ n_matches = 3, minutes = 1.0 }}\n'
                    '[criteria]\nchampion = "{}"\n'.format(sid, predictor, params, champion))
    return spec.load(path)


def test_runner_end_to_end_and_gate_against_stored_champion(tmp_path):
    quiet = lambda *a: None  # noqa: E731
    base = runner.run(_spec(tmp_path, "base", "baseline", "{ window = 25 }"), runs_dir=tmp_path, log=quiet)
    j0 = base["judges"]["J0"]
    assert j0["status"] == "ok" and set(j0["folds"]) == {"syn0", "syn1", "syn2"}
    assert 0 < j0["summary"]["spell_f1_mean"] <= 1
    assert json.loads((tmp_path / "base" / "result.json").read_text())["id"] == "base"
    assert "decision" not in base                            # champion = "none"

    cand = runner.run(_spec(tmp_path, "cand", "oracle_noise", "{ flip = 0.05 }", champion="base"),
                      runs_dir=tmp_path, log=quiet)
    assert cand["verdicts"]["J0"]["n_folds"] == 3
    assert cand["decision"] in (gate.ACCEPT, gate.NOISE, gate.REJECT)


def test_spec_file_name_must_match_id(tmp_path):
    path = tmp_path / "other.toml"
    path.write_text('id = "mine"\nhypothesis = "x"\n[predictor]\nname = "baseline"\n')
    with pytest.raises(ValueError):
        spec.load(path)


def test_oracle_predictor_refuses_real_data():
    fp = registry.get("predictor", "oracle_noise")()
    m = make_match(0, minutes=0.5)
    from dataclasses import replace
    with pytest.raises(RuntimeError):
        fp({}, replace(m, source="sportec"))
