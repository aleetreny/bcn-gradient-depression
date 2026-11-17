"""External cached predictions require the right fold and input fingerprint."""
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval import cached_external_report as report


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sportec_fold_checks_exclusion_and_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report, 'MATCH_IDS', ('a', 'b'))
    monkeypatch.setattr(report, 'match_fingerprint', lambda match: match.fingerprint)
    local = tmp_path / '_local'
    local.mkdir()
    manifest = local / 'external_fingerprints.json'
    manifest.write_text(json.dumps({'sportec_a_degrade_v3_s0': 'expected'}))
    bundle = local / report.FILES['original']
    bundle.write_bytes(pickle.dumps({'d3': {'preds_v3': {'a': np.array([0, 1, 2])}}}))
    predict = report.external_bundle_predictor('original', _sha(bundle), _sha(manifest))
    match = SimpleNamespace(source='sportec', meta={'match_id': 'a'}, fingerprint='expected', n_frames=3)
    np.testing.assert_array_equal(predict({'b_s1': None}, match), [0, 1, 2])
    with pytest.raises(ValueError, match='exclude'):
        predict({'a_s1': None}, match)
    match.fingerprint = 'other'
    with pytest.raises(ValueError, match='fingerprint'):
        predict({'b_s1': None}, match)
