"""J3 CSV reporting rejects unrelated matches and changed inputs."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval import public_csv_report as report


def test_csv_predictor_checks_match_and_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    local = tmp_path / '_local'
    local.mkdir()
    path = local / report.FILES['original']
    path.write_text('frame,label\n0,H\n1,A\n2,D\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    predict = report.public_csv_predictor('original', digest)
    match = SimpleNamespace(source='metrica_atd', meta={'game': 'BAR-ATH'}, n_frames=3)
    np.testing.assert_array_equal(predict({}, match), [0, 1, 2])
    with pytest.raises(ValueError, match='only accepts'):
        predict({}, SimpleNamespace(source='other', meta={}, n_frames=3))
    path.write_text('frame,label\n0,A\n1,A\n2,D\n')
    with pytest.raises(ValueError, match='registered hash'):
        predict({}, match)
