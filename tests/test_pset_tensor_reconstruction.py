"""Public tensor reconstruction respects realization seeds and output safety."""
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.models import player_set_net
from src.v9_swarm import rebuild_pset_tensor as reconstruction


@pytest.mark.parametrize(('tag', 'seed'), [('v3_s0', 0), ('v6_s1', 11)])
def test_reconstructs_requested_realization(tmp_path, monkeypatch, tag, seed):
    @dataclass
    class Config:
        static_hide_prob: float = 0.0

    monkeypatch.setattr(reconstruction, '__file__', str(tmp_path / 'src/v9_swarm/module.py'))
    monkeypatch.setattr(reconstruction, 'MATCH_IDS', ('public',))
    monkeypatch.setattr(reconstruction, '_load_degrade_configs', lambda root: {'v3': Config(), 'v6': Config()})
    match = SimpleNamespace(half=np.ones(3))
    monkeypatch.setattr(reconstruction, 'load', lambda *args, **kwargs: match)
    seen = []
    monkeypatch.setattr(reconstruction, 'degrade', lambda m, cfg, s: seen.append(s) or m)
    monkeypatch.setattr(reconstruction, 'match_fingerprint', lambda m: 'fingerprint')
    fields = ('p_h', 'p_a', 'mask_h', 'mask_a', 'g_anti', 'g_sym')
    monkeypatch.setattr(player_set_net, 'extract_player_set_tensors',
                        lambda m: SimpleNamespace(**{key: np.ones((3, 1)) for key in fields}))
    local = tmp_path / '_local'
    local.mkdir()
    (local / 'pset_tensor_request.json').write_text(json.dumps({'match_id': 'public', 'tag': tag}))
    output = local / f'pset_v{player_set_net.VERSION}_sportec_public_{tag}_fingerprint.npz'
    reconstruction.rebuild_pset_tensor(tmp_path, output)
    assert seen == [seed]
    with np.load(output) as data:
        assert set(data.files) == set(fields)
    with pytest.raises(FileExistsError):
        reconstruction.rebuild_pset_tensor(tmp_path, output)


def test_refuses_nonlocal_output(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, '__file__', str(tmp_path / 'src/v9_swarm/module.py'))
    with pytest.raises(ValueError, match='under _local'):
        reconstruction.rebuild_pset_tensor(tmp_path, tmp_path / 'data_external' / 'tensor.npz')
