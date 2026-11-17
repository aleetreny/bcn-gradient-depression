"""Checks for L1 inference symmetry and local-output protection."""
import numpy as np
import pytest

from src.v9_swarm import rebuild_l1_inference as reconstruction


class Classifier:
    def __init__(self, column):
        self.column = column

    def predict_proba(self, x):
        p = x[:, self.column]
        return np.column_stack((1 - p, p))


def test_twohead_and_binary_average_mirrored_predictions():
    x = np.array([[0.8, 0.7], [0.2, 0.3]], dtype=np.float32)
    xm = np.array([[0.6, 0.4], [0.4, 0.8]], dtype=np.float32)
    result = reconstruction._twohead_tta((Classifier(0), Classifier(1)), x, xm)
    a = np.column_stack((x[:, 0] * x[:, 1], x[:, 0] * (1 - x[:, 1]), 1 - x[:, 0]))
    b = np.column_stack((xm[:, 0] * (1 - xm[:, 1]), xm[:, 0] * xm[:, 1], 1 - xm[:, 0]))
    np.testing.assert_allclose(result, (a + b) / 2, atol=1e-7)
    np.testing.assert_allclose(reconstruction._binary_tta(Classifier(0), x, xm),
                               (x[:, 0] + xm[:, 0]) / 2)


def test_reconstruction_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(reconstruction, "__file__", str(tmp_path / "src/v9_swarm/module.py"))
    local = tmp_path / "_local"
    local.mkdir()
    output = local / "predictions.pkl"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="new"):
        reconstruction.rebuild_l1_inference(tmp_path, output)
    assert output.read_bytes() == b"keep"
