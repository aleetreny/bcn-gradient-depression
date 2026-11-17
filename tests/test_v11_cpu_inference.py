"""CPU adapters must control device selection and restore the model classes."""
import pytest
import torch

from src.v11_big_leap import rebuild_cpu_inference as cpu


@pytest.mark.parametrize("family", ["pset", "trajectory"])
def test_cpu_device_is_scoped_to_reconstruction(monkeypatch, family):
    if family == "pset":
        from src.v9_swarm import player_set_net_v2 as model_module
        from src.v9_swarm import rebuild_pset_inference as inference
        cls = model_module.SymPlayerSetNetV2
        name = "rebuild_pset_inference"
        reconstruct = cpu.rebuild_pset_cpu
    else:
        from src.v10_first_principles import traj_event_net_v10 as model_module
        from src.v10_first_principles import rebuild_trajectory_inference as inference
        cls = model_module.SymTrajectoryEventNetV10
        name = "rebuild_trajectory_inference"
        reconstruct = cpu.rebuild_trajectory_cpu

    original = cls._device

    def fake_rebuild(root, output):
        assert cls.__new__(cls)._device() == torch.device("cpu")
        raise RuntimeError("verify restoration after failure")

    monkeypatch.setattr(inference, name, fake_rebuild)
    with pytest.raises(RuntimeError, match="restoration"):
        reconstruct(None, None)
    assert cls._device is original


def test_phys_cpu_device_is_scoped_to_reconstruction(monkeypatch):
    from src.v11_big_leap import phys_conformer_v11 as model_module
    from src.v11_big_leap import rebuild_phys_inference as inference
    cls = model_module.SymPhysConformerV11

    def fake_predict(self, *args, **kwargs):
        assert self.device == torch.device("cpu")
        return "cpu"

    monkeypatch.setattr(cls, "predict_all_heads", fake_predict)

    def fake_rebuild(root, output):
        model = cls.__new__(cls)
        model.device = torch.device("mps")
        assert model.predict_all_heads() == "cpu"
        raise RuntimeError("verify restoration after failure")

    monkeypatch.setattr(inference, "rebuild_phys_inference", fake_rebuild)
    with pytest.raises(RuntimeError, match="restoration"):
        cpu.rebuild_phys_cpu(None, None)
    assert cls.predict_all_heads is fake_predict
