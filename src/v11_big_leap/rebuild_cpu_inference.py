"""Run the existing local neural inference reconstructors on CPU.

Inputs, output restrictions and hash checks are those of the underlying
reconstructors. This does not establish cross-machine determinism.
"""
from unittest.mock import patch

import torch


def rebuild_pset_cpu(root, output_path):
    from src.v9_swarm.player_set_net_v2 import SymPlayerSetNetV2
    from src.v9_swarm.rebuild_pset_inference import rebuild_pset_inference

    with patch.object(SymPlayerSetNetV2, "_device", lambda self: torch.device("cpu")):
        return rebuild_pset_inference(root, output_path)


def rebuild_trajectory_cpu(root, output_path):
    from src.v10_first_principles.traj_event_net_v10 import SymTrajectoryEventNetV10
    from src.v10_first_principles.rebuild_trajectory_inference import rebuild_trajectory_inference

    with patch.object(SymTrajectoryEventNetV10, "_device", lambda self: torch.device("cpu")):
        return rebuild_trajectory_inference(root, output_path)


def rebuild_phys_cpu(root, output_path):
    from src.v11_big_leap.phys_conformer_v11 import SymPhysConformerV11
    from src.v11_big_leap.rebuild_phys_inference import rebuild_phys_inference

    original = SymPhysConformerV11.predict_all_heads

    def predict_on_cpu(self, *args, **kwargs):
        self.device = torch.device("cpu")
        return original(self, *args, **kwargs)

    with patch.object(SymPhysConformerV11, "predict_all_heads", predict_on_cpu):
        return rebuild_phys_inference(root, output_path)
