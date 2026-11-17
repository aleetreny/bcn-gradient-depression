"""TCN: with norm="channel" the output only depends on frames inside the receptive field."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_channel_norm_is_local_and_group_norm_is_not():
    from src.models.tcn import _net
    torch.manual_seed(0)
    x = torch.randn(1, 6, 3000)
    far = x.clone()
    far[..., :900] = 0.0                       # > 511 frames before the checked window
    far[..., 2100:] = 0.0
    diffs = {}
    for norm in ("group", "channel"):
        net = _net(6, 16, 0.0, norm=norm).eval()
        with torch.no_grad():
            diffs[norm] = float((net(x)[..., 1450:1550] - net(far)[..., 1450:1550]).abs().max())
    assert diffs["channel"] == 0.0 and diffs["group"] > 1e-3
