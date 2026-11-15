import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "training/source/map_flow_matching_p1/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import train_p37_ordered_field as p37_training


@pytest.mark.parametrize("architecture", ("cnn", "transformer"))
@pytest.mark.parametrize("rasterization,reduction", [
    ("points", "pixel_mean"), ("segments", "support_balanced"),
])
def test_pairwise_cfm_loss_and_gradient_support_odd_rectangular_maps(architecture, rasterization, reduction):
    height, width, horizon = 13, 19, 8
    model = p37_training.OrderedDenseFieldFlow(
        map_height=height, map_width=width, hidden_dim=8, latent_dim=4,
        horizon=(32 if rasterization == "segments" else horizon),
        architecture=architecture, transformer_window_size=4,
    )
    maps = torch.zeros(2, 2, height, width)
    maps[:, 1] = 1.0
    starts = torch.tensor([[-1.0, -1.0], [-0.5, 0.0]])
    goals = torch.tensor([[1.0, 1.0], [0.5, 1.0]])
    progress = torch.linspace(0.0, 1.0, horizon)[:, None]
    routes = torch.stack([
        starts[index] + progress * (goals[index] - starts[index])
        for index in range(len(starts))
    ])
    routes = routes[:, None].expand(-1, 3, -1, -1).contiguous()
    latent = torch.randn(2, 2, model.latent_dim)

    loss = p37_training.pairwise_cfm_cost(
        model, maps, starts, goals, routes, latent, mask_thickness=1,
        endpoint_approach_weight=0.5,
        rasterization_mode=rasterization, field_loss_reduction=reduction,
        generator=torch.Generator().manual_seed(123),
    )

    assert loss.shape == (2, 2, 3)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.vector_field.out.weight.grad is not None
    assert torch.isfinite(model.vector_field.out.weight.grad).all()
