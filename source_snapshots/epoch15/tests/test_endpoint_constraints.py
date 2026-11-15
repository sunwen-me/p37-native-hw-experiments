import torch

from p37_model import OrderedDenseFieldFlow, endpoint_fields


def test_rectangular_endpoints_remain_fixed_at_every_sde_integration_step():
    height, width, steps = 9, 17, 4
    model = OrderedDenseFieldFlow(hidden_dim=8, latent_dim=4, horizon=8)
    maps = torch.zeros(1, 2, height, width)
    starts = torch.tensor([[-1.0, -1.0]])
    goals = torch.tensor([[1.0, 1.0]])
    latent = torch.zeros(1, 4)
    source = torch.randn(1, 2, height, width)
    schedule = torch.randn(steps, 1, 2, height, width)
    fixed = endpoint_fields(starts, goals, height, width)
    endpoint = fixed[:, :1].bool()
    seen = []

    def capture_state(_module, inputs):
        seen.append(inputs[0].detach().clone())

    handle = model.vector_field.register_forward_pre_hook(capture_state)
    result = model._integrate(
        maps, starts, goals, latent, num_steps=steps, noise_scale=1.0,
        source=source, sde_noise=0.2, sde_noise_schedule=schedule,
        generator=torch.Generator().manual_seed(23),
    )
    handle.remove()

    assert len(seen) == steps
    for state in [*seen, result]:
        torch.testing.assert_close(state * endpoint, fixed * endpoint)
    assert result.shape == (1, 2, height, width)
    assert torch.isfinite(result).all()
