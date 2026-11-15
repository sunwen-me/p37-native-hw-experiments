import torch

from p37_model import OrderedDenseFieldFlow, endpoint_fields


def test_vector_field_and_integrator_accept_rectangular_grids():
    torch.manual_seed(7)
    model = OrderedDenseFieldFlow(hidden_dim=8, latent_dim=4, horizon=6)
    for height, width in ((16, 20), (20, 12)):
        maps = torch.rand(2, 2, height, width)
        starts = torch.tensor([[-1.0, -1.0], [0.0, -0.5]])
        goals = torch.tensor([[1.0, 1.0], [0.5, 1.0]])
        latent = torch.zeros(2, 4)
        state = torch.zeros(2, 2, height, width)
        velocity = model.vector_field(
            state, maps, starts, goals, torch.zeros(2), latent,
        )
        assert velocity.shape == state.shape
        fixed = endpoint_fields(starts, goals, height, width)
        integrated = model._integrate(
            maps, starts, goals, latent, num_steps=1, noise_scale=0.0,
            source=torch.zeros_like(state),
        )
        assert integrated.shape == state.shape
        endpoint = fixed[:, :1].bool()
        torch.testing.assert_close(integrated * endpoint, fixed * endpoint)


def test_required_resolutions_run_one_flow_step_with_finite_shapes():
    torch.manual_seed(17)
    model = OrderedDenseFieldFlow(hidden_dim=8, latent_dim=4, horizon=6)
    sizes = (
        (64, 64), (128, 128), (256, 256), (512, 512),
        (128, 256), (256, 512), (512, 768), (127, 193), (255, 511),
    )
    for height, width in sizes:
        maps = torch.zeros(1, 2, height, width)
        start = torch.tensor([[-1.0, -1.0]])
        goal = torch.tensor([[1.0, 1.0]])
        result = model._integrate(
            maps, start, goal, torch.zeros(1, 4), num_steps=1,
            noise_scale=0.0, source=torch.zeros(1, 2, height, width),
        )
        assert result.shape == (1, 2, height, width)
        assert torch.isfinite(result).all()
        fixed = endpoint_fields(start, goal, height, width)
        endpoint = fixed[:, :1].bool()
        torch.testing.assert_close(result * endpoint, fixed * endpoint)
