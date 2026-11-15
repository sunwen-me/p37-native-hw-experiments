import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/source/map_flow_matching_p1/scripts"))
from route_progress_objective import arc_length_samples, route_progress_increment_cost
import train_p37_ordered_field as training


def test_rectangular_arc_samples_use_pixel_length_and_preserve_polyline_turns():
    # On H=11,W=21 this route travels 20px right, then 10px up.
    routes = torch.tensor([[[-1., -1.], [1., -1.], [1., 1.], [1., 1.]]])
    points, phase, valid = arc_length_samples(routes, 11, 21, 7)
    expected = torch.tensor([[[-1., -1.], [-.5, -1.], [0., -1.], [.5, -1.],
                              [1., -1.], [1., 0.], [1., 1.]]])
    assert torch.allclose(points, expected, atol=1e-6)
    assert valid.all() and torch.allclose(phase, torch.linspace(0, 1, 7))


def test_correct_terminal_progress_has_zero_cost_and_reversal_has_gradients():
    field = torch.zeros(2, 2, 9, 65)
    field[:, 1] = torch.linspace(0, 1, 65)[None]
    routes = torch.tensor([[[-1., 0.], [1., 0.]], [[1., 0.], [-1., 0.]]])
    field.requires_grad_()
    costs = route_progress_increment_cost(field, routes, samples=33)
    assert costs[0] < 1e-10 and costs[1] > .1
    costs.sum().backward()
    assert torch.isfinite(field.grad).all() and field.grad[:, 1].abs().sum() > 0
    assert not field.grad[:, 0].any()


def test_auxiliary_disabled_is_bitwise_compatible_and_enabled_reaches_network():
    model = training.OrderedDenseFieldFlow(hidden_dim=8, latent_dim=4, architecture="transformer", horizon=32)
    maps = torch.zeros(1, 2, 13, 19)
    maps[:, 1] = 1
    starts, goals = torch.tensor([[-.7, -.5]]), torch.tensor([[.6, .5]])
    routes = torch.stack([starts, goals], 1)[:, None]
    latent = torch.zeros(1, 2, 4)
    def cost(**kwargs):
        return training.pairwise_cfm_cost(model, maps, starts, goals, routes, latent, 1,
            rasterization_mode="segments", field_loss_reduction="support_balanced",
            generator=torch.Generator().manual_seed(37), **kwargs)
    old = cost()
    disabled = cost(route_increment_weight=0)
    enabled = cost(route_increment_weight=.1)
    assert torch.equal(old, disabled) and (enabled >= old).all()
    (enabled - disabled).mean().backward()
    gradient = model.vector_field.out.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
