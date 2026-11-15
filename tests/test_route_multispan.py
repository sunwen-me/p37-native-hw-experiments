import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/source/map_flow_matching_p1/scripts'))
import train_p37_ordered_field as training
from route_progress_objective import route_progress_increment_cost, route_progress_multispan_cost
from route_field_diagnostics import route_field_diagnostics


def test_dense_relative_slope_correct_flat_reversed_and_density_invariance():
    routes = torch.tensor([[[-1., 0.], [1., 0.]]]).repeat(3, 1, 1)
    fields = torch.zeros(3, 2, 7, 257)
    fields[0, 1] = torch.linspace(0, 1, 257)
    fields[2, 1] = torch.linspace(1, 0, 257)
    fields.requires_grad_()
    costs = route_progress_multispan_cost(fields, routes)
    assert costs[0] < 1e-10
    assert costs[1].item() == pytest.approx(.5)
    assert costs[2].item() == pytest.approx(1.5)
    assert torch.allclose(costs, route_progress_multispan_cost(fields, routes, spacing_px=.5), atol=1e-6)
    costs.sum().backward()
    assert torch.isfinite(fields.grad).all() and fields.grad[2, 1].abs().sum() > 0
    assert not fields.grad[:, 0].any()


def test_dense_objective_follows_turns_and_masks_padding_degenerate_and_short_routes():
    # Right 20px, then down 10px. A spatial chord would cut the corner.
    route = torch.tensor([[[-1., -1.], [1., -1.], [1., 1.], [1., 1.]]])
    field = torch.full((1, 2, 11, 21), 99.)
    field[0, 1, 0] = torch.arange(21) / 30
    field[0, 1, :, 20] = (20 + torch.arange(11)) / 30
    assert route_progress_multispan_cost(field, route).item() < 1e-10
    # A mixed batch has independently masked dense lengths; cap covers all L.
    fields = torch.zeros(3, 2, 7, 257)
    fields[:, 1] = torch.linspace(0, 1, 257)
    routes = torch.tensor([[[-1., 0.], [1., 0.]], [[-1., 0.], [-.99, 0.]], [[0., 0.], [0., 0.]]])
    costs = route_progress_multispan_cost(fields, routes, max_samples=32)
    assert costs[0] < 1e-10 and 0 < costs[1] < .5 and costs[2] == 0


def test_multispan_disabled_compatibility_and_network_gradient():
    model = training.OrderedDenseFieldFlow(hidden_dim=8, latent_dim=4, architecture='transformer', horizon=32)
    maps = torch.zeros(1, 2, 13, 19)
    maps[:, 1] = 1
    starts, goals = torch.tensor([[-.7, -.5]]), torch.tensor([[.6, .5]])
    routes = torch.stack([starts, goals], 1)[:, None]
    def cost(**kwargs):
        return training.pairwise_cfm_cost(model, maps, starts, goals, routes, torch.zeros(1, 2, 4), 1,
            rasterization_mode='segments', field_loss_reduction='support_balanced',
            generator=torch.Generator().manual_seed(37), **kwargs)
    original, disabled, enabled = cost(), cost(route_multispan_weight=0), cost(route_multispan_weight=.01)
    assert torch.equal(original, disabled) and (enabled >= original).all()
    (enabled - original).mean().backward()
    gradient = model.vector_field.out.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_dense_samples_detect_reversal_between_legacy_128_sample_sites():
    fields = torch.zeros(1, 2, 3, 2049)
    fields[:, 1] = torch.linspace(0, 1, 2049)
    fields[0, 1, :, 5:12] -= torch.tensor([.002, .004, .006, .008, .006, .004, .002])
    route = torch.tensor([[[-1., 0.], [1., 0.]]])
    assert route_progress_increment_cost(fields, route).item() < 1e-10
    assert route_progress_multispan_cost(fields, route).item() > .001


def test_cumulative_metric_detects_small_gradual_dip_and_measures_support_gap_in_pixels():
    fields = np.zeros((1, 2, 3, 201), dtype=np.float32)
    fields[:, 0] = 1
    fields[0, 0, :, 40:60] = 0
    # Every negative step is only -.0002; the old -.02 test counts none.
    progress = np.r_[np.linspace(0, .3, 51), np.linspace(.2998, .29, 50), np.linspace(.291, 1, 100)]
    fields[0, 1] = progress
    row = route_field_diagnostics([[0, 1], [200, 1]], fields)['candidates'][0]
    assert np.diff(progress).min() > -.02
    assert row['max_cumulative_progress_drawdown'] == pytest.approx(.01, abs=1e-6)
    assert row['longest_below_previous_peak_px'] > 49
    assert row['low_advance_16px_fraction'] > .1
    assert row['longest_missing_support_gap_px'] == pytest.approx(19.5)
    assert row['missing_support_total_px'] == pytest.approx(19.5)


def test_continuity_metric_handles_degeneracy_repetition_and_full_gap():
    fields = np.zeros((1, 2, 5, 65), np.float32)
    fields[:, 1] = np.linspace(0, 1, 65)
    result = route_field_diagnostics([[0, 2], [0, 2], [64, 2], [64, 2]], fields)
    row = result['candidates'][0]
    assert row['max_cumulative_progress_drawdown'] == 0
    assert row['longest_missing_support_gap_px'] == 64
    assert row['low_advance_16px_fraction'] == 0
    assert not route_field_diagnostics([[0, 2], [0, 2]], fields)['valid']
