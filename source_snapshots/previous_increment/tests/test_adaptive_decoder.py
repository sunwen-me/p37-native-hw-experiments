import numpy as np
import torch
from unittest.mock import patch

from adaptive_field_decoder import adaptive_ordered_field_to_path, _fixed_vertices
from safe_field_decoder import safe_ordered_field_to_path
from p37_model import OrderedDenseFieldFlow


def tail_fields(gap=False):
    height, width = 17, 301
    occupancy = torch.zeros(1, 1, height, width)
    occupancy[:, :, 7:10, 5:296] = 1
    progress = ((torch.arange(width) - 5) / 265).clamp(0, 1)[None, None, None].expand_as(occupancy).clone()
    start = torch.tensor([[2 * 5 / (width - 1) - 1, 0.]])
    goal = torch.tensor([[2 * 295 / (width - 1) - 1, 0.]])
    if gap:
        occupancy[:, :, :, 284:289] = 0
    return occupancy, progress, torch.ones_like(occupancy), start, goal


def test_supported_tail_is_actually_traversed_after_progress_saturates():
    fields = tail_fields()
    old, valid, diagnostics = safe_ordered_field_to_path(
        *fields, 128, mode="scale_aware", max_terminal_jump=.5, return_diagnostics=True)
    assert not valid.any() and not diagnostics["terminal_jump_ok"].any()
    path, valid, diagnostics = adaptive_ordered_field_to_path(
        *fields, 128, seed_paths=old, seed_lengths=torch.tensor([127]))
    assert valid.all() and diagnostics["adaptive_internal_steps"].item() > 1
    assert diagnostics["adaptive_remaining_distance_px"].item() <= 3
    assert torch.equal(path[:, 0], fields[3]) and torch.equal(path[:, -1], fields[4])
    assert path.shape == (1, 128, 2)


def test_subdividing_a_missing_tail_cannot_make_it_accepted():
    fields = tail_fields(gap=True)
    old, _, _ = safe_ordered_field_to_path(
        *fields, 128, mode="scale_aware", max_terminal_jump=.5, return_diagnostics=True)
    _, valid, diagnostics = adaptive_ordered_field_to_path(
        *fields, 128, seed_paths=old, seed_lengths=torch.tensor([127]))
    assert not valid.any()
    assert not diagnostics["adaptive_complete"].any()


def test_output_retains_corners_when_internal_trace_exceeds_point_budget():
    trace = np.concatenate([np.stack([np.linspace(0, 100, 601), np.zeros(601)], 1),
                            np.stack([np.full(600, 100), np.linspace(0, 100, 601)[1:]], 1)])
    output, fits = _fixed_vertices(trace, 512)
    assert fits and output.shape == (512, 2)
    assert ((output == [100, 0]).all(1)).any()
    # All segments stay on one of the two original straight legs.
    midpoints = (output[:-1] + output[1:]) / 2
    assert ((midpoints[:, 1] == 0) | (midpoints[:, 0] == 100)).all()


def test_vertex_budget_and_internal_step_budget_fail_closed():
    _, fits = _fixed_vertices(np.array([[0, 0], [10, 0], [10, 10], [20, 10]]), 3)
    assert not fits
    _, valid, diagnostics = adaptive_ordered_field_to_path(*tail_fields(), 128, max_internal_steps=2)
    assert not valid.any() and diagnostics["adaptive_failure_code"].item() == 3
    assert diagnostics["adaptive_internal_steps"].item() == 2


def test_progress_backoff_keeps_large_requested_increments_local():
    occupancy, _, clearance, start, goal = tail_fields()
    progress = ((torch.arange(301) - 5) / 290).clamp(0, 1)[None, None, None].expand_as(occupancy)
    path, valid, diagnostics = adaptive_ordered_field_to_path(
        occupancy, progress, clearance, start, goal, 32)
    assert valid.all()
    assert diagnostics["adaptive_backoffs"].item() > 0
    assert diagnostics["adaptive_internal_steps"].item() > 32
    assert path.shape == (1, 32, 2)


def test_fallback_only_changes_abstained_rows_and_uses_the_real_sampler():
    occupancy, progress, _, start, goal = tail_fields()
    fields = torch.cat([occupancy, progress], 1)[None].expand(2, 2, -1, -1, -1).clone()
    fields[0, :, 1] = ((torch.arange(301) - 5) / 290).clamp(0, 1)[None, None]
    maps = torch.zeros(2, 2, 17, 301)
    maps[:, 1] = 1
    start, goal = start.expand(2, -1), goal.expand(2, -1)
    model = OrderedDenseFieldFlow(hidden_dim=8, horizon=128, decoder_mode="adaptive_fallback")
    before = fields.clone()
    old = model.decode_fields(maps, start, goal, fields, mode="scale_aware")
    assert old["accepted"][0].all() and not old["accepted"][1].any()
    with patch.object(model, "_integrate", return_value=fields.reshape(4, 2, 17, 301)):
        result = model.sample(maps, start, goal, candidates=2, latent=torch.zeros(2, 2, model.latent_dim))
    assert torch.equal(result["fields"], before)
    assert result["accepted"].all()
    assert not result["diagnostics"]["adaptive_attempted"][0].any()
    assert result["diagnostics"]["adaptive_attempted"][1].all()
    for key in ("paths", "accepted", "selection_scores", "selected_index"):
        assert torch.equal(result[key][0], old[key][0])
