from unittest.mock import patch

import torch

import p37_model
from p37_model import OrderedDenseFieldFlow
from adaptive_field_decoder import adaptive_ordered_field_to_path
from test_adaptive_decoder import tail_fields


def test_bounded_retry_preserves_first_accepted_rows_and_reuses_original_seed():
    occupancy, progress, _, start, goal = tail_fields()
    fields = torch.cat([occupancy, progress], 1)[None].expand(2, 2, -1, -1, -1).clone()
    maps = torch.zeros(2, 2, 17, 301)
    maps[:, 1] = 1
    start, goal = start.expand(2, -1), goal.expand(2, -1)
    model = OrderedDenseFieldFlow(hidden_dim=8, horizon=128, decoder_mode="bounded_fallback")
    baseline = model.decode_fields(maps, start, goal, fields, mode="scale_aware")
    first_result = []
    calls = []
    def controlled(*args, **kwargs):
        clipped = kwargs["clip_internal_move"]
        calls.append(clipped)
        path, valid, diagnostics = adaptive_ordered_field_to_path(*args, **kwargs)
        if not clipped:
            first_result.append((path.clone(), valid.clone()))
            # Force one task to abstain to exercise independent second retry.
            valid = valid.clone()
            valid[2:] = False
        else:
            assert len(args[0]) == 2
            assert torch.equal(kwargs["seed_paths"], baseline["paths"][1])
        return path, valid, diagnostics
    with patch.object(p37_model, "adaptive_ordered_field_to_path", side_effect=controlled):
        result = model.decode_fields(maps, start, goal, fields)
    assert calls == [False, True]
    assert result["accepted"].all()
    assert not result["diagnostics"]["bounded_attempted"][0].any()
    assert result["diagnostics"]["bounded_attempted"][1].all()
    assert torch.equal(result["paths"][0], first_result[0][0][:2])


def test_clipped_moves_still_reject_a_missing_supported_tail():
    occupancy, progress, clearance, start, goal = tail_fields(gap=True)
    _, valid, diagnostics = adaptive_ordered_field_to_path(
        occupancy, progress, clearance, start, goal, 128, clip_internal_move=True)
    assert not valid.any() and not diagnostics["adaptive_complete"].any()


def test_bounded_mode_keeps_all_adaptive_accepted_outputs_bitwise():
    occupancy, progress, _, start, goal = tail_fields()
    maps = torch.zeros(1, 2, 17, 301)
    maps[:, 1] = 1
    fields = torch.cat([occupancy, progress], 1)[None]
    model = OrderedDenseFieldFlow(hidden_dim=8, horizon=128, decoder_mode="bounded_fallback")
    old = model.decode_fields(maps, start, goal, fields, mode="adaptive_fallback")
    new = model.decode_fields(maps, start, goal, fields)
    assert old["accepted"].all()
    for key in ("paths", "accepted", "selection_scores", "selected_index", "fields", "decoder_fields"):
        assert torch.equal(old[key], new[key]), key
    for key in old["diagnostics"]:
        assert torch.equal(old["diagnostics"][key], new["diagnostics"][key]), key
    assert not new["diagnostics"]["bounded_attempted"].any()
