from pathlib import Path

import pytest
import torch

from p37_model import OrderedDenseFieldFlow, candidate_selection_scores, endpoint_fields
from safe_field_decoder import support_weighted_progress_smoothing


@pytest.mark.parametrize("kernel", [-1, 2, 4, 2.5, True])
def test_invalid_smoothing_configuration_is_rejected(kernel):
    with pytest.raises(ValueError):
        OrderedDenseFieldFlow(hidden_dim=8, decode_progress_smoothing=kernel)


def test_low_support_background_does_not_dominate_route_progress():
    occupancy = torch.zeros(1, 1, 9, 15)
    occupancy[:, :, 4, :] = 1
    progress = torch.ones_like(occupancy)
    progress[:, :, 4, :] = .25
    before = progress.clone()
    corrected = support_weighted_progress_smoothing(occupancy, progress, 7)
    torch.testing.assert_close(corrected[:, :, 4, :], torch.full((1, 1, 15), .25))
    assert torch.equal(progress, before)
    assert torch.isfinite(corrected).all()


def test_smoothing_preserves_hard_endpoints_raw_fields_and_support_gap():
    model = OrderedDenseFieldFlow(hidden_dim=8, horizon=128, decoder_mode="scale_aware",
                                  decode_progress_smoothing=7)
    height, width = 17, 101
    maps = torch.zeros(1, 2, height, width)
    maps[:, 1] = 1
    start, goal = torch.tensor([[-.9, 0.]]), torch.tensor([[.9, 0.]])
    fields = torch.zeros(1, 2, 2, height, width)
    fields[:, :, 0, 8, 5:96] = 1
    fields[:, :, 1, 8, :] = torch.linspace(0, 1, width)
    fields[:, :, 0, :, 45:56] = 0  # missing route segment must remain missing
    fixed = endpoint_fields(start, goal, height, width)[:, None]
    fields = fields * (1 - fixed[:, :, :1]) + fixed
    before = fields.clone()
    result = model.decode_fields(maps, start, goal, fields)
    assert torch.equal(fields, before) and torch.equal(result["fields"], before)
    assert torch.equal(result["decoder_fields"][:, :, 0], fields[:, :, 0])
    endpoint = fixed[:, :, :1].expand_as(fields).bool()
    assert torch.equal(result["decoder_fields"][endpoint], fields[endpoint])
    assert not result["accepted"].any()  # smoothing cannot rescue a disconnected corridor
    assert result["abstain"].all()
    assert torch.equal(result["paths"][:, :, 0], start[:, None].expand(-1, 2, -1))
    assert torch.equal(result["paths"][:, :, -1], goal[:, None].expand(-1, 2, -1))
    expected_scores = candidate_selection_scores(
        result["decoder_fields"], result["paths"], maps, result["decoder_support_valid"],
    )
    assert torch.equal(result["selection_scores"], expected_scores)


def test_training_and_inference_share_the_repaired_implementation():
    root = Path(__file__).resolve().parents[1]
    for name in ("safe_field_decoder.py", "adaptive_field_decoder.py", "p37_model.py"):
        assert (root / "inference" / name).read_bytes() == (
            root / "training/source/map_flow_matching_p1" / name
        ).read_bytes()
