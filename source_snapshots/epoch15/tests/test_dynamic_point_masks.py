import torch
import pytest

from p10_model import _point_masks
from p37_model import endpoint_fields


def test_square_mask_preserves_legacy_indexing_and_size_keyword():
    points = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    expected = torch.zeros(3, 1, 64, 64)
    indices = torch.tensor([0, 4095, 32 * 64 + 32])
    expected.view(3, -1).scatter_(1, indices[:, None], 1.0)
    torch.testing.assert_close(_point_masks(points, 64), expected)
    torch.testing.assert_close(_point_masks(points, size=64), expected)


def test_rectangular_endpoint_fields_use_x_width_and_y_height():
    start = torch.tensor([[-1.0, -1.0], [1.0, -1.0]])
    goal = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
    fields = endpoint_fields(start, goal, height=7, width=11)
    assert fields.shape == (2, 2, 7, 11)
    assert fields[0, 0, 0, 0] == 1 and fields[0, 1, 6, 10] == 1
    assert fields[1, 0, 0, 10] == 1 and fields[1, 1, 6, 0] == 1


def test_endpoint_fields_reject_out_of_bounds_normalized_coordinates():
    with pytest.raises(ValueError, match="inside the map"):
        endpoint_fields(torch.tensor([[1.1, 0.0]]), torch.zeros(1, 2), 8, 12)
