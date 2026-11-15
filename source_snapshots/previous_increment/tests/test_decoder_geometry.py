import torch

from ordered_field_decoder import rasterize_ordered_fields


def test_square_legacy_size_and_explicit_hw_match():
    routes = torch.tensor([[[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]]])
    legacy = rasterize_ordered_fields(routes, size=64)
    explicit = rasterize_ordered_fields(routes, height=64, width=64)
    for left, right in zip(legacy, explicit):
        torch.testing.assert_close(left, right)
    assert legacy[0].shape == (1, 1, 64, 64)


def test_rectangular_rasterization_maps_normalized_axes_independently():
    routes = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]])
    occupancy, progress, valid = rasterize_ordered_fields(
        routes, height=7, width=11, subpixel=False,
    )
    assert occupancy.shape == progress.shape == valid.shape == (1, 1, 7, 11)
    assert occupancy[0, 0, 0, 0] == 1
    assert occupancy[0, 0, 6, 10] == 1
    assert progress[0, 0, 0, 0] == 0
    assert progress[0, 0, 6, 10] == 1
