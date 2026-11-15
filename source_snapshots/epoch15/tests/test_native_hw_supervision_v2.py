import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "training/source/map_flow_matching_p1/scripts"))
import train_p37_ordered_field as trainer
from ordered_field_decoder import rasterize_ordered_fields
from safe_field_decoder import safe_ordered_field_to_path
from utils.dataset import PathPlanningDataset
from utils.robot_geometry import resample_safe_polyline


def test_segment_labels_connect_sparse_polyline_without_shortcutting_corner():
    height, width = 127, 193
    route = torch.tensor([[[[-0.8, -0.8], [0.7, -0.8], [0.7, 0.8]]]])
    points = rasterize_ordered_fields(route, height=height, width=width, thickness=1)
    occupancy, progress, _ = rasterize_ordered_fields(
        route, height=height, width=width, thickness=1, mode="segments",
    )
    assert label(points[0][0, 0].numpy(), structure=np.ones((3, 3)))[1] == 3
    assert label(occupancy[0, 0].numpy(), structure=np.ones((3, 3)))[1] == 1
    assert occupancy[0, 0, height // 2, width // 2] == 0  # no diagonal shortcut
    # Horizontal leg is longer in pixels: progress at the corner exceeds 1/2.
    corner = progress[0, 0, round(.1 * (height - 1)), round(.85 * (width - 1))]
    assert .55 < corner < .65


@pytest.mark.parametrize("subpixel", [False, True])
def test_segment_padding_has_no_mass_and_repeated_vertices_stay_finite(subpixel):
    short = torch.tensor([[[[-.1, 0.0], [-.1, 0.0], [.1, 0.0]]]])
    long = torch.tensor([[[[-1., -1.], [1., -1.], [1., 1.]]]])
    kwargs = dict(height=31, width=65, thickness=1, mode="segments", subpixel=subpixel)
    single = rasterize_ordered_fields(short, **kwargs)
    batched = rasterize_ordered_fields(torch.cat([short, long]), **kwargs)
    for expected, actual in zip(single, batched):
        torch.testing.assert_close(actual[:1], expected)
        assert torch.isfinite(actual).all()


def test_scale_aware_decoder_tracks_long_oracle_without_relaxing_jump_limit():
    height, width = 64, 768
    route = torch.zeros(1, 1, 128, 2)
    route[0, 0, :, 0] = torch.linspace(-.9, .9, 128)
    occupancy, progress, _ = rasterize_ordered_fields(
        route, height=height, width=width, thickness=1, mode="segments",
    )
    start, goal = route[:, 0, 0], route[:, 0, -1]
    kwargs = dict(max_terminal_jump=.5, return_valid=True)
    _, old_valid = safe_ordered_field_to_path(
        occupancy, progress, torch.ones_like(occupancy), start, goal, 128, **kwargs,
    )
    path, valid = safe_ordered_field_to_path(
        occupancy, progress, torch.ones_like(occupancy), start, goal, 128,
        mode="scale_aware", **kwargs,
    )
    assert not old_valid.item()
    assert valid.item()
    assert torch.equal(path[:, 0], start) and torch.equal(path[:, -1], goal)
    assert torch.linalg.vector_norm((path[:, -1] - path[:, -2]) *
                                    torch.tensor([(width - 1) / 63, (height - 1) / 63])) <= .5
    # Missing route support still rejects instead of fabricating a connector.
    broken = occupancy.clone()
    broken[..., width // 3:2 * width // 3] = 0
    _, broken_valid = safe_ordered_field_to_path(
        broken, progress, torch.ones_like(occupancy), start, goal, 128,
        mode="scale_aware", **kwargs,
    )
    assert not broken_valid.item()


def test_balanced_loss_retains_route_error_as_background_area_grows():
    def loss(width, reduction):
        squared = torch.zeros(1, 8, width, requires_grad=True)
        corridor = torch.zeros_like(squared)
        corridor[:, 3, 2] = 1
        error = squared + corridor * 4
        return trainer.reduce_field_error(error, torch.ones_like(error), corridor, reduction)
    small = loss(16, "support_balanced")
    large = loss(256, "support_balanced")
    torch.testing.assert_close(small, large)
    assert small.item() == 2
    assert loss(256, "pixel_mean") < loss(16, "pixel_mean")
    large.sum().backward()


def test_progress_bins_cover_quantized_tail_that_doubles_back_near_goal():
    # Room-center labels can return near an earlier part of the final leg.
    # Thick raster progress then averages those nearby samples. A point-only
    # progress kernel can miss a slice even when spatial support is present.
    corners = np.asarray([
        [536, 334], [544, 351], [541, 374.5], [544, 397.5], [541, 420.5],
        [544, 441.5], [541, 462.5], [544, 486.5], [573.5, 490],
        [605, 486.5], [636.5, 488], [670, 486.5], [703.5, 486],
        [735, 486.5], [710, 485],
    ])
    path = resample_safe_polyline(corners, horizon=128)
    scale = np.asarray([767 / 2, 511 / 2], dtype=np.float32)
    route = torch.from_numpy(path / scale - 1)[None, None]
    occupancy, progress, _ = rasterize_ordered_fields(
        route, height=512, width=768, thickness=1, mode="segments",
    )
    _, valid, diagnostics = safe_ordered_field_to_path(
        occupancy, progress, torch.ones_like(occupancy), route[:, 0, 0],
        route[:, 0, -1], 128, mode="scale_aware", max_terminal_jump=.5,
        return_diagnostics=True,
    )
    assert valid.item()
    assert diagnostics["missing_support_steps"].item() == 0


def test_hard_endpoint_restoration_preserves_real_prediction_error(tmp_path):
    height, width = 512, 768
    scale = np.array([(width - 1) / 2, (height - 1) / 2], np.float32)
    xs = np.arange(width, dtype=np.float32)
    x = xs[np.argmax(np.abs(((xs / scale[0] - 1) + 1) * scale[0] - xs))]
    start, goal = np.array([30., 40.], np.float32), np.array([x, 200.], np.float32)
    source = tmp_path / "data.npz"
    np.savez(source, maps=np.zeros((1, height, width), np.float32),
             paths=np.array([[start, goal]]), starts=start[None], goals=goal[None],
             path_lengths=np.array([2]))
    dataset = PathPlanningDataset(str(source), horizon=2, normalize=True, validate_geometry=False)
    normalized = dataset.normalize_coordinates(np.array([[start, goal]]), task_ids=[0])[:, None]
    raw = dataset.denormalize_coordinates(normalized, task_ids=[0])
    assert np.linalg.norm(raw[0, 0, -1] - goal) > 1e-5
    fixed = dataset.denormalize_paths(normalized, task_ids=[0])
    np.testing.assert_array_equal(fixed[0, 0, 0], start)
    np.testing.assert_array_equal(fixed[0, 0, -1], goal)
    perturbed = normalized.copy()
    perturbed[..., -1, 0] += .001
    result = dataset.denormalize_paths(perturbed, task_ids=[0])
    assert np.linalg.norm(result[0, 0, -1] - goal) > .1
