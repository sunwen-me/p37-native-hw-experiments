import numpy as np

from utils.dataset import PathPlanningDataset


def _write_dataset(path):
    maps = np.empty(3, dtype=object)
    maps[0] = np.zeros((6, 10), dtype=np.float32)
    maps[1] = np.zeros((6, 10), dtype=np.float32)
    maps[2] = np.zeros((8, 5), dtype=np.float32)
    starts = np.asarray([[1, 2], [2, 3], [1, 1]], dtype=np.float32)
    goals = np.asarray([[8, 4], [7, 2], [3, 6]], dtype=np.float32)
    paths = np.empty(3, dtype=object)
    for index in range(3):
        paths[index] = np.linspace(starts[index], goals[index], 4, dtype=np.float32)
    np.savez(path, maps=maps, starts=starts, goals=goals, paths=paths,
             path_lengths=np.full(3, 4, dtype=np.int64))


def test_coordinate_round_trip_and_sample_shapes(tmp_path):
    source = tmp_path / "mixed_maps.npz"
    _write_dataset(source)
    dataset = PathPlanningDataset(str(source), horizon=4, normalize=True, validate_geometry=False)
    pixels = np.asarray([[9, 5], [0, 0]], dtype=np.float32)
    normalized = dataset.normalize_coordinates(pixels, height=6, width=10)
    np.testing.assert_allclose(normalized, [[1, 1], [-1, -1]])
    np.testing.assert_allclose(dataset.denormalize_coordinates(normalized, 6, 10), pixels)
    assert dataset[0]["map"].shape == (1, 6, 10)
    assert dataset[2]["map"].shape == (1, 8, 5)
    assert dataset[2]["spatial_size"].tolist() == [8, 5]


def test_quarter_turn_swaps_h_w_and_transforms_xy(tmp_path, monkeypatch):
    source = tmp_path / "mixed_maps.npz"
    _write_dataset(source)
    dataset = PathPlanningDataset(str(source), horizon=4, normalize=False, validate_geometry=False)
    monkeypatch.setattr(np.random, "randint", lambda *args: 1)
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    occupancy = np.zeros((6, 10), dtype=np.float32)
    start = np.asarray([2, 1], dtype=np.float32)
    goal = np.asarray([8, 4], dtype=np.float32)
    path = np.stack([start, goal])
    rotated, out_path, out_start, out_goal = dataset.augment_data(occupancy, path, start, goal)
    assert rotated.shape == (10, 6)
    np.testing.assert_array_equal(out_start, [1, 7])
    np.testing.assert_array_equal(out_goal, [4, 1])
    np.testing.assert_array_equal(out_path, [[1, 7], [4, 1]])
