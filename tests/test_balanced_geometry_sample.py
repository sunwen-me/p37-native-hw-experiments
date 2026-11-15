from collections import Counter

import numpy as np

from scripts.train_p37_ordered_field import balanced_geometry_sample


class _ShapeDataset:
    def __init__(self, shapes):
        self.shapes = list(shapes)

    def shape_at(self, index):
        return self.shapes[int(index)]


def test_geometry_sample_balances_all_available_shapes():
    shapes = [(64 + index, 96) for index in range(13) for _ in range(16)]
    dataset = _ShapeDataset(shapes)
    ids = balanced_geometry_sample(
        dataset, np.arange(len(shapes)), 26, np.random.default_rng(17),
    )
    counts = Counter(dataset.shape_at(index) for index in ids)
    assert len(ids) == 26
    assert len(counts) == 13
    assert set(counts.values()) == {2}


def test_geometry_sample_respects_small_bucket_capacity():
    shapes = [(64, 96)] + [
        (65 + index, 96) for index in range(12) for _ in range(2)
    ]
    dataset = _ShapeDataset(shapes)
    ids = balanced_geometry_sample(
        dataset, np.arange(len(shapes)), 26, np.random.default_rng(19),
    )
    counts = Counter(dataset.shape_at(index) for index in ids)
    assert len(ids) == 25
    assert counts[(64, 96)] == 1
    assert sorted(count for shape, count in counts.items() if shape != (64, 96)) == [2] * 12
