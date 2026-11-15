import pytest
import numpy as np

from utils.dataset import PathPlanningDataset, ShapeBucketBatchSampler, collate_fn
from test_rectangular_geometry import _write_dataset


def test_shape_sampler_never_mixes_h_w_and_collate_rejects_padding(tmp_path):
    source = tmp_path / "mixed_maps.npz"
    _write_dataset(source)
    dataset = PathPlanningDataset(str(source), horizon=4, validate_geometry=False)
    batches = list(ShapeBucketBatchSampler(dataset, batch_size=3, shuffle=False))
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]
    for batch in batches:
        assert len({dataset.shape_at(index) for index in batch}) == 1
    with pytest.raises(ValueError, match="same-H/W"):
        collate_fn([dataset[0], dataset[2]])
    assert collate_fn([dataset[0], dataset[1]])["map"].shape == (2, 1, 6, 10)
