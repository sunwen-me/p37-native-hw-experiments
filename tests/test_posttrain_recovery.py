import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from posttrain_p37_native_hw import (
    atomic_torch_save, batch_plan_sha, capture_rng, epoch_batches, make_loader, restore_rng,
)


def test_epoch_batch_cursor_is_reconstructible_and_loader_preserves_global_rng():
    class Dataset:
        def shape_at(self, index):
            return [(8, 12), (12, 8), (16, 24)][index % 3]

        def __getitem__(self, index):
            return torch.tensor(index)

    dataset = Dataset()
    ids = list(range(23))
    plan = epoch_batches(dataset, ids, 4, 192, 23, 1)
    rebuilt = epoch_batches(dataset, ids, 4, 192, 23, 1)
    assert sorted(task for batch in plan for task in batch) == ids
    assert batch_plan_sha(plan) == batch_plan_sha(rebuilt)
    before = torch.get_rng_state().clone()
    first = [item.tolist() for item in make_loader(dataset, plan[:3], 41)]
    resumed = [item.tolist() for item in make_loader(dataset, rebuilt[3:], 41)]
    assert first + resumed == plan
    assert torch.equal(torch.get_rng_state(), before)


def test_partial_checkpoint_restores_optimizer_and_all_training_rngs(tmp_path):
    random.seed(37)
    np.random.seed(37)
    torch.manual_seed(37)
    generator = torch.Generator().manual_seed(83)

    def model_and_optimizer():
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.25), torch.nn.Linear(4, 1))
        return model, torch.optim.AdamW(model.parameters(), lr=.001)

    def step(model, optimizer, gen):
        values = torch.randn(5, 3, generator=gen) * (random.random() + np.random.random())
        loss = model(values).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    model, optimizer = model_and_optimizer()
    for _ in range(2):
        step(model, optimizer, generator)
    checkpoint = tmp_path / 'partial.pth'
    atomic_torch_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'rng': capture_rng(generator)}, checkpoint)
    expected_losses = [step(model, optimizer, generator) for _ in range(3)]
    expected_weights = {key: value.clone() for key, value in model.state_dict().items()}
    restored_model, restored_optimizer = model_and_optimizer()
    restored_generator = torch.Generator()
    saved = torch.load(checkpoint, weights_only=False)
    restored_model.load_state_dict(saved['model'], strict=True)
    restored_optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'], restored_generator)
    actual_losses = [step(restored_model, restored_optimizer, restored_generator) for _ in range(3)]
    assert all(torch.equal(actual, expected) for actual, expected in zip(actual_losses, expected_losses))
    assert all(torch.equal(restored_model.state_dict()[key], value) for key, value in expected_weights.items())
