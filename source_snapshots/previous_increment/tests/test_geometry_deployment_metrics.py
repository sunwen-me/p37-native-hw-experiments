from types import SimpleNamespace

import numpy as np
import torch

from scripts import geometry_eval
from scripts.train_p37_ordered_field import geometry_success


def test_geometry_validation_distinguishes_abstention_unsafe_acceptance_and_selection(monkeypatch):
    safety = [[True, False], [True, False], [False, True], [True, False]]
    accepted = torch.tensor([[False, False], [False, True], [True, True], [True, False]])

    class Base:
        maps = [np.full((8, 12), i) for i in range(4)]
        starts = goals = np.zeros((4, 2))
        geometry = SimpleNamespace(radius_grid=1.5, to_json=lambda: '{}')

        def shape_at(self, _):
            return 8, 12

        def __getitem__(self, _):
            return {"map": torch.zeros(1, 8, 12), "distance_field": torch.ones(1, 8, 12),
                    "start": torch.zeros(2), "goal": torch.zeros(2)}

        def denormalize_paths(self, paths, task_ids):
            return paths

    class Model:
        def eval(self):
            return self

        def sample(self, *args, **kwargs):
            return {"paths": torch.zeros(4, 2, 8, 2), "selected_index": torch.tensor([0, 1, 0, 0]),
                    "decoder_valid": accepted, "accepted": accepted}

    monkeypatch.setattr(geometry_eval, 'score_task', lambda job: safety[int(job[2][0, 0])])
    args = SimpleNamespace(
        geometry_val_max_tasks=0, seed=17, geometry_val_batch_size=4,
        geometry_val_max_pixels=0, geometry_val_candidates=2, geometry_val_steps=1,
        geometry_val_noise_scale=1., distance_scale=8., geometry_val_clearance_temperature=.5,
        geometry_val_clearance_offset=2 ** -.5, geometry_val_clearance_power=2.,
        geometry_val_continuity_beta=60.,
    )
    result = geometry_success(Model(), Base(), range(4), 'cpu', args)
    for stats in [result, result['by_shape']['8x12']]:
        assert stats['tasks'] == 4
        assert stats['selected_path_success_rate'] == .5
        assert stats['deployment_success_rate'] == .25
        assert stats['abstention_rate'] == .25
        assert stats['selected_accepted_unsafe_rate'] == .5
        assert stats['accepted_oracle_best_of_k_success_rate'] == .5
        assert stats['deployment_selector_miss_rate'] == .25
        assert stats['selector_miss_rate'] == .5
        assert stats['accepted_safe_candidate_rate'] == .25
    empty = geometry_success(Model(), Base(), [], 'cpu', args)
    assert empty['tasks'] == 0 and empty['deployment_success_rate'] == 0
    assert empty['by_shape'] == {}
