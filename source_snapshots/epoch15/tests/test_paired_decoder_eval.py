import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from p37_paired_decoder_eval import candidate_outcomes, decode_fields
from p37_model import OrderedDenseFieldFlow


def test_cached_field_decode_matches_deployment_without_network_queries():
    torch.manual_seed(83)
    model = OrderedDenseFieldFlow(hidden_dim=8, latent_dim=4, horizon=8).eval()
    maps = torch.zeros(1, 2, 31, 47)
    maps[:, 1] = 1
    starts, goals = torch.tensor([[-.8, -.5]]), torch.tensor([[.7, .4]])
    with torch.no_grad():
        sampled = model.sample(maps, starts, goals, candidates=2, num_steps=1)
    before = sampled["fields"].clone()
    calls = []
    hook = model.vector_field.register_forward_hook(lambda *_: calls.append(1))
    decoded = decode_fields(model, maps, starts, goals, sampled["fields"], "legacy", 8)
    for key in ("paths", "selection_scores", "selected_index", "accepted"):
        assert torch.equal(decoded[key], sampled[key])
    assert torch.equal(decoded["decoder_valid"], sampled["decoder_support_valid"])
    longer = decode_fields(model, maps, starts, goals, sampled["fields"], "scale_aware", 32)
    assert longer["paths"].shape == (1, 2, 32, 2)
    assert torch.equal(sampled["fields"], before)
    assert not calls
    hook.remove()


def test_abstained_geometric_path_does_not_count_as_deployment_success():
    outcome = candidate_outcomes([True, False], [False, False], [False, False], 0)
    assert outcome["selected_geometry_success"]
    assert outcome["abstain"]
    assert not outcome["deployment_success"]
    assert not outcome["any_accepted_safe"]
    unsafe = candidate_outcomes([True, False], [False, True], [False, True], 1)
    assert unsafe["any_geometry_success"]
    assert not unsafe["deployment_success"]
    assert unsafe["selected_accepted_unsafe"]
    assert unsafe["accepted_unsafe_candidates"] == 1
