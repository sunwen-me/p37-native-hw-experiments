import torch

from flow_prior import load_flow_prior
from p37_model import OrderedDenseFieldFlow, endpoint_fields
from p37_planner import P37FlowPlanner


def _transformer_model():
    return OrderedDenseFieldFlow(
        map_size=64, horizon=12, hidden_dim=8, latent_dim=4,
        architecture="transformer", transformer_window_size=4,
    )


def test_transformer_vector_field_is_finite_and_trainable_on_odd_rectangles():
    torch.manual_seed(29)
    model = _transformer_model()
    for height, width in ((31, 47), (63, 95)):
        state = torch.randn(1, 2, height, width)
        maps = torch.rand(1, 2, height, width)
        start = torch.tensor([[-0.8, -0.7]])
        goal = torch.tensor([[0.8, 0.7]])
        velocity = model.vector_field(
            state, maps, start, goal, torch.tensor([0.25]), torch.randn(1, 4),
        )
        assert velocity.shape == state.shape
        assert torch.isfinite(velocity).all()
        velocity.square().mean().backward()
        gradients = [p.grad for p in model.vector_field.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(grad).all() for grad in gradients)
        model.zero_grad(set_to_none=True)


def test_transformer_flow_keeps_hard_endpoints_during_integration():
    torch.manual_seed(31)
    model = _transformer_model()
    height, width = 31, 47
    maps = torch.zeros(1, 2, height, width)
    start = torch.tensor([[-1.0, -1.0]])
    goal = torch.tensor([[1.0, 1.0]])
    fixed = endpoint_fields(start, goal, height, width)
    fields = model._integrate(
        maps, start, goal, torch.zeros(1, 4), num_steps=2,
        noise_scale=0.0, source=torch.zeros(1, 2, height, width),
    )
    assert fields.shape == (1, 2, height, width)
    assert torch.isfinite(fields).all()
    endpoint = fixed[:, :1].bool()
    torch.testing.assert_close(fields * endpoint, fixed * endpoint)


def test_transformer_checkpoint_round_trips_with_strict_loading(tmp_path):
    torch.manual_seed(37)
    original = _transformer_model()
    checkpoint = tmp_path / "transformer.pth"
    torch.save({
        "model_state_dict": original.state_dict(),
        "config": {
            "map_size": 64, "horizon": 12, "hidden_dim": 8,
            "latent_dim": 4, "architecture": "transformer",
            "transformer_window_size": 4,
        },
    }, checkpoint)
    loaded, state = load_flow_prior(checkpoint, "cpu")
    assert state["config"]["architecture"] == "transformer"
    assert loaded.architecture == "transformer"
    assert loaded.transformer_window_size == 4
    assert tuple(loaded.state_dict()) == tuple(original.state_dict())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_transforms_alias_selects_canonical_transformer_backbone():
    model = OrderedDenseFieldFlow(
        hidden_dim=8, latent_dim=4, architecture="transforms",
        transformer_window_size=4,
    )
    assert model.architecture == "transformer"
    assert model.vector_field.__class__.__name__ == "OrderedDenseFieldTransformerVectorField"


def test_transformer_checkpoint_runs_through_pixel_coordinate_planner(tmp_path):
    import numpy as np

    torch.manual_seed(41)
    original = _transformer_model()
    checkpoint = tmp_path / "transformer-planner.pth"
    torch.save({
        "model_state_dict": original.state_dict(),
        "config": {
            "map_size": 64, "horizon": 12, "hidden_dim": 8,
            "latent_dim": 4, "architecture": "transformer",
            "transformer_window_size": 4,
        },
    }, checkpoint)
    planner = P37FlowPlanner(checkpoint, device="cpu")
    occupancy = np.zeros((13, 19), dtype=np.float32)
    start = np.asarray([[1.0, 2.0]], dtype=np.float32)
    goal = np.asarray([[17.0, 10.0]], dtype=np.float32)

    result = planner.plan(
        occupancy, start, goal, candidates=1, seed=43, flow_steps=1,
        noise_scale=0.0,
    )

    assert result["paths"].shape == (1, 1, 12, 2)
    assert torch.isfinite(result["paths"]).all()
    torch.testing.assert_close(result["selected_path"][0, 0], torch.from_numpy(start[0]))
    torch.testing.assert_close(result["selected_path"][0, -1], torch.from_numpy(goal[0]))
