import numpy as np
import torch

from path_refinement import (
    DensePathRefiner,
    path_metrics,
    swept_clearance,
)
from p37_model import OrderedDenseFieldFlow
from refinement_planner import FlowRefinementPlanner
from run_manifest import file_sha256


def test_rectangular_path_metrics_and_offset_use_per_axis_pixel_scales():
    height, width = 9, 17
    path = torch.tensor([[[-1.0, 0.0], [1.0, 1.0]]])
    metrics = path_metrics(path, map_height=height, map_width=width)
    torch.testing.assert_close(metrics["length"], torch.tensor([272.0 ** 0.5]))

    base = torch.zeros(1, 4, 2)
    offsets = torch.zeros_like(base)
    offsets[:, 1, :] = torch.tensor([2.0, -1.0])
    refined = DensePathRefiner.apply_offset(
        base, offsets, map_height=height, map_width=width,
    )
    torch.testing.assert_close(refined[0, 1], torch.tensor([0.25, -0.25]))
    torch.testing.assert_close(refined[:, 0], base[:, 0])
    torch.testing.assert_close(refined[:, -1], base[:, -1])


def test_rectangular_swept_clearance_uses_height_for_vertical_boundary():
    height, width = 9, 17
    maps = torch.zeros(1, 4, height, width)
    maps[:, 1] = 100.0
    path = torch.tensor([[[-0.8, 1.0], [0.8, 1.0]]])
    clearance = swept_clearance(path, maps, distance_scale=1.0, offset=0.0)
    torch.testing.assert_close(clearance, torch.full_like(clearance, 0.5))


def test_refiner_and_deployment_planner_load_checkpoint_on_mixed_shapes(tmp_path):
    torch.manual_seed(23)
    horizon = 8
    prior = OrderedDenseFieldFlow(
        map_size=64, horizon=horizon, hidden_dim=8, latent_dim=4,
    )
    prior_path = tmp_path / "prior.pth"
    torch.save({
        "model_state_dict": prior.state_dict(),
        "config": {"map_size": 64, "horizon": horizon, "hidden_dim": 8, "latent_dim": 4},
    }, prior_path)
    refiner = DensePathRefiner(hidden=8)
    refiner_path = tmp_path / "refiner.pth"
    torch.save({
        "model_state_dict": refiner.state_dict(),
        "config": {"hidden": 8},
        "cache_audit": {
            "source_checkpoint_sha256": file_sha256(prior_path),
            "flow_steps": 1, "latent_scale": 1.0, "noise_scale": 0.0,
        },
        "phase": "warm",
        "update": 0,
    }, refiner_path)

    planner = FlowRefinementPlanner(prior_path, refiner_path, device="cpu")
    shapes = ((12, 20), (17, 9))
    maps = [np.zeros(shape, dtype=np.float32) for shape in shapes]
    starts = np.asarray([[1, 2], [2, 3]], dtype=np.float32)
    goals = np.asarray([[18, 10], [7, 15]], dtype=np.float32)
    result = planner.plan(
        maps, starts, goals, candidates=1, seed=9, flow_steps=1,
    )

    assert result["map_shapes"] == shapes
    assert result["paths"].shape == (2, 2, horizon, 2)
    assert torch.isfinite(result["paths"]).all()
    expected_starts = torch.from_numpy(starts)[:, None].expand(-1, 2, -1)
    expected_goals = torch.from_numpy(goals)[:, None].expand(-1, 2, -1)
    torch.testing.assert_close(result["paths"][:, :, 0], expected_starts)
    torch.testing.assert_close(result["paths"][:, :, -1], expected_goals)
    assert result["best_effort_path"].shape == (2, horizon, 2)
