import numpy as np
import torch

from p37_model import OrderedDenseFieldFlow
from p37_planner import P37FlowPlanner


def test_planner_supports_mixed_map_shapes_and_returns_pixel_paths(tmp_path):
    torch.manual_seed(3)
    model = OrderedDenseFieldFlow(hidden_dim=8, latent_dim=4, horizon=8)
    checkpoint = tmp_path / "prior.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"map_size": 64, "horizon": 8, "hidden_dim": 8, "latent_dim": 4},
    }, checkpoint)
    planner = P37FlowPlanner(checkpoint, device="cpu")
    maps = [np.zeros((12, 20), np.float32), np.zeros((16, 10), np.float32)]
    starts = np.asarray([[0, 3], [2, 1]], dtype=np.float32)
    goals = np.asarray([[19, 8], [8, 15]], dtype=np.float32)
    result = planner.plan(maps, starts, goals, candidates=1, seed=4, flow_steps=1, noise_scale=0)
    assert result["paths"].shape == (2, 1, 8, 2)
    assert result["map_shapes"] == ((12, 20), (16, 10))
    torch.testing.assert_close(result["selected_path"][:, 0], torch.from_numpy(starts))
    torch.testing.assert_close(result["selected_path"][:, -1], torch.from_numpy(goals))
