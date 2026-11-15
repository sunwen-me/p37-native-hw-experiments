import torch

from safe_field_decoder import safe_ordered_field_to_path


def test_rectangular_decoder_keeps_exact_endpoints_and_valid_support():
    height, width, horizon = 12, 20, 8
    occupancy = torch.ones(1, 1, height, width)
    progress = torch.linspace(0, 1, width)[None, None, None, :].expand(1, 1, height, -1).clone()
    clearance = torch.ones_like(occupancy)
    start = torch.tensor([[-1.0, 0.0]])
    goal = torch.tensor([[1.0, 0.0]])
    path, valid = safe_ordered_field_to_path(
        occupancy, progress, clearance, start, goal, horizon,
        continuity_beta=0.0, max_terminal_jump=2.0, return_valid=True,
    )
    assert path.shape == (1, horizon, 2)
    torch.testing.assert_close(path[:, 0], start)
    torch.testing.assert_close(path[:, -1], goal)
    assert valid.tolist() == [True]
