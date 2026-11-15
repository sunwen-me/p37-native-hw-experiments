import torch

from path_refinement import DensePathRefiner, path_metrics


def test_refiner_square_geometry_matches_legacy_64_pixel_formulas():
    torch.manual_seed(31)
    paths = torch.rand(3, 20, 2) * 2.0 - 1.0
    px = paths * 31.5
    step = px[:, 1:] - px[:, :-1]
    speed = step.norm(dim=-1)
    second = step[:, 1:] - step[:, :-1]
    dot = (step[:, 1:] * step[:, :-1]).sum(-1)
    cosine = (dot / (speed[:, 1:] * speed[:, :-1]).clamp_min(1e-8)).clamp(-1 + 1e-6, 1 - 1e-6)
    turns = torch.where(
        (speed[:, 1:] > 1e-5) & (speed[:, :-1] > 1e-5),
        cosine.acos(), torch.zeros_like(cosine),
    )
    legacy = {
        "length": speed.sum(-1),
        "smoothness": second.norm(dim=-1).mean(-1),
        "turning_angle": turns.sum(-1),
        "squared_curvature": second.square().sum(-1).mean(-1),
    }
    current = path_metrics(paths, map_size=64)
    for key in legacy:
        torch.testing.assert_close(current[key], legacy[key], rtol=0, atol=0)

    base = torch.rand(3, 20, 2) * 2.0 - 1.0
    offset = torch.randn_like(base)
    mask = torch.ones_like(offset[:, :, :1])
    mask[:, [0, -1]] = 0
    old_offset = base + offset * mask / 31.5
    legacy_api = DensePathRefiner.apply_offset(base, offset, map_size=64)
    explicit_hw = DensePathRefiner.apply_offset(
        base, offset, map_height=64, map_width=64,
    )
    torch.testing.assert_close(legacy_api, old_offset, rtol=0, atol=0)
    torch.testing.assert_close(explicit_hw, old_offset, rtol=0, atol=0)
