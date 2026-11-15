"""Dense path-mask Flow planner.

Unlike P9, the Flow state in P10 is the complete spatial path mask.  The
continuous vector field never operates on a compressed anchor representation;
anchors are extracted only after the final mask is generated for compatibility
with the existing path evaluator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _point_masks(
    points: torch.Tensor, height: int | None = None, width: int | None = None,
    *, size: int | None = None,
) -> torch.Tensor:
    """Rasterize normalized ``(x, y)`` points to a one-pixel mask.

    Passing only ``height`` retains the legacy square-map interface.
    """
    if size is not None:
        if height is not None and int(height) != int(size):
            raise ValueError("size and height disagree")
        height = int(size)
    if height is None:
        raise ValueError("height is required")
    height = int(height)
    width = height if width is None else int(width)
    if height < 1 or width < 1:
        raise ValueError("mask height and width must be positive")
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B,2]")
    scale = points.new_tensor([(width - 1) * 0.5, (height - 1) * 0.5])
    xy = ((points + 1.0) * scale).round().long()
    xy[:, 0].clamp_(0, width - 1)
    xy[:, 1].clamp_(0, height - 1)
    mask = torch.zeros(
        len(points), 1, height, width, device=points.device, dtype=points.dtype,
    )
    index = xy[:, 1] * width + xy[:, 0]
    mask.view(len(points), -1).scatter_(1, index[:, None], 1.0)
    return mask


class DenseMaskVectorField(nn.Module):
    """Small spatial U-Net conditioned on map, endpoints, time and route mode."""

    def __init__(self, map_channels=2, hidden_dim=64, route_count=4):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.route_embedding = nn.Embedding(route_count, hidden_dim * 4)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(map_channels + 3, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim), nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2), nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2), nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 4), nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4), nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4), nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4), nn.SiLU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 6, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2), nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim), nn.SiLU(),
        )
        self.out = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x, map_input, start, goal, t, mode_index):
        size = x.shape[-1]
        start_mask = _point_masks(start, size)
        goal_mask = _point_masks(goal, size)
        h0 = self.stem(torch.cat([x, map_input, start_mask, goal_mask], dim=1))
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        cond = self.time_embedding(t[:, None]) + self.route_embedding(mode_index)
        h2 = h2 + cond[:, :, None, None]
        h2 = self.mid(h2)
        u1 = F.interpolate(h2, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, h1], dim=1))
        u2 = F.interpolate(u1, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, h0], dim=1))
        return self.out(u2)


def mask_to_path(mask, start, goal, horizon, power=4.0, *, progress=None,
                 progress_beta=1280.0, force_endpoints=True):
    """Decode a dense path field into an ordered continuous path.

    A one-channel mask has no traversal order, so the legacy decoder falls
    back to chord projection.  New models should pass a second, same-shaped
    ``progress`` field; it preserves backtracking and obstacle-side turns
    without introducing an online search or geometry repair step.
    """
    if progress is not None:
        from ordered_field_decoder import ordered_field_to_path
        return ordered_field_to_path(
            mask, progress, start, goal, horizon,
            beta=progress_beta, power=power,
        )
    batch, _, height, width = mask.shape
    probability = mask.clamp(0.0, 1.0).pow(power)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=mask.device),
        torch.linspace(-1.0, 1.0, width, device=mask.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1).reshape(1, 1, height * width, 2)
    direction = goal - start
    denom = direction.square().sum(dim=-1, keepdim=True).clamp_min(1e-5)
    progress = ((points - start[:, None, None]) * direction[:, None, None]).sum(-1)
    progress = (progress / denom[:, None]).clamp(0.0, 1.0)
    centers = torch.linspace(0.0, 1.0, horizon, device=mask.device)
    paths = []
    for center in centers:
        weights = probability.reshape(batch, -1)
        weights = weights * torch.exp(-80.0 * (progress[:, 0] - center).square())
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        paths.append((weights.unsqueeze(-1) * points[:, 0]).sum(dim=-2))
    path = torch.stack(paths, dim=1)
    if force_endpoints:
        path[:, 0] = start
        path[:, -1] = goal
    return path.clamp(-1.0, 1.0)


class DenseMaskFlowPlanner(nn.Module):
    """GenPlanner-style dense path-mask Flow with multiple route modes."""

    variant = "p10_dense_path_mask_flow"

    def __init__(self, map_size=64, horizon=128, map_dim=256, hidden_dim=64,
                 route_count=4, coarse_points=8, accept_threshold=0.5,
                 hard_endpoints=True, **kwargs):
        super().__init__()
        self.map_size = int(map_size)
        self.horizon = int(horizon)
        self.route_count = int(route_count)
        self.coarse_points = int(coarse_points)
        self.accept_threshold = float(accept_threshold)
        self.hard_endpoints = bool(hard_endpoints)
        self.vector_field = DenseMaskVectorField(
            map_channels=2, hidden_dim=hidden_dim, route_count=route_count
        )

    def _integrate(self, map_input, start, goal, mode_index, num_steps, noise_scale):
        batch = len(start)
        x = noise_scale * torch.randn(
            batch, 1, self.map_size, self.map_size,
            device=map_input.device, dtype=map_input.dtype,
        )
        # Endpoint pixels can be held fixed for the controlled ablation.
        endpoint = _point_masks(torch.cat([start, goal], dim=0), self.map_size)
        endpoint = endpoint[:batch] + endpoint[batch:]
        dt = 1.0 / max(int(num_steps), 1)
        for step in range(int(num_steps)):
            t = torch.full((batch,), step * dt, device=x.device, dtype=x.dtype)
            velocity = self.vector_field(x, map_input, start, goal, t, mode_index)
            x = x + dt * velocity
            if self.hard_endpoints:
                x = x * (1.0 - endpoint) + endpoint
        return x

    @torch.no_grad()
    def sample(self, map_input, start, goal, num_steps=50, noise_scale=1.0):
        masks, paths = [], []
        for mode in range(self.route_count):
            mode_index = torch.full(
                (len(start),), mode, device=start.device, dtype=torch.long
            )
            final_mask = self._integrate(
                map_input, start, goal, mode_index, num_steps, noise_scale
            )
            masks.append(final_mask)
            paths.append(mask_to_path(final_mask, start, goal, self.horizon,
                                      force_endpoints=self.hard_endpoints))
        masks = torch.stack(masks, dim=1)
        paths = torch.stack(paths, dim=1)
        # P10 has no learned selector yet. All generated candidates are exposed
        # to the established evaluator; candidate coverage is the primary metric.
        logits = torch.zeros(
            len(start), self.route_count, device=start.device, dtype=start.dtype
        )
        accepted = logits.sigmoid() >= self.accept_threshold
        selected = torch.zeros(len(start), device=start.device, dtype=torch.long)
        return {
            "mask": masks, "paths": paths, "validity_logits": logits,
            "accepted": accepted, "abstain": ~accepted.any(dim=1),
            "selected_index": selected, "selected_path": paths[:, 0],
        }
