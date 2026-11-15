"""Ordered dense-field labels and decoder for path-mask ablations.

The existing dense-mask decoder orders pixels by projection onto the
start-to-goal chord.  That is ambiguous when a route bends around an
obstacle.  This module keeps the spatial occupancy representation, but adds a
per-pixel route progress value learned from the ordered path labels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rasterize_ordered_fields(
    routes: torch.Tensor,
    size: int = 64,
    thickness: int = 0,
    subpixel: bool = False,
    *,
    height: int | None = None,
    width: int | None = None,
    mode: str = "points",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize route occupancy and an ordered progress field.

    Args:
        routes: normalized paths with shape ``[B, K, T, 2]``.
        size: square raster resolution.
        thickness: radius used to spread both occupancy and progress.
        mode: legacy point splatting or continuous segment sampling. Segment
            mode preserves every corner, samples at most half a grid cell
            apart, and parameterizes progress by pixel arc length.

    Returns:
        ``occupancy``, ``progress``, and ``progress_valid`` with shape
        ``[B, K, size, size]``.  ``progress_valid`` identifies pixels whose
        progress is supported by a route point; it is useful for diagnostics
        and for constructing oracle fields.
    """
    if routes.ndim != 4 or routes.shape[-1] != 2:
        raise ValueError("routes must have shape [B,K,T,2]")
    if (height is None) != (width is None):
        raise ValueError("height and width must be supplied together")
    if height is None:
        height = width = int(size)
    else:
        height, width = int(height), int(width)
    if min(height, width) < 2 or thickness < 0:
        raise ValueError("height and width must be >= 2 and thickness non-negative")
    if mode not in {"points", "segments"}:
        raise ValueError("rasterization mode must be 'points' or 'segments'")
    batch, modes, horizon, _ = routes.shape
    if horizon < 2:
        raise ValueError("routes require at least two points")
    device, dtype = routes.device, routes.dtype
    center = torch.zeros(batch * modes, height * width, device=device, dtype=dtype)
    progress_sum = torch.zeros_like(center)
    values = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype)
    scale = routes.new_tensor([(width - 1) * 0.5, (height - 1) * 0.5])
    coordinates = ((routes + 1.0) * scale).clone()
    coordinates[..., 0].clamp_(0.0, width - 1)
    coordinates[..., 1].clamp_(0.0, height - 1)
    sample_mass = torch.ones_like(coordinates[..., 0])
    values = values[None, None].expand(batch, modes, -1)
    if mode == "segments":
        if not torch.isfinite(coordinates).all():
            raise ValueError("route coordinates must be finite")
        delta = coordinates[..., 1:, :] - coordinates[..., :-1, :]
        lengths = torch.linalg.vector_norm(delta, dim=-1)
        # Padding has zero mass, so short/zero segments cannot bias progress.
        counts = torch.ceil(delta.abs().amax(dim=-1) * 2).long().clamp_min(1)
        steps = torch.arange(int(counts.max()), device=device)
        fraction = (steps.to(dtype) / counts[..., None]).clamp_max(1.0)
        sample_mass = (steps < counts[..., None]).to(dtype)
        sampled = coordinates[..., :-1, None, :] + fraction[..., None] * delta[..., None, :]
        cumulative = torch.cat([torch.zeros_like(lengths[..., :1]), lengths.cumsum(-1)], dim=-1)
        sampled_progress = (
            cumulative[..., :-1, None] + fraction * lengths[..., None]
        ) / cumulative[..., -1:, None].clamp_min(1e-6)
        coordinates = torch.cat([
            sampled.flatten(-3, -2), coordinates[..., -1:, :],
        ], dim=-2)
        values = torch.cat([
            sampled_progress.flatten(-2), torch.ones_like(values[..., -1:]),
        ], dim=-1)
        sample_mass = torch.cat([
            sample_mass.flatten(-2), torch.ones_like(values[..., -1:]),
        ], dim=-1)
        horizon = coordinates.shape[-2]
    if subpixel:
        lower = coordinates.floor().long()
        upper = lower + 1
        upper[..., 0].clamp_(max=width - 1)
        upper[..., 1].clamp_(max=height - 1)
        fraction = coordinates - lower.to(dtype)
        x0, y0 = lower[..., 0], lower[..., 1]
        x1, y1 = upper[..., 0], upper[..., 1]
        wx, wy = fraction[..., 0], fraction[..., 1]
        indices = torch.stack((
            y0 * width + x0, y0 * width + x1,
            y1 * width + x0, y1 * width + x1,
        ), dim=-1).reshape(batch * modes, horizon * 4)
        splat_weights = torch.stack((
            (1.0 - wx) * (1.0 - wy), wx * (1.0 - wy),
            (1.0 - wx) * wy, wx * wy,
        ), dim=-1) * sample_mass[..., None]
        splat_weights = splat_weights.reshape(batch * modes, horizon * 4)
        center.scatter_add_(1, indices, splat_weights)
        progress_values = values.reshape(batch * modes, horizon, 1).expand(-1, -1, 4)
        progress_sum.scatter_add_(
            1, indices, (splat_weights.reshape(batch * modes, horizon, 4)
                         * progress_values).reshape(batch * modes, horizon * 4),
        )
    else:
        xy = coordinates.round().long()
        xy[..., 0].clamp_(0, width - 1)
        xy[..., 1].clamp_(0, height - 1)
        flat = (xy[..., 1] * width + xy[..., 0]).reshape(batch * modes, horizon)
        mass = sample_mass.reshape(batch * modes, horizon)
        center.scatter_add_(1, flat, mass)
        progress_sum.scatter_add_(
            1, flat, values.reshape(batch * modes, horizon) * mass
        )
    center = center.reshape(batch * modes, 1, height, width)
    progress_sum = progress_sum.reshape_as(center)
    center_binary = (center > 0).to(dtype)

    # Average duplicate samples first.  When thickness > 0, average the
    # nearby centerline values as well, so the progress channel is dense over
    # the same corridor that the occupancy channel exposes.
    if thickness:
        kernel = 2 * int(thickness) + 1
        # Multiple trajectory samples often land in one pixel.  Occupancy is
        # a support mask, so repeated samples must not make its target exceed
        # one even though their counts are still useful for averaging progress.
        occupancy = F.max_pool2d(center_binary, kernel, stride=1, padding=thickness)
        local_mass = F.avg_pool2d(center, kernel, stride=1, padding=thickness)
        local_progress = F.avg_pool2d(progress_sum, kernel, stride=1, padding=thickness)
        progress = local_progress / local_mass.clamp_min(1e-6)
        valid = local_mass > 0
    else:
        occupancy = center_binary
        progress = progress_sum / center.clamp_min(1.0)
        valid = center > 0
    shape = (batch, modes, height, width)
    return occupancy.reshape(shape), progress.reshape(shape), valid.reshape(shape)


def ordered_field_to_path(
    occupancy: torch.Tensor,
    progress: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    *,
    beta: float = 500.0,
    power: float = 2.0,
    occupancy_threshold: float = 0.0,
    valid: torch.Tensor | None = None,
    return_valid: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Decode ordered spatial fields into a continuous path.

    ``beta`` controls how narrowly each output point selects its target
    progress.  The fallback is the straight chord only when a field has no
    support at a requested progress value; endpoints are always hard-fixed.
    """
    if occupancy.ndim != 4 or progress.shape != occupancy.shape:
        raise ValueError("occupancy and progress must have shape [B,1,H,W]")
    if start.ndim != 2 or goal.shape != start.shape or start.shape[-1] != 2:
        raise ValueError("start and goal must have shape [B,2]")
    if horizon < 2 or beta <= 0 or power <= 0:
        raise ValueError("invalid horizon, beta, or power")
    if not 0.0 <= occupancy_threshold < 1.0:
        raise ValueError("occupancy_threshold must be in [0,1)")
    batch, channels, height, width = occupancy.shape
    if channels != 1 or progress.shape[1] != 1 or len(start) != batch:
        raise ValueError("field and endpoint batch/channel mismatch")
    probability = occupancy.clamp(0.0, 1.0)
    if occupancy_threshold:
        # Suppress low-amplitude background response in learned fields while
        # preserving binary oracle occupancy exactly.
        probability = (
            (probability - occupancy_threshold) /
            (1.0 - occupancy_threshold)
        ).clamp_min(0.0)
    probability = probability.pow(power)
    if valid is not None:
        if valid.shape != occupancy.shape:
            raise ValueError("valid must match occupancy shape")
        probability = probability * valid.to(probability.dtype)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=occupancy.device, dtype=occupancy.dtype),
        torch.linspace(-1.0, 1.0, width, device=occupancy.device, dtype=occupancy.dtype),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)
    flat_probability = probability.reshape(batch, -1)
    flat_progress = progress.clamp(0.0, 1.0).reshape(batch, -1)
    centers = torch.linspace(0.0, 1.0, horizon, device=occupancy.device, dtype=occupancy.dtype)
    direction = goal - start
    path = []
    support = []
    previous = start
    for center_value in centers:
        weights = flat_probability * torch.exp(
            -float(beta) * (flat_progress - center_value).square()
        )
        mass = weights.sum(dim=-1, keepdim=True)
        estimate = (weights[:, :, None] * points).sum(dim=1) / mass.clamp_min(1e-6)
        has_support = mass.squeeze(-1) > 1e-5
        # A missing progress slice is an invalid candidate.  Holding the last
        # supported point keeps the tensor finite without inventing a straight
        # obstacle-crossing segment; callers can reject the candidate via the
        # returned support flag.
        estimate = torch.where(has_support[:, None], estimate, previous)
        path.append(estimate)
        support.append(has_support)
        previous = estimate
    path = torch.stack(path, dim=1)
    path[:, 0] = start
    path[:, -1] = goal
    path = path.clamp(-1.0, 1.0)
    support = torch.stack(support, dim=1)
    support[:, 0] = True
    support[:, -1] = True
    if return_valid:
        return path, support.all(dim=1)
    return path
