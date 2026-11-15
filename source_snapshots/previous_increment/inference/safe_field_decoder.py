"""Safety-aware decoder for ordered dense Flow fields."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def support_weighted_progress_smoothing(
    occupancy: torch.Tensor, progress: torch.Tensor, kernel_size: int = 7,
) -> torch.Tensor:
    """Smooth progress using predicted route support without creating support.

    Occupancy here is the predicted route field, not the obstacle grid. The
    caller restores its hard endpoint fields after smoothing. A zero kernel
    disables this operation for historical checkpoints.
    """
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise ValueError("progress smoothing kernel must be an integer")
    if kernel_size < 0 or (kernel_size > 0 and kernel_size % 2 == 0):
        raise ValueError("progress smoothing kernel must be zero or positive odd")
    if occupancy.ndim != 4 or occupancy.shape[1] != 1 or progress.shape != occupancy.shape:
        raise ValueError("occupancy and progress must have shape [B,1,H,W]")
    if kernel_size == 0:
        return progress
    weights = occupancy.clamp(0.0, 1.0).pow(4)
    numerator = F.avg_pool2d(
        weights * progress.clamp(0.0, 1.0), kernel_size, 1, kernel_size // 2,
    )
    denominator = F.avg_pool2d(weights, kernel_size, 1, kernel_size // 2)
    return numerator / denominator.clamp_min(1e-8)


def _progress_geometry(probability: torch.Tensor, progress: torch.Tensor):
    """Estimate route length from progress variation along supported pixels.

    Arc-length labels have |grad(progress)| ~= 1 / route_length in pixel
    coordinates. A median suppresses turns and noisy outliers. This adapts to
    short routes on large maps as well as long routes, unlike map extent
    alone. It uses only predicted fields, never labels or graph search.
    """
    support = probability > 0
    x_ok = support[..., 1:] & support[..., :-1]
    y_ok = support[..., 1:, :] & support[..., :-1, :]
    dx = F.pad((progress[..., 1:] - progress[..., :-1]) * x_ok, (0, 1))
    dy = F.pad((progress[..., 1:, :] - progress[..., :-1, :]) * y_ok, (0, 0, 0, 1))
    # A raster pixel represents a finite progress interval. In particular,
    # averaging thick labels near a turn/hard endpoint can leave no pixel
    # whose scalar value equals a requested progress slice. Midpoints to
    # supported neighbors define its bin without adding spatial support.
    neighbor_delta = torch.maximum(dx.abs(), F.pad(dx[..., :-1].abs(), (1, 0)))
    neighbor_delta = torch.maximum(neighbor_delta, dy.abs())
    neighbor_delta = torch.maximum(neighbor_delta, F.pad(dy[..., :-1, :].abs(), (0, 0, 1, 0)))
    half_bin = .5 * neighbor_delta
    gradient = torch.sqrt(dx.square() + dy.square()).flatten(1)
    gradient = gradient.masked_fill(gradient <= 1e-6, float("nan"))
    median = gradient.nanmedian(dim=1).values
    height, width = probability.shape[-2:]
    fallback = median.new_full(median.shape, 1.0 / max(height, width))
    median = torch.where(torch.isfinite(median), median, fallback)
    return (1.0 / (63.0 * median)).clamp_min(1.0).square(), half_bin.flatten(1)


def distance_field_to_soft_clearance(
    distance_field: torch.Tensor,
    distance_scale: float,
    radius: float,
    temperature: float = 0.5,
    conservative_offset: float = 0.0,
) -> torch.Tensor:
    """Convert center-distance features into bounded safety weights.

    ``cv2.distanceTransform`` measures distance between pixel centers, while
    the geometry protocol measures distance to closed cell squares. A
    half-cell diagonal offset can therefore be used as a conservative lower
    bound when this feature drives safety weighting.
    """
    if distance_scale <= 0 or temperature <= 0 or radius < 0:
        raise ValueError("invalid distance-field clearance parameters")
    if conservative_offset < 0:
        raise ValueError("conservative_offset must be non-negative")
    distance = distance_field * float(distance_scale) - float(conservative_offset)
    return torch.sigmoid((distance - float(radius)) / float(temperature))


def safe_ordered_field_to_path(
    occupancy: torch.Tensor,
    progress: torch.Tensor,
    clearance: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    *,
    beta: float = 1280.0,
    power: float = 4.0,
    clearance_power: float = 2.0,
    continuity_beta: float = 60.0,
    occupancy_threshold: float = 0.1,
    terminal_support_window: int = 8,
    terminal_support_fraction: float = 0.75,
    max_terminal_jump: float = 0.35,
    max_path_step: float | None = None,
    mode: str = "legacy",
    return_valid: bool = False,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor],
]:
    """Decode an ordered field without averaging across distant branches.

    The original decoder computes a global weighted mean for every progress
    slice.  When a field contains two obstacle-side branches, that mean can
    fall between them.  This decoder multiplies the route support by the
    predicted clearance and uses a Gaussian centered at the previous point,
    so a branch is followed locally through the path.

    ``max_terminal_jump`` and ``max_path_step`` use normalized coordinate
    units calibrated to the legacy 64x64 grid. Per-axis scaling makes those
    limits represent the same grid distances on rectangular maps.

    ``scale_aware`` tightens progress slices with estimated route length. The local
    continuity kernel is in fixed grid units; leaving progress precision
    fixed would make that kernel dominate and lag behind long routes.
    Acceptance, support and jump limits are unchanged. Old checkpoints use
    ``legacy`` unless the new mode is explicitly recorded in their config.
    """
    if occupancy.ndim != 4 or progress.shape != occupancy.shape:
        raise ValueError("occupancy and progress must have shape [B,1,H,W]")
    if clearance.shape != occupancy.shape:
        raise ValueError("clearance must match occupancy shape")
    if start.ndim != 2 or goal.shape != start.shape or start.shape[-1] != 2:
        raise ValueError("start and goal must have shape [B,2]")
    if horizon < 2 or beta <= 0 or power <= 0 or clearance_power < 0:
        raise ValueError("invalid decoder parameters")
    if continuity_beta < 0 or not 0.0 <= occupancy_threshold < 1.0:
        raise ValueError("invalid continuity or occupancy threshold")
    if terminal_support_window < 1 or not 0.0 < terminal_support_fraction <= 1.0:
        raise ValueError("invalid terminal support parameters")
    if max_terminal_jump <= 0:
        raise ValueError("max_terminal_jump must be positive")
    if max_path_step is not None and max_path_step <= 0:
        raise ValueError("max_path_step must be positive when provided")
    if mode not in {"legacy", "scale_aware"}:
        raise ValueError("decoder mode must be 'legacy' or 'scale_aware'")

    batch, channels, height, width = occupancy.shape
    if channels != 1 or len(start) != batch:
        raise ValueError("field and endpoint batch/channel mismatch")

    probability = occupancy.clamp(0.0, 1.0)
    if occupancy_threshold:
        probability = (
            (probability - occupancy_threshold) /
            (1.0 - occupancy_threshold)
        ).clamp_min(0.0)
    probability = probability.pow(power)
    safety = clearance.clamp(0.0, 1.0).pow(clearance_power)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=occupancy.device,
                       dtype=occupancy.dtype),
        torch.linspace(-1.0, 1.0, width, device=occupancy.device,
                       dtype=occupancy.dtype),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)
    metric_scale = occupancy.new_tensor([(width - 1) / 63.0, (height - 1) / 63.0])
    flat_probability = (probability * safety).reshape(batch, -1)
    flat_progress = progress.clamp(0.0, 1.0).reshape(batch, -1)
    progress_beta = occupancy.new_full((batch, 1), float(beta))
    progress_half_bin = None
    if mode == "scale_aware":
        precision_scale, progress_half_bin = _progress_geometry(probability, progress)
        progress_beta = progress_beta * precision_scale[:, None]
    centers = torch.linspace(
        0.0, 1.0, horizon, device=occupancy.device, dtype=occupancy.dtype,
    )

    path = []
    support = []
    previous = start
    for center_value in centers:
        progress_delta = flat_progress - center_value
        if progress_half_bin is not None:
            progress_delta = (progress_delta.abs() - progress_half_bin).clamp_min(0.0)
        score = flat_probability * torch.exp(-progress_beta * progress_delta.square())
        # Keep the decode near the last supported point.  This prevents a
        # global average between two obstacle-side modes.
        local_delta = (points - previous[:, None, :]) * metric_scale
        local = torch.exp(-float(continuity_beta) * local_delta.square().sum(dim=-1))
        local_weights = score * local
        local_mass = local_weights.sum(dim=-1, keepdim=True)
        has_local = local_mass.squeeze(-1) > 1e-5
        # Never fall back to a global weighted mean.  If local support
        # disappears, a global fallback can jump between obstacle-side
        # branches and create a long, unsafe segment.  Mark that candidate
        # invalid instead; the selector can choose another independent draw.
        estimate = (local_weights[:, :, None] * points).sum(dim=1) / local_mass.clamp_min(1e-6)
        has_support = has_local
        estimate = torch.where(has_support[:, None], estimate, previous)
        path.append(estimate)
        support.append(has_support)
        previous = estimate

    path = torch.stack(path, dim=1)
    path[:, 0] = start
    path[:, -1] = goal
    path = path.clamp(-1.0, 1.0)
    # Keep the raw support before hard endpoint bookkeeping.  The goal pixel
    # is fixed by construction, so setting the final support bit to True would
    # otherwise hide a missing learned approach corridor.
    support = torch.stack(support, dim=1)
    raw_support = support.clone()
    support[:, 0] = True
    support[:, -1] = True
    window = min(int(terminal_support_window), horizon)
    terminal_support = raw_support[:, -window:].sum(dim=1)
    required_support = int(torch.ceil(
        torch.tensor(window * float(terminal_support_fraction), device=support.device)
    ).item())
    terminal_support_ok = terminal_support >= required_support
    terminal_delta = (path[:, -1] - path[:, -2]) * metric_scale
    terminal_jump_ok = torch.linalg.vector_norm(terminal_delta, dim=-1) <= float(max_terminal_jump)
    if max_path_step is None:
        path_step_ok = torch.ones(batch, dtype=torch.bool, device=path.device)
    else:
        path_delta = (path[:, 1:] - path[:, :-1]) * metric_scale
        path_step_ok = torch.linalg.vector_norm(path_delta, dim=-1).amax(dim=1) <= float(max_path_step)
    valid = (
        support.all(dim=1)
        & terminal_support_ok
        & terminal_jump_ok
        & path_step_ok
    )
    if return_diagnostics:
        return path, valid, {
            "missing_support_steps": (~support).sum(dim=1),
            "first_missing_support_step": torch.where(
                ~support, torch.arange(horizon, device=support.device)[None], horizon,
            ).amin(dim=1),
            "terminal_support_ok": terminal_support_ok,
            "terminal_jump_ok": terminal_jump_ok,
            "path_step_ok": path_step_ok,
            "progress_beta": progress_beta[:, 0],
        }
    if return_valid:
        return path, valid
    return path
