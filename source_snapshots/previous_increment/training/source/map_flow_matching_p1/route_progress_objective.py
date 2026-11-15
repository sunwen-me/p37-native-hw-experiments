"""Paired-route progress increments on a linear-CFM endpoint estimate.

Samples follow each teacher polyline in pixel arc length. Nearby branches
across walls are never paired by spatial proximity. This objective is only a
training hypothesis; the endpoint estimate is not an integrated Flow sample.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def arc_length_samples(routes, height, width, count=128):
    if routes.ndim != 3 or routes.shape[-1] != 2 or routes.shape[1] < 2 or count < 2:
        raise ValueError("expected [N,L,2] polylines and at least two samples")
    scale = routes.new_tensor([(width - 1) / 2, (height - 1) / 2])
    lengths = torch.linalg.vector_norm(torch.diff(routes, dim=1) * scale, dim=-1)
    cumulative = torch.cat([lengths.new_zeros(len(routes), 1), lengths.cumsum(-1)], -1)
    phase = torch.linspace(0, 1, count, device=routes.device, dtype=routes.dtype)
    requested = cumulative[:, -1:] * phase
    end = torch.searchsorted(cumulative.contiguous(), requested.contiguous(), right=True)
    end = end.clamp(1, routes.shape[1] - 1)
    begin = end - 1
    first = routes.gather(1, begin[..., None].expand(-1, -1, 2))
    last = routes.gather(1, end[..., None].expand(-1, -1, 2))
    fraction = (requested - cumulative.gather(1, begin)) / lengths.gather(1, begin).clamp_min(1e-8)
    points = first + fraction.clamp(0, 1)[..., None] * (last - first)
    return points, phase, cumulative[:, -1] > 1e-6


def route_progress_increment_cost(endpoint_estimate, routes, *, samples=128, scale=32.):
    """Return one paired cost per estimated field, retaining raw gradients.

    A factor of 32 makes a 1/127 phase increment numerically visible without
    dividing by tiny pixel displacements. Smooth-L1 limits outlier gradients.
    The caller supplies x_t + (1-t)*v, never velocity as terminal progress.
    """
    if (endpoint_estimate.ndim != 4 or endpoint_estimate.shape[1] != 2
            or len(endpoint_estimate) != len(routes) or scale <= 0):
        raise ValueError("endpoint fields and paired routes must match")
    points, phase, valid = arc_length_samples(routes, *endpoint_estimate.shape[-2:], samples)
    progress = F.grid_sample(endpoint_estimate[:, 1:2], points[:, None], align_corners=True)[:, 0, 0]
    error = (torch.diff(progress, dim=-1) - torch.diff(phase)) * scale
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean(-1)
    return loss * valid
