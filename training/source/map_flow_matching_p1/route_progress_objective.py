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


def route_progress_multispan_cost(endpoint_estimate, routes, *, spacing_px=1.,
                                 spans_px=(4., 16., 64.), max_samples=4096):
    """Dense arc-length relative-slope Smooth-L1, one cost per paired route.

    At each start arc s, compare p(s+d)-p(s) with d/L. Dividing the residual
    by d/L keeps a plateau's cost at 0.5 regardless of route length, sample
    density, or span. Each span and each route get equal weight. Spans shorten
    to L on short routes; padded starts and zero-length routes are masked.
    The cap increases spacing on very long routes, never truncates the route.
    No chords, nearby branches, occupancy gating, clamps, or extra targets.
    """
    if (endpoint_estimate.ndim != 4 or endpoint_estimate.shape[1] != 2
            or len(endpoint_estimate) != len(routes) or routes.ndim != 3
            or routes.shape[1] < 2 or routes.shape[-1] != 2
            or spacing_px <= 0 or max_samples < 2 or not spans_px
            or any(span <= 0 for span in spans_px)):
        raise ValueError("invalid paired fields, routes, or dense multispan settings")
    height, width = endpoint_estimate.shape[-2:]
    scale = routes.new_tensor([(width - 1) / 2, (height - 1) / 2])
    lengths = torch.linalg.vector_norm(torch.diff(routes, dim=1) * scale, dim=-1)
    cumulative = torch.cat([lengths.new_zeros(len(routes), 1), lengths.cumsum(-1)], -1)
    total = cumulative[:, -1]
    counts = (torch.ceil(total / spacing_px).long() + 1).clamp(2, max_samples)
    index = torch.arange(int(counts.max().item()), device=routes.device)
    arc = total[:, None] * (index[None] / (counts[:, None] - 1)).clamp_max(1)
    spans = torch.minimum(routes.new_tensor(spans_px)[None], total[:, None])
    queries = torch.cat([arc[..., None], arc[..., None] + spans[:, None]], -1)
    queries = torch.minimum(queries, total[:, None, None])
    end = torch.searchsorted(cumulative.contiguous(), queries.flatten(1).contiguous(), right=True)
    end = end.clamp(1, routes.shape[1] - 1)
    begin = end - 1
    first = routes.gather(1, begin[..., None].expand(-1, -1, 2))
    last = routes.gather(1, end[..., None].expand(-1, -1, 2))
    fraction = (queries.flatten(1) - cumulative.gather(1, begin)) / lengths.gather(1, begin).clamp_min(1e-8)
    points = first + fraction.clamp(0, 1)[..., None] * (last - first)
    progress = F.grid_sample(endpoint_estimate[:, 1:2], points[:, None], align_corners=True)[:, 0, 0]
    progress = progress.reshape_as(queries)
    expected = spans / total[:, None].clamp_min(1e-8)
    error = (progress[..., 1:] - progress[..., :1]) / expected[:, None].clamp_min(1e-8) - 1
    mask = ((index[None, :, None] < counts[:, None, None])
            & (arc[..., None] + spans[:, None] <= total[:, None, None] + 1e-5)
            & (total[:, None, None] > 1e-6))
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
    per_span = (loss * mask).sum(1) / mask.sum(1).clamp_min(1)
    return per_span.mean(-1)
