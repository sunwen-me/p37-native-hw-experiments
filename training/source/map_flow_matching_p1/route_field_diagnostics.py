"""Label-conditioned diagnostics; never used for decoding or acceptance."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def longest_mask_span(mask, arc):
    """Measure runs using midpoint cells, including half-cells at endpoints."""
    boundaries = np.r_[arc[0], (arc[:-1] + arc[1:]) / 2, arc[-1]]
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    begins, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return float(np.max(boundaries[ends] - boundaries[begins], initial=0.))


def route_field_diagnostics(route_px, fields, *, spacing_px=.5, support_threshold=.1):
    """Sample each channel on the exact polyline in pixel arc length.

    Reference-mode mismatch is possible: a predicted alternate safe route may
    differ. These metrics explain channels; geometry determines success.
    """
    route = np.asarray(route_px, dtype=np.float64)
    fields = np.asarray(fields)
    if (route.ndim != 2 or route.shape[1] != 2 or len(route) < 2
            or fields.ndim != 4 or fields.shape[1] != 2 or spacing_px <= 0
            or not np.isfinite(route).all() or not np.isfinite(fields).all()):
        raise ValueError("expected finite pixel route and [K,2,H,W] fields")
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cumulative = np.r_[0., lengths.cumsum()]
    length = float(cumulative[-1])
    if length <= 1e-6:
        return {"valid": False, "length_px": length, "candidates": []}
    keep = np.r_[True, lengths > 1e-8]
    arc = np.linspace(0, length, max(2, int(np.ceil(length / spacing_px)) + 1))
    xy = np.stack([np.interp(arc, cumulative[keep], route[keep, axis]) for axis in (0, 1)], -1)
    weights = np.diff(np.r_[0., (arc[:-1] + arc[1:]) / 2, length])
    rows = []
    for field in fields:
        occupancy, progress = [map_coordinates(channel, [xy[:, 1], xy[:, 0]], order=1,
                                                mode="nearest").astype(np.float64) for channel in field]
        drawdown = np.maximum.accumulate(progress) - progress
        supported = occupancy > support_threshold
        span = min(16., length)
        starts = arc[arc <= length - span + 1e-8]
        advance = np.interp(starts + span, arc, progress) - np.interp(starts, arc, progress)
        low = advance <= .1 * span / length
        rows.append({
            "max_cumulative_progress_drawdown": float(drawdown.max()),
            "max_drawdown_on_support": float(drawdown[supported].max(initial=0.)),
            "longest_below_previous_peak_px": longest_mask_span(drawdown > 1e-6, arc),
            "low_advance_16px_fraction": float(low.mean()),
            "longest_low_advance_window_start_run_px": longest_mask_span(low, starts),
            "support_coverage_length_fraction": float(weights[supported].sum() / length),
            "missing_support_total_px": float(weights[~supported].sum()),
            "longest_missing_support_gap_px": longest_mask_span(~supported, arc),
        })
    return {"valid": True, "length_px": length, "samples": len(arc),
            "spacing_px": float(arc[1] - arc[0]), "support_threshold": support_threshold,
            "low_advance_window_px": min(16., length), "low_advance_relative_slope_threshold": .1,
            "candidates": rows}
