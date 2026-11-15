"""Bounded local continuation of a predicted ordered route field.

This experimental decoder never queries labels or searches the obstacle grid.
It separates an internal supported trace from the requested output point count.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from safe_field_decoder import _progress_geometry


def _segment_supported(occupancy, clearance, begin, end, threshold, samples=17):
    """Check short internal moves, including their interiors, in pixel units."""
    height, width = occupancy.shape[-2:]
    t = torch.linspace(0, 1, samples, device=begin.device, dtype=begin.dtype)
    xy = begin[:, None, None] + (end - begin[:, None])[:, :, None] * t[None, None, :, None]
    scale = xy.new_tensor([(width - 1) / 2, (height - 1) / 2])
    grid = xy / scale - 1
    route = F.grid_sample(occupancy, grid, align_corners=True)[:, 0]
    safe = F.grid_sample(clearance, grid, align_corners=True)[:, 0]
    return ((route > threshold) & (safe >= .5)).all(-1)


def _rdp_indices(points, tolerance=.1):
    """Retain vertices with a bounded geometric error; callers check shortcuts."""
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue
        delta = points[last] - points[first]
        length2 = float(delta @ delta)
        section = points[first + 1:last]
        t = np.clip(((section - points[first]) @ delta) / max(length2, 1e-12), 0, 1)
        error = np.linalg.norm(section - (points[first] + t[:, None] * delta), axis=1)
        index = int(error.argmax())
        if error[index] > tolerance:
            middle = first + 1 + index
            keep.add(middle)
            pending.extend(((first, middle), (middle, last)))
    return sorted(keep)


def _fixed_vertices(trace, horizon):
    """Keep all simplified turns and subdivide their existing segments.

    Uniform arc-length interpolation can silently omit a corner. This method
    retains every selected vertex and rejects a trace exceeding the budget.
    """
    trace = np.asarray(trace, dtype=np.float64)
    trace = trace[np.r_[True, np.linalg.norm(np.diff(trace[:, :2], axis=0), axis=1) > 1e-7]]
    if len(trace) == 1:
        return np.repeat(trace, horizon, axis=0), True
    vertices = trace[_rdp_indices(trace[:, :2])]
    if len(vertices) > horizon:
        return np.repeat(trace[:1], horizon, axis=0), False
    lengths = np.linalg.norm(np.diff(vertices[:, :2], axis=0), axis=1)
    extra = horizon - len(vertices)
    quota = extra * lengths / max(float(lengths.sum()), 1e-12)
    additions = np.floor(quota).astype(int)
    remainder = extra - int(additions.sum())
    additions[np.argsort(-(quota - additions), kind="stable")[:remainder]] += 1
    output = [vertices[0]]
    for index, count in enumerate(additions):
        for part in range(1, int(count) + 2):
            output.append(vertices[index] + (vertices[index + 1] - vertices[index]) * part / (count + 1))
    return np.asarray(output), True


def _output_supported(occupancy, progress, clearance, paths_px, output_arc, profiles, threshold,
                      power, clearance_power, progress_beta, halfbin, continuity_beta):
    """Recheck final segments with the decoder's *joint local mass* criterion.

    A local weighted mean need not itself fall on a high-occupancy raster
    center. Requiring its interpolated pixel to exceed the occupancy threshold
    is a different rule and rejects supported baseline turns. Instead, carry
    the trace's target progress through simplification and check the original
    joint support threshold at every half-pixel sample, plus clearance.
    """
    height, width = occupancy.shape[-2:]
    result = []
    for index, path in enumerate(paths_px):
        path = path.detach().cpu().numpy()
        sections, targets = [], []
        arc = output_arc[index].detach().cpu().numpy()
        for step, (begin, end) in enumerate(zip(path[:-1], path[1:])):
            count = max(1, int(np.ceil(np.linalg.norm(end - begin) / .5)))
            t = np.linspace(0, 1, count + 1, dtype=np.float32)
            sections.append(begin + t[:, None] * (end - begin))
            sample_arc = arc[step] + t * (arc[step + 1] - arc[step])
            targets.append(np.interp(sample_arc, profiles[index][0], profiles[index][1]).astype(np.float32))
        xy = torch.as_tensor(np.concatenate(sections), device=occupancy.device, dtype=occupancy.dtype)
        scale = xy.new_tensor([(width - 1) / 2, (height - 1) / 2])
        grid = (xy / scale - 1)[None, None]
        safe = F.grid_sample(clearance[index:index + 1], grid, align_corners=True)
        target = torch.as_tensor(np.concatenate(targets), device=xy.device, dtype=xy.dtype)
        base = (((occupancy[index, 0].clamp(0, 1) - threshold) / (1 - threshold)).clamp_min(0).pow(power)
                * clearance[index, 0].clamp(0, 1).pow(clearance_power)).flatten()
        p = progress[index, 0].clamp(0, 1).flatten()
        yy, xx = torch.meshgrid(torch.arange(-16, 17, device=xy.device), torch.arange(-16, 17, device=xy.device), indexing="ij")
        offsets = torch.stack([xx, yy], -1).reshape(1, -1, 2)
        supported = (safe >= .5).all()
        for first in range(0, len(xy), 256):
            centers = xy[first:first + 256]
            local_xy = centers.floor().long()[:, None] + offsets
            inside = ((local_xy[..., 0] >= 0) & (local_xy[..., 0] < width)
                      & (local_xy[..., 1] >= 0) & (local_xy[..., 1] < height))
            indices = local_xy[..., 1].clamp(0, height - 1) * width + local_xy[..., 0].clamp(0, width - 1)
            delta = ((p[indices] - target[first:first + 256, None]).abs() - halfbin[index][indices]).clamp_min(0)
            distance2 = ((local_xy - centers[:, None]) * (2 / 63)).square().sum(-1)
            mass = (base[indices] * inside * torch.exp(-progress_beta[index] * delta.square() - continuity_beta * distance2)).sum(-1)
            supported &= (mass > 1e-5).all()
        result.append(supported)
    return torch.stack(result)


@torch.no_grad()
def adaptive_ordered_field_to_path(
    occupancy, progress, clearance, start, goal, horizon, *, beta=1280.,
    power=4., clearance_power=2., continuity_beta=60., occupancy_threshold=.1,
    terminal_support_window=8, terminal_support_fraction=.75,
    max_terminal_jump=.5, max_path_step=None, max_internal_steps=2048,
    internal_step_px=4., stagnation_steps=32, seed_paths=None, seed_lengths=None,
):
    """Continue locally using supported progress, with seven backoff increments.

    A hard four-pixel internal move limit and segment support/clearance checks
    supplement the original local mass threshold and final jump limits. A
    final link is attempted only within three pixels after actual continuation;
    a distant forced-goal chord is never subdivided into an accepted path.

    Failure codes: 0 complete, 1 no supported move, 2 stagnation, 3 step budget,
    4 output vertex budget, 5 final support/clearance check.
    """
    if occupancy.ndim != 4 or occupancy.shape[1] != 1 or progress.shape != occupancy.shape or clearance.shape != occupancy.shape:
        raise ValueError("fields must have shape [B,1,H,W]")
    batch, _, height, width = occupancy.shape
    if start.shape != (batch, 2) or goal.shape != start.shape or min(height, width) < 2:
        raise ValueError("invalid endpoints or map shape")
    if horizon < 2 or beta <= 0 or power <= 0 or clearance_power < 0 or continuity_beta < 0:
        raise ValueError("invalid decoder parameters")
    if not 0 <= occupancy_threshold < 1 or max_internal_steps < 1 or not 0 < internal_step_px <= 4 or stagnation_steps < 1:
        raise ValueError("invalid adaptive bounds")
    if terminal_support_window < 1 or not 0 < terminal_support_fraction <= 1 or max_terminal_jump <= 0:
        raise ValueError("invalid terminal parameters")
    if max_path_step is not None and max_path_step <= 0:
        raise ValueError("max_path_step must be positive")
    probability = ((occupancy.clamp(0, 1) - occupancy_threshold) / (1 - occupancy_threshold)).clamp_min(0).pow(power)
    precision, halfbin = _progress_geometry(probability, progress)
    progress_beta = beta * precision
    base = (probability * clearance.clamp(0, 1).pow(clearance_power)).flatten(1)
    p = progress.clamp(0, 1).flatten(1)
    scale = start.new_tensor([(width - 1) / 2, (height - 1) / 2])
    seed_px = None
    if seed_paths is not None:
        if seed_paths.shape != (batch, horizon, 2) or seed_lengths is None or seed_lengths.shape != (batch,):
            raise ValueError("seed_paths and seed_lengths must match the decode batch")
        if (seed_lengths < 1).any() or (seed_lengths >= horizon).any() or not torch.equal(seed_paths[:, 0], start):
            raise ValueError("seed must start exactly at start and exclude the forced goal")
        seed_px = (seed_paths + 1) * scale
        previous = seed_px[torch.arange(batch, device=start.device), seed_lengths - 1]
    else:
        previous = (start + 1) * scale
    goal_px = (goal + 1) * scale
    phase = start.new_zeros(batch) if seed_px is None else (seed_lengths - 1).to(start.dtype) / (horizon - 1)
    radius = 16
    yy, xx = torch.meshgrid(torch.arange(-radius, radius + 1, device=start.device),
                            torch.arange(-radius, radius + 1, device=start.device), indexing="ij")
    offsets = torch.stack([xx, yy], -1).reshape(1, -1, 2)
    backoff = .5 ** torch.arange(7, device=start.device, dtype=start.dtype)
    increment = 1 / (horizon - 1)
    active = torch.ones(batch, device=start.device, dtype=torch.bool)
    complete = torch.zeros_like(active)
    failure = torch.full((batch,), 3, device=start.device, dtype=torch.long)
    stagnant = torch.zeros_like(failure)
    iterations = torch.zeros_like(failure)
    backoffs = torch.zeros_like(failure)
    trace = [torch.cat([previous, phase[:, None]], -1)]
    trace_active = []
    for _ in range(max_internal_steps):
        near = (torch.linalg.vector_norm(previous - goal_px, dim=-1) <= min(3., max_terminal_jump * 63 / 2)) & active
        if near.any():
            link = _segment_supported(occupancy, clearance, previous, goal_px[:, None], occupancy_threshold)[:, 0]
            reached = near & link
            complete |= reached
            failure[reached] = 0
            active &= ~reached
        if not active.any():
            break
        xy = previous.floor().long()[:, None] + offsets
        inside = (xy[..., 0] >= 0) & (xy[..., 0] < width) & (xy[..., 1] >= 0) & (xy[..., 1] < height)
        indices = xy[..., 1].clamp(0, height - 1) * width + xy[..., 0].clamp(0, width - 1)
        local_base = base.gather(1, indices) * inside
        local_p = p.gather(1, indices)
        local_bin = halfbin.gather(1, indices)
        distance2 = ((xy - previous[:, None]) * (2 / 63)).square().sum(-1)
        local_base = local_base * torch.exp(-continuity_beta * distance2)
        targets = (phase[:, None] + increment * backoff).clamp_max(1)
        delta = ((local_p[:, None] - targets[..., None]).abs() - local_bin[:, None]).clamp_min(0)
        weights = local_base[:, None] * torch.exp(-progress_beta[:, None, None] * delta.square())
        # Once progress reaches its ceiling, repeating a weighted mean can
        # settle before the endpoint. A weak local goal potential continues
        # along the supported tail. Every resulting move is still bounded
        # and checked, including its interior; a missing tail cannot be filled.
        to_goal = goal_px - previous
        goal_distance = torch.linalg.vector_norm(to_goal, dim=-1)
        tail = (targets >= 1) & (goal_distance[:, None] <= 2 * radius)
        direction = to_goal / goal_distance[:, None].clamp_min(1e-6)
        attraction = .25 * ((xy - previous[:, None]) * direction[:, None]).sum(-1)
        weights *= torch.exp(torch.where(tail[..., None], attraction[:, None], 0.))
        mass = weights.sum(-1)
        estimate = (weights[..., None] * xy[:, None]).sum(-2) / mass[..., None].clamp_min(1e-6)
        movement = torch.linalg.vector_norm(estimate - previous[:, None], dim=-1)
        eligible = (mass > 1e-5) & (movement <= internal_step_px)
        eligible &= _segment_supported(occupancy, clearance, previous, estimate, occupancy_threshold)
        available = eligible.any(-1) & active
        choice = eligible.long().argmax(-1)
        arange = torch.arange(batch, device=start.device)
        new_point = estimate[arange, choice]
        new_phase = targets[arange, choice]
        slow = (movement[arange, choice] < .02) & ((new_phase - phase) < 1e-6)
        stagnant = torch.where(slow & available, stagnant + 1, torch.zeros_like(stagnant))
        stalled = (stagnant >= stagnation_steps) & active
        failure[active & ~available] = 1
        failure[stalled] = 2
        iterations += active.long()
        backoffs += (choice > 0).long() * available
        active &= available & ~stalled
        previous = torch.where(active[:, None], new_point, previous)
        phase = torch.where(active, new_phase, phase)
        trace.append(torch.cat([previous, phase[:, None]], -1))
        trace_active.append(active.clone())
    # The endpoint is still exact for rejected candidates, but its forced link
    # cannot affect completion or acceptance.
    trace_array = torch.stack(trace, 1).cpu().numpy()
    trace_mask = torch.stack(trace_active, 1).cpu().numpy() if trace_active else np.empty((batch, 0), bool)
    outputs, vertex_ok, profiles = [], [], []
    for index in range(batch):
        useful = trace_array[index, np.r_[True, trace_mask[index]]]
        if seed_px is not None:
            length = int(seed_lengths[index])
            seed = np.column_stack([seed_px[index, :length].cpu().numpy(), np.arange(length) / (horizon - 1)])
            useful = np.concatenate([seed, useful[1:]])
        endpoint = np.r_[goal_px[index].cpu().numpy(), 1.][None]
        useful = np.concatenate([useful, endpoint])
        arc = np.r_[0., np.linalg.norm(np.diff(useful[:, :2], axis=0), axis=1).cumsum()]
        profiles.append((arc, useful[:, 2]))
        output, fits = _fixed_vertices(np.column_stack([useful[:, :2], arc]), horizon)
        outputs.append(output)
        vertex_ok.append(fits)
    output = torch.as_tensor(np.stack(outputs), device=start.device, dtype=start.dtype)
    output_px, output_arc = output[..., :2], output[..., 2]
    paths = (output_px / scale - 1).clamp(-1, 1)
    paths[:, 0], paths[:, -1] = start, goal
    vertex_ok = torch.tensor(vertex_ok, device=start.device)
    # Check the actual final float32 outputs, after endpoint restoration.
    output_px = (paths + 1) * scale
    output_ok = _output_supported(occupancy, progress, clearance, output_px, output_arc, profiles,
                                  occupancy_threshold, power, clearance_power,
                                  progress_beta, halfbin, continuity_beta)
    terminal_ok = torch.linalg.vector_norm((paths[:, -1] - paths[:, -2]) * scale, dim=-1) <= max_terminal_jump * 63 / 2
    step_ok = torch.ones_like(complete)
    if max_path_step is not None:
        step_ok = torch.linalg.vector_norm(torch.diff(paths, dim=1) * (2 * scale / 63), dim=-1).amax(-1) <= max_path_step
    valid = complete & vertex_ok & output_ok & terminal_ok & step_ok
    failure[complete & ~vertex_ok] = 4
    failure[complete & vertex_ok & ~output_ok] = 5
    return paths, valid, {
        "missing_support_steps": (failure == 1).long(),
        "first_missing_support_step": torch.where(failure == 1, iterations, torch.full_like(iterations, horizon)),
        "terminal_support_ok": complete & output_ok,
        "terminal_jump_ok": terminal_ok & complete,
        "path_step_ok": step_ok,
        "progress_beta": progress_beta,
        "adaptive_internal_steps": iterations,
        "adaptive_backoffs": backoffs,
        "adaptive_final_progress": phase,
        "adaptive_remaining_distance_px": torch.linalg.vector_norm(previous - goal_px, dim=-1),
        "adaptive_complete": complete,
        "adaptive_output_supported": output_ok,
        "adaptive_failure_code": failure,
    }
