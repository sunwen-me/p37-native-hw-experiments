"""Constrained Flow policy primitives shared by P51 training and evaluation.

The base ordered-field generators are frozen.  A single residual policy is
trained against a mixture of generators, so the learned correction cannot
silently specialize to one checkpoint.  Safety is applied at every Flow-SDE
transition through a differentiable barrier projection built from the input
clearance channel.  The exact geometry checker remains an offline label and
evaluation tool; it is never called by this module.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from p37_model import endpoint_fields, flow_sde_transition


class ResidualVelocityPolicy(nn.Module):
    """Small shared residual field conditioned on the active base generator."""

    def __init__(self, hidden_dim: int = 32, residual_scale: float = 0.15):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be >= 8")
        self.hidden_dim = hidden_dim
        self.residual_scale = float(residual_scale)
        # x, map occupancy/distance, base velocity, endpoint occupancy/progress,
        # and a scalar Flow time channel.
        self.net = nn.Sequential(
            nn.Conv2d(9, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        map_input: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        t: torch.Tensor,
        base_velocity: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("x must have shape [B,2,H,W]")
        if map_input.shape[:1] != x.shape[:1] or map_input.shape[1] != 2:
            raise ValueError("map_input must have shape [B,2,H,W]")
        fixed = endpoint_fields(start, goal, x.shape[-1]).to(dtype=x.dtype)
        time = t.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        features = torch.cat([x, map_input, base_velocity, fixed, time], dim=1)
        return self.residual_scale * self.net(features)


class SafetyCostCritic(nn.Module):
    """Task-level baseline for the sampled safety cost.

    P51 uses bounded terminal costs and keeps the historical sigmoid output.
    P52 also includes discounted dense costs, so it requests an unbounded head
    to avoid silently clipping a valid cost-to-go target at one.
    """

    def __init__(self, hidden_dim: int = 32, bounded: bool = True):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.bounded = bool(bounded)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.GroupNorm(min(8, hidden_dim * 2), hidden_dim * 2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        head_layers = [
            nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        ]
        if self.bounded:
            head_layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*head_layers)

    def forward(self, map_input: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(map_input).flatten(1)
        return self.head(torch.cat([encoded, start, goal], dim=1)).squeeze(1)


class FlowSafetyCostCritic(nn.Module):
    """State-conditioned cost-to-go critic for full-Flow PPO.

    Unlike the task-level ``SafetyCostCritic``, this critic sees the current
    dense field and Flow time.  It therefore produces a separate baseline for
    each candidate and each transition, which is the quantity required by
    P52's per-step cost advantages.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be >= 8")
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(7, hidden_dim, 5, stride=2, padding=2),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.GroupNorm(min(8, hidden_dim * 2), hidden_dim * 2), nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.GroupNorm(min(8, hidden_dim * 4), hidden_dim * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 5, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(
        self,
        field: torch.Tensor,
        map_input: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if field.ndim != 4 or field.shape[1] != 2:
            raise ValueError("field must have shape [B,2,H,W]")
        if map_input.shape != field.shape or map_input.shape[1] != 2:
            raise ValueError("map_input must match field shape [B,2,H,W]")
        if time.ndim != 1 or len(time) != len(field):
            raise ValueError("time must have shape [B]")
        fixed = endpoint_fields(start, goal, field.shape[-1]).to(dtype=field.dtype)
        time_map = time[:, None, None, None].expand(
            -1, 1, field.shape[-2], field.shape[-1],
        )
        encoded = self.encoder(
            torch.cat([field, map_input, fixed, time_map], dim=1)
        ).flatten(1)
        return self.head(torch.cat([encoded, start, goal, time[:, None]], dim=1)).squeeze(1)


def barrier_penalty(
    x: torch.Tensor,
    map_input: torch.Tensor,
    robot_radius_grid: float,
    distance_scale: float,
    clearance_offset: float,
    margin: float,
    temperature: float,
    support_threshold: float = 0.5,
) -> torch.Tensor:
    """Return a differentiable occupancy mass inside the unsafe clearance band."""
    if temperature <= 0 or distance_scale <= 0:
        raise ValueError("temperature and distance_scale must be positive")
    clearance = map_input[:, 1:2] * float(distance_scale) - float(clearance_offset)
    unsafe_probability = torch.sigmoid(
        (float(robot_radius_grid) + float(margin) - clearance) / float(temperature)
    )
    occupancy_probability = x[:, :1].sigmoid()
    route_support = (occupancy_probability - float(support_threshold)).clamp_min(0.0)
    # Keep the quantity normalized per map so projection strength is independent
    # of the number of pixels.
    return (unsafe_probability * route_support.square()).flatten(1).mean(dim=1)


def project_velocity_with_barrier(
    x: torch.Tensor,
    velocity: torch.Tensor,
    map_input: torch.Tensor,
    endpoint: torch.Tensor,
    robot_radius_grid: float,
    distance_scale: float,
    clearance_offset: float,
    margin: float = 0.25,
    temperature: float = 0.25,
    support_threshold: float = 0.5,
    gain: float = 2.0,
    max_projection: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove only the inward velocity component of the field barrier.

    The previous implementation subtracted a fixed-size correction whenever
    unsafe field mass existed, even when the current velocity was already
    moving away from it.  That made the barrier an unconditional drift and
    could damage otherwise safe trajectories.  We now project onto the
    half-space ``<velocity, grad(barrier)> <= 0`` and cap the correction.
    This remains a field-level heuristic; decoded path geometry is checked
    separately by the offline terminal cost.
    """
    if gain < 0 or max_projection < 0:
        raise ValueError("gain and max_projection must be non-negative")
    if x.shape != velocity.shape or endpoint.shape[:1] != x.shape[:1]:
        raise ValueError("x, velocity and endpoint shapes do not match")
    if gain == 0 or max_projection == 0:
        return velocity, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    # The barrier is ``mean(sigmoid(clearance risk) * route_support**2)``.
    # Its occupancy derivative is closed form, so there is no reason to build
    # an autograd graph for every 64x64 field at every Flow step.
    clearance = map_input[:, 1:2] * float(distance_scale) - float(clearance_offset)
    unsafe_probability = torch.sigmoid(
        (float(robot_radius_grid) + float(margin) - clearance) / float(temperature)
    )
    occupancy_probability = x[:, :1].detach().sigmoid()
    route_support = (occupancy_probability - float(support_threshold)).clamp_min(0.0)
    penalty_values = (unsafe_probability * route_support.square()).flatten(1).mean(dim=1)
    gradient = torch.zeros_like(x)
    gradient[:, :1] = (
        2.0 * unsafe_probability * route_support * occupancy_probability
        * (1.0 - occupancy_probability)
        / float(x.shape[-1] * x.shape[-2])
    )
    gradient = gradient * (1.0 - endpoint)
    norm = torch.linalg.vector_norm(gradient.flatten(1), dim=1).clamp_min(1e-8)
    normal = gradient / norm[:, None, None, None]
    # A barrier should not fight a velocity that already reduces its value.
    inward = (velocity * normal).flatten(1).sum(dim=1).clamp_min(0.0)
    correction_norm = (float(gain) * inward).clamp(max=float(max_projection))
    correction = normal * correction_norm[:, None, None, None]
    return velocity - correction, penalty_values.detach()


def grouped_base_velocity(
    base_models: Sequence[nn.Module],
    generator_ids: torch.Tensor,
    x: torch.Tensor,
    map_input: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    t: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Evaluate frozen generators in groups while preserving candidate order."""
    if generator_ids.ndim != 1 or len(generator_ids) != len(x):
        raise ValueError("generator_ids must have shape [B]")
    if not len(base_models):
        raise ValueError("at least one base generator is required")
    output = torch.empty_like(x)
    for index in torch.unique(generator_ids).tolist():
        index = int(index)
        if index < 0 or index >= len(base_models):
            raise ValueError("generator id is out of range")
        rows = torch.nonzero(generator_ids == index, as_tuple=False).flatten()
        with torch.no_grad():
            output[rows] = base_models[index].vector_field(
                x[rows], map_input[rows], start[rows], goal[rows], t[rows], latent[rows]
            )
    return output


def constrained_transition(
    x: torch.Tensor,
    map_input: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    latent: torch.Tensor,
    generator_ids: torch.Tensor,
    base_models: Sequence[nn.Module],
    residual_policy: ResidualVelocityPolicy,
    step: int,
    num_steps: int,
    noise_level: float,
    robot_radius_grid: float,
    distance_scale: float,
    clearance_offset: float,
    barrier_margin: float,
    barrier_temperature: float,
    barrier_support_threshold: float,
    barrier_gain: float,
    barrier_max_projection: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One policy transition with residual correction and barrier projection."""
    fixed = endpoint_fields(start, goal, x.shape[-1]).to(dtype=x.dtype)
    endpoint = fixed[:, :1]
    t = torch.full((len(x),), step / float(num_steps), device=x.device, dtype=x.dtype)
    base_velocity = grouped_base_velocity(
        base_models, generator_ids, x, map_input, start, goal, t, latent,
    )
    velocity = base_velocity + residual_policy(
        x, map_input, start, goal, t, base_velocity
    )
    velocity, barrier = project_velocity_with_barrier(
        x, velocity, map_input, endpoint, robot_radius_grid, distance_scale,
        clearance_offset, margin=barrier_margin,
        temperature=barrier_temperature,
        support_threshold=barrier_support_threshold,
        gain=barrier_gain, max_projection=barrier_max_projection,
    )
    next_x, mean, std = flow_sde_transition(
        x, velocity, step, num_steps, noise_level, endpoint, fixed, generator,
    )
    return next_x, mean, std, barrier
