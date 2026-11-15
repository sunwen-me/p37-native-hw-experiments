"""Balanced Sinkhorn matching used by the P37 set-valued CFM loss."""

from __future__ import annotations

import torch


def sinkhorn_plan(cost: torch.Tensor, epsilon: float = 0.1,
                  iterations: int = 30, column_mass: torch.Tensor | None = None) -> torch.Tensor:
    """Return a soft matching plan with uniform rows and chosen columns."""
    if cost.ndim != 3 or min(cost.shape) < 1 or not torch.isfinite(cost).all():
        raise ValueError("cost must be finite with shape [B,R,T]")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    batch, routes, teachers = cost.shape
    log_kernel = -cost / epsilon
    row_mass = cost.new_full((batch, routes), 1.0 / routes)
    if column_mass is None:
        column_mass = cost.new_full((batch, teachers), 1.0 / teachers)
    if (column_mass.shape != (batch, teachers) or not torch.isfinite(column_mass).all()
            or (column_mass <= 0).any()):
        raise ValueError("column_mass must be finite and positive with shape [B,T]")
    column_mass = column_mass / column_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_row_mass = row_mass.log()
    log_column_mass = column_mass.log()
    log_u = torch.zeros_like(log_row_mass)
    log_v = torch.zeros_like(log_column_mass)
    for _ in range(iterations):
        log_u = log_row_mass - torch.logsumexp(
            log_kernel + log_v[:, None, :], dim=2
        )
        log_v = log_column_mass - torch.logsumexp(
            log_kernel + log_u[:, :, None], dim=1
        )
    return torch.exp(log_u[:, :, None] + log_kernel + log_v[:, None, :])
