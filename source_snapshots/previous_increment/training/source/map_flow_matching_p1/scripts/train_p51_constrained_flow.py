"""P51: constrained, generator-agnostic residual Flow policy.

This is a new RL baseline and does not overwrite P43.  Base P37/P50/P48
generators are frozen and a shared residual velocity policy is optimized with
PPO under an explicit expected safety-cost constraint:

    maximize E[R] subject to E[C] <= epsilon.

The implementation contains three auditable pieces that P43 lacked:

* a task-level safety cost critic and a dual Lagrange multiplier;
* a barrier projection on every Flow-SDE transition;
* generator-mixture rollouts and per-generator validation.

Exact geometry is used only for offline terminal labels/checkpoint selection.
Deployment uses the learned residual, the frozen input distance channel, and
the same barrier projection; it does not call a geometry checker or planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from constrained_flow import (  # noqa: E402
    ResidualVelocityPolicy,
    SafetyCostCritic,
    constrained_transition,
)
from p37_model import candidate_selection_scores, endpoint_fields  # noqa: E402
from safe_field_decoder import distance_field_to_soft_clearance  # noqa: E402
from scripts.train_p37_flow_sde_ppo import (  # noqa: E402
    compute_task_advantages,
    decode_fields,
    load_model,
    terminal_rewards,
    transition_logprob,
)


def normalize_cost_advantage(cost: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Convert candidate costs into a centered cost advantage."""
    if cost.ndim != 2 or baseline.ndim != 1 or len(cost) != len(baseline):
        raise ValueError("cost must be [tasks,candidates] and baseline [tasks]")
    advantage = cost - baseline[:, None].detach()
    flat = advantage.reshape(-1)
    return (flat - flat.mean()) / flat.std(unbiased=False).clamp_min(1e-4)


def constrained_ppo_loss(
    ratio: torch.Tensor,
    reward_advantage: torch.Tensor,
    cost_advantage: torch.Tensor,
    clip_eps: float,
    lagrange_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return PPO loss, reward surrogate and conservative cost surrogate.

    The cost term uses ``maximum`` so the Lagrangian update does not exploit a
    favorable clipping branch to hide a positive safety-cost increase.
    """
    if ratio.shape != reward_advantage.shape or ratio.shape != cost_advantage.shape:
        raise ValueError("ratio and advantages must have identical shapes")
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    reward_surrogate = torch.minimum(
        ratio * reward_advantage, clipped * reward_advantage,
    ).mean()
    cost_surrogate = torch.maximum(
        ratio * cost_advantage, clipped * cost_advantage,
    ).mean()
    loss = -reward_surrogate + float(lagrange_multiplier) * cost_surrogate
    return loss, reward_surrogate, cost_surrogate


def update_lagrange_multiplier(
    multiplier: float, mean_cost: float, epsilon: float, step_size: float,
    maximum: float,
) -> float:
    """Projected dual ascent for ``E[C] <= epsilon``."""
    if multiplier < 0 or epsilon < 0 or step_size < 0 or maximum <= 0:
        raise ValueError("invalid Lagrange parameters")
    return float(np.clip(
        multiplier + float(step_size) * (float(mean_cost) - float(epsilon)),
        0.0, float(maximum),
    ))


def rollout_constrained(
    base_models: Sequence[nn.Module],
    residual_policy: ResidualVelocityPolicy,
    maps: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    candidates: int,
    num_steps: int,
    noise_scale: float,
    sde_noise: float,
    barrier_config: dict[str, float],
    generator: torch.Generator,
    generator_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Roll out independent candidates, optionally choosing a base generator."""
    if candidates < 1 or num_steps < 1:
        raise ValueError("candidates and num_steps must be positive")
    batch = len(start)
    flat_maps = maps[:, None].expand(-1, candidates, -1, -1, -1).reshape(
        batch * candidates, *maps.shape[1:]
    )
    flat_start = start[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    flat_goal = goal[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    if generator_ids is None:
        # Use a balanced mixture instead of hoping a tiny PPO batch samples
        # every generator.  A random permutation keeps the assignment
        # independent of candidate slot while guaranteeing coverage.
        total = batch * candidates
        assignments = torch.arange(total, device=maps.device) % len(base_models)
        permutation = torch.randperm(total, device=maps.device, generator=generator)
        generator_ids = assignments[permutation].reshape(batch, candidates)
    if generator_ids.shape != (batch, candidates):
        raise ValueError("generator_ids must have shape [batch,candidates]")
    flat_generator_ids = generator_ids.reshape(-1).to(device=maps.device, dtype=torch.long)
    latent_dim = int(getattr(base_models[0], "latent_dim", 8))
    latent = torch.randn(
        batch * candidates, latent_dim, device=maps.device, dtype=maps.dtype,
        generator=generator,
    )
    fixed = endpoint_fields(flat_start, flat_goal, base_models[0].map_size)
    endpoint = fixed[:, :1]
    x = noise_scale * torch.randn(
        batch * candidates, 2, base_models[0].map_size, base_models[0].map_size,
        device=maps.device, dtype=maps.dtype, generator=generator,
    )
    x = x * (1.0 - endpoint) + fixed
    states: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    old_logprobs: list[torch.Tensor] = []
    barrier_values: list[torch.Tensor] = []
    for step in range(num_steps):
        next_x, mean, std, barrier = constrained_transition(
            x, flat_maps, flat_start, flat_goal, latent, flat_generator_ids,
            base_models, residual_policy, step, num_steps, sde_noise,
            **barrier_config, generator=generator,
        )
        states.append(x.detach())
        next_states.append(next_x.detach())
        old_logprobs.append(transition_logprob(
            next_x, mean, std, endpoint, reduction="mean",
        ).detach())
        barrier_values.append(barrier.detach())
        x = next_x
    return {
        "states": states,
        "next_states": next_states,
        "old_logprob": torch.stack(old_logprobs, dim=1).sum(dim=1),
        "final_fields": x.detach().reshape(
            batch, candidates, 2, base_models[0].map_size, base_models[0].map_size,
        ),
        "maps": flat_maps.detach(), "start": flat_start.detach(),
        "goal": flat_goal.detach(), "latent": latent.detach(),
        "endpoint": endpoint.detach(), "fixed_state": fixed.detach(),
        "generator_ids": flat_generator_ids.detach(),
        "generator_matrix": generator_ids.detach(), "batch": batch,
        "candidates": candidates, "sde_noise": float(sde_noise),
        "barrier_mean": torch.stack(barrier_values, dim=1).mean(dim=1).detach(),
    }


def recompute_logprob(
    base_models: Sequence[nn.Module],
    residual_policy: ResidualVelocityPolicy,
    data: dict[str, Any],
    num_steps: int,
    barrier_config: dict[str, float],
    activation_checkpointing: bool = True,
) -> torch.Tensor:
    values = []
    for step, (x, next_x) in enumerate(zip(data["states"], data["next_states"])):
        t = torch.full((len(x),), step / float(num_steps), device=x.device, dtype=x.dtype)
        generator_ids = data["generator_ids"]
        # The checkpointed closure keeps the large vector-field activation out
        # of the PPO graph while preserving gradients through the residual.
        def velocity_fn(x_value, map_value, start_value, goal_value, time_value, latent_value):
            from constrained_flow import grouped_base_velocity, project_velocity_with_barrier
            base = grouped_base_velocity(
                base_models, generator_ids, x_value, map_value, start_value,
                goal_value, time_value, latent_value,
            )
            value = base + residual_policy(
                x_value, map_value, start_value, goal_value, time_value, base,
            )
            projection_config = {
                "robot_radius_grid": barrier_config["robot_radius_grid"],
                "distance_scale": barrier_config["distance_scale"],
                "clearance_offset": barrier_config["clearance_offset"],
                "margin": barrier_config["barrier_margin"],
                "temperature": barrier_config["barrier_temperature"],
                "support_threshold": barrier_config["barrier_support_threshold"],
                "gain": barrier_config["barrier_gain"],
                "max_projection": barrier_config["barrier_max_projection"],
            }
            return project_velocity_with_barrier(
                x_value, value, map_value, data["endpoint"], **projection_config,
            )[0]
        if activation_checkpointing:
            velocity = activation_checkpoint(
                velocity_fn, x, data["maps"], data["start"], data["goal"], t,
                data["latent"], use_reentrant=False, preserve_rng_state=False,
            )
        else:
            velocity = velocity_fn(
                x, data["maps"], data["start"], data["goal"], t, data["latent"],
            )
        from p37_model import flow_sde_transition
        _, mean, std = flow_sde_transition(
            x, velocity, step, num_steps, data["sde_noise"],
            data["endpoint"], data["fixed_state"], None,
        )
        values.append(transition_logprob(
            next_x, mean, std, data["endpoint"], reduction="mean",
        ))
    return torch.stack(values, dim=1).sum(dim=1)


def terminal_costs(paths, decoder_valid, dataset, task_ids: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
    """Build bounded safety costs from the exact offline geometry protocol."""
    from utils.robot_geometry import FootprintChecker, evaluate_robot_path
    paths_np = paths.detach().cpu().numpy()
    valid_np = decoder_valid.detach().cpu().numpy().astype(bool)
    costs = np.zeros(paths_np.shape[:2], dtype=np.float32)
    successful = []
    clearance_violations = []
    for batch_index, task in enumerate(task_ids):
        occupancy = dataset.maps[int(task)]
        start = dataset.starts[int(task)]
        goal = dataset.goals[int(task)]
        checker = FootprintChecker(occupancy, dataset.geometry)
        for candidate in range(paths_np.shape[1]):
            world = dataset.denormalize_coordinates(paths_np[batch_index, candidate])
            result = evaluate_robot_path(
                world, occupancy, start, goal, dataset.geometry, endpoint_tolerance=1e-5,
            )
            segments = [checker.segment_clearance(a, b) for a, b in zip(world[:-1], world[1:])]
            minimum = min(segments) if segments else checker.segment_clearance(world[0], world[0])
            deficit = max(0.0, float(dataset.geometry.radius_grid) - float(minimum))
            invalid = not bool(valid_np[batch_index, candidate])
            # The binary failure term is deliberately dominant; the clearance
            # deficit supplies a dense ordering among otherwise safe samples.
            costs[batch_index, candidate] = np.clip(
                (0.85 if not bool(result["success"]) else 0.0)
                + (0.10 if invalid else 0.0)
                + 0.05 * min(deficit / max(float(dataset.geometry.radius_grid), 1e-6), 1.0),
                0.0, 1.0,
            )
            successful.append(bool(result["success"]))
            clearance_violations.append(float(deficit > 0))
    return torch.from_numpy(costs), {
        "mean_cost": float(costs.mean()),
        "task_cost_violation_rate": float((costs.mean(axis=1) > 0).mean()),
        "terminal_success_rate": float(np.mean(successful)) if successful else 0.0,
        "clearance_violation_rate": float(np.mean(clearance_violations)) if clearance_violations else 0.0,
    }


@torch.no_grad()
def validate_generators(
    base_models: Sequence[nn.Module], residual_policy: ResidualVelocityPolicy,
    dataset, task_ids: np.ndarray, device: torch.device, args,
) -> dict[str, float]:
    """Evaluate each base generator separately with the shared residual policy."""
    from utils.robot_geometry import evaluate_robot_path

    was_training = residual_policy.training
    residual_policy.eval()
    output: dict[str, float] = {}
    for generator_index, base in enumerate(base_models):
        successes = []; task_successes = []; selected_successes = []
        terminal_cost_values: list[np.ndarray] = []
        total_cost_values: list[np.ndarray] = []
        barrier_cost_values: list[np.ndarray] = []
        task_cost_values: list[float] = []
        selected_cost_values: list[float] = []
        for begin in range(0, len(task_ids), args.validation_batch_size):
            ids = task_ids[begin:begin + args.validation_batch_size]
            batch = [dataset[int(index)] for index in ids]
            maps = torch.stack([torch.cat([item["map"], item["distance_field"]], 0) for item in batch]).to(device)
            starts = torch.stack([item["start"] for item in batch]).to(device)
            goals = torch.stack([item["goal"] for item in batch]).to(device)
            distance = torch.stack([item["distance_field"] for item in batch]).to(device)
            ids_matrix = torch.full((len(ids), args.validation_candidates), generator_index,
                                    device=device, dtype=torch.long)
            data = rollout_constrained(
                base_models, residual_policy, maps, starts, goals,
                args.validation_candidates, args.validation_flow_steps,
                args.validation_noise_scale, args.sde_noise, args.barrier_config,
                torch.Generator(device=device).manual_seed(args.seed + 900_000 + generator_index + begin),
                generator_ids=ids_matrix,
            )
            paths, valid = decode_fields(
                base, data["final_fields"], distance, starts, goals,
                dataset.geometry.radius_grid, args.distance_scale,
                args.clearance_temperature, args.clearance_offset,
                args.decode_clearance_power, args.decode_continuity_beta,
            )
            scores = candidate_selection_scores(data["final_fields"], paths, maps, valid)
            selected = scores.argmax(dim=1).cpu().numpy()
            terminal_cost, _ = terminal_costs(
                paths, valid, dataset, ids,
            )
            barrier_cost = (
                float(args.barrier_cost_weight)
                * data["barrier_mean"].reshape(len(ids), args.validation_candidates).cpu()
            )
            total_cost = torch.clamp(terminal_cost + barrier_cost, 0.0, 1.0)
            terminal_cost_values.append(terminal_cost.numpy())
            barrier_cost_values.append(barrier_cost.numpy())
            total_cost_values.append(total_cost.numpy())
            task_cost_values.extend(total_cost.mean(dim=1).numpy().tolist())
            paths_np = dataset.denormalize_coordinates(paths.cpu().numpy())
            for local, task in enumerate(ids):
                results = []
                for candidate in range(args.validation_candidates):
                    result = evaluate_robot_path(
                        paths_np[local, candidate], dataset.maps[int(task)],
                        dataset.starts[int(task)], dataset.goals[int(task)], dataset.geometry,
                        endpoint_tolerance=1e-5,
                    )
                    results.append(bool(result["success"]))
                    successes.append(bool(result["success"]))
                task_successes.append(bool(np.any(results)))
                selected_successes.append(bool(results[int(selected[local])]))
                selected_cost_values.append(float(total_cost[local, int(selected[local])]))
        prefix = f"generator_{generator_index}"
        terminal_cost_array = np.concatenate(terminal_cost_values, axis=0)
        barrier_cost_array = np.concatenate(barrier_cost_values, axis=0)
        total_cost_array = np.concatenate(total_cost_values, axis=0)
        validation_mean_cost = float(total_cost_array.mean())
        output[f"{prefix}_candidate_success_rate"] = float(np.mean(successes))
        output[f"{prefix}_oracle_best_of_k_success_rate"] = float(np.mean(task_successes))
        output[f"{prefix}_selected_success_rate"] = float(np.mean(selected_successes))
        output[f"{prefix}_all_candidates_unsafe"] = float(np.sum(~np.asarray(task_successes)))
        # The robust constraint is checked on the same candidate-level cost
        # used by training, but independently for each frozen generator.
        # This prevents a good P37 average from hiding P50/P48 transfer cost.
        output[f"{prefix}_validation_mean_cost"] = validation_mean_cost
        output[f"{prefix}_validation_terminal_cost"] = float(terminal_cost_array.mean())
        output[f"{prefix}_validation_barrier_cost"] = float(barrier_cost_array.mean())
        output[f"{prefix}_validation_task_cost_violation_rate"] = float(
            np.mean(np.asarray(task_cost_values) > float(args.cost_epsilon))
        )
        output[f"{prefix}_validation_selected_cost"] = float(np.mean(selected_cost_values))
        output[f"{prefix}_validation_constraint_satisfied"] = float(
            validation_mean_cost <= float(args.cost_epsilon)
        )
    if was_training:
        residual_policy.train()
    return output


def save_checkpoint(path: Path, residual_policy, cost_critic, args, base_checkpoints, update, lagrange):
    torch.save({
        "variant": "p51_constrained_generator_agnostic_residual_flow",
        "residual_state_dict": residual_policy.state_dict(),
        "cost_critic_state_dict": cost_critic.state_dict(),
        "base_checkpoints": [str(Path(p).resolve()) for p in base_checkpoints],
        "config": {
            "residual_hidden_dim": args.residual_hidden_dim,
            "residual_scale": args.residual_scale,
            "cost_critic_hidden_dim": args.cost_critic_hidden_dim,
            "barrier_config": args.barrier_config,
            "barrier_cost_weight": args.barrier_cost_weight,
            "lagrange_multiplier": float(lagrange),
            "epsilon": args.cost_epsilon,
            "cost_ema_decay": args.cost_ema_decay,
        },
        "args": vars(args), "update": int(update),
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-checkpoint", action="append", dest="generator_checkpoints", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--sde-noise", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--critic-lr", type=float, default=2e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--cost-epsilon", type=float, default=0.05)
    parser.add_argument("--lagrange-init", type=float, default=0.1)
    parser.add_argument("--lagrange-step-size", type=float, default=0.1)
    parser.add_argument("--lagrange-max", type=float, default=20.0)
    parser.add_argument(
        "--cost-ema-decay", type=float, default=0.9,
        help="EMA decay for the robust generator-wise dual cost update.",
    )
    parser.add_argument("--critic-coef", type=float, default=0.5)
    parser.add_argument("--residual-hidden-dim", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.15)
    parser.add_argument("--cost-critic-hidden-dim", type=int, default=32)
    parser.add_argument("--length-weight", type=float, default=0.03)
    parser.add_argument("--smooth-weight", type=float, default=0.05)
    parser.add_argument("--clearance-weight", type=float, default=0.25)
    parser.add_argument("--distance-scale", type=float, default=8.0)
    parser.add_argument("--clearance-temperature", type=float, default=0.5)
    parser.add_argument("--clearance-offset", type=float, default=2 ** -0.5)
    parser.add_argument("--decode-clearance-power", type=float, default=2.0)
    parser.add_argument("--decode-continuity-beta", type=float, default=60.0)
    parser.add_argument("--barrier-margin", type=float, default=0.25)
    parser.add_argument("--barrier-temperature", type=float, default=0.25)
    parser.add_argument("--barrier-support-threshold", type=float, default=0.5)
    parser.add_argument("--barrier-gain", type=float, default=2.0)
    parser.add_argument("--barrier-max-projection", type=float, default=0.25)
    parser.add_argument(
        "--barrier-cost-weight", type=float, default=0.10,
        help="Weight of mean per-step barrier exposure in the constrained cost.",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument("--validation-tasks", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--validation-candidates", type=int, default=4)
    parser.add_argument("--validation-flow-steps", type=int, default=100)
    parser.add_argument("--validation-noise-scale", type=float, default=1.0)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.generator_checkpoints:
        raise ValueError("at least one --generator-checkpoint is required")
    if args.sde_noise <= 0:
        raise ValueError("constrained Flow-SDE requires positive --sde-noise")
    if not 0.0 <= args.cost_ema_decay < 1.0:
        raise ValueError("cost-ema-decay must be in [0, 1)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.barrier_config = {
        "robot_radius_grid": 1.5, "distance_scale": args.distance_scale,
        "clearance_offset": args.clearance_offset,
        "barrier_margin": args.barrier_margin,
        "barrier_temperature": args.barrier_temperature,
        "barrier_support_threshold": args.barrier_support_threshold,
        "barrier_gain": args.barrier_gain,
        "barrier_max_projection": args.barrier_max_projection,
    }

    source_project = None
    for parent in Path(args.generator_checkpoints[0]).resolve().parents:
        candidate = parent / "map_flow_matching"
        if (candidate / "utils" / "dataset.py").is_file():
            source_project = str(candidate); break
    if source_project is None:
        candidate = HERE.parent / "map_flow_matching"
        if (candidate / "utils" / "dataset.py").is_file(): source_project = str(candidate)
    if source_project is None:
        raise FileNotFoundError("cannot locate map_flow_matching/utils/dataset.py")
    sys.path.insert(0, source_project)
    from utils.dataset import PathPlanningDataset

    dataset = PathPlanningDataset(args.data_path, horizon=128, normalize=True,
                                  augment=False, distance_scale=args.distance_scale)
    if dataset.geometry is None:
        raise RuntimeError("P51 requires a geometry-aware dataset")
    args.barrier_config["robot_radius_grid"] = float(dataset.geometry.radius_grid)
    base_models = []
    for path in args.generator_checkpoints:
        model, _ = load_model(path, device)
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.distance_scale = args.distance_scale
        model.robot_radius_grid = float(dataset.geometry.radius_grid)
        model.clearance_temperature = args.clearance_temperature
        model.clearance_offset = args.clearance_offset
        model.decode_clearance_power = args.decode_clearance_power
        model.decode_continuity_beta = args.decode_continuity_beta
        base_models.append(model)

    residual_policy = ResidualVelocityPolicy(args.residual_hidden_dim, args.residual_scale).to(device)
    cost_critic = SafetyCostCritic(args.cost_critic_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(residual_policy.parameters(), lr=args.lr, weight_decay=1e-5)
    critic_optimizer = torch.optim.AdamW(cost_critic.parameters(), lr=args.critic_lr, weight_decay=1e-5)
    task_count = len(dataset) if args.max_tasks is None else min(len(dataset), args.max_tasks)
    task_ids = np.arange(task_count, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    validation_count = min(task_count, max(1, int(args.validation_tasks)))
    map_hashes = np.asarray([
        hashlib.sha1(np.ascontiguousarray(dataset.maps[int(index)]).tobytes()).hexdigest()
        for index in task_ids
    ])
    unique_hashes = np.unique(map_hashes)
    if len(unique_hashes) < 2 and validation_count < task_count:
        raise ValueError("map-disjoint validation requires at least two unique maps")
    order = rng.permutation(unique_hashes); chosen_hashes = []; count = 0
    for value in order:
        chosen_hashes.append(value); count += int(np.sum(map_hashes == value))
        if count >= validation_count: break
    validation_ids = np.flatnonzero(np.isin(map_hashes, chosen_hashes)).astype(np.int64)
    train_ids = np.setdiff1d(task_ids, validation_ids, assume_unique=True)
    if not len(train_ids): raise ValueError("validation split consumed all training tasks")
    output = Path(args.save_dir); output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "split.npz", train_ids=train_ids, validation_ids=validation_ids)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1009)
    lagrange = float(args.lagrange_init)
    history = []; best_safe = None; best_selection = None
    cost_ema = None
    for update in range(1, args.updates + 1):
        chosen = rng.choice(train_ids, size=args.batch_size, replace=True)
        batch = [dataset[int(index)] for index in chosen]
        maps = torch.stack([torch.cat([item["map"], item["distance_field"]], 0) for item in batch]).to(device)
        starts = torch.stack([item["start"] for item in batch]).to(device)
        goals = torch.stack([item["goal"] for item in batch]).to(device)
        distance = torch.stack([item["distance_field"] for item in batch]).to(device)
        data = rollout_constrained(
            base_models, residual_policy, maps, starts, goals, args.candidates,
            args.flow_steps, args.noise_scale, args.sde_noise, args.barrier_config,
            generator,
        )
        paths, decoder_valid = decode_fields(
            base_models[0], data["final_fields"], distance, starts, goals,
            dataset.geometry.radius_grid, args.distance_scale,
            args.clearance_temperature, args.clearance_offset,
            args.decode_clearance_power, args.decode_continuity_beta,
        )
        reward, reward_stats = terminal_rewards(
            paths, decoder_valid, dataset, chosen,
            args.length_weight, args.smooth_weight, args.clearance_weight,
        )
        costs, cost_stats = terminal_costs(paths, decoder_valid, dataset, chosen)
        barrier_exposure = data["barrier_mean"].reshape(data["batch"], args.candidates)
        barrier_cost = float(args.barrier_cost_weight) * barrier_exposure
        costs = torch.clamp(costs + barrier_cost.cpu(), 0.0, 1.0)
        # Use a robust generator-wise signal for the dual update.  The policy
        # still receives candidate-level PPO costs, but a strong P37 batch can
        # no longer hide a high P50/P48 cost from the multiplier.
        flat_costs = costs.flatten()
        flat_generator_ids = data["generator_matrix"].detach().cpu().flatten()
        generator_mean_costs = []
        for generator_index in range(len(base_models)):
            mask = flat_generator_ids == generator_index
            if not bool(mask.any()):
                raise RuntimeError("balanced rollout produced an empty generator group")
            generator_mean_costs.append(float(flat_costs[mask].mean()))
        dual_batch_cost = max(generator_mean_costs)
        cost_ema = (
            dual_batch_cost if cost_ema is None else
            args.cost_ema_decay * cost_ema + (1.0 - args.cost_ema_decay) * dual_batch_cost
        )
        cost_stats["mean_terminal_cost"] = cost_stats["mean_cost"]
        cost_stats["mean_barrier_cost"] = float(barrier_cost.mean().cpu())
        cost_stats["mean_cost"] = float(costs.mean().cpu())
        cost_stats["generator_mean_costs"] = generator_mean_costs
        cost_stats["worst_generator_cost"] = dual_batch_cost
        reward = reward.to(device=device, dtype=torch.float32)
        costs = costs.to(device=device, dtype=torch.float32)
        critic_value = cost_critic(maps, starts, goals)
        reward_advantage = compute_task_advantages(reward).detach()
        cost_advantage = normalize_cost_advantage(costs, critic_value).detach()
        old_logprob = data["old_logprob"].to(device)
        gradient_norm = 0.0; policy_loss = torch.zeros((), device=device)
        for _ in range(args.ppo_epochs):
            new_logprob = recompute_logprob(
                base_models, residual_policy, data, args.flow_steps,
                args.barrier_config, args.activation_checkpointing,
            )
            ratio = (new_logprob - old_logprob).clamp(-20.0, 20.0).exp()
            loss, reward_surrogate, cost_surrogate = constrained_ppo_loss(
                ratio, reward_advantage, cost_advantage, args.clip_eps, lagrange,
            )
            if not torch.isfinite(loss): raise FloatingPointError("non-finite constrained PPO loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(residual_policy.parameters(), 1.0).detach().cpu())
            optimizer.step(); policy_loss = loss.detach()
        critic_loss = (critic_value - costs.mean(dim=1).detach()).square().mean()
        critic_optimizer.zero_grad(set_to_none=True); (args.critic_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(cost_critic.parameters(), 1.0); critic_optimizer.step()
        lagrange = update_lagrange_multiplier(
            float(lagrange), float(cost_ema), args.cost_epsilon,
            args.lagrange_step_size, args.lagrange_max,
        )
        row = {
            "update": update, "policy_loss": float(policy_loss.cpu()),
            "reward_surrogate": float(reward_surrogate.detach().cpu()),
            "cost_surrogate": float(cost_surrogate.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "lagrange_multiplier": lagrange,
            "mean_barrier_penalty": float(data["barrier_mean"].mean().cpu()),
            "generator_usage": [int((data["generator_matrix"] == i).sum()) for i in range(len(base_models))],
            "cost_epsilon": args.cost_epsilon, "cost_constraint_satisfied": float(costs.mean()) <= args.cost_epsilon,
            "dual_batch_cost": dual_batch_cost, "cost_ema": cost_ema,
            "gradient_norm": gradient_norm, **reward_stats, **cost_stats,
        }
        if update == 1 or update % args.validation_interval == 0:
            validation = validate_generators(base_models, residual_policy, dataset, validation_ids, device, args)
            row.update({f"validation_{key}": value for key, value in validation.items()})
            validation_costs = [
                validation[f"generator_{index}_validation_mean_cost"]
                for index in range(len(base_models))
            ]
            validation_constraints_ok = all(
                value <= float(args.cost_epsilon) for value in validation_costs
            )
            worst_validation_cost = max(validation_costs)
            safe_values = [
                validation[f"generator_{index}_oracle_best_of_k_success_rate"]
                for index in range(len(base_models))
            ]
            selected_values = [
                validation[f"generator_{index}_selected_success_rate"]
                for index in range(len(base_models))
            ]
            safe_score = min(safe_values)
            selected_score = min(selected_values)
            selection_key = (safe_score, selected_score, -worst_validation_cost)
            row.update({
                "validation_worst_generator_cost": worst_validation_cost,
                "validation_constraint_satisfied": float(validation_constraints_ok),
                "validation_worst_generator_coverage": safe_score,
                "validation_worst_generator_selected": selected_score,
            })
            # A deployment checkpoint must satisfy the independent validation
            # constraint for every frozen generator.  Only then do coverage,
            # selected success, and lower cost break ties.
            if validation_constraints_ok and (
                best_selection is None or selection_key > best_selection
            ):
                best_selection = selection_key
                best_safe = safe_score
                save_checkpoint(output / "best_p51_deployment.pth", residual_policy, cost_critic,
                                args, args.generator_checkpoints, update, lagrange)
        history.append(row)
        if update == 1 or update % 5 == 0:
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    save_checkpoint(output / "last_p51.pth", residual_policy, cost_critic,
                    args, args.generator_checkpoints, args.updates, lagrange)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"output": str(output), "updates": args.updates,
                      "best_worst_generator_coverage": best_safe,
                      "best_selection": best_selection,
                      "validation_checkpoint_found": best_selection is not None,
                      "cost_ema": cost_ema,
                      "lagrange_multiplier": lagrange}), flush=True)


if __name__ == "__main__":
    main()
