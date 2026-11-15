"""P43: Flow-SDE PPO fine-tuning for the P37 ordered dense field model.

The policy state is P37's two-channel spatial field ``[occupancy, progress]``.
The decoder is kept fixed and is the same safe ordered decoder used by the
deployment evaluator.  Geometry is used only to construct an offline terminal
reward; it is never queried by the policy rollout.

P37 integrates from ``t=0`` (noise) to ``t=1`` (field).  The SDE transition is
written directly in that direction.  Endpoint pixels are deterministic and
are excluded from the Gaussian likelihood, otherwise the high-dimensional
PPO ratio is dominated by artificial endpoint terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p37_model import (  # noqa: E402
    OrderedDenseFieldFlow,
    candidate_selection_scores,
    endpoint_fields,
    flow_sde_transition,
)
from safe_field_decoder import (  # noqa: E402
    distance_field_to_soft_clearance,
    safe_ordered_field_to_path,
)


LOG_TWO_PI = math.log(2.0 * math.pi)

# Keep the historical script-level name used by tests and downstream tools,
# while sharing the implementation with deployment sampling.
sde_transition = flow_sde_transition


def transition_logprob(
    next_x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    endpoint: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return one likelihood per field sample, excluding fixed endpoints."""
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")
    non_endpoint = endpoint.expand_as(next_x) < 0.5
    safe_std = std.clamp_min(1e-8)
    element = (
        -torch.log(safe_std)
        - 0.5 * LOG_TWO_PI
        - 0.5 * ((next_x - mean) / safe_std).square()
    )
    element = torch.where(non_endpoint, element, torch.zeros_like(element))
    values = element.flatten(1).sum(dim=1)
    if reduction == "mean":
        values = values / non_endpoint.flatten(1).sum(dim=1).clamp_min(1)
    return values


def load_model(checkpoint: str, device: torch.device) -> tuple[OrderedDenseFieldFlow, dict[str, Any]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(state.get("config", {}))
    model = OrderedDenseFieldFlow(
        map_size=int(config.get("map_size", 64)),
        horizon=int(config.get("horizon", 128)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        latent_dim=int(config.get("latent_dim", 8)),
        decode_beta=float(config.get("decode_beta", 1280.0)),
        decode_power=float(config.get("decode_power", 4.0)),
        decode_threshold=float(config.get("decode_threshold", 0.1)),
        distance_scale=float(config.get("distance_scale", 8.0)),
        robot_radius_grid=float(config.get("robot_radius_grid", 1.5)),
        clearance_temperature=float(config.get("clearance_temperature", 0.5)),
        clearance_offset=float(config.get("clearance_offset", 2 ** -0.5)),
        decode_clearance_power=float(config.get("decode_clearance_power", 2.0)),
        decode_continuity_beta=float(config.get("decode_continuity_beta", 60.0)),
        decode_terminal_support_window=int(config.get("decode_terminal_support_window", 8)),
        decode_terminal_support_fraction=float(config.get("decode_terminal_support_fraction", 0.75)),
        decode_max_terminal_jump=float(config.get("decode_max_terminal_jump", 0.5)),
        decode_max_step_grid=config.get("decode_max_step_grid"),
        accept_threshold=float(config.get("accept_threshold", 0.5)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model, state


def rollout(
    model: OrderedDenseFieldFlow,
    maps: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    candidates: int,
    num_steps: int,
    noise_scale: float,
    sde_noise: float,
    generator: torch.Generator,
    logprob_reduction: str,
) -> dict[str, Any]:
    """Collect independent candidate trajectories and old transition scores."""
    if candidates < 1 or num_steps < 1 or noise_scale < 0:
        raise ValueError("invalid rollout parameters")
    batch = len(start)
    flat_maps = maps[:, None].expand(-1, candidates, -1, -1, -1).reshape(
        batch * candidates, *maps.shape[1:]
    )
    flat_start = start[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    flat_goal = goal[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    latent = torch.randn(
        batch, candidates, model.latent_dim, device=maps.device,
        dtype=maps.dtype, generator=generator,
    ).reshape(batch * candidates, model.latent_dim)
    fixed = endpoint_fields(flat_start, flat_goal, model.map_size)
    endpoint = fixed[:, :1]
    x = noise_scale * torch.randn(
        batch * candidates, 2, model.map_size, model.map_size,
        device=maps.device, dtype=maps.dtype, generator=generator,
    )
    x = x * (1.0 - endpoint) + fixed

    states: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    old_logprobs: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(num_steps):
            t = torch.full(
                (batch * candidates,), step / float(num_steps),
                device=maps.device, dtype=maps.dtype,
            )
            velocity = model.vector_field(
                x, flat_maps, flat_start, flat_goal, t, latent
            )
            xn, mean, std = sde_transition(
                x, velocity, step, num_steps, sde_noise, endpoint, fixed, generator
            )
            states.append(x.detach())
            next_states.append(xn.detach())
            old_logprobs.append(
                transition_logprob(
                    xn, mean, std, endpoint, reduction=logprob_reduction
                ).detach()
            )
            x = xn
    return {
        "states": states,
        "next_states": next_states,
        # Keep the per-transition scores for step-level PPO.  The summed
        # score remains for old callers and checkpoint compatibility.
        "old_logprob_steps": torch.stack(old_logprobs, dim=1),
        "old_logprob": torch.stack(old_logprobs, dim=1).sum(dim=1),
        "final_fields": x.detach().reshape(
            batch, candidates, 2, model.map_size, model.map_size
        ),
        "maps": flat_maps.detach(),
        "start": flat_start.detach(),
        "goal": flat_goal.detach(),
        "latent": latent.detach(),
        "endpoint": endpoint.detach(),
        "fixed_state": fixed.detach(),
        "batch": batch,
        "candidates": candidates,
        "sde_noise": float(sde_noise),
        "logprob_reduction": logprob_reduction,
    }


def recompute_logprob(
    model: OrderedDenseFieldFlow,
    data: dict[str, Any],
    num_steps: int,
    activation_checkpointing: bool = True,
    return_steps: bool = False,
) -> torch.Tensor:
    values = []
    for step, (x, xn) in enumerate(zip(data["states"], data["next_states"])):
        t = torch.full(
            (len(x),), step / float(num_steps), device=x.device, dtype=x.dtype
        )
        if activation_checkpointing:
            # Keep only the checkpoint inputs for each Flow step.  Without
            # this, all 100 U-Net activation graphs remain live until PPO
            # backward, which makes batch>1 OOM on a 24 GB GPU.
            velocity = activation_checkpoint(
                model.vector_field,
                x, data["maps"], data["start"], data["goal"], t,
                data["latent"],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            velocity = model.vector_field(
                x, data["maps"], data["start"], data["goal"], t,
                data["latent"],
            )
        _, mean, std = sde_transition(
            x, velocity, step, num_steps, data["sde_noise"],
            data["endpoint"], data["fixed_state"], None
        )
        values.append(
            transition_logprob(
                xn, mean, std, data["endpoint"], data["logprob_reduction"]
            )
        )
    values = torch.stack(values, dim=1)
    return values if return_steps else values.sum(dim=1)


def decode_fields(
    model: OrderedDenseFieldFlow,
    fields: torch.Tensor,
    distance: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    geometry_radius: float,
    distance_scale: float,
    clearance_temperature: float,
    clearance_offset: float,
    clearance_power: float,
    continuity_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode fields using the fixed deployment decoder."""
    batch, candidates = fields.shape[:2]
    flat_fields = fields.reshape(-1, 2, model.map_size, model.map_size)
    flat_start = start[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    flat_goal = goal[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    clearance = distance_field_to_soft_clearance(
        distance, distance_scale, geometry_radius,
        clearance_temperature, clearance_offset,
    )
    flat_clearance = clearance[:, None].expand(
        -1, candidates, -1, -1, -1
    ).reshape(-1, 1, model.map_size, model.map_size)
    paths, valid = safe_ordered_field_to_path(
        flat_fields[:, :1].clamp(0.0, 1.0),
        flat_fields[:, 1:2].clamp(0.0, 1.0),
        flat_clearance, flat_start, flat_goal, model.horizon,
        beta=model.decode_beta, power=model.decode_power,
        clearance_power=clearance_power, continuity_beta=continuity_beta,
        occupancy_threshold=model.decode_threshold, return_valid=True,
        terminal_support_window=model.decode_terminal_support_window,
        terminal_support_fraction=model.decode_terminal_support_fraction,
        max_terminal_jump=model.decode_max_terminal_jump,
        max_path_step=(
            None if model.decode_max_step_grid is None else
            2.0 * model.decode_max_step_grid / max(model.map_size - 1, 1)
        ),
    )
    return paths.reshape(batch, candidates, model.horizon, 2), valid.reshape(batch, candidates)


def terminal_rewards(
    paths: torch.Tensor,
    decoder_valid: torch.Tensor,
    dataset,
    task_ids: np.ndarray,
    length_weight: float,
    smooth_weight: float,
    clearance_weight: float,
    failure_goal_weight: float = 0.0,
    failure_start_weight: float = 0.0,
    failure_base_reward: float = -20.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a hard safety-gated offline reward for every candidate."""
    from utils.robot_geometry import evaluate_robot_path

    paths_np = paths.detach().cpu().numpy()
    valid_np = decoder_valid.detach().cpu().numpy().astype(bool)
    values = np.zeros(paths_np.shape[:2], dtype=np.float32)
    successes: list[bool] = []
    safes: list[bool] = []
    success_matrix = np.zeros(paths_np.shape[:2], dtype=bool)
    safe_matrix = np.zeros(paths_np.shape[:2], dtype=bool)
    valid_matrix = valid_np.copy()
    lengths: list[float] = []
    smoothness: list[float] = []
    clearances: list[float] = []
    goal_errors: list[float] = []
    start_errors: list[float] = []
    for batch_index, task in enumerate(task_ids):
        occupancy = dataset.maps[int(task)]
        start = dataset.starts[int(task)]
        goal = dataset.goals[int(task)]
        from utils.robot_geometry import FootprintChecker
        checker = FootprintChecker(occupancy, dataset.geometry)
        for candidate in range(paths_np.shape[1]):
            world = dataset.denormalize_coordinates(paths_np[batch_index, candidate])
            result = evaluate_robot_path(
                world, occupancy, start, goal, dataset.geometry,
                endpoint_tolerance=1e-5,
            )
            safe = bool(result["safe"])
            success = bool(result["success"])
            goal_error = float(result["goal_error"])
            start_error = float(result["start_error"])
            safe_matrix[batch_index, candidate] = safe
            success_matrix[batch_index, candidate] = success
            clearance = float(checker.segment_clearance(world[0], world[0]))
            segment_clearances = [checker.segment_clearance(a, b) for a, b in zip(world[:-1], world[1:])]
            if segment_clearances:
                clearance = min(clearance, min(segment_clearances))
            length = float(np.linalg.norm(np.diff(world, axis=0), axis=1).sum())
            smooth = float(np.linalg.norm(np.diff(world, n=2, axis=0), axis=1).mean())
            lengths.append(length); smoothness.append(smooth); clearances.append(clearance)
            goal_errors.append(goal_error); start_errors.append(start_error)
            successes.append(success); safes.append(safe)
            clearance_margin = clearance - float(dataset.geometry.radius_grid)
            if not valid_np[batch_index, candidate] or not success:
                # Safety is a gate, but failed candidates still receive a
                # graded signal from endpoint error and clearance severity.
                map_diagonal = float(np.hypot(*occupancy.shape))
                goal_term = failure_goal_weight * min(goal_error / max(map_diagonal, 1e-6), 1.0)
                start_term = failure_start_weight * min(start_error / max(map_diagonal, 1e-6), 1.0)
                values[batch_index, candidate] = (
                    float(failure_base_reward)
                    - 5.0 * float(not bool(result["completed_safe"]))
                    - goal_term
                    - start_term
                    - 2.0 * float(not bool(valid_np[batch_index, candidate]))
                    - 2.0 * max(0.0, -clearance_margin)
                )
            else:
                values[batch_index, candidate] = (
                    20.0
                    + clearance_weight * float(np.clip(clearance_margin, 0.0, 8.0))
                    - length_weight * length
                    - smooth_weight * smooth
                )
    return torch.from_numpy(values), {
        "candidate_success_rate": float(success_matrix.mean()),
        "oracle_best_of_k_success_rate": float(success_matrix.any(axis=1).mean()),
        "decoder_valid_rate": float(valid_matrix.mean()),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "safe_rate": float(np.mean(safes)) if safes else 0.0,
        "mean_reward": float(values.mean()),
        "best_reward": float(values.max()),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_smoothness": float(np.mean(smoothness)) if smoothness else 0.0,
        "mean_clearance": float(np.mean(clearances)) if clearances else 0.0,
        "mean_goal_error": float(np.mean(goal_errors)) if goal_errors else 0.0,
        "mean_start_error": float(np.mean(start_errors)) if start_errors else 0.0,
    }


def _normalize_advantage(value: torch.Tensor) -> torch.Tensor:
    value = value - value.mean()
    return value / value.std(unbiased=False).clamp_min(1e-4)


def compute_task_advantages(
    reward: torch.Tensor,
    task_weight: float = 1.0,
    candidate_weight: float = 0.25,
) -> torch.Tensor:
    """Build PPO advantages with task-level best-of-K credit assignment.

    The task term gives every candidate the signal from the best candidate of
    its task, so a task with no successful candidate is still pushed away from
    its current outcome.  A smaller within-task term retains path-quality
    ranking without making PPO optimize only selector ordering.
    """
    if reward.ndim != 2 or reward.shape[1] < 1:
        raise ValueError("reward must have shape [tasks, candidates]")
    if task_weight < 0 or candidate_weight < 0:
        raise ValueError("advantage weights must be non-negative")
    if task_weight == 0 and candidate_weight == 0:
        raise ValueError("at least one advantage weight must be positive")

    best_reward = reward.max(dim=1).values
    task_advantage = _normalize_advantage(best_reward)[:, None].expand_as(reward)
    candidate_advantage = reward - reward.mean(dim=1, keepdim=True)
    candidate_advantage = _normalize_advantage(candidate_advantage.reshape(-1)).reshape_as(reward)
    advantage = task_weight * task_advantage + candidate_weight * candidate_advantage
    return _normalize_advantage(advantage.reshape(-1))


def save_checkpoint(path: Path, model: OrderedDenseFieldFlow, args, source: str, update: int, variant: str) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "map_size": model.map_size, "horizon": model.horizon,
            "hidden_dim": model.hidden_dim, "latent_dim": model.latent_dim,
            "decode_beta": model.decode_beta, "decode_power": model.decode_power,
            "decode_threshold": model.decode_threshold,
            "distance_scale": model.distance_scale,
            "robot_radius_grid": model.robot_radius_grid,
            "clearance_temperature": model.clearance_temperature,
            "clearance_offset": model.clearance_offset,
            "decode_clearance_power": model.decode_clearance_power,
            "decode_continuity_beta": model.decode_continuity_beta,
            "accept_threshold": model.accept_threshold,
            "sde_noise": float(args.sde_noise),
        },
        "variant": variant,
        "source_checkpoint": str(source),
        "args": vars(args), "update": int(update),
    }, path)


@torch.no_grad()
def deployment_validation(
    model: OrderedDenseFieldFlow,
    dataset,
    task_ids: np.ndarray,
    device: torch.device,
    args,
) -> dict[str, float]:
    """Evaluate a fixed validation set with the deployment Flow kernel/decoder.

    Exact geometry is used only for this offline checkpoint-selection metric.
    The candidate selector itself sees only fields and the input distance
    channel, matching ``OrderedDenseFieldFlow.sample``.
    """
    from utils.robot_geometry import evaluate_robot_path

    was_training = model.training
    model.eval()
    candidate_successes: list[bool] = []
    task_successes: list[bool] = []
    selected_successes: list[bool] = []
    decoder_valids: list[bool] = []
    generator = torch.Generator(device=device).manual_seed(args.seed + 700_001)
    for begin in range(0, len(task_ids), args.validation_batch_size):
        ids = task_ids[begin:begin + args.validation_batch_size]
        batch = [dataset[int(index)] for index in ids]
        maps = torch.stack([
            torch.cat([item["map"], item["distance_field"]], dim=0)
            for item in batch
        ]).to(device)
        starts = torch.stack([item["start"] for item in batch]).to(device)
        goals = torch.stack([item["goal"] for item in batch]).to(device)
        distance = torch.stack([item["distance_field"] for item in batch]).to(device)
        candidates = args.validation_candidates
        output = model.sample(
            maps, starts, goals, candidates=candidates,
            num_steps=args.validation_flow_steps,
            noise_scale=args.validation_noise_scale,
            generator=generator, sde_noise=args.sde_noise,
        )
        paths_px = dataset.denormalize_coordinates(output["paths"].cpu().numpy())
        valid_np = output["decoder_valid"].cpu().numpy()
        selected = output["selected_index"].cpu().numpy()
        for local, task in enumerate(ids):
            task_results = []
            for candidate in range(candidates):
                result = evaluate_robot_path(
                    paths_px[local, candidate], dataset.maps[int(task)],
                    dataset.starts[int(task)], dataset.goals[int(task)],
                    dataset.geometry, endpoint_tolerance=1e-5,
                )
                task_results.append(bool(result["success"]))
                candidate_successes.append(bool(result["success"]))
                decoder_valids.append(bool(valid_np[local, candidate]))
            task_successes.append(bool(np.any(task_results)))
            selected_successes.append(bool(task_results[int(selected[local])]))
    if was_training:
        model.train()
    return {
        "candidate_success_rate": float(np.mean(candidate_successes)),
        "oracle_best_of_k_success_rate": float(np.mean(task_successes)),
        "selected_path_success_rate": float(np.mean(selected_successes)),
        "selector_miss_rate": float(np.mean(
            np.asarray(task_successes) & ~np.asarray(selected_successes)
        )),
        "decoder_valid_rate": float(np.mean(decoder_valids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--sde-noise", type=float, default=0.1)
    parser.add_argument("--logprob-reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument(
        "--activation-checkpointing", action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint vector-field activations during PPO log-prob recompute.",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--length-weight", type=float, default=0.03)
    parser.add_argument("--smooth-weight", type=float, default=0.05)
    parser.add_argument("--clearance-weight", type=float, default=0.25)
    parser.add_argument(
        "--task-advantage-weight", type=float, default=1.0,
        help="Weight for the task-level best-of-K PPO advantage.",
    )
    parser.add_argument(
        "--candidate-advantage-weight", type=float, default=0.25,
        help="Weight for within-task candidate quality ranking.",
    )
    parser.add_argument("--distance-scale", type=float, default=8.0)
    parser.add_argument("--clearance-temperature", type=float, default=0.5)
    parser.add_argument("--clearance-offset", type=float, default=2 ** -0.5)
    parser.add_argument("--decode-clearance-power", type=float, default=2.0)
    parser.add_argument("--decode-continuity-beta", type=float, default=60.0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument("--validation-tasks", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--validation-candidates", type=int, default=4)
    parser.add_argument("--validation-flow-steps", type=int, default=100)
    parser.add_argument("--validation-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--map-disjoint-validation", action=argparse.BooleanOptionalAction,
        default=True,
        help="Split validation by unique map bytes so duplicate maps never leak.",
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.sde_noise <= 0:
        raise ValueError("P37 Flow-SDE PPO requires positive --sde-noise")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # Import the source project's dataset without assuming a particular
    # checkpoint depth.  Formal checkpoints live under
    # ``<root>/flow_path_planning/map_flow_matching_p1/results/...``; older
    # runs may be nested differently, so locate the sibling project by its
    # dataset module instead of using a fixed ``parents[n]`` index.
    source_project = None
    local_project = HERE
    if (local_project / "utils" / "dataset.py").is_file():
        source_project = str(local_project)
    checkpoint_path = Path(args.checkpoint).resolve()
    if source_project is None:
        for parent in checkpoint_path.parents:
            candidate = parent / "map_flow_matching"
            if (candidate / "utils" / "dataset.py").is_file():
                source_project = str(candidate)
                break
    if source_project is None:
        candidate = HERE.parent / "map_flow_matching"
        if (candidate / "utils" / "dataset.py").is_file():
            source_project = str(candidate)
    if source_project is None:
        raise FileNotFoundError(
            "cannot locate map_flow_matching/utils/dataset.py relative to checkpoint"
        )
    sys.path.insert(0, source_project)
    from utils.dataset import PathPlanningDataset

    dataset = PathPlanningDataset(
        args.data_path, horizon=128, normalize=True, augment=False,
        distance_scale=args.distance_scale,
    )
    if dataset.geometry is None:
        raise RuntimeError("P37 PPO requires a geometry-aware dataset")
    model, _ = load_model(args.checkpoint, device)
    # The dataset geometry is the source of truth for this run.  Binding it
    # here keeps rollout rewards, validation, and deployment sampling aligned,
    # including checkpoints created before the geometry fields were persisted.
    model.distance_scale = float(args.distance_scale)
    model.robot_radius_grid = float(dataset.geometry.radius_grid)
    model.clearance_temperature = float(args.clearance_temperature)
    model.clearance_offset = float(args.clearance_offset)
    model.decode_clearance_power = float(args.decode_clearance_power)
    model.decode_continuity_beta = float(args.decode_continuity_beta)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    task_count = len(dataset) if args.max_tasks is None else min(len(dataset), args.max_tasks)
    task_ids = np.arange(task_count, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    validation_count = min(task_count, max(1, int(args.validation_tasks)))
    if args.map_disjoint_validation:
        map_hashes = np.asarray([
            hashlib.sha1(np.ascontiguousarray(dataset.maps[int(index)]).tobytes()).hexdigest()
            for index in task_ids
        ])
        unique_hashes = np.unique(map_hashes)
        if len(unique_hashes) < 2:
            raise ValueError("map-disjoint validation requires at least two unique maps")
        hash_order = rng.permutation(unique_hashes)
        selected_hashes = []
        selected_count = 0
        for value in hash_order:
            selected_hashes.append(value)
            selected_count += int(np.sum(map_hashes == value))
            if selected_count >= validation_count:
                break
        validation_ids = np.flatnonzero(np.isin(map_hashes, selected_hashes)).astype(np.int64)
    else:
        validation_ids = np.sort(rng.choice(
            task_ids, size=validation_count, replace=False,
        ))
    validation_ids = np.sort(validation_ids)
    train_ids = np.setdiff1d(task_ids, validation_ids, assume_unique=True)
    if not len(train_ids):
        raise ValueError("validation split consumed all PPO training tasks")
    output = Path(args.save_dir); output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "split.npz", train_ids=train_ids, validation_ids=validation_ids)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1009)
    history: list[dict[str, Any]] = []
    best_reward = -float("inf")
    best_success = -float("inf")

    for update in range(1, args.updates + 1):
        chosen = rng.choice(train_ids, size=args.batch_size, replace=True)
        batch = [dataset[int(index)] for index in chosen]
        maps = torch.stack([
            torch.cat([item["map"], item["distance_field"]], dim=0)
            for item in batch
        ]).to(device)
        starts = torch.stack([item["start"] for item in batch]).to(device)
        goals = torch.stack([item["goal"] for item in batch]).to(device)
        distance = torch.stack([item["distance_field"] for item in batch]).to(device)
        data = rollout(
            model, maps, starts, goals, args.candidates, args.flow_steps,
            args.noise_scale, args.sde_noise, generator, args.logprob_reduction,
        )
        paths, decoder_valid = decode_fields(
            model, data["final_fields"], distance, starts, goals,
            dataset.geometry.radius_grid, args.distance_scale,
            args.clearance_temperature, args.clearance_offset,
            args.decode_clearance_power, args.decode_continuity_beta,
        )
        reward, reward_stats = terminal_rewards(
            paths, decoder_valid, dataset, chosen,
            args.length_weight, args.smooth_weight, args.clearance_weight,
        )
        reward = reward.to(device=device, dtype=torch.float32)
        advantage = compute_task_advantages(
            reward,
            task_weight=args.task_advantage_weight,
            candidate_weight=args.candidate_advantage_weight,
        )
        old_logprob = data["old_logprob"].to(device)
        gradient_norm = 0.0
        for _ in range(args.ppo_epochs):
            new_logprob = recompute_logprob(
                model, data, args.flow_steps,
                activation_checkpointing=args.activation_checkpointing,
            )
            log_ratio = (new_logprob - old_logprob).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps)
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            if not torch.isfinite(policy_loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True); policy_loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            ).detach().cpu())
            optimizer.step()

        row = {
            "update": update,
            "policy_loss": float(policy_loss.detach().cpu()),
            "mean_log_ratio": float(log_ratio.detach().mean().cpu()),
            "clip_fraction": float((log_ratio.abs() > math.log(1.0 + args.clip_eps)).float().mean().cpu()),
            "gradient_norm": gradient_norm,
            **reward_stats,
        }
        if update == 1 or update % args.validation_interval == 0:
            validation_stats = deployment_validation(
                model, dataset, validation_ids, device, args,
            )
            row.update({f"validation_{key}": value
                        for key, value in validation_stats.items()})
            # Select the deployment checkpoint by the generator objective.
            # Selected-path success is reported separately and must not hide
            # tasks for which every candidate is unsafe.
            validation_score = validation_stats["oracle_best_of_k_success_rate"]
            if validation_score > best_success:
                best_success = validation_score
                save_checkpoint(
                    output / "best_p43_deployment.pth", model, args,
                    args.checkpoint, update, "p43_p37_flow_sde_ppo",
                )
        history.append(row)
        score = row["mean_reward"]
        if score > best_reward:
            best_reward = score
            save_checkpoint(output / "best_p43_reward.pth", model, args, args.checkpoint, update, "p43_p37_flow_sde_ppo")
        if update == 1 or update % 5 == 0:
            (output / "history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(row), flush=True)

    save_checkpoint(output / "last_p43.pth", model, args, args.checkpoint, args.updates, "p43_p37_flow_sde_ppo")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"output": str(output), "updates": args.updates,
                      "best_reward": best_reward,
                      "best_validation_oracle_success": best_success}), flush=True)


if __name__ == "__main__":
    main()
