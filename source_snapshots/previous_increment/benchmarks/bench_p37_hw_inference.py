"""Read-only GPU shape/latency smoke for P37 and Refiner deployment.

Example (run from the repository root):
  python benchmarks/bench_p37_hw_inference.py \
    --flow-checkpoint /path/to/flow_prior.pth \
    --refiner-checkpoint /path/to/refiner_warm.pth --device cuda:0

The default 2 candidates and 4 Flow steps are intended for shape and component
latency checks. They do not evaluate planning quality or replace full settings.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))

from p37_planner import P37FlowPlanner, _p37_map_input  # noqa: E402
from path_refinement import signed_distance_features  # noqa: E402
from refinement_planner import FlowRefinementPlanner  # noqa: E402
from safe_field_decoder import (  # noqa: E402
    distance_field_to_soft_clearance,
    safe_ordered_field_to_path,
)

SIZES = (
    (64, 64), (128, 128), (256, 256), (512, 512),
    (128, 256), (256, 512), (512, 768), (127, 193), (255, 511),
)


def _median_ms(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        del value
    return float(statistics.median(samples))


def _peak_memory_mib(function) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    value = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    del value
    return float(peak)


def _endpoint_error(paths: torch.Tensor, start: np.ndarray, goal: np.ndarray) -> float:
    start_tensor = torch.as_tensor(start, device=paths.device, dtype=paths.dtype)
    goal_tensor = torch.as_tensor(goal, device=paths.device, dtype=paths.dtype)
    return max(
        float((paths[..., 0, :] - start_tensor).abs().max().item()),
        float((paths[..., -1, :] - goal_tensor).abs().max().item()),
    )


def benchmark_shape(p37, refiner, height, width, candidates, steps, warmup, repeats):
    occupancy = np.zeros((1, height, width), dtype=np.float32)
    starts = np.asarray([[2.0, 2.0]], dtype=np.float32)
    goals = np.asarray([[width - 3.0, height - 3.0]], dtype=np.float32)
    device = p37.device

    map_tensor = torch.from_numpy(
        _p37_map_input(occupancy[0], p37.model.distance_scale)[None]
    ).to(device)
    scale = np.asarray([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    start = torch.from_numpy(starts / scale - 1.0).to(device)
    goal = torch.from_numpy(goals / scale - 1.0).to(device)
    flat_maps = map_tensor.expand(candidates, -1, -1, -1)
    flat_start = start.expand(candidates, -1)
    flat_goal = goal.expand(candidates, -1)
    latent = torch.zeros(candidates, p37.model.latent_dim, device=device)
    source = torch.zeros(candidates, 2, height, width, device=device)
    state = torch.zeros_like(source)
    times = torch.zeros(candidates, device=device)
    generator = torch.Generator(device=device).manual_seed(314159)

    vector_field_ms = _median_ms(
        lambda: p37.model.vector_field(
            state, flat_maps, flat_start, flat_goal, times, latent,
        ), warmup, repeats,
    )
    integrated = p37.model._integrate(
        flat_maps, flat_start, flat_goal, latent, steps, 1.0, source=source,
    )
    flow_integration_ms = _median_ms(
        lambda: p37.model._integrate(
            flat_maps, flat_start, flat_goal, latent, steps, 1.0, source=source,
        ), warmup, repeats,
    )
    distance = distance_field_to_soft_clearance(
        flat_maps[:, 1:2], p37.model.distance_scale, p37.model.robot_radius_grid,
        p37.model.clearance_temperature, p37.model.clearance_offset,
    )
    max_step = (
        None if p37.model.decode_max_step_grid is None else
        2.0 * p37.model.decode_max_step_grid / 63.0
    )

    def decode():
        return safe_ordered_field_to_path(
            integrated[:, :1].clamp(0.0, 1.0),
            integrated[:, 1:2].clamp(0.0, 1.0), distance,
            flat_start, flat_goal, p37.model.horizon,
            beta=p37.model.decode_beta, power=p37.model.decode_power,
            clearance_power=p37.model.decode_clearance_power,
            continuity_beta=p37.model.decode_continuity_beta,
            occupancy_threshold=p37.model.decode_threshold,
            terminal_support_window=p37.model.decode_terminal_support_window,
            terminal_support_fraction=p37.model.decode_terminal_support_fraction,
            max_terminal_jump=p37.model.decode_max_terminal_jump,
            max_path_step=max_step, return_valid=True,
        )

    decoder_ms = _median_ms(decode, warmup, repeats)

    def generate_candidates():
        return p37.model.sample(
            map_tensor, start, goal, candidates=candidates, num_steps=steps,
            generator=generator, noise_scale=1.0,
        )

    candidate_generation_ms = _median_ms(generate_candidates, warmup, repeats)
    base_sample = generate_candidates()
    refiner_maps = torch.from_numpy(
        signed_distance_features(occupancy[0], refiner.refiner.distance_scale)[None]
    ).to(device)
    base_paths = base_sample["paths"].reshape(candidates, -1, 2)
    repeated_features = refiner_maps.expand(candidates, -1, -1, -1)
    refiner_ms = _median_ms(
        lambda: refiner.refiner(repeated_features, base_paths), warmup, repeats,
    )

    p37_result = p37.plan(
        occupancy, starts, goals, candidates=candidates, seed=20261111,
        flow_steps=steps,
    )
    p37_endpoint_error = _endpoint_error(p37_result["paths"], starts[0], goals[0])
    if not torch.isfinite(p37_result["paths"]).all() or p37_endpoint_error > 1e-3:
        raise AssertionError(f"P37 output invalid for {height}x{width}")
    p37_ms = _median_ms(lambda: p37.plan(
        occupancy, starts, goals, candidates=candidates, seed=20261111,
        flow_steps=steps,
    ), warmup, repeats)
    p37_peak_mib = _peak_memory_mib(lambda: p37.plan(
        occupancy, starts, goals, candidates=candidates, seed=20261111,
        flow_steps=steps,
    ))

    refiner_result = refiner.plan(
        occupancy, starts, goals, candidates=candidates, seed=20261111,
        flow_steps=steps,
    )
    refiner_endpoint_error = _endpoint_error(
        refiner_result["paths"], starts[0], goals[0],
    )
    if not torch.isfinite(refiner_result["paths"]).all() or refiner_endpoint_error > 1e-3:
        raise AssertionError(f"Refiner output invalid for {height}x{width}")
    refiner_e2e_ms = _median_ms(lambda: refiner.plan(
        occupancy, starts, goals, candidates=candidates, seed=20261111,
        flow_steps=steps,
    ), warmup, repeats)
    refiner_peak_mib = _peak_memory_mib(lambda: refiner.plan(
        occupancy, starts, goals, candidates=candidates, seed=20261111,
        flow_steps=steps,
    ))

    return {
        "shape": [height, width],
        "vector_field_ms": vector_field_ms,
        "flow_integration_ms": flow_integration_ms,
        "decoder_ms": decoder_ms,
        "candidate_generation_ms": candidate_generation_ms,
        "refiner_ms": refiner_ms,
        "p37_end_to_end_ms": p37_ms,
        "p37_peak_allocated_mib": p37_peak_mib,
        "refiner_end_to_end_ms": refiner_e2e_ms,
        "refiner_peak_allocated_mib": refiner_peak_mib,
        "p37_accepted": bool(p37_result["accepted"].any().item()),
        "refiner_abstain": bool(refiner_result["abstain"][0].item()),
        "p37_max_endpoint_error_px": p37_endpoint_error,
        "refiner_max_endpoint_error_px": refiner_endpoint_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    p37 = P37FlowPlanner(args.flow_checkpoint, args.device)
    refiner = FlowRefinementPlanner(
        args.flow_checkpoint, args.refiner_checkpoint, device=args.device,
    ).eval()
    with torch.inference_mode():
        rows = [benchmark_shape(
            p37, refiner, height, width, args.candidates, args.steps,
            args.warmup, args.repeats,
        ) for height, width in SIZES]
    report = {
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flow_checkpoint": str(args.flow_checkpoint),
        "refiner_checkpoint": str(args.refiner_checkpoint),
        "candidates": args.candidates,
        "flow_steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "note": "Synthetic empty maps; shape/runtime smoke only, not planning-quality validation.",
        "results": rows,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
