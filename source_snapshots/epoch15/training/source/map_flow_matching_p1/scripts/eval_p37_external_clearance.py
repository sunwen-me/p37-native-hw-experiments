"""Evaluate a P37 checkpoint with an external clearance-aware decoder."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "map_flow_matching"))

from p37_model import (  # noqa: E402
    OrderedDenseFieldFlow,
)
from scripts.geometry_eval import score_task  # noqa: E402
from utils.dataset import PathPlanningDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paths-output", required=True)
    parser.add_argument("--fields-output", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-pixels-per-batch", type=int, default=1572864,
        help="Cap H*W*batch*candidates to bound native-H/W evaluation memory; 0 disables.",
    )
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--sde-noise", type=float, default=None,
        help="Flow-SDE transition noise; defaults to the checkpoint training value.",
    )
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--distance-scale", type=float, default=None)
    parser.add_argument(
        "--latent-scale", type=float, default=None,
        help="Override the checkpoint latent conditioning scale for an ablation.",
    )
    parser.add_argument(
        "--robot-radius-grid", type=float, default=None,
        help="Override dataset geometry radius; default reads geometry_json.",
    )
    parser.add_argument("--clearance-temperature", type=float, default=0.5)
    parser.add_argument(
        "--clearance-offset", type=float, default=2 ** -0.5,
        help="Conservative cell-center to closed-cell distance offset.",
    )
    parser.add_argument("--decode-beta", type=float, default=None)
    parser.add_argument("--decoder-mode", choices=("legacy", "scale_aware"), default=None)
    parser.add_argument(
        "--endpoint-protocol", choices=("raw_v1", "exact_hard_endpoints_v2"),
        default="exact_hard_endpoints_v2",
        help="v2 restores known hard endpoints without changing geometry tolerances.",
    )
    parser.add_argument("--decode-power", type=float, default=None)
    parser.add_argument("--decode-clearance-power", type=float, default=2.0)
    parser.add_argument("--decode-continuity-beta", type=float, default=60.0)
    parser.add_argument("--decode-terminal-support-window", type=int, default=None)
    parser.add_argument("--decode-terminal-support-fraction", type=float, default=None)
    parser.add_argument("--decode-max-terminal-jump", type=float, default=None)
    parser.add_argument(
        "--decode-max-step-grid", type=float, default=None,
        help="Optional max decoded step in grid pixels; model-only validity gate.",
    )
    parser.add_argument("--decode-threshold", type=float, default=None)
    args = parser.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    saved_args = state.get("args", {})
    sde_noise = (
        config.get("sde_noise", saved_args.get("sde_noise", 0.0))
        if args.sde_noise is None else args.sde_noise
    )
    if sde_noise < 0:
        raise ValueError("sde_noise must be non-negative")
    distance_scale = config.get("distance_scale", 8.0) if args.distance_scale is None else args.distance_scale
    beta = config.get("decode_beta", 1280.0) if args.decode_beta is None else args.decode_beta
    power = config.get("decode_power", 4.0) if args.decode_power is None else args.decode_power
    threshold = config.get("decode_threshold", 0.1) if args.decode_threshold is None else args.decode_threshold
    terminal_window = config.get("decode_terminal_support_window", 8) if args.decode_terminal_support_window is None else args.decode_terminal_support_window
    terminal_fraction = config.get("decode_terminal_support_fraction", 0.75) if args.decode_terminal_support_fraction is None else args.decode_terminal_support_fraction
    max_terminal_jump = config.get("decode_max_terminal_jump", 0.5) if args.decode_max_terminal_jump is None else args.decode_max_terminal_jump
    horizon = int(config.get("horizon", 128))
    map_size = int(config.get("map_size", 64))
    model = OrderedDenseFieldFlow(
        map_size=map_size, horizon=horizon,
        map_height=config.get("map_height"), map_width=config.get("map_width"),
        latent_dim=config.get("latent_dim", 8), hidden_dim=config.get("hidden_dim", 64),
        decode_beta=beta, decode_power=power, decode_threshold=threshold,
        distance_scale=distance_scale,
        robot_radius_grid=config.get("robot_radius_grid", 1.5),
        clearance_temperature=args.clearance_temperature,
        clearance_offset=args.clearance_offset,
        decode_clearance_power=args.decode_clearance_power,
        decode_continuity_beta=args.decode_continuity_beta,
        decode_terminal_support_window=terminal_window,
        decode_terminal_support_fraction=terminal_fraction,
        decode_max_terminal_jump=max_terminal_jump,
        decode_max_step_grid=(
            config.get("decode_max_step_grid")
            if args.decode_max_step_grid is None else args.decode_max_step_grid
        ),
        decoder_mode=(config.get("decoder_mode", "legacy")
                      if args.decoder_mode is None else args.decoder_mode),
        architecture=config.get("architecture", "cnn"),
        transformer_window_size=config.get("transformer_window_size", 8),
    ).to(args.device).eval()
    model.load_state_dict(state["model_state_dict"], strict=True)
    if args.latent_scale is not None:
        model.vector_field.latent_scale = float(args.latent_scale)
    dataset = PathPlanningDataset(
        args.test_data, horizon=horizon, normalize=True, augment=False,
        distance_scale=distance_scale,
    )
    if dataset.geometry is None:
        raise ValueError("test data must include geometry_json")
    robot_radius_grid = (
        dataset.geometry.radius_grid
        if args.robot_radius_grid is None else args.robot_radius_grid
    )
    # The evaluated dataset/CLI geometry is the source of truth.  Sampling,
    # decoding, and the offline checker must use the same radius and decoder
    # parameters even for older checkpoints with incomplete config metadata.
    model.distance_scale = float(distance_scale)
    model.robot_radius_grid = float(robot_radius_grid)
    model.clearance_temperature = float(args.clearance_temperature)
    model.clearance_offset = float(args.clearance_offset)
    model.decode_clearance_power = float(args.decode_clearance_power)
    model.decode_continuity_beta = float(args.decode_continuity_beta)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    shape_groups = {}
    for task_id in range(len(dataset)):
        shape_groups.setdefault(dataset.shape_at(task_id), []).append(task_id)
    paths_norm = np.empty(
        (len(dataset), args.candidates, horizon, 2), dtype=np.float32,
    )
    decoder_valid = np.empty((len(dataset), args.candidates), dtype=bool)
    selected_index = np.empty(len(dataset), dtype=np.int64)
    field_cache = {}
    with torch.no_grad():
        generated = 0
        for shape, group_ids in shape_groups.items():
            group_fields = []
            height, width = shape
            batch_size = args.batch_size
            if args.max_pixels_per_batch > 0:
                pixels_per_task = height * width * args.candidates
                batch_size = min(
                    batch_size,
                    max(1, args.max_pixels_per_batch // pixels_per_task),
                )
            for begin in range(0, len(group_ids), batch_size):
                ids = group_ids[begin:begin + batch_size]
                batch = [dataset[i] for i in ids]
                maps = torch.stack([
                    torch.cat([item["map"], item["distance_field"]], dim=0)
                    for item in batch
                ]).to(args.device)
                starts = torch.stack([item["start"] for item in batch]).to(args.device)
                goals = torch.stack([item["goal"] for item in batch]).to(args.device)
                sampled = model.sample(
                    maps, starts, goals, candidates=args.candidates,
                    num_steps=args.num_steps, noise_scale=args.noise_scale,
                    generator=generator, sde_noise=sde_noise,
                )
                paths_norm[ids] = sampled["paths"].cpu().numpy()
                decoder_valid[ids] = sampled["decoder_support_valid"].cpu().numpy()
                selected_index[ids] = sampled["selected_index"].cpu().numpy()
                if args.fields_output is not None:
                    group_fields.append(sampled["fields"].cpu().numpy())
                generated += len(ids)
                print(f"generated={generated}/{len(dataset)}", flush=True)
            if args.fields_output is not None:
                field_cache[shape] = (np.asarray(group_ids), np.concatenate(group_fields))
    task_ids = np.arange(len(dataset), dtype=np.int64)
    paths_px = (
        dataset.denormalize_paths(paths_norm, task_ids=task_ids)
        if args.endpoint_protocol == "exact_hard_endpoints_v2" else
        dataset.denormalize_coordinates(paths_norm, task_ids=task_ids)
    )
    jobs = [
        (str(Path(args.source_project).resolve()), paths_px[i], dataset.maps[i],
         dataset.starts[i], dataset.goals[i], dataset.geometry.to_json())
        for i in range(len(dataset))
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        safety = list(pool.map(score_task, jobs, chunksize=1))
    safe = np.asarray(safety, dtype=bool)
    per_task_any = safe.any(axis=1)
    selected_success = safe[np.arange(len(safe)), selected_index]
    selector_miss = per_task_any & ~selected_success
    by_shape = {}
    for shape, group_ids in shape_groups.items():
        ids = np.asarray(group_ids, dtype=np.int64)
        shape_any = per_task_any[ids]
        shape_selected = selected_success[ids]
        by_shape[f"{shape[0]}x{shape[1]}"] = {
            "tasks": int(len(ids)),
            "candidate_success_rate": float(safe[ids].mean()),
            "oracle_best_of_k_success_rate": float(shape_any.mean()),
            "selected_path_success_rate": float(shape_selected.mean()),
            "selector_miss_rate": float((shape_any & ~shape_selected).mean()),
            "decoder_invalid_rate": float((~decoder_valid[ids]).mean()),
        }
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_data": str(Path(args.test_data).resolve()),
        "tasks": int(len(dataset)), "candidates": int(args.candidates),
        "num_steps": int(args.num_steps), "noise_scale": float(args.noise_scale),
        "sde_noise": float(sde_noise),
        "seed": int(args.seed), "candidate_safety": safe.mean(axis=0).tolist(),
        "decode_beta": float(beta), "decode_power": float(power),
        "decoder_mode": model.decoder_mode, "endpoint_protocol": args.endpoint_protocol,
        "decode_clearance_power": float(args.decode_clearance_power),
        "decode_continuity_beta": float(args.decode_continuity_beta),
        "decode_terminal_support_window": int(terminal_window),
        "decode_terminal_support_fraction": float(terminal_fraction),
        "decode_max_terminal_jump": float(max_terminal_jump),
        "decode_max_step_grid": (
            None if model.decode_max_step_grid is None else
            float(model.decode_max_step_grid)
        ),
        "decode_threshold": float(threshold), "distance_scale": float(distance_scale),
        "robot_radius_grid": float(robot_radius_grid),
        "clearance_offset": float(args.clearance_offset),
        "C_N": float(per_task_any.mean()),
        "all_candidates_unsafe": int((~per_task_any).sum()),
        "candidate_success_rate": float(safe.mean()),
        "selected_path_success_rate": float(selected_success.mean()),
        "selector_miss_rate": float(selector_miss.mean()),
        "decoder_invalid_rate": float((~decoder_valid).mean()),
        "selected_index_histogram": np.bincount(
            selected_index, minlength=args.candidates,
        ).tolist(),
        "endpoint_error_max": float(np.max(np.linalg.norm(
            paths_px[:, :, [0, -1]] -
            np.stack([dataset.starts, dataset.goals], axis=1)[:, None], axis=-1,
        ))),
        "online_selector": True,
        "selector_type": "dense_segment_distance_field",
        "online_guidance": False, "online_search": False,
        "by_shape": by_shape,
        "dataset_limitations": [
            "Synthetic connected room-grid distribution.",
            "Only reachable positive-route tasks are included; this run does not measure unreachable-task rejection.",
        ],
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    paths_output = Path(args.paths_output); paths_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths_output, paths=paths_norm, safe=safe,
        decoder_valid=decoder_valid, selected_index=selected_index,
    )
    if args.fields_output is not None:
        fields_output = Path(args.fields_output)
        fields_output.parent.mkdir(parents=True, exist_ok=True)
        if len(field_cache) == 1:
            fields = next(iter(field_cache.values()))[1]
            np.savez_compressed(fields_output, fields=fields)
        else:
            field_payload = {}
            for (height, width), (ids, fields) in field_cache.items():
                key = f"{height}x{width}"
                field_payload[f"fields_{key}"] = fields
                field_payload[f"indices_{key}"] = ids
            np.savez_compressed(fields_output, **field_payload)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
