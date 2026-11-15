"""Re-evaluate saved P37 pilot checkpoints on the fixed geometry subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_project.resolve()))
    sys.path.insert(0, str(args.source_project.resolve().parent))
    sys.path.insert(0, str(args.source_project.resolve().parent / "map_flow_matching_p1"))
    from utils.dataset import PathPlanningDataset
    from utils.robot_geometry import evaluate_robot_path
    from map_flow_matching_p1.p37_model import OrderedDenseFieldFlow

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset = PathPlanningDataset(
        str(args.data_path), horizon=int(config["horizon"]), normalize=True,
        augment=False, distance_scale=float(config["distance_scale"]),
        validate_geometry=False, cache_distance_fields=True,
    )
    with np.load(args.split_path, allow_pickle=False) as archive:
        task_ids = archive["geometry_val_ids"].astype(np.int64)
    device = torch.device(args.device)
    model = OrderedDenseFieldFlow(
        map_height=int(config["map_height"]), map_width=int(config["map_width"]),
        horizon=int(config["horizon"]), hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        decode_beta=float(config["decode_beta"]),
        decode_power=float(config["decode_power"]),
        decode_threshold=float(config["decode_threshold"]),
        distance_scale=float(config["distance_scale"]),
        robot_radius_grid=float(config["robot_radius_grid"]),
        clearance_temperature=float(config["clearance_temperature"]),
        clearance_offset=float(config["clearance_offset"]),
        decode_clearance_power=float(config["decode_clearance_power"]),
        decode_continuity_beta=float(config["decode_continuity_beta"]),
        decode_terminal_support_window=int(config["decode_terminal_support_window"]),
        decode_terminal_support_fraction=float(config["decode_terminal_support_fraction"]),
        decode_max_terminal_jump=float(config["decode_max_terminal_jump"]),
        decode_max_step_grid=config.get("decode_max_step_grid"),
        architecture=config.get("architecture", "cnn"),
        transformer_window_size=int(config.get("transformer_window_size", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.vector_field.latent_scale = float(config["latent_scale"])

    generator = torch.Generator(device=device).manual_seed(args.seed + 200_003)
    all_candidates: list[bool] = []
    task_has_safe_candidate: list[bool] = []
    selected_safe: list[bool] = []
    accepted: list[bool] = []
    decoder_valid: list[bool] = []
    with torch.inference_mode():
        for begin in range(0, len(task_ids), args.batch_size):
            ids = task_ids[begin:begin + args.batch_size]
            batch = [dataset[int(index)] for index in ids]
            map_input = torch.stack([
                torch.cat([item["map"], item["distance_field"]], dim=0)
                for item in batch
            ]).to(device)
            starts = torch.stack([item["start"] for item in batch]).to(device)
            goals = torch.stack([item["goal"] for item in batch]).to(device)
            output = model.sample(
                map_input, starts, goals,
                candidates=int(config["geometry_val_candidates"]),
                num_steps=int(config["geometry_val_steps"]),
                noise_scale=float(config["geometry_val_noise_scale"]),
                generator=generator,
            )
            paths = dataset.denormalize_coordinates(
                output["paths"].cpu().numpy(), task_ids=ids,
            )
            selected = output["selected_index"].cpu().numpy()
            batch_accepted = output["accepted"].cpu().numpy()
            batch_valid = output["decoder_valid"].cpu().numpy()
            for local, task_id in enumerate(ids):
                candidate_safe = [bool(evaluate_robot_path(
                    candidate, dataset.maps[task_id], dataset.starts[task_id],
                    dataset.goals[task_id], dataset.geometry,
                    endpoint_tolerance=1e-5,
                )["success"]) for candidate in paths[local]]
                all_candidates.extend(candidate_safe)
                task_has_safe_candidate.append(any(candidate_safe))
                selected_safe.append(bool(candidate_safe[int(selected[local])]))
                accepted.append(bool(batch_accepted[local].any()))
                decoder_valid.extend(batch_valid[local].tolist())

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture": config.get("architecture", "cnn"),
        "seed": args.seed,
        "geometry_validation_tasks": int(len(task_ids)),
        "candidates_per_task": int(config["geometry_val_candidates"]),
        "flow_steps": int(config["geometry_val_steps"]),
        "train_loss": float(checkpoint["loss"]),
        "candidate_success_rate": float(np.mean(all_candidates)),
        "oracle_best_of_k_success_rate": float(np.mean(task_has_safe_candidate)),
        "selected_path_success_rate": float(np.mean(selected_safe)),
        "selector_miss_rate": float(np.mean(
            np.asarray(task_has_safe_candidate) & ~np.asarray(selected_safe)
        )),
        "decoder_valid_rate": float(np.mean(decoder_valid)),
        "decoder_invalid_rate": float(1.0 - np.mean(decoder_valid)),
        "accepted_task_rate": float(np.mean(accepted)),
        "abstain_rate": float(1.0 - np.mean(accepted)),
        "safety_evaluator": "evaluate_robot_path / disk_cells_v1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
