"""Audit route-label/decoder compatibility without loading a learned model.

Reads trusted local dataset archives, including their NumPy map objects. This
is an oracle diagnostic, not evidence of trained-model planning performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "training/source/map_flow_matching_p1/scripts"),
    str(ROOT / "training/source/map_flow_matching"),
]
import train_p37_ordered_field as trainer
from safe_field_decoder import distance_field_to_soft_clearance, safe_ordered_field_to_path
from scripts.geometry_eval import score_task
from utils.robot_geometry import RobotGeometry


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-shape", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--decode-horizon", type=int, default=None)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Inspect explicit validation failures instead of sampling.")
    parser.add_argument("--exclude-audit", type=Path, action="append", default=[],
                        help="Exclude task IDs used in a prior audit before sampling.")
    args = parser.parse_args()
    if args.per_shape < 1:
        parser.error("--per-shape must be positive")
    torch.set_num_threads(2)
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    decode_horizon = config["horizon"] if args.decode_horizon is None else args.decode_horizon
    if decode_horizon < 2:
        parser.error("--decode-horizon must be >= 2")
    with np.load(args.data_dir / "train_data.npz", allow_pickle=True) as data:
        maps, map_ids = data["unique_maps"], data["map_ids"]
        paths, starts, goals = data["paths"], data["starts"], data["goals"]
        geometry_json = str(data["geometry_json"])
    geometry = RobotGeometry.from_json(geometry_json)
    radius = geometry.radius_grid
    with np.load(args.split_path, allow_pickle=False) as split:
        validation_ids = split["val_ids"]
    excluded = set()
    for audit_path in args.exclude_audit:
        excluded.update(row["task_id"] for row in json.loads(audit_path.read_text())["tasks"])
    sampling_ids = validation_ids[~np.isin(validation_ids, list(excluded))]
    base = SimpleNamespace(shape_at=lambda i: tuple(maps[int(map_ids[int(i)])].shape))
    shape_count = len({base.shape_at(i) for i in validation_ids})
    seed = args.sample_seed if args.sample_seed is not None else config["seed"] + 600_007
    ids = trainer.balanced_geometry_sample(
        base, sampling_ids, shape_count * args.per_shape, np.random.default_rng(seed),
    )
    if args.task_ids is not None:
        ids = np.asarray(args.task_ids, dtype=np.int64)
        if not np.isin(ids, validation_ids).all():
            parser.error("explicit task IDs must belong to validation")
    variants = [("segments", "scale_aware")]
    if args.include_legacy:
        variants = [("points", "legacy"), ("segments", "legacy")] + variants
    rows = []
    for task_id in ids:
        task_id = int(task_id)
        image = maps[int(map_ids[task_id])]
        height, width = image.shape
        scale = np.asarray([(width - 1) / 2, (height - 1) / 2], dtype=np.float32)
        route = torch.from_numpy(paths[task_id] / scale - 1)[None, None]
        start = torch.from_numpy(starts[task_id] / scale - 1)[None]
        goal = torch.from_numpy(goals[task_id] / scale - 1)[None]
        distance = cv2.distanceTransform(
            np.pad((1 - image).astype(np.uint8) * 255, 1), cv2.DIST_L2, 5,
        )[1:-1, 1:-1]
        feature = torch.from_numpy(np.clip(distance / config["distance_scale"], 0, 1))[None, None]
        clearance = distance_field_to_soft_clearance(
            feature, config["distance_scale"], radius,
            config["geometry_val_clearance_temperature"], config["geometry_val_clearance_offset"],
        )
        for rasterization, decoder_mode in variants:
            target = trainer.ordered_targets(
                route, start, goal, config["mask_thickness"], height, width,
                subpixel=config.get("subpixel_rasterization", False),
                rasterization_mode=rasterization,
            )[:, 0]
            positive = target[0, 0].numpy() > 0
            components = int(label(positive, structure=np.ones((3, 3)))[1])
            target, _ = trainer.apply_clearance_target_gate(
                target, feature, trainer.endpoint_fields(start, goal, height, width),
                radius=radius, margin=config["target_clearance_margin"],
                temperature=config["target_clearance_temperature"], floor=config["target_clearance_floor"],
                distance_scale=config["distance_scale"], clearance_offset=config["clearance_offset"],
            )
            decoded, valid, diagnostics = safe_ordered_field_to_path(
                target[:, :1], target[:, 1:2], clearance, start, goal, decode_horizon,
                beta=config["decode_beta"], power=config["decode_power"],
                clearance_power=config["geometry_val_clearance_power"],
                continuity_beta=config["geometry_val_continuity_beta"],
                occupancy_threshold=config["decode_threshold"],
                terminal_support_window=config["decode_terminal_support_window"],
                terminal_support_fraction=config["decode_terminal_support_fraction"],
                max_terminal_jump=config["decode_max_terminal_jump"],
                max_path_step=(None if config.get("decode_max_step_grid") is None
                               else 2 * config["decode_max_step_grid"] / 63),
                mode=decoder_mode, return_diagnostics=True,
            )
            world = (decoded.numpy()[0] + 1) * scale
            raw_endpoint_error = float(max(
                np.linalg.norm(world[0] - starts[task_id]),
                np.linalg.norm(world[-1] - goals[task_id]),
            ))
            # Restore exact input endpoints only after verifying that the
            # decoder already obeyed the hard normalized-endpoint contract.
            assert torch.equal(decoded[:, 0], start) and torch.equal(decoded[:, -1], goal)
            world[0], world[-1] = starts[task_id], goals[task_id]
            safe = bool(score_task((
                str(ROOT / "training/source/map_flow_matching"), world[None],
                image, starts[task_id], goals[task_id], geometry_json,
            ))[0])
            rows.append({
                "task_id": task_id, "shape": f"{height}x{width}",
                "rasterization": rasterization, "decoder_mode": decoder_mode,
                "mask_components": components, "positive_fraction": float(positive.mean()),
                "decoder_valid": bool(valid.item()), "geometry_success": safe,
                "valid_and_safe": bool(valid.item()) and safe,
                "raw_endpoint_roundtrip_error_px": raw_endpoint_error,
                "final_step_px": float(np.linalg.norm(world[-1] - world[-2])),
                "decoder_diagnostics": {key: value.item() for key, value in diagnostics.items()},
            })
    summaries = {}
    for rasterization, decoder_mode in variants:
        selected = [r for r in rows if (r["rasterization"], r["decoder_mode"]) == (rasterization, decoder_mode)]
        by_shape = {}
        for shape in sorted({r["shape"] for r in selected}):
            group = [r for r in selected if r["shape"] == shape]
            by_shape[shape] = {"tasks": len(group), **{
                key: sum(int(r[key]) for r in group)
                for key in ("decoder_valid", "geometry_success", "valid_and_safe")
            }}
        summaries[f"{rasterization}/{decoder_mode}"] = {
            "tasks": len(selected), **{
                key: sum(int(r[key]) for r in selected)
                for key in ("decoder_valid", "geometry_success", "valid_and_safe")
            }, "by_shape": by_shape,
        }
    report = {
        "oracle_only": True, "sample_seed": int(seed), "per_shape": args.per_shape,
        "decode_horizon": decode_horizon,
        "progress_precision_policy": "supported_progress_gradient_and_pixel_bins_v2",
        "explicit_task_ids": args.task_ids,
        "excluded_tasks": len(excluded),
        "endpoint_protocol": "Restore exact hard normalized endpoints; keep strict geometric checks.",
        "limitations": "Synthetic reachable room grids; teacher fields do not measure learned-model quality.",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": trainer.file_sha256(args.checkpoint),
        "data_dir": str(args.data_dir.resolve()), "split_path": str(args.split_path.resolve()),
        "raster_settings": {key: config.get(key) for key in (
            "mask_thickness", "subpixel_rasterization", "decode_beta", "decode_max_terminal_jump",
        )},
        "summaries": summaries, "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
