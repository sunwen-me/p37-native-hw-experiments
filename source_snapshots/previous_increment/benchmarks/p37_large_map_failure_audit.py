"""Diagnose frozen P37 fields; oracle substitutions are analysis only.

No model training, deployment edits, threshold relaxation, or holdout use.
Teacher substitutions intentionally use labels and cannot be reported as
model planning performance. Their channels can also disagree on route mode.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field_component_metrics(occupancy, progress, target, image, distance, start, goal):
    """Pixel support topology is diagnostic, not a swept-disk path test."""
    support = occupancy > .1
    conservative_centers = distance - 2 ** -.5 > 1.5
    components, count = label(support & conservative_centers, np.ones((3, 3)))
    sx, sy = np.rint(start).astype(int)
    gx, gy = np.rint(goal).astype(int)
    start_component, goal_component = int(components[sy, sx]), int(components[gy, gx])
    teacher_support = target[0] > .1
    tail_support = teacher_support & (target[1] >= .875)
    teacher_values = occupancy[teacher_support]
    return {
        "support_components_conservative_centers": int(count),
        "start_goal_same_support_component": bool(start_component and start_component == goal_component),
        "support_fraction": float(support.mean()),
        "teacher_corridor_occupancy_recall_at_point1": float(support[teacher_support].mean()),
        "teacher_tail_occupancy_recall_at_point1": float(support[tail_support].mean()),
        "teacher_corridor_occupancy_quantiles": np.quantile(teacher_values, [.05, .5, .95]).tolist(),
        "teacher_corridor_progress_mae": float(np.abs(progress[teacher_support] - target[1][teacher_support]).mean()),
        "teacher_tail_progress_mae": float(np.abs(progress[tail_support] - target[1][tail_support]).mean()),
    }


def route_metrics(route, fields):
    delta = np.diff(route.astype(np.float64), axis=0)
    segment_length = np.linalg.norm(delta, axis=1)
    length = float(segment_length.sum())
    cumulative = np.r_[0., segment_length.cumsum()]
    arc = np.linspace(0, length, max(128, int(np.ceil(length * 2))))
    xy = np.stack([np.interp(arc, cumulative, route[:, axis]) for axis in (0, 1)], axis=1)
    reference_progress = arc / max(length, 1e-6)
    rows = []
    for field in fields:
        occupancy = map_coordinates(field[0], [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        progress = map_coordinates(field[1], [xy[:, 1], xy[:, 0]], order=1, mode="nearest")
        on = occupancy > .1
        drops = np.diff(progress)
        rows.append({
            "reference_centerline_occupancy_recall": float(on.mean()),
            "reference_centerline_progress_mae": float(np.abs(progress - reference_progress).mean()),
            "reference_centerline_large_backward_steps": int((drops < -.02).sum()),
            "reference_centerline_progress_mae_on_predicted_support": (
                float(np.abs(progress[on] - reference_progress[on]).mean()) if on.any() else None
            ),
        })
    directions = delta[np.linalg.norm(delta, axis=1) > 1e-6]
    direction_changes = 0
    if len(directions) > 1:
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        direction_changes = int((np.linalg.norm(np.diff(directions, axis=0), axis=1) > .1).sum())
    return {
        "reference_length_px": length,
        "endpoint_chord_px": float(np.linalg.norm(route[-1] - route[0])),
        "reference_turns": direction_changes,
        "candidates": rows,
    }


def collision_metrics(paths, image, radius, fast_segment_clearance):
    result = []
    for path in paths:
        clearance = np.asarray([
            fast_segment_clearance(image, a, b, radius)
            for a, b in zip(path[:-1], path[1:])
        ])
        hits = clearance <= radius + 1e-10
        hit_indices = np.flatnonzero(hits)
        steps = np.linalg.norm(np.diff(path.astype(np.float64), axis=0), axis=1)
        result.append({
            "geometry_safe": bool(not hits.any()),
            "prefix_before_forced_goal_safe": bool(not hits[:-1].any()),
            "forced_goal_segment_collision": bool(hits[-1]),
            "collision_segments": int(hits.sum()),
            "first_collision_segment": int(hit_indices[0] + 1) if len(hit_indices) else None,
            "last_collision_segment": int(hit_indices[-1] + 1) if len(hit_indices) else None,
            "final_step_px": float(steps[-1]),
            "max_step_px": float(steps.max()),
            "decoded_length_px": float(steps.sum()),
            "min_clearance_capped_at_radius_plus2": float(clearance.min()),
        })
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "paths").mkdir(exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Completed diagnostic exists; use a new output directory")
    root = run / "source"
    sys.path[:0] = [
        str(root / "training/source/map_flow_matching_p1"),
        str(root / "training/source/map_flow_matching"),
        str(root / "training/source/map_flow_matching_p1/scripts"),
    ]
    evaluator = load_module("frozen_p37_eval", root / "benchmarks/p37_paired_decoder_eval.py")
    import train_p37_ordered_field as trainer
    from scripts.generate_native_hw_dataset import fast_segment_clearance
    from safe_field_decoder import _progress_geometry

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    state = torch.load(run / "checkpoints/best_p37.pth", map_location="cpu", weights_only=False)
    config = state["config"]
    evaldir = run / "evaluation/paired_best_val_epoch15"
    original = json.loads((evaldir / "report.json").read_text())
    assert digest(run / "checkpoints/best_p37.pth") == original["manifest"]["binding"]["checkpoint_sha256"]
    settings = original["manifest"]["decoder_settings"]
    decoder = SimpleNamespace(**settings)
    decoder.distance_scale = original["manifest"]["binding"]["distance_scale"]
    decoder.decode_max_step_grid = settings["decode_max_step_grid"]
    data_path = Path(original["manifest"]["binding"]["data_path"])
    with np.load(data_path, allow_pickle=True) as d:
        maps, map_ids = d["unique_maps"], d["map_ids"]
        routes, starts_px, goals_px = d["paths"], d["starts"], d["goals"]
        geometry_json = str(d["geometry_json"])
    device = torch.device(args.device)
    large = [t for t in original["tasks"] if t["shape"] in ("512x768", "768x512")]
    controls = [
        t for shape in ("64x64", "127x193", "256x384", "384x256")
        for t in [row for row in original["tasks"] if row["shape"] == shape][:2]
    ]
    rows = []
    initial = {
        "status": "running", "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_root": str(root), "run_dir": str(run), "device": str(device),
        "script_sha256": digest(Path(__file__)), "checkpoint_sha256": digest(run / "checkpoints/best_p37.pth"),
        "paired_report_sha256": digest(evaldir / "report.json"), "large_task_ids": [t["task_id"] for t in large],
        "control_task_ids": [t["task_id"] for t in controls], "training": False,
        "oracle_substitutions_are_not_deployable": True,
        "safety_thresholds_relaxed": False, "holdout_used": False,
    }
    write_json(output / "manifest.json", initial)
    for position, task in enumerate(large + controls, 1):
        tid = task["task_id"]
        image = maps[int(map_ids[tid])]
        height, width = image.shape
        scale = np.asarray([(width - 1) / 2, (height - 1) / 2], dtype=np.float32)
        start_px, goal_px = starts_px[tid], goals_px[tid]
        start = torch.from_numpy(start_px / scale - 1)[None].to(device)
        goal = torch.from_numpy(goal_px / scale - 1)[None].to(device)
        reference = torch.from_numpy(routes[tid] / scale - 1)[None, None].to(device)
        distance = cv2.distanceTransform(np.pad((1 - image).astype(np.uint8) * 255, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        feature = torch.from_numpy(np.clip(distance / decoder.distance_scale, 0, 1)).to(device)[None, None]
        map_input = torch.cat([torch.from_numpy(image.astype(np.float32)).to(device)[None, None], feature], dim=1)
        target = trainer.ordered_targets(reference, start, goal, config["mask_thickness"], height, width,
                                         subpixel=config["subpixel_rasterization"], rasterization_mode="segments")[:, 0]
        target, _ = trainer.apply_clearance_target_gate(
            target, feature, trainer.endpoint_fields(start, goal, height, width),
            radius=decoder.robot_radius_grid, margin=config["target_clearance_margin"],
            temperature=config["target_clearance_temperature"], floor=config["target_clearance_floor"],
            distance_scale=decoder.distance_scale, clearance_offset=config["clearance_offset"],
        )
        target_np = target[0].cpu().numpy()
        with np.load(evaldir / task["field_cache"], allow_pickle=False) as d:
            fields_np = d["fields"]
        assert evaluator.array_sha256(fields_np) == task["fields_sha256"]
        predicted = torch.from_numpy(fields_np).to(device)
        # Nearest-corridor progress is explicitly a teacher-assisted upper-bound
        # intervention, not a prediction. Raw teacher progress is zero outside
        # its route; keeping both variants exposes that hybrid mismatch.
        nearest = distance_transform_edt(~(target_np[0] > 0), return_distances=False, return_indices=True)
        extended_progress_np = target_np[1][tuple(nearest)]
        extended_progress = torch.from_numpy(extended_progress_np).to(device)
        teacher = target[:, None]
        teacher_occ = predicted.clone()
        teacher_occ[:, :, :1] = teacher[:, :, :1]
        teacher_progress_raw = predicted.clone()
        teacher_progress_raw[:, :, 1:2] = teacher[:, :, 1:2]
        teacher_progress_extended = predicted.clone()
        teacher_progress_extended[:, :, 1] = extended_progress
        variants = [
            ("prediction", predicted, 512),
            ("oracle_occupancy_predicted_progress", teacher_occ, 512),
            ("predicted_occupancy_oracle_progress_raw", teacher_progress_raw, 512),
            ("predicted_occupancy_oracle_progress_extended", teacher_progress_extended, 512),
            ("oracle_both", teacher, 512),
            ("prediction_1024_points", predicted, 1024),
        ]
        row = {"task_id": tid, "shape": task["shape"], "is_large": task in large,
               "baseline_deployment_success": task["arms"]["scale_aware_512"]["deployment_success"],
               "field_sha256": task["fields_sha256"], "variants": {},
               "reference": route_metrics(routes[tid], fields_np[0])}
        field_stats = [field_component_metrics(f[0], f[1], target_np, image, distance, start_px, goal_px)
                       for f in fields_np[0]]
        teacher_stats = field_component_metrics(target_np[0], target_np[1], target_np, image, distance, start_px, goal_px)
        row["predicted_field_metrics"] = field_stats
        row["teacher_field_metrics"] = teacher_stats
        probability = ((target[:, :1] - decoder.decode_threshold) / (1 - decoder.decode_threshold)).clamp_min(0).pow(decoder.decode_power)
        teacher_precision, _ = _progress_geometry(probability, target[:, 1:2])
        row["teacher_progress_beta"] = float(teacher_precision.item() * decoder.decode_beta)
        path_payload = {}
        for name, fields, horizon in variants:
            decoded = evaluator.decode_fields(decoder, map_input, start, goal, fields, "scale_aware", horizon)
            normalized = decoded["paths"].cpu().numpy()[0]
            assert np.array_equal(normalized[:, 0], np.broadcast_to(start.cpu().numpy(), normalized[:, 0].shape))
            assert np.array_equal(normalized[:, -1], np.broadcast_to(goal.cpu().numpy(), normalized[:, -1].shape))
            world = (normalized + 1) * scale
            world[:, 0], world[:, -1] = start_px, goal_px
            safe = np.asarray(evaluator.score_task((str(root / "training/source/map_flow_matching"), world,
                                                    image, start_px, goal_px, geometry_json)))
            accepted = decoded["accepted"].cpu().numpy()[0]
            valid = decoded["decoder_valid"].cpu().numpy()[0]
            selected = int(decoded["selected_index"].item())
            outcomes = evaluator.candidate_outcomes(safe, valid, accepted, selected)
            outcomes.update(selected_index=selected, geometry_safe=safe.tolist(), accepted=accepted.tolist(),
                            diagnostics={k:v.cpu().tolist()[0] for k,v in decoded["diagnostics"].items()})
            if name == "prediction":
                expected = task["arms"]["scale_aware_512"]
                assert outcomes["deployment_success"] == expected["deployment_success"]
                assert safe.tolist() == expected["candidate_geometry_safe"]
                assert accepted.tolist() == expected["accepted"]
                with np.load(evaldir / "paths" / f"task_{tid:06d}.npz", allow_pickle=False) as saved:
                    error = float(np.abs(world - saved["paths_px_scale_aware_512"]).max())
                    assert error <= 1e-4, error
                row["baseline_reproduction_max_path_error_px"] = error
                row["baseline_collisions"] = collision_metrics(world, image, decoder.robot_radius_grid, fast_segment_clearance)
                assert [x["geometry_safe"] for x in row["baseline_collisions"]] == safe.tolist()
            row["variants"][name] = outcomes
            path_payload[name] = world
        np.savez_compressed(output / "paths" / f"task_{tid:06d}.npz", **path_payload)
        rows.append(row)
        with (output / "tasks.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"position":position,"tasks":len(large + controls),"task_id":tid,"shape":row["shape"],
                          "variant_deployment_success":{k:v["deployment_success"] for k,v in row["variants"].items()}},ensure_ascii=False),flush=True)
    summaries = {}
    for group in ("large", "controls", "512x768", "768x512"):
        selected = [r for r in rows if (r["is_large"] if group == "large" else not r["is_large"] if group == "controls" else r["shape"] == group)]
        if not selected:
            continue
        summaries[group] = {"tasks":len(selected),"variants":{}}
        for name in selected[0]["variants"]:
            values = [r["variants"][name] for r in selected]
            summaries[group]["variants"][name] = {key:sum(int(v[key]) for v in values) for key in (
                "deployment_success","any_geometry_success","any_accepted_safe","selected_accepted_unsafe","abstain")}
            summaries[group]["variants"][name]["rescued_baseline_failures"] = [r["task_id"] for r in selected if not r["baseline_deployment_success"] and r["variants"][name]["deployment_success"]]
            summaries[group]["variants"][name]["lost_baseline_successes"] = [r["task_id"] for r in selected if r["baseline_deployment_success"] and not r["variants"][name]["deployment_success"]]
    result = {"manifest":initial | {"status":"complete","finished_local":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()},
              "summaries":summaries,"tasks":rows,
              "limitations":["Oracle channel substitutions use reference labels and cannot be deployed or counted as model performance.",
                             "Single-channel substitutions can mismatch route modes and off-support progress; the raw and extended progress controls expose this limitation.",
                             "Connected components use a conservative cell-center mask and 8-neighbor connectivity; they are not the exact swept-disk safety test.",
                             "1024 output points alter sampling density, not network architecture; no claim about a trained 1024-point network.",
                             "Same development validation tasks; no untouched holdout evaluation."]}
    write_json(output / "report.json", result)
    write_json(output / "manifest.json", result["manifest"])
    print(json.dumps({"summaries":summaries},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
