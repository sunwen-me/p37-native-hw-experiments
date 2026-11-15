"""Compare decoders on identical fields from a frozen P37 checkpoint.

Only validation tasks from supplied audit manifests are evaluated. Predicted
fields are cached once per task; no teacher fields, optimization, or search
are used. Geometry scoring is offline and never participates in selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "training/source/map_flow_matching_p1"),
    str(ROOT / "training/source/map_flow_matching"),
    str(ROOT / "training/source/map_flow_matching_p1/scripts"),
]
from flow_prior import load_flow_prior
from scripts.geometry_eval import score_task
from train_p37_ordered_field import file_sha256, load_map_disjoint_split
from utils.dataset import PathPlanningDataset


ARMS = (("legacy", 128), ("legacy", 512), ("scale_aware", 128), ("scale_aware", 512))


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def decode_fields(model, maps, starts, goals, fields, mode, horizon):
    """Use the deployment decoder and selector without querying the network."""
    return model.decode_fields(maps, starts, goals, fields, mode=mode, horizon=horizon)


def candidate_outcomes(safe, valid, accepted, selected):
    """Distinguish geometric success from an accepted successful deployment."""
    safe, valid, accepted = [np.asarray(value, dtype=bool) for value in (safe, valid, accepted)]
    selected = int(selected)
    return {
        "candidate_geometry_successes": int(safe.sum()),
        "decoder_valid_candidates": int(valid.sum()),
        "accepted_candidates": int(accepted.sum()),
        "accepted_safe_candidates": int((accepted & safe).sum()),
        "accepted_unsafe_candidates": int((accepted & ~safe).sum()),
        "any_geometry_success": bool(safe.any()),
        "any_accepted_safe": bool((accepted & safe).any()),
        "selected_geometry_success": bool(safe[selected]),
        "selected_accepted": bool(accepted[selected]),
        "deployment_success": bool(accepted[selected] and safe[selected]),
        "selected_accepted_unsafe": bool(accepted[selected] and not safe[selected]),
        "abstain": bool(not accepted.any()),
    }


def summarize(rows, candidates):
    result = {}
    for mode, horizon in ARMS:
        name = f"{mode}_{horizon}"
        groups = {"overall": rows}
        groups.update({shape: [row for row in rows if row["shape"] == shape]
                       for shape in sorted({row["shape"] for row in rows})})
        per_group = {}
        for key, group in groups.items():
            if not group:
                continue
            values = [row["arms"][name] for row in group]
            counts = {metric: sum(int(value[metric]) for value in values) for metric in (
                "candidate_geometry_successes", "decoder_valid_candidates", "accepted_candidates",
                "accepted_safe_candidates", "accepted_unsafe_candidates", "any_geometry_success",
                "any_accepted_safe", "selected_geometry_success", "deployment_success",
                "selected_accepted_unsafe", "abstain",
            )}
            per_group[key] = {
                "tasks": len(group), "candidate_count": len(group) * candidates, **counts,
                "candidate_geometry_success_rate": counts["candidate_geometry_successes"] / (len(group) * candidates),
                "decoder_valid_rate": counts["decoder_valid_candidates"] / (len(group) * candidates),
                "selected_geometry_success_rate": counts["selected_geometry_success"] / len(group),
                "deployment_success_rate": counts["deployment_success"] / len(group),
                "abstention_rate": counts["abstain"] / len(group),
            }
        result[name] = {**per_group.pop("overall"), "by_shape": per_group}
    return result


def paired_changes(rows):
    baseline = "legacy_128"
    result = {}
    for mode, horizon in ARMS[1:]:
        name = f"{mode}_{horizon}"
        gained, lost = [], []
        for row in rows:
            old = row["arms"][baseline]["deployment_success"]
            new = row["arms"][name]["deployment_success"]
            if new and not old:
                gained.append(row["task_id"])
            if old and not new:
                lost.append(row["task_id"])
        result[name] = {"baseline": baseline, "gained_task_ids": gained, "lost_task_ids": lost}
    return result


def load_cohorts(paths):
    cohorts, ordered, seen = {}, [], set()
    for path in paths:
        cohort = path.stem
        if cohort in cohorts:
            raise ValueError("audit filenames must have distinct stems")
        ids = list(dict.fromkeys(int(row["task_id"]) for row in json.loads(path.read_text())["tasks"]))
        if not ids:
            raise ValueError("audit contains no tasks")
        cohorts[cohort] = ids
        for task_id in ids:
            if task_id not in seen:
                ordered.append(task_id)
                seen.add(task_id)
    return cohorts, ordered


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--task-audit", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.candidates < 1 or args.num_steps < 1:
        parser.error("candidates and num-steps must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists() and not args.resume:
        parser.error("output already contains a run; choose a new directory or --resume")
    (output / "fields").mkdir(exist_ok=True)
    (output / "paths").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    # Match the stopped local evaluator's precision policy for all four arms.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model, state = load_flow_prior(args.checkpoint, args.device)
    config = state["config"]
    model.eval().requires_grad_(False)
    model.vector_field.latent_scale = float(config.get("latent_scale", 1.0))
    model.horizon, model.decoder_mode = 128, "legacy"
    model.decode_progress_smoothing = 0  # historical four-arm protocol
    seed = int(config.get("seed", 20261112)) + 200_003 if args.seed is None else args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(args.device)
    base = PathPlanningDataset(
        str(args.data_path), horizon=int(config.get("horizon", 128)), normalize=True,
        augment=False, distance_scale=model.distance_scale,
        validate_geometry=False, cache_distance_fields=True,
    )
    if base.geometry is None:
        raise ValueError("data must declare geometry_json")
    _, validation_ids, _ = load_map_disjoint_split(base.maps, args.split_path)
    cohorts, task_ids = load_cohorts(args.task_audit)
    if not np.isin(task_ids, validation_ids).all():
        raise ValueError("every evaluated task must belong to validation")
    model.robot_radius_grid = float(base.geometry.radius_grid)
    model.clearance_temperature = float(config.get("geometry_val_clearance_temperature", model.clearance_temperature))
    model.clearance_offset = float(config.get("geometry_val_clearance_offset", model.clearance_offset))
    model.decode_clearance_power = float(config.get("geometry_val_clearance_power", model.decode_clearance_power))
    model.decode_continuity_beta = float(config.get("geometry_val_continuity_beta", model.decode_continuity_beta))
    checkpoint_sha = file_sha256(args.checkpoint)
    binding = {
        "checkpoint_sha256": checkpoint_sha, "seed": seed, "task_ids": task_ids,
        "candidates": args.candidates, "num_steps": args.num_steps,
        "noise_scale": args.noise_scale, "sde_noise": 0.0,
        "distance_scale": model.distance_scale, "latent_scale": model.vector_field.latent_scale,
        "data_path": str(args.data_path.resolve()), "split_sha256": file_sha256(args.split_path),
        "torch": torch.__version__, "tf32": True, "cudnn_benchmark": False,
        "model_source_sha256": file_sha256(ROOT / "training/source/map_flow_matching_p1/p37_model.py"),
    }
    cache_binding = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    manifest = {
        "status": "running", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": state.get("epoch"), "binding": binding,
        "cache_binding_sha256": cache_binding,
        "task_audits": {str(path.resolve()): file_sha256(path) for path in args.task_audit},
        "decoder_settings": {key: getattr(model, key) for key in (
            "decode_beta", "decode_power", "decode_threshold", "clearance_temperature",
            "clearance_offset", "robot_radius_grid", "decode_clearance_power", "decode_continuity_beta",
            "decode_terminal_support_window", "decode_terminal_support_fraction",
            "decode_max_terminal_jump", "decode_max_step_grid", "decode_progress_smoothing",
        )},
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in (
            Path(__file__), ROOT / "training/source/map_flow_matching_p1/p37_model.py",
            ROOT / "training/source/map_flow_matching_p1/safe_field_decoder.py",
            ROOT / "training/source/map_flow_matching/utils/dataset.py",
            ROOT / "training/source/map_flow_matching_p1/scripts/geometry_eval.py",
        )},
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else None,
        "batch_size": 1, "arms": [f"{mode}_{horizon}" for mode, horizon in ARMS],
        "uses_teacher_fields": False, "optimizer_steps": 0, "training_restarted": False,
        "endpoint_protocol": "exact_hard_endpoints_v2", "completed_tasks": 0,
        "cohorts": cohorts,
    }
    if args.resume:
        old_manifest = json.loads((output / "run_manifest.json").read_text())
        if old_manifest["cache_binding_sha256"] != cache_binding:
            raise ValueError("resume cache does not match checkpoint/data/sampling order")
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"stage": "start", "tasks": len(task_ids), "seed": seed, "epoch": state.get("epoch")}), flush=True)
    generator = torch.Generator(device=args.device).manual_seed(seed)
    rows, started = [], time.perf_counter()
    for position, task_id in enumerate(task_ids):
        item = base[task_id]
        maps = torch.cat([item["map"], item["distance_field"]], dim=0)[None].to(args.device)
        starts, goals = item["start"][None].to(args.device), item["goal"][None].to(args.device)
        cache_path = output / "fields" / f"task_{task_id:06d}.npz"
        baseline = None
        generation_started = time.perf_counter()
        if cache_path.exists():
            if not args.resume:
                raise ValueError("field cache exists without --resume")
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["cache_binding"]) != cache_binding or int(cached["task_id"]) != task_id:
                    raise ValueError("field cache metadata mismatch")
                fields_np = cached["fields"]
                if str(cached["fields_sha256"]) != array_sha256(fields_np):
                    raise ValueError("field cache hash mismatch")
                fields = torch.from_numpy(fields_np).to(args.device)
                generator.set_state(torch.from_numpy(cached["rng_state_after"]))
        else:
            baseline = model.sample(
                maps, starts, goals, candidates=args.candidates, num_steps=args.num_steps,
                noise_scale=args.noise_scale, generator=generator, sde_noise=0.0,
            )
            fields = baseline["fields"]
            fields_np = fields.cpu().numpy()
            if not np.isfinite(fields_np).all():
                raise ValueError("non-finite model field")
            temporary = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary, fields=fields_np, latent=baseline["latent"].cpu().numpy(),
                rng_state_after=generator.get_state().cpu().numpy(), task_id=np.asarray(task_id),
                cache_binding=np.asarray(cache_binding), fields_sha256=np.asarray(array_sha256(fields_np)),
            )
            temporary.replace(cache_path)
        generation_seconds = time.perf_counter() - generation_started
        field_sha = array_sha256(fields_np)
        row = {
            "task_id": task_id, "shape": f"{maps.shape[-2]}x{maps.shape[-1]}",
            "cohorts": [name for name, ids in cohorts.items() if task_id in ids],
            "fields_sha256": field_sha, "field_cache": str(cache_path.relative_to(output)),
            "loaded_cached_fields": baseline is None,
            "sampling_and_cache_seconds": generation_seconds, "arms": {},
        }
        path_payload = {}
        for mode, horizon in ARMS:
            arm_started = time.perf_counter()
            name = f"{mode}_{horizon}"
            decoded = decode_fields(model, maps, starts, goals, fields, mode, horizon)
            if baseline is not None and (mode, horizon) == ("legacy", 128):
                assert torch.equal(decoded["paths"], baseline["paths"])
                assert torch.equal(decoded["decoder_valid"], baseline["decoder_support_valid"])
                assert torch.equal(decoded["selected_index"], baseline["selected_index"])
                assert torch.equal(decoded["selection_scores"], baseline["selection_scores"])
            normalized = decoded["paths"].cpu().numpy()
            world = base.denormalize_paths(normalized, [task_id])[0]
            safe = np.asarray(score_task((
                str(ROOT / "training/source/map_flow_matching"), world, base.maps[task_id],
                base.starts[task_id], base.goals[task_id], base.geometry.to_json(),
            )), dtype=bool)
            valid = decoded["decoder_valid"][0].cpu().numpy()
            accepted = decoded["accepted"][0].cpu().numpy()
            selected = int(decoded["selected_index"].item())
            endpoint_error = float(np.linalg.norm(
                world[:, [0, -1]] - np.stack([base.starts[task_id], base.goals[task_id]])[None], axis=-1,
            ).max())
            selection_scores = decoded["selection_scores"][0].cpu().numpy()
            row["arms"][name] = {
                **candidate_outcomes(safe, valid, accepted, selected),
                "fields_sha256": field_sha, "selected_index": selected,
                "candidate_geometry_safe": safe.tolist(), "decoder_valid": valid.tolist(),
                "accepted": accepted.tolist(), "endpoint_error_max_px": endpoint_error,
                "selection_scores": [float(value) if np.isfinite(value) else None for value in selection_scores],
                "diagnostics": {key: value[0].cpu().tolist() for key, value in decoded["diagnostics"].items()},
                "decode_and_geometry_seconds": time.perf_counter() - arm_started,
            }
            path_payload[f"paths_px_{name}"] = world
        if field_sha != array_sha256(fields.cpu().numpy()):
            raise ValueError("decoder mutated shared predicted fields")
        np.savez_compressed(output / "paths" / f"task_{task_id:06d}.npz", **path_payload)
        rows.append(row)
        with (output / "tasks.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        manifest["completed_tasks"] = position + 1
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "run_manifest.json", manifest)
        print(json.dumps({"stage": "task_complete", "completed": position + 1, "total": len(task_ids),
                          "task_id": task_id, "shape": row["shape"],
                          "deployment_success": {key: value["deployment_success"] for key, value in row["arms"].items()},
                          "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    if file_sha256(args.checkpoint) != checkpoint_sha:
        raise ValueError("checkpoint changed during evaluation")
    report = {
        "manifest": manifest, "tasks": rows, "summaries": summarize(rows, args.candidates),
        "paired_changes": paired_changes(rows),
        "cohorts": {name: {"summaries": summarize([row for row in rows if row["task_id"] in ids], args.candidates),
                           "paired_changes": paired_changes([row for row in rows if row["task_id"] in ids])}
                    for name, ids in cohorts.items()},
        "limitations": ["Development validation tasks, not independent blind holdout.",
                        "Synthetic reachable room grids; no real-robot or unreachable-task claim.",
                        "No weights trained with continuous labels or support-balanced loss yet."],
        "peak_gpu_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 1024**3
                                   if str(args.device).startswith("cuda") else None),
    }
    manifest["status"] = "complete"
    manifest["checkpoint_sha256_verified_after"] = checkpoint_sha
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "report.json", report)
    print(json.dumps({"stage": "complete", "summaries": {
        key: {metric: value[metric] for metric in (
            "tasks", "deployment_success", "deployment_success_rate", "selected_geometry_success_rate",
            "decoder_valid_rate", "selected_accepted_unsafe", "abstain",
        )} for key, value in report["summaries"].items()
    }}, indent=2), flush=True)


if __name__ == "__main__":
    main()
