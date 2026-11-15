"""Frozen-field development comparison of bounded adaptive continuation.

Geometry labels are used only after decoding. The default fallback leaves a
task's existing accepted candidates unchanged; it runs only after abstention.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import shutil
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "training/source/map_flow_matching_p1"),
               str(ROOT / "training/source/map_flow_matching")]
from adaptive_field_decoder import adaptive_ordered_field_to_path
from package_p37_decoder import digest
from p37_model import candidate_selection_scores
from p37_planner import P37FlowPlanner, _p37_map_input
from safe_field_decoder import distance_field_to_soft_clearance
from scripts.geometry_eval import score_task


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("remaining", "large", "full"), default="remaining")
    parser.add_argument("--strategy", choices=("adaptive", "fallback"), default="fallback")
    parser.add_argument("--implementation", choices=("prototype", "sampler"), default="prototype")
    parser.add_argument("--prototype-dir", type=Path)
    parser.add_argument("--comparison-mode", choices=("scale_aware", "adaptive_fallback"), default="scale_aware")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("choose an empty output directory")
    (out / "paths").mkdir(parents=True)
    base_dir = args.run_dir / "evaluation/paired_best_val_epoch15"
    original = json.loads((base_dir / "report.json").read_text())
    baseline = json.loads((args.baseline_dir / "report.json").read_text())
    by_id = {r["task_id"]: r for r in baseline["tasks"]}
    tasks = [r for r in original["tasks"] if args.scope == "full" or (
        args.scope == "large" and r["shape"] in ("512x768", "768x512")) or (
        args.scope == "remaining" and not by_id[r["task_id"]]["deployment_success"])]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = P37FlowPlanner(args.checkpoint, device=args.device).model
    assert model.decoder_mode in {"scale_aware", "adaptive_fallback", "bounded_fallback"} and model.decode_progress_smoothing == 7 and model.horizon == 512
    if args.implementation == "sampler":
        assert model.decoder_mode in {"adaptive_fallback", "bounded_fallback"} and args.strategy == "fallback"
    if model.decoder_mode == "bounded_fallback":
        assert args.implementation == "sampler" and args.comparison_mode == "adaptive_fallback"
    settings = original["manifest"]["decoder_settings"]
    for key, expected in settings.items():
        assert getattr(model, key) == expected, key
    manifest = {
        "status": "running", "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
        "baseline_report_sha256": digest(args.baseline_dir / "report.json"),
        "fields_report_sha256": digest(base_dir / "report.json"),
        "scope": args.scope, "strategy": args.strategy, "implementation": args.implementation,
        "decoder_mode": model.decoder_mode,
        "comparison_mode": args.comparison_mode,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                          (Path(__file__), ROOT / "inference/adaptive_field_decoder.py", ROOT / "inference/p37_model.py")},
        "training": False, "teacher_fields_used": False, "network_queries": 0,
        "holdout_used": False, "independent_holdout": False,
        "safety_thresholds_unchanged": True, "internal_move_limit_px": 4,
        "max_internal_steps": 2048, "horizon": 512,
    }
    (out / "source").mkdir()
    for path in (Path(__file__), ROOT / "inference/adaptive_field_decoder.py", ROOT / "inference/p37_model.py",
                 ROOT / "inference/safe_field_decoder.py", ROOT / "inference/p37_planner.py"):
        shutil.copy2(path, out / "source" / path.name)
    write_json(out / "manifest.json", manifest)
    with np.load(original["manifest"]["binding"]["data_path"], allow_pickle=True) as data:
        maps, map_ids, starts, goals = data["unique_maps"], data["map_ids"], data["starts"], data["goals"]
        geometry = str(data["geometry_json"])
    rows = []
    def fail_network_query(*_):
        raise AssertionError("frozen-field replay must not query the network")
    model.vector_field.register_forward_pre_hook(fail_network_query)
    for number, task in enumerate(tasks, 1):
        begin = time.monotonic()
        tid = task["task_id"]
        previous = by_id[tid]
        image = maps[int(map_ids[tid])]
        height, width = image.shape
        scale = np.array([(width - 1) / 2, (height - 1) / 2], np.float32)
        map_input = torch.from_numpy(_p37_map_input(image, model.distance_scale))[None].to(args.device)
        start = torch.from_numpy(starts[tid] / scale - 1)[None].to(args.device)
        goal = torch.from_numpy(goals[tid] / scale - 1)[None].to(args.device)
        with np.load(base_dir / task["field_cache"], allow_pickle=False) as saved:
            fields_np = saved["fields"]
        assert hashlib.sha256(np.ascontiguousarray(fields_np).tobytes()).hexdigest() == task["fields_sha256"]
        fields = torch.from_numpy(fields_np).to(args.device)
        result = model.decode_fields(map_input, start, goal, fields, mode=args.comparison_mode)
        assert result["accepted"].tolist()[0] == previous["accepted"]
        assert int(result["selected_index"].item()) == previous["selected_index"]
        old_px = (result["paths"].cpu().numpy()[0] + 1) * scale
        old_px[:, 0], old_px[:, -1] = starts[tid], goals[tid]
        with np.load(args.baseline_dir / "paths" / f"task_{tid:06d}.npz") as saved:
            assert np.abs(old_px - saved["paths_px"]).max() <= 1e-4
            if args.comparison_mode == "adaptive_fallback":
                assert np.array_equal(old_px, saved["paths_px"])
        use_adaptive = args.strategy == "adaptive" or not result["accepted"].any()
        if args.implementation == "sampler":
            latent = torch.zeros(1, 4, model.latent_dim, device=args.device)
            with patch.object(model, "_integrate", return_value=fields.reshape(4, 2, height, width)):
                result = model.sample(map_input, start, goal, candidates=4, latent=latent)
            attempt_key = "bounded_attempted" if model.decoder_mode == "bounded_fallback" else "adaptive_attempted"
            assert bool(result["diagnostics"][attempt_key].any()) == bool(use_adaptive)
        elif use_adaptive:
            corrected = result["decoder_fields"][0]
            clearance = distance_field_to_soft_clearance(
                map_input[:, 1:2], model.distance_scale, model.robot_radius_grid,
                model.clearance_temperature, model.clearance_offset).expand(4, -1, -1, -1)
            paths, valid, diag = adaptive_ordered_field_to_path(
                corrected[:, :1].clamp(0, 1), corrected[:, 1:2].clamp(0, 1), clearance,
                start.expand(4, -1), goal.expand(4, -1), model.horizon,
                beta=model.decode_beta, power=model.decode_power,
                clearance_power=model.decode_clearance_power, continuity_beta=model.decode_continuity_beta,
                occupancy_threshold=model.decode_threshold,
                terminal_support_window=model.decode_terminal_support_window,
                terminal_support_fraction=model.decode_terminal_support_fraction,
                max_terminal_jump=model.decode_max_terminal_jump,
                max_path_step=None if model.decode_max_step_grid is None else 2 * model.decode_max_step_grid / 63,
                seed_paths=result["paths"][0],
                seed_lengths=result["diagnostics"]["first_missing_support_step"][0].clamp(1, model.horizon - 1))
            scores = candidate_selection_scores(result["decoder_fields"], paths[None], map_input, valid[None])
            accepted = valid[None] & torch.isfinite(scores)
            selected = torch.where(accepted.any(-1), scores.argmax(-1), torch.zeros(1, device=start.device, dtype=torch.long))
            result.update(paths=paths[None], accepted=accepted, selected_index=selected,
                          diagnostics={k: v[None] for k, v in diag.items()})
        assert np.array_equal(fields.cpu().numpy(), fields_np)
        px = (result["paths"].cpu().numpy()[0] + 1) * scale
        px[:, 0], px[:, -1] = starts[tid], goals[tid]
        assert px.shape == (4, 512, 2) and np.isfinite(px).all()
        assert torch.equal(result["paths"][:, :, 0], start[:, None].expand(-1, 4, -1))
        assert torch.equal(result["paths"][:, :, -1], goal[:, None].expand(-1, 4, -1))
        if not use_adaptive:
            assert np.array_equal(px, old_px)
        prototype_path = None if args.prototype_dir is None else args.prototype_dir / "paths" / f"task_{tid:06d}.npz"
        if prototype_path is not None and prototype_path.exists():
            with np.load(prototype_path) as saved:
                assert np.abs(px - saved["paths_px"]).max() <= 1e-4
        safe = np.asarray(score_task((str(ROOT / "training/source/map_flow_matching"), px,
                                     image, starts[tid], goals[tid], geometry)), dtype=bool)
        accepted = result["accepted"].cpu().numpy()[0]
        selected = int(result["selected_index"].item())
        row = {
            "task_id": tid, "shape": task["shape"], "fields_sha256": task["fields_sha256"],
            "baseline_success": previous["deployment_success"], "used_adaptive": bool(use_adaptive),
            "deployment_success": bool(accepted[selected] and safe[selected]),
            "geometry_safe": safe.tolist(), "accepted": accepted.tolist(), "selected_index": selected,
            "any_geometry_success": bool(safe.any()), "any_accepted_safe": bool((safe & accepted).any()),
            "accepted_unsafe_candidates": int((accepted & ~safe).sum()),
            "selected_accepted_unsafe": bool(accepted[selected] and not safe[selected]),
            "diagnostics": {k: v.cpu().tolist()[0] for k, v in result["diagnostics"].items()},
            "seconds": time.monotonic() - begin,
        }
        rows.append(row)
        np.savez_compressed(out / "paths" / f"task_{tid:06d}.npz", paths_px=px)
        with (out / "tasks.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({"completed": number, "total": len(tasks), "task": tid,
                          "success": row["deployment_success"], "accepted_unsafe": row["accepted_unsafe_candidates"],
                          "seconds": row["seconds"], "diagnostics": row["diagnostics"] if use_adaptive else {}}), flush=True)
    groups = {"overall": rows, "large": [r for r in rows if r["shape"] in ("512x768", "768x512")]}
    groups.update({shape: [r for r in rows if r["shape"] == shape] for shape in sorted({r["shape"] for r in rows})})
    summaries = {name: {"tasks": len(rs), "success": sum(r["deployment_success"] for r in rs),
                        "gains": [r["task_id"] for r in rs if r["deployment_success"] and not r["baseline_success"]],
                        "losses": [r["task_id"] for r in rs if r["baseline_success"] and not r["deployment_success"]],
                        "accepted_unsafe_candidates": sum(r["accepted_unsafe_candidates"] for r in rs)}
                 for name, rs in groups.items()}
    manifest.update(status="complete", finished_local=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(out / "manifest.json", manifest)
    write_json(out / "report.json", {"manifest": manifest, "summaries": summaries, "tasks": rows})
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
