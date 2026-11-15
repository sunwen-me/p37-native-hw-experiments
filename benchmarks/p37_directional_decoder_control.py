"""Frozen-field controls after exact adaptive rejection auditing.

Only previously abstained tasks are considered. Reverse decoding uses the
same predicted field with p -> 1-p and swaps endpoints; no labels guide it.
The clipped control bounds a proposed displacement before its original segment
checks. Neither control changes mass, support, clearance, or completion gates.
This is an offline experiment, not a packaged/deployed decoder.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "training/source/map_flow_matching_p1"),
               str(ROOT / "training/source/map_flow_matching")]
import p37_model
import adaptive_field_decoder
from p37_model import candidate_selection_scores
from p37_planner import P37FlowPlanner, _p37_map_input
from scripts.geometry_eval import score_task
from p37_adaptive_rejection_audit import digest, write_json


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-run", type=Path, required=True)
    p.add_argument("--current-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--control", choices=("clip", "reverse", "reverse_clip"), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    out = args.output_dir.resolve()
    (out / "paths").mkdir(parents=True, exist_ok=False)
    original_dir = args.original_run / "evaluation/paired_best_val_epoch15"
    current_dir = args.current_run / "evaluation/cached_full286_final"
    original = json.loads((original_dir / "report.json").read_text())
    current = json.loads((current_dir / "report.json").read_text())
    failed = {r["task_id"] for r in current["tasks"] if not r["deployment_success"]}
    tasks = [r for r in original["tasks"] if r["task_id"] in failed]
    checkpoint = args.current_run / "checkpoints/p37_adaptive_fallback.pth"
    manifest = {"status": "running", "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "control": args.control, "checkpoint_sha256": digest(checkpoint),
                "baseline_report_sha256": digest(current_dir / "report.json"),
                "script_sha256": digest(__file__), "network_queries": 0, "training": False,
                "teacher_fields_used": False, "holdout_used": False, "safety_thresholds_unchanged": True}
    source = Path(adaptive_field_decoder.__file__).read_text()
    if "clip" in args.control:
        line = "        movement = torch.linalg.vector_norm(estimate - previous[:, None], dim=-1)"
        assert source.count(line) == 1
        source = source.replace(line, """        displacement = estimate - previous[:, None]
        length = torch.linalg.vector_norm(displacement, dim=-1)
        fraction = ((internal_step_px * (1 - 1e-5)) / length.clamp_min(1e-6)).clamp_max(1)
        estimate = previous[:, None] + displacement * fraction[..., None]
""" + line)
    module_path = out / "adaptive_control.py"
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location("adaptive_control", module_path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    manifest["control_source_sha256"] = digest(module_path)
    write_json(out / "manifest.json", manifest)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = P37FlowPlanner(checkpoint, device=args.device).model
    def forbid_network(*args):
        raise AssertionError("offline control queried network")
    model.vector_field.register_forward_pre_hook(forbid_network)
    with np.load(original["manifest"]["binding"]["data_path"], allow_pickle=True) as data:
        maps, map_ids, starts, goals = (data[key] for key in ("unique_maps", "map_ids", "starts", "goals"))
        geometry = str(data["geometry_json"])
    rows = []
    for task in tasks:
        tid = task["task_id"]
        image = maps[int(map_ids[tid])]
        height, width = image.shape
        scale = np.array([(width - 1) / 2, (height - 1) / 2], np.float32)
        map_input = torch.from_numpy(_p37_map_input(image, model.distance_scale))[None].to(args.device)
        start = torch.from_numpy(starts[tid] / scale - 1)[None].to(args.device)
        goal = torch.from_numpy(goals[tid] / scale - 1)[None].to(args.device)
        with np.load(original_dir / task["field_cache"]) as saved:
            fields_np = saved["fields"]
        fields = torch.from_numpy(fields_np).to(args.device)
        base = model.decode_fields(map_input, start, goal, fields, mode="scale_aware")
        corrected = base["decoder_fields"]
        if "reverse" in args.control:
            reverse_fields = corrected.clone()
            reverse_fields[:, :, 1] = 1 - corrected[:, :, 1]
            with patch.object(p37_model, "adaptive_ordered_field_to_path", control.adaptive_ordered_field_to_path):
                result = model.decode_fields(map_input, goal, start, reverse_fields,
                                             mode="adaptive_fallback", progress_smoothing=0)
            paths = result["paths"].flip(2)
            valid = result["decoder_support_valid"].clone()
            terminal_px = (paths[:, :, -1] - paths[:, :, -2]) * torch.as_tensor(scale, device=paths.device)
            valid &= torch.linalg.vector_norm(terminal_px, dim=-1) <= model.decode_max_terminal_jump * 63 / 2
        else:
            with patch.object(p37_model, "adaptive_ordered_field_to_path", control.adaptive_ordered_field_to_path):
                result = model.decode_fields(map_input, start, goal, fields)
            paths, valid = result["paths"], result["decoder_support_valid"]
        scores = candidate_selection_scores(corrected, paths, map_input, valid)
        accepted = valid & torch.isfinite(scores)
        selected = int(scores.argmax(-1).item()) if accepted.any() else 0
        px = (paths.cpu().numpy()[0] + 1) * scale
        px[:, 0], px[:, -1] = starts[tid], goals[tid]
        safe = np.asarray(score_task((str(ROOT / "training/source/map_flow_matching"), px,
                                     image, starts[tid], goals[tid], geometry)), dtype=bool)
        accepted_np = accepted.cpu().numpy()[0]
        row = {"task_id": tid, "shape": task["shape"], "accepted": accepted_np.tolist(),
               "selected_index": selected, "geometry_safe": safe.tolist(),
               "deployment_success": bool(accepted_np[selected] and safe[selected]),
               "accepted_unsafe_candidates": int((accepted_np & ~safe).sum()),
               "diagnostics": {k: v.cpu().tolist()[0] for k, v in result["diagnostics"].items()}}
        rows.append(row)
        np.savez_compressed(out / "paths" / f"task_{tid:06d}.npz", paths_px=px)
        write_json(out / "partial.json", rows)
        print(json.dumps({"task": tid, "success": row["deployment_success"],
                          "accepted_unsafe": row["accepted_unsafe_candidates"]}), flush=True)
    summary = {"tasks": len(rows), "success": sum(r["deployment_success"] for r in rows),
               "gains": [r["task_id"] for r in rows if r["deployment_success"]],
               "accepted_unsafe_candidates": sum(r["accepted_unsafe_candidates"] for r in rows)}
    manifest.update(status="complete", finished_local=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(out / "manifest.json", manifest)
    write_json(out / "report.json", {"manifest": manifest, "summary": summary, "tasks": rows})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
