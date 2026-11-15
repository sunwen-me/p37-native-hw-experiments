"""Observe exact adaptive-decoder rejection gates without changing its code.

Python line tracing reads the production loop after its eligibility checks.
All seven proposals and overlapping rejection reasons are retained. Labels
are not consulted; pixel components are explanatory, not safety certificates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
from scipy.ndimage import label
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "training/source/map_flow_matching_p1"),
               str(ROOT / "training/source/map_flow_matching")]
import adaptive_field_decoder as decoder
import p37_model
from p37_planner import P37FlowPlanner, _p37_map_input


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def segment_minima(occupancy, clearance, begin, end):
    """Same 17 positions, interpolation, and float dtype as production."""
    height, width = occupancy.shape[-2:]
    t = torch.linspace(0, 1, 17, device=begin.device, dtype=begin.dtype)
    xy = begin[:, None, None] + (end - begin[:, None])[:, :, None] * t[None, None, :, None]
    scale = xy.new_tensor([(width - 1) / 2, (height - 1) / 2])
    grid = xy / scale - 1
    route = F.grid_sample(occupancy, grid, align_corners=True)[:, 0].amin(-1)
    safe = F.grid_sample(clearance, grid, align_corners=True)[:, 0].amin(-1)
    return route, safe


def describe_rejection(state, index, route_min, clearance_min):
    def cpu(name):
        return state[name][index].detach().cpu().numpy()
    threshold = state["occupancy_threshold"]
    mass, movement = cpu("mass"), cpu("movement")
    route_min = route_min[index].cpu().numpy()
    clearance_min = clearance_min[index].cpu().numpy()
    eligible = cpu("eligible")
    checks = np.stack((mass > 1e-5, movement <= state["internal_step_px"],
                       route_min > threshold, clearance_min >= .5), -1)
    assert np.array_equal(eligible, checks.all(-1))
    names = ("mass", "movement", "segment_occupancy", "segment_clearance")
    previous = cpu("previous")
    occupancy = state["occupancy"][index, 0].detach().cpu().numpy()
    clearance = state["clearance"][index, 0].detach().cpu().numpy()
    support = (occupancy > threshold) & (clearance >= .5)
    components, count = label(support, np.ones((3, 3)))
    height, width = occupancy.shape
    x, y = np.rint(previous).astype(int)
    gx, gy = np.rint(cpu("goal_px")).astype(int)
    component = int(components[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)])
    goal_component = int(components[np.clip(gy, 0, height - 1), np.clip(gx, 0, width - 1)])
    local_p, local_base = cpu("local_p"), cpu("local_base")
    supported = local_base > 1e-8
    proposals = []
    for backoff in range(7):
        proposals.append({
            "backoff": backoff, "target_phase": float(cpu("targets")[backoff]),
            "joint_mass": float(mass[backoff]), "movement_px": float(movement[backoff]),
            "estimate_px": cpu("estimate")[backoff].tolist(),
            "segment_occupancy_min": float(route_min[backoff]),
            "segment_clearance_min": float(clearance_min[backoff]),
            "failed_checks": [name for name, passed in zip(names, checks[backoff]) if not passed],
        })
    return {
        "candidate": index, "attempt": int(cpu("iterations")) + 1,
        "previous_px": previous.tolist(), "phase": float(cpu("phase")),
        "remaining_distance_px": float(torch.linalg.vector_norm(
            state["previous"][index] - state["goal_px"][index])),
        "local_mass_without_progress": float(local_base.sum()),
        "local_supported_progress_quantiles": np.quantile(local_p[supported], [0, .05, .5, .95, 1]).tolist()
            if supported.any() else None,
        "pixel_support_components": int(count), "stop_component": component,
        "goal_component": goal_component,
        "stop_goal_same_pixel_support_component": bool(component and component == goal_component),
        "stop_pixel_occupancy": float(occupancy[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]),
        "proposals": proposals,
    }


class RejectionTrace:
    """Trace only the exact production function, restoring any caller tracer."""
    def __init__(self):
        self.events = []
        function = decoder.adaptive_ordered_field_to_path.__wrapped__
        self.code = function.__code__
        source, first = inspect.getsourcelines(function)
        lines = [first + i for i, line in enumerate(source) if "available = eligible.any(-1) & active" in line]
        if len(lines) != 1:
            raise AssertionError("production observation point changed")
        self.line = lines[0]

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        if event == "line" and frame.f_lineno == self.line:
            state = frame.f_locals
            rejected = state["active"] & ~state["eligible"].any(-1)
            if rejected.any():
                route_min, safe_min = segment_minima(state["occupancy"], state["clearance"],
                                                     state["previous"], state["estimate"])
                for index in rejected.nonzero().flatten().tolist():
                    self.events.append(describe_rejection(state, index, route_min, safe_min))
        return self.trace

    def run(self, function, *args, **kwargs):
        previous = sys.gettrace()
        try:
            sys.settrace(self.trace)
            return function(*args, **kwargs)
        finally:
            sys.settrace(previous)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original_dir = args.original_run / "evaluation/paired_best_val_epoch15"
    current_dir = args.current_run / "evaluation/cached_full286_final"
    original = json.loads((original_dir / "report.json").read_text())
    current = json.loads((current_dir / "report.json").read_text())
    previous_by_id = {r["task_id"]: r for r in current["tasks"]}
    failed = {r["task_id"] for r in current["tasks"] if not r["deployment_success"]}
    controls = {82444, 69406, 66847, 40128}
    tasks = [r for r in original["tasks"] if r["task_id"] in failed | controls]
    checkpoint = args.current_run / "checkpoints/p37_adaptive_fallback.pth"
    bound = [checkpoint, current_dir / "report.json", original_dir / "report.json",
             ROOT / "inference/adaptive_field_decoder.py", ROOT / "inference/p37_model.py",
             ROOT / "inference/safe_field_decoder.py"]
    hashes = {str(p): digest(p) for p in bound}
    manifest = {"status": "running", "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "binding_sha256": hashes, "script_sha256": digest(__file__), "network_queries": 0,
                "teacher_fields_used": False, "holdout_used": False, "training": False,
                "failed_tasks": sorted(failed), "success_controls": sorted(controls)}
    write_json(out / "manifest.json", manifest)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = P37FlowPlanner(checkpoint, device=args.device).model
    def forbid_network(*args):
        raise AssertionError("offline audit queried network")
    model.vector_field.register_forward_pre_hook(forbid_network)
    with np.load(original["manifest"]["binding"]["data_path"], allow_pickle=True) as data:
        maps, map_ids, starts, goals = (data[key] for key in ("unique_maps", "map_ids", "starts", "goals"))
    rows = []
    for task in tasks:
        tid = task["task_id"]
        image = maps[int(map_ids[tid])]
        height, width = image.shape
        scale = np.array([(width - 1) / 2, (height - 1) / 2], np.float32)
        maps_t = torch.from_numpy(_p37_map_input(image, model.distance_scale))[None].to(args.device)
        start = torch.from_numpy(starts[tid] / scale - 1)[None].to(args.device)
        goal = torch.from_numpy(goals[tid] / scale - 1)[None].to(args.device)
        with np.load(original_dir / task["field_cache"]) as saved:
            fields_np = saved["fields"]
        assert hashlib.sha256(np.ascontiguousarray(fields_np).tobytes()).hexdigest() == task["fields_sha256"]
        fields = torch.from_numpy(fields_np).to(args.device)
        trace = RejectionTrace()
        production = p37_model.adaptive_ordered_field_to_path
        def observed(*a, **kw):
            return trace.run(production, *a, **kw)
        with patch.object(p37_model, "adaptive_ordered_field_to_path", side_effect=observed):
            result = model.decode_fields(maps_t, start, goal, fields)
        previous = previous_by_id[tid]
        assert result["accepted"].tolist()[0] == previous["accepted"]
        assert int(result["selected_index"].item()) == previous["selected_index"]
        for name, expected in previous["diagnostics"].items():
            assert result["diagnostics"][name].cpu().tolist()[0] == expected, name
        px = (result["paths"].cpu().numpy()[0] + 1) * scale
        px[:, 0], px[:, -1] = starts[tid], goals[tid]
        with np.load(current_dir / "paths" / f"task_{tid:06d}.npz") as saved:
            assert np.array_equal(px, saved["paths_px"]), tid
        assert np.array_equal(fields.cpu().numpy(), fields_np)
        expected_failures = sum(code == 1 for code in previous["diagnostics"]["adaptive_failure_code"])
        assert len(trace.events) == expected_failures
        row = {"task_id": tid, "shape": task["shape"], "baseline_success": previous["deployment_success"],
               "paths_bitwise_equal": True, "diagnostics_equal": True, "events": trace.events}
        rows.append(row)
        write_json(out / f"task_{tid:06d}.json", row)
        print(json.dumps({"task": tid, "events": len(trace.events), "baseline_success": previous["deployment_success"]}), flush=True)
    events = [event for row in rows if not row["baseline_success"] for event in row["events"]]
    proposals = [proposal for event in events for proposal in event["proposals"]]
    counts = Counter(name for proposal in proposals for name in proposal["failed_checks"])
    combinations = Counter("+".join(proposal["failed_checks"]) for proposal in proposals)
    summary = {"tasks": len(tasks), "remaining_failed_tasks": len(failed), "rejection_events": len(events),
               "backoff_proposals": len(proposals), "overlapping_failed_check_counts": dict(counts),
               "failed_check_combinations": dict(combinations),
               "events_all7_fail_each_gate": {name: sum(all(name in p["failed_checks"] for p in e["proposals"]) for e in events)
                   for name in ("mass", "movement", "segment_occupancy", "segment_clearance")},
               "stop_goal_same_pixel_support_component": sum(e["stop_goal_same_pixel_support_component"] for e in events),
               "paths_and_diagnostics_bitwise_reproduced": len(rows)}
    assert all(digest(Path(p)) == value for p, value in hashes.items())
    manifest.update(status="complete", finished_local=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(out / "manifest.json", manifest)
    write_json(out / "report.json", {"manifest": manifest, "summary": summary, "tasks": rows,
        "limitations": ["Pixel support components do not certify continuous swept-disk safety.",
                        "This reused development set cannot establish blind generalization.",
                        "Rejected forced-goal segments are excluded from the local continuation diagnosis."]})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
