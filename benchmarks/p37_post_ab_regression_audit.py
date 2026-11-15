"""Replay the epoch15/A/B regression and B's bounded decoder on frozen fields.

Network calls are forbidden. Cross-checkpoint channel substitutions are offline
mechanism controls, never model-quality estimates. Labels only score or describe
paths after decoding; no acceptance threshold or production code is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "benchmarks"),
               str(ROOT / "training/source/map_flow_matching_p1"),
               str(ROOT / "training/source/map_flow_matching")]
import p37_model
from p37_planner import P37FlowPlanner, _p37_map_input
from p37_adaptive_rejection_audit import RejectionTrace
from p37_large_map_failure_audit import route_metrics
from scripts.geometry_eval import score_task


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class FirstSupportLossTrace:
    """Read the first failed scale-aware support slice without replacing math."""
    def __init__(self):
        function = p37_model.safe_ordered_field_to_path
        function = getattr(function, "__wrapped__", function)
        self.code = function.__code__
        lines, first = inspect.getsourcelines(function)
        matching = [first + i for i, line in enumerate(lines)
                    if "estimate = (local_weights[:, :, None] * points)" in line]
        if len(matching) != 1:
            raise AssertionError("production scale-aware observation point changed")
        self.line = matching[0]
        self.events = []
        self.seen = set()

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        if event == "line" and frame.f_lineno == self.line:
            state = frame.f_locals
            step = len(state["path"])
            if not 0 < step < state["horizon"] - 1:
                return self.trace
            for index in (~state["has_local"]).nonzero().flatten().tolist():
                if index in self.seen:
                    continue
                self.seen.add(index)
                scale = state["previous"].new_tensor([(state["width"] - 1) / 2,
                                                     (state["height"] - 1) / 2])
                previous = state["previous"][index]
                distance2 = ((state["points"][0] - previous) * scale).square().sum(-1)
                base = state["flat_probability"][index]
                local_support = (base > 1e-8) & (distance2 <= 100)
                p = state["flat_progress"][index][local_support]
                strong = state["score"][index] > 1e-5
                self.events.append({
                    "candidate": index, "step": step,
                    "requested_progress": float(state["center_value"]),
                    "previous_px": ((previous + 1) * scale).cpu().tolist(),
                    "local_joint_mass": float(state["local_mass"][index, 0]),
                    "local_mass_without_progress": float((base * state["local"][index]).sum()),
                    "global_progress_mass": float(state["score"][index].sum()),
                    "nearest_strong_matching_progress_pixel_px":
                        float(distance2[strong].min().sqrt()) if strong.any() else None,
                    "nearby_supported_progress_range_10px":
                        [float(p.min()), float(p.max())] if p.numel() else None,
                })
        return self.trace


def traced_decode(model, maps, start, goal, fields, mode, *, smoothing=7):
    initial = FirstSupportLossTrace()
    adaptive = RejectionTrace()
    production = p37_model.adaptive_ordered_field_to_path
    stages = []

    def observed(*args, **kwargs):
        first = len(adaptive.events)
        result = production(*args, **kwargs)
        # The enclosing tracer observes both production decoder functions.
        stages.append({"clipped": bool(kwargs["clip_internal_move"]),
                       "events_begin": first, "events_end": len(adaptive.events),
                       "diagnostics": {k: v.cpu().tolist() for k, v in result[2].items()}})
        return result

    def combined(frame, event, arg):
        if frame.f_code is initial.code:
            initial.trace(frame, event, arg)
            return combined
        if frame.f_code is adaptive.code:
            adaptive.trace(frame, event, arg)
            return combined
        return None

    previous = sys.gettrace()
    try:
        with patch.object(p37_model, "adaptive_ordered_field_to_path", side_effect=observed):
            sys.settrace(combined)
            result = model.decode_fields(maps, start, goal, fields, mode=mode, progress_smoothing=smoothing)
    finally:
        sys.settrace(previous)
    return result, {"scale_aware_first_loss": initial.events,
                    "adaptive_rejections": adaptive.events, "adaptive_stages": stages}


def pixels(result, scale, start, goal):
    value = (result["paths"][0].cpu().numpy() + 1) * scale
    assert np.isfinite(value).all() and value.shape == (4, 512, 2)
    value[:, 0], value[:, -1] = start, goal
    return value


def outcome(result, paths, image, start, goal, geometry):
    safe = np.asarray(score_task((str(ROOT / "training/source/map_flow_matching"),
                                 paths, image, start, goal, geometry)), dtype=bool)
    accepted = result["accepted"][0].cpu().numpy()
    index = int(result["selected_index"].item())
    return {"deployment_success": bool(accepted[index] and safe[index]),
            "geometry_safe": safe.tolist(), "accepted": accepted.tolist(),
            "selected_index": index, "accepted_unsafe_candidates": int((accepted & ~safe).sum()),
            "diagnostics": {k: v.cpu().tolist()[0] for k, v in result["diagnostics"].items()}}


def check_replay(result, px, directory, previous):
    assert result["accepted"][0].cpu().tolist() == previous["accepted"]
    assert int(result["selected_index"].item()) == previous["selected_index"]
    for key, value in previous["diagnostics"].items():
        assert result["diagnostics"][key][0].cpu().tolist() == value, key
    with np.load(directory / "paths" / f"task_{previous['task_id']:06d}.npz") as saved:
        assert np.array_equal(px, saved["paths_px"]), previous["task_id"]


def summarize(rows, baseline):
    groups = {"overall": rows, "large": [r for r in rows if r["shape"] in ("512x768", "768x512")]}
    groups.update({shape: [r for r in rows if r["shape"] == shape]
                   for shape in sorted({r["shape"] for r in rows})})
    return {name: {"tasks": len(group), "success": sum(r["deployment_success"] for r in group),
                   "gains": [r["task_id"] for r in group if r["deployment_success"] and not baseline[r["task_id"]]["deployment_success"]],
                   "losses": [r["task_id"] for r in group if not r["deployment_success"] and baseline[r["task_id"]]["deployment_success"]],
                   "accepted_unsafe_candidates": sum(r["accepted_unsafe_candidates"] for r in group)}
            for name, group in groups.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=17831)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("source", "inputs", "case_paths", "bounded_paths"):
        (out / name).mkdir()
    source_paths = [Path(__file__), ROOT / "benchmarks/p37_adaptive_rejection_audit.py",
                    ROOT / "benchmarks/p37_large_map_failure_audit.py",
                    *(ROOT / "inference" / name for name in ("p37_model.py", "p37_planner.py",
                        "adaptive_field_decoder.py", "safe_field_decoder.py", "flow_prior.py")),
                    ROOT / "training/source/map_flow_matching_p1/scripts/geometry_eval.py",
                    ROOT / "training/source/map_flow_matching_p1/scripts/generate_native_hw_dataset.py",
                    ROOT / "training/source/map_flow_matching/utils/robot_geometry.py"]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(relative)] = digest(path)
    original_dir = args.original_run / "evaluation/paired_best_val_epoch15"
    original = read(original_dir / "report.json")
    binding = original["manifest"]["binding"]
    directories = {"epoch15": args.baseline_run / "evaluation/cached_full286_final",
                   "A_cfm": args.ab_run / "A_cfm/evaluation",
                   "B_route_increment": args.ab_run / "B_route_increment/evaluation"}
    reports = {key: read(path / "report.json") for key, path in directories.items()}
    by_id = {key: {row["task_id"]: row for row in value["tasks"]} for key, value in reports.items()}
    original_by_id = {row["task_id"]: row for row in original["tasks"]}
    checkpoint = args.ab_run / "B_route_increment/checkpoints/best_p37.pth"
    checkpoint_hash = digest(checkpoint)
    assert checkpoint_hash == reports["B_route_increment"]["manifest"]["binding"]["checkpoint_sha256"]
    for arm in ("A_cfm", "B_route_increment"):
        assert digest(args.ab_run / arm / "checkpoints/best_p37.pth") == reports[arm]["manifest"]["binding"]["checkpoint_sha256"]
    assert digest(args.baseline_run / "checkpoints/p37_adaptive_fallback.pth") == reports["epoch15"]["manifest"]["checkpoint_sha256"]
    for name, directory in directories.items():
        shutil.copy2(directory / "report.json", out / "inputs" / f"{name}_report.json")
    shutil.copy2(original_dir / "report.json", out / "inputs/original_report.json")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model = P37FlowPlanner(checkpoint, device=args.device).model
    for key, value in original["manifest"]["decoder_settings"].items():
        assert getattr(model, key) == value, key
    def forbid_network(*_):
        raise AssertionError("this audit must not query the network")
    model.vector_field.register_forward_pre_hook(forbid_network)
    with np.load(binding["data_path"], allow_pickle=True) as data:
        maps, map_ids, starts, goals, routes = (data[key] for key in ("unique_maps", "map_ids", "starts", "goals", "paths"))
        geometry = str(data["geometry_json"])
    manifest = {"status": "running", "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "source_sha256": source_hashes, "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash, "task_id": args.task_id,
                "input_report_sha256": {key: digest(path / "report.json") for key, path in directories.items()},
                "data_path": binding["data_path"], "network_queries": 0,
                "safety_thresholds_changed": False, "training": False, "holdout_used": False}
    write_json(out / "manifest.json", manifest)

    def inputs(tid):
        image = maps[int(map_ids[tid])]
        height, width = image.shape
        scale = np.asarray([(width - 1) / 2, (height - 1) / 2], dtype=np.float32)
        features = torch.from_numpy(_p37_map_input(image, model.distance_scale))[None].to(args.device)
        start = torch.from_numpy(starts[tid] / scale - 1)[None].to(args.device)
        goal = torch.from_numpy(goals[tid] / scale - 1)[None].to(args.device)
        return image, scale, features, start, goal

    def fields_for(name, tid):
        directory = original_dir if name == "epoch15" else directories[name]
        row = original_by_id[tid] if name == "epoch15" else by_id[name][tid]
        path = directory / (row["field_cache"] if name == "epoch15" else f"fields/task_{tid:06d}.npz")
        with np.load(path) as saved:
            fields, latent = saved["fields"], saved["latent"]
            assert int(saved["task_id"]) == tid
            field_hash = hashlib.sha256(np.ascontiguousarray(fields).tobytes()).hexdigest()
            assert field_hash == str(saved["fields_sha256"]) == row["fields_sha256"]
            if name != "epoch15":
                assert str(saved["checkpoint_sha256"]) == reports[name]["manifest"]["binding"]["checkpoint_sha256"]
        assert np.isfinite(fields).all() and fields.shape[0:3] == (1, 4, 2)
        return torch.from_numpy(fields).to(args.device), latent, path

    tid = args.task_id
    image, scale, features, start, goal = inputs(tid)
    case_rows, corrected, latents = {}, {}, {}
    for name in directories:
        fields, latent, path = fields_for(name, tid)
        latents[name] = latent
        shutil.copy2(path, out / "inputs" / f"{name}_task_{tid:06d}.npz")
        case_rows[name] = {}
        for mode in ("adaptive_fallback", "bounded_fallback"):
            result, trace = traced_decode(model, features, start, goal, fields, mode)
            px = pixels(result, scale, starts[tid], goals[tid])
            row = outcome(result, px, image, starts[tid], goals[tid], geometry)
            row["trace"] = trace
            if mode == "adaptive_fallback":
                check_replay(result, px, directories[name], by_id[name][tid])
                assert row["geometry_safe"] == by_id[name][tid]["geometry_safe"]
                corrected[name] = result["decoder_fields"].clone()
                row["reference_route_diagnostics"] = route_metrics(routes[tid], corrected[name][0].cpu().numpy())
            case_rows[name][mode] = row
            np.savez_compressed(out / "case_paths" / f"{name}_{mode}.npz", paths_px=px)
        print(json.dumps({"stage": "case", "arm": name,
                          "adaptive_success": case_rows[name]["adaptive_fallback"]["deployment_success"],
                          "bounded_success": case_rows[name]["bounded_fallback"]["deployment_success"]}), flush=True)
    assert all(np.array_equal(latents["epoch15"], value) for value in latents.values())

    controls = []
    for name in ("A_cfm", "B_route_increment"):
        for support_name, progress_name in (("epoch15", name), (name, "epoch15")):
            hybrid = torch.cat([corrected[support_name][:, :, :1], corrected[progress_name][:, :, 1:2]], 2)
            for mode in ("adaptive_fallback", "bounded_fallback"):
                result, trace = traced_decode(model, features, start, goal, hybrid, mode, smoothing=0)
                px = pixels(result, scale, starts[tid], goals[tid])
                row = outcome(result, px, image, starts[tid], goals[tid], geometry)
                row.update(support_from=support_name, corrected_progress_from=progress_name,
                           decoder=mode, trace=trace, diagnostic_only=True)
                controls.append(row)
                np.savez_compressed(out / "case_paths" / f"support_{support_name}_progress_{progress_name}_{mode}.npz", paths_px=px)
    case_report = {"task_id": tid, "shape": list(image.shape), "latents_identical": True,
                   "original_replays_bitwise_verified": True, "arms": case_rows,
                   "corrected_channel_controls": controls,
                   "limitations": ["Channel substitutions are diagnostic, not predictions of a deployable model.",
                                   "Predicted support/progress may encode different route modes.",
                                   "Pixel support components are not continuous geometry safety certificates."]}
    write_json(out / "case_report.json", case_report)

    rows = []
    for number, previous in enumerate(reports["B_route_increment"]["tasks"], 1):
        begin = time.monotonic()
        tid = previous["task_id"]
        image, scale, features, start, goal = inputs(tid)
        fields, _, _ = fields_for("B_route_increment", tid)
        old = model.decode_fields(features, start, goal, fields, mode="adaptive_fallback", horizon=512, progress_smoothing=7)
        old_px = pixels(old, scale, starts[tid], goals[tid])
        check_replay(old, old_px, directories["B_route_increment"], previous)
        result = model.decode_fields(features, start, goal, fields, mode="bounded_fallback", horizon=512, progress_smoothing=7)
        px = pixels(result, scale, starts[tid], goals[tid])
        row = outcome(result, px, image, starts[tid], goals[tid], geometry)
        row.update(task_id=tid, shape=previous["shape"], fields_sha256=previous["fields_sha256"],
                   baseline_success=previous["deployment_success"], seconds=time.monotonic() - begin)
        assert bool(result["diagnostics"]["bounded_attempted"].any()) == (not any(previous["accepted"]))
        if any(previous["accepted"]):
            for key in ("paths", "accepted", "selection_scores", "selected_index"):
                assert torch.equal(result[key], old[key]), (tid, key)
            for key, value in old["diagnostics"].items():
                assert torch.equal(result["diagnostics"][key], value), (tid, key)
            assert np.array_equal(px, old_px)
        rows.append(row)
        np.savez_compressed(out / "bounded_paths" / f"task_{tid:06d}.npz", paths_px=px)
        write_json(out / "bounded_partial.json", rows)
        if number % 22 == 0 or not previous["deployment_success"] or number == 286:
            print(json.dumps({"stage": "B_bounded", "completed": number, "task": tid,
                              "success": row["deployment_success"], "new_gain": row["deployment_success"] and not row["baseline_success"],
                              "accepted_unsafe": row["accepted_unsafe_candidates"]}), flush=True)
    for name, expected in source_hashes.items():
        assert digest(ROOT / name) == expected
        assert digest(out / "source" / name) == expected
    assert digest(checkpoint) == checkpoint_hash
    manifest.update(status="complete", finished_local=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    completed_tasks=len(rows), B_adaptive_bitwise_replays=len(rows),
                    old_B_successes_preserved_bitwise=sum(r["baseline_success"] for r in rows))
    report = {"manifest": manifest, "summaries": summarize(rows, by_id["B_route_increment"]),
              "vs_epoch15_adaptive": summarize(rows, by_id["epoch15"]), "tasks": rows,
              "remaining_failures": [r["task_id"] for r in rows if not r["deployment_success"]],
              "limitations": ["Reused development cases, not an independent holdout.",
                              "Frozen fields were generated by B's actual new weights; zero new network calls.",
                              "No candidate is accepted using independent geometry labels."]}
    write_json(out / "bounded_report.json", report)
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "summary": report["summaries"],
                      "remaining_failures": report["remaining_failures"]}), flush=True)


if __name__ == "__main__":
    main()
