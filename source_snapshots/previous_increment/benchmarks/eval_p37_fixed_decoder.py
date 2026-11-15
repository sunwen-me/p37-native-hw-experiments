"""Resample actual model fields under the frozen adaptive/smooth7/512 protocol."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training/source/map_flow_matching_p1"),
               str(ROOT / "training/source/map_flow_matching"),
               str(ROOT / "training/source/map_flow_matching_p1/scripts")]
from flow_prior import load_flow_prior
from utils.dataset import PathPlanningDataset
from train_p37_ordered_field import load_map_disjoint_split, file_sha256
from scripts.geometry_eval import score_task
from p37_paired_decoder_eval import candidate_outcomes
from p37_large_map_failure_audit import route_metrics


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(rows):
    groups = {"overall": rows, "large": [r for r in rows if r["shape"] in ("512x768", "768x512")]}
    groups.update({shape: [r for r in rows if r["shape"] == shape] for shape in sorted({r["shape"] for r in rows})})
    return {name: {"tasks": len(group), "success": sum(r["deployment_success"] for r in group),
                   "gains": [r["task_id"] for r in group if r["deployment_success"] and not r["baseline_success"]],
                   "losses": [r["task_id"] for r in group if not r["deployment_success"] and r["baseline_success"]],
                   "accepted_unsafe_candidates": sum(r["accepted_unsafe_candidates"] for r in group),
                   "selected_accepted_unsafe": sum(r["selected_accepted_unsafe"] for r in group)}
            for name, group in groups.items()}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-report", type=Path, required=True)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument("--split-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-tasks", type=int, default=0, help="smoke only; zero evaluates all 286")
    p.add_argument("--expect-source-fields", action="store_true", help="verify initial-model sampling protocol")
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and not args.resume:
        raise FileExistsError("choose a fresh evaluation directory or explicitly resume")
    for name in ("fields", "paths"):
        (out / name).mkdir(exist_ok=True)
    source = json.loads(args.source_report.read_text())
    old = json.loads(args.baseline_report.read_text())
    baseline = {r["task_id"]: r for r in old["tasks"]}
    settings = source["manifest"]["decoder_settings"]
    binding = source["manifest"]["binding"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model, state = load_flow_prior(args.checkpoint, args.device)
    model.eval().requires_grad_(False)
    model.vector_field.latent_scale = float(binding["latent_scale"])
    model.decoder_mode, model.horizon, model.decode_progress_smoothing = "adaptive_fallback", 512, 7
    for key, expected in settings.items():
        if key != "decode_progress_smoothing":
            assert getattr(model, key) == expected, key
    tasks = source["tasks"][:args.max_tasks or None]
    base = PathPlanningDataset(binding["data_path"], horizon=512, normalize=True, augment=False,
                               distance_scale=model.distance_scale, validate_geometry=False, cache_distance_fields=True)
    _, val_ids, _ = load_map_disjoint_split(base.maps, args.split_path)
    assert all(r["task_id"] in val_ids for r in tasks)
    assert file_sha256(args.split_path) == binding["split_sha256"]
    cache_binding = {"checkpoint_sha256": file_sha256(args.checkpoint),
                     "source_report_sha256": file_sha256(args.source_report),
                     "baseline_report_sha256": file_sha256(args.baseline_report),
                     "seed": binding["seed"], "decoder": "adaptive_fallback/smooth7/512", "K": 4, "steps": 100}
    if args.resume:
        assert json.loads((out / "manifest.json").read_text())["binding"] == cache_binding
    manifest = {"status": "running", "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "binding": cache_binding, "checkpoint": str(args.checkpoint.resolve()),
                "script_sha256": file_sha256(__file__), "new_model_predictions": True,
                "holdout_used": False, "teacher_fields_used_for_decoding": False,
                "reused_development_set": True, "expected_tasks": len(tasks),
                "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in (
                    ROOT / "training/source/map_flow_matching_p1/p37_model.py",
                    ROOT / "training/source/map_flow_matching_p1/safe_field_decoder.py",
                    ROOT / "training/source/map_flow_matching_p1/adaptive_field_decoder.py")}}
    write_json(out / "manifest.json", manifest)
    calls = [0]
    def count_network(*args):
        calls[0] += 1
    model.vector_field.register_forward_pre_hook(count_network)
    generator = torch.Generator(device=args.device).manual_seed(int(binding["seed"]))
    rows = []
    for number, task in enumerate(tasks, 1):
        started = time.monotonic()
        tid = task["task_id"]
        item = base[tid]
        maps = torch.cat([item["map"], item["distance_field"]], 0)[None].to(args.device)
        start, goal = item["start"][None].to(args.device), item["goal"][None].to(args.device)
        field_path = out / "fields" / f"task_{tid:06d}.npz"
        cached = field_path.exists()
        if cached:
            assert args.resume
            with np.load(field_path) as saved:
                fields_np = saved["fields"]
                assert str(saved["checkpoint_sha256"]) == cache_binding["checkpoint_sha256"]
                assert int(saved["task_id"]) == tid
                assert str(saved["fields_sha256"]) == hashlib.sha256(np.ascontiguousarray(fields_np).tobytes()).hexdigest()
                generator.set_state(torch.from_numpy(saved["rng_state_after"]))
            result = model.decode_fields(maps, start, goal, torch.from_numpy(fields_np).to(args.device))
        else:
            before = calls[0]
            result = model.sample(maps, start, goal, candidates=4, num_steps=100, noise_scale=1., generator=generator)
            assert calls[0] - before == 100
            fields_np = result["fields"].cpu().numpy()
            np.savez_compressed(field_path, fields=fields_np, latent=result["latent"].cpu().numpy(),
                                rng_state_after=generator.get_state().cpu().numpy(),
                                task_id=np.asarray(tid),
                                fields_sha256=np.asarray(hashlib.sha256(np.ascontiguousarray(fields_np).tobytes()).hexdigest()),
                                checkpoint_sha256=np.asarray(cache_binding["checkpoint_sha256"]))
        field_sha = hashlib.sha256(np.ascontiguousarray(fields_np).tobytes()).hexdigest()
        if args.expect_source_fields:
            assert field_sha == task["fields_sha256"], (tid, field_sha, task["fields_sha256"])
        px = base.denormalize_paths(result["paths"].cpu().numpy(), [tid])[0]
        safe = np.asarray(score_task((str(ROOT / "training/source/map_flow_matching"), px, base.maps[tid],
                                     base.starts[tid], base.goals[tid], base.geometry.to_json())), dtype=bool)
        valid, accepted = (result[key][0].cpu().numpy() for key in ("decoder_support_valid", "accepted"))
        selected = int(result["selected_index"].item())
        row = {"task_id": tid, "shape": task["shape"], "baseline_success": baseline[tid]["deployment_success"],
               **candidate_outcomes(safe, valid, accepted, selected), "geometry_safe": safe.tolist(),
               "accepted": accepted.tolist(), "selected_index": selected, "fields_sha256": field_sha,
               "loaded_new_model_cache": cached, "diagnostics": {k: v.cpu().tolist()[0] for k, v in result["diagnostics"].items()},
               "reference_route_diagnostics": route_metrics(base.paths[tid], result["decoder_fields"][0].cpu().numpy()),
               "seconds": time.monotonic() - started}
        assert px.shape == (4, 512, 2) and np.isfinite(px).all() and np.isfinite(fields_np).all()
        np.savez_compressed(out / "paths" / f"task_{tid:06d}.npz", paths_px=px)
        rows.append(row)
        manifest.update(completed_tasks=number, network_calls_this_invocation=calls[0])
        write_json(out / "manifest.json", manifest)
        write_json(out / "partial.json", rows)
        print(json.dumps({"completed": number, "total": len(tasks), "task": tid,
                          "success": row["deployment_success"], "unsafe": row["accepted_unsafe_candidates"]}), flush=True)
    assert file_sha256(args.checkpoint) == cache_binding["checkpoint_sha256"]
    manifest.update(status="complete", finished_local=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    summary = summarize(rows)
    write_json(out / "manifest.json", manifest)
    write_json(out / "report.json", {"manifest": manifest, "summaries": summary, "tasks": rows,
        "limitations": ["Development set reused for design; this is not blind generalization.",
                        "Reference-route diagnostics may disagree with a valid alternate predicted mode.",
                        "A single epoch and seed only screen the new objective."]})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
