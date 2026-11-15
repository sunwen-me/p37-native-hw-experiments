"""Replay cached validation fields through the packaged deployment sampler.

The real sampler and decoder/selector execute; only Flow integration is
replaced with the original cached prediction. Geometry is scored afterward.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
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
from package_p37_decoder import digest
from p37_planner import P37FlowPlanner, _p37_map_input
from scripts.geometry_eval import score_task


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("choose an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "paths").mkdir()
    base_dir = run / "evaluation/paired_best_val_epoch15"
    reference_dir = run / "diagnostics/progress_smooth7_full286_20261115"
    baseline = json.loads((base_dir / "report.json").read_text())
    reference = json.loads((reference_dir / "report.json").read_text())
    reference_by_id = {r["task_id"]: r for r in reference["tasks"]}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    planner = P37FlowPlanner(args.checkpoint, device=args.device)
    model = planner.model
    assert model.decoder_mode == "scale_aware" and model.horizon == 512
    assert model.decode_progress_smoothing == 7
    assert planner.checkpoint_state["decoder_revision"]["source_sha256"] == baseline["manifest"]["binding"]["checkpoint_sha256"]
    for key, expected in baseline["manifest"]["decoder_settings"].items():
        assert getattr(model, key) == expected, key
    manifest = {
        "status": "running", "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
        "baseline_report_sha256": digest(base_dir / "report.json"),
        "prototype_report_sha256": digest(reference_dir / "report.json"),
        "script_sha256": digest(__file__), "training": False, "network_queries": 0,
        "cached_integration_replay": True, "teacher_fields_used": False,
        "holdout_used": False, "independent_holdout": False, "safety_thresholds_unchanged": True,
        "decoder_mode": model.decoder_mode, "horizon": model.horizon,
        "decode_progress_smoothing": model.decode_progress_smoothing,
        "source_sha256": {name: digest(ROOT / "inference" / name) for name in
                          ("p37_model.py", "safe_field_decoder.py", "flow_prior.py", "p37_planner.py")},
    }
    write_json(out / "manifest.json", manifest)
    with np.load(baseline["manifest"]["binding"]["data_path"], allow_pickle=True) as data:
        maps, map_ids, starts, goals = data["unique_maps"], data["map_ids"], data["starts"], data["goals"]
        geometry = str(data["geometry_json"])
    rows, baseline_error, prototype_error = [], 0.0, 0.0

    def fail_network_query(*_):
        raise AssertionError("cached replay must not query the network")

    hook = model.vector_field.register_forward_pre_hook(fail_network_query)
    try:
        for index, task in enumerate(baseline["tasks"], 1):
            tid = task["task_id"]
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
            old = model.decode_fields(map_input, start, goal, fields, progress_smoothing=0)
            old_px = (old["paths"].cpu().numpy()[0] + 1) * scale
            old_px[:, 0], old_px[:, -1] = starts[tid], goals[tid]
            with np.load(base_dir / "paths" / f"task_{tid:06d}.npz") as saved:
                baseline_error = max(baseline_error, float(np.abs(old_px - saved["paths_px_scale_aware_512"]).max()))
            assert baseline_error <= 1e-4
            assert old["accepted"].tolist()[0] == task["arms"]["scale_aware_512"]["accepted"]
            assert int(old["selected_index"].item()) == task["arms"]["scale_aware_512"]["selected_index"]
            latent = torch.zeros(1, 4, model.latent_dim, device=args.device)
            with patch.object(model, "_integrate", return_value=fields.reshape(4, 2, height, width)):
                result = model.sample(map_input, start, goal, candidates=4, latent=latent)
            assert np.array_equal(result["fields"].cpu().numpy(), fields_np)
            px = (result["paths"].cpu().numpy()[0] + 1) * scale
            px[:, 0], px[:, -1] = starts[tid], goals[tid]
            assert px.shape == (4, 512, 2) and np.isfinite(px).all()
            assert torch.equal(result["paths"][:, :, 0], start[:, None].expand(-1, 4, -1))
            assert torch.equal(result["paths"][:, :, -1], goal[:, None].expand(-1, 4, -1))
            with np.load(reference_dir / "paths" / f"task_{tid:06d}.npz") as saved:
                prototype_error = max(prototype_error, float(np.abs(px - saved["paths_px"]).max()))
            assert prototype_error <= 1e-4
            safe = np.asarray(score_task((str(ROOT / "training/source/map_flow_matching"), px,
                                          image, starts[tid], goals[tid], geometry)), dtype=bool)
            accepted = result["accepted"].cpu().numpy()[0]
            selected = int(result["selected_index"].item())
            previous = reference_by_id[tid]
            assert safe.tolist() == previous["geometry_safe"]
            assert accepted.tolist() == previous["accepted"] and selected == previous["selected_index"]
            row = {
                "task_id": tid, "shape": task["shape"], "fields_sha256": task["fields_sha256"],
                "baseline_success": task["arms"]["scale_aware_512"]["deployment_success"],
                "deployment_success": bool(accepted[selected] and safe[selected]),
                "geometry_safe": safe.tolist(), "accepted": accepted.tolist(), "selected_index": selected,
                "any_geometry_success": bool(safe.any()), "any_accepted_safe": bool((safe & accepted).any()),
                "accepted_unsafe_candidates": int((accepted & ~safe).sum()),
                "selected_accepted_unsafe": bool(accepted[selected] and not safe[selected]),
                "diagnostics": {k: v.cpu().tolist()[0] for k, v in result["diagnostics"].items()},
            }
            rows.append(row)
            np.savez_compressed(out / "paths" / f"task_{tid:06d}.npz", paths_px=px)
            if index % 25 == 0:
                print(json.dumps({"completed": index, "total": len(baseline["tasks"]),
                                  "success": sum(r["deployment_success"] for r in rows)}), flush=True)
    finally:
        hook.remove()
    groups = {"overall": rows, "large": [r for r in rows if r["shape"] in ("512x768", "768x512")]}
    groups.update({shape: [r for r in rows if r["shape"] == shape] for shape in sorted({r["shape"] for r in rows})})
    summaries = {name: {"tasks": len(rs), "success": sum(r["deployment_success"] for r in rs),
                        "gains": [r["task_id"] for r in rs if r["deployment_success"] and not r["baseline_success"]],
                        "losses": [r["task_id"] for r in rs if r["baseline_success"] and not r["deployment_success"]],
                        "accepted_unsafe_candidates": sum(r["accepted_unsafe_candidates"] for r in rs)}
                 for name, rs in groups.items()}
    assert summaries["overall"]["success"] == 270
    assert not summaries["overall"]["losses"] and not summaries["overall"]["accepted_unsafe_candidates"]
    manifest.update(status="complete", finished_local=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    baseline_max_path_error_px=baseline_error, prototype_max_path_error_px=prototype_error)
    write_json(out / "manifest.json", manifest)
    write_json(out / "report.json", {"manifest": manifest, "summaries": summaries, "tasks": rows})
    print(json.dumps({"overall": summaries["overall"], "large": summaries["large"]}), flush=True)


if __name__ == "__main__":
    main()
