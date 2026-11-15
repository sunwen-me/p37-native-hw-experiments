"""Exercise the repaired pixel-coordinate planner with real Flow integration."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "training/source/map_flow_matching_p1"),
               str(ROOT / "training/source/map_flow_matching")]
from package_p37_decoder import digest
from p37_planner import P37FlowPlanner
from scripts.geometry_eval import score_task


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = json.loads((args.run_dir / "evaluation/paired_best_val_epoch15/report.json").read_text())
    tasks = [next(t for t in baseline["tasks"] if t["shape"] == shape)
             for shape in ("127x193", "512x768", "768x512")]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    planner = P37FlowPlanner(args.checkpoint, args.device)
    with np.load(baseline["manifest"]["binding"]["data_path"], allow_pickle=True) as data:
        maps, map_ids, starts, goals = data["unique_maps"], data["map_ids"], data["starts"], data["goals"]
        geometry = str(data["geometry_json"])
    calls = []
    hook = planner.model.vector_field.register_forward_pre_hook(lambda *_: calls.append(1))
    rows = []
    try:
        for task in tasks:
            tid = task["task_id"]
            image = maps[int(map_ids[tid])]
            before = len(calls)
            began = time.perf_counter()
            result = planner.plan(image, starts[tid], goals[tid], candidates=4,
                                  seed=20461115 + tid, flow_steps=100)
            px = result["paths"].cpu().numpy()[0]
            assert px.shape == (4, 512, 2) and np.isfinite(px).all()
            assert np.array_equal(px[:, 0], np.broadcast_to(starts[tid], (4, 2)))
            assert np.array_equal(px[:, -1], np.broadcast_to(goals[tid], (4, 2)))
            assert result["decoder_config"] == {"mode": "scale_aware", "horizon": 512, "progress_smoothing": 7}
            assert len(calls) - before == 100
            safe = np.asarray(score_task((str(ROOT / "training/source/map_flow_matching"), px,
                                          image, starts[tid], goals[tid], geometry)), dtype=bool)
            accepted = result["accepted"].cpu().numpy()[0]
            selected = int(result["selected_index"].item())
            assert not (accepted & ~safe).any()
            row = {"task_id": tid, "shape": task["shape"], "network_calls": len(calls) - before,
                   "seconds": time.perf_counter() - began, "geometry_safe": safe.tolist(),
                   "accepted": accepted.tolist(), "selected_index": selected,
                   "deployment_success": bool(safe[selected] and accepted[selected]),
                   "accepted_unsafe_candidates": int((accepted & ~safe).sum()),
                   "decoder_config": result["decoder_config"], "seed": 20461115 + tid}
            rows.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output.parent / f"smoke_task_{tid:06d}.npz", paths_px=px)
            print(json.dumps(row), flush=True)
    finally:
        hook.remove()
    report = {"status": "passed", "finished_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "checkpoint_sha256": digest(args.checkpoint), "script_sha256": digest(__file__),
              "network_calls": len(calls), "cached_fields_used": False, "training": False,
              "holdout_used": False, "scope": "Three development cases for API integration smoke; not a quality estimate",
              "tasks": rows}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
