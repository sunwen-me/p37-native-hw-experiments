"""Verify adaptive artifacts, their frozen inputs, and preservation of baselines."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from package_p37_decoder import digest


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--adaptive-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.adaptive_run.resolve()
    original_dir = args.original_run / "evaluation/paired_best_val_epoch15"
    baseline_dir = args.baseline_run / "evaluation/cached_full286_final"
    final_dir = run / "evaluation/cached_full286_final"
    original, baseline, final = [read(p / "report.json") for p in (original_dir, baseline_dir, final_dir)]
    receipt = read(run / "checkpoints/p37_adaptive_fallback.json")
    assert digest(run / "checkpoints/p37_adaptive_fallback.pth") == receipt["output_sha256"] == final["manifest"]["checkpoint_sha256"]
    assert digest(baseline_dir / "report.json") == final["manifest"]["baseline_report_sha256"]
    assert final["manifest"]["implementation"] == "sampler" and final["manifest"]["scope"] == "full"
    for relative, expected in final["manifest"]["source_sha256"].items():
        assert digest(ROOT / relative) == expected, relative
        assert digest(final_dir / "source" / Path(relative).name) == expected, relative
    preserved = read(args.baseline_run / "source_snapshot_manifest.json")
    for relative, expected in preserved["files_sha256"].items():
        assert digest(args.baseline_run / "source" / relative) == expected, relative
    best = args.original_run / "checkpoints/best_p37.pth"
    assert digest(best) == original["manifest"]["binding"]["checkpoint_sha256"]
    assert digest(args.baseline_run / "checkpoints/p37_progress_smooth7.pth") == receipt["source_sha256"]
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in
              (best, args.baseline_run / "checkpoints/p37_progress_smooth7.pth", run / "checkpoints/p37_adaptive_fallback.pth")]
    assert all(set(state["model_state_dict"]) == set(states[0]["model_state_dict"]) for state in states)
    for key, value in states[0]["model_state_dict"].items():
        assert all(torch.equal(value, state["model_state_dict"][key]) for state in states[1:]), key
    for key, expected in original["manifest"]["decoder_settings"].items():
        assert states[-1]["config"][key] == expected, key
    assert states[-1]["config"]["decoder_mode"] == "adaptive_fallback"
    assert states[-1]["config"]["decode_progress_smoothing"] == 7
    old_by_id = {r["task_id"]: r for r in baseline["tasks"]}
    fields_by_id = {r["task_id"]: r for r in original["tasks"]}
    assert [r["task_id"] for r in final["tasks"]] == [r["task_id"] for r in baseline["tasks"]]
    with np.load(original["manifest"]["binding"]["data_path"], allow_pickle=True) as data:
        starts, goals = data["starts"], data["goals"]
    unchanged = verified = 0
    recovered = []
    for row in final["tasks"]:
        tid = row["task_id"]
        old = old_by_id[tid]
        with np.load(original_dir / fields_by_id[tid]["field_cache"]) as saved:
            actual = hashlib.sha256(np.ascontiguousarray(saved["fields"]).tobytes()).hexdigest()
        assert actual == row["fields_sha256"] == fields_by_id[tid]["fields_sha256"]
        with np.load(final_dir / "paths" / f"task_{tid:06d}.npz") as saved:
            paths = saved["paths_px"]
        assert paths.shape == (4, 512, 2) and np.isfinite(paths).all()
        assert np.array_equal(paths[:, 0], np.broadcast_to(starts[tid], (4, 2)))
        assert np.array_equal(paths[:, -1], np.broadcast_to(goals[tid], (4, 2)))
        safe, accepted = np.asarray(row["geometry_safe"]), np.asarray(row["accepted"])
        selected = row["selected_index"]
        assert row["deployment_success"] == bool(accepted[selected] and safe[selected])
        assert row["accepted_unsafe_candidates"] == int((accepted & ~safe).sum()) == 0
        assert row["used_adaptive"] == (not any(old["accepted"]))
        if not row["used_adaptive"]:
            with np.load(baseline_dir / "paths" / f"task_{tid:06d}.npz") as saved:
                assert np.array_equal(paths, saved["paths_px"])
            assert row["accepted"] == old["accepted"] and selected == old["selected_index"]
            unchanged += 1
        if row["deployment_success"] and not old["deployment_success"]:
            with np.load(baseline_dir / "paths" / f"task_{tid:06d}.npz") as saved:
                previous_paths = saved["paths_px"]
            recovered.append({
                "task_id": tid, "shape": row["shape"], "selected_candidate": selected,
                "old_same_candidate_terminal_jump_px": float(np.linalg.norm(previous_paths[selected, -1] - previous_paths[selected, -2])),
                "new_terminal_jump_px": float(np.linalg.norm(paths[selected, -1] - paths[selected, -2])),
                "continuation_steps": row["diagnostics"]["adaptive_internal_steps"][selected],
                "distance_before_goal_link_px": row["diagnostics"]["adaptive_remaining_distance_px"][selected],
            })
        verified += 1
    summary = final["summaries"]
    assert summary["overall"]["success"] == 274 and summary["large"]["success"] == 32
    assert not summary["overall"]["losses"] and unchanged == 270
    assert {r["task_id"] for r in recovered} == {82444, 69406, 66847, 40128}
    result = {
        "status": "passed", "checked_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": digest(__file__), "field_hashes_verified": verified,
        "finite_512_point_candidates_with_exact_endpoints": verified * 4,
        "unchanged_accepted_tasks_bitwise": unchanged,
        "identical_parameter_tensors_vs_epoch15_and_smooth7": len(states[0]["model_state_dict"]),
        "previous_smoothing_source_files_preserved": len(preserved["files_sha256"]),
        "original_safety_config_preserved": True, "recovered": recovered,
        "summary": summary,
        "remaining_failures": [r["task_id"] for r in final["tasks"] if not r["deployment_success"]],
        "report_sha256": digest(final_dir / "report.json"),
    }
    (run / "integrity_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "recovered": recovered, "summary": summary["overall"]}))


if __name__ == "__main__":
    main()
