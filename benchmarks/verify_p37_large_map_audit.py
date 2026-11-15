"""Check cached diagnostic artifacts without network inference or training."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    base_dir = run / "evaluation/paired_best_val_epoch15"
    audit_dir = run / "diagnostics/large_map_failure_20261115"
    controls_dir = run / "diagnostics/large_map_decoder_controls_20261115"
    smooth_dir = run / "diagnostics/progress_smooth7_full286_20261115"
    base, audit, controls, smooth = [read(p / "report.json") for p in
                                    (base_dir, audit_dir, controls_dir, smooth_dir)]
    trace = read(audit_dir / "step_trace.json")
    by_id = {r["task_id"]: r for r in base["tasks"]}
    smooth_by_id = {r["task_id"]: r for r in smooth["tasks"]}
    assert len(by_id) == len(base["tasks"]) == len(smooth["tasks"]) == 286
    assert [r["task_id"] for r in base["tasks"]] == [r["task_id"] for r in smooth["tasks"]]
    assert digest(base_dir / "report.json") == smooth["manifest"]["baseline_report_sha256"]
    assert digest(audit_dir / "report.json") == controls["manifest"]["source_audit_sha256"]
    assert digest(audit_dir / "report.json") == trace["audit_sha256"]
    script_names = ("p37_large_map_failure_audit.py", "p37_large_map_decoder_controls.py",
                    "p37_progress_smoothing_eval.py", "p37_large_map_step_trace.py")
    for name, manifest in zip(script_names, [audit["manifest"], controls["manifest"], smooth["manifest"], trace]):
        assert digest(Path(__file__).parent / name) == manifest["script_sha256"]
    checkpoints = {name: digest(run / "checkpoints" / name)
                   for name in ("best_p37.pth", "latest_training_state.pth")}
    assert checkpoints["best_p37.pth"] == base["manifest"]["binding"]["checkpoint_sha256"]
    assert checkpoints["latest_training_state.pth"] == "e68fbf1123f303f047118a7520cc34b082c8a6ddb5ab08b3eba28e4db2443c35"
    source_hashes = {name: digest(run / "source" / name)
                     for name in base["manifest"]["source_sha256"]}
    assert source_hashes == base["manifest"]["source_sha256"]
    with np.load(base["manifest"]["binding"]["data_path"], allow_pickle=True) as data:
        starts, goals = data["starts"], data["goals"]
    for row in smooth["tasks"]:
        tid = row["task_id"]
        original = by_id[tid]
        with np.load(base_dir / original["field_cache"], allow_pickle=False) as cached:
            field_hash = hashlib.sha256(np.ascontiguousarray(cached["fields"]).tobytes()).hexdigest()
        assert field_hash == original["fields_sha256"] == row["fields_sha256"]
        assert row["baseline_success"] == original["arms"]["scale_aware_512"]["deployment_success"]
        with np.load(smooth_dir / "paths" / f"task_{tid:06d}.npz", allow_pickle=False) as saved:
            paths = saved["paths_px"]
        assert paths.shape == (4, 512, 2) and np.isfinite(paths).all()
        assert np.array_equal(paths[:, 0], np.broadcast_to(starts[tid], (4, 2)))
        assert np.array_equal(paths[:, -1], np.broadcast_to(goals[tid], (4, 2)))
        safe, accepted = np.asarray(row["geometry_safe"]), np.asarray(row["accepted"])
        selected = row["selected_index"]
        assert row["deployment_success"] == bool(safe[selected] and accepted[selected])
        assert row["accepted_unsafe_candidates"] == int((accepted & ~safe).sum())
    overlap_error = baseline_error = 0.0
    audit_by_id = {r["task_id"]: r for r in audit["tasks"]}
    assert len(audit_by_id) == len(controls["tasks"]) == 52
    for row in controls["tasks"]:
        tid = row["task_id"]
        first = audit_by_id[tid]["variants"]["prediction"]
        original = by_id[tid]["arms"]["scale_aware_512"]
        assert first["deployment_success"] == original["deployment_success"]
        assert first["geometry_safe"] == original["candidate_geometry_safe"]
        assert first["accepted"] == original["accepted"]
        with np.load(audit_dir / "paths" / f"task_{tid:06d}.npz") as saved, np.load(base_dir / "paths" / f"task_{tid:06d}.npz") as original_paths:
            baseline_error = max(baseline_error, float(np.abs(saved["prediction"] - original_paths["paths_px_scale_aware_512"]).max()))
        assert baseline_error <= 1e-4
        variant = row["variants"]["progress_support_weighted_smooth7"]
        for key in ("deployment_success", "geometry_safe", "accepted", "selected_index"):
            assert variant[key] == smooth_by_id[tid][key]
        with np.load(controls_dir / "paths" / f"task_{tid:06d}.npz") as saved, np.load(smooth_dir / "paths" / f"task_{tid:06d}.npz") as full:
            overlap_error = max(overlap_error, float(np.abs(saved["progress_support_weighted_smooth7"] - full["paths_px"]).max()))
        assert overlap_error <= 1e-4
        transpose = row["variants"]["transpose_cached_fields"]
        assert transpose["candidate_geometry_booleans_equal"] and transpose["acceptance_booleans_equal"]
    large = [r for r in audit["tasks"] if r["is_large"]]
    collisions = [c for r in large for c in r["baseline_collisions"] if not c["geometry_safe"]]
    assert len(large) == 44 and len(collisions) == 117
    assert all(c["prefix_before_forced_goal_safe"] and c["forced_goal_segment_collision"]
               and c["collision_segments"] == 1 and c["first_collision_segment"] == 511 for c in collisions)
    assert sum(r["variants"]["oracle_both"]["deployment_success"] for r in large) == 44
    summary = {"overall": {"tasks": 286, "baseline_success": sum(r["baseline_success"] for r in smooth["tasks"]),
                           "success": sum(r["deployment_success"] for r in smooth["tasks"]),
                           "gains": [r["task_id"] for r in smooth["tasks"] if r["deployment_success"] and not r["baseline_success"]],
                           "losses": [r["task_id"] for r in smooth["tasks"] if r["baseline_success"] and not r["deployment_success"]],
                           "accepted_unsafe_candidates": sum(r["accepted_unsafe_candidates"] for r in smooth["tasks"])}}
    for shape in sorted({r["shape"] for r in smooth["tasks"]}):
        rows = [r for r in smooth["tasks"] if r["shape"] == shape]
        summary[shape] = {"tasks": len(rows), "baseline_success": sum(r["baseline_success"] for r in rows),
                          "success": sum(r["deployment_success"] for r in rows)}
    assert summary["overall"]["success"] == smooth["summaries"]["overall"]["deployment_success"] == 270
    assert not summary["overall"]["losses"] and not summary["overall"]["accepted_unsafe_candidates"]
    result = {"status": "passed", "checked_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "task_order_preserved": True, "field_hashes_verified": 286, "finite_512_point_candidates_with_exact_endpoints": 1144,
              "baseline_reproduced_tasks": 52, "baseline_max_path_error_px": baseline_error,
              "smooth_overlap_verified_tasks": 52, "smooth_overlap_max_path_error_px": overlap_error,
              "transpose_outcome_equal_tasks": 52,
              "transpose_max_path_difference_px": max(r["variants"]["transpose_cached_fields"]["max_path_difference_px_vs_original"] for r in controls["tasks"]),
              "large_colliding_candidates_only_final_segment": len(collisions), "checkpoints": checkpoints,
              "frozen_source_sha256": source_hashes, "counts": summary,
              "remaining_failures": [r["task_id"] for r in smooth["tasks"] if not r["deployment_success"]],
              "remaining_failure_classes": dict(Counter("safe_candidate_rejected" if r["any_geometry_success"] else "no_safe_candidate"
                                                         for r in smooth["tasks"] if not r["deployment_success"])),
              "script_sha256": digest(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "overall": summary["overall"], "remaining_failure_classes": result["remaining_failure_classes"]}))


if __name__ == "__main__":
    main()
