"""Check the 275-case bounded fallback result and preserve the 274 baseline."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-run", type=Path, required=True)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--bounded-run", type=Path, required=True)
    args = p.parse_args()
    run = args.bounded_run
    final_dir = run / "evaluation/cached_full286"
    old_dir = args.baseline_run / "evaluation/cached_full286_final"
    original_dir = args.original_run / "evaluation/paired_best_val_epoch15"
    final, old, original = (read(path / "report.json") for path in (final_dir, old_dir, original_dir))
    receipt = read(run / "checkpoints/p37_bounded_fallback.json")
    assert digest(run / "checkpoints/p37_bounded_fallback.pth") == receipt["output_sha256"] == final["manifest"]["checkpoint_sha256"]
    assert digest(args.baseline_run / "checkpoints/p37_adaptive_fallback.pth") == receipt["source_sha256"]
    assert digest(old_dir / "report.json") == final["manifest"]["baseline_report_sha256"]
    assert final["manifest"]["implementation"] == "sampler" and final["manifest"]["comparison_mode"] == "adaptive_fallback"
    preserved = read(args.baseline_run / "source_snapshot_manifest.json")
    for name, expected in preserved["files_sha256"].items():
        assert digest(args.baseline_run / "source" / name) == expected, name
    for name, expected in final["manifest"]["source_sha256"].items():
        assert digest(ROOT / name) == expected
        assert digest(final_dir / "source" / Path(name).name) == expected
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in (
        args.original_run / "checkpoints/best_p37.pth",
        args.baseline_run / "checkpoints/p37_adaptive_fallback.pth", run / "checkpoints/p37_bounded_fallback.pth")]
    for name, value in states[0]["model_state_dict"].items():
        assert all(torch.equal(value, state["model_state_dict"][name]) for state in states[1:])
    for key, value in original["manifest"]["decoder_settings"].items():
        assert states[-1]["config"][key] == value
    old_by_id, field_by_id = ({r["task_id"]: r for r in report["tasks"]} for report in (old, original))
    assert [r["task_id"] for r in final["tasks"]] == [r["task_id"] for r in old["tasks"]]
    with np.load(original["manifest"]["binding"]["data_path"], allow_pickle=True) as data:
        starts, goals = data["starts"], data["goals"]
    unchanged = 0
    recovered = []
    for row in final["tasks"]:
        tid = row["task_id"]
        previous = old_by_id[tid]
        with np.load(original_dir / field_by_id[tid]["field_cache"]) as saved:
            field_hash = hashlib.sha256(np.ascontiguousarray(saved["fields"]).tobytes()).hexdigest()
        assert field_hash == field_by_id[tid]["fields_sha256"] == row["fields_sha256"]
        with np.load(final_dir / "paths" / f"task_{tid:06d}.npz") as saved:
            paths = saved["paths_px"]
        assert paths.shape == (4, 512, 2) and np.isfinite(paths).all()
        assert np.array_equal(paths[:, 0], np.broadcast_to(starts[tid], (4, 2)))
        assert np.array_equal(paths[:, -1], np.broadcast_to(goals[tid], (4, 2)))
        safe, accepted = np.asarray(row["geometry_safe"]), np.asarray(row["accepted"])
        assert int((accepted & ~safe).sum()) == row["accepted_unsafe_candidates"] == 0
        assert row["deployment_success"] == bool(safe[row["selected_index"]] and accepted[row["selected_index"]])
        attempted = row["diagnostics"]["bounded_attempted"]
        assert any(attempted) == (not any(previous["accepted"]))
        if any(previous["accepted"]):
            with np.load(old_dir / "paths" / f"task_{tid:06d}.npz") as saved:
                assert np.array_equal(paths, saved["paths_px"])
            assert row["accepted"] == previous["accepted"] and row["selected_index"] == previous["selected_index"]
            for key, value in previous["diagnostics"].items():
                assert row["diagnostics"][key] == value
            unchanged += 1
        if row["deployment_success"] and not previous["deployment_success"]:
            selected = row["selected_index"]
            recovered.append({"task_id": tid, "selected_candidate": selected,
                "internal_steps": row["diagnostics"]["adaptive_internal_steps"][selected],
                "remaining_before_actual_goal_link_px": row["diagnostics"]["adaptive_remaining_distance_px"][selected]})
    assert final["summaries"]["overall"]["success"] == 275
    assert final["summaries"]["large"]["success"] == 33
    assert not final["summaries"]["overall"]["losses"] and unchanged == 274
    assert [r["task_id"] for r in recovered] == [25782]
    result = {"status": "passed", "verified_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": digest(__file__), "field_hashes_verified": len(final["tasks"]),
        "finite_exact_endpoint_512_candidates": len(final["tasks"]) * 4,
        "unchanged_accepted_tasks_bitwise": unchanged, "identical_parameter_tensors": len(states[0]["model_state_dict"]),
        "baseline_source_files_preserved": len(preserved["files_sha256"]),
        "original_safety_config_preserved": True, "report_sha256": digest(final_dir / "report.json"),
        "recovered": recovered, "summaries": final["summaries"],
        "remaining_failures": [r["task_id"] for r in final["tasks"] if not r["deployment_success"]]}
    (run / "integrity_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
