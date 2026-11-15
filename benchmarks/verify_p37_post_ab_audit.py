"""Independently check the post-A/B cached replay, controls, and preserved inputs."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "training/source/map_flow_matching_p1"),
               str(ROOT / "training/source/map_flow_matching")]
from p37_post_ab_regression_audit import digest, read, write_json, summarize
from scripts.geometry_eval import score_task


def compare_success(rows, baseline):
    previous = {r["task_id"]: r for r in baseline["tasks"]}
    return {"gains": [r["task_id"] for r in rows if r["deployment_success"] and not previous[r["task_id"]]["deployment_success"]],
            "losses": [r["task_id"] for r in rows if not r["deployment_success"] and previous[r["task_id"]]["deployment_success"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--ab-run", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--bounded-baseline", type=Path, required=True)
    args = parser.parse_args()
    run = args.audit_dir
    report = read(run / "bounded_report.json")
    manifest = read(run / "manifest.json")
    assert manifest == report["manifest"] and manifest["status"] == "complete"
    assert manifest["network_queries"] == 0 and not manifest["training"] and not manifest["holdout_used"]
    assert digest(manifest["checkpoint"]) == manifest["checkpoint_sha256"]
    for path, expected in manifest["source_sha256"].items():
        assert digest(ROOT / path) == digest(run / "source" / path) == expected
    frozen = read(args.ab_run / "source_snapshot_manifest.json")["files_sha256"]
    for path, expected in frozen.items():
        assert digest(args.ab_run / "source" / path) == expected
    for name, expected in manifest["input_report_sha256"].items():
        assert digest(run / "inputs" / f"{name}_report.json") == expected
    previous = read(run / "inputs/B_route_increment_report.json")
    by_id = {r["task_id"]: r for r in previous["tasks"]}
    assert len(report["tasks"]) == len(previous["tasks"]) == 286
    assert [r["task_id"] for r in report["tasks"]] == [r["task_id"] for r in previous["tasks"]]
    expected_data_hash = read(args.ab_run / "B_route_increment/run_manifest.json")["data_sha256"]["train_data.npz"]
    assert digest(manifest["data_path"]) == expected_data_hash
    original = read(run / "inputs/original_report.json")
    original_rows = {r["task_id"]: r for r in original["tasks"]}
    original_dir = args.original_run / "evaluation/paired_best_val_epoch15"
    assert digest(original_dir / "report.json") == digest(run / "inputs/original_report.json")
    with np.load(manifest["data_path"], allow_pickle=True) as data:
        maps, map_ids, starts, goals = (data[k] for k in ("unique_maps", "map_ids", "starts", "goals"))
        geometry = str(data["geometry_json"])
    unchanged = 0
    for number, row in enumerate(report["tasks"], 1):
        tid = row["task_id"]
        old = by_id[tid]
        with np.load(args.ab_run / "B_route_increment/evaluation/fields" / f"task_{tid:06d}.npz") as saved:
            fields, latent = saved["fields"], saved["latent"]
            assert str(saved["checkpoint_sha256"]) == manifest["checkpoint_sha256"]
            assert np.isfinite(fields).all()
            assert hashlib.sha256(np.ascontiguousarray(fields).tobytes()).hexdigest() == row["fields_sha256"] == old["fields_sha256"]
        with np.load(original_dir / original_rows[tid]["field_cache"]) as saved:
            assert np.array_equal(latent, saved["latent"]), tid
        with np.load(run / "bounded_paths" / f"task_{tid:06d}.npz") as saved:
            paths = saved["paths_px"]
        assert paths.shape == (4, 512, 2) and np.isfinite(paths).all()
        assert np.array_equal(paths[:, 0], np.broadcast_to(starts[tid], (4, 2)))
        assert np.array_equal(paths[:, -1], np.broadcast_to(goals[tid], (4, 2)))
        safe = np.asarray(score_task((str(ROOT / "training/source/map_flow_matching"), paths,
                                     maps[map_ids[tid]], starts[tid], goals[tid], geometry)), bool)
        assert safe.tolist() == row["geometry_safe"]
        accepted = np.asarray(row["accepted"], bool)
        selected = row["selected_index"]
        assert row["accepted_unsafe_candidates"] == int((accepted & ~safe).sum())
        assert row["deployment_success"] == bool(accepted[selected] and safe[selected])
        assert bool(any(row["diagnostics"]["bounded_attempted"])) == (not any(old["accepted"]))
        if any(old["accepted"]):
            with np.load(args.ab_run / "B_route_increment/evaluation/paths" / f"task_{tid:06d}.npz") as saved:
                assert np.array_equal(paths, saved["paths_px"]), tid
            assert row["accepted"] == old["accepted"] and selected == old["selected_index"]
            for key, value in old["diagnostics"].items():
                assert row["diagnostics"][key] == value, (tid, key)
            unchanged += 1
        if number % 100 == 0:
            print({"verified_tasks": number}, flush=True)
    assert report["summaries"] == summarize(report["tasks"], by_id)
    assert report["summaries"]["overall"]["success"] == unchanged == 277
    assert report["summaries"]["large"]["success"] == 35
    assert not report["summaries"]["overall"]["gains"] and not report["summaries"]["overall"]["losses"]
    assert report["summaries"]["overall"]["accepted_unsafe_candidates"] == 0
    bounded_old = read(args.bounded_baseline / "evaluation/cached_full286/report.json")
    bounded_settings_match = all(digest(ROOT / name) == value for name, value in bounded_old["manifest"]["source_sha256"].items())
    assert bounded_settings_match

    case = read(run / "case_report.json")
    controls = read(run / "case_controls.json")
    assert case["latents_identical"] and case["original_replays_bitwise_verified"]
    assert controls["corrected_field_roundtrip_bitwise_verified"]
    tid = case["task_id"]
    for path in (run / "case_paths").glob("*.npz"):
        with np.load(path) as saved:
            paths = saved["paths_px"]
        assert paths.shape == (4, 512, 2) and np.isfinite(paths).all()
        assert np.array_equal(paths[:, 0], np.broadcast_to(starts[tid], (4, 2)))
        assert np.array_equal(paths[:, -1], np.broadcast_to(goals[tid], (4, 2)))
    stages = case["arms"]["B_route_increment"]["bounded_fallback"]["trace"]["adaptive_stages"]
    events = case["arms"]["B_route_increment"]["bounded_fallback"]["trace"]["adaptive_rejections"]
    final_events = events[stages[-1]["events_begin"]:stages[-1]["events_end"]]
    assert len(final_events) == 4
    assert all(all("mass" in proposal["failed_checks"] for proposal in e["proposals"]) for e in final_events)
    assert all(all(proposal["segment_occupancy_min"] > .1 and proposal["segment_clearance_min"] >= .5
                   for proposal in e["proposals"]) for e in final_events)
    window = (132., 212.)
    drawdowns = {}
    with np.load(run / "case_curves.npz") as saved:
        arc = saved["arc"]
        mask = (arc >= window[0]) & (arc <= window[1])
        for name in case["arms"]:
            drawdowns[name] = []
            for candidate, curve in enumerate(saved[name + "_progress"]):
                values = curve[mask]
                running_max = np.maximum.accumulate(values)
                trough = int(np.argmax(running_max - values))
                peak = int(np.argmax(values[:trough + 1]))
                drawdowns[name].append({"candidate": candidate,
                    "maximum_drawdown": float(running_max[trough] - values[trough]),
                    "peak_arc_px": float(arc[mask][peak]), "trough_arc_px": float(arc[mask][trough])})
    verification = {"status": "passed", "verified_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    "script_sha256": digest(__file__), "preserved_AB_source_files": len(frozen),
                    "new_B_field_hashes_verified": 286, "original_latents_bitwise_equal": 286,
                    "independent_geometry_rescored_tasks": 286, "finite_exact_endpoint_candidates": 1144,
                    "old_B_successes_bitwise_preserved": unchanged, "bounded_summary": report["summaries"],
                    "B_bounded_vs_old_bounded275": compare_success(report["tasks"], bounded_old),
                    "bounded_production_source_matches_old275": bounded_settings_match,
                    "B_case_final_stops_all_backoffs_mass_fail": len(final_events),
                    "B_case_final_stops_segment_occupancy_and_clearance_pass": len(final_events),
                    "diagnostic_old_safe_path_arc_window_px": window,
                    "diagnostic_progress_drawdowns": drawdowns,
                    "holdout_used": False, "training_started": False, "network_queries": 0}
    write_json(run / "integrity_verification.json", verification)
    print({key: verification[key] for key in ("status", "new_B_field_hashes_verified",
        "old_B_successes_bitwise_preserved", "B_bounded_vs_old_bounded275")}, flush=True)


if __name__ == "__main__":
    main()
