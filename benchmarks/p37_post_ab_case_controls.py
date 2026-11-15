"""Offline channel witnesses for one regressed task, using an old safe path.

The old accepted path and its successful candidate are diagnostic references;
channel-broadcast results cannot be reported as new model performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from p37_post_ab_regression_audit import (
    digest, read, write_json, pixels, outcome, traced_decode,
    P37FlowPlanner, _p37_map_input,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.audit_dir.resolve()
    assert not (run / "case_controls.json").exists()
    manifest = read(run / "manifest.json")
    case = read(run / "case_report.json")
    tid = case["task_id"]
    original = read(run / "inputs/original_report.json")
    model = P37FlowPlanner(manifest["checkpoint"], device=args.device).model
    assert digest(manifest["checkpoint"]) == manifest["checkpoint_sha256"]
    for key, value in original["manifest"]["decoder_settings"].items():
        assert getattr(model, key) == value
    def forbid_network(*_):
        raise AssertionError("channel controls must not query the network")
    model.vector_field.register_forward_pre_hook(forbid_network)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    with np.load(manifest["data_path"], allow_pickle=True) as data:
        image = data["unique_maps"][data["map_ids"][tid]]
        begin, end = data["starts"][tid], data["goals"][tid]
        geometry = str(data["geometry_json"])
    height, width = image.shape
    scale = np.array([(width - 1) / 2, (height - 1) / 2], np.float32)
    maps = torch.from_numpy(_p37_map_input(image, model.distance_scale))[None].to(args.device)
    start = torch.from_numpy(begin / scale - 1)[None].to(args.device)
    goal = torch.from_numpy(end / scale - 1)[None].to(args.device)
    corrected = {}
    for arm in case["arms"]:
        with np.load(run / "inputs" / f"{arm}_task_{tid:06d}.npz") as saved:
            fields = torch.from_numpy(saved["fields"]).to(args.device)
        result = model.decode_fields(maps, start, goal, fields, mode="adaptive_fallback", horizon=512, progress_smoothing=7)
        corrected[arm] = result["decoder_fields"].clone()
        # Validate the no-second-smoothing controls against real production paths.
        replay = model.decode_fields(maps, start, goal, corrected[arm], mode="adaptive_fallback", horizon=512, progress_smoothing=0)
        for key in ("paths", "accepted", "selection_scores", "selected_index"):
            assert torch.equal(replay[key], result[key]), (arm, key)
        for key, value in result["diagnostics"].items():
            assert torch.equal(replay["diagnostics"][key], value), (arm, key)

    successful = case["arms"]["epoch15"]["adaptive_fallback"]["selected_index"]
    assert case["arms"]["epoch15"]["adaptive_fallback"]["deployment_success"]
    with np.load(run / "case_paths/epoch15_adaptive_fallback.npz") as saved:
        old_path = saved["paths_px"][successful]
    arc = np.r_[0., np.linalg.norm(np.diff(old_path.astype(np.float64), axis=0), axis=1).cumsum()]
    targets = np.linspace(0, arc[-1], int(np.ceil(arc[-1] * 4)) + 1)
    unique = np.r_[True, np.diff(arc) > 1e-9]
    xy = np.stack([np.interp(targets, arc[unique], old_path[unique, axis]) for axis in (0, 1)], 1)
    tree = cKDTree(xy)
    curves = {"xy": xy, "arc": targets, "old_accepted_path": old_path,
              "image": image, "start": begin, "goal": end}
    measurements = {}
    for arm, fields in corrected.items():
        array = fields[0].cpu().numpy()
        occupancy = np.stack([map_coordinates(f[0], [xy[:, 1], xy[:, 0]], order=1, mode="nearest") for f in array])
        progress = np.stack([map_coordinates(f[1], [xy[:, 1], xy[:, 0]], order=1, mode="nearest") for f in array])
        curves[arm + "_occupancy"] = occupancy
        curves[arm + "_progress"] = progress
        candidates = []
        for c in range(4):
            missing = occupancy[c] <= model.decode_threshold
            edges = np.diff(np.r_[False, missing, False].astype(int))
            runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
            candidates.append({
                "candidate": c, "old_safe_path_support_recall": float((~missing).mean()),
                "minimum_support_on_old_safe_path": float(occupancy[c].min()),
                "longest_missing_interval_px": max((float(targets[min(b, len(targets) - 1)] - targets[a]) for a, b in runs), default=0.),
                "missing_intervals_arc_px": [[float(targets[a]), float(targets[min(b, len(targets) - 1)])] for a, b in runs],
            })
        stops = case["arms"][arm]["bounded_fallback"]["trace"]["adaptive_rejections"]
        stages = case["arms"][arm]["bounded_fallback"]["trace"]["adaptive_stages"]
        projections = []
        if stages:
            for event in stops[stages[-1]["events_begin"]:stages[-1]["events_end"]]:
                distance, index = tree.query(event["previous_px"])
                projections.append({"candidate": event["candidate"],
                                    "nearest_old_safe_arc_px": float(targets[index]),
                                    "nearest_old_safe_arc_fraction": float(targets[index] / targets[-1]),
                                    "distance_to_old_safe_path_px": float(distance),
                                    "decoder_phase": event["phase"]})
        measurements[arm] = {"candidates": candidates, "bounded_stop_projection": projections}
    np.savez_compressed(run / "case_curves.npz", **curves)

    controls = []
    for arm in ("A_cfm", "B_route_increment"):
        old = corrected["epoch15"]
        for kind in ("new_support_old_success_progress", "old_success_support_new_progress"):
            hybrid = corrected[arm].clone()
            if kind == "new_support_old_success_progress":
                hybrid[:, :, 1:2] = old[:, successful:successful + 1, 1:2]
            else:
                hybrid[:, :, :1] = old[:, successful:successful + 1, :1]
            for mode in ("adaptive_fallback", "bounded_fallback"):
                result, trace = traced_decode(model, maps, start, goal, hybrid, mode, smoothing=0)
                px = pixels(result, scale, begin, end)
                row = outcome(result, px, image, begin, end, geometry)
                row.update(arm=arm, control=kind, decoder=mode,
                           old_successful_candidate=successful, trace=trace, diagnostic_only=True)
                controls.append(row)
                np.savez_compressed(run / "case_paths" / f"{arm}_{kind}_{mode}.npz", paths_px=px)
                print({"arm": arm, "control": kind, "decoder": mode,
                       "success": row["deployment_success"], "accepted": row["accepted"]}, flush=True)
    write_json(run / "case_controls.json", {
        "task_id": tid, "script_sha256": digest(__file__),
        "old_successful_candidate": successful, "old_accepted_path_length_px": float(targets[-1]),
        "corrected_field_roundtrip_bitwise_verified": True,
        "measurements_on_old_accepted_safe_path": measurements, "controls": controls,
        "network_queries": 0, "diagnostic_only": True,
        "limitations": ["Old successful-candidate channels are explicitly chosen with prior evaluation knowledge.",
                        "These channel witnesses are not a deployable model or model-quality estimates.",
                        "A different predicted mode may not follow the old safe path."]})


if __name__ == "__main__":
    main()
