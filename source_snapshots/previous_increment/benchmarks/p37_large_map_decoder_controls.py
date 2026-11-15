"""Predicted-field-only controls for the large-map failure audit.

These are development diagnostics, not selected deployment settings. Keep the
same acceptance gates and report unsafe accepted outputs and regressions.
"""
from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from p37_large_map_failure_audit import digest, load_module, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "paths").mkdir(exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("completed report exists")
    run = args.run_dir.resolve()
    root = run / "source"
    sys.path[:0] = [str(root / "training/source/map_flow_matching_p1"),
                    str(root / "training/source/map_flow_matching"),
                    str(root / "training/source/map_flow_matching_p1/scripts")]
    evaluator = load_module("frozen_eval_controls", root / "benchmarks/p37_paired_decoder_eval.py")
    from p37_model import endpoint_fields
    from safe_field_decoder import _progress_geometry, distance_field_to_soft_clearance
    audit = json.loads((args.audit_dir / "report.json").read_text())
    evaldir = run / "evaluation/paired_best_val_epoch15"
    original = json.loads((evaldir / "report.json").read_text())
    task_by_id = {t["task_id"]:t for t in original["tasks"]}
    decoder = SimpleNamespace(**original["manifest"]["decoder_settings"])
    decoder.distance_scale = original["manifest"]["binding"]["distance_scale"]
    data_path = Path(original["manifest"]["binding"]["data_path"])
    with np.load(data_path, allow_pickle=True) as data:
        maps, map_ids = data["unique_maps"], data["map_ids"]
        starts_px, goals_px, geometry_json = data["starts"], data["goals"], str(data["geometry_json"])
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    rows = []
    manifest = {"status":"running","created_local":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "source_audit_sha256":digest(args.audit_dir / "report.json"),"script_sha256":digest(__file__),
                "uses_teacher_fields":False,"training":False,"thresholds_relaxed":False,
                "scope":"44 development large-map cases plus eight controls; exploratory settings are not deployment claims"}
    write_json(output / "manifest.json", manifest)
    for position, reference in enumerate(audit["tasks"],1):
        tid = reference["task_id"]
        task = task_by_id[tid]
        image = maps[int(map_ids[tid])]
        height, width = image.shape
        scale = np.asarray([(width-1)/2,(height-1)/2],dtype=np.float32)
        start = torch.from_numpy(starts_px[tid]/scale-1)[None].to(device)
        goal = torch.from_numpy(goals_px[tid]/scale-1)[None].to(device)
        distance = cv2.distanceTransform(np.pad((1-image).astype(np.uint8)*255,1),cv2.DIST_L2,5)[1:-1,1:-1]
        feature = torch.from_numpy(np.clip(distance/decoder.distance_scale,0,1)).to(device)[None,None]
        map_input = torch.cat([torch.from_numpy(image.astype(np.float32)).to(device)[None,None],feature],dim=1)
        with np.load(evaldir/task["field_cache"],allow_pickle=False) as cache:
            field_np = cache["fields"]
        assert evaluator.array_sha256(field_np)==task["fields_sha256"]
        fields = torch.from_numpy(field_np).to(device)
        fixed = endpoint_fields(start,goal,height,width)[:,None]
        variants=[]
        for multiplier in (.25,4.):
            altered_decoder=copy(decoder)
            altered_decoder.decode_beta*=multiplier
            variants.append((f"precision_x{multiplier:g}",fields,altered_decoder,False))
        occupancy=fields[:,:,0:1].clamp(0,1)
        progress=fields[:,:,1:2].clamp(0,1)
        for kernel in (3,7):
            weights=occupancy.reshape(-1,1,height,width).pow(4)
            numerator=F.avg_pool2d(weights*progress.reshape_as(weights),kernel,1,kernel//2)
            denominator=F.avg_pool2d(weights,kernel,1,kernel//2)
            modified=fields.clone()
            modified[:,:,1:2]=(numerator/denominator.clamp_min(1e-8)).reshape_as(progress)
            modified=modified*(1-fixed[:,:,:1])+fixed
            variants.append((f"progress_support_weighted_smooth{kernel}",modified,decoder,False))
        modified=fields.clone()
        modified[:,:,1:2]=(progress/occupancy.clamp_min(.1)).clamp(0,1)
        modified=modified*(1-fixed[:,:,:1])+fixed
        variants.append(("progress_divide_occupancy",modified,decoder,False))
        variants.append(("transpose_cached_fields",fields.transpose(-2,-1).contiguous(),decoder,True))
        row={"task_id":tid,"shape":reference["shape"],"is_large":reference["is_large"],
             "baseline_success":reference["baseline_deployment_success"],"variants":{}}
        payload={}
        for name,variant,settings,transpose in variants:
            input_map=map_input.transpose(-2,-1).contiguous() if transpose else map_input
            s=start.flip(-1) if transpose else start
            g=goal.flip(-1) if transpose else goal
            decoded=evaluator.decode_fields(settings,input_map,s,g,variant,"scale_aware",512)
            paths=decoded["paths"].cpu().numpy()[0]
            if transpose:paths=paths[...,::-1].copy()
            world=(paths+1)*scale
            world[:,0],world[:,-1]=starts_px[tid],goals_px[tid]
            safe=np.asarray(evaluator.score_task((str(root/"training/source/map_flow_matching"),world,
                                                 image,starts_px[tid],goals_px[tid],geometry_json)))
            accepted=decoded["accepted"].cpu().numpy()[0]
            valid=decoded["decoder_valid"].cpu().numpy()[0]
            selected=int(decoded["selected_index"].item())
            value=evaluator.candidate_outcomes(safe,valid,accepted,selected)
            value.update(selected_index=selected,geometry_safe=safe.tolist(),accepted=accepted.tolist(),
                         diagnostics={k:v.cpu().tolist()[0] for k,v in decoded["diagnostics"].items()})
            if transpose:
                with np.load(evaldir/"paths"/f"task_{tid:06d}.npz",allow_pickle=False) as saved:
                    value["max_path_difference_px_vs_original"]=float(np.abs(world-saved["paths_px_scale_aware_512"]).max())
                value["candidate_geometry_booleans_equal"]=safe.tolist()==task["arms"]["scale_aware_512"]["candidate_geometry_safe"]
                value["acceptance_booleans_equal"]=accepted.tolist()==task["arms"]["scale_aware_512"]["accepted"]
            payload[name]=world
            row["variants"][name]=value
        np.savez_compressed(output/"paths"/f"task_{tid:06d}.npz",**payload)
        rows.append(row)
        with (output/"tasks.jsonl").open("a") as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        print(json.dumps({"position":position,"total":len(audit["tasks"]),"task_id":tid,
                          "baseline":row["baseline_success"],"success":{k:v["deployment_success"] for k,v in row["variants"].items()}}),flush=True)
    summary={}
    for group in ("large","controls","512x768","768x512"):
        subset=[r for r in rows if (r["is_large"] if group=="large" else not r["is_large"] if group=="controls" else r["shape"]==group)]
        summary[group]={"tasks":len(subset),"variants":{}}
        for name in rows[0]["variants"]:
            summary[group]["variants"][name]={k:sum(int(r["variants"][name][k]) for r in subset) for k in ("deployment_success","any_geometry_success","selected_accepted_unsafe","accepted_unsafe_candidates","abstain")}
            summary[group]["variants"][name]["gains"]=[r["task_id"] for r in subset if not r["baseline_success"] and r["variants"][name]["deployment_success"]]
            summary[group]["variants"][name]["losses"]=[r["task_id"] for r in subset if r["baseline_success"] and not r["variants"][name]["deployment_success"]]
    manifest.update(status="complete",finished_local=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write_json(output/"manifest.json",manifest)
    write_json(output/"report.json",{"manifest":manifest,"summaries":summary,"tasks":rows})
    print(json.dumps({"summaries":summary}),flush=True)


if __name__=="__main__":
    main()
