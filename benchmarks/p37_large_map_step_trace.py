"""Inspect production decoder weights at cached failure positions.

Reference-route projection is used only to describe where a point stopped.
The production score and support threshold are reproduced without labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from p37_large_map_failure_audit import digest, write_json


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run=args.run_dir.resolve();root=run/'source'
    sys.path[:0]=[str(root/'training/source/map_flow_matching_p1'),str(root/'training/source/map_flow_matching')]
    from safe_field_decoder import _progress_geometry,distance_field_to_soft_clearance
    audit=json.loads((args.audit_dir/'report.json').read_text())
    evaldir=run/'evaluation/paired_best_val_epoch15'
    original=json.loads((evaldir/'report.json').read_text())
    cfg=original['manifest']['decoder_settings'];distance_scale=original['manifest']['binding']['distance_scale']
    with np.load(original['manifest']['binding']['data_path'],allow_pickle=True) as d:
        maps,map_ids,starts,goals,routes=d['unique_maps'],d['map_ids'],d['starts'],d['goals'],d['paths']
    task_by_id={t['task_id']:t for t in original['tasks']}
    torch.set_num_threads(4)
    result=[]
    for task in audit['tasks']:
        if not task['is_large']:continue
        tid=task['task_id'];image=maps[int(map_ids[tid])];H,W=image.shape
        with np.load(evaldir/task_by_id[tid]['field_cache'],allow_pickle=False) as d:field=torch.from_numpy(d['fields'][0]).cuda()
        with np.load(args.audit_dir/'paths'/f'task_{tid:06d}.npz',allow_pickle=False) as d:paths=d['prediction']
        probability=((field[:,:1].clamp(0,1)-cfg['decode_threshold'])/(1-cfg['decode_threshold'])).clamp_min(0).pow(cfg['decode_power'])
        progress=field[:,1:2].clamp(0,1)
        factor,halfbin=_progress_geometry(probability,progress)
        beta=cfg['decode_beta']*factor[:,None]
        distance=cv2.distanceTransform(np.pad((1-image).astype(np.uint8)*255,1),cv2.DIST_L2,5)[1:-1,1:-1]
        feature=torch.from_numpy(np.clip(distance/distance_scale,0,1)).cuda()[None,None]
        safety=distance_field_to_soft_clearance(feature,distance_scale,cfg['robot_radius_grid'],cfg['clearance_temperature'],cfg['clearance_offset']).pow(cfg['decode_clearance_power'])
        base=(probability*safety).flatten(1)
        yy,xx=torch.meshgrid(torch.arange(H,device='cuda'),torch.arange(W,device='cuda'),indexing='ij')
        pixel_points=torch.stack([xx,yy],dim=-1).reshape(1,-1,2).float()
        scale=torch.tensor([(W-1)/2,(H-1)/2],device='cuda')
        route=routes[tid]
        lens=np.linalg.norm(np.diff(route.astype(np.float64),axis=0),axis=1);cum=np.r_[0.,lens.cumsum()]
        arc=np.linspace(0,cum[-1],int(cum[-1]*4)+1)
        ref_xy=np.stack([np.interp(arc,cum,route[:,k]) for k in (0,1)],axis=1)
        tree=cKDTree(ref_xy)
        events=[]
        diagnostic=task['variants']['prediction']['diagnostics']
        for c in range(4):
            first=diagnostic['first_missing_support_step'][c]
            steps=sorted({int(first) if first<511 else 510,510})
            for step in steps:
                previous=paths[c,step-1].astype(np.float32)
                center=step/511
                delta=(progress[c].flatten()-center).abs().sub(halfbin[c]).clamp_min(0)
                score=base[c]*torch.exp(-beta[c]*delta.square())
                displacement=pixel_points[0]-torch.from_numpy(previous).cuda()
                dist2=displacement.square().sum(-1)
                local=torch.exp(-cfg['decode_continuity_beta']*(displacement*2/63).square().sum(-1))
                mass=float((score*local).sum())
                local_no_progress=float((base[c]*local).sum())
                global_mass=float(score.sum())
                strong=score>1e-5
                closest=float(torch.sqrt(dist2[strong].min())) if strong.any() else None
                local_support=(dist2<=100)&(probability[c].flatten()>0)
                nearby_progress=progress[c].flatten()[local_support]
                nearest_distance,nearest_index=tree.query(previous)
                events.append({'candidate':c,'step':step,'center_progress':center,'is_first_missing_support':step==first,
                               'previous_point_px':previous.tolist(),'distance_to_goal_px':float(np.linalg.norm(previous-goals[tid])),
                               'local_mass':mass,'production_support_threshold':1e-5,
                               'local_mass_without_progress_weight':local_no_progress,'progress_mass_without_locality_weight':global_mass,
                               'nearest_strong_matching_progress_pixel_distance_px':closest,
                               'local_supported_progress_range_within10px':([float(nearby_progress.min()),float(nearby_progress.max())] if len(nearby_progress) else None),
                               'reference_progress_nearest_to_previous_point':float(arc[nearest_index]/cum[-1]),
                               'distance_to_reference_route_px':float(nearest_distance)})
        result.append({'task_id':tid,'shape':task['shape'],'baseline_success':task['baseline_deployment_success'],'events':events})
    first=[e for r in result for e in r['events'] if e['is_first_missing_support']]
    summary={'first_missing_events':len(first),
             'local_support_available_without_progress':sum(e['local_mass_without_progress_weight']>1e-5 for e in first),
             'globally_available_matching_progress':sum(e['progress_mass_without_locality_weight']>1e-5 for e in first),
             'local_mass_reproduces_missing_support':sum(e['local_mass']<=1e-5 for e in first),
             'median_first_missing_requested_progress':float(np.median([e['center_progress'] for e in first]))}
    write_json(args.output,{'created_local':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                            'script_sha256':digest(__file__),'audit_sha256':digest(args.audit_dir/'report.json'),
                            'summary':summary,'tasks':result,
                            'limitations':['Previous points are reconstructed from saved float32 pixel paths; mass may differ by rounding near the support threshold.',
                                           'Nearest-reference progress is an offline diagnostic and does not exclude alternative valid route modes.']})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
