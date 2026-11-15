"""Fixed seven-pixel predicted-progress smoothing on all cached validation tasks.

Exploratory development evaluation. The setting was examined on the 44 large
cases first; this is not independent holdout evidence or a deployment update.
"""
from __future__ import annotations

import argparse
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();run=args.run_dir.resolve();root=run/'source';out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True);(out/'paths').mkdir(exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Completed report exists')
    sys.path[:0]=[str(root/'training/source/map_flow_matching_p1'),str(root/'training/source/map_flow_matching'),str(root/'training/source/map_flow_matching_p1/scripts')]
    evaluator=load_module('frozen_eval_smoothing',root/'benchmarks/p37_paired_decoder_eval.py')
    from p37_model import endpoint_fields
    evaldir=run/'evaluation/paired_best_val_epoch15';original=json.loads((evaldir/'report.json').read_text())
    settings=SimpleNamespace(**original['manifest']['decoder_settings']);settings.distance_scale=original['manifest']['binding']['distance_scale']
    with np.load(original['manifest']['binding']['data_path'],allow_pickle=True) as d:
        maps,map_ids,starts,goals,geometry=d['unique_maps'],d['map_ids'],d['starts'],d['goals'],str(d['geometry_json'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    rows=[];manifest={'status':'running','created_local':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                      'script_sha256':digest(__file__),'baseline_report_sha256':digest(evaldir/'report.json'),
                      'kernel':7,'weight':'clamp(predicted_occupancy,0,1)^4','denominator_floor':1e-8,
                      'hard_endpoint_fields_restored':True,'horizon':512,'safety_thresholds_unchanged':True,
                      'teacher_fields_used':False,'network_queries':0,'training':False,'holdout_used':False,
                      'setting_selected_on_same_development_tasks':True}
    write_json(out/'manifest.json',manifest)
    for i,t in enumerate(original['tasks'],1):
        tid=t['task_id'];image=maps[int(map_ids[tid])];H,W=image.shape
        scale=np.array([(W-1)/2,(H-1)/2],np.float32)
        s=torch.from_numpy(starts[tid]/scale-1)[None].cuda();g=torch.from_numpy(goals[tid]/scale-1)[None].cuda()
        distance=cv2.distanceTransform(np.pad((1-image).astype(np.uint8)*255,1),cv2.DIST_L2,5)[1:-1,1:-1]
        feature=torch.from_numpy(np.clip(distance/settings.distance_scale,0,1)).cuda()[None,None]
        map_input=torch.cat([torch.from_numpy(image.astype(np.float32)).cuda()[None,None],feature],dim=1)
        with np.load(evaldir/t['field_cache'],allow_pickle=False) as d:field_np=d['fields']
        assert evaluator.array_sha256(field_np)==t['fields_sha256']
        fields=torch.from_numpy(field_np).cuda();weights=fields[:,:,0:1].clamp(0,1).reshape(-1,1,H,W).pow(4)
        numerator=F.avg_pool2d(weights*fields[:,:,1:2].clamp(0,1).reshape_as(weights),7,1,3)
        denominator=F.avg_pool2d(weights,7,1,3)
        modified=fields.clone();modified[:,:,1:2]=(numerator/denominator.clamp_min(1e-8)).reshape(1,4,1,H,W)
        fixed=endpoint_fields(s,g,H,W)[:,None];modified=modified*(1-fixed[:,:,:1])+fixed
        decoded=evaluator.decode_fields(settings,map_input,s,g,modified,'scale_aware',512)
        world=(decoded['paths'].cpu().numpy()[0]+1)*scale;world[:,0],world[:,-1]=starts[tid],goals[tid]
        safe=np.array(evaluator.score_task((str(root/'training/source/map_flow_matching'),world,image,starts[tid],goals[tid],geometry)))
        valid=decoded['decoder_valid'].cpu().numpy()[0];accepted=decoded['accepted'].cpu().numpy()[0];selected=int(decoded['selected_index'].item())
        result=evaluator.candidate_outcomes(safe,valid,accepted,selected)
        result.update(task_id=tid,shape=t['shape'],baseline_success=t['arms']['scale_aware_512']['deployment_success'],
                      selected_index=selected,geometry_safe=safe.tolist(),accepted=accepted.tolist(),
                      fields_sha256=t['fields_sha256'],diagnostics={k:v.cpu().tolist()[0] for k,v in decoded['diagnostics'].items()})
        rows.append(result);np.savez_compressed(out/'paths'/f'task_{tid:06d}.npz',paths_px=world)
        with (out/'tasks.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
        if i%25==0:print(json.dumps({'completed':i,'tasks':len(original['tasks']),'success':sum(r['deployment_success'] for r in rows)}),flush=True)
    groups={'overall':rows,'large':[r for r in rows if r['shape'] in ('512x768','768x512')]}
    groups.update({shape:[r for r in rows if r['shape']==shape] for shape in sorted({r['shape'] for r in rows})})
    summary={}
    for name,rs in groups.items():
        summary[name]={'tasks':len(rs),**{k:sum(int(r[k]) for r in rs) for k in ('deployment_success','any_geometry_success','selected_accepted_unsafe','accepted_unsafe_candidates','abstain')},
                       'gains':[r['task_id'] for r in rs if not r['baseline_success'] and r['deployment_success']],
                       'losses':[r['task_id'] for r in rs if r['baseline_success'] and not r['deployment_success']]}
    manifest.update(status='complete',finished_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    write_json(out/'manifest.json',manifest);write_json(out/'report.json',{'manifest':manifest,'summaries':summary,'tasks':rows})
    print(json.dumps({'summaries':summary}),flush=True)


if __name__=='__main__':main()
