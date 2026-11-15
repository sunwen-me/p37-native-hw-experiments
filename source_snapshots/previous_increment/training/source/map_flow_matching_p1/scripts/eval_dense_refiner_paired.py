"""External paired evaluation using immutable, previously audited Flow draws.

Compares original selector, final-path neural reselector, geometry warm-start,
and RL refinement. Exact geometry is OFFLINE scoring only. Pooled K+K options
are disclosed separately from fixed-K refined-only comparisons.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np,torch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.append(str(HERE.parent/'map_flow_matching'))
from path_refinement import DensePathRefiner,signed_distance_features,neural_candidate_selection,refiner_from_config
from scripts.cache_refinement_rollouts import worker_init,exact_label
from rl_audit import quality_metrics,summarize,paired_summary,map_hashes,task_bootstrap
from run_manifest import write_run_manifest,file_sha256


def model_from_checkpoint(path,device):
    s=torch.load(path,map_location='cpu',weights_only=False);c=dict(s['config']);c.pop('representation',None)
    m=refiner_from_config(c).to(device);m.load_state_dict(s['model_state_dict'],strict=True);m.eval().requires_grad_(False)
    return m,s


def selected_pair(bs,ns,bq,nq,bi,ni,ba,seed,na=None):
    row=np.arange(len(bs));bold=bs[row,bi]&ba;new=ns[row,ni]
    if na is not None:new=new&na
    both=bold&new
    return {'selected_success_delta':task_bootstrap(new.astype(float)-bold,seed),
      'rescued_selected_tasks':int((~bold&new).sum()),'lost_selected_tasks':int((bold&~new).sum()),
      'common_successful_tasks':int(both.sum()),
      **{key+'_delta':task_bootstrap((nq[key][row,ni]-bq[key][row,bi])[both],seed) for key in bq}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--warm-checkpoint',required=True);p.add_argument('--rl-checkpoint',required=True)
    p.add_argument('--base-archive',required=True);p.add_argument('--base-result-json',required=True);p.add_argument('--data-path',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=6);p.add_argument('--seed',type=int,default=20260921);args=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda');warm,ws=model_from_checkpoint(args.warm_checkpoint,device);rl,rs=model_from_checkpoint(args.rl_checkpoint,device)
    raw=np.load(args.base_archive,allow_pickle=False);data=np.load(args.data_path,allow_pickle=False);baseinfo=json.loads(Path(args.base_result_json).read_text())
    assert file_sha256(args.data_path)==baseinfo['evaluation_data_sha256']
    assert ws['cache_audit']['source_checkpoint_sha256']==baseinfo['generator_checkpoint_sha256']==rs['cache_audit']['source_checkpoint_sha256']
    basepx=raw['paths'][0];base=(basepx/31.5-1).astype(np.float32);n,k,h,_=base.shape
    assert len(data['maps'])==n and len(set(map_hashes(data['maps'])))==n
    assert baseinfo['provenance']['rl_training_map_overlap']==0
    traincache=np.load(Path(rs['args']['cache_dir'])/'train.npz',allow_pickle=False);valcache=np.load(Path(rs['args']['cache_dir'])/'validation.npz',allow_pickle=False)
    evalhash=set(map_hashes(data['maps']));assert not evalhash&set(traincache['map_hashes']) and not evalhash&set(valcache['map_hashes'])
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    import utils.robot_geometry as geometry_module
    manifest=write_run_manifest(output,args,[__file__,HERE/'path_refinement.py',HERE/'scripts/cache_refinement_rollouts.py',HERE/'rl_audit.py',geometry_module.__file__],
      {'base_archive':args.base_archive,'base_summary':args.base_result_json,'evaluation_data':args.data_path,'warm_checkpoint':args.warm_checkpoint,'rl_checkpoint':args.rl_checkpoint})
    paths=[[],[],[]];labels=[[],[],[]];select={name:[] for name in ['base_warm_selector','base_rl_selector','warm_only','warm_pool','rl_only','rl_pool']};scores={name:[] for name in select};available={name:[] for name in select};start=time.perf_counter()
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        with torch.inference_mode():
            for begin in range(0,n,args.batch_size):
                ids=np.arange(begin,min(begin+args.batch_size,n));b=len(ids)
                maps=torch.from_numpy(np.stack([signed_distance_features(data['maps'][i]) for i in ids])).to(device)
                basepaths=torch.from_numpy(base[ids]).to(device);fm=maps[:,None].expand(-1,k,-1,-1,-1).flatten(0,1)
                wp=warm(fm,basepaths.flatten(0,1)).reshape_as(basepaths);rp=rl(fm,basepaths.flatten(0,1)).reshape_as(basepaths)
                world=[basepx[ids],((wp+1)*31.5).cpu().numpy(),((rp+1)*31.5).cpu().numpy()]
                for version,value in enumerate(world):
                    jobs=[(value[j],data['maps'][i],data['starts'][i],data['goals'][i],str(data['geometry_json'])) for j,i in enumerate(ids)]
                    paths[version].append(value);labels[version].append(np.asarray(list(pool.map(exact_label,jobs))))
                for name,m,value in [('base_warm_selector',warm,basepaths),('base_rl_selector',rl,basepaths),('warm_only',warm,wp),('warm_pool',warm,torch.cat([basepaths,wp],1)),('rl_only',rl,rp),('rl_pool',rl,torch.cat([basepaths,rp],1))]:
                    fallback=torch.from_numpy(raw['selected'][0,ids]).to(device) if name.endswith('_pool') else None
                    fallback_available=torch.from_numpy(raw['available'][0,ids]).to(device) if name.endswith('_pool') else None
                    si,sc,logits=neural_candidate_selection(m,maps,value,fallback,fallback_available,k if name.endswith('_pool') else None);select[name].append(si.cpu().numpy());scores[name].append(sc.cpu().numpy());available[name].append(((logits[torch.arange(b,device=device),si]>=0)|(fallback_available if fallback_available is not None else torch.zeros(b,dtype=torch.bool,device=device))).cpu().numpy())
                print(json.dumps({'finished':begin+b,'total':n,'seconds':time.perf_counter()-start}),flush=True)
    paths=np.stack([np.concatenate(v) for v in paths]);labels=np.stack([np.concatenate(v) for v in labels]);success=labels[...,0].astype(bool)
    quality=quality_metrics(paths);ref=np.linalg.norm(np.diff(data['paths'],axis=1),axis=2).sum(1)
    quality['length_ratio_to_reference']=quality['length']/ref[None,:,None].clip(1e-8)
    select={name:np.concatenate(v) for name,v in select.items()};scores={name:np.concatenate(v) for name,v in scores.items()};available={name:np.concatenate(v) for name,v in available.items()}
    original_si=raw['selected'][0];original_av=raw['available'][0]
    np.testing.assert_array_equal(success[0],raw['success'][0])
    bq={key:value[0] for key,value in quality.items()};wq={key:value[1] for key,value in quality.items()};rq={key:value[2] for key,value in quality.items()}
    metrics={'original':summarize(success[0],bq,original_si,original_av)}
    metrics['base_warm_selector']=summarize(success[0],bq,select['base_warm_selector'],available['base_warm_selector'])
    metrics['base_rl_selector']=summarize(success[0],bq,select['base_rl_selector'],available['base_rl_selector'])
    metrics['warm_only']=summarize(success[1],wq,select['warm_only'],available['warm_only']);metrics['rl_only']=summarize(success[2],rq,select['rl_only'],available['rl_only'])
    pools={}
    for name,v,q in [('warm_pool',1,wq),('rl_pool',2,rq)]:
        sq=np.concatenate([success[0],success[v]],1);pq={key:np.concatenate([bq[key],q[key]],1) for key in bq};pools[name]=(sq,pq)
        metrics[name]=summarize(sq,pq,select[name],available[name])
    pairs={}
    for name,v,q in [('warm_only',1,wq),('rl_only',2,rq)]:
        pairs[name+'_vs_original']=paired_summary(success[0],success[v],bq,q,original_si,select[name],original_av,available[name],args.seed)
    pairs['rl_only_vs_warm_only']=paired_summary(success[1],success[2],wq,rq,select['warm_only'],select['rl_only'],available['warm_only'],available['rl_only'],args.seed)
    for name,(sq,pq) in pools.items():
        pairs[name+'_vs_original']=selected_pair(success[0],sq,bq,pq,original_si,select[name],original_av,args.seed,available[name])
    wsq,wpq=pools['warm_pool'];rsq,rpq=pools['rl_pool']
    pairs['rl_pool_vs_warm_pool']=selected_pair(wsq,rsq,wpq,rpq,select['warm_pool'],select['rl_pool'],available['warm_pool'],args.seed,available['rl_pool'])
    result={'tasks':n,'unique_maps':n,'flow_draws':k,'refined_only_options':k,'pooled_options':2*k,'extra_flow_draws_for_pool':0,
      'exact_geometry_online':False,'online_search':False,'online_optimization_or_repair':False,
      'reference_is_archived_teacher_not_proven_shortest':True,'selector_protocol':'paired_quality_and_dense_EDT_margin_with_original_fallback_v7','selected_success_includes_neural_abstention':True,'map_overlap_training':0,'map_overlap_validation':0,
      'warm_phase':ws['phase'],'warm_update':ws['update'],'rl_phase':rs['phase'],'rl_update':rs['update'],
      'manifest_sha256':manifest,'metrics':metrics,'paired_comparisons':pairs,'seconds':time.perf_counter()-start}
    np.savez_compressed(output/'paths.npz',paths=paths,success=success,clearance=labels[...,1],collision_fraction=labels[...,2],original_selected=original_si,original_available=original_av,
       **{'selected_'+key:value for key,value in select.items()},**{'score_'+key:value for key,value in scores.items()},**{'available_'+key:value for key,value in available.items()},**quality)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
