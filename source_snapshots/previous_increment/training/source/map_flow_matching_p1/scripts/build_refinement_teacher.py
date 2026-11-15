"""TRAINING ONLY geometry teacher. Never imported by online deployment.

Optimize each fixed cached path independently on model-visible map geometry;
then exact offline footprint labels decide whether a target is usable. Keeps
identity targets when optimization harms safety or both quality objectives.
This teacher is an explicit auxiliary, not a claim that RL caused its gains.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np,torch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.append(str(HERE.parent/'map_flow_matching'))
from path_refinement import DensePathRefiner,path_metrics,swept_clearance
from scripts.cache_refinement_rollouts import exact_label,worker_init
from run_manifest import write_run_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-dir',required=True);p.add_argument('--save-dir',required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda');out=Path(args.save_dir);out.mkdir(parents=True,exist_ok=True)
    cache=np.load(Path(args.cache_dir)/'train.npz',allow_pickle=False);b,k,h,_=cache['base_paths'].shape
    # Only training split is used; validation labels never construct targets.
    manifest=write_run_manifest(out,args,[__file__,HERE/'path_refinement.py',HERE/'scripts/cache_refinement_rollouts.py'],{'training_cache':Path(args.cache_dir)/'train.npz'})
    targets=[];start=time.perf_counter()
    allbase=cache['base_paths'].reshape(-1,h,2)
    for begin in range(0,len(allbase),args.batch_size):
        flat=np.arange(begin,min(begin+args.batch_size,len(allbase)));ids=flat//k
        base=torch.from_numpy(allbase[flat]).to(device);maps=torch.from_numpy(cache['maps'][ids]).to(device);ref=torch.from_numpy(cache['reference_length'][ids]).to(device)
        off=torch.zeros_like(base,requires_grad=True);opt=torch.optim.Adam([off],lr=.04)
        for step in range(args.steps):
            path=DensePathRefiner.apply_offset(base,off)
            q=path_metrics(path);clearance=swept_clearance(path,maps)
            deficit=(1.5+.2-clearance).clamp_min(0)
            loss=.8*q['length']/ref.clamp_min(1)+.8*q['smoothness']/.1+10*(deficit.square().mean(1)+deficit.amax(1).square())+.01*off.square().mean((1,2))
            opt.zero_grad();loss.sum().backward();opt.step()
            with torch.no_grad():off.clamp_(-3,3);off[:,[0,-1]]=0
        targets.append(DensePathRefiner.apply_offset(base,off.detach()).cpu().numpy())
        print(json.dumps({'optimized':int(flat[-1])+1,'total':len(allbase),'seconds':time.perf_counter()-start}),flush=True)
    targets=np.concatenate(targets).reshape(b,k,h,2)
    px=(targets+1)*31.5
    jobs=[(px[j],cache['occupancy'][j],cache['starts'][j],cache['goals'][j],str(cache['geometry_json'])) for j in range(b)]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        label=np.asarray(list(pool.map(exact_label,jobs)),dtype=np.float32)
    tq=path_metrics(torch.from_numpy(targets.reshape(-1,h,2)));bq=path_metrics(torch.from_numpy(allbase))
    length=tq['length'].numpy().reshape(b,k);smooth=tq['smoothness'].numpy().reshape(b,k)
    bl=bq['length'].numpy().reshape(b,k);bs=bq['smoothness'].numpy().reshape(b,k)
    success=label[:,:,0].astype(bool);original_success=cache['success']
    # Teach quality only if BOTH length and smoothness did not worsen, and
    # exact safety is true. Failed originals also get safe rescue targets.
    usable=success&((~original_success)|((length<=bl+1e-4)&(smooth<=bs+1e-6)))
    target=np.where(usable[:,:,None,None],targets,cache['base_paths'])
    weights=np.where(usable,1.,np.where(original_success,1.,.1)).astype(np.float32)
    np.savez_compressed(out/'teacher.npz',targets=target,weights=weights,usable=usable,optimized_success=success,optimized_clearance=label[:,:,1])
    summary={'manifest_sha256':manifest,'train_only':True,'base_candidate_success':float(original_success.mean()),'optimized_candidate_success':float(success.mean()),
      'usable_fraction':float(usable.mean()),'rescued_candidates':int((~original_success&success).sum()),'rejected_unsafe_losses':int((original_success&~success).sum()),
      'usable_common_length_delta':float((length-bl)[usable&original_success].mean()),'usable_common_smoothness_delta':float((smooth-bs)[usable&original_success].mean())}
    (out/'teacher_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
