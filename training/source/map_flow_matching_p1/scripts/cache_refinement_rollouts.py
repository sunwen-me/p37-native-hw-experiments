"""Offline fixed-base cache for state-paired neural trajectory refinement.

Map-disjoint partitions selected before generating or inspecting rollouts.
Exact footprint labels are OFFLINE training/evaluation only. No test archive
is used for optimization or selecting model checkpoints.
"""
from __future__ import annotations
import argparse,json,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import numpy as np,torch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.append(str(HERE.parent/'map_flow_matching'))
from scripts.train_p37_flow_sde_ppo import load_model
from scripts.train_p56_path_chunk_fpo import frozen_base_paths
from scripts.train_p37_ordered_field import map_disjoint_split
from p37_model import candidate_selection_scores
from path_refinement import signed_distance_features
from run_manifest import write_run_manifest,file_sha256


def worker_init():
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def exact_label(job):
    from utils.robot_geometry import FootprintChecker,RobotGeometry
    paths,occ,start,goal,gjson=job
    checker=FootprintChecker(occ,RobotGeometry.from_json(gjson))
    results=[]
    for p in paths:
        values=np.asarray([checker.segment_clearance(a,b) for a,b in zip(np.vstack([start,p,goal])[:-1],np.vstack([start,p,goal])[1:])])
        minimum=float(values.min())
        success=minimum>checker.geometry.radius_grid+1e-10 and np.linalg.norm(p[0]-start)<=1e-5 and np.linalg.norm(p[-1]-goal)<=1e-5
        results.append([float(success),minimum,float((values<=checker.geometry.radius_grid+1e-10).mean())])
    return results


def balanced_subset(ids,hashes,maps_limit,tasks_per_map,seed):
    rng=np.random.default_rng(seed);groups=rng.permutation(np.unique(hashes[ids]))
    if maps_limit>0:groups=groups[:maps_limit]
    return np.asarray([int(i) for h in groups for i in rng.permutation(ids[hashes[ids]==h])[:tasks_per_map]],dtype=np.int64)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--data-path',required=True);p.add_argument('--save-dir',required=True)
    p.add_argument('--train-maps',type=int,default=128);p.add_argument('--validation-maps',type=int,default=0);p.add_argument('--tasks-per-map',type=int,default=2)
    p.add_argument('--candidates',type=int,default=4);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--flow-steps',type=int,default=100);p.add_argument('--latent-scale',type=float,default=8.);p.add_argument('--noise-scale',type=float,default=1.)
    p.add_argument('--seed',type=int,default=20261002);p.add_argument('--partition-seed',type=int,default=20260930);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda');base,state=load_model(args.checkpoint,device);base.eval().requires_grad_(False)
    from utils.dataset import PathPlanningDataset
    import utils.dataset as dataset_module,utils.robot_geometry as geometry_module
    ds=PathPlanningDataset(args.data_path,horizon=base.horizon,normalize=True,augment=False,distance_scale=base.distance_scale)
    train,val,hashes=map_disjoint_split(ds.maps,.1,args.partition_seed)
    train=balanced_subset(train,hashes,args.train_maps,args.tasks_per_map,args.seed)
    val=balanced_subset(val,hashes,args.validation_maps,args.tasks_per_map,args.seed+1)
    output=Path(args.save_dir);output.mkdir(parents=True,exist_ok=True)
    manifest=write_run_manifest(output,args,[__file__,HERE/'path_refinement.py',HERE/'scripts/train_p37_ordered_field.py',HERE/'scripts/train_p56_path_chunk_fpo.py',HERE/'scripts/train_p37_flow_sde_ppo.py',HERE/'p37_model.py',HERE/'safe_field_decoder.py',dataset_module.__file__,geometry_module.__file__],{'data':args.data_path,'checkpoint':args.checkpoint})
    generator=torch.Generator(device=device).manual_seed(args.seed)
    start_time=time.perf_counter()
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        for split,ids in [('train',train),('validation',val)]:
            paths,valids,selected,labels=[],[],[],[];feature_maps=[];references=[]
            for begin in range(0,len(ids),args.batch_size):
                tasks=ids[begin:begin+args.batch_size];batch=[ds[int(i)] for i in tasks]
                maps=torch.stack([torch.cat([x['map'],x['distance_field']],0) for x in batch]).to(device)
                starts=torch.stack([x['start'] for x in batch]).to(device);goals=torch.stack([x['goal'] for x in batch]).to(device);dist=torch.stack([x['distance_field'] for x in batch]).to(device)
                fields,path,valid=frozen_base_paths(base,maps,starts,goals,dist,args.candidates,args.flow_steps,args.noise_scale,args.latent_scale,generator)
                scores=candidate_selection_scores(fields,path,maps,valid)
                px=ds.denormalize_coordinates(path.cpu().numpy())
                jobs=[(px[j],ds.maps[int(task)],ds.starts[int(task)],ds.goals[int(task)],ds.geometry.to_json()) for j,task in enumerate(tasks)]
                rows=list(pool.map(exact_label,jobs));labels.extend(rows)
                paths.append(path.cpu().numpy());valids.append(valid.cpu().numpy());selected.append(scores.argmax(1).cpu().numpy())
                feature_maps.extend(signed_distance_features(ds.maps[int(t)],base.distance_scale) for t in tasks)
                references.extend(float(np.linalg.norm(np.diff(ds.paths[int(t)],axis=0),axis=1).sum()) for t in tasks)
                print(json.dumps({'split':split,'finished':begin+len(tasks),'total':len(ids),'seconds':time.perf_counter()-start_time}),flush=True)
            lab=np.asarray(labels,dtype=np.float32)
            np.savez_compressed(output/f'{split}.npz',task_ids=ids,map_hashes=hashes[ids],maps=np.asarray(feature_maps),base_paths=np.concatenate(paths),base_valid=np.concatenate(valids),base_selected=np.concatenate(selected),
                 success=lab[:,:,0].astype(bool),clearance=lab[:,:,1],collision_fraction=lab[:,:,2],reference_length=np.asarray(references,dtype=np.float32),starts=ds.starts[ids],goals=ds.goals[ids],occupancy=ds.maps[ids],geometry_json=ds.geometry.to_json())
    audit={'train_tasks':len(train),'validation_tasks':len(val),'train_unique_maps':len(set(hashes[train])),'validation_unique_maps':len(set(hashes[val])),'map_overlap':len(set(hashes[train])&set(hashes[val])),
      'manifest_sha256':manifest,'source_checkpoint_sha256':file_sha256(args.checkpoint),'seed':args.seed,'partition_seed':args.partition_seed,'flow_steps':args.flow_steps,'latent_scale':args.latent_scale,'noise_scale':args.noise_scale,'candidates':args.candidates}
    (output/'cache_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
