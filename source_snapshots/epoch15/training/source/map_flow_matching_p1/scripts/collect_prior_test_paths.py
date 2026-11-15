"""Generate immutable Flow base candidates for a fresh, strictly held-out test.

Contains exact geometry only as offline labels. Checkpoint, data and source
fingerprints are written before sampling. This does NOT train the generator.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np,torch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.append(str(HERE.parent/'map_flow_matching'))
from scripts.train_p37_flow_sde_ppo import load_model
from scripts.train_p56_path_chunk_fpo import frozen_base_paths
from scripts.cache_refinement_rollouts import worker_init,exact_label
from p37_model import candidate_selection_scores
from path_refinement import signed_distance_features
from rl_audit import quality_metrics,summarize,map_hashes
from run_manifest import write_run_manifest,file_sha256


def sampling_config(checkpoint_config, flow_steps=100, noise_scale=1., latent_scale=None):
    scale=float(checkpoint_config.get('latent_scale',1.) if latent_scale is None else latent_scale)
    if flow_steps<1 or not np.isfinite([noise_scale,scale]).all() or min(noise_scale,scale)<0:
        raise ValueError('positive flow steps and finite nonnegative sampling scales required')
    return {'flow_steps':flow_steps,'noise_scale':float(noise_scale),'latent_scale':scale}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--data-path',required=True);p.add_argument('--training-data-path',required=True);p.add_argument('--save-dir',required=True)
    p.add_argument('--seed',type=int,default=20260921);p.add_argument('--candidates',type=int,default=4);p.add_argument('--workers',type=int,default=8);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--flow-steps',type=int,default=100);p.add_argument('--noise-scale',type=float,default=1.);p.add_argument('--latent-scale',type=float,default=None,help='default: checkpoint latent_scale (1 for legacy checkpoints without it)');args=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda');model,state=load_model(args.checkpoint,device);model.eval().requires_grad_(False)
    sampling=sampling_config(state.get('config',{}),args.flow_steps,args.noise_scale,args.latent_scale)
    from utils.dataset import PathPlanningDataset
    ds=PathPlanningDataset(args.data_path,horizon=model.horizon,normalize=True,augment=False,distance_scale=model.distance_scale)
    th=map_hashes(np.load(args.training_data_path,allow_pickle=False)['maps']);eh=map_hashes(ds.maps);overlap=len(set(th)&set(eh));assert overlap==0
    output=Path(args.save_dir);output.mkdir(parents=True,exist_ok=True)
    manifest=write_run_manifest(output,args,[__file__,HERE/'p37_model.py',HERE/'safe_field_decoder.py',HERE/'scripts/train_p37_flow_sde_ppo.py',HERE/'scripts/train_p56_path_chunk_fpo.py'],{'checkpoint':args.checkpoint,'evaluation_data':args.data_path,'training_data':args.training_data_path})
    rng=torch.Generator(device=device).manual_seed(args.seed+710001);records={key:[] for key in ['paths','success','selected','available','valid']};start=time.perf_counter()
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        for begin in range(0,len(ds),args.batch_size):
            ids=np.arange(begin,min(begin+args.batch_size,len(ds)));batch=[ds[int(i)] for i in ids]
            maps=torch.stack([torch.cat([x['map'],x['distance_field']],0) for x in batch]).to(device);starts=torch.stack([x['start'] for x in batch]).to(device);goals=torch.stack([x['goal'] for x in batch]).to(device);dist=torch.stack([x['distance_field'] for x in batch]).to(device)
            fields,paths,valid=frozen_base_paths(model,maps,starts,goals,dist,args.candidates,sampling['flow_steps'],sampling['noise_scale'],sampling['latent_scale'],rng)
            score=candidate_selection_scores(fields,paths,maps,valid);px=ds.denormalize_coordinates(paths.cpu().numpy())
            jobs=[(px[j],ds.maps[t],ds.starts[t],ds.goals[t],ds.geometry.to_json()) for j,t in enumerate(ids)]
            labels=np.asarray(list(pool.map(exact_label,jobs)))
            records['paths'].append(px);records['success'].append(labels[:,:,0].astype(bool));records['selected'].append(score.argmax(1).cpu().numpy());records['available'].append(torch.isfinite(score).any(1).cpu().numpy());records['valid'].append(valid.cpu().numpy())
            print(json.dumps({'finished':begin+len(ids),'total':len(ds),'seconds':time.perf_counter()-start}),flush=True)
    arrays={key:np.concatenate(value) for key,value in records.items()};quality=quality_metrics(arrays['paths'])
    np.savez_compressed(output/'base.npz',paths=arrays['paths'][None],success=arrays['success'][None],selected=arrays['selected'][None],available=arrays['available'][None],decoder_valid=arrays['valid'],task_ids=np.arange(len(ds)),**quality)
    result={'tasks':len(ds),'unique_maps':len(set(eh)),'generator_checkpoint_sha256':file_sha256(args.checkpoint),'evaluation_data_sha256':file_sha256(args.data_path),'seed':args.seed,
       'sampling_config':sampling,'sampling_rng':{'seed_offset':710001,'stream':'continuous_across_batches','batch_size':args.batch_size,'candidates':args.candidates},'provenance':{'rl_training_map_overlap':0},'manifest_sha256':manifest,
       'metrics':summarize(arrays['success'],quality,arrays['selected'],arrays['available'])}
    (output/'base.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
