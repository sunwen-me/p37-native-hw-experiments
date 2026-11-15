"""Train full-resolution Flow-prior path refinement with exact Gaussian PPO.

Warm-start learns differentiable geometry, then RL uses exact offline footprint
rewards and same-base deterministic control variates. The two phases and their
checkpoints are kept SEPARATE so geometry training is not called an RL gain.
"""
from __future__ import annotations
import argparse,copy,json,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import numpy as np,torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));sys.path.append(str(HERE.parent/'map_flow_matching'))
from path_refinement import (DensePathRefiner,path_metrics,neural_candidate_selection,geometry_auxiliary_loss,
 exact_terminal_reward,joint_offset_logprob,paired_group_advantages,correlated_offset_noise,whiten_offset)
from scripts.cache_refinement_rollouts import worker_init,exact_label
from rl_audit import summarize,quality_metrics,paired_summary
from run_manifest import write_run_manifest


def load_cache(path):
    with np.load(path,allow_pickle=False) as f:return {key:f[key] for key in f.files}


def label_paths(pool,paths_px,cache,ids):
    # Each state contains one base candidate and several residual actions.
    jobs=[(paths_px[j],cache['occupancy'][int(t)],cache['starts'][int(t)],cache['goals'][int(t)],str(cache['geometry_json'])) for j,t in enumerate(ids)]
    return np.asarray(list(pool.map(exact_label,jobs)),dtype=np.float32)


def state_batch(cache,indices,candidates,device):
    return (torch.from_numpy(cache['maps'][indices]).to(device),
            torch.from_numpy(cache['base_paths'][indices,candidates]).to(device),
            torch.from_numpy(cache['reference_length'][indices]).to(device))


def save_checkpoint(path,model,args,phase,update,manifest,validation=None,extra=None):
    torch.save({'variant':'dense_path_refiner_gaussian_rl','model_state_dict':model.state_dict(),
       'config':{'hidden':model.hidden,'max_offset_px':model.max_offset_px,'distance_scale':model.distance_scale,'robot_radius':model.robot_radius,'safety_logit_scale':model.safety_logit_scale,'clearance_calibration_bias':model.clearance_calibration_bias,'representation':'multiscale_dense_v2'},
       'args':vars(args),'phase':phase,'update':update,'run_manifest_sha256':manifest,'validation':validation,
       'cache_audit':json.loads((Path(args.cache_dir)/'cache_audit.json').read_text()),**(extra or {})},path)


@torch.no_grad()
def validate(model,cache,device,pool,batch_size=16):
    records={key:[] for key in ['paths','labels','refined_selected','pooled_selected','reselected','refined_available','pooled_available','base_available']}
    for begin in range(0,len(cache['base_paths']),batch_size):
        ids=np.arange(begin,min(begin+batch_size,len(cache['base_paths'])))
        base=torch.from_numpy(cache['base_paths'][ids]).to(device)
        maps=torch.from_numpy(cache['maps'][ids]).to(device);b,k,h,_=base.shape
        flatmaps=maps[:,None].expand(-1,k,-1,-1,-1).flatten(0,1)
        refined=model(flatmaps,base.flatten(0,1)).reshape_as(base)
        paths=refined.cpu().numpy();px=(paths+1)*31.5
        jobs=[(px[j],cache['occupancy'][int(t)],cache['starts'][int(t)],cache['goals'][int(t)],str(cache['geometry_json'])) for j,t in enumerate(ids)]
        labels=np.asarray(list(pool.map(exact_label,jobs)),dtype=np.float32)
        si,_,slog=neural_candidate_selection(model,maps,refined)
        ri,_,rlog=neural_candidate_selection(model,maps,base)
        pi,_,plog=neural_candidate_selection(model,maps,torch.cat([base,refined],1),torch.from_numpy(cache['base_selected'][ids]).to(device),torch.from_numpy(cache['base_valid'][ids].any(1)).to(device),k)
        row=torch.arange(len(ids),device=device)
        records['refined_available'].append((slog[row,si]>=0).cpu().numpy());records['base_available'].append((rlog[row,ri]>=0).cpu().numpy());records['pooled_available'].append(((plog[row,pi]>=0)|torch.from_numpy(cache['base_valid'][ids].any(1)).to(device)).cpu().numpy())
        records['paths'].append(paths);records['labels'].append(labels);records['refined_selected'].append(si.cpu().numpy());records['reselected'].append(ri.cpu().numpy());records['pooled_selected'].append(pi.cpu().numpy())
    allpaths=np.concatenate(records['paths']);labels=np.concatenate(records['labels']);success=labels[:,:,0].astype(bool)
    bp=(cache['base_paths']+1)*31.5;rp=(allpaths+1)*31.5
    bq=quality_metrics(bp);rq=quality_metrics(rp);poolq={key:np.concatenate([bq[key],rq[key]],1) for key in bq}
    original=summarize(cache['success'],bq,cache['base_selected'],cache['base_valid'].any(1))
    reselected=summarize(cache['success'],bq,np.concatenate(records['reselected']),np.concatenate(records['base_available']))
    only=summarize(success,rq,np.concatenate(records['refined_selected']),np.concatenate(records['refined_available']))
    pooled_success=np.concatenate([cache['success'],success],1)
    pooled=summarize(pooled_success,poolq,np.concatenate(records['pooled_selected']),np.concatenate(records['pooled_available']))
    compare=paired_summary(cache['success'],success,bq,rq,cache['base_selected'],np.concatenate(records['refined_selected']),cache['base_valid'].any(1),np.concatenate(records['refined_available']),0)
    # Pooled action counts differ, so compare SELECTED outcomes directly rather
    # than pretend the original and pooled candidate arrays have same shape.
    rows=np.arange(len(success));bi=cache['base_selected'];pi=np.concatenate(records['pooled_selected'])
    bs=cache['success'][rows,bi]&cache['base_valid'].any(1);ps=pooled_success[rows,pi]&np.concatenate(records['pooled_available']);both=bs&ps
    changes={key:float((poolq[key][rows,pi]-bq[key][rows,bi])[both].mean()) if both.any() else None for key in bq}
    return {'original':original,'base_reselected':reselected,'refined_only':only,'pooled':pooled,
      'paired_refined_vs_original':compare,'pooled_common_selected_quality_delta':changes,
      'rescued_selected_tasks':int((~bs&ps).sum()),'lost_selected_tasks':int((bs&~ps).sum()),
      'flow_draws_per_task':cache['base_paths'].shape[1],'pooled_path_options_per_task':2*cache['base_paths'].shape[1]},allpaths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',required=True);p.add_argument('--save-dir',required=True)
    p.add_argument('--teacher-path');p.add_argument('--init-checkpoint');p.add_argument('--warm-updates',type=int,default=500);p.add_argument('--rl-updates',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--rl-batch-size',type=int,default=8);p.add_argument('--draws',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--rl-lr',type=float,default=1e-5);p.add_argument('--std-px',type=float,default=.08);p.add_argument('--noise-correlation',type=float,default=.9)
    p.add_argument('--clip-eps',type=float,default=.15);p.add_argument('--max-kl',type=float,default=.03);p.add_argument('--ppo-epochs',type=int,default=2)
    p.add_argument('--auxiliary-weight',type=float,default=.05);p.add_argument('--validation-interval',type=int,default=100)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--seed',type=int,default=20261002);p.add_argument('--overfit-states',type=int,default=0)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--max-offset-px',type=float,default=3.)
    p.add_argument('--feature-ablation',choices=('none','no_local_map_edt','no_derivatives','no_endpoint_progress'),default='none')
    args=p.parse_args()
    if args.draws<2 or args.draws%2:p.error('even draws >=2 required for antithetic SAME-base rollout')
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device('cuda')
    train=load_cache(Path(args.cache_dir)/'train.npz');val=load_cache(Path(args.cache_dir)/'validation.npz')
    assert not set(train['map_hashes'])&set(val['map_hashes'])
    model=DensePathRefiner(args.hidden,args.max_offset_px,feature_ablation=args.feature_ablation).to(device)
    if args.init_checkpoint:
        initial=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(initial['model_state_dict'],strict=True)
        if initial['cache_audit']['source_checkpoint_sha256']!=json.loads((Path(args.cache_dir)/'cache_audit.json').read_text())['source_checkpoint_sha256']:
            raise ValueError('init/checkpoint prior mismatch')
    actor_params=list(model.actor.parameters());safety_params=list(model.safety.parameters())
    opt=torch.optim.Adam(actor_params,lr=args.lr);safeopt=torch.optim.Adam(safety_params,lr=3e-4)
    rng=np.random.default_rng(args.seed);generator=torch.Generator(device=device).manual_seed(args.seed+7001)
    output=Path(args.save_dir);output.mkdir(parents=True,exist_ok=True)
    import utils.robot_geometry as geometry_module
    manifest=write_run_manifest(output,args,[__file__,HERE/'path_refinement.py',HERE/'scripts/cache_refinement_rollouts.py',HERE/'rl_audit.py',HERE/'run_manifest.py',geometry_module.__file__],
        {'train_cache':Path(args.cache_dir)/'train.npz','validation_cache':Path(args.cache_dir)/'validation.npz','cache_manifest':Path(args.cache_dir)/'run_manifest.json',**({'teacher':args.teacher_path} if args.teacher_path else {}),**({'init_checkpoint':args.init_checkpoint} if args.init_checkpoint else {})})
    teacher=load_cache(args.teacher_path) if args.teacher_path else None
    if teacher is not None and teacher['targets'].shape!=train['base_paths'].shape:raise ValueError('teacher/cache shape mismatch')
    successes=train['success'].reshape(-1);state_ids=np.arange(len(successes))
    if args.overfit_states>0:
        good=state_ids[successes];bad=state_ids[~successes]
        state_ids=np.concatenate([rng.choice(good,args.overfit_states//2,replace=False),rng.choice(bad,args.overfit_states-args.overfit_states//2,replace=False)])
    good=state_ids[successes[state_ids]];bad=state_ids[~successes[state_ids]]
    def sample_states(count):
        values=np.concatenate([rng.choice(good,count-count//3,replace=True),rng.choice(bad,count//3,replace=True)]) if len(bad) else rng.choice(good,count,replace=True)
        rng.shuffle(values);return values//train['base_paths'].shape[1],values%train['base_paths'].shape[1]
    baseline_quality=quality_metrics((val['base_paths']+1)*31.5)
    baseline=summarize(val['success'],baseline_quality,val['base_selected'],val['base_valid'].any(1))
    (output/'baseline_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    best_key=None;history=[];start=time.perf_counter();dual=5.
    def consider(phase,update,validation):
        nonlocal best_key
        m=validation['pooled'];change=validation['pooled_common_selected_quality_delta']
        # Not a deployment checkpoint if validation safety fell below original.
        eligible=(m['selected_success_rate']>=baseline['selected_success_rate'] and validation['lost_selected_tasks']==0
                  and change['length'] is not None and change['smoothness'] is not None
                  and change['length']<=0 and change['smoothness']<=0)
        qualitykey=-(float(change['length'] or 0)/50+float(change['smoothness'] or 0)/.1)
        key=(m['selected_success_rate'],qualitykey)
        if eligible and (best_key is None or key>best_key):
            best_key=key;save_checkpoint(output/'best_deployment.pth',model,args,phase,update,manifest,validation,{'dual':dual})
        save_checkpoint(output/f'{phase}_{update:05d}.pth',model,args,phase,update,manifest,validation,{'dual':dual})
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        for update in range(1,args.warm_updates+1):
            ids,candidates=sample_states(args.batch_size);maps,base,ref=state_batch(train,ids,candidates,device)
            auxiliary,predicted=geometry_auxiliary_loss(model,maps,base,ref)
            if teacher is not None:
                target=torch.from_numpy(teacher['targets'][ids,candidates]).to(device)
                weights=torch.from_numpy(teacher['weights'][ids,candidates]).to(device)
                # Explicit geometric BC warm-start; not counted as an RL gain.
                loss=(weights[:,None,None]*((predicted-target)*31.5).square()).mean()+.005*auxiliary
            else:
                loss=auxiliary
            opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(actor_params,1));opt.step()
            logits=model.success_logit(maps,base.detach());labels=torch.from_numpy(train['success'][ids,candidates].astype(np.float32)).to(device)
            sl=F.binary_cross_entropy_with_logits(logits,labels);safeopt.zero_grad();sl.backward();safeopt.step()
            row={'phase':'warm','update':update,'geometry_loss':float(loss.detach()),'actor_grad_norm':grad,'safety_loss':float(sl.detach()),'seconds':time.perf_counter()-start}
            if update%args.validation_interval==0 or update==args.warm_updates:
                model.eval();validation,paths=validate(model,val,device,pool);row['validation']=validation;consider('warm',update,validation);model.train()
            if update%20==0 or 'validation' in row:print(json.dumps(row),flush=True)
            history.append(row)
            (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if args.warm_updates==0:
            model.eval();validation,_=validate(model,val,device,pool);consider('initial',0,validation);model.train()
        save_checkpoint(output/'warm_final.pth',model,args,'warm',args.warm_updates,manifest,extra={'dual':dual})
        for group in opt.param_groups:group['lr']=args.rl_lr
        for update in range(1,args.rl_updates+1):
            ids,candidates=sample_states(args.rl_batch_size);maps,base,ref=state_batch(train,ids,candidates,device)
            with torch.no_grad():
                oldmean=model.mean_offset(maps,base)
                eps=correlated_offset_noise((len(base),args.draws//2,*oldmean.shape[1:]),generator,device,args.noise_correlation)
                eps=torch.cat([eps,-eps],1)
                actions=oldmean[:,None]+args.std_px*eps;actions[:,:,[0,-1]]=0
                oldlog=joint_offset_logprob(actions,oldmean[:,None],args.std_px,args.noise_correlation)
                alloff=torch.cat([oldmean[:,None],actions],1)
                paths=model.apply_offset(base[:,None].expand_as(alloff),alloff)
                px=((paths+1)*31.5).cpu().numpy()
            metrics=label_paths(pool,px,train,ids)
            exact_success=torch.from_numpy(metrics[:,:,0].astype(bool)).to(device)
            exact_clearance=torch.from_numpy(metrics[:,:,1]).to(device)
            quality=path_metrics(paths.flatten(0,1));q={key:value.reshape(len(base),args.draws+1) for key,value in quality.items()}
            reward=exact_terminal_reward(exact_success,q['length'],q['smoothness'],exact_clearance,ref[:,None])
            cost=(~exact_success).float()
            advantage=paired_group_advantages(reward[:,1:]-dual*cost[:,1:],reward[:,0]-dual*cost[:,0])
            losses=[];kl_rejections=0
            for epoch in range(args.ppo_epochs):
                mean=model.mean_offset(maps,base);newlog=joint_offset_logprob(actions,mean[:,None],args.std_px,args.noise_correlation)
                logratio=(newlog-oldlog).clamp(-20,20);ratio=logratio.exp()
                pg=-torch.minimum(ratio*advantage,ratio.clamp(1-args.clip_eps,1+args.clip_eps)*advantage).mean()
                aux,_=geometry_auxiliary_loss(model,maps,base,ref)
                pggrad=torch.autograd.grad(pg,actor_params,retain_graph=True)
                auxgrad=torch.autograd.grad(aux,actor_params,retain_graph=True)
                pgnorm=float(torch.stack([g.square().sum() for g in pggrad]).sum().sqrt())
                auxnorm=float(torch.stack([g.square().sum() for g in auxgrad]).sum().sqrt())
                loss=pg+args.auxiliary_weight*aux
                before={key:value.detach().clone() for key,value in model.actor.state_dict().items()};optbefore=copy.deepcopy(opt.state_dict())
                opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(actor_params,1));opt.step()
                with torch.no_grad():
                    newmean=model.mean_offset(maps,base)
                    kl=.5*whiten_offset(newmean-oldmean,args.std_px,args.noise_correlation).square().sum((1,2)).mean()
                if float(kl)>args.max_kl:
                    model.actor.load_state_dict(before);opt.load_state_dict(optbefore)
                    for group in opt.param_groups:group['lr']*=.5
                    kl_rejections+=1;break
                losses.append(float(loss.detach()))
            # Train neural safety on FINAL paths, not the stale decoder field.
            m=maps[:,None].expand(-1,args.draws+1,-1,-1,-1).flatten(0,1)
            logits=model.success_logit(m,paths.flatten(0,1).detach())
            safety_loss=F.binary_cross_entropy_with_logits(logits,exact_success.flatten().float())
            safeopt.zero_grad();safety_loss.backward();safeopt.step()
            violation=float(cost[:,1:].mean()-cost[:,0].mean());dual=float(np.clip(dual+2*violation,0,50))
            row={'phase':'rl','update':update,'joint_pg_loss':float(pg.detach()),'geometry_loss':float(aux.detach()),'pg_gradient_norm':pgnorm,
              'auxiliary_gradient_norm':auxnorm,'weighted_auxiliary_over_pg':args.auxiliary_weight*auxnorm/max(pgnorm,1e-12),'actor_grad_norm':grad,
              'actual_joint_gaussian_kl':float(kl),'kl_rejected_steps':kl_rejections,'lr':opt.param_groups[0]['lr'],
              'dual':dual,'sample_success':float(exact_success[:,1:].float().mean()),'deterministic_success':float(exact_success[:,0].float().mean()),
              'paired_reward_delta':float((reward[:,1:]-reward[:,0,None]).mean()),'seconds':time.perf_counter()-start}
            if update%args.validation_interval==0 or update==args.rl_updates:
                model.eval();validation,paths=validate(model,val,device,pool);row['validation']=validation;consider('rl',update,validation);model.train()
            history.append(row);print(json.dumps(row),flush=True);(output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        save_checkpoint(output/'last.pth',model,args,'rl',args.rl_updates,manifest,extra={'dual':dual})
    print(json.dumps({'best_key':best_key,'seconds':time.perf_counter()-start,'validation_only':True,'deployment_checkpoint_available':(output/'best_deployment.pth').is_file()}),flush=True)
if __name__=='__main__':main()
