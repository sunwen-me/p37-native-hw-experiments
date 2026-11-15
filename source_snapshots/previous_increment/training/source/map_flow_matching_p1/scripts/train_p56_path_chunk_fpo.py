"""P56: low-dimensional path-chunk FPO adapter around a P53 Flow prior.

The frozen P53 generator supplies a base path.  FPO controls an 8-anchor
(16-dimensional) residual path chunk, so CFM ratios and terminal geometry
rewards live at the same action scale instead of a 8192-pixel dense field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p10_model import _point_masks  # noqa: E402
from scripts.train_p37_flow_sde_ppo import decode_fields, load_model, terminal_rewards  # noqa: E402
from scripts.train_p37_flow_sde_ppo import compute_task_advantages  # noqa: E402
from scripts.train_p51_constrained_flow import terminal_costs  # noqa: E402
from p37_model import candidate_selection_scores  # noqa: E402


class PathChunkFlowPolicy(nn.Module):
    """Conditional Flow policy over low-dimensional anchor residuals."""

    def __init__(self, action_dim: int = 16, hidden_dim: int = 128, flow_steps: int = 10):
        super().__init__()
        self.action_dim = int(action_dim)
        self.flow_steps = int(flow_steps)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 4 * 4, hidden_dim * 2), nn.SiLU(),
        )
        self.base_context = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim * 2), nn.SiLU(),
        )
        self.time = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.vector = nn.Sequential(
            nn.Linear(self.action_dim + hidden_dim * 2, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, self.action_dim),
        )
        nn.init.zeros_(self.vector[-1].weight)
        nn.init.zeros_(self.vector[-1].bias)

    def context_features(self, maps, starts, goals, base_path=None):
        size = maps.shape[-1]
        x = torch.cat([maps, _point_masks(starts, size), _point_masks(goals, size)], dim=1)
        context = self.context(self.encoder(x))
        if base_path is not None:
            anchor_index = torch.linspace(0, base_path.shape[-2] - 1, self.action_dim // 2, device=base_path.device).round().long()
            if base_path.ndim == 3:
                base_anchor = base_path[:, anchor_index].reshape(len(base_path), -1)
                context = context + self.base_context(base_anchor)
            elif base_path.ndim == 4:
                batch, candidates = base_path.shape[:2]
                base_anchor = base_path[:, :, anchor_index].reshape(batch * candidates, -1)
                base_context = self.base_context(base_anchor).reshape(batch, candidates, -1)
                context = context[:, None, :].expand(-1, candidates, -1) + base_context
            else:
                raise ValueError("base_path must be [B,T,2] or [B,K,T,2]")
        return context

    def vector_field(self, context, x, t):
        return self._vector(context, x, t)

    def _vector(self, context, x, t):
        # Inject context through a separate additive projection so the action
        # vector field remains low-dimensional but map-conditioned.
        return self.vector(torch.cat([x, context + self.time(t[:, None])], dim=-1))

    @torch.no_grad()
    def sample(self, context, generator, stochastic=True, noise_scale=1.0):
        batch = len(context)
        # Training explores with fresh source noise. Zero-source is a
        # deterministic evaluation variant, NOT necessarily a zero residual;
        # validation sampling is explicitly controlled by its CLI flag.
        x = torch.randn(batch, self.action_dim, device=context.device,
                        dtype=context.dtype, generator=generator) * float(noise_scale) if stochastic else torch.zeros(
                            batch, self.action_dim, device=context.device, dtype=context.dtype
                        )
        dt = 1.0 / float(self.flow_steps)
        for step in range(self.flow_steps):
            t = torch.full((batch,), step * dt, device=context.device, dtype=context.dtype)
            x = x + dt * self._vector(context, x, t)
        return x

    def cfm_loss(self, context, action, eps, t):
        xt = (1.0 - t) * eps + t * action
        target = action - eps
        pred = self._vector(context, xt, t.squeeze(-1) if t.ndim > 1 else t)
        return F.smooth_l1_loss(pred, target, reduction="none", beta=1.0).mean(dim=-1)


def cfm_source_noise(action, mc_samples, noise_scale, generator):
    """Use the SAME source distribution as policy.sample for CFM pairs."""
    if mc_samples < 1 or noise_scale < 0:
        raise ValueError("mc_samples must be positive and noise_scale nonnegative")
    return torch.randn(action.numel() // action.shape[-1] * mc_samples,
                       action.shape[-1], device=action.device, dtype=action.dtype,
                       generator=generator) * float(noise_scale)


def paired_base_advantage(reward, base_reward):
    """State-dependent control variate, detached and scaled, NOT task ranking.

    No batch mean subtraction: when every action is worse than the zero-action
    base, all advantages should be negative. Base reward is independent of the
    sampled residual. This does not solve the low-rank action representation.
    """
    delta = (reward - base_reward).detach()
    return delta / delta.square().mean().sqrt().clamp_min(1e-4)


def map_disjoint_ids(dataset, validation_tasks, seed):
    if validation_tasks < 1:
        raise ValueError('validation_tasks must be positive')
    ids = np.arange(len(dataset), dtype=np.int64)
    hashes = np.asarray([
        hashlib.sha256(np.ascontiguousarray(dataset.maps[int(i)]).tobytes()).hexdigest()
        for i in ids
    ])
    rng = np.random.default_rng(seed); selected = []; count = 0
    for value in rng.permutation(np.unique(hashes)):
        selected.append(value); count += int(np.sum(hashes == value))
        if count >= min(validation_tasks, len(ids)): break
    val = np.flatnonzero(np.isin(hashes, selected)).astype(np.int64)
    train=np.setdiff1d(ids,val,assume_unique=True)
    if not len(train) or not len(val):
        raise ValueError('map-disjoint partition requires nonempty train and validation sets; reduce validation_tasks or add maps')
    return train,val


@torch.no_grad()
def frozen_base_paths(model, maps, starts, goals, distance, candidates, steps, noise_scale, latent_scale, generator):
    batch = len(starts)
    latent = torch.randn(batch, candidates, model.latent_dim, device=maps.device,
                          dtype=maps.dtype, generator=generator)
    flat_maps = maps[:, None].expand(-1, candidates, -1, -1, -1).reshape(batch*candidates, *maps.shape[1:])
    flat_start = starts[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    flat_goal = goals[:, None].expand(-1, candidates, -1).reshape(-1, 2)
    source = noise_scale * torch.randn(batch*candidates, 2, model.map_size, model.map_size,
                                       device=maps.device, dtype=maps.dtype, generator=generator)
    model.vector_field.latent_scale = float(latent_scale)
    fields = model._integrate(flat_maps, flat_start, flat_goal, latent.reshape(-1, model.latent_dim),
                               steps, noise_scale, source=source, sde_noise=0.0,
                               generator=generator).reshape(batch, candidates, 2, model.map_size, model.map_size)
    paths, valid = decode_fields(model, fields, distance, starts, goals,
                                 model.robot_radius_grid, model.distance_scale,
                                 model.clearance_temperature, model.clearance_offset,
                                 model.decode_clearance_power, model.decode_continuity_beta)
    return fields, paths, valid


def anchors_to_path(base, residual, residual_scale):
    """Add an interpolated residual to the full base path; zero is identity."""
    batch, candidates, horizon, _ = base.shape
    anchors = residual.shape[-1] // 2
    if anchors < 2:
        raise ValueError("residual action needs at least two anchors")
    residual = residual.reshape(batch, candidates, anchors, 2) * float(residual_scale)
    residual = residual.clone(); residual[:, :, 0] = 0.0; residual[:, :, -1] = 0.0
    query = torch.linspace(0.0, float(anchors - 1), horizon, device=base.device, dtype=base.dtype)
    left = query.floor().long().clamp_max(anchors - 1); right = query.ceil().long().clamp_max(anchors - 1)
    alpha = (query - left.to(query.dtype))[None, None, :, None]
    residual_curve = residual[:, :, left] * (1.0 - alpha) + residual[:, :, right] * alpha
    path = base + residual_curve
    path[:, :, 0] = base[:, :, 0]; path[:, :, -1] = base[:, :, -1]
    return path


def path_safety_shaping(paths, distance, distance_scale, clearance_offset,
                        robot_radius, margin, segment_samples=16):
    """Swept-segment distance-field barrier, including connector geometry."""
    batch, candidates, horizon, _ = paths.shape
    flat = paths.reshape(batch * candidates, horizon, 2)
    dist = distance[:, None].expand(-1, candidates, -1, -1, -1).reshape(
        batch * candidates, 1, distance.shape[-2], distance.shape[-1]
    )
    start = flat[:, :-1, None, :]
    delta = flat[:, 1:, None, :] - start
    fractions = torch.linspace(
        0.0, 1.0, int(segment_samples), device=paths.device, dtype=paths.dtype,
    ).view(1, 1, int(segment_samples), 1)
    samples = (start + fractions * delta).reshape(flat.shape[0], -1, 1, 2)
    clearance = torch.nn.functional.grid_sample(
        dist, samples, mode="bilinear", padding_mode="border", align_corners=True,
    ).squeeze(1).squeeze(-1)
    clearance = clearance.min(dim=1).values * float(distance_scale) - float(clearance_offset)
    deficit = (float(robot_radius) + float(margin) - clearance).clamp_min(0.0)
    return deficit.reshape(batch, candidates), clearance.reshape(batch, candidates)


def make_obs(maps, starts, goals):
    size = maps.shape[-1]
    return torch.cat([maps, _point_masks(starts, size), _point_masks(goals, size)], dim=1)


@torch.no_grad()
def validate(policy, base_model, dataset, task_ids, args, device, generator, latent_scale):
    """Paired base and residual validation with actual availability gating.

    ALL quality comparisons use jointly successful tasks, selector results
    use final paths, and action draws cannot change the next base draw.
    """
    from utils.robot_geometry import evaluate_robot_path
    from rl_audit import quality_metrics, summarize, paired_summary
    old_training=policy.training
    policy.eval();base_model.eval()
    action_rng=torch.Generator(device=device).manual_seed(args.seed+810001)
    record=[[],[]];success=[[],[]];selected=[[],[]];available=[[],[]]
    for begin in range(0,len(task_ids),args.validation_batch_size):
        ids=task_ids[begin:begin+args.validation_batch_size]
        batch=[dataset[int(i)] for i in ids]
        maps=torch.stack([torch.cat([x['map'],x['distance_field']],0) for x in batch]).to(device)
        starts=torch.stack([x['start'] for x in batch]).to(device);goals=torch.stack([x['goal'] for x in batch]).to(device)
        distance=torch.stack([x['distance_field'] for x in batch]).to(device)
        fields,base,valid=frozen_base_paths(base_model,maps,starts,goals,distance,args.validation_candidates,
                 args.flow_steps,args.noise_scale,latent_scale,generator)
        context=policy.context_features(maps,starts,goals,base).flatten(0,1)
        action=policy.sample(context,action_rng,stochastic=args.validation_stochastic,
                  noise_scale=args.action_noise_scale).reshape(len(ids),args.validation_candidates,-1)
        paths=anchors_to_path(base,action,args.residual_scale)
        for version,value in enumerate([base,paths]):
            scores=candidate_selection_scores(fields,value,maps,valid)
            si=scores.argmax(1).cpu().numpy();avail=torch.isfinite(scores).any(1).cpu().numpy()
            px=dataset.denormalize_coordinates(value.cpu().numpy())
            results=[[bool(evaluate_robot_path(px[j,k],dataset.maps[int(t)],dataset.starts[int(t)],
                 dataset.goals[int(t)],dataset.geometry,endpoint_tolerance=1e-5)['success'])
                 for k in range(args.validation_candidates)] for j,t in enumerate(ids)]
            record[version].append(px);success[version].append(results);selected[version].append(si);available[version].append(avail)
    summaries=[];arrays=[]
    for version in range(2):
        value=np.concatenate(record[version]);ok=np.concatenate(success[version]);si=np.concatenate(selected[version]);av=np.concatenate(available[version])
        q=quality_metrics(value);summaries.append(summarize(ok,q,si,av));arrays.append((ok,q,si,av))
    bo,bq,bi,ba=arrays[0];po,pq,pi,pa=arrays[1]
    paired=paired_summary(bo,po,bq,pq,bi,pi,ba,pa,args.seed)
    policy.train(old_training);base_model.eval()
    m=summaries[1]
    return {'candidate_success_rate':m['candidate_success_rate'],
       'oracle_best_of_k_success_rate':m['oracle_best_of_k_success_rate'],
       'selected_success_rate':m['selected_success_rate'],
       'selected_success_rate_proxy':m['selected_success_rate'],
       'candidate_mean_length':m['all_candidate_mean_length'],
       'base':summaries[0],'residual':m,'paired':paired}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint',required=True); parser.add_argument('--data-path',required=True); parser.add_argument('--save-dir',required=True); parser.add_argument('--updates',type=int,default=5); parser.add_argument('--batch-size',type=int,default=8); parser.add_argument('--train-candidates',type=int,default=1); parser.add_argument('--validation-candidates',type=int,default=8); parser.add_argument('--flow-steps',type=int,default=100); parser.add_argument('--noise-scale',type=float,default=1.0); parser.add_argument('--latent-scale',type=float,default=None); parser.add_argument('--anchors',type=int,default=8); parser.add_argument('--residual-scale',type=float,default=0.05); parser.add_argument('--action-noise-scale',type=float,default=0.2); parser.add_argument('--cfm-noise-scale',type=float,default=None,help='default matches action source; 1.0 reproduces historical mismatch'); parser.add_argument('--advantage-mode',choices=['paired-base','legacy-task'],default='paired-base'); parser.add_argument('--validation-stochastic', action=argparse.BooleanOptionalAction, default=True); parser.add_argument('--residual-penalty-weight',type=float,default=1.0); parser.add_argument('--clearance-barrier-weight',type=float,default=4.0); parser.add_argument('--clearance-margin',type=float,default=0.25); parser.add_argument('--policy-flow-steps',type=int,default=10); parser.add_argument('--mc-samples',type=int,default=8); parser.add_argument('--lr',type=float,default=3e-4); parser.add_argument('--clip-eps',type=float,default=0.05); parser.add_argument('--ppo-epochs',type=int,default=2); parser.add_argument('--validation-interval',type=int,default=5); parser.add_argument('--checkpoint-interval',type=int,default=5); parser.add_argument('--validation-tasks',type=int,default=32); parser.add_argument('--validation-batch-size',type=int,default=8); parser.add_argument('--length-weight',type=float,default=0.03); parser.add_argument('--smooth-weight',type=float,default=0.05); parser.add_argument('--clearance-weight',type=float,default=0.25); parser.add_argument('--seed',type=int,default=20260930); parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.anchors < 3:
        parser.error("at least three anchors are needed for an interior residual action")
    run_start=time.perf_counter(); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device(args.device)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False); config=state.get('config',{}); latent_scale=float(config.get('latent_scale',1.0) or 1.0) if args.latent_scale is None else args.latent_scale
    from utils.dataset import PathPlanningDataset
    dataset=PathPlanningDataset(args.data_path,horizon=128,normalize=True,augment=False,distance_scale=float(config.get('distance_scale',8.0)))
    train_ids,val_ids=map_disjoint_ids(dataset,args.validation_tasks,args.seed); base,_=load_model(args.checkpoint,device); base.eval(); base.requires_grad_(False)
    policy=PathChunkFlowPolicy(action_dim=args.anchors*2,flow_steps=args.policy_flow_steps).to(device); optimizer=torch.optim.AdamW(policy.parameters(),lr=args.lr); rng=np.random.default_rng(args.seed); generator=torch.Generator(device=device).manual_seed(args.seed+1009); output=Path(args.save_dir); output.mkdir(parents=True,exist_ok=True); history=[]
    from run_manifest import write_run_manifest
    import utils.dataset as dataset_module
    import utils.robot_geometry as geometry_module
    manifest_sha = write_run_manifest(output, args,
        [__file__, HERE / 'p37_model.py', HERE / 'safe_field_decoder.py',
         HERE / 'scripts/train_p37_flow_sde_ppo.py', HERE / 'scripts/train_p51_constrained_flow.py',
         HERE / 'run_manifest.py', dataset_module.__file__, geometry_module.__file__],
        {'data': args.data_path, 'initial_checkpoint': args.checkpoint})
    baseline_validation=validate(policy,base,dataset,val_ids,args,device,torch.Generator(device=device).manual_seed(args.seed+710001),latent_scale)
    (output/"baseline_validation.json").write_text(json.dumps(baseline_validation,indent=2)+"\n")
    best_key=(baseline_validation['selected_success_rate'],0.0); best_update=0
    torch.save({'variant':'p56_path_chunk_fpo','run_manifest_sha256':manifest_sha,
                'policy_state_dict':policy.state_dict(),'generator_checkpoint':str(Path(args.checkpoint).resolve()),
                'args':vars(args),'update':0,'validation':baseline_validation},output/'initial_p56_path_chunk.pth')
    for update in range(1,args.updates+1):
        chosen=rng.choice(train_ids,size=args.batch_size,replace=True); batch=[dataset[int(i)] for i in chosen]; maps=torch.stack([torch.cat([x['map'],x['distance_field']],0) for x in batch]).to(device); starts=torch.stack([x['start'] for x in batch]).to(device); goals=torch.stack([x['goal'] for x in batch]).to(device); distance=torch.stack([x['distance_field'] for x in batch]).to(device)
        with torch.no_grad(): _, base_paths,_=frozen_base_paths(base,maps,starts,goals,distance,args.train_candidates,args.flow_steps,args.noise_scale,latent_scale,generator)
        obs=make_obs(maps,starts,goals); context=policy.context_features(obs[:,:2],starts,goals,base_paths)
        context_k = context.reshape(-1, context.shape[-1]) if context.ndim == 3 else context[:, None].expand(-1, args.train_candidates, -1).reshape(-1, context.shape[-1]); action=policy.sample(context_k,generator,stochastic=True,noise_scale=args.action_noise_scale).reshape(args.batch_size,args.train_candidates,-1); paths=anchors_to_path(base_paths,action,args.residual_scale)
        reward,_=terminal_rewards(paths,torch.ones(args.batch_size,args.train_candidates,dtype=torch.bool,device=device),dataset,chosen,args.length_weight,args.smooth_weight,args.clearance_weight,3.0,1.0,-12.0); terminal_cost,_=terminal_costs(paths,torch.ones(args.batch_size,args.train_candidates,dtype=torch.bool,device=device),dataset,chosen); reward=reward.to(device); terminal_cost=terminal_cost.to(device); reward=torch.where(terminal_cost>1e-6,torch.full_like(reward,-30.0)-10.0*terminal_cost,reward); barrier,_=path_safety_shaping(paths,distance,base.distance_scale,base.clearance_offset,base.robot_radius_grid,args.clearance_margin); residual_cost=action[...,2:-2].square().mean(dim=-1); reward = reward - args.clearance_barrier_weight * barrier.to(device) - args.residual_penalty_weight * residual_cost
        if args.advantage_mode == 'paired-base':
            valid_for_reward = torch.ones_like(reward, dtype=torch.bool)
            base_reward, _ = terminal_rewards(base_paths, valid_for_reward, dataset, chosen,
                                              args.length_weight, args.smooth_weight,
                                              args.clearance_weight, 3.0, 1.0, -12.0)
            base_cost, _ = terminal_costs(base_paths, valid_for_reward, dataset, chosen)
            base_reward, base_cost = base_reward.to(device), base_cost.to(device)
            base_reward = torch.where(base_cost > 1e-6,
                                      torch.full_like(base_reward, -30.0) - 10.0 * base_cost,
                                      base_reward)
            base_barrier, _ = path_safety_shaping(base_paths, distance, base.distance_scale,
                                                 base.clearance_offset, base.robot_radius_grid,
                                                 args.clearance_margin)
            base_reward = base_reward - args.clearance_barrier_weight * base_barrier
            adv = paired_base_advantage(reward, base_reward)
        else:
            base_reward = None
            adv = compute_task_advantages(reward, task_weight=1.0, candidate_weight=0.25).reshape_as(reward).detach()
        cfm_noise_scale = args.action_noise_scale if args.cfm_noise_scale is None else args.cfm_noise_scale
        eps=cfm_source_noise(action,args.mc_samples,cfm_noise_scale,generator); t=torch.rand(len(eps),1,device=device,generator=generator); x1=action.reshape(-1,args.anchors*2)[:,None].expand(-1,args.mc_samples,-1).reshape(-1,args.anchors*2); ctx=context_k[:,None].expand(-1,args.mc_samples,-1).reshape(-1,context.shape[-1]); old=policy.cfm_loss(ctx,x1,eps,t).reshape(args.batch_size,args.train_candidates,args.mc_samples).detach(); loss_val=torch.zeros((),device=device)
        for _ in range(args.ppo_epochs):
            current_context = policy.context_features(obs[:,:2], starts, goals, base_paths)
            current_context = current_context.reshape(-1, current_context.shape[-1]) if current_context.ndim == 3 else current_context[:,None].expand(-1,args.train_candidates,-1).reshape(-1,current_context.shape[-1])
            current_context = current_context[:,None].expand(-1,args.mc_samples,-1).reshape(-1,current_context.shape[-1])
            new=policy.cfm_loss(current_context,x1,eps,t).reshape(args.batch_size,args.train_candidates,args.mc_samples); log_ratio=(old-new).clamp(-5,5); ratio=log_ratio.exp(); adv_mc=adv.unsqueeze(-1).expand(-1,-1,args.mc_samples); positive=torch.minimum(ratio*adv_mc,ratio.clamp(1-args.clip_eps,1+args.clip_eps)*adv_mc); negative=ratio*adv_mc-(adv_mc.abs()/(2*args.clip_eps))*(ratio-1).square(); loss_val=-torch.where(adv_mc>0,positive,negative).mean(); optimizer.zero_grad(set_to_none=True); loss_val.backward(); grad=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),1).detach().cpu()); optimizer.step()
        row={'update':update,'fpo_loss':float(loss_val.detach().cpu()),'gradient_norm':grad,'ratio_mean':float(ratio.mean().detach().cpu()),'train_reward_mean':float(reward.mean().detach().cpu()),'paired_reward_delta_mean':None if base_reward is None else float((reward-base_reward).mean().detach().cpu()),'cfm_noise_scale':float(cfm_noise_scale),'wall_seconds':float(time.perf_counter()-run_start)}
        if update==1 or update%args.validation_interval==0:
            validation=validate(policy,base,dataset,val_ids,args,device,torch.Generator(device=device).manual_seed(args.seed+710001),latent_scale)
            row['validation']=validation
            paired=validation['paired']
            length_delta=paired['common_selected_length_delta']['mean']
            smooth_delta=paired['common_selected_smoothness_delta']['mean']
            eligible=(validation['selected_success_rate']>=validation['base']['selected_success_rate']
                      and paired['lost_selected_tasks']==0 and length_delta is not None and smooth_delta is not None
                      and length_delta<=0 and smooth_delta<=0)
            key=(float(validation['selected_success_rate']),-(float(length_delta or 0)/50+float(smooth_delta or 0)/.1))
            if eligible and key > best_key:
                best_key=key; best_update=update
                torch.save({'variant':'p56_path_chunk_fpo','run_manifest_sha256':manifest_sha,'policy_state_dict':policy.state_dict(),'generator_checkpoint':str(Path(args.checkpoint).resolve()),'args':vars(args),'update':update,'validation':validation},output/'best_p56_path_chunk.pth')
        if update % args.checkpoint_interval == 0 or update == 1:
            torch.save({'variant':'p56_path_chunk_fpo','run_manifest_sha256':manifest_sha,'policy_state_dict':policy.state_dict(),'generator_checkpoint':str(Path(args.checkpoint).resolve()),'args':vars(args),'update':update},output/f'p56_update_{update:04d}.pth')
        history.append(row); print(json.dumps(row),flush=True)
    total_seconds=float(time.perf_counter()-run_start)
    torch.save({'variant':'p56_path_chunk_fpo','run_manifest_sha256':manifest_sha,'policy_state_dict':policy.state_dict(),'generator_checkpoint':str(Path(args.checkpoint).resolve()),'args':vars(args),'total_wall_seconds':total_seconds},output/'last_p56_path_chunk.pth'); (output/'history.json').write_text(json.dumps(history,indent=2)+'\n'); print(json.dumps({'best_update':best_update,'best_key':best_key,'total_wall_seconds':total_seconds,'mean_seconds_per_update':total_seconds/max(args.updates,1)}),flush=True)

if __name__=='__main__': main()
