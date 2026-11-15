"""Public neural-only Flow + continuous-path refinement deployment API.

Exact checker and offline teacher are deliberately NOT dependencies of this
module. Native map preprocessing is signed EDT; inference consists of a frozen
Flow draw plus one dense neural pass and final-path neural ranking.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from path_refinement import DensePathRefiner,signed_distance_features,neural_candidate_selection,refiner_from_config
from flow_prior import load_flow_prior
from run_manifest import file_sha256


def _maps_as_list(occupancy):
    if isinstance(occupancy, np.ndarray) and occupancy.dtype != object:
        values=np.asarray(occupancy,dtype=np.float32)
        if values.ndim==2:return [values]
        if values.ndim==3:return [values[index] for index in range(len(values))]
        raise ValueError('occupancy must have shape [H,W] or [N,H,W]')
    values=[np.asarray(value,dtype=np.float32) for value in occupancy]
    if not values:raise ValueError('occupancy must contain at least one map')
    if any(value.ndim!=2 for value in values):raise ValueError('each occupancy map must be two-dimensional')
    return values


class FlowRefinementPlanner(nn.Module):
    def __init__(self,generator_checkpoint,refiner_checkpoint,device='cuda',include_base=True,
                 compile_mode=None):
        super().__init__()
        device=torch.device(device)
        self.prior,self.prior_state=load_flow_prior(str(generator_checkpoint),device)
        state=torch.load(refiner_checkpoint,map_location='cpu',weights_only=False)
        c=dict(state['config']);c.pop('representation',None)
        self.refiner=refiner_from_config(c).to(device)
        self.refiner.load_state_dict(state['model_state_dict'],strict=True)
        if file_sha256(generator_checkpoint)!=state['cache_audit']['source_checkpoint_sha256']:
            raise ValueError('wrong frozen generator: refiner was trained on another prior')
        self.prior.requires_grad_(False).eval();self.refiner.requires_grad_(False).eval()
        if state.get('deployment_protocol') not in (None,'paired_quality_and_dense_EDT_margin_with_original_fallback_v7','paired_quality_and_dense_EDT_margin_with_original_fallback_v8'):
            raise ValueError('unsupported saved deployment selector protocol')
        self.cache_config=state['cache_audit'];self.refiner_phase=state['phase'];self.refiner_update=state['update']
        self.include_base=bool(include_base)
        self.compile_mode=compile_mode
        if compile_mode is not None:
            if not hasattr(torch,'compile'):
                raise RuntimeError('torch.compile is unavailable in this PyTorch build')
            self.prior.vector_field=torch.compile(self.prior.vector_field,mode=compile_mode)

    def _resolve_flow_steps(self, flow_steps):
        steps = self.cache_config['flow_steps'] if flow_steps is None else flow_steps
        try:
            resolved = int(steps)
        except (TypeError, ValueError) as exc:
            raise ValueError('flow_steps must be a positive integer') from exc
        if resolved != steps or resolved < 1:
            raise ValueError('flow_steps must be a positive integer')
        return resolved

    @torch.inference_mode()
    def _draw_base(self,prior_maps,starts,goals,candidates,generator,flow_steps=None):
        self.prior.vector_field.latent_scale=self.cache_config['latent_scale']
        return self.prior.sample(prior_maps,starts,goals,candidates=candidates,
             num_steps=self._resolve_flow_steps(flow_steps),noise_scale=self.cache_config['noise_scale'],generator=generator)

    @torch.inference_mode()
    def plan(self,occupancy,starts_px,goals_px,candidates=4,seed=20260921,flow_steps=None):
        """Return pixel paths + a LEARNED safety estimate, not certified safety.

        `best_effort_path` always exists. If `abstain` is true the learned
        estimate is negative; exact safety remains an OFFLINE evaluation.
        Pooled baseline options are explicitly counted (K flow draws, 2K paths).
        """
        device=next(self.refiner.parameters()).device
        maps=_maps_as_list(occupancy)
        starts=np.asarray(starts_px,dtype=np.float32).reshape(-1,2)
        goals=np.asarray(goals_px,dtype=np.float32).reshape(-1,2)
        count=len(maps)
        if len(starts)!=count or len(goals)!=count:
            raise ValueError('one start/goal pair per map required')
        if not np.isfinite(starts).all() or not np.isfinite(goals).all():
            raise ValueError('finite endpoints required')
        if int(candidates)!=candidates or candidates<1:
            raise ValueError('candidates must be a positive integer')
        candidates=int(candidates)
        buckets=defaultdict(list)
        for index,occ in enumerate(maps):
            if min(occ.shape)<4:
                raise ValueError('map height and width must both be at least 4')
            if not np.isfinite(occ).all() or not np.isin(occ,[0.,1.]).all():
                raise ValueError('binary occupancy (0 free, 1 obstacle) required')
            height,width=occ.shape
            if (not 0<=starts[index,0]<=width-1 or
                    not 0<=starts[index,1]<=height-1 or
                    not 0<=goals[index,0]<=width-1 or
                    not 0<=goals[index,1]<=height-1):
                raise ValueError('endpoints outside map')
            buckets[(height,width)].append(index)

        resolved_flow_steps=self._resolve_flow_steps(flow_steps)
        horizon=self.prior.horizon
        options_per_task=candidates*(2 if self.include_base else 1)
        dtype=next(self.refiner.parameters()).dtype
        paths=torch.empty((count,options_per_task,horizon,2),device=device,dtype=dtype)
        scores=torch.empty((count,options_per_task),device=device,dtype=dtype)
        logits=torch.empty_like(scores)
        selected=torch.empty(count,device=device,dtype=torch.long)
        base_valid=torch.empty((count,candidates),device=device,dtype=torch.bool)
        rng=torch.Generator(device=device).manual_seed(int(seed)+710001)

        for (height,width),indices in buckets.items():
            features=torch.from_numpy(np.stack([
                signed_distance_features(maps[index],self.refiner.distance_scale)
                for index in indices
            ])).to(device)
            prior_maps=features[:,:2].clone()
            prior_maps[:,1]=prior_maps[:,1].clamp(0,1)
            scale=np.asarray([(width-1)/2,(height-1)/2],dtype=np.float32)
            starts_norm=torch.from_numpy(starts[indices]/scale-1).to(device)
            goals_norm=torch.from_numpy(goals[indices]/scale-1).to(device)
            draw=self._draw_base(
                prior_maps,starts_norm,goals_norm,candidates,rng,resolved_flow_steps,
            )
            base,group_base_valid=draw['paths'],draw['decoder_support_valid']
            flat_maps=features[:,None].expand(-1,candidates,-1,-1,-1).flatten(0,1)
            refined=self.refiner(flat_maps,base.flatten(0,1)).reshape_as(base)
            options=torch.cat([base,refined],1) if self.include_base else refined
            baseline_available=group_base_valid.any(1)
            fallback=draw['selected_index'] if self.include_base else None
            group_selected,group_scores,group_logits=neural_candidate_selection(
                self.refiner,features,options,fallback,
                baseline_available if self.include_base else None,
                candidates if self.include_base else None,
            )
            rows=torch.as_tensor(indices,device=device,dtype=torch.long)
            path_scale=torch.as_tensor(scale,device=device,dtype=options.dtype)
            paths[rows]=(options+1)*path_scale
            scores[rows]=group_scores
            logits[rows]=group_logits
            selected[rows]=group_selected
            base_valid[rows]=group_base_valid

        rows=torch.arange(count,device=device)
        estimate=logits[rows,selected].sigmoid()
        baseline_available=base_valid.any(1)
        used_fallback=(estimate<.5)&baseline_available if self.include_base else torch.zeros_like(baseline_available)
        abstain=(estimate<.5)&(~baseline_available) if self.include_base else estimate<.5
        return {'paths':paths,'selected_index':selected,'best_effort_path':paths[rows,selected],
          'learned_success_probability':estimate,'used_original_fallback':used_fallback,
          'abstain':abstain,'base_decoder_support_valid':base_valid,'scores':scores,
          'flow_draws_per_task':candidates,'path_options_per_task':options_per_task,
          'flow_steps':resolved_flow_steps,
          'compile_mode':self.compile_mode,'map_shapes':tuple(tuple(map(int,m.shape)) for m in maps),
          'exact_geometry_online':False,'online_search':False,'online_optimizer':False,
          'safety_score_is_model_visible_not_certified':True,
          'refiner_phase':self.refiner_phase,'refiner_update':self.refiner_update}
