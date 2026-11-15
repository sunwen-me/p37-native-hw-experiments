"""Learned dense continuous refinement around a frozen global Flow prior.

Online inference is one neural pass with bilinear map queries: no exact
geometry oracle, search, iterative path repair or optimized teacher is called.
Unlike P56, ALL interior final trajectory points are controllable. The RL
actor is a Gaussian dense-trajectory policy (not an approximate field FPO).
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _resolve_map_shape(map_size=64, map_height=None, map_width=None):
    if map_height is None and map_width is None:
        if isinstance(map_size, (tuple, list)) and len(map_size) == 2:
            map_height, map_width = map_size
        else:
            map_height = map_width = map_size
    elif map_height is None or map_width is None:
        raise ValueError('map_height and map_width must be supplied together')
    height, width = int(map_height), int(map_width)
    if min(height, width) < 2:
        raise ValueError('map height and width must both be at least 2')
    return height, width


def _pixel_scale(reference, height, width):
    """Map normalized x/y deltas to grid-pixel deltas independently."""
    return reference.new_tensor([(width - 1) / 2, (height - 1) / 2])


def signed_distance_features(occupancy, distance_scale=8.0):
    """Map-visible EDT features; these are NOT an exact footprint checker."""
    import cv2
    m=np.asarray(occupancy,dtype=np.float32)
    free=(m<.5).astype(np.uint8)
    padded=np.pad(free,1,constant_values=0)
    outside=cv2.distanceTransform(padded,cv2.DIST_L2,5)[1:-1,1:-1]
    inside=cv2.distanceTransform((1-free),cv2.DIST_L2,5)
    signed=outside-inside
    gy,gx=np.gradient(signed)
    return np.stack([m,signed/float(distance_scale),gx,gy]).astype(np.float32)


def query_maps(maps,paths):
    return F.grid_sample(maps,paths.unsqueeze(2),mode='bilinear',padding_mode='border',align_corners=True).squeeze(-1).transpose(1,2)


def swept_clearance(paths,maps,distance_scale=8.,offset=2**-.5,samples=8):
    """Model-visible signed EDT and boundaries, not exact collision safety."""
    if maps.ndim != 4 or paths.ndim != 3 or paths.shape[0] != maps.shape[0] or paths.shape[-1] != 2:
        raise ValueError('maps and paths must have shapes [B,C,H,W] and [B,T,2]')
    height, width = maps.shape[-2:]
    frac=torch.linspace(0,1,samples,device=paths.device,dtype=paths.dtype)
    points=(paths[:,:-1,None,:]+frac[None,None,:,None]*(paths[:,1:,None,:]-paths[:,:-1,None,:])).flatten(1,2)
    values=query_maps(maps,points)[:,:,1]*distance_scale-offset
    px=(points+1)*_pixel_scale(points,height,width)
    border=torch.stack([px[:,:,0]+.5,px[:,:,1]+.5,width-.5-px[:,:,0],height-.5-px[:,:,1]],-1).amin(-1)
    return torch.minimum(values,border)


def path_metrics(paths,map_size=64,*,map_height=None,map_width=None):
    height,width=_resolve_map_shape(map_size,map_height,map_width)
    px=paths*_pixel_scale(paths,height,width)
    step=px[:,1:]-px[:,:-1]
    speed=step.norm(dim=-1)
    second=step[:,1:]-step[:,:-1]
    dot=(step[:,1:]*step[:,:-1]).sum(-1)
    cosine=(dot/(speed[:,1:]*speed[:,:-1]).clamp_min(1e-8)).clamp(-1+1e-6,1-1e-6)
    turn=torch.where((speed[:,1:]>1e-5)&(speed[:,:-1]>1e-5),cosine.acos(),torch.zeros_like(cosine))
    return {'length':speed.sum(-1),'smoothness':second.norm(dim=-1).mean(-1),'turning_angle':turn.sum(-1),
            'squared_curvature':second.square().sum(-1).mean(-1)}


class DensePathRefiner(nn.Module):
    """Spatially-conditioned Conv1D mean actor + separate path safety head.

    Dense 126*2 effective controls (horizon128), fixed endpoints, px units.
    Identity zero-head initialization. Local derivative inputs expose decoder
    jitter directly, rather than hoping pooled map context discovers it.
    """
    def __init__(self,hidden=64,max_offset_px=3.,distance_scale=8.,robot_radius=1.5,safety_logit_scale=10.,clearance_calibration_bias=.05,feature_ablation='none'):
        super().__init__()
        self.hidden=int(hidden);self.max_offset_px=float(max_offset_px)
        self.distance_scale=float(distance_scale);self.robot_radius=float(robot_radius);self.safety_logit_scale=float(safety_logit_scale);self.clearance_calibration_bias=float(clearance_calibration_bias)
        self.feature_ablation=str(feature_ablation)
        if self.feature_ablation not in ('none','no_local_map_edt','no_derivatives','no_endpoint_progress'):
            raise ValueError(f'unknown feature_ablation: {self.feature_ablation}')
        # x/y, local map 4, 3 patch distances, velocities 2, second diff2,
        # goal vector2, start vector2, progress1 =18 channels.
        self.actor=nn.Sequential(nn.Conv1d(18,hidden,5,padding=2),nn.SiLU(),
            nn.Conv1d(hidden,hidden,5,padding=2),nn.SiLU(),
            nn.Conv1d(hidden,hidden,5,padding=4,dilation=2),nn.SiLU(),
            nn.Conv1d(hidden,hidden,5,padding=8,dilation=4),nn.SiLU(),nn.Conv1d(hidden,6,1))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        self.safety=nn.Sequential(nn.Linear(8,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.safety[-1].weight);nn.init.zeros_(self.safety[-1].bias)

    def features(self,maps,base):
        height,width=maps.shape[-2:]
        scale=_pixel_scale(base,height,width).view(1,1,2)
        local=query_maps(maps,base)
        vel=torch.cat([base[:,1:]-base[:,:-1],base[:,-1:]-base[:,-2:-1]],1)*scale
        lap=F.pad((base[:,2:]-2*base[:,1:-1]+base[:,:-2]).transpose(1,2),(1,1)).transpose(1,2)*scale
        patch=[]
        for delta in [(-1.,0.),(1.,0.),(0.,1.)]:
            q=base+torch.tensor(delta,device=base.device,dtype=base.dtype)[None,None]/scale
            patch.append(query_maps(maps,q)[:,:,1:2])
        start=base[:,:1]-base;goal=base[:,-1:]-base
        progress=torch.linspace(0,1,base.shape[1],device=base.device)[None,:,None].expand(len(base),-1,-1)
        if self.feature_ablation == 'no_local_map_edt':
            local=torch.zeros_like(local); patch=[torch.zeros_like(x) for x in patch]
        elif self.feature_ablation == 'no_derivatives':
            vel=torch.zeros_like(vel); lap=torch.zeros_like(lap)
        elif self.feature_ablation == 'no_endpoint_progress':
            start=torch.zeros_like(start); goal=torch.zeros_like(goal); progress=torch.zeros_like(progress)
        return torch.cat([base,local,*patch,vel,lap,goal,start,progress],-1).transpose(1,2)

    def mean_offset(self,maps,base):
        raw=self.actor(self.features(maps,base)).transpose(1,2)
        height,width=maps.shape[-2:]
        scale=_pixel_scale(base,height,width).view(1,1,2)
        value=.5*self.max_offset_px*torch.tanh(raw[:,:,-2:])
        for i,window in enumerate([3,5,9,17]):
            smooth=F.avg_pool1d(F.pad(base.transpose(1,2),(window//2,window//2),mode='replicate'),window,stride=1).transpose(1,2)
            value=value+.5*torch.tanh(raw[:,:,i:i+1])*(smooth-base)*scale
        value=self.max_offset_px*torch.tanh(value/self.max_offset_px)
        endpoint=torch.ones_like(value[:,:,:1]);endpoint[:,[0,-1]]=0
        return value*endpoint

    def forward(self,maps,base):
        return self.apply_offset(
            base,self.mean_offset(maps,base),map_height=maps.shape[-2],
            map_width=maps.shape[-1],
        )

    @staticmethod
    def apply_offset(base,offset,map_size=64,*,map_height=None,map_width=None):
        height,width=_resolve_map_shape(map_size,map_height,map_width)
        mask=torch.ones_like(offset[:,:,:1]);mask[:,[0,-1]]=0
        return base+offset*mask/_pixel_scale(base,height,width)

    def safety_features(self,maps,paths):
        clearance=swept_clearance(paths,maps,self.distance_scale)
        height,width=maps.shape[-2:]
        quality=path_metrics(paths,map_height=height,map_width=width)
        direct=((paths[:,-1]-paths[:,0])*_pixel_scale(paths,height,width)).norm(dim=-1)
        deficit=(self.robot_radius-clearance).clamp_min(0)
        return torch.stack([clearance.amin(1)/4,clearance.mean(1)/8,
          torch.quantile(clearance,.05,dim=1)/4,deficit.mean(1),deficit.amax(1),
          quality['length']/direct.clamp_min(1),quality['smoothness'],quality['turning_angle']/32],-1)

    def success_logit(self,maps,paths):
        features=self.safety_features(maps,paths)
        # Calibrate a map-visible conservative clearance estimate instead of
        # letting an untrained pooled MLP arbitrarily rank already-safe paths.
        return self.safety_logit_scale*(features[:,0]*4-self.robot_radius)+3.*torch.tanh(self.safety(features).squeeze(-1))

    @torch.no_grad()
    def sample_offsets(self,maps,base,std_px,generator,draws=4):
        mean=self.mean_offset(maps,base)
        eps=torch.randn(len(base),draws,*mean.shape[1:],device=mean.device,dtype=mean.dtype,generator=generator)
        value=mean[:,None]+float(std_px)*eps
        value[:,:,[0,-1]]=0
        logprob=joint_offset_logprob(value,mean[:,None],std_px)
        return value,logprob,mean


def whiten_offset(value,std_px,correlation=0.):
    """Whiten an interior AR(1) Gaussian action, both x and y independently."""
    if std_px<=0 or not 0<=correlation<1:
        raise ValueError('positive std and 0<=correlation<1 required')
    interior=value[...,1:-1,:]/float(std_px)
    if correlation==0:return interior
    return torch.cat([interior[...,:1,:],
      (interior[...,1:,:]-correlation*interior[...,:-1,:])/math.sqrt(1-correlation**2)],-2)


def correlated_offset_noise(shape,generator,device,correlation=0.,dtype=torch.float32):
    """Smooth exploration with full-rank, exactly known joint likelihood."""
    eps=torch.randn(*shape,device=device,dtype=dtype,generator=generator)
    h=shape[-2]-2
    idx=torch.arange(h,device=device)
    power=(idx[:,None]-idx[None,:]).clamp_min(0)
    transform=(float(correlation)**power)*(idx[:,None]>=idx[None,:])
    transform[:,1:]*=math.sqrt(1-correlation**2)
    interior=torch.einsum('ij,...jc->...ic',transform.to(dtype),eps[...,1:-1,:])
    result=torch.zeros_like(eps);result[...,1:-1,:]=interior
    return result


def joint_offset_logprob(offset,mean,std_px,correlation=0.):
    """Exact JOINT Gaussian likelihood, only effective interior actions.

    AR(1) correlation is optional but must match the sampling distribution.
    Never average likelihood across action dimensions as field-PPO did.
    """
    white=whiten_offset(offset-mean,std_px,correlation)
    d=white.shape[-2];channels=white.shape[-1]
    logdet=channels*(d*math.log(std_px)+(d-1)*.5*math.log(1-correlation**2))
    return -.5*white.square().sum((-2,-1))-.5*d*channels*math.log(2*math.pi)-logdet


def paired_group_advantages(reward,baseline):
    """Baseline is SAME state/base deterministic policy, independent of draws.

    No cross-task reward ranking and no group normalization that makes a
    failed group's least-bad sample into artificial global success.
    """
    delta=(reward-baseline[:,None]).detach()
    scale=delta.square().mean().sqrt().clamp_min(.05)
    return delta/scale


def exact_terminal_reward(success,length,smoothness,clearance,reference_length,robot_radius=1.5):
    """Safety dominant, normalized quality; no bonus for unnecessary clearance.

    Failed paths retain an exact-clearance severity signal. Constant +4/-4
    safety separation exceeds the entire bounded successful quality penalty.
    Reference is archived teacher length, not claimed globally shortest.
    """
    length_penalty=(length/reference_length.clamp_min(1)).clamp(0,3)
    smooth_penalty=(smoothness/.1).clamp(0,3)
    good=4.-.8*length_penalty-.8*smooth_penalty
    deficit=(robot_radius-clearance).clamp_min(0).clamp_max(4)/robot_radius
    bad=-4.-deficit
    return torch.where(success,good,bad)


def geometry_auxiliary_loss(model,maps,base,reference_length):
    """Training-only differentiable geometry surrogate; never used to repair
    paths at inference. Quality gradients are restricted to likely-safe paths.
    """
    path=model(maps,base)
    quality=path_metrics(path,maps.shape[-1])
    clearance=swept_clearance(path,maps,model.distance_scale)
    deficit=(model.robot_radius+.15-clearance).clamp_min(0)
    barrier=deficit.square().mean(1)+deficit.amax(1).square()
    safe_weight=torch.sigmoid((clearance.amin(1).detach()-model.robot_radius)/.15)
    # Length and curvature terms have comparable scale after normalization.
    objective=.8*quality['length']/reference_length.clamp_min(1)+.8*quality['smoothness']/.1
    displacement=(path-base).square().mean((1,2))*((maps.shape[-1]-1)/2)**2
    return (safe_weight*objective+8.*barrier+.01*displacement).mean(),path


@torch.no_grad()
def neural_candidate_selection(model,maps,paths,fallback_selected=None,fallback_available=None,base_count=None):
    """Re-score FINAL paths, independent of old field decoder support.

    No exact checker; neural safety then path quality. Returning best-effort
    rather than silently accepting unsafe paths is recorded by the evaluator.
    """
    b,k,h,_=paths.shape
    m=maps[:,None].expand(-1,k,-1,-1,-1).flatten(0,1)
    p=paths.flatten(0,1)
    logits=model.success_logit(m,p)
    q=path_metrics(p,map_height=maps.shape[-2],map_width=maps.shape[-1])
    # Longer decoder connector segments cannot be checked at just 8 coarse
    # interpolation fractions. Match the training barrier margin and query a
    # denser map-visible sweep. This is still an approximate EDT feature, NOT
    # the exact footprint checker and never a path optimizer.
    clearance=swept_clearance(p,m,getattr(model,'distance_scale',8.),samples=64).amin(1)
    margin_logit=10*(clearance+getattr(model,'clearance_calibration_bias',.05)-getattr(model,'robot_radius',1.5))
    logits=torch.minimum(logits,margin_logit)
    if base_count is not None:
        if k!=2*base_count:raise ValueError('paired base/refined options required')
        length=q['length'].reshape(b,k);smooth=q['smoothness'].reshape(b,k)
        improved=(length[:,base_count:]<=length[:,:base_count]+1e-4)&(smooth[:,base_count:]<=smooth[:,:base_count]+1e-6)
        flat=logits.reshape(b,k)
        flat[:,base_count:]=flat[:,base_count:].masked_fill(~improved,-torch.inf)
        logits=flat.flatten()
    # Safety and quality are lexicographic, NOT competing reward scales.
    # If any final path is predicted safe, unsafe paths cannot win by being
    # short. If all predictions are unsafe, choose greatest safety estimate,
    # without a quality incentive for colliding shortcuts. No exact checker.
    logits=logits.reshape(b,k)
    # Once a candidate passes the safety gate, optimize route length first.
    # Smoothness only breaks near-length ties; the old weighted score could
    # prefer a substantially longer path merely because it was smoother.
    quality=(-q['length']-5.*q['smoothness']).reshape(b,k)
    predicted_safe=logits>=0
    safe_quality=quality.masked_fill(~predicted_safe,-torch.inf)
    any_safe=predicted_safe.any(1,keepdim=True)
    score=torch.where(any_safe,safe_quality,logits)
    if fallback_selected is not None:
        if fallback_available is None:raise ValueError('explicit fallback availability required')
        # Only baseline options may use the prior decoder's support evidence.
        # Refined options are ranked by their FINAL path, never stale validity.
        fallback_score=torch.full_like(score,-torch.inf)
        fallback_score.scatter_(1,fallback_selected[:,None],0.)
        use_fallback=(~any_safe[:,0])&fallback_available
        score=torch.where(use_fallback[:,None],fallback_score,score)
    return score.argmax(1),score,logits


class ControlledDenseRefiner(DensePathRefiner):
    """One effective RL action scales a FULL-resolution learned trajectory.

    The action changes high-frequency curvature everywhere, unlike P56's
    additive low-frequency anchors. This phase freezes the dense geometric
    refiner and learns only state-dependent blending via exact Gaussian PPO.
    Scalar action control is not claimed to create missing global topology.
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.controller=nn.Sequential(nn.Linear(16,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.controller[-1].weight);nn.init.zeros_(self.controller[-1].bias)

    def raw_refined(self,maps,base):
        return super().forward(maps,base)

    def control_features(self,maps,base,refined):
        return torch.cat([self.safety_features(maps,base),self.safety_features(maps,refined)],-1).clamp(-20,20)

    def control_mean(self,features):
        return self.controller(features).squeeze(-1)

    @staticmethod
    def apply_control(base,refined,action):
        strength=1.+.5*torch.tanh(action)
        return base+strength[...,None,None]*(refined-base)

    def forward(self,maps,base):
        refined=self.raw_refined(maps,base)
        action=self.control_mean(self.control_features(maps,base,refined))
        return self.apply_control(base,refined,action)


def refiner_from_config(config):
    """Checkpoint-compatible factory for dense and controlled-dense actors."""
    config=dict(config);config.pop('representation',None)
    controlled=config.pop('controller',None)
    if controlled not in (None,'scalar_blend_gaussian'):
        raise ValueError('unknown controller representation')
    return ControlledDenseRefiner(**config) if controlled else DensePathRefiner(**config)
