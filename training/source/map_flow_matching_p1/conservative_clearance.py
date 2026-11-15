"""Conservative distance-field bound for OPTIONAL proposal acceptance.

This is an ANALYTIC map-visible geometric guard, NOT learned safety and NOT
an exact footprint collision checker. It must be disclosed as such. It never
searches, optimizes or repairs a path. A negative bound means unknown, not
collision. Default v7 deployment is unchanged; this is for new experiments.

Proof sketch: exact EDT gives distance to occupied pixel CENTERS. Subtracting
sqrt(2)/2 lower-bounds distance to their closed square cells. At a continuous
point p, each grid center q yields D(q)-||p-q|| by 1-Lipschitzness. Taking the
maximum of four such lower bounds remains conservative. Uniform segment
samples at spacing<=h require another h/2 deduction to cover between samples.
The map exterior is separately bounded by its analytic boundary distance.
"""
from __future__ import annotations
import numpy as np
import torch


def center_distance_grid(occupancy):
    from scipy.ndimage import distance_transform_edt
    m=np.asarray(occupancy)
    if m.ndim!=2 or not np.isin(m,[0,1]).all():raise ValueError('binary 2D occupancy required')
    free=np.pad(m<.5,1,constant_values=False)
    return distance_transform_edt(free)[1:-1,1:-1].astype(np.float32)


def point_lower_bound(distance_grid,points_px):
    """distance_grid [N,H,W], points_px [N,P,2], distances in grid pixels."""
    if distance_grid.ndim!=3 or points_px.ndim!=3 or points_px.shape[-1]!=2 or len(distance_grid)!=len(points_px):
        raise ValueError('one distance grid per path point batch required')
    if not torch.isfinite(distance_grid).all() or not torch.isfinite(points_px).all():
        raise ValueError('finite distance grids and points required')
    if (distance_grid<0).any():raise ValueError('nonnegative center-distance grid required')
    n,h,w=distance_grid.shape
    lower=points_px.floor();upper=lower+1
    coordinates=torch.stack([lower,upper,torch.stack([lower[...,0],upper[...,1]],-1),torch.stack([upper[...,0],lower[...,1]],-1)],-2)
    coordinates[...,0].clamp_(0,w-1);coordinates[...,1].clamp_(0,h-1)
    index=coordinates[...,1].long()*w+coordinates[...,0].long()
    values=distance_grid.flatten(1).gather(1,index.flatten(1)).reshape_as(index)
    corner_bounds=values-(coordinates-points_px[...,None,:]).norm(dim=-1)
    obstacle=corner_bounds.amax(-1)-2**-.5
    border=torch.stack([points_px[...,0]+.5,points_px[...,1]+.5,w-.5-points_px[...,0],h-.5-points_px[...,1]],-1).amin(-1)
    return torch.minimum(obstacle,border)-1e-5 # Roundoff budget, never relaxation.


@torch.no_grad()
def swept_lower_bound(distance_grid,paths_px,spacing=.25):
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('positive finite sampling spacing required')
    if paths_px.ndim!=3 or paths_px.shape[-1]!=2 or paths_px.shape[1]<2:raise ValueError('NxTx2 path required')
    if not torch.isfinite(paths_px).all():raise ValueError('finite path required')
    delta=paths_px[:,1:]-paths_px[:,:-1]
    max_step=delta.norm(dim=-1).amax()
    samples=max(2,int(torch.ceil(max_step/spacing).item())+1)
    fractions=torch.linspace(0,1,samples,device=paths_px.device,dtype=paths_px.dtype)
    point=(paths_px[:,:-1,None,:]+fractions[None,None,:,None]*delta[:,:,None,:]).flatten(1,2)
    point_bounds=point_lower_bound(distance_grid,point)
    # Maximum nearest-sample arc distance <= spacing/2 on every segment.
    return point_bounds.amin(1)-spacing/2


@torch.no_grad()
def certified_new_proposal(distance_grid,paths_px,robot_radius=1.5,spacing=.25):
    if not np.isfinite(robot_radius) or robot_radius<0:raise ValueError('nonnegative finite robot radius required')
    lower=swept_lower_bound(distance_grid,paths_px,spacing)
    return lower>float(robot_radius),lower
