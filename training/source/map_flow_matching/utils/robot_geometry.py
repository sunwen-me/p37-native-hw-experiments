"""Versioned disk footprint in continuous grid coordinates.

Cell (row=y,col=x) is the closed square [x-.5,x+.5] x [y-.5,y+.5].
The map exterior is occupied. Contact counts as collision. Segments are
checked as swept disks, not a finite set of interpolation samples.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np

@dataclass(frozen=True)
class RobotGeometry:
    radius: float
    safety_margin: float = 0.0
    resolution: float = 1.0
    units: str = 'grid'
    version: str = 'disk_cells_v1'

    def __post_init__(self):
        if self.version != 'disk_cells_v1' or self.units not in ('grid','meters'):
            raise ValueError('unsupported geometry protocol/units')
        if not all(np.isfinite(v) for v in (self.radius,self.safety_margin,self.resolution)) or self.radius<0 or self.safety_margin<0 or self.resolution<=0:
            raise ValueError('invalid geometry dimensions')
        if self.units=='grid' and self.resolution!=1:
            raise ValueError('grid units require resolution=1; use meters for physical dimensions')

    @property
    def radius_grid(self):
        return (self.radius+self.safety_margin)/self.resolution

    def to_json(self):return json.dumps(asdict(self),sort_keys=True)

    @classmethod
    def from_json(cls,value):return cls(**json.loads(str(value)))

class FootprintChecker:
    def __init__(self,occupancy,geometry):
        self.map_data=np.asarray(occupancy)
        if self.map_data.ndim!=2 or not np.isfinite(self.map_data).all():raise ValueError('invalid map')
        self.geometry=geometry;self.height,self.width=self.map_data.shape
        y,x=np.where(self.map_data>.5)
        self.lo=np.column_stack((x,y)).astype(float)-.5;self.hi=self.lo+1

    def segment_clearance(self,a,b):
        a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
        if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite([a,b]).all():return -np.inf
        edge=min(a[0]+.5,b[0]+.5,a[1]+.5,b[1]+.5,self.width-.5-a[0],self.width-.5-b[0],self.height-.5-a[1],self.height-.5-b[1])
        if edge<=0:return float(edge)
        if not len(self.lo):return float(edge)
        # Minimum segment-to-square distance: intersection, endpoints to square,
        # or square vertices to segment. Includes exact tangencies within tolerance.
        d=b-a;t0=np.zeros(len(self.lo));t1=np.ones(len(self.lo));possible=np.ones(len(self.lo),bool)
        for axis in range(2):
            if d[axis]==0:possible &= (a[axis]>=self.lo[:,axis])&(a[axis]<=self.hi[:,axis])
            else:
                u=(self.lo[:,axis]-a[axis])/d[axis];v=(self.hi[:,axis]-a[axis])/d[axis]
                t0=np.maximum(t0,np.minimum(u,v));t1=np.minimum(t1,np.maximum(u,v))
        if np.any(possible&(t0<=t1)):return 0.
        endpoint=min(np.linalg.norm(np.maximum(np.maximum(self.lo-a,a-self.hi),0),axis=1).min(),np.linalg.norm(np.maximum(np.maximum(self.lo-b,b-self.hi),0),axis=1).min())
        corners=np.stack([self.lo,self.hi,np.column_stack([self.lo[:,0],self.hi[:,1]]),np.column_stack([self.hi[:,0],self.lo[:,1]])],axis=1).reshape(-1,2)
        den=float(d@d);t=np.clip(((corners-a)@d)/den,0,1) if den else np.zeros(len(corners))
        vertex=np.linalg.norm(corners-(a+t[:,None]*d),axis=1).min()
        return float(min(edge,endpoint,vertex))

    def check_point(self,x,y):return self.segment_clearance((x,y),(x,y))<=self.geometry.radius_grid+1e-10
    def check_path_segment(self,a,b):return self.segment_clearance(a,b)<=self.geometry.radius_grid+1e-10
    def check_path(self,path):
        p=np.asarray(path)
        if p.ndim!=2 or p.shape[1]!=2 or not len(p):raise ValueError('expected nonempty Nx2 path')
        hit=np.array([self.check_point(*p[0])]+[self.check_path_segment(a,b) for a,b in zip(p[:-1],p[1:])])
        return bool(not hit.any()),hit

def resample_safe_polyline(path,horizon=128):
    """Subdivide without removing corners; reject insufficient point budget."""
    p=np.asarray(path,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('invalid path')
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-12]]
    if not len(p):raise ValueError('empty path')
    if len(p)>horizon:raise ValueError('point budget would remove corners')
    if len(p)==1:return np.repeat(p,horizon,axis=0).astype(np.float32)
    length=np.linalg.norm(np.diff(p,axis=0),axis=1);counts=np.ones(len(length),dtype=int)
    for _ in range(horizon-len(p)):counts[np.argmax(length/counts)]+=1
    return np.concatenate([np.linspace(a,b,n+1)[:-1] for a,b,n in zip(p[:-1],p[1:],counts)]+[p[-1:]]).astype(np.float32)

def evaluate_robot_path(path,occupancy,start,goal,geometry,endpoint_tolerance=2.):
    """Use for new-protocol evaluation; never silently relabel historical results."""
    c=FootprintChecker(occupancy,geometry);p=np.asarray(path)
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all():raise ValueError('invalid path')
    safe=c.check_path(p)[0];completed=c.check_path(np.vstack([start,p,goal]))[0]
    se=float(np.linalg.norm(p[0]-start));ge=float(np.linalg.norm(p[-1]-goal))
    return dict(safe=safe,completed_safe=completed,success=bool(safe and completed and se<=endpoint_tolerance and ge<=endpoint_tolerance),start_error=se,goal_error=ge,geometry_version=geometry.version)
