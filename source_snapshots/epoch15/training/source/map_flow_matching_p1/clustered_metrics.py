"""Paired task-weighted statistics with whole-map clustered uncertainty.

Maps may contribute several start/goal tasks. Resampling tasks as independent
would understate uncertainty; each bootstrap draw copies complete map groups.
"""
from __future__ import annotations
import numpy as np


def map_cluster_bootstrap(delta,map_ids,seed=0,draws=2000):
    delta=np.asarray(delta,dtype=np.float64);groups=np.asarray(map_ids)
    if delta.ndim!=1 or groups.shape!=delta.shape:
        raise ValueError('one group ID per task delta required')
    finite=np.isfinite(delta);delta=delta[finite];groups=groups[finite]
    if not len(delta):return {'mean':None,'ci95':None,'tasks':0,'maps':0,'resampling_unit':'whole map'}
    _,inverse=np.unique(groups,return_inverse=True);counts=np.bincount(inverse);sums=np.bincount(inverse,weights=delta)
    rng=np.random.default_rng(seed)
    index=rng.integers(0,len(counts),(draws,len(counts)))
    replicates=sums[index].sum(1)/counts[index].sum(1)
    return {'mean':float(delta.mean()),'ci95':np.quantile(replicates,[.025,.975]).tolist(),
            'tasks':len(delta),'maps':len(counts),'resampling_unit':'whole map','point_estimate':'task weighted'}
