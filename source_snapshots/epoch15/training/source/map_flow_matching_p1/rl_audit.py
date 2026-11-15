"""Pure, unit-tested diagnostics. No online search, repair or geometry oracle."""
from __future__ import annotations
import hashlib
import numpy as np


def map_hashes(maps):
    return np.asarray([hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest() for m in maps])


def anchor_basis(horizon: int, anchors: int):
    """P56's linear interpolation, with the two fixed endpoint columns removed."""
    if horizon < 3 or anchors < 3:
        raise ValueError('need at least three path points and anchors')
    q = np.linspace(0, anchors - 1, horizon)
    left, right = np.floor(q).astype(int), np.ceil(q).astype(int)
    basis = np.zeros((horizon, anchors))
    for i, (lo, hi, alpha) in enumerate(zip(left, right, q - left)):
        basis[i, lo] += 1 - alpha
        basis[i, hi] += alpha
    return basis[:, 1:-1]


def curvature_capacity(paths, anchors=8):
    """Optimistic OFFLINE least-squares bound for sum-squared second differences.

    Uses ideal piecewise-linear interpolation, ignoring floating-point
    roundoff. Offsets are UNBOUNDED and safety is ignored. This is NOT a safe
    policy. A separate unchanged-row bound covers mean-norm smoothness;
    neither bound covers path length or total turning angle.
    """
    paths = np.asarray(paths, dtype=np.float64)
    basis = anchor_basis(paths.shape[-2], anchors)
    curvature_basis = np.diff(basis, n=2, axis=0)
    curvature = np.diff(paths, n=2, axis=-2)
    projection = curvature_basis @ np.linalg.pinv(curvature_basis)
    fit = np.einsum('ij,...jc->...ic', projection, curvature)
    before = np.square(curvature).sum(axis=(-2, -1))
    after = np.square(curvature - fit).sum(axis=(-2, -1))
    # Exactly unchanged rows provide a rigorous (optimistic) lower bound
    # for the ACTUAL mean-norm smoothness metric, independently of the
    # least-squares result above. We optimistically zero every affected row.
    affected = np.linalg.norm(curvature_basis, axis=1) > 1e-12
    curvature_norm = np.linalg.norm(curvature, axis=-1)
    norm_total = curvature_norm.sum(axis=-1)
    removable_norm = curvature_norm[..., affected].sum(axis=-1)
    norm_fraction = np.divide(removable_norm, norm_total,
                              out=np.zeros_like(norm_total), where=norm_total > 1e-20)
    return {'squared_curvature_before': before, 'squared_curvature_lower_bound': after,
            'maximum_removable_energy_fraction': np.divide(before - after, before,
                out=np.zeros_like(before), where=before > 1e-20),
            'maximum_removable_mean_norm_fraction_upper_bound': norm_fraction,
            'mean_norm_smoothness_lower_bound': curvature_norm[..., ~affected].sum(axis=-1) / curvature_norm.shape[-1],
            'affected_second_difference_rows': int(np.count_nonzero(affected)),
            'total_second_difference_rows': int(paths.shape[-2] - 2)}


def quality_metrics(paths):
    paths = np.asarray(paths, dtype=np.float64)
    velocity = np.diff(paths, axis=-2)
    speed = np.linalg.norm(velocity, axis=-1)
    dot = (velocity[..., 1:, :] * velocity[..., :-1, :]).sum(axis=-1)
    denom = speed[..., 1:] * speed[..., :-1]
    angle = np.arccos(np.clip(np.divide(dot, denom, out=np.ones_like(dot), where=denom > 1e-10), -1, 1))
    return {'length': speed.sum(axis=-1),
            'smoothness': np.linalg.norm(np.diff(paths, n=2, axis=-2), axis=-1).mean(axis=-1),
            'turning_angle': angle.sum(axis=-1)}


def _mean(values, mask=None):
    values = np.asarray(values)
    if mask is not None:
        values = values[mask]
    return float(values.mean()) if values.size else None


def summarize(success, quality, selected, available=None):
    success = np.asarray(success, dtype=bool)
    selected = np.asarray(selected, dtype=int)
    available = np.ones(len(selected), bool) if available is None else np.asarray(available, bool)
    rows = np.arange(len(success))
    chosen = success[rows, selected] & available
    out = {'candidate_success_rate': float(success.mean()),
           'oracle_best_of_k_success_rate': float(success.any(axis=1).mean()),
           'selected_success_rate': float(chosen.mean()),
           'all_candidates_unsafe': int((~success.any(axis=1)).sum()),
           'accept_coverage': float(available.mean()),
           'accepted_failure_rate': _mean(~chosen, available),
           'successful_candidates': int(success.sum())}
    for key, value in quality.items():
        out['all_candidate_mean_' + key] = _mean(value)
        out['successful_candidate_mean_' + key] = _mean(value, success)
        out['successful_selected_mean_' + key] = _mean(np.asarray(value)[rows, selected], chosen)
    return out


def task_bootstrap(delta, seed=0, draws=2000):
    """Paired TASK bootstrap: candidates are not independent trials."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        return {'mean': None, 'ci95': None, 'tasks': 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, (draws, len(delta)), replace=True).mean(axis=1)
    return {'mean': float(delta.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(), 'tasks': int(len(delta))}


def paired_summary(base_success, success, base_quality, quality, base_selected, selected,
                   base_available=None, available=None, seed=0):
    bs, ns = np.asarray(base_success, bool), np.asarray(success, bool)
    row = np.arange(len(bs))
    ba = np.ones(len(row), bool) if base_available is None else np.asarray(base_available, bool)
    na = np.ones(len(row), bool) if available is None else np.asarray(available, bool)
    bsel, nsel = bs[row, base_selected] & ba, ns[row, selected] & na
    out = {'rescued_candidates': int((~bs & ns).sum()), 'lost_candidates': int((bs & ~ns).sum()),
           'rescued_selected_tasks': int((~bsel & nsel).sum()), 'lost_selected_tasks': int((bsel & ~nsel).sum()),
           'candidate_success_delta': task_bootstrap((ns.astype(float)-bs).mean(axis=1), seed),
           'selected_success_delta': task_bootstrap(nsel.astype(float)-bsel, seed),
           'oracle_success_delta': task_bootstrap(ns.any(axis=1).astype(float)-bs.any(axis=1), seed)}
    common = bs & ns
    common_selected = bsel & nsel
    out['common_successful_candidates'] = int(common.sum())
    out['common_successful_selected_tasks'] = int(common_selected.sum())
    for key in quality:
        diff = np.asarray(quality[key]) - np.asarray(base_quality[key])
        counts = common.sum(axis=1)
        td = np.divide(np.where(common, diff, 0).sum(axis=1), counts,
                       out=np.full(len(row), np.nan), where=counts > 0)
        out['common_candidate_' + key + '_delta'] = task_bootstrap(td, seed)
        sd = np.asarray(quality[key])[row, selected] - np.asarray(base_quality[key])[row, base_selected]
        out['common_selected_' + key + '_delta'] = task_bootstrap(sd[common_selected], seed)
        out['same_base_selected_candidate_' + key + '_delta'] = _mean(diff[row, base_selected], common[row, base_selected])
    return out
