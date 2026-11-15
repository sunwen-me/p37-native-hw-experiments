"""Train P37 route-free Flow on occupancy plus ordered progress fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from ordered_field_decoder import rasterize_ordered_fields  # noqa: E402
from p37_model import OrderedDenseFieldFlow, endpoint_fields  # noqa: E402
from p37_sinkhorn import sinkhorn_plan  # noqa: E402
from route_progress_objective import route_progress_increment_cost  # noqa: E402
from safe_field_decoder import (  # noqa: E402
    distance_field_to_soft_clearance,
    safe_ordered_field_to_path,
)


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


class IndexedDataset(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def shape_at(self, index): return self.dataset.shape_at(index)
    def __getitem__(self, index):
        item = dict(self.dataset[index]); item["index"] = index; return item


class MaterializedDataset(Dataset):
    """Cache deterministic dataset transforms before the training loop.

    The source dataset recomputes the distance field and performs its offline
    geometry validation in ``__getitem__``.  Those values are unchanged over
    epochs, so materializing only the tensors consumed by the optimizer keeps
    the training protocol intact while avoiding repeated CPU work.
    """

    _keys = ("map", "distance_field", "start", "goal", "spatial_size")

    def __init__(self, dataset):
        if getattr(dataset, "augment", False):
            raise ValueError("input materialization requires deterministic non-augmented maps")
        task_map_ids = (np.asarray(dataset.map_ids, dtype=np.int64)
                        if dataset.map_ids is not None
                        else np.arange(len(dataset), dtype=np.int64))
        unique_ids, first_rows, inverse = np.unique(
            task_map_ids, return_index=True, return_inverse=True,
        )
        map_values = {key: [] for key in ("map", "distance_field", "spatial_size")}
        for position, _map_id in enumerate(unique_ids):
            sample_id = int(first_rows[position])
            item = dataset[sample_id]
            for key in map_values:
                map_values[key].append(item[key])
            if (position + 1) % 128 == 0 or position + 1 == len(unique_ids):
                print(f"Materialized unique maps: {position + 1}/{len(unique_ids)}", flush=True)

        starts, goals = [], []
        for index, shape in enumerate(dataset.map_shapes):
            height, width = shape
            start = np.asarray(dataset.starts[index], dtype=np.float32)
            goal = np.asarray(dataset.goals[index], dtype=np.float32)
            if dataset.normalize:
                start = dataset.normalize_coordinates(start, height, width)
                goal = dataset.normalize_coordinates(goal, height, width)
            starts.append(torch.from_numpy(np.asarray(start, dtype=np.float32)))
            goals.append(torch.from_numpy(np.asarray(goal, dtype=np.float32)))
        self._map_values = map_values
        self._sample_map_rows = inverse.astype(np.int64)
        self._starts = starts
        self._goals = goals

    def __len__(self):
        return len(self._sample_map_rows)

    def shape_at(self, index):
        size = self._map_values["spatial_size"][self._sample_map_rows[int(index)]]
        return int(size[0]), int(size[1])

    def __getitem__(self, index):
        index = int(index)
        map_row = int(self._sample_map_rows[index])
        item = {key: values[map_row] for key, values in self._map_values.items()}
        item["start"] = self._starts[index]
        item["goal"] = self._goals[index]
        item["index"] = torch.as_tensor(index, dtype=torch.long)
        return item


def map_disjoint_split(maps, val_fraction, seed):
    hashes = getattr(maps, "map_hashes", None)
    if hashes is None:
        hashes = np.asarray([
            hashlib.sha256(np.ascontiguousarray(np.asarray(m)).tobytes()).hexdigest()
            for m in maps
        ])
    else:
        hashes = np.asarray(hashes)
    unique = np.unique(hashes)
    rng = np.random.default_rng(seed); rng.shuffle(unique)
    val_count = max(1, int(round(val_fraction * len(unique))))
    val_hashes = set(unique[:val_count])
    val_mask = np.asarray([h in val_hashes for h in hashes])
    train_ids, val_ids = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    if not len(train_ids) or not len(val_ids):
        raise ValueError("map split produced an empty partition")
    return train_ids, val_ids, hashes


def load_map_disjoint_split(maps, path):
    """Load an inherited partition without changing validation-map ownership."""
    with np.load(path, allow_pickle=False) as archive:
        train_ids, val_ids = archive["train_ids"], archive["val_ids"]
        for ids in (train_ids, val_ids):
            if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0
                    or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(maps)):
                raise ValueError("split IDs must be unique, nonempty, in-range integers")
        if len(np.intersect1d(train_ids, val_ids)) or not np.array_equal(
            np.sort(np.concatenate([train_ids, val_ids])), np.arange(len(maps)),
        ):
            raise ValueError("split must partition all tasks exactly once")
        hashes = getattr(maps, "map_hashes", None)
        if hashes is None:
            hashes = np.asarray([
                hashlib.sha256(np.ascontiguousarray(np.asarray(item)).tobytes()).hexdigest()
                for item in maps
            ])
        else:
            hashes = np.asarray(hashes)
        if "training_map_hashes" not in archive or not np.array_equal(
            hashes, archive["training_map_hashes"],
        ):
            raise ValueError("split map hashes do not match data order")
        if set(hashes[train_ids]) & set(hashes[val_ids]):
            raise ValueError("training and validation maps overlap")
        if "geometry_val_ids" in archive:
            ids = archive["geometry_val_ids"]
            if (ids.ndim != 1 or ids.dtype.kind not in "iu"
                    or len(np.unique(ids)) != len(ids) or not np.isin(ids, val_ids).all()):
                raise ValueError("geometry validation IDs must be unique members of validation")
    return train_ids, val_ids, hashes


def balanced_geometry_sample(dataset, candidates, count, generator):
    """Select an approximately equal geometry-validation count per H/W."""
    candidates = np.asarray(candidates, dtype=np.int64)
    grouped = {}
    for task_id in candidates:
        grouped.setdefault(dataset.shape_at(int(task_id)), []).append(int(task_id))
    if not grouped or count <= 0:
        return np.asarray([], dtype=np.int64)
    shapes = sorted(grouped)
    for values in grouped.values():
        order = generator.permutation(len(values))
        grouped_values = [values[int(index)] for index in order]
        values[:] = grouped_values
    count = min(int(count), len(candidates))
    quota, _ = divmod(count, len(shapes))
    allocation = {shape: min(quota, len(grouped[shape])) for shape in shapes}
    remaining = count - sum(allocation.values())
    while remaining:
        progressed = False
        for shape in shapes:
            if allocation[shape] < len(grouped[shape]):
                allocation[shape] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    selected = [
        task_id for shape in shapes
        for task_id in grouped[shape][:allocation[shape]]
    ]
    return np.asarray(selected, dtype=np.int64)


def add_hard_replay_ids(train_ids, replay_ids, factor):
    """Repeat mined failures without crossing the map-disjoint split."""
    if factor < 1:
        raise ValueError("hard replay factor must be >= 1")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    replay_ids = np.asarray(replay_ids, dtype=np.int64).reshape(-1)
    hard = np.intersect1d(train_ids, replay_ids, assume_unique=False)
    if not len(hard) or factor == 1:
        return train_ids.copy(), hard
    return np.concatenate([train_ids, np.tile(hard, int(factor) - 1)]), hard


def ordered_targets(routes, starts, goals, thickness, height, width, subpixel=False,
                    rasterization_mode="points"):
    occupancy, progress, _ = rasterize_ordered_fields(
        routes, thickness=thickness, subpixel=subpixel, height=height, width=width,
        mode=rasterization_mode,
    )
    target = torch.stack([occupancy, progress], dim=2)
    fixed = endpoint_fields(starts, goals, height, width)
    endpoint = fixed[:, :1]
    return target * (1.0 - endpoint[:, None]) + fixed[:, None]


def apply_clearance_target_gate(
    target: torch.Tensor,
    clearance: torch.Tensor,
    fixed: torch.Tensor,
    radius: float,
    margin: float,
    temperature: float,
    floor: float,
    distance_scale: float = 8.0,
    clearance_offset: float = 2 ** -0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softly suppress route-field targets inside the footprint margin.

    ``target`` has shape ``[N,2,H,W]`` and ``clearance`` has shape
    ``[N,1,H,W]``.  The gate is deliberately soft: it keeps narrow but valid
    passages trainable while moving the field mass toward pixels with a
    clearance margin.  Fixed endpoint pixels are restored afterwards.
    """
    if target.ndim != 4 or target.shape[1] != 2:
        raise ValueError("target must have shape [N,2,H,W]")
    if clearance.shape != target[:, :1].shape:
        raise ValueError("clearance must have shape [N,1,H,W]")
    if fixed.shape != target.shape:
        raise ValueError("fixed must match target")
    if (radius < 0 or margin < 0 or temperature <= 0 or
            distance_scale <= 0 or clearance_offset < 0 or
            not 0.0 <= floor < 1.0):
        raise ValueError("invalid clearance target gate parameters")
    score = (
        clearance * float(distance_scale) - float(clearance_offset)
        - (float(radius) + float(margin))
    )
    gate = float(floor) + (1.0 - float(floor)) * torch.sigmoid(score / float(temperature))
    # Keep ordered progress amplitudes intact.  Suppressing progress near an
    # obstacle can create a missing progress slice and turn a valid route into
    # a decoder-invalid candidate; occupancy is the channel that should carry
    # the clearance preference.
    gated = target.clone()
    gated[:, :1] = gated[:, :1] * gate
    endpoint = fixed[:, :1]
    gated = gated * (1.0 - endpoint) + fixed
    return gated, gate


def field_loss_weights(
    target: torch.Tensor,
    field_positive_weight: float,
    progress_support_weight: float,
    progress_background_weight: float,
    tail_weight: float,
    tail_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return occupancy/progress weights and the supervised tail support."""
    if target.ndim != 4 or target.shape[1] != 2:
        raise ValueError("target must have shape [N,2,H,W]")
    if tail_weight < 0 or not 0.0 < tail_fraction <= 1.0:
        raise ValueError("invalid tail supervision parameters")
    corridor = target[:, 0].detach().clamp(0.0, 1.0)
    progress = target[:, 1].detach().clamp(0.0, 1.0)
    tail = corridor * (progress >= (1.0 - float(tail_fraction))).to(corridor.dtype)
    occupancy_weights = 1.0 + float(field_positive_weight) * corridor
    progress_weights = float(progress_background_weight) + float(progress_support_weight) * corridor
    if tail_weight:
        occupancy_weights = occupancy_weights + float(tail_weight) * tail
        progress_weights = progress_weights + float(tail_weight) * tail
    return occupancy_weights, progress_weights, tail


def reduce_field_error(squared, weights, corridor, reduction="pixel_mean",
                       background_weight=1.0):
    """Keep route supervision from vanishing in large background regions.

    Legacy reduction is retained for old experiments. Support-balanced mode
    normalizes foreground/background separately and then combines them;
    absent regions contribute neither a loss nor a denominator term.
    """
    if reduction == "pixel_mean":
        return (squared * weights).sum((-2, -1)) / weights.sum((-2, -1)).clamp_min(1e-6)
    if reduction != "support_balanced":
        raise ValueError("unknown field loss reduction")
    support = (corridor > 0).to(weights.dtype)
    result = squared.new_zeros(squared.shape[0])
    mass = torch.zeros_like(result)
    for mask, coefficient in ((support, 1.0), (1.0 - support, background_weight)):
        region_weights = weights * mask
        denominator = region_weights.sum((-2, -1))
        active = (denominator > 0).to(squared.dtype) * coefficient
        result = result + active * (
            (squared * region_weights).sum((-2, -1)) / denominator.clamp_min(1e-6)
        )
        mass = mass + active
    return result / mass.clamp_min(1e-6)


def pairwise_cfm_cost(model, maps, starts, goals, routes, latent,
                      mask_thickness, field_positive_weight=0.0,
                      progress_support_weight=4.0,
                      progress_background_weight=0.1,
                      endpoint_approach_weight=0.0,
                      endpoint_bandwidth=0.12,
                      endpoint_tail_weight=6.0,
                      endpoint_tail_fraction=0.125,
                      target_clearance_margin=0.25,
                      target_clearance_temperature=0.25,
                      target_clearance_floor=0.15,
                      share_source_across_latents=False,
                      generator=None, subpixel=False,
                      rasterization_mode="points", field_loss_reduction="pixel_mean",
                      route_increment_weight=0.0, route_increment_samples=128,
                      route_increment_scale=32.0):
    batch, teacher_count = routes.shape[:2]
    route_count = latent.shape[1]
    height, width = maps.shape[-2:]
    targets = ordered_targets(
        routes, starts, goals, mask_thickness, height, width, subpixel=subpixel,
        rasterization_mode=rasterization_mode,
    )
    # For a fixed latent row, every route comparison must use the same CFM
    # source and time.  Otherwise Sinkhorn can match routes based on sampled
    # noise or t instead of route compatibility.
    source_rows = 1 if share_source_across_latents else route_count
    x0 = torch.randn(
        batch, source_rows, 1, 2, height, width,
        device=maps.device, dtype=maps.dtype,
        generator=generator,
    )
    pair_start = starts[:, None].expand(-1, route_count, -1)
    pair_goal = goals[:, None].expand(-1, route_count, -1)
    fixed = endpoint_fields(
        pair_start.reshape(-1, 2), pair_goal.reshape(-1, 2), height, width,
    ).reshape(batch, route_count, 1, 2, height, width)
    endpoint = fixed[:, :, :, :1]
    x0 = (x0.expand(batch, route_count, 1, 2, height, width)
          * (1.0 - endpoint) + fixed).expand(
        -1, -1, teacher_count, -1, -1, -1,
    )
    t = torch.rand(batch, source_rows, 1, device=maps.device,
                   dtype=maps.dtype, generator=generator)
    t = t.expand(batch, route_count, 1)
    t = t.expand(-1, -1, teacher_count)
    target = targets[:, None].expand(-1, route_count, -1, -1, -1, -1)
    flat_maps = maps[:, None, None].expand(-1, route_count, teacher_count, -1, -1, -1).reshape(
        batch * route_count * teacher_count, *maps.shape[1:]
    )
    flat_start = starts[:, None, None].expand(-1, route_count, teacher_count, -1).reshape(-1, 2)
    flat_goal = goals[:, None, None].expand(-1, route_count, teacher_count, -1).reshape(-1, 2)
    flat_latent = latent[:, :, None].expand(-1, -1, teacher_count, -1).reshape(
        batch * route_count * teacher_count, model.latent_dim
    )
    if target_clearance_margin > 0.0:
        flat_target = target.reshape(-1, 2, height, width)
        flat_fixed = fixed.expand(-1, -1, teacher_count, -1, -1, -1).reshape(-1, 2, height, width)
        flat_clearance = flat_maps[:, 1:2]
        flat_target, _ = apply_clearance_target_gate(
            flat_target, flat_clearance, flat_fixed,
            radius=model.robot_radius_grid,
            margin=target_clearance_margin,
            temperature=target_clearance_temperature,
            floor=target_clearance_floor,
            distance_scale=model.distance_scale,
            clearance_offset=model.clearance_offset,
        )
        target = flat_target.reshape(batch, route_count, teacher_count, 2, height, width)
    xt = (1.0 - t[..., None, None, None]) * x0 + t[..., None, None, None] * target
    velocity_target = target - x0
    velocity = model.vector_field(
        xt.reshape(batch * route_count * teacher_count, 2, height, width),
        flat_maps, flat_start, flat_goal, t.reshape(-1), flat_latent,
    )
    squared = (velocity - velocity_target.reshape_as(velocity)).square()
    # Occupancy remains a full-image support objective.  Progress has no
    # semantic target off the route, so give corridor pixels most of its
    # supervision while retaining a small background penalty that suppresses
    # arbitrary progress noise.
    flat_target = target.reshape(-1, 2, height, width)
    corridor = flat_target[:, 0].detach().clamp(0.0, 1.0)
    occupancy_weights, progress_weights, _ = field_loss_weights(
        flat_target,
        field_positive_weight=field_positive_weight,
        progress_support_weight=progress_support_weight,
        progress_background_weight=progress_background_weight,
        tail_weight=endpoint_tail_weight,
        tail_fraction=endpoint_tail_fraction,
    )
    # Preserve the legacy reduction order for reproducibility. The corrected
    # balanced objective also applies approach weighting to occupancy below.
    occupancy_loss = reduce_field_error(
        squared[:, 0], occupancy_weights, corridor, field_loss_reduction,
    )
    if endpoint_approach_weight > 0.0:
        if endpoint_bandwidth <= 0.0:
            raise ValueError("endpoint_bandwidth must be positive")
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=maps.device, dtype=maps.dtype),
            torch.linspace(-1.0, 1.0, width, device=maps.device, dtype=maps.dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1)
        metric_scale = maps.new_tensor([(width - 1) / 63.0, (height - 1) / 63.0])
        start_delta = (grid[None] - starts[:, None, None]) * metric_scale
        goal_delta = (grid[None] - goals[:, None, None]) * metric_scale
        start_heat = torch.exp(-(start_delta.square().sum(-1)) /
                               (2.0 * endpoint_bandwidth ** 2))
        goal_heat = torch.exp(-(goal_delta.square().sum(-1)) /
                              (2.0 * endpoint_bandwidth ** 2))
        endpoint_heat = (start_heat + goal_heat).clamp_max(1.0)
        endpoint_heat = endpoint_heat[:, None, None].expand(
            -1, route_count, teacher_count, -1, -1,
        ).reshape_as(corridor)
        endpoint_weights = 1.0 + endpoint_approach_weight * endpoint_heat * corridor
        occupancy_weights = occupancy_weights * endpoint_weights
        progress_weights = progress_weights * endpoint_weights
    if field_loss_reduction == "support_balanced" and endpoint_approach_weight > 0:
        occupancy_loss = reduce_field_error(
            squared[:, 0], occupancy_weights, corridor, field_loss_reduction,
        )
    progress_loss = reduce_field_error(
        squared[:, 1], progress_weights, corridor, field_loss_reduction,
        background_weight=progress_background_weight,
    )
    pair_loss = occupancy_loss + progress_loss
    if route_increment_weight < 0:
        raise ValueError("route_increment_weight must be nonnegative")
    if route_increment_weight > 0:
        flat_t = t.reshape(-1, 1, 1, 1)
        endpoint_estimate = xt.reshape_as(velocity) + (1 - flat_t) * velocity
        paired_routes = routes[:, None].expand(-1, route_count, -1, -1, -1).reshape(
            -1, *routes.shape[-2:])
        increment_loss = route_progress_increment_cost(
            endpoint_estimate, paired_routes, samples=route_increment_samples,
            scale=route_increment_scale,
        )
        pair_loss = pair_loss + route_increment_weight * increment_loss
    return pair_loss.reshape(batch, route_count, teacher_count)


def latent_diversity_penalty(model, maps, starts, goals, latent, margin, generator):
    """Permutation-invariant separation of latent-conditioned vector fields."""
    if margin <= 0.0:
        return torch.zeros((), device=maps.device, dtype=maps.dtype)
    batch, route_count, _ = latent.shape
    height, width = maps.shape[-2:]
    x0 = torch.randn(
        batch, 2, height, width, device=maps.device, dtype=maps.dtype,
        generator=generator,
    )
    fixed = endpoint_fields(starts, goals, height, width)
    x0 = x0 * (1.0 - fixed[:, :1]) + fixed
    t = torch.rand(batch, device=maps.device, dtype=maps.dtype, generator=generator)
    flat_maps = maps[:, None].expand(-1, route_count, -1, -1, -1).reshape(
        batch * route_count, *maps.shape[1:]
    )
    flat_start = starts[:, None].expand(-1, route_count, -1).reshape(-1, 2)
    flat_goal = goals[:, None].expand(-1, route_count, -1).reshape(-1, 2)
    flat_x = x0[:, None].expand(-1, route_count, -1, -1, -1).reshape(
        batch * route_count, 2, height, width
    )
    flat_t = t[:, None].expand(-1, route_count).reshape(-1)
    velocity = model.vector_field(
        flat_x, flat_maps, flat_start, flat_goal, flat_t,
        latent.reshape(batch * route_count, -1),
    ).reshape(batch, route_count, -1)
    distances = torch.cdist(velocity, velocity, p=1) / float(velocity.shape[-1])
    upper = torch.triu(
        torch.ones(route_count, route_count, device=maps.device, dtype=torch.bool), 1,
    )[None]
    return (float(margin) - distances)[upper.expand(batch, -1, -1)].clamp_min(0.0).mean()


def sinkhorn_cfm_loss(model, maps, starts, goals, routes, valid, args,
                      generator=None, latent_prototypes=None):
    if (routes.ndim != 4 or routes.shape[-1] != 2 or valid.shape != routes.shape[:2]
            or not torch.isfinite(valid).all() or not torch.all((valid == 0) | (valid == 1))
            or not torch.all(valid.sum(dim=1) > 0)):
        raise ValueError("P37 requires a binary valid mask and at least one teacher per task")
    if not torch.isfinite(routes[valid.bool()]).all():
        raise ValueError("valid teachers must be finite")
    batch, route_count = valid.shape
    if latent_prototypes is None:
        latent = torch.randn(
            batch, route_count, model.latent_dim, device=maps.device,
            dtype=maps.dtype, generator=generator,
        )
    else:
        if tuple(latent_prototypes.shape) != (route_count, model.latent_dim):
            raise ValueError("latent_prototypes must match route count and latent dim")
        latent = latent_prototypes.to(device=maps.device, dtype=maps.dtype)[None].expand(
            batch, -1, -1,
        )
    # Pack only real teachers. Each latent has mass 1/R and each of the M
    # available teachers has mass 1/M; padding never becomes a Flow target.
    teacher_counts = valid.sum(dim=1).long()
    loss = maps.new_zeros(())
    for teacher_count in torch.unique(teacher_counts).tolist():
        ids = torch.nonzero(teacher_counts == teacher_count, as_tuple=True)[0]
        group_routes = routes[ids][valid[ids].bool()].reshape(
            len(ids), teacher_count, *routes.shape[2:],
        )
        costs = pairwise_cfm_cost(
            model, maps[ids], starts[ids], goals[ids], group_routes, latent[ids], args.mask_thickness,
            field_positive_weight=args.field_positive_weight,
            progress_support_weight=args.progress_support_weight,
            progress_background_weight=args.progress_background_weight,
            endpoint_approach_weight=args.endpoint_approach_weight,
            endpoint_bandwidth=args.endpoint_bandwidth,
            endpoint_tail_weight=args.endpoint_tail_weight,
            endpoint_tail_fraction=args.endpoint_tail_fraction,
            target_clearance_margin=args.target_clearance_margin,
            target_clearance_temperature=args.target_clearance_temperature,
            target_clearance_floor=args.target_clearance_floor,
            share_source_across_latents=args.share_source_across_latents,
            generator=generator,
            subpixel=args.subpixel_rasterization,
            rasterization_mode=getattr(args, "rasterization_mode", "points"),
            field_loss_reduction=getattr(args, "field_loss_reduction", "pixel_mean"),
            route_increment_weight=getattr(args, "route_increment_weight", 0.0),
            route_increment_samples=getattr(args, "route_increment_samples", 128),
            route_increment_scale=getattr(args, "route_increment_scale", 32.0),
        )
        plan = sinkhorn_plan(costs.detach(), args.sinkhorn_epsilon, args.sinkhorn_iterations)
        group_loss = (plan * costs).sum(dim=(1, 2)).mean()
        if args.latent_diversity_weight > 0.0 and teacher_count > 1:
            group_loss = group_loss + args.latent_diversity_weight * latent_diversity_penalty(
                model, maps[ids], starts[ids], goals[ids], latent[ids, :teacher_count],
                args.latent_diversity_margin, generator,
            )
        loss = loss + group_loss * (len(ids) / batch)
    return loss


def run_epoch(model, loader, labels, device, args, optimizer=None,
              generator=None, latent_prototypes=None):
    training = optimizer is not None; total = 0.0; count = 0
    model.train(training)
    for batch_index, batch in enumerate(loader, start=1):
        idx = batch["index"].long()
        maps = torch.cat([batch["map"], batch["distance_field"]], dim=1).to(device)
        starts, goals = batch["start"].to(device), batch["goal"].to(device)
        routes = labels["fine_routes"][idx].to(device)
        valid = labels["valid"][idx].to(device)
        loss = sinkhorn_cfm_loss(
            model, maps, starts, goals, routes, valid, args,
            generator=generator, latent_prototypes=latent_prototypes,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CFM loss; optimizer step skipped")
        if training:
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite CFM gradient; optimizer step skipped")
            optimizer.step()
        batch_count = int(len(idx))
        total += float(loss.detach()) * batch_count
        count += batch_count
        if (args.progress_every_batches > 0
                and batch_index % args.progress_every_batches == 0):
            print(json.dumps({
                "stage": "train_batch" if training else "validation_batch",
                "batch": batch_index, "batches": len(loader),
                "samples": count, "mean_loss": total / max(count, 1),
            }), flush=True)
    return total / max(count, 1)


def _geometry_summary(stats):
    def mean(key):
        return float(np.mean(stats[key])) if stats[key] else 0.0

    raw_oracle = np.asarray(stats["task_successes"], dtype=bool)
    raw_selected = np.asarray(stats["selected_successes"], dtype=bool)
    accepted_oracle = np.asarray(stats["accepted_task_successes"], dtype=bool)
    deployed = np.asarray(stats["deployment_successes"], dtype=bool)
    return {
        "tasks": len(raw_oracle),
        "candidate_success_rate": mean("candidate_successes"),
        "oracle_best_of_k_success_rate": mean("task_successes"),
        "selected_path_success_rate": mean("selected_successes"),
        "selector_miss_rate": float(np.mean(raw_oracle & ~raw_selected)) if len(raw_oracle) else 0.0,
        "decoder_valid_rate": mean("decoder_valids"),
        "accepted_candidate_rate": mean("accepted_candidates"),
        "accepted_safe_candidate_rate": mean("accepted_safe_candidates"),
        "accepted_oracle_best_of_k_success_rate": mean("accepted_task_successes"),
        "deployment_success_rate": mean("deployment_successes"),
        "deployment_selector_miss_rate": float(np.mean(accepted_oracle & ~deployed)) if len(deployed) else 0.0,
        "abstention_rate": mean("abstentions"),
        "selected_accepted_unsafe_rate": mean("selected_accepted_unsafe"),
    }


@torch.no_grad()
def geometry_success(model, base, task_ids, device, args):
    """Score the exact deployment sampler on a fixed validation subset."""
    from scripts.geometry_eval import score_task

    model.eval()
    task_ids = list(task_ids)
    if args.geometry_val_max_tasks > 0:
        task_ids = balanced_geometry_sample(
            base, task_ids, args.geometry_val_max_tasks,
            np.random.default_rng(args.seed + 600_007),
        ).tolist()
    generator = torch.Generator(device=device).manual_seed(args.seed + 200_003)
    totals = {key: [] for key in (
        "candidate_successes", "task_successes", "selected_successes", "decoder_valids",
        "accepted_candidates", "accepted_safe_candidates", "accepted_task_successes",
        "deployment_successes", "abstentions", "selected_accepted_unsafe",
    )}
    shape_stats = {}
    shape_buckets = {}
    for task_id in task_ids:
        shape_buckets.setdefault(base.shape_at(int(task_id)), []).append(int(task_id))
    ordered_ids = [np.asarray(values, dtype=np.int64)
                   for values in shape_buckets.values()]
    for bucket_ids in ordered_ids:
      height, width = base.shape_at(int(bucket_ids[0]))
      batch_size = args.geometry_val_batch_size
      if args.geometry_val_max_pixels > 0:
          pixels_per_task = height * width * args.geometry_val_candidates
          batch_size = min(batch_size, max(1, args.geometry_val_max_pixels // pixels_per_task))
      for begin in range(0, len(bucket_ids), batch_size):
        ids = bucket_ids[begin:begin + batch_size]
        batch = [base[i] for i in ids]
        maps = torch.stack([
            torch.cat([item["map"], item["distance_field"]], dim=0)
            for item in batch
        ]).to(device)
        starts = torch.stack([item["start"] for item in batch]).to(device)
        goals = torch.stack([item["goal"] for item in batch]).to(device)
        if base.geometry is None:
            raise ValueError("geometry validation requires geometry_json")
        model.distance_scale = float(args.distance_scale)
        model.robot_radius_grid = float(base.geometry.radius_grid)
        model.clearance_temperature = float(args.geometry_val_clearance_temperature)
        model.clearance_offset = float(args.geometry_val_clearance_offset)
        model.decode_clearance_power = float(args.geometry_val_clearance_power)
        model.decode_continuity_beta = float(args.geometry_val_continuity_beta)
        output = model.sample(
            maps, starts, goals, candidates=args.geometry_val_candidates,
            num_steps=args.geometry_val_steps,
            noise_scale=args.geometry_val_noise_scale,
            generator=generator,
        )
        paths = base.denormalize_paths(
            output["paths"].cpu().numpy(), task_ids=ids,
        )
        selected = output["selected_index"].cpu().numpy()
        valid_np = output["decoder_valid"].cpu().numpy()
        accepted_np = output["accepted"].cpu().numpy()
        shape_key = f"{height}x{width}"
        current = shape_stats.setdefault(shape_key, {key: [] for key in totals})
        for local, task_id in enumerate(ids):
            safe = score_task((
                str(HERE.parent / "map_flow_matching"),
                paths[local], base.maps[task_id], base.starts[task_id],
                base.goals[task_id], base.geometry.to_json(),
            ))
            safe = np.asarray(safe, dtype=bool)
            accepted = np.asarray(accepted_np[local], dtype=bool)
            chosen = int(selected[local])
            outcomes = {
                "candidate_successes": safe.tolist(),
                "task_successes": [bool(safe.any())],
                "selected_successes": [bool(safe[chosen])],
                "decoder_valids": valid_np[local].tolist(),
                "accepted_candidates": accepted.tolist(),
                "accepted_safe_candidates": (accepted & safe).tolist(),
                "accepted_task_successes": [bool((accepted & safe).any())],
                "deployment_successes": [bool(accepted[chosen] and safe[chosen])],
                "abstentions": [bool(not accepted.any())],
                "selected_accepted_unsafe": [bool(accepted[chosen] and not safe[chosen])],
            }
            for key, values in outcomes.items():
                totals[key].extend(values)
                current[key].extend(values)
    return {**_geometry_summary(totals),
            "by_shape": {shape: _geometry_summary(stats) for shape, stats in shape_stats.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--route-labels", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-path", default=None, help="Inherit audited map-disjoint train/val IDs.")
    parser.add_argument("--mask-thickness", type=int, default=1)
    parser.add_argument(
        "--decode-horizon", type=int, default=512,
        help="Output path points, independent of teacher sample count; use 128 for legacy runs.",
    )
    parser.add_argument(
        "--rasterization-mode", choices=("points", "segments"), default="segments",
        help="Continuous segment labels by default; points reproduces legacy labels.",
    )
    parser.add_argument(
        "--field-loss-reduction", choices=("pixel_mean", "support_balanced"),
        default="support_balanced", help="Normalize route/background error independently.",
    )
    parser.add_argument(
        "--subpixel-rasterization", action="store_true",
        help="Use bilinear route splatting; disabled by default pending oracle validation.",
    )
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument(
        "--latent-scale", type=float, default=1.0,
        help="Scale global latent conditioning in the vector field.",
    )
    parser.add_argument(
        "--fixed-latent-prototypes", action="store_true",
        help="Use one fixed latent prototype per route slot throughout training.",
    )
    parser.add_argument("--latent-diversity-weight", type=float, default=0.0)
    parser.add_argument("--latent-diversity-margin", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--architecture", choices=("cnn", "transformer"), default="cnn",
        help="P37 vector-field backbone; CNN preserves legacy checkpoint layout.",
    )
    parser.add_argument(
        "--transformer-window-size", type=int, default=8,
        help="Local attention window for the Transformer backbone.",
    )
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=30)
    parser.add_argument(
        "--field-positive-weight", type=float, default=8.0,
        help="Extra occupancy weight on route-supported pixels.",
    )
    parser.add_argument("--progress-support-weight", type=float, default=4.0)
    parser.add_argument("--progress-background-weight", type=float, default=0.1)
    parser.add_argument(
        "--endpoint-tail-weight", type=float, default=6.0,
        help="Extra occupancy/progress supervision for the final route tail.",
    )
    parser.add_argument(
        "--endpoint-tail-fraction", type=float, default=0.125,
        help="Fraction of ordered progress covered by tail supervision.",
    )
    parser.add_argument(
        "--endpoint-approach-weight", type=float, default=0.0,
        help="Extra CFM weight around start/goal route support.",
    )
    parser.add_argument("--endpoint-bandwidth", type=float, default=0.12)
    parser.add_argument(
        "--target-clearance-margin", type=float, default=0.25,
        help="Soft target-field clearance margin in grid pixels; <=0 disables it.",
    )
    parser.add_argument("--target-clearance-temperature", type=float, default=0.25)
    parser.add_argument(
        "--target-clearance-floor", type=float, default=0.35,
        help="Minimum target occupancy retained by the clearance gate.",
    )
    parser.add_argument("--decode-beta", type=float, default=1280.0)
    parser.add_argument(
        "--decoder-mode", choices=("legacy", "scale_aware"), default="scale_aware",
        help="Adapt progress precision to supported field geometry; retain safety gates.",
    )
    parser.add_argument("--decode-power", type=float, default=4.0)
    parser.add_argument(
        "--decode-progress-smoothing", type=int, default=0,
        help="Predicted-support weighted progress kernel; 0 preserves unsmoothed decoding, 7 enables the validated repair.",
    )
    parser.add_argument("--decode-threshold", type=float, default=0.1)
    parser.add_argument("--distance-scale", type=float, default=8.0)
    parser.add_argument("--share-source-across-latents", action="store_true")
    parser.add_argument("--min-teacher-clearance", type=float, default=None)
    parser.add_argument(
        "--hard-replay-ids", default=None,
        help="NPZ containing train-task ids mined as all-candidates-unsafe.",
    )
    parser.add_argument("--hard-replay-factor", type=int, default=1)
    parser.add_argument("--geometry-val-max-tasks", type=int, default=256)
    parser.add_argument("--geometry-val-batch-size", type=int, default=16)
    parser.add_argument(
        "--geometry-val-max-pixels", type=int, default=4194304,
        help="Cap H*W*batch*candidates in geometry validation; 0 disables it.",
    )
    parser.add_argument(
        "--max-pixels-per-batch", type=int, default=1572864,
        help="Adaptive training batch cap based on H*W; 0 uses --batch-size unchanged.",
    )
    parser.add_argument("--geometry-val-candidates", type=int, default=4)
    parser.add_argument("--geometry-val-steps", type=int, default=100)
    parser.add_argument(
        "--geometry-val-every", type=int, default=1,
        help="Run expensive deployment geometry validation every N epochs (also epoch 1 and final epoch).",
    )
    parser.add_argument(
        "--progress-every-batches", type=int, default=100,
        help="Print a compact training progress line every N batches; 0 disables.",
    )
    parser.add_argument("--geometry-val-noise-scale", type=float, default=1.0)
    parser.add_argument("--geometry-val-clearance-temperature", type=float, default=0.5)
    parser.add_argument(
        "--geometry-val-clearance-offset", type=float, default=2 ** -0.5,
        help="Conservative cell-center to closed-cell distance offset.",
    )
    parser.add_argument("--geometry-val-clearance-power", type=float, default=2.0)
    parser.add_argument("--geometry-val-continuity-beta", type=float, default=60.0)
    parser.add_argument("--decode-terminal-support-window", type=int, default=8)
    parser.add_argument("--decode-terminal-support-fraction", type=float, default=0.75)
    parser.add_argument("--decode-max-terminal-jump", type=float, default=0.5)
    parser.add_argument(
        "--decode-max-step-grid", type=float, default=None,
        help="Optional max decoded step in grid pixels for support validity.",
    )
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument(
        "--precompute-inputs", action="store_true",
        help="Materialize deterministic map features once before training.",
    )
    parser.add_argument(
        "--trusted-safety-audit", default=None,
        help="Use a matching full safety_audit.npz to skip repeated per-sample geometry checks and cache fields by map.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.decode_horizon < 2:
        parser.error("--decode-horizon must be >= 2")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    sys.path.insert(0, str(Path(args.source_project).resolve()))
    from utils.dataset import PathPlanningDataset
    raw = np.load(args.route_labels, allow_pickle=False)
    labels = {k: torch.from_numpy(raw[k]).float() for k in ("fine_routes", "valid")}
    if args.min_teacher_clearance is not None:
        if "route_point_clearance" not in raw:
            raise ValueError("clearance-filtered training requires route_point_clearance labels")
        route_clearance = np.asarray(raw["route_point_clearance"])
        eligible = np.all((route_clearance >= args.min_teacher_clearance) | (raw["valid"] == 0), axis=1)
    else:
        eligible = np.ones(len(labels["fine_routes"]), dtype=bool)
    base = PathPlanningDataset(
        args.data_path, horizon=labels["fine_routes"].shape[2], normalize=True,
        augment=False, distance_scale=args.distance_scale,
        validate_geometry=args.trusted_safety_audit is None,
        cache_distance_fields=args.trusted_safety_audit is not None,
    )
    dataset = MaterializedDataset(base) if args.precompute_inputs else IndexedDataset(base)
    if len(dataset) != len(labels["fine_routes"]): raise ValueError("route labels and dataset length differ")
    if args.split_path:
        train_ids, val_ids, hashes = load_map_disjoint_split(base.maps, args.split_path)
    else:
        train_ids, val_ids, hashes = map_disjoint_split(base.maps, args.val_fraction, args.seed)
    if args.trusted_safety_audit is not None:
        with np.load(args.trusted_safety_audit, allow_pickle=False) as safety:
            required = {"map_hashes", "safe", "minimum_clearance_lower_bound"}
            if not required.issubset(safety.files):
                raise ValueError("trusted safety audit lacks required fields")
            if (len(safety["safe"]) != len(base)
                    or not np.array_equal(safety["map_hashes"], hashes)
                    or not np.all(safety["safe"])
                    or not np.isfinite(safety["minimum_clearance_lower_bound"]).all()
                    or not np.all(safety["minimum_clearance_lower_bound"] > base.geometry.radius_grid + 1e-10)):
                raise ValueError("trusted safety audit does not match every training task")
    train_ids = train_ids[eligible[train_ids]]
    val_ids = val_ids[eligible[val_ids]]
    if not len(train_ids) or not len(val_ids):
        raise ValueError("clearance filtering produced an empty split")
    geometry_rng = np.random.default_rng(args.seed + 400_003)
    geometry_count = len(val_ids) if args.geometry_val_max_tasks <= 0 else min(
        len(val_ids), args.geometry_val_max_tasks
    )
    if args.split_path:
        with np.load(args.split_path, allow_pickle=False) as inherited:
            saved_geometry = inherited["geometry_val_ids"] if "geometry_val_ids" in inherited else np.asarray([], dtype=np.int64)
        geometry_val_ids = saved_geometry[np.isin(saved_geometry, val_ids)]
    else:
        geometry_val_ids = np.asarray([], dtype=np.int64)
    if not len(geometry_val_ids):
        geometry_val_ids = balanced_geometry_sample(
            dataset, val_ids, geometry_count, geometry_rng,
        )
    replay_ids = np.asarray([], dtype=np.int64)
    train_loader_ids = train_ids
    if args.hard_replay_ids:
        replay_archive = np.load(args.hard_replay_ids, allow_pickle=False)
        if "ids" not in replay_archive:
            raise ValueError("hard replay archive must contain an ids array")
        train_loader_ids, replay_ids = add_hard_replay_ids(
            train_ids, replay_archive["ids"], args.hard_replay_factor,
        )
    from utils.dataset import ShapeBucketBatchSampler
    train_loader = DataLoader(
        dataset,
        batch_sampler=ShapeBucketBatchSampler(
            dataset, args.batch_size, indices=train_loader_ids, shuffle=True,
            max_pixels_per_batch=(args.max_pixels_per_batch or None),
        ),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        dataset,
        batch_sampler=ShapeBucketBatchSampler(
            dataset, args.batch_size, indices=val_ids, shuffle=False,
            max_pixels_per_batch=(args.max_pixels_per_batch or None),
        ),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    if base.geometry is None:
        raise ValueError("geometry-aware data is required")
    model = OrderedDenseFieldFlow(
                                  map_height=base.map_height, map_width=base.map_width,
                                  horizon=args.decode_horizon,
                                  latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                                  decode_beta=args.decode_beta, decode_power=args.decode_power,
                                  decode_threshold=args.decode_threshold,
                                  distance_scale=args.distance_scale,
                                  robot_radius_grid=base.geometry.radius_grid,
                                  clearance_temperature=args.geometry_val_clearance_temperature,
                                  clearance_offset=args.geometry_val_clearance_offset,
                                  decode_clearance_power=args.geometry_val_clearance_power,
                                  decode_continuity_beta=args.geometry_val_continuity_beta,
                                  decode_terminal_support_window=args.decode_terminal_support_window,
                                  decode_terminal_support_fraction=args.decode_terminal_support_fraction,
                                  decode_max_terminal_jump=args.decode_max_terminal_jump,
                                  decode_max_step_grid=args.decode_max_step_grid,
                                  decoder_mode=args.decoder_mode,
                                  decode_progress_smoothing=args.decode_progress_smoothing,
                                  architecture=args.architecture,
                                  transformer_window_size=args.transformer_window_size).to(device)
    source_model_config=None;source_checkpoint_sha256=None
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        source_model_config = dict(state.get("config", {}))
        source_architecture = str(source_model_config.get("architecture", "cnn")).strip().lower()
        if source_architecture == "transforms":
            source_architecture = "transformer"
        if source_architecture != model.architecture:
            raise ValueError(
                f"init checkpoint architecture {source_architecture!r} does not match "
                f"requested architecture {model.architecture!r}"
            )
        if model.architecture == "transformer" and int(
            source_model_config.get("transformer_window_size", 8)
        ) != model.transformer_window_size:
            raise ValueError("init checkpoint transformer window size does not match")
        model.load_state_dict(state["model_state_dict"], strict=True)
        source_checkpoint_sha256=file_sha256(args.init_checkpoint)
    model.vector_field.latent_scale = float(args.latent_scale)
    checkpoint_config=vars(args) | {
        "map_size": model.map_size if model.map_size is not None else 64,
        "map_height": model.map_height, "map_width": model.map_width,
        "architecture": model.architecture,
        "transformer_window_size": model.transformer_window_size,
        "trained_map_shapes": [list(shape) for shape in base.unique_map_shapes],
        "spatial_version": 1,
        "supervision_version": 2 if args.rasterization_mode == "segments" else 1,
        "label_horizon": int(labels["fine_routes"].shape[2]),
        "endpoint_protocol": "exact_hard_endpoints_v2",
        "coordinate_convention": "x_norm=2*x/(W-1)-1;y_norm=2*y/(H-1)-1",
        "distance_unit": "grid_pixels",
        "source_model_config": source_model_config,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        # Keep the task-level schema names alongside the existing names used
        # by older inspection scripts.
        "original_model_config": source_model_config,
        "source_checkpoint_hash": source_checkpoint_sha256,
        "horizon": model.horizon, "latent_dim": model.latent_dim,
        "hidden_dim": model.hidden_dim, "decode_beta": model.decode_beta,
        "decode_power": model.decode_power, "decode_threshold": model.decode_threshold,
        "distance_scale": model.distance_scale,
        "robot_radius_grid": model.robot_radius_grid,
        "clearance_temperature": model.clearance_temperature,
        "clearance_offset": model.clearance_offset,
        "decode_clearance_power": model.decode_clearance_power,
        "decode_continuity_beta": model.decode_continuity_beta,
        "accept_threshold": model.accept_threshold,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.save_dir); output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "split.npz", train_ids=train_ids, val_ids=val_ids,
        geometry_val_ids=geometry_val_ids,
    )
    split_audit = {"total_tasks": len(dataset), "unique_maps": len(np.unique(hashes)),
                   "train_tasks": len(train_ids), "val_tasks": len(val_ids),
                   "train_loader_tasks": int(len(train_loader_ids)),
                   "hard_replay_tasks": int(len(replay_ids)),
                   "hard_replay_factor": int(args.hard_replay_factor),
                   "map_overlap": int(len(set(hashes[train_ids]) & set(hashes[val_ids]))),
                   "sampling": "uniform", "min_teacher_clearance": args.min_teacher_clearance,
                   "eligible_tasks": int(eligible.sum()),
                   "geometry_val_tasks": int(len(geometry_val_ids)),
                   "geometry_val_seed": int(args.seed + 400_003)}
    split_audit["inherited_split_path"] = args.split_path
    split_audit["teacher_count_histogram"] = dict(zip(
        *[items.tolist() for items in np.unique(raw["valid"].sum(axis=1).astype(int), return_counts=True)]
    ))
    (output / "split_audit.json").write_text(json.dumps(split_audit, indent=2))
    best = float("inf"); best_geometry = -float("inf"); history = []
    train_generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
    latent_prototypes = None
    if args.fixed_latent_prototypes:
        prototype_generator = torch.Generator(device=device).manual_seed(args.seed + 500_003)
        latent_prototypes = torch.randn(
            labels["fine_routes"].shape[1], args.latent_dim,
            device=device, dtype=torch.float32, generator=prototype_generator,
        )
        latent_prototypes = latent_prototypes / latent_prototypes.std().clamp_min(1e-6)
    history_jsonl = output / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        print(json.dumps({
            "epoch": epoch, "stage": "train_start",
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
        }), flush=True)
        train = run_epoch(
            model, train_loader, labels, device, args, optimizer,
            generator=train_generator, latent_prototypes=latent_prototypes,
        )
        print(json.dumps({"epoch": epoch, "stage": "train_complete", "train_loss": train}), flush=True)
        val_generator = torch.Generator(device=device).manual_seed(args.seed + 300_003)
        print(json.dumps({"epoch": epoch, "stage": "validation_start"}), flush=True)
        with torch.no_grad():
            val = run_epoch(
                model, val_loader, labels, device, args,
                generator=val_generator, latent_prototypes=latent_prototypes,
            )
        print(json.dumps({"epoch": epoch, "stage": "validation_complete", "val_loss": val}), flush=True)
        if val < best:
            best = val
            torch.save({"epoch": epoch, "variant": model.variant,
                        "model_state_dict": model.state_dict(),
                        "config": checkpoint_config | {"split_audit": split_audit}, "loss": best}, output / "best_p37.pth")
        geometry_due = (
            epoch == 1 or epoch == args.epochs
            or epoch % args.geometry_val_every == 0
        )
        geometry = None
        if geometry_due:
            print(json.dumps({
                "epoch": epoch, "stage": "geometry_validation_start",
                "tasks": min(len(geometry_val_ids), args.geometry_val_max_tasks)
                if args.geometry_val_max_tasks > 0 else len(geometry_val_ids),
            }), flush=True)
            geometry = geometry_success(model, base, geometry_val_ids, device, args)
            print(json.dumps({
                "epoch": epoch, "stage": "geometry_validation_complete",
                "selected_path_success_rate": geometry["selected_path_success_rate"],
                "deployment_success_rate": geometry["deployment_success_rate"],
            }), flush=True)
        row = {
            "epoch": epoch, "train_loss": train, "val_loss": val,
            "geometry_val_ran": bool(geometry_due),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
        }
        if geometry is not None:
            row["geometry_val_C_N"] = geometry["oracle_best_of_k_success_rate"]
            row.update({f"geometry_val_{key}": value for key, value in geometry.items()})
        history.append(row)
        print(json.dumps(row), flush=True)
        with history_jsonl.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        if geometry is not None and geometry["deployment_success_rate"] > best_geometry:
            best_geometry = geometry["deployment_success_rate"]
            torch.save({"epoch": epoch, "variant": model.variant,
                        "model_state_dict": model.state_dict(),
                        "config": checkpoint_config | {"split_audit": split_audit},
                        "loss": val, "geometry_val": geometry,
                        "geometry_best_metric": "deployment_success_rate"},
                       output / "best_p37_geometry.pth")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"best_val_loss": best,
                      "best_geometry_val_deployment_success": best_geometry,
                      "checkpoint": str(output / "best_p37.pth"),
                      "geometry_checkpoint": str(output / "best_p37_geometry.pth"),
                      "split": split_audit}))


if __name__ == "__main__": main()
