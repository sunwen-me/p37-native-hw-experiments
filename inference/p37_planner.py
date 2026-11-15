"""Pixel-coordinate deployment wrapper for the non-RL P37 Flow prior.

The prior's learned selector only sees its fields and distance channel. This
module performs deterministic occupancy preprocessing and coordinate scaling;
it does not certify collision safety.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from flow_prior import load_flow_prior


def _maps_as_list(occupancy):
    if isinstance(occupancy, np.ndarray) and occupancy.dtype != object:
        values = np.asarray(occupancy, dtype=np.float32)
        if values.ndim == 2:
            return [values]
        if values.ndim == 3:
            return [values[index] for index in range(len(values))]
        raise ValueError("occupancy must have shape [H,W] or [N,H,W]")
    values = [np.asarray(value, dtype=np.float32) for value in occupancy]
    if not values:
        raise ValueError("occupancy must contain at least one map")
    if any(value.ndim != 2 for value in values):
        raise ValueError("each occupancy map must be two-dimensional")
    return values


def _p37_map_input(occupancy: np.ndarray, distance_scale: float) -> np.ndarray:
    import cv2

    free = (occupancy < 0.5).astype(np.uint8)
    padded = np.pad(free, 1, mode="constant", constant_values=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    distance = np.clip(distance / float(distance_scale), 0.0, 1.0)
    return np.stack([occupancy, distance]).astype(np.float32)


class P37FlowPlanner:
    """Load a P37 checkpoint and plan in pixel coordinates on H/W grids.

    A single model call always contains one exact H/W because the P37 U-Net's
    GroupNorm statistics must not include padded map regions. A mixed-size
    input sequence is therefore split into exact-shape buckets and restored
    to its original row order.
    """

    def __init__(self, checkpoint, device="cpu"):
        self.model, self.checkpoint_state = load_flow_prior(checkpoint, device)
        self.model.eval().requires_grad_(False)
        self.device = torch.device(device)

    @torch.inference_mode()
    def plan(self, occupancy, starts_px, goals_px, *, candidates=4, seed=20260921,
             flow_steps=100, noise_scale=1.0, sde_noise=0.0):
        maps = _maps_as_list(occupancy)
        if not maps:
            raise ValueError("occupancy must contain at least one map")
        starts = np.asarray(starts_px, dtype=np.float32).reshape(-1, 2)
        goals = np.asarray(goals_px, dtype=np.float32).reshape(-1, 2)
        if len(starts) != len(maps) or len(goals) != len(maps):
            raise ValueError("one start and goal pair is required for each map")
        if candidates < 1 or flow_steps < 1 or noise_scale < 0 or sde_noise < 0:
            raise ValueError("invalid sampling parameters")
        if not np.isfinite(starts).all() or not np.isfinite(goals).all():
            raise ValueError("endpoint coordinates must be finite")

        buckets = defaultdict(list)
        for index, value in enumerate(maps):
            if min(value.shape) < 4:
                raise ValueError("map height and width must both be at least 4")
            if not np.isfinite(value).all() or not np.isin(value, [0.0, 1.0]).all():
                raise ValueError("occupancy maps must be finite binary grids")
            height, width = value.shape
            if (not 0 <= starts[index, 0] <= width - 1 or
                    not 0 <= starts[index, 1] <= height - 1 or
                    not 0 <= goals[index, 0] <= width - 1 or
                    not 0 <= goals[index, 1] <= height - 1):
                raise ValueError("endpoint outside its map")
            buckets[(height, width)].append(index)

        count = len(maps)
        horizon = self.model.horizon
        dtype=next(self.model.parameters()).dtype
        paths = torch.empty((count, candidates, horizon, 2), device=self.device, dtype=dtype)
        scores = torch.empty((count, candidates), device=self.device, dtype=dtype)
        decoder_valid = torch.empty((count, candidates), dtype=torch.bool, device=self.device)
        accepted = torch.empty_like(decoder_valid)
        selected = torch.empty(count, dtype=torch.long, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        for (height, width), indices in buckets.items():
            features = np.stack([_p37_map_input(maps[index], self.model.distance_scale)
                                 for index in indices])
            scale = np.asarray([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float32)
            starts_norm = torch.from_numpy(starts[indices] / scale - 1.0).to(self.device)
            goals_norm = torch.from_numpy(goals[indices] / scale - 1.0).to(self.device)
            map_tensor = torch.from_numpy(features).to(self.device)
            result = self.model.sample(
                map_tensor, starts_norm, goals_norm, candidates=candidates,
                num_steps=flow_steps, noise_scale=noise_scale,
                generator=generator, sde_noise=sde_noise,
            )
            rows = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            paths[rows] = (result["paths"] + 1.0) * torch.as_tensor(
                scale, device=self.device, dtype=result["paths"].dtype,
            )
            # The model hard-fixes normalized endpoints. Restore their exact
            # pixel inputs only when that contract holds; retain any actual
            # prediction error for downstream rejection.
            for position, normalized, original in (
                (0, starts_norm, starts[indices]), (-1, goals_norm, goals[indices]),
            ):
                fixed = (result["paths"][:, :, position] == normalized[:, None]).all(-1)
                paths[rows, :, position] = torch.where(
                    fixed[..., None],
                    torch.as_tensor(original, device=self.device, dtype=dtype)[:, None],
                    paths[rows, :, position],
                )
            scores[rows] = result["selection_scores"]
            decoder_valid[rows] = result["decoder_support_valid"]
            accepted[rows] = result["accepted"]
            selected[rows] = result["selected_index"]

        row = torch.arange(count, device=self.device)
        return {
            "paths": paths,
            "selected_index": selected,
            "selected_path": paths[row, selected],
            "selection_scores": scores,
            "decoder_support_valid": decoder_valid,
            "accepted": accepted,
            "abstain": ~accepted.any(dim=1),
            "map_shapes": tuple(tuple(map(int, value.shape)) for value in maps),
            "decoder_config": {
                "mode": self.model.decoder_mode,
                "horizon": self.model.horizon,
                "progress_smoothing": self.model.decode_progress_smoothing,
            },
            "safety_score_is_not_certified": True,
        }
