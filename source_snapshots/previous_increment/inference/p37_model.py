"""P37 route-free Flow with occupancy and ordered progress fields."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from p10_model import _point_masks
from adaptive_field_decoder import adaptive_ordered_field_to_path
from safe_field_decoder import (
    distance_field_to_soft_clearance,
    safe_ordered_field_to_path,
    support_weighted_progress_smoothing,
)


class OrderedDenseFieldVectorField(nn.Module):
    """Spatial U-Net whose Flow state is ``[occupancy, progress]``."""

    def __init__(self, map_channels=2, hidden_dim=64, latent_dim=8):
        super().__init__()
        hidden_dim = int(hidden_dim)
        latent_dim = int(latent_dim)
        self.latent_dim = latent_dim
        # Diagnostic/control knob for global latent conditioning.  The
        # checkpoint-compatible default preserves the original P37 behavior.
        self.latent_scale = 1.0
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
        )
        self.latent_embedding = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
        )
        # Two state channels + map channels + start/goal masks.
        self.stem = nn.Sequential(
            nn.Conv2d(map_channels + 4, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim), nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2), nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2), nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 4), nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4), nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4), nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 4), nn.SiLU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 6, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2), nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim), nn.SiLU(),
        )
        self.out = nn.Conv2d(hidden_dim, 2, 1)

    def forward(self, x, map_input, start, goal, t, latent):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Flow state must have shape [B,2,H,W]")
        if map_input.ndim != 4 or map_input.shape[0] != len(x):
            raise ValueError("map_input must have shape [B,C,H,W]")
        if map_input.shape[-2:] != x.shape[-2:]:
            raise ValueError("Flow state and map spatial shapes must match")
        height, width = x.shape[-2:]
        start_mask = _point_masks(start, height, width)
        goal_mask = _point_masks(goal, height, width)
        h0 = self.stem(torch.cat([x, map_input, start_mask, goal_mask], dim=1))
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        cond = (
            self.time_embedding(t[:, None])
            + float(self.latent_scale) * self.latent_embedding(latent)
        )
        h2 = h2 + cond[:, :, None, None]
        h2 = self.mid(h2)
        u1 = F.interpolate(h2, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, h1], dim=1))
        u2 = F.interpolate(u1, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, h0], dim=1))
        return self.out(u2)


def _attention_head_count(channels: int) -> int:
    """Choose a small valid head count for the current feature width."""
    for heads in range(min(8, int(channels)), 0, -1):
        if int(channels) % heads == 0:
            return heads
    return 1


class WindowTransformerBlock(nn.Module):
    """Shiftable local attention over a 2D feature map.

    Padding is explicit instead of using cyclic rolls, so tokens at opposite
    image borders never attend to one another. Padded keys are masked.
    """

    def __init__(self, channels: int, window_size: int, shift_size: int = 0):
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if self.window_size < 2 or not 0 <= self.shift_size < self.window_size:
            raise ValueError("invalid transformer window or shift size")
        heads = _attention_head_count(self.channels)
        self.heads = heads
        self.head_dim = self.channels // heads
        self.norm1 = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, self.channels * 3)
        self.proj = nn.Linear(self.channels, self.channels)
        relative_span = 2 * self.window_size - 1
        self.relative_position_bias = nn.Parameter(
            torch.zeros(relative_span * relative_span, heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.window_size), torch.arange(self.window_size),
            indexing="ij",
        )).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative + self.window_size - 1
        relative_index = relative[0] * relative_span + relative[1]
        self.register_buffer("relative_position_index", relative_index, persistent=False)
        self.norm2 = nn.LayerNorm(self.channels)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.channels * 4), nn.GELU(),
            nn.Linear(self.channels * 4, self.channels),
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def _partition(self, value: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = value.shape
        window = self.window_size
        return value.reshape(
            batch, height // window, window, width // window, window, channels,
        ).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)

    def _unpartition(
        self, windows: torch.Tensor, batch: int, height: int, width: int,
    ) -> torch.Tensor:
        window = self.window_size
        channels = self.channels
        return windows.reshape(
            batch, height // window, width // window, window, window, channels,
        ).permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        if channels != self.channels:
            raise ValueError("transformer feature width does not match block")
        window, shift = self.window_size, self.shift_size
        normalized = self.norm1(value.permute(0, 2, 3, 1))
        pad_left = pad_top = shift
        pad_right = (window - (width + pad_left) % window) % window
        pad_bottom = (window - (height + pad_top) % window) % window
        padded = F.pad(
            normalized.permute(0, 3, 1, 2),
            (pad_left, pad_right, pad_top, pad_bottom),
        ).permute(0, 2, 3, 1)
        valid = torch.ones(
            (batch, 1, height, width), dtype=torch.bool, device=value.device,
        )
        valid = F.pad(
            valid, (pad_left, pad_right, pad_top, pad_bottom), value=False,
        ).permute(0, 2, 3, 1)
        padded_height, padded_width = padded.shape[1:3]
        tokens = self._partition(padded)
        key_valid = self._partition(valid).squeeze(-1)
        window_count, token_count, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(
            window_count, token_count, 3, self.heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, content = qkv.unbind(0)
        attention = (query * (self.head_dim ** -0.5)) @ key.transpose(-2, -1)
        bias = self.relative_position_bias[self.relative_position_index]
        bias = bias.permute(2, 0, 1).to(dtype=attention.dtype)
        attention = attention + bias.unsqueeze(0)
        attention = attention.masked_fill(
            ~key_valid[:, None, None, :], torch.finfo(attention.dtype).min,
        ).softmax(dim=-1)
        attended = (attention @ content).transpose(1, 2).reshape(
            window_count, token_count, channels,
        )
        attended = self._unpartition(
            self.proj(attended), batch, padded_height, padded_width,
        )
        attended = attended[:, pad_top:pad_top + height, pad_left:pad_left + width]
        value = value + attended.permute(0, 3, 1, 2)
        tokens = value.permute(0, 2, 3, 1)
        return (tokens + self.mlp(self.norm2(tokens))).permute(0, 3, 1, 2)


class GlobalTransformerBlock(nn.Module):
    """Global self-attention at the /32 bottleneck, where tokens stay few."""

    def __init__(self, channels: int):
        super().__init__()
        heads = _attention_head_count(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        tokens = value.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(math.gcd(8, int(out_channels)), out_channels),
            nn.GELU(),
        )


class _Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(math.gcd(8, int(out_channels)), out_channels),
            nn.GELU(),
            _ConvNormAct(out_channels, out_channels),
        )


class OrderedDenseFieldTransformerVectorField(nn.Module):
    """Hierarchical window Transformer vector field for P37 Flow Matching.

    The encoder uses shifted-window attention at 1/4, 1/8 and 1/16 scale,
    followed by global attention at 1/32 scale. A convolutional skip decoder
    returns the same two-channel dense velocity field as the CNN backbone.
    """

    def __init__(
        self, map_channels=2, hidden_dim=64, latent_dim=8,
        window_size=8,
    ):
        super().__init__()
        hidden_dim, latent_dim = int(hidden_dim), int(latent_dim)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.window_size = int(window_size)
        if hidden_dim < 1 or latent_dim < 1:
            raise ValueError("hidden_dim and latent_dim must be positive")
        if self.window_size < 2:
            raise ValueError("transformer window_size must be at least 2")
        channels = hidden_dim * 4
        self.latent_scale = 1.0
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, channels),
        )
        self.latent_embedding = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, channels),
        )
        self.stem = nn.Sequential(
            _ConvNormAct(map_channels + 4, hidden_dim),
            _ConvNormAct(hidden_dim, hidden_dim),
        )
        self.down1 = _Downsample(hidden_dim, hidden_dim * 2)
        self.down2 = _Downsample(hidden_dim * 2, channels)
        self.down3 = _Downsample(channels, channels)
        self.down4 = _Downsample(channels, channels)
        self.down5 = _Downsample(channels, channels)
        self.stage4 = nn.ModuleList([
            WindowTransformerBlock(channels, self.window_size, shift)
            for shift in (0, self.window_size // 2)
        ])
        self.stage8 = nn.ModuleList([
            WindowTransformerBlock(channels, self.window_size, shift)
            for shift in (0, self.window_size // 2)
        ])
        self.stage16 = nn.ModuleList([
            WindowTransformerBlock(channels, self.window_size, shift)
            for shift in (0, self.window_size // 2)
        ])
        self.global_stage = nn.ModuleList([
            GlobalTransformerBlock(channels), GlobalTransformerBlock(channels),
        ])
        self.up16 = _ConvNormAct(channels * 2, channels)
        self.up8 = _ConvNormAct(channels * 2, channels)
        self.up4 = _ConvNormAct(channels * 2, hidden_dim * 2)
        self.up2 = _ConvNormAct(hidden_dim * 4, hidden_dim)
        self.up1 = _ConvNormAct(hidden_dim * 2, hidden_dim)
        self.out = nn.Conv2d(hidden_dim, 2, 1)

    @staticmethod
    def _run_stage(value: torch.Tensor, stage: nn.ModuleList) -> torch.Tensor:
        for block in stage:
            value = block(value)
        return value

    @staticmethod
    def _merge(value: torch.Tensor, skip: torch.Tensor, block: nn.Module) -> torch.Tensor:
        value = F.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False,
        )
        return block(torch.cat([value, skip], dim=1))

    def forward(self, x, map_input, start, goal, t, latent):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Flow state must have shape [B,2,H,W]")
        if map_input.ndim != 4 or map_input.shape[0] != len(x):
            raise ValueError("map_input must have shape [B,C,H,W]")
        if map_input.shape[-2:] != x.shape[-2:]:
            raise ValueError("Flow state and map spatial shapes must match")
        if map_input.shape[1] != 2:
            raise ValueError("map_input must have two channels")
        height, width = x.shape[-2:]
        start_mask = _point_masks(start, height, width)
        goal_mask = _point_masks(goal, height, width)
        value = torch.cat([x, map_input, start_mask, goal_mask], dim=1)
        pad_height, pad_width = (-height) % 32, (-width) % 32
        value = F.pad(value, (0, pad_width, 0, pad_height))
        x0 = self.stem(value)
        x1 = self.down1(x0)
        x2 = self._run_stage(self.down2(x1), self.stage4)
        x3 = self._run_stage(self.down3(x2), self.stage8)
        x4 = self._run_stage(self.down4(x3), self.stage16)
        x5 = self.down5(x4)
        condition = (
            self.time_embedding(t[:, None])
            + float(self.latent_scale) * self.latent_embedding(latent)
        )
        x5 = x5 + condition[:, :, None, None]
        x5 = self._run_stage(x5, self.global_stage)
        value = self._merge(x5, x4, self.up16)
        value = self._merge(value, x3, self.up8)
        value = self._merge(value, x2, self.up4)
        value = self._merge(value, x1, self.up2)
        value = self._merge(value, x0, self.up1)
        return self.out(value)[..., :height, :width]


def endpoint_fields(
    start: torch.Tensor, goal: torch.Tensor, height: int | None = None,
    width: int | None = None, *, size: int | None = None,
) -> torch.Tensor:
    """Return fixed endpoint values, preserving ``size=`` square calls.

    Normalized endpoints use x/width and y/height independently.
    """
    if size is not None:
        if height is not None and int(height) != int(size):
            raise ValueError("size and height disagree")
        height = int(size)
    if height is None:
        raise ValueError("height is required")
    height = int(height)
    width = height if width is None else int(width)
    if height < 2 or width < 2:
        raise ValueError("endpoint map dimensions must be at least 2")
    if start.ndim != 2 or goal.shape != start.shape or start.shape[-1] != 2:
        raise ValueError("start and goal must have shape [B,2]")
    if not torch.isfinite(start).all() or not torch.isfinite(goal).all():
        raise ValueError("endpoint coordinates must be finite")
    tolerance = 1e-6
    if ((start < -1.0 - tolerance) | (start > 1.0 + tolerance)).any() or (
        (goal < -1.0 - tolerance) | (goal > 1.0 + tolerance)
    ).any():
        raise ValueError("normalized endpoints must lie inside the map")
    start_mask = _point_masks(start, height, width)
    goal_mask = _point_masks(goal, height, width)
    occupied = (start_mask + goal_mask).clamp(0.0, 1.0)
    progress = goal_mask
    return torch.cat([occupied, progress], dim=1)


def flow_sde_transition(
    x: torch.Tensor,
    velocity_forward: torch.Tensor,
    step: int,
    num_steps: int,
    noise_level: float,
    endpoint: torch.Tensor,
    fixed_state: torch.Tensor,
    generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one forward-time Flow-SDE transition.

    This is shared by PPO and deployment sampling so that the policy is
    trained under the same transition kernel that is used at evaluation time.
    Endpoint values remain deterministic and are excluded from the stochastic
    transition by masking both the mean and standard deviation.
    """
    if num_steps < 1 or not 0 <= step < num_steps:
        raise ValueError("invalid Flow-SDE step")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    dt = 1.0 / float(num_steps)
    t = float(step) * dt
    tau = 1.0 - t
    tau_next = max(tau - dt, 0.0)

    x0_pred = x + tau * velocity_forward
    x1_pred = x - t * velocity_forward
    denominator = tau_next if abs(tau - 1.0) < 1e-8 else tau
    sigma_ratio = tau / max(1.0 - denominator, 1e-8)
    sigma = float(noise_level) * math.sqrt(max(sigma_ratio, 0.0))
    mean = x0_pred * (1.0 - tau_next)
    mean = mean + x1_pred * (
        tau_next - (sigma * sigma) * dt / max(2.0 * tau, 1e-8)
    )
    std_value = math.sqrt(dt) * sigma
    std = torch.full_like(x, std_value)

    mean = mean * (1.0 - endpoint) + fixed_state
    std = std * (1.0 - endpoint)
    if std_value == 0.0:
        next_x = mean
    else:
        if noise is None:
            noise = torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
        if tuple(noise.shape) != tuple(x.shape):
            raise ValueError("noise must match the Flow state shape")
        next_x = mean + noise * std
        next_x = next_x * (1.0 - endpoint) + fixed_state
    return next_x, mean, std


def candidate_selection_scores(
    fields: torch.Tensor,
    paths: torch.Tensor,
    map_input: torch.Tensor,
    decoder_valid: torch.Tensor,
    segment_samples: int = 16,
) -> torch.Tensor:
    """Score decoded candidates with model-visible field evidence.

    This is deliberately not an exact collision checker.  It uses only the
    predicted ordered field and the distance-field channel already supplied to
    the model, so deployment remains search-free and geometry-checker-free.
    Distance evidence is sampled densely along every segment; checking only
    the segment midpoint can miss a collision on a long decoded segment.
    Invalid decoder outputs are never selected when a valid candidate exists.
    """
    if fields.ndim != 5 or paths.ndim != 4 or decoder_valid.ndim != 2:
        raise ValueError("invalid candidate tensor ranks")
    if int(segment_samples) < 2:
        raise ValueError("segment_samples must be at least 2")
    batch, candidates = fields.shape[:2]
    if paths.shape[:2] != (batch, candidates) or decoder_valid.shape != (batch, candidates):
        raise ValueError("candidate dimensions do not match")
    if map_input.ndim != 4 or len(map_input) != batch or map_input.shape[1] < 2:
        raise ValueError("map_input must contain occupancy and distance channels")

    flat_fields = fields.reshape(-1, *fields.shape[2:]).clamp(0.0, 1.0)
    height, width = fields.shape[-2:]
    if map_input.shape[-2:] != (height, width):
        raise ValueError("fields and map_input spatial shapes must match")
    flat_paths = paths.reshape(-1, paths.shape[2], 2).clamp(-1.0, 1.0)
    flat_distance = map_input[:, 1:2].unsqueeze(1).expand(
        -1, candidates, -1, -1, -1,
    ).reshape(-1, 1, map_input.shape[-2], map_input.shape[-1])
    grid = flat_paths.unsqueeze(2)
    segment_start = flat_paths[:, :-1, None, :]
    segment_delta = flat_paths[:, 1:, None, :] - segment_start
    fractions = torch.linspace(
        0.0, 1.0, int(segment_samples), device=paths.device, dtype=paths.dtype,
    ).view(1, 1, int(segment_samples), 1)
    clearance_points = segment_start + fractions * segment_delta
    clearance_grid = clearance_points.reshape(
        flat_paths.shape[0], -1, 1, 2,
    )
    occupancy = F.grid_sample(
        flat_fields[:, :1], grid, mode="bilinear", padding_mode="border",
        align_corners=True,
    ).squeeze(1).squeeze(-1)
    progress = F.grid_sample(
        flat_fields[:, 1:2], grid, mode="bilinear", padding_mode="border",
        align_corners=True,
    ).squeeze(1).squeeze(-1)
    distance = F.grid_sample(
        flat_distance, clearance_grid, mode="bilinear", padding_mode="border",
        align_corners=True,
    ).squeeze(1).squeeze(-1)
    target_progress = torch.linspace(
        0.0, 1.0, paths.shape[2], device=paths.device, dtype=paths.dtype,
    )[None]
    progress_error = (progress - target_progress).abs().mean(dim=1)
    occupancy_support = occupancy.mean(dim=1)
    clearance_mean = distance.mean(dim=1)
    clearance_min = distance.min(dim=1).values
    # Express normalized deltas in legacy 64-grid units. This is identical
    # for 64x64 and measures the correct anisotropic pixel distance otherwise.
    metric_scale = flat_paths.new_tensor([(width - 1) / 63.0, (height - 1) / 63.0])
    path_length = torch.linalg.vector_norm(
        (flat_paths[:, 1:] - flat_paths[:, :-1]) * metric_scale, dim=-1,
    ).sum(dim=1)
    # A single low-clearance bottleneck can invalidate the entire path, so
    # minimum clearance dominates the mean.  Both are model-visible distance
    # field evidence; no exact geometry checker is used online.
    scores = (
        3.0 * clearance_min
        + 0.5 * clearance_mean
        + 0.5 * occupancy_support
        - 0.5 * progress_error
        - 0.02 * path_length
    ).reshape(batch, candidates)
    scores = scores.masked_fill(~decoder_valid, -torch.inf)
    return scores


class OrderedDenseFieldFlow(nn.Module):
    """Route-free dense Flow with a global latent and ordered field decoder."""

    variant = "p37_ordered_dense_field_flow"

    def __init__(self, map_size=64, horizon=128, hidden_dim=64, latent_dim=8,
                 map_height=None, map_width=None,
                 decode_beta=1280.0, decode_power=4.0, accept_threshold=0.5,
                 decode_threshold=0.1, distance_scale=8.0,
                 robot_radius_grid=1.5, clearance_temperature=0.5,
                 clearance_offset=2 ** -0.5, decode_clearance_power=2.0,
                 decode_continuity_beta=60.0,
                 decode_terminal_support_window=8,
                 decode_terminal_support_fraction=0.75,
                 decode_max_terminal_jump=0.5,
                 decode_max_step_grid=None,
                 decoder_mode="legacy",
                 decode_progress_smoothing=0,
                 architecture="cnn", transformer_window_size=8,
                 **kwargs):
        super().__init__()
        legacy_size = int(map_size)
        self.map_height = legacy_size if map_height is None else int(map_height)
        self.map_width = legacy_size if map_width is None else int(map_width)
        if self.map_height < 4 or self.map_width < 4:
            raise ValueError("map height and width must both be at least 4")
        if (map_height is None) != (map_width is None):
            raise ValueError("map_height and map_width must be supplied together")
        if map_height is not None and legacy_size != 64 and (
            legacy_size != self.map_height or legacy_size != self.map_width
        ):
            raise ValueError("map_size cannot be combined with different explicit H/W")
        # Legacy consumers expect map_size on square models. New code uses the
        # runtime map tensor shape, so rectangular models expose both axes.
        self.map_size = self.map_height if self.map_height == self.map_width else None
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.decode_beta = float(decode_beta)
        self.decode_power = float(decode_power)
        self.decode_threshold = float(decode_threshold)
        self.accept_threshold = float(accept_threshold)
        self.distance_scale = float(distance_scale)
        self.robot_radius_grid = float(robot_radius_grid)
        self.clearance_temperature = float(clearance_temperature)
        self.clearance_offset = float(clearance_offset)
        self.decode_clearance_power = float(decode_clearance_power)
        self.decode_continuity_beta = float(decode_continuity_beta)
        self.decode_terminal_support_window = int(decode_terminal_support_window)
        self.decode_terminal_support_fraction = float(decode_terminal_support_fraction)
        self.decode_max_terminal_jump = float(decode_max_terminal_jump)
        self.decode_max_step_grid = (
            None if decode_max_step_grid is None else float(decode_max_step_grid)
        )
        if decoder_mode not in {"legacy", "scale_aware", "adaptive_fallback"}:
            raise ValueError("decoder_mode must be 'legacy', 'scale_aware' or 'adaptive_fallback'")
        self.decoder_mode = decoder_mode
        if (isinstance(decode_progress_smoothing, bool)
                or not isinstance(decode_progress_smoothing, int)
                or decode_progress_smoothing < 0
                or (decode_progress_smoothing > 0 and decode_progress_smoothing % 2 == 0)):
            raise ValueError("decode_progress_smoothing must be zero or positive odd")
        self.decode_progress_smoothing = decode_progress_smoothing
        architecture = str(architecture).strip().lower()
        if architecture == "transforms":
            architecture = "transformer"
        if architecture not in {"cnn", "transformer"}:
            raise ValueError("architecture must be 'cnn' or 'transformer'")
        self.architecture = architecture
        self.transformer_window_size = int(transformer_window_size)
        if architecture == "cnn":
            # Keep the legacy module tree and parameter names unchanged so old
            # checkpoints continue to strict-load without conversion.
            self.vector_field = OrderedDenseFieldVectorField(
                map_channels=2, hidden_dim=hidden_dim, latent_dim=latent_dim,
            )
        else:
            self.vector_field = OrderedDenseFieldTransformerVectorField(
                map_channels=2, hidden_dim=hidden_dim, latent_dim=latent_dim,
                window_size=self.transformer_window_size,
            )

    def _integrate(self, map_input, start, goal, latent, num_steps, noise_scale,
                   generator=None, source=None, sde_noise=0.0,
                   sde_noise_schedule=None):
        batch = len(start)
        if sde_noise < 0:
            raise ValueError("sde_noise must be non-negative")
        if map_input.ndim != 4 or map_input.shape[0] != batch or map_input.shape[1] != 2:
            raise ValueError("map_input must have shape [B,2,H,W]")
        height, width = map_input.shape[-2:]
        if height < 4 or width < 4:
            raise ValueError("map height and width must both be at least 4")
        if sde_noise_schedule is not None:
            expected_schedule = (int(num_steps), batch, 2, height, width)
            if tuple(sde_noise_schedule.shape) != expected_schedule:
                raise ValueError("sde_noise_schedule must have shape [steps,batch,2,H,W]")
        if source is None:
            x = noise_scale * torch.randn(
                batch, 2, height, width,
                device=map_input.device, dtype=map_input.dtype, generator=generator,
            )
        else:
            expected = (batch, 2, height, width)
            if tuple(source.shape) != expected:
                raise ValueError("source must have shape [batch,2,H,W]")
            x = source.to(device=map_input.device, dtype=map_input.dtype)
        fixed = endpoint_fields(start, goal, height, width)
        endpoint = fixed[:, :1]
        # The CFM source fixes endpoint pixels before the first vector-field
        # query.  Keep inference on the same path as training.
        x = x * (1.0 - endpoint) + fixed
        dt = 1.0 / max(int(num_steps), 1)
        for step in range(int(num_steps)):
            t = torch.full((batch,), step * dt, device=x.device, dtype=x.dtype)
            velocity = self.vector_field(x, map_input, start, goal, t, latent)
            if sde_noise > 0.0:
                x, _, _ = flow_sde_transition(
                    x, velocity, step, int(num_steps), float(sde_noise),
                    endpoint, fixed, generator,
                    None if sde_noise_schedule is None else sde_noise_schedule[step],
                )
            else:
                x = x + dt * velocity
                x = x * (1.0 - endpoint) + fixed
        return x

    @torch.no_grad()
    def sample(self, map_input, start, goal, candidates=4, num_steps=100,
               noise_scale=1.0, latent=None, generator=None,
               shared_source=False, sde_noise=0.0, source=None,
               sde_noise_schedule=None):
        if candidates < 1 or num_steps < 1:
            raise ValueError("candidates and num_steps must be positive")
        batch = len(start)
        if map_input.ndim != 4 or map_input.shape[0] != batch or map_input.shape[1] != 2:
            raise ValueError("map_input must have shape [B,2,H,W]")
        height, width = map_input.shape[-2:]
        if tuple(start.shape) != (batch, 2) or tuple(goal.shape) != (batch, 2):
            raise ValueError("start and goal must have shape [B,2]")
        if latent is None:
            latent = torch.randn(
                batch, candidates, self.latent_dim, device=start.device,
                dtype=start.dtype, generator=generator,
            )
        expected = (batch, candidates, self.latent_dim)
        if tuple(latent.shape) != expected or latent.device != start.device or latent.dtype != start.dtype:
            raise ValueError("latent shape, device and dtype must match candidates")
        flat_maps = map_input[:, None].expand(-1, candidates, -1, -1, -1).reshape(
            batch * candidates, *map_input.shape[1:]
        )
        flat_start = start[:, None].expand(-1, candidates, -1).reshape(-1, 2)
        flat_goal = goal[:, None].expand(-1, candidates, -1).reshape(-1, 2)
        if source is not None:
            expected_source = (batch, candidates, 2, height, width)
            if tuple(source.shape) != expected_source:
                raise ValueError("source must have shape [B,K,2,H,W]")
            if source.device != map_input.device or source.dtype != map_input.dtype:
                raise ValueError("source device and dtype must match map_input")
            source = source.reshape(batch * candidates, 2, height, width)
        elif shared_source:
            shared = noise_scale * torch.randn(
                batch, 2, height, width,
                device=map_input.device, dtype=map_input.dtype, generator=generator,
            )
            source = shared[:, None].expand(
                -1, candidates, -1, -1, -1,
            ).reshape(batch * candidates, 2, height, width)
        flat_schedule = None
        if sde_noise_schedule is not None:
            expected_schedule = (num_steps, batch, candidates, 2, height, width)
            if tuple(sde_noise_schedule.shape) != expected_schedule:
                raise ValueError("sde_noise_schedule must have shape [steps,B,K,2,H,W]")
            flat_schedule = sde_noise_schedule.reshape(
                num_steps, batch * candidates, 2, height, width,
            )
        fields = self._integrate(
            flat_maps, flat_start, flat_goal, latent.reshape(-1, self.latent_dim),
            num_steps, noise_scale, generator=generator, source=source,
            sde_noise=sde_noise, sde_noise_schedule=flat_schedule,
        ).reshape(batch, candidates, 2, height, width)
        result = self.decode_fields(map_input, start, goal, fields)
        result["latent"] = latent
        return result

    @torch.no_grad()
    def decode_fields(self, map_input, start, goal, fields, *, mode=None,
                      horizon=None, progress_smoothing=None):
        """Decode and score cached or newly sampled fields through one path.

        ``fields`` remains the raw Flow output. ``decoder_fields`` records the
        optional support-weighted progress correction used by both decoding
        and selection. Neither input fields nor checkpoint weights are changed.
        """
        if fields.ndim != 5 or fields.shape[2] != 2:
            raise ValueError("fields must have shape [B,K,2,H,W]")
        batch, candidates, _, height, width = fields.shape
        if candidates < 1 or tuple(map_input.shape) != (batch, 2, height, width):
            raise ValueError("map_input must match fields with shape [B,2,H,W]")
        if tuple(start.shape) != (batch, 2) or tuple(goal.shape) != (batch, 2):
            raise ValueError("start and goal must have shape [B,2]")
        mode = self.decoder_mode if mode is None else mode
        if mode not in {"legacy", "scale_aware", "adaptive_fallback"}:
            raise ValueError("unknown decoder mode")
        horizon = self.horizon if horizon is None else horizon
        kernel = self.decode_progress_smoothing if progress_smoothing is None else progress_smoothing
        flat_fields = fields.reshape(batch * candidates, 2, height, width)
        smoothed = support_weighted_progress_smoothing(
            flat_fields[:, :1], flat_fields[:, 1:2], kernel,
        )
        decoder_fields = fields
        if kernel:
            decoder_fields = fields.clone()
            decoder_fields[:, :, 1:2] = smoothed.reshape(batch, candidates, 1, height, width)
            fixed = endpoint_fields(start, goal, height, width)[:, None]
            decoder_fields = decoder_fields * (1.0 - fixed[:, :, :1]) + fixed
            flat_fields = decoder_fields.reshape(batch * candidates, 2, height, width)
        flat_maps = map_input[:, None].expand(-1, candidates, -1, -1, -1).reshape(
            batch * candidates, 2, height, width,
        )
        flat_start = start[:, None].expand(-1, candidates, -1).reshape(-1, 2)
        flat_goal = goal[:, None].expand(-1, candidates, -1).reshape(-1, 2)
        distance = distance_field_to_soft_clearance(
            flat_maps[:, 1:2], self.distance_scale, self.robot_radius_grid,
            self.clearance_temperature, self.clearance_offset,
        )
        paths, decoder_support_valid, diagnostics = safe_ordered_field_to_path(
            flat_fields[:, :1].clamp(0.0, 1.0),
            flat_fields[:, 1:2].clamp(0.0, 1.0),
            distance,
            flat_start, flat_goal, horizon,
            beta=self.decode_beta, power=self.decode_power,
            clearance_power=self.decode_clearance_power,
            continuity_beta=self.decode_continuity_beta,
            occupancy_threshold=self.decode_threshold,
            terminal_support_window=self.decode_terminal_support_window,
            terminal_support_fraction=self.decode_terminal_support_fraction,
            max_terminal_jump=self.decode_max_terminal_jump,
            max_path_step=(
                None if self.decode_max_step_grid is None else
                2.0 * self.decode_max_step_grid / 63.0
            ),
            return_diagnostics=True,
            mode="scale_aware" if mode == "adaptive_fallback" else mode,
        )
        paths = paths.reshape(batch, candidates, horizon, 2)
        decoder_support_valid = decoder_support_valid.reshape(batch, candidates)
        selection_scores = candidate_selection_scores(
            decoder_fields, paths, map_input, decoder_support_valid,
        )
        accepted = decoder_support_valid & torch.isfinite(selection_scores)
        if mode == "adaptive_fallback":
            # Preserve every task with an existing accepted candidate, including
            # its selector. Continuation is only attempted after full abstention.
            retry_rows = ~accepted.any(dim=1)
            retry = retry_rows[:, None].expand(-1, candidates).reshape(-1)
            diagnostics["adaptive_attempted"] = retry.clone()
            diagnostics["baseline_decoder_support_valid"] = decoder_support_valid.reshape(-1).clone()
            for key, default in (("adaptive_internal_steps", 0), ("adaptive_backoffs", 0),
                                 ("adaptive_failure_code", -1)):
                diagnostics[key] = torch.full_like(retry, default, dtype=torch.long)
            for key in ("adaptive_final_progress", "adaptive_remaining_distance_px"):
                diagnostics[key] = flat_start.new_zeros(batch * candidates)
            for key in ("adaptive_complete", "adaptive_output_supported"):
                diagnostics[key] = torch.zeros_like(retry)
            if retry.any():
                seed_paths = paths.reshape(-1, horizon, 2)[retry]
                seed_lengths = diagnostics["first_missing_support_step"][retry].clamp(1, horizon - 1)
                retry_paths, retry_valid, retry_diagnostics = adaptive_ordered_field_to_path(
                    flat_fields[retry, :1].clamp(0, 1), flat_fields[retry, 1:2].clamp(0, 1),
                    distance[retry], flat_start[retry], flat_goal[retry], horizon,
                    beta=self.decode_beta, power=self.decode_power,
                    clearance_power=self.decode_clearance_power,
                    continuity_beta=self.decode_continuity_beta,
                    occupancy_threshold=self.decode_threshold,
                    terminal_support_window=self.decode_terminal_support_window,
                    terminal_support_fraction=self.decode_terminal_support_fraction,
                    max_terminal_jump=self.decode_max_terminal_jump,
                    max_path_step=(None if self.decode_max_step_grid is None else 2 * self.decode_max_step_grid / 63),
                    seed_paths=seed_paths, seed_lengths=seed_lengths,
                )
                paths = paths.clone()
                paths.reshape(-1, horizon, 2)[retry] = retry_paths
                decoder_support_valid = decoder_support_valid.clone()
                decoder_support_valid.reshape(-1)[retry] = retry_valid
                for key, value in retry_diagnostics.items():
                    diagnostics[key] = diagnostics[key].clone()
                    diagnostics[key][retry] = value
                selection_scores = candidate_selection_scores(
                    decoder_fields, paths, map_input, decoder_support_valid)
                accepted = decoder_support_valid & torch.isfinite(selection_scores)
        selected_index = selection_scores.argmax(dim=1)
        selected_index = torch.where(accepted.any(dim=1), selected_index,
                                    torch.zeros_like(selected_index))
        return {
            "fields": fields, "decoder_fields": decoder_fields, "paths": paths,
            "validity_logits": selection_scores,
            "selection_scores": selection_scores,
            # Keep decoder_valid for checkpoint/evaluator compatibility.  It
            # means field support validity, not collision safety.
            "decoder_valid": decoder_support_valid,
            "decoder_support_valid": decoder_support_valid,
            "accepted": accepted,
            "abstain": ~accepted.any(dim=1),
            "selected_index": selected_index,
            "selected_path": paths[torch.arange(batch, device=start.device), selected_index],
            "diagnostics": {key: value.reshape(batch, candidates) for key, value in diagnostics.items()},
        }
