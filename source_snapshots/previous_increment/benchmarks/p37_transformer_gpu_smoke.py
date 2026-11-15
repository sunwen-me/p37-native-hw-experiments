"""Run one real supervised P37 CFM optimizer step per requested CUDA shape.

This is a bounded architecture smoke test, not a training or quality benchmark.
It expects a CUDA-enabled PyTorch runtime and the repository's P37 training
source tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training/source/map_flow_matching_p1"
sys.path.insert(0, str(TRAINING / "scripts"))

from train_p37_ordered_field import sinkhorn_cfm_loss  # noqa: E402
from p37_model import OrderedDenseFieldFlow  # noqa: E402


def run_shape(height: int, width: int, args) -> dict:
    device = torch.device("cuda")
    model = OrderedDenseFieldFlow(
        map_size=64, horizon=args.horizon, hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim, architecture=args.architecture,
        transformer_window_size=args.window_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    maps = torch.zeros(1, 2, height, width, device=device)
    maps[:, 1] = 1.0
    starts = torch.tensor([[-0.8, -0.8]], device=device)
    goals = torch.tensor([[0.8, 0.8]], device=device)
    fraction = torch.linspace(0.0, 1.0, args.horizon, device=device)
    routes = starts[:, None, None, :] + fraction[None, None, :, None] * (
        goals - starts
    )[:, None, None, :]
    valid = torch.ones(1, 1, device=device)
    loss_args = SimpleNamespace(
        mask_thickness=1,
        field_positive_weight=8.0,
        progress_support_weight=4.0,
        progress_background_weight=0.1,
        endpoint_approach_weight=0.0,
        endpoint_bandwidth=0.12,
        endpoint_tail_weight=6.0,
        endpoint_tail_fraction=0.125,
        target_clearance_margin=0.0,
        target_clearance_temperature=0.25,
        target_clearance_floor=0.15,
        share_source_across_latents=False,
        subpixel_rasterization=False,
        sinkhorn_epsilon=0.05,
        sinkhorn_iterations=5,
        latent_diversity_weight=0.0,
        latent_diversity_margin=0.01,
    )
    steps_ms = []
    peak_allocated = peak_reserved = 0
    last_loss = None
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        loss = sinkhorn_cfm_loss(
            model, maps, starts, goals, routes, valid, loss_args,
            generator=torch.Generator(device=device).manual_seed(args.seed + step),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if step > 0:
            steps_ms.append(elapsed_ms)
        last_loss = float(loss.detach())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at {height}x{width}")
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"non-finite gradient at {height}x{width}")
        peak_allocated = max(
            peak_allocated, torch.cuda.max_memory_allocated(device),
        )
        peak_reserved = max(
            peak_reserved, torch.cuda.max_memory_reserved(device),
        )
    result = {
        "height": height,
        "width": width,
        "architecture": args.architecture,
        "hidden_dim": args.hidden_dim,
        "window_size": args.window_size,
        "warmup_steps": 1,
        "measured_steps": args.steps,
        "step_ms": steps_ms,
        "median_step_ms": sorted(steps_ms)[len(steps_ms) // 2],
        "peak_allocated_mib": peak_allocated / (1024 ** 2),
        "peak_reserved_mib": peak_reserved / (1024 ** 2),
        "loss": last_loss,
        "finite_loss_and_gradients": True,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    del optimizer, model, maps, starts, goals, routes, valid
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=("64x64", "512x768"))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--architecture", choices=("cnn", "transformer"), default="transformer",
    )
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8173)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    output = []
    for shape in args.sizes:
        height, width = map(int, shape.lower().split("x"))
        torch.cuda.reset_peak_memory_stats()
        output.append(run_shape(height, width, args))
    print(json.dumps({
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "device_memory_total_mib": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
        "results": output,
        "scope": "synthetic supervised CFM optimizer-step smoke; not a quality evaluation",
    }, indent=2))


if __name__ == "__main__":
    main()
