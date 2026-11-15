"""Real training-batch gradient/memory checks for the one-objective A/B run."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "training/source/map_flow_matching_p1/scripts"),
               str(ROOT / "training/source/map_flow_matching")]
from posttrain_p37_native_hw import make_model, sha
import train_p37_ordered_field as trainer
from utils.dataset import PathPlanningDataset
from route_progress_objective import arc_length_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    source_hash = sha(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(state["config"]) | {"route_increment_weight": .1, "route_increment_samples": 128,
        "route_increment_scale": 32., "decoder_mode": "adaptive_fallback", "decode_progress_smoothing": 7,
        "rasterization_mode": "segments", "field_loss_reduction": "support_balanced"}
    model = make_model(config).cuda().train()
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.vector_field.latent_scale = float(config.get("latent_scale", 1.))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    base = PathPlanningDataset(str(args.data_dir / "train_data.npz"), normalize=True, augment=False,
        validate_geometry=False, cache_distance_fields=True, distance_scale=config["distance_scale"])
    train_ids, val_ids, _ = trainer.load_map_disjoint_split(base.maps, args.split_path)
    with np.load(args.data_dir / "p37_routes.npz") as archive:
        routes, valid = archive["fine_routes"], archive["valid"]
    rows = []
    for height, width in ((64, 64), (127, 193), (512, 768), (768, 512)):
        ids = [int(i) for i in train_ids if base.shape_at(int(i)) == (height, width)]
        # The largest teacher count gives the largest packed pair batch.
        ids.sort(key=lambda i: -valid[i].sum())
        ids = ids[:max(1, min(32, 196608 // (height * width)))]
        items = [base[i] for i in ids]
        maps = torch.stack([torch.cat([item["map"], item["distance_field"]], 0) for item in items]).cuda()
        starts, goals = (torch.stack([item[key] for item in items]).cuda() for key in ("start", "goal"))
        label_routes, label_valid = torch.from_numpy(routes[ids]).cuda(), torch.from_numpy(valid[ids]).cuda()
        torch.cuda.reset_peak_memory_stats()
        loss = trainer.sinkhorn_cfm_loss(model, maps, starts, goals, label_routes, label_valid,
            SimpleNamespace(**config), generator=torch.Generator(device="cuda").manual_seed(37))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(loss) and torch.isfinite(gradient)
        assert model.vector_field.out.weight.grad.abs().sum() > 0
        optimizer.step()
        points, _, active = arc_length_samples(label_routes[label_valid.bool()], height, width, 128)
        assert active.all() and torch.isfinite(points).all()
        row = {"shape": f"{height}x{width}", "tasks": ids, "batch_size": len(ids),
               "teacher_counts": label_valid.sum(-1).tolist(), "loss": float(loss.detach()),
               "gradient_norm_before_clip": float(gradient),
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del maps, starts, goals, label_routes, label_valid, loss
    assert sha(args.checkpoint) == source_hash
    report = {"status": "complete", "finished_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "source_checkpoint_sha256": source_hash, "source_checkpoint_unchanged": True,
              "smoke_weights_discarded": True, "formal_training": False,
              "train_tasks": len(train_ids), "validation_tasks": len(val_ids), "batches": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
