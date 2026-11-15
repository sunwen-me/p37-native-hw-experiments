"""Create a map-disjoint, scene-stratified P37 supervised pilot subset.

The script reads an audited source bundle and writes derived NPZ files plus a
hash/selection manifest into a new output directory. It never edits the source
bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_maps(pool_by_scene, count: int, rng: np.random.Generator) -> list[int]:
    scenes = sorted(pool_by_scene)
    base, remainder = divmod(int(count), len(scenes))
    chosen = []
    for index, scene in enumerate(scenes):
        quota = base + int(index < remainder)
        pool = np.asarray(pool_by_scene[scene], dtype=np.int64)
        if len(pool) < quota:
            raise ValueError(
                f"scene {scene!r} has {len(pool)} maps but {quota} were requested"
            )
        chosen.extend(rng.choice(pool, size=quota, replace=False).tolist())
    return sorted(map(int, chosen))


def write_pilot(source_dir: Path, output_dir: Path, train_map_count: int,
                val_map_count: int, geometry_val_count: int, seed: int) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"pilot data output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ("train_data.npz", "p37_routes.npz", "split.npz", "safety_audit.npz")
    source_paths = {name: source_dir / name for name in required}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source bundle is missing: {missing}")

    data = np.load(source_paths["train_data.npz"], allow_pickle=True)
    labels = np.load(source_paths["p37_routes.npz"], allow_pickle=False)
    split = np.load(source_paths["split.npz"], allow_pickle=False)
    safety = np.load(source_paths["safety_audit.npz"], allow_pickle=False)
    row_count = len(data["maps"])
    map_ids = np.asarray(data["map_ids"], dtype=np.int64)
    map_hashes = np.asarray(data["map_hashes"])
    route_map_ids = np.asarray(labels["map_ids"], dtype=np.int64)
    route_hashes = np.asarray(labels["map_hashes"])
    if len(labels["fine_routes"]) != row_count or not np.array_equal(map_ids, route_map_ids):
        raise ValueError("train_data and P37 route rows are not aligned")
    if not np.array_equal(map_hashes, route_hashes):
        raise ValueError("train_data and P37 route map hashes do not align")
    if not np.array_equal(safety["map_hashes"], map_hashes):
        raise ValueError("safety audit hashes do not align with the source data")
    if not np.all(safety["safe"]):
        raise ValueError("the audited source contains unsafe route rows")
    if not np.all((labels["valid"] == 0) | (labels["valid"] == 1)):
        raise ValueError("route validity must be binary")
    if not np.all(labels["valid"].sum(axis=1) > 0):
        raise ValueError("every pilot row must have at least one valid teacher")
    height, width = data["maps"].shape[-2:]
    pixel_scale = np.asarray(
        [(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32,
    )
    normalized_paths = data["paths"] / pixel_scale - 1.0
    if not np.allclose(
        normalized_paths, labels["fine_routes"][:, 0], rtol=0, atol=2e-6,
    ):
        raise ValueError("pixel paths and normalized fine_routes do not match")

    train_ids = np.asarray(split["train_ids"], dtype=np.int64)
    val_ids = np.asarray(split["val_ids"], dtype=np.int64)
    if (len(np.unique(np.concatenate([train_ids, val_ids]))) != row_count
            or not np.array_equal(
                np.sort(np.concatenate([train_ids, val_ids])),
                np.arange(row_count, dtype=np.int64),
            )):
        raise ValueError("official split does not partition source rows exactly once")
    if not np.array_equal(split["training_map_hashes"], map_hashes):
        raise ValueError("official split hashes do not align with source data")

    scenes = np.asarray(data["scene_types"]).astype(str)
    map_to_scene = {}
    for map_id in np.unique(map_ids):
        rows = np.flatnonzero(map_ids == map_id)
        values = np.unique(scenes[rows])
        hashes = np.unique(map_hashes[rows])
        if len(values) != 1 or len(hashes) != 1:
            raise ValueError(f"map {map_id} has inconsistent scene or hash metadata")
        map_to_scene[int(map_id)] = str(values[0])

    train_pool, val_pool = {}, {}
    for ids, pool in ((train_ids, train_pool), (val_ids, val_pool)):
        for map_id in np.unique(map_ids[ids]):
            scene = map_to_scene[int(map_id)]
            pool.setdefault(scene, []).append(int(map_id))
    if set(train_pool) != set(val_pool):
        raise ValueError("official train/validation split has different scene categories")

    rng = np.random.default_rng(int(seed))
    selected_train_maps = sample_maps(train_pool, train_map_count, rng)
    selected_val_maps = sample_maps(val_pool, val_map_count, rng)
    train_rows = np.flatnonzero(np.isin(map_ids, selected_train_maps))
    val_rows = np.flatnonzero(np.isin(map_ids, selected_val_maps))
    if len(np.intersect1d(
        np.unique(map_hashes[train_rows]), np.unique(map_hashes[val_rows]),
    )):
        raise ValueError("selected training and validation maps overlap")
    selected_rows = np.concatenate([train_rows, val_rows])
    train_local = np.arange(len(train_rows), dtype=np.int64)
    val_local = np.arange(len(train_rows), len(selected_rows), dtype=np.int64)
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError("pilot selection produced an empty split")
    if not np.allclose(
        data["paths"][selected_rows, 0], data["starts"][selected_rows],
        rtol=0, atol=1e-5,
    ) or not np.allclose(
        data["paths"][selected_rows, -1], data["goals"][selected_rows],
        rtol=0, atol=1e-5,
    ):
        raise ValueError("pilot path endpoints do not match task endpoints")

    data_keys = (
        "maps", "starts", "goals", "paths", "path_lengths", "map_ids",
        "map_hashes", "scene_types", "geometry_json", "generation_json",
    )
    label_keys = (
        "fine_routes", "valid", "map_ids", "map_hashes", "scene_types",
        "geometry_json", "generation_json",
    )
    np.savez_compressed(
        output_dir / "train_data.npz",
        **{key: data[key][selected_rows] if np.asarray(data[key]).ndim
           and len(data[key]) == row_count else data[key] for key in data_keys},
    )
    np.savez_compressed(
        output_dir / "p37_routes.npz",
        **{key: labels[key][selected_rows] if np.asarray(labels[key]).ndim
           and len(labels[key]) == row_count else labels[key] for key in label_keys},
    )
    source_hashes = np.asarray([
        hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for value in data["maps"][selected_rows]
    ])
    geometry_count = min(int(geometry_val_count), len(val_local))
    geometry_local = np.sort(rng.choice(val_local, size=geometry_count, replace=False))
    np.savez_compressed(
        output_dir / "split.npz",
        train_ids=train_local,
        val_ids=val_local,
        training_map_hashes=source_hashes,
        geometry_val_ids=geometry_local,
    )
    np.savez_compressed(
        output_dir / "safety_audit.npz",
        map_hashes=source_hashes,
        safe=safety["safe"][selected_rows],
        minimum_clearance_lower_bound=safety["minimum_clearance_lower_bound"][selected_rows],
    )
    scene_histogram = {
        scene: {
            "train_maps": sum(map_to_scene[map_id] == scene for map_id in selected_train_maps),
            "val_maps": sum(map_to_scene[map_id] == scene for map_id in selected_val_maps),
        }
        for scene in sorted(train_pool)
    }
    output_hashes = {
        name: sha256(output_dir / name)
        for name in ("train_data.npz", "p37_routes.npz", "split.npz", "safety_audit.npz")
    }
    manifest = {
        "seed": int(seed),
        "source_dir": str(source_dir),
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "output_sha256": output_hashes,
        "source_rows": int(row_count),
        "train_maps": selected_train_maps,
        "validation_maps": selected_val_maps,
        "train_tasks": int(len(train_rows)),
        "validation_tasks": int(len(val_rows)),
        "geometry_validation_tasks": int(geometry_count),
        "map_shape": list(data["maps"].shape[1:]),
        "scene_histogram": scene_histogram,
        "all_selected_routes_safety_audited": True,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-maps", type=int, default=64)
    parser.add_argument("--val-maps", type=int, default=16)
    parser.add_argument("--geometry-val-tasks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    if min(args.train_maps, args.val_maps, args.geometry_val_tasks) < 1:
        parser.error("map and geometry-validation counts must be positive")
    print(json.dumps(write_pilot(
        args.source_dir, args.output_dir, args.train_maps, args.val_maps,
        args.geometry_val_tasks, args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
