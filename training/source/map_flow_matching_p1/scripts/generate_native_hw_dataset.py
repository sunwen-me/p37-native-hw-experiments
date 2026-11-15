"""Generate native-resolution rectangular P37 supervision with exact safety labels.

Maps are generated directly in their requested H/W shape as connected room
layouts.  No existing square raster is resized or stretched.  Every route is
constructed through opened doors, resampled without removing corners, and
checked against the project's closed-cell swept-disk geometry protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_PROJECT = HERE.parents[1] / "map_flow_matching"
sys.path.insert(0, str(SOURCE_PROJECT))

from utils.robot_geometry import RobotGeometry, resample_safe_polyline  # noqa: E402


DEFAULT_SHAPES = (
    (64, 64), (64, 96), (96, 64), (96, 160), (160, 96),
    (127, 193), (193, 127), (128, 192), (192, 128),
    (256, 384), (384, 256), (512, 768), (768, 512),
)


def map_hash(map_data: np.ndarray) -> str:
    value = np.ascontiguousarray(map_data, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _segments_intersect_square(a, b, lo, hi):
    direction = b - a
    lower_t, upper_t = 0.0, 1.0
    for axis in range(2):
        if direction[axis] == 0:
            if a[axis] < lo[axis] or a[axis] > hi[axis]:
                return False
        else:
            left = (lo[axis] - a[axis]) / direction[axis]
            right = (hi[axis] - a[axis]) / direction[axis]
            lower_t = max(lower_t, min(left, right))
            upper_t = min(upper_t, max(left, right))
    return lower_t <= upper_t


def _segment_square_distance(a, b, lo, hi):
    if _segments_intersect_square(a, b, lo, hi):
        return 0.0
    endpoint_distance = min(
        np.linalg.norm(np.maximum(np.maximum(lo - point, point - hi), 0.0))
        for point in (a, b)
    )
    direction = b - a
    denominator = float(direction @ direction)
    corners = np.asarray([
        [lo[0], lo[1]], [lo[0], hi[1]],
        [hi[0], lo[1]], [hi[0], hi[1]],
    ])
    if denominator:
        fractions = np.clip(((corners - a) @ direction) / denominator, 0.0, 1.0)
    else:
        fractions = np.zeros(len(corners), dtype=np.float64)
    corner_distance = np.linalg.norm(corners - (a + fractions[:, None] * direction), axis=1).min()
    return float(min(endpoint_distance, corner_distance))


def fast_segment_clearance(map_data: np.ndarray, a, b, radius: float) -> float:
    """Exact segment-to-closed-cell clearance, restricted to a local AABB.

    The cap is a certified lower bound: any occupied square outside the search
    box is farther than ``radius + 2`` from the segment.
    """
    height, width = map_data.shape
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    boundary = min(
        a[0] + 0.5, b[0] + 0.5, a[1] + 0.5, b[1] + 0.5,
        width - 0.5 - a[0], width - 0.5 - b[0],
        height - 0.5 - a[1], height - 0.5 - b[1],
    )
    cap = float(radius + 2.0)
    if boundary <= 0:
        return float(boundary)
    reach = cap + 0.5
    x0 = max(0, int(math.floor(min(a[0], b[0]) - reach)))
    x1 = min(width - 1, int(math.ceil(max(a[0], b[0]) + reach)))
    y0 = max(0, int(math.floor(min(a[1], b[1]) - reach)))
    y1 = min(height - 1, int(math.ceil(max(a[1], b[1]) + reach)))
    local_y, local_x = np.nonzero(map_data[y0:y1 + 1, x0:x1 + 1] > 0.5)
    if not len(local_x):
        return float(min(boundary, cap))
    clearance = min(boundary, cap)
    for x, y in zip(local_x + x0, local_y + y0):
        lo = np.asarray([x - 0.5, y - 0.5], dtype=np.float64)
        clearance = min(clearance, _segment_square_distance(a, b, lo, lo + 1.0))
        if clearance <= radius:
            return float(clearance)
    return float(clearance)


def _grid_edges(rows: int, cols: int, rng: np.random.Generator):
    edges = []
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + cols))
    order = rng.permutation(len(edges))
    parent = list(range(rows * cols))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    opened = set()
    for index in order:
        left, right = edges[int(index)]
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left
            opened.add((min(left, right), max(left, right)))
    for index, edge in enumerate(edges):
        normalized = (min(edge), max(edge))
        if normalized not in opened and rng.random() < 0.28:
            opened.add(normalized)
    return edges, opened


def _make_layout(height: int, width: int, rng: np.random.Generator):
    rows = max(2, min(12, int(round(height / 48))))
    cols = max(2, min(12, int(round(width / 48))))
    def jittered_bounds(length, cells):
        bounds = np.rint(np.linspace(0, length, cells + 1)).astype(int)
        nominal_gap = int(np.min(np.diff(bounds)))
        jitter = min(4, max(1, nominal_gap // 8))
        for index in range(1, cells):
            lower = bounds[index - 1] + 16
            upper = bounds[index + 1] - 16
            bounds[index] = int(np.clip(
                bounds[index] + rng.integers(-jitter, jitter + 1),
                lower, upper,
            ))
        return bounds

    # Native pixel-level boundary jitter prevents small maps from collapsing
    # into a tiny set of repeated raster layouts while preserving room width.
    y_bounds = jittered_bounds(height, rows)
    x_bounds = jittered_bounds(width, cols)
    occupancy = np.zeros((height, width), dtype=np.uint8)
    edges, opened = _grid_edges(rows, cols, rng)
    doors = {}
    door_width = 9
    half = door_width // 2

    for boundary_index in range(1, cols):
        boundary_x = int(x_bounds[boundary_index])
        occupancy[:, boundary_x - 1:boundary_x + 1] = 1
    for boundary_index in range(1, rows):
        boundary_y = int(y_bounds[boundary_index])
        occupancy[boundary_y - 1:boundary_y + 1, :] = 1

    for left, right in sorted(opened):
        row_left, col_left = divmod(left, cols)
        row_right, col_right = divmod(right, cols)
        if row_left == row_right:
            boundary_x = int(x_bounds[max(col_left, col_right)])
            row = row_left
            low, high = int(y_bounds[row]), int(y_bounds[row + 1])
            center = (low + high) // 2 + int(rng.integers(-3, 4))
            center = int(np.clip(center, low + half + 2, high - half - 2))
            occupancy[center - half:center + half + 1, boundary_x - 1:boundary_x + 1] = 0
            point = (boundary_x - 0.5, float(center))
        else:
            boundary_y = int(y_bounds[max(row_left, row_right)])
            col = col_left
            low, high = int(x_bounds[col]), int(x_bounds[col + 1])
            center = (low + high) // 2 + int(rng.integers(-3, 4))
            center = int(np.clip(center, low + half + 2, high - half - 2))
            occupancy[boundary_y - 1:boundary_y + 1, center - half:center + half + 1] = 0
            point = (float(center), boundary_y - 0.5)
        doors[(left, right)] = point

    adjacency = {node: [] for node in range(rows * cols)}
    for left, right in opened:
        point = doors[(min(left, right), max(left, right))]
        adjacency[left].append((right, point))
        adjacency[right].append((left, point))
    rooms = []
    for row in range(rows):
        for col in range(cols):
            x0 = int(x_bounds[col]) + (1 if col else 2)
            x1 = int(x_bounds[col + 1]) - (2 if col + 1 < cols else 3)
            y0 = int(y_bounds[row]) + (1 if row else 2)
            y1 = int(y_bounds[row + 1]) - (2 if row + 1 < rows else 3)
            rooms.append((x0, x1, y0, y1))
    return occupancy, adjacency, rooms, rows, cols


def _room_route(adjacency, start_room: int, goal_room: int):
    parent = {start_room: None}
    via = {}
    queue = deque([start_room])
    while queue:
        node = queue.popleft()
        if node == goal_room:
            break
        for neighbor, door_point in adjacency[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                via[neighbor] = door_point
                queue.append(neighbor)
    if goal_room not in parent:
        raise RuntimeError("room graph must be connected")
    nodes = [goal_room]
    while nodes[-1] != start_room:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    door_points = [via[nodes[index + 1]] for index in range(len(nodes) - 1)]
    return nodes, door_points


def _sample_in_room(room, rng):
    x0, x1, y0, y1 = room
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise RuntimeError(f"room too small for footprint-safe endpoints: {room}")
    return np.asarray([
        float(rng.integers(x0 + 3, x1 - 2)),
        float(rng.integers(y0 + 3, y1 - 2)),
    ], dtype=np.float64)


def _make_map_tasks(height, width, seed, pairs_per_map, horizon, geometry):
    rng = np.random.default_rng(seed)
    occupancy, adjacency, rooms, rows, cols = _make_layout(height, width, rng)
    task_starts, task_goals, task_paths, clearances, lengths = [], [], [], [], []
    room_count = rows * cols
    for task_index in range(pairs_per_map):
        start_room = int(rng.integers(room_count))
        goal_room = int(rng.integers(room_count - 1))
        if goal_room >= start_room:
            goal_room += 1
        nodes, door_points = _room_route(adjacency, start_room, goal_room)
        start = _sample_in_room(rooms[start_room], rng)
        goal = _sample_in_room(rooms[goal_room], rng)
        # Keep each graph transition inside its own room.  Connecting two
        # door centers directly can cut across a room corner and clip the
        # closed cells at a perpendicular wall, especially on rectangular
        # maps.  A room-center waypoint makes every room segment stay in its
        # free rectangular interior before crossing the selected door.
        waypoints = [start]
        for room_position, room_id in enumerate(nodes):
            x0, x1, y0, y1 = rooms[room_id]
            center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            waypoints.append(center)
            if room_position < len(door_points):
                waypoints.append(door_points[room_position])
        waypoints.append(goal)
        waypoints = np.asarray(waypoints, dtype=np.float64)
        segment_clearances = [
            fast_segment_clearance(occupancy, left, right, geometry.radius_grid)
            for left, right in zip(waypoints[:-1], waypoints[1:])
        ]
        minimum_clearance = float(min(segment_clearances))
        if minimum_clearance <= geometry.radius_grid + 1e-10:
            raise RuntimeError(
                f"unsafe generated route map={height}x{width}, seed={seed}, "
                f"task={task_index}, clearance={minimum_clearance:.6f}"
            )
        path = resample_safe_polyline(waypoints, horizon=horizon)
        task_starts.append(start.astype(np.float32))
        task_goals.append(goal.astype(np.float32))
        task_paths.append(path)
        clearances.append(minimum_clearance)
        lengths.append(float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum()))
    return occupancy, np.asarray(task_starts), np.asarray(task_goals), np.asarray(task_paths), np.asarray(clearances), np.asarray(lengths), rows, cols


def _parse_shapes(raw):
    if not raw:
        return list(DEFAULT_SHAPES)
    output = []
    for item in raw.split(","):
        fields = item.lower().split("x")
        if len(fields) != 2:
            raise ValueError(f"shape must be HxW, got {item!r}")
        height, width = map(int, fields)
        if min(height, width) < 64:
            raise ValueError("native training map dimensions must be at least 64")
        output.append((height, width))
    if not output:
        raise ValueError("at least one shape is required")
    return output


def _object_array(values):
    output = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        output[index] = value
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--map-count", type=int, default=9300)
    parser.add_argument("--pairs-per-map", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--shapes", default=None, help="Comma-separated HxW list")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.map_count < 2 or args.pairs_per_map < 1 or args.horizon < 2:
        parser.error("map-count >= 2, pairs-per-map >= 1, horizon >= 2 required")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction must lie strictly between 0 and 1")
    shapes = _parse_shapes(args.shapes)
    geometry = RobotGeometry(radius=1.0, safety_margin=0.5, resolution=1.0)
    rng = np.random.default_rng(args.seed)
    shape_indices = np.arange(args.map_count) % len(shapes)
    rng.shuffle(shape_indices)

    unique_maps, unique_hashes, scene_types, map_shapes = [], [], [], []
    map_seeds, map_rows, map_cols = [], [], []
    all_starts, all_goals, all_paths, all_map_ids = [], [], [], []
    all_clearances, all_lengths, all_hashes = [], [], []
    map_hash_set = set()
    duplicate_retries = 0
    started = time.time()
    for map_index, shape_index in enumerate(shape_indices):
        height, width = shapes[int(shape_index)]
        for attempt in range(1000):
            map_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            result = _make_map_tasks(
                height, width, map_seed, args.pairs_per_map, args.horizon, geometry,
            )
            occupancy, starts, goals, paths, clearances, lengths, rows, cols = result
            digest = map_hash(occupancy)
            if digest not in map_hash_set:
                break
            duplicate_retries += 1
        else:
            raise RuntimeError(
                f"could not generate a unique {height}x{width} map after 1000 attempts"
            )
        map_hash_set.add(digest)
        unique_maps.append(occupancy)
        unique_hashes.append(digest)
        scene_types.append("connected_room_grid")
        map_shapes.append((height, width))
        map_seeds.append(map_seed)
        map_rows.append(rows)
        map_cols.append(cols)
        all_starts.extend(starts)
        all_goals.extend(goals)
        all_paths.extend(paths)
        all_clearances.extend(clearances)
        all_lengths.extend(lengths)
        all_map_ids.extend([map_index] * args.pairs_per_map)
        all_hashes.extend([digest] * args.pairs_per_map)
        if args.progress_every > 0 and ((map_index + 1) % args.progress_every == 0 or map_index + 1 == args.map_count):
            elapsed = time.time() - started
            print(json.dumps({
                "maps": map_index + 1, "total_maps": args.map_count,
                "shape": f"{height}x{width}",
                "tasks": len(all_starts), "elapsed_seconds": round(elapsed, 1),
            }), flush=True)

    map_hash_array = np.asarray(unique_hashes)
    task_hash_array = np.asarray(all_hashes)
    task_map_ids = np.asarray(all_map_ids, dtype=np.int32)
    starts = np.asarray(all_starts, dtype=np.float32)
    goals = np.asarray(all_goals, dtype=np.float32)
    paths = np.asarray(all_paths, dtype=np.float32)
    clearances = np.asarray(all_clearances, dtype=np.float64)
    lengths = np.asarray(all_lengths, dtype=np.float64)
    task_count = len(starts)
    split_rng = np.random.default_rng(args.seed + 900_001)
    map_order = split_rng.permutation(args.map_count)
    val_map_count = max(1, int(round(args.val_fraction * args.map_count)))
    val_map_ids = np.sort(map_order[:val_map_count])
    train_map_ids = np.sort(map_order[val_map_count:])
    val_mask = np.isin(task_map_ids, val_map_ids)
    train_ids = np.flatnonzero(~val_mask).astype(np.int64)
    val_ids = np.flatnonzero(val_mask).astype(np.int64)
    if set(map_hash_array[train_map_ids]) & set(map_hash_array[val_map_ids]):
        raise RuntimeError("map-level train/validation hash overlap detected")

    geometry_rng = np.random.default_rng(args.seed + 900_003)
    selected_geometry = []
    for shape in sorted(set(map_shapes)):
        maps_of_shape = np.asarray([
            index for index, value in enumerate(map_shapes) if tuple(value) == tuple(shape)
            and index in set(val_map_ids.tolist())
        ], dtype=np.int64)
        geometry_rng.shuffle(maps_of_shape)
        for map_id in maps_of_shape[:16]:
            selected_geometry.append(int(map_id) * args.pairs_per_map)
    geometry_val_ids = np.asarray(selected_geometry, dtype=np.int64)
    if len(geometry_val_ids) > 256:
        geometry_val_ids = np.sort(geometry_rng.choice(geometry_val_ids, 256, replace=False))

    routes_normalized = paths.copy()
    for map_id, (height, width) in enumerate(map_shapes):
        begin = map_id * args.pairs_per_map
        end = begin + args.pairs_per_map
        routes_normalized[begin:end, :, 0] = routes_normalized[begin:end, :, 0] / ((width - 1) / 2.0) - 1.0
        routes_normalized[begin:end, :, 1] = routes_normalized[begin:end, :, 1] / ((height - 1) / 2.0) - 1.0

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    generation_config = {
        "generator": "native_room_grid_v1",
        "seed": int(args.seed), "map_count": int(args.map_count),
        "pairs_per_map": int(args.pairs_per_map), "task_count": int(task_count),
        "horizon": int(args.horizon), "shapes": [list(v) for v in shapes],
        "shape_counts": {
            f"{height}x{width}": int(sum(tuple(v) == (height, width) for v in map_shapes))
            for height, width in sorted(set(map_shapes))
        },
        "geometry": json.loads(geometry.to_json()),
        "route_protocol": "shortest path in the generated connected room-door graph; exact swept-disk checked",
        "split_protocol": "map-disjoint random 90/10 split; hashes include H/W",
        "limitations": [
            "Synthetic connected room-grid distribution; not a substitute for real robot occupancy-map domains.",
            "One route label per endpoint task; the label follows a shortest path in the generated room graph.",
            "No unreachable endpoint tasks are included in this supervised positive-route set.",
        ],
    }
    np.savez_compressed(
        output / "train_data.npz",
        unique_maps=_object_array(unique_maps),
        unique_map_hashes=map_hash_array,
        map_ids=task_map_ids,
        map_hashes=task_hash_array,
        starts=starts, goals=goals, paths=paths,
        path_lengths=np.full(task_count, args.horizon, dtype=np.int32),
        geometry_json=np.asarray(geometry.to_json()),
        generation_json=np.asarray(json.dumps(generation_config, sort_keys=True)),
        scene_types=np.asarray(scene_types, dtype="U32")[task_map_ids],
    )
    np.savez_compressed(
        output / "p37_routes.npz",
        fine_routes=routes_normalized[:, None].astype(np.float32),
        routes=routes_normalized[:, None, np.linspace(0, args.horizon - 1, 8).round().astype(int)].astype(np.float32),
        valid=np.ones((task_count, 1), dtype=np.float32),
        map_hashes=task_hash_array, map_ids=task_map_ids,
        geometry_json=np.asarray(geometry.to_json()),
    )
    np.savez_compressed(
        output / "split.npz",
        train_ids=train_ids, val_ids=val_ids,
        training_map_hashes=task_hash_array,
        train_map_ids=train_map_ids, val_map_ids=val_map_ids,
        geometry_val_ids=geometry_val_ids,
    )
    np.savez_compressed(
        output / "safety_audit.npz",
        safe=np.ones(task_count, dtype=bool),
        minimum_clearance_lower_bound=clearances,
        path_length=lengths,
        map_hashes=task_hash_array,
    )

    checksums = {}
    for name in ("train_data.npz", "p37_routes.npz", "split.npz", "safety_audit.npz"):
        digest = hashlib.sha256()
        with (output / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        checksums[name] = digest.hexdigest()
    manifest = generation_config | {
        "status": "complete",
        "unique_maps": len(unique_maps), "tasks": task_count,
        "train_maps": len(train_map_ids), "validation_maps": len(val_map_ids),
        "train_tasks": len(train_ids), "validation_tasks": len(val_ids),
        "train_val_hash_overlap": 0,
        "duplicate_map_retries": duplicate_retries,
        "minimum_certified_clearance": float(clearances.min()),
        "maximum_certified_clearance_lower_bound": float(clearances.max()),
        "elapsed_seconds": round(time.time() - started, 3),
        "sha256": checksums,
    }
    (output / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
