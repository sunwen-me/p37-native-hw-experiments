"""Offline geometry scoring helpers used by the path evaluators.

The boolean scorer is kept for backwards compatibility.  The detailed scorer
uses the same geometry protocol and additionally exposes collision, length,
smoothness and clearance metrics for selected and oracle comparisons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def candidate_metrics(candidate, occupancy, start, goal, geometry):
    """Return exact geometry and path-quality metrics for one candidate."""
    from utils.robot_geometry import FootprintChecker, evaluate_robot_path

    import numpy as np

    world = np.asarray(candidate, dtype=np.float64)
    result = evaluate_robot_path(
        world, occupancy, start, goal, geometry, endpoint_tolerance=1e-5,
    )
    checker = FootprintChecker(occupancy, geometry)
    segment_clearances = [
        checker.segment_clearance(a, b) for a, b in zip(world[:-1], world[1:])
    ]
    point_clearance = checker.segment_clearance(world[0], world[0])
    minimum_clearance = min([point_clearance, *segment_clearances])
    differences = np.diff(world, axis=0)
    second_differences = np.diff(world, n=2, axis=0)
    return {
        "success": bool(result["success"]),
        "safe": bool(result["safe"]),
        "completed_safe": bool(result["completed_safe"]),
        "start_error": float(result["start_error"]),
        "goal_error": float(result["goal_error"]),
        "length": float(np.linalg.norm(differences, axis=1).sum()) if len(differences) else 0.0,
        "smoothness": float(np.linalg.norm(second_differences, axis=1).mean())
        if len(second_differences) else 0.0,
        "minimum_clearance": float(minimum_clearance),
    }


def score_task_metrics(job):
    """Return detailed exact metrics for every candidate in one task."""
    source_project, candidates, occupancy, start, goal, geometry_json = job
    sys.path.insert(0, source_project)
    from utils.robot_geometry import RobotGeometry

    geometry = RobotGeometry.from_json(geometry_json)
    return [candidate_metrics(candidate, occupancy, start, goal, geometry)
            for candidate in candidates]


def score_task(job):
    """Return only success booleans for legacy benchmark callers."""
    source_project, candidates, occupancy, start, goal, geometry_json = job
    sys.path.insert(0, source_project)
    from utils.robot_geometry import RobotGeometry

    # This exact local-AABB implementation avoids comparing every path segment
    # with every occupied cell in a large map.  It uses the same closed-cell,
    # swept-disk geometry as the detailed full-map scorer; the radius+2 cap is
    # only used to certify clearance safely above the collision threshold.
    source_root = Path(source_project).resolve()
    sys.path.insert(0, str(source_root.parent / "map_flow_matching_p1"))
    from scripts.generate_native_hw_dataset import fast_segment_clearance

    geometry = RobotGeometry.from_json(geometry_json)
    radius = geometry.radius_grid
    image = np.asarray(occupancy)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    result = []
    for candidate in candidates:
        path = np.asarray(candidate, dtype=np.float64)
        if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 1
                or not np.isfinite(path).all()):
            result.append(False)
            continue
        start_error = float(np.linalg.norm(path[0] - start))
        goal_error = float(np.linalg.norm(path[-1] - goal))
        if start_error > 1e-5 or goal_error > 1e-5:
            result.append(False)
            continue
        clearances = [fast_segment_clearance(image, path[0], path[0], radius)]
        clearances.extend(
            fast_segment_clearance(image, a, b, radius)
            for a, b in zip(path[:-1], path[1:])
        )
        if min(clearances) <= radius + 1e-10:
            result.append(False)
            continue
        # Match evaluate_robot_path's completed-path check, including the tiny
        # endpoint links rather than assuming exact floating-point equality.
        endpoint_links = (
            fast_segment_clearance(image, start, path[0], radius),
            fast_segment_clearance(image, path[-1], goal, radius),
        )
        result.append(min(endpoint_links) > radius + 1e-10)
    return result
