import numpy as np

from scripts.geometry_eval import score_task
from utils.robot_geometry import FootprintChecker, RobotGeometry


def test_local_aabb_score_matches_full_swept_disk_checker():
    occupancy = np.zeros((48, 64), dtype=np.uint8)
    occupancy[17:31, 29:34] = 1
    geometry = RobotGeometry(radius=1.0, safety_margin=0.5, resolution=1.0)
    start = np.asarray([5.0, 8.0])
    goal = np.asarray([58.0, 39.0])
    safe_path = np.asarray([
        start, [20.0, 12.0], [25.0, 10.0],
        [38.0, 10.0], [45.0, 20.0], goal,
    ])
    colliding_path = np.asarray([
        start, [15.0, 14.0], [20.0, 18.0],
        [30.0, 22.0], [40.0, 26.0], goal,
    ])
    checker = FootprintChecker(occupancy, geometry)
    expected = [checker.check_path(path)[0] for path in (safe_path, colliding_path)]
    actual = score_task((
        "training/source/map_flow_matching",
        [safe_path, colliding_path], occupancy, start, goal,
        geometry.to_json(),
    ))
    assert actual == expected == [True, False]


def test_local_aabb_score_respects_map_exterior():
    occupancy = np.zeros((32, 48), dtype=np.uint8)
    geometry = RobotGeometry(radius=1.0, safety_margin=0.5, resolution=1.0)
    start = np.asarray([1.0, 12.0])
    goal = np.asarray([45.0, 12.0])
    path = np.asarray([start, [24.0, 12.0], goal])
    result = score_task((
        "training/source/map_flow_matching", np.asarray([path]), occupancy,
        start, goal, geometry.to_json(),
    ))
    assert result == [False]
