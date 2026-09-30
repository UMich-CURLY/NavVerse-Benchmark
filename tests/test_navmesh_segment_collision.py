import numpy as np

from navverse.navmesh import navmesh_segment_is_clear


class _FakeNavmesh:
    navmesh_v = np.array(
        [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    navmesh_t = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    def __init__(self, wall_distance):
        self.wall_distance = wall_distance

    def distance_to_wall(self, point):
        return self.wall_distance(np.asarray(point, dtype=np.float32))


def test_segment_rejects_obstacle_between_clear_endpoints():
    navmesh = _FakeNavmesh(lambda point: abs(float(point[0]) - 0.5))

    assert not navmesh_segment_is_clear(
        navmesh,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        clearance=0.2,
        resolution=0.05,
        max_step_height=0.05,
    )


def test_segment_accepts_clear_straight_motion():
    navmesh = _FakeNavmesh(lambda _point: 1.0)

    assert navmesh_segment_is_clear(
        navmesh,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        clearance=0.28,
        resolution=0.05,
        max_step_height=0.05,
    )


def test_segment_rejects_motion_outside_navmesh_even_if_wall_query_is_clear():
    navmesh = _FakeNavmesh(lambda _point: 1.0)

    assert not navmesh_segment_is_clear(
        navmesh,
        [0.5, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        clearance=0.28,
        resolution=0.05,
        max_step_height=0.05,
    )
