import numpy as np

from navverse.navmesh import navmesh_surface_is_reachable, navmesh_surface_z


def _two_level_navmesh():
    lower = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    upper = lower.copy()
    upper[:, 2] = 3.5
    vertices = np.concatenate([upper, lower], axis=0).astype(np.float32)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return vertices, triangles


def test_overlapping_navmesh_selects_surface_nearest_current_level():
    vertices, triangles = _two_level_navmesh()

    assert navmesh_surface_z(vertices, triangles, [0.2, 0.2], reference_z=0.1) == 0.0
    assert navmesh_surface_z(vertices, triangles, [0.2, 0.2], reference_z=3.6) == 3.5


def test_nearest_vertex_fallback_stays_on_current_level():
    vertices, triangles = _two_level_navmesh()

    assert navmesh_surface_z(vertices, triangles, [1.1, 1.1], reference_z=0.1) == 0.0
    assert navmesh_surface_z(vertices, triangles, [1.1, 1.1], reference_z=3.6) == 3.5


def test_exact_surface_query_rejects_point_outside_navmesh():
    vertices, triangles = _two_level_navmesh()

    assert navmesh_surface_z(
        vertices,
        triangles,
        [1.1, 1.1],
        reference_z=0.1,
        allow_nearest_fallback=False,
    ) is None


def test_surface_height_continuity_rejects_floor_changes():
    assert navmesh_surface_is_reachable(0.04, 0.0, max_step_height=0.05)
    assert not navmesh_surface_is_reachable(0.06, 0.0, max_step_height=0.05)
    assert not navmesh_surface_is_reachable(3.5, 0.0, max_step_height=0.05)
