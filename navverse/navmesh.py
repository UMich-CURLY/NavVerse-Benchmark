import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
try:
    from pxr import UsdGeom, Gf, Usd
except ImportError:
    UsdGeom = None
    Gf = None
    Usd = None
try:
    import PyRecastDetour as pyrecast
except ImportError:
    raise ImportError("PyRecastDetour not found; run bash navverse_tools/install_pyrecastdetour.sh")


_NAVMESH_CACHE = {}
INNOUT_NAVMESH_BUILD_VERSION = 4


def _require_pxr():
    if UsdGeom is None or Gf is None or Usd is None:
        raise ImportError("pxr is required for USD/stage-based navmesh operations")


def _config_value(config, *keys, default=None):
    value = config
    for key in keys:
        if value is None:
            return default
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            value = getattr(value, key, default)
    return value


def _navmesh_preprocess_enabled(scene_type, preprocess_config):
    enabled_scene_types = _config_value(preprocess_config, "enabled_scene_types", default=()) or ()
    return scene_type in set(enabled_scene_types)


def extract_distance_to_wall(value):
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return extract_distance_to_wall(value.tolist())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("distance_to_wall returned an empty value")
        return extract_distance_to_wall(value[0])
    raise TypeError(f"Unsupported distance_to_wall return type: {type(value)}")


def navmesh_surface_z(
    vertices,
    triangles,
    xy,
    reference_z=None,
    allow_nearest_fallback=True,
):
    xy = np.asarray(xy, dtype=np.float32)
    candidates = []
    for tri in triangles:
        pts = vertices[tri]
        a, b, c = pts[0, :2], pts[1, :2], pts[2, :2]
        v0, v1, v2 = b - a, c - a, xy - a
        denom = float(v0[0] * v1[1] - v1[0] * v0[1])
        if abs(denom) < 1e-8:
            continue
        inv = 1.0 / denom
        u = float((v2[0] * v1[1] - v1[0] * v2[1]) * inv)
        v = float((v0[0] * v2[1] - v2[0] * v0[1]) * inv)
        w = 1.0 - u - v
        if u >= -1e-4 and v >= -1e-4 and w >= -1e-4:
            candidates.append(float(w * pts[0, 2] + u * pts[1, 2] + v * pts[2, 2]))
    if candidates:
        if reference_z is None:
            return candidates[0]
        return min(candidates, key=lambda z: abs(z - float(reference_z)))

    if not allow_nearest_fallback:
        return None
    distances = np.linalg.norm(vertices[:, :2] - xy[None, :], axis=1)
    if reference_z is None:
        return float(vertices[int(np.argmin(distances)), 2])
    height_error = np.abs(vertices[:, 2] - float(reference_z))
    return float(vertices[int(np.argmin(distances + height_error)), 2])


def navmesh_surface_is_reachable(surface_z, reference_z, max_step_height):
    return abs(float(surface_z) - float(reference_z)) <= float(max_step_height) + 1e-4


def navmesh_segment_is_clear(
    navmesh_interface,
    start,
    target,
    clearance,
    resolution,
    max_step_height,
):
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    distance = float(np.linalg.norm(target[:2] - start[:2]))
    steps = max(int(np.ceil(distance / max(float(resolution), 1e-3))), 1)
    reference_z = float(start[2])
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        sample = start + (target - start) * alpha
        surface_z = navmesh_surface_z(
            navmesh_interface.navmesh_v,
            navmesh_interface.navmesh_t,
            sample[:2],
            reference_z=reference_z,
            allow_nearest_fallback=False,
        )
        if surface_z is None or not navmesh_surface_is_reachable(
            surface_z, reference_z, max_step_height
        ):
            return False
        sample[2] = surface_z
        wall_distance = extract_distance_to_wall(
            navmesh_interface.distance_to_wall(sample.tolist())
        )
        if not np.isfinite(wall_distance) or wall_distance + 1e-4 < float(clearance):
            return False
        reference_z = float(surface_z)
    return True


def load_navmesh_for_scene(scene_folder, scene_id, up_axis="Z"):
    if scene_folder is None or scene_id is None:
        return None

    navmesh_path = Path(scene_folder) / "navmesh" / f"{scene_id}_navmesh.bin"
    cache_key = (str(navmesh_path), up_axis)
    if cache_key in _NAVMESH_CACHE:
        return _NAVMESH_CACHE[cache_key]

    if not navmesh_path.exists():
        _NAVMESH_CACHE[cache_key] = None
        return None

    navmesh_interface = NavmeshInterface(up_axis=up_axis)
    navmesh_interface.load_navmesh(str(navmesh_path))
    _NAVMESH_CACHE[cache_key] = navmesh_interface
    return navmesh_interface


def load_navmesh_for_episode(env, episode, up_axis="Z"):
    args = getattr(env, "args", None)
    scene_folder = getattr(args, "scene_folder", None)
    scene_id = episode.get("scene_id", None)
    return load_navmesh_for_scene(scene_folder, scene_id, up_axis=up_axis)

def _path_is_excluded(path, exclude_paths, exclude_markers=()):
    path = str(path)
    for exclude_path in exclude_paths:
        exclude_path = str(exclude_path).rstrip("/")
        if path == exclude_path or path.startswith(f"{exclude_path}/"):
            return True
    for marker in exclude_markers:
        if path.endswith(marker) or f"{marker}/" in path:
            return True
    return False


def _non_physical_geometry_markers(scene_type, preprocess_config=None):
    if not _navmesh_preprocess_enabled(scene_type, preprocess_config):
        return ()
    config = preprocess_config
    return tuple(_config_value(config, "excluded_geometry_markers", default=()) or ())


def get_all_stage_mesh(
    stage,
    prims,
    exclude_paths=None,
    exclude_markers=(),
    include_invisible_roots=False,
):
    _require_pxr()
    exclude_paths = exclude_paths or ()

    found_meshes = []

    # For each selected prim, go through its children and figure out if they are meshes
    for prim in prims:
        if (
            UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
            and not (include_invisible_roots and prim.IsA(UsdGeom.Mesh))
        ):
            continue

        if _path_is_excluded(prim.GetPath(), exclude_paths, exclude_markers):
            continue

        if prim.IsA(UsdGeom.Mesh):
            found_meshes.append(prim)
            continue
        # Traverse the scene graph and print the paths of prims, including instance proxies
        for x in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if UsdGeom.Imageable(x).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            if _path_is_excluded(x.GetPath(), exclude_paths, exclude_markers):
                continue
            if x.IsA(UsdGeom.Mesh):
                found_meshes.append(x)
    print("Length of found meshes: ", len(found_meshes))
    points, faces = get_mesh(found_meshes)
    # print("points",points)
    # print("faces",faces)
    return points, faces


def navmesh_preprocess_cache_key(scene_type, preprocess_config=None):
    if not _navmesh_preprocess_enabled(scene_type, preprocess_config):
        return None
    config = preprocess_config
    return {
        "version": 2,
        "type": "terrain_relative_height_window",
        "grid_cell_size": float(_config_value(config, "grid_cell_size")),
        "lower_margin": float(_config_value(config, "lower_margin")),
        "upper_margin": float(_config_value(config, "upper_margin")),
        "excluded_geometry_markers": list(_non_physical_geometry_markers(scene_type, preprocess_config)),
    }


def get_mesh(objs):
    points, faces = [],[]

    for obj in tqdm(objs, total=len(objs)):
        f_offset = len(points)
        # f, p = convert_to_mesh(obj)#usd_stage.GetPrimAtPath(obj))
        f, p = meshconvert(obj)#usd_stage.GetPrimAtPath(obj))
        p = np.array(p)
        f = np.array(f)

        if len(p) == 0:
            continue

        if p.max()-p.min() > 1000:
            print(f"[WARNING]: Large mesh found: {obj.GetPath()}, Max: {p.max()}, Min: {p.min()}")

        if p.max()>1000 or p.min()<-1000:
            print(f"[WARNING]: Remote mesh found: {obj.GetPath()}, Max: {p.max()}, Min: {p.min()}")
        # print("points shape: ", p.shape)  n x 3
        # print("faces shape: ", f.shape)   n x 3
        points.extend(p)
        faces.extend(f + f_offset)
    print("points",len(points))
    print("faces",len(faces))
    return points, faces

def meshconvert(prim):
    _require_pxr()

    # Create an XformCache object to efficiently compute world transforms
    xform_cache = UsdGeom.XformCache()

    # Get the mesh schema
    mesh = UsdGeom.Mesh(prim)
    
    # Get verts and triangles
    tris = mesh.GetFaceVertexIndicesAttr().Get()
    if not tris:
        return [], []
    tris_cnt = mesh.GetFaceVertexCountsAttr().Get()

    # Get the vertices in local space
    points_attr = mesh.GetPointsAttr()
    local_points = points_attr.Get()
    
    # Convert the VtVec3fArray to a NumPy array
    points_np = np.array(local_points, dtype=np.float64)
    
    # Add a fourth component (with value 1.0) to make the points homogeneous
    num_points = len(local_points)
    ones = np.ones((num_points, 1), dtype=np.float64)
    points_np = np.hstack((points_np, ones))

    # Compute the world transform for this prim
    world_transform = xform_cache.GetLocalToWorldTransform(prim)

    # Convert the GfMatrix to a NumPy array
    matrix_np = np.array(world_transform, dtype=np.float64).reshape((4, 4))

    # Transform all vertices to world space using matrix multiplication
    world_points = np.dot(points_np, matrix_np)
    if np.any(np.isnan(world_points)):
        print("world_points: ", world_points)
        print("world_points shape: ", world_points.shape)
        print("points_np: ", points_np)
        print("matrix_np: ", matrix_np)
        print("prim path: ", prim.GetPath())
        print("prim: ", prim)

    tri_list = convert_to_triangle_mesh(tris, tris_cnt)
    # tri_list = tri_list.flatten()

    world_points = world_points[:,:3]

    return np.array(tri_list), world_points

def convert_to_triangle_mesh(FaceVertexIndices, FaceVertexCounts):
    """
    Convert a list of vertices and a list of faces into a triangle mesh.
    
    A list of triangle faces, where each face is a list of indices of the vertices that form the face.
    """
    
    # Parse the face vertex indices into individual face lists based on the face vertex counts.

    faces = []
    start = 0
    for count in FaceVertexCounts:
        end = start + count
        face = FaceVertexIndices[start:end]
        faces.append(face)
        start = end

    # Convert all faces to triangles
    triangle_faces = []
    for face in faces:
        if len(face) < 3:
            newface = []  # Invalid face
        elif len(face) == 3:
            newface = [face]  # Already a triangle
        else:
            # Fan triangulation: pick the first vertex and connect it to all other vertices
            v0 = face[0]
            newface = [[v0, face[i], face[i + 1]] for i in range(1, len(face) - 1)]

        triangle_faces.extend(newface)
    
    return np.array(triangle_faces)
def get_bbox_center(prim):
    _require_pxr()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    bound = bbox_cache.ComputeWorldBound(prim)
    range3d = bound.ComputeAlignedBox()
    min_point = range3d.GetMin()
    max_point = range3d.GetMax()
    center = (min_point + max_point) / 2.0
    return np.array([center[0], center[1], center[2]])

class NavmeshInterface:
    def __init__(self, up_axis='Y', preprocess_config=None):
        self.built = False
        self.input_prim = None
        self.input_vert = None
        self.input_tri = None
        self.random_points = None
        self.wall_outline = []
        self.path_points = None
        self.start_pos = None
        self.end_pos = None
        self.navmesh_origin = np.zeros(3, dtype=np.float32)
        self.navmesh_preprocess_metadata = None
        self.scene_type = None
        self.preprocess_config = preprocess_config
        self.building_exclusion_bboxes = np.empty((0, 6), dtype=np.float32)
        self.nm = pyrecast.Navmesh()
        settings = self.nm.get_settings()
        settings["cellSize"] = 0.08
        settings["cellHeight"] = 0.1
        settings["agentHeight"] = 0.3
        settings["agentRadius"] = 0.3
        settings["agentMaxClimb"] = 0.05
        settings["agentMaxSlope"] = 26.0
        settings["regionMinSize"] = 0.5
        settings["regionMergeSize"] = 0.5
        settings["edgeMaxLen"] = 5.0
        settings["edgeMaxError"] = 1.3
        settings["vertsPerPoly"] = 6.0
        settings["detailSampleDist"] = 1.0
        settings["detailSampleMaxError"] = 1.0
        self.settings = settings
        # if z_up is true, we will need to do some conversion before sending to 
        # recast, then, we will convert it back to y_up (all functions will need to do that)
        self.z_up = up_axis == 'Z'
    
    def _snake_to_camel(self, s):
        parts = s.split('_')
        return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])
    
    def _camel_to_snake(self, s):
        return ''.join(['_' + char.lower() if char.isupper() else char for char in s]).lstrip('_')

    def update_settings(self, settings):
        for key, value in settings.items():
            self.settings[self._snake_to_camel(key)] = value

    def serialized_settings(self):
        return {key: float(value) for key, value in sorted(self.settings.items())}

    def _preprocess_config_value(self, key):
        return _config_value(self.preprocess_config, key)

    def _preprocess_enabled(self, scene_type):
        return _navmesh_preprocess_enabled(scene_type, self.preprocess_config)

    def _filter_degenerate_faces(self, area_epsilon=1e-8, chunk_size=1_000_000):
        faces_arr = np.array(self.input_tri, dtype=np.int32)
        if faces_arr.ndim == 1:
            faces_arr = faces_arr.reshape(-1, 3)
        valid_chunks = []
        removed_count = 0
        for start in range(0, len(faces_arr), chunk_size):
            faces = faces_arr[start:start + chunk_size]
            v0 = self.input_vert[faces[:, 0]]
            v1 = self.input_vert[faces[:, 1]]
            v2 = self.input_vert[faces[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            valid_mask = np.einsum("ij,ij->i", cross, cross) > area_epsilon
            removed_count += int((~valid_mask).sum())
            if np.any(valid_mask):
                valid_chunks.append(faces[valid_mask])
        if removed_count:
            self.input_tri = np.concatenate(valid_chunks, axis=0) if valid_chunks else np.empty((0, 3), dtype=np.int32)
            print(f"[INFO]: Removed {removed_count} degenerate faces after height constraint")

    def _set_local_navmesh_origin(self):
        self.navmesh_origin = np.zeros(3, dtype=np.float32)
        if self.input_vert is None or len(self.input_vert) == 0:
            return
        min_corner = self.input_vert.min(axis=0)
        max_corner = self.input_vert.max(axis=0)
        self.navmesh_origin = np.array(
            [
                (min_corner[0] + max_corner[0]) * 0.5,
                0.0,
                (min_corner[2] + max_corner[2]) * 0.5,
            ],
            dtype=np.float32,
        )
        self.input_vert = self.input_vert - self.navmesh_origin
        print(f"[INFO]: Shifted navmesh input by local origin {self.navmesh_origin.tolist()}")

    def _find_terrain_surface_prim(self, selected_prims):
        _require_pxr()
        terrain_candidates = []
        fallback_candidates = []
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
        for prim in selected_prims:
            for candidate in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if not candidate.IsA(UsdGeom.Mesh):
                    continue
                path = str(candidate.GetPath())
                if (
                    path.endswith("/city/Terrain/Terrain")
                    or "/city/Terrain/" in path
                    or path.endswith("/TerrainRegions/Terrain")
                    or path.endswith("/NavVerseTerrainCollision")
                ):
                    terrain_candidates.append(candidate)
                    continue
                if "Terrain" not in path or "/TerrainRegions/" in path:
                    continue
                try:
                    bound = bbox_cache.ComputeWorldBound(candidate).ComputeAlignedBox()
                    min_point = bound.GetMin()
                    max_point = bound.GetMax()
                    area = float(max_point[0] - min_point[0]) * float(max_point[1] - min_point[1])
                except Exception:
                    area = 0.0
                fallback_candidates.append((area, candidate))
        if terrain_candidates:
            return terrain_candidates[0]
        if fallback_candidates:
            return max(fallback_candidates, key=lambda item: item[0])[1]
        return None

    def _terrain_top_surface_points(self, terrain_prim):
        triangles, points = meshconvert(terrain_prim)
        if len(points) == 0 or len(triangles) == 0:
            return np.empty((0, 3), dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        v0 = points[triangles[:, 0]]
        v1 = points[triangles[:, 1]]
        v2 = points[triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area = np.linalg.norm(cross, axis=1)
        normal_z = np.divide(cross[:, 2], area, out=np.zeros_like(area), where=area > 1e-12)
        top_mask = normal_z > 0.5
        if not np.any(top_mask):
            top_mask = np.abs(normal_z) > 0.8
        if not np.any(top_mask):
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate([v0[top_mask], v1[top_mask], v2[top_mask]], axis=0)

    def _build_ground_height_grid(self, surface_points):
        from scipy import ndimage

        xy = surface_points[:, :2]
        z = surface_points[:, 2].astype(np.float32)
        cell_size = float(self._preprocess_config_value("grid_cell_size"))
        min_xy = xy.min(axis=0) - cell_size
        max_xy = xy.max(axis=0) + cell_size
        grid_shape = np.ceil((max_xy - min_xy) / cell_size).astype(np.int64) + 1
        grid_shape = np.maximum(grid_shape, 1)
        ix = np.clip(((xy[:, 0] - min_xy[0]) / cell_size).astype(np.int64), 0, grid_shape[0] - 1)
        iy = np.clip(((xy[:, 1] - min_xy[1]) / cell_size).astype(np.int64), 0, grid_shape[1] - 1)
        grid = np.full((grid_shape[1], grid_shape[0]), -np.inf, dtype=np.float32)
        np.maximum.at(grid, (iy, ix), z)
        missing = ~np.isfinite(grid)
        if np.any(missing):
            nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
            grid = grid[tuple(nearest)]
        return grid, min_xy.astype(np.float32), float(cell_size)

    @staticmethod
    def _lookup_ground_height(points, grid, min_xy, cell_size):
        ix = np.clip(((points[:, 0] - min_xy[0]) / cell_size).astype(np.int64), 0, grid.shape[1] - 1)
        iy = np.clip(((points[:, 1] - min_xy[1]) / cell_size).astype(np.int64), 0, grid.shape[0] - 1)
        return grid[iy, ix]

    def _apply_relative_height_window(self, selected_prims, scene_type):
        if not self._preprocess_enabled(scene_type):
            return
        preprocess_key = navmesh_preprocess_cache_key(scene_type, self.preprocess_config)
        terrain_prim = self._find_terrain_surface_prim(selected_prims)
        if terrain_prim is None:
            print("[WARNING]: Terrain surface prim not found; skipping relative height window")
            self.navmesh_preprocess_metadata = {
                **preprocess_key,
                "applied": False,
                "reason": "terrain_surface_prim_not_found",
            }
            return
        surface_points = self._terrain_top_surface_points(terrain_prim)
        if len(surface_points) == 0:
            print("[WARNING]: Terrain top surface points not found; skipping relative height window")
            self.navmesh_preprocess_metadata = {
                **preprocess_key,
                "applied": False,
                "reason": "terrain_top_surface_empty",
                "terrain_prim": str(terrain_prim.GetPath()),
            }
            return

        grid, min_xy, cell_size = self._build_ground_height_grid(surface_points)
        ground_z = self._lookup_ground_height(self.input_vert, grid, min_xy, cell_size)
        lower = ground_z - float(self._preprocess_config_value("lower_margin"))
        upper = ground_z + float(self._preprocess_config_value("upper_margin"))
        original_z = self.input_vert[:, 2].astype(np.float32, copy=True)

        faces_arr = np.asarray(self.input_tri, dtype=np.int32)
        if faces_arr.ndim == 1:
            faces_arr = faces_arr.reshape(-1, 3)
        kept_chunks = []
        removed_below = 0
        removed_above = 0
        chunk_size = 1_000_000
        for start in range(0, len(faces_arr), chunk_size):
            faces = faces_arr[start:start + chunk_size]
            face_z = original_z[faces]
            face_lower = lower[faces]
            face_upper = upper[faces]
            below = np.all(face_z < face_lower, axis=1)
            above = np.all(face_z > face_upper, axis=1)
            keep = ~(below | above)
            removed_below += int(np.sum(below))
            removed_above += int(np.sum(above))
            if np.any(keep):
                kept_chunks.append(faces[keep])
        self.input_tri = np.concatenate(kept_chunks, axis=0) if kept_chunks else np.empty((0, 3), dtype=np.int32)
        clipped_low = int(np.sum(original_z < lower))
        clipped_high = int(np.sum(original_z > upper))
        self.input_vert[:, 2] = np.clip(original_z, lower, upper)

        top_z = surface_points[:, 2]
        self.navmesh_preprocess_metadata = {
            **preprocess_key,
            "applied": True,
            "terrain_prim": str(terrain_prim.GetPath()),
            "terrain_top_min": float(np.min(top_z)),
            "terrain_top_max": float(np.max(top_z)),
            "terrain_top_p50": float(np.percentile(top_z, 50)),
            "removed_faces_below_window": removed_below,
            "removed_faces_above_window": removed_above,
            "clipped_vertices_below_window": clipped_low,
            "clipped_vertices_above_window": clipped_high,
        }
        print(
            "[INFO]: Applied terrain-relative height window "
            f"terrain={terrain_prim.GetPath()} top_z=[{np.min(top_z):.3f}, {np.max(top_z):.3f}] "
            f"removed_faces below={removed_below} above={removed_above} "
            f"clipped_vertices below={clipped_low} above={clipped_high}"
        )

    def configure_building_exclusion(self, selected_paths, stage, scene_type=None):
        self.scene_type = scene_type
        if not self._preprocess_enabled(scene_type):
            self.building_exclusion_bboxes = np.empty((0, 6), dtype=np.float32)
            return
        _require_pxr()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
        bboxes = []
        for selected_path in selected_paths:
            root = stage.GetPrimAtPath(selected_path)
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                prim_path = str(prim.GetPath()).lower()
                if "building" not in prim_path and "buildings" not in prim_path:
                    continue
                try:
                    bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                    min_point = bound.GetMin()
                    max_point = bound.GetMax()
                except Exception:
                    continue
                bbox = [
                    float(min_point[0]),
                    float(min_point[1]),
                    float(min_point[2]),
                    float(max_point[0]),
                    float(max_point[1]),
                    float(max_point[2]),
                ]
                if bbox[3] <= bbox[0] or bbox[4] <= bbox[1]:
                    continue
                bboxes.append(bbox)
        self.building_exclusion_bboxes = (
            np.asarray(bboxes, dtype=np.float32)
            if bboxes
            else np.empty((0, 6), dtype=np.float32)
        )
        if len(self.building_exclusion_bboxes):
            print(f"[INFO]: Loaded {len(self.building_exclusion_bboxes)} building exclusion bboxes")

    def _load_building_exclusion_metadata(self, metadata):
        self.scene_type = metadata.get("scene_type")
        bboxes = metadata.get("building_exclusion_bboxes", [])
        if bboxes:
            self.building_exclusion_bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 6)
        else:
            self.building_exclusion_bboxes = np.empty((0, 6), dtype=np.float32)

    def _point_in_excluded_building(self, points):
        bboxes = self.building_exclusion_bboxes
        if bboxes is None or len(bboxes) == 0:
            return np.zeros(len(points), dtype=bool)
        points = np.asarray(points, dtype=np.float32)
        excluded = np.zeros(len(points), dtype=bool)
        chunk_size = 256
        margin = 0.05
        for start in range(0, len(points), chunk_size):
            chunk = points[start:start + chunk_size]
            inside_xy = (
                (chunk[:, None, 0] >= bboxes[None, :, 0] - margin)
                & (chunk[:, None, 0] <= bboxes[None, :, 3] + margin)
                & (chunk[:, None, 1] >= bboxes[None, :, 1] - margin)
                & (chunk[:, None, 1] <= bboxes[None, :, 4] + margin)
            )
            excluded[start:start + chunk_size] = np.any(inside_xy, axis=1)
        return excluded

    def is_path_valid(self):
        if self.path_points is None:
            print("[WARNING]: No path found")
            return False
        if len(self.path_points) < 2:
            return False
        if self.path_points[0] != self.start_pos or self.path_points[-1] != self.end_pos:
            print("[WARNING]: Path does not match start or end position")
            return False
        return True

    def setup_navmesh(self, selected_paths, exclude_paths, stage, scene_type=None, preprocess_config=None):
        if preprocess_config is not None:
            self.preprocess_config = preprocess_config
        self.configure_building_exclusion(selected_paths, stage, scene_type=scene_type)
        self.input_prim = [stage.GetPrimAtPath(x) for x in selected_paths]
        self.input_vert, self.input_tri = get_all_stage_mesh(
            stage,
            self.input_prim,
            exclude_paths=exclude_paths,
            exclude_markers=_non_physical_geometry_markers(scene_type, self.preprocess_config),
        )
        if len(self.input_vert) == 0:
            print('[INFO]: No mesh found')
        self.input_vert = np.array(self.input_vert)
        print(f"[INFO]: Loaded {len(self.input_vert)} vertices and {len(self.input_tri)} triangles")
        print(f'[INFO]: bounding box: max={self.input_vert.max(axis=0)}, min={self.input_vert.min(axis=0)}')
        if np.any(np.isnan(self.input_vert)):
            print("[WARNING]: NaNs found in input vertices")
        print(f"[INFO]: scene_type: {scene_type}")
        self._apply_relative_height_window(self.input_prim, scene_type)
        self.input_vert = self._convert_up_axis(self.input_vert)
        if self._preprocess_enabled(scene_type):
            self._filter_degenerate_faces()
            self._set_local_navmesh_origin()
        
        # Optimize face flattening
        verts_flat = self.input_vert.flatten().tolist()
        faces_arr = np.array(self.input_tri, dtype=np.int32)
        if faces_arr.ndim == 1:
             faces_arr = faces_arr.reshape(-1, 3)
             
        threes = np.full((faces_arr.shape[0], 1), 3, dtype=np.int32)
        faces_flat = np.hstack((threes, faces_arr)).flatten().tolist()

        # Initialize the navmesh with raw data
        print("[INFO]: Loading geometry from vertices and triangles. This will take a while, please wait.")
        self.nm.init_by_raw(verts_flat, faces_flat)
        print("[INFO]: Geometry loaded")

    def save_geometry(self, selected_paths, exclude_paths, stage, output_dir, scene_type=None, preprocess_config=None):
        if preprocess_config is not None:
            self.preprocess_config = preprocess_config
        self.configure_building_exclusion(selected_paths, stage, scene_type=scene_type)
        self.input_prim = [stage.GetPrimAtPath(x) for x in selected_paths]
        self.input_vert, self.input_tri = get_all_stage_mesh(
            stage,
            self.input_prim,
            exclude_paths=exclude_paths,
            exclude_markers=_non_physical_geometry_markers(scene_type, self.preprocess_config),
        )
        if len(self.input_vert) == 0:
            print('[INFO]: No mesh found')
        self.input_vert = np.array(self.input_vert)
        print(f"[INFO]: Loaded {len(self.input_vert)} vertices and {len(self.input_tri)} triangles")
        print(f'[INFO]: bounding box: max={self.input_vert.max(axis=0)}, min={self.input_vert.min(axis=0)}')
        if np.any(np.isnan(self.input_vert)):
            print("[WARNING]: NaNs found in input vertices")
        print(f"[INFO]: scene_type: {scene_type}")
        self._apply_relative_height_window(self.input_prim, scene_type)
        self.input_vert = self._convert_up_axis(self.input_vert)
        if self._preprocess_enabled(scene_type):
            self._filter_degenerate_faces()
        
        # Convert back to original axis (Z-up) for saving to be consistent with generate_navmesh.py
        self.input_vert = self._convert_up_axis(self.input_vert, inverse=True)

        # Save to bin files
        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        verts_path = os.path.join(output_dir, f"{scene_type}_vertices.bin")
        faces_path = os.path.join(output_dir, f"{scene_type}_faces.bin")
        
        print(f"[INFO]: Saving geometry to {verts_path}")
        self.input_vert.astype(np.float32).tofile(verts_path)
        
        if len(self.input_tri) > 0:
            # Flatten faces
            faces_arr = np.array(self.input_tri, dtype=np.int32)
            faces_arr.tofile(faces_path)
        else:
            with open(faces_path, 'wb'):
                pass
                
        print("[INFO]: Geometry saved")


    def build_navmesh(self):
        print(f"[INFO]: settings: {self.settings}")
        self.nm.set_settings(self.settings)
        # Try watershed (0); if it fails, switch to monotone (1)
        self.nm.set_partition_type(1)
        print("[INFO]: Building navmesh, will take a while, please wait.")
        self.nm.build_navmesh()
        v, t, = self.get_navmesh_polygons()
        print(f'v shape: {v.shape}')
        print(f't shape: {t.shape}')
        log = self.nm.get_log()
        print(log)
        if not self.built:
            print('[WARNING]: Failed to build navmesh')

    @staticmethod
    def _prim_intersects_xy_radius(prim, bbox_cache, center_xy, radius):
        try:
            bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = np.asarray(bounds.GetMin(), dtype=float)
            maximum = np.asarray(bounds.GetMax(), dtype=float)
        except Exception:
            return False
        delta = np.maximum(np.maximum(minimum[:2] - center_xy, 0.0), center_xy - maximum[:2])
        return float(delta @ delta) <= float(radius) ** 2

    @staticmethod
    def _innout_child(stage, roots, name):
        for root in roots:
            candidate = stage.GetPrimAtPath(f"{root.GetPath()}/{name}")
            if candidate.IsValid():
                return candidate
        return None

    def _nearby_children(self, parent, bbox_cache, center_xy, radius):
        if parent is None:
            return []
        return [
            candidate
            for candidate in parent.GetChildren()
            if self._prim_intersects_xy_radius(candidate, bbox_cache, center_xy, radius)
        ]

    def _innout_input_prims(self, selected_paths, stage, room_center, max_dist):
        roots = [stage.GetPrimAtPath(path) for path in selected_paths]
        roots = [prim for prim in roots if prim.IsValid()]
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )

        prims = []
        collision_surface = self._innout_child(stage, roots, "NavVerseTerrainCollision")
        terrain_regions = self._innout_child(stage, roots, "TerrainRegions")
        if collision_surface is not None:
            prims.append(collision_surface)
        elif terrain_regions is not None:
            prims.append(terrain_regions)

        for name in ("start_result_navigation", "InnOutDoorConnector", "InnOutThreshold"):
            candidate = self._innout_child(stage, roots, name)
            if candidate is not None:
                prims.append(candidate)

        nearby_radius = float(max_dist) + 5.0
        for name in ("city", "store"):
            parent = self._innout_child(stage, roots, name)
            prims.extend(self._nearby_children(parent, bbox_cache, room_center[:2], nearby_radius))

        bbox_prims = []
        for name in ("Objaverse", "RoadObjects"):
            parent = self._innout_child(stage, roots, name)
            bbox_prims.extend(self._nearby_children(parent, bbox_cache, room_center[:2], nearby_radius))

        unique = {}
        for prim in prims:
            unique[str(prim.GetPath())] = prim
        print(
            "[INFO]: InNOut navmesh input roots: "
            + ", ".join(sorted(unique))
        )
        bbox_unique = {str(prim.GetPath()): prim for prim in bbox_prims}
        print(f"[INFO]: InNOut bounding-box obstacle count: {len(bbox_unique)}")
        return list(unique.values()), list(bbox_unique.values())

    @staticmethod
    def _bbox_obstacle_geometry(
        prims,
        xy_padding=0.0,
        min_height=0.0,
        ground_overlap=0.0,
    ):
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        vertices = []
        faces = []
        box_faces = np.asarray(
            [
                [0, 2, 1], [1, 2, 3],
                [4, 5, 6], [5, 7, 6],
                [0, 1, 4], [1, 5, 4],
                [2, 6, 3], [3, 6, 7],
                [0, 4, 2], [2, 4, 6],
                [1, 3, 5], [3, 7, 5],
            ],
            dtype=np.int32,
        )
        for prim in prims:
            try:
                bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                minimum = np.asarray(bounds.GetMin(), dtype=float)
                maximum = np.asarray(bounds.GetMax(), dtype=float)
            except Exception:
                continue
            if (
                not np.all(np.isfinite([*minimum, *maximum]))
                or np.any(maximum[:2] <= minimum[:2])
                or maximum[2] < minimum[2]
            ):
                continue
            source_min_z = float(minimum[2])
            minimum[:2] -= float(xy_padding)
            maximum[:2] += float(xy_padding)
            minimum[2] = source_min_z - float(ground_overlap)
            maximum[2] = max(float(maximum[2]), source_min_z + float(min_height))
            offset = len(vertices)
            vertices.extend([
                [minimum[0], minimum[1], minimum[2]],
                [maximum[0], minimum[1], minimum[2]],
                [minimum[0], maximum[1], minimum[2]],
                [maximum[0], maximum[1], minimum[2]],
                [minimum[0], minimum[1], maximum[2]],
                [maximum[0], minimum[1], maximum[2]],
                [minimum[0], maximum[1], maximum[2]],
                [maximum[0], maximum[1], maximum[2]],
            ])
            faces.extend(box_faces + offset)
        return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    def _crop_geometry_to_xy_radius(self, center_xy, max_dist):
        faces = np.asarray(self.input_tri, dtype=np.int32).reshape(-1, 3)
        distances_sq = np.sum((self.input_vert[:, :2] - center_xy) ** 2, axis=1)
        near_vertex = distances_sq <= float(max_dist) ** 2
        kept_faces = faces[np.any(near_vertex[faces], axis=1)]
        if len(kept_faces) == 0:
            self.input_vert = np.empty((0, 3), dtype=np.float64)
            self.input_tri = np.empty((0, 3), dtype=np.int32)
            return
        used_vertices, remapped = np.unique(kept_faces.reshape(-1), return_inverse=True)
        removed_vertices = len(self.input_vert) - len(used_vertices)
        self.input_vert = self.input_vert[used_vertices]
        self.input_tri = remapped.reshape(-1, 3).astype(np.int32)
        print(
            f"[INFO]: Cropped InNOut geometry to {max_dist:.1f}m: "
            f"removed_vertices={removed_vertices} remaining_vertices={len(self.input_vert)} "
            f"remaining_faces={len(self.input_tri)}"
        )

    def build_navmesh_innout(
        self,
        selected_paths,
        exclude_paths,
        stage,
        max_dist=60.0,
        preprocess_config=None,
    ):
        if preprocess_config is not None:
            self.preprocess_config = preprocess_config
        self.scene_type = "innout"
        room_prim = None
        for root_path in selected_paths:
            candidate = stage.GetPrimAtPath(f"{str(root_path).rstrip('/')}/start_result_navigation")
            if candidate.IsValid():
                room_prim = candidate
                break
        if room_prim is None:
            raise RuntimeError("InNOut room prim is missing from the selected navmesh roots")
        room_center = get_bbox_center(room_prim)

        self.configure_building_exclusion(selected_paths, stage, scene_type="innout")
        self.input_prim, bbox_obstacle_prims = self._innout_input_prims(
            selected_paths,
            stage,
            room_center,
            max_dist,
        )
        if not self.input_prim:
            raise RuntimeError("No InNOut navmesh input prims found")
        self.input_vert, self.input_tri = get_all_stage_mesh(
            stage,
            self.input_prim,
            exclude_paths=exclude_paths,
            exclude_markers=_non_physical_geometry_markers("innout", self.preprocess_config),
            include_invisible_roots=True,
        )
        self.input_vert = np.asarray(self.input_vert, dtype=np.float64)
        self.input_tri = np.asarray(self.input_tri, dtype=np.int32).reshape(-1, 3)
        bbox_padding = float(
            _config_value(
                self.preprocess_config,
                "innout_bbox_obstacle_xy_padding_m",
                default=0.0,
            )
            or 0.0
        )
        bbox_min_height = float(
            _config_value(
                self.preprocess_config,
                "innout_bbox_obstacle_min_height_m",
                default=0.0,
            )
            or 0.0
        )
        bbox_ground_overlap = float(
            _config_value(
                self.preprocess_config,
                "innout_bbox_obstacle_ground_overlap_m",
                default=0.0,
            )
            or 0.0
        )
        if min(bbox_padding, bbox_min_height, bbox_ground_overlap) < 0.0:
            raise ValueError("InNOut bounding-box obstacle margins must be non-negative")
        bbox_vertices, bbox_faces = self._bbox_obstacle_geometry(
            bbox_obstacle_prims,
            xy_padding=bbox_padding,
            min_height=bbox_min_height,
            ground_overlap=bbox_ground_overlap,
        )
        if len(bbox_vertices):
            bbox_faces = bbox_faces + len(self.input_vert)
            self.input_vert = np.concatenate((self.input_vert, bbox_vertices), axis=0)
            self.input_tri = np.concatenate((self.input_tri, bbox_faces), axis=0)
        if len(self.input_vert) == 0 or len(self.input_tri) == 0:
            raise RuntimeError("InNOut navmesh geometry is empty")
        if np.any(~np.isfinite(self.input_vert)):
            raise RuntimeError("InNOut navmesh geometry contains non-finite vertices")

        self._crop_geometry_to_xy_radius(room_center[:2], max_dist)
        if len(self.input_vert) == 0 or len(self.input_tri) == 0:
            raise RuntimeError("InNOut navmesh geometry is empty after local cropping")
        self._apply_relative_height_window(self.input_prim, "innout")
        preprocess_metadata = dict(self.navmesh_preprocess_metadata or {})
        preprocess_metadata.update({
            "innout_build_version": INNOUT_NAVMESH_BUILD_VERSION,
            "innout_local_radius_m": float(max_dist),
            "innout_input_root_count": len(self.input_prim),
            "innout_bbox_obstacle_count": len(bbox_obstacle_prims),
            "innout_bbox_obstacle_xy_padding_m": bbox_padding,
            "innout_bbox_obstacle_min_height_m": bbox_min_height,
            "innout_bbox_obstacle_ground_overlap_m": bbox_ground_overlap,
            "innout_input_vertex_count": len(self.input_vert),
            "innout_input_face_count": len(self.input_tri),
        })
        self.navmesh_preprocess_metadata = preprocess_metadata

        self.input_vert = self._convert_up_axis(self.input_vert)
        self._filter_degenerate_faces()
        self._set_local_navmesh_origin()
        faces = np.asarray(self.input_tri, dtype=np.int32).reshape(-1, 3)
        face_data = np.column_stack((np.full(len(faces), 3, dtype=np.int32), faces)).reshape(-1)
        print("[INFO]: Loading local InNOut geometry into Recast")
        self.nm.init_by_raw(self.input_vert.reshape(-1).tolist(), face_data.tolist())
        self.build_navmesh()


    def visualize_navmesh(self):
        _require_pxr()
        from navverse.utils.vis import visualize_mesh
        if self.built:
            v, t, = self.navmesh_v, self.navmesh_t
            v = v.flatten()
            # create a usd color of blue with transparency
            color = Gf.Vec3f(0.051208995, 0.774935, 0.94585985)
            opacity = 0.89
            visualize_mesh('/World/ground/navmeshmesh', v, t, color, opacity)
            print("[INFO]: Visualized navmesh")
        else:
            print('[WARNING]: Navmesh not built') 

    def get_navmesh_polygons(self):
        trivert, polygon_indices, polygon_sizes = self.nm.get_navmesh_polygonization()
        trivert = np.asarray(trivert).reshape(-1,3)

        # Parse polygon_indices using polygon_sizes
        t = []
        index_offset = 0
        
        for poly_size in polygon_sizes:
            if poly_size == 3:
                # Triangle - add directly
                triangle = [polygon_indices[index_offset], 
                        polygon_indices[index_offset + 1], 
                        polygon_indices[index_offset + 2]]
                t.append(triangle)
            elif poly_size > 3:
                # Polygon with more than 3 vertices - use fan triangulation
                first_vertex = polygon_indices[index_offset]
                for i in range(1, poly_size - 1):
                    triangle = [first_vertex,
                            polygon_indices[index_offset + i],
                            polygon_indices[index_offset + i + 1]]
                    t.append(triangle)
            # Skip polygons with less than 3 vertices (invalid)
            index_offset += poly_size
        
        self.navmesh_v = np.array(trivert, dtype=np.float32)
        self.navmesh_t = np.array(t, dtype=np.int32)
        # print(f'navmesh_v shape: {self.navmesh_v.shape}')
        # print(f'navmesh_t shape: {self.navmesh_t.shape}')
        # print(f'navmesh_v: {self.navmesh_v[0]}')
        # print(f'navmesh_t: {self.navmesh_t[0]}')
        self.navmesh_v = self._convert_up_axis(self.navmesh_v + self.navmesh_origin, inverse=True)
        
        self.built = self.navmesh_v.shape[0] > 0
        return self.navmesh_v, self.navmesh_t
    
    def find_paths(self, start, end):
        """
        Input: start, end are lists of points [x, y, z]
        """
        start = self._convert_up_axis([start])[0] - self.navmesh_origin
        end = self._convert_up_axis([end])[0] - self.navmesh_origin

        path_points = self.nm.pathfind_straight(start, end, 1)
        path_points = np.array(path_points).reshape(-1, 3)
        path_points = path_points + self.navmesh_origin
        path_points = self._convert_up_axis(path_points, inverse=True)
        # print(f"Path points: {path_points.shape}")
        
        # if path_points.shape[0] <= 1:
        #     print("[WARNING]: No valid path found")
        return path_points

    def distance_to_wall(self, point):
        point = np.asarray(point, dtype=np.float32).reshape(1, 3)
        point = self._convert_up_axis(point)[0] - self.navmesh_origin
        distance = self.nm.distance_to_wall(point.tolist())
        return extract_distance_to_wall(distance)

    def save_navmesh(self, save_path, metadata=None):
        print("[INFO]: Exported navmesh to", save_path)
        self.nm.save_navmesh(save_path)
        navmesh_metadata = {
            "origin": self.navmesh_origin.tolist(),
            "settings": self.serialized_settings(),
            "preprocess_details": self.navmesh_preprocess_metadata,
        }
        if len(self.building_exclusion_bboxes):
            navmesh_metadata["building_exclusion_bboxes"] = self.building_exclusion_bboxes.tolist()
        if metadata:
            navmesh_metadata.update(metadata)
        with open(f"{save_path}.meta.json", "w") as f:
            json.dump(navmesh_metadata, f)

    def load_navmesh(self, load_path):
        print("[INFO]: Loaded navmesh from", load_path)
        self.navmesh_origin = np.zeros(3, dtype=np.float32)
        meta_path = f"{load_path}.meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
            self.navmesh_origin = np.array(metadata.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32)
            self._load_building_exclusion_metadata(metadata)
            print(f"[INFO]: Loaded navmesh origin {self.navmesh_origin.tolist()}")
        else:
            self.scene_type = None
            self.building_exclusion_bboxes = np.empty((0, 6), dtype=np.float32)
        self.nm.load_navmesh(load_path)
        self.built = True
        print(self.nm.get_log())
        print("[INFO]: Updating navmesh polygons")
        self.get_navmesh_polygons()

    """ ----- Navmesh helper functions ----- """
    # ----- Helper functions for up axis conversion ----- #
    def _convert_up_axis(self, vertices, inverse=False):
        '''
        Convert all data between navmesh interface and end-user to be in the correct up axis

        pyrecast assumes y-up, so only change when z-up is true
        inverse = True will convert y-up output back to z-up
        '''
        if not self.z_up:
            return vertices
        
        vertices = np.array(vertices)
        v_copy = np.empty(shape=vertices.shape)
        if inverse:
            # Convert data we have been given in y-up into the z-up of scene (e.g. output from recast)
            v_copy[:, 0], v_copy[:, 1], v_copy[:, 2] = vertices[:, 0], -vertices[:, 2], vertices[:, 1]
        else: 
            # Convert the up axis from Y to Z
            v_copy[:, 0], v_copy[:, 1], v_copy[:, 2] = vertices[:, 0], vertices[:, 2], -vertices[:, 1]

        vertices = v_copy
        return vertices

    # ----- Helper functions for getting random points ----- #
    def sample_random_points(self, num_points):
        if not self.built or self.navmesh_v.shape[0] == 0:
            print("[WARNING]: Navmesh not built")
            return None

        def sample_raw(count):
            random_poly_indices = np.random.randint(0, len(t), count)
            random_poly = t[random_poly_indices]
            weights = np.random.rand(random_poly.shape[0]*3).reshape(-1, 3)
            weights = weights[:,:,np.newaxis].repeat(3, axis=2)
            return np.average(v[random_poly], weights=weights, axis=1)

        v = self.navmesh_v
        t = self.navmesh_t
        if len(self.building_exclusion_bboxes) == 0:
            return sample_raw(num_points)

        accepted = []
        remaining = int(num_points)
        max_attempts = 12
        for _ in range(max_attempts):
            batch_count = max(remaining * 4, num_points)
            candidates = sample_raw(batch_count)
            keep = ~self._point_in_excluded_building(candidates)
            if np.any(keep):
                kept = candidates[keep]
                accepted.append(kept[:remaining])
                remaining -= min(remaining, len(kept))
                if remaining <= 0:
                    break
        if remaining > 0:
            print(f"[WARNING]: Only sampled {num_points - remaining}/{num_points} non-building navmesh points")
            if not accepted:
                return None
        vertices = np.concatenate(accepted, axis=0)[:num_points]
        # print(f'random points: {vertices}')

        return vertices
