"""Deterministic terrain geometry for controlled embodiment evaluation."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


TERRAIN_TYPES = ("original", "flat", "slope", "stairs", "uneven")
TERRAIN_ROOT_PATH = "/World/NavVerseControlledTerrain"
PATCH_LENGTH_M = 4.0
PATCH_WIDTH_M = 2.0
PATCH_CENTER_FRACTIONS = (0.25, 0.50, 0.75)
SLOPE_ANGLE_DEG = 8.0
STAIR_HEIGHT_M = 0.10
TERRAIN_PROFILES = {
    "standard": {
        "patch_length_m": PATCH_LENGTH_M,
        "patch_width_m": PATCH_WIDTH_M,
        "slope_angle_deg": SLOPE_ANGLE_DEG,
        "stair_height_m": STAIR_HEIGHT_M,
        "uneven_height_range_m": (0.02, 0.06),
    },
    "hard": {
        "patch_length_m": PATCH_LENGTH_M,
        "patch_width_m": PATCH_WIDTH_M,
        "slope_angle_deg": 15.0,
        "stair_height_m": 0.15,
        "uneven_height_range_m": (0.05, 0.12),
    },
    "moderate": {
        "patch_length_m": 4.5,
        "patch_width_m": 2.25,
        "slope_angle_deg": 15.0,
        "stair_height_m": 0.12,
        "uneven_height_range_m": (0.04, 0.10),
    },
    "moderate_v2": {
        "patch_length_m": 4.5,
        "patch_width_m": 2.25,
        "slope_angle_deg": 15.0,
        "stair_height_m": 0.11,
        "uneven_height_range_m": (0.03, 0.08),
    },
    "moderate_final": {
        "patch_length_m": 4.5,
        "patch_width_m": 2.25,
        "slope_angle_deg": 15.0,
        "stair_height_m": 0.10,
        "uneven_height_range_m": (0.03, 0.08),
    },
    "moderate_compact": {
        "patch_length_m": 4.25,
        "patch_width_m": 2.25,
        "slope_angle_deg": 15.0,
        "stair_height_m": 0.10,
        "uneven_height_range_m": (0.03, 0.08),
    },
    "moderate_wide": {
        "patch_length_m": 4.0,
        "patch_width_m": 2.5,
        "slope_angle_deg": 15.0,
        "stair_height_m": 0.10,
        "uneven_height_range_m": (0.03, 0.08),
    },
    "moderate_traversable": {
        "patch_length_m": 4.0,
        "patch_width_m": 2.5,
        "slope_angle_deg": 15.0,
        "stair_height_m": 0.02,
        "uneven_height_range_m": (0.03, 0.08),
    },
    "balanced": {
        "patch_length_m": 4.5,
        "patch_width_m": 2.25,
        "slope_angle_deg": 17.0,
        "stair_height_m": 0.17,
        "uneven_height_range_m": (0.07, 0.16),
    },
    "extra_hard": {
        "patch_length_m": 5.0,
        "patch_width_m": 2.5,
        "slope_angle_deg": 20.0,
        "stair_height_m": 0.20,
        "uneven_height_range_m": (0.10, 0.22),
    },
    "micro_rough": {
        "patch_length_m": 4.0,
        "patch_width_m": 2.5,
        "slope_angle_deg": 0.0,
        "stair_height_m": 0.0,
        "uneven_height_range_m": (0.002, 0.006),
        "resolution_m": 0.10,
        "z_offset_m": 0.0002,
    },
    "micro_rough_fine": {
        "patch_length_m": 4.0,
        "patch_width_m": 2.5,
        "slope_angle_deg": 0.0,
        "stair_height_m": 0.0,
        "uneven_height_range_m": (0.0005, 0.0015),
        "resolution_m": 0.10,
        "z_offset_m": 0.0002,
    },
    "micro_rough_medium": {
        "patch_length_m": 4.0,
        "patch_width_m": 2.5,
        "slope_angle_deg": 0.0,
        "stair_height_m": 0.0,
        "uneven_height_range_m": (0.001, 0.003),
        "resolution_m": 0.10,
        "z_offset_m": 0.0002,
    },
}


def path_length_xy(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def path_sha256(points):
    payload = json.dumps(points, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _path_arrays(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ValueError("A reference path must contain at least two XY points")
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return points, segment_lengths, cumulative


def sample_path_pose(points, distance_m):
    points, segment_lengths, cumulative = _path_arrays(points)
    distance_m = float(np.clip(distance_m, 0.0, cumulative[-1]))
    segment_idx = int(np.searchsorted(cumulative, distance_m, side="right") - 1)
    segment_idx = min(segment_idx, len(segment_lengths) - 1)
    while segment_idx < len(segment_lengths) - 1 and segment_lengths[segment_idx] <= 1e-9:
        segment_idx += 1
    segment_length = max(float(segment_lengths[segment_idx]), 1e-9)
    ratio = (distance_m - cumulative[segment_idx]) / segment_length
    point = points[segment_idx] + ratio * (points[segment_idx + 1] - points[segment_idx])
    delta = points[segment_idx + 1, :2] - points[segment_idx, :2]
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    z = float(point[2]) if point.shape[0] > 2 else 0.0
    return [float(point[0]), float(point[1]), z], yaw


def _angle_delta(a, b):
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def shared_patch_placements(points, max_turn_deg=25.0, max_relief_m=0.30):
    length = path_length_xy(points)
    if length < 20.0:
        return None
    placements = []
    half_length = PATCH_LENGTH_M * 0.5
    for fraction in PATCH_CENTER_FRACTIONS:
        center_s = length * fraction
        center, yaw = sample_path_pose(points, center_s)
        samples = [sample_path_pose(points, center_s + offset) for offset in np.linspace(-half_length, half_length, 9)]
        sample_yaws = [pose[1] for pose in samples]
        if max(math.degrees(_angle_delta(sample_yaw, yaw)) for sample_yaw in sample_yaws) > max_turn_deg:
            return None
        sample_z = [pose[0][2] for pose in samples]
        if max(sample_z) - min(sample_z) > max_relief_m:
            return None
        placements.append(
            {
                "center_s_m": center_s,
                "center_fraction": fraction,
                "position": center,
                "yaw_rad": yaw,
                "length_m": PATCH_LENGTH_M,
                "width_m": PATCH_WIDTH_M,
            }
        )
    return placements


def episode_terrain_seed(seed, episode_label):
    digest = hashlib.sha256(f"{seed}:{episode_label}".encode()).hexdigest()
    return int(digest[:8], 16)


def load_controlled_terrain_manifest(path):
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest.get("episodes", [])
    if len(episodes) != 100:
        raise ValueError(f"Controlled terrain manifest must contain 100 episodes, got {len(episodes)}")
    labels = [row["episode_label"] for row in episodes]
    if len(set(labels)) != len(labels):
        raise ValueError("Controlled terrain manifest contains duplicate episode labels")
    scenes = {row["scene_id"] for row in episodes}
    if len(scenes) != 10:
        raise ValueError(f"Controlled terrain manifest must contain 10 scenes, got {len(scenes)}")
    for scene_id in scenes:
        count = sum(row["scene_id"] == scene_id for row in episodes)
        if count != 10:
            raise ValueError(f"Scene {scene_id} has {count} episodes instead of 10")
    for row in episodes:
        if len(row.get("patches", [])) != 3:
            raise ValueError(f"Episode {row['episode_label']} does not have exactly three patches")
    manifest["_path"] = str(manifest_path)
    manifest["_episode_map"] = {row["episode_label"]: row for row in episodes}
    return manifest


def terrain_profile_config(profile):
    try:
        return TERRAIN_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported controlled terrain profile: {profile!r}") from exc


def _height_grid(terrain_type, seed, resolution=0.25, profile="standard"):
    config = terrain_profile_config(profile)
    resolution = float(config.get("resolution_m", resolution))
    patch_length = config["patch_length_m"]
    patch_width = config["patch_width_m"]
    x = np.linspace(-patch_length * 0.5, patch_length * 0.5, int(patch_length / resolution) + 1)
    y = np.linspace(-patch_width * 0.5, patch_width * 0.5, int(patch_width / resolution) + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if terrain_type == "flat":
        heights = np.zeros_like(xx)
    elif terrain_type == "slope":
        heights = np.tan(np.deg2rad(config["slope_angle_deg"])) * (
            patch_length * 0.5 - np.abs(xx)
        )
    elif terrain_type == "uneven":
        rng = np.random.default_rng(seed)
        heights = rng.uniform(*config["uneven_height_range_m"], size=xx.shape)
        edge_distance = np.minimum(patch_length * 0.5 - np.abs(xx), patch_width * 0.5 - np.abs(yy))
        heights *= np.clip(edge_distance / 0.50, 0.0, 1.0)
    else:
        raise ValueError(f"Height grid is not used for terrain type {terrain_type!r}")
    return x, y, heights


def _grid_mesh_data(terrain_type, seed, profile="standard"):
    x, y, heights = _height_grid(terrain_type, seed, profile=profile)
    points = [[float(xv), float(yv), float(heights[i, j])] for i, xv in enumerate(x) for j, yv in enumerate(y)]
    face_counts = []
    face_indices = []
    ny = len(y)
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v11 = (i + 1) * ny + j + 1
            v01 = i * ny + j + 1
            face_counts.extend((3, 3))
            face_indices.extend((v00, v10, v11, v00, v11, v01))
    return points, face_counts, face_indices


class ControlledTerrainManager:
    def __init__(self, manifest_path, terrain_type, terrain_profile="standard"):
        if terrain_type not in TERRAIN_TYPES:
            raise ValueError(f"Unsupported controlled terrain type: {terrain_type!r}")
        self.manifest = load_controlled_terrain_manifest(manifest_path)
        self.terrain_type = terrain_type
        self.terrain_profile = terrain_profile
        self.terrain_config = terrain_profile_config(terrain_profile)
        self.current = None
        self.reached = [False, False, False]
        self.completed = [False, False, False]

    @staticmethod
    def _apply_collision(prim):
        from pxr import UsdPhysics

        UsdPhysics.CollisionAPI.Apply(prim)

    def _create_surface_patch(self, stage, patch_path, seed):
        from pxr import Gf, UsdGeom

        points, face_counts, face_indices = _grid_mesh_data(
            self.terrain_type,
            seed,
            profile=self.terrain_profile,
        )
        mesh = UsdGeom.Mesh.Define(stage, f"{patch_path}/Surface")
        mesh.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
        if not mesh.GetFaceVertexCountsAttr().HasAuthoredValue():
            mesh.CreateFaceVertexCountsAttr(face_counts)
            mesh.CreateFaceVertexIndicesAttr(face_indices)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.32, 0.18)])
            self._apply_collision(mesh.GetPrim())

    def _create_stair_patch(self, stage, patch_path):
        from pxr import Gf, UsdGeom

        cube = UsdGeom.Cube.Define(stage, f"{patch_path}/SingleStep")
        if not cube.GetSizeAttr().HasAuthoredValue():
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            stair_height = self.terrain_config["stair_height_m"]
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, stair_height * 0.5))
            xform.AddScaleOp().Set(
                Gf.Vec3d(
                    self.terrain_config["patch_length_m"],
                    self.terrain_config["patch_width_m"],
                    stair_height,
                )
            )
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.40, 0.28, 0.16)])
            self._apply_collision(cube.GetPrim())

    def _ensure_patch(self, stage, patch_path, seed):
        from pxr import UsdGeom

        patch = UsdGeom.Xform.Define(stage, patch_path)
        prim = patch.GetPrim()
        xform = UsdGeom.Xformable(prim)
        if not prim.GetAttribute("xformOp:translate").IsValid():
            xform.AddTranslateOp()
        if not prim.GetAttribute("xformOp:rotateZ").IsValid():
            xform.AddRotateZOp()
        if self.terrain_type == "stairs":
            self._create_stair_patch(stage, patch_path)
        else:
            self._create_surface_patch(stage, patch_path, seed)
        return prim

    def configure_episode(self, stage, episode_label):
        if episode_label not in self.manifest["_episode_map"]:
            raise KeyError(f"Episode {episode_label!r} is not in the shared controlled terrain manifest")
        self.current = self.manifest["_episode_map"][episode_label]
        self.reached = [False, False, False]
        self.completed = [False, False, False]
        if self.terrain_type == "original":
            return self.metric_values()

        from pxr import Gf, UsdGeom

        root = UsdGeom.Xform.Define(stage, TERRAIN_ROOT_PATH)
        root.GetPrim().SetCustomDataByKey("navverse_manifest", self.manifest["_path"])
        root.GetPrim().SetCustomDataByKey("navverse_terrain_type", self.terrain_type)
        root.GetPrim().SetCustomDataByKey("navverse_terrain_profile", self.terrain_profile)
        for index, placement in enumerate(self.current["patches"]):
            patch_path = f"{TERRAIN_ROOT_PATH}/Patch_{index}"
            patch_prim = self._ensure_patch(
                stage, patch_path, self.current["terrain_seed"]
            )
            position = placement["position"]
            patch_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]) + float(self.terrain_config.get("z_offset_m", 0.005)),
                )
            )
            patch_prim.GetAttribute("xformOp:rotateZ").Set(
                math.degrees(float(placement["yaw_rad"]))
            )
        return self.metric_values()

    def update_progress(self, position_xy):
        if self.current is None:
            return self.metric_values()
        x, y = float(position_xy[0]), float(position_xy[1])
        for index, placement in enumerate(self.current["patches"]):
            px, py = placement["position"][:2]
            yaw = float(placement["yaw_rad"])
            dx, dy = x - px, y - py
            local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
            local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
            half_length = float(self.terrain_config["patch_length_m"]) * 0.5
            half_width = float(self.terrain_config["patch_width_m"]) * 0.5
            if abs(local_x) <= half_length and abs(local_y) <= half_width + 0.35:
                self.reached[index] = True
            if self.reached[index] and local_x >= half_length + 0.20:
                self.completed[index] = True
        return self.metric_values()

    def metric_values(self):
        metrics = {
            "terrain_patches_reached": float(sum(self.reached)),
            "terrain_patches_completed": float(sum(self.completed)),
        }
        for index in range(3):
            metrics[f"terrain_patch_{index + 1}_reached"] = float(self.reached[index])
            metrics[f"terrain_patch_{index + 1}_completed"] = float(self.completed[index])
        return metrics
