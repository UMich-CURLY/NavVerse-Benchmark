from pathlib import Path
import re
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporter
from navverse.measures import add_measurement
import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import math as math_utils
import isaacsim.core.utils.bounds as bounds_utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
import torch
from navverse.termination_cfg import VLNTerminationsCfg
from navverse.navmesh import (
    NavmeshInterface,
    extract_distance_to_wall,
    load_navmesh_for_episode,
    navmesh_segment_is_clear,
    navmesh_surface_is_reachable,
    navmesh_surface_z,
)
from navverse.config import load_config, resolve_task_config_path
from navverse_tools.terrain_eval.runtime import (
    TerrainEvaluationRuntime,
    configure_env_cfg as configure_terrain_eval_env_cfg,
)
from isaaclab.managers import TerminationManager
import isaacsim.core.utils.prims as prim_utils
import numpy as np
from scipy.spatial.transform import Rotation as R
from isaaclab.sensors import TiledCamera, TiledCameraCfg
import time
import os
import carb

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

INNOUT_COLLISION_ROLE_KEY = "navverse_collision_role"
INNOUT_VISUAL_ONLY_COLLISION_ROLE = "visual_only"
NAVMESH_BOUNDARY_COLLISION_PATH = "/World/NavVerseNavmeshBoundaryCollision"
MISSING_COLLISION_SUBTREES = object()
DEFAULT_PHYSICAL_COLLISION_SUBTREES = {
    "vcp": ("city", "RoadObjects", "store", "Objaverse"),
}


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


def init_env_cfg(env_cfg, args, episode):
    load_scene(env_cfg, args, episode)
    configure_terrain_eval_env_cfg(env_cfg, args)
    set_robot_pose(env_cfg, episode)
    
def load_scene(env_cfg, args, episode):
    scene_path = episode["path"]
    print(f"[INFO] Loading scene {scene_path}...")
    if scene_path == "generator":
        env_cfg.load_generator()
    else:
        usd_path = str(Path(args.scene_folder) / scene_path)
        print(f"[DEBUG] Loading USD from {usd_path}")
        env_cfg.load_usd(usd_path)


def disable_embedded_scene_lighting(stage, scene_root_path="/World/ground/terrain"):
    scene_root = stage.GetPrimAtPath(scene_root_path)
    if not scene_root.IsValid():
        return []

    sky_paths = []
    for prim in Usd.PrimRange(scene_root):
        path = prim.GetPath()
        if prim.GetName().lower() in {"sky", "skylight"}:
            sky_paths.append(path)

    disabled_paths = []
    for path in sorted(sky_paths, key=lambda item: item.pathElementCount):
        if any(path.HasPrefix(parent) for parent in sky_paths if parent != path):
            continue
        stage.GetPrimAtPath(path).SetActive(False)
        disabled_paths.append(str(path))

    return disabled_paths


def configure_runtime_lighting(stage, scene_folder):
    default_dome_light = stage.GetPrimAtPath("/World/skyLight")
    if default_dome_light.IsValid():
        default_dome_light.SetActive(False)

    for path in ("/World/sky", "/World/NavVerseLighting"):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)

    cloudy_sky_path = Path(scene_folder) / "nvidia" / "sky" / "CloudySky.usd"
    if not cloudy_sky_path.is_file():
        raise FileNotFoundError(f"CloudySky asset not found: {cloudy_sky_path}")

    sky = stage.DefinePrim("/World/sky", "Xform")
    sky.GetReferences().AddReference(str(cloudy_sky_path.resolve()), Sdf.Path("/World"))
    sky.SetCustomDataByKey("navverse_runtime_lighting", "CloudySky")
    carb.settings.get_settings().set("/rtx/sceneDb/ambientLightIntensity", 1.0)
    return str(sky.GetPath())


def configure_legacy_runtime_lighting(stage, scene_folder):
    for prim in stage.Traverse():
        if (
            str(prim.GetPath()).startswith("/World/ground/terrain")
            and prim.GetTypeName() == "DomeLight"
        ):
            prim.SetActive(False)
    default_dome_light = stage.GetPrimAtPath("/World/skyLight")
    if default_dome_light.IsValid():
        default_dome_light.SetActive(False)
    if stage.GetPrimAtPath("/World/sky").IsValid():
        stage.RemovePrim("/World/sky")
    sky_path = Path(scene_folder) / "nvidia" / "sky" / "CloudySky.usd"
    sky = stage.DefinePrim("/World/sky", "Xform")
    sky.GetReferences().AddReference(str(sky_path.resolve()))
    return str(sky.GetPath())


def set_robot_pose(env_cfg, episode, robot=None):
    pos = list(episode["start_position"])
    spawn_height = getattr(env_cfg, "robot_spawn_height", 0.6)
    if getattr(env_cfg, "use_episode_robot_spawn_height", True):
        spawn_height = episode.get("robot_spawn_height", spawn_height)
    pos[2] += float(spawn_height)
    rot = list(episode["start_rotation"]) # wxyz
    env_cfg.scene.robot.init_state.pos = pos
    env_cfg.scene.robot.init_state.rot = rot
    if robot is not None:
        with torch.inference_mode():
            robot_root_state = robot.data.default_root_state.clone()
            robot_root_state[:, 0:3] = torch.tensor(pos, device=robot.device, dtype=robot_root_state.dtype)
            robot_root_state[:, 3:7] = torch.tensor(rot, device=robot.device, dtype=robot_root_state.dtype)
            robot_root_state[:, 7:] = 0.
            if getattr(env_cfg, "control_mode", None) == "kinematic_displacement" and hasattr(robot, "write_root_pose_to_sim"):
                robot.write_root_pose_to_sim(robot_root_state[:, :7])
            else:
                robot.write_root_state_to_sim(robot_root_state)
            # robot.reset()
            robot.write_data_to_sim()

class VLNEnvWrapper:
    """Wrapper to configure an :class:`RslRlVecEnvWrapper` instance to VLN environment."""

    def __init__(self, args, env, low_level_policy, robot_name, measure_names=None):
        self.env = env
        self.manager_env = env.unwrapped
        self.sim = self.manager_env.sim
        self.device = self.manager_env.device
        self.scene = self.manager_env.scene
        self.robot_name = robot_name
        self.first_init = True
        self.usd_path = None
        self.scene_render_warmup_required = False
        self.scene_scale = None
        self.scene_setting = None
        self.args = args
        self.num_envs = self.manager_env.num_envs

        if measure_names is None:
            measure_names = [
                "PathLength",
                "HorizontalPathLength",
                "CoveredArea",
                "ExplorationEfficiency",
                "DistanceToGoal",
                "DistanceToWall",
                "AverageDistanceToWall",
                "Collision",
                "CollisionRate",
                "ClosestGoal",
                "NavigableSurface",
                "NavigableSurfaceRatio",
                "Success",
                "SoftSuccess",
                "SPL",
                "SoftSPL",
                "OracleNavigationError",
                "OracleSuccess",
                "SimDuration",
                "AverageHorizontalSpeed",
            ]
        self.measure_names = measure_names

        self.env_step = 0
        self.step_dt = self.manager_env.step_dt
        self.profile_stats = {}
        self.profile_step_count = 0
        self._terrain_collision_config_cache = {}
        self.terrain_eval = TerrainEvaluationRuntime.from_args(args)

        self.high_level_obs_key = "camera"
        if not self.args.disable_camera and not self.args.on_demand_render:
             # check if camera observation is available
            assert self.high_level_obs_key in self.env.observation_space.spaces.keys()

        self.low_level_policy = low_level_policy
        self.low_level_type = getattr(self.manager_env.cfg, "low_level_type", "direct")
        self.direct_control = self.low_level_type == "direct"
        self.control_mode = getattr(self.manager_env.cfg, "control_mode", "velocity")
        self.collision_response = getattr(self.manager_env.cfg, "collision_response", "block")
        self.arm_control_mode = getattr(self.manager_env.cfg, "arm_control_mode", None)
        self.arm_ik_body_name = getattr(self.manager_env.cfg, "arm_ik_body_name", "bracelet_link")
        self.arm_ik_planning_body_name = getattr(self.manager_env.cfg, "arm_ik_planning_body_name", "arm_base_link")
        self.arm_ik_planning_frame_pos = getattr(self.manager_env.cfg, "arm_ik_planning_frame_pos", (0.12, 0.0, 0.374775))
        self.arm_ik_planning_frame_rot = getattr(self.manager_env.cfg, "arm_ik_planning_frame_rot", (1.0, 0.0, 0.0, 0.0))
        self.arm_ik_method = getattr(self.manager_env.cfg, "arm_ik_method", "dls")
        self.base_joint_names = tuple(getattr(self.manager_env.cfg, "base_joint_names", ()))
        self.arm_joint_names = tuple(getattr(self.manager_env.cfg, "arm_joint_names", ()))
        self.gripper_joint_names = tuple(getattr(self.manager_env.cfg, "gripper_joint_names", ()))
        self.arm_home_joint_positions = tuple(getattr(self.manager_env.cfg, "arm_home_joint_positions", ()))
        self.gripper_home_joint_positions = tuple(getattr(self.manager_env.cfg, "gripper_home_joint_positions", ()))
        self.arm_ik_max_target_delta = float(getattr(self.manager_env.cfg, "arm_ik_max_target_delta", 1.0))
        self.arm_ik_max_joint_step = float(getattr(self.manager_env.cfg, "arm_ik_max_joint_step", 1.0))
        self.arm_ik_joint_limit_margin = float(getattr(self.manager_env.cfg, "arm_ik_joint_limit_margin", 0.02))
        self.navmesh_interface = None
        self._static_prim_bounding_box_cache = {}
        self.low_level_action = None
        self.last_tidybot_arm_result = None
        self._tidybot_ik_controller = None
        self._tidybot_ik_joint_ids = None
        self._tidybot_ik_body_idx = None
        self._tidybot_ik_jacobi_body_idx = None
        self._tidybot_ik_jacobi_joint_ids = None
        self._tidybot_ik_planning_body_idx = None
        self._tidybot_hold_joint_ids = None
        self._tidybot_hold_joint_pos = None
        self._tidybot_base_joint_ids = None
        self._tidybot_hold_base_root_pose = None
        self._tidybot_hold_base_joint_pos = None
        self._tidybot_base_navmesh_z_offset = None
        self._tidybot_base_body_idx = None
        self._tidybot_hold_base_this_step = False
        self.robot_height = getattr(self.manager_env.cfg, "robot_height", 0.61)

        if self.low_level_type == "rsl_rl" and self.low_level_policy is None:
            raise ValueError("low_level_type='rsl_rl' requires a low_level_policy")

        self.is_stop_called = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.measure_manager = None
        self.termination_states = {
            "stuck_prev_pos": torch.zeros(self.num_envs, 3, device=self.args.device),
            "stuck_same_pos_count": torch.zeros(self.num_envs, dtype=torch.int32, device=self.args.device),
            "stuck_dist_history": [[] for _ in range(self.num_envs)],
            "stuck_xy_dist_history": [[] for _ in range(self.num_envs)],
            "stuck_yaw_dist_history": [[] for _ in range(self.num_envs)],
            "back_n_forth_prev_pos": torch.zeros(self.num_envs, 3, device=self.args.device),
            "back_n_forth_same_pos_count": torch.zeros(self.num_envs, dtype=torch.int32, device=self.args.device),
        }
        self.terminations_cfg = VLNTerminationsCfg()
        if getattr(self.manager_env.cfg, "ignore_bad_orientation", False):
            self.terminations_cfg.bad_orientation = None
        self.termination_manager = TerminationManager(self.terminations_cfg, self)

    def _set_collision_enabled(self, prim, enabled):
        collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=bool(enabled))
        sim_utils.define_collision_properties(prim.GetPrimPath(), collider_cfg)

    def _remove_collision_properties(self, prim):
        removed = False
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            removed = True
        if PhysxSchema is not None and prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
            removed = True
        return removed

    def _disable_collision_subtree(self, root_prim):
        disabled_meshes = 0
        disabled_prims = 0
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if prim == root_prim or prim.IsA(UsdGeom.Mesh):
                try:
                    if self._remove_collision_properties(prim):
                        disabled_prims += 1
                    if prim.IsA(UsdGeom.Mesh):
                        disabled_meshes += 1
                except Exception as exc:
                    print(f"[ENV] Failed to disable collision on {prim.GetPath()}: {exc}")
        return disabled_prims, disabled_meshes

    def _path_is_under(self, path, root_path):
        path = str(path)
        root_path = str(root_path).rstrip("/")
        return path == root_path or path.startswith(f"{root_path}/")

    def _enable_physical_terrain_mesh_colliders(self, terrain_prim, excluded_paths):
        enabled_meshes = 0
        for prim in Usd.PrimRange(terrain_prim, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if (
                prim.GetCustomDataByKey(INNOUT_COLLISION_ROLE_KEY)
                == INNOUT_VISUAL_ONLY_COLLISION_ROLE
            ):
                continue
            prim_path = str(prim.GetPath())
            if any(self._path_is_under(prim_path, excluded_path) for excluded_path in excluded_paths):
                continue
            try:
                self._set_collision_enabled(prim, True)
                enabled_meshes += 1
            except Exception as exc:
                print(f"[ENV] Failed to enable collision on {prim_path}: {exc}")
        return enabled_meshes

    def _terrain_collision_config(self, scene_type=None):
        if scene_type:
            config_path = resolve_task_config_path(
                self.episode.get("config"),
                scene_id=self.episode.get("scene_id"),
                scene_type=scene_type,
            )
            if not config_path.exists():
                return _config_value(
                    getattr(self.args, "navverse", None),
                    "terrain_collision",
                    default=None,
                )
            cache_key = str(config_path)
            if cache_key not in self._terrain_collision_config_cache:
                runtime_config = load_config(config_path)
                self._terrain_collision_config_cache[cache_key] = _config_value(
                    runtime_config.get("navverse"),
                    "terrain_collision",
                    default=None,
                )
            return self._terrain_collision_config_cache[cache_key]
        return _config_value(
            getattr(self.args, "navverse", None),
            "terrain_collision",
            default=None,
        )

    def _replacement_surface_mesh_config(self, scene_type=None):
        config = self._terrain_collision_config(scene_type)
        by_scene_type = _config_value(config, "replacement_surface_mesh_by_scene_type", default={}) or {}
        scene_config = _config_value(by_scene_type, scene_type, default=None) if scene_type else None
        return scene_config or _config_value(config, "replacement_surface_mesh", default=None)

    def _navmesh_boundary_collision_config(self, scene_type):
        if self._use_legacy_vcp_collision(scene_type):
            return None
        config = self._terrain_collision_config(scene_type)
        by_scene_type = _config_value(
            config,
            "navmesh_boundary_collision_by_scene_type",
            default={},
        ) or {}
        return _config_value(by_scene_type, scene_type, default=None)

    def _physical_collision_subtrees(self, scene_type):
        if self._use_legacy_vcp_collision(scene_type):
            return ()
        config = self._terrain_collision_config(scene_type)
        by_scene_type = _config_value(config, "physical_subtrees_by_scene_type", default={}) or {}
        subtrees = _config_value(
            by_scene_type,
            scene_type,
            default=MISSING_COLLISION_SUBTREES,
        )
        if subtrees is MISSING_COLLISION_SUBTREES:
            return DEFAULT_PHYSICAL_COLLISION_SUBTREES.get(scene_type)
        return subtrees or ()

    @staticmethod
    def _use_legacy_vcp_collision(scene_type):
        return scene_type == "vcp" and os.environ.get(
            "NAVVERSE_LEGACY_VCP_COLLISION",
            "0",
        ).lower() in {"1", "true", "yes"}

    def _configure_terrain_collision(self, terrain_prim, scene_type):
        config = self._terrain_collision_config(scene_type)
        filtered_scene_types = set(_config_value(config, "filtered_scene_types", default=()) or ())
        if scene_type not in filtered_scene_types:
            self._set_collision_enabled(terrain_prim, True)
            return

        self._remove_collision_properties(terrain_prim)
        physical_subtrees = self._physical_collision_subtrees(scene_type)
        replacement_by_scene_type = _config_value(config, "replacement_surface_mesh_by_scene_type", default={}) or {}
        has_scene_replacement_surface = (
            _config_value(replacement_by_scene_type, scene_type, default=None) is not None
        )
        if has_scene_replacement_surface and physical_subtrees is None:
            raise ValueError(
                "A scene-specific replacement collision surface requires an explicit "
                f"physical_subtrees_by_scene_type entry for {scene_type!r}"
            )

        if physical_subtrees is not None:
            disabled_prims, disabled_meshes = self._disable_collision_subtree(terrain_prim)
            if has_scene_replacement_surface:
                self._create_replacement_surface_mesh(terrain_prim, scene_type)
            enabled_meshes = 0
            terrain_path = str(terrain_prim.GetPrimPath())
            for relative_path in physical_subtrees:
                subtree_path = f"{terrain_path}/{str(relative_path).strip('/')}"
                subtree_prim = self.scene.stage.GetPrimAtPath(subtree_path)
                if not subtree_prim.IsValid():
                    print(f"[ENV] Physical terrain collision subtree not found: {subtree_path}")
                    continue
                enabled_meshes += self._enable_physical_terrain_mesh_colliders(subtree_prim, [])
            print(
                "[ENV] Filtered terrain collision to physical subtrees: "
                f"disabled_prims={disabled_prims}, disabled_meshes={disabled_meshes}, "
                f"enabled_meshes={enabled_meshes}"
            )
            return

        disabled_prims = 0
        disabled_meshes = 0
        terrain_path = str(terrain_prim.GetPrimPath())
        excluded_paths = []
        for relative_path in _config_value(config, "non_physical_subtrees", default=()) or ():
            subtree_path = f"{terrain_path}/{relative_path}"
            excluded_paths.append(subtree_path)
            subtree_prim = self.scene.stage.GetPrimAtPath(subtree_path)
            if not subtree_prim.IsValid():
                continue
            prim_count, mesh_count = self._disable_collision_subtree(subtree_prim)
            disabled_prims += prim_count
            disabled_meshes += mesh_count
        if disabled_prims or disabled_meshes:
            print(
                "[ENV] Disabled non-physical terrain collision layers: "
                f"prims={disabled_prims}, meshes={disabled_meshes}"
            )
        self._create_replacement_surface_mesh(terrain_prim, scene_type)
        surface_config = self._replacement_surface_mesh_config(scene_type)
        if bool(_config_value(surface_config, "enabled", default=False)):
            excluded_paths.append(f"{terrain_path}/{_config_value(surface_config, 'source')}")
        enabled_meshes = self._enable_physical_terrain_mesh_colliders(terrain_prim, excluded_paths)
        print(f"[ENV] Enabled physical terrain mesh colliders: meshes={enabled_meshes}")

    def _triangulated_mesh_faces(self, mesh_prim):
        mesh = UsdGeom.Mesh(mesh_prim)
        raw_counts = mesh.GetFaceVertexCountsAttr().Get()
        raw_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if raw_counts is None or raw_indices is None:
            return np.asarray([], dtype=np.int32).reshape(0, 3)
        counts = np.asarray(raw_counts, dtype=np.int32)
        indices = np.asarray(raw_indices, dtype=np.int32)
        if counts.ndim != 1 or indices.ndim != 1:
            return np.asarray([], dtype=np.int32).reshape(0, 3)
        faces = []
        offset = 0
        for count in counts:
            polygon = indices[offset: offset + count]
            offset += int(count)
            if count < 3:
                continue
            for index in range(1, int(count) - 1):
                faces.append([polygon[0], polygon[index], polygon[index + 1]])
        return np.asarray(faces, dtype=np.int32)

    def _mesh_points_relative_to_terrain(self, mesh_prim, terrain_prim):
        mesh = UsdGeom.Mesh(mesh_prim)
        raw_points = mesh.GetPointsAttr().Get()
        if raw_points is None:
            return np.asarray([], dtype=np.float32).reshape(0, 3)
        points = np.asarray(raw_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[-1] != 3:
            return np.asarray([], dtype=np.float32).reshape(0, 3)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        mesh_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
        world_to_terrain = xform_cache.GetLocalToWorldTransform(terrain_prim).GetInverse()
        return np.asarray(
            [
                world_to_terrain.Transform(
                    mesh_to_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                )
                for point in points
            ],
            dtype=np.float32,
        )

    def _lookup_grid_height(self, points, grid, min_xy, surface_config):
        cell_size = float(_config_value(surface_config, "grid_cell_size"))
        ix = np.clip(((points[:, 0] - min_xy[0]) / cell_size).astype(np.int64), 0, grid.shape[1] - 1)
        iy = np.clip(((points[:, 1] - min_xy[1]) / cell_size).astype(np.int64), 0, grid.shape[0] - 1)
        return grid[iy, ix]

    def _build_collision_height_grid(self, points, faces, upward_mask, surface_config):
        from scipy import ndimage

        upward_vertices = points[np.unique(faces[upward_mask].reshape(-1))]
        xy = upward_vertices[:, :2]
        z = upward_vertices[:, 2].astype(np.float32)
        cell_size = float(_config_value(surface_config, "grid_cell_size"))
        min_xy = xy.min(axis=0) - cell_size
        max_xy = xy.max(axis=0) + cell_size
        grid_shape = np.ceil((max_xy - min_xy) / cell_size).astype(np.int64) + 1
        grid = np.full((grid_shape[1], grid_shape[0]), -np.inf, dtype=np.float32)
        ix = np.clip(((xy[:, 0] - min_xy[0]) / cell_size).astype(np.int64), 0, grid_shape[0] - 1)
        iy = np.clip(((xy[:, 1] - min_xy[1]) / cell_size).astype(np.int64), 0, grid_shape[1] - 1)
        np.maximum.at(grid, (iy, ix), z)
        missing = ~np.isfinite(grid)
        if np.any(missing):
            nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
            grid = grid[tuple(nearest)]
        return grid, min_xy.astype(np.float32)

    def _collect_surface_source_meshes(self, source_prim, terrain_prim):
        mesh_prims = [source_prim] if source_prim.IsA(UsdGeom.Mesh) else [
            prim for prim in Usd.PrimRange(source_prim, Usd.TraverseInstanceProxies()) if prim.IsA(UsdGeom.Mesh)
        ]
        all_points = []
        all_faces = []
        vertex_offset = 0
        for mesh_prim in mesh_prims:
            points = self._mesh_points_relative_to_terrain(mesh_prim, terrain_prim)
            faces = self._triangulated_mesh_faces(mesh_prim)
            if len(points) == 0 or len(faces) == 0:
                continue
            all_points.append(points)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(points)
        if not all_points:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int32), len(mesh_prims)
        return np.concatenate(all_points, axis=0), np.concatenate(all_faces, axis=0), len(mesh_prims)

    def _create_replacement_surface_mesh(self, terrain_prim, scene_type=None):
        surface_config = self._replacement_surface_mesh_config(scene_type)
        if not bool(_config_value(surface_config, "enabled", default=False)):
            return
        terrain_path = str(terrain_prim.GetPrimPath())
        source_path = f"{terrain_path}/{_config_value(surface_config, 'source')}"
        source_prim = self.scene.stage.GetPrimAtPath(source_path)
        if not source_prim.IsValid():
            print(f"[ENV] Terrain surface collision source not found: {source_path}")
            return

        points, faces, source_mesh_count = self._collect_surface_source_meshes(source_prim, terrain_prim)
        if len(points) == 0 or len(faces) == 0:
            print(f"[ENV] Terrain surface collision source is empty: {source_path}")
            return

        triangles = points[faces]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(edge_a, edge_b)
        normal_lengths = np.linalg.norm(normals, axis=1)
        normal_z = np.divide(normals[:, 2], normal_lengths, out=np.zeros_like(normal_lengths), where=normal_lengths > 1e-12)
        upward_mask = normal_z > float(_config_value(surface_config, "normal_z_min"))
        if not np.any(upward_mask):
            print(f"[ENV] No upward terrain faces found for collision source: {source_path}")
            return

        grid, min_xy = self._build_collision_height_grid(points, faces, upward_mask, surface_config)
        centroids = triangles.mean(axis=1)
        height_margin = float(_config_value(surface_config, "height_margin"))
        surface_mask = np.zeros(len(triangles), dtype=bool)
        sample_points = (
            triangles[:, 0],
            triangles[:, 1],
            triangles[:, 2],
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
            centroids,
        )
        # Preserve a large ground triangle when only part of it overlaps a higher grid cell.
        for sample in sample_points:
            ground_z = self._lookup_grid_height(sample, grid, min_xy, surface_config)
            surface_mask |= sample[:, 2] >= ground_z - height_margin
        surface_mask &= upward_mask
        surface_faces = faces[surface_mask]
        if len(surface_faces) == 0:
            print(f"[ENV] No filtered terrain surface collision faces kept for: {source_path}")
            return

        used_vertices, remapped_indices = np.unique(surface_faces.reshape(-1), return_inverse=True)
        collision_points = points[used_vertices]
        collision_indices = remapped_indices.astype(np.int32).tolist()
        collision_counts = [3] * len(surface_faces)

        collision_path = f"{terrain_path}/{_config_value(surface_config, 'prim_name')}"
        if self.scene.stage.GetPrimAtPath(collision_path).IsValid():
            self.scene.stage.RemovePrim(collision_path)
        collision_mesh = UsdGeom.Mesh.Define(self.scene.stage, collision_path)
        collision_mesh.CreatePointsAttr(
            [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in collision_points]
        )
        collision_mesh.CreateFaceVertexCountsAttr(collision_counts)
        collision_mesh.CreateFaceVertexIndicesAttr(collision_indices)
        collision_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdGeom.Imageable(collision_mesh.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

        collision_prim = collision_mesh.GetPrim()
        self._remove_collision_properties(source_prim)
        self._set_collision_enabled(collision_prim, True)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
        mesh_collision_api.CreateApproximationAttr(UsdPhysics.Tokens.none)
        print(
            "[ENV] Replaced terrain collision mesh with top surface: "
            f"source_meshes={source_mesh_count}, source_faces={len(faces)}, kept_faces={len(surface_faces)}, "
            f"vertices={len(collision_points)}"
        )

    def _remove_navmesh_boundary_collision(self):
        prim = self.scene.stage.GetPrimAtPath(NAVMESH_BOUNDARY_COLLISION_PATH)
        if prim.IsValid():
            self.scene.stage.RemovePrim(NAVMESH_BOUNDARY_COLLISION_PATH)

    def _configure_navmesh_boundary_collision(self, scene_type):
        config = self._navmesh_boundary_collision_config(scene_type)
        if not bool(_config_value(config, "enabled", default=False)):
            self._remove_navmesh_boundary_collision()
            return
        if self.navmesh_interface is None or not getattr(self.navmesh_interface, "built", False):
            raise RuntimeError(
                f"Navmesh boundary collision is enabled for {scene_type!r}, but no navmesh is loaded"
            )

        vertices = np.asarray(self.navmesh_interface.navmesh_v, dtype=np.float32)
        triangles = np.asarray(self.navmesh_interface.navmesh_t, dtype=np.int32).reshape(-1, 3)
        if len(vertices) == 0 or len(triangles) == 0:
            raise RuntimeError(f"Navmesh boundary collision geometry is empty for {scene_type!r}")

        edge_pairs = np.concatenate(
            (
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            ),
            axis=0,
        )
        edge_owners = np.tile(np.arange(len(triangles), dtype=np.int32), 3)
        undirected_edges = np.sort(edge_pairs, axis=1)
        _, inverse, edge_counts = np.unique(
            undirected_edges,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        boundary_mask = edge_counts[inverse] == 1
        boundary_edges = edge_pairs[boundary_mask]
        boundary_owners = edge_owners[boundary_mask]

        p0 = vertices[boundary_edges[:, 0]]
        p1 = vertices[boundary_edges[:, 1]]
        triangle_centers = vertices[triangles[boundary_owners]].mean(axis=1)
        edge_xy = p1[:, :2] - p0[:, :2]
        edge_lengths = np.linalg.norm(edge_xy, axis=1)
        edge_unit = np.divide(
            edge_xy,
            edge_lengths[:, None],
            out=np.zeros_like(edge_xy),
            where=edge_lengths[:, None] > 1e-8,
        )
        midpoint_xy = 0.5 * (p0[:, :2] + p1[:, :2])
        inward_xy = triangle_centers[:, :2] - midpoint_xy
        inward_xy -= np.sum(inward_xy * edge_unit, axis=1)[:, None] * edge_unit
        inward_lengths = np.linalg.norm(inward_xy, axis=1)

        min_edge_length = float(_config_value(config, "min_edge_length", default=0.05))
        valid = (edge_lengths >= min_edge_length) & (inward_lengths > 1e-8)
        p0 = p0[valid]
        p1 = p1[valid]
        outward_xy = -np.divide(
            inward_xy[valid],
            inward_lengths[valid, None],
            out=np.zeros_like(inward_xy[valid]),
            where=inward_lengths[valid, None] > 1e-8,
        )

        outward_offset = float(_config_value(config, "outward_offset", default=0.0))
        wall_thickness = float(_config_value(config, "wall_thickness", default=0.2))
        below_surface = float(_config_value(config, "below_surface", default=0.3))
        above_surface = float(_config_value(config, "above_surface", default=1.0))
        if min(min_edge_length, outward_offset, wall_thickness, below_surface, above_surface) < 0.0:
            raise ValueError("Navmesh boundary collision dimensions must be non-negative")
        if wall_thickness == 0.0:
            raise ValueError("Navmesh boundary collision wall_thickness must be positive")
        if len(p0) == 0:
            raise RuntimeError(f"Navmesh boundary collision has no valid edges for {scene_type!r}")

        center_p0 = p0.copy()
        center_p1 = p1.copy()
        center_p0[:, :2] += outward_xy * outward_offset
        center_p1[:, :2] += outward_xy * outward_offset
        half_thickness_xy = outward_xy * (0.5 * wall_thickness)
        wall_points = np.empty((len(p0), 8, 3), dtype=np.float32)
        for side, side_offset in enumerate((-half_thickness_xy, half_thickness_xy)):
            base_index = side * 4
            wall_points[:, base_index] = center_p0
            wall_points[:, base_index + 1] = center_p1
            wall_points[:, base_index + 2] = center_p1
            wall_points[:, base_index + 3] = center_p0
            wall_points[:, base_index:base_index + 4, :2] += side_offset[:, None, :]
            wall_points[:, base_index, 2] -= below_surface
            wall_points[:, base_index + 1, 2] -= below_surface
            wall_points[:, base_index + 2, 2] += above_surface
            wall_points[:, base_index + 3, 2] += above_surface

        quad_indices = np.asarray(
            (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (3, 7, 6, 2),
                (0, 4, 7, 3),
                (1, 2, 6, 5),
            ),
            dtype=np.int32,
        )
        triangle_indices = np.concatenate(
            (
                quad_indices[:, [0, 1, 2]],
                quad_indices[:, [0, 2, 3]],
            ),
            axis=0,
        )
        base = (np.arange(len(wall_points), dtype=np.int32) * 8)[:, None, None]
        wall_faces = (base + triangle_indices[None, :, :]).reshape(-1, 3)
        self._remove_navmesh_boundary_collision()
        collision_mesh = UsdGeom.Mesh.Define(
            self.scene.stage,
            NAVMESH_BOUNDARY_COLLISION_PATH,
        )
        collision_mesh.CreatePointsAttr(
            [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in wall_points.reshape(-1, 3)]
        )
        collision_mesh.CreateFaceVertexCountsAttr([3] * len(wall_faces))
        collision_mesh.CreateFaceVertexIndicesAttr(wall_faces.reshape(-1).tolist())
        collision_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        collision_mesh.CreateDoubleSidedAttr(True)
        UsdGeom.Imageable(collision_mesh.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        collision_prim = collision_mesh.GetPrim()
        self._set_collision_enabled(collision_prim, True)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
        mesh_collision_api.CreateApproximationAttr(UsdPhysics.Tokens.none)
        print(
            "[ENV] Created navmesh boundary collision: "
            f"edges={len(wall_points)}, vertices={len(wall_points) * 8}, "
            f"faces={len(wall_faces)}, outward_offset={outward_offset:.2f}, "
            f"wall_thickness={wall_thickness:.2f}"
        )

    def _reset_low_level_policy_state(self):
        policy_owner = getattr(self.low_level_policy, "__self__", None)
        if policy_owner is not None and hasattr(policy_owner, "reset"):
            dones = torch.ones(self.num_envs, dtype=torch.bool, device=self.manager_env.device)
            policy_owner.reset(dones)
        self.low_level_action = self._empty_env_action()

    def _profile_record(self, key, duration_s):
        if not getattr(self.args, "profile_step_timing", False):
            return
        total_s, count = self.profile_stats.get(key, (0.0, 0))
        self.profile_stats[key] = (total_s + float(duration_s), count + 1)

    def _profile_summary(self):
        if not getattr(self.args, "profile_step_timing", False):
            return
        interval = max(1, int(getattr(self.args, "profile_step_interval", 100) or 100))
        if self.profile_step_count == 0 or self.profile_step_count % interval != 0:
            return
        step_total = self.profile_stats.get("wrapper/total", (0.0, 0))[0]
        print(f"[PROFILE][wrapper] steps={self.profile_step_count}, window_keys={len(self.profile_stats)}")
        for key, (total_s, count) in sorted(self.profile_stats.items()):
            pct = 100.0 * total_s / step_total if step_total > 0 else 0.0
            avg_ms = 1000.0 * total_s / max(count, 1)
            print(f"[PROFILE][wrapper] {key}: total={total_s:.3f}s count={count} avg={avg_ms:.2f}ms pct={pct:.1f}%")
        self.profile_stats.clear()

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.
        """
        return self.manager_env
    
    def _get_high_level_obs(self, obs):
        if self.high_level_obs_key in obs:
            return obs[self.high_level_obs_key]
        if self.args.on_demand_render or self.args.disable_camera:
            return {}
        raise KeyError(self.high_level_obs_key)

    def get_prim_bounding_box(self, prim_path):
        """Return min_x, min_y, min_z, max_x, max_y, max_z of the prim."""
        prim_path_str = str(prim_path)
        cache_static = prim_path_str == "/World/ground/terrain"
        if cache_static and prim_path_str in self._static_prim_bounding_box_cache:
            return self._static_prim_bounding_box_cache[prim_path_str]
        prim = self.scene.stage.GetPrimAtPath(prim_path)
        if prim is None:
            raise ValueError(f"Prim at path {prim_path} not found")
        bb_cache = bounds_utils.create_bbox_cache()
        bounds = bounds_utils.compute_combined_aabb(bb_cache, prim_paths=[prim_path])
        if cache_static:
            self._static_prim_bounding_box_cache[prim_path_str] = bounds
        return bounds

    def remove_prim(self, rule):
        prim_list = prim_utils.find_matching_prim_paths(rule)
        for prim_path in prim_list:
            prim_utils.delete_prim(prim_path)

    def get_prim_position(self, prim_path):
        """Return position of the prim (center of the bounding box)."""
        min_x, min_y, min_z, max_x, max_y, max_z = self.get_prim_bounding_box(prim_path)
        return (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2

    def get_prim_radius(self, prim_path):
        """Return radius of the prim."""
        min_x, min_y, min_z, max_x, max_y, max_z = self.get_prim_bounding_box(prim_path)
        return max(max_x - min_x, max_y - min_y, max_z - min_z) / 2

    def get_prim_orientation(self, prim_path):
        """Return orientation of the prim."""
        prim = self.scene.stage.GetPrimAtPath(prim_path)
        if prim is None:
            raise ValueError(f"Prim at path {prim_path} not found")
        return prim.GetAttribute('xformOp:orient').Get()
    
    def _pose_state_w(self):
        robot = self.manager_env.scene["robot"]
        if self.robot_name == "tidybot" and self.control_mode == "physics_velocity":
            if self._tidybot_base_body_idx is None:
                body_ids, body_names = robot.find_bodies("base")
                if len(body_ids) != 1:
                    raise RuntimeError(
                        f"Expected one TidyBot base body, found {body_names}"
                    )
                self._tidybot_base_body_idx = int(body_ids[0])
            return robot.data.body_state_w[:, self._tidybot_base_body_idx, :7]
        return robot.data.root_state_w[:, :7]

    def get_body_pose(self):
        pose_state = self._pose_state_w()
        pos_robot = pose_state[0, 0:3].cpu().numpy().astype(np.float32)
        quat_robot = pose_state[0, 3:7].cpu().numpy().astype(np.float32) # wxyz
        quat_robot = np.concatenate([quat_robot[1:],quat_robot[:1]])
        return pos_robot, quat_robot

    def get_cam_pose(self):
        manager_env = self.manager_env
        pose_state = self._pose_state_w()
        pos_robot = pose_state[0, 0:3].cpu().numpy().astype(np.float32)
        quat_robot = pose_state[0, 3:7].cpu().numpy().astype(np.float32) # wxyz
        quat_robot = R.from_quat(np.concatenate([quat_robot[1:],quat_robot[:1]]))
        # transform from robot to camera
        pos_cam_body = manager_env.scene["pov_camera"].cfg.offset.pos
        quat_cam_body = manager_env.scene["pov_camera"].cfg.offset.rot # wxyz
        quat_cam_body = R.from_quat(np.concatenate([quat_cam_body[1:],quat_cam_body[:1]]))
        # camera pose in world frame
        pos_cam_world = pos_robot + quat_robot.as_matrix() @ pos_cam_body
        quat_cam_world = quat_robot * quat_cam_body
        quat_cam_world = quat_cam_world.as_quat() # xyzw
        # pose = manager_env.scene["pov_camera"]._view.get_world_poses()
        # pos = pose[0][0].detach().cpu().numpy()
        # quat = convert_camera_frame_orientation_convention(pose[1][0], origin="opengl", target="world").detach().cpu().numpy()
        # quat = np.concatenate([quat[1:],quat[:1]])
        # return pos, quat
        return pos_cam_world, quat_cam_world

    def get_camera_prim_path(self, camera_name: str, env_index: int = 0) -> str:
        prim_path = self.manager_env.scene[camera_name].cfg.prim_path
        prim_path = prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_index}")
        prim_path = re.sub(r"/World/envs/env_\.\*", f"/World/envs/env_{env_index}", prim_path)
        prim_path = re.sub(r"env_\.\*", f"env_{env_index}", prim_path)
        return prim_path
    
    def create_camera(
        self,
        prim_path="/World/camera",
        perspective=True,
        pos=None,
        quat_opengl=None,
        focal_length=24.0,
        horizontal_aperture=20.955,
        clipping_range_min=0.1,
        clipping_range_max=20.0,
        width=640,
        height=480,
        data_types=None,
        offset_convention="opengl",
        colorize_instance_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_semantic_segmentation=True,
    ):
        if prim_utils.get_prim_at_path(prim_path):
            prim_utils.delete_prim(prim_path)
        camera_cfg = TiledCameraCfg(
            prim_path=prim_path,
            update_period=0.1,
            update_latest_camera_pose=True,
            offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=quat_opengl, convention=offset_convention),
            width=width,
            height=height,
            data_types=data_types or ["rgb", "distance_to_image_plane"],
            colorize_instance_segmentation=colorize_instance_segmentation,
            colorize_instance_id_segmentation=colorize_instance_id_segmentation,
            colorize_semantic_segmentation=colorize_semantic_segmentation,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, horizontal_aperture=horizontal_aperture,
                clipping_range=(clipping_range_min, clipping_range_max)
            ),
        )
        tiled_camera = TiledCamera(camera_cfg)
        if not perspective:
            camera_prim = self.manager_env.scene.stage.GetPrimAtPath(prim_path)
            camera_prim.GetAttribute('projection').Set("orthographic")
        tiled_camera._initialize_callback(None)
        return tiled_camera

    def _step_low_level_env(self, env_action):
        """Normalize step outputs using the configured low-level control type."""
        profile_start = time.perf_counter()
        carb.profiler.begin(1, "NavVerse.wrapper/raw_env_step")
        try:
            step_result = self.env.step(env_action)
        finally:
            carb.profiler.end(1)
        self._profile_record("wrapper/raw_env_step", time.perf_counter() - profile_start)
        if self.low_level_type == "rsl_rl":
            if len(step_result) != 4:
                raise ValueError(f"Expected 4 values from rsl_rl env.step, got {len(step_result)}")
            return step_result
        if self.low_level_type == "direct":
            if len(step_result) != 5:
                raise ValueError(f"Expected 5 values from direct env.step, got {len(step_result)}")
            obs, reward, terminated, truncated, info = step_result
            done = terminated | truncated
            return obs, reward, done, info
        raise ValueError(f"Unsupported low_level_type: {self.low_level_type}")

    def _pump_ui_events_if_available(self):
        pump = getattr(self, "ui_event_pump", None)
        if pump is None:
            return
        try:
            pump()
        except Exception as exc:
            print(f"[ENV] UI event pump failed during reset: {exc}")

    def reset(self, episode=None, warmup_steps=0) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
        zero_cmd = torch.tensor([0., 0., 0.], device=self.args.device)
        self._reset_low_level_policy_state()
        if episode is not None:
            self.episode = episode
        else:
            if self.episode is None:
                raise ValueError("Episode is not set")
        episode = self.episode
        # access episode settings
        collider = self.episode.get("collider", True)
        scene_scale = self.episode.get("scene_scale", 1.0)
        align_ground = self.episode.get("align_ground", True)
        scene_settings = {
            "scale": scene_scale,
            "align_ground": align_ground,
            "collider": collider,
        }

        # load measures
        self.measure_manager = add_measurement(self, self.episode, self.measure_names)

        # load scene
        scene_changed = self.usd_path != self.episode["path"]
        settings_changed = (self.scene_setting is None) or (self.scene_setting != scene_settings)
        scene_load_start_time = time.perf_counter()

        # do not update scene if it is the first init
        if scene_changed or settings_changed:
            self._static_prim_bounding_box_cache.clear()
            self._remove_navmesh_boundary_collision()
            # remove previous scene
            while len(self.scene.terrain.terrain_prim_paths) > 0:
                self.scene.stage.RemovePrim(self.scene.terrain.terrain_prim_paths[0])
                while self.scene.stage.GetPrimAtPath(self.scene.terrain.terrain_prim_paths[0]).IsValid():
                    env_action = self._command_to_env_action(zero_cmd)
                    _, _, _, infos = self._step_low_level_env(env_action)
                self.scene.terrain.terrain_prim_paths.pop(0)
            print("[ENV] Waiting for previous scene to be removed...")
            self._pump_ui_events_if_available()
            # load new scene
            load_scene(self.manager_env.cfg, self.args, self.episode)
            self._pump_ui_events_if_available()
            self.manager_env.scene._terrain = TerrainImporter(self.manager_env.cfg.scene.terrain)
            self._pump_ui_events_if_available()
            if self.episode["path"] == "generator":
                self.scene.terrain.terrain_prim_paths.append("/World/ground/terrain")
            if self._use_legacy_vcp_collision(self.episode.get("scene_type")):
                runtime_lighting = configure_legacy_runtime_lighting(
                    self.scene.stage,
                    self.args.scene_folder,
                )
                print(f"[INFO] Enabled legacy NavVerse runtime lighting: {runtime_lighting}")
                self.scene_render_warmup_required = False
            else:
                disabled_lighting = disable_embedded_scene_lighting(self.scene.stage)
                print(f"[INFO] Disabled {len(disabled_lighting)} embedded scene lighting roots")
                runtime_lighting = configure_runtime_lighting(self.scene.stage, self.args.scene_folder)
                print(f"[INFO] Enabled NavVerse runtime lighting: {runtime_lighting}")
                self.scene_render_warmup_required = True
            # apply scale and translation
            terrain_prim = self.scene.stage.GetPrimAtPath('/World/ground/terrain')
            if collider:
                self._configure_terrain_collision(terrain_prim, self.episode.get("scene_type"))
            terrain_prim.GetAttribute('xformOp:scale').Set(Gf.Vec3f(scene_scale, scene_scale, scene_scale))
            if align_ground:
                min_x, min_y, min_z, max_x, max_y, max_z = self.get_prim_bounding_box(terrain_prim.GetPrimPath())
                print(f"Bounding box: min_x: {min_x}, min_y: {min_y}, min_z: {min_z}, max_x: {max_x}, max_y: {max_y}, max_z: {max_z}")
                terrain_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0, 0, -min_z-20))
                self._static_prim_bounding_box_cache.clear()
        self.scene_setting = scene_settings
        self.usd_path = self.episode["path"]

        # reset robot position
        robot = self.scene["robot"]
        robot_episode = self.terrain_eval.configure_episode(
            self.scene.stage, episode, self.manager_env.cfg
        )
        set_robot_pose(self.manager_env.cfg, robot_episode, robot)
        self.navmesh_interface = self._load_navmesh_for_current_scene()
        if collider:
            self._configure_navmesh_boundary_collision(self.episode.get("scene_type"))

        # reset termination states
        self.reset_termination_states([*range(self.num_envs)])
        self.termination_manager.reset()

        # reset low-level environment
        print("[INFO] Updating environment, this may take a while...")
        self._pump_ui_events_if_available()
        obs, infos = self.env.reset()
        self._pump_ui_events_if_available()
        self._reset_low_level_policy_state()
        self.reset_tidybot_arm_pose()
        self.reset_tidybot_base_pose()
        self.obs = obs
        for i in range(self.num_envs):
            self.set_stop_called(i, False)

        for i in range(warmup_steps+1):
            if i==0:
                print("Resetting environment...")
            elif i%10==0:
                print(f"Warmup {i} / {warmup_steps}...")
            env_action = self._command_to_env_action(zero_cmd)
            obs, _, _, infos = self._step_low_level_env(env_action)
            self.obs = obs
            self.low_level_action = env_action
            if i==0:
                print(f"[INFO] Scene loading completed in {time.perf_counter() - scene_load_start_time:.2f}s for {self.episode['path']}")
                self._pump_ui_events_if_available()

        self.reset_tidybot_base_pose()
        self.env_step = 0
        self.same_pos_count = 0

        self.measure_manager.reset_measures()
        measurements = self.measure_manager.get_measurements()
        measurements.update(self.terrain_eval.measurements())
        infos["measurements"] = measurements
        if self.last_tidybot_arm_result is not None:
            infos["tidybot_arm"] = self.last_tidybot_arm_result

        self.first_init = False
        high_level_obs = self._get_high_level_obs(self.obs)
        return high_level_obs, infos
    
    def reset_termination_states(self, reset_env_ids):
        for env_id in reset_env_ids:
            for key in self.termination_states.keys():
                if "dist_history" in key:
                    self.termination_states[key][int(env_id)] = []
                else:
                    self.termination_states[key][env_id] = 0
    
    def update_command(self, command) -> None:
        """Update the command for the low-level policy."""

        # make sure command is a tensor on the same device as low_level_obs
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=self.manager_env.device)
        if command.ndim == 1:
            command = command.unsqueeze(0).repeat(self.num_envs, 1)
        if self.obs is None or "policy" not in self.obs:
            return
        self.obs["policy"] = self.obs["policy"].clone()
        self.obs["policy"][:, 9:12] = command
        self.scene["robot"].velocity_command = command

    def parse_transform_command(self, transform):
        if isinstance(transform, dict):
            if "transform" in transform:
                transform = transform["transform"]
            elif "pose" in transform:
                transform = transform["pose"]
            elif "target" in transform:
                transform = transform["target"]
            if "position" in transform and "orientation" in transform:
                pos = transform["position"]
                rot = transform["orientation"]
                return torch.tensor([
                    pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0),
                    rot.get("x", 0.0), rot.get("y", 0.0), rot.get("z", 0.0), rot.get("w", 1.0),
                ], device=self.manager_env.device, dtype=torch.float32)
            if "translation" in transform and "rotation" in transform:
                pos = transform["translation"]
                rot = transform["rotation"]
                return torch.tensor([
                    pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0),
                    rot.get("x", 0.0), rot.get("y", 0.0), rot.get("z", 0.0), rot.get("w", 1.0),
                ], device=self.manager_env.device, dtype=torch.float32)
            if "matrix" in transform:
                transform = transform["matrix"]
            else:
                return torch.tensor([
                    transform.get("x", 0.0), transform.get("y", 0.0), transform.get("z", 0.0),
                    transform.get("qx", 0.0), transform.get("qy", 0.0), transform.get("qz", 0.0), transform.get("qw", 1.0),
                ], device=self.manager_env.device, dtype=torch.float32)
        if torch.is_tensor(transform):
            arr = transform.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(transform, dtype=np.float32)
        if arr.shape == (4, 4):
            quat_xyzw = R.from_matrix(arr[:3, :3]).as_quat()
            arr = np.concatenate([arr[:3, 3], quat_xyzw]).astype(np.float32)
        arr = arr.reshape(-1)
        if arr.size != 7:
            raise ValueError(f"Transform command must contain 7 values or a 4x4 matrix, got {arr.size}")
        return torch.tensor(arr, device=self.manager_env.device, dtype=torch.float32)

    def _empty_env_action(self):
        return torch.zeros((self.num_envs, 0), device=self.manager_env.device, dtype=torch.float32)

    def _get_robot_joint_names(self):
        robot = self.scene["robot"]
        names = getattr(robot.data, "joint_names", None)
        if names is not None:
            return list(names)
        names = getattr(robot, "joint_names", None)
        return list(names) if names is not None else []

    def _resolve_joint_ids(self, joint_names):
        robot = self.scene["robot"]
        if hasattr(robot, "find_joints"):
            joint_ids = []
            resolved_names = []
            for name in joint_names:
                ids, names = robot.find_joints(name, preserve_order=True)
                if len(ids) != 1:
                    raise ValueError(f"Expected one joint named {name}, found {names}")
                joint_ids.append(int(ids[0]))
                resolved_names.append(names[0])
            return joint_ids, resolved_names
        available = self._get_robot_joint_names()
        joint_ids = [available.index(name) for name in joint_names]
        return joint_ids, list(joint_names)

    def _normalise_joint_positions(self, value, default_names):
        if value is None:
            return [], []
        if isinstance(value, dict):
            return list(value.keys()), [float(v) for v in value.values()]
        if torch.is_tensor(value):
            values = value.detach().cpu().numpy().reshape(-1).astype(np.float32).tolist()
        else:
            values = np.asarray(value, dtype=np.float32).reshape(-1).tolist()
        if len(values) > len(default_names):
            raise ValueError(f"Got {len(values)} joint values but only {len(default_names)} default joints")
        return list(default_names[: len(values)]), [float(v) for v in values]

    def _parse_tidybot_pose_command(self, value):
        if isinstance(value, dict):
            frame = value.get("frame", value.get("frame_id", self.arm_ik_planning_body_name))
            pose_value = value.get("pose", value)
            pos = pose_value.get("position", pose_value.get("translation", pose_value))
            rot = pose_value.get("orientation", pose_value.get("rotation", pose_value))
            values = [
                pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0),
                rot.get("w", 1.0), rot.get("x", 0.0), rot.get("y", 0.0), rot.get("z", 0.0),
            ]
        else:
            frame = self.arm_ik_planning_body_name
            arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value, dtype=np.float32)
            arr = arr.reshape(-1)
            if arr.size != 7:
                raise ValueError(f"Arm target pose must contain 7 values, got {arr.size}")
            values = [arr[0], arr[1], arr[2], arr[6], arr[3], arr[4], arr[5]]
        return torch.tensor([values], device=self.device, dtype=torch.float32), frame

    def _ensure_tidybot_ik_controller(self):
        if self._tidybot_ik_controller is not None:
            return
        robot = self.scene["robot"]
        joint_ids, _ = self._resolve_joint_ids(self.arm_joint_names)
        body_ids, body_names = robot.find_bodies(self.arm_ik_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one IK body named {self.arm_ik_body_name}, found {body_names}")
        try:
            planning_body_ids, planning_body_names = robot.find_bodies(self.arm_ik_planning_body_name)
        except ValueError:
            planning_body_ids, planning_body_names = [], []
        if len(planning_body_ids) > 1:
            raise ValueError(f"Expected at most one planning body named {self.arm_ik_planning_body_name}, found {planning_body_names}")
        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method=self.arm_ik_method)
        self._tidybot_ik_controller = DifferentialIKController(cfg=cfg, num_envs=self.num_envs, device=self.device)
        self._tidybot_ik_joint_ids = [int(i) for i in joint_ids]
        self._tidybot_ik_body_idx = int(body_ids[0])
        self._tidybot_ik_planning_body_idx = int(planning_body_ids[0]) if planning_body_ids else None
        if robot.is_fixed_base:
            self._tidybot_ik_jacobi_body_idx = self._tidybot_ik_body_idx - 1
            self._tidybot_ik_jacobi_joint_ids = self._tidybot_ik_joint_ids
        else:
            self._tidybot_ik_jacobi_body_idx = self._tidybot_ik_body_idx
            self._tidybot_ik_jacobi_joint_ids = [i + 6 for i in self._tidybot_ik_joint_ids]

    def _body_pose_in_root(self, body_idx):
        robot = self.scene["robot"]
        body_pos_w = robot.data.body_pos_w[:, body_idx]
        body_quat_w = robot.data.body_quat_w[:, body_idx]
        return math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            body_pos_w,
            body_quat_w,
        )

    def _planning_frame_pose_in_root(self):
        if self._tidybot_ik_planning_body_idx is not None:
            return self._body_pose_in_root(self._tidybot_ik_planning_body_idx)
        pos = torch.tensor(self.arm_ik_planning_frame_pos, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        rot = torch.tensor(self.arm_ik_planning_frame_rot, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        return pos, rot

    def _target_pose_in_root(self, target_pose, frame):
        if frame in (self.arm_ik_planning_body_name, "arm_base", "planning"):
            base_pos, base_quat = self._planning_frame_pose_in_root()
            return math_utils.combine_frame_transforms(base_pos, base_quat, target_pose[:, :3], target_pose[:, 3:7])
        if frame in ("root", "base"):
            return target_pose[:, :3], target_pose[:, 3:7]
        if frame == "world":
            robot = self.scene["robot"]
            return math_utils.subtract_frame_transforms(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                target_pose[:, :3],
                target_pose[:, 3:7],
            )
        raise ValueError(f"Unsupported Tidybot arm target frame: {frame}")

    def _compute_tidybot_ik_jacobian(self):
        robot = self.scene["robot"]
        jacobian = robot.root_physx_view.get_jacobians()[
            :, self._tidybot_ik_jacobi_body_idx, :, self._tidybot_ik_jacobi_joint_ids
        ]
        base_rot = robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def reset_tidybot_arm_pose(self):
        if self.robot_name != "tidybot":
            return None
        joint_names = list(self.arm_joint_names)
        joint_values = list(self.arm_home_joint_positions)
        if not joint_names and not self.gripper_joint_names:
            return None
        if len(joint_values) != len(joint_names):
            return None
        if self.gripper_home_joint_positions:
            joint_names.extend(self.gripper_joint_names[: len(self.gripper_home_joint_positions)])
            joint_values.extend(float(v) for v in self.gripper_home_joint_positions)
        joint_ids, resolved_names = self._resolve_joint_ids(joint_names)
        robot = self.scene["robot"]
        joint_pos = torch.tensor([joint_values], device=robot.device, dtype=robot.data.joint_pos.dtype)
        self._write_tidybot_hold_pose(joint_ids, joint_pos, teleport=True)
        self.last_tidybot_arm_result = {"mode": "home_joint_pose", "joint_names": resolved_names, "joint_positions": joint_values}
        return self.last_tidybot_arm_result

    def _write_tidybot_hold_pose(self, joint_ids, joint_pos, remember=True, teleport=False):
        robot = self.scene["robot"]
        with torch.inference_mode():
            joint_vel = torch.zeros_like(joint_pos)
            if teleport:
                robot.write_joint_position_to_sim(joint_pos, joint_ids=joint_ids)
                robot.write_joint_velocity_to_sim(joint_vel, joint_ids=joint_ids)
            if hasattr(robot, "set_joint_position_target"):
                robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)
            else:
                robot.write_joint_position_to_sim(joint_pos, joint_ids=joint_ids)
            if hasattr(robot, "set_joint_velocity_target"):
                robot.set_joint_velocity_target(joint_vel, joint_ids=joint_ids)
            robot.write_data_to_sim()
        if teleport:
            self._forward_tidybot_kinematics()
        if remember:
            self._tidybot_hold_joint_ids = [int(i) for i in joint_ids]
            self._tidybot_hold_joint_pos = joint_pos.detach().clone()

    def _forward_tidybot_kinematics(self):
        sim = getattr(self.manager_env, "sim", None)
        if sim is not None:
            sim.forward()

    def _hold_tidybot_arm_pose(self):
        if self.robot_name != "tidybot":
            return
        if self._tidybot_hold_joint_ids is None or self._tidybot_hold_joint_pos is None:
            self.reset_tidybot_arm_pose()
            return
        robot = self.scene["robot"]
        joint_pos = self._tidybot_hold_joint_pos.to(device=robot.device, dtype=robot.data.joint_pos.dtype)
        with torch.inference_mode():
            joint_vel = torch.zeros_like(joint_pos)
            if hasattr(robot, "set_joint_position_target"):
                robot.set_joint_position_target(joint_pos, joint_ids=self._tidybot_hold_joint_ids)
            else:
                robot.write_joint_position_to_sim(joint_pos, joint_ids=self._tidybot_hold_joint_ids)
            if hasattr(robot, "set_joint_velocity_target"):
                robot.set_joint_velocity_target(joint_vel, joint_ids=self._tidybot_hold_joint_ids)
            robot.write_data_to_sim()

    def reset_tidybot_base_pose(self):
        if self.robot_name != "tidybot":
            return
        self._remember_tidybot_base_pose()
        self._tidybot_hold_base_this_step = False

    def _resolve_tidybot_base_joint_ids(self):
        if self._tidybot_base_joint_ids is not None:
            return self._tidybot_base_joint_ids
        if not self.base_joint_names:
            self._tidybot_base_joint_ids = []
            return self._tidybot_base_joint_ids
        joint_ids, _ = self._resolve_joint_ids(self.base_joint_names)
        self._tidybot_base_joint_ids = [int(i) for i in joint_ids]
        return self._tidybot_base_joint_ids

    def _remember_tidybot_base_pose(self, root_pose=None):
        if self.robot_name != "tidybot":
            return
        robot = self.scene["robot"]
        if root_pose is None:
            root_pose = robot.data.root_state_w[:, :7]
        self._tidybot_hold_base_root_pose = root_pose.detach().clone()
        base_joint_ids = self._resolve_tidybot_base_joint_ids()
        if base_joint_ids:
            self._tidybot_hold_base_joint_pos = robot.data.joint_pos[:, base_joint_ids].detach().clone()
        root_z = float(self._tidybot_hold_base_root_pose[0, 2].detach().cpu())
        surface_z = self._navmesh_surface_z_at_xy(
            self._tidybot_hold_base_root_pose[0, :2].detach().cpu().numpy(),
            reference_z=root_z,
        )
        if surface_z is not None:
            self._tidybot_base_navmesh_z_offset = root_z - float(surface_z)

    def _write_tidybot_base_root_pose(self, root_pose):
        robot = self.scene["robot"]
        root_state = robot.data.root_state_w.clone()
        root_state[:, :7] = root_pose.to(device=robot.device, dtype=root_state.dtype)
        root_state[:, 7:] = 0.0
        if hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root_state[:, :7])
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(root_state[:, 7:])
        else:
            robot.write_root_state_to_sim(root_state)

    def _hold_tidybot_base_pose(self):
        if self.robot_name != "tidybot":
            return
        if self._tidybot_hold_base_root_pose is None:
            self.reset_tidybot_base_pose()
        robot = self.scene["robot"]
        with torch.inference_mode():
            self._write_tidybot_base_root_pose(self._tidybot_hold_base_root_pose)
            base_joint_ids = self._resolve_tidybot_base_joint_ids()
            if base_joint_ids and self._tidybot_hold_base_joint_pos is not None:
                joint_pos = self._tidybot_hold_base_joint_pos.to(device=robot.device, dtype=robot.data.joint_pos.dtype)
                joint_vel = torch.zeros_like(joint_pos)
                robot.write_joint_position_to_sim(joint_pos, joint_ids=base_joint_ids)
                robot.write_joint_velocity_to_sim(joint_vel, joint_ids=base_joint_ids)
                if hasattr(robot, "set_joint_position_target"):
                    robot.set_joint_position_target(joint_pos, joint_ids=base_joint_ids)
                if hasattr(robot, "set_joint_velocity_target"):
                    robot.set_joint_velocity_target(joint_vel, joint_ids=base_joint_ids)
            robot.write_data_to_sim()

    def _tidybot_rejected_arm_result(self, reason, frame, target_pos=None, joint_pos_des=None, joint_pos=None):
        result = {
            "mode": "ik_pose_rejected",
            "rejected": True,
            "reason": reason,
            "body_name": self.arm_ik_body_name,
            "frame": frame,
        }
        if target_pos is not None:
            result["target_pos_root"] = [float(v) for v in target_pos[0].detach().cpu().tolist()]
        if joint_pos_des is not None:
            result["joint_positions_desired"] = [float(v) for v in joint_pos_des[0].detach().cpu().tolist()]
        if joint_pos is not None:
            result["joint_positions_current"] = [float(v) for v in joint_pos[0].detach().cpu().tolist()]
        self.last_tidybot_arm_result = result
        return result

    def _validate_tidybot_ik_result(self, frame, target_pos, target_quat, ee_pos, joint_pos, joint_pos_des):
        if not torch.isfinite(target_pos).all() or not torch.isfinite(target_quat).all():
            return self._tidybot_rejected_arm_result("non_finite_target_pose", frame, target_pos, joint_pos_des, joint_pos)
        if not torch.isfinite(joint_pos_des).all():
            return self._tidybot_rejected_arm_result("non_finite_joint_solution", frame, target_pos, joint_pos_des, joint_pos)

        target_delta = torch.linalg.norm(target_pos - ee_pos, dim=1)
        if self.arm_ik_max_target_delta > 0 and torch.any(target_delta > self.arm_ik_max_target_delta):
            result = self._tidybot_rejected_arm_result("target_delta_too_large", frame, target_pos, joint_pos_des, joint_pos)
            result["target_delta"] = float(target_delta.max().detach().cpu())
            result["max_target_delta"] = self.arm_ik_max_target_delta
            return result

        robot = self.scene["robot"]
        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is not None:
            joint_limits = limits[:, self._tidybot_ik_joint_ids, :]
            finite_limits = torch.isfinite(joint_limits).all(dim=-1)
            valid_limits = finite_limits & (joint_limits[..., 0] <= joint_limits[..., 1])
            if valid_limits.any():
                margin = self.arm_ik_joint_limit_margin
                lower = joint_limits[..., 0] + margin
                upper = joint_limits[..., 1] - margin
                outside = valid_limits & ((joint_pos_des < lower) | (joint_pos_des > upper))
                if outside.any():
                    result = self._tidybot_rejected_arm_result(
                        "joint_limit_violation",
                        frame,
                        target_pos,
                        joint_pos_des,
                        joint_pos,
                    )
                    result["joint_limit_margin"] = margin
                    return result
        return None

    def _limit_tidybot_ik_joint_step(self, joint_pos, joint_pos_des):
        if self.arm_ik_max_joint_step <= 0:
            return joint_pos_des, None
        joint_delta = joint_pos_des - joint_pos
        max_joint_delta = float(torch.abs(joint_delta).max().detach().cpu())
        if max_joint_delta <= self.arm_ik_max_joint_step:
            return joint_pos_des, None
        clipped_delta = torch.clamp(joint_delta, -self.arm_ik_max_joint_step, self.arm_ik_max_joint_step)
        clipped_joint_pos = joint_pos + clipped_delta
        return clipped_joint_pos, {
            "joint_step_clipped": True,
            "max_joint_delta_raw": max_joint_delta,
            "max_joint_step": self.arm_ik_max_joint_step,
        }

    def _apply_tidybot_ik_pose_command(self, command):
        pose_value = None
        for key in ("arm_target_pose", "target_pose", "ee_pose"):
            if key in command:
                pose_value = command[key]
                break
        if pose_value is None:
            return None
        self._ensure_tidybot_ik_controller()
        target_pose, frame = self._parse_tidybot_pose_command(pose_value)
        ee_pos, ee_quat = self._body_pose_in_root(self._tidybot_ik_body_idx)
        target_pos, target_quat = self._target_pose_in_root(target_pose, frame)
        ik_command = torch.cat((target_pos, target_quat), dim=1)
        self._tidybot_ik_controller.set_command(ik_command, ee_pos, ee_quat)
        joint_pos = self.scene["robot"].data.joint_pos[:, self._tidybot_ik_joint_ids]
        jacobian = self._compute_tidybot_ik_jacobian()
        joint_pos_des = self._tidybot_ik_controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
        joint_pos_des, joint_step_info = self._limit_tidybot_ik_joint_step(joint_pos, joint_pos_des)
        rejected_result = self._validate_tidybot_ik_result(
            frame,
            target_pos,
            target_quat,
            ee_pos,
            joint_pos,
            joint_pos_des,
        )
        if rejected_result is not None:
            self._hold_tidybot_arm_pose()
            return rejected_result
        self._write_tidybot_hold_pose(self._tidybot_ik_joint_ids, joint_pos_des)
        result = {
            "mode": "ik_pose",
            "body_name": self.arm_ik_body_name,
            "frame": frame,
            "joint_names": list(self.arm_joint_names),
            "joint_positions": [float(v) for v in joint_pos_des[0].detach().cpu().tolist()],
            "target_pose_root_wxyz": [float(v) for v in ik_command[0].detach().cpu().tolist()],
        }
        if joint_step_info is not None:
            result.update(joint_step_info)
        self.last_tidybot_arm_result = result
        return result

    def _apply_tidybot_arm_command(self, command):
        if self.robot_name != "tidybot" or not isinstance(command, dict):
            return None
        ik_result = self._apply_tidybot_ik_pose_command(command)
        if ik_result is not None:
            if ik_result.get("rejected", False):
                return ik_result
            gripper_value = command.get("gripper", command.get("gripper_position", None))
            if gripper_value is not None:
                gripper_result = self._apply_tidybot_arm_command({"gripper": gripper_value})
                ik_result["gripper"] = gripper_result
                self.last_tidybot_arm_result = ik_result
            return ik_result
        arm_value = None
        for key in ("arm_joint_positions", "arm_joints", "joint_positions"):
            if key in command:
                arm_value = command[key]
                break
        gripper_value = command.get("gripper", command.get("gripper_position", None))
        joint_names, joint_values = self._normalise_joint_positions(arm_value, self.arm_joint_names)
        if gripper_value is not None:
            gripper_names, gripper_values = self._normalise_joint_positions(
                [gripper_value] if np.isscalar(gripper_value) else gripper_value,
                self.gripper_joint_names,
            )
            joint_names.extend(gripper_names)
            joint_values.extend(gripper_values)
        if not joint_names:
            return None
        joint_ids, resolved_names = self._resolve_joint_ids(joint_names)
        robot = self.scene["robot"]
        joint_pos = torch.tensor([joint_values], device=robot.device, dtype=robot.data.joint_pos.dtype)
        remember_hold = any(name in self.arm_joint_names for name in resolved_names)
        self._write_tidybot_hold_pose(joint_ids, joint_pos, remember=remember_hold)
        result = {"joint_names": resolved_names, "joint_positions": [float(v) for v in joint_values]}
        self.last_tidybot_arm_result = result
        return result

    def _extract_base_transform_command(self, command):
        if not isinstance(command, dict):
            return command
        for key in ("base_transform", "transform", "pose", "target"):
            if key in command:
                return command[key]
        if "position" in command or "translation" in command or "matrix" in command:
            return command
        return None

    def _navmesh_settings_for_episode(self):
        try:
            task_config_path = resolve_task_config_path(
                getattr(self.args, "tg_config_path", None),
                scene_id=self.episode.get("scene_id"),
                scene_type=self.episode.get("scene_type"),
            )
            task_config = load_config(task_config_path)
        except OSError as exc:
            print(f"[WARNING]: Failed to read task config for dynamic navmesh: {exc}")
            return {}
        return task_config.get("navmesh", {})

    def _navmesh_preprocess_config_for_episode(self):
        try:
            task_config_path = resolve_task_config_path(
                getattr(self.args, "tg_config_path", None),
                scene_id=self.episode.get("scene_id"),
                scene_type=self.episode.get("scene_type"),
            )
            task_config = load_config(task_config_path)
        except OSError as exc:
            print(f"[WARNING]: Failed to read task config for dynamic navmesh preprocess: {exc}")
            return {}
        return task_config.get("navmesh_preprocess", {})

    def _rebuild_navmesh_for_current_stage(self):
        selected_paths = ["/World/ground/terrain"]
        exclude_paths = self.episode.get("navmesh_exclude", [])
        scene_type = self.episode.get("scene_type", None)
        preprocess_config = self._navmesh_preprocess_config_for_episode()
        navmesh_interface = NavmeshInterface(up_axis="Z", preprocess_config=preprocess_config)
        settings = self._navmesh_settings_for_episode()
        if settings:
            navmesh_interface.update_settings(settings)
        print(f"[ENV] Rebuilding navmesh from loaded stage for scene_id={self.episode.get('scene_id')}")
        if scene_type == "innout":
            navmesh_interface.build_navmesh_innout(selected_paths, exclude_paths, self.scene.stage)
        else:
            navmesh_interface.setup_navmesh(
                selected_paths,
                exclude_paths,
                self.scene.stage,
                scene_type=scene_type,
                preprocess_config=preprocess_config,
            )
            navmesh_interface.build_navmesh()
        if not navmesh_interface.built:
            print("[WARNING]: Dynamic navmesh rebuild failed")
            return None
        return navmesh_interface

    def _load_navmesh_for_current_scene(self):
        if self.episode.get("path") == "generator":
            if self.args.rebuild_dynamic_navmesh:
                return self._rebuild_navmesh_for_current_stage()
            return load_navmesh_for_episode(self, self.episode)
        return load_navmesh_for_episode(self, self.episode)

    def _navmesh_surface_z_at_xy(
        self,
        xy,
        reference_z=None,
        allow_nearest_fallback=True,
    ):
        if self.navmesh_interface is None or not hasattr(self.navmesh_interface, "navmesh_v"):
            return None
        vertices = self.navmesh_interface.navmesh_v
        triangles = self.navmesh_interface.navmesh_t
        return navmesh_surface_z(
            vertices,
            triangles,
            xy,
            reference_z=reference_z,
            allow_nearest_fallback=allow_nearest_fallback,
        )

    def _navmesh_surface_pose(self, pose):
        surface_pose = np.asarray(pose, dtype=np.float32).copy()
        surface_z = self._navmesh_surface_z_at_xy(surface_pose[:2], reference_z=surface_pose[2])
        if surface_z is not None:
            surface_pose[2] = surface_z
        return surface_pose

    def _robot_root_z_for_navmesh_xy(self, xy, fallback_z):
        root_z_offset = self._tidybot_base_navmesh_z_offset
        if self.robot_name != "tidybot" or root_z_offset is None:
            root_z_offset = float(getattr(self.manager_env.cfg, "robot_spawn_height", 0.0))
        reference_surface_z = float(fallback_z) - float(root_z_offset)
        surface_z = self._navmesh_surface_z_at_xy(
            xy,
            reference_z=reference_surface_z,
        )
        if surface_z is None:
            return float(fallback_z)
        max_step_height = float(getattr(self.manager_env.cfg, "max_step_height", 0.05))
        if not navmesh_surface_is_reachable(surface_z, reference_surface_z, max_step_height):
            return float(fallback_z)
        return surface_z + float(root_z_offset)

    def _nearest_navmesh_robot_root_pose(self, pose):
        if self.navmesh_interface is None or not hasattr(self.navmesh_interface, "navmesh_v"):
            return pose.copy()
        vertices = self.navmesh_interface.navmesh_v
        distances = np.linalg.norm(vertices[:, :2] - pose[:2][None, :], axis=1)
        root_z_offset = self._tidybot_base_navmesh_z_offset or 0.0
        surface_z = float(pose[2]) - float(root_z_offset)
        nearest_idx = int(np.argmin(distances + np.abs(vertices[:, 2] - surface_z)))
        snapped = pose.copy()
        snapped[:2] = vertices[nearest_idx, :2]
        snapped[2] = self._robot_root_z_for_navmesh_xy(snapped[:2], pose[2])
        return snapped

    def _is_navmesh_point_clear(self, point):
        if self.navmesh_interface is None:
            return False
        clearance = max(
            float(getattr(self.manager_env.cfg, "navmesh_additional_clearance", 0.0)),
            0.0,
        )
        surface_point = np.asarray(point, dtype=np.float32).copy()
        surface_z = self._navmesh_surface_z_at_xy(
            surface_point[:2],
            reference_z=surface_point[2],
            allow_nearest_fallback=False,
        )
        if surface_z is None:
            return False
        surface_point[2] = surface_z
        try:
            wall_distance = extract_distance_to_wall(
                self.navmesh_interface.distance_to_wall(surface_point.tolist())
            )
            return np.isfinite(wall_distance) and wall_distance + 1e-4 >= clearance
        except Exception:
            return False

    def _is_navmesh_pose_valid(self, start, target):
        if self.navmesh_interface is None:
            return False
        clearance = max(
            float(getattr(self.manager_env.cfg, "navmesh_additional_clearance", 0.0)),
            0.0,
        )
        tolerance = float(
            getattr(self.manager_env.cfg, "kinematic_navmesh_endpoint_tolerance", 0.05)
        )
        start_surface = self._navmesh_surface_pose(start)
        target_surface = self._navmesh_surface_pose(target)
        max_step_height = float(getattr(self.manager_env.cfg, "max_step_height", 0.05))
        if not navmesh_surface_is_reachable(
            target_surface[2], start_surface[2], max_step_height
        ):
            return False
        path = self.navmesh_interface.find_paths(start_surface.tolist(), target_surface.tolist())
        if path is None or len(path) < 2:
            return False
        if np.linalg.norm(np.asarray(path[-1])[:2] - target_surface[:2]) > tolerance:
            return False
        try:
            return navmesh_segment_is_clear(
                self.navmesh_interface,
                start_surface,
                target_surface,
                clearance=clearance,
                resolution=float(getattr(self.manager_env.cfg, "navmesh_sample_resolution", 0.1)),
                max_step_height=max_step_height,
            )
        except Exception:
            return False

    def _slide_target_on_navmesh(self, start, target):
        resolution = float(getattr(self.manager_env.cfg, "navmesh_sample_resolution", 0.1))
        distance = float(np.linalg.norm(target[:2] - start[:2]))
        steps = max(int(np.ceil(distance / resolution)), 1)
        best = start.copy()
        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            candidate = start + (target - start) * alpha
            if self._is_navmesh_pose_valid(start, candidate):
                best = candidate
            else:
                break
        return best

    def _write_robot_pose(self, pose, remember_tidybot_base=True):
        robot = self.scene["robot"]
        root_state = robot.data.root_state_w.clone()
        root_state[:, 0:3] = torch.tensor(pose[:3], device=robot.device, dtype=root_state.dtype)
        root_state[:, 3:7] = torch.tensor([pose[6], pose[3], pose[4], pose[5]], device=robot.device, dtype=root_state.dtype)
        root_state[:, 7:] = 0.0
        if hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root_state[:, :7])
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(root_state[:, 7:])
        else:
            robot.write_root_state_to_sim(root_state)
        robot.write_data_to_sim()
        if remember_tidybot_base and self.robot_name == "tidybot":
            self._remember_tidybot_base_pose(root_state[:, :7])

    def _empty_kinematic_action(self, arm_result=None):
        if self.robot_name == "tidybot":
            self._tidybot_hold_base_this_step = True
            self._hold_tidybot_base_pose()
        self.low_level_action = self._empty_env_action()
        return self.low_level_action

    def _kinematic_displacement_action(self, command):
        self._tidybot_hold_base_this_step = False
        arm_result = self._apply_tidybot_arm_command(command)
        base_command = self._extract_base_transform_command(command)
        if base_command is None:
            return self._empty_kinematic_action(arm_result)
        if torch.is_tensor(base_command):
            flat_command = base_command[0] if base_command.ndim == 2 else base_command
            if flat_command.numel() == 3:
                return self._empty_kinematic_action(arm_result)
            if flat_command.numel() == 7:
                is_position_unset = torch.allclose(flat_command[:3], torch.zeros_like(flat_command[:3]))
                is_identity_quat = torch.allclose(
                    flat_command[3:7],
                    torch.tensor([0.0, 0.0, 0.0, 1.0], device=flat_command.device, dtype=flat_command.dtype),
                )
                if is_position_unset and is_identity_quat:
                    return self._empty_kinematic_action(arm_result)
        target = self.parse_transform_command(base_command[0] if torch.is_tensor(base_command) and base_command.ndim == 2 else base_command)
        current_pos, current_quat = self.get_body_pose()
        target_np = target.detach().cpu().numpy().astype(np.float32)
        start = np.asarray(current_pos, dtype=np.float32)
        candidate = target_np[:3].copy()
        if self.robot_name == "drone":
            pose = np.concatenate([candidate, target_np[3:7]])
            self._write_robot_pose(pose)
            self.low_level_action = self._empty_env_action()
            self.last_kinematic_result = {
                "blocked": False,
                "target": candidate.tolist(),
                "accepted": candidate.tolist(),
            }
            return self.low_level_action
        if self.navmesh_interface is not None:
            candidate[2] = self._robot_root_z_for_navmesh_xy(candidate[:2], start[2])
            start_on_navmesh = self._is_navmesh_point_clear(start)
            if not start_on_navmesh:
                start = self._nearest_navmesh_robot_root_pose(start)
        elif candidate[2] == 0.0:
            candidate[2] = start[2]
        accepted = candidate
        blocked = False
        if self.navmesh_interface is not None:
            if not self._is_navmesh_pose_valid(start, candidate):
                blocked = True
                if self.collision_response == "block_or_slide":
                    accepted = self._slide_target_on_navmesh(start, candidate)
                else:
                    accepted = start
            accepted[2] = self._robot_root_z_for_navmesh_xy(accepted[:2], accepted[2])
        pose = np.concatenate([accepted, target_np[3:7]])
        self._write_robot_pose(pose)
        if self.robot_name == "tidybot":
            self._tidybot_hold_base_this_step = True
        self.low_level_action = self._empty_env_action()
        self.last_kinematic_result = {"blocked": blocked, "target": candidate.tolist(), "accepted": accepted.tolist()}
        if arm_result is not None:
            self.last_kinematic_result["arm"] = arm_result
        return self.low_level_action

    def _command_to_env_action(self, command) -> torch.Tensor:
        if self.control_mode == "kinematic_displacement" and isinstance(command, dict):
            return self._kinematic_displacement_action(command)
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=self.manager_env.device)
        command = command.to(device=self.manager_env.device, dtype=torch.float32)
        if command.ndim == 1:
            command = command.unsqueeze(0).repeat(self.num_envs, 1)

        if not self.direct_control:
            self.update_command(command)
            env_action = self.low_level_policy(self.obs)
            self.low_level_action = env_action
            return env_action

        if self.control_mode == "physics_velocity":
            if self.robot_name != "tidybot" or command.shape[-1] != 3:
                raise ValueError(
                    "physics_velocity requires TidyBot commands shaped (num_envs, 3)"
                )
            robot = self.scene["robot"]
            base_joint_ids = self._resolve_tidybot_base_joint_ids()
            if len(base_joint_ids) != 3:
                raise RuntimeError(
                    "physics_velocity requires joint_x, joint_y, and joint_th"
                )
            robot.set_joint_velocity_target(command, joint_ids=base_joint_ids)
            robot.velocity_command = command
            self._tidybot_hold_base_this_step = False
            self.low_level_action = self._empty_env_action()
            return self.low_level_action

        if self.control_mode == "kinematic_displacement":
            return self._kinematic_displacement_action(command)

        if self.robot_name == "turtlebot":
            max_cmd_lin_vel = getattr(self.manager_env.cfg, "max_cmd_lin_vel", None)
            max_cmd_ang_vel = getattr(self.manager_env.cfg, "max_cmd_ang_vel", None)
            if max_cmd_lin_vel is not None:
                command[:, 0] = torch.clamp(command[:, 0], min=-max_cmd_lin_vel, max=max_cmd_lin_vel)
            if max_cmd_ang_vel is not None:
                command[:, 2] = torch.clamp(command[:, 2], min=-max_cmd_ang_vel, max=max_cmd_ang_vel)
            wheel_radius = getattr(self.manager_env.cfg, "wheel_radius", 0.033)
            wheel_base = getattr(self.manager_env.cfg, "wheel_base", 0.16)
            wheel_command_signs = getattr(self.manager_env.cfg, "wheel_command_signs", (1.0, 1.0))
            left_wheel = (command[:, 0] - 0.5 * wheel_base * command[:, 2]) / wheel_radius
            right_wheel = (command[:, 0] + 0.5 * wheel_base * command[:, 2]) / wheel_radius
            env_action = torch.stack(
                [left_wheel * wheel_command_signs[0], right_wheel * wheel_command_signs[1]],
                dim=-1,
            )
            self.low_level_action = env_action
            return env_action

        raise ValueError(f"Unsupported direct-control robot: {self.robot_name}")

    def step(self, action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in the environment.

        Args:
            action: The action of high-level planner, which should be velocity command to the low-level policy.

        Returns:
            obs: The observation of the high-level planner.
            reward: The reward of the environment.
            done: Whether the episode is done.
            info: Additional information of the environment.
        
        """

        profile_total_start = time.perf_counter()

        profile_start = time.perf_counter()
        carb.profiler.begin(1, "NavVerse.wrapper/command_to_action")
        try:
            env_action = self._command_to_env_action(action)
        finally:
            carb.profiler.end(1)
        self._profile_record("wrapper/command_to_action", time.perf_counter() - profile_start)

        profile_start = time.perf_counter()
        carb.profiler.begin(1, "NavVerse.wrapper/low_level_env_step")
        try:
            obs, reward, done, info = self._step_low_level_env(env_action)
        finally:
            carb.profiler.end(1)
        self._profile_record("wrapper/low_level_env_step", time.perf_counter() - profile_start)

        if self.robot_name == "tidybot":
            profile_start = time.perf_counter()
            carb.profiler.begin(1, "NavVerse.wrapper/tidybot_hold_after_step")
            try:
                if self._tidybot_hold_base_this_step:
                    self._hold_tidybot_base_pose()
                self._hold_tidybot_arm_pose()
                self._tidybot_hold_base_this_step = False
            finally:
                carb.profiler.end(1)
            self._profile_record("wrapper/tidybot_hold_after_step", time.perf_counter() - profile_start)
        self.obs = obs
        high_level_obs = self._get_high_level_obs(obs)
        self.env_step += 1

        # update measures
        profile_start = time.perf_counter()
        carb.profiler.begin(1, "NavVerse.wrapper/measure_update")
        try:
            stop_terminal = bool(self.is_stop_called.any().item())
            if not stop_terminal:
                self.measure_manager.update_measures()
            measurements = self.measure_manager.get_measurements()
            position_xy = None if stop_terminal else self.get_body_pose()[0][:2]
            measurements.update(self.terrain_eval.measurements(position_xy))
        finally:
            carb.profiler.end(1)
        self._profile_record("wrapper/measure_update", time.perf_counter() - profile_start)
        info["measurements"] = measurements
        if self.last_tidybot_arm_result is not None:
            info["tidybot_arm"] = self.last_tidybot_arm_result
        if hasattr(self, "last_kinematic_result"):
            info["kinematic"] = self.last_kinematic_result
        # update terminations
        profile_start = time.perf_counter()
        carb.profiler.begin(1, "NavVerse.wrapper/termination_update")
        try:
            if not self.args.disable_termination:
                self.reset_buf = self.termination_manager.compute()
                termination_terms = self.termination_manager.get_active_iterable_terms(0)
                done = done.any() or self.reset_buf.any()
                if self.reset_buf.any():
                    termination_reason = max(termination_terms, key=lambda x:x[1])[0]
                    info["terminations"] = {"termination_reason": termination_reason}
                    self._refresh_stop_terminal_measurements(info)
                    # print termination reason if it is not the same as the last time
                    print_termination_reason = (self.episode['episode_label'], termination_reason)
                    if not hasattr(self, 'print_termination_reason') or self.print_termination_reason != print_termination_reason:
                        self.print_termination_reason = print_termination_reason
                        print(f"Episode {self.episode['episode_label']} terminated due to {termination_reason}")
                        print("Measures: ", ", ".join([f"{k}={v:.2f}" for k, v in info["measurements"].items()]))
                else:
                    if "terminations" in info:
                        del info["terminations"]
            else:
                # termination disabled, only respond to stop_called
                done = done.any() or self.is_stop_called.any()
                if self.is_stop_called.any():
                    info["terminations"] = {"termination_reason": "stop_called"}
                    self._refresh_stop_terminal_measurements(info)
                    print(f"Episode {self.episode['episode_label']} terminated due to stop_called")
                else:
                    if "terminations" in info:
                        del info["terminations"]
        finally:
            carb.profiler.end(1)
        self._profile_record("wrapper/termination_update", time.perf_counter() - profile_start)

        self.profile_step_count += 1
        self._profile_record("wrapper/total", time.perf_counter() - profile_total_start)
        self._profile_summary()
        return high_level_obs, reward, done, info

    def _refresh_stop_terminal_measurements(self, info):
        terminations = info.get("terminations", {})
        if terminations.get("termination_reason") != "stop_called":
            return

        for measure_name in ("success", "soft_success"):
            measure = self.measure_manager.measures.get(measure_name)
            if measure is not None:
                measure.update_metric()
        for measure_name in ("spl", "soft_spl"):
            measure = self.measure_manager.measures.get(measure_name)
            if measure is not None:
                measure.update_metric(update_path=False)
        measurements = self.measure_manager.get_measurements()
        measurements.update(self.terrain_eval.measurements())
        info["measurements"] = measurements
    
    def reset_idx(self, reset_env_ids):
        reset_env_ids = torch.tensor(reset_env_ids, device=self.device, dtype=torch.int32)
        self.reset_termination_states(reset_env_ids)
        self.reset_buf = self.termination_manager.compute()
        for env_id in reset_env_ids:
            self.set_stop_called(env_id, False)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.manager_env.recorder_manager.record_pre_reset(reset_env_ids)
            self.manager_env._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.manager_env.sim.has_rtx_sensors():
                num_rerenders_on_reset = 1
                for _ in range(num_rerenders_on_reset):
                    self.manager_env.sim.render()
            # trigger recorder terms for post-reset calls
            self.manager_env.recorder_manager.record_post_reset(reset_env_ids)

    def set_stop_called(self, robot_index: int, is_stop_called: bool) -> None:
        """Set the stop called flag."""
        self.is_stop_called[robot_index] = is_stop_called
    
    def close(self) -> None:
        self.env.close()
