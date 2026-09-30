import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=14,
    num_cols=14,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
        "down_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.25),
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
        ),
        "up_stairs": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.25),
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
        ),
    },
)
@configclass
class BaseTerminationsCfg:
    pass


def robot_pose(
    env,
    asset_cfg=SceneEntityCfg("robot"),
):
    robot = env.scene[asset_cfg.name]
    if getattr(env.cfg, "control_mode", None) == "physics_velocity":
        body_ids, body_names = robot.find_bodies("base")
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one physics base body, found {body_names}")
        pose_state = robot.data.body_state_w[:, int(body_ids[0]), :7]
    else:
        pose_state = robot.data.root_state_w[:, :7]
    pos = pose_state[:, 0:3]
    quat = pose_state[:, 3:7]
    quat = torch.concat([quat[:, 1:], quat[:, :1]], dim=1)
    return torch.concat([pos, quat], dim=1)


def camera_tf(
    env,
    sensor_cfg=SceneEntityCfg("tiled_camera"),
):
    sensor = env.scene.sensors[sensor_cfg.name]
    pos = sensor.cfg.offset.pos
    quat = sensor.cfg.offset.rot
    quat = [*quat[1:], *quat[:1]]
    num_envs = sensor.data.pos_w.shape[0]
    return torch.tensor([[*pos, *quat]] * num_envs, device=sensor.device)


def camera_pose(
    env,
    sensor_cfg=SceneEntityCfg("tiled_camera"),
):
    sensor = env.scene.sensors[sensor_cfg.name]
    pos = sensor.data.pos_w.detach().cpu()
    quat = sensor.data.quat_w_world.detach().cpu()
    quat = torch.concat([quat[:, 1:], quat[:, :1]], dim=1)
    return torch.concat([pos, quat], dim=1)


@configclass
class CameraObservationsCfg(ObsGroup):
    body_pose = ObsTerm(func=robot_pose, params={"asset_cfg": SceneEntityCfg("robot")})
    pov_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("pov_camera"), "data_type": "rgb", "normalize": False})
    pov_depth = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("pov_camera"), "data_type": "distance_to_image_plane"})
    pov_tf = ObsTerm(func=camera_tf, params={"sensor_cfg": SceneEntityCfg("pov_camera")})
    pov_pose = ObsTerm(func=camera_pose, params={"sensor_cfg": SceneEntityCfg("pov_camera")})
    third_person_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("third_person_camera"), "data_type": "rgb", "normalize": False})
    third_person_depth = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("third_person_camera"), "data_type": "distance_to_image_plane"})
    third_person_tf = ObsTerm(func=camera_tf, params={"sensor_cfg": SceneEntityCfg("third_person_camera")})
    third_person_pose = ObsTerm(func=camera_pose, params={"sensor_cfg": SceneEntityCfg("third_person_camera")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class EmptyCfg:
    pass

@configclass
class BaseEnvCfg(LocomotionVelocityRoughEnvCfg):

    terminations: BaseTerminationsCfg = BaseTerminationsCfg()
    robot_spawn_height = 0.8
    use_episode_robot_spawn_height = True
    robot_warmup_steps = 0
    robot_height = 0.61
    robot_width = 0.5
    low_level_type = "direct"

    def load_generator(self):
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=self._physics_material,
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

    def load_usd(self, usd_path: str):
        usd_path = str(usd_path)
        self.usd_path = usd_path
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=usd_path,
            physics_material=self._physics_material,
            env_spacing=self.scene.env_spacing,
        )

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 10
        self.episode_length_s = 10000.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.sim.render.dlss_mode = 2
        self.sim.render.enable_reflections = True
        self.sim.render.enable_direct_lighting = True
        self.sim.render.enable_translucency = True
        self.sim.render.enable_shadows = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_dl_denoiser = True

        self.scene.contact_forces.update_period = self.sim.dt

        self._physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )

        self.load_generator()
        self.scene.height_scanner = None
