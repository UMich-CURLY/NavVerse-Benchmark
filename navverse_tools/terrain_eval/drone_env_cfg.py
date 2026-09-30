"""Fixed-height collision-free drone used by terrain evaluation."""

from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

from navverse.robots.base_env_cfg import BaseEnvCfg, CameraObservationsCfg, EmptyCfg


DRONE_CFG = RigidObjectCfg(
    spawn=sim_utils.CylinderCfg(
        radius=0.25,
        height=0.12,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=True,
            max_linear_velocity=2.5,
            max_angular_velocity=4.0,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.85)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
)


@configclass
class DroneObservationsCfg:
    camera: CameraObservationsCfg = CameraObservationsCfg()


@configclass
class DroneEnvCfg(BaseEnvCfg):
    observations: DroneObservationsCfg = DroneObservationsCfg()
    actions: EmptyCfg = EmptyCfg()
    commands: EmptyCfg = EmptyCfg()
    rewards: EmptyCfg = EmptyCfg()
    terminations: EmptyCfg = EmptyCfg()
    events: EmptyCfg = EmptyCfg()
    viewer = ViewerCfg(eye=(6.0, 6.0, 3.0), origin_type="world", env_index=0, asset_name="robot")
    usd_path: Optional[str] = None

    control_mode = "kinematic_displacement"
    collision_response = "none"
    dynamics = "none"
    robot_spawn_height = 2.0
    controlled_robot_spawn_height = 2.0
    robot_warmup_steps = 0
    robot_height = 0.12
    robot_width = 0.50
    low_level_type = "direct"
    max_horizontal_speed = 2.5
    max_yaw_speed = 4.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.contact_forces = None
        self.scene.robot = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class DroneEnvCfg_PLAY(DroneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1000000.0
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.pov_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Camera",
            update_period=0.1,
            update_latest_camera_pose=True,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1000.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.25, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )
        self.scene.third_person_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ThirdPersonCamera",
            update_period=0.1,
            update_latest_camera_pose=True,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1000.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-1.5, 0.0, 0.8),
                rot=(0.5609855, 0.4304593, -0.4304593, -0.5609855),
                convention="opengl",
            ),
        )

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
