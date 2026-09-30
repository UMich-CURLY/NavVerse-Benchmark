from pathlib import Path
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from navverse.robots.base_env_cfg import BaseEnvCfg, CameraObservationsCfg, EmptyCfg


_TIDYBOT_URDF_PATH = Path(__file__).resolve().parent / "assets/tidybot_description/urdf/tidybot_isaac.urdf"
_TIDYBOT_HABITAT_PE_URDF_PATH = (
    Path(__file__).resolve().parent
    / "assets/tidybot_description/urdf/tidybot_habitat_pe.urdf"
)
_TIDYBOT_ARM_HOME_JOINT_POS = {
    "joint_1": 0.0,
    "joint_2": 0.55,
    "joint_3": 0.0,
    "joint_4": -1.65,
    "joint_5": 0.0,
    "joint_6": 1.10,
    "joint_7": 0.0,
    "finger_joint": 0.0,
}

TIDYBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_TIDYBOT_URDF_PATH),
        fix_base=False,
        merge_fixed_joints=True,
        collision_from_visuals=False,
        joint_drive=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.625), joint_pos=_TIDYBOT_ARM_HOME_JOINT_POS),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["joint_x", "joint_y", "joint_th"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
            stiffness=0.0,
            damping=1e5,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            effort_limit_sim={
                "joint_[1-4]": 87.0,
                "joint_[5-7]": 20.0,
            },
            velocity_limit_sim=2.0,
            stiffness=800.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "left_outer_finger_joint",
                "left_inner_finger_joint",
                "left_inner_finger_knuckle_joint",
                "right_outer_knuckle_joint",
                "right_outer_finger_joint",
                "right_inner_finger_joint",
                "right_inner_finger_knuckle_joint",
            ],
            effort_limit_sim=40.0,
            velocity_limit_sim=5.0,
            stiffness=500.0,
            damping=50.0,
        ),
    },
)

TIDYBOT_HABITAT_PE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_TIDYBOT_HABITAT_PE_URDF_PATH),
        fix_base=True,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        collision_from_visuals=False,
        joint_drive=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=True,
            max_linear_velocity=3.0,
            max_angular_velocity=360.0,
            max_depenetration_velocity=2.0,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.085),
        joint_pos={"joint_x": 0.0, "joint_y": 0.0, "joint_th": 0.0},
        joint_vel={"joint_x": 0.0, "joint_y": 0.0, "joint_th": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["joint_x", "joint_y", "joint_th"],
            effort_limit_sim={
                "joint_x": 1000.0,
                "joint_y": 1000.0,
                "joint_th": 500.0,
            },
            velocity_limit_sim={
                "joint_x": 5.0,
                "joint_y": 5.0,
                "joint_th": 10.0,
            },
            stiffness=0.0,
            damping=1.0e5,
        ),
    },
)


@configclass
class TidybotObservationsCfg:
    camera: CameraObservationsCfg = CameraObservationsCfg()


@configclass
class TidybotRoughEnvCfg(BaseEnvCfg):
    observations: TidybotObservationsCfg = TidybotObservationsCfg()
    actions: EmptyCfg = EmptyCfg()
    commands: EmptyCfg = EmptyCfg()
    rewards: EmptyCfg = EmptyCfg()
    terminations: EmptyCfg = EmptyCfg()
    events: EmptyCfg = EmptyCfg()
    viewer = ViewerCfg(eye=(6.0, 6.0, 2.0), origin_type="world", env_index=0, asset_name="robot")
    usd_path: Optional[str] = None

    control_mode = "kinematic_displacement"
    collision_proxy = "cylinder"
    radius = 0.32
    height = 1.25
    max_step_height = 0.05
    max_slope_angle = 26.0
    dynamics = "none"
    collision_response = "block_or_slide"
    arm_control_mode = "joint_position"
    base_joint_names = ("joint_x", "joint_y", "joint_th")
    arm_ik_body_name = "bracelet_link"
    arm_ik_planning_body_name = "arm_base_link"
    arm_ik_planning_frame_pos = (0.12, 0.0, 0.374775)
    arm_ik_planning_frame_rot = (1.0, 0.0, 0.0, 0.0)
    arm_ik_method = "dls"
    arm_joint_names = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7")
    gripper_joint_names = ("finger_joint",)
    arm_home_joint_positions = (0.0, 0.55, 0.0, -1.65, 0.0, 1.10, 0.0)
    gripper_home_joint_positions = (0.0,)
    arm_ik_max_target_delta = 1.0
    arm_ik_max_joint_step = 0.25
    arm_ik_joint_limit_margin = 0.02
    robot_spawn_height = 0.625
    robot_warmup_steps = 0
    robot_height = 1.25
    robot_width = 0.64
    low_level_type = "direct"
    navmesh_endpoint_tolerance = 0.40
    kinematic_navmesh_endpoint_tolerance = 0.05
    navmesh_sample_resolution = 0.10

    def __post_init__(self):
        super().__post_init__()
        self.scene.contact_forces = None
        self.scene.robot = TIDYBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class TidybotRoughEnvCfg_PLAY(TidybotRoughEnvCfg):
    def __post_init__(self) -> None:
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
                pos=(0.32, 0.0, 1.05),
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


@configclass
class TidybotHabitatPERoughEnvCfg_PLAY(TidybotRoughEnvCfg_PLAY):
    """Base-only TidyBot geometry and camera used by Habitat PE."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = TIDYBOT_HABITAT_PE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.0,
            history_length=self.decimation,
            track_air_time=False,
        )
        self.control_mode = "physics_velocity"
        self.dynamics = "physx"
        self.radius = 0.28
        self.height = 1.0
        self.robot_height = 1.0
        self.robot_width = 0.56
        self.navmesh_additional_clearance = 0.0
        self.robot_spawn_height = 0.085
        self.controlled_robot_spawn_height = 0.085
        self.use_episode_robot_spawn_height = False
        self.arm_control_mode = None
        self.base_joint_names = ("joint_x", "joint_y", "joint_th")
        self.ignore_bad_orientation = True
        self.arm_joint_names = ()
        self.gripper_joint_names = ()
        self.arm_home_joint_positions = ()
        self.gripper_home_joint_positions = ()
        # Parent cameras to the articulated body that moves in the world plane.
        self.scene.pov_camera.prim_path = "{ENV_REGEX_NS}/Robot/base/Camera"
        self.scene.pov_camera.offset = CameraCfg.OffsetCfg(
            pos=(-0.1, 0.0, 1.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        )
        self.scene.third_person_camera.prim_path = (
            "{ENV_REGEX_NS}/Robot/base/ThirdPersonCamera"
        )
