from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from navverse.robots.base_env_cfg import BaseEnvCfg, CameraObservationsCfg, EmptyCfg


TURTLEBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "wheel_left_joint": 0.0,
            "wheel_right_joint": 0.0,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.3,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
        ),
    },
)


@configclass
class TurtlebotActionsCfg:
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_left_joint", "wheel_right_joint"],
        scale=1.0,
        preserve_order=True,
    )


@configclass
class TurtlebotCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 4.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.5, 1.5),
        ),
    )


@configclass
class TurtlebotObservationsCfg:
    camera: CameraObservationsCfg = CameraObservationsCfg()


@configclass
class TurtlebotRoughEnvCfg(BaseEnvCfg):
    observations: TurtlebotObservationsCfg = TurtlebotObservationsCfg()
    actions: TurtlebotActionsCfg = TurtlebotActionsCfg()
    commands: TurtlebotCommandsCfg = TurtlebotCommandsCfg()
    rewards: EmptyCfg = EmptyCfg()
    terminations: EmptyCfg = EmptyCfg()
    events: EmptyCfg = EmptyCfg()
    viewer = ViewerCfg(eye=(6.0, 6.0, 2.0), origin_type="world", env_index=0, asset_name="robot")
    usd_path: Optional[str] = None

    robot_spawn_height = 0.05
    controlled_robot_spawn_height = 0.05
    use_episode_robot_spawn_height = False
    robot_warmup_steps = 0
    robot_height = 0.18
    robot_width = 0.35
    wheel_radius = 0.025
    wheel_base = 0.16
    wheel_command_signs = (1.0, 1.0)
    max_cmd_lin_vel = 2.0
    max_cmd_ang_vel = 2.0
    low_level_type = "direct"

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = TURTLEBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class TurtlebotRoughEnvCfg_PLAY(TurtlebotRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        self.episode_length_s = 1000000.0
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.scene.pov_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/Camera",
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
                pos=(0.08, 0.0, 0.22),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )
        self.scene.third_person_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/ThirdPersonCamera",
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
