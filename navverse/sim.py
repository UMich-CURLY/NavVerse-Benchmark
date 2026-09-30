import torch
import numpy as np
import cv2
import json
import os
import time
from threading import Thread, Event
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from navverse.utils.astar import BevAStarReferencePlanner
# Isaac Import
import carb
import omni
import isaacsim.core.utils.prims as prim_utils
try:
    from omni.kit.viewport.utility import get_viewport_from_window_name
except ModuleNotFoundError:
    get_viewport_from_window_name = None
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Local imports
import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
from navverse.path_following import KinematicPIDPoseFollower, WaypointFollower, calc_yaw
from navverse.timed_trajectory import control_decimation, trajectory_tick_index
from navverse.utils.socket_server import run_server, format_data
from navverse.utils.vis import visualize_curve, visualize_arrow
from navverse.episode import VLNEpisode, episode_instruction, episode_target_type, load_episode_set
from navverse.config import load_config, resolve_scene_spec, resolve_task_config_path
from navverse.vln_args import load_runtime_config
from navverse.vln_env_wrapper import VLNEnvWrapper, init_env_cfg
from navverse.utils.foxglove_utils import FoxgloveVisualizer
from pathlib import Path
from omegaconf import OmegaConf
from navverse.termination_cfg import DEFAULT_EPISODE_TIMEOUT_S
from navverse.robots.base_env_cfg import EmptyCfg
from navverse.robots.spot.spot_rough_env_cfg import SpotRoughEnvCfg_PLAY
from navverse.robots.turtlebot_rough_env_cfg import TurtlebotRoughEnvCfg_PLAY
from navverse.robots.tidybot_rough_env_cfg import (
    TidybotHabitatPERoughEnvCfg_PLAY,
    TidybotRoughEnvCfg_PLAY,
)
from navverse.robots.cylinder_rough_env_cfg import CylinderRobotEnvCfg_PLAY
from navverse_tools.terrain_eval.drone_env_cfg import DroneEnvCfg_PLAY
from threading import Lock
from navverse.measures import add_measurement

class VLNSim:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.robot_name = args.robot_name
        self._init_runtime_state()
        self._init_episode_state()
        self._init_control_state(args)
        self._init_viewport_state()
        self._init_server_state(args)

    def _init_runtime_state(self):
        self.init = False
        self.env = None
        self.manager_env = None
        self.info = None
        self.obs = None
        self.reward = None
        self.done = None
        self.load_env_lock = Lock()
        self.sim_step_lock = Lock()
        self.on_demand_render_lock = Lock()
        self.on_demand_render_pending = False
        self.on_demand_render_event = None
        self.on_demand_render_error = None
        self.obs_index = 0
        self.cam_frame = -1
        self.closed = False
        self.profile_stats = {}
        self.profile_step_count = 0
        self.telemetry_start_wall_time = time.time()
        self.telemetry_start_step = 0
        self.astar_visualization_index = 0
        self.astar_capture_debug_index = 0
        self.last_astar_world_path = None
        self.last_astar_plan_info = {}
        self.last_astar_waypoints = None
        self._runtime_config_cache = {}
        self._astar_config_cache = {}
        self._active_astar_config = None
        self.runtime_config_path = None
        camera_names = os.environ.get(
            "NAVVERSE_ON_DEMAND_CAMERAS",
            "pov_camera,third_person_camera",
        )
        self.on_demand_camera_names = tuple(
            name.strip() for name in camera_names.split(",") if name.strip()
        )

    def _init_episode_state(self):
        self.reset_flag = True
        self.sim_state = "init" # init, loading, running, terminated
        self.next_episode = None # this is updated when starting a new episode
        self.current_episode = None # this is updated after an episode is loaded
        self.episode_list = VLNEpisode.from_json_folder(self.args.episode_folder)
        self.episode_label_list = [x.episode_label for x in self.episode_list]

        # Shared buffers for server callbacks
        self._latest_data = None
        self.reset_server_cache()

    def _init_control_state(self, args):
        self.waypoints = []
        if self.robot_name == "turtlebot":
            follower_max_vel = [0.8, 0.0, 2.0]
            follower_min_vel = [0.3, 0.1, 0.3]
            follower_kp = [2.0, 0.5, 1.5]
            arrive_dist = 0.1
            arrive_yaw = np.pi / 180.0 * 15.0
            term_dist = 0.05
            term_yaw = np.pi / 180.0 * 5.0
            lookahead_distance = 0.5
            prefer_forward = True
        elif self.robot_name == "spot":
            follower_max_vel = [2.5, 1.5, 2.5]
            follower_min_vel = [0.35, 0.30, 0.45]
            follower_kp = [5.0, 4.0, 5.0]
            arrive_dist = 0.25
            arrive_yaw = np.pi / 2.0
            term_dist = 0.45
            term_yaw = np.pi / 2.0
            lookahead_distance = 0.8
            prefer_forward = True
        else:
            follower_max_vel = [2.5, 0.6, 2.5]
            follower_min_vel = [0.35, 0.15, 0.45]
            follower_kp = [5.0, 1.5, 5.0]
            arrive_dist = 0.1
            arrive_yaw = np.pi / 180.0 * 15.0
            term_dist = 0.05
            term_yaw = np.pi / 180.0 * 5.0
            lookahead_distance = 0.5
            prefer_forward = False
        self.waypoint_follower = WaypointFollower(
            device=self.device,
            max_vel=follower_max_vel,
            min_vel=follower_min_vel,
            kp=follower_kp,
            arrive_dist=arrive_dist,
            arrive_yaw=arrive_yaw,
            term_dist=term_dist,
            term_yaw=term_yaw,
            lookahead_distance=lookahead_distance,
            prefer_forward=prefer_forward,
        )
        self.timed_trajectory_follower = WaypointFollower(
            device=self.device,
            max_vel=follower_max_vel,
            min_vel=[0.0, 0.0, 0.0],
            kp=follower_kp,
            arrive_dist=0.0,
            arrive_yaw=0.0,
            term_dist=0.0,
            term_yaw=0.0,
            lookahead_distance=0.0,
            prefer_forward=False,
        )
        self.tidybot_timed_trajectory_follower = KinematicPIDPoseFollower()
        self.timed_trajectory = None
        self.last_timed_trajectory_status = None
        self._default_waypoint_follower_config = {
            "max_vel": np.asarray(follower_max_vel, dtype=float),
            "min_vel": np.asarray(follower_min_vel, dtype=float),
            "kp": np.asarray(follower_kp, dtype=float),
            "arrive_dist": arrive_dist,
            "arrive_yaw": arrive_yaw,
            "term_dist": term_dist,
            "term_yaw": term_yaw,
            "lookahead_distance": lookahead_distance,
            "prefer_forward": prefer_forward,
            "lateral_deadband": 0.0,
            "lateral_turn_damping_yaw": 0.0,
        }
        self.waypoint_path_planner = None
        tidybot_uses_physics_velocity = (
            self.robot_name == "tidybot"
            and getattr(args, "tidybot_embodiment", "full") == "habitat_pe"
        )
        self.command_dim = (
            7
            if self.robot_name in ["tidybot", "cylinder", "drone"]
            and not tidybot_uses_physics_velocity
            else 3
        )
        self.commands = torch.zeros((args.num_envs, self.command_dim), device=self.device)
        if self.command_dim == 7:
            self.commands[:, 6] = 1.0
        self.commands_source = 'server'
        self.pending_transform_command = None
        self.transform_root_z_offset = 0.0
        self.waypoint_command_frames = 0
        self.waiting_for_first_action = False
        self.arm_command = None
        self.robot_index = 0
        self.keyboard_base_linear_speed = 0.5
        self.keyboard_base_yaw_speed = 0.8
        self.keyboard_arm_linear_speed = 0.50
        self.keyboard_arm_vertical_speed = 0.40
        self._tidybot_keyboard_base_pos = None
        self._tidybot_keyboard_base_yaw = None
        self._tidybot_keyboard_ee_pos = None
        self._tidybot_keyboard_ee_quat_wxyz = None
        self._tidybot_base_command_active = False
        self._tidybot_keyboard_base_keys = {"W", "A", "S", "D", "Q", "E"}
        self._tidybot_keyboard_arm_keys = {"I", "J", "K", "L", "U", "O"}
        self._tidybot_keyboard_keys = self._tidybot_keyboard_base_keys | self._tidybot_keyboard_arm_keys
        self._keyboard_control_keys = set()

    def _init_viewport_state(self):
        self.viewport = get_viewport_from_window_name("Viewport") if get_viewport_from_window_name is not None else None
        if self.args.headless:
            print("[SIM] Headless mode, disabling viewport updates")
            if self.viewport is not None:
                self.viewport.updates_enabled = False
        self.viewport_third_person = True
        self.callbacks = {}
        if self.args.on_demand_render:
            self.add_callback('pre_step', self.service_on_demand_render_request)
            self.add_callback('step_finished', self.service_on_demand_render_request)
        self.visualize_waypoints = False

    def _init_server_state(self, args):
        self.server_data_lock = Lock()
        self.server_thread = None
        self.server_stop_event = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        self._pressed_control_keys = set()
        if not args.disable_socket_server:
            self.start_server(host=args.host, port=args.port, server_name="BenchmarkServer")
        if args.foxglove_port>0:
            self.visualizer = FoxgloveVisualizer(host=args.host, port=args.foxglove_port)
        else:
            self.visualizer = None

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
        step_total = self.profile_stats.get("step/total", (0.0, 0))[0]
        render_total = self.profile_stats.get("render/total", (0.0, 0))[0]
        print(f"[PROFILE] steps={self.profile_step_count}, window_keys={len(self.profile_stats)}")
        for key, (total_s, count) in sorted(self.profile_stats.items()):
            denom = render_total if key.startswith("render/") and render_total > 0 else step_total
            pct = 100.0 * total_s / denom if denom > 0 else 0.0
            avg_ms = 1000.0 * total_s / max(count, 1)
            print(f"[PROFILE] {key}: total={total_s:.3f}s count={count} avg={avg_ms:.2f}ms pct={pct:.1f}%")
        self.profile_stats.clear()

    def reset_telemetry_baseline(self):
        self.telemetry_start_wall_time = time.time()
        if self.manager_env is None:
            self.telemetry_start_step = 0
        else:
            self.telemetry_start_step = int(getattr(self.manager_env, "common_step_counter", 0) or 0)
    
    def init_env(self):
        print("[SIM] Initializing Isaac Lab environment...")
        if self.next_episode is None:
            raise ValueError("Current episode must be set before initializing the environment")
        # isaac lab manager
        args = self.args
        robot_cfg = {
            "spot": {
                "env_cfg_cls": SpotRoughEnvCfg_PLAY,
                "task": "Isaac-Velocity-Rough-Spot-v0",
                "checkpoint": Path(__file__).resolve().parent / "robots" / "spot" / "model_dec7.pt",
            },
            "turtlebot": {
                "env_cfg_cls": TurtlebotRoughEnvCfg_PLAY,
                "task": None,
                "checkpoint": None,
            },
            "tidybot": {
                "env_cfg_cls": (
                    TidybotHabitatPERoughEnvCfg_PLAY
                    if getattr(args, "tidybot_embodiment", "full") == "habitat_pe"
                    else TidybotRoughEnvCfg_PLAY
                ),
                "task": None,
                "checkpoint": None,
            },
            "cylinder": {
                "env_cfg_cls": CylinderRobotEnvCfg_PLAY,
                "task": None,
                "checkpoint": None,
            },
            "drone": {
                "env_cfg_cls": DroneEnvCfg_PLAY,
                "task": None,
                "checkpoint": None,
            },
        }
        if args.robot_name not in robot_cfg:
            raise ValueError(f"Unsupported robot_name: {args.robot_name}")
        cfg = robot_cfg[args.robot_name]
        env_cfg = cfg["env_cfg_cls"]()
        init_env_cfg(env_cfg, args, self.next_episode)
        self._configure_evaluation_controls(env_cfg, args)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.curriculum = None
        manager_env = ManagerBasedRLEnv(cfg=env_cfg)
        self._sync_on_demand_render_products(manager_env)
        if cfg["task"] is not None:
            task_name = cfg["task"]
            agent_cfg = rsl_rl_cli_args.parse_rsl_rl_cfg(task_name, args)
            env = RslRlVecEnvWrapper(manager_env)
            ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            ppo_runner.load(str(cfg["checkpoint"]))
            policy = ppo_runner.get_inference_policy(device=args.device)
        else:
            env = manager_env
            policy = None

        # vln env
        env = VLNEnvWrapper(args, env, policy, args.robot_name)
        self.env = env
        self.manager_env = manager_env
        if self.robot_name == "spot":
            self.transform_root_z_offset = 0.5 * float(getattr(manager_env.cfg, "robot_height", 0.0) or 0.0)
        else:
            self.transform_root_z_offset = 0.0
        if args.on_demand_render or getattr(args, "gui_high_fps", False):
            self._disable_step_rendering(manager_env)
        print("[SIM] Env setup complete")

        # viewpoint
        self.update_viewer()

        # keyboard
        self.set_up_keyboard()

    def _configure_evaluation_controls(self, env_cfg, args):
        control_dt = getattr(args, "timed_trajectory_control_dt", None)
        if control_dt is not None:
            control_dt = float(control_dt)
            env_cfg.decimation = control_decimation(env_cfg.sim.dt, control_dt)
            env_cfg.sim.render_interval = env_cfg.decimation
            print(
                f"[SIM] Timed trajectory control: physics_dt={env_cfg.sim.dt:.4f}s, "
                f"decimation={env_cfg.decimation}, control_dt={control_dt:.4f}s"
            )
        if os.environ.get("NAVVERSE_CAMERA_RGB_ONLY", "0").lower() in {
            "1",
            "true",
            "yes",
        }:
            for camera_name in ("pov_camera", "third_person_camera"):
                camera_cfg = getattr(env_cfg.scene, camera_name, None)
                if camera_cfg is not None:
                    camera_cfg.data_types = ["rgb"]
        if args.disable_camera:
            self._configure_no_camera_rendering(env_cfg)
        elif args.on_demand_render:
            self._configure_on_demand_rendering(env_cfg)
        if not args.enable_locomotion_reward:
            env_cfg.rewards = EmptyCfg()
            needs_contact_sensor = (
                getattr(env_cfg, "control_mode", None) == "physics_velocity"
            )
            if (
                not getattr(args, "enable_contact_sensor", False)
                and not needs_contact_sensor
                and hasattr(env_cfg.scene, "contact_forces")
            ):
                env_cfg.scene.contact_forces = None

    def _configure_no_camera_rendering(self, env_cfg):
        if hasattr(env_cfg.observations, "camera"):
            env_cfg.observations.camera = None
        env_cfg.scene.pov_camera = None
        env_cfg.scene.third_person_camera = None
        env_cfg.num_rerenders_on_reset = 0
        env_cfg.wait_for_textures = False
        env_cfg.robot_warmup_steps = 0

    def _configure_on_demand_rendering(self, env_cfg):
        if hasattr(env_cfg.observations, "camera"):
            env_cfg.observations.camera = None
        env_cfg.num_rerenders_on_reset = 0
        env_cfg.wait_for_textures = False
        for camera_name in ("pov_camera", "third_person_camera"):
            camera_cfg = getattr(env_cfg.scene, camera_name, None)
            if camera_cfg is not None:
                camera_cfg.update_period = 10**9

    def _sync_on_demand_render_products(self, manager_env=None):
        if not self.args.on_demand_render:
            return
        from omni.replicator.core.scripts.utils.viewport_manager import ViewportManager

        manager_env = manager_env or self.manager_env
        if manager_env is None:
            return
        viewport_manager = ViewportManager()
        requested_cameras = set(self.on_demand_camera_names)
        for camera_name in ("pov_camera", "third_person_camera"):
            if camera_name not in manager_env.scene.sensors:
                continue
            sensor = manager_env.scene.sensors[camera_name]
            for render_product_path in sensor.render_product_paths:
                render_product = viewport_manager._hydra_textures.get(
                    viewport_manager._context, render_product_path
                )
                render_product.hydra_texture.set_updates_enabled(
                    camera_name in requested_cameras
                )

    def _disable_step_rendering(self, manager_env):
        manager_env.cfg.sim.render_interval = 10**9

    def reset(self, episode):
        with self.load_env_lock:
            print("[Sim] Reset the simulation...")
            with self.server_data_lock:
                self.reset_server_cache()
            self.next_episode = episode
            self.reset_flag = True
            if not self.init:
                self.init = True
                self.init_env()
            self.clear_waypoints()
            self.commands.zero_()
            if self.command_dim == 7:
                self.commands[:, 6] = 1.0
            self.warmup_steps = getattr(self.manager_env.cfg, "robot_warmup_steps", 0)
            self._configure_controlled_episode_timeout()

    def _configure_controlled_episode_timeout(self):
        timeout_s = getattr(self.args, "controlled_episode_timeout_s", None)
        timeout_s = DEFAULT_EPISODE_TIMEOUT_S if timeout_s is None else float(timeout_s)
        if timeout_s <= 0.0:
            raise ValueError("Controlled episode timeout must be positive")
        self.env.terminations_cfg.time_out.params["max_time"] = timeout_s
        term_cfg = self.env.termination_manager.get_term_cfg("time_out")
        term_cfg.params["max_time"] = timeout_s
        self.env.termination_manager.set_term_cfg("time_out", term_cfg)
        if timeout_s != DEFAULT_EPISODE_TIMEOUT_S:
            print(f"[SIM] Applied controlled episode timeout: {timeout_s:.1f}s")

    def load_episode(self, episode_label):
        print(f"[SIM] Loading episode: {episode_label}")
        self.reset(self.episode_list[self.episode_label_list.index(episode_label)])

    def reset_server_cache(self):
        self._latest_data = {
            "rgb": None,
            "depth": None,
            "position": None,
            "quat_xyzw": None,
            "timestamp": None,
            "info": {}
        }

    def latest_data_ready(self):
        if self._latest_data is None:
            return False
        for key in self._latest_data:
            if self._latest_data[key] is None:
                return False
        return True

    def _handle_velocity_command(self, message):
        self._clear_timed_trajectory("replaced_by_velocity")
        vx, vy, vw = message["vx"], message["vy"], message["vw"]
        if self.command_dim == 3:
            self.commands[self.robot_index] = torch.tensor([float(vx), float(vy), float(vw)], device=self.device)
        self.commands_source = 'server_vel'

    def _handle_waypoint_command(self, message):
        self._clear_timed_trajectory("replaced_by_waypoint")
        self.configure_benchmark_waypoint_follower()
        self.configure_controlled_waypoint_follower()
        self.set_waypoints(message["waypoint"], verbose=False)
        self.commands_source = 'server_waypoint'

    def _handle_timed_trajectory_command(self, message):
        if self.command_dim != 3 and self.robot_name != "tidybot":
            raise ValueError(
                "TIMED_TRAJECTORY requires a velocity-controlled base or TidyBot"
            )
        waypoints = np.asarray(message.get("waypoints"), dtype=float)
        dt = float(message.get("dt", 0.0))
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) == 0:
            raise ValueError(f"TIMED_TRAJECTORY waypoints must be non-empty (N, 3), got {waypoints.shape}")
        if not np.isfinite(waypoints).all():
            raise ValueError("TIMED_TRAJECTORY waypoints must be finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"TIMED_TRAJECTORY dt must be positive, got {dt}")

        self.clear_waypoints()
        start_step = int(getattr(self.manager_env, "common_step_counter", 0) or 0)
        step_dt = float(getattr(self.manager_env, "step_dt", 0.0) or 0.0)
        if not np.isclose(step_dt, dt, rtol=0.0, atol=1e-9):
            raise ValueError(
                "TIMED_TRAJECTORY requires one reference per simulator control tick: "
                f"trajectory dt={dt:.6f}s, simulator step_dt={step_dt:.6f}s"
            )
        duration_s = len(waypoints) * dt
        self.timed_trajectory_follower.reset()
        self.tidybot_timed_trajectory_follower.reset()
        self.timed_trajectory = {
            "waypoints": waypoints,
            "dt": dt,
            "start_step": start_step,
            "duration_s": duration_s,
        }
        self.last_timed_trajectory_status = {
            "active": True,
            "start_sim_time": start_step * step_dt,
            "duration_s": duration_s,
            "point_count": int(len(waypoints)),
            "dt": dt,
            "interpolation": "none",
            "execution": "one_setpoint_per_control_tick",
            "reason": "executing",
        }
        self.commands_source = 'server_timed_trajectory'

    def _clear_timed_trajectory(self, reason):
        if self.timed_trajectory is None:
            return
        self.timed_trajectory = None
        if self.last_timed_trajectory_status is not None:
            self.last_timed_trajectory_status["active"] = False
            self.last_timed_trajectory_status["reason"] = reason

    def _uses_internal_task_generation_reference(self):
        return (
            getattr(self.args, "controlled_reference_execution", "")
            == "task_generation_internal"
        )

    def _configure_internal_task_generation_timeout(self, episode):
        timeout_s = float(episode.get("episode_timeout", DEFAULT_EPISODE_TIMEOUT_S))
        self.env.terminations_cfg.time_out.params["max_time"] = timeout_s
        term_cfg = self.env.termination_manager.get_term_cfg("time_out")
        term_cfg.params["max_time"] = timeout_s
        self.env.termination_manager.set_term_cfg("time_out", term_cfg)
        print(f"[SIM] Applied task-generation episode timeout: {timeout_s:.1f}s")

    def _handle_transform_command(self, message):
        self._clear_timed_trajectory("replaced_by_transform")
        self.pending_transform_command = self.env.parse_transform_command(message)
        self.commands_source = 'server_transform'

    def _handle_arm_command(self, msg_type, message):
        if msg_type == 'ARM_TARGET_POSE' and 'arm_target_pose' not in message:
            self.arm_command = {'arm_target_pose': message}
        else:
            self.arm_command = message
        self.commands_source = 'server_arm'

    def _handle_gripper_command(self, message):
        self.arm_command = {"gripper": message.get("position", message.get("value", message))}
        self.commands_source = 'server_gripper'

    def _handle_stop_command(self):
        self._clear_timed_trajectory("stopped")
        self.clear_waypoints()
        self.commands[self.robot_index] = torch.zeros(self.command_dim, device=self.device)
        if self.command_dim == 7:
            self.commands[self.robot_index, 6] = 1.0
        self.commands_source = 'server_stop'
        self.env.set_stop_called(self.robot_index, True)

    def _handle_episode_command(self, message):
        self._clear_timed_trajectory("episode_changed")
        self.load_episode(message["episode_label"])
        self.call_callbacks('client_episode_changed', message["episode_label"])

    def _sensor_error_payload(self, message="Unknown error"):
        error_message = {
            "success": False,
            "message": message,
        }
        if self.current_episode is not None:
            error_message["episode_id"] = self.current_episode.get("episode_id", "")
            error_message["scene_id"] = self.current_episode.get("scene_id", "")
        return error_message

    def _state_sensor_error_payload(self):
        if self.sim_state == "init":
            return self._sensor_error_payload("Simulator is initializing")
        if self.sim_state == "loading":
            return self._sensor_error_payload("Episode is loading")
        if self.sim_state == "terminated":
            error_message = self._sensor_error_payload("Episode has been terminated")
            error_message["sim_state"] = self.sim_state
            if self.manager_env is not None:
                error_message.update(self._state_timing_payload())
            if self.env is not None:
                try:
                    position, quat_xyzw = self.env.get_body_pose()
                    error_message["pose"] = {
                        "position": [float(x) for x in position],
                        "orientation": [float(x) for x in quat_xyzw],
                    }
                except Exception as exc:
                    error_message["pose_error"] = str(exc)
            error_message.update(self._latest_data["info"])
            return error_message
        return None

    def _on_demand_render_error_payload(self):
        if not self.args.on_demand_render:
            return None
        if self.waiting_for_first_action and self.latest_data_ready():
            return None
        render_error = self.request_on_demand_sensor_data()
        if render_error is None:
            return None
        return self._sensor_error_payload(render_error)

    def _sensor_data_payload(self):
        with self.server_data_lock:
            for key in self._latest_data:
                if self._latest_data[key] is None:
                    return self._sensor_error_payload("Episode is not ready yet")
            return format_data(
                self._latest_data["rgb"],
                self._latest_data["depth"],
                self._latest_data["position"],
                self._latest_data["quat_xyzw"],
                self._latest_data["info"],
                self._latest_data["timestamp"],
            )

    def _sensor_payload(self):
        error_payload = self._state_sensor_error_payload()
        if error_payload is not None:
            return error_payload
        error_payload = self._on_demand_render_error_payload()
        if error_payload is not None:
            return error_payload
        return self._sensor_data_payload()

    def _episode_list_payload(self):
        episode_set_list = load_episode_set(self.args.episode_folder)
        episode_set_list["all"] = self.episode_label_list
        return episode_set_list

    def action_callback(self, msg_type, message):
        if self.closed:
            return
        print(f"[SIM] Received command: {msg_type}")
        command_handlers = {
            'VEL': self._handle_velocity_command,
            'WAYPOINT': self._handle_waypoint_command,
            'TIMED_TRAJECTORY': self._handle_timed_trajectory_command,
            'TRANSFORM': self._handle_transform_command,
            'ARM': lambda payload: self._handle_arm_command(msg_type, payload),
            'ARM_JOINTS': lambda payload: self._handle_arm_command(msg_type, payload),
            'ARM_TARGET_POSE': lambda payload: self._handle_arm_command(msg_type, payload),
            'GRIPPER': self._handle_gripper_command,
            'STOP': lambda _payload: self._handle_stop_command(),
        }
        if msg_type in command_handlers:
            self.waiting_for_first_action = False
            command_handlers[msg_type](message)
        elif msg_type == 'CLEAR_WAYPOINTS':
            self.clear_waypoints()
            self.commands_source = 'server_clear_waypoints'
        elif msg_type == 'EPISODE':
            self._handle_episode_command(message)

    def data_callback(self, request_type):
        if self.closed:
            return {
                "success": False,
                "message": "Simulator is shutting down",
            }
        if request_type == "GET_SIM_STATE":
            return self._state_payload()
        if request_type == "GET_SENSOR_DATA":
            return self._sensor_payload()
        elif request_type == "GET_EPISODE_LIST":
            return self._episode_list_payload()

    def _state_payload(self):
        payload = {
            "success": self.sim_state == "running",
            "message": self.sim_state,
            "sim_state": self.sim_state,
            "timestamp_server_ns": time.time_ns(),
        }
        if self.current_episode is not None:
            payload.update({
                "episode_label": self.current_episode.episode_label,
                "episode_id": self.current_episode.get("episode_id", ""),
                "scene_id": self.current_episode.get("scene_id", ""),
            })
        if self.manager_env is not None:
            payload.update(self._state_timing_payload())
            payload.update({
                "commands_source": self.commands_source,
                "waypoint_count": len(self.waypoints) if self.waypoints is not None else 0,
                "timed_trajectory_status": self.last_timed_trajectory_status,
            })
            try:
                payload["command"] = self.commands[self.robot_index].detach().cpu().tolist()
            except Exception:
                pass
        if self.env is not None and self.sim_state in {"running", "terminated"}:
            try:
                position, quat_xyzw = self.env.get_body_pose()
                payload["pose"] = {
                    "position": [float(x) for x in position],
                    "orientation": [float(x) for x in quat_xyzw],
                }
                if hasattr(self.env, "low_level_action"):
                    payload["low_level_action"] = self.env.low_level_action.detach().cpu().tolist()
            except Exception as exc:
                payload["pose_error"] = str(exc)
        if self._latest_data is not None:
            info = self._latest_data.get("info") or {}
            for key in ("termination_reason", "metrics", "gt_goal_position", "instruction", "instruction_type", "target_type"):
                if key in info:
                    payload[key] = info[key]
        return payload

    def _state_timing_payload(self):
        step_dt = float(getattr(self.manager_env, "step_dt", 0.0) or 0.0)
        sim_step = int(getattr(self.manager_env, "common_step_counter", 0) or 0)
        sim_frame_count = max(0, sim_step - self.telemetry_start_step)
        sim_total_time_s = sim_frame_count * step_dt
        wall_elapsed_s = max(1e-6, time.time() - self.telemetry_start_wall_time)
        return {
            "sim_step": sim_step,
            "step_dt": step_dt,
            "sim_time": sim_step * step_dt,
            "sim_frame_count": sim_frame_count,
            "sim_fps": sim_frame_count / wall_elapsed_s,
            "sim_to_wall_ratio": sim_total_time_s / wall_elapsed_s,
            "sim_total_time_s": sim_total_time_s,
        }

    def add_callback(self, callback_name, func):
        available_callbacks = ['client_episode_changed', 'pre_step', 'step_finished', 'reset_finished', 'on_render']
        if callback_name not in available_callbacks:
            raise ValueError(f"Invalid callback name: {callback_name}")
        if func in self.callbacks.get(callback_name, []):
            print(f"[SIM] Callback {callback_name} already exists")
            return len(self.callbacks[callback_name])-1
        self.callbacks.setdefault(callback_name, []).append(func)
        return len(self.callbacks[callback_name])-1

    def remove_callback(self, callback_name, func):
        if callback_name in self.callbacks:
            if func in self.callbacks[callback_name]:
                self.callbacks[callback_name].remove(func)
            else:
                raise ValueError(f"Function {func} not found in callback {callback_name}")

    def call_callbacks(self, callback_name, *args, **kwargs):
        for func in list(self.callbacks.get(callback_name, [])):
            func(*args, **kwargs)

    def update_viewer(self):
        if self.args.headless or self.env is None or self.viewport is None:
            return False

        if self.viewport_third_person:
            camera_path = self.env.get_camera_prim_path("third_person_camera", self.robot_index)
        else:
            camera_path = self.env.get_camera_prim_path("pov_camera", self.robot_index)

        camera_prim = self.manager_env.scene.stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            if self.args.headless:
                return True
            print(f"[SIM] Camera prim not ready: {camera_path}")
            return False

        self.viewport.set_active_camera(camera_path)
        return True

    def start_server(self, host="localhost", port=12300, server_name="IsaacLabServer"):
        # Start socket server in background
        self.server_stop_event = Event()
        server_thread = Thread(target=run_server, kwargs={
            "data_cb": self.data_callback, 
            "action_cb": self.action_callback, 
            "stop_flag": self.server_stop_event,
            "host": host,
            "port": port,
            "server_name": server_name
        })
        server_thread.daemon = True
        server_thread.start()
        self.server_thread = server_thread
        print("[SIM] Socket server started")

    def _cancel_pending_render_request(self):
        with self.on_demand_render_lock:
            pending_event = self.on_demand_render_event
            self.on_demand_render_error = "Simulator is shutting down"
            self.on_demand_render_pending = False
            self.on_demand_render_event = None
        if pending_event is not None:
            pending_event.set()

    def _release_keyboard_subscription(self):
        self._pressed_control_keys.clear()
        sub_keyboard = getattr(self, "_sub_keyboard", None)
        unsubscribe = getattr(sub_keyboard, "unsubscribe", None)
        if callable(unsubscribe):
            try:
                unsubscribe()
            except Exception as exc:
                print(f"[SIM] Failed to unsubscribe keyboard events: {exc}")
        self._sub_keyboard = None
        self._keyboard = None
        self._input = None

    def _stop_socket_server(self):
        if self.server_stop_event is not None:
            self.server_stop_event.set()
        if self.server_thread is not None and self.server_thread.is_alive():
            self.server_thread.join(timeout=1.0)

    def _stop_visualizer(self):
        if self.visualizer is None:
            return
        server = getattr(self.visualizer, "server", None)
        stop = getattr(server, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as exc:
                print(f"[SIM] Failed to stop Foxglove server: {exc}")
        self.visualizer = None

    def _close_env(self):
        if self.env is None:
            return
        try:
            self.env.close()
        except Exception as exc:
            print(f"[SIM] Failed to close environment cleanly: {exc}")
        finally:
            self.env = None
            self.manager_env = None

    def close(self):
        if self.closed:
            return
        print("[SIM] Closing simulator")
        self.closed = True
        self.sim_state = "terminated"
        self._cancel_pending_render_request()
        self._release_keyboard_subscription()
        self._stop_socket_server()
        self._stop_visualizer()
        self._close_env()
    
    def set_waypoints(self, waypoints, verbose=False):
        """Waypoint should be in the format of [x, y, yaw], set yaw to inf if angle is ignored"""
        if verbose:
            body_pos, body_quat = self.env.get_body_pose()
            base_xy = np.array(body_pos[:2])
            base_yaw = R.from_quat(body_quat).as_euler('ZYX')[0]
            print(f"current pose: {base_xy}, {base_yaw}")
            print(f"next waypoint: {waypoints[0]}")
        self.waypoints = waypoints
        self.waypoint_follower.reset()
        self.waypoint_command_frames = 0

    def configure_benchmark_waypoint_follower(self):
        if self.robot_name == "turtlebot":
            self.reset_waypoint_follower_config()
            return
        self.configure_waypoint_follower_profile("benchmark", reset_to_default=True)
        if getattr(self.args, "controlled_terrain_manifest", ""):
            if self.robot_name == "drone":
                self.waypoint_follower.max_vel = np.asarray([2.5, 0.0, 4.0], dtype=float)

    def configure_controlled_waypoint_follower(self):
        profile = getattr(self.args, "controlled_waypoint_follower_profile", "")
        if not profile:
            return
        if not self.configure_waypoint_follower_profile(profile, reset_to_default=False):
            raise ValueError(f"Unknown controlled waypoint follower profile: {profile}")
        print(f"[SIM] Applied controlled waypoint follower profile: {profile}")

    def set_discrete_waypoint_action(self, action):
        action = str(action).upper()
        if action == "STOP":
            self._handle_stop_command()
            return True
        if self.env is None or self.manager_env is None:
            print(f"[SIM] Ignoring {action}: environment is not ready")
            return False
        episode_label = self.current_episode.episode_label if self.current_episode is not None else ""
        scene_id = "_".join(str(episode_label).split("_")[:-1])
        if scene_id.startswith(("vc_", "vcp_", "innout_")):
            forward_distance = 1.0
        else:
            forward_distance = 0.25
        turn_angle = np.deg2rad(30.0)
        action_to_relative_pose = {
            "FORWARD": (forward_distance, 0.0, 0.0),
            "BACKWARD": (-forward_distance, 0.0, 0.0),
            "LEFT": (0.0, forward_distance, 0.0),
            "RIGHT": (0.0, -forward_distance, 0.0),
            "TURN_LEFT": (0.0, 0.0, turn_angle),
            "TURN_RIGHT": (0.0, 0.0, -turn_angle),
        }
        if action not in action_to_relative_pose:
            print(f"[SIM] Unknown discrete action from UI: {action}")
            return False

        body_pos, body_quat = self.env.get_body_pose()
        base_yaw = R.from_quat(body_quat).as_euler("ZYX", degrees=False)[0]
        rel_x, rel_y, rel_yaw = action_to_relative_pose[action]
        cos_yaw = np.cos(base_yaw)
        sin_yaw = np.sin(base_yaw)
        target_x = float(body_pos[0] + cos_yaw * rel_x - sin_yaw * rel_y)
        target_y = float(body_pos[1] + sin_yaw * rel_x + cos_yaw * rel_y)
        target_yaw = float(np.arctan2(np.sin(base_yaw + rel_yaw), np.cos(base_yaw + rel_yaw)))
        if hasattr(self.env, "set_stop_called"):
            self.env.set_stop_called(self.robot_index, False)
        self.configure_benchmark_waypoint_follower()
        if action == "BACKWARD":
            self.waypoint_follower.prefer_forward = False
        self.waypoint_path_planner = "discrete"
        self.set_waypoints([[target_x, target_y, target_yaw]], verbose=True)
        self.commands_source = f"ui_{action.lower()}"
        return True
    
    def generate_astar_path(self, scene_id, ref_path, astar_overrides=None, goal_position=None, goal_radius=None):
        self._configure_astar_follower_for_scene(scene_id)
        self.last_astar_world_path = None
        self.last_astar_waypoints = None
        astar_config = self._astar_config(scene_id, overrides=astar_overrides)
        self._active_astar_config = astar_config
        astar_bev_mode = astar_config.bev_mode
        if astar_bev_mode not in {"global", "local"}:
            raise ValueError(f"Unsupported astar_bev_mode: {astar_bev_mode}")

        ref_points = self._prepare_astar_reference_points(ref_path)
        if len(ref_points) < 2:
            return []
        target_goal = None
        if goal_position is not None:
            target_goal = np.asarray(goal_position, dtype=float)[:3]
            if target_goal.shape[0] >= 2:
                if target_goal.shape[0] < 3:
                    target_goal = np.array([target_goal[0], target_goal[1], ref_points[-1][2]], dtype=float)
                if goal_radius is None:
                    if np.linalg.norm(ref_points[-1][:2] - target_goal[:2]) > 0.05:
                        ref_points.append(target_goal)
                    else:
                        ref_points[-1] = target_goal

        goal_stop_radius_m = self._apply_astar_goal_stop_radius(goal_radius)

        global_time_budget = float(astar_config.global_planner.time_budget_s)
        segment_time_budget = float(astar_config.segmentation.segment_time_budget_s)
        segment_total_budget = float(astar_config.segmentation.segment_total_budget_s)
        goal_tolerance = float(astar_config.segmentation.goal_tolerance_m)
        strict_endpoint = bool((self.current_episode or {}).get("task_generation_strict_endpoint", False))
        if strict_endpoint:
            goal_tolerance = min(
                goal_tolerance,
                float((self.current_episode or {}).get("task_generation_path_endpoint_tolerance_m", goal_tolerance)),
            )
        elif goal_stop_radius_m is not None:
            goal_tolerance = goal_stop_radius_m
        total_path = None
        planner = None
        plan_source = None
        plan_errors = []
        self.last_astar_plan_info = {
            "scene_id": scene_id,
            "source": None,
            "fallback": False,
            "errors": plan_errors,
        }
        global_planner = None
        if bool(astar_config.global_planner.first):
            try:
                global_planner = self._create_global_bev_astar_planner(scene_id)
                if bool(astar_config.global_planner.grid2d):
                    grid2d_budget = float(astar_config.global_planner.grid2d_time_budget_s)
                    grid2d_clearance = float(astar_config.global_planner.grid2d_clearance_m)
                    grid2d_min_clearance = float(
                        OmegaConf.select(
                            astar_config,
                            "global_planner.grid2d_min_clearance_m",
                            default=grid2d_clearance,
                        )
                    )
                    grid2d_retries = max(
                        1,
                        int(
                            OmegaConf.select(
                                astar_config,
                                "global_planner.grid2d_clearance_retries",
                                default=1,
                            )
                        ),
                    )
                    if grid2d_retries > 1 and grid2d_min_clearance < grid2d_clearance:
                        grid2d_clearances = np.linspace(grid2d_clearance, grid2d_min_clearance, grid2d_retries)
                    else:
                        grid2d_clearances = [grid2d_clearance]
                    for clearance_m in grid2d_clearances:
                        clearance_m = float(clearance_m)
                        grid2d_path = global_planner.plan_reference_path_2d(
                            ref_points[0],
                            ref_points[-1],
                            time_budget=grid2d_budget,
                            resolution_m=float(astar_config.global_planner.grid2d_resolution_m),
                            clearance_m=clearance_m,
                            snap_distance_m=float(astar_config.global_planner.grid2d_snap_m),
                            footprint_check_neighbors=bool(
                                OmegaConf.select(
                                    astar_config,
                                    "global_planner.grid2d_footprint_check_neighbors",
                                    default=False,
                                )
                            ),
                            collision_spacing_m=float(
                                OmegaConf.select(
                                    astar_config,
                                    "global_planner.grid2d_collision_spacing_m",
                                    default=0.15,
                                )
                            ),
                        )
                        if not self._astar_segment_reaches_goal(grid2d_path, ref_points[-1], goal_tolerance):
                            plan_errors.append(f"global_grid2d_failed:{clearance_m:.2f}")
                            continue
                        if not global_planner.is_world_path_collision_free(grid2d_path):
                            plan_errors.append(f"global_grid2d_collision:{clearance_m:.2f}")
                            continue
                        planner = global_planner
                        total_path = grid2d_path
                        plan_source = "global_grid2d"
                        self._save_astar_path_visualization(planner, total_path, 0, plan_source)
                        print(f"[SIM] A* coarse global 2D path succeeded clearance={clearance_m:.2f}m")
                        break
                if total_path is None and bool(OmegaConf.select(astar_config, "global_planner.direct", default=True)):
                    global_path = self._try_plan_astar_segment(
                        global_planner,
                        ref_points[0],
                        ref_points[-1],
                        global_time_budget,
                        goal_tolerance,
                        "global:start_goal",
                    )
                    if self._astar_segment_reaches_goal(global_path, ref_points[-1], goal_tolerance):
                        planner = global_planner
                        total_path = self._select_safer_astar_segment_path(
                            global_planner,
                            global_path,
                            ref_points[0],
                            ref_points[-1],
                            "global:start_goal",
                        )
                        plan_source = "global_direct"
                        self._save_astar_path_visualization(planner, total_path, 0, plan_source)
                        print("[SIM] A* global start-goal path succeeded")
                    else:
                        plan_errors.append("global_direct_failed")
            except Exception as exc:
                plan_errors.append(f"global_direct_error:{exc}")
                print(f"[SIM] A* global start-goal path failed: {exc}")

        if (
            total_path is None
            and global_planner is not None
            and bool(OmegaConf.select(astar_config, "segmentation.enabled", default=True))
        ):
            segmented_path = self._plan_ref_segments_with_planner(
                global_planner,
                ref_points,
                segment_time_budget,
                goal_tolerance,
                total_time_budget=segment_total_budget,
            )
            if segmented_path is not None:
                planner = global_planner
                total_path = segmented_path
                plan_source = "global_segmented"
                self._save_astar_path_visualization(planner, total_path, 0, plan_source)
                print("[SIM] A* global segmented path succeeded")
            else:
                plan_errors.append("global_segmented_failed")

        if total_path is None and astar_bev_mode == "local" and bool(astar_config.local_single_capture.enabled):
            single_capture_path = self._plan_local_single_capture_path(ref_points, segment_time_budget, goal_tolerance, segment_total_budget)
            if single_capture_path is not None:
                planner, total_path = single_capture_path
                plan_source = "local_single"
                self._save_astar_path_visualization(planner, total_path, 0, plan_source)

        if total_path is None and astar_bev_mode == "local":
            total_path = []
            start = ref_points[0]
            local_segment_start_time = time.time()
            for segment_idx, goal in enumerate(ref_points[1:]):
                remaining_segment_time = segment_total_budget - (time.time() - local_segment_start_time)
                if remaining_segment_time <= 0.0:
                    plan_errors.append(f"local_segmented_budget_exhausted:{segment_idx}")
                    total_path = []
                    break
                try:
                    planner, path = self._plan_astar_segment(
                        "local",
                        None,
                        start,
                        goal,
                        segment_idx,
                        min(segment_time_budget, remaining_segment_time),
                        goal_tolerance,
                    )
                except Exception as exc:
                    plan_errors.append(f"local_segmented_error:{segment_idx}:{exc}")
                    total_path = []
                    break
                self._save_astar_path_visualization(planner, path, segment_idx, "local_segmented")
                if total_path:
                    total_path.extend(path[1:])
                else:
                    total_path.extend(path)
                start = path[-1]
            if len(total_path) >= 2:
                plan_source = "local_segmented"

        if total_path is None or len(total_path) < 2:
            if bool((self.current_episode or {}).get("task_generation_require_astar_path", False)):
                raise RuntimeError(f"A* planning failed for task-generation episode: {plan_errors}")
            if not bool(astar_config.fallback.allow_reference):
                raise RuntimeError(f"A* planning failed and reference fallback is disabled: {plan_errors}")
            total_path = [np.asarray(point, dtype=float) for point in ref_points]
            plan_source = "navmesh_fallback"
            self.last_astar_plan_info["fallback"] = True
            print("[SIM] A* planning failed; using navmesh reference fallback path")

        total_path, fallback_reason = self._maybe_use_reference_fallback_path(total_path, ref_points)
        if fallback_reason is not None:
            if bool((self.current_episode or {}).get("task_generation_require_astar_path", False)):
                raise RuntimeError(f"A* reference fallback rejected for task-generation episode: {fallback_reason}")
            plan_source = "navmesh_fallback"
            self.last_astar_plan_info["fallback"] = True
            self.last_astar_plan_info["fallback_reason"] = fallback_reason
        smoothing_info = self._smooth_astar_world_path(planner, total_path, plan_source)
        total_path = smoothing_info.pop("path")
        if target_goal is not None:
            final_gap = float(np.linalg.norm(np.asarray(total_path[-1])[:2] - target_goal[:2]))
            default_target_tolerance = goal_stop_radius_m if goal_stop_radius_m is not None else 0.05
            target_tolerance = float(
                (self.current_episode or {}).get(
                    "task_generation_path_endpoint_tolerance_m",
                    default_target_tolerance,
                )
            )
            if final_gap > target_tolerance:
                target_connection = np.asarray([total_path[-1], target_goal], dtype=float)
                if planner is not None and planner.is_world_path_collision_free(target_connection):
                    total_path = list(total_path) + [target_goal]
                else:
                    raise RuntimeError(
                        "A* final target connection is unsafe "
                        f"(gap={final_gap:.2f}m, tolerance={target_tolerance:.2f}m)"
                    )
        spacing = float(astar_config.waypoint.spacing_m)
        total_path = self._resample_astar_world_path(total_path, spacing)
        if planner is not None and plan_source != "navmesh_fallback":
            path_array = np.asarray(total_path, dtype=float)
            if not planner.is_world_path_collision_free(path_array):
                raise RuntimeError("A* generated path failed full footprint collision check")
            min_path_clearance = float(astar_config.planner.min_path_clearance_m)
            if min_path_clearance > 0.0:
                path_clearance, _ = planner.world_path_clearance(path_array)
                if path_clearance < min_path_clearance:
                    raise RuntimeError(
                        "A* generated path failed clearance check "
                        f"(min={path_clearance:.2f}m, required={min_path_clearance:.2f}m)"
                    )
        self.last_astar_world_path = np.asarray(total_path, dtype=float).tolist()
        waypoints = []
        for i, point in enumerate(total_path[1:], start=1):
            prev = total_path[i - 1]
            yaw = np.arctan2(point[1] - prev[1], point[0] - prev[0])
            waypoints.append([float(point[0]), float(point[1]), float(yaw)])
        self.last_astar_waypoints = [list(waypoint) for waypoint in waypoints]
        self.last_astar_plan_info.update(
            {
                "source": plan_source,
                "waypoints": int(len(waypoints)),
                "path_length_m": self._path_length_xy(total_path),
                "global_time_budget_s": global_time_budget,
                "segment_time_budget_s": segment_time_budget,
                "segment_total_budget_s": segment_total_budget,
                "config_name": astar_config.config_name,
                "grid2d_time_budget_s": float(astar_config.global_planner.grid2d_time_budget_s),
                "grid2d_resolution_m": float(astar_config.global_planner.grid2d_resolution_m),
                "grid2d_clearance_m": float(astar_config.global_planner.grid2d_clearance_m),
                "max_climb_height_m": float(astar_config.planner.max_climb_height_m),
                "min_obstacle_area_px": int(astar_config.planner.min_obstacle_area_px),
                "goal_position": None if target_goal is None else np.asarray(target_goal, dtype=float).tolist(),
                "goal_radius_m": None if goal_radius is None else float(goal_radius),
                "goal_stop_radius_m": goal_stop_radius_m,
                "goal_tolerance_m": goal_tolerance,
                "smoothing": smoothing_info,
            }
        )
        print(f"[SIM] Generated {len(waypoints)} A* waypoints from {plan_source}")
        self._save_astar_world_path(total_path, waypoints, plan_source or astar_bev_mode)
        return waypoints

    def _reference_path_to_waypoints(self, ref_path):
        points = [np.asarray(point, dtype=float) for point in ref_path]
        if len(points) < 2:
            return []
        waypoints = []
        for i, point in enumerate(points[1:], start=1):
            prev = points[i - 1]
            yaw = np.arctan2(point[1] - prev[1], point[0] - prev[0])
            waypoints.append([float(point[0]), float(point[1]), float(yaw)])
        return waypoints

    def _configure_astar_follower_for_scene(self, scene_id):
        self._apply_astar_follower_config(scene_id)

    def _apply_astar_goal_stop_radius(self, goal_radius):
        if goal_radius is None:
            return None
        goal_stop_radius_m = max(0.0, float(goal_radius)) + 1.0
        self.waypoint_follower.term_dist = max(float(self.waypoint_follower.term_dist), goal_stop_radius_m)
        return goal_stop_radius_m

    def _apply_astar_follower_config(self, scene_id=None):
        if self.robot_name != "spot":
            return
        astar_config = self._astar_config(scene_id)
        follower_config = astar_config.execution.follower
        self.waypoint_follower.max_vel = np.asarray(follower_config.max_vel, dtype=float)
        self.waypoint_follower.min_vel = np.asarray(follower_config.min_vel, dtype=float)
        self.waypoint_follower.kp = np.asarray(follower_config.kp, dtype=float)
        self.waypoint_follower.arrive_dist = float(follower_config.arrive_dist)
        self.waypoint_follower.arrive_yaw = np.deg2rad(float(follower_config.arrive_yaw_deg))
        self.waypoint_follower.term_dist = float(follower_config.term_dist)
        self.waypoint_follower.term_yaw = np.deg2rad(float(follower_config.term_yaw_deg))
        self.waypoint_follower.lookahead_distance = float(follower_config.lookahead_distance)
        self.waypoint_follower.lateral_deadband = float(follower_config.lateral_deadband)
        self.waypoint_follower.lateral_turn_damping_yaw = np.deg2rad(float(follower_config.lateral_turn_damping_yaw_deg))
        prefer_forward = bool(follower_config.prefer_forward)
        if str(scene_id or "").startswith("grCommercial"):
            prefer_forward = False
        self.waypoint_follower.prefer_forward = prefer_forward

    def reset_waypoint_follower_config(self):
        for key, value in self._default_waypoint_follower_config.items():
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(self.waypoint_follower, key, value)

    def _runtime_config(self, scene_id=None):
        config_path = self._runtime_config_path(scene_id)
        dotlist_overrides = tuple(getattr(self.args, "_config_dotlist_overrides", []) or [])
        cache_key = (str(Path(config_path).resolve()), dotlist_overrides)
        if cache_key not in self._runtime_config_cache:
            self._runtime_config_cache[cache_key] = load_runtime_config(
                config_path=config_path,
                dotlist_overrides=list(dotlist_overrides),
            )
        return self._runtime_config_cache[cache_key]

    def configure_waypoint_follower_profile(self, profile_name, reset_to_default=False, scene_id=None):
        if reset_to_default:
            self.reset_waypoint_follower_config()
        runtime_config = self._runtime_config(scene_id)
        profiles = getattr(getattr(runtime_config, "waypoint_follower", {}), "profiles", {})
        profile = profiles.get(profile_name) if hasattr(profiles, "get") else None
        if profile is None:
            return False
        array_fields = {"max_vel", "min_vel", "kp"}
        degree_fields = {
            "arrive_yaw_deg": "arrive_yaw",
            "term_yaw_deg": "term_yaw",
            "lateral_turn_damping_yaw_deg": "lateral_turn_damping_yaw",
        }
        for key, value in profile.items():
            if key in array_fields:
                setattr(self.waypoint_follower, key, np.asarray(value, dtype=float))
            elif key in degree_fields:
                setattr(self.waypoint_follower, degree_fields[key], np.deg2rad(float(value)))
            elif hasattr(self.waypoint_follower, key):
                setattr(self.waypoint_follower, key, bool(value) if key == "prefer_forward" else float(value))
        return True

    def _path_length_xy(self, path):
        if len(path) < 2:
            return 0.0
        return float(sum(np.linalg.norm(np.asarray(cur)[:2] - np.asarray(prev)[:2]) for prev, cur in zip(path[:-1], path[1:])))

    def _runtime_config_path(self, scene_id=None):
        explicit_args = getattr(self.args, "_explicit_args", set())
        if "config" in explicit_args:
            return getattr(self.args, "config", "configs/default.yaml")
        episode = self.current_episode
        if episode is not None and episode.get("config"):
            return str(
                resolve_task_config_path(
                    episode["config"],
                    scene_id=episode.get("scene_id"),
                    scene_type=episode.get("scene_type"),
                )
            )
        runtime_config_path = getattr(self, "runtime_config_path", None)
        if runtime_config_path:
            return runtime_config_path
        scene_id = scene_id or (episode.get("scene_id") if episode is not None else None)
        if scene_id:
            inferred_path = resolve_task_config_path(None, scene_id=scene_id)
            if Path(inferred_path).exists():
                return str(inferred_path)
        return getattr(self.args, "config", "configs/default.yaml")

    def _astar_config(self, scene_id=None, overrides=None):
        if scene_id is None and overrides is None and self._active_astar_config is not None:
            self.args.astar = self._active_astar_config
            return self._active_astar_config
        config_path = self._runtime_config_path(scene_id)
        dotlist_overrides = tuple(getattr(self.args, "_config_dotlist_overrides", []) or [])
        cache_key = (str(Path(config_path).resolve()), dotlist_overrides)
        if cache_key not in self._astar_config_cache:
            runtime_config = self._runtime_config(scene_id)
            config = runtime_config.astar
            if scene_id:
                scene_spec = resolve_scene_spec(load_config(config_path), scene_id)
                config = OmegaConf.merge(config, scene_spec.get("astar") or {})
            self._astar_config_cache[cache_key] = config
        config = self._astar_config_cache[cache_key]
        episode = self.current_episode
        episode_overrides = episode.get("astar", {}) if episode is not None else {}
        runtime_overrides = getattr(self.args, "astar_runtime_overrides", None)
        config = OmegaConf.merge(config, runtime_overrides or {})
        config = OmegaConf.merge(config, episode_overrides or {})
        config = OmegaConf.merge(config, overrides or {})
        self.args.astar = config
        return config

    def _maybe_use_reference_fallback_path(self, total_path, ref_points):
        astar_config = self._astar_config()
        if not bool(astar_config.fallback.allow_reference):
            return total_path, None
        astar_length = self._path_length_xy(total_path)
        ref_length = self._path_length_xy(ref_points)
        if ref_length <= 1e-6:
            return total_path, None
        max_ratio = float(astar_config.fallback.reference_fallback_max_ratio)
        ratio = astar_length / ref_length
        if ratio > max_ratio:
            print(f"[SIM] A* path length ratio {ratio:.2f} exceeds {max_ratio:.2f}; using reference fallback path")
            return [np.asarray(point, dtype=float) for point in ref_points], f"path_ratio:{ratio:.2f}>{max_ratio:.2f}"
        return total_path, None

    def _plan_astar_segment(self, astar_bev_mode, global_planner, start, goal, segment_idx, time_budget, goal_tolerance):
        astar_config = self._astar_config()
        if astar_bev_mode == "global":
            planner = global_planner
            path = self._try_plan_astar_segment(planner, start, goal, time_budget, goal_tolerance, segment_idx)
            if self._astar_segment_reaches_goal(path, goal, goal_tolerance):
                path = self._select_safer_astar_segment_path(planner, path, start, goal, segment_idx)
                return planner, path
            if bool(astar_config.fallback.allow_reference):
                direct_path = planner.direct_world_path(
                    start,
                    goal,
                    spacing=float(astar_config.waypoint.spacing_m) * 0.5,
                )
                if planner.is_world_path_collision_free(direct_path) and self._astar_segment_reaches_goal(
                    direct_path,
                    goal,
                    goal_tolerance,
                ):
                    print(f"[SIM] A* global segment {segment_idx} using direct reference fallback")
                    return planner, direct_path
            distance = float(np.linalg.norm(path[-1, :2] - goal[:2])) if path is not None and len(path) else float("inf")
            raise RuntimeError(
                f"A* global segment {segment_idx} did not reach subgoal "
                f"(distance={distance:.2f}m, tolerance={goal_tolerance:.2f}m)"
            )

        base_clearance, clearance_schedule = self._local_astar_clearance_schedule()

        last_path = np.empty((0, 3), dtype=float)
        for clearance in clearance_schedule:
            planner = self._create_local_bev_astar_planner(start, goal, clearance_m=clearance)
            path = self._try_plan_astar_segment(planner, start, goal, time_budget, goal_tolerance, segment_idx)
            if self._astar_segment_reaches_goal(path, goal, goal_tolerance):
                path = self._select_safer_astar_segment_path(planner, path, start, goal, segment_idx)
                if abs(clearance - base_clearance) > 1e-6:
                    print(f"[SIM] A* segment {segment_idx} succeeded with relaxed clearance={clearance:.3f}m")
                return planner, path
            last_path = path

        distance = float(np.linalg.norm(last_path[-1, :2] - goal[:2])) if len(last_path) else float("inf")
        raise RuntimeError(
            f"A* segment {segment_idx} did not reach subgoal "
            f"(distance={distance:.2f}m, tolerance={goal_tolerance:.2f}m)"
        )

    def _local_astar_clearance_schedule(self):
        astar_config = self._astar_config()
        base_clearance = float(astar_config.planner.clearance_m)
        if bool(astar_config.fallback.no_clearance_retries):
            return base_clearance, [base_clearance]
        clearance_schedule = []
        for clearance in (base_clearance, base_clearance * 0.5, 0.0):
            clearance = max(0.0, clearance)
            if not any(abs(clearance - existing) < 1e-6 for existing in clearance_schedule):
                clearance_schedule.append(clearance)
        return base_clearance, clearance_schedule

    def _plan_local_single_capture_path(self, ref_points, segment_time_budget, goal_tolerance, segment_total_budget=None):
        astar_config = self._astar_config()
        padding_m = float(astar_config.local_single_capture.padding_m)
        min_window_m = float(astar_config.local_bev.window_m)
        resolution_m = float(astar_config.local_single_capture.resolution_m)
        points = np.asarray(ref_points, dtype=float)
        xy_extent = np.max(points[:, :2], axis=0) - np.min(points[:, :2], axis=0)
        window_m = max(min_window_m, float(xy_extent.max() + 2.0 * padding_m))
        if window_m <= 16.0:
            resolution_m = min(resolution_m, float(astar_config.local_bev.resolution_m))
        captured = self._capture_local_bev_depth_for_points(
            ref_points,
            padding_m=padding_m,
            min_window_m=min_window_m,
            resolution_m=resolution_m,
            max_window_m=float(astar_config.local_single_capture.max_window_m),
            log_label="single-capture local",
        )
        if captured is None:
            return None
        bev_depth, info = captured
        base_clearance, clearance_schedule = self._local_astar_clearance_schedule()

        for clearance in clearance_schedule:
            planner = self._create_local_bev_astar_planner_from_depth(bev_depth, info, clearance_m=clearance)
            path = self._plan_ref_segments_with_planner(
                planner,
                ref_points,
                segment_time_budget,
                goal_tolerance,
                total_time_budget=segment_total_budget,
            )
            if path is not None:
                if abs(clearance - base_clearance) > 1e-6:
                    print(f"[SIM] A* single-capture guided path succeeded with relaxed clearance={clearance:.3f}m")
                else:
                    print("[SIM] A* single-capture guided path succeeded")
                return planner, path

        print("[SIM] A* single-capture local BEV failed; falling back to per-segment local BEV captures")
        return None

    def _plan_ref_segments_with_planner(self, planner, ref_points, time_budget, goal_tolerance, total_time_budget=None):
        astar_config = self._astar_config()
        total_path = []
        start = ref_points[0]
        start_time = time.time()
        for segment_idx, goal in enumerate(ref_points[1:]):
            if total_time_budget is not None:
                remaining_time = float(total_time_budget) - (time.time() - start_time)
                if remaining_time <= 0.0:
                    print(f"[SIM] A* segmented total budget exhausted before segment {segment_idx}")
                    return None
                segment_budget = min(float(time_budget), remaining_time)
            else:
                segment_budget = float(time_budget)
            path = self._try_plan_astar_segment(planner, start, goal, segment_budget, goal_tolerance, f"single:{segment_idx}")
            if self._astar_segment_reaches_goal(path, goal, goal_tolerance):
                path = self._select_safer_astar_segment_path(planner, path, start, goal, f"single:{segment_idx}")
            else:
                direct_path = planner.direct_world_path(
                    start,
                    goal,
                    spacing=float(astar_config.waypoint.spacing_m) * 0.5,
                )
                if planner.is_world_path_collision_free(direct_path) and self._astar_segment_reaches_goal(
                    direct_path,
                    goal,
                    goal_tolerance,
                ):
                    print(f"[SIM] A* single-capture segment {segment_idx} using direct reference fallback")
                    path = direct_path
                else:
                    return None
            if total_path:
                total_path.extend(path[1:])
            else:
                total_path.extend(path)
            start = path[-1]
        return [np.asarray(point, dtype=float) for point in total_path] if len(total_path) >= 2 else None

    def _try_plan_astar_segment(self, planner, start, goal, time_budget, goal_tolerance, segment_idx):
        path = planner.plan_reference_path(start, goal, time_budget=time_budget)
        if self._astar_segment_reaches_goal(path, goal, goal_tolerance):
            return path
        astar_config = self._astar_config()
        retry_multiplier = float(astar_config.segmentation.retry_time_budget_multiplier)
        if retry_multiplier <= 1.0:
            return path
        print(f"[SIM] Retrying A* segment {segment_idx} with extra time budget")
        path = planner.plan_reference_path(start, goal, time_budget=time_budget * retry_multiplier)
        return path

    def _select_safer_astar_segment_path(self, planner, path, start, goal, segment_idx):
        astar_config = self._astar_config()
        direct_path = planner.direct_world_path(start, goal, spacing=float(astar_config.waypoint.spacing_m) * 0.5)
        if not planner.is_world_path_collision_free(direct_path):
            return path
        path_min, path_avg = planner.world_path_clearance(path)
        direct_min, direct_avg = planner.world_path_clearance(direct_path)
        min_clearance = float(astar_config.planner.min_path_clearance_m)
        if direct_avg > path_avg + 0.05 and (path_min < min_clearance or direct_min >= path_min):
            print(
                f"[SIM] A* segment {segment_idx} using clearer direct segment "
                f"(astar min/avg={path_min:.2f}/{path_avg:.2f}, direct min/avg={direct_min:.2f}/{direct_avg:.2f})"
            )
            return direct_path
        return path

    def _direct_astar_segment_if_clear(self, path, start, goal):
        astar_config = self._astar_config()
        if astar_config.bev_mode != "local":
            return None
        planner = self._create_local_bev_astar_planner(start, goal, clearance_m=0.0)
        direct_path = planner.direct_world_path(start, goal, spacing=float(astar_config.waypoint.spacing_m) * 0.5)
        if not planner.is_world_path_collision_free(direct_path):
            return None
        if not self._astar_segment_reaches_goal(direct_path, goal, float(astar_config.segmentation.goal_tolerance_m)):
            return None
        return planner, direct_path

    def _smooth_astar_world_path(self, planner, path, plan_source):
        path = [np.asarray(point, dtype=float) for point in path]
        info = {
            "enabled": False,
            "source": plan_source,
            "input_points": int(len(path)),
            "output_points": int(len(path)),
            "shortcut_points": int(len(path)),
            "jitter_points": int(len(path)),
            "input_length_m": self._path_length_xy(path),
            "output_length_m": self._path_length_xy(path),
        }
        astar_config = self._astar_config()
        smoothing_config = getattr(astar_config, "smoothing", None)
        if (
            smoothing_config is None
            or not bool(getattr(smoothing_config, "enabled", False))
            or planner is None
            or plan_source == "navmesh_fallback"
            or len(path) < 3
        ):
            info["path"] = path
            return info

        info["enabled"] = True
        smoothed = path
        if bool(getattr(smoothing_config, "shortcut_enabled", True)):
            smoothed = self._shortcut_astar_world_path(planner, smoothed, smoothing_config)
        info["shortcut_points"] = int(len(smoothed))
        if bool(getattr(smoothing_config, "jitter_prune_enabled", True)):
            smoothed = self._prune_astar_jitter(planner, smoothed, smoothing_config)
        info["jitter_points"] = int(len(smoothed))
        info["output_points"] = int(len(smoothed))
        info["output_length_m"] = self._path_length_xy(smoothed)
        if info["output_points"] < info["input_points"]:
            print(
                "[SIM] A* smoothing "
                f"{info['input_points']}->{info['output_points']} points, "
                f"{info['input_length_m']:.2f}->{info['output_length_m']:.2f}m"
            )
        info["path"] = smoothed
        return info

    def _shortcut_astar_world_path(self, planner, path, smoothing_config):
        if len(path) < 3:
            return path
        max_span_m = float(getattr(smoothing_config, "max_shortcut_span_m", 8.0))
        check_spacing_m = float(getattr(smoothing_config, "shortcut_check_spacing_m", 0.25))
        min_clearance_m = float(getattr(smoothing_config, "min_shortcut_clearance_m", 0.0))
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            best_j = i + 1
            for j in range(len(path) - 1, i + 1, -1):
                if max_span_m > 0.0 and np.linalg.norm(path[j][:2] - path[i][:2]) > max_span_m:
                    continue
                if self._astar_direct_connection_ok(planner, path[i], path[j], check_spacing_m, min_clearance_m):
                    best_j = j
                    break
            out.append(path[best_j])
            i = best_j
        return out

    def _prune_astar_jitter(self, planner, path, smoothing_config):
        if len(path) < 3:
            return path
        max_passes = max(1, int(getattr(smoothing_config, "jitter_max_passes", 3)))
        max_chord_m = float(getattr(smoothing_config, "jitter_max_chord_m", 3.0))
        max_offset_m = float(getattr(smoothing_config, "jitter_max_offset_m", 0.45))
        max_turn_rad = np.deg2rad(float(getattr(smoothing_config, "jitter_max_turn_deg", 60.0)))
        check_spacing_m = float(getattr(smoothing_config, "shortcut_check_spacing_m", 0.25))
        min_clearance_m = float(getattr(smoothing_config, "min_shortcut_clearance_m", 0.0))
        points = list(path)
        for _ in range(max_passes):
            changed = False
            out = [points[0]]
            i = 1
            while i < len(points) - 1:
                prev = out[-1]
                cur = points[i]
                nxt = points[i + 1]
                chord = float(np.linalg.norm(nxt[:2] - prev[:2]))
                if chord <= max_chord_m:
                    offset = self._point_line_offset_xy(cur, prev, nxt)
                    turn = self._turn_angle_abs(prev, cur, nxt)
                    if (
                        offset <= max_offset_m
                        and turn <= max_turn_rad
                        and self._astar_direct_connection_ok(planner, prev, nxt, check_spacing_m, min_clearance_m)
                    ):
                        changed = True
                        i += 1
                        continue
                out.append(cur)
                i += 1
            out.append(points[-1])
            points = out
            if not changed:
                break
        return points

    def _astar_direct_connection_ok(self, planner, start, goal, spacing_m, min_clearance_m):
        direct_path = planner.direct_world_path(start, goal, spacing=max(0.05, float(spacing_m)))
        if not planner.is_world_path_collision_free(direct_path):
            return False
        if min_clearance_m > 0.0:
            min_clearance, _ = planner.world_path_clearance(direct_path)
            if min_clearance < min_clearance_m:
                return False
        return True

    @staticmethod
    def _point_line_offset_xy(point, start, goal):
        point = np.asarray(point, dtype=float)
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        segment = goal[:2] - start[:2]
        denom = float(np.linalg.norm(segment))
        if denom < 1e-9:
            return float(np.linalg.norm(point[:2] - start[:2]))
        return float(abs(np.cross(segment, point[:2] - start[:2]) / denom))

    @staticmethod
    def _turn_angle_abs(prev, cur, nxt):
        prev = np.asarray(prev, dtype=float)
        cur = np.asarray(cur, dtype=float)
        nxt = np.asarray(nxt, dtype=float)
        v1 = cur[:2] - prev[:2]
        v2 = nxt[:2] - cur[:2]
        if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
            return 0.0
        yaw1 = np.arctan2(v1[1], v1[0])
        yaw2 = np.arctan2(v2[1], v2[0])
        return float(abs((yaw2 - yaw1 + np.pi) % (2 * np.pi) - np.pi))

    def _prepare_astar_reference_points(self, ref_path):
        astar_config = self._astar_config()
        max_segment_m = float(astar_config.segmentation.max_segment_m)
        min_spacing_m = min(0.5, max_segment_m * 0.5)
        compact = []
        last_point = None
        for point in ref_path:
            point = np.asarray(point, dtype=float)
            if last_point is not None and np.linalg.norm(point[:2] - last_point[:2]) < min_spacing_m:
                continue
            compact.append(point)
            last_point = point
        if len(compact) < 2 or max_segment_m <= 0.0:
            return compact

        densified = [compact[0]]
        for start, goal in zip(compact[:-1], compact[1:]):
            distance = float(np.linalg.norm(goal[:2] - start[:2]))
            steps = max(1, int(np.ceil(distance / max_segment_m)))
            for step_idx in range(1, steps + 1):
                t = step_idx / steps
                densified.append(start + t * (goal - start))
        print(f"[SIM] A* reference points: {len(compact)} compact -> {len(densified)} subdivided")
        return densified

    def _astar_segment_reaches_goal(self, path, goal, tolerance):
        if path is None or len(path) == 0:
            return False
        return float(np.linalg.norm(np.asarray(path[-1])[:2] - np.asarray(goal)[:2])) <= tolerance

    def _resample_astar_world_path(self, path, spacing):
        path = [np.asarray(point, dtype=float) for point in path]
        if len(path) < 2 or spacing <= 0.0:
            return path
        resampled = [path[0]]
        distance_since_sample = 0.0
        current = path[0].copy()
        for start, goal in zip(path[:-1], path[1:]):
            current = start.copy()
            vec = goal - current
            distance = float(np.linalg.norm(vec[:2]))
            while distance > 1e-6:
                remaining = spacing - distance_since_sample
                if distance < remaining:
                    distance_since_sample += distance
                    break
                direction = vec / distance
                current = current + direction * remaining
                resampled.append(current.copy())
                distance_since_sample = 0.0
                vec = goal - current
                distance = float(np.linalg.norm(vec[:2]))
        if np.linalg.norm(resampled[-1][:2] - path[-1][:2]) > 1e-6:
            resampled.append(path[-1])
        return resampled

    def _create_global_bev_astar_planner(self, scene_id):
        astar_config = self._astar_config(scene_id)
        namespace = None
        if self.current_episode is not None:
            namespace = self.current_episode.get("episode_data_namespace")
        if not namespace:
            namespace = getattr(self, "episode_data_namespace", None)
        scene_data_dir = Path(self.args.scene_folder) / "episode_data"
        if namespace:
            scene_data_dir /= str(namespace)
        scene_data_dir /= scene_id
        bev_map_path = scene_data_dir / "nav_bev_map.npz"
        if not bev_map_path.exists():
            bev_map_path = scene_data_dir / "bev_map.npz"
        return BevAStarReferencePlanner(
            bev_map_path,
            max_climb_height=float(astar_config.planner.max_climb_height_m),
            robot_width=float(getattr(astar_config.planner, "robot_width_m", getattr(self.manager_env.cfg, "robot_width", 0.5))),
            robot_height=float(getattr(astar_config.planner, "robot_height_m", getattr(self.manager_env.cfg, "robot_height", 0.9))),
            step_size=float(astar_config.planner.step_size_m),
            step_size_yaw=int(astar_config.planner.step_size_yaw_deg),
            reach_threshold=float(astar_config.planner.reach_threshold_m),
            reach_yaw=float(astar_config.planner.reach_yaw_deg),
            clearance_m=float(astar_config.planner.clearance_m),
            preferred_clearance_m=float(astar_config.planner.preferred_clearance_m),
            clearance_cost=float(astar_config.planner.clearance_cost),
            min_obstacle_area_px=int(astar_config.planner.min_obstacle_area_px),
        )

    def _create_local_bev_astar_planner(self, start, goal, clearance_m=None):
        bev_depth, info = self._capture_local_bev_depth(start, goal)
        return self._create_local_bev_astar_planner_from_depth(bev_depth, info, clearance_m=clearance_m)

    def _create_local_bev_astar_planner_from_depth(self, bev_depth, info, clearance_m=None):
        astar_config = self._astar_config()
        return BevAStarReferencePlanner.from_bev_depth(
            bev_depth,
            info,
            max_climb_height=float(astar_config.planner.max_climb_height_m),
            robot_width=float(getattr(astar_config.planner, "robot_width_m", getattr(self.manager_env.cfg, "robot_width", 0.5))),
            robot_height=float(getattr(astar_config.planner, "robot_height_m", getattr(self.manager_env.cfg, "robot_height", 0.9))),
            step_size=float(astar_config.planner.step_size_m),
            step_size_yaw=int(astar_config.planner.step_size_yaw_deg),
            reach_threshold=float(astar_config.planner.reach_threshold_m),
            reach_yaw=float(astar_config.planner.reach_yaw_deg),
            clearance_m=float(astar_config.planner.clearance_m if clearance_m is None else clearance_m),
            preferred_clearance_m=float(astar_config.planner.preferred_clearance_m),
            clearance_cost=float(astar_config.planner.clearance_cost),
            min_obstacle_area_px=int(astar_config.planner.min_obstacle_area_px),
        )

    def _render_transient_camera(self, camera, render_frames):
        sim = self.manager_env.sim
        for _ in range(max(1, int(render_frames))):
            sim.forward()
            sim.render()
            camera._is_outdated[:] = True
            camera.update(0.0, force_recompute=True)

    def _destroy_transient_camera(self, camera):
        if camera is None:
            return
        try:
            annotators = getattr(camera, "_annotators", {})
            render_product_paths = list(getattr(camera, "render_product_paths", []))
            for annotator in annotators.values():
                annotator.detach(render_product_paths)
            annotators.clear()
        except Exception as exc:
            print(f"[SIM] Failed to detach transient camera annotators: {exc}")
        try:
            if hasattr(camera, "_clear_callbacks"):
                camera._clear_callbacks()
        except Exception as exc:
            print(f"[SIM] Failed to clear transient camera callbacks: {exc}")

    def _snapshot_robot_sim_state(self):
        robot = self.manager_env.scene["robot"]
        with torch.inference_mode(False):
            return {
                "root_state": robot.data.root_state_w.detach().clone(),
                "joint_pos": robot.data.joint_pos.detach().clone(),
                "joint_vel": robot.data.joint_vel.detach().clone(),
            }

    def _restore_robot_sim_state(self, state):
        robot = self.manager_env.scene["robot"]
        with torch.inference_mode():
            root_state = state["root_state"].to(device=robot.device, dtype=robot.data.root_state_w.dtype).clone()
            joint_pos = state["joint_pos"].to(device=robot.device, dtype=robot.data.joint_pos.dtype).clone()
            joint_vel = state["joint_vel"].to(device=robot.device, dtype=robot.data.joint_vel.dtype).clone()
            if hasattr(robot, "write_root_state_to_sim"):
                robot.write_root_state_to_sim(root_state)
            else:
                robot.write_root_pose_to_sim(root_state[:, :7])
                if hasattr(robot, "write_root_velocity_to_sim"):
                    robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_position_to_sim(joint_pos)
            robot.write_joint_velocity_to_sim(joint_vel)
            if hasattr(robot, "set_joint_position_target"):
                robot.set_joint_position_target(joint_pos)
            if hasattr(robot, "set_joint_velocity_target"):
                robot.set_joint_velocity_target(torch.zeros_like(joint_vel))
            robot.write_data_to_sim()

    def _capture_local_bev_depth(self, start, goal):
        astar_config = self._astar_config()
        return self._capture_local_bev_depth_for_points(
            [start, goal],
            padding_m=float(astar_config.local_bev.padding_m),
            min_window_m=float(astar_config.local_bev.window_m),
            resolution_m=float(astar_config.local_bev.resolution_m),
            max_window_m=0.0,
            log_label="local",
        )

    def _capture_local_bev_depth_for_points(
        self,
        points,
        padding_m=None,
        min_window_m=None,
        resolution_m=None,
        max_window_m=None,
        log_label="local",
    ):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        astar_config = self._astar_config()
        padding = float(astar_config.local_bev.padding_m if padding_m is None else padding_m)
        min_window = float(astar_config.local_bev.window_m if min_window_m is None else min_window_m)
        resolution = float(astar_config.local_bev.resolution_m if resolution_m is None else resolution_m)
        camera_height = float(astar_config.local_bev.camera_height_m)
        render_frames = max(1, int(astar_config.local_bev.render_frames))
        xy_min = np.min(points[:, :2], axis=0)
        xy_max = np.max(points[:, :2], axis=0)
        center_xy = (xy_min + xy_max) / 2.0
        path_extent = xy_max - xy_min
        window_m = max(min_window, float(path_extent.max() + 2.0 * padding))
        max_window = float(0.0 if max_window_m is None else max_window_m)
        if max_window > 0.0 and window_m > max_window:
            print(
                f"[SIM] Skipping {log_label} BEV for A*: requested window={window_m:.2f}m "
                f"exceeds max={max_window:.2f}m"
            )
            return None
        image_size = max(32, int(np.ceil(window_m / resolution)))
        if image_size > 1024:
            image_size = 1024
            resolution = window_m / image_size
        px_per_meter = 1.0 / resolution
        z_ref = float(np.max(points[:, 2])) if points.shape[1] > 2 else 0.0
        camera_pos = np.array([center_xy[0], center_xy[1], z_ref + camera_height], dtype=float)
        try:
            cam_z = float(self.env.get_cam_pose()[0][2])
        except Exception:
            cam_z = z_ref + 0.8
        clipping_range_min = max(0.1, camera_pos[2] - (cam_z + 0.3))
        clipping_range_max = clipping_range_min + 1000.0
        prim_path = "/World/LocalAStarBEVCamera"
        robot_state = self._snapshot_robot_sim_state()
        robot_prim = self.manager_env.scene.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        robot_hidden = False
        camera = None
        try:
            if robot_prim.IsValid():
                prim_utils.set_prim_visibility(robot_prim, False)
                robot_hidden = True
            camera = self.env.create_camera(
                prim_path=prim_path,
                perspective=False,
                pos=camera_pos.tolist(),
                quat_opengl=[1, 0, 0, 0],
                horizontal_aperture=window_m * 10.0,
                clipping_range_min=clipping_range_min,
                clipping_range_max=clipping_range_max,
                width=image_size,
                height=image_size,
            )
            self._render_transient_camera(camera, render_frames)
            depth = camera.data.output["distance_to_image_plane"].detach().cpu().numpy()[0]
            rgb_output = camera.data.output.get("rgb")
            rgb = None if rgb_output is None else rgb_output.detach().cpu().numpy()[0]
        finally:
            self._destroy_transient_camera(camera)
            if robot_hidden:
                prim_utils.set_prim_visibility(robot_prim, True)
            self._restore_robot_sim_state(robot_state)
            try:
                if prim_utils.get_prim_at_path(prim_path):
                    prim_utils.delete_prim(prim_path)
            except Exception as exc:
                print(f"[SIM] Failed to delete local A* BEV camera {prim_path}: {exc}")
        info = {
            "px_per_meter": px_per_meter,
            "world_center": [float(center_xy[0]), float(center_xy[1]), z_ref],
            "world_size": [window_m, window_m, camera_height],
            "world_bb": [
                float(center_xy[0] - window_m / 2.0),
                float(center_xy[1] - window_m / 2.0),
                z_ref,
                float(center_xy[0] + window_m / 2.0),
                float(center_xy[1] + window_m / 2.0),
                z_ref + camera_height,
            ],
            "image_width": image_size,
            "image_height": image_size,
        }
        print(
            f"[SIM] Captured {log_label} BEV for A*: center=({center_xy[0]:.2f}, {center_xy[1]:.2f}), "
            f"window={window_m:.2f}m, resolution={resolution:.3f}m/px, size={image_size}px"
        )
        self._save_astar_bev_capture_debug(depth, rgb, info, log_label)
        return depth, info

    def _astar_debug_root(self):
        astar_config = self._astar_config()
        debug_root = Path(astar_config.debug.dir)
        episode_label = self.current_episode.episode_label if self.current_episode is not None else "unknown_episode"
        return debug_root / episode_label

    def _save_astar_bev_capture_debug(self, depth, rgb, info, label):
        astar_config = self._astar_config()
        if not bool(astar_config.debug.save_visualizations):
            return
        debug_root = self._astar_debug_root()
        debug_root.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in str(label)).strip("_") or "bev"
        stem = f"{self.astar_capture_debug_index:03d}_{safe_label}"

        depth_img = np.asarray(depth)
        if depth_img.ndim == 3:
            depth_img = depth_img[..., 0]
        valid = np.isfinite(depth_img) & (depth_img > 0)
        if np.any(valid):
            lo, hi = np.percentile(depth_img[valid], [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            depth_u8 = np.clip((depth_img - lo) / (hi - lo), 0.0, 1.0)
            depth_u8 = (depth_u8 * 255.0).astype(np.uint8)
            depth_u8[~valid] = 0
        else:
            depth_u8 = np.zeros(depth_img.shape[:2], dtype=np.uint8)
        depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(str(debug_root / f"{stem}_depth.png"), depth_vis)

        if rgb is not None:
            rgb_img = np.asarray(rgb)
            if rgb_img.dtype != np.uint8:
                if rgb_img.max(initial=0) <= 1.0:
                    rgb_img = rgb_img * 255.0
                rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
            rgb_img = rgb_img[..., :3]
            cv2.imwrite(str(debug_root / f"{stem}_rgb.png"), rgb_img[..., ::-1])

        with (debug_root / f"{stem}_info.json").open("w") as f:
            json.dump(info, f, indent=2)
        self.astar_capture_debug_index += 1

    def _save_astar_path_visualization(self, planner, path, segment_idx, mode):
        astar_config = self._astar_config()
        if not bool(astar_config.debug.save_visualizations):
            return
        debug_root = self._astar_debug_root()
        debug_root.mkdir(parents=True, exist_ok=True)
        obstacle_map = planner.obstacle_map
        vis = np.zeros((*obstacle_map.shape, 3), dtype=np.uint8)
        vis[:, :] = 245
        vis[obstacle_map > 0] = (35, 35, 35)
        if len(path) > 0:
            pts = []
            for point in path:
                x, y, _ = planner.world_to_grid(point)
                pts.append([int(round(x)), int(round(y))])
            pts = np.asarray(pts, dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(vis, [pts], False, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(vis, tuple(pts[0]), 5, (0, 180, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, tuple(pts[-1]), 5, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        text = f"{mode} segment {segment_idx} points={len(path)}"
        cv2.putText(vis, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        out_path = debug_root / f"{self.astar_visualization_index:03d}_{mode}_segment_{segment_idx:02d}.png"
        cv2.imwrite(str(out_path), vis)
        print(f"[SIM] Saved A* path visualization: {out_path}")
        self.astar_visualization_index += 1

    def _save_astar_world_path(self, total_path, waypoints, mode):
        astar_config = self._astar_config()
        if not bool(astar_config.debug.save_visualizations):
            return
        debug_root = self._astar_debug_root()
        debug_root.mkdir(parents=True, exist_ok=True)
        out_path = debug_root / f"{mode}_world_path.json"
        payload = {
            "episode_label": self.current_episode.episode_label if self.current_episode is not None else None,
            "mode": mode,
            "path": np.asarray(total_path, dtype=float).tolist(),
            "waypoints": waypoints,
        }
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"[SIM] Saved A* world path: {out_path}")

    def set_ref_waypoints(self, episode):
        self.waypoint_path_planner = None
        self.last_astar_world_path = None
        self.last_astar_waypoints = None
        self.last_astar_plan_info = {}
        measure_manager = add_measurement(self.env, episode, measure_names=["DistanceToGoal", "ClosestGoal"])
        measure_manager.update_measures()
        measurements = measure_manager.get_measurements()
        closest_goal_idx = measurements["closest_goal"]
        episode["closest_goal_idx"] = int(closest_goal_idx)
        goal = episode["goals"][closest_goal_idx]
        ref_path = goal["reference_path"]
        self._configure_astar_follower_for_scene(episode["scene_id"])
        self._apply_astar_goal_stop_radius(goal.get("radius"))
        # notice that ref_path has [x, y, z], while waypoints has [x, y, yaw]
        preserve_reference_path = bool(episode.get("task_generation_preserve_reference_path", False))
        if preserve_reference_path or goal.get("reference_path_planner") == "astar":
            waypoints = self._reference_path_to_waypoints(ref_path)
            self.waypoint_path_planner = "astar"
            self.last_astar_world_path = np.asarray(ref_path, dtype=float).tolist()
            self.last_astar_waypoints = [list(waypoint) for waypoint in waypoints]
            self.last_astar_plan_info = {
                "scene_id": episode["scene_id"],
                "source": "stored_reference" if preserve_reference_path else "stored_astar",
                "fallback": False,
                "waypoints": int(len(waypoints)),
                "path_length_m": self._path_length_xy(ref_path),
            }
        elif getattr(self.args, "trajectory_planner", None) == "astar":
            guide_path = goal.get("source_reference_path", ref_path)
            waypoints = self.generate_astar_path(
                episode["scene_id"],
                guide_path,
                goal_position=goal.get("location"),
                goal_radius=goal.get("radius"),
            )
            self.waypoint_path_planner = "astar"
        else:
            waypoints = [[p[0], p[1], np.inf] for p in ref_path[1:]]
            self.waypoint_path_planner = "navmesh"
        # print current waypoint
        self.set_waypoints(waypoints)
        self.configure_controlled_waypoint_follower()
    
    def clear_waypoint_visuals(self):
        if not self.visualize_waypoints:
            return
        self.env.remove_prim("/World/WaypointPath")
        self.env.remove_prim("/World/Arrow")
        self.remove_callback('pre_step', self.clear_waypoint_visuals)

    def clear_waypoints(self):
        self.waypoints = None
        if self.command_dim == 7:
            body_pos, body_quat = self.env.get_body_pose()
            self.commands[self.robot_index] = torch.tensor(
                [*body_pos, *body_quat],
                device=self.device,
                dtype=self.commands.dtype,
            )
        else:
            self.commands[self.robot_index] = torch.zeros(self.command_dim, device=self.device)
        if self.visualize_waypoints:
            self.add_callback('pre_step', self.clear_waypoint_visuals)

    def _set_drone_waypoint_command(self, body_pos, body_quat):
        wp_idx = min(self.waypoint_follower.current_wp_idx, len(self.waypoints) - 1)
        target_wp = np.asarray(self.waypoints[wp_idx], dtype=float)
        current_xy = np.asarray(body_pos[:2], dtype=float)
        delta_xy = target_wp[:2] - current_xy
        distance = float(np.linalg.norm(delta_xy))
        step_dt = float(getattr(self.manager_env, "step_dt", 0.05) or 0.05)
        max_speed = float(getattr(self.manager_env.cfg, "max_horizontal_speed", 2.5))
        max_step = max_speed * step_dt
        next_xy = target_wp[:2]
        if distance > max_step and distance > 1e-9:
            next_xy = current_xy + delta_xy * (max_step / distance)
        target_yaw = float(target_wp[2])
        if target_yaw > 10.0:
            target_yaw = calc_yaw(current_xy, target_wp[:2])
        current_yaw = float(R.from_quat(body_quat).as_euler("xyz")[2])
        yaw_error = (target_yaw - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
        max_yaw_step = float(getattr(self.manager_env.cfg, "max_yaw_speed", 4.0)) * step_dt
        next_yaw = current_yaw + float(np.clip(yaw_error, -max_yaw_step, max_yaw_step))
        target_quat = R.from_euler("z", next_yaw).as_quat()
        self.commands[self.robot_index] = torch.tensor(
            [float(next_xy[0]), float(next_xy[1]), float(body_pos[2]), *target_quat],
            device=self.device,
            dtype=self.commands.dtype,
        )
    
    def follow_waypoints(self):
        body_pos, body_quat = self.env.get_body_pose()
        if self.command_dim == 3 and self._should_ramp_astar_command():
            astar_config = self._astar_config()
            settle_frames = max(0, int(astar_config.execution.start_settle_frames))
            if self.waypoint_command_frames < settle_frames:
                self.commands[self.robot_index] = torch.zeros(self.command_dim, device=self.device)
                self.waypoint_command_frames += 1
                return
        waypoint_command = self.waypoint_follower.update(body_pos, body_quat, self.waypoints, verbose=False)
        if self.command_dim == 3:
            if self._should_ramp_astar_command():
                astar_config = self._astar_config()
                ramp_frames = max(1, int(astar_config.execution.command_ramp_frames))
                min_scale = float(astar_config.execution.command_ramp_min_scale)
                settle_frames = max(0, int(astar_config.execution.start_settle_frames))
                ramp_step = max(0, self.waypoint_command_frames - settle_frames)
                progress = min(1.0, (ramp_step + 1) / ramp_frames)
                scale = min_scale + (1.0 - min_scale) * progress
                waypoint_command = waypoint_command * scale
            self.waypoint_command_frames += 1
            self.commands[self.robot_index] = waypoint_command
        elif self.robot_name == "drone" and self.waypoints is not None and len(self.waypoints) > 0:
            self._set_drone_waypoint_command(body_pos, body_quat)
        elif self.command_dim == 7 and self.waypoints is not None and len(self.waypoints) > 0:
            wp_idx = min(self.waypoint_follower.current_wp_idx, len(self.waypoints) - 1)
            target_wp = np.asarray(self.waypoints[wp_idx], dtype=float)
            target_yaw = float(target_wp[2])
            if target_yaw > 10.0:
                target_yaw = calc_yaw(np.asarray(body_pos[:2]), target_wp[:2])
            target_quat = R.from_euler("z", target_yaw).as_quat()
            transform = torch.tensor(
                [[float(target_wp[0]), float(target_wp[1]), float(body_pos[2]), *target_quat]],
                device=self.device,
                dtype=self.commands.dtype,
            )
            self.commands[self.robot_index] = transform
            if self.robot_name == "tidybot":
                self._tidybot_base_command_active = True
        if self.waypoint_follower.arrived_at_goal:
            if self._uses_internal_task_generation_reference():
                self._handle_stop_command()
                self.commands_source = "task_generation_internal_stop"
            else:
                self.clear_waypoints()
        elif self.visualize_waypoints:
            points = [body_pos]+[[wp[0], wp[1], body_pos[2]] for wp in self.waypoints[self.waypoint_follower.current_wp_idx:]]
            remaining_waypoints = np.array(self.waypoints[self.waypoint_follower.current_wp_idx:])
            visualize_curve(points, prim_path="/World/WaypointPath", width=0.02)
            if (remaining_waypoints[:,2]<10.0).all():
                visualize_arrow(remaining_waypoints, body_pos[2], prim_path="/World/Arrow", color=(0.0, 1.0, 0.0), scale=(0.2, 0.2, 0.2))

    def follow_timed_trajectory(self):
        trajectory = self.timed_trajectory
        if trajectory is None:
            return
        current_step = int(getattr(self.manager_env, "common_step_counter", 0) or 0)
        target_index = trajectory_tick_index(
            current_step, trajectory["start_step"], len(trajectory["waypoints"])
        )
        if target_index is None:
            self._clear_timed_trajectory("completed")
            if self.command_dim == 3:
                self.commands[self.robot_index] = torch.zeros(
                    self.command_dim, device=self.device
                )
            else:
                self._tidybot_base_command_active = False
            self.commands_source = "server_timed_trajectory_complete"
            return
        target = trajectory["waypoints"][target_index]
        body_pos, body_quat = self.env.get_body_pose()
        if self.robot_name == "tidybot":
            current_yaw = float(R.from_quat(body_quat).as_euler("xyz")[2])
            next_pose = self.tidybot_timed_trajectory_follower.update(
                np.array([body_pos[0], body_pos[1], current_yaw], dtype=float),
                target,
                trajectory["dt"],
            )
            if self.env.control_mode == "physics_velocity":
                self.commands[self.robot_index] = torch.tensor(
                    self.tidybot_timed_trajectory_follower.previous_command,
                    device=self.device,
                    dtype=self.commands.dtype,
                )
                self._tidybot_base_command_active = True
                return
            target_quat = R.from_euler("z", next_pose[2]).as_quat()
            self.commands[self.robot_index] = torch.tensor(
                [next_pose[0], next_pose[1], body_pos[2], *target_quat],
                device=self.device,
                dtype=self.commands.dtype,
            )
            self._tidybot_base_command_active = True
            return
        command = self.timed_trajectory_follower.update(
            body_pos, body_quat, [target], verbose=False
        )
        self.commands[self.robot_index] = command.reshape(-1)

    def _should_ramp_astar_command(self):
        if getattr(self, "waypoint_path_planner", None) != "astar":
            return False
        astar_config = self._astar_config()
        if int(astar_config.execution.command_ramp_frames) <= 0:
            return False
        return True

    def apply_pending_transform(self):
        if self.pending_transform_command is None:
            return
        transform = self.pending_transform_command
        self.pending_transform_command = None
        if self.command_dim == 7:
            self.commands[self.robot_index] = transform
            self._tidybot_base_command_active = True
            return
        pose = transform.detach().cpu().numpy().astype(np.float32)
        current_pos, _ = self.env.get_body_pose()
        if pose[2] == 0.0:
            pose[2] = current_pos[2]
        elif self.transform_root_z_offset > 0.0:
            pose[2] += self.transform_root_z_offset
        self.env._write_robot_pose(pose)
        self.commands[self.robot_index] = torch.zeros(self.command_dim, device=self.device)

    def _reset_if_requested(self):
        with self.load_env_lock:
            if not self.reset_flag:
                return False
            print(f"[SIM] Resetting env state to {self.next_episode.episode_label}..")
            self.timed_trajectory = None
            self.last_timed_trajectory_status = None
            self.tidybot_timed_trajectory_follower.reset()
            self.sim_state = "loading"
            self.env.reset_idx([self.robot_index])
            obs, info = self.env.reset(self.next_episode, self.warmup_steps)
            if isinstance(info, dict):
                info.pop("terminations", None)
            self.obs_index = 0
            self.reset_flag = False
            self.current_episode = self.next_episode
            self.next_episode = None
            if self._uses_internal_task_generation_reference():
                self._configure_internal_task_generation_timeout(self.current_episode)
                self.set_ref_waypoints(self.current_episode)
                self.commands_source = "task_generation_internal_reference"
            self.update_viewer()
            self.info = info
            self.obs = obs
            self.reward = None
            self.done = None
            self.sim_state = "running"
            self._reset_tidybot_keyboard_targets()
            self.reset_telemetry_baseline()
            self.waiting_for_first_action = bool(self.args.on_demand_render and not getattr(self.args, "gui_high_fps", False))
            if self._uses_internal_task_generation_reference():
                self.waiting_for_first_action = False
            if self.args.on_demand_render and not getattr(self.args, "gui_high_fps", False):
                with self.sim_step_lock:
                    self._hold_tidybot_arm_pose_if_available()
                    self.update_on_demand_sensor_data()
                    if self.env.scene_render_warmup_required:
                        self.update_on_demand_sensor_data()
                        self.env.scene_render_warmup_required = False
                    self._hold_tidybot_arm_pose_if_available()
            self.update_obs(obs, info)
            self.call_callbacks('reset_finished')
            return True

    def _finish_step(self, profile_total_start):
        profile_start = time.perf_counter()
        self.call_callbacks('step_finished')
        self._profile_record("step/step_finished_callbacks", time.perf_counter() - profile_start)
        self.profile_step_count += 1
        self._profile_record("step/total", time.perf_counter() - profile_total_start)
        self._profile_summary()

    def _run_pre_step_callbacks(self):
        profile_start = time.perf_counter()
        self.call_callbacks('pre_step')
        self._profile_record("step/pre_step_callbacks", time.perf_counter() - profile_start)

    def _wait_for_first_action_if_needed(self):
        if not self.waiting_for_first_action:
            return False
        self._hold_tidybot_arm_pose_if_available()
        time.sleep(0.01)
        return True

    def _hold_tidybot_arm_pose_if_available(self):
        if self.robot_name == "tidybot" and self.env is not None:
            hold = getattr(self.env, "_hold_tidybot_arm_pose", None)
            if hold is not None:
                hold()

    def _update_step_command(self):
        profile_start = time.perf_counter()
        self.apply_pending_transform()
        self._apply_tidybot_keyboard_controls()
        if self.timed_trajectory is not None:
            self.follow_timed_trajectory()
        elif self.waypoints is not None and len(self.waypoints) > 0:
            self.follow_waypoints()
        if self.robot_name == "turtlebot":
            self.commands[:, 1] = 0.0
        self._profile_record("step/command_update", time.perf_counter() - profile_start)

    def _step_command(self):
        if self.robot_name == "tidybot":
            if self.env.control_mode == "physics_velocity":
                self._tidybot_base_command_active = False
                return self.commands
            command = {}
            if self._tidybot_base_command_active:
                command["base_transform"] = self.commands[self.robot_index].detach().clone()
                self._tidybot_base_command_active = False
            if self.arm_command is not None:
                command.update(self.arm_command)
            return command
        return self.commands

    def _apply_camera_pose_fix(self, obs):
        profile_start = time.perf_counter()
        if not self.args.disable_camera:
            pos_cam_world, quat_cam_world = self.env.get_cam_pose()
            obs['pov_pose'] = torch.tensor(np.concatenate([pos_cam_world, quat_cam_world]).reshape(1,-1), device=self.device)
        self._profile_record("step/camera_pose_fix", time.perf_counter() - profile_start)

    def _store_step_result(self, obs, reward, done, info):
        self.info = info
        self.obs = obs
        self.reward = reward
        self.done = done
        with self.server_data_lock:
            profile_start = time.perf_counter()
            if "terminations" in info:
                self.sim_state = "terminated"
                self._latest_data["info"] = self._build_info_payload(info)
            else:
                self.sim_state = "running"
                self.update_obs(obs, info)
            self._profile_record("step/update_obs", time.perf_counter() - profile_start)

    def _advance_env_step(self):
        with self.sim_step_lock:
            profile_start = time.perf_counter()
            carb.profiler.begin(1, "NavVerse.step/env_step")
            try:
                obs, reward, done, info = self.env.step(self._step_command())
            finally:
                carb.profiler.end(1)
            self._profile_record("step/env_step", time.perf_counter() - profile_start)
            self._apply_camera_pose_fix(obs)
            self._store_step_result(obs, reward, done, info)

    def step(self):
        if self.closed:
            return
        profile_total_start = time.perf_counter()
        carb.profiler.begin(1, "NavVerse.step")
        try:
            with torch.inference_mode():
                profile_start = time.perf_counter()
                carb.profiler.begin(1, "NavVerse.step/reset_if_requested")
                try:
                    reset_completed = self._reset_if_requested()
                finally:
                    carb.profiler.end(1)
                self._profile_record("step/reset", time.perf_counter() - profile_start)
                if reset_completed:
                    self._finish_step(profile_total_start)
                    return

                self._run_pre_step_callbacks()
                if self._wait_for_first_action_if_needed():
                    self._finish_step(profile_total_start)
                    return

                self._update_step_command()
                self._advance_env_step()
                self._finish_step(profile_total_start)
        finally:
            carb.profiler.end(1)
    
    def _build_info_payload(self, info):
        current_episode = self.current_episode
        payload_info = {
            "scene_id": current_episode["scene_id"],
            "episode_id": current_episode["episode_id"],
            "metrics": info["measurements"],
            "gt_goal_position": self.current_episode["goals"][self.current_episode["closest_goal_idx"]]["location"],
        }
        if "terminations" in info:
            payload_info["terminations"] = info["terminations"]
        payload_info["instruction_type"] = self.args.instruction_type
        payload_info["target_type"] = episode_target_type(current_episode)
        payload_info["instruction"] = episode_instruction(current_episode, self.args.instruction_type)
        return payload_info

    def _camera_hfov_deg(self):
        camera_cfg = self.manager_env.scene["pov_camera"].cfg.spawn
        return 2 * np.arctan(camera_cfg.horizontal_aperture / 2.0 / camera_cfg.focal_length) * 180.0 / np.pi

    def _camera_tf_dict(self):
        def camera_tf(camera_name):
            sensor_cfg = self.manager_env.scene[camera_name].cfg
            pos = sensor_cfg.offset.pos
            quat = sensor_cfg.offset.rot
            quat_xyzw = [*quat[1:], *quat[:1]]
            return [*pos, *quat_xyzw]

        def extract_xyz(pose):
            return {key: pose[i] for i, key in enumerate(["x", "y", "z"])}

        def extract_quat(pose):
            return {key: pose[i + 3] for i, key in enumerate(["x", "y", "z", "w"])}

        return {
            "pov": {
                "parent_frame": "body",
                "child_frame_id": "pov",
                "transform": {
                    "translation": extract_xyz(camera_tf("pov_camera")),
                    "rotation": extract_quat(camera_tf("pov_camera")),
                },
            },
            "third_person": {
                "parent_frame": "body",
                "child_frame_id": "third_person",
                "transform": {
                    "translation": extract_xyz(camera_tf("third_person_camera")),
                    "rotation": extract_quat(camera_tf("third_person_camera")),
                },
            },
        }

    def _depth_to_uint16(self, depth):
        depth = depth.cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
        return np.clip(depth, 0, 65535).astype(np.uint16)

    def set_on_demand_camera_names(self, camera_names):
        self.on_demand_camera_names = tuple(camera_names)
        self._sync_on_demand_render_products()

    def _copy_camera_data_from_sensors(self, camera_names):
        rgb = {}
        depth = {}
        for camera_name in camera_names:
            sensor = self.manager_env.scene.sensors[camera_name]
            output = sensor.data.output
            output_name = (
                "pov" if camera_name == "pov_camera" else "third_person"
            )
            if "rgb" in output:
                rgb[output_name] = (
                    output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
                )
            if "distance_to_image_plane" in output:
                depth[output_name] = self._depth_to_uint16(
                    output["distance_to_image_plane"][0, :, :, 0]
                )
        self._latest_data["rgb"] = rgb
        self._latest_data["depth"] = depth
        if "pov_camera" in camera_names:
            self.cam_frame = self.manager_env.scene.sensors[
                "pov_camera"
            ].frame.clone()

    def update_on_demand_sensor_data(self, camera_names=None):
        if self.env is None or self.manager_env is None or self.current_episode is None:
            return
        profile_total_start = time.perf_counter()
        sim = self.manager_env.sim
        camera_names = tuple(camera_names or self.on_demand_camera_names)
        sensors = [
            self.manager_env.scene.sensors[name]
            for name in camera_names
            if name in self.manager_env.scene.sensors
        ]
        if not sensors:
            return
        profile_start = time.perf_counter()
        sim.forward()
        self._profile_record("render/sim_forward", time.perf_counter() - profile_start)
        profile_start = time.perf_counter()
        sim.render()
        self._profile_record("render/sim_render", time.perf_counter() - profile_start)
        profile_start = time.perf_counter()
        for sensor in sensors:
            sensor._is_outdated[:] = True
            sensor.update(0.0, force_recompute=True)
        self._profile_record("render/sensor_update", time.perf_counter() - profile_start)
        profile_start = time.perf_counter()
        self._copy_camera_data_from_sensors(camera_names)
        self._profile_record("render/copy_camera_data", time.perf_counter() - profile_start)
        self._latest_data["timestamp"] = time.time_ns()
        self.call_callbacks('on_render')
        self._profile_record("render/total", time.perf_counter() - profile_total_start)

    def request_on_demand_sensor_data(self, timeout_s=5.0):
        with self.on_demand_render_lock:
            if not self.on_demand_render_pending:
                self.on_demand_render_pending = True
                self.on_demand_render_event = Event()
                self.on_demand_render_error = None
            event = self.on_demand_render_event
        if event is None or not event.wait(timeout_s):
            return "Timed out waiting for on-demand render on simulation thread"
        with self.on_demand_render_lock:
            return self.on_demand_render_error

    def service_on_demand_render_request(self):
        if not self.args.on_demand_render:
            return False
        with self.on_demand_render_lock:
            if not self.on_demand_render_pending:
                return False
            event = self.on_demand_render_event
        error = None
        try:
            with self.sim_step_lock:
                self.update_on_demand_sensor_data()
        except Exception as exc:
            error = f"On-demand render failed: {exc}"
            print(error)
            import traceback
            traceback.print_exc()
        with self.on_demand_render_lock:
            self.on_demand_render_error = error
            self.on_demand_render_pending = False
            self.on_demand_render_event = None
            if event is not None:
                event.set()
        return error is None

    def update_obs(self, obs, info):
        if self.args.disable_camera or self.args.on_demand_render:
            should_update = True
        else:
            should_update = self.cam_frame != self.manager_env.scene.sensors['pov_camera'].frame
        if not should_update:
            return
        if not self.args.disable_socket_server:
            try:
                if self.args.disable_camera:
                    self._latest_data["rgb"] = {
                        "pov": np.zeros((1, 1, 3), dtype=np.uint8),
                    }
                    self._latest_data["depth"] = {
                        "pov": np.zeros((1, 1), dtype=np.uint16),
                    }
                elif not self.args.on_demand_render:
                    self._latest_data["rgb"] = {
                        "pov": obs['pov_rgb'][0, :, :, :3].cpu().numpy().astype(np.uint8),
                        "third_person": obs['third_person_rgb'][0, :, :, :3].cpu().numpy().astype(np.uint8),
                    }
                    self._latest_data["depth"] = {
                        "pov": self._depth_to_uint16(obs['pov_depth'][0, :, :, 0]),
                        "third_person": self._depth_to_uint16(obs['third_person_depth'][0, :, :, 0]),
                    }
                    self.cam_frame = self.manager_env.scene.sensors['pov_camera'].frame.clone()
                pos_robot, quat_robot = self.env.get_body_pose()
                self._latest_data["position"] = pos_robot
                self._latest_data["quat_xyzw"] = quat_robot
                self._latest_data["timestamp"] = time.time_ns()
                has_camera_metadata = not self.args.disable_camera
                info_payload = {
                    "intrinsic": None,
                    "robot_height": self.env.robot_height,
                    "hfov_deg": self._camera_hfov_deg() if has_camera_metadata else None,
                    "tfs": self._camera_tf_dict() if has_camera_metadata else {},
                    "sim_step": int(getattr(self.manager_env, "common_step_counter", 0) or 0),
                    "step_dt": float(getattr(self.manager_env, "step_dt", 0.0) or 0.0),
                }
                info_payload["sim_time"] = info_payload["sim_step"] * info_payload["step_dt"]
                info_payload.update(self._build_info_payload(info))
                self._latest_data["info"] = info_payload
            except Exception as e:
                print(f"Error updating obs: {e}")
                import traceback
                traceback.print_exc()

        self.obs_index += 1
        # save pcd
        # self.save_debug_data(obs)
        if self.visualizer is not None:
            self.visualize_obs(obs)
    
    def save_debug_data(self, obs):
        if self.obs_index%10==0:
            data = {
                "rgb": obs['pov_rgb'][0, :, :, :3].cpu().numpy().astype(np.uint8),
                "depth": obs['pov_depth'][0, :, :, 0].cpu().numpy(),
                "pose": obs['pov_pose'].cpu().numpy(),
            }
            np.savez(f"data/obs_{self.obs_index//10}.npz", **data)
    
    def visualize_obs(self, obs):
        if not self.visualizer.listener.has_subscribers() and self.obs_index>1:
            return
        time_start = time.time()
        rgb = obs['pov_rgb'][0, :, :, :3].cpu().numpy().astype(np.uint8)
        depth = obs['pov_depth'][0, :, :, 0].cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        pose = obs['pov_pose'][self.robot_index].cpu().numpy()
        pose_matrix = np.eye(4)
        pose_matrix[:3, 3] = pose[:3]
        pose_matrix[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        # convert pose: x (forward), y (left), z (up) -> x (right), y (down), z (forward)
        S = np.eye(4)
        S[:3, :3] = np.array(
            [[0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]]
        )
        pose_matrix = S @ pose_matrix @ S.T
        # log tf, rgb, depth
        self.visualizer.log({
            'type': 'tf',
            'channel_name': 'robot_tf',
            'pose': pose_matrix,
            'parent_frame_id': 'world',
            'child_frame_id': 'robot',
        })
        self.visualizer.log({
            'type': 'image',
            'channel_name': 'obs_rgb',
            'image': rgb,
            'frame_id': 'robot',
        })
        self.visualizer.log({
            'type': 'image',
            'channel_name': 'obs_depth',
            'image': depth,
            'frame_id': 'robot',
        })
        # log point cloud
        if self.obs_index%5==0:
            rgb_image = o3d.geometry.Image(rgb)
            depth_image = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, convert_rgb_to_intensity=False, depth_scale=1., depth_trunc=10000.)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=rgb.shape[1],
                height=rgb.shape[0],
                fx=733,
                fy=733,
                cx=rgb.shape[1]//2,
                cy=rgb.shape[0]//2,
            )
            obs_pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            obs_pc = o3d.geometry.PointCloud.voxel_down_sample(obs_pc, voxel_size=0.03)
            points = np.asarray(obs_pc.points)
            colors = np.asarray(obs_pc.colors)
            self.visualizer.log({
                'type': 'point_cloud',
                'channel_name': 'obs_pc',
                'points': points,
                'color': colors,
                'pose': pose_matrix,
                'frame_id': 'world',
            })
            print(f"Visualize obs time: {time.time() - time_start}")

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        try:
            self._input = carb.input.acquire_input_interface()
            app_window = omni.appwindow.get_default_app_window()
            self._keyboard = app_window.get_keyboard() if app_window is not None else None
        except RuntimeError as exc:
            if self.args.headless:
                print(f"[SIM] Keyboard input unavailable in headless mode; skipping setup: {exc}")
                self._input = None
                self._keyboard = None
                self._sub_keyboard = None
                self._pressed_control_keys = set()
                self._keyboard_control_keys = set()
                return
            raise
        if self._keyboard is None:
            print("[SIM] Keyboard window unavailable; skipping setup")
            self._sub_keyboard = None
            self._pressed_control_keys = set()
            self._keyboard_control_keys = set()
            return
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._pressed_control_keys = set()
        if self.robot_name == "turtlebot":
            T = 0.8
            R = 2.0
        else:
            T = 2
            R = 2
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0], device=self.device), # forward
            "DOWN": torch.tensor([-T, 0.0, 0.0], device=self.device), # backward
            "LEFT": torch.tensor([0, 0.0, R], device=self.device), # turn left
            "RIGHT": torch.tensor([0, 0.0, -R], device=self.device), # turn right
            "W": torch.tensor([T, 0.0, 0.0], device=self.device), # forward
            "S": torch.tensor([-T, 0.0, 0.0], device=self.device), # backward
            "A": torch.tensor([0, 0.5*T, 0.0], device=self.device), # left
            "D": torch.tensor([0, -0.5*T, 0.0], device=self.device), # right
            "Q": torch.tensor([0, 0.0, R], device=self.device), # turn left
            "E": torch.tensor([0, 0.0, -R], device=self.device), # turn right
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device),
        }
        self._keyboard_control_keys = set(self._key_to_control)
        if self.robot_name == "tidybot":
            self._keyboard_control_keys |= self._tidybot_keyboard_keys

    @staticmethod
    def _normalise_keyboard_input(raw_input):
        key_name = getattr(raw_input, "name", str(raw_input))
        key_name = key_name.rsplit(".", 1)[-1].upper()
        if key_name.startswith("KEY_"):
            key_name = key_name[4:]
        return key_name

    @staticmethod
    def _yaw_from_quat_xyzw(quat):
        return float(R.from_quat(quat).as_euler("ZYX")[0])

    @staticmethod
    def _quat_xyzw_from_yaw(yaw):
        return np.array([0.0, 0.0, np.sin(yaw * 0.5), np.cos(yaw * 0.5)], dtype=np.float32)

    def _keyboard_axis(self, positive, negative):
        return float(positive in self._pressed_control_keys) - float(negative in self._pressed_control_keys)

    def _reset_tidybot_keyboard_targets(self):
        self._tidybot_keyboard_base_pos = None
        self._tidybot_keyboard_base_yaw = None
        self._tidybot_keyboard_ee_pos = None
        self._tidybot_keyboard_ee_quat_wxyz = None
        self._tidybot_base_command_active = False

    def _ensure_tidybot_keyboard_base_target(self):
        if self.robot_name != "tidybot" or self.env is None:
            return False
        if self._tidybot_keyboard_base_pos is not None:
            return True
        base_pos, base_quat = self.env.get_body_pose()
        self._tidybot_keyboard_base_pos = base_pos.astype(np.float32).copy()
        self._tidybot_keyboard_base_yaw = self._yaw_from_quat_xyzw(base_quat)
        return True

    def _ensure_tidybot_keyboard_arm_target(self):
        if self.robot_name != "tidybot" or self.env is None:
            return False
        if self._tidybot_keyboard_ee_pos is not None and self._tidybot_keyboard_ee_quat_wxyz is not None:
            return True
        return self._sync_tidybot_keyboard_arm_target()

    def _sync_tidybot_keyboard_arm_target(self):
        if self.robot_name != "tidybot" or self.env is None:
            return False
        self.env._ensure_tidybot_ik_controller()
        ee_pos, ee_quat = self.env._body_pose_in_root(self.env._tidybot_ik_body_idx)
        planning_pos, planning_quat = self.env._planning_frame_pose_in_root()
        ee_pos_arm_base, ee_quat_arm_base = math_utils.subtract_frame_transforms(
            planning_pos, planning_quat, ee_pos, ee_quat
        )
        self._tidybot_keyboard_ee_pos = ee_pos_arm_base[0].detach().cpu().numpy().astype(np.float32).copy()
        self._tidybot_keyboard_ee_quat_wxyz = ee_quat_arm_base[0].detach().cpu().numpy().astype(np.float32).copy()
        return True

    def _apply_tidybot_keyboard_controls(self):
        if self.robot_name != "tidybot":
            return False
        base_active = bool(self._pressed_control_keys.intersection(self._tidybot_keyboard_base_keys))
        arm_active = bool(self._pressed_control_keys.intersection(self._tidybot_keyboard_arm_keys))
        if not base_active and not arm_active:
            if self.commands_source == "keyboard_tidybot":
                self.arm_command = None
            return False
        if not self._ensure_tidybot_keyboard_base_target():
            return False
        dt = float(getattr(self.manager_env, "step_dt", 0.05) or 0.05)
        base_vx = self.keyboard_base_linear_speed * self._keyboard_axis("W", "S")
        base_vy = self.keyboard_base_linear_speed * self._keyboard_axis("A", "D")
        base_wz = self.keyboard_base_yaw_speed * self._keyboard_axis("Q", "E")
        ee_vx = self.keyboard_arm_linear_speed * self._keyboard_axis("I", "K")
        ee_vy = self.keyboard_arm_linear_speed * self._keyboard_axis("J", "L")
        ee_vz = self.keyboard_arm_vertical_speed * self._keyboard_axis("U", "O")
        if base_vx or base_vy:
            heading = np.array([np.cos(self._tidybot_keyboard_base_yaw), np.sin(self._tidybot_keyboard_base_yaw)], dtype=np.float32)
            lateral = np.array([-np.sin(self._tidybot_keyboard_base_yaw), np.cos(self._tidybot_keyboard_base_yaw)], dtype=np.float32)
            self._tidybot_keyboard_base_pos[:2] += (heading * base_vx + lateral * base_vy) * dt
        if base_wz:
            self._tidybot_keyboard_base_yaw += base_wz * dt
        base_transform = np.concatenate(
            [self._tidybot_keyboard_base_pos, self._quat_xyzw_from_yaw(self._tidybot_keyboard_base_yaw)]
        ).astype(np.float32)
        self.commands[self.robot_index] = torch.tensor(base_transform, device=self.device, dtype=torch.float32)
        self._tidybot_base_command_active = base_active
        self.arm_command = None
        if arm_active and self._sync_tidybot_keyboard_arm_target():
            self._tidybot_keyboard_ee_pos[0] += ee_vx * dt
            self._tidybot_keyboard_ee_pos[1] += ee_vy * dt
            self._tidybot_keyboard_ee_pos[2] += ee_vz * dt
            ee_quat = self._tidybot_keyboard_ee_quat_wxyz
            self.arm_command = {
                "arm_target_pose": {
                    "frame": "arm_base_link",
                    "position": {
                        "x": float(self._tidybot_keyboard_ee_pos[0]),
                        "y": float(self._tidybot_keyboard_ee_pos[1]),
                        "z": float(self._tidybot_keyboard_ee_pos[2]),
                    },
                    "orientation": {
                        "w": float(ee_quat[0]),
                        "x": float(ee_quat[1]),
                        "y": float(ee_quat[2]),
                        "z": float(ee_quat[3]),
                    },
                }
            }
        self.commands_source = 'keyboard_tidybot'
        return True

    def _apply_keyboard_controls(self):
        if self.robot_name == "tidybot":
            self._apply_tidybot_keyboard_controls()
            return
        if self.command_dim != 3:
            return
        command = self._key_to_control["ZEROS"].clone()
        for key_name in self._pressed_control_keys:
            if key_name in self._key_to_control:
                command += self._key_to_control[key_name]
        self.commands[self.robot_index] = command
        self.commands_source = 'keyboard'

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        key_name = self._normalise_keyboard_input(event.input)
        key_press_types = {carb.input.KeyboardEventType.KEY_PRESS}
        key_repeat_type = getattr(carb.input.KeyboardEventType, "KEY_REPEAT", None)
        if key_repeat_type is not None:
            key_press_types.add(key_repeat_type)

        if event.type in key_press_types:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if key_name in self._keyboard_control_keys:
                print("keyboard event: ", key_name, flush=True)
                self.waiting_for_first_action = False
                self.clear_waypoints()
                self._pressed_control_keys.add(key_name)
                self._apply_keyboard_controls()
            if key_name.isdigit() and int(key_name) < min(self.env.unwrapped.num_envs, 10):
                self.robot_index = int(key_name)
                self._reset_tidybot_keyboard_targets()
                self.update_viewer()
            if key_name == 'C':
                self.viewport_third_person = not self.viewport_third_person
                self.update_viewer()
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in self._keyboard_control_keys:
                self._pressed_control_keys.discard(key_name)
                self._apply_keyboard_controls()
        return True
