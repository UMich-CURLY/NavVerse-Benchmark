"""Verify the original Turtlebot can move forward and turn in simulation."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher

import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
import navverse.vln_args as vln_cli_args


parser = argparse.ArgumentParser(description="Validate Turtlebot forward and turn motion in Isaac Lab.")
rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--test_episode", default="vcp_brussels_29")
parser.add_argument("--spawn_height", type=float)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--motion_steps", type=int, default=40)
parser.add_argument("--linear_speed", type=float, default=0.3)
parser.add_argument("--yaw_speed", type=float, default=0.6)
parser.add_argument("--turn_linear_speed", type=float, default=0.0)
parser.add_argument("--actuator_damping", type=float)
args = vln_cli_args.parse_args(parser)
args.robot_name = "turtlebot"
args.episode_label = args.test_episode
args.disable_socket_server = True
args.disable_camera = True
args.disable_termination = False

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
summary = {
    "robot": args.robot_name,
    "episode": args.test_episode,
    "requested_spawn_height_m": args.spawn_height,
    "requested_actuator_damping": args.actuator_damping,
    "phases": [],
}


def pose(vln_sim):
    position, quat_xyzw = vln_sim.env.get_body_pose()
    rotation = R.from_quat(quat_xyzw)
    yaw = float(rotation.as_euler("ZYX")[0])
    up = rotation.apply([0.0, 0.0, 1.0])
    tilt_deg = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    return position.astype(float), yaw, tilt_deg


def angle_delta(after, before):
    return float(np.arctan2(np.sin(after - before), np.cos(after - before)))


app = AppLauncher(args).app
vln_sim = None
try:
    from navverse.robots.turtlebot_rough_env_cfg import TURTLEBOT_CFG
    from navverse.sim import VLNSim

    if args.actuator_damping is not None:
        TURTLEBOT_CFG.actuators["wheels"].damping = float(args.actuator_damping)
    VLNSim.set_up_keyboard = lambda self: None
    vln_sim = VLNSim(args)
    episode_index = vln_sim.episode_label_list.index(args.test_episode)
    episode = vln_sim.episode_list[episode_index]
    if args.spawn_height is not None:
        vln_sim.manager_env.cfg.robot_spawn_height = float(args.spawn_height)
        episode["robot_spawn_height"] = float(args.spawn_height)
    vln_sim.reset(episode)
    vln_sim.step()
    vln_sim.waiting_for_first_action = False
    robot = vln_sim.manager_env.scene["robot"]
    action_term = vln_sim.manager_env.action_manager._terms["wheel_vel"]
    summary["robot_joint_names"] = list(robot.joint_names)
    summary["action_joint_names"] = list(action_term._joint_names)
    summary["action_joint_ids"] = list(action_term._joint_ids)
    summary["joint_properties"] = {
        "stiffness": robot.data.joint_stiffness[0].detach().cpu().tolist(),
        "damping": robot.data.joint_damping[0].detach().cpu().tolist(),
        "friction": robot.data.joint_friction_coeff[0].detach().cpu().tolist(),
        "dynamic_friction": robot.data.joint_dynamic_friction_coeff[0].detach().cpu().tolist(),
        "viscous_friction": robot.data.joint_viscous_friction_coeff[0].detach().cpu().tolist(),
        "armature": robot.data.joint_armature[0].detach().cpu().tolist(),
        "effort_limits": robot.data.joint_effort_limits[0].detach().cpu().tolist(),
        "velocity_limits": robot.data.joint_vel_limits[0].detach().cpu().tolist(),
    }
    print(
        "[TURTLEBOT_SMOKE] "
        f"robot_joints={summary['robot_joint_names']} "
        f"action_joints={summary['action_joint_names']} "
        f"action_joint_ids={summary['action_joint_ids']}",
        flush=True,
    )

    state = {"max_tilt_deg": 0.0, "termination_reason": None}

    def run_phase(name, command, steps):
        before_position, before_yaw, _ = pose(vln_sim)
        phase_max_tilt = 0.0
        completed_steps = 0
        wheel_velocity_samples = []
        low_level_action = None
        for _ in range(steps):
            vln_sim.commands[vln_sim.robot_index] = torch.tensor(
                command,
                device=vln_sim.device,
                dtype=torch.float32,
            )
            vln_sim.step()
            completed_steps += 1
            _, _, tilt_deg = pose(vln_sim)
            robot = vln_sim.manager_env.scene["robot"]
            wheel_velocity_samples.append(robot.data.joint_vel[0].detach().cpu().tolist())
            low_level_action = vln_sim.env.low_level_action[0].detach().cpu().tolist()
            phase_max_tilt = max(phase_max_tilt, tilt_deg)
            state["max_tilt_deg"] = max(state["max_tilt_deg"], tilt_deg)
            terminations = (vln_sim.info or {}).get("terminations", {})
            if terminations:
                state["termination_reason"] = terminations.get("termination_reason", "unknown")
                break
        after_position, after_yaw, _ = pose(vln_sim)
        heading = np.asarray([np.cos(before_yaw), np.sin(before_yaw)])
        displacement = after_position[:2] - before_position[:2]
        result = {
            "name": name,
            "command": list(command),
            "requested_steps": int(steps),
            "completed_steps": int(completed_steps),
            "forward_displacement_m": float(np.dot(displacement, heading)),
            "xy_displacement_m": float(np.linalg.norm(displacement)),
            "yaw_delta_rad": angle_delta(after_yaw, before_yaw),
            "max_tilt_deg": float(phase_max_tilt),
            "termination_reason": state["termination_reason"],
            "low_level_action": low_level_action,
            "mean_joint_velocity": np.mean(wheel_velocity_samples, axis=0).tolist(),
            "final_joint_velocity": wheel_velocity_samples[-1],
        }
        summary["phases"].append(result)
        print(f"[TURTLEBOT_SMOKE] {json.dumps(result, sort_keys=True)}", flush=True)
        return result

    initial_position, initial_yaw, initial_tilt = pose(vln_sim)
    summary["initial_pose"] = {
        "position": initial_position.tolist(),
        "yaw_rad": initial_yaw,
        "tilt_deg": initial_tilt,
    }
    settle = run_phase("settle", [0.0, 0.0, 0.0], args.settle_steps)
    motion_start_root_state = robot.data.root_state_w.clone()
    motion_start_joint_pos = robot.data.joint_pos.clone()

    def restore_motion_start():
        with torch.inference_mode():
            robot.write_root_state_to_sim(motion_start_root_state.clone())
            robot.write_joint_state_to_sim(
                motion_start_joint_pos.clone(),
                torch.zeros_like(robot.data.joint_vel),
            )
        for _ in range(5):
            vln_sim.commands[vln_sim.robot_index] = 0.0
            vln_sim.step()

    forward = run_phase("forward", [args.linear_speed, 0.0, 0.0], args.motion_steps)
    restore_motion_start()
    left = run_phase("left", [args.turn_linear_speed, 0.0, args.yaw_speed], args.motion_steps)
    restore_motion_start()
    right = run_phase("right", [args.turn_linear_speed, 0.0, -args.yaw_speed], args.motion_steps)
    final_position, final_yaw, final_tilt = pose(vln_sim)
    summary["final_pose"] = {
        "position": final_position.tolist(),
        "yaw_rad": final_yaw,
        "tilt_deg": final_tilt,
    }
    summary["max_tilt_deg"] = state["max_tilt_deg"]
    summary["termination_reason"] = state["termination_reason"]
    summary["checks"] = {
        "settled_upright": settle["max_tilt_deg"] < 25.0,
        "forward": forward["forward_displacement_m"] >= 0.3,
        "left": left["yaw_delta_rad"] >= 0.5,
        "right": right["yaw_delta_rad"] <= -0.5,
        "no_termination": state["termination_reason"] is None,
    }
    summary["passed"] = all(summary["checks"].values())
except BaseException as exc:
    summary["error"] = repr(exc)
    summary["traceback"] = traceback.format_exc()
    raise
finally:
    output_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    if vln_sim is not None:
        vln_sim.close()
    app.close()

if not summary.get("passed", False):
    raise SystemExit(1)
