import os

import numpy as np
import torch

DEFAULT_EPISODE_TIMEOUT_S = 300.0

try:
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    from isaaclab.assets import RigidObject
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils import configclass
    _ISAACLAB_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _ISAACLAB_IMPORT_ERROR = exc

    class _MissingMdp:
        def __getattr__(self, _name):
            def _missing(*_args, **_kwargs):
                raise _ISAACLAB_IMPORT_ERROR

            return _missing

    class RigidObject:
        pass

    class SceneEntityCfg:
        def __init__(self, name, *_args, **_kwargs):
            self.name = name

    class DoneTerm:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def configclass(cls):
        return cls

    mdp = _MissingMdp()

def terrain_out_of_bounds(
    env, asset_cfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
):
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    source: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/terminations.py
    """
    terrain_type = env.scene.cfg.terrain.terrain_type
    if terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        position_w = (
            env._pose_state_w()[:, :3]
            if hasattr(env, "_pose_state_w")
            else asset.data.root_pos_w
        )

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(position_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(position_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    elif terrain_type == "usd":
        asset: RigidObject = env.scene[asset_cfg.name]
        position_w = (
            env._pose_state_w()[:, :3]
            if hasattr(env, "_pose_state_w")
            else asset.data.root_pos_w
        )
        min_x, min_y, min_z, max_x, max_y, max_z = env.get_prim_bounding_box("/World/ground/terrain")
        return (position_w[:, 0] > max_x - distance_buffer) | \
            (position_w[:, 0] < min_x + distance_buffer) | \
            (position_w[:, 1] > max_y - distance_buffer) | \
            (position_w[:, 1] < min_y + distance_buffer) | \
            (position_w[:, 2] < min_z)
    else:
        raise ValueError(f"Received unsupported terrain type: {terrain_type}, must be either 'plane' or 'generator'.")

def time_out(env, max_time: float = DEFAULT_EPISODE_TIMEOUT_S) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    if env.measure_manager:
        return env.measure_manager.get_measure("sim_duration").get_metric() >= max_time
    else:
        return False
    
def _yaw_from_quat_wxyz(quat):
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def robot_stuck(
    env,
    asset_cfg = SceneEntityCfg("robot"),
    label: str = "stuck",
    max_time: float = 30.0,
    dist_threshold: float = 0.01,
    yaw_threshold: float = 0.01,
    vel_threshold: float = 0.1,
):
    """Terminate when a command fails to produce either XY or yaw motion."""
    robot = env.scene[asset_cfg.name]
    curr_xy = robot.data.root_pos_w[:, :2].clone()
    curr_yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w).unsqueeze(-1)
    curr_pose = torch.cat([curr_xy, curr_yaw], dim=-1)

    prev_pos = env.termination_states[f"{label}_prev_pos"]
    same_pos_count = env.termination_states[f"{label}_same_pos_count"]
    if hasattr(robot, "velocity_command"):
        command = robot.velocity_command
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape[-1] >= 3:
            robot_vel = torch.norm(command[:, :3], dim=-1)
        elif command.shape[-1] >= 2:
            robot_vel = torch.norm(command[:, :2], dim=-1)
        else:
            robot_vel = torch.abs(command.squeeze(-1))
    else:
        robot_vel = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    pose_delta = curr_pose - prev_pos
    pose_delta[:, 2] = torch.atan2(torch.sin(pose_delta[:, 2]), torch.cos(pose_delta[:, 2]))
    xy_dist = torch.norm(pose_delta[:, :2], dim=-1)
    yaw_dist = torch.abs(pose_delta[:, 2])
    xy_dist_history = env.termination_states.setdefault(f"{label}_xy_dist_history", [[] for _ in range(env.num_envs)])
    yaw_dist_history = env.termination_states.setdefault(f"{label}_yaw_dist_history", [[] for _ in range(env.num_envs)])
    pose_dist = torch.sqrt(xy_dist * xy_dist + yaw_dist * yaw_dist)
    dist_history = env.termination_states[f"{label}_dist_history"]
    for i, d in enumerate(pose_dist):
        dist_history[i].append(d)
        xy_dist_history[i].append(xy_dist[i])
        yaw_dist_history[i].append(yaw_dist[i])
        if len(dist_history[i]) > 5:
            dist_history[i].pop(0)
        if len(xy_dist_history[i]) > 5:
            xy_dist_history[i].pop(0)
        if len(yaw_dist_history[i]) > 5:
            yaw_dist_history[i].pop(0)
    avg_pose_dist = torch.stack([torch.mean(torch.stack(d)) for d in dist_history])
    avg_xy_dist = torch.stack([torch.mean(torch.stack(d)) for d in xy_dist_history])
    avg_yaw_dist = torch.stack([torch.mean(torch.stack(d)) for d in yaw_dist_history])
    no_pose_progress = torch.logical_and(avg_xy_dist < dist_threshold, avg_yaw_dist < yaw_threshold)
    same_pos = torch.logical_and(no_pose_progress, robot_vel > vel_threshold)
    same_pos_count[same_pos] += 1
    same_pos_count[~same_pos] = 0
    if os.environ.get("NAVVERSE_STUCK_DEBUG", "").lower() in {"1", "true", "yes"}:
        debug_count = env.termination_states.setdefault(
            f"{label}_debug_step_count",
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device),
        )
        debug_count += 1
        interval = max(1, int(os.environ.get("NAVVERSE_STUCK_DEBUG_INTERVAL", "20")))
        should_print = torch.logical_or(same_pos, debug_count % interval == 0)
        for i in range(env.num_envs):
            if bool(should_print[i].item()):
                print(
                    "[STUCK_DEBUG] "
                    f"env={i} same_pos={bool(same_pos[i].item())} "
                    f"count={int(same_pos_count[i].item())} "
                    f"xy_delta={float(xy_dist[i].item()):.4f} "
                    f"yaw_delta={float(yaw_dist[i].item()):.4f} "
                    f"avg_xy_delta={float(avg_xy_dist[i].item()):.4f} "
                    f"avg_yaw_delta={float(avg_yaw_dist[i].item()):.4f} "
                    f"avg_pose_delta={float(avg_pose_dist[i].item()):.4f} "
                    f"cmd_norm={float(robot_vel[i].item()):.4f} "
                    f"thresholds=(xy<{dist_threshold:.4f}, yaw<{yaw_threshold:.4f}, cmd>{vel_threshold:.4f})",
                    flush=True,
                )
    env.termination_states[f"{label}_prev_pos"] = curr_pose

    # if the robot has stayed in the same location for 1000 steps
    return same_pos_count*env.manager_env.step_dt >= max_time

def stop_called(env):
    """Terminate the episode when the stop is called."""
    return env.is_stop_called

@configclass
class VLNTerminationsCfg:
    """Termination terms for the MDP.
    The env should have a termination_states dictionary to store the termination states.
    Once a termination is activated, the env will put termination reason into info.
    VLNSim will receive the info in update_obs() and pause the socket server until episode resets.
    """
    time_out = DoneTerm(func=time_out, time_out=True, params={"max_time": DEFAULT_EPISODE_TIMEOUT_S})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": float(np.deg2rad(45.0))},
    )
    terrain_out_of_bounds = DoneTerm(
        func=terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    # stuck = DoneTerm(
    #     func=robot_stuck,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "max_time": 100.0,
    #         "label": "stuck",
    #         "dist_threshold": 0.015,
    #         "yaw_threshold": 0.01,
    #         "vel_threshold": 0.01,
    #     },
    # )
    # back_n_forth = DoneTerm(
    #     func=robot_stuck,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "max_time": 30.0,
    #         "label": "back_n_forth",
    #         "dist_threshold": 2.0,
    #         "vel_threshold": 0,
    #     },
    # )
    stop_called = DoneTerm(
        func=stop_called,
    )
