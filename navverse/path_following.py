import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))

def world_to_body(dx, dy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy

def calc_yaw(current_pos, target_pos):
    return np.arctan2((target_pos[1] - current_pos[1]), target_pos[0] - current_pos[0])

def set_yaw_to_forward(cam_pos, waypoints):
    last_pos = cam_pos[:2]
    new_waypoints = []
    for wp in waypoints:
        yaw_angle = calc_yaw(last_pos, wp[:2])
        new_waypoints.append([wp[0], wp[1], yaw_angle])
        last_pos = wp[:2]
    return new_waypoints


class KinematicPIDPoseFollower:
    """Habitat-PE PID pose tracking for an ideal holonomic kinematic base."""

    def __init__(
        self,
        kp_pos=2.5,
        ki_pos=0.0,
        kd_pos=0.3,
        kp_yaw=4.0,
        ki_yaw=0.0,
        kd_yaw=0.2,
        max_linear_velocity=2.0,
        max_linear_acceleration=8.0,
        max_yaw_velocity=np.pi,
        max_yaw_acceleration=np.deg2rad(400.0),
        integral_limit_pos=0.5,
        integral_limit_yaw=np.deg2rad(30.0),
        feedforward_gain=1.0,
    ):
        self.kp_pos = float(kp_pos)
        self.ki_pos = float(ki_pos)
        self.kd_pos = float(kd_pos)
        self.kp_yaw = float(kp_yaw)
        self.ki_yaw = float(ki_yaw)
        self.kd_yaw = float(kd_yaw)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_linear_acceleration = float(max_linear_acceleration)
        self.max_yaw_velocity = float(max_yaw_velocity)
        self.max_yaw_acceleration = float(max_yaw_acceleration)
        self.integral_limit_pos = float(integral_limit_pos)
        self.integral_limit_yaw = float(integral_limit_yaw)
        self.feedforward_gain = float(feedforward_gain)
        self.reset()

    def reset(self):
        self.integral_pos = np.zeros(2, dtype=float)
        self.integral_yaw = 0.0
        self.previous_command = np.zeros(3, dtype=float)
        self.previous_pose = None
        self.previous_reference = None

    def update(self, current_pose, reference_pose, dt):
        current = np.asarray(current_pose, dtype=float)
        reference = np.asarray(reference_pose, dtype=float)
        if current.shape != (3,) or reference.shape != (3,):
            raise ValueError("current_pose and reference_pose must both have shape (3,)")
        if not np.isfinite(current).all() or not np.isfinite(reference).all():
            raise ValueError("PID poses must be finite")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("PID dt must be positive")

        if self.previous_pose is None:
            current_velocity = np.zeros(3, dtype=float)
        else:
            current_velocity = np.empty(3, dtype=float)
            current_velocity[:2] = (current[:2] - self.previous_pose[:2]) / dt
            current_velocity[2] = wrap_to_pi(current[2] - self.previous_pose[2]) / dt

        previous_reference = current if self.previous_reference is None else self.previous_reference
        reference_velocity = np.empty(3, dtype=float)
        reference_velocity[:2] = (reference[:2] - previous_reference[:2]) / dt
        reference_velocity[2] = wrap_to_pi(reference[2] - previous_reference[2]) / dt

        error_pos = reference[:2] - current[:2]
        error_yaw = wrap_to_pi(reference[2] - current[2])
        self.integral_pos = np.clip(
            self.integral_pos + error_pos * dt,
            -self.integral_limit_pos,
            self.integral_limit_pos,
        )
        self.integral_yaw = float(
            np.clip(
                self.integral_yaw + error_yaw * dt,
                -self.integral_limit_yaw,
                self.integral_limit_yaw,
            )
        )

        command_xy = (
            self.kp_pos * error_pos
            + self.ki_pos * self.integral_pos
            - self.kd_pos * current_velocity[:2]
            + self.feedforward_gain * reference_velocity[:2]
        )
        command_yaw = (
            self.kp_yaw * error_yaw
            + self.ki_yaw * self.integral_yaw
            - self.kd_yaw * current_velocity[2]
            + self.feedforward_gain * reference_velocity[2]
        )

        speed = float(np.linalg.norm(command_xy))
        if speed > self.max_linear_velocity:
            command_xy *= self.max_linear_velocity / speed
        command_yaw = float(np.clip(command_yaw, -self.max_yaw_velocity, self.max_yaw_velocity))

        delta_xy = command_xy - self.previous_command[:2]
        max_delta_xy = self.max_linear_acceleration * dt
        delta_speed = float(np.linalg.norm(delta_xy))
        if delta_speed > max_delta_xy:
            delta_xy *= max_delta_xy / delta_speed
        delta_yaw = float(
            np.clip(
                command_yaw - self.previous_command[2],
                -self.max_yaw_acceleration * dt,
                self.max_yaw_acceleration * dt,
            )
        )
        command = self.previous_command + np.array([delta_xy[0], delta_xy[1], delta_yaw])

        next_pose = current.copy()
        next_pose[:2] += command[:2] * dt
        next_pose[2] = wrap_to_pi(current[2] + command[2] * dt)
        self.previous_pose = current.copy()
        self.previous_reference = reference.copy()
        self.previous_command = command
        return next_pose

class WaypointFollower:
    def __init__(self,
        device,
        max_vel=[1.5, 1.0, 1.0],
        min_vel=[0.3, 0.1, 0.3],
        kp=[2.0, 0.5, 2.0],
        ki=[0.0, 0.0, 0.0], # [0.1, 0.1, 0.1]
        kd=[0.0, 0.0, 0.0], # [3, 3, 1]
        max_integral=[5.0, 5.0, 3.0],
        arrive_dist=0.1,
        arrive_yaw=np.pi/180.0*15.0,
        term_dist=0.03,
        term_yaw=np.pi/180.0*3.0,
        lookahead_distance=0,
        prefer_forward=True,
        lateral_deadband=0.0,
        lateral_turn_damping_yaw=0.0,
    ):
        self.device = device
        self.max_vel = np.array(max_vel)
        self.min_vel = np.array(min_vel)
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.max_integral = np.array(max_integral)
        self.arrive_dist = arrive_dist
        self.arrive_yaw = arrive_yaw
        self.term_dist = term_dist
        self.term_yaw = term_yaw
        self.lookahead_distance = lookahead_distance
        self.lateral_deadband = lateral_deadband
        self.lateral_turn_damping_yaw = lateral_turn_damping_yaw
        self.current_wp_idx = 0
        self.error_integral = np.zeros(3)
        self.last_error = np.zeros(3)
        self.arrived_at_goal = False
        self.prefer_forward = prefer_forward
    
    def reset(self):
        self.arrived_at_goal = False
        self.current_wp_idx = 0
        self.error_integral = np.zeros(3)
        self.last_error = np.zeros(3)
    
    def update(self, cam_pos, cam_quat, waypoints_world, verbose=False):
        device = self.device
        current_wp_idx = self.current_wp_idx

        # Empty or invalid path
        if waypoints_world is None or len(waypoints_world) == 0:
            return torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)

        num_wps = len(waypoints_world)
        # Clamp current index to valid range (do not append/alter original path)
        if current_wp_idx >= num_wps:
            current_wp_idx = num_wps - 1

        base_xy = np.array(cam_pos[:2])
        base_yaw = R.from_quat(cam_quat).as_euler('ZYX')[0]
        waypoints_world = np.array(waypoints_world)

        final_wp = waypoints_world[-1]
        final_dist = np.linalg.norm(final_wp[:2] - base_xy)
        final_yaw = final_wp[2] if final_wp.shape[0] > 2 else np.inf
        final_yaw_ignored = (not np.isfinite(final_yaw)) or final_yaw > 10.0
        final_yaw_reached = final_yaw_ignored or abs(wrap_to_pi(final_yaw - base_yaw)) < self.term_yaw
        if final_dist < self.term_dist and final_yaw_reached:
            self.current_wp_idx = num_wps - 1
            self.arrived_at_goal = True
            print(f"[PLAN] Reached goal")
            return torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)

        # 1) Find nearest forward waypoint to avoid going backwards
        #    Only consider waypoints from current_wp_idx forward
        dists = [
            (i, np.linalg.norm(waypoints_world[i, :2] - base_xy))
            for i in range(num_wps)
        ]
        nearest_idx, nearest_dist = min(dists, key=lambda t: t[1])
        current_wp_idx = max(current_wp_idx, nearest_idx)

        # 2) Choose a lookahead target a few meters ahead along the path
        #    Use a simple rule: first waypoint at distance >= self.lookahead_distance from robot
        target_idx = current_wp_idx
        for i in range(current_wp_idx, num_wps):
            if dists[i][1] >= self.lookahead_distance:
                target_idx = i
                break
        if target_idx != current_wp_idx:
            self.error_integral = np.zeros(3)
            self.last_error = np.zeros(3)
        current_wp_idx = target_idx
        is_final_segment = (current_wp_idx == num_wps - 1)
        target_wp = waypoints_world[target_idx].copy()

        # set different arrive thresholds for final segment
        if is_final_segment:
            arrive_dist = self.term_dist
            arrive_yaw = self.term_yaw
        else:
            arrive_dist = self.arrive_dist
            arrive_yaw = self.arrive_yaw

        # calculate xy diff
        diff_to_goal = target_wp[:2] - base_xy
        dist_to_goal = np.linalg.norm(diff_to_goal)
        
        if dist_to_goal < 0.5 and target_wp[2]>10.0:
            # if set yaw is inf and robot is close to wp, set target yaw to current yaw
            target_wp[2] = base_yaw
        elif (dist_to_goal > 0.5 and self.prefer_forward) or target_wp[2]>10.0:
            # set yaw to next waypoint if far away from the waypoint position or set yaw is inf
            target_wp = np.array(target_wp)
            target_wp[2] = calc_yaw(base_xy, target_wp[:2])

        # 3) Compute control to the lookahead target in body frame (pure-pursuit style)
        dx = target_wp[0] - base_xy[0]
        dy = target_wp[1] - base_xy[1]
        ang_err = wrap_to_pi(target_wp[2] - base_yaw)

        ex_b, ey_b = world_to_body(dx, dy, base_yaw)

        # Update error integral and derivative
        lateral_error = 0.0 if abs(ey_b) < self.lateral_deadband else ey_b
        error = np.array([ex_b, lateral_error, ang_err])
        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -self.max_integral, self.max_integral)
        error_derivative = error - self.last_error
        self.last_error = error

        # Linear velocities: proportional to lookahead vector, clipped by given maxima
        # TODO: I and D terms should consider dt
        vel = self.kp * error + self.ki * self.error_integral + self.kd * error_derivative

        # enforce minimum velocities when far from goal
        for i, diff in enumerate([ex_b, ey_b]):
            if abs(diff) > arrive_dist and abs(vel[i]) < self.min_vel[i]:
                sign_source = vel[i] if abs(vel[i]) > 1e-6 else diff
                vel[i] = math.copysign(self.min_vel[i], sign_source)
        if abs(ang_err) > arrive_yaw and abs(vel[2]) < self.min_vel[2]:
            sign_source = vel[2] if abs(vel[2]) > 1e-6 else ang_err
            vel[2] = math.copysign(self.min_vel[2], sign_source)

        # if the angle error is too large, set linear velocity to 0
        if abs(ang_err) > np.deg2rad(30.0) and self.prefer_forward:
            vel[0] = 0.0
            vel[1] = 0.0
        elif self.lateral_turn_damping_yaw > 0.0 and abs(ang_err) > self.lateral_turn_damping_yaw:
            vel[1] *= self.lateral_turn_damping_yaw / abs(ang_err)

        vel = np.clip(vel, -self.max_vel, self.max_vel)
        vx, vy, vw = vel

        if verbose:
            print("================================================")
            print(f"cam_pos: {cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}")
            print(f"target_wp: [{target_idx}] {target_wp[0]:.2f}, {target_wp[1]:.2f}, {np.rad2deg(target_wp[2]):.2f}")
            print(f"base_xy: {base_xy[0]:.2f}, {base_xy[1]:.2f} base_yaw: {np.rad2deg(base_yaw):.2f}")
            print(f"dx: {dx:.2f}, dy: {dy:.2f}, ang_err: {np.rad2deg(ang_err):.2f}")
            print(f"ex_b: {ex_b:.2f}, ey_b: {ey_b:.2f}")
            print(f"vx: {vx:.2f}, vy: {vy:.2f}, vw: {vw:.2f}")
            print(f"current_wp_idx: {current_wp_idx}, num_wps: {num_wps}")


        if dist_to_goal < arrive_dist and abs(ang_err) < arrive_yaw:
            if is_final_segment:
                # vx, vy, vw = 0.0, 0.0, 0.0
                current_wp_idx = num_wps - 1
                self.arrived_at_goal = True
                print(f"[PLAN] Reached goal")
            else:
                current_wp_idx += 1
                print(f"[PLAN] Reached waypoint {current_wp_idx}/{num_wps}")

        command = torch.tensor([[vx, vy, vw]], device=device, dtype=torch.float32)
        self.current_wp_idx = current_wp_idx
        return command
