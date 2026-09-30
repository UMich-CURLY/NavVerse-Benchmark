from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _yaw_from_quat_xyzw(quat_xyzw):
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _numeric_measurements(measurements):
    names = sorted(measurements)
    values = []
    for name in names:
        value = measurements[name]
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
    return names, values


def update_episode_with_astar_path(episode, vln_sim, goal_index=None):
    astar_path = getattr(vln_sim, "last_astar_world_path", None)
    if astar_path is None or len(astar_path) < 2:
        return False

    if goal_index is None:
        goal_index = int(episode.get("closest_goal_idx", 0) or 0)
    goal_index = max(0, min(goal_index, len(episode.get("goals", [])) - 1))
    goal = episode["goals"][goal_index]
    if "source_reference_path" not in goal:
        goal["source_reference_path"] = deepcopy(goal.get("reference_path", []))
    goal["reference_path"] = np.asarray(astar_path, dtype=float).tolist()
    goal["reference_path_planner"] = "astar"
    episode["closest_goal_idx"] = int(goal_index)
    episode["executed_goal_idx"] = int(goal_index)
    episode["executed_path_planner"] = "astar"
    episode["astar_plan_info"] = deepcopy(getattr(vln_sim, "last_astar_plan_info", {}) or {})
    return True


class EpisodeRecorder:
    def __init__(self, vln_sim, episode_dir, episode, scene_folder, clear_existing=True):
        self.vln_sim = vln_sim
        self.episode_dir = Path(episode_dir)
        self.episode = episode
        self.scene_folder = Path(scene_folder)
        self.scene_id = episode["scene_id"]
        if clear_existing and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        (self.episode_dir / "pov_rgb").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "third_person_rgb").mkdir(parents=True, exist_ok=True)
        self.pose_path = self.episode_dir / "pose.txt"
        if self.pose_path.exists():
            self.pose_path.unlink()
        self.frame_count = 0
        self.scene_state_path = self._ensure_scene_state()
        self.records = []
        self.frame_quality = []

    def _ensure_scene_state(self):
        scene_dir = self.scene_folder / "episode_data" / self.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        out_path = scene_dir / "scene_state.npz"
        if out_path.exists():
            return out_path

        payload = {}
        for prefix, filename in (("bev", "bev_map"), ("nav_bev", "nav_bev_map")):
            npz_path = scene_dir / f"{filename}.npz"
            info_path = scene_dir / f"{filename}_info.txt"
            if npz_path.exists():
                data = np.load(npz_path)
                payload[f"{prefix}_rgb"] = data["rgb"] if "rgb" in data else np.empty((0,), dtype=np.uint8)
                payload[f"{prefix}_depth"] = data["depth"] if "depth" in data else np.empty((0,), dtype=np.float32)
            else:
                payload[f"{prefix}_rgb"] = np.empty((0,), dtype=np.uint8)
                payload[f"{prefix}_depth"] = np.empty((0,), dtype=np.float32)
            payload[f"{prefix}_path"] = np.asarray(str(npz_path))
            payload[f"{prefix}_info_json"] = np.asarray(info_path.read_text() if info_path.exists() else "{}")
        np.savez_compressed(out_path, **payload)
        return out_path

    def _latest_images(self):
        obs = getattr(self.vln_sim, "obs", {}) or {}
        if "pov_rgb" in obs and "third_person_rgb" in obs:
            return {
                "pov": obs["pov_rgb"].cpu().numpy()[0, :, :, :3].astype(np.uint8),
                "third_person": obs["third_person_rgb"].cpu().numpy()[0, :, :, :3].astype(np.uint8),
            }

        if getattr(self.vln_sim.args, "on_demand_render", False):
            self.vln_sim.update_on_demand_sensor_data()
        latest_data = self.vln_sim._latest_data or {}
        latest_rgb = latest_data.get("rgb") or {}
        if "pov" not in latest_rgb or "third_person" not in latest_rgb:
            return None
        return {
            "pov": latest_rgb["pov"],
            "third_person": latest_rgb["third_person"],
        }

    def _target_waypoint(self, body_position):
        waypoints = self.vln_sim.waypoints or []
        waypoint_idx = getattr(self.vln_sim.waypoint_follower, "current_wp_idx", -1)
        if waypoint_idx is None:
            waypoint_idx = -1
        waypoint_idx = int(waypoint_idx)
        if 0 <= waypoint_idx < len(waypoints):
            waypoint = np.asarray(waypoints[waypoint_idx], dtype=np.float32)
            distance = float(np.linalg.norm(np.asarray(body_position[:2], dtype=float) - waypoint[:2]))
            return waypoint_idx, waypoint, distance
        return -1, np.full((3,), np.nan, dtype=np.float32), np.nan

    def capture_frame(self):
        images = self._latest_images()
        if images is None:
            return False

        frame_index = self.frame_count
        cv2.imwrite(str(self.episode_dir / "pov_rgb" / f"{frame_index}.png"), images["pov"][..., ::-1])
        cv2.imwrite(str(self.episode_dir / "third_person_rgb" / f"{frame_index}.png"), images["third_person"][..., ::-1])

        pov_mean = float(np.mean(images["pov"]))
        third_person_mean = float(np.mean(images["third_person"]))
        self.frame_quality.append(
            {
                "pov_rgb_mean": pov_mean,
                "third_person_rgb_mean": third_person_mean,
                "pov_black_frame": pov_mean < 5.0,
                "third_person_black_frame": third_person_mean < 5.0,
            }
        )

        body_pos, body_quat = self.vln_sim.env.get_body_pose()
        pov_pos, pov_quat = self.vln_sim.env.get_cam_pose()
        waypoint_idx, waypoint, waypoint_distance = self._target_waypoint(body_pos)
        measurements = {}
        measure_manager = getattr(self.vln_sim.env, "measure_manager", None)
        if measure_manager is not None:
            measurements = measure_manager.get_measurements()
        measurement_names, measurement_values = _numeric_measurements(measurements)
        command = self.vln_sim.commands[self.vln_sim.robot_index].detach().cpu().numpy().astype(np.float32)
        sim_step = int(getattr(self.vln_sim.manager_env, "common_step_counter", 0) or 0)
        step_dt = float(getattr(self.vln_sim.manager_env, "step_dt", 0.0) or 0.0)

        pose = [*pov_pos.tolist(), *pov_quat.tolist()]
        with self.pose_path.open("a") as f:
            f.write(f"{[frame_index] + pose}\n")

        self.records.append(
            {
                "frame_index": frame_index,
                "obs_index": int(getattr(self.vln_sim, "obs_index", 0) or 0),
                "sim_step": sim_step,
                "sim_time_s": sim_step * step_dt,
                "timestamp_ns": time.time_ns(),
                "body_position": np.asarray(body_pos, dtype=np.float32),
                "body_quat_xyzw": np.asarray(body_quat, dtype=np.float32),
                "body_yaw": _yaw_from_quat_xyzw(body_quat),
                "pov_position": np.asarray(pov_pos, dtype=np.float32),
                "pov_quat_xyzw": np.asarray(pov_quat, dtype=np.float32),
                "target_waypoint_index": waypoint_idx,
                "target_waypoint_xyyaw": waypoint,
                "target_waypoint_distance_m": waypoint_distance,
                "command": command,
                "measurement_names": measurement_names,
                "measurement_values": measurement_values,
            }
        )
        self.frame_count += 1
        return True

    def quality_summary(self):
        if not self.frame_quality:
            return {
                "frame_count": 0,
                "pov_black_ratio": 0.0,
                "third_person_black_ratio": 0.0,
            }
        return {
            "frame_count": len(self.frame_quality),
            "pov_black_ratio": float(np.mean([r["pov_black_frame"] for r in self.frame_quality])),
            "third_person_black_ratio": float(np.mean([r["third_person_black_frame"] for r in self.frame_quality])),
        }

    def finalize(self, astar_world_path=None, astar_plan_info=None):
        if not self.records:
            return None
        measurement_names = self.records[0]["measurement_names"]
        astar_waypoints = getattr(self.vln_sim, "last_astar_waypoints", None)
        if astar_waypoints is None:
            astar_waypoints = self.vln_sim.waypoints or []
        astar_world_path = [] if astar_world_path is None else astar_world_path
        payload = {
            "frame_index": np.asarray([r["frame_index"] for r in self.records], dtype=np.int32),
            "obs_index": np.asarray([r["obs_index"] for r in self.records], dtype=np.int32),
            "sim_step": np.asarray([r["sim_step"] for r in self.records], dtype=np.int64),
            "sim_time_s": np.asarray([r["sim_time_s"] for r in self.records], dtype=np.float32),
            "timestamp_ns": np.asarray([r["timestamp_ns"] for r in self.records], dtype=np.int64),
            "body_position": np.stack([r["body_position"] for r in self.records]).astype(np.float32),
            "body_quat_xyzw": np.stack([r["body_quat_xyzw"] for r in self.records]).astype(np.float32),
            "body_yaw": np.asarray([r["body_yaw"] for r in self.records], dtype=np.float32),
            "pov_position": np.stack([r["pov_position"] for r in self.records]).astype(np.float32),
            "pov_quat_xyzw": np.stack([r["pov_quat_xyzw"] for r in self.records]).astype(np.float32),
            "target_waypoint_index": np.asarray([r["target_waypoint_index"] for r in self.records], dtype=np.int32),
            "target_waypoint_xyyaw": np.stack([r["target_waypoint_xyyaw"] for r in self.records]).astype(np.float32),
            "target_waypoint_distance_m": np.asarray([r["target_waypoint_distance_m"] for r in self.records], dtype=np.float32),
            "command": np.stack([r["command"] for r in self.records]).astype(np.float32),
            "pov_rgb_mean": np.asarray([r["pov_rgb_mean"] for r in self.frame_quality], dtype=np.float32),
            "third_person_rgb_mean": np.asarray([r["third_person_rgb_mean"] for r in self.frame_quality], dtype=np.float32),
            "pov_black_frame": np.asarray([r["pov_black_frame"] for r in self.frame_quality], dtype=np.bool_),
            "third_person_black_frame": np.asarray(
                [r["third_person_black_frame"] for r in self.frame_quality], dtype=np.bool_
            ),
            "measurement_names": np.asarray(measurement_names),
            "measurements": np.asarray([r["measurement_values"] for r in self.records], dtype=np.float32),
            "reference_path": np.asarray(
                self.episode["goals"][int(self.episode.get("closest_goal_idx", 0) or 0)].get("reference_path", []),
                dtype=np.float32,
            ),
            "astar_world_path": np.asarray(astar_world_path, dtype=np.float32),
            "astar_waypoints": np.asarray(astar_waypoints, dtype=np.float32),
            "astar_plan_info_json": np.asarray(json.dumps(astar_plan_info or {}, default=_json_default)),
            "episode_info_json": np.asarray(json.dumps(self.episode, default=_json_default)),
            "scene_state_path": np.asarray(str(self.scene_state_path)),
        }
        out_path = self.episode_dir / "agent_state.npz"
        np.savez_compressed(out_path, **payload)
        return out_path
