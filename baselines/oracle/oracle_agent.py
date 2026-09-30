import json
import math
import os
import time

from baselines.base.base_agent import BaseAgent
from navverse.episode import find_episode_json


class OracleAgent(BaseAgent):
    STOP_DISTANCE_THRESHOLD = 0.5
    WAYPOINT_SPACING_M = 1.5
    NO_PROGRESS_EPS_M = 0.2
    NO_PROGRESS_STEPS = 4
    STUCK_SKIP_DISTANCE_M = 5.0
    STUCK_SKIP_MIN_WALL_S = 8.0

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.action_type = getattr(args, "oracle_action_type", "waypoint")
        self.stale_waypoint_skip = bool(getattr(args, "oracle_stale_waypoint_skip", True))
        self.capture_internal_frames = bool(
            getattr(args, "oracle_capture_internal_frames", False)
        )
        self.episodes_dir = os.path.abspath(
            os.path.expanduser(getattr(args, "episode_folder", None) or "navverse_data/episodes")
        )
        self.scene_episode_cache = {}
        self.reset()

    def reset(self):
        self.l_step = 0
        self.current_episode_label = None
        self.reference_waypoints = None
        self.reference_path = None
        self.goal_radius = 0.0
        self.current_waypoint_idx = 0
        self.last_sent_waypoint_idx = None
        self.current_target_best_distance = None
        self.current_target_no_progress_steps = 0
        self.current_target_started_at = None
        self.vis_frames = {}
        print("OracleAgent reset")

    def _load_scene_episode_map(self, scene_id):
        if scene_id in self.scene_episode_cache:
            return self.scene_episode_cache[scene_id]

        scene_episode_path = find_episode_json(self.episodes_dir, scene_id)
        if scene_episode_path is None:
            raise FileNotFoundError(
                f"Episode file not found for scene '{scene_id}' under {self.episodes_dir}"
            )

        with open(scene_episode_path, "r") as f:
            scene_episodes = json.load(f)

        episode_map = {}
        for episode in scene_episodes:
            episode_label = f"{episode['scene_id']}_{episode['episode_id']}"
            episode_map[episode_label] = episode
        self.scene_episode_cache[scene_id] = episode_map
        return episode_map

    def _load_reference_path(self, obs):
        scene_id = obs["scene_id"]
        episode_label = obs["episode_label"]
        episode_map = self._load_scene_episode_map(scene_id)
        if episode_label not in episode_map:
            raise KeyError(f"Episode '{episode_label}' not found in {scene_id}.json")

        episode = episode_map[episode_label]
        goals = episode.get("goals", [])
        if len(goals) == 0:
            raise ValueError(f"Episode '{episode_label}' does not contain any goals")

        closest_goal_idx = int(episode.get("closest_goal_idx", 0))
        if closest_goal_idx < 0 or closest_goal_idx >= len(goals):
            raise IndexError(
                f"Episode '{episode_label}' has invalid closest_goal_idx={closest_goal_idx}"
            )

        goal = goals[closest_goal_idx]
        self.goal_radius = float(goal.get("radius", 0.0) or 0.0)
        return goal.get("reference_path", [])

    def _densify_reference_path(self, reference_path):
        if len(reference_path) < 2:
            return reference_path

        densified_path = [reference_path[0]]
        for start, end in zip(reference_path[:-1], reference_path[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dz = end[2] - start[2] if len(start) > 2 and len(end) > 2 else 0.0
            segment_length = math.hypot(dx, dy)
            subdivisions = max(1, int(math.ceil(segment_length / self.WAYPOINT_SPACING_M)))
            for step_idx in range(1, subdivisions + 1):
                ratio = step_idx / subdivisions
                point = [
                    start[0] + dx * ratio,
                    start[1] + dy * ratio,
                ]
                if len(start) > 2 and len(end) > 2:
                    point.append(start[2] + dz * ratio)
                densified_path.append(point)
        return densified_path

    def _distance_to_waypoint(self, obs, waypoint):
        current_position = obs.get("position")
        if current_position is None or len(current_position) < 2:
            return None

        return math.hypot(
            current_position[0] - waypoint[0],
            current_position[1] - waypoint[1],
        )

    def _yaw_for_reference_index(self, reference_idx):
        if not self.reference_path:
            return 0.0

        point = self.reference_path[reference_idx]
        if reference_idx + 1 < len(self.reference_path):
            next_point = self.reference_path[reference_idx + 1]
            return math.atan2(next_point[1] - point[1], next_point[0] - point[0])
        if reference_idx > 0:
            previous_point = self.reference_path[reference_idx - 1]
            return math.atan2(point[1] - previous_point[1], point[0] - previous_point[0])
        return 0.0

    def _transform_for_reference_index(self, reference_idx):
        point = self.reference_path[reference_idx]
        yaw = self._yaw_for_reference_index(reference_idx)
        half_yaw = 0.5 * yaw
        return {
            "position": {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]) if len(point) > 2 else 0.0,
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(half_yaw),
                "w": math.cos(half_yaw),
            },
        }

    def _update_current_waypoint_idx(self, obs):
        if not self.reference_waypoints:
            return

        while self.current_waypoint_idx < len(self.reference_waypoints):
            reached_idx = None
            reached_distance = None
            for waypoint_idx in range(len(self.reference_waypoints) - 1, self.current_waypoint_idx - 1, -1):
                target_waypoint = self.reference_waypoints[waypoint_idx]
                distance_to_waypoint = self._distance_to_waypoint(obs, target_waypoint)
                if distance_to_waypoint is None:
                    return
                waypoint_threshold = self.STOP_DISTANCE_THRESHOLD
                if waypoint_idx == len(self.reference_waypoints) - 1:
                    waypoint_threshold = max(waypoint_threshold, self.goal_radius)
                if distance_to_waypoint <= waypoint_threshold:
                    reached_idx = waypoint_idx
                    reached_distance = distance_to_waypoint
                    break

            if reached_idx is None:
                return

            print(
                f"OracleAgent act: episode={obs['episode_label']}, "
                f"reached waypoint {reached_idx + 1}/"
                f"{len(self.reference_waypoints)} at {reached_distance:.3f}m"
            )
            self.current_waypoint_idx = reached_idx + 1
            self.current_target_best_distance = None
            self.current_target_no_progress_steps = 0
            self.current_target_started_at = None

    def _skip_stale_intermediate_waypoint(self, obs):
        if not self.reference_waypoints:
            return
        if self.current_waypoint_idx >= len(self.reference_waypoints) - 1:
            return

        target_waypoint = self.reference_waypoints[self.current_waypoint_idx]
        distance_to_target = self._distance_to_waypoint(obs, target_waypoint)
        if distance_to_target is None:
            return

        if (
            self.current_target_best_distance is None
            or distance_to_target < self.current_target_best_distance - self.NO_PROGRESS_EPS_M
        ):
            self.current_target_best_distance = distance_to_target
            self.current_target_no_progress_steps = 0
            self.current_target_started_at = time.time()
            return

        if self.current_target_started_at is None:
            self.current_target_started_at = time.time()

        self.current_target_no_progress_steps += 1
        if (
            self.current_target_no_progress_steps >= self.NO_PROGRESS_STEPS
            and time.time() - self.current_target_started_at >= self.STUCK_SKIP_MIN_WALL_S
            and distance_to_target <= self.STUCK_SKIP_DISTANCE_M
        ):
            print(
                f"OracleAgent act: episode={obs['episode_label']}, "
                f"skipping stale waypoint {self.current_waypoint_idx + 1}/"
                f"{len(self.reference_waypoints)} at {distance_to_target:.3f}m"
            )
            self.current_waypoint_idx += 1
            self.current_target_best_distance = None
            self.current_target_no_progress_steps = 0
            self.current_target_started_at = None

    def act(self, obs, step_data_queue=None):
        episode_label = obs["episode_label"]
        if obs.get("rgb") is not None:
            self.vis_frames.setdefault("rgb", []).append(obs["rgb"].copy())

        if self.action_type == "task_generation_internal":
            if self.capture_internal_frames:
                for camera_name, frame in (obs.get("rgb_dict") or {}).items():
                    if frame is not None:
                        self.vis_frames.setdefault(
                            f"diagnostic_{camera_name}", []
                        ).append(frame.copy())
            self.current_episode_label = episode_label
            self.l_step += 1
            return None

        if self.current_episode_label != episode_label:
            self.current_episode_label = episode_label
            self.reference_waypoints = None
            self.reference_path = None
            self.goal_radius = 0.0
            self.current_waypoint_idx = 0
            self.last_sent_waypoint_idx = None
            self.current_target_best_distance = None
            self.current_target_no_progress_steps = 0
            self.current_target_started_at = None
        self.l_step += 1

        if self.reference_waypoints is None:
            self.reference_path = self._densify_reference_path(self._load_reference_path(obs))
            self.reference_waypoints = [
                [point[0], point[1], float("inf")]
                for point in self.reference_path[1:]
            ]
            print(
                f"OracleAgent act: episode={episode_label}, "
                f"loaded {len(self.reference_waypoints)} reference waypoints"
            )
        if len(self.reference_waypoints) == 0:
            print(f"OracleAgent act: episode={episode_label}, no reference waypoints")
            return None

        self._update_current_waypoint_idx(obs)
        if self.action_type == "waypoint" and self.stale_waypoint_skip:
            self._skip_stale_intermediate_waypoint(obs)
        if self.current_waypoint_idx >= len(self.reference_waypoints):
            print(
                f"OracleAgent act: episode={episode_label}, "
                f"all waypoints reached from reference path, sending STOP"
            )
            return {"action": "STOP"}

        if self.last_sent_waypoint_idx == self.current_waypoint_idx:
            return None

        self.last_sent_waypoint_idx = self.current_waypoint_idx
        if self.action_type == "transform":
            reference_idx = self.current_waypoint_idx + 1
            print(
                f"OracleAgent act: episode={episode_label}, "
                f"sending GT transform {reference_idx + 1}/{len(self.reference_path)}"
            )
            return {"transform": self._transform_for_reference_index(reference_idx)}

        remaining_waypoints = self.reference_waypoints[self.current_waypoint_idx :]
        print(
            f"OracleAgent act: episode={episode_label}, "
            f"sending {len(remaining_waypoints)} sim waypoints"
        )
        return {"waypoint": remaining_waypoints}
