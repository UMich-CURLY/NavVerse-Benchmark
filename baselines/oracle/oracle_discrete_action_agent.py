import math

from baselines.oracle.oracle_agent import OracleAgent


class OracleDiscreteActionAgent(OracleAgent):
    DEFAULT_ANGLE_THRESHOLD_DEG = 30.0
    DEFAULT_WAYPOINT_SWITCH_DISTANCE_M = 1.0
    DEFAULT_STOP_DISTANCE_M = 0.5

    def reset(self):
        super().reset()
        self.action_counts = {}

    @staticmethod
    def _wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _yaw_from_xyzw(quat_xyzw):
        x, y, z, w = quat_xyzw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _angle_threshold_rad(self):
        threshold_deg = getattr(
            self.args,
            "oracle_discrete_angle_threshold_deg",
            self.DEFAULT_ANGLE_THRESHOLD_DEG,
        )
        return math.radians(float(threshold_deg))

    def _waypoint_switch_distance_m(self):
        return float(
            getattr(
                self.args,
                "oracle_discrete_waypoint_switch_distance_m",
                self.DEFAULT_WAYPOINT_SWITCH_DISTANCE_M,
            )
        )

    def _stop_distance_m(self):
        configured_stop_distance = getattr(
            self.args,
            "oracle_discrete_stop_distance_m",
            self.DEFAULT_STOP_DISTANCE_M,
        )
        return max(float(configured_stop_distance), float(self.goal_radius or 0.0))

    def _ensure_reference_waypoints(self, obs):
        if self.reference_waypoints is not None:
            return
        self.reference_path = self._densify_reference_path(self._load_reference_path(obs))
        self.reference_waypoints = [
            [point[0], point[1], float("inf")]
            for point in self.reference_path[1:]
        ]
        print(
            f"OracleDiscreteActionAgent act: episode={obs['episode_label']}, "
            f"loaded {len(self.reference_waypoints)} reference waypoints"
        )

    def _target_error(self, obs, waypoint):
        current_position = obs.get("position")
        rotation = obs.get("rotation")
        if current_position is None or rotation is None:
            return None, None
        dx = waypoint[0] - current_position[0]
        dy = waypoint[1] - current_position[1]
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw = self._yaw_from_xyzw(rotation)
        return distance, self._wrap_pi(target_yaw - yaw)

    def _record_action(self, action):
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        return {"action": action}

    def _reset_episode_state(self, episode_label):
        self.current_episode_label = episode_label
        self.reference_waypoints = None
        self.reference_path = None
        self.goal_radius = 0.0
        self.current_waypoint_idx = 0
        self.current_target_best_distance = None
        self.current_target_no_progress_steps = 0
        self.current_target_started_at = None
        self.action_counts = {}

    def _select_action(self, distance, yaw_error, is_final_waypoint, angle_threshold, waypoint_switch_distance, stop_distance):
        if is_final_waypoint and distance <= stop_distance:
            return "STOP"
        if not is_final_waypoint and distance <= waypoint_switch_distance:
            return "NEXT_WAYPOINT"
        if yaw_error > angle_threshold:
            return "TURN_LEFT"
        if yaw_error < -angle_threshold:
            return "TURN_RIGHT"
        distance_threshold = stop_distance if is_final_waypoint else waypoint_switch_distance
        if distance > distance_threshold:
            return "FORWARD"
        return "NEXT_WAYPOINT"

    def act(self, obs, step_data_queue=None):
        episode_label = obs["episode_label"]
        if obs.get("rgb") is not None:
            self.vis_frames.setdefault("rgb", []).append(obs["rgb"].copy())

        if self.current_episode_label != episode_label:
            self._reset_episode_state(episode_label)
        self.l_step += 1

        self._ensure_reference_waypoints(obs)
        if len(self.reference_waypoints) == 0:
            print(f"OracleDiscreteActionAgent act: episode={episode_label}, no reference waypoints")
            return self._record_action("STOP")

        self._update_current_waypoint_idx(obs)
        self._skip_stale_intermediate_waypoint(obs)
        if self.current_waypoint_idx >= len(self.reference_waypoints):
            print(f"OracleDiscreteActionAgent act: episode={episode_label}, all waypoints reached, sending STOP")
            return self._record_action("STOP")

        angle_threshold = self._angle_threshold_rad()
        waypoint_switch_distance = self._waypoint_switch_distance_m()
        stop_distance = self._stop_distance_m()

        while self.current_waypoint_idx < len(self.reference_waypoints):
            target_waypoint = self.reference_waypoints[self.current_waypoint_idx]
            distance, yaw_error = self._target_error(obs, target_waypoint)
            if distance is None or yaw_error is None:
                print(f"OracleDiscreteActionAgent act: episode={episode_label}, pose unavailable")
                return None

            is_final_waypoint = self.current_waypoint_idx == len(self.reference_waypoints) - 1
            action = self._select_action(
                distance,
                yaw_error,
                is_final_waypoint,
                angle_threshold,
                waypoint_switch_distance,
                stop_distance,
            )
            if action == "STOP":
                print(
                    f"OracleDiscreteActionAgent act: episode={episode_label}, "
                    f"final waypoint reached at {distance:.3f}m, sending STOP"
                )
                return self._record_action("STOP")
            if action == "NEXT_WAYPOINT":
                print(
                    f"OracleDiscreteActionAgent act: episode={episode_label}, "
                    f"switching waypoint {self.current_waypoint_idx + 1}/"
                    f"{len(self.reference_waypoints)} at {distance:.3f}m"
                )
                self.current_waypoint_idx += 1
                continue

            print(
                f"OracleDiscreteActionAgent act: episode={episode_label}, "
                f"waypoint={self.current_waypoint_idx + 1}/{len(self.reference_waypoints)}, "
                f"distance={distance:.3f}m, yaw_error={math.degrees(yaw_error):.1f}deg, "
                f"action={action}"
            )
            return self._record_action(action)

        print(f"OracleDiscreteActionAgent act: episode={episode_label}, all waypoints consumed, sending STOP")
        return self._record_action("STOP")

    def get_result_info(self):
        return {
            "oracle_discrete_waypoint_idx": self.current_waypoint_idx,
            "oracle_discrete_action_counts": self.action_counts,
        }
