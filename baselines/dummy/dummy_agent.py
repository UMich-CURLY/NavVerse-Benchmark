import math
import time
from baselines.base.base_agent import BaseAgent
import numpy as np

class DummyAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.l_step = 0
        self.vis_frames = {"rgb": []}
        self.action_plan = []
        self.pending_action = None
        self.action_test_records = []
        self.forward_heading_yaw = None

    @staticmethod
    def _split_csv(value):
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @staticmethod
    def _wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _yaw_from_xyzw(quat_xyzw):
        x, y, z, w = quat_xyzw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _forward_distance(self, episode_label):
        override = getattr(self.args, "discrete_action_forward_distance_m", None)
        if override is not None:
            return float(override)
        scene_id = "_".join(str(episode_label).split("_")[:-1])
        if scene_id.startswith(("vc_", "vcp_", "innout_")):
            return 1.0
        return 0.25

    def _pose_from_obs(self, obs):
        pos = np.asarray(obs["position"], dtype=np.float64)
        yaw = self._yaw_from_xyzw(obs["rotation"])
        return pos, yaw

    def _target_for_action(self, obs, action):
        pos, yaw = self._pose_from_obs(obs)
        forward_distance = self._forward_distance(obs.get("episode_label", ""))
        lateral_distance = 0.25
        turn_angle = math.radians(float(getattr(self.args, "dummy_action_turn_deg", 30.0)))
        target = pos.copy()
        target_yaw = yaw
        local_dx = 0.0
        local_dy = 0.0
        if action == "FORWARD":
            local_dx = forward_distance
            if self.forward_heading_yaw is None:
                self.forward_heading_yaw = yaw
            yaw = self.forward_heading_yaw
        elif action == "LEFT":
            local_dy = lateral_distance
            self.forward_heading_yaw = None
        elif action == "RIGHT":
            local_dy = -lateral_distance
            self.forward_heading_yaw = None
        elif action == "TURN_LEFT":
            target_yaw = yaw + turn_angle
            self.forward_heading_yaw = None
        elif action == "TURN_RIGHT":
            target_yaw = yaw - turn_angle
            self.forward_heading_yaw = None
        elif action == "TURN_LEFT_SMALL":
            target_yaw = yaw + turn_angle / 2.0
            self.forward_heading_yaw = None
        elif action == "TURN_RIGHT_SMALL":
            target_yaw = yaw - turn_angle / 2.0
            self.forward_heading_yaw = None
        else:
            raise ValueError(f"Unsupported dummy action test action: {action}")
        if local_dx or local_dy:
            target[0] += local_dx * math.cos(yaw) - local_dy * math.sin(yaw)
            target[1] += local_dx * math.sin(yaw) + local_dy * math.cos(yaw)
        return [float(target[0]), float(target[1]), float(target_yaw)]

    def _action_error(self, obs, target):
        pos, yaw = self._pose_from_obs(obs)
        pos_error = float(np.linalg.norm(pos[:2] - np.asarray(target[:2], dtype=np.float64)))
        yaw_error = abs(self._wrap_pi(yaw - target[2]))
        return pos_error, yaw_error

    def _record_pending_action(self, obs):
        if self.pending_action is None:
            return
        pending = self.pending_action
        self.pending_action = None
        pos_error, yaw_error = self._action_error(obs, pending["target"])
        sim_time = obs.get("sim_time")
        start_sim_time = pending.get("start_sim_time")
        sim_elapsed = None
        if sim_time is not None and start_sim_time is not None:
            sim_elapsed = float(sim_time) - float(start_sim_time)
        wall_elapsed = time.time() - pending["start_wall_time"]
        pos_threshold = float(getattr(self.args, "dummy_action_pos_threshold_m", 0.12))
        yaw_threshold = math.radians(float(getattr(self.args, "dummy_action_yaw_threshold_deg", 7.0)))
        max_sim_action_s = float(getattr(self.args, "dummy_action_max_sim_time_s", 3.0))
        turn_action = str(pending["action"]).startswith("TURN_")
        reached = yaw_error <= yaw_threshold if turn_action else pos_error <= pos_threshold
        in_time = sim_elapsed is not None and sim_elapsed <= max_sim_action_s + 1e-6
        status = obs.get("last_action_status") or {}
        if reached and in_time:
            reason = "reached"
        elif reached:
            reason = "reached_after_sim_time_limit"
        elif status.get("timed_out"):
            reason = "worker_action_timeout"
        else:
            reason = "not_reached_before_next_observation"
        record = {
            "episode_label": obs.get("episode_label", ""),
            "trajectory": pending.get("trajectory", "manual"),
            "action": pending["action"],
            "action_index": pending["action_index"],
            "passed": bool(reached and in_time),
            "reason": reason,
            "target": pending["target"],
            "sim_elapsed": sim_elapsed,
            "wall_elapsed": wall_elapsed,
            "final_pos_error": pos_error,
            "final_yaw_error_deg": math.degrees(yaw_error),
            "worker_action_status": status,
        }
        self.action_test_records.append(record)
        print(
            "DummyAgent action_test: "
            f"trajectory={record['trajectory']} "
            f"action={record['action']} passed={record['passed']} reason={record['reason']} "
            f"sim_elapsed={sim_elapsed if sim_elapsed is not None else 'n/a'} "
            f"pos_error={pos_error:.3f} yaw_error_deg={record['final_yaw_error_deg']:.2f}"
        )

    def _build_action_plan(self):
        suite = self._split_csv(getattr(self.args, "dummy_trajectory_suite", ""))
        if suite:
            return self._build_trajectory_suite(suite)
        sequence = self._split_csv(getattr(self.args, "dummy_action_sequence", "FORWARD,TURN_LEFT,TURN_RIGHT"))
        repeats = max(1, int(getattr(self.args, "dummy_action_repeats", 1) or 1))
        return [{"trajectory": "manual", "action": action} for action in sequence * repeats]

    @staticmethod
    def _repeat_actions(trajectory, sequence, repeats=1):
        return [
            {"trajectory": trajectory, "action": action}
            for _ in range(repeats)
            for action in sequence
        ]

    def _build_trajectory_suite(self, suite):
        plan = []
        for name in suite:
            if name == "circle":
                plan.extend(self._repeat_actions(name, ["FORWARD", "TURN_LEFT"], repeats=12))
            elif name == "square":
                plan.extend(self._repeat_actions(name, ["FORWARD", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT"], repeats=4))
            elif name.startswith("zigzag"):
                try:
                    z_count = int(name.removeprefix("zigzag") or 3)
                except ValueError as exc:
                    raise ValueError(f"Invalid dummy trajectory name: {name}") from exc
                for index in range(max(1, z_count)):
                    trajectory = f"{name}_{index + 1}"
                    turn_out = "TURN_LEFT" if index % 2 == 0 else "TURN_RIGHT"
                    turn_back = "TURN_RIGHT" if index % 2 == 0 else "TURN_LEFT"
                    plan.extend(self._repeat_actions(trajectory, ["FORWARD", turn_out, "FORWARD", turn_back], repeats=1))
            else:
                raise ValueError(f"Unsupported dummy trajectory name: {name}")
        return plan

    def reset(self):
        self.l_step = 0
        self.vis_frames = {"rgb": []}
        self.action_plan = self._build_action_plan()
        self.pending_action = None
        self.action_test_records = []
        self.forward_heading_yaw = None
        printable_plan = [f"{item['trajectory']}:{item['action']}" for item in self.action_plan]
        print(f"DummyAgent reset: action_plan={printable_plan}")

    def act(self, obs, step_data_queue=None):
        self._record_pending_action(obs)
        sleep_s = float(getattr(self.args, "dummy_action_sleep_s", 0.0) or 0.0)
        if sleep_s > 0:
            time.sleep(sleep_s)
        self.l_step += 1
        if obs.get("rgb") is not None:
            self.vis_frames["rgb"].append(obs["rgb"].copy())
        print(f"DummyAgent act: step={self.l_step}, target={obs['objectgoal']}")
        if self.l_step <= len(self.action_plan):
            plan_item = self.action_plan[self.l_step - 1]
            action_name = plan_item["action"]
            self.pending_action = {
                "action": action_name,
                "trajectory": plan_item["trajectory"],
                "action_index": self.l_step,
                "target": self._target_for_action(obs, action_name),
                "start_sim_time": obs.get("sim_time"),
                "start_wall_time": time.time(),
            }
            print(f"DummyAgent act: trajectory={plan_item['trajectory']} action={action_name}")
            return {"action": action_name}
        else:
            return {"action": "STOP"}

    def get_result_info(self):
        total = len(self.action_test_records)
        passed = sum(1 for record in self.action_test_records if record.get("passed"))
        sim_elapsed_values = [
            record["sim_elapsed"]
            for record in self.action_test_records
            if record.get("sim_elapsed") is not None
        ]
        pos_errors = [record["final_pos_error"] for record in self.action_test_records]
        return {
            "dummy_action_total": total,
            "dummy_action_passed": passed,
            "dummy_action_failed": total - passed,
            "dummy_action_pass_rate": passed / total if total > 0 else None,
            "dummy_action_max_sim_elapsed": max(sim_elapsed_values) if sim_elapsed_values else None,
            "dummy_action_max_pos_error": max(pos_errors) if pos_errors else None,
            "action_test_records": self.action_test_records,
        }
