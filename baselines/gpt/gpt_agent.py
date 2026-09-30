import math
from pathlib import Path
import time

import numpy as np

from baselines.base.http_action_agent import HTTPActionServerAgent


class GPTAgent(HTTPActionServerAgent):
    def __init__(self, args):
        repo_root = Path(__file__).resolve().parents[2]
        super().__init__(
            args,
            name="GPT",
            argument_prefix="gpt",
            serve_script=repo_root / "baselines" / "gpt" / "serve_gpt.py",
            default_checkpoint=getattr(args, "gpt_model", "gpt-6-astra"),
            default_port=8300,
            default_env_name="gpt",
            python_env_var="GPT_SERVER_PYTHON",
            server_cli_args=(
                "--reasoning-effort",
                getattr(args, "gpt_reasoning_effort", "low") or "low",
                "--compact-at-fraction",
                str(getattr(args, "gpt_compact_at_fraction", 0.70)),
                "--stop-range-m",
                str(getattr(args, "gpt_stop_range_m", 1.6)),
                "--max-steps",
                str(getattr(args, "gpt_max_steps", 175)),
            ),
        )
        expected_actions = ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        if self.action_space_list != expected_actions:
            raise ValueError(
                "GPT requires action ids ordered as "
                f"{expected_actions}, got {self.action_space_list}"
            )

    def act(self, obs, step_data_queue=None):
        del step_data_queue
        started = time.perf_counter()
        result = self._request_action(obs)
        latency_s = time.perf_counter() - started
        self.l_step += 1
        self.vis_frames["rgb"].append(obs["rgb"].copy())

        if self.max_steps > 0 and self.l_step >= self.max_steps:
            return {"action": "STOP"}

        action_id = int(result.get("action_id", -1))
        if not 0 <= action_id < len(self.action_space_list):
            raise RuntimeError(f"GPT returned invalid action_id={action_id}: {result}")
        action = self.action_space_list[action_id]
        if str(result.get("action", "")).upper() != action:
            raise RuntimeError(f"GPT action/action_id mismatch: {result}")
        if action == "STOP":
            return {"action": "STOP"}

        step_size = float(result.get("step_size", float("nan")))
        min_size, max_size = (0.1, 5.0) if action == "FORWARD" else (5.0, 180.0)
        if not math.isfinite(step_size) or not min_size <= step_size <= max_size:
            raise RuntimeError(f"GPT returned invalid {action} step_size={step_size}: {result}")

        position = np.asarray(obs["position"], dtype=np.float64)
        yaw = self._yaw_from_xyzw(obs["rotation"])
        target = position.copy()
        target_yaw = yaw
        if action == "FORWARD":
            target[0] += step_size * math.cos(yaw)
            target[1] += step_size * math.sin(yaw)
            unit = "m"
        else:
            direction = 1.0 if action == "TURN_LEFT" else -1.0
            target_yaw = (yaw + direction * math.radians(step_size) + math.pi) % (
                2.0 * math.pi
            ) - math.pi
            unit = "deg"

        waypoint = [[float(target[0]), float(target[1]), float(target_yaw)]]
        print(
            f"[{self.name} step {self.l_step - 1}] latency={latency_s * 1000:.2f}ms "
            f"action={action} step_size={step_size:g}{unit} waypoint={waypoint[0]}"
        )
        return {"waypoint": waypoint}
