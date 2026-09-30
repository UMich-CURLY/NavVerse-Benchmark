import atexit
import ctypes
import io
import math
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from baselines.base.base_agent import BaseAgent
from navverse.termination_cfg import DEFAULT_EPISODE_TIMEOUT_S


def _set_pdeathsig():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


class HTTPActionServerAgent(BaseAgent):
    _SERVER_HEALTH_TIMEOUT_S = 900

    def __init__(
        self,
        args,
        *,
        name,
        argument_prefix,
        serve_script,
        default_checkpoint,
        default_port,
        default_env_name,
        python_env_var,
        uses_depth=False,
        server_cli_args=(),
    ):
        super().__init__(args)
        self.args = args
        self.name = name
        self.argument_prefix = argument_prefix
        self.uses_depth = uses_depth
        self.l_step = 0
        self.worker_id = int(getattr(args, "process_label", 0) or 0)
        self._server_proc = None
        self._server_log_handle = None
        self._needs_reset = True

        base_port = int(self._arg("base_port", default_port) or default_port)
        self._server_port = base_port + self.worker_id
        explicit_url = self._arg("server_url", "") or ""
        if explicit_url:
            try:
                self.server_url = explicit_url.format(port=self._server_port, worker_id=self.worker_id)
            except (KeyError, IndexError):
                self.server_url = explicit_url
            print(f"[{self.name} w{self.worker_id}] using external server {self.server_url}")
        else:
            python = self._resolve_server_python(default_env_name, python_env_var)
            checkpoint = self._arg("checkpoint", "") or default_checkpoint
            self.server_url = self._spawn_internal_server(
                python,
                Path(serve_script),
                checkpoint,
                list(server_cli_args),
            )

        action_map = self._arg("action_map", "STOP,FORWARD,TURN_LEFT_SMALL,TURN_RIGHT_SMALL")
        self.action_space_list = [item.strip() for item in action_map.split(",") if item.strip()]
        if len(self.action_space_list) != 4:
            raise ValueError(
                f"--{self.argument_prefix}_action_map must contain exactly four comma-separated "
                "NavVerse actions for model action ids [0, 1, 2, 3]"
            )
        self.request_timeout = self._arg("request_timeout", None) or DEFAULT_EPISODE_TIMEOUT_S
        self.max_steps = int(self._arg("max_steps", 500) or 0)
        self.depth_scale_m = float(self._arg("depth_scale_m", 10.0) or 10.0)
        self.reset()

    def _arg(self, suffix, default=None):
        return getattr(self.args, f"{self.argument_prefix}_{suffix}", default)

    def _resolve_server_python(self, env_name, env_var):
        for candidate in (
            self._arg("server_python", "") or "",
            os.environ.get(env_var, ""),
            os.path.expanduser(f"~/miniconda3/envs/{env_name}/bin/python"),
            os.path.expanduser(f"~/anaconda3/envs/{env_name}/bin/python"),
        ):
            if candidate and os.path.exists(candidate):
                return candidate
        raise RuntimeError(
            f"Could not locate the {self.name} server python. Set "
            f"--{self.argument_prefix}_server_python or {env_var}."
        )

    def _spawn_internal_server(self, python, serve_script, checkpoint, extra_args):
        if not serve_script.exists():
            raise FileNotFoundError(serve_script)

        log_dir = os.path.dirname(getattr(self.args, "log_file", "") or "") or os.path.join(
            "dump", getattr(self.args, "exp_name", "") or self.argument_prefix
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.argument_prefix}_server_{self.worker_id}.log")
        self._server_log_handle = open(log_path, "a", encoding="utf-8", buffering=1)

        cmd = [
            python,
            "-u",
            str(serve_script),
            "--checkpoint",
            checkpoint,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._server_port),
            *extra_args,
        ]
        print(
            f"[{self.name} w{self.worker_id}] spawning server on port={self._server_port} "
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} log={log_path}"
        )
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=self._server_log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            preexec_fn=_set_pdeathsig,
        )
        atexit.register(self._cleanup_server)

        health_url = f"http://127.0.0.1:{self._server_port}/health"
        self._wait_for_server_health(health_url, log_path)
        return f"http://127.0.0.1:{self._server_port}/act"

    def _wait_for_server_health(self, health_url, log_path):
        deadline = time.time() + self._SERVER_HEALTH_TIMEOUT_S
        last_log_s = 0.0
        while time.time() < deadline:
            if self._server_proc is not None and self._server_proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} server exited with code {self._server_proc.returncode} before becoming "
                    f"healthy; see {log_path}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    response.read()
                print(f"[{self.name} w{self.worker_id}] server ready at {health_url}")
                return
            except Exception:
                pass
            now = time.time()
            if now - last_log_s >= 30.0:
                last_log_s = now
                print(f"[{self.name} w{self.worker_id}] waiting for {health_url}")
            time.sleep(3.0)
        self._cleanup_server()
        raise TimeoutError(f"{self.name} server did not become healthy: {health_url}")

    def _cleanup_server(self):
        proc = self._server_proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception:
                pass
            self._server_proc = None
        if self._server_log_handle is not None:
            try:
                self._server_log_handle.close()
            except Exception:
                pass
            self._server_log_handle = None

    def shutdown(self):
        self._cleanup_server()

    def __del__(self):
        try:
            self._cleanup_server()
        except Exception:
            pass

    def reset(self):
        self.l_step = 0
        self.vis_frames = {"rgb": []}
        self._needs_reset = True
        print(f"{self.name} reset; worker={self.worker_id} server={self.server_url}")

    @staticmethod
    def _encode_rgb(rgb):
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="JPEG")
        return buffer.getvalue()

    @staticmethod
    def _encode_depth(depth):
        depth_mm = np.nan_to_num(np.asarray(depth), nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
        depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        buffer = io.BytesIO()
        Image.fromarray(depth_mm).save(buffer, format="PNG")
        return buffer.getvalue()

    def _instruction(self, goal, instruction_type=None, target_type=None):
        if instruction_type is None:
            instruction_type = "vln" if getattr(self.args, "task_type", None) == "vln" else "target"
        if target_type is None:
            target_type = "place" if getattr(self.args, "task_type", None) == "placenav" else "obj"
        if instruction_type == "target" and target_type == "obj":
            return f"Find the {goal} and stop when you are close to it."
        return str(goal)

    def _request_action(self, obs):
        files = {"rgb": ("obs.jpg", self._encode_rgb(obs["rgb"]), "image/jpeg")}
        if self.uses_depth:
            if obs.get("depth") is None:
                raise ValueError(f"{self.name} requires a depth observation")
            files["depth"] = ("depth.png", self._encode_depth(obs["depth"]), "image/png")
        data = {
            "instruction": self._instruction(obs["objectgoal"], obs.get("instruction_type"), obs.get("target_type")),
            "reset": "true" if self._needs_reset else "false",
            "hfov": str(obs.get("hfov") or getattr(self.args, "hfov", 79.0)),
            "depth_scale_m": str(self.depth_scale_m),
        }
        try:
            response = requests.post(
                self.server_url,
                files=files,
                data=data,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"{self.name} inference request failed: {exc}") from exc
        self._needs_reset = False
        return result

    @staticmethod
    def _yaw_from_xyzw(quat_xyzw):
        x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @classmethod
    def _trajectory_to_waypoints(cls, obs, trajectory):
        if obs.get("position") is None or obs.get("rotation") is None:
            raise RuntimeError("A model trajectory requires position and rotation observations")

        local_xy = np.asarray(trajectory, dtype=np.float64)
        if local_xy.ndim != 2 or local_xy.shape[1] != 2 or not np.isfinite(local_xy).all():
            raise RuntimeError(f"Model returned an invalid local trajectory with shape {local_xy.shape}")

        position = np.asarray(obs["position"], dtype=np.float64)
        base_yaw = cls._yaw_from_xyzw(obs["rotation"])
        cos_yaw = math.cos(base_yaw)
        sin_yaw = math.sin(base_yaw)
        waypoints = []
        previous_local = local_xy[0] if len(local_xy) else np.zeros(2, dtype=np.float64)
        previous_yaw = base_yaw
        for local_point in local_xy[1:]:
            delta = local_point - previous_local
            if np.linalg.norm(delta) > 1e-6:
                previous_yaw = base_yaw + math.atan2(delta[1], delta[0])
            world_x = position[0] + cos_yaw * local_point[0] - sin_yaw * local_point[1]
            world_y = position[1] + sin_yaw * local_point[0] + cos_yaw * local_point[1]
            target_yaw = (previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
            waypoints.append([float(world_x), float(world_y), float(target_yaw)])
            previous_local = local_point
        if not waypoints:
            raise RuntimeError("Model returned an empty local trajectory")
        return waypoints

    def act(self, obs, step_data_queue=None):
        start_time = time.time()
        result = self._request_action(obs)
        latency_ms = (time.time() - start_time) * 1000.0
        self.l_step += 1
        self.vis_frames["rgb"].append(obs["rgb"].copy())
        if self.max_steps > 0 and self.l_step >= self.max_steps:
            return {"action": "STOP"}

        if result.get("trajectory") is not None:
            waypoints = self._trajectory_to_waypoints(obs, result["trajectory"])
            print(
                f"[{self.name} step {self.l_step - 1}] latency={latency_ms:.2f}ms "
                f"waypoints={len(waypoints)} target={waypoints[-1]}"
            )
            return {"waypoint": waypoints}

        action_id = int(result.get("action_id", -1))
        if not 0 <= action_id < len(self.action_space_list):
            raise RuntimeError(f"{self.name} server returned invalid action_id={action_id}: {result}")
        action = self.action_space_list[action_id]
        print(
            f"[{self.name} step {self.l_step - 1}] latency={latency_ms:.2f}ms "
            f"action_id={action_id} action={action}"
        )
        return {"action": action}
