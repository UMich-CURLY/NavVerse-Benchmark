import atexit
import ctypes
import io
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import requests
from PIL import Image

from baselines.base.base_agent import BaseAgent
from navverse.termination_cfg import DEFAULT_EPISODE_TIMEOUT_S


def _set_pdeathsig():
    """Have the kernel kill this child if the parent process dies (Linux)."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # prctl(PR_SET_PDEATHSIG=1, signo=SIGKILL)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


class LongNavAgent(BaseAgent):
    _SERVER_HEALTH_TIMEOUT_S = 600

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.l_step = 0
        self.worker_id = int(getattr(args, "process_label", 0) or 0)
        base_port = int(getattr(args, "longnav_base_port", 8000) or 8000)
        self._server_port = base_port + self.worker_id
        self._server_proc = None
        self._server_log_handle = None
        explicit_url = getattr(args, "longnav_server_url", "") or ""
        if explicit_url:
            try:
                self.server_url = explicit_url.format(
                    port=self._server_port, worker_id=self.worker_id
                )
            except (KeyError, IndexError):
                self.server_url = explicit_url
            print(f"[LongNavAgent w{self.worker_id}] using external server {self.server_url}")
        else:
            self.server_url = self._spawn_internal_server(args)

        action_map = getattr(args, "longnav_action_map", "STOP,FORWARD,TURN_LEFT,TURN_RIGHT")
        self.action_space_list = [item.strip() for item in action_map.split(",") if item.strip()]
        if len(self.action_space_list) != 4:
            raise ValueError(
                "--longnav_action_map must contain exactly 4 comma-separated actions "
                "for action ids [0, 1, 2, 3]"
            )
        self.request_timeout = getattr(args, "longnav_request_timeout", None) or DEFAULT_EPISODE_TIMEOUT_S
        self.max_steps = getattr(args, "longnav_max_steps", 200)

    def _resolve_server_python(self, args):
        configured = getattr(args, "longnav_server_python", "") or os.environ.get("LONGNAV_SERVER_PYTHON", "")
        python = Path(configured).expanduser() if configured else Path(__file__).parent / ".venv" / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(
                "LongNav R1 environment missing. Run bash baselines/longnav_r1/setup.sh "
                "or set --longnav_server_python."
            )
        return str(python)

    def _spawn_internal_server(self, args):
        server_python = self._resolve_server_python(args)
        checkpoint = getattr(args, "longnav_checkpoint", "") or "Aasdfip/hm3d_rpp_ke_standard-checkpoint_231"
        repo_root = Path(__file__).resolve().parents[2]
        serve_script = repo_root / "baselines" / "longnav_r1" / "serve_longnav.py"
        if not serve_script.exists():
            raise FileNotFoundError(f"serve_longnav.py not found at {serve_script}")

        log_dir = os.path.dirname(getattr(args, "log_file", "") or "") or os.path.join(
            "dump", getattr(args, "exp_name", "") or "longnav"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"longnav_server_{self.worker_id}.log")
        self._server_log_handle = open(log_path, "a", encoding="utf-8", buffering=1)

        cmd = [
            server_python, "-u", str(serve_script),
            "--checkpoint", checkpoint,
            "--host", "127.0.0.1",
            "--port", str(self._server_port),
            "--attn_impl", getattr(args, "longnav_attn_impl", "sdpa") or "sdpa",
        ]
        stop_prob_threshold = getattr(args, "longnav_stop_prob_threshold", None)
        if stop_prob_threshold is not None:
            cmd.extend(["--stop_prob_threshold", str(stop_prob_threshold)])
        print(
            f"[LongNavAgent w{self.worker_id}] spawning server: pid<unknown> port={self._server_port} "
            f"attn_impl={getattr(args, 'longnav_attn_impl', 'sdpa') or 'sdpa'} "
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} log={log_path}"
        )

        server_env = os.environ.copy()
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=self._server_log_handle,
            stderr=subprocess.STDOUT,
            env=server_env,
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
                    f"LongNav server (worker {self.worker_id}) exited with "
                    f"code {self._server_proc.returncode} before becoming healthy; "
                    f"see {log_path}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    resp.read()
                print(f"[LongNavAgent w{self.worker_id}] server ready at {health_url}")
                return
            except Exception:
                pass
            now = time.time()
            if now - last_log_s >= 30.0:
                last_log_s = now
                print(f"[LongNavAgent w{self.worker_id}] waiting for {health_url}")
            time.sleep(3.0)
        self._cleanup_server()
        raise TimeoutError(f"LongNav server did not become healthy: {health_url}")

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

    def __del__(self):
        try:
            self._cleanup_server()
        except Exception:
            pass

    def shutdown(self):
        self._cleanup_server()

    def reset(self):
        self.l_step = 0
        self.vis_frames = {"rgb": []}
        print(
            f"LongNavAgent reset; worker={self.worker_id} "
            f"server={self.server_url}"
        )

    def get_image(self, arr):
        """Helper to encode an obs frame as JPEG bytes for the server."""
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()

    def get_action(self, image_bytes, instruction=None):
        """
        Stateless client function.
        Args:
            image_bytes (bytes): The JPEG/PNG encoded bytes of the image.
            instruction (str, optional): Text instruction. If present, resets server state.
        Returns:
            dict: The JSON response containing 'action_id' and 'action_probs'.
        """
        files = {'file': ('obs.jpg', image_bytes, 'image/jpeg')}
        data = {}
        if instruction is not None:
            data['instr_or_goal'] = instruction
        try:
            response = requests.post(self.server_url, files=files, data=data, timeout=self.request_timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LongNav inference request failed: {e}") from e

    def act(self, obs, step_data_queue=None):
        if self.max_steps > 0 and self.l_step >= self.max_steps:
            self.l_step += 1
            return {"action": "STOP"}

        instruction = obs["objectgoal"] if self.l_step == 0 else None
        if self.l_step == 0:
            print(f"\n[Step 0] Sending RESET + Instruction: {instruction}")

        img_bytes = self.get_image(obs["rgb"])
        start_time = time.time()
        result = self.get_action(img_bytes, instruction)
        latency = (time.time() - start_time) * 1000
        if result is None:
            raise RuntimeError("LongNav inference server returned no result")

        action_id = int(result['action_id'])
        action_probs = result.get("action_probs") or []
        stop_prob = action_probs[0] if len(action_probs) > 0 else None
        chosen_prob = action_probs[action_id] if 0 <= action_id < len(action_probs) else None
        prob_suffix = ""
        if stop_prob is not None and chosen_prob is not None:
            prob_suffix = f" | StopP: {float(stop_prob):.4f} | ChosenP: {float(chosen_prob):.4f}"
        print(f"[Step {self.l_step}] Latency: {latency:.2f} ms | Action ID: {action_id}{prob_suffix}")
        if action_id < 0 or action_id >= len(self.action_space_list):
            raise RuntimeError(f"Invalid LongNav action_id={action_id}")
        action = self.action_space_list[action_id]

        self.l_step += 1
        self.vis_frames["rgb"].append(obs["rgb"].copy())
        return {"action": action}

    def get_result_info(self):
        return {"longnav_policy": "LongNav-R1", "longnav_output": "discrete_action"}
