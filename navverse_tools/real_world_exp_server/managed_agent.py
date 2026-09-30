from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from navverse_tools.real_world_exp_server.agent import AgentError, HTTPAgent


StateCallback = Callable[[], None]


@dataclass(frozen=True)
class LaunchSpec:
    env_name: str
    script: Path
    args: tuple[str, ...]
    health_url: str
    startup_timeout_s: float = 900.0
    environment: dict[str, str] = field(default_factory=dict)
    python: Path | None = None


class ManagedHTTPAgent:
    def __init__(
        self,
        profile_id: str,
        delegate: HTTPAgent,
        launch: LaunchSpec,
        repo_root: Path,
        log_dir: Path,
    ):
        self.profile_id = profile_id
        self.delegate = delegate
        self.launch = launch
        self.repo_root = repo_root
        self.log_dir = log_dir
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._state = "stopped"
        self._detail = "Select this agent to start it on GPU 0."
        self._idle_timer: threading.Timer | None = None
        self._idle_deadline: float | None = None
        self._lock = threading.RLock()
        atexit.register(self.shutdown)

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._state == "ready" and not self._healthy():
                self._state = "stopped"
                self._detail = "Model server is not running."
            detail = self._detail
            if self._state == "ready" and self._idle_deadline is not None:
                remaining_s = max(0, math.ceil(self._idle_deadline - time.monotonic()))
                detail = f"Model ready on GPU 0; idle shutdown in {remaining_s // 60}:{remaining_s % 60:02d}."
            return {
                "state": self._state,
                "detail": detail,
                "auto_start": True,
                "selectable": True,
                "gpu": "0",
            }

    def ensure_ready(self, on_state_change: StateCallback | None = None) -> None:
        with self._lock:
            if self._process is not None and self._healthy():
                self._cancel_idle_shutdown()
                self._set_state("ready", "Model server is ready on GPU 0.", on_state_change)
                return
            if self._process is None and self._healthy():
                self._set_state(
                    "error",
                    "The configured port is occupied by an unmanaged server; refusing it because GPU 0 cannot be verified.",
                    on_state_change,
                )
                raise AgentError(self._detail)
            if self._process is not None and self._process.poll() is not None:
                self._close_process()
            if self._process is None:
                self._launch(on_state_change)

        deadline = time.monotonic() + self.launch.startup_timeout_s
        last_publish = 0.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._healthy():
                    self._set_state("ready", "Model server is ready on GPU 0.", on_state_change)
                    return
                if self._process is not None and self._process.poll() is not None:
                    code = self._process.returncode
                    self._set_state(
                        "error",
                        f"Model process exited with code {code}; see {self.log_path}.",
                        on_state_change,
                    )
                    self._close_process()
                    raise AgentError(self._detail)
            if time.monotonic() - last_publish >= 5.0:
                last_publish = time.monotonic()
                if on_state_change is not None:
                    on_state_change()
            time.sleep(2.0)

        with self._lock:
            self._set_state(
                "error",
                f"Model did not become healthy within {self.launch.startup_timeout_s:.0f}s.",
                on_state_change,
            )
            self._terminate_process()
        raise AgentError(self._detail)

    def act(self, **kwargs: Any) -> dict[str, Any]:
        self.ensure_ready()
        return self.delegate.act(**kwargs)

    def reset(self) -> bool:
        with self._lock:
            if self._process is None or self._process.poll() is not None or not self._healthy():
                return False
            reset = getattr(self.delegate, "reset", None)
            if reset is None:
                return False
            applied = bool(reset())
            if applied:
                self._detail = "Model state reset; ready on GPU 0."
            return applied

    def schedule_idle_shutdown(
        self,
        timeout_s: float,
        on_state_change: StateCallback | None = None,
    ) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            self._cancel_idle_shutdown()
            self._idle_deadline = time.monotonic() + timeout_s
            self._idle_timer = threading.Timer(
                timeout_s,
                self._shutdown_after_idle,
                args=(timeout_s, on_state_change),
            )
            self._idle_timer.daemon = True
            self._idle_timer.start()
            self._set_state(
                "ready",
                f"Episode finished; keeping model on GPU 0 for {timeout_s / 60:g} idle minutes.",
                on_state_change,
            )

    def stop(self, reason: str) -> bool:
        with self._lock:
            changed = self._process is not None or self._state != "stopped"
            self._cancel_idle_shutdown()
            self._terminate_process()
            self._set_state("stopped", reason, None)
            return changed

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"{self.profile_id}.log"

    def _launch(self, on_state_change: StateCallback | None) -> None:
        python = str(self.launch.python) if self.launch.python else resolve_env_python(self.launch.env_name)
        if self.launch.python and not self.launch.python.is_file():
            raise AgentError(f"Python interpreter does not exist: {self.launch.python}")
        if not self.launch.script.is_file():
            raise AgentError(f"Launch script does not exist: {self.launch.script}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab", buffering=0)
        command = [python, "-u", str(self.launch.script), *self.launch.args]
        environment = os.environ.copy()
        environment.update(self.launch.environment)
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "0",
            }
        )
        self._process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=environment,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._set_state(
            "starting",
            f"Starting on physical GPU 0; log: {self.log_path}.",
            on_state_change,
        )

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.launch.health_url, timeout=2) as response:
                response.read(256)
                return 200 <= response.status < 300
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _set_state(
        self,
        state: str,
        detail: str,
        on_state_change: StateCallback | None,
    ) -> None:
        changed = state != self._state or detail != self._detail
        self._state = state
        self._detail = detail
        if changed and on_state_change is not None:
            on_state_change()

    def _terminate_process(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._close_process()

    def _close_process(self) -> None:
        self._cancel_idle_shutdown()
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def shutdown(self) -> None:
        self.stop("Bridge stopped the model server.")

    def _shutdown_after_idle(
        self,
        timeout_s: float,
        on_state_change: StateCallback | None,
    ) -> None:
        with self._lock:
            if self._idle_timer is None:
                return
            self._idle_timer = None
            self._idle_deadline = None
            self._terminate_process()
            self._set_state(
                "stopped",
                f"Stopped after {timeout_s / 60:g} minutes without another episode.",
                on_state_change,
            )

    def _cancel_idle_shutdown(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = None
        self._idle_deadline = None


class BlockedAgent:
    def __init__(self, reason: str):
        self.reason = reason

    def status(self) -> dict[str, Any]:
        return {
            "state": "blocked",
            "detail": self.reason,
            "auto_start": False,
            "selectable": False,
            "gpu": "0",
        }

    def act(self, **_kwargs: Any) -> dict[str, Any]:
        raise AgentError(self.reason)


def resolve_env_python(env_name: str) -> str:
    current_env = Path(sys.prefix).name
    if current_env == env_name:
        return sys.executable
    for root in (Path.home() / "miniconda3", Path.home() / "anaconda3"):
        candidate = root / "envs" / env_name / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    raise AgentError(f"Conda environment {env_name!r} is not installed")
