import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import ray

# Ensure utils can be imported
sys.path.append(os.getcwd())

from arguments import get_args
from evaluation_worker import EvaluationWorker
from main_remote import foxglove_vis_thread_func
from preset_config import list_presets, resolve_preset_argv
from navverse.episode import VLNEpisode, load_episode_set
from navverse.launcher.config import base_config
from navverse.launcher.discovery import get_split, resolve_split_episode_folder
from navverse.launcher.registry import AGENT_REGISTRY
from navverse.termination_cfg import DEFAULT_EPISODE_TIMEOUT_S
from utils.ray_utils import (
    RayEvent,
    RayProcess,
    RayQueue,
    RayTaskQueue,
    best_effort_kill,
    best_effort_remote_call,
    can_connect,
    find_available_port_block,
    is_tcp_port_available,
    ray_ref_ready,
    socket_request_responds,
    tcp_listener_pids,
    terminate_process_group,
    terminate_processes_matching_cmdline,
    terminate_tcp_port_listeners,
    detect_first_gpu_total_vram_gb,
    wait_for_tcp_port_release,
)
from navverse.utils.socket_client import request_data, send_message
from utils.logging import RuntimeLogger, ensure_repo_log_symlink, setup_tee
from utils.wandb_logging import WANDB_PROGRESS_INTERVAL_S, save_result_thread_func, save_step_data_thread_func


@dataclass
class WorkerLayout:
    """Static launch plan for one worker slot.

    Responsibility:
    - Binds a logical worker id to the sim port and per-agent log file.

    Flow:
    - `BenchmarkManager.allocate_workers()` creates one layout per worker.
    - The same layout is passed to both `SimWorker` and `AgentWorker`.
    - On restart, the manager may replace the layout with a new port.
    """
    worker_id: int
    sim_port: int
    agent_log_file: str
    sim_log_file: str
    gpu_hint: str = ""


@dataclass
class WorkerRuntimeState:
    """Mutable status snapshot for one worker pair.

    Responsibility:
    - Stores the latest sim/agent state that the manager and watchdog inspect.

    Flow:
    - Initialized when a worker pair starts.
    - Updated by `SimWorker`, `AgentWorker`, and `BenchmarkManager`.
    - Read by the watchdog to decide whether a worker should continue or restart.
    """
    worker_id: int
    sim_port: int
    assigned_episode: Optional[str] = None
    scene_id: Optional[str] = None
    sim_status: str = "init"
    agent_status: str = "init"
    sim_pid: Optional[int] = None
    sim_cuda_visible_devices: str = ""
    agent_cuda_visible_devices: str = ""
    agent_step: int = 0
    agent_last_action: str = ""
    sim_frame_count: int = 0
    sim_fps: float = 0.0
    sim_to_wall_ratio: float = 0.0
    sim_total_time_s: float = 0.0
    last_message: str = ""
    updated_at_s: float = 0.0
    sim_status_changed_at_s: float = 0.0
    last_sim_response_s: float = 0.0
    restart_count: int = 0


def apply_requested_cuda_visible_devices(eval_args):
    requested_gpu_ids = ",".join(item.strip() for item in str(eval_args.gpu_ids or "").split(",") if item.strip())
    env_gpu_ids = ",".join(
        item.strip()
        for item in str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
        if item.strip()
    )
    if requested_gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_gpu_ids
        eval_args.cuda_visible_devices = requested_gpu_ids
    else:
        eval_args.cuda_visible_devices = env_gpu_ids
    return eval_args.cuda_visible_devices


def resolve_worker_cuda_visible_devices(requested_visible_devices, preferred_gpu_id=""):
    if preferred_gpu_id:
        return str(preferred_gpu_id)
    accelerator_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    assigned_gpu_ids = [str(gpu_id) for gpu_id in accelerator_ids]
    if not assigned_gpu_ids:
        return requested_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")

    visible_gpu_ids = [
        item.strip()
        for item in str(requested_visible_devices or "").split(",")
        if item.strip()
    ]
    if visible_gpu_ids:
        if all(gpu_id in visible_gpu_ids for gpu_id in assigned_gpu_ids):
            return ",".join(assigned_gpu_ids)

        mapped_gpu_ids = []
        for gpu_id in assigned_gpu_ids:
            if gpu_id.isdigit():
                logical_index = int(gpu_id)
                if logical_index < len(visible_gpu_ids):
                    mapped_gpu_ids.append(visible_gpu_ids[logical_index])
                    continue
            mapped_gpu_ids.append(gpu_id)
        return ",".join(mapped_gpu_ids)

    return ",".join(assigned_gpu_ids)


def resolve_conda_executable():
    return (
        os.environ.get("CONDA_EXE")
        or shutil.which("conda")
        or os.path.expanduser("~/miniconda3/bin/conda")
    )


READY_SIM_STATUSES = {"ready", "running"}


@ray.remote(num_cpus=0)
class AgentManager:
    """Shared runtime services for all agent workers.

    Responsibility:
    - Owns the result/log queues and the background processes that drain them.

    Flow:
    - `BenchmarkManager.start()` creates this actor once for the whole run.
    - `start_background_processes()` launches result saving and optional Foxglove logging.
    - `AgentWorker.run()` fetches the shared queues through `get_context()`.
    - `shutdown()` stops the background processes during benchmark teardown.
    """
    def __init__(self, args):
        self.logger = RuntimeLogger("AgentManager")
        self.args = args
        self.result_queue = RayQueue()
        self.progress_queue = RayQueue()
        self.step_data_queue = RayQueue()
        self.foxglove_log_queue = RayQueue()
        self.quit_event = RayEvent()
        self.save_result_process = None
        self.save_step_data_process = None
        self.foxglove_vis_process = None
        self.logger.step("init")

    def __repr__(self):
        return "agent-runtime"

    def start_background_processes(self):
        self.logger.step("start_background_processes_begin")
        self.save_result_process = RayProcess(
            name="SaveResultProcess",
            target=save_result_thread_func,
            args=(self.args, self.result_queue, self.quit_event, self.progress_queue),
        )
        self.save_result_process.start()
        self.logger.step("save_result_started")
        if self.args.save_step_data:
            self.save_step_data_process = RayProcess(
                name="SaveStepDataProcess",
                target=save_step_data_thread_func,
                args=(self.args, self.step_data_queue, self.quit_event),
            )
            self.save_step_data_process.start()
            self.logger.step("save_step_data_started")
        if self.args.foxglove:
            self.foxglove_vis_process = RayProcess(
                name="FoxgloveVisProcess",
                target=foxglove_vis_thread_func,
                args=(self.args, self.foxglove_log_queue, self.quit_event),
            )
            self.foxglove_vis_process.start()
            self.logger.step("foxglove_started")

    def get_context(self):
        return {
            "args": self.args,
            "result_queue": self.result_queue,
            "progress_queue": self.progress_queue,
            "step_data_queue": self.step_data_queue,
            "foxglove_log_queue": self.foxglove_log_queue,
        }

    def log_progress(self, snapshot):
        self.progress_queue.put(snapshot)

    def shutdown(self):
        self.logger.step("shutdown_start")
        self.quit_event.set()
        if self.save_result_process is not None:
            self.save_result_process.join()
        if self.save_step_data_process is not None:
            self.save_step_data_process.join()
        if self.foxglove_vis_process is not None:
            self.foxglove_vis_process.join()
        self.logger.step("shutdown_done")


@ray.remote(num_cpus=0)
class SimWorker:
    """Ray actor that owns exactly one simulator process.

    Responsibility:
    - Starts `scripts/navverse_evaluation.py`, keeps it bound to a port, and
      switches episodes for the paired agent worker.

    Flow:
    - The manager creates the actor with a GPU fraction and an initial task.
    - `ensure_episode()` either launches a fresh simulator or forwards a new episode.
    - The manager/watchdog polls `get_process_status()` to detect failures.
    - `cleanup()` terminates the child process when the worker finishes or restarts.
    """
    def __init__(self, worker_layout, eval_args, args, manager_name):
        self.worker_layout = worker_layout
        self.worker_id = worker_layout.worker_id
        self.sim_port = worker_layout.sim_port
        self.eval_args = eval_args
        self.args = args
        self.host = args.host
        self.manager = ray.get_actor(manager_name)
        self.logger = RuntimeLogger(f"SimWorker[{self.worker_id}]")
        self.process = None
        self.output_file = None
        self.current_task = None
        self.process_lock = threading.Lock()
        self.cuda_visible_devices = resolve_worker_cuda_visible_devices(
            getattr(self.eval_args, "cuda_visible_devices", ""),
            self.worker_layout.gpu_hint,
        )
        if self.cuda_visible_devices:
            self._update_state(sim_cuda_visible_devices=self.cuda_visible_devices)
        self.logger.step("init")

    def __repr__(self):
        return f"sim-{self.worker_id}"

    def _update_state(self, **kwargs):
        self.manager.update_worker_state.remote(self.worker_id, **kwargs)

    def _record_sim_response(self):
        self.manager.record_sim_response.remote(self.worker_id)

    def _terminate_process(self):
        if self.process is None:
            self._close_output_file()
            return
        if self.process.poll() is not None:
            self._close_output_file()
            return
        terminate_process_group(self.process.pid, timeout_s=15.0)
        try:
            self.process.wait(timeout=3.0)
        except Exception:
            pass
        if not wait_for_tcp_port_release(self.host, self.sim_port, timeout_s=20.0):
            self.logger.warning(f"[Sim {self.worker_id}] port {self.sim_port} did not release after process termination")
            terminate_tcp_port_listeners(
                self.host,
                self.sim_port,
                timeout_s=15.0,
                required_cmd_substrings=("navverse_evaluation.py",),
            )
        self._close_output_file()

    def _release_stale_port_listener(self):
        if is_tcp_port_available(self.host, self.sim_port):
            return
        pids = tcp_listener_pids(self.host, self.sim_port)
        self.logger.warning(
            f"[Sim {self.worker_id}] port {self.sim_port} is occupied before startup; listener_pids={pids}"
        )
        released = terminate_tcp_port_listeners(
            self.host,
            self.sim_port,
            timeout_s=20.0,
            required_cmd_substrings=("navverse_evaluation.py",),
        )
        if not released:
            raise RuntimeError(f"Port {self.sim_port} is unavailable for sim startup")

    def _close_output_file(self):
        if self.output_file is None:
            return
        try:
            self.output_file.close()
        except Exception:
            pass
        self.output_file = None

    def _wait_for_socket_ready(self, timeout_s=DEFAULT_EPISODE_TIMEOUT_S):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process is None:
                raise RuntimeError(f"[Sim {self.worker_id}] process record disappeared while waiting for socket")
            if self.process.poll() is not None:
                raise RuntimeError(f"[Sim {self.worker_id}] exited before socket on port {self.sim_port} became ready")
            if can_connect(self.host, self.sim_port, timeout=1.0):
                self._update_state(sim_status="ready", last_message="socket ready")
                self._record_sim_response()
                return
            time.sleep(0.5)
        raise TimeoutError(f"[Sim {self.worker_id}] timed out waiting for socket on port {self.sim_port}")

    def _start_sim_process(self, task):
        self._release_stale_port_listener()

        episode_label = task["episode_label"]
        cmd = [
            resolve_conda_executable(),
            "run",
            "--no-capture-output",
            "-n",
            self.eval_args.conda_env,
            "python",
            "scripts/navverse_evaluation.py",
            "--episode_label",
            episode_label,
            "--port",
            str(self.sim_port),
        ]
        if getattr(self.eval_args, "episode_folder", ""):
            cmd.extend(["--episode_folder", self.eval_args.episode_folder])
        if getattr(self.eval_args, "scene_folder", ""):
            cmd.extend(["--scene_folder", self.eval_args.scene_folder])
        if self.eval_args.sim_robot_name:
            cmd.extend(["--robot_name", self.eval_args.sim_robot_name])
        if self.eval_args.sim_tidybot_embodiment:
            cmd.extend(
                ["--tidybot_embodiment", self.eval_args.sim_tidybot_embodiment]
            )
        if self.eval_args.sim_timed_trajectory_dt is not None:
            cmd.extend(
                [
                    "--timed_trajectory_control_dt",
                    str(self.eval_args.sim_timed_trajectory_dt),
                ]
            )
        if self.eval_args.controlled_terrain_manifest:
            cmd.extend(
                [
                    "--controlled_terrain_manifest",
                    self.eval_args.controlled_terrain_manifest,
                    "--controlled_terrain_type",
                    self.eval_args.controlled_terrain_type,
                ]
            )
        if self.eval_args.disable_sim_camera:
            cmd.extend(["--disable_camera", "true"])
        cmd.extend(["--instruction_type", self.args.instruction_type])
        if self.eval_args.headless:
            cmd.append("--headless")
            if int(getattr(self.eval_args, "sim_livestream", 0) or 0) > 0:
                cmd.extend(["--livestream", str(self.eval_args.sim_livestream)])
            else:
                cmd.extend(["--experience", "isaaclab.python.rendering.kit"])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.getcwd(), env.get("PYTHONPATH", "")]))

        self._update_state(
            assigned_episode=episode_label,
            scene_id=task["scene_id"],
            sim_status="starting",
            sim_cuda_visible_devices=self.cuda_visible_devices,
            last_message=f"launching sim on port {self.sim_port}",
        )
        self._record_sim_response()
        self.logger.info(f"[Sim {self.worker_id}] Command: {' '.join(cmd)}")
        self.logger.info(f"[Sim {self.worker_id}] CUDA_VISIBLE_DEVICES={self.cuda_visible_devices}")
        os.makedirs(os.path.dirname(self.worker_layout.sim_log_file), exist_ok=True)
        self.output_file = open(self.worker_layout.sim_log_file, "a", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=env,
            start_new_session=True,
            stdout=self.output_file,
            stderr=subprocess.STDOUT,
        )
        self._update_state(sim_pid=self.process.pid)
        self.logger.step(f"sim_launched pid={self.process.pid} port={self.sim_port}")
        self._wait_for_socket_ready()

    def ensure_episode(self, task):
        with self.process_lock:
            previous_task = self.current_task
            current_scene_id = previous_task.get("scene_id") if previous_task is not None else None
            self.current_task = task
            self._update_state(
                assigned_episode=task["episode_label"],
                scene_id=task["scene_id"],
                last_message=f"ensure episode {task['episode_label']}",
            )
            if self.process is None or self.process.poll() is not None:
                # start new process
                self._start_sim_process(task)
                return {"started_new_process": True, "sim_port": self.sim_port}

            if previous_task is not None and task["episode_label"] == previous_task.get("episode_label"):
                return {"started_new_process": False, "sim_port": self.sim_port, "already_loaded": True}

            if current_scene_id is not None and task["scene_id"] != current_scene_id:
                # scene changed, restart sim
                self.logger.info(
                    f"[Sim {self.worker_id}] scene changed from {current_scene_id} to {task['scene_id']}; "
                    "restarting sim to avoid cross-scene OOM"
                )
                self._update_state(
                    sim_status="starting",
                    sim_pid=None,
                    last_message=f"restart sim for scene change {current_scene_id} -> {task['scene_id']}",
                )
                self._terminate_process()
                self.process = None
                self._start_sim_process(task)
                return {"started_new_process": True, "sim_port": self.sim_port}
            else:
                # same scene, forward episode via socket server
                self._update_state(sim_status="loading", last_message=f"forward episode {task['episode_label']}")
                self._record_sim_response()
                send_message("EPISODE", {"episode_label": task["episode_label"]}, self.host, self.sim_port)
                return {"started_new_process": False, "sim_port": self.sim_port}

    def get_process_status(self):
        process_alive = self.process is not None and self.process.poll() is None
        process_pid = self.process.pid if self.process is not None else None
        return {
            "process_alive": process_alive,
            "sim_pid": process_pid,
            "sim_port": self.sim_port,
            "current_task": self.current_task,
        }

    def cleanup(self):
        with self.process_lock:
            self._terminate_process()
            self.process = None
            self.current_task = None
            self._close_output_file()
            self._update_state(sim_status="stopped", sim_pid=None, last_message="cleanup")
            self.logger.step("cleanup")


@ray.remote(num_cpus=0)
class AgentWorker:
    """Ray actor that runs one evaluation agent against one simulator port.

    Responsibility:
    - Configures the agent-side GPU environment and drives `EvaluationWorker`.

    Flow:
    - The manager creates the actor after the paired `SimWorker` is reserved.
    - `run()` pulls shared runtime state from `AgentManager`.
    - It waits until the simulator reports `ready`, then constructs `EvaluationWorker`.
    - When the agent exits, the manager treats the result as success or restart signal.
    """
    def __init__(self, worker_layout, agent_runtime_manager, task_queue, manager_name, sys_argv):
        self.worker_layout = worker_layout
        self.worker_id = worker_layout.worker_id
        self.sim_port = worker_layout.sim_port
        self.agent_runtime_manager = agent_runtime_manager
        self.task_queue = task_queue
        self.manager = ray.get_actor(manager_name)
        self.sys_argv = sys_argv
        self.logger = RuntimeLogger(f"AgentWorker[{self.worker_id}]")
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            self._update_state(agent_cuda_visible_devices=cuda_visible_devices)
        self.logger.step("init")

    def __repr__(self):
        return f"agent-{self.worker_id}"

    def _update_state(self, **kwargs):
        # Avoid blocking on manager state writes while the benchmark is active.
        self.manager.update_worker_state.remote(self.worker_id, **kwargs)

    def _configure_agent_gpu(self, args):
        cuda_visible_devices = resolve_worker_cuda_visible_devices(
            getattr(args, "cuda_visible_devices", ""),
            self.worker_layout.gpu_hint,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        args.cuda_visible_devices = cuda_visible_devices
        args.gpu_id = 0
        self._update_state(agent_cuda_visible_devices=cuda_visible_devices)
        self.logger.info(f"[Agent {self.worker_id}] CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    def _wait_for_sim_socket(self, args, timeout_s=DEFAULT_EPISODE_TIMEOUT_S):
        deadline_s = time.time() + timeout_s
        last_log_s = 0.0
        last_error = ""
        while time.time() < deadline_s:
            try:
                if socket_request_responds(args.host, self.sim_port, "GET_EPISODE_LIST", timeout=1.0):
                    self._update_state(
                        sim_status="ready",
                        agent_status="running",
                        last_message=f"sim socket ready on port {self.sim_port}",
                    )
                    return
            except Exception as exc:
                last_error = str(exc)

            now_s = time.time()
            if now_s - last_log_s >= 10.0:
                last_log_s = now_s
                detail = f": {last_error}" if last_error else ""
                self.logger.info(f"[Agent {self.worker_id}] waiting for sim socket {self.sim_port}{detail}")
            time.sleep(0.5)

        raise TimeoutError(f"Timed out waiting for simulator socket on port {self.sim_port}")

    def run(self):
        try:
            runtime_context = ray.get(self.agent_runtime_manager.get_context.remote())
            args = copy.deepcopy(runtime_context["args"])
            args.process_label = self.worker_id
            args.port = self.sim_port
            args.log_file = self.worker_layout.agent_log_file
            self._configure_agent_gpu(args)
            sys.argv = ["eval_navverse_worker", *self.sys_argv]

            self._update_state(agent_status="waiting_for_sim", last_message=f"waiting for sim port {self.sim_port}")
            self._wait_for_sim_socket(args)
            self._update_state(agent_status="running", last_message=f"starting agent on port {self.sim_port}")

            worker = EvaluationWorker(
                args,
                self.task_queue,
                runtime_context["result_queue"],
                runtime_context["step_data_queue"],
                runtime_context["foxglove_log_queue"],
                progress_queue=runtime_context["progress_queue"],
                episode_controller=self.manager,
            )
            self.logger.step("evaluation_worker_created")
            worker.run()
            self._update_state(agent_status="finished", last_message="evaluation worker finished")
            self.logger.step("evaluation_worker_finished")
            return {"status": "finished"}
        except Exception as exc:
            self.logger.error(f"[Agent {self.worker_id}] crashed with error: {exc}", exc_info=True)
            self._update_state(agent_status="failed", last_message=str(exc))
            raise


@ray.remote(num_cpus=0)
class BenchmarkManager:
    """Top-level coordinator for one distributed benchmark run.

    `start()` loads the requested split, computes how many worker pairs fit on
    the visible GPUs, allocates one simulator port per pair, starts the shared
    result/progress writer, then launches each `SimWorker` and paired
    `AgentWorker`.

    During the run, agents reserve tasks from the shared queue and ask the
    manager to load episodes on their paired simulator. The watchdog loop checks
    each worker pair every few seconds: finished agents are cleaned up or
    restarted, starting/loading simulators are queried through `GET_SIM_STATE`,
    running simulators are checked through both the Ray actor and socket server,
    and unhealthy pairs are restarted with their current task requeued.

    The run is complete only when no worker pairs remain active and the task
    queue is empty. `shutdown()` stops the watchdog, terminates every active
    worker pair, and closes the shared background writer.
    """
    def __init__(self, manager_name, eval_args, project_args, unknown, episode_folder, first_gpu_total_vram_gb):
        self.manager_name = manager_name
        self.eval_args = eval_args
        self.project_args = project_args
        self.unknown = unknown
        self.episode_folder = episode_folder
        self.first_gpu_total_vram_gb = first_gpu_total_vram_gb
        self.host = project_args.host
        self.logger = RuntimeLogger("BenchmarkManager")
        self.state_lock = threading.Lock()
        self.worker_layouts: Dict[int, WorkerLayout] = {}
        self.worker_states: Dict[int, WorkerRuntimeState] = {}
        self.sim_workers = {}
        self.agent_workers = {}
        self.agent_run_refs = {}
        self.active_worker_ids = set()
        self.task_queue = RayTaskQueue(project_args)
        self.agent_runtime_manager = None
        self.watchdog_stop = False
        self.watchdog_thread = None
        self.failed_message = None
        self.all_done = False
        self.agent_gpu_fraction = None
        self.sim_gpu_fraction = None
        self.worker_slots_per_gpu = 1
        self.start_time_s = time.time()
        self.last_progress_publish_s = 0.0
        self.logger.step("init")

    def __repr__(self):
        return "benchmark-manager"

    def _set_failed(self, message):
        with self.state_lock:
            if self.failed_message is None:
                self.failed_message = message

    def _set_worker_state_locked(self, worker_id, **kwargs):
        state = self.worker_states[worker_id]
        now_s = time.time()
        for key, value in kwargs.items():
            if key == "sim_status" and value != getattr(state, "sim_status", None):
                state.sim_status_changed_at_s = now_s
            setattr(state, key, value)
        if not state.sim_status_changed_at_s:
            state.sim_status_changed_at_s = now_s
        state.updated_at_s = now_s
        self.worker_states[worker_id] = state
        return asdict(state)

    def update_worker_state(self, worker_id, **kwargs):
        with self.state_lock:
            return self._set_worker_state_locked(worker_id, **kwargs)

    def record_sim_response(self, worker_id):
        with self.state_lock:
            state = self.worker_states[worker_id]
            state.last_sim_response_s = time.time()
            self.worker_states[worker_id] = state
            return asdict(state)

    def record_agent_progress(
        self,
        worker_id,
        agent_step=0,
        episode_label=None,
        scene_id=None,
        last_action="",
        last_message="agent progress",
    ):
        with self.state_lock:
            state = self.worker_states[worker_id]
            state.agent_step = int(agent_step or 0)
            if episode_label:
                state.assigned_episode = episode_label
            if scene_id:
                state.scene_id = scene_id
            if last_action:
                state.agent_last_action = last_action
            state.last_message = last_message or state.last_message
            state.updated_at_s = time.time()
            self.worker_states[worker_id] = state
            return asdict(state)

    def get_all_worker_states(self):
        with self.state_lock:
            return {worker_id: asdict(state) for worker_id, state in self.worker_states.items()}

    def get_run_status(self):
        with self.state_lock:
            return {
                "failed_message": self.failed_message,
                "all_done": self.all_done,
                "worker_states": {worker_id: asdict(state) for worker_id, state in self.worker_states.items()},
            }

    def _progress_snapshot(self):
        now_s = time.time()
        with self.state_lock:
            return {
                "wall_time_s": now_s,
                "elapsed_s": now_s - self.start_time_s,
                "failed_message": self.failed_message,
                "all_done": self.all_done,
                "worker_states": {
                    worker_id: asdict(state)
                    for worker_id, state in self.worker_states.items()
                },
            }

    def _publish_progress_snapshot(self, force=False):
        if self.agent_runtime_manager is None:
            return
        now_s = time.time()
        interval_s = float(getattr(self.project_args, "wandb_progress_interval_s", WANDB_PROGRESS_INTERVAL_S))
        if not force and (now_s - self.last_progress_publish_s) < interval_s:
            return
        self.last_progress_publish_s = now_s
        self.agent_runtime_manager.log_progress.remote(self._progress_snapshot())

    def _configure_gpu_fractions(self):
        total_ray_gpus = float(ray.cluster_resources().get("GPU", 0))
        if total_ray_gpus <= 0:
            raise RuntimeError("No GPUs available for benchmark runtime.")
        first_gpu_total_vram_gb = self.first_gpu_total_vram_gb
        if first_gpu_total_vram_gb <= 0:
            raise RuntimeError("Failed to receive first GPU VRAM from the driver process.")

        self.agent_gpu_fraction = float(self.eval_args.agent_vram_gb) / first_gpu_total_vram_gb
        self.sim_gpu_fraction = float(self.eval_args.sim_vram_gb) / first_gpu_total_vram_gb
        pair_fraction = self.agent_gpu_fraction + self.sim_gpu_fraction
        if pair_fraction <= 0:
            raise RuntimeError("The combined sim/agent VRAM requirement must be positive.")
        if pair_fraction > 1.0:
            raise RuntimeError(
                f"Combined GPU fraction {pair_fraction:.3f} exceeds 1.0 based on first GPU VRAM {first_gpu_total_vram_gb:.1f} GB."
            )

        worker_slots_per_gpu = int(1.0 / pair_fraction)
        if worker_slots_per_gpu <= 0:
            raise RuntimeError(
                f"Combined GPU fraction {pair_fraction:.3f} leaves no room for one worker pair per GPU."
            )
        num_workers = max(1, int(total_ray_gpus) * worker_slots_per_gpu)
        self.worker_slots_per_gpu = worker_slots_per_gpu
        self.logger.info(
            f"Using first_gpu_total_vram_gb={first_gpu_total_vram_gb:.1f}, "
            f"agent_gpu_fraction={self.agent_gpu_fraction:.3f}, "
            f"sim_gpu_fraction={self.sim_gpu_fraction:.3f}, "
            f"worker_slots_per_gpu={worker_slots_per_gpu}, "
            f"num_workers={num_workers}"
        )
        return num_workers

    def allocate_workers(self):
        num_workers = self._configure_gpu_fractions()
        sim_ports = find_available_port_block(
            self.host,
            self.eval_args.port,
            block_size=num_workers,
        )
        log_dir = os.path.dirname(os.path.abspath(self.project_args.log_file))
        visible_gpu_ids = [
            item.strip()
            for item in str(getattr(self.eval_args, "cuda_visible_devices", "")).split(",")
            if item.strip()
        ]
        if not visible_gpu_ids:
            visible_gpu_ids = [str(gpu_id) for gpu_id in range(int(ray.cluster_resources().get("GPU", 0)))]
        worker_layouts = [
            WorkerLayout(
                worker_id=worker_id,
                sim_port=sim_port,
                agent_log_file=os.path.join(log_dir, f"agent_{worker_id}.log"),
                sim_log_file=os.path.join(log_dir, f"sim_{worker_id}.log"),
                gpu_hint=visible_gpu_ids[worker_id % len(visible_gpu_ids)] if visible_gpu_ids else "",
            )
            for worker_id, sim_port in enumerate(sim_ports)
        ]
        self.logger.info(
            "Allocated workers="
            f"{[(layout.worker_id, layout.sim_port, layout.gpu_hint) for layout in worker_layouts]}"
        )
        return worker_layouts

    def load_tasks(self):
        episode_set_list = load_episode_set(self.episode_folder)
        episode_list = VLNEpisode.from_json_folder(self.episode_folder)
        episode_set_list["all"] = [episode.episode_label for episode in episode_list]
        if len(episode_set_list["all"]) == 0:
            raise RuntimeError("Received episode list is invalid.")
        all_episode_list = sorted(list(set(label for labels in episode_set_list.values() for label in labels)))
        if "," in self.project_args.episode_labels:
            target_labels = self.project_args.episode_labels.split(",")
        elif self.project_args.episode_labels in episode_set_list:
            target_labels = episode_set_list[self.project_args.episode_labels]
        elif self.project_args.episode_labels in all_episode_list:
            target_labels = [self.project_args.episode_labels]
        else:
            raise ValueError(f"Episode label {self.project_args.episode_labels} not found.")

        if self.eval_args.resume_existing_results:
            result_path = os.path.join(
                self.project_args.dump_location,
                self.project_args.exp_name,
                "result.json",
            )
            completed_labels = []
            if os.path.isfile(result_path):
                with open(result_path, "r", encoding="utf-8") as handle:
                    existing_results = json.load(handle)
                if not isinstance(existing_results, list):
                    raise RuntimeError(f"Existing result file must contain a list: {result_path}")
                completed_labels = [
                    str(row.get("episode_label", ""))
                    for row in existing_results
                    if isinstance(row, dict) and row.get("episode_label")
                ]
                duplicate_labels = sorted(
                    label for label in set(completed_labels) if completed_labels.count(label) > 1
                )
                if duplicate_labels:
                    raise RuntimeError(
                        f"Existing result file has duplicate episode labels: {duplicate_labels[:10]}"
                    )
                unexpected_labels = sorted(set(completed_labels) - set(target_labels))
                if unexpected_labels:
                    raise RuntimeError(
                        f"Existing result file has labels outside the selected tasks: {unexpected_labels[:10]}"
                    )
            completed_set = set(completed_labels)
            target_labels = [label for label in target_labels if label not in completed_set]
            self.logger.info(
                f"Resume existing results: skipping {len(completed_set)} completed tasks; "
                f"{len(target_labels)} remaining"
            )

        self.logger.info(f"Loaded {len(target_labels)} tasks")
        self.task_queue.add_tasks(
            [
                {
                    "episode_label": episode_label,
                    "scene_id": "_".join(episode_label.split("_")[:-1]),
                    "episode_id": episode_label.split("_")[-1],
                }
                for episode_label in target_labels
            ]
        )
        self.logger.step("tasks_loaded")

    def _reserve_new_sim_port(self, worker_id):
        blocked_ports = set()
        with self.state_lock:
            for other_worker_id, layout in self.worker_layouts.items():
                if other_worker_id == worker_id:
                    continue
                blocked_ports.add(layout.sim_port)
        return find_available_port_block(self.host, self.eval_args.port, block_size=1, blocked_ports=blocked_ports)[0]

    def ensure_worker_episode(self, worker_id, task):
        sim_worker = self.sim_workers.get(worker_id)
        if sim_worker is None:
            raise RuntimeError(f"Sim worker {worker_id} is not available.")
        with self.state_lock:
            if worker_id in self.worker_states:
                self._set_worker_state_locked(
                    worker_id,
                    assigned_episode=task["episode_label"],
                    scene_id=task.get("scene_id"),
                    sim_status="loading",
                    last_message=f"dispatch episode {task['episode_label']}",
                )
        self.logger.step(f"worker={worker_id} ensure_episode episode={task['episode_label']}")
        return ray.get(sim_worker.ensure_episode.remote(task))

    def _start_worker_pair(self, worker_layout, initial_task=None, wait_for_sim_ready=False):
        if initial_task is None:
            initial_task = self.task_queue.reserve_task(worker_layout.worker_id)
        if initial_task is None:
            with self.state_lock:
                if worker_layout.worker_id not in self.worker_states:
                    self.worker_layouts[worker_layout.worker_id] = worker_layout
                    self.worker_states[worker_layout.worker_id] = WorkerRuntimeState(
                        worker_id=worker_layout.worker_id,
                        sim_port=worker_layout.sim_port,
                        sim_cuda_visible_devices=worker_layout.gpu_hint,
                        agent_cuda_visible_devices=worker_layout.gpu_hint,
                        updated_at_s=time.time(),
                        last_sim_response_s=time.time(),
                    )
                self._set_worker_state_locked(
                    worker_layout.worker_id,
                    sim_status="idle",
                    agent_status="idle",
                    last_message="no task reserved",
                )
            return False

        self.task_queue.assign_task(worker_layout.worker_id, initial_task)

        with self.state_lock:
            self.worker_layouts[worker_layout.worker_id] = worker_layout
            self.worker_states[worker_layout.worker_id] = WorkerRuntimeState(
                worker_id=worker_layout.worker_id,
                sim_port=worker_layout.sim_port,
                sim_cuda_visible_devices=worker_layout.gpu_hint,
                agent_cuda_visible_devices=worker_layout.gpu_hint,
                updated_at_s=time.time(),
                last_sim_response_s=time.time(),
            )
            self._set_worker_state_locked(
                worker_layout.worker_id,
                assigned_episode=initial_task["episode_label"],
                scene_id=initial_task.get("scene_id"),
                sim_status="starting",
                agent_status="waiting_for_sim",
                last_message=f"launching sim on port {worker_layout.sim_port}",
            )

        sim_worker = SimWorker.options(num_gpus=self.sim_gpu_fraction).remote(
            worker_layout, self.eval_args, self.project_args, self.manager_name
        )
        self.sim_workers[worker_layout.worker_id] = sim_worker
        sim_ready_ref = sim_worker.ensure_episode.remote(initial_task)
        if wait_for_sim_ready:
            ray.get(sim_ready_ref)

        agent_worker = AgentWorker.options(num_gpus=self.agent_gpu_fraction).remote(
            worker_layout,
            self.agent_runtime_manager,
            self.task_queue,
            self.manager_name,
            self.unknown,
        )
        agent_run_ref = agent_worker.run.remote()
        self.agent_workers[worker_layout.worker_id] = agent_worker
        self.agent_run_refs[worker_layout.worker_id] = agent_run_ref
        self.active_worker_ids.add(worker_layout.worker_id)
        self.logger.step(
            f"worker_pair_started worker={worker_layout.worker_id} port={worker_layout.sim_port}"
        )
        return True

    def _cleanup_worker_pair(self, worker_id):
        agent_run_ref = self.agent_run_refs.pop(worker_id, None)
        if agent_run_ref is not None:
            try:
                ray.cancel(agent_run_ref, force=True)
            except Exception:
                pass

        agent_worker = self.agent_workers.pop(worker_id, None)
        if agent_worker is not None:
            best_effort_kill(self.logger, agent_worker, f"agent_worker[{worker_id}]")

        sim_worker = self.sim_workers.pop(worker_id, None)
        sim_pid = None
        sim_port = None
        with self.state_lock:
            fallback_state = self.worker_states.get(worker_id, WorkerRuntimeState(worker_id, 0))
            sim_pid = fallback_state.sim_pid
            sim_port = fallback_state.sim_port
        if sim_worker is not None:
            cleaned = best_effort_remote_call(self.logger, sim_worker, "cleanup", timeout_s=20.0)
            if not cleaned:
                if sim_pid is not None:
                    terminate_process_group(sim_pid, timeout_s=15.0)
            # `cleanup()` only stops the child simulator process; the Ray actor
            # itself still holds its reserved GPU resources until it is killed.
            best_effort_kill(self.logger, sim_worker, f"sim_worker[{worker_id}]")

        if sim_pid is not None and sim_port is not None and can_connect(self.host, sim_port, timeout=1.0):
            terminate_process_group(sim_pid, timeout_s=15.0)
        if sim_port is not None:
            terminated = terminate_processes_matching_cmdline(
                ("navverse_evaluation.py", f"--port {sim_port}"),
                timeout_s=15.0,
            )
            released = wait_for_tcp_port_release(self.host, sim_port, timeout_s=20.0)
            if not terminated or not released:
                self.logger.warning(
                    f"[Worker {worker_id}] simulator processes or port {sim_port} remain after cleanup"
                )
        self.active_worker_ids.discard(worker_id)

    def _restart_worker_pair(self, worker_id, reason):
        task_queue_empty = self.task_queue.empty()
        current_task = None
        sim_worker = self.sim_workers.get(worker_id)
        if sim_worker is not None:
            try:
                current_task = ray.get(sim_worker.get_process_status.remote(), timeout=5.0).get("current_task")
            except Exception:
                current_task = None
        with self.state_lock:
            state = self.worker_states[worker_id]
            if current_task is None and state.assigned_episode is not None:
                current_task = {
                    "episode_label": state.assigned_episode,
                    "scene_id": state.scene_id or "_".join(state.assigned_episode.split("_")[:-1]),
                    "episode_id": state.assigned_episode.split("_")[-1],
                }
            if task_queue_empty and state.agent_status == "finished":
                self.logger.info(f"[Worker {worker_id}] finished normally with no remaining tasks; skipping restart.")
                return False
            restart_count = state.restart_count + 1
        self.logger.warning(f"[Worker {worker_id}] restarting pair due to: {reason}")
        if current_task is not None:
            self.task_queue.release_task(worker_id, current_task.get("episode_label"))
            self.task_queue.requeue_task(current_task, front=True)
        old_layout = self.worker_layouts[worker_id]
        self._cleanup_worker_pair(worker_id)
        new_port = self._reserve_new_sim_port(worker_id)
        new_layout = WorkerLayout(
            worker_id=worker_id,
            sim_port=new_port,
            agent_log_file=old_layout.agent_log_file,
            sim_log_file=old_layout.sim_log_file,
            gpu_hint=old_layout.gpu_hint,
        )
        with self.state_lock:
            self.worker_layouts[worker_id] = new_layout
            self.worker_states[worker_id] = WorkerRuntimeState(
                worker_id=worker_id,
                sim_port=new_layout.sim_port,
                sim_cuda_visible_devices=new_layout.gpu_hint,
                agent_cuda_visible_devices=new_layout.gpu_hint,
                restart_count=restart_count,
                updated_at_s=time.time(),
                last_sim_response_s=time.time(),
            )
        self._start_worker_pair(new_layout)
        return True

    def _is_sim_timeout_exceeded(self, state, now_s):
        last_activity_s = max(state.last_sim_response_s, state.updated_at_s)
        return (now_s - last_activity_s) > DEFAULT_EPISODE_TIMEOUT_S

    def _is_sim_state_timeout_exceeded(self, state, now_s):
        status_since_s = state.sim_status_changed_at_s or state.updated_at_s
        return (now_s - status_since_s) > DEFAULT_EPISODE_TIMEOUT_S

    def _poll_sim_state_payload(self, sim_port):
        try:
            payload = request_data(self.host, sim_port, request_type="GET_SIM_STATE", verbose=False)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _sync_sim_state_from_payload(self, worker_id, payload):
        sim_state = payload.get("sim_state") or payload.get("message") or "unknown"
        if sim_state == "running" or payload.get("success", False):
            sim_status = "running"
        elif sim_state in {"init", "loading", "terminated"}:
            sim_status = sim_state
        else:
            sim_status = "loading"

        message = f"sim state={sim_state}"
        if payload.get("episode_label"):
            message = f"{message} episode={payload['episode_label']}"
        updates = {
            "sim_status": sim_status,
            "last_message": message,
        }
        telemetry_fields = {
            "sim_frame_count": int,
            "sim_fps": float,
            "sim_to_wall_ratio": float,
            "sim_total_time_s": float,
        }
        for field_name, caster in telemetry_fields.items():
            if field_name in payload and payload[field_name] is not None:
                updates[field_name] = caster(payload[field_name])
        with self.state_lock:
            self._set_worker_state_locked(worker_id, **updates)
        self.record_sim_response(worker_id)
        return sim_status

    def _worker_snapshot(self, worker_id):
        with self.state_lock:
            return (
                self.worker_states[worker_id],
                self.worker_layouts[worker_id],
                self.agent_run_refs.get(worker_id),
                self.sim_workers.get(worker_id),
            )

    def _mark_complete_when_idle(self, worker_ids):
        if worker_ids or not self.task_queue.empty():
            return False
        with self.state_lock:
            self.all_done = True
        return True

    def _process_completed_agent(self, worker_id, state, agent_run_ref):
        if agent_run_ref is None or not ray_ref_ready(agent_run_ref, timeout_s=0.0):
            return False
        try:
            ray.get(agent_run_ref)
        except Exception as exc:
            self._restart_worker_pair(worker_id, f"agent crashed: {exc}")
            return True

        if state.agent_status != "finished":
            return False

        task_queue_empty = self.task_queue.empty()
        self._cleanup_worker_pair(worker_id)
        if task_queue_empty:
            self.logger.info(f"[Worker {worker_id}] completed all work.")
            return True
        self._restart_worker_pair(worker_id, "agent finished early while tasks remain")
        return True

    def _monitor_starting_sim(self, worker_id, state, layout, sim_worker, now_s):
        if state.sim_status not in {"starting", "init", "loading"}:
            return False
        if can_connect(self.host, layout.sim_port, timeout=1.0):
            payload = self._poll_sim_state_payload(layout.sim_port)
            if payload is not None:
                sim_status = self._sync_sim_state_from_payload(worker_id, payload)
                with self.state_lock:
                    state = self.worker_states[worker_id]
                if sim_status in READY_SIM_STATUSES or not self._is_sim_state_timeout_exceeded(state, now_s):
                    return True
        else:
            sim_status = self._read_sim_actor_status(worker_id, sim_worker, now_s)
            if sim_status is not None and not sim_status["process_alive"]:
                self._restart_worker_pair(worker_id, "sim process exited during startup")
                return True
        if self._is_sim_state_timeout_exceeded(state, now_s):
            self._restart_worker_pair(
                worker_id,
                f"sim stayed {state.sim_status} for over {DEFAULT_EPISODE_TIMEOUT_S:.0f} seconds",
            )
        return True

    def _read_sim_actor_status(self, worker_id, sim_worker, now_s):
        try:
            return ray.get(sim_worker.get_process_status.remote(), timeout=5.0)
        except Exception as exc:
            with self.state_lock:
                latest_state = self.worker_states.get(worker_id)
            if latest_state is not None and not self._is_sim_timeout_exceeded(latest_state, now_s):
                return None
            self._restart_worker_pair(worker_id, f"sim worker actor unavailable: {exc}")
            return None

    def _monitor_running_sim_socket(self, worker_id, state, layout, now_s):
        if can_connect(self.host, layout.sim_port, timeout=1.0):
            payload = self._poll_sim_state_payload(layout.sim_port)
            if payload is not None:
                sim_status = self._sync_sim_state_from_payload(worker_id, payload)
                with self.state_lock:
                    latest_state = self.worker_states[worker_id]
                if sim_status in {"init", "loading"} and self._is_sim_state_timeout_exceeded(latest_state, now_s):
                    self._restart_worker_pair(
                        worker_id,
                        f"sim stayed {sim_status} for over {DEFAULT_EPISODE_TIMEOUT_S:.0f} seconds",
                    )
                return
            if self._is_sim_timeout_exceeded(state, now_s):
                self._restart_worker_pair(
                    worker_id,
                    f"sim socket unresponsive for over {DEFAULT_EPISODE_TIMEOUT_S:.0f} seconds",
                )
            return
        if (now_s - state.last_sim_response_s) <= DEFAULT_EPISODE_TIMEOUT_S:
            return
        self._restart_worker_pair(
            worker_id,
            f"sim socket was not connectable for over {DEFAULT_EPISODE_TIMEOUT_S:.0f} seconds",
        )

    def _check_worker_pair_health(self, worker_id, now_s):
        state, layout, agent_run_ref, sim_worker = self._worker_snapshot(worker_id)

        if self._process_completed_agent(worker_id, state, agent_run_ref):
            return
        if sim_worker is None:
            self._restart_worker_pair(worker_id, "sim worker handle missing")
            return
        if self._monitor_starting_sim(worker_id, state, layout, sim_worker, now_s):
            return

        sim_status = self._read_sim_actor_status(worker_id, sim_worker, now_s)
        if sim_status is None:
            return
        if not sim_status["process_alive"]:
            self._restart_worker_pair(worker_id, "sim process exited")
            return
        self._monitor_running_sim_socket(worker_id, state, layout, now_s)

    def _check_all_worker_pairs_health(self):
        with self.state_lock:
            worker_ids = list(self.active_worker_ids)
        if self._mark_complete_when_idle(worker_ids):
            return

        now_s = time.time()
        for worker_id in worker_ids:
            self._check_worker_pair_health(worker_id, now_s)

    def _watchdog_loop(self):
        while not self.watchdog_stop:
            try:
                self._check_all_worker_pairs_health()
                self._publish_progress_snapshot()
            except Exception as exc:
                self.logger.error(f"[Watchdog] error: {exc}", exc_info=True)
                self._set_failed(str(exc))
            time.sleep(5.0)

    def start(self):
        worker_layouts = self.allocate_workers()
        self.load_tasks()

        self.agent_runtime_manager = AgentManager.remote(self.project_args)
        ray.get(self.agent_runtime_manager.start_background_processes.remote())
        self.logger.step("agent_runtime_manager_started")

        started_count = 0
        for worker_layout in worker_layouts:
            try:
                started = self._start_worker_pair(worker_layout, wait_for_sim_ready=True)
            except Exception as exc:
                self.logger.warning(
                    f"[Worker {worker_layout.worker_id}] initial simulator startup failed: {exc}; restarting pair"
                )
                started = self._restart_worker_pair(worker_layout.worker_id, f"initial startup failed: {exc}")
            if started:
                started_count += 1
                startup_stagger_s = float(self.eval_args.sim_startup_stagger_s)
                if startup_stagger_s > 0 and started_count < len(worker_layouts):
                    self.logger.info(
                        f"Waiting {startup_stagger_s:.1f}s before starting the next simulator"
                    )
                    time.sleep(startup_stagger_s)

        self.watchdog_thread = threading.Thread(name="Watchdog", target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        self.logger.step(f"benchmark_started workers={started_count}")
        self._publish_progress_snapshot(force=True)

        if started_count == 0 and self.task_queue.empty():
            with self.state_lock:
                self.all_done = True

        return {
            "worker_count": started_count,
        }

    def wait(self):
        while True:
            with self.state_lock:
                if self.failed_message is not None:
                    raise RuntimeError(self.failed_message)
                if self.all_done:
                    return self.get_all_worker_states()
            time.sleep(1.0)

    def shutdown(self):
        self.watchdog_stop = True
        self._publish_progress_snapshot(force=True)
        if self.watchdog_thread is not None:
            self.watchdog_thread.join(timeout=1.0)

        for worker_id in list(self.active_worker_ids):
            self._cleanup_worker_pair(worker_id)

        if self.agent_runtime_manager is not None:
            cleaned = best_effort_remote_call(self.logger, self.agent_runtime_manager, "shutdown", timeout_s=5.0)
            if not cleaned:
                best_effort_kill(self.logger, self.agent_runtime_manager, "agent_runtime_manager")

        self.logger.step("shutdown_done")


def add_runtime_args(parser):
    runtime_inputs = base_config("benchmark")["inputs"]
    parser.add_argument("--port", type=int, default=12300, help="Base port for communication")
    parser.add_argument("--agent_vram_gb", type=float, default=20.0, help="Per-agent VRAM budget in GB")
    parser.add_argument("--sim_vram_gb", type=float, default=20.0, help="Per-simulator VRAM budget in GB")
    parser.add_argument(
        "--sim_startup_stagger_s",
        type=float,
        default=0.0,
        help="Wait after each initial simulator becomes ready before starting the next one.",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--sim_livestream",
        type=int,
        default=0,
        help="Simulator livestream mode. Defaults to 0 to avoid fixed livestream port conflicts in parallel headless runs.",
    )
    parser.add_argument("--dashboard_port", type=int, default=4913, help="Port for dashboard")
    parser.add_argument("--conda_env", type=str, default="navverse", help="Conda environment to use for simulators")
    parser.add_argument(
        "--episode_folder",
        type=str,
        default=runtime_inputs["episode_folder"],
        help="Episode folder installed from the NavVerse dataset",
    )
    parser.add_argument(
        "--scene_folder",
        type=str,
        default=runtime_inputs["scene_folder"],
        help="Scene asset root used by benchmark workers",
    )
    parser.add_argument(
        "--sim_robot_name",
        type=str,
        default="",
        help="Robot name passed to scripts/navverse_evaluation.py for simulator workers",
    )
    parser.add_argument(
        "--sim_tidybot_embodiment",
        choices=["full", "habitat_pe"],
        default="",
        help="TidyBot embodiment profile passed to simulator workers",
    )
    parser.add_argument(
        "--sim_timed_trajectory_dt",
        type=float,
        default=None,
        help="Control period passed to TIMED_TRAJECTORY simulator workers",
    )
    parser.add_argument(
        "--controlled_terrain_manifest",
        type=str,
        default="",
        help="Shared controlled terrain manifest passed to simulator workers",
    )
    parser.add_argument(
        "--controlled_terrain_type",
        choices=["flat", "slope", "stairs", "uneven"],
        default="flat",
        help="Controlled terrain condition passed to simulator workers",
    )
    parser.add_argument(
        "--disable_sim_camera",
        action="store_true",
        help="Disable simulator cameras for trajectory-only terrain evaluation",
    )
    parser.add_argument("--ray_num_cpus", type=int, default=16, help="CPU resources to expose to Ray")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="Comma-separated physical GPU ids to expose to Ray, e.g. 2,3. Overrides CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--resume_existing_results",
        action="store_true",
        help="Skip episode labels already present in the run result.json.",
    )


def add_wandb_args(parser):
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging. Enabled by default; use --no-wandb to disable.",
    )
    parser.add_argument("--wandb_project", type=str, default="navverse-benchmark", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="", help="wandb run name, defaults to exp_name")
    parser.add_argument(
        "--wandb_upload_videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload per-episode videos to wandb when wandb logging is enabled",
    )
    parser.add_argument("--disable_runtime_log", action="store_true", help="Disable runtime step logging")


def build_eval_parser():
    parser = argparse.ArgumentParser(description="Distributed Training with Ray", allow_abbrev=False)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset", help="benchmark preset name under baselines/presets")
    source.add_argument("--config", help="path to a benchmark preset YAML file")
    parser.add_argument("--list-presets", action="store_true")
    add_runtime_args(parser)
    add_wandb_args(parser)
    return parser


def resolve_requested_episode_split(eval_args, project_args):
    selector = str(project_args.episode_labels or "")
    if not selector or selector == "all" or "," in selector:
        return
    episode_folder = Path(eval_args.episode_folder).expanduser()
    if not episode_folder.is_absolute():
        episode_folder = Path(__file__).resolve().parents[1] / episode_folder
    if not episode_folder.is_dir():
        return
    try:
        split = get_split(episode_folder, selector)
    except ValueError:
        return
    if not split.valid:
        raise ValueError(f"Invalid split {split.name!r}: {split.errors + split.missing_labels[:5]}")
    runtime_folder = resolve_split_episode_folder(episode_folder, split)
    if runtime_folder != episode_folder.resolve():
        eval_args.episode_folder = str(runtime_folder)
        project_args.episode_folder = str(runtime_folder)
        project_args.episode_labels = ",".join(split.labels)
        print(f"Resolved split {split.name!r} to isolated episode folder {runtime_folder}")


def apply_eval_args_to_project_args(project_args, eval_args, visible_devices):
    project_args.port = eval_args.port
    project_args.cuda_visible_devices = visible_devices
    project_args.wandb = eval_args.wandb
    project_args.wandb_project = eval_args.wandb_project
    project_args.wandb_run_name = eval_args.wandb_run_name or project_args.exp_name
    project_args.wandb_upload_videos = eval_args.wandb_upload_videos
    project_args.wandb_progress_interval_s = WANDB_PROGRESS_INTERVAL_S


def init_runtime_logging(eval_args, project_args, visible_devices):
    runtime_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runtime.log"))
    RuntimeLogger.configure_runtime(enabled=not eval_args.disable_runtime_log, path=runtime_log_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    setup_tee(project_args.log_file)
    ensure_repo_log_symlink(repo_root, project_args.log_file)
    logger = RuntimeLogger("main")
    logger.step("args_parsed")
    if visible_devices:
        logger.info(f"[Main] CUDA_VISIBLE_DEVICES={visible_devices}")
    return logger, repo_root


def init_ray(eval_args, logger):
    logger.step("ray_init_begin")
    ray_tmp_folder = os.environ.get("NAVVERSE_RAY_TMP_DIR", os.path.join("/dev/shm", "ray_tmp"))
    os.makedirs(ray_tmp_folder, exist_ok=True)
    ray.init(
        include_dashboard=False,
        dashboard_port=eval_args.dashboard_port,
        _temp_dir=ray_tmp_folder,
        _node_ip_address="127.0.0.1",
        num_cpus=eval_args.ray_num_cpus,
    )
    logger.step("ray_init_done")


def run_manager(eval_args, project_args, unknown, repo_root, manager_name):
    first_gpu_total_vram_gb = detect_first_gpu_total_vram_gb()
    episode_folder = eval_args.episode_folder
    if not os.path.isabs(episode_folder):
        episode_folder = os.path.join(repo_root, episode_folder)
    eval_args.episode_folder = episode_folder
    project_args.episode_folder = episode_folder
    if eval_args.controlled_terrain_manifest and not os.path.isabs(eval_args.controlled_terrain_manifest):
        eval_args.controlled_terrain_manifest = os.path.join(repo_root, eval_args.controlled_terrain_manifest)
    manager = BenchmarkManager.options(name=manager_name).remote(
        manager_name,
        eval_args,
        project_args,
        unknown,
        episode_folder,
        first_gpu_total_vram_gb,
    )
    ray.get(manager.start.remote())
    return manager


def wait_for_manager(manager, logger):
    logger.step("manager_started")
    while True:
        run_status = ray.get(manager.get_run_status.remote())
        if run_status["failed_message"] is not None:
            raise RuntimeError(run_status["failed_message"])
        if run_status["all_done"]:
            break
        time.sleep(1.0)
    logger.step("all_workers_finished")


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--list-presets" in raw_argv:
        for preset in list_presets():
            print(f"{preset['name']}: {preset.get('description', '')}")
        return
    resolved_argv, _ = resolve_preset_argv(raw_argv)
    parser = build_eval_parser()
    eval_args, unknown = parser.parse_known_args(resolved_argv)
    _, unknown = get_args(unknown)
    visible_devices = apply_requested_cuda_visible_devices(eval_args)

    project_args, _ = get_args(resolved_argv)
    agent_spec = AGENT_REGISTRY.get(str(project_args.agent))
    if agent_spec and agent_spec.validation_warning:
        print(f"WARNING: {agent_spec.validation_warning}", file=sys.stderr, flush=True)
    resolve_requested_episode_split(eval_args, project_args)
    apply_eval_args_to_project_args(project_args, eval_args, visible_devices)
    logger, repo_root = init_runtime_logging(eval_args, project_args, visible_devices)

    manager = None
    manager_name = f"benchmark-manager-{project_args.exp_name}-{os.getpid()}"
    try:
        init_ray(eval_args, logger)
        manager = run_manager(eval_args, project_args, unknown, repo_root, manager_name)
        wait_for_manager(manager, logger)
    except KeyboardInterrupt:
        logger.info("[Main] KeyboardInterrupt received.")
        logger.step("keyboard_interrupt")
    except Exception as exc:
        logger.error(f"Error during execution: {exc}", exc_info=True)
        logger.step(f"exception {type(exc).__name__}")
    finally:
        logger.info("[Main] Cleaning up...")
        logger.step("cleanup_begin")
        if ray.is_initialized():
            if manager is not None:
                cleaned = best_effort_remote_call(logger, manager, "shutdown", timeout_s=10.0)
                if not cleaned:
                    best_effort_kill(logger, manager, "benchmark_manager")
            ray.shutdown()
        logger.step("cleanup_done")


if __name__ == "__main__":
    main()
