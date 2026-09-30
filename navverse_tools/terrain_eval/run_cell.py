#!/usr/bin/env python3
"""Run one controlled terrain and embodiment evaluation cell."""
import argparse
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import dotenv_values


SIMULATOR_FATAL_MARKERS = (
    "[ERROR] Benchmark initialization or loop failed",
    "Warp CUDA error 700",
    "CUDA error: an illegal memory access was encountered",
    "Fatal Python error: Aborted",
)


class LocalTaskQueue:
    def __init__(self, episode_labels):
        self.tasks = [
            {
                "episode_label": label,
                "scene_id": "_".join(label.split("_")[:-1]),
                "episode_id": label.split("_")[-1],
            }
            for label in episode_labels
        ]
        self.completed = []

    def get_task(self, _worker_id, _scene_id=None):
        return self.tasks.pop(0) if self.tasks else None

    def empty(self):
        return not self.tasks

    def complete_task(self, _worker_id, episode_label):
        self.completed.append(episode_label)


class ResultSink:
    def __init__(self, result_path, existing=None):
        self.result_path = result_path
        self.rows = list(existing or [])

    def put(self, row):
        self.rows.append(row)
        pending = self.result_path.with_suffix(".tmp")
        pending.write_text(json.dumps(self.rows, indent=2) + "\n")
        pending.replace(self.result_path)


def runtime_path(repo_root, name, default):
    value = os.environ.get(name) or dotenv_values(repo_root / ".env").get(name, default)
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--terrain",
        choices=("original", "flat", "slope", "stairs", "uneven"),
        required=True,
    )
    parser.add_argument("--robot", choices=("drone", "spot", "turtlebot"), required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--port", type=int, default=13000)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--episode-folder",
        type=Path,
        default=runtime_path(repo_root, "EPISODE_FOLDER", "navverse_data/episodes"),
    )
    parser.add_argument(
        "--scene-folder",
        type=Path,
        default=runtime_path(repo_root, "SCENE_FOLDER", "navverse_data"),
    )
    parser.add_argument("--cell-name", default="")
    parser.add_argument("--episode-labels", default="")
    parser.add_argument("--material-friction", type=float, default=None)
    parser.add_argument("--material-static-friction", type=float, default=None)
    parser.add_argument("--material-dynamic-friction", type=float, default=None)
    parser.add_argument("--material-restitution", type=float, default=0.0)
    parser.add_argument("--material-compliant-stiffness", type=float, default=0.0)
    parser.add_argument("--material-compliant-damping", type=float, default=0.0)
    parser.add_argument(
        "--material-scope", choices=("default", "terrain"), default="default"
    )
    parser.add_argument(
        "--material-friction-combine-mode",
        choices=("average", "min", "multiply", "max"),
        default="multiply",
    )
    parser.add_argument("--turtlebot-joint-dynamic-friction", type=float, default=None)
    parser.add_argument("--turtlebot-joint-viscous-friction", type=float, default=None)
    parser.add_argument(
        "--terrain-profile",
        choices=(
            "standard",
            "hard",
            "moderate",
            "moderate_v2",
            "moderate_final",
            "moderate_compact",
            "moderate_wide",
            "moderate_traversable",
            "balanced",
            "extra_hard",
            "micro_rough",
            "micro_rough_fine",
            "micro_rough_medium",
        ),
        default="standard",
    )
    parser.add_argument("--episode-timeout-s", type=float, default=None)
    parser.add_argument(
        "--waypoint-follower-profile",
        choices=("", "task_generation"),
        default="",
    )
    parser.add_argument(
        "--reference-execution",
        choices=("external_oracle", "task_generation_internal"),
        default="external_oracle",
    )
    parser.add_argument("--diagnostic-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scene-retries", type=int, default=3)
    return parser.parse_args()


def resolve_cell_name(args):
    return args.cell_name or f"{args.terrain}_{args.robot}"


def wait_for_server(process, host, port, timeout_s=180.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Simulator exited before opening port {port}: returncode={process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Simulator did not open {host}:{port} within {timeout_s:.0f}s")


def terminate_process_group(process, timeout_s=30.0):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


def simulator_log_has_fatal_error(log_path, start_offset=0):
    try:
        with Path(log_path).open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(start_offset, size - 131072))
            tail = log_file.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in tail for marker in SIMULATOR_FATAL_MARKERS)


def run_worker_with_simulator(worker, simulator, simulator_log_path, simulator_log_offset=0):
    errors = []

    def worker_target():
        try:
            worker.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker_target, name="controlled-evaluation-worker")
    thread.start()
    while thread.is_alive():
        thread.join(timeout=1.0)
        if simulator.poll() is not None:
            worker.evaluation_over.set()
            thread.join(timeout=10.0)
            raise RuntimeError(
                f"Simulator exited during evaluation: returncode={simulator.returncode}"
            )
        if simulator_log_has_fatal_error(simulator_log_path, simulator_log_offset):
            worker.evaluation_over.set()
            terminate_process_group(simulator, timeout_s=5.0)
            thread.join(timeout=10.0)
            raise RuntimeError("Simulator reported a fatal runtime error")
    if errors:
        raise errors[0]


def build_worker_args(
    repo_root,
    run_root,
    cell,
    port,
    gpu_id,
    reference_execution,
    diagnostic_video,
):
    baselines_dir = repo_root / "baselines"
    sys.path.insert(0, str(baselines_dir))
    from arguments import get_args

    argv = [
        "controlled-direct",
        "--agent", "Oracle",
        "--backend", "isaac",
        "--host", "localhost",
        "--port", str(port),
        "--gpu_id", "0",
        "--task_type", "placenav",
        "--dump_location", str(run_root),
        "--exp_name", cell,
        "--log_file", str(run_root / cell / "agent.log"),
        "--oracle_action_type", (
            "task_generation_internal"
            if reference_execution == "task_generation_internal"
            else "waypoint"
        ),
        "--no-oracle_stale_waypoint_skip",
        "--save_video" if diagnostic_video else "--no-save_video",
        "--no-obs_action_bev",
        "--no-modern_obs_video",
    ]
    if diagnostic_video:
        argv.append("--oracle_capture_internal_frames")
    old_argv = sys.argv
    try:
        sys.argv = argv
        worker_args, unknown = get_args()
    finally:
        sys.argv = old_argv
    if unknown:
        raise RuntimeError(f"Unparsed direct worker arguments: {unknown}")
    worker_args.process_label = 0
    worker_args.cuda_visible_devices = str(gpu_id)
    worker_args.foxglove = False
    worker_args.disable_action = False
    return worker_args


def run_scene(repo_root, args, scene_labels, result_sink, scene_index):
    cell = resolve_cell_name(args)
    cell_dir = Path(args.run_root).resolve() / cell
    sim_log_path = cell_dir / f"sim_scene_{scene_index:02d}.log"
    port = args.port + scene_index
    simulator_command = [
        sys.executable,
        "scripts/navverse_evaluation.py",
        "--episode_label", scene_labels[0],
        "--port", str(port),
        "--episode_folder", str(args.episode_folder),
        "--scene_folder", str(args.scene_folder),
        "--robot_name", args.robot,
        "--controlled_terrain_manifest", str(Path(args.manifest).resolve()),
        "--controlled_terrain_type", args.terrain,
        "--controlled_terrain_profile", args.terrain_profile,
        "--controlled_waypoint_follower_profile", (
            "task_generation"
            if args.reference_execution == "task_generation_internal"
            else args.waypoint_follower_profile
        ),
        "--controlled_reference_execution", (
            args.reference_execution
            if args.reference_execution == "task_generation_internal"
            else ""
        ),
        "--disable_camera", "false" if args.diagnostic_video else "true",
        "--task_type", "placenav",
        "--headless",
        "--experience", "isaaclab.python.rendering.kit",
    ]
    if args.material_friction is not None:
        simulator_command.extend(
            ["--controlled_material_friction", str(args.material_friction)]
        )
    if args.material_static_friction is not None:
        simulator_command.extend(
            ["--controlled_material_static_friction", str(args.material_static_friction)]
        )
    if args.material_dynamic_friction is not None:
        simulator_command.extend(
            ["--controlled_material_dynamic_friction", str(args.material_dynamic_friction)]
        )
    if any(
        value is not None
        for value in (
            args.material_friction,
            args.material_static_friction,
            args.material_dynamic_friction,
        )
    ):
        simulator_command.extend(
            [
                "--controlled_material_scope", args.material_scope,
                "--controlled_material_restitution", str(args.material_restitution),
                "--controlled_material_compliant_stiffness", str(args.material_compliant_stiffness),
                "--controlled_material_compliant_damping", str(args.material_compliant_damping),
                "--controlled_material_friction_combine_mode", args.material_friction_combine_mode,
            ]
        )
    if args.episode_timeout_s is not None:
        simulator_command.extend(
            ["--controlled_episode_timeout_s", str(args.episode_timeout_s)]
        )
    if args.turtlebot_joint_dynamic_friction is not None:
        simulator_command.extend(
            [
                "--controlled_turtlebot_joint_dynamic_friction",
                str(args.turtlebot_joint_dynamic_friction),
            ]
        )
    if args.turtlebot_joint_viscous_friction is not None:
        simulator_command.extend(
            [
                "--controlled_turtlebot_joint_viscous_friction",
                str(args.turtlebot_joint_viscous_friction),
            ]
        )
    process_env = os.environ.copy()
    process_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    process_env["PYTHONUNBUFFERED"] = "1"
    with sim_log_path.open("a", buffering=1) as sim_log:
        sim_log.write(f"COMMAND {' '.join(simulator_command)}\n")
        sim_log.flush()
        simulator_log_offset = sim_log.tell()
        simulator = subprocess.Popen(
            simulator_command,
            cwd=repo_root,
            env=process_env,
            stdout=sim_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(simulator, "localhost", port)
            worker_args = build_worker_args(
                repo_root,
                Path(args.run_root).resolve(),
                cell,
                port,
                args.gpu_id,
                args.reference_execution,
                args.diagnostic_video,
            )
            from evaluation_worker import EvaluationWorker

            task_queue = LocalTaskQueue(scene_labels)
            worker = EvaluationWorker(
                worker_args,
                task_queue,
                result_sink,
                queue.Queue(),
                queue.Queue(),
                episode_controller=None,
            )
            run_worker_with_simulator(
                worker,
                simulator,
                sim_log_path,
                simulator_log_offset,
            )
            if task_queue.completed != scene_labels:
                raise RuntimeError(
                    f"Scene completion mismatch: completed={task_queue.completed}, expected={scene_labels}"
                )
        finally:
            terminate_process_group(simulator)


def run_scene_with_retries(repo_root, args, scene_labels, result_sink, scene_index):
    remaining = list(scene_labels)
    for attempt in range(args.scene_retries + 1):
        try:
            run_scene(repo_root, args, remaining, result_sink, scene_index)
            return
        except Exception as exc:
            completed = {row["episode_label"] for row in result_sink.rows}
            remaining = [label for label in scene_labels if label not in completed]
            if not remaining:
                return
            if attempt >= args.scene_retries:
                raise
            print(
                f"Scene {scene_index} simulator attempt {attempt + 1} failed with "
                f"{type(exc).__name__}: {exc}; restarting for {len(remaining)} remaining episodes",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(5.0)


def main():
    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    args = parse_args()
    if args.scene_retries < 0:
        raise ValueError("--scene-retries must be non-negative")
    if args.material_friction is not None and args.material_friction < 0.0:
        raise ValueError("--material-friction must be non-negative")
    material_values = (
        args.material_static_friction,
        args.material_dynamic_friction,
        args.material_restitution,
        args.material_compliant_stiffness,
        args.material_compliant_damping,
    )
    if any(value is not None and value < 0.0 for value in material_values):
        raise ValueError("Controlled material values must be non-negative")
    if (
        args.material_compliant_damping > 0.0
        and args.material_compliant_stiffness <= 0.0
    ):
        raise ValueError("Compliant damping requires positive compliant stiffness")
    joint_friction_values = (
        args.turtlebot_joint_dynamic_friction,
        args.turtlebot_joint_viscous_friction,
    )
    if any(value is not None and value < 0.0 for value in joint_friction_values):
        raise ValueError("Turtlebot joint friction values must be non-negative")
    if args.episode_timeout_s is not None and args.episode_timeout_s <= 0.0:
        raise ValueError("--episode-timeout-s must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(args.manifest).resolve().read_text())
    manifest_labels = [row["episode_label"] for row in manifest["episodes"]]
    requested = [label for label in args.episode_labels.split(",") if label]
    target_labels = requested or manifest_labels
    if len(set(target_labels)) != len(target_labels) or not set(target_labels).issubset(manifest_labels):
        raise ValueError("Requested episode labels must be a unique subset of the controlled manifest")

    cell = resolve_cell_name(args)
    cell_dir = Path(args.run_root).resolve() / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    result_path = cell_dir / "result.json"
    existing = json.loads(result_path.read_text()) if args.resume and result_path.is_file() else []
    existing_labels = {row["episode_label"] for row in existing}
    remaining = [label for label in target_labels if label not in existing_labels]
    if not remaining:
        return

    scene_groups = []
    for label in remaining:
        scene_id = "_".join(label.split("_")[:-1])
        if not scene_groups or scene_groups[-1][0] != scene_id:
            scene_groups.append((scene_id, []))
        scene_groups[-1][1].append(label)

    result_sink = ResultSink(result_path, existing)
    for scene_index, (_scene_id, scene_labels) in enumerate(scene_groups):
        run_scene_with_retries(
            repo_root, args, scene_labels, result_sink, scene_index
        )

    actual = {row["episode_label"] for row in result_sink.rows}
    if not set(target_labels).issubset(actual):
        raise RuntimeError(f"Missing results: {sorted(set(target_labels) - actual)}")


if __name__ == "__main__":
    main()
