#!/usr/bin/env python3
"""Run task generation across multiple per-scene workers."""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from navverse.config import list_scene_ids, load_config, resolve_scene_spec


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _target_episode_count(config: dict, config_path: Path, scene_id: str, task_type: str) -> tuple[Path, int, int]:
    scene_spec = resolve_scene_spec(config, scene_id)
    flow = scene_spec.get("task_generation", {}).get("flow", {})
    output_dir = Path(flow.get("episode_output_dir", "episodes"))
    target_count = int(flow.get("episode_number", 50))
    episode_mix = flow.get("episode_mix") or {}
    if task_type in episode_mix:
        target_count = int(episode_mix[task_type])
    episode_path = output_dir / f"{scene_id}.json"
    if not episode_path.is_absolute():
        episode_path = config_path.parent.parent / episode_path
    count = 0
    if episode_path.exists():
        try:
            episodes = json.loads(episode_path.read_text())
            count = sum(1 for episode in episodes if episode.get("task_type", task_type) == task_type)
        except Exception:
            count = 0
    return episode_path, count, target_count


def _killed_after_success(log_path: Path) -> tuple[bool, str | None]:
    """Detect navverse_task_generation's SIGKILL-on-clean-exit convention."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return False, None

    success_markers = (
        ("Episode generation target reached", "target_reached"),
        ("[TG] Episode data replay completed successfully", "episode_data_replay_completed"),
        ("Task already generated", "already_generated"),
        ("Task skipped", "skipped"),
    )
    for marker, reason in success_markers:
        if marker in text:
            return True, reason
    return False, None


def _episode_data_replay_succeeded(log_path: Path) -> bool:
    try:
        return "[TG] Episode data replay completed successfully" in log_path.read_text(errors="replace")
    except OSError:
        return False


def _run_worker(worker_id: int, args: argparse.Namespace, jobs: queue.Queue[str]) -> None:
    gpu = args.gpu_offset + worker_id // args.workers_per_gpu
    port = args.base_port + worker_id
    while True:
        try:
            scene_id = jobs.get_nowait()
        except queue.Empty:
            print(f"[worker {worker_id}] queue empty", flush=True)
            return

        start = time.time()
        active_path = args.run_root / "active" / f"{scene_id}.json"
        log_path = args.run_root / "logs" / f"{scene_id}.worker{worker_id}.log"
        payload = {
            "gpu": gpu,
            "port": port,
            "scene_id": scene_id,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
            "worker": worker_id,
        }
        _write_json(active_path, payload)

        cmd = [
            args.python,
            "scripts/navverse_task_generation.py",
            "--enable_cameras",
            "--headless",
            "--scene_folder",
            args.scene_folder,
            "--config",
            str(args.config_path),
            "--tg_config_path",
            str(args.config_path),
            "--test_scene_id",
            scene_id,
            "--tg_mode",
            "auto",
            "--task_type",
            args.task_type,
            "--port",
            str(port),
            "--trajectory_planner",
            args.trajectory_planner,
        ]
        if args.replay_episode_data:
            cmd.extend(["--replay_episode_data", "true"])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("DISPLAY", None)
        env.pop("XAUTHORITY", None)

        print(f"[worker {worker_id}] start {scene_id} gpu={gpu} port={port}", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log_file:
            header = f"# scene_id={scene_id} worker={worker_id} gpu={gpu} port={port}\n# cmd={cmd}\n"
            log_file.write(header.encode())
            log_file.flush()
            proc = subprocess.run(
                cmd,
                cwd=args.repo_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        elapsed = round(time.time() - start, 2)
        result = {
            **payload,
            "elapsed_seconds": elapsed,
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "log": str(log_path),
            "returncode": proc.returncode,
        }
        try:
            active_path.unlink()
        except FileNotFoundError:
            pass

        killed_after_success = False
        reason = None
        if proc.returncode == -signal.SIGKILL:
            killed_after_success, reason = _killed_after_success(log_path)
        if killed_after_success:
            result["completion_reason"] = reason
        episode_path, generated_count, target_count = _target_episode_count(
            args.loaded_config,
            args.config_path,
            scene_id,
            args.task_type,
        )
        if args.replay_episode_data and episode_path.exists():
            try:
                generated_count = len(json.loads(episode_path.read_text()))
                target_count = generated_count
            except (OSError, json.JSONDecodeError, TypeError):
                generated_count = 0
                target_count = 1
        result["episode_path"] = str(episode_path)
        result["generated_episodes"] = generated_count
        result["target_episodes"] = target_count
        target_reached = generated_count >= target_count
        replay_succeeded = not args.replay_episode_data or _episode_data_replay_succeeded(log_path)
        if (proc.returncode == 0 or killed_after_success) and target_reached and replay_succeeded:
            _write_json(args.run_root / "done" / f"{scene_id}.json", result)
            reason_text = f" reason={result['completion_reason']}" if killed_after_success else ""
            print(f"[worker {worker_id}] done {scene_id} elapsed={elapsed}s{reason_text}", flush=True)
        else:
            if proc.returncode == 0 or killed_after_success:
                result["failure_reason"] = (
                    "episode_data_replay_failed" if not replay_succeeded else "target_episode_count_not_reached"
                )
            _write_json(args.run_root / "failed" / f"{scene_id}.json", result)
            print(
                f"[worker {worker_id}] failed {scene_id} rc={proc.returncode} "
                f"episodes={generated_count}/{target_count} elapsed={elapsed}s log={log_path}",
                flush=True,
            )
        jobs.task_done()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("configs/vcp.yaml"))
    parser.add_argument("--scene-folder", default="navverse_data")
    parser.add_argument("--scene-type", default="vcp")
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--scene-ids-file", type=Path)
    parser.add_argument("--exclude-scene-ids", nargs="*", default=None)
    parser.add_argument("--task-type", default="objnav", choices=["objnav", "vln", "locnav", "placenav"])
    parser.add_argument("--trajectory-planner", default="astar", choices=["navmesh", "astar"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--gpu-offset", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=49000)
    parser.add_argument("--replay-episode-data", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    if not args.config_path.is_absolute():
        args.config_path = args.repo_root / args.config_path
    args.run_root.mkdir(parents=True, exist_ok=True)
    for dirname in ("active", "done", "failed", "logs"):
        (args.run_root / dirname).mkdir(exist_ok=True)

    args.loaded_config = load_config(args.config_path)
    file_scene_ids = None
    if args.scene_ids_file:
        file_scene_ids = [line.strip() for line in args.scene_ids_file.read_text().splitlines() if line.strip()]
    scene_ids = args.scene_ids or file_scene_ids or [
        scene_id for scene_id in list_scene_ids(args.loaded_config)
        if scene_id.split("_", 1)[0] == args.scene_type
    ]
    if args.exclude_scene_ids:
        excluded = set(args.exclude_scene_ids)
        scene_ids = [scene_id for scene_id in scene_ids if scene_id not in excluded]
    jobs: queue.Queue[str] = queue.Queue()
    for scene_id in scene_ids:
        jobs.put(scene_id)

    _write_json(args.run_root / "scenes.json", {"scene_type": args.scene_type, "scene_ids": scene_ids})
    _write_json(
        args.run_root / "worker_map.json",
        {
            str(worker_id): {
                "gpu": args.gpu_offset + worker_id // args.workers_per_gpu,
                "port": args.base_port + worker_id,
            }
            for worker_id in range(args.workers)
        },
    )
    print(f"[supervisor] run_root={args.run_root}", flush=True)
    print(f"[supervisor] queued={len(scene_ids)} workers={args.workers}", flush=True)

    threads = [threading.Thread(target=_run_worker, args=(worker_id, args, jobs)) for worker_id in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print("[supervisor] finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
