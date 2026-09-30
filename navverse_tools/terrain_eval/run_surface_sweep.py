#!/usr/bin/env python3
"""Run small controlled surface-physics calibration sweeps."""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROBOTS = ("spot", "turtlebot")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="navverse_tools/terrain_eval/configs/vcp_shared_100_seed42.json",
    )
    parser.add_argument(
        "--selection",
        default="navverse_tools/terrain_eval/configs/vcp_surface_gate_5.json",
    )
    parser.add_argument(
        "--sweep",
        default="navverse_tools/terrain_eval/configs/surface_sweep_v1.json",
    )
    parser.add_argument("--gpu-ids", default="1,2,3,4,5,6,7")
    parser.add_argument("--robots", default=",".join(ROBOTS))
    parser.add_argument("--episode-timeout-s", type=float, default=75.0)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cell-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve_path(repo_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def write_json_atomic(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def load_rows(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def validate_rows(rows, expected_labels):
    labels = [row.get("episode_label") for row in rows if isinstance(row, dict)]
    if labels != expected_labels:
        return False, f"labels={labels} expected={expected_labels}"
    for row in rows:
        try:
            values = (float(row["success"]), float(row["average_horizontal_speed"]))
        except (KeyError, TypeError, ValueError):
            return False, f"invalid metrics for {row.get('episode_label')}"
        if not all(math.isfinite(value) for value in values):
            return False, f"non-finite metrics for {row.get('episode_label')}"
    return True, f"{len(rows)}/{len(expected_labels)} exact ordered results"


def cell_name(condition, robot):
    return f"{condition['name']}_{robot}"


def launch_cell(repo_root, args, manifest, labels, condition, robot, gpu_id, cell_index):
    run_root = Path(args.run_root).resolve()
    cell = cell_name(condition, robot)
    cell_dir = run_root / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "navverse_tools.terrain_eval.run_cell",
        "--manifest", str(manifest),
        "--terrain", condition["terrain"],
        "--terrain-profile", condition.get("terrain_profile", "standard"),
        "--robot", robot,
        "--gpu-id", gpu_id,
        "--port", str(17000 + cell_index * 100),
        "--run-root", str(run_root),
        "--cell-name", cell,
        "--episode-labels", ",".join(labels),
        "--episode-timeout-s", str(args.episode_timeout_s),
        "--material-scope", "terrain",
        "--material-static-friction", str(condition["static_friction"]),
        "--material-dynamic-friction", str(condition["dynamic_friction"]),
        "--material-restitution", str(condition.get("restitution", 0.0)),
        "--material-compliant-stiffness", str(condition.get("compliant_stiffness", 0.0)),
        "--material-compliant-damping", str(condition.get("compliant_damping", 0.0)),
        "--material-friction-combine-mode", condition.get("friction_combine_mode", "multiply"),
        "--resume",
    ]
    if "turtlebot_joint_dynamic_friction" in condition:
        command.extend(
            [
                "--turtlebot-joint-dynamic-friction",
                str(condition["turtlebot_joint_dynamic_friction"]),
            ]
        )
    if "turtlebot_joint_viscous_friction" in condition:
        command.extend(
            [
                "--turtlebot-joint-viscous-friction",
                str(condition["turtlebot_joint_viscous_friction"]),
            ]
        )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONUNBUFFERED"] = "1"
    log_file = (cell_dir / "sweep_launch.log").open("a", buffering=1)
    log_file.write(f"[{utc_now()}] {' '.join(command)}\n")
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_file, command


def summarize(run_root, specs, labels):
    cells = {}
    for condition, robot in specs:
        cell = cell_name(condition, robot)
        rows = load_rows(run_root / cell / "result.json")
        valid, validation = validate_rows(rows, labels)
        if not valid:
            raise RuntimeError(f"{cell}: {validation}")
        cells[cell] = {
            "condition": condition,
            "robot": robot,
            "episodes": len(rows),
            "success_rate": sum(float(row["success"]) for row in rows) / len(rows),
            "average_horizontal_speed_mps": sum(
                float(row["average_horizontal_speed"]) for row in rows
            ) / len(rows),
            "successful_only_speed_mps": (
                sum(float(row["average_horizontal_speed"]) for row in rows if float(row["success"]))
                / max(1, sum(bool(float(row["success"])) for row in rows))
            ),
            "termination_reasons": dict(Counter(row.get("termination_reason", "missing") for row in rows)),
            "validation": validation,
        }
    summary = {"episode_labels": labels, "cells": cells}
    write_json_atomic(run_root / "summary.json", summary)
    return summary


def stop_running(running):
    for process, _log_file, _gpu_id, _spec in running.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process, log_file, _gpu_id, _spec in running.values():
        try:
            process.wait(timeout=45.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10.0)
        log_file.close()


def main():
    args = parse_args()
    if args.cell_retries < 0:
        raise ValueError("--cell-retries must be non-negative")
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_path(repo_root, args.manifest)
    selection_path = resolve_path(repo_root, args.selection)
    sweep_path = resolve_path(repo_root, args.sweep)
    manifest_labels = [row["episode_label"] for row in json.loads(manifest_path.read_text())["episodes"]]
    labels = json.loads(selection_path.read_text())["episode_labels"]
    if not labels or len(set(labels)) != len(labels) or not set(labels).issubset(manifest_labels):
        raise RuntimeError("Selection must be a non-empty unique manifest subset")
    conditions = json.loads(sweep_path.read_text())["conditions"]
    condition_names = [condition["name"] for condition in conditions]
    if len(condition_names) != len(set(condition_names)):
        raise RuntimeError("Sweep condition names must be unique")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise RuntimeError("GPU ids must be unique")

    robots = [value.strip() for value in args.robots.split(",") if value.strip()]
    if not robots or len(robots) != len(set(robots)) or not set(robots).issubset(ROBOTS):
        raise RuntimeError(f"Robots must be unique values from {ROBOTS}")
    specs = [(condition, robot) for condition in conditions for robot in robots]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    status = {
        "state": "running",
        "started_at": utc_now(),
        "manifest": str(manifest_path),
        "selection": str(selection_path),
        "sweep": str(sweep_path),
        "episode_labels": labels,
        "gpu_ids": gpu_ids,
        "robots": robots,
        "cells": {},
    }
    if args.resume and status_path.is_file():
        status["started_at"] = json.loads(status_path.read_text()).get("started_at", status["started_at"])

    pending = []
    for index, (condition, robot) in enumerate(specs):
        cell = cell_name(condition, robot)
        valid, validation = validate_rows(load_rows(run_root / cell / "result.json"), labels)
        if valid:
            status["cells"][cell] = {"state": "complete", "validation": validation}
        else:
            pending.append((index, condition, robot, 0))
    write_json_atomic(status_path, status)

    available_gpus = list(gpu_ids)
    running = {}
    failure = None
    try:
        while pending or running:
            while pending and available_gpus:
                index, condition, robot, attempts = pending.pop(0)
                gpu_id = available_gpus.pop(0)
                cell = cell_name(condition, robot)
                process, log_file, command = launch_cell(
                    repo_root, args, manifest_path, labels, condition, robot, gpu_id, index
                )
                running[cell] = (process, log_file, gpu_id, (index, condition, robot, attempts + 1))
                status["cells"][cell] = {
                    "state": "running", "gpu_id": gpu_id, "pid": process.pid,
                    "attempt": attempts + 1, "command": command,
                }
                write_json_atomic(status_path, status)

            time.sleep(2.0)
            for cell, (process, log_file, gpu_id, spec) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_file.close()
                del running[cell]
                available_gpus.append(gpu_id)
                index, condition, robot, attempts = spec
                valid, validation = validate_rows(load_rows(run_root / cell / "result.json"), labels)
                if returncode == 0 and valid:
                    status["cells"][cell] = {"state": "complete", "validation": validation}
                elif attempts <= args.cell_retries:
                    status["cells"][cell] = {
                        "state": "retrying", "returncode": returncode, "validation": validation,
                    }
                    pending.append((index, condition, robot, attempts))
                else:
                    failure = f"{cell} failed after {attempts} attempts: rc={returncode}, {validation}"
                    status["cells"][cell] = {"state": "failed", "error": failure}
                    raise RuntimeError(failure)
                write_json_atomic(status_path, status)
    except BaseException as exc:
        failure = failure or f"{type(exc).__name__}: {exc}"
        stop_running(running)
    if failure:
        status["state"] = "failed"
        status["error"] = failure
        status["finished_at"] = utc_now()
        write_json_atomic(status_path, status)
        raise RuntimeError(failure)

    summarize(run_root, specs, labels)
    status["state"] = "complete"
    status["finished_at"] = utc_now()
    write_json_atomic(status_path, status)


if __name__ == "__main__":
    main()
