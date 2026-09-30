#!/usr/bin/env python3
"""Schedule the controlled friction matrix across GPUs."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROBOTS = ("drone", "spot", "turtlebot")
FRICTIONS = (0.5, 1.0)
DEFAULT_EPISODE_TIMEOUT_S = 300.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="navverse_tools/terrain_eval/configs/vcp_shared_100_seed42.json",
    )
    parser.add_argument(
        "--exclusions",
        default="navverse_tools/terrain_eval/configs/vcp_exclude_spot_taskgen_timeout20.json",
    )
    parser.add_argument(
        "--selection",
        default=None,
        help="Optional JSON with episode_labels; overrides --exclusions.",
    )
    parser.add_argument("--robots", default=",".join(ROBOTS))
    parser.add_argument(
        "--episode-timeout-s", type=float, default=DEFAULT_EPISODE_TIMEOUT_S
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cell-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def friction_slug(friction):
    return f"{friction:g}".replace(".", "p")


def cell_name(friction, robot):
    return f"material_mu{friction_slug(friction)}_{robot}"


def write_json_atomic(path, payload):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def validate_result(result_path, expected_labels):
    if not result_path.is_file():
        return False, "missing result.json"
    try:
        rows = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if not isinstance(rows, list):
        return False, "result.json is not a list"
    labels = [row.get("episode_label") for row in rows if isinstance(row, dict)]
    expected = set(expected_labels)
    if len(rows) != len(expected) or len(set(labels)) != len(expected):
        return False, f"rows={len(rows)} unique={len(set(labels))} expected={len(expected)}"
    if set(labels) != expected:
        return False, "episode labels do not match the shared filtered set"
    required = ("success", "average_horizontal_speed")
    for row in rows:
        try:
            values = [float(row[name]) for name in required]
        except (KeyError, TypeError, ValueError):
            return False, f"missing or non-numeric metric in {row.get('episode_label')}"
        if not all(math.isfinite(value) for value in values):
            return False, f"non-finite metric in {row.get('episode_label')}"
    return True, f"{len(expected)}/{len(expected)} exact unique results"


def launch_cell(
    repo_root,
    manifest_path,
    run_root,
    expected_labels,
    friction,
    robot,
    gpu_id,
    port,
    episode_timeout_s,
):
    cell = cell_name(friction, robot)
    cell_dir = run_root / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "navverse_tools.terrain_eval.run_cell",
        "--manifest",
        str(manifest_path),
        "--terrain",
        "original",
        "--robot",
        robot,
        "--gpu-id",
        gpu_id,
        "--port",
        str(port),
        "--run-root",
        str(run_root),
        "--cell-name",
        cell,
        "--material-friction",
        str(friction),
        "--episode-timeout-s",
        str(episode_timeout_s),
        "--episode-labels",
        ",".join(expected_labels),
        "--resume",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONUNBUFFERED"] = "1"
    log_file = (cell_dir / "matrix_launch.log").open("a", buffering=1)
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


def stop_running(running):
    for process, _log_file, _gpu_id, _spec in running.values():
        if process.poll() is None:
            process.terminate()
    for process, log_file, _gpu_id, _spec in running.values():
        try:
            process.wait(timeout=45.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
        log_file.close()


def summarize(run_root, expected_labels, robots, episode_timeout_s):
    cells = {}
    for friction in FRICTIONS:
        for robot in robots:
            cell = cell_name(friction, robot)
            rows = json.loads((run_root / cell / "result.json").read_text())
            by_label = {row["episode_label"]: row for row in rows}
            ordered = [by_label[label] for label in expected_labels]
            cells[cell] = {
                "friction": friction,
                "robot": robot,
                "episodes": len(ordered),
                "success_rate": sum(float(row["success"]) for row in ordered) / len(ordered),
                "average_horizontal_speed_mps": sum(
                    float(row["average_horizontal_speed"]) for row in ordered
                )
                / len(ordered),
                "termination_reasons": {
                    reason: sum(row.get("termination_reason") == reason for row in ordered)
                    for reason in sorted({row.get("termination_reason", "missing") for row in ordered})
                },
            }
    summary = {
        "expected_episodes": len(expected_labels),
        "episode_labels": expected_labels,
        "episode_timeout_s": episode_timeout_s,
        "cells": cells,
        "validation_errors": [],
    }
    write_json_atomic(run_root / "summary.json", summary)
    return summary


def main():
    args = parse_args()
    if args.cell_retries < 0:
        raise ValueError("--cell-retries must be non-negative")
    if args.episode_timeout_s <= 0.0:
        raise ValueError("--episode-timeout-s must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest_labels = [row["episode_label"] for row in manifest["episodes"]]
    selection_path = None
    exclusions_path = None
    excluded_labels = []
    if args.selection:
        selection_path = Path(args.selection)
        if not selection_path.is_absolute():
            selection_path = repo_root / selection_path
        expected_labels = json.loads(selection_path.read_text())["episode_labels"]
        if (
            not expected_labels
            or len(set(expected_labels)) != len(expected_labels)
            or not set(expected_labels).issubset(manifest_labels)
        ):
            raise RuntimeError("Selection must contain unique frozen-manifest labels")
    else:
        exclusions_path = Path(args.exclusions)
        if not exclusions_path.is_absolute():
            exclusions_path = repo_root / exclusions_path
        excluded_labels = json.loads(exclusions_path.read_text())["episode_labels"]
        if len(excluded_labels) != 20 or len(set(excluded_labels)) != 20:
            raise RuntimeError("Expected exactly 20 unique excluded episode labels")
        if not set(excluded_labels).issubset(manifest_labels):
            raise RuntimeError("Exclusions are not a subset of the frozen manifest")
        excluded = set(excluded_labels)
        expected_labels = [label for label in manifest_labels if label not in excluded]
        if len(expected_labels) != 80:
            raise RuntimeError(f"Expected 80 filtered labels, got {len(expected_labels)}")

    robots = [item.strip() for item in args.robots.split(",") if item.strip()]
    if not robots or len(set(robots)) != len(robots) or not set(robots).issubset(ROBOTS):
        raise ValueError(f"--robots must be unique values from {ROBOTS}")

    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-ids must contain unique GPU ids")
    specs = [(friction, robot) for friction in FRICTIONS for robot in robots]

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    status = {
        "state": "running",
        "started_at": utc_now(),
        "run_root": str(run_root),
        "manifest": str(manifest_path.resolve()),
        "selection": str(selection_path.resolve()) if selection_path else None,
        "exclusions": str(exclusions_path.resolve()) if exclusions_path else None,
        "excluded_episode_count": len(excluded_labels),
        "expected_episode_count": len(expected_labels),
        "episode_timeout_s": args.episode_timeout_s,
        "robots": robots,
        "cells": {},
    }
    if args.resume and status_path.is_file():
        previous = json.loads(status_path.read_text())
        status["started_at"] = previous.get("started_at", status["started_at"])

    pending = []
    for cell_index, (friction, robot) in enumerate(specs):
        cell = cell_name(friction, robot)
        valid, message = validate_result(
            run_root / cell / "result.json", expected_labels
        )
        if valid:
            status["cells"][cell] = {"state": "complete", "validation": message}
        else:
            pending.append((cell_index, friction, robot, 0))
    write_json_atomic(status_path, status)

    available_gpus = list(gpu_ids)
    running = {}
    failure = None
    try:
        while pending or running:
            while pending and available_gpus:
                cell_index, friction, robot, previous_attempts = pending.pop(0)
                gpu_id = available_gpus.pop(0)
                cell = cell_name(friction, robot)
                process, log_file, command = launch_cell(
                    repo_root,
                    manifest_path.resolve(),
                    run_root,
                    expected_labels,
                    friction,
                    robot,
                    gpu_id,
                    16000 + cell_index * 100,
                    args.episode_timeout_s,
                )
                running[cell] = (
                    process,
                    log_file,
                    gpu_id,
                    (cell_index, friction, robot, previous_attempts + 1),
                )
                status["cells"][cell] = {
                    "state": "running",
                    "gpu_id": gpu_id,
                    "pid": process.pid,
                    "attempt": previous_attempts + 1,
                    "friction": friction,
                    "robot": robot,
                    "command": command,
                }
                write_json_atomic(status_path, status)

            for cell, (process, log_file, gpu_id, spec) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_file.close()
                del running[cell]
                available_gpus.append(gpu_id)
                cell_index, friction, robot, attempt = spec
                valid, message = validate_result(
                    run_root / cell / "result.json", expected_labels
                )
                if valid:
                    status["cells"][cell] = {
                        "state": "complete",
                        "finished_at": utc_now(),
                        "returncode": returncode,
                        "validation": message,
                    }
                elif attempt <= args.cell_retries:
                    pending.append((cell_index, friction, robot, attempt))
                    status["cells"][cell] = {
                        "state": "retrying",
                        "returncode": returncode,
                        "validation": message,
                    }
                else:
                    failure = f"Cell {cell} failed validation after {attempt} attempts: {message}"
                    status["cells"][cell] = {
                        "state": "failed",
                        "returncode": returncode,
                        "validation": message,
                    }
                write_json_atomic(status_path, status)
            if failure:
                break
            time.sleep(2.0)
    finally:
        if running:
            stop_running(running)
    if failure:
        status["state"] = "failed"
        status["error"] = failure
        write_json_atomic(status_path, status)
        raise RuntimeError(failure)

    summarize(run_root, expected_labels, robots, args.episode_timeout_s)
    status["state"] = "complete"
    status["finished_at"] = utc_now()
    write_json_atomic(status_path, status)
    print(run_root)


if __name__ == "__main__":
    main()
