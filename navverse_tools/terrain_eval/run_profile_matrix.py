#!/usr/bin/env python3
"""Schedule a selected terrain-profile matrix across GPUs."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROBOTS = ("spot", "turtlebot")
TERRAINS = ("slope", "stairs", "uneven")
DEFAULT_EPISODE_TIMEOUT_S = 75.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="navverse_tools/terrain_eval/configs/vcp_shared_100_seed42.json",
    )
    parser.add_argument(
        "--selection",
        default="navverse_tools/terrain_eval/configs/vcp_gate_10_seed43.json",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument(
        "--terrain-profile",
        choices=(
            "hard",
            "moderate",
            "moderate_v2",
            "moderate_final",
            "moderate_compact",
            "moderate_wide",
            "moderate_traversable",
            "balanced",
            "extra_hard",
        ),
        default="hard",
    )
    parser.add_argument("--terrains", default=",".join(TERRAINS))
    parser.add_argument(
        "--episode-timeout-s", type=float, default=DEFAULT_EPISODE_TIMEOUT_S
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cell-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cell_name(profile, terrain, robot):
    return f"{profile}_{terrain}_{robot}"


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
    if labels != expected_labels or len(set(labels)) != len(expected_labels):
        return False, f"rows={len(rows)} unique={len(set(labels))} expected={len(expected_labels)}"
    for row in rows:
        try:
            values = (float(row["success"]), float(row["average_horizontal_speed"]))
        except (KeyError, TypeError, ValueError):
            return False, f"missing or non-numeric metric in {row.get('episode_label')}"
        if not all(math.isfinite(value) for value in values):
            return False, f"non-finite metric in {row.get('episode_label')}"
    return True, f"{len(expected_labels)}/{len(expected_labels)} exact ordered unique results"


def launch_cell(
    repo_root,
    manifest,
    run_root,
    labels,
    terrain,
    robot,
    gpu_id,
    port,
    terrain_profile,
    episode_timeout_s,
):
    cell = cell_name(terrain_profile, terrain, robot)
    cell_dir = run_root / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "navverse_tools.terrain_eval.run_cell",
        "--manifest",
        str(manifest),
        "--terrain",
        terrain,
        "--terrain-profile",
        terrain_profile,
        "--episode-timeout-s",
        str(episode_timeout_s),
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
        "--episode-labels",
        ",".join(labels),
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


def summarize(run_root, labels, terrain_profile, episode_timeout_s, terrains):
    cells = {}
    for terrain in terrains:
        for robot in ROBOTS:
            cell = cell_name(terrain_profile, terrain, robot)
            rows = json.loads((run_root / cell / "result.json").read_text())
            cells[cell] = {
                "terrain": terrain,
                "terrain_profile": terrain_profile,
                "robot": robot,
                "episodes": len(rows),
                "success_rate": sum(float(row["success"]) for row in rows) / len(rows),
                "average_horizontal_speed_mps": sum(
                    float(row["average_horizontal_speed"]) for row in rows
                )
                / len(rows),
                "termination_reasons": dict(
                    sorted(Counter(row.get("termination_reason", "missing") for row in rows).items())
                ),
            }
    summary = {
        "expected_episodes": len(labels),
        "episode_labels": labels,
        "terrain_profile": terrain_profile,
        "episode_timeout_s": episode_timeout_s,
        "cells": cells,
        "validation_errors": [],
    }
    write_json_atomic(run_root / "summary.json", summary)
    return summary


def launch_pending_cells(
    args,
    repo_root,
    manifest,
    run_root,
    labels,
    pending,
    available_gpus,
    running,
    status,
    status_path,
):
    while pending and available_gpus:
        index, terrain, robot, previous_attempts = pending.pop(0)
        gpu_id = available_gpus.pop(0)
        cell = cell_name(args.terrain_profile, terrain, robot)
        process, log_file, command = launch_cell(
            repo_root,
            manifest,
            run_root,
            labels,
            terrain,
            robot,
            gpu_id,
            21000 + index * 100,
            args.terrain_profile,
            args.episode_timeout_s,
        )
        running[cell] = (
            process,
            log_file,
            gpu_id,
            (index, terrain, robot, previous_attempts + 1),
        )
        status["cells"][cell] = {
            "state": "running",
            "gpu_id": gpu_id,
            "pid": process.pid,
            "attempt": previous_attempts + 1,
            "command": command,
        }
        write_json_atomic(status_path, status)


def reap_finished_cells(
    args,
    run_root,
    labels,
    pending,
    available_gpus,
    running,
    status,
    status_path,
):
    failure = None
    for cell, (process, log_file, gpu_id, spec) in list(running.items()):
        returncode = process.poll()
        if returncode is None:
            continue
        log_file.close()
        del running[cell]
        available_gpus.append(gpu_id)
        index, terrain, robot, attempt = spec
        valid, message = validate_result(run_root / cell / "result.json", labels)
        if valid:
            status["cells"][cell] = {
                "state": "complete",
                "finished_at": utc_now(),
                "returncode": returncode,
                "validation": message,
            }
        elif attempt <= args.cell_retries:
            pending.append((index, terrain, robot, attempt))
            status["cells"][cell] = {
                "state": "retrying",
                "returncode": returncode,
                "validation": message,
            }
        else:
            failure = f"Cell {cell} failed after {attempt} attempts: {message}"
            status["cells"][cell] = {
                "state": "failed",
                "returncode": returncode,
                "validation": message,
            }
        write_json_atomic(status_path, status)
    return failure


def run_scheduler(
    args,
    repo_root,
    manifest,
    run_root,
    labels,
    gpu_ids,
    pending,
    status,
    status_path,
):
    available_gpus = list(gpu_ids)
    running = {}
    failure = None
    try:
        while pending or running:
            launch_pending_cells(
                args,
                repo_root,
                manifest,
                run_root,
                labels,
                pending,
                available_gpus,
                running,
                status,
                status_path,
            )
            failure = reap_finished_cells(
                args,
                run_root,
                labels,
                pending,
                available_gpus,
                running,
                status,
                status_path,
            )
            if failure:
                break
            time.sleep(2.0)
    finally:
        for process, log_file, _gpu_id, _spec in running.values():
            if process.poll() is None:
                process.terminate()
            log_file.close()
    if failure:
        status["state"] = "failed"
        status["error"] = failure
        write_json_atomic(status_path, status)
        raise RuntimeError(failure)


def main():
    args = parse_args()
    if args.cell_retries < 0:
        raise ValueError("--cell-retries must be non-negative")
    if args.episode_timeout_s <= 0.0:
        raise ValueError("--episode-timeout-s must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from navverse_tools.terrain_eval.terrain import terrain_profile_config

    manifest = (repo_root / args.manifest).resolve()
    selection = (repo_root / args.selection).resolve()
    manifest_rows = json.loads(manifest.read_text())["episodes"]
    manifest_labels = [row["episode_label"] for row in manifest_rows]
    labels = json.loads(selection.read_text())["episode_labels"]
    if not labels or len(set(labels)) != len(labels) or not set(labels).issubset(manifest_labels):
        raise RuntimeError("Hard-terrain selection must contain unique manifest labels")
    by_label = {row["episode_label"]: row for row in manifest_rows}
    patch_length = terrain_profile_config(args.terrain_profile)["patch_length_m"]
    too_short = [
        label
        for label in labels
        if float(by_label[label]["path_length_m"]) + 1e-9 < 4.0 * patch_length
    ]
    if too_short:
        raise RuntimeError(
            f"Terrain patches would overlap on {len(too_short)} selected paths"
        )

    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    terrains = [item.strip() for item in args.terrains.split(",") if item.strip()]
    if (
        not terrains
        or len(set(terrains)) != len(terrains)
        or not set(terrains).issubset(TERRAINS)
    ):
        raise ValueError("--terrains must contain unique supported terrain names")
    specs = [(terrain, robot) for terrain in terrains for robot in ROBOTS]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-ids must contain at least one unique GPU id")

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    status = {
        "state": "running",
        "started_at": utc_now(),
        "run_root": str(run_root),
        "manifest": str(manifest),
        "selection": str(selection),
        "expected_episode_count": len(labels),
        "terrain_profile": args.terrain_profile,
        "terrains": terrains,
        "episode_timeout_s": args.episode_timeout_s,
        "cells": {},
    }
    if args.resume and status_path.is_file():
        previous = json.loads(status_path.read_text())
        status["started_at"] = previous.get("started_at", status["started_at"])

    pending = []
    for index, (terrain, robot) in enumerate(specs):
        cell = cell_name(args.terrain_profile, terrain, robot)
        valid, message = validate_result(run_root / cell / "result.json", labels)
        if valid:
            status["cells"][cell] = {"state": "complete", "validation": message}
        else:
            pending.append((index, terrain, robot, 0))
    write_json_atomic(status_path, status)

    run_scheduler(
        args,
        repo_root,
        manifest,
        run_root,
        labels,
        gpu_ids,
        pending,
        status,
        status_path,
    )

    summarize(
        run_root,
        labels,
        args.terrain_profile,
        args.episode_timeout_s,
        terrains,
    )
    status["state"] = "complete"
    status["finished_at"] = utc_now()
    write_json_atomic(status_path, status)
    print(run_root)


if __name__ == "__main__":
    main()
