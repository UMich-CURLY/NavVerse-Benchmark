#!/usr/bin/env python3
"""Schedule the standard terrain-evaluation matrix across GPUs."""
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
TERRAINS = ("original", "flat", "slope", "stairs", "uneven")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="navverse_tools/terrain_eval/configs/vcp_shared_100_seed42.json",
    )
    parser.add_argument("--gpu-ids", default="1,2,3,4,5,6,7")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--robots", default=",".join(ROBOTS))
    parser.add_argument("--terrains", default=",".join(TERRAINS))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, status):
    status["updated_at"] = utc_now()
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def validate_cell(result_path, expected):
    if not result_path.is_file():
        return False, "missing result.json"
    try:
        rows = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, str(exc)
    labels = [row.get("episode_label") for row in rows if isinstance(row, dict)]
    if len(rows) != 100 or len(set(labels)) != 100 or set(labels) != expected:
        return False, f"rows={len(rows)} unique={len(set(labels))} expected_match={set(labels) == expected}"
    required = {"success", "average_horizontal_speed", "terrain_patches_completed"}
    if any(not required.issubset(row) for row in rows):
        return False, "required controlled-evaluation metrics are missing"
    for row in rows:
        try:
            values = [float(row[name]) for name in required]
        except (TypeError, ValueError):
            return False, f"non-numeric controlled metric in {row.get('episode_label')}"
        if not all(math.isfinite(value) for value in values):
            return False, f"non-finite controlled metric in {row.get('episode_label')}"
        if not 0.0 <= float(row["success"]) <= 1.0:
            return False, f"invalid success in {row.get('episode_label')}"
        if result_path.parent.name.endswith("_drone") and float(row["average_horizontal_speed"]) > 2.52:
            return False, f"drone speed exceeds 2.5 m/s in {row.get('episode_label')}"
    return True, "100/100 exact unique results"


def collect_pending_cells(cells, run_root, expected, status):
    pending = []
    for cell_index, cell in enumerate(cells):
        valid, message = validate_cell(run_root / cell / "result.json", expected)
        if valid:
            status["cells"][cell] = {"state": "complete", "validation": message}
        else:
            pending.append((cell_index, cell, message))
    return pending


def launch_cell(repo_root, run_root, manifest_path, cell_spec, gpu_id, resume):
    cell_index, cell, previous_validation = cell_spec
    terrain, robot = cell.split("_", 1)
    cell_dir = run_root / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "navverse_tools.terrain_eval.run_cell",
        "--manifest", str(manifest_path),
        "--terrain", terrain,
        "--robot", robot,
        "--gpu-id", gpu_id,
        "--port", str(13000 + cell_index * 100),
        "--run-root", str(run_root),
    ]
    if resume:
        command.append("--resume")
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
    state = {
        "state": "running",
        "started_at": utc_now(),
        "gpu_id": gpu_id,
        "pid": process.pid,
        "command": command,
        "previous_validation": previous_validation,
    }
    return process, log_file, state


def stop_running_cells(running):
    for process, log_file, _gpu_id in running.values():
        process.terminate()
        try:
            process.wait(timeout=45.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
        log_file.close()


def run_pending_cells(repo_root, run_root, manifest_path, expected, pending, gpu_ids, status, status_path, resume):
    available_gpus = list(gpu_ids)
    running = {}
    failed = None
    try:
        while pending or running:
            while pending and available_gpus and failed is None:
                cell_spec = pending.pop(0)
                cell = cell_spec[1]
                gpu_id = available_gpus.pop(0)
                process, log_file, state = launch_cell(
                    repo_root, run_root, manifest_path, cell_spec, gpu_id, resume
                )
                running[cell] = (process, log_file, gpu_id)
                status["cells"][cell] = state
                write_status(status_path, status)

            for cell, (process, log_file, gpu_id) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_file.close()
                del running[cell]
                available_gpus.append(gpu_id)
                valid, message = validate_cell(run_root / cell / "result.json", expected)
                status["cells"][cell] = {
                    "state": "complete" if valid else "failed",
                    "finished_at": utc_now(),
                    "returncode": returncode,
                    "validation": message,
                }
                write_status(status_path, status)
                if not valid and failed is None:
                    failed = f"Cell {cell} failed validation: {message}"
            if failed is not None:
                break
            time.sleep(2.0)
    finally:
        if failed is not None:
            stop_running_cells(running)
    if failed is not None:
        raise RuntimeError(failed)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    expected = {row["episode_label"] for row in manifest["episodes"]}
    if len(expected) != 100:
        raise RuntimeError(f"Expected 100 manifest labels, got {len(expected)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root or repo_root / "dump" / f"vcp_controlled_terrain_10x10_seed42_{stamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    robots = tuple(item for item in args.robots.split(",") if item)
    terrains = tuple(item for item in args.terrains.split(",") if item)
    if not set(robots).issubset(ROBOTS) or not set(terrains).issubset(TERRAINS):
        raise ValueError("Unsupported robot or terrain selection")
    cells = [f"{terrain}_{robot}" for terrain in terrains for robot in robots]
    status = {
        "run_root": str(run_root),
        "manifest": str(manifest_path),
        "manifest_episode_count": len(expected),
        "gpu_ids": args.gpu_ids,
        "cells": {cell: {"state": "pending"} for cell in cells},
        "started_at": utc_now(),
    }
    if args.resume and status_path.is_file():
        status = json.loads(status_path.read_text())
    write_status(status_path, status)

    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    pending_cells = collect_pending_cells(cells, run_root, expected, status)
    write_status(status_path, status)
    run_pending_cells(
        repo_root,
        run_root,
        manifest_path,
        expected,
        pending_cells,
        gpu_ids,
        status,
        status_path,
        args.resume,
    )

    summary_command = [
        sys.executable,
        "-m",
        "navverse_tools.terrain_eval.summarize",
        str(run_root),
        "--manifest",
        str(manifest_path),
        "--robots",
        ",".join(robots),
        "--terrains",
        ",".join(terrains),
    ]
    subprocess.run(summary_command, cwd=repo_root, check=True)
    status["state"] = "complete"
    status["finished_at"] = utc_now()
    write_status(status_path, status)
    print(run_root)


if __name__ == "__main__":
    main()
