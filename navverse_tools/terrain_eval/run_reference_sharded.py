#!/usr/bin/env python3
"""Execute a frozen reference-path selection in contiguous GPU shards."""
import argparse
from collections import Counter
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="navverse_tools/terrain_eval/configs/vcp_shared_100_seed42.json",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--episode-labels", default="")
    parser.add_argument(
        "--reference-execution",
        choices=("external_oracle", "task_generation_internal"),
        default="external_oracle",
    )
    parser.add_argument("--diagnostic-video", action="store_true")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--port-base", type=int, default=15000)
    parser.add_argument("--shard-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, payload):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def contiguous_shards(labels, shard_count):
    quotient, remainder = divmod(len(labels), shard_count)
    shards = []
    offset = 0
    for shard_index in range(shard_count):
        size = quotient + (1 if shard_index < remainder else 0)
        shards.append(labels[offset : offset + size])
        offset += size
    return shards


def load_rows(result_path):
    try:
        rows = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return rows if isinstance(rows, list) else []


def validate_rows(rows, expected_labels):
    labels = [row.get("episode_label") for row in rows if isinstance(row, dict)]
    expected = set(expected_labels)
    if len(rows) != len(expected) or len(set(labels)) != len(expected):
        return False, f"rows={len(rows)} unique={len(set(labels))} expected={len(expected)}"
    if set(labels) != expected:
        return False, "episode labels do not match shard"
    required = {"success", "average_horizontal_speed", "terrain_patches_completed"}
    for row in rows:
        if not required.issubset(row):
            return False, f"missing metrics in {row.get('episode_label')}"
        try:
            values = [float(row[name]) for name in required]
        except (TypeError, ValueError):
            return False, f"non-numeric metrics in {row.get('episode_label')}"
        if not all(math.isfinite(value) for value in values):
            return False, f"non-finite metrics in {row.get('episode_label')}"
    return True, f"{len(expected)}/{len(expected)} exact unique results"


def merge_shard_results(run_root, shards):
    ordered_labels = [label for shard in shards for label in shard]
    by_label = {}
    for shard_index, expected_labels in enumerate(shards):
        result_path = (
            run_root
            / "shards"
            / f"shard_{shard_index:02d}"
            / "original_spot"
            / "result.json"
        )
        rows = load_rows(result_path)
        valid, message = validate_rows(rows, expected_labels)
        if not valid:
            raise RuntimeError(f"Shard {shard_index:02d} failed validation: {message}")
        for row in rows:
            label = row["episode_label"]
            if label in by_label:
                raise RuntimeError(f"Duplicate result across shards: {label}")
            by_label[label] = row
    merged = [by_label[label] for label in ordered_labels]
    cell_dir = run_root / "original_spot"
    cell_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cell_dir / "result.json", merged)
    return merged


def launch_shard(
    repo_root,
    manifest_path,
    run_root,
    shard_index,
    labels,
    gpu_id,
    port,
    reference_execution,
    diagnostic_video,
):
    shard_root = run_root / "shards" / f"shard_{shard_index:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "navverse_tools.terrain_eval.run_cell",
        "--manifest",
        str(manifest_path),
        "--terrain",
        "original",
        "--robot",
        "spot",
        "--gpu-id",
        gpu_id,
        "--port",
        str(port),
        "--run-root",
        str(shard_root),
        "--episode-labels",
        ",".join(labels),
        "--waypoint-follower-profile",
        "task_generation",
        "--reference-execution",
        reference_execution,
        "--resume",
    ]
    if diagnostic_video:
        command.append("--diagnostic-video")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONUNBUFFERED"] = "1"
    log_file = (shard_root / "shard.log").open("a", buffering=1)
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


def stop_processes(running):
    for process, log_file, _shard_index in running.values():
        if process.poll() is None:
            process.terminate()
    for process, log_file, _shard_index in running.values():
        try:
            process.wait(timeout=45.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10.0)
        log_file.close()


def main():
    args = parse_args()
    if args.shard_retries < 0:
        raise ValueError("--shard-retries must be non-negative")
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest_labels = [row["episode_label"] for row in manifest["episodes"]]
    if len(manifest_labels) != 100 or len(set(manifest_labels)) != 100:
        raise RuntimeError("Controlled reference evaluation requires 100 unique labels")
    requested = [label for label in args.episode_labels.split(",") if label]
    if len(set(requested)) != len(requested) or not set(requested).issubset(manifest_labels):
        raise ValueError("--episode-labels must be a unique subset of the manifest")
    requested_set = set(requested)
    labels = (
        [label for label in manifest_labels if label in requested_set]
        if requested
        else manifest_labels
    )
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-ids must contain unique GPU ids")
    shards = contiguous_shards(labels, len(gpu_ids))
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    status = {
        "state": "running",
        "started_at": utc_now(),
        "run_root": str(run_root),
        "manifest": str(manifest_path),
        "manifest_episode_count": len(manifest_labels),
        "target_episode_count": len(labels),
        "robot": "spot",
        "terrain": "original",
        "path_source": "stored_reference_path",
        "waypoint_follower_profile": "task_generation",
        "reference_execution": args.reference_execution,
        "gpu_ids": gpu_ids,
        "shards": {},
    }
    if args.resume and status_path.is_file():
        previous = json.loads(status_path.read_text())
        status["started_at"] = previous.get("started_at", status["started_at"])

    pending = []
    for shard_index, (gpu_id, shard_labels) in enumerate(zip(gpu_ids, shards)):
        result_path = (
            run_root
            / "shards"
            / f"shard_{shard_index:02d}"
            / "original_spot"
            / "result.json"
        )
        valid, message = validate_rows(load_rows(result_path), shard_labels)
        if valid:
            status["shards"][f"shard_{shard_index:02d}"] = {
                "state": "complete",
                "gpu_id": gpu_id,
                "validation": message,
            }
        else:
            pending.append((shard_index, gpu_id, shard_labels, 0))
    write_json_atomic(status_path, status)

    running = {}
    failed = None
    try:
        while pending or running:
            while pending:
                shard_index, gpu_id, shard_labels, attempt = pending.pop(0)
                process, log_file, command = launch_shard(
                    repo_root,
                    manifest_path,
                    run_root,
                    shard_index,
                    shard_labels,
                    gpu_id,
                    args.port_base + shard_index * 100,
                    args.reference_execution,
                    args.diagnostic_video,
                )
                running[gpu_id] = (process, log_file, shard_index)
                status["shards"][f"shard_{shard_index:02d}"] = {
                    "state": "running",
                    "gpu_id": gpu_id,
                    "pid": process.pid,
                    "attempt": attempt + 1,
                    "episode_count": len(shard_labels),
                    "first_episode": shard_labels[0],
                    "last_episode": shard_labels[-1],
                    "command": command,
                }
                write_json_atomic(status_path, status)

            for gpu_id, (process, log_file, shard_index) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_file.close()
                del running[gpu_id]
                shard_labels = shards[shard_index]
                result_path = (
                    run_root
                    / "shards"
                    / f"shard_{shard_index:02d}"
                    / "original_spot"
                    / "result.json"
                )
                valid, message = validate_rows(load_rows(result_path), shard_labels)
                current_attempt = int(
                    status["shards"][f"shard_{shard_index:02d}"]["attempt"]
                )
                if valid:
                    status["shards"][f"shard_{shard_index:02d}"] = {
                        "state": "complete",
                        "gpu_id": gpu_id,
                        "returncode": returncode,
                        "finished_at": utc_now(),
                        "validation": message,
                    }
                elif current_attempt <= args.shard_retries:
                    pending.append(
                        (shard_index, gpu_id, shard_labels, current_attempt)
                    )
                    status["shards"][f"shard_{shard_index:02d}"] = {
                        "state": "retrying",
                        "gpu_id": gpu_id,
                        "returncode": returncode,
                        "validation": message,
                    }
                else:
                    failed = f"Shard {shard_index:02d} failed: {message}"
                    status["shards"][f"shard_{shard_index:02d}"] = {
                        "state": "failed",
                        "gpu_id": gpu_id,
                        "returncode": returncode,
                        "validation": message,
                    }
                write_json_atomic(status_path, status)
            if failed:
                break
            time.sleep(2.0)
    finally:
        if running:
            stop_processes(running)
    if failed:
        status["state"] = "failed"
        status["error"] = failed
        write_json_atomic(status_path, status)
        raise RuntimeError(failed)

    merged = merge_shard_results(run_root, shards)
    if labels == manifest_labels:
        summary_command = [
            sys.executable,
            "-m",
            "navverse_tools.terrain_eval.summarize",
            str(run_root),
            "--manifest",
            str(manifest_path),
            "--robots",
            "spot",
            "--terrains",
            "original",
        ]
        subprocess.run(summary_command, cwd=repo_root, check=True)
    else:
        termination_reasons = Counter(
            row.get("termination_reason", "unknown") for row in merged
        )
        write_json_atomic(
            run_root / "summary.json",
            {
                "episodes": len(merged),
                "unique_episodes": len({row["episode_label"] for row in merged}),
                "success_rate": sum(float(row["success"]) for row in merged)
                / len(merged),
                "average_horizontal_speed_mps": sum(
                    float(row["average_horizontal_speed"]) for row in merged
                )
                / len(merged),
                "termination_reasons": dict(sorted(termination_reasons.items())),
                "reference_execution": args.reference_execution,
                "waypoint_follower_profile": "task_generation",
            },
        )
    status["state"] = "complete"
    status["finished_at"] = utc_now()
    status["merged_result_count"] = len(merged)
    write_json_atomic(status_path, status)
    print(run_root)


if __name__ == "__main__":
    main()
