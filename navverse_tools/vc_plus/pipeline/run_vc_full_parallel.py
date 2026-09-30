#!/usr/bin/env python3
"""Run VC Full generation in city shards bound to visible GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def now() -> str:
    return time.strftime("%F %T %Z")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def detect_gpus() -> list[int]:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5, check=False)
    except Exception:
        return [0]
    gpus: list[int] = []
    for line in result.stdout.splitlines():
        match = re.match(r"GPU\s+(\d+):", line.strip())
        if match:
            gpus.append(int(match.group(1)))
    return gpus or [0]


def split_round_robin(items: list[str], count: int) -> list[list[str]]:
    buckets = [[] for _ in range(max(1, count))]
    for index, item in enumerate(items):
        buckets[index % len(buckets)].append(item)
    return [bucket for bucket in buckets if bucket]


def strip_generate_cities(args: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(args):
        if args[index] in {"--cities", "--assignment-cities"}:
            index += 1
            while index < len(args) and not args[index].startswith("--"):
                index += 1
            continue
        stripped.append(args[index])
        index += 1
    return stripped


def city_status(work_root: Path, city: str) -> dict[str, Any]:
    manifest = work_root / city / "manifest.json"
    data = read_json(manifest, {})
    commands = data.get("commands", []) if isinstance(data, dict) else []
    last_command = commands[-1] if commands else {}
    return {
        "city": city,
        "manifest": str(manifest),
        "manifest_exists": manifest.exists(),
        "manifest_mtime": manifest.stat().st_mtime if manifest.exists() else None,
        "last_stage": last_command.get("stage"),
        "last_finished_at": last_command.get("finished_at"),
        "warnings": data.get("warnings", []) if isinstance(data, dict) else [],
    }


def worker_done_cities(work_root: Path, cities: list[str], started_at: float) -> list[str]:
    done = []
    for city in cities:
        manifest = work_root / city / "manifest.json"
        if manifest.exists() and manifest.stat().st_mtime >= started_at:
            done.append(city)
    return done


def newest_log_mtime(log_root: Path) -> float:
    newest = 0.0
    if not log_root.exists():
        return newest
    for path in log_root.rglob("*.log"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--assignment-cities", nargs="*", default=[])
    parser.add_argument("--exclude-cities", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpus", default="", help="Comma-separated GPU IDs. Defaults to all GPUs from nvidia-smi.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--generate-script", type=Path, default=Path(__file__).with_name("generate_vc_full.py"))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=1800)
    parser.add_argument("--stale-seconds", type=int, default=2700)
    parser.add_argument("--generate-args", nargs=argparse.REMAINDER, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cities = [city for city in args.cities if city not in set(args.exclude_cities)]
    assignment_cities = [city for city in (args.assignment_cities or cities) if city not in set(args.exclude_cities)]
    if not cities:
        raise SystemExit("no cities to process")
    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()] if args.gpus else detect_gpus()
    workers = args.workers if args.workers > 0 else len(gpu_ids)
    buckets = split_round_robin(cities, workers)
    generate_args = strip_generate_cities(args.generate_args)
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.progress_log.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    processes: list[dict[str, Any]] = []

    for worker_index, bucket in enumerate(buckets):
        gpu = gpu_ids[worker_index % len(gpu_ids)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["NVIDIA_VISIBLE_DEVICES"] = str(gpu)
        log_path = args.log_root / f"parallel_worker_{worker_index:02d}_gpu{gpu}.log"
        cmd = [
            args.python,
            str(args.generate_script),
            *generate_args,
            "--assignment-cities",
            *assignment_cities,
            "--cities",
            *bucket,
        ]
        log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        processes.append({"index": worker_index, "gpu": gpu, "cities": bucket, "pid": proc.pid, "proc": proc, "log": str(log_path)})

    def snapshot() -> dict[str, Any]:
        worker_rows = []
        newest_manifest_mtime = 0.0
        for row in processes:
            proc = row["proc"]
            done = worker_done_cities(args.work_root, row["cities"], started_at)
            pending = [city for city in row["cities"] if city not in set(done)]
            statuses = [city_status(args.work_root, city) for city in row["cities"]]
            mtimes = [item["manifest_mtime"] for item in statuses if item["manifest_mtime"]]
            if mtimes:
                newest_manifest_mtime = max(newest_manifest_mtime, max(mtimes))
            worker_rows.append(
                {
                    "index": row["index"],
                    "gpu": row["gpu"],
                    "pid": row["pid"],
                    "returncode": proc.poll(),
                    "done_count": len(done),
                    "pending_count": len(pending),
                    "done": done,
                    "pending": pending,
                    "log": row["log"],
                    "last_pending_status": city_status(args.work_root, pending[0]) if pending else None,
                }
            )
        newest_activity = max(newest_manifest_mtime, newest_log_mtime(args.log_root), started_at)
        stale_seconds = time.time() - newest_activity
        return {
            "started_at": started_at,
            "updated_at": time.time(),
            "updated_at_text": now(),
            "cities_total": len(cities),
            "done_total": sum(row["done_count"] for row in worker_rows),
            "pending_total": sum(row["pending_count"] for row in worker_rows),
            "workers": worker_rows,
            "stale_seconds": round(stale_seconds, 1),
            "stale": stale_seconds > args.stale_seconds,
        }

    while True:
        state = snapshot()
        write_json(args.status_json, state)
        with args.progress_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(state, sort_keys=True) + "\n")
        if all(row["proc"].poll() is not None for row in processes):
            break
        time.sleep(max(30, args.progress_interval))

    final_state = snapshot()
    write_json(args.status_json, final_state)
    return 1 if any(row["proc"].returncode for row in processes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
