#!/usr/bin/env python3
"""Run VC scene photo renders through parallel Blender workers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_index_set(value: str) -> list[int]:
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def facade_indices(plan: dict) -> list[int]:
    indices: list[int] = []
    for index, item in enumerate(plan.get("placements", [])):
        if item.get("status") == "planned" and item.get("facade_status") == "facade_aligned":
            indices.append(index)
    return indices


def split_indices(indices: list[int], workers: int) -> list[list[int]]:
    buckets = [[] for _ in range(max(1, workers))]
    for offset, index in enumerate(indices):
        buckets[offset % len(buckets)].append(index)
    return [bucket for bucket in buckets if bucket]


def detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return 1
    count = sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))
    return max(1, count)


def run_worker(cmd: list[str], log_path: Path, gpu_index: int | None = None) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["NVIDIA_VISIBLE_DEVICES"] = str(gpu_index)
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)


def blender_cmd(args: argparse.Namespace, only: str, store_indices: list[int] | None = None) -> list[str]:
    cmd = [
        args.blender,
        "-b",
        "--python",
        str(args.renderer_script),
        "--",
        "--city",
        args.city,
        "--outdoor-usd",
        str(args.outdoor_usd),
        "--innout-usd",
        str(args.innout_usd),
        "--storefront-plan",
        str(args.storefront_plan),
        "--innout-plan",
        str(args.innout_plan),
        "--output-root",
        str(args.output_root),
        "--only",
        only,
        "--engine",
        args.engine,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--store-camera-distance",
        str(args.store_camera_distance),
    ]
    if store_indices:
        cmd.extend(["--store-indices", ",".join(str(index) for index in store_indices)])
    return cmd


def store_image_indices(store_dir: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for path in store_dir.glob("store_*.png"):
        parts = path.name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            images[int(parts[1])] = path
    return images


def low_detail_store_indices(store_dir: Path, expected_indices: list[int], min_std: float) -> list[int]:
    try:
        import cv2
    except Exception:
        return []
    images = store_image_indices(store_dir)
    bad: list[int] = []
    for index in expected_indices:
        path = images.get(index)
        if path is None:
            bad.append(index)
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            bad.append(index)
            continue
        score = float(image.reshape(-1, 3).std(axis=0).mean())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_density = float(cv2.Canny(gray, 50, 150).mean() / 255.0)
        if score < min_std or edge_density < 0.006:
            bad.append(index)
    return bad


def write_flipped_normal_plan(source_plan: Path, output_root: Path, city: str, indices: list[int]) -> Path:
    plan = read_json(source_plan, {"placements": []})
    for index in indices:
        if index >= len(plan.get("placements", [])):
            continue
        item = plan["placements"][index]
        normal = item.get("outward_normal_xy")
        if normal and len(normal) >= 2:
            item["outward_normal_xy"] = [-float(normal[0]), -float(normal[1])]
    plan_dir = output_root / city / "_photo_retry_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    path = plan_dir / f"{city}_flipped_{os.getpid()}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def clear_pngs(path: Path) -> None:
    if path.exists():
        for image_path in path.glob("*.png"):
            image_path.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--outdoor-usd", type=Path, required=True)
    parser.add_argument("--innout-usd", type=Path, required=True)
    parser.add_argument("--storefront-plan", type=Path, required=True)
    parser.add_argument("--innout-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--renderer-script", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--engine", choices=["workbench", "eevee"], default="eevee")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--store-camera-distance", type=float, default=6.0)
    parser.add_argument("--retry-store-camera-distances", default="18,28")
    parser.add_argument("--min-store-image-std", type=float, default=8.0)
    parser.add_argument("--store-indices", default="")
    parser.add_argument("--max-stores", type=int, default=0)
    parser.add_argument("--skip-stores", action="store_true")
    parser.add_argument("--skip-innout", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    log_dir = args.log_dir or (args.output_root / args.city / "_photo_logs")
    gpu_count = detect_gpu_count()
    workers = max(1, args.workers)
    if args.workers <= 0:
        workers = gpu_count

    store_dir = args.output_root / args.city / "store"
    innout_dir = args.output_root / args.city / "innout"
    store_count = 0
    if not args.skip_stores:
        plan = read_json(args.storefront_plan, {"placements": []})
        indices = parse_index_set(args.store_indices) if args.store_indices else facade_indices(plan)
        if args.max_stores > 0:
            indices = indices[: args.max_stores]
        clear_pngs(store_dir)
        processes: list[tuple[subprocess.Popen, Path]] = []
        for worker_index, bucket in enumerate(split_indices(indices, min(workers, max(1, len(indices))))):
            cmd = blender_cmd(args, "stores", bucket)
            log_path = log_dir / f"{args.city}_stores_worker_{worker_index:02d}.log"
            gpu_index = worker_index % gpu_count
            processes.append((run_worker(cmd, log_path, gpu_index), log_path))
        for proc, log_path in processes:
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"store photo worker failed with code {code}; see {log_path}")
        store_count = len(list(store_dir.glob("*.png"))) if store_dir.exists() else 0
        if store_count != len(indices):
            raise RuntimeError(f"store photo workers wrote {store_count}/{len(indices)} images; see {log_dir}")
        bad = low_detail_store_indices(store_dir, indices, args.min_store_image_std)
        retry_specs: list[tuple[float, bool]] = [(args.store_camera_distance, True)]
        retry_specs.extend((float(value), False) for value in args.retry_store_camera_distances.split(",") if value.strip())
        retry_specs.extend((float(value), True) for value in args.retry_store_camera_distances.split(",") if value.strip())
        original_plan = args.storefront_plan
        for retry_distance, flip_normal in retry_specs:
            if not bad:
                break
            original_distance = args.store_camera_distance
            args.store_camera_distance = retry_distance
            args.storefront_plan = write_flipped_normal_plan(original_plan, args.output_root, args.city, bad) if flip_normal else original_plan
            retry_processes: list[tuple[subprocess.Popen, Path]] = []
            for worker_index, bucket in enumerate(split_indices(bad, min(workers, max(1, len(bad))))):
                cmd = blender_cmd(args, "stores", bucket)
                suffix = f"{retry_distance:g}_{'flip' if flip_normal else 'orig'}"
                log_path = log_dir / f"{args.city}_stores_retry_{suffix}_{worker_index:02d}.log"
                gpu_index = worker_index % gpu_count
                retry_processes.append((run_worker(cmd, log_path, gpu_index), log_path))
            for proc, log_path in retry_processes:
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(f"store photo retry worker failed with code {code}; see {log_path}")
            args.store_camera_distance = original_distance
            args.storefront_plan = original_plan
            bad = low_detail_store_indices(store_dir, indices, args.min_store_image_std)
        if bad:
            raise RuntimeError(f"store photo quality check failed for indices {bad}; see {log_dir}")

    innout_count = 0
    if not args.skip_innout:
        if innout_dir.exists():
            shutil.rmtree(innout_dir)
        cmd = blender_cmd(args, "innout")
        log_path = log_dir / f"{args.city}_innout.log"
        proc = run_worker(cmd, log_path, 0 if gpu_count else None)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"innout photo worker failed with code {code}; see {log_path}")
        innout_count = len(list(innout_dir.glob("*.png"))) if innout_dir.exists() else 0

    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "city": args.city,
                "elapsed_seconds": round(elapsed, 3),
                "engine": args.engine,
                "gpu_count": gpu_count,
                "height": args.height,
                "innout_photos": innout_count,
                "store_photos": store_count,
                "workers": workers,
                "width": args.width,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
