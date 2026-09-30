#!/usr/bin/env python3

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from navverse.config import list_scene_ids, load_config


BEV_FILE_SUFFIXES = (".npz", "_rgb.png", "_depth.png", "_info.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate VCP BEV maps with isolated Isaac Sim workers.")
    parser.add_argument("--config", default="configs/vcp.yaml")
    parser.add_argument("--scene-folder", default=os.environ.get("SCENE_FOLDER", "navverse_data"))
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cities", default="", help="Comma-separated scene ids or city names.")
    parser.add_argument("--legacy-only", action="store_true", help="Skip maps already written by the raised-camera generator.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-dir", default="")
    return parser.parse_args()


def output_paths(scene_folder, scene_id):
    scene_dir = scene_folder / "episode_data" / scene_id
    return [
        scene_dir / f"{name}{suffix}"
        for name in ("bev_map", "nav_bev_map")
        for suffix in BEV_FILE_SUFFIXES
    ]


def has_raised_camera_metadata(scene_folder, scene_id):
    info_path = scene_folder / "episode_data" / scene_id / "bev_map_info.txt"
    try:
        info = json.loads(info_path.read_text())
        camera_position = info["camera_position"]
        world_bb = info["world_bb"]
        return float(camera_position[2]) > float(world_bb[5])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def normalize_scene_ids(config, requested):
    available = list_scene_ids(config)
    if not requested:
        return available
    by_city = {scene_id.split("_", 1)[1].lower(): scene_id for scene_id in available}
    selected = []
    for value in requested.split(","):
        value = value.strip().lower()
        if not value:
            continue
        scene_id = value if value.startswith("vcp_") else by_city.get(value)
        if scene_id not in available:
            raise ValueError(f"Unknown VCP scene: {value}")
        selected.append(scene_id)
    return selected


def validate_outputs(scene_folder, scene_id):
    missing = [str(path) for path in output_paths(scene_folder, scene_id) if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return f"missing or empty outputs: {missing}"
    if not has_raised_camera_metadata(scene_folder, scene_id):
        return "BEV metadata does not place the camera above the scene"
    return None


def run_scene(repo_root, config_path, scene_folder, scene_id, gpu_id, force, log_dir):
    log_path = log_dir / f"{scene_id}.log"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "navverse_task_generation.py"),
        "--headless",
        "--experience",
        "isaaclab.python.rendering.kit",
        "--scene_folder",
        str(scene_folder),
        "--config",
        str(config_path),
        "--tg_config_path",
        str(config_path),
        "--test_scene_id",
        scene_id,
        "--tg_mode",
        "auto",
        "--bev_only",
    ]
    if force:
        command.append("--force_regenerate_bev")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w") as log_file:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    error = None if result.returncode == 0 else f"worker exit code {result.returncode}"
    if error is None:
        error = validate_outputs(scene_folder, scene_id)
    return {
        "scene_id": scene_id,
        "gpu_id": str(gpu_id),
        "returncode": result.returncode,
        "error": error,
        "log": str(log_path),
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    scene_folder = Path(args.scene_folder).expanduser().resolve()
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    config = load_config(config_path)
    scene_ids = normalize_scene_ids(config, args.cities)
    if args.legacy_only:
        scene_ids = [
            scene_id
            for scene_id in scene_ids
            if not has_raised_camera_metadata(scene_folder, scene_id)
        ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else Path("/tmp") / f"navverse_vcp_bev_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[BEV] scenes={len(scene_ids)}, workers={args.workers}, "
        f"gpus={gpu_ids}, logs={log_dir}"
    )
    if not scene_ids:
        return 0

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {}
        for index, scene_id in enumerate(scene_ids):
            gpu_id = gpu_ids[index % len(gpu_ids)]
            future = executor.submit(
                run_scene,
                repo_root,
                config_path,
                scene_folder,
                scene_id,
                gpu_id,
                args.force,
                log_dir,
            )
            pending[future] = scene_id

        completed = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                scene_id = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "scene_id": scene_id,
                        "gpu_id": "",
                        "returncode": -1,
                        "error": str(exc),
                        "log": "",
                    }
                results.append(result)
                completed += 1
                status = "ok" if result["error"] is None else f"failed: {result['error']}"
                print(f"[BEV] {completed}/{len(scene_ids)} {scene_id}: {status}", flush=True)

    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    failures = [result for result in results if result["error"] is not None]
    print(f"[BEV] complete={len(results) - len(failures)}, failed={len(failures)}, summary={summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
