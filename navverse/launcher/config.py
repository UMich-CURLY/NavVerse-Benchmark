from __future__ import annotations

import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("setup", "download", "build", "benchmark")
BACKENDS = ("native", "docker", "sagemaker")
DEFAULT_BENCHMARK_PRESET = "benchmark_uninavid_450"


def _runtime_path(name: str, default: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                return value.strip().strip('"').strip("'")
    return default


def base_config(task: str) -> dict[str, Any]:
    return {
        "task": task,
        "execution": {
            "backend": "native",
            "run_name": None,
            "output_dir": "dump",
            "image": "navverse-benchmark:local",
        },
        "inputs": {
            "episode_folder": _runtime_path("EPISODE_FOLDER", "navverse_data/episodes"),
            "scene_folder": _runtime_path("SCENE_FOLDER", "navverse_data"),
            "scenes_s3_uri": None,
            "uninavid_s3_uri": None,
        },
        "resources": {
            "gpu_ids": "0,1,2,3,4,5,6,7",
            "num_gpus": 8,
            "instance_type": "ml.g7e.48xlarge",
            "instance_count": 1,
            "shm_size": "256g",
        },
        "sagemaker": {
            "profile": os.environ.get("AWS_PROFILE", ""),
            "region": os.environ.get("AWS_REGION", "us-west-2"),
            "role_arn": os.environ.get("SAGEMAKER_ROLE_ARN", ""),
            "bucket": os.environ.get("SAGEMAKER_BUCKET", ""),
            "image_uri": None,
            "max_runtime_seconds": None,
            "wait": True,
        },
    }
