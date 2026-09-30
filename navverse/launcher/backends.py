from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .tasks import TaskSpec


def render_command(command: list[str], env: dict[str, str] | None = None) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted((env or {}).items()))
    body = shlex.join(command)
    return f"{prefix} {body}".strip()


def _conda_command(task: TaskSpec) -> list[str]:
    conda = shutil.which("conda")
    if task.conda_env and conda and os.environ.get("CONDA_DEFAULT_ENV") != task.conda_env:
        return [conda, "run", "-n", task.conda_env, "--no-capture-output", *task.command]
    return task.command


def run_native(task: TaskSpec, dry_run: bool) -> dict[str, Any]:
    command = _conda_command(task)
    result = {"backend": "native", "cwd": str(task.cwd), "command": command, "env": task.env}
    if not dry_run:
        subprocess.run(command, cwd=task.cwd, env={**os.environ, **task.env}, check=True)
    return result


def docker_command(task: TaskSpec, config: dict[str, Any]) -> list[str]:
    execution = config["execution"]
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--runtime", "nvidia",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
        "--shm-size", str(config["resources"]["shm_size"]),
        "-w", str(task.cwd), "-v", f"{REPO_ROOT}:{REPO_ROOT}",
    ]
    repo_root = REPO_ROOT.resolve()
    for raw_path in sorted(set(task.inputs.values())):
        path = Path(raw_path).expanduser().resolve()
        if path.exists() and path != repo_root and repo_root not in path.parents:
            command.extend(["-v", f"{path}:{path}:ro"])
    for key, value in sorted(task.env.items()):
        command.extend(["-e", f"{key}={value}"])
    command.extend([str(execution["image"]), *task.command])
    return command


def run_docker(task: TaskSpec, config: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    command = docker_command(task, config)
    result = {"backend": "docker", "command": command}
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return result


def _s3_channel(name: str, uri: str) -> dict[str, Any]:
    return {
        "ChannelName": name,
        "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": uri,
            "S3DataDistributionType": "FullyReplicated",
        }},
    }


def sagemaker_request(task: TaskSpec, config: dict[str, Any], config_s3_uri: str) -> dict[str, Any]:
    sm = config["sagemaker"]
    image_uri = sm.get("image_uri") or config["execution"].get("image")
    if not image_uri or ".dkr.ecr." not in image_uri:
        raise ValueError("SageMaker requires sagemaker.image_uri (an ECR image URI)")
    if not sm.get("role_arn") or not sm.get("bucket"):
        raise ValueError("SageMaker requires a role ARN and output bucket")
    inputs = config["inputs"]
    channels = [_s3_channel("config", config_s3_uri)]
    for channel, key in (
        ("scenes", "scenes_s3_uri"), ("uninavid", "uninavid_s3_uri"),
    ):
        if inputs.get(key):
            channels.append(_s3_channel(channel, str(inputs[key])))
    request: dict[str, Any] = {
        "TrainingJobName": task.run_name.replace("_", "-")[:63],
        "RoleArn": sm["role_arn"],
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", "navverse.launcher.container_entrypoint"],
            "ContainerArguments": ["--config", "/opt/ml/input/data/config/config.yaml"],
        },
        "InputDataConfig": channels,
        "OutputDataConfig": {"S3OutputPath": f"s3://{sm['bucket']}/outputs/{task.run_name}"},
        "ResourceConfig": {
            "InstanceType": config["resources"]["instance_type"],
            "InstanceCount": int(config["resources"]["instance_count"]),
            "VolumeSizeInGB": 500,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": int(sm.get("max_runtime_seconds") or 432000)},
        "EnableManagedSpotTraining": False,
    }
    return request


def run_sagemaker(task: TaskSpec, config: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    sm = config["sagemaker"]
    config_s3_uri = f"s3://{sm['bucket']}/launch/{task.run_name}/config.yaml"
    request = sagemaker_request(task, config, config_s3_uri)
    result: dict[str, Any] = {"backend": "sagemaker", "config_s3_uri": config_s3_uri, "request": request}
    if dry_run:
        return result
    try:
        import boto3
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError("Install boto3 and omegaconf to submit SageMaker jobs") from exc
    session = boto3.Session(profile_name=sm.get("profile") or None, region_name=sm["region"])
    payload = OmegaConf.to_yaml(OmegaConf.create(config)).encode()
    bucket, key = config_s3_uri.removeprefix("s3://").split("/", 1)
    session.client("s3").put_object(Bucket=bucket, Key=key, Body=payload)
    client = session.client("sagemaker")
    response = client.create_training_job(**request)
    result["training_job_arn"] = response["TrainingJobArn"]
    if sm.get("wait", True):
        while True:
            status = client.describe_training_job(TrainingJobName=request["TrainingJobName"])
            state = status["TrainingJobStatus"]
            print(json.dumps({"training_job": request["TrainingJobName"], "status": state}), flush=True)
            if state in {"Completed", "Failed", "Stopped"}:
                if state != "Completed":
                    raise RuntimeError(status.get("FailureReason", f"Training job ended in {state}"))
                break
            time.sleep(30)
    return result


def run_backend(task: TaskSpec, config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    backend = config["execution"]["backend"]
    if backend == "native":
        return run_native(task, dry_run)
    if backend == "docker":
        return run_docker(task, config, dry_run)
    if backend == "sagemaker":
        return run_sagemaker(task, config, dry_run)
    raise ValueError(f"Unknown backend {backend!r}")
