from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from .config import REPO_ROOT


@dataclass
class TaskSpec:
    task: str
    run_name: str
    command: list[str]
    cwd: Path
    conda_env: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return self.task

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "run_name": self.run_name,
            "command": self.command,
            "cwd": str(self.cwd),
            "conda_env": self.conda_env,
            "env": self.env,
            "inputs": self.inputs,
            "notes": self.notes,
        }


def _run_name(config: dict[str, Any], prefix: str) -> str:
    configured = config["execution"].get("run_name")
    if configured:
        return str(configured)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def _setup(config: dict[str, Any]) -> TaskSpec:
    env_name = str(config["setup"]["env_name"])
    return TaskSpec(
        task="setup",
        run_name=f"setup-{env_name}",
        command=["bash", "scripts/setup_navverse.sh", env_name, *config["extra_args"]],
        cwd=REPO_ROOT,
        notes=[f"conda environment: {env_name}"],
    )


def _download(config: dict[str, Any]) -> TaskSpec:
    download = config["download"]
    command = ["python", "scripts/download_navverse_data.py"]
    if download.get("raw"):
        command.append("--raw")
    for option in (
        "repo_id",
        "archive",
        "output_dir",
        "download_dir",
        "env_file",
        "archive_path",
        "checksum_path",
        "max_workers",
    ):
        if download.get(option):
            command.extend([f"--{option.replace('_', '-')}", str(download[option])])
    for flag in (
        "no_hf_transfer",
        "verify_only",
        "overwrite",
        "keep_archive",
        "skip_episodes",
        "no_update_env",
    ):
        if download.get(flag):
            command.append(f"--{flag.replace('_', '-')}")
    for option in ("include", "ignore"):
        if download.get(option):
            command.append(f"--{option}")
            command.extend(str(item) for item in download[option])
    command.extend(config["extra_args"])
    mode = "raw snapshot" if download.get("raw") else "runnable release"
    return TaskSpec(
        task="download",
        run_name="download-navverse-data",
        command=command,
        cwd=REPO_ROOT,
        conda_env=config["download"].get("conda_env") or "navverse",
        notes=[f"dataset mode: {mode}"],
    )


def _build(config: dict[str, Any]) -> TaskSpec:
    image = str(config["execution"]["image"])
    return TaskSpec(
        task="build",
        run_name="build-navverse-image",
        command=["bash", "docker/build_docker.sh", image, *config["extra_args"]],
        cwd=REPO_ROOT,
        notes=[f"Docker image: {image}"],
    )


def _benchmark(config: dict[str, Any]) -> TaskSpec:
    from baselines.preset_config import load_preset
    from .registry import AGENT_REGISTRY

    benchmark = config["benchmark"]
    source = benchmark.get("config_path") or benchmark.get("preset")
    preset = load_preset(str(source))
    run_name = _run_name(config, preset["name"])
    command = ["python", "-u", "baselines/eval_navverse.py"]
    if benchmark.get("config_path"):
        command.extend(["--config", str(benchmark["config_path"])])
    else:
        command.extend(["--preset", preset["name"]])
    command.extend(["--exp_name", run_name])
    preset_arguments = preset["arguments"]
    episode_folder = (
        benchmark.get("episode_folder")
        or preset_arguments.get("episode_folder")
        or config["inputs"]["episode_folder"]
    )
    scene_folder = (
        benchmark.get("scene_folder")
        or preset_arguments.get("scene_folder")
        or config["inputs"]["scene_folder"]
    )
    overrides = {
        "episode_folder": episode_folder,
        "scene_folder": scene_folder,
        "episode_labels": benchmark.get("split_set"),
        "agent": benchmark.get("agent"),
        "gpu_ids": benchmark.get("gpu_ids"),
        "num_gpu": benchmark.get("num_gpus"),
        "dump_location": config["execution"].get("output_dir"),
        "log_file": str(
            Path(str(config["execution"].get("output_dir") or "dump"))
            / run_name
            / "agent.log"
        ),
    }
    for key, value in overrides.items():
        if value is not None:
            command.extend([f"--{key}", str(value)])
    command.extend(config["extra_args"])
    launcher = preset.get("launcher", {})
    conda_env = (
        benchmark.get("conda_env")
        or os.environ.get(str(launcher.get("conda_env_variable", "")))
        or launcher.get("conda_env")
        or "navverse"
    )
    config["inputs"]["episode_folder"] = str(episode_folder)
    config["inputs"]["scene_folder"] = str(scene_folder)
    notes = [f"preset={preset['name']}: {preset.get('description', '')}"]
    agent_name = str(benchmark.get("agent") or preset_arguments.get("agent") or "")
    agent_spec = AGENT_REGISTRY.get(agent_name)
    if agent_spec and agent_spec.validation_warning:
        notes.append(f"WARNING: {agent_spec.validation_warning}")
    return TaskSpec(
        task="benchmark",
        run_name=run_name,
        command=command,
        cwd=REPO_ROOT,
        conda_env=str(conda_env),
        inputs={
            "episodes": str(episode_folder),
            "scenes": str(scene_folder),
        },
        notes=notes,
    )


def build_task(config: dict[str, Any]) -> TaskSpec:
    task = str(config["task"])
    if task != "benchmark" and config["execution"]["backend"] != "native":
        raise ValueError(f"{task} only supports the native execution backend")
    builders = {
        "setup": _setup,
        "download": _download,
        "build": _build,
        "benchmark": _benchmark,
    }
    try:
        return builders[task](config)
    except KeyError as exc:
        raise ValueError(f"Unknown launcher task: {task}") from exc
