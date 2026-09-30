from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .backends import render_command, run_backend
from .config import BACKENDS, DEFAULT_BENCHMARK_PRESET, TASKS, base_config
from .discovery import discover_splits
from .tasks import build_task


TASK_DESCRIPTIONS = {
    "setup": "create the NavVerse conda environment",
    "download": "download and verify the public dataset",
    "build": "build the warmed NavVerse Docker image",
    "benchmark": "run a benchmark preset",
}


def _list_presets() -> list[dict[str, Any]]:
    from baselines.preset_config import list_presets

    return list_presets()


def _select(message: str, choices: list[str], default: str | None = None) -> str:
    try:
        import questionary

        answer = questionary.select(message, choices=choices, default=default).ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer or default or choices[0]
    except ImportError:
        print(message)
        for index, choice in enumerate(choices, 1):
            marker = " (default)" if choice == default else ""
            print(f"  {index}. {choice}{marker}")
        answer = input("> ").strip()
        return default or choices[0] if not answer else choices[int(answer) - 1]


def _text(message: str, default: str = "") -> str:
    try:
        import questionary

        answer = questionary.text(message, default=default).ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer or default
    except ImportError:
        suffix = f" [{default}]" if default else ""
        return input(f"{message}{suffix}: ").strip() or default


def _available_gpus() -> list[tuple[str, str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [("0", "GPU 0")]

    gpus = []
    for line in result.stdout.splitlines():
        index, separator, name = line.partition(",")
        if separator and index.strip().isdigit():
            gpus.append((index.strip(), name.strip()))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        allowed = {item.strip() for item in visible.split(",")}
        gpus = [gpu for gpu in gpus if gpu[0] in allowed]
    return gpus or [("0", "GPU 0")]


def _checkbox(
    message: str,
    choices: list[tuple[str, str]],
    default: list[str],
) -> list[str]:
    try:
        import questionary
        from questionary.prompts import common as questionary_common

        previous_indicators = (
            questionary_common.INDICATOR_SELECTED,
            questionary_common.INDICATOR_UNSELECTED,
        )
        questionary_common.INDICATOR_SELECTED = "[✓]"
        questionary_common.INDICATOR_UNSELECTED = "[ ]"
        try:
            answer = questionary.checkbox(
                message,
                choices=[
                    questionary.Choice(
                        title=f"{value} — {label}",
                        value=value,
                        checked=value in default,
                    )
                    for value, label in choices
                ],
                instruction="(Space toggles, Enter confirms)",
            ).ask()
        finally:
            (
                questionary_common.INDICATOR_SELECTED,
                questionary_common.INDICATOR_UNSELECTED,
            ) = previous_indicators
        if answer is None:
            raise KeyboardInterrupt
        return list(answer)
    except ImportError:
        print(f"{message} (comma-separated ids; blank keeps all)")
        for value, label in choices:
            print(f"  [✓] {value} — {label}")
        answer = input("> ").strip()
        if not answer:
            return default
        selected = [item.strip() for item in answer.split(",") if item.strip()]
        valid = {value for value, _ in choices}
        unknown = sorted(set(selected) - valid)
        if unknown:
            raise ValueError(f"Unknown GPU ids: {', '.join(unknown)}")
        return selected


def _confirm(message: str, default: bool = True) -> bool:
    try:
        import questionary

        answer = questionary.confirm(message, default=default).ask()
        if answer is None:
            raise KeyboardInterrupt
        return bool(answer)
    except ImportError:
        suffix = "[Y/n]" if default else "[y/N]"
        answer = input(f"{message} {suffix} ").strip().lower()
        return default if not answer else answer in {"y", "yes"}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="validate and print without starting")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navverse-run",
        description="Set up, download, build, or benchmark NavVerse.",
        allow_abbrev=False,
    )
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    subparsers = parser.add_subparsers(dest="task")

    setup_parser = subparsers.add_parser("setup", help=TASK_DESCRIPTIONS["setup"])
    setup_parser.add_argument("--env-name", default="navverse")
    _add_common(setup_parser)

    download_parser = subparsers.add_parser("download", help=TASK_DESCRIPTIONS["download"])
    download_parser.add_argument("--raw", action="store_true")
    download_parser.add_argument("--repo-id")
    download_parser.add_argument("--archive")
    download_parser.add_argument("--output-dir")
    download_parser.add_argument("--download-dir")
    download_parser.add_argument("--env-file")
    download_parser.add_argument("--archive-path")
    download_parser.add_argument("--checksum-path")
    download_parser.add_argument("--include", nargs="*")
    download_parser.add_argument("--ignore", nargs="*")
    download_parser.add_argument("--max-workers", type=int)
    download_parser.add_argument("--no-hf-transfer", action="store_true")
    download_parser.add_argument("--verify-only", action="store_true")
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.add_argument("--keep-archive", action="store_true")
    download_parser.add_argument("--skip-episodes", action="store_true")
    download_parser.add_argument("--no-update-env", action="store_true")
    download_parser.add_argument("--conda-env", default="navverse")
    _add_common(download_parser)

    build_parser = subparsers.add_parser("build", help=TASK_DESCRIPTIONS["build"])
    build_parser.add_argument("--image", default="umich-curly/navverse:latest")
    _add_common(build_parser)

    benchmark_parser = subparsers.add_parser("benchmark", help=TASK_DESCRIPTIONS["benchmark"])
    source = benchmark_parser.add_mutually_exclusive_group()
    source.add_argument("--preset")
    source.add_argument("--config", dest="config_path")
    benchmark_parser.add_argument("--backend", choices=BACKENDS, default="native")
    benchmark_parser.add_argument("--run-name")
    benchmark_parser.add_argument("--episode-folder")
    benchmark_parser.add_argument("--scene-folder")
    benchmark_parser.add_argument("--split-set")
    benchmark_parser.add_argument("--agent")
    benchmark_parser.add_argument("--gpu-ids")
    benchmark_parser.add_argument("--num-gpus", type=int)
    benchmark_parser.add_argument("--conda-env")
    benchmark_parser.add_argument("--image", default="navverse-benchmark:local")
    benchmark_parser.add_argument("--scenes-s3-uri")
    benchmark_parser.add_argument("--uninavid-s3-uri")
    benchmark_parser.add_argument("--sagemaker-role-arn")
    benchmark_parser.add_argument("--sagemaker-bucket")
    benchmark_parser.add_argument("--aws-profile")
    benchmark_parser.add_argument("--aws-region")
    _add_common(benchmark_parser)
    return parser


def _interactive_argv() -> list[str]:
    task_labels = [f"{task} — {TASK_DESCRIPTIONS[task]}" for task in TASKS]
    task = _select("Task", task_labels, task_labels[-1]).split(" — ", 1)[0]
    if task == "setup":
        return [task, "--env-name", _text("Conda environment name", "navverse")]
    if task == "download":
        mode = _select("Dataset", ["release — verified runnable data", "raw — raw repository snapshot"], "release — verified runnable data")
        argv = [task]
        if mode.startswith("raw"):
            argv.append("--raw")
        output_dir = _text("Output directory", "navverse_data")
        return [*argv, "--output-dir", output_dir]
    if task == "build":
        return [task, "--image", _text("Docker image tag", "umich-curly/navverse:latest")]

    presets = _list_presets()
    labels = [f"{item['name']} — {item.get('description', '')}" for item in presets]
    default = next((label for label in labels if label.startswith(DEFAULT_BENCHMARK_PRESET)), labels[0])
    preset = _select("Benchmark preset", labels, default).split(" — ", 1)[0]
    preset_config = next(item for item in presets if item["name"] == preset)
    preset_arguments = preset_config.get("arguments", {})
    backend = _select("Execution backend", list(BACKENDS), "native")
    defaults = base_config("benchmark")["inputs"]
    episode_folder = _text(
        "Episode folder",
        str(preset_arguments.get("episode_folder") or defaults["episode_folder"]),
    )
    argv = [task, "--preset", preset, "--backend", backend, "--episode-folder", episode_folder]
    folder = Path(episode_folder).expanduser()
    if folder.is_dir():
        splits = discover_splits(folder)
        if splits:
            split_labels = [
                f"{item.name} — {len(item.labels)} episodes / {item.scene_count} scenes / "
                f"{'ready' if item.valid else 'invalid'}"
                for item in splits
            ]
            preset_split = str(preset_arguments.get("episode_labels") or "")
            split_default = next(
                (label for label in split_labels if label.startswith(f"{preset_split} —")),
                split_labels[0],
            )
            split = _select("Episode split", split_labels, split_default).split(" — ", 1)[0]
            argv.extend(["--split-set", split])
    gpus = _available_gpus()
    gpu_ids = _checkbox("Visible GPUs", gpus, [index for index, _ in gpus])
    if not gpu_ids:
        raise ValueError("Select at least one GPU")
    argv.extend(["--gpu-ids", ",".join(gpu_ids)])
    run_name = _text("Run name (leave blank for timestamped default)")
    if run_name:
        argv.extend(["--run-name", run_name])
    if backend == "docker":
        argv.extend(["--image", _text("Docker image", "navverse-benchmark:local")])
    return argv


def _config_from_args(args: argparse.Namespace, extra_args: list[str]) -> dict[str, Any]:
    config = base_config(args.task)
    config["extra_args"] = extra_args
    if args.task == "setup":
        config["setup"] = {"env_name": args.env_name}
    elif args.task == "download":
        config["download"] = {
            key: getattr(args, key)
            for key in (
                "raw", "repo_id", "archive", "output_dir", "download_dir", "env_file",
                "archive_path", "checksum_path", "verify_only", "overwrite", "keep_archive",
                "include", "ignore", "max_workers", "no_hf_transfer", "skip_episodes",
                "no_update_env", "conda_env",
            )
        }
    elif args.task == "build":
        config["execution"]["image"] = args.image
    else:
        config["execution"].update(
            backend=args.backend,
            run_name=args.run_name,
            image=args.image,
        )
        config["benchmark"] = {
            "preset": args.preset or (None if args.config_path else DEFAULT_BENCHMARK_PRESET),
            "config_path": args.config_path,
            "split_set": args.split_set,
            "agent": args.agent,
            "conda_env": args.conda_env,
            "episode_folder": args.episode_folder,
            "scene_folder": args.scene_folder,
            "gpu_ids": args.gpu_ids,
            "num_gpus": args.num_gpus,
        }
        if args.episode_folder:
            config["inputs"]["episode_folder"] = args.episode_folder
        if args.scene_folder:
            config["inputs"]["scene_folder"] = args.scene_folder
        config["inputs"]["scenes_s3_uri"] = args.scenes_s3_uri
        config["inputs"]["uninavid_s3_uri"] = args.uninavid_s3_uri
        if args.gpu_ids:
            config["resources"]["gpu_ids"] = args.gpu_ids
            if args.num_gpus is None:
                config["resources"]["num_gpus"] = len(
                    [item for item in args.gpu_ids.split(",") if item.strip()]
                )
                config["benchmark"]["num_gpus"] = config["resources"]["num_gpus"]
        if args.num_gpus is not None:
            config["resources"]["num_gpus"] = args.num_gpus
        config["sagemaker"].update(
            role_arn=args.sagemaker_role_arn or config["sagemaker"]["role_arn"],
            bucket=args.sagemaker_bucket or config["sagemaker"]["bucket"],
            profile=args.aws_profile or config["sagemaker"]["profile"],
            region=args.aws_region or config["sagemaker"]["region"],
            image_uri=args.image,
        )
    return config


def _show_summary(config: dict[str, Any], task: Any, result: dict[str, Any] | None = None) -> None:
    payload = {"config": config, "task": task.as_dict()}
    if result is not None:
        payload["launch"] = result
    print(json.dumps(payload, indent=2, default=str))


def _handle_listing(args: argparse.Namespace) -> int | None:
    if args.list_tasks:
        print("\n".join(f"{task}: {TASK_DESCRIPTIONS[task]}" for task in TASKS))
        return 0
    if args.list_presets:
        print(
            "\n".join(
                f"{item['name']}: {item.get('description', '')}"
                for item in _list_presets()
            )
        )
        return 0
    return None


def _resolve_invocation(
    parser: argparse.ArgumentParser, raw_argv: list[str]
) -> tuple[argparse.Namespace, list[str], bool] | int:
    args, extra_args = parser.parse_known_args(raw_argv)
    listing_result = _handle_listing(args)
    if listing_result is not None:
        return listing_result
    interactive = args.task is None
    if not interactive:
        return args, extra_args, False
    if not sys.stdin.isatty():
        parser.error("a task is required when stdin is not a TTY")
    try:
        selected_argv = _interactive_argv()
    except KeyboardInterrupt:
        print("Cancelled.")
        return 130
    selected_args, selected_extra = parser.parse_known_args(selected_argv)
    return selected_args, selected_extra, True


def _print_task_preview(config: dict[str, Any], task: Any) -> None:
    print(f"task={task.task} backend={config['execution']['backend']} run={task.run_name}")
    for note in task.notes:
        print(f"  {note}")
    print(f"  command: {render_command(task.command, task.env)}")


def _confirm_interactive_task(interactive: bool) -> bool | None:
    if not interactive:
        return True
    try:
        return _confirm("Start this task?")
    except KeyboardInterrupt:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    invocation = _resolve_invocation(parser, raw_argv)
    if isinstance(invocation, int):
        return invocation
    args, extra_args, interactive = invocation

    config = _config_from_args(args, extra_args)
    task = build_task(config)
    result = run_backend(task, config, dry_run=True)
    if args.dry_run:
        _show_summary(config, task, result)
        return 0

    _print_task_preview(config, task)
    confirmation = _confirm_interactive_task(interactive)
    if confirmation is not True:
        print("Cancelled.")
        return 130 if confirmation is None else 1
    result = run_backend(task, config, dry_run=False)
    if args.json:
        _show_summary(config, task, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
