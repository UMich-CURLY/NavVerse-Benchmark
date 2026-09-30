from __future__ import annotations

import json
import runpy
from pathlib import Path

import yaml

from baselines.preset_config import list_presets, load_preset, resolve_preset_argv
from navverse.episode import VLNEpisode
from navverse.launcher.backends import docker_command, sagemaker_request
from navverse.launcher.cli import main
from navverse.launcher.config import DEFAULT_BENCHMARK_PRESET, TASKS, base_config
from navverse.launcher.container_entrypoint import _prepare_uninavid
from navverse.launcher.discovery import discover_splits, resolve_split_episode_folder
from navverse.launcher.tasks import build_task


write_release_marker = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "download_navverse_data.py")
)["write_release_marker"]


def benchmark_config(**overrides):
    config = base_config("benchmark")
    config["extra_args"] = overrides.pop("extra_args", [])
    config["benchmark"] = {
        "preset": overrides.pop("preset", DEFAULT_BENCHMARK_PRESET),
        "config_path": overrides.pop("config_path", None),
        "split_set": overrides.pop("split_set", None),
        "agent": overrides.pop("agent", None),
        "conda_env": overrides.pop("conda_env", None),
        "episode_folder": overrides.pop("episode_folder", None),
        "scene_folder": overrides.pop("scene_folder", None),
        "gpu_ids": overrides.pop("gpu_ids", None),
        "num_gpus": overrides.pop("num_gpus", None),
    }
    config["execution"].update(overrides)
    return config


def test_release_marker_records_verified_install(tmp_path):
    marker = tmp_path / "NAVVERSE_DATA_RELEASE_july23.json"
    write_release_marker(
        marker,
        repo_id="example/navverse",
        archive_name="release.tar.gz",
        checksum="a" * 64,
        counts={"outdoor": 52},
    )
    payload = json.loads(marker.read_text())
    assert payload["sha256"] == "a" * 64
    assert payload["counts"] == {"outdoor": 52}


def test_relocates_uninavid_absolute_asset_paths(tmp_path):
    model = tmp_path / "uninavid-7b"
    model.mkdir()
    (tmp_path / "eva_vit_g.pth").write_bytes(b"weights")
    (model / "config.json").write_text(
        json.dumps(
            {
                "mm_vision_tower": "/old/eva_vit_g.pth",
                "image_processor": "/old/processor",
            }
        )
    )
    assert _prepare_uninavid(tmp_path) == model
    config = json.loads((model / "config.json").read_text())
    assert config["mm_vision_tower"] == str(tmp_path / "eva_vit_g.pth")
    assert "/old/" not in config["image_processor"]


def test_discovers_split_metadata_and_duplicates(tmp_path):
    (tmp_path / "scene.json").write_text(
        json.dumps([{"scene_id": "scene", "episode_id": 1, "task_type": "objnav"}])
    )
    (tmp_path / "tiny_set.txt").write_text("scene_1\nscene_1\n")
    split = next(item for item in discover_splits(tmp_path) if item.name == "tiny")
    assert split.valid
    assert split.unique_count == 1
    assert split.duplicate_count == 1
    assert split.scene_count == 1


def test_resolves_split_to_isolated_episode_subfolder(tmp_path):
    root = tmp_path / "episodes"
    legacy = root / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "scene.json").write_text(
        json.dumps([{"scene_id": "legacy_scene", "episode_id": 1}])
    )
    (root / "legacy_set.txt").write_text("legacy_scene_1\n")

    split = next(item for item in discover_splits(root) if item.name == "legacy")
    runtime_folder = resolve_split_episode_folder(root, split)
    labels = {
        episode.episode_label for episode in VLNEpisode.from_json_folder(runtime_folder)
    }
    assert split.valid
    assert split.labels == ["legacy_scene_1"]
    assert runtime_folder == legacy
    assert "legacy_scene_1" in labels


def test_all_presets_parse_and_inheritance_is_applied():
    presets = list_presets()
    expected = {'benchmark_gpt', 'benchmark_poliformer', 'benchmark_internvla_n1', 'benchmark_longnav', 'benchmark_streamvln', 'benchmark_uninavid', 'benchmark_dummy', 'benchmark_uninavid_vln', 'benchmark_oracle_gt', 'benchmark_uninavid_450', 'benchmark_uninavid_placenav', 'benchmark_oracle', 'benchmark_oracle_gt_discrete', 'benchmark_warmup', 'benchmark_smoke'}
    assert {preset["name"] for preset in presets} == expected
    assert len(presets) == len(expected)
    inherited = load_preset("benchmark_uninavid_450")
    assert inherited["arguments"]["agent"] == "UniNaVid"
    assert inherited["arguments"]["episode_labels"] == "benchmark450"
    assert inherited["arguments"]["save_video"] is False


def test_unverified_poliformer_preset_warns_without_disabling_launch():
    task = build_task(benchmark_config(preset="benchmark_poliformer"))
    assert "--preset" in task.command
    assert any("WARNING" in note and "PoliFormer" in note for note in task.notes)


def test_external_preset_inheritance_and_explicit_argument_precedence(tmp_path):
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("arguments:\n  agent: UniNaVid\n  gpu_ids: '0,1'\n")
    child.write_text("extends: parent.yaml\narguments:\n  gpu_ids: '2'\n")
    assert load_preset(child)["arguments"] == {"agent": "UniNaVid", "gpu_ids": "2"}

    argv, _ = resolve_preset_argv(
        ["--preset", "benchmark_uninavid", "--agent", "Dummy"]
    )
    agent_positions = [index for index, value in enumerate(argv) if value == "--agent"]
    assert len(agent_positions) == 2
    assert argv[agent_positions[-1] + 1] == "Dummy"


def test_builds_all_four_native_task_specs():
    setup = base_config("setup")
    setup.update(extra_args=[], setup={"env_name": "navverse"})
    download = base_config("download")
    download.update(
        extra_args=[],
        download={
            "raw": False,
            "repo_id": None,
            "archive": None,
            "output_dir": None,
            "download_dir": None,
            "env_file": None,
            "archive_path": None,
            "checksum_path": None,
            "max_workers": None,
            "include": None,
            "ignore": None,
            "no_hf_transfer": False,
            "verify_only": False,
            "overwrite": False,
            "keep_archive": False,
            "skip_episodes": False,
            "no_update_env": False,
            "conda_env": "navverse",
        },
    )
    build = base_config("build")
    build["extra_args"] = []
    benchmark = benchmark_config(run_name="smoke")

    specs = [build_task(config) for config in (setup, download, build, benchmark)]
    assert [spec.task for spec in specs] == list(TASKS)
    assert specs[0].command[:2] == ["bash", "scripts/setup_navverse.sh"]
    assert specs[1].command[:2] == ["python", "scripts/download_navverse_data.py"]
    assert specs[2].command[:2] == ["bash", "docker/build_docker.sh"]
    assert specs[3].command[:3] == ["python", "-u", "baselines/eval_navverse.py"]


def test_benchmark_native_docker_and_sagemaker_generation(tmp_path):
    config = benchmark_config(
        preset="benchmark_smoke",
        run_name="smoke",
        episode_folder=str(tmp_path / "episodes"),
        scene_folder=str(tmp_path / "scenes"),
        gpu_ids="0,1",
        num_gpus=2,
    )
    task = build_task(config)
    assert task.conda_env == "navverse"
    assert task.command[task.command.index("--scene_folder") + 1] == str(
        tmp_path / "scenes"
    )
    assert task.command[task.command.index("--gpu_ids") + 1] == "0,1"
    assert task.command[task.command.index("--num_gpu") + 1] == "2"

    config["execution"].update(backend="docker", image="navverse:test")
    docker = docker_command(task, config)
    assert docker[:4] == ["docker", "run", "--rm", "--gpus"]
    assert docker[docker.index("--runtime") + 1] == "nvidia"
    assert "NVIDIA_DRIVER_CAPABILITIES=all" in docker
    assert "navverse:test" in docker

    image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/navverse:test"
    config["execution"].update(backend="sagemaker", image=image)
    config["inputs"].update(
        scenes_s3_uri="s3://bucket/scenes/",
        uninavid_s3_uri="s3://bucket/uninavid/",
    )
    config["sagemaker"].update(
        role_arn="arn:aws:iam::123456789012:role/navverse",
        bucket="bucket",
        image_uri=image,
    )
    request = sagemaker_request(task, config, "s3://bucket/config/config.yaml")
    assert request["AlgorithmSpecification"]["ContainerEntrypoint"] == [
        "python",
        "-m",
        "navverse.launcher.container_entrypoint",
    ]
    assert [item["ChannelName"] for item in request["InputDataConfig"]] == [
        "config",
        "scenes",
        "uninavid",
    ]


def test_cli_dry_runs_every_public_task(capsys):
    commands = (
        ["setup", "--dry-run"],
        ["download", "--dry-run"],
        ["build", "--dry-run"],
        ["benchmark", "--preset", "benchmark_smoke", "--dry-run"],
    )
    for argv in commands:
        assert main(argv) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["task"]["task"] == argv[0]
        assert payload["launch"]["backend"] == "native"


def test_launcher_interface_contains_only_supported_tasks(capsys):
    assert main(["--list-tasks"]) == 0
    output = capsys.readouterr().out
    assert [line.split(":", 1)[0] for line in output.splitlines()] == list(TASKS)
    assert "sft:" not in output
    assert "rl:" not in output


def test_default_benchmark_is_canonical_450_without_video():
    assert DEFAULT_BENCHMARK_PRESET == "benchmark_uninavid_450"
    preset = load_preset(DEFAULT_BENCHMARK_PRESET)
    assert preset["arguments"]["episode_labels"] == "benchmark450"
    assert preset["arguments"]["save_video"] is False
    assert preset["arguments"]["wandb"] is False


def test_packaged_yaml_configuration_is_valid():
    for preset in list_presets():
        payload = yaml.safe_load(Path(preset["path"]).read_text())
        assert isinstance(payload, dict)
        assert isinstance(preset["arguments"], dict)
