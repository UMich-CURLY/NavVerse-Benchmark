from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_SCENE_TYPE_CONFIGS = {
    "grHome": "grh.yaml",
    "grCommercial": "grc.yaml",
    "vcp": "vcp.yaml",
    "innout": "innout.yaml",
}


def deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_path(path, *, base_dir: Path | None = None) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path.resolve()


def load_config(path):
    path = resolve_path(path, base_dir=Path.cwd())
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        config = compose(config_name=path.stem)
    OmegaConf.set_struct(config, False)
    loaded = OmegaConf.to_container(config, resolve=True)
    loaded["_config_path"] = str(path)
    return loaded


def scene_type_from_scene_id(scene_id):
    return str(scene_id).split("_", 1)[0]


def config_path_for_scene_type(scene_type):
    filename = DEFAULT_SCENE_TYPE_CONFIGS.get(scene_type)
    if filename is None:
        filename = f"{scene_type}.yaml"
    return DEFAULT_CONFIG_DIR / filename


def resolve_task_config_path(path=None, *, scene_id=None, scene_type=None):
    if path not in (None, "", "auto"):
        path = Path(path)
        if path.is_absolute() or (Path.cwd() / path).exists():
            return resolve_path(path, base_dir=Path.cwd())
        return resolve_path(path, base_dir=REPO_ROOT)
    if scene_type is None and scene_id not in (None, "", "none"):
        scene_type = scene_type_from_scene_id(scene_id)
    if scene_type is None:
        scene_type = "vcp"
    return config_path_for_scene_type(scene_type).resolve()


def list_scene_ids(config):
    scene_type = config.get("scene_type")
    scenes = ((config.get("scene") or {}).get("scenes") or {})
    return [f"{scene_type}_{scene_name}" for scene_name in scenes]


def parse_scene_id(scene_id):
    scene_type = scene_type_from_scene_id(scene_id)
    scene_name = str(scene_id)[len(scene_type) + 1 :]
    return scene_type, scene_name


SCENE_MODULE_OVERRIDE_KEYS = {
    "task_generation",
    "astar",
    "navmesh",
    "navmesh_preprocess",
    "sim",
    "trajectory_planner",
    "sampling_trajectory_planner",
    "waypoint_follower",
}


def _scene_only_config(config):
    return {
        key: value
        for key, value in (config or {}).items()
        if key not in SCENE_MODULE_OVERRIDE_KEYS
    }


def resolve_scene_spec(config, scene_id):
    scene_type, scene_name = parse_scene_id(scene_id)
    config_scene_type = config.get("scene_type")
    if config_scene_type and scene_type != config_scene_type:
        raise KeyError(f"Scene {scene_id!r} is not in config for scene_type {config_scene_type!r}")
    scene_section = config.get("scene") or {}
    scene_entries = scene_section.get("scenes") or {}
    if scene_name not in scene_entries:
        raise KeyError(f"Scene {scene_id!r} not found in config")

    scene_default = scene_section.get("default") or {}
    scene_entry = scene_entries[scene_name] or {}
    scene_config = deep_merge(_scene_only_config(scene_default), _scene_only_config(scene_entry))
    scene_config["scene_id"] = f"{scene_type}_{scene_name}"
    scene_config["scene_type"] = scene_type
    path_template = scene_config.pop("path_template", None)
    if path_template:
        scene_config["path"] = str(path_template).format(
            scene_name=scene_name,
            scene_name_upper=scene_name.upper(),
            scene_type=scene_type,
        )
    if "ceiling_height" not in scene_config or scene_config["ceiling_height"] is None:
        scene_config["ceiling_height"] = 1.0
    if "navmesh_preset" not in scene_config:
        scene_config["navmesh_preset"] = "default"

    task_generation = deep_merge(config.get("task_generation") or {}, scene_default.get("task_generation") or {})
    task_generation = deep_merge(task_generation, scene_entry.get("task_generation") or {})
    astar = deep_merge(config.get("astar") or {}, scene_entry.get("astar") or {})
    navmesh = deep_merge(config.get("navmesh") or {}, scene_entry.get("navmesh") or {})
    navmesh_preprocess = deep_merge(
        config.get("navmesh_preprocess") or {},
        scene_entry.get("navmesh_preprocess") or {},
    )
    sim = deep_merge(config.get("sim") or {}, scene_entry.get("sim") or {})
    waypoint_follower = deep_merge(
        config.get("waypoint_follower") or {},
        scene_entry.get("waypoint_follower") or {},
    )
    return {
        "scene": scene_config,
        "task_generation": task_generation,
        "astar": astar,
        "navmesh": navmesh,
        "navmesh_preprocess": navmesh_preprocess,
        "sim": sim,
        "waypoint_follower": waypoint_follower,
        "scene_type": scene_type,
        "scene_name": scene_name,
        "config_path": config.get("_config_path"),
    }
