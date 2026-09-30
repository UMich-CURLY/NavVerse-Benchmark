from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_ROOT = Path(__file__).resolve().parent / "presets"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def preset_path(name_or_path: str | Path) -> Path:
    raw = Path(name_or_path).expanduser()
    if raw.suffix in {".yaml", ".yml"} or raw.parent != Path("."):
        path = raw if raw.is_absolute() else REPO_ROOT / raw
    else:
        path = PRESET_ROOT / f"{raw.name}.yaml"
    if not path.is_file():
        available = ", ".join(item.stem for item in sorted(PRESET_ROOT.glob("*.yaml")))
        raise FileNotFoundError(f"Unknown benchmark preset {name_or_path!r}; available: {available}")
    return path.resolve()


def load_preset(name_or_path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = preset_path(name_or_path)
    if path in _stack:
        chain = " -> ".join(item.stem for item in (*_stack, path))
        raise ValueError(f"Cyclic benchmark preset inheritance: {chain}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark preset must contain a YAML mapping: {path}")
    parent = payload.pop("extends", None)
    if parent:
        parent_path = path.parent / str(parent)
        if parent_path.suffix not in {".yaml", ".yml"}:
            parent_path = parent_path.with_suffix(".yaml")
        payload = _merge(load_preset(parent_path, (*_stack, path)), payload)
    payload["name"] = path.stem
    payload["path"] = str(path)
    arguments = payload.setdefault("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Preset arguments must be a mapping: {path}")
    for argument, env_name in payload.get("argument_env", {}).items():
        if os.environ.get(str(env_name)):
            arguments[str(argument)] = os.environ[str(env_name)]
    return payload


def list_presets() -> list[dict[str, Any]]:
    return [load_preset(path) for path in sorted(PRESET_ROOT.glob("*.yaml"))]


def _expand(value: str) -> str:
    value = value.replace("${REPO_ROOT}", str(REPO_ROOT))
    return os.path.expandvars(os.path.expanduser(value))


def apply_preset_environment(preset: dict[str, Any]) -> None:
    for key, value in preset.get("environment_defaults", {}).items():
        os.environ.setdefault(str(key), _expand(str(value)))
    for key, value in preset.get("environment", {}).items():
        os.environ[str(key)] = _expand(str(value))


def arguments_to_argv(arguments: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in arguments.items():
        option = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            argv.append(option if value else f"--no-{key}")
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            argv.extend([option, str(item)])
    return argv


def resolve_preset_argv(argv: list[str]) -> tuple[list[str], dict[str, Any] | None]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset")
    source.add_argument("--config")
    known, remaining = parser.parse_known_args(argv)
    source_value = known.config or known.preset
    if not source_value:
        return argv, None

    preset = load_preset(source_value)
    apply_preset_environment(preset)
    preset_argv = arguments_to_argv(preset["arguments"])
    if "--exp_name" not in remaining and "exp_name" not in preset["arguments"]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        preset_argv.extend(["--exp_name", f"{preset['name']}_{stamp}"])
    if "--log_file" not in remaining and "log_file" not in preset["arguments"]:
        source_argv = remaining if "--exp_name" in remaining else preset_argv
        exp_name = source_argv[source_argv.index("--exp_name") + 1]
        preset_argv.extend(["--log_file", f"dump/{exp_name}/agent.log"])
    return [*preset_argv, *remaining], preset
