#!/usr/bin/env python3
"""Convert road-object GLB assets to USD with Blender."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def convert_one(source: Path, target: Path) -> None:
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(source))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type in {"CAMERA", "LIGHT"}:
            obj.select_set(True)
    bpy.ops.object.delete()
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(filepath=str(target), selected_objects_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert road-object GLB assets to USD.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    for filename in args.files:
        source = args.input_dir / filename
        target = args.output_dir / (source.stem + ".usd")
        if not source.exists():
            print(f"missing {source}")
            continue
        if target.exists():
            print(f"skip existing {target}")
            continue
        print(f"convert {source} -> {target}")
        convert_one(source, target)


if __name__ == "__main__":
    main()
