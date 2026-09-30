#!/usr/bin/env python3
"""Render raw VC terrain GLBs for quick visual QA."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


@contextlib.contextmanager
def suppress_native_output(enabled: bool):
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render(width: int, height: int) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 15
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 1
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.65, 0.65, 0.65)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 1000.0))
    sun = bpy.context.object
    sun.name = "GroundQASun"
    sun.data.energy = 3.0
    sun.data.use_shadow = False

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 1000.0))
    area = bpy.context.object
    area.name = "GroundQAFill"
    area.data.energy = 2000.0
    area.data.size = 2000.0
    area.data.use_shadow = False

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.lens = 24
    camera.data.type = "ORTHO"
    camera.data.clip_end = 1_000_000.0
    return camera


def import_scene(path: Path) -> bool:
    if not path.exists():
        return False
    suffix = path.suffix.lower()
    with suppress_native_output(True):
        if suffix in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=str(path))
        elif suffix in {".usd", ".usda", ".usdc", ".usdz"}:
            bpy.ops.wm.usd_import(filepath=str(path))
        else:
            raise ValueError(f"Unsupported terrain scene format: {path}")
    return True


def object_or_parent_name_contains(obj: bpy.types.Object, needles: tuple[str, ...]) -> bool:
    current = obj
    while current:
        name = current.name.lower()
        if any(needle in name for needle in needles):
            return True
        current = current.parent
    return False


def hide_non_terrain_objects() -> None:
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        keep = object_or_parent_name_contains(obj, ("terrain", "ground"))
        obj.hide_viewport = not keep
        obj.hide_render = not keep


def mesh_bounds() -> tuple[Vector, Vector] | None:
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        found = True
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
    if not found:
        return None
    return mins, maxs


def look_at(camera: bpy.types.Object, eye: Vector, target: Vector) -> None:
    direction = target - eye
    camera.location = eye
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    with suppress_native_output(True):
        bpy.ops.render.render(write_still=True)


def render_city(city: str, terrain_path: Path, output_dir: Path, width: int, height: int, terrain_only: bool) -> dict:
    reset_scene()
    camera = setup_render(width, height)
    result = {
        "city": city,
        "terrain": str(terrain_path),
        "status": "missing",
        "renders": {},
    }
    if not import_scene(terrain_path):
        return result
    if terrain_only:
        hide_non_terrain_objects()
    bounds = mesh_bounds()
    if bounds is None:
        result["status"] = "no_mesh"
        return result
    mins, maxs = bounds
    center = (mins + maxs) * 0.5
    dims = maxs - mins
    xy_extent = max(dims.x, dims.y, 1.0)
    z_extent = max(dims.z, 1.0)
    result["status"] = "rendered"
    result["bounds_min"] = [mins.x, mins.y, mins.z]
    result["bounds_max"] = [maxs.x, maxs.y, maxs.z]
    result["dimensions"] = [dims.x, dims.y, dims.z]

    top_path = output_dir / city / f"{city}_terrain_top.png"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = xy_extent * 1.08
    eye = Vector((center.x, center.y, maxs.z + xy_extent * 1.4 + z_extent))
    look_at(camera, eye, center)
    render(top_path)
    result["renders"]["top"] = str(top_path)

    oblique_path = output_dir / city / f"{city}_terrain_oblique.png"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = xy_extent * 0.85
    eye = Vector((center.x - xy_extent * 0.55, center.y - xy_extent * 0.85, maxs.z + xy_extent * 0.65 + z_extent))
    look_at(camera, eye, center)
    render(oblique_path)
    result["renders"]["oblique"] = str(oblique_path)
    return result


def discover_cities(data_root: Path) -> list[str]:
    return sorted(path.name for path in data_root.iterdir() if (path / "terrain.glb").exists())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument(
        "--input-files",
        nargs="*",
        default=None,
        help="Optional CITY=/path/to/terrain.glb-or-scene.usd entries. Overrides --data-root/--cities discovery.",
    )
    parser.add_argument("--terrain-only", action="store_true", help="Hide non terrain/ground meshes after importing scene files.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--manifest", type=Path, default=None)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.input_files:
        city_paths = []
        for item in args.input_files:
            if "=" not in item:
                raise ValueError(f"--input-files entries must be CITY=PATH, got {item!r}")
            city, path = item.split("=", 1)
            city_paths.append((city, Path(path)))
    else:
        cities = args.cities or discover_cities(args.data_root)
        city_paths = [(city, args.data_root / city / "terrain.glb") for city in cities]
    results = []
    for index, (city, terrain_path) in enumerate(city_paths, 1):
        print(f"[{index}/{len(city_paths)}] {city}", flush=True)
        try:
            result = render_city(city, terrain_path, args.output_dir, args.width, args.height, args.terrain_only)
        except Exception as exc:
            result = {
                "city": city,
                "terrain": str(terrain_path),
                "status": "error",
                "error": str(exc),
                "renders": {},
            }
        results.append(result)
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
