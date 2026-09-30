#!/usr/bin/env python3
"""Render storefront and innout QA photos from generated VC USD scenes."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_usd_child_names(path: Path, parent_paths: list[str]) -> set[str]:
    try:
        from pxr import Usd
    except Exception:
        return set()
    stage = Usd.Stage.Open(str(path))
    if not stage:
        return set()
    names: set[str] = set()
    for parent_path in parent_paths:
        parent = stage.GetPrimAtPath(parent_path)
        if parent:
            names.update(child.GetName() for child in parent.GetChildren())
    return names


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return name or "item"


def parse_index_set(value: str) -> set[int]:
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.add(int(part))
    return indices


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


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


def import_usd(path: Path) -> None:
    with suppress_native_output(True):
        bpy.ops.wm.usd_import(filepath=str(path))


def setup_render(width: int, height: int, engine: str) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.35
    scene.view_settings.gamma = 1.0
    if scene.world:
        scene.world.color = (0.45, 0.45, 0.45)
    if engine in {"opengl", "workbench"}:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
    else:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
        scene.eevee.use_gtao = True
    if engine not in {"opengl", "workbench"}:
        scene.render.engine = "BLENDER_EEVEE"
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 20.0))
    bpy.context.object.name = "DefaultSun"
    bpy.context.object.data.energy = 3.0
    bpy.context.object.data.use_shadow = False
    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 180.0))
    bpy.context.object.name = "GlobalFillArea"
    bpy.context.object.data.energy = 2500.0
    bpy.context.object.data.size = 1000.0
    bpy.context.object.data.use_shadow = False
    bpy.ops.object.light_add(type="POINT", location=(0.0, -6.0, 2.0))
    bpy.context.object.name = "CameraFillLight"
    bpy.context.object.data.energy = 250.0
    bpy.context.object.data.shadow_soft_size = 8.0
    bpy.context.object.data.use_shadow = False
    bpy.ops.object.camera_add(location=(0.0, -6.0, 2.0))
    camera = bpy.context.object
    scene.camera = camera
    camera.data.lens = 24
    return camera


def look_at(camera: bpy.types.Object, eye: Vector, target: Vector) -> None:
    direction = target - eye
    camera.location = eye
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    fill = bpy.data.objects.get("CameraFillLight")
    if fill:
        fill.location = eye


def render(path: Path, engine: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    if engine == "opengl":
        raise RuntimeError("OpenGL viewport renders require a foreground Blender context; use --engine eevee in background mode.")
    else:
        with suppress_native_output(True):
            bpy.ops.render.render(write_still=True)


def hide_objects_named(names: set[str]) -> None:
    for obj in bpy.data.objects:
        current = obj
        matched = False
        while current:
            if current.name in names or any(current.name.startswith(f"{name}.") for name in names):
                matched = True
                break
            current = current.parent
        if matched:
            obj.hide_render = True
            obj.hide_viewport = True


def hide_root_objects_named(names: set[str]) -> None:
    if not names:
        return
    for obj in bpy.data.objects:
        root = obj
        while root.parent:
            root = root.parent
        if root.name in names or any(root.name.startswith(f"{name}.") for name in names):
            obj.hide_render = True
            obj.hide_viewport = True


def disable_existing_lights() -> None:
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True
            obj.hide_viewport = True


def clear_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for image_path in path.glob("*.png"):
        image_path.unlink()


def render_stores(args: argparse.Namespace) -> None:
    plan = read_json(args.storefront_plan, {"placements": []})
    if not args.outdoor_usd.exists():
        return
    hidden_roots = read_usd_child_names(args.outdoor_usd, ["/World/Objaverse", "/World/RoadObjects"])
    reset_scene()
    import_usd(args.outdoor_usd)
    hide_objects_named({"Objaverse", "RoadObjects"})
    hide_root_objects_named(hidden_roots)
    disable_existing_lights()
    camera = setup_render(args.width, args.height, args.engine)
    out_dir = args.output_root / args.city / "store"
    selected_indices = parse_index_set(args.store_indices) if args.store_indices else set()
    if not selected_indices:
        clear_pngs(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for index, item in enumerate(plan.get("placements", [])):
        if selected_indices and index not in selected_indices:
            continue
        if item.get("status") != "planned" or item.get("facade_status") != "facade_aligned":
            continue
        loc = item.get("final_location")
        outward = item.get("outward_normal_xy")
        if not loc or not outward:
            continue
        ox, oy = float(outward[0]), float(outward[1])
        ground = float(loc[2])
        target = Vector((float(loc[0]), float(loc[1]), ground + 1.4))
        eye = Vector((float(loc[0]) + ox * args.store_camera_distance, float(loc[1]) + oy * args.store_camera_distance, ground + 1.7))
        look_at(camera, eye, target)
        amenity = safe_name(str(item.get("amenity") or item.get("asset_type") or "store"))
        render(out_dir / f"store_{index:04d}_{amenity}.png", args.engine)
        rendered += 1
        if args.max_stores and rendered >= args.max_stores:
            break


def render_innout(args: argparse.Namespace) -> None:
    plan = read_json(args.innout_plan, {})
    if not args.innout_usd.exists() or not plan:
        return
    hidden_roots = read_usd_child_names(args.innout_usd, ["/World/Objaverse", "/World/RoadObjects"])
    reset_scene()
    import_usd(args.innout_usd)
    hide_objects_named({"Objaverse", "RoadObjects"})
    hide_root_objects_named(hidden_roots)
    disable_existing_lights()
    camera = setup_render(args.width, args.height, args.engine)
    out_dir = args.output_root / args.city / "innout"
    center = plan.get("door_center_xy")
    outward = plan.get("outward_normal_xy")
    if not center or not outward:
        return
    clear_pngs(out_dir)
    ox, oy = float(outward[0]), float(outward[1])
    z = float(plan.get("terrain_z", plan.get("door_bottom_z", 0.0)))
    door = Vector((float(center[0]), float(center[1]), z + 1.3))
    outside = Vector((float(center[0]) + ox * 3.5, float(center[1]) + oy * 3.5, z + 1.55))
    inside = Vector((float(center[0]) - ox * 2.5, float(center[1]) - oy * 2.5, z + 1.35))
    look_at(camera, outside, door)
    render(out_dir / "outside_to_inside.png", args.engine)
    look_at(camera, inside, outside)
    render(out_dir / "inside_to_outside.png", args.engine)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--outdoor-usd", type=Path, required=True)
    parser.add_argument("--innout-usd", type=Path, required=True)
    parser.add_argument("--storefront-plan", type=Path, required=True)
    parser.add_argument("--innout-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--store-camera-distance", type=float, default=6.0)
    parser.add_argument("--engine", choices=["opengl", "workbench", "eevee"], default="eevee")
    parser.add_argument("--only", choices=["all", "stores", "innout"], default="all")
    parser.add_argument("--max-stores", type=int, default=0)
    parser.add_argument("--store-indices", default="")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.only in {"all", "stores"}:
        render_stores(args)
    if args.only in {"all", "innout"}:
        render_innout(args)


if __name__ == "__main__":
    main()
