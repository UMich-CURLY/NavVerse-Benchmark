#!/usr/bin/env python3
"""Render traffic-light orientation QA views from a generated VC USD."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "traffic_light"


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_usd(path: Path) -> None:
    with suppress_native_output(True):
        bpy.ops.wm.usd_import(filepath=str(path))


def setup_render(width: int, height: int) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.2
    scene.view_settings.gamma = 1.0
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 1
        scene.eevee.use_gtao = True
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.55, 0.55, 0.55)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 80.0))
    sun = bpy.context.object
    sun.name = "TrafficQaSun"
    sun.data.energy = 2.8
    sun.data.use_shadow = False

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 120.0))
    area = bpy.context.object
    area.name = "TrafficQaGlobalFill"
    area.data.energy = 2800.0
    area.data.size = 1000.0
    area.data.use_shadow = False

    bpy.ops.object.light_add(type="POINT", location=(0.0, 0.0, 5.0))
    fill = bpy.context.object
    fill.name = "TrafficQaCameraFill"
    fill.data.energy = 450.0
    fill.data.shadow_soft_size = 8.0
    fill.data.use_shadow = False

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.lens = 32
    camera.data.clip_end = 5000.0
    return camera


def look_at(camera: bpy.types.Object, eye: Vector, target: Vector) -> None:
    direction = target - eye
    camera.location = eye
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    fill = bpy.data.objects.get("TrafficQaCameraFill")
    if fill:
        fill.location = eye


def make_arrow_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("TrafficQaHeadingArrowMat")
    mat.diffuse_color = (1.0, 0.02, 0.02, 1.0)
    return mat


def add_heading_arrow(name: str, location: list[float], heading: float, length: float = 5.0) -> None:
    mat = bpy.data.materials.get("TrafficQaHeadingArrowMat") or make_arrow_material()
    base = Vector((float(location[0]), float(location[1]), float(location[2]) + 0.08))
    direction = Vector((math.cos(heading), math.sin(heading), 0.0))
    end = base + direction * length

    curve = bpy.data.curves.new(f"{name}_heading_line", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = 0.08
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (base.x, base.y, base.z, 1.0)
    spline.points[1].co = (end.x, end.y, end.z, 1.0)
    line = bpy.data.objects.new(f"{name}_heading_line", curve)
    line.data.materials.append(mat)
    bpy.context.collection.objects.link(line)

    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.35, depth=0.9, location=end)
    cone = bpy.context.object
    cone.name = f"{name}_heading_tip"
    cone.data.materials.append(mat)
    cone.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()


def render(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    with suppress_native_output(True):
        bpy.ops.render.render(write_still=True)


def choose_lights(objects: list[dict], max_lights: int) -> list[dict]:
    lights = [obj for obj in objects if obj.get("category") in {"traffic_signals", "crossing"} and obj.get("final_location")]
    chosen: list[dict] = []
    heading_bins: set[int] = set()
    for light in lights:
        heading = float(light.get("heading", 0.0))
        bin_id = int(round((math.degrees(heading) % 360.0) / 45.0)) % 8
        if bin_id in heading_bins:
            continue
        chosen.append(light)
        heading_bins.add(bin_id)
        if len(chosen) >= max_lights:
            return chosen
    for light in lights:
        if light not in chosen:
            chosen.append(light)
        if len(chosen) >= max_lights:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Render traffic light orientation QA screenshots.")
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--road-objects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-lights", type=int, default=6)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=13.0)
    parser.add_argument("--camera-side", type=float, default=3.5)
    parser.add_argument("--driver-view", action="store_true", help="Place the camera on the side that the traffic light faces.")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    data = json.loads(args.road_objects.read_text(encoding="utf-8"))
    lights = choose_lights(data.get("objects", []), args.max_lights)
    reset_scene()
    import_usd(args.usd)
    camera = setup_render(args.width, args.height)

    manifest = []
    for idx, light in enumerate(lights):
        loc = [float(v) for v in light["final_location"]]
        heading = float(light.get("heading", 0.0))
        forward = Vector((math.cos(heading), math.sin(heading), 0.0))
        side = Vector((-forward.y, forward.x, 0.0))
        target = Vector((loc[0], loc[1], loc[2] + 3.0))
        direction = forward if args.driver_view else -forward
        eye = target + direction * args.camera_distance + side * args.camera_side + Vector((0.0, 0.0, 2.2))
        add_heading_arrow(f"traffic_light_{idx:02d}", loc, heading)
        look_at(camera, eye, target)
        name = safe_name(f"{idx:02d}_{light.get('category')}_{light.get('node_id')}_heading_{math.degrees(heading) % 360.0:.0f}")
        out_path = args.output_dir / f"{name}.png"
        render(out_path)
        manifest.append({
            "path": str(out_path),
            "category": light.get("category"),
            "node_id": light.get("node_id"),
            "heading_degrees": math.degrees(heading) % 360.0,
            "final_location": loc,
            "camera_position": [eye.x, eye.y, eye.z],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rendered": len(manifest), "output_dir": str(args.output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
