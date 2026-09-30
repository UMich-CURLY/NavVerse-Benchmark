#!/usr/bin/env python3
"""Render actual final innout USD views around the selected doorway."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render actual VC innout views from a final USD.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine", choices=["workbench", "eevee"], default="workbench")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_lighting() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.88, 0.9, 0.93)

    sun_data = bpy.data.lights.new("actual_qa_sun", "SUN")
    sun = bpy.data.objects.new("actual_qa_sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(48.0), 0.0, math.radians(-35.0))
    sun.data.energy = 2.5

    area_data = bpy.data.lights.new("actual_qa_area", "AREA")
    area = bpy.data.objects.new("actual_qa_area", area_data)
    bpy.context.collection.objects.link(area)
    area.location = (0.0, 0.0, 180.0)
    area.data.size = 120.0
    area.data.energy = 500.0


def render_view(name: str, location: Vector, target: Vector, output_path: Path, lens: float = 24.0) -> None:
    cam_data = bpy.data.cameras.new(f"{name}_camera")
    camera = bpy.data.objects.new(f"{name}_camera", cam_data)
    bpy.context.collection.objects.link(camera)
    camera.location = location
    cam_data.lens = lens
    cam_data.clip_end = 2000.0
    look_at(camera, target)
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    plan = json.loads(Path(args.plan).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(Path(args.usd).resolve()))
    add_lighting()

    if args.engine == "eevee":
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.gtao_distance = 4.0
        bpy.context.scene.eevee.gtao_factor = 1.2
    else:
        bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
        bpy.context.scene.display.shading.light = "STUDIO"
        bpy.context.scene.display.shading.color_type = "MATERIAL"
        bpy.context.scene.display.shading.background_type = "VIEWPORT"
        bpy.context.scene.display.shading.background_color = (0.88, 0.9, 0.93)
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 900
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "Medium High Contrast"

    door = Vector((plan["door_center_xy"][0], plan["door_center_xy"][1], plan["door_bottom_z"] + 1.65))
    room = Vector((plan["room_translate"][0], plan["room_translate"][1], plan["room_translate"][2] + 1.55))
    outward = Vector((plan["outward_normal_xy"][0], plan["outward_normal_xy"][1], 0.0)).normalized()
    tangent = Vector((plan["tangent_xy"][0], plan["tangent_xy"][1], 0.0)).normalized()

    render_view(
        "actual_outside_to_door",
        door + outward * 11.0 + tangent * 1.5 + Vector((0.0, 0.0, 1.3)),
        door - outward * 2.0,
        output_dir / "actual_outside_to_door.png",
        lens=32.0,
    )
    render_view(
        "actual_inside_to_outside",
        room - outward * 1.8 + Vector((0.0, 0.0, 0.35)),
        door + outward * 6.0,
        output_dir / "actual_inside_to_outside.png",
        lens=28.0,
    )
    render_view(
        "actual_indoor_space",
        room - outward * 4.0 + tangent * 3.0 + Vector((0.0, 0.0, 1.0)),
        room + tangent * 0.5,
        output_dir / "actual_indoor_space.png",
        lens=24.0,
    )
    print(f"Wrote actual innout renders to {output_dir}")


if __name__ == "__main__":
    main()
