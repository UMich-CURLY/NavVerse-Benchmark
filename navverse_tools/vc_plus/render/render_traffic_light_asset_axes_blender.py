#!/usr/bin/env python3
"""Render standalone traffic-light asset axes for orientation debugging."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def look_at(camera, eye: Vector, target: Vector) -> None:
    camera.location = eye
    camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.asset))
    scene = bpy.context.scene
    scene.render.resolution_x = 900
    scene.render.resolution_y = 700
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.6, 0.6, 0.6)
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 1
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
    bpy.context.object.data.energy = 3
    bpy.ops.object.light_add(type="AREA", location=(0, -5, 8))
    bpy.context.object.data.energy = 500
    bpy.context.object.data.size = 10
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    scene.camera = cam
    cam.data.lens = 42

    target = Vector((0, 0, 3.5))
    views = {
        "from_pos_x": Vector((12, 0, 4)),
        "from_neg_x": Vector((-12, 0, 4)),
        "from_pos_y": Vector((0, 12, 4)),
        "from_neg_y": Vector((0, -12, 4)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, eye in views.items():
        look_at(cam, eye, target)
        scene.render.filepath = str(args.output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    print(f"Wrote asset axis renders to {args.output_dir}")


if __name__ == "__main__":
    main()
