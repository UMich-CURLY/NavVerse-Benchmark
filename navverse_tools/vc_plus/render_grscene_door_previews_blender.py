#!/usr/bin/env python3
"""Render quick door-focused previews for GRScenes USDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GRScene door previews.")
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def object_path(obj: bpy.types.Object) -> str:
    names = [obj.name]
    parent = obj.parent
    while parent is not None:
        names.append(parent.name)
        parent = parent.parent
    return "/".join(reversed(names))


def hide_environment() -> None:
    for obj in bpy.data.objects:
        path = object_path(obj)
        if "__default_setting" in path or "HDR_Sphere" in path:
            obj.hide_render = True
            obj.hide_viewport = True


def mesh_world_corners(obj: bpy.types.Object) -> list[Vector]:
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def bbox_for(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = []
    for obj in objects:
        if obj.type == "MESH":
            points.extend(mesh_world_corners(obj))
    if not points:
        points = [Vector((0.0, 0.0, 0.0))]
    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_v, max_v


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_view(name: str, location: Vector, target: Vector, ortho_scale: float, output_path: Path) -> None:
    cam_data = bpy.data.cameras.new(f"{name}_camera")
    camera = bpy.data.objects.new(f"{name}_camera", cam_data)
    bpy.context.collection.objects.link(camera)
    camera.location = location
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_end = 10000.0
    look_at(camera, target)
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(Path(args.scene_usd).resolve()))
    hide_environment()

    door_objects = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and "/door/" in object_path(obj).lower()
    ]
    if not door_objects:
        door_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    min_v, max_v = bbox_for(door_objects)
    center = (min_v + max_v) * 0.5
    dims = max_v - min_v
    scale = max(5.0, min(35.0, max(dims.x, dims.y, dims.z) * 1.25))

    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.display.shading.light = "STUDIO"
    bpy.context.scene.display.shading.color_type = "MATERIAL"
    bpy.context.scene.display.shading.background_type = "VIEWPORT"
    bpy.context.scene.display.shading.background_color = (0.9, 0.92, 0.95)
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 675
    bpy.context.scene.view_settings.view_transform = "Standard"

    views = {
        "front_y": Vector((center.x, center.y - scale * 1.8, center.z + scale * 0.18)),
        "back_y": Vector((center.x, center.y + scale * 1.8, center.z + scale * 0.18)),
        "side_x": Vector((center.x + scale * 1.8, center.y, center.z + scale * 0.18)),
        "top": Vector((center.x, center.y, center.z + scale * 2.3)),
    }
    for label, location in views.items():
        render_view(
            f"{args.scene_name}_{label}",
            location,
            center,
            scale,
            output_dir / f"{args.scene_name}_{label}.png",
        )
    print(f"Wrote {args.scene_name} door previews to {output_dir}; door_meshes={len(door_objects)}")


if __name__ == "__main__":
    main()
