#!/usr/bin/env python3
"""Render a localized material view of the generated innout doorway."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render local material views for an innout plan.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def import_glb(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    correction = Matrix(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    objects = [obj for obj in bpy.data.objects if obj not in before]
    for obj in objects:
        if obj.parent is None:
            obj.matrix_world = correction @ obj.matrix_world
    return objects


def import_transformed_usd(path: Path, translate: list[float], yaw_deg: float, scale: float) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(path))
    objects = [obj for obj in bpy.data.objects if obj not in before]
    transform = (
        Matrix.Translation(Vector(translate))
        @ Matrix.Rotation(math.radians(yaw_deg), 4, "Z")
        @ Matrix.Diagonal((scale, scale, scale, 1.0))
    )
    for obj in objects:
        if obj.parent is None:
            obj.matrix_world = transform @ obj.matrix_world
    return objects


def add_lighting() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.78, 0.8, 0.84)
    for name, loc, energy, size in [
        ("key", (0.0, -6.0, 9.0), 550.0, 8.0),
        ("fill", (6.0, 5.0, 5.0), 180.0, 10.0),
    ]:
        light_data = bpy.data.lights.new(f"{name}_area", "AREA")
        light = bpy.data.objects.new(f"{name}_area", light_data)
        bpy.context.collection.objects.link(light)
        light.location = loc
        light.data.energy = energy
        light.data.size = size


def object_path(obj: bpy.types.Object) -> str:
    names = [obj.name]
    parent = obj.parent
    while parent is not None:
        names.append(parent.name)
        parent = parent.parent
    return "/".join(reversed(names))


def hide_default_environment_geometry() -> None:
    for obj in bpy.data.objects:
        path = object_path(obj)
        if "__default_setting" in path or "HDR_Sphere" in path:
            obj.hide_render = True
            obj.hide_viewport = True


def render_view(name: str, location: Vector, target: Vector, output_path: Path, lens: float) -> None:
    camera_data = bpy.data.cameras.new(f"{name}_camera")
    camera = bpy.data.objects.new(f"{name}_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = location
    camera_data.lens = lens
    camera_data.clip_end = 800.0
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
    import_glb(Path(plan["cut_glb"]))
    import_transformed_usd(Path(plan["interior_usd"]), plan["room_translate"], plan["room_yaw_deg"], plan["room_scale"])
    hide_default_environment_geometry()
    add_lighting()

    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.taa_render_samples = 8
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.gtao_distance = 3.0
    bpy.context.scene.eevee.gtao_factor = 1.0
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 675
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "Medium High Contrast"

    door = Vector((plan["door_center_xy"][0], plan["door_center_xy"][1], plan["door_bottom_z"] + 1.5))
    room = Vector((plan["room_translate"][0], plan["room_translate"][1], plan["room_translate"][2] + 1.5))
    outward = Vector((plan["outward_normal_xy"][0], plan["outward_normal_xy"][1], 0.0)).normalized()
    tangent = Vector((plan["tangent_xy"][0], plan["tangent_xy"][1], 0.0)).normalized()

    render_view(
        "material_outside_to_door",
        door + outward * 10.0 + tangent * 1.2 + Vector((0.0, 0.0, 1.4)),
        door - outward * 1.5,
        output_dir / "material_outside_to_door.png",
        30.0,
    )
    render_view(
        "material_inside_to_outside",
        room - outward * 2.0 + Vector((0.0, 0.0, 0.25)),
        door + outward * 5.0,
        output_dir / "material_inside_to_outside.png",
        28.0,
    )
    render_view(
        "material_indoor_space",
        room - outward * 4.0 + tangent * 2.8 + Vector((0.0, 0.0, 1.0)),
        room + tangent * 0.8,
        output_dir / "material_indoor_space.png",
        24.0,
    )
    print(f"Wrote local material renders to {output_dir}")


if __name__ == "__main__":
    main()
