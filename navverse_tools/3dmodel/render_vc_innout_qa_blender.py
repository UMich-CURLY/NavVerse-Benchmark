#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args():
    parser = argparse.ArgumentParser(description="Render local QA views for a generated VC innout doorway.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--storefront-glb")
    parser.add_argument("--storefront-usd")
    parser.add_argument("--output-dir", required=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path, preserve_axes=False):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = [obj for obj in bpy.data.objects if obj not in before]
    if preserve_axes:
        correction = Matrix(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        for obj in objects:
            if obj.parent is None:
                obj.matrix_world = correction @ obj.matrix_world
    return objects


def import_usd(path):
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(path))
    return [obj for obj in bpy.data.objects if obj not in before]


def make_material(name, color, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, alpha)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*color, alpha)
        bsdf.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = True
    return mat


def apply_material(objects, material):
    for obj in objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)


def add_door_marker(plan, material):
    center_x, center_y = plan["door_center_xy"]
    bottom_z = plan["door_bottom_z"]
    width = float(plan.get("door_width", 4.5))
    height = float(plan.get("door_height", 3.2))
    depth = float(plan.get("door_depth", 8.0))
    yaw = math.radians(plan["doorway_yaw_deg"])

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(center_x, center_y, bottom_z + height * 0.5))
    cube = bpy.context.object
    cube.name = "door_opening_marker"
    cube.rotation_euler = (0.0, 0.0, yaw)
    cube.dimensions = (width, depth, height)
    cube.data.materials.append(material)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    cube.display_type = "WIRE"
    return cube


def add_room_marker(plan, material):
    x, y, z = plan["room_translate"]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, z + 1.5))
    room = bpy.context.object
    room.name = "indoor_room_proxy"
    room.rotation_euler = (0.0, 0.0, math.radians(plan["room_yaw_deg"]))
    room.dimensions = (8.0, 8.0, 3.0)
    room.data.materials.append(material)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    room.display_type = "WIRE"
    return room


def add_storefront(plan, storefront_glb, material):
    if storefront_glb is None:
        return None
    objects = import_glb(storefront_glb, preserve_axes=True)
    apply_material(objects, material)
    root = bpy.data.objects.new("storefront_root", None)
    bpy.context.collection.objects.link(root)
    for obj in objects:
        if obj.parent is None:
            obj.parent = root
    tx, ty, tz = plan["storefront_translate"]
    root.location = (tx / 100.0, ty / 100.0, tz / 100.0)
    root.rotation_euler = (0.0, 0.0, math.radians(plan["storefront_yaw_deg"]))
    root.scale = (1.0, 1.0, 1.0)
    return root


def add_storefront_usd(plan, storefront_usd, material):
    if storefront_usd is None:
        return None
    objects = import_usd(storefront_usd)
    apply_material(objects, material)
    root = bpy.data.objects.new("storefront_root", None)
    bpy.context.collection.objects.link(root)
    for obj in objects:
        if obj.parent is None:
            obj.parent = root
    tx, ty, tz = plan["storefront_translate"]
    root.location = (tx / 100.0, ty / 100.0, tz / 100.0)
    root.rotation_euler = (0.0, 0.0, math.radians(plan["storefront_yaw_deg"]))
    root.scale = (1.0, 1.0, 1.0)
    return root


def add_lighting():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.92, 0.94, 0.96)

    sun_data = bpy.data.lights.new("qa_sun", "SUN")
    sun = bpy.data.objects.new("qa_sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(45.0), 0.0, math.radians(-35.0))
    sun.data.energy = 3.0

    area_data = bpy.data.lights.new("qa_area", "AREA")
    area = bpy.data.objects.new("qa_area", area_data)
    bpy.context.collection.objects.link(area)
    area.location = (0.0, 0.0, 160.0)
    area.data.size = 80.0
    area.data.energy = 400.0


def set_tree_hide_render(obj, hide_render):
    if obj is None:
        return
    obj.hide_render = hide_render
    for child in obj.children_recursive:
        child.hide_render = hide_render


def look_at(camera, target):
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_view(name, location, target, output_path, orthographic_scale=None):
    cam_data = bpy.data.cameras.new(f"{name}_camera")
    camera = bpy.data.objects.new(f"{name}_camera", cam_data)
    bpy.context.collection.objects.link(camera)
    camera.location = Vector(location)
    look_at(camera, target)
    if orthographic_scale is not None:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = orthographic_scale
    else:
        cam_data.lens = 28
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def write_qa_legend(output_dir, plan):
    legend = {
        "colors": {
            "dark_gray": "cut building mesh",
            "yellow": "optional exterior storefront/facade overlay; not the GR indoor scene",
            "green_translucent": "door opening marker; this is the cut volume, not the store",
            "blue_translucent": "indoor room proxy for placement review, not final room geometry",
        },
        "views": {
            "front_building.png": "cut building only",
            "front_clean.png": "cut building plus optional exterior storefront/facade when enabled",
            "front_clearance.png": "cut building plus green door opening marker",
            "front.png": "all QA overlays: building, optional exterior storefront/facade, door marker, and room proxy",
            "perspective.png": "oblique all-overlay review",
            "outside_oblique.png": "wider exterior all-overlay review",
            "topdown.png": "top-down all-overlay review",
        },
        "door": {
            "width_m": plan.get("door_width"),
            "height_m": plan.get("door_height"),
            "depth_m": plan.get("door_depth"),
            "size_source": plan.get("door_size_source"),
        },
        "storefront_bbox": {
            "min": plan.get("storefront_bbox_min"),
            "max": plan.get("storefront_bbox_max"),
            "dims": plan.get("storefront_bbox_dims"),
            "exterior_storefront_enabled": plan.get("exterior_storefront_enabled"),
        },
    }
    (output_dir / "qa_legend.json").write_text(json.dumps(legend, indent=2, sort_keys=True) + "\n")
    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                "QA color legend",
                "dark gray: cut building mesh",
                "yellow: optional exterior storefront/facade overlay; not the GR indoor scene",
                "green translucent: door opening marker / cut volume, not the store",
                "blue translucent: indoor room proxy for placement review, not final room geometry",
                "",
                "View files",
                "front_building.png: cut building only",
                "front_clean.png: cut building plus optional exterior storefront/facade when enabled",
                "front_clearance.png: cut building plus green door opening marker",
                "front.png: all QA overlays",
                "perspective.png/outside_oblique.png/topdown.png: all-overlay review views",
                "",
                f"door_width_m: {plan.get('door_width')}",
                f"door_height_m: {plan.get('door_height')}",
                f"door_depth_m: {plan.get('door_depth')}",
                f"door_size_source: {plan.get('door_size_source')}",
            ]
        )
        + "\n"
    )


def main():
    args = parse_args()
    plan = json.loads(Path(args.plan).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_qa_legend(output_dir, plan)

    clear_scene()
    cut_objects = import_glb(plan["cut_glb"], preserve_axes=True)
    apply_material(cut_objects, make_material("cut_building_gray", (0.28, 0.30, 0.32), 1.0))
    storefront_glb = Path(args.storefront_glb).resolve() if args.storefront_glb else None
    storefront_usd = Path(args.storefront_usd).resolve() if args.storefront_usd else None
    storefront = None
    storefront_material = make_material("storefront_visible", (0.95, 0.62, 0.10), 1.0)
    if storefront_usd is not None and storefront_usd.exists():
        storefront = add_storefront_usd(plan, storefront_usd, storefront_material)
    elif storefront_glb is not None and storefront_glb.exists():
        storefront = add_storefront(plan, storefront_glb, storefront_material)
    door_marker = add_door_marker(plan, make_material("door_marker", (0.05, 0.8, 0.2), 0.12))
    room_marker = add_room_marker(plan, make_material("room_marker", (0.1, 0.35, 1.0), 0.07))
    add_lighting()

    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.display.shading.light = "STUDIO"
    bpy.context.scene.display.shading.color_type = "MATERIAL"
    bpy.context.scene.display.shading.background_type = "VIEWPORT"
    bpy.context.scene.display.shading.background_color = (0.92, 0.94, 0.96)
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 900
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "Medium High Contrast"

    center = Vector((plan["door_center_xy"][0], plan["door_center_xy"][1], plan["door_bottom_z"] + 2.0))
    outward = Vector((plan["outward_normal_xy"][0], plan["outward_normal_xy"][1], 0.0)).normalized()
    tangent = Vector((plan["tangent_xy"][0], plan["tangent_xy"][1], 0.0)).normalized()

    set_tree_hide_render(storefront, True)
    door_marker.hide_render = True
    room_marker.hide_render = True
    render_view(
        "front_building",
        center + outward * 22.0 + Vector((0.0, 0.0, 5.0)),
        center,
        output_dir / "front_building.png",
        orthographic_scale=14.0,
    )
    set_tree_hide_render(storefront, False)
    door_marker.hide_render = True
    room_marker.hide_render = True
    render_view(
        "front_clean",
        center + outward * 22.0 + Vector((0.0, 0.0, 5.0)),
        center,
        output_dir / "front_clean.png",
        orthographic_scale=14.0,
    )
    set_tree_hide_render(storefront, True)
    door_marker.hide_render = False
    room_marker.hide_render = True
    render_view(
        "front_clearance",
        center + outward * 22.0 + Vector((0.0, 0.0, 5.0)),
        center,
        output_dir / "front_clearance.png",
        orthographic_scale=14.0,
    )
    set_tree_hide_render(storefront, False)
    door_marker.hide_render = False
    room_marker.hide_render = False
    render_view(
        "front",
        center + outward * 22.0 + Vector((0.0, 0.0, 5.0)),
        center,
        output_dir / "front.png",
        orthographic_scale=14.0,
    )
    render_view(
        "perspective",
        center + outward * 18.0 + tangent * 7.0 + Vector((0.0, 0.0, 8.0)),
        center,
        output_dir / "perspective.png",
    )
    render_view(
        "outside_oblique",
        center + outward * 28.0 - tangent * 14.0 + Vector((0.0, 0.0, 10.0)),
        center,
        output_dir / "outside_oblique.png",
    )
    render_view(
        "topdown",
        center + Vector((0.0, 0.0, 95.0)),
        center,
        output_dir / "topdown.png",
        orthographic_scale=45.0,
    )
    print(f"Wrote QA renders to {output_dir}")


if __name__ == "__main__":
    main()
