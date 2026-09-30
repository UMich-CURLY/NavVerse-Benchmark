#!/usr/bin/env python3
"""Apply scored storefront boolean cuts and export a USD layer."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.data.objects if obj not in before]


def import_usd(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(path))
    return [obj for obj in bpy.data.objects if obj not in before]


def mesh_objects(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    return [obj for obj in objects if obj.type == "MESH"]


def create_cutter(facade: dict[str, Any]) -> bpy.types.Object:
    center = Vector(facade["center"])
    normal = Vector(facade["normal"])
    width = float(facade.get("opening_width", 3.0))
    height = float(facade.get("opening_height", 2.8))
    depth = 3.0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center - normal * 0.15)
    cutter = bpy.context.object
    cutter.name = "storefront_boolean_cutter"
    if abs(normal.x) > abs(normal.y):
        cutter.dimensions = (depth, width, height)
    else:
        cutter.dimensions = (width, depth, height)
    bpy.context.view_layer.objects.active = cutter
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return cutter


def apply_boolean(building_meshes: list[bpy.types.Object], cutter: bpy.types.Object) -> list[str]:
    cut_names = []
    for obj in building_meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        mod = obj.modifiers.new("storefront_opening", "BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.object = cutter
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
            cut_names.append(obj.name)
        except Exception:
            obj.modifiers.remove(mod)
        obj.select_set(False)
    cutter.hide_viewport = True
    cutter.hide_render = True
    return cut_names


def place_storefront(asset_path: Path, facade: dict[str, Any]) -> dict[str, Any]:
    objects = import_usd(asset_path)
    normal = Vector(facade["normal"])
    center = Vector(facade["center"])
    target = center + normal * 0.35
    for obj in objects:
        obj.location += target
        if abs(normal.x) > abs(normal.y):
            obj.rotation_euler[2] = math.pi / 2 if normal.x > 0 else -math.pi / 2
        else:
            obj.rotation_euler[2] = 0.0 if normal.y > 0 else math.pi
    return {
        "placement_count": 1,
        "imported_object_count": len(objects),
        "imported_object_names": [obj.name for obj in objects],
    }


def export_usd(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.usd_export(filepath=str(path), selected_objects_only=False, export_materials=True, export_textures=True)
    except TypeError:
        bpy.ops.wm.usd_export(filepath=str(path), selected_objects_only=False, export_materials=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    args = parser.parse_args(argv)

    score = read_json(args.score)
    selected = score.get("selected")
    if not selected:
        raise RuntimeError("No selected building in score JSON")
    building_path = Path(selected["building_path"])
    facade = selected["facade"]
    storefront = selected.get("nearest_storefront") or {}
    asset_path = Path(storefront.get("asset_path", ""))
    if not asset_path.exists():
        raise RuntimeError(f"Missing storefront asset: {asset_path}")

    reset_scene()
    building_objects = import_glb(building_path)
    cutter = create_cutter(facade)
    cut_names = apply_boolean(mesh_objects(building_objects), cutter)
    storefront_result = place_storefront(asset_path, facade)
    export_usd(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "status": "ok",
        "method": "single scored building Blender boolean cut",
        "building_path": str(building_path),
        "storefront_asset": str(asset_path),
        "facade": facade,
        "cut_meshes": cut_names,
        "storefront_placement_count": storefront_result["placement_count"],
        "imported_storefront_object_count": storefront_result["imported_object_count"],
        "imported_storefront_objects": storefront_result["imported_object_names"],
        "output": str(args.output),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
