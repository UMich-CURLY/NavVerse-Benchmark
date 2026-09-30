#!/usr/bin/env python3
"""Render Objaverse asset size comparison sheets with a 1m cube."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from objaverse_asset_rules import objaverse_asset_enabled, objaverse_asset_scale


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def look_at(camera: bpy.types.Object, eye: Vector, target: Vector) -> None:
    camera.location = eye
    camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()


def setup_render(width: int, height: int) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.1
    scene.view_settings.gamma = 1.0
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 1
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.62, 0.62, 0.62)
    bpy.ops.object.light_add(type="SUN", location=(0, -5, 12))
    bpy.context.object.data.energy = 2.5
    bpy.ops.object.light_add(type="AREA", location=(0, -4, 8))
    bpy.context.object.data.energy = 650.0
    bpy.context.object.data.size = 8.0
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.type = "ORTHO"
    return camera


def object_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    coords = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mn = Vector((min(p.x for p in coords), min(p.y for p in coords), min(p.z for p in coords)))
    mx = Vector((max(p.x for p in coords), max(p.y for p in coords), max(p.z for p in coords)))
    return mn, mx


def subtree_bbox(root: bpy.types.Object) -> tuple[Vector, Vector]:
    mins = []
    maxs = []
    for obj in [root, *root.children_recursive]:
        if obj.type != "MESH":
            continue
        mn, mx = object_bbox(obj)
        mins.append(mn)
        maxs.append(mx)
    if not mins:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    return (
        Vector((min(v.x for v in mins), min(v.y for v in mins), min(v.z for v in mins))),
        Vector((max(v.x for v in maxs), max(v.y for v in maxs), max(v.z for v in maxs))),
    )


def import_asset(path: Path, name: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    if path.suffix.lower() == ".usd":
        bpy.ops.wm.usd_import(filepath=str(path))
    elif path.suffix.lower() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise RuntimeError(f"Unsupported asset type: {path}")
    new_objects = [obj for obj in bpy.data.objects if obj not in before]
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    for obj in new_objects:
        if obj.parent is None:
            obj.parent = root
    return root


def align_to_ground_and_center(root: bpy.types.Object, x: float, y: float) -> dict:
    mn, mx = subtree_bbox(root)
    center = (mn + mx) / 2.0
    root.location += Vector((x - center.x, y - center.y, -mn.z))
    mn2, mx2 = subtree_bbox(root)
    size = mx2 - mn2
    return {
        "min": [mn2.x, mn2.y, mn2.z],
        "max": [mx2.x, mx2.y, mx2.z],
        "size": [size.x, size.y, size.z],
        "center": [(mn2.x + mx2.x) / 2.0, (mn2.y + mx2.y) / 2.0, (mn2.z + mx2.z) / 2.0],
    }


def add_label(text: str, x: float, y: float, z: float) -> None:
    bpy.ops.object.text_add(location=(x, y, z))
    obj = bpy.context.object
    obj.name = f"label_{text[:32]}"
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = 0.28
    obj.rotation_euler = (math.radians(70), 0, 0)


def add_reference_cube(x: float, y: float) -> dict:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, 0.5))
    cube = bpy.context.object
    cube.name = "reference_1m_cube"
    mat = bpy.data.materials.new("reference_1m_cube_mat")
    mat.diffuse_color = (0.1, 0.3, 1.0, 1.0)
    cube.data.materials.append(mat)
    add_label("1m cube", x, y - 1.0, 0.05)
    return {"size": [1.0, 1.0, 1.0], "center": [x, y, 0.5]}


def render_group(assets: list[Path], output_path: Path, width: int, height: int) -> list[dict]:
    reset_scene()
    camera = setup_render(width, height)
    records = []
    xs = [-4.5, -1.5, 1.5, 4.5]
    add_reference_cube(xs[0], 0.0)
    for idx, asset in enumerate(assets):
        root = import_asset(asset, f"asset_{asset.stem}")
        scale = objaverse_asset_scale(asset)
        root.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        bbox = align_to_ground_and_center(root, xs[idx + 1], 0.0)
        label = asset.name if scale == 1.0 else f"{asset.name} x{scale:.2f}"
        add_label(label, xs[idx + 1], -1.5, 0.05)
        records.append({"asset": str(asset), "name": asset.name, "scale": scale, **bbox})

    all_mins = []
    all_maxs = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mn, mx = object_bbox(obj)
        all_mins.append(mn)
        all_maxs.append(mx)
    min_x = min(v.x for v in all_mins)
    max_x = max(v.x for v in all_maxs)
    max_z = max(v.z for v in all_maxs)
    target = Vector(((min_x + max_x) / 2.0, 0.0, max_z * 0.45))
    camera.data.ortho_scale = max(6.0, max_x - min_x + 2.0, max_z + 2.0)
    look_at(camera, target + Vector((0.0, -8.5, max(3.0, max_z * 0.65))), target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Objaverse asset size comparison images.")
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets", nargs="*", default=None, help="Optional asset filenames to render.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    if args.assets:
        assets = [args.asset_dir / name for name in args.assets]
    else:
        assets = sorted(args.asset_dir.glob("*.usd"))
    assets = [asset for asset in assets if asset.exists() and objaverse_asset_enabled(asset)]
    manifest = []
    for group_idx in range(0, len(assets), 3):
        group = assets[group_idx:group_idx + 3]
        out_path = args.output_dir / f"size_group_{group_idx // 3:03d}.png"
        records = render_group(group, out_path, args.width, args.height)
        manifest.append({"image": str(out_path), "records": records})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"assets": len(assets), "images": len(manifest), "output_dir": str(args.output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
