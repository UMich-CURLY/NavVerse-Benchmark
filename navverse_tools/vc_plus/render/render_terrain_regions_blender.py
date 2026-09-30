#!/usr/bin/env python3
"""Render TerrainRegions overlays for quick VC terrain QA."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


REGION_COLORS = {
    "road": (0.95, 0.08, 0.08, 0.58),
    "sidewalk": (1.0, 0.78, 0.12, 0.58),
    "crosswalk": (1.0, 1.0, 1.0, 0.72),
    "green": (0.0, 0.72, 0.20, 0.58),
    "water": (0.05, 0.28, 1.0, 0.62),
}


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
    scene.render.image_settings.compression = 0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.15
    scene.view_settings.gamma = 1.0
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 1
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.60, 0.60, 0.60)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 20.0))
    sun = bpy.context.object
    sun.name = "TerrainRegionSun"
    sun.data.energy = 2.0
    sun.data.use_shadow = False

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 12.0))
    area = bpy.context.object
    area.name = "TerrainRegionFill"
    area.data.energy = 600.0
    area.data.size = 12.0
    area.data.use_shadow = False

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.clip_end = 1_000_000.0
    return camera


def import_usd(path: Path) -> None:
    with suppress_native_output(True):
        bpy.ops.wm.usd_import(filepath=str(path))


def add_region_meshes_from_usd(path: Path, lift: float = 0.006) -> None:
    from pxr import Usd, UsdGeom, Gf

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for category in REGION_COLORS:
        prim = stage.GetPrimAtPath(f"/World/TerrainRegions/{category}")
        if not prim:
            continue
        usd_mesh = UsdGeom.Mesh(prim)
        points = usd_mesh.GetPointsAttr().Get() or []
        counts = usd_mesh.GetFaceVertexCountsAttr().Get() or []
        indices = usd_mesh.GetFaceVertexIndicesAttr().Get() or []
        if not points or not counts or not indices:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        verts = []
        for point in points:
            world = local_to_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            verts.append((float(world[0]), float(world[1]), float(world[2]) + lift))
        faces = []
        cursor = 0
        for count in counts:
            face = tuple(int(index) for index in indices[cursor:cursor + count])
            cursor += count
            if len(face) >= 3:
                faces.append(face)
        mesh = bpy.data.meshes.new(f"TerrainRegions_{category}_mesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(f"TerrainRegions_{category}", mesh)
        bpy.context.collection.objects.link(obj)


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        try:
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Alpha"].default_value = color[3]
            bsdf.inputs["Roughness"].default_value = 0.82
        except Exception:
            pass
    mat.blend_method = "BLEND"
    mat.use_screen_refraction = False
    return mat


def object_family(obj: bpy.types.Object) -> str:
    names = []
    current = obj
    while current:
        names.append(current.name.lower())
        current = current.parent
    joined = "/".join(names)
    for category in REGION_COLORS:
        if category in joined:
            return category
    if "terrain" in joined or "ground" in joined:
        return "terrain"
    if "building" in joined:
        return "building"
    return "other"


def style_scene() -> None:
    mats = {name: material(f"QA_{name}", color) for name, color in REGION_COLORS.items()}
    terrain_mat = material("QA_terrain_gray", (0.34, 0.34, 0.34, 1.0))
    building_mat = material("QA_building_gray", (0.54, 0.54, 0.54, 1.0))
    other_mat = material("QA_other_gray", (0.45, 0.45, 0.45, 1.0))
    for obj in bpy.data.objects:
        obj.hide_viewport = False
        obj.hide_render = False
        if obj.type != "MESH":
            continue
        family = object_family(obj)
        if family in mats:
            obj.data.materials.clear()
            obj.data.materials.append(mats[family])
            obj.show_transparent = True
        elif family == "terrain":
            obj.data.materials.clear()
            obj.data.materials.append(terrain_mat)
        elif family == "building":
            obj.data.materials.clear()
            obj.data.materials.append(building_mat)
        else:
            obj.data.materials.clear()
            obj.data.materials.append(other_mat)


def mesh_bounds(region_only: bool = False, road_only: bool = False) -> tuple[Vector, Vector] | None:
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        family = object_family(obj)
        if region_only and family not in REGION_COLORS:
            continue
        if road_only and family != "road":
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


def render_views(usd: Path, output_dir: Path, width: int, height: int) -> dict[str, str]:
    reset_scene()
    camera = setup_render(width, height)
    import_usd(usd)
    add_region_meshes_from_usd(usd)
    style_scene()

    bounds = mesh_bounds(region_only=False)
    region_bounds = mesh_bounds(region_only=True)
    road_bounds = mesh_bounds(road_only=True)
    if bounds is None or region_bounds is None:
        raise RuntimeError("Imported scene did not contain renderable terrain region meshes")

    mins, maxs = bounds
    center = (mins + maxs) * 0.5
    dims = maxs - mins
    extent = max(dims.x, dims.y, 1.0)

    outputs = {}
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = extent * 1.08
    look_at(camera, Vector((center.x, center.y, maxs.z + extent * 1.5)), center)
    top = output_dir / "terrain_regions_top.png"
    render(top)
    outputs["top"] = str(top)

    camera.data.ortho_scale = extent * 0.82
    look_at(
        camera,
        Vector((center.x - extent * 0.55, center.y - extent * 0.75, maxs.z + extent * 0.55)),
        center,
    )
    oblique = output_dir / "terrain_regions_oblique.png"
    render(oblique)
    outputs["oblique"] = str(oblique)

    if road_bounds is not None:
        rmins, rmaxs = road_bounds
        road_center = (rmins + rmaxs) * 0.5
        road_extent = max(rmaxs.x - rmins.x, rmaxs.y - rmins.y, 1.0)
        camera.data.ortho_scale = min(max(road_extent * 0.24, 1.2), extent * 0.35)
        look_at(
            camera,
            Vector((road_center.x - 1.8, road_center.y - 2.4, road_center.z + 1.6)),
            road_center,
        )
        close = output_dir / "road_lowering_close.png"
        render(close)
        outputs["road_close"] = str(close)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    outputs = render_views(args.usd, args.output_dir, args.width, args.height)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
