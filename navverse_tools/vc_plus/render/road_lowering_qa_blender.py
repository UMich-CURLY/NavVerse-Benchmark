#!/usr/bin/env python3
"""Run a focused OSM road-lowering QA pass on one VC terrain GLB."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from blender_process_cities_with_roads import (
    build_osm_road_segments_for_lowering,
    clear_scene,
    cut_terrain_for_road_lowering,
    import_glb,
    lower_terrain_under_osm_roads,
    subdivide_terrain_for_road_lowering,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-dir", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=float, default=0.2)
    parser.add_argument("--subdivide-cuts", type=int, default=1)
    parser.add_argument("--cell-size", type=float, default=25.0)
    parser.add_argument("--max-segments", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args(argv)


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.86
    return mat


def world_vertex_map(objects: list[bpy.types.Object]) -> dict[tuple[str, int], Vector]:
    points = {}
    for obj in objects:
        if obj.type != "MESH" or not obj.data:
            continue
        for vertex in obj.data.vertices:
            points[(obj.name, vertex.index)] = obj.matrix_world @ vertex.co
    return points


def vertex_deltas(objects: list[bpy.types.Object], before: dict[tuple[str, int], Vector]) -> dict[tuple[str, int], float]:
    deltas = {}
    for obj in objects:
        if obj.type != "MESH" or not obj.data:
            continue
        for vertex in obj.data.vertices:
            key = (obj.name, vertex.index)
            if key in before:
                after = obj.matrix_world @ vertex.co
                deltas[key] = float(after.z - before[key].z)
    return deltas


def classify_faces(objects: list[bpy.types.Object], deltas: dict[tuple[str, int], float], depth: float) -> None:
    lowered_mat = material("QA_lowered_road_dark", (0.04, 0.045, 0.05, 1.0))
    curb_mat = material("QA_curb_sidewalk_light", (0.72, 0.70, 0.62, 1.0))
    for obj in objects:
        if obj.type != "MESH" or not obj.data:
            continue
        obj.data.materials.clear()
        obj.data.materials.append(curb_mat)
        obj.data.materials.append(lowered_mat)
        for poly in obj.data.polygons:
            face_deltas = [deltas.get((obj.name, vertex_index), 0.0) for vertex_index in poly.vertices]
            lowered_votes = sum(1 for delta in face_deltas if delta <= -depth * 0.5)
            poly.material_index = 1 if lowered_votes >= max(1, math.ceil(len(face_deltas) * 0.5)) else 0


def best_segment_profile(
    objects: list[bpy.types.Object],
    before: dict[tuple[str, int], Vector],
    deltas: dict[tuple[str, int], float],
    segments: list[dict],
    depth: float,
) -> dict:
    candidates = []
    vertices = []
    for obj in objects:
        if obj.type != "MESH" or not obj.data:
            continue
        for vertex in obj.data.vertices:
            key = (obj.name, vertex.index)
            if key not in before:
                continue
            after = obj.matrix_world @ vertex.co
            vertices.append((before[key], after, deltas.get(key, 0.0)))

    for seg_index, seg in enumerate(segments):
        sx, sy = seg["start"]
        dx, dy = seg["delta"]
        length = math.sqrt(seg["length_sq"])
        if length <= 1e-6:
            continue
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx
        half_width = float(seg["half_width"])
        inside = []
        outside = []
        for before_pt, after_pt, delta in vertices:
            rel_x = before_pt.x - sx
            rel_y = before_pt.y - sy
            s = rel_x * tx + rel_y * ty
            if s < 0.0 or s > length:
                continue
            lateral = rel_x * nx + rel_y * ny
            if abs(lateral) <= half_width:
                if delta <= -depth * 0.5:
                    inside.append((s, lateral, before_pt, after_pt, delta))
            elif half_width < abs(lateral) <= half_width + 1.2:
                outside.append((s, lateral, before_pt, after_pt, delta))
        if inside and outside:
            center_s = sorted(item[0] for item in inside)[len(inside) // 2]
            nearby_inside = [item for item in inside if abs(item[0] - center_s) <= 2.5]
            nearby_outside = [item for item in outside if abs(item[0] - center_s) <= 2.5]
            if nearby_inside and nearby_outside:
                candidates.append((len(nearby_inside) + len(nearby_outside), seg_index, center_s, nearby_inside, nearby_outside))

    if not candidates:
        raise RuntimeError("No road-lowering boundary profile with both lowered road and non-lowered roadside vertices was found")
    _score, seg_index, center_s, inside, outside = max(candidates, key=lambda item: item[0])
    seg = segments[seg_index]
    sx, sy = seg["start"]
    dx, dy = seg["delta"]
    length = math.sqrt(seg["length_sq"])
    tx, ty = dx / length, dy / length
    nx, ny = -ty, tx
    center = Vector((sx + tx * center_s, sy + ty * center_s, 0.0))

    def median(values: list[float]) -> float:
        values = sorted(values)
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return 0.5 * (values[mid - 1] + values[mid])

    samples = []
    for item in sorted(inside + outside, key=lambda entry: entry[1]):
        s, lateral, before_pt, after_pt, delta = item
        samples.append({
            "s": float(s),
            "lateral": float(lateral),
            "z_before": float(before_pt.z),
            "z_after": float(after_pt.z),
            "delta_z": float(delta),
            "inside_road": abs(float(lateral)) <= float(seg["half_width"]),
        })

    return {
        "segment_index": seg_index,
        "way_id": seg.get("way_id"),
        "center_xy": [float(center.x), float(center.y)],
        "tangent_xy": [float(tx), float(ty)],
        "normal_xy": [float(nx), float(ny)],
        "half_width": float(seg["half_width"]),
        "inside_count": len(inside),
        "outside_count": len(outside),
        "inside_delta_median": median([float(item[4]) for item in inside]),
        "outside_delta_median": median([float(item[4]) for item in outside]),
        "samples": samples[:120],
    }


def add_profile_markers(profile: dict) -> None:
    cx, cy = profile["center_xy"]
    nx, ny = profile["normal_xy"]
    half_width = profile["half_width"]
    z = max(sample["z_after"] for sample in profile["samples"]) + 0.03
    curve = bpy.data.curves.new("QA_profile_cross_section", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = 0.035
    curve.materials.append(material("QA_profile_blue", (0.0, 0.25, 1.0, 1.0)))
    spl = curve.splines.new("POLY")
    spl.points.add(1)
    span = half_width + 1.2
    for point, offset in zip(spl.points, (-span, span)):
        point.co = (cx + nx * offset, cy + ny * offset, z, 1.0)
    bpy.context.collection.objects.link(bpy.data.objects.new("QA_profile_cross_section", curve))

    for name, offset in (("left_curb", -half_width), ("right_curb", half_width)):
        curve = bpy.data.curves.new(f"QA_{name}", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.025
        curve.materials.append(material(f"QA_{name}_yellow", (1.0, 0.82, 0.0, 1.0)))
        spl = curve.splines.new("POLY")
        spl.points.add(1)
        tx, ty = profile["tangent_xy"]
        for point, s in zip(spl.points, (-2.5, 2.5)):
            point.co = (cx + nx * offset + tx * s, cy + ny * offset + ty * s, z + 0.02, 1.0)
        bpy.context.collection.objects.link(bpy.data.objects.new(f"QA_{name}", curve))


def look_at(camera: bpy.types.Object, eye: Vector, target: Vector) -> None:
    direction = target - eye
    camera.location = eye
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render(profile: dict, output_path: Path, width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 1
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.78, 0.80, 0.82)
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 20.0))
    bpy.context.object.data.energy = 2.5
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(profile["half_width"] * 2.8, 5.0)
    cx, cy = profile["center_xy"]
    nx, ny = profile["normal_xy"]
    tx, ty = profile["tangent_xy"]
    z = max(sample["z_after"] for sample in profile["samples"])
    target = Vector((cx, cy, z))
    eye = Vector((cx - nx * 3.2 - tx * 2.6, cy - ny * 3.2 - ty * 2.6, z + 2.2))
    look_at(camera, eye, target)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    clear_scene()
    terrain_glb = args.city_dir / "terrain.glb"
    osm = args.city_dir / "road_data" / "road_data_with_height.osm"
    terrain_objects = import_glb(str(terrain_glb))
    if not terrain_objects:
        raise RuntimeError(f"No terrain objects imported from {terrain_glb}")
    segments = build_osm_road_segments_for_lowering(str(osm))
    if args.max_segments > 0:
        segments = sorted(segments, key=lambda seg: float(seg["length_sq"]), reverse=True)[:args.max_segments]
    subdivision = subdivide_terrain_for_road_lowering(terrain_objects, cuts=args.subdivide_cuts)
    boundary_cuts = cut_terrain_for_road_lowering(terrain_objects, segments)
    before_lowering = world_vertex_map(terrain_objects)
    stats_path = args.output_dir / f"{args.city}_road_lowering_stats.json"
    stats = lower_terrain_under_osm_roads(
        terrain_objects,
        str(osm),
        depth=args.depth,
        cell_size=args.cell_size,
        stats_path=str(stats_path),
        subdivision=subdivision,
        boundary_cuts=boundary_cuts,
        segments=segments,
    )
    deltas = vertex_deltas(terrain_objects, before_lowering)
    classify_faces(terrain_objects, deltas, args.depth)
    profile = best_segment_profile(terrain_objects, before_lowering, deltas, segments, args.depth)
    add_profile_markers(profile)
    render_path = args.output_dir / "road_lowering_curb_close.png"
    render(profile, render_path, args.width, args.height)
    qa = {
        "city": args.city,
        "depth": args.depth,
        "stats_path": str(stats_path),
        "render_path": str(render_path),
        "road_segments": stats.get("road_segments", 0),
        "total_vertices": stats.get("total_vertices", 0),
        "total_lowered_vertices": stats.get("total_lowered_vertices", 0),
        "profile": profile,
    }
    qa_path = args.output_dir / "road_lowering_curb_qa.json"
    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(qa, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
