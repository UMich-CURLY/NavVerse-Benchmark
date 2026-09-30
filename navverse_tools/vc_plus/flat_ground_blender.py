#!/usr/bin/env python3
"""Compress VC city ground relief and realign outdoor geometry.

Run with Blender, for example:
blender --background --python navverse_tools/vc_plus/flat_ground_blender.py -- \
  --data-root navverse_data/vc_plus --output-root navverse_data/vc_plus_smoke \
  --cities AMSTERDAM --height-scale 0.1 --max-buildings 5 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import bmesh
import bpy
import mathutils
import numpy as np


CITY_EXCLUDES = {"generated", "objects", "road_objects", "storefront", "__pycache__"}
ROAD_CURB_WIDTH_M = 0.2
ROAD_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "living_street",
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compress VC Full terrain relief and align buildings/objects.")
    parser.add_argument("--data-root", type=Path, default=repo_root / "navverse_data" / "vc_plus")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cities", nargs="+", default=None, help="City names, or 'all'.")
    parser.add_argument("--in-place", action="store_true", help="Modify the input city directories directly.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output city directory.")
    parser.add_argument(
        "--ground-height",
        default="auto",
        help="'auto'/'median' for per-city median height, 'min' for minimum terrain height, or a numeric z value.",
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=0.1,
        help="Scale height differences around ground height. Use 0.1 for 10x flatter relief, 0 for fully flat.",
    )
    parser.add_argument("--road-depth", type=float, default=0.2, help="Meters to lower drivable OSM road corridors.")
    parser.add_argument("--road-cell-size", type=float, default=25.0)
    parser.add_argument("--terrain-subdivide-cuts", type=int, default=0)
    parser.add_argument(
        "--road-boundary-cuts",
        action="store_true",
        help="Add bmesh cuts along OSM road and curb-strip boundaries before lowering road faces. Enabled by default.",
    )
    parser.add_argument(
        "--skip-road-boundary-cuts",
        action="store_true",
        help="Disable road boundary cuts. This is faster but can create jagged road lowering on coarse terrain.",
    )
    parser.set_defaults(road_boundary_cuts=True)
    parser.add_argument("--building-offset", type=float, default=0.0)
    parser.add_argument("--building-base-percentile", type=float, default=1.0)
    parser.add_argument("--max-buildings", type=int, default=None, help="Smoke-test limit for building GLBs.")
    parser.add_argument("--skip-buildings", action="store_true")
    parser.add_argument("--skip-terrain", action="store_true")
    parser.add_argument("--skip-json", action="store_true")
    parser.add_argument("--skip-height-field", action="store_true")
    parser.add_argument("--skip-osm", action="store_true")
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def discover_cities(data_root: Path, requested: list[str] | None) -> list[str]:
    if requested and len(requested) == 1 and requested[0].lower() == "all":
        return sorted(p.name for p in data_root.iterdir() if p.is_dir() and p.name not in CITY_EXCLUDES)
    if requested:
        return requested
    return sorted(p.name for p in data_root.iterdir() if p.is_dir() and p.name not in CITY_EXCLUDES)


def copy_city(src: Path, dst: Path, overwrite: bool) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output city exists, pass --overwrite to replace it: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def restore_original_height_field(src_dir: Path, dst_dir: Path) -> Path | None:
    backups = sorted(src_dir.glob("height_field.bak_flat_ground_*.npz"))
    source = backups[0] if backups else src_dir / "height_field.npz"
    if not source.exists():
        return None
    target = dst_dir / "height_field.npz"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return source


def auto_ground_height(city_dir: Path) -> float:
    height_path = city_dir / "height_field.npz"
    if height_path.exists():
        data = np.load(height_path)
        heights = np.asarray(data["terrain_alt"], dtype=float).reshape(-1)
        heights = heights[np.isfinite(heights)]
        if len(heights):
            return float(np.median(heights))
    terrain_path = city_dir / "terrain.glb"
    if terrain_path.exists():
        objects = import_glb(terrain_path)
        zs = [
            float((obj.matrix_world @ vertex.co).z)
            for obj in objects
            if obj.type == "MESH"
            for vertex in obj.data.vertices
        ]
        clear_scene()
        if zs:
            return float(np.median(np.asarray(zs, dtype=float)))
    return 0.0


def min_ground_height(city_dir: Path) -> float:
    height_path = city_dir / "height_field.npz"
    if height_path.exists():
        data = np.load(height_path)
        heights = np.asarray(data["terrain_alt"], dtype=float).reshape(-1)
        heights = heights[np.isfinite(heights)]
        if len(heights):
            return float(np.min(heights))
    terrain_path = city_dir / "terrain.glb"
    if terrain_path.exists():
        objects = import_glb(terrain_path)
        zs = [
            float((obj.matrix_world @ vertex.co).z)
            for obj in objects
            if obj.type == "MESH"
            for vertex in obj.data.vertices
        ]
        clear_scene()
        if zs:
            return float(np.min(np.asarray(zs, dtype=float)))
    return 0.0


def compressed_height(z: float, ground_height: float, height_scale: float) -> float:
    return ground_height + (float(z) - ground_height) * height_scale


def parse_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    feet = text.endswith("'") or text.lower().endswith("ft")
    if text.endswith("'"):
        text = text[:-1]
    elif text.lower().endswith("ft"):
        text = text[:-2]
    elif text.lower().endswith("m"):
        text = text[:-1]
    try:
        number = float(text.strip())
    except ValueError:
        return default
    return number * 0.3048 if feet else number


def terrain_region_width(tags: dict[str, str]) -> float:
    width = parse_float(tags.get("width") or tags.get("width:carriageway"), None)
    if width and width > 0:
        return float(width)
    lanes = int(parse_float(tags.get("lanes"), 1.0) or 1)
    highway = tags.get("highway")
    lane_width = 3.75 if highway in {"motorway", "trunk"} else 3.0 if highway in {"residential", "service"} else 3.4
    return max(lanes, 1) * lane_width


def is_drivable_road(tags: dict[str, str]) -> bool:
    highway = tags.get("highway")
    return highway in ROAD_HIGHWAYS and tags.get("area") != "yes"


def osm_path_for(city_dir: Path) -> Path | None:
    for path in [city_dir / "road_data" / "road_data_with_height.osm", city_dir / "road_data" / "road_data.osm"]:
        if path.exists():
            return path
    return None


def build_road_segments(osm_path: Path | None) -> tuple[list[dict[str, Any]], set[str]]:
    if osm_path is None or not osm_path.exists():
        return [], set()
    root = ET.parse(osm_path).getroot()
    nodes: dict[str, tuple[float | None, float | None]] = {}
    ways = []
    road_node_ids: set[str] = set()
    for element in root:
        if element.tag == "node":
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            nodes[element.get("id")] = (parse_float(tags.get("utm_x"), None), parse_float(tags.get("utm_y"), None))
        elif element.tag == "way":
            refs = [nd.get("ref") for nd in element.findall("nd")]
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            ways.append((refs, tags))

    segments = []
    for refs, tags in ways:
        if not is_drivable_road(tags):
            continue
        width = terrain_region_width(tags)
        road_node_ids.update(ref for ref in refs if ref)
        for start_ref, end_ref in zip(refs, refs[1:]):
            start = nodes.get(start_ref)
            end = nodes.get(end_ref)
            if not start or not end:
                continue
            sx, sy = start
            ex, ey = end
            if sx is None or sy is None or ex is None or ey is None:
                continue
            dx, dy = ex - sx, ey - sy
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-8:
                continue
            half_width = max(width * 0.5, 0.1)
            segments.append(
                {
                    "start": (sx, sy),
                    "end": (ex, ey),
                    "delta": (dx, dy),
                    "length_sq": length_sq,
                    "half_width": half_width,
                    "bounds": (
                        min(sx, ex) - half_width,
                        min(sy, ey) - half_width,
                        max(sx, ex) + half_width,
                        max(sy, ey) + half_width,
                    ),
                }
            )
    return segments, road_node_ids


def build_segment_grid(segments: list[dict[str, Any]], cell_size: float) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    safe_cell = max(float(cell_size or 25.0), 1.0)
    for index, segment in enumerate(segments):
        min_x, min_y, max_x, max_y = segment["bounds"]
        for ix in range(math.floor(min_x / safe_cell), math.floor(max_x / safe_cell) + 1):
            for iy in range(math.floor(min_y / safe_cell), math.floor(max_y / safe_cell) + 1):
                grid.setdefault((ix, iy), []).append(index)
    return grid


def point_hits_segment(x: float, y: float, segment: dict[str, Any]) -> bool:
    return point_hits_segment_width(x, y, segment, segment["half_width"])


def point_hits_segment_width(x: float, y: float, segment: dict[str, Any], half_width: float) -> bool:
    sx, sy = segment["start"]
    dx, dy = segment["delta"]
    t = ((x - sx) * dx + (y - sy) * dy) / segment["length_sq"]
    t = max(0.0, min(1.0, t))
    px, py = sx + t * dx, sy + t * dy
    dist_sq = (x - px) * (x - px) + (y - py) * (y - py)
    return dist_sq <= half_width * half_width


def road_lower_transition_width(half_width: float) -> float:
    return min(ROAD_CURB_WIDTH_M, max(0.01, half_width * 0.5))


def road_lower_inner_half_width(segment: dict[str, Any]) -> float:
    half_width = float(segment["half_width"])
    return max(half_width - road_lower_transition_width(half_width), half_width * 0.5)


def point_hits_any_road(
    x: float,
    y: float,
    segments: list[dict[str, Any]],
    grid: dict[tuple[int, int], list[int]],
    cell_size: float,
    *,
    inner_only: bool = False,
) -> bool:
    if not segments:
        return False
    safe_cell = max(float(cell_size or 25.0), 1.0)
    cell = (math.floor(x / safe_cell), math.floor(y / safe_cell))
    for index in grid.get(cell, []):
        segment = segments[index]
        half_width = road_lower_inner_half_width(segment) if inner_only else segment["half_width"]
        if point_hits_segment_width(x, y, segment, half_width):
            return True
    return False


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)


def import_glb(path: Path) -> list[bpy.types.Object]:
    clear_scene()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.data.objects if obj not in before]


def export_glb(path: Path) -> None:
    ensure_dir(path.parent)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=False)


def terrain_surface_z_threshold(obj: bpy.types.Object) -> float:
    zs = sorted(float((obj.matrix_world @ vertex.co).z) for vertex in obj.data.vertices)
    if len(zs) < 2:
        return float("-inf")
    for index in range(len(zs) - 1):
        if zs[index + 1] - zs[index] >= 2.0:
            return (zs[index] + zs[index + 1]) * 0.5
    return min(zs) - 1.0


def subdivide_mesh(obj: bpy.types.Object, cuts: int) -> dict[str, int] | None:
    safe_cuts = max(int(cuts or 0), 0)
    if safe_cuts <= 0 or obj.type != "MESH" or not obj.data.polygons:
        return None
    mesh = obj.data
    before_vertices = len(mesh.vertices)
    before_faces = len(mesh.polygons)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=safe_cuts, use_grid_fill=True)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return {
        "vertices_before": before_vertices,
        "vertices_after": len(mesh.vertices),
        "faces_before": before_faces,
        "faces_after": len(mesh.polygons),
    }


def weld_mesh_vertices(obj: bpy.types.Object, distance: float = 1e-5) -> dict[str, Any] | None:
    if obj.type != "MESH" or not obj.data or not obj.data.polygons:
        return None
    mesh = obj.data
    before_vertices = len(mesh.vertices)
    before_faces = len(mesh.polygons)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=max(float(distance), 0.0))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    after_vertices = len(mesh.vertices)
    if after_vertices == before_vertices:
        return None
    summary = {
        "distance": distance,
        "vertices_before": before_vertices,
        "vertices_after": after_vertices,
        "faces_before": before_faces,
        "faces_after": len(mesh.polygons),
        "merged_vertices": before_vertices - after_vertices,
    }
    print(
        f"[INFO] Welded terrain {obj.name}: "
        f"vertices {before_vertices}->{after_vertices}, merged={summary['merged_vertices']}"
    )
    return summary


def face_overlaps_xy_bbox(
    obj: bpy.types.Object,
    face: bmesh.types.BMFace,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    surface_z_threshold: float,
) -> bool:
    world_points = [obj.matrix_world @ vert.co for vert in face.verts]
    max_z = max(float(point.z) for point in world_points)
    if max_z < surface_z_threshold:
        return False
    face_min_x = min(float(point.x) for point in world_points)
    face_max_x = max(float(point.x) for point in world_points)
    face_min_y = min(float(point.y) for point in world_points)
    face_max_y = max(float(point.y) for point in world_points)
    return not (
        face_max_x < min_x
        or face_min_x > max_x
        or face_max_y < min_y
        or face_min_y > max_y
    )


def local_bisect_plane(
    obj: bpy.types.Object,
    point_xy: tuple[float, float],
    normal_xy: tuple[float, float],
) -> tuple[mathutils.Vector | None, mathutils.Vector | None]:
    inv = obj.matrix_world.inverted()
    normal_matrix = obj.matrix_world.inverted().transposed().to_3x3()
    plane_co = inv @ mathutils.Vector((float(point_xy[0]), float(point_xy[1]), 0.0))
    plane_no = normal_matrix @ mathutils.Vector((float(normal_xy[0]), float(normal_xy[1]), 0.0))
    if plane_no.length <= 1e-9:
        return None, None
    plane_no.normalize()
    return plane_co, plane_no


def collect_bisect_geom(
    obj: bpy.types.Object,
    bm: bmesh.types.BMesh,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    surface_z_threshold: float,
) -> list[Any]:
    faces = [
        face
        for face in bm.faces
        if face_overlaps_xy_bbox(obj, face, min_x, min_y, max_x, max_y, surface_z_threshold)
    ]
    if not faces:
        return []
    geom = set(faces)
    for face in faces:
        geom.update(face.edges)
        geom.update(face.verts)
    return list(geom)


def cut_terrain_for_road_lowering(
    terrain_objects: list[bpy.types.Object],
    segments: list[dict[str, Any]],
    margin: float = 1.0,
) -> list[dict[str, Any]]:
    if not segments:
        return []
    summaries = []
    for obj in terrain_objects:
        if obj.type != "MESH" or not obj.data or not obj.data.polygons:
            continue
        mesh = obj.data
        before_vertices = len(mesh.vertices)
        before_faces = len(mesh.polygons)
        surface_z_threshold = terrain_surface_z_threshold(obj)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        total_cuts = 0
        cut_vertices = 0
        for segment in segments:
            sx, sy = segment["start"]
            ex, ey = segment["end"]
            dx, dy = segment["delta"]
            length = math.sqrt(segment["length_sq"])
            if length <= 1e-6:
                continue
            tx, ty = dx / length, dy / length
            nx, ny = -ty, tx
            half_width = float(segment["half_width"])
            inner_half_width = road_lower_inner_half_width(segment)
            pad = half_width + max(float(margin), 0.0)
            min_x, max_x = min(sx, ex) - pad, max(sx, ex) + pad
            min_y, max_y = min(sy, ey) - pad, max(sy, ey) + pad
            geom = collect_bisect_geom(obj, bm, min_x, min_y, max_x, max_y, surface_z_threshold)
            if not geom:
                continue
            offsets = sorted({-half_width, -inner_half_width, inner_half_width, half_width})
            for offset in offsets:
                point_xy = (sx + nx * offset, sy + ny * offset)
                normal_xy = (nx, ny)
                plane_co, plane_no = local_bisect_plane(obj, point_xy, normal_xy)
                if plane_co is None or plane_no is None:
                    continue
                try:
                    result = bmesh.ops.bisect_plane(
                        bm,
                        geom=geom,
                        plane_co=plane_co,
                        plane_no=plane_no,
                        clear_inner=False,
                        clear_outer=False,
                    )
                except Exception as exc:
                    print(f"[WARNING] Road boundary cut failed for {obj.name}: {exc}")
                    continue
                total_cuts += 1
                cut_geom = result.get("geom_cut", [])
                cut_vertices += len(cut_geom)
                expanded_geom = {element for element in geom if element.is_valid}
                for element in cut_geom:
                    if not element.is_valid:
                        continue
                    expanded_geom.add(element)
                    if isinstance(element, bmesh.types.BMVert):
                        expanded_geom.update(face for face in element.link_faces if face.is_valid)
                        expanded_geom.update(edge for edge in element.link_edges if edge.is_valid)
                    elif isinstance(element, bmesh.types.BMEdge):
                        expanded_geom.update(face for face in element.link_faces if face.is_valid)
                        expanded_geom.update(vertex for vertex in element.verts if vertex.is_valid)
                geom = list(expanded_geom)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        summary = {
            "name": obj.name,
            "vertices_before": before_vertices,
            "vertices_after": len(mesh.vertices),
            "faces_before": before_faces,
            "faces_after": len(mesh.polygons),
            "bisect_calls": total_cuts,
            "cut_geom_count": cut_vertices,
            "surface_z_threshold": surface_z_threshold,
        }
        summaries.append(summary)
        print(
            f"[INFO] Cut terrain {obj.name} along OSM road strips: "
            f"vertices {before_vertices}->{summary['vertices_after']}, "
            f"faces {before_faces}->{summary['faces_after']}, "
            f"bisect_calls={total_cuts}"
        )
    return summaries


def flatten_terrain_glb(
    path: Path,
    ground_height: float,
    height_scale: float,
    road_depth: float,
    segments: list[dict[str, Any]],
    cell_size: float,
    subdivide_cuts: int,
    road_boundary_cuts: bool,
) -> dict[str, Any]:
    objects = import_glb(path)
    grid = build_segment_grid(segments, cell_size)
    stats: dict[str, Any] = {
        "path": str(path),
        "ground_height": ground_height,
        "height_scale": height_scale,
        "road_depth": road_depth,
        "road_segments": len(segments),
        "road_curb_width": ROAD_CURB_WIDTH_M,
        "road_lowering_mode": "face_center_inner_road_corridor",
        "road_boundary_cuts_enabled": road_boundary_cuts,
        "objects": [],
        "total_surface_vertices": 0,
        "total_road_vertices": 0,
        "total_road_faces": 0,
        "total_welded_vertices": 0,
    }
    for obj in objects:
        if obj.type != "MESH" or not obj.data.vertices:
            continue
        weld = weld_mesh_vertices(obj)
        subdivision = subdivide_mesh(obj, subdivide_cuts)
        boundary_cuts = cut_terrain_for_road_lowering([obj], segments) if road_boundary_cuts else []
        threshold = terrain_surface_z_threshold(obj)
        inv = obj.matrix_world.inverted()
        surface_vertices = 0
        road_vertices = 0
        road_faces = 0
        min_before = float("inf")
        max_before = float("-inf")
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            min_before = min(min_before, float(world.z))
            max_before = max(max_before, float(world.z))
            if float(world.z) < threshold:
                continue
            surface_vertices += 1
            world.z = compressed_height(float(world.z), ground_height, height_scale)
            vertex.co = inv @ world
        obj.data.update()

        lowered_indices: set[int] = set()
        safe_cell = max(float(cell_size or 25.0), 1.0)
        for polygon in obj.data.polygons:
            face_world_points = [obj.matrix_world @ obj.data.vertices[index].co for index in polygon.vertices]
            if not face_world_points or max(point.z for point in face_world_points) < threshold:
                continue
            center = sum(face_world_points, mathutils.Vector((0.0, 0.0, 0.0))) / len(face_world_points)
            cell = (math.floor(center.x / safe_cell), math.floor(center.y / safe_cell))
            if not any(
                point_hits_segment_width(
                    center.x,
                    center.y,
                    segments[index],
                    road_lower_inner_half_width(segments[index]),
                )
                for index in grid.get(cell, [])
            ):
                continue
            road_faces += 1
            lowered_indices.update(polygon.vertices)
        for vertex_index in lowered_indices:
            world = obj.matrix_world @ obj.data.vertices[vertex_index].co
            world.z -= road_depth
            obj.data.vertices[vertex_index].co = inv @ world
        if lowered_indices:
            obj.data.update()
        road_vertices = len(lowered_indices)

        stats["total_surface_vertices"] += surface_vertices
        stats["total_road_vertices"] += road_vertices
        stats["total_road_faces"] += road_faces
        if weld:
            stats["total_welded_vertices"] += int(weld["merged_vertices"])
        stats["objects"].append(
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "surface_vertices": surface_vertices,
                "road_vertices": road_vertices,
                "road_faces": road_faces,
                "surface_z_threshold": threshold,
                "z_before_min": min_before,
                "z_before_max": max_before,
                "weld": weld,
                "subdivision": subdivision,
                "boundary_cuts": boundary_cuts,
            }
        )
    export_glb(path)
    clear_scene()
    return stats


def object_base_z(objects: list[bpy.types.Object], percentile: float) -> float | None:
    zs = [
        float((obj.matrix_world @ vertex.co).z)
        for obj in objects
        if obj.type == "MESH"
        for vertex in obj.data.vertices
    ]
    if not zs:
        return None
    return float(np.percentile(np.asarray(zs, dtype=float), percentile))


def shift_glb_to_ground(
    path: Path,
    ground_height: float,
    height_scale: float,
    offset: float,
    base_percentile: float,
) -> dict[str, Any]:
    objects = import_glb(path)
    meshes = [obj for obj in objects if obj.type == "MESH" and obj.data.vertices]
    base_z = object_base_z(meshes, base_percentile)
    stats = {"path": str(path), "mesh_objects": len(meshes), "base_z": base_z, "delta_z": 0.0}
    if base_z is None:
        clear_scene()
        return stats
    target_base_z = compressed_height(base_z, ground_height, height_scale) + offset
    delta = target_base_z - base_z
    for obj in meshes:
        inv = obj.matrix_world.inverted()
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            world.z += delta
            vertex.co = inv @ world
        obj.data.update()
    export_glb(path)
    clear_scene()
    stats["delta_z"] = delta
    stats["target_base_z"] = target_base_z
    return stats


def update_height_field(
    path: Path,
    ground_height: float,
    height_scale: float,
    road_depth: float,
    segments: list[dict[str, Any]],
    cell_size: float,
) -> dict[str, Any]:
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    before = np.asarray(arrays["terrain_alt"], dtype=float)
    after = ground_height + (before - ground_height) * height_scale
    road_points = 0
    if road_depth > 0.0 and segments and "plane_coord" in arrays:
        coords = np.asarray(arrays["plane_coord"], dtype=float).reshape(-1, 2)
        grid = build_segment_grid(segments, cell_size)
        flat_after = after.reshape(-1).copy()
        for index, (x, y) in enumerate(coords):
            if point_hits_any_road(float(x), -float(y), segments, grid, cell_size, inner_only=True):
                flat_after[index] -= road_depth
                road_points += 1
        after = flat_after.reshape(before.shape)
    arrays["terrain_alt"] = after.astype(before.dtype, copy=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, **arrays)
    tmp.with_suffix(tmp.suffix + ".npz").replace(path)
    return {
        "path": str(path),
        "z_before_min": float(np.nanmin(before)),
        "z_before_median": float(np.nanmedian(before)),
        "z_before_max": float(np.nanmax(before)),
        "z_after_min": float(np.nanmin(arrays["terrain_alt"])),
        "z_after_median": float(np.nanmedian(arrays["terrain_alt"])),
        "z_after_max": float(np.nanmax(arrays["terrain_alt"])),
        "height_scale": height_scale,
        "road_depth": road_depth,
        "road_lowered_points": road_points,
    }


def update_object_json(path: Path, ground_height: float, height_scale: float) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    changed = 0
    before_values = []
    for item in data.get("objects", []):
        loc = item.get("location")
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        if len(loc) >= 3:
            before_values.append(loc[2])
            loc[2] = compressed_height(float(loc[2]), ground_height, height_scale)
        else:
            loc.append(ground_height)
        changed += 1
    write_json(path, data)
    return {
        "path": str(path),
        "objects_changed": changed,
        "z_before_min": float(min(before_values)) if before_values else None,
        "z_before_max": float(max(before_values)) if before_values else None,
        "z_after_min": float(min(compressed_height(float(v), ground_height, height_scale) for v in before_values)) if before_values else ground_height,
        "z_after_max": float(max(compressed_height(float(v), ground_height, height_scale) for v in before_values)) if before_values else ground_height,
        "height_scale": height_scale,
    }


def set_node_height_tag(element: ET.Element, height: float) -> None:
    for tag in element.findall("tag"):
        if tag.get("k") == "height":
            tag.set("v", f"{height:.3f}")
            return
    tag = ET.SubElement(element, "tag")
    tag.set("k", "height")
    tag.set("v", f"{height:.3f}")


def node_height(element: ET.Element, fallback: float) -> float:
    for tag in element.findall("tag"):
        if tag.get("k") == "height":
            return float(parse_float(tag.get("v"), fallback))
    return fallback


def update_osm(
    path: Path,
    road_node_ids: set[str],
    ground_height: float,
    height_scale: float,
    road_depth: float,
) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    road_nodes = 0
    ground_nodes = 0
    for element in root:
        if element.tag != "node":
            continue
        target_height = compressed_height(node_height(element, ground_height), ground_height, height_scale)
        if element.get("id") in road_node_ids:
            set_node_height_tag(element, target_height - road_depth)
            road_nodes += 1
        else:
            set_node_height_tag(element, target_height)
            ground_nodes += 1
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return {"path": str(path), "road_nodes": road_nodes, "ground_nodes": ground_nodes}


def process_city(src_dir: Path, dst_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    copy_city(src_dir, dst_dir, args.overwrite)
    source_height_field = restore_original_height_field(src_dir, dst_dir)
    if args.ground_height in {"auto", "median"}:
        ground_height = auto_ground_height(dst_dir)
    elif args.ground_height == "min":
        ground_height = min_ground_height(dst_dir)
    else:
        ground_height = float(args.ground_height)
    osm_path = osm_path_for(dst_dir)
    segments, road_node_ids = build_road_segments(osm_path)
    summary: dict[str, Any] = {
        "city": dst_dir.name,
        "source_dir": str(src_dir),
        "output_dir": str(dst_dir),
        "ground_height": ground_height,
        "height_scale": args.height_scale,
        "road_depth": args.road_depth,
        "road_curb_width": ROAD_CURB_WIDTH_M,
        "started_at": time.strftime("%F %T %Z"),
        "road_segments": len(segments),
        "source_height_field": str(source_height_field) if source_height_field else None,
    }
    if not args.skip_height_field and (dst_dir / "height_field.npz").exists():
        summary["height_field"] = update_height_field(
            dst_dir / "height_field.npz",
            ground_height,
            args.height_scale,
            args.road_depth,
            segments,
            args.road_cell_size,
        )
    if not args.skip_osm and osm_path is not None:
        summary["osm"] = update_osm(osm_path, road_node_ids, ground_height, args.height_scale, args.road_depth)
    if not args.skip_json:
        object_summaries = []
        for json_path in [dst_dir / "objects" / "amenities_with_z.json", dst_dir / "objects" / "natural_with_z.json"]:
            if json_path.exists():
                object_summaries.append(update_object_json(json_path, ground_height, args.height_scale))
        summary["objects_json"] = object_summaries
    if not args.skip_terrain and (dst_dir / "terrain.glb").exists():
        summary["terrain"] = flatten_terrain_glb(
            dst_dir / "terrain.glb",
            ground_height,
            args.height_scale,
            args.road_depth,
            segments,
            args.road_cell_size,
            args.terrain_subdivide_cuts,
            args.road_boundary_cuts and not args.skip_road_boundary_cuts,
        )
    if not args.skip_buildings:
        building_paths = sorted((dst_dir / "buildings").glob("*.glb"))
        if args.max_buildings is not None:
            building_paths = building_paths[: args.max_buildings]
        building_stats = []
        for index, building_path in enumerate(building_paths, 1):
            print(f"[{dst_dir.name}] Aligning building {index}/{len(building_paths)}: {building_path.name}")
            building_stats.append(
                shift_glb_to_ground(
                    building_path,
                    ground_height,
                    args.height_scale,
                    args.building_offset,
                    args.building_base_percentile,
                )
            )
        summary["buildings"] = {"processed": len(building_stats), "items": building_stats[:50]}
    summary["finished_at"] = time.strftime("%F %T %Z")
    write_json(dst_dir / "flat_ground_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.in_place and args.output_root is not None:
        raise ValueError("Use either --in-place or --output-root, not both.")
    if not args.in_place and args.output_root is None:
        raise ValueError("Pass --output-root for non-destructive output, or --in-place to modify inputs.")
    data_root = args.data_root.resolve()
    output_root = data_root if args.in_place else args.output_root.resolve()
    cities = discover_cities(data_root, args.cities)
    print(f"[INFO] Flat-ground processing {len(cities)} city/cities from {data_root} to {output_root}")
    summaries = []
    for city in cities:
        src_dir = data_root / city
        if not src_dir.exists():
            print(f"[WARNING] Missing city directory: {src_dir}")
            continue
        dst_dir = output_root / city
        print(f"[INFO] Processing {city}")
        summaries.append(process_city(src_dir, dst_dir, args))
    ensure_dir(output_root)
    write_json(output_root / "flat_ground_run_summary.json", {"cities": summaries})
    print(f"[INFO] Wrote run summary: {output_root / 'flat_ground_run_summary.json'}")


if __name__ == "__main__":
    main()
