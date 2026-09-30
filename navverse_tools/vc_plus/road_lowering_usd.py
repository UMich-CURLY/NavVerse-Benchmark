#!/usr/bin/env python3
"""Lower road faces in a USD terrain mesh and stitch road edges with vertical faces."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from terrain_regions import build_segment_grid, parse_osm_regions, point_hits_segment, source_face_materials


def road_category_at(
    x: float,
    y: float,
    segments: list[Any],
    segment_grid: dict[tuple[int, int], list[int]],
    cell_size: float,
) -> str | None:
    safe_cell = max(float(cell_size or 25.0), 1.0)
    candidates = segment_grid.get((math.floor(x / safe_cell), math.floor(y / safe_cell)), [])
    for category in ("crosswalk", "road"):
        for index in candidates:
            segment = segments[index]
            if segment.category == category and point_hits_segment(x, y, segment):
                return category
    return None


def face_offsets(counts: list[int]) -> list[int]:
    offsets = []
    offset = 0
    for count in counts:
        offsets.append(offset)
        offset += count
    return offsets


def face_centers(points: list[Gf.Vec3f], counts: list[int], indices: list[int]) -> list[Gf.Vec3d]:
    centers = []
    offsets = face_offsets(counts)
    for face_index, count in enumerate(counts):
        center = Gf.Vec3d(0.0, 0.0, 0.0)
        start = offsets[face_index]
        for local_index in range(count):
            point = points[indices[start + local_index]]
            center += Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
        centers.append(center / max(count, 1))
    return centers


def old_face_material_lookup(material_faces: dict[str, list[int]]) -> dict[int, str]:
    lookup = {}
    for material_path, faces in material_faces.items():
        for face in faces:
            lookup[int(face)] = material_path
    return lookup


def create_material_subsets(
    stage: Usd.Stage,
    mesh_prim: Usd.Prim,
    material_faces: dict[str, list[int]],
    wall_faces: list[int],
    wall_material_path: str | None,
) -> None:
    for child in list(mesh_prim.GetChildren()):
        if child.GetTypeName() == "GeomSubset":
            stage.RemovePrim(child.GetPath())
    for material_path, faces in material_faces.items():
        if not faces:
            continue
        subset = UsdGeom.Subset.Define(stage, f"{mesh_prim.GetPath().pathString}/{Sdf.Path(material_path).name}")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(faces)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if material:
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    if wall_faces and wall_material_path:
        subset = UsdGeom.Subset.Define(stage, f"{mesh_prim.GetPath().pathString}/NavVerse_RoadLoweringWalls")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(wall_faces)
        material = UsdShade.Material(stage.GetPrimAtPath(wall_material_path))
        if material:
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)


def set_extent(mesh: UsdGeom.Mesh, points: list[Gf.Vec3f]) -> None:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    zs = [float(point[2]) for point in points]
    mesh.CreateExtentAttr([Gf.Vec3f(min(xs), min(ys), min(zs)), Gf.Vec3f(max(xs), max(ys), max(zs))])


def face_normal(points: list[Gf.Vec3f], face_indices: list[int]) -> Gf.Vec3f:
    if len(face_indices) < 3:
        return Gf.Vec3f(0.0, 0.0, 1.0)
    origin = Gf.Vec3d(points[face_indices[0]])
    for i in range(1, len(face_indices) - 1):
        a = Gf.Vec3d(points[face_indices[i]]) - origin
        b = Gf.Vec3d(points[face_indices[i + 1]]) - origin
        normal = Gf.Cross(a, b)
        if normal.GetLength() > 1e-9:
            normal.Normalize()
            return Gf.Vec3f(float(normal[0]), float(normal[1]), float(normal[2]))
    return Gf.Vec3f(0.0, 0.0, 1.0)


def face_varying_normals(points: list[Gf.Vec3f], counts: list[int], indices: list[int]) -> list[Gf.Vec3f]:
    normals: list[Gf.Vec3f] = []
    offset = 0
    for count in counts:
        face_indices = indices[offset : offset + count]
        normal = face_normal(points, face_indices)
        normals.extend([normal] * count)
        offset += count
    return normals


def lower_roads_in_stage(
    stage: Usd.Stage,
    osm_path: Path,
    *,
    terrain_path: str = "/World/city/Terrain/Terrain",
    depth: float = 0.2,
    cell_size: float = 25.0,
    force: bool = False,
) -> dict[str, Any]:
    terrain_prim = stage.GetPrimAtPath(terrain_path)
    if not terrain_prim or terrain_prim.GetTypeName() != "Mesh":
        raise RuntimeError(f"terrain mesh not found: {terrain_path}")
    existing = terrain_prim.GetCustomDataByKey("navverse_usd_road_lowering_depth")
    if existing is not None and not force:
        raise RuntimeError(f"{terrain_path} already has USD road lowering metadata; pass --force to reapply")

    mesh = UsdGeom.Mesh(terrain_prim)
    points = list(mesh.GetPointsAttr().Get() or [])
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if not points or not counts or not indices:
        raise RuntimeError(f"terrain mesh has no authored geometry: {terrain_path}")

    segments, _ = parse_osm_regions(osm_path)
    road_segments = [segment for segment in segments if segment.category in {"road", "crosswalk"}]
    segment_grid = build_segment_grid(road_segments, cell_size)
    centers = face_centers(points, counts, indices)
    face_categories = [
        road_category_at(float(center[0]), float(center[1]), road_segments, segment_grid, cell_size)
        for center in centers
    ]
    road_faces = {index for index, category in enumerate(face_categories) if category in {"road", "crosswalk"}}

    offsets = face_offsets(counts)
    uv_attr = terrain_prim.GetAttribute("primvars:UVMap")
    uv_values = list(uv_attr.Get() or []) if uv_attr else []
    material_faces = source_face_materials(terrain_prim, len(counts))
    face_to_material = old_face_material_lookup(material_faces)
    road_material_counter = Counter(face_to_material.get(face) for face in road_faces)
    road_material_counter.pop(None, None)
    wall_material_path = road_material_counter.most_common(1)[0][0] if road_material_counter else next(iter(material_faces.keys()), None)

    new_points: list[Gf.Vec3f] = []
    new_counts: list[int] = []
    new_indices: list[int] = []
    new_uvs: list[Any] = []
    new_material_faces: dict[str, list[int]] = defaultdict(list)
    lowered_vertex_records: dict[tuple[int, int], int] = {}

    safe_depth = max(float(depth or 0.0), 0.0)
    for face_index, count in enumerate(counts):
        material_path = face_to_material.get(face_index)
        new_face_index = len(new_counts)
        if material_path:
            new_material_faces[material_path].append(new_face_index)
        new_counts.append(count)
        start = offsets[face_index]
        lower = safe_depth if face_index in road_faces else 0.0
        for local_index in range(count):
            old_vertex_index = indices[start + local_index]
            point = points[old_vertex_index]
            new_point = Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]) - lower)
            new_points.append(new_point)
            new_indices.append(len(new_points) - 1)
            if uv_values:
                new_uvs.append(uv_values[start + local_index])
            if face_index in road_faces:
                lowered_vertex_records[(face_index, old_vertex_index)] = len(new_points) - 1

    edge_faces: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for face_index, count in enumerate(counts):
        start = offsets[face_index]
        face_vertices = [indices[start + local_index] for local_index in range(count)]
        for local_index, a in enumerate(face_vertices):
            b = face_vertices[(local_index + 1) % count]
            edge_faces[tuple(sorted((a, b)))].append((face_index, a, b))

    wall_faces: list[int] = []

    def add_wall_quad(top_a: Gf.Vec3f, top_b: Gf.Vec3f, bottom_b: Gf.Vec3f, bottom_a: Gf.Vec3f) -> None:
        wall_faces.append(len(new_counts))
        new_counts.append(4)
        for point in (top_a, top_b, bottom_b, bottom_a):
            new_points.append(point)
            new_indices.append(len(new_points) - 1)
            new_uvs.append(Gf.Vec2f(float(point[0]) * 0.05, float(point[1]) * 0.05))

    for edge, records in edge_faces.items():
        if len(records) != 2:
            continue
        left, right = records
        left_is_road = left[0] in road_faces
        right_is_road = right[0] in road_faces
        if left_is_road == right_is_road:
            continue
        road_record = left if left_is_road else right
        nonroad_record = right if left_is_road else left
        road_face, road_a, road_b = road_record
        _, top_a_old, top_b_old = nonroad_record
        road_a_new = lowered_vertex_records.get((road_face, road_a))
        road_b_new = lowered_vertex_records.get((road_face, road_b))
        if road_a_new is None or road_b_new is None:
            continue
        top_a = Gf.Vec3f(*points[top_a_old])
        top_b = Gf.Vec3f(*points[top_b_old])
        bottom_b = new_points[road_b_new]
        bottom_a = new_points[road_a_new]
        add_wall_quad(top_a, top_b, bottom_b, bottom_a)

    mesh.CreatePointsAttr(new_points)
    mesh.CreateFaceVertexCountsAttr(new_counts)
    mesh.CreateFaceVertexIndicesAttr(new_indices)
    mesh.CreateNormalsAttr(face_varying_normals(new_points, new_counts, new_indices))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    if uv_values:
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "UVMap",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        primvar.Set(new_uvs)
    set_extent(mesh, new_points)
    create_material_subsets(stage, terrain_prim, dict(new_material_faces), wall_faces, wall_material_path)
    terrain_prim.SetCustomDataByKey("navverse_usd_road_lowering_depth", safe_depth)
    terrain_prim.SetCustomDataByKey("navverse_usd_road_lowering_wall_faces", len(wall_faces))
    terrain_prim.SetCustomDataByKey("navverse_usd_road_lowering_road_faces", len(road_faces))
    return {
        "terrain_path": terrain_path,
        "osm_path": str(osm_path),
        "depth": safe_depth,
        "original_face_count": len(counts),
        "new_face_count": len(new_counts),
        "original_point_count": len(points),
        "new_point_count": len(new_points),
        "road_face_count": len(road_faces),
        "wall_face_count": len(wall_faces),
        "wall_material_path": wall_material_path,
    }


def repair_usd(path: Path, osm_path: Path, depth: float, force: bool = False) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")
    summary = lower_roads_in_stage(stage, osm_path, depth=depth, force=force)
    stage.GetRootLayer().Save()
    return {"usd": str(path), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lower USD terrain road faces and stitch road edges.")
    parser.add_argument("usd", nargs="+", type=Path)
    parser.add_argument("--osm", type=Path, required=True)
    parser.add_argument("--depth", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    summaries = [repair_usd(path, args.osm, args.depth, force=args.force) for path in args.usd]
    text = json.dumps(summaries, indent=2, sort_keys=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
