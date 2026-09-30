#!/usr/bin/env python3
"""Compose a final VC Full innout USD from a scored doorway plan."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import Any

import sys

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrain_utils import TerrainHeightQuery
from exit_signs import apply_exit_sign_to_stage
from storefront_collision import apply_collision_apis_under
from terrain_regions import (
    ClipVertex,
    RegionMeshData,
    append_polygon,
    carve_primitive,
    polygon_xy_area,
    primitive,
    source_face_materials,
    subtract_convex_polygon_from_mesh,
)
from preview_scene import add_innout_preview_cameras, add_preview_sky
from storefront_facade_planner import (
    edge_front_clearance_ok,
    load_building_edges,
    load_road_segments,
    nearest_road,
)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
except ImportError as exc:
    raise SystemExit("pxr is required; run this script from the navverse USD environment.") from exc


DEFAULT_STORE_ASSET = "restaurant_6"
_BUILDING_FOOTPRINT_CACHE: dict[str, list[dict[str, Any]]] = {}
_TERRAIN_MESH_CACHE: dict[str, trimesh.Trimesh | None] = {}
DOOR_CONNECTOR_OVERLAP_M = 0.05
THRESHOLD_START_DISTANCE_M = DOOR_CONNECTOR_OVERLAP_M
THRESHOLD_LANDING_DISTANCE_M = 0.35
THRESHOLD_OUTER_DISTANCE_M = 2.0
THRESHOLD_LATERAL_BLEND_M = 0.6
THRESHOLD_TERRAIN_STEP_LIMIT_M = 0.12
THRESHOLD_MIN_OUTER_DISTANCE_M = 0.65
THRESHOLD_BLEND_DISTANCE_M = 0.75
THRESHOLD_SURFACE_OFFSET_M = 0.002
THRESHOLD_BASE_CLEARANCE_M = 0.02
DOORWAY_TERRAIN_LONGITUDINAL_SAMPLES_M = (0.0, 0.5, 1.2)
DOORWAY_TERRAIN_LATERAL_FRACTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)
INNOUT_ROOM_OBJECT_CLEARANCE_M = 0.5
CUT_INTERIOR_MATERIAL_NAME = "NavVerseCutInterior"
INNOUT_FLOOR_BASE_TOLERANCE_M = 0.03
INNOUT_FLOOR_FLAT_TOLERANCE_M = 0.03
INNOUT_FLOOR_MAX_DECORATION_HEIGHT_M = 0.35
INNOUT_FLOOR_DUPLICATE_AREA_RATIO = 0.8
INNOUT_FLOOR_DUPLICATE_OVERLAP_RATIO = 0.9
INNOUT_FLOOR_MIN_SURFACE_COVERAGE_RATIO = 0.65
INNOUT_FLOOR_MIN_HULL_COVERAGE_RATIO = 0.65
INNOUT_COLLISION_ROLE_KEY = "navverse_collision_role"
INNOUT_VISUAL_ONLY_COLLISION_ROLE = "visual_only"
INNOUT_INTERIOR_OPENING_SIDE_CLEARANCE_M = 0.2
INNOUT_INTERIOR_OPENING_VERTICAL_CLEARANCE_M = 0.05
INNOUT_INTERIOR_CORRIDOR_SIDE_CLEARANCE_M = 0.35
INNOUT_INTERIOR_CORRIDOR_DEPTH_M = 5.0


@dataclass
class DoorPlan:
    city: str
    content_radius_m: float
    local_storefront_capacity_estimate: int
    base_usd: str
    output_usd: str
    building_glb: str
    door_cut_source_glb: str
    door_cut_source_z_up: bool
    door_cut_mode: str
    target_prim_path: str
    target_match_distance: float | None
    interior_scene: str
    interior_usd: str
    exterior_storefront_enabled: bool
    storefront_usd: str | None
    storefront_bbox_min: list[float]
    storefront_bbox_max: list[float]
    storefront_bbox_dims: list[float]
    storefront_raw_bbox_dims: list[float]
    storefront_scale_multiplier: float
    excluded_storefront_plan_indices: list[int]
    excluded_storefront_buildings: list[str]
    removed_storefront_prim_paths: list[str]
    interior_door_name: str
    interior_door_prim_path: str
    interior_door_bbox_min: list[float]
    interior_door_bbox_max: list[float]
    interior_door_center: list[float]
    interior_scene_bbox_min: list[float]
    interior_scene_bbox_max: list[float]
    interior_scene_bbox_dims: list[float]
    interior_fit_status: str
    interior_fit_outside_corner_count: int
    interior_fit_outside_distance_m: float
    interior_fit_inward_nudge_m: float
    interior_door_anchor_inside_building: bool
    interior_doorway_inset_deficit_m: float
    removed_interior_door_path: str
    interior_wall_opening_summary: dict[str, Any]
    removed_interior_obstacle_paths: list[str]
    removed_exterior_obstacle_paths: list[str]
    door_size_source: str
    door_center_xy: list[float]
    door_bottom_z: float
    door_width: float
    door_height: float
    door_depth: float
    interior_frontage_width: float
    interior_world_top_z: float
    building_opening_source: str
    building_opening_width: float
    building_opening_bottom_z: float
    building_opening_top_z: float
    terrain_z: float
    exterior_surface_z: float
    terrain_samples: list[list[float]]
    terrain_sample_z: list[float]
    exterior_public_surface_samples: list[dict[str, float | bool]]
    building_min_z: float
    building_footprint_xy: list[list[float]]
    terrain_hole_margin_m: float
    terrain_hole_summary: list[dict[str, Any]]
    threshold_summary: dict[str, Any]
    door_connector_summary: dict[str, Any]
    tangent_xy: list[float]
    outward_normal_xy: list[float]
    doorway_yaw_deg: float
    room_translate: list[float]
    room_yaw_deg: float
    room_scale: float
    room_effective_scale: float
    storefront_translate: list[float]
    storefront_yaw_deg: float
    cut_glb: str
    cut_usd: str
    qa_dir: str
    exterior_wall_opening_summary: dict[str, Any] = field(default_factory=dict)
    local_object_target_capacity_estimate: int = 0
    interior_floor_cleanup_summary: dict[str, Any] = field(default_factory=dict)
    terrain_height_summary: dict[str, Any] = field(default_factory=dict)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def normalize(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm <= 1e-9:
        raise ValueError(f"zero-length vector: {vec}")
    return arr / norm


def doorway_terrain_alignment(
    terrain_query: TerrainHeightQuery,
    center: np.ndarray,
    tangent: np.ndarray,
    outward: np.ndarray,
    door_width: float,
    fallback_z: float,
) -> tuple[list[np.ndarray], list[float], float]:
    half_width = float(door_width) * 0.5
    sample_xy = [
        center + outward * distance + tangent * (half_width * lateral_fraction)
        for distance in DOORWAY_TERRAIN_LONGITUDINAL_SAMPLES_M
        for lateral_fraction in DOORWAY_TERRAIN_LATERAL_FRACTIONS
    ]
    sample_z = [
        terrain_query.height_at(float(point[0]), float(point[1]), fallback_z)
        for point in sample_xy
    ]
    if not sample_z or not np.isfinite(sample_z).all():
        raise RuntimeError("could not sample finite terrain heights around the InNOut doorway")
    return sample_xy, sample_z, float(max(sample_z))


def frontage_terrain_samples(
    terrain_query: TerrainHeightQuery,
    center: np.ndarray,
    tangent: np.ndarray,
    outward: np.ndarray,
    opening_width: float,
    fallback_z: float,
) -> tuple[list[np.ndarray], list[float]]:
    half_width = float(opening_width) * 0.5
    sample_xy = [
        center,
        center - tangent * half_width,
        center + tangent * half_width,
        center + outward * 0.5,
        center + outward * 1.2,
        center - tangent * half_width + outward * 0.5,
        center + tangent * half_width + outward * 0.5,
        center - tangent * half_width + outward * 1.2,
        center + tangent * half_width + outward * 1.2,
    ]
    sample_z = [
        terrain_query.height_at(float(point[0]), float(point[1]), fallback_z)
        for point in sample_xy
    ]
    if not sample_z or not np.isfinite(sample_z).all():
        raise RuntimeError("could not sample finite terrain heights across the InNOut frontage")
    return sample_xy, sample_z


def terrain_height_summary(
    doorway_sample_xy: list[np.ndarray],
    doorway_sample_z: list[float],
    frontage_sample_z: list[float],
    door_width: float,
    opening_width: float,
    selected_z: float,
) -> dict[str, Any]:
    return {
        "selection": "doorway_local_max",
        "selected_z": float(selected_z),
        "door_width_m": float(door_width),
        "frontage_width_m": float(opening_width),
        "longitudinal_distances_m": list(DOORWAY_TERRAIN_LONGITUDINAL_SAMPLES_M),
        "lateral_fractions": list(DOORWAY_TERRAIN_LATERAL_FRACTIONS),
        "doorway_samples": [
            [float(point[0]), float(point[1]), float(height)]
            for point, height in zip(doorway_sample_xy, doorway_sample_z, strict=True)
        ],
        "doorway_min_z": float(min(doorway_sample_z)),
        "doorway_max_z": float(max(doorway_sample_z)),
        "frontage_min_z": float(min(frontage_sample_z)),
        "frontage_max_z": float(max(frontage_sample_z)),
    }


def realign_anchor_plan_to_doorway_terrain(
    args: argparse.Namespace,
    plan: DoorPlan,
) -> float:
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    terrain_query = TerrainHeightQuery.from_roots(
        args.data_root.resolve(),
        args.city,
        preferred_source=args.source_dataset,
    )
    doorway_sample_xy, doorway_sample_z, selected_z = doorway_terrain_alignment(
        terrain_query,
        center,
        tangent,
        outward,
        plan.door_width,
        plan.building_min_z,
    )
    frontage_sample_xy, frontage_sample_z = frontage_terrain_samples(
        terrain_query,
        center,
        tangent,
        outward,
        plan.building_opening_width,
        plan.building_min_z,
    )
    delta = float(selected_z - plan.door_bottom_z)
    plan.door_bottom_z = float(selected_z)
    plan.terrain_z = float(selected_z)
    plan.exterior_surface_z = float(selected_z)
    plan.room_translate[2] = float(plan.room_translate[2] + delta)
    plan.interior_world_top_z = float(plan.interior_world_top_z + delta)
    plan.building_opening_top_z = float(plan.building_opening_top_z + delta)
    if plan.storefront_translate:
        plan.storefront_translate[2] = float(plan.storefront_translate[2] + delta * 100.0)
    plan.terrain_samples = [
        [float(point[0]), float(point[1])] for point in frontage_sample_xy
    ]
    plan.terrain_sample_z = [float(value) for value in frontage_sample_z]
    plan.building_opening_bottom_z = float(
        min(frontage_sample_z[7], frontage_sample_z[8])
        + THRESHOLD_SURFACE_OFFSET_M
        - THRESHOLD_BASE_CLEARANCE_M
    )
    plan.terrain_height_summary = terrain_height_summary(
        doorway_sample_xy,
        doorway_sample_z,
        frontage_sample_z,
        plan.door_width,
        plan.building_opening_width,
        selected_z,
    )
    return delta


def yaw_to_map_xy(local_vec: np.ndarray, world_vec: np.ndarray) -> float:
    local = normalize(local_vec)
    world = normalize(world_vec)
    return math.degrees(math.atan2(world[1], world[0]) - math.atan2(local[1], local[0]))


def rotate_xy(point: np.ndarray, yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.asarray([c * point[0] - s * point[1], s * point[0] + c * point[1]], dtype=float)


def mapped_vertices_from_glb(glb_path: Path) -> np.ndarray:
    scene = trimesh.load(glb_path, force="scene")
    if hasattr(scene, "geometry"):
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh = scene
    vertices = np.asarray(mesh.vertices)
    return np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])


def mapped_trimesh_from_glb(glb_path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(glb_path, force="scene")
    if hasattr(scene, "geometry"):
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh = scene
    mapped = mesh.copy()
    vertices = np.asarray(mapped.vertices, dtype=float)
    mapped.vertices = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
    return mapped


def load_cut_mesh_data(glb_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[int]]:
    scene = trimesh.load(glb_path, force="scene", process=False)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    interior_faces: list[int] = []
    vertex_offset = 0
    face_offset = 0
    all_have_uvs = True

    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        part_vertices = np.asarray(geometry.vertices, dtype=float)
        part_faces = np.asarray(geometry.faces, dtype=np.int64)
        if not len(part_vertices) or not len(part_faces):
            continue
        material = getattr(getattr(geometry, "visual", None), "material", None)
        material_name = str(getattr(material, "name", "") or "")
        if material_name.startswith(CUT_INTERIOR_MATERIAL_NAME):
            interior_faces.extend(range(face_offset, face_offset + len(part_faces)))
        part_uvs = getattr(getattr(geometry, "visual", None), "uv", None)
        if part_uvs is None or len(part_uvs) != len(part_vertices):
            all_have_uvs = False
        else:
            uvs.append(np.asarray(part_uvs, dtype=float))
        vertices.append(part_vertices)
        faces.append(part_faces + vertex_offset)
        vertex_offset += len(part_vertices)
        face_offset += len(part_faces)

    if not vertices or not faces:
        raise RuntimeError(f"cut mesh is empty: {glb_path}")
    merged_uvs = np.concatenate(uvs) if all_have_uvs and len(uvs) == len(vertices) else None
    return np.concatenate(vertices), np.concatenate(faces), merged_uvs, interior_faces


def write_cut_usd_from_glb(
    glb_path: Path,
    usd_path: Path,
    *,
    require_interior_faces: bool = False,
) -> None:
    vertices, faces, uv, interior_faces = load_cut_mesh_data(glb_path)
    if require_interior_faces and not interior_faces:
        raise RuntimeError(f"cut mesh has no {CUT_INTERIOR_MATERIAL_NAME} faces: {glb_path}")

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    if usd_path.exists():
        usd_path.unlink()

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root = stage.DefinePrim("/cut_building", "Xform")
    stage.SetDefaultPrim(root)
    usd_mesh = UsdGeom.Mesh.Define(stage, "/cut_building/mesh")
    usd_mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
    usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
    usd_mesh.CreateFaceVertexIndicesAttr([int(index) for face in faces for index in face])
    usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
    if uv is not None and len(uv) == len(vertices):
        face_uv = [Gf.Vec2f(float(uv[index][0]), float(uv[index][1])) for face in faces for index in face]
        UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "UVMap",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        ).Set(face_uv)
    if interior_faces:
        subset = UsdGeom.Subset.Define(stage, "/cut_building/mesh/NavVerseCutInteriorFaces")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(interior_faces)
        material = UsdShade.Material.Define(stage, "/cut_building/Looks/NavVerseCutInterior")
        shader = UsdShade.Shader.Define(stage, "/cut_building/Looks/NavVerseCutInterior/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.72, 0.73, 0.74))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
    root.SetCustomDataByKey("navverse_cut_interior_face_count", len(interior_faces))
    stage.GetRootLayer().Save()


def first_bound_material_path(prim: Usd.Prim) -> Sdf.Path | None:
    for candidate in Usd.PrimRange(prim):
        if not candidate.IsA(UsdGeom.Mesh):
            continue
        targets = UsdShade.MaterialBindingAPI(candidate).GetDirectBindingRel().GetTargets()
        if targets:
            return targets[0]
    return None


def simplify_closed_footprint(points: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
    simplified = np.asarray(points, dtype=float)
    if len(simplified) > 3 and np.linalg.norm(simplified[0] - simplified[-1]) <= tolerance:
        simplified = simplified[:-1]
    changed = True
    while changed and len(simplified) > 3:
        changed = False
        kept = []
        for index, point in enumerate(simplified):
            previous = simplified[index - 1]
            following = simplified[(index + 1) % len(simplified)]
            span = following - previous
            length = float(np.linalg.norm(span))
            distance = abs(float(np.cross(span, point - previous))) / length if length > 1e-9 else 0.0
            if distance <= tolerance and float(np.dot(point - previous, following - point)) >= -1e-6:
                changed = True
                continue
            kept.append(point)
        if len(kept) >= 3:
            simplified = np.asarray(kept, dtype=float)
        else:
            break
    return simplified


def section_footprint(mesh: trimesh.Trimesh, zmin: float, zmax: float) -> np.ndarray | None:
    section_z = zmin + min(max(0.8, 0.08 * (zmax - zmin)), 2.0)
    section = mesh.section(
        plane_origin=np.asarray([0.0, 0.0, section_z], dtype=float),
        plane_normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    if section is None:
        return None
    closed = []
    for path in section.discrete:
        points = np.asarray(path, dtype=float)
        if len(points) < 4 or np.linalg.norm(points[0, :2] - points[-1, :2]) > 0.25:
            continue
        perimeter = float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())
        closed.append((perimeter, points[:, :2]))
    if not closed:
        return None
    return simplify_closed_footprint(max(closed, key=lambda item: item[0])[1])


def building_footprint(
    mapped_vertices: np.ndarray,
    mapped_mesh: trimesh.Trimesh | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    zmin = float(mapped_vertices[:, 2].min())
    zmax = float(mapped_vertices[:, 2].max())
    if mapped_mesh is not None:
        footprint = section_footprint(mapped_mesh, zmin, zmax)
        if footprint is not None and len(footprint) >= 3:
            return footprint, footprint.mean(axis=0), zmin
    low = mapped_vertices[:, 2] <= zmin + max(0.25, 0.05 * (zmax - zmin))
    points = mapped_vertices[low][:, :2]
    if len(points) < 4:
        points = mapped_vertices[:, :2]
    hull = ConvexHull(points)
    footprint = points[hull.vertices]
    centroid = footprint.mean(axis=0)
    return footprint, centroid, zmin


def load_city_building_footprints(city_dir: Path) -> list[dict[str, Any]]:
    cache_key = str(city_dir.resolve())
    if cache_key in _BUILDING_FOOTPRINT_CACHE:
        return _BUILDING_FOOTPRINT_CACHE[cache_key]
    footprints = []
    for path in sorted((city_dir / "buildings").glob("*.glb")):
        try:
            mapped = mapped_vertices_from_glb(path)
            footprint, _centroid, min_z = building_footprint(mapped)
        except Exception:
            continue
        footprints.append({
            "path": str(path.resolve()),
            "name": path.stem,
            "footprint": footprint,
            "min_z": float(min_z),
            "max_z": float(mapped[:, 2].max()),
        })
    _BUILDING_FOOTPRINT_CACHE[cache_key] = footprints
    return footprints


def load_mapped_terrain_mesh(city_dir: Path) -> trimesh.Trimesh | None:
    path = city_dir / "terrain.glb"
    cache_key = str(path.resolve())
    if cache_key in _TERRAIN_MESH_CACHE:
        return _TERRAIN_MESH_CACHE[cache_key]
    if not path.exists():
        _TERRAIN_MESH_CACHE[cache_key] = None
        return None
    try:
        mesh = mapped_trimesh_from_glb(path)
    except Exception:
        mesh = None
    _TERRAIN_MESH_CACHE[cache_key] = mesh
    return mesh


def load_public_points(work_dir: Path) -> np.ndarray:
    weighted_points = []
    for filename, weight in (("sidewalk_points.json", 1.0), ("lane_points.json", 0.35)):
        path = work_dir / filename
        if not path.exists():
            continue
        data = read_json(path)
        for point in data.get("points", []):
            weighted_points.append(
                (
                    float(point["utm_x"]),
                    float(point["utm_y"]),
                    float(point.get("height", 0.0)),
                    weight,
                )
            )
    if not weighted_points:
        raise RuntimeError(f"no sidewalk/lane points found in {work_dir}")
    return np.asarray(weighted_points, dtype=float)


def best_wall_for_footprint(
    footprint: np.ndarray,
    centroid: np.ndarray,
    public_points: np.ndarray,
    min_wall_length: float,
    door_width: float,
) -> dict[str, Any]:
    best = None
    xy = public_points[:, :2]
    weights = public_points[:, 3]
    edge_margin = 1.5

    for index in range(len(footprint)):
        a = footprint[index]
        b = footprint[(index + 1) % len(footprint)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < max(min_wall_length, door_width + 1.0):
            continue

        tangent = edge / length
        normals = [np.array([-tangent[1], tangent[0]]), np.array([tangent[1], -tangent[0]])]
        for normal in normals:
            if np.sign(np.dot((a + b) * 0.5 + normal - centroid, normal)) < 0:
                normal = -normal

            rel = xy - a
            proj = rel @ tangent
            lateral = rel @ normal
            mask = (proj >= -edge_margin) & (proj <= length + edge_margin) & (lateral >= 0.8) & (lateral <= 14.0)
            if not np.any(mask):
                continue

            masked_weights = weights[mask]
            weighted_count = float(masked_weights.sum())
            mean_dist = float(np.average(lateral[mask], weights=masked_weights))
            raw_center = float(np.average(proj[mask], weights=masked_weights))
            center_s = float(np.clip(raw_center, door_width * 0.5 + 0.8, length - door_width * 0.5 - 0.8))
            center = a + tangent * center_s
            score = weighted_count / (mean_dist + 0.5) + 0.02 * length
            candidate = {
                "score": score,
                "edge_index": index,
                "length": length,
                "weighted_count": weighted_count,
                "mean_public_distance": mean_dist,
                "center": center,
                "tangent": tangent,
                "outward_normal": normalize(normal),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        raise RuntimeError("no suitable road-facing wall found")
    return best


def candidate_stage_paths(stage: Usd.Stage) -> list[tuple[str, np.ndarray, np.ndarray]]:
    candidates = []
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in stage.TraverseAll():
        path = prim.GetPath().pathString
        if not path.startswith("/World/city"):
            continue
        if prim.GetTypeName() not in {"Mesh", "Xform"}:
            continue
        name = prim.GetName().lower()
        if not any(token in name for token in ("building", "element")):
            continue
        try:
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        except Exception:
            continue
        mn = np.asarray(box.GetMin(), dtype=float)
        mx = np.asarray(box.GetMax(), dtype=float)
        if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
            continue
        if float(np.linalg.norm(mx - mn)) <= 1e-6:
            continue
        candidates.append((path, mn, mx))
    return candidates


def name_fallback_path(candidates: list[tuple[str, np.ndarray, np.ndarray]], building_glb: Path) -> str | None:
    stem = building_glb.stem
    tokens = {stem.lower(), stem.replace(".", "_").lower()}
    if stem.startswith("buildings_element."):
        suffix = stem.split(".", 1)[1]
        tokens.add(f"element_{suffix}".lower())
        tokens.add(f"buildings_element_{suffix}".lower())
    for path, _mn, _mx in candidates:
        lowered = path.lower()
        if any(token in lowered for token in tokens):
            return path
    return None


def normalized_prim_name(value: str) -> str:
    return "_".join(part for part in re.split(r"[^a-z0-9]+", value.lower()) if part)


def match_stage_path_by_name(stage: Usd.Stage, building_glb: Path) -> str | None:
    target = normalized_prim_name(building_glb.stem)
    matches = []
    for prim in stage.TraverseAll():
        path = prim.GetPath().pathString
        in_city = path.startswith("/World/city") or (path.count("/") == 1 and path.startswith("/buildings_"))
        if not in_city or prim.GetTypeName() not in {"Mesh", "Xform"}:
            continue
        name = normalized_prim_name(prim.GetName())
        if name == target or name.startswith(target + "_"):
            matches.append((not prim.IsActive(), name != target, path.count("/"), len(name), path))
    if not matches:
        return None
    matches.sort()
    best = matches[0]
    if best[1] and len(matches) > 1 and matches[1][:4] == best[:4]:
        return None
    return best[4]


def match_stage_path(base_usd: Path, building_glb: Path, max_dist: float) -> tuple[str, float | None]:
    stage = Usd.Stage.Open(str(base_usd))
    if stage is None:
        raise RuntimeError(f"could not open base USD: {base_usd}")
    name_match = match_stage_path_by_name(stage, building_glb)
    if name_match:
        return name_match, None
    mapped = mapped_vertices_from_glb(building_glb)
    bbox_min = mapped.min(axis=0)
    bbox_max = mapped.max(axis=0)
    candidates = candidate_stage_paths(stage)
    best_path = None
    best_dist = float("inf")
    for path, mn, mx in candidates:
        dist = float(np.linalg.norm(mn - bbox_min) + np.linalg.norm(mx - bbox_max))
        if dist < best_dist:
            best_path = path
            best_dist = dist
    if best_path is not None and best_dist <= max_dist:
        return best_path, best_dist
    fallback = name_fallback_path(candidates, building_glb)
    if fallback:
        return fallback, best_dist if np.isfinite(best_dist) else None
    raise RuntimeError(f"could not match {building_glb.name} to a prim in {base_usd}; best_dist={best_dist:.3f}")


def first_root_prim(layer_path: Path) -> Sdf.Path:
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    roots = list(layer.rootPrims)
    if not roots:
        raise RuntimeError(f"no root prims in {layer_path}")
    if layer.defaultPrim:
        return Sdf.Path(f"/{layer.defaultPrim}")
    return Sdf.Path(f"/{roots[0].name}")


def choose_storefront_usd(selected: dict[str, Any], store_root: Path) -> Path:
    nearest = selected.get("nearest_storefront") or {}
    asset = nearest.get("asset_path")
    if asset and Path(asset).exists():
        return Path(asset)
    fallback = store_root / DEFAULT_STORE_ASSET / f"{DEFAULT_STORE_ASSET}.usd"
    if fallback.exists():
        return fallback
    matches = sorted(store_root.glob("*/*.usd"))
    if matches:
        return matches[0]
    raise RuntimeError(f"no storefront USD found under {store_root}")


def storefront_bbox(path: Path, target_height: float | None = None) -> tuple[list[float], list[float], list[float], list[float], float]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open storefront USD: {path}")
    root = stage.GetDefaultPrim()
    if not root:
        roots = list(stage.GetPseudoRoot().GetChildren())
        if not roots:
            raise RuntimeError(f"no root prims in storefront USD: {path}")
        root = roots[0]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = bbox_cache.ComputeWorldBound(root).ComputeAlignedBox()
    mn = np.asarray(box.GetMin(), dtype=float)
    mx = np.asarray(box.GetMax(), dtype=float)
    if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
        raise RuntimeError(f"invalid storefront bbox: {path}")
    dims = mx - mn
    if dims[0] <= 0 or dims[2] <= 0:
        raise RuntimeError(f"degenerate storefront bbox: {path}")
    raw_dims = dims.copy()
    scale = 1.0
    if target_height is not None and target_height > 0.0:
        scale = float(target_height) / float(raw_dims[2])
        dims = raw_dims * scale
        mn = mn * scale
        mx = mx * scale
    return [float(v) for v in mn], [float(v) for v in mx], [float(v) for v in dims], [float(v) for v in raw_dims], float(scale)


def stage_meters_per_unit(path: Path) -> float:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {path}")
    return float(UsdGeom.GetStageMetersPerUnit(stage))


def load_grscene_door_info(interior_usd: Path) -> dict[str, Any]:
    door_info_path = interior_usd.parent / "door_info.json"
    if not door_info_path.exists():
        raise RuntimeError(f"missing GR scene door_info.json: {door_info_path}; run detect_grscene_doors.py first")
    info = read_json(door_info_path)
    doors = [door for door in info.get("doors", []) if door.get("likely_entry_door", True)]
    scene_bbox = info.get("scene_bbox_xyz")
    if not doors:
        raise RuntimeError(f"no likely entry doors in {door_info_path}")
    if not scene_bbox:
        raise RuntimeError(f"missing scene_bbox_xyz in {door_info_path}; rerun detect_grscene_doors.py")
    info["doors"] = doors
    return info


def choose_grscene_door(doors: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        doors,
        key=lambda door: (
            float(door.get("width_m", 0.0)),
            float(door.get("height_m", 0.0)),
            int(door.get("point_count", 0)),
        ),
    )


def door_size_from_grscene(
    args: argparse.Namespace,
    door: dict[str, Any],
    logical_scale: float,
) -> tuple[float, float, str]:
    width = args.door_width
    height = args.door_height
    source = "manual"
    if width is None:
        raw_width = float(door["width_m"]) * logical_scale
        width = clamp(raw_width, args.min_door_width, args.max_door_width)
        source = "grscene_door_info"
        if width != raw_width:
            source += "_clamped"
    if height is None:
        raw_height = float(door["height_m"]) * logical_scale
        height = clamp(raw_height, args.min_door_height, args.max_door_height)
        if source == "manual":
            source = "grscene_door_info_height"
        if height != raw_height and not source.endswith("_clamped"):
            source += "_clamped"
    return float(width), float(height), source


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def door_size_from_storefront(args: argparse.Namespace, dims: list[float]) -> tuple[float, float, str]:
    width = args.door_width
    height = args.door_height
    source = "manual"
    if width is None:
        width = clamp(float(dims[0]) + args.door_width_margin, args.min_door_width, args.max_door_width)
        source = "storefront_bbox"
    if height is None:
        height = clamp(float(dims[2]) + args.door_height_margin, args.min_door_height, args.max_door_height)
        source = "storefront_bbox" if source != "manual" else "storefront_bbox_height"
    return float(width), float(height), source


def door_size_from_args(args: argparse.Namespace, dims: list[float] | None) -> tuple[float, float, str]:
    if args.fit_door_to_storefront:
        if not dims:
            raise RuntimeError("--fit-door-to-storefront requires a storefront asset")
        return door_size_from_storefront(args, dims)
    width = args.door_width if args.door_width is not None else 4.5
    height = args.door_height if args.door_height is not None else 3.2
    source = "manual" if args.door_width is not None or args.door_height is not None else "fixed_gr_scene_door"
    return float(width), float(height), source


def interior_frontage_width(
    scene_bbox: dict[str, Any],
    door: dict[str, Any],
    logical_scale: float,
) -> float:
    dims = scene_bbox["dims_xyz"]
    tangent_axis = 1 if door.get("normal_axis_guess") == "x" else 0
    return float(dims[tangent_axis]) * float(logical_scale)


def building_opening_width_from_room(
    args: argparse.Namespace,
    door_width: float,
    room_frontage_width: float,
) -> float:
    return clamp(
        max(
            float(room_frontage_width) + args.building_opening_width_margin,
            float(door_width) + args.building_opening_width_margin,
            args.min_building_opening_width,
        ),
        float(door_width),
        args.max_building_opening_width,
    )


def validate_door_vertical_alignment(args: argparse.Namespace, bottom_z: float, building_min_z: float, building_max_z: float) -> None:
    if bottom_z < building_min_z - args.max_door_below_building_bottom_m:
        raise RuntimeError(
            "door bottom is below the selected building base; "
            f"door_bottom_z={bottom_z:.3f} building_min_z={building_min_z:.3f}"
        )
    if bottom_z > building_min_z + args.max_door_above_building_bottom_m:
        raise RuntimeError(
            "door bottom is too high above the selected building base; "
            f"door_bottom_z={bottom_z:.3f} building_min_z={building_min_z:.3f} "
            f"delta={bottom_z - building_min_z:.3f}"
        )
    if bottom_z > building_max_z + args.max_door_above_building_top_m:
        raise RuntimeError(
            "door bottom is above the selected building top; "
            f"door_bottom_z={bottom_z:.3f} building_max_z={building_max_z:.3f}"
        )


def validate_exterior_building_clearance(
    city_dir: Path,
    selected_building: Path,
    door_center_xy: np.ndarray,
    tangent: np.ndarray,
    outward: np.ndarray,
    door_width: float,
    bottom_z: float,
    height: float,
) -> None:
    selected_path = str(selected_building.resolve())
    for record in load_city_building_footprints(city_dir):
        if record["path"] == selected_path:
            continue
        if bottom_z + min(height, 2.0) < record["min_z"] or bottom_z > record["max_z"] + 0.5:
            continue
        for distance in (0.8, 1.5, 3.0, 5.0):
            for lateral in (-0.45 * door_width, 0.0, 0.45 * door_width):
                point = door_center_xy + outward * distance + tangent * lateral
                if point_in_polygon(point, record["footprint"], tolerance=0.15):
                    raise RuntimeError(
                        "door exterior corridor intersects another building; "
                        f"building={record['name']} distance={distance:.2f} lateral={lateral:.2f}"
                    )


def exterior_terrain_clearance_rays(
    door_center_xy: np.ndarray,
    tangent: np.ndarray,
    outward: np.ndarray,
    door_width: float,
    bottom_z: float,
    door_height: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    origins = []
    directions = []
    max_lengths = []
    heights = [min(door_height - 0.35, value) for value in (1.2, 2.0) if door_height > value + 0.15]
    for height in heights:
        for lateral in (-0.45 * door_width, 0.0, 0.45 * door_width):
            start_xy = door_center_xy - outward * 0.8 + tangent * lateral
            end_xy = door_center_xy + outward * 7.0 + tangent * lateral
            start = np.asarray([start_xy[0], start_xy[1], bottom_z + height], dtype=float)
            end = np.asarray([end_xy[0], end_xy[1], bottom_z + height], dtype=float)
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length <= 1e-9:
                continue
            origins.append(start)
            directions.append(direction / length)
            max_lengths.append(length)
    return np.asarray(origins), np.asarray(directions), max_lengths


def validate_exterior_terrain_clearance(
    city_dir: Path,
    door_center_xy: np.ndarray,
    tangent: np.ndarray,
    outward: np.ndarray,
    door_width: float,
    bottom_z: float,
    door_height: float,
) -> None:
    mesh = load_mapped_terrain_mesh(city_dir)
    if mesh is None:
        return
    origins, directions, max_lengths = exterior_terrain_clearance_rays(
        door_center_xy,
        tangent,
        outward,
        door_width,
        bottom_z,
        door_height,
    )
    if not len(origins):
        return
    try:
        locations, ray_indices, _tri_indices = mesh.ray.intersects_location(
            origins,
            directions,
            multiple_hits=False,
        )
    except Exception:
        return
    segment_hits = []
    for location, ray_index in zip(locations, ray_indices, strict=True):
        distance = float(np.dot(location - origins[ray_index], directions[ray_index]))
        if -1e-4 <= distance <= max_lengths[ray_index] + 1e-4:
            segment_hits.append(location)
    if segment_hits:
        first = segment_hits[0]
        raise RuntimeError(
            "door exterior corridor intersects terrain above walking clearance; "
            f"hit=({first[0]:.3f},{first[1]:.3f},{first[2]:.3f})"
        )


def validate_exterior_public_surface(
    args: argparse.Namespace,
    public_points: np.ndarray,
    door_center_xy: np.ndarray,
    tangent: np.ndarray,
    outward: np.ndarray,
    door_width: float,
    bottom_z: float,
) -> list[dict[str, float | bool]]:
    samples: list[dict[str, float | bool]] = []
    if len(public_points) == 0:
        return samples
    xy = public_points[:, :2]
    heights = public_points[:, 2]
    valid_distances: set[float] = set()
    for distance in (1.5, 3.0, 5.0, 7.0):
        distance_has_match = False
        for lateral in (-0.35 * door_width, 0.0, 0.35 * door_width):
            point = door_center_xy + outward * distance + tangent * lateral
            deltas = xy - point
            nearest_index = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
            nearest_xy = xy[nearest_index]
            nearest_distance = float(np.linalg.norm(nearest_xy - point))
            height_delta = float(bottom_z - heights[nearest_index])
            valid = (
                nearest_distance <= args.max_exterior_public_distance_m
                and abs(height_delta) <= args.max_exterior_public_height_delta_m
            )
            distance_has_match = distance_has_match or valid
            samples.append({
                "distance_m": float(distance),
                "lateral_m": float(lateral),
                "sample_x": float(point[0]),
                "sample_y": float(point[1]),
                "nearest_public_x": float(nearest_xy[0]),
                "nearest_public_y": float(nearest_xy[1]),
                "nearest_public_distance_m": nearest_distance,
                "height_delta_m": height_delta,
                "valid": bool(valid),
            })
        if distance_has_match:
            valid_distances.add(float(distance))
    if len(valid_distances) < args.min_exterior_public_distance_hits:
        best = min(samples, key=lambda item: (item["nearest_public_distance_m"], abs(item["height_delta_m"])))
        raise RuntimeError(
            "door exterior direction is not sufficiently supported by sidewalk/lane points; "
            f"valid_distances={len(valid_distances)} "
            f"best_distance={best['nearest_public_distance_m']:.3f} "
            f"best_height_delta={best['height_delta_m']:.3f}"
        )
    return samples


def normalized_building_keys(value: Any) -> set[str]:
    if not value:
        return set()
    text = str(value)
    keys = {text.lower()}
    path = Path(text)
    if path.name:
        keys.add(path.name.lower())
    if path.suffix.lower() in {".glb", ".usd", ".usda", ".usdc"} and path.stem:
        keys.add(path.stem.lower())
    return {key for key in keys if key}


def storefront_overlaps_doorway(
    item: dict[str, Any],
    door_center_xy: np.ndarray,
    tangent_xy: np.ndarray,
    outward_xy: np.ndarray,
    door_width: float,
    normal_clearance: float,
) -> bool:
    location = item.get("opening_center") or item.get("final_location")
    if not location:
        return True
    delta = np.asarray(location[:2], dtype=float) - door_center_xy
    lateral = abs(float(delta @ tangent_xy))
    normal = abs(float(delta @ outward_xy))
    storefront_width = float(item.get("opening_width") or item.get("storefront_facade_width") or 4.0)
    return lateral <= 0.5 * (door_width + storefront_width) + 1.0 and normal <= normal_clearance


def excluded_storefronts_for_selected(
    plan_path: Path,
    selected: dict[str, Any],
    door_center_xy: np.ndarray,
    tangent_xy: np.ndarray,
    outward_xy: np.ndarray,
    door_width: float,
    door_depth: float,
) -> tuple[list[int], list[str]]:
    if not plan_path.exists():
        return [], []

    selected_keys = set()
    selected_keys.update(normalized_building_keys(selected.get("building_name")))
    selected_keys.update(normalized_building_keys(selected.get("building_path")))
    if not selected_keys:
        return [], []

    indices = []
    buildings = []
    plan = read_json(plan_path)
    for array_index, item in enumerate(plan.get("placements", [])):
        item_keys = set()
        item_keys.update(normalized_building_keys(item.get("building_name")))
        item_keys.update(normalized_building_keys(item.get("building_path")))
        if not selected_keys.intersection(item_keys):
            continue
        if item.get("status") != "planned":
            continue
        if item.get("facade_status") and item.get("facade_status") != "facade_aligned":
            continue
        if not storefront_overlaps_doorway(
            item,
            door_center_xy,
            tangent_xy,
            outward_xy,
            door_width,
            max(2.0, float(door_depth)),
        ):
            continue
        try:
            indices.append(int(item.get("index", array_index)))
        except (TypeError, ValueError):
            indices.append(array_index)
        building = item.get("building_name") or item.get("building_path")
        if building:
            buildings.append(str(building))
    return sorted(set(indices)), sorted(set(buildings))


def storefront_cut_source_glb(original: Path, manifest_path: Path | None) -> Path:
    if not manifest_path or not manifest_path.exists():
        return original
    manifest = read_json(manifest_path)
    original_resolved = original.resolve()
    for record in manifest.get("records", []):
        building = Path(str(record.get("building") or ""))
        cut_glb = Path(str(record.get("cut_glb") or ""))
        if building.exists() and building.resolve() == original_resolved and cut_glb.exists():
            return cut_glb.resolve()
    return original


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(((point - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + ab * t)))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray, tolerance: float = 0.05) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if point_segment_distance(point, a, b) <= tolerance:
            return True
        xi, yi = float(a[0]), float(a[1])
        xj, yj = float(b[0]), float(b[1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y:
                inside = not inside
    return inside


def local_normal_options(door: dict[str, Any]) -> list[np.ndarray]:
    axis = door.get("normal_axis_guess")
    if axis == "x":
        return [np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0])]
    return [np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0])]


def door_outer_face_anchor_xy(door: dict[str, Any], local_normal: np.ndarray) -> np.ndarray:
    mn = door["bbox_min_xyz"]
    mx = door["bbox_max_xyz"]
    center = door["center_xyz"]
    if abs(float(local_normal[0])) >= abs(float(local_normal[1])):
        x = mx[0] if local_normal[0] > 0 else mn[0]
        return np.asarray([x, center[1]], dtype=float)
    y = mx[1] if local_normal[1] > 0 else mn[1]
    return np.asarray([center[0], y], dtype=float)


def bbox_xy_corners(bbox: dict[str, Any]) -> list[np.ndarray]:
    mn = bbox["bbox_min_xyz"]
    mx = bbox["bbox_max_xyz"]
    return [
        np.asarray([mn[0], mn[1]], dtype=float),
        np.asarray([mn[0], mx[1]], dtype=float),
        np.asarray([mx[0], mn[1]], dtype=float),
        np.asarray([mx[0], mx[1]], dtype=float),
    ]


def fit_grscene_to_building(
    doors: list[dict[str, Any]],
    scene_bbox: dict[str, Any],
    footprint: np.ndarray,
    door_center_xy: np.ndarray,
    door_bottom_z: float,
    outward: np.ndarray,
    logical_scale: float,
    min_doorway_inset_m: float,
) -> dict[str, Any]:
    candidates = []
    scene_corners = bbox_xy_corners(scene_bbox)
    inward_nudges = [0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    for door in doors:
        door_bottom = float(door["bbox_min_xyz"][2])
        for local_normal in local_normal_options(door):
            yaw = yaw_to_map_xy(local_normal, outward)
            door_anchor = door_outer_face_anchor_xy(door, local_normal)
            rotated_anchor = rotate_xy(door_anchor * logical_scale, yaw)
            base_translate_xy = door_center_xy - rotated_anchor
            for inward_nudge in inward_nudges:
                translate_xy = base_translate_xy - outward * inward_nudge
                transformed_door_anchor = door_center_xy - outward * inward_nudge
                transformed_corners = [
                    translate_xy + rotate_xy(corner * logical_scale, yaw)
                    for corner in scene_corners
                ]
                outside = [
                    corner for corner in transformed_corners
                    if not point_in_polygon(corner, footprint)
                ]
                outside_distance = 0.0
                for corner in outside:
                    distances = [
                        point_segment_distance(corner, footprint[i], footprint[(i + 1) % len(footprint)])
                        for i in range(len(footprint))
                    ]
                    outside_distance += min(distances) if distances else 0.0
                candidates.append(
                    {
                        "door": door,
                        "room_translate": [
                            float(translate_xy[0]),
                            float(translate_xy[1]),
                            float(door_bottom_z - door_bottom * logical_scale),
                        ],
                        "room_yaw_deg": float(yaw),
                        "outside_corner_count": len(outside),
                        "outside_distance_m": float(outside_distance),
                        "inward_nudge_m": float(inward_nudge),
                        "door_anchor_inside": bool(point_in_polygon(transformed_door_anchor, footprint, tolerance=0.02)),
                        "doorway_inset_deficit_m": float(max(0.0, min_doorway_inset_m - inward_nudge)),
                        "transformed_scene_corners_xy": [[float(c[0]), float(c[1])] for c in transformed_corners],
                    }
                )

    candidates.sort(
        key=lambda item: (
            item["outside_corner_count"],
            item["doorway_inset_deficit_m"],
            0 if item["door_anchor_inside"] else 1,
            item["outside_distance_m"],
            abs(item["inward_nudge_m"] - min_doorway_inset_m),
            -float(item["door"].get("width_m", 0.0)),
        )
    )
    best = candidates[0]
    best["fit_status"] = "inside_building" if best["outside_corner_count"] == 0 else "outside_building"
    return best


def composed_interior_path(room_path: str, source_prim_path: str) -> str:
    if source_prim_path.startswith("/Root/"):
        return room_path + source_prim_path.removeprefix("/Root")
    if source_prim_path == "/Root":
        return room_path
    return room_path + source_prim_path


def unique_building_candidates(score: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = []
    if score.get("selected"):
        raw_candidates.append(score["selected"])
    raw_candidates.extend(score.get("top_candidates", []))
    candidates = []
    seen = set()
    for candidate in raw_candidates:
        key = candidate.get("building_path")
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def build_plan(args: argparse.Namespace) -> DoorPlan:
    if args.anchor_plan:
        return build_plan_from_anchor(args)

    score = read_json(args.score)
    candidates = unique_building_candidates(score)
    if not candidates:
        raise RuntimeError(f"no selected building candidates in {args.score}")
    errors = []
    best_plan = None
    best_capacity = -1
    for idx, selected in enumerate(candidates):
        for selected_variant in selected_grscene_variants(selected, score.get("grscene_catalog", [])):
            try:
                plan = build_plan_for_selected(args, selected_variant)
                capacity = estimate_local_storefront_capacity(plan, args)
                plan.local_storefront_capacity_estimate = capacity
                object_capacity = estimate_local_object_target_capacity(plan, args)
                plan.local_object_target_capacity_estimate = object_capacity
                if object_capacity < args.min_local_object_targets:
                    errors.append({
                        "rank": idx,
                        "building": selected.get("building_name") or selected.get("building_path"),
                        "grscene": (selected_variant.get("selected_grscene") or {}).get("scene"),
                        "error": (
                            f"local public-side object capacity {object_capacity} is below "
                            f"the required {args.min_local_object_targets}"
                        ),
                    })
                    break
                if capacity > best_capacity:
                    best_plan = plan
                    best_capacity = capacity
                if capacity >= args.min_local_storefront_capacity:
                    return plan
                break
            except RuntimeError as exc:
                grscene = selected_variant.get("selected_grscene") or {}
                errors.append({
                    "rank": idx,
                    "building": selected.get("building_name") or selected.get("building_path"),
                    "grscene": grscene.get("scene"),
                    "error": str(exc),
                })
    if best_plan is not None:
        return best_plan
    detail = "; ".join(
        f"#{item['rank']} {item['building']}[{item.get('grscene') or 'no_grscene'}]: {item['error']}"
        for item in errors[:8]
    )
    raise RuntimeError(f"could not build innout plan from {len(candidates)} candidates: {detail}")


def build_plan_from_anchor(args: argparse.Namespace) -> DoorPlan:
    anchor_data = read_json(args.anchor_plan)
    plan_fields = DoorPlan.__dataclass_fields__
    required = {
        name
        for name, definition in plan_fields.items()
        if definition.default is MISSING and definition.default_factory is MISSING
    }
    missing = sorted(required - set(anchor_data))
    if missing:
        raise RuntimeError(f"anchor plan is missing DoorPlan fields: {missing}")
    plan = DoorPlan(**{name: anchor_data[name] for name in plan_fields if name in anchor_data})
    if plan.city != args.city:
        raise RuntimeError(f"anchor plan city {plan.city!r} does not match {args.city!r}")

    if args.interior_scene or args.interior_usd:
        score = read_json(args.score)
        building_glb = Path(plan.building_glb).resolve()
        candidates = [score.get("selected"), *(score.get("top_candidates") or [])]
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate
                and Path(candidate.get("building_path", "")).resolve() == building_glb
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(
                f"anchor building is absent from geometry score candidates: {building_glb}"
            )
        rebuilt_plan = build_plan_for_selected(args, selected)
        rebuilt_plan.local_storefront_capacity_estimate = estimate_local_storefront_capacity(
            rebuilt_plan, args
        )
        rebuilt_plan.local_object_target_capacity_estimate = estimate_local_object_target_capacity(
            rebuilt_plan, args
        )
        if rebuilt_plan.local_storefront_capacity_estimate < args.min_local_storefront_capacity:
            raise RuntimeError(
                "local storefront capacity is below the required minimum: "
                f"{rebuilt_plan.local_storefront_capacity_estimate} < "
                f"{args.min_local_storefront_capacity}"
            )
        if rebuilt_plan.local_object_target_capacity_estimate < args.min_local_object_targets:
            raise RuntimeError(
                "local public-side object capacity is below the required minimum: "
                f"{rebuilt_plan.local_object_target_capacity_estimate} < "
                f"{args.min_local_object_targets}"
            )
        return rebuilt_plan

    original_door_size = (float(plan.door_width), float(plan.door_height))
    plan.door_width = min(max(original_door_size[0], args.min_door_width), args.max_door_width)
    plan.door_height = min(max(original_door_size[1], args.min_door_height), args.max_door_height)
    if (plan.door_width, plan.door_height) != original_door_size:
        plan.door_size_source = f"{plan.door_size_source}+anchor_current_limits"
    plan.building_opening_width = max(
        float(plan.building_opening_width),
        float(args.min_building_opening_width),
        float(plan.door_width + args.building_opening_width_margin),
    )
    realign_anchor_plan_to_doorway_terrain(args, plan)
    plan.building_opening_top_z = max(
        float(plan.building_opening_top_z),
        float(plan.door_bottom_z + plan.door_height + DOOR_CONNECTOR_OVERLAP_M),
    )

    building_glb = Path(plan.building_glb).resolve()
    if not building_glb.exists():
        raise RuntimeError(f"anchor building GLB does not exist: {building_glb}")
    target_path, target_dist = match_stage_path(args.base_usd, building_glb, args.max_bbox_match_dist)
    storefront_plan = args.storefront_plan or args.work_dir / "storefront_plan.json"
    selected = {
        "building_path": str(building_glb),
        "building_name": building_glb.stem,
    }
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    plan.exterior_public_surface_samples = validate_exterior_public_surface(
        args,
        load_public_points(args.work_dir.resolve()),
        center,
        tangent,
        outward,
        plan.door_width,
        plan.door_bottom_z,
    )
    excluded_indices, excluded_buildings = excluded_storefronts_for_selected(
        storefront_plan,
        selected,
        center,
        tangent,
        outward,
        plan.building_opening_width,
        plan.door_depth,
    )
    cut_source_glb = storefront_cut_source_glb(building_glb, args.storefront_openings_manifest)
    cut_dir = args.work_dir.resolve() / "generated_innout"

    plan.content_radius_m = float(args.content_radius_m)
    plan.base_usd = str(args.base_usd.resolve())
    plan.output_usd = str(args.output.resolve())
    plan.target_prim_path = target_path
    plan.target_match_distance = target_dist
    plan.door_cut_source_glb = str(cut_source_glb)
    plan.door_cut_source_z_up = cut_source_glb.resolve() != building_glb
    plan.door_cut_mode = "boolean_tunnel"
    plan.excluded_storefront_plan_indices = excluded_indices
    plan.excluded_storefront_buildings = excluded_buildings
    plan.removed_storefront_prim_paths = []
    plan.removed_interior_obstacle_paths = []
    plan.removed_exterior_obstacle_paths = []
    plan.interior_wall_opening_summary = {}
    plan.exterior_wall_opening_summary = {}
    plan.terrain_hole_summary = []
    plan.threshold_summary = {}
    plan.door_connector_summary = {}
    plan.cut_glb = str(cut_dir / f"{args.city}_{building_glb.stem}_cut.glb")
    plan.cut_usd = str(cut_dir / f"{args.city}_{building_glb.stem}_cut.usd")
    plan.qa_dir = str((args.qa_root / args.city).resolve())
    return plan


def estimate_local_storefront_capacity(plan: DoorPlan, args: argparse.Namespace) -> int:
    city_dir = args.data_root.resolve() / args.source_dataset / args.city
    osm_path = args.work_dir.resolve() / "road_data_with_height.osm"
    if not city_dir.exists() or not osm_path.exists():
        return 0
    edges, _skipped = load_building_edges(city_dir, 2.5, 1.25)
    roads = load_road_segments(osm_path)
    if not edges or not roads:
        return 0

    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    radius = float(args.content_radius_m)
    facade_width = 7.25
    edge_margin = 0.25
    spacing = facade_width + edge_margin * 2.0
    capacity = 0
    doorway_normal_clearance = max(2.0, float(plan.door_depth))
    for edge in edges:
        half_width = facade_width * 0.5 + edge_margin
        if edge.length < half_width * 2.0:
            continue
        edge_s = half_width
        while edge_s <= edge.length - half_width + 1e-6:
            point = edge.start + edge.tangent * edge_s
            if float(np.linalg.norm(point - center)) > radius + 1e-6:
                edge_s += spacing
                continue
            doorway_delta = point - center
            if (
                abs(float(doorway_delta @ tangent))
                <= 0.5 * (facade_width + float(plan.building_opening_width)) + 1.0
                and abs(float(doorway_delta @ outward)) <= doorway_normal_clearance
            ):
                edge_s += spacing
                continue
            road = nearest_road(point, roads, min(radius, 50.0))
            road_normal = road.get("normal")
            if road.get("distance") is None or road_normal is None:
                edge_s += spacing
                continue
            if float(edge.outward @ road_normal) < 0.05:
                edge_s += spacing
                continue
            if edge_front_clearance_ok(edge, edge_s, facade_width, edges, edge_margin):
                capacity += 1
            edge_s += spacing
    return capacity


def estimate_local_object_target_capacity(plan: DoorPlan, args: argparse.Namespace) -> int:
    points_path = args.work_dir.resolve() / "sidewalk_points.json"
    if not points_path.exists():
        return 0
    payload = read_json(points_path)
    points = payload.get("points", []) if isinstance(payload, dict) else []
    center = np.asarray(plan.door_center_xy, dtype=float)
    outward = normalize(plan.outward_normal_xy)
    footprint = np.asarray(plan.building_footprint_xy, dtype=float)
    candidates = []
    for item in points:
        if item.get("explicit_sidewalk") is not True:
            continue
        point = np.asarray([item.get("utm_x"), item.get("utm_y")], dtype=float)
        if not np.isfinite(point).all():
            continue
        delta = point - center
        if float(np.linalg.norm(delta)) > float(args.content_radius_m) + 1e-6:
            continue
        if float(delta @ outward) < 0.0:
            continue
        if len(footprint) >= 3 and point_in_polygon(point, footprint, tolerance=0.5):
            continue
        candidates.append(point)

    candidates.sort(key=lambda point: float(np.linalg.norm(point - center)))
    selected: list[np.ndarray] = []
    required = max(1, int(args.min_local_object_targets))
    for point in candidates:
        if all(float(np.linalg.norm(point - other)) >= 1.5 for other in selected):
            selected.append(point)
            if len(selected) >= required:
                break
    return len(selected)


def grscene_with_scale_for_bbox(scene: dict[str, Any], bbox: dict[str, Any]) -> dict[str, Any]:
    bx, by, _bz = bbox["dims"]
    margin = 1.5
    width = float(scene.get("width", 0.0))
    depth = float(scene.get("depth", 0.0))
    fits_xy = width + margin <= bx and depth + margin <= by
    scale = min((bx - margin) / width, (by - margin) / depth) if width > 0 and depth > 0 else 1.0
    chosen = dict(scene)
    chosen["scale"] = max(0.5, min(1.15, scale))
    chosen["fits_xy"] = fits_xy
    return chosen


def selected_grscene_variants(selected: dict[str, Any], catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen: set[tuple[str | None, float | None]] = set()

    def add(scene: dict[str, Any] | None) -> None:
        if not scene:
            return
        key = (scene.get("scene"), round(float(scene.get("scale", 1.0)), 6))
        if key in seen:
            return
        seen.add(key)
        variant = dict(selected)
        variant["selected_grscene"] = scene
        variants.append(variant)

    add(selected.get("selected_grscene"))
    bbox = selected.get("bbox") or {}
    if catalog and bbox:
        scored = [grscene_with_scale_for_bbox(scene, bbox) for scene in catalog]
        scored.sort(key=lambda item: (bool(item.get("fits_xy")), float(item.get("scale", 1.0))), reverse=True)
        for scene in scored:
            add(scene)
    return variants or [selected]


def build_plan_for_selected(args: argparse.Namespace, selected: dict[str, Any]) -> DoorPlan:
    data_root = args.data_root.resolve()
    work_dir = args.work_dir.resolve()
    if not selected:
        raise RuntimeError("empty selected building candidate")

    grscene = selected.get("selected_grscene") or {}
    interior_scene = args.interior_scene or grscene.get("scene")
    if args.interior_usd:
        interior_usd = Path(args.interior_usd)
    elif args.interior_scene:
        interior_usd = (
            args.data_root.resolve()
            / "grscenes_commercial"
            / "scenes"
            / args.interior_scene
            / "start_result_navigation.usd"
        )
    else:
        interior_usd = Path(grscene.get("path", ""))
    if not interior_scene or not interior_usd.exists():
        raise RuntimeError(f"missing GR commercial scene for {args.city}")
    resolved_scene = interior_usd.parent.name
    if resolved_scene != interior_scene:
        raise RuntimeError(
            "GR scene label and USD path disagree: "
            f"scene={interior_scene!r} path_scene={resolved_scene!r}"
        )
    door_info = load_grscene_door_info(interior_usd)
    gr_doors = door_info["doors"]
    primary_gr_door = choose_grscene_door(gr_doors)
    logical_gr_scale = float(grscene.get("scale", 1.0))
    if grscene.get("scene") != interior_scene and selected.get("bbox"):
        scene_dims = door_info["scene_bbox_xyz"]["dims_xyz"]
        override_scene = {
            "width": float(scene_dims[0]),
            "depth": float(scene_dims[1]),
        }
        logical_gr_scale = float(
            grscene_with_scale_for_bbox(override_scene, selected["bbox"])["scale"]
        )
    source_meters_per_unit = stage_meters_per_unit(interior_usd)

    storefront_usd = None
    storefront_min: list[float] = []
    storefront_max: list[float] = []
    storefront_dims: list[float] = []
    storefront_raw_dims: list[float] = []
    storefront_scale = 1.0
    if args.fit_door_to_storefront or args.add_exterior_storefront:
        storefront_usd = choose_storefront_usd(selected, args.store_root.resolve())
        storefront_min, storefront_max, storefront_dims, storefront_raw_dims, storefront_scale = storefront_bbox(
            storefront_usd,
            args.storefront_target_height,
        )
    if args.fit_door_to_storefront:
        door_width, door_height, door_size_source = door_size_from_args(args, storefront_dims)
    else:
        door_width, door_height, door_size_source = door_size_from_grscene(args, primary_gr_door, logical_gr_scale)
    room_frontage_width = interior_frontage_width(
        door_info["scene_bbox_xyz"], primary_gr_door, logical_gr_scale
    )
    building_opening_width = building_opening_width_from_room(
        args, door_width, room_frontage_width
    )

    building_glb = Path(selected["building_path"]).resolve()
    if not building_glb.exists():
        raise RuntimeError(f"selected building GLB does not exist: {building_glb}")
    public_points = load_public_points(work_dir)
    mapped_mesh = mapped_trimesh_from_glb(building_glb)
    mapped = np.asarray(mapped_mesh.vertices, dtype=float)
    footprint, centroid, building_min_z = building_footprint(mapped, mapped_mesh)
    building_max_z = float(mapped[:, 2].max())
    wall = best_wall_for_footprint(
        footprint,
        centroid,
        public_points,
        args.min_wall_length,
        building_opening_width,
    )
    center = np.asarray(wall["center"], dtype=float)
    tangent = normalize(wall["tangent"])
    outward = normalize(wall["outward_normal"])
    terrain_query = TerrainHeightQuery.from_roots(data_root, args.city, preferred_source=args.source_dataset)
    doorway_sample_xy, doorway_sample_z, terrain_z = doorway_terrain_alignment(
        terrain_query,
        center,
        tangent,
        outward,
        door_width,
        building_min_z,
    )
    exterior_surface_z = float(terrain_z)
    bottom_z = float(exterior_surface_z)
    validate_door_vertical_alignment(args, bottom_z, building_min_z, building_max_z)
    room_fit = fit_grscene_to_building(
        gr_doors,
        door_info["scene_bbox_xyz"],
        footprint,
        center,
        bottom_z,
        outward,
        logical_gr_scale,
        args.min_indoor_doorway_inset_m,
    )
    if room_fit["fit_status"] != "inside_building":
        raise RuntimeError(
            "could not place GR scene fully inside building footprint; "
            f"outside_corners={room_fit['outside_corner_count']} "
            f"outside_distance_m={room_fit['outside_distance_m']:.3f}"
        )
    aligned_door = room_fit["door"]
    yaw = float(room_fit["room_yaw_deg"])
    if not args.fit_door_to_storefront:
        door_width, door_height, door_size_source = door_size_from_grscene(args, aligned_door, logical_gr_scale)
    room_frontage_width = interior_frontage_width(
        door_info["scene_bbox_xyz"], aligned_door, logical_gr_scale
    )
    building_opening_width = building_opening_width_from_room(
        args, door_width, room_frontage_width
    )
    final_doorway_sample_xy, final_doorway_sample_z, final_terrain_z = doorway_terrain_alignment(
        terrain_query,
        center,
        tangent,
        outward,
        door_width,
        building_min_z,
    )
    terrain_delta = float(final_terrain_z - terrain_z)
    if abs(terrain_delta) > 1e-9:
        room_fit["room_translate"][2] = float(room_fit["room_translate"][2] + terrain_delta)
        terrain_z = float(final_terrain_z)
        exterior_surface_z = float(final_terrain_z)
        bottom_z = float(final_terrain_z)
        validate_door_vertical_alignment(args, bottom_z, building_min_z, building_max_z)
    doorway_sample_xy = final_doorway_sample_xy
    doorway_sample_z = final_doorway_sample_z
    terrain_sample_xy, terrain_sample_z = frontage_terrain_samples(
        terrain_query,
        center,
        tangent,
        outward,
        building_opening_width,
        building_min_z,
    )
    terrain_alignment_summary = terrain_height_summary(
        doorway_sample_xy,
        doorway_sample_z,
        terrain_sample_z,
        door_width,
        building_opening_width,
        terrain_z,
    )
    interior_world_top_z = float(
        room_fit["room_translate"][2]
        + float(door_info["scene_bbox_xyz"]["bbox_max_xyz"][2]) * logical_gr_scale
    )
    building_opening_height = float(
        max(
            door_height + DOOR_CONNECTOR_OVERLAP_M,
            interior_world_top_z - bottom_z,
        )
        + args.building_opening_height_margin
    )
    building_opening_top_z = float(bottom_z + building_opening_height)
    if building_opening_top_z + 0.15 > building_max_z:
        raise RuntimeError(
            "transformed GR scene and opening do not fit below the selected building roof; "
            f"opening_top={building_opening_top_z:.3f} building_top={building_max_z:.3f}"
        )
    source_dir = data_root / args.source_dataset / args.city
    exterior_public_surface_samples = validate_exterior_public_surface(
        args, public_points, center, tangent, outward, door_width, bottom_z
    )
    validate_exterior_building_clearance(
        source_dir,
        building_glb,
        center,
        tangent,
        outward,
        building_opening_width,
        bottom_z,
        building_opening_height,
    )
    validate_exterior_terrain_clearance(source_dir, center, tangent, outward, door_width, bottom_z, door_height)

    target_path, target_dist = match_stage_path(args.base_usd, building_glb, args.max_bbox_match_dist)
    usd_scale = logical_gr_scale * source_meters_per_unit
    storefront_plan = args.storefront_plan or work_dir / "storefront_plan.json"
    excluded_storefront_indices, excluded_storefront_buildings = excluded_storefronts_for_selected(
        storefront_plan,
        selected,
        center,
        tangent,
        outward,
        building_opening_width,
        float(args.door_depth),
    )
    cut_source_glb = storefront_cut_source_glb(building_glb, args.storefront_openings_manifest)

    store_xy = center + outward * args.storefront_offset_m
    storefront_bottom_offset = float(storefront_min[2]) if storefront_min else 0.0
    cut_dir = work_dir / "generated_innout"
    cut_glb = cut_dir / f"{args.city}_{building_glb.stem}_cut.glb"
    cut_usd = cut_dir / f"{args.city}_{building_glb.stem}_cut.usd"
    output_usd = args.output.resolve()
    qa_dir = (args.qa_root / args.city).resolve()
    removed_door_path = composed_interior_path("/World/start_result_navigation", str(aligned_door.get("prim_path", "")))
    building_opening_bottom_z = min(float(terrain_sample_z[7]), float(terrain_sample_z[8]))
    building_opening_bottom_z += THRESHOLD_SURFACE_OFFSET_M - THRESHOLD_BASE_CLEARANCE_M
    return DoorPlan(
        city=args.city,
        content_radius_m=float(args.content_radius_m),
        local_storefront_capacity_estimate=0,
        base_usd=str(args.base_usd.resolve()),
        output_usd=str(output_usd),
        building_glb=str(building_glb),
        door_cut_source_glb=str(cut_source_glb),
        door_cut_source_z_up=cut_source_glb.resolve() != building_glb.resolve(),
        door_cut_mode="boolean_tunnel",
        target_prim_path=target_path,
        target_match_distance=target_dist,
        interior_scene=interior_scene,
        interior_usd=str(interior_usd.resolve()),
        exterior_storefront_enabled=bool(args.add_exterior_storefront),
        storefront_usd=str(storefront_usd.resolve()) if storefront_usd else None,
        storefront_bbox_min=storefront_min,
        storefront_bbox_max=storefront_max,
        storefront_bbox_dims=storefront_dims,
        storefront_raw_bbox_dims=storefront_raw_dims,
        storefront_scale_multiplier=float(storefront_scale),
        excluded_storefront_plan_indices=excluded_storefront_indices,
        excluded_storefront_buildings=excluded_storefront_buildings,
        removed_storefront_prim_paths=[],
        interior_door_name=str(aligned_door.get("name", "")),
        interior_door_prim_path=str(aligned_door.get("prim_path", "")),
        interior_door_bbox_min=[float(v) for v in aligned_door["bbox_min_xyz"]],
        interior_door_bbox_max=[float(v) for v in aligned_door["bbox_max_xyz"]],
        interior_door_center=[float(v) for v in aligned_door["center_xyz"]],
        interior_scene_bbox_min=[float(v) for v in door_info["scene_bbox_xyz"]["bbox_min_xyz"]],
        interior_scene_bbox_max=[float(v) for v in door_info["scene_bbox_xyz"]["bbox_max_xyz"]],
        interior_scene_bbox_dims=[float(v) for v in door_info["scene_bbox_xyz"]["dims_xyz"]],
        interior_fit_status=str(room_fit["fit_status"]),
        interior_fit_outside_corner_count=int(room_fit["outside_corner_count"]),
        interior_fit_outside_distance_m=float(room_fit["outside_distance_m"]),
        interior_fit_inward_nudge_m=float(room_fit.get("inward_nudge_m", 0.0)),
        interior_door_anchor_inside_building=bool(room_fit.get("door_anchor_inside", False)),
        interior_doorway_inset_deficit_m=float(room_fit.get("doorway_inset_deficit_m", 0.0)),
        removed_interior_door_path=removed_door_path,
        interior_wall_opening_summary={},
        removed_interior_obstacle_paths=[],
        removed_exterior_obstacle_paths=[],
        door_size_source=door_size_source,
        door_center_xy=[float(center[0]), float(center[1])],
        door_bottom_z=float(bottom_z),
        door_width=float(door_width),
        door_height=float(door_height),
        door_depth=float(args.door_depth),
        interior_frontage_width=float(room_frontage_width),
        interior_world_top_z=float(interior_world_top_z),
        building_opening_source="interior_room_frontage",
        building_opening_width=float(building_opening_width),
        building_opening_bottom_z=building_opening_bottom_z,
        building_opening_top_z=building_opening_top_z,
        terrain_z=float(terrain_z),
        exterior_surface_z=float(exterior_surface_z),
        terrain_samples=[[float(pt[0]), float(pt[1])] for pt in terrain_sample_xy],
        terrain_sample_z=[float(value) for value in terrain_sample_z],
        exterior_public_surface_samples=exterior_public_surface_samples,
        building_min_z=float(building_min_z),
        building_footprint_xy=[[float(point[0]), float(point[1])] for point in footprint],
        terrain_hole_margin_m=0.0,
        terrain_hole_summary=[],
        threshold_summary={},
        door_connector_summary={},
        tangent_xy=[float(tangent[0]), float(tangent[1])],
        outward_normal_xy=[float(outward[0]), float(outward[1])],
        doorway_yaw_deg=float(yaw),
        room_translate=[float(v) for v in room_fit["room_translate"]],
        room_yaw_deg=float(yaw),
        room_scale=float(usd_scale),
        room_effective_scale=float(logical_gr_scale),
        storefront_translate=[
            float(100.0 * store_xy[0]),
            float(100.0 * store_xy[1]),
            float(100.0 * (bottom_z - storefront_bottom_offset)),
        ],
        storefront_yaw_deg=float(yaw),
        cut_glb=str(cut_glb),
        cut_usd=str(cut_usd),
        qa_dir=str(qa_dir),
        terrain_height_summary=terrain_alignment_summary,
    )


def run_blender_cut(plan: DoorPlan, args: argparse.Namespace) -> None:
    blender = shutil.which(args.blender) or args.blender
    if plan.door_cut_source_z_up:
        source_vertices, _faces, _uvs, _interior_faces = load_cut_mesh_data(
            Path(plan.door_cut_source_glb)
        )
    else:
        source_vertices = mapped_vertices_from_glb(Path(plan.door_cut_source_glb))
    center = np.asarray(plan.door_center_xy, dtype=float)
    outward = normalize(plan.outward_normal_xy)
    source_outward_max = float(np.max((source_vertices[:, :2] - center) @ outward))
    outside_depth = max(
        THRESHOLD_OUTER_DISTANCE_M + 1.0,
        source_outward_max + 0.5,
    )
    inside_depth = max(plan.door_depth - 0.25, 0.01)
    print(
        "InNOut building cut depth: "
        f"inside={inside_depth:.3f}m outside={outside_depth:.3f}m "
        f"source_outward_max={source_outward_max:.3f}m"
    )
    cmd = [
        blender,
        "--background",
        "--python",
        str(args.cut_helper),
        "--",
        "--input-glb",
        plan.door_cut_source_glb,
        "--output-glb",
        plan.cut_glb,
        "--cut-mode",
        plan.door_cut_mode,
        f"--center={plan.door_center_xy[0]},{plan.door_center_xy[1]}",
        f"--bottom-z={plan.building_opening_bottom_z}",
        f"--tangent={plan.tangent_xy[0]},{plan.tangent_xy[1]}",
        f"--outward-normal={plan.outward_normal_xy[0]},{plan.outward_normal_xy[1]}",
        f"--door-width={plan.building_opening_width}",
        f"--door-height={plan.building_opening_top_z - plan.building_opening_bottom_z}",
        f"--door-depth={plan.door_depth}",
        f"--inside-depth={inside_depth}",
        f"--outside-depth={outside_depth}",
    ]
    if plan.door_cut_source_z_up:
        cmd.append("--input-z-up")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CONDA_PREFIX"):
        env.pop(key, None)
    subprocess.check_call(cmd, env=env)
    cut_glb = Path(plan.cut_glb)
    if not cut_glb.exists():
        raise RuntimeError(f"Blender finished without producing {cut_glb}")
    cut_vertices, cut_faces, _uvs, _interior_faces = load_cut_mesh_data(cut_glb)
    blocking_faces = doorway_blocking_faces(plan, cut_vertices, cut_faces)
    if blocking_faces:
        raise RuntimeError(
            f"cut building still has {len(blocking_faces)} faces crossing the exterior doorway corridor: "
            f"sample={blocking_faces[:8]}"
        )
    write_cut_usd_from_glb(cut_glb, Path(plan.cut_usd), require_interior_faces=True)


def clip_polygon_axis(
    polygon: np.ndarray,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output = []
    previous = polygon[-1]
    previous_value = float(previous[axis] - boundary)
    previous_inside = previous_value >= -1e-9 if keep_greater else previous_value <= 1e-9
    for current in polygon:
        current_value = float(current[axis] - boundary)
        current_inside = current_value >= -1e-9 if keep_greater else current_value <= 1e-9
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-12:
                output.append(previous + (previous_value / denominator) * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return np.asarray(output, dtype=float).reshape(-1, 3)


def doorway_blocking_faces(
    plan: DoorPlan,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> list[int]:
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    relative_xy = vertices[:, :2] - center
    portal_vertices = np.column_stack(
        (
            relative_xy @ tangent,
            relative_xy @ outward,
            vertices[:, 2],
        )
    )
    half_width = max(0.05, 0.5 * float(plan.door_width))
    bounds = (
        (0, -half_width, True),
        (0, half_width, False),
        (1, -0.05, True),
        (1, THRESHOLD_OUTER_DISTANCE_M + 1.0, False),
        (2, float(plan.door_bottom_z) - 0.02, True),
        (2, float(plan.door_bottom_z + plan.door_height) - 0.05, False),
    )
    blocking = []
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64).reshape(-1, 3)):
        polygon = portal_vertices[face]
        for axis, boundary, keep_greater in bounds:
            polygon = clip_polygon_axis(polygon, axis, boundary, keep_greater)
            if len(polygon) < 3:
                break
        if len(polygon) < 3:
            continue
        origin = polygon[0]
        area = sum(
            0.5 * float(np.linalg.norm(np.cross(polygon[index] - origin, polygon[index + 1] - origin)))
            for index in range(1, len(polygon) - 1)
        )
        if area > 1e-6:
            blocking.append(face_index)
    return blocking


def remove_excluded_storefront_prims(stage: Usd.Stage, plan: DoorPlan) -> list[str]:
    excluded = set(plan.excluded_storefront_plan_indices or [])
    if not excluded:
        return []
    parent = stage.GetPrimAtPath("/World/store")
    if not parent:
        return []

    removed = []
    for child in list(parent.GetChildren()):
        custom = child.GetCustomData()
        values = []
        for key in ("navverse_plan_index", "navverse_plan_array_index"):
            try:
                values.append(int(custom.get(key)))
            except (TypeError, ValueError):
                pass
        if excluded.intersection(values):
            removed.append(child.GetPath().pathString)
    for path in removed:
        stage.RemovePrim(path)
    return removed


def bounds_intersect_door_corridor(
    bounds: Gf.Range3d,
    center: np.ndarray,
    tangent: np.ndarray,
    outward: np.ndarray,
    half_width: float,
    outward_min: float,
    outward_max: float,
    bottom: float,
    top: float,
) -> bool:
    if bounds.IsEmpty():
        return False
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    corners = np.asarray(
        [
            [x, y, z]
            for x in (float(minimum[0]), float(maximum[0]))
            for y in (float(minimum[1]), float(maximum[1]))
            for z in (float(minimum[2]), float(maximum[2]))
        ],
        dtype=float,
    )
    local_xy = corners[:, :2] - center
    tangent_range = local_xy @ tangent
    outward_range = local_xy @ outward
    return not (
        tangent_range.max() < -half_width
        or tangent_range.min() > half_width
        or outward_range.max() < outward_min
        or outward_range.min() > outward_max
        or corners[:, 2].max() < bottom
        or corners[:, 2].min() > top
    )


def doorway_bbox_cache() -> UsdGeom.BBoxCache:
    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
    )


def clear_interior_doorway_corridor(stage: Usd.Stage, plan: DoorPlan) -> list[str]:
    room_path = "/World/start_result_navigation"
    room = stage.GetPrimAtPath(room_path)
    if not room:
        return []
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    half_width = max(
        0.55,
        float(plan.door_width) * 0.5 + INNOUT_INTERIOR_CORRIDOR_SIDE_CLEARANCE_M,
    )
    bottom = float(plan.door_bottom_z) - INNOUT_INTERIOR_OPENING_VERTICAL_CLEARANCE_M
    top = float(plan.door_bottom_z) + min(float(plan.door_height) - 0.1, 1.8)
    candidates = []
    for prim in Usd.PrimRange(room):
        path = prim.GetPath().pathString
        if prim.GetName().startswith("Instance") and (
            "/Meshes/Furnitures/" in path
            or "/Meshes/Animation/" in path
            or "/Meshes/BaseAnimation/" in path
        ):
            candidates.append(prim)

    bbox_cache = doorway_bbox_cache()
    removed = []
    for prim in candidates:
        if prim.GetPath().pathString == "/World/Objaverse/NavVerseExitSigns":
            continue
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not bounds_intersect_door_corridor(
            bounds,
            center,
            tangent,
            outward,
            half_width,
            -INNOUT_INTERIOR_CORRIDOR_DEPTH_M,
            0.2,
            bottom,
            top,
        ):
            continue
        removed.append(prim.GetPath().pathString)
    for path in removed:
        stage.OverridePrim(path).SetActive(False)
    room.SetCustomDataByKey("navverse_removed_doorway_obstacles", json.dumps(removed))
    return removed


def face_corner_primvar_values(
    primvar: UsdGeom.Primvar,
    point_count: int,
    counts: list[int],
    indices: list[int],
) -> list[Gf.Vec2f] | None:
    if not primvar or not primvar.HasValue():
        return None
    values = list(primvar.ComputeFlattened() or [])
    interpolation = primvar.GetInterpolation()
    if interpolation == UsdGeom.Tokens.faceVarying and len(values) == len(indices):
        return [Gf.Vec2f(value) for value in values]
    if interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(values) == point_count:
        return [Gf.Vec2f(values[index]) for index in indices]
    if interpolation == UsdGeom.Tokens.uniform and len(values) == len(counts):
        return [Gf.Vec2f(values[face]) for face, count in enumerate(counts) for _ in range(count)]
    if interpolation == UsdGeom.Tokens.constant and len(values) == 1:
        return [Gf.Vec2f(values[0])] * len(indices)
    raise RuntimeError(
        f"unsupported {primvar.GetPrimvarName()} layout on {primvar.GetPrim().GetPath()}: "
        f"interpolation={interpolation!r} values={len(values)}"
    )


def wall_corner_uv_sets(
    prim: Usd.Prim,
    point_count: int,
    counts: list[int],
    indices: list[int],
) -> dict[str, tuple[Any, list[Gf.Vec2f]]]:
    supported_types = {
        Sdf.ValueTypeNames.TexCoord2fArray,
        Sdf.ValueTypeNames.Float2Array,
    }
    result = {}
    for primvar in sorted(
        UsdGeom.PrimvarsAPI(prim).GetPrimvars(),
        key=lambda value: value.GetPrimvarName(),
    ):
        if primvar.GetTypeName() not in supported_types:
            continue
        values = face_corner_primvar_values(primvar, point_count, counts, indices)
        if values is not None:
            result[primvar.GetPrimvarName()] = (primvar.GetTypeName(), values)
    return result


@dataclass(frozen=True)
class WallOpeningClip:
    center: np.ndarray
    tangent: np.ndarray
    outward: np.ndarray
    half_width: float
    bottom: float
    top: float
    outward_min: float
    outward_max: float
    shape: Any


def build_wall_opening_clip(plan: DoorPlan) -> WallOpeningClip:
    half_width = max(
        0.45,
        float(plan.door_width) * 0.5 + INNOUT_INTERIOR_OPENING_SIDE_CLEARANCE_M,
    )
    bottom = float(plan.door_bottom_z) - INNOUT_INTERIOR_OPENING_VERTICAL_CLEARANCE_M
    top = (
        float(plan.door_bottom_z + plan.door_height)
        + INNOUT_INTERIOR_OPENING_VERTICAL_CLEARANCE_M
    )
    shape = primitive(
        "interior_doorway",
        [(-half_width, bottom), (half_width, bottom), (half_width, top), (-half_width, top)],
    )
    if shape is None:
        raise RuntimeError("invalid interior doorway opening rectangle")
    return WallOpeningClip(
        center=np.asarray(plan.door_center_xy, dtype=float),
        tangent=normalize(plan.tangent_xy),
        outward=normalize(plan.outward_normal_xy),
        half_width=half_width,
        bottom=bottom,
        top=top,
        outward_min=-(max(float(plan.interior_fit_inward_nudge_m), 0.02) + 0.5),
        outward_max=0.25,
        shape=shape,
    )


def doorway_space_points(
    points: list[Gf.Vec3f],
    local_to_world: Gf.Matrix4d,
    clip: WallOpeningClip,
) -> list[Gf.Vec3f]:
    result = []
    for source in points:
        point = local_to_world.Transform(Gf.Vec3d(source))
        offset = np.asarray([float(point[0]), float(point[1])]) - clip.center
        result.append(
            Gf.Vec3f(
                float(np.dot(offset, clip.tangent)),
                float(point[2]),
                float(np.dot(offset, clip.outward)),
            )
        )
    return result


def wall_mesh_overlaps_opening(points: list[Gf.Vec3f], clip: WallOpeningClip) -> bool:
    u_values = [float(point[0]) for point in points]
    z_values = [float(point[1]) for point in points]
    v_values = [float(point[2]) for point in points]
    return not (
        max(u_values) < -clip.half_width
        or min(u_values) > clip.half_width
        or max(z_values) < clip.bottom
        or min(z_values) > clip.top
        or max(v_values) < clip.outward_min
        or min(v_values) > clip.outward_max
    )


def wall_mesh_is_parallel_corridor_panel(
    points: list[Gf.Vec3f],
    clip: WallOpeningClip,
) -> bool:
    u_values = np.asarray([float(point[0]) for point in points], dtype=float)
    z_values = np.asarray([float(point[1]) for point in points], dtype=float)
    v_values = np.asarray([float(point[2]) for point in points], dtype=float)
    return bool(
        np.ptp(u_values) <= 0.05
        and np.ptp(z_values) >= 0.5
        and abs(float(np.mean(u_values))) < clip.half_width
        and z_values.max() >= clip.bottom
        and z_values.min() <= clip.top
        and v_values.min() >= clip.outward_min - 0.05
        and v_values.max() <= clip.outward_max + 0.05
    )


def mesh_interior_opening_blocking_faces(
    prim: Usd.Prim,
    clip: WallOpeningClip,
    xform_cache: UsdGeom.XformCache,
) -> list[int]:
    mesh = UsdGeom.Mesh(prim)
    points = list(mesh.GetPointsAttr().Get() or [])
    counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get() or []]
    indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get() or []]
    if not points or not counts or not indices:
        return []
    doorway_points = doorway_space_points(
        points,
        xform_cache.GetLocalToWorldTransform(prim),
        clip,
    )
    blocking = []
    offset = 0
    for face_index, count in enumerate(counts):
        face = indices[offset : offset + count]
        offset += count
        for triangle_index in range(1, count - 1):
            triangle = [
                ClipVertex(doorway_points[index], None, None)
                for index in (face[0], face[triangle_index], face[triangle_index + 1])
            ]
            outward_values = [float(vertex.point[2]) for vertex in triangle]
            if max(outward_values) < clip.outward_min or min(outward_values) > clip.outward_max:
                continue
            inside, _outside = carve_primitive(triangle, clip.shape)
            if any(polygon_xy_area(part) > 1e-6 for part in inside):
                blocking.append(face_index)
                break
    return blocking


def append_carved_wall_triangle(
    data: RegionMeshData,
    triangle: list[ClipVertex],
    material_path: str,
    clip: WallOpeningClip,
) -> bool:
    triangle_v = [float(vertex.point[2]) for vertex in triangle]
    if max(triangle_v) < clip.outward_min or min(triangle_v) > clip.outward_max:
        append_polygon(data, triangle, material_path)
        return False
    inside, outside = carve_primitive(triangle, clip.shape)
    if not inside:
        append_polygon(data, triangle, material_path)
        return False
    for part in outside:
        append_polygon(data, part, material_path)
    return True


def carve_wall_mesh_data(
    prim: Usd.Prim,
    points: list[Gf.Vec3f],
    counts: list[int],
    indices: list[int],
    doorway_points: list[Gf.Vec3f],
    clip: WallOpeningClip,
    corner_uvs: list[Gf.Vec2f] | None,
) -> tuple[RegionMeshData, int, str]:
    subset_faces = source_face_materials(prim, len(counts))
    direct_material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath().pathString
    face_materials = {
        int(face): material_path
        for material_path, faces in subset_faces.items()
        for face in faces
    }
    data = RegionMeshData()
    carved_faces = 0
    offset = 0
    for face_index, count in enumerate(counts):
        corners = [
            ClipVertex(
                doorway_points[indices[offset + corner]],
                corner_uvs[offset + corner] if corner_uvs is not None else None,
                None,
            )
            for corner in range(count)
        ]
        material_path = face_materials.get(face_index, direct_material)
        for triangle_index in range(1, count - 1):
            triangle = [corners[0], corners[triangle_index], corners[triangle_index + 1]]
            carved_faces += append_carved_wall_triangle(data, triangle, material_path, clip)
        offset += count
    return data, carved_faces, direct_material


def bind_carved_wall_materials(
    stage: Usd.Stage,
    prim: Usd.Prim,
    data: RegionMeshData,
    direct_material: str,
) -> None:
    for child in prim.GetChildren():
        if child.GetTypeName() == "GeomSubset":
            stage.OverridePrim(child.GetPath()).SetActive(False)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    for material_index, (material_path, faces) in enumerate(sorted(data.material_faces.items())):
        if not material_path or material_path == direct_material or not faces:
            continue
        subset = binding_api.CreateMaterialBindSubset(
            f"navverse_door_material_{material_index:03d}",
            faces,
            UsdGeom.Tokens.face,
        )
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if material:
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)


def write_carved_wall_mesh(
    stage: Usd.Stage,
    prim: Usd.Prim,
    mesh: UsdGeom.Mesh,
    data: RegionMeshData,
    direct_material: str,
    world_to_local: Gf.Matrix4d,
    clip: WallOpeningClip,
    uv_sets: dict[str, tuple[Any, list[Gf.Vec2f]]],
) -> None:
    local_points = []
    for point in data.points:
        world = Gf.Vec3d(
            float(clip.center[0] + clip.tangent[0] * point[0] + clip.outward[0] * point[2]),
            float(clip.center[1] + clip.tangent[1] * point[0] + clip.outward[1] * point[2]),
            float(point[1]),
        )
        local_points.append(Gf.Vec3f(world_to_local.Transform(world)))
    source_corner_count = len(local_points)
    for name, (_type_name, values) in uv_sets.items():
        if len(values) != source_corner_count:
            raise RuntimeError(
                f"carved UV set {name!r} on {prim.GetPath()} has {len(values)} values; "
                f"expected {source_corner_count}"
            )
    valid_face_indices = []
    normals = []
    for face_index, index in enumerate(range(0, source_corner_count, 3)):
        edge_a = Gf.Vec3d(local_points[index + 1]) - Gf.Vec3d(local_points[index])
        edge_b = Gf.Vec3d(local_points[index + 2]) - Gf.Vec3d(local_points[index])
        normal = Gf.Cross(edge_a, edge_b)
        if normal.GetLength() <= 1e-9:
            continue
        normal.Normalize()
        normal_f = Gf.Vec3f(normal)
        valid_face_indices.append(face_index)
        normals.extend([normal_f, normal_f, normal_f])
    if not valid_face_indices:
        raise RuntimeError(f"carving removed every non-degenerate face on {prim.GetPath()}")
    source_face_count = source_corner_count // 3
    skipped_face_count = source_face_count - len(valid_face_indices)
    if skipped_face_count:
        local_points = [
            point
            for face_index in valid_face_indices
            for point in local_points[face_index * 3 : face_index * 3 + 3]
        ]
        uv_sets = {
            name: (
                type_name,
                [
                    value
                    for face_index in valid_face_indices
                    for value in values[face_index * 3 : face_index * 3 + 3]
                ],
            )
            for name, (type_name, values) in uv_sets.items()
        }
        face_remap = {source: target for target, source in enumerate(valid_face_indices)}
        data.material_faces = {
            material_path: [face_remap[face] for face in faces if face in face_remap]
            for material_path, faces in data.material_faces.items()
        }
        data.skipped_degenerate_triangles += skipped_face_count
        prim.SetCustomDataByKey("navverse_carved_degenerate_face_count", skipped_face_count)
    face_count = len(local_points) // 3
    mesh.GetPointsAttr().Set(local_points)
    mesh.GetFaceVertexCountsAttr().Set([3] * face_count)
    mesh.GetFaceVertexIndicesAttr().Set(list(range(len(local_points))))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    for primvar in primvars_api.GetPrimvars():
        if primvar.GetInterpolation() == UsdGeom.Tokens.constant:
            continue
        primvar.GetAttr().Block()
        indices_attr = primvar.GetIndicesAttr()
        if indices_attr:
            indices_attr.Block()
    for name, (type_name, values) in uv_sets.items():
        uv_primvar = primvars_api.CreatePrimvar(
            name,
            type_name,
            UsdGeom.Tokens.faceVarying,
        )
        uv_primvar.Set(values)
        indices_attr = uv_primvar.GetIndicesAttr()
        if indices_attr:
            indices_attr.Block()
    mesh.GetNormalsAttr().Set(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    xs = [float(point[0]) for point in local_points]
    ys = [float(point[1]) for point in local_points]
    zs = [float(point[2]) for point in local_points]
    mesh.GetExtentAttr().Set(
        [Gf.Vec3f(min(xs), min(ys), min(zs)), Gf.Vec3f(max(xs), max(ys), max(zs))]
    )
    bind_carved_wall_materials(stage, prim, data, direct_material)
    prim.SetCustomDataByKey("navverse_interior_doorway_carved", True)


def carve_wall_mesh(
    stage: Usd.Stage,
    prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
    clip: WallOpeningClip,
) -> tuple[bool, int]:
    mesh = UsdGeom.Mesh(prim)
    points = list(mesh.GetPointsAttr().Get() or [])
    counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get() or []]
    indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get() or []]
    if not points or not counts or not indices:
        return False, 0
    local_to_world = xform_cache.GetLocalToWorldTransform(prim)
    doorway_points = doorway_space_points(points, local_to_world, clip)
    if not wall_mesh_overlaps_opening(doorway_points, clip):
        return False, 0
    source_uv_sets = wall_corner_uv_sets(prim, len(points), counts, indices)
    primary_uv_name = next(iter(source_uv_sets), None)
    primary_corner_uvs = source_uv_sets[primary_uv_name][1] if primary_uv_name else None
    data, carved_faces, direct_material = carve_wall_mesh_data(
        prim,
        points,
        counts,
        indices,
        doorway_points,
        clip,
        primary_corner_uvs,
    )
    if carved_faces:
        carved_uv_sets = {}
        if primary_uv_name:
            carved_uv_sets[primary_uv_name] = (
                source_uv_sets[primary_uv_name][0],
                list(data.uvs),
            )
        for name, (type_name, corner_uvs) in source_uv_sets.items():
            if name == primary_uv_name:
                continue
            uv_data, uv_carved_faces, _ = carve_wall_mesh_data(
                prim,
                points,
                counts,
                indices,
                doorway_points,
                clip,
                corner_uvs,
            )
            if uv_carved_faces != carved_faces or len(uv_data.points) != len(data.points):
                raise RuntimeError(
                    f"carved UV topology mismatch for {name!r} on {prim.GetPath()}"
                )
            carved_uv_sets[name] = (type_name, list(uv_data.uvs))
        write_carved_wall_mesh(
            stage,
            prim,
            mesh,
            data,
            direct_material,
            local_to_world.GetInverse(),
            clip,
            carved_uv_sets,
        )
    return True, carved_faces


def carve_interior_wall_opening(stage: Usd.Stage, plan: DoorPlan) -> dict[str, Any]:
    root_paths = (
        "/World/start_result_navigation/Meshes/Base/wall",
        "/World/start_result_navigation/Meshes/Base/other",
    )
    roots = [stage.GetPrimAtPath(path) for path in root_paths]
    roots = [root for root in roots if root]
    if not roots:
        return {"candidate_mesh_count": 0, "carved_mesh_count": 0, "carved_source_triangle_count": 0}
    clip = build_wall_opening_clip(plan)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    candidate_mesh_count = 0
    carved_source_triangle_count = 0
    carved_paths = []
    deactivated_parallel_paths = []
    for root in roots:
        is_other_root = root.GetPath().pathString.endswith("/other")
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() != "Mesh":
                continue
            points = list(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [])
            if points:
                doorway_points = doorway_space_points(
                    points,
                    xform_cache.GetLocalToWorldTransform(prim),
                    clip,
                )
                if wall_mesh_is_parallel_corridor_panel(doorway_points, clip):
                    candidate_mesh_count += 1
                    deactivated_parallel_paths.append(prim.GetPath().pathString)
                    continue
            if is_other_root and not mesh_interior_opening_blocking_faces(prim, clip, xform_cache):
                continue
            candidate, carved_faces = carve_wall_mesh(stage, prim, xform_cache, clip)
            candidate_mesh_count += int(candidate)
            carved_source_triangle_count += carved_faces
            if carved_faces:
                carved_paths.append(prim.GetPath().pathString)
    for path in deactivated_parallel_paths:
        stage.OverridePrim(path).SetActive(False)

    def find_remaining_blockers() -> list[tuple[Usd.Prim, list[int]]]:
        verify_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        remaining = []
        for root in roots:
            for prim in Usd.PrimRange(root):
                if prim.GetTypeName() != "Mesh" or not prim.IsActive():
                    continue
                blocking_faces = mesh_interior_opening_blocking_faces(prim, clip, verify_cache)
                if blocking_faces:
                    remaining.append((prim, blocking_faces))
        return remaining

    corrective_pass_count = 0
    remaining_blockers = find_remaining_blockers()
    for _attempt in range(2):
        if not remaining_blockers:
            break
        repair_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        repaired = False
        for prim, _blocking_faces in remaining_blockers:
            _candidate, carved_faces = carve_wall_mesh(stage, prim, repair_cache, clip)
            if not carved_faces:
                continue
            repaired = True
            carved_source_triangle_count += carved_faces
            path = prim.GetPath().pathString
            if path not in carved_paths:
                carved_paths.append(path)
        if not repaired:
            break
        corrective_pass_count += 1
        remaining_blockers = find_remaining_blockers()
    if remaining_blockers:
        remaining_summary = [
            {
                "prim_path": prim.GetPath().pathString,
                "face_count": len(blocking_faces),
            }
            for prim, blocking_faces in remaining_blockers
        ]
        raise RuntimeError(
            f"interior wall meshes still cross the doorway after carving: {remaining_summary}"
        )

    summary = {
        "candidate_mesh_count": candidate_mesh_count,
        "carved_mesh_count": len(carved_paths),
        "carved_source_triangle_count": carved_source_triangle_count,
        "carved_mesh_paths": carved_paths,
        "corrective_pass_count": corrective_pass_count,
        "deactivated_parallel_mesh_count": len(deactivated_parallel_paths),
        "deactivated_parallel_mesh_paths": deactivated_parallel_paths,
        "source_root_paths": [root.GetPath().pathString for root in roots],
        "half_width_m": clip.half_width,
        "bottom_z": clip.bottom,
        "top_z": clip.top,
        "outward_min_m": clip.outward_min,
        "outward_max_m": clip.outward_max,
        "verified_blocking_face_count": 0,
    }
    roots[0].SetCustomDataByKey("navverse_interior_wall_opening_summary", json.dumps(summary, sort_keys=True))
    return summary


def mesh_doorway_blocking_faces(
    prim: Usd.Prim,
    plan: DoorPlan,
    xform_cache: UsdGeom.XformCache,
) -> list[int]:
    mesh = UsdGeom.Mesh(prim)
    points = list(mesh.GetPointsAttr().Get() or [])
    counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get() or []]
    indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get() or []]
    if not points or not counts or not indices:
        return []
    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, count - 1))
        offset += count
    if not triangles:
        return []
    transform = xform_cache.GetLocalToWorldTransform(prim)
    world_points = np.asarray(
        [tuple(transform.Transform(Gf.Vec3d(point))) for point in points],
        dtype=float,
    )
    return doorway_blocking_faces(plan, world_points, np.asarray(triangles, dtype=np.int64))


def build_exterior_wall_opening_clip(plan: DoorPlan) -> WallOpeningClip:
    half_width = max(float(plan.door_width) * 0.5, float(plan.building_opening_width) * 0.5 - 0.05)
    bottom = float(plan.building_opening_bottom_z) + 0.02
    top = float(plan.building_opening_top_z) - 0.05
    shape = primitive(
        "exterior_doorway",
        [(-half_width, bottom), (half_width, bottom), (half_width, top), (-half_width, top)],
    )
    if shape is None:
        raise RuntimeError("invalid exterior doorway opening rectangle")
    return WallOpeningClip(
        center=np.asarray(plan.door_center_xy, dtype=float),
        tangent=normalize(plan.tangent_xy),
        outward=normalize(plan.outward_normal_xy),
        half_width=half_width,
        bottom=bottom,
        top=top,
        outward_min=-0.25,
        outward_max=THRESHOLD_OUTER_DISTANCE_M + 1.0,
        shape=shape,
    )


def carve_overlapping_exterior_walls(stage: Usd.Stage, plan: DoorPlan) -> dict[str, Any]:
    city_root = stage.GetPrimAtPath("/World/city")
    if not city_root:
        return {"candidate_mesh_count": 0, "carved_mesh_count": 0, "carved_source_triangle_count": 0}
    selected_cut_path = Sdf.Path(plan.target_prim_path + "_cut")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    clip = build_exterior_wall_opening_clip(plan)
    candidates = []
    for prim in Usd.PrimRange(city_root):
        if prim.GetTypeName() != "Mesh" or not prim.IsActive():
            continue
        if prim.GetPath().HasPrefix(selected_cut_path):
            continue
        blocking = mesh_doorway_blocking_faces(prim, plan, xform_cache)
        if blocking:
            candidates.append((prim, len(blocking)))

    carved_paths = []
    carved_source_triangle_count = 0
    for prim, _blocking_face_count in candidates:
        _candidate, carved_faces = carve_wall_mesh(stage, prim, xform_cache, clip)
        carved_source_triangle_count += carved_faces
        if carved_faces:
            carved_paths.append(prim.GetPath().pathString)

    remaining = []
    verify_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim, _blocking_face_count in candidates:
        blocking = mesh_doorway_blocking_faces(prim, plan, verify_cache)
        if blocking:
            remaining.append({"prim_path": prim.GetPath().pathString, "face_count": len(blocking)})
    if remaining:
        raise RuntimeError(f"city meshes still cross the exterior doorway after carving: {remaining}")

    summary = {
        "candidate_mesh_count": len(candidates),
        "carved_mesh_count": len(carved_paths),
        "carved_source_triangle_count": carved_source_triangle_count,
        "carved_mesh_paths": carved_paths,
        "half_width_m": clip.half_width,
        "bottom_z": clip.bottom,
        "top_z": clip.top,
        "outward_min_m": clip.outward_min,
        "outward_max_m": clip.outward_max,
    }
    city_root.SetCustomDataByKey("navverse_exterior_wall_opening_summary", json.dumps(summary, sort_keys=True))
    return summary


def clear_exterior_doorway_corridor(stage: Usd.Stage, plan: DoorPlan) -> list[str]:
    roots = [stage.GetPrimAtPath(path) for path in ("/World/RoadObjects", "/World/Objaverse")]
    candidates = [child for root in roots if root for child in root.GetChildren()]
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    half_width = max(0.7, float(plan.door_width) * 0.45)
    bottom = float(plan.door_bottom_z) + 0.05
    top = float(plan.door_bottom_z) + min(float(plan.door_height) - 0.1, 2.0)
    room_footprint = np.asarray(interior_room_hole_footprint(plan), dtype=float)
    room_bottom = float(plan.door_bottom_z) - INNOUT_ROOM_OBJECT_CLEARANCE_M
    room_top = float(plan.interior_world_top_z) + INNOUT_ROOM_OBJECT_CLEARANCE_M
    bbox_cache = doorway_bbox_cache()
    removed = []
    for prim in candidates:
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        corridor_overlap = bounds_intersect_door_corridor(
            bounds,
            center,
            tangent,
            outward,
            half_width,
            -0.1,
            2.5,
            bottom,
            top,
        )
        room_overlap = False
        if not bounds.IsEmpty():
            minimum = bounds.GetMin()
            maximum = bounds.GetMax()
            bounds_center = np.asarray(
                [
                    (float(minimum[0]) + float(maximum[0])) * 0.5,
                    (float(minimum[1]) + float(maximum[1])) * 0.5,
                ],
                dtype=float,
            )
            bounds_radius = 0.5 * math.hypot(
                float(maximum[0]) - float(minimum[0]),
                float(maximum[1]) - float(minimum[1]),
            )
            z_overlap = not (float(maximum[2]) < room_bottom or float(minimum[2]) > room_top)
            room_overlap = z_overlap and point_in_polygon(
                bounds_center,
                room_footprint,
                tolerance=bounds_radius + INNOUT_ROOM_OBJECT_CLEARANCE_M,
            )
        if corridor_overlap or room_overlap:
            removed.append(prim.GetPath().pathString)
    for path in removed:
        stage.OverridePrim(path).SetActive(False)
    stage.GetPrimAtPath("/World").SetCustomDataByKey(
        "navverse_removed_exterior_doorway_obstacles",
        json.dumps(removed),
    )
    stage.GetPrimAtPath("/World").SetCustomDataByKey(
        "navverse_innout_room_object_clearance",
        json.dumps(
            {
                "clearance_m": INNOUT_ROOM_OBJECT_CLEARANCE_M,
                "footprint_xy": room_footprint.tolist(),
                "removed_paths": removed,
            },
            sort_keys=True,
        ),
    )
    return removed


def threshold_smoothstep(value: float) -> float:
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def threshold_grid(
    plan: DoorPlan,
    landing_distance: float = THRESHOLD_LANDING_DISTANCE_M,
    outer_distance: float = THRESHOLD_OUTER_DISTANCE_M,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    half_width = float(plan.door_width) * 0.5
    outer_half_width = half_width + THRESHOLD_LATERAL_BLEND_M
    longitudinal = np.concatenate(
        (
            np.linspace(THRESHOLD_START_DISTANCE_M, landing_distance, 5),
            np.linspace(landing_distance, outer_distance, 6)[1:],
        )
    )
    lateral = np.asarray(
        [
            -outer_half_width,
            -half_width,
            -half_width * 0.5,
            0.0,
            half_width * 0.5,
            half_width,
            outer_half_width,
        ],
        dtype=float,
    )
    sample_xy = [
        [float(value) for value in center + outward * distance + tangent * side]
        for distance in longitudinal
        for side in lateral
    ]
    return longitudinal, lateral, sample_xy


def carve_threshold_terrain(
    stage: Usd.Stage,
    world_polygon: list[list[float]],
) -> list[dict[str, Any]]:
    terrain_regions = stage.GetPrimAtPath("/World/TerrainRegions")
    summaries = []
    if not terrain_regions:
        return summaries
    for child in list(terrain_regions.GetChildren()):
        if child.GetTypeName() != "Mesh":
            continue
        summary = subtract_convex_polygon_from_mesh(
            stage,
            child.GetPath().pathString,
            world_polygon,
        )
        if summary:
            summaries.append(summary)
    terrain_regions.SetCustomDataByKey(
        "navverse_innout_threshold_cut",
        json.dumps({"polygon_xy": world_polygon, "meshes": summaries}, sort_keys=True),
    )
    return summaries


def bind_threshold_materials(
    stage: Usd.Stage,
    mesh: UsdGeom.Mesh,
    face_materials: list[str],
) -> str:
    material_counts: dict[str, int] = {}
    for material_path in face_materials:
        if material_path:
            material_counts[material_path] = material_counts.get(material_path, 0) + 1
    if material_counts:
        direct_material_path = max(material_counts, key=material_counts.get)
        direct_material = UsdShade.Material(stage.GetPrimAtPath(direct_material_path))
    else:
        direct_material_path = "/World/Looks/NavVerseInnOutThreshold"
        direct_material = UsdShade.Material.Define(stage, direct_material_path)
        shader = UsdShade.Shader.Define(stage, f"{direct_material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.03, 0.04, 0.04))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        direct_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding_api.Bind(
        direct_material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    subset_faces: dict[str, list[int]] = {}
    for face_index, material_path in enumerate(face_materials):
        if material_path and material_path != direct_material_path:
            subset_faces.setdefault(material_path, []).append(face_index)
    for material_index, (material_path, face_indices) in enumerate(sorted(subset_faces.items())):
        subset = binding_api.CreateMaterialBindSubset(
            f"threshold_material_{material_index:03d}",
            face_indices,
            UsdGeom.Tokens.face,
        )
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if material:
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    return direct_material_path


def threshold_surface_profile(
    plan: DoorPlan,
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    samples: list[dict[str, Any]],
    landing_distance: float,
    outer_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    row_count = len(longitudinal)
    column_count = len(lateral)
    terrain_z = np.asarray([float(sample["z"]) for sample in samples], dtype=float).reshape(
        row_count,
        column_count,
    )
    floor_z = float(plan.door_bottom_z) + THRESHOLD_SURFACE_OFFSET_M
    top_z = np.empty_like(terrain_z)
    half_width = float(plan.door_width) * 0.5
    for row, distance in enumerate(longitudinal):
        length_fraction = threshold_smoothstep(
            (float(distance) - landing_distance)
            / (outer_distance - landing_distance)
        )
        for column, side in enumerate(lateral):
            center_profile_z = floor_z + (terrain_z[row, column] - floor_z) * length_fraction
            side_fraction = threshold_smoothstep(
                (abs(float(side)) - half_width) / THRESHOLD_LATERAL_BLEND_M
            )
            top_z[row, column] = center_profile_z + (
                terrain_z[row, column] - center_profile_z
            ) * side_fraction
    return terrain_z, top_z, top_z - THRESHOLD_BASE_CLEARANCE_M, floor_z


def threshold_point_index(
    row: int,
    column: int,
    row_count: int,
    column_count: int,
    *,
    bottom: bool = False,
) -> int:
    index = row * column_count + column
    return index + row_count * column_count if bottom else index


def threshold_mesh_faces(
    row_count: int,
    column_count: int,
    sample_materials: list[str],
    *,
    reverse_top_winding: bool,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    faces: list[tuple[int, int, int]] = []
    face_materials: list[str] = []
    for row in range(row_count - 1):
        for column in range(column_count - 1):
            top_a = threshold_point_index(row, column, row_count, column_count)
            top_b = threshold_point_index(row, column + 1, row_count, column_count)
            top_c = threshold_point_index(row + 1, column + 1, row_count, column_count)
            top_d = threshold_point_index(row + 1, column, row_count, column_count)
            material_path = sample_materials[top_a]
            top_faces = [(top_a, top_b, top_c), (top_a, top_c, top_d)]
            if reverse_top_winding:
                top_faces = [tuple(reversed(face)) for face in top_faces]
            faces.extend(top_faces)
            face_materials.extend((material_path, material_path))
            point_count = row_count * column_count
            faces.extend(
                tuple(index + point_count for index in reversed(face))
                for face in top_faces
            )
            face_materials.extend((material_path, material_path))

    perimeter = (
        [threshold_point_index(0, column, row_count, column_count) for column in range(column_count)]
        + [
            threshold_point_index(row, column_count - 1, row_count, column_count)
            for row in range(1, row_count)
        ]
        + [
            threshold_point_index(row_count - 1, column, row_count, column_count)
            for column in range(column_count - 2, -1, -1)
        ]
        + [
            threshold_point_index(row, 0, row_count, column_count)
            for row in range(row_count - 2, 0, -1)
        ]
    )
    if reverse_top_winding:
        perimeter = list(reversed(perimeter))
    point_count = row_count * column_count
    for edge_index, top_a in enumerate(perimeter):
        top_b = perimeter[(edge_index + 1) % len(perimeter)]
        bottom_a = top_a + point_count
        bottom_b = top_b + point_count
        faces.extend(((top_a, bottom_a, bottom_b), (top_a, bottom_b, top_b)))
        face_materials.extend((sample_materials[top_a], sample_materials[top_a]))
    return faces, face_materials


def threshold_mesh_geometry(
    plan: DoorPlan,
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    top_z: np.ndarray,
    bottom_z: np.ndarray,
    samples: list[dict[str, Any]],
) -> tuple[list[list[float]], list[list[float]], list[tuple[int, int, int]], list[str]]:
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    top_points = []
    for row, distance in enumerate(longitudinal):
        for column, side in enumerate(lateral):
            xy = center + outward * float(distance) + tangent * float(side)
            top_points.append([float(xy[0]), float(xy[1]), float(top_z[row, column])])
    points = top_points + [
        [point[0], point[1], float(bottom_z.reshape(-1)[index])]
        for index, point in enumerate(top_points)
    ]
    point_uvs = [sample["uv"] for sample in samples] * 2
    sample_materials = [str(sample.get("material_path") or "") for sample in samples]
    tangent_outward_cross_z = float(tangent[0] * outward[1] - tangent[1] * outward[0])
    faces, face_materials = threshold_mesh_faces(
        len(longitudinal),
        len(lateral),
        sample_materials,
        reverse_top_winding=tangent_outward_cross_z < 0.0,
    )
    return points, point_uvs, faces, face_materials


def polygon_normal_z(points: list[list[float]]) -> float:
    vertices = np.asarray(points, dtype=float)
    return float(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])[2])


def write_threshold_mesh(
    stage: Usd.Stage,
    root_path: str,
    points: list[list[float]],
    point_uvs: list[list[float]],
    faces: list[tuple[int, int, int]],
    face_materials: list[str],
) -> tuple[UsdGeom.Mesh, str]:
    indices = [index for face in faces for index in face]
    mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/ramp")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(*(min(point[index] for point in points) for index in range(3))),
            Gf.Vec3f(*(max(point[index] for point in points) for index in range(3))),
        ]
    )
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "UVMap",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    ).Set([Gf.Vec2f(*point_uvs[index]) for index in indices])
    return mesh, bind_threshold_materials(stage, mesh, face_materials)


def write_threshold_collision_proxy(
    stage: Usd.Stage,
    root_path: str,
    plan: DoorPlan,
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    top_z: np.ndarray,
    landing_distance: float,
) -> dict[str, Any]:
    row_indices = [
        0,
        int(np.argmin(np.abs(longitudinal - float(landing_distance)))),
        len(longitudinal) - 1,
    ]
    column_indices = [0, 1, len(lateral) - 2, len(lateral) - 1]
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    points = []
    for row in row_indices:
        for column in column_indices:
            xy = center + outward * float(longitudinal[row]) + tangent * float(lateral[column])
            points.append([float(xy[0]), float(xy[1]), float(top_z[row, column])])

    row_count = len(row_indices)
    column_count = len(column_indices)
    reverse_winding = float(tangent[0] * outward[1] - tangent[1] * outward[0]) < 0.0
    faces = []
    for row in range(row_count - 1):
        for column in range(column_count - 1):
            a = row * column_count + column
            b = a + 1
            d = (row + 1) * column_count + column
            c = d + 1
            cell_faces = [(a, b, c), (a, c, d)]
            if reverse_winding:
                cell_faces = [tuple(reversed(face)) for face in cell_faces]
            faces.extend(cell_faces)

    proxy_path = f"{root_path}/collision_proxy"
    mesh = UsdGeom.Mesh.Define(stage, proxy_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(*(min(point[index] for point in points) for index in range(3))),
            Gf.Vec3f(*(max(point[index] for point in points) for index in range(3))),
        ]
    )
    UsdGeom.Imageable(mesh.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    collision = apply_collision_apis_under(stage, proxy_path)
    selected_longitudinal = longitudinal[row_indices]
    selected_lateral = lateral[column_indices]
    selected_top_z = top_z[np.ix_(row_indices, column_indices)]
    _boundary_gap, max_grade = threshold_surface_metrics(
        selected_longitudinal,
        selected_lateral,
        selected_top_z,
        selected_top_z,
    )
    summary = {
        "prim_path": proxy_path,
        "top_only": True,
        "row_count": row_count,
        "column_count": column_count,
        "triangle_count": len(faces),
        "top_normal_z": polygon_normal_z([points[index] for index in faces[0]]),
        "max_surface_grade": max_grade,
        "max_surface_slope_deg": math.degrees(math.atan(max_grade)),
        "longitudinal_distances_m": [float(value) for value in selected_longitudinal],
        "lateral_distances_m": [float(value) for value in selected_lateral],
        "collision": collision,
    }
    mesh.GetPrim().SetCustomDataByKey(
        "navverse_threshold_collision_proxy",
        json.dumps(summary, sort_keys=True),
    )
    return summary


def threshold_cut_polygon(plan: DoorPlan, outer_distance: float) -> list[list[float]]:
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    outer_half_width = float(plan.door_width) * 0.5 + THRESHOLD_LATERAL_BLEND_M
    inner_center = center + outward * THRESHOLD_START_DISTANCE_M
    outer_center = center + outward * outer_distance
    return [
        [float(value) for value in inner_center - tangent * outer_half_width],
        [float(value) for value in inner_center + tangent * outer_half_width],
        [float(value) for value in outer_center + tangent * outer_half_width],
        [float(value) for value in outer_center - tangent * outer_half_width],
    ]


def threshold_surface_metrics(
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    terrain_z: np.ndarray,
    top_z: np.ndarray,
) -> tuple[float, float]:
    row_count, column_count = top_z.shape
    boundary_errors = []
    for row in range(row_count):
        for column in range(column_count):
            if row == row_count - 1 or column in {0, column_count - 1}:
                boundary_errors.append(abs(float(top_z[row, column] - terrain_z[row, column])))
    grades = []
    for row in range(row_count - 1):
        distance = float(longitudinal[row + 1] - longitudinal[row])
        grades.extend(np.abs(top_z[row + 1] - top_z[row]) / max(distance, 1e-9))
    for column in range(column_count - 1):
        distance = float(lateral[column + 1] - lateral[column])
        grades.extend(np.abs(top_z[:, column + 1] - top_z[:, column]) / max(distance, 1e-9))
    max_grade = max((float(value) for value in grades), default=0.0)
    return max(boundary_errors, default=0.0), max_grade


def adaptive_threshold_distances(
    longitudinal: np.ndarray,
    terrain_z: np.ndarray,
) -> tuple[float, float, dict[str, Any] | None]:
    center_columns = terrain_z[:, 1:-1] if terrain_z.shape[1] > 2 else terrain_z
    terrain_profile = np.median(center_columns, axis=1)
    terrain_steps = np.abs(np.diff(terrain_profile))
    for index in np.flatnonzero(terrain_steps > THRESHOLD_TERRAIN_STEP_LIMIT_M):
        outer_distance = float(longitudinal[index])
        if outer_distance < THRESHOLD_MIN_OUTER_DISTANCE_M:
            raise RuntimeError(
                "terrain step is too close to the doorway for a safe threshold transition: "
                f"distance={outer_distance:.3f}m step={terrain_steps[index]:.3f}m"
            )
        landing_distance = max(
            THRESHOLD_START_DISTANCE_M,
            outer_distance - THRESHOLD_BLEND_DISTANCE_M,
        )
        return landing_distance, outer_distance, {
            "step_m": float(terrain_steps[index]),
            "step_after_distance_m": outer_distance,
            "next_sample_distance_m": float(longitudinal[index + 1]),
        }
    return THRESHOLD_LANDING_DISTANCE_M, THRESHOLD_OUTER_DISTANCE_M, None


def sample_threshold_profile(
    stage: Usd.Stage,
    plan: DoorPlan,
    landing_distance: float,
    outer_distance: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    longitudinal, lateral, sample_xy = threshold_grid(plan, landing_distance, outer_distance)
    samples = sample_terrain_uv_material(stage, sample_xy)
    if len(samples) != len(sample_xy) or any(sample.get("uv") is None for sample in samples):
        raise RuntimeError(
            f"could not sample complete terrain profile for InNOut threshold: "
            f"{len(samples)}/{len(sample_xy)}"
        )
    return longitudinal, lateral, samples


def add_threshold_ramp(stage: Usd.Stage, plan: DoorPlan) -> dict[str, Any]:
    root_path = "/World/InnOutThreshold"
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    root = stage.DefinePrim(root_path, "Xform")
    half_width = float(plan.door_width) * 0.5
    outer_half_width = half_width + THRESHOLD_LATERAL_BLEND_M
    landing_distance = THRESHOLD_LANDING_DISTANCE_M
    outer_distance = THRESHOLD_OUTER_DISTANCE_M
    longitudinal, lateral, samples = sample_threshold_profile(
        stage,
        plan,
        landing_distance,
        outer_distance,
    )
    initial_terrain_z = np.asarray([float(sample["z"]) for sample in samples], dtype=float).reshape(
        len(longitudinal),
        len(lateral),
    )
    landing_distance, outer_distance, terrain_step = adaptive_threshold_distances(
        longitudinal,
        initial_terrain_z,
    )
    if terrain_step is not None:
        longitudinal, lateral, samples = sample_threshold_profile(
            stage,
            plan,
            landing_distance,
            outer_distance,
        )
    terrain_z, top_z, bottom_z, floor_z = threshold_surface_profile(
        plan,
        longitudinal,
        lateral,
        samples,
        landing_distance,
        outer_distance,
    )
    points, point_uvs, faces, face_materials = threshold_mesh_geometry(
        plan,
        longitudinal,
        lateral,
        top_z,
        bottom_z,
        samples,
    )
    mesh, material_path = write_threshold_mesh(
        stage,
        root_path,
        points,
        point_uvs,
        faces,
        face_materials,
    )
    mesh.GetPrim().SetCustomDataByKey(
        INNOUT_COLLISION_ROLE_KEY,
        INNOUT_VISUAL_ONLY_COLLISION_ROLE,
    )
    collision_proxy = write_threshold_collision_proxy(
        stage,
        root_path,
        plan,
        longitudinal,
        lateral,
        top_z,
        landing_distance,
    )
    cut_polygon = threshold_cut_polygon(plan, outer_distance)
    carve_summary = carve_threshold_terrain(stage, cut_polygon)
    boundary_gap, max_grade = threshold_surface_metrics(longitudinal, lateral, terrain_z, top_z)
    row_count, column_count = top_z.shape
    summary = {
        "prim_path": mesh.GetPath().pathString,
        "inner_z": floor_z,
        "outer_left_z": float(top_z[-1, 1]),
        "outer_right_z": float(top_z[-1, -2]),
        "facade_left_z": floor_z,
        "facade_right_z": floor_z,
        "facade_base_z": floor_z - THRESHOLD_BASE_CLEARANCE_M,
        "bottom_z": float(np.min(bottom_z)),
        "building_opening_bottom_z": float(plan.building_opening_bottom_z),
        "solid": True,
        "depth_m": outer_distance - THRESHOLD_START_DISTANCE_M,
        "width_m": outer_half_width * 2.0,
        "door_width_m": half_width * 2.0,
        "start_distance_m": THRESHOLD_START_DISTANCE_M,
        "landing_distance_m": landing_distance,
        "outer_distance_m": outer_distance,
        "terrain_step_limit_m": THRESHOLD_TERRAIN_STEP_LIMIT_M,
        "terrain_step_limited": terrain_step is not None,
        "terrain_step": terrain_step,
        "lateral_blend_m": THRESHOLD_LATERAL_BLEND_M,
        "row_count": row_count,
        "column_count": column_count,
        "top_triangle_count": (row_count - 1) * (column_count - 1) * 2,
        "top_normal_z": polygon_normal_z([points[index] for index in faces[0]]),
        "bottom_normal_z": polygon_normal_z([points[index] for index in faces[2]]),
        "max_terrain_boundary_gap_m": boundary_gap,
        "max_surface_grade": max_grade,
        "max_surface_slope_deg": math.degrees(math.atan(max_grade)),
        "material_path": material_path,
        "collision_proxy": collision_proxy,
        "terrain_cut_polygon_xy": cut_polygon,
        "terrain_cut_meshes": carve_summary,
    }
    root.SetCustomDataByKey("navverse_threshold_summary", json.dumps(summary, sort_keys=True))
    return summary


def interior_door_depth_m(plan: DoorPlan) -> float:
    local_outward = rotate_xy(normalize(plan.outward_normal_xy), -float(plan.room_yaw_deg))
    axis = 0 if abs(float(local_outward[0])) >= abs(float(local_outward[1])) else 1
    local_depth = abs(
        float(plan.interior_door_bbox_max[axis]) - float(plan.interior_door_bbox_min[axis])
    )
    return local_depth * float(plan.room_effective_scale)


def add_door_connector(stage: Usd.Stage, plan: DoorPlan) -> dict[str, Any]:
    root_path = "/World/InnOutDoorConnector"
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    root = stage.DefinePrim(root_path, "Xform")
    center = np.asarray(plan.door_center_xy, dtype=float)
    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    door_depth = interior_door_depth_m(plan)
    depth = max(float(plan.interior_fit_inward_nudge_m) + door_depth, 0.02)
    outer_center = center + outward * DOOR_CONNECTOR_OVERLAP_M
    inner_center = center - outward * (depth + DOOR_CONNECTOR_OVERLAP_M)
    half_width = float(plan.door_width) * 0.5
    bottom = float(plan.door_bottom_z) + THRESHOLD_SURFACE_OFFSET_M
    top = float(plan.door_bottom_z + plan.door_height)

    outer_left = outer_center - tangent * half_width
    outer_right = outer_center + tangent * half_width
    inner_left = inner_center - tangent * half_width
    inner_right = inner_center + tangent * half_width
    surfaces = {
        "floor": [
            [*outer_left, bottom],
            [*outer_right, bottom],
            [*inner_right, bottom],
            [*inner_left, bottom],
        ],
        "ceiling": [
            [*outer_left, top],
            [*inner_left, top],
            [*inner_right, top],
            [*outer_right, top],
        ],
        "left_wall": [
            [*outer_left, bottom],
            [*inner_left, bottom],
            [*inner_left, top],
            [*outer_left, top],
        ],
        "right_wall": [
            [*outer_right, bottom],
            [*outer_right, top],
            [*inner_right, top],
            [*inner_right, bottom],
        ],
    }
    if polygon_normal_z(surfaces["floor"]) < 0.0:
        surfaces["floor"].reverse()
    if polygon_normal_z(surfaces["ceiling"]) > 0.0:
        surfaces["ceiling"].reverse()
    material_path = "/World/Looks/NavVerseInnOutDoorConnector"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.28, 0.29, 0.3))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for name, points in surfaces.items():
        mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr(
            [
                Gf.Vec3f(*(min(point[index] for point in points) for index in range(3))),
                Gf.Vec3f(*(max(point[index] for point in points) for index in range(3))),
            ]
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
    apply_collision_apis_under(stage, root_path)
    summary = {
        "prim_path": root_path,
        "depth_m": float(np.linalg.norm(outer_center - inner_center)),
        "width_m": half_width * 2.0,
        "bottom_z": bottom,
        "top_z": top,
        "outer_overlap_m": DOOR_CONNECTOR_OVERLAP_M,
        "inner_overlap_m": DOOR_CONNECTOR_OVERLAP_M,
        "interior_door_depth_m": door_depth,
        "outer_center_xy": [float(value) for value in outer_center],
        "inner_center_xy": [float(value) for value in inner_center],
        "surface_count": len(surfaces),
        "floor_normal_z": polygon_normal_z(surfaces["floor"]),
        "ceiling_normal_z": polygon_normal_z(surfaces["ceiling"]),
        "material_path": material_path,
    }
    root.SetCustomDataByKey("navverse_door_connector_summary", json.dumps(summary, sort_keys=True))
    return summary


def sample_terrain_uv_material(
    stage: Usd.Stage,
    sample_xy: list[list[float]],
) -> list[dict[str, Any]]:
    prim = stage.GetPrimAtPath("/World/TerrainRegions/Terrain")
    if not prim or prim.GetTypeName() != "Mesh":
        return []
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray([tuple(point) for point in mesh.GetPointsAttr().Get() or []], dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=int)
    uv_attr = prim.GetAttribute("primvars:UVMap")
    uv_values = np.asarray(uv_attr.Get() if uv_attr else [], dtype=float)
    if not len(points) or not len(counts) or not len(indices) or not len(uv_values) or not np.all(counts == 3):
        return []
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    world_points = np.asarray([tuple(transform.Transform(Gf.Vec3d(*point))) for point in points], dtype=float)
    triangles = world_points[indices.reshape(-1, 3)]
    triangle_xy = triangles[:, :, :2]
    lower = triangle_xy.min(axis=1)
    upper = triangle_xy.max(axis=1)
    material_faces = source_face_materials(prim, len(counts))
    face_materials = {
        int(face_index): material_path
        for material_path, face_indices in material_faces.items()
        for face_index in face_indices
    }
    direct_material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath().pathString
    results = []
    for values in sample_xy:
        query = np.asarray(values, dtype=float)
        candidates = np.flatnonzero(
            (lower[:, 0] <= query[0])
            & (upper[:, 0] >= query[0])
            & (lower[:, 1] <= query[1])
            & (upper[:, 1] >= query[1])
        )
        best = None
        for face_index in candidates:
            a, b, c = triangle_xy[face_index]
            denominator = np.cross(b - a, c - a)
            if abs(denominator) <= 1e-12:
                continue
            weight_b = np.cross(query - a, c - a) / denominator
            weight_c = np.cross(b - a, query - a) / denominator
            weight_a = 1.0 - weight_b - weight_c
            if min(weight_a, weight_b, weight_c) < -1e-7:
                continue
            z = float(
                weight_a * triangles[face_index, 0, 2]
                + weight_b * triangles[face_index, 1, 2]
                + weight_c * triangles[face_index, 2, 2]
            )
            uv_triangle = uv_values[face_index * 3 : face_index * 3 + 3]
            uv = weight_a * uv_triangle[0] + weight_b * uv_triangle[1] + weight_c * uv_triangle[2]
            if best is None or z > best["z"]:
                best = {
                    "z": z,
                    "uv": [float(uv[0]), float(uv[1])],
                    "material_path": face_materials.get(int(face_index), direct_material),
                }
        if best is None:
            return []
        results.append(best)
    return results


def interior_room_hole_footprint(plan: DoorPlan) -> list[list[float]]:
    local_min = np.asarray(plan.interior_scene_bbox_min, dtype=float)[:2]
    local_max = np.asarray(plan.interior_scene_bbox_max, dtype=float)[:2]
    local_corners = np.asarray(
        [
            [local_min[0], local_min[1]],
            [local_max[0], local_min[1]],
            [local_max[0], local_max[1]],
            [local_min[0], local_max[1]],
        ],
        dtype=float,
    )
    yaw = math.radians(float(plan.room_yaw_deg))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    footprint = (
        local_corners * float(plan.room_effective_scale)
    ) @ rotation.T + np.asarray(plan.room_translate, dtype=float)[:2]
    centroid = footprint.mean(axis=0)
    expanded_footprint = []
    for point in footprint:
        direction = point - centroid
        length = float(np.linalg.norm(direction))
        expanded = point if length <= 1e-9 else point + direction / length * plan.terrain_hole_margin_m
        expanded_footprint.append([float(expanded[0]), float(expanded[1])])
    return expanded_footprint


def carve_terrain_for_innout(stage: Usd.Stage, plan: DoorPlan) -> None:
    terrain_regions = stage.GetPrimAtPath("/World/TerrainRegions")
    plan.terrain_hole_summary = []
    if not terrain_regions:
        return
    footprint = interior_room_hole_footprint(plan)
    for child in list(terrain_regions.GetChildren()):
        if child.GetTypeName() != "Mesh":
            continue
        summary = subtract_convex_polygon_from_mesh(stage, child.GetPath().pathString, footprint)
        if summary:
            plan.terrain_hole_summary.append(summary)
    terrain_regions.SetCustomDataByKey(
        "navverse_innout_terrain_hole",
        json.dumps(
            {
                "building": plan.building_glb,
                "interior_scene": plan.interior_scene,
                "margin_m": plan.terrain_hole_margin_m,
                "meshes": plan.terrain_hole_summary,
            },
            sort_keys=True,
        ),
    )


def replace_innout_building(stage: Usd.Stage, plan: DoorPlan, output_usd: Path) -> None:
    original = stage.OverridePrim(plan.target_prim_path)
    target_material_path = first_bound_material_path(original)
    original.SetActive(False)
    replacement_path = Sdf.Path(plan.target_prim_path + "_cut")
    replacement = stage.DefinePrim(replacement_path, "Xform")
    replacement.GetReferences().AddReference(
        os.path.relpath(plan.cut_usd, output_usd.parent),
        first_root_prim(Path(plan.cut_usd)),
    )
    if target_material_path is not None:
        material = UsdShade.Material(stage.GetPrimAtPath(target_material_path))
        replacement_mesh = stage.OverridePrim(replacement_path.AppendChild("mesh"))
        UsdShade.MaterialBindingAPI(replacement_mesh).Bind(material)


def add_innout_storefront(stage: Usd.Stage, plan: DoorPlan, output_usd: Path) -> None:
    store_path = "/World/store/innout_storefront"
    if stage.GetPrimAtPath(store_path):
        stage.RemovePrim(store_path)
    if plan.exterior_storefront_enabled and plan.storefront_usd:
        store = stage.DefinePrim(store_path, "Xform")
        store_xform = UsdGeom.XformCommonAPI(store)
        store_xform.SetTranslate(Gf.Vec3d(*plan.storefront_translate))
        store_xform.SetRotate(Gf.Vec3f(0.0, 0.0, plan.storefront_yaw_deg))
        scale = 100.0 * float(plan.storefront_scale_multiplier or 1.0)
        store_xform.SetScale(Gf.Vec3f(scale, scale, scale))
        store.GetReferences().AddReference(os.path.relpath(plan.storefront_usd, output_usd.parent))
    apply_collision_apis_under(stage, "/World/store")


def xy_bbox_overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_min = np.asarray(first["bbox_min"], dtype=float)
    first_max = np.asarray(first["bbox_max"], dtype=float)
    second_min = np.asarray(second["bbox_min"], dtype=float)
    second_max = np.asarray(second["bbox_max"], dtype=float)
    overlap = np.maximum(
        np.minimum(first_max[:2], second_max[:2]) - np.maximum(first_min[:2], second_min[:2]),
        0.0,
    )
    overlap_area = float(overlap[0] * overlap[1])
    smaller_area = min(float(first["xy_area_m2"]), float(second["xy_area_m2"]))
    return overlap_area / smaller_area if smaller_area > 1e-9 else 0.0


def descendant_mesh_world_bbox(
    prim: Usd.Prim,
    bbox_cache: UsdGeom.BBoxCache,
) -> tuple[np.ndarray, np.ndarray] | None:
    minima = []
    maxima = []
    for candidate in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not candidate.IsA(UsdGeom.Mesh):
            continue
        try:
            aligned = bbox_cache.ComputeWorldBound(candidate).ComputeAlignedBox()
            bbox_min = np.asarray(aligned.GetMin(), dtype=float)
            bbox_max = np.asarray(aligned.GetMax(), dtype=float)
        except Exception:
            continue
        if not np.all(np.isfinite([*bbox_min, *bbox_max])) or np.any(bbox_max < bbox_min):
            continue
        minima.append(bbox_min)
        maxima.append(bbox_max)
    if not minima:
        return None
    return np.min(np.asarray(minima), axis=0), np.max(np.asarray(maxima), axis=0)


def descendant_floor_geometry(
    prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
) -> tuple[float, int, np.ndarray, list[str]]:
    projected_area = 0.0
    triangle_count = 0
    world_points = []
    mesh_paths = []
    for candidate in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not candidate.IsActive() or not candidate.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(candidate)
        points = list(mesh.GetPointsAttr().Get() or [])
        counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get() or []]
        indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get() or []]
        if not points or not counts or not indices:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(candidate)
        transformed = np.asarray(
            [local_to_world.Transform(Gf.Vec3d(point)) for point in points],
            dtype=float,
        )
        world_points.append(transformed)
        mesh_paths.append(candidate.GetPath().pathString)
        offset = 0
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            for triangle_index in range(1, count - 1):
                triangle = transformed[[face[0], face[triangle_index], face[triangle_index + 1]]]
                edge_a = triangle[1, :2] - triangle[0, :2]
                edge_b = triangle[2, :2] - triangle[0, :2]
                projected_area += abs(
                    float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
                ) * 0.5
                triangle_count += 1
    points = np.concatenate(world_points, axis=0) if world_points else np.empty((0, 3), dtype=float)
    return projected_area, triangle_count, points, mesh_paths


def set_floor_collision_role(stage: Usd.Stage, mesh_paths: list[str], role: str) -> None:
    for path in mesh_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetCustomDataByKey(INNOUT_COLLISION_ROLE_KEY, role)


def add_synthetic_interior_floor(
    stage: Usd.Stage,
    plan: DoorPlan,
    room_path: str,
    footprint_xy: np.ndarray,
    material_path: Sdf.Path | None,
) -> dict[str, Any]:
    floor_path = f"{room_path}/NavVerseInteriorFloor"
    if stage.GetPrimAtPath(floor_path).IsValid():
        stage.RemovePrim(floor_path)

    footprint = simplify_closed_footprint(np.asarray(footprint_xy, dtype=float), tolerance=0.02)
    if len(footprint) < 3:
        raise RuntimeError("synthetic interior floor footprint has fewer than three vertices")
    signed_area = 0.5 * float(
        np.sum(
            footprint[:, 0] * np.roll(footprint[:, 1], -1)
            - footprint[:, 1] * np.roll(footprint[:, 0], -1)
        )
    )
    if signed_area < 0.0:
        footprint = footprint[::-1]

    floor_z = float(plan.door_bottom_z)
    points = np.column_stack([footprint, np.full(len(footprint), floor_z, dtype=float)])
    faces = [[0, index, index + 1] for index in range(1, len(points) - 1)]
    mesh = UsdGeom.Mesh.Define(stage, floor_path)
    UsdGeom.Xformable(mesh.GetPrim()).SetResetXformStack(True)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * len(points))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(*np.min(points, axis=0)),
            Gf.Vec3f(*np.max(points, axis=0)),
        ]
    )

    tangent = normalize(plan.tangent_xy)
    outward = normalize(plan.outward_normal_xy)
    centered = footprint - np.mean(footprint, axis=0)
    coordinates = np.column_stack([centered @ tangent, centered @ outward])
    spans = np.maximum(np.ptp(coordinates, axis=0), 1e-6)
    uv_values = (coordinates - np.min(coordinates, axis=0)) / max(float(spans[0]), float(spans[1]))
    primvars = UsdGeom.PrimvarsAPI(mesh)
    for name in ("st", "st1", "st2"):
        primvars.CreatePrimvar(
            name,
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        ).Set([Gf.Vec2f(float(value[0]), float(value[1])) for value in uv_values])

    bound_material_path = None
    if material_path is not None:
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if material:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            bound_material_path = material_path.pathString
    if bound_material_path is None:
        bound_material_path = "/World/Looks/NavVerseInteriorFloor"
        material = UsdShade.Material.Define(stage, bound_material_path)
        shader = UsdShade.Shader.Define(stage, f"{bound_material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.78, 0.79, 0.8)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    mesh.GetPrim().SetCustomDataByKey(INNOUT_COLLISION_ROLE_KEY, "physical")
    mesh.GetPrim().SetCustomDataByKey("navverse_generated_interior_floor", True)
    collision_summary = apply_collision_apis_under(stage, floor_path)
    return {
        "prim_path": floor_path,
        "vertex_count": len(points),
        "triangle_count": len(faces),
        "xy_area_m2": abs(signed_area),
        "floor_z_m": floor_z,
        "material_path": bound_material_path,
        "collision": collision_summary,
    }


def clean_duplicate_interior_floor(stage: Usd.Stage, plan: DoorPlan, room_path: str) -> dict[str, Any]:
    ground_path = f"{room_path}/Meshes/Base/ground"
    ground = stage.GetPrimAtPath(ground_path)
    summary: dict[str, Any] = {
        "applied": False,
        "ground_path": ground_path,
        "canonical_floor_path": None,
        "duplicate_floor_paths": [],
        "deactivated_paths": [],
    }
    if not ground.IsValid():
        summary["reason"] = "ground_prim_missing"
        return summary

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    records = []
    for child in ground.GetChildren():
        if not child.IsActive():
            continue
        bounds = descendant_mesh_world_bbox(child, bbox_cache)
        if bounds is None:
            continue
        bbox_min, bbox_max = bounds
        dims = bbox_max - bbox_min
        if not np.all(np.isfinite([*bbox_min, *bbox_max])) or np.any(dims[:2] <= 0.0):
            continue
        projected_area, triangle_count, world_points, mesh_paths = descendant_floor_geometry(
            child,
            xform_cache,
        )
        if triangle_count == 0:
            continue
        records.append(
            {
                "path": child.GetPath().pathString,
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "xy_area_m2": float(dims[0] * dims[1]),
                "triangle_xy_area_m2": projected_area,
                "triangle_count": triangle_count,
                "height_m": float(dims[2]),
                "world_points": world_points,
                "mesh_paths": mesh_paths,
                "material_path": first_bound_material_path(child),
            }
        )

    floor_z = float(plan.door_bottom_z)
    local_dims = np.asarray(plan.interior_scene_bbox_max, dtype=float) - np.asarray(
        plan.interior_scene_bbox_min,
        dtype=float,
    )
    room_area = float(local_dims[0] * local_dims[1] * plan.room_effective_scale**2)
    min_floor_area = max(10.0, 0.4 * room_area)
    large_flat_candidates = [
        record
        for record in records
        if abs(float(record["bbox_min"][2]) - floor_z) <= INNOUT_FLOOR_BASE_TOLERANCE_M
        and float(record["height_m"]) <= INNOUT_FLOOR_FLAT_TOLERANCE_M
        and float(record["xy_area_m2"]) >= min_floor_area
    ]
    large_flat_candidates.sort(
        key=lambda item: (-float(item["triangle_xy_area_m2"]), item["path"])
    )
    minimum_surface_area = room_area * INNOUT_FLOOR_MIN_SURFACE_COVERAGE_RATIO
    complete_candidates = [
        record
        for record in large_flat_candidates
        if float(record["triangle_xy_area_m2"]) >= minimum_surface_area
    ]
    summary["floor_z_m"] = floor_z
    summary["expected_room_area_m2"] = room_area
    summary["minimum_full_floor_area_m2"] = min_floor_area
    summary["minimum_surface_area_m2"] = minimum_surface_area
    summary["full_floor_candidate_paths"] = [item["path"] for item in complete_candidates]
    summary["large_flat_candidates"] = [
        {
            "path": item["path"],
            "bbox_area_m2": float(item["xy_area_m2"]),
            "triangle_area_m2": float(item["triangle_xy_area_m2"]),
            "triangle_count": int(item["triangle_count"]),
        }
        for item in large_flat_candidates
    ]

    if not complete_candidates:
        if not large_flat_candidates:
            summary["reason"] = "no_large_flat_floor_candidates"
            return summary
        hull_source = np.concatenate(
            [item["world_points"][:, :2] for item in large_flat_candidates],
            axis=0,
        )
        hull_source = np.unique(np.round(hull_source, 6), axis=0)
        if len(hull_source) < 3:
            raise RuntimeError("incomplete interior floor has too few footprint points")
        hull = ConvexHull(hull_source)
        hull_points = hull_source[hull.vertices]
        hull_coverage = float(hull.volume) / room_area if room_area > 1e-9 else 0.0
        if hull_coverage < INNOUT_FLOOR_MIN_HULL_COVERAGE_RATIO:
            raise RuntimeError(
                "incomplete interior floor cannot be reconstructed safely: "
                f"hull coverage {hull_coverage:.3f} is below "
                f"{INNOUT_FLOOR_MIN_HULL_COVERAGE_RATIO:.3f}"
            )
        base_visual = large_flat_candidates[0]
        for record in large_flat_candidates:
            set_floor_collision_role(
                stage,
                record["mesh_paths"],
                INNOUT_VISUAL_ONLY_COLLISION_ROLE,
            )
        base_instance_path = f"{base_visual['path']}/Instance"
        base_visual_path = (
            base_instance_path
            if stage.GetPrimAtPath(base_instance_path).IsValid()
            else base_visual["path"]
        )
        stage.OverridePrim(base_visual_path).SetActive(False)
        synthetic = add_synthetic_interior_floor(
            stage,
            plan,
            room_path,
            hull_points,
            base_visual["material_path"],
        )
        summary.update(
            {
                "applied": True,
                "reason": "sparse_floor_reconstructed",
                "canonical_floor_path": synthetic["prim_path"],
                "deactivated_paths": [base_visual_path],
                "deactivated_count": 1,
                "visual_only_mesh_paths": sorted(
                    path
                    for record in large_flat_candidates
                    for path in record["mesh_paths"]
                ),
                "source_floor_path": base_visual["path"],
                "source_triangle_area_m2": float(base_visual["triangle_xy_area_m2"]),
                "hull_area_m2": float(hull.volume),
                "hull_coverage_ratio": hull_coverage,
                "synthetic_floor": synthetic,
            }
        )
        ground.SetCustomDataByKey(
            "navverse_interior_floor_cleanup",
            json.dumps(summary, sort_keys=True),
        )
        return summary

    canonical = complete_candidates[0]
    duplicates = [
        candidate
        for candidate in complete_candidates[1:]
        if float(candidate["triangle_xy_area_m2"]) >= (
            float(canonical["triangle_xy_area_m2"]) * INNOUT_FLOOR_DUPLICATE_AREA_RATIO
        )
        and xy_bbox_overlap_ratio(canonical, candidate) >= INNOUT_FLOOR_DUPLICATE_OVERLAP_RATIO
        and abs(float(candidate["bbox_min"][2]) - float(canonical["bbox_min"][2]))
        <= INNOUT_FLOOR_BASE_TOLERANCE_M
    ]
    if not duplicates:
        summary["reason"] = "complete_floor"
        summary["canonical_floor_path"] = canonical["path"]
        summary["canonical_triangle_area_m2"] = float(canonical["triangle_xy_area_m2"])
        return summary

    deactivated = []
    for record in duplicates:
        instance_path = f"{record['path']}/Instance"
        target_path = instance_path if stage.GetPrimAtPath(instance_path).IsValid() else record["path"]
        stage.OverridePrim(target_path).SetActive(False)
        deactivated.append(target_path)

    summary.update(
        {
            "applied": True,
            "reason": "duplicate_full_floor_detected",
            "canonical_floor_path": canonical["path"],
            "duplicate_floor_paths": [item["path"] for item in duplicates],
            "deactivated_paths": sorted(deactivated),
            "deactivated_count": len(deactivated),
            "canonical_triangle_area_m2": float(canonical["triangle_xy_area_m2"]),
        }
    )
    ground.SetCustomDataByKey("navverse_interior_floor_cleanup", json.dumps(summary, sort_keys=True))
    return summary


def add_innout_room(stage: Usd.Stage, plan: DoorPlan, output_usd: Path) -> None:
    room_path = "/World/start_result_navigation"
    if stage.GetPrimAtPath(room_path):
        stage.RemovePrim(room_path)
    room = stage.DefinePrim(room_path, "Xform")
    room_xform = UsdGeom.Xformable(room)
    room_xform.AddTranslateOp().Set(Gf.Vec3d(*plan.room_translate))
    yaw = math.radians(plan.room_yaw_deg)
    room_xform.AddOrientOp().Set(
        Gf.Quatf(math.cos(yaw * 0.5), Gf.Vec3f(0.0, 0.0, math.sin(yaw * 0.5)))
    )
    room_xform.AddScaleOp().Set(Gf.Vec3d(plan.room_scale, plan.room_scale, plan.room_scale))
    room.GetReferences().AddReference(os.path.relpath(plan.interior_usd, output_usd.parent))
    stage.OverridePrim(f"{room_path}/__default_setting/HDR_Sphere").SetActive(False)
    if plan.removed_interior_door_path:
        stage.OverridePrim(plan.removed_interior_door_path).SetActive(False)
    plan.interior_floor_cleanup_summary = clean_duplicate_interior_floor(stage, plan, room_path)


def finish_innout_doorway(stage: Usd.Stage, plan: DoorPlan, output_usd: Path) -> None:
    plan.exterior_wall_opening_summary = carve_overlapping_exterior_walls(stage, plan)
    plan.interior_wall_opening_summary = carve_interior_wall_opening(stage, plan)
    plan.removed_interior_obstacle_paths = clear_interior_doorway_corridor(stage, plan)
    plan.removed_exterior_obstacle_paths = clear_exterior_doorway_corridor(stage, plan)
    plan.door_connector_summary = add_door_connector(stage, plan)
    plan.threshold_summary = add_threshold_ramp(stage, plan)
    exit_sign_summary = apply_exit_sign_to_stage(stage, asdict(plan), output_dir=output_usd.parent)
    stage.DefinePrim("/World/Objaverse", "Xform").SetCustomDataByKey("navverse_has_innout_exit_sign", True)
    stage.GetPrimAtPath(exit_sign_summary["prim_path"]).SetCustomDataByKey(
        "navverse_summary",
        exit_sign_summary,
    )


def compose_scene(plan: DoorPlan) -> list[str]:
    base_usd = Path(plan.base_usd)
    output_usd = Path(plan.output_usd)
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    temporary_usd = output_usd.with_name(
        f".{output_usd.stem}.{os.getpid()}.tmp{output_usd.suffix}"
    )
    temporary_usd.unlink(missing_ok=True)

    root_layer = Sdf.Layer.FindOrOpen(str(base_usd))
    if root_layer is None:
        raise RuntimeError(f"could not open base layer: {base_usd}")
    try:
        out_layer = Sdf.Layer.CreateNew(str(temporary_usd))
        out_layer.TransferContent(root_layer)
        stage = Usd.Stage.Open(out_layer)
        removed_storefronts = remove_excluded_storefront_prims(stage, plan)
        carve_terrain_for_innout(stage, plan)
        replace_innout_building(stage, plan, output_usd)
        add_innout_storefront(stage, plan, output_usd)
        add_innout_room(stage, plan, output_usd)
        finish_innout_doorway(stage, plan, output_usd)

        add_preview_sky(
            stage,
            output_usd,
            output_usd.parents[1].parent / "nvidia" / "sky" / "CloudySky.usd",
        )
        add_innout_preview_cameras(stage, plan)
        out_layer.Save()
        os.replace(temporary_usd, output_usd)
        return removed_storefronts
    finally:
        temporary_usd.unlink(missing_ok=True)


def render_qa(plan_path: Path, plan: DoorPlan, args: argparse.Namespace) -> None:
    if args.skip_qa_render:
        return
    blender = shutil.which(args.blender) or args.blender
    cmd = [
        blender,
        "--background",
        "--python",
        str(args.render_helper),
        "--",
        "--plan",
        str(plan_path),
        "--output-dir",
        plan.qa_dir,
    ]
    if plan.exterior_storefront_enabled and plan.storefront_usd:
        cmd.extend(["--storefront-usd", plan.storefront_usd])
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CONDA_PREFIX"):
        env.pop(key, None)
    subprocess.check_call(cmd, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose final VC Full innout USD and doorway QA.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-dataset", default="vc_full")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--base-usd", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--storefront-plan", type=Path)
    parser.add_argument("--storefront-openings-manifest", type=Path)
    parser.add_argument("--anchor-plan", type=Path)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--interior-scene")
    parser.add_argument("--interior-usd")
    parser.add_argument("--door-width", type=float, default=None, help="Manual outdoor cut width override. Defaults to a fixed GR-scene doorway width.")
    parser.add_argument("--door-height", type=float, default=None, help="Manual outdoor cut height override. Defaults to a fixed GR-scene doorway height.")
    parser.add_argument("--door-depth", type=float, default=16.0)
    parser.add_argument("--fit-door-to-storefront", action="store_true", help="Size the cut from the nearest storefront bbox for storefront-specific QA.")
    parser.add_argument("--add-exterior-storefront", action="store_true", help="Debug/legacy option: add an exterior storefront facade at the innout doorway.")
    parser.add_argument("--door-width-margin", type=float, default=0.35)
    parser.add_argument("--door-height-margin", type=float, default=0.45)
    parser.add_argument("--min-door-width", type=float, default=2.6)
    parser.add_argument("--max-door-width", type=float, default=5.2)
    parser.add_argument("--min-door-height", type=float, default=2.4)
    parser.add_argument("--max-door-height", type=float, default=3.4)
    parser.add_argument("--building-opening-width-margin", type=float, default=0.8)
    parser.add_argument("--building-opening-height-margin", type=float, default=0.3)
    parser.add_argument("--min-building-opening-width", type=float, default=3.2)
    parser.add_argument("--max-building-opening-width", type=float, default=64.0)
    parser.add_argument("--max-door-above-building-bottom-m", type=float, default=4.5)
    parser.add_argument("--max-door-below-building-bottom-m", type=float, default=0.5)
    parser.add_argument("--max-door-above-building-top-m", type=float, default=0.3)
    parser.add_argument("--max-exterior-public-distance-m", type=float, default=6.0)
    parser.add_argument("--max-exterior-public-height-delta-m", type=float, default=2.5)
    parser.add_argument("--min-exterior-public-distance-hits", type=int, default=2)
    parser.add_argument("--min-indoor-doorway-inset-m", type=float, default=0.02)
    parser.add_argument("--min-wall-length", type=float, default=6.0)
    parser.add_argument("--storefront-offset-m", type=float, default=0.15)
    parser.add_argument("--storefront-target-height", type=float, default=4.0)
    parser.add_argument("--content-radius-m", type=float, default=50.0)
    parser.add_argument("--min-local-storefront-capacity", type=int, default=8)
    parser.add_argument("--min-local-object-targets", type=int, default=4)
    parser.add_argument("--max-bbox-match-dist", type=float, default=2.0)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--cut-helper", type=Path, required=True)
    parser.add_argument("--render-helper", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-qa-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    if args.plan_only:
        write_json(args.plan_output, asdict(plan))
        return
    run_blender_cut(plan, args)
    plan.removed_storefront_prim_paths = compose_scene(plan)
    write_json(args.plan_output, asdict(plan))
    render_qa(args.plan_output, plan, args)
    print(f"Wrote final innout scene: {plan.output_usd}")
    print(f"Wrote QA renders: {plan.qa_dir}")


if __name__ == "__main__":
    main()
