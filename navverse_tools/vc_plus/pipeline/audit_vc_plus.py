#!/usr/bin/env python3
"""Audit generated VC+ outdoor and innout scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from compose_vc_full_innout import (
    INNOUT_COLLISION_ROLE_KEY,
    INNOUT_VISUAL_ONLY_COLLISION_ROLE,
    mapped_vertices_from_glb,
    point_in_polygon,
    point_segment_distance,
)


REGIONS = ("water", "green", "crosswalk", "curb", "sidewalk", "road")
VEHICLE_FOOTPRINT_CLEARANCE = 2.5
PROP_FOOTPRINT_CLEARANCE = 0.75


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_path(value: Any) -> str | None:
    if isinstance(value, Sdf.AssetPath):
        return value.path
    return None


def ground_texture_refs(stage: Usd.Stage, usd_path: Path) -> tuple[set[str], list[str]]:
    refs: set[str] = set()
    missing: list[str] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        for shader_input in UsdShade.Shader(prim).GetInputs():
            ref = asset_path(shader_input.Get())
            if not ref or "Material_Ground" not in Path(ref).name:
                continue
            refs.add(ref)
            resolved = Path(ref) if Path(ref).is_absolute() else usd_path.parent / ref
            if not resolved.exists():
                missing.append(ref)
    return refs, sorted(set(missing))


def city_texture_refs(
    stage: Usd.Stage,
    usd_path: Path,
) -> tuple[set[str], list[str], list[str]]:
    refs: set[str] = set()
    non_dds: list[str] = []
    missing: list[str] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader" or not str(prim.GetPath()).startswith("/World/city/"):
            continue
        for shader_input in UsdShade.Shader(prim).GetInputs():
            ref = asset_path(shader_input.Get())
            if not ref:
                continue
            refs.add(ref)
            if not ref.lower().endswith(".dds"):
                non_dds.append(ref)
            resolved = Path(ref) if Path(ref).is_absolute() else usd_path.parent / ref
            if not resolved.exists():
                missing.append(ref)
    return refs, sorted(set(non_dds)), sorted(set(missing))


def mesh_error(prim: Usd.Prim, *, require_triangles: bool = True) -> str | None:
    mesh = UsdGeom.Mesh(prim)
    points = list(mesh.GetPointsAttr().Get() or [])
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if not points or not counts or not indices:
        return "empty mesh"
    if any(count < 3 for count in counts):
        return "degenerate face"
    if require_triangles and any(count != 3 for count in counts):
        return "non-triangle face"
    if len(indices) != sum(counts):
        return "face/index count mismatch"
    if min(indices) < 0 or max(indices) >= len(points):
        return "out-of-range face index"
    if not np.isfinite(np.asarray(points, dtype=float)).all():
        return "non-finite point"
    return None


def audit_category_mesh(
    stage: Usd.Stage,
    category: str,
    label: str,
    required: bool,
    errors: list[str],
) -> int | None:
    prim = stage.GetPrimAtPath(f"/World/TerrainRegions/{category}")
    if not prim or prim.GetTypeName() != "Mesh":
        if required:
            errors.append(f"{label}: missing TerrainRegions/{category} mesh")
        return None
    if UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.inherited:
        errors.append(f"{label}: TerrainRegions/{category} is hidden")
    problem = mesh_error(prim)
    if problem:
        errors.append(f"{label}: TerrainRegions/{category}: {problem}")
    return len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or [])


def audit_region_meshes(
    stage: Usd.Stage,
    label: str,
    expected_regions: set[str],
    errors: list[str],
) -> dict[str, int]:
    root = stage.GetPrimAtPath("/World/TerrainRegions")
    if not root:
        errors.append(f"{label}: missing /World/TerrainRegions")
        return {}
    root_visibility = UsdGeom.Imageable(root).ComputeVisibility()
    if root_visibility != UsdGeom.Tokens.inherited:
        errors.append(f"{label}: TerrainRegions is not visible")
    face_counts: dict[str, int] = {}
    for category in REGIONS:
        count = audit_category_mesh(stage, category, label, category in expected_regions, errors)
        if count is not None:
            face_counts[category] = count
    source = stage.GetPrimAtPath("/World/TerrainRegions/Terrain")
    if not source or source.GetTypeName() != "Mesh":
        errors.append(f"{label}: missing hidden TerrainRegions/Terrain mesh")
    else:
        if UsdGeom.Imageable(source).ComputeVisibility() != UsdGeom.Tokens.invisible:
            errors.append(f"{label}: source Terrain is not hidden")
        problem = mesh_error(source)
        if problem:
            errors.append(f"{label}: TerrainRegions/Terrain: {problem}")
        source_face_count = len(UsdGeom.Mesh(source).GetFaceVertexCountsAttr().Get() or [])
        expected_face_count = sum(face_counts.values())
        if source_face_count != expected_face_count:
            errors.append(
                f"{label}: combined Terrain face count {source_face_count} != regions {expected_face_count}"
            )
        face_counts["Terrain"] = source_face_count
    return face_counts


def audit_region_summary(city_dir: Path, label: str, errors: list[str]) -> dict[str, Any]:
    summary = read_json(city_dir / f"{city_dir.name}_terrain_regions_summary.json")
    if not summary:
        errors.append(f"{label}: missing terrain-region summary")
        return {}
    if summary.get("region_geometry") != "semantic_boundary_clipped_source_terrain":
        errors.append(f"{label}: unexpected region geometry mode")
    if summary.get("unclassified_region") != "sidewalk":
        errors.append(f"{label}: unclassified terrain is not assigned to sidewalk")
    if float(summary.get("coverage_relative_error", 1.0)) > 1e-6:
        errors.append(f"{label}: region coverage error exceeds 1e-6")
    if int(summary.get("source_surface_triangle_count", 0)) <= 0:
        errors.append(f"{label}: terrain summary has no source surface triangles")
    if int(summary.get("generated_curb_walls", 0)) <= 0:
        errors.append(f"{label}: terrain summary has no generated curb walls")
    if summary.get("region_visibility") != "visible" or summary.get("source_terrain_visibility") != "invisible":
        errors.append(f"{label}: terrain summary visibility mode is invalid")
    return summary


def audit_preview(stage: Usd.Stage, label: str, errors: list[str]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    expected_focal_length = 30.0 if label == "outdoor" else 24.0
    expected_source = "low_obstruction_central_" if label == "outdoor" else "innout_entrance_street_view"
    sky = stage.GetPrimAtPath("/World/sky")
    if not sky or not stage.GetPrimAtPath("/World/sky/DomeLight"):
        errors.append(f"{label}: embedded CloudySky lighting is missing")
    else:
        details["lighting"] = sky.GetCustomDataByKey("navverse_preview_lighting")
        if details["lighting"] != "CloudySky":
            errors.append(f"{label}: unexpected preview lighting {details['lighting']!r}")
    camera_details = {}
    for camera_path in ("/World/DefaultCamera", "/OmniverseKit_Persp"):
        prim = stage.GetPrimAtPath(camera_path)
        if not prim or not prim.IsA(UsdGeom.Camera):
            errors.append(f"{label}: missing preview camera {camera_path}")
            continue
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float)
        if not np.isfinite(matrix).all():
            errors.append(f"{label}: non-finite preview camera transform {camera_path}")
        source = prim.GetCustomDataByKey("navverse_preview_camera_source")
        if not source or (label == "outdoor" and not str(source).startswith(expected_source)):
            errors.append(f"{label}: invalid preview camera source on {camera_path}: {source!r}")
        if label == "innout" and source != expected_source:
            errors.append(f"{label}: invalid preview camera source on {camera_path}: {source!r}")
        focal_length = float(UsdGeom.Camera(prim).GetFocalLengthAttr().Get())
        if abs(focal_length - expected_focal_length) > 1e-5:
            errors.append(
                f"{label}: preview camera focal length {focal_length} != {expected_focal_length}"
            )
        camera_details[camera_path] = {"source": source, "focal_length": focal_length}
    details["cameras"] = camera_details
    return details


def audit_terrain(stage: Usd.Stage, city_dir: Path, label: str, errors: list[str]) -> dict[str, Any]:
    if stage.GetPrimAtPath("/World/city/Terrain/Terrain"):
        errors.append(f"{label}: stale /World/city/Terrain/Terrain")
    summary = audit_region_summary(city_dir, label, errors)
    expected_regions = {
        category
        for category, count in (summary.get("region_face_counts") or {}).items()
        if int(count) > 0
    }
    face_counts = audit_region_meshes(stage, label, expected_regions, errors)
    return {"region_face_counts": face_counts}


def audit_storefront_colliders(stage: Usd.Stage, label: str, errors: list[str]) -> dict[str, int]:
    root = stage.GetPrimAtPath("/World/store")
    meshes = [prim for prim in Usd.PrimRange(root) if prim.GetTypeName() == "Mesh"] if root else []
    colliders = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes)
    if not meshes:
        errors.append(f"{label}: no storefront meshes")
    if colliders != len(meshes):
        errors.append(f"{label}: storefront colliders {colliders}/{len(meshes)}")
    return {"meshes": len(meshes), "colliders": colliders}


def active_child_count(stage: Usd.Stage, path: str, *, excluded_names: set[str] | None = None) -> int:
    root = stage.GetPrimAtPath(path)
    excluded_names = excluded_names or set()
    if not root:
        return 0
    return sum(child.IsActive() and child.GetName() not in excluded_names for child in root.GetChildren())


def point_distance_xy(point: list[float], center: list[float]) -> float:
    return math.hypot(float(point[0]) - float(center[0]), float(point[1]) - float(center[1]))


def merge_eligible_road_object_paths(objects: list[dict[str, Any]]) -> set[str]:
    placed_traffic_lights: list[list[float]] = []
    paths: set[str] = set()
    for index, item in enumerate(objects):
        location = item.get("final_location") or item.get("location")
        is_traffic_light = item.get("category") in {"crossing", "traffic_signals"}
        if is_traffic_light and location:
            if any(point_distance_xy(location, other) < 6.0 for other in placed_traffic_lights):
                continue
            placed_traffic_lights.append(location)
        asset = str(item.get("asset_path") or "")
        if not asset:
            continue
        paths.add(f"/World/RoadObjects/{Path(asset.replace(chr(92), '/')).stem}_{index}")
    return paths


def audit_object_spacing(
    objects: list[dict[str, Any]],
    label: str,
    errors: list[str],
) -> dict[str, Any]:
    invalid_footprints = 0
    overlaps = 0
    minimum_margin = math.inf
    for index, item in enumerate(objects):
        radius = float(item.get("footprint_radius_m", 0.0))
        dims = item.get("footprint_dims_m") or []
        if radius <= 0.0 or len(dims) < 2 or min(float(dims[0]), float(dims[1])) <= 0.0:
            invalid_footprints += 1
            continue
        for other in objects[:index]:
            other_radius = float(other.get("footprint_radius_m", 0.0))
            if other_radius <= 0.0:
                continue
            clearance = (
                VEHICLE_FOOTPRINT_CLEARANCE
                if "vehicle" in {item.get("category"), other.get("category")}
                else PROP_FOOTPRINT_CLEARANCE
            )
            distance = point_distance_xy(item["location"], other["location"])
            margin = distance - radius - other_radius - clearance
            minimum_margin = min(minimum_margin, margin)
            if margin < -1e-4:
                overlaps += 1
    if invalid_footprints:
        errors.append(f"{label}: {invalid_footprints} objects have missing or invalid footprint metadata")
    if overlaps:
        errors.append(f"{label}: {overlaps} object footprint pairs violate placement clearance")
    return {
        "objects": len(objects),
        "invalid_footprints": invalid_footprints,
        "overlap_pairs": overlaps,
        "minimum_clearance_margin_m": None if not math.isfinite(minimum_margin) else minimum_margin,
    }


def object_intersects_exclusion(item: dict[str, Any], exclusion: dict[str, Any]) -> bool:
    location = np.asarray((item.get("location") or [])[:2], dtype=float)
    center = np.asarray(exclusion.get("center_xy") or [], dtype=float)
    tangent = np.asarray(exclusion.get("tangent_xy") or [], dtype=float)
    outward = np.asarray(exclusion.get("outward_normal_xy") or [], dtype=float)
    if location.shape != (2,) or center.shape != (2,) or tangent.shape != (2,) or outward.shape != (2,):
        return False
    tangent_length = float(np.linalg.norm(tangent))
    outward_length = float(np.linalg.norm(outward))
    if tangent_length <= 1e-6 or outward_length <= 1e-6:
        return False
    tangent /= tangent_length
    outward /= outward_length
    radius = float(item.get("footprint_radius_m", 0.0))
    delta = location - center
    lateral = abs(float(np.dot(delta, tangent)))
    normal = float(np.dot(delta, outward))
    half_width = (
        float(exclusion.get("width_m", 0.0)) * 0.5
        + float(exclusion.get("lateral_clearance_m", 0.0))
        + radius
    )
    inward = float(exclusion.get("inward_clearance_m", 0.0)) + radius
    outward_distance = float(exclusion.get("outward_clearance_m", 0.0)) + radius
    return lateral <= half_width and -inward <= normal <= outward_distance


def audit_local_innout_content(
    city_dir: Path,
    plan: dict[str, Any],
    actual_storefronts: int,
    actual_objects: int,
    actual_road_objects: int,
    details: dict[str, Any],
    errors: list[str],
) -> bool:
    radius = float(plan.get("content_radius_m", 0.0))
    center = plan.get("door_center_xy")
    store_plan = read_json(city_dir / "innout_storefront_plan.json")
    object_plan = read_json(city_dir / "innout_objaverse_objects.json")
    road_plan = read_json(city_dir / "innout_road_objects_info.json")
    if radius <= 0.0 or not isinstance(center, list) or len(center) < 2:
        return False
    if not store_plan or not object_plan:
        errors.append("innout: local content plan is missing")
        return True

    excluded_indices = set(plan.get("excluded_storefront_plan_indices") or [])
    storefronts = [
        row
        for row in store_plan.get("placements", [])
        if row.get("status") == "planned"
        and row.get("facade_status") == "facade_aligned"
        and row.get("index") not in excluded_indices
    ]
    storefront_distances = [
        point_distance_xy(row["final_location"], center)
        for row in storefronts
        if len(row.get("final_location") or []) >= 2
    ]
    estimated_capacity = int(plan.get("local_storefront_capacity_estimate", 0))
    minimum_storefronts = min(
        int(store_plan.get("min_storefronts", 0)),
        estimated_capacity,
        7,
    )
    if len(storefronts) < minimum_storefronts:
        errors.append(
            f"innout: only {len(storefronts)} facade-aligned storefronts, expected at least "
            f"{minimum_storefronts}"
        )
    if actual_storefronts != len(storefronts):
        errors.append(
            f"innout: storefront mesh count {actual_storefronts} != local plan {len(storefronts)}"
        )
    if storefront_distances and max(storefront_distances) > radius + 1e-4:
        errors.append("innout: storefront plan contains placements outside the content radius")

    objects = object_plan.get("objects", [])
    placement_exclusions = object_plan.get("placement_exclusions") or []
    if not placement_exclusions:
        errors.append("innout: local object plan has no doorway placement exclusion")
    doorway_violations = [
        row
        for row in objects
        if any(object_intersects_exclusion(row, exclusion) for exclusion in placement_exclusions)
    ]
    if doorway_violations:
        errors.append(
            f"innout: {len(doorway_violations)} objects intersect the doorway placement exclusion"
        )
    target_counts = object_plan.get("target_counts") or {"prop": 120, "vehicle": 30}
    removed_object_paths = {
        str(path)
        for path in plan.get("removed_exterior_obstacle_paths") or []
        if str(path).startswith("/World/Objaverse/")
    }
    expected_active_objects = len(objects) - len(removed_object_paths)
    object_categories = Counter(row.get("category") for row in objects)
    object_distances = [
        point_distance_xy(row["location"], center)
        for row in objects
        if len(row.get("location") or []) >= 2
    ]
    for category in ("prop", "vehicle"):
        if object_categories.get(category, 0) > int(target_counts.get(category, 0)):
            errors.append(
                f"innout: local {category} count exceeds target: "
                f"{object_categories.get(category, 0)} > {target_counts.get(category, 0)}"
            )
    invalid_props = [
        row
        for row in objects
        if row.get("category") == "prop"
        and (
            row.get("placement_semantic") not in {"footway=sidewalk", "road_sidewalk_tag"}
            or not row.get("placement_way_id")
        )
    ]
    if invalid_props:
        errors.append(
            f"innout: local object plan has {len(invalid_props)} props without explicit OSM sidewalk provenance"
        )
    if actual_objects != expected_active_objects:
        errors.append(
            "innout: active object count "
            f"{actual_objects} != local plan {len(objects)} - removed doorway obstacles "
            f"{len(removed_object_paths)}"
        )
    if object_distances and max(object_distances) > radius + 1e-4:
        errors.append("innout: object plan contains placements outside the content radius")

    road_objects = road_plan.get("objects", [])
    removed_road_paths = {
        str(path)
        for path in plan.get("removed_exterior_obstacle_paths") or []
        if str(path).startswith("/World/RoadObjects/")
    }
    road_distances = [
        point_distance_xy(row["location"], center)
        for row in road_objects
        if len(row.get("location") or []) >= 2
    ]
    merge_eligible_road_paths = merge_eligible_road_object_paths(road_objects)
    expected_active_road_objects = len(merge_eligible_road_paths - removed_road_paths)
    if actual_road_objects != expected_active_road_objects:
        errors.append(
            "innout: active road-object count "
            f"{actual_road_objects} != merge-eligible plan {len(merge_eligible_road_paths)} - removed doorway obstacles "
            f"{len(removed_road_paths)}"
        )
    if road_distances and max(road_distances) > radius + 1e-4:
        errors.append("innout: road-object plan contains placements outside the content radius")

    details["innout_local_content"] = {
        "center_xy": center,
        "radius_m": radius,
        "storefronts": len(storefronts),
        "storefront_capacity_estimate": estimated_capacity,
        "minimum_storefronts": minimum_storefronts,
        "storefront_max_distance_m": max(storefront_distances, default=0.0),
        "object_categories": dict(object_categories),
        "object_targets": target_counts,
        "object_spacing": audit_object_spacing(objects, "innout", errors),
        "doorway_placement_exclusions": len(placement_exclusions),
        "doorway_object_violations": len(doorway_violations),
        "explicit_sidewalk_props": object_categories.get("prop", 0) - len(invalid_props),
        "removed_doorway_objects": len(removed_object_paths),
        "expected_active_objects": expected_active_objects,
        "object_max_distance_m": max(object_distances, default=0.0),
        "road_objects": len(road_objects),
        "merge_eligible_road_objects": len(merge_eligible_road_paths),
        "removed_doorway_road_objects": len(removed_road_paths),
        "expected_active_road_objects": expected_active_road_objects,
        "road_object_max_distance_m": max(road_distances, default=0.0),
    }
    return True


def audit_city_transform(stage: Usd.Stage, label: str, errors: list[str]) -> dict[str, Any]:
    prim = stage.GetPrimAtPath("/World/city")
    if not prim:
        errors.append(f"{label}: missing /World/city")
        return {}
    scale = None
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale = np.asarray(op.Get(), dtype=float)
            break
    if scale is None or not np.allclose(scale, np.ones(3), atol=1e-6):
        errors.append(f"{label}: /World/city scale is {None if scale is None else scale.tolist()}, expected 1.0")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
    )
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    size = np.asarray(bounds.GetMax(), dtype=float) - np.asarray(bounds.GetMin(), dtype=float)
    if not np.isfinite(size).all() or max(size[:2], default=0.0) < 50.0 or size[2] < 5.0:
        errors.append(f"{label}: /World/city world bounds are implausibly small: {size.tolist()}")
    return {
        "scale": None if scale is None else scale.tolist(),
        "world_size_m": size.tolist(),
    }


def audit_innout_plan(plan: dict[str, Any], details: dict[str, Any], errors: list[str]) -> None:
    height_delta = abs(float(plan.get("door_bottom_z", math.inf)) - float(plan.get("terrain_z", -math.inf)))
    details["door_terrain_height_delta_m"] = height_delta
    if not math.isfinite(height_delta) or height_delta > 1e-4:
        errors.append(f"innout: door/terrain height delta {height_delta}")
    if plan.get("interior_fit_status") != "inside_building":
        errors.append(f"innout: interior fit status {plan.get('interior_fit_status')!r}")
    if not plan.get("interior_door_anchor_inside_building"):
        errors.append("innout: interior door anchor is outside selected building")
    if plan.get("door_cut_mode") != "boolean_tunnel":
        errors.append(f"innout: unexpected door cut mode {plan.get('door_cut_mode')!r}")
    room_frontage_width = float(plan.get("interior_frontage_width", math.inf))
    opening_width = float(plan.get("building_opening_width", -math.inf))
    room_top_z = float(plan.get("interior_world_top_z", math.inf))
    opening_top_z = float(plan.get("building_opening_top_z", -math.inf))
    details["room_sized_opening"] = {
        "source": plan.get("building_opening_source"),
        "interior_frontage_width_m": room_frontage_width,
        "opening_width_m": opening_width,
        "interior_world_top_z_m": room_top_z,
        "opening_top_z_m": opening_top_z,
    }
    if plan.get("building_opening_source") != "interior_room_frontage":
        errors.append("innout: building opening is not derived from the interior room frontage")
    if not math.isfinite(room_frontage_width) or opening_width < room_frontage_width:
        errors.append(
            "innout: building opening is narrower than the interior frontage: "
            f"opening={opening_width} frontage={room_frontage_width}"
        )
    if not math.isfinite(room_top_z) or opening_top_z < room_top_z:
        errors.append(
            "innout: building opening is lower than the interior room top: "
            f"opening_top={opening_top_z} room_top={room_top_z}"
        )


def audit_exterior_doorway(plan: dict[str, Any], details: dict[str, Any], errors: list[str]) -> None:
    footprint = np.asarray(plan.get("building_footprint_xy") or [], dtype=float)
    center = np.asarray(plan.get("door_center_xy") or [], dtype=float)
    outward = np.asarray(plan.get("outward_normal_xy") or [], dtype=float)
    if footprint.ndim != 2 or footprint.shape[0] < 3 or footprint.shape[1] != 2:
        errors.append("innout: invalid selected-building footprint")
        return
    if center.shape != (2,) or outward.shape != (2,) or not np.isfinite(outward).all():
        errors.append("innout: invalid exterior doorway position or normal")
        return
    outward_length = float(np.linalg.norm(outward))
    if outward_length <= 1e-9:
        errors.append("innout: zero-length exterior doorway normal")
        return
    outward /= outward_length

    boundary_distance = min(
        point_segment_distance(center, footprint[index], footprint[(index + 1) % len(footprint)])
        for index in range(len(footprint))
    )
    probe_distance = 0.5
    inward_probe = center - outward * probe_distance
    outward_probe = center + outward * probe_distance
    inward_inside = point_in_polygon(inward_probe, footprint, tolerance=0.02)
    outward_inside = point_in_polygon(outward_probe, footprint, tolerance=0.02)
    details["exterior_doorway"] = {
        "building_boundary_distance_m": boundary_distance,
        "probe_distance_m": probe_distance,
        "inward_probe_inside_building": inward_inside,
        "outward_probe_inside_building": outward_inside,
    }
    if boundary_distance > 0.15:
        errors.append(f"innout: doorway is {boundary_distance:.3f} m from the selected building exterior")
    if not inward_inside:
        errors.append("innout: doorway inward probe is outside the selected building")
    if outward_inside:
        errors.append("innout: doorway outward probe remains inside the selected building")

    valid_samples = [row for row in plan.get("exterior_public_surface_samples") or [] if row.get("valid")]
    if not valid_samples:
        errors.append("innout: doorway has no valid exterior public-surface samples")
        return
    projected_samples = []
    for row in valid_samples:
        public_point = np.asarray([row["nearest_public_x"], row["nearest_public_y"]], dtype=float)
        public_delta = public_point - center
        projected_samples.append((float(public_delta @ outward), float(np.linalg.norm(public_delta))))
    outward_projection, public_distance = max(projected_samples)
    details["exterior_doorway"].update({
        "best_public_surface_distance_m": public_distance,
        "best_public_surface_outward_projection_m": outward_projection,
    })
    if outward_projection <= probe_distance:
        errors.append("innout: nearest public surface is not on the exterior side of the doorway")

    cut_usd = Path(str(plan.get("cut_usd") or ""))
    if not cut_usd.exists():
        errors.append(f"innout: cut building USD is missing: {cut_usd}")
        return
    cut_stage = Usd.Stage.Open(str(cut_usd))
    cut_xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    shell_points: list[np.ndarray] = []
    for prim in cut_stage.Traverse():
        if prim.GetTypeName() != "Mesh" or not prim.IsActive():
            continue
        transform = cut_xforms.GetLocalToWorldTransform(prim)
        for point in UsdGeom.Mesh(prim).GetPointsAttr().Get() or []:
            world = transform.Transform(point)
            shell_points.append(np.asarray([world[0], world[1], world[2]], dtype=float))
    if not shell_points:
        errors.append("innout: cut building USD has no mesh points")
        return
    shell = np.asarray(shell_points, dtype=float)
    local_xy = shell[:, :2] - center
    lateral = np.abs(local_xy @ np.asarray(plan.get("tangent_xy"), dtype=float))
    vertical = (shell[:, 2] >= float(plan.get("building_opening_bottom_z", -math.inf)) - 0.1) & (
        shell[:, 2] <= float(plan.get("building_opening_top_z", math.inf)) + 0.1
    )
    near_opening = shell[(lateral <= float(plan.get("building_opening_width", 0.0)) * 0.5 + 0.5) & vertical]
    if not len(near_opening):
        errors.append("innout: cut building has no geometry around the doorway opening")
        return
    shell_facade_offset = float(np.min(np.abs((near_opening[:, :2] - center) @ outward)))
    details["exterior_doorway"]["cut_shell_facade_offset_m"] = shell_facade_offset
    facade_tolerance = 0.3 if plan.get("door_cut_mode") == "boolean_tunnel" else 0.05
    if shell_facade_offset > facade_tolerance:
        errors.append(
            "innout: cut-building facade does not meet the exterior doorway: "
            f"offset={shell_facade_offset:.3f} m"
        )


def audit_cut_building_alignment(
    stage: Usd.Stage,
    plan: dict[str, Any],
    details: dict[str, Any],
    errors: list[str],
) -> None:
    source_path = str(plan.get("target_prim_path") or "")
    replacement_path = source_path + "_cut"
    source = stage.GetPrimAtPath(source_path)
    replacement = stage.GetPrimAtPath(replacement_path)
    if source and source.IsActive():
        errors.append(f"innout: original cut target remains active: {source_path}")
    if not replacement or not replacement.IsActive():
        errors.append(f"innout: active cut-building replacement is missing: {replacement_path}")
        return

    interior_subset_path = replacement_path + "/mesh/NavVerseCutInteriorFaces"
    interior_subset = stage.GetPrimAtPath(interior_subset_path)
    interior_face_count = 0
    interior_material_path = ""
    if interior_subset and interior_subset.GetTypeName() == "GeomSubset":
        interior_face_count = len(UsdGeom.Subset(interior_subset).GetIndicesAttr().Get() or [])
        interior_material_path = (
            UsdShade.MaterialBindingAPI(interior_subset)
            .GetDirectBinding()
            .GetMaterialPath()
            .pathString
        )
    details["cut_building_interior_material"] = {
        "subset_path": interior_subset_path,
        "face_count": interior_face_count,
        "material_path": interior_material_path,
    }
    if interior_face_count <= 0:
        errors.append("innout: cut-building interior face subset is missing or empty")
    if "NavVerseCutInterior" not in interior_material_path:
        errors.append(
            f"innout: cut-building interior faces do not use the gray-white material: {interior_material_path}"
        )

    door_width = float(plan.get("door_width", math.inf))
    building_opening_width = float(plan.get("building_opening_width", -math.inf))
    details["building_opening_width_m"] = building_opening_width
    if not math.isfinite(building_opening_width) or building_opening_width < door_width + 0.5:
        errors.append(
            "innout: building opening does not provide enough lateral clearance: "
            f"opening={building_opening_width} door={door_width}"
        )

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
    )
    bounds = cache.ComputeWorldBound(replacement).ComputeAlignedRange()
    actual_min = np.asarray(bounds.GetMin(), dtype=float)
    actual_max = np.asarray(bounds.GetMax(), dtype=float)
    building_glb = Path(str(plan.get("building_glb") or ""))
    if not building_glb.exists():
        errors.append(f"innout: selected building GLB is missing: {building_glb}")
        return
    expected_vertices = mapped_vertices_from_glb(building_glb)
    expected_min = expected_vertices.min(axis=0)
    expected_max = expected_vertices.max(axis=0)
    xy_error = float(
        max(
            np.max(np.abs(actual_min[:2] - expected_min[:2])),
            np.max(np.abs(actual_max[:2] - expected_max[:2])),
        )
    )
    building_min_z = float(plan.get("building_min_z", math.nan))
    z_min_error = abs(float(actual_min[2]) - building_min_z)
    details["cut_building_alignment"] = {
        "replacement_path": replacement_path,
        "bbox_min": actual_min.tolist(),
        "bbox_max": actual_max.tolist(),
        "source_bbox_min": expected_min.tolist(),
        "source_bbox_max": expected_max.tolist(),
        "xy_extent_error_m": xy_error,
        "z_min_error_m": z_min_error,
    }
    if not np.isfinite(actual_min).all() or not np.isfinite(actual_max).all():
        errors.append("innout: cut-building replacement has non-finite world bounds")
        return
    if xy_error > 0.5:
        errors.append(f"innout: cut-building XY extent is misaligned by {xy_error:.3f} m")
    if not math.isfinite(z_min_error) or z_min_error > 1.0:
        errors.append(f"innout: cut-building vertical axis/minimum is misaligned by {z_min_error:.3f} m")
    if actual_max[2] < float(plan.get("building_opening_top_z", math.inf)) + 0.15 - 1e-4:
        errors.append("innout: cut building does not extend above the doorway")

    room = stage.GetPrimAtPath("/World/start_result_navigation")
    if not room:
        return
    room_bounds = cache.ComputeWorldBound(room).ComputeAlignedRange()
    room_min = np.asarray(room_bounds.GetMin(), dtype=float)
    room_max = np.asarray(room_bounds.GetMax(), dtype=float)
    details["cut_building_alignment"].update({
        "room_bbox_min": room_min.tolist(),
        "room_bbox_max": room_max.tolist(),
    })
    if float(plan.get("interior_world_top_z", math.inf)) > actual_max[2] + 0.05:
        errors.append("innout: transformed room top extends above the selected building")


def audit_interior_floor_alignment(
    stage: Usd.Stage,
    plan: dict[str, Any],
    details: dict[str, Any],
    errors: list[str],
) -> None:
    ground_root = stage.GetPrimAtPath("/World/start_result_navigation/Meshes/Base/ground")
    if not ground_root:
        errors.append("innout: missing interior ground root")
        return

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    floor_z_values: list[float] = []
    mesh_count = 0
    for prim in Usd.PrimRange(ground_root):
        if prim.GetTypeName() != "Mesh" or not prim.IsActive():
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not points:
            continue
        transform = xform_cache.GetLocalToWorldTransform(prim)
        floor_z_values.extend(float(transform.Transform(point)[2]) for point in points)
        mesh_count += 1

    details["interior_ground_mesh_count"] = mesh_count
    if not floor_z_values:
        errors.append("innout: interior ground meshes have no points")
        return

    door_bottom_z = float(plan.get("door_bottom_z", math.inf))
    nearest_floor_z = min(floor_z_values, key=lambda value: abs(value - door_bottom_z))
    height_delta = abs(nearest_floor_z - door_bottom_z)
    details["interior_floor_z_range_m"] = [min(floor_z_values), max(floor_z_values)]
    details["interior_floor_nearest_door_z_m"] = nearest_floor_z
    details["interior_floor_door_height_delta_m"] = height_delta
    if not math.isfinite(height_delta) or height_delta > 0.05:
        errors.append(
            "innout: interior floor is not aligned with the doorway: "
            f"floor_z={nearest_floor_z} door_bottom_z={door_bottom_z} delta={height_delta}"
        )


def audit_terrain_hole(
    stage: Usd.Stage,
    plan: dict[str, Any],
    details: dict[str, Any],
    errors: list[str],
) -> None:
    holes = plan.get("terrain_hole_summary") or []
    details["terrain_hole_meshes"] = len(holes)
    expected_paths = set()
    for name in (*REGIONS, "Terrain"):
        path = f"/World/TerrainRegions/{name}"
        prim = stage.GetPrimAtPath(path)
        if prim and prim.GetTypeName() == "Mesh":
            expected_paths.add(path)
    actual_paths = {str(row.get("mesh_path")) for row in holes}
    if actual_paths != expected_paths or len(holes) != len(actual_paths):
        errors.append(
            "innout: terrain-hole mesh records do not match stage meshes: "
            f"actual={sorted(actual_paths)} expected={sorted(expected_paths)}"
        )
    for row in holes:
        balance = float(row.get("kept_xy_area", 0.0)) + float(row.get("removed_xy_area", 0.0))
        source = float(row.get("source_xy_area", 0.0))
        if abs(balance - source) > max(1e-5, source * 1e-7):
            errors.append(f"innout: terrain-hole area mismatch on {row.get('mesh_path')}")


def audit_threshold(
    stage: Usd.Stage,
    plan: dict[str, Any],
    details: dict[str, Any],
    errors: list[str],
) -> None:
    ramp = stage.GetPrimAtPath("/World/InnOutThreshold/ramp")
    collision_proxy = stage.GetPrimAtPath("/World/InnOutThreshold/collision_proxy")
    if not ramp or ramp.GetTypeName() != "Mesh":
        errors.append("innout: missing threshold ramp")
    else:
        problem = mesh_error(ramp)
        if problem:
            errors.append(f"innout: threshold ramp: {problem}")
        if collision_proxy and collision_proxy.GetTypeName() == "Mesh":
            if (
                ramp.GetCustomDataByKey(INNOUT_COLLISION_ROLE_KEY)
                != INNOUT_VISUAL_ONLY_COLLISION_ROLE
            ):
                errors.append("innout: visual threshold ramp is not marked visual-only")
            if ramp.HasAPI(UsdPhysics.CollisionAPI) or ramp.HasAPI(UsdPhysics.MeshCollisionAPI):
                errors.append("innout: visual threshold ramp still has collision APIs")
        elif not ramp.HasAPI(UsdPhysics.CollisionAPI) or not ramp.HasAPI(UsdPhysics.MeshCollisionAPI):
            errors.append("innout: threshold ramp collision APIs missing")
        material_path = UsdShade.MaterialBindingAPI(ramp).GetDirectBinding().GetMaterialPath().pathString
        details["threshold_material"] = material_path
        if "Material_Ground" not in material_path:
            errors.append(f"innout: threshold is not using a ground material: {material_path}")
        if not ramp.GetAttribute("primvars:UVMap").HasAuthoredValueOpinion():
            errors.append("innout: threshold UVMap missing")
    if collision_proxy:
        if collision_proxy.GetTypeName() != "Mesh":
            errors.append("innout: threshold collision proxy is not a mesh")
        else:
            problem = mesh_error(collision_proxy)
            if problem:
                errors.append(f"innout: threshold collision proxy: {problem}")
            if not collision_proxy.HasAPI(UsdPhysics.CollisionAPI) or not collision_proxy.HasAPI(
                UsdPhysics.MeshCollisionAPI
            ):
                errors.append("innout: threshold collision proxy APIs missing")
            if UsdGeom.Imageable(collision_proxy).ComputeVisibility() != UsdGeom.Tokens.invisible:
                errors.append("innout: threshold collision proxy is visible")
    summary = plan.get("threshold_summary") or {}
    details["threshold_summary"] = summary
    if not summary.get("solid"):
        errors.append("innout: threshold ramp is not a closed solid")
    if float(summary.get("top_normal_z", 0.0)) <= 0.0:
        errors.append("innout: threshold top faces do not point upward")
    if float(summary.get("bottom_normal_z", 0.0)) >= 0.0:
        errors.append("innout: threshold bottom faces do not point downward")
    proxy_summary = summary.get("collision_proxy") or {}
    if collision_proxy:
        if proxy_summary.get("prim_path") != str(collision_proxy.GetPath()):
            errors.append("innout: threshold collision proxy summary path mismatch")
        if not proxy_summary.get("top_only"):
            errors.append("innout: threshold collision proxy is not top-only")
        if float(proxy_summary.get("top_normal_z", 0.0)) <= 0.0:
            errors.append("innout: threshold collision proxy faces do not point upward")
    if float(summary.get("width_m", 0.0)) + 1e-4 < float(plan.get("door_width", math.inf)):
        errors.append("innout: threshold ramp is narrower than the doorway")
    opening_bottom = float(plan.get("building_opening_bottom_z", math.inf))
    ramp_bottom = float(summary.get("facade_base_z", summary.get("bottom_z", -math.inf)))
    facade_bottom = min(
        float(summary.get("facade_left_z", -math.inf)),
        float(summary.get("facade_right_z", -math.inf)),
    )
    details["building_opening_ramp_clearance_m"] = facade_bottom - opening_bottom
    if not math.isfinite(opening_bottom) or opening_bottom > facade_bottom - 0.005:
        errors.append("innout: building opening bottom does not clear the threshold profile")
    if not math.isfinite(ramp_bottom) or opening_bottom > ramp_bottom + 0.005:
        errors.append("innout: building opening bottom is above the threshold facade base")


def audit_door_connector(
    stage: Usd.Stage,
    plan: dict[str, Any],
    details: dict[str, Any],
    errors: list[str],
) -> None:
    root = stage.GetPrimAtPath("/World/InnOutDoorConnector")
    expected_names = {"floor", "ceiling", "left_wall", "right_wall"}
    if not root:
        errors.append("innout: missing door connector")
        return
    meshes = {child.GetName(): child for child in root.GetChildren() if child.GetTypeName() == "Mesh"}
    if set(meshes) != expected_names:
        errors.append(f"innout: door connector surfaces {sorted(meshes)} != {sorted(expected_names)}")
    for name, prim in meshes.items():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            errors.append(f"innout: door connector collider missing on {name}")
        problem = mesh_error(prim, require_triangles=False)
        if problem:
            errors.append(f"innout: door connector {name}: {problem}")

    summary = plan.get("door_connector_summary") or {}
    details["door_connector"] = summary
    depth = float(summary.get("depth_m", 0.0))
    width = float(summary.get("width_m", 0.0))
    if int(summary.get("surface_count", 0)) != 4 or depth < 0.07:
        errors.append(f"innout: invalid door connector summary: {summary}")
    if float(summary.get("floor_normal_z", 0.0)) <= 0.0:
        errors.append("innout: door connector floor does not point upward")
    if float(summary.get("ceiling_normal_z", 0.0)) >= 0.0:
        errors.append("innout: door connector ceiling does not point downward")
    if abs(width - float(plan.get("door_width", math.inf))) > 1e-4:
        errors.append(f"innout: connector width {width} != door width {plan.get('door_width')}")
    outer_overlap = float(summary.get("outer_overlap_m", 0.0))
    inner_overlap = float(summary.get("inner_overlap_m", 0.0))
    expected_depth = (
        float(plan.get("interior_fit_inward_nudge_m", 0.0))
        + float(summary.get("interior_door_depth_m", 0.0))
        + outer_overlap
        + inner_overlap
    )
    if outer_overlap < 0.04 or inner_overlap < 0.04 or abs(depth - expected_depth) > 1e-4:
        errors.append(f"innout: door connector does not overlap both interfaces: {summary}")

    room = stage.GetPrimAtPath("/World/start_result_navigation")
    if room and meshes:
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        )
        connector_box = cache.ComputeWorldBound(root).ComputeAlignedBox()
        room_box = cache.ComputeWorldBound(room).ComputeAlignedBox()
        connector_min = np.asarray(connector_box.GetMin(), dtype=float)
        connector_max = np.asarray(connector_box.GetMax(), dtype=float)
        room_min = np.asarray(room_box.GetMin(), dtype=float)
        room_max = np.asarray(room_box.GetMax(), dtype=float)
        if np.any(np.minimum(connector_max, room_max) < np.maximum(connector_min, room_min) - 1e-4):
            errors.append("innout: door connector does not overlap the indoor scene bounds")


def audit_innout(stage: Usd.Stage, city_dir: Path, errors: list[str]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    plan = read_json(city_dir / "innout_final_plan.json")
    if not plan:
        errors.append("innout: missing innout_final_plan.json")
        return details
    audit_innout_plan(plan, details, errors)
    audit_exterior_doorway(plan, details, errors)
    audit_cut_building_alignment(stage, plan, details, errors)
    audit_interior_floor_alignment(stage, plan, details, errors)
    audit_terrain_hole(stage, plan, details, errors)
    audit_threshold(stage, plan, details, errors)
    audit_door_connector(stage, plan, details, errors)
    excluded_buildings = set(plan.get("excluded_storefront_buildings") or [])
    if len(excluded_buildings) > 1:
        errors.append(f"innout: storefronts from {len(excluded_buildings)} buildings were excluded")
    if not stage.GetPrimAtPath("/World/start_result_navigation"):
        errors.append("innout: missing interior root")
    if not stage.GetPrimAtPath("/World/Objaverse/NavVerseExitSigns/green_exit_sign_0000"):
        errors.append("innout: missing exit sign")
    return details


def audit_scene(city_dir: Path, label: str, errors: list[str]) -> dict[str, Any]:
    usd_path = city_dir / f"{city_dir.name}_{label}.usd"
    details: dict[str, Any] = {"path": str(usd_path)}
    if not usd_path.exists() or usd_path.stat().st_size == 0:
        errors.append(f"{label}: missing or empty USD")
        return details
    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        errors.append(f"{label}: root layer cannot be opened")
        return details
    stage = Usd.Stage.Open(layer)
    if stage is None:
        errors.append(f"{label}: stage cannot be opened")
        return details
    details["bytes"] = usd_path.stat().st_size
    details["city"] = audit_city_transform(stage, label, errors)
    details["preview"] = audit_preview(stage, label, errors)
    details["terrain"] = audit_terrain(stage, city_dir, label, errors)
    details["storefront_colliders"] = audit_storefront_colliders(stage, label, errors)
    details["active_content"] = {
        "storefronts": active_child_count(stage, "/World/store"),
        "objects": active_child_count(
            stage,
            "/World/Objaverse",
            excluded_names={"NavVerseExitSigns"},
        ),
        "road_objects": active_child_count(stage, "/World/RoadObjects"),
    }
    refs, missing = ground_texture_refs(stage, usd_path)
    details["ground_dds_refs"] = len(refs)
    if len(refs) != 64 or any(not ref.lower().endswith(".bc7.dds") for ref in refs):
        errors.append(f"{label}: expected 64 unique BC7 ground refs, found {len(refs)}")
    if missing:
        errors.append(f"{label}: {len(missing)} missing ground DDS files")
    city_refs, city_non_dds, city_missing = city_texture_refs(stage, usd_path)
    details["city_texture_refs"] = len(city_refs)
    details["city_texture_non_dds"] = len(city_non_dds)
    if len(city_refs) <= len(refs):
        errors.append(f"{label}: city texture set contains no non-ground materials")
    if city_non_dds:
        errors.append(f"{label}: {len(city_non_dds)} city texture refs are not DDS")
    if city_missing:
        errors.append(f"{label}: {len(city_missing)} city DDS files are missing")
    if label == "innout":
        details["innout"] = audit_innout(stage, city_dir, errors)
    return details


def audit_flat_parameters(city_dir: Path, errors: list[str]) -> None:
    flat = read_json(city_dir / "flat_ground_summary.json")
    terrain = flat.get("terrain", {}) if flat else {}
    if not flat:
        errors.append("missing flat_ground_summary.json")
    elif (
        float(flat.get("height_scale", -1.0)) != 0.3
        or float(terrain.get("road_depth", -1.0)) != 0.2
        or not terrain.get("road_boundary_cuts_enabled")
        or float(terrain.get("road_curb_width", -1.0)) != 0.2
    ):
        errors.append("flat-ground parameters do not match 0.3 scale / 0.2 m road+curb")


def audit_source_height_field(city_dir: Path, source_root: Path | None, errors: list[str]) -> None:
    if source_root is not None:
        flat_height = city_dir / "height_field.npz"
        source_height = source_root / city_dir.name / "height_field.npz"
        if not flat_height.exists() or not source_height.exists():
            errors.append("flat/source height_field.npz is missing")
        elif file_digest(flat_height) != file_digest(source_height):
            errors.append("flat/source height_field.npz differs")


def audit_objects_manifest(city_dir: Path, details: dict[str, Any], errors: list[str]) -> None:
    objects = read_json(city_dir / "objaverse_objects.json").get("objects", [])
    categories = Counter(row.get("category") for row in objects)
    details["object_categories"] = dict(categories)
    details["object_spacing"] = audit_object_spacing(objects, "outdoor", errors)
    sidewalk_constrained_empty = (
        categories == Counter({"vehicle": 200})
        and read_json(city_dir / "sidewalk_points.json").get("points", []) == []
    )
    if categories != Counter({"prop": 600, "vehicle": 200}) and not sidewalk_constrained_empty:
        errors.append(f"object plan count mismatch: {dict(categories)}")
    invalid_props = [
        row
        for row in objects
        if row.get("category") == "prop"
        and (
            row.get("placement_semantic") not in {"footway=sidewalk", "road_sidewalk_tag"}
            or not row.get("placement_way_id")
        )
    ]
    details["explicit_sidewalk_props"] = categories.get("prop", 0) - len(invalid_props)
    if invalid_props:
        errors.append(f"object plan has {len(invalid_props)} props without explicit OSM sidewalk provenance")
    manifest = read_json(city_dir / "manifest.json")
    if not manifest:
        errors.append("missing manifest.json")
    elif manifest.get("warnings"):
        details["manifest_warnings"] = manifest["warnings"]


def audit_city(city_dir: Path, source_root: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    audit_flat_parameters(city_dir, errors)
    audit_source_height_field(city_dir, source_root, errors)
    audit_objects_manifest(city_dir, details, errors)

    for label in ("outdoor", "innout"):
        details[label] = audit_scene(city_dir, label, errors)
    plan = read_json(city_dir / "innout_final_plan.json")
    outdoor_storefronts = details["outdoor"].get("storefront_colliders", {}).get("meshes", 0)
    innout_storefronts = details["innout"].get("storefront_colliders", {}).get("meshes", 0)
    innout_active_content = details["innout"].get("active_content", {})
    local_content = audit_local_innout_content(
        city_dir,
        plan,
        innout_storefronts,
        int(innout_active_content.get("objects", 0)),
        int(innout_active_content.get("road_objects", 0)),
        details,
        errors,
    )
    removed_storefronts = len(plan.get("removed_storefront_prim_paths") or [])
    if not local_content and innout_storefronts != outdoor_storefronts - removed_storefronts:
        errors.append(
            "innout: storefront mesh count does not match outdoor minus selected-building removals: "
            f"{innout_storefronts} != {outdoor_storefronts} - {removed_storefronts}"
        )
    return {"city": city_dir.name, "passed": not errors, "errors": errors, "details": details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cities", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = args.cities or sorted(
        path.name
        for path in args.root.iterdir()
        if path.is_dir() and (path / f"{path.name}_outdoor.usd").exists()
    )
    rows = [audit_city(args.root / name, args.source_root) for name in names]
    result = {
        "root": str(args.root),
        "city_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "failed_count": sum(not row["passed"] for row in rows),
        "cities": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("city_count", "passed_count", "failed_count")}, indent=2))
    raise SystemExit(1 if result["failed_count"] else 0)


if __name__ == "__main__":
    main()
