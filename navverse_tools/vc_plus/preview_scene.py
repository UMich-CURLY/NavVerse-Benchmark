#!/usr/bin/env python3
"""Author embedded preview lighting and street-level cameras for VC scenes."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom


CAMERA_PATHS = ("/World/DefaultCamera", "/OmniverseKit_Persp")


def add_preview_sky(stage: Usd.Stage, output_usd: Path, sky_usd: Path) -> str | None:
    sky_path = "/World/sky"
    if stage.GetPrimAtPath(sky_path):
        stage.RemovePrim(sky_path)
    if not sky_usd.exists():
        return None
    sky = stage.DefinePrim(sky_path, "Xform")
    asset_path = os.path.relpath(sky_usd.resolve(), output_usd.resolve().parent)
    sky.GetReferences().AddReference(asset_path, Sdf.Path("/World"))
    sky.SetCustomDataByKey("navverse_preview_lighting", "CloudySky")
    return asset_path


def set_preview_cameras(
    stage: Usd.Stage,
    eye: np.ndarray,
    target: np.ndarray,
    *,
    source: str,
    focal_length: float = 30.0,
) -> dict[str, Any]:
    direction = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    horizontal = max(float(np.linalg.norm(direction[:2])), 1e-9)
    heading = math.atan2(float(direction[1]), float(direction[0]))
    downward_pitch = math.atan2(float(eye[2] - target[2]), horizontal)
    rotate_x = 90.0 - math.degrees(downward_pitch)
    rotate_z = math.degrees(heading) - 90.0
    for camera_path in CAMERA_PATHS:
        if stage.GetPrimAtPath(camera_path):
            stage.RemovePrim(camera_path)
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(float(focal_length))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 5000.0))
        xform = UsdGeom.XformCommonAPI(camera.GetPrim())
        xform.SetTranslate(Gf.Vec3d(*(float(value) for value in eye)))
        xform.SetRotate(Gf.Vec3f(float(rotate_x), 0.0, float(rotate_z)))
        camera.GetPrim().SetCustomDataByKey("navverse_preview_camera_source", source)
    return {
        "camera_paths": list(CAMERA_PATHS),
        "eye": [float(value) for value in eye],
        "target": [float(value) for value in target],
        "heading_rad": heading,
        "source": source,
    }


def _read_points(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return list(json.load(stream).get("points", []))


def _preview_obstacle_bounds(
    stage: Usd.Stage,
    root_paths: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
    )
    minimums = []
    maximums = []
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root:
            continue
        for child in root.GetChildren():
            try:
                bounds = cache.ComputeWorldBound(child).ComputeAlignedBox()
            except Exception:
                continue
            minimum = np.asarray(bounds.GetMin(), dtype=float)
            maximum = np.asarray(bounds.GetMax(), dtype=float)
            if np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum)):
                minimums.append(minimum[:2])
                maximums.append(maximum[:2])
    if not minimums:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    return np.asarray(minimums, dtype=float), np.asarray(maximums, dtype=float)


def _camera_corridor_is_clear(
    origin: np.ndarray,
    forward: np.ndarray,
    obstacle_minimums: np.ndarray,
    obstacle_maximums: np.ndarray,
) -> bool:
    if not len(obstacle_minimums):
        return True
    for distance, margin in ((0.0, 5.0), (2.0, 1.5), (4.0, 1.0), (6.0, 0.7)):
        sample = origin + forward * distance
        hits = np.all(sample >= obstacle_minimums - margin, axis=1) & np.all(
            sample <= obstacle_maximums + margin,
            axis=1,
        )
        if np.any(hits):
            return False
    return True


def _camera_view_obstruction_score(
    origin: np.ndarray,
    forward: np.ndarray,
    obstacle_minimums: np.ndarray,
    obstacle_maximums: np.ndarray,
    *,
    max_distance: float = 18.0,
) -> float:
    if not len(obstacle_minimums):
        return 0.0
    centers = (obstacle_minimums + obstacle_maximums) * 0.5
    half_extents = (obstacle_maximums - obstacle_minimums) * 0.5
    right = np.asarray([-forward[1], forward[0]], dtype=float)
    offsets = centers - origin
    along = offsets @ forward
    lateral = offsets @ right
    along_radius = half_extents @ np.abs(forward)
    lateral_radius = half_extents @ np.abs(right)
    near = along - along_radius
    far = along + along_radius
    lateral_clearance = np.maximum(np.abs(lateral) - lateral_radius, 0.0)
    cone_half_width = 0.8 + np.maximum(near, 0.25) * 0.48
    visible = (far >= 0.25) & (near <= max_distance) & (lateral_clearance < cone_half_width)
    if not np.any(visible):
        return 0.0
    depth_weight = np.clip(
        (max_distance - np.maximum(near, 0.25)) / max(max_distance, 0.25),
        0.0,
        1.0,
    )
    overlap_weight = np.clip(
        (cone_half_width - lateral_clearance) / np.maximum(cone_half_width, 0.1),
        0.0,
        1.0,
    )
    size_weight = 1.0 + np.minimum(np.linalg.norm(half_extents, axis=1), 5.0)
    return float(np.sum(depth_weight[visible] * overlap_weight[visible] * size_weight[visible]))


def add_outdoor_preview_cameras(stage: Usd.Stage, city_dir: Path, terrain_query: Any) -> dict[str, Any]:
    points = _read_points(city_dir / "lane_points.json")
    point_source = "lane"
    if not points:
        points = _read_points(city_dir / "sidewalk_points.json")
        point_source = "sidewalk"
    if not points:
        eye = np.asarray([0.0, 0.0, 1.5], dtype=float)
        target = np.asarray([8.0, 0.0, 1.5], dtype=float)
        return set_preview_cameras(stage, eye, target, source="fallback_origin")

    coordinates = np.asarray([[float(point["utm_x"]), float(point["utm_y"])] for point in points], dtype=float)
    center = np.median(coordinates, axis=0)
    obstacle_minimums, obstacle_maximums = _preview_obstacle_bounds(
        stage,
        ("/World/Objaverse", "/World/RoadObjects", "/World/store", "/World/city"),
    )
    view_minimums, view_maximums = _preview_obstacle_bounds(
        stage,
        ("/World/Objaverse", "/World/RoadObjects"),
    )
    building_minimums, building_maximums = _preview_obstacle_bounds(
        stage,
        ("/World/city", "/World/store"),
    )
    candidate_order = np.argsort(np.sum((coordinates - center) ** 2, axis=1))
    index = int(candidate_order[0])
    candidate_rank = 0
    selected_forward = None
    selected_score = math.inf
    search_limit = min(len(candidate_order), 1024)
    for rank, candidate_index in enumerate(candidate_order[:search_limit]):
        candidate = points[int(candidate_index)]
        heading = float(candidate.get("heading", 0.0))
        forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
        for direction in (forward, -forward):
            origin = coordinates[int(candidate_index)]
            if not _camera_corridor_is_clear(
                origin,
                direction,
                obstacle_minimums,
                obstacle_maximums,
            ):
                continue
            score = _camera_view_obstruction_score(origin, direction, view_minimums, view_maximums)
            score += 3.0 * _camera_view_obstruction_score(
                origin,
                direction,
                building_minimums,
                building_maximums,
                max_distance=12.0,
            )
            if (score, rank) < (selected_score, candidate_rank):
                index = int(candidate_index)
                candidate_rank = rank
                selected_forward = direction
                selected_score = score
    point = points[index]
    x, y = coordinates[index]
    fallback_z = float(point.get("height", 0.0))
    z = float(terrain_query.height_at(float(x), float(y), fallback_z))
    if selected_forward is None:
        heading = float(point.get("heading", 0.0))
        selected_forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
        selected_score = _camera_view_obstruction_score(
            coordinates[index],
            selected_forward,
            view_minimums,
            view_maximums,
        )
        selected_score += 3.0 * _camera_view_obstruction_score(
            coordinates[index],
            selected_forward,
            building_minimums,
            building_maximums,
            max_distance=12.0,
        )
    eye = np.asarray([x, y, z + 1.5], dtype=float)
    target = np.asarray(
        [x + selected_forward[0] * 8.0, y + selected_forward[1] * 8.0, z + 1.5],
        dtype=float,
    )
    summary = set_preview_cameras(
        stage,
        eye,
        target,
        source=f"low_obstruction_central_{point_source}_view",
    )
    summary["candidate_rank"] = candidate_rank
    summary["obstacle_count"] = len(obstacle_minimums)
    summary["view_obstacle_count"] = len(view_minimums)
    summary["building_obstacle_count"] = len(building_minimums)
    summary["view_obstruction_score"] = selected_score
    summary["point_source"] = point_source
    return summary


def add_innout_preview_cameras(stage: Usd.Stage, plan: Any) -> dict[str, Any]:
    center = np.asarray(plan.door_center_xy, dtype=float)
    outward = np.asarray(plan.outward_normal_xy, dtype=float)
    outward /= max(float(np.linalg.norm(outward)), 1e-9)
    tangent = np.asarray(plan.tangent_xy, dtype=float)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
    eye_xy = center + outward * 4.0 - tangent * 2.0
    eye = np.asarray([eye_xy[0], eye_xy[1], float(plan.terrain_z) + 1.5], dtype=float)
    target = np.asarray([center[0], center[1], float(plan.door_bottom_z) + 1.5], dtype=float)
    return set_preview_cameras(
        stage,
        eye,
        target,
        source="innout_entrance_street_view",
        focal_length=24.0,
    )
