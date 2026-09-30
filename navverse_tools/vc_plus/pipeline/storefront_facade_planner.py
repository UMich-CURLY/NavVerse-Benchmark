#!/usr/bin/env python3
"""Align storefront POIs to nearby road-facing building facades."""

from __future__ import annotations

import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from terrain_utils import TerrainHeightQuery


REAL_STOREFRONT_ASSET_TYPES = {
    "restaurant",
    "cafe",
    "fast_food",
    "convenience_store",
    "bank",
    "pharmacy",
    "bar",
    "library",
    "theatre",
    "cinema",
    "post_office",
}


@dataclass
class RoadSegment:
    start: np.ndarray
    end: np.ndarray
    start_z: float
    end_z: float


@dataclass
class BuildingEdge:
    building_path: Path
    building_name: str
    edge_index: int
    start: np.ndarray
    end: np.ndarray
    tangent: np.ndarray
    outward: np.ndarray
    length: float
    min_z: float
    footprint: np.ndarray
    mesh_support_distance: float


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize(vec: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        return np.zeros_like(vec)
    return vec / length


def project_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float, float]:
    edge = end - start
    length_sq = float(edge @ edge)
    if length_sq <= 1e-9:
        return start.copy(), 0.0, float(np.linalg.norm(point - start))
    t = float(np.clip(((point - start) @ edge) / length_sq, 0.0, 1.0))
    projected = start + t * edge
    return projected, t, float(np.linalg.norm(point - projected))


def mapped_vertices_from_glb(path: Path) -> np.ndarray | None:
    try:
        scene = trimesh.load(path, force="scene")
    except Exception:
        return None
    if hasattr(scene, "geometry"):
        meshes = [mesh for mesh in scene.geometry.values() if len(getattr(mesh, "vertices", []))]
        if not meshes:
            return None
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float)
    if vertices.size == 0:
        return None
    return np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])


def mapped_mesh_from_glb(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        scene = trimesh.load(path, force="scene")
    except Exception:
        return None
    if hasattr(scene, "geometry"):
        meshes = [mesh for mesh in scene.geometry.values() if len(getattr(mesh, "vertices", []))]
        if not meshes:
            return None
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float)
    faces = np.asarray(getattr(mesh, "faces", []), dtype=int)
    if vertices.size == 0 or faces.size == 0:
        return None
    mapped = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
    return mapped, faces


def footprint_polygon(vertices: np.ndarray) -> tuple[np.ndarray, float] | None:
    min_z = float(vertices[:, 2].min())
    max_z = float(vertices[:, 2].max())
    lower = vertices[:, 2] <= min_z + max(0.25, 0.05 * (max_z - min_z))
    points = vertices[lower][:, :2]
    if len(points) < 3:
        points = vertices[:, :2]
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
    except Exception:
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        polygon = np.asarray([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]], dtype=float)
    if len(polygon) < 3:
        return None
    return polygon, min_z


def min_point_to_projected_triangles_distance(point: np.ndarray, triangles: np.ndarray) -> float:
    points = triangles[:, :, :2]
    a = points[:, 0, :]
    b = points[:, 1, :]
    c = points[:, 2, :]
    v0 = c - a
    v1 = b - a
    v2 = point - a
    denom = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    valid = np.abs(denom) > 1e-10
    if np.any(valid):
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / np.where(valid, denom, 1.0)
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / np.where(valid, denom, 1.0)
        inside = valid & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1.0 + 1e-8)
        if np.any(inside):
            return 0.0

    min_distance = float("inf")
    for index in range(3):
        start = points[:, index, :]
        end = points[:, (index + 1) % 3, :]
        segment = end - start
        denom = np.einsum("ij,ij->i", segment, segment)
        raw_t = np.einsum("ij,ij->i", point - start, segment) / np.where(denom > 1e-12, denom, 1.0)
        t = np.clip(raw_t, 0.0, 1.0)
        projected = start + segment * t[:, None]
        distances = np.linalg.norm(projected - point, axis=1)
        if len(distances):
            min_distance = min(min_distance, float(np.min(distances)))
    return min_distance


def edge_mesh_support_distance(
    vertices: np.ndarray,
    faces: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    min_z: float,
) -> float:
    triangles = vertices[faces]
    lower_band = max(1.0, 0.08 * max(1.0, float(vertices[:, 2].max() - min_z)))
    mask = (triangles[:, :, 2].min(axis=1) <= min_z + lower_band) & (
        triangles[:, :, 2].max(axis=1) >= min_z - 0.2
    )
    candidates = triangles[mask] if np.any(mask) else triangles
    edge = end - start
    length = float(np.linalg.norm(edge))
    if length <= 1e-9 or len(candidates) == 0:
        return float("inf")
    samples = [0.5] if length < 6.0 else [0.25, 0.5, 0.75]
    max_distance = 0.0
    for sample in samples:
        point = start + edge * sample
        min_distance = min_point_to_projected_triangles_distance(point, candidates)
        max_distance = max(max_distance, min_distance)
    return float(max_distance)


def load_building_edges(
    city_dir: Path,
    min_edge_length: float,
    mesh_wall_support_tolerance: float,
) -> tuple[list[BuildingEdge], int]:
    edges: list[BuildingEdge] = []
    skipped_unsupported = 0
    building_dir = city_dir / "buildings"
    for glb_path in sorted(building_dir.glob("*.glb")):
        if glb_path.stem == "buildings_element":
            continue
        mapped_mesh = mapped_mesh_from_glb(glb_path)
        if mapped_mesh is None:
            continue
        vertices, faces = mapped_mesh
        footprint = footprint_polygon(vertices)
        if footprint is None:
            continue
        polygon, min_z = footprint
        centroid = polygon.mean(axis=0)
        for edge_index in range(len(polygon)):
            start = polygon[edge_index]
            end = polygon[(edge_index + 1) % len(polygon)]
            edge = end - start
            length = float(np.linalg.norm(edge))
            if length < min_edge_length:
                continue
            support_distance = edge_mesh_support_distance(vertices, faces, start, end, min_z)
            if support_distance > mesh_wall_support_tolerance:
                skipped_unsupported += 1
                continue
            tangent = edge / length
            normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
            midpoint = (start + end) * 0.5
            if float(normal @ (midpoint - centroid)) < 0.0:
                normal = -normal
            edges.append(
                BuildingEdge(
                    building_path=glb_path,
                    building_name=glb_path.stem,
                    edge_index=edge_index,
                    start=start,
                    end=end,
                    tangent=tangent,
                    outward=normal,
                    length=length,
                    min_z=min_z,
                    footprint=polygon,
                    mesh_support_distance=support_distance,
                )
            )
    return edges, skipped_unsupported


def load_road_segments(osm_path: Path) -> list[RoadSegment]:
    root = ET.parse(osm_path).getroot()
    nodes: dict[str, np.ndarray] = {}
    heights: dict[str, float] = {}
    for element in root:
        if element.tag != "node":
            continue
        tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
        node_id = element.get("id", "")
        nodes[node_id] = np.asarray([parse_float(tags.get("utm_x")), parse_float(tags.get("utm_y"))], dtype=float)
        heights[node_id] = parse_float(tags.get("height"))

    segments: list[RoadSegment] = []
    for element in root:
        if element.tag != "way":
            continue
        tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
        if "highway" not in tags:
            continue
        refs = [nd.get("ref") for nd in element.findall("nd")]
        for index in range(len(refs) - 1):
            start = nodes.get(refs[index] or "")
            end = nodes.get(refs[index + 1] or "")
            if start is None or end is None:
                continue
            if float(np.linalg.norm(end - start)) <= 1e-6:
                continue
            segments.append(
                RoadSegment(
                    start=start,
                    end=end,
                    start_z=heights.get(refs[index] or "", 0.0),
                    end_z=heights.get(refs[index + 1] or "", 0.0),
                )
            )
    return segments


def nearest_road(point: np.ndarray, segments: list[RoadSegment], max_distance: float) -> dict[str, Any]:
    best: dict[str, Any] = {"distance": None, "direction": None, "normal": None, "height": None, "segment_t": None}
    best_distance = float("inf")
    for segment in segments:
        projected, segment_t, distance = project_point_to_segment(point, segment.start, segment.end)
        if distance >= best_distance or distance > max_distance:
            continue
        direction = normalize(segment.end - segment.start)
        normal = normalize(projected - point)
        if np.linalg.norm(direction) <= 1e-9 or np.linalg.norm(normal) <= 1e-9:
            continue
        height = float(segment.start_z + (segment.end_z - segment.start_z) * segment_t)
        best_distance = distance
        best = {
            "distance": distance,
            "direction": direction,
            "normal": normal,
            "height": height,
            "segment_t": segment_t,
        }
    return best


def storefront_ground_height(
    terrain: TerrainHeightQuery,
    roads: list[RoadSegment],
    final_xy: np.ndarray,
    outward: np.ndarray,
    fallback: float,
    *,
    road_max_distance: float,
) -> tuple[float, dict[str, Any]]:
    samples = [final_xy, final_xy + outward * 1.2, final_xy + outward * 3.0]
    road_hits = [nearest_road(sample, roads, road_max_distance) for sample in samples]
    road_hits = [hit for hit in road_hits if hit.get("height") is not None]
    if road_hits:
        road_hit = min(road_hits, key=lambda hit: float(hit["distance"]))
        return float(road_hit["height"]), road_hit
    terrain_z = terrain.height_near([(float(pt[0]), float(pt[1])) for pt in samples], fallback, mode="max")
    return float(terrain_z), {"distance": None, "height": None}


def yaw_for_storefront_front(outward: np.ndarray) -> float:
    # Storefront assets use local +Y as the facade/front direction.
    return math.atan2(-float(outward[0]), float(outward[1]))


def choose_facade(
    poi_xy: np.ndarray,
    road: dict[str, Any],
    edges: list[BuildingEdge],
    max_distance: float,
    min_facing_dot: float,
    min_facade_width: float,
    edge_margin: float,
    excluded_edge_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    road_normal = road.get("normal")
    best = None
    best_score = float("inf")
    for edge in edges:
        if excluded_edge_keys and f"{edge.building_name}:{edge.edge_index}" in excluded_edge_keys:
            continue
        if edge.length < min_facade_width + edge_margin * 2.0:
            continue
        projected, edge_t, poi_distance = project_point_to_segment(poi_xy, edge.start, edge.end)
        if poi_distance > max_distance:
            continue
        edge_s = edge_t * edge.length
        half_width = min_facade_width * 0.5 + edge_margin
        if edge_s < half_width or edge.length - edge_s < half_width:
            continue
        if not edge_front_clearance_ok(edge, edge_s, min_facade_width, edges, edge_margin):
            continue
        if road_normal is not None:
            facing_dot = float(edge.outward @ road_normal)
        else:
            facing_dot = float(edge.outward @ normalize(poi_xy - projected))
        if facing_dot < min_facing_dot:
            continue
        score = poi_distance + max(0.0, 0.8 - facing_dot) * 8.0 - min(edge.length, 25.0) * 0.03
        if score < best_score:
            best_score = score
            best = {
                "edge": edge,
                "projected": projected,
                "edge_t": edge_t,
                "poi_to_facade_distance": poi_distance,
                "facing_dot": facing_dot,
                "score": score,
            }
    return best


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def edge_front_clearance_ok(
    edge: BuildingEdge,
    edge_s: float,
    facade_width: float,
    edges: list[BuildingEdge],
    edge_margin: float,
) -> bool:
    center = edge.start + edge.tangent * edge_s
    half_width = facade_width * 0.5 + edge_margin
    lateral_samples = [-half_width, 0.0, half_width]
    front_samples = [1.2, 2.5, 4.0, 6.0]
    for lateral in lateral_samples:
        base = center + edge.tangent * lateral
        for distance in front_samples:
            point = base + edge.outward * distance
            for other in edges:
                if point_in_polygon(point, other.footprint):
                    return False
    return True


def avoid_same_edge_overlap(
    selected: dict[str, Any],
    placed: dict[str, list[float]],
    spacing: float,
    footprint_width: float,
    edge_margin: float,
) -> tuple[np.ndarray, float]:
    edge: BuildingEdge = selected["edge"]
    key = f"{edge.building_name}:{edge.edge_index}"
    half_width = footprint_width * 0.5 + edge_margin
    required_spacing = max(spacing, footprint_width + edge_margin * 2.0)
    min_s = min(max(half_width, 0.0), edge.length * 0.5)
    max_s = max(edge.length - half_width, edge.length * 0.5)
    edge_s = float(np.clip(selected["edge_t"] * edge.length, min_s, max_s))
    used = placed.setdefault(key, [])
    if used:
        candidates = [edge_s]
        max_steps = max(2, int(math.ceil(edge.length / required_spacing)) + 2)
        for step in range(1, max_steps + 1):
            candidates.append(float(np.clip(edge_s - step * required_spacing, min_s, max_s)))
            candidates.append(float(np.clip(edge_s + step * required_spacing, min_s, max_s)))
        candidates = sorted(set(candidates), key=lambda value: abs(value - edge_s))
        edge_s = max(candidates, key=lambda value: min(abs(value - other) for other in used))
        for candidate in candidates:
            if all(abs(candidate - other) >= required_spacing for other in used):
                edge_s = candidate
                break
    used.append(edge_s)
    return edge.start + edge.tangent * edge_s, edge_s


def resolve_asset_path(asset_path: Any, work_dir: Path) -> Path | None:
    if not asset_path:
        return None
    path = Path(str(asset_path))
    if path.is_absolute():
        return path
    return work_dir / path


def asset_bbox_metrics(
    asset_path: Path | None,
    default_width: float,
    default_height: float,
    target_height: float | None = None,
) -> dict[str, float]:
    fallback_height = float(target_height or default_height)
    fallback = {
        "width": float(default_width),
        "height": fallback_height,
        "raw_width": float(default_width),
        "raw_height": float(default_height),
        "scale_multiplier": 1.0,
        "inward_y": 0.0,
        "y_min": 0.0,
        "y_max": 0.0,
    }
    if asset_path is None or not asset_path.exists():
        return fallback
    try:
        from pxr import Usd, UsdGeom
    except Exception:
        return fallback
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        return fallback
    root = stage.GetDefaultPrim()
    if not root:
        roots = list(stage.GetPseudoRoot().GetChildren())
        root = roots[0] if roots else None
    if not root:
        return fallback
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root).ComputeAlignedBox()
    width = float(box.GetMax()[0] - box.GetMin()[0])
    height = float(box.GetMax()[2] - box.GetMin()[2])
    y_min = float(box.GetMin()[1])
    y_max = float(box.GetMax()[1])
    inward_y = max(0.0, -y_min)
    if not math.isfinite(width) or width <= 0.0:
        width = float(default_width)
    if not math.isfinite(height) or height <= 0.0:
        height = float(default_height)
    if not math.isfinite(y_min):
        y_min = 0.0
    if not math.isfinite(y_max):
        y_max = 0.0
    if not math.isfinite(inward_y):
        inward_y = 0.0
    raw_width = width
    raw_height = height
    scale_multiplier = 1.0
    if target_height is not None and target_height > 0.0 and raw_height > 0.0:
        scale_multiplier = float(target_height) / raw_height
        width *= scale_multiplier
        height *= scale_multiplier
        inward_y *= scale_multiplier
        y_min *= scale_multiplier
        y_max *= scale_multiplier
    return {
        "width": width,
        "height": height,
        "raw_width": raw_width,
        "raw_height": raw_height,
        "scale_multiplier": scale_multiplier,
        "inward_y": inward_y,
        "y_min": y_min,
        "y_max": y_max,
    }


def asset_facade_width(asset_path: Path | None, default_width: float) -> float:
    return asset_bbox_metrics(asset_path, default_width, 3.0)["width"]


def align_storefront_plan(
    plan: dict[str, Any],
    city_dir: Path,
    osm_path: Path,
    work_dir: Path,
    *,
    max_poi_to_facade_distance: float = 50.0,
    max_road_distance: float = 14.0,
    min_edge_length: float = 2.5,
    min_facing_dot: float = 0.65,
    storefront_offset: float = 0.01,
    same_edge_spacing: float = 2.2,
    opening_width: float = 3.2,
    opening_height: float = 3.0,
    opening_width_margin: float = 0.04,
    opening_height_margin: float = 0.04,
    edge_margin: float = 2.0,
    fallback_storefront_width: float = 4.0,
    mesh_wall_support_tolerance: float = 0.75,
    storefront_target_height: float | None = 4.0,
) -> dict[str, Any]:
    edges, skipped_unsupported_edges = load_building_edges(city_dir, min_edge_length, mesh_wall_support_tolerance)
    source_edge_count = len(edges)
    roads = load_road_segments(osm_path)
    terrain = TerrainHeightQuery(city_dir)
    placed_by_edge: dict[str, list[float]] = {}
    asset_metric_cache: dict[str, dict[str, float]] = {}
    records = []
    status_counts: dict[str, int] = {}
    excluded_edge_keys = set(plan.get("render_rejected_edges") or [])
    excluded_edge_keys.update({
        f"{item.get('building_name')}:{item.get('building_edge_index')}"
        for item in plan.get("placements", [])
        if item.get("render_rejected") and item.get("building_name") is not None and item.get("building_edge_index") is not None
    })
    spatial_filter = plan.get("spatial_filter") or {}
    focus_values = spatial_filter.get("center_xy")
    focus_radius = parse_float(spatial_filter.get("radius_m"))
    focus_center = np.asarray(focus_values, dtype=float) if focus_values and focus_radius is not None else None

    def inside_focus(point: np.ndarray) -> bool:
        return focus_center is None or float(np.linalg.norm(point - focus_center)) <= float(focus_radius) + 1e-6

    reserved_openings = []
    for opening in plan.get("reserved_openings") or []:
        center = opening.get("center_xy")
        tangent = opening.get("tangent_xy")
        outward = opening.get("outward_normal_xy")
        if not center or not tangent or not outward:
            continue
        reserved_openings.append({
            "center": np.asarray(center, dtype=float),
            "tangent": np.asarray(tangent, dtype=float),
            "outward": np.asarray(outward, dtype=float),
            "width": float(parse_float(opening.get("width_m"), 0.0)),
            "lateral_clearance": float(parse_float(opening.get("lateral_clearance_m"), 1.0)),
            "normal_clearance": float(parse_float(opening.get("normal_clearance_m"), 2.0)),
        })

    def overlaps_reserved_opening(point: np.ndarray, width: float) -> bool:
        for opening in reserved_openings:
            delta = point - opening["center"]
            lateral = abs(float(delta @ opening["tangent"]))
            normal = abs(float(delta @ opening["outward"]))
            half_width = 0.5 * (float(width) + opening["width"]) + opening["lateral_clearance"]
            if lateral <= half_width and normal <= opening["normal_clearance"]:
                return True
        return False

    def edge_intersects_focus(edge: BuildingEdge) -> bool:
        if focus_center is None:
            return True
        delta = edge.end - edge.start
        denom = float(delta @ delta)
        if denom <= 1e-12:
            nearest = edge.start
        else:
            t = float(np.clip(((focus_center - edge.start) @ delta) / denom, 0.0, 1.0))
            nearest = edge.start + delta * t
        return float(np.linalg.norm(nearest - focus_center)) <= float(focus_radius) + 1e-6

    excluded_building_keys = set()
    for value in plan.get("excluded_buildings") or []:
        path = Path(str(value))
        excluded_building_keys.update({str(value).lower(), path.name.lower(), path.stem.lower()})

    def building_is_excluded(edge: BuildingEdge) -> bool:
        values = {
            str(edge.building_path).lower(),
            edge.building_path.name.lower(),
            edge.building_path.stem.lower(),
            str(edge.building_name).lower(),
        }
        return bool(values.intersection(excluded_building_keys))

    edges = [edge for edge in edges if edge_intersects_focus(edge) and not building_is_excluded(edge)]

    for item in plan.get("placements", []):
        record = dict(item)
        if item.get("status") != "planned":
            records.append(record)
            status = str(item.get("facade_status") or item.get("status") or "skipped")
            status_counts[status] = status_counts.get(status, 0) + 1
            continue
        poi_xy = np.asarray([parse_float(item.get("utm_x")), parse_float(item.get("utm_y"))], dtype=float)
        asset_path = resolve_asset_path(item.get("asset_path"), work_dir)
        asset_key = str(asset_path) if asset_path else ""
        if asset_key not in asset_metric_cache:
            asset_metric_cache[asset_key] = asset_bbox_metrics(
                asset_path,
                max(opening_width, fallback_storefront_width),
                opening_height,
                storefront_target_height,
            )
        asset_metrics = asset_metric_cache[asset_key]
        facade_width = max(asset_metrics["width"], opening_width)
        cut_width = facade_width + opening_width_margin
        cut_height = max(1.6, asset_metrics["height"] + opening_height_margin)
        item_offset = storefront_offset
        road = nearest_road(poi_xy, roads, max_road_distance)
        selected = choose_facade(
            poi_xy,
            road,
            edges,
            max_poi_to_facade_distance,
            min_facing_dot,
            facade_width,
            edge_margin,
            excluded_edge_keys,
        )
        if selected is None:
            road_normal = road.get("normal")
            outward = road_normal if road_normal is not None else np.asarray([0.0, 1.0], dtype=float)
            final_xy = poi_xy
            status = "fallback_no_road_facing_facade"
            building_path = None
            building_name = None
            edge_index = None
            edge_length = None
            edge_s = None
            poi_distance = None
            facing_dot = None
        else:
            edge = selected["edge"]
            outward = edge.outward
            facade_xy, edge_s = avoid_same_edge_overlap(
                selected,
                placed_by_edge,
                same_edge_spacing,
                facade_width,
                edge_margin,
            )
            final_xy = facade_xy
            final_xy = final_xy + outward * item_offset
            if inside_focus(final_xy) and not overlaps_reserved_opening(final_xy, facade_width):
                status = "facade_aligned"
                building_path = os.path.relpath(edge.building_path, work_dir)
                building_name = edge.building_name
                edge_index = edge.edge_index
                edge_length = edge.length
                poi_distance = selected["poi_to_facade_distance"]
                facing_dot = selected["facing_dot"]
            else:
                final_xy = poi_xy
                status = "reserved_opening_overlap" if inside_focus(final_xy) else "outside_spatial_filter"
                used = placed_by_edge.get(f"{edge.building_name}:{edge.edge_index}", [])
                if edge_s in used:
                    used.remove(edge_s)
                building_path = None
                building_name = None
                edge_index = None
                edge_length = None
                edge_s = None
                poi_distance = selected["poi_to_facade_distance"]
                facing_dot = selected["facing_dot"]

        height, ground_road = storefront_ground_height(
            terrain,
            roads,
            final_xy,
            outward,
            parse_float(item.get("height")),
            road_max_distance=max_road_distance + 4.0,
        )
        final_location = [float(final_xy[0]), float(final_xy[1]), height]
        facade_center = [
            float(final_xy[0] - outward[0] * item_offset),
            float(final_xy[1] - outward[1] * item_offset),
            float(height + cut_height * 0.5),
        ]
        record.update(
            {
                "facade_status": status,
                "final_location": final_location,
                "heading": yaw_for_storefront_front(outward),
                "outward_normal_xy": [float(outward[0]), float(outward[1])],
                "opening_center": facade_center,
                "opening_bottom_z": float(height),
                "opening_width": float(cut_width),
                "opening_height": float(cut_height),
                "storefront_facade_width": float(facade_width),
                "storefront_offset": float(item_offset),
                "storefront_asset_height": float(asset_metrics["height"]),
                "storefront_asset_raw_height": float(asset_metrics["raw_height"]),
                "storefront_asset_raw_width": float(asset_metrics["raw_width"]),
                "storefront_scale_multiplier": float(asset_metrics["scale_multiplier"]),
                "storefront_asset_inward_y": float(asset_metrics["inward_y"]),
                "storefront_asset_y_min": float(asset_metrics["y_min"]),
                "storefront_asset_y_max": float(asset_metrics["y_max"]),
                "opening_inside_depth": 1.35,
                "opening_outside_depth": 0.25,
                "storefront_ground_source": "road" if ground_road.get("height") is not None else "terrain",
                "storefront_ground_road_distance": ground_road.get("distance"),
                "storefront_ground_road_height": ground_road.get("height"),
                "storefront_edge_margin": float(edge_margin),
                "building_edge_s": float(edge_s) if edge_s is not None else None,
                "building_edge_start_xy": [float(value) for value in edge.start] if selected is not None else None,
                "building_edge_end_xy": [float(value) for value in edge.end] if selected is not None else None,
                "road_distance": road.get("distance"),
                "building_path": building_path,
                "building_name": building_name,
                "building_edge_index": edge_index,
                "building_edge_length": edge_length,
                "poi_to_facade_distance": poi_distance,
                "road_facing_dot": facing_dot,
            }
        )
        records.append(record)
        status_counts[status] = status_counts.get(status, 0) + 1

    result = dict(plan)
    target_aligned = int(parse_float(plan.get("min_storefronts"), 0.0))
    if target_aligned > status_counts.get("facade_aligned", 0):
        catalog = plan.get("asset_catalog") or {}
        rng_seed = int(parse_float(plan.get("seed"), 0.0)) + sum(ord(ch) for ch in str(plan.get("city") or "")) + 9173
        rng = random.Random(rng_seed)
        templates = [
            record
            for record in records
            if record.get("status") == "planned"
            and record.get("asset_path")
            and (record.get("asset_type") or record.get("amenity")) in REAL_STOREFRONT_ASSET_TYPES
        ]
        if not templates:
            for asset_type, asset_paths in sorted(catalog.items()):
                if asset_type not in REAL_STOREFRONT_ASSET_TYPES:
                    continue
                for asset_path in asset_paths:
                    templates.append(
                        {
                            "amenity": asset_type,
                            "asset_type": asset_type,
                            "asset_path": asset_path,
                            "height": 0.0,
                            "status": "planned",
                        }
                    )
        edge_candidates = []
        synthetic_road_limit = max(
            max_road_distance,
            min(float(focus_radius), 50.0) if focus_radius is not None else max_road_distance,
        )
        for edge in edges:
            if f"{edge.building_name}:{edge.edge_index}" in excluded_edge_keys:
                continue
            if edge.length < opening_width + edge_margin * 2.0:
                continue
            midpoint = (edge.start + edge.end) * 0.5
            road = nearest_road(midpoint, roads, synthetic_road_limit)
            road_normal = road.get("normal")
            road_distance = float(road["distance"]) if road.get("distance") is not None else math.inf
            facing_dot = float(edge.outward @ road_normal) if road_normal is not None else -1.0
            relaxed = road_distance > max_road_distance or facing_dot < min_facing_dot
            edge_candidates.append((relaxed, road_distance, -edge.length, edge))
        edge_candidates.sort(key=lambda value: (value[0], value[1], value[2], value[3].building_name, value[3].edge_index))

        next_index = max((int(parse_float(record.get("index"), -1)) for record in records), default=-1) + 1
        synthetic_edge_type_counts: dict[str, dict[str, int]] = {}
        synthetic_edge_asset_counts: dict[str, dict[str, int]] = {}
        synthetic_edge_last_type: dict[str, str] = {}
        synthetic_edge_last_asset: dict[str, str] = {}
        added = 0

        def choose_synthetic_fill(
            edge_key: str,
            max_facade_width: float,
        ) -> tuple[dict[str, Any], Path | None, str, str | None] | None:
            type_counts = synthetic_edge_type_counts.setdefault(edge_key, {})
            asset_counts = synthetic_edge_asset_counts.setdefault(edge_key, {})
            choices = []
            last_type = synthetic_edge_last_type.get(edge_key)
            for candidate in templates:
                candidate_type = str(candidate.get("asset_type") or candidate.get("amenity") or "")
                candidate_amenity = str(candidate.get("amenity") or candidate_type)
                asset_candidates = catalog.get(candidate_type) or catalog.get(candidate_amenity) or []
                if not asset_candidates and candidate.get("asset_path"):
                    asset_candidates = [candidate.get("asset_path")]
                fitting_assets = []
                for asset in asset_candidates:
                    asset_path = resolve_asset_path(asset, work_dir)
                    asset_key = str(asset_path) if asset_path else ""
                    if asset_key not in asset_metric_cache:
                        asset_metric_cache[asset_key] = asset_bbox_metrics(
                            asset_path,
                            max(opening_width, fallback_storefront_width),
                            opening_height,
                            storefront_target_height,
                        )
                    width = max(asset_metric_cache[asset_key]["width"], opening_width)
                    if width <= max_facade_width + 1e-6:
                        fitting_assets.append(asset)
                if not fitting_assets:
                    continue
                type_count = type_counts.get(candidate_type, 0)
                choices.append((type_count, candidate_type == last_type, rng.random(), candidate, candidate_type, fitting_assets))
            if not choices:
                return None

            min_count = min(choice[0] for choice in choices)
            balanced = [choice for choice in choices if choice[0] == min_count]
            if len({choice[4] for choice in balanced}) > 1:
                balanced = [choice for choice in balanced if not choice[1]] or balanced
            _count, _same_last, _tie, template, template_type, asset_candidates = min(balanced, key=lambda value: value[2])

            last_asset = synthetic_edge_last_asset.get(edge_key)
            asset_options = list(asset_candidates)
            if len(asset_options) > 1:
                asset_options = [asset for asset in asset_options if str(asset) != last_asset] or asset_options
            min_asset_count = min(asset_counts.get(str(asset), 0) for asset in asset_options)
            asset_options = [asset for asset in asset_options if asset_counts.get(str(asset), 0) == min_asset_count]
            selected_asset = rng.choice(asset_options)
            selected_asset_path = resolve_asset_path(selected_asset, work_dir)
            return template, selected_asset_path, template_type, str(selected_asset)

        for _relaxed, _road_distance, _neg_length, edge in edge_candidates:
            if status_counts.get("facade_aligned", 0) >= target_aligned:
                break
            if not templates:
                break
            key = f"{edge.building_name}:{edge.edge_index}"
            used = placed_by_edge.setdefault(key, [])
            candidate_slots = []
            slot_spacing = max(0.5, same_edge_spacing)
            min_s = min(max(opening_width * 0.5 + edge_margin, 0.0), edge.length * 0.5)
            max_s = max(edge.length - min_s, edge.length * 0.5)
            slot = min_s
            while slot <= max_s + 1e-6:
                candidate_slots.append(float(slot))
                slot += slot_spacing
            if not candidate_slots:
                candidate_slots = [float((min_s + max_s) * 0.5)]
            for edge_s in candidate_slots:
                if status_counts.get("facade_aligned", 0) >= target_aligned:
                    break
                max_facade_width = max(
                    0.0,
                    2.0 * (min(edge_s, edge.length - edge_s) - edge_margin),
                )
                selected_fill = choose_synthetic_fill(key, max_facade_width)
                if selected_fill is None:
                    continue
                template, asset_path, template_asset_type, selected_asset_key = selected_fill
                asset_key = str(asset_path) if asset_path else ""
                if asset_key not in asset_metric_cache:
                    asset_metric_cache[asset_key] = asset_bbox_metrics(
                        asset_path,
                        max(opening_width, fallback_storefront_width),
                        opening_height,
                        storefront_target_height,
                    )
                asset_metrics = asset_metric_cache[asset_key]
                facade_width = max(asset_metrics["width"], opening_width)
                cut_width = facade_width + opening_width_margin
                cut_height = max(1.6, asset_metrics["height"] + opening_height_margin)
                item_offset = storefront_offset
                half_width = facade_width * 0.5 + edge_margin
                required_spacing = max(same_edge_spacing, facade_width + edge_margin * 2.0)
                if edge_s < half_width or edge_s > edge.length - half_width:
                    continue
                if any(abs(edge_s - other) < required_spacing for other in used):
                    continue
                if not edge_front_clearance_ok(edge, edge_s, facade_width, edges, edge_margin):
                    continue
                candidate_facade_xy = edge.start + edge.tangent * edge_s
                road = nearest_road(candidate_facade_xy, roads, synthetic_road_limit)
                road_normal = road.get("normal")
                if road.get("distance") is None or road_normal is None:
                    continue
                facing_dot = float(edge.outward @ road_normal)
                if facing_dot < 0.0:
                    continue
                candidate_final_xy = candidate_facade_xy + edge.outward * item_offset
                if not inside_focus(candidate_final_xy) or overlaps_reserved_opening(candidate_final_xy, facade_width):
                    continue
                used.append(edge_s)
                synthetic_edge_type_counts.setdefault(key, {})[template_asset_type] = (
                    synthetic_edge_type_counts.setdefault(key, {}).get(template_asset_type, 0) + 1
                )
                if selected_asset_key:
                    synthetic_edge_asset_counts.setdefault(key, {})[selected_asset_key] = (
                        synthetic_edge_asset_counts.setdefault(key, {}).get(selected_asset_key, 0) + 1
                    )
                    synthetic_edge_last_asset[key] = selected_asset_key
                synthetic_edge_last_type[key] = template_asset_type
                facade_xy = candidate_facade_xy
                final_xy = candidate_final_xy
                height, ground_road = storefront_ground_height(
                    terrain,
                    roads,
                    final_xy,
                    edge.outward,
                    parse_float(template.get("height")),
                    road_max_distance=max_road_distance + 4.0,
                )
                final_location = [float(final_xy[0]), float(final_xy[1]), height]
                facade_center = [
                    float(final_xy[0] - edge.outward[0] * item_offset),
                    float(final_xy[1] - edge.outward[1] * item_offset),
                    float(height + cut_height * 0.5),
                ]
                record = dict(template)
                record.update(
                    {
                        "index": next_index,
                        "source": "synthetic_facade_fill",
                        "synthetic": True,
                        "asset_path": os.path.relpath(asset_path, work_dir) if asset_path else selected_asset_key,
                        "asset_type": template_asset_type or template.get("asset_type"),
                        "utm_x": float(facade_xy[0]),
                        "utm_y": float(facade_xy[1]),
                        "height": height,
                        "status": "planned",
                        "facade_status": "facade_aligned",
                        "final_location": final_location,
                        "heading": yaw_for_storefront_front(edge.outward),
                        "outward_normal_xy": [float(edge.outward[0]), float(edge.outward[1])],
                        "opening_center": facade_center,
                        "opening_bottom_z": float(height),
                        "opening_width": float(cut_width),
                        "opening_height": float(cut_height),
                        "storefront_facade_width": float(facade_width),
                        "storefront_offset": float(item_offset),
                        "storefront_asset_height": float(asset_metrics["height"]),
                        "storefront_asset_raw_height": float(asset_metrics["raw_height"]),
                        "storefront_asset_raw_width": float(asset_metrics["raw_width"]),
                        "storefront_scale_multiplier": float(asset_metrics["scale_multiplier"]),
                        "storefront_asset_inward_y": float(asset_metrics["inward_y"]),
                        "storefront_asset_y_min": float(asset_metrics["y_min"]),
                        "storefront_asset_y_max": float(asset_metrics["y_max"]),
                        "opening_inside_depth": 1.35,
                        "opening_outside_depth": 0.25,
                        "storefront_ground_source": "road" if ground_road.get("height") is not None else "terrain",
                        "storefront_ground_road_distance": ground_road.get("distance"),
                        "storefront_ground_road_height": ground_road.get("height"),
                        "storefront_edge_margin": float(edge_margin),
                        "building_edge_s": float(edge_s),
                        "building_edge_start_xy": [float(value) for value in edge.start],
                        "building_edge_end_xy": [float(value) for value in edge.end],
                        "road_distance": road.get("distance"),
                        "building_path": os.path.relpath(edge.building_path, work_dir),
                        "building_name": edge.building_name,
                        "building_edge_index": edge.edge_index,
                        "building_edge_length": edge.length,
                        "poi_to_facade_distance": 0.0,
                        "road_facing_dot": facing_dot,
                    }
                )
                records.append(record)
                next_index += 1
                added += 1
                status_counts["facade_aligned"] = status_counts.get("facade_aligned", 0) + 1
        if added:
            result["synthetic_added"] = int(parse_float(result.get("synthetic_added"), 0.0)) + added
            result["synthetic_facade_fill_added"] = added

    result["placements"] = records
    result["planned_count"] = len(records)
    result["facade_alignment"] = {
        "method": "simplified OSM amenity to nearest road-facing building footprint edge",
        "building_edge_count": len(edges),
        "source_building_edge_count": source_edge_count,
        "road_segment_count": len(roads),
        "status_counts": status_counts,
        "max_poi_to_facade_distance": max_poi_to_facade_distance,
        "max_road_distance": max_road_distance,
        "min_edge_length": min_edge_length,
        "min_facing_dot": min_facing_dot,
        "mesh_wall_support_tolerance": mesh_wall_support_tolerance,
        "skipped_unsupported_edge_count": skipped_unsupported_edges,
        "storefront_offset": storefront_offset,
        "same_edge_spacing": same_edge_spacing,
        "opening_width_margin": opening_width_margin,
        "opening_height_margin": opening_height_margin,
        "target_facade_aligned": target_aligned,
        "storefront_target_height": storefront_target_height,
    }
    return result
