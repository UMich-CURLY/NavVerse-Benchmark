#!/usr/bin/env python3
"""Render level street-view QA images from generated VC USD scenes in Isaac Sim."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "navverse_data"
VC_PLUS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VC_PLUS_ROOT))

from terrain_utils import TerrainHeightQuery  # noqa: E402


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
BuildingPolygon = tuple[np.ndarray, np.ndarray, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vc-folder", default="vc_plus")
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("outdoor", "innout"), default="outdoor")
    parser.add_argument("--views-per-city", type=int, default=10)
    parser.add_argument("--view-index", type=int)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--resolution", nargs=2, type=int, default=(960, 540))
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--topdown-height", type=float, default=0.0)
    parser.add_argument("--lookahead", type=float, default=18.0)
    parser.add_argument("--camera-back", type=float, default=4.0)
    parser.add_argument("--lateral-offset", type=float, default=0.8)
    parser.add_argument("--focal-length", type=float, default=22.0)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--quality-retries", type=int, default=2)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--surface-mode", choices=("stage", "terrain", "regions"), default="stage")
    parser.add_argument("--hide-regions", nargs="*", default=())
    parser.add_argument("--terrain-only", action="store_true")
    parser.add_argument("--debug-region-colors", action="store_true")
    parser.add_argument("--authored-camera", action="store_true")
    parser.add_argument("--scene-lighting", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def parse_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def osm_path_for_city(city_dir: Path) -> Path:
    for path in (
        city_dir / "road_data" / "road_data_with_height.osm",
        city_dir / "road_data_with_height.osm",
        city_dir / "road_data" / "road_data.osm",
    ):
        if path.exists():
            return path
    raise FileNotFoundError(f"no OSM road file found in {city_dir}")


def read_road_segments(city_dir: Path) -> list[dict[str, Any]]:
    root = ET.parse(osm_path_for_city(city_dir)).getroot()
    nodes: dict[str, dict[str, float | None]] = {}
    ways: list[dict[str, Any]] = []
    for element in root:
        if element.tag == "node":
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            nodes[element.get("id") or ""] = {
                "x": parse_float(tags.get("utm_x"), None),
                "y": parse_float(tags.get("utm_y"), None),
                "z": parse_float(tags.get("height"), 0.0),
            }
        elif element.tag == "way":
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            if tags.get("highway") not in ROAD_HIGHWAYS or tags.get("area") == "yes":
                continue
            ways.append(
                {
                    "id": element.get("id"),
                    "highway": tags.get("highway"),
                    "nodes": [nd.get("ref") for nd in element.findall("nd")],
                }
            )

    segments: list[dict[str, Any]] = []
    for way in ways:
        refs = way["nodes"]
        for start_ref, end_ref in zip(refs, refs[1:]):
            start = nodes.get(start_ref or "")
            end = nodes.get(end_ref or "")
            if not start or not end:
                continue
            if start["x"] is None or start["y"] is None or end["x"] is None or end["y"] is None:
                continue
            sx, sy = float(start["x"]), float(start["y"])
            ex, ey = float(end["x"]), float(end["y"])
            dx, dy = ex - sx, ey - sy
            length = math.hypot(dx, dy)
            if length < 22.0:
                continue
            segments.append(
                {
                    "way_id": way["id"],
                    "highway": way["highway"],
                    "start": (sx, sy, float(start["z"] or 0.0)),
                    "end": (ex, ey, float(end["z"] or 0.0)),
                    "length": length,
                    "heading": math.atan2(dy, dx),
                    "mid": ((sx + ex) * 0.5, (sy + ey) * 0.5, (float(start["z"] or 0.0) + float(end["z"] or 0.0)) * 0.5),
                }
            )
    return segments


def read_building_polygons(city_dir: Path) -> list[BuildingPolygon]:
    root = ET.parse(osm_path_for_city(city_dir)).getroot()
    nodes: dict[str, np.ndarray] = {}
    for element in root.findall("node"):
        tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
        x = parse_float(tags.get("utm_x"), None)
        y = parse_float(tags.get("utm_y"), None)
        if x is not None and y is not None:
            nodes[element.get("id") or ""] = np.asarray([x, y], dtype=float)

    polygons = []
    for element in root.findall("way"):
        tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
        if tags.get("building") in (None, "no"):
            continue
        points = [nodes[ref] for nd in element.findall("nd") if (ref := nd.get("ref") or "") in nodes]
        if len(points) < 3:
            continue
        polygon = np.asarray(points, dtype=float)
        if np.linalg.norm(polygon[0] - polygon[-1]) <= 1e-6:
            polygon = polygon[:-1]
        if len(polygon) >= 3:
            polygons.append((polygon, polygon.min(axis=0), polygon.max(axis=0)))
    return polygons


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = len(polygon) - 1
    for index in range(len(polygon)):
        xi, yi = polygon[index]
        xj, yj = polygon[previous]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        previous = index
    return inside


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    length_sq = float(np.dot(delta, delta))
    if length_sq <= 1e-9:
        return float(np.linalg.norm(point - start))
    amount = float(np.clip(np.dot(point - start, delta) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + delta * amount)))


def building_clearance(point: np.ndarray, polygons: list[BuildingPolygon]) -> float:
    if not polygons:
        return float("inf")
    bbox_distances = []
    for _polygon, bbox_min, bbox_max in polygons:
        outside = np.maximum(np.maximum(bbox_min - point, point - bbox_max), 0.0)
        bbox_distances.append(float(np.linalg.norm(outside)))
    clearance = float("inf")
    for polygon_index in np.argsort(bbox_distances):
        if bbox_distances[polygon_index] >= clearance:
            break
        polygon = polygons[polygon_index][0]
        if point_in_polygon(point, polygon):
            return -1.0
        clearance = min(
            clearance,
            min(
                point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
                for index in range(len(polygon))
            ),
        )
    return clearance


def sightline_crosses_building(
    start: np.ndarray,
    end: np.ndarray,
    polygons: list[BuildingPolygon],
) -> bool:
    sight_min = np.minimum(start, end) - 0.05
    sight_max = np.maximum(start, end) + 0.05
    candidates = [
        polygon
        for polygon, bbox_min, bbox_max in polygons
        if np.all(bbox_max >= sight_min) and np.all(bbox_min <= sight_max)
    ]
    for amount in np.linspace(0.08, 0.92, 18):
        point = start + (end - start) * float(amount)
        if any(point_in_polygon(point, polygon) for polygon in candidates):
            return True
    return False


def read_scene_objects(city_dir: Path, variant: str) -> list[dict[str, Any]]:
    plan_name = "objaverse_objects.json" if variant == "outdoor" else "innout_objaverse_objects.json"
    return [
        item
        for item in read_json(city_dir / plan_name).get("objects", [])
        if len(item.get("location") or []) >= 3
    ]


def object_clearance(
    point: np.ndarray,
    objects: list[dict[str, Any]],
    excluded: dict[str, Any] | None = None,
) -> float:
    clearances = [
        float(np.linalg.norm(point - np.asarray(item["location"][:2], dtype=float)))
        - max(float(item.get("footprint_radius_m", 0.5)), 0.25)
        for item in objects
        if item is not excluded
    ]
    return min(clearances, default=float("inf"))


def object_sightline_clearance(
    start: np.ndarray,
    end: np.ndarray,
    objects: list[dict[str, Any]],
    excluded: dict[str, Any] | None = None,
) -> float:
    delta = end - start
    length_sq = float(np.dot(delta, delta))
    if length_sq <= 1e-6:
        return float("inf")
    clearance = float("inf")
    for item in objects:
        if item is excluded:
            continue
        center = np.asarray(item["location"][:2], dtype=float)
        radius = max(float(item.get("footprint_radius_m", 0.5)), 0.25)
        amount = float(np.clip(np.dot(center - start, delta) / length_sq, 0.05, 0.95))
        closest = start + amount * delta
        clearance = min(clearance, float(np.linalg.norm(center - closest)) - radius)
    return clearance


def choose_clear_road_eye(
    preferred: np.ndarray,
    forward: np.ndarray,
    side: np.ndarray,
    objects: list[dict[str, Any]],
    building_polygons: list[BuildingPolygon],
) -> np.ndarray:
    candidates = []
    for longitudinal in (0.0, -6.0, 6.0, -12.0, 12.0, -24.0, 24.0):
        for lateral in (0.0, 2.2, -2.2, 4.0, -4.0, 8.0, -8.0):
            point = preferred + forward[:2] * longitudinal + side[:2] * lateral
            object_space = object_clearance(point, objects)
            building_space = building_clearance(point, building_polygons)
            score = (
                object_space >= 1.5,
                building_space >= 0.75,
                min(object_space, 10.0) + min(building_space, 10.0),
                -float(np.linalg.norm(point - preferred)),
            )
            candidates.append((score, point))
    return max(candidates, key=lambda item: item[0])[1]


def select_segments(segments: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not segments:
        raise RuntimeError("no suitable road segments found")
    priority = {"primary": 0, "secondary": 1, "tertiary": 2, "residential": 3, "unclassified": 4, "service": 5}
    ranked = sorted(
        segments,
        key=lambda item: (
            priority.get(str(item["highway"]), 6),
            abs(float(item["mid"][0])) + abs(float(item["mid"][1])),
            -float(item["length"]),
        ),
    )
    pool = ranked[: max(count * 12, count)]
    if len(pool) <= count:
        return pool[:count]
    # Keep a bias toward central major roads but still vary the sampled views.
    selected = []
    for bucket in np.array_split(np.asarray(pool, dtype=object), count):
        selected.append(rng.choice(list(bucket)))
    return selected


def build_views(
    city_dir: Path,
    count: int,
    rng: random.Random,
    camera_height: float,
    target_height: float,
    lookahead: float,
    camera_back: float,
    lateral_offset: float,
    topdown_height: float,
    objects: list[dict[str, Any]],
    building_polygons: list[BuildingPolygon],
) -> list[dict[str, Any]]:
    terrain = TerrainHeightQuery(city_dir)
    selected = select_segments(read_road_segments(city_dir), count, rng)
    views: list[dict[str, Any]] = []
    for index, segment in enumerate(selected, 1):
        mid = np.asarray(segment["mid"], dtype=float)
        heading = float(segment["heading"])
        forward = np.asarray([math.cos(heading), math.sin(heading), 0.0], dtype=float)
        side = np.asarray([-forward[1], forward[0], 0.0], dtype=float)
        if topdown_height > 0.0:
            eye_xy = mid + np.asarray([0.0, -0.1, 0.0], dtype=float)
            target_xy = mid
        else:
            eye_xy = mid - forward * float(camera_back) + side * float(lateral_offset)
            eye_xy[:2] = choose_clear_road_eye(
                eye_xy[:2], forward, side, objects, building_polygons
            )
            target_xy = mid + forward * float(lookahead)
        fallback_z = float(mid[2])
        ground_eye = terrain.height_at(float(eye_xy[0]), float(eye_xy[1]), fallback_z)
        ground_target = terrain.height_at(float(target_xy[0]), float(target_xy[1]), fallback_z)
        eye_z = ground_eye + (float(topdown_height) if topdown_height > 0.0 else float(camera_height))
        target_z = ground_target + float(target_height)
        views.append(
            {
                "name": f"{index:02d}_{segment['highway']}",
                "eye": [float(eye_xy[0]), float(eye_xy[1]), float(eye_z)],
                "target": [float(target_xy[0]), float(target_xy[1]), float(target_z)],
                "way_id": segment["way_id"],
                "highway": segment["highway"],
                "segment_length": float(segment["length"]),
            }
        )
    return views


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def select_spread_objects(
    objects: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = list(objects)
    rng.shuffle(candidates)
    selected: list[dict[str, Any]] = []
    while candidates and len(selected) < count:
        if not selected:
            first_index = max(
                range(len(candidates)),
                key=lambda index: float(candidates[index].get("footprint_radius_m", 0.0)),
            )
            selected.append(candidates.pop(first_index))
            continue
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                min(
                    math.hypot(
                        float(candidates[index]["location"][0]) - float(item["location"][0]),
                        float(candidates[index]["location"][1]) - float(item["location"][1]),
                    )
                    for item in selected
                )
                + 4.0 * float(candidates[index].get("footprint_radius_m", 0.0)),
                candidates[index].get("asset_path") not in {item.get("asset_path") for item in selected},
            ),
        )
        selected.append(candidates.pop(best_index))
    return selected


def road_camera_eye(
    target_xy: np.ndarray,
    segments: list[dict[str, Any]],
    distance: float,
    reverse: bool,
    objects: list[dict[str, Any]],
    target_object: dict[str, Any],
    building_polygons: list[BuildingPolygon],
) -> np.ndarray:
    road_candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for segment in segments:
        start = np.asarray(segment["start"][:2], dtype=float)
        end = np.asarray(segment["end"][:2], dtype=float)
        delta = end - start
        length_sq = float(np.dot(delta, delta))
        if length_sq <= 1e-6:
            continue
        amount = float(np.clip(np.dot(target_xy - start, delta) / length_sq, 0.0, 1.0))
        projection = start + delta * amount
        separation = float(np.linalg.norm(target_xy - projection))
        forward = delta / math.sqrt(length_sq)
        road_candidates.append((separation, projection, forward))
    if not road_candidates:
        raise RuntimeError("no road segment available for object QA camera")

    obstacle_circles = [
        (
            np.asarray(item["location"][:2], dtype=float),
            max(float(item.get("footprint_radius_m", 0.5)), 0.25),
        )
        for item in objects
        if item is not target_object and len(item.get("location") or []) >= 2
    ]

    def obstacle_clearance(point: np.ndarray) -> float:
        if not obstacle_circles:
            return float("inf")
        return min(float(np.linalg.norm(point - center)) - radius for center, radius in obstacle_circles)

    def sightline_clearance(point: np.ndarray) -> float:
        delta = target_xy - point
        length_sq = float(np.dot(delta, delta))
        if length_sq <= 1e-6 or not obstacle_circles:
            return float("inf")
        clearance = float("inf")
        for center, radius in obstacle_circles:
            amount = float(np.clip(np.dot(center - point, delta) / length_sq, 0.0, 1.0))
            closest = point + amount * delta
            clearance = min(clearance, float(np.linalg.norm(center - closest)) - radius)
        return clearance

    preferred_direction = 1.0 if reverse else -1.0
    target_radius = max(float(target_object.get("footprint_radius_m", 0.5)), 0.5)
    ranked: list[
        tuple[tuple[bool, bool, bool, bool, bool, float, float, float], np.ndarray]
    ] = []

    def rank_eye(eye: np.ndarray, separation: float, distance_scale: float) -> None:
        camera_clearance = obstacle_clearance(eye)
        target_clearance = float(np.linalg.norm(eye - target_xy)) - target_radius
        line_clearance = sightline_clearance(eye)
        building_space = building_clearance(eye, building_polygons)
        building_sightline = sightline_crosses_building(eye, target_xy, building_polygons)
        score = (
            camera_clearance >= 1.5,
            target_clearance >= 1.5,
            building_space >= 0.75,
            not building_sightline,
            line_clearance >= 0.35,
            -separation,
            min(camera_clearance, 8.0)
            + min(target_clearance, 8.0)
            + min(building_space, 8.0)
            + min(line_clearance, 8.0),
            -abs(distance_scale - 1.0),
        )
        ranked.append((score, eye))

    for separation, projection, forward in sorted(road_candidates, key=lambda item: item[0])[:12]:
        side = np.asarray([-forward[1], forward[0]], dtype=float)
        for direction in (preferred_direction, -preferred_direction):
            for side_scale in (0.6, 2.0, -2.0):
                for distance_scale in (1.0, 1.35, 1.7, 2.2):
                    eye = (
                        projection
                        + forward * distance * distance_scale * direction
                        + side * (side_scale * direction)
                    )
                    rank_eye(eye, separation, distance_scale)

    for angle in np.linspace(0.0, math.tau, 16, endpoint=False):
        direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        for distance_scale in (1.0, 1.5, 2.0, 2.8):
            eye = target_xy + direction * distance * distance_scale
            road_distance = min(
                float(np.linalg.norm(eye - projection))
                for _separation, projection, _forward in road_candidates
            )
            rank_eye(eye, road_distance, distance_scale)
    return max(ranked, key=lambda item: item[0])[1]


def build_object_views(
    city_dir: Path,
    variant: str,
    category: str,
    count: int,
    rng: random.Random,
    terrain: TerrainHeightQuery,
    camera_height: float,
    objects: list[dict[str, Any]],
    building_polygons: list[BuildingPolygon],
) -> list[dict[str, Any]]:
    targets = [item for item in objects if item.get("category") == category]
    segments = read_road_segments(city_dir)
    views = []
    for index, item in enumerate(select_spread_objects(targets, count, rng), 1):
        location = np.asarray(item["location"], dtype=float)
        radius = max(float(item.get("footprint_radius_m", 0.5)), 0.5)
        distance = radius * 2.0 + (5.0 if category == "vehicle" else 3.0)
        eye_xy = road_camera_eye(
            location[:2],
            segments,
            distance,
            reverse=index % 2 == 0,
            objects=objects,
            target_object=item,
            building_polygons=building_polygons,
        )
        eye_z = terrain.height_at(float(eye_xy[0]), float(eye_xy[1]), float(location[2])) + camera_height
        views.append(
            {
                "name": f"{category}_{index:02d}",
                "eye": [float(eye_xy[0]), float(eye_xy[1]), float(eye_z)],
                "target": [float(location[0]), float(location[1]), float(location[2] + min(radius, 1.2))],
                "target_location": location.tolist(),
                "target_asset": item.get("asset_path"),
                "target_category": category,
                "camera_object_clearance_m": object_clearance(eye_xy, objects, excluded=item),
                "camera_building_clearance_m": building_clearance(eye_xy, building_polygons),
                "camera_object_sightline_clearance_m": object_sightline_clearance(
                    eye_xy,
                    location[:2],
                    objects,
                    excluded=item,
                ),
                "camera_building_sightline_blocked": sightline_crosses_building(
                    eye_xy,
                    location[:2],
                    building_polygons,
                ),
            }
        )
    return views


def build_innout_door_views(
    city_dir: Path,
    terrain: TerrainHeightQuery,
    camera_height: float,
    objects: list[dict[str, Any]],
    building_polygons: list[BuildingPolygon],
) -> list[dict[str, Any]]:
    plan = read_json(city_dir / "innout_final_plan.json")
    if not plan:
        return []
    center = np.asarray(
        [plan["door_center_xy"][0], plan["door_center_xy"][1], plan["door_bottom_z"]],
        dtype=float,
    )
    outward = np.asarray([*plan["outward_normal_xy"], 0.0], dtype=float)
    outward /= np.linalg.norm(outward)
    tangent = np.asarray([*plan["tangent_xy"], 0.0], dtype=float)
    tangent /= np.linalg.norm(tangent)

    room_eye = center - outward * 3.0 + np.asarray([0.0, 0.0, camera_height])
    interior_door = center - outward * 0.6
    bbox_min = plan.get("interior_scene_bbox_min")
    bbox_max = plan.get("interior_scene_bbox_max")
    room_translate = plan.get("room_translate")
    local_door = plan.get("interior_door_center")
    if bbox_min and bbox_max and room_translate and local_door:
        yaw = math.radians(float(plan.get("room_yaw_deg", 0.0)))
        scale = float(plan.get("room_effective_scale", plan.get("room_scale", 1.0)))
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=float,
        )
        local_center = (np.asarray(bbox_min[:2], dtype=float) + np.asarray(bbox_max[:2], dtype=float)) * 0.5
        room_eye[:2] = np.asarray(room_translate[:2], dtype=float) + rotation @ (local_center * scale)
        room_eye[2] = float(plan["door_bottom_z"]) + camera_height
        interior_door[:2] = np.asarray(room_translate[:2], dtype=float) + rotation @ (
            np.asarray(local_door[:2], dtype=float) * scale
        )

    def outside_eye(distance: float, lateral: float) -> list[float]:
        preferred = center + outward * distance + tangent * lateral
        candidates = []
        for extra_distance in (0.0, 3.0, 6.0):
            for lateral_shift in (0.0, 3.0, -3.0, 6.0, -6.0):
                point = preferred + outward * extra_distance + tangent * lateral_shift
                object_space = object_clearance(point[:2], objects)
                line_space = object_sightline_clearance(point[:2], center[:2], objects)
                building_space = building_clearance(point[:2], building_polygons)
                score = (
                    object_space >= 1.5,
                    building_space >= 0.75,
                    line_space >= 0.35,
                    min(object_space, 10.0)
                    + min(building_space, 10.0)
                    + min(line_space, 10.0),
                    -float(np.linalg.norm(point - preferred)),
                )
                candidates.append((score, point))
        point = max(candidates, key=lambda item: item[0])[1]
        point[2] = terrain.height_at(float(point[0]), float(point[1]), float(center[2])) + camera_height
        return point.tolist()

    target = (center - outward * 0.7 + np.asarray([0.0, 0.0, 1.3])).tolist()
    return [
        {
            "name": "door_outside_direct",
            "eye": outside_eye(8.0, 0.0),
            "target": target,
            "target_category": "innout_door",
        },
        {
            "name": "door_outside_oblique",
            "eye": outside_eye(6.0, 6.0),
            "target": target,
            "target_category": "innout_door",
        },
        {
            "name": "door_inside_out",
            "eye": room_eye.tolist(),
            "target": (interior_door + outward * 2.0 + np.asarray([0.0, 0.0, 1.3])).tolist(),
            "target_category": "innout_door",
        },
    ]


def build_quality_views(
    city_dir: Path,
    variant: str,
    count: int,
    rng: random.Random,
    camera_height: float,
    lookahead: float,
    camera_back: float,
    lateral_offset: float,
    topdown_height: float,
) -> list[dict[str, Any]]:
    if count < 10:
        raise ValueError("quality QA requires at least 10 views per scene")
    terrain = TerrainHeightQuery(city_dir)
    objects = read_scene_objects(city_dir, variant)
    building_polygons = read_building_polygons(city_dir)
    target_views = []
    if variant == "innout":
        target_views.extend(
            build_innout_door_views(
                city_dir,
                terrain,
                camera_height,
                objects,
                building_polygons,
            )
        )
    target_views.extend(
        build_object_views(
            city_dir,
            variant,
            "vehicle",
            2 if variant == "innout" else 3,
            rng,
            terrain,
            camera_height,
            objects,
            building_polygons,
        )
    )
    target_views.extend(
        build_object_views(
            city_dir,
            variant,
            "prop",
            2 if variant == "innout" else 3,
            rng,
            terrain,
            camera_height,
            objects,
            building_polygons,
        )
    )
    road_count = max(1, count - len(target_views))
    road_views = build_views(
        city_dir,
        max(road_count, count),
        rng,
        camera_height,
        camera_height,
        lookahead,
        camera_back,
        lateral_offset,
        topdown_height,
        objects,
        building_polygons,
    )
    for index, view in enumerate(road_views, 1):
        view["name"] = f"road_{index:02d}_{view['highway']}"
        view["target_category"] = "road"
    return (road_views[:road_count] + target_views)[:count]


async def attach_stage(stage_to_attach):
    import omni.usd

    return await omni.usd.get_context().attach_stage_async(stage_to_attach)


def wait_for_stage_load() -> None:
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    for _ in range(900):
        app.update()
        if omni.usd.get_context().get_stage_loading_status()[2] == 0:
            return
    raise TimeoutError("USD stage did not finish loading")


def camera_matrix(position: np.ndarray, target: np.ndarray):
    from isaacsim.core.utils.rotations import lookat_to_quatf
    from pxr import Gf

    quat = lookat_to_quatf(
        Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])),
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    matrix = Gf.Matrix4d().SetRotateOnly(quat)
    matrix.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    return matrix


def render_rgb(app, rgb_annotator, warmup_frames: int):
    import omni.replicator.core as rep

    app.update()
    rep.orchestrator.step(rt_subframes=max(1, warmup_frames), delta_time=0.0)
    rgb = rgb_annotator.get_data()
    if hasattr(rgb, "numpy"):
        rgb = rgb.numpy()
    if isinstance(rgb, dict):
        rgb = rgb.get("data")
    if rgb is None or rgb.size == 0:
        return None
    return rgb[:, :, :3].astype(np.uint8)


def add_lighting(stage) -> Any:
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom, UsdLux

    rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", 1.35)
    dome = UsdLux.DomeLight.Define(stage, "/World/IsaacStreetGridDome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/IsaacStreetGridSun")
    sun.CreateIntensityAttr(2800.0)
    sun.CreateAngleAttr(0.5)
    headlight = UsdLux.SphereLight.Define(stage, "/World/IsaacStreetGridHeadlight")
    headlight.CreateIntensityAttr(2600.0)
    headlight.CreateRadiusAttr(5.0)
    headlight_xform = UsdGeom.Xformable(headlight.GetPrim())
    headlight_xform.ClearXformOpOrder()
    return headlight_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)


def set_surface_mode(stage, mode: str, hidden_regions: set[str], terrain_only: bool, debug_colors: bool) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    if terrain_only:
        world = stage.GetPrimAtPath("/World")
        for child in world.GetChildren():
            if child.GetName() != "TerrainRegions" and child.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(child).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    prim = stage.GetPrimAtPath("/World/TerrainRegions")
    if not prim or not prim.IsValid() or mode == "stage":
        return
    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
    for child in prim.GetChildren():
        visible = child.GetName() == "Terrain" if mode == "terrain" else child.GetName() != "Terrain"
        if child.GetName() in hidden_regions:
            visible = False
        UsdGeom.Imageable(child).CreateVisibilityAttr(
            UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
        )
        if not visible or not debug_colors or child.GetTypeName() != "Mesh":
            continue
        color = {
            "water": (0.05, 0.35, 0.95),
            "green": (0.08, 0.65, 0.18),
            "crosswalk": (0.95, 0.85, 0.12),
            "curb": (0.95, 0.12, 0.1),
            "sidewalk": (0.15, 0.75, 0.9),
            "road": (0.12, 0.12, 0.12),
            "Terrain": (0.45, 0.45, 0.45),
        }.get(child.GetName(), (0.7, 0.7, 0.7))
        material_path = f"/World/IsaacTerrainRegionDebugMaterials/{child.GetName()}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(child).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )


def write_montage(city: str, image_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
    tile_w, tile_h = images[0].size
    label_h = 34
    columns = min(5, len(images))
    rows = math.ceil(len(images) / columns)
    montage = Image.new("RGB", (tile_w * columns, (tile_h + label_h) * rows), (18, 18, 18))
    draw = ImageDraw.Draw(montage)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for index, image in enumerate(images):
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + label_h)
        montage.paste(image, (x, y + label_h))
        label = image_paths[index].stem.removeprefix(f"{city}_")
        draw.text((x + 12, y + 7), f"{city} {label}", fill=(245, 245, 245), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)


def image_quality_metrics(rgb: np.ndarray) -> dict[str, Any]:
    luminance = rgb.astype(np.float32).mean(axis=2)
    mean = float(luminance.mean())
    std = float(luminance.std())
    dark_fraction = float((luminance < 8.0).mean())
    bright_fraction = float((luminance > 247.0).mean())
    issues = []
    if std < 4.0:
        issues.append("low_contrast")
    if dark_fraction > 0.65:
        issues.append("mostly_black")
    if bright_fraction > 0.95:
        issues.append("mostly_white")
    return {
        "mean_luminance": mean,
        "luminance_std": std,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "issues": issues,
        "passed": not issues,
    }


def setup_render_camera(stage, args: argparse.Namespace, views: list[dict[str, Any]]):
    from pxr import Gf, Usd, UsdGeom

    headlight_translate = None if args.scene_lighting else add_lighting(stage)
    if args.authored_camera:
        camera_path = "/World/DefaultCamera"
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"missing authored camera: {camera_path}")
        matrix = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        eye = np.asarray(matrix.ExtractTranslation(), dtype=float)
        return camera_path, None, headlight_translate, [
            {"name": "authored_default", "eye": eye.tolist(), "target": None}
        ]

    camera_path = "/World/IsaacStreetGridCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(float(args.focal_length))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 6000.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    transform = camera_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    return camera_path, transform, headlight_translate, views


def place_render_camera(transform, headlight_translate, view: dict[str, Any]) -> np.ndarray:
    from pxr import Gf

    eye = np.asarray(view["eye"], dtype=float)
    if transform is not None:
        transform.Set(camera_matrix(eye, np.asarray(view["target"], dtype=float)))
    if headlight_translate is not None:
        headlight_translate.Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2] + 0.2)))
    return eye


def render_city(args: argparse.Namespace, city: str, rng: random.Random) -> dict[str, Any]:
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd

    city_dir = (args.data_root / args.vc_folder / city).resolve()
    usd_path = city_dir / f"{city}_{args.variant}.usd"
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)
    city_out = (args.output_dir / city).resolve()
    city_out.mkdir(parents=True, exist_ok=True)

    views = build_quality_views(
        city_dir,
        args.variant,
        int(args.views_per_city),
        rng,
        float(args.camera_height),
        float(args.lookahead),
        float(args.camera_back),
        float(args.lateral_offset),
        float(args.topdown_height),
    )
    if args.view_index is not None:
        if not 0 <= args.view_index < len(views):
            raise ValueError(f"view index {args.view_index} is outside 0..{len(views) - 1}")
        views = [views[args.view_index]]

    print(f"[isaac-street-grid] opening {usd_path}", flush=True)
    stage_to_attach = Usd.Stage.Open(str(usd_path))
    if stage_to_attach is None:
        raise RuntimeError(f"failed to open USD stage: {usd_path}")
    success, _error = asyncio.get_event_loop().run_until_complete(attach_stage(stage_to_attach))
    if not success:
        raise RuntimeError(f"failed to attach USD stage: {usd_path}")
    wait_for_stage_load()
    stage = omni.usd.get_context().get_stage()
    set_surface_mode(
        stage,
        args.surface_mode,
        set(args.hide_regions),
        bool(args.terrain_only),
        bool(args.debug_region_colors),
    )
    camera_path, transform, headlight_translate, views = setup_render_camera(stage, args, views)

    render_product = rep.create.render_product(camera_path, tuple(args.resolution), name=f"StreetGrid_{city}")
    rgb_annotator = rep.annotators.get("rgb")
    rgb_annotator.attach([render_product])
    app = omni.kit.app.get_app()

    renders = []
    paths: list[Path] = []
    for view in views:
        place_render_camera(transform, headlight_translate, view)
        rgb = None
        quality = None
        for attempt in range(max(0, int(args.quality_retries)) + 1):
            warmup_frames = int(args.warmup_frames) * (attempt + 1)
            rgb = render_rgb(app, rgb_annotator, warmup_frames)
            if rgb is None:
                continue
            quality = image_quality_metrics(rgb)
            if quality["passed"]:
                break
            print(
                f"[isaac-street-grid] retrying {city} {view['name']} "
                f"after {quality['issues']}",
                flush=True,
            )
        if rgb is None or quality is None:
            raise RuntimeError(f"empty render for {city} {view['name']}")
        image_path = city_out / f"{city}_{view['name']}.png"
        Image.fromarray(rgb).save(image_path)
        paths.append(image_path)
        renders.append({**view, "path": str(image_path), "quality": quality})
        print(f"[isaac-street-grid] wrote {image_path}", flush=True)

    montage_path = args.output_dir / f"{city}_{args.variant}_street_grid.png"
    write_montage(city, paths, montage_path)
    return {"city": city, "usd": str(usd_path), "renders": renders, "montage": str(montage_path)}


def render_all(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    summaries = []
    for city in args.cities:
        summaries.append(render_city(args, city, rng))
    summary = {
        "cities": args.cities,
        "variant": args.variant,
        "views_per_city": int(args.views_per_city),
        "camera_height": float(args.camera_height),
        "quality_retries": int(args.quality_retries),
        "resolution": list(args.resolution),
        "surface_mode": args.surface_mode,
        "authored_camera": bool(args.authored_camera),
        "scene_lighting": bool(args.scene_lighting),
        "summaries": summaries,
    }
    (args.output_dir / "street_grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    print("[isaac-street-grid] launching Isaac app", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
            "width": int(args.resolution[0]),
            "height": int(args.resolution[1]),
            "multi_gpu": bool(args.multi_gpu),
            "samples_per_pixel_per_frame": 1,
            "anti_aliasing": 0,
        }
    )
    try:
        render_all(args)
    except BaseException:
        print("[isaac-street-grid] failed:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        print("[isaac-street-grid] closing app", flush=True)
        app.close()


if __name__ == "__main__":
    main()
