#!/usr/bin/env python3
"""Generate VC Full outdoor/innout USD variants with quality manifests.

This script is the clean orchestration entrypoint for the VC Full pipeline. It keeps
legacy geometry behavior delegated to the Blender scripts while making dataset
preparation, point sampling, variant selection, validation, and VC+ comparison
explicit and reproducible.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from objaverse_asset_rules import (
    OBJAVERSE_VEHICLE_ASSETS,
    objaverse_asset_enabled,
    objaverse_outdoor_group_enabled,
    objaverse_outdoor_ungrouped_enabled,
    objaverse_asset_scale,
)
from storefront_facade_planner import align_storefront_plan
from terrain_utils import TerrainHeightQuery
from canonical_city_textures import (
    copy_canonical_texture_files,
    load_canonical_texture_map,
    rewrite_city_texture_refs,
)


CITY_EXCLUDES = {"generated", "objects", "road_objects", "storefront", "__pycache__"}
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = DEFAULT_REPO_ROOT / "navverse_data"
DEFAULT_WORK_ROOT = DEFAULT_DATA_ROOT / "vc_full_generated"
DEFAULT_LOG_ROOT = DEFAULT_WORK_ROOT / "logs"
DEFAULT_STORE_ROOT = DEFAULT_DATA_ROOT / "storefront"
DEFAULT_GRSCENE_ROOT = DEFAULT_DATA_ROOT / "grscenes_commercial"
DEFAULT_OBJAVERSE_ROOT = DEFAULT_DATA_ROOT / "objaverse"
DEFAULT_QA_ROOT = DEFAULT_DATA_ROOT / "vc_qa"
DEFAULT_SCENE_DATA_ROOT = DEFAULT_DATA_ROOT.parent / "scene_data"

COMMERCIAL_AMENITIES = [
    "restaurant", "cafe", "fast_food", "parking_entrance", "bar", "bank", "pub", "atm",
    "bicycle_rental", "parking", "pharmacy", "toilets", "theatre", "library", "dentist",
    "school", "post_office", "bureau_de_change", "doctors", "bicycle_repair_station",
    "clinic", "community_centre", "car_rental", "police", "arts_centre", "cinema",
    "kindergarten", "university", "coworking_space", "college", "convenience_store",
    "food_court", "lab",
]

SYNTHETIC_STOREFRONT_TYPES = [
    "restaurant", "cafe", "fast_food", "convenience_store", "bank",
    "pharmacy", "bar", "library", "theatre",
]

REAL_STOREFRONT_ASSET_TYPES = {
    "restaurant", "cafe", "fast_food", "convenience_store", "bank",
    "pharmacy", "bar", "library", "theatre", "cinema", "post_office",
}

STOREFRONT_RENDER_REJECT_REASONS = {
    "empty_rgb",
    "missing_target_store_prim",
    "low_store_visual_signal",
    "not_close_to_street",
    "target_not_visible",
    "target_too_small_or_clipped",
}
MAX_STOREFRONT_RENDER_REPAIR_ITERATIONS = 3

AMENITY_ASSET_ALIASES = {
    "pub": "bar",
    "car_sharing": "car_rental",
    "college": "lab",
}

EXCLUDED_LANE_HIGHWAYS = {
    "footway", "cycleway", "path", "steps", "pedestrian", "track", "bridleway", "corridor",
}

ROAD_OBJECT_ASSETS = {
    "crossing": "nyc_traffic_light.usd",
    "traffic_signals": "nyc_traffic_light.usd",
    "bus_stop": "Bus_stop_sign.usd",
    "stop": "Only_straight_road_sign.usd",
}

DEFAULT_SIDEWALK_WIDTH = 1.8
VEHICLE_SAME_ASSET_MIN_DISTANCE = 10.0
VEHICLE_FOOTPRINT_CLEARANCE = 2.5
PROP_FOOTPRINT_CLEARANCE = 0.75
PROP_GROUP_MIN_DISTANCE = 8.0
PROP_GROUP_REPEAT_MIN_DISTANCE = 10.0
PROP_GROUP_SIDEWALK_RADIUS = 3.0
OBJAVERSE_TARGET_PROPS = 600
OBJAVERSE_TARGET_VEHICLES = 200
INNOUT_DEFAULT_RADIUS_M = 50.0
INNOUT_DEFAULT_STOREFRONTS = 12
INNOUT_STOREFRONT_OPENING_RESERVE = 2
INNOUT_DEFAULT_PROPS = 120
INNOUT_DEFAULT_VEHICLES = 30
INNOUT_DOORWAY_LATERAL_CLEARANCE_M = 1.5
INNOUT_DOORWAY_INWARD_CLEARANCE_M = 1.0
INNOUT_DOORWAY_OUTWARD_CLEARANCE_M = 10.0
INNOUT_ROOM_OBJECT_CLEARANCE_M = 0.5
TRAFFIC_SIGNAL_CURB_OFFSET = 0.8
TRAFFIC_SIGNAL_FAR_SIDE_OFFSET = 4.0
TRAFFIC_SIGNAL_CLUSTER_DISTANCE = 18.0
TRAFFIC_SIGNAL_APPROACH_RADIUS = 30.0
TRAFFIC_SIGNAL_APPROACH_ANGLE_DEDUP_DEG = 32.0
_OBJECT_DIM_CACHE: dict[str, list[float]] = {}

OBJAVERSE_CATEGORY_ASSETS = {
    "vehicle": ["garbage_truck.usd", "van.usd", "electric_car.usd", "bus.usd", "ambulance.usd", "truck.usd"],
    "street": ["bicycle.usd", "bench.usd", "trash_can.usd", "umbrella.usd", "food_truck.usd"],
    "sidewalk": ["bicycle.usd", "bench.usd", "trash_can.usd", "umbrella.usd", "backpack.usd"],
}

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_width(value: Any) -> float | None:
    if not value:
        return None
    text = str(value).strip().lower()
    try:
        if text.endswith("m"):
            return float(text[:-1].strip())
        if "ft" in text:
            return float(text.replace("ft", "").strip()) * 0.3048
        if "'" in text:
            return float(text.replace("'", "").strip()) * 0.3048
        return float(text)
    except (TypeError, ValueError):
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def point_segment_distance_xy(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    edge = end - start
    denominator = float(edge @ edge)
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = max(0.0, min(1.0, float((point - start) @ edge) / denominator))
    return float(np.linalg.norm(point - (start + t * edge)))


def point_in_polygon_xy(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    previous = polygon[-1]
    for current in polygon:
        if (float(current[1]) > y) != (float(previous[1]) > y):
            crossing_x = float(previous[0]) + (y - float(previous[1])) * (
                float(current[0]) - float(previous[0])
            ) / (float(current[1]) - float(previous[1]))
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def innout_room_footprint(anchor: dict[str, Any]) -> list[list[float]]:
    local_min = np.asarray(anchor["interior_scene_bbox_min"], dtype=float)[:2]
    local_max = np.asarray(anchor["interior_scene_bbox_max"], dtype=float)[:2]
    corners = np.asarray(
        [
            [local_min[0], local_min[1]],
            [local_max[0], local_min[1]],
            [local_max[0], local_max[1]],
            [local_min[0], local_max[1]],
        ],
        dtype=float,
    )
    yaw = math.radians(float(anchor["room_yaw_deg"]))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    footprint = (
        corners * float(anchor["room_effective_scale"])
    ) @ rotation.T + np.asarray(anchor["room_translate"], dtype=float)[:2]
    return [[float(value) for value in point] for point in footprint]


def usd_file_error(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size <= 0:
        return "empty"
    try:
        from pxr import Sdf
    except ImportError:
        return None
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        return "PXR could not open layer"
    return None


def require_valid_usd(path: Path) -> None:
    error = usd_file_error(path)
    if error:
        raise RuntimeError(f"invalid USD output ({error}): {path}")


def usd_has_active_prims(path: Path, prim_paths: list[str]) -> bool:
    try:
        from pxr import Usd
    except ImportError:
        return True
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if stage is None:
        return False
    return all((prim := stage.GetPrimAtPath(prim_path)) and prim.IsActive() for prim_path in prim_paths)


def run_logged(cmd: list[str], log_path: Path, dry_run: bool = False, cwd: Path | None = None) -> dict[str, Any]:
    record = {"cmd": cmd, "log": str(log_path), "dry_run": dry_run, "started_at": time.strftime("%F %T %Z")}
    if dry_run:
        record.update({"returncode": 0, "skipped": True, "finished_at": time.strftime("%F %T %Z")})
        return record
    ensure_dir(log_path.parent)
    env = None
    if cmd and Path(cmd[0]).name == "blender":
        env = os.environ.copy()
        for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CONDA_PREFIX"):
            env.pop(key, None)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=lambda: os.umask(0o002),
        )
    record.update({"returncode": proc.returncode, "finished_at": time.strftime("%F %T %Z")})
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}; see {log_path}")
    return record


@dataclass
class RunConfig:
    repo_root: Path
    data_root: Path
    source_dataset: str
    work_root: Path
    store_root: Path
    grscene_root: Path
    objaverse_root: Path
    vc_plus_root: Path
    log_root: Path
    variant: str
    stage: str
    cities: list[str]
    assignment_cities: list[str]
    seed: int
    lane_points: int
    sidewalk_points: int
    dry_run: bool
    overwrite: bool
    skip_existing: bool
    blender: str
    usd_python: str
    compare_vc_plus: bool
    max_storefronts: int | None
    min_storefronts: int
    max_building_scan: int
    texture_material_op: str
    texture_variant: str
    texture_scene_root: Path | None
    city_texture_source_root: Path | None
    compressonator: str
    skip_texture_optimization: bool
    qa_root: Path
    skip_qa_render: bool
    scene_data_root: Path
    skip_scene_photo_render: bool
    scene_photo_workers: int
    scene_photo_engine: str
    write_cut_smoke: bool
    lower_osm_roads: bool
    road_lower_depth: float
    road_lower_subdivide_cuts: int
    flat_ground: bool
    flat_ground_height_scale: float
    flat_ground_reference: str
    flat_ground_copy_height_field_to_source: bool
    flat_ground_road_boundary_cuts: bool
    storefront_target_height: float
    innout_radius_m: float
    innout_storefronts: int
    innout_props: int
    innout_vehicles: int

    @property
    def source_root(self) -> Path:
        return self.data_root / self.source_dataset

    @property
    def tools_dir(self) -> Path:
        return self.repo_root / "navverse_tools" / "vc_plus"

    @property
    def pipeline_tools_dir(self) -> Path:
        return self.tools_dir / "pipeline"

    @property
    def render_tools_dir(self) -> Path:
        return self.tools_dir / "render"

    @property
    def texture_tools_dir(self) -> Path:
        return self.repo_root / "navverse_tools" / "texture_downsampling"

    @property
    def run_manifest(self) -> Path:
        return self.work_root / "manifests" / f"run_{self.variant}_{now_stamp()}_{os.getpid()}.json"


@dataclass
class CityPaths:
    city: str
    original_source_dir: Path
    source_dir: Path
    work_dir: Path
    osm: Path
    raw_usd: Path
    normalized_usd: Path
    outdoor_usd: Path
    innout_base_usd: Path
    innout_usd: Path
    lane_points: Path
    sidewalk_points: Path
    road_objects: Path
    objaverse_objects: Path
    store_manifest: Path
    innout_lane_points: Path
    innout_sidewalk_points: Path
    innout_road_objects: Path
    innout_objaverse_objects: Path
    innout_store_manifest: Path
    innout_anchor_plan: Path
    city_manifest: Path


@dataclass
class CitySummary:
    city: str
    source_exists: bool
    buildings: int = 0
    has_roof: bool = False
    has_terrain: bool = False
    has_objects_json: bool = False
    has_osm: bool = False
    amenity_total: int = 0
    commercial_total: int = 0
    highway_ways: int = 0
    outputs: dict[str, str] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)


class OsmIndex:
    def __init__(self, osm_path: Path) -> None:
        self.osm_path = osm_path
        self.nodes: dict[str, dict[str, Any]] = {}
        self.ways: list[dict[str, Any]] = []
        self.amenities: list[dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        tree = ET.parse(self.osm_path)
        root = tree.getroot()
        for element in root:
            if element.tag != "node":
                continue
            node = {
                "id": element.get("id"),
                "lat": parse_float(element.get("lat"), 0.0),
                "lon": parse_float(element.get("lon"), 0.0),
                "utm_x": 0.0,
                "utm_y": 0.0,
                "height": 0.0,
                "tags": {},
            }
            for tag in element.findall("tag"):
                key, value = tag.get("k"), tag.get("v")
                node["tags"][key] = value
                if key == "utm_x":
                    node["utm_x"] = parse_float(value, 0.0)
                elif key == "utm_y":
                    node["utm_y"] = parse_float(value, 0.0)
                elif key == "height":
                    node["height"] = parse_float(value, 0.0)
            self.nodes[node["id"]] = node
            amenity = node["tags"].get("amenity")
            if amenity:
                self.amenities.append({**node, "amenity": amenity})
        for element in root:
            if element.tag != "way":
                continue
            way = {"id": element.get("id"), "nodes": [nd.get("ref") for nd in element.findall("nd")], "tags": {}}
            for tag in element.findall("tag"):
                way["tags"][tag.get("k")] = tag.get("v")
            self.ways.append(way)
        self._loaded = True

    def summary(self) -> dict[str, int]:
        self.load()
        commercial = {a for a in COMMERCIAL_AMENITIES}
        return {
            "amenity_total": len(self.amenities),
            "commercial_total": sum(1 for item in self.amenities if item["amenity"] in commercial),
            "highway_ways": sum(1 for way in self.ways if "highway" in way["tags"]),
        }

    def commercial_points(self) -> list[dict[str, Any]]:
        self.load()
        allowed = set(COMMERCIAL_AMENITIES)
        return [item for item in self.amenities if item["amenity"] in allowed]


class PointSampler:
    @staticmethod
    def interpolate_point(
        start: dict[str, Any],
        end: dict[str, Any],
        t: float,
    ) -> dict[str, Any]:
        point = dict(start)
        for key in ("utm_x", "utm_y", "height", "lat", "lon"):
            if start.get(key) is None or end.get(key) is None:
                continue
            point[key] = float(start[key]) + t * (float(end[key]) - float(start[key]))
        return point

    @staticmethod
    def is_lane_way(tags: dict[str, str]) -> bool:
        highway = tags.get("highway")
        if not highway or highway in EXCLUDED_LANE_HIGHWAYS:
            return False
        return tags.get("motor_vehicle") != "no" and tags.get("vehicle") != "no"

    @staticmethod
    def lane_width(tags: dict[str, str]) -> float:
        width = parse_width(tags.get("width"))
        if width and width > 0:
            return width
        lanes = None
        try:
            lanes = int(tags.get("lanes")) if tags.get("lanes") else None
        except (TypeError, ValueError):
            lanes = None
        highway = tags.get("highway")
        lane_width = 3.5
        if highway in {"residential", "service"}:
            lane_width = 3.0
        elif highway in {"motorway", "trunk"}:
            lane_width = 3.75
        return lanes * lane_width if lanes and lanes > 0 else lane_width

    @staticmethod
    def is_sidewalk_way(tags: dict[str, str]) -> bool:
        return (tags.get("highway") == "footway" and tags.get("footway") != "crossing") or tags.get("footway") == "sidewalk"

    @staticmethod
    def is_explicit_sidewalk_way(tags: dict[str, str]) -> bool:
        return tags.get("footway") == "sidewalk"

    @staticmethod
    def sidewalk_width(tags: dict[str, str], include_generic_width: bool = True) -> float:
        width = None
        keys = ["sidewalk:left:width", "sidewalk:right:width", "footway:width", "width:sidewalk"]
        if include_generic_width:
            keys.append("width")
        for key in keys:
            if width is not None:
                break
            width = parse_width(tags.get(key))
        return width if width and width > 0 else DEFAULT_SIDEWALK_WIDTH

    @staticmethod
    def sidewalk_sides(tags: dict[str, str]) -> list[float]:
        sidewalk = (tags.get("sidewalk") or "").lower()
        sides: list[float] = []
        if sidewalk in {"both", "yes"}:
            sides = [1.0, -1.0]
        elif sidewalk in {"left"}:
            sides = [1.0]
        elif sidewalk in {"right"}:
            sides = [-1.0]
        if (tags.get("sidewalk:left") or "").lower() in {"yes", "designated"} and 1.0 not in sides:
            sides.append(1.0)
        if (tags.get("sidewalk:right") or "").lower() in {"yes", "designated"} and -1.0 not in sides:
            sides.append(-1.0)
        return sides

    @staticmethod
    def build_segments(
        osm: OsmIndex,
        kind: str,
        *,
        explicit_sidewalk_only: bool = False,
    ) -> list[dict[str, Any]]:
        osm.load()
        segments = []
        for way in osm.ways:
            tags = way["tags"]
            sidewalk_semantic = None
            if kind == "lane":
                if not PointSampler.is_lane_way(tags):
                    continue
                width = PointSampler.lane_width(tags)
            else:
                is_explicit_path = PointSampler.is_explicit_sidewalk_way(tags)
                if PointSampler.is_sidewalk_way(tags) and (is_explicit_path or not explicit_sidewalk_only):
                    width = PointSampler.sidewalk_width(tags)
                    side_offsets = [0.0]
                    sidewalk_semantic = "footway=sidewalk" if is_explicit_path else "generic_footway"
                elif PointSampler.is_lane_way(tags):
                    sides = PointSampler.sidewalk_sides(tags)
                    if not sides:
                        continue
                    width = PointSampler.sidewalk_width(tags, include_generic_width=False)
                    road_width = PointSampler.lane_width(tags)
                    side_offsets = [side * (road_width / 2.0 + width / 2.0) for side in sides]
                    sidewalk_semantic = "road_sidewalk_tag"
                else:
                    continue
                highway = "sidewalk"
            refs = way["nodes"]
            for idx in range(len(refs) - 1):
                start, end = osm.nodes.get(refs[idx]), osm.nodes.get(refs[idx + 1])
                if not start or not end:
                    continue
                dx = end["utm_x"] - start["utm_x"]
                dy = end["utm_y"] - start["utm_y"]
                length = math.hypot(dx, dy)
                if length <= 0:
                    continue
                heading = math.atan2(dy, dx)
                if kind == "lane":
                    side_offsets = [0.0]
                    highway = tags.get("highway")
                unit_x, unit_y = dx / length, dy / length
                perp_x, perp_y = -unit_y, unit_x
                for side_offset in side_offsets:
                    if side_offset:
                        shifted_start = {
                            **start,
                            "utm_x": start["utm_x"] + perp_x * side_offset,
                            "utm_y": start["utm_y"] + perp_y * side_offset,
                        }
                        shifted_end = {
                            **end,
                            "utm_x": end["utm_x"] + perp_x * side_offset,
                            "utm_y": end["utm_y"] + perp_y * side_offset,
                        }
                    else:
                        shifted_start, shifted_end = start, end
                    segment = {
                        "start": shifted_start,
                        "end": shifted_end,
                        "length": length,
                        "width": width,
                        "heading": heading,
                        "way_id": way["id"],
                        "highway": highway,
                        "source_highway": tags.get("highway"),
                        "side_offset": side_offset,
                        "oneway": tags.get("oneway"),
                    }
                    if sidewalk_semantic:
                        segment["explicit_sidewalk"] = sidewalk_semantic != "generic_footway"
                        segment["sidewalk_semantic"] = sidewalk_semantic
                    segments.append(segment)
        return segments

    @staticmethod
    def build_explicit_sidewalk_segments(osm: OsmIndex) -> list[dict[str, Any]]:
        return PointSampler.build_segments(osm, "sidewalk", explicit_sidewalk_only=True)

    @staticmethod
    def filter_explicit_sidewalk_points(
        osm: OsmIndex,
        points: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        osm.load()
        semantics = {}
        for way in osm.ways:
            tags = way["tags"]
            if PointSampler.is_explicit_sidewalk_way(tags):
                semantics[str(way["id"])] = "footway=sidewalk"
            elif PointSampler.sidewalk_sides(tags):
                semantics[str(way["id"])] = "road_sidewalk_tag"
        filtered = []
        for point in points:
            semantic = semantics.get(str(point.get("way_id")))
            if not semantic:
                continue
            item = dict(point)
            item["explicit_sidewalk"] = True
            item["sidewalk_semantic"] = semantic
            filtered.append(item)
        return filtered

    @staticmethod
    def clip_segments_to_radius(
        segments: list[dict[str, Any]],
        center_xy: list[float],
        radius_m: float,
    ) -> list[dict[str, Any]]:
        clipped = []
        cx, cy = float(center_xy[0]), float(center_xy[1])
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            dx = float(end["utm_x"]) - float(start["utm_x"])
            dy = float(end["utm_y"]) - float(start["utm_y"])
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                continue
            safe_radius = float(radius_m) - float(segment.get("width", 0.0)) * 0.5
            if safe_radius <= 0.0:
                continue
            sx = float(start["utm_x"]) - cx
            sy = float(start["utm_y"]) - cy
            b = 2.0 * (sx * dx + sy * dy)
            c = sx * sx + sy * sy - safe_radius * safe_radius
            discriminant = b * b - 4.0 * length_sq * c
            if discriminant < 0.0:
                continue
            root = math.sqrt(max(discriminant, 0.0))
            t0 = max(0.0, (-b - root) / (2.0 * length_sq))
            t1 = min(1.0, (-b + root) / (2.0 * length_sq))
            if t1 - t0 <= 1e-9:
                continue

            item = dict(segment)
            item["start"] = PointSampler.interpolate_point(start, end, t0)
            item["end"] = PointSampler.interpolate_point(start, end, t1)
            item["length"] = math.sqrt(length_sq) * (t1 - t0)
            clipped.append(item)
        return clipped

    @staticmethod
    def clip_segments_to_half_plane(
        segments: list[dict[str, Any]],
        center_xy: list[float],
        normal_xy: list[float],
    ) -> list[dict[str, Any]]:
        center = np.asarray(center_xy, dtype=float)
        normal = np.asarray(normal_xy, dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        clipped = []
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            start_xy = np.asarray([start["utm_x"], start["utm_y"]], dtype=float)
            end_xy = np.asarray([end["utm_x"], end["utm_y"]], dtype=float)
            start_signed = float((start_xy - center) @ normal)
            end_signed = float((end_xy - center) @ normal)
            if start_signed < 0.0 and end_signed < 0.0:
                continue
            t0, t1 = 0.0, 1.0
            if start_signed < 0.0 or end_signed < 0.0:
                crossing = -start_signed / (end_signed - start_signed)
                if start_signed < 0.0:
                    t0 = crossing
                else:
                    t1 = crossing
            if t1 - t0 <= 1e-9:
                continue
            item = dict(segment)
            item["start"] = PointSampler.interpolate_point(start, end, t0)
            item["end"] = PointSampler.interpolate_point(start, end, t1)
            item["length"] = float(np.linalg.norm(end_xy - start_xy)) * (t1 - t0)
            clipped.append(item)
        return clipped

    @staticmethod
    def filter_points_to_public_side(
        points: list[dict[str, Any]],
        center_xy: list[float],
        normal_xy: list[float],
        building_footprint_xy: list[list[float]],
        limit: int,
    ) -> list[dict[str, Any]]:
        center = np.asarray(center_xy, dtype=float)
        normal = np.asarray(normal_xy, dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        footprint = np.asarray(building_footprint_xy, dtype=float)
        filtered = []
        for item in points:
            point = np.asarray([item["utm_x"], item["utm_y"]], dtype=float)
            if float((point - center) @ normal) < 0.0:
                continue
            if len(footprint) >= 3 and point_in_polygon_xy(point, footprint):
                continue
            filtered.append(item)
            if len(filtered) >= limit:
                break
        return filtered

    @staticmethod
    def sample(segments: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
        if not segments or count <= 0:
            return []
        total = sum(seg["length"] for seg in segments)
        cumulative = []
        running = 0.0
        for seg in segments:
            running += seg["length"]
            cumulative.append(running)
        points = []
        for _ in range(count):
            target = rng.uniform(0.0, total)
            seg_idx = 0
            while seg_idx < len(cumulative) and cumulative[seg_idx] < target:
                seg_idx += 1
            seg = segments[min(seg_idx, len(segments) - 1)]
            start, end = seg["start"], seg["end"]
            dx, dy = end["utm_x"] - start["utm_x"], end["utm_y"] - start["utm_y"]
            length = seg["length"]
            if length <= 0:
                continue
            t = rng.random()
            unit_x, unit_y = dx / length, dy / length
            perp_x, perp_y = -unit_y, unit_x
            offset = rng.uniform(-seg["width"] / 2.0, seg["width"] / 2.0)
            point = {
                "utm_x": start["utm_x"] + t * dx + perp_x * offset,
                "utm_y": start["utm_y"] + t * dy + perp_y * offset,
                "height": start["height"] + t * (end["height"] - start["height"]),
                "lat": start["lat"] + t * (end["lat"] - start["lat"]),
                "lon": start["lon"] + t * (end["lon"] - start["lon"]),
                "width": seg["width"],
                "way_id": seg["way_id"],
                "highway": seg.get("highway"),
                "heading": seg["heading"],
            }
            if seg.get("sidewalk_semantic"):
                point["explicit_sidewalk"] = bool(seg.get("explicit_sidewalk"))
                point["sidewalk_semantic"] = seg["sidewalk_semantic"]
            points.append(point)
        return points


class StorefrontPlanner:
    def __init__(self, store_root: Path, seed: int, max_storefronts: int | None, min_storefronts: int) -> None:
        self.store_root = store_root
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_storefronts = max_storefronts
        self.min_storefronts = min_storefronts
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> dict[str, list[Path]]:
        catalog: dict[str, list[Path]] = {}
        if not self.store_root.exists():
            return catalog
        for usd in sorted(self.store_root.glob("*/*.usd")):
            if not self._is_usable_storefront_asset(usd):
                continue
            folder = usd.parent.name
            parts = folder.split("_")
            amenity = "_".join(parts[:-1]) if parts and parts[-1].isdigit() else folder
            catalog.setdefault(amenity, []).append(usd)
        return catalog

    @staticmethod
    def _is_usable_storefront_asset(usd: Path) -> bool:
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            return True
        stage = Usd.Stage.Open(str(usd))
        if stage is None:
            return False
        root = stage.GetDefaultPrim()
        if root is None:
            roots = list(stage.GetPseudoRoot().GetChildren())
            root = roots[0] if roots else None
        if root is None:
            return False
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root).ComputeAlignedBox()
        dims = box.GetSize()
        return float(dims[1]) >= 0.03

    def asset_type(self, amenity: str) -> str:
        if amenity == "restaurant" and self.rng.random() < 0.3 and self.catalog.get("convenience_store"):
            return "convenience_store"
        return AMENITY_ASSET_ALIASES.get(amenity, amenity)

    def target_count(self) -> int:
        target = max(self.min_storefronts, 0)
        if self.max_storefronts is not None:
            target = min(target, self.max_storefronts)
        return target

    @staticmethod
    def far_enough(loc: list[float], others: list[list[float]], min_dist: float) -> bool:
        return all(math.hypot(loc[0] - other[0], loc[1] - other[1]) >= min_dist for other in others)

    def synthetic_candidates(
        self,
        osm: OsmIndex,
        focus_center_xy: list[float] | None = None,
        focus_radius_m: float | None = None,
    ) -> list[dict[str, Any]]:
        segments = PointSampler.build_segments(osm, "sidewalk") or PointSampler.build_segments(osm, "lane")
        if focus_center_xy is not None and focus_radius_m is not None:
            segments = PointSampler.clip_segments_to_radius(segments, focus_center_xy, focus_radius_m)
        sample_count = max(self.target_count() * 12, 80)
        return PointSampler.sample(segments, sample_count, self.rng)

    def plan(
        self,
        city: str,
        osm: OsmIndex,
        city_work_dir: Path,
        focus_center_xy: list[float] | None = None,
        focus_radius_m: float | None = None,
        excluded_buildings: list[str] | None = None,
    ) -> dict[str, Any]:
        records = []
        missing = {}
        points = osm.commercial_points()
        if focus_center_xy is not None and focus_radius_m is not None:
            points = [
                point
                for point in points
                if math.hypot(
                    float(point["utm_x"]) - float(focus_center_xy[0]),
                    float(point["utm_y"]) - float(focus_center_xy[1]),
                ) <= float(focus_radius_m)
            ]
        for idx, point in enumerate(points):
            if self.max_storefronts is not None and len(records) >= self.max_storefronts:
                break
            amenity = point["amenity"]
            asset_type = self.asset_type(amenity)
            if asset_type not in REAL_STOREFRONT_ASSET_TYPES:
                continue
            assets = self.catalog.get(asset_type, [])
            if not assets:
                missing[asset_type] = missing.get(asset_type, 0) + 1
                continue
            asset = self.rng.choice(assets)
            records.append({
                "index": idx,
                "city": city,
                "amenity": amenity,
                "asset_type": asset_type,
                "asset_path": os.path.relpath(asset, city_work_dir),
                "utm_x": point["utm_x"],
                "utm_y": point["utm_y"],
                "height": point["height"],
                "status": "planned",
                "source": "osm_amenity",
            })
        target = self.target_count()
        synthetic_added = 0
        synthetic_types = [name for name in SYNTHETIC_STOREFRONT_TYPES if self.catalog.get(name)]
        placed_locs = [[float(item["utm_x"]), float(item["utm_y"])] for item in records]
        next_index = max((int(item["index"]) for item in records), default=-1) + 1
        if synthetic_types and len(records) < target:
            for point in self.synthetic_candidates(osm, focus_center_xy, focus_radius_m):
                if len(records) >= target:
                    break
                loc = [float(point["utm_x"]), float(point["utm_y"])]
                if not self.far_enough(loc, placed_locs, 10.0):
                    continue
                amenity = self.rng.choice(synthetic_types)
                asset_type = self.asset_type(amenity)
                assets = self.catalog.get(asset_type, [])
                if not assets:
                    continue
                asset = self.rng.choice(assets)
                records.append({
                    "index": next_index,
                    "city": city,
                    "amenity": amenity,
                    "asset_type": asset_type,
                    "asset_path": os.path.relpath(asset, city_work_dir),
                    "utm_x": point["utm_x"],
                    "utm_y": point["utm_y"],
                    "height": point.get("height", 0.0),
                    "status": "planned",
                    "source": "synthetic_roadside",
                    "synthetic": True,
                })
                next_index += 1
                placed_locs.append(loc)
                synthetic_added += 1
        result = {
            "city": city,
            "planned_count": len(records),
            "missing_asset_counts": missing,
            "synthetic_added": synthetic_added,
            "min_storefronts": self.min_storefronts,
            "seed": self.seed,
            "asset_catalog": {
                name: [os.path.relpath(asset, city_work_dir) for asset in assets]
                for name, assets in sorted(self.catalog.items())
            },
            "placements": records,
        }
        if focus_center_xy is not None and focus_radius_m is not None:
            result["spatial_filter"] = {
                "center_xy": [float(focus_center_xy[0]), float(focus_center_xy[1])],
                "radius_m": float(focus_radius_m),
            }
        if excluded_buildings:
            result["excluded_buildings"] = sorted(set(str(value) for value in excluded_buildings))
        return result


class ObjaversePlanner:
    def __init__(self, objaverse_root: Path, seed: int) -> None:
        self.objaverse_root = objaverse_root
        self.rng = random.Random(seed)
        self.catalog = self._load_catalog()
        self.semantic_groups = self._load_semantic_groups()
        self.ungrouped_assets = self._load_ungrouped_assets()
        self.object_dims = self._load_object_dims()

    @staticmethod
    def is_motor_vehicle_asset_name(name: str) -> bool:
        return Path(name).name.lower() in OBJAVERSE_VEHICLE_ASSETS

    def _load_usd_files(self) -> dict[str, Path]:
        usd_dir = self.objaverse_root / "usd"
        return {
            p.name: p
            for p in usd_dir.glob("*.usd")
            if objaverse_asset_enabled(p)
        } if usd_dir.exists() else {}

    def _load_catalog(self) -> dict[str, list[Path]]:
        files = self._load_usd_files()
        catalog: dict[str, list[Path]] = {}
        fallback = sorted(files.values())[:20]
        for category, names in OBJAVERSE_CATEGORY_ASSETS.items():
            matches = [files[name] for name in names if name in files]
            catalog[category] = matches or fallback
        vehicle_names = {asset.name for asset in catalog.get("vehicle", [])}
        extra_vehicles = [
            path for name, path in files.items()
            if name not in vehicle_names and self.is_motor_vehicle_asset_name(name)
        ]
        if extra_vehicles:
            catalog["vehicle"] = sorted(catalog.get("vehicle", []) + extra_vehicles)
        return catalog

    def relative_asset(self, asset: Path, city_work_dir: Path) -> str:
        return os.path.relpath(asset, city_work_dir)

    def _load_semantic_groups(self) -> list[dict[str, Any]]:
        for path in [
            Path(__file__).resolve().with_name("semantic_groups.json"),
            Path(__file__).resolve().parents[2] / "3dmodel" / "semantic_groups.json",
        ]:
            if path.exists():
                return [
                    group
                    for group in read_json(path, []) or []
                    if objaverse_outdoor_group_enabled(group.get("name"))
                ]
        return []

    def _load_ungrouped_assets(self) -> list[Path]:
        files = self._load_usd_files()
        grouped_names = {
            name
            for group in self.semantic_groups
            for name in group.get("objects", [])
        }
        vehicle_names = {asset.name for asset in self.catalog.get("vehicle", [])}
        return [
            path
            for name, path in sorted(files.items())
            if name not in grouped_names
            and name not in vehicle_names
            and not self.is_motor_vehicle_asset_name(name)
            and objaverse_outdoor_ungrouped_enabled(name)
        ]

    def _load_object_dims(self) -> dict[str, list[float]]:
        info_path = self.objaverse_root / "usd" / "object_info.json"
        dims: dict[str, list[float]] = {}
        for item in read_json(info_path, []) or []:
            usd_path = item.get("usd_path")
            values = item.get("dims", [1.0, 1.0, 1.0])
            if not usd_path:
                continue
            dims[str(usd_path)] = values
            dims[Path(str(usd_path)).name] = values
        return dims

    def object_dims_for(self, asset: Path) -> list[float]:
        dims = self.object_dims.get(str(asset)) or self.object_dims.get(asset.name) or [1.0, 1.0, 1.0]
        if len(dims) < 3 or (abs(float(dims[0]) - 1.0) < 1e-6 and abs(float(dims[1]) - 1.0) < 1e-6):
            dims = self.actual_object_dims(asset, dims)
        scale = self.asset_scale_for(asset)
        return [float(value) * scale for value in dims]

    @staticmethod
    def actual_object_dims(asset: Path, fallback: list[float]) -> list[float]:
        cache_key = str(asset.resolve())
        if cache_key in _OBJECT_DIM_CACHE:
            return _OBJECT_DIM_CACHE[cache_key]
        try:
            from pxr import Usd, UsdGeom

            stage = Usd.Stage.Open(cache_key)
            root = stage.GetDefaultPrim() if stage else None
            if stage and not root:
                roots = list(stage.GetPseudoRoot().GetChildren())
                root = roots[0] if roots else None
            if not root:
                raise RuntimeError("USD has no root prim")
            box = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
            ).ComputeWorldBound(root).ComputeAlignedBox()
            lower = box.GetMin()
            upper = box.GetMax()
            dims = [float(upper[index] - lower[index]) for index in range(3)]
            if not all(math.isfinite(value) and value > 0.0 for value in dims):
                raise RuntimeError("USD has invalid bounds")
        except Exception:
            dims = [float(value) for value in fallback[:3]]
        _OBJECT_DIM_CACHE[cache_key] = dims
        return dims

    @staticmethod
    def asset_scale_for(asset: Path) -> float:
        return objaverse_asset_scale(asset)

    def object_size_for(self, asset: Path) -> float:
        dims = self.object_dims_for(asset)
        return max(float(dims[0]), float(dims[1]))

    def footprint_radius_for(self, asset: Path) -> float:
        dims = self.object_dims_for(asset)
        return 0.5 * math.hypot(float(dims[0]), float(dims[1]))

    @staticmethod
    def far_enough(loc: list[float], others: list[list[float]], min_dist: float) -> bool:
        for other in others:
            if math.hypot(loc[0] - other[0], loc[1] - other[1]) < min_dist:
                return False
        return True

    @staticmethod
    def footprint_clear(
        loc: list[float],
        radius: float,
        occupied: list[dict[str, Any]],
        clearance: float,
    ) -> bool:
        return all(
            math.hypot(loc[0] - item["location"][0], loc[1] - item["location"][1])
            >= radius
            + float(item["radius"])
            + max(
                clearance,
                VEHICLE_FOOTPRINT_CLEARANCE if item.get("category") == "vehicle" else 0.0,
            )
            for item in occupied
        )

    @staticmethod
    def footprint_clear_of_exclusions(
        loc: list[float],
        radius: float,
        exclusions: list[dict[str, Any]],
    ) -> bool:
        for exclusion in exclusions:
            polygon_values = exclusion.get("polygon_xy") or []
            if len(polygon_values) >= 3 and len(loc) >= 2:
                point = np.asarray(loc[:2], dtype=float)
                polygon = np.asarray(polygon_values, dtype=float)
                clearance = radius + float(exclusion.get("clearance_m", 0.0))
                edge_distance = min(
                    point_segment_distance_xy(point, polygon[index], polygon[(index + 1) % len(polygon)])
                    for index in range(len(polygon))
                )
                if point_in_polygon_xy(point, polygon) or edge_distance <= clearance:
                    return False
                continue
            center = exclusion.get("center_xy") or []
            tangent = exclusion.get("tangent_xy") or []
            outward = exclusion.get("outward_normal_xy") or []
            if len(loc) < 2 or len(center) != 2 or len(tangent) != 2 or len(outward) != 2:
                continue
            tangent_length = math.hypot(float(tangent[0]), float(tangent[1]))
            outward_length = math.hypot(float(outward[0]), float(outward[1]))
            if tangent_length <= 1e-6 or outward_length <= 1e-6:
                continue
            delta_x = float(loc[0]) - float(center[0])
            delta_y = float(loc[1]) - float(center[1])
            lateral = abs(
                delta_x * float(tangent[0]) / tangent_length
                + delta_y * float(tangent[1]) / tangent_length
            )
            normal = (
                delta_x * float(outward[0]) / outward_length
                + delta_y * float(outward[1]) / outward_length
            )
            half_width = (
                float(exclusion.get("width_m", 0.0)) * 0.5
                + float(exclusion.get("lateral_clearance_m", 0.0))
                + radius
            )
            inward = float(exclusion.get("inward_clearance_m", 0.0)) + radius
            outward_distance = float(exclusion.get("outward_clearance_m", 0.0)) + radius
            if lateral <= half_width and -inward <= normal <= outward_distance:
                return False
        return True

    def choose_vehicle_asset(
        self,
        loc: list[float],
        vehicle_assets: list[Path],
        vehicle_history: dict[str, list[list[float]]],
        occupied: list[dict[str, Any]],
        exclusions: list[dict[str, Any]],
    ) -> Path | None:
        candidates = vehicle_assets[:]
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda asset: len(vehicle_history.get(asset.name, [])))
        for asset in candidates:
            radius = self.footprint_radius_for(asset)
            if (
                self.far_enough(loc, vehicle_history.get(asset.name, []), VEHICLE_SAME_ASSET_MIN_DISTANCE)
                and self.footprint_clear(loc, radius, occupied, VEHICLE_FOOTPRINT_CLEARANCE)
                and self.footprint_clear_of_exclusions(loc, radius, exclusions)
            ):
                return asset
        return None

    def nearby_sidewalk_candidates(
        self,
        center: list[float],
        sidewalk_points: list[dict[str, Any]],
        terrain: TerrainHeightQuery,
    ) -> list[dict[str, Any]]:
        candidates = []
        radius_sq = PROP_GROUP_SIDEWALK_RADIUS * PROP_GROUP_SIDEWALK_RADIUS
        for point in sidewalk_points:
            dx = float(point["utm_x"]) - center[0]
            dy = float(point["utm_y"]) - center[1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > radius_sq:
                continue
            loc = [
                float(point["utm_x"]),
                float(point["utm_y"]),
                terrain.height_at(point["utm_x"], point["utm_y"], point.get("height", center[2])),
            ]
            candidates.append({
                "loc": loc,
                "heading": point.get("heading", 0.0),
                "dist_sq": dist_sq,
                "way_id": point.get("way_id"),
                "sidewalk_semantic": point.get("sidewalk_semantic"),
            })
        candidates.sort(key=lambda item: item["dist_sq"])
        if len(candidates) > 1:
            near_center = candidates[:1]
            rest = candidates[1:]
            self.rng.shuffle(rest)
            candidates = near_center + rest
        return candidates

    def balanced_fill_assets(self, filename_to_path: dict[str, Path]) -> list[Path]:
        assets = [
            path
            for name, path in filename_to_path.items()
            if objaverse_asset_enabled(name)
            and not self.is_motor_vehicle_asset_name(name)
            and (
                objaverse_outdoor_ungrouped_enabled(name)
                or any(name in group.get("objects", []) for group in self.semantic_groups)
            )
        ]
        compact = [asset for asset in assets if self.object_size_for(asset) <= 4.0]
        return sorted(compact or assets)

    def fill_balanced_sidewalk_props(
        self,
        objects: list[dict[str, Any]],
        occupied: list[dict[str, Any]],
        target_count: int,
        sidewalk_points: list[dict[str, Any]],
        terrain: TerrainHeightQuery,
        store_locs: list[list[float]],
        city_work_dir: Path,
        assets: list[Path],
        asset_usage: dict[str, int],
        max_store_clearance: float,
        exclusions: list[dict[str, Any]],
    ) -> int:
        if not assets:
            return sum(item.get("category") == "prop" for item in objects)
        prop_objects = [item for item in objects if item.get("category") == "prop"]
        asset_history: dict[str, list[list[float]]] = {}
        for item in prop_objects:
            asset_history.setdefault(Path(item["asset_path"]).name, []).append(list(item["location"]))

        candidates = list(sidewalk_points)
        self.rng.shuffle(candidates)
        passes = tuple(
            (same_asset_spacing, min(store_clearance, max_store_clearance))
            for same_asset_spacing, store_clearance in (
                (10.0, 12.0),
                (8.0, 10.0),
                (5.0, 8.0),
                (2.0, 5.0),
                (0.0, 3.0),
            )
        )
        for same_asset_spacing, store_clearance in passes:
            for point in candidates:
                if len(prop_objects) >= target_count:
                    return len(prop_objects)
                loc = [
                    float(point["utm_x"]),
                    float(point["utm_y"]),
                    terrain.height_at(point["utm_x"], point["utm_y"], point.get("height", 0.0)),
                ]
                if not self.far_enough(loc, store_locs, store_clearance):
                    continue
                ordered_assets = list(assets)
                self.rng.shuffle(ordered_assets)
                ordered_assets.sort(key=lambda asset: asset_usage.get(asset.name, 0))
                asset = next(
                    (
                        candidate
                        for candidate in ordered_assets
                        if (
                            not same_asset_spacing
                            or self.far_enough(
                                loc, asset_history.get(candidate.name, []), same_asset_spacing
                            )
                        )
                        and self.footprint_clear(
                            loc,
                            self.footprint_radius_for(candidate),
                            occupied,
                            PROP_FOOTPRINT_CLEARANCE,
                        )
                        and self.footprint_clear_of_exclusions(
                            loc,
                            self.footprint_radius_for(candidate),
                            exclusions,
                        )
                    ),
                    None,
                )
                if asset is None:
                    continue
                item = {
                    "category": "prop",
                    "location": loc,
                    "heading": point.get("heading", 0.0),
                    "asset_path": self.relative_asset(asset, city_work_dir),
                    "semantic_group": "balanced_fill",
                    "placement_surface": "sidewalk",
                    "placement_way_id": point.get("way_id"),
                    "placement_semantic": point.get("sidewalk_semantic"),
                    "ungrouped_asset": asset in self.ungrouped_assets,
                    "scale": [self.asset_scale_for(asset)] * 3,
                    "footprint_dims_m": self.object_dims_for(asset)[:2],
                    "footprint_radius_m": self.footprint_radius_for(asset),
                }
                objects.append(item)
                prop_objects.append(item)
                occupied.append(
                    {"location": loc, "radius": item["footprint_radius_m"], "category": "prop"}
                )
                asset_history.setdefault(asset.name, []).append(loc)
                asset_usage[asset.name] = asset_usage.get(asset.name, 0) + 1
        return len(prop_objects)

    def place_vehicles(
        self,
        objects: list[dict[str, Any]],
        occupied: list[dict[str, Any]],
        target_vehicles: int,
        lane_points: list[dict[str, Any]],
        terrain: TerrainHeightQuery,
        store_locs: list[list[float]],
        store_clearance: float,
        placement_exclusions: list[dict[str, Any]],
        city_work_dir: Path,
    ) -> int:
        vehicle_count = min(max(int(target_vehicles), 0), len(lane_points)) if lane_points else 0
        vehicle_assets = self.catalog.get("vehicle", [])
        vehicle_history: dict[str, list[list[float]]] = {asset.name: [] for asset in vehicle_assets}
        if not vehicle_assets or not lane_points:
            return 0

        vehicle_candidates = list(lane_points)
        self.rng.shuffle(vehicle_candidates)
        placed_vehicle_count = 0
        for point in vehicle_candidates:
            if placed_vehicle_count >= vehicle_count:
                break
            ground_z = terrain.height_at(point["utm_x"], point["utm_y"], point.get("height", 0.0))
            loc = [point["utm_x"], point["utm_y"], ground_z]
            asset = self.choose_vehicle_asset(
                loc,
                vehicle_assets,
                vehicle_history,
                occupied,
                placement_exclusions,
            )
            if asset is None:
                continue
            footprint_radius = self.footprint_radius_for(asset)
            if not self.far_enough(loc, store_locs, store_clearance + footprint_radius):
                continue
            vehicle_history.setdefault(asset.name, []).append(loc)
            objects.append({
                "category": "vehicle",
                "location": loc,
                "heading": point.get("heading", 0.0),
                "asset_path": self.relative_asset(asset, city_work_dir),
                "placement_surface": "road",
                "scale": [self.asset_scale_for(asset)] * 3,
                "footprint_dims_m": self.object_dims_for(asset)[:2],
                "footprint_radius_m": footprint_radius,
            })
            occupied.append({"location": loc, "radius": footprint_radius, "category": "vehicle"})
            placed_vehicle_count += 1
        return placed_vehicle_count

    def plan(
        self,
        city: str,
        city_work_dir: Path,
        lane_points: list[dict[str, Any]],
        sidewalk_points: list[dict[str, Any]],
        terrain: TerrainHeightQuery | None = None,
        store_locs: list[list[float]] | None = None,
        target_props: int = OBJAVERSE_TARGET_PROPS,
        target_vehicles: int = OBJAVERSE_TARGET_VEHICLES,
        store_clearance: float = 12.0,
        placement_exclusions: list[dict[str, Any]] | None = None,
        prioritize_sidewalk_props: bool = False,
    ) -> dict[str, Any]:
        objects = []
        store_locs = store_locs or []
        placement_exclusions = placement_exclusions or []
        invalid_sidewalk_points = [
            point for point in sidewalk_points if point.get("explicit_sidewalk") is not True
        ]
        if invalid_sidewalk_points:
            raise ValueError(
                f"Objaverse sidewalk placement received {len(invalid_sidewalk_points)} "
                "points without explicit OSM sidewalk semantics"
            )
        terrain = terrain or TerrainHeightQuery(None)
        occupied: list[dict[str, Any]] = []
        if not prioritize_sidewalk_props:
            self.place_vehicles(
                objects,
                occupied,
                target_vehicles,
                lane_points,
                terrain,
                store_locs,
                store_clearance,
                placement_exclusions,
                city_work_dir,
            )

        sidewalk_count = min(max(int(target_props), 0), len(sidewalk_points)) if sidewalk_points else 0
        usd_dir = self.objaverse_root / "usd"
        filename_to_path = {p.name: p for p in usd_dir.glob("*.usd")} if usd_dir.exists() else {}
        if sidewalk_count <= 0 or not sidewalk_points:
            if prioritize_sidewalk_props:
                self.place_vehicles(
                    objects, occupied, target_vehicles, lane_points, terrain, store_locs,
                    store_clearance, placement_exclusions, city_work_dir,
                )
            return {
                "city": city,
                "sidewalk_constraint": "explicit_osm_sidewalk",
                "placement_exclusions": placement_exclusions,
                "objects": objects,
            }
        if not self.semantic_groups:
            sidewalk_assets = self.catalog.get("sidewalk", [])
            if not sidewalk_assets:
                if prioritize_sidewalk_props:
                    self.place_vehicles(
                        objects, occupied, target_vehicles, lane_points, terrain, store_locs,
                        store_clearance, placement_exclusions, city_work_dir,
                    )
                return {
                    "city": city,
                    "sidewalk_constraint": "explicit_osm_sidewalk",
                    "placement_exclusions": placement_exclusions,
                    "objects": objects,
                }
            prop_asset_usage: dict[str, int] = {asset.name: 0 for asset in sidewalk_assets}
            prop_candidates = list(sidewalk_points)
            self.rng.shuffle(prop_candidates)
            placed_prop_count = 0
            for point in prop_candidates:
                if placed_prop_count >= sidewalk_count:
                    break
                candidates = sidewalk_assets[:]
                self.rng.shuffle(candidates)
                candidates.sort(key=lambda asset: prop_asset_usage.get(asset.name, 0))
                asset = candidates[0]
                loc = [
                    point["utm_x"],
                    point["utm_y"],
                    terrain.height_at(point["utm_x"], point["utm_y"], point.get("height", 0.0)),
                ]
                footprint_radius = self.footprint_radius_for(asset)
                if not self.far_enough(loc, store_locs, store_clearance + footprint_radius):
                    continue
                if not self.footprint_clear(
                    loc, footprint_radius, occupied, PROP_FOOTPRINT_CLEARANCE
                ):
                    continue
                if not self.footprint_clear_of_exclusions(
                    loc, footprint_radius, placement_exclusions
                ):
                    continue
                item = {
                    "category": "prop",
                    "location": loc,
                    "heading": point.get("heading", 0.0),
                    "asset_path": self.relative_asset(asset, city_work_dir),
                    "placement_surface": "sidewalk",
                    "placement_way_id": point.get("way_id"),
                    "placement_semantic": point.get("sidewalk_semantic"),
                    "scale": [self.asset_scale_for(asset)] * 3,
                    "footprint_dims_m": self.object_dims_for(asset)[:2],
                    "footprint_radius_m": footprint_radius,
                }
                objects.append(item)
                occupied.append({"location": loc, "radius": footprint_radius, "category": "prop"})
                prop_asset_usage[asset.name] = prop_asset_usage.get(asset.name, 0) + 1
                placed_prop_count += 1
            if prioritize_sidewalk_props:
                self.place_vehicles(
                    objects, occupied, target_vehicles, lane_points, terrain, store_locs,
                    store_clearance, placement_exclusions, city_work_dir,
                )
            return {
                "city": city,
                "sidewalk_constraint": "explicit_osm_sidewalk",
                "placement_exclusions": placement_exclusions,
                "objects": objects,
            }

        candidate_points = list(sidewalk_points)
        self.rng.shuffle(candidate_points)
        groups = []
        for point in candidate_points:
            loc = [
                point["utm_x"],
                point["utm_y"],
                terrain.height_at(point["utm_x"], point["utm_y"], point.get("height", 0.0)),
            ]
            if not self.far_enough(loc, store_locs, store_clearance):
                continue
            if not self.far_enough(loc, [group["center"] for group in groups], PROP_GROUP_MIN_DISTANCE):
                continue
            groups.append({"center": loc, "heading": point.get("heading", 0.0)})

        placement_groups = list(self.semantic_groups)
        placement_groups.extend(
            {"name": f"ungrouped_{asset.stem}", "objects": [asset.name], "ungrouped": True}
            for asset in self.ungrouped_assets
        )
        semantic_group_history = {group["name"]: [] for group in placement_groups}
        semantic_group_usage = {group["name"]: 0 for group in placement_groups}
        prop_asset_usage: dict[str, int] = {}
        prop_count = 0
        for group in groups:
            if prop_count >= sidewalk_count:
                break
            center = group["center"]
            candidates = list(placement_groups)
            self.rng.shuffle(candidates)
            candidates.sort(
                key=lambda semantic_group: (
                    semantic_group_usage.get(semantic_group["name"], 0),
                    sum(prop_asset_usage.get(name, 0) for name in semantic_group.get("objects", [])),
                )
            )
            selected = None
            for semantic_group in candidates:
                previous = semantic_group_history.get(semantic_group["name"], [])
                if self.far_enough(center, previous, PROP_GROUP_REPEAT_MIN_DISTANCE):
                    selected = semantic_group
                    break
            if not selected:
                continue
            semantic_group_history[selected["name"]].append(center)
            group_assets = [
                filename_to_path[name]
                for name in selected.get("objects", [])
                if name in filename_to_path and objaverse_asset_enabled(name)
            ]
            if not group_assets:
                continue
            self.rng.shuffle(group_assets)
            group_assets.sort(key=lambda asset: prop_asset_usage.get(asset.name, 0))

            group_start_count = prop_count
            sidewalk_candidates = self.nearby_sidewalk_candidates(center, sidewalk_points, terrain)
            for asset in group_assets:
                if prop_count >= sidewalk_count:
                    break
                asset_dims = self.object_dims_for(asset)
                footprint_radius = self.footprint_radius_for(asset)
                best_loc = None
                best_heading = group["heading"]
                best_candidate = None
                for candidate in sidewalk_candidates:
                    candidate_loc = list(candidate["loc"])
                    if not self.far_enough(
                        candidate_loc, store_locs, store_clearance + footprint_radius
                    ):
                        continue
                    if not self.footprint_clear(
                        candidate_loc,
                        footprint_radius,
                        occupied,
                        PROP_FOOTPRINT_CLEARANCE,
                    ):
                        continue
                    if not self.footprint_clear_of_exclusions(
                        candidate_loc,
                        footprint_radius,
                        placement_exclusions,
                    ):
                        continue
                    best_loc = candidate_loc
                    best_heading = candidate["heading"]
                    best_candidate = candidate
                    break
                if not best_loc:
                    continue
                objects.append({
                    "category": "prop",
                    "location": best_loc,
                    "heading": best_heading,
                    "asset_path": self.relative_asset(asset, city_work_dir),
                    "semantic_group": selected["name"],
                    "placement_surface": "sidewalk",
                    "placement_way_id": best_candidate.get("way_id") if best_candidate else None,
                    "placement_semantic": best_candidate.get("sidewalk_semantic") if best_candidate else None,
                    "ungrouped_asset": bool(selected.get("ungrouped")),
                    "scale": [self.asset_scale_for(asset)] * 3,
                    "footprint_dims_m": asset_dims[:2],
                    "footprint_radius_m": footprint_radius,
                })
                occupied.append({"location": best_loc, "radius": footprint_radius, "category": "prop"})
                prop_asset_usage[asset.name] = prop_asset_usage.get(asset.name, 0) + 1
                prop_count += 1
            if prop_count > group_start_count:
                semantic_group_usage[selected["name"]] = semantic_group_usage.get(selected["name"], 0) + 1
        if prop_count < sidewalk_count:
            prop_count = self.fill_balanced_sidewalk_props(
                objects,
                occupied,
                sidewalk_count,
                sidewalk_points,
                terrain,
                store_locs,
                city_work_dir,
                self.balanced_fill_assets(filename_to_path),
                prop_asset_usage,
                store_clearance,
                placement_exclusions,
            )
        if prioritize_sidewalk_props:
            self.place_vehicles(
                objects, occupied, target_vehicles, lane_points, terrain, store_locs,
                store_clearance, placement_exclusions, city_work_dir,
            )
        return {
            "city": city,
            "sidewalk_constraint": "explicit_osm_sidewalk",
            "placement_exclusions": placement_exclusions,
            "objects": objects,
        }


class RoadObjectPlanner:
    def __init__(self, road_source_root: Path, road_output_root: Path) -> None:
        self.road_source_root = road_source_root
        self.road_output_root = road_output_root

    def prepare_assets(self) -> None:
        ensure_dir(self.road_output_root)
        for usd_filename in sorted(set(ROAD_OBJECT_ASSETS.values())):
            source = self.road_source_root / usd_filename.replace(".usd", ".glb")
            target = self.road_output_root / source.name
            if source.exists() and not target.exists():
                shutil.copy2(source, target)

    @staticmethod
    def road_angle_at_node(node: dict[str, Any], osm: OsmIndex) -> float:
        for way in osm.ways:
            refs = way.get("nodes", [])
            if node["id"] not in refs:
                continue
            index = refs.index(node["id"])
            prev_node = osm.nodes.get(refs[index - 1]) if index > 0 else None
            next_node = osm.nodes.get(refs[index + 1]) if index < len(refs) - 1 else None
            if prev_node and next_node:
                dx = next_node["utm_x"] - prev_node["utm_x"]
                dy = next_node["utm_y"] - prev_node["utm_y"]
            elif next_node:
                dx = next_node["utm_x"] - node["utm_x"]
                dy = next_node["utm_y"] - node["utm_y"]
            elif prev_node:
                dx = node["utm_x"] - prev_node["utm_x"]
                dy = node["utm_y"] - prev_node["utm_y"]
            else:
                continue
            if dx or dy:
                return math.atan2(dy, dx)
        return 0.0

    @staticmethod
    def project_to_segment(x: float, y: float, segment: dict[str, Any]) -> dict[str, float] | None:
        start, end = segment["start"], segment["end"]
        dx = end["utm_x"] - start["utm_x"]
        dy = end["utm_y"] - start["utm_y"]
        length_sq = dx * dx + dy * dy
        if length_sq <= 0.0:
            return None
        t = ((x - start["utm_x"]) * dx + (y - start["utm_y"]) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        px = start["utm_x"] + t * dx
        py = start["utm_y"] + t * dy
        dist = math.hypot(x - px, y - py)
        length = math.sqrt(length_sq)
        return {
            "x": px,
            "y": py,
            "height": start["height"] + t * (end["height"] - start["height"]),
            "distance": dist,
            "heading": segment["heading"],
            "nx": -dy / length,
            "ny": dx / length,
            "width": segment["width"],
        }

    @classmethod
    def nearest_segment_projection(
        cls,
        x: float,
        y: float,
        segments: list[dict[str, Any]],
        max_distance: float,
    ) -> dict[str, float] | None:
        best = None
        for segment in segments:
            projection = cls.project_to_segment(x, y, segment)
            if not projection or projection["distance"] > max_distance:
                continue
            if best is None or projection["distance"] < best["distance"]:
                best = projection
        return best

    @staticmethod
    def signal_category(tags: dict[str, str]) -> str | None:
        if tags.get("highway") == "traffic_signals":
            return "traffic_signals"
        if tags.get("highway") == "crossing" and "traffic_signals" in (tags.get("crossing") or ""):
            return "crossing"
        return None

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def travel_headings_for_segment(segment: dict[str, Any]) -> list[float]:
        heading = float(segment["heading"])
        oneway = (segment.get("oneway") or "").lower()
        if oneway in {"yes", "true", "1"}:
            return [heading]
        if oneway in {"-1", "reverse"}:
            return [heading + math.pi]
        return [heading, heading + math.pi]

    @staticmethod
    def cluster_signal_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clusters: list[list[dict[str, Any]]] = []
        for node in nodes:
            assigned = False
            for cluster in clusters:
                if any(
                    math.hypot(node["utm_x"] - other["utm_x"], node["utm_y"] - other["utm_y"])
                    <= TRAFFIC_SIGNAL_CLUSTER_DISTANCE
                    for other in cluster
                ):
                    cluster.append(node)
                    assigned = True
                    break
            if not assigned:
                clusters.append([node])

        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                if merged:
                    break
                for j in range(i + 1, len(clusters)):
                    if any(
                        math.hypot(a["utm_x"] - b["utm_x"], a["utm_y"] - b["utm_y"])
                        <= TRAFFIC_SIGNAL_CLUSTER_DISTANCE
                        for a in clusters[i]
                        for b in clusters[j]
                    ):
                        clusters[i].extend(clusters[j])
                        del clusters[j]
                        merged = True
                        break

        result = []
        for idx, cluster in enumerate(clusters):
            xs = [float(node["utm_x"]) for node in cluster]
            ys = [float(node["utm_y"]) for node in cluster]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            radius = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys, strict=True))
            result.append({"id": idx, "nodes": cluster, "center": [cx, cy], "radius": radius})
        return result

    def approach_candidates_for_cluster(
        self,
        cluster: dict[str, Any],
        lane_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cx, cy = cluster["center"]
        candidates = []
        for segment in lane_segments:
            projection = self.project_to_segment(cx, cy, segment)
            if not projection:
                continue
            max_distance = max(TRAFFIC_SIGNAL_APPROACH_RADIUS, projection["width"] / 2.0 + 12.0)
            if projection["distance"] > max_distance:
                continue
            for travel_heading in self.travel_headings_for_segment(segment):
                candidates.append({
                    "travel_heading": travel_heading,
                    "distance": projection["distance"],
                    "width": projection["width"],
                    "height": projection["height"],
                    "way_id": segment.get("way_id"),
                    "highway": segment.get("highway"),
                })
        candidates.sort(key=lambda item: item["distance"])

        groups: list[dict[str, Any]] = []
        threshold = math.radians(TRAFFIC_SIGNAL_APPROACH_ANGLE_DEDUP_DEG)
        for candidate in candidates:
            if any(self.angle_diff(candidate["travel_heading"], group["travel_heading"]) < threshold for group in groups):
                continue
            groups.append(candidate)
            if len(groups) >= 8:
                break
        return groups

    def signal_cluster_objects(
        self,
        city: str,
        clusters: list[dict[str, Any]],
        lane_segments: list[dict[str, Any]],
        terrain: TerrainHeightQuery,
    ) -> list[dict[str, Any]]:
        asset_name = ROAD_OBJECT_ASSETS["traffic_signals"]
        asset_path = os.path.relpath(self.road_output_root / asset_name, self.road_output_root.parent / city)
        objects = []
        for cluster in clusters:
            cx, cy = cluster["center"]
            source_node_ids = [node["id"] for node in cluster["nodes"]]
            source_categories = sorted({self.signal_category(node.get("tags", {})) or "unknown" for node in cluster["nodes"]})
            approaches = self.approach_candidates_for_cluster(cluster, lane_segments)
            if not approaches:
                nearest = self.nearest_segment_projection(cx, cy, lane_segments, TRAFFIC_SIGNAL_APPROACH_RADIUS)
                if not nearest:
                    continue
                approaches = [{
                    "travel_heading": nearest["heading"],
                    "distance": nearest["distance"],
                    "width": nearest["width"],
                    "height": nearest["height"],
                    "way_id": None,
                    "highway": None,
                }]
            far_offset = min(max(float(cluster["radius"]) + 2.0, TRAFFIC_SIGNAL_FAR_SIDE_OFFSET), 12.0)
            for approach_idx, approach in enumerate(approaches):
                travel_heading = float(approach["travel_heading"])
                right_x = math.sin(travel_heading)
                right_y = -math.cos(travel_heading)
                forward_x = math.cos(travel_heading)
                forward_y = math.sin(travel_heading)
                curb_offset = float(approach["width"]) / 2.0 + TRAFFIC_SIGNAL_CURB_OFFSET
                sx = cx + forward_x * far_offset + right_x * curb_offset
                sy = cy + forward_y * far_offset + right_y * curb_offset
                fallback_z = float(approach.get("height", 0.0))
                z = terrain.height_at(sx, sy, fallback_z)
                objects.append({
                    "category": "traffic_signals",
                    "location": [cx, cy, terrain.height_at(cx, cy, fallback_z)],
                    "final_location": [sx, sy, z],
                    "heading": travel_heading + math.pi,
                    "asset_path": asset_path,
                    "node_id": f"cluster_{cluster['id']}_approach_{approach_idx}",
                    "source_node_ids": source_node_ids,
                    "source_categories": source_categories,
                    "placement_source": "intersection_approach",
                    "cluster_id": cluster["id"],
                    "cluster_radius_m": cluster["radius"],
                    "approach_way_id": approach.get("way_id"),
                    "approach_highway": approach.get("highway"),
                    "approach_distance_m": approach["distance"],
                    "approach_travel_heading": travel_heading,
                    "road_side": "right",
                })
        return objects

    def road_object_pose(
        self,
        category: str,
        node: dict[str, Any],
        osm: OsmIndex,
        terrain: TerrainHeightQuery,
        lane_segments: list[dict[str, Any]],
        sidewalk_segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        x, y = float(node["utm_x"]), float(node["utm_y"])
        base_z = terrain.height_at(x, y, node["height"])
        heading = self.road_angle_at_node(node, osm)
        pose = {
            "location": [x, y, base_z],
            "final_location": [x, y, base_z],
            "heading": heading + math.pi,
            "placement_source": "osm_node",
        }
        nearest_lane = self.nearest_segment_projection(x, y, lane_segments, 20.0)
        if nearest_lane:
            pose["heading"] = nearest_lane["heading"] + math.pi
        if category == "bus_stop":
            nearest_sidewalk = self.nearest_segment_projection(x, y, sidewalk_segments, 12.0)
            if nearest_sidewalk:
                sx, sy = nearest_sidewalk["x"], nearest_sidewalk["y"]
                z = terrain.height_at(sx, sy, nearest_sidewalk["height"])
                pose.update({
                    "final_location": [sx, sy, z],
                    "heading": nearest_sidewalk["heading"] + math.pi,
                    "placement_source": "nearest_sidewalk",
                    "placement_distance_m": nearest_sidewalk["distance"],
                })
            elif nearest_lane:
                side = 1.0
                dot = (x - nearest_lane["x"]) * nearest_lane["nx"] + (y - nearest_lane["y"]) * nearest_lane["ny"]
                if abs(dot) > 0.2:
                    side = 1.0 if dot > 0.0 else -1.0
                offset = nearest_lane["width"] / 2.0 + DEFAULT_SIDEWALK_WIDTH / 2.0
                sx = nearest_lane["x"] + nearest_lane["nx"] * side * offset
                sy = nearest_lane["y"] + nearest_lane["ny"] * side * offset
                z = terrain.height_at(sx, sy, nearest_lane["height"])
                pose.update({
                    "final_location": [sx, sy, z],
                    "placement_source": "road_edge",
                    "placement_distance_m": nearest_lane["distance"],
                })
        elif category in {"crossing", "traffic_signals", "stop"} and nearest_lane:
            side = 1.0
            dot = (x - nearest_lane["x"]) * nearest_lane["nx"] + (y - nearest_lane["y"]) * nearest_lane["ny"]
            if abs(dot) > 0.2:
                side = 1.0 if dot > 0.0 else -1.0
            offset = nearest_lane["width"] / 2.0 + 0.5
            heading = pose["heading"]
            placement_rule = "road_edge"
            if category in {"crossing", "traffic_signals"}:
                if abs(dot) <= 0.2:
                    side = -1.0
                travel_heading = nearest_lane["heading"] if side < 0.0 else nearest_lane["heading"] + math.pi
                heading = travel_heading + math.pi
                offset = nearest_lane["width"] / 2.0 + TRAFFIC_SIGNAL_CURB_OFFSET
                placement_rule = "far_side_right_hand"
                sx = (
                    nearest_lane["x"]
                    + nearest_lane["nx"] * side * offset
                    + math.cos(travel_heading) * TRAFFIC_SIGNAL_FAR_SIDE_OFFSET
                )
                sy = (
                    nearest_lane["y"]
                    + nearest_lane["ny"] * side * offset
                    + math.sin(travel_heading) * TRAFFIC_SIGNAL_FAR_SIDE_OFFSET
                )
            else:
                sx = nearest_lane["x"] + nearest_lane["nx"] * side * offset
                sy = nearest_lane["y"] + nearest_lane["ny"] * side * offset
            z = terrain.height_at(sx, sy, nearest_lane["height"])
            pose.update({
                "final_location": [sx, sy, z],
                "heading": heading,
                "placement_source": placement_rule,
                "placement_distance_m": nearest_lane["distance"],
                "road_side": "right" if side < 0.0 else "left",
            })
        return pose

    def plan(self, city: str, osm: OsmIndex, terrain: TerrainHeightQuery | None = None) -> dict[str, Any]:
        osm.load()
        terrain = terrain or TerrainHeightQuery(None)
        objects = []
        stats = {key: 0 for key in ROAD_OBJECT_ASSETS}
        placement_stats: dict[str, int] = {}
        lane_segments = PointSampler.build_segments(osm, "lane")
        sidewalk_segments = PointSampler.build_segments(osm, "sidewalk")
        signal_nodes = [node for node in osm.nodes.values() if self.signal_category(node.get("tags", {}))]
        signal_clusters = self.cluster_signal_nodes(signal_nodes)
        signal_objects = self.signal_cluster_objects(city, signal_clusters, lane_segments, terrain)
        objects.extend(signal_objects)
        stats["traffic_signals"] += len(signal_objects)
        for obj in signal_objects:
            placement_stats[obj["placement_source"]] = placement_stats.get(obj["placement_source"], 0) + 1
        for node in osm.nodes.values():
            tags = node.get("tags", {})
            category = None
            if self.signal_category(tags):
                continue
            if tags.get("highway") in {"bus_stop", "stop"}:
                category = tags.get("highway")
            elif tags.get("public_transport") == "platform" or tags.get("bus") == "yes":
                category = "bus_stop"
            if not category or category not in ROAD_OBJECT_ASSETS:
                continue
            asset_name = ROAD_OBJECT_ASSETS[category]
            asset_path = os.path.relpath(self.road_output_root / asset_name, self.road_output_root.parent / city)
            pose = self.road_object_pose(category, node, osm, terrain, lane_segments, sidewalk_segments)
            placement_stats[pose["placement_source"]] = placement_stats.get(pose["placement_source"], 0) + 1
            objects.append({
                "category": category,
                **pose,
                "asset_path": asset_path,
                "node_id": node["id"],
            })
            stats[category] += 1
        return {
            "city": city,
            "total_objects": len(objects),
            "statistics": stats,
            "placement_statistics": placement_stats,
            "traffic_signal_clusters": len(signal_clusters),
            "objects": objects,
        }


class QualityComparator:
    def __init__(self, vc_plus_root: Path) -> None:
        self.vc_plus_root = vc_plus_root

    def reference(self, city: str) -> dict[str, Any]:
        city_dir = self.vc_plus_root / city
        if not city_dir.exists():
            return {"exists": False}
        obj = read_json(city_dir / "objaverse_objects.json", {}) or {}
        road = read_json(city_dir / "road_objects_info.json", {}) or {}
        cut = read_json(city_dir / "cut_buildings.json", {}) or {}
        return {
            "exists": True,
            "files": sorted(p.name for p in city_dir.glob("*.usd")),
            "objaverse_count": len(obj.get("objects", [])),
            "vehicle_count": sum(1 for item in obj.get("objects", []) if item.get("category") == "vehicle"),
            "road_object_count": road.get("total_objects", len(road.get("objects", [])) if isinstance(road, dict) else 0),
            "cut_building_count": cut.get("total_cut_buildings"),
        }


class VcFullPipeline:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        if config.stage in {"all", "plan", "compose"}:
            required_roots = {
                "storefront": config.store_root,
                "objaverse": config.objaverse_root,
                "vc_plus": config.vc_plus_root,
            }
            if config.variant in {"innout", "both"}:
                required_roots["grscene"] = config.grscene_root
            missing = [f"{name}={path}" for name, path in required_roots.items() if not path.is_dir()]
            if missing:
                raise FileNotFoundError("missing VC Full asset roots: " + ", ".join(missing))
        self.compare = QualityComparator(config.vc_plus_root)
        self.storefronts = StorefrontPlanner(config.store_root, config.seed, config.max_storefronts, config.min_storefronts)
        self.objaverse = ObjaversePlanner(config.objaverse_root, config.seed)
        self.road_planner = RoadObjectPlanner(config.vc_plus_root / "road_objects", config.work_root / "road_objects")
        self.grscene_assignments: dict[str, str] = {}

    def configure_grscene_assignments(self, cities: list[str]) -> None:
        scenes = sorted(p.parent.name for p in self.config.grscene_root.glob("scenes/scene*/start_result_navigation.usd"))
        if not scenes:
            self.grscene_assignments = {}
            return
        rng = random.Random(self.config.seed)
        shuffled = scenes[:]
        rng.shuffle(shuffled)
        self.grscene_assignments = {city: shuffled[index % len(shuffled)] for index, city in enumerate(cities)}

    def convert_road_assets(self) -> dict[str, Any]:
        files = sorted({name.replace(".usd", ".glb") for name in ROAD_OBJECT_ASSETS.values()})
        missing = [name for name in ROAD_OBJECT_ASSETS.values() if not (self.config.work_root / "road_objects" / name).exists()]
        if not missing:
            return {"skipped": "existing"}
        script = self.config.tools_dir / "convert_road_objects_blender.py"
        cmd = [
            self.config.blender, "-b", "--python", str(script), "--",
            "--input-dir", str(self.config.work_root / "road_objects"),
            "--output-dir", str(self.config.work_root / "road_objects"),
            "--files", *files,
        ]
        return run_logged(cmd, self.config.log_root / "convert_road_objects.log", self.config.dry_run, self.config.repo_root)

    def city_paths(self, city: str) -> CityPaths:
        work_dir = self.config.work_root / city
        has_processed_ground = all(
            (work_dir / name).exists()
            for name in ("flat_ground_summary.json", "height_field.npz", "terrain.glb")
        )
        source_dir = (
            work_dir
            if self.config.flat_ground or has_processed_ground
            else self.config.source_root / city
        )
        return CityPaths(
            city=city,
            original_source_dir=self.config.source_root / city,
            source_dir=source_dir,
            work_dir=work_dir,
            osm=source_dir / "road_data" / "road_data_with_height.osm",
            raw_usd=work_dir / f"{city}_with_road_objects.usd",
            normalized_usd=work_dir / f"{city}.usd",
            outdoor_usd=work_dir / f"{city}_outdoor.usd",
            innout_base_usd=work_dir / f"{city}_innout_base.usd",
            innout_usd=work_dir / f"{city}_innout.usd",
            lane_points=work_dir / "lane_points.json",
            sidewalk_points=work_dir / "sidewalk_points.json",
            road_objects=work_dir / "road_objects_info.json",
            objaverse_objects=work_dir / "objaverse_objects.json",
            store_manifest=work_dir / "storefront_plan.json",
            innout_lane_points=work_dir / "innout_lane_points.json",
            innout_sidewalk_points=work_dir / "innout_sidewalk_points.json",
            innout_road_objects=work_dir / "innout_road_objects_info.json",
            innout_objaverse_objects=work_dir / "innout_objaverse_objects.json",
            innout_store_manifest=work_dir / "innout_storefront_plan.json",
            innout_anchor_plan=work_dir / "innout_anchor_plan.json",
            city_manifest=work_dir / "manifest.json",
        )

    def discover_cities(self) -> list[str]:
        if self.config.cities:
            return self.config.cities
        return sorted(p.name for p in self.config.source_root.iterdir() if p.is_dir() and p.name not in CITY_EXCLUDES)

    def summarize_source(self, paths: CityPaths) -> CitySummary:
        summary = CitySummary(city=paths.city, source_exists=paths.source_dir.exists())
        summary.has_roof = (paths.source_dir / "roof.glb").exists()
        summary.has_terrain = (paths.source_dir / "terrain.glb").exists()
        summary.has_objects_json = (paths.source_dir / "objects" / "amenities_with_z.json").exists() and (paths.source_dir / "objects" / "natural_with_z.json").exists()
        summary.has_osm = paths.osm.exists()
        building_dir = paths.source_dir / "buildings"
        summary.buildings = len(list(building_dir.glob("*.glb"))) if building_dir.exists() else 0
        if paths.osm.exists():
            osm_summary = OsmIndex(paths.osm).summary()
            summary.amenity_total = osm_summary["amenity_total"]
            summary.commercial_total = osm_summary["commercial_total"]
            summary.highway_ways = osm_summary["highway_ways"]
        for key, ok in {
            "roof.glb": summary.has_roof,
            "terrain.glb": summary.has_terrain,
            "objects json": summary.has_objects_json,
            "road_data_with_height.osm": summary.has_osm,
        }.items():
            if not ok:
                summary.warnings.append(f"missing {key}")
        return summary

    def sync_flat_height_field_to_source(self, paths: CityPaths) -> dict[str, Any] | None:
        if self.config.dry_run or not self.config.flat_ground_copy_height_field_to_source:
            return None
        processed_height = paths.work_dir / "height_field.npz"
        source_height = paths.original_source_dir / "height_field.npz"
        if not processed_height.exists():
            return {
                "stage": "flat-ground-height-field-source-sync",
                "skipped": "missing processed height_field",
                "processed_height_field": str(processed_height),
            }
        if source_height.exists() and filecmp.cmp(processed_height, source_height, shallow=False):
            return {
                "stage": "flat-ground-height-field-source-sync",
                "skipped": "already identical",
                "source_height_field": str(source_height),
                "processed_height_field": str(processed_height),
            }
        ensure_dir(source_height.parent)
        record: dict[str, Any] = {
            "stage": "flat-ground-height-field-source-sync",
            "source_height_field": str(source_height),
            "processed_height_field": str(processed_height),
        }
        if source_height.exists():
            backup = source_height.with_name(f"{source_height.stem}.bak_flat_ground_{now_stamp()}{source_height.suffix}")
            shutil.copy2(source_height, backup)
            record["backup"] = str(backup)
        shutil.copy2(processed_height, source_height)
        return record

    def copy_osm_to_work_root(self, paths: CityPaths) -> dict[str, Any] | None:
        if self.config.dry_run or not paths.osm.exists():
            return None
        target = paths.work_dir / "road_data_with_height.osm"
        if target.exists() and filecmp.cmp(paths.osm, target, shallow=False):
            return None
        shutil.copy2(paths.osm, target)
        return {"stage": "copy-osm-to-work-root", "source": str(paths.osm), "target": str(target)}

    def prepare_city_dir(self, paths: CityPaths) -> list[dict[str, Any]]:
        commands: list[dict[str, Any]] = []
        if self.config.flat_ground:
            ensure_dir(paths.work_dir.parent)
            summary_path = paths.work_dir / "flat_ground_summary.json"
            if summary_path.exists() and self.config.skip_existing:
                commands.append({"stage": "flat-ground", "skipped": "existing", "path": str(summary_path)})
                sync_record = self.sync_flat_height_field_to_source(paths)
                if sync_record:
                    commands.append(sync_record)
                osm_record = self.copy_osm_to_work_root(paths)
                if osm_record:
                    commands.append(osm_record)
                return commands
            script = self.config.tools_dir / "flat_ground_blender.py"
            cmd = [
                self.config.blender,
                "-b",
                "--python",
                str(script),
                "--",
                "--data-root",
                str(self.config.source_root),
                "--output-root",
                str(self.config.work_root),
                "--cities",
                paths.city,
                "--ground-height",
                self.config.flat_ground_reference,
                "--height-scale",
                str(self.config.flat_ground_height_scale),
                "--road-depth",
                str(self.config.road_lower_depth),
            ]
            if self.config.overwrite or paths.work_dir.exists():
                cmd.append("--overwrite")
            if self.config.road_lower_subdivide_cuts:
                cmd.extend(["--terrain-subdivide-cuts", str(self.config.road_lower_subdivide_cuts)])
            if self.config.flat_ground_road_boundary_cuts:
                cmd.append("--road-boundary-cuts")
            log = self.config.log_root / f"{paths.city}_flat_ground.log"
            commands.append({"stage": "flat-ground", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
            if not self.config.dry_run and not summary_path.exists():
                raise RuntimeError(f"flat-ground preprocessing did not produce {summary_path}; see {log}")
            sync_record = self.sync_flat_height_field_to_source(paths)
            if sync_record:
                commands.append(sync_record)
            osm_record = self.copy_osm_to_work_root(paths)
            if osm_record:
                commands.append(osm_record)
            return commands
        ensure_dir(paths.work_dir)
        if paths.osm.exists():
            shutil.copy2(paths.osm, paths.work_dir / "road_data_with_height.osm")
        return commands

    def generate_raw_usd(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "raw-usd"}:
            return
        raw_usd_error = usd_file_error(paths.raw_usd)
        if raw_usd_error is None and self.config.skip_existing:
            self.apply_canonical_city_textures(paths, summary)
            if not self.config.dry_run:
                shutil.copy2(paths.raw_usd, paths.normalized_usd)
                require_valid_usd(paths.normalized_usd)
            summary.commands.append({"stage": "raw-usd", "skipped": "existing", "path": str(paths.raw_usd)})
            return
        if paths.raw_usd.exists() and raw_usd_error:
            summary.warnings.append(f"removing invalid raw USD ({raw_usd_error}): {paths.raw_usd}")
            if not self.config.dry_run:
                paths.raw_usd.unlink()
        script = self.config.pipeline_tools_dir / "blender_process_cities_with_roads.py"
        cmd = [
            self.config.blender, "-b", "-P", str(script), "--",
            "--data-dir", str(paths.source_dir.parent),
            "--output-dir", str(self.config.work_root),
            "--cities", paths.city,
        ]
        if self.config.skip_existing:
            cmd.append("--skip-existing")
        if self.config.lower_osm_roads and not self.config.flat_ground:
            cmd.extend([
                "--lower-osm-roads",
                "--road-lower-depth",
                str(self.config.road_lower_depth),
                "--road-lower-subdivide-cuts",
                str(self.config.road_lower_subdivide_cuts),
            ])
        elif self.config.lower_osm_roads and self.config.flat_ground:
            summary.commands.append(
                {
                    "stage": "raw-usd-road-lowering",
                    "skipped": "flat-ground preprocessing already lowered OSM road corridors",
                }
            )
        log = self.config.log_root / f"{paths.city}_raw_usd.log"
        summary.commands.append({"stage": "raw-usd", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
        if not self.config.dry_run:
            try:
                require_valid_usd(paths.raw_usd)
            except RuntimeError as exc:
                raise RuntimeError(f"raw USD generation failed validation; see {log}: {exc}") from exc
            self.apply_canonical_city_textures(paths, summary)
        if paths.raw_usd.exists() and not self.config.dry_run:
            shutil.copy2(paths.raw_usd, paths.normalized_usd)
            require_valid_usd(paths.normalized_usd)
        road_lowering_stats = paths.work_dir / f"{paths.city}_road_lowering_stats.json"
        if road_lowering_stats.exists():
            stats = read_json(road_lowering_stats, {}) or {}
            summary.outputs["road_lowering_stats"] = str(road_lowering_stats)
            summary.counts["road_lowering_segments"] = stats.get("road_segments", 0)
            summary.counts["road_lowering_vertices"] = stats.get("total_lowered_vertices", 0)
            summary.counts["road_lowering_depth"] = stats.get("depth", 0.0)

    def apply_canonical_city_textures(
        self,
        paths: CityPaths,
        summary: CitySummary,
        target_scenes: dict[str, Path] | None = None,
    ) -> None:
        source_root = self.config.city_texture_source_root or self.config.data_root / "vc_full_generated"
        source_dir = source_root / paths.city / "textures"
        source_scene = source_root / paths.city / f"{paths.city}_outdoor.usd"
        destination_dir = paths.work_dir / "textures"
        png_files = sorted(source_dir.glob("Material_Ground*.png"))
        if len(png_files) != 64:
            raise RuntimeError(
                f"canonical ground textures incomplete for {paths.city}: "
                f"png={len(png_files)}/64 source={source_dir}"
            )
        texture_map = load_canonical_texture_map(source_scene)
        ground_refs = {
            texture.source_path.name
            for texture in texture_map.values()
            if texture.source_path.name.startswith("Material_Ground")
        }
        if len(ground_refs) != 64:
            raise RuntimeError(
                f"canonical city scene has {len(ground_refs)}/64 ground DDS refs: {source_scene}"
            )
        targets = target_scenes or {"raw": paths.raw_usd}
        existing_targets = {label: scene for label, scene in targets.items() if scene.exists()}
        if not self.config.dry_run:
            ensure_dir(destination_dir)
            for source in png_files:
                destination = destination_dir / source.name
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
            copy_result = copy_canonical_texture_files(texture_map, destination_dir)
            rewrite_results = {
                label: rewrite_city_texture_refs(scene, texture_map, destination_dir)
                for label, scene in existing_targets.items()
            }
        else:
            copy_result = {"files": len({item.source_path for item in texture_map.values()})}
            rewrite_results = {
                label: {"texture_refs": len(texture_map), "rewritten": 0, "unchanged": 0}
                for label in targets
            }
        summary.commands.append(
            {
                "stage": "canonical-city-textures",
                "source": str(source_dir),
                "source_scene": str(source_scene),
                "destination": str(destination_dir),
                "ground_png_count": len(png_files),
                "ground_dds_count": len(ground_refs),
                "city_dds_count": copy_result["files"],
                "targets": {
                    label: {
                        "scene": str(targets[label]),
                        **result,
                    }
                    for label, result in rewrite_results.items()
                },
            }
        )

    def apply_canonical_city_textures_to_composed(
        self,
        paths: CityPaths,
        summary: CitySummary,
    ) -> None:
        if self.config.stage not in {"all", "compose", "textures"}:
            return
        targets: dict[str, Path] = {}
        if self.config.variant in {"outdoor", "both"}:
            targets["outdoor"] = paths.outdoor_usd
        if self.config.variant in {"innout", "both"}:
            targets["innout_base"] = paths.innout_base_usd
            targets["innout"] = paths.innout_usd
        self.apply_canonical_city_textures(paths, summary, targets)

    def generate_points_and_road_plan(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "points", "plan", "compose", "validate", "compare"}:
            return
        if not paths.osm.exists():
            summary.warnings.append("cannot sample points without OSM")
            return
        osm = OsmIndex(paths.osm)
        terrain = TerrainHeightQuery(paths.source_dir)
        rng = random.Random(self.config.seed + sum(ord(ch) for ch in paths.city))
        lane_segments = PointSampler.build_segments(osm, "lane")
        sidewalk_segments = PointSampler.build_explicit_sidewalk_segments(osm)
        sidewalk_source = "osm_explicit_sidewalk"
        lane_points = PointSampler.sample(lane_segments, self.config.lane_points, rng)
        sidewalk_points = PointSampler.sample(sidewalk_segments, self.config.sidewalk_points, rng)
        for point in lane_points + sidewalk_points:
            point["height"] = terrain.height_at(
                float(point["utm_x"]),
                float(point["utm_y"]),
                float(point.get("height", 0.0)),
            )
        write_json(paths.lane_points, {"city": paths.city, "input": str(paths.osm), "lane_segments": len(lane_segments), "points": lane_points})
        write_json(
            paths.sidewalk_points,
            {
                "city": paths.city,
                "input": str(paths.osm),
                "sidewalk_segments": len(sidewalk_segments),
                "sidewalk_source": sidewalk_source,
                "points": sidewalk_points,
            },
        )
        road_plan = self.road_planner.plan(paths.city, osm, terrain)
        write_json(paths.road_objects, road_plan)
        summary.counts.update({
            "lane_points": len(lane_points),
            "sidewalk_points": len(sidewalk_points),
            "road_objects_planned": road_plan["total_objects"],
        })
        if not sidewalk_segments:
            summary.warnings.append("no explicit OSM sidewalk semantics; sidewalk prop placement disabled")

    def generate_outdoor_content_plans(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.variant not in {"outdoor", "both"}:
            return
        if not paths.osm.exists() or not paths.lane_points.exists() or not paths.sidewalk_points.exists():
            summary.warnings.append("cannot generate outdoor content plans without OSM sampling files")
            return
        osm = OsmIndex(paths.osm)
        terrain = TerrainHeightQuery(paths.source_dir)
        lane_points = (read_json(paths.lane_points, {}) or {}).get("points", [])
        sidewalk_points = PointSampler.filter_explicit_sidewalk_points(
            osm,
            (read_json(paths.sidewalk_points, {}) or {}).get("points", []),
        )
        store_plan = self.storefronts.plan(paths.city, osm, paths.work_dir)
        store_plan = align_storefront_plan(
            store_plan,
            paths.source_dir,
            paths.osm,
            paths.work_dir,
            storefront_target_height=self.config.storefront_target_height,
        )
        write_json(paths.store_manifest, store_plan)
        store_locs = [
            [float(item["final_location"][0]), float(item["final_location"][1]), float(item["final_location"][2])]
            for item in store_plan.get("placements", [])
            if item.get("status") == "planned"
            and item.get("facade_status") == "facade_aligned"
            and item.get("final_location")
        ]
        objaverse_plan = self.objaverse.plan(paths.city, paths.work_dir, lane_points, sidewalk_points, terrain, store_locs)
        vehicle_placed = sum(1 for item in objaverse_plan["objects"] if item.get("category") == "vehicle")
        prop_placed = sum(1 for item in objaverse_plan["objects"] if item.get("category") != "vehicle")
        expected_vehicles = min(OBJAVERSE_TARGET_VEHICLES, len(lane_points))
        expected_props = min(OBJAVERSE_TARGET_PROPS, len(sidewalk_points))
        if vehicle_placed != expected_vehicles or prop_placed != expected_props:
            raise RuntimeError(
                "Objaverse placement count mismatch: "
                f"vehicles={vehicle_placed}/{expected_vehicles}, props={prop_placed}/{expected_props}"
            )
        write_json(paths.objaverse_objects, objaverse_plan)
        summary.counts.update({
            "storefronts_planned": store_plan["planned_count"],
            "storefronts_facade_aligned": store_plan.get("facade_alignment", {}).get("status_counts", {}).get("facade_aligned", 0),
            "objaverse_objects_planned": len(objaverse_plan["objects"]),
            "objaverse_vehicles_planned": vehicle_placed,
            "objaverse_props_planned": prop_placed,
        })

    @staticmethod
    def filter_objects_to_radius(objects: list[dict[str, Any]], center_xy: list[float], radius_m: float) -> list[dict[str, Any]]:
        return [
            item
            for item in objects
            if item.get("location")
            and math.hypot(
                float(item["location"][0]) - float(center_xy[0]),
                float(item["location"][1]) - float(center_xy[1]),
            ) <= float(radius_m) + 1e-6
        ]

    @staticmethod
    def filter_objects_to_public_side(
        objects: list[dict[str, Any]],
        center_xy: list[float],
        normal_xy: list[float],
        building_footprint_xy: list[list[float]],
    ) -> list[dict[str, Any]]:
        center = np.asarray(center_xy, dtype=float)
        normal = np.asarray(normal_xy, dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        footprint = np.asarray(building_footprint_xy, dtype=float)
        filtered = []
        for item in objects:
            location = item.get("location") or []
            if len(location) < 2:
                continue
            point = np.asarray(location[:2], dtype=float)
            if float((point - center) @ normal) < 0.0:
                continue
            if len(footprint) >= 3 and point_in_polygon_xy(point, footprint):
                continue
            filtered.append(item)
        return filtered

    def generate_innout_content_plans(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.variant not in {"innout", "both"}:
            return
        if not paths.innout_anchor_plan.exists():
            summary.warnings.append("cannot generate innout content plans without a doorway anchor")
            return
        anchor = read_json(paths.innout_anchor_plan, {}) or {}
        center_xy = anchor.get("door_center_xy")
        building_glb = anchor.get("building_glb")
        if not center_xy or not building_glb:
            raise RuntimeError(f"invalid innout doorway anchor: {paths.innout_anchor_plan}")

        osm = OsmIndex(paths.osm)
        terrain = TerrainHeightQuery(paths.source_dir)
        lane_segments = PointSampler.clip_segments_to_radius(
            PointSampler.build_segments(osm, "lane"), center_xy, self.config.innout_radius_m
        )
        sidewalk_segments = PointSampler.build_explicit_sidewalk_segments(osm)
        sidewalk_source = "osm_explicit_sidewalk"
        sidewalk_segments = PointSampler.clip_segments_to_radius(
            sidewalk_segments, center_xy, self.config.innout_radius_m
        )
        outward_normal_xy = anchor["outward_normal_xy"]
        building_footprint_xy = anchor.get("building_footprint_xy") or []
        lane_segments = PointSampler.clip_segments_to_half_plane(
            lane_segments, center_xy, outward_normal_xy
        )
        sidewalk_segments = PointSampler.clip_segments_to_half_plane(
            sidewalk_segments, center_xy, outward_normal_xy
        )
        rng = random.Random(self.config.seed + sum(ord(ch) for ch in paths.city) + 104729)
        lane_sample_count = min(
            self.config.lane_points,
            max(self.config.innout_vehicles * 20, self.config.innout_vehicles),
        )
        sidewalk_sample_count = min(
            self.config.sidewalk_points,
            max(self.config.innout_props * 20, self.config.innout_props),
        )
        lane_points = PointSampler.filter_points_to_public_side(
            PointSampler.sample(lane_segments, lane_sample_count * 2, rng),
            center_xy,
            outward_normal_xy,
            building_footprint_xy,
            lane_sample_count,
        )
        sidewalk_points = PointSampler.filter_points_to_public_side(
            PointSampler.sample(sidewalk_segments, sidewalk_sample_count * 2, rng),
            center_xy,
            outward_normal_xy,
            building_footprint_xy,
            sidewalk_sample_count,
        )
        for point in lane_points + sidewalk_points:
            point["height"] = terrain.height_at(
                float(point["utm_x"]),
                float(point["utm_y"]),
                float(point.get("height", 0.0)),
            )
        spatial_filter = {
            "center_xy": [float(center_xy[0]), float(center_xy[1])],
            "radius_m": float(self.config.innout_radius_m),
            "public_side_only": True,
        }
        write_json(paths.innout_lane_points, {
            "city": paths.city,
            "input": str(paths.osm),
            "lane_segments": len(lane_segments),
            "spatial_filter": spatial_filter,
            "points": lane_points,
        })
        write_json(paths.innout_sidewalk_points, {
            "city": paths.city,
            "input": str(paths.osm),
            "sidewalk_segments": len(sidewalk_segments),
            "sidewalk_source": sidewalk_source,
            "spatial_filter": spatial_filter,
            "points": sidewalk_points,
        })
        if not sidewalk_segments:
            summary.warnings.append(
                "no explicit OSM sidewalk semantics within the innout radius; local sidewalk prop placement disabled"
            )

        local_seed = self.config.seed + sum(ord(ch) for ch in paths.city) + 130363
        local_storefronts = StorefrontPlanner(
            self.config.store_root,
            local_seed,
            self.config.innout_storefronts + INNOUT_STOREFRONT_OPENING_RESERVE,
            self.config.innout_storefronts + INNOUT_STOREFRONT_OPENING_RESERVE,
        )
        store_plan = local_storefronts.plan(
            paths.city,
            osm,
            paths.work_dir,
            center_xy,
            self.config.innout_radius_m,
        )
        store_plan["reserved_openings"] = [{
            "center_xy": [float(center_xy[0]), float(center_xy[1])],
            "tangent_xy": [float(value) for value in anchor["tangent_xy"]],
            "outward_normal_xy": [float(value) for value in anchor["outward_normal_xy"]],
            "width_m": float(anchor.get("building_opening_width") or anchor["door_width"]),
            "lateral_clearance_m": 1.0,
            "normal_clearance_m": max(2.0, float(anchor.get("door_depth") or 0.0)),
            "purpose": "innout_doorway",
        }]
        store_plan["planning_reserve"] = INNOUT_STOREFRONT_OPENING_RESERVE
        store_plan = align_storefront_plan(
            store_plan,
            paths.source_dir,
            paths.osm,
            paths.work_dir,
            max_road_distance=20.0,
            min_facing_dot=0.3,
            mesh_wall_support_tolerance=1.25,
            same_edge_spacing=1.0,
            edge_margin=0.25,
            storefront_target_height=self.config.storefront_target_height,
        )
        store_plan["min_storefronts"] = self.config.innout_storefronts
        write_json(paths.innout_store_manifest, store_plan)
        store_locs = [
            [float(item["final_location"][0]), float(item["final_location"][1]), float(item["final_location"][2])]
            for item in store_plan.get("placements", [])
            if item.get("status") == "planned"
            and item.get("facade_status") == "facade_aligned"
            and item.get("final_location")
        ]

        local_objaverse = ObjaversePlanner(self.config.objaverse_root, local_seed + 32771)
        doorway_exclusion = {
            "center_xy": [float(center_xy[0]), float(center_xy[1])],
            "tangent_xy": [float(value) for value in anchor["tangent_xy"]],
            "outward_normal_xy": [float(value) for value in anchor["outward_normal_xy"]],
            "width_m": float(anchor["door_width"]),
            "lateral_clearance_m": INNOUT_DOORWAY_LATERAL_CLEARANCE_M,
            "inward_clearance_m": INNOUT_DOORWAY_INWARD_CLEARANCE_M,
            "outward_clearance_m": INNOUT_DOORWAY_OUTWARD_CLEARANCE_M,
            "purpose": "innout_doorway",
        }
        room_exclusion = {
            "polygon_xy": innout_room_footprint(anchor),
            "clearance_m": INNOUT_ROOM_OBJECT_CLEARANCE_M,
            "purpose": "innout_room",
        }
        placement_exclusions = [doorway_exclusion, room_exclusion]
        objaverse_plan = local_objaverse.plan(
            paths.city,
            paths.work_dir,
            lane_points,
            sidewalk_points,
            terrain,
            store_locs,
            target_props=self.config.innout_props,
            target_vehicles=self.config.innout_vehicles,
            store_clearance=3.0,
            placement_exclusions=placement_exclusions,
            prioritize_sidewalk_props=True,
        )
        objaverse_plan["spatial_filter"] = spatial_filter
        vehicle_placed = sum(item.get("category") == "vehicle" for item in objaverse_plan["objects"])
        prop_placed = sum(item.get("category") != "vehicle" for item in objaverse_plan["objects"])
        expected_vehicles = min(self.config.innout_vehicles, len(lane_points))
        expected_props = min(self.config.innout_props, len(sidewalk_points))
        objaverse_plan["target_counts"] = {
            "vehicle": int(expected_vehicles),
            "prop": int(expected_props),
        }
        objaverse_plan["placement_policy"] = {
            "footprint_aware": True,
            "vehicle_clearance_m": VEHICLE_FOOTPRINT_CLEARANCE,
            "prop_clearance_m": PROP_FOOTPRINT_CLEARANCE,
            "allow_target_shortfall": True,
            "doorway_exclusion": doorway_exclusion,
            "room_exclusion": room_exclusion,
        }
        if vehicle_placed != expected_vehicles or prop_placed != expected_props:
            objaverse_plan["safety_limited_shortfall"] = {
                "vehicle": int(expected_vehicles - vehicle_placed),
                "prop": int(expected_props - prop_placed),
            }
        write_json(paths.innout_objaverse_objects, objaverse_plan)

        road_plan = read_json(paths.road_objects, {}) or {}
        local_road_objects = self.filter_objects_to_radius(
            road_plan.get("objects", []), center_xy, self.config.innout_radius_m
        )
        local_road_objects = self.filter_objects_to_public_side(
            local_road_objects,
            center_xy,
            outward_normal_xy,
            building_footprint_xy,
        )
        room_clear_road_objects = []
        for item in local_road_objects:
            asset_reference = Path(str(item.get("asset_path") or ""))
            asset_path = (
                asset_reference
                if asset_reference.is_absolute()
                else (paths.work_dir / asset_reference).resolve()
            )
            dimensions = ObjaversePlanner.actual_object_dims(asset_path, [2.0, 2.0, 4.0])
            scale = item.get("scale", 1.0)
            if isinstance(scale, (list, tuple)):
                scale_xy = [float(scale[0]), float(scale[1] if len(scale) > 1 else scale[0])]
            else:
                scale_xy = [float(scale), float(scale)]
            footprint_radius = 0.5 * math.hypot(
                float(dimensions[0]) * scale_xy[0],
                float(dimensions[1]) * scale_xy[1],
            )
            if not ObjaversePlanner.footprint_clear_of_exclusions(
                item.get("final_location") or item.get("location") or [],
                footprint_radius,
                placement_exclusions,
            ):
                continue
            room_clear_road_objects.append(
                {**item, "footprint_radius_m": footprint_radius}
            )
        local_road_objects = room_clear_road_objects
        local_road_plan = dict(road_plan)
        local_road_plan["objects"] = local_road_objects
        local_road_plan["total_objects"] = len(local_road_objects)
        local_road_plan["spatial_filter"] = spatial_filter
        write_json(paths.innout_road_objects, local_road_plan)

        facade_aligned = store_plan.get("facade_alignment", {}).get("status_counts", {}).get("facade_aligned", 0)
        summary.counts.update({
            "innout_radius_m": self.config.innout_radius_m,
            "innout_lane_points": len(lane_points),
            "innout_sidewalk_points": len(sidewalk_points),
            "innout_road_objects_planned": len(local_road_objects),
            "innout_storefronts_facade_aligned": facade_aligned,
            "innout_objaverse_vehicles_planned": vehicle_placed,
            "innout_objaverse_props_planned": prop_placed,
        })

    def compose_with_existing_merge(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "compose"}:
            return
        if self.config.variant not in {"outdoor", "both"}:
            return
        if not paths.normalized_usd.exists() and paths.raw_usd.exists():
            shutil.copy2(paths.raw_usd, paths.normalized_usd)
        script = self.config.pipeline_tools_dir / "merge_usd.py"
        cmd = [
            self.config.usd_python, str(script),
            "--scene_folder", str(self.config.work_root.parent),
            "--vc_folder", self.config.work_root.name,
            "--city", paths.city,
            "--source_suffix", "",
            "--output_suffix", "_outdoor",
            "--skip_convert",
            "--city_scale", "1.0",
        ]
        log = self.config.log_root / f"{paths.city}_merge_outdoor.log"
        if paths.normalized_usd.exists():
            summary.commands.append({"stage": "compose-outdoor", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
            if not self.config.dry_run and not paths.outdoor_usd.exists():
                raise RuntimeError(f"outdoor compose did not produce {paths.outdoor_usd}; see {log}")
            self.apply_storefront_openings_to(
                paths,
                summary,
                paths.outdoor_usd,
                paths.store_manifest,
                paths.work_dir / "storefront_openings_manifest.json",
                "outdoor",
                "generated_storefront_openings/outdoor",
            )
        else:
            summary.warnings.append("outdoor compose skipped: missing normalized source USD")

    def apply_storefront_openings_to(
        self,
        paths: CityPaths,
        summary: CitySummary,
        usd_path: Path,
        store_plan: Path,
        manifest: Path,
        label: str,
        output_subdir: str,
    ) -> None:
        if not usd_path.exists() or not store_plan.exists():
            return
        script = self.config.pipeline_tools_dir / "apply_storefront_openings.py"
        cmd = [
            self.config.usd_python,
            str(script),
            "--city",
            paths.city,
            "--work-dir",
            str(paths.work_dir),
            "--usd",
            str(usd_path),
            "--storefront-plan",
            str(store_plan),
            "--manifest",
            str(manifest),
            "--blender",
            self.config.blender,
            "--cut-helper",
            str(self.config.repo_root / "navverse_tools" / "3dmodel" / "cut_vc_building_blender.py"),
            "--output-subdir",
            output_subdir,
        ]
        log = self.config.log_root / f"{paths.city}_{label}_storefront_openings.log"
        summary.commands.append({"stage": f"{label}-storefront-openings", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
        result = read_json(manifest, {}) or {}
        summary.outputs[f"{label}_storefront_openings_manifest"] = str(manifest)
        summary.counts[f"{label}_storefront_opening_buildings"] = result.get("cut_building_count", 0)

    def compose_innout_base(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "compose"}:
            return
        if self.config.variant not in {"innout", "both"}:
            return
        required = [
            paths.normalized_usd,
            paths.innout_lane_points,
            paths.innout_sidewalk_points,
            paths.innout_road_objects,
            paths.innout_store_manifest,
            paths.innout_objaverse_objects,
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            summary.warnings.append(f"innout base compose skipped: missing {', '.join(path.name for path in missing)}")
            return
        newest_input = max(path.stat().st_mtime for path in required)
        if paths.innout_base_usd.exists() and self.config.skip_existing and paths.innout_base_usd.stat().st_mtime >= newest_input:
            summary.outputs["innout_base"] = str(paths.innout_base_usd)
            return
        script = self.config.pipeline_tools_dir / "merge_usd.py"
        cmd = [
            self.config.usd_python,
            str(script),
            "--scene_folder", str(self.config.work_root.parent),
            "--vc_folder", self.config.work_root.name,
            "--city", paths.city,
            "--source_suffix", "",
            "--output_suffix", "_innout_base",
            "--skip_convert",
            "--city_scale", "1.0",
            "--lane-points", str(paths.innout_lane_points),
            "--sidewalk-points", str(paths.innout_sidewalk_points),
            "--road-objects", str(paths.innout_road_objects),
            "--storefront-plan", str(paths.innout_store_manifest),
            "--storefront-merge-summary", str(paths.work_dir / "innout_storefront_merge_summary.json"),
            "--objaverse-objects", str(paths.innout_objaverse_objects),
        ]
        log = self.config.log_root / f"{paths.city}_merge_innout_base.log"
        summary.commands.append({"stage": "compose-innout-base", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
        if not self.config.dry_run and not paths.innout_base_usd.exists():
            raise RuntimeError(f"innout base compose did not produce {paths.innout_base_usd}; see {log}")
        self.apply_storefront_openings_to(
            paths,
            summary,
            paths.innout_base_usd,
            paths.innout_store_manifest,
            paths.work_dir / "innout_storefront_openings_manifest.json",
            "innout",
            "generated_storefront_openings/innout",
        )
        summary.outputs["innout_base"] = str(paths.innout_base_usd)

    def score_innout_geometry(self, paths: CityPaths, summary: CitySummary) -> dict[str, Any]:
        score_path = paths.work_dir / "building_geometry_score.json"
        if score_path.exists() and self.config.skip_existing:
            existing = read_json(score_path, {}) or {}
            context = existing.get("selection_context") or {}
            if context.get("mode") == "pre_content" and context.get("storefront_anchor_count") == 0:
                return existing
        script = self.config.tools_dir / "geometry_score_blender.py"
        cmd = [
            self.config.blender, "-b", "--python", str(script), "--",
            "--city", paths.city,
            "--city-dir", str(paths.source_dir),
            "--work-dir", str(paths.work_dir),
            "--sidewalk-points", str(paths.sidewalk_points),
            "--grscene-root", str(self.config.grscene_root),
            "--output", str(score_path),
            "--max-buildings", str(self.config.max_building_scan),
        ]
        preferred_grscene = self.grscene_assignments.get(paths.city)
        if preferred_grscene:
            cmd.extend(["--preferred-grscene", preferred_grscene])
        log = self.config.log_root / f"{paths.city}_geometry_score.log"
        summary.commands.append({"stage": "geometry-score", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
        return read_json(score_path, {}) or {}

    def plan_innout(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "compose", "plan"}:
            return
        if self.config.variant not in {"innout", "both"}:
            return
        score = self.score_innout_geometry(paths, summary)
        selected = score.get("selected")
        selected_grscene = (selected or {}).get("selected_grscene")
        innout_plan = {
            "city": paths.city,
            "status": "planned" if selected and selected_grscene else "blocked",
            "selected_building": selected,
            "selected_grscene": selected_grscene,
            "assigned_grscene": self.grscene_assignments.get(paths.city),
            "geometry_score_path": str(paths.work_dir / "building_geometry_score.json"),
            "method": "automatic vc_full geometry scoring + facade selection; vc_plus cut_buildings is reference-only and not used for target selection",
        }
        write_json(paths.work_dir / "innout_plan.json", innout_plan)
        summary.counts["geometry_buildings_scanned"] = score.get("building_count_scanned", 0)
        summary.counts["innout_anchor_planned"] = 1 if selected else 0
        if not selected:
            summary.warnings.append("innout blocked: no geometry-scored building candidate")
        if not selected_grscene:
            summary.warnings.append("innout blocked: no GRScenes commercial scene")

    def innout_compose_command(
        self,
        paths: CityPaths,
        base_usd: Path,
        plan_output: Path,
        storefront_plan: Path,
        *,
        plan_only: bool = False,
    ) -> list[str]:
        helper = self.config.pipeline_tools_dir / "compose_vc_full_innout.py"
        cmd = [
            self.config.usd_python,
            str(helper),
            "--city", paths.city,
            "--data-root", str(self.config.data_root),
            "--work-dir", str(paths.work_dir),
            "--base-usd", str(base_usd),
            "--score", str(paths.work_dir / "building_geometry_score.json"),
            "--output", str(paths.innout_usd),
            "--store-root", str(self.config.store_root),
            "--storefront-plan", str(storefront_plan),
            "--storefront-openings-manifest", str(paths.work_dir / "innout_storefront_openings_manifest.json"),
            "--source-dataset", paths.source_dir.parent.name,
            "--qa-root", str(self.config.qa_root),
            "--blender", self.config.blender,
            "--cut-helper", str(self.config.repo_root / "navverse_tools" / "3dmodel" / "cut_vc_building_blender.py"),
            "--render-helper", str(self.config.repo_root / "navverse_tools" / "3dmodel" / "render_vc_innout_qa_blender.py"),
            "--plan-output", str(plan_output),
            "--storefront-target-height", str(self.config.storefront_target_height),
            "--content-radius-m", str(self.config.innout_radius_m),
            "--min-local-storefront-capacity", str(self.config.innout_storefronts),
        ]
        if plan_only:
            cmd.extend(["--plan-only", "--skip-qa-render"])
        else:
            cmd.extend(["--anchor-plan", str(paths.innout_anchor_plan)])
            if self.config.skip_qa_render:
                cmd.append("--skip-qa-render")
        return cmd

    def plan_innout_anchor(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "compose", "plan"}:
            return
        if self.config.variant not in {"innout", "both"}:
            return
        if not paths.normalized_usd.exists():
            summary.warnings.append("innout doorway anchor skipped: missing normalized city USD")
            return
        if paths.innout_anchor_plan.exists() and self.config.skip_existing:
            existing = read_json(paths.innout_anchor_plan, {}) or {}
            if (
                float(existing.get("content_radius_m") or -1.0) == float(self.config.innout_radius_m)
                and Path(existing.get("base_usd", "")).resolve() == paths.normalized_usd.resolve()
            ):
                summary.outputs["innout_anchor_plan"] = str(paths.innout_anchor_plan)
                return
        cmd = self.innout_compose_command(
            paths,
            paths.normalized_usd,
            paths.innout_anchor_plan,
            paths.innout_store_manifest,
            plan_only=True,
        )
        log = self.config.log_root / f"{paths.city}_innout_anchor.log"
        summary.commands.append({"stage": "innout-anchor", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
        if paths.innout_anchor_plan.exists():
            summary.outputs["innout_anchor_plan"] = str(paths.innout_anchor_plan)


    def compose_storefront_cut_layer(self, paths: CityPaths, summary: CitySummary) -> None:
        if not self.config.write_cut_smoke:
            return
        if self.config.stage not in {"all", "compose"}:
            return
        if self.config.variant not in {"outdoor", "innout", "both"}:
            return
        score_path = paths.work_dir / "building_geometry_score.json"
        if not score_path.exists():
            score = self.score_innout_geometry(paths, summary)
            if not score.get("selected"):
                summary.warnings.append("storefront cut skipped: no geometry-scored building candidate")
                return
        script = self.config.tools_dir / "boolean_cut_building_blender.py"
        output = paths.work_dir / f"{paths.city}_storefront_cut.usd"
        manifest = paths.work_dir / "storefront_cut_manifest.json"
        if output.exists() and manifest.exists() and self.config.skip_existing:
            summary.outputs["storefront_cut"] = str(output)
            summary.commands.append({"stage": "storefront-cut", "skipped": "existing", "path": str(output)})
            return
        cmd = [
            self.config.blender, "-b", "--python", str(script), "--",
            "--score", str(score_path),
            "--output", str(output),
            "--manifest", str(manifest),
        ]
        log = self.config.log_root / f"{paths.city}_storefront_cut.log"
        summary.commands.append({"stage": "storefront-cut", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
        cut_manifest = read_json(manifest, {}) or {}
        summary.outputs["storefront_cut"] = str(output)
        summary.counts["storefront_cut_building_meshes"] = len(cut_manifest.get("cut_meshes", []))
        summary.counts["storefront_cut_placements"] = cut_manifest.get("storefront_placement_count", 0)
        summary.counts["storefront_imported_objects"] = cut_manifest.get("imported_storefront_object_count", 0)

    def compose_innout_scene(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "compose"}:
            return
        if self.config.variant not in {"innout", "both"}:
            return
        score_path = paths.work_dir / "building_geometry_score.json"
        plan_path = paths.work_dir / "innout_final_plan.json"
        if paths.innout_usd.exists() and plan_path.exists() and self.config.skip_existing:
            existing = read_json(plan_path, {}) or {}
            required_inputs = [paths.innout_base_usd, paths.innout_anchor_plan, paths.innout_store_manifest]
            inputs_are_older = all(
                path.exists() and path.stat().st_mtime <= paths.innout_usd.stat().st_mtime
                for path in required_inputs
            )
            if (
                inputs_are_older
                and float(existing.get("content_radius_m") or -1.0)
                == float(self.config.innout_radius_m)
                and Path(existing.get("base_usd", "")).resolve() == paths.innout_base_usd.resolve()
                and existing.get("door_cut_source_glb")
                and existing.get("door_cut_mode") == "boolean_tunnel"
                and usd_has_active_prims(
                    paths.innout_usd,
                    ["/World/store", "/World/Objaverse", "/World/start_result_navigation"],
                )
            ):
                summary.outputs["innout"] = str(paths.innout_usd)
                summary.outputs["innout_plan"] = str(plan_path)
                summary.outputs["qa_dir"] = str(self.config.qa_root / paths.city)
                summary.commands.append({"stage": "compose-innout", "skipped": "existing", "path": str(paths.innout_usd)})
                return
        if not paths.innout_base_usd.exists():
            summary.warnings.append("innout compose skipped: missing local innout base USD")
            return
        if not score_path.exists():
            score = self.score_innout_geometry(paths, summary)
            if not score.get("selected"):
                summary.warnings.append("innout compose skipped: no geometry-scored building candidate")
                return

        cmd = self.innout_compose_command(
            paths,
            paths.innout_base_usd,
            plan_path,
            paths.innout_store_manifest,
        )
        log = self.config.log_root / f"{paths.city}_compose_innout.log"
        summary.commands.append({"stage": "compose-innout", **run_logged(cmd, log, self.config.dry_run, self.config.repo_root)})
        if paths.innout_usd.exists():
            summary.outputs["innout"] = str(paths.innout_usd)
        if plan_path.exists():
            plan = read_json(plan_path, {}) or {}
            summary.outputs["innout_plan"] = str(plan_path)
            summary.outputs["qa_dir"] = plan.get("qa_dir", str(self.config.qa_root / paths.city))
            summary.counts["innout_target_match_distance"] = plan.get("target_match_distance")

    def render_scene_photos(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "compose", "validate"}:
            return
        if self.config.skip_scene_photo_render:
            summary.commands.append({"stage": "scene-photo-render", "skipped": "disabled"})
            return
        if not paths.outdoor_usd.exists():
            summary.warnings.append("scene photo render skipped: missing outdoor USD")
            return
        if not paths.store_manifest.exists():
            summary.warnings.append("scene photo render skipped: missing storefront plan")
            return
        store_dir = self.config.scene_data_root / paths.city / "store"
        store_script = self.config.render_tools_dir / "render_isaac_storefront_photos.py"
        store_cmd = [
            self.config.usd_python,
            str(store_script),
            "--usd", str(paths.outdoor_usd),
            "--storefront-plan", str(paths.store_manifest),
            "--output-dir", str(store_dir),
            "--max-stores", "0",
            "--resolution", "960", "540",
            "--warmup-frames", "4",
            "--camera-distances", "6,8,12,18,24",
            "--camera-lateral-offsets=-8,-4,0,4,8",
        ]
        store_log = self.config.log_root / f"{paths.city}_storefront_isaac_photos.log"
        summary.commands.append({"stage": "storefront-isaac-photo-render", **run_logged(store_cmd, store_log, self.config.dry_run, self.config.repo_root)})

        innout_dir = self.config.scene_data_root / paths.city / "innout"
        innout_plan = paths.work_dir / "innout_final_plan.json"
        if self.config.variant in {"innout", "both"}:
            if paths.innout_usd.exists() and innout_plan.exists():
                exit_script = self.config.render_tools_dir / "render_isaac_exit_views.py"
                exit_cmd = [
                    self.config.usd_python,
                    str(exit_script),
                    "--usd", str(paths.innout_usd),
                    "--plan", str(innout_plan),
                    "--output-dir", str(innout_dir),
                    "--resolution", "960", "540",
                    "--warmup-frames", "4",
                ]
                exit_log = self.config.log_root / f"{paths.city}_innout_isaac_photos.log"
                summary.commands.append({"stage": "innout-isaac-photo-render", **run_logged(exit_cmd, exit_log, self.config.dry_run, self.config.repo_root)})
            else:
                summary.warnings.append("innout scene photo render skipped: missing innout USD or plan")
        summary.outputs["store_photos_dir"] = str(store_dir)
        summary.outputs["innout_photos_dir"] = str(innout_dir)
        summary.counts["store_photos"] = len(list(store_dir.glob("*.png"))) if store_dir.exists() else 0
        summary.counts["innout_photos"] = len(list(innout_dir.glob("*.png"))) if innout_dir.exists() else 0

    def storefront_render_failures(self, paths: CityPaths) -> list[dict[str, Any]]:
        summary_path = self.config.scene_data_root / paths.city / "store" / "isaac_storefront_views.json"
        records = read_json(summary_path, []) or []
        failures = []
        for record in records:
            if record.get("qa_status") == "pass":
                continue
            reasons = set(record.get("qa_reasons") or [])
            if reasons & STOREFRONT_RENDER_REJECT_REASONS:
                failures.append(record)
        return failures

    def repair_storefront_render_failures(self, paths: CityPaths, summary: CitySummary) -> bool:
        if self.config.dry_run or self.config.skip_scene_photo_render:
            return False
        if self.config.stage not in {"all", "compose"}:
            return False
        repaired = False
        initial_failures: int | None = None
        for iteration in range(1, MAX_STOREFRONT_RENDER_REPAIR_ITERATIONS + 1):
            failures = self.storefront_render_failures(paths)
            if initial_failures is None:
                initial_failures = len(failures)
            if not failures:
                break
            plan = read_json(paths.store_manifest, {}) or {}
            placements = plan.get("placements", [])
            if not placements:
                return repaired
            failed_by_index = {int(row["index"]): row for row in failures if row.get("index") is not None}
            rejected_edges = set(plan.get("render_rejected_edges") or [])
            rejected_count = 0
            for item in placements:
                item.pop("render_rejected", None)
                item.pop("render_reject_reasons", None)
                item.pop("render_reject_iteration", None)
                item.pop("render_reject_metrics", None)
            for item in placements:
                plan_index = item.get("index")
                if plan_index is None:
                    continue
                failure = failed_by_index.get(int(plan_index))
                if failure is None:
                    continue
                if item.get("building_name") is None or item.get("building_edge_index") is None:
                    continue
                edge_key = f"{item.get('building_name')}:{item.get('building_edge_index')}"
                item["render_rejected"] = True
                item["render_reject_reasons"] = failure.get("qa_reasons") or []
                item["render_reject_iteration"] = iteration
                item["render_reject_metrics"] = {
                    "visibility_score": failure.get("visibility_score"),
                    "mask_area_ratio": failure.get("mask_area_ratio"),
                    "camera_distance": failure.get("camera_distance"),
                    "camera_lateral_offset": failure.get("camera_lateral_offset"),
                }
                rejected_edges.add(edge_key)
                rejected_count += 1
            if rejected_count == 0:
                summary.warnings.append(f"storefront render repair skipped: {len(failures)} failures had no rejectable facade edge")
                return repaired

            plan["render_rejected_edges"] = sorted(rejected_edges)
            realigned = align_storefront_plan(
                plan,
                paths.source_dir,
                paths.osm,
                paths.work_dir,
                storefront_target_height=self.config.storefront_target_height,
            )
            realigned["render_rejected_edges"] = sorted(rejected_edges)
            realigned["render_feedback_repair"] = {
                "iteration": iteration,
                "failed_render_count": len(failures),
                "rejected_placement_count": rejected_count,
                "rejected_edge_count": len(rejected_edges),
            }
            write_json(paths.store_manifest, realigned)
            summary.warnings.append(
                f"storefront render repair pass {iteration}: rejected {rejected_count} placements on {len(rejected_edges)} facade edges and regenerated storefront plan"
            )
            repaired = True

            self.compose_with_existing_merge(paths, summary)
            self.plan_innout(paths, summary)
            self.compose_innout_scene(paths, summary)
            self.optimize_textures(paths, summary)
            self.adjust_storefront_materials(paths, summary)
            self.render_scene_photos(paths, summary)

        remaining = self.storefront_render_failures(paths)
        if initial_failures:
            summary.counts["storefront_render_failures_before_repair"] = initial_failures
        summary.counts["storefront_render_failures_after_repair"] = len(remaining)
        if repaired:
            final_plan = read_json(paths.store_manifest, {}) or {}
            summary.counts["storefront_render_rejected_edges"] = len(final_plan.get("render_rejected_edges") or [])
        if remaining:
            sample = [row.get("index") for row in remaining[:12]]
            summary.warnings.append(f"storefront render repair left {len(remaining)} failures; sample indices: {sample}")
        return repaired

    def optimize_textures(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "textures"}:
            return
        if self.config.skip_texture_optimization:
            summary.commands.append({"stage": "texture-optimization", "skipped": "disabled"})
            return

        helper = self.config.texture_tools_dir / "process_material.py"
        if not helper.exists():
            raise FileNotFoundError(helper)
        texture_scene_root = self.config.texture_scene_root or self.config.data_root

        layer_candidates: dict[str, Path] = {}
        if self.config.variant in {"outdoor", "both"}:
            layer_candidates["outdoor"] = paths.outdoor_usd
        if self.config.variant in {"innout", "both"}:
            layer_candidates["innout"] = paths.innout_usd
            layer_candidates["storefront_cut"] = paths.work_dir / f"{paths.city}_storefront_cut.usd"

        existing_layers = {label: path for label, path in layer_candidates.items() if path.exists()}
        if not existing_layers:
            summary.warnings.append("texture optimization skipped: no composed final layer exists")
            return

        for label, scene_usd in existing_layers.items():
            material_json = paths.work_dir / f"{paths.city}_{label}_{self.config.texture_material_op}.json"
            texture_json = paths.work_dir / f"{paths.city}_{label}_{self.config.texture_variant}.json"
            if self.config.skip_existing and material_json.exists() and texture_json.exists():
                summary.commands.append(
                    {
                        "stage": "texture-optimization",
                        "label": label,
                        "skipped": "existing",
                        "material_json": str(material_json),
                        "texture_json": str(texture_json),
                    }
                )
                continue

            material_cmd = [
                self.config.usd_python,
                str(helper),
                "--op",
                self.config.texture_material_op,
                "--scene_usd",
                str(scene_usd),
                "--scene_root",
                str(texture_scene_root),
                "--output_json",
                str(material_json),
            ]
            texture_cmd = [
                self.config.usd_python,
                str(helper),
                "--op",
                "convert",
                "--variant",
                self.config.texture_variant,
                "--scene_usd",
                str(scene_usd),
                "--scene_root",
                str(texture_scene_root),
                "--output_json",
                str(texture_json),
            ]
            if self.config.compressonator:
                texture_cmd.extend(["--compressonator", self.config.compressonator])

            log = self.config.log_root / f"{paths.city}_{label}_texture_optimization.log"
            summary.commands.append(
                {
                    "stage": "texture-materials",
                    "label": label,
                    **run_logged(material_cmd, log, self.config.dry_run, self.config.repo_root),
                }
            )
            summary.commands.append(
                {
                    "stage": "texture-convert",
                    "label": label,
                    **run_logged(texture_cmd, log, self.config.dry_run, self.config.repo_root),
                }
            )
            summary.outputs[f"{label}_material_json"] = str(material_json)
            summary.outputs[f"{label}_texture_json"] = str(texture_json)

    def adjust_storefront_materials(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "compose", "textures", "validate"}:
            return
        helper = self.config.pipeline_tools_dir / "adjust_storefront_materials.py"
        if not helper.exists():
            raise FileNotFoundError(helper)
        layer_candidates: dict[str, Path] = {}
        if self.config.variant in {"outdoor", "both"}:
            layer_candidates["outdoor"] = paths.outdoor_usd
        if self.config.variant in {"innout", "both"}:
            layer_candidates["innout"] = paths.innout_usd
        for label, scene_usd in layer_candidates.items():
            if not scene_usd.exists():
                continue
            material_json = paths.work_dir / f"{paths.city}_{label}_storefront_material_adjust.json"
            cmd = [
                self.config.usd_python,
                str(helper),
                "--usd", str(scene_usd),
                "--store-root", str(self.config.store_root),
                "--summary", str(material_json),
            ]
            log = self.config.log_root / f"{paths.city}_{label}_storefront_material_adjust.log"
            summary.commands.append(
                {
                    "stage": "storefront-material-adjust",
                    "label": label,
                    **run_logged(cmd, log, self.config.dry_run, self.config.repo_root),
                }
            )
            if material_json.exists():
                data = read_json(material_json, {}) or {}
                records = data.get("records", [])
                summary.outputs[f"{label}_storefront_material_adjust_json"] = str(material_json)
                summary.counts[f"{label}_storefront_material_adjusted"] = sum(
                    1 for row in records if row.get("status") == "adjusted"
                )

    def finalize_texture_bindings(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "compose", "textures", "validate"}:
            return
        if self.config.skip_texture_optimization:
            summary.commands.append({"stage": "texture-finalize", "skipped": "disabled"})
            return

        helper = self.config.pipeline_tools_dir / "finalize_texture_bindings.py"
        if not helper.exists():
            raise FileNotFoundError(helper)
        layer_candidates: list[Path] = []
        if self.config.variant in {"outdoor", "both"} and paths.outdoor_usd.exists():
            layer_candidates.append(paths.outdoor_usd)
        if self.config.variant in {"innout", "both"} and paths.innout_usd.exists():
            layer_candidates.append(paths.innout_usd)
        if not layer_candidates:
            summary.warnings.append("final texture binding pass skipped: no final scene exists")
            return

        output_json = paths.work_dir / f"{paths.city}_final_texture_bindings.json"
        cmd = [
            self.config.usd_python,
            str(helper),
            "--usd",
            *(str(path) for path in layer_candidates),
            "--output-json",
            str(output_json),
            "--workers",
            "2",
        ]
        if self.config.compressonator:
            cmd.extend(["--compressonator", self.config.compressonator])
        log = self.config.log_root / f"{paths.city}_final_texture_bindings.log"
        summary.commands.append(
            {
                "stage": "texture-finalize",
                **run_logged(cmd, log, self.config.dry_run, self.config.repo_root),
            }
        )
        if output_json.exists():
            data = read_json(output_json, {}) or {}
            summary.outputs["final_texture_bindings_json"] = str(output_json)
            summary.counts["final_texture_bindings"] = int(data.get("binding_count", 0))
            summary.counts["final_texture_replacements"] = int(data.get("replacement_count", 0))

    def validate_and_compare(self, paths: CityPaths, summary: CitySummary) -> None:
        if self.config.stage not in {"all", "validate", "compare", "textures"}:
            return
        for label, path in {"raw": paths.raw_usd, "source": paths.normalized_usd, "outdoor": paths.outdoor_usd, "innout": paths.innout_usd, "storefront_cut": paths.work_dir / f"{paths.city}_storefront_cut.usd"}.items():
            if path.exists():
                summary.outputs[label] = str(path)
                summary.counts[f"{label}_bytes"] = path.stat().st_size
        if self.config.compare_vc_plus:
            write_json(paths.work_dir / "vc_plus_reference.json", self.compare.reference(paths.city))

    def process_city(self, city: str) -> CitySummary:
        paths = self.city_paths(city)
        prepare_commands = self.prepare_city_dir(paths)
        summary = self.summarize_source(paths)
        summary.commands.extend(prepare_commands)
        self.generate_raw_usd(paths, summary)
        self.generate_points_and_road_plan(paths, summary)
        self.plan_innout(paths, summary)
        self.plan_innout_anchor(paths, summary)
        self.generate_outdoor_content_plans(paths, summary)
        self.generate_innout_content_plans(paths, summary)
        self.compose_with_existing_merge(paths, summary)
        self.compose_innout_base(paths, summary)
        self.compose_storefront_cut_layer(paths, summary)
        self.compose_innout_scene(paths, summary)
        self.apply_canonical_city_textures_to_composed(paths, summary)
        self.optimize_textures(paths, summary)
        self.adjust_storefront_materials(paths, summary)
        self.render_scene_photos(paths, summary)
        self.repair_storefront_render_failures(paths, summary)
        self.finalize_texture_bindings(paths, summary)
        self.validate_and_compare(paths, summary)
        write_json(paths.city_manifest, asdict(summary))
        return summary

    def run(self) -> dict[str, Any]:
        ensure_dir(self.config.work_root)
        ensure_dir(self.config.log_root)
        self.road_planner.prepare_assets()
        road_asset_conversion = self.convert_road_assets()
        cities = self.discover_cities()
        assignment_cities = self.config.assignment_cities or cities
        self.configure_grscene_assignments(assignment_cities)
        result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.config).items()}, "road_asset_conversion": road_asset_conversion, "cities": [], "errors": []}
        result["grscene_assignments"] = self.grscene_assignments
        for city in cities:
            try:
                summary = self.process_city(city)
                result["cities"].append(asdict(summary))
            except Exception as exc:
                result["errors"].append({"city": city, "error": repr(exc)})
        manifest = self.config.run_manifest
        write_json(manifest, result)
        result["manifest"] = str(manifest)
        print(json.dumps({"manifest": str(manifest), "cities": len(result["cities"]), "errors": result["errors"]}, indent=2, sort_keys=True))
        return result


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Generate VC Full outdoor/innout USD variants with manifests.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-dataset", default="vc_full")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--grscene-root", type=Path, default=DEFAULT_GRSCENE_ROOT)
    parser.add_argument("--objaverse-root", type=Path, default=DEFAULT_OBJAVERSE_ROOT)
    parser.add_argument("--vc-plus-root", type=Path, default=DEFAULT_DATA_ROOT / "vc_apr14")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--variant", choices=["outdoor", "innout", "both"], default="both")
    parser.add_argument("--stage", choices=["all", "raw-usd", "points", "plan", "compose", "textures", "validate", "compare"], default="plan")
    parser.add_argument("--cities", nargs="*", default=[])
    parser.add_argument("--assignment-cities", nargs="*", default=[], help="Full city list used for deterministic GRScene assignment when running city shards in parallel.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lane-points", type=int, default=10000)
    parser.add_argument("--sidewalk-points", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--usd-python", default=sys.executable)
    parser.add_argument("--compare-vc-plus", action="store_true", default=True)
    parser.add_argument("--no-compare-vc-plus", dest="compare_vc_plus", action="store_false")
    parser.add_argument("--max-storefronts", type=int, default=None)
    parser.add_argument("--min-storefronts", type=int, default=50, help="Minimum storefront count per city; synthetic roadside storefronts are added when OSM commercial POIs are sparse.")
    parser.add_argument("--max-building-scan", type=int, default=0, help="Limit building GLBs scanned by Blender for quick smoke tests; 0 scans all.")
    parser.add_argument("--texture-material-op", choices=["merge_materials", "merge_materials_prune", "merge_materials_deactivate"], default="merge_materials_prune")
    parser.add_argument("--texture-variant", choices=["a_bc7_nomip", "b_bc7_mip", "c_bc7_mip_skip64"], default="a_bc7_nomip")
    parser.add_argument("--texture-scene-root", type=Path, default=None, help="Root containing textures to rewrite; defaults to --data-root so shared assets are included.")
    parser.add_argument(
        "--city-texture-source-root",
        type=Path,
        default=None,
        help="Canonical per-city upscaled city DDS and Nano Banana ground PNG root; defaults to <data-root>/vc_full_generated.",
    )
    parser.add_argument("--compressonator", default="compressonatorcli")
    parser.add_argument("--skip-texture-optimization", action="store_true")
    parser.add_argument("--qa-root", type=Path, default=DEFAULT_QA_ROOT)
    parser.add_argument("--skip-qa-render", action="store_true")
    parser.add_argument("--scene-data-root", type=Path, default=DEFAULT_SCENE_DATA_ROOT)
    parser.add_argument("--skip-scene-photo-render", action="store_true")
    parser.add_argument("--scene-photo-workers", type=int, default=1, help="Parallel Blender workers for scene photo rendering; use 0 to auto-detect GPUs.")
    parser.add_argument("--scene-photo-engine", choices=["workbench", "eevee"], default="eevee")
    parser.add_argument("--write-cut-smoke", action="store_true")
    parser.add_argument(
        "--lower-osm-roads",
        dest="lower_osm_roads",
        action="store_true",
        default=True,
        help="Lower terrain under OSM road corridors during raw USD generation. Enabled by default.",
    )
    parser.add_argument(
        "--skip-lower-osm-roads",
        dest="lower_osm_roads",
        action="store_false",
        help="Disable OSM road terrain lowering during raw USD generation.",
    )
    parser.add_argument("--road-lower-depth", type=float, default=0.2, help="Meters to lower OSM road terrain when --lower-osm-roads is enabled.")
    parser.add_argument("--road-lower-subdivide-cuts", type=int, default=1, help="Uniform terrain edge subdivision cuts before OSM road lowering.")
    parser.add_argument(
        "--flat-ground",
        action="store_true",
        help="Preprocess each source city into --work-root with compressed ground before raw USD export.",
    )
    parser.add_argument("--flat-ground-height-scale", type=float, default=0.3)
    parser.add_argument("--flat-ground-reference", choices=["auto", "median", "min"], default="min")
    parser.add_argument(
        "--flat-ground-road-boundary-cuts",
        action="store_true",
        help="Enable bmesh road-boundary cuts during flat-ground preprocessing. Enabled by default.",
    )
    parser.add_argument(
        "--skip-flat-ground-road-boundary-cuts",
        dest="flat_ground_road_boundary_cuts",
        action="store_false",
        help="Disable flat-ground road-boundary cuts. Faster, but can create jagged road lowering on coarse terrain.",
    )
    parser.set_defaults(flat_ground_road_boundary_cuts=True)
    parser.add_argument(
        "--flat-ground-copy-height-field-to-source",
        action="store_true",
        help="Also replace the source city's height_field.npz with the processed flat-ground height field, keeping a .bak_flat_ground_* backup.",
    )
    parser.add_argument(
        "--storefront-target-height",
        type=float,
        default=4.0,
        help="Uniform final storefront height in meters; width/depth scale proportionally and opening cuts use the scaled size.",
    )
    parser.add_argument("--innout-radius-m", type=float, default=INNOUT_DEFAULT_RADIUS_M)
    parser.add_argument("--innout-storefronts", type=int, default=INNOUT_DEFAULT_STOREFRONTS)
    parser.add_argument("--innout-props", type=int, default=INNOUT_DEFAULT_PROPS)
    parser.add_argument("--innout-vehicles", type=int, default=INNOUT_DEFAULT_VEHICLES)
    args = parser.parse_args()
    return RunConfig(**vars(args))


def main() -> None:
    config = parse_args()
    pipeline = VcFullPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
