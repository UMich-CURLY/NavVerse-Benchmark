import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


_OSM_SURFACE_MAP_CACHE: Dict[str, Optional["OSMSurfaceMap"]] = {}


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_width(value: Any) -> Optional[float]:
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


def parse_lane_count(value: Any) -> Optional[int]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    for sep in (";", "|", ","):
        if sep in text:
            text = text.split(sep)[0].strip()
            break

    try:
        return max(1, int(float(text)))
    except (TypeError, ValueError):
        return None


def point_to_segment_distance_2d(
    point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray
) -> float:
    segment = end_xy - start_xy
    denom = float(np.dot(segment, segment))
    if denom <= 1e-8:
        return float(np.linalg.norm(point_xy - start_xy))

    t = float(np.dot(point_xy - start_xy, segment) / denom)
    t = min(1.0, max(0.0, t))
    projection = start_xy + t * segment
    return float(np.linalg.norm(point_xy - projection))


class OSMSurfaceMap:
    NAVIGABLE_SURFACE_TYPES = {"sidewalk", "pedestrian"}
    ROAD_HIGHWAY_TYPES = {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "service",
        "busway",
        "living_street",
        "unclassified",
        "construction",
    }
    SURFACE_PRIORITY = [
        "sidewalk",
        "crossing",
        "pedestrian",
        "platform",
        "steps",
        "cycleway",
        "road",
        "parking",
        "water",
        "grass",
    ]
    DRAW_ORDER = [
        "water",
        "grass",
        "parking",
        "road",
        "cycleway",
        "pedestrian",
        "platform",
        "steps",
        "crossing",
        "sidewalk",
    ]
    SURFACE_COLORS = {
        "unknown": (32, 32, 32),
        "sidewalk": (80, 200, 255),
        "crossing": (255, 255, 255),
        "pedestrian": (255, 170, 40),
        "platform": (182, 120, 255),
        "steps": (255, 110, 110),
        "cycleway": (60, 220, 120),
        "road": (100, 100, 100),
        "parking": (155, 155, 60),
        "water": (220, 120, 70),
        "grass": (95, 170, 95),
    }
    ROAD_WIDTH_DEFAULTS = {
        "motorway": 14.0,
        "trunk": 12.0,
        "primary": 12.0,
        "secondary": 10.5,
        "residential": 8.0,
        "tertiary": 10.5,
        "service": 3.0,
        "busway": 7.0,
        "living_street": 5.0,
        "unclassified": 6.0,
        "construction": 6.0,
        "platform": 3.5,
    }
    DEFAULT_WIDTHS = {
        "sidewalk": 1.5,
        "crossing": 3.0,
        "pedestrian": 4.0,
        "platform": 3.5,
        "steps": 2.0,
        "cycleway": 2.0,
        "road": 6.0,
    }
    DEBUG_IMAGE_NAME = "osm_semantic_bev.png"
    DEBUG_METADATA_NAME = "osm_semantic_bev.json"

    @classmethod
    def from_episode(cls, env, episode) -> Optional["OSMSurfaceMap"]:
        args = getattr(env, "args", None)
        scene_folder = getattr(args, "scene_folder", None)
        scene_path = episode.get("path", None)
        if scene_folder is None or scene_path is None:
            return None

        scene_dir = Path(scene_folder) / Path(scene_path).parent
        osm_path = scene_dir / "road_data_with_height.osm"
        cache_key = str(osm_path)
        if cache_key not in _OSM_SURFACE_MAP_CACHE:
            if not osm_path.exists():
                _OSM_SURFACE_MAP_CACHE[cache_key] = None
            else:
                _OSM_SURFACE_MAP_CACHE[cache_key] = cls(osm_path)
        return _OSM_SURFACE_MAP_CACHE[cache_key]

    def __init__(self, osm_path: Union[str, Path], save_debug: bool = True) -> None:
        self.osm_path = Path(osm_path)
        self.scene_dir = self.osm_path.parent
        self.save_debug = save_debug
        self.segment_features: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.polygon_features: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.segment_index: Dict[str, Dict[Tuple[int, int], List[Dict[str, Any]]]] = {}
        self.polygon_index: Dict[str, Dict[Tuple[int, int], List[Dict[str, Any]]]] = {}
        self.spatial_cell_size = 5.0
        self.tag_counters = {
            "highway": Counter(),
            "footway": Counter(),
            "landuse": Counter(),
            "leisure": Counter(),
            "amenity": Counter(),
            "natural": Counter(),
            "surface": Counter(),
            "sidewalk": Counter(),
            "sidewalk:left": Counter(),
            "sidewalk:right": Counter(),
            "sidewalk:both": Counter(),
        }
        self.present_surface_types: set[str] = set()
        self.nodes, self.ways = self._load_osm()
        self.bounds = self._compute_bounds()
        self._build_features()
        self._build_spatial_indexes()
        if self.save_debug:
            self._save_debug_semantic_bev()

    def _load_osm(self) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        tree = ET.parse(self.osm_path)
        root = tree.getroot()

        nodes: Dict[str, Dict[str, float]] = {}
        ways: List[Dict[str, Any]] = []

        for element in root:
            if element.tag != "node":
                continue

            node_data = {
                "id": element.get("id", ""),
                "lat": parse_float(element.get("lat")) or 0.0,
                "lon": parse_float(element.get("lon")) or 0.0,
                "height": 0.0,
                "utm_x": 0.0,
                "utm_y": 0.0,
            }
            for tag in element.findall("tag"):
                key = tag.get("k")
                value = tag.get("v")
                if key == "height":
                    node_data["height"] = parse_float(value) or 0.0
                elif key == "utm_x":
                    node_data["utm_x"] = parse_float(value) or 0.0
                elif key == "utm_y":
                    node_data["utm_y"] = parse_float(value) or 0.0
            nodes[node_data["id"]] = node_data

        for element in root:
            if element.tag != "way":
                continue

            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            for key, counter in self.tag_counters.items():
                value = tags.get(key)
                if value is not None:
                    counter[value] += 1

            ways.append(
                {
                    "id": element.get("id", ""),
                    "nodes": [nd.get("ref") for nd in element.findall("nd")],
                    "tags": tags,
                }
            )

        return nodes, ways

    def _compute_bounds(self) -> Tuple[float, float, float, float]:
        coords = np.array(
            [[node["utm_x"], node["utm_y"]] for node in self.nodes.values()],
            dtype=np.float32,
        )
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        return float(min_x), float(min_y), float(max_x), float(max_y)

    def _is_sidewalk_way(self, tags: Dict[str, str]) -> bool:
        highway = tags.get("highway")
        footway = tags.get("footway")
        if highway == "footway" and footway != "crossing":
            return True
        if footway == "sidewalk":
            return True
        return False

    def _classify_way(self, tags: Dict[str, str]) -> Optional[str]:
        highway = tags.get("highway")
        footway = tags.get("footway")
        amenity = tags.get("amenity")
        landuse = tags.get("landuse")
        natural = tags.get("natural")

        if natural == "water":
            return "water"
        if landuse == "grass":
            return "grass"
        if amenity in {"parking", "parking_space", "motorcycle_parking"}:
            return "parking"
        if highway == "crossing" or footway == "crossing":
            return "crossing"
        if self._is_sidewalk_way(tags):
            return "sidewalk"
        if highway in {"pedestrian", "path"}:
            return "pedestrian"
        if highway == "platform":
            return "platform"
        if highway in {"steps", "elevator"}:
            return "steps"
        if highway == "cycleway":
            return "cycleway"
        if highway in self.ROAD_HIGHWAY_TYPES:
            return "road"
        return None

    def _resolve_width(self, tags: Dict[str, str], surface_type: str) -> float:
        if surface_type == "sidewalk":
            left = parse_width(tags.get("sidewalk:left:width"))
            right = parse_width(tags.get("sidewalk:right:width"))
            if left is not None and right is not None:
                return max(0.8, (left + right) / 2.0)
            if left is not None:
                return max(0.8, left)
            if right is not None:
                return max(0.8, right)

            for key in ("footway:width", "width:sidewalk", "width"):
                width = parse_width(tags.get(key))
                if width is not None:
                    return max(0.8, width)
            return self.DEFAULT_WIDTHS["sidewalk"]

        if surface_type == "road":
            width = parse_width(tags.get("width"))
            if width is not None:
                return max(2.0, width)

            lanes = parse_lane_count(tags.get("lanes"))
            if lanes is not None:
                return max(3.0, 3.5 * lanes)

            highway = tags.get("highway", "")
            return self.ROAD_WIDTH_DEFAULTS.get(highway, self.DEFAULT_WIDTHS["road"])

        width = parse_width(tags.get("width"))
        if width is not None:
            return max(0.5, width)

        return self.DEFAULT_WIDTHS.get(surface_type, 2.0)

    def _sidewalk_width(self, tags: Dict[str, str], side: str) -> float:
        for key in (f"sidewalk:{side}:width", "sidewalk:width", "width:sidewalk"):
            width = parse_width(tags.get(key))
            if width is not None:
                return max(0.8, width)
        return self.DEFAULT_WIDTHS["sidewalk"]

    def _sidewalk_sides(self, tags: Dict[str, str]) -> List[str]:
        sides: set[str] = set()

        sidewalk = str(tags.get("sidewalk", "")).strip().lower()
        if sidewalk in {"both", "yes"}:
            sides.update(("left", "right"))
        elif sidewalk in {"left", "right"}:
            sides.add(sidewalk)

        both = str(tags.get("sidewalk:both", "")).strip().lower()
        if both in {"yes", "both"}:
            sides.update(("left", "right"))

        for side in ("left", "right"):
            value = str(tags.get(f"sidewalk:{side}", "")).strip().lower()
            if value in {"yes", side}:
                sides.add(side)

        return sorted(sides)

    def _add_segment_feature(
        self,
        surface_type: str,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        half_width: float,
    ) -> None:
        self.segment_features[surface_type].append(
            {
                "start": start_xy,
                "end": end_xy,
                "half_width": half_width,
                "bbox": self._feature_bbox(
                    np.stack([start_xy, end_xy], axis=0), padding=half_width
                ),
            }
        )

    def _add_attached_sidewalk_segments(
        self,
        tags: Dict[str, str],
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        road_half_width: float,
    ) -> None:
        sides = self._sidewalk_sides(tags)
        if not sides:
            return

        segment = end_xy - start_xy
        length = float(np.linalg.norm(segment))
        if length <= 1e-8:
            return

        normal = np.array([-segment[1], segment[0]], dtype=np.float32) / length
        for side in sides:
            sidewalk_width = self._sidewalk_width(tags, side)
            offset = normal * (road_half_width + sidewalk_width / 2.0)
            if side == "right":
                offset = -offset
            self.present_surface_types.add("sidewalk")
            self._add_segment_feature(
                "sidewalk",
                start_xy + offset,
                end_xy + offset,
                sidewalk_width / 2.0,
            )

    def _is_polygon_feature(
        self, tags: Dict[str, str], surface_type: str, coords: np.ndarray
    ) -> bool:
        if len(coords) < 4:
            return False

        is_closed = np.allclose(coords[0], coords[-1])
        if not is_closed:
            return False

        if tags.get("area") == "yes":
            return True

        return surface_type in {"water", "grass", "parking"}

    def _feature_bbox(
        self, coords: np.ndarray, padding: float = 0.0
    ) -> Tuple[float, float, float, float]:
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        return (
            float(min_xy[0] - padding),
            float(min_xy[1] - padding),
            float(max_xy[0] + padding),
            float(max_xy[1] + padding),
        )

    def _build_features(self) -> None:
        for way in self.ways:
            tags = way["tags"]
            surface_type = self._classify_way(tags)
            if surface_type is None:
                continue

            coords = []
            for node_id in way["nodes"]:
                node = self.nodes.get(node_id)
                if node is None:
                    continue
                coords.append([node["utm_x"], node["utm_y"]])

            if len(coords) < 2:
                continue

            coords_np = np.array(coords, dtype=np.float32)
            self.present_surface_types.add(surface_type)

            if self._is_polygon_feature(tags, surface_type, coords_np):
                self.polygon_features[surface_type].append(
                    {
                        "contour": coords_np,
                        "bbox": self._feature_bbox(coords_np),
                    }
                )
                continue

            width = self._resolve_width(tags, surface_type)
            half_width = width / 2.0
            for start_xy, end_xy in zip(coords_np[:-1], coords_np[1:]):
                if np.allclose(start_xy, end_xy):
                    continue
                self._add_segment_feature(surface_type, start_xy, end_xy, half_width)
                if surface_type == "road":
                    self._add_attached_sidewalk_segments(
                        tags, start_xy, end_xy, half_width
                    )

    def _point_in_bbox(
        self, point_xy: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> bool:
        x, y = float(point_xy[0]), float(point_xy[1])
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    def _bbox_cells(
        self, bbox: Tuple[float, float, float, float]
    ) -> List[Tuple[int, int]]:
        min_x, min_y, max_x, max_y = bbox
        cell = self.spatial_cell_size
        x0 = math.floor(min_x / cell)
        y0 = math.floor(min_y / cell)
        x1 = math.floor(max_x / cell)
        y1 = math.floor(max_y / cell)
        return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    def _point_cell(self, point_xy: np.ndarray) -> Tuple[int, int]:
        return (
            math.floor(float(point_xy[0]) / self.spatial_cell_size),
            math.floor(float(point_xy[1]) / self.spatial_cell_size),
        )

    def _build_spatial_indexes(self) -> None:
        self.segment_index = {}
        self.polygon_index = {}

        for surface_type, features in self.segment_features.items():
            index: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
            for feature in features:
                for cell in self._bbox_cells(feature["bbox"]):
                    index[cell].append(feature)
            self.segment_index[surface_type] = dict(index)

        for surface_type, features in self.polygon_features.items():
            index = defaultdict(list)
            for feature in features:
                for cell in self._bbox_cells(feature["bbox"]):
                    index[cell].append(feature)
            self.polygon_index[surface_type] = dict(index)

    def _point_in_polygon(self, point_xy: np.ndarray, contour: np.ndarray) -> bool:
        if cv2 is not None:
            inside = cv2.pointPolygonTest(
                contour, (float(point_xy[0]), float(point_xy[1])), False
            )
            return inside >= 0

        x, y = float(point_xy[0]), float(point_xy[1])
        inside = False
        previous = contour[-1]
        for current in contour:
            x1, y1 = float(current[0]), float(current[1])
            x2, y2 = float(previous[0]), float(previous[1])
            if point_to_segment_distance_2d(
                point_xy,
                np.asarray([x1, y1], dtype=np.float32),
                np.asarray([x2, y2], dtype=np.float32),
            ) <= 1e-6:
                return True
            if (y1 > y) != (y2 > y):
                x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < x_intersection:
                    inside = not inside
            previous = current
        return inside

    def get_surface_type(self, point_xyz: Union[List[float], np.ndarray]) -> str:
        point_xy = np.asarray(point_xyz[:2], dtype=np.float32)
        cell = self._point_cell(point_xy)

        for surface_type in self.SURFACE_PRIORITY:
            for polygon in self.polygon_index.get(surface_type, {}).get(cell, []):
                if not self._point_in_bbox(point_xy, polygon["bbox"]):
                    continue
                if self._point_in_polygon(point_xy, polygon["contour"]):
                    return surface_type

            for segment in self.segment_index.get(surface_type, {}).get(cell, []):
                if not self._point_in_bbox(point_xy, segment["bbox"]):
                    continue
                dist = point_to_segment_distance_2d(
                    point_xy, segment["start"], segment["end"]
                )
                if dist <= segment["half_width"]:
                    return surface_type

        return "unknown"

    def is_navigable_surface(
        self, point_xyz: Union[List[float], np.ndarray]
    ) -> Tuple[bool, str]:
        surface_type = self.get_surface_type(point_xyz)
        return surface_type in self.NAVIGABLE_SURFACE_TYPES, surface_type

    def _world_to_pixel(
        self,
        points_xy: np.ndarray,
        resolution_m_per_px: float,
        image_height: int,
        min_x: float,
        min_y: float,
    ) -> np.ndarray:
        px = np.round((points_xy[:, 0] - min_x) / resolution_m_per_px).astype(np.int32)
        py = np.round((points_xy[:, 1] - min_y) / resolution_m_per_px).astype(np.int32)
        py = image_height - 1 - py
        return np.stack([px, py], axis=1)

    def _save_debug_semantic_bev(self) -> None:
        if cv2 is None:
            return

        min_x, min_y, max_x, max_y = self.bounds
        padding = 2.0
        resolution_m_per_px = 0.2
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

        width = max(32, int(math.ceil((max_x - min_x) / resolution_m_per_px)) + 1)
        height = max(32, int(math.ceil((max_y - min_y) / resolution_m_per_px)) + 1)
        canvas = np.full((height, width, 3), 24, dtype=np.uint8)

        for surface_type in self.DRAW_ORDER:
            color = self.SURFACE_COLORS[surface_type]

            for polygon in self.polygon_features.get(surface_type, []):
                contour_px = self._world_to_pixel(
                    polygon["contour"], resolution_m_per_px, height, min_x, min_y
                )
                cv2.fillPoly(canvas, [contour_px], color)

            for segment in self.segment_features.get(surface_type, []):
                line_px = self._world_to_pixel(
                    np.stack([segment["start"], segment["end"]], axis=0),
                    resolution_m_per_px,
                    height,
                    min_x,
                    min_y,
                )
                thickness = max(
                    1, int(round((2.0 * segment["half_width"]) / resolution_m_per_px))
                )
                cv2.line(
                    canvas,
                    tuple(line_px[0]),
                    tuple(line_px[1]),
                    color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

        legend_y = 24
        for surface_type in self.DRAW_ORDER[::-1]:
            if surface_type not in self.present_surface_types:
                continue
            color = self.SURFACE_COLORS[surface_type]
            cv2.rectangle(canvas, (16, legend_y - 10), (34, legend_y + 8), color, -1)
            cv2.putText(
                canvas,
                surface_type,
                (42, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            legend_y += 24

        image_path = self.scene_dir / self.DEBUG_IMAGE_NAME
        metadata_path = self.scene_dir / self.DEBUG_METADATA_NAME
        cv2.imwrite(str(image_path), canvas)

        metadata = {
            "osm_path": str(self.osm_path),
            "resolution_m_per_px": resolution_m_per_px,
            "bounds_xy": {
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
            },
            "surface_types_present": sorted(self.present_surface_types),
            "surface_type_count": len(self.present_surface_types),
            "raw_tag_type_counts": {
                f"{key}_types": len(counter) for key, counter in self.tag_counters.items()
            },
            "raw_tag_histograms": {
                key: counter.most_common() for key, counter in self.tag_counters.items()
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
