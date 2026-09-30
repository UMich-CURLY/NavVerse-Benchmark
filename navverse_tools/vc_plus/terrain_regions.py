#!/usr/bin/env python3
"""Build semantic terrain-region meshes from an existing terrain mesh."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


REGION_ORDER = ("water", "green", "crosswalk", "curb", "sidewalk", "road")
DEFAULT_TERRAIN_REGION = "sidewalk"
DEFAULT_CURB_WIDTH_M = 0.2
MIN_TRIANGLE_AREA_M2 = 1e-10
MIN_SURFACE_XY_AREA_M2 = 1e-8
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


@dataclass
class Segment:
    category: str
    start: tuple[float, float]
    delta: tuple[float, float]
    length_sq: float
    half_width: float
    inner_half_width: float = 0.0
    outer_half_width: float = 0.0


@dataclass
class PolygonRegion:
    category: str
    points: list[tuple[float, float]]
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class ClipPrimitive:
    category: str
    points: tuple[tuple[float, float], ...]
    bounds: tuple[float, float, float, float]


@dataclass
class ClipVertex:
    point: Gf.Vec3f
    uv: Gf.Vec2f | None
    normal: Gf.Vec3f | None


@dataclass
class RegionMeshData:
    points: list[Gf.Vec3f] = field(default_factory=list)
    uvs: list[Gf.Vec2f] = field(default_factory=list)
    normals: list[Gf.Vec3f] = field(default_factory=list)
    material_faces: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    xy_area: float = 0.0
    skipped_degenerate_triangles: int = 0


def parse_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    text = str(value).strip().lower().replace(",", "")
    if not text:
        return default
    feet = text.endswith("'") or text.endswith("ft")
    if text.endswith("'"):
        text = text[:-1]
    elif text.endswith("ft"):
        text = text[:-2]
    elif text.endswith("m"):
        text = text[:-1]
    try:
        number = float(text.strip())
    except ValueError:
        return default
    return number * 0.3048 if feet else number


def road_width(tags: dict[str, str]) -> float:
    width = parse_float(tags.get("width") or tags.get("width:carriageway"), None)
    if width and width > 0:
        return float(width)
    lanes = int(parse_float(tags.get("lanes"), 1.0) or 1)
    highway = tags.get("highway")
    lane_width = 3.75 if highway in {"motorway", "trunk"} else 3.0 if highway in {"residential", "service"} else 3.4
    return max(lanes, 1) * lane_width


def path_width(tags: dict[str, str], default: float) -> float:
    for key in ("sidewalk:left:width", "sidewalk:right:width", "footway:width", "width:sidewalk", "width"):
        width = parse_float(tags.get(key), None)
        if width and width > 0:
            return max(float(width), 0.2)
    return default


def is_drivable_road(tags: dict[str, str]) -> bool:
    return tags.get("highway") in ROAD_HIGHWAYS and tags.get("area") != "yes"


def sidewalk_sides(tags: dict[str, str]) -> list[float]:
    sidewalk = (tags.get("sidewalk") or "").lower()
    if sidewalk in {"both", "yes"}:
        return [1.0, -1.0]
    if sidewalk == "left":
        return [1.0]
    if sidewalk == "right":
        return [-1.0]
    return []


def classify_polygon(tags: dict[str, str]) -> str | None:
    if tags.get("natural") in {"water", "bay"} or tags.get("waterway") or tags.get("landuse") in {"reservoir", "basin"}:
        return "water"
    if (
        tags.get("leisure") in {"park", "garden"}
        or tags.get("landuse") in {"grass", "forest", "meadow", "recreation_ground", "village_green"}
        or tags.get("natural") in {"wood", "grassland", "scrub"}
    ):
        return "green"
    return None


def road_lower_transition_width(half_width: float, curb_width: float = DEFAULT_CURB_WIDTH_M) -> float:
    return min(max(float(curb_width), 0.0), max(0.01, float(half_width) * 0.5))


def road_inner_half_width(half_width: float, curb_width: float = DEFAULT_CURB_WIDTH_M) -> float:
    safe_half = max(float(half_width), 0.05)
    return max(safe_half - road_lower_transition_width(safe_half, curb_width), safe_half * 0.5)


def parse_osm_regions(osm_path: Path, curb_width: float = DEFAULT_CURB_WIDTH_M) -> tuple[list[Segment], list[PolygonRegion]]:
    root = ET.parse(osm_path).getroot()
    nodes: dict[str, tuple[float, float]] = {}
    ways: list[tuple[list[str], dict[str, str]]] = []
    for element in root:
        if element.tag == "node":
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            x = parse_float(tags.get("utm_x"), None)
            y = parse_float(tags.get("utm_y"), None)
            if x is not None and y is not None:
                nodes[element.get("id")] = (float(x), float(y))
        elif element.tag == "way":
            refs = [nd.get("ref") for nd in element.findall("nd") if nd.get("ref")]
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            ways.append((refs, tags))

    segments: list[Segment] = []
    polygons: list[PolygonRegion] = []

    def add_segment(category: str, start: tuple[float, float], end: tuple[float, float], width: float, *, inner: float = 0.0, outer: float = 0.0) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-8:
            return
        segments.append(
            Segment(
                category=category,
                start=start,
                delta=(dx, dy),
                length_sq=length_sq,
                half_width=max(width * 0.5, 0.05),
                inner_half_width=max(inner, 0.0),
                outer_half_width=max(outer, 0.0),
            )
        )

    for refs, tags in ways:
        points = [nodes[ref] for ref in refs if ref in nodes]
        if len(points) < 2:
            continue
        closed = len(points) >= 4 and refs[0] == refs[-1]
        poly_category = classify_polygon(tags)
        if closed and poly_category:
            ring = points[:-1] if points[0] == points[-1] else points
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            polygons.append(PolygonRegion(poly_category, ring, (min(xs), min(ys), max(xs), max(ys))))
            continue

        highway = tags.get("highway")
        if tags.get("footway") == "crossing" or highway == "crossing" or tags.get("crossing"):
            category = "crosswalk"
            width = path_width(tags, 4.0)
        elif highway in {"footway", "pedestrian", "path", "steps"} or tags.get("footway") == "sidewalk":
            category = "sidewalk"
            width = path_width(tags, 1.8)
        elif is_drivable_road(tags):
            category = "road"
            width = road_width(tags)
        else:
            continue

        for start, end in zip(points, points[1:]):
            if category == "road":
                half = width * 0.5
                inner_half = road_inner_half_width(half, curb_width)
                add_segment("road", start, end, inner_half * 2.0)
                add_segment("curb", start, end, width, inner=inner_half, outer=half)
                sidewalk_width = path_width(tags, 1.8)
                for side in sidewalk_sides(tags):
                    length = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    if length <= 1e-6:
                        continue
                    tx = (end[0] - start[0]) / length
                    ty = (end[1] - start[1]) / length
                    nx, ny = -ty, tx
                    sidewalk_center = side * (half + sidewalk_width * 0.5)
                    sidewalk_start = (start[0] + nx * sidewalk_center, start[1] + ny * sidewalk_center)
                    sidewalk_end = (end[0] + nx * sidewalk_center, end[1] + ny * sidewalk_center)
                    add_segment("sidewalk", sidewalk_start, sidewalk_end, sidewalk_width)
            else:
                add_segment(category, start, end, width)

    return segments, polygons


def point_hits_segment(x: float, y: float, segment: Segment) -> bool:
    sx, sy = segment.start
    dx, dy = segment.delta
    t = ((x - sx) * dx + (y - sy) * dy) / segment.length_sq
    t = max(0.0, min(1.0, t))
    px = sx + t * dx
    py = sy + t * dy
    dist = math.hypot(x - px, y - py)
    if segment.inner_half_width or segment.outer_half_width:
        return segment.inner_half_width <= dist <= segment.outer_half_width
    return dist <= segment.half_width


def point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    inside = False
    count = len(points)
    j = count - 1
    for i in range(count):
        xi, yi = points[i]
        xj, yj = points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def build_segment_grid(segments: list[Segment], cell_size: float) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    safe_cell = max(float(cell_size or 25.0), 1.0)
    for index, segment in enumerate(segments):
        sx, sy = segment.start
        ex = sx + segment.delta[0]
        ey = sy + segment.delta[1]
        pad = max(segment.half_width, segment.outer_half_width, 0.1)
        min_x, max_x = min(sx, ex) - pad, max(sx, ex) + pad
        min_y, max_y = min(sy, ey) - pad, max(sy, ey) + pad
        for ix in range(math.floor(min_x / safe_cell), math.floor(max_x / safe_cell) + 1):
            for iy in range(math.floor(min_y / safe_cell), math.floor(max_y / safe_cell) + 1):
                grid.setdefault((ix, iy), []).append(index)
    return grid


def classify_point(
    x: float,
    y: float,
    segments: list[Segment],
    segment_grid: dict[tuple[int, int], list[int]],
    polygons: list[PolygonRegion],
    cell_size: float,
) -> str:
    for category in ("water", "green"):
        for polygon in polygons:
            if polygon.category != category:
                continue
            min_x, min_y, max_x, max_y = polygon.bounds
            if min_x <= x <= max_x and min_y <= y <= max_y and point_in_polygon(x, y, polygon.points):
                return category

    safe_cell = max(float(cell_size or 25.0), 1.0)
    candidates = segment_grid.get((math.floor(x / safe_cell), math.floor(y / safe_cell)), [])
    # Road core must win before curb so overlapping road corridors do not create
    # false curb strips through intersections or polyline joins.
    for category in ("crosswalk", "road", "curb", "sidewalk"):
        for index in candidates:
            segment = segments[index]
            if segment.category == category and point_hits_segment(x, y, segment):
                return category
    return DEFAULT_TERRAIN_REGION


def bound_material_for_subset(subset_prim: Usd.Prim) -> Sdf.Path | None:
    material = UsdShade.MaterialBindingAPI(subset_prim).ComputeBoundMaterial()[0]
    if material:
        return material.GetPath()
    targets = subset_prim.GetRelationship("material:binding").GetTargets()
    return targets[0] if targets else None


def source_face_materials(mesh_prim: Usd.Prim, face_count: int) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for child in mesh_prim.GetChildren():
        if child.GetTypeName() != "GeomSubset":
            continue
        material_path = bound_material_for_subset(child)
        indices_attr = child.GetAttribute("indices")
        indices = indices_attr.Get() if indices_attr else None
        if material_path is None or indices is None:
            continue
        key = material_path.pathString
        result.setdefault(key, [])
        result[key].extend(int(index) for index in indices if 0 <= int(index) < face_count)
    return result


def signed_xy_area(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )


def ccw_points(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    cleaned = []
    for point in points:
        xy = (float(point[0]), float(point[1]))
        if not cleaned or math.dist(xy, cleaned[-1]) > 1e-8:
            cleaned.append(xy)
    if len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) <= 1e-8:
        cleaned.pop()
    if signed_xy_area(cleaned) < 0.0:
        cleaned.reverse()
    return tuple(cleaned)


def primitive(category: str, points: list[tuple[float, float]]) -> ClipPrimitive | None:
    ring = ccw_points(points)
    if len(ring) < 3 or abs(signed_xy_area(ring)) <= 1e-8:
        return None
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return ClipPrimitive(category, ring, (min(xs), min(ys), max(xs), max(ys)))


def point_in_triangle(point: tuple[float, float], triangle: tuple[tuple[float, float], ...]) -> bool:
    signs = []
    for index in range(3):
        start = triangle[index]
        end = triangle[(index + 1) % 3]
        signs.append((end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]))
    return all(value >= -1e-9 for value in signs) or all(value <= 1e-9 for value in signs)


def triangulate_polygon(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], ...]]:
    ring = list(ccw_points(points))
    triangles = []
    while len(ring) > 3:
        ear_found = False
        for index, current in enumerate(ring):
            previous = ring[index - 1]
            following = ring[(index + 1) % len(ring)]
            cross = (current[0] - previous[0]) * (following[1] - current[1]) - (
                current[1] - previous[1]
            ) * (following[0] - current[0])
            if cross <= 1e-9:
                continue
            candidate = (previous, current, following)
            adjacent = {(index - 1) % len(ring), index, (index + 1) % len(ring)}
            if any(
                point_in_triangle(point, candidate)
                for check_index, point in enumerate(ring)
                if check_index not in adjacent
            ):
                continue
            triangles.append(candidate)
            del ring[index]
            ear_found = True
            break
        if not ear_found:
            return []
    if len(ring) == 3 and abs(signed_xy_area(ring)) > 1e-8:
        triangles.append(tuple(ring))
    return triangles


def offset_polyline(
    points: list[tuple[float, float]],
    offset: float,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return []
    segment_normals = []
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        segment_normals.append((0.0, 0.0) if length <= 1e-8 else (-dy / length, dx / length))
    result = []
    for index, point in enumerate(points):
        if index == 0:
            nx, ny = segment_normals[0]
            scale = offset
        elif index == len(points) - 1:
            nx, ny = segment_normals[-1]
            scale = offset
        else:
            previous = segment_normals[index - 1]
            following = segment_normals[index]
            mx = previous[0] + following[0]
            my = previous[1] + following[1]
            miter_length = math.hypot(mx, my)
            if miter_length <= 1e-8:
                nx, ny = following
                scale = offset
            else:
                nx, ny = mx / miter_length, my / miter_length
                denominator = nx * following[0] + ny * following[1]
                scale = offset if abs(denominator) <= 1e-4 else offset / denominator
                limit = max(abs(offset) * 4.0, abs(offset))
                scale = max(-limit, min(limit, scale))
        result.append((point[0] + nx * scale, point[1] + ny * scale))
    return result


def ribbon_primitives(
    category: str,
    points: list[tuple[float, float]],
    first_offset: float,
    second_offset: float,
) -> list[ClipPrimitive]:
    first = offset_polyline(points, first_offset)
    second = offset_polyline(points, second_offset)
    result = []
    for index in range(min(len(first), len(second)) - 1):
        shape = primitive(
            category,
            [first[index], second[index], second[index + 1], first[index + 1]],
        )
        if shape is not None:
            result.append(shape)
    return result


def parse_osm_clip_primitives(
    osm_path: Path,
    curb_width: float = DEFAULT_CURB_WIDTH_M,
) -> dict[str, list[ClipPrimitive]]:
    root = ET.parse(osm_path).getroot()
    nodes: dict[str, tuple[float, float]] = {}
    ways: list[tuple[list[str], dict[str, str]]] = []
    for element in root:
        if element.tag == "node":
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            x = parse_float(tags.get("utm_x"), None)
            y = parse_float(tags.get("utm_y"), None)
            if x is not None and y is not None:
                nodes[element.get("id")] = (float(x), float(y))
        elif element.tag == "way":
            refs = [nd.get("ref") for nd in element.findall("nd") if nd.get("ref")]
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            ways.append((refs, tags))

    result: dict[str, list[ClipPrimitive]] = {category: [] for category in REGION_ORDER}
    for refs, tags in ways:
        points = [nodes[ref] for ref in refs if ref in nodes]
        if len(points) < 2:
            continue
        closed = len(points) >= 4 and refs[0] == refs[-1]
        polygon_category = classify_polygon(tags)
        if closed and polygon_category:
            ring = points[:-1] if points[0] == points[-1] else points
            for triangle in triangulate_polygon(ring):
                shape = primitive(polygon_category, list(triangle))
                if shape is not None:
                    result[polygon_category].append(shape)
            continue

        highway = tags.get("highway")
        if tags.get("footway") == "crossing" or highway == "crossing" or tags.get("crossing"):
            width = path_width(tags, 4.0)
            result["crosswalk"].extend(ribbon_primitives("crosswalk", points, width * 0.5, -width * 0.5))
        elif highway in {"footway", "pedestrian", "path", "steps"} or tags.get("footway") == "sidewalk":
            width = path_width(tags, 1.8)
            result["sidewalk"].extend(ribbon_primitives("sidewalk", points, width * 0.5, -width * 0.5))
        elif is_drivable_road(tags):
            half_width = road_width(tags) * 0.5
            inner = road_inner_half_width(half_width, curb_width)
            result["road"].extend(ribbon_primitives("road", points, inner, -inner))
            result["curb"].extend(ribbon_primitives("curb", points, half_width, inner))
            result["curb"].extend(ribbon_primitives("curb", points, -inner, -half_width))
            sidewalk_width = path_width(tags, 1.8)
            for side in sidewalk_sides(tags):
                center = side * (half_width + sidewalk_width * 0.5)
                result["sidewalk"].extend(
                    ribbon_primitives(
                        "sidewalk",
                        points,
                        center + sidewalk_width * 0.5,
                        center - sidewalk_width * 0.5,
                    )
                )
    return result


def build_primitive_grid(
    primitives: list[ClipPrimitive],
    cell_size: float,
) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    safe_cell = max(float(cell_size or 25.0), 1.0)
    for index, shape in enumerate(primitives):
        min_x, min_y, max_x, max_y = shape.bounds
        for ix in range(math.floor(min_x / safe_cell), math.floor(max_x / safe_cell) + 1):
            for iy in range(math.floor(min_y / safe_cell), math.floor(max_y / safe_cell) + 1):
                grid.setdefault((ix, iy), []).append(index)
    return grid


def clip_candidates(
    polygon: list[ClipVertex],
    primitives: list[ClipPrimitive],
    grid: dict[tuple[int, int], list[int]],
    cell_size: float,
) -> list[ClipPrimitive]:
    safe_cell = max(float(cell_size or 25.0), 1.0)
    xs = [float(vertex.point[0]) for vertex in polygon]
    ys = [float(vertex.point[1]) for vertex in polygon]
    polygon_bounds = (min(xs), min(ys), max(xs), max(ys))
    indices = set()
    for ix in range(math.floor(min(xs) / safe_cell), math.floor(max(xs) / safe_cell) + 1):
        for iy in range(math.floor(min(ys) / safe_cell), math.floor(max(ys) / safe_cell) + 1):
            indices.update(grid.get((ix, iy), []))
    return [
        primitives[index]
        for index in sorted(indices)
        if not (
            primitives[index].bounds[2] < polygon_bounds[0]
            or primitives[index].bounds[0] > polygon_bounds[2]
            or primitives[index].bounds[3] < polygon_bounds[1]
            or primitives[index].bounds[1] > polygon_bounds[3]
        )
    ]


def interpolate_vertex(start: ClipVertex, end: ClipVertex, amount: float) -> ClipVertex:
    point = Gf.Vec3f(start.point * (1.0 - amount) + end.point * amount)
    uv = None if start.uv is None or end.uv is None else Gf.Vec2f(start.uv * (1.0 - amount) + end.uv * amount)
    normal = None
    if start.normal is not None and end.normal is not None:
        normal = Gf.Vec3f(start.normal * (1.0 - amount) + end.normal * amount)
        if normal.GetLength() > 1e-9:
            normal.Normalize()
    return ClipVertex(point, uv, normal)


def clean_clip_polygon(polygon: list[ClipVertex]) -> list[ClipVertex]:
    result = []
    for vertex in polygon:
        if not result or math.hypot(
            float(vertex.point[0] - result[-1].point[0]),
            float(vertex.point[1] - result[-1].point[1]),
        ) > 1e-7:
            result.append(vertex)
    if len(result) > 1 and math.hypot(
        float(result[0].point[0] - result[-1].point[0]),
        float(result[0].point[1] - result[-1].point[1]),
    ) <= 1e-7:
        result.pop()
    return result


def split_polygon_by_line(
    polygon: list[ClipVertex],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[list[ClipVertex], list[ClipVertex]]:
    inside: list[ClipVertex] = []
    outside: list[ClipVertex] = []
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    def signed_distance(vertex: ClipVertex) -> float:
        return dx * (float(vertex.point[1]) - start[1]) - dy * (float(vertex.point[0]) - start[0])

    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        current_distance = signed_distance(current)
        following_distance = signed_distance(following)
        if current_distance >= -1e-8:
            inside.append(current)
        if current_distance <= 1e-8:
            outside.append(current)
        if (current_distance > 1e-8 and following_distance < -1e-8) or (
            current_distance < -1e-8 and following_distance > 1e-8
        ):
            intersection = interpolate_vertex(
                current,
                following,
                current_distance / (current_distance - following_distance),
            )
            inside.append(intersection)
            outside.append(intersection)
    return clean_clip_polygon(inside), clean_clip_polygon(outside)


def polygon_xy_area(polygon: list[ClipVertex]) -> float:
    return abs(
        signed_xy_area([(float(vertex.point[0]), float(vertex.point[1])) for vertex in polygon])
    )


def carve_primitive(
    polygon: list[ClipVertex],
    shape: ClipPrimitive,
) -> tuple[list[list[ClipVertex]], list[list[ClipVertex]]]:
    inside_parts = [polygon]
    outside_parts: list[list[ClipVertex]] = []
    for index, start in enumerate(shape.points):
        end = shape.points[(index + 1) % len(shape.points)]
        next_inside = []
        for part in inside_parts:
            inside, outside = split_polygon_by_line(part, start, end)
            if len(inside) >= 3 and polygon_xy_area(inside) > 1e-9:
                next_inside.append(inside)
            if len(outside) >= 3 and polygon_xy_area(outside) > 1e-9:
                outside_parts.append(outside)
        inside_parts = next_inside
        if not inside_parts:
            break
    return inside_parts, outside_parts


def append_polygon(
    data: RegionMeshData,
    polygon: list[ClipVertex],
    material_path: str,
) -> None:
    for index in range(1, len(polygon) - 1):
        triangle = [polygon[0], polygon[index], polygon[index + 1]]
        area = polygon_xy_area(triangle)
        edge_a = Gf.Vec3d(triangle[1].point) - Gf.Vec3d(triangle[0].point)
        edge_b = Gf.Vec3d(triangle[2].point) - Gf.Vec3d(triangle[0].point)
        if Gf.Cross(edge_a, edge_b).GetLength() * 0.5 <= MIN_TRIANGLE_AREA_M2:
            data.skipped_degenerate_triangles += 1
            continue
        face_index = len(data.points) // 3
        for vertex in triangle:
            data.points.append(vertex.point)
            if vertex.uv is not None:
                data.uvs.append(vertex.uv)
            if vertex.normal is not None:
                data.normals.append(vertex.normal)
        data.material_faces[material_path].append(face_index)
        data.xy_area += area


def split_triangle_into_regions(
    triangle: list[ClipVertex],
    material_path: str,
    primitives_by_category: dict[str, list[ClipPrimitive]],
    grids_by_category: dict[str, dict[tuple[int, int], list[int]]],
    cell_size: float,
    region_data: dict[str, RegionMeshData],
) -> None:
    if polygon_xy_area(triangle) <= 1e-9:
        append_polygon(region_data[DEFAULT_TERRAIN_REGION], triangle, material_path)
        return
    remaining = [triangle]
    for category in ("crosswalk", "road", "curb", "sidewalk", "water", "green"):
        candidates = clip_candidates(
            triangle,
            primitives_by_category[category],
            grids_by_category[category],
            cell_size,
        )
        for shape in candidates:
            next_remaining = []
            for part in remaining:
                inside, outside = carve_primitive(part, shape)
                for clipped in inside:
                    append_polygon(region_data[category], clipped, material_path)
                next_remaining.extend(outside)
            remaining = next_remaining
            if not remaining:
                return
    for part in remaining:
        append_polygon(region_data[DEFAULT_TERRAIN_REGION], part, material_path)


def collect_polygon_inside_primitives(
    polygon: list[ClipVertex],
    primitives: list[ClipPrimitive],
    grid: dict[tuple[int, int], list[int]],
    cell_size: float,
) -> list[list[ClipVertex]]:
    remaining = [polygon]
    collected = []
    for shape in clip_candidates(polygon, primitives, grid, cell_size):
        next_remaining = []
        for part in remaining:
            inside, outside = carve_primitive(part, shape)
            collected.extend(inside)
            next_remaining.extend(outside)
        remaining = next_remaining
        if not remaining:
            break
    return collected


def road_restore_offset(
    x: float,
    y: float,
    road_segments: list[Segment],
    segment_grid: dict[tuple[int, int], list[int]],
    cell_size: float,
    road_depth: float,
) -> float:
    if road_depth <= 0.0:
        return 0.0
    safe_cell = max(float(cell_size or 25.0), 1.0)
    candidates = segment_grid.get((math.floor(x / safe_cell), math.floor(y / safe_cell)), [])
    result = 0.0
    for index in candidates:
        segment = road_segments[index]
        sx, sy = segment.start
        dx, dy = segment.delta
        t = max(0.0, min(1.0, ((x - sx) * dx + (y - sy) * dy) / segment.length_sq))
        distance = math.hypot(x - (sx + t * dx), y - (sy + t * dy))
        inner = float(segment.half_width)
        outer = inner + DEFAULT_CURB_WIDTH_M
        if distance <= inner:
            result = max(result, road_depth)
        elif distance < outer:
            result = max(result, road_depth * (outer - distance) / DEFAULT_CURB_WIDTH_M)
    return result


def lift_curb_surface(
    data: RegionMeshData,
    road_segments: list[Segment],
    segment_grid: dict[tuple[int, int], list[int]],
    cell_size: float,
    road_depth: float,
) -> int:
    lifted = 0
    for index, point in enumerate(data.points):
        offset = road_restore_offset(
            float(point[0]),
            float(point[1]),
            road_segments,
            segment_grid,
            cell_size,
            road_depth,
        )
        if offset <= 1e-6:
            continue
        data.points[index] = Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]) + offset)
        lifted += 1
    return lifted


def material_for_faces(data: RegionMeshData) -> dict[int, str]:
    return {
        int(face_index): material_path
        for material_path, face_indices in data.material_faces.items()
        for face_index in face_indices
    }


def point_in_clip_primitive(x: float, y: float, shape: ClipPrimitive, tolerance: float = 1e-8) -> bool:
    if x < shape.bounds[0] - tolerance or x > shape.bounds[2] + tolerance:
        return False
    if y < shape.bounds[1] - tolerance or y > shape.bounds[3] + tolerance:
        return False
    for index, start in enumerate(shape.points):
        end = shape.points[(index + 1) % len(shape.points)]
        if (end[0] - start[0]) * (y - start[1]) - (end[1] - start[1]) * (x - start[0]) < -tolerance:
            return False
    return True


def point_in_primitive_union(
    x: float,
    y: float,
    primitives: list[ClipPrimitive],
    grid: dict[tuple[int, int], list[int]],
    cell_size: float,
) -> bool:
    safe_cell = max(float(cell_size or 25.0), 1.0)
    candidates = grid.get((math.floor(x / safe_cell), math.floor(y / safe_cell)), [])
    return any(point_in_clip_primitive(x, y, primitives[index]) for index in candidates)


def edge_crosses_primitive_union_boundary(
    start: Gf.Vec3f,
    end: Gf.Vec3f,
    primitives: list[ClipPrimitive],
    grid: dict[tuple[int, int], list[int]],
    cell_size: float,
) -> bool:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-7:
        return False
    midpoint_x = float(start[0] + end[0]) * 0.5
    midpoint_y = float(start[1] + end[1]) * 0.5
    sample_distance = min(0.01, max(0.001, length * 0.1))
    offset_x = -dy / length * sample_distance
    offset_y = dx / length * sample_distance
    first_inside = point_in_primitive_union(
        midpoint_x + offset_x,
        midpoint_y + offset_y,
        primitives,
        grid,
        cell_size,
    )
    second_inside = point_in_primitive_union(
        midpoint_x - offset_x,
        midpoint_y - offset_y,
        primitives,
        grid,
        cell_size,
    )
    return first_inside != second_inside


def add_road_boundary_walls(
    road_surface: RegionMeshData,
    curb_data: RegionMeshData,
    road_primitives: list[ClipPrimitive],
    road_primitive_grid: dict[tuple[int, int], list[int]],
    road_segments: list[Segment],
    segment_grid: dict[tuple[int, int], list[int]],
    cell_size: float,
    road_depth: float,
) -> int:
    if road_depth <= 0.0:
        return 0
    face_materials = material_for_faces(road_surface)
    edges: dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[tuple[int, int, int]]] = defaultdict(list)

    def key(point: Gf.Vec3f) -> tuple[int, int, int]:
        return tuple(int(round(float(value) * 100000.0)) for value in point)

    for face_index in range(len(road_surface.points) // 3):
        start = face_index * 3
        for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
            index_a = start + local_a
            index_b = start + local_b
            edge_key = tuple(sorted((key(road_surface.points[index_a]), key(road_surface.points[index_b]))))
            edges[edge_key].append((index_a, index_b, face_index))

    wall_count = 0
    for records in edges.values():
        if len(records) != 1:
            continue
        index_a, index_b, face_index = records[0]
        bottom_a = road_surface.points[index_a]
        bottom_b = road_surface.points[index_b]
        if not edge_crosses_primitive_union_boundary(
            bottom_a,
            bottom_b,
            road_primitives,
            road_primitive_grid,
            cell_size,
        ):
            continue
        offset_a = road_restore_offset(
            float(bottom_a[0]), float(bottom_a[1]), road_segments, segment_grid, cell_size, road_depth
        )
        offset_b = road_restore_offset(
            float(bottom_b[0]), float(bottom_b[1]), road_segments, segment_grid, cell_size, road_depth
        )
        if min(offset_a, offset_b) < road_depth * 0.75:
            continue
        top_a = Gf.Vec3f(float(bottom_a[0]), float(bottom_a[1]), float(bottom_a[2]) + offset_a)
        top_b = Gf.Vec3f(float(bottom_b[0]), float(bottom_b[1]), float(bottom_b[2]) + offset_b)
        uv_a = road_surface.uvs[index_a] if road_surface.uvs else None
        uv_b = road_surface.uvs[index_b] if road_surface.uvs else None
        edge_a = Gf.Vec3d(top_b) - Gf.Vec3d(top_a)
        edge_b = Gf.Vec3d(bottom_a) - Gf.Vec3d(top_a)
        normal = Gf.Cross(edge_a, edge_b)
        if normal.GetLength() <= 1e-9:
            continue
        normal.Normalize()
        normal_f = Gf.Vec3f(float(normal[0]), float(normal[1]), float(normal[2]))
        polygon = [
            ClipVertex(top_a, uv_a, normal_f),
            ClipVertex(top_b, uv_b, normal_f),
            ClipVertex(bottom_b, uv_b, normal_f),
            ClipVertex(bottom_a, uv_a, normal_f),
        ]
        append_polygon(curb_data, polygon, face_materials.get(face_index, ""))
        wall_count += 1
    return wall_count


def combine_region_data(region_data: dict[str, RegionMeshData]) -> RegionMeshData:
    combined = RegionMeshData()
    for category in REGION_ORDER:
        source = region_data[category]
        face_offset = len(combined.points) // 3
        combined.points.extend(source.points)
        combined.uvs.extend(source.uvs)
        combined.normals.extend(source.normals)
        for material_path, faces in source.material_faces.items():
            combined.material_faces[material_path].extend(face_offset + int(face) for face in faces)
        combined.xy_area += source.xy_area
        combined.skipped_degenerate_triangles += source.skipped_degenerate_triangles
    return combined


def create_region_mesh(
    stage: Usd.Stage,
    path: str,
    source_mesh: UsdGeom.Mesh,
    data: RegionMeshData,
) -> int:
    mesh = UsdGeom.Mesh.Define(stage, path)
    face_count = len(data.points) // 3
    mesh.CreatePointsAttr(data.points)
    mesh.CreateFaceVertexCountsAttr([3] * face_count)
    mesh.CreateFaceVertexIndicesAttr(list(range(len(data.points))))
    mesh.CreateSubdivisionSchemeAttr(source_mesh.GetSubdivisionSchemeAttr().Get() or UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(source_mesh.GetOrientationAttr().Get() or UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(bool(source_mesh.GetDoubleSidedAttr().Get()))
    source_binding = UsdShade.MaterialBindingAPI(source_mesh.GetPrim()).GetDirectBinding()
    region_binding_api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    source_material_path = source_binding.GetMaterialPath()
    if source_material_path:
        source_material = UsdShade.Material(stage.GetPrimAtPath(source_material_path))
        if source_material:
            source_relationship = source_mesh.GetPrim().GetRelationship("material:binding")
            strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(source_relationship)
            region_binding_api.Bind(
                source_material,
                bindingStrength=strength,
            )
    if data.uvs:
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "UVMap",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        primvar.Set(data.uvs)
    if data.normals:
        mesh.CreateNormalsAttr(data.normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    if data.points:
        xs = [float(point[0]) for point in data.points]
        ys = [float(point[1]) for point in data.points]
        zs = [float(point[2]) for point in data.points]
        mesh.CreateExtentAttr([Gf.Vec3f(min(xs), min(ys), min(zs)), Gf.Vec3f(max(xs), max(ys), max(zs))])

    family_type = UsdGeom.Subset.GetFamilyType(source_mesh, "materialBind")
    if family_type:
        region_binding_api.SetMaterialBindSubsetsFamilyType(family_type)
    for material_path, faces in data.material_faces.items():
        if not faces:
            continue
        name = Sdf.Path(material_path).name
        subset = region_binding_api.CreateMaterialBindSubset(name, faces, UsdGeom.Tokens.face)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if material:
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    return face_count


def subtract_convex_polygon_from_mesh(
    stage: Usd.Stage,
    mesh_path: str,
    world_polygon: list[list[float]] | list[tuple[float, float]],
) -> dict[str, Any] | None:
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim or prim.GetTypeName() != "Mesh":
        return None
    mesh = UsdGeom.Mesh(prim)
    points = list(mesh.GetPointsAttr().Get() or [])
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if not points or not counts or not indices:
        return None

    world_to_local = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim).GetInverse()
    local_polygon = []
    for point in world_polygon:
        local = world_to_local.Transform(Gf.Vec3d(float(point[0]), float(point[1]), 0.0))
        local_polygon.append((float(local[0]), float(local[1])))
    hole = primitive("terrain_hole", local_polygon)
    if hole is None:
        raise RuntimeError(f"invalid terrain-hole polygon for {mesh_path}")
    hole_min_x, hole_min_y, hole_max_x, hole_max_y = hole.bounds

    face_count = len(counts)
    uv_attr = prim.GetAttribute("primvars:UVMap")
    uv_values = uv_attr.Get() if uv_attr else None
    normal_attr = prim.GetAttribute("normals")
    normal_values = normal_attr.Get() if normal_attr else None
    normal_interpolation = normal_attr.GetMetadata("interpolation") if normal_attr else None
    if normal_values is not None and normal_interpolation != UsdGeom.Tokens.faceVarying:
        raise RuntimeError(f"unsupported normal interpolation on {mesh_path}: {normal_interpolation!r}")
    material_faces = source_face_materials(prim, face_count)
    direct_material_path = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath().pathString
    face_materials = {
        int(face_index): material_path
        for material_path, face_indices in material_faces.items()
        for face_index in face_indices
    }

    data = RegionMeshData()
    source_triangle_count = 0
    source_xy_area = 0.0
    offset = 0
    for face_index, count in enumerate(counts):
        corners = []
        for local_index in range(count):
            corner_offset = offset + local_index
            corners.append(
                ClipVertex(
                    Gf.Vec3f(points[indices[corner_offset]]),
                    Gf.Vec2f(uv_values[corner_offset]) if uv_values is not None else None,
                    Gf.Vec3f(normal_values[corner_offset]) if normal_values is not None else None,
                )
            )
        material_path = face_materials.get(face_index, direct_material_path)
        for triangle_index in range(1, len(corners) - 1):
            triangle = [corners[0], corners[triangle_index], corners[triangle_index + 1]]
            source_triangle_count += 1
            source_xy_area += polygon_xy_area(triangle)
            xs = [float(vertex.point[0]) for vertex in triangle]
            ys = [float(vertex.point[1]) for vertex in triangle]
            if (
                max(xs) < hole_min_x
                or min(xs) > hole_max_x
                or max(ys) < hole_min_y
                or min(ys) > hole_max_y
            ):
                append_polygon(data, triangle, material_path)
                continue
            inside, outside = carve_primitive(triangle, hole)
            if not inside:
                append_polygon(data, triangle, material_path)
                continue
            for part in outside:
                append_polygon(data, part, material_path)
        offset += count

    for child in list(prim.GetChildren()):
        if child.GetTypeName() == "GeomSubset":
            stage.RemovePrim(child.GetPath())
    output_face_count = len(data.points) // 3
    mesh.GetPointsAttr().Set(data.points)
    mesh.GetFaceVertexCountsAttr().Set([3] * output_face_count)
    mesh.GetFaceVertexIndicesAttr().Set(list(range(len(data.points))))
    if data.uvs:
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "UVMap",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        primvar.Set(data.uvs)
        primvar.GetIndicesAttr().Clear()
    if data.normals:
        mesh.GetNormalsAttr().Set(data.normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    if data.points:
        xs = [float(point[0]) for point in data.points]
        ys = [float(point[1]) for point in data.points]
        zs = [float(point[2]) for point in data.points]
        mesh.GetExtentAttr().Set(
            [Gf.Vec3f(min(xs), min(ys), min(zs)), Gf.Vec3f(max(xs), max(ys), max(zs))]
        )
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    family_type = UsdGeom.Subset.GetFamilyType(mesh, "materialBind")
    if family_type:
        binding_api.SetMaterialBindSubsetsFamilyType(family_type)
    for material_path, faces in data.material_faces.items():
        if not faces:
            continue
        subset = binding_api.CreateMaterialBindSubset(Sdf.Path(material_path).name, faces, UsdGeom.Tokens.face)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if material:
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)

    kept_xy_area = data.xy_area
    return {
        "mesh_path": mesh_path,
        "source_triangle_count": source_triangle_count,
        "output_triangle_count": output_face_count,
        "source_xy_area": source_xy_area,
        "kept_xy_area": kept_xy_area,
        "removed_xy_area": max(0.0, source_xy_area - kept_xy_area),
        "skipped_degenerate_triangles": data.skipped_degenerate_triangles,
    }


def move_prim(stage: Usd.Stage, source_path: str, destination_path: str) -> None:
    editor = Usd.NamespaceEditor(stage)
    editor.MovePrimAtPath(Sdf.Path(source_path), Sdf.Path(destination_path))
    result = editor.CanApplyEdits()
    if not result:
        raise RuntimeError(f"cannot move {source_path} to {destination_path}: {result}")
    if not editor.ApplyEdits():
        raise RuntimeError(f"failed to move {source_path} to {destination_path}")


def build_terrain_regions_from_stage(
    stage: Usd.Stage,
    osm_path: Path,
    *,
    terrain_path: str = "/World/city/Terrain/Terrain",
    regions_path: str = "/World/TerrainRegions",
    road_depth: float = 0.0,
    cell_size: float = 25.0,
) -> dict[str, Any]:
    stored_terrain_path = f"{regions_path}/Terrain"
    if not stage.GetPrimAtPath(terrain_path) and stage.GetPrimAtPath(stored_terrain_path):
        stage.DefinePrim(Sdf.Path(terrain_path).GetParentPath(), "Xform")
        move_prim(stage, stored_terrain_path, terrain_path)

    terrain_prim = stage.GetPrimAtPath(terrain_path)
    if not terrain_prim or terrain_prim.GetTypeName() != "Mesh":
        raise RuntimeError(f"terrain mesh not found: {terrain_path}")
    if not osm_path.exists():
        raise RuntimeError(f"OSM file not found: {osm_path}")

    mesh = UsdGeom.Mesh(terrain_prim)
    points = list(mesh.GetPointsAttr().Get() or [])
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if not points or not counts or not indices:
        raise RuntimeError(f"terrain mesh has no authored geometry: {terrain_path}")
    previous_depth = terrain_prim.GetCustomDataByKey("navverse_terrain_regions_road_lowered_depth")
    if previous_depth is not None:
        raise RuntimeError(
            f"{terrain_path} has stale split-stage road lowering metadata; restore from backup or regenerate before rebuilding TerrainRegions"
        )
    face_count = len(counts)
    point_z_values = sorted(float(point[2]) for point in points)
    surface_z_threshold = float("-inf")
    if len(point_z_values) > 1:
        gaps = [point_z_values[index + 1] - point_z_values[index] for index in range(len(point_z_values) - 1)]
        gap_index = max(range(len(gaps)), key=gaps.__getitem__)
        if gaps[gap_index] >= 2.0:
            surface_z_threshold = (point_z_values[gap_index] + point_z_values[gap_index + 1]) * 0.5
    source_terrain_already_lowered = bool(
        terrain_prim.GetCustomDataByKey("navverse_usd_road_lowering_depth")
        or terrain_prim.GetCustomDataByKey("navverse_flat_ground")
    )
    uv_attr = terrain_prim.GetAttribute("primvars:UVMap")
    uv_values = uv_attr.Get() if uv_attr else None
    normal_attr = terrain_prim.GetAttribute("normals")
    normal_values = normal_attr.Get() if normal_attr else None
    normal_interpolation = normal_attr.GetMetadata("interpolation") if normal_attr else None
    if normal_values is not None and normal_interpolation != UsdGeom.Tokens.faceVarying:
        raise RuntimeError(
            f"unsupported terrain normal interpolation {normal_interpolation!r}; expected faceVarying"
        )
    material_faces = source_face_materials(terrain_prim, face_count)
    direct_material_path = UsdShade.MaterialBindingAPI(terrain_prim).GetDirectBinding().GetMaterialPath().pathString
    face_materials = {
        int(face_index): material_path
        for material_path, face_indices in material_faces.items()
        for face_index in face_indices
    }

    segments, polygons = parse_osm_regions(osm_path)
    road_segments = [segment for segment in segments if segment.category == "road"]
    road_segment_grid = build_segment_grid(road_segments, cell_size)
    primitives_by_category = parse_osm_clip_primitives(osm_path)
    grids_by_category = {
        category: build_primitive_grid(primitives, cell_size)
        for category, primitives in primitives_by_category.items()
    }
    region_data = {category: RegionMeshData() for category in REGION_ORDER}
    road_surface_data = RegionMeshData()
    source_triangle_count = 0
    source_surface_triangle_count = 0
    skipped_below_surface_triangles = 0
    skipped_non_curb_vertical_triangles = 0
    source_xy_area = 0.0
    offset = 0
    for face_index, count in enumerate(counts):
        corners = []
        for local_index in range(count):
            corner_offset = offset + local_index
            corners.append(
                ClipVertex(
                    Gf.Vec3f(points[indices[corner_offset]]),
                    Gf.Vec2f(uv_values[corner_offset]) if uv_values is not None else None,
                    Gf.Vec3f(normal_values[corner_offset]) if normal_values is not None else None,
                )
            )
        material_path = face_materials.get(face_index, direct_material_path)
        for triangle_index in range(1, len(corners) - 1):
            triangle = [corners[0], corners[triangle_index], corners[triangle_index + 1]]
            source_triangle_count += 1
            z_values = [float(vertex.point[2]) for vertex in triangle]
            if min(z_values) < surface_z_threshold:
                skipped_below_surface_triangles += 1
                continue
            xy_area = polygon_xy_area(triangle)
            if xy_area <= MIN_SURFACE_XY_AREA_M2:
                skipped_non_curb_vertical_triangles += 1
                continue
            source_surface_triangle_count += 1
            source_xy_area += xy_area
            for road_part in collect_polygon_inside_primitives(
                triangle,
                primitives_by_category["road"],
                grids_by_category["road"],
                cell_size,
            ):
                append_polygon(road_surface_data, road_part, material_path)
            split_triangle_into_regions(
                triangle,
                material_path,
                primitives_by_category,
                grids_by_category,
                cell_size,
                region_data,
            )
        offset += count

    lifted_curb_vertices = lift_curb_surface(
        region_data["curb"],
        road_segments,
        road_segment_grid,
        cell_size,
        max(float(road_depth or 0.0), 0.0),
    )
    generated_curb_walls = add_road_boundary_walls(
        road_surface_data,
        region_data["curb"],
        primitives_by_category["road"],
        grids_by_category["road"],
        road_segments,
        road_segment_grid,
        cell_size,
        max(float(road_depth or 0.0), 0.0),
    )

    if stage.GetPrimAtPath(regions_path):
        stage.RemovePrim(regions_path)
    parent = stage.DefinePrim(regions_path, "Xform")
    UsdGeom.Imageable(parent).CreateVisibilityAttr(UsdGeom.Tokens.inherited)

    mesh_counts: dict[str, int] = {}
    for category in REGION_ORDER:
        data = region_data[category]
        if not data.points:
            continue
        mesh_counts[category] = create_region_mesh(
            stage,
            f"{regions_path}/{category}",
            mesh,
            data,
        )

    combined_data = combine_region_data(region_data)
    create_region_mesh(stage, stored_terrain_path, mesh, combined_data)
    stored_terrain = stage.GetPrimAtPath(stored_terrain_path)
    UsdGeom.Imageable(stored_terrain).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    generated_xy_area = sum(data.xy_area for data in region_data.values())
    coverage_error = abs(generated_xy_area - source_xy_area)
    coverage_relative_error = coverage_error / max(source_xy_area, 1e-12)
    coverage_tolerance = max(1e-4, source_xy_area * 1e-6)
    if coverage_error > coverage_tolerance:
        raise RuntimeError(
            "terrain region coverage mismatch: "
            f"source_xy_area={source_xy_area}, generated_xy_area={generated_xy_area}, error={coverage_error}"
        )

    stage.RemovePrim(terrain_path)
    old_parent_path = Sdf.Path(terrain_path).GetParentPath()
    old_parent = stage.GetPrimAtPath(old_parent_path)
    if old_parent and not old_parent.GetChildren():
        stage.RemovePrim(old_parent_path)

    return {
        "terrain_path": stored_terrain_path,
        "regions_path": regions_path,
        "osm_path": str(osm_path),
        "road_depth": max(float(road_depth or 0.0), 0.0),
        "curb_width": DEFAULT_CURB_WIDTH_M,
        "source_terrain_lowered": False,
        "source_terrain_already_lowered": source_terrain_already_lowered,
        "region_geometry": "semantic_boundary_clipped_source_terrain",
        "unclassified_region": DEFAULT_TERRAIN_REGION,
        "region_visibility": "visible",
        "source_terrain_visibility": "invisible",
        "face_count": face_count,
        "source_triangle_count": source_triangle_count,
        "source_surface_triangle_count": source_surface_triangle_count,
        "surface_z_threshold": surface_z_threshold,
        "semantic_road_surface_triangle_count": len(road_surface_data.points) // 3,
        "lifted_curb_vertices": lifted_curb_vertices,
        "generated_curb_walls": generated_curb_walls,
        "skipped_below_surface_triangles": skipped_below_surface_triangles,
        "skipped_non_curb_vertical_triangles": skipped_non_curb_vertical_triangles,
        "source_xy_area": source_xy_area,
        "generated_xy_area": generated_xy_area,
        "coverage_error": coverage_error,
        "coverage_relative_error": coverage_relative_error,
        "coverage_tolerance": coverage_tolerance,
        "region_face_counts": {
            category: len(region_data[category].points) // 3 for category in REGION_ORDER
        },
        "region_xy_areas": {category: region_data[category].xy_area for category in REGION_ORDER},
        "region_mesh_counts": mesh_counts,
        "skipped_degenerate_triangles": {
            category: region_data[category].skipped_degenerate_triangles for category in REGION_ORDER
        },
        "clip_primitive_counts": {
            category: len(primitives_by_category[category]) for category in REGION_ORDER
        },
        "segment_count": len(segments),
        "polygon_count": len(polygons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a USD terrain mesh into semantic TerrainRegions meshes.")
    parser.add_argument("usd", type=Path)
    parser.add_argument("--osm", type=Path, required=True)
    parser.add_argument("--terrain-path", default="/World/city/Terrain/Terrain")
    parser.add_argument("--regions-path", default="/World/TerrainRegions")
    parser.add_argument("--road-depth", type=float, default=0.0)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        raise RuntimeError(f"could not open USD: {args.usd}")
    summary = build_terrain_regions_from_stage(
        stage,
        args.osm,
        terrain_path=args.terrain_path,
        regions_path=args.regions_path,
        road_depth=args.road_depth,
    )
    stage.GetRootLayer().Save()
    text = json.dumps({"usd": str(args.usd), **summary}, indent=2, sort_keys=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
