#!/usr/bin/env python3
"""Compress VC USD terrain heights and move scene objects to the new ground."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom


SURFACE_ROOTS = ("/World/city/Terrain", "/World/TerrainRegions")
OBJECT_PARENT_PATHS = ("/World/store", "/World/RoadObjects", "/World/Objaverse")
EXTRA_OBJECT_ROOT_PATHS = ("/World/start_result_navigation",)
CITY_OBJECT_EXCLUDES = {"Terrain", "_materials"}
MARKER_KEY = "navverse_flat_ground"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, nargs="+", help="USD files to edit in place")
    parser.add_argument("--height-field", type=Path, required=True)
    parser.add_argument("--osm", type=Path, default=None)
    parser.add_argument("--height-scale", type=float, default=0.3)
    parser.add_argument("--height-reference", choices=("min", "median"), default="min")
    parser.add_argument("--road-depth", type=float, default=0.2)
    parser.add_argument("--road-cell-size", type=float, default=25.0)
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="allow processing a stage that already has the marker")
    parser.add_argument("--summary-suffix", default=".flat_ground_0_3_summary.json")
    return parser.parse_args()


def height_stats(height_field: Path) -> tuple[float, float]:
    with np.load(height_field) as data:
        if "terrain_alt" in data:
            alt = data["terrain_alt"].astype(float)
        elif "height_field" in data:
            alt = data["height_field"].astype(float)
        else:
            alt = data[data.files[0]].astype(float)
    finite = alt[np.isfinite(alt)]
    if finite.size == 0:
        raise ValueError(f"no finite terrain heights in {height_field}")
    return float(np.min(finite)), float(np.median(finite))


def compressed_z(z: float, ground_z: float, scale: float) -> float:
    return ground_z + (float(z) - ground_z) * scale


def parse_float(value: str | None, default: float | None = 0.0) -> float | None:
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
    return tags.get("highway") in {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "living_street",
    } and tags.get("area") != "yes"


def build_road_segments(osm_path: Path | None) -> list[dict]:
    if osm_path is None or not osm_path.exists():
        return []
    root = ET.parse(osm_path).getroot()
    nodes: dict[str, tuple[float | None, float | None]] = {}
    ways = []
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
        half_width = max(terrain_region_width(tags) * 0.5, 0.1)
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
            segments.append(
                {
                    "start": (sx, sy),
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
    return segments


def build_segment_grid(segments: list[dict], cell_size: float) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    safe_cell = max(float(cell_size or 25.0), 1.0)
    for index, segment in enumerate(segments):
        min_x, min_y, max_x, max_y = segment["bounds"]
        for ix in range(math.floor(min_x / safe_cell), math.floor(max_x / safe_cell) + 1):
            for iy in range(math.floor(min_y / safe_cell), math.floor(max_y / safe_cell) + 1):
                grid.setdefault((ix, iy), []).append(index)
    return grid


def point_hits_segment(x: float, y: float, segment: dict) -> bool:
    sx, sy = segment["start"]
    dx, dy = segment["delta"]
    t = ((x - sx) * dx + (y - sy) * dy) / segment["length_sq"]
    t = max(0.0, min(1.0, t))
    px, py = sx + t * dx, sy + t * dy
    dist_sq = (x - px) * (x - px) + (y - py) * (y - py)
    return dist_sq <= segment["half_width"] * segment["half_width"]


def point_hits_any_road(
    x: float,
    y: float,
    segments: list[dict],
    grid: dict[tuple[int, int], list[int]],
    cell_size: float,
) -> bool:
    if not segments:
        return False
    safe_cell = max(float(cell_size or 25.0), 1.0)
    cell = (math.floor(x / safe_cell), math.floor(y / safe_cell))
    return any(point_hits_segment(x, y, segments[index]) for index in grid.get(cell, []))


def bbox_cache() -> UsdGeom.BBoxCache:
    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )


def path_bbox(stage: Usd.Stage, path: str) -> dict[str, list[float]] | None:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    box = bbox_cache().ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return None
    return {
        "min": [round(float(v), 6) for v in box.GetMin()],
        "max": [round(float(v), 6) for v in box.GetMax()],
    }


def key_bboxes(stage: Usd.Stage) -> dict[str, dict[str, list[float]]]:
    paths = (
        "/World",
        "/World/city",
        "/World/city/Terrain",
        "/World/TerrainRegions",
        "/World/city/Roof",
        "/World/store",
        "/World/RoadObjects",
        "/World/Objaverse",
        "/World/start_result_navigation",
    )
    return {path: box for path in paths if (box := path_bbox(stage, path))}


def surface_mesh_paths(stage: Usd.Stage) -> list[str]:
    paths = []
    for root_path in SURFACE_ROOTS:
        root = stage.GetPrimAtPath(root_path)
        if not root:
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                paths.append(prim.GetPath().pathString)
    return paths


def compress_mesh_points(
    stage: Usd.Stage,
    mesh_paths: list[str],
    ground_z: float,
    height_scale: float,
    road_segments: list[dict],
    road_depth: float,
    road_cell_size: float,
) -> dict[str, int]:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    grid = build_segment_grid(road_segments, road_cell_size)
    changed = 0
    road_vertices = 0
    total_vertices = 0
    for path in mesh_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        local_to_world = cache.GetLocalToWorldTransform(prim)
        world_to_local = local_to_world.GetInverse()
        new_points = []
        for point in points:
            world = local_to_world.Transform(Gf.Vec3d(point[0], point[1], point[2]))
            z = compressed_z(world[2], ground_z, height_scale)
            if road_depth > 0 and point_hits_any_road(float(world[0]), float(world[1]), road_segments, grid, road_cell_size):
                z -= road_depth
                road_vertices += 1
            world = Gf.Vec3d(world[0], world[1], z)
            new_points.append(world_to_local.Transform(world))
        mesh.GetPointsAttr().Set(new_points)
        changed += 1
        total_vertices += len(new_points)
    return {"compressed_meshes": changed, "surface_vertices": total_vertices, "road_vertices": road_vertices}


def object_roots(stage: Usd.Stage) -> list[Usd.Prim]:
    roots = []
    seen = set()

    def add(prim: Usd.Prim) -> None:
        path = prim.GetPath().pathString
        if path not in seen:
            seen.add(path)
            roots.append(prim)

    for parent_path in OBJECT_PARENT_PATHS:
        parent = stage.GetPrimAtPath(parent_path)
        if parent:
            for child in parent.GetChildren():
                add(child)
    for root_path in EXTRA_OBJECT_ROOT_PATHS:
        prim = stage.GetPrimAtPath(root_path)
        if prim:
            add(prim)
    city = stage.GetPrimAtPath("/World/city")
    if city:
        for child in city.GetChildren():
            if child.GetName() not in CITY_OBJECT_EXCLUDES:
                add(child)
    return roots


def bbox_ref_z(prim: Usd.Prim) -> float | None:
    box = bbox_cache().ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return None
    z = float(box.GetMin()[2])
    return z if math.isfinite(z) else None


def local_delta_for_parent(prim: Usd.Prim, world_delta_z: float) -> Gf.Vec3d:
    parent = prim.GetParent()
    if not parent:
        return Gf.Vec3d(0.0, 0.0, world_delta_z)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    parent_world = cache.GetLocalToWorldTransform(parent)
    return parent_world.GetInverse().TransformDir(Gf.Vec3d(0.0, 0.0, world_delta_z))


def translate_root(prim: Usd.Prim, world_delta_z: float) -> bool:
    if abs(world_delta_z) < 1e-8 or not prim or not prim.IsA(UsdGeom.Xformable):
        return False
    local_delta = local_delta_for_parent(prim, world_delta_z)
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        op = translate_ops[0]
        value = op.Get()
        op.Set(Gf.Vec3d(value[0] + local_delta[0], value[1] + local_delta[1], value[2] + local_delta[2]))
    else:
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "flatGroundShift").Set(local_delta)
    return True


def move_object_roots(stage: Usd.Stage, ground_z: float, height_scale: float) -> dict[str, object]:
    moved = []
    skipped = 0
    for prim in object_roots(stage):
        ref_z = bbox_ref_z(prim)
        if ref_z is None:
            skipped += 1
            continue
        target_z = compressed_z(ref_z, ground_z, height_scale)
        delta = target_z - ref_z
        if translate_root(prim, delta):
            moved.append(
                {
                    "path": prim.GetPath().pathString,
                    "old_ref_z": round(ref_z, 6),
                    "new_ref_z": round(target_z, 6),
                    "delta_z": round(delta, 6),
                }
            )
    deltas = [item["delta_z"] for item in moved]
    refs = [item["new_ref_z"] for item in moved]
    return {
        "moved_roots": len(moved),
        "skipped_roots": skipped,
        "delta_z_min": min(deltas) if deltas else 0.0,
        "delta_z_max": max(deltas) if deltas else 0.0,
        "new_ref_z_min": min(refs) if refs else None,
        "new_ref_z_max": max(refs) if refs else None,
        "samples": moved[:20],
    }


def process_usd(
    usd_path: Path,
    ground_z: float,
    height_scale: float,
    road_segments: list[dict],
    road_depth: float,
    road_cell_size: float,
    backup: bool,
    force: bool,
    summary_suffix: str,
) -> dict[str, object]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD stage: {usd_path}")
    world = stage.GetDefaultPrim() or stage.GetPrimAtPath("/World")
    marker = world.GetCustomDataByKey(MARKER_KEY) if world else None
    if marker and not force:
        raise RuntimeError(f"{usd_path} already has {MARKER_KEY}; use --force to reprocess")

    before = key_bboxes(stage)
    meshes = surface_mesh_paths(stage)
    mesh_summary = compress_mesh_points(stage, meshes, ground_z, height_scale, road_segments, road_depth, road_cell_size)
    object_summary = move_object_roots(stage, ground_z, height_scale)
    after = key_bboxes(stage)

    summary = {
        "usd": str(usd_path),
        "height_scale": height_scale,
        "ground_z": ground_z,
        "road_depth": road_depth,
        "road_cell_size": road_cell_size,
        "road_segments": len(road_segments),
        "surface_mesh_paths": meshes,
        "before_bboxes": before,
        "after_bboxes": after,
        **mesh_summary,
        **object_summary,
    }
    if world:
        world.SetCustomDataByKey(
            MARKER_KEY,
            {
                "height_scale": float(height_scale),
                "ground_z": float(ground_z),
                "road_depth": float(road_depth),
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    backup_path = None
    if backup:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        backup_path = usd_path.with_name(f"{usd_path.stem}.bak_flat_ground_{timestamp}{usd_path.suffix}")
        shutil.copy2(usd_path, backup_path)
        summary["backup"] = str(backup_path)
    stage.GetRootLayer().Save()
    summary_path = usd_path.with_name(f"{usd_path.name}{summary_suffix}")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    min_height, median_height = height_stats(args.height_field.resolve())
    ground_z = min_height if args.height_reference == "min" else median_height
    road_segments = build_road_segments(args.osm.resolve() if args.osm else None)
    summaries = []
    for usd_path in args.usd:
        summaries.append(
            process_usd(
                usd_path.resolve(),
                ground_z,
                args.height_scale,
                road_segments,
                args.road_depth,
                args.road_cell_size,
                args.backup,
                args.force,
                args.summary_suffix,
            )
        )
    print(json.dumps({"height_field": str(args.height_field.resolve()), "osm": str(args.osm.resolve()) if args.osm else None, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
