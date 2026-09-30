#!/usr/bin/env python3
"""Score VC building geometry for automatic storefront and GRScenes placement."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.data.objects if obj not in before]


def world_bbox(objects: list[bpy.types.Object]) -> dict[str, Any] | None:
    points: list[Vector] = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        return None
    mins = [min(p[i] for p in points) for i in range(3)]
    maxs = [max(p[i] for p in points) for i in range(3)]
    dims = [maxs[i] - mins[i] for i in range(3)]
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    return {"min": mins, "max": maxs, "dims": dims, "center": center, "footprint_area": dims[0] * dims[1]}


def distance_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_point(center: list[float], points: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    best = None
    best_dist = None
    for item in points:
        loc = [float(item.get("utm_x", item.get("location", [0, 0, 0])[0])), float(item.get("utm_y", item.get("location", [0, 0, 0])[1]))]
        dist = distance_xy(center, loc)
        if best_dist is None or dist < best_dist:
            best = item
            best_dist = dist
    return best, best_dist


def facade_from_bbox(bbox: dict[str, Any], anchor: dict[str, Any] | None) -> dict[str, Any]:
    min_x, min_y, min_z = bbox["min"]
    max_x, max_y, max_z = bbox["max"]
    cx, cy, _ = bbox["center"]
    if anchor:
        ax = float(anchor.get("utm_x", anchor.get("location", [cx, cy, 0])[0]))
        ay = float(anchor.get("utm_y", anchor.get("location", [cx, cy, 0])[1]))
    else:
        ax, ay = cx, min_y - 1.0
    candidates = [
        (abs(ax - min_x), "west", [min_x, min(max(ay, min_y), max_y), min_z], [-1.0, 0.0, 0.0], max_y - min_y),
        (abs(ax - max_x), "east", [max_x, min(max(ay, min_y), max_y), min_z], [1.0, 0.0, 0.0], max_y - min_y),
        (abs(ay - min_y), "south", [min(max(ax, min_x), max_x), min_y, min_z], [0.0, -1.0, 0.0], max_x - min_x),
        (abs(ay - max_y), "north", [min(max(ax, min_x), max_x), max_y, min_z], [0.0, 1.0, 0.0], max_x - min_x),
    ]
    _, side, base_center, normal, width = min(candidates, key=lambda x: x[0])
    opening_height = min(3.2, max(2.4, (max_z - min_z) * 0.35))
    opening_width = min(5.0, max(2.5, width * 0.35))
    return {
        "side": side,
        "center": [base_center[0], base_center[1], min_z + opening_height * 0.5],
        "normal": normal,
        "available_width": width,
        "opening_width": opening_width,
        "opening_height": opening_height,
    }


def load_grscene_catalog(root: Path) -> list[dict[str, Any]]:
    scenes = []
    for usd in sorted(root.glob("scenes/scene*/start_result_navigation.usd")):
        door_info = read_json(usd.parent / "door_info.json", None)
        if not door_info:
            continue
        doors = [door for door in door_info.get("doors", []) if door.get("likely_entry_door", True)]
        if not doors:
            continue
        scene_bbox = door_info.get("scene_bbox_xyz") or {}
        dims = scene_bbox.get("dims_xyz") or [8.0, 8.0, 3.0]
        scenes.append({
            "path": str(usd),
            "scene": usd.parent.name,
            "width": float(dims[0]),
            "depth": float(dims[1]),
            "height": float(dims[2]),
            "door_count": len(doors),
            "door_info": str(usd.parent / "door_info.json"),
            "source": "door_info",
        })
    return scenes


def grscene_with_scale(scene: dict[str, Any], bbox: dict[str, Any]) -> dict[str, Any]:
    bx, by, _bz = bbox["dims"]
    margin = 1.5
    fits_xy = scene["width"] + margin <= bx and scene["depth"] + margin <= by
    scale = min((bx - margin) / scene["width"], (by - margin) / scene["depth"]) if scene["width"] > 0 and scene["depth"] > 0 else 1.0
    chosen = dict(scene)
    chosen["scale"] = max(0.5, min(1.15, scale))
    chosen["fits_xy"] = fits_xy
    return chosen


def choose_grscene(catalog: list[dict[str, Any]], bbox: dict[str, Any], preferred_scene: str | None = None) -> dict[str, Any] | None:
    if not catalog:
        return None
    if preferred_scene:
        for scene in catalog:
            if scene["scene"] == preferred_scene:
                return grscene_with_scale(scene, bbox)

    fits = []
    for scene in catalog:
        chosen = grscene_with_scale(scene, bbox)
        fits.append((chosen["fits_xy"], chosen["scale"], chosen))
    fits.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return dict(fits[0][2])


def score_building(bbox: dict[str, Any], store_dist: float | None, sidewalk_dist: float | None) -> float:
    dx, dy, dz = bbox["dims"]
    area = bbox["footprint_area"]
    if dz < 2.5 or min(dx, dy) < 4.0 or area < 25.0:
        return -1e9
    area_score = min(area / 400.0, 2.0)
    height_score = min(dz / 12.0, 1.5)
    store_score = 2.5 / (1.0 + (store_dist or 80.0) / 20.0)
    sidewalk_score = 1.5 / (1.0 + (sidewalk_dist or 80.0) / 15.0)
    aspect = max(dx, dy) / max(min(dx, dy), 1e-6)
    aspect_penalty = max(0.0, aspect - 8.0) * 0.15
    return area_score + height_score + store_score + sidewalk_score - aspect_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--city-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--storefront-plan", type=Path)
    parser.add_argument("--sidewalk-points", type=Path, required=True)
    parser.add_argument("--grscene-root", type=Path, required=True)
    parser.add_argument("--preferred-grscene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-buildings", type=int, default=0)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    args = parser.parse_args(argv)

    buildings = sorted((args.city_dir / "buildings").glob("*.glb"))
    if args.max_buildings > 0:
        buildings = buildings[: args.max_buildings]
    store_points = []
    if args.storefront_plan and args.storefront_plan.exists():
        store_points = [
            item
            for item in (read_json(args.storefront_plan, {"placements": []}) or {}).get("placements", [])
            if item.get("status") == "planned"
            and item.get("facade_status") == "facade_aligned"
            and item.get("final_location")
        ]
    sidewalk_points = (read_json(args.sidewalk_points, {"points": []}) or {}).get("points", [])
    gr_catalog = load_grscene_catalog(args.grscene_root)
    candidates = []
    for index, building in enumerate(buildings):
        reset_scene()
        try:
            objects = import_glb(building)
            bbox = world_bbox(objects)
        except Exception as exc:
            candidates.append({"building_path": str(building), "status": "import_failed", "error": repr(exc)})
            continue
        if not bbox:
            candidates.append({"building_path": str(building), "status": "no_mesh"})
            continue
        store, store_dist = nearest_point(bbox["center"], store_points)
        sidewalk, sidewalk_dist = nearest_point(bbox["center"], sidewalk_points)
        score = score_building(bbox, store_dist, sidewalk_dist)
        facade = facade_from_bbox(bbox, store or sidewalk)
        grscene = choose_grscene(gr_catalog, bbox, args.preferred_grscene)
        candidates.append({
            "rank_input_index": index,
            "status": "scored",
            "building_path": str(building),
            "building_name": building.stem,
            "bbox": bbox,
            "nearest_storefront_distance": store_dist,
            "nearest_sidewalk_distance": sidewalk_dist,
            "nearest_storefront": store,
            "facade": facade,
            "selected_grscene": grscene,
            "score": score,
        })
    scored = [item for item in candidates if item.get("status") == "scored"]
    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = scored[0] if scored else None
    write_json(args.output, {
        "city": args.city,
        "method": "pre-content geometry scoring over vc_full building GLBs; vc_plus cut_buildings is not used",
        "selection_context": {
            "mode": "pre_content",
            "storefront_anchor_count": len(store_points),
        },
        "building_count_scanned": len(buildings),
        "storefront_anchor_count": len(store_points),
        "sidewalk_point_count": len(sidewalk_points),
        "grscene_catalog_count": len(gr_catalog),
        "grscene_catalog": gr_catalog,
        "preferred_grscene": args.preferred_grscene,
        "selected": selected,
        "top_candidates": scored[:20],
        "failed_candidates": [item for item in candidates if item.get("status") != "scored"][:50],
    })


if __name__ == "__main__":
    main()
