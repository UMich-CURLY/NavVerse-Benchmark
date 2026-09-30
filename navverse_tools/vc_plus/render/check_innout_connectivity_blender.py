#!/usr/bin/env python3
"""Probe line-of-sight corridor connectivity for a generated innout USD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether sampled doorway rays are blocked.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--usd")
    parser.add_argument("--glb")
    parser.add_argument("--output", required=True)
    parser.add_argument("--extent", type=float, default=7.0)
    parser.add_argument("--inside-extent", type=float, default=None)
    parser.add_argument("--outside-extent", type=float, default=None)
    parser.add_argument("--width-samples", type=int, default=5)
    parser.add_argument("--height-samples", type=int, default=4)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def object_path(obj: bpy.types.Object | None) -> str | None:
    if obj is None:
        return None
    names = [obj.name]
    parent = obj.parent
    while parent is not None:
        names.append(parent.name)
        parent = parent.parent
    return "/".join(reversed(names))


def ray_hits(origin: Vector, target: Vector, max_hits: int = 32) -> list[dict[str, object]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    direction = target - origin
    length = direction.length
    if length <= 0:
        return []
    direction.normalize()
    cursor = Vector(origin)
    travelled = 0.0
    hits: list[dict[str, object]] = []
    for _ in range(max_hits):
        ok, loc, normal, face_index, obj, _matrix = bpy.context.scene.ray_cast(
            depsgraph, cursor, direction, distance=max(0.0, length - travelled)
        )
        if not ok:
            break
        distance = (loc - origin).length
        hits.append(
            {
                "object": obj.name if obj else None,
                "object_path": object_path(obj),
                "distance": float(distance),
                "location": [float(loc.x), float(loc.y), float(loc.z)],
                "normal": [float(normal.x), float(normal.y), float(normal.z)],
                "face_index": int(face_index),
            }
        )
        cursor = loc + direction * 0.03
        travelled = (cursor - origin).length
        if travelled >= length:
            break
    return hits


def default_inside_extent(plan: dict[str, object], requested_extent: float) -> float:
    dimensions = plan.get("interior_scene_bbox_dims") or []
    if len(dimensions) < 2:
        return min(requested_extent, 3.0)
    room_scale = float(plan.get("room_effective_scale") or 1.0)
    safe_room_depth = min(float(dimensions[0]), float(dimensions[1])) * room_scale * 0.35
    return min(requested_extent, max(1.5, min(safe_room_depth, 3.0)))


def main() -> None:
    args = parse_args()
    plan = json.loads(Path(args.plan).read_text())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if args.usd:
        imported_path = Path(args.usd).resolve()
        bpy.ops.wm.usd_import(filepath=str(imported_path))
    elif args.glb:
        imported_path = Path(args.glb).resolve()
        bpy.ops.import_scene.gltf(filepath=str(imported_path))
    else:
        raise SystemExit("one of --usd or --glb is required")
    bpy.context.view_layer.update()

    door_xy = plan["door_center_xy"]
    door_bottom_z = float(plan["door_bottom_z"])
    door_width = float(plan["door_width"])
    door_height = float(plan["door_height"])
    outward = Vector((plan["outward_normal_xy"][0], plan["outward_normal_xy"][1], 0.0)).normalized()
    tangent = Vector((plan["tangent_xy"][0], plan["tangent_xy"][1], 0.0)).normalized()
    door_center = Vector((door_xy[0], door_xy[1], door_bottom_z))
    inside_extent = float(
        args.inside_extent
        if args.inside_extent is not None
        else default_inside_extent(plan, args.extent)
    )
    outside_extent = float(args.outside_extent if args.outside_extent is not None else args.extent)

    width_count = max(1, args.width_samples)
    height_count = max(1, args.height_samples)
    width_offsets = [0.0] if width_count == 1 else [
        (-0.42 * door_width) + i * (0.84 * door_width / (width_count - 1)) for i in range(width_count)
    ]
    height_offsets = [
        0.35 + i * ((max(0.35, door_height - 0.5)) / max(1, height_count - 1)) for i in range(height_count)
    ]

    samples = []
    unblocked = []
    for w in width_offsets:
        for h in height_offsets:
            center = door_center + tangent * w + Vector((0.0, 0.0, h))
            inside = center - outward * inside_extent
            outside = center + outward * outside_extent
            hits = ray_hits(inside, outside)
            blocked = len(hits) > 0
            row = {
                "width_offset": float(w),
                "height_offset": float(h),
                "inside": [float(inside.x), float(inside.y), float(inside.z)],
                "outside": [float(outside.x), float(outside.y), float(outside.z)],
                "blocked": blocked,
                "hit_count": len(hits),
                "first_hit": hits[0] if hits else None,
                "hits": hits[:8],
            }
            samples.append(row)
            if not blocked:
                unblocked.append(row)

    result = {
        "city": plan.get("city"),
        "input": str(imported_path),
        "plan": str(Path(args.plan).resolve()),
        "door_width": door_width,
        "door_height": door_height,
        "extent_each_side_m": args.extent,
        "inside_extent_m": inside_extent,
        "inside_extent_source": "explicit" if args.inside_extent is not None else "room_bbox",
        "outside_extent_m": outside_extent,
        "sample_count": len(samples),
        "unblocked_count": len(unblocked),
        "has_any_clear_line": bool(unblocked),
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ["sample_count", "unblocked_count", "has_any_clear_line"]}, indent=2))


if __name__ == "__main__":
    main()
