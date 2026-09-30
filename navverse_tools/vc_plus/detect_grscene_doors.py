#!/usr/bin/env python3
"""Detect semantic door meshes in GRScenes and write per-scene door_info.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from pxr import Gf, Usd, UsdGeom


DOOR_ROOT_PATH = "/Root/Meshes/BaseAnimation/door"


def is_environment_mesh(path: str) -> bool:
    lowered = path.lower()
    return "__default_setting" in lowered or "hdr_sphere" in lowered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write door_info.json files for GRScenes.")
    parser.add_argument(
        "--grscene-root",
        default="navverse_data/grscenes_commercial",
        help="GRScenes dataset root containing a scenes/ directory.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Optional scene folder names to scan, such as scene0 scene24. Defaults to all scene* folders.",
    )
    parser.add_argument("--output-name", default="door_info.json")
    return parser.parse_args()


def scene_dirs(grscene_root: Path, requested: list[str] | None) -> list[Path]:
    scenes_root = grscene_root / "scenes"
    if requested:
        return [scenes_root / name for name in requested]
    return sorted(
        (path for path in scenes_root.glob("scene*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("scene")) if path.name.removeprefix("scene").isdigit() else path.name,
    )


def iter_meshes(root: Usd.Prim) -> Iterable[Usd.Prim]:
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            yield prim


def transformed_points(mesh_prim: Usd.Prim) -> list[Gf.Vec3d]:
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
    if not points:
        return []
    matrix = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return [matrix.Transform(Gf.Vec3d(point[0], point[1], point[2])) for point in points]


def bbox_from_points(points: list[Gf.Vec3d]) -> tuple[list[float], list[float]]:
    min_xyz = [min(point[i] for point in points) for i in range(3)]
    max_xyz = [max(point[i] for point in points) for i in range(3)]
    return min_xyz, max_xyz


def transformed_extent_corners(mesh_prim: Usd.Prim) -> tuple[list[Gf.Vec3d], int]:
    mesh = UsdGeom.Mesh(mesh_prim)
    extent = mesh.GetExtentAttr().Get(Usd.TimeCode.Default())
    point_count = 0
    if extent and len(extent) == 2:
        local_min = [float(extent[0][i]) for i in range(3)]
        local_max = [float(extent[1][i]) for i in range(3)]
    else:
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
        point_count = len(points)
        if not points:
            return [], 0
        local_min = [min(float(point[i]) for point in points) for i in range(3)]
        local_max = [max(float(point[i]) for point in points) for i in range(3)]

    matrix = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    corners = []
    for x in (local_min[0], local_max[0]):
        for y in (local_min[1], local_max[1]):
            for z in (local_min[2], local_max[2]):
                corners.append(matrix.Transform(Gf.Vec3d(x, y, z)))
    return corners, point_count


def round_xyz(values: list[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def door_axis_guess(dims: list[float]) -> dict[str, object]:
    dx, dy, _dz = dims
    if dx <= dy:
        return {
            "normal_axis_guess": "x",
            "tangent_axis_guess": "y",
            "normal_xy_guess": [1.0, 0.0],
            "tangent_xy_guess": [0.0, 1.0],
        }
    return {
        "normal_axis_guess": "y",
        "tangent_axis_guess": "x",
        "normal_xy_guess": [0.0, 1.0],
        "tangent_xy_guess": [1.0, 0.0],
    }


def door_info_for_model(model_prim: Usd.Prim) -> dict[str, object] | None:
    points: list[Gf.Vec3d] = []
    mesh_count = 0
    for mesh_prim in iter_meshes(model_prim):
        mesh_points = transformed_points(mesh_prim)
        if mesh_points:
            mesh_count += 1
            points.extend(mesh_points)
    if not points:
        return None

    min_xyz, max_xyz = bbox_from_points(points)
    dims = [max_xyz[i] - min_xyz[i] for i in range(3)]
    center = [(min_xyz[i] + max_xyz[i]) * 0.5 for i in range(3)]
    horizontal_dims = sorted([dims[0], dims[1]])
    thickness = horizontal_dims[0]
    width = horizontal_dims[1]
    height = dims[2]
    likely_entry_door = height >= 1.5 and width >= 0.6 and thickness <= 1.0

    return {
        "prim_path": str(model_prim.GetPath()),
        "name": model_prim.GetName(),
        "bbox_min_xyz": round_xyz(min_xyz),
        "bbox_max_xyz": round_xyz(max_xyz),
        "center_xyz": round_xyz(center),
        "dims_xyz": round_xyz(dims),
        "width_m": round(float(width), 6),
        "height_m": round(float(height), 6),
        "thickness_m": round(float(thickness), 6),
        "mesh_count": mesh_count,
        "point_count": len(points),
        "likely_entry_door": likely_entry_door,
        **door_axis_guess(dims),
    }


def scene_bbox(stage: Usd.Stage) -> dict[str, object] | None:
    points: list[Gf.Vec3d] = []
    mesh_count = 0
    fallback_point_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if is_environment_mesh(str(prim.GetPath())):
            continue
        mesh_points, used_points = transformed_extent_corners(prim)
        if mesh_points:
            mesh_count += 1
            fallback_point_count += used_points
            points.extend(mesh_points)
    if not points:
        return None

    min_xyz, max_xyz = bbox_from_points(points)
    dims = [max_xyz[i] - min_xyz[i] for i in range(3)]
    center = [(min_xyz[i] + max_xyz[i]) * 0.5 for i in range(3)]
    return {
        "bbox_min_xyz": round_xyz(min_xyz),
        "bbox_max_xyz": round_xyz(max_xyz),
        "center_xyz": round_xyz(center),
        "dims_xyz": round_xyz(dims),
        "mesh_count": mesh_count,
        "extent_corner_count": len(points),
        "fallback_point_count": fallback_point_count,
        "excluded": ["__default_setting", "HDR_Sphere"],
    }


def scan_scene(scene_dir: Path, output_name: str) -> dict[str, object]:
    scene_usd = scene_dir / "start_result_navigation.usd"
    result: dict[str, object] = {
        "scene": scene_dir.name,
        "scene_usd": str(scene_usd),
        "door_root_path": DOOR_ROOT_PATH,
        "door_root_found": False,
        "door_model_count": 0,
        "valid_door_count": 0,
        "scene_bbox_xyz": None,
        "doors": [],
        "detector": {
            "method": "semantic /Root/Meshes/BaseAnimation/door children with mesh-point world-space bounding boxes",
            "caveat": "This detects modeled door objects and their positions, not whether the door is open or walkable.",
        },
    }
    if not scene_usd.exists():
        result["error"] = "missing start_result_navigation.usd"
        return result

    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        result["error"] = "failed to open USD stage"
        return result

    result["scene_bbox_xyz"] = scene_bbox(stage)
    door_root = stage.GetPrimAtPath(DOOR_ROOT_PATH)
    if not door_root or not door_root.IsValid():
        return result

    result["door_root_found"] = True
    doors = []
    children = list(door_root.GetChildren())
    result["door_model_count"] = len(children)
    for child in children:
        info = door_info_for_model(child)
        if info is not None:
            doors.append(info)

    result["doors"] = doors
    result["valid_door_count"] = len(doors)
    return result


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    grscene_root = Path(args.grscene_root)
    scenes = scene_dirs(grscene_root, args.scenes)
    if not scenes:
        raise SystemExit(f"No scenes found under {grscene_root / 'scenes'}")

    summaries = []
    for scene_dir in scenes:
        info = scan_scene(scene_dir, args.output_name)
        output = scene_dir / args.output_name
        write_json(output, info)
        summaries.append(
            {
                "scene": scene_dir.name,
                "door_root_found": info["door_root_found"],
                "valid_door_count": info["valid_door_count"],
                "output": str(output),
            }
        )

    total_doors = sum(int(row["valid_door_count"]) for row in summaries)
    print(
        json.dumps(
            {
                "scene_count": len(summaries),
                "total_valid_doors": total_doors,
                "scenes_with_doors": sum(1 for row in summaries if int(row["valid_door_count"]) > 0),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
