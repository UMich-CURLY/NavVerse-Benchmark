#!/usr/bin/env python3
"""Place a green exit sign above the composed innout doorway."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
except ImportError as exc:
    raise SystemExit("pxr is required; run this script from the navverse USD environment.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXIT_SIGN_ASSET = REPO_ROOT / "navverse_data" / "objaverse" / "usd" / "green_exit_sign.usd"
EXIT_SIGN_PARENT = "/World/Objaverse/NavVerseExitSigns"
EXIT_SIGN_PRIM = f"{EXIT_SIGN_PARENT}/green_exit_sign_0000"
EXIT_SIGN_BACKING_PRIM = f"{EXIT_SIGN_PRIM}/ExteriorBacking"
EXIT_SIGN_BACKING_MATERIAL = f"{EXIT_SIGN_PARENT}/Looks/ExteriorBacking"
EXIT_SIGN_WALL_CLEARANCE_M = 0.015
EXIT_SIGN_MIN_WIDTH_M = 0.25
EXIT_SIGN_MAX_VERTICAL_OVERHANG_M = 0.12


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_xy(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] < 2:
        raise ValueError(f"expected at least 2 values, got {values}")
    xy = arr[:2]
    norm = np.linalg.norm(xy)
    if norm <= 1e-9:
        raise ValueError(f"zero-length xy vector: {values}")
    return xy / norm


def first_root_prim(path: Path) -> str | None:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return None
    default = stage.GetDefaultPrim()
    if default:
        return default.GetPath().pathString
    roots = list(stage.GetPseudoRoot().GetChildren())
    return roots[0].GetPath().pathString if roots else None


def asset_bbox(path: Path) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open exit sign asset: {path}")
    root = stage.GetDefaultPrim()
    if root is None:
        roots = list(stage.GetPseudoRoot().GetChildren())
        root = roots[0] if roots else None
    if root is None:
        raise RuntimeError(f"exit sign asset has no root prim: {path}")
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root).ComputeAlignedBox()
    return box.GetMin(), box.GetMax()


def transformed_interior_door_anchor(plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    center = np.asarray(plan["interior_door_center"], dtype=float)
    room_translate = np.asarray(plan["room_translate"], dtype=float)
    scale = float(plan.get("room_effective_scale") or 1.0)
    yaw = math.radians(float(plan["room_yaw_deg"]))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    center_xy = room_translate[:2] + rotation @ (center[:2] * scale)

    outer_center_xy = np.asarray(plan["door_center_xy"], dtype=float)
    outward = normalize_xy(plan["outward_normal_xy"])
    door_bbox_min = np.asarray(plan["interior_door_bbox_min"], dtype=float)
    door_bbox_max = np.asarray(plan["interior_door_bbox_max"], dtype=float)
    bbox_corners = np.asarray(
        [
            [door_bbox_min[0], door_bbox_min[1]],
            [door_bbox_min[0], door_bbox_max[1]],
            [door_bbox_max[0], door_bbox_min[1]],
            [door_bbox_max[0], door_bbox_max[1]],
        ],
        dtype=float,
    )
    world_corners = room_translate[:2] + (rotation @ (bbox_corners * scale).T).T
    surface_offset = min(float(np.dot(corner - outer_center_xy, outward)) for corner in world_corners)
    center_offset = float(np.dot(center_xy - outer_center_xy, outward))
    inward_surface_xy = center_xy + outward * (surface_offset - center_offset)

    scene_bbox_max = np.asarray(plan["interior_scene_bbox_max"], dtype=float)
    door_top_z = float(room_translate[2] + door_bbox_max[2] * scale)
    ceiling_z = float(room_translate[2] + scene_bbox_max[2] * scale)
    return center_xy, inward_surface_xy, door_top_z, ceiling_z


def inward_face_topology(
    points: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
) -> tuple[list[int], list[int]]:
    kept_counts: list[int] = []
    kept_indices: list[int] = []
    cursor = 0
    for count in counts:
        face_indices = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        face_points = points[face_indices]
        normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9 or float(normal[1] / norm) > -0.9:
            continue
        kept_counts.append(int(count))
        kept_indices.extend(int(value) for value in face_indices)
    return kept_counts, kept_indices


def make_exit_sign_interior_only(stage: Usd.Stage, sign_prim: Usd.Prim) -> int:
    """Keep only the textured face whose local normal points into the room."""
    selected_faces = 0
    for prim in Usd.PrimRange(sign_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if points.size == 0 or counts.size == 0 or indices.size == 0:
            continue

        kept_counts, kept_indices = inward_face_topology(points, counts, indices)

        if not kept_counts:
            raise RuntimeError(f"exit sign mesh has no inward-facing surface: {prim.GetPath()}")
        mesh.GetFaceVertexCountsAttr().Set(kept_counts)
        mesh.GetFaceVertexIndicesAttr().Set(kept_indices)
        mesh.CreateDoubleSidedAttr(False)

        st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        if st and st.Get() is not None and not st.IsIndexed():
            values = list(st.Get())
            for index in set(kept_indices):
                values[index] = Gf.Vec2f(1.0 - float(values[index][0]), float(values[index][1]))
            st.Set(values)
        selected_faces += len(kept_counts)
    if selected_faces == 0:
        raise RuntimeError(f"exit sign reference contains no mesh: {sign_prim.GetPath()}")
    return selected_faces


def add_exit_sign_exterior_backing(stage: Usd.Stage, bbox_min: Gf.Vec3d, bbox_max: Gf.Vec3d) -> str:
    stage.DefinePrim(f"{EXIT_SIGN_PARENT}/Looks", "Scope")
    material = UsdShade.Material.Define(stage, EXIT_SIGN_BACKING_MATERIAL)
    shader = UsdShade.Shader.Define(stage, f"{EXIT_SIGN_BACKING_MATERIAL}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.86, 0.86, 0.84))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    margin = 0.04
    y = float(bbox_max[1]) + 0.004
    x0 = float(bbox_min[0]) - margin
    x1 = float(bbox_max[0]) + margin
    z0 = float(bbox_min[2]) - margin
    z1 = float(bbox_max[2]) + margin
    mesh = UsdGeom.Mesh.Define(stage, EXIT_SIGN_BACKING_PRIM)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(x0, y, z0),
            Gf.Vec3f(x0, y, z1),
            Gf.Vec3f(x1, y, z1),
            Gf.Vec3f(x1, y, z0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    return EXIT_SIGN_BACKING_PRIM


def mesh_world_triangles(mesh: UsdGeom.Mesh, xform_cache: UsdGeom.XformCache) -> np.ndarray:
    point_values = mesh.GetPointsAttr().Get()
    count_values = mesh.GetFaceVertexCountsAttr().Get()
    index_values = mesh.GetFaceVertexIndicesAttr().Get()
    if not point_values or not count_values or not index_values:
        return np.empty((0, 3, 3), dtype=float)
    points = np.asarray(point_values, dtype=float)
    matrix = np.asarray(xform_cache.GetLocalToWorldTransform(mesh.GetPrim()), dtype=float)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)]) @ matrix
    points = homogeneous[:, :3] / homogeneous[:, 3, None]
    counts = np.asarray(count_values, dtype=int)
    indices = np.asarray(index_values, dtype=int)
    if np.all(counts == 3):
        return points[indices.reshape(-1, 3)]

    triangles: list[np.ndarray] = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        for index in range(1, int(count) - 1):
            triangles.append(points[[face[0], face[index], face[index + 1]]])
    return np.asarray(triangles, dtype=float)


def first_ray_hit(origin: np.ndarray, direction: np.ndarray, length: float, triangles: np.ndarray) -> float | None:
    if triangles.size == 0:
        return None
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(determinant) > 1e-8
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    offset = origin - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", offset, h)
    valid &= (u >= 0.0) & (u <= 1.0)
    q = np.cross(offset, edge1)
    v = inverse * np.einsum("j,ij->i", direction, q)
    valid &= (v >= 0.0) & (u + v <= 1.0)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    valid &= (distance > 1e-4) & (distance < length - 1e-4)
    return float(np.min(distance[valid])) if np.any(valid) else None


def collect_room_mesh_triangles(room: Usd.Prim, path_fragment: str) -> np.ndarray:
    xform_cache = UsdGeom.XformCache()
    chunks = []
    for prim in Usd.PrimRange(room):
        if not prim.IsA(UsdGeom.Mesh) or path_fragment not in prim.GetPath().pathString:
            continue
        triangles = mesh_world_triangles(UsdGeom.Mesh(prim), xform_cache)
        if triangles.size:
            chunks.append(triangles)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3, 3), dtype=float)


def probe_ceiling_heights(
    plan: dict[str, Any],
    inward_surface_xy: np.ndarray,
    ceiling_triangles: np.ndarray,
) -> list[float]:
    outward = normalize_xy(plan["outward_normal_xy"])
    tangent = normalize_xy(plan["tangent_xy"])
    origin_z = float(plan["door_bottom_z"]) + 0.5
    hits = []
    for inward in (0.03, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5):
        for lateral in (-0.3, 0.0, 0.3):
            xy = inward_surface_xy - outward * inward + tangent * lateral
            origin = np.asarray([float(xy[0]), float(xy[1]), origin_z], dtype=float)
            distance = first_ray_hit(origin, np.asarray([0.0, 0.0, 1.0]), 6.0, ceiling_triangles)
            if distance is not None:
                hits.append(origin_z + distance)
    return hits


def refine_sign_height_from_stage(stage: Usd.Stage, plan: dict[str, Any], transform: dict[str, Any]) -> None:
    room = stage.GetPrimAtPath("/World/start_result_navigation")
    if not room:
        return
    _, inward_surface_xy, door_top_z, _ = transformed_interior_door_anchor(plan)
    outward = normalize_xy(plan["outward_normal_xy"])
    ceiling_triangles = collect_room_mesh_triangles(room, "/Meshes/Base/ceiling/")
    if not ceiling_triangles.size:
        return

    ceiling_hits = probe_ceiling_heights(plan, inward_surface_xy, ceiling_triangles)
    if not ceiling_hits:
        return

    actual_ceiling_z = min(ceiling_hits)
    actual_clearance = max(0.0, actual_ceiling_z - door_top_z)
    current_width = float(transform["target_width_m"])
    current_height = float(transform["target_height_m"])
    current_depth = float(transform["target_depth_m"])
    aspect = current_height / current_width
    max_height = actual_clearance + EXIT_SIGN_MAX_VERTICAL_OVERHANG_M
    refined_width = max(EXIT_SIGN_MIN_WIDTH_M, min(current_width, max_height / aspect))
    scale_factor = refined_width / current_width
    refined_height = current_height * scale_factor
    refined_depth = current_depth * scale_factor

    old_face_offset = float(transform["signed_offset_from_door_center_m"]) - current_depth * 0.5
    refined_center_offset = old_face_offset + refined_depth * 0.5
    center_delta = refined_center_offset - float(transform["signed_offset_from_door_center_m"])
    transform["translation"][0] += float(outward[0] * center_delta)
    transform["translation"][1] += float(outward[1] * center_delta)
    transform["signed_offset_from_door_center_m"] = refined_center_offset

    old_bottom = float(transform["sign_bottom_z"])
    if refined_height <= actual_clearance:
        refined_bottom = max(door_top_z, min(door_top_z + 0.035, actual_ceiling_z - refined_height))
    else:
        refined_bottom = actual_ceiling_z - refined_height
    transform["translation"][2] += refined_bottom - old_bottom
    transform["scale"] = [float(value * scale_factor) for value in transform["scale"]]
    transform["target_width_m"] = refined_width
    transform["target_height_m"] = refined_height
    transform["target_depth_m"] = refined_depth
    transform["estimated_ceiling_z"] = float(transform["ceiling_z"])
    transform["ceiling_z"] = actual_ceiling_z
    transform["wall_clearance_z_m"] = actual_clearance
    transform["sign_bottom_z"] = refined_bottom
    transform["sign_top_z"] = refined_bottom + refined_height
    transform["vertical_overhang_below_m"] = max(0.0, door_top_z - refined_bottom)
    transform["vertical_overhang_above_m"] = max(0.0, refined_bottom + refined_height - actual_ceiling_z)
    transform["ceiling_probe_hit_count"] = len(ceiling_hits)


def doorway_sign_transform(plan: dict[str, Any], asset_path: Path) -> dict[str, Any]:
    bbox_min, bbox_max = asset_bbox(asset_path)
    asset_width = max(float(bbox_max[0] - bbox_min[0]), 1e-6)
    asset_depth = max(float(bbox_max[1] - bbox_min[1]), 1e-6)
    asset_height = max(float(bbox_max[2] - bbox_min[2]), 1e-6)

    outer_center_xy = np.asarray(plan["door_center_xy"], dtype=float)
    tangent = normalize_xy(plan["tangent_xy"])
    outward = normalize_xy(plan["outward_normal_xy"])
    door_width = float(plan.get("door_width") or 2.4)
    center_xy, inward_surface_xy, door_top_z, ceiling_z = transformed_interior_door_anchor(plan)

    target_width = min(max(door_width * 0.22, 0.42), 0.55)
    aspect = asset_height / asset_width
    wall_clearance = max(0.0, ceiling_z - door_top_z)
    max_height_with_overhang = wall_clearance + EXIT_SIGN_MAX_VERTICAL_OVERHANG_M
    target_width = max(EXIT_SIGN_MIN_WIDTH_M, min(target_width, max_height_with_overhang / aspect))
    scale = target_width / asset_width
    sign_depth = asset_depth * scale
    sign_height = asset_height * scale
    if sign_height <= wall_clearance:
        preferred_bottom = door_top_z + 0.035
        sign_bottom = max(door_top_z, min(preferred_bottom, ceiling_z - sign_height))
    else:
        sign_bottom = ceiling_z - sign_height

    wall_offset = float(np.dot(center_xy - outer_center_xy, outward))
    wall_surface_offset = float(np.dot(inward_surface_xy - outer_center_xy, outward))
    desired_center_signed = wall_surface_offset - EXIT_SIGN_WALL_CLEARANCE_M + sign_depth * 0.5
    sign_center_xy = outer_center_xy + outward * desired_center_signed
    yaw_deg = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
    mapped_local_y = np.asarray([-tangent[1], tangent[0]], dtype=float)
    if float(np.dot(mapped_local_y, outward)) < 0.0:
        yaw_deg += 180.0
    yaw = math.radians(yaw_deg)
    rot = np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=float)
    local_center_xy = np.asarray(
        [
            float((bbox_min[0] + bbox_max[0]) * 0.5),
            float((bbox_min[1] + bbox_max[1]) * 0.5),
        ],
        dtype=float,
    )
    scale_xy = np.asarray([scale, scale], dtype=float)
    root_xy = sign_center_xy - rot @ (local_center_xy * scale_xy)
    root_z = sign_bottom - float(bbox_min[2]) * scale

    return {
        "translation": [float(root_xy[0]), float(root_xy[1]), float(root_z)],
        "rotation_xyz_deg": [0.0, 0.0, float(yaw_deg)],
        "scale": [float(scale), float(scale), float(scale)],
        "mount_source": "transformed_interior_door_wall",
        "interior_door_wall_offset_m": float(wall_offset),
        "interior_door_wall_surface_offset_m": float(wall_surface_offset),
        "wall_surface_clearance_m": float(EXIT_SIGN_WALL_CLEARANCE_M),
        "target_width_m": float(target_width),
        "target_depth_m": float(sign_depth),
        "target_height_m": float(sign_height),
        "door_top_z": float(door_top_z),
        "ceiling_z": float(ceiling_z),
        "wall_clearance_z_m": float(wall_clearance),
        "sign_bottom_z": float(sign_bottom),
        "sign_top_z": float(sign_bottom + sign_height),
        "vertical_overhang_below_m": float(max(0.0, door_top_z - sign_bottom)),
        "vertical_overhang_above_m": float(max(0.0, sign_bottom + sign_height - ceiling_z)),
        "signed_offset_from_door_center_m": float(desired_center_signed),
    }


def apply_exit_sign_to_stage(
    stage: Usd.Stage,
    plan: dict[str, Any],
    *,
    output_dir: Path,
    asset_path: Path = DEFAULT_EXIT_SIGN_ASSET,
) -> dict[str, Any]:
    if not asset_path.exists():
        raise FileNotFoundError(f"missing exit sign asset: {asset_path}")

    if stage.GetPrimAtPath(EXIT_SIGN_PRIM):
        stage.RemovePrim(EXIT_SIGN_PRIM)
    stage.DefinePrim(EXIT_SIGN_PARENT, "Xform")

    xform_prim = stage.DefinePrim(EXIT_SIGN_PRIM, "Xform")
    xform = UsdGeom.XformCommonAPI(xform_prim)
    transform = doorway_sign_transform(plan, asset_path)
    refine_sign_height_from_stage(stage, plan, transform)
    xform.SetTranslate(Gf.Vec3d(*transform["translation"]))
    xform.SetRotate(Gf.Vec3f(*transform["rotation_xyz_deg"]))
    xform.SetScale(Gf.Vec3f(*transform["scale"]))

    root_prim = first_root_prim(asset_path)
    rel_path = os.path.relpath(asset_path, output_dir)
    if root_prim:
        xform_prim.GetReferences().AddReference(rel_path, root_prim)
    else:
        xform_prim.GetReferences().AddReference(rel_path)
    interior_face_count = make_exit_sign_interior_only(stage, xform_prim)
    backing_prim = add_exit_sign_exterior_backing(stage, *asset_bbox(asset_path))
    xform_prim.SetCustomDataByKey("navverse_asset_type", "green_exit_sign")
    xform_prim.SetCustomDataByKey("navverse_mount", "innout_interior_door_above")

    summary = {
        "prim_path": EXIT_SIGN_PRIM,
        "asset_path": rel_path,
        "interior_face_count": interior_face_count,
        "exterior_backing_prim": backing_prim,
        "rendering_mode": "interior_face_with_neutral_exterior_backing",
        **transform,
    }
    xform_prim.SetCustomDataByKey("navverse_summary", summary)
    return summary


def apply_exit_sign_to_usd(usd_path: Path, plan_path: Path, asset_path: Path = DEFAULT_EXIT_SIGN_ASSET) -> dict[str, Any]:
    plan = read_json(plan_path)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {usd_path}")
    summary = apply_exit_sign_to_stage(stage, plan, output_dir=usd_path.parent, asset_path=asset_path)
    stage.GetRootLayer().Save()
    return {"usd": str(usd_path), "plan": str(plan_path), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--asset", type=Path, default=DEFAULT_EXIT_SIGN_ASSET)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    summary = apply_exit_sign_to_usd(args.usd, args.plan, args.asset)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
