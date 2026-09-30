#!/usr/bin/env python3
"""Scale VC store meshes uniformly while keeping their bases on the ground."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, nargs="+", help="USD files to edit in place")
    parser.add_argument("--store-root", default="/World/store")
    parser.add_argument("--target-height", type=float, default=3.0)
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-extents-only", action="store_true")
    parser.add_argument("--summary-suffix", default=".store_scale_3m_summary.json")
    return parser.parse_args()


def bbox_cache() -> UsdGeom.BBoxCache:
    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )


def world_bbox(prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
    box = bbox_cache().ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return None
    return box.GetMin(), box.GetMax()


def local_points_extent(points) -> tuple[Gf.Vec3f, Gf.Vec3f] | None:
    if not points:
        return None
    arr = np.array([[point[0], point[1], point[2]] for point in points], dtype=float)
    min_pt = arr.min(axis=0)
    max_pt = arr.max(axis=0)
    return Gf.Vec3f(*min_pt.tolist()), Gf.Vec3f(*max_pt.tolist())


def update_mesh_extent(mesh: UsdGeom.Mesh) -> bool:
    extent = local_points_extent(mesh.GetPointsAttr().Get())
    if extent is None:
        return False
    mesh.GetExtentAttr().Set([extent[0], extent[1]])
    return True


def store_dimensions(stage: Usd.Stage, store_root: str) -> list[dict[str, object]]:
    parent = stage.GetPrimAtPath(store_root)
    if not parent:
        return []
    rows = []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for child in parent.GetChildren():
        world_points = []
        for mesh_prim in descendant_meshes(child):
            mesh = UsdGeom.Mesh(mesh_prim)
            points = mesh.GetPointsAttr().Get()
            if not points:
                continue
            local_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
            for point in points:
                world_points.append(local_to_world.Transform(Gf.Vec3d(point[0], point[1], point[2])))
        if not world_points:
            bbox = world_bbox(child)
            if bbox is None:
                continue
            min_pt, max_pt = bbox
        else:
            arr = np.array([[point[0], point[1], point[2]] for point in world_points], dtype=float)
            min_pt = arr.min(axis=0)
            max_pt = arr.max(axis=0)
        dims = [float(max_pt[i] - min_pt[i]) for i in range(3)]
        if dims[2] <= 1e-6:
            continue
        rows.append(
            {
                "path": child.GetPath().pathString,
                "dimensions": [round(v, 6) for v in dims],
                "base_z": round(float(min_pt[2]), 6),
                "front_width": round(max(dims[0], dims[1]), 6),
                "depth": round(min(dims[0], dims[1]), 6),
            }
        )
    return rows


def summarize_dimensions(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"count": 0}
    values = np.array(
        [
            [
                row["front_width"],
                row["depth"],
                row["dimensions"][2],
                row["base_z"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    keys = ("front_width", "depth", "height", "base_z")
    summary: dict[str, object] = {"count": len(rows)}
    for index, key in enumerate(keys):
        summary[key] = {
            "min": round(float(np.min(values[:, index])), 6),
            "median": round(float(np.median(values[:, index])), 6),
            "max": round(float(np.max(values[:, index])), 6),
        }
    return summary


def descendant_meshes(root: Usd.Prim) -> list[Usd.Prim]:
    return [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]


def scale_store_meshes(stage: Usd.Stage, store_root: str, target_height: float) -> list[dict[str, object]]:
    parent = stage.GetPrimAtPath(store_root)
    if not parent:
        raise RuntimeError(f"missing store root: {store_root}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    results = []
    for store in parent.GetChildren():
        bbox = world_bbox(store)
        if bbox is None:
            continue
        min_pt, max_pt = bbox
        current_height = float(max_pt[2] - min_pt[2])
        if current_height <= 1e-6:
            continue
        scale = float(target_height) / current_height
        anchor = Gf.Vec3d(
            (float(min_pt[0]) + float(max_pt[0])) * 0.5,
            (float(min_pt[1]) + float(max_pt[1])) * 0.5,
            float(min_pt[2]),
        )
        changed_meshes = 0
        changed_points = 0
        for mesh_prim in descendant_meshes(store):
            mesh = UsdGeom.Mesh(mesh_prim)
            points = mesh.GetPointsAttr().Get()
            if not points:
                continue
            local_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
            world_to_local = local_to_world.GetInverse()
            new_points = []
            for point in points:
                world = local_to_world.Transform(Gf.Vec3d(point[0], point[1], point[2]))
                scaled = anchor + (world - anchor) * scale
                new_points.append(world_to_local.Transform(scaled))
            mesh.GetPointsAttr().Set(new_points)
            update_mesh_extent(mesh)
            changed_meshes += 1
            changed_points += len(new_points)
        results.append(
            {
                "path": store.GetPath().pathString,
                "old_height": round(current_height, 6),
                "scale": round(scale, 6),
                "base_z": round(float(min_pt[2]), 6),
                "changed_meshes": changed_meshes,
                "changed_points": changed_points,
            }
        )
    return results


def refresh_store_extents(stage: Usd.Stage, store_root: str) -> dict[str, int]:
    parent = stage.GetPrimAtPath(store_root)
    if not parent:
        raise RuntimeError(f"missing store root: {store_root}")
    stores = 0
    meshes = 0
    for store in parent.GetChildren():
        stores += 1
        for mesh_prim in descendant_meshes(store):
            if update_mesh_extent(UsdGeom.Mesh(mesh_prim)):
                meshes += 1
    return {"refreshed_stores": stores, "refreshed_meshes": meshes}


def process_usd(
    usd_path: Path,
    store_root: str,
    target_height: float,
    backup: bool,
    refresh_extents_only: bool,
    summary_suffix: str,
) -> dict[str, object]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD stage: {usd_path}")
    before_rows = store_dimensions(stage, store_root)
    backup_path = None
    if backup:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        backup_path = usd_path.with_name(f"{usd_path.stem}.bak_store_scale_{timestamp}{usd_path.suffix}")
        shutil.copy2(usd_path, backup_path)
    if refresh_extents_only:
        operation = refresh_store_extents(stage, store_root)
        scaled = []
    else:
        scaled = scale_store_meshes(stage, store_root, target_height)
        operation = {"scaled_count": len(scaled)}
    stage.GetRootLayer().Save()
    after_rows = store_dimensions(stage, store_root)
    summary = {
        "usd": str(usd_path),
        "backup": str(backup_path) if backup_path else None,
        "store_root": store_root,
        "target_height": float(target_height),
        "before": summarize_dimensions(before_rows),
        "after": summarize_dimensions(after_rows),
        **operation,
        "scaled_samples": scaled[:20],
    }
    summary_path = usd_path.with_name(f"{usd_path.name}{summary_suffix}")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    summaries = [
        process_usd(
            usd.resolve(),
            args.store_root,
            args.target_height,
            args.backup,
            args.refresh_extents_only,
            args.summary_suffix,
        )
        for usd in args.usd
    ]
    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
