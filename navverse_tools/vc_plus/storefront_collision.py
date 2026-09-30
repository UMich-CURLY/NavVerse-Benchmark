#!/usr/bin/env python3
"""Apply physics collision APIs to storefront meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pxr import Usd, UsdPhysics


def apply_collision_apis_under(stage: Usd.Stage, root_path: str = "/World/store") -> dict[str, Any]:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {
            "root_path": root_path,
            "root_exists": False,
            "mesh_count": 0,
            "collision_count": 0,
            "authored_count": 0,
        }

    mesh_count = 0
    collision_count = 0
    authored_count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh_count += 1
        if not UsdPhysics.CollisionAPI(prim):
            UsdPhysics.CollisionAPI.Apply(prim)
            authored_count += 1
        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_api.CreateCollisionEnabledAttr(True)

        if not UsdPhysics.MeshCollisionAPI(prim):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        mesh_api.CreateApproximationAttr().Set("none")
        collision_count += 1

    return {
        "root_path": root_path,
        "root_exists": True,
        "mesh_count": mesh_count,
        "collision_count": collision_count,
        "authored_count": authored_count,
    }


def repair_usd(path: Path, root_path: str = "/World/store") -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"could not open USD: {path}")
    summary = apply_collision_apis_under(stage, root_path)
    stage.GetRootLayer().Save()
    return {"usd": str(path), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add storefront collision APIs to one or more USD files.")
    parser.add_argument("usd", nargs="+", type=Path)
    parser.add_argument("--root-path", default="/World/store")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    summaries = [repair_usd(path, args.root_path) for path in args.usd]
    text = json.dumps(summaries, indent=2, sort_keys=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
