#!/usr/bin/env python3
"""Replace storefront buildings with hollow shells containing facade openings."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdShade

from compose_vc_full_innout import (
    first_bound_material_path,
    first_root_prim,
    match_stage_path,
    write_cut_usd_from_glb,
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def resolve_building_path(work_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (work_dir / path).resolve()
    return path if path.exists() else None


def make_opening(item: dict[str, Any]) -> dict[str, Any] | None:
    outward = item.get("outward_normal_xy")
    if not outward:
        return None
    ox, oy = float(outward[0]), float(outward[1])
    tangent = [-oy, ox]
    center = item.get("opening_center") or item.get("final_location")
    if not center:
        return None
    bottom_z = float(item.get("opening_bottom_z", center[2] if len(center) > 2 else 0.0))
    return {
        "center": [float(center[0]), float(center[1])],
        "bottom_z": bottom_z,
        "tangent": tangent,
        "outward_normal": [ox, oy],
        "width": float(item.get("opening_width", 3.2)),
        "height": float(item.get("opening_height", 3.0)),
        "inside_depth": float(item.get("opening_inside_depth", 1.35)),
        "outside_depth": float(item.get("opening_outside_depth", 0.25)),
        "source_index": item.get("index"),
        "amenity": item.get("amenity"),
    }


def run_cut(blender: str, cut_helper: Path, building: Path, openings: list[dict[str, Any]], output_glb: Path) -> None:
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    openings_json = output_glb.with_suffix(".openings.json")
    write_json(openings_json, openings)
    cmd = [
        shutil.which(blender) or blender,
        "--background",
        "--python",
        str(cut_helper),
        "--",
        "--input-glb",
        str(building),
        "--output-glb",
        str(output_glb),
        "--openings-json",
        str(openings_json),
        "--cut-mode",
        "boolean_tunnel",
    ]
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CONDA_PREFIX"):
        env.pop(key, None)
    subprocess.check_call(cmd, env=env)


def apply_replacements(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    plan = read_json(args.storefront_plan)
    by_building: dict[Path, list[dict[str, Any]]] = {}
    skipped = []
    for item in plan.get("placements", []):
        if item.get("facade_status") != "facade_aligned":
            skipped.append({"reason": "not_facade_aligned", "item": item.get("index")})
            continue
        building = resolve_building_path(work_dir, item.get("building_path"))
        opening = make_opening(item)
        if not building or not opening:
            skipped.append({"reason": "missing_building_or_opening", "item": item.get("index")})
            continue
        by_building.setdefault(building, []).append(opening)

    layer = Sdf.Layer.FindOrOpen(str(args.usd))
    if layer is None:
        raise RuntimeError(f"could not open USD layer: {args.usd}")
    stage = Usd.Stage.Open(layer)
    records = []
    cut_dir = work_dir / args.output_subdir

    for building, openings in sorted(by_building.items(), key=lambda kv: str(kv[0])):
        stem = building.stem.replace(" ", "_")
        cut_glb = cut_dir / f"{args.city}_{stem}_storefront_cut.glb"
        cut_usd = cut_dir / f"{args.city}_{stem}_storefront_cut.usd"
        try:
            target_path, target_dist = match_stage_path(args.usd, building, args.max_bbox_match_dist)
        except RuntimeError as exc:
            skipped.append(
                {
                    "reason": "building_stage_match_failed",
                    "building": str(building),
                    "opening_count": len(openings),
                    "error": str(exc),
                }
            )
            continue
        run_cut(args.blender, args.cut_helper, building, openings, cut_glb)
        if cut_usd.exists():
            cut_usd.unlink()
        write_cut_usd_from_glb(cut_glb, cut_usd)

        original = stage.OverridePrim(target_path)
        material_path = first_bound_material_path(original)
        original.SetActive(False)
        replacement_path = Sdf.Path(target_path + "_storefront_cut")
        if stage.GetPrimAtPath(replacement_path):
            stage.RemovePrim(replacement_path)
        replacement = stage.DefinePrim(replacement_path, "Xform")
        replacement.GetReferences().AddReference(os.path.relpath(cut_usd, args.usd.parent), first_root_prim(cut_usd))
        if material_path is not None:
            material = UsdShade.Material(stage.GetPrimAtPath(material_path))
            mesh = stage.OverridePrim(replacement_path.AppendChild("mesh"))
            UsdShade.MaterialBindingAPI(mesh).Bind(material)
        records.append(
            {
                "building": str(building),
                "target_path": target_path,
                "target_match_distance": target_dist,
                "opening_count": len(openings),
                "cut_glb": str(cut_glb),
                "cut_usd": str(cut_usd),
            }
        )

    layer.Save()
    return {"city": args.city, "usd": str(args.usd), "cut_building_count": len(records), "records": records, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply storefront opening shell replacements to an outdoor USD.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--storefront-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--cut-helper", type=Path, required=True)
    parser.add_argument("--output-subdir", default="generated_storefront_openings")
    parser.add_argument("--max-bbox-match-dist", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = apply_replacements(args)
    write_json(args.manifest, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
