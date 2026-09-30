"""Repair generated storefront USD meshes without replacing the source assets.

Run with a Python environment that provides Blender's ``bpy`` module, for example:

    python repair_usd_meshes_blender.py --root navverse_data/storefront

The repaired file is written next to each input as ``<name>.repaired.usd``.  The
source USD and its textures are left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args() -> argparse.Namespace:
    argv = (
        sys.argv[sys.argv.index("--") + 1 :]
        if "--" in sys.argv
        else sys.argv[1:]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suffix", default=".repaired")
    parser.add_argument("--planarity-threshold", type=float, default=1e-3)
    parser.add_argument("--depth-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--depth-limit", type=float, default=0.025)
    parser.add_argument("--mad-multiplier", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def planarize_nonplanar_faces(
    mesh: bpy.types.Mesh, threshold: float, depth_axis: int
) -> tuple[np.ndarray, int]:
    coordinates = np.array([vertex.co[:] for vertex in mesh.vertices], dtype=float)
    projected_sum = np.zeros(len(coordinates), dtype=float)
    projected_weight = np.zeros(len(coordinates), dtype=float)
    planarized_faces = 0

    for face in mesh.polygons:
        if len(face.vertices) < 4:
            continue
        indices = np.asarray(face.vertices[:], dtype=int)
        points = coordinates[indices]
        center = points.mean(axis=0)
        _, singular_values, basis = np.linalg.svd(
            points - center, full_matrices=False
        )
        nonplanarity = singular_values[-1] / max(singular_values[0], 1e-12)
        if nonplanarity < threshold:
            continue

        normal = basis[-1]
        if abs(normal[depth_axis]) < 0.5:
            continue
        projected_depth = points[:, depth_axis] - (
            (points - center) @ normal
        ) / normal[depth_axis]
        weight = max(float(face.area), 1e-8)
        projected_sum[indices] += projected_depth * weight
        projected_weight[indices] += weight
        planarized_faces += 1

    changed = projected_weight > 0
    coordinates[changed, depth_axis] = (
        projected_sum[changed] / projected_weight[changed]
    )
    return coordinates, planarized_faces


def clamp_depth_spikes(
    mesh: bpy.types.Mesh,
    coordinates: np.ndarray,
    axis: int,
    minimum_limit: float,
    mad_multiplier: float,
) -> tuple[int, float]:
    adjacency: list[set[int]] = [set() for _ in coordinates]
    for edge in mesh.edges:
        first, second = edge.vertices
        adjacency[first].add(second)
        adjacency[second].add(first)

    next_depth = coordinates[:, axis].copy()
    movements: list[float] = []
    for vertex_index, neighbors in enumerate(adjacency):
        if len(neighbors) < 3:
            continue
        neighbor_depths = coordinates[list(neighbors), axis]
        median = float(np.median(neighbor_depths))
        mad = float(np.median(np.abs(neighbor_depths - median)))
        limit = max(minimum_limit, mad_multiplier * mad)
        offset = coordinates[vertex_index, axis] - median
        if abs(offset) <= limit:
            continue
        repaired = median + float(np.clip(offset, -limit, limit))
        movements.append(abs(repaired - coordinates[vertex_index, axis]))
        next_depth[vertex_index] = repaired

    coordinates[:, axis] = next_depth
    return len(movements), max(movements, default=0.0)


def repair_mesh(mesh: bpy.types.Mesh, args: argparse.Namespace) -> dict[str, int | float]:
    input_vertices = len(mesh.vertices)
    input_faces = len(mesh.polygons)
    input_ngons = sum(len(face.vertices) > 4 for face in mesh.polygons)

    depth_axis = {"x": 0, "y": 1, "z": 2}[args.depth_axis]
    coordinates, planarized_faces = planarize_nonplanar_faces(
        mesh, args.planarity_threshold, depth_axis
    )
    clamped_vertices, maximum_movement = clamp_depth_spikes(
        mesh,
        coordinates,
        depth_axis,
        args.depth_limit,
        args.mad_multiplier,
    )
    for vertex, coordinate in zip(mesh.vertices, coordinates):
        vertex.co = coordinate

    return {
        "input_vertices": input_vertices,
        "input_faces": input_faces,
        "input_ngons": input_ngons,
        "planarized_faces": planarized_faces,
        "clamped_depth_vertices": clamped_vertices,
        "maximum_depth_movement": maximum_movement,
    }


def triangulate_object(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new(name="Storefront repair triangulation", type="TRIANGULATE")
    modifier.quad_method = "BEAUTY"
    modifier.ngon_method = "BEAUTY"
    modifier.keep_custom_normals = False
    result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    if result != {"FINISHED"}:
        raise RuntimeError(f"Could not triangulate {obj.name}")
    obj.data.validate(clean_customdata=True)
    obj.data.update()
    wrapper_name, separator, numeric_suffix = obj.name.rpartition("_")
    if separator and numeric_suffix.isdigit() and obj.data.name == obj.name:
        obj.name = wrapper_name


def output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def repair_file(input_path: Path, args: argparse.Namespace) -> dict[str, object]:
    destination = output_path(input_path, args.suffix)
    if destination.exists() and not args.overwrite:
        return {
            "input": str(input_path),
            "output": str(destination),
            "status": "skipped_existing",
        }

    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.wm.usd_import(filepath=str(input_path), import_materials=True)
    if result != {"FINISHED"}:
        raise RuntimeError(f"USD import failed: {input_path}")

    mesh_reports = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            mesh_report = {"object": obj.name, **repair_mesh(obj.data, args)}
            triangulate_object(obj)
            mesh_report.update(
                output_vertices=len(obj.data.vertices),
                output_triangles=len(obj.data.polygons),
            )
            mesh_reports.append(mesh_report)
    if not mesh_reports:
        raise RuntimeError(f"No mesh objects found: {input_path}")

    result = bpy.ops.wm.usd_export(
        filepath=str(destination),
        check_existing=False,
        export_animation=False,
        export_hair=False,
        export_uvmaps=True,
        export_normals=True,
        export_materials=True,
        export_textures=False,
        relative_paths=True,
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"USD export failed: {destination}")
    if not destination.exists() or destination.stat().st_size < 1024:
        raise RuntimeError(f"USD exporter did not produce a valid file: {destination}")
    return {
        "input": str(input_path),
        "output": str(destination),
        "status": "repaired",
        "meshes": mesh_reports,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    inputs = sorted(
        path
        for path in root.glob("*/*.usd")
        if not path.stem.endswith(args.suffix)
    )
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit(f"No source USD files found below {root}")

    reports = []
    for index, input_path in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] {input_path.name}", flush=True)
        try:
            reports.append(repair_file(input_path, args))
        except Exception as error:
            reports.append(
                {"input": str(input_path), "status": "failed", "error": str(error)}
            )
            print(f"ERROR: {input_path}: {error}", file=sys.stderr, flush=True)

    report_path = args.report or root / "storefront_mesh_repair_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    repaired = sum(report["status"] == "repaired" for report in reports)
    failed = sum(report["status"] == "failed" for report in reports)
    print(f"Repaired {repaired}/{len(reports)}; failed {failed}; report: {report_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
