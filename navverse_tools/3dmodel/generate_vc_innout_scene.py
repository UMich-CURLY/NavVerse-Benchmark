#!/usr/bin/env python
import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree

try:
    from pxr import Gf, Sdf, Usd, UsdGeom
except ImportError as exc:
    raise SystemExit(
        "pxr is required. Run this through the navverse env with the Isaac USD libs on PYTHONPATH."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INTERIORS = {
    "BELGRADE": "scene11",
    "MADRID": "scene27",
}


@dataclass
class DoorPlan:
    city: str
    building_name: str
    building_glb: str
    target_prim_path: str
    interior_scene: str
    storefront_asset: str
    door_center_xy: list[float]
    door_bottom_z: float
    door_width: float
    door_height: float
    door_depth: float
    terrain_z: float
    building_min_z: float
    tangent_xy: list[float]
    outward_normal_xy: list[float]
    doorway_yaw_deg: float
    room_translate: list[float]
    room_yaw_deg: float
    storefront_translate: list[float]
    storefront_yaw_deg: float
    cut_glb: str
    cut_usd: str
    output_usd: str


def normalize(vec):
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm <= 1e-9:
        raise ValueError(f"zero-length vector: {vec}")
    return arr / norm


def yaw_for_local_minus_y(outward_normal):
    normal = normalize(outward_normal)
    return math.degrees(math.atan2(normal[0], -normal[1]))


def mapped_vertices_from_glb(glb_path):
    scene = trimesh.load(glb_path, force="scene")
    if hasattr(scene, "geometry"):
        mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh = scene
    vertices = np.asarray(mesh.vertices)
    mapped = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
    return mapped


def load_trimesh_mesh(glb_path):
    mesh = trimesh.load(glb_path, force="mesh")
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return mesh

    scene = trimesh.load(glb_path, force="scene")
    if hasattr(scene, "dump"):
        return scene.dump(concatenate=True)
    if hasattr(scene, "geometry"):
        return trimesh.util.concatenate(list(scene.geometry.values()))
    raise RuntimeError(f"could not load mesh from {glb_path}")


def write_cut_usd_from_glb(glb_path, usd_path):
    mesh = load_trimesh_mesh(glb_path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"cut mesh is empty: {glb_path}")

    usd_vertices = vertices
    usd_path = Path(usd_path)
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    if usd_path.exists():
        usd_path.unlink()

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root = stage.DefinePrim("/cut_building", "Xform")
    stage.SetDefaultPrim(root)
    usd_mesh = UsdGeom.Mesh.Define(stage, "/cut_building/mesh")
    usd_mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in usd_vertices])
    usd_mesh.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
    usd_mesh.CreateFaceVertexIndicesAttr([int(index) for face in faces for index in face])
    usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
    stage.GetRootLayer().Save()


def mapped_bbox_from_glb(glb_path):
    vertices = mapped_vertices_from_glb(glb_path)
    return vertices.min(axis=0), vertices.max(axis=0)


def building_footprint(mapped_vertices):
    zmin = float(mapped_vertices[:, 2].min())
    zmax = float(mapped_vertices[:, 2].max())
    low = mapped_vertices[:, 2] <= zmin + max(0.25, 0.05 * (zmax - zmin))
    points = mapped_vertices[low][:, :2]
    if len(points) < 4:
        points = mapped_vertices[:, :2]
    hull = ConvexHull(points)
    footprint = points[hull.vertices]
    centroid = footprint.mean(axis=0)
    return footprint, centroid, zmin


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def load_public_points(city_dir):
    weighted_points = []
    for filename, weight in (("sidewalk_points.json", 1.0), ("lane_points.json", 0.35)):
        path = city_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for point in data.get("points", []):
            weighted_points.append(
                (
                    float(point["utm_x"]),
                    float(point["utm_y"]),
                    float(point.get("height", 0.0)),
                    weight,
                )
            )
    if not weighted_points:
        raise RuntimeError(f"no sidewalk/lane points found in {city_dir}")
    return np.asarray(weighted_points, dtype=float)


def best_wall_for_footprint(footprint, centroid, public_points, min_wall_length=6.0, door_width=4.5):
    best = None
    xy = public_points[:, :2]
    weights = public_points[:, 3]
    edge_margin = 1.5

    for index in range(len(footprint)):
        a = footprint[index]
        b = footprint[(index + 1) % len(footprint)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < max(min_wall_length, door_width + 1.0):
            continue

        tangent = edge / length
        normals = [np.array([-tangent[1], tangent[0]]), np.array([tangent[1], -tangent[0]])]
        for normal in normals:
            side = np.sign(np.dot((a + b) * 0.5 + normal - centroid, normal))
            if side < 0:
                normal = -normal

            rel = xy - a
            proj = rel @ tangent
            lateral = rel @ normal
            mask = (proj >= -edge_margin) & (proj <= length + edge_margin) & (lateral >= 0.8) & (lateral <= 14.0)
            if not np.any(mask):
                continue

            masked_weights = weights[mask]
            weighted_count = float(masked_weights.sum())
            mean_dist = float(np.average(lateral[mask], weights=masked_weights))
            raw_center = float(np.average(proj[mask], weights=masked_weights))
            center_s = float(np.clip(raw_center, door_width * 0.5 + 0.8, length - door_width * 0.5 - 0.8))
            center = a + tangent * center_s
            score = weighted_count / (mean_dist + 0.5) + 0.02 * length

            candidate = {
                "score": score,
                "edge_index": index,
                "length": length,
                "weighted_count": weighted_count,
                "mean_public_distance": mean_dist,
                "center": center,
                "tangent": tangent,
                "outward_normal": normalize(normal),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        raise RuntimeError("no suitable road-facing wall found")
    return best


def terrain_height(data_root, city, xy, fallback_z):
    npz_path = data_root / "vc_full_raw" / city / "height_field.npz"
    if not npz_path.exists():
        return fallback_z
    data = np.load(npz_path)
    coords = np.asarray(data["plane_coord"], dtype=float)
    heights = np.asarray(data["terrain_alt"], dtype=float)
    _, idx = cKDTree(coords).query(np.asarray(xy, dtype=float), k=1)
    return float(heights[int(idx)])


def target_prim_bbox(stage, cut_name):
    root = f"/World/city/root/{cut_name}"
    prim = stage.GetPrimAtPath(root)
    if not prim:
        return None, None, None

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for child in Usd.PrimRange(prim):
        if child.GetTypeName() == "Mesh":
            box = bbox_cache.ComputeWorldBound(child).ComputeAlignedBox()
            return root, np.asarray(box.GetMin()), np.asarray(box.GetMax())
    box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return root, np.asarray(box.GetMin()), np.asarray(box.GetMax())


def resolve_building(data_root, city, requested):
    city_dir = data_root / "vc_apr14" / city
    raw_dir = data_root / "vc_full_raw" / city / "buildings"
    cut_name = requested
    if cut_name == "cut_buildings.json":
        cut_data = json.loads((city_dir / "cut_buildings.json").read_text())
        cut_name = cut_data["cut_building_names"][0]

    if cut_name.startswith("element_"):
        suffix = cut_name.split("_", 1)[1]
        glb = raw_dir / f"buildings_element.{suffix}.glb"
        if glb.exists():
            return cut_name, glb, f"/World/city/root/{cut_name}"

    direct = raw_dir / f"buildings_{cut_name}.glb"
    if direct.exists():
        return cut_name, direct, f"/World/city/root/{cut_name}"

    stage = Usd.Stage.Open(str(city_dir / f"{city}_buildings.usd"))
    target_path, target_min, target_max = target_prim_bbox(stage, cut_name)
    if target_min is None:
        raise RuntimeError(f"could not find {cut_name} in {city}_buildings.usd")

    best = None
    for glb in raw_dir.glob("*.glb"):
        try:
            bbox_min, bbox_max = mapped_bbox_from_glb(glb)
        except Exception:
            continue
        dist = float(np.linalg.norm(bbox_min - target_min) + np.linalg.norm(bbox_max - target_max))
        if best is None or dist < best[0]:
            best = (dist, glb)
    if best is None:
        raise RuntimeError(f"no raw GLB candidate matched {cut_name}")
    return cut_name, best[1], target_path


def collect_stage_mesh_bboxes(stage):
    mesh_bboxes = []
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in stage.TraverseAll():
        if prim.GetTypeName() != "Mesh" or not prim.GetPath().pathString.startswith("/World/city/root/"):
            continue
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        parent = prim.GetParent().GetPath().pathString
        mesh_bboxes.append((parent, np.asarray(box.GetMin()), np.asarray(box.GetMax())))
    return mesh_bboxes


def match_stage_path(mesh_bboxes, bbox_min, bbox_max, max_dist=0.1):
    matched_path = None
    matched_dist = float("inf")
    for path, mn, mx in mesh_bboxes:
        dist = float(np.linalg.norm(mn - bbox_min) + np.linalg.norm(mx - bbox_max))
        if dist < matched_dist:
            matched_dist = dist
            matched_path = path
    if matched_path is None or matched_dist > max_dist:
        return None
    return matched_path


def auto_building_candidate(glb, mesh_bboxes, public_points, door_width):
    vertices = mapped_vertices_from_glb(glb)
    footprint, centroid, _ = building_footprint(vertices)
    area = polygon_area(footprint)
    if area < 1200.0:
        return None

    wall = best_wall_for_footprint(footprint, centroid, public_points, door_width=door_width)
    matched_path = match_stage_path(mesh_bboxes, vertices.min(axis=0), vertices.max(axis=0))
    if matched_path is None:
        return None

    score = wall["score"] * min(area / 1800.0, 4.0)
    return score, glb.stem.replace("buildings_", ""), glb, matched_path, area, wall


def choose_auto_building(data_root, city, public_points, door_width):
    raw_dir = data_root / "vc_full_raw" / city / "buildings"
    stage = Usd.Stage.Open(str(data_root / "vc_apr14" / city / f"{city}_buildings.usd"))
    mesh_bboxes = collect_stage_mesh_bboxes(stage)
    best = None

    for glb in raw_dir.glob("*.glb"):
        try:
            candidate = auto_building_candidate(glb, mesh_bboxes, public_points, door_width)
        except Exception:
            continue
        if candidate is not None and (best is None or candidate[0] > best[0]):
            best = candidate

    if best is None:
        raise RuntimeError(f"could not auto-select a building for {city}")
    return best


def run_blender_cut(plan, door_width, door_height, door_depth):
    blender = shutil.which("blender")
    if blender is None:
        raise RuntimeError("blender is not installed or not on PATH")

    cmd = [
        blender,
        "--background",
        "--python",
        str(SCRIPT_DIR / "cut_vc_building_blender.py"),
        "--",
        "--input-glb",
        plan.building_glb,
        "--output-glb",
        plan.cut_glb,
        f"--center={plan.door_center_xy[0]},{plan.door_center_xy[1]}",
        f"--bottom-z={plan.door_bottom_z}",
        f"--tangent={plan.tangent_xy[0]},{plan.tangent_xy[1]}",
        f"--outward-normal={plan.outward_normal_xy[0]},{plan.outward_normal_xy[1]}",
        f"--door-width={door_width}",
        f"--door-height={door_height}",
        f"--door-depth={door_depth}",
    ]
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CONDA_PREFIX"):
        env.pop(key, None)
    subprocess.check_call(cmd, env=env)
    if not Path(plan.cut_glb).exists():
        raise RuntimeError(f"Blender finished without producing {plan.cut_glb}")
    write_cut_usd_from_glb(plan.cut_glb, plan.cut_usd)
    if not Path(plan.cut_usd).exists():
        raise RuntimeError(f"USD conversion did not produce {plan.cut_usd}")


def first_root_prim(layer_path):
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    roots = list(layer.rootPrims)
    if not roots:
        raise RuntimeError(f"no root prims in {layer_path}")
    if layer.defaultPrim:
        return Sdf.Path(f"/{layer.defaultPrim}")
    return Sdf.Path(f"/{roots[0].name}")


def compose_scene(data_root, city, plan):
    city_dir = data_root / "vc_apr14" / city
    base_usd = city_dir / f"{city}_innnot.usd"
    output_usd = Path(plan.output_usd)
    if output_usd.exists():
        output_usd.unlink()

    root_layer = Sdf.Layer.FindOrOpen(str(base_usd))
    out_layer = Sdf.Layer.CreateNew(str(output_usd))
    out_layer.TransferContent(root_layer)
    stage = Usd.Stage.Open(out_layer)

    original = stage.OverridePrim(plan.target_prim_path)
    original.SetActive(False)

    replacement_path = Sdf.Path(plan.target_prim_path + "_cut")
    replacement = stage.DefinePrim(replacement_path, "Xform")
    replacement.GetReferences().AddReference(
        os.path.relpath(plan.cut_usd, city_dir),
        first_root_prim(plan.cut_usd),
    )

    store_path = "/World/store/innout_storefront"
    if stage.GetPrimAtPath(store_path):
        stage.RemovePrim(store_path)
    store = stage.DefinePrim(store_path, "Xform")
    store_xform = UsdGeom.XformCommonAPI(store)
    store_xform.SetTranslate(Gf.Vec3d(*plan.storefront_translate))
    store_xform.SetRotate(Gf.Vec3f(0.0, 0.0, plan.storefront_yaw_deg))
    store_xform.SetScale(Gf.Vec3f(100.0, 100.0, 100.0))
    store.GetReferences().AddReference(
        f"../../storefront/{plan.storefront_asset}/{plan.storefront_asset}.usd"
    )

    room_path = "/World/start_result_navigation"
    if stage.GetPrimAtPath(room_path):
        stage.RemovePrim(room_path)
    room = stage.DefinePrim(room_path, "Xform")
    room_xform = UsdGeom.Xformable(room)
    room_xform.AddTranslateOp().Set(Gf.Vec3d(*plan.room_translate))
    yaw = math.radians(plan.room_yaw_deg)
    room_xform.AddOrientOp().Set(
        Gf.Quatf(math.cos(yaw * 0.5), Gf.Vec3f(0.0, 0.0, math.sin(yaw * 0.5)))
    )
    room_xform.AddScaleOp().Set(Gf.Vec3d(0.01, 0.01, 0.01))
    room.GetReferences().AddReference(
        f"../../grscenes_commercial/scenes/{plan.interior_scene}/start_result_navigation.usd"
    )

    out_layer.Save()


def build_plan(args):
    data_root = Path(args.data_root).resolve()
    city = args.city.upper()
    city_dir = data_root / "vc_apr14" / city
    public_points = load_public_points(city_dir)

    if args.building == "auto":
        _, building_name, glb, target_prim_path, _, wall = choose_auto_building(
            data_root, city, public_points, args.door_width
        )
        mapped = mapped_vertices_from_glb(glb)
        _, _, building_min_z = building_footprint(mapped)
    else:
        building_name, glb, target_prim_path = resolve_building(data_root, city, args.building)
        mapped = mapped_vertices_from_glb(glb)
        footprint, centroid, building_min_z = building_footprint(mapped)
        wall = best_wall_for_footprint(footprint, centroid, public_points, door_width=args.door_width)

    center = np.asarray(wall["center"], dtype=float)
    outward = normalize(wall["outward_normal"])
    tangent = normalize(wall["tangent"])
    terrain_z = terrain_height(data_root, city, center, building_min_z)
    bottom_z = max(terrain_z, building_min_z) - 0.08
    yaw = yaw_for_local_minus_y(outward)

    room_xy = center - outward * args.room_offset_m
    store_xy = center + outward * args.storefront_offset_m
    room_translate = [float(room_xy[0]), float(room_xy[1]), float(bottom_z)]
    storefront_translate = [
        float(100.0 * store_xy[0]),
        float(100.0 * store_xy[1]),
        float(100.0 * bottom_z),
    ]

    interior_scene = args.interior_scene or DEFAULT_INTERIORS.get(city, "scene11")
    output_dir = city_dir / args.output_dir
    cut_glb = output_dir / f"{city}_{Path(glb).stem}_cut.glb"
    cut_usd = output_dir / f"{city}_{Path(glb).stem}_cut.usd"
    output_usd = city_dir / f"{city}_innout_generated.usd"

    return DoorPlan(
        city=city,
        building_name=building_name,
        building_glb=str(glb),
        target_prim_path=target_prim_path,
        interior_scene=interior_scene,
        storefront_asset=args.storefront_asset,
        door_center_xy=[float(center[0]), float(center[1])],
        door_bottom_z=float(bottom_z),
        door_width=float(args.door_width),
        door_height=float(args.door_height),
        door_depth=float(args.door_depth),
        terrain_z=float(terrain_z),
        building_min_z=float(building_min_z),
        tangent_xy=[float(tangent[0]), float(tangent[1])],
        outward_normal_xy=[float(outward[0]), float(outward[1])],
        doorway_yaw_deg=float(yaw),
        room_translate=room_translate,
        room_yaw_deg=float(yaw),
        storefront_translate=storefront_translate,
        storefront_yaw_deg=float(yaw),
        cut_glb=str(cut_glb),
        cut_usd=str(cut_usd),
        output_usd=str(output_usd),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a staged VC innout scene with a cut storefront.")
    parser.add_argument("--data-root", default="navverse_data")
    parser.add_argument("--city", required=True)
    parser.add_argument("--building", default="cut_buildings.json", help="cut_buildings.json, auto, or a building prim/raw name")
    parser.add_argument("--interior-scene")
    parser.add_argument("--storefront-asset", default="restaurant_6")
    parser.add_argument("--output-dir", default="generated_innout")
    parser.add_argument("--door-width", type=float, default=4.5)
    parser.add_argument("--door-height", type=float, default=3.2)
    parser.add_argument("--door-depth", type=float, default=16.0)
    parser.add_argument("--room-offset-m", type=float, default=5.0)
    parser.add_argument("--storefront-offset-m", type=float, default=0.15)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    plan = build_plan(args)
    plan_path = Path(plan.output_usd).with_suffix(".plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(asdict(plan), indent=2))
    print(json.dumps(asdict(plan), indent=2))
    if args.plan_only:
        return
    run_blender_cut(plan, args.door_width, args.door_height, args.door_depth)
    compose_scene(Path(args.data_root).resolve(), plan.city, plan)
    print(f"Wrote staged innout scene: {plan.output_usd}")
    print(f"Wrote plan: {plan_path}")


if __name__ == "__main__":
    main()
