#!/usr/bin/env python3
"""Replay recorded episode camera poses on a VC USD scene in Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


SURFACE_MESH_PATHS = {
    "/World/TerrainRegions/Terrain",
    "/World/TerrainRegions/water",
    "/World/TerrainRegions/green",
    "/World/TerrainRegions/sidewalk",
    "/World/TerrainRegions/road",
    "/World/TerrainRegions/crosswalk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--height-field", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height-scale", type=float, required=True)
    parser.add_argument("--height-reference", choices=["min", "median"], default="median")
    parser.add_argument("--osm", type=Path, default=None)
    parser.add_argument("--road-depth", type=float, default=0.0)
    parser.add_argument("--road-cell-size", type=float, default=25.0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 20, 40, 60, 80, 100, 120, 140])
    parser.add_argument("--resolution", nargs=2, type=int, default=(640, 480))
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--warmup-frames", type=int, default=12)
    parser.add_argument("--focal-length", type=float, default=18.0)
    parser.add_argument(
        "--camera-convention",
        choices=["raw", "world_to_opengl", "world_to_opengl_isaaclab", "world_to_opengl_inverse"],
        default="world_to_opengl",
    )
    parser.add_argument("--hide-terrain-regions", action="store_true", default=True)
    parser.add_argument("--no-compress-camera-z", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def load_episode_states(episode_dir: Path) -> dict[str, np.ndarray]:
    state_path = episode_dir / "agent_state.npz"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    with np.load(state_path, allow_pickle=True) as data:
        return {
            "frame_index": data["frame_index"].astype(int),
            "pov_position": data["pov_position"].astype(float),
            "pov_quat_xyzw": data["pov_quat_xyzw"].astype(float),
        }


def height_stats(height_field: Path) -> tuple[float, float]:
    with np.load(height_field) as data:
        alt = data["terrain_alt"].astype(float)
    return float(np.min(alt)), float(np.median(alt))


def compressed_z(z: float, ground_z: float, scale: float) -> float:
    return ground_z + (float(z) - ground_z) * scale


def parse_float(value, default: float | None = 0.0) -> float | None:
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


def xform_cache():
    from pxr import Usd
    from pxr import UsdGeom

    return UsdGeom.XformCache(Usd.TimeCode.Default())


def add_world_alignment(stage, align_offset: float) -> None:
    from pxr import Gf
    from pxr import UsdGeom

    world = stage.GetPrimAtPath("/World")
    if not world:
        return
    xform = UsdGeom.Xformable(world)
    existing = [op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate:episodeAlign"]
    op = existing[0] if existing else xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "episodeAlign")
    op.Set(Gf.Vec3d(0.0, 0.0, -float(align_offset)))


def compress_mesh_points(
    stage,
    mesh_paths: set[str],
    ground_z: float,
    height_scale: float,
    road_segments: list[dict],
    road_depth: float,
    road_cell_size: float,
) -> dict[str, int]:
    from pxr import Gf
    from pxr import UsdGeom

    cache = xform_cache()
    grid = build_segment_grid(road_segments, road_cell_size)
    changed = 0
    road_vertices = 0
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
    return {"compressed_meshes": changed, "road_vertices": road_vertices}


def local_delta_for_parent(stage, prim, world_delta_z: float):
    from pxr import Gf

    parent = prim.GetParent()
    if not parent:
        return Gf.Vec3d(0.0, 0.0, world_delta_z)
    cache = xform_cache()
    parent_world = cache.GetLocalToWorldTransform(parent)
    return parent_world.GetInverse().TransformDir(Gf.Vec3d(0.0, 0.0, world_delta_z))


def translate_root(stage, prim, world_delta_z: float) -> bool:
    from pxr import Gf
    from pxr import UsdGeom

    if abs(world_delta_z) < 1e-8 or not prim or not prim.IsA(UsdGeom.Xformable):
        return False
    local_delta = local_delta_for_parent(stage, prim, world_delta_z)
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        op = translate_ops[0]
        value = op.Get()
        op.Set(Gf.Vec3d(value[0] + local_delta[0], value[1] + local_delta[1], value[2] + local_delta[2]))
    else:
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "flatGroundShift").Set(local_delta)
    return True


def bbox_ref_z(stage, prim) -> float | None:
    from pxr import Usd
    from pxr import UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return None
    return float(box.GetMin()[2])


def origin_ref_z(stage, prim) -> float | None:
    cache = xform_cache()
    try:
        pos = cache.GetLocalToWorldTransform(prim).Transform((0.0, 0.0, 0.0))
    except Exception:
        return None
    z = float(pos[2])
    if math.isfinite(z) and abs(z) > 1e-4:
        return z
    return None


def object_roots(stage):
    roots = []
    for parent_path in ["/World/store", "/World/RoadObjects", "/World/Objaverse"]:
        parent = stage.GetPrimAtPath(parent_path)
        if parent:
            roots.extend(list(parent.GetChildren()))
    city = stage.GetPrimAtPath("/World/city")
    if city:
        for child in city.GetChildren():
            name = child.GetName()
            if name in {"Terrain", "_materials"}:
                continue
            roots.append(child)
    return roots


def move_object_roots(stage, ground_z: float, height_scale: float) -> int:
    moved = 0
    for prim in object_roots(stage):
        ref_z = bbox_ref_z(stage, prim)
        if ref_z is None:
            ref_z = origin_ref_z(stage, prim)
        if ref_z is None:
            continue
        target_z = compressed_z(ref_z, ground_z, height_scale)
        if translate_root(stage, prim, target_z - ref_z):
            moved += 1
    return moved


def prepare_stage(
    stage,
    align_offset: float,
    ground_raw: float,
    height_scale: float,
    hide_regions: bool,
    road_segments: list[dict],
    road_depth: float,
    road_cell_size: float,
) -> dict[str, float]:
    from pxr import UsdGeom

    add_world_alignment(stage, align_offset)
    ground_world = ground_raw - align_offset
    mesh_summary = {"compressed_meshes": 0, "road_vertices": 0}
    root_count = 0
    if abs(height_scale - 1.0) > 1e-8:
        mesh_summary = compress_mesh_points(stage, SURFACE_MESH_PATHS, ground_world, height_scale, road_segments, road_depth, road_cell_size)
        root_count = move_object_roots(stage, ground_world, height_scale)
    if hide_regions:
        regions = stage.GetPrimAtPath("/World/TerrainRegions")
        if regions:
            UsdGeom.Imageable(regions).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
            for child in regions.GetChildren():
                UsdGeom.Imageable(child).CreateVisibilityAttr(
                    UsdGeom.Tokens.inherited if child.GetName() == "Terrain" else UsdGeom.Tokens.invisible
                )
    return {
        "align_offset": align_offset,
        "ground_world": ground_world,
        "height_scale": height_scale,
        "compressed_meshes": mesh_summary["compressed_meshes"],
        "road_vertices": mesh_summary["road_vertices"],
        "road_segments": len(road_segments),
        "road_depth": road_depth,
        "moved_roots": root_count,
    }


def convert_camera_quat(quat_xyzw: np.ndarray, convention: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    x, y, z, w = [float(v) for v in quat_xyzw]
    camera_rot = R.from_quat([x, y, z, w])
    if convention == "raw":
        return np.array([x, y, z, w], dtype=float)
    if convention == "world_to_opengl_isaaclab":
        return (camera_rot * R.from_euler("XYZ", [math.pi / 2.0, -math.pi / 2.0, 0.0])).as_quat()

    # IsaacLab's world camera frame is x-forward, y-left, z-up. USD/OpenGL
    # cameras use x-right, y-up, z-back.
    target_axes_in_world_camera = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    world_to_opengl = R.from_matrix(target_axes_in_world_camera)
    if convention == "world_to_opengl":
        return (camera_rot * world_to_opengl).as_quat()
    if convention == "world_to_opengl_inverse":
        return (camera_rot * world_to_opengl.inv()).as_quat()
    raise ValueError(f"unsupported camera convention: {convention}")


def camera_matrix_from_pose(position: np.ndarray, quat_xyzw: np.ndarray, convention: str):
    from pxr import Gf

    quat_opengl_xyzw = convert_camera_quat(quat_xyzw, convention)
    ox, oy, oz, ow = [float(v) for v in quat_opengl_xyzw]
    quat = Gf.Quatd(ow, Gf.Vec3d(ox, oy, oz))
    return Gf.Matrix4d().SetRotate(quat).SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))


def render_rgb(app, rgb_annotator, warmup_frames: int):
    import omni.replicator.core as rep

    app.update()
    rep.orchestrator.step(rt_subframes=max(1, warmup_frames), delta_time=0.0)
    rgb = rgb_annotator.get_data()
    if hasattr(rgb, "numpy"):
        rgb = rgb.numpy()
    if isinstance(rgb, dict):
        rgb = rgb.get("data")
    if rgb is None or rgb.size == 0:
        return None
    return rgb[:, :, :3].astype(np.uint8)


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def wait_for_stage_load() -> None:
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    for _ in range(900):
        app.update()
        if omni.usd.get_context().get_stage_loading_status()[2] == 0:
            return
    raise TimeoutError("USD stage did not finish loading")


async def attach_stage(stage_to_attach):
    import omni.usd

    return await omni.usd.get_context().attach_stage_async(stage_to_attach)


def render_episode(args: argparse.Namespace) -> None:
    import asyncio

    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdLux

    usd = args.usd.resolve()
    episode_dir = args.episode_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    align_offset, median_height = height_stats(args.height_field.resolve())
    ground_raw = align_offset if args.height_reference == "min" else median_height
    states = load_episode_states(episode_dir)
    frame_to_index = {int(frame): index for index, frame in enumerate(states["frame_index"])}

    print(f"[isaac-episode] opening {usd}", flush=True)
    stage_to_attach = Usd.Stage.Open(str(usd))
    if stage_to_attach is None:
        raise RuntimeError(f"failed to open USD stage: {usd}")
    success, _error = asyncio.get_event_loop().run_until_complete(attach_stage(stage_to_attach))
    if not success:
        raise RuntimeError(f"failed to attach USD stage: {usd}")
    wait_for_stage_load()
    stage = omni.usd.get_context().get_stage()
    road_segments = build_road_segments(args.osm.resolve() if args.osm else None)
    prep_summary = prepare_stage(
        stage,
        align_offset,
        ground_raw,
        args.height_scale,
        args.hide_terrain_regions,
        road_segments,
        args.road_depth,
        args.road_cell_size,
    )
    wait_for_stage_load()

    rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", 1.15)
    dome = UsdLux.DomeLight.Define(stage, "/EpisodeReplayDome")
    dome.CreateIntensityAttr(850.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/EpisodeReplaySun")
    sun.CreateIntensityAttr(2600.0)
    sun.CreateAngleAttr(0.45)

    camera_path = "/EpisodeReplayCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(float(args.focal_length))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 5000.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    transform = camera_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

    render_product = rep.create.render_product(camera_path, tuple(args.resolution), name="EpisodeReplayRender")
    rgb_annotator = rep.annotators.get("rgb")
    rgb_annotator.attach([render_product])
    app = omni.kit.app.get_app()

    renders = []
    for frame in args.frames:
        if frame not in frame_to_index:
            print(f"[isaac-episode] skip missing frame {frame}", flush=True)
            continue
        idx = frame_to_index[frame]
        position = np.array(states["pov_position"][idx], dtype=float)
        if not args.no_compress_camera_z and abs(args.height_scale - 1.0) > 1e-8:
            position[2] = compressed_z(position[2], prep_summary["ground_world"], args.height_scale)
        quat = states["pov_quat_xyzw"][idx]
        transform.Set(camera_matrix_from_pose(position, quat, args.camera_convention))
        rgb = render_rgb(app, rgb_annotator, args.warmup_frames)
        if rgb is None:
            raise RuntimeError(f"empty render for frame {frame}")
        path = output_dir / f"{frame:04d}.png"
        write_png(path, rgb)
        renders.append({"frame": int(frame), "path": str(path), "camera_position": position.tolist(), "camera_quat_xyzw": quat.tolist()})
        print(f"[isaac-episode] wrote {path}", flush=True)

    summary = {
        "usd": str(usd),
        "episode_dir": str(episode_dir),
        "height_field": str(args.height_field.resolve()),
        "height_reference": args.height_reference,
        "osm": str(args.osm.resolve()) if args.osm else None,
        "stage": prep_summary,
        "frames": renders,
        "focal_length": float(args.focal_length),
        "camera_convention": args.camera_convention,
    }
    (output_dir / "episode_pose_render_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    print("[isaac-episode] launching Isaac app", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
            "width": int(args.resolution[0]),
            "height": int(args.resolution[1]),
            "multi_gpu": True,
            "samples_per_pixel_per_frame": 1,
            "anti_aliasing": 0,
        }
    )
    try:
        render_episode(args)
    except BaseException:
        print("[isaac-episode] failed:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        print("[isaac-episode] closing app", flush=True)
        app.close()


if __name__ == "__main__":
    main()
