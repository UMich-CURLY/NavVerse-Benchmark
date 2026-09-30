#!/usr/bin/env python3
"""Render street-level flat-ground terrain QA views in Isaac Sim."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROAD_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "living_street",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--converted-usd", type=Path, default=None)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--views-json", type=Path, default=None, help="Reuse camera views from a previous render summary.")
    parser.add_argument("--write-views-json", type=Path, default=None, help="Write selected camera views for later reuse.")
    parser.add_argument("--include-buildings", action="store_true")
    parser.add_argument("--max-buildings", type=int, default=None)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--usd-scale", type=float, default=100.0, help="Isaac GLB converter scale from VC meters to USD units.")
    parser.add_argument("--camera-height", type=float, default=1.55, help="Camera height above the road segment midpoint, in meters.")
    parser.add_argument("--target-height", type=float, default=1.25, help="Look-at target height above the road midpoint, in meters.")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def parse_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def read_osm_segments(city_dir: Path) -> list[dict[str, Any]]:
    osm_path = city_dir / "road_data" / "road_data_with_height.osm"
    if not osm_path.exists():
        osm_path = city_dir / "road_data" / "road_data.osm"
    root = ET.parse(osm_path).getroot()
    nodes = {}
    ways = []
    for element in root:
        if element.tag == "node":
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            nodes[element.get("id")] = {
                "x": parse_float(tags.get("utm_x"), None),
                "y": parse_float(tags.get("utm_y"), None),
                "z": parse_float(tags.get("height"), 0.0),
            }
        elif element.tag == "way":
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            if tags.get("highway") not in ROAD_HIGHWAYS:
                continue
            ways.append(
                {
                    "id": element.get("id"),
                    "nodes": [nd.get("ref") for nd in element.findall("nd")],
                    "tags": tags,
                }
            )
    segments = []
    for way in ways:
        refs = way["nodes"]
        for start_ref, end_ref in zip(refs, refs[1:]):
            start = nodes.get(start_ref)
            end = nodes.get(end_ref)
            if not start or not end:
                continue
            if start["x"] is None or start["y"] is None or end["x"] is None or end["y"] is None:
                continue
            dx = float(end["x"]) - float(start["x"])
            dy = float(end["y"]) - float(start["y"])
            length = math.hypot(dx, dy)
            if length < 25.0:
                continue
            segments.append(
                {
                    "way_id": way["id"],
                    "highway": way["tags"].get("highway"),
                    "start": start,
                    "end": end,
                    "length": length,
                    "heading": math.atan2(dy, dx),
                    "mid": {
                        "x": (float(start["x"]) + float(end["x"])) * 0.5,
                        "y": (float(start["y"]) + float(end["y"])) * 0.5,
                        "z": (float(start["z"]) + float(end["z"])) * 0.5,
                    },
                }
            )
    return sorted(segments, key=lambda item: (abs(item["mid"]["x"]) + abs(item["mid"]["y"]), -item["length"]))


async def convert_asset(input_path: Path, output_path: Path) -> None:
    import omni.kit.asset_converter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
        return
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.ignore_cameras = True
    context.ignore_lights = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(input_path),
        str(output_path),
        None,
        context,
    )
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(f"asset conversion failed: {input_path} -> {output_path}")


def safe_ref_name(index: int, path: Path) -> str:
    clean = "".join(char if char.isalnum() else "_" for char in path.stem)
    clean = clean.strip("_") or "asset"
    return f"b_{index:04d}_{clean[:64]}"


def add_reference(stage, prim_path: str, asset_path: Path) -> None:
    from pxr import UsdGeom

    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    prim.GetReferences().AddReference(str(asset_path))


async def convert_and_add_buildings(stage, city_dir: Path, converted_root: Path, max_buildings: int | None) -> list[str]:
    from pxr import UsdGeom

    building_paths = sorted((city_dir / "buildings").glob("*.glb"))
    if max_buildings is not None:
        building_paths = building_paths[:max_buildings]
    if not building_paths:
        return []
    UsdGeom.Xform.Define(stage, "/World/Buildings")
    added = []
    for index, building_path in enumerate(building_paths, 1):
        converted = converted_root / "buildings" / f"{building_path.stem}.usd"
        await convert_asset(building_path, converted)
        ref_path = f"/World/Buildings/{safe_ref_name(index, building_path)}"
        add_reference(stage, ref_path, converted)
        added.append(str(converted))
    return added


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


def camera_matrix(position: np.ndarray, target: np.ndarray):
    from isaacsim.core.utils.rotations import lookat_to_quatf
    from pxr import Gf

    quat = lookat_to_quatf(
        Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])),
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])),
        Gf.Vec3d(0.0, 1.0, 0.0),
    )
    return Gf.Matrix4d().SetRotate(quat).SetTranslateOnly(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )


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


def vc_to_usd(point: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray([point[0] * scale, point[2] * scale, point[1] * scale], dtype=float)


def build_views(
    segments: list[dict[str, Any]],
    count: int,
    usd_scale: float,
    camera_height: float,
    target_height: float,
) -> list[dict[str, Any]]:
    if not segments:
        raise RuntimeError("no suitable OSM road segments found")
    preferred_order = {"primary": 0, "secondary": 1, "tertiary": 2, "residential": 3, "unclassified": 4, "service": 5}
    ranked = sorted(
        segments,
        key=lambda item: (
            preferred_order.get(str(item["highway"]), 6),
            abs(item["mid"]["x"]) + abs(item["mid"]["y"]),
            -item["length"],
        ),
    )
    candidates = ranked[: max(count * 8, count)]
    stride = max(1, len(candidates) // max(1, count))
    selected = candidates[::stride][:count]
    views = []
    for index, segment in enumerate(selected, 1):
        mid = segment["mid"]
        heading = float(segment["heading"])
        forward = np.asarray([math.cos(heading), math.sin(heading), 0.0], dtype=float)
        right = np.asarray([-forward[1], forward[0], 0.0], dtype=float)
        center = np.asarray([mid["x"], mid["y"], mid["z"]], dtype=float)
        eye_vc = center - forward * 7.5 + right * 1.4 + np.asarray([0.0, 0.0, camera_height])
        target_vc = center + forward * 16.0 + np.asarray([0.0, 0.0, target_height])
        views.append(
            {
                "name": f"street_{index:02d}_{segment['highway']}",
                "eye": vc_to_usd(eye_vc, usd_scale),
                "target": vc_to_usd(target_vc, usd_scale),
                "eye_vc": eye_vc,
                "target_vc": target_vc,
                "segment": segment,
            }
        )
    return views


def serialize_view(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": view["name"],
        "eye_vc": view["eye_vc"].tolist(),
        "target_vc": view["target_vc"].tolist(),
        "way_id": view["segment"]["way_id"],
        "highway": view["segment"]["highway"],
        "segment_length": view["segment"]["length"],
    }


def load_views(path: Path, usd_scale: float) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    raw_views = raw.get("views", raw.get("renders", raw if isinstance(raw, list) else []))
    views = []
    for index, item in enumerate(raw_views, 1):
        eye_vc = np.asarray(item.get("eye_vc") or item.get("camera_position_vc"), dtype=float)
        target_vc = np.asarray(item.get("target_vc") or item.get("target_position_vc"), dtype=float)
        highway = item.get("highway", "road")
        name = item.get("name") or f"street_{index:02d}_{highway}"
        views.append(
            {
                "name": name,
                "eye": vc_to_usd(eye_vc, usd_scale),
                "target": vc_to_usd(target_vc, usd_scale),
                "eye_vc": eye_vc,
                "target_vc": target_vc,
                "segment": {
                    "way_id": item.get("way_id"),
                    "highway": highway,
                    "length": item.get("segment_length"),
                },
            }
        )
    if not views:
        raise RuntimeError(f"no reusable camera views found in {path}")
    return views


def render_streets(args: argparse.Namespace) -> None:
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux

    city_dir = args.city_dir.resolve()
    output_dir = args.output_dir.resolve()
    terrain_glb = city_dir / "terrain.glb"
    converted_usd = args.converted_usd.resolve() if args.converted_usd else output_dir / "converted" / "terrain.usd"
    loop = asyncio.get_event_loop()
    print(f"[isaac-flat-street] converting {terrain_glb} -> {converted_usd}", flush=True)
    loop.run_until_complete(convert_asset(terrain_glb, converted_usd))

    print(f"[isaac-flat-street] opening {converted_usd}", flush=True)
    stage_to_attach = Usd.Stage.Open(str(converted_usd))
    if stage_to_attach is None:
        raise RuntimeError(f"failed to open converted USD: {converted_usd}")
    success, _error = loop.run_until_complete(attach_stage(stage_to_attach))
    if not success:
        raise RuntimeError(f"failed to attach converted USD: {converted_usd}")
    wait_for_stage_load()
    stage = omni.usd.get_context().get_stage()
    building_refs: list[str] = []
    if args.include_buildings:
        print("[isaac-flat-street] converting/adding buildings", flush=True)
        building_refs = loop.run_until_complete(
            convert_and_add_buildings(stage, city_dir, output_dir / "converted", args.max_buildings)
        )
        wait_for_stage_load()
        print(f"[isaac-flat-street] added {len(building_refs)} building reference(s)", flush=True)

    rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", 1.1)
    dome = UsdLux.DomeLight.Define(stage, "/World/FlatGroundStreetDome")
    dome.CreateIntensityAttr(850.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/FlatGroundStreetSun")
    sun.CreateIntensityAttr(3200.0)
    sun.CreateAngleAttr(0.45)

    camera_path = "/World/FlatGroundStreetCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 5000.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    transform = camera_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

    render_product = rep.create.render_product(camera_path, tuple(args.resolution), name="FlatGroundStreetRender")
    rgb_annotator = rep.annotators.get("rgb")
    rgb_annotator.attach([render_product])
    app = omni.kit.app.get_app()

    if args.views_json:
        views = load_views(args.views_json.resolve(), float(args.usd_scale))
    else:
        views = build_views(
            read_osm_segments(city_dir),
            int(args.views),
            float(args.usd_scale),
            float(args.camera_height),
            float(args.target_height),
        )
    if args.write_views_json:
        args.write_views_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_views_json.write_text(json.dumps({"views": [serialize_view(view) for view in views]}, indent=2) + "\n")
    renders = []
    for view in views:
        transform.Set(camera_matrix(view["eye"], view["target"]))
        rgb = render_rgb(app, rgb_annotator, args.warmup_frames)
        if rgb is None:
            raise RuntimeError(f"empty render for {view['name']}")
        path = output_dir / f"{view['name']}.png"
        write_png(path, rgb)
        renders.append(
            {
                "path": str(path),
                "camera_position": view["eye"].tolist(),
                "target_position": view["target"].tolist(),
                "camera_position_vc": view["eye_vc"].tolist(),
                "target_position_vc": view["target_vc"].tolist(),
                "way_id": view["segment"]["way_id"],
                "highway": view["segment"]["highway"],
                "segment_length": view["segment"]["length"],
            }
        )
        print(f"[isaac-flat-street] wrote {path}", flush=True)

    summary = {
        "city_dir": str(city_dir),
        "converted_usd": str(converted_usd),
        "building_refs": building_refs,
        "renders": renders,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "flat_ground_street_renders.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    print("[isaac-flat-street] launching Isaac app", flush=True)
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
        render_streets(args)
    except BaseException:
        print("[isaac-flat-street] failed:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        print("[isaac-flat-street] closing app", flush=True)
        app.close()


if __name__ == "__main__":
    main()
