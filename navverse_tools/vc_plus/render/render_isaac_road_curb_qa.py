#!/usr/bin/env python3
"""Render a focused road-curb QA view from a VC USD scene in Isaac Sim."""

from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Columbus road lowering curb QA view in Isaac Sim.")
    parser.add_argument("--usd", required=True)
    parser.add_argument("--geometry-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 720))
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--surface-mode", choices=("terrain", "regions"), default="terrain")
    parser.add_argument("--hide-regions", nargs="*", default=())
    parser.add_argument("--terrain-only", action="store_true")
    parser.add_argument("--debug-region-colors", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--multi-gpu", action="store_true")
    return parser.parse_args()


async def attach_stage(stage_to_attach):
    import omni.usd

    return await omni.usd.get_context().attach_stage_async(stage_to_attach)


def wait_for_stage_load() -> None:
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    for _ in range(900):
        app.update()
        if omni.usd.get_context().get_stage_loading_status()[2] == 0:
            return
    raise TimeoutError("USD stage did not finish loading")


def camera_matrix(position: np.ndarray, target: np.ndarray):
    from isaacsim.core.utils.rotations import lookat_to_quatf
    from pxr import Gf

    quat = lookat_to_quatf(
        Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])),
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    matrix = Gf.Matrix4d().SetRotateOnly(quat)
    matrix.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    return matrix


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


def annotator_array(annotator):
    data = annotator.get_data()
    if hasattr(data, "numpy"):
        data = data.numpy()
    if isinstance(data, dict):
        data = data.get("data")
    if data is None or not getattr(data, "size", 0):
        return None
    return np.asarray(data)


def write_depth(path: Path, depth: np.ndarray) -> dict[str, float]:
    values = np.asarray(depth, dtype=np.float32).squeeze()
    finite = np.isfinite(values) & (values > 0.0)
    if not np.any(finite):
        raise RuntimeError("empty depth render")
    low, high = np.percentile(values[finite], [1.0, 99.0])
    scaled = np.zeros(values.shape, dtype=np.uint16)
    if high > low:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        scaled[finite] = np.round((1.0 - normalized[finite]) * 65535.0).astype(np.uint16)
    Image.fromarray(scaled).save(path)
    preview_low, preview_high = np.percentile(values[finite], [2.0, 95.0])
    preview = np.zeros(values.shape, dtype=np.uint8)
    if preview_high > preview_low:
        normalized = np.clip((values - preview_low) / (preview_high - preview_low), 0.0, 1.0)
        preview[finite] = np.round((1.0 - normalized[finite]) * 255.0).astype(np.uint8)
    preview_path = path.with_name(f"{path.stem}_preview.png")
    Image.fromarray(preview).save(preview_path)
    np.save(path.with_suffix(".npy"), values)
    return {
        "min_m": float(np.min(values[finite])),
        "max_m": float(np.max(values[finite])),
        "p01_m": float(low),
        "p99_m": float(high),
        "preview_path": str(preview_path),
        "preview_p02_m": float(preview_low),
        "preview_p95_m": float(preview_high),
    }


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def add_marker(stage, path: str, position: np.ndarray, color: tuple[float, float, float], radius: float = 0.16) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    material = UsdShade.Material.Define(stage, f"{path}_mat")
    shader = UsdShade.Shader.Define(stage, f"{path}_mat/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )
    UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)


def render_curb(args: argparse.Namespace) -> None:
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    usd_path = Path(args.usd).resolve()
    geometry_path = Path(args.geometry_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = json.loads(geometry_path.read_text())
    best = geometry["best"]
    inside_xy = np.asarray(best["inside_xy"], dtype=float)
    outside_xy = np.asarray(best["outside_xy"], dtype=float)
    inside_z = float(best["inside_z"])
    outside_z = float(best["outside_z"])
    inside = np.asarray([inside_xy[0], inside_xy[1], inside_z + 0.05], dtype=float)
    outside = np.asarray([outside_xy[0], outside_xy[1], outside_z + 0.05], dtype=float)
    normal_xy = outside_xy - inside_xy
    normal_xy = normal_xy / max(1e-9, np.linalg.norm(normal_xy))
    tangent_xy = np.asarray([-normal_xy[1], normal_xy[0]], dtype=float)
    midpoint_xy = (inside_xy + outside_xy) * 0.5
    midpoint_z = (inside_z + outside_z) * 0.5
    views = [
        {
            "name": "wide",
            "eye": np.asarray(
                [
                    midpoint_xy[0] + normal_xy[0] * 4.6 - tangent_xy[0] * 5.0,
                    midpoint_xy[1] + normal_xy[1] * 4.6 - tangent_xy[1] * 5.0,
                    midpoint_z + 1.15,
                ],
                dtype=float,
            ),
            "target": np.asarray(
                [
                    midpoint_xy[0] + tangent_xy[0] * 1.2,
                    midpoint_xy[1] + tangent_xy[1] * 1.2,
                    midpoint_z + 0.18,
                ],
                dtype=float,
            ),
            "focal_length": 22.0,
        },
        {
            "name": "roadside",
            "eye": np.asarray(
                [
                    midpoint_xy[0] - normal_xy[0] * 2.6 - tangent_xy[0] * 2.0,
                    midpoint_xy[1] - normal_xy[1] * 2.6 - tangent_xy[1] * 2.0,
                    midpoint_z + 0.72,
                ],
                dtype=float,
            ),
            "target": np.asarray(
                [
                    midpoint_xy[0] + normal_xy[0] * 0.25 + tangent_xy[0] * 0.2,
                    midpoint_xy[1] + normal_xy[1] * 0.25 + tangent_xy[1] * 0.2,
                    midpoint_z + 0.06,
                ],
                dtype=float,
            ),
            "focal_length": 38.0,
        },
        {
            "name": "outside",
            "eye": np.asarray(
                [
                    midpoint_xy[0] + normal_xy[0] * 2.4 - tangent_xy[0] * 1.7,
                    midpoint_xy[1] + normal_xy[1] * 2.4 - tangent_xy[1] * 1.7,
                    midpoint_z + 0.82,
                ],
                dtype=float,
            ),
            "target": np.asarray(
                [
                    midpoint_xy[0] - normal_xy[0] * 0.15 + tangent_xy[0] * 0.2,
                    midpoint_xy[1] - normal_xy[1] * 0.15 + tangent_xy[1] * 0.2,
                    midpoint_z + 0.08,
                ],
                dtype=float,
            ),
            "focal_length": 38.0,
        },
        {
            "name": "top",
            "eye": np.asarray(
                [
                    midpoint_xy[0] + normal_xy[0] * 1.0 - tangent_xy[0] * 0.8,
                    midpoint_xy[1] + normal_xy[1] * 1.0 - tangent_xy[1] * 0.8,
                    midpoint_z + 5.2,
                ],
                dtype=float,
            ),
            "target": np.asarray(
                [
                    midpoint_xy[0],
                    midpoint_xy[1],
                    midpoint_z,
                ],
                dtype=float,
            ),
            "focal_length": 55.0,
        },
    ]

    print(f"[isaac-curb] opening stage: {usd_path}", flush=True)
    stage_to_attach = Usd.Stage.Open(str(usd_path))
    if stage_to_attach is None:
        raise RuntimeError(f"failed to open USD stage: {usd_path}")
    success, _error = asyncio.get_event_loop().run_until_complete(attach_stage(stage_to_attach))
    if not success:
        raise RuntimeError(f"failed to attach USD stage: {usd_path}")
    wait_for_stage_load()
    stage = omni.usd.get_context().get_stage()
    print("[isaac-curb] stage loaded", flush=True)

    if args.terrain_only:
        world = stage.GetPrimAtPath("/World")
        for child in world.GetChildren():
            if child.GetName() != "TerrainRegions" and child.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(child).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    terrain_regions = stage.GetPrimAtPath("/World/TerrainRegions")
    if terrain_regions:
        UsdGeom.Imageable(terrain_regions).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
        hidden_regions = set(args.hide_regions)
        for child in terrain_regions.GetChildren():
            visible = child.GetName() == "Terrain" if args.surface_mode == "terrain" else child.GetName() != "Terrain"
            if child.GetName() in hidden_regions:
                visible = False
            UsdGeom.Imageable(child).CreateVisibilityAttr(
                UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
            )
            if visible and args.debug_region_colors and child.GetTypeName() == "Mesh":
                color = {
                    "water": (0.05, 0.35, 0.95),
                    "green": (0.08, 0.65, 0.18),
                    "crosswalk": (0.95, 0.85, 0.12),
                    "curb": (0.95, 0.12, 0.1),
                    "sidewalk": (0.15, 0.75, 0.9),
                    "road": (0.12, 0.12, 0.12),
                    "Terrain": (0.45, 0.45, 0.45),
                }.get(child.GetName(), (0.7, 0.7, 0.7))
                material = UsdShade.Material.Define(stage, f"/World/IsaacTerrainRegionDebugMaterials/{child.GetName()}")
                shader = UsdShade.Shader.Define(
                    stage,
                    f"/World/IsaacTerrainRegionDebugMaterials/{child.GetName()}/PreviewSurface",
                )
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(child).Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )

    rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", 1.2)
    dome = UsdLux.DomeLight.Define(stage, "/World/IsaacCurbRenderDome")
    dome.CreateIntensityAttr(700.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/IsaacCurbRenderSun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(0.5)
    headlight = UsdLux.SphereLight.Define(stage, "/World/IsaacCurbRenderHeadlight")
    headlight.CreateIntensityAttr(3500.0)
    headlight.CreateRadiusAttr(3.0)
    headlight_xform = UsdGeom.Xformable(headlight.GetPrim())
    headlight_xform.ClearXformOpOrder()
    headlight_translate = headlight_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

    camera_path = "/World/IsaacCurbRenderCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    focal_length_attr = camera.CreateFocalLengthAttr(22.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 5000.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    transform = camera_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

    render_product = rep.create.render_product(camera_path, tuple(args.resolution), name="IsaacCurbRenderProduct")
    rgb_annotator = rep.annotators.get("rgb")
    rgb_annotator.attach([render_product])
    depth_annotator = rep.annotators.get("distance_to_image_plane")
    depth_annotator.attach([render_product])

    app = omni.kit.app.get_app()
    depth_stats = {}

    def render_view(view, suffix: str) -> Path:
        eye = view["eye"]
        target = view["target"]
        focal_length_attr.Set(float(view["focal_length"]))
        transform.Set(camera_matrix(eye, target))
        headlight_translate.Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2] + 0.3)))
        rgb = render_rgb(app, rgb_annotator, args.warmup_frames)
        if rgb is None:
            raise RuntimeError(f"empty curb render for {view['name']} {suffix}")
        path = output_dir / f"curb_{view['name']}_{suffix}.png"
        write_png(path, rgb)
        depth = annotator_array(depth_annotator)
        if depth is None:
            raise RuntimeError(f"empty curb depth render for {view['name']} {suffix}")
        depth_path = output_dir / f"curb_{view['name']}_{suffix}_depth.png"
        depth_stats[f"{view['name']}_{suffix}"] = {
            "path": str(depth_path),
            **write_depth(depth_path, depth),
        }
        return path

    actual_paths = {}
    for view in views:
        actual_paths[view["name"]] = str(render_view(view, "actual"))

    add_marker(stage, "/World/IsaacCurbInsideRoadMarker", inside, (1.0, 0.05, 0.02))
    add_marker(stage, "/World/IsaacCurbOutsideRoadMarker", outside, (0.0, 0.9, 0.15))
    marked_paths = {}
    for view in views:
        marked_paths[view["name"]] = str(render_view(view, "marked"))

    summary = {
        "usd": str(usd_path),
        "geometry_json": str(geometry_path),
        "actual_renders": actual_paths,
        "marked_renders": marked_paths,
        "depth_renders": depth_stats,
        "inside_road_xyz": inside.tolist(),
        "outside_road_xyz": outside.tolist(),
        "outside_minus_inside_z": outside_z - inside_z,
        "views": [
            {
                "name": view["name"],
                "camera_position": view["eye"].tolist(),
                "target_position": view["target"].tolist(),
                "focal_length": view["focal_length"],
            }
            for view in views
        ],
        "way_id": best.get("way_id"),
    }
    (output_dir / "curb_isaac_qa.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    print("[isaac-curb] launching Isaac app", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
            "width": int(args.resolution[0]),
            "height": int(args.resolution[1]),
            "multi_gpu": bool(args.multi_gpu),
            "samples_per_pixel_per_frame": 1,
            "anti_aliasing": 0,
        }
    )
    try:
        render_curb(args)
    except BaseException:
        print("[isaac-curb] failed:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        print("[isaac-curb] closing app", flush=True)
        app.close()


if __name__ == "__main__":
    main()
