#!/usr/bin/env python3
"""Render Isaac/Kit camera views around a VC innout doorway."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render exit views from an innout USD in Isaac Sim.")
    parser.add_argument("--usd", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(1280, 900))
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--light-exit-stage", action="store_true", help="Render a compact stage with only the exit cut and room.")
    parser.add_argument("--proxy-room", action="store_true", help="Use simple aligned floor/walls instead of the full GR interior.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--light-scale", type=float, default=1.0)
    parser.add_argument("--capture-depth", action="store_true")
    parser.add_argument("--view-names", nargs="*", default=None)
    parser.add_argument("--kit_args", default="")
    parser.add_argument("--multi-gpu", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize_xy(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    norm = np.linalg.norm(arr)
    if norm <= 1e-9:
        raise ValueError(f"zero-length vector: {values}")
    return arr / norm


def camera_quat(position: np.ndarray, target: np.ndarray):
    from isaacsim.core.utils.rotations import lookat_to_quatf
    from pxr import Gf

    quat = lookat_to_quatf(
        Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])),
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    return quat


def planned_views(plan: dict, stage=None) -> list[dict[str, np.ndarray | str]]:
    center = np.asarray([plan["door_center_xy"][0], plan["door_center_xy"][1], plan["door_bottom_z"] + 1.35], dtype=float)
    outward = normalize_xy(plan["outward_normal_xy"])
    tangent = normalize_xy(plan["tangent_xy"])
    outward3 = np.asarray([outward[0], outward[1], 0.0])
    tangent3 = np.asarray([tangent[0], tangent[1], 0.0])
    z = np.asarray([0.0, 0.0, 1.0])
    sign_center = center + z * 1.4
    try:
        if stage is not None:
            from pxr import Usd, UsdGeom

            sign_prim = stage.GetPrimAtPath("/World/Objaverse/NavVerseExitSigns/green_exit_sign_0000")
            if not sign_prim:
                raise RuntimeError("composed stage has no exit sign")
            box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(sign_prim)
            aligned = box.ComputeAlignedBox()
            sign_center = (np.asarray(aligned.GetMin(), dtype=float) + np.asarray(aligned.GetMax(), dtype=float)) * 0.5
            if not np.all(np.isfinite(sign_center)):
                raise RuntimeError("exit sign world bbox is not finite")
        else:
            raise RuntimeError("stage is required for the composed sign world bbox")
    except Exception:
        try:
            from navverse_tools.vc_plus.exit_signs import DEFAULT_EXIT_SIGN_ASSET, doorway_sign_transform

            transform = doorway_sign_transform(plan, DEFAULT_EXIT_SIGN_ASSET)
            sign_center = np.asarray(transform["translation"], dtype=float)
            sign_center[2] = float(transform["sign_bottom_z"] + transform["sign_top_z"]) * 0.5
        except Exception:
            sign_center = np.asarray(
                [
                    plan["door_center_xy"][0],
                    plan["door_center_xy"][1],
                    plan["door_bottom_z"] + min(float(plan.get("door_height", 2.6)) + 0.18, 3.05),
                ],
                dtype=float,
            )
    views = [
        {
            "name": "sidewalk_north_to_room",
            "position": center + tangent3 * 14.0 + outward3 * 2.0 + z * 0.5,
            "target": center - outward3 * 0.3,
        },
        {
            "name": "sidewalk_south_to_room",
            "position": center - tangent3 * 14.0 + outward3 * 2.0 + z * 0.5,
            "target": center - outward3 * 0.3,
        },
        {
            "name": "street_wide_to_room",
            "position": center + outward3 * 14.0 + tangent3 * 4.0 + z * 0.6,
            "target": center - outward3 * 0.5 + z * 0.1,
        },
        {
            "name": "street_oblique_to_room",
            "position": center + outward3 * 11.5 + tangent3 * 8.0 + z * 0.55,
            "target": center - outward3 * 0.7 + z * 0.05,
        },
        {
            "name": "street_corner_to_room",
            "position": center + outward3 * 7.5 + tangent3 * 11.5 + z * 0.45,
            "target": center - outward3 * 0.8,
        },
        {
            "name": "street_direct_to_room",
            "position": center + outward3 * 9.0 + tangent3 * 0.9 + z * 0.45,
            "target": center - outward3 * 0.8,
        },
        {
            "name": "outside_to_exit",
            "position": center + outward3 * 5.0 + tangent3 * 0.4 + z * 0.35,
            "target": center - outward3 * 0.8,
        },
        {
            "name": "threshold_oblique",
            "position": center + outward3 * 2.1 + tangent3 * 1.7 + z * 0.55,
            "target": center - outward3 * 1.0 + z * 0.05,
        },
        {
            "name": "inside_to_outside",
            "position": center - outward3 * 4.2 - tangent3 * 0.25 + z * 0.45,
            "target": center + outward3 * 1.2,
        },
        {
            "name": "side_cutaway",
            "position": center + tangent3 * 4.8 + z * 2.0,
            "target": center,
        },
        {
            "name": "threshold_profile_left",
            "position": center + tangent3 * 3.2 + outward3 * 0.7 - z * 0.85,
            "target": center + outward3 * 0.45 - z * 1.1,
        },
        {
            "name": "threshold_profile_right",
            "position": center - tangent3 * 3.2 + outward3 * 0.7 - z * 0.85,
            "target": center + outward3 * 0.45 - z * 1.1,
        },
        {
            "name": "facade_ground_close",
            "position": center + outward3 * 4.0 + tangent3 * 0.5 - z * 0.65,
            "target": center + outward3 * 0.3 - z * 1.12,
        },
        {
            "name": "exit_sign_close",
            "position": sign_center - outward3 * 1.65 + tangent3 * 0.18 - z * 0.42,
            "target": sign_center - z * 0.03,
        },
        {
            "name": "exit_sign_outer_side",
            "position": sign_center + outward3 * 1.65 + tangent3 * 0.18 - z * 0.42,
            "target": sign_center - z * 0.03,
        },
    ]
    return views


def door_vectors(plan: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray([plan["door_center_xy"][0], plan["door_center_xy"][1], plan["door_bottom_z"] + 1.35], dtype=float)
    outward = normalize_xy(plan["outward_normal_xy"])
    tangent = normalize_xy(plan["tangent_xy"])
    return center, np.asarray([outward[0], outward[1], 0.0]), np.asarray([tangent[0], tangent[1], 0.0])


def select_views(plan: dict, stage, names: list[str] | None) -> list[dict[str, np.ndarray | str]]:
    views = planned_views(plan, stage)
    if not names:
        return views
    allowed = set(names)
    selected = [view for view in views if str(view["name"]) in allowed]
    missing = sorted(allowed - {str(view["name"]) for view in selected})
    if missing:
        raise ValueError(f"unknown view names: {missing}")
    return selected


def rgb_array(data, view_name: str) -> np.ndarray:
    if hasattr(data, "numpy"):
        data = data.numpy()
    if isinstance(data, dict):
        data = data.get("data")
    if data is None or data.size == 0:
        raise RuntimeError(f"empty RGB frame for {view_name}")
    return data


def depth_array(data, view_name: str) -> np.ndarray:
    if hasattr(data, "numpy"):
        data = data.numpy()
    if isinstance(data, dict):
        data = data.get("data")
    if data is None or data.size == 0:
        raise RuntimeError(f"empty depth frame for {view_name}")
    return np.asarray(data, dtype=np.float32).squeeze()


def write_depth_preview(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(depth[valid], [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        preview[valid] = np.asarray((1.0 - normalized[valid]) * 255.0, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(preview, cv2.COLORMAP_TURBO))


def make_preview_material(stage, path: str, color: tuple[float, float, float]):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_cube(stage, path: str, material, translate: tuple[float, float, float], scale: tuple[float, float, float]):
    from pxr import Gf, UsdGeom, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
    return cube


def build_light_exit_stage(plan: dict, output_dir: Path, proxy_room: bool = False):
    from pxr import Gf, Usd, UsdGeom, UsdShade

    stage_path = output_dir / "CHICAGO_exit_light_stage.usda"
    if stage_path.exists():
        stage_path.unlink()
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    stage.DefinePrim("/World/Looks", "Scope")

    cut = stage.DefinePrim("/World/cut_building", "Xform")
    cut.GetReferences().AddReference(str(Path(plan["cut_usd"]).resolve()), "/cut_building")

    center = plan["door_center_xy"]
    if proxy_room:
        room = stage.DefinePrim("/World/proxy_room", "Xform")
        room_xform = UsdGeom.Xformable(room)
        room_xform.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], plan["door_bottom_z"]))
        tangent = normalize_xy(plan["tangent_xy"])
        yaw = math.atan2(float(tangent[1]), float(tangent[0]))
        room_xform.AddOrientOp().Set(Gf.Quatf(math.cos(yaw * 0.5), Gf.Vec3f(0.0, 0.0, math.sin(yaw * 0.5))))
        wall_mat = make_preview_material(stage, "/World/Looks/proxy_wall", (0.78, 0.76, 0.69))
        floor_mat = make_preview_material(stage, "/World/Looks/proxy_floor", (0.38, 0.40, 0.38))
        width = 4.4
        depth = 6.6
        height = 3.0
        bind_cube(stage, "/World/proxy_room/floor", floor_mat, (0.0, depth * 0.5, 0.02), (width, depth, 0.04))
        bind_cube(stage, "/World/proxy_room/left_wall", wall_mat, (-width * 0.5, depth * 0.5, height * 0.5), (0.06, depth, height))
        bind_cube(stage, "/World/proxy_room/right_wall", wall_mat, (width * 0.5, depth * 0.5, height * 0.5), (0.06, depth, height))
        bind_cube(stage, "/World/proxy_room/back_wall", wall_mat, (0.0, depth, height * 0.5), (width, 0.06, height))
        bind_cube(stage, "/World/proxy_room/ceiling", wall_mat, (0.0, depth * 0.5, height), (width, depth, 0.04))
    else:
        room = stage.DefinePrim("/World/start_result_navigation", "Xform")
        room_xform = UsdGeom.Xformable(room)
        room_xform.AddTranslateOp().Set(Gf.Vec3d(*plan["room_translate"]))
        yaw = math.radians(plan["room_yaw_deg"])
        room_xform.AddOrientOp().Set(Gf.Quatf(math.cos(yaw * 0.5), Gf.Vec3f(0.0, 0.0, math.sin(yaw * 0.5))))
        room_xform.AddScaleOp().Set(Gf.Vec3d(plan["room_scale"], plan["room_scale"], plan["room_scale"]))
        room.GetReferences().AddReference(str(Path(plan["interior_usd"]).resolve()))
        removed = plan.get("removed_interior_door_path")
        if removed:
            stage.OverridePrim(removed).SetActive(False)

    ground = UsdGeom.Cube.Define(stage, "/World/exit_ground")
    ground.CreateSizeAttr(1.0)
    ground_xform = UsdGeom.Xformable(ground.GetPrim())
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], plan["door_bottom_z"] - 0.035))
    ground_xform.AddScaleOp().Set(Gf.Vec3d(9.0, 9.0, 0.035))
    material = make_preview_material(stage, "/World/Looks/ground_matte", (0.46, 0.45, 0.42))
    UsdShade.MaterialBindingAPI(ground.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()
    return stage


async def attach_stage(stage_to_attach):
    import omni.usd

    return await omni.usd.get_context().attach_stage_async(stage_to_attach)


def wait_for_stage_load() -> None:
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    for _ in range(600):
        app.update()
        if omni.usd.get_context().get_stage_loading_status()[2] == 0:
            return
    raise TimeoutError("USD stage did not finish loading")


def render_views(args: argparse.Namespace) -> None:
    import omni.kit.app
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux

    usd_path = Path(args.usd).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()
    plan = read_json(Path(args.plan))
    if args.light_exit_stage:
        print("[isaac-exit] building compact exit stage", flush=True)
        stage_to_attach = build_light_exit_stage(plan, output_dir, proxy_room=args.proxy_room)
        print(f"[isaac-exit] compact stage: {stage_to_attach.GetRootLayer().identifier}", flush=True)
    else:
        print(f"[isaac-exit] opening stage: {usd_path}", flush=True)
        start_time = time.time()
        stage_to_attach = Usd.Stage.Open(str(usd_path))
        print(f"[isaac-exit] pxr stage opened in {time.time() - start_time:.2f}s", flush=True)
        if stage_to_attach is None:
            raise RuntimeError(f"failed to open USD stage: {usd_path}")
    start_time = time.time()
    success, _error = asyncio.get_event_loop().run_until_complete(attach_stage(stage_to_attach))
    print(f"[isaac-exit] stage attached in {time.time() - start_time:.2f}s", flush=True)
    if not success:
        raise RuntimeError(f"failed to open USD stage: {usd_path}")
    wait_for_stage_load()
    stage = omni.usd.get_context().get_stage()
    print("[isaac-exit] stage loaded", flush=True)

    hdr = stage.GetPrimAtPath("/World/start_result_navigation/__default_setting/HDR_Sphere")
    if hdr and hdr.IsValid():
        hdr.SetActive(False)

    import omni.replicator.core as rep

    light_scale = float(args.light_scale)
    rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", max(0.2, light_scale))
    dome = UsdLux.DomeLight.Define(stage, "/World/IsaacExitRenderDome")
    dome.CreateIntensityAttr(5000.0 * light_scale)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    light = UsdLux.DistantLight.Define(stage, "/World/IsaacExitRenderLight")
    light.CreateIntensityAttr(14000.0 * light_scale)
    light.CreateAngleAttr(0.45)
    center, outward3, _tangent3 = door_vectors(plan)
    for name, position, intensity in [
        ("IsaacExitOutdoorFill", center + outward3 * 3.0 + np.asarray([0.0, 0.0, 2.8]), 32000.0 * light_scale),
        ("IsaacExitIndoorFill", center - outward3 * 2.3 + np.asarray([0.0, 0.0, 1.9]), 48000.0 * light_scale),
    ]:
        fill = UsdLux.SphereLight.Define(stage, f"/World/{name}")
        fill.CreateIntensityAttr(intensity)
        fill.CreateRadiusAttr(4.0)
        fill_xform = UsdGeom.Xformable(fill.GetPrim())
        fill_xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    headlight = UsdLux.SphereLight.Define(stage, "/World/IsaacExitCameraHeadlight")
    headlight.CreateIntensityAttr(65000.0 * light_scale)
    headlight.CreateRadiusAttr(3.0)
    headlight_xform = UsdGeom.Xformable(headlight.GetPrim())
    headlight_xform.ClearXformOpOrder()
    headlight_translate = headlight_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

    camera_path = "/World/IsaacExitRenderCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(18.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    camera_xform = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    render_product = rep.create.render_product(camera_path, tuple(args.resolution), name="IsaacExitRenderProduct")
    rgb_annotator = rep.annotators.get("rgb")
    rgb_annotator.attach([render_product])
    depth_annotator = None
    if args.capture_depth:
        depth_annotator = rep.annotators.get("distance_to_camera")
        depth_annotator.attach([render_product])
    print("[isaac-exit] render product ready", flush=True)

    summary = []
    selected_views = select_views(plan, stage, args.view_names)
    for view in selected_views:
        name = str(view["name"])
        position = np.asarray(view["position"], dtype=float)
        target = np.asarray(view["target"], dtype=float)
        mat = Gf.Matrix4d().SetRotateOnly(camera_quat(position, target))
        mat.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        camera_xform.Set(mat)
        headlight_translate.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2] + 0.25)))
        omni.kit.app.get_app().update()
        rep.orchestrator.step(rt_subframes=max(1, args.warmup_frames), delta_time=0.0)
        rgb = rgb_array(rgb_annotator.get_data(), name)
        path = output_dir / f"{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"[isaac-exit] wrote {path}", flush=True)
        row = {"name": name, "path": str(path), "position": position.tolist(), "target": target.tolist()}
        if depth_annotator is not None:
            depth = depth_array(depth_annotator.get_data(), name)
            depth_path = output_dir / f"{name}_depth.npy"
            preview_path = output_dir / f"{name}_depth.png"
            np.save(depth_path, depth)
            write_depth_preview(preview_path, depth)
            row.update({"depth_path": str(depth_path), "depth_preview_path": str(preview_path)})
        summary.append(row)
        omni.kit.app.get_app().update()

    (output_dir / "isaac_exit_views.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[isaac-exit] wrote summary: {output_dir / 'isaac_exit_views.json'}", flush=True)


def main() -> None:
    args = parse_args()
    print("[isaac-exit] launching Isaac app", flush=True)
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
        print("[isaac-exit] app launched", flush=True)
        render_views(args)
    except BaseException:
        print("[isaac-exit] failed:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        print("[isaac-exit] closing app", flush=True)
        app.close()


if __name__ == "__main__":
    main()
