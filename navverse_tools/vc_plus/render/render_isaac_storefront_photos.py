#!/usr/bin/env python3
"""Render storefront inspection screenshots from VC USD scenes in Isaac Sim."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render storefront screenshots from a VC outdoor USD in Isaac Sim.")
    parser.add_argument("--usd", required=True)
    parser.add_argument("--storefront-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(960, 540))
    parser.add_argument("--indices", default="")
    parser.add_argument("--max-stores", type=int, default=10)
    parser.add_argument("--camera-distance", type=float, default=8.0)
    parser.add_argument("--camera-distances", default="", help="Comma-separated candidate camera distances in meters.")
    parser.add_argument("--camera-lateral-offsets", default="0", help="Comma-separated candidate lateral offsets in meters.")
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--ambient-intensity", type=float, default=1.5)
    parser.add_argument("--dome-intensity", type=float, default=900.0)
    parser.add_argument("--sun-intensity", type=float, default=2500.0)
    parser.add_argument("--headlight-intensity", type=float, default=4500.0)
    parser.add_argument("--headlight-radius", type=float, default=5.0)
    parser.add_argument("--allow-flip", action="store_true", help="Try the opposite normal only for diagnostics.")
    parser.add_argument(
        "--preview-building-material",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily bind a neutral material to city buildings for geometry QA renders.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def selected_storefronts(plan: dict, indices_text: str, max_stores: int) -> list[tuple[int, dict]]:
    if indices_text:
        wanted = {int(part.strip()) for part in indices_text.split(",") if part.strip()}
    else:
        wanted = None
    selected: list[tuple[int, dict]] = []
    store_prim_index = 0
    for array_index, item in enumerate(plan.get("placements", [])):
        plan_index = int(item.get("index", array_index))
        if item.get("status") != "planned" or item.get("facade_status") != "facade_aligned":
            continue
        if not item.get("final_location") or not item.get("outward_normal_xy"):
            continue
        if wanted is not None and plan_index not in wanted:
            store_prim_index += 1
            continue
        record = dict(item)
        record["_store_array_index"] = array_index
        record["_plan_index"] = plan_index
        record["_store_prim_index"] = store_prim_index
        selected.append((plan_index, record))
        store_prim_index += 1
        if wanted is None and max_stores > 0 and len(selected) >= max_stores:
            break
    return selected


def safe_name(value: str) -> str:
    import re

    name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return name or "store"


def parse_distances(value: str, fallback: float) -> list[float]:
    distances = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        distances.append(float(part))
    return distances or [fallback]


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

    # USD cameras look along local -Z; this helper aligns local +Z with target-camera.
    quat = lookat_to_quatf(
        Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])),
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    mat = Gf.Matrix4d().SetRotateOnly(quat)
    mat.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    return mat


def image_detail_score(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
    return float(cv2.Canny(gray, 50, 150).mean() / 255.0)


def street_view_quality(rgb: np.ndarray) -> tuple[float, dict[str, float]]:
    image = rgb[:, :, :3].astype(np.float32)
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    white_ratio = float((gray > 238.0).mean())
    dark_ratio = float((gray < 12.0).mean())
    saturation = (image.max(axis=2) - image.min(axis=2)) / 255.0
    color_ratio = float((saturation > 0.12).mean())
    detail = image_detail_score(rgb)
    quality = detail * 5.0 + min(color_ratio / 0.22, 1.0) * 0.22
    quality -= max(0.0, white_ratio - 0.35) * 0.85
    quality -= max(0.0, 0.12 - color_ratio) * 0.75
    quality -= max(0.0, 0.010 - detail) * 8.0
    return float(quality), {
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "color_ratio": color_ratio,
        "detail": detail,
    }


def bind_preview_building_material(stage) -> int:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, "/World/IsaacStoreRenderBuildingPreviewMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/IsaacStoreRenderBuildingPreviewMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.44, 0.44, 0.42))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    city = stage.GetPrimAtPath("/World/city")
    if not city.IsValid():
        return 0
    bound = 0
    for child in city.GetChildren():
        name = child.GetName()
        if name == "Roof" or name.startswith("buildings_"):
            UsdShade.MaterialBindingAPI(child).Bind(material, UsdShade.Tokens.strongerThanDescendants)
            bound += 1
    return bound


def store_prim_for_item(stage, item: dict):
    store = stage.GetPrimAtPath("/World/store")
    if not store.IsValid():
        return None
    plan_index = int(item.get("_plan_index", item.get("index", 0)))
    plan_suffix = f"_{plan_index:04d}"
    for child in store.GetChildren():
        try:
            if child.GetCustomDataByKey("navverse_plan_index") == plan_index:
                return child
        except Exception:
            pass
        if child.GetName().endswith(plan_suffix):
            return child
    suffix = f"_{int(item.get('_store_prim_index', item.get('_store_array_index', plan_index))):04d}"
    for child in store.GetChildren():
        if not child.GetName().endswith(suffix):
            continue
        return child
    return None


def store_bbox_center(stage, bbox_cache, item: dict) -> np.ndarray | None:
    child = store_prim_for_item(stage, item)
    if child is None:
        return None
    box = bbox_cache.ComputeWorldBound(child).ComputeAlignedBox()
    mn = np.asarray(box.GetMin(), dtype=float)
    mx = np.asarray(box.GetMax(), dtype=float)
    if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
        return None
    if float(np.linalg.norm(mx - mn)) <= 1e-6:
        return None
    return (mn + mx) * 0.5


def make_preview_material(stage, path: str, color: tuple[float, float, float]):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_override_material(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def unbind_override_material(prim) -> None:
    from pxr import UsdShade

    api = UsdShade.MaterialBindingAPI(prim)
    try:
        api.UnbindDirectBinding()
    except Exception:
        pass


def mask_visibility_score(rgb: np.ndarray) -> tuple[float, float, dict[str, float]]:
    image = rgb[:, :, :3].astype(np.float32)
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    mask = (red > 160.0) & (green < 95.0) & (blue > 145.0) & ((red - green) > 70.0) & ((blue - green) > 55.0)
    area = int(mask.sum())
    height, width = mask.shape
    area_ratio = float(area / max(1, width * height))
    if area <= 0:
        return 0.0, area_ratio, {"area_ratio": area_ratio}
    ys, xs = np.nonzero(mask)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bbox_w = (x1 - x0 + 1.0) / width
    bbox_h = (y1 - y0 + 1.0) / height
    cx = ((x0 + x1) * 0.5) / width
    cy = ((y0 + y1) * 0.5) / height
    center_penalty = abs(cx - 0.5) * 0.55 + abs(cy - 0.53) * 0.25
    size_score = min(area_ratio / 0.045, 1.0) * 0.55 + min(bbox_w / 0.42, 1.0) * 0.30 + min(bbox_h / 0.30, 1.0) * 0.15
    border_penalty = 0.25 if x0 <= 2 or x1 >= width - 3 or y0 <= 2 or y1 >= height - 3 else 0.0
    score = max(0.0, size_score - center_penalty - border_penalty)
    metrics = {
        "area_ratio": area_ratio,
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "center_x": float(cx),
        "center_y": float(cy),
    }
    return float(score), area_ratio, metrics


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


def render_storefronts(args: argparse.Namespace) -> None:
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux

    usd_path = Path(args.usd).resolve()
    plan_path = Path(args.storefront_plan).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("store_*.png"):
        stale.unlink()
    plan = read_json(plan_path)
    items = selected_storefronts(plan, args.indices, args.max_stores)
    if not items:
        raise RuntimeError("no facade-aligned storefronts selected")
    camera_distances = parse_distances(args.camera_distances, args.camera_distance)
    lateral_offsets = parse_distances(args.camera_lateral_offsets, 0.0)

    print(f"[isaac-store] opening stage: {usd_path}", flush=True)
    start = time.time()
    stage_to_attach = Usd.Stage.Open(str(usd_path))
    print(f"[isaac-store] pxr stage opened in {time.time() - start:.2f}s", flush=True)
    if stage_to_attach is None:
        raise RuntimeError(f"failed to open USD stage: {usd_path}")
    success, _error = asyncio.get_event_loop().run_until_complete(attach_stage(stage_to_attach))
    if not success:
        raise RuntimeError(f"failed to attach USD stage: {usd_path}")
    wait_for_stage_load()
    stage = omni.usd.get_context().get_stage()
    print("[isaac-store] stage loaded", flush=True)
    deactivated_store_lights = 0
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/store") and "Light" in prim.GetTypeName():
            prim.SetActive(False)
            deactivated_store_lights += 1
    if deactivated_store_lights:
        print(f"[isaac-store] deactivated {deactivated_store_lights} store lights", flush=True)

    rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", float(args.ambient_intensity))
    if args.preview_building_material:
        bound = bind_preview_building_material(stage)
        print(f"[isaac-store] preview building material bound to {bound} city prims", flush=True)
    dome = UsdLux.DomeLight.Define(stage, "/World/IsaacStoreRenderDome")
    dome.CreateIntensityAttr(float(args.dome_intensity))
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    distant = UsdLux.DistantLight.Define(stage, "/World/IsaacStoreRenderSun")
    distant.CreateIntensityAttr(float(args.sun_intensity))
    distant.CreateAngleAttr(0.5)
    headlight = UsdLux.SphereLight.Define(stage, "/World/IsaacStoreRenderHeadlight")
    headlight.CreateIntensityAttr(float(args.headlight_intensity))
    headlight.CreateRadiusAttr(float(args.headlight_radius))
    headlight_xform = UsdGeom.Xformable(headlight.GetPrim())
    headlight_xform.ClearXformOpOrder()
    headlight_translate = headlight_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

    camera_path = "/World/IsaacStoreRenderCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(18.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_transform = camera_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    render_product = rep.create.render_product(camera_path, tuple(args.resolution), name="IsaacStoreRenderProduct")
    rgb_annotator = rep.annotators.get("rgb")
    rgb_annotator.attach([render_product])
    mask_material = make_preview_material(stage, "/World/IsaacStoreRenderMaskMaterial", (1.0, 0.0, 1.0))
    print("[isaac-store] render product ready", flush=True)

    summary = []
    for index, item in items:
        loc = np.asarray(item["final_location"], dtype=float)
        normal = np.asarray(item["outward_normal_xy"], dtype=float)
        normal = normal / max(1e-9, np.linalg.norm(normal))
        tangent = np.asarray([-normal[1], normal[0]], dtype=float)
        target_store = store_prim_for_item(stage, item)
        target = np.asarray([loc[0], loc[1], loc[2] + 1.35], dtype=float)
        best_score = -1.0
        best_edge_score = -1.0
        best_mask_area = 0.0
        best_metrics: dict[str, float] = {}
        best_quality_metrics: dict[str, float] = {}
        best_flipped = False
        best_distance = None
        best_lateral = None
        best_eye = None
        directions = [(False, normal)]
        if args.allow_flip:
            directions.append((True, -normal))

        if target_store is not None:
            bind_override_material(target_store, mask_material)
        try:
            for flipped, direction in directions:
                for distance in camera_distances:
                    for lateral in lateral_offsets:
                        eye = np.asarray(
                            [
                                target[0] + direction[0] * distance + tangent[0] * lateral,
                                target[1] + direction[1] * distance + tangent[1] * lateral,
                                target[2] + 0.35,
                            ],
                            dtype=float,
                        )
                        camera_transform.Set(camera_matrix(eye, target))
                        headlight_translate.Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2] + 0.3)))
                        rgb = render_rgb(omni.kit.app.get_app(), rgb_annotator, args.warmup_frames)
                        if rgb is None:
                            continue
                        edge_score = image_detail_score(rgb)
                        quality_score, quality_metrics = street_view_quality(rgb)
                        if target_store is not None:
                            score, mask_area, metrics = mask_visibility_score(rgb)
                            if mask_area <= 0.0:
                                combined_score = -1.0 + quality_score * 0.01
                            else:
                                combined_score = score + quality_score * 0.35
                        else:
                            mask_area = 0.0
                            metrics = {}
                            combined_score = quality_score
                        if combined_score > best_score:
                            best_score = combined_score
                            best_edge_score = edge_score
                            best_mask_area = mask_area
                            best_metrics = metrics
                            best_quality_metrics = quality_metrics
                            best_flipped = flipped
                            best_distance = distance
                            best_lateral = lateral
                            best_eye = eye
            if target_store is not None:
                unbind_override_material(target_store)
                for _ in range(2):
                    omni.kit.app.get_app().update()
        except Exception:
            if target_store is not None:
                unbind_override_material(target_store)
            raise

        if best_eye is None:
            print(f"[isaac-store] storefront {index} failed: empty RGB for all camera candidates", flush=True)
            summary.append(
                {
                    "index": index,
                    "path": None,
                    "qa_status": "failed",
                    "qa_reasons": ["empty_rgb"],
                    "qa_warnings": [],
                    "visibility_score": None,
                    "edge_score": None,
                    "mask_area_ratio": 0.0,
                    "visibility_metrics": {},
                    "quality_metrics": {},
                    "target_store_prim": target_store.GetPath().pathString if target_store is not None else None,
                    "flipped": False,
                    "camera_distance": None,
                    "camera_lateral_offset": None,
                    "camera_position": None,
                    "target_position": target.tolist(),
                    "outward_normal_xy": normal.tolist(),
                }
            )
            continue
        camera_transform.Set(camera_matrix(best_eye, target))
        headlight_translate.Set(Gf.Vec3d(float(best_eye[0]), float(best_eye[1]), float(best_eye[2] + 0.3)))
        best_rgb = render_rgb(omni.kit.app.get_app(), rgb_annotator, args.warmup_frames)
        if best_rgb is None:
            raise RuntimeError(f"empty final RGB for storefront {index}")
        qa_reasons = []
        qa_warnings = []
        if target_store is None:
            qa_reasons.append("missing_target_store_prim")
        if best_mask_area < 0.004:
            qa_reasons.append("target_not_visible")
        if best_metrics.get("bbox_w", 0.0) < 0.08 or best_metrics.get("bbox_h", 0.0) < 0.08:
            qa_reasons.append("target_too_small_or_clipped")
        if best_mask_area >= 0.004 and best_quality_metrics.get("color_ratio", 0.0) < 0.04 and best_quality_metrics.get("detail", 0.0) < 0.008:
            qa_reasons.append("low_store_visual_signal")
        road_distance = item.get("road_distance")
        if road_distance is None:
            road_distance = item.get("storefront_ground_road_distance")
        if road_distance is not None and float(road_distance) > 14.0:
            qa_reasons.append("not_close_to_street")
        if best_quality_metrics.get("color_ratio", 0.0) < 0.12 or best_quality_metrics.get("detail", 0.0) < 0.010:
            qa_warnings.append("low_street_context_signal")
        if best_quality_metrics.get("dark_ratio", 0.0) > 0.60:
            qa_warnings.append("dark_render")
        if best_quality_metrics.get("white_ratio", 0.0) > 0.35:
            qa_warnings.append("much_blank_wall")
        qa_status = "pass" if not qa_reasons else "failed"
        amenity = safe_name(str(item.get("amenity") or item.get("asset_type") or "store"))
        path = output_dir / f"store_{index:04d}_{amenity}.png"
        cv2.imwrite(str(path), cv2.cvtColor(best_rgb, cv2.COLOR_RGB2BGR))
        print(
            f"[isaac-store] wrote {path} qa={qa_status} visibility={best_score:.4f} edge={best_edge_score:.4f} area={best_mask_area:.4f} flipped={best_flipped} distance={best_distance} lateral={best_lateral}",
            flush=True,
        )
        summary.append(
            {
                "index": index,
                "path": str(path),
                "qa_status": qa_status,
                "qa_reasons": qa_reasons,
                "qa_warnings": qa_warnings,
                "visibility_score": best_score,
                "edge_score": best_edge_score,
                "mask_area_ratio": best_mask_area,
                "visibility_metrics": best_metrics,
                "quality_metrics": best_quality_metrics,
                "target_store_prim": target_store.GetPath().pathString if target_store is not None else None,
                "flipped": best_flipped,
                "camera_distance": best_distance,
                "camera_lateral_offset": best_lateral,
                "camera_position": best_eye.tolist(),
                "target_position": target.tolist(),
                "outward_normal_xy": normal.tolist(),
            }
        )
        omni.kit.app.get_app().update()

    (output_dir / "isaac_storefront_views.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[isaac-store] wrote summary: {output_dir / 'isaac_storefront_views.json'}", flush=True)


def main() -> None:
    args = parse_args()
    print("[isaac-store] launching Isaac app", flush=True)
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
        print("[isaac-store] app launched", flush=True)
        render_storefronts(args)
    except BaseException:
        print("[isaac-store] failed:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        print("[isaac-store] closing app", flush=True)
        app.close()


if __name__ == "__main__":
    main()
