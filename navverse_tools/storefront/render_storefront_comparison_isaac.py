#!/usr/bin/env python3
"""Render original/repaired storefront pairs with Isaac Sim RTX."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import traceback
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, default=(640, 480))
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--names", default="", help="Comma-separated storefront names.")
    parser.add_argument("--dome-intensity", type=float, default=250.0)
    parser.add_argument("--sun-intensity", type=float, default=800.0)
    parser.add_argument("--headlight-intensity", type=float, default=1200.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


async def attach_stage(stage) -> tuple[bool, str]:
    import omni.usd

    return await omni.usd.get_context().attach_stage_async(stage)


def wait_for_stage_load(app) -> None:
    import omni.usd

    for _ in range(900):
        app.update()
        if omni.usd.get_context().get_stage_loading_status()[2] == 0:
            return
    raise TimeoutError("USD stage did not finish loading")


def camera_matrix(position: np.ndarray, target: np.ndarray):
    from isaacsim.core.utils.rotations import lookat_to_quatf
    from pxr import Gf

    quat = lookat_to_quatf(
        Gf.Vec3d(*target.tolist()),
        Gf.Vec3d(*position.tolist()),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    matrix = Gf.Matrix4d().SetRotateOnly(quat)
    matrix.SetTranslateOnly(Gf.Vec3d(*position.tolist()))
    return matrix


def render_rgb(app, annotator, warmup_frames: int) -> np.ndarray:
    import omni.replicator.core as rep

    for _ in range(4):
        app.update()
        rep.orchestrator.step(rt_subframes=max(1, warmup_frames), delta_time=0.0)
        data = annotator.get_data()
        if hasattr(data, "numpy"):
            data = data.numpy()
        if isinstance(data, dict):
            data = data.get("data")
        if data is not None and data.size > 0:
            return np.asarray(data[:, :, :3], dtype=np.uint8)
    raise RuntimeError("RGB annotator returned no pixels after four attempts")


def asset_bounds(stage, asset_prim) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    aligned = cache.ComputeWorldBound(asset_prim).ComputeAlignedBox()
    lower = np.asarray(aligned.GetMin(), dtype=float)
    upper = np.asarray(aligned.GetMax(), dtype=float)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise RuntimeError("non-finite asset bounds")
    if float(np.linalg.norm(upper - lower)) <= 1e-6:
        raise RuntimeError("empty asset bounds")
    return lower, upper


def safe_prim_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def describe_asset(stage, asset_prim, usd_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    meshes = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and prim.GetPath().HasPrefix(asset_prim.GetPath())
    ]
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        textures = []
        if material:
            for descendant in Usd.PrimRange(material.GetPrim()):
                shader = UsdShade.Shader(descendant)
                if not shader:
                    continue
                file_input = shader.GetInput("file")
                if file_input:
                    asset_path = file_input.Get()
                    textures.append(getattr(asset_path, "resolvedPath", "") or str(asset_path))
        print(
            "  VERIFY"
            f" usd={usd_path} mesh={prim.GetPath()}"
            f" points={len(mesh.GetPointsAttr().Get() or [])}"
            f" faces={len(mesh.GetFaceVertexCountsAttr().Get() or [])}"
            f" material={material.GetPath() if material else '<none>'}"
            f" textures={textures}",
            flush=True,
        )


def make_contact_sheet(
    pairs: list[tuple[str, Path, Path, int]], destination: Path, columns: int
) -> None:
    view_width, view_height = 400, 300
    title_height, label_height = 60, 36
    tile_width, tile_height = 2 * view_width, view_height + label_height
    rows = math.ceil(len(pairs) / columns)
    sheet = np.full(
        (title_height + rows * tile_height, columns * tile_width, 3),
        24,
        dtype=np.uint8,
    )
    cv2.putText(
        sheet,
        "ISAAC SIM RTX: BEFORE (left) / REPAIRED (right), +Y front",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (242, 242, 242),
        2,
        cv2.LINE_AA,
    )
    for index, (name, before_path, after_path, vertex_count) in enumerate(pairs):
        column, row = index % columns, index // columns
        left = column * tile_width
        top = title_height + row * tile_height
        before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
        after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
        if before is None or after is None:
            raise RuntimeError(f"could not read comparison images for {name}")
        sheet[top : top + view_height, left : left + view_width] = cv2.resize(
            before, (view_width, view_height), interpolation=cv2.INTER_AREA
        )
        sheet[
            top : top + view_height, left + view_width : left + tile_width
        ] = cv2.resize(after, (view_width, view_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(sheet, (left + 8, top + 8), (left + 105, top + 35), (42, 42, 135), -1)
        cv2.putText(sheet, "BEFORE", (left + 17, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (left + view_width + 8, top + 8), (left + view_width + 118, top + 35), (69, 115, 31), -1)
        cv2.putText(sheet, "REPAIRED", (left + view_width + 16, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        placeholder = vertex_count <= 4
        if placeholder:
            cv2.rectangle(sheet, (left + 2, top + 2), (left + tile_width - 2, top + view_height - 2), (55, 55, 235), 6)
        label = f"{name}{' - SOURCE PLACEHOLDER' if placeholder else ''}"
        color = (90, 90, 255) if placeholder else (238, 238, 238)
        cv2.putText(sheet, label, (left + 8, top + view_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
        raise RuntimeError(f"failed to write {destination}")


def render_all(args: argparse.Namespace, app) -> Path:
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    originals = sorted(
        path for path in root.glob("*/*.usd") if not path.stem.endswith(".repaired")
    )
    requested_names = {name.strip() for name in args.names.split(",") if name.strip()}
    if requested_names:
        originals = [path for path in originals if path.stem in requested_names]
        missing = requested_names - {path.stem for path in originals}
        if missing:
            raise FileNotFoundError(f"unknown storefront names: {sorted(missing)}")
    if args.limit is not None:
        originals = originals[: args.limit]
    report = json.loads((root / "storefront_mesh_repair_report.json").read_text())
    vertex_counts = {
        Path(item["input"]).stem: sum(mesh["input_vertices"] for mesh in item.get("meshes", []))
        for item in report
    }

    stage = Usd.Stage.CreateInMemory("storefront_isaac_comparison.usda")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(float(args.dome_intensity))
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(args.sun_intensity))
    sun.CreateAngleAttr(0.5)
    headlight = UsdLux.SphereLight.Define(stage, "/World/Headlight")
    headlight.CreateIntensityAttr(float(args.headlight_intensity))
    headlight.CreateRadiusAttr(4.0)
    headlight_xform = UsdGeom.Xformable(headlight.GetPrim())
    headlight_xform.ClearXformOpOrder()
    headlight_translate = headlight_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

    camera_path = "/World/Camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(28.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_transform = camera_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

    success, error = asyncio.get_event_loop().run_until_complete(attach_stage(stage))
    if not success:
        raise RuntimeError(f"failed to attach stage: {error}")
    wait_for_stage_load(app)
    stage = omni.usd.get_context().get_stage()
    rep.settings.carb_settings("/rtx/sceneDb/ambientLightIntensity", 0.35)
    render_product = rep.create.render_product(camera_path, tuple(args.resolution), name="StorefrontComparison")
    annotator = rep.annotators.get("rgb")
    annotator.attach([render_product])

    pairs = []
    for index, original in enumerate(originals, start=1):
        name = original.stem
        repaired = original.with_name(f"{name}.repaired.usd")
        pair_root = f"/World/Assets/A{index:03d}_{safe_prim_name(name)}"
        before_asset = UsdGeom.Xform.Define(stage, f"{pair_root}/Before")
        repaired_asset = UsdGeom.Xform.Define(stage, f"{pair_root}/Repaired")
        before_asset.GetPrim().GetReferences().AddReference(str(original.resolve()))
        repaired_asset.GetPrim().GetReferences().AddReference(str(repaired.resolve()))
        wait_for_stage_load(app)
        for asset_prim in (before_asset.GetPrim(), repaired_asset.GetPrim()):
            for prim in Usd.PrimRange(asset_prim):
                if "Light" in prim.GetTypeName():
                    prim.SetActive(False)
        describe_asset(stage, before_asset.GetPrim(), original)
        describe_asset(stage, repaired_asset.GetPrim(), repaired)

        before_lower, before_upper = asset_bounds(stage, before_asset.GetPrim())
        repaired_lower, repaired_upper = asset_bounds(stage, repaired_asset.GetPrim())
        lower = np.minimum(before_lower, repaired_lower)
        upper = np.maximum(before_upper, repaired_upper)
        center = (lower + upper) * 0.5
        size = upper - lower
        major = max(float(size[0]), float(size[2]))
        eye = center + np.asarray([0.9 * major, 1.6 * major, 0.25 * size[2]])
        camera_transform.Set(camera_matrix(eye, center))
        headlight_translate.Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2] + 0.2)))

        pair_paths = []
        variants = (
            ("before", original, before_asset.GetPrim(), repaired_asset.GetPrim()),
            ("repaired", repaired, repaired_asset.GetPrim(), before_asset.GetPrim()),
        )
        for _label, _usd_path, visible_prim, hidden_prim in variants:
            UsdGeom.Imageable(visible_prim).MakeVisible()
            UsdGeom.Imageable(hidden_prim).MakeInvisible()
            for _ in range(3):
                render_rgb(app, annotator, args.warmup_frames)

        for label, usd_path, visible_prim, hidden_prim in variants:
            print(f"[{index}/{len(originals)}] {name}: {label}", flush=True)
            UsdGeom.Imageable(visible_prim).MakeVisible()
            UsdGeom.Imageable(hidden_prim).MakeInvisible()
            for _ in range(2):
                app.update()
            render_rgb(app, annotator, args.warmup_frames)
            rgb = render_rgb(app, annotator, args.warmup_frames)
            output_path = output_dir / f"{name}.{label}.png"
            if not cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"failed to write {output_path}")
            pair_paths.append(output_path)
        pairs.append((name, pair_paths[0], pair_paths[1], vertex_counts[name]))
        UsdGeom.Imageable(before_asset.GetPrim()).MakeInvisible()
        UsdGeom.Imageable(repaired_asset.GetPrim()).MakeInvisible()

    destination = output_dir / "storefront_isaac_before_after_contact_sheet.png"
    make_contact_sheet(pairs, destination, args.columns)
    return destination


def main() -> None:
    args = parse_args()
    print("[isaac-comparison] launching Isaac Sim", flush=True)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": args.renderer,
            "width": int(args.resolution[0]),
            "height": int(args.resolution[1]),
            "multi_gpu": True,
            "samples_per_pixel_per_frame": 1,
            "anti_aliasing": 0,
        }
    )
    try:
        destination = render_all(args, app)
        print(f"CONTACT_SHEET: {destination}", flush=True)
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        print("[isaac-comparison] closing Isaac Sim", flush=True)
        app.close()


if __name__ == "__main__":
    main()
