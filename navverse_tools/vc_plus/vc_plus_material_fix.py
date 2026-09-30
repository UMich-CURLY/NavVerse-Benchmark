"""Repair a vc_plus city USD's building materials.

Fixes the regressions that were observed in the composed BELGRADE_merged.usd output:

1. ``upAxis`` is forced to ``Z`` (the navverse sim/eval convention). Some refactors
   produced layers without an explicit upAxis, which silently defaults to Y and tips
   the scene on its side.
2. ``/World/store`` gets ``xformOp:scale = (0.01, 0.01, 0.01)`` if no xform ops were
   authored. The split-refactor that produced the small manifest dropped this scale,
   sending storefronts 100x further from the city.
3. Per-building textures are re-extracted from ``vc_full_raw/<CITY>/buildings/*.glb``
   and written to ``<city_dir>/textures_from_glb/<safe_glb_stem>.png``. This bypasses
   the Blender export's ``Material_element`` image-name collision and any downstream
   upscaling that changed the embedded photos.
4. Every ``element_NNN`` and ``Mesh_Data_NNN`` prim under ``/World/city/root`` is
   spatially matched to its nearest GLB; its bound UsdUVTexture inputs:file is
   overridden to the GLB-extracted PNG. The shader's metallic/roughness are set to
   1.0/1.0 to match the visual look of the other building materials.

The fix is applied as overrides in the supplied root layer (typically the city's
manifest USD like ``BELGRADE_merged.usd``), so the heavier ``<city>_buildings.usd``
data layer is left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# pxr is loaded lazily because Isaac/USD takes seconds to import.


def _glb_chunks(path: Path) -> dict[bytes, bytes]:
    with open(path, "rb") as f:
        f.read(12)
        chunks: dict[bytes, bytes] = {}
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            clen, ctype = struct.unpack("<I4s", head)
            chunks[ctype] = f.read(clen)
    return chunks


def glb_extract_image(path: Path) -> tuple[str, bytes] | tuple[None, None]:
    """Return (image_name, image_bytes) for the GLB's first image, or (None, None)."""
    chunks = _glb_chunks(path)
    if b"JSON" not in chunks:
        return None, None
    j = json.loads(chunks[b"JSON"].rstrip(b"\x00 \t\r\n"))
    if not j.get("images"):
        return None, None
    img = j["images"][0]
    bv = j["bufferViews"][img["bufferView"]]
    offset = bv.get("byteOffset", 0)
    blob = chunks.get(b"BIN\x00", b"")
    return img.get("name", "?"), blob[offset : offset + bv["byteLength"]]


def glb_centroid_world(path: Path) -> tuple[float, float, float] | None:
    """Return GLB centroid in USD world-frame coordinates (axis-swapped from glTF).

    Mapping observed in vc_full_raw: ``(usd_x, usd_y, usd_z) = (glb_x, -glb_z, glb_y)``.
    """
    chunks = _glb_chunks(path)
    if b"JSON" not in chunks:
        return None
    j = json.loads(chunks[b"JSON"].rstrip(b"\x00 \t\r\n"))
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    for mesh in j.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is None:
                continue
            acc = j["accessors"][pos]
            if "min" in acc and "max" in acc:
                for i in range(3):
                    mn[i] = min(mn[i], acc["min"][i])
                    mx[i] = max(mx[i], acc["max"][i])
    if mn[0] == float("inf"):
        return None
    cx = (mn[0] + mx[0]) / 2.0
    cy = (mn[1] + mx[1]) / 2.0
    cz = (mn[2] + mx[2]) / 2.0
    return (cx, -cz, cy)


def safe_stem(glb_filename: str) -> str:
    """USD/FS-safe stem from a GLB filename. Keeps Unicode word chars + dot/hyphen."""
    stem = Path(glb_filename).stem
    stem = re.sub(r"[^\w.-]", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem)
    return stem


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def extract_glb_textures(glb_dir: Path, out_dir: Path) -> dict[str, str]:
    """Extract first PNG image from each GLB into ``out_dir``. Return {glb_name: out_filename}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for glb in sorted(glb_dir.iterdir()):
        if glb.suffix.lower() != ".glb":
            continue
        name, data = glb_extract_image(glb)
        if not data:
            continue
        fname = safe_stem(glb.name) + ".png"
        (out_dir / fname).write_bytes(data)
        mapping[glb.name] = fname
    return mapping


STORE_SCALE_KEY = "navverse_store_child_scale_multiplier"


def apply_material_fix(
    root_usd: Path,
    glb_dir: Path,
    textures_subdir: str = "textures_from_glb",
    set_metallic: float = 1.0,
    set_roughness: float = 1.0,
    extract_textures: bool = True,
    *,
    set_up_axis: bool = True,
    add_store_scale: bool = True,
    store_child_scale_multiplier: float = 2.0,
    storefront_metallic: float = 0.0,
    storefront_roughness: float = 0.82,
) -> dict:
    """Open ``root_usd`` and apply all the material/shader/xform fixes.

    Returns a summary dict.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    root_usd = Path(root_usd).resolve()
    glb_dir = Path(glb_dir).resolve()
    city_dir = root_usd.parent
    tex_out_dir = city_dir / textures_subdir

    summary = {
        "root_usd": str(root_usd),
        "glb_dir": str(glb_dir),
        "textures_subdir": textures_subdir,
        "glbs_seen": 0,
        "textures_extracted": 0,
        "up_axis_set": False,
        "store_scale_set": False,
        "store_children_rescaled": 0,
        "store_child_scale_multiplier": store_child_scale_multiplier,
        "storefront_shader_overrides": 0,
        "element_overrides": 0,
        "mesh_data_overrides": 0,
        "unmatched_meshes": [],
    }

    # 1) Extract embedded GLB images to <city_dir>/<textures_subdir>/.
    glb_centroids: dict[str, tuple[float, float, float]] = {}
    glb_to_filename: dict[str, str] = {}
    for glb in sorted(glb_dir.iterdir()):
        if glb.suffix.lower() != ".glb":
            continue
        summary["glbs_seen"] += 1
        c = glb_centroid_world(glb)
        if c is None:
            continue
        glb_centroids[glb.name] = c
        glb_to_filename[glb.name] = safe_stem(glb.name) + ".png"

    if extract_textures:
        tex_out_dir.mkdir(parents=True, exist_ok=True)
        for glb in sorted(glb_dir.iterdir()):
            if glb.suffix.lower() != ".glb":
                continue
            name, data = glb_extract_image(glb)
            if not data:
                continue
            out = tex_out_dir / glb_to_filename[glb.name]
            out.write_bytes(data)
            summary["textures_extracted"] += 1

    # 2) Open the USD stage for editing.
    stage = Usd.Stage.Open(str(root_usd), Usd.Stage.LoadAll)
    if not stage:
        raise RuntimeError(f"failed to open {root_usd}")

    # 2a) Force upAxis = Z.
    if set_up_axis and UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        summary["up_axis_set"] = True

    # 2b) /World/store xformOp:scale fallback.
    store = stage.GetPrimAtPath("/World/store")
    if add_store_scale and store and store.IsValid():
        xf = UsdGeom.Xformable(store)
        ops = xf.GetOrderedXformOps()
        has_scale = any(op.GetOpType() == UsdGeom.XformOp.TypeScale for op in ops)
        if not has_scale:
            xf.ClearXformOpOrder()
            from pxr import Gf as _Gf
            xf.AddScaleOp().Set(_Gf.Vec3f(0.01, 0.01, 0.01))
            summary["store_scale_set"] = True

    # 2c) Storefront per-child scale.
    # The current storefront/<cat>/<cat>.usd assets are ~50% the size of the apr14
    # storefronts that the building holes were cut for. Multiply each child's
    # xformOp:scale by ~2x so the storefront fills the hole. Stored as customData
    # on /World/store so subsequent fix runs don't double-apply.
    if store and store.IsValid() and store_child_scale_multiplier and store_child_scale_multiplier != 1.0:
        already = store.GetCustomDataByKey(STORE_SCALE_KEY)
        if not already:
            from pxr import Gf as _Gf
            for child in store.GetChildren():
                cxf = UsdGeom.Xformable(child)
                scale_op = next(
                    (op for op in cxf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale),
                    None,
                )
                if not scale_op:
                    continue
                cur = scale_op.Get()
                k = float(store_child_scale_multiplier)
                scale_op.Set(_Gf.Vec3f(cur[0] * k, cur[1] * k, cur[2] * k))
                summary["store_children_rescaled"] += 1
            store.SetCustomDataByKey(STORE_SCALE_KEY, float(store_child_scale_multiplier))

    # 2d) Storefront shader params. The source storefront/<cat>/<cat>.usd materials
    # use roughness=0.5, which can look too glossy next to reconstructed city
    # textures. Keep storefronts non-metallic and make them a little rougher.
    if store and store.IsValid() and (storefront_metallic is not None or storefront_roughness is not None):
        for child in store.GetChildren():
            for d in Usd.PrimRange(child):
                if not d.IsA(UsdShade.Shader):
                    continue
                sh = UsdShade.Shader(d)
                if sh.GetIdAttr().Get() != "UsdPreviewSurface":
                    continue
                if storefront_metallic is not None:
                    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(storefront_metallic))
                if storefront_roughness is not None:
                    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(storefront_roughness))
                summary["storefront_shader_overrides"] += 1

    # 3) Bbox cache + helpers for traversal.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def find_bound_material(prim):
        for sub in [prim] + list(prim.GetChildren()):
            mb = UsdShade.MaterialBindingAPI(sub).GetDirectBinding()
            if mb.GetMaterial():
                return mb.GetMaterial()
        return None

    def set_texture_and_shader(mat, asset_path: str):
        """Override UsdUVTexture inputs:file + UsdPreviewSurface metallic/roughness."""
        for c in mat.GetPrim().GetChildren():
            if not c.IsA(UsdShade.Shader):
                continue
            sh = UsdShade.Shader(c)
            sid = sh.GetIdAttr().Get()
            if sid == "UsdUVTexture":
                f = sh.GetInput("file")
                if f:
                    f.Set(Sdf.AssetPath(asset_path))
            elif sid == "UsdPreviewSurface":
                if set_metallic is not None:
                    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(set_metallic))
                if set_roughness is not None:
                    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(set_roughness))

    # 4) Re-bind every element_NNN and Mesh_Data_NNN under /World/city/root via spatial match.
    root = stage.GetPrimAtPath("/World/city/root")
    if not root:
        return summary

    for prim in root.GetChildren():
        name = prim.GetName()
        is_element = name.startswith("element_")
        is_mesh_data = name.startswith("Mesh_Data_")
        if not (is_element or is_mesh_data):
            continue
        bb = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn, mx = bb.GetMin(), bb.GetMax()
        ctr = ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)
        if not glb_centroids:
            continue
        nearest = min(glb_centroids.items(), key=lambda kv: _dist2(ctr, kv[1]))
        d2 = _dist2(ctr, nearest[1])
        if d2 > 25.0:  # 5m threshold
            summary["unmatched_meshes"].append({"prim": str(prim.GetPath()), "dist_m": d2 ** 0.5})
            continue
        mat = find_bound_material(prim)
        if not mat:
            continue
        new_asset = f"./{textures_subdir}/{glb_to_filename[nearest[0]]}"
        set_texture_and_shader(mat, new_asset)
        if is_element:
            summary["element_overrides"] += 1
        else:
            summary["mesh_data_overrides"] += 1

    stage.GetRootLayer().Save()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Repair a vc_plus city USD's building materials.")
    parser.add_argument("--root_usd", required=True, help="path to <CITY>_merged.usd or <CITY>_outdoor.usd to fix")
    parser.add_argument("--glb_dir", required=True, help="vc_full_raw/<CITY>/buildings directory")
    parser.add_argument("--textures_subdir", default="textures_from_glb")
    parser.add_argument("--metallic", type=float, default=1.0)
    parser.add_argument("--roughness", type=float, default=1.0)
    parser.add_argument("--no_extract", action="store_true", help="skip texture extraction (textures must already exist)")
    parser.add_argument("--no_up_axis", action="store_true")
    parser.add_argument("--no_store_scale", action="store_true")
    parser.add_argument("--store_child_scale", type=float, default=2.0,
                        help="multiplier applied to each /World/store child's xformOp:scale (default 2.0)")
    parser.add_argument("--storefront_metallic", type=float, default=0.0,
                        help="metallic override on storefront UsdPreviewSurface shaders (default 0.0)")
    parser.add_argument("--storefront_roughness", type=float, default=0.82,
                        help="roughness override on storefront UsdPreviewSurface shaders (default 0.82)")
    args = parser.parse_args()
    s = apply_material_fix(
        Path(args.root_usd),
        Path(args.glb_dir),
        textures_subdir=args.textures_subdir,
        set_metallic=args.metallic,
        set_roughness=args.roughness,
        extract_textures=not args.no_extract,
        set_up_axis=not args.no_up_axis,
        add_store_scale=not args.no_store_scale,
        store_child_scale_multiplier=args.store_child_scale,
        storefront_metallic=args.storefront_metallic,
        storefront_roughness=args.storefront_roughness,
    )
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
