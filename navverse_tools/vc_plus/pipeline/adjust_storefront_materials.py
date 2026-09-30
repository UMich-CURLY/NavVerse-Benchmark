#!/usr/bin/env python3
"""Apply scene-local storefront albedo and shader normalization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Sdf, Usd, UsdShade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize storefront materials in a generated VC USD.")
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--target-mean", type=float, default=92.0)
    parser.add_argument("--target-p90", type=float, default=132.0)
    parser.add_argument("--min-factor", type=float, default=0.34)
    parser.add_argument("--roughness", type=float, default=0.88)
    parser.add_argument("--specular", type=float, default=0.08)
    return parser.parse_args()


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def texture_stats(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("RGB").resize((128, 128))
    rgb = np.asarray(image, dtype=np.float32)
    lum = luminance(rgb)
    return {
        "mean": float(lum.mean()),
        "p90": float(np.percentile(lum, 90.0)),
        "white_ratio": float((lum > 235.0).mean()),
    }


def exposure_factor(stats: dict[str, float], target_mean: float, target_p90: float, min_factor: float) -> float:
    factor = 1.0
    if stats["mean"] > target_mean:
        factor = min(factor, target_mean / max(stats["mean"], 1e-6))
    if stats["p90"] > target_p90:
        factor = min(factor, target_p90 / max(stats["p90"], 1e-6))
    return float(np.clip(factor, min_factor, 1.0))


def write_adjusted_texture(src: Path, dst: Path, factor: float) -> dict[str, float]:
    before = texture_stats(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    source_image = Image.open(src)
    image = source_image.convert("RGBA" if source_image.mode in {"RGBA", "LA"} else "RGB")
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim == 3:
        arr[..., :3] *= factor
    else:
        arr *= factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(dst)
    after = texture_stats(dst)
    return {f"source_{key}": value for key, value in before.items()} | {f"adjusted_{key}": value for key, value in after.items()}


def resolve_store_texture(store_root: Path, asset_path: Sdf.AssetPath | None, asset_name: str) -> Path | None:
    if not asset_path:
        return None
    candidates: list[Path] = []
    raw = asset_path.path
    if raw:
        raw_path = Path(raw)
        candidates.append(store_root / asset_name / raw_path.name)
        candidates.append(store_root / asset_name / "textures" / raw_path.name)
        candidates.extend(sorted(store_root.glob(f"*/textures/{raw_path.name}")))
        if raw_path.is_absolute() and "storefront_albedo_adjusted" not in raw_path.parts:
            candidates.append(raw_path)
    if asset_path.resolvedPath:
        resolved = Path(asset_path.resolvedPath)
        if "storefront_albedo_adjusted" not in resolved.parts:
            candidates.append(resolved)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def asset_name_for_store_child(child: Usd.Prim) -> str:
    name = child.GetName()
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name


def relative_asset_path(path: Path, layer_dir: Path) -> str:
    try:
        return os.path.relpath(path, layer_dir)
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    usd_path = args.usd.resolve()
    store_root = args.store_root.resolve()
    output_dir = (args.output_dir or (usd_path.parent / "storefront_albedo_adjusted")).resolve()
    summary_path = (args.summary or (usd_path.parent / f"{usd_path.stem}_storefront_material_adjust.json")).resolve()

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open USD: {usd_path}")
    stage.SetEditTarget(stage.GetRootLayer())
    layer_dir = usd_path.parent

    records = []
    texture_cache: dict[tuple[Path, float], Path] = {}
    store = stage.GetPrimAtPath("/World/store")
    if not store or not store.IsValid():
        raise RuntimeError(f"{usd_path} has no /World/store prim")

    for child in store.GetChildren():
        asset_name = asset_name_for_store_child(child)
        for prim in Usd.PrimRange(child):
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            shader_id = shader.GetIdAttr().Get()
            if shader_id == "UsdPreviewSurface":
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(args.roughness))
                shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(float(args.specular))
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
                continue
            if shader_id != "UsdUVTexture":
                continue
            file_input = shader.GetInput("file")
            src = resolve_store_texture(store_root, file_input.Get() if file_input else None, asset_name)
            if src is None:
                records.append({"prim": prim.GetPath().pathString, "asset_name": asset_name, "status": "missing_texture"})
                continue
            stats = texture_stats(src)
            factor = exposure_factor(stats, args.target_mean, args.target_p90, args.min_factor)
            dst = output_dir / asset_name / src.name
            cache_key = (src, factor)
            if cache_key not in texture_cache:
                write_stats = write_adjusted_texture(src, dst, factor)
                texture_cache[cache_key] = dst
            else:
                write_stats = {}
            file_input.Set(Sdf.AssetPath(relative_asset_path(dst, layer_dir)))
            shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            records.append(
                {
                    "prim": prim.GetPath().pathString,
                    "asset_name": asset_name,
                    "status": "adjusted",
                    "source_texture": str(src),
                    "adjusted_texture": str(dst),
                    "factor": factor,
                    **stats,
                    **write_stats,
                }
            )

    stage.GetRootLayer().Save()
    summary_path.write_text(json.dumps({"usd": str(usd_path), "records": records}, indent=2, sort_keys=True) + "\n")
    adjusted = sum(1 for row in records if row["status"] == "adjusted")
    print(f"[storefront-materials] adjusted {adjusted} texture bindings in {usd_path}")
    print(f"[storefront-materials] wrote {summary_path}")


if __name__ == "__main__":
    main()
