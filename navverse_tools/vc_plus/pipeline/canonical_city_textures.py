#!/usr/bin/env python3
"""Synchronize canonical DDS textures for the VC city mesh."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from pxr import Sdf, Usd


CITY_ROOT = "/World/city"
RAW_MATERIAL_ROOT = "/_materials"


@dataclass(frozen=True)
class CanonicalTexture:
    asset_path: str
    source_path: Path


def _city_asset_attributes(stage: Usd.Stage):
    for prim in stage.TraverseAll():
        if prim.GetTypeName() != "Shader":
            continue
        prim_path = str(prim.GetPath())
        if prim_path.startswith(CITY_ROOT):
            canonical_path = prim_path
        elif prim_path.startswith(RAW_MATERIAL_ROOT):
            canonical_path = CITY_ROOT + prim_path
        else:
            continue
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                yield (canonical_path, attribute.GetName()), attribute, value


def load_canonical_texture_map(scene_usd: Path) -> dict[tuple[str, str], CanonicalTexture]:
    scene_usd = scene_usd.resolve()
    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"could not open canonical scene: {scene_usd}")

    texture_map: dict[tuple[str, str], CanonicalTexture] = {}
    for key, _attribute, value in _city_asset_attributes(stage):
        source_path = Path(value.resolvedPath) if value.resolvedPath else scene_usd.parent / value.path
        source_path = source_path.resolve()
        if source_path.suffix.lower() != ".dds":
            raise RuntimeError(
                f"canonical city texture is not DDS: {key[0]} {key[1]}={value.path}"
            )
        if not source_path.exists():
            raise FileNotFoundError(
                f"canonical city texture is missing: {key[0]} {key[1]}={source_path}"
            )
        texture_map[key] = CanonicalTexture(value.path, source_path)

    if not texture_map:
        raise RuntimeError(f"canonical scene has no {CITY_ROOT} texture inputs: {scene_usd}")
    return texture_map


def inspect_target_texture_refs(
    scene_usd: Path,
    texture_map: dict[tuple[str, str], CanonicalTexture],
) -> dict[str, object]:
    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"could not open target scene: {scene_usd}")

    refs = list(_city_asset_attributes(stage))
    unmapped = [f"{key[0]}:{key[1]}" for key, _attribute, _value in refs if key not in texture_map]
    return {
        "scene": str(scene_usd),
        "texture_refs": len(refs),
        "unmapped": unmapped,
    }


def copy_canonical_texture_files(
    texture_map: dict[tuple[str, str], CanonicalTexture],
    destination_dir: Path,
) -> dict[str, object]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted({texture.source_path for texture in texture_map.values()})
    copied = 0
    skipped = 0
    copied_bytes = 0
    for source in sources:
        destination = destination_dir / source.name
        source_stat = source.stat()
        if destination.exists():
            destination_stat = destination.stat()
            if (
                destination_stat.st_size == source_stat.st_size
                and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
            ):
                skipped += 1
                continue
        shutil.copy2(source, destination)
        copied += 1
        copied_bytes += source_stat.st_size
    return {
        "files": len(sources),
        "copied": copied,
        "skipped": skipped,
        "copied_bytes": copied_bytes,
    }


def rewrite_city_texture_refs(
    scene_usd: Path,
    texture_map: dict[tuple[str, str], CanonicalTexture],
    destination_dir: Path,
) -> dict[str, object]:
    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"could not open target scene: {scene_usd}")

    refs = list(_city_asset_attributes(stage))
    unmapped = [key for key, _attribute, _value in refs if key not in texture_map]
    if unmapped:
        sample = ", ".join(f"{path}:{name}" for path, name in unmapped[:8])
        raise RuntimeError(f"{scene_usd} has {len(unmapped)} unmapped city texture refs: {sample}")

    rewritten = 0
    unchanged = 0
    for key, attribute, value in refs:
        canonical = texture_map[key]
        destination = destination_dir / canonical.source_path.name
        if not destination.exists():
            raise FileNotFoundError(f"target city DDS is missing: {destination}")
        target_asset = str(Path("textures") / canonical.source_path.name)
        if value.path == target_asset or value.path == f"./{target_asset}":
            unchanged += 1
            continue
        attribute.Set(Sdf.AssetPath(target_asset))
        rewritten += 1

    stage.GetRootLayer().Save()
    return {
        "scene": str(scene_usd),
        "texture_refs": len(refs),
        "rewritten": rewritten,
        "unchanged": unchanged,
    }
