#!/usr/bin/env python3
"""Replace low-resolution VC city texture refs with canonical upscaled DDS files."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from canonical_city_textures import (
    inspect_target_texture_refs,
    load_canonical_texture_map,
    rewrite_city_texture_refs,
)


LAYER_PATTERNS = (
    "{city}.usd",
    "{city}_with_road_objects.usd",
    "{city}_outdoor.usd",
    "{city}_innout.usd",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--cities", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def discover_cities(args: argparse.Namespace) -> list[str]:
    if args.cities:
        return sorted(set(args.cities))
    return sorted(
        path.name
        for path in args.flat_root.iterdir()
        if path.is_dir() and (path / f"{path.name}_outdoor.usd").exists()
    )


def copy_one(source: Path, destination: Path) -> tuple[str, int]:
    source_stat = source.stat()
    if destination.exists():
        destination_stat = destination.stat()
        if (
            destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        ):
            return "skipped", 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return "copied", source_stat.st_size


def build_plans(args: argparse.Namespace, cities: list[str]):
    plans = []
    copy_jobs: dict[Path, Path] = {}
    for city in cities:
        city_dir = args.flat_root / city
        canonical_scene = args.canonical_root / city / f"{city}_outdoor.usd"
        texture_map = load_canonical_texture_map(canonical_scene)
        layers = [city_dir / pattern.format(city=city) for pattern in LAYER_PATTERNS]
        layers = [path for path in layers if path.exists()]
        if not layers:
            raise FileNotFoundError(f"no target USD layers found for {city}")
        inspections = [inspect_target_texture_refs(path, texture_map) for path in layers]
        unmapped = [item for inspection in inspections for item in inspection["unmapped"]]
        if unmapped:
            raise RuntimeError(f"{city} has {len(unmapped)} unmapped city texture refs")
        destination_dir = city_dir / "textures"
        for texture in texture_map.values():
            copy_jobs[destination_dir / texture.source_path.name] = texture.source_path
        plans.append(
            {
                "city": city,
                "canonical_scene": str(canonical_scene),
                "canonical_refs": len(texture_map),
                "canonical_files": len({item.source_path for item in texture_map.values()}),
                "layers": layers,
                "inspections": inspections,
                "texture_map": texture_map,
            }
        )
        print(
            f"[preflight {len(plans):02d}/{len(cities)}] {city}: "
            f"refs={len(texture_map)} files={plans[-1]['canonical_files']} layers={len(layers)}",
            flush=True,
        )
    return plans, copy_jobs


def backup_layers(plans: list[dict[str, object]], backup_root: Path) -> None:
    for plan in plans:
        city_backup = backup_root / str(plan["city"])
        city_backup.mkdir(parents=True, exist_ok=True)
        for layer in plan["layers"]:
            shutil.copy2(layer, city_backup / layer.name)


def copy_textures(copy_jobs: dict[Path, Path], workers: int) -> tuple[int, int, int]:
    copied = skipped = copied_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(copy_one, source, destination): destination
            for destination, source in copy_jobs.items()
        }
        for index, future in enumerate(as_completed(futures), 1):
            status, byte_count = future.result()
            copied += status == "copied"
            skipped += status == "skipped"
            copied_bytes += byte_count
            if index % 500 == 0 or index == len(futures):
                print(
                    f"[textures {index}/{len(futures)}] copied={copied} skipped={skipped} "
                    f"bytes={copied_bytes}",
                    flush=True,
                )
    return copied, skipped, copied_bytes


def rewrite_layers(
    plans: list[dict[str, object]],
    flat_root: Path,
) -> list[dict[str, object]]:
    results = []
    for index, plan in enumerate(plans, 1):
        destination_dir = flat_root / str(plan["city"]) / "textures"
        layer_results = [
            rewrite_city_texture_refs(layer, plan["texture_map"], destination_dir)
            for layer in plan["layers"]
        ]
        results.append(
            {
                "city": plan["city"],
                "canonical_refs": plan["canonical_refs"],
                "canonical_files": plan["canonical_files"],
                "layers": layer_results,
            }
        )
        print(f"[rewrite {index:02d}/{len(plans)}] {plan['city']}", flush=True)
    return results


def main() -> None:
    args = parse_args()
    args.flat_root = args.flat_root.resolve()
    args.canonical_root = args.canonical_root.resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = (args.backup_root or args.flat_root.parent / f"{args.flat_root.name}_backup_pre_city_dds_{timestamp}").resolve()
    manifest_path = (args.manifest or args.flat_root / f"city_texture_repair_{timestamp}.json").resolve()
    cities = discover_cities(args)
    plans, copy_jobs = build_plans(args, cities)

    manifest = {
        "flat_root": str(args.flat_root),
        "canonical_root": str(args.canonical_root),
        "cities": cities,
        "apply": bool(args.apply),
        "backup_root": str(backup_root) if args.apply else None,
        "copy_files": len(copy_jobs),
        "copy_bytes_planned": sum(source.stat().st_size for source in copy_jobs.values()),
        "city_results": [],
    }
    if not args.apply:
        print(json.dumps(manifest, indent=2), flush=True)
        return

    backup_layers(plans, backup_root)
    copied, skipped, copied_bytes = copy_textures(copy_jobs, args.workers)
    manifest["city_results"] = rewrite_layers(plans, args.flat_root)

    manifest.update(
        {
            "copied": copied,
            "skipped": skipped,
            "copied_bytes": copied_bytes,
            "finished_at": time.strftime("%F %T %Z"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "copied": copied, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
