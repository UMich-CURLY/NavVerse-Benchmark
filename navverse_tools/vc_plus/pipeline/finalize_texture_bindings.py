#!/usr/bin/env python3
"""Convert effective image bindings to DDS after final USD composition."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pxr import Ar, Sdf, Usd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tga", ".bmp"}
NORMAL_HINTS = ("normal", "norm", "bump")
SCALAR_HINTS = ("rough", "metal", "ao", "occlusion", "height", "disp", "opacity", "alpha")


@dataclass(frozen=True)
class Binding:
    stage: str
    attribute: str
    layer: str
    spec_path: str
    asset: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, nargs="+", required=True, help="Final USD entrypoints to process.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--compressonator", default="compressonatorcli")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Regenerate DDS files even when a valid output exists.")
    parser.add_argument("--audit-only", action="store_true", help="Scan bindings without converting or editing layers.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing source images or conversion errors.")
    parser.add_argument("--lock-dir", type=Path, default=Path(tempfile.gettempdir()) / "navverse_texture_locks")
    return parser.parse_args()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def classify_texture(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if any(hint in name for hint in NORMAL_HINTS):
        return "normal", "BC5"
    if any(hint in name for hint in SCALAR_HINTS):
        return "scalar", "BC4"
    return "color", "BC7"


def output_path(source: Path, texture_format: str) -> Path:
    return source.with_suffix(f".{texture_format.lower()}.dds")


def lock_path(lock_dir: Path, namespace: str, value: str) -> Path:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return lock_dir / namespace / f"{digest}.lock"


def resolve_authored_asset(layer: Sdf.Layer, asset_path: str) -> Path:
    identifier = Sdf.ComputeAssetPathRelativeToLayer(layer, asset_path)
    resolved = Ar.GetResolver().Resolve(identifier)
    return Path(str(resolved or identifier))


def effective_binding(stage_path: Path, attr: Usd.Attribute) -> Binding | None:
    value = attr.Get()
    if not isinstance(value, Sdf.AssetPath) or Path(value.path).suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    for spec in attr.GetPropertyStack(Usd.TimeCode.Default()):
        authored = spec.default
        if not isinstance(authored, Sdf.AssetPath):
            continue
        if Path(authored.path).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        layer_path = spec.layer.realPath or spec.layer.identifier
        if not layer_path:
            continue
        source = resolve_authored_asset(spec.layer, authored.path)
        return Binding(
            stage=str(stage_path),
            attribute=str(attr.GetPath()),
            layer=str(Path(layer_path).resolve()),
            spec_path=str(spec.path),
            asset=authored.path,
            source=str(source),
        )
    return None


def scan_stages(stage_paths: list[Path]) -> tuple[list[Binding], list[dict[str, str]]]:
    bindings: dict[tuple[str, str], Binding] = {}
    stage_errors: list[dict[str, str]] = []
    for stage_path in stage_paths:
        stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
        if stage is None:
            stage_errors.append({"stage": str(stage_path), "error": "failed to open USD stage"})
            continue
        for prim in stage.TraverseAll():
            for attr in prim.GetAttributes():
                binding = effective_binding(stage_path, attr)
                if binding is not None:
                    bindings.setdefault((binding.layer, binding.spec_path), binding)
    return list(bindings.values()), stage_errors


def pad_to_block_size(image: Image.Image, block: int = 4) -> tuple[Image.Image, bool]:
    width, height = image.size
    padded_width = (width + block - 1) // block * block
    padded_height = (height + block - 1) // block * block
    if (padded_width, padded_height) == (width, height):
        return image, False

    padded = Image.new(image.mode, (padded_width, padded_height))
    padded.paste(image, (0, 0))
    if padded_height > height:
        bottom = image.crop((0, height - 1, width, height))
        for y in range(height, padded_height):
            padded.paste(bottom, (0, y))
    if padded_width > width:
        right = padded.crop((width - 1, 0, width, padded_height))
        for x in range(width, padded_width):
            padded.paste(right, (x, 0))
    return padded, True


def valid_dds(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 128:
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"DDS "
    except OSError:
        return False


def compress_command(executable: str, source: Path, destination: Path, texture_format: str) -> list[str]:
    resolved = shutil.which(executable) or executable
    executable_path = Path(resolved)
    prefix = b""
    if executable_path.is_file():
        try:
            prefix = executable_path.read_bytes()[:4]
        except OSError:
            pass
    command = ["bash", str(executable_path)] if prefix and not (
        prefix.startswith(b"#!") or prefix == b"\x7fELF"
    ) else [str(executable_path)]
    return [*command, "-fd", texture_format, "-nomipmap", str(source), str(destination)]


def convert_texture(
    source_text: str,
    compressonator: str,
    force: bool,
    lock_dir_text: str,
) -> dict[str, Any]:
    source = Path(source_text)
    kind, texture_format = classify_texture(source)
    destination = output_path(source, texture_format)
    record: dict[str, Any] = {
        "source": str(source),
        "output": str(destination),
        "kind": kind,
        "format": texture_format,
    }
    if not source.is_file():
        record["status"] = "missing_source"
        return record

    record["source_bytes"] = source.stat().st_size
    lock_file = lock_path(Path(lock_dir_text), "outputs", str(destination.resolve()))
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("w", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        should_convert = force or not valid_dds(destination)
        if not should_convert and source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
            should_convert = True
        if not should_convert:
            record.update({"status": "existing", "output_bytes": destination.stat().st_size})
            return record

        temporary_output = destination.with_name(
            f".{destination.stem}.tmp.{os.getpid()}.{time.time_ns()}{destination.suffix}"
        )
        padded_source: Path | None = None
        try:
            with Image.open(source) as image:
                width, height = image.size
                padded, was_padded = pad_to_block_size(image)
                record.update(
                    {
                        "width": width,
                        "height": height,
                        "padded_width": padded.width,
                        "padded_height": padded.height,
                    }
                )
                input_path = source
                if was_padded:
                    with tempfile.NamedTemporaryFile(
                        prefix="navverse_bc_pad_",
                        suffix=".png",
                        delete=False,
                    ) as temporary:
                        padded_source = Path(temporary.name)
                    padded.save(padded_source)
                    input_path = padded_source
            result = subprocess.run(
                compress_command(compressonator, input_path, temporary_output, texture_format),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
            if result.returncode != 0 or not valid_dds(temporary_output):
                record.update(
                    {
                        "status": "failed",
                        "returncode": result.returncode,
                        "error": result.stderr[-2000:],
                    }
                )
                return record
            os.replace(temporary_output, destination)
            record.update({"status": "converted", "output_bytes": destination.stat().st_size})
            return record
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            record.update({"status": "failed", "error": repr(exc)})
            return record
        finally:
            temporary_output.unlink(missing_ok=True)
            if padded_source is not None:
                padded_source.unlink(missing_ok=True)


def replace_binding(
    layer: Sdf.Layer,
    binding: Binding,
    conversions: dict[str, dict[str, Any]],
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    spec = layer.GetAttributeAtPath(Sdf.Path(binding.spec_path))
    if spec is None:
        return None, {
            "layer": binding.layer,
            "spec_path": binding.spec_path,
            "error": "attribute spec missing",
        }
    current = spec.default
    if not isinstance(current, Sdf.AssetPath):
        return None, None
    if Path(current.path).suffix.lower() not in IMAGE_EXTENSIONS:
        return None, None

    current_source = resolve_authored_asset(layer, current.path)
    conversion = conversions.get(os.path.realpath(current_source))
    if conversion is None or conversion.get("status") not in {"converted", "existing"}:
        return None, None
    new_asset = str(Path(current.path).with_suffix(f".{conversion['format'].lower()}.dds"))
    spec.default = Sdf.AssetPath(new_asset)
    return {
        "layer": binding.layer,
        "spec_path": binding.spec_path,
        "old": current.path,
        "new": new_asset,
    }, None


def edit_binding_layers(
    bindings: list[Binding],
    conversions: dict[str, dict[str, Any]],
    lock_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[Binding]] = defaultdict(list)
    for binding in bindings:
        grouped[binding.layer].append(binding)

    replacements: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for layer_path, layer_bindings in sorted(grouped.items()):
        layer_lock = lock_path(lock_dir, "layers", str(Path(layer_path).resolve()))
        layer_lock.parent.mkdir(parents=True, exist_ok=True)
        with layer_lock.open("w", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            layer = Sdf.Layer.FindOrOpen(layer_path)
            if layer is None:
                errors.append({"layer": layer_path, "error": "failed to open layer"})
                continue
            layer.Reload()
            changed = False
            for binding in layer_bindings:
                replacement, error = replace_binding(layer, binding, conversions)
                if error is not None:
                    errors.append(error)
                if replacement is not None:
                    replacements.append(replacement)
                    changed = True
            if changed and not layer.Save():
                errors.append({"layer": layer_path, "error": "failed to save layer"})
    return replacements, errors


def main() -> int:
    args = parse_args()
    stage_paths = [path.expanduser().resolve() for path in args.usd]
    missing_stages = [str(path) for path in stage_paths if not path.is_file()]
    existing_stages = [path for path in stage_paths if path.is_file()]
    bindings, stage_errors = scan_stages(existing_stages)

    unique_sources = sorted({os.path.realpath(binding.source) for binding in bindings})
    conversions: list[dict[str, Any]] = []
    if args.audit_only:
        for source_text in unique_sources:
            source = Path(source_text)
            _, texture_format = classify_texture(source)
            destination = output_path(source, texture_format)
            conversions.append(
                {
                    "source": str(source),
                    "output": str(destination),
                    "status": "audit",
                    "source_exists": source.is_file(),
                    "output_exists": valid_dds(destination),
                }
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    convert_texture,
                    source_text,
                    args.compressonator,
                    args.force,
                    str(args.lock_dir),
                )
                for source_text in unique_sources
            ]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                conversions.append(future.result())
                if index % 50 == 0 or index == len(futures):
                    print(f"[finalize-textures] converted {index}/{len(futures)} unique sources", flush=True)

    conversion_map = {os.path.realpath(record["source"]): record for record in conversions}
    replacements: list[dict[str, str]] = []
    layer_errors: list[dict[str, str]] = []
    if not args.audit_only:
        replacements, layer_errors = edit_binding_layers(bindings, conversion_map, args.lock_dir)

    status_counts: dict[str, int] = defaultdict(int)
    for record in conversions:
        status_counts[record["status"]] += 1
    result = {
        "stages": [str(path) for path in stage_paths],
        "stage_count": len(existing_stages),
        "missing_stages": missing_stages,
        "binding_count": len(bindings),
        "layer_count": len({binding.layer for binding in bindings}),
        "unique_source_count": len(unique_sources),
        "status_counts": dict(sorted(status_counts.items())),
        "replacement_count": len(replacements),
        "stage_errors": stage_errors,
        "layer_errors": layer_errors,
        "bindings": [asdict(binding) for binding in bindings],
        "conversions": sorted(conversions, key=lambda record: record["source"]),
        "replacements": replacements,
    }
    atomic_write_json(args.output_json, result)
    print(
        "[finalize-textures] "
        f"stages={result['stage_count']} bindings={result['binding_count']} "
        f"sources={result['unique_source_count']} replacements={result['replacement_count']} "
        f"statuses={result['status_counts']}",
        flush=True,
    )

    failures = (
        len(missing_stages)
        + len(stage_errors)
        + len(layer_errors)
        + status_counts.get("failed", 0)
        + status_counts.get("missing_source", 0)
    )
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
