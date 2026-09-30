#!/usr/bin/env python3
"""Make the VC+ dataset portable with one sibling GRScenes directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdUtils


USD_SUFFIXES = {".usd", ".usda", ".usdc"}
URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


@dataclass(frozen=True)
class AssetMapping:
    name: str
    source: Path
    destination: Path
    copy_children: tuple[str, ...] | None = None


@dataclass
class AuthoredAssetAudit:
    categories: Counter[str] = field(default_factory=Counter)
    stale_source_paths: list[dict[str, str]] = field(default_factory=list)
    unexpected_external_paths: list[dict[str, str]] = field(default_factory=list)
    missing_paths: list[dict[str, str]] = field(default_factory=list)
    absolute_authored_paths: list[dict[str, str]] = field(default_factory=list)

    def inspect(
        self,
        raw_path: str,
        *,
        layer: Sdf.Layer,
        layer_path: Path,
        layer_label: str,
        flat_root: Path,
        grscene_root: Path,
        mappings: list[AssetMapping],
    ) -> tuple[Path | None, str | None]:
        if not raw_path:
            return None, None
        if is_runtime_asset(raw_path):
            self.categories["runtime_or_uri"] += 1
            return None, "runtime_or_uri"
        if Path(raw_path).is_absolute():
            self.absolute_authored_paths.append(
                {"layer": layer_label, "asset": raw_path}
            )
        resolved = resolve_asset_path(layer, layer_path, raw_path)
        entry = {
            "layer": layer_label,
            "asset": raw_path,
            "resolved": str(resolved),
        }
        category = classify_asset_path(resolved, flat_root, grscene_root, mappings)
        self.categories[category] += 1
        if category == "stale_source":
            self.stale_source_paths.append(entry)
        elif category == "unexpected_external":
            self.unexpected_external_paths.append(entry)
        if not resolved.exists():
            self.categories["missing"] += 1
            self.missing_paths.append(entry)
        return resolved, category

    def result(self, *, layers_scanned: int) -> dict[str, Any]:
        return {
            "layers_scanned": layers_scanned,
            "asset_paths_by_category": dict(sorted(self.categories.items())),
            "stale_source_paths": self.stale_source_paths,
            "unexpected_external_paths": self.unexpected_external_paths,
            "missing_paths": self.missing_paths,
            "absolute_authored_paths": self.absolute_authored_paths,
        }


@dataclass
class EffectiveAssetAudit:
    categories: Counter[str] = field(default_factory=Counter)
    problems: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    failed_scenes: list[str] = field(default_factory=list)

    def record_problem(
        self,
        kind: str,
        scene: Path,
        flat_root: Path,
        source_layer: Sdf.Layer,
        raw_path: str,
        resolved: Path,
    ) -> None:
        key = (kind, source_layer.identifier, raw_path)
        entry = self.problems.setdefault(
            key,
            {
                "kind": kind,
                "source_layer": source_layer.identifier,
                "asset": raw_path,
                "resolved": str(resolved),
                "scenes": set(),
            },
        )
        entry["scenes"].add(str(scene.relative_to(flat_root)))

    def inspect(
        self,
        attribute: Usd.Attribute,
        scene: Path,
        flat_root: Path,
        grscene_root: Path,
    ) -> None:
        value = attribute.Get()
        if not isinstance(value, Sdf.AssetPath) or not value.path:
            return
        raw_path = value.path
        if is_runtime_asset(raw_path):
            self.categories["runtime_or_uri"] += 1
            return
        source_layer = authored_asset_layer(attribute, value)
        source_path = Path(source_layer.realPath or source_layer.identifier)
        resolved = resolve_asset_path(source_layer, source_path, raw_path)
        if Path(raw_path).is_absolute():
            self.categories["absolute"] += 1
            self.record_problem(
                "absolute", scene, flat_root, source_layer, raw_path, resolved
            )
        category = classify_asset_path(resolved, flat_root, grscene_root, [])
        self.categories[category] += 1
        if category == "unexpected_external":
            self.record_problem(
                category, scene, flat_root, source_layer, raw_path, resolved
            )
        if not resolved.exists():
            self.categories["missing"] += 1
            self.record_problem(
                "missing", scene, flat_root, source_layer, raw_path, resolved
            )

    def serialized_problems(self) -> list[dict[str, Any]]:
        serialized_problems = []
        for entry in self.problems.values():
            serialized = dict(entry)
            serialized["scenes"] = sorted(serialized["scenes"])
            serialized_problems.append(serialized)
        return sorted(
            serialized_problems,
            key=lambda item: (item["kind"], item["source_layer"], item["asset"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy VC+ runtime assets into the dataset and rewrite all "
            "authored USD paths to the local copies."
        )
    )
    parser.add_argument("--flat-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Source data root. Defaults to the parent of --flat-root.",
    )
    parser.add_argument(
        "--grscene-root",
        type=Path,
        help="Allowed sibling GRScenes directory. Defaults to <data-root>/grscenes_commercial.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Output manifest. Defaults to <flat-root>/portable_assets_manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned copies and rewrites without changing files.",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Only rewrite and audit paths; localized assets must already exist.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail on unresolved local asset paths as well as external paths.",
    )
    return parser.parse_args()


def canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def classify_asset_path(
    resolved: Path,
    flat_root: Path,
    grscene_root: Path,
    mappings: list[AssetMapping],
) -> str:
    if any(is_relative_to(resolved, mapping.source) for mapping in mappings):
        return "stale_source"
    if is_relative_to(resolved, flat_root):
        return "vc_plus"
    if is_relative_to(resolved, grscene_root):
        return "grscenes_commercial"
    return "unexpected_external"


def bundle_layer_label(
    layer_path: Path,
    flat_root: Path,
    grscene_root: Path,
) -> str:
    if is_relative_to(layer_path, flat_root):
        return f"{flat_root.name}/{layer_path.relative_to(flat_root)}"
    if is_relative_to(layer_path, grscene_root):
        return f"{grscene_root.name}/{layer_path.relative_to(grscene_root)}"
    return str(layer_path)


def selected_sources(mapping: AssetMapping) -> list[tuple[Path, Path]]:
    if mapping.copy_children is None:
        return [(mapping.source, mapping.destination)]
    return [
        (mapping.source / child, mapping.destination / child)
        for child in mapping.copy_children
    ]


def tree_stats(root: Path) -> dict[str, int]:
    stats = {"files": 0, "symlinks": 0, "bytes": 0}
    if not root.exists():
        return stats
    for path in root.rglob("*"):
        if path.is_symlink():
            stats["symlinks"] += 1
        elif path.is_file():
            stats["files"] += 1
            stats["bytes"] += path.stat().st_size
    return stats


def copy_assets(
    mappings: list[AssetMapping],
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for mapping in mappings:
        for source, destination in selected_sources(mapping):
            if not source.is_dir():
                raise FileNotFoundError(f"Missing asset source: {source}")
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    source,
                    destination,
                    copy_function=shutil.copy2,
                    dirs_exist_ok=True,
                    symlinks=True,
                )
            source_stats = tree_stats(source)
            destination_stats = source_stats if dry_run else tree_stats(destination)
            copied.append(
                {
                    "asset_root": mapping.name,
                    "source": str(source),
                    "destination": str(destination),
                    "source_stats": source_stats,
                    "destination_stats": destination_stats,
                }
            )
            if not dry_run and source_stats != destination_stats:
                raise RuntimeError(
                    f"Copied tree does not match source statistics: {source} -> {destination}"
                )
    return copied


def iter_usd_layers(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in USD_SUFFIXES
        and ".localize-tmp" not in path.name
    )


def is_runtime_asset(raw_path: str) -> bool:
    if URI_PATTERN.match(raw_path) or raw_path.startswith(("anon:", "data:")):
        return True
    normalized = raw_path.replace("\\", "/")
    return "/" not in normalized and normalized.lower().endswith(".mdl")


def resolve_asset_path(layer: Sdf.Layer, layer_path: Path, raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    computed = Sdf.ComputeAssetPathRelativeToLayer(layer, normalized)
    resolved = Path(computed)
    if not resolved.is_absolute():
        resolved = layer_path.parent / resolved
    return canonical(resolved)


def mapped_destination(
    resolved: Path,
    mappings: list[AssetMapping],
) -> tuple[AssetMapping, Path] | None:
    for mapping in mappings:
        if is_relative_to(resolved, mapping.source):
            return mapping, mapping.destination / resolved.relative_to(mapping.source)
    return None


def relative_asset_path(asset_path: Path, layer_path: Path) -> str:
    return Path(os.path.relpath(asset_path, layer_path.parent)).as_posix()


def export_layer_atomically(layer: Sdf.Layer, layer_path: Path) -> None:
    temporary = layer_path.with_name(
        f".{layer_path.stem}.localize-tmp{layer_path.suffix}"
    )
    temporary.unlink(missing_ok=True)
    try:
        file_format = layer.GetFileFormat().formatId
        if file_format not in {"usda", "usdc"}:
            file_format = (
                "usdc" if layer_path.read_bytes()[:8] == b"PXR-USDC" else "usda"
            )
        if not layer.Export(str(temporary), args={"format": file_format}):
            raise RuntimeError(f"Could not export rewritten layer: {layer_path}")
        shutil.copystat(layer_path, temporary)
        os.replace(temporary, layer_path)
    finally:
        temporary.unlink(missing_ok=True)


def rewrite_layer(
    layer_path: Path,
    mappings: list[AssetMapping],
    *,
    dry_run: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        return [], [f"Could not open USD layer: {layer_path}"]

    changes: list[dict[str, str]] = []
    errors: list[str] = []

    def rewrite(raw_path: str) -> str:
        if not raw_path or is_runtime_asset(raw_path) or "[" in raw_path:
            return raw_path
        resolved = resolve_asset_path(layer, layer_path, raw_path)
        mapped = mapped_destination(resolved, mappings)
        replacement: Path | None = None
        reason = ""
        if mapped is not None:
            mapping, replacement = mapped
            reason = mapping.name
            if not dry_run and not replacement.exists():
                errors.append(
                    f"Missing localized target for {layer_path}: {raw_path} -> {replacement}"
                )
                return raw_path
        elif "\\" in raw_path and resolved.exists():
            replacement = resolved
            reason = "normalize_path_separators"

        if replacement is None:
            return raw_path
        rewritten = relative_asset_path(replacement, layer_path)
        if rewritten == raw_path:
            return raw_path
        changes.append(
            {
                "old": raw_path,
                "new": rewritten,
                "reason": reason,
            }
        )
        return rewritten

    UsdUtils.ModifyAssetPaths(layer, rewrite)
    if changes and not errors and not dry_run:
        export_layer_atomically(layer, layer_path)
    return changes, errors


def rewrite_layers(
    flat_root: Path,
    mappings: list[AssetMapping],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    layer_changes: list[dict[str, Any]] = []
    errors: list[str] = []
    layers = iter_usd_layers(flat_root)
    root_counts: Counter[str] = Counter()
    for layer_path in layers:
        changes, layer_errors = rewrite_layer(layer_path, mappings, dry_run=dry_run)
        errors.extend(layer_errors)
        if not changes:
            continue
        root_counts.update(change["reason"] for change in changes)
        layer_changes.append(
            {
                "layer": str(layer_path.relative_to(flat_root)),
                "changes": changes,
            }
        )
    return {
        "layers_scanned": len(layers),
        "layers_changed": len(layer_changes),
        "paths_changed": sum(len(item["changes"]) for item in layer_changes),
        "changes_by_reason": dict(sorted(root_counts.items())),
        "layer_changes": layer_changes,
        "errors": errors,
    }


def audit_layers(
    flat_root: Path,
    grscene_root: Path,
    mappings: list[AssetMapping],
) -> dict[str, Any]:
    audit = AuthoredAssetAudit()
    layers = iter_usd_layers(flat_root)

    for layer_path in layers:
        layer = Sdf.Layer.FindOrOpen(str(layer_path))
        if layer is None:
            audit.missing_paths.append(
                {"layer": str(layer_path), "asset": "<layer>", "resolved": ""}
            )
            continue

        def inspect(raw_path: str) -> str:
            audit.inspect(
                raw_path,
                layer=layer,
                layer_path=layer_path,
                layer_label=str(layer_path.relative_to(flat_root)),
                flat_root=flat_root,
                grscene_root=grscene_root,
                mappings=mappings,
            )
            return raw_path

        UsdUtils.ModifyAssetPaths(layer, inspect)

    return audit.result(layers_scanned=len(layers))


def final_scene_roots(flat_root: Path) -> list[Path]:
    roots: list[Path] = []
    for suffix in USD_SUFFIXES:
        roots.extend(flat_root.glob(f"*/*_outdoor{suffix}"))
        roots.extend(flat_root.glob(f"*/*_innout{suffix}"))
    return sorted(set(roots))


def audit_final_scene_closure(
    flat_root: Path,
    grscene_root: Path,
    mappings: list[AssetMapping],
) -> dict[str, Any]:
    roots = final_scene_roots(flat_root)
    pending = list(roots)
    visited: set[Path] = set()
    audit = AuthoredAssetAudit()
    failed_layers: list[str] = []

    while pending:
        layer_path = canonical(pending.pop())
        if layer_path in visited:
            continue
        visited.add(layer_path)
        layer = Sdf.Layer.FindOrOpen(str(layer_path))
        if layer is None:
            failed_layers.append(str(layer_path))
            continue

        def inspect(raw_path: str) -> str:
            resolved, category = audit.inspect(
                raw_path,
                layer=layer,
                layer_path=layer_path,
                layer_label=bundle_layer_label(layer_path, flat_root, grscene_root),
                flat_root=flat_root,
                grscene_root=grscene_root,
                mappings=mappings,
            )
            if (
                resolved is not None
                and category in {"vc_plus", "grscenes_commercial"}
                and resolved.exists()
                and resolved.suffix.lower() in USD_SUFFIXES
            ):
                pending.append(resolved)
            return raw_path

        UsdUtils.ModifyAssetPaths(layer, inspect)

    return {
        "scene_roots": len(roots),
        "reachable_layers": len(visited),
        "reachable_flat_layers": sum(
            is_relative_to(path, flat_root) for path in visited
        ),
        "reachable_vc_plus_layers": sum(
            is_relative_to(path, flat_root) for path in visited
        ),
        "reachable_grscene_layers": sum(
            is_relative_to(path, grscene_root) for path in visited
        ),
        "asset_paths_by_category": dict(sorted(audit.categories.items())),
        "stale_source_paths": audit.stale_source_paths,
        "unexpected_external_paths": audit.unexpected_external_paths,
        "missing_paths": audit.missing_paths,
        "absolute_authored_paths": audit.absolute_authored_paths,
        "failed_layers": failed_layers,
    }


def authored_asset_layer(attribute: Usd.Attribute, value: Sdf.AssetPath) -> Sdf.Layer:
    for spec in attribute.GetPropertyStack():
        if not spec.HasInfo("default"):
            continue
        authored = spec.GetInfo("default")
        if isinstance(authored, Sdf.AssetPath) and authored.path == value.path:
            return spec.layer
    stack = attribute.GetPropertyStack()
    return stack[0].layer if stack else attribute.GetStage().GetRootLayer()


def audit_effective_scene_assets(
    flat_root: Path,
    grscene_root: Path,
) -> dict[str, Any]:
    audit = EffectiveAssetAudit()
    roots = final_scene_roots(flat_root)
    for scene in roots:
        stage = Usd.Stage.Open(str(scene), load=Usd.Stage.LoadAll)
        if stage is None:
            audit.failed_scenes.append(str(scene.relative_to(flat_root)))
            continue
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Shader":
                continue
            for attribute in prim.GetAttributes():
                audit.inspect(attribute, scene, flat_root, grscene_root)
    return {
        "scene_roots": len(roots),
        "asset_values_by_category": dict(sorted(audit.categories.items())),
        "problems": audit.serialized_problems(),
        "failed_scenes": audit.failed_scenes,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_asset_mappings(
    data_root: Path,
    flat_root: Path,
) -> tuple[list[AssetMapping], list[AssetMapping]]:
    copy_mappings = [
        AssetMapping(
            "objaverse",
            canonical(data_root / "objaverse"),
            canonical(flat_root / "objaverse"),
            ("usd",),
        ),
        AssetMapping(
            "storefront",
            canonical(data_root / "storefront"),
            canonical(flat_root / "storefront"),
        ),
        AssetMapping(
            "nvidia",
            canonical(data_root / "nvidia"),
            canonical(flat_root / "nvidia"),
            ("sky", "Sky_Elements"),
        ),
    ]
    rewrite_mappings = list(copy_mappings)
    legacy_flat_root = canonical(data_root / "vc_full_flat")
    if legacy_flat_root != flat_root:
        rewrite_mappings.append(
            AssetMapping("legacy_flat_root", legacy_flat_root, flat_root)
        )
    return copy_mappings, rewrite_mappings


def audit_failed(
    rewrite: dict[str, Any],
    audit: dict[str, Any],
    closure_audit: dict[str, Any],
    effective_audit: dict[str, Any],
    *,
    strict_missing: bool,
) -> bool:
    failures = [bool(rewrite["errors"])]
    if audit:
        failures.extend(
            bool(audit[key])
            for key in (
                "stale_source_paths",
                "unexpected_external_paths",
                "absolute_authored_paths",
            )
        )
        failures.append(strict_missing and bool(audit["missing_paths"]))
    if closure_audit:
        failures.extend(
            bool(closure_audit[key])
            for key in (
                "stale_source_paths",
                "unexpected_external_paths",
                "absolute_authored_paths",
                "failed_layers",
            )
        )
    if effective_audit:
        failures.append(bool(effective_audit["failed_scenes"]))
        failures.append(
            any(
                problem["kind"] in {"absolute", "unexpected_external"}
                for problem in effective_audit["problems"]
            )
        )
        failures.append(
            strict_missing
            and any(
                problem["kind"] == "missing" for problem in effective_audit["problems"]
            )
        )
    return any(failures)


def main() -> int:
    args = parse_args()
    flat_root = canonical(args.flat_root)
    data_root = canonical(args.data_root or flat_root.parent)
    grscene_root = canonical(args.grscene_root or data_root / "grscenes_commercial")
    manifest_path = canonical(
        args.manifest or flat_root / "portable_assets_manifest.json"
    )
    if not flat_root.is_dir():
        raise FileNotFoundError(f"Missing flat dataset root: {flat_root}")
    if not grscene_root.is_dir():
        raise FileNotFoundError(f"Missing GRScenes root: {grscene_root}")

    mappings, rewrite_mappings = build_asset_mappings(data_root, flat_root)

    copies: list[dict[str, Any]] = []
    if not args.skip_copy:
        copies = copy_assets(mappings, dry_run=args.dry_run)
    rewrite = rewrite_layers(flat_root, rewrite_mappings, dry_run=args.dry_run)
    audit = (
        {} if args.dry_run else audit_layers(flat_root, grscene_root, rewrite_mappings)
    )
    closure_audit = (
        {}
        if args.dry_run
        else audit_final_scene_closure(flat_root, grscene_root, rewrite_mappings)
    )
    effective_audit = (
        {} if args.dry_run else audit_effective_scene_assets(flat_root, grscene_root)
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "flat_root": str(flat_root),
        "grscene_root": str(grscene_root),
        "portable_layout": [flat_root.name, grscene_root.name],
        "copies": copies,
        "rewrite": rewrite,
        "audit": audit,
        "final_scene_closure_audit": closure_audit,
        "effective_scene_asset_audit": effective_audit,
    }
    if not args.dry_run:
        write_manifest(manifest_path, payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    failed = audit_failed(
        rewrite,
        audit,
        closure_audit,
        effective_audit,
        strict_missing=args.strict_missing,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
