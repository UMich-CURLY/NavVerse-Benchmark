import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher
from PIL import Image

parser = argparse.ArgumentParser(description="Build cumulative GRScenes material/texture ablations.")
parser.add_argument("--op", required=True, choices=["convert", "merge_materials", "merge_materials_prune", "merge_materials_deactivate"])
parser.add_argument("--scene_usd", required=True)
parser.add_argument("--scene_root", required=True)
parser.add_argument("--output_json", required=True)
parser.add_argument("--compressonator", default="compressonatorcli")
parser.add_argument("--variant", choices=["a_bc7_nomip", "b_bc7_mip", "c_bc7_mip_skip64"], default="a_bc7_nomip")
parser.add_argument("--small_threshold", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Sdf, Usd, UsdShade, UsdUtils

TEXTURE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp"}
COLOR_HINTS = ("color", "albedo", "basecolor", "base_color", "diffuse", "day")
NORMAL_HINTS = ("normal", "norm", "bump")
SCALAR_HINTS = ("rough", "metal", "ao", "occlusion", "height", "disp", "opacity", "alpha")
scene_usd = Path(args.scene_usd).resolve()
scene_root = Path(args.scene_root).resolve()
sha_cache = {}


def resolve_asset(asset):
    p = Path(str(asset))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append((scene_usd.parent / p).resolve())
    candidates.append((scene_root / p).resolve())
    for marker in (
        "/grscenes_commercial_downsample_texture/",
        "/grscenes_commercial_ablation_abcd_20260505/",
        "/grscenes_commercial/",
    ):
        s = str(p)
        if marker in s:
            candidates.append((scene_root / s.split(marker, 1)[1]).resolve())
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def deps_textures():
    _layers, assets, _unresolved = UsdUtils.ComputeAllDependencies(str(scene_usd))
    out = []
    seen = set()
    for asset in assets:
        if Path(str(asset)).suffix.lower() not in TEXTURE_EXTS:
            continue
        p = resolve_asset(str(asset))
        if not p.exists():
            continue
        try:
            p.relative_to(scene_root)
        except ValueError:
            continue
        key = os.path.realpath(str(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return sorted(out)


def classify(path):
    name = path.name.lower()
    if any(x in name for x in NORMAL_HINTS):
        return "normal"
    if any(x in name for x in COLOR_HINTS):
        return "color"
    if any(x in name for x in SCALAR_HINTS):
        return "scalar"
    return "other"


def make_relative(path):
    return os.path.relpath(str(path), str(scene_usd.parent))


def apply_asset_mapping(mapping):
    stage = Usd.Stage.Open(str(scene_usd))
    replacements = []
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath):
                resolved = os.path.realpath(str(resolve_asset(value.path)))
                new_path = mapping.get(resolved)
                if new_path:
                    new_rel = make_relative(new_path)
                    attr.Set(Sdf.AssetPath(new_rel))
                    replacements.append({"prim": str(prim.GetPath()), "attr": attr.GetName(), "old": value.path, "new": new_rel})
    stage.GetRootLayer().Save()
    return replacements


def run_compressonator(src, dst, fmt, mipmaps):
    exe = shutil.which(args.compressonator) or args.compressonator
    exe_path = Path(exe)
    use_shell_wrapper = False
    if exe_path.is_file():
        try:
            prefix = exe_path.read_bytes()[:4]
            use_shell_wrapper = not (prefix.startswith(b"#!") or prefix == b"\x7fELF")
        except OSError:
            use_shell_wrapper = False
    cmd = (["bash", str(exe_path)] if use_shell_wrapper else [str(exe_path)]) + ["-fd", fmt]
    if mipmaps:
        cmd += ["-miplevels", "20"]
    else:
        cmd += ["-nomipmap"]
    cmd += [str(src), str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)


def with_output_lock(path):
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return lock_path.open("w", encoding="utf-8")


def pad_to_block_size(im, block=4):
    """Pad bottom/right edges by replicating the last row/column so dims are a multiple of `block`.

    BC7/BC5/BC4 are 4x4 block formats; compressing a texture whose dims aren't a multiple of 4
    produces files that decode incorrectly (visible as missing/blue textures in some renderers).
    """
    w, h = im.size
    new_w = (w + block - 1) // block * block
    new_h = (h + block - 1) // block * block
    if (new_w, new_h) == (w, h):
        return im, False
    canvas = Image.new(im.mode, (new_w, new_h))
    canvas.paste(im, (0, 0))
    if new_h > h:
        bottom = im.crop((0, h - 1, w, h))
        for y in range(h, new_h):
            canvas.paste(bottom, (0, y))
    if new_w > w:
        right = canvas.crop((w - 1, 0, w, new_h))
        for x in range(w, new_w):
            canvas.paste(right, (x, 0))
    return canvas, True


def convert_textures():
    mipmaps = args.variant in {"b_bc7_mip", "c_bc7_mip_skip64"}
    skip_small = args.variant == "c_bc7_mip_skip64"
    records = []
    mapping = {}
    with tempfile.TemporaryDirectory(prefix="bc7_pad_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for path in deps_textures():
            kind = classify(path)
            before = path.stat().st_size
            with Image.open(path) as im:
                width, height = im.size
                im_rgb_mode = im.mode
                if skip_small and max(width, height) <= args.small_threshold:
                    records.append({"path": str(path), "kind": kind, "width": width, "height": height, "before_bytes": before, "after_bytes": before, "output": str(path), "skipped_small": True})
                    continue
                padded_im, padded = pad_to_block_size(im, block=4)
                padded_w, padded_h = padded_im.size
            if kind == "normal":
                fmt = "BC5"
            elif kind == "scalar":
                fmt = "BC4"
            else:
                fmt = "BC7"
            suffix = f".{fmt.lower()}_mip.dds" if mipmaps else f".{fmt.lower()}.dds"
            out = path.with_suffix(suffix)
            src_for_compress = path
            if padded:
                src_for_compress = tmpdir_path / f"{path.stem}_{padded_w}x{padded_h}.png"
                padded_im.save(src_for_compress)
            failed = None
            with with_output_lock(out) as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if not out.exists():
                    try:
                        run_compressonator(src_for_compress, out, fmt, mipmaps)
                    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                        failed = repr(exc)
            if failed or not out.exists():
                records.append({"path": str(path), "kind": kind, "format": fmt, "width": width, "height": height, "padded_width": padded_w, "padded_height": padded_h, "before_bytes": before, "after_bytes": before, "output": str(path), "skipped_small": False, "failed": failed or "missing_output"})
                continue
            mapping[os.path.realpath(str(path))] = out
            records.append({"path": str(path), "kind": kind, "format": fmt, "width": width, "height": height, "padded_width": padded_w, "padded_height": padded_h, "before_bytes": before, "after_bytes": out.stat().st_size, "output": str(out), "skipped_small": False})
    replacements = apply_asset_mapping(mapping)
    failed_count = sum(1 for r in records if r.get("failed"))
    return {"variant": args.variant, "texture_count": len(records), "converted_count": sum(1 for r in records if not r.get("skipped_small") and not r.get("failed")), "skipped_small_count": sum(1 for r in records if r.get("skipped_small")), "failed_count": failed_count, "padded_count": sum(1 for r in records if r.get("padded_width") not in (None, r.get("width"))), "before_total_bytes": sum(r["before_bytes"] for r in records), "after_total_bytes": sum(r["after_bytes"] for r in records), "records": records, "replacements": replacements}


def sha_file(path):
    path = str(path)
    if path in sha_cache:
        return sha_cache[path]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    sha_cache[path] = h.hexdigest()
    return sha_cache[path]


def norm_value(v):
    if isinstance(v, Sdf.AssetPath):
        p = resolve_asset(v.path)
        if p.exists() and p.suffix.lower() in TEXTURE_EXTS:
            return ("asset_sha", p.suffix.lower(), sha_file(p), p.stat().st_size)
        return ("asset", os.path.realpath(str(p)) if p.exists() else v.path)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return tuple(norm_value(x) for x in v)
    return repr(v)


def material_signature(mat_prim):
    prefix = str(mat_prim.GetPath())
    rows = []
    for prim in Usd.PrimRange(mat_prim):
        rel_path = str(prim.GetPath())[len(prefix):] or "."
        attrs = []
        for attr in prim.GetAttributes():
            conns = tuple(sorted(str(c)[len(prefix):] if str(c).startswith(prefix) else str(c) for c in attr.GetConnections()))
            attrs.append((attr.GetName(), norm_value(attr.Get()), conns))
        rels = []
        for rel in prim.GetRelationships():
            targets = tuple(sorted(str(t)[len(prefix):] if str(t).startswith(prefix) else str(t) for t in rel.GetTargets()))
            rels.append((rel.GetName(), targets))
        rows.append((rel_path, prim.GetTypeName(), tuple(sorted(attrs)), tuple(sorted(rels))))
    return hashlib.sha256(repr(tuple(rows)).encode()).hexdigest()


def merge_materials(prune=False, deactivate=False):
    stage = Usd.Stage.Open(str(scene_usd))
    groups = defaultdict(list)
    for prim in stage.TraverseAll():
        if prim.IsA(UsdShade.Material):
            groups[material_signature(prim)].append(str(prim.GetPath()))
    mapping = {}
    for paths in groups.values():
        if len(paths) <= 1:
            continue
        canonical = sorted(paths)[0]
        for path in paths:
            if path != canonical:
                mapping[Sdf.Path(path)] = Sdf.Path(canonical)
    replacements = []
    for prim in stage.TraverseAll():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if not targets:
                continue
            new_targets = [mapping.get(t, t) for t in targets]
            if new_targets != targets:
                rel.SetTargets(new_targets)
                replacements.append({"prim": str(prim.GetPath()), "rel": rel.GetName(), "old": [str(t) for t in targets], "new": [str(t) for t in new_targets]})
    removed_prims = []
    if prune or deactivate:
        for old_path in sorted(mapping.keys(), key=lambda x: len(str(x)), reverse=True):
            prim = stage.GetPrimAtPath(old_path)
            if prim and prim.IsValid():
                removed_prims.append(str(old_path))
                if deactivate:
                    prim.SetActive(False)
                else:
                    stage.RemovePrim(old_path)
    stage.GetRootLayer().Save()
    duplicate_groups = [v for v in groups.values() if len(v) > 1]
    remaining_materials = sum(1 for prim in stage.TraverseAll() if prim.IsA(UsdShade.Material))
    return {"material_count": sum(len(v) for v in groups.values()), "signature_group_count": len(groups), "duplicate_group_count": len(duplicate_groups), "duplicate_material_count": sum(len(v) - 1 for v in duplicate_groups), "max_group_size": max((len(v) for v in groups.values()), default=0), "remaining_material_count": remaining_materials, "prune": prune, "deactivate": deactivate, "removed_prim_count": len(removed_prims), "removed_prims": removed_prims, "replacements": replacements}

if args.op == "convert":
    result = convert_textures()
elif args.op == "merge_materials":
    result = merge_materials(prune=False)
elif args.op == "merge_materials_prune":
    result = merge_materials(prune=True)
else:
    result = merge_materials(deactivate=True)
Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
summary = {k: v for k, v in result.items() if k not in ("records", "replacements")}
summary["replacements_count"] = len(result.get("replacements", []))
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
os._exit(0)
