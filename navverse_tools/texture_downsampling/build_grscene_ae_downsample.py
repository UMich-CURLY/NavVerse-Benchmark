#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, log_file, cwd=None):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_file.open("a") as f:
        f.write("\n[CMD] " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"[EXIT] code={proc.returncode} elapsed_s={time.time() - start:.3f}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(map(str, cmd))}; log={log_file}")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent.parent
LOCAL_COMPRESSONATOR = SCRIPT_DIR / ".local_tools" / "compressonator" / "compressonatorcli-4.5.52-Linux" / "compressonatorcli"


def resolve_compressonator(value):
    if value:
        return Path(value).expanduser().resolve()
    candidates = [
        LOCAL_COMPRESSONATOR,
        Path("/usr/bin/compressonatorcli"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("compressonatorcli")


def ensure_compressonator(initial):
    """Locate compressonatorcli; if missing, run setup.sh to install it locally."""
    resolved = resolve_compressonator(initial)
    if resolved.exists() or resolved.name != "compressonatorcli":
        return resolved
    setup = SCRIPT_DIR / "setup.sh"
    if setup.exists():
        print(f"[INFO] compressonatorcli not found; running {setup}")
        subprocess.run(["bash", str(setup)], check=True)
        resolved = resolve_compressonator(initial)
    return resolved


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    _data_root = DEFAULT_REPO_ROOT / "navverse_data"
    default_source = _data_root / "grscenes_commercial"
    default_dest = _data_root / f"grscenes_commercial_downsample_texture_ae_{time.strftime('%Y%m%d_%H%M%S')}"
    default_log_dir = DEFAULT_REPO_ROOT / "logs" / f"grscene_ae_downsample_{time.strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Build the GRScenes A+E optimized copy: material deactivation + BC7/BC5/BC4 no-mip textures.")
    parser.add_argument("--source_root", default=str(default_source))
    parser.add_argument("--dest_root", default=str(default_dest))
    parser.add_argument("--repo_root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--log_dir", default=str(default_log_dir))
    parser.add_argument("--compressonator", default="")
    parser.add_argument("--usd_glob", default="scenes/scene*/start_result_navigation.usd")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_copy", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args, passthrough = parser.parse_known_args()

    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    helper = repo_root / "navverse_tools" / "texture_downsampling" / "process_material.py"
    compressonator = ensure_compressonator(args.compressonator)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not helper.exists():
        raise FileNotFoundError(helper)

    summary = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "repo_root": str(repo_root),
        "log_dir": str(log_dir),
        "compressonator": str(compressonator),
        "usd_glob": args.usd_glob,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "steps": [],
    }
    write_summary(log_dir / "summary.json", summary)

    if not args.skip_copy:
        if dest_root.exists() and not args.resume:
            raise FileExistsError(f"Destination exists; pass --resume or choose a new --dest_root: {dest_root}")
        dest_root.mkdir(parents=True, exist_ok=True)
        run(["rsync", "-a", "--info=progress2", f"{source_root}/", f"{dest_root}/"], log_dir / "copy.log")
        summary["steps"].append({"name": "copy", "status": "done"})
        write_summary(log_dir / "summary.json", summary)

    usds = sorted(dest_root.glob(args.usd_glob), key=lambda p: str(p))
    if args.limit > 0:
        usds = usds[: args.limit]
    if not usds:
        raise RuntimeError(f"No USDs matched {dest_root / args.usd_glob}")
    summary["usd_count"] = len(usds)
    write_summary(log_dir / "summary.json", summary)

    python = sys.executable
    for idx, usd in enumerate(usds, 1):
        scene_name = usd.parent.name
        material_json = log_dir / f"{idx:02d}_{scene_name}_E_material_deactivate.json"
        texture_json = log_dir / f"{idx:02d}_{scene_name}_A_texture_bc7_bc5_bc4_nomip.json"
        scene_log = log_dir / f"{idx:02d}_{scene_name}.log"
        if args.resume and material_json.exists() and texture_json.exists():
            summary["steps"].append({"name": scene_name, "status": "skipped_existing", "scene_usd": str(usd)})
            write_summary(log_dir / "summary.json", summary)
            continue
        run([
            python, str(helper),
            "--op", "merge_materials_deactivate",
            "--scene_usd", str(usd),
            "--scene_root", str(dest_root),
            "--output_json", str(material_json),
            *passthrough,
        ], scene_log, cwd=repo_root)
        run([
            python, str(helper),
            "--op", "convert",
            "--variant", "a_bc7_nomip",
            "--scene_usd", str(usd),
            "--scene_root", str(dest_root),
            "--output_json", str(texture_json),
            "--compressonator", str(compressonator),
            *passthrough,
        ], scene_log, cwd=repo_root)
        step = {"name": scene_name, "status": "done", "scene_usd": str(usd), "material_json": str(material_json), "texture_json": str(texture_json)}
        summary["steps"].append(step)
        write_summary(log_dir / "summary.json", summary)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    summary["status"] = "done"
    write_summary(log_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
