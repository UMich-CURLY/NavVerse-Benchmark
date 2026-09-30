#!/usr/bin/env python3
"""Download the runnable NavVerse release or a raw Hugging Face snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "tccoin/navverse-benchmark"
DEFAULT_ARCHIVE = "navverse_v1_july23.tar.gz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "navverse_data"
DEFAULT_RAW_OUTPUT_DIR = REPO_ROOT / "navverse_data"
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "navverse_data" / "downloads"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_RAW_IGNORE = ["navverse_v1_*.tar.gz", "navverse_v1_*.tar.gz.sha256"]
RELEASE_MARKER = "NAVVERSE_DATA_RELEASE_july23.json"
EPISODE_PREFIX = "episodes"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Destination directory. Defaults to navverse_data for both the runnable release "
            "and --raw."
        ),
    )
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Download the raw Hugging Face dataset snapshot without release verification or extraction.",
    )
    parser.add_argument("--include", nargs="*", help="Raw mode only: glob patterns to include.")
    parser.add_argument("--ignore", nargs="*", help="Raw mode only: glob patterns to exclude.")
    parser.add_argument("--max-workers", type=int, default=8, help="Raw mode download workers. Default: 8.")
    parser.add_argument(
        "--no-hf-transfer",
        action="store_true",
        help="Raw mode only: disable the hf_transfer accelerator.",
    )
    parser.add_argument("--archive-path", type=Path, help="Use an existing local archive instead of downloading.")
    parser.add_argument("--checksum-path", type=Path, help="Use an existing local .sha256 file.")
    parser.add_argument("--verify-only", action="store_true", help="Verify the archive checksum without extracting.")
    parser.add_argument("--overwrite", action="store_true", help="Allow extraction over an incomplete existing layout.")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument(
        "--skip-episodes",
        action="store_true",
        help="Install only scene assets; skip episode files from the dataset repository.",
    )
    parser.add_argument("--no-update-env", action="store_true")
    return parser.parse_args(argv)


def download_raw_snapshot(args: argparse.Namespace, output_dir: Path) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub first: python -m pip install 'huggingface_hub>=0.24' hf_transfer")

    if not args.no_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    output_dir.mkdir(parents=True, exist_ok=True)
    ignore_patterns = None if args.include else (args.ignore if args.ignore is not None else DEFAULT_RAW_IGNORE)
    print(
        f"[raw] {args.repo_id} -> {output_dir} "
        f"workers={args.max_workers} hf_transfer={not args.no_hf_transfer}",
        flush=True,
    )
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        token=args.token,
        allow_patterns=args.include,
        ignore_patterns=ignore_patterns,
        max_workers=args.max_workers,
    )
    print(f"[raw] Done: {output_dir}", flush=True)
    return 0


def download_file(repo_id: str, filename: str, directory: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Install huggingface_hub first: python -m pip install 'huggingface_hub>=0.24' hf-xet")

    directory.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id}/{filename}", flush=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(directory),
            token=token,
        )
    )


def download_episodes(repo_id: str, output_dir: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub first: python -m pip install 'huggingface_hub>=0.24' hf-xet")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id}/{EPISODE_PREFIX}/", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{EPISODE_PREFIX}/**"],
        local_dir=str(output_dir),
        token=token,
    )
    return output_dir / EPISODE_PREFIX


def expected_checksum(path: Path) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
        raise ValueError(f"Invalid SHA-256 file: {path}")
    return fields[0].lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    processed = 0
    next_report = 16 * 1024**3
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(chunk)
            processed += len(chunk)
            if processed >= next_report:
                print(f"[verify] {processed / 1024**3:.0f} GiB read", flush=True)
                next_report += 16 * 1024**3
    return digest.hexdigest()


def verify_archive(archive: Path, checksum: Path) -> str:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    wanted = expected_checksum(checksum)
    print(f"[verify] SHA-256 for {archive.name} ({archive.stat().st_size / 1024**3:.1f} GiB)", flush=True)
    actual = sha256_file(archive)
    if actual != wanted:
        raise RuntimeError(f"SHA-256 mismatch: expected {wanted}, got {actual}")
    print(f"[verify] OK {actual}", flush=True)
    return actual


def write_release_marker(
    marker: Path,
    *,
    repo_id: str,
    archive_name: str,
    checksum: str,
    counts: dict[str, int],
) -> None:
    payload = {
        "repo_id": repo_id,
        "archive": archive_name,
        "sha256": checksum,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": counts,
    }
    temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)


def validate_layout(root: Path) -> dict[str, int]:
    required_dirs = [root / "vc_plus", root / "grscenes_commercial", root / "navmesh"]
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise RuntimeError(f"Missing release directories: {missing_dirs}")

    sky = root / "nvidia" / "sky" / "CloudySky.usd"
    if not sky.is_file():
        raise RuntimeError(f"Missing runtime sky asset: {sky}")
    if root not in sky.resolve().parents:
        raise RuntimeError(f"Runtime sky link escapes the release root: {sky}")

    outdoor = sorted((root / "vc_plus").glob("*/*_outdoor.usd"))
    innout = sorted((root / "vc_plus").glob("*/*_innout.usd"))
    commercial = sorted(
        (root / "grscenes_commercial" / "scenes").glob("*/start_result_navigation.usd")
    )
    if (len(outdoor), len(innout), len(commercial)) != (52, 52, 30):
        raise RuntimeError(
            "Unexpected scene counts: "
            f"outdoor={len(outdoor)} innout={len(innout)} grCommercial={len(commercial)}"
        )

    required_navmeshes = []
    for scene in outdoor:
        city = scene.parent.name.lower()
        required_navmeshes.append(root / "navmesh" / f"vcp_{city}_navmesh.bin")
        required_navmeshes.append(root / "navmesh" / f"innout_{city}_navmesh.bin")
    for scene in commercial:
        required_navmeshes.append(root / "navmesh" / f"grCommercial_{scene.parent.name}_navmesh.bin")
    missing_navmeshes = [str(path) for path in required_navmeshes if not path.is_file()]
    if missing_navmeshes:
        raise RuntimeError(f"Missing navmeshes: {missing_navmeshes[:10]}")
    required_metadata = [path.with_name(f"{path.name}.meta.json") for path in required_navmeshes]
    missing_metadata = [str(path) for path in required_metadata if not path.is_file()]
    if missing_metadata:
        raise RuntimeError(f"Missing navmesh metadata: {missing_metadata[:10]}")

    return {
        "outdoor": len(outdoor),
        "innout": len(innout),
        "grCommercial": len(commercial),
        "required_navmeshes": len(required_navmeshes),
        "navmesh_metadata": len(required_metadata),
    }


def validate_episode_layout(root: Path) -> dict[str, int]:
    required_dirs = [root / name for name in ("grCommercial", "innout", "legacy", "vcp")]
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise RuntimeError(f"Missing episode directories: {missing_dirs}")

    json_files = sorted(root.glob("*/*.json"))
    episode_count = 0
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid episode file: {path}") from exc
        if not isinstance(payload, list):
            raise RuntimeError(f"Episode file must contain a JSON list: {path}")
        episode_count += len(payload)

    split_files = sorted(root.glob("*_set.txt"))
    if len(json_files) < 100 or episode_count < 9_000 or not split_files:
        raise RuntimeError(
            "Incomplete episode release: "
            f"json_files={len(json_files)} episodes={episode_count} splits={len(split_files)}"
        )
    return {
        "episode_json_files": len(json_files),
        "episodes": episode_count,
        "episode_splits": len(split_files),
    }


def update_data_folders(env_file: Path, scene_folder: Path) -> None:
    if env_file.exists():
        text = env_file.read_text(encoding="utf-8")
        mode = env_file.stat().st_mode
    else:
        template = REPO_ROOT / ".env_template"
        text = template.read_text(encoding="utf-8") if template.exists() else ""
        mode = 0o100644

    scene_value = f"{scene_folder.resolve().as_posix().rstrip('/')}/"
    episode_value = f"{(scene_folder / EPISODE_PREFIX).resolve().as_posix().rstrip('/')}/"
    updated = text
    for key, value in (("SCENE_FOLDER", scene_value), ("EPISODE_FOLDER", episode_value)):
        line = f"{key}={value}"
        pattern = re.compile(rf"^{key}=.*$", re.MULTILINE)
        updated = pattern.sub(line, updated, count=1) if pattern.search(updated) else f"{line}\n{updated}"
    if not updated.endswith("\n"):
        updated += "\n"

    temporary = env_file.with_name(f".{env_file.name}.tmp.{os.getpid()}")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(updated, encoding="utf-8")
    temporary.chmod(mode & 0o777)
    os.replace(temporary, env_file)
    print(f"[config] Updated SCENE_FOLDER and EPISODE_FOLDER in {env_file}", flush=True)


def main() -> int:
    args = parse_args()
    default_output_dir = DEFAULT_RAW_OUTPUT_DIR if args.raw else DEFAULT_OUTPUT_DIR
    output_dir = (args.output_dir or default_output_dir).expanduser().resolve()
    if args.raw:
        return download_raw_snapshot(args, output_dir)

    download_dir = args.download_dir.expanduser().resolve()
    archive_name = Path(args.archive).name
    checksum_name = f"{archive_name}.sha256"

    marker = output_dir / RELEASE_MARKER
    if marker.is_file() and not args.verify_only and args.archive_path is None:
        counts = validate_layout(output_dir)
        if not args.skip_episodes:
            episode_root = output_dir / EPISODE_PREFIX
            if not episode_root.is_dir():
                download_episodes(args.repo_id, output_dir, args.token)
            counts.update(validate_episode_layout(episode_root))
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            write_release_marker(
                marker,
                repo_id=args.repo_id,
                archive_name=marker_payload.get("archive", archive_name),
                checksum=marker_payload.get("sha256", ""),
                counts=counts,
            )
        if not args.no_update_env:
            update_data_folders(args.env_file.expanduser().resolve(), output_dir)
        print(f"[data] NavVerse release is already installed: {counts}", flush=True)
        return 0

    archive = (
        args.archive_path.expanduser().resolve()
        if args.archive_path
        else download_file(args.repo_id, args.archive, download_dir, args.token)
    )
    checksum = (
        args.checksum_path.expanduser().resolve()
        if args.checksum_path
        else download_file(args.repo_id, checksum_name, download_dir, args.token)
    )
    archive_checksum = verify_archive(archive, checksum)
    if args.verify_only:
        return 0

    occupied = [output_dir / name for name in ("vc_plus", "grscenes_commercial", "navmesh")]
    if any(path.exists() for path in occupied) and not args.overwrite:
        raise RuntimeError(
            f"{output_dir} already contains scene data without a valid {RELEASE_MARKER}; "
            "move it aside or rerun with --overwrite."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive.name} -> {output_dir}", flush=True)
    subprocess.run(
        [
            "tar",
            "--extract",
            "--gzip",
            "--no-same-owner",
            "--file",
            str(archive),
            "--directory",
            str(output_dir),
        ],
        check=True,
    )
    counts = validate_layout(output_dir)
    write_release_marker(
        marker,
        repo_id=args.repo_id,
        archive_name=archive_name,
        checksum=archive_checksum,
        counts=counts,
    )
    if not args.skip_episodes:
        episode_root = download_episodes(args.repo_id, output_dir, args.token)
        counts.update(validate_episode_layout(episode_root))
        write_release_marker(
            marker,
            repo_id=args.repo_id,
            archive_name=archive_name,
            checksum=archive_checksum,
            counts=counts,
        )
    print(f"[extract] Layout verified: {counts}", flush=True)
    if not args.no_update_env:
        update_data_folders(args.env_file.expanduser().resolve(), output_dir)
    if args.archive_path is None and not args.keep_archive:
        archive.unlink(missing_ok=True)
    print(f"[data] Ready for navverse_evaluation ({time.strftime('%F %T')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
