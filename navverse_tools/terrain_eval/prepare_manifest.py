#!/usr/bin/env python3
"""Create a frozen shared manifest for controlled terrain evaluation."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from navverse_tools.terrain_eval.terrain import (
    PATCH_CENTER_FRACTIONS,
    PATCH_LENGTH_M,
    PATCH_WIDTH_M,
    SLOPE_ANGLE_DEG,
    STAIR_HEIGHT_M,
    episode_terrain_seed,
    path_length_xy,
    path_sha256,
    shared_patch_placements,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-dir", default="navverse_data/episodes/vcp"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--episodes-per-scene", type=int, default=10)
    parser.add_argument(
        "--manifest",
        default="navverse_tools/terrain_eval/configs/vcp_shared_100_seed42.json",
    )
    parser.add_argument(
        "--set-file",
        default="navverse_tools/terrain_eval/configs/vcp_shared_100_seed42.txt",
    )
    return parser.parse_args()


def episode_hash(episode):
    payload = json.dumps(episode, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    by_scene = defaultdict(list)
    source_hashes = {}
    for path in sorted(Path(args.episode_dir).glob("*.json")):
        raw = path.read_bytes()
        source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for episode in json.loads(raw):
            reference_path = episode["goals"][0]["reference_path"]
            placements = shared_patch_placements(reference_path)
            if placements is None:
                continue
            label = f"{episode['scene_id']}_{episode['episode_id']}"
            by_scene[episode["scene_id"]].append(
                {
                    "scene_id": episode["scene_id"],
                    "episode_id": episode["episode_id"],
                    "episode_label": label,
                    "path_length_m": path_length_xy(reference_path),
                    "path_sha256": path_sha256(reference_path),
                    "episode_sha256": episode_hash(episode),
                    "terrain_seed": episode_terrain_seed(args.seed, label),
                    "patches": placements,
                }
            )

    scenes = sorted(scene for scene, rows in by_scene.items() if len(rows) >= args.episodes_per_scene)
    rng.shuffle(scenes)
    selected_scenes = scenes[: args.scene_count]
    if len(selected_scenes) != args.scene_count:
        raise RuntimeError(f"Only {len(selected_scenes)} scenes have enough eligible episodes")

    selected = []
    for scene in selected_scenes:
        rows = sorted(by_scene[scene], key=lambda row: int(row["episode_id"]))
        rng.shuffle(rows)
        selected.extend(rows[: args.episodes_per_scene])

    manifest = {
        "version": 1,
        "seed": args.seed,
        "selection": {
            "scene_count": args.scene_count,
            "episodes_per_scene": args.episodes_per_scene,
            "minimum_path_length_m": 20.0,
            "selected_scenes": selected_scenes,
        },
        "terrain": {
            "patch_count": 3,
            "patch_length_m": PATCH_LENGTH_M,
            "patch_width_m": PATCH_WIDTH_M,
            "patch_center_fractions": PATCH_CENTER_FRACTIONS,
            "slope_angle_deg": SLOPE_ANGLE_DEG,
            "stair_height_m": STAIR_HEIGHT_M,
            "uneven_height_range_m": [0.02, 0.06],
            "identical_mesh_copies_per_episode": True,
        },
        "source_json_sha256": source_hashes,
        "episodes": selected,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    set_path = Path(args.set_file)
    set_path.write_text("\n".join(row["episode_label"] for row in selected) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "set_file": str(set_path), "scenes": selected_scenes, "episodes": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
