#!/usr/bin/env python3
"""Create PlaceNav store episodes with intent-driven goals."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path


DEFAULT_CATALOG = Path("navverse_tools/storefront/storefront_intent_catalog.json")
DEFAULT_EPISODE_FOLDER = Path("navverse_data/episodes")
DEFAULT_OUTPUT_FOLDER = Path("episodes_placenav")
TRAILING_INSTANCE_RE = re.compile(r"_\d+$")


def category_from_objnav(objnav: str) -> str:
    return TRAILING_INSTANCE_RE.sub("", objnav or "")


def stable_instruction_index(seed: int, episode_label: str, instruction_count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{episode_label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % instruction_count


def load_catalog(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = data.get("categories", {})
    return {
        category: value["instructions"]
        for category, value in categories.items()
        if value.get("instructions")
    }


def iter_episode_files(episode_folder: Path) -> list[Path]:
    return sorted(path for path in episode_folder.glob("*.json") if path.is_file())


def convert_episode(
    episode: dict,
    instructions_by_category: dict[str, list[str]],
    seed: int,
    preserve_instruction: bool,
) -> dict | None:
    category = category_from_objnav(episode.get("objnav", ""))
    instructions = instructions_by_category.get(category)
    if not instructions:
        return None

    converted = deepcopy(episode)
    episode_label = f"{converted['scene_id']}_{converted['episode_id']}"
    instruction_id = stable_instruction_index(seed, episode_label, len(instructions))
    placenav_goal = instructions[instruction_id]

    original_instruction = converted.get("instruction", "")
    if not preserve_instruction and original_instruction and original_instruction != placenav_goal:
        converted["route_instruction"] = original_instruction
        converted["instruction"] = placenav_goal
    converted["placenav_goal"] = placenav_goal
    converted["placenav_category"] = category
    converted["placenav_instruction_id"] = instruction_id
    return converted


def write_episode_set(output_folder: Path, labels: list[str], split_name: str) -> None:
    content = "\n".join(labels) + ("\n" if labels else "")
    (output_folder / f"{split_name}_set.txt").write_text(content, encoding="utf-8")


def convert_folder(
    episode_folder: Path,
    output_folder: Path,
    catalog_path: Path,
    seed: int,
    dry_run: bool,
    update_source: bool,
    split_folder: Path,
    split_name: str,
    preserve_instruction: bool,
) -> tuple[int, dict[str, int], list[str]]:
    instructions_by_category = load_catalog(catalog_path)
    output_by_file: dict[Path, list[dict]] = {}
    updated_source_by_file: dict[Path, list[dict]] = {}
    category_counts: dict[str, int] = {}
    labels: list[str] = []
    missing_categories: set[str] = set()

    for episode_file in iter_episode_files(episode_folder):
        data = json.loads(episode_file.read_text(encoding="utf-8"))
        converted_episodes = []
        updated_data = []
        file_changed = False
        for episode in data:
            converted = convert_episode(episode, instructions_by_category, seed, preserve_instruction)
            if converted is None:
                updated_data.append(episode)
                continue
            converted_episodes.append(converted)
            updated_data.append(converted if update_source else episode)
            file_changed = True
            labels.append(f"{converted['scene_id']}_{converted['episode_id']}")
            category_counts[converted["placenav_category"]] = category_counts.get(converted["placenav_category"], 0) + 1
        if converted_episodes:
            output_by_file[episode_file.name] = converted_episodes
        if file_changed:
            updated_source_by_file[episode_file] = updated_data

    labels.sort()
    if not dry_run:
        if update_source:
            for episode_file, episodes in sorted(updated_source_by_file.items()):
                episode_file.write_text(json.dumps(episodes, indent=4) + "\n", encoding="utf-8")
        else:
            output_folder.mkdir(parents=True, exist_ok=True)
            for filename, episodes in sorted(output_by_file.items()):
                (output_folder / filename).write_text(json.dumps(episodes, indent=4) + "\n", encoding="utf-8")
            write_episode_set(output_folder, labels, "test")
        split_folder.mkdir(parents=True, exist_ok=True)
        write_episode_set(split_folder, labels, split_name)

    return len(labels), dict(sorted(category_counts.items())), sorted(missing_categories)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-folder", type=Path, default=DEFAULT_EPISODE_FOLDER)
    parser.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-source", action="store_true", help="Add PlaceNav fields to matching source episodes in place.")
    parser.add_argument("--split-folder", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--split-name", type=str, default="placenav")
    parser.add_argument("--preserve-instruction", action="store_true", help="Keep existing route instruction text and only add placenav_goal fields.")
    args = parser.parse_args()

    total, category_counts, missing_categories = convert_folder(
        args.episode_folder,
        args.output_folder,
        args.catalog,
        args.seed,
        args.dry_run,
        args.update_source,
        args.split_folder,
        args.split_name,
        args.preserve_instruction,
    )
    mode = "Would write" if args.dry_run else "Wrote"
    target = args.episode_folder if args.update_source else args.output_folder
    print(f"{mode} {total} PlaceNav episodes to {target}")
    print(json.dumps(category_counts, indent=2))
    if missing_categories:
        print(f"Missing storefront instruction categories: {', '.join(missing_categories)}")


if __name__ == "__main__":
    main()
