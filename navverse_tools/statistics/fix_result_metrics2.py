#!/usr/bin/env python3
"""Write result_fix2.json files with a distance-only success criterion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from navverse.utils.osm_surface import OSMSurfaceMap
from navverse_tools.statistics.fix_result_metrics import extract_records, load_trajectory_points, to_float


INPUT_FILENAMES = ("result_fix.json", "result.json", "results.json")
OUTPUT_NAME = "result_fix2.json"
SUCCESS_DISTANCE = 1.6
_SURFACE_MAP_CACHE: dict[Path, OSMSurfaceMap | None] = {}


def find_input_files(dump_dir: Path) -> list[Path]:
    result_files: list[Path] = []
    for child in sorted(dump_dir.iterdir()):
        if not child.is_dir():
            continue
        for filename in INPUT_FILENAMES:
            candidate = child / filename
            if candidate.is_file():
                result_files.append(candidate)
                break
    return result_files


def episode_label(record: dict[str, Any]) -> str | None:
    label = record.get("episode_label")
    if isinstance(label, str) and label:
        return label

    scene_id = record.get("scene_id")
    episode_id = record.get("episode_id")
    if isinstance(scene_id, str) and scene_id and episode_id is not None:
        return f"{scene_id}_{episode_id}"

    return None


def episode_id(record: dict[str, Any]) -> int | None:
    value = record.get("episode_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass

    label = episode_label(record)
    if label:
        try:
            return int(label.rsplit("_", 1)[-1])
        except ValueError:
            return None

    return None


def load_episode_data(episode_dir: Path) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {}
    if not episode_dir.is_dir():
        return data

    for path in episode_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            data[path.stem] = payload
        elif isinstance(payload, dict):
            data[path.stem] = [
                payload[str(index)] if str(index) in payload else payload[index]
                for index in range(len(payload))
            ]
    return data


def episode_record(
    record: dict[str, Any], episode_data: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    scene_id = record.get("scene_id")
    eid = episode_id(record)
    if isinstance(scene_id, str) and eid is not None:
        episodes = episode_data.get(scene_id)
        if episodes is not None and 0 <= eid < len(episodes):
            return episodes[eid]
    return None


def xyz_distance(a: list[Any] | tuple[Any, ...], b: list[Any] | tuple[Any, ...]) -> float:
    return sum((float(b[index]) - float(a[index])) ** 2 for index in range(3)) ** 0.5


def distance_to_goal_from_position(
    position: list[Any] | tuple[Any, ...], goals: list[dict[str, Any]]
) -> float | None:
    candidates: list[tuple[float, int]] = []
    for goal_index, goal in enumerate(goals):
        reference_path = goal.get("reference_path")
        if not isinstance(reference_path, list) or not reference_path:
            continue

        path_distance_to_goal = 0.0
        for reverse_index in range(len(reference_path)):
            index = len(reference_path) - reverse_index - 1
            point = reference_path[index]
            if not isinstance(point, list) or len(point) < 3:
                continue
            if reverse_index > 0:
                path_distance_to_goal += xyz_distance(point, reference_path[index + 1])
            candidates.append(
                (xyz_distance(position, point) + path_distance_to_goal, goal_index)
            )

    if not candidates:
        return None

    closest_distance, closest_goal_index = min(candidates, key=lambda item: item[0])
    radius = to_float(goals[closest_goal_index].get("radius")) or 0.0
    return max(1e-5, closest_distance - radius)


def initial_distance_to_goal(episode: dict[str, Any] | None) -> float | None:
    if episode is None:
        return None
    start_position = episode.get("start_position")
    goals = episode.get("goals")
    if not isinstance(start_position, list) or len(start_position) < 3:
        return None
    if not isinstance(goals, list):
        return None
    return distance_to_goal_from_position(start_position, goals)


def load_surface_map(
    scene_folder: Path, episode: dict[str, Any] | None
) -> OSMSurfaceMap | None:
    if episode is None:
        return None

    scene_path = episode.get("path")
    if not isinstance(scene_path, str) or not scene_path:
        return None

    osm_path = scene_folder / Path(scene_path).parent / "road_data_with_height.osm"
    osm_path = osm_path.resolve()
    if osm_path in _SURFACE_MAP_CACHE:
        return _SURFACE_MAP_CACHE[osm_path]

    if not osm_path.is_file():
        _SURFACE_MAP_CACHE[osm_path] = None
        return None

    try:
        surface_map = OSMSurfaceMap(osm_path, save_debug=False)
    except Exception as exc:
        print(f"[WARNING] Failed to load OSM surface map {osm_path}: {exc}")
        surface_map = None

    _SURFACE_MAP_CACHE[osm_path] = surface_map
    return surface_map


def fix_navigable_surface_metrics(
    fixed: dict[str, Any],
    result_dir: Path,
    episode: dict[str, Any] | None,
    scene_folder: Path,
) -> None:
    surface_map = load_surface_map(scene_folder, episode)
    if surface_map is None:
        return

    trajectory_points = load_trajectory_points(result_dir, fixed)
    if not trajectory_points:
        fixed["navigable_surface"] = float("nan")
        fixed["navigable_surface_ratio"] = float("nan")
        return

    values = []
    for _, point_xy in trajectory_points:
        on_sidewalk, surface_type = surface_map.is_navigable_surface(point_xy)
        if surface_type == "unknown":
            continue
        values.append(float(on_sidewalk))

    if not values:
        fixed["navigable_surface"] = float("nan")
        fixed["navigable_surface_ratio"] = float("nan")
        return

    fixed["navigable_surface"] = values[-1]
    fixed["navigable_surface_ratio"] = sum(values) / len(values)


def fix_record(
    record: dict[str, Any],
    episode: dict[str, Any] | None,
    success_distance: float,
    result_dir: Path,
    scene_folder: Path,
    fix_nsr: bool,
) -> dict[str, Any]:
    fixed = dict(record)
    distance_to_goal = to_float(fixed.get("distance_to_goal"))
    success = float(distance_to_goal is not None and distance_to_goal < success_distance)
    fixed["success"] = success

    path_length = to_float(fixed.get("path_length"))
    start_distance = initial_distance_to_goal(episode)
    if start_distance is not None and path_length is not None:
        fixed["spl"] = success * (start_distance / max(start_distance, path_length))

    if success == 1.0:
        fixed["termination_reason"] = "success"

    if fix_nsr:
        fix_navigable_surface_metrics(fixed, result_dir, episode, scene_folder)

    return fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate result_fix2.json using distance_to_goal < threshold."
    )
    parser.add_argument("--dump-dir", type=Path, default=Path("dump"))
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=Path("navverse_data/episodes"),
    )
    parser.add_argument("--scene-folder", type=Path, default=Path("navverse_data"))
    parser.add_argument("--success-distance", type=float, default=SUCCESS_DISTANCE)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument(
        "--skip-nsr",
        action="store_true",
        help="Skip offline navigable_surface_ratio recomputation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_data = load_episode_data(args.episode_dir)
    written = 0

    for input_file in find_input_files(args.dump_dir):
        with input_file.open("r", encoding="utf-8") as f:
            records = extract_records(json.load(f))
        if args.expected_episodes is not None and len(records) != args.expected_episodes:
            continue

        fixed_records = []
        for record in records:
            episode = episode_record(record, episode_data)
            fixed_records.append(
                fix_record(
                    record,
                    episode,
                    args.success_distance,
                    input_file.parent,
                    args.scene_folder,
                    not args.skip_nsr,
                )
            )
        out_path = input_file.parent / OUTPUT_NAME
        out_path.write_text(
            json.dumps(fixed_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written += 1
        print(f"Wrote {out_path} ({len(fixed_records)} episodes)")

    print(f"Wrote {written} fixed result files.")


if __name__ == "__main__":
    main()
