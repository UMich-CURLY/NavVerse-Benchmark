#!/usr/bin/env python3
"""Write result_fix.json files with offline metric corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RESULT_FILENAMES = ("result.json", "results.json")
OUTPUT_NAME = "result_fix.json"
INNOUT_EXIT_POINTS: dict[str, tuple[tuple[float, float], ...]] = {
    "innout_amsterdam": ((212.481, 160.140), (-14.944, 254.028)),
    "innout_austin": ((50.784, -46.221),),
    "innout_barcelona": ((-189.309, 10.748),),
    "innout_berlin": ((-61.313, 4.639),),
    "innout_ny": ((-130.714, 180.777),),
    "innout_paris": ((28.461, -270.853),),
}
INNOUT_INDOOR_REGIONS: dict[str, tuple[dict[str, Any], ...]] = {
    "innout_amsterdam": (
        {
            "type": "polygon",
            "points": (
                (204.73, 161.98),
                (205.10, 156.34),
                (210.10, 158.30),
                (213.81, 159.79),
                (209.91, 165.52),
                (205.81, 162.97),
            ),
        },
        {"type": "circle", "center": (-16.42, 252.31), "radius": 2.0},
    ),
    "innout_austin": (
        {
            "type": "polygon",
            "points": (
                (50.77, -46.48),
                (57.34, -48.84),
                (64.35, -48.90),
                (62.27, -44.77),
                (51.16, -44.35),
                (50.96, -45.07),
            ),
        },
    ),
    "innout_barcelona": (
        {
            "type": "polygon",
            "points": (
                (-197.45, 9.74),
                (-189.06, 2.42),
                (-184.05, 3.11),
                (-184.26, 4.35),
                (-185.05, 6.61),
                (-187.93, 11.11),
                (-195.23, 10.80),
            ),
        },
    ),
    "innout_berlin": (
        {
            "type": "polygon",
            "points": (
                (-63.80, 6.25),
                (-62.41, 4.42),
                (-61.25, 4.35),
                (-59.95, 4.58),
                (-59.85, 7.30),
                (-60.28, 7.59),
                (-63.77, 7.38),
            ),
        },
    ),
    "innout_ny": (
        {
            "type": "polygon",
            "points": (
                (-132.05, 181.04),
                (-129.58, 181.17),
                (-129.18, 181.20),
                (-124.93, 186.89),
                (-125.05, 189.46),
                (-127.71, 191.21),
                (-129.28, 191.92),
            ),
        },
    ),
    "innout_paris": (
        {
            "type": "polygon",
            "points": (
                (19.37, -266.73),
                (24.55, -276.99),
                (25.60, -276.65),
                (30.16, -271.03),
                (25.95, -265.84),
                (20.48, -265.63),
            ),
        },
    ),
}
INNOUT_NEAR_EXIT_RADIUS = 2.0
INNOUT_OUTDOOR_EXIT_DISTANCE = 2.0


def extract_records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
        return obj
    if isinstance(obj, dict):
        for key in ("results", "episodes", "data", "records"):
            value = obj.get(key)
            if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                return value
        if obj and all(isinstance(value, dict) for value in obj.values()):
            records = []
            for episode_key, value in obj.items():
                record = dict(value)
                record.setdefault("episode_id", episode_key)
                records.append(record)
            return records
    raise ValueError("unsupported result JSON schema")


def find_result_files(dump_dir: Path) -> list[Path]:
    result_files: list[Path] = []
    for child in sorted(dump_dir.iterdir()):
        if not child.is_dir():
            continue
        for filename in RESULT_FILENAMES:
            candidate = child / filename
            if candidate.is_file():
                result_files.append(candidate)
                break
    return result_files


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def innout_scene_key(record: dict[str, Any]) -> str | None:
    scene_id = record.get("scene_id")
    if isinstance(scene_id, str) and scene_id.startswith("innout_"):
        return scene_id

    episode_label = record.get("episode_label")
    if isinstance(episode_label, str) and episode_label.startswith("innout_"):
        parts = episode_label.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])

    return None


def episode_label(record: dict[str, Any]) -> str | None:
    label = record.get("episode_label")
    if isinstance(label, str) and label:
        return label

    scene_id = record.get("scene_id")
    episode_id = record.get("episode_id")
    if isinstance(scene_id, str) and scene_id and episode_id is not None:
        return f"{scene_id}_{episode_id}"

    return None


def load_trajectory_points(
    result_dir: Path, record: dict[str, Any]
) -> list[tuple[int, tuple[float, float]]]:
    label = episode_label(record)
    if label is None:
        return []

    data_path = result_dir / "data" / f"{label}.json"
    if not data_path.is_file():
        return []

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trajectory = data.get("trajectory")
    if not isinstance(trajectory, dict):
        return []

    points: list[tuple[int, tuple[float, float]]] = []
    for step, matrix in sorted(trajectory.items(), key=lambda item: int(item[0])):
        if (
            isinstance(matrix, list)
            and len(matrix) >= 2
            and isinstance(matrix[0], list)
            and isinstance(matrix[1], list)
            and len(matrix[0]) >= 4
            and len(matrix[1]) >= 4
        ):
            x = to_float(matrix[0][3])
            y = to_float(matrix[1][3])
            if x is not None and y is not None:
                points.append((int(step), (x, y)))

    return points


def xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def polygon_contains_point(
    polygon: tuple[tuple[float, float], ...], point: tuple[float, float]
) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        crosses_y = (current[1] > point[1]) != (previous[1] > point[1])
        if crosses_y:
            x_intersection = (previous[0] - current[0]) * (
                point[1] - current[1]
            ) / (previous[1] - current[1]) + current[0]
            if point[0] < x_intersection:
                inside = not inside
        previous = current
    return inside


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    segment_length_sq = segment_x * segment_x + segment_y * segment_y
    if segment_length_sq <= 1e-12:
        return xy_distance(point, start)

    point_x = point[0] - start[0]
    point_y = point[1] - start[1]
    t = (point_x * segment_x + point_y * segment_y) / segment_length_sq
    t = max(0.0, min(1.0, t))
    closest = (start[0] + t * segment_x, start[1] + t * segment_y)
    return xy_distance(point, closest)


def closest_exit_segment(
    points: list[tuple[int, tuple[float, float]]],
    exits: tuple[tuple[float, float], ...],
) -> tuple[float, int]:
    if not points:
        return float("inf"), 0
    if len(points) == 1:
        point = points[0][1]
        return min(xy_distance(point, exit_point) for exit_point in exits), 0

    best_distance = float("inf")
    best_index = 0
    for index in range(len(points) - 1):
        start = points[index][1]
        end = points[index + 1][1]
        distance = min(
            point_to_segment_distance(exit_point, start, end)
            for exit_point in exits
        )
        if distance < best_distance:
            best_distance = distance
            best_index = index

    return best_distance, best_index


def point_is_in_indoor_region(
    point: tuple[float, float], regions: tuple[dict[str, Any], ...]
) -> bool:
    for region in regions:
        region_type = region.get("type")
        if region_type == "polygon":
            polygon = region.get("points")
            if isinstance(polygon, tuple) and polygon_contains_point(polygon, point):
                return True
        elif region_type == "circle":
            center = region.get("center")
            radius = region.get("radius")
            if (
                isinstance(center, tuple)
                and isinstance(radius, (int, float))
                and xy_distance(point, center) <= float(radius)
            ):
                return True
    return False


def min_exit_distance(
    point: tuple[float, float], exits: tuple[tuple[float, float], ...]
) -> float:
    return min(xy_distance(point, exit_point) for exit_point in exits)


def innout_state(
    record: dict[str, Any],
    trajectory_points: list[tuple[int, tuple[float, float]]],
) -> tuple[str, int | None] | None:
    scene_key = innout_scene_key(record)
    if scene_key is None:
        return None

    exits = INNOUT_EXIT_POINTS.get(scene_key)
    regions = INNOUT_INDOOR_REGIONS.get(scene_key)
    if exits is None or regions is None or not trajectory_points:
        return None

    exit_distance, exit_segment_index = closest_exit_segment(trajectory_points, exits)
    if exit_distance > INNOUT_NEAR_EXIT_RADIUS:
        return "indoor", None

    for step, point in trajectory_points[exit_segment_index + 1 :]:
        if point_is_in_indoor_region(point, regions):
            continue
        if min_exit_distance(point, exits) >= INNOUT_OUTDOOR_EXIT_DISTANCE:
            return "outdoor", step

    return "near_exit", None


def fix_record(
    record: dict[str, Any],
    min_ce_path_length: float,
    trajectory_points: list[tuple[int, tuple[float, float]]],
) -> dict[str, Any]:
    fixed = dict(record)
    covered_area = to_float(fixed.get("covered_area"))
    path_length = to_float(fixed.get("path_length"))
    if covered_area is not None and path_length is not None:
        if path_length <= 1e-9:
            fixed["exploration_efficiency"] = 0.0
        else:
            fixed["exploration_efficiency"] = covered_area / max(
                path_length, min_ce_path_length
            )

    state = innout_state(fixed, trajectory_points)
    if state is not None:
        innout_state_value, exit_time = state
        fixed["innout_state"] = innout_state_value
        fixed["innout_exit_time"] = exit_time

    return fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate result_fix.json files using offline metric fixes."
    )
    parser.add_argument("--dump-dir", type=Path, default=Path("dump"))
    parser.add_argument("--min-ce-path-length", type=float, default=1.0)
    parser.add_argument("--expected-episodes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = 0
    for result_file in find_result_files(args.dump_dir):
        with result_file.open("r", encoding="utf-8") as f:
            records = extract_records(json.load(f))
        if args.expected_episodes is not None and len(records) != args.expected_episodes:
            continue

        fixed_records = []
        for record in records:
            trajectory_points = load_trajectory_points(result_file.parent, record)
            fixed_records.append(
                fix_record(record, args.min_ce_path_length, trajectory_points)
            )
        out_path = result_file.parent / OUTPUT_NAME
        out_path.write_text(
            json.dumps(fixed_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written += 1
        print(f"Wrote {out_path} ({len(fixed_records)} episodes)")

    print(f"Wrote {written} fixed result files.")


if __name__ == "__main__":
    main()
