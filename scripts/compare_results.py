#!/usr/bin/env python3
"""Summarize complete benchmark result dumps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_DUMP_DIR = Path("dump")
DEFAULT_EPISODES = 400
COMPLETE_LIST_NAME = "complete_result_folders.json"
METRICS_TABLE_NAME = "complete_result_metrics.md"
RESULT_FILENAMES = ("result_fix.json", "result.json", "results.json")
NON_METRIC_KEYS = {"episode_id", "step"}


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


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return extract_records(json.load(f))


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


def to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None

    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metric_names(records: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for record in records:
        for key, value in record.items():
            if key in NON_METRIC_KEYS:
                continue
            if to_number(value) is not None:
                names.add(key)
    return names


def average_metrics(records: list[dict[str, Any]], metrics: list[str]) -> dict[str, float]:
    averages: dict[str, float] = {}
    for metric in metrics:
        values = [
            number
            for record in records
            if (number := to_number(record.get(metric))) is not None
        ]
        if values:
            averages[metric] = statistics.fmean(values)
    return averages


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def write_complete_list(
    path: Path,
    complete_runs: list[dict[str, Any]],
    expected_episodes: int,
) -> None:
    payload = {
        "expected_episodes": expected_episodes,
        "complete_count": len(complete_runs),
        "complete_folders": complete_runs,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_metrics_table(
    path: Path,
    complete_runs: list[dict[str, Any]],
    metrics: list[str],
    expected_episodes: int,
) -> None:
    columns = ["folder", "episodes", "result_file", *metrics]
    lines = [
        "# Complete Result Metrics",
        "",
        f"Complete folders are runs with exactly {expected_episodes} episodes.",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]

    for run in complete_runs:
        averages = run["metrics"]
        row = [
            run["folder"],
            str(run["episodes"]),
            run["result_file"],
            *[format_number(averages.get(metric)) for metric in metrics],
        ]
        lines.append("|" + "|".join(row) + "|")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find complete benchmark result folders and average their metrics."
    )
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--expected-episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--list-name", default=COMPLETE_LIST_NAME)
    parser.add_argument("--table-name", default=METRICS_TABLE_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_files = find_result_files(args.dump_dir)
    complete_runs: list[dict[str, Any]] = []
    all_metrics: set[str] = set()

    loaded: list[tuple[Path, list[dict[str, Any]]]] = []
    for result_file in result_files:
        records = load_records(result_file)
        loaded.append((result_file, records))
        if len(records) == args.expected_episodes:
            all_metrics.update(metric_names(records))

    metrics = sorted(all_metrics)
    for result_file, records in loaded:
        if len(records) != args.expected_episodes:
            continue
        complete_runs.append(
            {
                "folder": result_file.parent.name,
                "path": str(result_file.parent),
                "result_file": result_file.name,
                "episodes": len(records),
                "metrics": average_metrics(records, metrics),
            }
        )

    list_path = args.dump_dir / args.list_name
    table_path = args.dump_dir / args.table_name
    write_complete_list(list_path, complete_runs, args.expected_episodes)
    write_metrics_table(table_path, complete_runs, metrics, args.expected_episodes)

    print(f"Scanned {len(result_files)} result files.")
    print(f"Found {len(complete_runs)} complete folders with {args.expected_episodes} episodes.")
    print(f"Wrote {list_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
