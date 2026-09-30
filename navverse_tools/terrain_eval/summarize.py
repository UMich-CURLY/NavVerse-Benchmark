#!/usr/bin/env python3
"""Summarize terrain-evaluation cells without publishing run artifacts."""
import argparse
import json
import math
from pathlib import Path


ROBOTS = ("drone", "spot", "turtlebot")
TERRAINS = ("original", "flat", "slope", "stairs", "uneven")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument(
        "--manifest",
        default="navverse_tools/terrain_eval/configs/vcp_shared_100_seed42.json",
    )
    parser.add_argument("--robots", default=",".join(ROBOTS))
    parser.add_argument("--terrains", default=",".join(TERRAINS))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main():
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    expected = {row["episode_label"] for row in manifest["episodes"]}
    robots = tuple(item.strip() for item in args.robots.split(",") if item.strip())
    terrains = tuple(item.strip() for item in args.terrains.split(",") if item.strip())
    if not robots or not terrains or not set(robots).issubset(ROBOTS) or not set(terrains).issubset(TERRAINS):
        raise ValueError("Unsupported or empty robot/terrain selection")
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "expected_episodes": len(expected),
        "robots": list(robots),
        "terrains": list(terrains),
        "cells": {},
    }
    errors = []
    for terrain in terrains:
        for robot in robots:
            cell = f"{terrain}_{robot}"
            result_path = run_root / cell / "result.json"
            if not result_path.is_file():
                errors.append(f"{cell}: missing result.json")
                continue
            rows = json.loads(result_path.read_text())
            labels = [row.get("episode_label") for row in rows]
            duplicates = sorted(label for label in set(labels) if labels.count(label) > 1)
            missing = sorted(expected - set(labels))
            unexpected = sorted(set(labels) - expected)
            if duplicates or missing or unexpected or len(rows) != 100:
                errors.append(
                    f"{cell}: rows={len(rows)} duplicate={len(duplicates)} missing={len(missing)} unexpected={len(unexpected)}"
                )
            successes = [float(row.get("success", 0.0)) for row in rows]
            speeds = [float(row.get("average_horizontal_speed", float("nan"))) for row in rows]
            finite_speeds = [value for value in speeds if math.isfinite(value)]
            completed = [float(row.get("terrain_patches_completed", 0.0)) for row in rows]
            summary["cells"][cell] = {
                "terrain": terrain,
                "robot": robot,
                "episodes": len(rows),
                "unique_episodes": len(set(labels)),
                "success_rate": mean(successes),
                "average_horizontal_speed_mps": mean(finite_speeds),
                "mean_patches_completed": mean(completed),
                "all_three_patches_completed_rate": mean([value == 3.0 for value in completed]),
                "termination_reasons": {
                    reason: sum(row.get("termination_reason") == reason for row in rows)
                    for reason in sorted({row.get("termination_reason", "missing") for row in rows})
                },
            }
    summary["validation_errors"] = errors
    output_path = run_root / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"{'Terrain':<10} {'Robot':<10} {'N':>4} {'SR':>8} {'Vel':>8} {'3patch':>8}")
    for terrain in terrains:
        for robot in robots:
            row = summary["cells"].get(f"{terrain}_{robot}")
            if row is None:
                continue
            print(
                f"{terrain:<10} {robot:<10} {row['episodes']:>4} "
                f"{row['success_rate']:>8.4f} {row['average_horizontal_speed_mps']:>8.4f} "
                f"{row['all_three_patches_completed_rate']:>8.4f}"
            )
    if errors and not args.allow_incomplete:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
