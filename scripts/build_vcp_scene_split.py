#!/usr/bin/env python3
"""Build the canonical scene-disjoint NavVerse train/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


SOURCE_METADATA = {
    "grCommercial": {"domain": "indoor", "forced_task": "objnav"},
    "vcp": {"domain": "outdoor", "forced_task": None},
    "innout": {"domain": "innout", "forced_task": None},
}
DOMAINS = ("indoor", "outdoor", "innout")
TASKS = ("objnav", "placenav")
EXTRA_TRAIN_CITIES = {"rome", "whitehouse"}
PREVIOUSLY_HELD_OUT_CITIES = (
    "barcelona",
    "brussels",
    "denver",
    "frankfurt",
    "harvard",
    "lasvegas",
    "philadelphia",
)
PREVIOUSLY_HELD_OUT_INDOOR = ("grCommercial_scene1", "grCommercial_scene3")
BENCHMARK500_QUOTAS = {
    ("indoor", "objnav"): 160,
    ("outdoor", "objnav"): 45,
    ("outdoor", "placenav"): 115,
    ("innout", "objnav"): 45,
    ("innout", "placenav"): 135,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-root", type=Path, default=Path("navverse_data/episodes")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("episodes"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indoor-test-scenes",
        type=int,
        default=5,
        help="Number of indoor scenes assigned to test. Default: 5.",
    )
    parser.add_argument(
        "--city-test-scenes",
        type=int,
        default=10,
        help="Number of paired outdoor/InOut cities assigned to test. Default: 10.",
    )
    parser.add_argument(
        "--skip-benchmark500",
        action="store_true",
        help="Do not create the balanced 500-episode benchmark subset.",
    )
    return parser.parse_args()


def label_digest(lines: list[str]) -> str:
    payload = "" if not lines else "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "" if not lines else "\n".join(lines) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def load_scene_file(
    path: Path, source: str, metadata: dict, known_labels: set[str]
) -> tuple[str, list[dict]]:
    episodes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Expected a non-empty episode list: {path}")
    scene_ids = {episode.get("scene_id") for episode in episodes}
    if len(scene_ids) != 1 or None in scene_ids:
        raise ValueError(f"Expected one scene_id in {path}: {scene_ids}")
    scene_id = scene_ids.pop()
    records = []
    episode_ids = set()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if episode_id is None or episode_id in episode_ids:
            raise ValueError(f"Invalid or duplicate episode_id in {path}: {episode_id}")
        episode_ids.add(episode_id)
        label = f"{scene_id}_{episode_id}"
        if label in known_labels:
            raise ValueError(f"Duplicate episode label: {label}")
        known_labels.add(label)
        task = metadata["forced_task"] or str(episode.get("task_type", "")).lower()
        if task not in TASKS:
            raise ValueError(f"Unsupported task {task!r} in {path}")
        records.append(
            {
                "label": label,
                "scene": scene_id,
                "source": source,
                "domain": metadata["domain"],
                "task": task,
            }
        )
    return scene_id, records


def load_episodes(root: Path) -> tuple[list[dict], dict[str, set[str]]]:
    records = []
    source_scenes: dict[str, set[str]] = {}
    labels = set()
    for source, metadata in SOURCE_METADATA.items():
        paths = sorted((root / source).glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No episode JSON files found in {root / source}")
        source_scenes[source] = set()
        for path in paths:
            scene_id, scene_records = load_scene_file(path, source, metadata, labels)
            if scene_id in source_scenes[source]:
                raise ValueError(f"Duplicate scene_id in {source}: {scene_id}")
            source_scenes[source].add(scene_id)
            records.extend(scene_records)
    records.sort(key=lambda record: record["label"])
    return records, source_scenes


def choose_test_groups(
    source_scenes: dict[str, set[str]], seed: int, indoor_count: int, city_count: int
) -> tuple[set[str], set[str]]:
    vcp_cities = {scene.removeprefix("vcp_") for scene in source_scenes["vcp"]}
    innout_cities = {
        scene.removeprefix("innout_") for scene in source_scenes["innout"]
    }
    if vcp_cities != innout_cities:
        raise ValueError(
            "VCP and InOut city sets differ: "
            f"vcp_only={sorted(vcp_cities - innout_cities)}, "
            f"innout_only={sorted(innout_cities - vcp_cities)}"
        )
    eligible_cities = sorted(vcp_cities - EXTRA_TRAIN_CITIES)
    if not 0 < city_count < len(vcp_cities) or city_count > len(eligible_cities):
        raise ValueError("city-test-scenes must leave at least one city in train")
    test_cities = [
        city for city in PREVIOUSLY_HELD_OUT_CITIES if city in eligible_cities
    ][:city_count]
    remaining_cities = sorted(set(eligible_cities) - set(test_cities))
    rng = random.Random(seed)
    test_cities.extend(rng.sample(remaining_cities, city_count - len(test_cities)))

    indoor_scenes = sorted(source_scenes["grCommercial"])
    if not 0 < indoor_count < len(indoor_scenes):
        raise ValueError("indoor-test-scenes must leave at least one scene in train")
    test_indoor = [
        scene for scene in PREVIOUSLY_HELD_OUT_INDOOR if scene in indoor_scenes
    ][:indoor_count]
    remaining_indoor = sorted(set(indoor_scenes) - set(test_indoor))
    test_indoor.extend(
        rng.sample(remaining_indoor, indoor_count - len(test_indoor))
    )
    return set(test_cities), set(test_indoor)


def build_benchmark500(records: list[dict], seed: int) -> list[dict]:
    test_records = [record for record in records if record["split"] == "test"]
    rng = random.Random(seed + 500)
    selected = []
    for (domain, task), quota in BENCHMARK500_QUOTAS.items():
        candidates = [
            record
            for record in test_records
            if record["domain"] == domain and record["task"] == task
        ]
        if len(candidates) < quota:
            raise ValueError(
                f"Benchmark500 needs {quota} {domain}/{task} episodes, "
                f"but the test split has {len(candidates)}"
            )
        selected.extend(rng.sample(candidates, quota))
    selected.sort(key=lambda record: record["label"])
    return selected


def count_records(records: list[dict]) -> dict:
    by_domain_task = Counter(
        (record["domain"], record["task"]) for record in records
    )
    return {
        "scenes": len({record["scene"] for record in records}),
        "episodes": len(records),
        "tasks": dict(sorted(Counter(record["task"] for record in records).items())),
        "domains": dict(
            sorted(Counter(record["domain"] for record in records).items())
        ),
        "domain_tasks": {
            f"{domain}_{task}": by_domain_task[domain, task]
            for domain in DOMAINS
            for task in TASKS
            if by_domain_task[domain, task]
        },
    }


def assign_and_validate_splits(
    records: list[dict], test_cities: set[str], test_indoor: set[str]
) -> None:
    for record in records:
        group = (
            record["scene"]
            if record["domain"] == "indoor"
            else record["scene"].split("_", 1)[1]
        )
        test_groups = test_indoor if record["domain"] == "indoor" else test_cities
        record["split"] = "test" if group in test_groups else "train"
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    train_scenes = {record["scene"] for record in train}
    test_scenes = {record["scene"] for record in test}
    if train_scenes & test_scenes:
        raise ValueError("Scene leakage between train and test")
    train_labels = {record["label"] for record in train}
    test_labels = {record["label"] for record in test}
    if train_labels & test_labels or len(train_labels | test_labels) != len(records):
        raise ValueError("Episode leakage or incomplete split coverage")


def build_outputs(records: list[dict], benchmark_records: list[dict]) -> dict[str, list[str]]:
    outputs = {
        f"{split}_set.txt": [
            record["label"] for record in records if record["split"] == split
        ]
        for split in ("train", "test")
    }
    if not benchmark_records:
        return outputs
    outputs["benchmark500_set.txt"] = [
        record["label"] for record in benchmark_records
    ]
    for field, values in (("domain", DOMAINS), ("task", TASKS)):
        for value in values:
            outputs[f"benchmark500_{value}_set.txt"] = [
                record["label"]
                for record in benchmark_records
                if record[field] == value
            ]
    return outputs


def load_benchmark450(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing legacy benchmark: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 450 or len(set(labels)) != 450:
        raise ValueError(f"Expected 450 unique labels in {path}, found {len(labels)}")
    return labels


def main() -> None:
    args = parse_args()
    records, source_scenes = load_episodes(args.episodes_root)
    test_cities, test_indoor = choose_test_groups(
        source_scenes,
        args.seed,
        args.indoor_test_scenes,
        args.city_test_scenes,
    )
    assign_and_validate_splits(records, test_cities, test_indoor)

    benchmark_records = []
    if not args.skip_benchmark500:
        benchmark_records = build_benchmark500(records, args.seed)
    outputs = build_outputs(records, benchmark_records)
    benchmark450_path = args.output_dir / "benchmark450_set.txt"
    outputs[benchmark450_path.name] = load_benchmark450(benchmark450_path)

    for name, lines in outputs.items():
        write_lines(args.output_dir / name, lines)

    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "test")
    }
    manifest = {
        "version": 1,
        "seed": args.seed,
        "protocol": {
            "scene_disjoint": True,
            "indoor_scene_counts": {
                "train": len(source_scenes["grCommercial"] - test_indoor),
                "test": len(test_indoor),
            },
            "indoor_test_fraction": len(test_indoor)
            / len(source_scenes["grCommercial"]),
            "paired_city_scene_counts": {
                "train": len(source_scenes["vcp"]) - len(test_cities),
                "test": len(test_cities),
            },
            "invariant": "vcp_<city> and innout_<city> always use the same split",
            "extra_local_cities_forced_to_train": sorted(
                EXTRA_TRAIN_CITIES
                & {scene.removeprefix("vcp_") for scene in source_scenes["vcp"]}
            ),
            "selection": (
                "retain previously held-out scenes, then use a seeded sample to fill "
                "the requested test counts"
            ),
            "test_cities": sorted(test_cities),
            "test_indoor_scenes": sorted(test_indoor),
        },
        "sources": {
            source: {
                "domain": SOURCE_METADATA[source]["domain"],
                "scenes": len(scenes),
                "episodes": sum(1 for record in records if record["source"] == source),
            }
            for source, scenes in source_scenes.items()
        },
        "splits": {
            split: count_records(split_records[split]) for split in ("train", "test")
        },
        "benchmark500": (
            {
                **count_records(benchmark_records),
                "subset_of": "test_set.txt",
                "quotas": {
                    f"{domain}_{task}": quota
                    for (domain, task), quota in BENCHMARK500_QUOTAS.items()
                },
                "note": "The subset contains equal ObjNav and PlaceNav counts.",
            }
            if benchmark_records
            else None
        ),
        "benchmark450": {
            "episodes": len(outputs["benchmark450_set.txt"]),
            "source": "legacy benchmark protocol",
        },
        "outputs": {
            name: {"line_count": len(lines), "sha256": label_digest(lines)}
            for name, lines in sorted(outputs.items())
        },
    }
    manifest_path = args.output_dir / "split_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(json.dumps({**manifest["splits"], "benchmark500": manifest["benchmark500"]}, indent=2))


if __name__ == "__main__":
    main()
