from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SplitInfo:
    name: str
    path: Path | None
    labels: list[str]
    unique_count: int
    duplicate_count: int
    scene_count: int
    scene_types: dict[str, int]
    task_types: dict[str, int]
    sha256: str | None
    missing_labels: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return bool(self.labels) and not self.errors and not self.missing_labels

    def as_dict(self) -> dict:
        return {
            "name": self.name, "path": str(self.path) if self.path else None, "count": len(self.labels),
            "unique_count": self.unique_count, "duplicate_count": self.duplicate_count,
            "scene_count": self.scene_count, "scene_types": self.scene_types, "task_types": self.task_types,
            "sha256": self.sha256, "missing_labels": self.missing_labels, "errors": self.errors, "valid": self.valid,
        }


def _read_episode_index(folder: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in sorted(folder.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for episode in payload:
            if not isinstance(episode, dict):
                continue
            scene_id, episode_id = episode.get("scene_id"), episode.get("episode_id")
            if scene_id is not None and episode_id is not None:
                index[f"{scene_id}_{episode_id}"] = episode
    return index


def _split_info(name: str, path: Path | None, labels: list[str], index: dict[str, dict]) -> SplitInfo:
    labels = [label.strip() for label in labels if label.strip() and not label.lstrip().startswith("#")]
    unique_labels = list(dict.fromkeys(labels))
    missing = [label for label in unique_labels if label not in index]
    scenes = [label.rsplit("_", 1)[0] for label in unique_labels]
    scene_types = Counter(scene.split("_", 1)[0] for scene in scenes)
    task_types = Counter(str(index[label].get("task_type")) for label in unique_labels if label in index and index[label].get("task_type"))
    return SplitInfo(
        name=name, path=path, labels=labels, unique_count=len(unique_labels), duplicate_count=len(labels) - len(unique_labels),
        scene_count=len(set(scenes)), scene_types=dict(sorted(scene_types.items())), task_types=dict(sorted(task_types.items())),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else None, missing_labels=missing,
        errors=[] if labels else ["empty split"],
    )


def discover_splits(episode_folder: str | Path) -> list[SplitInfo]:
    folder = Path(episode_folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Episode folder does not exist: {folder}")
    index = _read_episode_index(folder)
    splits = [
        _split_info(path.name.removesuffix("_set.txt"), path, path.read_text(encoding="utf-8").splitlines(), index)
        for path in sorted(folder.glob("*_set.txt"))
    ]
    splits.append(_split_info("all", None, sorted(index), index))
    return splits


def get_split(episode_folder: str | Path, name: str) -> SplitInfo:
    for split in discover_splits(episode_folder):
        if split.name == name:
            return split
    raise ValueError(f"Split set {name!r} was not found in {Path(episode_folder).resolve()}")


def resolve_split_episode_folder(episode_folder: str | Path, split: SplitInfo) -> Path:
    folder = Path(episode_folder).expanduser().resolve()
    labels = set(split.labels)
    candidates = []
    for child in sorted(path for path in folder.iterdir() if path.is_dir()):
        if labels.issubset(_read_episode_index(child)):
            candidates.append(child)
    return candidates[0] if len(candidates) == 1 else folder
