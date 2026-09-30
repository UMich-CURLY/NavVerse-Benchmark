from __future__ import annotations

import json
import os
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINATION_REASONS = {
    "success",
    "collision",
    "stuck",
    "timeout",
    "agent_stop",
    "user_abort",
}


class StoreError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_episode_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not value:
        value = datetime.now(timezone.utc).strftime("phone_%Y%m%d_%H%M%S")
    return value[:120]


class EpisodeStore:
    def __init__(self, root: Path, *, allow_concurrent: bool = False):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_concurrent = allow_concurrent
        self._lock = threading.RLock()

    def start(
        self,
        *,
        episode_id: str,
        instruction: str,
        hfov: float,
        client_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        with self._lock:
            if not instruction.strip():
                raise StoreError("instruction is required")
            identifier = safe_episode_id(episode_id)
            path = self.root / identifier
            if path.exists():
                raise StoreError(f"episode {identifier!r} already exists")
            active = self._active_ids()
            if active and not self.allow_concurrent:
                raise StoreError(f"episode {active[0]!r} is still active")
            (path / "frames").mkdir(parents=True)
            now = utc_now()
            episode = {
                "schema_version": 1,
                "episode_id": identifier,
                "instruction": instruction.strip(),
                "status": "active",
                "started_at": now,
                "updated_at": now,
                "hfov": float(hfov),
                "client_metadata": client_metadata or {},
                "steps": [],
                "termination": None,
            }
            self._write_json(path / "episode.json", episode)
            return deepcopy(episode)

    def add_step(
        self,
        *,
        episode_id: str,
        request_id: str,
        image_bytes: bytes,
        captured_at: str | None,
        sensor: dict[str, Any] | None,
        action: dict[str, Any],
        observation_assets: dict[str, bytes] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            episode = self._read(episode_id)
            if episode["status"] != "active":
                raise StoreError("episode is already finished")
            for step in episode["steps"]:
                if step["request_id"] == request_id:
                    return deepcopy(step)
            index = len(episode["steps"])
            relative_image = f"frames/{index:04d}.jpg"
            self._atomic_write(self.root / episode["episode_id"] / relative_image, image_bytes)
            asset_paths: dict[str, str] = {}
            for name, content in (observation_assets or {}).items():
                if name not in {"depth", "confidence"}:
                    raise StoreError(f"unsupported observation asset {name!r}")
                relative_asset = f"frames/{index:04d}.{name}.png"
                self._atomic_write(self.root / episode["episode_id"] / relative_asset, content)
                asset_paths[name] = relative_asset
            arkit = (sensor or {}).get("arkit")
            if isinstance(arkit, dict):
                if isinstance(arkit.get("intrinsics"), dict):
                    relative_intrinsics = f"frames/{index:04d}.intrinsics.json"
                    self._write_json(
                        self.root / episode["episode_id"] / relative_intrinsics,
                        arkit["intrinsics"],
                    )
                    asset_paths["intrinsics"] = relative_intrinsics
                if isinstance(arkit.get("pose"), dict):
                    relative_pose = f"frames/{index:04d}.pose.json"
                    self._write_json(
                        self.root / episode["episode_id"] / relative_pose,
                        arkit["pose"],
                    )
                    asset_paths["pose"] = relative_pose
            step = {
                "step_index": index,
                "request_id": request_id,
                "captured_at": captured_at,
                "received_at": utc_now(),
                "image": relative_image,
                "observation_assets": asset_paths,
                "sensor": sensor or {},
                **action,
            }
            episode["steps"].append(step)
            episode["updated_at"] = utc_now()
            self._write_episode(episode)
            return deepcopy(step)

    def finish(
        self,
        *,
        episode_id: str,
        reason: str,
        notes: str,
        sensor: dict[str, Any] | None,
        client_time: str | None,
    ) -> dict[str, Any]:
        with self._lock:
            if reason not in TERMINATION_REASONS:
                raise StoreError(f"unsupported termination reason {reason!r}")
            episode = self._read(episode_id)
            if episode["status"] == "finished":
                return deepcopy(episode)
            episode["status"] = "finished"
            episode["updated_at"] = utc_now()
            episode["termination"] = {
                "reason": reason,
                "success": reason == "success",
                "notes": notes.strip()[:1000],
                "client_time": client_time,
                "server_time": episode["updated_at"],
                "sensor": sensor or {},
            }
            self._write_episode(episode)
            result = {
                "episode_label": episode["episode_id"],
                "instruction": episode["instruction"],
                "success": reason == "success",
                "termination_reason": reason,
                "num_steps": len(episode["steps"]),
                "action_history": [step["action"] for step in episode["steps"]],
                "imu_odometry": (sensor or {}).get("imu_odometry", {}),
                "started_at": episode["started_at"],
                "finished_at": episode["updated_at"],
            }
            self._write_json(self.root / episode["episode_id"] / "result.json", result)
            return deepcopy(episode)

    def get(self, episode_id: str) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._read(episode_id))

    def update_result(self, episode_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            identifier = safe_episode_id(episode_id)
            path = self.root / identifier / "result.json"
            if not path.is_file():
                raise StoreError(f"episode {identifier!r} has no result")
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StoreError(f"episode {identifier!r} result is unreadable") from exc
            result.update(values)
            self._write_json(path, result)
            return deepcopy(result)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            summaries = []
            for manifest in sorted(self.root.glob("*/episode.json"), reverse=True):
                try:
                    episode = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                summaries.append(
                    {
                        "episode_id": episode.get("episode_id"),
                        "instruction": episode.get("instruction"),
                        "status": episode.get("status"),
                        "started_at": episode.get("started_at"),
                        "updated_at": episode.get("updated_at"),
                        "num_steps": len(episode.get("steps", [])),
                        "termination": episode.get("termination"),
                    }
                )
            return summaries[:50]

    def _active_ids(self) -> list[str]:
        return [row["episode_id"] for row in self.list() if row["status"] == "active"]

    def _read(self, episode_id: str) -> dict[str, Any]:
        identifier = safe_episode_id(episode_id)
        manifest = self.root / identifier / "episode.json"
        if not manifest.is_file():
            raise StoreError(f"episode {identifier!r} does not exist")
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"episode {identifier!r} is unreadable") from exc

    def _write_episode(self, episode: dict[str, Any]) -> None:
        self._write_json(self.root / episode["episode_id"] / "episode.json", episode)

    def _write_json(self, path: Path, value: Any) -> None:
        payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self._atomic_write(path, payload)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with open(temp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
