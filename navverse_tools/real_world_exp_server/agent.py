from __future__ import annotations

import base64
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any


ACTION_ALIASES = {
    "FORWARD": "MOVE_FORWARD",
    "MOVE_FORWARD": "MOVE_FORWARD",
    "LEFT": "TURN_LEFT",
    "TURN_LEFT": "TURN_LEFT",
    "TURN_LEFT_SMALL": "TURN_LEFT",
    "RIGHT": "TURN_RIGHT",
    "TURN_RIGHT": "TURN_RIGHT",
    "TURN_RIGHT_SMALL": "TURN_RIGHT",
    "BACKWARD": "MOVE_BACKWARD",
    "MOVE_BACKWARD": "MOVE_BACKWARD",
    "STOP": "STOP",
}


class AgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class GuidanceConfig:
    forward_distance_m: float = 0.25
    backward_distance_m: float = 0.25
    turn_degrees: float = 30.0


class DemoAgent:
    def __init__(self, actions: list[str], guidance: GuidanceConfig):
        if not actions:
            raise ValueError("Demo action list must not be empty")
        self.actions = actions
        self.guidance = guidance
        self._episode_steps: dict[str, int] = {}

    def act(
        self,
        *,
        image_bytes: bytes,
        instruction: str,
        episode_id: str,
        reset: bool,
        hfov: float,
        observation: dict[str, Any] | None = None,
        depth_bytes: bytes | None = None,
        confidence_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        del image_bytes, instruction, hfov, observation, depth_bytes, confidence_bytes
        if reset:
            self._episode_steps[episode_id] = 0
        index = self._episode_steps.get(episode_id, 0)
        self._episode_steps[episode_id] = index + 1
        raw = {"action": self.actions[min(index, len(self.actions) - 1)], "demo": True}
        return normalize_agent_response(raw, self.actions, self.guidance)

    def reset(self) -> bool:
        self._episode_steps.clear()
        return True


class HTTPAgent:
    def __init__(
        self,
        *,
        url: str,
        protocol: str,
        action_map: list[str],
        timeout_s: float,
        guidance: GuidanceConfig,
        reset_url: str | None = None,
    ):
        self.url = url
        self.protocol = protocol
        self.action_map = action_map
        self.timeout_s = timeout_s
        self.guidance = guidance
        self.reset_url = reset_url
        self._lock = threading.Lock()

    def act(
        self,
        *,
        image_bytes: bytes,
        instruction: str,
        episode_id: str,
        reset: bool,
        hfov: float,
        observation: dict[str, Any] | None = None,
        depth_bytes: bytes | None = None,
        confidence_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        del episode_id
        started = time.perf_counter()
        try:
            with self._lock:
                if self.protocol == "json":
                    response = self._post_json(
                        image_bytes,
                        instruction,
                        reset,
                        hfov,
                        observation,
                        depth_bytes,
                        confidence_bytes,
                    )
                else:
                    response = self._post_multipart(
                        image_bytes,
                        instruction,
                        reset,
                        hfov,
                        observation,
                        depth_bytes,
                        confidence_bytes,
                    )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentError(f"Agent request failed: {exc}") from exc
        response["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return normalize_agent_response(response, self.action_map, self.guidance)

    def reset(self) -> bool:
        if not self.reset_url:
            return False
        request = urllib.request.Request(
            self.reset_url,
            data=b"",
            headers={"Accept": "application/json"},
            method="POST",
        )
        try:
            with self._lock:
                result = self._read_json(request)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentError(f"Agent reset failed: {exc}") from exc
        return result.get("ok") is True

    def _post_json(
        self,
        image_bytes: bytes,
        instruction: str,
        reset: bool,
        hfov: float,
        observation: dict[str, Any] | None,
        depth_bytes: bytes | None,
        confidence_bytes: bytes | None,
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "instruction": instruction,
                "reset": reset,
                "hfov": hfov,
                "observation": observation,
                "depth_png_base64": (
                    base64.b64encode(depth_bytes).decode("ascii") if depth_bytes else None
                ),
                "confidence_png_base64": (
                    base64.b64encode(confidence_bytes).decode("ascii")
                    if confidence_bytes
                    else None
                ),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        return self._read_json(request)

    def _post_multipart(
        self,
        image_bytes: bytes,
        instruction: str,
        reset: bool,
        hfov: float,
        observation: dict[str, Any] | None,
        depth_bytes: bytes | None,
        confidence_bytes: bytes | None,
    ) -> dict[str, Any]:
        if self.protocol == "longnav":
            fields = {"instr_or_goal": instruction} if reset else {}
            file_field = "file"
        else:
            fields = {
                "instruction": instruction,
                "reset": "true" if reset else "false",
                "hfov": str(hfov),
                "depth_scale_m": "10.0",
            }
            file_field = "rgb"
            if observation:
                fields["observation_json"] = json.dumps(observation, separators=(",", ":"))
                depth_scale = observation.get("depth", {}).get("scale_m")
                if isinstance(depth_scale, (int, float)):
                    fields["depth_scale_m"] = str(depth_scale)
        extra_files = {}
        if self.protocol != "longnav" and depth_bytes:
            extra_files["depth"] = ("depth.png", "image/png", depth_bytes)
        if self.protocol != "longnav" and confidence_bytes:
            extra_files["confidence"] = ("confidence.png", "image/png", confidence_bytes)
        body, content_type = encode_multipart(
            fields,
            file_field,
            image_bytes,
            extra_files=extra_files,
        )
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": content_type, "Accept": "application/json"},
            method="POST",
        )
        return self._read_json(request)

    def _read_json(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise AgentError(f"Agent returned HTTP {exc.code}: {detail}") from exc
        try:
            result = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AgentError("Agent did not return valid JSON") from exc
        if not isinstance(result, dict):
            raise AgentError("Agent response must be a JSON object")
        return result


def encode_multipart(
    fields: dict[str, str],
    file_field: str,
    image_bytes: bytes,
    *,
    extra_files: dict[str, tuple[str, str, bytes]] | None = None,
) -> tuple[bytes, str]:
    boundary = f"----NavVersePhone{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    files = {file_field: ("observation.jpg", "image/jpeg", image_bytes), **(extra_files or {})}
    for name, (filename, content_type, content) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def normalize_agent_response(
    response: dict[str, Any], action_map: list[str], guidance: GuidanceConfig
) -> dict[str, Any]:
    if response.get("trajectory") is not None:
        trajectory = _valid_trajectory(response["trajectory"])
        target_x, target_y = trajectory[-1]
        return {
            "action": "FOLLOW_TRAJECTORY",
            "label": "沿箭头方向移动",
            "target_delta_deg": round(math.degrees(math.atan2(target_y, target_x)), 2),
            "distance_m": round(math.hypot(target_x, target_y), 3),
            "trajectory": trajectory,
            "agent_response": response,
        }

    raw_action = response.get("action")
    if raw_action is None and response.get("action_id") is not None:
        try:
            action_id = int(response["action_id"])
            raw_action = action_map[action_id]
        except (TypeError, ValueError, IndexError) as exc:
            raise AgentError(f"Invalid action_id in agent response: {response}") from exc
    if not isinstance(raw_action, str):
        raise AgentError(f"Agent returned neither action/action_id nor trajectory: {response}")
    canonical = ACTION_ALIASES.get(raw_action.strip().upper())
    if canonical is None:
        raise AgentError(f"Unsupported agent action {raw_action!r}")

    labels = {
        "MOVE_FORWARD": "向前移动",
        "MOVE_BACKWARD": "向后移动",
        "TURN_LEFT": "向左转",
        "TURN_RIGHT": "向右转",
        "STOP": "停止",
    }
    deltas = {
        "MOVE_FORWARD": 0.0,
        "MOVE_BACKWARD": 180.0,
        "TURN_LEFT": -guidance.turn_degrees,
        "TURN_RIGHT": guidance.turn_degrees,
        "STOP": 0.0,
    }
    distances = {
        "MOVE_FORWARD": guidance.forward_distance_m,
        "MOVE_BACKWARD": guidance.backward_distance_m,
        "TURN_LEFT": 0.0,
        "TURN_RIGHT": 0.0,
        "STOP": 0.0,
    }
    return {
        "action": canonical,
        "label": labels[canonical],
        "target_delta_deg": deltas[canonical],
        "distance_m": distances[canonical],
        "agent_response": response,
    }


def _valid_trajectory(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise AgentError("Agent trajectory must be a non-empty list")
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise AgentError("Every trajectory point must contain x and y")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise AgentError("Trajectory coordinates must be numbers") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise AgentError("Trajectory coordinates must be finite")
        result.append([x, y])
    return result
