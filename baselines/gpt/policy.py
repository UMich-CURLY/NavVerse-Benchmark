"""Episode-scoped GPT policy through the Codex Python SDK."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image


ACTIONS = ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT")
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "step_size": {
            "type": "number",
            "minimum": 0,
            "maximum": 180,
            "description": (
                "FORWARD distance in metres (0.1 to 5); turn angle in degrees "
                "(5 to 180); 0 for STOP."
            ),
        },
    },
    "required": ["action", "step_size"],
    "additionalProperties": False,
}
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "explored_regions": {"type": "array", "items": {"type": "string"}},
        "visual_landmarks": {"type": "array", "items": {"type": "string"}},
        "unexplored_routes": {"type": "array", "items": {"type": "string"}},
        "next_search_plan": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "goal",
        "explored_regions",
        "visual_landmarks",
        "unexplored_routes",
        "next_search_plan",
        "uncertainties",
    ],
    "additionalProperties": False,
}

DEVELOPER_INSTRUCTIONS = """You are a vision-language navigation policy being evaluated in a simulator.
Use only the episode instruction, RGB observations, and actions in this conversation.
Do not call tools, inspect files, browse, run code, or interact with the simulator.
Most turns ask you to choose one action and one step_size for the newest RGB
observation. step_size is a forward distance in metres for FORWARD, a rotation
angle in degrees for turns, and zero for STOP.
A turn may instead request a compact navigation-state summary in the supplied schema.
The robot can descend staircases, but it cannot climb stairs. Never plan or attempt
an upward stair route. If the goal appears to be upstairs, search only accessible
areas and do not assume the robot can reach it by climbing. Each action runs for
at most five seconds of simulator time; if a motion target is not reached, a new
observation follows so you can replan.
Success means completing the stated episode goal. For an object goal, the stated
STOP range is only a rough outer range, not a reason to stop. Once the requested
object is identified, keep approaching with cautious, short moves and fresh
observations until you are almost touching it (roughly 0.3 to 0.5 m from its
visible surface) and can clearly confirm it. Then call STOP. If an obstacle or
collision risk prevents closer approach, stop at the closest safe position. Never
push into or through the object. For a destination goal, stop only when the
instructed destination is reached. Do not stop early just to save the remaining
action budget.
The evaluation uses SPL: success multiplied by shortest-path distance divided by
the greater of shortest-path distance and traveled distance. You do not know the
exact shortest path, so use SPL as a planning objective: preserve the chance of
success while avoiding unnecessary distance, loops, and backtracking.
First search the most likely target-related areas using the instruction, current
view, and episode history, choosing an efficient route. Do not revisit checked
regions without new evidence. If the main likely areas have been checked without
finding the goal, expand systematically and inspect secondary rooms, side routes,
and occluded corners more carefully. Distinguish observed facts from guesses and
never invent a map, pose, or unseen route."""

COMPACT_PROMPT = """Preserve the navigation goal, explored regions, visual landmarks, routes that remain unexplored, likely next search areas, and uncertainty. Preserve the mobility constraint: the robot may descend stairs but cannot climb stairs, so exclude upward stair routes from the plan. For object goals, remember that the stated STOP range is only an outer range: approach cautiously until almost touching the visible object, then stop; do not collide with it. Preserve the most recent raw observations and actions when possible. Do not turn visual guesses into confirmed map facts."""

_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def parse_action_response(raw_response: str) -> tuple[str, float]:
    try:
        payload = json.loads(raw_response)
        if not isinstance(payload, dict) or set(payload) != {"action", "step_size"}:
            raise TypeError("expected exactly action and step_size")
        action = payload["action"]
        raw_step_size = payload["step_size"]
        if isinstance(raw_step_size, bool):
            raise TypeError("step_size must be numeric")
        step_size = float(raw_step_size)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"GPT returned an invalid structured action: {raw_response!r}"
        ) from exc
    if action not in ACTIONS:
        raise ValueError(f"GPT returned unsupported action {action!r}")
    if not np.isfinite(step_size):
        raise ValueError("GPT returned a non-finite step_size")
    if action == "FORWARD":
        if not 0.1 <= step_size <= 5.0:
            raise ValueError("FORWARD step_size must be in [0.1, 5] metres")
    elif action in {"TURN_LEFT", "TURN_RIGHT"}:
        if not 5.0 <= step_size <= 180.0:
            raise ValueError("turn step_size must be in [5, 180] degrees")
    elif step_size != 0:
        raise ValueError("STOP requires step_size=0")
    return action, step_size


def _image_data_url(image: bytes | np.ndarray) -> str:
    if isinstance(image, bytes):
        source = Image.open(io.BytesIO(image)).convert("RGB")
    else:
        source = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    source.thumbnail((512, 512))
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _usage_summary(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    latest = getattr(usage, "last", None) or usage
    summary = {
        name: int(value)
        for name in _USAGE_FIELDS
        if (value := getattr(latest, name, None)) is not None
    }
    context_window = getattr(usage, "model_context_window", None)
    if context_window is not None:
        summary["model_context_window"] = int(context_window)
    return summary


def _context_tokens(usage: Any) -> tuple[int | None, int | None]:
    if usage is None:
        return None, None
    latest = getattr(usage, "last", None) or usage
    used = getattr(latest, "total_tokens", None)
    if used is None:
        used = getattr(latest, "input_tokens", None)
    window = getattr(usage, "model_context_window", None)
    if used is None or window is None:
        return None, int(window) if window is not None else None
    return int(used), int(window)


class GPTPolicy:
    """One tool-denied Codex thread per episode, with VLN-aware compaction."""

    def __init__(
        self,
        model: str = "gpt-6-astra",
        reasoning_effort: str = "low",
        stop_range_m: float = 1.6,
        max_steps: int = 500,
        compact_at_fraction: float = 0.70,
        compact_timeout_s: float = 300.0,
    ):
        try:
            from openai_codex import ApprovalMode, Codex, ImageInput, Sandbox, TextInput
        except ImportError as exc:
            raise RuntimeError(
                "GPT support requires the optional dependencies in "
                "baselines/gpt/requirements.txt"
            ) from exc

        if not 0.5 <= float(compact_at_fraction) < 0.9:
            raise ValueError("compact_at_fraction must be in [0.5, 0.9)")
        if float(compact_timeout_s) <= 0:
            raise ValueError("compact_timeout_s must be positive")
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")

        self._ApprovalMode = ApprovalMode
        self._ImageInput = ImageInput
        self._Sandbox = Sandbox
        self._TextInput = TextInput
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.stop_range_m = float(stop_range_m)
        self.max_steps = int(max_steps)
        self.compact_at_fraction = float(compact_at_fraction)
        self.compact_timeout_s = float(compact_timeout_s)
        self._working_dir = tempfile.TemporaryDirectory(prefix="gpt-codex-vln-")
        self.codex = Codex()
        self._thread = None
        self._instruction = ""
        self._step = 0
        self._needs_compaction = False
        self._compaction_error: str | None = None
        self._compaction_trigger: dict[str, int] = {}
        self._pending_summary: dict[str, Any] | None = None
        self._episode_usage = {name: 0 for name in _USAGE_FIELDS}

    def _start_thread(self):
        return self.codex.thread_start(
            model=self.model,
            config={
                "model_reasoning_effort": self.reasoning_effort,
                "compact_prompt": COMPACT_PROMPT,
            },
            cwd=self._working_dir.name,
            developer_instructions=DEVELOPER_INSTRUCTIONS,
            ephemeral=True,
            approval_mode=self._ApprovalMode.deny_all,
            sandbox=self._Sandbox.read_only,
        )

    def _run_turn(self, inputs: list[Any], output_schema: dict[str, Any] | None = None):
        if self._thread is None:
            raise RuntimeError("A navigation episode must be reset before inference")
        started = time.perf_counter()
        result = self._thread.run(
            inputs,
            output_schema=output_schema,
            approval_mode=self._ApprovalMode.deny_all,
            cwd=self._working_dir.name,
            sandbox=self._Sandbox.read_only,
        )
        return result, time.perf_counter() - started

    def _record_usage(self, usage: Any) -> dict[str, int]:
        latest = _usage_summary(usage)
        for name in _USAGE_FIELDS:
            self._episode_usage[name] += latest.get(name, 0)
        return latest

    def _wait_for_compaction(self, thread, updated_at: int) -> None:
        deadline = time.monotonic() + self.compact_timeout_s
        saw_active = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Codex compaction did not finish within {self.compact_timeout_s:g} seconds"
                )
            state = thread.read().thread
            status = state.status.root.type
            if status == "active":
                saw_active = True
            elif status == "idle" and saw_active:
                return
            elif status == "systemError":
                raise RuntimeError("Codex thread entered a system error during compaction")
            elif (
                status == "idle"
                and state.updated_at > updated_at
                and self.compact_timeout_s - remaining >= 2.0
            ):
                return
            time.sleep(min(0.25, remaining))

    def _summarize_and_compact(self) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("Cannot compact without an active episode thread")
        trigger = self._compaction_trigger.copy()
        next_step = self._step + 1
        prompt = (
            "Create a concise navigation memory for the next context window. "
            f"The next action decision is step {next_step} of at most {self.max_steps}. "
            f"Episode goal: {self._instruction}\n"
            "Use the entire observation and action history. Summarize which visually "
            "distinct regions have been explored, landmarks and goal evidence, which "
            "routes appear unexplored, and where to search next and for what. The robot "
            "can descend stairs but cannot climb them; exclude routes that require "
            "upward stair travel. Separate "
            "observations from hypotheses; do not invent a floorplan or pose. The last "
            "action in the conversation has been executed by the simulator, but its "
            "visual result has not yet been sent. Use at most four entries per list, "
            "with each entry no longer than fifteen words."
        )
        summary_result, summary_latency_s = self._run_turn(
            [self._TextInput(prompt)], output_schema=SUMMARY_SCHEMA
        )
        summary_usage = self._record_usage(summary_result.usage)
        raw_summary = summary_result.final_response
        if not raw_summary:
            raise RuntimeError("Codex returned no navigation summary")
        try:
            summary = json.loads(raw_summary)
        except json.JSONDecodeError as exc:
            raise ValueError("Codex returned an invalid navigation summary") from exc

        compact_started = time.perf_counter()
        state = self._thread.read().thread
        status = state.status.root.type
        if status != "idle":
            raise RuntimeError(f"Cannot compact a GPT thread while its status is {status}")
        self._thread.compact()
        self._wait_for_compaction(self._thread, state.updated_at)
        compact_latency_s = time.perf_counter() - compact_started
        self._pending_summary = summary
        self._needs_compaction = False
        self._compaction_error = None
        self._compaction_trigger = {}
        return {
            "summary": summary,
            "summary_usage": summary_usage,
            "summary_latency_s": summary_latency_s,
            "compact_latency_s": compact_latency_s,
            "trigger": trigger,
        }

    def reset(self, instruction: str) -> None:
        self._instruction = str(instruction)
        if not self._instruction:
            raise ValueError("A navigation instruction is required before inference")
        self._thread = self._start_thread()
        self._step = 0
        self._needs_compaction = False
        self._compaction_error = None
        self._compaction_trigger = {}
        self._pending_summary = None
        self._episode_usage = {name: 0 for name in _USAGE_FIELDS}

    def act(self, image: bytes | np.ndarray, instruction: str | None = None) -> dict[str, Any]:
        if instruction is not None and str(instruction) != self._instruction:
            self.reset(str(instruction))
        if not self._instruction:
            raise ValueError("A navigation instruction is required before inference")
        if self._compaction_error is not None:
            raise RuntimeError(self._compaction_error)

        maintenance = {}
        if self._needs_compaction:
            try:
                maintenance = self._summarize_and_compact()
            except Exception as exc:
                self._compaction_error = f"GPT context compaction failed: {exc}"
                raise RuntimeError(self._compaction_error) from exc

        current_step = self._step + 1
        step_budget = (
            f"Current action decision: {current_step} of at most {self.max_steps}. "
            f"STOP range: about {self.stop_range_m:g} m is only an outer range for object goals; "
            "approach to near-touching before STOP, using short moves and fresh observations. "
            "Choose exactly one action for the newest RGB observation."
        )
        if self._step == 0:
            prompt = (
                f"Navigation goal: {self._instruction}\n"
                "Return exactly action and step_size. FORWARD step_size is a distance "
                "from 0.1 to 5 metres. For TURN_LEFT and TURN_RIGHT, step_size is an "
                "angle from 5 to 180 degrees; the action selects the direction. STOP "
                "uses step_size=0. Choose the travel "
                "distance from the visible free space and goal proximity; prefer short "
                "moves near obstacles or the target. The robot follows a smooth timed "
                "trajectory for at most 5 seconds of simulator time, then receives a "
                "new observation to replan.\n"
                f"{step_budget} Allowed actions: STOP, FORWARD, TURN_LEFT, TURN_RIGHT."
            )
        else:
            prompt = (
                "This is the next RGB observation in the same navigation episode. "
                "Use the retained observations and actions to assess progress, then "
                "choose one action and step_size for this current image. "
                "Each action runs for at most 5 seconds of simulator time; a motion "
                f"may be interrupted for a new observation. Use the same units and ranges as before. {step_budget}"
            )

        inputs = []
        reinjected_summary = self._pending_summary is not None
        if reinjected_summary:
            inputs.append(
                self._TextInput(
                    "Navigation memory carried across context compaction:\n"
                    + json.dumps(self._pending_summary, ensure_ascii=False, separators=(",", ":"))
                )
            )
        inputs.extend(
            [
                self._TextInput(prompt),
                self._ImageInput(url=_image_data_url(image)),
            ]
        )

        result, latency_s = self._run_turn(inputs, output_schema=ACTION_SCHEMA)
        usage = self._record_usage(result.usage)
        raw_response = result.final_response
        if not raw_response:
            raise RuntimeError("Codex returned no final response")
        action, step_size = parse_action_response(raw_response)

        if reinjected_summary:
            self._pending_summary = None
        self._step += 1
        used_tokens, context_window = _context_tokens(result.usage)
        if (
            used_tokens is not None
            and context_window is not None
            and context_window > 0
            and used_tokens >= self.compact_at_fraction * context_window
        ):
            self._needs_compaction = True
            self._compaction_trigger = {
                "context_tokens": used_tokens,
                "model_context_window": context_window,
            }

        return {
            "action": action,
            "step_size": step_size,
            "step": current_step,
            "max_steps": self.max_steps,
            "model": self.model,
            "runtime": "codex",
            "reasoning_effort": self.reasoning_effort,
            "raw_response": raw_response,
            "call_id": result.id,
            "latency_s": latency_s,
            "duration_ms": result.duration_ms,
            "usage": usage,
            "episode_usage": self._episode_usage.copy(),
            "cache_hit_ratio": (
                usage.get("cached_input_tokens", 0) / usage["input_tokens"]
                if usage.get("input_tokens", 0) > 0
                else 0.0
            ),
            "episode_cache_hit_ratio": (
                self._episode_usage["cached_input_tokens"]
                / self._episode_usage["input_tokens"]
                if self._episode_usage["input_tokens"] > 0
                else 0.0
            ),
            "context_management": {
                "compact_at_fraction": self.compact_at_fraction,
                "compaction_pending": self._needs_compaction,
                "summary_reinjected": reinjected_summary,
                "maintenance": maintenance,
                "trigger": maintenance.get("trigger", self._compaction_trigger.copy()),
                "next_trigger": self._compaction_trigger.copy() if maintenance else {},
            },
        }

    def close(self) -> None:
        self.codex.close()
        self._working_dir.cleanup()
