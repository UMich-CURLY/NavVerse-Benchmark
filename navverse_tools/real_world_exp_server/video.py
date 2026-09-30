from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from baselines.utils.modern_video import (
    ACCENT,
    BACKGROUND,
    CARD,
    CARD_BORDER,
    MUTED,
    TEXT,
    _fit_instruction,
    _font,
    _paste_contained,
    _rounded_card,
    _text_width,
)


CANVAS_SIZE = (1920, 1080)
ACTION_LABELS = ("STOP", "FORWARD", "LEFT", "RIGHT")
ACTION_COLORS = {
    "STOP": "#ff675f",
    "MOVE_FORWARD": "#57d7ff",
    "MOVE_BACKWARD": "#ff9d5c",
    "TURN_LEFT": "#c8ff62",
    "TURN_RIGHT": "#c8ff62",
    "FOLLOW_TRAJECTORY": "#57d7ff",
}


class RealWorldEpisodeVideoRenderer:
    view_card = (32, 104, 1270, 1048)
    info_card = (1294, 104, 1888, 1048)

    def __init__(self, episode: dict[str, Any]):
        self.episode = episode
        self.steps = episode.get("steps") or []
        agent = (episode.get("client_metadata") or {}).get("agent") or {}
        self.agent_name = str(agent.get("label") or agent.get("id") or "Agent")
        self.base = self._build_base()

    def _build_base(self) -> Image.Image:
        canvas = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 26), "NAVVERSE", font=_font(34, bold=True), fill=TEXT)
        draw.text((230, 36), "PHONE EPISODE REPLAY", font=_font(18, bold=True), fill=ACCENT)
        episode_text = f"{self.agent_name}  ·  {self.episode['episode_id']}"
        episode_font = _font(20)
        draw.text(
            (1888 - _text_width(draw, episode_text, episode_font), 34),
            episode_text,
            font=episode_font,
            fill=MUTED,
        )
        _rounded_card(draw, self.view_card)
        _rounded_card(draw, self.info_card)
        draw.text((54, 123), "FIRST-PERSON VIEW", font=_font(17, bold=True), fill=TEXT)
        draw.text((1318, 123), "AGENT DECISION", font=_font(17, bold=True), fill=TEXT)
        return canvas

    def render(self, episode_dir: Path, step: dict[str, Any], index: int) -> np.ndarray:
        canvas = self.base.copy()
        image = Image.open(episode_dir / step["image"]).convert("RGB")
        _paste_contained(canvas, image, (48, 164, 1254, 1032))
        draw = ImageDraw.Draw(canvas)
        action = str(step.get("action") or "UNKNOWN")
        color = ACTION_COLORS.get(action, ACCENT)

        action_box = (1318, 164, 1864, 302)
        draw.rounded_rectangle(action_box, radius=18, fill="#0b1728", outline=CARD_BORDER, width=2)
        draw.text((1342, 184), f"STEP {index + 1:03d} / {len(self.steps):03d}", font=_font(17, bold=True), fill=MUTED)
        draw.text((1342, 224), action.replace("MOVE_", "").replace("TURN_", ""), font=_font(39, bold=True), fill=color)
        label = str(step.get("label") or "")
        if label and label.isascii():
            draw.text((1344, 274), label, font=_font(15), fill=TEXT)

        self._draw_instruction(draw)
        self._draw_probabilities(draw, step)
        self._draw_telemetry(draw, step)
        self._draw_outcome(draw, index)

        progress = (index + 1) / max(1, len(self.steps))
        draw.rounded_rectangle((48, 1016, 1254, 1028), radius=6, fill="#1b2b40")
        draw.rounded_rectangle((48, 1016, 48 + round(1206 * progress), 1028), radius=6, fill=ACCENT)
        return np.asarray(canvas)

    def _draw_instruction(self, draw: ImageDraw.ImageDraw) -> None:
        draw.text((1318, 332), "INSTRUCTION", font=_font(15, bold=True), fill=MUTED)
        font, lines, line_height = _fit_instruction(
            draw,
            self.episode.get("instruction", ""),
            510,
            150,
        )
        y = 368
        for line in lines:
            draw.text((1318, y), line, font=font, fill=TEXT)
            y += line_height

    def _draw_probabilities(self, draw: ImageDraw.ImageDraw, step: dict[str, Any]) -> None:
        draw.text((1318, 530), "ACTION PROBABILITIES", font=_font(15, bold=True), fill=MUTED)
        probabilities = ((step.get("agent_response") or {}).get("action_probs") or [])[:4]
        for row, label in enumerate(ACTION_LABELS):
            probability = float(probabilities[row]) if row < len(probabilities) else 0.0
            y = 574 + row * 54
            draw.text((1318, y), label, font=_font(15, bold=True), fill=TEXT)
            draw.rounded_rectangle((1436, y + 2, 1800, y + 22), radius=10, fill="#1b2b40")
            if probability > 0:
                draw.rounded_rectangle(
                    (1436, y + 2, 1436 + round(364 * min(1.0, probability)), y + 22),
                    radius=10,
                    fill=ACCENT,
                )
            draw.text((1812, y - 1), f"{probability:5.1%}", font=_font(14), fill=MUTED)

    def _draw_telemetry(self, draw: ImageDraw.ImageDraw, step: dict[str, Any]) -> None:
        response = step.get("agent_response") or {}
        orientation = (step.get("sensor") or {}).get("orientation") or {}
        heading = orientation.get("heading_deg")
        heading_text = "—" if not isinstance(heading, (int, float)) else f"{heading:.1f}°"
        latency = response.get("latency_ms")
        latency_text = "—" if not isinstance(latency, (int, float)) else f"{latency:.0f} ms"
        distance = ((step.get("sensor") or {}).get("imu_odometry") or {}).get("path_distance_m")
        distance_text = "—" if not isinstance(distance, (int, float)) else f"{distance:.2f} m"
        draw.text((1318, 806), "PHONE HEADING", font=_font(14, bold=True), fill=MUTED)
        draw.text((1318, 834), heading_text, font=_font(22, bold=True), fill=TEXT)
        draw.text((1500, 806), "IMU DISTANCE", font=_font(14, bold=True), fill=MUTED)
        draw.text((1500, 834), distance_text, font=_font(22, bold=True), fill=TEXT)
        draw.text((1682, 806), "LATENCY", font=_font(14, bold=True), fill=MUTED)
        draw.text((1682, 834), latency_text, font=_font(22, bold=True), fill=TEXT)

    def _draw_outcome(self, draw: ImageDraw.ImageDraw, index: int) -> None:
        termination = self.episode.get("termination") or {}
        is_last = index == len(self.steps) - 1
        status = "RUNNING"
        color = ACCENT
        if is_last and termination:
            status = "SUCCESS" if termination.get("success") else str(termination.get("reason", "FINISHED")).upper()
            color = "#c8ff62" if termination.get("success") else "#ff9d5c"
        draw.rounded_rectangle((1318, 904, 1864, 1024), radius=18, fill=CARD, outline=color, width=3)
        draw.text((1342, 925), "EPISODE STATUS", font=_font(14, bold=True), fill=MUTED)
        draw.text((1342, 958), status, font=_font(29, bold=True), fill=color)


def render_episode_video(
    episode_dir: Path,
    *,
    output_name: str = "episode.mp4",
    fps: int = 2,
) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    episode = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    steps = episode.get("steps") or []
    if not steps:
        raise ValueError("episode has no frames to render")
    output_path = episode_dir / output_name
    temp_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    temp_path.unlink(missing_ok=True)
    renderer = RealWorldEpisodeVideoRenderer(episode)
    started = time.perf_counter()
    writer = imageio.get_writer(
        temp_path,
        format="FFMPEG",
        mode="I",
        fps=max(1, int(fps)),
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=8,
        ffmpeg_params=["-preset", "veryfast", "-movflags", "+faststart"],
    )
    try:
        for index, step in enumerate(steps):
            frame = renderer.render(episode_dir, step, index)
            writer.append_data(frame)
            if index == len(steps) - 1:
                for _ in range(max(1, int(fps) * 2 - 1)):
                    writer.append_data(frame)
    except Exception:
        writer.close()
        temp_path.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(temp_path, output_path)
    repeated_final_frames = max(1, int(fps) * 2 - 1)
    return {
        "path": output_name,
        "frames": len(steps),
        "fps": max(1, int(fps)),
        "duration_s": (len(steps) + repeated_final_frames) / max(1, int(fps)),
        "render_elapsed_s": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a NavVerse-style phone episode replay")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output-name", default="episode.mp4")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(render_episode_video(args.episode_dir, output_name=args.output_name, fps=args.fps)))


if __name__ == "__main__":
    main()
