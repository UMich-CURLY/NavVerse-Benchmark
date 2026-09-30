import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#07111f"
CARD = "#101d30"
CARD_BORDER = "#223550"
MUTED = "#91a4bd"
TEXT = "#f5f8fc"
ACCENT = "#57d7ff"


@dataclass(frozen=True)
class ModernVideoSample:
    pov_jpeg: bytes
    third_person_jpeg: bytes | None
    step: int
    action: str


def _font(size, bold=False):
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / family,
        Path("/usr/share/fonts/dejavu") / family,
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _encode_jpeg(image, quality=88):
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=False,
        subsampling=1,
    )
    return buffer.getvalue()


def capture_modern_video_sample(pov, third_person, step, action):
    return ModernVideoSample(
        pov_jpeg=_encode_jpeg(pov),
        third_person_jpeg=None if third_person is None else _encode_jpeg(third_person),
        step=int(step),
        action=str(action),
    )


def _text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in str(text or "").splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if _text_width(draw, candidate, font) <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def _fit_instruction(draw, instruction, max_width, max_height):
    for font_size in range(27, 11, -1):
        font = _font(font_size)
        lines = _wrap_text(draw, instruction, font, max_width)
        line_height = font_size + 9
        if len(lines) * line_height <= max_height:
            return font, lines, line_height
    font = _font(11)
    lines = _wrap_text(draw, instruction, font, max_width)
    return font, lines, 16


def _rounded_card(draw, box, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=CARD, outline=CARD_BORDER, width=2)


def _compact_action(action):
    value = str(action).strip()
    lower = value.lower()
    if "waypoints" in lower:
        value = value.split(",", 1)[0]
    elif lower.startswith("vx:"):
        value = "VELOCITY"
    return value.upper().replace("TURN_LEFT_SMALL", "LEFT · 15°").replace(
        "TURN_RIGHT_SMALL", "RIGHT · 15°"
    )


def _paste_contained(canvas, image, box, background="#091526"):
    left, top, right, bottom = box
    width = max(1, right - left)
    height = max(1, bottom - top)
    panel = Image.new("RGB", (width, height), background)
    source = image.convert("RGB")
    scale = min(width / source.width, height / source.height)
    source = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    canvas.paste(panel, (left, top))


class ModernVideoRenderer:
    def __init__(self, instruction, episode_label, agent_name, view_sizes=None):
        self.instruction = str(instruction or "")
        self.episode_label = str(episode_label or "unknown")
        self.agent_name = str(agent_name or "agent")
        self.base = None
        self.view_boxes = {}
        if view_sizes:
            self._build_base(view_sizes)

    def _build_base(self, view_sizes):
        gap = 4
        ratios = {name: width / height for name, (width, height) in view_sizes.items()}
        view_height = min(600, int((1600 - gap * (len(ratios) - 1)) / sum(ratios.values())))
        widths = {name: max(1, round(view_height * ratio)) for name, ratio in ratios.items()}
        width = sum(widths.values()) + gap * (len(widths) - 1)
        width = (width + 7) // 8 * 8
        text_canvas = Image.new("RGB", (width, 1))
        text_draw = ImageDraw.Draw(text_canvas)
        self.instruction_font = _font(20)
        lines = _wrap_text(text_draw, self.instruction, self.instruction_font, width - 24)
        footer_height = 16 + 27 * len(lines) if self.instruction else 0
        height = (64 + view_height + footer_height + 7) // 8 * 8
        canvas = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        labels = {"pov": "FIRST PERSON", "third": "THIRD PERSON", "bev": "BEV / TRAJECTORY"}
        left = 0
        for name, panel_width in widths.items():
            draw.text((left + 10, 43), labels[name], font=_font(13, bold=True), fill=MUTED)
            self.view_boxes[name] = (left, 64, left + panel_width, 64 + view_height)
            left += panel_width + gap
        for index, line in enumerate(lines if self.instruction else []):
            draw.text((12, 72 + view_height + index * 27), line, font=self.instruction_font, fill=TEXT)
        self.base = canvas

    def render(self, sample, bev_image=None):
        views = {"pov": Image.open(io.BytesIO(sample.pov_jpeg))}
        if sample.third_person_jpeg is not None:
            views["third"] = Image.open(io.BytesIO(sample.third_person_jpeg))
        if bev_image is not None:
            views["bev"] = Image.fromarray(np.asarray(bev_image, dtype=np.uint8))
        if self.base is None:
            self._build_base({name: image.size for name, image in views.items()})
        canvas = self.base.copy()
        for name, box in self.view_boxes.items():
            if name in views:
                _paste_contained(canvas, views[name], box)
            else:
                self._draw_placeholder(canvas, box, "Frame unavailable")
        draw = ImageDraw.Draw(canvas)
        status = f"STEP {sample.step:03d}  ·  {_compact_action(sample.action)}"
        title = f"NavVerse  ·  {self.agent_name}  ·  {self.episode_label}"
        font = _font(18, bold=True)
        for size in range(18, 8, -1):
            font = _font(size, bold=True)
            if _text_width(draw, title + "    " + status, font) <= canvas.width - 24:
                break
        draw.text((12, 11), title, font=font, fill=TEXT)
        draw.text((canvas.width - 12 - _text_width(draw, status, font), 11), status, font=font, fill=ACCENT)
        return np.asarray(canvas)

    @staticmethod
    def _draw_placeholder(canvas, box, message):
        left, top, right, bottom = box
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(box, fill="#091526")
        font = _font(18)
        width = _text_width(draw, message, font)
        draw.text(
            (left + (right - left - width) // 2, top + (bottom - top) // 2 - 10),
            message,
            font=font,
            fill=MUTED,
        )


def save_modern_video(
    output_path,
    samples,
    bev_images,
    instruction,
    episode_label,
    agent_name,
    fps,
):
    if not samples:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    temp_path.unlink(missing_ok=True)
    view_sizes = {"pov": Image.open(io.BytesIO(samples[0].pov_jpeg)).size}
    third_sample = next((sample for sample in samples if sample.third_person_jpeg is not None), None)
    if third_sample is not None:
        view_sizes["third"] = Image.open(io.BytesIO(third_sample.third_person_jpeg)).size
    first_bev = next((frame for frame in bev_images if frame is not None), None) if bev_images else None
    if first_bev is not None:
        view_sizes["bev"] = (first_bev.shape[1], first_bev.shape[0])
    renderer = ModernVideoRenderer(instruction, episode_label, agent_name, view_sizes)
    start_time = time.perf_counter()
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
        for index, sample in enumerate(samples):
            bev = None if not bev_images else bev_images[min(index, len(bev_images) - 1)]
            writer.append_data(renderer.render(sample, bev))
    except Exception:
        writer.close()
        temp_path.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(temp_path, output_path)
    return {
        "path": str(output_path),
        "frames": len(samples),
        "elapsed_s": time.perf_counter() - start_time,
        "fps": max(1, int(fps)),
    }
