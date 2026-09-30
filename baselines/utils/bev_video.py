import json
import math
import os

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import numpy as np

from navverse.episode import find_episode_json
from PIL import Image, ImageDraw


BEV_IMAGE_CANDIDATES = ("bev_map_rgb.png", "nav_bev_map_rgb.png")
BEV_INFO_CANDIDATES = ("bev_map_info.txt", "nav_bev_map_info.txt")


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _find_existing(directory, names):
    for name in names:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


def _as_points(values):
    points = []
    for value in values or []:
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.shape == (4, 4):
            points.append([arr[0, 3], arr[1, 3], arr[2, 3]])
        elif arr.size >= 3:
            flat = arr.reshape(-1)
            points.append([flat[0], flat[1], flat[2]])
    if not points:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def _trajectory_pose_data(trajectory):
    steps = []
    points = []
    yaws = []
    for index, (step, value) in enumerate(_sorted_trajectory_items(trajectory)):
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.shape == (4, 4):
            points.append([arr[0, 3], arr[1, 3], arr[2, 3]])
            yaws.append(float(math.atan2(arr[1, 0], arr[0, 0])))
        elif arr.size >= 3:
            flat = arr.reshape(-1)
            points.append([flat[0], flat[1], flat[2]])
            yaws.append(float("nan"))
        else:
            continue
        steps.append(int(step) if str(step).isdigit() else index)
    if not points:
        return [], np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    return steps, np.asarray(points, dtype=float), np.asarray(yaws, dtype=float)


def _sorted_trajectory_items(trajectory):
    if not isinstance(trajectory, dict):
        return []
    return sorted(
        trajectory.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
    )


def _sorted_trajectory_points(trajectory):
    return _as_points(value for _, value in _sorted_trajectory_items(trajectory))


def _world_to_pixel(points, world_center, px_per_meter, width, height):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)
    pixel_x = (pts[:, 0] - world_center[0]) * px_per_meter + width / 2.0
    pixel_y = height / 2.0 - (pts[:, 1] - world_center[1]) * px_per_meter
    return np.column_stack([pixel_x, pixel_y])


def _resize_and_pad(image, size, color=(255, 255, 255)):
    target_w, target_h = size
    pil_image = Image.fromarray(image).convert("RGB")
    pil_image.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color)
    x0 = (target_w - pil_image.width) // 2
    y0 = (target_h - pil_image.height) // 2
    canvas.paste(pil_image, (x0, y0))
    return np.asarray(canvas)


def append_bev_panel_to_frames(frames, bev_image, panel_width=640):
    if not frames or bev_image is None:
        return frames
    return append_bev_panels_to_frames(frames, [bev_image] * len(frames), panel_width=panel_width)


def append_bev_panels_to_frames(frames, bev_images, panel_width=760, margin=8):
    if not frames or not bev_images:
        return frames
    output = []
    for index, frame in enumerate(frames):
        frame_image = Image.fromarray(frame).convert("RGB")
        canvas = Image.new("RGB", (frame_image.width + panel_width, frame_image.height), (255, 255, 255))
        canvas.paste(frame_image, (0, 0))

        panel = Image.new("RGB", (panel_width, frame_image.height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.line(
            [(frame_image.width, 0), (frame_image.width, frame_image.height)],
            fill=(210, 210, 210),
            width=2,
        )
        target_size = (max(1, panel_width - 2 * margin), max(1, frame_image.height - 2 * margin))
        bev_image = bev_images[min(index, len(bev_images) - 1)]
        bev_panel = _resize_and_pad(bev_image, target_size)
        panel.paste(Image.fromarray(bev_panel), (margin, margin))
        canvas.paste(panel, (frame_image.width, 0))
        output.append(np.asarray(canvas))
    return output


class BEVVideoRenderer:
    def __init__(
        self,
        episode_folder="navverse_data/episodes",
        episode_data_root=None,
        output_size=(744, 1072),
    ):
        root = _repo_root()
        if episode_data_root is None:
            episode_data_root = os.path.join(root, "navverse_data", "episode_data")
        if not os.path.isabs(episode_folder):
            episode_folder = os.path.join(root, episode_folder)
        if not os.path.isabs(episode_data_root):
            episode_data_root = os.path.join(root, episode_data_root)
        self.episode_folder = episode_folder
        self.episode_data_root = episode_data_root
        self.output_size = output_size
        self.minimum_crop_span_m = 50.0
        self.scene_cache = {}
        self.episode_cache = {}
        self.missing_scenes = set()

    def load_scene(self, scene_id):
        if scene_id in self.scene_cache:
            return self.scene_cache[scene_id]
        if scene_id in self.missing_scenes:
            return None

        scene_dir = os.path.join(self.episode_data_root, scene_id)
        image_path = _find_existing(scene_dir, BEV_IMAGE_CANDIDATES)
        info_path = _find_existing(scene_dir, BEV_INFO_CANDIDATES)
        if image_path is None or info_path is None:
            self.missing_scenes.add(scene_id)
            return None

        try:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            info = _read_json(info_path)
        except Exception:
            self.missing_scenes.add(scene_id)
            return None
        height, width = image.shape[:2]
        scene = {
            "image": image,
            "world_center": np.asarray(info["world_center"], dtype=float),
            "px_per_meter": float(info["px_per_meter"]),
            "width": int(info.get("image_width", width)),
            "height": int(info.get("image_height", height)),
        }
        self.scene_cache[scene_id] = scene
        return scene

    def load_episode(self, scene_id, episode_id):
        key = (scene_id, str(episode_id))
        if key in self.episode_cache:
            return self.episode_cache[key]

        path = find_episode_json(self.episode_folder, scene_id)
        episode = None
        try:
            if path is None:
                raise FileNotFoundError(scene_id)
            data = _read_json(path)
            for candidate in data:
                if str(candidate.get("episode_id")) == str(episode_id):
                    episode = candidate
                    break
        except Exception:
            episode = None
        self.episode_cache[key] = episode
        return episode

    def _episode_points(self, episode):
        if not episode:
            return {
                "start": np.zeros((0, 3), dtype=float),
                "targets": np.zeros((0, 3), dtype=float),
                "reference_paths": [],
            }
        goals = episode.get("goals", [])
        reference_paths = []
        for goal in goals:
            path = _as_points(goal.get("reference_path") or [])
            if len(path):
                reference_paths.append(path)
        return {
            "start": _as_points([episode.get("start_position")]),
            "targets": _as_points(goal.get("location") for goal in goals),
            "reference_paths": reference_paths,
        }

    def _expand_bounds_to_output_aspect(self, left_top, right_bottom):
        span = np.maximum(right_bottom - left_top, 1.0)
        target_w, target_h = self.output_size
        target_aspect = max(1.0, float(target_w)) / max(1.0, float(target_h))
        current_aspect = span[0] / span[1]
        center = (left_top + right_bottom) / 2.0
        if current_aspect > target_aspect:
            span[1] = span[0] / target_aspect
        else:
            span[0] = span[1] * target_aspect
        return center - span / 2.0, center + span / 2.0

    def _crop_bounds(self, pixels, scene_shape):
        height, width = scene_shape
        non_empty = [points for points in pixels.values() if len(points)]
        if not non_empty:
            return 0, 0, width, height

        base = np.vstack(non_empty)
        min_xy = np.nanmin(base, axis=0)
        max_xy = np.nanmax(base, axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        px_per_meter = float(getattr(self, "_last_px_per_meter", 0.0) or 0.0)
        padding = np.maximum(span * 0.25, px_per_meter * 0.5)
        left_top = min_xy - padding
        right_bottom = max_xy + padding
        minimum_span = max(1.0, px_per_meter * self.minimum_crop_span_m)
        crop_span = np.maximum(right_bottom - left_top, minimum_span)
        crop_center = (left_top + right_bottom) / 2.0
        left_top = crop_center - crop_span / 2.0
        right_bottom = crop_center + crop_span / 2.0

        left_top, right_bottom = self._expand_bounds_to_output_aspect(left_top, right_bottom)
        left = int(np.floor(left_top[0]))
        top = int(np.floor(left_top[1]))
        right = int(np.ceil(right_bottom[0]))
        bottom = int(np.ceil(right_bottom[1]))
        return left, top, right, bottom

    @staticmethod
    def _crop_with_padding(image, bounds):
        left, top, right, bottom = bounds
        crop_width = max(1, right - left)
        crop_height = max(1, bottom - top)
        height, width = image.shape[:2]
        corners = np.asarray([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]])
        background = np.median(corners, axis=0).astype(image.dtype)
        crop = np.empty((crop_height, crop_width, 3), dtype=image.dtype)
        crop[:] = background

        source_left = max(0, left)
        source_top = max(0, top)
        source_right = min(width, right)
        source_bottom = min(height, bottom)
        if source_right > source_left and source_bottom > source_top:
            dest_left = source_left - left
            dest_top = source_top - top
            crop[
                dest_top:dest_top + source_bottom - source_top,
                dest_left:dest_left + source_right - source_left,
            ] = image[source_top:source_bottom, source_left:source_right]
        return crop

    def _prepare_context(self, scene_id, episode_id, trajectory):
        scene = self.load_scene(scene_id)
        if scene is None:
            return None
        episode = self.load_episode(scene_id, episode_id)
        episode_points = self._episode_points(episode)
        trajectory_steps, trajectory_points, trajectory_yaws = _trajectory_pose_data(trajectory)

        self._last_px_per_meter = scene["px_per_meter"]
        pixels = {
            "trajectory": _world_to_pixel(
                trajectory_points[:, :2],
                scene["world_center"],
                scene["px_per_meter"],
                scene["width"],
                scene["height"],
            ),
            "start": _world_to_pixel(
                episode_points["start"][:, :2],
                scene["world_center"],
                scene["px_per_meter"],
                scene["width"],
                scene["height"],
            ),
            "targets": _world_to_pixel(
                episode_points["targets"][:, :2],
                scene["world_center"],
                scene["px_per_meter"],
                scene["width"],
                scene["height"],
            ),
        }
        reference_pixels = [
            _world_to_pixel(
                path[:, :2],
                scene["world_center"],
                scene["px_per_meter"],
                scene["width"],
                scene["height"],
            )
            for path in episode_points["reference_paths"]
        ]
        crop_pixels = dict(pixels)
        crop_pixels["reference"] = (
            np.vstack(reference_pixels)
            if reference_pixels
            else np.zeros((0, 2), dtype=float)
        )
        bounds = self._crop_bounds(crop_pixels, scene["image"].shape[:2])
        left, top, _, _ = bounds
        crop = self._crop_with_padding(scene["image"], bounds)
        shifted = {key: value - np.asarray([left, top], dtype=float) for key, value in pixels.items()}
        shifted["reference_paths"] = [
            path - np.asarray([left, top], dtype=float)
            for path in reference_pixels
        ]

        return {
            "crop": crop,
            "shifted": shifted,
            "trajectory_steps": trajectory_steps,
            "trajectory_yaws": trajectory_yaws,
        }

    @staticmethod
    def _fallback_yaw(past, future):
        points = past
        if len(points) < 2 and len(future) >= 2:
            points = future[:2]
        if len(points) < 2:
            return 0.0
        delta = points[-1] - points[-2]
        if np.linalg.norm(delta) < 1e-6 and len(future) >= 2:
            delta = future[1] - future[0]
        if np.linalg.norm(delta) < 1e-6:
            return 0.0
        return math.atan2(-float(delta[1]), float(delta[0]))

    @staticmethod
    def _draw_pose_marker(ax, xy, yaw):
        cone_vertices = np.asarray([
            [0.0, 0.0],
            [1.0, 0.48],
            [1.0, -0.48],
            [0.0, 0.0],
        ])
        cone = Path(cone_vertices).transformed(Affine2D().rotate(yaw))
        ax.scatter(
            [float(xy[0])],
            [float(xy[1])],
            s=1800,
            marker=cone,
            facecolor="#00b4d8",
            edgecolor="#0077b6",
            linewidth=1.0,
            alpha=0.28,
            zorder=7,
        )

        cursor_vertices = np.asarray([
            [1.0, 0.0],
            [-0.62, 0.56],
            [-0.26, 0.0],
            [-0.62, -0.56],
            [1.0, 0.0],
        ])
        cursor = Path(cursor_vertices).transformed(Affine2D().rotate(yaw))
        ax.scatter(
            [float(xy[0])],
            [float(xy[1])],
            s=650,
            marker=cursor,
            facecolor="white",
            edgecolor="white",
            linewidth=1.5,
            zorder=8,
        )
        ax.scatter(
            [float(xy[0])],
            [float(xy[1])],
            s=460,
            marker=cursor,
            facecolor="#0a84ff",
            edgecolor="#0b2545",
            linewidth=1.4,
            zorder=8,
            label="current",
        )

    @staticmethod
    def _draw_reference_paths(ax, reference_paths):
        labeled = False
        for reference_path in reference_paths:
            if len(reference_path) < 2:
                continue
            ax.plot(
                reference_path[:, 0],
                reference_path[:, 1],
                color="#2ca02c",
                linewidth=2.5,
                linestyle=(0, (4, 4)),
                alpha=0.95,
                label=None if labeled else "GT path",
            )
            labeled = True

    def _draw_context(self, context, current_step=None):
        crop = context["crop"]
        shifted = context["shifted"]
        steps = context["trajectory_steps"]
        yaws = context["trajectory_yaws"]
        trajectory = shifted["trajectory"]
        if current_step is None or not steps:
            past_count = len(trajectory)
        else:
            past_count = sum(step <= current_step for step in steps)
            past_count = max(1, min(past_count, len(trajectory)))
        past = trajectory[:past_count]
        future = trajectory[past_count - 1:] if past_count > 0 else trajectory

        target_w, target_h = self.output_size
        dpi = 140
        fig = Figure(figsize=(target_w / dpi, target_h / dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.imshow(crop, aspect="equal")
        self._draw_reference_paths(ax, shifted["reference_paths"])
        if len(future) >= 2:
            ax.plot(
                future[:, 0],
                future[:, 1],
                color="#1f77b4",
                linewidth=3.4,
                solid_capstyle="round",
                alpha=0.25,
                label="future robot",
            )
        if len(past) >= 2:
            ax.plot(
                past[:, 0],
                past[:, 1],
                color="#1f77b4",
                linewidth=4.0,
                solid_capstyle="round",
                label="robot",
            )
        if len(shifted["start"]):
            ax.scatter(
                shifted["start"][:1, 0],
                shifted["start"][:1, 1],
                s=130,
                marker="o",
                color="#2ca02c",
                edgecolor="white",
                linewidth=1.5,
                label="start",
                zorder=5,
            )
        if len(shifted["targets"]):
            ax.scatter(
                shifted["targets"][:, 0],
                shifted["targets"][:, 1],
                s=180,
                marker="*",
                color="#d62728",
                edgecolor="white",
                linewidth=1.3,
                label="target",
                zorder=6,
            )
        if len(past):
            current_yaw = yaws[past_count - 1] if past_count - 1 < len(yaws) else float("nan")
            if not np.isfinite(current_yaw):
                current_yaw = self._fallback_yaw(past, future)
            self._draw_pose_marker(ax, past[-1], current_yaw)
        ax.set_xlim(0, max(1, crop.shape[1]))
        ax.set_ylim(max(1, crop.shape[0]), 0)
        ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = {}
            for handle, label in zip(handles, labels):
                unique.setdefault(label, handle)
            ax.legend(
                unique.values(),
                unique.keys(),
                loc="lower left",
                frameon=True,
                framealpha=0.85,
                facecolor="white",
                edgecolor="none",
                fontsize=8,
                markerscale=0.65,
            )
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        image = rgba[:, :, :3].copy()
        return image

    def render(self, scene_id, episode_id, trajectory, current_step=None):
        context = self._prepare_context(scene_id, episode_id, trajectory)
        if context is None:
            return None
        return self._draw_context(context, current_step=current_step)

    def render_sequence(self, scene_id, episode_id, trajectory, frame_steps):
        context = self._prepare_context(scene_id, episode_id, trajectory)
        if context is None:
            return None
        return [
            self._draw_context(context, current_step=step)
            for step in frame_steps
        ]
