import datetime
import json
import logging
import os
import time
from math import cos, sin
import multiprocessing as mp
import cv2
import numpy as np
import torch
import queue

from scipy.spatial.transform import Rotation as R
import foxglove
from foxglove import Channel, Schema
from foxglove.channels import (
    CompressedImageChannel,
    PointCloudChannel,
    FrameTransformChannel,
)
from foxglove.schemas import (
    Color,
    CubePrimitive,
    Duration,
    FrameTransform,
    FrameTransforms,
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    Quaternion,
    CompressedImage,
    SceneEntity,
    SceneUpdate,
    Timestamp,
    Vector3,
)
from foxglove.websocket import (
    Capability,
    ChannelView,
    Client,
    ClientChannel,
    ServerListener,
)

class AgentListener(ServerListener):
    def __init__(self) -> None:
        # Map client id -> set of subscribed topics
        self.subscribers: dict[int, set[str]] = {}

    def has_subscribers(self) -> bool:
        return len(self.subscribers) > 0

    def on_subscribe(
        self,
        client: Client,
        channel: ChannelView,
    ) -> None:
        """
        Called by the server when a client subscribes to a channel.
        We'll use this and on_unsubscribe to simply track if we have any subscribers at all.
        """
        print(f"Client {client} subscribed to channel {channel.topic}")
        self.subscribers.setdefault(client.id, set()).add(channel.topic)

    def on_unsubscribe(
        self,
        client: Client,
        channel: ChannelView,
    ) -> None:
        """
        Called by the server when a client unsubscribes from a channel.
        """
        print(f"Client {client} unsubscribed from channel {channel.topic}")
        self.subscribers[client.id].remove(channel.topic)
        if not self.subscribers[client.id]:
            del self.subscribers[client.id]

    def on_client_advertise(
        self,
        client: Client,
        channel: ClientChannel,
    ) -> None:
        """
        Called when a client advertises a new channel.
        """
        print(f"Client {client.id} advertised channel: {channel.id}")
        print(f"  Topic: {channel.topic}")
        print(f"  Encoding: {channel.encoding}")
        print(f"  Schema name: {channel.schema_name}")
        print(f"  Schema encoding: {channel.schema_encoding}")
        print(f"  Schema: {channel.schema!r}")

    def on_message_data(
        self,
        client: Client,
        client_channel_id: int,
        data: bytes,
    ) -> None:
        """
        This handler demonstrates receiving messages from the client.
        You can send messages from Foxglove app in the publish panel:
        https://docs.foxglove.dev/docs/visualization/panels/publish
        """
        print(f"Message from client {client.id} on channel {client_channel_id}")
        print(f"Data: {data!r}")

    def on_client_unadvertise(
        self,
        client: Client,
        client_channel_id: int,
    ) -> None:
        """
        Called when a client unadvertises a new channel.
        """
        print(f"Client {client.id} unadvertised channel: {client_channel_id}")


class FoxgloveVisualizer():
    def __init__(self, args=None, log_queue=None, host=None, port=None):
        if args is not None:
            host = args.foxglove_host
            port = args.foxglove_port
            if args.foxglove_port == 0:
                port = args.port + 1000
        if log_queue is None:
            self.log_queue = mp.Queue()
        else:
            self.log_queue = log_queue
        self.listener = AgentListener()
        self.server = foxglove.start_server(
            host=host,
            port=port,
            server_listener=self.listener,
            capabilities=[Capability.ClientPublish],
            supported_encodings=["json"],
        )
        self.channels = {}
        print(f"Foxglove server started on {host}:{port}")
    
    def run(self):
        try:
            log_data_dict = {}
            while not self.log_queue.empty():
                log_data = self.log_queue.get_nowait()
                log_data_dict[log_data['channel_name']] = log_data
            for channel_name, log_data in log_data_dict.items():
                self.log(log_data)
        except queue.Empty:
            time.sleep(0.01)
    def log(self, log_data):
        if log_data is None:
            return
        if log_data['type'] == 'image':
            self.log_image(log_data['channel_name'], log_data['image'], log_data.get('frame_id', 'robot'))
        elif log_data['type'] == 'json':
            self.log_json(log_data['channel_name'], log_data['json'])
        elif log_data['type'] == 'point_cloud':
            self.log_pc(log_data['channel_name'], log_data['points'], log_data['color'], log_data.get('pose', np.eye(4)), log_data.get('frame_id', 'robot'))
        elif log_data['type'] == 'tf':
            self.log_tf(log_data['channel_name'], log_data['pose'], log_data['parent_frame_id'], log_data['child_frame_id'])

    def log_image(self, channel_name, image, frame_id):
        if not channel_name in self.channels:
            self.channels[channel_name] = CompressedImageChannel(topic=f"/{channel_name}")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_jpg = cv2.imencode('.jpeg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        self.channels[channel_name].log(
            CompressedImage(
                data=img_jpg,
                format="jpeg",
                frame_id=frame_id,
            )
        )
    
    def log_json(self, channel_name, json_data):
        if not channel_name in self.channels:
            self.channels[channel_name] = Channel(topic=f"/{channel_name}")
        json_data = self.convert_data(json_data)
        self.channels[channel_name].log(json_data)
    
    def log_pc(self, channel_name, points, color, pose, frame_id):
        """
        Log a point cloud to Foxglove.
        
        Args:
            channel_name (str): Name of the channel/topic
            points (np.ndarray): Point cloud points of shape (n, 3) with x, y, z coordinates
            color (np.ndarray): RGB colors of shape (n, 3) with values in range [0, 255] or [0, 1]
        """
        if not channel_name in self.channels:
            self.channels[channel_name] = PointCloudChannel(topic=f"/{channel_name}")

        if points.shape[0]==0:
            print("[FOXGLOVE] no points to visualize, skipping")
            return
        
        # Ensure points and color are numpy arrays
        points = np.asarray(points, dtype=np.float32)
        color = np.asarray(color, dtype=np.float32)
        
        # Validate shapes
        assert points.shape[1] == 3, f"Points must be (n, 3), got {points.shape}"
        assert color.shape == points.shape, f"Color shape {color.shape} must match points shape {points.shape}"
        
        # Convert color to uint8, handling both [0, 1] and [0, 255] ranges
        if color.max() <= 1.0:
            color = (color * 255).astype(np.uint8)
        else:
            color = np.clip(color, 0, 255).astype(np.uint8)
        
        n_points = points.shape[0]
        
        # Define fields: x, y, z (float32), r, g, b, a (uint8)
        f32 = PackedElementFieldNumericType.Float32
        u8 = PackedElementFieldNumericType.Uint8
        
        fields = [
            PackedElementField(name="x", offset=0, type=f32),
            PackedElementField(name="y", offset=4, type=f32),
            PackedElementField(name="z", offset=8, type=f32),
            PackedElementField(name="red", offset=12, type=u8),
            PackedElementField(name="green", offset=13, type=u8),
            PackedElementField(name="blue", offset=14, type=u8),
            PackedElementField(name="alpha", offset=15, type=u8),
        ]
        
        # Pack data efficiently using a structured NumPy array (Vectorized)
        # Each point is 16 bytes: 12 (xyz as f4) + 4 (rgba as u1)
        dt = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')
        ])
        
        structured_array = np.empty(n_points, dtype=dt)
        structured_array['x'] = points[:, 0]
        structured_array['y'] = points[:, 1]
        structured_array['z'] = points[:, 2]
        structured_array['r'] = color[:, 0]
        structured_array['g'] = color[:, 1]
        structured_array['b'] = color[:, 2]
        structured_array['a'] = 255
        
        packed_data = structured_array.tobytes()

        if pose is not None:
            quat = R.from_matrix(pose[:3, :3]).as_quat()
            pos = Vector3(x=pose[0, 3], y=pose[1, 3], z=pose[2, 3])
            quat = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            pose = Pose(
                position=pos,
                orientation=quat,
            )
        
        # Create PointCloud message
        point_cloud = PointCloud(
            timestamp=Timestamp(sec=int(time.time()), nsec=0),
            frame_id=frame_id,
            point_stride=16,
            fields=fields,
            data=packed_data,
            pose=pose,
        )

        self.channels[channel_name].log(point_cloud)
    
    def log_tf(self, channel_name, pose, parent_frame_id, child_frame_id):
        if not channel_name in self.channels:
            self.channels[channel_name] = FrameTransformChannel(topic=f"/{channel_name}")
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        pos = Vector3(x=pose[0, 3], y=pose[1, 3], z=pose[2, 3])
        quat = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        self.channels[channel_name].log(FrameTransform(
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
            translation=pos,
            rotation=quat,
        ))
    
    def convert_data(self, data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_data(v) for v in data]
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data

import sys
from pathlib import Path
import re


class RuntimeFileHandler(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def emit(self, record):
        msg = self.format(record) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, msg.encode("utf-8"))
        finally:
            os.close(fd)


class RuntimeLogger(logging.LoggerAdapter):
    ENV_ENABLED = "NAVVERSE_RUNTIME_LOG_ENABLED"
    ENV_PATH = "NAVVERSE_RUNTIME_LOG_PATH"
    ENV_START_TS = "NAVVERSE_RUNTIME_LOG_START_TS"

    @staticmethod
    def _make_formatter():
        class _CompactFormatter(logging.Formatter):
            def format(self, record):
                timestamp = self.formatTime(record, self.datefmt)
                level = record.levelname
                if level in {"INFO", "DEBUG"}:
                    return f"[{timestamp}] {record.getMessage()}"
                return f"[{timestamp}] [{level}] {record.getMessage()}"

        return _CompactFormatter(datefmt="%H:%M:%S")

    def __init__(self, runtime_name="NavVerse", logger_name="NavVerse", log_file=None):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        has_console_handler = any(
            isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
            for handler in logger.handlers
        )
        if not has_console_handler:
            handler = logging.StreamHandler()
            handler.setFormatter(self._make_formatter())
            logger.addHandler(handler)
        logger.propagate = False
        super().__init__(logger, {})
        self.runtime_name = runtime_name
        self.runtime_enabled = os.environ.get(self.ENV_ENABLED, "0") == "1"
        self.runtime_path = os.environ.get(self.ENV_PATH)
        self.runtime_start_ts = float(os.environ.get(self.ENV_START_TS, time.time()))
        self.runtime_last_ts = None
        self._runtime_logger = None

    @classmethod
    def configure_runtime(cls, enabled=True, path=None):
        if path is None:
            path = os.path.join(os.getcwd(), "runtime.log")
        os.environ[cls.ENV_ENABLED] = "1" if enabled else "0"
        os.environ[cls.ENV_PATH] = path
        os.environ[cls.ENV_START_TS] = str(time.time())
        if enabled:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("elapsed_s | delta_s | source | step\n")

    def _get_runtime_logger(self):
        if not self.runtime_enabled or not self.runtime_path:
            return None
        if self._runtime_logger is not None:
            return self._runtime_logger
        logger = logging.getLogger(f"{self.logger.name}.runtime.{self.runtime_name}")
        if getattr(logger, "_runtime_path", None) != self.runtime_path:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            handler = RuntimeFileHandler(self.runtime_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger._runtime_path = self.runtime_path
        self._runtime_logger = logger
        return logger

    def step(self, step, level=logging.INFO):
        runtime_logger = self._get_runtime_logger()
        if runtime_logger is None:
            return
        now = time.time()
        delta = 0.0 if self.runtime_last_ts is None else now - self.runtime_last_ts
        self.runtime_last_ts = now
        runtime_logger.log(
            level,
            f"{now - self.runtime_start_ts:8.3f} | {delta:7.3f} | {self.runtime_name} | {step}",
        )

class Tee:
    """File-like object that duplicates writes to multiple streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            # Remove ANSI color codes before writing to log file
            clean_data = re.sub(r'\x1b\[[0-9;]*m', '', data)
            s.write(clean_data if hasattr(s, 'name') and s.name.endswith('.log') else data)
        # Flush so logs appear promptly
        for s in self.streams:
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        for stream in self.streams:
            if hasattr(stream, "fileno"):
                return stream.fileno()
        raise AttributeError("No underlying stream supports fileno()")


def setup_tee(log_path: str):
    """
    Redirect sys.stdout and sys.stderr so that everything printed goes to:
      1) the original terminal
      2) the given log file

    This affects all threads in this process.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # line-buffered file so lines appear quickly
    log_file = log_path.open("a", buffering=1, encoding="utf-8")

    # Save original stdout/stderr (the real terminal)
    orig_stdout = sys.__stdout__
    orig_stderr = sys.__stderr__

    # Replace sys.stdout / sys.stderr with a tee object
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)


def ensure_repo_log_symlink(repo_root, target_log_path):
    link_path = os.path.join(repo_root, "eval_nav_verse.log")
    target_abs_path = os.path.abspath(target_log_path)
    target_rel_path = os.path.relpath(target_abs_path, repo_root)
    if os.path.lexists(link_path):
        os.remove(link_path)
    os.symlink(target_rel_path, link_path)
    return link_path
