import datetime
import json
import logging
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
        log_data = None
        while True:
            try:
                log_data = self.log_queue.get_nowait()
                self.log(log_data)
            except queue.Empty:
                time.sleep(0.01)
                break
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
        
        # Define fields: x, y, z (float32), r, g, b (uint8)
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
        
        # Pack data: each point is 16 bytes (12 for xyz float32, 3 for rgba uint8)
        point_step = 16
        
        # Create structured array for efficient packing
        # Pack xyz as float32 and rgb as uint8
        xyz_bytes = points.astype(np.float32).tobytes()
        rgb_bytes = color.astype(np.uint8).tobytes()
        
        # Interleave xyz and rgb data
        packed_data = bytearray(n_points * point_step)
        for i in range(n_points):
            offset = i * point_step
            xyz_offset = i * 12  # 3 floats * 4 bytes
            rgb_offset = i * 3   # 3 uint8
            packed_data[offset:offset+12] = xyz_bytes[xyz_offset:xyz_offset+12]
            packed_data[offset+12:offset+15] = rgb_bytes[rgb_offset:rgb_offset+3]
            packed_data[offset+15:offset+16] = bytes([255])
        
        packed_data = bytes(packed_data)

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
            point_stride=point_step,
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