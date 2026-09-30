import socket
import pickle
import threading
import struct
import time
import numpy as np
import cv2
import traceback
import json
import multiprocessing as mp

"""
# Socket Server

Set `host` to:
- on client side: "localhost" for local network, ip address of server for remote network
- on server side: "localhost" for local network, "0.0.0.0" for public network

Set `port` to 12300+thread_id, e.g. 12300 for the first thread, 12301 for the second thread, etc.

# Protocol Message Types

Client to Server messages are handled in `handle_client_connection()`.
Formatted as: "{message_type} {payload}", where message_type is one of the following:
- GET_SENSOR_DATA: call data_cb(), no payload
- GET_SIM_STATE: call data_cb() for a lightweight non-image simulator state payload
- DISCRETE_ACTION: {action_id}
- VEL: {vx, vy, vw}
- WAYPOINT: {x_list, y_list}
- TIMED_TRAJECTORY: {waypoints: [[x, y, yaw], ...], dt: seconds}
- STOP: {}
- [Other ACTIONS]: handled by specific server, payload is JSON

Server to Client messages are handled in `format_data()`.
Formatted as: "{payload_len}{payload}". Payload includes:
- rgb_images: Dict of RGB images keyed by camera topic
- depth_images: Dict of depth images keyed by camera topic
- tfs: Dict of transforms keyed by child_frame (e.g. camera transforms)
- pose: Pose
- timestamp_server_ns: Server-side timestamp when data was packed
- success: Whether the data was successfully generated
- message: Message from the server
- [Other INFO]: handled by specific server (e.g. instruction, episode_label, etc.)

"""

# --- Global frame counter for dummy data ---
# Use a list to pass by reference to threads, or a Lock with a simple int
frame_counter_lock = threading.Lock()
_frame_count = 0

def get_and_increment_frame_count():
    global _frame_count
    with frame_counter_lock:
        current_count = _frame_count
        _frame_count += 1
    return current_count


def _compress_rgb_image(rgb_image_np):
    rgb_image_np = rgb_image_np[:, :, ::-1]
    success, encoded_image = cv2.imencode('.jpg', rgb_image_np, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        print("Warning: RGB image JPG encoding failed.")
        return None, None
    return encoded_image.tobytes(), {
        "shape": rgb_image_np.shape,
        "dtype": str(rgb_image_np.dtype),
        "compressed_format": "jpg",
    }


def _compress_depth_image(depth_image_np):
    if depth_image_np.dtype not in [np.uint8, np.uint16]:
        print(f"Warning: Depth image dtype {depth_image_np.dtype} might not be perfectly preserved by PNG. "
              "Consider converting to uint16 if precision loss is acceptable, or use a different compression.")
    success, encoded_image = cv2.imencode('.png', depth_image_np)
    if not success:
        print("Warning: Depth image PNG encoding failed.")
        return None, None
    return encoded_image.tobytes(), {
        "shape": depth_image_np.shape,
        "dtype": str(depth_image_np.dtype),
        "compressed_format": "png",
    }


def compress_payload(payload_dict):
    """
    Compresses image fields in the payload dictionary.
    Supports single images ('rgb_image', 'depth_image') and dicts of images
    ('rgb_images', 'depth_images') keyed by camera topic.

    Args:
        payload_dict (dict): The dictionary containing sensor data.
                             Expected to have 'rgb_image' and/or 'depth_image'
                             as NumPy arrays.

    Returns:
        dict: A new dictionary with images replaced by their PNG-compressed bytes.
              Original metadata like 'shape' and 'dtype' for images are stored
              to aid in perfect reconstruction if needed, though cv2.imdecode
              with IMREAD_UNCHANGED often handles this for PNG.
    """
    compressed_dict = payload_dict.copy() # Work on a copy

    # Compress RGB Image
    if 'rgb_image' in compressed_dict and isinstance(compressed_dict['rgb_image'], np.ndarray):
        encoded_bytes, meta = _compress_rgb_image(compressed_dict['rgb_image'])
        if encoded_bytes is not None:
            compressed_dict['rgb_image'] = encoded_bytes
            compressed_dict['rgb_image_shape'] = meta["shape"]
            compressed_dict['rgb_image_dtype'] = meta["dtype"]
            compressed_dict['rgb_image_compressed_format'] = meta["compressed_format"]
        else:
            compressed_dict['rgb_image'] = None

    # Compress Depth Image
    if 'depth_image' in compressed_dict and isinstance(compressed_dict['depth_image'], np.ndarray):
        encoded_bytes, meta = _compress_depth_image(compressed_dict['depth_image'])
        if encoded_bytes is not None:
            compressed_dict['depth_image'] = encoded_bytes
            compressed_dict['depth_image_shape'] = meta["shape"]
            compressed_dict['depth_image_dtype'] = meta["dtype"]
            compressed_dict['depth_image_compressed_format'] = meta["compressed_format"]
        else:
            compressed_dict['depth_image'] = None

    # Compress RGB Images (dict)
    if isinstance(compressed_dict.get('rgb_images'), dict):
        rgb_images = {}
        for cam_name, image in compressed_dict['rgb_images'].items():
            if isinstance(image, np.ndarray):
                encoded_bytes, _ = _compress_rgb_image(image)
                rgb_images[cam_name] = encoded_bytes
            else:
                rgb_images[cam_name] = image
        compressed_dict['rgb_images'] = rgb_images

    # Compress Depth Images (dict)
    if isinstance(compressed_dict.get('depth_images'), dict):
        depth_images = {}
        for cam_name, image in compressed_dict['depth_images'].items():
            if isinstance(image, np.ndarray):
                encoded_bytes, _ = _compress_depth_image(image)
                depth_images[cam_name] = encoded_bytes
            else:
                depth_images[cam_name] = image
        compressed_dict['depth_images'] = depth_images

    return compressed_dict

def generate_dummy_data(server_name = "DummyServer"):
    """Generates a set of dummy sensor data."""
    frame_id = get_and_increment_frame_count()
    timestamp_ns = time.time_ns()

    # Dummy RGB Image
    rgb_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(rgb_arr, f"Standalone RGB Frame: {frame_id}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)
    cv2.rectangle(rgb_arr, (100 + frame_id % 200, 100), (200 + frame_id % 200, 200), (0,0,255), 3)


    # Dummy Depth Image (16-bit grayscale, representing millimeters)
    depth_arr = np.full((480, 640), 3000, dtype=np.uint16) # Base depth 3 meters
    cv2.circle(depth_arr, (320, 240), 50 + (frame_id % 50), int(1000 + (frame_id % 10) * 100) , -1)
    cv2.putText(depth_arr, f"D:{frame_id}", (30, 450), # Will be noisy on actual depth display
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5000), 1)


    # Dummy Pose
    pose_dict = {
        "header": {
            "stamp_sec": int(timestamp_ns / 1_000_000_000),
            "stamp_nanosec": int(timestamp_ns % 1_000_000_000),
            "frame_id": "dummy_odom"
        },
        "pose": {
            "position": {"x": 1.0 + frame_id * 0.02, "y": 0.5 - frame_id * 0.01, "z": 0.1},
            "orientation": {"x": 0.0, "y": 0.0, "z": np.sin(frame_id * 0.05 / 2.0), "w": np.cos(frame_id * 0.05 / 2.0)}
        }
    }

    return compress_payload(
    {
        "rgb_image": rgb_arr,
        "depth_image": depth_arr,
        "pose": pose_dict,
        "timestamp_server_ns": timestamp_ns, # Server-side timestamp when data was packed
        "success": True,
        "message": f"Dummy data generated successfully by {server_name}."
    }
    )

def format_data(
    rgb,
    depth,
    position,
    quat_xyzw,
    info,
    server_name="DummyServer",
    timestamp=None,
):
    if timestamp is None:
        timestamp_ns = time.time_ns()
    else:
        timestamp_ns = timestamp*1e9

    pose_dict = {
        "header": {
            "stamp_sec": int(timestamp_ns / 1_000_000_000),
            "stamp_nanosec": int(timestamp_ns % 1_000_000_000),
            "frame_id": "dummy_odom"
        },
        "pose": {
            "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
            "orientation": {"x": float(quat_xyzw[0]), "y": float(quat_xyzw[1]), "z": float(quat_xyzw[2]), "w": float(quat_xyzw[3])}
        }
    }

    payload = {
        "pose": pose_dict,
        "timestamp_server_ns": int(timestamp_ns),
        "success": True,
        "message": f"Dummy data generated successfully by {server_name}."
    }
    if isinstance(rgb, dict):
        payload["rgb_images"] = rgb
    else:
        payload["rgb_image"] = rgb
    if isinstance(depth, dict):
        payload["depth_images"] = depth
    else:
        payload["depth_image"] = depth
    payload.update(info)
    payload = {k: v for k, v in payload.items() if v is not None}

    return compress_payload(payload)

def handle_client_connection(client_socket, client_address, data_cb=None, action_cb=None, server_name="DummyServer"):
    """Handles a single client connection."""
    # print(f"[{time.strftime('%H:%M:%S')}] Accepted connection from {client_address}")
    try:
        # 1. Wait for a request from the client (e.g., "GET_SENSOR_DATA")
        request = client_socket.recv(8172) # Expecting a small request string
        if not request:
            # Benign case: readiness probes can connect and immediately close
            # without sending a request payload.
            return

        request_str = request.decode().strip()
        # print(f"[{time.strftime('%H:%M:%S')}] Received request: '{request_str}' from {client_address}")

        request_types = ["GET_SENSOR_DATA", "GET_SIM_STATE", "GET_EPISODE_LIST"]
        if request_str in request_types:
            # handle request
            try:
                data_payload = data_cb(request_str)
                if data_payload is None:
                    raise ValueError("data not ready")
            except Exception as e:
                data_payload = {
                    "success": False,
                    "message": str(e)
                }
            pickled_payload = pickle.dumps(data_payload)
            payload_len = len(pickled_payload)
            # Send the pickled data
            header = struct.pack('>Q', payload_len)
            client_socket.sendall(header+pickled_payload)
        else:
            # handle action
            header = request_str.split(' ')[0]
            payload_index = len(header)+1
            data = json.loads(request_str[payload_index:].strip())
            action_cb(header.strip(), data)

    except ConnectionResetError:
        print(f"[{time.strftime('%H:%M:%S')}] Client {client_address} reset the connection.")
    except socket.timeout:
        print(f"[{time.strftime('%H:%M:%S')}] Socket timeout for {client_address}.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Error handling client {client_address}: {e}")
        raise
    finally:
        client_socket.close()

def _stop_requested(stop_flag):
    if hasattr(stop_flag, "is_set"):
        return stop_flag.is_set()
    if hasattr(stop_flag, "value"):
        return bool(stop_flag.value)
    return bool(stop_flag)

def run_server(data_cb=lambda:None, action_cb=lambda:None, stop_flag = None, host = "localhost", port = 12300, server_name = "StandaloneSensorActionServer"):
    """Main server loop to listen for and handle connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow address reuse immediately after server closes
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    if stop_flag is None:
        stop_flag = mp.Value('b', False)
    
    try:
        print(f"{server_name} listening on {host}:{port}...")
        server_socket.bind((host, port))
        server_socket.listen(5) # Allow up to 5 queued connections
        server_socket.settimeout(0.2)

        while not _stop_requested(stop_flag):
            try:
                client_socket, client_address = server_socket.accept()
                if data_cb is not None:
                    handle_client_connection(client_socket, client_address, data_cb, action_cb, server_name=server_name) # Sequential handling
                else:
                    handle_client_connection(client_socket, client_address)
            except socket.timeout: # server_socket.accept() can timeout if set
                continue 
            except KeyboardInterrupt:
                print(f"\n{server_name} received KeyboardInterrupt. Shutting down.")
                break
            except Exception as e:
                print(f"Error in server accept loop: {e}")
                print(traceback.format_exc())
                continue # Or continue, depending on desired robustness

    finally:
        print(f"{server_name} is shutting down.")
        server_socket.close()

if __name__ == "__main__":
    # You might need to install OpenCV and NumPy if you haven't:
    # pip install opencv-python numpy
    run_server(data_cb=generate_dummy_data)
