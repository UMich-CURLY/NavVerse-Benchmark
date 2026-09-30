import socket
import pickle
import json
import time
import numpy as np
import cv2
import struct

def recv_all(sock, n):
    """Helper function to receive n bytes from a socket."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None # Connection closed
        data.extend(packet)
    return data

def _decode_rgb_image(encoded_bytes):
    np_arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
    rgb_image_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if rgb_image_np is None:
        return None
    return rgb_image_np[:, :, ::-1]


def _decode_depth_image(encoded_bytes):
    np_arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def decompress_payload(compressed_payload_dict):
    """
    Decompresses 'rgb_image' and 'depth_image' in the payload dictionary
    if they were PNG-compressed.

    Args:
        compressed_payload_dict (dict): The dictionary received from the server,
                                        potentially with compressed image data as bytes.

    Returns:
        dict: A new dictionary with image bytes replaced by NumPy arrays.
    """
    decompressed_dict = compressed_payload_dict.copy()

    # Decompress RGB Image
    if isinstance(decompressed_dict.get('rgb_image'), bytes):
        encoded_bytes = decompressed_dict['rgb_image']
        rgb_image_np = _decode_rgb_image(encoded_bytes)
        if rgb_image_np is not None:
            decompressed_dict['rgb_image'] = rgb_image_np
            # Optionally, verify against stored shape/dtype if they were sent
            # stored_shape = decompressed_dict.get('rgb_image_shape')
            # stored_dtype = decompressed_dict.get('rgb_image_dtype')
            # if stored_shape and rgb_image_np.shape != stored_shape:
            #     print(f"Warning: Decompressed RGB shape {rgb_image_np.shape} differs from original {stored_shape}")
            # if stored_dtype and str(rgb_image_np.dtype) != stored_dtype:
            #     print(f"Warning: Decompressed RGB dtype {rgb_image_np.dtype} differs from original {stored_dtype}")
        else:
            print("Warning: RGB image decoding failed.")
            decompressed_dict['rgb_image'] = None # Or handle error
        # Clean up compression-specific keys
        decompressed_dict.pop('rgb_image_compressed_format', None)
        decompressed_dict.pop('rgb_image_shape', None)
        decompressed_dict.pop('rgb_image_dtype', None)


    # Decompress Depth Image
    if decompressed_dict.get('depth_image_compressed_format') == 'png' and \
       isinstance(decompressed_dict.get('depth_image'), bytes):
        encoded_bytes = decompressed_dict['depth_image']
        # cv2.IMREAD_UNCHANGED is crucial for 16-bit depth images
        depth_image_np = _decode_depth_image(encoded_bytes)
        if depth_image_np is not None:
            decompressed_dict['depth_image'] = depth_image_np
        else:
            print("Warning: Depth image PNG decoding failed.")
            decompressed_dict['depth_image'] = None
        # Clean up compression-specific keys
        decompressed_dict.pop('depth_image_compressed_format', None)
        decompressed_dict.pop('depth_image_shape', None)
        decompressed_dict.pop('depth_image_dtype', None)

    # Decompress RGB Images (dict)
    if isinstance(decompressed_dict.get('rgb_images'), dict):
        rgb_images = {}
        for cam_name, value in decompressed_dict['rgb_images'].items():
            if isinstance(value, bytes):
                rgb_image_np = _decode_rgb_image(value)
                if rgb_image_np is None:
                    print(f"Warning: RGB image decoding failed for {cam_name}.")
                rgb_images[cam_name] = rgb_image_np
            else:
                rgb_images[cam_name] = value
        decompressed_dict['rgb_images'] = rgb_images

    # Decompress Depth Images (dict)
    if isinstance(decompressed_dict.get('depth_images'), dict):
        depth_images = {}
        for cam_name, value in decompressed_dict['depth_images'].items():
            if isinstance(value, bytes):
                depth_image_np = _decode_depth_image(value)
                if depth_image_np is None:
                    print(f"Warning: Depth image decoding failed for {cam_name}.")
                depth_images[cam_name] = depth_image_np
            else:
                depth_images[cam_name] = value
        decompressed_dict['depth_images'] = depth_images

    return decompressed_dict

def send_message(msg_type, msg, host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5.0) # Timeout for connection and operations
    retry_count = 0
    max_retries = 10
    while retry_count < max_retries:
        try:
            client_socket.connect((host, port))
            break
        except Exception as e:
            print(f"Waiting for server ({host}:{port}) to be ready: {e}")
            retry_count += 1
            time.sleep(0.01)
    if retry_count >= max_retries:
        print(f"Failed to connect to server ({host}:{port}) after {max_retries} retries")
        return
    # print("Connected to server.")
    message = msg_type+" "+json.dumps(msg)
    client_socket.sendall(message.encode())
    client_socket.close()

def request_data(host, port, request_type="GET_SENSOR_DATA", verbose=False):
    # print(f"\n[{time.strftime('%H:%M:%S')}] Attempting to connect to {host}:{port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5.0) # Timeout for connection and operations
    client_socket.connect((host, port))
    # print("Connected to server.")

    # 1. Send request
    start_time = time.time()
    client_socket.sendall(request_type.encode())
    end_time = time.time()
    if verbose:
        print(f"[Request] Time: {end_time - start_time:.3f}s")

    # 2. Receive the length of the pickled data (8 bytes, unsigned long long)
    start_time = time.time()
    header = recv_all(client_socket, 8)
    end_time = time.time()
    if verbose:
        print(f"[Recv Len] Time: {end_time - start_time:.3f}s")
    if not header:
        print("Connection closed by server before sending data length.")
        return None 
    

    # 3. Receive the pickled data
    start_time = time.time()
    msglen = struct.unpack('>Q', header)[0]
    pickled_payload = recv_all(client_socket, msglen)
    client_socket.close()
    end_time = time.time()
    if verbose:
        print(f"[Recv] Time: {end_time - start_time:.3f}s")
    if not pickled_payload:
        print("Connection closed by server before sending full payload.")
        return None

    # 4. Deserialize
    start_time = time.time()
    payload = pickle.loads(pickled_payload)
    end_time = time.time()
    if verbose:
        print(f"[Deserialize] Time: {end_time - start_time:.3f}s")
    start_time = time.time()
    payload = decompress_payload(payload)
    end_time = time.time()
    if verbose:
        print(f"[Decompress] Time: {end_time - start_time:.3f}s")

    return payload


def request_sensor_data(host, port, verbose=False):
    return request_data(host, port, request_type="GET_SENSOR_DATA", verbose=verbose)

def request_planner_state(host, port, verbose=False):
    # print(f"\n[{time.strftime('%H:%M:%S')}] Attempting to connect to {host}:{port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5.0) # Timeout for connection and operations
    client_socket.connect((host, port))
    # print("Connected to server.")

    # 1. Send request
    client_socket.sendall(b"GET_PLANNER_STATE")
    if verbose:
        print(f"Sent request: {b'GET_PLANNER_STATE'.decode()}")

    # 2. Receive the length of the pickled data (8 bytes, unsigned long long)
    header = recv_all(client_socket, 8)
    if not header:
        print("Connection closed by server before sending data length.")
        return None 
    
    # 3. Receive the pickled data
    msglen = struct.unpack('>Q', header)[0]
    json_payload = recv_all(client_socket, msglen).decode()
    client_socket.close()
    if not json_payload:
        print("Connection closed by server before sending full payload.")
        return None

    # 4. Deserialize
    payload = json.loads(json_payload)
    return payload


def extract_tf_dict(payload):
    """tfs is a dict of {camera_name: {parent_frame: str, child_frame_id: str, transform: dict}}"""
    
    tf_payload = None
    if "tfs" in payload and isinstance(payload.get("tfs"), dict):
        tf_payload = payload.get("tfs")
        tf_dict = {}
        for camera_name, value in tf_payload.items():
            correct_format = True
            for key in ["parent_frame", "child_frame_id", "transform"]:
                if key not in value:
                    print(f"[SOCKET_CLIENT] {key} not found in {camera_name}")
                    correct_format = False
                    break
            if correct_format:
                tf_dict[camera_name] = value
        return tf_dict
    else:
        # print("[SOCKET_CLIENT] tfs not found")
        return {}


def match_tf_key(tf_dict, camera_name):
    if camera_name in tf_dict:
        return camera_name
    for key in tf_dict.keys():
        if key.endswith(camera_name) or camera_name.endswith(key):
            return key
    return None

def main():
    print("Agent Socket Client Started")
    host = "localhost"
    port = 12340
    while True:
        start_cycle_time = time.time()
        payload = None # Initialize payload to None for this cycle

        try:
            payload = request_sensor_data(host, port)

            # print("Successfully received and unpickled data.")
            
            # --- Process Data ---
            if payload and payload.get("success", False):
                rgb_image = payload.get("rgb_image")
                depth_image = payload.get("depth_image")
                pose = payload.get("pose")
                tfs = payload.get("tfs")
                server_timestamp_ns = payload.get("timestamp_server_ns")
                
                latency_ms = (time.time_ns() - server_timestamp_ns) / 1_000_000 if server_timestamp_ns else -1
                if(latency_ms>500):
                    print(f"  Server timestamp (ns): {server_timestamp_ns}, Approx E2E Latency: {latency_ms:.2f} ms")

                if rgb_image is not None:
                    print(f"  RGB Image: {rgb_image.shape}, dtype: {rgb_image.dtype}")
                else:
                    print("  RGB Image: None")

                if depth_image is not None:
                    print(f"  Depth Image: {depth_image.shape}, dtype: {depth_image.dtype}")
                    # Normalize depth for display (example: if 16UC1 in mm, display 0-5m)
                    if depth_image.dtype == np.uint16:
                        depth_display = np.clip(depth_image, 0, 5000) # 0 to 5 meters
                        depth_display = (depth_display / 5000.0 * 255).astype(np.uint8)
                    elif depth_image.dtype == np.float32: # Assuming meters
                        depth_display = np.clip(depth_image, 0, 5.0)
                        depth_display = (depth_display / 5.0 * 255).astype(np.uint8)
                    else:
                        print(f"  Cannot display depth image with dtype {depth_image.dtype} automatically.")
                else:
                    print("  Depth Image: None")
                if pose:
                    p = pose['pose']['position']
                    o = pose['pose']['orientation']
                    print(f"  Pose ({pose['header']['frame_id']} @ {pose['header']['stamp_sec']}.{pose['header']['stamp_nanosec']}):")
                    print(f"    Pos: x={p['x']:.2f}, y={p['y']:.2f}, z={p['z']:.2f}")
                    print(f"    Ori: x={o['x']:.2f}, y={o['y']:.2f}, z={o['z']:.2f}, w={o['w']:.2f}")
                else:
                    print("  Pose: None")
                
                if cv2.waitKey(1) & 0xFF == ord('q'): # Allow quitting via OpenCV window
                    break
            elif payload: # success == False or no success key
                print(f"  Server responded with an issue: {payload.get('message', 'Unknown error')}")
            else: # Should not happen if recv_all worked
                print("  Received empty or invalid payload from server.")


        except socket.timeout:
            print(f"Socket timeout during operation with {host}:{port}")
        except socket.error as e:
            print(f"Socket error: {e}")
        except pickle.UnpicklingError as e:
            print(f"Pickle error: {e}. Received data might be corrupt or not a pickle.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}", exc_info=True)
        finally:
            # Maintain 1Hz cycle
            elapsed_cycle_time = time.time() - start_cycle_time
            sleep_time = 1.0 - elapsed_cycle_time
            if sleep_time > 0:
                # print(f"Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

    cv2.destroyAllWindows()
    print("Agent Socket Client Shutting Down.")

if __name__ == '__main__':
    main()