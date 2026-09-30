import io
import multiprocessing as mp
import math
import os
from concurrent.futures import ThreadPoolExecutor
import ray
import socket
import threading
from queue import Full, Empty
import signal
import time
import torch
import imageio
import numpy as np
from navverse.termination_cfg import DEFAULT_EPISODE_TIMEOUT_S
from navverse.utils.socket_client import request_data, send_message, extract_tf_dict, match_tf_key
from baselines.utils.bev_video import BEVVideoRenderer, append_bev_panels_to_frames
from baselines.utils.camera import Camera, warp_to_pi
from baselines.utils.logging import setup_tee
from baselines.utils.modern_video import capture_modern_video_sample, save_modern_video
from baselines.utils.vis import pad_frames_to_same_size, render_grid, text_args
import json


ON_DEMAND_RENDER_TIMEOUT_MESSAGE = "Timed out waiting for on-demand render on simulation thread"


class EvaluationWorker(mp.Process):
    def __init__(
        self,
        args,
        task_queue,
        result_queue,
        step_data_queue,
        foxglove_log_queue,
        progress_queue=None,
        episode_controller=None,
        daemon=False,
    ):
        super().__init__(name=f"EvalWorker{args.process_label}", daemon=daemon)
        self.args = args
        ctx = mp.get_context('spawn')
        self.reset_flag = ctx.Event()
        self.reset_flag.set()
        self.command_queue = ctx.Queue()
        self.evaluation_over = ctx.Event()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.progress_queue = progress_queue
        self.episode_controller = episode_controller
        
        # Proxy queues to handle Ray <-> Multiprocessing communication
        self.ray_step_data_queue = step_data_queue
        self.ray_foxglove_log_queue = foxglove_log_queue
        self.step_data_queue = ctx.Queue()
        self.foxglove_log_queue = ctx.Queue()
        
        # defer agent construction to child process to avoid pickling unpicklable objects (e.g., Open3D)
        self.agent = None
        # runtime variables
        self.initialized = False
        self.episode_label = None
        self.scene_id = None
        self.episode_id = None
        self.obs = {
            "timestamp": None,
        }
        self.last_timestamp = None
        self.action = None
        self.instruction = None
        self.instruction_type = None
        self.target_type = None
        self.camera = None
        self.wait_for_action = False
        self.wait_for_action_start_time = 0
        self.wait_for_action_start_sim_time = None
        self.wait_for_action_mode = None
        self.timed_action_duration_s = None
        self.waiting_for_stop_termination = False
        self.expected_episode = None
        self.wrong_episode_count = 0
        self.last_action_send_time = 0.0
        self.last_progress_report_time = 0.0
        self.last_status_log_time = 0.0
        self.last_action_status = None
        self.server_error_streak_message = None
        self.server_error_streak_count = 0
        self.action_history = {}
        self.trajectory = {}
        self.pose_history = []
        self.modern_video_samples = []
        self.parquet_rows = []
        self.forward_heading_yaw = None
        self.bev_renderer = None

    def _parquet_enabled(self):
        return bool(getattr(self.args, "save_parquet", False))

    @staticmethod
    def _jsonable(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): EvaluationWorker._jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [EvaluationWorker._jsonable(item) for item in value]
        return str(value)

    def _json_dumps(self, value):
        return json.dumps(self._jsonable(value), ensure_ascii=True)

    def _base_parquet_row(self, row_type):
        episode_id = self.episode_id
        return {
            "row_type": row_type,
            "worker_id": int(getattr(self.args, "process_label", -1)),
            "episode_label": str(self.episode_label or self.expected_episode or ""),
            "scene_id": "" if self.scene_id is None else str(self.scene_id),
            "episode_id": "" if episode_id is None else str(episode_id),
            "instruction": self.instruction,
            "instruction_type": self.instruction_type,
            "target_type": self.target_type,
        }

    def _append_parquet_row(self, row):
        if not self._parquet_enabled():
            return
        row["row_index"] = len(self.parquet_rows)
        self.parquet_rows.append(row)

    def _flatten_parquet_values(self, prefix, values):
        flattened = {}
        for key, value in (values or {}).items():
            column_name = f"{prefix}_{key}"
            if value is None or isinstance(value, (str, int, float, bool, np.generic)):
                flattened[column_name] = self._jsonable(value)
            else:
                flattened[f"{column_name}_json"] = self._json_dumps(value)
        return flattened

    @staticmethod
    def _array_to_npy_bytes(array):
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(array), allow_pickle=False)
        return buffer.getvalue()

    def _encode_parquet_image(self, image, image_kind):
        if image is None:
            return None
        array = np.asarray(image)
        if image_kind == "rgb" and array.ndim in (2, 3):
            rgb_array = array
            if rgb_array.dtype != np.uint8:
                rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
            try:
                import imageio.v3 as iio

                buffer = io.BytesIO()
                iio.imwrite(buffer, rgb_array, extension=".png")
                return "png", buffer.getvalue(), list(array.shape), str(array.dtype)
            except Exception as exc:
                self.log(f"Failed to encode {image_kind} image as PNG: {exc}", level="warning")
        return "npy", self._array_to_npy_bytes(array), list(array.shape), str(array.dtype)

    def _record_parquet_images(self, obs, agent_step):
        image_sources = [
            ("rgb", obs.get("rgb_dict") or {"primary": obs.get("rgb")}),
            ("depth", obs.get("depth_dict") or {"primary": obs.get("depth")}),
        ]
        for image_kind, image_dict in image_sources:
            if not isinstance(image_dict, dict):
                continue
            for camera_name, image in image_dict.items():
                encoded = self._encode_parquet_image(image, image_kind)
                if encoded is None:
                    continue
                encoding, image_bytes, shape, dtype = encoded
                row = self._base_parquet_row("image")
                row.update({
                    "agent_step": int(agent_step),
                    "sim_time": obs.get("sim_time"),
                    "sim_step": obs.get("sim_step"),
                    "step_dt": obs.get("step_dt"),
                    "timestamp": obs.get("timestamp"),
                    "camera_name": camera_name,
                    "image_kind": image_kind,
                    "image_encoding": encoding,
                    "image_shape_json": self._json_dumps(shape),
                    "image_dtype": dtype,
                    "image_bytes": image_bytes,
                })
                self._append_parquet_row(row)

    def _record_pose_sample(self, source):
        if not self._parquet_enabled() or not getattr(self, "obs", None):
            return
        position = self.obs.get("position")
        rotation = self.obs.get("rotation")
        if position is None or rotation is None:
            return
        agent_step = int(getattr(self.agent, "l_step", 0) or 0) if self.agent is not None else 0
        body_pose = getattr(self, "body_pose", None)
        pose_record = {
            "source": source,
            "agent_step": agent_step,
            "timestamp": self.obs.get("timestamp"),
            "sim_time": self.obs.get("sim_time"),
            "sim_step": self.obs.get("sim_step"),
            "step_dt": self.obs.get("step_dt"),
            "position": self._jsonable(position),
            "rotation": self._jsonable(rotation),
            "body_pose": self._jsonable(body_pose),
            "last_action_status": self._jsonable(self.last_action_status),
        }
        self.pose_history.append(pose_record)

        row = self._base_parquet_row("pose")
        row.update({
            "source": source,
            "agent_step": agent_step,
            "timestamp": pose_record["timestamp"],
            "sim_time": pose_record["sim_time"],
            "sim_step": pose_record["sim_step"],
            "step_dt": pose_record["step_dt"],
            "position_json": self._json_dumps(position),
            "rotation_json": self._json_dumps(rotation),
            "body_pose_json": self._json_dumps(body_pose),
            "last_action_status_json": self._json_dumps(self.last_action_status),
        })
        self._append_parquet_row(row)

    def record_parquet_step(self, action):
        if not self._parquet_enabled():
            return
        obs = self.obs or {}
        agent_step = int(getattr(self.agent, "l_step", 0) or 0)
        try:
            action_name = self.get_action_name(**action) if action is not None else None
        except Exception:
            action_name = str(action)

        row = self._base_parquet_row("step")
        row.update({
            "agent_step": agent_step,
            "timestamp": obs.get("timestamp"),
            "sim_time": obs.get("sim_time"),
            "sim_step": obs.get("sim_step"),
            "step_dt": obs.get("step_dt"),
            "primary_camera": obs.get("primary_camera"),
            "position_json": self._json_dumps(obs.get("position")),
            "rotation_json": self._json_dumps(obs.get("rotation")),
            "body_pose_json": self._json_dumps(getattr(self, "body_pose", None)),
            "metrics_json": self._json_dumps(obs.get("metrics", {})),
            "action_json": self._json_dumps(action),
            "action_name": action_name,
            "last_action_status_json": self._json_dumps(self.last_action_status),
        })
        if isinstance(action, dict) and "sg_vln_debug" in action:
            row["sg_vln_debug_json"] = self._json_dumps(action.get("sg_vln_debug"))
        if isinstance(action, dict) and "sg_vln_debug_error" in action:
            row["sg_vln_debug_error"] = str(action.get("sg_vln_debug_error"))
        row.update(self._flatten_parquet_values("metric", obs.get("metrics", {})))
        self._append_parquet_row(row)
        self._record_pose_sample("sensor_data")
        self._record_parquet_images(obs, agent_step)

    @staticmethod
    def _safe_episode_filename(episode_label):
        return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(episode_label))

    def _save_episode_parquet(self, result):
        if not self._parquet_enabled():
            return None
        row = self._base_parquet_row("episode_summary")
        row.update({
            "agent_step": int(getattr(self.agent, "l_step", 0) or 0),
            "summary_json": self._json_dumps(result),
            "action_history_json": self._json_dumps(self.action_history),
            "trajectory_json": self._json_dumps(self.trajectory),
            "pose_history_json": self._json_dumps(self.pose_history),
            "action_test_records_json": self._json_dumps(getattr(self.agent, "action_test_records", None)),
        })
        row.update(self._flatten_parquet_values("summary", result))
        self._append_parquet_row(row)

        import pandas as pd

        parquet_dir = f"{self.args.dump_location}/{self.args.exp_name}/parquet"
        os.makedirs(parquet_dir, exist_ok=True)
        episode_name = self._safe_episode_filename(self.episode_label or self.expected_episode or "unknown")
        parquet_path = f"{parquet_dir}/{episode_name}_{self.args.agent}.parquet"
        dataframe = pd.DataFrame(self.parquet_rows)
        for column_name in dataframe.columns:
            if dataframe[column_name].dtype != "object":
                continue
            non_null_values = dataframe[column_name].dropna()
            has_binary = non_null_values.map(lambda value: isinstance(value, (bytes, bytearray))).any()
            if not has_binary:
                dataframe[column_name] = dataframe[column_name].astype("string")
        dataframe.to_parquet(parquet_path, index=False, engine="pyarrow")
        self.log(f"Parquet saved to {parquet_path}")
        return parquet_path

    @staticmethod
    def _payload_episode_label(payload):
        if payload is None:
            return None
        scene_id = payload.get("scene_id")
        episode_id = payload.get("episode_id")
        if scene_id is None or episode_id is None:
            return None
        return f"{scene_id}_{episode_id}"

    def report_agent_progress(self, action=None, reason="step", force=False):
        now = time.time()
        interval_s = float(getattr(self.args, "wandb_progress_interval_s", 10.0) or 10.0)
        if not force and (now - self.last_progress_report_time) < interval_s:
            return
        self.last_progress_report_time = now
        if self.episode_controller is None or self.agent is None:
            return

        action_name = ""
        if action is not None:
            try:
                action_name = self.get_action_name(**action)
            except Exception:
                action_name = str(action)

        try:
            self.episode_controller.record_agent_progress.remote(
                self.args.process_label,
                agent_step=int(getattr(self.agent, "l_step", 0) or 0),
                episode_label=self.episode_label,
                scene_id=self.scene_id,
                last_action=action_name,
                last_message=reason,
            )
        except Exception as exc:
            self.log(f"Failed to report progress: {exc}", level="debug")

    def load_agent(self):
        from navverse.launcher.registry import instantiate_agent

        return instantiate_agent(
            self.args.agent,
            self.args,
            self.evaluation_over,
            self.step_data_queue,
            self.foxglove_log_queue,
        )
    
    def update_vis(self, action=None):
        action_name = None
        if action is not None:
            action_name = self.get_action_name(**action)
        if action_name is not None:
            self.action_history[self.agent.l_step] = action
        self.trajectory[self.agent.l_step] = self.body_pose.tolist()
        if (
            action_name is not None
            and getattr(self.args, "save_video", False)
            and getattr(self.args, "modern_obs_video", True)
            and self.args.backend == "isaac"
            and self.obs
            and "rgb_dict" in self.obs
            and self.obs["rgb_dict"].get("pov") is not None
        ):
            self.modern_video_samples.append(
                capture_modern_video_sample(
                    self.obs["rgb_dict"]["pov"],
                    self.obs["rgb_dict"].get("third_person"),
                    self.agent.l_step,
                    action_name,
                )
            )
        if not self.args.foxglove:
            return

        info_data = {
            "type": "json",
            "channel_name": "info",
            "json": {
                "step": self.agent.l_step,
            }
        }
        if action is not None:
            info_data["json"]["action"] = action
            info_data["json"]["action_name"] = action_name
        self._safe_queue_put(self.foxglove_log_queue, info_data, "foxglove_log")
        if hasattr(self.agent, "vis_frames"):
            for img_key, img_list in self.agent.vis_frames.items():
                if len(img_list) > 0:
                    img = img_list[-1]
                    img_data = {
                        "type": "image",
                        "channel_name": img_key,
                        "image": img
                    }
                    self._safe_queue_put(self.foxglove_log_queue, img_data, "foxglove_log")
        else:
            self.agent.vis_frames = {}
        if self.args.backend == "isaac":
            if (self.obs and "rgb_dict" in self.obs and action_name):
                # add a new line every 30 letters in instruction
                instruction = self.instruction
                instruction_lines = ["Instruction: "]
                for word in instruction.split():
                    if len(instruction_lines) == 0 or len(instruction_lines[-1]) + len(word) > 30:
                        instruction_lines.append(word)
                    else:
                        instruction_lines[-1] += f" {word}"
                instruction_text = [{"type": "text", "args": text_args, "text": line} for line in instruction_lines]
                vis_grid = [
                    {
                        "type": "text",
                        "args": text_args,
                        "text": f"Episode: {self.obs['episode_label']}"
                    },
                    *instruction_text,
                    {
                        "type": "text",
                        "args": text_args,
                        "text": f"Step: {self.agent.l_step}, action: {action_name}"
                    },
                    {
                        "type": "image_row",
                        "height": int(self.obs['rgb_dict']["pov"].shape[0]),
                        "image": [
                            ["First person view", self.obs['rgb_dict']["pov"]],
                            ["Third person view", self.obs['rgb_dict']["third_person"]],
                        ],
                    },
                ]
                self.agent.vis_frames.setdefault('obs_action', []).append(render_grid(vis_grid))
        elif self.args.backend == "tiamat":
            if (self.obs and "rgb_dict" in self.obs and action_name):
                # add a new line every 30 letters in instruction
                instruction = self.instruction
                instruction_lines = ["Instruction: "]
                for word in instruction.split():
                    if len(instruction_lines) == 0 or len(instruction_lines[-1]) + len(word) > 30:
                        instruction_lines.append(word)
                    else:
                        instruction_lines[-1] += f" {word}"
                instruction_text = [{"type": "text", "args": text_args, "text": line} for line in instruction_lines]
                print([*self.obs['rgb_dict'].keys()])
                vis_grid = [
                    {
                        "type": "text",
                        "args": text_args,
                        "text": f"Episode: {self.obs['episode_label']}"
                    },
                    *instruction_text,
                    {
                        "type": "text",
                        "args": text_args,
                        "text": f"Step: {self.agent.l_step}, action: {action_name}"
                    },
                    {
                        "type": "image_row",
                        "height": int(self.obs['rgb_dict']["frontright"].shape[0]),
                        "image": [
                            ["main view", self.obs['rgb']],
                        ],
                    },
                ]
                self.agent.vis_frames.setdefault('obs_action', []).append(render_grid(vis_grid))

        # if self.obs.get('rgb_dict', None):
        #     for cam_name, rgb in self.obs['rgb_dict'].items():
        #         if rgb is None:
        #             continue
        #         img_data = {"type": "image", "channel_name": f"rgb/{cam_name}", "image": rgb}
        #         self.foxglove_log_queue.put(img_data)
        # elif self.obs.get('rgb', None) is not None:
        #     img_data = {"type": "image", "channel_name": "rgb", "image": self.obs['rgb']}
        #     self.foxglove_log_queue.put(img_data)
        # if self.obs.get('depth_dict', None):
        #     for cam_name, depth_image in self.obs['depth_dict'].items():
        #         if depth_image is None:
        #             continue
        #         depth_max = depth_image.max() if depth_image.max() > 0 else 1.0
        #         depth_vis = (depth_image / depth_max * 255).astype(np.uint8)
        #         depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        #         depth_data = {"type": "image", "channel_name": f"depth/{cam_name}", "image": depth_vis}
        #         self.foxglove_log_queue.put(depth_data)
        # elif self.obs.get('depth', None) is not None:
        #     depth_image = self.obs['depth']
        #     depth_max = depth_image.max() if depth_image.max() > 0 else 1.0
        #     depth_vis = (depth_image / depth_max * 255).astype(np.uint8)
        #     depth_visere = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        #     depth_data = {"type": "image", "channel_name": "depth", "image": depth_vis}
        #     self.foxglove_log_queue.put(depth_data)

    def _forward_queue(self, source, target):
        while not self.evaluation_over.is_set():
            try:
                data = source.get(timeout=0.1)
                target.put(data)
            except Empty:
                continue
            except Exception as e:
                print(f"Error forwarding queue: {e}")

    def run(self):
        # Start forwarding threads
        t1 = threading.Thread(target=self._forward_queue, args=(self.step_data_queue, self.ray_step_data_queue), daemon=True)
        t1.start()
        t2 = threading.Thread(target=self._forward_queue, args=(self.foxglove_log_queue, self.ray_foxglove_log_queue), daemon=True)
        t2.start()

        args = self.args
        setup_tee(args.log_file)
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except ValueError:
            # When run inside a Ray actor thread, signal handlers can't be changed.
            pass
        episode_start_time = None
        run_error = None
        try:
            cuda_visible_devices = getattr(args, "cuda_visible_devices", "")
            if cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu_id)
            else:
                self.log("No GPU available, exiting...", level="error")
                return
            while not self.evaluation_over.is_set():
                # reset agent if needed
                if self.reset_flag.is_set():
                    if self.initialized:
                        self.save_result(info={
                            "episode_duration": time.time() - episode_start_time
                        })
                    if self.args.backend == "real_world":
                        self.ask_for_instruction()
                    else:
                        if not self.load_next_task():
                            break
                    if not self.initialized:
                        self.agent = self.load_agent()
                        self.initialized = True
                    self.log(f"Resetting agent at {time.strftime('%H:%M:%S')}")
                    episode_start_time = time.time()
                    self.agent.reset()
                    self.action_history = {}
                    self.trajectory = {}
                    self.pose_history = []
                    self.modern_video_samples = []
                    self.parquet_rows = []
                    self.waiting_for_stop_termination = False
                    # Clear stale action-wait state from the previous episode so the
                    # new episode does not deadlock comparing the fresh spawn pose
                    # against the prior target waypoint until wall_time timeout.
                    self.wait_for_action = False
                    self.wait_for_action_start_time = 0
                    self.wait_for_action_start_sim_time = None
                    self.wait_for_action_mode = None
                    self.timed_action_duration_s = None
                    self.target_waypoint = None
                    self.last_action_status = None
                    self.forward_heading_yaw = None
                    self.server_error_streak_message = None
                    self.server_error_streak_count = 0
                    self.reset_flag.clear()

                # get obs
                obs = self.get_obs()
                if obs is None:
                    self._periodic_status("no_obs")
                    self.log("No obs received, skipping", level="debug")
                    continue
                self.obs = obs
                if self.waiting_for_stop_termination:
                    self._periodic_status("waiting_for_stop_termination")
                    continue

                # update agent and get action
                action = self.agent.act(obs)
                self.update_vis(action)
                self.record_parquet_step(action)
                self.report_agent_progress(action=action, reason="agent step")
                if action is not None:
                    # check if episode is over
                    if self._is_stop_action(action.get("action", None)):
                        if self.args.continuous:
                            self.log("Agent is in continuous mode, continuing after stop action")
                        else:
                            self.log("Agent sent a stop action, waiting for final simulator metrics")
                            self.waiting_for_stop_termination = True
                    # send action
                    self.action = action
                    self.send_action(**action, verbose=True)
                    self.last_action_send_time = time.time()
                else:
                    self._periodic_status("no_action")
        except Exception as e:
            run_error = e
            self.log(f"Error: {e}")
            import traceback
            traceback.print_exc()
        self.evaluation_over.set()
        if self.agent:
            self.agent.shutdown()
        self.log(f"Evaluation worker {self.args.process_label} finished.")
        if run_error is not None:
            raise run_error
    
    def ask_for_instruction(self):
        print("Please type in the target object:")
        instruction = self.command_queue.get()
        self.log(f"Instruction set to {instruction}")
        self.instruction = instruction

    def _request_payload(self, request_type):
        args = self.args
        try:
            return request_data(args.host, args.port, request_type=request_type, verbose=False)
        except socket.timeout:
            self.log(f"Socket timeout during connection to {args.host}:{args.port}", level="debug")
        except socket.error as e:
            self.log(f"Socket error: {e}", level="debug")
        except Exception as e:
            self.log(f"Error: {e}", level="error")
        return None

    def _pose_from_payload(self, payload):
        pose = payload.get("pose")
        if pose is None:
            return None, None
        if isinstance(pose, dict) and "pose" in pose:
            pose = pose["pose"]
            position = [pose["position"][_] for _ in ["x", "y", "z"]]
            rotation = [pose["orientation"][_] for _ in ["x", "y", "z", "w"]]
        else:
            position = pose.get("position")
            rotation = pose.get("orientation")
        if position is None or rotation is None:
            return None, None
        return position, rotation

    def _update_episode_from_payload(self, payload):
        scene_id = payload.get("scene_id", None)
        episode_id = payload.get("episode_id", None)
        if self.args.backend == "real_world":
            episode_id = f"{self.instruction}_{time.strftime('%H%M%S')}"
        else:
            self.instruction = payload.get("instruction", self.instruction or "None")
            self.instruction_type = payload.get("instruction_type")
            self.target_type = payload.get("target_type")
        episode_label = f"{scene_id}_{episode_id}"
        if not self.is_episode_correct(episode_label):
            self.log(f"Was waiting for episode {self.expected_episode}, but got {episode_label}", level="warning")
            return None, None, None
        self.episode_label = episode_label
        self.scene_id = scene_id
        self.episode_id = episode_id
        return scene_id, episode_id, episode_label

    def _set_episode_from_label(self, episode_label):
        if not episode_label:
            return
        self.episode_label = episode_label
        self.scene_id = "_".join(episode_label.split("_")[:-1])
        self.episode_id = episode_label.split("_")[-1]
        self.instruction = self.instruction or "None"

    def _poll_action_state(self, payload):
        if payload is None:
            self._periodic_status("no_state_payload_from_server")
            return True
        if not payload.get("success", False):
            if payload.get("sim_state") == "terminated":
                return False
            self._periodic_status(f"state_err: {payload.get('message', 'Unknown error')}")
            return True

        if self._update_episode_from_payload(payload)[2] is None:
            return True
        position, rotation = self._pose_from_payload(payload)
        if position is None or rotation is None:
            self._periodic_status("state_missing_pose")
            return True

        self.obs.update({
            "position": position,
            "rotation": rotation,
            "timestamp": payload.get("timestamp_server_ns", time.time_ns()) / 1000.0,
            "sim_time": payload.get("sim_time"),
            "sim_step": payload.get("sim_step"),
            "step_dt": payload.get("step_dt"),
            "metrics": payload.get("metrics", {}),
            "timed_trajectory_status": payload.get("timed_trajectory_status"),
        })
        self.camera = Camera(self.args, no_init_pose=True)
        self.body_pose = self.camera.get_pose_agent(position=position, quat_xyzw=rotation)
        self.wait_for_action = not self.check_action_status(self.body_pose)
        self._record_pose_sample("sim_state")
        if self.wait_for_action:
            self._periodic_status("waiting_action_state")
            return True
        return False

    def get_obs(self):
        args = self.args
        if self.wait_for_action:
            state_payload = self._request_payload("GET_SIM_STATE")
            if self._poll_action_state(state_payload):
                return None

        # Request Data
        payload = self._request_payload("GET_SENSOR_DATA")
        
        # Process Data
        position = np.zeros(3)
        obs = None
        if payload is not None and payload.get("success", False):
            self.server_error_streak_message = None
            self.server_error_streak_count = 0
            self.payload = payload

            # get timestamp
            server_timestamp_ns = payload.get("timestamp_server_ns")
            if server_timestamp_ns == self.last_timestamp:
                self.log("Timestamp is the same as the last timestamp, skipping", level="info")
                return None
            self.last_timestamp = server_timestamp_ns
            sim_time = payload.get("sim_time")
            sim_step = payload.get("sim_step")
            step_dt = payload.get("step_dt")
            
            # get info from payload
            scene_id = payload.get("scene_id", None)
            episode_id = payload.get("episode_id", None)
            robot_height = payload.get("robot_height", None)
            if robot_height is not None:
                args.ROBOT_HEIGHT = float(robot_height)
                robot_height = args.ROBOT_HEIGHT
            else:
                robot_height = float(args.ROBOT_HEIGHT)
            scene_id, episode_id, episode_label = self._update_episode_from_payload(payload)
            if episode_label is None:
                return None

            # get rgb and depth images
            rgb_images = payload.get("rgb_images")
            depth_images = payload.get("depth_images")
            if rgb_images is None and payload.get("rgb_image") is not None:
                rgb_images = {"default": payload.get("rgb_image")}
            if depth_images is None and payload.get("depth_image") is not None:
                depth_images = {"default": payload.get("depth_image")}
            if rgb_images is None or depth_images is None:
                self.log("rgb_images or depth_images is None", level="error")
                return None
            camera_names = sorted(set(rgb_images.keys()) & set(depth_images.keys()))

            # get gt goal position if args.gt_goal_position is True
            if args.gt_goal_position:
                gt_goal_position = payload.get("gt_goal_position", None)
                if gt_goal_position is None:
                    self.log("gt_goal_position is not available in obs", level="error")
                    return None

            # get pose
            position, rotation = self._pose_from_payload(payload)
            if position is None or rotation is None:
                self.log("pose is missing in obs", level="error")
                return None

            # check action status
            self.camera = Camera(args, no_init_pose=True)
            self.body_pose = self.camera.get_pose_agent(position=position, quat_xyzw=rotation)
            self.wait_for_action = not self.check_action_status(self.body_pose)     # Check if current waypoint is acheived
            if self.wait_for_action:
                self._periodic_status("waiting_action")
                return None

            # get primary camera
            if args.primary_camera != "":
                # use specific camera
                primary_camera = args.primary_camera
                if primary_camera not in camera_names:
                    self.log(f"Primary camera {primary_camera} not available (available: {camera_names}), skipping", level="error")
                    return None
            else:
                # try front left first
                primary_camera = "frontleft"
                if primary_camera not in camera_names:
                    # front left not available, use first camera
                    if len(camera_names) > 0:
                        primary_camera = camera_names[0]
                    else:
                        self.log("No cameras available, skipping", level="error")
                        return None 

            # get camera transforms
            tf_dict = extract_tf_dict(payload)
            camera_tfs = {}
            for cam_name in camera_names:
                match_key = match_tf_key(tf_dict, cam_name)
                if match_key is not None:
                    camera_tfs[cam_name] = tf_dict.get(match_key)

            # get mapping cameras
            mapping_cameras = []
            if args.mapping_cameras:
                mapping_cameras = [c.strip() for c in args.mapping_cameras.split(",") if c.strip() in camera_names]
            if len(mapping_cameras) == 0:
                mapping_cameras = camera_names

            # get hfov
            hfov_deg = payload.get("hfov_deg", None)
            hfov_dict = {}
            if isinstance(hfov_deg, dict):
                hfov_dict = hfov_deg
            else:
                if hfov_deg is not None:
                    hfov_dict = {cam_name: hfov_deg for cam_name in camera_names}
            if not hfov_dict:
                hfov_dict = {cam_name: float(args.hfov) for cam_name in camera_names}
            hfov_scalar = hfov_dict.get(primary_camera)
            if hfov_scalar is None and hfov_dict:
                hfov_scalar = next(iter(hfov_dict.values()))

            # get obs from payload
            rgb_dict = {}
            depth_dict = {}
            for cam_name in camera_names:
                rgb = rgb_images.get(cam_name)
                depth = depth_images.get(cam_name)
                if depth is not None:
                    if depth.dtype == np.uint16:
                        depth = depth.astype(float) / 1000.0
                    else:
                        depth = depth.astype(float)
                rgb_dict[cam_name] = rgb
                depth_dict[cam_name] = depth

            rgb_image = rgb_dict.get(primary_camera) if primary_camera else None
            depth_image = depth_dict.get(primary_camera) if primary_camera else None

            obs = {
                "rgb": rgb_image,
                "depth": depth_image,
                "rgb_dict": rgb_dict,
                "depth_dict": depth_dict,
                "hfov_dict": hfov_dict,
                "camera_tfs": camera_tfs,
                "primary_camera": primary_camera,
                "mapping_cameras": mapping_cameras,
                "objectgoal": self.instruction,
                "instruction_type": self.instruction_type,
                "target_type": self.target_type,
                "position": position,
                "rotation": rotation,
                "hfov": hfov_scalar,
                "episode_label": episode_label,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "timestamp": server_timestamp_ns/1000.0,
                "robot_height": robot_height,
                "sim_time": sim_time,
                "sim_step": sim_step,
                "step_dt": step_dt,
                "metrics": payload.get("metrics", {}),
                "last_action_status": self.last_action_status,
            }
            if args.gt_goal_position:
                obs["gt_goal_position"] = gt_goal_position
        elif payload is not None and "terminations" in payload:
            payload_episode_label = self._payload_episode_label(payload)
            if (
                self.expected_episode is not None
                and payload_episode_label is not None
                and payload_episode_label != self.expected_episode
            ):
                self.log(
                    f"Ignoring stale termination for {payload_episode_label} while waiting for {self.expected_episode}",
                    level="warning",
                )
                return None
            self.log(f"Received terminations from server: {payload.get('terminations')}")
            self._set_episode_from_label(payload_episode_label or self.expected_episode)
            self.instruction = payload.get("instruction", self.instruction or "None")
            self.instruction_type = payload.get("instruction_type", self.instruction_type)
            self.target_type = payload.get("target_type", self.target_type)
            if not hasattr(self, "payload"):
                self.payload = {}
            self.payload.update(payload)
            self.waiting_for_stop_termination = False
            self.reset_flag.set()
            return None
        elif payload is not None:
            self._handle_server_error_payload(payload)
            # self.log(f"Server responded with an issue: {payload.get('message', 'Unknown error')}", level="info")
        else:
            self._periodic_status("no_payload_from_server")
            # self.log("Received empty or invalid payload from server.", level="info")
        return obs

    def _handle_server_error_payload(self, payload):
        message = str(payload.get("message") or "Unknown error")
        if message == self.server_error_streak_message:
            self.server_error_streak_count += 1
        else:
            self.server_error_streak_message = message
            self.server_error_streak_count = 1

        if message == ON_DEMAND_RENDER_TIMEOUT_MESSAGE:
            max_repeats = int(getattr(self.args, "fatal_server_error_repeats", 6) or 0)
            if max_repeats > 0 and self.server_error_streak_count >= max_repeats:
                raise RuntimeError(
                    f"{message} repeated {self.server_error_streak_count} times for "
                    f"{self.expected_episode or self.episode_label or '<unknown episode>'}"
                )

        self._periodic_status(f"server_err: {message}")

    @staticmethod
    def _scene_uses_long_forward_step(scene_id):
        scene_id = str(scene_id or "").lower()
        return scene_id.startswith(("vc_", "vcp_", "innout_"))

    def _forward_distance_m(self):
        override = getattr(self.args, "discrete_action_forward_distance_m", None)
        if override is not None:
            return float(override)
        scene_id = self.scene_id
        if scene_id is None and self.expected_episode:
            scene_id = "_".join(str(self.expected_episode).split("_")[:-1])
        if self._scene_uses_long_forward_step(scene_id):
            return 1.0
        return 0.25
    
    def _convert_action_to_waypoint(self, action):
        forward_distance, lateral_distance, turn_angle = self._forward_distance_m(), 0.25, np.deg2rad(30)
        discrete_action_list = {
            'FORWARD': [forward_distance, 0.0, 0.0],
            'LEFT': [0.0, lateral_distance, 0.0],
            'RIGHT': [0.0, -lateral_distance, 0.0],
            'TURN_RIGHT': [0.0, 0.0, -turn_angle],
            'TURN_LEFT': [0.0, 0.0, turn_angle],
            'TURN_RIGHT_SMALL': [0.0, 0.0, -turn_angle/2.],
            'TURN_LEFT_SMALL': [0.0, 0.0, turn_angle/2.]
        }
        if not isinstance(action, list):
            action = [action]
        next_waypoints = []
        for a in action:
            if a not in discrete_action_list:
                self.log(f"Unknown action: {a}", level="warning")
                continue
            action_waypoint = discrete_action_list[a]
            current_waypoint = self.camera.convert_agent_pose_to_waypoint(self.body_pose)
            if a == "FORWARD":
                if self.forward_heading_yaw is None:
                    self.forward_heading_yaw = current_waypoint[2]
                heading = self.forward_heading_yaw
                next_waypoint = [
                    current_waypoint[0] + action_waypoint[0] * math.cos(heading),
                    current_waypoint[1] + action_waypoint[0] * math.sin(heading),
                    heading,
                ]
            elif action_waypoint[0] == 0.0 and action_waypoint[1] == 0.0 and action_waypoint[2] != 0.0:
                self.forward_heading_yaw = None
                next_waypoint = [
                    current_waypoint[0],
                    current_waypoint[1],
                    warp_to_pi(current_waypoint[2] + action_waypoint[2]),
                ]
            else:
                self.forward_heading_yaw = None
                T_rel = self.camera.parse_pose(position=[action_waypoint[0], action_waypoint[1], 0], yaw_angle=action_waypoint[2])
                next_pose = self.body_pose @ T_rel
                next_waypoint = self.camera.convert_agent_pose_to_waypoint(next_pose)
            next_waypoints.append(next_waypoint)
            print(f"current_waypoint:{current_waypoint}")
            print(f"next_waypoint: {next_waypoint}")
        return next_waypoints
    
    def _is_stop_action(self, action):
        if isinstance(action, list):
            return "STOP" in action
        elif isinstance(action, str):
            return action == "STOP"
        else:
            return False
    
    def get_action_name(self, vel=None, waypoint=None, timed_trajectory=None, transform=None, action=None, **kwargs):
        if vel is not None:
            return f"vx: {vel[0]}, vy: {vel[1]}, vw: {vel[2]}"
        elif waypoint is not None:
            return f"{len(waypoint)} waypoints, next waypoint: {waypoint[0]}"
        elif timed_trajectory is not None:
            points = timed_trajectory.get("waypoints", [])
            dt = timed_trajectory.get("dt")
            return f"{len(points)} timed SE(2) setpoints at dt={dt}s"
        elif transform is not None:
            if isinstance(transform, dict):
                position = transform.get("position", transform.get("translation", transform))
                x = position.get("x", None)
                y = position.get("y", None)
                z = position.get("z", None)
                if x is not None and y is not None:
                    return f"transform: [{x:.2f}, {y:.2f}, {z if z is not None else 0.0:.2f}]"
            return "transform"
        elif action is not None:
            return action
        else:
            return "None"
    
    def send_action(self, vel=None, waypoint=None, timed_trajectory=None, transform=None, action=None, action_args=None, verbose=False, **kwargs):
        args = self.args
        # Handle stop action
        if timed_trajectory is not None:
            points = np.asarray(timed_trajectory.get("waypoints"), dtype=float)
            dt = float(timed_trajectory.get("dt", 0.0))
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                raise ValueError(f"timed_trajectory waypoints must be non-empty (N, 3), got {points.shape}")
            if not np.isfinite(points).all() or not np.isfinite(dt) or dt <= 0.0:
                raise ValueError("timed_trajectory requires finite waypoints and a positive dt")
            self.set_wait_for_timed_trajectory(len(points) * dt)
            msg_type = "TIMED_TRAJECTORY"
            msg = {"waypoints": points.tolist(), "dt": dt}
        elif waypoint is not None:
            if self.args.agent == "GPT" and len(waypoint) == 1:
                self.set_wait_for_action(waypoint[0])
            msg_type = "WAYPOINT"
            msg = {"waypoint": waypoint}
        elif transform is not None:
            msg_type = "TRANSFORM"
            msg = {"transform": transform}
        elif action is not None:
            if self._is_stop_action(action):
                self.forward_heading_yaw = None
                msg_type = "STOP"
                msg = {}
                self.send_action(vel=(0.0, 0.0, 0.0), verbose=verbose) 
            elif action == "RESET":
                self.forward_heading_yaw = None
                msg_type = "RESET"
                msg = {}
            else:
                waypoints = self._convert_action_to_waypoint(action)
                if len(waypoints) > 0:
                    self.set_wait_for_action(waypoints[-1])
                    msg_type = "WAYPOINT"
                    msg = {"waypoint": waypoints}
                else:
                    self.log("invalid action.", level="warning")
                    return
        elif vel is not None:
            msg_type = "VEL"
            msg = {
                "vx": vel[0],
                "vy": vel[1],
                "vw": vel[2]
            }
        else:
            self.log("invalid action.", level="warning")
            return

        if action_args is not None and isinstance(action_args, dict):
            msg.update(action_args)

        if not args.disable_action:
            retry_count = 0
            max_retries = 100
            while retry_count < max_retries:
                try:
                    send_message(msg_type, msg, args.host, args.port)
                    break
                except Exception as e:
                    retry_count += 1
                    self.log(f"Error sending action (retry {retry_count}/{max_retries}): {e}", level="warning")
                    time.sleep(0.01)
            if retry_count >= max_retries:
                self.log(f"Failed to send action after {max_retries} retries", level="error")
                return
            if verbose:
                self.log(
                    f"Action sent: {msg_type} "
                    f"{self.get_action_name(vel=vel, waypoint=waypoint, timed_trajectory=timed_trajectory, transform=transform, action=action)}",
                    level="info",
                )
    
    def set_wait_for_action(self, target_waypoint):
        self.wait_for_action_start_time = time.time()
        self.wait_for_action_start_sim_time = None
        self.wait_for_action_mode = "waypoint"
        self.timed_action_duration_s = None
        self.target_waypoint = target_waypoint
        start_waypoint = self.camera.convert_agent_pose_to_waypoint(self.body_pose)
        target_diff = [self.target_waypoint[i] - start_waypoint[i] for i in range(3)]
        target_diff[2] = warp_to_pi(target_diff[2])
        self.wait_for_action_ignores_position = (
            self.args.agent != "GPT"
            and np.linalg.norm(target_diff[:2]) <= 0.02
            and abs(target_diff[2]) > np.deg2rad(1.0)
        )
        self.wait_for_action = True
        self.last_action_status = {
            "status": "pending",
            "target_waypoint": target_waypoint,
            "ignore_position": self.wait_for_action_ignores_position,
            "started_wall_time": self.wait_for_action_start_time,
            "started_sim_time": self.wait_for_action_start_sim_time,
        }

    def set_wait_for_timed_trajectory(self, duration_s):
        self.wait_for_action_start_time = time.time()
        self.wait_for_action_start_sim_time = None
        self.wait_for_action_mode = "timed_trajectory"
        self.timed_action_duration_s = float(duration_s)
        self.target_waypoint = None
        self.wait_for_action = True
        self.last_action_status = {
            "status": "pending",
            "mode": "timed_trajectory",
            "duration_s": self.timed_action_duration_s,
            "started_wall_time": self.wait_for_action_start_time,
            "started_sim_time": None,
        }

    def _check_timed_trajectory_status(self):
        wall_elapsed = time.time() - self.wait_for_action_start_time
        status = self.obs.get("timed_trajectory_status") or {}
        start_sim_time = status.get("start_sim_time")
        duration_s = float(status.get("duration_s", self.timed_action_duration_s or 0.0))
        current_sim_time = self.obs.get("sim_time")
        sim_elapsed = None
        if start_sim_time is not None and current_sim_time is not None:
            self.wait_for_action_start_sim_time = float(start_sim_time)
            sim_elapsed = max(0.0, float(current_sim_time) - float(start_sim_time))
        complete = status.get("reason") == "completed" or (
            sim_elapsed is not None and duration_s > 0.0 and sim_elapsed >= duration_s
        )
        if complete:
            self.last_action_status = {
                "status": "complete",
                "mode": "timed_trajectory",
                "timed_out": False,
                "wall_elapsed": wall_elapsed,
                "sim_elapsed": sim_elapsed,
                "duration_s": duration_s,
                "simulator_status": status,
            }
            self.wait_for_action_mode = None
            return True
        wall_timeout = float(
            getattr(self.args, "action_wait_timeout_s", None) or DEFAULT_EPISODE_TIMEOUT_S
        )
        if wall_elapsed > wall_timeout:
            self.last_action_status = {
                "status": "timeout",
                "mode": "timed_trajectory",
                "timed_out": True,
                "timeout_source": "wall_time",
                "wall_elapsed": wall_elapsed,
                "sim_elapsed": sim_elapsed,
                "duration_s": duration_s,
                "simulator_status": status,
            }
            self.wait_for_action_mode = None
            return True
        self.last_action_status = {
            "status": "pending",
            "mode": "timed_trajectory",
            "timed_out": False,
            "wall_elapsed": wall_elapsed,
            "sim_elapsed": sim_elapsed,
            "duration_s": duration_s,
            "simulator_status": status,
        }
        return False

    def check_action_status(self, agent_pose):
        """return True if the action is complete or timed out, False otherwise"""
        if not self.wait_for_action:
            return True
        if self.wait_for_action_mode == "timed_trajectory":
            return self._check_timed_trajectory_status()
        wall_timeout = float(getattr(self.args, "action_wait_timeout_s", None) or DEFAULT_EPISODE_TIMEOUT_S)
        sim_timeout = getattr(self.args, "action_wait_timeout_sim_s", None)
        sim_timeout = None if sim_timeout is None else float(sim_timeout)
        pos_threshold = float(getattr(self.args, "action_wait_pos_threshold_m", None) or 0.05)
        angle_threshold = np.deg2rad(float(getattr(self.args, "action_wait_yaw_threshold_deg", None) or 5.0))
        if self.args.backend=="tiamat" and self.args.agent=="LongNav":
            wall_timeout = float(getattr(self.args, "action_wait_timeout_s", None) or DEFAULT_EPISODE_TIMEOUT_S)
            pos_threshold = float(getattr(self.args, "action_wait_pos_threshold_m", None) or 0.15)
            angle_threshold = np.deg2rad(float(getattr(self.args, "action_wait_yaw_threshold_deg", None) or 10.0))
        curr_waypoint = self.camera.convert_agent_pose_to_waypoint(agent_pose)
        diff = [self.target_waypoint[i]-curr_waypoint[i] for i in range(3)]
        diff[2] = warp_to_pi(diff[2])
        backend_pose = self.camera.get_pose_backend(position=agent_pose[:3, 3], rotation=agent_pose[:3, :3])
        # print("="*20)
        # print(f"target_waypoint: {self.target_waypoint[:2]}, {np.rad2deg(self.target_waypoint[2])}")
        # print(f"curr_waypoint: {curr_waypoint[:2]}, {np.rad2deg(curr_waypoint[2])}")
        # print(f"diff: {diff[:2]}, {np.rad2deg(diff[2])}")
        self._safe_queue_put(self.foxglove_log_queue, {
            "type": "json",
            "channel_name": "action_status",
            "json": {
                "target_waypoint": self.target_waypoint[:2],
                "target_yaw": np.rad2deg(self.target_waypoint[2]),
                "curr_waypoint": curr_waypoint[:2],
                "curr_yaw": np.rad2deg(curr_waypoint[2]),
                "diff": diff[:2],
                "diff_yaw": np.rad2deg(diff[2]),
                "obs_position": self.obs["position"],
                "backend_pos": backend_pose[:3, 3],
            }
        }, "foxglove_log")
        action_complete = True
        wall_elapsed = time.time() - self.wait_for_action_start_time
        current_sim_time = self.obs.get("sim_time")
        sim_elapsed = None
        if self.wait_for_action_start_sim_time is None and current_sim_time is not None:
            self.wait_for_action_start_sim_time = float(current_sim_time)
            self.last_action_status["started_sim_time"] = self.wait_for_action_start_sim_time
        if self.wait_for_action_start_sim_time is not None and current_sim_time is not None:
            sim_elapsed = max(0.0, float(current_sim_time) - float(self.wait_for_action_start_sim_time))
        ignore_position = bool(getattr(self, "wait_for_action_ignores_position", False))
        if (not ignore_position) and np.linalg.norm(diff[:2]) > pos_threshold:
            # self.log(f"Waiting for action to complete... pos_diff={np.linalg.norm(diff[:2]):.2f} ({pos_threshold:.2f}), ang_diff={np.rad2deg(diff[2]):.2f} ({np.rad2deg(angle_threshold):.2f})", level="info")
            action_complete = False
        if np.abs(diff[2]) > angle_threshold:
            # self.log(f"Waiting for action to complete... pos_diff={np.linalg.norm(diff[:2]):.2f} ({pos_threshold:.2f}), ang_diff={np.rad2deg(diff[2]):.2f} ({np.rad2deg(angle_threshold):.2f})", level="info")
            action_complete = False
        # print("================================================")
        # print(f"target_waypoint: {self.target_waypoint[:2]}, {np.rad2deg(self.target_waypoint[2])}")
        # print(f"curr_waypoint: {curr_waypoint[:2]}, {np.rad2deg(curr_waypoint[2])}")
        # print(f"diff: {diff[:2]}, {np.rad2deg(diff[2])}")
        if action_complete:
            self.last_action_status = {
                "status": "complete",
                "timed_out": False,
                "wall_elapsed": wall_elapsed,
                "sim_elapsed": sim_elapsed,
                "target_waypoint": [float(x) for x in self.target_waypoint],
                "curr_waypoint": [float(x) for x in curr_waypoint],
                "ignore_position": ignore_position,
                "pos_error": float(np.linalg.norm(diff[:2])),
                "yaw_error_deg": float(np.rad2deg(diff[2])),
            }
            print(f"Action complete after {wall_elapsed} seconds")
            return True
        else:
            timed_out_by_sim = sim_timeout is not None and sim_elapsed is not None and sim_elapsed > sim_timeout
            timed_out_by_wall = wall_elapsed > wall_timeout
            if timed_out_by_sim or timed_out_by_wall:
                timeout_source = "sim_time" if timed_out_by_sim else "wall_time"
                self.last_action_status = {
                    "status": "timeout",
                    "timed_out": True,
                    "timeout_source": timeout_source,
                    "wall_elapsed": wall_elapsed,
                    "sim_elapsed": sim_elapsed,
                    "target_waypoint": [float(x) for x in self.target_waypoint],
                    "curr_waypoint": [float(x) for x in curr_waypoint],
                    "ignore_position": ignore_position,
                    "pos_error": float(np.linalg.norm(diff[:2])),
                    "yaw_error_deg": float(np.rad2deg(diff[2])),
                }
                self.log(
                    f"Action timeout by {timeout_source}. "
                    f"wall_elapsed={wall_elapsed:.2f}s, sim_elapsed={sim_elapsed}, "
                    f"pos_diff={np.linalg.norm(diff[:2]):.2f}, ang_diff={np.rad2deg(diff[2]):.2f}",
                    level="info",
                )
                return True
            else:
                self.last_action_status = {
                    "status": "pending",
                    "timed_out": False,
                    "wall_elapsed": time.time() - self.wait_for_action_start_time,
                    "sim_elapsed": sim_elapsed,
                    "target_waypoint": [float(x) for x in self.target_waypoint],
                    "curr_waypoint": [float(x) for x in curr_waypoint],
                    "ignore_position": ignore_position,
                    "pos_error": float(np.linalg.norm(diff[:2])),
                    "yaw_error_deg": float(np.rad2deg(diff[2])),
                }
                return False

    def load_next_task(self, episode_label=None):
        args = self.args
        if episode_label is None:
            retry_count = 0
            max_retries = 10
            while retry_count < max_retries:
                try:
                    task = self.task_queue.get_task(args.process_label, self.scene_id)
                    if task is not None:
                        break
                    if self.task_queue.empty():
                        break
                except Exception as e:
                    self.log(f"{e}", level="error")
                    self.evaluation_over.set()
                    return False
                time.sleep(0.1)
                retry_count += 1
            if task is None:
                self.log("No task found, exiting...")
                self.evaluation_over.set()
                return False
            
        else:
            task = {
                "episode_label": episode_label,
                "scene_id": "_".join(episode_label.split("_")[:-1]),
                "episode_id": episode_label.split("_")[-1],
            }
        if self.episode_controller is not None:
            ray.get(self.episode_controller.ensure_worker_episode.remote(args.process_label, task))
        else:
            control_port = task.get("control_port", args.port)
            episode_info = {
                "episode_label": task['episode_label'],
                "worker_id": args.process_label,
                "scene_id": task.get("scene_id"),
            }
            send_message("EPISODE", episode_info, args.host, control_port)
        self.expected_episode = task['episode_label']
        self.wrong_episode_count = 0
        self.last_timestamp = None
        self.log(f"Loading task: {task['episode_label']}@{args.port}")
        self.report_agent_progress(reason=f"loading {task['episode_label']}", force=True)
        return True

    def is_episode_correct(self, episode_label):
        # only isaac backend has episode switching functionality right now
        if self.args.backend not in ["isaac", "tiamat"]:
            return True
        if self.expected_episode is not None:
            if episode_label != self.expected_episode:
                self.wrong_episode_count += 1
                if self.wrong_episode_count >= 5:
                    mismatch_count = self.wrong_episode_count
                    self.log(
                        f"Wrong episode count {mismatch_count} reached for {self.expected_episode}; "
                        "requesting worker restart",
                        level="warning",
                    )
                    self.load_next_task(episode_label=self.expected_episode)
                    if mismatch_count >= int(getattr(self.args, "fatal_wrong_episode_repeats", 20) or 20):
                        raise RuntimeError(
                            f"Simulator kept serving {episode_label} while waiting for {self.expected_episode}"
                        )
                    self.wrong_episode_count = mismatch_count
                return False
            else:
                self.expected_episode = None
        self.wrong_episode_count = 0
        return True

    def _make_early_termination_frame(self):
        terminations = getattr(self, "payload", {}).get("terminations", {})
        reason = terminations.get("termination_reason", "unknown")
        lines = [
            f"Episode: {self.episode_label or self.expected_episode or 'unknown'}",
            "Termination before first usable observation",
            f"Reason: {reason}",
            f"Step: {getattr(self.agent, 'l_step', 0)}",
        ]
        return render_grid([
            {"type": "text", "args": text_args, "text": line}
            for line in lines
        ])

    @staticmethod
    def _sorted_numeric_keys(mapping):
        if not isinstance(mapping, dict):
            return []
        keys = []
        for key in mapping.keys():
            try:
                keys.append(int(key))
            except (TypeError, ValueError):
                continue
        return sorted(keys)

    def _obs_action_frame_steps(self, frame_count):
        steps = self._sorted_numeric_keys(self.action_history)
        if not steps:
            steps = self._sorted_numeric_keys(self.trajectory)
        if not steps:
            return [None] * frame_count
        if len(steps) >= frame_count:
            return steps[:frame_count]
        return steps + [steps[-1]] * (frame_count - len(steps))

    def _render_obs_action_bev_sequence(self, frame_count, output_size=None, force=False):
        if not force and not getattr(self.args, "obs_action_bev", True):
            return None
        if not self.scene_id or self.episode_id is None:
            return None
        if self.bev_renderer is None:
            self.bev_renderer = BEVVideoRenderer(
                episode_folder=getattr(
                    self.args, "episode_folder", "navverse_data/episodes"
                ),
            )
        if output_size is not None:
            self.bev_renderer.output_size = output_size
        frame_steps = self._obs_action_frame_steps(frame_count)
        return self.bev_renderer.render_sequence(
            self.scene_id,
            self.episode_id,
            self.trajectory,
            frame_steps,
        )

    def _augment_obs_action_frames_with_bev(self, frame_list, bev_images=None):
        if not frame_list or not getattr(self.args, "obs_action_bev", True):
            return frame_list
        start_time = time.perf_counter()
        panel_width = 760
        panel_margin = 8
        frame_height = int(frame_list[0].shape[0])
        bev_output_size = (panel_width - 2 * panel_margin, frame_height - 2 * panel_margin)
        if bev_images is None:
            try:
                bev_images = self._render_obs_action_bev_sequence(
                    len(frame_list), output_size=bev_output_size
                )
            except Exception as exc:
                self.log(f"Failed to render BEV panel for obs_action video: {exc}", level="warning")
                return frame_list
        if not bev_images:
            return frame_list
        augmented = append_bev_panels_to_frames(frame_list, bev_images, panel_width=panel_width, margin=panel_margin)
        elapsed_s = time.perf_counter() - start_time
        self.log(
            f"Added stepwise BEV panel to obs_action video in {elapsed_s:.2f}s "
            f"for {len(frame_list)} frames"
        )
        return augmented

    def save_result(self, info=None):
        args = self.args
        agent = self.agent
        has_termination = "terminations" in getattr(self, "payload", {})
        if agent.l_step == 0 and not has_termination:
            return
        if self.episode_label is None:
            self._set_episode_from_label(self.expected_episode)
        # get result
        result = {
            "step": agent.l_step,
            "episode_label": self.episode_label,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "instruction_type": self.instruction_type,
            "target_type": self.target_type,
        }
        if "metrics" in self.payload:
            result.update(self.payload.get("metrics", {}))
        if "terminations" in self.payload:
            try:
                result.update(self.payload.get("terminations"))
            except Exception:
                print(self.payload.get("terminations"))
        result["final_pose_debug"] = {
            "payload_episode_label": self._payload_episode_label(getattr(self, "payload", None)),
            "payload_sim_state": getattr(self, "payload", {}).get("sim_state"),
            "payload_sim_time": getattr(self, "payload", {}).get("sim_time"),
            "payload_sim_step": getattr(self, "payload", {}).get("sim_step"),
            "payload_step_dt": getattr(self, "payload", {}).get("step_dt"),
            "payload_pose": self._jsonable(getattr(self, "payload", {}).get("pose")),
            "payload_terminations": self._jsonable(getattr(self, "payload", {}).get("terminations")),
            "obs_position": self._jsonable(getattr(self, "obs", {}).get("position")),
            "obs_rotation": self._jsonable(getattr(self, "obs", {}).get("rotation")),
            "body_pose": self._jsonable(getattr(self, "body_pose", None)),
            "last_action_status": self._jsonable(self.last_action_status),
            "wait_for_action": bool(getattr(self, "wait_for_action", False)),
            "target_waypoint": self._jsonable(getattr(self, "target_waypoint", None)),
        }
        if info is not None:
            result.update(info)
        if hasattr(agent, "get_result_info") and callable(agent.get_result_info):
            result.update(agent.get_result_info())
        # save action history and trajectory
        data_dir = f'{args.dump_location}/{args.exp_name}/data'
        os.makedirs(data_dir, exist_ok=True)
        episode_data_path = f'{data_dir}/{self.episode_label}.json'
        with open(episode_data_path, 'w') as f:
            json.dump({
                "action_history": self.action_history,
                "trajectory": self.trajectory,
                "action_test_records": getattr(agent, "action_test_records", None),
            }, f)
        # save video
        if args.save_video:
            frames = getattr(agent, "vis_frames", {})
            if has_termination and not any(len(frame_list) > 0 for frame_list in frames.values()):
                frames.setdefault("obs_action", []).append(self._make_early_termination_frame())
            bev_images = None
            bev_frame_count = len(frames.get("obs_action") or self.modern_video_samples)
            if bev_frame_count:
                try:
                    bev_images = self._render_obs_action_bev_sequence(
                        bev_frame_count,
                        output_size=(744, 1064),
                        force=bool(self.modern_video_samples),
                    )
                except Exception as exc:
                    self.log(f"Failed to render BEV video sequence: {exc}", level="warning")
                    bev_images = []
            if frames.get("obs_action"):
                frames["obs_action"] = self._augment_obs_action_frames_with_bev(
                    frames["obs_action"], bev_images=bev_images
                )
            video_save_paths = []
            modern_executor = None
            modern_future = None
            modern_video_path = None
            if getattr(args, "modern_obs_video", True) and self.modern_video_samples:
                frame_count = len(self.modern_video_samples)
                modern_fps = max(2, math.ceil(frame_count / 60))
                vid_dir = f'{args.dump_location}/{args.exp_name}/vid'
                os.makedirs(vid_dir, exist_ok=True)
                modern_video_path = f'{vid_dir}/{self.episode_label}_modern_{args.agent}.mp4'
                self.log("Saving modern fixed-layout video in parallel...")
                modern_executor = ThreadPoolExecutor(max_workers=1)
                modern_future = modern_executor.submit(
                    save_modern_video,
                    modern_video_path,
                    self.modern_video_samples,
                    bev_images,
                    self.instruction,
                    self.episode_label,
                    args.agent,
                    modern_fps,
                )
            for frame_type, frame_list in frames.items():
                if frame_type == 'rgb':
                    continue
                if len(frame_list) > 0:
                    self.log(f"Saving {frame_type} video...")
                    vid_dir = f'{args.dump_location}/{args.exp_name}/vid'
                    os.makedirs(vid_dir, exist_ok=True)
                    video_save_path = f'{vid_dir}/{self.episode_label}_{frame_type}_{args.agent}.mp4'
                    frame_list_bgr = [frame for frame in frame_list]
                    same_size_frames = pad_frames_to_same_size(frame_list_bgr)
                    if frame_type == 'mapping':
                        fps = 10
                    else:
                        fps = 2
                    frame_count = len(same_size_frames)
                    max_duration_sec = 60
                    if frame_count / fps > max_duration_sec:
                        fps = max(fps, math.ceil(frame_count / max_duration_sec))
                        self.log(
                            f"{frame_type} video has {frame_count} frames, "
                            f"increasing fps to {fps} to keep duration within {max_duration_sec}s"
                        )
                    imageio.mimsave(video_save_path, same_size_frames, fps=fps)
                    self.log(f"{frame_type} video saved to {video_save_path}")
                    video_save_paths.append(video_save_path)
            original_video_save_path = video_save_paths[-1] if video_save_paths else None
            if modern_future is not None:
                try:
                    modern_result = modern_future.result()
                except Exception as exc:
                    self.log(f"Failed to save modern video: {exc}", level="warning")
                else:
                    self.log(
                        f"Modern video saved in {modern_result['elapsed_s']:.2f}s to "
                        f"{modern_video_path}"
                    )
                    result["modern_video_save_path"] = modern_video_path
                    result["modern_video_save_duration_s"] = modern_result["elapsed_s"]
                    video_save_paths.append(modern_video_path)
                finally:
                    modern_executor.shutdown(wait=True)
            if video_save_paths:
                result["video_save_paths"] = video_save_paths
                result["video_save_path"] = original_video_save_path or video_save_paths[-1]

        parquet_save_path = self._save_episode_parquet(result)
        if parquet_save_path is not None:
            result["parquet_save_path"] = parquet_save_path

        # gather result
        self.result_queue.put(result)
        self.task_queue.complete_task(args.process_label, self.episode_label)
    def log(self, message, level="info"):
        level = level.lower()
        if level not in ["debug", "info", "warning", "error"]:
            level = "info"
        # color codes
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        WARNING = '\033[93m'
        ERROR = '\033[91m'
        ENDC = '\033[0m'
        timestamp = time.strftime("%H:%M:%S")
        # print message with color
        if level == "debug":
            if self.args.verbose:
                print(f"{HEADER}[{timestamp} agent_{self.args.process_label}] {message}{ENDC}")
        elif level == "info":
            print(f"{OKBLUE}[{timestamp} agent_{self.args.process_label}] {message}{ENDC}")
        elif level == "warning":
            print(f"{WARNING}[{timestamp} agent_{self.args.process_label}] [WARNING] {message}{ENDC}")
        elif level == "error":
            print(f"{ERROR}[{timestamp} agent_{self.args.process_label}] [ERROR] {message}{ENDC}")

    def _safe_queue_put(self, queue, item, name, timeout=0.2):
        try:
            queue.put(item, block=True, timeout=timeout)
            return True
        except Full:
            self.log(f"{name} queue full, dropping item", level="warning")
            return False
        except Exception as e:
            self.log(f"Failed to enqueue to {name}: {e}", level="warning")
            return False

    def _format_status_message(self, reason, now):
        details = []
        if reason == "no_payload_from_server":
            message = "Server did not return a payload."
            if self.last_action_send_time:
                details.append(f"Last action was sent {now - self.last_action_send_time:.2f}s ago.")
        elif reason == "waiting_action":
            message = "Still waiting for the previously sent action to finish."
            if self.wait_for_action_start_time:
                details.append(f"Waiting for {now - self.wait_for_action_start_time:.2f}s.")
        elif reason == "no_obs":
            message = "No usable observation is available yet."
        elif reason == "no_action":
            message = "The agent did not produce an action for the current observation."
        elif reason == "waiting_for_stop_termination":
            message = "Stop was sent; waiting for final simulator termination and metrics."
        elif reason.startswith("server_err:"):
            detail = reason.split(":", 1)[1].strip() or "Unknown error"
            message = f"Server reported an error: {detail}."
        else:
            message = f"Worker status update: {reason.replace('_', ' ')}."
        return " ".join([message, *details]).strip()

    def _periodic_status(self, reason, interval=5.0):
        now = time.time()
        if now - self.last_status_log_time < interval:
            return
        self.last_status_log_time = now
        self.log(
            self._format_status_message(reason, now),
            level="warning",
        )
