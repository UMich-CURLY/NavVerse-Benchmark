#!/usr/bin/env python3
import argparse
import atexit
import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


def trajectory_to_action_ids(
    trajectory,
    *,
    step_size: float = 0.25,
    turn_angle_deg: float = 15.0,
    lookahead: int = 4,
) -> list[int]:
    points = []
    for point in trajectory:
        if len(point) < 2:
            raise RuntimeError("InternVLA-N1 trajectory points require x and y")
        points.append((float(point[0]), float(point[1])))
    if not points:
        raise RuntimeError("InternVLA-N1 returned an empty trajectory")

    actions: list[int] = []
    yaw = 0.0
    pos = points[0]
    goal = points[-1]
    turn_angle = math.radians(turn_angle_deg)

    for _ in range(4096):
        if math.dist(pos, goal) <= 0.2:
            return actions or [0]
        nearest = min(range(len(points)), key=lambda index: math.dist(points[index], pos))
        target = points[min(nearest + lookahead, len(points) - 1)]
        target_dx = target[0] - pos[0]
        target_dy = target[1] - pos[1]
        if math.hypot(target_dx, target_dy) < 1e-6:
            return actions or [0]
        target_yaw = math.atan2(target_dy, target_dx)
        delta_yaw = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        turns = int(round(delta_yaw / turn_angle))
        actions.extend([2] * turns if turns > 0 else [3] * -turns)
        yaw = (yaw + turns * turn_angle + math.pi) % (2 * math.pi) - math.pi

        next_pos = (
            pos[0] + step_size * math.cos(yaw),
            pos[1] + step_size * math.sin(yaw),
        )
        if math.dist(next_pos, goal) > math.dist(pos, goal):
            return actions or [0]
        actions.append(1)
        pos = next_pos
    raise RuntimeError("InternVLA-N1 trajectory discretization exceeded its safety limit")


def discrete_action_ids(output_actions, output_trajectory) -> list[int]:
    if output_actions is not None and len(output_actions) > 0:
        actions = [int(action_id) for action_id in output_actions]
    elif output_trajectory is not None:
        actions = trajectory_to_action_ids(output_trajectory)
    else:
        raise RuntimeError("InternVLA-N1 produced no action")
    if any(action_id not in {0, 1, 2, 3} for action_id in actions):
        raise RuntimeError(f"InternVLA-N1 produced unsupported actions: {actions}")
    return actions


def _upstream_path():
    path = Path(__file__).resolve().parents[1] / "submodules" / "InternNav"
    if not (path / "internnav").is_dir():
        raise FileNotFoundError(
            f"InternNav checkout not found at {path}; run baselines/internvla_n1/setup.sh first"
        )
    sys.path.insert(0, str(path))
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="NavVerse InternVLA-N1 action server")
    parser.add_argument("--checkpoint", default="InternRobotics/InternVLA-N1-DualVLN")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    _upstream_path()

    import numpy as np
    import torch
    import uvicorn
    from fastapi import FastAPI, File, Form, UploadFile
    from PIL import Image

    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

    model_args = SimpleNamespace(
        device=args.device,
        model_path=args.checkpoint,
        resize_w=384,
        resize_h=384,
        num_history=args.num_history,
        plan_step_gap=args.plan_step_gap,
    )
    agent = InternVLAN1AsyncAgent(model_args)
    debug_dir = tempfile.mkdtemp(prefix="navverse_internvla_n1_")
    atexit.register(shutil.rmtree, debug_dir, ignore_errors=True)
    action_queue = []

    def reset_agent():
        working_dir = os.getcwd()
        try:
            os.chdir(debug_dir)
            agent.reset()
        finally:
            os.chdir(working_dir)
        agent.save_dir = debug_dir
        agent.last_s2_idx = -100
        action_queue.clear()

    reset_agent()
    lock = threading.Lock()
    app = FastAPI(title="NavVerse InternVLA-N1 server")

    @app.get("/health")
    def health():
        return {"ok": True, "checkpoint": args.checkpoint, "output": "discrete_action"}

    @app.post("/reset")
    def reset():
        with lock:
            reset_agent()
        return {"ok": True}

    @app.post("/act")
    def act(
        rgb: UploadFile = File(...),
        depth: Optional[UploadFile] = File(None),
        confidence: Optional[UploadFile] = File(None),
        instruction: str = Form(""),
        reset: bool = Form(False),
        hfov: float = Form(79.0),
        depth_scale_m: float = Form(10.0),
        observation_json: Optional[str] = Form(None),
    ):
        del confidence
        observation = json.loads(observation_json) if observation_json else {}
        rgb_array = np.asarray(Image.open(io.BytesIO(rgb.file.read())).convert("RGB"))
        height, width = rgb_array.shape[:2]
        if depth is None:
            if "nextdit" not in str(agent.model.config.system1):
                raise ValueError("This InternVLA-N1 System 1 requires a depth observation")
            depth_array = np.zeros((height, width), dtype=np.float32)
        else:
            depth_array = np.asarray(
                Image.open(io.BytesIO(depth.file.read())), dtype=np.float32
            ) * depth_scale_m
            if depth_array.shape != (height, width):
                depth_array = np.asarray(
                    Image.fromarray(depth_array, mode="F").resize(
                        (width, height),
                        Image.Resampling.NEAREST,
                    ),
                    dtype=np.float32,
                )
        intrinsic = np.asarray(
            observation.get("intrinsics", {}).get("rgb", []),
            dtype=np.float32,
        )
        if intrinsic.shape != (3, 3):
            focal = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
            intrinsic = np.array(
                [
                    [focal, 0.0, width / 2.0],
                    [0.0, focal, height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        camera_to_world = np.asarray(
            observation.get("pose", {}).get("camera_to_world", []),
            dtype=np.float32,
        )
        if camera_to_world.shape != (4, 4):
            camera_to_world = np.eye(4, dtype=np.float32)

        with lock:
            if reset:
                reset_agent()
            if action_queue:
                return {"action_id": action_queue.pop(0)}

            with torch.inference_mode():
                output = agent.step(
                    rgb_array,
                    depth_array,
                    camera_to_world,
                    instruction,
                    intrinsic,
                )
            output_actions = output.output_action
            if output_actions is not None and len(output_actions) > 0 and int(output_actions[0]) == 5:
                with torch.inference_mode():
                    output = agent.step(
                        rgb_array,
                        depth_array,
                        camera_to_world,
                        instruction,
                        intrinsic,
                        look_down=True,
                    )
                output_actions = output.output_action

            action_queue.extend(
                discrete_action_ids(output_actions, output.output_trajectory)
            )
            return {"action_id": action_queue.pop(0)}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
