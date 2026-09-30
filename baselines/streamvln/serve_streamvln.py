#!/usr/bin/env python3
import argparse
import io
import sys
import threading
from collections import deque
from pathlib import Path


def _upstream_path():
    path = Path(__file__).resolve().parents[1] / "submodules" / "StreamVLN"
    if not (path / "streamvln").is_dir():
        raise FileNotFoundError(
            f"StreamVLN checkout not found at {path}; run baselines/streamvln/setup.sh first"
        )
    sys.path.insert(0, str(path / "streamvln"))
    sys.path.insert(0, str(path))
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="NavVerse StreamVLN action server")
    parser.add_argument(
        "--checkpoint",
        default="mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn_impl",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
    )
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--model_max_length", type=int, default=4096)
    return parser.parse_args()


def _intrinsic(width, height, hfov):
    import numpy as np

    focal = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    return np.array(
        [
            [focal, 0.0, (width - 1.0) / 2.0, 0.0],
            [0.0, focal, (height - 1.0) / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def main():
    args = parse_args()
    _upstream_path()

    import numpy as np
    import torch
    import transformers
    import uvicorn
    from fastapi import FastAPI, File, Form, UploadFile
    from PIL import Image

    from streamvln.streamvln_agent import VLNEvaluator
    from model.stream_video_vln import StreamVLNForCausalLM

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    config = transformers.AutoConfig.from_pretrained(args.checkpoint)
    model = StreamVLNForCausalLM.from_pretrained(
        args.checkpoint,
        attn_implementation=args.attn_impl,
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = args.num_history
    model.reset(1)
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()

    sensor_config = {
        "rgb_height": 1.25,
        "camera_intrinsic": _intrinsic(640, 480, 79.0),
    }
    evaluator = VLNEvaluator(sensor_config, model=model, tokenizer=tokenizer, args=args)
    action_queue = deque()
    lock = threading.Lock()
    app = FastAPI(title="NavVerse StreamVLN server")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "checkpoint": args.checkpoint,
            "tokenizer": args.tokenizer,
            "attn_impl": args.attn_impl,
        }

    @app.post("/act")
    def act(
        rgb: UploadFile = File(...),
        instruction: str = Form(""),
        reset: bool = Form(False),
        hfov: float = Form(79.0),
        depth_scale_m: float = Form(10.0),
    ):
        del depth_scale_m
        rgb_array = np.asarray(Image.open(io.BytesIO(rgb.file.read())).convert("RGB"))
        with lock:
            if reset:
                evaluator.reset_memory()
                action_queue.clear()
            height, width = rgb_array.shape[:2]
            evaluator.intrinsic_matrix = _intrinsic(width, height, hfov)
            run_model = not action_queue
            actions, generate_time, _llm_output = evaluator.step(
                0,
                rgb_array,
                instruction,
                run_model=run_model,
            )
            evaluator.step_id += 1
            if actions is not None:
                action_queue.extend(int(action) for action in actions)
            action_id = action_queue.popleft() if action_queue else 0
            if action_id == 0:
                action_queue.clear()
        return {
            "action_id": action_id,
            "queued_actions": list(action_queue),
            "generate_time_s": generate_time,
        }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
