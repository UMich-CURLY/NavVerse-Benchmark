#!/usr/bin/env python3
"""HTTP evaluation bridge for the public LongNav-R1 policy."""

import argparse
import importlib.util
import io
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional


def longnav_package_root():
    spec = importlib.util.find_spec("longnav")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Run bash baselines/longnav_r1/setup.sh and use its .venv/bin/python")
    return Path(next(iter(spec.submodule_search_locations)))


def resolve_checkpoint(checkpoint):
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=checkpoint,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "config.yaml"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Aasdfip/hm3d_rpp_ke_standard-checkpoint_231")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--attn_impl", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--stop_prob_threshold", type=float, default=None)
    args = parser.parse_args()
    if args.stop_prob_threshold is not None and not 0 <= args.stop_prob_threshold <= 1:
        parser.error("--stop_prob_threshold must be in [0, 1]")
    return args


def discrete_action_response(action_id, action_probs) -> dict:
    action_id = int(action_id)
    if action_id not in {0, 1, 2, 3}:
        raise RuntimeError(f"LongNav produced unsupported action_id: {action_id}")
    probabilities = (
        action_probs.tolist() if hasattr(action_probs, "tolist") else list(action_probs)
    )
    return {"action_id": action_id, "action_probs": probabilities}

def _create_discrete_backend(args, package_root):
    import numpy as np

    from longnav.config_schema import RolloutConfig, VLMConfig, VLMTrainingConfig
    from longnav.utils.factories import get_base_model
    from longnav.utils.rollout_core import RLWorker, substitute_convo_template

    rollout_cfg = RolloutConfig()
    rollout_cfg.stop_prob_threshold = args.stop_prob_threshold
    vlm_cfg = VLMConfig()
    vlm_cfg.attn_impl = args.attn_impl
    training_cfg = VLMTrainingConfig()

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    vlm_cfg.model_id = get_base_model(checkpoint_path)
    system_prompt_path = (
        Path(args.system_prompt).expanduser().resolve()
        if args.system_prompt
        else package_root / "conf" / "prompts" / "objectnav_prompt.txt"
    )
    system_prompt = system_prompt_path.read_text()
    rollout_cfg.convo_start_template = [
        {"role": "user", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
    ]

    worker = RLWorker(asdict(rollout_cfg), **asdict(vlm_cfg))
    worker._setup_peft(training_cfg)
    worker.load_checkpoint(checkpoint_path, False, False)

    def reset():
        worker.reset()

    def infer(rgb_pil, instr_or_goal=None):
        if instr_or_goal is not None:
            worker.reset()
            worker.messages = substitute_convo_template(
                worker.rollout_config["convo_start_template"],
                {"instr_or_goal": instr_or_goal} | worker.rollout_config,
            )
        action_probs, _action_logprobs, _ = worker.infer_probs(
            images=[rgb_pil],
            messages=worker.messages,
            temperature=worker.rollout_config["temperature"],
            pos_id_kwargs=None,
        )
        action_id = np.random.choice(len(action_probs), p=action_probs)
        stop_threshold = worker.rollout_config.get("stop_prob_threshold")
        if action_id == 0 and stop_threshold is not None and action_probs[0] < stop_threshold:
            tail = action_probs[1:] / np.sum(action_probs[1:])
            action_id = np.random.choice(len(action_probs) - 1, p=tail) + 1
        worker.messages = substitute_convo_template(
            worker.rollout_config["convo_turn_template"],
            {"action": worker.rollout_config["action_space"][action_id]},
        )
        return discrete_action_response(action_id, action_probs)

    return reset, infer, "discrete_action"


def main() -> None:
    args = parse_args()
    package_root = longnav_package_root()

    import uvicorn
    from fastapi import FastAPI, File, Form, UploadFile
    from PIL import Image

    reset_backend, infer_step, output_type = _create_discrete_backend(args, package_root)
    lock = threading.Lock()
    app = FastAPI(title="LongNav-R1 inference server")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "port": args.port,
            "policy_backend": "discrete",
            "output": output_type,
        }

    @app.post("/reset")
    def reset():
        with lock:
            reset_backend()
        return {"ok": True}

    @app.post("/act")
    def get_action(
        file: UploadFile = File(...),
        instr_or_goal: Optional[str] = Form(None),
    ):
        image_bytes = file.file.read()
        rgb_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with lock:
            return infer_step(rgb_pil, instr_or_goal)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
