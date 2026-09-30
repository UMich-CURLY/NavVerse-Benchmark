from __future__ import annotations

import argparse
import io
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image


POLIFORMER_ACTIONS = {
    "STOP",
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "TURN_LEFT_SMALL",
    "TURN_RIGHT_SMALL",
}


def discrete_action_response(result) -> dict[str, str]:
    action = result.get("action") if isinstance(result, dict) else None
    if action not in POLIFORMER_ACTIONS:
        raise RuntimeError(f"PoliFormer returned no supported action: {result}")
    return {"action": action}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NavVerse PoliFormer RGB action server")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "scratch" / "poliformer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"PoliFormer checkpoint not found: {args.checkpoint}")

    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "OBJAVERSE_HOUSES_DIR",
        str(args.data_dir / "objaverse_houses" / "houses_2023_07_28"),
    )
    os.environ.setdefault("OBJAVERSE_DATA_DIR", str(args.data_dir / "objaverse_assets"))
    os.environ.setdefault("DETIC_REPO_PATH", str(repo_root / "baselines" / "submodules" / "Detic"))
    os.environ.setdefault("HF_HOME", str(Path.home() / "scratch" / "hugging_face_cache"))
    sys.path.insert(0, str(repo_root / "baselines"))
    from baselines.poliformer.poliformer_agent import PoliFormerAgent

    upstream_argv = [
        sys.argv[0],
        "--ckpt_path",
        str(args.checkpoint),
        "--num_workers",
        "1",
        "--training_tag",
        "text-nav",
        "--house_set",
        "objaverse",
    ]
    if args.greedy:
        upstream_argv.append("--greedy")
    original_argv = sys.argv
    try:
        sys.argv = upstream_argv
        agent = PoliFormerAgent(SimpleNamespace())
    finally:
        sys.argv = original_argv

    lock = threading.Lock()
    app = FastAPI(title="NavVerse PoliFormer server")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": "PoliFormer",
            "checkpoint": str(args.checkpoint),
            "output": "discrete_action",
        }

    @app.post("/reset")
    def reset():
        with lock:
            agent.reset()
        return {"ok": True}

    @app.post("/act")
    def act(
        rgb: UploadFile = File(...),
        instruction: str = Form(...),
        reset: bool = Form(False),
        hfov: float = Form(79.0),
        depth_scale_m: float = Form(10.0),
    ):
        del hfov, depth_scale_m
        rgb_array = np.asarray(Image.open(io.BytesIO(rgb.file.read())).convert("RGB"))
        with lock:
            if reset:
                agent.reset()
            result = agent.act({"rgb": rgb_array, "objectgoal": instruction})
        return discrete_action_response(result)

    uvicorn.run(app, host=args.host, port=args.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
