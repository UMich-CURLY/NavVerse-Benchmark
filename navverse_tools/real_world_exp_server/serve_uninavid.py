from __future__ import annotations

import argparse
import io
import threading
from types import SimpleNamespace

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uni-NaVid RGB action server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--task-type", choices=("vln", "placenav", "objnav"), default="vln")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from baselines.uninavid.uninavid_agent import UniNaVidAgent

    agent_args = SimpleNamespace(action_type="single", task_type=args.task_type)
    agent = UniNaVidAgent(agent_args)
    lock = threading.Lock()
    app = FastAPI(title="NavVerse Uni-NaVid server")

    @app.get("/health")
    def health():
        return {"ok": True, "model": "Uni-NaVid", "task_type": args.task_type}

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
        if not result or not isinstance(result.get("action"), str):
            raise RuntimeError(f"Uni-NaVid returned no supported action: {result}")
        return result

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
