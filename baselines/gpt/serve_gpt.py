"""HTTP action server for GPT models through the Codex Python SDK."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.gpt.policy import GPTPolicy, ACTIONS  # noqa: E402


def create_app(args: argparse.Namespace) -> FastAPI:
    policy = GPTPolicy(
        model=args.checkpoint,
        reasoning_effort=args.reasoning_effort,
        stop_range_m=args.stop_range_m,
        max_steps=args.max_steps,
        compact_at_fraction=args.compact_at_fraction,
    )
    app = FastAPI(title="NavVerse GPT action server")
    current_instruction = ""

    @app.get("/health")
    def health():
        return {"status": "ok", "model": policy.model, "runtime": "codex"}

    @app.post("/act")
    def act(
        instruction: str = Form(...),
        reset: str = Form("false"),
        rgb: UploadFile = File(...),
    ):
        nonlocal current_instruction
        if reset.lower() == "true" or instruction != current_instruction:
            current_instruction = instruction
            policy.reset(instruction)

        started = time.perf_counter()
        stats = policy.act(rgb.file.read())
        stats["server_latency_s"] = time.perf_counter() - started
        stats["action_id"] = ACTIONS.index(stats["action"])
        print(json.dumps(stats, ensure_ascii=False), flush=True)
        return JSONResponse(stats)

    @app.on_event("shutdown")
    def shutdown():
        policy.close()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--stop-range-m", type=float, default=1.6)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--compact-at-fraction", type=float, default=0.70)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
