import argparse
import cProfile
import io
import json
import os
import pstats
import time
from pathlib import Path

from isaaclab.app import AppLauncher
import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
import navverse.vln_args as vln_cli_args

parser = argparse.ArgumentParser()
rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
parser.add_argument("--scene_usd", required=True)
parser.add_argument("--output_json", required=True)
parser.add_argument("--mode", choices=["normal_camera", "no_camera"], default="normal_camera")
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--external_render", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)
args.enable_cameras = args.mode != "no_camera"
args.disable_camera = args.mode == "no_camera"

app_launcher = AppLauncher(args)
app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from navverse.robots.spot.spot_rough_env_cfg import SpotRoughEnvCfg_PLAY

env_cfg = SpotRoughEnvCfg_PLAY()
env_cfg.load_usd(args.scene_usd)
env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.0)
env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
env_cfg.curriculum = None
env_cfg.num_rerenders_on_reset = 0
env_cfg.wait_for_textures = False
if args.mode == "no_camera":
    if hasattr(env_cfg.observations, "camera"):
        env_cfg.observations.camera = None
    env_cfg.scene.pov_camera = None
    env_cfg.scene.third_person_camera = None

t0 = time.perf_counter()
env = ManagerBasedRLEnv(cfg=env_cfg)
construct_s = time.perf_counter() - t0
env.reset()
action_dim = getattr(getattr(env, "action_manager", None), "total_action_dim", None)
if action_dim is None:
    action_dim = env.action_space.shape[-1]
actions = torch.zeros((env.num_envs, action_dim), device=env.device)
for _ in range(args.warmup):
    env.step(actions)
    if args.external_render:
        env.sim.render()

prof = cProfile.Profile()
t0 = time.perf_counter()
prof.enable()
for _ in range(args.frames):
    env.step(actions)
    if args.external_render:
        env.sim.render()
prof.disable()
elapsed_s = time.perf_counter() - t0
stream = io.StringIO()
stats = pstats.Stats(prof, stream=stream).strip_dirs().sort_stats("cumtime")
stats.print_stats(80)
rows = []
for func, stat in stats.stats.items():
    cc, nc, tt, ct, callers = stat
    filename, line, name = func
    rows.append({"func": f"{filename}:{line}:{name}", "primitive_calls": cc, "calls": nc, "tottime_s": tt, "cumtime_s": ct})
rows.sort(key=lambda r: r["cumtime_s"], reverse=True)
result = {
    "scene_usd": args.scene_usd,
    "construct_s": construct_s,
    "frames": args.frames,
    "external_render": args.external_render,
    "elapsed_s": elapsed_s,
    "step_render_fps": args.frames / elapsed_s,
    "top_cumtime": rows[:80],
    "pstats_text": stream.getvalue(),
}
Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"output_json": args.output_json, "construct_s": construct_s, "step_render_fps": result["step_render_fps"], "top10": rows[:10]}, indent=2), flush=True)
os._exit(0)
