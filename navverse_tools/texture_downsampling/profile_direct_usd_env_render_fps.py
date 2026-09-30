import argparse
import cProfile
import io
import json
import os
import pstats
import subprocess
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
parser.add_argument("--render_warmup", type=int, default=100)
parser.add_argument("--render_frames", type=int, default=500)
parser.add_argument("--step_warmup", type=int, default=20)
parser.add_argument("--step_frames", type=int, default=500)
parser.add_argument("--step_external_render", action="store_true")
parser.add_argument("--texture_streaming", choices=["default", "on", "off"], default="default")
parser.add_argument("--hold_after_profile", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)
args.enable_cameras = args.mode != "no_camera"
args.disable_camera = args.mode == "no_camera"
if args.texture_streaming != "default":
    old = "--/rtx-transient/resourcemanager/enableTextureStreaming=false"
    new = "--/rtx-transient/resourcemanager/enableTextureStreaming=true" if args.texture_streaming == "on" else old
    args.kit_args = args.kit_args.replace(old, new)
start = time.perf_counter()
events = []

def vram_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception as exc:
        return {"error": repr(exc)}

def ev(name, extra=None):
    rec = {"name": name, "elapsed_total_s": time.perf_counter() - start}
    if extra:
        rec.update(extra)
    events.append(rec)
    print("[PROFILE]", json.dumps(rec, sort_keys=True), flush=True)

app_launcher = AppLauncher(args)
app = app_launcher.app
ev("app_launcher_ready", {"vram_mb": vram_mb()})
import carb
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors.camera.tiled_camera import TiledCamera
from isaaclab.sensors.camera.camera import Camera
from isaaclab.sim import SimulationContext
from isaaclab.envs import manager_based_env as mbe
from isaaclab.scene.interactive_scene import InteractiveScene
from navverse.robots.spot.spot_rough_env_cfg import SpotRoughEnvCfg_PLAY

def wrap(cls, method, label=None):
    if not hasattr(cls, method):
        return
    original = getattr(cls, method)
    label = label or f"{cls.__name__}.{method}"
    def wrapped(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            carb.profiler.begin(1, label)
        except Exception:
            pass
        try:
            return original(self, *a, **kw)
        finally:
            try:
                carb.profiler.end(1)
            except Exception:
                pass
            ev(label, {"elapsed_s": time.perf_counter() - t0})
    setattr(cls, method, wrapped)

for cls, methods in [
    (SimulationContext, ["reset", "play", "render", "forward"]),
    (InteractiveScene, ["__init__", "reset", "update"]),
    (mbe.ManagerBasedEnv, ["__init__", "load_managers", "reset"]),
    (ManagerBasedRLEnv, ["__init__", "load_managers", "_reset_idx"]),
    (TiledCamera, ["__init__", "_initialize_impl", "reset"]),
    (Camera, ["__init__", "_initialize_impl", "reset"]),
]:
    for method in methods:
        wrap(cls, method)

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
ev("config_ready", {"scene_usd": args.scene_usd, "vram_mb": vram_mb()})
prof = cProfile.Profile()
t0 = time.perf_counter()
env = prof.runcall(ManagerBasedRLEnv, cfg=env_cfg)
ev("construct_total", {"elapsed_s": time.perf_counter() - t0, "vram_mb": vram_mb()})
s = io.StringIO()
pstats.Stats(prof, stream=s).strip_dirs().sort_stats("cumtime").print_stats(60)
construct = s.getvalue()
print("[CPROFILE_CONSTRUCT]\n" + construct, flush=True)
t0 = time.perf_counter()
obs, info = env.reset()
ev("reset_total", {"elapsed_s": time.perf_counter() - t0, "vram_mb": vram_mb()})
for _ in range(args.render_warmup):
    env.sim.render()
t0 = time.perf_counter()
for _ in range(args.render_frames):
    env.sim.render()
render_elapsed = time.perf_counter() - t0
render_fps = args.render_frames / render_elapsed if render_elapsed > 0 else None
ev("render_benchmark", {"warmup_frames": args.render_warmup, "frames": args.render_frames, "elapsed_s": render_elapsed, "fps": render_fps, "vram_mb": vram_mb()})
action_dim = getattr(getattr(env, "action_manager", None), "total_action_dim", None)
if action_dim is None:
    action_dim = env.action_space.shape[-1]
actions = torch.zeros((env.num_envs, action_dim), device=env.device)
for _ in range(args.step_warmup):
    env.step(actions)
    if args.step_external_render:
        env.sim.render()
t0 = time.perf_counter()
for _ in range(args.step_frames):
    env.step(actions)
    if args.step_external_render:
        env.sim.render()
step_elapsed = time.perf_counter() - t0
step_fps = args.step_frames / step_elapsed if step_elapsed > 0 else None
ev("step_render_benchmark", {"warmup_frames": args.step_warmup, "frames": args.step_frames, "elapsed_s": step_elapsed, "fps": step_fps, "external_render": args.step_external_render, "vram_mb": vram_mb()})
Path(args.output_json).write_text(json.dumps({"events": events, "construct_cprofile_top": construct, "scene_usd": args.scene_usd, "mode": args.mode, "render_elapsed_s": render_elapsed, "render_fps": render_fps, "step_render_elapsed_s": step_elapsed, "step_render_fps": step_fps}, indent=2, sort_keys=True) + "\n")
print(json.dumps({"output_json": args.output_json, "render_fps": render_fps, "step_render_fps": step_fps, "last_events": events[-12:]}, indent=2), flush=True)
if args.hold_after_profile:
    ev("hold_after_profile_start", {"message": "Keeping GUI alive; close the app or press Ctrl-C to exit."})
    try:
        while app.is_running():
            env.sim.render()
            app.update()
            time.sleep(0.02)
    except KeyboardInterrupt:
        ev("hold_after_profile_stop", {"reason": "KeyboardInterrupt"})
    os._exit(0)
os._exit(0)
