# Usage (run from repo root):
# python scripts/navverse_task_generation.py --enable_cameras --scene_folder navverse_data --tg_config_path configs/vcp.yaml --test_scene_id vcp_amsterdam
import argparse
import json
import os
import signal
import traceback
from pathlib import Path

# start simulation
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Isaac Lab Server for Spot robot with USD scene")

import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
import navverse.vln_args as vln_cli_args

rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)

AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)
args.enable_contact_sensor = True

# Launch Isaac Lab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
args.disable_socket_server = True
if not args.headless:
    args.on_demand_render = False

import carb  # noqa: E402

settings = carb.settings.get_settings()


def app_is_running():
    try:
        return simulation_app.is_running()
    except Exception:
        return False


def pump_gui_events():
    if args.headless:
        return app_is_running()
    if not app_is_running():
        return False
    settings.set("/app/player/playSimulations", False)
    try:
        simulation_app.update()
    except Exception as exc:
        if app_is_running():
            raise
        print(f"[INFO] GUI update stopped during shutdown: {exc}")
        return False
    finally:
        if app_is_running():
            try:
                settings.set("/app/player/playSimulations", True)
            except Exception as exc:
                print(f"[INFO] Failed to restore simulation playback during shutdown: {exc}")
    return app_is_running()

# Local imports

from navverse.sim import VLNSim
from navverse.task_generator import TaskGenerator
from navverse.episode import VLNEpisode


def wait_for_bev_map(task_generator, vln_sim, scene_id, bev_map_path, *, max_steps, label):
    if getattr(vln_sim, "waiting_for_first_action", False):
        vln_sim.waiting_for_first_action = False
    for step in range(max_steps):
        vln_sim.step()
        if bev_map_path.exists():
            print(f"[TG] {label} BEV map ready for {scene_id}: {bev_map_path}")
            return
        if hasattr(task_generator, "bev_camera_lock") and not task_generator.bev_camera_lock.locked():
            print(f"[TG] Waiting for {label} BEV map file after camera job unlocked: {bev_map_path}")
    raise RuntimeError(
        f"Timed out waiting for {label} BEV map for {scene_id}: {bev_map_path} "
        f"after {max_steps} sim steps"
    )


# setup environment
task_generator = TaskGenerator(args)
if args.test_scene_id != "none":
    default_scene_id = args.test_scene_id
else:
    default_scene_id = task_generator.scene_id_list[0]
test_episode = VLNEpisode(task_generator.episode_data(default_scene_id))
vln_sim = VLNSim(args)
vln_sim.reset(test_episode)
vln_sim.step()
if vln_sim.env is not None:
    vln_sim.env.ui_event_pump = pump_gui_events
task_generator.bind_vln_sim(vln_sim)

# disable socket server
print("[INFO] Socket server disabled in task generation")
if not args.headless:
    print("[INFO] On-demand rendering disabled for interactive task generation UI")


# Setup UI
task_generator_ui = None
if not args.headless:
    from navverse.ui import TaskGeneratorUI
    task_generator_ui = TaskGeneratorUI(vln_sim, task_generator)
    task_generator_ui.update_ui("scene_id", test_episode.scene_id)


def update_task_generator_ui(scene_id):
    if task_generator_ui is not None:
        task_generator_ui.update_ui("scene_id", scene_id)


def teleport_robot_from_ui():
    if task_generator_ui is not None:
        task_generator_ui.teleport_robot()


def episode_file_target_reached(scene_id, cfg):
    episode_path = Path(cfg.flow.get("episode_output_dir", "episodes")) / f"{scene_id}.json"
    if not episode_path.exists():
        return False
    try:
        episodes = json.loads(episode_path.read_text())
    except Exception as exc:
        print(f"[TG] Failed to read existing episode file {episode_path}: {exc}")
        return False

    episode_mix = cfg.flow.get("episode_mix")
    if not episode_mix:
        return len(episodes) >= int(cfg.flow.get("episode_number", 50))

    for task_type, target_count in episode_mix.items():
        current_count = sum(episode.get("task_type") == task_type for episode in episodes)
        if current_count < int(target_count):
            return False
    return True


def bev_output_paths(scene_data_dir, file_name):
    return {
        "npz": scene_data_dir / f"{file_name}.npz",
        "rgb": scene_data_dir / f"{file_name}_rgb.png",
        "depth": scene_data_dir / f"{file_name}_depth.png",
        "info": scene_data_dir / f"{file_name}_info.txt",
    }


def bev_outputs_complete(paths):
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def generate_bev_output(scene_id, file_name, *, clip_range, ceiling_height=None, force=False):
    scene_data_dir = task_generator.episode_data_scene_folder(scene_id)
    target_paths = bev_output_paths(scene_data_dir, file_name)
    if not force and bev_outputs_complete(target_paths):
        print(f"[TG] Complete {file_name} outputs already exist for {scene_id}, skipping")
        return

    generated_name = file_name
    generated_paths = target_paths
    if force:
        generated_name = f".{file_name}.regenerate_{os.getpid()}"
        generated_paths = bev_output_paths(scene_data_dir, generated_name)
        for path in generated_paths.values():
            path.unlink(missing_ok=True)

    task_generator.create_bev_map(
        scene_id,
        file_name=generated_name,
        clip_range=clip_range,
        ceiling_height=ceiling_height,
    )
    max_steps = int(task_generator.cfg.flow.get("bev_wait_steps", 10000))
    wait_for_bev_map(
        task_generator,
        vln_sim,
        scene_id,
        generated_paths["npz"],
        max_steps=max_steps,
        label=file_name,
    )
    if not bev_outputs_complete(generated_paths):
        missing = [str(path) for path in generated_paths.values() if not path.is_file() or path.stat().st_size == 0]
        raise RuntimeError(f"Incomplete regenerated BEV outputs for {scene_id}: {missing}")

    if force:
        for output_name, generated_path in generated_paths.items():
            os.replace(generated_path, target_paths[output_name])
        print(f"[TG] Atomically replaced {file_name} outputs for {scene_id}")


def ensure_bev_maps(scene_id, *, force=False):
    if not bool(task_generator.cfg.flow.get("generate_bev_if_missing", True)):
        print(f"[TG] BEV auto-generation disabled for {scene_id}")
        return
    generate_bev_output(
        scene_id,
        "bev_map",
        clip_range="ceiling",
        ceiling_height=task_generator.cfg.scene.get("ceiling_height", 1.0),
        force=force,
    )
    generate_bev_output(
        scene_id,
        "nav_bev_map",
        clip_range="robot",
        force=force,
    )

task_failed = False
if args.tg_mode == "auto":
    scene_id = args.test_scene_id
    try:
        if scene_id != "none":
            task_generator._set_runtime_config(scene_id)
            test_episode = VLNEpisode(task_generator._episode_base_data())
            update_task_generator_ui(scene_id)
            if not args.bev_only:
                existing_status = task_generator.load_status(scene_id)
                reset_attempt_counter = bool(args.replay_episode_data)
                if not args.replay_episode_data and existing_status == "skipped":
                    if episode_file_target_reached(scene_id, task_generator.cfg):
                        print(f"[TG] Task skipped and already generated for {scene_id}, quitting...")
                        os.kill(os.getpid(), signal.SIGKILL)
                    print(f"[TG] Existing skipped status is stale for {scene_id}; continuing generation")
                    reset_attempt_counter = True
                if not args.replay_episode_data and existing_status == "success":
                    if episode_file_target_reached(scene_id, task_generator.cfg):
                        print(f"[TG] Task already generated for {scene_id}, quitting...")
                        task_generator.save_status(scene_id, status="success")
                        os.kill(os.getpid(), signal.SIGKILL)
                    print(f"[TG] Existing success status is stale for {scene_id}; continuing generation")
                    reset_attempt_counter = True
                start_status = {"status": "start"}
                if reset_attempt_counter:
                    start_status.update(
                        checked_episode_attempts=0,
                        last_attempt_stage=None,
                        last_attempt_reason=None,
                        skip_reason=None,
                    )
                task_generator.save_status(scene_id, **start_status)

            if args.bev_only:
                vln_sim.reset(test_episode)
                vln_sim.step()
            else:
                with task_generator.timing_status(scene_id, "load_scene"):
                    vln_sim.reset(test_episode)
                    vln_sim.step()
            if not args.bev_only and bool(task_generator.cfg.flow.get("check_navmesh", True)):
                with task_generator.timing_status(scene_id, "check_navmesh"):
                    task_generator.check_navmesh(scene_id)
                    teleport_robot_from_ui()
            if args.bev_only:
                ensure_bev_maps(scene_id, force=args.force_regenerate_bev)
                print(f"[TG] BEV-only generation completed successfully for {scene_id}")
            elif not args.replay_episode_data:
                with task_generator.timing_status(scene_id, "create_bev"):
                    ensure_bev_maps(scene_id)
            if not args.bev_only:
                timing_name = "replay_episode_data" if args.replay_episode_data else "generate_episodes"
                with task_generator.timing_status(scene_id, timing_name):
                    if args.replay_episode_data:
                        task_generator.replay_episode_data(scene_id)
                    else:
                        task_generator.generate_episodes(scene_id)
                    while True:
                        vln_sim.step()
                        if task_generator.generate_finished:
                            break
                if args.replay_episode_data:
                    if task_generator.replay_episode_failures:
                        raise RuntimeError(
                            f"Episode data replay failed for ids: {sorted(task_generator.replay_episode_failures)}"
                        )
                    print("[TG] Episode data replay completed successfully")
                if task_generator.load_status(scene_id) != "skipped":
                    task_generator.save_status(scene_id, status="success")
    except Exception as e:
        task_failed = True
        print(f"[TG] Error: {e}")
        traceback.print_exc()
        if scene_id != "none" and not args.bev_only:
            task_generator.save_status(scene_id, status="error", error=str(e))
else:
    """Main simulation loop"""
    print("[INFO]: Starting simulation")
    scene_id = args.test_scene_id
    if scene_id != "none":
        task_generator._set_runtime_config(scene_id)
        test_episode = VLNEpisode(task_generator._episode_base_data())
        update_task_generator_ui(scene_id)
        vln_sim.reset(test_episode)
    while app_is_running():
        if not pump_gui_events():
            break
        vln_sim.step()
        if not pump_gui_events():
            break

if args.tg_mode == "auto" and args.bev_only:
    vln_sim.close()
    if app_is_running():
        simulation_app.close(wait_for_replicator=False)
    raise SystemExit(1 if task_failed else 0)

os.kill(os.getpid(), signal.SIGKILL)
