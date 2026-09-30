
import argparse
import json
import time
import traceback

# start simulation
from isaaclab.app import AppLauncher
import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
import navverse.vln_args as vln_cli_args

# Add command line arguments
parser = argparse.ArgumentParser(description="Benchmark")
rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--gui_max_fps", type=float, default=30.0, help="Limit the GUI loop to keep Kit input/window events responsive. Set <=0 to disable.")
parser.add_argument("--gui_high_fps", action="store_true", help="Maximize GUI FPS by disabling agent-side services and sensor requests while keeping viewport camera binding.")
parser.set_defaults(enable_locomotion_reward=False, disable_camera=False)
args = vln_cli_args.parse_args(parser)
if args.gui_high_fps:
    args.disable_socket_server = True
    args.foxglove_port = 0
    args.enable_locomotion_reward = False
    args.disable_camera = False
    args.on_demand_render = True
    args.enable_cameras = True
    args.gui_max_fps = 0.0
    print("[INFO] GUI high FPS mode: disabled agent socket/reward work, kept robot camera prims for viewport binding, and uncapped GUI loop")

# Launch Isaac Lab app
sim_start_time = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Enable Extension and setup settings
import carb  # noqa: E402
# from isaacsim.core.utils.extensions import enable_extension
# enable_extension("omni.anim.navigation.bundle")
settings = carb.settings.get_settings()
with open("settings.json", "w+") as f:
    json.dump(settings.get_settings_dictionary().get_dict(), f, indent=4)
# settings.set("/renderer/multiGPU/enabled", False)
# settings.set("/renderer/activeGpu", 0) 

def app_is_running():
    try:
        return simulation_app.is_running()
    except Exception:
        return False


def app_is_exiting():
    try:
        return simulation_app.is_exiting()
    except Exception:
        return True


def app_backend_is_running():
    try:
        return simulation_app.app.is_running()
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


vln_sim = None

try:
    if not pump_gui_events():
        raise SystemExit(0)
    # Local imports
    from navverse.sim import VLNSim

    # setup simulation
    default_episode = "test_generator_0"
    vln_sim = VLNSim(args)
    if args.episode_label != "none":
        initial_episode = args.episode_label
    elif default_episode in vln_sim.episode_label_list:
        initial_episode = default_episode
    else:
        initial_episode = vln_sim.episode_label_list[0]
    vln_sim.load_episode(initial_episode)
    # vln_sim.env.env.cfg.sim.render_interval = 10
    vln_sim.step()

    # Setup UI
    benchmark_ui = None
    if not args.headless:
        from navverse.ui import BenchmarkUI
        benchmark_ui = BenchmarkUI(vln_sim)
        benchmark_ui.update_ui("episode_label", initial_episode)

    # Load episode if specified
    if args.episode_label != "none" and args.episode_label != initial_episode:
        vln_sim.load_episode(args.episode_label)
        if benchmark_ui is not None:
            benchmark_ui.update_ui("episode_label", args.episode_label)
        vln_sim.step()

    """Main simulation loop"""
    print(f"[INFO] Starting simulation took {time.time()-sim_start_time:.2f}s")
    step_render_disabled = (args.on_demand_render and args.headless) or args.gui_high_fps
    print(f"[INFO] Evaluation mode: on_demand_render={args.on_demand_render}, headless={args.headless}, step_render_disabled={step_render_disabled}, rebuild_dynamic_navmesh={args.rebuild_dynamic_navmesh}, enable_locomotion_reward={args.enable_locomotion_reward}, disable_camera={args.disable_camera}, gui_high_fps={args.gui_high_fps}, gui_max_fps={args.gui_max_fps}")
    print("[INFO] Starting benchmark loop")
    start_time = time.time()
    start_episode_step = 0
    frame_count = 0
    end_time = 0
    gui_dt = 1.0 / args.gui_max_fps if (not args.headless and args.gui_max_fps > 0) else 0.0
    while app_is_running():
        loop_start = time.time()
        if not pump_gui_events():
            break
        if not app_is_running():
            break
        vln_sim.step()
        if not pump_gui_events():
            break
        # print infos
        frame_count += 1
        if frame_count == 1:
            print(f"[INFO] Loading default scene took {time.time() - start_time:.2f}s")
            pass
        log_fps_interval = 100
        if frame_count % log_fps_interval == 0:
            duration = time.time() - start_time
            total_sim_time = vln_sim.manager_env.common_step_counter * vln_sim.manager_env.step_dt
            sim_time = total_sim_time - start_episode_step * vln_sim.manager_env.step_dt
            print(f"[INFO] Frame count: {frame_count}, Time: {duration:.2f}s, FPS: {log_fps_interval / duration:.2f}, Sim Time: {sim_time:.2f}s ({sim_time/duration:.2f}x), step_dt={vln_sim.manager_env.step_dt}, count={vln_sim.manager_env.common_step_counter}, total_sim_time={total_sim_time:.2f}s")
            start_time = time.time()
            start_episode_step = int(vln_sim.manager_env.common_step_counter)
        if gui_dt > 0:
            sleep_time = gui_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
except KeyboardInterrupt:
    print("[INFO] Keyboard interrupt received, shutting down")
except Exception as exc:
    print(f"[ERROR] Benchmark initialization or loop failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    raise
finally:
    print("[INFO] Shutting down benchmark")
    try:
        settings.set("/app/player/playSimulations", False)
    except Exception:
        pass
    if vln_sim is not None:
        vln_sim.close()

    if app_backend_is_running() and not app_is_exiting():
        simulation_app.close(wait_for_replicator=False)
    else:
        print("[INFO] Kit is already exiting; skipping SimulationApp.close()")
