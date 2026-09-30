"""This script is used to warm-up the simulation.
NOTE:
    If the local disk space or the preset local mesh cache size is not enough,
    the warm-up process is likely to fail.

python warmup.py --dirs /data/isaac_scenes_v1 --enable_cameras
"""
import argparse
import os
import time
import torch
import tqdm

parser = argparse.ArgumentParser(description='Warmup the simulation.')
parser.add_argument(
    '-r',
    '--reset',
    required=False,
    action='store_true',
    help='If specified, it will release the old local mesh cache first.',
)
parser.add_argument('-f', '--files', required=False, nargs='*', help='the usd file(s) to warmup')
parser.add_argument('-d', '--dirs', required=False, nargs='*', help='the folders including usd file(s) to warmup')

import utils.rsl_rl_cli_args as rsl_rl_cli_args  # isort: skip
import utils.vln_args as vln_cli_args
from isaaclab.app import AppLauncher


rsl_rl_cli_args.add_rsl_rl_args(parser)
vln_cli_args.add_vln_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = vln_cli_args.parse_args(parser)

# Check cli args
if not args.files and not args.dirs:
    print('No path specified!')
    exit(1)


log_file = f"warmup_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
if os.path.exists(log_file):
    os.remove(log_file)

# SimulationApp should be initialized first before importing omni plugins
app_launcher = AppLauncher(args)
kit = app_launcher.app


from isaacsim.core.utils.prims import define_prim, get_prim_at_path
import omni.physx
import omni.physx.bindings._physx as physx_bindings
from isaaclab.sim import SimulationContext
from pxr import UsdUtils
import isaaclab.sim as sim_utils
from isaacsim.core.api import World


kit.set_setting(physx_bindings.SETTING_UJITSO_COLLISION_COOKING, False)
kit.set_setting(physx_bindings.SETTING_USE_LOCAL_MESH_CACHE, True)
kit.set_setting(physx_bindings.SETTING_LOCAL_MESH_CACHE_SIZE_MB, 1024*100)

import carb, os
settings = carb.settings.get_settings()
settings.set("/renderer/multiGPU/enabled", True)
settings.set("/renderer/activeGpu", 0)
settings.set("/rtx/post/dlss/execMode", 1) # 0: Performance, 1: Balanced, 2: Quality, 3: Auto
# settings.set("/UJITSO/enabled", True)
# settings.set("/UJITSO/geometry", True)
# settings.set("/UJITSO/textures", True)
# settings.set("/UJITSO/materials", True)
# settings.set("/UJITSO/profileCache", True)
# settings.set("/UJITSO/profileCache", True)
# settings.set("/UJITSO/datastore/localCachePath", "/data/isaac_scenes_v1/cache")


ujitso_cooking_enabled = kit._carb_settings.get_as_bool(physx_bindings.SETTING_UJITSO_COLLISION_COOKING)
use_local_cache = kit._carb_settings.get_as_bool(physx_bindings.SETTING_USE_LOCAL_MESH_CACHE)
local_cache_MB = kit._carb_settings.get(physx_bindings.SETTING_LOCAL_MESH_CACHE_SIZE_MB)

print(f'========== ujitso_cooking_enabled: {ujitso_cooking_enabled} ==========')
print(f'========== use_local_cache: {use_local_cache} ==========')
print(f'========== local_cache_MB: {local_cache_MB} ==========')
# with open(log_file, "a+") as f:
#     f.write(f"========== ujitso_cooking_enabled: {ujitso_cooking_enabled} ==========\n")
#     f.write(f"========== use_local_cache: {use_local_cache} ==========\n")
#     f.write(f"========== local_cache_MB: {local_cache_MB} ==========\n")


def test_single(scene_abs_path):
    global world
    """Main simulation loop"""

    
    start_time = 0
    total_start_time = time.time()
    print("[INFO]: Starting simulation")

    kit.context.open_stage(scene_abs_path)
    world = World()
    stage = kit.context.get_stage()
    stageId = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    sim_context = SimulationContext()
    # sim_context._physics_context._physx_sim_interface.attach_stage(stageId)
    # sim_context.play()
    # sim_context.stop()

    # while True:
    #     my_world.step(render=True)

    sim_init = False
    world.reset()
    for frame_count in range(1000000):
        with torch.inference_mode():
            world.step(render=True)
        if frame_count==50:
            start_time = time.time()
        if frame_count==1:
            open(log_file, "a+").write(f"  Load time: {time.time() - total_start_time:.2f}s\n")
    frame_count = 50
    open(log_file, "a+").write(f"  Frame count: {frame_count}\n")
    open(log_file, "a+").write(f"  Simulation time: {time.time() - start_time:.2f}s\n")
    open(log_file, "a+").write(f"  FPS: {frame_count / (time.time() - start_time):.2f}\n")
    kit.context.close_stage()

def test_all(scene_abs_folder):
    # scene_dirs = [_ for _ in os.listdir(scene_abs_folder) if os.path.isdir(os.path.join(scene_abs_folder, _))]
    # for scene_dir in tqdm.tqdm(scene_dirs):
    #     navigation_path = os.path.join(scene_abs_folder, scene_dir, 'start_result_navigation.usd')
    #     interaction_path = os.path.join(scene_abs_folder, scene_dir, 'start_result_interaction.usd')
    #     test_single(navigation_path)
    #     test_single(interaction_path)
    vc_folders = os.path.join(scene_abs_folder, 'virtual_community')
    vc_scenes = [_ for _ in os.listdir(vc_folders) if os.path.isdir(os.path.join(vc_folders, _))]
    vc_scenes = [os.path.join(vc_folders, f"{_}/{_}.usd") for _ in vc_scenes]
    print("========== vc_scenes: ==========\n"+'\n'.join(vc_scenes))
    nvidia_scenes = [
        "nvidia/AECDemo_NVD@10012/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Morning(20Gb).usd",
        "nvidia/AECDemo_NVD@10012/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Night(20Gb).usd",
        "nvidia/AECDemo_NVD@10012/Demos/AEC/BrownstoneDemo/World_BrownstoneDemopack_Brownstone(8Gb).usd",
        "nvidia/AECO_CityDemoPack_NVD@10011/Demos/AEC/TowerDemo/CityDemopack/World_CityDemopack.usd",
        "nvidia/AECO_TowerDemoPack_NVD@10012/Demos/AEC/TowerDemo/TowerDemopack/World_TowerDemopack.usd",
    ]
    nvidia_folder = [os.path.join(scene_abs_folder, _) for _ in nvidia_scenes]
    print("========== nvidia_folder: ==========\n"+'\n'.join(nvidia_folder))
    gr_commercial_folder = os.path.join(scene_abs_folder, 'grscenes_commercial/scenes/')
    gr_commercial_scenes = [_ for _ in os.listdir(gr_commercial_folder) if os.path.isdir(os.path.join(gr_commercial_folder, _))]
    gr_commercial_scenes = [os.path.join(gr_commercial_folder, _+'/start_result_navigation.usd') for _ in gr_commercial_scenes]
    print("========== gr_commercial_scenes: ==========\n"+'\n'.join(gr_commercial_scenes))
    gr_home_folder = os.path.join(scene_abs_folder, 'grscenes_home/scenes/')
    gr_home_scenes = [_ for _ in os.listdir(gr_home_folder) if os.path.isdir(os.path.join(gr_home_folder, _))]
    gr_home_scenes = [os.path.join(gr_home_folder, _+'/start_result_navigation.usd') for _ in gr_home_scenes]
    print("========== gr_home_scenes: ==========\n"+'\n'.join(gr_home_scenes))
    all_scenes = vc_scenes + nvidia_folder + gr_commercial_scenes + gr_home_scenes
    for scene in tqdm.tqdm(all_scenes):
        with open(log_file, "a+") as f:
            f.write(f"Processing {scene} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  File exists: {os.path.exists(scene)}\n")
            f.flush()
            if os.path.exists(scene):
                try:
                    test_single(scene)
                except Exception as e:
                    f.write(f"  Error: {e}\n")
                    import traceback
                    traceback.print_exc()
                    f.write(f"  Traceback: {traceback.format_exc()}\n")
            f.write(f"\n")
if __name__ == '__main__':
    # Release the old local mesh cache if specified
    if args.reset:
        omni.physx.get_physx_cooking_interface().release_local_mesh_cache()

    if args.files:
        for f in args.files:
            f_abs_path = os.path.abspath(f)
            if not os.path.exists(f_abs_path):
                print(f'Error! {f} not found!')
            else:
                test_single(f_abs_path)

    if args.dirs:
        for d in args.dirs:
            d_abs_path = os.path.abspath(d)
            if not os.path.exists(d_abs_path):
                print(f'Error! {d} not found!')
            else:
                test_all(d_abs_path)
    kit.close()