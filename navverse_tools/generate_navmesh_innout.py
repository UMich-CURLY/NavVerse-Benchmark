import argparse
import os
import sys
import glob
import subprocess
import tempfile
import numpy as np
from pathlib import Path

try:
    from pxr import Usd, UsdGeom
except ImportError:
    print("Warning: pxr module not found. USD operations will be skipped.")
    Usd = None

class SuppressFD:
    def __init__(self, fd=2):
        self.fd = fd
        self.devnull = None
        self.save_fd = None

    def __enter__(self):
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        self.save_fd = os.dup(self.fd)
        os.dup2(self.devnull, self.fd)

    def __exit__(self, *args):
        os.dup2(self.save_fd, self.fd)
        os.close(self.devnull)
        os.close(self.save_fd)

# Add repo root to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

from navverse.navmesh import NavmeshInterface
from navverse.config import load_config, resolve_path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate navmesh from USD files")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing files")
    parser.add_argument("--filter", type=str, default=None, help="Filter string to match files (e.g., 'innout')")
    parser.add_argument("--config_path", type=str, default="configs/innout.yaml", help="Path to task config yaml")
    return parser.parse_args()

def extract_mesh_with_blender(usd_path, output_dir):
    # Blender script content
    blender_script = f"""
import bpy
import json
import numpy as np
import os

def convert_to_mesh(obj):
    # Apply all modifiers and convert to mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    object_eval = obj.evaluated_get(depsgraph)
    mesh = object_eval.to_mesh()
    return mesh

def main():
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import USD
    bpy.ops.wm.usd_import(filepath="{usd_path}")
    
    verts_path = os.path.join("{output_dir}", "vertices.bin")
    faces_path = os.path.join("{output_dir}", "faces.bin")
    
    f_verts = open(verts_path, 'wb')
    f_faces = open(faces_path, 'wb')
    
    vert_offset = 0
    
    # Collect mesh objects first
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    for obj in mesh_objects:
        # Get world matrix
        matrix_world = obj.matrix_world
        
        # Get mesh data
        mesh = convert_to_mesh(obj)
        
        # Transform vertices to world space
        # mesh.vertices coordinates are local
        # Doing this list comprehension is cleaner than low-level numpy for now to ensure transform correctness
        verts = [matrix_world @ v.co for v in mesh.vertices]
        
        n_verts = len(verts)
        if n_verts == 0:
            obj.to_mesh_clear()
            continue
            
        # Convert to numpy and write directly
        # Flattening (x, y, z)
        verts_np = np.empty((n_verts, 3), dtype=np.float32)
        for i, v in enumerate(verts):
            verts_np[i] = (v.x, v.y, v.z)
            
        verts_np.tofile(f_verts)
        
        # Get faces (loop triangles are triangulated faces)
        mesh.calc_loop_triangles()
        
        # Collect faces
        # We need to flatten the list of indices
        faces_list = []
        for tri in mesh.loop_triangles:
            # Add offset to indices
            faces_list.extend([v_index + vert_offset for v_index in tri.vertices])
            
        if faces_list:
            np.array(faces_list, dtype=np.int32).tofile(f_faces)
        
        vert_offset += n_verts
        
        # Clean up
        obj.to_mesh_clear()
            
    f_verts.close()
    f_faces.close()

if __name__ == "__main__":
    main()
"""
    # Write blender script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(blender_script)
        script_path = f.name
        
    # Run Blender
    cmd = ["blender", "--background", "--python", script_path]
    
    try:
        print(f"Running Blender extraction for {usd_path}...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Blender failed: {e}")
        # print stderr for debugging
        print(e.stderr.decode())
        raise
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def find_scene_config(file_path, task_config):
    # Normalize file path to compare with config paths
    # The config paths seem to be relative to 'data' folder typically, but let's check
    # e.g., vc_plus/AMSTERDAM/AMSTERDAM_gr_merged.usd
    
    abs_file_path = os.path.abspath(file_path)
    
    for scene_type, scene_data in task_config.get('scene', {}).items():
        episodes = scene_data.get('episodes', {})
        
        for scene_id, episode_data in episodes.items():
            config_path = episode_data.get('path')
            if not config_path:
                continue
                
            # We check if the config_path is a suffix of the file_path
            # This handles cases where file_path is absolute and config_path is relative
            if abs_file_path.endswith(config_path):
                return {
                    'scene_type': scene_type,
                    'scene_id': scene_id,
                    'navmesh_preset': episode_data.get('navmesh_preset', 'default'),
                }
    return None


def convert_up_axis(vertices, inverse=False):
    '''
    Convert all data between navmesh interface and end-user to be in the correct up axis

    pyrecast assumes y-up, so only change when z-up is true
    inverse = True will convert y-up output back to z-up
    '''
    
    vertices = np.array(vertices)
    v_copy = np.empty(shape=vertices.shape)
    if inverse:
        # Convert data we have been given in y-up into the z-up of scene (e.g. output from recast)
        v_copy[:, 0], v_copy[:, 1], v_copy[:, 2] = vertices[:, 0], -vertices[:, 2], vertices[:, 1]
    else: 
        # Convert the up axis from Y to Z
        v_copy[:, 0], v_copy[:, 1], v_copy[:, 2] = vertices[:, 0], vertices[:, 2], -vertices[:, 1]

    vertices = v_copy
    return vertices

def generate_navmesh_from_data(vertices, faces, output_path, navmesh_settings={}):
    # Prepare for NavmeshInterface
    if len(faces) > 0:
        faces_reshaped = faces.reshape(-1, 3)
        threes = np.full((faces_reshaped.shape[0], 1), 3, dtype=np.int32)
        formatted_faces_np = np.hstack((threes, faces_reshaped))
        formatted_faces = formatted_faces_np.flatten().tolist()
    else:
        formatted_faces = []
        
    # Initialize NavmeshInterface
    nm_interface = NavmeshInterface(up_axis='Z')
    
    # Check for NaNs
    if np.isnan(vertices).any():
        print("Error: Input vertices contain NaNs")
        return
    
    # Update settings
    # Ensure tileSize is set for large scenes if not provided
    if 'tileSize' not in navmesh_settings:
        navmesh_settings['tileSize'] = 256
        
    nm_interface.update_settings(navmesh_settings)
    
    # Convert coordinate system
    # verts_y_up = nm_interface._convert_up_axis(vertices)
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    print(f"Geometry Bounds: Min {min_bounds}, Max {max_bounds}")
    print(f"Geometry Size: {max_bounds - min_bounds}")
    
    verts_flat = vertices.flatten().tolist()
    
    print("Building navmesh...")
    nm_interface.nm.init_by_raw(verts_flat, formatted_faces)
    nm_interface.build_navmesh()
    nm_interface.save_navmesh(output_path)
    print("Done.")

def get_bbox_center(prim):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    bound = bbox_cache.ComputeWorldBound(prim)
    range3d = bound.ComputeAlignedBox()
    min_point = range3d.GetMin()
    max_point = range3d.GetMax()
    center = (min_point + max_point) / 2.0
    return center

def filter_by_room_distance(vertices, faces, usd_path, max_dist=100.0):
    if not Usd:
        return vertices, faces
        
    print(f"Loading {usd_path} to filter geometry...")
    try:
        with SuppressFD():
            stage = Usd.Stage.Open(usd_path)
            room_prim = stage.GetPrimAtPath("/start_result_navigation")
            if not room_prim:
                room_prim = stage.GetPrimAtPath("/World/start_result_navigation")
            
            if not room_prim:
                print("Warning: /start_result_navigation not found in USD. Skipping filtering.")
                return vertices, faces
                
            center_gf = get_bbox_center(room_prim)
            room_center = np.array([center_gf[0], center_gf[1], center_gf[2]])
            print(f"Found room center at: {room_center}")
            
            # Calculate distances (only XY)
            # vertices are typically (x, y, z) here (Z-up in raw bin, but check expectation)
            # The generate_navmesh script loads raw bin which is Z-up (from Blender export of Z-up stage)
            # So indices 0 and 1 are X and Y.
            
            diff = vertices[:, :2] - room_center[:2]
            dist_sq = np.sum(diff**2, axis=1)
            
            valid_mask = dist_sq <= max_dist**2
            
            if np.sum(valid_mask) < len(vertices):
                num_removed = len(vertices) - np.sum(valid_mask)
                print(f"[INFO]: Removing {num_removed} vertices > {max_dist}m from room")
                
                # Re-index faces
                old_to_new = np.full(len(vertices), -1, dtype=np.int32)
                valid_indices = np.where(valid_mask)[0]
                old_to_new[valid_indices] = np.arange(len(valid_indices))
                
                # Update vertices
                new_vertices = vertices[valid_mask]
                
                # Update faces
                if len(faces) > 0:
                    faces_arr = faces.reshape(-1, 3)
                    new_faces = old_to_new[faces_arr]
                    
                    # Keep face only if all vertices are valid
                    valid_faces_mask = np.all(new_faces != -1, axis=1)
                    new_faces_filtered = new_faces[valid_faces_mask]
                    new_faces_flat = new_faces_filtered.flatten()
                    
                    print(f"[INFO]: Remaining faces: {len(new_faces_filtered)}")
                    return new_vertices, new_faces_flat
                else:
                    return new_vertices, faces
            else:
                print(f"[INFO]: All vertices within {max_dist}m of room.")
                return vertices, faces

    except Exception as e:
        print(f"Error reading USD for filtering: {e}")
        return vertices, faces

def process_bin_pair(v_path, f_path, task_config):
    # Determine output path
    dir_name = os.path.dirname(v_path)
    base_name = os.path.basename(v_path)
    
    if base_name.endswith("_vertices.bin"):
        prefix = base_name[:-13] # remove "_vertices.bin"
        output_filename = f"{prefix}_navmesh.bin"
    else:
        output_filename = os.path.splitext(base_name)[0] + "_navmesh.bin"
        
    output_path = os.path.join(dir_name, output_filename)
    
    print(f"Processing {v_path} & {f_path} -> {output_path}")
    
    try:
        vertices = np.fromfile(v_path, dtype=np.float32).reshape(-1, 3)
        vertices = convert_up_axis(vertices)
        if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
            faces = np.fromfile(f_path, dtype=np.int32)
        else:
            faces = np.array([], dtype=np.int32)
            
        print(f"Loaded {len(vertices)} vertices and {len(faces)//3} triangles")
        
        # Filter vertices based on room distance if USD file exists
        usd_path = None
                         
        # Check in ../vc_apr14 from the bin file directory
        if not usd_path and prefix.startswith("innout_"):
             city_name = prefix[7:] # e.g. "amsterdam"
             
             # Go up one level from bin dir to find vc_apr14
             # e.g. data/navmesh -> data/vc_apr14
             parent_dir = os.path.dirname(dir_name)
             vc_plus_dir = os.path.join(parent_dir, "vc_apr14")
             
             if os.path.exists(vc_plus_dir):
                 # Find city dir (case insensitive)
                 target_city_dir = None
                 for d in os.listdir(vc_plus_dir):
                     if d.lower() == city_name.lower():
                         target_city_dir = os.path.join(vc_plus_dir, d)
                         break
                 
                 if target_city_dir:
                     # Construct USD path: CITY_gr_merged.usd
                     # The filename usually matches the directory name case
                     city_dir_name = os.path.basename(target_city_dir)
                     candidate_usd = os.path.join(target_city_dir, f"{city_dir_name}_gr_merged.usd")
                     if os.path.exists(candidate_usd):
                         usd_path = candidate_usd
        
        if usd_path:
            print(f"Using USD file for room filtering: {usd_path}")
            vertices, faces = filter_by_room_distance(vertices, faces, usd_path, max_dist=100.0)
        else:
            print(f"Warning: USD file not found for {prefix}. Skipping distance filtering.")

        vertices = convert_up_axis(vertices)

        # Try to find config based on filename
        navmesh_settings = dict(task_config.get("navmesh") or {})
        
        # Split prefix by underscore
        parts = prefix.split('_')
        
        found_config = bool(navmesh_settings)
        
        if 'scene' in task_config:
            # Try to match parts to scene types
            for scene_type in task_config['scene']:
                if scene_type in parts:
                    # found scene type, now look for scene id in the rest
                    episodes_config = task_config['scene'][scene_type].get('episodes', {})
                    
                    # Try to match any remaining part or combination of parts to episode keys
                    # This is a bit heuristic
                    
                    # Iterate through all episode keys and check if they exist in parts
                    for episode_key, episode_data in episodes_config.items():
                        # exact match in parts
                        if episode_key in parts:
                            preset_name = episode_data.get('navmesh_preset')
                            if preset_name:
                                if 'navmesh' in task_config and preset_name in task_config['navmesh']:
                                    navmesh_settings = task_config['navmesh'][preset_name]
                                    print(f"Found config for {prefix} (matched {scene_type}/{episode_key}): using preset '{preset_name}'")
                                    found_config = True
                                    break
                    
                    if found_config:
                        break
                        
            # If not found by explicit scene type match, try brute force search across all scenes
            if not found_config:
                for scene_type, scene_data in task_config['scene'].items():
                    episodes_config = scene_data.get('episodes', {})
                    for episode_key, episode_data in episodes_config.items():
                        # Check if episode_key is a substring of prefix or vice versa
                        # or if episode_key is in the parts
                         if episode_key in parts or episode_key == prefix:
                            preset_name = episode_data.get('navmesh_preset')
                            if preset_name:
                                if 'navmesh' in task_config and preset_name in task_config['navmesh']:
                                    navmesh_settings = task_config['navmesh'][preset_name]
                                    print(f"Found config for {prefix} (matched {scene_type}/{episode_key}): using preset '{preset_name}'")
                                    found_config = True
                                    break
                    if found_config:
                        break
        
        if not found_config:
             print(f"Warning: No matching config found for {prefix}")
        
        generate_navmesh_from_data(vertices, faces, output_path, navmesh_settings)

    except Exception as e:
        print(f"Error processing {v_path}: {e}")
        raise

def main():
    args = parse_args()
    
    # Load config
    config_path = resolve_path(args.config_path, base_dir=Path(repo_root))
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    task_config = load_config(config_path)
    
    if args.filter:
        # Search for bin files
        search_pattern = os.path.join(args.input_folder, "**", f"{args.filter}*_vertices.bin")
        print(f"Searching for {search_pattern}...")
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
             # Try flat search
             search_pattern = os.path.join(args.input_folder, f"{args.filter}*_vertices.bin")
             files = glob.glob(search_pattern)
             
        if not files:
             print("No matching vertices files found.")
             return
             
        print(f"Found {len(files)} vertex files to process.")
        for v_path in files:
            # Construct faces path
            # Assume it's in the same dir and has _faces.bin suffix instead of _vertices.bin
            # Check if it ends with _vertices.bin
            if v_path.endswith("_vertices.bin"):
                base = v_path[:-13] 
                f_path = base + "_faces.bin"
            else:
                 # Should not happen given glob, but safety
                 continue
                 
            process_bin_pair(v_path, f_path, task_config)

if __name__ == "__main__":
    main()
