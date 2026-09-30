## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu
## Usage (run from repo root): python replace_usd_models_isaac.py --input_folder navverse_data/grscenes_commercial/models/

import os
import shutil
import sys
import argparse
from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf, Vt, Gf
import time
from utils.mesh_utils import count_total_faces_in_mesh_prim, get_parent_prim_name
import tqdm

def swap_mesh_geometry(original_usd_path, decimated_usd_path, output_usd_path):

    stage_orig = Usd.Stage.Open(original_usd_path)
    try:
        stage_dec = Usd.Stage.Open(decimated_usd_path)
    except Exception as e:
        print(f"[WARNING] Failed to open decimated USD: {decimated_usd_path}\n  {e}\n  Skipping this instance.")
        return

    # Build a name-to-prim map for decimated meshes
    dec_mesh_map = {}
    for prim in stage_dec.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            parent_prim_name = get_parent_prim_name(prim)
            if parent_prim_name is not None:
                if str(parent_prim_name) in prim.GetName():
                    dec_mesh_map[parent_prim_name] = prim
                else:
                    dec_mesh_map[prim.GetName()] = prim

    # For each mesh in the original, if a decimated mesh with the same name exists, swap geometry
    total_faces_original = 0
    total_faces_swapped = 0
    to_remove_prims = []
    for prim in stage_orig.Traverse():
        total_faces_original += count_total_faces_in_mesh_prim(prim)

        # check prim type and mesh face count
        faces = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
        if not prim.IsA(UsdGeom.Mesh) or faces is not None and len(faces)<10:
            continue

        mesh_name = prim.GetName()

        if mesh_name in dec_mesh_map:
            dec_prim = dec_mesh_map[mesh_name]
        else:
            found_mesh_name = []
            found_mesh_path = []
            mesh_path = []
            for key in dec_mesh_map.keys():
                if mesh_name+"_" in key:
                    found = True
                    mesh_path = str(prim.GetPath()).replace("/Root/Instance/", "").split("/")
                    # there are some random SM_XXXXXXX or Group_XXXXXXX path in last but second position
                    if "Group_" in mesh_path[-2] or "SM_" in mesh_path[-2] and len(mesh_path[-2])<=5:
                        mesh_path = mesh_path[:-1]
                    else:
                        mesh_path = mesh_path[:-2]
                    for path_part in mesh_path:
                        if not path_part in str(dec_mesh_map[key].GetPath()):
                            found = False
                            break
                    if found:
                        found_mesh_name.append(key)
                        found_mesh_path.append(dec_mesh_map[key].GetPath())
            if len(found_mesh_name) > 1:
                print("--------------------------------")
                print("usd path:", original_usd_path)
                print(f"Multiple possible meshes found in decimated meshes: {mesh_name}, path: {prim.GetPath()}")
                print(f"matching path: {mesh_path}")
                print("possible mesh:")
                for name, path in zip(found_mesh_name, found_mesh_path):
                    print(f"  {name}: {path}")
                # if os.path.exists(original_usd_path):
                #     print(f"Remove renamed usd file: {original_usd_path}")
                #     os.remove(original_usd_path)
                continue
            elif len(found_mesh_name) == 1:
                dec_prim = dec_mesh_map[found_mesh_name[0]]
            else:
                print("--------------------------------")
                print("usd path:", original_usd_path)
                print(f"Prim not found in decimated meshes: {mesh_name}, path: {prim.GetPath()}")
                print(f"matching path: {mesh_path}")
                for key in dec_mesh_map.keys():
                    if mesh_name+"_" in key:
                        print(f"path not matched:{dec_mesh_map[key].GetPath()}")
                # if os.path.exists(original_usd_path):
                #     print(f"Remove renamed usd file: {original_usd_path}")
                #     os.remove(original_usd_path)
                continue
        for attr_name in [
            "points", "normals", "faceVertexIndices", "faceVertexCounts",
            "primvars:normals", "primvars:st", "primvars:uv"
        ]:
            orig_attr = prim.GetAttribute(attr_name)
            dec_attr = dec_prim.GetAttribute(attr_name)
            if orig_attr and dec_attr and dec_attr.HasAuthoredValue():
                orig_attr.Set(dec_attr.Get())
        total_faces_swapped += count_total_faces_in_mesh_prim(prim)
    
    for path in to_remove_prims:
        stage_orig.RemovePrim(path)

    # print(f"[INFO] Total faces in original meshes: {total_faces_original}")
    # print(f"[INFO] Total faces in swapped meshes: {total_faces_swapped}")

    # Save the modified original stage as the output
    stage_orig.GetRootLayer().Export(output_usd_path)
    # save a "optimized.txt" file in that folder
    with open(f"{os.path.dirname(output_usd_path)}/optimized.txt", "w") as f:
        f.write(f"Total faces in original meshes: {total_faces_original}\n")
    # print(f"Exported USD with original hierarchy/materials and swapped mesh geometry: {output_usd_path}")


def swap_all_decimated_meshes_in_folder(models_folder):
    """
    Iterate through all subfolders in models_folder, find instance.usd and
    corresponding instance_decimated.usd, and swap mesh geometry.
    """
    processed_count = 0
    
    all_folders = []
    for root, dirs, files in os.walk(models_folder):
        # if 'optimized.txt' in files:
        #     continue
        if 'skipped.txt' in files:
            continue
        if 'instance_renamed_decimated.usd' in files and 'instance_renamed.usd' in files:
            all_folders.append(root)
    pbar = tqdm.tqdm(all_folders)
    for folder in pbar:
        pbar.set_description(f"Processing {folder[-20:]}")
        original_usd = os.path.join(folder, 'instance_renamed.usd')
        decimated_usd = os.path.join(folder, 'instance_renamed_decimated.usd')
        output_usd = os.path.join(folder, 'instance.usd')
        swap_mesh_geometry(original_usd, decimated_usd, output_usd)
        processed_count += 1
    
    print(f"[INFO] Processing complete. Total USD files processed: {processed_count}")

# --- SCRIPT USAGE EXAMPLE ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Replace mesh geometry in USD files with decimated versions")
    parser.add_argument("--input_folder", 
                        type=str, 
                        required=True,
                        help="Path to the input folder containing USD files to process")
    
    args = parser.parse_args()
    
    # Use parsed arguments
    input_folder = args.input_folder
    
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Starting mesh replacement process...")
    
    # Validate input folder exists
    if not os.path.exists(input_folder):
        print(f"[ERROR] Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    # Main function call
    swap_all_decimated_meshes_in_folder(input_folder)