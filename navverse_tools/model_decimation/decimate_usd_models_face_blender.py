# Usage (run from repo root):
# blender --background --python decimate_usd_models_face_blender.py -- --input_folder navverse_data/grscenes_commercial/models/ --ratio 0.1

## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu
import shutil
import sys
import os
print(sys.executable)
print(sys.version)
import site
sys.path.append(site.getusersitepackages())
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import json
import random
import bpy
import os
import argparse
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)
try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf, Vt, Gf
    from utils.mesh_utils import count_total_faces_in_mesh_prim, get_parent_prim_name
except ModuleNotFoundError:
    Usd = UsdGeom = UsdShade = Sdf = Tf = Vt = Gf = None

    def count_total_faces_in_mesh_prim(_prim):
        return 0

    def get_parent_prim_name(_prim):
        return None
from utils.uv_utils import apply_decimate_modifiers, apply_dissolve_modifiers, apply_remesh_modifiers, protect_uv_seams
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)  # also flush errors immediately


class SuppressBlenderOutput:
    def __enter__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

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

        dec_prim = None
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
                    if len(mesh_path) >= 2:
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
                # print(f"matching path: {mesh_path}")
                # for key in dec_mesh_map.keys():
                #     if mesh_name+"_" in key:
                #         print(f"path not matched:{dec_mesh_map[key].GetPath()}")
                # if os.path.exists(original_usd_path):
                #     print(f"Remove renamed usd file: {original_usd_path}")
                #     os.remove(original_usd_path)
                continue
        
        if dec_prim:
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
    # print(f"Exported USD with original hierarchy/materials and swapped mesh geometry: {output_usd_path}")

def export_usd(output_usd_path):
    """Export current Blender scene to USD."""
    export_kwargs = {
        "filepath": output_usd_path,
        "relative_paths": True,
        "export_materials": False,
        "export_animation": False,
        "export_uvmaps": True,
        "use_instancing": True,
        "triangulate_meshes": True,
    }
    try:
        with SuppressBlenderOutput():
            try:
                bpy.ops.wm.usd_export(**export_kwargs)
            except TypeError as exc:
                if "triangulate_meshes" not in str(exc):
                    raise
                export_kwargs.pop("triangulate_meshes", None)
                bpy.ops.wm.usd_export(**export_kwargs)
        print(f"[INFO] Decimation and export complete: {output_usd_path}")
    except Exception as e:
        print(f"[ERROR] Export USD failed: {e}")

def decimate_usd_meshes(
    input_usd_path,
    output_usd_path,
    original_usd_path,
    info_path,
    ratio=0.1,
    skip_uv_seams=False,
    disable_remesh=False,
):
    """
    Imports a USD file, decimates all meshes, and exports to a new USD file.
    
    Args:
        input_usd_path (str): The full path to the USD file to import.
        output_usd_path (str): The full path for the decimated USD file.
        original_usd_path (str): The full path for the original USD file to swap geometry into.
        info_path (str): The full path for the info json file.
        ratio (float): The decimation ratio. A value of 0.1 means 10% of the original faces.
    """
    print(f"================================================")
    
    # Check if the input file exists
    if not os.path.exists(input_usd_path):
        print(f"Error: Input file not found at {input_usd_path}")
        return
    
    # Check if the output file exists
    if os.path.exists(output_usd_path):
        print(f"Output file found at {output_usd_path}, skipping...")
        return

    # Clear the scene of existing objects before import
    bpy.ops.scene.new()

    print(f"Importing USD file: {input_usd_path}")
    # Import the USD file
    try:
        with SuppressBlenderOutput():
            bpy.ops.wm.usd_import(filepath=input_usd_path)
    except Exception as e:
        print(f"Error importing USD file: {e}")
        return

    original_face_count = 0
    decimated_face_count = 0
    num_mesh = len([x.type == 'MESH' for x in bpy.context.scene.objects])
    success = True
    print(f"Processing {num_mesh} meshes")
    
    # Iterate through all objects in the scene
    processed_meshes = set()

    # thresholds for decimation
    threshold1 = 300 # first decimation
    threshold2 = 1000 # second decimation
    threshold3 = 5000 # remesh
    threshold4 = 10000 # problem mesh
    threshold5 = 30000 # problem object
    angle_limit = 15

    # ratio for decimation
    if "person" in input_usd_path:
        print("Person mesh detected, using custom thresholds and ratio")
        threshold1 = 1000
        threshold2 = 30000//min(num_mesh, 3)
        threshold3 = 50000//min(num_mesh, 5)
        threshold4 = 30000
        threshold5 = 50000
        ratio = 0.2
        angle_limit = 15
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            original_face_count += len(obj.data.polygons)
    
    if original_face_count < 100:
        print(f"Original face count is too low: {original_face_count}, skipping...")
        # For "all", original_usd_path is `foo.usd`. input is `foo_renamed.usd`.
        # If we copy input to original, we get unique names in original.
        # Maybe that's acceptable if face count is low.
        shutil.copy(input_usd_path, original_usd_path) 
        
        with open(f"{os.path.dirname(output_usd_path)}/skipped.txt", "w") as f:
            f.write(f"Face count is too low: {original_face_count}, skipping...")
        return

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.name not in processed_meshes:
            if "component1_SRC_UV" in obj.name:
                continue
            if len(obj.data.polygons) > threshold1:
                print(f"Processing mesh object: {obj.name}, face count: {len(obj.data.polygons)}")
            processed_meshes.add(obj.name)
            # Count number of faces
            original_mesh_face_count = len(obj.data.polygons)
            # Select the object and make it active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)

            if len(obj.data.polygons) > threshold5 and not "person" in input_usd_path:
                ratio0 = threshold5/len(obj.data.polygons)
                apply_decimate_modifiers(obj, ratio0, modifier_name="Decimate_Faces0")
                print(f"  > Applied decimation: face count = {original_mesh_face_count} → {len(obj.data.polygons)}, ratio = {ratio0}")

            if len(obj.data.polygons) > threshold1 and not skip_uv_seams:
                print(f"  > Protecting UV seams for {obj.name}")
                protect_uv_seams(obj)

            if len(obj.data.polygons) > threshold1:
                num_polygons = len(obj.data.polygons)
                apply_decimate_modifiers(obj, ratio, modifier_name="Decimate_Faces1")
                apply_dissolve_modifiers(obj, angle_limit, modifier_name="Dissolve_Faces1")
                print(f"  > Applied first decimation: face count = {num_polygons} → {len(obj.data.polygons)}, ratio = {ratio}")

            if len(obj.data.polygons)>threshold2:
                num_polygons = len(obj.data.polygons)
                ratio2 = threshold2/len(obj.data.polygons)
                apply_decimate_modifiers(obj, ratio2, modifier_name="Decimate_Faces2")
                apply_dissolve_modifiers(obj, angle_limit, modifier_name="Dissolve_Faces2")
                print(f"  > Applied second decimation: face count = {num_polygons} → {len(obj.data.polygons)}, ratio = {ratio2}")

            if len(obj.data.polygons) > threshold3 and disable_remesh:
                print(f"  > Remesh disabled, keeping {len(obj.data.polygons)} faces for {obj.name}")

            if len(obj.data.polygons)>threshold3 and not disable_remesh:
                # try remesh
                voxel_size = 5.0
                adaptivity = 5.0
                print(f"  > Applying remesh, original face count: {len(obj.data.polygons)}, voxel_size: {voxel_size}")
                
                # Create a backup of the mesh data instead of relying on undo
                mesh_backup = obj.data.copy()
                
                apply_remesh_modifiers(obj, voxel_size, adaptivity, modifier_name="Remesh_Faces3")
                if len(obj.data.polygons)==0:
                    print(f"  > Remesh failed, original face count: {len(obj.data.polygons)}, skipping...")
                    # Restore from backup
                    failed_mesh = obj.data
                    obj.data = mesh_backup
                    # Remove the failed mesh data
                    bpy.data.meshes.remove(failed_mesh)
                    
                    success = False
                    continue
                else:
                    # Success, remove the backup
                    bpy.data.meshes.remove(mesh_backup)

                ratio3 = min(threshold3/len(obj.data.polygons), 0.5)
                print(f"  > Applying third decimation, original face count: {len(obj.data.polygons)}, ratio: {ratio3}")
                apply_decimate_modifiers(obj, ratio3, modifier_name="Decimate_Faces3")
                apply_dissolve_modifiers(obj, 10, modifier_name="Dissolve_Faces3")
                print("  > Decimated faces after remesh:", len(obj.data.polygons))

            if len(obj.data.polygons)>threshold4:
                with open("problem_mesh.log", "a") as f:
                    f.write(f"{obj.name}, {input_usd_path}, {len(obj.data.polygons)}\n")
                print(f"  > Problem mesh: {obj.name}, {input_usd_path}, {len(obj.data.polygons)}")

            if len(obj.data.polygons) > threshold1:
                print(f"  > Processed mesh object {obj.name}, face count: {original_mesh_face_count} → {len(obj.data.polygons)}")

            decimated_face_count += len(obj.data.polygons)

            # Deselect the object
            obj.select_set(False)
    print(f"Object total faces: {original_face_count} → {decimated_face_count}")
    if decimated_face_count > threshold5 or not success:
        success = False
        with open("problem_objects.log", "a") as f:
            f.write(f"{input_usd_path}, {decimated_face_count}\n")
            print(f"Problem object: {input_usd_path}, {decimated_face_count}")
    
        # save blend file
        output_blend_path = os.path.splitext(output_usd_path)[0] + ".blend"
        with SuppressBlenderOutput():
            bpy.ops.wm.save_mainfile(filepath=output_blend_path)
        print(f"Saved blend file: {output_blend_path}")

    # Export decimated geometry
    export_usd(output_usd_path)

    if Usd is None:
        print("[INFO] pxr is unavailable in Blender Python; skipping in-process geometry swap.")
        info = {
            "original_face_count": original_face_count,
            "decimated_face_count": decimated_face_count,
            "decimation_success": success,
            "swap_deferred": True,
        }
        with open(info_path, "w") as f:
            json.dump(info, f)
        return
    
    # merge decimated geometry into original hierarchy
    print(f"Merging decimated geometry into original hierarchy...")
    swap_mesh_geometry(input_usd_path, output_usd_path, original_usd_path)

    # Verify face count
    bpy.ops.scene.new()
    faces = count_total_faces_in_usd(original_usd_path)
    if faces == 0:
        success = False
        print(f"Error: Resulting USD has 0 triangles")
        if os.path.exists(output_usd_path):
            os.remove(output_usd_path)
    else:
        print(f"Resulting USD has {faces} triangles.")
        # save a "optimized.txt" file in that folder
        with open(f"{os.path.dirname(output_usd_path)}/optimized.txt", "w") as f:
            f.write(f"Total triangles in optimized meshes: {faces}\n")

        info = {
            "original_face_count": original_face_count,
            "decimated_face_count": decimated_face_count,
            "decimation_success": success,
        }

        # save to info.json
        with open(info_path, "w") as f:
            json.dump(info, f)

def count_total_faces_in_usd(usd_path):
    """Count total faces in a USD file."""
    bpy.ops.scene.new()
    if not os.path.exists(usd_path):
        return None
    with SuppressBlenderOutput():
        bpy.ops.wm.usd_import(filepath=usd_path)
    total_faces = 0
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            total_faces += len(obj.data.polygons)
    return total_faces

def decimate_all_usd_in_folder(
    input_folder,
    ratio=0.1,
    worker_id=0,
    total_workers=1,
    source="gr",
    skip_uv_seams=False,
    disable_remesh=False,
):
    """Recursively decimate all USD files in a folder."""
    all_files = []
    
    if source == "gr":
        for root, dirs, files in os.walk(input_folder):
            if "instance_renamed.usd" in files:
                all_files.append((root, files))
    elif source == "all":
        for root, dirs, files in os.walk(input_folder):
            renamed_files = [f for f in files if f.endswith("_renamed.usd")]
            if renamed_files:
                all_files.append((root, files))

    usd_files = []
    random.shuffle(all_files)
    pbar = tqdm(all_files, desc="Checking status", maxinterval=0.5)
    
    for root, files in pbar:
        # Determine targets in this folder
        targets = []
        if source == "gr":
            if "instance_renamed.usd" in files:
                targets.append("instance_renamed.usd")
        elif source == "all":
            targets = [f for f in files if f.endswith("_renamed.usd")]
            
        for target_file in targets:
            base_name = target_file.replace("_renamed.usd", "")
            # info_file logic
            if source == "gr":
                info_filename = "info.json"
                output_filename = "instance_renamed_decimated.usd"
            else:
                info_filename = f"{base_name}_info.json"
                output_filename = f"{base_name}_decimated.usd"
                
            decimation_success = None
            if info_filename in files:
                try:
                    with open(os.path.join(root, info_filename), "r") as f:
                        info = json.load(f)
                        decimation_success = info.get("decimated_face_count", 0) > 0 and (output_filename in files) or ("skipped.txt" in files)
                        # decimation_success = info.get("decimated_face_count", 0) > 0 and (output_filename in files) or ("skipped.txt" in files)
                except Exception as e:
                    print(f"Error loading {os.path.join(root, info_filename)}: {e}")
            
            if decimation_success is None:
                decimated_face_count = count_total_faces_in_usd(os.path.join(root, output_filename))
                decimation_success = (decimated_face_count is not None) and (decimated_face_count < 30000) or ("skipped.txt" in files)
                decimated_face_count = 0 if decimated_face_count is None else decimated_face_count
                info = {
                    "decimated_face_count": decimated_face_count,
                    "decimation_success": decimation_success,
                }
                # Update info file just in case
                with open(os.path.join(root, info_filename), "w") as f:
                    json.dump(info, f)
            
            if not decimation_success:
                usd_files.append(os.path.join(root, target_file))
                
    usd_files.sort()
    # Partition files among workers
    # if len(usd_files) < 500:
    #     my_files = usd_files
    # else:
    my_files = [f for i, f in enumerate(usd_files) if i % total_workers == worker_id]
    # shuffle the files
    random.shuffle(my_files)

    print(f"TERMINAL: Worker {worker_id}/{total_workers-1} processing {len(my_files)}/{len(usd_files)} USD files in folder: {input_folder}")

    for idx, usd_path in tqdm(enumerate(my_files)):
        # usd_path is like ".../foo_renamed.usd"
        base_name_ext = os.path.basename(usd_path) # foo_renamed.usd
        base_name = base_name_ext.replace("_renamed.usd", "") # foo
        
        dir_name = os.path.dirname(usd_path)
        
        if source == "gr":
            info_filename = "info.json"
            output_filename = "instance_renamed_decimated.usd"
        else:
            info_filename = f"{base_name}_info.json"
            output_filename = f"{base_name}_decimated.usd"
        output_path = os.path.join(dir_name, output_filename)
        if source == "gr":
             # Original logic for GR: instance_renamed.usd -> instance.usd
             original_usd_path = os.path.join(dir_name, "instance.usd")
             info_path = os.path.join(dir_name, "info.json")
        else:
             original_usd_path = os.path.join(dir_name, f"{base_name}.usd")
             info_path = os.path.join(dir_name, f"{base_name}_info.json")

        print(f"[INFO] Worker {worker_id} ({idx}/{len(my_files)}) Decimating: {usd_path} → {output_path}")
        try:
            decimate_usd_meshes(
                usd_path,
                output_path,
                original_usd_path,
                info_path,
                ratio,
                skip_uv_seams=skip_uv_seams,
                disable_remesh=disable_remesh,
            )
        except Exception as e:
            print(f"[ERROR] Worker {worker_id} Failed to process {usd_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    # Parse command line arguments
    # Note: When running with Blender, arguments after '--' are passed to the script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # Get arguments after '--'
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Recursively decimate all USD files in a folder using Blender")
    parser.add_argument("--input_folder", 
                        type=str, 
                        required=True,
                        help="Path to the input folder containing USD files to decimate")
    parser.add_argument("--ratio", 
                        type=float, 
                        default=0.1,
                        help="Decimation ratio (default: 0.1)")
    parser.add_argument("--worker_id",
                        type=int,
                        default=0,
                        help="ID of the current worker (0-indexed)")
    parser.add_argument("--total_workers",
                        type=int,
                        default=1,
                        help="Total number of workers")
    parser.add_argument("--source",
                        type=str,
                        default="gr",
                        choices=["gr", "all"],
                        help="Source type: 'gr' or 'all'")
    parser.add_argument("--skip_uv_seams",
                        action="store_true",
                        help="Skip UV seam protection for crash-prone meshes")
    parser.add_argument("--disable_remesh",
                        action="store_true",
                        help="Skip remesh modifier for crash-prone meshes")

    args = parser.parse_args(argv)
    
    # Use parsed arguments
    input_folder = args.input_folder
    decimation_ratio = args.ratio
    worker_id = args.worker_id
    total_workers = args.total_workers
    source = args.source
    skip_uv_seams = args.skip_uv_seams
    disable_remesh = args.disable_remesh
    
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Decimation ratio: {decimation_ratio}")
    print(f"[INFO] Worker ID: {worker_id}, Total Workers: {total_workers}")
    print(f"[INFO] Source: {source}")
    print(f"[INFO] Skip UV seams: {skip_uv_seams}")
    print(f"[INFO] Disable remesh: {disable_remesh}")
    
    # Main function call
    decimate_all_usd_in_folder(
        input_folder,
        decimation_ratio,
        worker_id,
        total_workers,
        source,
        skip_uv_seams=skip_uv_seams,
        disable_remesh=disable_remesh,
    )
