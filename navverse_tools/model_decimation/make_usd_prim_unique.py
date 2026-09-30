# Usage: python make_usd_prim_unique.py --input_folder /path/to/folder

## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu

import os
import random
import string
import sys
import argparse
from pxr import Usd, Sdf
# import omni.kit.commands
# import omni.kit
import tqdm

def make_unique_name(name, existing):
    """
    If 'name' exists in existing set, append a suffix to make it unique.
    """
    if name not in existing:
        existing.add(name)
        return name
    i = 1
    while f"{name}_{i}" in existing:
        i += 1
    # 5 random letters
    random_suffix = ''.join(random.choices(string.ascii_letters, k=5))
    unique_name = f"{name}_{i}_{random_suffix}"
    existing.add(unique_name)
    return unique_name

def limit_name_length(name, max_length=60):
    """
    Limit name length to max_length characters.
    If name is longer, truncate and add hash suffix to maintain uniqueness.
    """
    if len(name) <= max_length:
        return name, False
    return name[:max_length], True

def rename_prims(stage):
    """
    Recursively rename all prims in the stage to ensure unique names.
    Uses omni.kit.commands.MovePrim to safely handle references/payloads.
    """
    existing_names = set()
    existing_names.add("Looks")
    existing_names.add("Materials")
    prim_paths = [prim.GetPath() for prim in stage.Traverse() if '/Root/Looks' not in str(prim.GetPath()) and '/Root/Materials' not in str(prim.GetPath()) and 'Looks/WorldGridMaterial' not in str(prim.GetPath())]
    
    prim_paths.reverse()
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        # omni.kit.commands.execute("MovePrim", path_from=path, path_to=pathTo)
        if not prim.IsValid():
            # print(f"Invalid prim at path: {path}")
            # print("prim name:", prim.GetName())
            continue

        old_name = prim.GetName()
        old_name, name_pruned = limit_name_length(old_name)
        # print("original path:", prim.GetPath())
        new_name = make_unique_name(old_name, existing_names)

        if old_name != new_name or name_pruned:
            parent_path = prim.GetPath().GetParentPath()
            new_path = parent_path.AppendChild(new_name)
            # print("new path:", new_path)
            # Copy Mesh spec to new path
            Sdf.CopySpec(stage.GetRootLayer(), prim.GetPath(), stage.GetRootLayer(), new_path)

            # Remove old Mesh
            stage.RemovePrim(prim.GetPath())

            # print(f"Renamed Mesh: {prim.GetPath()} -> {new_path}")


def process_usd_file(usd_path, output_usd_path):
    """
    Open a USD file in Isaac Lab, rename all prims uniquely, and save to a new USD file.
    """
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as e:
        print(f"Failed to open USD stage: {usd_path}: {e}")
        return

    if not stage:
        print(f"Failed to open USD stage: {usd_path}")
        return

    rename_prims(stage)

    # Save to a new USD file
    stage.GetRootLayer().Export(output_usd_path)
    # print(f"Saved renamed USD: {output_usd_path}")

def make_name_unique_all_usds(input_folder, source="gr"):
    """
    Iterate through all subfolders in input_folder,
    find 'instance.usd' files, rename prims uniquely, and save as 'instance_renamed.usd'.
    """
    all_usd_paths = []
    
    print(f"Scanning folder: {input_folder} with source={source}")
    
    for root_dir, dirs, files in os.walk(input_folder):
        if source == "gr":
            if "instance_renamed.usd" in files:
                continue
            if "instance.usd" in files:
                all_usd_paths.append((os.path.join(root_dir, "instance.usd"), os.path.join(root_dir, "instance_renamed.usd")))
        elif source == "all":
            for f in files:
                if f.endswith(".usd") and not f.endswith("_renamed.usd") and not f.endswith("_decimated.usd"):
                    base_name = os.path.splitext(f)[0]
                    renamed_name = f"{base_name}_renamed.usd"
                    if renamed_name in files:
                        continue
                    all_usd_paths.append((os.path.join(root_dir, f), os.path.join(root_dir, renamed_name)))

    all_usd_paths.sort()
    print(f"Found {len(all_usd_paths)} files to process.")
    
    pbar = tqdm.tqdm(all_usd_paths)
    for usd_path, output_path in pbar:
        process_usd_file(usd_path, output_path)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Make USD prim names unique by adding suffixes and limiting length")
    parser.add_argument("--input_folder", 
                        type=str, 
                        required=True,
                        help="Path to the input folder containing USD files to process")
    parser.add_argument("--source",
                        type=str,
                        default="gr",
                        choices=["gr", "all"],
                        help="Source type: 'gr' for instance.usd, 'all' for all .usd files")
    
    args = parser.parse_args()
    
    # Use parsed arguments
    input_folder = args.input_folder
    source = args.source
    
    print(f"Processing USD files in folder: {input_folder}")
    
    # Validate input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    # Main function call
    make_name_unique_all_usds(input_folder, source)
