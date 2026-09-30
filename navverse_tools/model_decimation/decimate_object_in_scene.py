# Usage: blender --background --quiet --python decimate_object_in_scene.py
import os
import shutil
import sys
import re
from pxr import Usd, UsdGeom, Sdf
import bpy
from mathutils import Vector
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils.usd_utils import decimate_usd
from utils.mesh_utils import swap_mesh_geometry

# ----------------------------------------------------------
#  Output Logger
# ----------------------------------------------------------
log_file_path = "my_script_modified.log"
os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
log_file = open(log_file_path, "w")
sys.stdout = log_file


def decimate_object_in_scene(stage, dataset_path):
    
    ## Iterate through all objects in the scene
    prim_paths = [prim.GetPath() for prim in stage.Traverse()]
    for prim_path in prim_paths:
        print(prim_path)
        prim = stage.GetPrimAtPath(prim_path)

        # Only match leaf Instance prims
        if not prim_path or not str(prim_path).endswith("/Instance"):
            continue

        # Get a list of all prepended references
        prim = prim.GetParent()
        references = []
        for prim_spec in prim.GetPrimStack():
            references.extend(prim_spec.referenceList.prependedItems)

        object_file = os.path.join(dataset_path, references[0].assetPath)
        print("object file:", object_file)
        object_folder = Path(object_file).parent
        print("object folder:", object_folder)

        ## decimate the object USD file
        renamed_usd = os.path.join(str(object_folder), "instance_renamed.usd")
        decimated_usd = os.path.join(str(object_folder), "instance_renamed_decimated_001.usd")
        if os.path.exists(decimated_usd):
            continue
        decimation_ratio = 0.001  # 0.1% of original faces
        decimate_usd(renamed_usd, decimated_usd, decimation_ratio)

        ## swap the decimated mesh geometry back to the original scene USD
        output_usd = os.path.join(str(object_folder), 'instance.usd')
        swap_mesh_geometry(renamed_usd, decimated_usd, output_usd)
    

# ----------------------------------------------------------
#  MAIN ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.environ.get(
        "GRSCENES_COMMERCIAL_ROOT",
        os.path.join(_REPO_ROOT, "navverse_data", "grscenes_commercial_decimated"),
    )
    usd_file = os.path.join(dataset_path, "scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation.usd")    # Hospital scene

    if not os.path.exists(usd_file):
        print("ERROR: USD file not found")
        exit(1)

    stage = Usd.Stage.Open(usd_file)

    # Decimate object USD in scene
    decimate_object_in_scene(stage, dataset_path)

