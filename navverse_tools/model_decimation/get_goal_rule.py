# Usage: blender --background --quiet --python get_goal_rule.py 
import os
import shutil
import sys
import re
from pxr import Usd, UsdGeom, Sdf
import bpy
from mathutils import Vector
from pathlib import Path

# ----------------------------------------------------------
#  Output Logger
# ----------------------------------------------------------
log_file_path = "my_script_modified.log"
os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
log_file = open(log_file_path, "w")
sys.stdout = log_file

# ----------------------------------------------------------
#  PART 1: USD FILE PRIM CATEGORY EXTRACTION
# ----------------------------------------------------------
def extract_category_from_path(prim_path):
    parts = prim_path.strip("/").split("/")

    if "Instance" not in parts:
        return None

    idx = parts.index("Instance")
    if idx - 2 < 0:
        return None

    return parts[idx - 2], parts[idx - 1]


def build_rule(category, model_name):
    if category == "other":
        return f"other/{model_name}"
    else:
        return f"{category}/model_[^/]*/Instance$"

def get_all_prims(prim):
    """
    Recursively yield a prim and all its children, including those from references.
    """
    yield prim
    for child in prim.GetChildren():  # stage prims only
        yield from get_all_prims(child)


def write_category_rules(stage, output_txt, dataset_path):
    
    category_rules = {}
    other_models = set()
    prim_paths = [prim.GetPath() for prim in stage.Traverse()]
    for prim_path in prim_paths:
        print(prim_path)
        prim = stage.GetPrimAtPath(prim_path)

        # Only match leaf Instance prims
        if not prim_path or not str(prim_path).endswith("/Instance"):
            continue
            
        # Extract category and model name
        extracted = extract_category_from_path(str(prim_path))
        if not extracted:
            continue
        category, model_name = extracted

        # Unknown category handling
        if category == "other":
            
            rule = build_rule(category, model_name)
            other_models.add(rule)
            # Get a list of all prepended references
            prim = prim.GetParent()
            references = []
            for prim_spec in prim.GetPrimStack():
                references.extend(prim_spec.referenceList.prependedItems)

            object_file = os.path.join(dataset_path, references[0].assetPath)
            object_folder = Path(object_file).parent

            ## TODO: Render object usd file to image
            #     img_out = os.path.join("renders", f"{folder_name}.png")
            #     render_usd_image(object_file, img_out)

            ## TODO: Use Dino for detection
            detected_category = "detected_category"  # Placeholder for detected category

            ## Copy object folder to detected category folder
            detected_path = os.path.join(dataset_path, "models/object/others", detected_category)
            os.makedirs(detected_path, exist_ok=True)
            dest_folder = os.path.join(detected_path, os.path.basename(object_folder))
            if not os.path.exists(dest_folder):
                shutil.copytree(object_folder, dest_folder)

            ## Rename prim
            root_layer = stage.GetRootLayer()
            old_prim_path = prim.GetPath()
            print("old prim path:", old_prim_path)

            base_path = old_prim_path.GetParentPath().GetParentPath()  
            old_prim_name = old_prim_path.name        

            new_parent_path = base_path.AppendChild(detected_category)
            if not root_layer.GetPrimAtPath(new_parent_path):
                # Define an empty prim at that path
                stage.DefinePrim(new_parent_path)

            # New prim path
            new_prim_path = new_parent_path.AppendChild(old_prim_name)
            print("new prim path:", new_prim_path)

            # Copy old prim spec to new path
            Sdf.CopySpec(root_layer, old_prim_path, root_layer, new_prim_path)

            # Remove the old prim spec
            parent = root_layer.GetPrimAtPath(old_prim_path.GetParentPath())
            del parent.nameChildren[old_prim_path.name]

            ## Add new reference to new prim
            prim = stage.GetPrimAtPath(new_prim_path)
            new_ref_path = os.path.join(dest_folder, "instance.usd")
            rel_new_ref_path = os.path.relpath(new_ref_path, os.path.dirname(root_layer.realPath))
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(rel_new_ref_path)

        # Known category handling
        else:
            if category not in category_rules:
                category_rules[category] = build_rule(category, model_name)

    with open(output_txt, "w") as f:
        for cat, rule in sorted(category_rules.items()):
            f.write(f"{cat}: {rule}\n")

        for rule in sorted(other_models):
            f.write(f"other: {rule}\n")

    print(f"[DONE] Category rules written to {output_txt}")

    ## Export modified USD file (optional)
    # flattened = stage.Flatten() 
    # exported_usd_path = os.path.join(dataset_path, "scenes/modified_scene.usd")
    # flattened.Export(exported_usd_path)

# ----------------------------------------------------------
#  PART 2: TODO RENDER THE USD FILE IN BLENDER
# ----------------------------------------------------------
def render_usd_image(usd_file, output_image):
    pass


# ----------------------------------------------------------
#  MAIN ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.environ.get("GRSCENES_COMMERCIAL_ROOT",
                                  os.path.join(_REPO_ROOT, "navverse_data", "grscenes_commercial"))
    usd_file = os.path.join(dataset_path, "scenes/MV4AFHQKTKJZ2AABAAAAADQ8_usd/start_result_navigation.usd")
    # usd_file = os.path.join(dataset_path, "scenes/modified_scene.usd")

    txt_out = "prim_rules.txt"
    img_out = "renders/render.png"

    if not os.path.exists(usd_file):
        print("ERROR: USD file not found")
        exit(1)

    stage = Usd.Stage.Open(usd_file)

    # Parse USD
    write_category_rules(stage, txt_out, dataset_path)

