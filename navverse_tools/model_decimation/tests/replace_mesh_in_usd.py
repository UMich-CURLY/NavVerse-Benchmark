# Usage: python tests/replace_mesh_in_usd.py

## Author: Po-Hsun Chang
## Contact: pohsun@umich.edu

import os
import sys
from pathlib import Path

from pxr import Usd, UsdGeom
# Ensure the package root (model_decimation) is on sys.path so `utils` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.mesh_utils import count_total_faces_in_mesh_prim, get_parent_prim_name

def swap_mesh_geometry(original_usd_path, decimated_usd_path, output_usd_path):

    stage_orig = Usd.Stage.Open(original_usd_path)
    stage_dec = Usd.Stage.Open(decimated_usd_path)

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
            print("decimated prim path:", prim.GetPath())
    print("hash:", dec_mesh_map.keys())
    total_faces_original = 0
    total_faces_swapped = 0
    # For each mesh in the original, if a decimated mesh with the same name exists, swap geometry
    # original: /Root/Instance/Group_accac44c_2749_43ad_9490_24084eefd0bc/SM_04/MUI3NKIKTJS66AABAAAAAAY8_primitiveModel_LoftModel_mesh_07c8d02b__DD__31e1__DD__508a__DD__9d9b__DD__0d8d4af6ce26_roomId_null_0
    # /root/Root/Instance/Group_accac44c_2749_43ad_9490_24084eefd0bc/SM_04/MUI3NKIKTJS66AABAAAAAAY8_primitiveModel_LoftModel_mesh_07c8d02b/MUI3NKIKTJS66AABAAAAAAY8_primitiveModel_LoftModel_mesh_07c8d02b
    for prim in stage_orig.Traverse():
        # if prim.IsA(UsdGeom.Mesh):
        # print("original prim faces:", count_total_faces_in_mesh_prim(prim))
        total_faces_original += count_total_faces_in_mesh_prim(prim)
        mesh_name = prim.GetName()
        print("original prim path:", prim.GetPath())
        print("original prim name:", prim.GetName())
        if mesh_name in dec_mesh_map:
            dec_prim = dec_mesh_map[mesh_name]
            print("decimated prim path:", dec_prim.GetPath())
            # Copy geometry attributes
            for attr_name in [
                "points", "normals", "faceVertexIndices", "faceVertexCounts",
                "primvars:normals", "primvars:st", "primvars:uv"
            ]:
                orig_attr = prim.GetAttribute(attr_name)
                dec_attr = dec_prim.GetAttribute(attr_name)
                if orig_attr and dec_attr and dec_attr.HasAuthoredValue():
                    orig_attr.Set(dec_attr.Get())
        total_faces_swapped += count_total_faces_in_mesh_prim(prim)
        print("decimated prim faces:", count_total_faces_in_mesh_prim(prim))

    print(f"[INFO] Total faces in original meshes: {total_faces_original}")
    print(f"[INFO] Total faces in swapped meshes: {total_faces_swapped}")
    # Save the modified original stage as the output
    # stage_orig.GetRootLayer().Export(output_usd_path)
    # print(f"✅ Exported USD with original hierarchy/materials and swapped mesh geometry: {output_usd_path}")


def count_total_faces_in_instance(stage):
    total_faces = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith("/Root/Instance"):
            mesh = UsdGeom.Mesh(prim)
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            if face_counts is not None:
                total_faces += len(face_counts)
    return total_faces


# --- Example usage ---
if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parents[3]
    model_folder = Path("navverse_data/grscenes_commercial_decimated/models/object/articulated/refrigerator/07e0ddb233eee0d2e3eae00b7e28afa8/")

    original_usd = str(repo_root / model_folder / "instance_renamed.usd")
    decimated_usd = str(repo_root / model_folder / "instance_renamed_decimated.usd")
    output_usd = str(repo_root / model_folder / "instance_test.usd")

    # Print face count in original
    stage_orig = Usd.Stage.Open(original_usd)
    orig_faces = count_total_faces_in_instance(stage_orig)
    print(f"Original USD total face count under /Root/Instance: {orig_faces}")

    swap_mesh_geometry(original_usd, decimated_usd, output_usd)

    # Print face count in swapped output
    stage_out = Usd.Stage.Open(output_usd)
    out_faces = count_total_faces_in_instance(stage_out)
    print(f"Output USD total face count under /Root/Instance: {out_faces}")
