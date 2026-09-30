import os
from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf

def count_total_faces_in_mesh_prim(prim):
    total_faces = 0
    mesh = UsdGeom.Mesh(prim)
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    if face_counts is not None:
        total_faces += len(face_counts)
    return total_faces

def get_parent_prim_name(prim):
    """Get the parent prim name."""
    parent_path = prim.GetPath().GetParentPath()
    if parent_path == Sdf.Path.absoluteRootPath:
        return None  # No parent (root prim)
    
    parent_prim = prim.GetStage().GetPrimAtPath(parent_path)
    return parent_prim.GetName() if parent_prim else None

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
    for prim in stage_orig.Traverse():
        
        total_faces_original += count_total_faces_in_mesh_prim(prim)

        mesh_name = prim.GetName()
        if mesh_name in dec_mesh_map:
            dec_prim = dec_mesh_map[mesh_name]
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

    print(f"[INFO] Total faces in original meshes: {total_faces_original}")
    print(f"[INFO] Total faces in swapped meshes: {total_faces_swapped}")

    # Save the modified original stage as the output
    stage_orig.GetRootLayer().Export(output_usd_path)
    print(f"✅ Exported USD with original hierarchy/materials and swapped mesh geometry: {output_usd_path}")