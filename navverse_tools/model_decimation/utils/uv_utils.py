import bpy
import bmesh
from mathutils import Vector
import numpy as np


EPS = 1e-6

def uv_equal(a: Vector, b: Vector, eps=EPS) -> bool:
    return (a - b).length <= eps

def mark_seams_from_uv_islands(obj: bpy.types.Object, uv_name: str | None = None, eps=EPS) -> int:
    """
    Mark mesh edges as seam if UVs are discontinuous across that edge.
    Returns number of edges marked as seam.
    """
    me = obj.data
    if not me.uv_layers:
        # print(f"[WARN] {obj.name}: no UV layers, skipping seam-from-islands.")
        return 0

    # pick UV layer
    if uv_name is None:
        uv_name = me.uv_layers.active.name
    if uv_name not in me.uv_layers:
        raise ValueError(f"{obj.name}: UV layer '{uv_name}' not found. Available: {[u.name for u in me.uv_layers]}")

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    uv_layer = bm.loops.layers.uv.get(uv_name)
    if uv_layer is None:
        bm.free()
        raise RuntimeError(f"{obj.name}: bmesh has no uv layer '{uv_name}' (unexpected).")

    marked = 0

    # optional: clear existing seams first (comment out if you want to keep existing seams too)
    for e in bm.edges:
        e.seam = False

    for e in bm.edges:
        # only consider manifold edges with exactly 2 linked loops (2 faces)
        loops = e.link_loops
        if len(loops) != 2:
            continue

        l1, l2 = loops[0], loops[1]

        # UV endpoints of the edge in each face loop orientation
        u1a = l1[uv_layer].uv.copy()
        u1b = l1.link_loop_next[uv_layer].uv.copy()

        u2a = l2[uv_layer].uv.copy()
        u2b = l2.link_loop_next[uv_layer].uv.copy()

        # If UVs match (same or flipped), it's continuous => same island across this edge.
        continuous = (uv_equal(u1a, u2a, eps) and uv_equal(u1b, u2b, eps)) or \
                     (uv_equal(u1a, u2b, eps) and uv_equal(u1b, u2a, eps))

        if not continuous:
            e.seam = True
            marked += 1

    bm.to_mesh(me)
    bm.free()
    me.update()
    return marked

def build_protect_seams_vertex_group(obj: bpy.types.Object, vg_name="protect_seams",
                                     seam_weight=0.0, other_weight=1.0) -> bpy.types.VertexGroup:
    """
    Creates/overwrites a vertex group where seam vertices have weight seam_weight,
    all others have other_weight.
    """
    me = obj.data

    # collect seam vertices from mesh edges
    seam_verts = set()
    for e in me.edges:
        if e.use_seam:
            seam_verts.add(e.vertices[0])
            seam_verts.add(e.vertices[1])

    vg = obj.vertex_groups.get(vg_name) or obj.vertex_groups.new(name=vg_name)

    # clear existing weights in this group by recreating it (safest)
    # Blender doesn't have a direct "clear group" API, so we remove & recreate.
    obj.vertex_groups.remove(vg)
    vg = obj.vertex_groups.new(name=vg_name)

    all_vids = [v.index for v in me.vertices]
    if all_vids:
        vg.add(all_vids, other_weight, 'REPLACE')
    if seam_verts:
        vg.add(list(seam_verts), seam_weight, 'REPLACE')

    # print(f"[INFO] {obj.name}: vg '{vg_name}' created. seam verts={len(seam_verts)} / total verts={len(me.vertices)}")
    return vg

def assign_vg_to_collapse_decimate(obj: bpy.types.Object, vg_name: str, factor: float = 1.0):
    """
    Assign the vertex group to all Decimate modifiers in COLLAPSE mode.
    """
    found = 0
    for m in obj.modifiers:
        if m.type == 'DECIMATE' and getattr(m, "decimate_type", None) == 'COLLAPSE':
            m.vertex_group = vg_name
            # property name is 'vertex_group_factor' in Blender
            if hasattr(m, "vertex_group_factor"):
                m.vertex_group_factor = factor
            found += 1
    print(f"[INFO] {obj.name}: assigned vg '{vg_name}' to {found} COLLAPSE decimate modifier(s).")
    return found

def protect_uv_seams(obj: bpy.types.Object, uv_name=None, vg_name="protect_seams", eps=EPS):
    if obj.type != "MESH":
        return
    marked = mark_seams_from_uv_islands(obj, uv_name=uv_name, eps=eps)
    # print(f"{obj.name}: seams marked from UV islands = {marked}")
    build_protect_seams_vertex_group(obj, vg_name=vg_name, seam_weight=0.0, other_weight=1.0)

def apply_decimate_modifiers(obj, ratio, modifier_name="Decimate_Faces", apply=True, vg_name="protect_seams"):
    decimate_modifier = obj.modifiers.new(name=modifier_name, type='DECIMATE')
    decimate_modifier.ratio = ratio
    decimate_modifier.use_collapse_triangulate = True
    decimate_modifier.vertex_group = vg_name
    decimate_modifier.vertex_group_factor = 1.0
    if apply:
        bpy.ops.object.modifier_apply(modifier=modifier_name)

def apply_dissolve_modifiers(obj, angle_limit, modifier_name="Dissolve_Faces", apply=True):
    dissolve_modifier = obj.modifiers.new(name=modifier_name, type='DECIMATE')
    dissolve_modifier.decimate_type = "DISSOLVE"
    dissolve_modifier.use_dissolve_boundaries = False
    dissolve_modifier.angle_limit = np.deg2rad(angle_limit)
    dissolve_modifier.delimit = set(["UV", "SEAM", "MATERIAL"])
    if apply:
        bpy.ops.object.modifier_apply(modifier=modifier_name)

def apply_remesh_modifiers(obj, voxel_size, adaptivity, modifier_name="Remesh_Faces", apply=True, transfer_uv=True):
    transfer_uv = obj.data.uv_layers.active is not None
    if transfer_uv:
        # Duplicate original as the UV source (keeps original intact)
        src = obj.copy()
        src.data = obj.data.copy()
        src.name = obj.name + "_SRC_UV"
        bpy.context.scene.collection.objects.link(src)
    remesh_modifier = obj.modifiers.new(name=modifier_name, type='REMESH')
    remesh_modifier.voxel_size = voxel_size
    remesh_modifier.adaptivity = adaptivity
    if apply:
        bpy.ops.object.modifier_apply(modifier=modifier_name)
    if transfer_uv:
        dt = obj.modifiers.new(name="UV_Transfer", type="DATA_TRANSFER")
        dt.object = src
        dt.use_loop_data = True
        dt.data_types_loops = {"UV"}
        # Use active UV layers
        src_uv = src.data.uv_layers.active.name
        dst_uv = src_uv
        if not obj.data.uv_layers.active:
            obj.data.uv_layers.new(name=src_uv)
        else:
            dst_uv = obj.data.uv_layers.active.name
        dt.layers_uv_select_src = src_uv
        dt.layers_uv_select_dst = dst_uv

        # Robust mapping for changed topology
        # (best general choice for UVs when topology differs)
        dt.loop_mapping = "POLYINTERP_NEAREST"
        dt.mix_mode = "REPLACE"
        dt.mix_factor = 1.0

        bpy.ops.object.modifier_apply(modifier=dt.name)
        bpy.data.objects.remove(src, do_unlink=True)
