import bpy
import os
import mathutils

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
path = os.environ.get(
    "VC_STORE_ROAD_OBJECTS",
    os.path.join(_REPO_ROOT, "navverse_data", "vc_store", "road_objects"),
)
assets = ['nyc_traffic_light.glb', 'Danger_road_sign.glb', 'Only_straight_road_sign.glb', 'Bus_stop_sign.glb', 'Speed_50_road_sign.glb']

for a in assets:
    p = os.path.join(path, a)
    if os.path.exists(p):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=p)
        min_v = [float('inf')] * 3
        max_v = [float('-inf')] * 3
        has_mesh = False
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                has_mesh = True
                mw = obj.matrix_world
                for c in obj.bound_box:
                    wc = mw @ mathutils.Vector(c)
                    for i in range(3):
                        min_v[i] = min(min_v[i], wc[i])
                        max_v[i] = max(max_v[i], wc[i])
        if has_mesh:
            print(f"ASSET: {a} HEIGHT: {max_v[2] - min_v[2]}")
        else:
            print(f"ASSET: {a} NO MESH FOUND")
