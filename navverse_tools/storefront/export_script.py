import bpy
import sys
import math
import os

print("PYTHON SCRIPT STARTED", flush=True)

# Get args
# Blender args are after "--", so we find the index of "--" and take subsequent args
print("DEBUG: Checking arguments...", flush=True)
try:
    argv = sys.argv
    # print(f"DEBUG: argv={argv}", flush=True) 
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
        output_usd = argv[0]
        output_glb = argv[1]
        print(f"DEBUG: Output USD: {output_usd}", flush=True)
    else:
        raise ValueError("Missing '--' separator for arguments")
except Exception as e:
    print(f"Error parsing arguments: {e}", flush=True)
    # Force blender to exit with error code if possible, or just re-raise
    import sys
    sys.exit(1)

print("DEBUG: Deselecting...", flush=True)
# 1. Deselect everything
try:
    bpy.ops.object.select_all(action='DESELECT')
except Exception as e:
    print(f"Error deselecting: {e}", flush=True)

count = len(bpy.data.objects)
print(f"DEBUG: Found {count} objects in the scene", flush=True)

total_faces = 0

# 2. Iterate through all MESH objects
print("DEBUG: Starting loop...", flush=True)
for obj in bpy.data.objects:
    print(f"Processing object: {obj.name} with type: {obj.type}", flush=True)
    if obj.type == 'MESH':
        print(f"Processing object: {obj.name}", flush=True)
        
        # Set active and select
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # Apply Modifiers
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
            # Iterate over a copy of the modifiers list to avoid issues as they are removed
            for modifier in list(obj.modifiers):
                print(f"  Applying modifier: {modifier.name}", flush=True)
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            print(f"  Modifiers applied: {obj.modifiers}", flush=True)
            faces = len(obj.data.polygons)
            print(f"  Faces after modifiers: {faces}", flush=True)
            total_faces += faces
        except Exception as e:
            print(f"  Error applying modifiers on {obj.name}: {e}", flush=True)

        # Cleanup: Limited Dissolve
        # We need to be in EDIT mode for mesh operations
        # try:
        #     bpy.ops.object.mode_set(mode='EDIT')
            
        #     # Select all geometry
        #     bpy.ops.mesh.select_all(action='SELECT')
            
        #     # Limited Dissolve
        #     # 20 degrees = 0.349066 radians
        #     bpy.ops.mesh.dissolve_limited(angle_limit=0.349066)
            
        #     # Return to Object mode
        #     bpy.ops.object.mode_set(mode='OBJECT')
        #     print(f"  Faces after cleanup: {len(obj.data.polygons)}", flush=True)
            
        # except Exception as e:
        #     print(f"  Error during cleanup of {obj.name}: {e}", flush=True)
        #     # Ensure we attempt to return to object mode if failed
        #     if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        #         bpy.ops.object.mode_set(mode='OBJECT')
            
        # Deselect for next iteration
        obj.select_set(False)

# 3. Export
print(f"Exporting USD to {output_usd}", flush=True)
bpy.ops.wm.usd_export(filepath=output_usd, check_existing=False)

print(f"Exporting GLB to {output_glb}", flush=True)
bpy.ops.export_scene.gltf(filepath=output_glb, check_existing=False, export_format='GLB')

# 4. Save the blend file
print("Saving blend file as model.blend...", flush=True)
# Derive model.blend path from output_usd path
output_dir = os.path.dirname(output_usd)
model_blend_path = os.path.join(output_dir, "model.blend")
bpy.ops.wm.save_mainfile(filepath=model_blend_path)

print(f"TOTAL_FACES: {total_faces}", flush=True)
