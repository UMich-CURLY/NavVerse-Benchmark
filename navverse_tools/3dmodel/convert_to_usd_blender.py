import bpy
import os
import json
import glob
import sys
import mathutils

try:
    from pxr import Usd, UsdGeom, Gf
except ImportError:
    Usd = None
    print("pxr module not found. Will skip gallery creation or fallback.")

# Track used names to ensure uniqueness
used_names = {}

def get_unique_name(base_name):
    # Replace spaces with underscores
    clean_name = base_name.replace(" ", "_")
    
    if clean_name not in used_names:
        used_names[clean_name] = 1
        return clean_name
    else:
        count = used_names[clean_name]
        used_names[clean_name] += 1
        return f"{clean_name}_{count}"

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def process_file(filepath, meta_data, output_dir):
    filename = os.path.basename(filepath)
    if filename == "all_selected_objects.json":
        return None

    # Determine output name and target height
    target_height = None
    if filename in meta_data:
        item = meta_data[filename]
        if 'custom_name' in item:
            unique_name = get_unique_name(item['custom_name'])
        else:
            unique_name = get_unique_name(os.path.splitext(filename)[0])
        
        if 'height' in item:
            target_height = item['height']
    else:
        # Fallback if not in json
        base = os.path.splitext(filename)[0]
        unique_name = get_unique_name(base)
    
    output_path = os.path.join(output_dir, unique_name + ".usd")

    # If it exists, skip processing but return info
    if os.path.exists(output_path):
        print(f"Skipping {filename} because it already exists")
        # We still want to know the dimensions if possible, but for now we'll just return basic info
        return {
            "source_path": filepath,
            "usd_path": output_path,
            "target_height": target_height,
            "name": unique_name,
            "dims": [1, 1, target_height if target_height else 1] # Fallback dims
        }

    print(f"Processing: {filename}")
    reset_scene()

    ext = os.path.splitext(filename)[1].lower()
    
    try:
        if ext == ".blend":
            with bpy.data.libraries.load(filepath) as (data_from, data_to):
                data_to.objects = data_from.objects
            
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)

        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=filepath)
        elif ext in [".gltf", ".glb"]:
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif ext == ".obj":
            bpy.ops.import_scene.obj(filepath=filepath)
        else:
            print(f"Skipping unsupported format: {ext}")
            return None
        
        # Remove cameras and lights
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.type in ['CAMERA', 'LIGHT']:
                obj.select_set(True)
        bpy.ops.object.delete()

        # Select all objects
        bpy.ops.object.select_all(action='SELECT')
        selected_objects = bpy.context.selected_objects
        
        if not selected_objects:
            print(f"No objects imported from {filename}")
            return None

        # Ensure we are in object mode
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Make single user to avoid "Cannot apply to a multi user" error
        bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)

        # Update view layer to ensure matrices are fresh
        bpy.context.view_layer.update()
        
        # Calculate current bounds
        min_v = [float('inf'), float('inf'), float('inf')]
        max_v = [float('-inf'), float('-inf'), float('-inf')]
        
        has_geometry = False
        for obj in selected_objects:
            if obj.type == 'MESH':
                has_geometry = True
                # Get world coordinates of vertices
                mw = obj.matrix_world
                # Use bound_box (8 corners) to approximate extent
                for corner in obj.bound_box:
                    world_corner = mw @ mathutils.Vector(corner)
                    for i in range(3):
                        min_v[i] = min(min_v[i], world_corner[i])
                        max_v[i] = max(max_v[i], world_corner[i])
        
        final_dims = [0, 0, 0]
        if has_geometry:
            # Detect Width and Length (X and Y)
            size_x = max_v[0] - min_v[0]
            size_y = max_v[1] - min_v[1]
            
            # If the object is longer along Y, rotate it to align with X (length axis)
            if size_y > size_x:
                print(f"Rotating {filename} to align length with X axis (size_x: {size_x:.2f}, size_y: {size_y:.2f})")
                # Rotate 90 degrees around Z axis (local center)
                # To keep it simple and not mess up centering, we rotate first then re-calc bounds
                bpy.ops.transform.rotate(value=1.5708, orient_axis='Z') # 90 degrees in radians
                
                # Update matrices and re-calculate bounds after rotation
                bpy.context.view_layer.update()
                min_v = [float('inf'), float('inf'), float('inf')]
                max_v = [float('-inf'), float('-inf'), float('-inf')]
                for obj in selected_objects:
                    if obj.type == 'MESH':
                        mw = obj.matrix_world
                        for corner in obj.bound_box:
                            world_corner = mw @ mathutils.Vector(corner)
                            for i in range(3):
                                min_v[i] = min(min_v[i], world_corner[i])
                                max_v[i] = max(max_v[i], world_corner[i])

            # Center and Align to Ground
            center_x = (min_v[0] + max_v[0]) / 2
            center_y = (min_v[1] + max_v[1]) / 2
            min_z = min_v[2]
            
            translation = mathutils.Vector((-center_x, -center_y, -min_z))
            
            # Apply translation to all selected objects
            bpy.ops.transform.translate(value=translation)
            
            # Apply location to reset origins to 0,0,0 (pivot point)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
            
            current_h = max_v[2] - min_v[2]
            print(f"Current Height: {current_h}, Target Height: {target_height}")
            
            if target_height and current_h > 0.0001:
                scale_factor = target_height / current_h
                print(f"Scaling by {scale_factor}")

                # Parent all to a temp empty to scale together
                bpy.ops.object.select_all(action='DESELECT')
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
                root_empty = bpy.context.active_object
                
                for obj in selected_objects:
                    obj.select_set(True)
                    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
                    obj.select_set(False)
                
                # Scale root
                root_empty.scale = (scale_factor, scale_factor, scale_factor)
                
                # Select everything including root
                root_empty.select_set(True)
                for obj in selected_objects:
                    obj.select_set(True)
                
                # Apply scale logic
                bpy.ops.object.select_all(action='DESELECT')
                for obj in selected_objects:
                    obj.select_set(True)
                
                bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
                
                # Delete root
                bpy.ops.object.select_all(action='DESELECT')
                root_empty.select_set(True)
                bpy.ops.object.delete()
                
                # Now select objects and apply scale
                bpy.ops.object.select_all(action='DESELECT')
                for obj in selected_objects:
                    obj.select_set(True)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                
                final_dims = [
                    (max_v[0] - min_v[0]) * scale_factor,
                    (max_v[1] - min_v[1]) * scale_factor,
                    target_height
                ]
            else:
                final_dims = [
                    max_v[0] - min_v[0],
                    max_v[1] - min_v[1],
                    current_h
                ]

        # Select all objects again for export
        bpy.ops.object.select_all(action='SELECT')
        
        # Export USD
        if hasattr(bpy.ops.wm, "usd_export"):
            bpy.ops.wm.usd_export(filepath=output_path)
        else:
            print("USD export not supported in this Blender version.")
            
        print(f"Exported to {output_path}")

        return {
            "source_path": filepath,
            "usd_path": output_path,
            "target_height": target_height,
            "name": unique_name,
            "dims": final_dims
        }

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        return None

def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    if len(argv) < 3:
        print("Usage: blender -b -P convert_to_usd_blender.py -- <input_dir> <output_dir> <json_path>")
        base_path = os.getcwd()
        input_dir = os.path.join(base_path, "tools/3dmodel/data/result")
        output_dir = os.path.join(base_path, "tools/3dmodel/data/usd")
        json_path = os.path.join(input_dir, "all_selected_objects.json")
    else:
        input_dir = argv[0]
        output_dir = argv[1]
        json_path = argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Input Dir: {input_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"JSON Path: {json_path}")

    # Load JSON
    meta_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                if 'filename' in item:
                    meta_data[item['filename']] = item

    # Get all files
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for f in filenames:
            if f.lower().endswith(('.blend', '.fbx', '.glb', '.gltf', '.obj')):
                files.append(os.path.join(root, f))

    # Sort to ensure deterministic order for numbering
    files.sort()

    processed_items = []
    for f in files:
        res = process_file(f, meta_data, output_dir)
        if res:
            processed_items.append(res)

    # Save processed items info to a JSON for merge_usd.py
    info_path = os.path.join(output_dir, "object_info.json")
    with open(info_path, 'w') as f:
        json.dump(processed_items, f, indent=2)
    print(f"Saved object info to {info_path}")

    print("Conversion complete.")
    
    # --- Create Gallery ---
    print("Creating Gallery with USD References...")
    reset_scene()
    
    # Parent of result folder (input_dir)
    gallery_dir = os.path.dirname(input_dir.rstrip(os.sep))
    gallery_usd_path = os.path.join(gallery_dir, "all_objects_gallery.usd")
    
    if Usd:
        if os.path.exists(gallery_usd_path):
            os.remove(gallery_usd_path)
            
        stage = Usd.Stage.CreateNew(gallery_usd_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        
        root = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(root)
        
        current_x = 0.0
        cube_interval = 5
        
        for i, item in enumerate(processed_items):
            dims = item.get("dims", [1,1,1])
            
            # Add Reference Cube
            if i % cube_interval == 0:
                cube_path = f"/World/ReferenceCube_{i}"
                cube = UsdGeom.Cube.Define(stage, cube_path)
                cube.GetSizeAttr().Set(1.0)
                
                xform = UsdGeom.XformCommonAPI(cube)
                xform.SetTranslate(Gf.Vec3d(current_x, 0, 0.5))
                
                current_x += 2.0 
                
            # Add Referenced Object
            name = item["name"]
            usd_path = item["usd_path"]
            rel_path = os.path.relpath(usd_path, os.path.dirname(gallery_usd_path))
            
            safe_name = name.replace(" ", "_").replace("-", "_").replace(".", "_").replace("(", "").replace(")", "")
            
            prim_path = f"/World/Obj_{i}_{safe_name}"
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(f"./{rel_path}")
            
            xform = UsdGeom.XformCommonAPI(prim)
            xform.SetTranslate(Gf.Vec3d(current_x, 0, 0))
            
            size_x = dims[0]
            size_y = dims[1]
            max_footprint = max(size_x, size_y)
            
            spacing = max(2.0, max_footprint + 1.0)
            current_x += spacing
            
        stage.GetRootLayer().Save()
        print(f"Gallery saved to {gallery_usd_path}")
    else:
        print("Skipping gallery creation because pxr is missing.")
        
    print("Gallery creation complete.")

if __name__ == "__main__":
    main()
