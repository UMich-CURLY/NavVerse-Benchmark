import bpy
import sys
import os
import math
import argparse
import mathutils
import json
import shutil

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_scene():
    # Camera
    cam_data = bpy.data.cameras.new(name='Camera')
    cam = bpy.data.objects.new(name='Camera', object_data=cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (0, -2, 1)
    
    cam.data.lens = 35
    bpy.context.scene.camera = cam
    
    # White background
    if bpy.context.scene.world is None:
        new_world = bpy.data.worlds.new("World")
        bpy.context.scene.world = new_world
    
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes['Background']
    bg_node.inputs['Color'].default_value = (1, 1, 1, 1) # White
    bg_node.inputs['Strength'].default_value = 1.0

    # Uniform Lighting
    # 3-Point Light System + Fill
    
    # Key Light
    key_light_data = bpy.data.lights.new(name="Key Light", type='SUN')
    key_light = bpy.data.objects.new(name="Key Light", object_data=key_light_data)
    bpy.context.collection.objects.link(key_light)
    key_light.location = (5, -5, 10)
    key_light.data.energy = 5
    
    # Fill Light 1
    fill1_data = bpy.data.lights.new(name="Fill Light 1", type='POINT')
    fill1 = bpy.data.objects.new(name="Fill Light 1", object_data=fill1_data)
    bpy.context.collection.objects.link(fill1)
    fill1.location = (-5, -5, 5)
    fill1.data.energy = 200
    
    # Fill Light 2 (Back/Rim)
    fill2_data = bpy.data.lights.new(name="Fill Light 2", type='POINT')
    fill2 = bpy.data.objects.new(name="Fill Light 2", object_data=fill2_data)
    bpy.context.collection.objects.link(fill2)
    fill2.location = (0, 5, 5)
    fill2.data.energy = 150

    # Ambient Light (Area light from top)
    top_light_data = bpy.data.lights.new(name="Top Light", type='AREA')
    top_light = bpy.data.objects.new(name="Top Light", object_data=top_light_data)
    bpy.context.collection.objects.link(top_light)
    top_light.location = (0, 0, 10)
    top_light.rotation_euler = (0, 0, 0)
    top_light.data.shape = 'SQUARE'
    top_light.data.size = 10
    top_light.data.energy = 500

    return cam

def import_object(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.stl':
            # Blender 4.0+ changed the API for STL import to bpy.ops.wm.stl_import
            # We try the new API first.
            try:
                bpy.ops.wm.stl_import(filepath=file_path)
            except (AttributeError, RuntimeError):
                # Fallback to old API
                try:
                    bpy.ops.import_mesh.stl(filepath=file_path)
                except (AttributeError, RuntimeError) as e:
                    print(f"Failed to import STL using both wm.stl_import and import_mesh.stl: {e}")
                    return False

        elif ext == '.obj':
            # Blender 4.0+ changed the API for OBJ import to bpy.ops.wm.obj_import
            try:
                bpy.ops.wm.obj_import(filepath=file_path)
            except (AttributeError, RuntimeError):
                # Fallback to old API
                try:
                    bpy.ops.import_scene.obj(filepath=file_path)
                except (AttributeError, RuntimeError) as e:
                    print(f"Failed to import OBJ using both wm.obj_import and import_scene.obj: {e}")
                    return False

        elif ext == '.fbx':
            try:
                bpy.ops.import_scene.fbx(filepath=file_path)
            except (AttributeError, RuntimeError) as e:
                print(f"Failed to import FBX: {e}")
                return False

        elif ext == '.blend':
            try:
                bpy.ops.wm.open_mainfile(filepath=file_path)
            except (AttributeError, RuntimeError) as e:
                print(f"Failed to open Blend file: {e}")
                return False

        elif ext in ['.usd', '.usdc', '.usda', '.usdz']:
            try:
                bpy.ops.wm.usd_import(filepath=file_path)
            except (AttributeError, RuntimeError) as e:
                print(f"Failed to import USD: {e}")
                return False

        elif ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=file_path)
        else:
            print(f"Unsupported file format: {ext}")
            return False
    except Exception as e:
        print(f"Error importing {file_path}: {e}")
        return False
    return True

def setup_material(obj):
    # Only setup material if the object doesn't have one, or if we want to enforce it.
    # The requirement says "make sure the meshes are render with material".
    # If the imported model has textures/materials, we might want to keep them.
    # If it has none, we add a default one.
    
    # Check if object has any material slots
    if not obj.data.materials:
        # Create a new material
        mat = bpy.data.materials.new(name="ObjectMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        
        # Set a random color or a specific color
        # Let's use a nice blueish color for now
        bsdf.inputs['Base Color'].default_value = (0.1, 0.4, 0.8, 1) # RGBA
        bsdf.inputs['Roughness'].default_value = 0.5
        bsdf.inputs['Metallic'].default_value = 0.0
        
        obj.data.materials.append(mat)
    else:
        # Ensure existing materials use nodes and are visible
        # Sometimes imported materials are not set up for Eevee/Cycles properly
        for mat in obj.data.materials:
            if mat:
                if not mat.use_nodes:
                     mat.use_nodes = True
                
                # Check for image textures and ensure they are compatible
                if mat.node_tree:
                     for node in mat.node_tree.nodes:
                         if node.type == 'TEX_IMAGE':
                             # Sometimes image nodes are present but image is missing or corrupt
                             # causing "Failed to create GPU texture"
                             if not node.image:
                                 print(f"Warning: Missing image in material {mat.name}")
                             else:
                                 # Try to reload or pack if needed?
                                 pass

def normalize_object():
    # Select all mesh objects
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        return None
    
    # Join into one if multiple
    bpy.ops.object.select_all(action='DESELECT')
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    
    obj = bpy.context.active_object
    
    # Center origin to geometry
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # Move to (0,0,0)
    obj.location = (0, 0, 0)
    
    # Scale to fit in unit box
    max_dim = max(obj.dimensions)
    if max_dim > 0:
        scale = 1.0 / max_dim
        obj.scale = (scale, scale, scale)
        
    # Apply transform
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Setup material
    setup_material(obj)
    
    return obj

def render_angles(output_dir, num_angles=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cam = bpy.context.scene.camera
    
    # Find the object to look at (assumed normalized at 0,0,0)
    # If no object found, we can't really frame it properly, but we'll use (0,0,0) as target
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if mesh_objs:
        target_obj = mesh_objs[0]
        
        # Calculate camera distance to fit object
        # Taking max dimension which is normalized to ~1.0
        # Field of view of 35mm lens is approx 0.64 rad horizontal
        # Distance = (size / 2) / tan(fov / 2)
        # With size ~1.4 (diagonal of unit cube), and some margin
        # Let's set a safe radius
        # Or simpler: position camera and use 'View Selected' logic if possible
        pass
    else:
        # If no mesh, nothing to render properly
        import shutil
        print("No mesh found to render.")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        return

    # Create an empty at center to track
    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0, 0, 0)
    
    # Add track to constraint to camera
    const = cam.constraints.new(type='TRACK_TO')
    const.target = target
    const.track_axis = 'TRACK_NEGATIVE_Z'
    const.up_axis = 'UP_Y'
    
    # Adjust radius to ensure object is fully visible
    # We normalized the object to max dimension of 1.0.
    # The bounding box is roughly centered at 0,0,0.
    # With a unit cube (size 1), the max distance from origin is sqrt(0.5^2 + 0.5^2 + 0.5^2) ~= 0.87.
    # To fit this sphere in camera view:
    # Sensor width = 36mm (default), Lens = 35mm.
    # Horizontal FOV = 2 * atan(36/(2*35)) ~= 54.4 degrees.
    # Distance D = Radius / sin(FOV/2).
    # sin(27.2) ~= 0.457.
    # D = 0.87 / 0.457 ~= 1.9.
    # We add a safety factor.
    radius = 2.2
    
    # Check bounding box dimensions to be sure
    if mesh_objs:
        dims = mesh_objs[0].dimensions
        max_dim = max(dims)
        # Recalculate if for some reason normalization failed or we want to be super precise
        # But we trust normalize_object to scale it to 1.0.
        # Just in case, let's use the actual dimensions relative to camera.
        # Since we rotate around, we want the bounding sphere radius.
        bbox_corners = [mesh_objs[0].matrix_world @ mathutils.Vector(corner) for corner in mesh_objs[0].bound_box]
        max_dist = max([(v - mesh_objs[0].location).length for v in bbox_corners])
        
        # Recalculate radius
        fov = 2 * math.atan(bpy.context.scene.camera.data.sensor_width / (2 * bpy.context.scene.camera.data.lens))
        required_dist = max_dist / math.sin(fov / 2)
        radius = required_dist * 1.2 # 20% margin
        
        # Clamp minimum radius to avoid clipping
        radius = max(radius, 1.5)
    
    rendered = False
    try:
        for i in range(num_angles):
            angle = (i / num_angles) * 2 * math.pi
            x = radius * math.sin(angle)
            y = -radius * math.cos(angle)
            z = radius * 0.5 # Elevated view proportional to distance
            
            cam.location = (x, y, z)
            bpy.context.view_layer.update()
            
            bpy.context.scene.render.filepath = os.path.join(output_dir, f"render_{i}.png")
            # Faster render settings
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            bpy.context.scene.render.resolution_x = 512
            bpy.context.scene.render.resolution_y = 512
            
            bpy.ops.render.render(write_still=True)
            rendered = True
    except Exception as e:
        print(f"Render failed: {e}")
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        return

    if not rendered:
         import shutil
         if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    # Get arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metadata", required=False, help="JSON string of metadata")
    args = parser.parse_args(argv)


    output_dir = args.output_dir
    output_dir = output_dir.rstrip(os.sep)
    no_texture_output_dir = output_dir.split(os.sep)[:-1] + ["no_texture_" + output_dir.split(os.sep)[-1]]
    no_texture_output_dir = os.path.join(*no_texture_output_dir)

    if os.path.exists(output_dir) or os.path.exists(no_texture_output_dir):
        print(f"Output directory {output_dir} already exist. Exiting.")
        exit(0)
    
    reset_scene()
    if import_object(args.input):
        setup_scene()
        
        # Normalize and get object
        obj = normalize_object()
        
        # Check for textures
        has_texture = False
        if obj and obj.data.materials:
            for mat in obj.data.materials:
                if mat and mat.node_tree:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            # Verify if image data is actually loaded or packed
                            if node.image.source == 'FILE':
                                try:
                                    # Try to access the size to see if it's loaded
                                    image_size = node.image.size[0]
                                    if image_size > 0:
                                        has_texture = True
                                        break
                                except:
                                    # Image might be missing or broken link
                                    pass
                            else:
                                # Generated or Packed
                                has_texture = True
                                break
                if has_texture:
                    print(f"Texture found for object: {obj.name}, type: {node.image.type}, source: {node.image.source}")
                    break
        
        
        # Update metadata and output directory if no texture
        metadata = {}
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError:
                print("Error parsing metadata JSON")
        
        if not has_texture:
            print("No texture found for object.")
            # Rename output directory to end with _no_texture
            # We assume output_dir is the intended directory
            # If it already exists, we might need to handle it, but script usually creates it
            # The calling script creates the directory, so we might need to rename it
            # But wait, render_angles takes output_dir. 
            # If we change output_dir here, we should ensure the directory exists.
            
            # Remove trailing slash if any
            new_output_dir = no_texture_output_dir
            
            # If original dir exists, rename it
            if os.path.exists(output_dir) and output_dir != new_output_dir:
                 # Check if new destination already exists
                if os.path.exists(new_output_dir):
                    shutil.rmtree(new_output_dir)
                os.rename(output_dir, new_output_dir)
                output_dir = new_output_dir
            elif not os.path.exists(output_dir):
                 # Create the new directory
                 os.makedirs(new_output_dir, exist_ok=True)
                 output_dir = new_output_dir
            
            metadata["has_texture"] = False
        else:
            metadata["has_texture"] = True
            
        # Save metadata
        if metadata:
            # Ensure output dir exists (in case rename logic didn't trigger or failed)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "info.json"), "w") as f:
                json.dump(metadata, f, indent=4)
        
        render_angles(output_dir)
