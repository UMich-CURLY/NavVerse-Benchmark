import bpy
import os

def decimate_usd(input_usd_path, output_usd_path, ratio=0.1):
    """
    Imports a USD file, decimates all meshes, and exports to a new USD file.
    
    Args:
        input_usd_path (str): The full path to the USD file to import.
        output_usd_path (str): The full path for the decimated USD file.
        ratio (float): The decimation ratio. A value of 0.1 means 10% of the original faces.
    """
    
    # Check if the input file exists
    if not os.path.exists(input_usd_path):
        print(f"Error: Input file not found at {input_usd_path}")
        return

    # Clear the scene of existing objects before import
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print(f"Importing USD file: {input_usd_path}")
    # Import the USD file
    try:
        bpy.ops.wm.usd_import(filepath=input_usd_path)
    except Exception as e:
        print(f"Error importing USD file: {e}")
        return

    original_face_count = 0
    decimated_face_count = 0
    # Iterate through all objects in the scene
    processed_meshes = set()
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.data not in processed_meshes:
            processed_meshes.add(obj.data)
            print(f"Processing mesh object: {obj.name}")
            
            # Count number of faces
            face_count = len(obj.data.polygons)
            original_face_count += face_count
            print(f"Number of faces: {face_count}")
            if face_count < 500:
                decimated_face_count += face_count
                print(f"Number of faces after decimation: {face_count}")
                continue
            

            # Select the object and make it active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)

            # Add the Decimate modifier
            decimate_modifier = obj.modifiers.new(name="Decimate_Faces", type='DECIMATE')
            decimate_modifier.ratio = ratio
            decimate_modifier.use_dissolve_boundaries = False

            # Apply the modifier
            # Using bpy.ops.object.modifier_apply(modifier="Modifier Name") can fail in scripts.
            # A more robust method is to use the object.convert() operator.
            bpy.ops.object.modifier_apply(modifier="Decimate_Faces")
            
            # Count number of faces after decimation
            face_count = len(obj.data.polygons)
            print(f"Number of faces after decimation: {face_count}")
            decimated_face_count += face_count

            # Deselect the object
            obj.select_set(False)

    print(f"Total original faces: {original_face_count}")
    print(f"Total decimated faces: {decimated_face_count}")
    print(f"Exporting decimated USD file: {output_usd_path}")
    # Export the scene as a new USD file
    try:
        bpy.ops.wm.usd_export(filepath=output_usd_path,
                              #export_textures=True,
                                relative_paths=True,
                                export_materials=False,
                                export_animation=False,
                                export_uvmaps=True,
                                use_instancing=True,
                            )
    except Exception as e:
        print(f"Error exporting USD file: {e}")
        return

    print("Decimation and export complete.")