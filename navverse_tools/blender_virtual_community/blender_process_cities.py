import bpy
import os
import json
import numpy as np
from tqdm import tqdm
import sys,contextlib

# ========== Config ==========
cities = ["AMSTERDAM", "AUSTIN", "BALTIMORE", "BARCELONA", "BELGRADE", "BERLIN", "BOISE", "BOSTON", "BRATISLAVA", "BRUSSELS", "BUDAPEST", "CALGARY", "CHARLOTTE", "CHICAGO", "CHRISTCHURCH", "COLUMBUS", "DENVER", "DETROIT", "EL_PASO", "FLORENCE", "FORT_WORTH", "FRANKFURT", "HAMBURG", "HARVARD", "KANSAS_CITY", "LASVEGAS", "LONDON", "LONGISLAND", "MADISON", "MADRID", "MADRID2", "MILAN", "MINNEAPOLIS", "MIT", "MONTREAL", "NY", "ORLANDO", "PARIS", "PHILADELPHIA", "PORTLAND", "ROME", "SANFRANCISCO", "SANFRANCISCO2", "SILICONVALLEY", "STANFORD", "SYDNEY", "TORONTO", "UCLA", "UMASS", "WHITEHOUSE", "YALE", "ZURICH"]


cities = ["AMSTERDAM"]

DATA_DIR = "D:/Desktop/ViCo"
OUTPUT_BASE_DIR = f"{DATA_DIR}/generated"  # Base directory for exported USD
HDRI_PATH = f"{DATA_DIR}/qwantani_sunrise_puresky_2k.exr"  # HDRI file path
OBJECTS_BASE_DIR = f"{DATA_DIR}/objects/outdoor_objects/retrieved"

# ==========================

def clear_scene():
    """Clear current scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear mesh data blocks
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)

def import_glb(filepath):
    """Import GLB"""
    if os.path.exists(filepath):
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        print(f"[WARNING] File does not exist: {filepath}")

def import_usd(filepath):
    """Import USD"""
    if os.path.exists(filepath):
        # Record the number of objects before import
        objects_before = len(bpy.data.objects)
        materials_before = len(bpy.data.materials)
        images_before = len(bpy.data.images)
        
        bpy.ops.wm.usd_import(filepath=filepath)
        
        # Record the state after import
        objects_after = len(bpy.data.objects)
        materials_after = len(bpy.data.materials)
        images_after = len(bpy.data.images)
        
        print(f"[INFO] Import USD {os.path.basename(filepath)} - Objects: {objects_before}→{objects_after}, Materials: {materials_before}→{materials_after}, Textures: {images_before}→{images_after}")
    else:
        print(f"[WARNING] File not found: {filepath}")

def setup_hdri_world(hdri_path):
    """Setup HDRI sky"""
    world = bpy.data.worlds.get("World")
    if not world:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Get existing nodes (default has Background and World Output)
    bg = nodes.get("Background")
    output = nodes.get("World Output")

    env = nodes.new(type="ShaderNodeTexEnvironment")
    env.location = (-300, 0)
    env.image = bpy.data.images.load(hdri_path)

    # Connect nodes: Environment → Background → World Output
    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # Adjust brightness intensity (default 10.0, can be changed)
    bg.inputs["Strength"].default_value = 10.0
    
    
def get_object_height(obj):
    z_values = [v[2] for v in obj.bound_box]
    return max(z_values) - min(z_values)

def import_objects_from_json(json_files, base_dir):
    """Import objects from JSON files"""
    import time 
    
    [amenities_file_with_z, natural_file_with_z] = json_files 
    with open(amenities_file_with_z, "r") as f: 
        amenities_data = json.load(f) 
    with open(natural_file_with_z, "r") as f: 
        natural_data = json.load(f) 
    merged_data = { "objects": amenities_data["objects"] + natural_data["objects"] } 
    total_objects = len(merged_data["objects"]) 
    print(f"[INFO] Preparing to import {total_objects} objects") # Analyze repetition 
    unique_glb_files = {} 
    for obj in merged_data["objects"]: 
        obj_path = obj["asset_path"] 
        if obj_path in unique_glb_files: 
            unique_glb_files[obj_path] += 1 
        else: 
            unique_glb_files[obj_path] = 1 
            
    unique_count = len(unique_glb_files) 
    print(f"[INFO] Found {unique_count} unique GLB files") 
    print(f"[INFO] Repetition rate: {((total_objects - unique_count) / total_objects * 100):.1f}%") 
    
    # Show the most repeated files 
    sorted_files = sorted(unique_glb_files.items(), key=lambda x: x[1], reverse=True) 
    print(f"[INFO] Most repeated GLB files:") 
    for i, (file_path, count) in enumerate(sorted_files[:5]): 
        print(f" {i+1}. {os.path.basename(file_path)}: {count} times") #
        
    # GLB file cache - store the object data of the imported GLB files 
    glb_cache = {} 
    
    # Line performance analysis variables 
    line_times = { "obj_path_extraction": [], "location_extraction": [], "rotation_extraction": [], "tags_extraction": [], "file_exists_check": [], "glb_import": [], "glb_cache_hit": [], "get_selected_objects": [], "get_object_height": [], "height_calculation": [], "object_transform": [], "object_naming": [], "deselect_all": [] } 
    
    iteration_times = [] 
    
    for i, obj in enumerate(tqdm(merged_data["objects"], desc="Importing objects")): 
        iteration_start = time.time() 
        
        # 1. Extract object path 
        step_start = time.time() 
        obj_path = obj["asset_path"] 
        line_times["obj_path_extraction"].append(time.time() - step_start) 
        
        # 2. Extract location information 
        step_start = time.time() 
        location = obj.get("location", [0, 0, 0]) 
        line_times["location_extraction"].append(time.time() - step_start) 
        
        # 3. Extract rotation information 
        step_start = time.time() 
        rotation = obj.get("rotation", [0, 0, 0]) 
        line_times["rotation_extraction"].append(time.time() - step_start) 
        
        # 4. Extract tag information 
        step_start = time.time() 
        tags = obj.get("tags", {}) 
        line_times["tags_extraction"].append(time.time() - step_start) 
        
        # 5. Check if the file exists 
        step_start = time.time() 
        if not os.path.exists(obj_path): 
            print(f"GLB file not found: {obj_path}") 
            continue 
        line_times["file_exists_check"].append(time.time() - step_start) 
        
        # 6. Check GLB cache or import GLB file 
        step_start = time.time() 
        if obj_path in glb_cache: 
            # Copy objects from cache 
            cache_data = glb_cache[obj_path] 
            imported_objs = [] 
            
            # Batch create objects (optimized version) 
            objects_to_link = [] 
            for cached_obj_data in cache_data: 
                if "mesh" in cached_obj_data: # Ensure cache has valid mesh data 
                    # Create new object 
                    new_obj = bpy.data.objects.new(cached_obj_data["name"], cached_obj_data["mesh"]) 
                    objects_to_link.append(new_obj) 
                    imported_objs.append(new_obj) 
                else: # Ensure cache has valid mesh data 
                    # Create new object 
                    new_obj = bpy.data.objects.new(cached_obj_data["name"], None) 
                    objects_to_link.append(new_obj) 
                    imported_objs.append(new_obj) 
                    
                # Batch link objects to scene (faster than individual linking) 
            for obj in objects_to_link: 
                bpy.context.collection.objects.link(obj) 
            line_times["glb_cache_hit"].append(time.time() - step_start) 
        else: # First import, save to cache 
            bpy.ops.import_scene.gltf(filepath=obj_path) 
            imported_objs = bpy.context.selected_objects.copy() 
            
            # Save to cache 
            cache_data = [] 
            skipped_objects = [] 
            for obj in imported_objs: 
                if obj.data is not None: # Ensure object has mesh data 
                    cache_data.append({ "name": obj.name, "mesh": obj.data.copy()}) 
                else: 
                    cache_data.append({ "name": obj.name }) 
                    skipped_objects.append(obj.name) 
            if skipped_objects: 
                print(f"[WARNING] GLB file {os.path.basename(obj_path)} has {len(skipped_objects)} objects without mesh data: {skipped_objects}") 
            glb_cache[obj_path] = cache_data 
            line_times["glb_import"].append(time.time() - step_start) 
            
            # 7. Get imported objects (already handled above for cache hit) 
            step_start = time.time() 
            line_times["get_selected_objects"].append(time.time() - step_start) 
            
            # 8. Calculate object height 
            step_start = time.time() 
            obj_height = get_object_height(imported_objs[0]) 
            line_times["get_object_height"].append(time.time() - step_start) 
            
            # 9. Height calculation and scaling 
            step_start = time.time() 
            rescaled_obj_height = np.clip(obj_height, tags["rescale"]["height"][0], tags["rescale"]["height"][1]) 
            rescale_ratio = rescaled_obj_height / obj_height 
            scale = [rescale_ratio, rescale_ratio, rescale_ratio] 
            line_times["height_calculation"].append(time.time() - step_start) 
            
            # 10. Object transformation setting and naming (merged optimization) 
            step_start = time.time() 
            for imported_obj in imported_objs: 
                # Set location, rotation, and scale 
                imported_obj.location = location 
                imported_obj.rotation_euler = rotation 
                imported_obj.scale = scale 
                
                # Optional: Set object name 
                if "name" in tags: 
                    imported_obj.name = tags["name"] 
            for imported_obj in imported_objs: 
                if hasattr(imported_obj, 'select_set'): 
                    imported_obj.select_set(False) 
                elif hasattr(imported_obj, 'select'): 
                    imported_obj.select = False 
            line_times["deselect_all"].append(time.time() - step_start)
    
    

def import_buildings_from_directory(scene_dir):
    """Import all GLB files in the buildings directory"""
    buildings_dir = os.path.join(scene_dir, "buildings")
    
    if not os.path.exists(buildings_dir):
        print(f"[WARNING] Buildings directory does not exist: {buildings_dir}")
        return
    
    # Get all GLB files
    glb_files = []
    for file in os.listdir(buildings_dir):
        if file.lower().endswith('.glb'):
            glb_files.append(os.path.join(buildings_dir, file))
    
    if not glb_files:
        print(f"[INFO] No GLB files found in buildings directory: {buildings_dir}")
        return
    
    print(f"[INFO] Found {len(glb_files)} building GLB files")
    
    # Create error log file
    error_log_path = os.path.join(scene_dir, "building_import_errors.txt")
    failed_buildings = []
    
    # Import all building GLB files
    imported_count = 0
    for glb_path in tqdm(glb_files, desc="Importing buildings"):
        if imported_count >= 200:
            break
        try:
            if os.path.exists(glb_path):
                bpy.ops.import_scene.gltf(filepath=glb_path)
                imported_count += 1
                
                # Optional: Set prefix for building object names
                imported_objs = bpy.context.selected_objects
                for obj in imported_objs:
                    if not obj.name.startswith("buildings_"):
                        obj.name = f"buildings_{obj.name}"
                
                # Optimized: Only deselect the current imported object (instead of all scene objects)
                for obj in imported_objs:
                    if hasattr(obj, 'select_set'):
                        obj.select_set(False)
                    elif hasattr(obj, 'select'):
                        obj.select = False
            else:
                error_msg = f"File not found: {glb_path}"
                print(f"[WARNING] {error_msg}")
                failed_buildings.append(f"{os.path.basename(glb_path)} - {error_msg}")
                
        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            print(f"[ERROR] Import GLB file failed {glb_path}: {e}")
            failed_buildings.append(f"{os.path.basename(glb_path)} - {error_msg}")
    
    # Write error log file
    if failed_buildings:
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Building import error log\n")
            f.write(f"Processing time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files: {len(glb_files)}\n")
            f.write(f"Successfully imported: {imported_count}\n")
            f.write(f"Failed count: {len(failed_buildings)}\n")
            f.write(f"Success rate: {(imported_count/len(glb_files)*100):.1f}%\n\n")
            f.write("Failed file list:\n")
            f.write("=" * 50 + "\n")
            for failed in failed_buildings:
                f.write(f"{failed}\n")
        
        print(f"[WARNING] {len(failed_buildings)} building files imported failed, details saved to: {error_log_path}")
    else:
        print(f"[SUCCESS] All building files imported successfully!")
    
    print(f"[COMPLETED] Successfully imported {imported_count}/{len(glb_files)} building files")
    
    

@contextlib.contextmanager
def suppress_output():
    class DevNull:
        def write(self, msg): pass
        def flush(self): pass
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = DevNull(), DevNull()
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def process_scene(scene_dir, output_dir, city):
    """Process a single city scene"""
    print(f"\n====== Processing city: {city} ======")
    
    # Check if there are objects data
    amenities_json = os.path.join(scene_dir, "objects/amenities_with_z.json")
    natural_json = os.path.join(scene_dir, "objects/natural_with_z.json")
    
    has_objects = False
    if os.path.exists(amenities_json) and os.path.exists(natural_json):
        try:
            with open(amenities_json, "r") as f:
                amenities_data = json.load(f)
            with open(natural_json, "r") as f:
                natural_data = json.load(f)
            
            total_objects = len(amenities_data.get("objects", [])) + len(natural_data.get("objects", []))
            if total_objects > 0:
                has_objects = True
                print(f"[INFO] Found {total_objects} objects")
            else:
                print(f"[INFO] No objects found")
        except Exception as e:
            print(f"[WARNING] Failed to read objects JSON file: {e}")
    else:
        print(f"[INFO] Objects JSON file does not exist")
    
    # Check if there are buildings
    buildings_dir = os.path.join(scene_dir, "buildings")
    has_buildings = False
    building_count = 0
    if os.path.exists(buildings_dir):
        glb_files = [f for f in os.listdir(buildings_dir) if f.lower().endswith('.glb')]
        building_count = len(glb_files)
        if building_count > 0:
            has_buildings = True
            print(f"[INFO] Found {building_count} building files")
        else:
            print(f"[INFO] No building files found")
    else:
        print(f"[INFO] Buildings directory does not exist")
    
    # If there are no objects and buildings, skip processing and generate info.txt
    if not has_objects or not has_buildings:
        print(f"[SKIP] {city} has no objects and buildings, skipping processing")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate info.txt file
        info_file = os.path.join(output_dir, f"{city}_info.txt")
        with open(info_file, "w", encoding="utf-8") as f:
            f.write(f"City processing information - {city}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Processing time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing status: Skip\n")
            f.write(f"Skip reason: No objects or buildings\n\n")
            f.write("Check results:\n")
            f.write(f"  Objects count: {total_objects if 'total_objects' in locals() else 0}\n")
            f.write(f"  Buildings count: {building_count}\n")
            f.write(f"  Amenities JSON: {'exists' if os.path.exists(amenities_json) else 'does not exist'}\n")
            f.write(f"  Natural JSON: {'exists' if os.path.exists(natural_json) else 'does not exist'}\n")
            f.write(f"  Buildings directory: {'exists' if os.path.exists(buildings_dir) else 'does not exist'}\n")
        
        print(f"[INFO] Generated skip information file: {info_file}")
        return
    
    print(f"[CONTINUE] Starting to process {city}")
    print(bpy.ops.wm.usd_export.get_rna_type().properties.keys())
    clear_scene()

    # Import roof and terrain
    
    print("Importing roof and terrain")
    
    import_glb(os.path.join(scene_dir, "roof.glb"))
    import_glb(os.path.join(scene_dir, "terrain.glb"))

    # Import all GLB files in the buildings directory
    print("Importing buildings")
    import_buildings_from_directory(scene_dir)

    # Import objects from JSON
    print("Importing objects from JSON")
    amenities_json = os.path.join(scene_dir, "objects/amenities_with_z.json")
    natural_json = os.path.join(scene_dir, "objects/natural_with_z.json")
    # import_objects_from_json([amenities_json, natural_json], OBJECTS_BASE_DIR)

    # Set HDRI
    print("Setting up HDRI")
    setup_hdri_world(HDRI_PATH)

    # Export USD
    print("Exporting USD")
    os.makedirs(output_dir, exist_ok=True)
    usd_out = os.path.join(output_dir, f"{city}_no_obj_200_buildings.usd")
    
    win = bpy.context.window_manager.windows[0]
    area = None
    for a in win.screen.areas:
        if a.type in {'VIEW_3D', 'OUTLINER', 'PROPERTIES', 'TOPBAR'}:
            area = a
            break
    if area is None:
        area = win.screen.areas[0]

    region = None
    for r in area.regions:
        if r.type == 'WINDOW':
            region = r
            break
    if region is None:
        region = area.regions[-1]

    
    bpy.ops.preferences.addon_enable(module="omni_nucleus")
    bpy.ops.preferences.addon_enable(module="omni_optimization_panel")
    bpy.ops.preferences.addon_enable(module="omni_panel")
    bpy.ops.preferences.addon_enable(module="omni_audio2face")
    bpy.ops.preferences.addon_enable(module="umm2")

    # 2) (Optional) Open generate_mdl in the plugin preferences to prevent operator parameters from being incorrectly passed
    try:
        prefs = bpy.context.preferences.addons["omni_usd"].preferences
        if hasattr(prefs, "generate_mdl"):
            prefs.generate_mdl = True
            print("Set prefs.generate_mdl = True")
    except KeyError:
        pass



    # 3) Use temp_override to execute export
    with bpy.context.temp_override(window=win, area=area, region=region):
        with suppress_output():
            res = bpy.ops.wm.usd_export(
                'EXEC_DEFAULT',
                filepath=usd_out,
                export_materials=True,
                export_textures=True,
                generate_preview_surface=True,  
                generate_mdl=True,               
                relative_paths=True,
                convert_to_cm=False
            )
    print("usd_export result:", res)
    print("Loaded addons:", list(bpy.context.preferences.addons.keys()))
    print("User config dir:", bpy.utils.resource_path('USER'))
    print("Scripts dir:", bpy.utils.script_paths())
    import sys
    print("Blender binary:", bpy.app.binary_path)
    print("Python exec:", sys.executable)
    print(bpy.utils.script_paths())
    
    op = bpy.ops.wm.usd_export.get_rna_type().bl_rna
    print("Operator idname:", op.identifier)
    print("Python type:", op)   
    
    print("generate_mdl =", bpy.ops.wm.usd_export.get_rna_type().properties['generate_mdl'].default)
    
    
    print(f"[COMPLETED] {city} exported: {usd_out}")
    

def main():
    for city in cities:
        # if os.path.exists(os.path.join(OUTPUT_BASE_DIR, city)):
        #     continue
        scene_dir = os.path.join(DATA_DIR, city)
        output_dir = os.path.join(OUTPUT_BASE_DIR, city)
        process_scene(scene_dir, output_dir, city)

if __name__ == "__main__":
    main()
