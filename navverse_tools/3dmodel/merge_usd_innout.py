#!/usr/bin/env python
import argparse
import os
import json
import random
import glob
import subprocess
import sys
import shutil
import math

try:
    from pxr import Usd, UsdGeom, Sdf, Gf
except ImportError:
    print("Error: pxr module not found. Please run with an environment that has USD (e.g. env_isaaclab).")
    sys.exit(1)

# Helper to check proximity
def is_far_enough(loc, existing_objects, min_dist):
    for other_loc in existing_objects:
        dist = math.sqrt(sum((loc[i] - other_loc[i])**2 for i in range(2)))
        if dist < min_dist:
            return False
    return True

class SuppressFD:
    def __init__(self, fd=2):
        self.fd = fd
        self.devnull = None
        self.save_fd = None

    def __enter__(self):
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        self.save_fd = os.dup(self.fd)
        os.dup2(self.devnull, self.fd)

    def __exit__(self, *args):
        os.dup2(self.save_fd, self.fd)
        os.close(self.devnull)
        os.close(self.save_fd)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scene_folder", type=str, default="./data", help="Input folder")
parser.add_argument("--city", type=str, default="all", help="City name (all, AMSTERDAM...)")
args = parser.parse_args()

SCENE_DATA_DIR = args.scene_folder
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERT_SCRIPT = os.path.join(TOOLS_DIR, "convert_to_usd_blender.py")
RESULT_DIR = os.path.join(SCENE_DATA_DIR, "objaverse/result")
USD_MODELS_DIR = os.path.join(SCENE_DATA_DIR, "objaverse/usd")
ALL_SELECTED = os.path.join(RESULT_DIR, "all_selected_objects.json")

def convert_models():
    print("--- Step 1 & 2: Converting models with Blender ---")
    if not os.path.exists(USD_MODELS_DIR):
        os.makedirs(USD_MODELS_DIR)
    
    # Check if blender is available
    blender_cmd = "blender"
    # Try to find blender in common paths if not in PATH
    if shutil.which(blender_cmd) is None:
        if os.path.exists("/snap/bin/blender"):
            blender_cmd = "/snap/bin/blender"
        elif os.path.exists("/usr/bin/blender"):
            blender_cmd = "/usr/bin/blender"
        # Add mac/windows paths if needed
    
    print(f"Using Blender: {blender_cmd}")
    
    cmd = [
        blender_cmd, 
        "-b", 
        "-P", CONVERT_SCRIPT, 
        "--", 
        RESULT_DIR, 
        USD_MODELS_DIR, 
        ALL_SELECTED
    ]
    
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Blender conversion failed: {e}")
        # Proceeding anyway as some models might have been converted
    except FileNotFoundError:
        print("Blender not found. Skipping conversion (assuming files exist).")

# Configuration
def main(CITY):
    print(f"Processing {CITY}")
    
    VC_SCENE_DIR = os.path.join(SCENE_DATA_DIR, f"vc_plus/{CITY}")
    LANE_POINTS = os.path.join(VC_SCENE_DIR, "lane_points.json")
    SIDEWALK_POINTS = os.path.join(VC_SCENE_DIR, "sidewalk_points.json")
    ROAD_OBJECTS = os.path.join(VC_SCENE_DIR, "road_objects_info.json")
    SCENE_USD = os.path.join(VC_SCENE_DIR, f"{CITY}_gr_merged.usd")
    OUTPUT_MERGED_USD = os.path.join(VC_SCENE_DIR, f"{CITY}_innout.usd")

    # Store types from store_front.ipynb
    STORE_TYPES = [
        "restaurant", "cafe", "fast_food", "convenience_store", "parking_entrance",
        "bar", "bank", "atm", "bicycle_rental", "pharmacy", "toilets", "theatre",
        "library", "dentist", "school", "post_office", "bureau_de_change", "doctors",
        "bicycle_repair_station", "clinic", "community_centre", "car_rental",
        "police", "arts_centre", "cinema", "kindergarten", "university",
        "coworking_space", "lab"
    ]

    SEMANTIC_GROUPS_JSON = os.path.join(SCENE_DATA_DIR, "../tools/3dmodel/semantic_groups.json")
    if not os.path.exists(SEMANTIC_GROUPS_JSON):
        # Fallback relative path
        SEMANTIC_GROUPS_JSON = os.path.join(TOOLS_DIR, "data/usd/semantic_groups.json")

    def generate_objaverse_json(room_loc):
        print("--- Step 3: Sampling objects ---")
        
        if not os.path.exists(LANE_POINTS) or not os.path.exists(SIDEWALK_POINTS):
            print(f"Error: Points files not found at {LANE_POINTS} or {SIDEWALK_POINTS}")
            return []

        with open(LANE_POINTS) as f:
            lane_data = json.load(f)
        with open(SIDEWALK_POINTS) as f:
            sidewalk_data = json.load(f)

        # Categorize USD models and map filenames to full paths
        usd_files = glob.glob(os.path.join(USD_MODELS_DIR, "*.usd"))
        if not usd_files:
            print("No USD files found. Cannot sample.")
            return []

        cars = []
        others = []
        filename_to_path = {}
        
        # keywords for cars
        car_keywords = ["car", "truck", "van", "ambulance", "taxi", "bus"]
        
        for filepath in usd_files:
            filename = os.path.basename(filepath) # keep case sensitive for matching
            filename_lower = filename.lower()
            filename_to_path[filename] = filepath
            
            if any(x in filename_lower for x in car_keywords):
                cars.append(filepath)
            else:
                others.append(filepath)
                
        print(f"Found {len(cars)} cars and {len(others)} other objects.")
        
        # Load semantic groups
        semantic_groups = []
        if os.path.exists(SEMANTIC_GROUPS_JSON):
            try:
                with open(SEMANTIC_GROUPS_JSON, 'r') as f:
                    semantic_groups = json.load(f)
                print(f"Loaded {len(semantic_groups)} semantic groups.")
            except Exception as e:
                print(f"Error loading semantic groups: {e}")
        else:
            print(f"Warning: Semantic groups file not found at {SEMANTIC_GROUPS_JSON}")

        # Load traffic light locations from road_objects_info.json for exclusion
        tl_locs = []
        if os.path.exists(ROAD_OBJECTS):
            with open(ROAD_OBJECTS) as f:
                road_data = json.load(f)
                for obj in road_data.get('objects', []):
                    if obj.get('category') in ['crossing', 'traffic_signals']:
                        loc = obj.get('final_location') or obj.get('location')
                        if loc:
                            tl_locs.append(loc)
            print(f"Found {len(tl_locs)} traffic light locations from road objects.")

        # Collect store locations from SCENE_USD for exclusion
        store_locs = []
        if os.path.exists(SCENE_USD):
            print(f"Loading {SCENE_USD} to find store fronts...")
            temp_stage = Usd.Stage.Open(SCENE_USD)
            for prim in temp_stage.Traverse():
                name = prim.GetName().lower()
                if any(s_type in name for s_type in STORE_TYPES):
                    xformable = UsdGeom.Xformable(prim)
                    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = world_transform.ExtractTranslation()
                    store_locs.append([translation[0], translation[1], translation[2]])
            print(f"Found {len(store_locs)} store fronts.")
        else:
            print(f"Warning: Scene USD not found at {SCENE_USD}. Skipping store front exclusion.")

        # Load object dimensions for dynamic distance check
        obj_info_map = {}
        info_path = os.path.join(USD_MODELS_DIR, "object_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info_list = json.load(f)
                for item in info_list:
                    usd_path = item['usd_path']
                    dims = item.get('dims', [1, 1, 1])
                    # footprint size is the max of width and length
                    obj_info_map[usd_path] = {'dims': dims, 'size': max(dims[0], dims[1])}
        
        def get_obj_size(filepath):
            info = obj_info_map.get(filepath)
            return info['size'] if info else 2.0 # Default 2m if not found
        
        def get_obj_dims(filepath):
            info = obj_info_map.get(filepath)
            return info['dims'] if info else [2.0, 2.0, 2.0]

        new_objects = []

        # Track placed objects with full info
        placed_objects_info = [] # List of dicts: {'loc': loc, 'size': size, 'dims': dims}
        
        def is_far_enough_dynamic(loc, size, existing_placed, extra_gap=1.0):
            for other_obj in existing_placed:
                other_loc = other_obj['loc']
                other_size = other_obj['size']
                # Min distance = (size_a/2) + (size_b/2) + extra_gap
                min_dist = (size / 2.0) + (other_size / 2.0) + extra_gap
                dist = math.sqrt(sum((loc[i] - other_loc[i])**2 for i in range(2)))
                if dist < min_dist:
                    return False
            return True

        car_locs = []
        X_DIST = 15.0  # Min distance between cars
        STORE_DIST = 5.0 # Min distance from store fronts
        TL_PROP_DIST = 3.0 # Min distance between props and traffic lights
        CAR_PROP_DIST = 5.0 # Min distance between props and cars

        # Sample Lane (m=20) - Sample cars first to establish exclusion zones
        if cars:
            n_lane = min(200, len(lane_data.get('points', [])))
            # Randomize the whole list so we can pick candidates that satisfy distance
            all_lane_points = list(lane_data.get('points', []))
            print(f"Found {len(all_lane_points)} lane points")
            random.shuffle(all_lane_points)
            
            count = 0
            for p in all_lane_points:
                if count >= n_lane:
                    break
                
                loc = [p['utm_x'], p['utm_y'], p['height']]

                # Filter by room distance (50m)
                if room_loc:
                    dist_room = math.sqrt((loc[0]-room_loc[0])**2 + (loc[1]-room_loc[1])**2)
                    if dist_room > 50.0:
                        continue
                
                # Check car-to-car distance (X range)
                if not is_far_enough(loc, car_locs, X_DIST):
                    continue
                
                # Check store front distance
                if not is_far_enough(loc, store_locs, STORE_DIST):
                    continue

                asset = random.choice(cars)
                asset_size = get_obj_size(asset)
                
                # Check dynamic distance from other already placed objaverse objects
                if not is_far_enough_dynamic(loc, asset_size, placed_objects_info):
                    continue

                rel_path = os.path.relpath(asset, VC_SCENE_DIR)
                
                new_objects.append({
                    "category": "vehicle",
                    "location": loc,
                    "heading": p.get('heading', 0),
                    "asset_path": rel_path
                })
                car_locs.append(loc)
                placed_objects_info.append({'loc': loc, 'size': asset_size, 'dims': get_obj_dims(asset)})
                count += 1

        # Group logic
        # 1. Determine number of groups based on available sidewalk points and objects
        # 2. For each group, pick a center location from sidewalk points (far from other groups)
        # 3. Pick 2-4 objects for the group
        # 4. Place objects around the group center with stacking logic
        
        GROUP_DIST = 8.0 # Min distance between group centers
        
        # Collect all valid sidewalk points first
        valid_sidewalk_points = []
        n_sidewalk = min(1000, len(sidewalk_data.get('points', [])))
        all_sidewalk_points = list(sidewalk_data.get('points', []))
        random.shuffle(all_sidewalk_points)
        
        for p in all_sidewalk_points:
            loc = [p['utm_x'], p['utm_y'], p['height']]
            
            # Filter by room distance (50m)
            if room_loc:
                dist_room = math.sqrt((loc[0]-room_loc[0])**2 + (loc[1]-room_loc[1])**2)
                if dist_room > 50.0:
                    continue
            
            # Check store front distance
            if not is_far_enough(loc, store_locs, STORE_DIST):
                continue
            
            # Check traffic light distance
            if not is_far_enough(loc, tl_locs, TL_PROP_DIST):
                continue
                
            # Check car distance
            if not is_far_enough(loc, car_locs, CAR_PROP_DIST):
                continue
                
            valid_sidewalk_points.append({'loc': loc, 'heading': p.get('heading', 0)})
            if len(valid_sidewalk_points) >= n_sidewalk:
                break
        
        print(f"Found {len(valid_sidewalk_points)} valid sidewalk points for grouping")
        
        # Form Groups
        groups = []
        for point in valid_sidewalk_points:
            loc = point['loc']
            # Check distance from existing groups
            if not is_far_enough(loc, [g['center'] for g in groups], GROUP_DIST):
                continue
            
            groups.append({'center': loc, 'heading': point['heading'], 'objects': []})
            
        print(f"Formed {len(groups)} groups")
        
        # Distribute objects to groups
        # Only use semantic groups. Allow reuse with 20m distance constraint.
        
        if not semantic_groups:
            print("Error: No semantic groups available.")
            return []

        # Track locations of each semantic group to enforce 20m separation
        # key: group name, value: list of locations
        semantic_group_history = {sg['name']: [] for sg in semantic_groups}
        
        for group in groups:
            center = group['center']
            base_heading = group['heading']
            
            # Find a valid semantic group
            candidates = list(semantic_groups)
            random.shuffle(candidates)
            
            selected_sg = None
            
            for sg in candidates:
                name = sg['name']
                history = semantic_group_history.get(name, [])
                
                # Check distance constraint (20m from previous instances of SAME group)
                valid = True
                for prev_loc in history:
                    dist = math.sqrt((center[0] - prev_loc[0])**2 + (center[1] - prev_loc[1])**2)
                    if dist < 10.0:
                        valid = False
                        break
                
                if valid:
                    selected_sg = sg
                    break
            
            if not selected_sg:
                print(f"Warning: Could not find a valid semantic group for location {center} (all within 20m of existing instances). Skipping.")
                continue
                
            # Record usage
            semantic_group_history[selected_sg['name']].append(center)
            
            # Resolve objects for this group
            group_objects = []
            for obj_name in selected_sg.get("objects", []):
                # Try exact match first
                if obj_name in filename_to_path:
                    group_objects.append(filename_to_path[obj_name])
                else:
                    # Try case-insensitive match if not found directly
                    pass
            
            if not group_objects:
                print(f"Warning: No objects found for group {selected_sg['name']}. Skipping.")
                continue

            # Local placed objects for this group to handle stacking/offset
            # Coordinates are WORLD coordinates
            group_placed_info = [] 
            
            for asset in group_objects:
                asset_size = get_obj_size(asset)
                asset_dims = get_obj_dims(asset)
                
                # Try to place in group
                # Strategy: 
                # 1. First object at center (or slightly offset)
                # 2. Subsequent objects: try to stack on existing, or place nearby
                
                # Offset from group center (within ~1.5m radius)
                # We try a few random positions around the center
                
                best_loc = None
                
                for attempt in range(10):
                    # Sample offset
                    # First object attempts to be at center (attempt 0)
                    if not group_placed_info and attempt == 0:
                        ox, oy = 0, 0
                    else:
                        r = random.uniform(0, 1.5)
                        theta = random.uniform(0, 2*math.pi)
                        ox = r * math.cos(theta)
                        oy = r * math.sin(theta)
                    
                    candidate_loc = [center[0] + ox, center[1] + oy, center[2]]
                    
                    # Check stacking with THIS GROUP'S objects
                    # (We assume groups are far enough apart that we don't need to check other groups)
                    
                    # Check proximity/stacking
                    nearby_objs = []
                    for obj_info in group_placed_info:
                        dist = math.sqrt((candidate_loc[0] - obj_info['loc'][0])**2 + (candidate_loc[1] - obj_info['loc'][1])**2)
                        # Overlap check?
                        # If dist < (size_a + size_b)/2, they overlap horizontally.
                        # We allow overlap ONLY if stacking.
                        
                        min_dist_touch = (asset_size/2.0 + obj_info['size']/2.0) * 0.8 # slightly permissive
                        if dist < min_dist_touch:
                            nearby_objs.append(obj_info)
                    
                    is_stacked = False
                    if nearby_objs:
                        # Try stack
                        for other_obj in nearby_objs:
                             other_dims = other_obj['dims']
                             other_size = other_obj['size']
                             
                             # Condition: New is Small (<1m), Old is Large (>2m)
                             if asset_size < 1.0 and other_size > 2.0:
                                 # Stack
                                 new_z = other_obj['loc'][2] + other_dims[2]
                                 
                                 # Shift to edge of base object?
                                 # reusing logic
                                 vec = [candidate_loc[0] - other_obj['loc'][0], candidate_loc[1] - other_obj['loc'][1]]
                                 dist_v = math.sqrt(vec[0]**2 + vec[1]**2)
                                 if dist_v < 0.001:
                                     vec = [1, 0]
                                     dist_v = 1
                                 
                                 target_dist = (other_size / 2.0) * 0.6 # slightly inwards
                                 new_x = other_obj['loc'][0] + (vec[0]/dist_v) * target_dist
                                 new_y = other_obj['loc'][1] + (vec[1]/dist_v) * target_dist
                                 
                                 candidate_loc = [new_x, new_y, new_z]
                                 is_stacked = True
                                 break
                        
                        if not is_stacked:
                            # Conflict (overlap but couldn't stack)
                            continue 
                    
                    # If not stacked, check if it's a valid ground placement (no overlap)
                    # We already checked overlap above (nearby_objs). If nearby_objs is not empty and we didn't stack, we 'continue'd.
                    # So here, either nearby_objs is empty, or we stacked.
                    
                    best_loc = candidate_loc
                    break
                
                if best_loc:
                    rel_path = os.path.relpath(asset, VC_SCENE_DIR)
                    new_objects.append({
                        "category": "prop",
                        "location": best_loc,
                        "heading": base_heading, # aligned with sidewalk
                        "asset_path": rel_path
                    })
                    # Update local group info
                    group_placed_info.append({'loc': best_loc, 'size': asset_size, 'dims': asset_dims})
                    
                    # Also add to global placed info for future steps (though we are done with props)
                    placed_objects_info.append({'loc': best_loc, 'size': asset_size, 'dims': asset_dims})

        output_json = os.path.join(VC_SCENE_DIR, "objaverse_objects.json")
        with open(output_json, 'w') as f:
            json.dump({"city": CITY, "objects": new_objects}, f, indent=2)
        
        print(f"Saved {len(new_objects)} objaverse objects to {output_json}")
        return new_objects

    def merge_usds(room_loc):
        print("--- Step 4: Merging USDs ---")
        
        if not os.path.exists(SCENE_USD):
            print(f"Error: Scene USD not found at {SCENE_USD}")
            return

        # Load existing city scene layer
        root_layer = Sdf.Layer.FindOrOpen(SCENE_USD)
        if not root_layer:
            print(f"Error: Could not open {SCENE_USD}")
            return

        # Create a new layer for the merged result (or overwrite if exists)
        if os.path.exists(OUTPUT_MERGED_USD):
            os.remove(OUTPUT_MERGED_USD)
        merged_layer = Sdf.Layer.CreateNew(OUTPUT_MERGED_USD)
        
        # Transfer content from the original layer to the new one
        merged_layer.TransferContent(root_layer)
        
        # Now open the merged layer as a stage for manipulation
        stage = Usd.Stage.Open(merged_layer)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        
        # Ensure /World exists as root
        world_path = Sdf.Path("/World")
        if not stage.GetPrimAtPath(world_path):
            stage.DefinePrim(world_path, "Xform")
        
        stage.SetDefaultPrim(stage.GetPrimAtPath(world_path))

        # Remove old road objects and objaverse xform
        for path in ["/World/RoadObjects", "/World/Objaverse"]:
             if stage.GetPrimAtPath(path):
                 print(f"Removing old {path}...")
                 stage.RemovePrim(path)

        # Define store and city parents
        store_parent_path = world_path.AppendChild("store")
        city_parent_path = world_path.AppendChild("city")
        prim = stage.DefinePrim(store_parent_path, "Xform")
        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetScale(Gf.Vec3f(0.01, 0.01, 0.01))
        prim = stage.DefinePrim(city_parent_path, "Xform")
        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetScale(Gf.Vec3f(0.01, 0.01, 0.01))

        # Collect prims to move
        to_move = []
        path_map = {}
        
        pseudo_root = stage.GetPseudoRoot()
        for prim in pseudo_root.GetChildren():
            p_path = prim.GetPath()
            if p_path == world_path:
                for child in prim.GetChildren():
                    c_name = child.GetName()
                    if c_name in ["store", "city", "RoadObjects", "Objaverse", "start_result_navigation"]:
                        continue
                    
                    target_parent = city_parent_path
                    for s_type in STORE_TYPES:
                        if s_type in c_name.lower():
                            target_parent = store_parent_path
                            break
                    new_p = target_parent.AppendChild(c_name)
                    to_move.append((child.GetPath(), new_p))
                    path_map[child.GetPath()] = new_p
            else:
                name = prim.GetName()
                if name == "start_result_navigation":
                    continue
                target_parent = city_parent_path
                for s_type in STORE_TYPES:
                    if s_type in name.lower():
                        target_parent = store_parent_path
                        break
                new_p = target_parent.AppendChild(name)
                to_move.append((p_path, new_p))
                path_map[p_path] = new_p

        # Apply all moves
        if to_move:
            edit = Sdf.BatchNamespaceEdit()
            for old_p, new_p in to_move:
                if not stage.GetPrimAtPath(new_p):
                    edit.Add(old_p, new_p)
                else:
                    print(f"Warning: Destination path {new_p} already exists. Skipping move for {old_p}")
            
            if not merged_layer.Apply(edit):
                print("Warning: Failed to apply some namespace edits.")

        # FIX RELATIONSHIPS AND MATERIAL BINDINGS
        print("Fixing material bindings and relationships...")
        sorted_old_paths = sorted(path_map.keys(), key=lambda p: len(p.pathString), reverse=True)
        
        fix_count = 0
        def fix_path(path):
            nonlocal fix_count
            if not isinstance(path, Sdf.Path):
                return path
            for old_p in sorted_old_paths:
                if path == old_p or path.HasPrefix(old_p):
                    new_path = path.ReplacePrefix(old_p, path_map[old_p])
                    if new_path != path:
                        fix_count += 1
                    return new_path
            return path

        for prim in stage.TraverseAll():
            # Fix Internal References
            prim_spec = merged_layer.GetPrimAtPath(prim.GetPath())
            if prim_spec:
                # Fix references
                refs = prim_spec.referenceList.prependedItems
                if refs:
                    new_refs = []
                    changed = False
                    for ref in refs:
                        if not ref.assetPath: # Internal reference
                            new_target = fix_path(ref.primPath)
                            if new_target != ref.primPath:
                                new_refs.append(Sdf.Reference(ref.assetPath, new_target, ref.customData))
                                changed = True
                            else:
                                new_refs.append(ref)
                        else:
                            new_refs.append(ref)
                    if changed:
                        prim_spec.referenceList.prependedItems = new_refs

            # Fix Properties
            for prop in prim.GetProperties():
                if isinstance(prop, Usd.Relationship):
                    targets = prop.GetTargets()
                    if targets:
                        new_targets = [fix_path(t) for t in targets]
                        if new_targets != targets:
                            prop.SetTargets(new_targets)
                
                elif isinstance(prop, Usd.Attribute):
                    # Fix attribute value if it's a path or asset path
                    try:
                        val = prop.Get()
                        if isinstance(val, Sdf.Path):
                            new_val = fix_path(val)
                            if new_val != val:
                                prop.Set(new_val)
                        elif isinstance(val, list) and val and isinstance(val[0], Sdf.Path):
                            new_val = [fix_path(v) for v in val]
                            if new_val != val:
                                prop.Set(new_val)
                        elif isinstance(val, Sdf.AssetPath):
                            # Usually asset paths are relative, but check if they are absolute prim paths
                            p_str = val.path
                            if p_str.startswith('/'):
                                new_p = fix_path(Sdf.Path(p_str))
                                if new_p.pathString != p_str:
                                    prop.Set(Sdf.AssetPath(new_p.pathString))
                    except:
                        pass
                    
                    # Fix attribute connections (Shader inputs/outputs)
                    try:
                        conns = prop.GetConnections()
                        if conns:
                            new_conns = [fix_path(c) for c in conns]
                            if new_conns != conns:
                                prop.SetConnections(new_conns)
                    except:
                        pass
        
        print(f"Fixed {fix_count} paths in relationships and attributes.")

        # Helper to add objects
        def add_objects(object_list, parent_path, source="unknown"):
            for i, obj in enumerate(object_list):
                asset_path = obj.get('asset_path')
                if not asset_path:
                    continue
                if "ViCo" in asset_path or ":" in asset_path:
                    filename = os.path.basename(asset_path.replace("\\", "/"))
                    asset_path = os.path.join(SCENE_DATA_DIR, "vc_plus/road_objects", filename)
                    asset_path = os.path.relpath(asset_path, VC_SCENE_DIR)
                
                filename = os.path.basename(asset_path).split(".")[0]
                prim_path = f"{parent_path}/{filename}_{i}"
                prim = stage.DefinePrim(prim_path, "Xform")
                xform = UsdGeom.XformCommonAPI(prim)
                
                loc = obj.get('location') or obj.get('final_location')
                if loc:
                    xform.SetTranslate(Gf.Vec3d(loc[0], loc[1], loc[2]))
                
                rot_deg = Gf.Vec3f(0, 0, 0)
                if 'heading' in obj:
                    rot_deg = Gf.Vec3f(0, 0, math.degrees(obj['heading']))
                elif 'rotation' in obj and 'euler_xyz' in obj['rotation']:
                    euler = obj['rotation']['euler_xyz']
                    rot_deg = Gf.Vec3f(math.degrees(euler[0]), math.degrees(euler[1]), math.degrees(euler[2]))
                xform.SetRotate(rot_deg)
                
                if 'scale' in obj:
                    s = obj['scale']
                    xform.SetScale(Gf.Vec3f(s[0], s[1], s[2]))
                
                if os.path.exists(os.path.join(VC_SCENE_DIR, asset_path)):
                    prim.GetReferences().AddReference(asset_path)
                else:
                    print(f"Warning: Asset not found at {asset_path}")

        # Helper to add objects with category-specific offsets and rotation logic
        def add_road_objects(object_list, parent_path):
            # Unscaled heights measured from assets in AMSTERDAM
            ROAD_ASSET_HEIGHTS = {
                "nyc_traffic_light.glb": 9.507,
                "Danger_road_sign.glb": 3.045,
                "Only_straight_road_sign.glb": 3.019,
                "Bus_stop_sign.glb": 2.383,
                "Speed_50_road_sign.glb": 2.946
            }
            
            placed_tl_locs = []
            TL_MIN_DIST = 6.0

            for i, obj in enumerate(object_list):
                is_tl = obj.get('category') in ['crossing', 'traffic_signals']
                
                # Use final_location if available
                final_loc = obj.get('final_location')
                loc = final_loc or obj.get('location')
                
                # Traffic light distance check
                if is_tl and loc:
                    if not is_far_enough(loc, placed_tl_locs, TL_MIN_DIST):
                        continue
                    placed_tl_locs.append(loc)

                asset_path = obj.get('asset_path')
                if not asset_path:
                    continue
                if "ViCo" in asset_path or ":" in asset_path:
                    filename = os.path.basename(asset_path.replace("\\", "/"))
                    asset_path = os.path.join(SCENE_DATA_DIR, "vc_plus/road_objects", filename)
                    asset_path = os.path.relpath(asset_path, VC_SCENE_DIR)
                
                filename = os.path.basename(asset_path).split(".")[0]
                prim_path = f"{parent_path}/{filename}_{i}"
                prim = stage.DefinePrim(prim_path, "Xform")
                xform = UsdGeom.XformCommonAPI(prim)
                
                # Use final_location if available
                final_loc = obj.get('final_location')
                if final_loc:
                    # User reported Z should be original value - 2
                    xform.SetTranslate(Gf.Vec3d(final_loc[0], final_loc[1], final_loc[2] - 2.0))
                else:
                    loc = obj.get('location')
                    if loc:
                        # If final_location not present, location is already the ground height
                        xform.SetTranslate(Gf.Vec3d(loc[0], loc[1], loc[2]))
                
                # Handle rotation based on user's manual fix: X=+90, Z=original+90+180
                rot_deg = Gf.Vec3f(90, 0, 0)
                
                if 'road_angle_radians' in obj:
                    angle_deg = math.degrees(obj['road_angle_radians']) + 90 + 180
                    rot_deg = Gf.Vec3f(90, 0, angle_deg)
                elif 'rotation' in obj and 'euler_xyz' in obj['rotation']:
                    euler = obj['rotation']['euler_xyz']
                    # Note: euler[2] is the Z rotation in radians
                    rot_deg = Gf.Vec3f(90, 0, math.degrees(euler[2]) + 180)
                
                # XformCommonAPI default convention is XYZ
                xform.SetRotate(rot_deg)
                
                # Scale based on target heights: 7m for traffic signals, 2.5m for others
                asset_filename = os.path.basename(asset_path)
                unscaled_h = ROAD_ASSET_HEIGHTS.get(asset_filename, 3.0)
                
                target_h = 7.0 if obj.get('category') in ['crossing', 'traffic_signals'] else 2.5
                s = target_h / unscaled_h
                xform.SetScale(Gf.Vec3f(s, s, s))
                
                if os.path.exists(os.path.join(VC_SCENE_DIR, asset_path)):
                    prim.GetReferences().AddReference(asset_path)
                else:
                    print(f"Warning: Road asset not found at {asset_path}")

        # Add Road Objects
        if os.path.exists(ROAD_OBJECTS):
            with open(ROAD_OBJECTS) as f:
                data = json.load(f)
                if 'objects' in data:
                    filtered_objects = []
                    for obj in data['objects']:
                        final_loc = obj.get('final_location')
                        loc = final_loc or obj.get('location')
                        if loc and room_loc:
                            dist_room = math.sqrt((loc[0]-room_loc[0])**2 + (loc[1]-room_loc[1])**2)
                            if dist_room > 100.0:
                                continue
                        filtered_objects.append(obj)
                    add_road_objects(filtered_objects, "/World/RoadObjects")

        # Add Objaverse Objects
        objaverse_json = os.path.join(VC_SCENE_DIR, "objaverse_objects.json")
        if os.path.exists(objaverse_json):
            with open(objaverse_json) as f:
                data = json.load(f)
                if 'objects' in data:
                    add_objects(data['objects'], "/World/Objaverse", "objaverse")

        # Remove DomeLights
        print("Removing DomeLights...")
        prims_to_remove = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "DomeLight":
                prims_to_remove.append(prim.GetPath())
        
        for path in prims_to_remove:
            print(f"Removing DomeLight at: {path}")
            stage.RemovePrim(path)

        # Add Test Cube at room location
        # if room_loc:
        #     print(f"Adding testcube at {room_loc}...")
        #     cube_path = "/World/testcube"
        #     # Ensure unique name if exists
        #     if stage.GetPrimAtPath(cube_path):
        #         stage.RemovePrim(cube_path)
                
        #     cube_prim = stage.DefinePrim(cube_path, "Cube")
        #     # Set size and color
        #     UsdGeom.Cube(cube_prim).GetSizeAttr().Set(2.0)
            
        #     xform = UsdGeom.XformCommonAPI(cube_prim)
        #     xform.SetTranslate(Gf.Vec3d(room_loc[0], room_loc[1], room_loc[2]))
            
        #     # Add a red color for visibility
        #     color_attr = UsdGeom.Gprim(cube_prim).GetDisplayColorAttr()
        #     color_attr.Set([Gf.Vec3f(1.0, 0.0, 0.0)])


        # Save changes to the layer
        merged_layer.Save()
        print(f"Saved merged USD to {OUTPUT_MERGED_USD}")

    # if os.path.exists(OUTPUT_MERGED_USD):
    #     return
    # if CITY != "AMSTERDAM":
    #     return
    print(f"Processing {CITY}")

    # Get room location from SCENE_USD
    room_loc = None
    if os.path.exists(SCENE_USD):
        # Helper to compute center from BBoxCache (handles descendants)
        def get_bbox_center(prim):
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
            bound = bbox_cache.ComputeWorldBound(prim)
            range3d = bound.ComputeAlignedBox()
            min_point = range3d.GetMin()
            max_point = range3d.GetMax()
            return (min_point + max_point) / 2.0

        print(f"Loading {SCENE_USD} to find room location...")
        
        # Silence USD warnings by redirecting stderr
        try:
            with SuppressFD():
                # Open stage
                temp_stage = Usd.Stage.Open(SCENE_USD)
                room_prim = temp_stage.GetPrimAtPath("/World/start_result_navigation")
                
                if room_prim:
                    # Use BBoxCache to get the center of the room and its children
                    center_world = get_bbox_center(room_prim)
                    room_loc = [center_world[0], center_world[1], center_world[2]]
                    print(f"Found room center at: {room_loc} (using BBoxCache)")
                else:
                    print("Warning: room not found.")
                
        except Exception as e:
            print(f"Error finding room location: {e}")

    if room_loc is None:
        print("Warning: Room location not found")
        return

    generate_objaverse_json(room_loc)
    merge_usds(room_loc)

if __name__ == "__main__":
    cities = []
    if args.city == "all":
        # iterate over all cities in the vc_plus folder
        for city in os.listdir(os.path.join(args.scene_folder, "vc_apr14")):
            if os.path.isdir(os.path.join(args.scene_folder, "vc_apr14", city)) and city != "road_objects":
                cities.append(city)
        print("Processing cities:", cities)
        input("Press Enter to continue...")
    else:
        cities.append(args.city)
    convert_models()
    for city in cities:
        main(city)
