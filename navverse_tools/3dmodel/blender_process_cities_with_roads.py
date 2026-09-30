import bpy
import os
import json
import numpy as np
from tqdm import tqdm
import sys, contextlib
import xml.etree.ElementTree as ET
import random
import glob
import math
import mathutils
import time
# ========== Config ==========
cities = ["YALE", "ZURICH"]

# Test with single city
# cities = ["NY"]#"NY", "AMSTERDAM", "AUSTIN", "BALTIMORE","BARCELONA",  "BERLIN", "BOISE", "BELGRADE", "BOSTON", "BRATISLAVA", "BRUSSELS", "BUDAPEST", "CALGARY", "CHARLOTTE", "CHICAGO", "CHRISTCHURCH", "COLUMBUS", "DENVER", "DETROIT", "EL_PASO", "FLORENCE", "FORT_WORTH", "FRANKFURT", "HAMBURG", "HARVARD"(bug), "KANSAS_CITY", "LASVEGAS", "LONDON", "LONGISLAND", "MADISON", "MADRID", "MADRID2", "MILAN", "MINNEAPOLIS", "MIT", "MONTREAL", "ORLANDO", "PARIS", "PHILADELPHIA", "PORTLAND", "ROME", "SANFRANCISCO", "SANFRANCISCO2","SILICONVALLEY", "STANFORD"(bug),  "SYDNEY", "TORONTO", "UCLA", "UMASS", "WHITEHOUSE", 

DATA_DIR = "D:/Desktop/ViCo"
OUTPUT_BASE_DIR = f"{DATA_DIR}/generated"  # Base directory for exported USD
HDRI_PATH = f"{DATA_DIR}/qwantani_sunrise_puresky_2k.exr"  # HDRI file path
OBJECTS_BASE_DIR = f"{DATA_DIR}/objects/outdoor_objects/retrieved"

# ==========================

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

def clear_scene():
    """Clear current scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear mesh data blocks
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)

def install_psutil_if_needed():
    """Install psutil if not available"""
    try:
        import psutil
        return True
    except ImportError:
        print("[INFO] psutil not found, attempting to install...")
        try:
            import subprocess
            import sys
            
            # Get Blender's Python executable
            python_exe = sys.executable
            print(f"[INFO] Using Python: {python_exe}")
            
            # Install psutil
            result = subprocess.run([
                python_exe, "-m", "pip", "install", "psutil"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("[SUCCESS] psutil installed successfully!")
                return True
            else:
                print(f"[ERROR] Failed to install psutil: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Could not install psutil: {e}")
            return False

def get_memory_usage():
    """Get current memory usage information"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # Try to install psutil if not available
        if install_psutil_if_needed():
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            except ImportError:
                memory_mb = 0
        else:
            memory_mb = 0
    
    # Also get Blender scene statistics
    scene_stats = {
        'objects': len(bpy.data.objects),
        'meshes': len(bpy.data.meshes),
        'materials': len(bpy.data.materials),
        'textures': len(bpy.data.textures),
        'images': len(bpy.data.images),
        'collections': len(bpy.data.collections)
    }
    
    return memory_mb, scene_stats

def print_memory_stats(stage, city_name=""):
    """Print memory usage statistics"""
    try:
        memory_mb, scene_stats = get_memory_usage()
        print(f"[MEMORY] {stage} - {city_name}: {memory_mb:.1f}MB | "
              f"Objects: {scene_stats['objects']}, Meshes: {scene_stats['meshes']}, "
              f"Materials: {scene_stats['materials']}, Images: {scene_stats['images']}")
    except Exception as e:
        print(f"[MEMORY] {stage} - {city_name}: Unable to get memory stats ({e})")

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

# ========== Road Processing Functions ==========

def load_osm_with_height(osm_file_path):
    """Load OSM data with pre-processed height and coordinate information"""
    print(f"Loading OSM with height data from: {osm_file_path}")
    
    tree = ET.parse(osm_file_path)
    root = tree.getroot()
    
    nodes = {}
    ways = []
    highway_nodes = []  # Store nodes with highway tags
    
    # Parse nodes with height and coordinate data
    for element in root:
        if element.tag == 'node':
            node_data = {
                'id': element.get('id'),
                'lat': float(element.get('lat')) if element.get('lat') else 0,
                'lon': float(element.get('lon')) if element.get('lon') else 0,
                'height': 0.0,
                'utm_x': 0.0,
                'utm_y': 0.0,
                'highway': None,
                'tags': {}
            }
            
            # Extract all tags including height, coordinates, and highway
            for tag in element.findall('tag'):
                k = tag.get('k')
                v = tag.get('v')
                node_data['tags'][k] = v
                
                if k == 'height':
                    node_data['height'] = float(v)
                elif k == 'utm_x':
                    node_data['utm_x'] = float(v)
                elif k == 'utm_y':
                    node_data['utm_y'] = float(v)
                elif k == 'highway':
                    node_data['highway'] = v
            
            nodes[node_data['id']] = node_data
            
            # Store nodes with highway tags for object placement
            if node_data['highway'] in ['crossing', 'traffic_signals', 'bus_stop', 'stop']:
                highway_nodes.append(node_data)
    
    # Parse ways
    for element in root:
        if element.tag == 'way':
            way_data = {
                'id': element.get('id'),
                'nodes': [nd.get('ref') for nd in element.findall('nd')],
                'tags': {}
            }
            
            for tag in element.findall('tag'):
                way_data['tags'][tag.get('k')] = tag.get('v')
            
            ways.append(way_data)
    
    print(f"Loaded {len(nodes)} nodes and {len(ways)} ways with height data")
    print(f"Found {len(highway_nodes)} nodes with highway tags for object placement")
    return nodes, ways, highway_nodes

def get_available_objects():
    """Get available GLB objects for different highway types"""
    objects_dir = f"{DATA_DIR}/objects/outdoor_objects/retrieved/mesh_renamed"
    
    available_objects = {
        'crossing': [],
        'traffic_signals': [],  # Intentionally unused for now
        'bus_stop': [],          # Use a pool: bus_stop_sign + *_road_sign.glb
        'stop': None
    }
    
    if not os.path.exists(objects_dir):
        print(f"Warning: Objects directory not found: {objects_dir}")
        return available_objects
    
    # Get traffic light files for crossing
    traffic_light_files = glob.glob(os.path.join(objects_dir, "*_traffic_light.glb"))
    available_objects['crossing'] = traffic_light_files
    
    # Get road sign files (will be used under bus_stop selection)
    road_sign_files = glob.glob(os.path.join(objects_dir, "*_road_sign.glb"))
    
    # Get specific bus stop sign and combine with road_sign_files as a pool
    bus_stop_file = os.path.join(objects_dir, "Bus_stop_sign.glb")
    if os.path.exists(bus_stop_file):
        available_objects['bus_stop'].append(bus_stop_file)
    # Extend bus_stop pool with road_sign files
    available_objects['bus_stop'].extend(road_sign_files)
    
    stop_sign_file = os.path.join(objects_dir, "stop_sign.glb")
    if os.path.exists(stop_sign_file):
        available_objects['stop'] = stop_sign_file
    
    return available_objects

def load_object_from_glb(file_path):
    """Load a GLB object and return all imported objects"""
    try:
        # Import the GLB file
        bpy.ops.import_scene.gltf(filepath=file_path)
        
        # Get ALL imported objects (mesh, lights, cameras, etc.)
        imported_objects = list(bpy.context.selected_objects)
        
        if imported_objects:
            print(f"Imported {len(imported_objects)} objects from {file_path}")
            return imported_objects  # Return all objects
        else:
            print(f"Warning: No objects found in {file_path}")
            return []
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def calculate_road_direction_at_node(node, nodes, ways):
    """Calculate the road direction at a given node"""
    # Find ways that contain this node
    connected_ways = []
    for way in ways:
        if node['id'] in way.get('nodes', []):
            connected_ways.append(way)
    
    if not connected_ways:
        return 0  # Default direction if no connected ways
    
    # Get the first connected way and find the direction
    way = connected_ways[0]
    way_nodes = way.get('nodes', [])
    
    try:
        node_index = way_nodes.index(node['id'])
        
        # Get previous and next nodes to determine direction
        prev_node = None
        next_node = None
        
        if node_index > 0:
            prev_node_id = way_nodes[node_index - 1]
            if prev_node_id in nodes:
                prev_node = nodes[prev_node_id]
        
        if node_index < len(way_nodes) - 1:
            next_node_id = way_nodes[node_index + 1]
            if next_node_id in nodes:
                next_node = nodes[next_node_id]
        
        # Calculate direction vector
        if prev_node and next_node:
            # Use both directions to get average
            dir_x = next_node['utm_x'] - prev_node['utm_x']
            dir_y = next_node['utm_y'] - prev_node['utm_y']
        elif next_node:
            # Use only forward direction
            dir_x = next_node['utm_x'] - node['utm_x']
            dir_y = next_node['utm_y'] - node['utm_y']
        elif prev_node:
            # Use only backward direction
            dir_x = node['utm_x'] - prev_node['utm_x']
            dir_y = node['utm_y'] - prev_node['utm_y']
        else:
            return 0  # No direction available
        
        # Calculate angle in radians
        angle = math.atan2(dir_y, dir_x)
        return angle
        
    except (ValueError, IndexError):
        return 0  # Default direction if calculation fails


def get_road_width_at_node(node, ways):
    """Get road width for the road containing this node"""
    # Find ways that contain this node
    for way in ways:
        if node['id'] in way.get('nodes', []):
            tags = way.get('tags', {})
            # Try to get width from OSM tags
            if 'width' in tags:
                try:
                    width_str = tags['width']
                    # Handle different width formats (meters, feet, etc.)
                    if 'm' in width_str:
                        return float(width_str.replace('m', '').strip())
                    elif 'ft' in width_str or "'" in width_str:
                        return float(width_str.replace('ft', '').replace("'", '').strip()) * 0.3048  # Convert feet to meters
                    else:
                        return float(width_str)
                except:
                    pass
            
            # Fallback: estimate width based on highway type
            highway_type = tags.get('highway', 'residential')
            lane_width = 3.5  # Standard lane width
            
            if highway_type in ['motorway', 'trunk']:
                lane_width = 3.75
            elif highway_type in ['residential', 'service']:
                lane_width = 3.0
            elif highway_type in ['footway', 'cycleway']:
                lane_width = 2.0
            
            lanes = 1
            if 'lanes' in tags:
                try:
                    lanes = int(tags['lanes'])
                except:
                    lanes = 1
            
            # Calculate estimated width
            estimated_width = lanes * lane_width
            
            # Add margins based on road type
            if highway_type in ['motorway', 'trunk', 'primary']:
                estimated_width += 2.0
            elif highway_type in ['secondary', 'tertiary']:
                estimated_width += 1.0
            
            return estimated_width
    
    # Default width if no road found
    return 6.0  # Default 6 meters


def place_objects_at_node(objects, x, y, z, highway_type, node, nodes, ways):
    """Place all objects from a GLB import at a node location with appropriate rotation and scale"""
    if not objects:
        return []
    
    # Calculate road direction
    road_angle = calculate_road_direction_at_node(node, nodes, ways)
    
    # Debug: Print input coordinates
    print(f"DEBUG: Placing {highway_type} at node coordinates: x={x}, y={y}, z={z}, road_angle={math.degrees(road_angle):.1f}°")
    
    # Calculate rotation and scale based on highway type
    rotation_quat = mathutils.Quaternion((1, 0, 0))  # Identity quaternion
    scale = (1.0, 1.0, 1.0)
    offset_x, offset_y = 0, 0  # Offset for positioning
    
    if highway_type == 'crossing':
        # Traffic lights - face opposite to traffic flow (like other objects)
        # Face opposite to traffic flow (180 degrees from road direction)
        rotation_angle = road_angle + math.pi
        # Create Z-axis rotation quaternion for world coordinate system
        scale = (0.33, 0.33, 0.33)
        # Get road width and calculate offset
        road_width = get_road_width_at_node(node, ways)
        # Offset distance = road width / 2 + 1.5m (move to roadside)
        offset_distance = (road_width / 2) + 1.5
        offset_x = -math.sin(road_angle) * offset_distance
        offset_y = math.cos(road_angle) * offset_distance
        print(f"DEBUG: {highway_type} offset calculation: road_width={road_width:.1f}m, offset_distance={offset_distance:.1f}m, offset=({offset_x:.2f}, {offset_y:.2f})")
        
    elif highway_type == 'traffic_signals':
        # Road signs - face road direction and adjust position based on road width
        # Face opposite to traffic flow (180 degrees from road direction)
        rotation_angle = road_angle + math.pi
        rotation_quat = mathutils.Quaternion((0, 0, 1), rotation_angle)  # Z-axis rotation
        scale = (2.0, 2.0, 2.0)
        # No offset - stay at original position
    elif highway_type == 'bus_stop':
        # Bus stop signs - face road direction, no position adjustment
        # Face opposite to traffic flow (180 degrees from road direction)
        rotation_angle = road_angle + math.pi
        rotation_quat = mathutils.Quaternion((0, 0, 1), rotation_angle)  # Z-axis rotation
        scale = (2.0, 2.0, 2.0)
        # No offset - stay at original position
    elif highway_type == 'stop':
        # Stop signs - face road direction, no position adjustment
        # Face opposite to traffic flow (180 degrees from road direction)
        rotation_angle = road_angle + math.pi
        rotation_quat = mathutils.Quaternion((0, 0, 1), rotation_angle)  # Z-axis rotation
        scale = (2.0, 2.0, 2.0)
        # No offset - stay at original position
    
    # Find root objects (objects without parents)
    root_objects = []
    child_objects = []
    
    for obj in objects:
        if obj.parent is None:
            root_objects.append(obj)
        else:
            child_objects.append(obj)
    # Create copies of objects to avoid modifying cached originals
    placed_objects = []
    for obj in root_objects:
        # Check if object is valid
        if obj is None:
            print(f"Warning: Skipping None object in root_objects")
            continue
            
        try:
            # Position the object directly (no copying needed since we load fresh each time)
            final_x = x + offset_x
            final_y = y + offset_y
            final_z = z + 2  # 2 meters above the node height
            
            obj.location = (final_x, final_y, final_z)
            
            # Apply rotation and scale
            if highway_type == 'crossing':
                # For crossing objects, use Euler angles instead of quaternions
                # First change rotation mode to Euler, then apply Z-axis rotation
                obj.rotation_mode = 'XYZ'  # Set rotation mode to Euler XYZ
                obj.rotation_euler = (-math.pi/2, 0, rotation_angle)  # Z-axis rotation in Euler
                print(f"DEBUG: {obj.name} - Set rotation mode to Euler XYZ, Z={math.degrees(rotation_angle):.1f}°")
            else:
                # For other objects, use the calculated rotation
                obj.rotation_quaternion = rotation_quat
            obj.scale = scale
            
            # Debug: Verify position
            actual_location = obj.location
            print(f"DEBUG: Object {obj.name} placed at ({actual_location.x:.2f}, {actual_location.y:.2f}, {actual_location.z:.2f})")
            
            placed_objects.append(obj)
            
        except Exception as e:
            print(f"Warning: Failed to process object {obj.name if obj else 'Unknown'}: {e}")
            continue
    
    # Handle child objects - duplicate them and maintain parent relationships
    for child_obj in child_objects:
        # Check if child object is valid
        if child_obj is None:
            print(f"Warning: Skipping None child object")
            continue
            
        try:
            # Child objects are already in the scene from the GLB import, just add to placed list
            placed_objects.append(child_obj)
            
        except Exception as e:
            print(f"Warning: Failed to process child object {child_obj.name if child_obj else 'Unknown'}: {e}")
            continue
    
    # Debug output
    if len(placed_objects) > 0:
        main_obj = placed_objects[0]
        if highway_type == 'crossing':
            road_width = get_road_width_at_node(node, ways)
            print(f"Placed {len(placed_objects)} objects for {highway_type} at ({x:.2f}, {y:.2f}, {z:.2f}) with road width {road_width:.1f}m, road angle {math.degrees(road_angle):.1f}° -> ({main_obj.location.x:.2f}, {main_obj.location.y:.2f}, {main_obj.location.z:.2f})")
        else:
            print(f"Placed {len(placed_objects)} objects for {highway_type} at ({x:.2f}, {y:.2f}, {z:.2f}) with road angle {math.degrees(road_angle):.1f}° -> ({main_obj.location.x:.2f}, {main_obj.location.y:.2f}, {main_obj.location.z:.2f})")
    else:
        print(f"Warning: No objects were placed for {highway_type} at ({x:.2f}, {y:.2f}, {z:.2f})")
    
    return placed_objects

def load_object_from_glb_with_cache(file_path, glb_cache, timing_stats):
    """Load a GLB object with caching support"""
    try:
        # Check GLB cache first
        if file_path in glb_cache:
            # Cache hit - create objects from cached data
            step_start = time.time()
            cache_data = glb_cache[file_path]
            imported_objs = []
            
            # Create new objects from cache
            objects_to_link = []
            for cached_obj_data in cache_data:
                if "mesh" in cached_obj_data:  # Ensure cache has valid mesh data
                    # Create new object
                    new_obj = bpy.data.objects.new(cached_obj_data["name"], cached_obj_data["mesh"])
                    objects_to_link.append(new_obj)
                    imported_objs.append(new_obj)
                else:  # Object without mesh data
                    # Create new object
                    new_obj = bpy.data.objects.new(cached_obj_data["name"], None)
                    objects_to_link.append(new_obj)
                    imported_objs.append(new_obj)
            
            # Batch link objects to scene (faster than individual linking)
            for obj in objects_to_link:
                bpy.context.collection.objects.link(obj)
            
            timing_stats['glb_cache_hit'] += time.time() - step_start
            return imported_objs
            
        else:
            # Cache miss - import GLB file and save to cache
            step_start = time.time()
            bpy.ops.import_scene.gltf(filepath=file_path)
            imported_objs = bpy.context.selected_objects.copy()
            
            # Save to cache
            cache_data = []
            skipped_objects = []
            for obj in imported_objs:
                if obj.data is not None:  # Ensure object has mesh data
                    cache_data.append({"name": obj.name, "mesh": obj.data.copy()})
                else:
                    cache_data.append({"name": obj.name})
                    skipped_objects.append(obj.name)
            
            if skipped_objects:
                print(f"[WARNING] GLB file {os.path.basename(file_path)} has {len(skipped_objects)} objects without mesh data: {skipped_objects}")
            
            glb_cache[file_path] = cache_data
            timing_stats['glb_import'] += time.time() - step_start
            
            return imported_objs
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def process_highway_nodes(highway_nodes, city_name, nodes, ways):
    """Process highway nodes and place appropriate 3D objects with caching"""
    import time
    
    print(f"Processing {len(highway_nodes)} highway nodes for {city_name}")
    
    # Timing statistics
    timing_stats = {
        'get_available_objects': 0,
        'crossing_processing': 0,
        'bus_stop_processing': 0,
        'stop_processing': 0,
        'glb_cache_hit': 0,
        'glb_import': 0,
        'total_processing': 0
    }
    
    start_time = time.time()
    
    # Get available objects
    step_start = time.time()
    available_objects = get_available_objects()
    timing_stats['get_available_objects'] = time.time() - step_start
    print(f"[TIMING] get_available_objects: {timing_stats['get_available_objects']:.3f}s")
    
    # GLB file cache - store the object data of the imported GLB files
    glb_cache = {}
    
    # Statistics
    objects_placed = {
        'crossing': 0,
        'traffic_signals': 0,
        'bus_stop': 0,
        'stop': 0
    }
    
    # Process each node
    crossing_time = 0
    bus_stop_time = 0
    stop_time = 0
    
    for node in tqdm(highway_nodes, desc="Placing highway objects"):
        highway_type = node['highway']
        
        if highway_type == 'crossing' and available_objects['crossing']:
            # Randomly select a traffic light
            step_start = time.time()
            traffic_light_file = random.choice(available_objects['crossing'])
            objects = load_object_from_glb_with_cache(traffic_light_file, glb_cache, timing_stats)
            print("objects",len(objects))
            if objects:
                place_objects_at_node(objects, node['utm_x'], node['utm_y'], node['height'], highway_type, node, nodes, ways)
                objects_placed['crossing'] += 1
            crossing_time += time.time() - step_start
                
        # elif highway_type == 'traffic_signals' and available_objects['traffic_signals']:
        #     # Randomly select a road sign
        #     road_sign_file = random.choice(available_objects['traffic_signals'])
        #     objects = load_object_from_glb_with_cache(road_sign_file, glb_cache, timing_stats)
        #     if objects:
        #         place_objects_at_node(objects, node['utm_x'], node['utm_y'], node['height'], highway_type, node, nodes, ways)
        #         objects_placed['traffic_signals'] += 1
        #         
        elif highway_type == 'bus_stop' and available_objects['bus_stop']:
            # Randomly choose between bus_stop_sign and *_road_sign.glb files
            step_start = time.time()
            chosen_file = random.choice(available_objects['bus_stop'])
            objects = load_object_from_glb_with_cache(chosen_file, glb_cache, timing_stats)
            if objects:
                place_objects_at_node(objects, node['utm_x'], node['utm_y'], node['height'], highway_type, node, nodes, ways)
                objects_placed['bus_stop'] += 1
            bus_stop_time += time.time() - step_start
                
        elif highway_type == 'stop' and available_objects['stop']:
            # Use stop sign
            step_start = time.time()
            objects = load_object_from_glb_with_cache(available_objects['stop'], glb_cache, timing_stats)
            if objects:
                place_objects_at_node(objects, node['utm_x'], node['utm_y'], node['height'], highway_type, node, nodes, ways)
                objects_placed['stop'] += 1
            stop_time += time.time() - step_start
    
    timing_stats['crossing_processing'] = crossing_time
    timing_stats['bus_stop_processing'] = bus_stop_time
    timing_stats['stop_processing'] = stop_time
    timing_stats['total_processing'] = time.time() - start_time
    
    # Calculate cache efficiency
    cache_hits = timing_stats['glb_cache_hit']
    cache_misses = timing_stats['glb_import']
    total_cache_operations = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / total_cache_operations * 100) if total_cache_operations > 0 else 0
    
    # Print timing statistics
    print(f"\n[TIMING] Highway objects processing timing for {city_name}:")
    print(f"  get_available_objects: {timing_stats['get_available_objects']:.3f}s")
    print(f"  glb_cache_hit: {timing_stats['glb_cache_hit']:.3f}s")
    print(f"  glb_import: {timing_stats['glb_import']:.3f}s")
    print(f"  cache_hit_rate: {cache_hit_rate:.1f}% ({cache_hits:.3f}s hits / {total_cache_operations:.3f}s total)")
    print(f"  crossing_processing: {timing_stats['crossing_processing']:.3f}s ({objects_placed['crossing']} objects)")
    print(f"  bus_stop_processing: {timing_stats['bus_stop_processing']:.3f}s ({objects_placed['bus_stop']} objects)")
    print(f"  stop_processing: {timing_stats['stop_processing']:.3f}s ({objects_placed['stop']} objects)")
    print(f"  total_processing: {timing_stats['total_processing']:.3f}s")
    
    # Print placement statistics
    print(f"\nHighway objects placed for {city_name}:")
    total_placed = 0
    for obj_type, count in objects_placed.items():
        if count > 0:
            print(f"  {obj_type}: {count}")
        total_placed += count
    
    print(f"  Total objects placed: {total_placed}")
    
    return objects_placed, timing_stats

# ========== Original Functions from blender_process_cities.py ==========


def import_buildings_from_directory(scene_dir):
    """Import all GLB files in the buildings directory"""
    import time
    
    buildings_dir = os.path.join(scene_dir, "buildings")
    
    # Timing statistics
    timing_stats = {
        'directory_check': 0,
        'file_discovery': 0,
        'glb_import': 0,
        'object_naming': 0,
        'object_deselection': 0,
        'error_logging': 0,
        'total_processing': 0
    }
    
    start_time = time.time()
    
    # Check directory existence
    step_start = time.time()
    if not os.path.exists(buildings_dir):
        print(f"[WARNING] Buildings directory does not exist: {buildings_dir}")
        return
    timing_stats['directory_check'] = time.time() - step_start
    
    # Get all GLB files
    step_start = time.time()
    glb_files = []
    for file in os.listdir(buildings_dir):
        if file.lower().endswith('.glb'):
            glb_files.append(os.path.join(buildings_dir, file))
    
    if not glb_files:
        print(f"[INFO] No GLB files found in buildings directory: {buildings_dir}")
        return
    
    print(f"[INFO] Found {len(glb_files)} building GLB files")
    timing_stats['file_discovery'] = time.time() - step_start
    
    # Create error log file
    error_log_path = os.path.join(scene_dir, "building_import_errors.txt")
    failed_buildings = []
    
    # Import all building GLB files
    imported_count = 0
    glb_import_time = 0
    object_naming_time = 0
    object_deselection_time = 0
    
    for glb_path in tqdm(glb_files, desc="Importing buildings"):
        # if imported_count >= 200:
        #     break
        try:
            if os.path.exists(glb_path):
                # GLB import timing
                step_start = time.time()
                bpy.ops.import_scene.gltf(filepath=glb_path)
                glb_import_time += time.time() - step_start
                imported_count += 1
                
                # Object naming timing
                step_start = time.time()
                imported_objs = bpy.context.selected_objects
                for obj in imported_objs:
                    if not obj.name.startswith("buildings_"):
                        obj.name = f"buildings_{obj.name}"
                object_naming_time += time.time() - step_start
                
                # Object deselection timing
                step_start = time.time()
                for obj in imported_objs:
                    if hasattr(obj, 'select_set'):
                        obj.select_set(False)
                    elif hasattr(obj, 'select'):
                        obj.select = False
                object_deselection_time += time.time() - step_start
            else:
                error_msg = f"File not found: {glb_path}"
                print(f"[WARNING] {error_msg}")
                failed_buildings.append(f"{os.path.basename(glb_path)} - {error_msg}")
                
        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            print(f"[ERROR] Import GLB file failed {glb_path}: {e}")
            failed_buildings.append(f"{os.path.basename(glb_path)} - {error_msg}")
    
    timing_stats['glb_import'] = glb_import_time
    timing_stats['object_naming'] = object_naming_time
    timing_stats['object_deselection'] = object_deselection_time
    
    # Write error log file
    step_start = time.time()
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
    timing_stats['error_logging'] = time.time() - step_start
    
    timing_stats['total_processing'] = time.time() - start_time
    
    # Print timing statistics
    print(f"\n[TIMING] Building import timing for {os.path.basename(scene_dir)}:")
    print(f"  directory_check: {timing_stats['directory_check']:.3f}s")
    print(f"  file_discovery: {timing_stats['file_discovery']:.3f}s ({len(glb_files)} files)")
    print(f"  glb_import: {timing_stats['glb_import']:.3f}s ({imported_count} files)")
    print(f"  object_naming: {timing_stats['object_naming']:.3f}s")
    print(f"  object_deselection: {timing_stats['object_deselection']:.3f}s")
    print(f"  error_logging: {timing_stats['error_logging']:.3f}s")
    print(f"  total_processing: {timing_stats['total_processing']:.3f}s")
    
    print(f"[COMPLETED] Successfully imported {imported_count}/{len(glb_files)} building files")
    
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
        if location[-1] < 1:
            print(f"Location height is less than 1: {obj_path}")
            continue
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
                    cache_data.append({ "name": obj.name, "mesh": obj.data.copy()  }) 
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
    print("bg",bg)
    output = nodes.get("World Output")

    env = nodes.new(type="ShaderNodeTexEnvironment")
    env.location = (-300, 0)
    env.image = bpy.data.images.load(hdri_path)

    # Connect nodes: Environment → Background → World Output
    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # Adjust brightness intensity (default 10.0, can be changed)
    bg.inputs["Strength"].default_value = 10.0

def generate_city_statistics_report(city, output_dir, objects_placed, timing_stats):
    """Generate detailed statistics report for a city"""
    import datetime
    import json
    
    # Get current scene statistics
    scene_stats = {
        'total_objects': len(bpy.data.objects),
        'total_meshes': len(bpy.data.meshes),
        'total_materials': len(bpy.data.materials),
        'total_textures': len(bpy.data.textures),
        'total_images': len(bpy.data.images),
        'total_collections': len(bpy.data.collections),
        'total_lights': len(bpy.data.lights),
        'total_cameras': len(bpy.data.cameras)
    }
    
    # Load amenities and natural objects data
    amenities_data = {}
    natural_data = {}
    amenities_stats = {}
    natural_stats = {}
    
    try:
        amenities_json_path = os.path.join(DATA_DIR, city, "objects/amenities_with_z.json")
        natural_json_path = os.path.join(DATA_DIR, city, "objects/natural_with_z.json")
        
        if os.path.exists(amenities_json_path):
            with open(amenities_json_path, 'r', encoding='utf-8') as f:
                amenities_data = json.load(f)
                
        if os.path.exists(natural_json_path):
            with open(natural_json_path, 'r', encoding='utf-8') as f:
                natural_data = json.load(f)
                
        # Analyze amenities objects
        if amenities_data and 'objects' in amenities_data:
            amenities_objects = amenities_data['objects']
            amenities_stats = {
                'total_count': len(amenities_objects),
                'type_breakdown': {},
                'successful_imports': 0,
                'failed_imports': 0
            }
            
            for obj in amenities_objects:
                obj_type = obj.get('name', 'unknown')
                amenities_stats['type_breakdown'][obj_type] = amenities_stats['type_breakdown'].get(obj_type, 0) + 1
                
                # Check if object was successfully imported (has valid location)
                location = obj.get('location', [0, 0, 0])
                if location != [0, 0, 0]:
                    amenities_stats['successful_imports'] += 1
                else:
                    amenities_stats['failed_imports'] += 1
        
        # Analyze natural objects
        if natural_data and 'objects' in natural_data:
            natural_objects = natural_data['objects']
            natural_stats = {
                'total_count': len(natural_objects),
                'type_breakdown': {},
                'successful_imports': 0,
                'failed_imports': 0
            }
            
            for obj in natural_objects:
                obj_type = obj.get('name', 'unknown')
                natural_stats['type_breakdown'][obj_type] = natural_stats['type_breakdown'].get(obj_type, 0) + 1
                
                # Check if object was successfully imported (has valid location)
                location = obj.get('location', [0, 0, 0])
                if location != [0, 0, 0]:
                    natural_stats['successful_imports'] += 1
                else:
                    natural_stats['failed_imports'] += 1
                    
    except Exception as e:
        print(f"[WARNING] Failed to analyze amenities/natural data: {e}")
        amenities_stats = {'total_count': 0, 'type_breakdown': {}, 'successful_imports': 0, 'failed_imports': 0}
        natural_stats = {'total_count': 0, 'type_breakdown': {}, 'successful_imports': 0, 'failed_imports': 0}
    
    # Get memory usage
    try:
        memory_mb, memory_scene_stats = get_memory_usage()
    except:
        memory_mb = 0
        memory_scene_stats = scene_stats
    
    # Calculate cache efficiency
    cache_hits = timing_stats.get('glb_cache_hit', 0)
    cache_misses = timing_stats.get('glb_import', 0)
    total_cache_operations = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / total_cache_operations * 100) if total_cache_operations > 0 else 0
    
    # Create statistics file
    stats_file = os.path.join(output_dir, f"{city}_statistics.txt")
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"City Processing Statistics Report - {city}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Processing Summary
        f.write("PROCESSING SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Processing status: COMPLETED\n")
        f.write(f"Total processing time: {timing_stats.get('total_processing', 0):.3f} seconds\n")
        f.write(f"Memory usage: {memory_mb:.1f} MB\n\n")
        
        # Highway Objects Statistics
        f.write("HIGHWAY OBJECTS PLACED\n")
        f.write("-" * 30 + "\n")
        total_highway_objects = 0
        for obj_type, count in objects_placed.items():
            if count > 0:
                f.write(f"{obj_type}: {count} objects\n")
                total_highway_objects += count
        f.write(f"Total highway objects: {total_highway_objects}\n\n")
        
        # Amenities Objects Statistics
        f.write("AMENITIES OBJECTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total amenities objects: {amenities_stats.get('total_count', 0)}\n")
        f.write(f"Successfully imported: {amenities_stats.get('successful_imports', 0)}\n")
        f.write(f"Failed imports: {amenities_stats.get('failed_imports', 0)}\n")
        
        if amenities_stats.get('successful_imports', 0) > 0:
            success_rate = (amenities_stats['successful_imports'] / amenities_stats['total_count'] * 100) if amenities_stats['total_count'] > 0 else 0
            f.write(f"Success rate: {success_rate:.1f}%\n")
        
        if amenities_stats.get('type_breakdown'):
            f.write("\nAmenities by type:\n")
            for obj_type, count in sorted(amenities_stats['type_breakdown'].items()):
                f.write(f"  {obj_type}: {count} objects\n")
        f.write("\n")
        
        # Natural Objects Statistics
        f.write("NATURAL OBJECTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total natural objects: {natural_stats.get('total_count', 0)}\n")
        f.write(f"Successfully imported: {natural_stats.get('successful_imports', 0)}\n")
        f.write(f"Failed imports: {natural_stats.get('failed_imports', 0)}\n")
        
        if natural_stats.get('successful_imports', 0) > 0:
            success_rate = (natural_stats['successful_imports'] / natural_stats['total_count'] * 100) if natural_stats['total_count'] > 0 else 0
            f.write(f"Success rate: {success_rate:.1f}%\n")
        
        if natural_stats.get('type_breakdown'):
            f.write("\nNatural objects by type:\n")
            for obj_type, count in sorted(natural_stats['type_breakdown'].items()):
                f.write(f"  {obj_type}: {count} objects\n")
        f.write("\n")
        
        # Cache Performance
        f.write("CACHE PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"Cache hit time: {cache_hits:.3f} seconds\n")
        f.write(f"Cache miss time: {cache_misses:.3f} seconds\n")
        f.write(f"Total cache operations: {total_cache_operations:.3f} seconds\n")
        f.write(f"Cache hit rate: {cache_hit_rate:.1f}%\n\n")
        
        # Detailed Timing Statistics
        f.write("DETAILED TIMING STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"get_available_objects: {timing_stats.get('get_available_objects', 0):.3f}s\n")
        f.write(f"crossing_processing: {timing_stats.get('crossing_processing', 0):.3f}s\n")
        f.write(f"bus_stop_processing: {timing_stats.get('bus_stop_processing', 0):.3f}s\n")
        f.write(f"stop_processing: {timing_stats.get('stop_processing', 0):.3f}s\n")
        f.write(f"glb_cache_hit: {timing_stats.get('glb_cache_hit', 0):.3f}s\n")
        f.write(f"glb_import: {timing_stats.get('glb_import', 0):.3f}s\n")
        f.write(f"total_processing: {timing_stats.get('total_processing', 0):.3f}s\n\n")
        
        # Scene Statistics
        f.write("SCENE STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total objects in scene: {scene_stats['total_objects']}\n")
        f.write(f"Total meshes: {scene_stats['total_meshes']}\n")
        f.write(f"Total materials: {scene_stats['total_materials']}\n")
        f.write(f"Total textures: {scene_stats['total_textures']}\n")
        f.write(f"Total images: {scene_stats['total_images']}\n")
        f.write(f"Total collections: {scene_stats['total_collections']}\n")
        f.write(f"Total lights: {scene_stats['total_lights']}\n")
        f.write(f"Total cameras: {scene_stats['total_cameras']}\n\n")
        
        # Object Type Breakdown
        f.write("OBJECT TYPE BREAKDOWN\n")
        f.write("-" * 30 + "\n")
        object_types = {}
        for obj in bpy.data.objects:
            obj_type = obj.type
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        for obj_type, count in sorted(object_types.items()):
            f.write(f"{obj_type}: {count} objects\n")
        f.write("\n")
        
        # Material Statistics
        f.write("MATERIAL STATISTICS\n")
        f.write("-" * 30 + "\n")
        material_types = {}
        for mat in bpy.data.materials:
            if mat.use_nodes:
                material_types['shader_material'] = material_types.get('shader_material', 0) + 1
            else:
                material_types['basic_material'] = material_types.get('basic_material', 0) + 1
        
        for mat_type, count in material_types.items():
            f.write(f"{mat_type}: {count} materials\n")
        f.write("\n")
        
        # File Information
        f.write("FILE INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"USD output file: {os.path.join(output_dir, f'{city}_with_road_objects.usd')}\n")
        f.write(f"Statistics file: {stats_file}\n")
        f.write(f"Report generated by: blender_process_cities_with_roads.py\n")
        f.write(f"Script version: 1.0\n")
        f.write(f"Processing date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Processing time: {datetime.datetime.now().strftime('%H:%M:%S')}\n\n")
        
        # Performance Analysis
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        if timing_stats.get('total_processing', 0) > 0:
            objects_per_second = total_highway_objects / timing_stats['total_processing']
            f.write(f"Highway objects per second: {objects_per_second:.2f}\n")
            
            # Total objects processed
            total_processed_objects = (
                total_highway_objects + 
                amenities_stats.get('successful_imports', 0) + 
                natural_stats.get('successful_imports', 0)
            )
            if total_processed_objects > 0:
                total_objects_per_second = total_processed_objects / timing_stats['total_processing']
                f.write(f"Total objects per second: {total_objects_per_second:.2f}\n")
        
        if cache_hit_rate > 0:
            f.write(f"Cache efficiency: {'Excellent' if cache_hit_rate > 80 else 'Good' if cache_hit_rate > 60 else 'Fair' if cache_hit_rate > 40 else 'Poor'}\n")
        
        if memory_mb > 0:
            f.write(f"Memory efficiency: {'Good' if memory_mb < 1000 else 'Moderate' if memory_mb < 2000 else 'High' if memory_mb < 4000 else 'Very High'} ({memory_mb:.1f} MB)\n")
        
        # Import success analysis
        total_amenities = amenities_stats.get('total_count', 0)
        total_natural = natural_stats.get('total_count', 0)
        if total_amenities > 0 or total_natural > 0:
            f.write(f"\nImport success analysis:\n")
            if total_amenities > 0:
                amenities_success_rate = (amenities_stats.get('successful_imports', 0) / total_amenities * 100)
                f.write(f"  Amenities import rate: {amenities_success_rate:.1f}% ({amenities_stats.get('successful_imports', 0)}/{total_amenities})\n")
            if total_natural > 0:
                natural_success_rate = (natural_stats.get('successful_imports', 0) / total_natural * 100)
                f.write(f"  Natural objects import rate: {natural_success_rate:.1f}% ({natural_stats.get('successful_imports', 0)}/{total_natural})\n")
        
        f.write("\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    print(f"[STATS] Generated detailed statistics report: {stats_file}")

def process_scene(scene_dir, output_dir, city):
    """Process a single city scene"""
    print(f"\n====== Processing city: {city} ======")
    
    # Print initial memory stats
    print_memory_stats("START", city)
    
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
    print_memory_stats("AFTER_CLEANUP", city)

    # Import roof and terrain
    print("Importing roof and terrain")
    import_glb(os.path.join(scene_dir, "roof.glb"))
    import_glb(os.path.join(scene_dir, "terrain.glb"))
    print_memory_stats("AFTER_TERRAIN", city)

    # Import all GLB files in the buildings directory
    print("Importing buildings")
    import_buildings_from_directory(scene_dir)
    print_memory_stats("AFTER_BUILDINGS", city)

    # Import objects from JSON
    print("Importing objects from JSON")
    amenities_json = os.path.join(scene_dir, "objects/amenities_with_z.json")
    natural_json = os.path.join(scene_dir, "objects/natural_with_z.json")
    import_objects_from_json([amenities_json, natural_json], OBJECTS_BASE_DIR)
    print_memory_stats("AFTER_OBJECTS", city)

    # Set HDRI
    print("Setting up HDRI")
    setup_hdri_world(HDRI_PATH)
    print_memory_stats("AFTER_HDRI", city)
    
    # Process highway nodes and place road objects
    print("Processing highway nodes and placing road objects")
    osm_file_path = os.path.join(scene_dir, "road_data/road_data_with_height.osm")
    objects_placed = {'crossing': 0, 'traffic_signals': 0, 'bus_stop': 0, 'stop': 0}
    timing_stats = {'total_processing': 0}
    
    if os.path.exists(osm_file_path):
        nodes, ways, highway_nodes = load_osm_with_height(osm_file_path)
        objects_placed, timing_stats = process_highway_nodes(highway_nodes, city, nodes, ways)
    else:
        print(f"[WARNING] OSM file not found: {osm_file_path}")

    # Export USD
    print("Exporting USD")
    os.makedirs(output_dir, exist_ok=True)
    usd_out = os.path.join(output_dir, f"{city}_with_road_objects.usd")
    
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
    
    # Print final memory stats
    print_memory_stats("FINAL", city)
    
    # Generate detailed statistics report
    generate_city_statistics_report(city, output_dir, objects_placed, timing_stats)
    
    print(f"[COMPLETED] {city} exported: {usd_out}")
    
    
def main():
    import gc
    
    for i, city in enumerate(cities):
        print(f"\n{'='*60}")
        print(f"PROCESSING CITY {i+1}/{len(cities)}: {city}")
        print(f"{'='*60}")
        
        scene_dir = os.path.join(DATA_DIR, city)
        output_dir = os.path.join(OUTPUT_BASE_DIR, city)
        
        # Process the city
        process_scene(scene_dir, output_dir, city)
        
        # Force garbage collection between cities
        if i < len(cities) - 1:  # Don't do this after the last city
            print(f"\n[OPTIMIZATION] Forcing garbage collection after {city}...")
            gc.collect()
            print_memory_stats("AFTER_GC", city)
            print(f"[OPTIMIZATION] Ready to process next city")

if __name__ == "__main__":
    main()

