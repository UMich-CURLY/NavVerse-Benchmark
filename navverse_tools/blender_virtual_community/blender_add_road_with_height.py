# Blender OSM Roads Visualization - Using pre-processed OSM with height data
# Run this script inside Blender (Scripting workspace)
#
# New Features:
# 1. Road width is calculated from OSM tags (width data embedded during preprocessing)
# 2. Materials are created based on actual road names from OSM data
# 3. Each road gets a unique material named after its actual name (e.g., "road_West 29th Street_residential")
# 4. Width calculation priority:
#    - First: Direct width tags from OSM (pre-processed from roads.pkl)
#    - Second: Lane count multiplied by standard lane widths
#    - Road type-specific adjustments and minimum width requirements
# 5. Debug output shows calculated width and OSM width tag for verification
# 6. Run process_road_for_height.py first to embed width data into OSM file
# 7. Terrain import: Automatically loads terrain.glb for each city
# 8. Highway node objects: Places 3D objects at highway nodes:
#    - crossing: Traffic lights (*_traffic_light.glb)
#    - traffic_signals: Road signs (*_road_sign.glb)
#    - bus_stop: Bus stop signs (Bus_stop_sign.glb)
#    - stop: Stop signs (stop_sign.glb)

import bpy
import bmesh
import os
import math
import xml.etree.ElementTree as ET
import random
import glob
from tqdm import tqdm
import sys

# Try to import mathutils with fallback
try:
    import mathutils
    HAS_MATHUTILS = True
    print("✅ mathutils imported successfully")
except ImportError:
    HAS_MATHUTILS = False
    print("⚠️ mathutils not available, using Euler angles as fallback")

cities = ["AMSTERDAM", "AUSTIN", "BALTIMORE", "BARCELONA", "BELGRADE", "BERLIN", "BOISE", "BOSTON", "BRATISLAVA", "BRUSSELS", "BUDAPEST", "CALGARY", "CHARLOTTE", "CHICAGO", "CHRISTCHURCH", "COLUMBUS", "DENVER", "DETROIT", "EL_PASO", "FLORENCE", "FORT_WORTH", "FRANKFURT", "HAMBURG", "HARVARD", "KANSAS_CITY", "LASVEGAS", "LONDON", "LONGISLAND", "MADISON", "MADRID", "MADRID2", "MILAN", "MINNEAPOLIS", "MIT", "MONTREAL", "NY", "ORLANDO", "PARIS", "PHILADELPHIA", "PORTLAND", "ROME", "SANFRANCISCO", "SANFRANCISCO2", "SILICONVALLEY", "STANFORD", "SYDNEY", "TORONTO", "UCLA", "UMASS", "WHITEHOUSE", "YALE", "ZURICH"]

DATA_DIR = "D:/Desktop/ViCo"

def suppress_output():
    """Context manager to suppress stdout and stderr"""
    class DevNull:
        def write(self, msg): pass
        def flush(self): pass
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = DevNull(), DevNull()
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def import_usd(filepath):
    """Import USD file"""
    if os.path.exists(filepath):
        print(f"Importing USD: {filepath}")
        try:
            bpy.ops.wm.usd_import(filepath=filepath)
            print(f"✅ Successfully imported USD: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"❌ Error importing USD {filepath}: {e}")
    else:
        print(f"❌ USD file not found: {filepath}")

def clear_scene():
    """Clear the default scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_simple_material(name, color):
    """Create a simple material"""
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    
    # Add Principled BSDF
    bsdf = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    
    # Add output
    output = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return material

def create_road_materials():
    """Create a cache for road materials"""
    return {}

def get_or_create_material(material_name, road_type, road_name=None):
    """Get existing material or create new one based on road type and name"""
    materials = bpy.data.materials
    
    # Create material name based on road name if available, otherwise use road type
    if road_name:
        # Clean the road name to make it a valid material name
        clean_name = "".join(c for c in road_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        material_name = f"road_{clean_name}_{road_type}"
    else:
        material_name = f"road_{road_type}"
    
    # Check if material already exists
    if material_name in materials:
        return materials[material_name]
    
    # Get color based on road type
    road_colors = {
        'motorway': (1.0, 0.0, 0.0),      # Red
        'trunk': (1.0, 0.5, 0.0),         # Orange
        'primary': (1.0, 1.0, 0.0),       # Yellow
        'secondary': (0.0, 1.0, 0.0),     # Green
        'tertiary': (0.0, 0.0, 1.0),      # Blue
        'residential': (0.5, 0.0, 1.0),   # Purple
        'unclassified': (0.5, 0.5, 0.5),  # Gray
        'service': (0.8, 0.4, 0.2),       # Brown
        'footway': (1.0, 0.8, 0.8),       # Pink
        'cycleway': (0.0, 1.0, 1.0),      # Cyan
        'default': (0.3, 0.3, 0.3)        # Dark Gray
    }
    
    color = road_colors.get(road_type, road_colors['default'])
    
    # Create new material
    material = create_simple_material(material_name, color)
    return material

def calculate_road_width(tags, road_type):
    """Calculate road width based on OSM tags (width data now embedded in OSM)"""
    
    # First priority: Try to get width directly from OSM tags
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
    
    # Second priority: Calculate width based on lanes
    lanes = 1
    if 'lanes' in tags:
        try:
            lanes = int(tags['lanes'])
        except:
            lanes = 1
    
    # Default lane width in meters
    lane_width = 3.5  # Standard lane width
    
    # Adjust lane width based on road type
    if road_type in ['motorway', 'trunk']:
        lane_width = 3.75  # Wider lanes for highways
    elif road_type in ['residential', 'service']:
        lane_width = 3.0   # Narrower lanes for local roads
    elif road_type in ['footway', 'cycleway']:
        lane_width = 2.0   # Very narrow for pedestrian/cycle paths
    
    # Calculate total width
    total_width = lanes * lane_width
    
    # Add some buffer for road shoulders/margins
    if road_type in ['motorway', 'trunk', 'primary']:
        total_width += 2.0  # Add 1m margin on each side
    elif road_type in ['secondary', 'tertiary']:
        total_width += 1.0  # Add 0.5m margin on each side
    
    # Ensure minimum width
    min_widths = {
        'motorway': 8.0,
        'trunk': 7.0,
        'primary': 6.0,
        'secondary': 5.0,
        'tertiary': 4.0,
        'residential': 3.0,
        'service': 2.5,
        'footway': 1.5,
        'cycleway': 2.0,
        'unclassified': 3.0
    }
    
    min_width = min_widths.get(road_type, 3.0)
    return max(total_width, min_width)

def create_road_with_height(way_points, road_type, road_name, tags, road_id):
    """Create a road mesh with height information from pre-processed data"""
    if len(way_points) < 2:
        return None
    
    # Calculate road width from OSM tags (width data now embedded in OSM)
    width = calculate_road_width(tags, road_type)
    
    # Get or create material based on road name and type
    material = get_or_create_material(f"road_{road_id}", road_type, road_name)
    
    # Create mesh with road name in object name
    mesh_name = f"road_{road_name}_{road_type}" if road_name else f"road_{road_type}_{road_id}"
    mesh = bpy.data.meshes.new(mesh_name)
    obj = bpy.data.objects.new(mesh_name, mesh)
    bpy.context.collection.objects.link(obj)
    
    bm = bmesh.new()
    
    # Create vertices
    vertices = []
    for i, point in enumerate(way_points):
        x, y, z = point[0], point[1], point[2]  # x, y are already relative to city center, z is height
        
        # Calculate perpendicular direction
        if i == 0 and len(way_points) > 1:
            next_point = way_points[1]
            dx = next_point[0] - x
            dy = next_point[1] - y
        elif i == len(way_points) - 1:
            prev_point = way_points[i-1]
            dx = x - prev_point[0]
            dy = y - prev_point[1]
        else:
            prev_point = way_points[i-1]
            next_point = way_points[i+1]
            dx = (next_point[0] - prev_point[0]) / 2
            dy = (next_point[1] - prev_point[1]) / 2
        
        # Normalize
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            perp_x = -dy / length * width / 2
            perp_y = dx / length * width / 2
        else:
            perp_x = perp_y = 0
        
        # Create vertices with height (add small offset above terrain)
        v1 = bm.verts.new((x + perp_x, y + perp_y, z + 5))  # offset 5 meters above terrain
        v2 = bm.verts.new((x - perp_x, y - perp_y, z + 5))
        vertices.append((v1, v2))
    
    # Create faces
    for i in range(len(vertices) - 1):
        v1_curr, v2_curr = vertices[i]
        v1_next, v2_next = vertices[i + 1]
        
        face = bm.faces.new([v1_curr, v1_next, v2_next, v2_curr])
        face.smooth = True
    
    # Update mesh
    bm.to_mesh(mesh)
    bm.free()
    
    # Assign material
    obj.data.materials.append(material)
    
    return obj

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

def process_highway_nodes(highway_nodes, city_name, nodes, ways):
    """Process highway nodes and place appropriate 3D objects"""
    print(f"Processing {len(highway_nodes)} highway nodes for {city_name}")
    
    # Get available objects
    available_objects = get_available_objects()
    
    # Statistics
    objects_placed = {
        'crossing': 0,
        'traffic_signals': 0,
        'bus_stop': 0,
        'stop': 0
    }
    
    for node in tqdm(highway_nodes, desc="Placing highway objects"):
        highway_type = node['highway']
        
        if highway_type == 'crossing' and available_objects['crossing']:
            # Randomly select a traffic light
            traffic_light_file = random.choice(available_objects['crossing'])
            objects = load_object_from_glb(traffic_light_file)
            if objects:
                place_objects_at_node(objects, node['utm_x'], node['utm_y'], node['height'], highway_type, node, nodes, ways)
                objects_placed['crossing'] += 1
                
        # elif highway_type == 'traffic_signals' and available_objects['traffic_signals']:
        #     # Randomly select a road sign
        #     road_sign_file = random.choice(available_objects['traffic_signals'])
        #     objects = load_object_from_glb(road_sign_file)
        #     if objects:
        #         place_objects_at_node(objects, node['utm_x'], node['utm_y'], node['height'], highway_type, node, nodes, ways)
        #         objects_placed['traffic_signals'] += 1
        #         
        elif highway_type == 'bus_stop' and available_objects['bus_stop']:
            # Randomly choose between bus_stop_sign and *_road_sign.glb files
            chosen_file = random.choice(available_objects['bus_stop'])
            objects = load_object_from_glb(chosen_file)
            if objects:
                place_objects_at_node(objects, node['utm_x'], node['utm_y'], node['height'], highway_type, node, nodes, ways)
                objects_placed['bus_stop'] += 1
                
        elif highway_type == 'stop' and available_objects['stop']:
            # Use stop sign
            objects = load_object_from_glb(available_objects['stop'])
            if objects:
                place_objects_at_node(objects, node['utm_x'], node['utm_y'], node['height'], highway_type, node, nodes, ways)
                objects_placed['stop'] += 1
    
    # Print statistics
    print(f"\nHighway objects placed for {city_name}:")
    total_placed = 0
    for obj_type, count in objects_placed.items():
        if count > 0:
            print(f"  {obj_type}: {count}")
        total_placed += count
    
    print(f"  Total objects placed: {total_placed}")
    
    return objects_placed

def process_roads_with_height(osm_file_path, city_name, max_roads=2000):
    """Process roads using pre-processed height data with embedded width information"""
    print(f"Processing roads for {city_name} with pre-processed height and width data")
    
    # Load OSM data (now includes width information and highway nodes)
    nodes, ways, highway_nodes = load_osm_with_height(osm_file_path)
    
    road_count = 0
    road_stats = {}
    road_names_stats = {}
    
    # Process roads using pre-processed coordinates and heights
    for way in tqdm(ways, desc="Processing roads"):
        if road_count >= max_roads:
            break
        
        tags = way.get('tags', {})
        if 'highway' not in tags:
            continue
        
        highway_type = tags['highway']
        way_nodes = way.get('nodes', [])
        
        if len(way_nodes) < 2:
            continue
        
        # Get road name
        road_name = tags.get('name', None)
        
        # Convert to coordinates using pre-processed data
        way_points = []
        for node_id in way_nodes:
            if node_id in nodes:
                node = nodes[node_id]
                x = node.get('utm_x', 0)  # Already relative to city center
                y = node.get('utm_y', 0)  # Already relative to city center
                z = node.get('height', 0)  # Pre-calculated height
                
                way_points.append((x, y, z))
        
        if len(way_points) >= 2:
            road_obj = None #create_road_with_height(way_points, highway_type, road_name, tags, road_count)
            if road_obj:
                road_count += 1
                road_stats[highway_type] = road_stats.get(highway_type, 0) + 1
                
                # Track road names for statistics
                if road_name:
                    road_names_stats[road_name] = road_names_stats.get(road_name, 0) + 1
                
                # Debug info for first few roads
                if road_count <= 5:
                    width = calculate_road_width(tags, highway_type)
                    osm_width = tags.get('width', 'N/A')
                    print(f"Road {road_count}: {road_name or 'Unnamed'} ({highway_type}) - Width: {width:.1f}m (OSM: {osm_width}), Lanes: {tags.get('lanes', 'N/A')}")
                
                if road_count % 100 == 0:
                    print(f"Created {road_count} roads...")
    
    # Process highway nodes and place objects
    print(f"\nProcessing highway nodes for {city_name}...")
    highway_objects_stats = process_highway_nodes(highway_nodes, city_name, nodes, ways)
    
    print(f"Total roads created: {road_count}")
    print("Road types:")
    for road_type, count in sorted(road_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {road_type}: {count}")
    
    print("\nTop road names:")
    for road_name, count in sorted(road_names_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {road_name}: {count}")
    
    print(f"\nHighway objects placed:")
    for obj_type, count in highway_objects_stats.items():
        if count > 0:
            print(f"  {obj_type}: {count}")
    
    return road_count

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
            # else:
            #     # For other objects, use the calculated rotation
            #     obj.rotation_quaternion = rotation_quat
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

def setup_scene(city_name):
    """Setup scene with USD terrain, lighting, and camera"""
    # Import USD file from generated folder
    usd_file = f"{DATA_DIR}/generated/{city_name}/{city_name}.usd"
    if os.path.exists(usd_file):
        print(f"Loading USD scene: {usd_file}")
        import_usd(usd_file)
    else:
        print(f"Warning: USD file not found: {usd_file}")
        # Fallback to plane
        bpy.ops.mesh.primitive_plane_add(size=2000)
        ground = bpy.context.active_object
        ground.name = "Ground"
    

def export_usd(city_name):
    """Export the complete scene as USD file"""
    print(f"\nExporting USD for {city_name}...")
    
    # Create output directory
    output_dir = f"{DATA_DIR}/generated/{city_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up context for USD export
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
    
    # Enable required addons
    bpy.ops.preferences.addon_enable(module="omni_nucleus")
    bpy.ops.preferences.addon_enable(module="omni_optimization_panel")
    bpy.ops.preferences.addon_enable(module="omni_panel")
    bpy.ops.preferences.addon_enable(module="omni_audio2face")
    bpy.ops.preferences.addon_enable(module="umm2")

    # Set generate_mdl preference
    try:
        prefs = bpy.context.preferences.addons["omni_usd"].preferences
        if hasattr(prefs, "generate_mdl"):
            prefs.generate_mdl = True
            print("Set prefs.generate_mdl = True")
    except KeyError:
        pass

    # Export USD
    usd_out = os.path.join(output_dir, f"{city_name}_with_road_objects.usd")
    
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
    
    print(f"USD export result: {res}")
    print(f"✅ Exported USD: {usd_out}")

# def get_available_cities():
#     """Get list of available cities with processed OSM files"""
#     base_path = "D:/Desktop/ViCo"
#     cities = []
    
#     if os.path.exists(base_path):
#         for item in os.listdir(base_path):
#             item_path = os.path.join(base_path, item)
#             # Only process directories that are all uppercase with possible numbers
#             if (os.path.isdir(item_path) and 
#                 item.isupper() and 
#                 item != "Virtual-Community" and 
#                 item != "objects"):
                
#                 # Check if it has the processed OSM file
#                 osm_file = os.path.join(item_path, "road_data", "road_data_with_height.osm")
#                 if os.path.exists(osm_file):
#                     cities.append(item)
    
#     return sorted(cities)

def main():
    """Main function"""
    print("Blender OSM Roads Visualization - Using Pre-processed Height Data")
    print("=" * 70)
    
    # Clear scene
    clear_scene()
    
    
    print(f"\nAvailable cities for processing:")
    for i, city in enumerate(cities, 1):
        print(f"  {i}. {city}")
    
    try:
        choice = input(f"Select city (1-{len(cities)}, default for NY): ").strip()
        if not choice:
            choice = "0"
        
        city_index = int(choice) - 1
        if 0 <= city_index < len(cities):
            selected_city = cities[city_index]
        else:
            selected_city = "NY"
    except:
        selected_city = "NY"
    
    print(f"Selected city: {selected_city}")
    
    # OSM file path
    osm_file_path = f"{DATA_DIR}/{selected_city}/road_data/road_data_with_height.osm"
    
    if not os.path.exists(osm_file_path):
        print(f"Error: Processed OSM file not found: {osm_file_path}")
        print("Please run process_osm_with_height.py first to generate height data")
        return
    
    # Setup scene with terrain
    setup_scene(selected_city)
    
    # Process roads
    road_count = process_roads_with_height(osm_file_path, selected_city, max_roads=2000)
    
    # Set viewport shading
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    break
    
    print(f"\nVisualization complete! Created {road_count} roads for {selected_city}")
    print("Using pre-processed height data - no real-time calculation needed!")
    
    # Export USD
    export_usd(selected_city)

# Run the script
if __name__ == "__main__":
    main()
