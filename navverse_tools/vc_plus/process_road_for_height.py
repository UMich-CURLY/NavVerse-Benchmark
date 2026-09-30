# Process OSM files with height information - Offline processing
# This script processes all cities and adds height data to OSM nodes

import os
import math
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

cities = ["AMSTERDAM", "AUSTIN", "BALTIMORE", "BARCELONA", "BELGRADE", "BERLIN", "BOISE", "BOSTON", "BRATISLAVA", "BRUSSELS", "BUDAPEST", "CALGARY", "CHARLOTTE", "CHICAGO", "CHRISTCHURCH", "COLUMBUS", "DENVER", "DETROIT", "EL_PASO", "FLORENCE", "FORT_WORTH", "FRANKFURT", "HAMBURG", "HARVARD", "KANSAS_CITY", "LASVEGAS", "LONDON", "LONGISLAND", "MADISON", "MADRID", "MADRID2", "MILAN", "MINNEAPOLIS", "MIT", "MONTREAL", "NY", "ORLANDO", "PARIS", "PHILADELPHIA", "PORTLAND", "ROME", "SANFRANCISCO", "SANFRANCISCO2", "SILICONVALLEY", "STANFORD", "SYDNEY", "TORONTO", "UCLA", "UMASS", "WHITEHOUSE", "YALE", "ZURICH"]

cities = ["BOSTON"]

DATA_DIR = "D:/Desktop/ViCo"

def get_city_center(city_name):
    """Get city center coordinates from datapoint file"""
    datapoint_file = f"{DATA_DIR}/Virtual-Community/ViCo/scene-generation/datapoint/{city_name}.txt"
    
    if os.path.exists(datapoint_file):
        try:
            with open(datapoint_file, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    print(f"Loaded city center from {datapoint_file}: ({lat}, {lon})")
                    return lat, lon
        except Exception as e:
            print(f"Error reading {datapoint_file}: {e}")
            return None, None
    else:
        return None, None

def load_height_field(height_file_path):
    """Load height field data from npz file"""
    print(f"Loading height field from: {height_file_path}")
    
    if not os.path.exists(height_file_path):
        print(f"Warning: Height field file not found: {height_file_path}")
        return None, None, None
    
    try:
        height_field = np.load(height_file_path)
        plane_coord = height_field["plane_coord"]
        terrain_alt = height_field["terrain_alt"]
        
        # Convert to simple arrays
        xs = plane_coord[..., 0].flatten()
        ys = plane_coord[..., 1].flatten() * -1  # Flip Y coordinate
        zs = terrain_alt.flatten()
        
        print(f"Height field loaded: {len(xs)} points")
        return xs, ys, zs
    except Exception as e:
        print(f"Error loading height field: {e}")
        return None, None, None

def create_height_interpolator(xs, ys, zs):
    """Create height interpolator using scipy for better performance"""
    if xs is None or ys is None or zs is None:
        return None
    
    # Create interpolator using scipy
    points = np.stack([xs.flatten(), ys.flatten()], axis=-1)
    heights = zs.flatten()
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(points).any(axis=1) | np.isnan(heights))
    points = points[valid_mask]
    heights = heights[valid_mask]
    
    if len(points) == 0:
        print("Warning: No valid height data points")
        return None
    
    print(f"Height interpolator created with {len(points)} valid points using scipy")
    
    # Create both linear and nearest interpolators
    interp_linear = LinearNDInterpolator(points, heights)
    interp_nearest = NearestNDInterpolator(points, heights)
    
    def get_height(x, y):
        """Efficient height interpolation using scipy"""
        z = interp_linear(x, y)
        if np.isnan(z):
            z = interp_nearest(x, y)
        return float(z) if not np.isnan(z) else 0.0
    
    return get_height

def lat_lon_to_utm(lat, lon, center_lat, center_lon):
    """Convert lat/lon to UTM-like coordinates relative to city center"""
    # Calculate actual scale based on latitude
    lat_scale = 111320.0  # meters per degree latitude
    # lon_scale = 111320.0 * math.cos(math.radians(center_lat))  # meters per degree longitude at this latitude
    
    # Convert to coordinates relative to city center
    x = (lon - center_lon) * lat_scale * math.cos(math.radians(center_lat))
    y = (lat - center_lat) * lat_scale
    
    return x, y

def process_osm_with_height(city_name):
    """Process OSM file and add height information to nodes"""
    print(f"\n{'='*60}")
    print(f"Processing {city_name}")
    print(f"{'='*60}")
    
    # File paths
    osm_file_path = f"{DATA_DIR}/{city_name}/road_data/road_data.osm"
    height_file_path = f"{DATA_DIR}/{city_name}/height_field.npz"
    output_file_path = f"{DATA_DIR}/{city_name}/road_data/road_data_with_height.osm"
    
    # Check if input files exist
    if not os.path.exists(osm_file_path):
        print(f"Error: OSM file not found: {osm_file_path}")
        return False
    
    if not os.path.exists(height_file_path):
        print(f"Error: Height field file not found: {height_file_path}")
        return False
    
    # Get city center
    center_lat, center_lon = get_city_center(city_name)
    if center_lat is None or center_lon is None:
        print(f"Error: Could not get city center for {city_name}")
        return False
    
    # Load height field
    xs, ys, zs = load_height_field(height_file_path)
    height_interpolator = create_height_interpolator(xs, ys, zs)
    
    if height_interpolator is None:
        print(f"Error: Could not create height interpolator for {city_name}")
        return False
    
    # Parse OSM file
    print(f"Parsing OSM file: {osm_file_path}")
    tree = ET.parse(osm_file_path)
    root = tree.getroot()
    
    # Process nodes and add height information
    nodes_processed = 0
    nodes_with_height = 0
    
    print("Processing nodes and adding height information...")
    for element in tqdm(root, desc="Processing nodes"):
        if element.tag == 'node':
            nodes_processed += 1
            
            # Get node coordinates
            lat = float(element.get('lat', 0))
            lon = float(element.get('lon', 0))
            
            # Convert to UTM coordinates
            x, y = lat_lon_to_utm(lat, lon, center_lat, center_lon)
            
            # Get height from interpolator
            height = height_interpolator(x, y)
            
            # Add height as a tag
            height_tag = ET.SubElement(element, 'tag')
            height_tag.set('k', 'height')
            height_tag.set('v', f"{height:.3f}")
            
            # Add UTM coordinates as tags for reference
            utm_x_tag = ET.SubElement(element, 'tag')
            utm_x_tag.set('k', 'utm_x')
            utm_x_tag.set('v', f"{x:.3f}")
            
            utm_y_tag = ET.SubElement(element, 'tag')
            utm_y_tag.set('k', 'utm_y')
            utm_y_tag.set('v', f"{y:.3f}")
            
            nodes_with_height += 1
    
    # Save the modified OSM file
    print(f"Saving modified OSM file: {output_file_path}")
    tree.write(output_file_path, encoding='utf-8', xml_declaration=True)
    
    # Print statistics
    print(f"\nProcessing complete for {city_name}:")
    print(f"  Nodes processed: {nodes_processed}")
    print(f"  Nodes with height: {nodes_with_height}")
    print(f"  Output file: {output_file_path}")
    
    return True

# def get_available_cities():
#     """Get list of available cities from the directory structure"""
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
                
#                 # Check if it has the required files
#                 osm_file = os.path.join(item_path, "road_data", "road_data.osm")
#                 height_file = os.path.join(item_path, "height_field.npz")
#                 if os.path.exists(osm_file) and os.path.exists(height_file):
#                     cities.append(item)
    
#     return sorted(cities)

def main():
    """Main function to process all cities"""
    print("OSM Height Processing Tool")
    print("=" * 60)
    
    
    print(f"Found {len(cities)} cities with required data:")
    for i, city in enumerate(cities, 1):
        print(f"  {i}. {city}")
    
    # Process all cities by default
    print(f"\nProcessing all {len(cities)} cities...")
    success_count = 0
    
    
    for city in cities:
        if process_osm_with_height(city):
            success_count += 1
        else:
            print(f"Error: Could not process {city}")
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Successfully processed: {success_count}/{len(cities)} cities")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
