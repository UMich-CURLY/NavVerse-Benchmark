# Node Highway Statistics Analyzer
# This script analyzes highway tags in OSM nodes across all cities
# It processes each city's road_data_with_height.osm file and generates statistics

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

# All available cities
cities = ["AMSTERDAM", "AUSTIN", "BALTIMORE", "BARCELONA", "BELGRADE", "BERLIN", "BOISE", "BOSTON", 
          "BRATISLAVA", "BRUSSELS", "BUDAPEST", "CALGARY", "CHARLOTTE", "CHICAGO", "CHRISTCHURCH", 
          "COLUMBUS", "DENVER", "DETROIT", "EL_PASO", "FLORENCE", "FORT_WORTH", "FRANKFURT", "HAMBURG", 
          "HARVARD", "KANSAS_CITY", "LASVEGAS", "LONDON", "LONGISLAND", "MADISON", "MADRID", "MADRID2", 
          "MILAN", "MINNEAPOLIS", "MIT", "MONTREAL", "NY", "ORLANDO", "PARIS", "PHILADELPHIA", "PORTLAND", 
          "ROME", "SANFRANCISCO", "SANFRANCISCO2", "SILICONVALLEY", "STANFORD", "SYDNEY", "TORONTO", 
          "UCLA", "UMASS", "WHITEHOUSE", "YALE", "ZURICH"]

DATA_DIR = "D:/Desktop/ViCo"

def get_available_cities():
    """Get list of cities that have processed OSM files"""
    available_cities = []
    
    for city in cities:
        osm_file_path = f"{DATA_DIR}/{city}/road_data/road_data_with_height.osm"
        if os.path.exists(osm_file_path):
            available_cities.append(city)
    
    return available_cities

def analyze_city_highway_stats(city_name):
    """Analyze highway statistics for a single city"""
    osm_file_path = f"{DATA_DIR}/{city_name}/road_data/road_data_with_height.osm"
    
    if not os.path.exists(osm_file_path):
        print(f"OSM file not found for {city_name}: {osm_file_path}")
        return None
    
    try:
        print(f"Analyzing {city_name}...")
        tree = ET.parse(osm_file_path)
        root = tree.getroot()
        
        # Statistics collection
        total_nodes = 0
        nodes_with_highway = 0
        highway_stats = {}
        node_details = []  # Store detailed node information
        
        # Process all nodes
        for element in tqdm(root, desc=f"Processing {city_name} nodes", leave=False):
            if element.tag == 'node':
                total_nodes += 1
                
                # Get node basic info
                node_info = {
                    'id': element.get('id'),
                    'lat': element.get('lat'),
                    'lon': element.get('lon'),
                    'highway': None,
                    'other_tags': {}
                }
                
                # Check for highway tags and collect all tags
                for tag in element.findall('tag'):
                    k = tag.get('k')
                    v = tag.get('v')
                    
                    if k == 'highway':
                        node_info['highway'] = v
                        nodes_with_highway += 1
                        highway_stats[v] = highway_stats.get(v, 0) + 1
                    else:
                        node_info['other_tags'][k] = v
                
                # Only store nodes with highway tags for detailed analysis
                if node_info['highway']:
                    node_details.append(node_info)
        
        # Calculate statistics
        stats = {
            'city': city_name,
            'total_nodes': total_nodes,
            'nodes_with_highway': nodes_with_highway,
            'highway_percentage': (nodes_with_highway / total_nodes * 100) if total_nodes > 0 else 0,
            'highway_distribution': highway_stats,
            'node_details': node_details
        }
        
        print(f"  Total nodes: {total_nodes}")
        print(f"  Nodes with highway tags: {nodes_with_highway} ({stats['highway_percentage']:.1f}%)")
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing {city_name}: {e}")
        return None

def save_city_statistics(stats):
    """Save detailed statistics for a single city"""
    if not stats:
        return
    
    city_name = stats['city']
    output_dir = f"{DATA_DIR}/{city_name}/road_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON statistics
    json_file = f"{output_dir}/node_highway_detailed_stats.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Save human-readable text statistics
    txt_file = f"{output_dir}/node_highway_statistics.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Node Highway Statistics for {city_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total nodes processed: {stats['total_nodes']}\n")
        f.write(f"Nodes with highway tags: {stats['nodes_with_highway']}\n")
        f.write(f"Percentage with highway tags: {stats['highway_percentage']:.1f}%\n\n")
        
        if stats['highway_distribution']:
            f.write("Highway types distribution:\n")
            f.write("-" * 30 + "\n")
            
            # Sort by count (descending)
            sorted_stats = sorted(stats['highway_distribution'].items(), key=lambda x: x[1], reverse=True)
            for highway_type, count in sorted_stats:
                percentage = (count / stats['nodes_with_highway']) * 100
                f.write(f"{highway_type:20s}: {count:6d} ({percentage:5.1f}%)\n")
        
        f.write(f"\nDetailed node data saved to: {json_file}\n")
        f.write(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"  Statistics saved to: {txt_file}")

def create_global_summary(all_stats):
    """Create global summary across all cities"""
    if not all_stats:
        return
    
    # Collect global statistics
    global_highway_types = {}
    city_summaries = {}
    total_nodes_global = 0
    total_highway_nodes_global = 0
    
    for stats in all_stats:
        if stats:
            city_name = stats['city']
            city_summaries[city_name] = {
                'total_nodes': stats['total_nodes'],
                'highway_nodes': stats['nodes_with_highway'],
                'percentage': stats['highway_percentage']
            }
            
            total_nodes_global += stats['total_nodes']
            total_highway_nodes_global += stats['nodes_with_highway']
            
            # Merge highway type distributions
            for highway_type, count in stats['highway_distribution'].items():
                global_highway_types[highway_type] = global_highway_types.get(highway_type, 0) + count
    
    # Save global summary
    global_file = f"{DATA_DIR}/global_node_highway_statistics.txt"
    with open(global_file, 'w', encoding='utf-8') as f:
        f.write("Global Node Highway Statistics Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Analysis Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Cities Analyzed: {len([s for s in all_stats if s])}\n")
        f.write(f"Total Nodes Globally: {total_nodes_global:,}\n")
        f.write(f"Total Highway Nodes Globally: {total_highway_nodes_global:,}\n")
        f.write(f"Global Highway Percentage: {(total_highway_nodes_global/total_nodes_global)*100:.1f}%\n\n")
        
        # Global highway type distribution
        if global_highway_types:
            f.write("Global Highway Types Distribution:\n")
            f.write("-" * 40 + "\n")
            
            sorted_global = sorted(global_highway_types.items(), key=lambda x: x[1], reverse=True)
            for highway_type, count in sorted_global:
                percentage = (count / total_highway_nodes_global) * 100
                f.write(f"{highway_type:20s}: {count:8d} ({percentage:5.1f}%)\n")
        
        # Per-city summary
        f.write(f"\nPer-City Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'City':15s} {'Total Nodes':>12s} {'Highway Nodes':>14s} {'Percentage':>10s}\n")
        f.write("-" * 60 + "\n")
        
        sorted_cities = sorted(city_summaries.items(), key=lambda x: x[1]['highway_nodes'], reverse=True)
        for city, summary in sorted_cities:
            f.write(f"{city:15s} {summary['total_nodes']:12,d} {summary['highway_nodes']:14,d} {summary['percentage']:9.1f}%\n")
    
    print(f"\nGlobal summary saved to: {global_file}")
    
    # Save global JSON summary
    global_json_file = f"{DATA_DIR}/global_node_highway_statistics.json"
    global_summary = {
        'analysis_date': __import__('datetime').datetime.now().isoformat(),
        'cities_analyzed': len([s for s in all_stats if s]),
        'global_totals': {
            'total_nodes': total_nodes_global,
            'highway_nodes': total_highway_nodes_global,
            'percentage': (total_highway_nodes_global/total_nodes_global)*100
        },
        'global_highway_distribution': global_highway_types,
        'city_summaries': city_summaries
    }
    
    with open(global_json_file, 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    
    print(f"Global JSON summary saved to: {global_json_file}")

def main():
    """Main function to analyze all cities"""
    print("Node Highway Statistics Analyzer")
    print("=" * 60)
    
    # Get available cities
    available_cities = get_available_cities()
    
    if not available_cities:
        print("No cities found with processed OSM files!")
        print("Required file: road_data/road_data_with_height.osm")
        return
    
    print(f"Found {len(available_cities)} cities with processed OSM files:")
    for i, city in enumerate(available_cities, 1):
        print(f"  {i}. {city}")
    
    print(f"\nAnalyzing highway statistics for all {len(available_cities)} cities...")
    
    # Analyze each city
    all_stats = []
    success_count = 0
    
    for city in available_cities:
        stats = analyze_city_highway_stats(city)
        if stats:
            all_stats.append(stats)
            save_city_statistics(stats)
            success_count += 1
        else:
            print(f"Failed to analyze {city}")
    
    # Create global summary
    if all_stats:
        print(f"\nCreating global summary...")
        create_global_summary(all_stats)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Successfully analyzed: {success_count}/{len(available_cities)} cities")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()






