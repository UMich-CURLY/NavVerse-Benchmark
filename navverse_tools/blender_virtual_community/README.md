# Blender Virtual Community

This repository contains tools for generating 3D city environments in Blender using OpenStreetMap data and height field information.

## Workflow Overview

### 1. City Objects and Buildings Processing

**Step 1: Height Field Processing**

- `process_height_field.py` - Run outside Blender
- Processes height data from `height_field.npz` files for each city
- Assigns elevation values to objects

**Step 2: Blender Scene Generation**

- `blender_process_cities.py` - Run inside Blender (paste code into Scripting workspace)
- Generates complete USD scenes for each city including:
  - Roof geometry (`roof.glb`)
  - Terrain mesh (`terrain.glb`)
  - Building models
  - Objects positioned with elevation data

- `blender_process_cities_with_roads.py` Run inside Blender (paste code into Scripting workspace)
- Generates complete USD scenes for each city including:
  - Roof geometry (`roof.glb`)
  - Terrain mesh (`terrain.glb`)
  - Building models
  - Objects positioned with elevation data
  - Add road objects based on road data - traffic lights, signs

### 2. Road Network Visualization

**Step 1: Road Height Processing**

- `process_road_for_height.py` - Run outside Blender
- Extracts road elevation data from `height_field.npz` files and road width data from road_tada/roads.pkl
- Prepares road geometry with accurate height and width information

**Step 2: Road Scene Creation**

- `blender_add_road_with_height.py` - Run inside Blender (paste code into Scripting workspace)
- Generates road network and terrain for visualization
- Creates viewable road infrastructure with proper elevation

## Usage Notes

- Run external Python scripts first to prepare height data
- Copy and paste Blender scripts into Blender's Scripting workspace for execution

## TODO

- Segment terrain with different heights based on road infomation
