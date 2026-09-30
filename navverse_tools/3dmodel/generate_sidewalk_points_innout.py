import argparse
import json
import math
import os
import random
import sys
import bisect

try:
    from pxr import Usd, UsdGeom, Gf
except ImportError:
    print("Error: pxr module not found. Please run with an environment that has USD (e.g. env_isaaclab).")
    sys.exit(1)

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

N_POINTS = 10000

def get_bbox_center_and_range(prim):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    bound = bbox_cache.ComputeWorldBound(prim)
    range3d = bound.ComputeAlignedBox()
    min_point = range3d.GetMin()
    max_point = range3d.GetMax()
    center = (min_point + max_point) / 2.0
    return center, range3d

def get_building_bboxes(stage, room_center, max_dist):
    bboxes = []
    root_prim = stage.GetPrimAtPath("/World/city/root")
    if not root_prim:
        return bboxes
    
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    
    for child in root_prim.GetChildren():
        bound = bbox_cache.ComputeWorldBound(child)
        range3d = bound.ComputeAlignedBox()
        
        # Optimization: Check if bbox overlaps with sampling area roughly
        # Check closest point distance or simple center distance
        # We'll just include all for simplicity unless it's too slow, 
        # but filtering by distance is safer for large scenes.
        
        min_p = range3d.GetMin()
        max_p = range3d.GetMax()
        
        # Check if the building bbox intersects with the room's sampling sphere (max_dist)
        # Closest point on AABB to room_center
        closest = Gf.Vec3d(
            max(min_p[0], min(room_center[0], max_p[0])),
            max(min_p[1], min(room_center[1], max_p[1])),
            max(min_p[2], min(room_center[2], max_p[2]))
        )
        dist_sq = (closest[0] - room_center[0])**2 + (closest[1] - room_center[1])**2 + (closest[2] - room_center[2])**2
        
        if dist_sq <= max_dist**2:
            bboxes.append(range3d)
        
    return bboxes

def sample_points_from_mesh(stage, mesh_parent_path, room_center, room_range, building_bboxes, n_points, rng, max_dist=50.0):
    terrain_parent = stage.GetPrimAtPath(mesh_parent_path)
    if not terrain_parent:
        print(f"  Mesh parent not found: {mesh_parent_path}")
        return []

    triangles = []
    
    # Iterate all meshes under terrain_parent
    # This handles split meshes like TerrainOther_001, TerrainOther_002, etc.
    mesh_prims = []
    if terrain_parent.IsA(UsdGeom.Mesh):
        mesh_prims.append(terrain_parent)
    
    # Recursively find all Mesh descendants
    # For simple hierarchy, iterating children is enough, but recursive is safer
    to_visit = [terrain_parent]
    while to_visit:
        prim = to_visit.pop(0)
        for child in prim.GetChildren():
            if child.IsA(UsdGeom.Mesh):
                mesh_prims.append(child)
            to_visit.append(child)
            
    if not mesh_prims:
        print(f"  No meshes found under {mesh_parent_path}")
        return []
        
    print(f"  Found {len(mesh_prims)} terrain mesh parts.")

    for mesh_prim in mesh_prims:
        mesh = UsdGeom.Mesh(mesh_prim)
        
        # Get Mesh Data
        points_attr = mesh.GetPointsAttr().Get()
        if not points_attr:
            continue
        
        # Transform points to world space
        xform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_points = [xform.Transform(p) for p in points_attr]
        
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        
        if not face_vertex_counts or not face_vertex_indices:
            continue

        # Iterate faces and triangulate
        idx_pointer = 0
        for count in face_vertex_counts:
            # Fan triangulation for polygons with > 3 vertices
            for i in range(count - 2):
                idx0 = face_vertex_indices[idx_pointer]
                idx1 = face_vertex_indices[idx_pointer + i + 1]
                idx2 = face_vertex_indices[idx_pointer + i + 2]
                
                try:
                    p0 = world_points[idx0]
                    p1 = world_points[idx1]
                    p2 = world_points[idx2]
                except IndexError:
                    continue # Skip invalid indices
                
                # Check if triangle is potentially within range
                # Calculate centroid for rough check
                center = (p0 + p1 + p2) / 3.0
                dist_sq = (center[0] - room_center[0])**2 + (center[1] - room_center[1])**2
                
                # Use buffer for large triangles
                if dist_sq <= (max_dist + 10.0)**2:
                    # Calculate Area
                    v1 = p1 - p0
                    v2 = p2 - p0
                    cross = Gf.Cross(v1, v2)
                    area = 0.5 * cross.GetLength()
                    
                    if area > 1e-6:
                        triangles.append({
                            'p0': p0, 'p1': p1, 'p2': p2,
                            'area': area
                        })
            
            idx_pointer += count

    if not triangles:
        print("  No valid triangles found near room.")
        return []

    print(f"  Found {len(triangles)} triangles near room.")

    total_area = sum(t['area'] for t in triangles)
    cumulative_areas = []
    current = 0
    for t in triangles:
        current += t['area']
        cumulative_areas.append(current)
    
    sampled_points = []
    attempts = 0
    # Heuristic: try enough times to fill n_points, but avoid infinite loops
    max_attempts = n_points * 50 
    
    while len(sampled_points) < n_points and attempts < max_attempts:
        attempts += 1
        
        # Weighted sample triangle
        r = rng.uniform(0, total_area)
        idx = bisect.bisect_left(cumulative_areas, r)
        tri = triangles[min(idx, len(triangles)-1)]
        
        # Random point in triangle
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        
        w = 1 - u - v
        p = tri['p0'] * u + tri['p1'] * v + tri['p2'] * w
        
        # Check constraints
        
        # 1. Distance (Strict 50m)
        if (p[0] - room_center[0])**2 + (p[1] - room_center[1])**2 > max_dist**2:
            continue
        
        # 2. Z-level check (within 2m of room center)
        if abs(p[2] - room_range.GetMin()[2]) > 3.0:
            continue
            
        # 3. Not in Room
        if room_range and room_range.Contains(p):
            continue
            
        # 4. Not in Buildings
        in_building = False
        for bbox in building_bboxes:
            if bbox.Contains(p):
                in_building = True
                break
        if in_building:
            continue
            
        # Valid point
        sampled_points.append({
            "utm_x": p[0],
            "utm_y": p[1],
            "height": p[2],
            "lat": 0.0, # Dummy
            "lon": 0.0, # Dummy
            "width": 1.5,
            "heading": rng.uniform(0, 2*math.pi)
        })
        
    return sampled_points

def process_city(city, n_points, seed, input_folder):
    vc_plus_dir = os.path.join(input_folder, "vc_apr14", city)
    
    # Load USD to get room location
    usd_file = os.path.join(vc_plus_dir, f"{city}_gr_merged.usd")
    room_center = None
    room_range = None
    
    if os.path.exists(usd_file):
        print(f"Loading {usd_file}...")
        try:
            with SuppressFD():
                stage = Usd.Stage.Open(usd_file)
                room_prim = stage.GetPrimAtPath("/start_result_navigation")
                if not room_prim:
                    room_prim = stage.GetPrimAtPath("/World/start_result_navigation")
                
                if room_prim:
                    center_gf, room_range = get_bbox_center_and_range(room_prim)
                    room_center = [center_gf[0], center_gf[1], center_gf[2]]
                    print(f"  Found room center at: {room_center}")
                    
                    # Get Building BBoxes
                    building_bboxes = get_building_bboxes(stage, center_gf, 50.0)
                    print(f"  Found {len(building_bboxes)} buildings near room.")
                    
                    # Sample Mesh
                    mesh_parent_path = "/World/city/TerrainOther"
                    rng = random.Random(seed)
                    points = sample_points_from_mesh(stage, mesh_parent_path, center_gf, room_range, building_bboxes, n_points, rng, max_dist=50.0)
                    
                    return {
                        "city": city,
                        "sidewalk_segments": 0, # Legacy field
                        "sidewalk_length_m": 0, # Legacy field
                        "points": points,
                    }
                else:
                    print("  Warning: /start_result_navigation not found in USD.")
        except Exception as e:
            print(f"  Error reading USD: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  Warning: USD file not found at {usd_file}")

    return None

def main():
    parser = argparse.ArgumentParser(description="Generate random points on sidewalk/terrain.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True, 
        help="Input folder containing city data (e.g. ./data or /path/to/isaac_scenes_v1)"
    )
    args = parser.parse_args()

    # Scan for cities in input_folder/vc_apr14
    vc_plus_path = os.path.join(args.input_folder, "vc_apr14")
    if not os.path.exists(vc_plus_path):
        print(f"Error: {vc_plus_path} does not exist.")
        return

    city_list = []
    if os.path.isdir(vc_plus_path):
        for item in os.listdir(vc_plus_path):
            if os.path.isdir(os.path.join(vc_plus_path, item)) and item != "road_objects":
                city_list.append(item)
    
    print(f"Found cities: {city_list}")

    for city in city_list:
        print(f"Processing {city}...")
        result = process_city(city, N_POINTS, args.seed, args.input_folder)
        if not result or not result['points']:
            print(f"  No points generated for {city}")
            continue
        
        # Save directly to the city folder
        out_dir = os.path.join(vc_plus_path, city)
        out_path = os.path.join(out_dir, "sidewalk_points.json")
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {len(result['points'])} points to {out_path}")

if __name__ == "__main__":
    main()
