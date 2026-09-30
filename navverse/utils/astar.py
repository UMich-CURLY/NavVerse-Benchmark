import heapq
import json
import math
from pathlib import Path
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def generate_kernels():
    """
    Generate 4 different 3x3 kernels where each has one 1 and one -1 at opposite ends.

    Returns:
        np.ndarray: An array of shape (4, 3, 3) containing the four kernels.
    """
    kernels = np.zeros((4, 3, 3))  # Initialize 4 kernels with zeros
    
    positions = [((0, 0), (2, 2)),  # Top-left to bottom-right
                ((0, 2), (2, 0)),  # Top-right to bottom-left
                ((0, 1), (2, 1)),  # Top-center to bottom-center
                ((1, 0), (1, 2))]  # Middle-left to middle-right

    for i, (pos1, pos2) in enumerate(positions):
        kernels[i, pos1[0], pos1[1]] = 1   # Set 1 at one end
        # kernels[i, 1, 1] = 1   # Set 1 at one end
        kernels[i, pos2[0], pos2[1]] = -1  # Set -1 at the opposite end
    
    return kernels

def load_bev_map(bev_map_path):
    bev_map_path = str(bev_map_path)
    bev_data = np.load(bev_map_path)
    bev_depth = bev_data['depth'][:, :, 0]
    with open(f"{bev_map_path.replace('.npz', '_info.txt')}") as f:
        info = json.load(f)
    return bev_depth, info

def remove_small_obstacle_components(obstacle_map, min_component_area_px=0):
    if min_component_area_px <= 1:
        return obstacle_map.astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(obstacle_map.astype(np.uint8), 8)
    filtered = np.zeros_like(obstacle_map, dtype=np.uint8)
    for component_id in range(1, component_count):
        if stats[component_id, cv2.CC_STAT_AREA] >= min_component_area_px:
            filtered[labels == component_id] = 1
    return filtered


def generate_obstacle_map(bev_depth, max_climb_height=0.25, min_component_area_px=0):
    image_width, image_height = bev_depth.shape
    height_map = np.ones((image_width, image_height))*(-1000.0)
    valid_depth = np.isfinite(bev_depth) & (bev_depth > 0)
    height_map[valid_depth] = bev_depth[valid_depth]

    # hand desgined gradient kernel that has physical unit
    kernels = generate_kernels()
    filtered_img = []
    for kernel in kernels:
        img = cv2.filter2D(src=height_map, ddepth=-1, kernel=kernel)
        filtered_img.append(img)
    filtered_img = np.stack(filtered_img)  # Shape: (4, height, width)
    # take the max across the 4 kernels to get the final gradient map
    gradient_map = np.max(np.abs(filtered_img), axis=0)

    # calculate obstacle map
    obstacle_map = (gradient_map > max_climb_height).astype(np.uint8)
    obstacle_map = remove_small_obstacle_components(obstacle_map, min_component_area_px)

    return obstacle_map


def inflate_obstacle_map(obstacle_map, clearance_m, meters_per_cell):
    if clearance_m <= 0.0:
        return obstacle_map
    radius_px = int(np.ceil(clearance_m / meters_per_cell))
    if radius_px <= 0:
        return obstacle_map
    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(obstacle_map.astype(np.uint8), kernel, iterations=1)


class AStarPlanner:
    def __init__(
        self,
        meters_per_cell=0.05,
        step_size=0.1,
        step_size_yaw=45,
        reach_threshold=1.0,
        reach_yaw=45.0,
        robot_width=0.4,
        robot_height=0.8,
        preferred_clearance_m=0.45,
        clearance_cost=80.0,
    ):

        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.step_size_px = max(1, int(round(step_size / meters_per_cell)))
        self.step_size_yaw = step_size_yaw
        self.reach_threshold = max(1, int(round(reach_threshold / meters_per_cell)))
        self.reach_yaw = np.deg2rad(reach_yaw)
        self.robot_width = robot_width # meters
        self.robot_height = robot_height # meters
        self.preferred_clearance_m = preferred_clearance_m
        self.clearance_cost = clearance_cost

        # resolution, default 5cm per cell
        self.meters_per_cell = meters_per_cell

        # conservative circle radius: half diagonal (rectangle -> circumscribed circle)
        self.robot_radius = 0.5 * np.hypot(self.robot_width, self.robot_height)
        # when farther than this margin, we skip expensive rectangle test
        self.safe_margin = self.robot_radius * 1.15  # 15% cushion

    def plan(self, occupancy_grid, start, goal, time_budget=0.5):
        self.last_plan_reached_goal = False
        # compute nearest distance to obstacle
        dist_px = distance_transform_edt(occupancy_grid.astype(np.uint8) == 0)
        self.dist_m = dist_px * self.meters_per_cell

        # Adjust start and goal yaw
        start = self.find_closest_yaw(start)
        start = self.find_noncollide_neighbor(start, occupancy_grid)
        goal = self.find_closest_yaw(goal)
        goal = self.find_noncollide_neighbor(goal, occupancy_grid)

        # start A* planning
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        cost_so_far = {self.get_id(start): 0} # g_cost
        all_nodes = set()
        start_time = time.time()

        while open_list:
            if time.time() - start_time > time_budget:
                print("Time budget exceeded, returning furthest node")
                break
            _, current_node = heapq.heappop(open_list)

            distance_to_goal = np.linalg.norm(np.array(current_node[:2]) - np.array(goal[:2]))
            yaw_diff = np.abs(self.warp_to_pi(current_node[2] - goal[2]))
            if distance_to_goal<self.reach_threshold and yaw_diff<self.reach_yaw:
                # reached goal!
                came_from[goal] = current_node
                raw_path = self.reconstruct_path(came_from, start, goal)
                self.raw_path = raw_path
                self.final_path = self.simplify_path(raw_path, occupancy_grid)
                self.last_plan_reached_goal = True
                return self.final_path

            current_cost = cost_so_far[self.get_id(current_node)]
            for neighbor in self.get_neighbors(current_node, occupancy_grid):
                new_cost = current_cost + self.g_cost(current_node, neighbor)
                new_cost += self.clearance_penalty(neighbor)
                dist = np.linalg.norm([neighbor[1] - current_node[1], neighbor[0] - current_node[0]])
                if dist >= 0.5*self.step_size_px:
                    heading_angle = np.arctan2(neighbor[1] - current_node[1], neighbor[0] - current_node[0])
                    angle_diff = np.abs(self.warp_to_pi(heading_angle - neighbor[2]))
                    new_cost += angle_diff*50.0
                # new_cost = current_cost + self.g_cost(current_node, neighbor) + angle_diff*5.0
                # print(f"current node: {current_node}, cost to goal: {self.g_cost(neighbor, goal)}")
                neighbor_grid = self.get_id(neighbor)

                if neighbor_grid not in cost_so_far or new_cost < cost_so_far[neighbor_grid]:
                    cost_so_far[neighbor_grid] = new_cost
                    all_nodes.add(neighbor)
                    priority = new_cost + self.h_cost(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current_node
        print("No path found, returning furthest node")
        if len(all_nodes) == 0:
            return
        best_node = min(all_nodes, key=lambda x: self.h_cost(x, goal))
        raw_path = self.reconstruct_path(came_from, start, best_node)
        self.raw_path = raw_path

        # process the path
        print(f"A* planning time: {time.time() - start_time} seconds")
        if raw_path is None:
            print("A* failed to find a path")
            return
        start_time = time.time()
        final_path = self.simplify_path(raw_path, occupancy_grid)
        self.final_path = final_path
        return final_path

    def get_neighbors(self, node, occupancy_grid, relax=False, collision_check=True):
        neighbors = []
        step_size_px = self.step_size_px
        step_size_yaw = self.step_size_yaw
        for rel_yaw in range(-step_size_yaw, step_size_yaw+1, step_size_yaw):
            new_yaw = np.deg2rad(rel_yaw)+node[2]
            dx = np.cos(new_yaw)*step_size_px
            dy = np.sin(new_yaw)*step_size_px
            # rotate and forward one step size
            neighbors.append((dx, dy,  np.deg2rad(rel_yaw)))
            if rel_yaw != 0:
                # pure rotation
                neighbors.append((0, 0, np.deg2rad(rel_yaw)))
        # backward one step size
        # neighbors.append((0, -step_size_px, 0))

        if collision_check:
            if relax:
                check_func = lambda x: self.check_collision(x, occupancy_grid, relax=relax)
            else:
                check_func = lambda x: self.check_collision(x, occupancy_grid, relax=relax, parent_node=node)
        else:
            check_func = lambda x: False
        result = []
        for dx, dy, dw in neighbors:
            neighbor = (node[0] + dx, node[1] + dy, self.warp_to_pi(node[2] + dw))
            if 0 <= neighbor[0] < occupancy_grid.shape[0] and 0 <= neighbor[1] < occupancy_grid.shape[1]:
                if not check_func(neighbor):
                    # Only traverse free space
                    result.append(neighbor)
        # print(f"{len(neighbors)}->{len(result)}")
        # print(f"neighbors: {node}->{result}")
        return result
    
    def find_closest_yaw(self, node):
        yaw_list = [np.deg2rad(yaw) for yaw in range(-180, 180+1, self.step_size_yaw)]
        best_yaw = np.argmin([np.abs(self.warp_to_pi(yaw - node[2])) for yaw in yaw_list])
        return (node[0], node[1], yaw_list[best_yaw])

    def find_noncollide_neighbor(self, start, occupancy_grid):
        max_attempts = 5
        open_list = []
        visited_list = []

        start_hit = self.check_collision_roi(start, occupancy_grid)[1]
        heapq.heappush(open_list, (start_hit, start))
        best_node = start
        best_hit = start_hit

        def get_neighbors(node):
            return [
                (node[0] + self.step_size_px, node[1], node[2]),
                (node[0] - self.step_size_px, node[1], node[2]),
                (node[0], node[1] + self.step_size_px, node[2]),
                (node[0], node[1] - self.step_size_px, node[2]),
                # (node[0], node[1], node[2] + self.step_size_yaw),
                # (node[0], node[1], node[2] - self.step_size_yaw),
            ]

        for i in range(max_attempts):
            if best_hit == 0 or len(open_list) == 0:
                break
            node = heapq.heappop(open_list)[1]
            visited_list.append(self.get_id(node))
            neighbors = get_neighbors(node)
            for neighbor in neighbors:
                if self.get_id(neighbor) not in visited_list:
                    hit = self.check_collision_roi(neighbor, occupancy_grid, robot_width=self.robot_width, robot_height=self.robot_height)[1]
                    if hit < best_hit:
                        best_hit = hit
                        best_node = neighbor
                    heapq.heappush(open_list, (hit, neighbor))
        
        # print(f"start: {start}, start hit: {start_hit}, best node: {best_node}, best hit: {best_hit}")
        return tuple(best_node)

    def check_collision_roi(self, node, obstacle_map, parent_node=None, robot_width=None, robot_height=None):
        # Build rotated rect
        if robot_width is None:
            robot_width = self.robot_width
        if robot_height is None:
            robot_height = self.robot_height
        robot_w = max(1, int(round(robot_width / self.meters_per_cell)))
        robot_h = max(1, int(round(robot_height / self.meters_per_cell)))
        rect = (node[:2], (robot_w, robot_h), np.rad2deg(node[2]) + 90.0)

        # check out of bounds
        box = cv2.boxPoints(rect)
        
        if parent_node is not None:
            rect_parent = (parent_node[:2], (robot_w, robot_h), np.rad2deg(parent_node[2]) + 90.0)
            box_parent = cv2.boxPoints(rect_parent)
            
            # Ensure both boxes are float32 before concatenation for convexHull
            box = box.astype(np.float32)
            box_parent = box_parent.astype(np.float32)
            
            # ConvexHull expects float32 or int32 points
            box = cv2.convexHull(np.concatenate((box, box_parent)))
            box = box.reshape(-1, 2)

        # Convert to integer coordinates for boundingRect
        box_int = np.int32(box)
        x, y, w, h = cv2.boundingRect(box_int)
        
        H, W = obstacle_map.shape[:2]
        if x >= W or y >= H or x + w <= 0 or y + h <= 0:
            return True
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return True

        # Crop ROI once; build a local mask by shifting polygon into ROI coordinates
        roi = obstacle_map[y0:y1, x0:x1]
        
        # Ensure box is in correct format for fillConvexPoly
        # It needs to be integer coordinates relative to the ROI
        box_shifted = (box - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, box_shifted, 255)
        hit = cv2.countNonZero((roi > 0).astype(np.uint8) & (mask > 0))
        return hit>5, hit, roi, mask

    def crop_minAreaRect(self, img, rect):
        # check out of bounds
        box = cv2.boxPoints(rect)
        x, y, w, h = cv2.boundingRect(box)
        H, W = img.shape[:2]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)

        # Crop ROI once; build a local mask by shifting polygon into ROI coordinates
        roi = img[y0:y1, x0:x1]
        box_shifted = (box - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, box_shifted, 255)
        return roi
    

    def check_collision(self, node, obstacle_map, relax=False, parent_node=None):
        """Check if the node is in collision with the obstacle map.
        Return True if in collision, False otherwise.
        """
        # r = int(round(node[1]))
        # c = int(round(node[0]))
        r = int(round(node[1]))
        c = int(round(node[0]))
        if r < 0 or r >= self.dist_m.shape[0] or c < 0 or c >= self.dist_m.shape[1]:
            return True  # out of bounds collides
        # cheap early-out: far from obstacles => no collision
        if parent_node is None and self.dist_m[r, c] >= self.safe_margin:
            return False

        # skip roi check if relax is True
        if relax:
            return False

        # near obstacles => do exact rectangle test on original grid
        return self.check_collision_roi(node, obstacle_map.astype(np.uint8), parent_node=parent_node)[0]

    def plot_rect(self, node, obstacle_map, color=100, width=1):
        robot_w = max(1, int(round(self.robot_width / self.meters_per_cell)))
        robot_h = max(1, int(round(self.robot_height / self.meters_per_cell)))
        rect = (node[:2], (robot_w, robot_h), np.rad2deg(node[2])+90)
        # flip obstacle_map vertically
        wall = self.crop_minAreaRect(obstacle_map, rect)
        vis_img = obstacle_map.copy()
        if vis_img.dtype == bool:
            vis_img = vis_img.astype(np.uint8) * 255
        elif vis_img.max() <= 1:
            vis_img = (vis_img*255).astype(np.uint8)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(vis_img,[box],0,color,1)
        return vis_img, wall
    
    def get_id(self, node):
        return (int(node[0]), int(node[1]), round(np.rad2deg(node[2])))
        # return (int(node[0]), int(node[1]))

    def g_cost(self, p1, p2):
        diff = [p1[0]-p2[0], p1[1]-p2[1], self.warp_to_pi(p1[2]-p2[2])]
        diff[2] *= 10
        return np.linalg.norm(diff)

    def clearance_penalty(self, node):
        if self.clearance_cost <= 0.0 or self.preferred_clearance_m <= 0.0:
            return 0.0
        r = int(round(node[1]))
        c = int(round(node[0]))
        if r < 0 or r >= self.dist_m.shape[0] or c < 0 or c >= self.dist_m.shape[1]:
            return self.clearance_cost
        deficit = max(0.0, self.preferred_clearance_m - float(self.dist_m[r, c]))
        if deficit <= 0.0:
            return 0.0
        return self.clearance_cost * deficit * deficit

    def h_cost(self, p1, p2):
        return np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]])

    def reconstruct_path(self, came_from, start, goal):
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path
    
    def warp_to_pi(self, yaw):
        return (yaw + np.pi) % (2 * np.pi) - np.pi

    def _world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid indices."""
        grid_x = int((world_x - self.map_origin_x) / self.meters_per_cell)
        grid_y = int((world_y - self.map_origin_y) / self.meters_per_cell)
        return (grid_y, grid_x)  # Note: A* expects (row, col)

    def _grid_to_world(self, grid_row, grid_col):
        """Convert grid indices to world coordinates."""
        # world_x = self.map_origin_x + grid_col * self.meters_per_cell
        # world_y = self.map_origin_y + grid_row * self.meters_per_cell
        world_x = self.map_origin_x + grid_col * 0.25
        world_y = self.map_origin_y + grid_row * 0.25

        return (world_x, world_y)

    def simplify_path(self, path, occupancy_grid):
        """Remove unnecessary points from the path."""
        if len(path) < 3:
            return path
        
        simplified = [path[0]]  # Always keep start point
        
        i = 0
        while i < len(path) - 1:
            furthest = i + 1        # Look ahead to see how far we can go in a straight line
            
            for j in range(i + 2, min(i + 6, len(path))):  # Look ahead max 5 points
                if self._can_connect_directly(path[i], path[j], occupancy_grid) and path[i][2] == path[j][2]:
                    furthest = j
                else:
                    break
            
            simplified.append(path[furthest])
            i = furthest
        
        return simplified

    def _can_connect_directly(self, point1, point2, occupancy_grid, interval=2.0):
        """Check if two points can be connected with a straight line (no obstacles)."""
        point1, point2 = np.array(point1), np.array(point2)
        
        # Sample points along the line between point1 and point2
        dist = np.linalg.norm(point2[:2] - point1[:2])
        steps = dist/interval
        
        for i in range(1, int(steps)):  # Skip start and end points
            t = i / steps
            x, y, yaw = point1 + t*(point2 - point1)
            if self.check_collision((x, y, yaw), occupancy_grid):
                return False
        return True
    
    def _create_waypoints(self, path_points, spacing=0.1):
        """Create evenly spaced waypoints along the path."""
        if len(path_points) < 2:
            # Single point path
            world_x, world_y = self._grid_to_world(path_points[0][0], path_points[0][1])
            return [(world_x, world_y)]
        
        # Convert all points to world coordinates
        world_path = []
        for row, col, yaw in path_points:
            world_x, world_y = self._grid_to_world(col, row)
            world_path.append(np.array([world_x, world_y, yaw]))
        return world_path


class BevAStarReferencePlanner:
    def __init__(
        self,
        bev_map_path,
        max_climb_height=0.25,
        robot_width=0.5,
        robot_height=0.9,
        step_size=0.2,
        step_size_yaw=45,
        reach_threshold=0.5,
        reach_yaw=45.0,
        clearance_m=0.0,
        preferred_clearance_m=0.45,
        clearance_cost=80.0,
        min_obstacle_area_px=0,
    ):
        self.bev_map_path = Path(bev_map_path)
        self.bev_depth, self.info = load_bev_map(self.bev_map_path)
        self._init_planner(
            max_climb_height=max_climb_height,
            robot_width=robot_width,
            robot_height=robot_height,
            step_size=step_size,
            step_size_yaw=step_size_yaw,
            reach_threshold=reach_threshold,
            reach_yaw=reach_yaw,
            clearance_m=clearance_m,
            preferred_clearance_m=preferred_clearance_m,
            clearance_cost=clearance_cost,
            min_obstacle_area_px=min_obstacle_area_px,
        )

    @classmethod
    def from_bev_depth(
        cls,
        bev_depth,
        info,
        max_climb_height=0.25,
        robot_width=0.5,
        robot_height=0.9,
        step_size=0.2,
        step_size_yaw=45,
        reach_threshold=0.5,
        reach_yaw=45.0,
        clearance_m=0.0,
        preferred_clearance_m=0.45,
        clearance_cost=80.0,
        min_obstacle_area_px=0,
    ):
        obj = cls.__new__(cls)
        obj.bev_map_path = None
        obj.bev_depth = bev_depth
        obj.info = info
        obj._init_planner(
            max_climb_height=max_climb_height,
            robot_width=robot_width,
            robot_height=robot_height,
            step_size=step_size,
            step_size_yaw=step_size_yaw,
            reach_threshold=reach_threshold,
            reach_yaw=reach_yaw,
            clearance_m=clearance_m,
            preferred_clearance_m=preferred_clearance_m,
            clearance_cost=clearance_cost,
            min_obstacle_area_px=min_obstacle_area_px,
        )
        return obj

    def _init_planner(
        self,
        max_climb_height,
        robot_width,
        robot_height,
        step_size,
        step_size_yaw,
        reach_threshold,
        reach_yaw,
        clearance_m,
        preferred_clearance_m,
        clearance_cost,
        min_obstacle_area_px,
    ):
        if self.bev_depth.ndim == 3:
            self.bev_depth = self.bev_depth[:, :, 0]
        self.px_per_meter = float(np.asarray(self.info["px_per_meter"]).mean())
        self.world_center = np.asarray(self.info["world_center"], dtype=float)
        self.meters_per_cell = 1.0 / self.px_per_meter
        self.raw_obstacle_map = generate_obstacle_map(
            self.bev_depth,
            max_climb_height=max_climb_height,
            min_component_area_px=min_obstacle_area_px,
        )
        self.obstacle_map = inflate_obstacle_map(self.raw_obstacle_map, clearance_m, self.meters_per_cell)
        self.image_height, self.image_width = self.obstacle_map.shape
        self.planner = AStarPlanner(
            meters_per_cell=self.meters_per_cell,
            robot_width=robot_width,
            robot_height=robot_height,
            step_size=step_size,
            step_size_yaw=step_size_yaw,
            reach_threshold=reach_threshold,
            reach_yaw=reach_yaw,
            preferred_clearance_m=preferred_clearance_m,
            clearance_cost=clearance_cost,
        )

    def world_to_grid(self, point, yaw=0.0):
        point = np.asarray(point, dtype=float)
        xy_px = (point[:2] - self.world_center[:2]) * self.px_per_meter
        return (
            float(xy_px[0] + self.image_width / 2.0),
            float(self.image_height / 2.0 - xy_px[1]),
            float(yaw),
        )

    def grid_to_world(self, node, z):
        x_px, y_px = node[:2]
        world_x = (x_px - self.image_width / 2.0) / self.px_per_meter + self.world_center[0]
        world_y = (self.image_height / 2.0 - y_px) / self.px_per_meter + self.world_center[1]
        return np.array([world_x, world_y, z], dtype=float)

    def _grid_yaw_between(self, start, goal):
        start_grid_xy = self.world_to_grid(start)[:2]
        goal_grid_xy = self.world_to_grid(goal)[:2]
        return float(np.arctan2(goal_grid_xy[1] - start_grid_xy[1], goal_grid_xy[0] - start_grid_xy[0]))

    def plan_reference_path(self, start_pos, goal_pos, time_budget=1.0):
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        grid_yaw = self._grid_yaw_between(start_pos, goal_pos)
        start = self.world_to_grid(start_pos, yaw=grid_yaw)
        goal = self.world_to_grid(goal_pos, yaw=grid_yaw)
        path = self.planner.plan(self.obstacle_map, start, goal, time_budget=time_budget)
        if not path:
            return np.empty((0, 3), dtype=float)
        z = float(start_pos[2]) if start_pos.shape[0] > 2 else 0.0
        world_path = np.array([self.grid_to_world(node, z) for node in path], dtype=float)
        if len(world_path) > 0:
            world_path[0, 2] = z
        return world_path

    def plan_reference_path_2d(
        self,
        start_pos,
        goal_pos,
        time_budget=0.8,
        resolution_m=0.05,
        clearance_m=0.25,
        snap_distance_m=1.0,
        footprint_check_neighbors=False,
        collision_spacing_m=0.15,
    ):
        start_time = time.time()
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        self._ensure_distance_map()
        raw_dist_m = distance_transform_edt(self.raw_obstacle_map.astype(np.uint8) == 0) * self.meters_per_cell

        factor = max(1, int(round(float(resolution_m) / self.meters_per_cell)))
        coarse_resolution_m = self.meters_per_cell * factor
        height = (raw_dist_m.shape[0] // factor) * factor
        width = (raw_dist_m.shape[1] // factor) * factor
        if height <= 0 or width <= 0:
            return np.empty((0, 3), dtype=float)
        coarse_dist = raw_dist_m[:height, :width].reshape(
            height // factor,
            factor,
            width // factor,
            factor,
        ).min(axis=(1, 3))

        def world_to_coarse(point):
            x, y, _ = self.world_to_grid(point)
            row = int(round(y)) // factor
            col = int(round(x)) // factor
            return (
                int(np.clip(row, 0, coarse_dist.shape[0] - 1)),
                int(np.clip(col, 0, coarse_dist.shape[1] - 1)),
            )

        def coarse_to_world(cell):
            row, col = cell
            x_px = col * factor + 0.5 * factor
            y_px = row * factor + 0.5 * factor
            return self.grid_to_world((x_px, y_px, 0.0), z)

        def nearest_free(cell):
            row, col = cell
            if coarse_dist[row, col] >= clearance_m:
                return cell
            max_radius = max(1, int(math.ceil(float(snap_distance_m) / coarse_resolution_m)))
            for radius in range(1, max_radius + 1):
                row0 = max(0, row - radius)
                row1 = min(coarse_dist.shape[0] - 1, row + radius)
                col0 = max(0, col - radius)
                col1 = min(coarse_dist.shape[1] - 1, col + radius)
                candidates = np.argwhere(coarse_dist[row0 : row1 + 1, col0 : col1 + 1] >= clearance_m)
                if len(candidates) == 0:
                    continue
                candidates = candidates + np.array([row0, col0])
                best_idx = int(np.argmin(np.sum((candidates - np.array([row, col])) ** 2, axis=1)))
                return tuple(int(value) for value in candidates[best_idx])
            return None

        start_cell = nearest_free(world_to_coarse(start_pos))
        goal_cell = nearest_free(world_to_coarse(goal_pos))
        if start_cell is None or goal_cell is None:
            return np.empty((0, 3), dtype=float)

        z = float(start_pos[2]) if start_pos.shape[0] > 2 else 0.0
        edge_collision_cache = {}

        def edge_is_safe(current, neighbor):
            if not footprint_check_neighbors:
                return True
            key = (current, neighbor)
            if key not in edge_collision_cache:
                segment = np.asarray([coarse_to_world(current), coarse_to_world(neighbor)], dtype=float)
                edge_collision_cache[key] = self.is_world_path_collision_free(
                    segment,
                    spacing_m=float(collision_spacing_m),
                )
            return edge_collision_cache[key]

        moves = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        open_list = [(0.0, start_cell)]
        came_from = {}
        cost_so_far = {start_cell: 0.0}
        best_cell = start_cell
        best_heuristic = math.hypot(start_cell[0] - goal_cell[0], start_cell[1] - goal_cell[1])
        reached = False
        end_cell = None

        while open_list:
            if time.time() - start_time > time_budget:
                break
            _, current = heapq.heappop(open_list)
            heuristic = math.hypot(current[0] - goal_cell[0], current[1] - goal_cell[1])
            if heuristic < best_heuristic:
                best_heuristic = heuristic
                best_cell = current
            if heuristic <= 1.5:
                reached = True
                end_cell = current
                break
            for drow, dcol, move_cost in moves:
                neighbor = (current[0] + drow, current[1] + dcol)
                if (
                    neighbor[0] < 0
                    or neighbor[0] >= coarse_dist.shape[0]
                    or neighbor[1] < 0
                    or neighbor[1] >= coarse_dist.shape[1]
                ):
                    continue
                if coarse_dist[neighbor] < clearance_m:
                    continue
                if drow != 0 and dcol != 0:
                    if coarse_dist[current[0] + drow, current[1]] < clearance_m:
                        continue
                    if coarse_dist[current[0], current[1] + dcol] < clearance_m:
                        continue
                if not edge_is_safe(current, neighbor):
                    continue
                clearance_deficit = max(0.0, self.planner.preferred_clearance_m - float(coarse_dist[neighbor]))
                new_cost = cost_so_far[current] + move_cost + 3.0 * clearance_deficit * clearance_deficit
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    priority = new_cost + math.hypot(neighbor[0] - goal_cell[0], neighbor[1] - goal_cell[1])
                    heapq.heappush(open_list, (priority, neighbor))

        if not reached:
            if best_heuristic * coarse_resolution_m > max(0.75, float(snap_distance_m)):
                return np.empty((0, 3), dtype=float)
            end_cell = best_cell

        cells = [end_cell]
        while cells[-1] != start_cell:
            cells.append(came_from[cells[-1]])
        cells.reverse()

        world_path = []
        for cell in cells:
            world_path.append(coarse_to_world(cell))
        world_path = np.asarray(world_path, dtype=float)

        if len(world_path) > 0 and self.is_world_path_collision_free(np.asarray([start_pos, world_path[0]], dtype=float)):
            world_path = np.vstack([start_pos, world_path])
        elif len(world_path) > 0:
            return np.empty((0, 3), dtype=float)
        if len(world_path) > 0 and self.is_world_path_collision_free(np.asarray([world_path[-1], goal_pos], dtype=float)):
            world_path = np.vstack([world_path, goal_pos])
        return world_path

    def direct_world_path(self, start_pos, goal_pos, spacing=0.4):
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        distance = float(np.linalg.norm(goal_pos[:2] - start_pos[:2]))
        if distance <= 1e-6:
            return np.asarray([goal_pos], dtype=float)
        steps = max(1, int(np.ceil(distance / spacing)))
        return np.asarray([start_pos + (goal_pos - start_pos) * (i / steps) for i in range(steps + 1)], dtype=float)

    def _sample_world_path(self, world_path, spacing_m=0.2):
        world_path = np.asarray(world_path, dtype=float)
        if len(world_path) < 2:
            return world_path
        sampled = [world_path[0]]
        spacing_m = max(0.05, float(spacing_m))
        for start, goal in zip(world_path[:-1], world_path[1:]):
            segment = self.direct_world_path(start, goal, spacing=spacing_m)
            sampled.extend(segment[1:])
        return np.asarray(sampled, dtype=float)

    def world_path_clearance(self, world_path):
        self._ensure_distance_map()
        world_path = self._sample_world_path(world_path)
        clearances = []
        for point in world_path:
            x, y, _ = self.world_to_grid(point)
            r = int(round(y))
            c = int(round(x))
            if r < 0 or r >= self.planner.dist_m.shape[0] or c < 0 or c >= self.planner.dist_m.shape[1]:
                return 0.0, 0.0
            clearances.append(float(self.planner.dist_m[r, c]))
        if not clearances:
            return 0.0, 0.0
        return min(clearances), float(np.mean(clearances))

    def is_world_path_collision_free(self, world_path, spacing_m=0.2):
        self._ensure_distance_map()
        world_path = np.asarray(world_path, dtype=float)
        if len(world_path) < 2:
            return True
        for start, goal in zip(world_path[:-1], world_path[1:]):
            yaw = self._grid_yaw_between(start, goal)
            for point in self.direct_world_path(start, goal, spacing=spacing_m)[1:]:
                x, y, _ = self.world_to_grid(point, yaw=yaw)
                if self.planner.check_collision((x, y, yaw), self.obstacle_map):
                    return False
        return True

    def _ensure_distance_map(self):
        if not hasattr(self.planner, "dist_m"):
            dist_px = distance_transform_edt(self.obstacle_map.astype(np.uint8) == 0)
            self.planner.dist_m = dist_px * self.meters_per_cell


if __name__ == "__main__":
    # Create a 100x100 occupancy grid (1 = obstacle, 0 = free)
    occupancy_grid_map = np.zeros((100, 100), dtype=np.uint8)

    ## Add a few large obstacle clusters
    occupancy_grid_map[2:20, 16:50] = 1
    occupancy_grid_map[50:75, 40:75] = 1
    occupancy_grid_map[70:98, 6:36] = 1
    occupancy_grid_map[10:15, 70:78] = 1
    occupancy_grid_map[48:55, 75:85] = 1
    occupancy_grid_map[15:25, 50:55] = 1

    ## Add boundary walls
    occupancy_grid_map[0, :] = 1
    occupancy_grid_map[-1, :] = 1
    occupancy_grid_map[:, 0] = 1
    occupancy_grid_map[:, -1] = 1

    # ----------------------------------------------------------
    # Initialize A* planner
    astar = AStarPlanner(meters_per_cell=0.05)
    astar.occupancy_grid = occupancy_grid_map == 0

    plt.figure()
    plt.imshow(occupancy_grid_map, cmap='binary')
    plt.title("Inflated Obstacles With Original Boundaries in Grid Frame")
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.colorbar(label="Occupancy")

    ## Plot original boundaries
    obstacle_mask = (occupancy_grid_map > 0).astype(np.uint8)
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    label_added = False
    for contour in contours:
        contour = contour.squeeze(axis=1)
        if contour.ndim == 2 and contour.shape[0] > 1:
            xs = np.r_[contour[:, 0], contour[0, 0]]
            ys = np.r_[contour[:, 1], contour[0, 1]]
            plt.plot(
                xs,
                ys,
                color='cyan',
                linewidth=1.2,
                label='Original Boundary' if not label_added else None,
            )
            label_added = True

    start = (15,40, np.deg2rad(0))
    goal = (85, 85, np.deg2rad(90))

    vis_img = astar.occupancy_grid.copy()
    for point in [start, goal]:
        vis_img, wall = astar.plot_rect(point, vis_img)
    plt.imshow(vis_img, cmap='binary')
    for x, y, yaw in [start, goal]:
        plt.quiver(x, y, np.cos(yaw), np.sin(yaw),
                angles='xy', scale_units='xy', scale=0.1, color='blue', width=0.01, label='Yaw Direction')
    plt.show()

    print("in utils.py, the start is", start, "and goal is ", goal)
    path = astar.plan(occupancy_grid_map, start, goal)

    print(f"path: {np.array(path).shape}")
    # path = astar.plan(start, goal)
    vis_img = astar.occupancy_grid.copy()
    for x, y, yaw in astar.path_grid:
        vis_img, wall = astar.plot_rect((x, y, yaw), vis_img, 30)
    plt.imshow(vis_img, cmap='binary')
    if astar.path_grid:
        path = np.array(astar.path_grid)
        plt.scatter(path[:, 0], path[:, 1], color='red', s=3, label='Planned Path')
        plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1.5)
        plt.scatter(start[0], start[1], color='blue', s=50, marker='.', label='Start')
        plt.scatter(goal[0], goal[1], color='blue', s=50, marker='*', label='Goal')
        plt.legend()

        x,y,yaw = path[:,0], path[:,1], path[:,2]

        plt.quiver(x, y, np.cos(yaw), np.sin(yaw), np.arange(len(yaw)),
                angles='xy', scale_units='xy', scale=0.1, cmap='viridis', width=0.01, label='Yaw Direction')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.axis("equal")
        plt.title("Waypoints with Yaw Directions in World Frame")
        plt.show()
