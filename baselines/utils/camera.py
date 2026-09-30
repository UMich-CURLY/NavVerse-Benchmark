import numpy as np
from scipy.spatial.transform import Rotation as R


def warp_to_pi(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi

class Camera:
    def __init__(self, args, no_init_pose=False):
        self.args = args
        self.last_pose = np.eye(4)
        self.initial_pose = None
        self.no_init_pose = no_init_pose

    def set_camera_matrix(self, width, height, hfov, vfov=None):
        self.cx = width / 2.
        self.cy = height / 2.
        self.fx = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
        if vfov is not None:
            self.fy = (height / 2.) / np.tan(np.deg2rad(vfov / 2.))
        else:
            self.fy = self.fx
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                      [0, self.fy, self.cy],
                                      [0, 0, 1]])

    def get_change_of_basis(self, backend):
        """S should be orthogonal (S.T@S = I)
        @note in foxglove, red is x axis, green is y, blue is z axis
        """
        S = np.eye(4)
        if backend == "habitat":
            # x (right), y (up), z (back) -> x (forward), y (left), z (up)
            S[:3,:3] = np.array([[0, 0, -1],
                                [-1, 0, 0],
                                [0, 1, 0]])
        elif backend == "opencv":
            # x (right), y (down), z (forward) -> x (forward), y (left), z (up)
            S[:3,:3] = np.array([[0, 0, 1],
                                [-1, 0, 0],
                                [0, -1, 0]])
        else:
            # real_world, isaac, tiamat use x (forward), y (left), z (up)
            S[:3,:3] = np.eye(3)
        return S
    
    @staticmethod
    def parse_pose(quat_xyzw=None, position=None, pose=None, rotation=None, yaw_angle=None, euler_order="ZYX"):
        new_pose = np.eye(4)
        if quat_xyzw is not None:
            new_pose[:3,:3] = R.from_quat(quat_xyzw).as_matrix()
        elif rotation is not None:
            new_pose[:3,:3] = rotation
        elif yaw_angle is not None:
            new_pose[:3,:3] = R.from_euler(euler_order[0], yaw_angle).as_matrix()
        if position is not None:
            new_pose[:3,3] = position
        if pose is not None:
            new_pose = pose
        return new_pose
    
    def get_pose_agent(self, quat_xyzw=None, position=None, pose=None, rotation=None, yaw_angle=None, T_body_cam=None, backend=None, child_frame_id=None):
        """convert pose from backend coordinate system to agent coordinate system
        @note this function will also update the initial pose and last pose
        """
        args = self.args
        if backend is None:
            backend = args.backend
        # get current pose
        curr_pose = self.parse_pose(quat_xyzw=quat_xyzw, position=position, pose=pose, rotation=rotation, yaw_angle=yaw_angle)
        if self.no_init_pose:
            self.initial_pose = np.eye(4)
        else:
            # set initial pose
            if self.initial_pose is None:
                self.initial_pose = curr_pose
                self.last_pose = curr_pose
            else:
                # reject odometry update if the robot moves too far
                odom_translation = curr_pose[:3, 3] - self.last_pose[:3, 3]
                if np.linalg.norm(odom_translation) > 0.8:
                    rel_pose = curr_pose @ np.linalg.inv(self.last_pose)
                    self.initial_pose = rel_pose @ self.initial_pose
        self.last_pose = curr_pose
        # convert to agent coordinate system
        S1 = self.get_change_of_basis(backend)
        if child_frame_id is not None:
            S2 = self.get_change_of_basis("opencv" if "optical" in child_frame_id else backend)
        else:
            S2 = self.get_change_of_basis(backend)
        camera_pose = np.linalg.inv(self.initial_pose) @ curr_pose
        if T_body_cam is not None:
            camera_pose = camera_pose @ T_body_cam
        camera_pose =  S1 @ camera_pose @ S2.T
        if backend == "real_world":
            # z drift is large in real world, ignore it here
            camera_pose[1, 3] = args.ROBOT_HEIGHT
        # if verbose:
        #     print("input pose:", curr_pose[:3, 3], R.from_matrix(curr_pose[:3, :3]).as_euler("xyz", degrees=True))
        #     print("transformed pose:", camera_pose[:3, 3], R.from_matrix(camera_pose[:3, :3]).as_euler("xyz", degrees=True))
        return camera_pose
    
    def get_pose_backend(self, quat_xyzw=None, position=None, pose=None, rotation=None, yaw_angle=None, backend=None):
        """convert pose from agent coordinate system to backend coordinate system"""
        args = self.args
        if backend is None:
            backend = args.backend
        agent_pose = self.parse_pose(quat_xyzw=quat_xyzw, position=position, pose=pose, rotation=rotation, yaw_angle=yaw_angle)
        S = np.linalg.inv(self.get_change_of_basis(backend))
        backend_pose = S @ agent_pose @ S.T
        if not self.no_init_pose:
            backend_pose = self.initial_pose @ backend_pose
        return backend_pose
    
    def convert_agent_pose_to_waypoint(self, agent_pose):
        """convert agent pose to absolute waypoints
        agent & isaac & real_world & tiamat: [forward, left, yaw], rotating from +x to +y is nagative for yaw
        """
        args = self.args
        if args.backend in ["real_world", "isaac", "tiamat"]:
            backend_pose = self.get_pose_backend(position=agent_pose[:3, 3], rotation=agent_pose[:3, :3])
            return [
                backend_pose[0, 3],
                backend_pose[1, 3],
                R.from_matrix(backend_pose[:3, :3]).as_euler("ZYX")[0]
            ]
        elif args.backend == "habitat":
            raise NotImplementedError("Habitat backend is not supported for waypoints")

def get_valid_mask(points=None, map_size=25.0, map_origin=None):
    mask = (points[:, 0] >= map_origin[0] - 0.5 * map_size) & \
            (points[:, 0] <= map_origin[0] + 0.5 * map_size) & \
            (points[:, 1] >= map_origin[1] - 0.5 * map_size) & \
            (points[:, 1] <= map_origin[1] + 0.5 * map_size)
    return mask

def get_pcd_height(points):
    return points[:, 2]
    
def project_to_map(points=None, colors=None, pcd=None, map_size=25.0, map_resolution=0.05, map_origin=None):
    """project 3D points to 2D bev image coordinates
    points: [x, y, z]
    colors: [r, g, b]
    pcd: open3d.geometry.PointCloud
    map_size: int, shape of bev map in meters
    map_resolution: float, meters per pixel
    map_origin: [x, y], origin of the map in meters
    """
    if points is not None:
        points = np.array(points)
    if colors is not None:
        colors = np.array(colors)
    if not pcd is None:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

    single_point = len(points.shape)==1
    if single_point:
        points_filtered = points.reshape(1, -1)
    else:
        mask = get_valid_mask(points, map_size, map_origin)
        points_filtered = points[mask]
        if colors is not None:
            colors_filtered = colors[mask]

    x_coords = np.floor((points_filtered[:, 0] - map_origin[0] + 0.5 * map_size) / map_resolution).astype(int)
    y_coords = np.floor((points_filtered[:, 1] - map_origin[1] + 0.5 * map_size) / map_resolution).astype(int)
    map_size_px = int(map_size / map_resolution)
    x_coords = np.clip(x_coords, 0, map_size_px-1)
    y_coords = np.clip(y_coords, 0, map_size_px-1)

    if single_point:
        x_coords = x_coords[0]
        y_coords = y_coords[0]

    if colors is not None:
        return x_coords, y_coords, colors_filtered
    else:
        return x_coords, y_coords

def back_project_to_3d(xy_coords, height, map_size, map_resolution, map_origin):
    """back project 2D bev coordinates to 3D points"""
    points = np.zeros((len(xy_coords), 3))
    map_size_px = int(map_size / map_resolution)
    points[:, 0] = (xy_coords[:, 0] - 0.5 * map_size_px) * map_resolution + map_origin[0]
    points[:, 1] = (xy_coords[:, 1] - 0.5 * map_size_px) * map_resolution + map_origin[1]
    points[:, 2] = height
    return points