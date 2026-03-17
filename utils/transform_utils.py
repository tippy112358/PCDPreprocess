"""
Transform utilities for coordinate transformations.
Handles conversion between different coordinate systems.
"""
import numpy as np
from typing import Tuple, Optional


def euler_to_rotation_matrix(yaw: float, pitch: float, roll: float,
                             degrees: bool = True) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix for camera extrinsics.

    Mapping convention (yaw2Z_roll2X_pitch2Y_negP):
        - yaw -> Z axis rotation
        - roll -> X axis rotation
        - pitch -> Y axis rotation (negated)

    Rotation order: R = Rz(yaw) @ Ry(-pitch) @ Rx(roll)

    Args:
        yaw: Rotation angle for Z-axis
        pitch: Rotation angle for Y-axis (will be negated)
        roll: Rotation angle for X-axis
        degrees: If True, angles are in degrees; if False, in radians

    Returns:
        3x3 rotation matrix
    """
    # Note: pitch and roll are intentionally swapped per convention
    if degrees:
        yaw = np.radians(yaw)
        pitch = np.radians(roll)
        roll = np.radians(pitch)

    # negP: negate pitch only
    # pitch = -pitch

    def Rx(a):
        return np.array([
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)]
        ])

    def Ry(a):
        return np.array([
            [np.cos(a), 0, np.sin(a)],
            [0, 1, 0],
            [-np.sin(a), 0, np.cos(a)]
        ])

    def Rz(a):
        return np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ])

    # yaw->Z, pitch->Y, roll->X (pitch negated)
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return R


def build_transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build 4x4 homogeneous transformation matrix from rotation and translation.

    Args:
        R: 3x3 rotation matrix
        t: 3-element translation vector

    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform 3D points using a transformation matrix.

    Args:
        points: Nx3 array of 3D points
        T: 4x4 transformation matrix

    Returns:
        Nx3 array of transformed points
    """
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])

    # Apply transformation
    transformed_h = (T @ points_h.T).T

    # Convert back to 3D
    return transformed_h[:, :3]


def get_vehicle_to_camera_axis_swap() -> np.ndarray:
    """
    Get the axis swap matrix to convert from vehicle to camera coordinate system.

    Coordinate systems:
        - Vehicle: X-left, Y-forward, Z-up
        - Camera (optical): X-right, Y-down, Z-forward

    Transformation:
        camera_X = vehicle_X (保持左右一致)
        camera_Y = -vehicle_Z (down = -up)
        camera_Z = vehicle_Y (forward = forward)

    Returns:
        4x4 homogeneous transformation matrix for axis swap
    """
    axis_swap = np.array([
        [1, 0, 0, 0],    # cam_x = veh_x
        [0, 0, -1, 0],   # cam_y = -veh_z
        [0, 1, 0, 0],    # cam_z = veh_y
        [0, 0, 0, 1]
    ])
    return axis_swap


def vehicle_pose_to_rotation_matrix(yaw: float) -> np.ndarray:
    """
    Convert vehicle pose yaw angle to rotation matrix.

    The vehicle pose uses a convention where:
        - Forward direction (Y_vehicle) in world coords is [cos(yaw), sin(yaw), 0]
        - Left direction (X_vehicle) in world coords is [-sin(yaw), cos(yaw), 0]

    Args:
        yaw: Yaw angle in radians

    Returns:
        3x3 rotation matrix from vehicle to world coordinates
    """
    R = np.array([
        [-np.sin(yaw), np.cos(yaw), 0],
        [np.cos(yaw), np.sin(yaw), 0],
        [0, 0, 1]
    ])
    return R


def compute_lidar_to_world_matrix(lidar_params: dict, vehicle_pose: dict) -> np.ndarray:
    """
    Compute the transformation matrix from LiDAR to world coordinate system.

    Args:
        lidar_params: LiDAR calibration parameters with 'extrinsics' containing
                      x, y, z, yaw, pitch, roll (in degrees)
        vehicle_pose: Vehicle pose in world coordinates with
                      x, y, z, yaw (in radians)

    Returns:
        4x4 transformation matrix from LiDAR to world coordinates
    """
    # LiDAR to vehicle (with rotation, angles in degrees)
    lidar_ext = lidar_params['extrinsics']
    lidar_t = np.array([lidar_ext['x'], lidar_ext['y'], lidar_ext['z']])
    lidar_R = euler_to_rotation_matrix(
        lidar_ext['yaw'], lidar_ext['pitch'], lidar_ext['roll'],
        degrees=True  # LiDAR angles are in degrees
    )
    lidar_to_vehicle = build_transform_matrix(lidar_R, lidar_t)

    # Vehicle to world (using vehicle pose with correct rotation convention)
    vehicle_t = np.array([vehicle_pose['x'], vehicle_pose['y'], vehicle_pose['z']])
    vehicle_R = vehicle_pose_to_rotation_matrix(vehicle_pose['yaw'])
    vehicle_to_world = build_transform_matrix(vehicle_R, vehicle_t)

    # Combine transformations
    lidar_to_world = vehicle_to_world @ lidar_to_vehicle

    return lidar_to_world


def compute_world_to_cam_matrix(vehicle_pose: dict, camera_params: dict) -> np.ndarray:
    """
    Compute the transformation matrix from world to camera coordinate system.

    Args:
        vehicle_pose: Vehicle pose in world coordinates with
                      x, y, z, yaw (in radians)
        camera_params: Camera calibration parameters

    Returns:
        4x4 transformation matrix from world to camera coordinates
    """
    # World to vehicle (inverse of vehicle_to_world)
    vehicle_t = np.array([vehicle_pose['x'], vehicle_pose['y'], vehicle_pose['z']])
    vehicle_R = vehicle_pose_to_rotation_matrix(vehicle_pose['yaw'])
    vehicle_to_world = build_transform_matrix(vehicle_R, vehicle_t)
    world_to_vehicle = np.linalg.inv(vehicle_to_world)

    # Vehicle to camera
    cam_ext = camera_params['camera_to_vehicle_extrinsics']
    cam_t = np.array([cam_ext['x'], cam_ext['y'], cam_ext['z']])
    cam_R = euler_to_rotation_matrix(
        cam_ext['yaw'], cam_ext['pitch'], cam_ext['roll'],
        degrees=True
    )
    cam_to_vehicle = build_transform_matrix(cam_R, cam_t)
    vehicle_to_cam_raw = np.linalg.inv(cam_to_vehicle)

    # Apply axis swap
    axis_swap = get_vehicle_to_camera_axis_swap()
    vehicle_to_cam = axis_swap @ vehicle_to_cam_raw

    # Combine transformations
    world_to_cam = vehicle_to_cam @ world_to_vehicle

    return world_to_cam


def merge_point_clouds(points_list: list, colors_list: list = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Merge multiple point clouds into one.

    Args:
        points_list: List of Nx3 point arrays
        colors_list: Optional list of Nx3 color arrays

    Returns:
        Tuple of (merged_points, merged_colors or None)
    """
    merged_points = np.vstack(points_list)

    if colors_list is not None:
        # Filter out None entries
        valid_colors = [c for c in colors_list if c is not None]
        if len(valid_colors) == len(colors_list):
            merged_colors = np.vstack(colors_list)
        else:
            merged_colors = None
    else:
        merged_colors = None

    return merged_points, merged_colors


def compute_lidar_to_cam_matrix(lidar_params: dict, camera_params: dict) -> np.ndarray:
    """
    Compute the transformation matrix from LiDAR to camera coordinate system.

    The transformation chain is:
        lidar_2_cam = axis_swap @ inv(cam_2_vehicle) @ lidar_2_vehicle

    Coordinate systems:
        - Vehicle: X-left, Y-forward, Z-up
        - Camera: X-right, Y-down, Z-forward

    Note:
        - Camera extrinsics use euler_to_rotation_matrix with yaw2Z_roll2X_pitch2Y_negP convention
        - LiDAR extrinsics angles are in degrees

    Args:
        lidar_params: LiDAR calibration parameters with 'extrinsics' containing
                      x, y, z, yaw, pitch, roll (in degrees)
        camera_params: Camera calibration parameters with 'camera_to_vehicle_extrinsics'
                       containing x, y, z, yaw, pitch, roll (in degrees)

    Returns:
        4x4 transformation matrix from LiDAR to camera coordinates
    """
    # Extract LiDAR extrinsics (angles in degrees)
    lidar_ext = lidar_params['extrinsics']
    lidar_t = np.array([lidar_ext['x'], lidar_ext['y'], lidar_ext['z']])
    lidar_R = euler_to_rotation_matrix(
        lidar_ext['yaw'], lidar_ext['pitch'], lidar_ext['roll'],
        degrees=True  # LiDAR angles are in degrees
    )
    lidar_to_vehicle = build_transform_matrix(lidar_R, lidar_t)

    # Extract camera extrinsics (angles in degrees)
    cam_ext = camera_params['camera_to_vehicle_extrinsics']
    cam_t = np.array([cam_ext['x'], cam_ext['y'], cam_ext['z']])
    cam_R = euler_to_rotation_matrix(
        cam_ext['yaw'], cam_ext['pitch'], cam_ext['roll'],
        degrees=True  # Camera angles are in degrees
    )
    cam_to_vehicle = build_transform_matrix(cam_R, cam_t)

    # Compute vehicle to camera transformation
    vehicle_to_cam = np.linalg.inv(cam_to_vehicle)

    # Get axis swap matrix for coordinate system conversion
    axis_swap = get_vehicle_to_camera_axis_swap()

    # Compute lidar to camera transformation with axis swap
    lidar_to_cam = axis_swap @ vehicle_to_cam @ lidar_to_vehicle

    return lidar_to_cam
