"""
Calibration utilities for loading and parsing camera and LiDAR parameters.
"""
import json
import numpy as np
from typing import Dict, List, Any


def load_camera_params(json_path: str) -> Dict[str, Any]:
    """
    Load camera calibration parameters from JSON file.

    Args:
        json_path: Path to camera_params.json

    Returns:
        Dictionary containing camera parameters for all cameras
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params


def load_lidar_params(json_path: str) -> Dict[str, Any]:
    """
    Load LiDAR calibration parameters from JSON file.

    Args:
        json_path: Path to lidar_params.json

    Returns:
        Dictionary containing LiDAR parameters
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params['installation']


def get_front_cameras(camera_params: Dict[str, Any],
                      yaw_threshold: float = 50) -> List[str]:
    """
    Get list of front-facing camera names based on yaw angle.

    Front cameras are defined as those with yaw angle close to 0 degrees.

    Args:
        camera_params: Dictionary of camera parameters
        yaw_threshold: Maximum absolute yaw angle to consider as front-facing

    Returns:
        List of front-facing camera names
    """
    front_cameras = []
    for cam_name, cam_data in camera_params.items():
        yaw = cam_data['camera_to_vehicle_extrinsics']['yaw']
        if abs(yaw) < yaw_threshold:
            front_cameras.append(cam_name)

    # Sort by FOV (H110, H60, H30, H15 - widest to narrowest)
    def get_fov(name):
        # Extract FOV from name like "H110", "H60", etc.
        for part in name.split('_'):
            if part.startswith('H'):
                return int(part[1:])
        return 0

    front_cameras.sort(key=get_fov, reverse=True)
    return front_cameras


def build_intrinsic_matrix(intrinsics: Dict[str, float]) -> np.ndarray:
    """
    Build 3x3 camera intrinsic matrix from intrinsics dictionary.

    The intrinsic matrix K is:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    Args:
        intrinsics: Dictionary containing fx, fy, cx, cy

    Returns:
        3x3 intrinsic matrix
    """
    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ])
    return K


def get_camera_intrinsics(camera_params: Dict[str, Any],
                          camera_name: str) -> Dict[str, float]:
    """
    Get intrinsics for a specific camera.

    Args:
        camera_params: Dictionary of all camera parameters
        camera_name: Name of the camera

    Returns:
        Dictionary containing intrinsics (fx, fy, cx, cy, width, height, etc.)
    """
    return camera_params[camera_name]['intrinsics']


def get_camera_extrinsics(camera_params: Dict[str, Any],
                          camera_name: str) -> Dict[str, float]:
    """
    Get extrinsics for a specific camera.

    Args:
        camera_params: Dictionary of all camera parameters
        camera_name: Name of the camera

    Returns:
        Dictionary containing extrinsics (x, y, z, yaw, pitch, roll)
    """
    return camera_params[camera_name]['camera_to_vehicle_extrinsics']
