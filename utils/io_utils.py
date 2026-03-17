"""
I/O utilities for loading point clouds and images.
"""
import os
import re
import glob
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2
import open3d as o3d


def load_pcd(pcd_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load point cloud from PCD file using Open3D.

    Args:
        pcd_path: Path to PCD file

    Returns:
        Tuple of (points Nx3, colors Nx3 or None)
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors


def load_image(image_path: str) -> np.ndarray:
    """
    Load image using OpenCV.

    Args:
        image_path: Path to image file

    Returns:
        Image array in BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def find_lidar_front_file(folder: str) -> Optional[str]:
    """
    Find LDR_FRONT PCD file in a folder.

    Args:
        folder: Path to folder containing data files

    Returns:
        Path to LDR_FRONT PCD file or None if not found
    """
    pattern = os.path.join(folder, "*-LDR_FRONT-*.pcd")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def find_camera_image(folder: str, camera_name: str) -> Optional[str]:
    """
    Find camera image file in a folder.

    Args:
        folder: Path to folder containing data files
        camera_name: Name of the camera (e.g., "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110")

    Returns:
        Path to camera image file or None if not found
    """
    pattern = os.path.join(folder, f"*{camera_name}*.jpg")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def get_frame_id_from_folder(folder: str) -> str:
    """
    Extract frame ID from folder name.

    Args:
        folder: Path to frame folder

    Returns:
        Frame ID (folder name)
    """
    return os.path.basename(folder)


def parse_vehicle_pose(pb_path: str) -> Dict[str, float]:
    """
    Parse vehicle pose from data_frame.pb.txt file.

    Args:
        pb_path: Path to data_frame.pb.txt file

    Returns:
        Dictionary with x, y, z, yaw, pitch, roll (angles in radians)
    """
    with open(pb_path, 'r') as f:
        content = f.read()

    # Parse vehicle_pose block
    pose_match = re.search(
        r'vehicle_pose\s*\{([^}]+)\}',
        content,
        re.DOTALL
    )

    if not pose_match:
        raise ValueError(f"Could not find vehicle_pose in {pb_path}")

    pose_content = pose_match.group(1)

    # Extract values
    def extract_value(name: str) -> float:
        match = re.search(rf'{name}:\s*([\d.\-+e]+)', pose_content)
        if match:
            return float(match.group(1))
        raise ValueError(f"Could not find {name} in vehicle_pose")

    pose = {
        'x': extract_value('x'),
        'y': extract_value('y'),
        'z': extract_value('z'),
        'yaw': extract_value('yaw'),      # already in radians
        'pitch': extract_value('pitch'),  # already in radians
        'roll': extract_value('roll'),    # already in radians
    }

    return pose


def get_frame_folders(sequence_path: str, start_frame: int, end_frame: int) -> List[str]:
    """
    Get list of frame folder paths for a range of frames.

    Args:
        sequence_path: Path to sequence directory
        start_frame: Starting frame ID (inclusive)
        end_frame: Ending frame ID (inclusive)

    Returns:
        List of frame folder paths
    """
    folders = []
    for frame_id in range(start_frame, end_frame + 1):
        frame_path = os.path.join(sequence_path, str(frame_id))
        if os.path.exists(frame_path):
            folders.append(frame_path)
    return folders
