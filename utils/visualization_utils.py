"""
Visualization utilities for projecting and visualizing point clouds on images.
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from matplotlib import cm


def project_points_to_image(points_camera: np.ndarray, K: np.ndarray,
                            width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points in camera coordinates to 2D image plane.

    Args:
        points_camera: Nx3 array of 3D points in camera coordinates
        K: 3x3 camera intrinsic matrix
        width: Image width
        height: Image height

    Returns:
        Tuple of (uv Nx2 pixel coordinates, depths N, valid_mask N)
    """
    # Filter points behind the camera
    valid_depth = points_camera[:, 2] > 0.1

    # Project points
    uv_h = (K @ points_camera.T).T
    depths = uv_h[:, 2].copy()
    depths[~valid_depth] = -1

    # Avoid division by zero
    uv_h[:, 2] = np.where(uv_h[:, 2] > 0.1, uv_h[:, 2], 1.0)
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    # Check bounds
    valid_x = (uv[:, 0] >= 0) & (uv[:, 0] < width)
    valid_y = (uv[:, 1] >= 0) & (uv[:, 1] < height)
    valid = valid_depth & valid_x & valid_y

    return uv.astype(np.int32), depths, valid


def depth_to_color(depths: np.ndarray, min_depth: float = 0.0,
                   max_depth: float = 100.0) -> np.ndarray:
    """
    Convert depth values to RGB colors using a colormap.

    Args:
        depths: Array of depth values
        min_depth: Minimum depth for color mapping
        max_depth: Maximum depth for color mapping

    Returns:
        Nx3 array of RGB colors (0-255)
    """
    # Normalize depths
    normalized = (depths - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0, 1)

    # Use plasma colormap (red = near, blue = far)
    colors = cm.plasma(1 - normalized)[:, :3]  # RGBA to RGB, invert for red=near

    return (colors * 255).astype(np.uint8)


def visualize_points_on_image(image: np.ndarray, uv: np.ndarray,
                              depths: np.ndarray, valid: np.ndarray,
                              point_size: int = 2,
                              min_depth: float = 0.0,
                              max_depth: float = 100.0) -> np.ndarray:
    """
    Visualize projected points on image with depth-based coloring.

    Args:
        image: BGR image array
        uv: Nx2 pixel coordinates
        depths: N depth values
        valid: N boolean mask for valid points
        point_size: Size of points to draw
        min_depth: Minimum depth for color mapping
        max_depth: Maximum depth for color mapping

    Returns:
        Image with points visualized
    """
    result = image.copy()

    # Get valid points
    valid_uv = uv[valid]
    valid_depths = depths[valid]

    # Get colors based on depth
    colors = depth_to_color(valid_depths, min_depth, max_depth)

    # Convert colors from RGB to BGR for OpenCV
    colors_bgr = colors[:, ::-1]

    # Draw points
    for i, (pt, color) in enumerate(zip(valid_uv, colors_bgr)):
        cv2.circle(result, (int(pt[0]), int(pt[1])), point_size,
                   color.tolist(), -1)

    return result


def create_depth_map(image: np.ndarray, uv: np.ndarray,
                     depths: np.ndarray, valid: np.ndarray,
                     min_depth: float = 0.0,
                     max_depth: float = 100.0) -> np.ndarray:
    """
    Create a depth map visualization.

    Args:
        image: BGR image array
        uv: Nx2 pixel coordinates
        depths: N depth values
        valid: N boolean mask for valid points
        min_depth: Minimum depth for color mapping
        max_depth: Maximum depth for color mapping

    Returns:
        Depth map image
    """
    height, width = image.shape[:2]
    depth_map = np.zeros((height, width), dtype=np.float32)

    # Fill depth map
    valid_uv = uv[valid]
    valid_depths = depths[valid]

    for pt, depth in zip(valid_uv, valid_depths):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < width and 0 <= y < height:
            depth_map[y, x] = depth

    # Normalize and colorize
    depth_normalized = (depth_map - min_depth) / (max_depth - min_depth)
    depth_normalized = np.clip(depth_normalized, 0, 1)

    # Apply colormap (use COLORMAP_PLASMA for better visualization)
    depth_colored = cv2.applyColorMap(
        (depth_normalized * 255).astype(np.uint8),
        cv2.COLORMAP_PLASMA
    )

    # Invert colormap so red = near, blue = far
    depth_colored = cv2.bitwise_not(depth_colored)

    return depth_colored


def create_side_by_side_comparison(image: np.ndarray,
                                   projected: np.ndarray,
                                   depth_map: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side comparison of original, projected, and depth map.

    Args:
        image: Original BGR image
        projected: Image with projected points
        depth_map: Depth map visualization

    Returns:
        Side-by-side comparison image
    """
    # Resize all to same height if needed
    h = image.shape[0]

    def resize_to_height(img, target_h):
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            return cv2.resize(img, (new_w, target_h))
        return img

    image = resize_to_height(image, h)
    projected = resize_to_height(projected, h)
    depth_map = resize_to_height(depth_map, h)

    # Concatenate horizontally
    result = np.hstack([image, projected, depth_map])

    return result
