"""
Static/Dynamic Point Cloud Segmentation Module (Task 2.1).

Projects LiDAR point cloud onto camera images with segmentation masks,
then splits into static and dynamic point clouds.

Segmentation masks: white = dynamic objects, black = static background
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import open3d as o3d

from utils.transform_utils import (
    compute_lidar_to_world_matrix,
    compute_world_to_cam_matrix,
    compute_lidar_to_cam_matrix,
    transform_points,
)
from utils.calibration_utils import (
    load_camera_params,
    load_lidar_params,
    build_intrinsic_matrix,
    get_camera_intrinsics,
)
from utils.io_utils import (
    load_pcd,
    load_image,
    find_lidar_front_file,
    find_camera_image,
    parse_vehicle_pose,
)
from utils.visualization_utils import project_points_to_image


class StaticDynamicSegmentation:
    """
    Splits point cloud into static and dynamic parts using segmentation masks.
    """

    # Default test data path
    DEFAULT_DATA_ROOT = "/moganshan/afs_a/yuhan/repo/chery_preprocess/pcd_reproj/test/lijiaoqiao_20260205_02"
    DEFAULT_SEQUENCE = "lijiaoqiao_20260205_02_offset_0.0m"
    DEFAULT_FRAME = "17000010"

    # Segmentation results path
    DEFAULT_SEG_ROOT = "/home/yuhan/yuchen/repos/processed_chery_data/lijiaoqiao_20260205_02"

    # Use FRONT_WIDE camera for segmentation (0.0m offset)
    SEGMENTATION_CAMERA = 'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110'

    def __init__(self, data_root: Optional[str] = None,
                 sequence: Optional[str] = None,
                 frame: Optional[str] = None,
                 seg_root: Optional[str] = None):
        """
        Initialize the segmentation module.

        Args:
            data_root: Root directory containing test data
            sequence: Sequence folder name
            frame: Frame folder name
            seg_root: Root directory containing segmentation results
        """
        self.data_root = data_root or self.DEFAULT_DATA_ROOT
        self.sequence = sequence or self.DEFAULT_SEQUENCE
        self.frame = frame or self.DEFAULT_FRAME
        self.seg_root = seg_root or self.DEFAULT_SEG_ROOT

        self.sequence_path = os.path.join(self.data_root, self.sequence)
        self.frame_path = os.path.join(self.sequence_path, self.frame)

        # Load calibration parameters
        self.camera_params = load_camera_params(
            os.path.join(self.sequence_path, "camera_params.json")
        )
        self.lidar_params = load_lidar_params(
            os.path.join(self.sequence_path, "lidar_params.json")
        )

    def load_segmentation_mask(self, offset: str, camera_name: str, frame_id: str) -> np.ndarray:
        """
        Load segmentation mask for a specific camera and offset.

        Args:
            offset: Offset string (e.g., "+1.0m")
            camera_name: Camera name
            frame_id: Frame ID

        Returns:
            Segmentation mask (binary: 1=dynamic, 0=static)
        """
        # Construct segmentation result path
        seg_sequence = f"lijiaoqiao_20260205_02_offset_{offset}"
        seg_frame_path = os.path.join(self.seg_root, seg_sequence, frame_id)

        # Find segmentation mask file
        # Pattern matches: *CAM_PBQ_<camera_name>*<frame_id>.png
        pattern = f"*{camera_name}*{frame_id}.png"
        import glob
        matches = glob.glob(os.path.join(seg_frame_path, pattern))

        if not matches:
            raise FileNotFoundError(f"No segmentation mask found for {camera_name} in {seg_frame_path}")

        # Load mask (white = dynamic, black = static)
        mask = cv2.imread(matches[0], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {matches[0]}")

        # Convert to binary (1=dynamic, 0=static)
        binary_mask = (mask > 128).astype(np.uint8)

        return binary_mask

    def segment_point_cloud(self, offset: str = None,
                           camera_name: str = None,
                           frame_id: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment point cloud into static and dynamic parts.

        Args:
            offset: Offset for segmentation mask (e.g., "0.0m", "+1.0m")
            camera_name: Camera to use for segmentation
            frame_id: Frame ID

        Returns:
            Tuple of (static_points, static_colors, dynamic_points, dynamic_colors)
        """
        if offset is None:
            offset = "0.0m"  # Use 0.0m segmentation by default
        if camera_name is None:
            camera_name = self.SEGMENTATION_CAMERA
        if frame_id is None:
            frame_id = self.frame

        print(f"Segmenting point cloud for frame {frame_id} using {camera_name}")
        print(f"  Segmentation mask offset: {offset}")

        # Build frame path for the specified frame_id
        frame_path = os.path.join(self.sequence_path, frame_id)

        # Load point cloud
        pcd_path = find_lidar_front_file(frame_path)
        if pcd_path is None:
            raise FileNotFoundError(f"No LDR_FRONT found in {frame_path}")

        points, _ = load_pcd(pcd_path)
        print(f"  Loaded {len(points)} points from {pcd_path}")

        # Load vehicle pose
        pb_path = os.path.join(frame_path, "data_frame.pb.txt")
        vehicle_pose = parse_vehicle_pose(pb_path)

        # Load camera image for colorization
        image_path = find_camera_image(frame_path, camera_name)
        if image_path is None:
            raise FileNotFoundError(f"No image found for camera {camera_name}")
        image = load_image(image_path)
        print(f"  Loaded image: {image_path}")

        # Load camera parameters and intrinsics
        cam_params = self.camera_params[camera_name]
        intrinsics = get_camera_intrinsics(self.camera_params, camera_name)
        K = build_intrinsic_matrix(intrinsics)
        width, height = intrinsics['width'], intrinsics['height']

        # Transform LiDAR -> Camera
        lidar_to_cam = compute_lidar_to_cam_matrix(self.lidar_params, cam_params)
        points_camera = transform_points(points, lidar_to_cam)

        # Project points to image
        uv, depths, valid = project_points_to_image(points_camera, K, width, height)

        # Colorize points from image
        colors = np.zeros((len(points), 3), dtype=np.float32)
        for i in range(len(points)):
            if valid[i]:
                u, v = int(uv[i, 0]), int(uv[i, 1])
                if 0 <= v < height and 0 <= u < width:
                    # OpenCV uses BGR, convert to RGB (0-1 range)
                    bgr = image[v, u]
                    colors[i] = [bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0]

        # Load segmentation mask
        seg_mask = self.load_segmentation_mask(offset, camera_name, frame_id)
        print(f"  Loaded segmentation mask: {seg_mask.shape}, dynamic ratio: {seg_mask.mean():.3f}")

        # Classify points based on mask
        # A point is dynamic if it projects onto or near a white pixel in the mask
        # Use dilation to expand dynamic regions for better matching
        from scipy.ndimage import binary_dilation
        kernel_size = 15  # Expand dynamic region by 15 pixels to capture nearby points
        expanded_mask = binary_dilation(seg_mask, iterations=kernel_size)

        dynamic_flags = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if valid[i]:
                u, v = int(uv[i, 0]), int(uv[i, 1])
                if 0 <= v < expanded_mask.shape[0] and 0 <= u < expanded_mask.shape[1]:
                    if expanded_mask[v, u] == 1:
                        dynamic_flags[i] = True

        # Split point cloud with image colors
        static_indices = ~dynamic_flags
        dynamic_indices = dynamic_flags

        static_points = points[static_indices]
        static_colors = colors[static_indices]

        dynamic_points = points[dynamic_indices]
        dynamic_colors = colors[dynamic_indices]

        print(f"  Static points: {len(static_points)} ({len(static_points)/len(points)*100:.1f}%)")
        print(f"  Dynamic points: {len(dynamic_points)} ({len(dynamic_indices)/len(points)*100:.1f}%)")

        return static_points, static_colors, dynamic_points, dynamic_colors

    def save_segmented_point_clouds(self, static_points: np.ndarray, static_colors: np.ndarray,
                                    dynamic_points: np.ndarray, dynamic_colors: np.ndarray,
                                    output_dir: str = None, frame_id: str = None):
        """
        Save segmented point clouds as PLY files.

        Args:
            static_points: Static point cloud
            static_colors: Static point colors
            dynamic_points: Dynamic point cloud
            dynamic_colors: Dynamic point colors
            output_dir: Output directory
            frame_id: Frame ID for filename
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output", "task2_1"
            )
        if frame_id is None:
            frame_id = self.frame

        os.makedirs(output_dir, exist_ok=True)

        # Save static point cloud
        static_ply = os.path.join(output_dir, f"{frame_id}_static.ply")
        save_point_cloud_as_ply(static_points, static_colors, static_ply)
        print(f"  Saved static point cloud: {static_ply}")

        # Save dynamic point cloud
        dynamic_ply = os.path.join(output_dir, f"{frame_id}_dynamic.ply")
        save_point_cloud_as_ply(dynamic_points, dynamic_colors, dynamic_ply)
        print(f"  Saved dynamic point cloud: {dynamic_ply}")

    def run_segmentation(self, offset: str = None, frame_id: str = None):
        """
        Run the complete segmentation pipeline.

        Args:
            offset: Offset for segmentation mask
            frame_id: Frame ID to process
        """
        print("=" * 60)
        print("Static/Dynamic Point Cloud Segmentation (Task 2.1)")
        print("=" * 60)
        print()

        # Segment point cloud
        static_points, static_colors, dynamic_points, dynamic_colors = \
            self.segment_point_cloud(offset=offset, frame_id=frame_id)

        # Save results
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "task2_1"
        )
        self.save_segmented_point_clouds(
            static_points, static_colors,
            dynamic_points, dynamic_colors,
            output_dir=output_dir,
            frame_id=frame_id or self.frame
        )

        print()
        print("=" * 60)
        print("Segmentation complete!")
        print(f"Output saved to: {output_dir}")
        print("=" * 60)

        return static_points, static_colors, dynamic_points, dynamic_colors


def save_point_cloud_as_ply(points: np.ndarray, colors: np.ndarray, filepath: str):
    """
    Save point cloud as PLY file.

    Args:
        points: Nx3 array of point positions
        colors: Nx3 array of point colors (0-1 range)
        filepath: Output PLY file path
    """
    if len(points) == 0:
        # Create empty PLY file
        pcd = o3d.geometry.PointCloud()
        o3d.io.write_point_cloud(filepath, pcd)
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filepath, pcd)


def main():
    """Main entry point."""
    # Available offsets with segmentation results
    # (Check which ones have data)
    segmentation = StaticDynamicSegmentation()

    # Use 0.0m offset for segmentation with image coloring
    results = segmentation.run_segmentation(offset="0.0m", frame_id="17000010")

    return results


if __name__ == "__main__":
    main()
