"""
LiDAR-Image Alignment Module.

Projects LiDAR point cloud onto camera images using calibration parameters.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from utils.transform_utils import compute_lidar_to_cam_matrix, transform_points
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
    get_frame_id_from_folder,
)
from utils.visualization_utils import (
    project_points_to_image,
    visualize_points_on_image,
    create_depth_map,
)


class LidarImageAligner:
    """
    Aligns LiDAR point clouds with camera images.

    Projects LiDAR points onto camera images using calibration parameters.
    """

    # Default test data path
    DEFAULT_DATA_ROOT = "/moganshan/afs_a/yuhan/repo/chery_preprocess/pcd_reproj/test/lijiaoqiao_20260205_02"
    DEFAULT_SEQUENCE = "lijiaoqiao_20260205_02_offset_0.0m"
    DEFAULT_FRAME = "17000010"

    # All camera names
    ALL_CAMERAS = [
        'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110',
        'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60',
        'CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30',
        'CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15',
        'CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99',
        'CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99',
        'CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30',
        'CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99',
        'CAM_PBQ_REAR_RESET_OPTICAL_H50',
        'CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30',
        'CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99',
    ]

    def __init__(self, data_root: Optional[str] = None,
                 sequence: Optional[str] = None,
                 frame: Optional[str] = None):
        """
        Initialize the aligner.

        Args:
            data_root: Root directory containing test data
            sequence: Sequence folder name
            frame: Frame folder name
        """
        self.data_root = data_root or self.DEFAULT_DATA_ROOT
        self.sequence = sequence or self.DEFAULT_SEQUENCE
        self.frame = frame or self.DEFAULT_FRAME

        self.sequence_path = os.path.join(self.data_root, self.sequence)
        self.frame_path = os.path.join(self.sequence_path, self.frame)

        # Load calibration parameters
        self.camera_params = load_camera_params(
            os.path.join(self.sequence_path, "camera_params.json")
        )
        self.lidar_params = load_lidar_params(
            os.path.join(self.sequence_path, "lidar_params.json")
        )

        # Load point cloud
        pcd_path = find_lidar_front_file(self.frame_path)
        if pcd_path is None:
            raise FileNotFoundError(f"No LDR_FRONT PCD found in {self.frame_path}")

        self.points, self.colors = load_pcd(pcd_path)
        print(f"Loaded {len(self.points)} points from {pcd_path}")

    def align_single_camera(self, camera_name: str,
                            min_depth: float = 0.0,
                            max_depth: float = 100.0,
                            point_size: int = 2) -> Optional[Dict[str, np.ndarray]]:
        """
        Align point cloud to a single camera.

        Args:
            camera_name: Name of the camera
            min_depth: Minimum depth for visualization
            max_depth: Maximum depth for visualization
            point_size: Size of points in visualization

        Returns:
            Dictionary with 'projected' and 'depth_map' images, or None if failed
        """
        # Get camera image
        image_path = find_camera_image(self.frame_path, camera_name)
        if image_path is None:
            print(f"No image found for camera {camera_name}")
            return None

        # Load image
        image = load_image(image_path)

        # Get camera parameters
        cam_params = self.camera_params[camera_name]
        intrinsics = get_camera_intrinsics(self.camera_params, camera_name)
        K = build_intrinsic_matrix(intrinsics)
        width = intrinsics['width']
        height = intrinsics['height']

        # Compute LiDAR to camera transformation
        lidar_to_cam = compute_lidar_to_cam_matrix(self.lidar_params, cam_params)

        # Transform points to camera coordinates
        points_camera = transform_points(self.points, lidar_to_cam)

        # Project points to image
        uv, depths, valid = project_points_to_image(points_camera, K, width, height)

        # Create visualizations
        projected = visualize_points_on_image(
            image, uv, depths, valid, point_size, min_depth, max_depth
        )
        depth_map = create_depth_map(
            image, uv, depths, valid, min_depth, max_depth
        )

        return {
            'original': image,
            'projected': projected,
            'depth_map': depth_map,
            'num_valid_points': np.sum(valid),
        }

    def align_all_cameras(self, output_dir: Optional[str] = None,
                          cameras: Optional[List[str]] = None,
                          min_depth: float = 0.0,
                          max_depth: float = 100.0,
                          point_size: int = 2) -> Dict[str, str]:
        """
        Align point cloud to all cameras and save results.

        Args:
            output_dir: Directory to save output images
            cameras: List of camera names to process (default: all cameras)
            min_depth: Minimum depth for visualization
            max_depth: Maximum depth for visualization
            point_size: Size of points in visualization

        Returns:
            Dictionary mapping camera name to output file paths
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output", "task1_1"
            )

        os.makedirs(output_dir, exist_ok=True)

        if cameras is None:
            cameras = self.ALL_CAMERAS

        frame_id = get_frame_id_from_folder(self.frame_path)
        results = {}

        print(f"\nProcessing {len(cameras)} cameras:")
        for cam_name in cameras:
            short_name = cam_name.replace("CAM_PBQ_", "").replace("_RESET_OPTICAL", "")
            print(f"  - {short_name}")

            result = self.align_single_camera(
                cam_name, min_depth, max_depth, point_size
            )

            if result is None:
                continue

            # Generate output file names
            base_name = f"{frame_id}_{short_name}"

            # Save projected image
            projected_path = os.path.join(output_dir, f"{base_name}_projected.jpg")
            cv2.imwrite(projected_path, result['projected'])

            # Save depth map
            depth_path = os.path.join(output_dir, f"{base_name}_depth.jpg")
            cv2.imwrite(depth_path, result['depth_map'])

            results[cam_name] = {
                'projected': projected_path,
                'depth_map': depth_path,
                'num_valid_points': result['num_valid_points'],
            }

            print(f"    Valid points: {result['num_valid_points']}")

        return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("LiDAR-Image Alignment")
    print("=" * 60)

    # Create aligner with default test data
    aligner = LidarImageAligner()

    print(f"\nSequence: {aligner.sequence}")
    print(f"Frame: {aligner.frame}")
    print(f"Point cloud size: {len(aligner.points)} points")

    # Process all cameras
    results = aligner.align_all_cameras(
        min_depth=0.0,
        max_depth=100.0,
        point_size=2
    )

    print("\n" + "=" * 60)
    print("Alignment complete!")
    print(f"Output saved to: output/task1_1/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
