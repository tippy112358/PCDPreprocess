"""
Multi-frame LiDAR Fusion Module.

Merges multiple frames of LiDAR point clouds and projects to camera images.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from utils.transform_utils import (
    compute_lidar_to_world_matrix,
    compute_world_to_cam_matrix,
    transform_points,
    merge_point_clouds,
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
    get_frame_folders,
)
from utils.visualization_utils import (
    project_points_to_image,
    visualize_points_on_image,
    create_depth_map,
)


class MultiFrameLidarFusion:
    """
    Multi-frame LiDAR point cloud fusion and projection.
    """

    # Default test data path
    DEFAULT_DATA_ROOT = "/moganshan/afs_a/yuhan/repo/chery_preprocess/pcd_reproj/test/lijiaoqiao_20260205_02"
    DEFAULT_SEQUENCE = "lijiaoqiao_20260205_02_offset_0.0m"
    DEFAULT_START_FRAME = 17000010
    DEFAULT_END_FRAME = 17000040

    # All cameras
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
                 start_frame: Optional[int] = None,
                 end_frame: Optional[int] = None):
        """
        Initialize the fusion module.

        Args:
            data_root: Root directory containing test data
            sequence: Sequence folder name
            start_frame: Starting frame ID
            end_frame: Ending frame ID
        """
        self.data_root = data_root or self.DEFAULT_DATA_ROOT
        self.sequence = sequence or self.DEFAULT_SEQUENCE
        self.start_frame = start_frame or self.DEFAULT_START_FRAME
        self.end_frame = end_frame or self.DEFAULT_END_FRAME

        self.sequence_path = os.path.join(self.data_root, self.sequence)

        # Load calibration parameters
        self.camera_params = load_camera_params(
            os.path.join(self.sequence_path, "camera_params.json")
        )
        self.lidar_params = load_lidar_params(
            os.path.join(self.sequence_path, "lidar_params.json")
        )

        # Get frame folders
        self.frame_folders = get_frame_folders(
            self.sequence_path, self.start_frame, self.end_frame
        )
        print(f"Found {len(self.frame_folders)} frames: {self.start_frame} - {self.end_frame}")

    def load_frame_data(self, frame_folder: str) -> Dict:
        """
        Load data for a single frame.

        Args:
            frame_folder: Path to frame folder

        Returns:
            Dictionary with points, colors, vehicle_pose, frame_id
        """
        frame_id = os.path.basename(frame_folder)

        # Load point cloud
        pcd_path = find_lidar_front_file(frame_folder)
        if pcd_path is None:
            print(f"No LDR_FRONT found in {frame_folder}")
            return None

        points, colors = load_pcd(pcd_path)

        # Load vehicle pose
        pb_path = os.path.join(frame_folder, "data_frame.pb.txt")
        if not os.path.exists(pb_path):
            print(f"No data_frame.pb.txt found in {frame_folder}")
            return None

        vehicle_pose = parse_vehicle_pose(pb_path)

        return {
            'frame_id': frame_id,
            'points': points,
            'colors': colors,
            'vehicle_pose': vehicle_pose,
        }

    def transform_to_world(self, frame_data: Dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform frame point cloud to world coordinates.

        Args:
            frame_data: Frame data dictionary

        Returns:
            Tuple of (world_points, colors)
        """
        lidar_to_world = compute_lidar_to_world_matrix(
            self.lidar_params, frame_data['vehicle_pose']
        )
        world_points = transform_points(frame_data['points'], lidar_to_world)
        return world_points, frame_data['colors']

    def merge_all_frames(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load and merge all frame point clouds in world coordinates.

        Returns:
            Tuple of (merged_world_points, merged_colors)
        """
        world_points_list = []
        colors_list = []

        print("\nLoading and transforming frames to world coordinates:")
        for frame_folder in self.frame_folders:
            frame_data = self.load_frame_data(frame_folder)
            if frame_data is None:
                continue

            world_points, colors = self.transform_to_world(frame_data)
            world_points_list.append(world_points)
            colors_list.append(colors)

            print(f"  Frame {frame_data['frame_id']}: {len(world_points)} points")

        # Merge all point clouds
        merged_points, merged_colors = merge_point_clouds(world_points_list, colors_list)
        print(f"\nTotal merged points: {len(merged_points)}")

        return merged_points, merged_colors

    def project_merged_to_camera(self, merged_points: np.ndarray,
                                  target_frame_folder: str,
                                  camera_name: str,
                                  min_depth: float = 0.0,
                                  max_depth: float = 150.0,
                                  point_size: int = 2) -> Optional[Dict]:
        """
        Project merged world points to a camera at target frame.

        Args:
            merged_points: Merged point cloud in world coordinates
            target_frame_folder: Target frame folder for camera pose
            camera_name: Name of the camera
            min_depth: Minimum depth for visualization
            max_depth: Maximum depth for visualization
            point_size: Size of points in visualization

        Returns:
            Dictionary with projection results
        """
        # Load target frame data
        target_data = self.load_frame_data(target_frame_folder)
        if target_data is None:
            return None

        # Load camera image
        image_path = find_camera_image(target_frame_folder, camera_name)
        if image_path is None:
            print(f"No image found for {camera_name}")
            return None
        image = load_image(image_path)

        # Get camera intrinsics
        cam_params = self.camera_params[camera_name]
        intrinsics = get_camera_intrinsics(self.camera_params, camera_name)
        K = build_intrinsic_matrix(intrinsics)
        width, height = intrinsics['width'], intrinsics['height']

        # Compute world to camera transformation
        world_to_cam = compute_world_to_cam_matrix(
            target_data['vehicle_pose'], cam_params
        )

        # Transform world points to camera coordinates
        points_camera = transform_points(merged_points, world_to_cam)

        # Project to image
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
            'total_points': len(merged_points),
        }

    def run_fusion_and_projection(self, output_dir: Optional[str] = None,
                                   target_camera: str = None,
                                   cameras: List[str] = None):
        """
        Run the complete fusion and projection pipeline.

        Args:
            output_dir: Directory to save output images
            target_camera: (deprecated) Use cameras instead
            cameras: List of camera names to project to
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output", "task1_2"
            )
        os.makedirs(output_dir, exist_ok=True)

        if cameras is None:
            cameras = self.ALL_CAMERAS

        # Merge all frames
        merged_points, merged_colors = self.merge_all_frames()

        # Use the last frame as target
        target_frame = self.frame_folders[-1]
        target_frame_id = os.path.basename(target_frame)

        print(f"\nProjecting to target frame: {target_frame_id}")

        results = {}
        for cam_name in cameras:
            short_name = cam_name.replace("CAM_PBQ_", "").replace("_RESET_OPTICAL", "")
            print(f"\n  Processing camera: {short_name}")

            result = self.project_merged_to_camera(
                merged_points, target_frame, cam_name,
                min_depth=0.0, max_depth=150.0, point_size=2
            )

            if result is None:
                continue

            # Save outputs
            base_name = f"{target_frame_id}_{short_name}"
            projected_path = os.path.join(output_dir, f"{base_name}_projected.jpg")
            cv2.imwrite(projected_path, result['projected'])

            depth_path = os.path.join(output_dir, f"{base_name}_depth.jpg")
            cv2.imwrite(depth_path, result['depth_map'])

            results[cam_name] = {
                'projected': projected_path,
                'depth_map': depth_path,
                'num_valid_points': result['num_valid_points'],
            }

            print(f"    Valid points: {result['num_valid_points']} / {result['total_points']}")
            print(f"    Saved: {projected_path}")

        return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("Multi-frame LiDAR Fusion")
    print("=" * 60)

    # Create fusion module
    fusion = MultiFrameLidarFusion()

    print(f"\nSequence: {fusion.sequence}")
    print(f"Frames: {fusion.start_frame} - {fusion.end_frame}")

    # Run fusion and projection
    results = fusion.run_fusion_and_projection()

    print("\n" + "=" * 60)
    print("Fusion complete!")
    print(f"Output saved to: output/task1_2/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
