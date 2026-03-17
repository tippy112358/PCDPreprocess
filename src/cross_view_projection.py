"""
Cross-view Projection Module (Task 1-3).

Projects LiDAR point cloud from source viewpoint (0m offset) to target viewpoints
with different offsets (+1m, +2m, +4m, -1m, -2m, -4m).
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
    transform_points,
    merge_point_clouds,
)
from utils.calibration_utils import (
    load_camera_params,
    load_lidar_params,
    build_intrinsic_matrix,
    get_camera_intrinsics,
    get_camera_extrinsics,
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


def save_point_cloud_as_ply(points: np.ndarray,
                             colors: np.ndarray = None,
                             output_path: str = None):
    """
    Save point cloud as PLY file.

    Args:
        points: Nx3 array of points
        colors: Nx3 array of colors (0-1 range) or None
        output_path: Output file path
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"  Saved PLY: {output_path}")


class CrossViewProjection:
    """
    Cross-view point cloud projection.

    Projects LiDAR points from source viewpoint to target viewpoints with different offsets.
    """

    # Default paths
    DEFAULT_DATA_ROOT = "/moganshan/afs_a/yuhan/repo/chery_preprocess/pcd_reproj/test/lijiaoqiao_20260205_02"

    # Available offsets
    OFFSETS = ['+1.0m', '+2.0m', '+4.0m', '-1.0m', '-2.0m', '-4.0m']

    # Test cameras
    TEST_CAMERAS = [
        'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110',
        'CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99',
        'CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99',
    ]

    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize the cross-view projection module.

        Args:
            data_root: Root directory containing test data
        """
        self.data_root = data_root or self.DEFAULT_DATA_ROOT

        # Source: 0m offset
        self.source_sequence = "lijiaoqiao_20260205_02_offset_0.0m"
        self.source_path = os.path.join(self.data_root, self.source_sequence)

        # Load source calibration
        self.lidar_params = load_lidar_params(
            os.path.join(self.source_path, "lidar_params.json")
        )

    def load_source_lidar(self, frame_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Load source LiDAR data and vehicle pose.

        Args:
            frame_id: Frame ID (e.g., "17000010")

        Returns:
            Tuple of (points, colors, vehicle_pose)
        """
        frame_path = os.path.join(self.source_path, frame_id)

        # Load point cloud
        pcd_path = find_lidar_front_file(frame_path)
        if pcd_path is None:
            raise FileNotFoundError(f"No LDR_FRONT found in {frame_path}")

        points, colors = load_pcd(pcd_path)

        # Load vehicle pose
        pb_path = os.path.join(frame_path, "data_frame.pb.txt")
        vehicle_pose = parse_vehicle_pose(pb_path)

        return points, colors, vehicle_pose

    def load_target_data(self, offset: str, frame_id: str) -> Tuple[Dict, Dict, np.ndarray, Dict]:
        """
        Load target viewpoint data (camera params, pose, image).

        Args:
            offset: Offset string (e.g., "+1.0m")
            frame_id: Frame ID

        Returns:
            Tuple of (camera_params, vehicle_pose, image, intrinsics)
        """
        target_sequence = f"lijiaoqiao_20260205_02_offset_{offset}"
        target_path = os.path.join(self.data_root, target_sequence)
        frame_path = os.path.join(target_path, frame_id)

        # Load camera params
        camera_params = load_camera_params(
            os.path.join(target_path, "camera_params.json")
        )

        # Load vehicle pose
        pb_path = os.path.join(frame_path, "data_frame.pb.txt")
        vehicle_pose = parse_vehicle_pose(pb_path)

        return camera_params, vehicle_pose, frame_path

    def project_to_target_view(self, points: np.ndarray,
                                source_pose: Dict,
                                target_pose: Dict,
                                all_camera_params: Dict,
                                camera_name: str,
                                image: np.ndarray) -> Dict:
        """
        Project points from source view to target camera view.

        Args:
            points: Source LiDAR points (in LiDAR frame)
            source_pose: Source vehicle pose
            target_pose: Target vehicle pose
            all_camera_params: All camera parameters dict
            camera_name: Camera name
            image: Target camera image

        Returns:
            Dictionary with projection results
        """
        # Get camera params for specific camera
        cam_params = all_camera_params[camera_name]

        # Get camera intrinsics
        intrinsics = get_camera_intrinsics(all_camera_params, camera_name)
        K = build_intrinsic_matrix(intrinsics)
        width, height = intrinsics['width'], intrinsics['height']

        # Transform: LiDAR -> World (using source pose)
        lidar_to_world = compute_lidar_to_world_matrix(self.lidar_params, source_pose)
        world_points = transform_points(points, lidar_to_world)

        # Transform: World -> Camera (using target pose)
        world_to_cam = compute_world_to_cam_matrix(target_pose, cam_params)
        points_camera = transform_points(world_points, world_to_cam)

        # Project to image
        uv, depths, valid = project_points_to_image(points_camera, K, width, height)

        # Create visualizations
        projected = visualize_points_on_image(
            image, uv, depths, valid, 2, 0, 150
        )
        depth_map = create_depth_map(
            image, uv, depths, valid, 0, 150
        )

        return {
            'original': image,
            'projected': projected,
            'depth_map': depth_map,
            'num_valid_points': np.sum(valid),
        }

    def run_cross_view_projection(self, frame_id: str = "17000010",
                                    offsets: List[str] = None,
                                    cameras: List[str] = None,
                                    output_dir: str = None,
                                    save_ply: bool = True):
        """
        Run cross-view projection for multiple offsets.

        Args:
            frame_id: Frame ID to process
            offsets: List of offsets to project to
            cameras: List of cameras to project to
            output_dir: Output directory
            save_ply: Whether to save point cloud as PLY
        """
        if offsets is None:
            offsets = ['+1.0m', '+2.0m', '+4.0m']
        if cameras is None:
            cameras = self.TEST_CAMERAS
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output", "task1_3"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Load source LiDAR
        print(f"Loading source LiDAR from offset_0.0m/{frame_id}")
        source_points, source_colors, source_pose = self.load_source_lidar(frame_id)
        print(f"  Points: {len(source_points)}")

        # Transform to world coordinates (only once)
        lidar_to_world = compute_lidar_to_world_matrix(self.lidar_params, source_pose)
        world_points = transform_points(source_points, lidar_to_world)

        # Save world point cloud as PLY
        if save_ply:
            ply_dir = os.path.join(output_dir, "point_clouds")
            os.makedirs(ply_dir, exist_ok=True)
            ply_path = os.path.join(ply_dir, f"{frame_id}_world.ply")
            save_point_cloud_as_ply(world_points, source_colors, ply_path)

        results = {}

        for offset in offsets:
            print(f"\n{'='*50}")
            print(f"Projecting to offset_{offset}")
            print('='*50)

            # Load target data
            try:
                camera_params, target_pose, frame_path = self.load_target_data(offset, frame_id)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

            for cam_name in cameras:
                short_name = cam_name.replace("CAM_PBQ_", "").replace("_RESET_OPTICAL", "")

                # Load target image
                image_path = find_camera_image(frame_path, cam_name)
                if image_path is None:
                    print(f"  No image for {short_name}")
                    continue
                image = load_image(image_path)

                print(f"\n  Camera: {short_name}")

                # Project
                result = self.project_to_target_view(
                    source_points, source_pose, target_pose,
                    camera_params, cam_name, image
                )

                # Save results
                offset_clean = offset.replace('+', 'pos').replace('-', 'neg')
                base_name = f"{frame_id}_offset_{offset_clean}_{short_name}"

                projected_path = os.path.join(output_dir, f"{base_name}_projected.jpg")
                cv2.imwrite(projected_path, result['projected'])

                depth_path = os.path.join(output_dir, f"{base_name}_depth.jpg")
                cv2.imwrite(depth_path, result['depth_map'])

                print(f"    Valid points: {result['num_valid_points']}")
                print(f"    Saved: {projected_path}")

                results[f"{offset}_{cam_name}"] = {
                    'projected': projected_path,
                    'depth_map': depth_path,
                    'num_valid_points': result['num_valid_points'],
                }

        print(f"\n{'='*50}")
        print(f"Cross-view projection complete!")
        print(f"Output saved to: {output_dir}")
        print('='*50)

        return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("Cross-view Projection (Task 1-3)")
    print("=" * 60)

    projection = CrossViewProjection()

    # Run for all offsets
    results = projection.run_cross_view_projection(
        frame_id="17000010",
        offsets=['+1.0m', '+2.0m', '+4.0m', '-1.0m', '-2.0m', '-4.0m'],
        cameras=[
            'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110',
            'CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99',
            'CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99',
        ]
    )

    return results


if __name__ == "__main__":
    main()
