"""
Colored Point Cloud Reprojection Module (Task 1-4).

1. Project LiDAR points to source camera image
2. Colorize the point cloud using image colors
3. Project the colored point cloud to target viewpoints
4. Save intermediate colored point cloud as PLY
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
    print(f"  Saved PLY: {output_path} ({len(points)} points)")


def colorize_points_from_image(points_camera: np.ndarray,
                                image: np.ndarray,
                                K: np.ndarray,
                                width: int,
                                height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Colorize points using image colors.

    Args:
        points_camera: Nx3 points in camera coordinates
        image: BGR image
        K: Camera intrinsic matrix
        width: Image width
        height: Image height

    Returns:
        Tuple of (colors Nx3 in 0-1 range, valid_mask N)
    """
    # Project points to image
    uv, depths, valid = project_points_to_image(points_camera, K, width, height)

    # Initialize colors
    colors = np.zeros((len(points_camera), 3), dtype=np.float32)

    # Get valid points
    valid_uv = uv[valid]

    # Sample colors from image (BGR -> RGB, 0-255 -> 0-1)
    for i, (pt, is_valid) in enumerate(zip(uv, valid)):
        if is_valid:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < width and 0 <= y < height:
                # BGR to RGB, normalize to 0-1
                bgr = image[y, x]
                colors[i] = [bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0]

    return colors, valid


class ColoredPointCloudReprojection:
    """
    Colored point cloud reprojection to new viewpoints.
    """

    DEFAULT_DATA_ROOT = "/moganshan/afs_a/yuhan/repo/chery_preprocess/pcd_reproj/test/lijiaoqiao_20260205_02"
    SOURCE_SEQUENCE = "lijiaoqiao_20260205_02_offset_0.0m"

    # Colorization camera (front wide for best coverage)
    COLORIZATION_CAMERA = 'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110'

    # Projection cameras
    PROJECTION_CAMERAS = [
        'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110',
        'CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99',
        'CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99',
    ]

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = data_root or self.DEFAULT_DATA_ROOT
        self.source_path = os.path.join(self.data_root, self.SOURCE_SEQUENCE)

        # Load calibration
        self.lidar_params = load_lidar_params(
            os.path.join(self.source_path, "lidar_params.json")
        )
        self.camera_params = load_camera_params(
            os.path.join(self.source_path, "camera_params.json")
        )

    def load_source_data(self, frame_id: str) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """
        Load source LiDAR, pose, and colorization image.

        Returns:
            Tuple of (points, source_pose, image)
        """
        frame_path = os.path.join(self.source_path, frame_id)

        # Load point cloud
        pcd_path = find_lidar_front_file(frame_path)
        points, _ = load_pcd(pcd_path)

        # Load vehicle pose
        pb_path = os.path.join(frame_path, "data_frame.pb.txt")
        source_pose = parse_vehicle_pose(pb_path)

        # Load colorization image
        image_path = find_camera_image(frame_path, self.COLORIZATION_CAMERA)
        image = load_image(image_path)

        return points, source_pose, image

    def colorize_point_cloud(self, points: np.ndarray,
                              source_pose: Dict,
                              image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Colorize point cloud using source camera image.

        Args:
            points: LiDAR points (in LiDAR frame)
            source_pose: Source vehicle pose
            image: Colorization camera image

        Returns:
            Tuple of (world_points, colors)
        """
        # Get camera intrinsics
        intrinsics = get_camera_intrinsics(self.camera_params, self.COLORIZATION_CAMERA)
        K = build_intrinsic_matrix(intrinsics)
        width, height = intrinsics['width'], intrinsics['height']

        # Transform LiDAR -> Camera
        cam_params = self.camera_params[self.COLORIZATION_CAMERA]
        lidar_to_cam = compute_lidar_to_cam_matrix(self.lidar_params, cam_params)
        points_camera = transform_points(points, lidar_to_cam)

        # Colorize
        colors, valid = colorize_points_from_image(points_camera, image, K, width, height)

        # Transform to world coordinates
        lidar_to_world = compute_lidar_to_world_matrix(self.lidar_params, source_pose)
        world_points = transform_points(points, lidar_to_world)

        return world_points, colors

    def load_target_data(self, offset: str, frame_id: str):
        """Load target viewpoint data."""
        target_sequence = f"lijiaoqiao_20260205_02_offset_{offset}"
        target_path = os.path.join(self.data_root, target_sequence)
        frame_path = os.path.join(target_path, frame_id)

        # Load camera params
        camera_params = load_camera_params(os.path.join(target_path, "camera_params.json"))

        # Load vehicle pose
        pb_path = os.path.join(frame_path, "data_frame.pb.txt")
        vehicle_pose = parse_vehicle_pose(pb_path)

        return camera_params, vehicle_pose, frame_path

    def project_colored_points(self, world_points: np.ndarray,
                                colors: np.ndarray,
                                target_pose: Dict,
                                camera_params: Dict,
                                camera_name: str,
                                image: np.ndarray) -> Dict:
        """
        Project colored world points to target camera.
        """
        # Get camera intrinsics
        intrinsics = get_camera_intrinsics(camera_params, camera_name)
        K = build_intrinsic_matrix(intrinsics)
        width, height = intrinsics['width'], intrinsics['height']

        # Transform World -> Camera
        world_to_cam = compute_world_to_cam_matrix(target_pose, camera_params[camera_name])
        points_camera = transform_points(world_points, world_to_cam)

        # Project to image
        uv, depths, valid = project_points_to_image(points_camera, K, width, height)

        # Create visualization with colors
        result_image = image.copy()
        valid_uv = uv[valid]
        valid_colors = colors[valid]

        for pt, color in zip(valid_uv, valid_colors):
            if np.sum(color) > 0:  # Only draw if colored
                # Convert 0-1 RGB to 0-255 BGR
                bgr = (color[::-1] * 255).astype(np.uint8).tolist()
                cv2.circle(result_image, (int(pt[0]), int(pt[1])), 2, bgr, -1)

        # Also create depth visualization
        depth_map = create_depth_map(image, uv, depths, valid, 0, 150)

        return {
            'original': image,
            'projected': result_image,
            'depth_map': depth_map,
            'num_valid_points': np.sum(valid),
            'num_colored_points': np.sum(valid & (np.sum(colors, axis=1) > 0)),
        }

    def run(self, frame_id: str = "17000010",
            offsets: List[str] = None,
            cameras: List[str] = None,
            output_dir: str = None):
        """
        Run the complete colored reprojection pipeline.
        """
        if offsets is None:
            offsets = ['+1.0m', '+2.0m', '+4.0m', '-1.0m', '-2.0m', '-4.0m']
        if cameras is None:
            cameras = self.PROJECTION_CAMERAS
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output", "task1_4"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Load source data
        print("=" * 60)
        print("Colored Point Cloud Reprojection (Task 1-4)")
        print("=" * 60)
        print(f"\nLoading source data from offset_0.0m/{frame_id}")
        points, source_pose, image = self.load_source_data(frame_id)
        print(f"  Points: {len(points)}")

        # Step 2: Colorize point cloud
        print(f"\nColorizing with {self.COLORIZATION_CAMERA}...")
        world_points, colors = self.colorize_point_cloud(points, source_pose, image)
        colored_count = np.sum(np.sum(colors, axis=1) > 0)
        print(f"  Colored points: {colored_count} / {len(colors)}")

        # Step 3: Save colored point cloud as PLY
        ply_dir = os.path.join(output_dir, "point_clouds")
        os.makedirs(ply_dir, exist_ok=True)
        ply_path = os.path.join(ply_dir, f"{frame_id}_colored_world.ply")
        save_point_cloud_as_ply(world_points, colors, ply_path)

        # Step 4: Project to target viewpoints
        results = {}
        for offset in offsets:
            print(f"\n{'='*50}")
            print(f"Projecting to offset_{offset}")
            print('='*50)

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
                    continue
                target_image = load_image(image_path)

                print(f"\n  Camera: {short_name}")

                # Project colored points
                result = self.project_colored_points(
                    world_points, colors, target_pose,
                    camera_params, cam_name, target_image
                )

                # Save results
                offset_clean = offset.replace('+', 'pos').replace('-', 'neg')
                base_name = f"{frame_id}_offset_{offset_clean}_{short_name}"

                projected_path = os.path.join(output_dir, f"{base_name}_colored.jpg")
                cv2.imwrite(projected_path, result['projected'])

                depth_path = os.path.join(output_dir, f"{base_name}_depth.jpg")
                cv2.imwrite(depth_path, result['depth_map'])

                print(f"    Valid points: {result['num_valid_points']}")
                print(f"    Colored points: {result['num_colored_points']}")
                print(f"    Saved: {projected_path}")

                results[f"{offset}_{cam_name}"] = result

        print(f"\n{'='*60}")
        print("Colored reprojection complete!")
        print(f"Output: {output_dir}")
        print('='*60)

        return results


def main():
    projection = ColoredPointCloudReprojection()
    results = projection.run()
    return results


if __name__ == "__main__":
    main()
