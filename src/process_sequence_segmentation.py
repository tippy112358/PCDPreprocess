"""
Process Full Sequence with Static/Dynamic Segmentation.

Processes all frames in offset_0.0m sequence:
1. Segments each frame into static/dynamic point clouds
2. Saves to output/processed/ with original structure
3. Merges all static point clouds into one large point cloud
4. Generates visualization for validation
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

from utils.transform_utils import (
    compute_lidar_to_world_matrix,
    compute_world_to_cam_matrix,
    compute_lidar_to_cam_matrix,
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
from utils.visualization_utils import project_points_to_image


def create_mask_projection_visualization(mask: np.ndarray, uv: np.ndarray,
                                          valid: np.ndarray,
                                          camera_name: str) -> np.ndarray:
    """
    Create a visualization of point cloud projection on segmentation mask.

    Args:
        mask: Binary segmentation mask (1=dynamic, 0=static)
        uv: Nx2 UV coordinates
        valid: N boolean array indicating valid projections
        camera_name: Camera name for title

    Returns:
        Visualization image with projections overlaid on mask
    """
    # Convert mask to BGR for visualization
    # Static = green (0, 255, 0), Dynamic = red (0, 0, 255)
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_vis[mask == 0] = [0, 255, 0]  # Static = green
    mask_vis[mask == 1] = [0, 0, 255]  # Dynamic = red

    # Draw projected points as yellow dots
    for i in range(len(uv)):
        if valid[i]:
            u, v = int(uv[i, 0]), int(uv[i, 1])
            if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
                cv2.circle(mask_vis, (u, v), 3, (0, 255, 255), -1)  # Yellow

    # Add title
    cv2.putText(mask_vis, camera_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return mask_vis


def create_projection_visualization(image: np.ndarray, uv: np.ndarray,
                                     valid: np.ndarray, dynamic_mask: np.ndarray,
                                     camera_name: str) -> np.ndarray:
    """
    Create a visualization of point cloud projection on camera image.

    Args:
        image: Camera image (BGR)
        uv: Nx2 UV coordinates
        valid: N boolean array indicating valid projections
        dynamic_mask: Binary segmentation mask (1=dynamic, 0=static)
        camera_name: Camera name for title

    Returns:
        Visualization image with projections overlaid
    """
    vis = image.copy()

    # Draw projected points
    # Green = static, Red = dynamic
    for i in range(len(uv)):
        if valid[i]:
            u, v = int(uv[i, 0]), int(uv[i, 1])
            if 0 <= v < dynamic_mask.shape[0] and 0 <= u < dynamic_mask.shape[1]:
                is_dynamic = dynamic_mask[v, u] == 1
                color = (0, 0, 255) if is_dynamic else (0, 255, 0)  # BGR
                cv2.circle(vis, (u, v), 2, color, -1)

    # Add title
    cv2.putText(vis, camera_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return vis


class SequenceSegmentationProcessor:
    """
    Process entire sequence with static/dynamic segmentation.
    """

    # Default paths
    DEFAULT_DATA_ROOT = "/moganshan/afs_a/yuhan/repo/chery_preprocess/pcd_reproj/test/lijiaoqiao_20260205_02"
    DEFAULT_SEQUENCE = "lijiaoqiao_20260205_02_offset_0.0m"
    DEFAULT_SEG_ROOT = "/home/yuhan/yuchen/repos/processed_chery_data/lijiaoqiao_20260205_02"

    # Use FRONT_WIDE camera for segmentation
    SEGMENTATION_CAMERA = 'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110'

    # Camera priority for colorization (in order)
    COLORIZATION_CAMERAS = [
        'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60',    # 1st priority
        'CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110',   # 2nd priority
        'CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99',   # 3rd priority
        'CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99',    # 4th priority
    ]

    def __init__(self, data_root: Optional[str] = None,
                 sequence: Optional[str] = None,
                 seg_root: Optional[str] = None):
        """
        Initialize the processor.

        Args:
            data_root: Root directory containing test data
            sequence: Sequence folder name
            seg_root: Root directory containing segmentation results
        """
        self.data_root = data_root or self.DEFAULT_DATA_ROOT
        self.sequence = sequence or self.DEFAULT_SEQUENCE
        self.seg_root = seg_root or self.DEFAULT_SEG_ROOT

        self.sequence_path = os.path.join(self.data_root, self.sequence)

        # Load calibration parameters
        self.camera_params = load_camera_params(
            os.path.join(self.sequence_path, "camera_params.json")
        )
        self.lidar_params = load_lidar_params(
            os.path.join(self.sequence_path, "lidar_params.json")
        )

        # Output directory
        self.output_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "processed"
        )

    def load_segmentation_mask(self, frame_id: str, camera_name: str) -> np.ndarray:
        """Load segmentation mask for a frame."""
        seg_sequence = f"lijiaoqiao_20260205_02_offset_0.0m"
        seg_frame_path = os.path.join(self.seg_root, seg_sequence, frame_id)

        # Find segmentation mask file
        pattern = f"*{camera_name}*{frame_id}.png"
        import glob
        matches = glob.glob(os.path.join(seg_frame_path, pattern))

        if not matches:
            # Return all-zeros mask if file not found (all static)
            print(f"    Warning: No segmentation mask found, assuming all static")
            return np.zeros((512, 1024), dtype=np.uint8)

        # Load mask
        mask = cv2.imread(matches[0], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((512, 1024), dtype=np.uint8)

        # Convert to binary (1=dynamic, 0=static)
        binary_mask = (mask > 128).astype(np.uint8)

        return binary_mask

    def colorize_points_from_cameras(self, frame_path: str, points: np.ndarray) -> np.ndarray:
        """
        Colorize points from multiple cameras in priority order.

        Priority: Front_H60 -> Front_H110 -> FR -> FL
        Each camera only colors points that haven't been colored yet.

        Args:
            frame_path: Path to frame folder
            points: Nx3 point cloud in LiDAR coordinates

        Returns:
            Nx3 color array (RGB, 0-1 range)
        """
        colors = np.zeros((len(points), 3), dtype=np.float32)
        colored = np.zeros(len(points), dtype=bool)  # Track which points are colored

        # Process cameras in priority order
        for camera_name in self.COLORIZATION_CAMERAS:
            # Load camera image
            image_path = find_camera_image(frame_path, camera_name)
            if image_path is None:
                continue

            image = load_image(image_path)

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

            # Color only uncolored points
            for i in range(len(points)):
                if not colored[i] and valid[i]:
                    u, v = int(uv[i, 0]), int(uv[i, 1])
                    if 0 <= v < height and 0 <= u < width:
                        bgr = image[v, u]
                        colors[i] = [bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0]
                        colored[i] = True

        return colors

    def create_mask_projection_visualization(self, frame_path: str, frame_id: str,
                                             points: np.ndarray, dynamic_flags: np.ndarray,
                                             seg_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Create visualization showing point cloud projection on segmentation mask.

        Args:
            frame_path: Path to frame folder
            frame_id: Frame ID
            points: Nx3 point cloud in LiDAR coordinates
            dynamic_flags: N boolean array indicating dynamic points
            seg_mask: Segmentation mask used for classification

        Returns:
            Visualization image with LiDAR points projected on mask
        """
        try:
            # Get the segmentation camera (FRONT_H110)
            camera_name = self.SEGMENTATION_CAMERA

            # Load camera parameters and intrinsics
            cam_params = self.camera_params[camera_name]
            intrinsics = get_camera_intrinsics(self.camera_params, camera_name)
            K = build_intrinsic_matrix(intrinsics)
            width, height = intrinsics['width'], intrinsics['height']

            # Transform and project points
            lidar_to_cam = compute_lidar_to_cam_matrix(self.lidar_params, cam_params)
            points_camera = transform_points(points, lidar_to_cam)
            uv, depths, valid = project_points_to_image(points_camera, K, width, height)

            # Create mask visualization
            camera_label = camera_name.replace('CAM_PBQ_', '').replace('_RESET_OPTICAL', '')
            vis = create_mask_projection_visualization(seg_mask, uv, valid, camera_label)

            return vis

        except Exception as e:
            print(f"Warning: Failed to create mask visualization: {e}")
            return None

    def create_validation_visualization(self, frame_path: str, frame_id: str,
                                       points: np.ndarray, dynamic_flags: np.ndarray,
                                       seg_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Create validation visualization showing point cloud projections on all camera views
        and the segmentation mask.

        Args:
            frame_path: Path to frame folder
            frame_id: Frame ID
            points: Nx3 point cloud in LiDAR coordinates
            dynamic_flags: N boolean array indicating dynamic points
            seg_mask: Segmentation mask used for classification

        Returns:
            Combined visualization image or None if visualization failed
        """
        try:
            vis_images = []
            camera_labels = []

            # Process each camera in priority order
            for camera_name in self.COLORIZATION_CAMERAS:
                # Load camera image
                image_path = find_camera_image(frame_path, camera_name)
                if image_path is None:
                    continue

                image = load_image(image_path)

                # Load camera parameters and intrinsics
                cam_params = self.camera_params[camera_name]
                intrinsics = get_camera_intrinsics(self.camera_params, camera_name)
                K = build_intrinsic_matrix(intrinsics)
                width, height = intrinsics['width'], intrinsics['height']

                # Transform and project points
                lidar_to_cam = compute_lidar_to_cam_matrix(self.lidar_params, cam_params)
                points_camera = transform_points(points, lidar_to_cam)
                uv, depths, valid = project_points_to_image(points_camera, K, width, height)

                # Create visualization
                camera_label = camera_name.replace('CAM_PBQ_', '').replace('_RESET_OPTICAL', '')
                vis = create_projection_visualization(image, uv, valid, seg_mask, camera_label)
                vis_images.append(vis)
                camera_labels.append(camera_label)

            if not vis_images:
                return None

            # Create a 2x2 grid for 4 cameras
            rows = []
            for i in range(0, len(vis_images), 2):
                if i + 1 < len(vis_images):
                    # Resize images to same width and concatenate horizontally
                    h1, w1 = vis_images[i].shape[:2]
                    h2, w2 = vis_images[i+1].shape[:2]
                    if h1 != h2:
                        # Resize to same height
                        new_h = min(h1, h2)
                        vis_images[i] = cv2.resize(vis_images[i], (int(w1 * new_h/h1), new_h))
                        vis_images[i+1] = cv2.resize(vis_images[i+1], (int(w2 * new_h/h2), new_h))
                    row = np.hstack([vis_images[i], vis_images[i+1]])
                else:
                    row = vis_images[i]
                rows.append(row)

            # Concatenate rows vertically
            combined = np.vstack(rows)

            # Add frame ID title
            cv2.putText(combined, f"Frame: {frame_id}", (10, combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            return combined

        except Exception as e:
            print(f"Warning: Failed to create visualization: {e}")
            return None

    def segment_single_frame(self, frame_path: str, frame_id: str,
                            camera_name: str = None) -> Dict:
        """
        Segment a single frame.

        Args:
            frame_path: Path to frame folder
            frame_id: Frame ID
            camera_name: Camera to use for segmentation

        Returns:
            Dictionary with static and dynamic point clouds
        """
        if camera_name is None:
            camera_name = self.SEGMENTATION_CAMERA

        # Load point cloud
        pcd_path = find_lidar_front_file(frame_path)
        if pcd_path is None:
            return None

        points, _ = load_pcd(pcd_path)

        # Load vehicle pose
        pb_path = os.path.join(frame_path, "data_frame.pb.txt")
        vehicle_pose = parse_vehicle_pose(pb_path)

        # Colorize points from multiple cameras in priority order
        colors = self.colorize_points_from_cameras(frame_path, points)

        # Load segmentation mask (use FRONT_H110 for segmentation)
        seg_mask = self.load_segmentation_mask(frame_id, camera_name)

        # Use dilation to expand dynamic regions
        from scipy.ndimage import binary_dilation
        expanded_mask = binary_dilation(seg_mask, iterations=15)

        # Project points to segmentation camera for classification
        seg_cam_params = self.camera_params[camera_name]
        seg_intrinsics = get_camera_intrinsics(self.camera_params, camera_name)
        seg_K = build_intrinsic_matrix(seg_intrinsics)
        seg_width, seg_height = seg_intrinsics['width'], seg_intrinsics['height']

        lidar_to_seg_cam = compute_lidar_to_cam_matrix(self.lidar_params, seg_cam_params)
        points_seg_cam = transform_points(points, lidar_to_seg_cam)
        uv_seg, depths_seg, valid_seg = project_points_to_image(points_seg_cam, seg_K, seg_width, seg_height)

        # Classify points as static/dynamic based on segmentation mask
        dynamic_flags = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if valid_seg[i]:
                u, v = int(uv_seg[i, 0]), int(uv_seg[i, 1])
                if 0 <= v < expanded_mask.shape[0] and 0 <= u < expanded_mask.shape[1]:
                    if expanded_mask[v, u] == 1:
                        dynamic_flags[i] = True

        # Split point cloud
        static_indices = ~dynamic_flags
        dynamic_indices = dynamic_flags

        static_points = points[static_indices]
        static_colors = colors[static_indices]

        dynamic_points = points[dynamic_indices]
        dynamic_colors = colors[dynamic_indices]

        # Transform to world coordinates
        lidar_to_world = compute_lidar_to_world_matrix(self.lidar_params, vehicle_pose)
        static_points_world = transform_points(static_points, lidar_to_world)
        dynamic_points_world = transform_points(dynamic_points, lidar_to_world)

        return {
            'frame_id': frame_id,
            'static_points': static_points_world,
            'static_colors': static_colors,
            'dynamic_points': dynamic_points_world,
            'dynamic_colors': dynamic_colors,
            'camera_name': camera_name,
            'points': points,  # Original points for visualization
            'dynamic_flags': dynamic_flags,  # For visualization
            'seg_mask': seg_mask,  # For visualization
        }

    def save_frame_point_clouds(self, frame_data: Dict, output_dir: str,
                                frame_path: str = None, save_vis: bool = False):
        """
        Save point clouds for a single frame.

        Args:
            frame_data: Frame data dictionary
            output_dir: Output directory
            frame_path: Path to frame folder (for visualization)
            save_vis: Whether to save visualization image
        """
        frame_id = frame_data['frame_id']

        # Create frame output directory
        frame_output_dir = os.path.join(output_dir, frame_id)
        os.makedirs(frame_output_dir, exist_ok=True)

        # Save static point cloud
        static_ply = os.path.join(frame_output_dir, f"{frame_id}_static.ply")
        if len(frame_data['static_points']) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(frame_data['static_points'])
            pcd.colors = o3d.utility.Vector3dVector(frame_data['static_colors'])
            o3d.io.write_point_cloud(static_ply, pcd)

        # Save dynamic point cloud
        dynamic_ply = os.path.join(frame_output_dir, f"{frame_id}_dynamic.ply")
        if len(frame_data['dynamic_points']) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(frame_data['dynamic_points'])
            pcd.colors = o3d.utility.Vector3dVector(frame_data['dynamic_colors'])
            o3d.io.write_point_cloud(dynamic_ply, pcd)

        # Save visualization if requested
        if save_vis and frame_path is not None:
            # Camera image visualization
            vis = self.create_validation_visualization(
                frame_path, frame_id,
                frame_data['points'],
                frame_data['dynamic_flags'],
                frame_data['seg_mask']
            )
            if vis is not None:
                vis_path = os.path.join(frame_output_dir, f"{frame_id}_visualization.png")
                cv2.imwrite(vis_path, vis)

            # Mask projection visualization
            mask_vis = self.create_mask_projection_visualization(
                frame_path, frame_id,
                frame_data['points'],
                frame_data['dynamic_flags'],
                frame_data['seg_mask']
            )
            if mask_vis is not None:
                mask_vis_path = os.path.join(frame_output_dir, f"{frame_id}_mask_projection.png")
                cv2.imwrite(mask_vis_path, mask_vis)

    def process_sequence(self, start_frame: int = 17000010, end_frame: int = 17000040,
                        save_visualization: bool = False):
        """
        Process entire sequence.

        Args:
            start_frame: Starting frame ID
            end_frame: Ending frame ID
            save_visualization: Whether to save projection visualization images
        """
        print("=" * 60)
        print("Sequence Segmentation Processing")
        print("=" * 60)
        print(f"Sequence: {self.sequence}")
        print(f"Frames: {start_frame} - {end_frame}")
        print(f"Save visualization: {save_visualization}")
        print()

        # Get frame folders directly from directory
        import os
        frame_folders = []
        for item in os.listdir(self.sequence_path):
            item_path = os.path.join(self.sequence_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                frame_id = int(item)
                if start_frame <= frame_id <= end_frame:
                    frame_folders.append(item_path)

        frame_folders.sort()
        print(f"Found {len(frame_folders)} frames to process")
        print()

        # Create output directory structure
        sequence_output_dir = os.path.join(self.output_root, self.sequence)
        os.makedirs(sequence_output_dir, exist_ok=True)

        # Process each frame
        all_static_points = []
        all_static_colors = []

        for frame_folder in tqdm(frame_folders, desc="Processing frames"):
            frame_id = os.path.basename(frame_folder)

            # Segment frame
            frame_data = self.segment_single_frame(frame_folder, frame_id)

            if frame_data is None:
                continue

            # Save frame point clouds (with optional visualization)
            self.save_frame_point_clouds(frame_data, sequence_output_dir,
                                        frame_path=frame_folder,
                                        save_vis=save_visualization)

            # Collect static points for merging
            all_static_points.append(frame_data['static_points'])
            all_static_colors.append(frame_data['static_colors'])

            print(f"  Frame {frame_id}: Static={len(frame_data['static_points'])}, "
                  f"Dynamic={len(frame_data['dynamic_points'])}, "
                  f"Camera={frame_data['camera_name']}")

        print()
        print("=" * 60)
        print("Merging all static point clouds...")
        print("=" * 60)

        # Merge all static point clouds
        if all_static_points:
            merged_points, merged_colors = merge_point_clouds(all_static_points, all_static_colors)

            print(f"Merged {len(all_static_points)} frames")
            print(f"Total static points: {len(merged_points)}")

            # Save merged point cloud
            merged_ply = os.path.join(sequence_output_dir, "merged_static.ply")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged_points)
            pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            o3d.io.write_point_cloud(merged_ply, pcd)

            print(f"Saved merged point cloud: {merged_ply}")
        else:
            print("No static point clouds to merge")

        print()
        print("=" * 60)
        print("Processing complete!")
        print(f"Output saved to: {sequence_output_dir}")
        print("=" * 60)

        return {
            'output_dir': sequence_output_dir,
            'num_frames': len(all_static_points),
            'merged_points': len(merged_points) if all_static_points else 0,
        }


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Process sequence with segmentation')
    parser.add_argument('--no-vis', action='store_true',
                        help='Disable visualization output')
    args = parser.parse_args()

    processor = SequenceSegmentationProcessor()

    # Process all frames in the sequence
    # Frame range: 17000010 to 17000907 (approx 900 frames)
    save_vis = not args.no_vis
    results = processor.process_sequence(
        start_frame=17000010,
        end_frame=17100000,
        save_visualization=save_vis
    )

    return results


if __name__ == "__main__":
    main()
