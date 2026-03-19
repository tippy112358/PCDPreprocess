import torch
import os
import pandas as pd
import imageio
from torchvision.transforms import v2
import torchvision
from PIL import Image
import numpy as np
from einops import rearrange
import re
import random
import json

class Camera(object):
    def __init__(self, c2w):
        c2w = np.linalg.inv(np.array(c2w).reshape(4, 4))
        c2w[:3,3]*=1.5
        # print(c2w)
        swap_axes = np.array([
            [0, 0, -1, 0],  # x -> z
            [0, 1, 0, 0],  # y -> x
            [-1, 0, 0, 0],  # z -> y
            [0, 0, 0, 1]
        ])
        
        # # 应用轴交换（左乘）
        c2w = swap_axes @ c2w @ np.linalg.inv(swap_axes)
        # print(c2w)
        # import pdb
        # pdb.set_trace()
        self.c2w_mat = c2w
        self.w2c_mat = np.linalg.inv(c2w)


 

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832):
        self.base_path = base_path
        self.metadata= pd.read_csv(os.path.join(base_path, "metadata.csv"))
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = False  # Set to True if the dataset is for image-to-video tasks
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        start_frame_id=  reader.count_frames() - 1 - (num_frames - 1) * interval
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame
    
    def __getitem__(self, data_id):
        data_id=(data_id)%len(self.metadata)
        path = self.metadata["data_path"][data_id]
        while True:
            try:
                assert os.path.exists(os.path.join(self.base_path, path)), f"Path {path} does not exist."
                assert os.path.exists(os.path.join(self.base_path, path, "source_rgb.mp4")), f"Source video not found in {os.path.join(self.base_path, path, "source_rgb.mp4")}."
                assert os.path.exists(os.path.join(self.base_path, path, "render_rgb.mp4")), f"Render video not found in {path}."
                assert os.path.exists(os.path.join(self.base_path, path, "render_lidar.mp4")), f"Render video not found in {path}."
                assert os.path.exists(os.path.join(self.base_path, path, "render_lidar_mask.mp4")), f"Mask video not found in {path}."
                assert os.path.exists(os.path.join(self.base_path, path, "render_rgb_mask.mp4")), f"Mask video not found in {path}."
                assert os.path.exists(os.path.join(self.base_path, path, "target_rgb.mp4")), f"Target video not found in {path}."
                source_video = self.load_video(os.path.join(self.base_path, path, "source_rgb.mp4"))
                render_rgb = self.load_video(os.path.join(self.base_path, path, "render_rgb.mp4"))
                render_lidar = self.load_video(os.path.join(self.base_path, path, "render_lidar.mp4"))
                target_video = self.load_video(os.path.join(self.base_path, path, "target_rgb.mp4"))
                data = {
                    "text": self.metadata["prompt"][data_id],
                    "source_video": source_video,
                    "render_rgb": render_rgb,
                    "render_lidar": render_lidar,
                    "target_video": target_video,
                    "path": os.path.join(self.base_path, path)
                }
                break
            except Exception as e:
                print(f"Error loading {path}: {e}")
                data_id += 1
                if data_id >= len(self.metadata):
                    raise IndexError("No more videos to load.")
        return data

    def __len__(self):
        return len(self.metadata)
    
    
class TensorControlDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch, height=480, width=832,render_type="rgb",need_extra_cond=True,need_raw_lidar=True):
        self.base_path=base_path
            
        self.metadata= pd.read_csv(os.path.join(base_path, "metadata.csv"))
        self.mask_name =f"render_{render_type}_mask.mp4"
        self.render_name = f"render_{render_type}_latents"
        extra_cond = "lidar" if render_type == "rgb" else "rgb"
        if need_extra_cond:
            self.extra_cond_name = f"render_{extra_cond}_latents"
            self.extra_cond_mask_name=f"render_{extra_cond}_mask.mp4"
        else:
            self.extra_cond_name=None
            self.extra_cond_mask_name = None
        
        if need_raw_lidar:
            self.lidar_name=f"render_lidar.mp4"
        else:
            self.lidar_name=None
        print(len(self.metadata), "tensors cached in metadata.")
        assert len(self.metadata) > 0
        self.steps_per_epoch = steps_per_epoch
        self.num_frames=81
        self.height = height
        self.width = width
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def crop_and_resize(self, image, zero_as_one=False):
        width, height = image.size
        scale = max(self.width / width, self.height / height)

        new_h, new_w = round(height * scale), round(width * scale)

        if not zero_as_one:
            # 普通 resize
            image = torchvision.transforms.functional.resize(
                image,
                (new_h, new_w),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )
            return image

        # -------- zero_padding 模式 (RGB) --------
        arr = np.array(image)  # H, W, 3
        h, w = arr.shape[:2]
        scale_h = new_h / h  # 计算高度缩放比例
        scale_w = new_w / w  # 计算宽度缩放比例
        
        # 非零点位置 (不是全 0 的像素)
        nonzero_mask = np.any(arr != 0, axis=-1)
        nonzero_y, nonzero_x = np.nonzero(nonzero_mask)
        
        # 计算新位置 - 向量化操作，避免循环
        ny = np.rint(nonzero_y * scale_h).astype(np.int32)
        nx = np.rint(nonzero_x * scale_w).astype(np.int32)
        
        # 过滤掉超出边界的索引
        valid = (ny >= 0) & (ny < new_h) & (nx >= 0) & (nx < new_w)
        ny, nx, nonzero_y, nonzero_x = ny[valid], nx[valid], nonzero_y[valid], nonzero_x[valid]
        
        # 初始化新图
        new_arr = np.zeros((new_h, new_w, 3), dtype=np.float32)
        count_arr = np.zeros((new_h, new_w), dtype=np.int32)
        
        # 使用高级索引进行累加 - 向量化操作
        new_arr[ny, nx] += arr[nonzero_y, nonzero_x]
        count_arr[ny, nx] += 1
        
        # 取平均值（只对非零点位置）
        mask = count_arr > 0
        new_arr[mask] /= count_arr[mask, None]
        
        return Image.fromarray(new_arr.astype(np.uint8))

        
    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.eye(4, dtype=np.float32)
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_mask_video(self, file_path,zero_as_one=False):
        # 加载mask视频，做resize，转tensor，格式为 (C=1, T, H, W)
        reader = imageio.get_reader(file_path)
        frames = []
        for frame_id in range(self.num_frames):
            frame = reader.get_data(frame_id)  # shape (H, W) or (H, W, 1)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame,zero_as_one=zero_as_one)
            frame = self.frame_process(frame)  # 0-1 float tensor
            frames.append(frame)
        reader.close()
        frames = torch.stack(frames, dim=1)  # (C, T, H, W)
        # print(frames.shape)
        return frames
    
    
    def __getitem__(self, index):
        while True:
            try:
                data_id = torch.randint(0, len(self.metadata ), (1,))[0]
                data_id = (data_id + int(index)) % len(self.metadata)
                data_id = int(data_id)
                path = self.metadata["data_path"][data_id]
                assert os.path.exists(os.path.join(self.base_path, path, self.mask_name)), f"Mask video not found in {path}."
                # Load the source video
                render_mask = self.load_mask_video(os.path.join(self.base_path, path, self.mask_name))

                

                # Load the processed tensors
                path_processed = os.path.join(self.base_path, path, "processed.tensors.pth")
                if not os.path.exists(path_processed):
                    raise FileNotFoundError(f"Processed tensors not found at {path_processed}")
                data = torch.load(path_processed, weights_only=True, map_location="cpu")


                
                # Compose render_latent: [hidden_latent, cond_latent]
                x = torch.cat([data['target_latents'], data['latents']], dim=1)
                cam_data= json.load(open(os.path.join(self.base_path, path, "cameras_extrinsics.json"), 'r'))
                # Parse relative poses
                cam_idx = list(range(81))[::4]
                if self.extra_cond_name and self.extra_cond_mask_name:
                    try:
                        extra_cond_mask=self.load_mask_video(os.path.join(self.base_path, path, self.extra_cond_mask_name),zero_as_one=True)
                        extra_cond=data[self.extra_cond_name]
                    except:
                        extra_cond=None
                        extra_cond_mask= None
                

                if self.lidar_name:
                    try:
                        lidar_raw=self.load_mask_video(os.path.join(self.base_path, path, self.lidar_name),zero_as_one=True)
                        # print(lidar_raw.shape)
                    except:
                        lidar_raw=None

                cond_cam_params = [Camera(cam) for cam in cam_data['source']]
                tgt_cam_params = [Camera(cam) for cam in cam_data['target']]
                cam_idx = list(range(len(tgt_cam_params)))[::4]
                relative_poses = []
                for i in cam_idx:
                    relative_pose = self.get_relative_pose([cond_cam_params[0], tgt_cam_params[i]])
                    relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
                pose_embedding = torch.stack(relative_poses, dim=0)  # (21, 3, 4)
                camera_embedding = rearrange(pose_embedding, 'b c d -> b (c d)').to(torch.bfloat16)

                

                return {
                    "x": x,                     # (B, T*2, C, H, W)
                    "latents": data['latents'],               # (B, T, C, H, W)
                    "render_latents": data[self.render_name],         # (B, T, C, H, W)
                    "render_mask": render_mask,             # (B, T, 1, H, W)
                    "target_latents": data['target_latents'],  # (B, T, C, H, W)
                    "camera_embedding": camera_embedding,   # (B, 12)
                    "prompt_emb": data["prompt_emb"],
                    "image_emb": {},  # Optional
                    "extra_cond":extra_cond,
                    "extra_cond_mask":extra_cond_mask,
                    "lidar_raw":lidar_raw


                }

            except Exception as e:
                # raise e
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.metadata))

    def __len__(self):
        return self.steps_per_epoch

class TextVideoCameraDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832,render_type="rgb"):
        self.base_path = base_path
        self.metadata= pd.read_csv(os.path.join(base_path, "metadata.csv"))
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mask_name =f"render_{render_type}_mask.mp4"
        self.render_name = f"render_{render_type}.mp4"
        self.is_i2v = False  # Set to True if the dataset is for image-to-video tasks
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.lidar_name=f"render_lidar.mp4"

    # def crop_and_resize(self, image):
    #     width, height = image.size
    #     scale = max(self.width / width, self.height / height)
    #     image = torchvision.transforms.functional.resize(
    #         image,
    #         (round(height*scale), round(width*scale)),
    #         interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    #     )
    #     return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < 81:
            reader.close()
            return None
        start_frame_id=  reader.count_frames() - 1 - (num_frames - 1) * interval
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * 1)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.eye(4, dtype=np.float32)
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame
    
    def load_mask_video(self, file_path,zero_as_one=False):
        # 加载mask视频，做resize，转tensor，格式为 (C=1, T, H, W)
        reader = imageio.get_reader(file_path)
        frames = []
        for frame_id in range(self.num_frames):
            frame = reader.get_data(frame_id)  # shape (H, W) or (H, W, 1)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame,zero_as_one=zero_as_one)
            frame = self.frame_process(frame)  # 0-1 float tensor
            frames.append(frame)
        reader.close()
        frames = torch.stack(frames, dim=1)  # (C, T, H, W)
        # print(frames.shape)
        return frames
    
    def crop_and_resize(self, image, zero_as_one=False):
        width, height = image.size
        scale = max(self.width / width, self.height / height)

        new_h, new_w = round(height * scale), round(width * scale)

        if not zero_as_one:
            # 普通 resize
            image = torchvision.transforms.functional.resize(
                image,
                (new_h, new_w),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )
            return image

        # -------- zero_padding 模式 (RGB) --------
        arr = np.array(image)  # H, W, 3
        h, w = arr.shape[:2]
        scale_h = new_h / h  # 计算高度缩放比例
        scale_w = new_w / w  # 计算宽度缩放比例
        
        # 非零点位置 (不是全 0 的像素)
        nonzero_mask = np.any(arr != 0, axis=-1)
        nonzero_y, nonzero_x = np.nonzero(nonzero_mask)
        
        # 计算新位置 - 向量化操作，避免循环
        ny = np.rint(nonzero_y * scale_h).astype(np.int32)
        nx = np.rint(nonzero_x * scale_w).astype(np.int32)
        
        # 过滤掉超出边界的索引
        valid = (ny >= 0) & (ny < new_h) & (nx >= 0) & (nx < new_w)
        ny, nx, nonzero_y, nonzero_x = ny[valid], nx[valid], nonzero_y[valid], nonzero_x[valid]
        
        # 初始化新图
        new_arr = np.zeros((new_h, new_w, 3), dtype=np.float32)
        count_arr = np.zeros((new_h, new_w), dtype=np.int32)
        
        # 使用高级索引进行累加 - 向量化操作
        new_arr[ny, nx] += arr[nonzero_y, nonzero_x]
        count_arr[ny, nx] += 1
        
        # 取平均值（只对非零点位置）
        mask = count_arr > 0
        new_arr[mask] /= count_arr[mask, None]
        
        return Image.fromarray(new_arr.astype(np.uint8))


    def __getitem__(self, data_id):
        path = self.metadata["data_path"][data_id]
        while True:
            try:
                assert os.path.exists(os.path.join(self.base_path, path)), f"Path {os.path.join(self.base_path, path)} does not exist."
                assert os.path.exists(os.path.join(self.base_path, path, "source_rgb.mp4")), f"Source video not found in {path}."
                assert os.path.exists(os.path.join(self.base_path, path, self.render_name)), f"Render video not found in {path}."
                assert os.path.exists(os.path.join(self.base_path, path, self.mask_name)), f"Mask video not found in {path}."
                assert os.path.exists(os.path.join(self.base_path, path, "target_rgb.mp4")), f"Mask video not found in {path}."
                source_video = self.load_video(os.path.join(self.base_path, path, "source_rgb.mp4"))
                target_video = self.load_video(os.path.join(self.base_path, path, "target_rgb.mp4"))
                render_video = self.load_video(os.path.join(self.base_path, path, self.render_name))
                lidar_video=self.load_mask_video(os.path.join(self.base_path, path, self.lidar_name),zero_as_one=True)
                
                mask_video = self.load_video(os.path.join(self.base_path, path, self.mask_name))
                cam_data= json.load(open(os.path.join(self.base_path, path, "cameras_extrinsics.json"), 'r'))
                # Parse relative poses
                cond_cam_params = [Camera(cam) for cam in cam_data['source']]
                tgt_cam_params = [Camera(cam) for cam in cam_data['target']]
                cam_idx = list(range(len(tgt_cam_params)-81,len(tgt_cam_params)))[::4]

                # cam_idx = list(range(len(tgt_cam_params)))[::4]
                relative_poses = []
                for i in cam_idx:
                    relative_pose = self.get_relative_pose([cond_cam_params[i], tgt_cam_params[i]])
                    relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
                pose_embedding = torch.stack(relative_poses, dim=0)  # (21, 3, 4)
                camera_embedding = rearrange(pose_embedding, 'b c d -> b (c d)').to(torch.bfloat16)
                data = {
                    "text": self.metadata["prompt"][data_id],
                    "source_video": source_video,
                    "render_video": render_video,
                    "target_video": target_video,
                    "mask_video": mask_video,
                    "lidar_video": lidar_video,
                    "camera": camera_embedding,  # (B, 12)
                    "path": path
                }
                break
            except Exception as e:
                print(f"Error loading {path}: {e}")
                raise e
                data_id += 1
                if data_id >= len(self.metadata):
                    raise IndexError("No more videos to load.")
        return data

    def __len__(self):
        return len(self.metadata)
