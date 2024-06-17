"""
Script for the stage of 3D Prompt Proposal in the paper

Author: Mutian Xu (mutianxu@link.cuhk.edu.cn) and Xingyilang Yin
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
import os
import cv2
import argparse
import torch
import numpy as np
import open3d as o3d
import pointops
from utils.main_utils import *
from utils.sam_utils import *
from segment_anything import sam_model_registry, SamPredictor
from tqdm import trange
from PIL import Image
import json
from pathlib import Path

def create_output_folders(args):
    # Create folder to save SAM outputs:
    create_folder(args.sam_output_path)
    # Create folder to save SAM outputs for a specific scene:
    create_folder(os.path.join(args.sam_output_path))
    # Create subfolder for saving different output types:
    create_folder(os.path.join(args.sam_output_path, 'points_npy'))
    create_folder(os.path.join(args.sam_output_path, 'iou_preds_npy'))
    create_folder(os.path.join(args.sam_output_path, 'masks_npy'))
    create_folder(os.path.join(args.sam_output_path, 'corre_3d_ins_npy'))


def prompt_init(xyz, rgb, voxel_size, device):
    # Here we only use voxelization to decide the number of fps-sampled points, \
    # since voxel_size is more controllable. We use fps later for prompt initialization
    # if len(xyz) > 2_000_000:
    #     xyz = torch.from_numpy(xyz).cuda().contiguous()
    #     idx = pointops.farthest_point_sampling(xyz, torch.cuda.IntTensor([len(xyz)]), torch.cuda.IntTensor([2_000_000]))
    #     xyz = xyz[idx.long(), :]
    #     xyz = xyz.cpu().numpy()
    idx_sort, num_pt = voxelize(xyz, voxel_size, mode=1)
    print("the number of initial 3D prompts:", len(num_pt))
    xyz = torch.from_numpy(xyz).cuda().contiguous()
    o, n_o = len(xyz), len(num_pt)
    o, n_o = torch.cuda.IntTensor([o]), torch.cuda.IntTensor([n_o])
    idx = pointops.farthest_point_sampling(xyz, o, n_o)
    fps_points = xyz[idx.long(), :]
    fps_points = torch.from_numpy(fps_points.cpu().numpy()).to(device=device)
    rgb = rgb / 256.
    rgb = torch.from_numpy(rgb).cuda().contiguous()
    fps_colors = rgb[idx.long(), :]
    
    return fps_points, fps_colors
    

def save_init_prompt(xyz, rgb, args):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    point_cloud.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
    prompt_ply_file = os.path.join(args.prompt_path, 'init_prompt.ply')
    o3d.io.write_point_cloud(prompt_ply_file, point_cloud)
    
    
def process_batch(
    mask,
    points: torch.Tensor,
    ins_idxs: torch.Tensor,
    im_size: Tuple[int, ...],
) -> MaskData:
    masks = []
    iou_preds = []
    device = mask.device
    batch_size = points.shape[0]
    h, w = mask.shape
    
    x = torch.clamp(points[:, 1].round().long(), 0, h-1)
    y = torch.clamp(points[:, 0].round().long(), 0, w-1)
    ins_id = mask[x, y]
    
    iou_preds = torch.ones(batch_size, dtype=torch.float)
    iou_preds[ins_id == 0] = 0.0
    iou_preds = iou_preds.view(batch_size, 1).to(device)
    
    ins_id = ins_id.view(batch_size, 1)
    input_mask = mask.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    masks = ins_id.repeat(1, h*w).view(batch_size, h, w).to(device)
    masks[~(masks == input_mask)] = 0
    masks = masks.view(batch_size, 1, h, w)
    
    data_original = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=points, 
        corre_3d_ins=ins_idxs 
    )

    return data_original
    
def generate_sam_gt_output(transforms, frame_id_init, frame_id_end, init_prompt, args):
    for i in trange(frame_id_init, frame_id_end):
        frame_id = i
        frame_name = "frame_{:05d}".format(frame_id + 1)
        
        image_dir = args.data_path / 'images' / f"{frame_name}.png"
        image = cv2.imread(image_dir.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load the intrinsic
        depth_intrinsic = torch.eye(4, dtype=torch.float64).to(device=args.device)
        depth_intrinsic[0, 0] = transforms["fl_x"]
        depth_intrinsic[1, 1] = transforms["fl_y"]
        depth_intrinsic[0, 2] = transforms["cx"]
        depth_intrinsic[1, 2] = transforms["cy"]
        # Load the depth, and pose
        depth_dir = args.data_path / 'depth' / f"{frame_name}.tif"
        depth = cv2.imread(depth_dir.as_posix(), -1)
        depth = torch.from_numpy(depth.astype(np.float64)).to(device=args.device)
        pose = torch.tensor(transforms["frames"][i]["transform_matrix"], dtype=torch.float64).to(device=args.device)
        mask_dir = args.data_path / 'instance' / f"{frame_name}.png"
        mask = torch.from_numpy(np.array(Image.open(mask_dir.as_posix()))).to(device=args.device)
        # print(pose)
        # print(depth.max())
        # exit()
        if str(pose[0, 0].item()) == '-inf': # skip frame with '-inf' pose
            print(f'skip frame {frame_id}')
            continue

        # 3D-2D projection
        input_point_pos, corre_ins_idx = transform_pt_depth_scannet_torch(init_prompt, depth_intrinsic, depth, pose, args.device)  # [valid, 2], [valid]
        if input_point_pos.shape[0] == 0 or input_point_pos.shape[1] == 0:
            print(f'skip frame {frame_id}')
            continue

        image_size = image.shape[:2]
        data_original = MaskData()
        for (points, ins_idxs) in batch_iterator(64, input_point_pos, corre_ins_idx):
            batch_data_original = process_batch(mask, points, ins_idxs, image_size)
            data_original.cat(batch_data_original)
            del batch_data_original
        data_original.to_numpy()
        save_file_name = str(frame_id) + ".npy"
        np.save(args.sam_output_path / "points_npy" / save_file_name, data_original["points"])
        np.save(args.sam_output_path / "masks_npy" / save_file_name, data_original["masks"])  
        np.save(args.sam_output_path / "iou_preds_npy" / save_file_name, data_original["iou_preds"])  
        np.save(args.sam_output_path / "corre_3d_ins_npy" / save_file_name, data_original["corre_3d_ins"])

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate 3d prompt proposal on ScanNet.")
    # for voxelization to decide the number of fps-sampled points:
    parser.add_argument('--voxel_size', default=0.2, type=float, help='Size of voxels.')
    # path arguments:
    parser.add_argument('--data_path', default="/data/luoly/dataset/assist/530_scannet_table0_copy", type=Path, help='Path to the dataset.')
    parser.add_argument("--device", default="cuda:0", type=str, help="The device to run generation on.")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = get_args()
    args.prompt_path = args.data_path / "sampro3d_outputs" / "initial_prompt"
    args.sam_output_path  = args.data_path / "sampro3d_outputs" / "SAM_outputs"
    print("Arguments:")
    print(args)
    # Initialize SAM:
    device = torch.device(args.device)
    # Load all 3D points of the input scene:
    scene_plypath = args.data_path / 'dense_pc.ply'
    xyz, rgb = load_ply(scene_plypath)

    # 3D prompt initialization:
    init_prompt, init_color = prompt_init(xyz, rgb, args.voxel_size, device)
    # save the initial 3D prompts for later use:
    create_folder(args.prompt_path)
    save_init_prompt(init_prompt, init_color, args)

    # SAM segmentation on image frames:
    # create folder to save diffrent SAM output types for later use (note that this is the only stage to perform SAM):
    create_output_folders(args)  # we use npy files to save different output types for faster i/o and clear split
    # perform SAM on each 2D RGB frame:
    frame_id_init = 0
    with open(args.data_path / "transforms.json") as f:
        transforms = json.load(f)
    frame_id_end = len(transforms["frames"])
    # You can define frame_id_init and frame_id_end by yourself for segmenting partial point clouds from limited frames. Sometimes partial result is better!
    print("Start performing SAM segmentations on {} 2D frames...".format(frame_id_end))

    generate_sam_gt_output(transforms, frame_id_init, frame_id_end, init_prompt, args)
    print("Finished performing SAM segmentations!")