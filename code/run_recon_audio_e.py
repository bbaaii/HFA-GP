import torch
import torch.nn as nn
# from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# import torch
# from networks.discriminator import Discriminator
# from networks.generator import Generator
from networks.headnerf import HeadNeRF, HeadNeRF_Audio, Weights_3DMM, AudioNet, AudioAttNet
import torch.nn.functional as F
# from torch import nn, optim
# import os
import argparse
import os
import torch
from torch.utils import data
from dataset import HeadData, HeadData_Audio, HeadData_Audio_french
import torchvision
import torchvision.transforms as transforms
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp

import math

import torchvision.utils as utils

import wandb
import torch
import imageio
from PIL import Image
import numpy as np
import math
import mrcfile
import torch.nn.functional as F
from tqdm import tqdm
import argparse

import glob
def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

trans =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def save_video(vid_target_recon, save_path, fps=50):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)

def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)
def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """

    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)

    else:
        # Just use the mean.
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta


# def points2hv(points,device, n=1, r=2.7):
#     phi = torch.acos(points[:, 1:2]/r) 

#     sin_theta = points[:, 2:3] / (r*torch.sin(phi))
#     sin_theta = torch.clamp(sin_theta, -1, 1)
#     theta = torch.asin(sin_theta)
#     h = theta / math.pi
#     v = phi / math.pi
#     return h, v
def arctan(y,x):
    if x > 0:
        return torch.atan(y/x)
    elif x < 0 and y >= 0:
        return torch.atan(y/x) + math.pi
    elif x < 0 and y < 0:
        return torch.atan(y/x) - math.pi
    elif x == 0 and y > 0:
        return  0.5 * math.pi
    elif x == 0 and y < 0:
        return  -0.5 * math.pi
    elif x == 0 and y == 0:
        return  0


def points2hv(points,device, n=1, r=2.7):
    # theta = torch.acos(points[:, 1:2]/torch.sqrt( (points[:, 0:1])*(points[:, 0:1]) +  (points[:, 1:2])*(points[:, 1:2])  + (points[:, 2:3])*(points[:, 2:3]) ))
    phi = torch.acos(points[:, 1:2]/r) 
    # theta = arctan(points[:, 1:2],points[:, 0:1])

    sin_phi = torch.sqrt(1- (points[:, 1:2]/r)*(points[:, 1:2]/r))
    sin_theta = points[:, 2:3] / (r*sin_phi)
    cos_theta = points[:, 0:1] / (r*sin_phi)
    # print('11111111111111')
    # print(sin_theta*sin_theta+cos_theta*cos_theta)
    # print(sin_theta)
    # print(cos_theta)
    theta = torch.acos(cos_theta)
    if sin_theta > 1 or sin_theta<-1:
        theta = torch.acos(cos_theta)

    # sin_theta = torch.clamp(sin_theta, -1, 1)
    if cos_theta > 1 or cos_theta <-1:
        theta = torch.asin(sin_theta)
    h = theta / math.pi
    v = phi / math.pi
    return h, v    
# def create_cam2world_matrix(forward_vector, origin, device=None):
#     """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

#     forward_vector = normalize_vecs(forward_vector)
#     print('forward_vector')
#     print(forward_vector)
#     up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)
#     print('up_vector')
#     print(up_vector)

#     left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
#     print('left_vector')
#     print(left_vector)

#     up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))
#     print('up_vector')
#     print(up_vector)

#     rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
#     print('rotation_matrix')
#     print(rotation_matrix)
#     rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)
#     print('rotation_matrix')
#     print(rotation_matrix)

#     translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
#     print('translation_matrix')
#     print(translation_matrix)
#     translation_matrix[:, :3, 3] = origin
#     print('translation_matrix')
#     print(translation_matrix)

#     cam2world = translation_matrix @ rotation_matrix
#     print('cam2world')
#     print(cam2world)

#     return cam2world

def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world

def create_world2cam_matrix(forward_vector, origin):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    cam2world = create_cam2world_matrix(forward_vector, origin, device=device)
    world2cam = torch.inverse(cam2world)
    return world2cam

def main( args):
    # init distributed computing
    # ddp_setup(args, rank, world_size)
    # torch.cuda.set_device(rank)
    device = torch.device("cuda")

    transform = torchvision.transforms.Compose([
        transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset_test = HeadData_Audio(args.dataset_type, transform, dataset = args.dataset , person = args.person_1 )

    loader_test = data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=1, rank=0, shuffle=False),
        pin_memory=True,
        drop_last=False,
    )
    model_path = args.model_path
    # model_path = '/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/exps/nerface2-v1/checkpoint/874999.pt'
    save_path = f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/demo/{args.demo_name}/'
    # embedding_dir = f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/embeddings/'
    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(embedding_dir, exist_ok=True)
    ckpt = torch.load(model_path)

    # weights_3dmm = Weights_3DMM(input_dim = args.params_len, use_softmax = False).to(
    #         device)
    AudNet = AudioNet(args.dim_aud, args.win_size).to(device)
    AudAttNet = AudioAttNet().to(device)
    # gen = HeadNeRF3(args, args.size, device, args.latent_dim_style, args.latent_dim_shape, args.run_id, args.emb_dir).cuda()
    # gen = HeadNeRF2(args, args.size, device, args.latent_dim_style, args.latent_dim_shape, args.run_id, args.emb_dir).cuda()
    # gen = HeadNeRF2(args, args.size, device, args.latent_dim_style, args.latent_dim_shape, args.run_id, args.emb_dir, use_softmax = False).to(
    #         device)
    gen = HeadNeRF_Audio(args, args.size, device, args.latent_dim_style, args.latent_dim_shape, args.run_id, args.emb_dir).to(
            device)
    # weights_3dmm.load_state_dict(ckpt["weights_3dmm"])
    AudNet.load_state_dict(ckpt["AudNet"])
    AudAttNet.load_state_dict(ckpt["AudAttNet"])
    gen.load_state_dict(ckpt["gen"])
    # weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
    # gen.load_state_dict(weight)
    gen.eval()
    auds = np.load(os.path.join('/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/datasets/'+ args.dataset +'/'+args.person_1, 'aud.npy'))
    auds = torch.Tensor(auds).to(device).float()
    # weights_3dmm.eval()
    AudNet.eval()
    AudAttNet.eval()
    ckpt_name = os.path.basename(model_path)
    global_step = int(os.path.splitext(ckpt_name)[0])
    frame_idx = 0
    vid_target_recon = []
    # for real_image, label in loader_test:
    # img_i = 
    get_start = True
    for real_image, label, params, img_i in loader_test:
        real_image = real_image.to(device)
        label = label.to(device)
        params = params.to(device)
        img_i = img_i.to(device)
        if get_start:
            start = int(img_i.squeeze(0))
            get_start = False

        if img_i.squeeze(0) >= len(auds):
            break
        aud = auds[img_i.squeeze(0)]
        with torch.no_grad():
            print(global_step)
            if global_step >= args.nosmo_iters:
                smo_half_win = int(args.smo_size / 2)
                left_i = img_i - smo_half_win
                right_i = img_i + smo_half_win
                pad_left, pad_right = 0, 0
                if left_i < 0:
                    pad_left = -left_i
                    left_i = 0
                if right_i > auds.shape[0]:
                    pad_right = right_i - auds.shape[0]
                    right_i = auds.shape[0]
                auds_win = auds[left_i:right_i]
                if pad_left > 0:
                    auds_win = torch.cat(
                        (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                if pad_right > 0:
                    auds_win = torch.cat(
                        (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                auds_win = AudNet(auds_win)
                aud = auds_win[smo_half_win]
                aud_smo = AudAttNet(auds_win)
                if len(aud_smo.shape) == 1:
                    aud_smo = aud_smo.unsqueeze(0)
                generated_image = gen(aud_smo, label)
            else:
                aud = AudNet(aud.squeeze(1))
                if len(aud.shape) == 1:
                    aud = aud.unsqueeze(0)
                generated_image = gen(aud_smo, label)

            img_recon = generated_image


        # if args.viz_3d2:
        #     r = 2.7
        #     # frame_idx
        #     inter = frame_idx % 120
        #     h = math.pi*(0.5 + 0.1*math.cos(2*math.pi*inter/(0.5 * 240)))
        #     v = math.pi*(0.5 - 0.05*math.sin(2*math.pi*inter/(0.5 * 240)))
        #     camera_points, phi, theta = sample_camera_positions(device=device, n=1, r=r, horizontal_mean=h, vertical_mean=v, mode=None)
        #     c = create_cam2world_matrix(-camera_points, camera_points, device=device)
        #     c = c.reshape(1, -1)
        #     label1 = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).repeat(1, 1).to(c)), -1)

        #     inter2 = frame_idx % 100
        #     h2 = math.pi*(0.5 + 0.4*math.cos(2*math.pi*inter2/(0.5 * 200)))
        #     v2 = math.pi*0.5#(0.5 - 0.05*math.sin(2*math.pi*inter/(0.5 * 240)))
        #     camera_points2, phi2, theta2 = sample_camera_positions(device=device, n=1, r=r, horizontal_mean=h2, vertical_mean=v2, mode=None)
        #     c2 = create_cam2world_matrix(-camera_points2, camera_points2, device=device)
        #     c2 = c2.reshape(1, -1)
        #     label2 = torch.cat((c2, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).repeat(1, 1).to(c2)), -1)


        # with torch.no_grad():
        #     # img_recon = gen(real_image, label)
        #     if args.out_pose :
        #         generated_weights,pose  = gen.get_weights(real_image)
            
        #         latent= gen.get_latent(generated_weights)
        #         pose[:, [1,2,5,6,9,10]] *= -1 #################################
        #         img_recon = gen.get_image(latent, label)
        #     else:
        #         generated_weights = gen.get_weights(real_image)
        #         latent = gen.get_latent(generated_weights)
        #         img_recon = gen.get_image(latent, label)
        #     img_recon_1_1 = gen.get_image(latent, label1)
        #     img_recon_1_2 = gen.get_image(latent, label2)

            # torch.save(latent, f'{embedding_dir}/{frame_idx:05d}.pt')
            torchvision.utils.save_image(img_recon, f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/demo/{args.demo_name}/{frame_idx:05d}.png', normalize=True,
                        range=(-1,1))
            # torchvision.utils.save_image(img_recon_1_1, f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/demo/{args.demo_name}/person1_1_{frame_idx:05d}.png', normalize=True,
            #             range=(-1,1))
            # torchvision.utils.save_image(img_recon_1_2, f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/demo/{args.demo_name}/person1_2_{frame_idx:05d}.png', normalize=True,
            #             range=(-1,1))
            
            # vid_target_recon.append(img_recon.unsqueeze(2))
        frame_idx += 1
        # if frame_idx>=200:
        #     break



    # vid_target_recon = torch.cat(vid_target_recon, dim=2)
    # save_video(vid_target_recon, save_path + 'video.mp4',fps = args.fps)

    if args.cat_video:
        video_out = imageio.get_writer(f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/demo/{args.demo_name}/{args.demo_name}cat.mp4', mode='I', fps=24, codec='libx264', bitrate='12M')
        # length = len(os.listdir(save_path))
        if args.viz_3d2:
            length = int(len(glob.glob(save_path + '*.png'))/3)
        else:
            length = len(glob.glob(save_path + '*.png'))
        
        for i in range(length):
            if args.dataset_type == 'train':
                person1_image_path1 = f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/datasets/{args.dataset}/{args.person_1}/train/cropped_images/{str(i + start)}.jpg'
            else:
                person1_image_path1 = f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/datasets/{args.dataset}/{args.person_1}/test/cropped_images/{str(i + start)}.jpg'
            # person1_image_path1 = f'{args.video_dir1}/{str(i + start)}.jpg'
            person1_image_path2 = f'{save_path}/{i:05d}.png'
            print(person1_image_path1)
            if not os.path.isfile(person1_image_path1):
                continue
            # print(image_path2)
            person1_image1 = Image.open(person1_image_path1).convert('RGB')
            person1_image2 = Image.open(person1_image_path2).convert('RGB')
            person1_image1 = trans(person1_image1).unsqueeze(0)
            person1_image2 = trans(person1_image2).unsqueeze(0)
            # image1 = Image.open(image_path1).convert('RGB')
            # image2 = Image.open(image_path2).convert('RGB')
            # image1 = trans(image1).unsqueeze(0)
            # image2 = trans(image2).unsqueeze(0)
            if args.viz_3d2:
                person1_image_path3 = f'{save_path}/person1_1_{i:05d}.png'
                person1_image_path4 = f'{save_path}/person1_2_{i:05d}.png'
                person1_image3 = Image.open(person1_image_path3).convert('RGB')
                person1_image4 = Image.open(person1_image_path4).convert('RGB')
                person1_image3 = trans(person1_image3).unsqueeze(0)
                person1_image4 = trans(person1_image4).unsqueeze(0)
                cat_frame_1 = torch.cat( (torch.cat((person1_image1, person1_image2),3),torch.cat((person1_image3, person1_image4),3)),2)
            else:
                cat_frame_1 = torch.cat((person1_image1, person1_image2),3)
        # print(image2.shape)
            # cat_frame = torch.cat((image1,image2),3)
        # print(cat_frame.shape)
            video_out.append_data(layout_grid(cat_frame_1, grid_w=1, grid_h=1))
        video_out.close()
    else:
        video_out = imageio.get_writer(f'/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/demo/{args.demo_name}/{args.demo_name}rec.mp4', mode='I', fps=24, codec='libx264', bitrate='12M')
        length = len(glob.glob(save_path + '*.png'))
        for i in range(length):
            # image_path1 = f'{args.video_dir1}/f_{i:04d}.png'
            image_path2 = f'{save_path}/{i:05d}.png'
        # print(image_path1)
            print(image_path2)
            # image1 = Image.open(image_path1).convert('RGB')
            image2 = Image.open(image_path2).convert('RGB')
            # image1 = trans(image1).unsqueeze(0)
            image2 = trans(image2).unsqueeze(0)
        # print(image2.shape)
            # cat_frame = torch.cat((image1,image2),3)
        # print(cat_frame.shape)
            video_out.append_data(layout_grid(image2, grid_w=1, grid_h=1))
        video_out.close()



if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='ad_dataset')

    parser.add_argument("--dataset_type", type=str, default='val')
    parser.add_argument("--suffix", type=str, default='.png')
    parser.add_argument("--ds_path", type=str, default=None)
    
    
    parser.add_argument("--person_1", type=str, default='french')
    parser.add_argument("--run_id", type=str, default='nerface2')
    parser.add_argument("--run_id_2", type=str, default=None)
    parser.add_argument("--out_pose", action='store_true', default=False)
    parser.add_argument("--person_2", type=str, default=None)

    # parser.add_argument("--run_id", type=str, default='nerface2')
    parser.add_argument("--params_len", type=int, default=64)

    parser.add_argument("--emb_dir", type=str, default='/apdcephfs_cq2/share_1290939/kitbai/PTI/embeddings/')
    parser.add_argument("--model_path", type=str, default='/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/exps/nerface2-v1/checkpoint/874999.pt')
    parser.add_argument("--video_dir1", type=str, default="/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/datasets/nerface_dataset/person_1/test/cropped_images/")
    parser.add_argument("--old", action='store_true', default=True)
    parser.add_argument("--tune", action='store_true', default=False)
    parser.add_argument("--init", action='store_true', default=False)
    parser.add_argument("--same_bases", action='store_true', default=False)
    parser.add_argument("--finetune", action='store_true', default=False)

    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    # parser.add_argument("--lr", type=float, default=0.002)
    
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_shape", type=int, default=50)
    # parser.add_argument("--dataset", type=str, default='vox')

    parser.add_argument("--demo_name", type=str, default='nerface2-3d3')
    parser.add_argument("--exp_path", type=str, default='/apdcephfs_cq2/share_1290939/kitbai/LIA-3d/exps/')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')


    parser.add_argument("--dim_aud", type=int, default=64,
                        help='dimension of audio features for NeRF')
    parser.add_argument("--win_size", type=int, default=16,
                        help="windows size of audio feature")
    parser.add_argument('--nosmo_iters', type=int, default=300000,
                        help='number of iterations befor applying smoothing on audio features')
    parser.add_argument("--smo_size", type=int, default=8,
                        help="window size for smoothing audio features")
    parser.add_argument("--tune_iter", type=int, default=50000)
    parser.add_argument("--two_stage", type=int, default=200000)



    parser.add_argument("--viz_3d1", action='store_true', default=False)
    parser.add_argument("--viz_3d2", action='store_true', default=False)
    parser.add_argument("--fix_cam", action='store_true', default=False)
    parser.add_argument("--cat_video", action='store_true', default=False)
    parser.add_argument("--cam_angle", type=int, default=61)
    parser.add_argument("--fps", type=int, default=24)

# torch.save(w_pivot, f'{embedding_dir}/0.pt')
    opts = parser.parse_args()

    main( args=opts)