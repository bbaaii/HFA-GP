import torch
import torch.nn as nn
import torchvision
import os
from PIL import Image

from networks.headnerf import HeadNeRF_final
import torch.nn.functional as F

import os
import torch
from torch.utils import data
from dataset import HeadData_test
import torchvision
import torchvision.transforms as transforms
import math

import argparse
import torch
import imageio
from PIL import Image
import math

import torch.nn.functional as F


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
    phi = torch.acos(points[:, 1:2]/r) 
   

    sin_phi = torch.sqrt(1- (points[:, 1:2]/r)*(points[:, 1:2]/r))
    sin_theta = points[:, 2:3] / (r*sin_phi)
    cos_theta = points[:, 0:1] / (r*sin_phi)

    theta = torch.acos(cos_theta)
    if sin_theta > 1 or sin_theta<-1:
        theta = torch.acos(cos_theta)


    if cos_theta > 1 or cos_theta <-1:
        theta = torch.asin(sin_theta)
    h = theta / math.pi
    v = phi / math.pi
    return h, v    


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

    device = torch.device("cuda")

    transform = torchvision.transforms.Compose([
        transforms.Resize(args.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset_test = HeadData_test(args.dataset_type, transform, dataset = args.dataset , person = args.person, ds_path = args.ds_path, suffix= args.suffix )

    loader_test = data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=1, rank=0, shuffle=False),
        pin_memory=True,
        drop_last=False,
    )
    model_path = args.model_path

    save_path = f'./demo/{args.demo_name}/'

    os.makedirs(save_path, exist_ok=True)

    ckpt = torch.load(model_path)
    gen = HeadNeRF_final(args, args.size, device, args.latent_dim_style, args.latent_dim_shape, args.run_id, args.emb_dir).cuda()

    gen.load_state_dict(ckpt["gen"])

    gen.eval()
    frame_idx = 0
    vid_target_recon = []
    for real_image, label in loader_test:
        real_image = real_image.to(device)
        label = label.to(device)


        with torch.no_grad():
            if args.out_pose :
                generated_weights,pose  = gen.get_weights(real_image)
            
                latent= gen.get_latent(generated_weights)
                pose[:, [1,2,5,6,9,10]] *= -1 #################################
                img_recon = gen.get_image(latent, label)
            else:
                generated_weights = gen.get_weights(real_image)
                latent = gen.get_latent(generated_weights)
                img_recon = gen.get_image(latent, label)

            torchvision.utils.save_image(img_recon, f'./demo/{args.demo_name}/{frame_idx:05d}.png', normalize=True,
                        range=(-1,1))

        frame_idx += 1


    if args.cat_video:
        video_out = imageio.get_writer(f'./demo/{args.demo_name}/{args.demo_name}cat.mp4', mode='I', fps=24, codec='libx264', bitrate='12M')

        length = len(glob.glob(save_path + '*.png'))
        
        for i in range(length):
            person1_image_path1 = f'./datasets/{args.dataset}/{args.person}/test/cropped_images/f_{i:04d}.png'
            person1_image_path2 = f'{save_path}/{i:05d}.png'
            print(person1_image_path1)

            person1_image1 = Image.open(person1_image_path1).convert('RGB')
            person1_image2 = Image.open(person1_image_path2).convert('RGB')
            person1_image1 = trans(person1_image1).unsqueeze(0)
            person1_image2 = trans(person1_image2).unsqueeze(0)

            cat_frame_1 = torch.cat((person1_image1, person1_image2),3)

            video_out.append_data(layout_grid(cat_frame_1, grid_w=1, grid_h=1))
        video_out.close()
    else:
        video_out = imageio.get_writer(f'./demo/{args.demo_name}/rec.mp4', mode='I', fps=24, codec='libx264', bitrate='12M')
        length = len(glob.glob(save_path + '*.png'))
        for i in range(length):

            image_path2 = f'{save_path}/{i:05d}.png'

            print(image_path2)

            image2 = Image.open(image_path2).convert('RGB')

            image2 = trans(image2).unsqueeze(0)

            video_out.append_data(layout_grid(image2, grid_w=1, grid_h=1))
        video_out.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='nerface_dataset')

    parser.add_argument("--dataset_type", type=str, default='test')
    parser.add_argument("--suffix", type=str, default='.png')
    parser.add_argument("--ds_path", type=str, default='./datasets/our_dataset/person_2/test/cam_003/cropped_images')
    parser.add_argument("--person", type=str, default='person_2')
    parser.add_argument("--run_id", type=str, default='nerface2')
    parser.add_argument("--person_2", type=str, default=None)
    # parser.add_argument("--run_id", type=str, default='nerface2')
    parser.add_argument("--run_id_2", type=str, default=None)
    parser.add_argument("--emb_dir", type=str, default='/apdcephfs_cq2/share_1290939/kitbai/PTI/embeddings/')
    parser.add_argument("--model_path", type=str, default='./exps/nerface2-v1/checkpoint/874999.pt')
    parser.add_argument("--video_dir1", type=str, default="./datasets/nerface_dataset/person_1/test/cropped_images/")
    parser.add_argument("--old", action='store_true', default=True)
    parser.add_argument("--tune", action='store_true', default=False)
    parser.add_argument("--init", action='store_true', default=False)
    parser.add_argument("--same_bases", action='store_true', default=False)
    parser.add_argument("--finetune", action='store_true', default=False)
    parser.add_argument("--out_pose", action='store_true', default=False)
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
    parser.add_argument("--exp_path", type=str, default='./exps/')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--viz_3d1", action='store_true', default=False)
    parser.add_argument("--viz_3d2", action='store_true', default=False)
    parser.add_argument("--fix_cam", action='store_true', default=False)
    parser.add_argument("--cat_video", action='store_true', default=False)
    parser.add_argument("--cam_angle", type=int, default=61)
    parser.add_argument("--fps", type=int, default=24)

# torch.save(w_pivot, f'{embedding_dir}/0.pt')
    opts = parser.parse_args()

    main( args=opts)