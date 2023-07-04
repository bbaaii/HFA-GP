import torch
# from networks.discriminator import Discriminator
# from networks.generator import Generator
from networks.headnerf import HeadNeRF_final
import torch.nn.functional as F
from torch import nn, optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP

from lpips import LPIPS
def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag

l2_criterion = torch.nn.MSELoss(reduction='mean')

from cam_utils import sample_camera_positions, create_cam2world_matrix
import math

import torchvision.transforms as transforms






def cam_sampler(batch, device):
    camera_points, phi, theta = sample_camera_positions(device, n=batch, r=2.7, horizontal_mean=0.5*math.pi, 
                                                        vertical_mean=0.5*math.pi, horizontal_stddev=0.3, vertical_stddev=0.155, mode='gaussian')
    c = create_cam2world_matrix(-camera_points, camera_points, device=device)
    c = c.reshape(batch, -1)
    c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).repeat(batch, 1).to(c)), -1)
    return c


def cam_sampler_pose(batch, horizontal_mean, vertical_mean , device):
    camera_points, phi, theta = sample_camera_positions(device, n=batch, r=2.7, horizontal_mean=horizontal_mean*math.pi, 
                                                        vertical_mean=vertical_mean*math.pi, horizontal_stddev=0.15, vertical_stddev=0.155, mode='gaussian')
    c = create_cam2world_matrix(-camera_points, camera_points, device=device)
    c = c.reshape(batch, -1)
    c = torch.cat((c, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).repeat(batch, 1).to(c)), -1)
    return c



class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.device = device

        self.gen = HeadNeRF_final(args, args.size, device, args.latent_dim_style, args.latent_dim_shape, args.run_id, args.emb_dir).to(
            device)
        self.gen = DDP(self.gen, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True)

        self.g_optim = torch.optim.Adam(self.gen.parameters(), lr= args.lr)
        for param in self.gen.module.generator.parameters():
            param.requires_grad = False

        self.lpips_loss = LPIPS(net='alex').to(device).eval()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((args.size, args.size))

    def l2_loss(self, real_images, generated_images):
        loss = l2_criterion(real_images, generated_images)
        return loss

    def tune_generator(self):
        for param in self.gen.module.generator.parameters():
            param.requires_grad = True

    def gen_update(self, real_image, label, person_2 = False, mask = None):
        self.gen.train()
        self.g_optim.zero_grad()

        weights_i = self.gen.module.get_weights(real_image)
        latent_i = self.gen.module.get_latent(weights_i, person_2)
        generated_image = self.gen.module.get_image(latent_i, label)




        generated_image =  self.face_pool(generated_image)
        l2_loss =  self.l2_loss(real_image, generated_image)
        loss_lpips = self.lpips_loss( real_image, generated_image)
        loss_lpips = torch.squeeze(loss_lpips).mean()



        g_loss = (l2_loss + loss_lpips )
        
        g_loss.backward()

        
        self.g_optim.step()

        return l2_loss, loss_lpips, generated_image 

    def sample(self, real_image, label, person_2 = False ):
        with torch.no_grad():
            self.gen.eval()
            img_recon = self.gen(real_image, label, person_2)


        return img_recon

    def sample_bases(self, person_2 = False ):
        img_recons = []
        with torch.no_grad():
            r = 2.7
            points, _, _ = sample_camera_positions(device=self.device, n=1, r=r, horizontal_mean=0.5*math.pi, vertical_mean=0.5*math.pi, mode=None)
            label = create_cam2world_matrix(-points, points, device=self.device)
            label = label.reshape(1, -1)
            label = torch.cat((label, torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).reshape(1, -1).repeat(1, 1).to(label)), -1)
            self.gen.eval()

            for base_id in range(self.args.latent_dim_shape):
                weights = torch.zeros(self.args.latent_dim_shape).to(self.device)
                weights[base_id] = 10
                weights = weights.unsqueeze(0)
            
                latent = self.gen.module.get_latent(weights,person_2)
                img_recon = self.gen.module.get_image(latent, label)
                img_recons.append(img_recon)

        return img_recons


    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt)
        ckpt_name = os.path.basename(resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])

        self.gen.module.load_state_dict(ckpt["gen"])

        self.g_optim.load_state_dict(ckpt["g_optim"])


        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
