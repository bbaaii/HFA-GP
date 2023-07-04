import torch
# from networks.discriminator import Discriminator
# from networks.generator import Generator
from networks.headnerf import HeadNeRF_Audio, AudioNet, AudioAttNet 
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

class Trainer(nn.Module):
    def __init__(self, auds, i_train, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size

        self.gen = HeadNeRF_Audio(args, args.size, device, args.latent_dim_style, args.latent_dim_shape, args.run_id, args.emb_dir).to(
            device)
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        self.AudNet = AudioNet(args.dim_aud, args.win_size).to(device)
        self.AudAttNet = AudioAttNet().to(device)
        self.AudNet = DDP(self.AudNet, device_ids=[rank], find_unused_parameters=True)
        self.AudAttNet = DDP(self.AudAttNet, device_ids=[rank], find_unused_parameters=True)
        self.optimizer_Aud = torch.optim.Adam(
                                params=list(self.AudNet.parameters()), lr=args.lr, betas=(0.9, 0.999))
        self.optimizer_AudAtt = torch.optim.Adam(
                                params=list(self.AudAttNet.parameters()), lr=args.lr, betas=(0.9, 0.999))


        self.w_optim = torch.optim.Adam(self.gen.parameters(), lr= args.lr)
        for param in self.gen.module.generator.parameters():
            param.requires_grad = False
        self.lpips_loss = LPIPS(net='alex').to(device).eval()
        self.device = device
        self.auds = torch.Tensor(auds).to(device).float()
        self.i_train = i_train
        self.face_pool = torch.nn.AdaptiveAvgPool2d((args.size, args.size))


    def l2_loss(self, real_images, generated_images):
        loss = l2_criterion(real_images, generated_images)
        return loss
    def tune_generator(self):
        for param in self.gen.module.generator.parameters():
            param.requires_grad = True
    def gen_update(self, real_image, label, params, global_step, img_i, person_2 = False):
        self.gen.train()
        # self.weights_3dmm.train()
        self.AudNet.train()
        self.AudAttNet.train()
        self.w_optim.zero_grad()
        self.optimizer_Aud.zero_grad()
        self.optimizer_AudAtt.zero_grad()

        aud = self.auds[img_i]
        if global_step >= self.args.nosmo_iters:
            smo_half_win = int(self.args.smo_size / 2)
            left_i = img_i - smo_half_win
            right_i = img_i + smo_half_win
            pad_left, pad_right = 0, 0
            if left_i < 0:
                pad_left = -left_i
                left_i = 0
            if right_i > self.i_train:
                pad_right = right_i-self.i_train
                right_i = self.i_train
            auds_win = self.auds[left_i:right_i]
            if pad_left > 0:
                auds_win = torch.cat(
                    (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
            if pad_right > 0:
                auds_win = torch.cat(
                    (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
            auds_win = self.AudNet(auds_win)
            aud = auds_win[smo_half_win]
            aud_smo = self.AudAttNet(auds_win)
            if len(aud_smo.shape) == 1:
                aud_smo = aud_smo.unsqueeze(0)
            generated_image = self.gen(aud_smo, label, person_2)
        else:
            aud = self.AudNet(aud.squeeze(1))
            if len(aud.shape) == 1:
                aud = aud.unsqueeze(0)
            generated_image = self.gen(aud, label, person_2)

        generated_image =  self.face_pool(generated_image)
        l2_loss_3dmm = torch.zeros(1).to(self.device) #self.l2_loss(weights, generated_weights)
        l2_loss =  self.l2_loss(real_image, generated_image)
        loss_lpips = self.lpips_loss( real_image, generated_image)
        loss_lpips = torch.squeeze(loss_lpips).mean()
        
        g_loss = l2_loss_3dmm + l2_loss + loss_lpips
        
        g_loss.backward()

        
        self.w_optim.step()
        self.optimizer_Aud.step()
        if global_step >= self.args.nosmo_iters:
            self.optimizer_AudAtt.step()

        return l2_loss_3dmm, l2_loss, loss_lpips, generated_image 

    def sample(self, real_image, label, params, global_step, img_i, person_2 = False):
        with torch.no_grad():
            self.gen.eval()
            self.AudNet.eval()
            self.AudAttNet.eval()
            aud = self.auds[img_i]
            if global_step >= self.args.nosmo_iters:
                smo_half_win = int(self.args.smo_size / 2)
                left_i = img_i - smo_half_win
                right_i = img_i + smo_half_win
                pad_left, pad_right = 0, 0
                if left_i < 0:
                    pad_left = -left_i
                    left_i = 0
                if right_i > self.auds.shape[0]:
                    pad_right = right_i - self.auds.shape[0]
                    right_i = self.auds.shape[0]
                auds_win = self.auds[left_i:right_i]
                if pad_left > 0:
                    auds_win = torch.cat(
                        (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                if pad_right > 0:
                    auds_win = torch.cat(
                        (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                auds_win = self.AudNet(auds_win)
                aud = auds_win[smo_half_win]
                aud_smo = self.AudAttNet(auds_win)
                if len(aud_smo.shape) == 1:
                    aud_smo = aud_smo.unsqueeze(0)
                img_recon = self.gen(aud_smo, label, person_2)
            else:
                aud = self.AudNet(aud.squeeze(1))
                if len(aud.shape) == 1:
                    aud = aud.unsqueeze(0)

                img_recon = self.gen(aud, label, person_2)


        return img_recon#
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
                weights[base_id] = 5
                weights = weights.unsqueeze(0)
            

                latent = self.gen.module.get_latent(weights,person_2)
                img_recon = self.gen.module.get_image(latent, label)
                img_recons.append(img_recon)

        return img_recons#

    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt)
        ckpt_name = os.path.basename(resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])

        self.gen.module.load_state_dict(ckpt["gen"])
        self.AudNet.module.load_state_dict(ckpt["AudNet"])
        self.AudAttNet.module.load_state_dict(ckpt["AudAttNet"])
        self.w_optim.load_state_dict(ckpt["w_optim"])

        self.optimizer_Aud.load_state_dict(ckpt["optimizer_Aud"])
        self.optimizer_AudAtt.load_state_dict(ckpt["optimizer_AudAtt"])

        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "AudAttNet": self.AudAttNet.module.state_dict(),
                "AudNet": self.AudNet.module.state_dict(),
                "w_optim": self.w_optim.state_dict(),
                "optimizer_Aud": self.optimizer_Aud.state_dict(),
                "optimizer_AudAtt": self.optimizer_AudAtt.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
