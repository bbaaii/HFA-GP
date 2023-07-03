from torch import nn
from .encoder3d import Encoder, EqualLinear, Encoder_whole

import os 

import dnnlib
import legacy
import torch
import pickle
import copy
# import torch_utils
def load_bases(device, base_dir, dim_shape):

    bases= torch.randn(dim_shape, 18,512)

    latent_dirs = os.listdir(base_dir)
    for i in range(dim_shape):
        w_potential_path = base_dir + latent_dirs[i] + '/0.pt'

        base = torch.load(w_potential_path).squeeze(0)#.to(device)
        bases[i] = base.detach()

    return bases.to(device)
        
def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



def load_G_official(args, device, eg3d_ffhq = './pretrained_models/eg3d/ffhqrebalanced512-128.pkl'):
    with dnnlib.util.open_url(eg3d_ffhq) as f:
    # with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device) 

        old_G = copy.deepcopy(old_G).requires_grad_(False).to(device)

    return old_G

        



class HeadNeRF_final(nn.Module):
    def __init__(self,args ,size, device, dim=512, dim_shape=20, run_id = 'nerface2',emb_dir = './PTI/embeddings/',use_softmax = False):
        super(HeadNeRF_final, self).__init__()
        self.base_dir = emb_dir + run_id
        self.device = device
        self.encoder = Encoder(size, dim, dim_shape, use_softmax, args.out_pose)
        self.args = args
        self.out_pose = args.out_pose
        self.dim  = dim
        self.dim_shape = dim_shape

        bases = torch.randn(self.dim_shape, 14*self.dim).to(device)
        self.bases = torch.nn.Parameter(bases, requires_grad=True)
        self.delta = torch.nn.Parameter(bases.mean(dim = 0), requires_grad=True)

        # print(self.bases)
        if args.person_2 :
            
            base_dir_2 =  emb_dir + args.run_id_2 + '/PTI/'
            if args.init:
                bases_2 = load_bases(self.device, base_dir_2, dim_shape).view(self.dim_shape,-1)
            else:
                bases_2 = torch.randn(self.dim_shape, 14*self.dim).to(device)
            if not args.same_bases:
                self.bases_2 = torch.nn.Parameter(bases_2, requires_grad=True)
            self.delta_2 = torch.nn.Parameter(bases_2.mean(dim = 0), requires_grad=True)
            # print(self.bases_2)
        

        self.generator = load_G_official(args, self.device )

        
    def get_delta(self, person_2 = False):
        if not person_2:
            return self.delta.view(-1, self.dim)
        else:
            return self.delta_2.view(-1, self.dim)
    def get_latent(self, weights, person_2 = False):
        b = weights.shape[0]

        if not person_2:
            bases = self.bases + 1e-8
        else:
            if not self.args.same_bases:
                bases = self.bases_2 + 1e-8
            else:
                bases = self.bases + 1e-8
        Q, R = torch.qr(bases.T)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if weights is None:
            return Q
        else:
            input_diag = torch.diag_embed(weights)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            if not person_2:
                return out.view(b, -1, self.dim) + self.delta.view(-1, self.dim)
            else:
                return out.view(b, -1, self.dim) + self.delta_2.view(-1, self.dim)



    def forward(self, image, label, person_2 = False):

        label[:, [1,2,5,6,9,10]] *= -1
        if self.out_pose:
            weights, pose = self.encoder(image)
            latent = self.get_latent(weights, person_2)#.unsqueeze(0)
            img_recon = self.generator.synthesis(latent, c=label, noise_mode='const')['image']
            return img_recon, pose
        else:
            weights = self.encoder(image)
            latent = self.get_latent(weights, person_2)#.unsqueeze(0)

            img_recon = self.generator.synthesis(latent, c=label, noise_mode='const')['image']

            return img_recon

        
    def get_weights(self, image):
        if self.out_pose:
            weights, pose = self.encoder(image)
            return weights, pose
        else:
            weights = self.encoder(image)
            return weights
        # return 
    def get_image(self, latent, label):
        label[:, [1,2,5,6,9,10]] *= -1
        img_recon = self.generator.synthesis(latent, c=label, noise_mode='const')['image']
        return img_recon 



class Weights_3DMM(nn.Module):
    def __init__(self, input_dim=76, dim=512, dim_shape=50, use_softmax = False):
        super(Weights_3DMM, self).__init__()
        fc = [EqualLinear(input_dim, dim)]

        for i in range(5):
            fc.append(EqualLinear(dim, dim))

        fc.append(EqualLinear(dim, dim_shape))
        self.fc = nn.Sequential(*fc)
        self.softmax = nn.Softmax(dim=1)
        self.use_softmax = use_softmax


    def forward(self, input):

        weights = self.fc(input)
        if self.use_softmax:
            weights = self.softmax(weights)
        # weights = self.softmax(weights)
        return weights



class HeadNeRF_3DMM(nn.Module):
    def __init__(self, args ,size, device, dim=512, dim_shape=20, run_id = 'nerface2',emb_dir = './PTI/embeddings/',use_softmax = False):
        super(HeadNeRF_3DMM, self).__init__()
        self.base_dir = emb_dir + run_id
        self.device = device
        self.weights_3dmm = Weights_3DMM(input_dim = args.params_len, dim =dim, dim_shape = dim_shape, use_softmax = use_softmax)
        

        self.dim  = dim
        self.dim_shape = dim_shape
        bases = torch.randn(self.dim_shape, 14*self.dim).to(device)
        self.bases = torch.nn.Parameter(bases, requires_grad=True)
        self.delta = torch.nn.Parameter(bases.mean(dim = 0), requires_grad=True)



        self.generator = load_G_official(args, self.device )

        

    def get_latent(self, weights, person_2 = False):
        b = weights.shape[0]

        bases = self.bases + 1e-8

        Q, R = torch.qr(bases.T)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if weights is None:
            return Q
        else:
            input_diag = torch.diag_embed(weights)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out.view(b, -1, self.dim) + self.delta.view(-1, self.dim)




    def forward(self, params, label, person_2 = False):
        label[:, [1,2,5,6,9,10]] *= -1
        weights = self.weights_3dmm(params)


        latent = self.get_latent(weights, person_2)#.unsqueeze(0)

        img_recon = self.generator.synthesis(latent, c=label, noise_mode='const')['image']

        return img_recon
    def get_weights(self, params):
        weights = self.weights_3dmm(params)
        return weights

    def get_image(self, latent, label):
        label[:, [1,2,5,6,9,10]] *= -1


        img_recon = self.generator.synthesis(latent, c=label, noise_mode='const')['image']
        return img_recon 


class HeadNeRF_Audio(nn.Module):
    def __init__(self, args ,size, device, dim=512, dim_shape=20, run_id = 'nerface2',emb_dir = './PTI/embeddings/',use_softmax = False):
        super(HeadNeRF_Audio, self).__init__()
        self.base_dir = emb_dir + run_id 
        self.device = device
        self.weights_3dmm = Weights_3DMM(input_dim = args.params_len, dim =dim, dim_shape = dim_shape, use_softmax = use_softmax)
        

        self.dim  = dim
        self.dim_shape = dim_shape
        bases = torch.randn(self.dim_shape, 14*self.dim).to(device)
        self.bases = torch.nn.Parameter(bases, requires_grad=True)
        self.delta = torch.nn.Parameter(bases.mean(dim = 0), requires_grad=True)

        

        self.generator = load_G_official(args, self.device )

        

    def get_latent(self, weights, person_2 = False):
        b = weights.shape[0]

        bases = self.bases + 1e-8
        Q, R = torch.qr(bases.T)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if weights is None:
            return Q
        else:
            input_diag = torch.diag_embed(weights)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out.view(b, -1, self.dim) + self.delta.view(-1, self.dim)




    def forward(self, params, label, person_2 = False):
        label[:, [1,2,5,6,9,10]] *= -1

        weights = self.weights_3dmm(params)

        latent = self.get_latent(weights, person_2)#.unsqueeze(0)

        img_recon = self.generator.synthesis(latent, c=label, noise_mode='const')['image']


        return img_recon
    def get_weights(self, params):
        weights = self.weights_3dmm(params)
        return weights

    def get_image(self, latent, label):
        label[:, [1,2,5,6,9,10]] *= -1
        img_recon = self.generator.synthesis(latent, c=label, noise_mode='const')['image']

        return img_recon 



# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=32, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len,
                      out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = x[..., :self.dim_aud].permute(1, 0).unsqueeze(
            0)  # 2 x subspace_dim x seq_len
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(self.seq_len, 1)

        return torch.sum(y*x, dim=0)
# Model


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_aud=76, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(29, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, 8-half_w:8+half_w, :].permute(0, 2, 1)
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).squeeze()
        return x

