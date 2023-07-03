import math
import torch
from torch import nn
from torch.nn import functional as F


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
        return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0), ]

    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1, )

    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        return F.leaky_relu(input, negative_slope=self.negative_slope)


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):

        return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride,
                                  bias=bias and not activate))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class EncoderApp(nn.Module):
    def __init__(self, size, w_dim=512):
        super(EncoderApp, self).__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }

        self.w_dim = w_dim
        log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channels[size], 1))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs.append(EqualConv2d(in_channel, self.w_dim, 4, padding=0, bias=False))

    def forward(self, x):

        # res = []
        h = x
        for conv in self.convs:
            h = conv(h)
            # res.append(h)

        return h.squeeze(-1).squeeze(-1) #, res[::-1][2:]


class Encoder(nn.Module):
    def __init__(self, size, dim=512, dim_motion=20, use_softmax = False, out_pose = False):
        super(Encoder, self).__init__()

        # appearance netmork
        self.net_app = EncoderApp(size, dim)

        # motion network
        fc = [EqualLinear(dim, dim)]
        for i in range(3):
            fc.append(EqualLinear(dim, dim))

        fc.append(EqualLinear(dim, dim_motion))
        self.fc = nn.Sequential(*fc)
        self.out_pose = out_pose
        if out_pose:
            fc2 = [EqualLinear(dim, dim)]
            for i in range(3):
                fc2.append(EqualLinear(dim, dim))

            fc2.append(EqualLinear(dim, 25))
            self.pose = nn.Sequential(*fc2)
        self.fc = nn.Sequential(*fc)
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=1)

    def enc_app(self, x):

        h_source = self.net_app(x)

        return h_source

    # def enc_motion(self, x):

    #     h, _ = self.net_app(x)
    #     h_motion = self.fc(h)

    #     return h_motion
    def get_weights(self, x):

        h = self.net_app(x)
        h_weights = self.fc(h)
        if self.use_softmax:
            h_weights = self.softmax(h_weights)
        if self.out_pose:
            pose = self.pose(h)
            return h_weights, pose

        return h_weights

    def forward(self, input_source, h_start=None):
        if self.out_pose:
            weights, pose = self.get_weights(input_source)
            return weights, pose
        else:
            weights = self.get_weights(input_source)
            return weights
        # print(weights)
        # if self.use_softmax:
        #     weights = self.softmax(weights)
        # # print(weights)
        # return weights
        # if input_target is not None:

        #     h_source, feats = self.net_app(input_source)
        #     h_target, _ = self.net_app(input_target)

        #     h_motion_target = self.fc(h_target)

        #     if h_start is not None:
        #         h_motion_source = self.fc(h_source)
        #         h_motion = [h_motion_target, h_motion_source, h_start]
        #     else:
        #         h_motion = [h_motion_target]

        #     return h_source, h_motion, feats
        # else:
        #     h_source, feats = self.net_app(input_source)

        #     return h_source, None, feats
    #


class Encoder_whole(nn.Module):
    def __init__(self, size, dim=512, dim_motion=20, latent_warp =32, use_softmax = False, out_pose = False):
        super(Encoder_whole, self).__init__()

        # appearance netmork
        self.net_app = EncoderApp(size, dim)

        # motion network
        fc = [EqualLinear(dim, dim)]
        for i in range(3):
            fc.append(EqualLinear(dim, dim))

        fc.append(EqualLinear(dim, dim_motion))
        self.fc = nn.Sequential(*fc)

        self.out_pose = out_pose

        # fc_warp = [EqualLinear(dim + 25, dim)]

        # for i in range(3):
        #     fc_warp.append(EqualLinear(dim, dim))

        # fc_warp.append(EqualLinear(dim, latent_warp))
        # self.fc_warp = nn.Sequential(*fc_warp)

        
        if out_pose:
            fc2 = [EqualLinear(dim, dim)]
            for i in range(3):
                fc2.append(EqualLinear(dim, dim))

            fc2.append(EqualLinear(dim, 25))
            self.pose = nn.Sequential(*fc2)
        # self.fc = nn.Sequential(*fc)
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=1)

    def enc_app(self, x):

        h_source = self.net_app(x)
        weights = self.fc(h_source)

        return h_source, weights

    # def enc_motion(self, x):

    #     h, _ = self.net_app(x)
    #     h_motion = self.fc(h)

    #     return h_motion

    # def get_latent(self,h, pose):
    #     latent = self.fc_warp(torch.cat((h,pose),-1))
    #     return latent

    def get_weights(self, x):

        h = self.net_app(x)
        h_weights = self.fc(h)
        # h_latent = self.fc_warp(h)
        if self.out_pose:
            pose = self.pose(h)
            return h_weights, h, pose

        return h_weights, h

    def forward(self, input_source, pose_input=None):
        h_source, weights =  self.enc_app(input_source)
        # weights = self.fc(h)
        if self.out_pose and  pose_input is None:
            # pose = self.pose(h)
            
            pose = self.pose(h_source)
            # latent = self.get_latent(h_source, pose) #self.fc_warp(torch.cat((h,pose),-1))

            # weights, pose = self.get_weights(input_source)
            return weights, h_source, pose
        else:
            # latent = self.get_latent(h_source, pose_input)
            # pose = self.pose(h)
            # latent = self.fc_warp(h)
            
            # weights, latent = self.get_weights(input_source)
            return weights, h_source



class pose2latent(nn.Module):
    def __init__(self, latent_warp =32, len_pose = 25, dim=512):
        super(pose2latent, self).__init__()

        fc_warp = [EqualLinear(len_pose, dim)]

        for i in range(3):
            fc_warp.append(EqualLinear(dim, dim))

        fc_warp.append(EqualLinear(dim, latent_warp))
        self.fc_warp = nn.Sequential(*fc_warp)

    def forward(self, pose):
        latent = self.fc_warp(pose)
        return latent

class pose2latent_o(nn.Module):
    def __init__(self, latent_warp =32, dim=512):
        super(pose2latent_o, self).__init__()

        fc_warp = [EqualLinear(dim + 25, dim)]

        for i in range(3):
            fc_warp.append(EqualLinear(dim, dim))

        fc_warp.append(EqualLinear(dim, latent_warp))
        self.fc_warp = nn.Sequential(*fc_warp)

    def forward(self, h, pose):
        latent = self.fc_warp(torch.cat((h,pose),-1))
        return latent

# model = Encoder_whole(512, dim=512, dim_motion=20, latent_warp =32, use_softmax = False, out_pose = True)
# # print(model)
# input = torch.randn(1,3,512,512)
# pose = torch.randn(1, 25)
# weights, latent,pose = model(input)
# print(weights.shape)
# print(latent.shape)
# print(pose.shape)


# model = pose2latent(latent_warp = 32)
# # print(model)
# input = torch.randn(1,512)
# pose = torch.randn(1, 25)
# latent = model(input, pose)
# # print(weights.shape)
# print(latent.shape)
# print(pose.shape)
