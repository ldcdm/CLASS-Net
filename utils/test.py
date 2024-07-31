from collections import OrderedDict

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
from torch.nn import init
from torch.nn.init import trunc_normal_
import numpy as np

from utils.Attention import SimplifiedScaledDotProductAttention


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Shift_conv(nn.Module):
    def __init__(self, in_features=16,out_features=32,kernel=3,stride=1,pad=1):
        super(Shift_conv, self).__init__()

        self.shift = Shift_six(in_features,shift_size=5)
        self.attn = SimplifiedScaledDotProductAttention(in_features,h=12)
        self.channel_proj = channel_proj( in_features*2,in_features)
        self.conv_branch = nn.Sequential( nn.Conv3d(in_features, in_features, kernel, stride, pad),
                                          nn.BatchNorm3d(in_features),
                                          nn.ReLU(inplace=True),
                                         )
        self.out_conv = nn.Sequential( nn.Conv3d(in_features, out_features, kernel, stride, pad),
                                          nn.BatchNorm3d(out_features),
                                          nn.ReLU(inplace=True),
                                         )
        self.pixel_dot = pixel_dot(in_features)
    def forward(self, x):
        # B C D H W
        h = x
        B, C, D, H, W = x.shape

        x_s_all = self.shift(x)

        out1 = self.channel_proj(torch.cat((x,x_s_all),dim=1))
        # print(h.shape)
        out2 = self.conv_branch(h)
        out3 = torch.matmul(self.pixel_dot(x,x_s_all),h)

        out = out1 + out2
        out = self.out_conv(out)
        return out


class Basic_conv(nn.Module):
    def __init__(self, in_features=32, out_features=64, kernel=3, stride=1, pad=1):
        super(Basic_conv, self).__init__()

        self.conv_branch = nn.Sequential(nn.Conv3d(in_features, in_features, kernel, stride, pad),
                                         nn.BatchNorm3d(in_features),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(in_features, out_features, kernel, stride, pad),
                                         nn.BatchNorm3d(out_features),
                                         nn.ReLU(inplace=True),
                                         )
    def forward(self, x):
        # B C D H W
        h = x
        B, C, D, H, W = x.shape

        out2 = self.conv_branch(h)

        out = out2

        return out


class channel_proj(nn.Module):

    def __init__(self, in_features, out_features,dropout=0):
        super(channel_proj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Conv3d(in_features,out_features,1,1,0)

    def forward(self, x):
        # input: B N D H W

        x = self.linear(x)

        return x



class pixel_dot(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.soft_max = nn.Softmax(dim=1)
        self.in_features = in_features
    def forward(self,x1,x2):

        assert x1.shape == x2.shape
        out = torch.matmul(x1,x2) / np.sqrt(self.in_features)
        out = self.soft_max(out)
        return out

class Shift(nn.Module): #channel&token mixing
    def __init__(self, in_features, shift_size=2):
        super().__init__()
        self.dim = in_features

        self.shift_group = shift_size
        self.pad = shift_size // 2

    def forward(self, x):

        # in:  B, C, D, H, W
        B, C, D, H, W = x.shape

        # xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad,self.pad, self.pad), "constant", 0)
        # xs = torch.chunk(xn, self.shift_group, 1)
        # x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        # x_cat = torch.cat(x_shift, 1)
        # x_cat = torch.narrow(x_cat, 2, self.pad, D)
        # x_cat = torch.narrow(x_cat, 3, self.pad, H)
        # x_s_d = torch.narrow(x_cat, 4, self.pad, W)
        #
        # xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        # xs = torch.chunk(xn, self.shift_group, 1)
        # x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        # x_cat = torch.cat(x_shift, 1)
        # x_cat = torch.narrow(x_cat, 2, self.pad, D)
        # x_cat = torch.narrow(x_cat, 3, self.pad, H)
        # x_s_h = torch.narrow(x_cat, 4, self.pad, W)
        #
        # xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        # xs = torch.chunk(xn, self.shift_group, 1)
        # x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        # x_cat = torch.cat(x_shift, 1)
        # x_cat = torch.narrow(x_cat, 2, self.pad, D)
        # x_cat = torch.narrow(x_cat, 3, self.pad, H)
        # x_s_w = torch.narrow(x_cat, 4, self.pad, W)

        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_group, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(x_shift, range(-self.pad, self.pad + 1))]
        x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(x_shift, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s_all = torch.narrow(x_cat, 4, self.pad, W)

        return x_s_all


class Shift_six(nn.Module): #channel&token mixing
    def __init__(self, in_features,shift_size=2):
        super().__init__()
        self.dim = in_features

        self.shift_group = 6
        self.pad = shift_size // 2

    def forward(self, x):

        # in:  B, C, D, H, W
        B, C, D, H, W = x.shape


        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_group, 1)


        x_shift = [torch.roll(xs[0],-self.pad,2),
                   torch.roll(xs[1],self.pad+1,2),
                   torch.roll(xs[2], -self.pad, 3),
                   torch.roll(xs[3], self.pad + 1, 3),
                   torch.roll(xs[4], -self.pad, 4),
                   torch.roll(xs[5], self.pad + 1, 4),
                   ]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s_all = torch.narrow(x_cat, 4, self.pad, W)

        return x_s_all


class shift_u(nn.Module):
    def __init__(self, img_shape=(128,256,256),in_channels=1, out_channels=6,dims=[16,32,64,128]):


        super(shift_u, self).__init__()

        self.img_shape = img_shape
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.encoder1 = Basic_conv(in_channels,dims[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = Shift_conv(dims[0], dims[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = Shift_conv(dims[1], dims[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = Shift_conv(dims[2], dims[3])
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = Basic_conv(dims[3],dims[3]*2)


        self.upconv4 = nn.ConvTranspose3d(
            dims[-1] * 2, dims[-1], kernel_size=2, stride=2
        )
        self.decoder4 = Basic_conv(dims[-1] * 2, dims[-1])

        self.upconv3 = nn.ConvTranspose3d(
            dims[-1], dims[-2], kernel_size=2, stride=2
        )
        self.decoder3 = Basic_conv(dims[-2] * 2, dims[-2])

        self.upconv2 = nn.ConvTranspose3d(
            dims[-2], dims[-3], kernel_size=2, stride=2
        )
        self.decoder2 = Basic_conv(dims[-3] * 2, dims[-3])

        self.upconv1 = nn.ConvTranspose3d(
            dims[-3] , dims[-4], kernel_size=2, stride=2
        )
        self.decoder1 = Basic_conv(dims[-4]*2, dims[-4])

        self.conv = nn.Conv3d(
            in_channels=dims[-4], out_channels=out_channels, kernel_size=1
        )




    def forward(self, x):
        # print("===========encoder start============")
        # print(x.shape)
        t1 = x
        t1 = self.encoder1(t1)

        # print('t1',t1.shape)

        t2 = self.encoder2(self.pool1(t1))
        # print('t2',t2.shape)

        t3 = self.encoder3(self.pool2(t2))
        # print('t3',t3.shape)

        t4 = self.encoder4(self.pool3(t3))
        # print('t4',t4.shape)

        btn = self.pool4(t4)
        btn = self.bottleneck(btn)

        de4 = self.upconv4(btn)

        # print(de4.shape,t4.shape)

        de4 = self.decoder4(torch.cat((de4,t4),dim=1))

        de3 = self.upconv3(de4)
        de3 = self.decoder3(torch.cat((de3,t3),dim=1))

        de2 = self.upconv2(de3)
        de2 = self.decoder2(torch.cat((de2,t2),dim=1))

        de1 = self.upconv1(de2)
        de1 = self.decoder1(torch.cat((de1,t1),dim=1))

        out = self.conv(de1)


        # print("===========decoder end============")
        return out




if __name__ == '__main__':

    sm = shift_u(in_channels=1,out_channels=6).cuda()
    a = torch.Tensor(2,1,128,256,256).cuda()
    H = 16
    W = 16
    D = 16
    out = sm(a)

    print(out.shape)