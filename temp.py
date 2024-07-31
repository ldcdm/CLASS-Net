import os

import torch
from monai.transforms import Compose, EnsureType, AsDiscrete, KeepLargestConnectedComponent
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv3d


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = (self.conv(x) + x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class Encoder_block(nn.Module):
    def __init__(self, stage=1, in_channel=1, out_channel=16, normalization='none', dw=True):
        super(Encoder_block, self).__init__()
        if dw:
            scale = 2
        else:
            scale = 1
        self.block = ConvBlock(stage, in_channel, out_channel // scale, normalization=normalization)
        self.block_dw = DownsamplingConvBlock(out_channel // scale, out_channel, normalization=normalization)
        self.dw = dw

    def forward(self, x):
        x = self.block(x)
        skip = x
        if self.dw:
            x = self.block_dw(x)

        return skip, x


class Decoder_block(nn.Module):
    def __init__(self, stage=1, in_channel=1, out_channel=16, normalization='none', mode='ConvTranspose'):
        super(Decoder_block, self).__init__()

        if mode == 'ConvTranspose':
            self.block_up = UpsamplingDeconvBlock(in_channel, out_channel, normalization=normalization)
        if mode == 'Upsampling':
            self.block_up = Upsampling(in_channel, out_channel, normalization=normalization)

        self.block = ConvBlock(stage, out_channel, out_channel, normalization=normalization)

    def forward(self, x, x2):

        x = self.block_up(x)
        x = x + x2
        skip = x
        x = self.block(x)

        return skip, x


class Deep_up(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(Deep_up, self).__init__()
        self.up = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True), )
    def forward(self, input):
        return self.up(input)

class DeepSupervise_up(nn.Module):
    def __init__(self, n_filters, n_classes):
        super(DeepSupervise_up, self).__init__()

        self.up1 = Deep_up(n_filters * 8, n_classes, 8)
        self.up2 = Deep_up(n_filters * 4, n_classes, 4)
        self.up3 = Deep_up(n_filters * 2, n_classes, 2)

    def forward(self, input):
        [x5_up, x6_up, x7_up, out] = input
        x5_up = self.up1(x5_up)
        x6_up = self.up2(x6_up)
        x7_up = self.up3(x7_up)

        output = [x5_up, x6_up, x7_up, out]
        return output

class Deep_down(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(Deep_down, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, kernel_size=(kernel_size,kernel_size,kernel_size), stride=(stride,stride,stride), padding=padding, groups=channel_in, bias=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=(1,1,1), padding=0, bias=False),
            nn.BatchNorm3d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=(kernel_size,kernel_size,kernel_size), stride=(1,1,1), padding=padding, groups=channel_in, bias=False),
            nn.Conv3d(channel_in, channel_out, kernel_size=(1,1,1), padding=0, bias=False),
            nn.BatchNorm3d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class DeepContrast_down(nn.Module):
    def __init__(self, n_filters):
        super(DeepContrast_down, self).__init__()

        self.down1 = nn.Sequential(
            Deep_down(
                channel_in= n_filters * 2,
                channel_out=n_filters * 4
            ),
            Deep_down(
                channel_in=n_filters * 4,
                channel_out=n_filters * 8
            ),
            Deep_down(
                channel_in=n_filters * 8,
                channel_out=n_filters * 16
            ),
            Deep_down(
                channel_in=n_filters * 16,
                channel_out=n_filters * 32
            ),
            nn.AdaptiveAvgPool3d(1)
        )
        self.down2 = nn.Sequential(
            Deep_down(
                channel_in=n_filters * 4,
                channel_out=n_filters * 8
            ),
            Deep_down(
                channel_in=n_filters * 8,
                channel_out=n_filters * 16
            ),
            Deep_down(
                channel_in=n_filters * 16,
                channel_out=n_filters * 32
            ),
            nn.AdaptiveAvgPool3d(1)
        )
        self.down3 = nn.Sequential(
            Deep_down(
                channel_in=n_filters * 8,
                channel_out=n_filters * 16
            ),
            Deep_down(
                channel_in=n_filters * 16,
                channel_out=n_filters * 32
            ),
            nn.AdaptiveAvgPool3d(1)
        )
        self.down4 = nn.Sequential(
            Deep_down(
                channel_in=n_filters * 16,
                channel_out=n_filters * 32
            ),
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self,x5, input):
        [x5_up, x6_up, x7_up, out] = input

        x5_down = self.down4(x5)
        x6_down = self.down3(x5_up)
        x7_down = self.down2(x6_up)
        x8_down = self.down1(x7_up)

        output = [x5_down, x6_down, x7_down,x8_down, out]
        return output

class Skip_Gate_v3(nn.Module):
    def __init__(self, n_filters_in, classes=5, reduction=2):
        super(Skip_Gate_v3, self).__init__()
        self.Conv1 = nn.Conv3d(classes, n_filters_in // reduction, 3, 1, 1)
        self.Conv2 = nn.Conv3d(n_filters_in, n_filters_in, 3, 1, 1)

        self.alpha = nn.Conv3d(n_filters_in // reduction, n_filters_in, 3, 1, 1)
        self.beta = nn.Conv3d(n_filters_in // reduction, n_filters_in, 3, 1, 1)

        self.Conv3 = nn.Conv3d(n_filters_in, n_filters_in, 3, 1, 1)
        self.Relu = nn.ReLU(True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, seg=None):
        seg = F.interpolate(seg.float(), x.size()[2:], mode='trilinear', align_corners=True)
        res = x
        seg = self.Conv1(seg)
        # x = self.Conv2(x)

        alpha = self.alpha(seg)
        beta = self.beta(seg)
        # print(x.shape,alpha.shape,beta.shape)
        x = self.Relu(x * (1 + alpha) + beta)
        x = self.Conv3(x)
        x = self.Sigmoid(x)
        x = x * res
        return x


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dv=False,
                 dc=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        if dv and not dc:
            self.mode = 1
        elif not dv and dc:
            self.mode = 2
        elif not dv and not dc:
            self.mode = 0
        else:
            raise Exception

        # shared Encoder
        self.e_1 = Encoder_block(1, n_channels, n_filters * 2, normalization)
        self.e_2 = Encoder_block(2, n_filters * 2, n_filters * 4, normalization)
        self.e_3 = Encoder_block(3, n_filters * 4, n_filters * 8, normalization)
        self.e_4 = Encoder_block(3, n_filters * 8, n_filters * 16, normalization)
        self.e_5 = Encoder_block(3, n_filters * 16, n_filters * 16, normalization, False)

        # Decoder1
        self.d1_1 = Decoder_block(3, n_filters * 16, n_filters * 8, normalization, 'ConvTranspose')
        self.d1_2 = Decoder_block(3, n_filters * 8, n_filters * 4, normalization, 'ConvTranspose')
        self.d1_3 = Decoder_block(2, n_filters * 4, n_filters * 2, normalization, 'ConvTranspose')
        self.d1_4 = Decoder_block(1, n_filters * 2, n_filters, normalization, 'ConvTranspose')
        self.out_conv1 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dv_up = DeepSupervise_up(n_filters, n_classes)
        self.dc_down = DeepContrast_down(n_filters)

        self.skip_gate1 = Skip_Gate_v3(n_filters * 2, reduction=2)
        self.skip_gate2 = Skip_Gate_v3(n_filters * 4, reduction=2)
        self.skip_gate3 = Skip_Gate_v3(n_filters * 8, reduction=2)

        self.dropout = nn.Dropout3d(0.5)
        if self.mode !=0:
            self.drop1 = nn.Dropout3d(0.1)
        else:
            self.drop1 = nn.Dropout3d(0.5)
        self.drop2 = nn.Dropout3d(0.2)
        self.drop3 = nn.Dropout3d(0.3)
        self.drop4 = nn.Dropout3d(0.5)

    def encoder(self, input):
        x1, x1_dw = self.e_1(input)
        x2, x2_dw = self.e_2(x1_dw)
        x3, x3_dw = self.e_3(x2_dw)
        x4, x4_dw = self.e_4(x3_dw)
        x5, x5_ = self.e_5(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)
        res = [x1, x2, x3, x4, x5]
        return res

    def decoder(self, features, seg=None):
        # 预测分支1
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_, x5_up = self.d1_1(x5, x4)
        x6_, x6_up = self.d1_2(x5_up, x3)
        x7_, x7_up = self.d1_3(x6_up, x2)
        x8_, x8_up = self.d1_4(x7_up, x1)
        x9 = x8_up
        if self.has_dropout:
            x9 = self.drop1(x9)
            x5_up = self.drop4(x5_up)
            x6_up = self.drop3(x6_up)
            x7_up = self.drop2(x7_up)
        out = self.out_conv1(x9)

        res = [x5_up, x6_up, x7_up, out]

        if self.mode == 1:
            res = self.dv_up(res)
            [x5_up, x6_up, x7_up, out] = res
            res = [x5_up, x6_up, x7_up, out]
        elif self.mode == 2:
            res = self.dc_down(x5,res)
            [x5_down, x6_down, x7_down,x8_down, out] = res
            res = [x5_down, x6_down, x7_down,x8_down, out]

        return res

    def decoder_dual(self, features, seg=None):
        # 预测分支1
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_, x5_up = self.d1_1(x5, self.skip_gate3(x4, seg))
        x6_, x6_up = self.d1_2(x5_up, self.skip_gate2(x3, seg))
        x7_, x7_up = self.d1_3(x6_up, self.skip_gate1(x2, seg))
        x8_, x8_up = self.d1_4(x7_up, x1)
        x9 = x8_up
        if self.has_dropout:
            x9 = self.drop1(x9)
            x5_up = self.drop4(x5_up)
            x6_up = self.drop3(x6_up)
            x7_up = self.drop2(x7_up)
        out = self.out_conv1(x9)

        res = [x5_up, x6_up, x7_up, out]

        if self.mode == 1:
            res = self.dv_up(res)
            [x5_up, x6_up, x7_up, out] = res
            res = [x5_up, x6_up, x7_up, out]
        elif self.mode == 2:
            res = self.dc_down(x5,res)
            [x5_down, x6_down, x7_down,x8_down, out] = res
            res = [x5_down, x6_down, x7_down,x8_down, out]
        return res

    def net_forward(self, image, seg=None, dual=False):
        features = self.encoder(image)
        if dual:
            assert any(seg[seg > 0])
            out = self.decoder_dual(features, seg)
        else:
            out = self.decoder(features)
        return out

    def forward(self, image, turnoff_drop=True, is_train=False, dual=False, seg=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        if is_train and dual:
            out = self.net_forward(image, seg, True)
        elif not is_train and dual:
            out = self.net_forward(image, seg, False)
            seg = out[-1]
            out = self.net_forward(image, seg, True)
            out = out[-1]
        elif is_train and not dual:
            out = self.net_forward(image, seg, False)
        else:
            out = self.net_forward(image, seg, False)
            out = out[-1]

        if turnoff_drop:
            self.has_dropout = has_dropout
        return out
import numpy as np

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(8, 1, 96, 96, 96)
    y = torch.randn(2, 5, 96, 96, 96)
    x = x.to(device)
    y = y.to(device)
    # print("x size: {}".format(x.size()))

    model = VNet(n_channels=1, n_classes=5, dv=False,dc=True).to(device)
    # out1 = model(x, False, True, False, y)
    # # outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4 = out1
    # for i in out1:
    #     print(i.shape)

    # post_output = Compose([
    #     EnsureType(),
    #     AsDiscrete(argmax=True, to_onehot=5),
    #     KeepLargestConnectedComponent(is_onehot=True, applied_labels=[1, 2, 3])
    # ])
    # print(model)
    # print()
    # print(out1.shape)
    # print(post_output(out1).shape)


    unlabeled_volume_batch = x
    noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
    ema_inputs = unlabeled_volume_batch + noise

    with torch.no_grad():
        ema_output = model(ema_inputs,True,False,False)

    T = 4
    volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
    stride = volume_batch_r.shape[0] // 2

    preds = torch.zeros([stride * T, 5, 96,96,96]).cuda()

    for i in range(T // 2):

        ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)

        with torch.no_grad():
            preds[2 * stride * i:2 * stride * (i + 1)] = model(ema_inputs)


    preds = torch.softmax(preds, dim=1)
    preds = preds.reshape(T, stride, 5, 96,96,96)
    preds = torch.mean(preds, dim=0)

    uncertainty = -1.0 * (preds * torch.log(preds + 1e-6))

    consistency_weight = 0.1

    threshold = (0.75 + 0.25 * 1) * np.log(2)
    mask = (uncertainty < threshold).float()
    ema_output = consistency_weight * (mask * ema_output)

    print(ema_output.shape)