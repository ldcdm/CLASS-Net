import os

import torch
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, EnsureType, AsDiscrete, KeepLargestConnectedComponent, RandSpatialCropSamplesd
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        skip=x
        x = self.conv(x)
        x=x+skip
        x = self.relu(x)
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
        skip=x
        x = self.block(x)

        x = x + skip
        skip=x
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
        x = self.block(x) + skip
        skip=x

        return skip, x


def soft_dilate(img):
    return F.max_pool3d(img, (5, 5, 5), (1, 1, 1), (2, 2, 2))


def soft_erode(img):
    p1 = -F.max_pool3d(-img, (5, 1, 1), (1, 1, 1), (2, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 5, 1), (1, 1, 1), (0, 2, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 5), (1, 1, 1), (0, 0, 2))

    return torch.min(torch.min(p1, p2), p3)


class CSAM(nn.Module):
    def __init__(self, n_filters,n_classes=4):
        super(CSAM, self).__init__()
        self.n_classes = n_classes
        self.sigmoid = nn.Sigmoid()
        self.attn = cont_attn()
        self.norm = nn.BatchNorm3d(n_filters)
        # self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
    def forward(self, feat,x):
        # x=self.out_conv(x)
        b, c, d, w, h = x.shape
        # out=torch.zeros_like(feat)
        out = feat
        for i in range(0, c):
            out = out + self.attn(feat[:, :, ...], self.sigmoid(x[:, i:i + 1, ...]))

        return self.norm(out)


class cont_attn(nn.Module):
    def __init__(self):
        super(cont_attn, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.k = nn.Parameter(torch.ones(1))

    def forward(self, feat_area, fore_mask):

        dilate_mask = fore_mask
        erode_mask = fore_mask
        N, C, D, W, H = feat_area.size()
        iters = 1

        for i in range(iters):
            dilate_mask = soft_dilate(fore_mask)
        for i in range(iters):
            erode_mask = soft_erode(fore_mask)

        back_mask = 1 - dilate_mask

        erode_feat = erode_mask.contiguous().view(N, 1, -1)  # N,1,DHW
        erode_feat = erode_feat.permute(0, 2, 1).contiguous()  # N,DHW,1

        dilate_feat = dilate_mask.contiguous().view(N, 1, -1)  # N,1,DHW
        dilate_feat = dilate_feat.permute(0, 2, 1).contiguous()  # N,DHW,1

        back_feat = back_mask.contiguous().view(N, 1, -1)  # N,1,DHW
        back_feat = back_feat.permute(0, 2, 1).contiguous()  # N,DHW,1
        feat = feat_area.contiguous().view(N, C, -1)  # N,C,DHW

        erode_num = torch.sum(erode_feat, dim=1, keepdim=True) + 1e-5
        dilate_num = torch.sum(dilate_feat, dim=1, keepdim=True) + 1e-5
        back_num = torch.sum(back_feat, dim=1, keepdim=True) + 1e-5

        erode_cluster = torch.bmm(feat, erode_feat) / erode_num  # N,C,1
        dilate_cluster = torch.bmm(feat, dilate_feat) / dilate_num  # N,C,1
        back_cluster = torch.bmm(feat, back_feat) / back_num  # N,C,1
        fore_cluster = erode_cluster + dilate_cluster
        feat_cluster = torch.cat((fore_cluster, back_cluster), dim=-1)  # N,C,2

        feat_key = feat_area  # N,C,H,W,S
        feat_key = feat_key.contiguous().view(N, C, -1)  # N,C,DHW
        feat_key = feat_key.permute(0, 2, 1).contiguous()  # N,DHW,C

        feat_cluster = feat_cluster.permute(0, 2, 1).contiguous()  # N,2,C
        feat_query = feat_cluster  # N,2,C
        feat_value = feat_cluster  # N,2,C

        feat_query = feat_query.permute(0, 2, 1).contiguous()  # N,C,2
        feat_sim = torch.bmm(feat_key, feat_query)  # N,DHW,2
        feat_sim = self.softmax(feat_sim)

        feat_atten = torch.bmm(feat_sim, feat_value)  # N,DHW,C
        feat_atten = feat_atten.permute(0, 2, 1).contiguous()  # N,C,DHW
        feat_atten = feat_atten.view(N, C, D, W, H)
        feat_area = self.k * feat_atten + feat_area

        return feat_area


class Dsv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(Dsv, self).__init__()
        self.sideout = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)

    def forward(self, input):
        skip = self.sideout(input)
        out = self.up(skip)
        return skip,out

class Deep_down(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
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

        self.down1 = Deep_down(
                channel_in=n_filters * 4,
                channel_out=n_filters * 8)

        self.down2 = Deep_down(
                channel_in=n_filters * 8,
                channel_out=n_filters * 16
            )

        # self.down3 = Deep_down(
        #         channel_in=n_filters * 16,
        #         channel_out=n_filters * 32
        #     )

        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self,x5, input):
        [x5_up, x6_up, x7_up, out] = input

        x7_down = self.down1(x6_up)

        x6_down = self.down2(x5_up+x7_down)

        # x5_down = self.down3(x5+x6_down)

        output = self.pool(x5+x6_down)
        return output


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dc=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        self.dc = dc

        # shared Encoder
        self.e_1 = Encoder_block(1, n_channels, n_filters * 2, normalization)
        self.e_2 = Encoder_block(2, n_filters * 2, n_filters * 4, normalization)
        self.e_3 = Encoder_block(3, n_filters * 4, n_filters * 8, normalization)
        self.e_4 = Encoder_block(3, n_filters * 8, n_filters * 16, normalization)
        self.e_5 = Encoder_block(3, n_filters * 16, n_filters * 16, normalization, False)

        self.d1_1 = Decoder_block(3, n_filters * 16, n_filters * 8, normalization, 'ConvTranspose')
        self.d1_2 = Decoder_block(3, n_filters * 8, n_filters * 4, normalization, 'ConvTranspose')
        self.d1_3 = Decoder_block(2, n_filters * 4, n_filters * 2, normalization, 'ConvTranspose')
        self.d1_4 = Decoder_block(1, n_filters * 2, n_filters, normalization, 'ConvTranspose')
        self.out_conv1 = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dc_down = DeepContrast_down(n_filters)
        self.CSAM =CSAM(n_filters * 2,n_classes)
        self.Dsv=Dsv(n_filters * 2,n_classes,2)

        self.dropout = nn.Dropout3d(0.5)
        if self.dc:
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


    def decoder(self, features):
        # 预测分支1
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_, x5_up = self.d1_1(x5, x4)
        x6_, x6_up = self.d1_2(x5_up, x3)
        x7_, x7_up = self.d1_3(x6_up, x2)
        skip,side_out = self.Dsv(x7_up)
        x7_up = self.CSAM(x7_up,skip)
        x8_, x8_up = self.d1_4(x7_up, x1)
        x9 = x8_up
        if self.has_dropout:
            x9 = self.drop1(x9)
            x5_up = self.drop4(x5_up)
            x6_up = self.drop3(x6_up)
            x7_up = self.drop2(x7_up)
        out = self.out_conv1(x9)

        res = [x5_up, x6_up, x7_up, out]



        if self.dc:
            x5_dowm = self.dc_down(x5,res)
            res = [x5_dowm, x5_dowm, side_out, out]

        return res


    def forward(self, input, turnoff_drop=False, is_train=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        if is_train:
            self.dc=True
        else:
            self.dc=False
        features = self.encoder(input)
        out = self.decoder(features)

        if turnoff_drop:
            self.has_dropout = has_dropout
        if is_train:
            self.dc=True
            return out
        else:
            self.dc=True
            return out[-1]



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    devicess = [0]

    model = VNet(n_channels=1, n_classes=4,normalization='batchnorm',has_dropout=True, dc=True).to(device)
    model = torch.nn.DataParallel(model, device_ids=devicess)
    x = torch.Tensor(4, 1, 80, 144, 144)
    x.to(device)
    out = model(x,False,True)
    for i in out:
        print((i.size()))

