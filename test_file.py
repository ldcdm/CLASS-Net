import math
import os

import torch
from monai.losses import ContrastiveLoss
from monai.transforms import Compose, EnsureType, AsDiscrete, KeepLargestConnectedComponent
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.conv import Conv3d
from tqdm import tqdm

from utils.dataloader_backup import MedData_test
from utils.dataloader_monai import MedData_train


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # print("num_attention_heads", self.num_attention_heads) # 12
        # print("attention_head_size", self.attention_head_size) # 24
        # print("all_head_size", self.all_head_size) # 288
        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        #print('x.size()[:-1]',x.size()[:-1]) # x.size()[:-1] torch.Size([2, 1440])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q,K,V):
        mixed_query_layer = self.query(Q)
        mixed_key_layer = self.key(K)
        mixed_value_layer = self.value(V)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # print("value_layer", value_layer.shape) # value_layer torch.Size([2, 12, 1440, 48])
        # print("key_layer", key_layer.shape) # key_layer torch.Size([2, 12, 1440, 48])

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        #print("attention_probs",attention_probs.shape) # attention_probs torch.Size([2, 12, 1440, 1440])
        context_layer = torch.matmul(attention_probs, value_layer)
        #print("context_layer1", context_layer.shape) # context_layer1 torch.Size([2, 12, 1440, 48])
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #print("context_layer2", context_layer.shape) # context_layer2 torch.Size([2, 1440, 12, 48])
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        #print("new_context_layer_shape", new_context_layer_shape) # new_context_layer_shape torch.Size([2, 1440, 576])
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights



class tensor2patch(nn.Module):
    def __init__(self, n_filters_in, n_filters_out,img_size = 96,patch_size=16,dropout=0.1):
        super(tensor2patch, self).__init__()
        self.patch = nn.Conv3d(n_filters_in,n_filters_out,patch_size,patch_size,0)
        self.n_patches = int((img_size/patch_size)**3)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_filters_out, self.n_patches))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        x = self.patch(x)
        x = x.view(x.size()[0],x.size()[1],-1)

        embeddings = x + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings

class patch2tensor(nn.Module):
    def __init__(self, n_filters_in, n_filters_out,img_size = 96,patch_size=16,to_size = 12,dropout=0.1):
        super(patch2tensor, self).__init__()
        self.n = img_size//patch_size
        t = img_size // to_size
        k = patch_size // t
        s = patch_size // t

        self.patch = nn.ConvTranspose3d(n_filters_in,n_filters_out,k,s,0)

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.view(x.size()[0],x.size()[1],self.n,self.n,self.n)
        x = self.patch(x)
        embeddings = self.dropout(x)
        return embeddings

class NormBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none',dual=False):
        super(NormBlock, self).__init__()
        self.dual = dual
        ops = []
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        self.l1 =nn.Linear(1,16)
        self.l2 = nn.Linear(16, n_filters_out)
        self.l3 = nn.Linear(16, n_filters_out)
        self.norm = nn.BatchNorm3d(n_filters_out)

    def forward(self, x,seg=None):

        if self.dual:

            seg = F.interpolate(seg.float(),x.size()[2:],mode='trilinear',align_corners=True)
            seg = seg.permute(0,2,3,4,1)

            l1 = self.l1(seg)
            l2 = self.l2(l1)
            l2 = l2.permute(0, 4, 1, 2, 3)

            l3 = self.l3(l1)
            l3 = l3.permute(0, 4, 1, 2, 3)
            x = self.norm(x)
            x = x * (1+l2) + l3
        else:
            x = self.norm(x)
        return x

class NormBlock_v2(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none',dual=False):
        super(NormBlock_v2, self).__init__()
        self.dual = dual
        ops = []
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        self.l1 =nn.Conv3d(5,16,3,1,1)
        self.l2 = nn.Conv3d(16,n_filters_out,3,1,1)
        self.l3 = nn.Conv3d(16,n_filters_out,3,1,1)
        self.norm = nn.BatchNorm3d(n_filters_out)

    def forward(self, x,seg=None):

        if self.dual:

            seg = F.interpolate(seg.float(),x.size()[2:],mode='trilinear',align_corners=True)


            l1 = self.l1(seg)
            l2 = self.l2(l1)
            l3 = self.l3(l1)

            x = self.norm(x)
            x = x * (1+l2) + l3
        else:
            x = self.norm(x)
        return x

class Skip_Gate(nn.Module):
    def __init__(self, n_filters_in, reduction,dual_skip=False):
        super(Skip_Gate, self).__init__()
        self.dual_skip = dual_skip
        self.Conv1 = nn.Conv3d(n_filters_in,n_filters_in//reduction,3,1,1)
        self.Conv2 = nn.Conv3d(n_filters_in,n_filters_in//reduction,3,1,1)
        self.Conv3 = nn.Conv3d(n_filters_in//reduction,n_filters_in,3,1,1)
        self.Relu = nn.ReLU(True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x,seg=None,weight=1):


        if self.dual_skip:
            if seg is None:
                return x
            else:
                seg = F.interpolate(seg.float(),x.size()[2:],mode='trilinear',align_corners=True)
                res = x
                seg = self.Conv1(seg)
                x = self.Conv2(x)
                x = self.Relu(x + seg)
                x = self.Conv3(x)
                x = self.Sigmoid(x)
                x = x * res
                return x

        else:
            return x

class Skip_Gate_v2(nn.Module):
    def __init__(self, n_filters_in, reduction,dual_skip=False):
        super(Skip_Gate_v2, self).__init__()
        self.dual_skip = dual_skip
        self.Conv1 = nn.Conv3d(n_filters_in,n_filters_in//reduction,3,1,1)
        self.Conv2 = nn.Conv3d(n_filters_in,n_filters_in//reduction,3,1,1)

        self.alpha = nn.Conv3d(n_filters_in//reduction,n_filters_in//reduction,3,1,1)
        self.beta = nn.Conv3d(n_filters_in//reduction,n_filters_in//reduction,3,1,1)

        self.Conv3 = nn.Conv3d(n_filters_in//reduction,n_filters_in,3,1,1)
        self.Relu = nn.ReLU(True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x,seg=None,weight=1):


        if self.dual_skip:
            if seg is None:
                return x
            else:
                seg = F.interpolate(seg.float(),x.size()[2:],mode='trilinear',align_corners=True)
                res = x
                seg = self.Conv1(seg)
                x = self.Conv2(x)

                alpha = self.alpha(seg)
                beta = self.beta(seg)

                x = self.Relu(x * (1+alpha) + beta)
                x = self.Conv3(x)
                x = self.Sigmoid(x)
                x = x * res
                return x

        else:
            return x

class Skip_Gate_v3(nn.Module):
    def __init__(self, n_filters_in, reduction,dual_skip=False):
        super(Skip_Gate_v3, self).__init__()
        self.dual_skip = dual_skip
        self.Conv1 = nn.Conv3d(n_filters_in,n_filters_in//reduction,3,1,1)
        self.Conv2 = nn.Conv3d(n_filters_in,n_filters_in,3,1,1)

        self.alpha = nn.Conv3d(n_filters_in//reduction,n_filters_in,3,1,1)
        self.beta = nn.Conv3d(n_filters_in//reduction,n_filters_in,3,1,1)

        self.Conv3 = nn.Conv3d(n_filters_in,n_filters_in,3,1,1)
        self.Relu = nn.ReLU(True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x,seg=None,weight=1):


        if self.dual_skip:
            if seg is None:
                return x
            else:
                seg = F.interpolate(seg.float(),x.size()[2:],mode='trilinear',align_corners=True)
                res = x
                seg = self.Conv1(seg)
                # x = self.Conv2(x)

                alpha = self.alpha(seg)
                beta = self.beta(seg)

                x = self.Relu(x * (1+alpha) + beta)
                x = self.Conv3(x)
                x = self.Sigmoid(x)
                x = x * res
                return x

        else:
            return x

class Data_fetcher():
    def __init__(self,loader1,loader2,mode=1,loader_nums=2):
        self.loader1 = loader1 #labeled
        self.loader2 = loader2 #unlabeled
        self.mode = mode
        if loader_nums == 1:
            self.loader_nums = 1
            self.iter1 = iter(loader1)
            self.length = len(loader1)+len(loader2)

        elif loader_nums == 2:
            self.loader_nums = 2
            self.iter1 = iter(loader1)
            self.iter2 = iter(loader2)
            if mode == 1:  # 全监督使用
                self.length = len(loader1)+len(loader2)
                # self.length = len(loader1) + len((loader2))
            elif mode == 0:  # 半监督使用
                self.length = len(loader1)+len(loader2)

    def fetch(self):

        if self.loader_nums==2:
            if self.mode == 1:
                try:
                    batch1 = next(self.iter1)
                except StopIteration:
                    try:
                        batch1 = next(self.iter2)
                    except StopIteration:
                        self.iter1 = iter(self.loader1)
                        self.iter2 = iter(self.loader2)
                        batch1 = next(self.iter1)
                try:
                    batch2 = next(self.iter1)
                except StopIteration:
                    try:
                        batch2 = next(self.iter2)
                    except StopIteration:
                        self.iter1 = iter(self.loader1)
                        self.iter2 = iter(self.loader2)
                        batch2 = next(self.iter1)

                return (batch1,batch2)

            elif self.mode == 0:
                try:
                    la_batch = next(self.iter1)
                except StopIteration:
                    self.iter1 = iter(self.loader1)
                    la_batch = next(self.iter1)
                try:
                    un_batch = next(self.iter2)
                except StopIteration:
                    self.iter2  = iter(self.loader2)
                    un_batch = next(self.iter2)
                return (la_batch,un_batch)
        elif self.loader_nums==1:
            if self.mode == 1:
                try:
                    batch1 = next(self.iter1)
                except StopIteration:
                    self.iter1 = iter(self.loader1)
                    batch1 = next(self.iter1)
                try:
                    batch2 = next(self.iter1)
                except StopIteration:
                    self.iter1 = iter(self.loader1)
                    batch2 = next(self.iter1)

                return (batch1,batch2)
            else:
                raise ValueError

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        features = F.normalize(features,dim=2)
        # print(features.shape)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # print(mask.shape)

        contrast_count = features.shape[1]  #   2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(torch.unbind(features, dim=1)[0].shape)
        #   256 x 512
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   #   256 x   512
            anchor_count = contrast_count   #   2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print(anchor_feature.shape,contrast_feature.shape)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # print (anchor_dot_contrast)  #256 x 256

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print(logits)
        # print(anchor_dot_contrast)
        # print(logits_max)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(mask)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print(mask)
        # print(logits_mask)
        mask = mask * logits_mask
        # print(mask)

        # compute log_prob

        exp_logits = torch.exp(logits) * logits_mask
        # print(torch.exp(logits))
        # print(logits_mask)
        # print(exp_logits)
        # print('exp_logits',torch.log(exp_logits.sum(1, keepdim=True)))

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # print(torch.log(exp_logits.sum(1, keepdim=True)))


        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class My_ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

def soft_dilate(img):
    return F.max_pool3d(img,(5,5,5),(1,1,1),(2,2,2))

def soft_erode(img):
    p1 = -F.max_pool3d(-img,(5,1,1),(1,1,1),(2,0,0))
    p2 = -F.max_pool3d(-img,(1,5,1),(1,1,1),(0,2,0))
    p3 = -F.max_pool3d(-img,(1,1,5),(1,1,1),(0,0,2))

    return torch.min(torch.min(p1,p2),p3)

class clus_atten(nn.Module):
    def __init__(self,device):
        super(clus_atten,self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.k = nn.Parameter(torch.ones(1)).to(device)

    def forward(self, feat_area, fore_mask):

        dilate_mask = fore_mask
        erode_mask = fore_mask
        N,C,H,W,S = feat_area.size()
        iters = 1

        for i in range(iters):
            dilate_mask = soft_dilate(fore_mask)
        for i in range(iters):
            erode_mask = soft_erode(fore_mask)

        fore_mask = erode_mask #N,1,H,W,S cand_mask *
        back_mask = 1 - dilate_mask #N,1,H,W,S
        #feat_area = feat_area * cand_mask #N,C,H,W,S

        fore_feat = fore_mask.contiguous().view(N,1,-1) #N,1,HWS
        fore_feat = fore_feat.permute(0,2,1).contiguous() #N,HWS,1
        back_feat = back_mask.contiguous().view(N,1,-1) #N,1,HWS
        back_feat = back_feat.permute(0,2,1).contiguous() #N,HWS,1
        feat = feat_area.contiguous().view(N,C,-1) #N,C,HWS

        fore_num = torch.sum(fore_feat,dim=1,keepdim=True) + 1e-5
        back_num = torch.sum(back_feat,dim=1,keepdim=True) + 1e-5

        fore_cluster = torch.bmm(feat,fore_feat) / fore_num #N,C,1
        back_cluster = torch.bmm(feat,back_feat) / back_num #N,C,1
        feat_cluster = torch.cat((fore_cluster,back_cluster),dim=-1) #N,C,2

        feat_key = feat_area #N,C,H,W,S
        feat_key = feat_key.contiguous().view(N,C,-1) #N,C,HWS
        feat_key = feat_key.permute(0,2,1).contiguous() #N,HWS,C

        feat_cluster = feat_cluster.permute(0,2,1).contiguous() #N,2,C
        feat_query = feat_cluster #N,2,C
        feat_value = feat_cluster #N,2,C

        feat_query = feat_query.permute(0,2,1).contiguous() #N,C,2
        feat_sim = torch.bmm(feat_key,feat_query) #N,HWS,2
        feat_sim = self.softmax(feat_sim)

        feat_atten = torch.bmm(feat_sim,feat_value) #N,HWS,C
        feat_atten = feat_atten.permute(0,2,1).contiguous() #N,C,HWS
        feat_atten = feat_atten.view(N,C,H,W,S)
        feat_area = self.k * feat_atten + feat_area

        return feat_area

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(4, 5, 96, 96, 96).to(device)


    y = torch.randn(4, 128, 12, 12, 12).to(device)
    z = torch.randn(4, 5, 96, 96, 96).to(device)
    # contra_criterion = SupConLoss()

    bsz = z.size(0)
    # print(bsz)

    x = torch.randn([4, 512,1,1,1]).to(device)
    z = torch.rand([4, 512,1,1,1]).to(device)
    z = x + x
    temp = torch.cat([x,z],dim=0)

    f1, f2 = torch.split(temp, [bsz, bsz], dim=0)

    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    # print(features.shape)
    # print(contra_criterion(features))

    # con = ContrastiveLoss(temperature=0.5, batch_size=4)
    con2 = My_ContrastiveLoss(temperature=0.5, batch_size=4)
    con3 = SupConLoss(0.5,base_temperature=0.5)
    # print(con(x,z))
    print(con2(x.squeeze(),z.squeeze()))
    print(con3(features))
    # x = torch.randn(4, 1, 96, 96, 96).to(device)
    # y = torch.randn(4, 1, 96, 96, 96).to(device)
    # res = clus_atten(device)
    # print(res(x,y).shape)

    # print("x size: {}".format(x.size()))

    # model = NormBlock_v2(1, 16, 16, normalization='batchnorm',dual=True)
    # x = x.permute(0,2,3,4,1).reshape(-1,16)
    # y = y.permute(0,2,3,4,1).reshape(-1,16)
    # print(x.shape)
    # model = SelfAttention(8,16,0)

    # model = Skip_Gate_v3(16,4,True)
    # out = model(x,y)
    # print(out.shape)

    # patch = tensor2patch(5, 512 ,img_size = 96,patch_size=16,dropout=0.1)
    #
    # patch2 = tensor2patch(128, 512 ,img_size = 12,patch_size=2,dropout=0.1)
    #
    # tensor = patch2tensor(512, 128 ,img_size = 12,patch_size=2,to_size=12,dropout=0.1)
    #
    # attn = SelfAttention(8, 512, 0.1)
    # px = patch(x)
    # py = patch2(y)
    #
    # px = px.permute(0,2,1)
    # py = py.permute(0,2,1)
    # px,weights = attn(py,px,py)
    # px = px.permute(0,2,1)
    # tx = tensor(px)
    #
    # print((tx.shape))


    # from hparam1 import hparams as hp
    # unlabeled_image_dir = hp.unlabeled_image_dir
    # unlabeled_label_dir = hp.unlabeled_label_dir
    # labeled_image_dir = hp.labeled_image_dir
    # labeled_label_dir = hp.labeled_label_dir
    # test_image_dir = hp.test_image_dir
    # test_label_dir = hp.test_label_dir
    # val_image_dir = hp.val_image_dir
    # val_label_dir = hp.val_label_dir
    # batch_size = 4
    # labeled_bs = 2
    #
    # un_util = MedData_train(unlabeled_image_dir, unlabeled_label_dir, True, batch_size=batch_size - labeled_bs)
    # un_loader = un_util.train_loader
    #
    # la_util = MedData_train(labeled_image_dir, labeled_label_dir, False, batch_size=labeled_bs)
    # la_loader = la_util.train_loader
    #
    #
    # # fetcher = Data_fetcher(la_loader,un_loader,0,2)
    # fetcher1 = Data_fetcher(la_loader,un_loader,0,2)
    # fetcher2 = Data_fetcher(la_loader,un_loader,0,1)
    # fetcher3 = Data_fetcher(la_loader,un_loader,1,2)
    # fetcher4 = Data_fetcher(la_loader,un_loader,1,1)
    #
    # print(fetcher1.length)
    # print(fetcher2.length)
    # print(fetcher3.length)
    # print(fetcher4.length)

    # pbar = tqdm(range(8), dynamic_ncols=True)

    # for i in pbar:
    #     batch1, batch2 = fetcher.fetch()
    #
    #     input1, label1 = (
    #         batch1["image"].to(device),
    #         batch1["label"].to(device),
    #     )
    #     # print()
    #     input2, label2 = (
    #         batch2["image"].to(device),
    #         batch2["label"].to(device),
    #     )
    #     print(torch.unique(label1),torch.unique(label2))
        # input = torch.cat([input1, input2], dim=0)
        # label = torch.cat([label1, label2], dim=0)
        #
        # x = input.type(torch.FloatTensor).cuda()
        # y = label.type(torch.FloatTensor).cuda()

