
import numpy as np
import torch
import torchio
from scipy import ndimage
from scipy.ndimage import distance_transform_edt as distance, morphology
from skimage import segmentation as skimage_seg
from skimage import filters
from skimage.morphology import disk, erosion, ball, dilation


def compute_sdf_soft(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)
    batch_size, channel, x, y, z = img_gt.shape
    if channel>out_shape[1]:
        stride = channel - out_shape[1]
    else:
        stride = 0
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c+stride].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                # print(sdf.shape)
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def compute_dtm01(img_gt, out_shape):

    normalized_dtm = np.zeros(out_shape)
    batch_size, channel, x, y, z = img_gt.shape
    if channel>out_shape[1]:
        stride = channel - out_shape[1]
    else:
        stride = 0
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c+stride].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis/np.max(posdis)

    return normalized_dtm

def compute_boundary(img_gt, out_shape):

    img_gt = img_gt.astype(np.uint8)
    boundary = np.zeros(out_shape)
    batch_size, channel, x, y, z = img_gt.shape
    if channel>out_shape[1]:
        stride = channel - out_shape[1]
    else:
        stride = 0
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c+stride].astype(np.bool)
            if posmask.any():

                boundary_ = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                boundary[b][c] = boundary_

    return boundary

def compute_border2(img_gt, out_shape):

    img_gt = img_gt.astype(np.uint8)
    boundary = np.zeros(out_shape)
    batch_size, channel, x, y, z = img_gt.shape
    if channel>out_shape[1]:
        stride = channel - out_shape[1]
    else:
        stride = 0
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c+stride]
            if posmask.any():
                kernel = ball(2)
                ori = posmask
                edges = erosion(posmask, kernel, )
                edges = ori - edges
                # boundary_ = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                boundary[b][c] = edges

    return boundary

def compute_erosion(img_gt):

    outout = np.zeros(img_gt)
    batch_size,c,  x, y, z = img_gt.shape
    for b in range(img_gt.shape[0]): # batch size
        for c in range(img_gt.shape[1]):
            posmask = img_gt[b][c]
            kernel = ball(1)
            edges = erosion(posmask, kernel,)
            edges = dilation(edges, kernel, )
            # boundary_ = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            outout[b][c] = edges

    return outout

def compute_sdf_robust_nonorm(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation

    """

    img_gt = img_gt.astype(np.uint8)
    sdf_out = np.zeros(out_shape)

    batch_size, channel, x, y, z = img_gt.shape
    if channel>out_shape[1]:
        stride = channel - out_shape[1]
    elif channel==out_shape[1]:
        stride = 0
    else:
        raise ValueError

    for b in range(out_shape[0]): # batch size

        for c in range(out_shape[1]):

            posmask = img_gt[b][c+stride].astype(np.bool)

            if posmask.any():
                # print(np.unique(img_gt[b][c + stride]))
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                sdf_out[b][c] = sdf

    return sdf_out
def compute_sdf_robust(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    batch_size, channel, x, y, z = img_gt.shape
    if channel>out_shape[1]:
        stride = channel - out_shape[1]
    elif channel==out_shape[1]:
        stride = 0
    else:
        raise ValueError

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            print(np.unique(img_gt[b][c+stride]))
            posmask = img_gt[b][c+stride].astype(np.bool)
            kernel = ball(1)
            posmask = erosion(posmask, kernel, )
            posmask = dilation(posmask, kernel, )
            if posmask.any():

                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
    return normalized_sdf


def compute_sdf_pre(img_gt, out_shape):

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    channel, x, y, z = img_gt.shape


    if channel>out_shape[0]:
        stride = channel - out_shape[0]
    elif channel==out_shape[0]:
        stride = 0
    else:
        raise ValueError
    # print(img_gt.shape, out_shape,stride)
     # batch size
    for c in range(out_shape[0]):

        posmask = img_gt[c+stride].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[c] = sdf
            assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
    return normalized_sdf

import cv2 as cv
if __name__ == '__main__':
    img_name = "/home/dluser/ZDY/bone/pytorch_medical/Pytorch-Medical-Segmentation/logs/0/image.nii.gz"
    # label_name = "/home/dluser/ZDY/bone/pytorch_medical/Pytorch-Medical-Segmentation/logs/Knee_mymodel_my/0/0_seg299.nii.gz"
    label_name = "/home/dluser/ZDY/bone/pytorch_medical/Pytorch-Medical-Segmentation/logs/0/label.nii.gz"


    img_patch = (96,96,96)
    img_size = (160,384,384)
    data_dicts = [{"img": img_name,"label": label_name}]
    from monai.transforms import LoadImaged, EnsureTyped, AsDiscreted, Compose, ToTensord, EnsureChannelFirstd, \
    Orientationd, SaveImage, CopyItemsd, Lambdad, RandSpatialCropSamplesd, RandSpatialCrop, RandSpatialCropd

    transform = Compose([
        LoadImaged(keys=['img','label']),
        EnsureChannelFirstd(keys=['img',"label"]),
        Orientationd(keys=['img',"label"], axcodes="RAS"),
        EnsureTyped(keys=['img','label']),

        AsDiscreted(keys=['label'],to_onehot=5),

        RandSpatialCropd(keys=["img","label"],roi_size=img_patch,random_size=False),
        CopyItemsd(keys=["label"], times=1, names=["gt_sdf"], allow_missing_keys=False),

        Lambdad(keys='gt_sdf', func=lambda x: compute_sdf_pre(x.numpy(),(5,*img_patch))),
        # RandSpatialCropSamplesd(keys=["img","label","gt_sdf"], roi_size=img_patch, random_size=False,
        #                         num_samples=1),
    ])

    data = transform(data_dicts)
    label = data[0]['label']
    img = data[0]['img']
    sdf = data[0]['gt_sdf']
    print(sdf.shape)
    data = label.unsqueeze(0)
    img = img.unsqueeze(1)

    # out_ = compute_sdf_soft(data.numpy(),(1, 5 , 96, 96, 96))
    # out_1 = compute_border2(data.numpy(), (1, 5, 96, 96, 96))
    # out_2 = compute_boundary(data.numpy(), (1, 5, 96, 96, 96))

    # out_1 = compute_erosion(data.numpy(), (1, 5, 96, 96, 96))
    # print(np.unique(data.numpy()))
    # out_ = compute_sdf_robust_nonorm(data.numpy(), (1, 5, 96, 96, 96))
    # # out_1 = data.numpy()
    #
    output_dir = 'logs/0'
    #
    saver = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="dis", print_log=True)
    # saver1 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="sobel", print_log=True)
    # saver2 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="border", print_log=True)
    # saver3 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="ori", print_log=True)
    # saver4 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="erosion", print_log=True)
    #
    # img0 = out_[:, 0, ...]
    # img1 = out_[:, 1, ...]
    # img2 = out_[:, 2, ...]
    # img3 = out_[:, 3, ...]
    # img4 = out_[:, 4, ...]
    # img5 = np.argmax(out_,axis=1)
    #
    # saver4(img0)
    # saver4(img1)
    # saver4(img2)
    # saver4(img3)
    # saver4(img4)
    # saver4(img5)


    img0 = sdf[0:1, ...]
    img1 = sdf[1:2, ...]
    img2 = sdf[2:3, ...]
    img3 = sdf[3:4, ...]
    img4 = sdf[4:5, ...]


    saver(img0)
    saver(img1)
    saver(img2)
    saver(img3)
    saver(img4)
