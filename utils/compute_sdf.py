import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from skimage.morphology import ball, erosion, dilation


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
    elif channel==out_shape[1]:
        stride = 0
    else:
        raise ValueError
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c+stride].astype(np.bool)
            print(np.unique(img_gt[b][c+stride]))
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                # print(sdf.shape)
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                print(np.min(sdf),np.max(sdf))
                assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

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

            posmask = img_gt[b][c+stride].astype(np.bool)
            kernel = ball(1)

            posmask = erosion(posmask, kernel)
            posmask = dilation(posmask, kernel)
            if posmask.any():
                # print(np.unique(img_gt[b][c + stride]))
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

def compute_dtm_soft(img_gt, out_shape):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """

    img_gt = img_gt.astype(np.uint8)
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
                normalized_dtm[b][c] = posdis / np.max(posdis)

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

def compute_erosion(img_gt, out_shape):

    outout = np.zeros(out_shape)
    batch_size,c,  x, y, z = img_gt.shape
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b][c]
            kernel = ball(1)
            edges = erosion(posmask, kernel,)
            edges = dilation(edges, kernel, )
            # boundary_ = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            outout[b][c] = edges

    return outout