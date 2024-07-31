import monai
import numpy as np
import torch
import torchio
from monai.data import Dataset, DataLoader, decollate_batch
from monai.handlers import from_engine
from scipy import ndimage, fft
from scipy.fft import dct, idct, dst, idst
from scipy.ndimage import distance_transform_edt as distance, morphology, sobel
from skimage import segmentation as skimage_seg
from skimage import filters
from skimage.feature import canny
from skimage.morphology import disk, erosion, ball
from skimage.transform import radon,iradon

def compute_dct(img_gt):

    output = np.zeros(img_gt.shape)
    batch_size, channel, x, y, z = img_gt.shape

    for b in range(batch_size): # batch size
        # for c in range(channel):
        #     temp = img_gt[b][c+stride]
        #     temp = dct(temp)
        #     output[b][c] = temp

        temp = img_gt[b]
        temp = dct(temp,axis=1,type=1)
        output[b] = temp
    return output

def compute_idct(img_gt):

    output = np.zeros(img_gt.shape)
    batch_size, channel, x, y, z = img_gt.shape

    for b in range(batch_size): # batch size
        # for c in range(channel):
        #     temp = img_gt[b][c+stride]
        #     temp = idct(temp)
        #     output[b][c] = temp

        temp = img_gt[b]
        temp = idct(temp,axis=1,type=1)
        output[b] = temp
    return output
def compute_dst(img_gt):

    output = np.zeros(img_gt.shape)
    batch_size, channel, x, y, z = img_gt.shape

    for b in range(batch_size): # batch size
        # for c in range(channel):
        #     temp = img_gt[b][c+stride]
        #     temp = dst(temp)
        #     output[b][c] = temp

        temp = img_gt[b]
        temp = dst(temp)
        output[b] = temp
    return output

def compute_idst(img_gt, out_shape):

    output = np.zeros(img_gt.shape)
    batch_size, channel, x, y, z = img_gt.shape

    for b in range(batch_size): # batch size
        # for c in range(channel):
        #     temp = img_gt[b][c+stride]
        #     temp = idst(temp)
        #     output[b][c] = temp

        temp = img_gt[b]
        temp = idst(temp)
        output[b] = temp
    return output



def compute_sobel(img_gt):

    output = np.zeros(img_gt.shape)
    batch_size, channel, x, y, z = img_gt.shape
    for b in range(batch_size): # batch size
        for c in range(channel):
            temp = img_gt[b][c]
            temp = sobel(temp)
            output[b][c] = temp
    return output
def compute_distance(img_gt):

    output = np.zeros(img_gt.shape)
    batch_size, channel, x, y, z = img_gt.shape
    for b in range(batch_size): # batch size
        for c in range(channel):
            temp = img_gt[b][c]
            temp = distance(temp)
            output[b][c] = temp
    return output

def compute_tanh(img_gt, out_shape):

    output = np.zeros(out_shape)
    batch_size, channel, x, y, z = img_gt.shape
    if channel>out_shape[1]:
        stride = channel - out_shape[1]
    else:
        stride = 0
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            temp = img_gt[b][c+stride]
            temp = np.tanh(temp)
            output[b][c] = temp
    return output

def compute_itanh(img_gt, out_shape):

    output = np.zeros(out_shape)
    batch_size, channel, x, y, z = img_gt.shape
    if channel>out_shape[1]:
        stride = channel - out_shape[1]
    else:
        stride = 0
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            temp = img_gt[b][c+stride]
            temp = 1 / (1 + np.exp(-1500 * temp))
            output[b][c] = temp
    return output


def canny_edges_3d(grayImage_):
    MIN_CANNY_THRESHOLD = 0
    MAX_CANNY_THRESHOLD = 1

    # print(dim)
    b,c = np.shape(grayImage_)[0],np.shape(grayImage_)[1]
    edges_ = np.zeros(grayImage_.shape, dtype=bool)
    for i1 in range(b):
        for i2 in range(c):
            grayImage = grayImage_[i1][i2]
            dim = np.shape(grayImage)
            edges_x = np.zeros(grayImage.shape, dtype=bool)
            edges_y = np.zeros(grayImage.shape, dtype=bool)
            edges_z = np.zeros(grayImage.shape, dtype=bool)
            edges = np.zeros(grayImage.shape, dtype=bool)

            # print(np.shape(edges))

            for i in range(dim[0]):
                edges_x[i, :, :] = canny(grayImage[i, :, :], low_threshold=MIN_CANNY_THRESHOLD,
                                         high_threshold=MAX_CANNY_THRESHOLD, sigma=0)

            for j in range(dim[1]):
                edges_y[:, j, :] = canny(grayImage[:, j, :], low_threshold=MIN_CANNY_THRESHOLD,
                                         high_threshold=MAX_CANNY_THRESHOLD, sigma=0)

            for k in range(dim[2]):
                edges_z[:, :, k] = canny(grayImage[:, :, k], low_threshold=MIN_CANNY_THRESHOLD,
                                         high_threshold=MAX_CANNY_THRESHOLD, sigma=0)

            # edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD)
            for i in range(dim[0]):
                for j in range(dim[1]):
                    for k in range(dim[2]):
                        edges[i, j, k] = (edges_x[i, j, k] and edges_y[i, j, k]) or (edges_x[i, j, k] and edges_z[i, j, k]) or (
                                    edges_y[i, j, k] and edges_z[i, j, k])
                        # edges[i,j,k] = (edges_x[i,j,k]) or (edges_y[i,j,k]) or (edges_z[i,j,k])
            edges_[i1][i2] = edges
    return edges_


# def canny_edges_3d(grayImage):
#     MIN_CANNY_THRESHOLD = 0
#     MAX_CANNY_THRESHOLD = 1
#
#     dim = np.shape(grayImage)
#
#     edges_x = np.zeros(grayImage.shape, dtype=bool)
#     edges_y = np.zeros(grayImage.shape, dtype=bool)
#     edges_z = np.zeros(grayImage.shape, dtype=bool)
#     edges = np.zeros(grayImage.shape, dtype=bool)
#
#     # print(np.shape(edges))
#
#     for i in range(dim[0]):
#         edges_x[i, :, :] = canny(grayImage[i, :, :], low_threshold=MIN_CANNY_THRESHOLD,
#                                  high_threshold=MAX_CANNY_THRESHOLD, sigma=0)
#
#     for j in range(dim[1]):
#         edges_y[:, j, :] = canny(grayImage[:, j, :], low_threshold=MIN_CANNY_THRESHOLD,
#                                  high_threshold=MAX_CANNY_THRESHOLD, sigma=0)
#
#     for k in range(dim[2]):
#         edges_z[:, :, k] = canny(grayImage[:, :, k], low_threshold=MIN_CANNY_THRESHOLD,
#                                  high_threshold=MAX_CANNY_THRESHOLD, sigma=0)
#
#     # edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD)
#     for i in range(dim[0]):
#         for j in range(dim[1]):
#             for k in range(dim[2]):
#                 edges[i, j, k] = (edges_x[i, j, k] and edges_y[i, j, k]) or (edges_x[i, j, k] and edges_z[i, j, k]) or (
#                             edges_y[i, j, k] and edges_z[i, j, k])
#                 # edges[i,j,k] = (edges_x[i,j,k]) or (edges_y[i,j,k]) or (edges_z[i,j,k])
#
#     return edges


import cv2 as cv

import torch.nn.functional as F
def soft_dilate(img):
    return F.max_pool3d(img,(5,5,5),(1,1,1),(2,2,2))

def soft_erode(img):
    p1 = -F.max_pool3d(-img,(5,1,1),(1,1,1),(2,0,0))
    p2 = -F.max_pool3d(-img,(1,5,1),(1,1,1),(0,2,0))
    p3 = -F.max_pool3d(-img,(1,1,5),(1,1,1),(0,0,2))

    return torch.min(torch.min(p1,p2),p3)

if __name__ == '__main__':
    img_name = "/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/imageLa/9247140.nii.gz"
    label_name = "/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/labelLa/9247140.nii.gz"
    data_dicts = [{"image": img_name,"label": label_name}]
    from monai.transforms import LoadImaged, EnsureTyped, AsDiscreted, Compose, ToTensord, EnsureChannelFirstd, \
    Orientationd, SaveImage, ScaleIntensityd, RandCropByLabelClassesd, RandFlipd, RandAffined, RandRotate, RandRotated, \
    NormalizeIntensityd, RandZoomd, EnsureType, CopyItemsd, RandCoarseDropoutd, RandGaussianNoised, RandAdjustContrastd, \
    RandScaleIntensityd, Spacingd, Resized, Invertd, Activationsd, KeepLargestConnectedComponentd, CenterSpatialCropd

    transform = Compose([
        LoadImaged(keys=['image','label']),
        EnsureChannelFirstd(keys=['image',"label"]),
        Orientationd(keys=['image',"label"], axcodes="RAS"),
        EnsureTyped(keys=['image',"label"]),
        ScaleIntensityd(keys=["image"]),
        # AsDiscreted(keys=['label'],to_onehot=5),
    ])
    transform_test = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # NormalizeIntensityd(
            #     keys=["image", ],nonzero=True,
            # ),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
            # ToDeviced(keys=["image", "label"], device="cuda:0")
        ]
    )

    import SimpleITK as itk

    i = itk.ReadImage(img_name)
    print(i.GetSpacing())

    training_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # NormalizeIntensityd(
        #     keys=["image"], nonzero=True,
        # ),
        # Resized(keys=['image','label'],spatial_size=(80,160,160),mode=('trilinear', 'nearest'),align_corners=[True,None]),
        # Spacingd(keys=['image','label'],pixdim=[0.7, 0.3645, 0.3645],mode=('bilinear', 'nearest'),align_corners=[True,None]),
        # Spacingd(keys=['label'], pixdim=[0.7, 0.365, 0.365], mode='nearest', ),
        # CenterSpatialCropd(keys=['image','label'],roi_size=(80,160,160)),

        ScaleIntensityd(keys=["image"]),

        # RandCropByLabelClassesd(
        #     keys=["image", "label"],
        #     label_key="label",
        #     image_key="image",
        #     spatial_size=(96,96,96),
        #     num_samples=1,
        #     num_classes=5,
        #     ratios=[1, 2, 4, 2, 4]
        # ),
        # CopyItemsd(keys=["image", "label"],times=1,names=["image_ori","label_ori"]),
        # # RandFlipd(
        # #     keys=["image", "label"],
        # #     spatial_axis=[0, 1, 2],
        # #     prob=0.1,
        # # ),
        # # RandRotated(
        # #     range_x=np.pi / 15,
        # #     range_y=np.pi / 15,
        # #     range_z=np.pi / 15,
        # #     prob=0.5,
        # #     keep_size=True,
        # #     keys=['image', 'label'],
        # #     mode=('bilinear', 'nearest'), padding_mode='zeros', ),
        # RandAdjustContrastd(keys=["image"],prob=0.8),
        # RandScaleIntensityd(keys=["image"],prob=0.2,factors=0.5),
        # RandCoarseDropoutd(keys=["image", "label"],holes=1,spatial_size=16,fill_value=0,prob=0.1),
        # # RandScaleIntensityd(),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[0],
        #     prob=0.10,
        # ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[1],
        #     prob=0.10,
        # ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[2],
        #     prob=0.10,
        # ),
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=0.5, spatial_size=(96, 96, 96),
        #     rotate_range=np.pi / 15,
        #     scale_range=(0.1, 0.1, 0.1),
        #     padding_mode='zeros',
        #     as_tensor_output=True,
        # ),

        EnsureTyped(keys=["image", "label"]),
    ])

    check_ds = Dataset(data=data_dicts, transform=transform_test)
    check_loader = DataLoader(check_ds, batch_size=1)

    # data = training_transform(data_dicts)
    # label = data[0]['label']
    # img = data[0]['img']
    # data = label.unsqueeze(0)
    # img = img.unsqueeze(1)
    output_dir = 'logs'

    save0 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="aug", print_log=True)
    save1 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="aug_label", print_log=True)
    save2 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="ori", print_log=True)
    save3 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="ori_label", print_log=True)
    # out_r = compute_dct(img.numpy())
    # out_ir = compute_idct(out_r)
    for i in check_loader:
        image = i['image']
        label = i['label']
        # image_ori = i['image_ori'][0]
        # label_ori = i['label_ori'][0]

        print(image[0].shape)
        print(label[0].shape)

        # print(image_ori.shape)
        # print(label_ori.shape)
        meta_data = decollate_batch(i["image_meta_dict"])
        save0(image[0],meta_data=meta_data[0])
        # save1(label[0],meta_data=meta_data[0])

        save2(image[0])

        # img1 = itk.GetImageFromArray(image)
        # img1.SetSpacing((0.7, 0.3645, 0.3645))
        # itk.WriteImage(img1,'0_img.nii.gz')
        # seg1 = itk.GetImageFromArray(label)
        # seg1.SetSpacing((0.7, 0.3645, 0.3645))
        # itk.WriteImage(seg1,'0_seg.nii.gz')

        # print(i["image_meta_dict"])
        # print(i.keys())

        # save2(image_ori)
        # save3(label_ori)
        print(torch.max(image),torch.min(image))

    i = itk.ReadImage("logs/9247140/9247140_aug.nii.gz")
    print(i.GetSpacing())
    i = itk.ReadImage("logs/0/0_ori.nii.gz")
    print(i.GetSpacing())
    # out_r = canny_edges_3d(img.numpy())
    #
    # out_r = compute_distance(img.numpy())
    # neg_r = compute_distance(1 - img.numpy())
    # # _ = out_r.shape
    # # out_r=out_r.reshape(1,1,*_)
    # # print(out_r.shape)
    #
    # saver_o = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="ori", print_log=True)
    # saver_or = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="neg", print_log=True)
    # saver_r = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="dis", print_log=True)
    # saver_ir = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="negdis", print_log=True)
    #
    # img_o = img.numpy()[:, 0, ...]
    # img_r = out_r[:, 0, ...]
    # imgneg_r = neg_r[:, 0, ...]
    # # img_ir = out_ir[:, 0, ...]
    # saver_o(img_o)
    # saver_r(img_r)
    # saver_or(1-img_o)
    # saver_ir(imgneg_r)

    # out_1 = compute_dct(img.numpy(), (1, 1, 96, 96, 96))
    # out_2 = compute_idct(out_1, (1, 1, 96, 96, 96))
    #
    # out_3 = compute_dst(img.numpy(), (1, 1, 96, 96, 96))
    # out_4 = compute_idst(out_3, (1, 1, 96, 96, 96))
    #
    #
    #
    # saver1 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="dct", print_log=True)
    # saver2 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="idct", print_log=True)
    # saver_e1 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="error_dct", print_log=True)
    #
    # saver3 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="dst", print_log=True)
    # saver4 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="idst", print_log=True)
    # saver_e2 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="error_dst", print_log=True)
    #
    # img1 = out_1[:, 0, ...]
    # img2 = out_2[:, 0, ...]
    # saver1(img1)
    # saver2(img2)
    # saver_e1((out_2-img.numpy())[:, 0, ...])
    #
    # img3 = out_3[:, 0, ...]
    # img4 = out_4[:, 0, ...]
    # saver3(img3)
    # saver4(img4)
    # saver_e2((out_4-img.numpy())[:, 0, ...])
