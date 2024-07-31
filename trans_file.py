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
from skimage.morphology import disk, erosion, ball, dilation
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
import torch.nn.functional as F

def soft_dilate(img):
    return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

def soft_erode(img):
    p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
    p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
    p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))

    return torch.min(torch.min(p1,p2),p3)
def tensor_erode(bin_img, ksize=5):
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, D, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad, pad, pad], mode='constant', value=0)

    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    patches = patches.unfold(dimension=4, size=ksize, step=1)
    # B x C x H x W x k x k

    # 取每个 patch 中最小的值，i.e., 0
    eroded, _ = patches.reshape(B, C, D, H, W, -1).min(dim=-1)
    return eroded
def tensor_dilate(bin_img, ksize=5):
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, D, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad, pad, pad], mode='constant', value=0)

    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    patches = patches.unfold(dimension=4, size=ksize, step=1)
    # B x C x H x W x k x k

    # 取每个 patch 中最小的值，i.e., 0
    dilated, _ = patches.reshape(B, C, D, H, W, -1).max(dim=-1)
    return dilated

def compute_erosion(img_gt):

    batch_size,c,  x, y, z = img_gt.shape
    out = torch.zeros_like(img_gt)
    print(img_gt.shape)
    for i in range(c):
        out[:,i:i+1,...]= tensor_erode(img_gt[:,i:i+1,...])
    return out

def compute_dilation(img_gt):

    batch_size,c,  x, y, z = img_gt.shape
    out = torch.zeros_like(img_gt)
    print(img_gt.shape)
    for i in range(c):
        out[:,i:i+1,...]= tensor_dilate(img_gt[:,i:i+1,...])
    return out
if __name__ == '__main__':
    img_name = "/home/dluser/dataset/ZDY_Dataset/dataset/oai/OAI-ZIB/labelVal/9993833.nii.gz"
    label_name = "/home/dluser/dataset/ZDY_Dataset/dataset/oai/OAI-ZIB/labelVal/9993833.nii.gz"
    data_dicts = [{"image": img_name,"label": label_name}]
    from monai.transforms import LoadImaged, EnsureTyped, AsDiscreted, Compose, ToTensord, EnsureChannelFirstd, \
    Orientationd, SaveImage, ScaleIntensityd, RandCropByLabelClassesd, RandFlipd, RandAffined, RandRotate, RandRotated, \
    NormalizeIntensityd, RandZoomd, EnsureType, CopyItemsd, RandCoarseDropoutd, RandGaussianNoised, RandAdjustContrastd, \
    RandScaleIntensityd, Spacingd, Resized, Invertd, Activationsd, KeepLargestConnectedComponentd, CenterSpatialCropd


    training_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # CenterSpatialCropd(keys=['image','label'],roi_size=(96,96,96)),
        AsDiscreted(keys="label",to_onehot=5),
        # ScaleIntensityd(keys=["image"]),
        EnsureType(),
    ])


    check_ds = Dataset(data=data_dicts, transform=training_transform)
    check_loader = DataLoader(check_ds, batch_size=1)

    output_dir = 'logs'

    # post_output = Compose([
    #     Invertd(
    #         keys="pred",
    #         transform=training_transform,
    #         orig_keys="image",
    #         meta_keys="pred_meta_dict",
    #         orig_meta_keys="image_meta_dict",
    #         meta_key_postfix="meta_dict",
    #         nearest_interp=False,
    #         to_tensor=True,
    #     ),
    #     AsDiscreted(keys="pred",argmax=True, to_onehot=5),
    #     KeepLargestConnectedComponentd(keys="pred",is_onehot=True,applied_labels=[1,2,3])
    # ])
#
#
    save0 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="aug", print_log=True)
    save1 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="label", print_log=True)
    save2 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="ero1", print_log=True)
    save3 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="dila1", print_log=True)
    save4 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="ero2", print_log=True)
    save5 = SaveImage(output_dir=output_dir, output_ext=".nii.gz", output_postfix="dila2", print_log=True)
    # out_r = compute_dct(img.numpy())
    # out_ir = compute_idct(out_r)
    for i in check_loader:
        image = i['image']
        label = i['label']

        # print(image.shape)
        print(label.shape)
        # print(image_ori.shape)
        # print(label_ori.shape)
        # save0(image)
        # print(torch.nn.functional.softmax(label,dim=1).shape)
        # print(torch.argmax(torch.nn.functional.softmax(label,dim=1),dim=1,keepdim=True).shape)
        a=torch.argmax(torch.nn.functional.softmax(label,dim=1),dim=1,keepdim=False)
        save1(a)
        b=torch.argmax((torch.nn.functional.softmax(soft_dilate(label),dim=1)),dim=1,keepdim=False)
        # save1(torch.argmax(torch.nn.functional.softmax(label, dim=1), dim=1, keepdim=False))
        # save2(torch.argmax(compute_erosion(torch.nn.functional.softmax(label,dim=1)),dim=1,keepdim=False))
        # save3(torch.argmax(compute_dilation(torch.nn.functional.softmax(label,dim=1)),dim=1,keepdim=False))
        save2(b)
        c = torch.argmax((torch.nn.functional.softmax(soft_erode(label),dim=1)),dim=1,keepdim=False)
        save3(c)
        save0(a-c)
        # save4(torch.argmax(tensor_erode(torch.nn.functional.softmax((label),dim=1)),dim=1,keepdim=False))
        # save5(torch.argmax(tensor_dilate(torch.nn.functional.softmax((label),dim=1)),dim=1,keepdim=False))

        # print(torch.max(image),torch.min(image))
#

# val_org_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image"]),
#         Orientationd(keys=["image"], axcodes="RAS"),
#         Spacingd(keys=["image"], pixdim=[1.4, 0.85, 0.85], mode="bilinear"),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#     ]
# )

# val_org_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(keys=['image'], pixdim=[1.4, 0.85, 0.85], mode="bilinear",align_corners=True),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#         # ScaleIntensityd(keys=["image"]),
#         ScaleIntensityd(keys=["image"]),
#         EnsureTyped(keys=["image", "label"]),
#     ]
# )
# img_name = "/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/imageLa/9247140.nii.gz"
# label_name = "/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/labelLa/9247140.nii.gz"
# data_dicts = [{"image": img_name, "label": label_name}]
#
# val_org_ds = Dataset(data=data_dicts, transform=val_org_transforms)
# val_org_loader = DataLoader(val_org_ds, batch_size=1, )
# post_transforms = Compose([
#     Invertd(
#         keys="pred",
#         transform=val_org_transforms,
#         orig_keys="image",
#         meta_keys="pred_meta_dict",
#         orig_meta_keys="image_meta_dict",
#         meta_key_postfix="meta_dict",
#         nearest_interp=False,
#         to_tensor=True,
#     ),
#     AsDiscreted(keys="pred",argmax=True, to_onehot=5),
#     KeepLargestConnectedComponentd(keys="pred",is_onehot=True,applied_labels=[1,2,3])
# ])
# for val_data in val_org_loader:
#     val_inputs = val_data["image"]
#
#     val_data["pred"] = torch.zeros_like(val_inputs)
#     val_data = [post_transforms(i) for i in decollate_batch(val_data)]
#     val_outputs, val_labels = from_engine(["pred", "label"])(val_data)