from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    ScaleIntensityd,
    EnsureType,
    Invertd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandCropByLabelClassesd,
    RandAffined,
    NormalizeIntensityd,
    RandFlip,
    ToDeviced,
    SaveImage, CopyItemsd, OneOf, RandCoarseDropoutd, RandCoarseShuffled, RandFlipd, RandRotated, RandAdjustContrastd,
    RandGaussianNoised, RandScaleIntensityd, LabelFilterd, Resized, Zoomd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
from hparam_h import hparams as hp


base_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/'

class MedData_train():
    def __init__(self, images_dir, labels_dir,unlabeled=True,batch_size = 2,contrast=False):

        self.train_images = sorted(
            glob.glob(os.path.join(images_dir, hp.fold_arch)))
        self.train_labels = sorted(
            glob.glob(os.path.join(labels_dir, hp.fold_arch)))

        self.data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(self.train_images, self.train_labels)
        ]
        self.contrast = contrast

        if unlabeled:
            self.transforms = self.transform(True)
            self.labeled = False
        else:
            self.transforms = self.transform(False)
            self.labeled = True

        check_ds = Dataset(data=self.data_dicts, transform=self.transforms)
        self.check_loader = DataLoader(check_ds, batch_size=1)


        train_ds = CacheDataset(
            data=self.data_dicts, transform=self.transforms,cache_num=8,
            num_workers=6,
            copy_cache = True,
        )

        g = torch.Generator()
        g.manual_seed(1337)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True,pin_memory=True,generator=g)

    def transform(self,unlabeled=True):
        train_Transforms=get_transforms(trainortest='train', aug=hp.aug, labeled=not unlabeled, contrast=self.contrast)
        return train_Transforms


class MedData_test():
    def __init__(self, images_dir, labels_dir):

        self.subjects = []


        self.train_images = sorted(
            glob.glob(os.path.join(images_dir, hp.fold_arch)))
        self.train_labels = sorted(
            glob.glob(os.path.join(labels_dir, hp.fold_arch)))

        self.data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(self.train_images, self.train_labels)
        ]

        # self.transforms =None
        self.transforms = self.transform()

        check_ds = Dataset(data=self.data_dicts, transform=self.transforms)
        self.check_loader = DataLoader(check_ds, batch_size=1)

        test_ds = CacheDataset(
            data=self.data_dicts, transform=self.transforms,
            cache_num=6, num_workers=4,
            copy_cache = True,
        )
        g = torch.Generator()
        g.manual_seed(1337)
        self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,drop_last=True, num_workers=4,pin_memory=True,generator=g)

    def transform(self):
        val_transforms = get_transforms(trainortest='test',aug=True,labeled=True,contrast=False)
        return val_transforms

class MedData_val():
    def __init__(self, images_dir, labels_dir):

        self.subjects = []


        self.train_images = sorted(
            glob.glob(os.path.join(images_dir, hp.fold_arch)))
        self.train_labels = sorted(
            glob.glob(os.path.join(labels_dir, hp.fold_arch)))

        self.data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(self.train_images, self.train_labels)
        ]

        # self.transforms =None
        self.transforms = self.transform()

        check_ds = Dataset(data=self.data_dicts, transform=self.transforms)
        self.check_loader = DataLoader(check_ds, batch_size=1)

        val_ds = CacheDataset(
            data=self.data_dicts, transform=self.transforms,
            cache_num=6, num_workers=4,
            copy_cache = True,
        )
        g = torch.Generator()
        g.manual_seed(1337)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,drop_last=True, num_workers=4,pin_memory=True,generator=g)

    def transform(self):
        val_transforms = get_transforms(trainortest='val',aug=True,labeled=True,contrast=False)
        return val_transforms
class Data_fetcher():
    def __init__(self,loader1,loader2,mode=1,loader_nums=2,length=0):
        self.loader1 = loader1 #labeled
        self.loader2 = loader2 #unlabeled
        self.mode = mode #0为半监督 1为全监督
        if length == 0:
            self.length = len(loader1)+len(loader2)
        else:
            self.length= length

        if loader_nums == 1:
            self.loader_nums = 1
            self.iter1 = iter(loader1)
        elif loader_nums == 2:
            self.loader_nums = 2
            self.iter1 = iter(loader1)
            self.iter2 = iter(loader2)

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


def get_transforms(trainortest='train',aug=True,labeled=True,contrast=False):

    if trainortest=='train':
        if labeled:
            if aug :
                if contrast:
                    transform = Compose([
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="RAS"),
                        ScaleIntensityd(keys=["image"]),

                        # Resized(
                        #     keys=["image", "label"],
                        #     spatial_size=hp.patch_size,
                        #     size_mode="all",
                        #     mode=["area", 'nearest']
                        # ),
                        CopyItemsd(keys=["image", "label"], times=1, names=["image_aug", "label_aug"]),
                        RandAdjustContrastd(keys=["image_aug"], prob=0.8),
                        RandScaleIntensityd(keys=["image_aug"], prob=0.2, factors=0.5),
                        RandCoarseDropoutd(keys=["image_aug", "label_aug"], holes=2, spatial_size=16, fill_value=0, prob=0.1),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[0],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[1],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[2],
                            prob=0.50,
                        ),
                        # RandAffined(
                        #     keys=['image', 'label'],
                        #     mode=('bilinear', 'nearest'),
                        #     prob=0.5, spatial_size=(96, 96, 96),
                        #     rotate_range=np.pi / 15,
                        #     scale_range=(0.1, 0.1, 0.1),
                        #     padding_mode='zeros',
                        #     as_tensor_output=True,
                        # ),
                        RandFlipd(
                            keys=["image_aug", "label_aug"],
                            spatial_axis=[0],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image_aug", "label_aug"],
                            spatial_axis=[1],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image_aug", "label_aug"],
                            spatial_axis=[2],
                            prob=0.50,
                        ),
                        RandAffined(
                            keys=['image_aug', 'label_aug'],
                            mode=('bilinear', 'nearest'),
                            prob=0.5, spatial_size=(160,288,288),
                            rotate_range=np.pi / 15,
                            scale_range=(0.1, 0.1, 0.1),
                            padding_mode='zeros',
                            as_tensor_output=True,
                        ),

                        EnsureType(),
                    ])
                else:
                    transform = Compose([
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="RAS"),
                        ScaleIntensityd(keys=["image"]),
                        # Resized(
                        #     keys=["image", "label"],
                        #     spatial_size=hp.patch_size,
                        #     size_mode="all",
                        #     mode=["area", 'nearest']
                        # ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[0],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[1],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[2],
                            prob=0.50,
                        ),
                        # RandAffined(
                        #     keys=['image', 'label'],
                        #     mode=('bilinear', 'nearest'),
                        #     prob=0.5, spatial_size=(96, 96, 96),
                        #     rotate_range=np.pi / 15,
                        #     scale_range=(0.1, 0.1, 0.1),
                        #     padding_mode='zeros',
                        #     as_tensor_output=True,
                        # ),
                        EnsureType(),
                    ])
            else:
                transform = Compose([
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    ScaleIntensityd(keys=["image"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=hp.patch_size,
                        size_mode="all",
                        mode=["area", 'nearest']
                    ),
                    EnsureType(),
                ])

        elif not labeled:
            if aug :
                if contrast:
                    transform = Compose([
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="RAS"),
                        ScaleIntensityd(keys=["image"]),

                        # Resized(
                        #     keys=["image", "label"],
                        #     spatial_size=hp.patch_size,
                        #     size_mode="all",
                        #     mode=["area", 'nearest']
                        # ),
                        CopyItemsd(keys=["image", "label"], times=1, names=["image_aug", "label_aug"]),
                        RandAdjustContrastd(keys=["image_aug"], prob=0.8),
                        RandScaleIntensityd(keys=["image_aug"], prob=0.2, factors=0.5),
                        RandCoarseDropoutd(keys=["image_aug", "label_aug"], holes=2, spatial_size=16, fill_value=0,
                                           prob=0.1),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[0],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[1],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[2],
                            prob=0.50,
                        ),
                        # RandAffined(
                        #     keys=['image', 'label'],
                        #     mode=('bilinear', 'nearest'),
                        #     prob=0.5, spatial_size=(96, 96, 96),
                        #     rotate_range=np.pi / 15,
                        #     scale_range=(0.1, 0.1, 0.1),
                        #     padding_mode='zeros',
                        #     as_tensor_output=True,
                        # ),
                        RandFlipd(
                            keys=["image_aug", "label_aug"],
                            spatial_axis=[0],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image_aug", "label_aug"],
                            spatial_axis=[1],
                            prob=0.50,
                        ),
                        RandFlipd(
                            keys=["image_aug", "label_aug"],
                            spatial_axis=[2],
                            prob=0.50,
                        ),
                        RandAffined(
                            keys=['image_aug', 'label_aug'],
                            mode=('bilinear', 'nearest'),
                            prob=0.5, spatial_size=(160,288,288),
                            rotate_range=np.pi / 15,
                            scale_range=(0.1, 0.1, 0.1),
                            padding_mode='zeros',
                            as_tensor_output=True,
                        ),

                        EnsureType(),
                    ])
                else:
                    transform = Compose([
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="RAS"),
                        LabelFilterd(keys=["label"],applied_labels=[0]),
                        ScaleIntensityd(keys=["image"]),
                        # Resized(
                        #     keys=["image", "label"],
                        #     spatial_size=hp.patch_size,
                        #     size_mode="all",
                        #     mode=["area", 'nearest']
                        # ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[0],
                            prob=0.10,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[1],
                            prob=0.10,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[2],
                            prob=0.10,
                        ),
                        # RandAffined(
                        #     keys=['image', 'label'],
                        #     mode=('bilinear', 'nearest'),
                        #     prob=0.5, spatial_size=sample_size,
                        #     rotate_range=np.pi / 15,
                        #     scale_range=(0.1, 0.1, 0.1),
                        #     padding_mode='zeros',
                        #     as_tensor_output=True,
                        # ),

                        EnsureType(),
                    ])


            else:
                transform = Compose([
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    LabelFilterd(keys=["label"], applied_labels=[0]),
                    ScaleIntensityd(keys=["image"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=hp.patch_size,
                        size_mode="all",
                        mode=["area", 'nearest']
                    ),
                    EnsureType(),
                ])

    elif trainortest=='test':
            transform = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    # NormalizeIntensityd(
                    #     keys=["image", ],nonzero=True,
                    # ),
                    # Resized(
                    #     keys=["image", "label"],
                    #     spatial_size=hp.patch_size,
                    #     size_mode="all",
                    #     mode=["area", 'nearest']
                    # ),

                    ScaleIntensityd(keys=["image"]),
                    EnsureTyped(keys=["image", "label"]),
                    # ToDeviced(keys=["image", "label"], device="cuda:0")
                ])
    elif trainortest == 'val':
        transform = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(
                keys=["image"],
                spatial_size=hp.patch_size,
                size_mode="all",
                mode=["area"]
            ),
            ScaleIntensityd(keys=["image"]),
            EnsureType(),
        ])
    return transform



if __name__ == '__main__':
    unlabeled_image_dir = '/home/dluser/dataset/ZDY_Dataset/dataset/oai/OAI-ZIB/imageUn96'
    unlabeled_label_dir = '/home/dluser/dataset/ZDY_Dataset/dataset/oai/OAI-ZIB/labelUn96'
    labeled_image_dir = '/home/dluser/dataset/ZDY_Dataset/dataset/oai/OAI-ZIB/imageLa4'
    labeled_label_dir = '/home/dluser/dataset/ZDY_Dataset/dataset/oai/OAI-ZIB/labelLa4'

    save_dir="/home/dluser/ZDY/bone/pytorch_medical/Pytorch-Medical-Segmentation/results/test"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    un_util = MedData_train(unlabeled_image_dir, unlabeled_label_dir, True, batch_size=2,contrast=True)
    un_loader = un_util.train_loader

    la_util = MedData_train(labeled_image_dir, labeled_label_dir, False, batch_size=2,contrast=True)
    la_loader = la_util.train_loader

    fetcher = Data_fetcher(la_loader,un_loader,1,2,hp.iters_perepoch)
    # ts_util = MedData_test(labeled_image_dir,labeled_label_dir)
    # ts_loader = ts_util.test_loader

    iters = hp.iters_perepoch


    index = 0


    saver_la = SaveImage(output_dir=save_dir, output_ext=".nii.gz", output_postfix="la",print_log=True)
    saver_lala = SaveImage(output_dir=save_dir, output_ext=".nii.gz", output_postfix="lala",print_log=True)
    saver_un = SaveImage(output_dir=save_dir, output_ext=".nii.gz", output_postfix="un",print_log=True)
    saver_unla = SaveImage(output_dir=save_dir, output_ext=".nii.gz", output_postfix="unla",print_log=True)

    saver_la_aug = SaveImage(output_dir=save_dir, output_ext=".nii.gz", output_postfix="la_aug",print_log=True)
    saver_lala_aug = SaveImage(output_dir=save_dir, output_ext=".nii.gz", output_postfix="lala_aug",print_log=True)
    saver_un_aug = SaveImage(output_dir=save_dir, output_ext=".nii.gz", output_postfix="un_aug",print_log=True)
    saver_unla_aug = SaveImage(output_dir=save_dir, output_ext=".nii.gz", output_postfix="unla_aug",print_log=True)
    for epoch in range(1, 2):
        print("epoch:" + str(epoch))

        epoch_loss = 0
        step = 0

        # save_ = 1
        for i in range(0,2):

            batch1, batch2 = fetcher.fetch()
            input1, label1 = (
                batch1["image"].to(device),
                batch1["label"].to(device),
            )
            input2, label2 = (
                batch2["image"].to(device),
                batch2["label"].to(device),
            )
            input1_aug, label1_aug = (
                batch1["image_aug"].to(device),
                batch1["label_aug"].to(device),
            )
            input2_aug, label2_aug = (
                batch2["image_aug"].to(device),
                batch2["label_aug"].to(device),
            )
            meta_data1 = decollate_batch(batch1["image_meta_dict"])

            saver_la(input1[0], {"filename_or_obj":batch1["image_meta_dict"]["filename_or_obj"][0],"affine":batch1["image_meta_dict"]["affine"][0]})
            saver_lala(label1[0],{"filename_or_obj":batch1["label_meta_dict"]["filename_or_obj"][0],"affine":batch1["label_meta_dict"]["affine"][0]})
            saver_la_aug(input1_aug[0],{"filename_or_obj":batch1["image_meta_dict"]["filename_or_obj"][0],"affine":batch1["image_meta_dict"]["affine"][0]})
            saver_lala_aug(label1_aug[0],{"filename_or_obj":batch1["image_meta_dict"]["filename_or_obj"][0],"affine":batch1["image_meta_dict"]["affine"][0]})


            saver_un(input2[0], {"filename_or_obj":batch2["image_meta_dict"]["filename_or_obj"][0],"affine":batch2["image_meta_dict"]["affine"][0]})
            saver_unla(label2[0], {"filename_or_obj":batch2["image_meta_dict"]["filename_or_obj"][0],"affine":batch2["image_meta_dict"]["affine"][0]})
            saver_un_aug(input2_aug[0], {"filename_or_obj":batch2["image_meta_dict"]["filename_or_obj"][0],"affine":batch2["image_meta_dict"]["affine"][0]})
            saver_unla_aug(label2_aug[0], {"filename_or_obj":batch2["image_meta_dict"]["filename_or_obj"][0],"affine":batch2["image_meta_dict"]["affine"][0]})
            print(input1.shape,input1_aug.shape)
            print(label1.shape,label1_aug.shape)