import copy
import math
import os
import glob
import random
import shutil
from os.path import join
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# from utils import read_list, read_nifti, config
np.random.seed(1337)
random.seed(1337)


class Data_Preprocess:
    def __init__(self, Base_path):
        self.Base_path = Base_path
        self.x_min = 999
        self.x_max = 0
        self.y_min = 999
        self.y_max = 0
        self.z_min = 999
        self.z_max = 0

    def write_txt(self, list, path):

        file = open(path, 'w')
        for data in list:
            file.write(str(data) + '\n')

    def do_select(self):
        self.find_bound(self.Train_ids)
        self.find_bound(self.Test_ids)
        self.find_bound(self.Val_ids)
        print(self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.z_range = self.z_max - self.z_min
        if self.x_range < self.y_range:
            self.x_range = self.y_range

        if not isinstance(self.x_range / 16, int):
            self.x_range = math.ceil(self.x_range / 16) * 16
        if not isinstance(self.z_range / 16, int):
            self.z_range = math.ceil(self.z_range / 16) * 16

        # print(self.x_range,self.y_range,self.z_range)
        self.do_select_dataset(self.Train_ids, 'Train')
        self.do_select_dataset(self.Test_ids, 'Test')
        self.do_select_dataset(self.Val_ids, 'Val')

    def do_default(self):

        self.do_process(self.unlabeled_ids, f'Un{self.un}')
        self.do_process(self.labeled_ids, f'La{self.la}')

    def select_dataset(self):
        self.img_ids = []
        for path in glob.glob(os.path.join(base_dir, f'image', '*.mhd')):
            img_id = path.split('/')[-1].split('image')[-1].split('.mhd')[0]
            self.img_ids.append(img_id)

        print(len( self.img_ids))
        self.Train_ids = []
        for i in range(0, 60):
            img_idx = random.choices(list(range(len(self.img_ids))), k=1)[0]
            self.Train_ids.append(self.img_ids[img_idx])
            self.img_ids.pop(int(img_idx))
        self.write_txt(
            self.Train_ids,
            os.path.join(base_dir, 'splits/Train.txt')
        )
        self.Test_ids = []
        for i in range(0, 0):
            img_idx = random.choices(list(range(len(self.img_ids))), k=1)[0]
            self.Test_ids.append(self.img_ids[img_idx])
            self.img_ids.pop(int(img_idx))
        self.write_txt(
            self.Test_ids,
            os.path.join(base_dir, 'splits/Test.txt')
        )
        self.Val_ids = []
        for i in range(0, 0):
            img_idx = random.choices(list(range(len(self.img_ids))), k=1)[0]
            self.Val_ids.append(self.img_ids[img_idx])
            self.img_ids.pop(int(img_idx))
        self.write_txt(
            self.Val_ids,
            os.path.join(base_dir, 'splits/Val.txt')
        )

    def split_dataset(self, la):
        self.la = la
        self.un = len(self.Train_ids) - la
        self.temp_ids = copy.copy(self.Train_ids)
        self.labeled_ids = []
        for i in range(0, la):
            img_idx = random.choices(list(range(len(self.temp_ids))), k=1)[0]
            self.labeled_ids.append(self.temp_ids[img_idx])
            self.temp_ids.pop(int(img_idx))
        self.write_txt(
            self.labeled_ids,
            os.path.join(base_dir, f'splits/labeled{self.la}.txt')
        )
        self.unlabeled_ids = self.temp_ids
        self.write_txt(
            self.unlabeled_ids,
            os.path.join(base_dir, f'splits/unlabeled{self.un}.txt')
        )

    def find_bound(self, data_list):
        for i, img_id in enumerate(data_list):
            label_path = os.path.join(base_dir, f'label', f'labels{img_id}.mhd')
            label = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label)
            label_array[label_array == 1] = 0
            label_array[label_array == 3] = 0
            label_array[label_array == 2] = 1
            label_array[label_array == 4] = 2
            label_array[label_array == 5] = 3
            nozero = label_array.nonzero()
            xmin = min(nozero[1])
            xmax = max(nozero[1])
            ymin = min(nozero[0])
            ymax = max(nozero[0])
            zmin = min(nozero[2])
            zmax = max(nozero[2])
            if self.x_max < xmax: self.x_max = xmax
            if self.x_min > xmin: self.x_min = xmin
            if self.y_max < ymax: self.y_max = ymax
            if self.y_min > ymin: self.y_min = ymin
            if self.z_max < zmax: self.z_max = zmax
            if self.z_min > zmin: self.z_min = zmin

    def do_select_dataset(self, data_list, tag):
        data_num = len(data_list)
        print(tag, 'set has {} images'.format(data_num))

        if not os.path.exists(join(self.Base_path, f'image{tag}')):  # 创建保存目录
            os.makedirs(join(join(self.Base_path, f'image{tag}')))
        if not os.path.exists(join(self.Base_path, f'label{tag}')):  # 创建保存目录
            os.makedirs(join(join(self.Base_path, f'label{tag}')))

        for i, img_id in enumerate(data_list):
            print("==== {}/{} ====".format(i + 1, data_num))
            image_path = os.path.join(base_dir, f'image', f'image{img_id}.mhd')
            label_path = os.path.join(base_dir, f'label', f'labels{img_id}.mhd')
            # shutil.copy(image_path,join(self.Base_path, f'image{tag}', f'{img_id}.nii.gz'))
            label = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label)

            print(label_array.shape)
            # label_array[label_array == 1] = 0
            # label_array[label_array == 3] = 0
            # label_array[label_array == 2] = 1
            # label_array[label_array == 4] = 2
            # label_array[label_array == 5] = 3
            # nozero = label_array.nonzero()
            # xmin = min(nozero[1])
            # xmax = max(nozero[1])
            # ymin = min(nozero[0])
            # ymax = max(nozero[0])
            # zmin = min(nozero[2])
            # zmax = max(nozero[2])
            # x_range = xmax - xmin
            # y_range = ymax - ymin
            # if x_range < y_range:
            #     x_range = y_range
            # z_range = zmax - zmin
            # x_temp = math.floor((self.x_range - x_range) / 2)
            # z_temp = math.floor((self.z_range - z_range) / 2)
            #
            # x_start = max(0, ymin - x_temp)
            # x_end = x_start + self.x_range
            # y_start = max(0, xmin - x_temp)
            # y_end = y_start + self.x_range
            #
            # if (x_end > 384):
            #     x_end = 384
            #     x_start = x_end - self.x_range
            # if (y_end > 384):
            #     y_end = 384
            #     y_start = y_end - self.x_range
            # temp_array = label_array[x_start:x_end, y_start:y_end, :]
            # print(temp_array.shape)
            # temp_image = sitk.GetImageFromArray(temp_array.astype(np.uint8))
            # temp_image.SetOrigin(label.GetOrigin())
            # temp_image.SetSpacing(label.GetSpacing())
            # temp_image.SetDirection(label.GetDirection())
            # sitk.WriteImage(temp_image, join(self.Base_path, f'label{tag}', f'{img_id}.nii.gz'))
            #
            # image = sitk.ReadImage(image_path, sitk.sitkFloat32)
            # image_array = sitk.GetArrayFromImage(image)
            # temp_array = image_array[x_start:x_end, y_start:y_end, :]
            # temp_image = sitk.GetImageFromArray(temp_array)
            # temp_image.SetOrigin(label.GetOrigin())
            # temp_image.SetSpacing(label.GetSpacing())
            # temp_image.SetDirection(label.GetDirection())
            # sitk.WriteImage(temp_image, join(self.Base_path, f'image{tag}', f'{img_id}.nii.gz'))

    def do_process(self, data_list, tag):
        data_num = len(data_list)
        print(tag, 'set has {} images'.format(data_num))

        if not os.path.exists(join(self.Base_path, f'image{tag}')):  # 创建保存目录
            os.makedirs(join(join(self.Base_path, f'image{tag}')))
        if not os.path.exists(join(self.Base_path, f'label{tag}')):  # 创建保存目录
            os.makedirs(join(join(self.Base_path, f'label{tag}')))

        for i, img_id in enumerate(data_list):
            print("==== {}/{} ====".format(i + 1, data_num))
            image_path = os.path.join(base_dir, f'imageTrain', f'{img_id}.nii.gz')
            label_path = os.path.join(base_dir, f'labelTrain', f'{img_id}.nii.gz')
            shutil.copy(image_path, join(self.Base_path, f'image{tag}', f'{img_id}.nii.gz'))
            shutil.copy(label_path, join(self.Base_path, f'label{tag}', f'{img_id}.nii.gz'))


if __name__ == '__main__':

    base_dir =  r'/home/dluser/dataset/ZDY_Dataset/dataset/oai/OAI-ZIB'#r"/home/dluser/dataset/ZDY_Dataset/dataset/ski10"  #

    splits_path = os.path.join(base_dir, 'splits')
    if not os.path.exists(splits_path):
        os.makedirs(splits_path)
    data_preprocess = Data_Preprocess(base_dir)
    data_preprocess.select_dataset()
    data_preprocess.do_select()
    # data_preprocess.split_dataset(4)
    # data_preprocess.do_default()
    # data_preprocess.split_dataset(8)
    # data_preprocess.do_default()
    # data_preprocess.split_dataset(20)
    # data_preprocess.do_default()
