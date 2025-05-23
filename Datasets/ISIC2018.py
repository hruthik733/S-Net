import os
import imageio.v2 as imageio
import PIL
import torch
import numpy as np
# import nibabel as nib
# import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
# from utils.transform import itensity_normalize
from torch.utils.data.dataset import Dataset


class ISIC2018_dataset(Dataset):
    def __init__(self, path_list, index, train_type='train', image_size=(224, 224), transform=None):
        # 根据传入的路径列表和索引生成文件地址列表
        self.image_list, self.label_list = path_list
        self.transform = transform
        self.img_size = image_size
        self.train_type = train_type
        self.index = index
        # 生成图像文件路径列表
        if self.train_type in ['train', 'val', 'test']:
            self.image_path_list = [self.image_list[x] for x in self.index]
            self.label_path_list = [self.label_list[x] for x in self.index]
        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")
        assert len(self.image_path_list) == len(self.label_path_list)
        # 读取图像测试
        # image = np.load(self.image_list[1])
        # label = np.load(self.label_list[1])
        # print(image.shape)    # 224 320 3
        # print(label.shape)    # 224 320
        # plt.imshow(label)
        # plt.show()

    def __getitem__(self, item: int):
        image_name = self.image_path_list[item]
        label_name = self.label_path_list[item]

        fallback_shape = (224, 224)

        # Load image
        try:
            ext = os.path.splitext(image_name)[-1].lower()
            if ext == '.npy':
                image = np.load(image_name)
            elif ext in ['.png', '.jpg', '.jpeg']:
                image = imageio.imread(image_name)
            else:
                raise ValueError("Unsupported image format")
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_name}: {e}")
            image = np.zeros((*fallback_shape, 3), dtype=np.uint8)

        # Load label
        try:
            ext = os.path.splitext(label_name)[-1].lower()
            if ext == '.npy':
                label = np.load(label_name)
            elif ext in ['.png', '.jpg', '.jpeg']:
                label = imageio.imread(label_name)
            else:
                raise ValueError("Unsupported label format")
        except Exception as e:
            print(f"[ERROR] Failed to load label {label_name}: {e}")
            label = np.zeros(fallback_shape, dtype=np.uint8)

        name = os.path.basename(image_name)
        sample = {'image': image, 'label': label}

        if self.transform is not None:
            sample = self.transform(sample, self.train_type, self.img_size)

        if self.train_type in ['test']:
            return name, sample['image'], sample['label']
        else:
            return sample['image'], sample['label']


    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    print(os.listdir('../segment_result/gold_standard'))