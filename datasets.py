import sys
import glob
from os.path import join
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import config as c
from natsort import natsorted


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if self.mode == "train":
            # TRAIN SETTING
            if c.Dataset_mode == 'DIV2K':
                self.TRAIN_PATH = c.TRAIN_PATH_DIV2K
                self.format_train = 'png'
                print('TRAIN DATASET is DIV2K')

            if c.Dataset_mode == 'COCO':
                self.TRAIN_PATH = c.TEST_PATH_COCO
                self.format_train = 'jpg'
                print('TRAIN DATASET is COCO')

            # train
            self.files = natsorted(sorted(glob.glob(self.TRAIN_PATH + "/*." + self.format_train)))

        if self.mode == "val":
            # VAL SETTING
            if c.Dataset_VAL_mode == 'DIV2K':
                self.VAL_PATH = c.VAL_PATH_DIV2K
                self.format_val = 'png'
                print('VAL DATASET is DIV2K')

            if c.Dataset_VAL_mode == 'COCO':
                self.VAL_PATH = c.VAL_PATH_COCO
                self.format_val = 'jpg'
                print('VAL DATASET is COCO')

            if c.Dataset_VAL_mode == 'ImageNet':
                self.VAL_PATH = c.VAL_PATH_IMAGENET
                self.format_val = 'JPEG'
                print('VAL DATASET is ImageNet')

            # test
            self.files = sorted(glob.glob(self.VAL_PATH + "/*." + self.format_val))

    def __getitem__(self, index):
        try:
                image = Image.open(self.files[index])
                image = to_rgb(image)
                item = self.transform(image)
                return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)


if c.Dataset_VAL_mode == 'DIV2K':
    cropsize_val = c.cropsize_val_div2k
if c.Dataset_VAL_mode == 'COCO':
    cropsize_val = c.cropsize_val_coco
if c.Dataset_VAL_mode == 'ImageNet':
    cropsize_val = c.cropsize_val_imagenet

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(cropsize_val),
    T.ToTensor(),
])


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)