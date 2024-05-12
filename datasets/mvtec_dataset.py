from __future__ import division, print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms
import random

from PIL import Image

import logging
import os.path
import pickle
import random
from typing import Any, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from glob import glob
import pandas as pd

logger = logging.getLogger("global_logger")


def build_mvtec_dataloader(cfg, training, distributed=False, category='carpet'):
    logger.info("building mvtec dataset")
    transform = transforms.Compose([
        transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ), ])

    if training:
        train_data = MVTEC(root='/kaggle/input/mvtec-ad', train=True, transform=transform, category=category,
                       resize=224, use_imagenet=True, select_random_image_from_imagenet=True,
                       shrink_factor=1)
        padded = MVTEC(root='/kaggle/input/mvtec-ad', train=True, transform=transform, category=category,
                       resize=224, use_imagenet=True, select_random_image_from_imagenet=True,
                       shrink_factor=0.9, shuffle=True, ratio=0.05, pad_train=True)
        dataset = torch.utils.data.ConcatDataset([train_data, padded])
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )
        return data_loader
    dataset_main = MVTEC(root='/kaggle/input/mvtec-ad/', train=False, transform=transform, category=category,
                         resize=224, use_imagenet=True, select_random_image_from_imagenet=True,
                         shrink_factor=1)
    dataset_shifted = MVTEC(root='/kaggle/input/mvtec-ad/', train=False, transform=transform, category=category,
                            resize=224, use_imagenet=True, select_random_image_from_imagenet=True,
                            shrink_factor=0.9)
    sampler = RandomSampler(dataset_main)
    data_loader = DataLoader(
        dataset_main,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )
    sampler = RandomSampler(dataset_shifted)
    data_loader2 = DataLoader(
        dataset_shifted,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )
    return data_loader, data_loader2


from torch.utils.data import Dataset
from PIL import ImageFilter, Image, ImageOps
from torchvision.datasets.folder import default_loader
import os


class IMAGENET30_TEST_DATASET(Dataset):
    def __init__(self, root_dir="/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        # print(f"self.class_to_idx in ImageNet30_Test_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                if instance_path != "/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/airliner/._1.JPEG":
                    for img_name in os.listdir(instance_path):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(instance_path, img_name)
                            # image = Image.open(img_path).convert('RGB')
                            self.img_path_list.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = default_loader(img_path)
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# This is a modified version of original  https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
# This file and the mvtec data directory must be in the same directory, such that:
# /.../this_directory/mvtecDataset.py
# /.../this_directory/mvtec/bottle/...
# /.../this_directory/mvtec/cable/...
# and so on

from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms
import random

from PIL import Image


def center_paste(large_img, small_img):
    # Calculate the center position
    large_width, large_height = large_img.size
    small_width, small_height = small_img.size

    # Calculate the top-left position
    left = (large_width - small_width) // 2
    top = (large_height - small_height) // 2

    # Create a copy of the large image to keep the original unchanged
    result_img = large_img.copy()

    # Paste the small image onto the large one at the calculated position
    result_img.paste(small_img, (left, top))

    return result_img


class MVTEC(data.Dataset):

    def __init__(self, root, train=True,
                 transform=None,
                 category='carpet', resize=None, use_imagenet=False,
                 select_random_image_from_imagenet=False, shrink_factor=0.9, shuffle=False, ratio=1, pad_train=False):
        self.root = root
        self.transform = transform
        self.image_files = []
        self.category = category
        print("category MVTecDataset:", category)
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.image_files = image_files
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train
        self.resize = resize
        self.use_imagenet = use_imagenet
        if use_imagenet:
            self.resize = int(resize * shrink_factor)
        self.select_random_image_from_imagenet = select_random_image_from_imagenet
        self.imagenet30_testset = IMAGENET30_TEST_DATASET()
        self.pad_train = pad_train

        if shuffle:
            random.seed(42)
            random.shuffle(self.image_files)
            sep = int(ratio * len(self.image_files))
            self.image_files = self.image_files[:sep]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        to_pil = transforms.ToPILImage()

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        if self.train and not self.pad_train:
            image = self.transform(image)
            height = image.shape[1]
            width = image.shape[2]
            state = 'train' if self.train else 'test'
            ret = {
                'filename': f'{state}_{self.pad_train}_{self.category}_{index}',
                'image': image,
                'height': height,
                'width': width,
                'label': target,
                'clsname': 'mvtec',
                'mask': torch.zeros((1, height, width)) if target == 0 else torch.ones((1, height, width))
            }
            return ret

        if self.select_random_image_from_imagenet:
            imagenet30_img = self.imagenet30_testset[int(random.random() * len(self.imagenet30_testset))][0].resize(
                (224, 224))
        else:
            imagenet30_img = self.imagenet30_testset[100][0].resize((224, 224))

        # if resizing image
        if self.use_imagenet and self.resize is not None:
            resizeTransf = transforms.Resize((self.resize, self.resize))
            image = resizeTransf(image)

        #         print(f"imagenet30_img.size: {imagenet30_img.size}")
        #         print(f"img.size: {img.size}")
        image = center_paste(imagenet30_img, image)
        image = self.transform(image)

        height = image.shape[1]
        width = image.shape[2]

        state = 'train' if self.train else 'test'

        ret = {
            'filename': f'{state}_{self.pad_train}_{self.category}_{index}',
            'image': image,
            'height': height,
            'width': width,
            'label': target,
            'clsname': 'mvtec',
            'mask': torch.zeros((1, height, width)) if target == 0 else torch.ones((1, height, width))
        }

        return ret

    def __len__(self):
        return len(self.image_files)
