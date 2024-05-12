from __future__ import division
import pandas as pd
import logging
import os.path
import pickle
import random
from glob import glob
from typing import Any, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

logger = logging.getLogger("global_logger")


def build_visa_dataloader(cfg, training, distributed=True):
    logger.info("building wbc dataset")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ), ])

    test_set1 = MNIST_Dataset(train=False, transform=transform, test_id=1)
    test_set2 = MNIST_Dataset(train=False, transform=transform, test_id=2)
    train_set = MNIST_Dataset(train=True, transform=transform)

    if training:
        dataset = train_set
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )
        return data_loader
    dataset_main = test_set1
    dataset_shifted = test_set2
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


from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import random

def center_paste_2(large_img, small_img, shrink_factor):
    width , height = small_img.size
    large_img = large_img.resize((width, height))

    new_width = int(width * shrink_factor)
    new_height = int(height * shrink_factor)

    small_img = small_img.resize((new_width, new_height))
    small_width, small_height = small_img.size

    left = (width - small_width) // 2
    top = (height - small_height) // 2

    result_img = large_img.copy()
    result_img.paste(small_img, (left, top))
    return result_img

class IMAGENET30_TEST_DATASET(Dataset):
    def __init__(self, root_dir="./one_class_test/one_class_test/", transform=None):
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
                if instance_path != "./one_class_test/one_class_test/airliner/._1.JPEG":
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

class Train_Visa(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):

        self.transform = transform
        self.img_paths = glob.glob(root)
        self.labels = [0]*len(self.img_paths)
        print(len(self.img_paths))
        self.imagenet_30 = IMAGENET30_TEST_DATASET()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        r = random.uniform(0, 1)
        if r < 0.05:
          random_index = int(random.random() * len(self.imagenet_30))
          imagenet30_img, _ = self.imagenet_30[random_index]
          imagenet30_img = imagenet30_img.convert('RGB')
          factors = [0.98, 0.95, 0.93, 0.91, 0.88, 0.82, 0.90, 0.97, 0.85, 0.80]
          image  = center_paste_2(imagenet30_img, image, random.choice(factors))

        image = self.transform(image)
        return image, label


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

class VisaDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase, shrink_factor=None):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.imagenet30_testset = IMAGENET30_TEST_DATASET()
        self.shrink_factor = shrink_factor
        print(f"self.shrink_factor: {self.shrink_factor}")

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        if self.shrink_factor:
            pad_img, _ = self.imagenet30_testset[int(random.random() * len(self.imagenet30_testset))]
            pad_img = pad_img.resize(img.size)

            img = img.resize((int(img.size[0] * self.shrink_factor), int(img.size[1] * self.shrink_factor)))

            img = center_paste(pad_img, img)

        img = self.transform(img)

        return img, label
