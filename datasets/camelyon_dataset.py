from __future__ import division

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

ones = torch.ones((1, 224, 224))
zeros = torch.zeros((1, 224, 224))


def build_camelyon_dataloader(cfg, training, distributed=True):
    logger.info("building camelyon dataset")
    transform = transforms.Compose([
        transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ), ])

    node0_train = glob('/kaggle/input/camelyon17-clean/node0/train/normal/*')
    node1_train = glob('/kaggle/input/camelyon17-clean/node1/train/normal/*')
    node2_train = glob('/kaggle/input/camelyon17-clean/node2/train/normal/*')

    train_path = node0_train + node1_train + node2_train
    train_label = [0] * len(train_path)

    node0_test_normal = glob('/kaggle/input/camelyon17-clean/node0/test/normal/*')
    node0_test_anomaly = glob('/kaggle/input/camelyon17-clean/node0/test/anomaly/*')

    node1_test_normal = glob('/kaggle/input/camelyon17-clean/node1/test/normal/*')
    node1_test_anomaly = glob('/kaggle/input/camelyon17-clean/node1/test/anomaly/*')

    node2_test_normal = glob('/kaggle/input/camelyon17-clean/node2/test/normal/*')
    node2_test_anomaly = glob('/kaggle/input/camelyon17-clean/node2/test/anomaly/*')

    test_path_normal = node0_test_normal + node1_test_normal + node2_test_normal
    test_path_anomaly = node0_test_anomaly + node1_test_anomaly + node2_test_anomaly

    test_path = test_path_normal + test_path_anomaly
    test_label = [0] * len(test_path_normal) + [1] * len(test_path_anomaly)

    node3_test_normal = glob('/kaggle/input/camelyon17-clean/node3/test/normal/*')
    node3_test_anomaly = glob('/kaggle/input/camelyon17-clean/node3/test/anomaly/*')

    node4_test_normal = glob('/kaggle/input/camelyon17-clean/node4/test/normal/*')
    node4_test_anomaly = glob('/kaggle/input/camelyon17-clean/node4/test/anomaly/*')

    shifted_test_path_normal = node3_test_normal + node4_test_normal
    shifted_test_path_anomaly = node3_test_anomaly + node4_test_anomaly

    shifted_test_path = shifted_test_path_normal + shifted_test_path_anomaly
    shifted_test_label = [0] * len(shifted_test_path_normal) + [1] * len(shifted_test_path_anomaly)

    if training:
        dataset = Camelyon(image_path=train_path, labels=train_label, transform=transform)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )
        return data_loader
    dataset_main = Camelyon(image_path=test_path, labels=test_label, transform=transform)
    dataset_shifted = Camelyon(image_path=shifted_test_path, labels=shifted_test_label, transform=transform)
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


class Camelyon(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        height = image.shape[1]
        width = image.shape[2]

        ret = {
            'filename': os.path.basename(image_file),
            'image': image,
            'height': height,
            'width': width,
            'label': self.labels[index],
            'clsname': 'isic',
            'mask': zeros if self.labels[index] == 0 else ones
        }
        return ret

    def __len__(self):
        return len(self.image_files)

