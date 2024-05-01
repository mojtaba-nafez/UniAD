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


def build_brain_dataloader(cfg, training, distributed=True):
    logger.info("building waterbirds dataset")
    transform = transforms.Compose([
        transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ), ])

    train_normal_path = glob('./Br35H/dataset/train/normal/*')
    test_normal_path = glob('./Br35H/dataset/test/normal/*')
    test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')
    test_path = test_normal_path + test_anomaly_path

    normal_train_brats = glob('./brats/dataset/train/normal/*')
    normal_test_brats = glob('./brats/dataset/test/normal/*')
    anomaly_brats = glob('./brats/dataset/test/anomaly/*')
    brats_test_path = normal_test_brats + anomaly_brats


    ### ADDING 150 Exposures
    random.seed(1)
    random_brats_images = random.sample(normal_train_brats, 150)
    train_normal_path.extend(random_brats_images)
    #########################


    if training:
        dataset = Brain_MRI(image_path=train_normal_path, labels=[0]*len(train_normal_path),
                            transform=transform)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )
        return data_loader
    dataset_main = Brain_MRI(image_path=test_path, labels=[0]*len(test_normal_path)+[1]*len(test_anomaly_path),
                            transform=transform)
    dataset_shifted = Brain_MRI(image_path=brats_test_path, labels=[0]*len(normal_test_brats)+[1]*len(anomaly_brats),
                            transform=transform)
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


class Brain_MRI(Dataset):
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
            'clsname': 'brain',
            'mask': torch.zeros((1, height, width)) if self.labels[index] == 0 else torch.ones((1, height, width))
        }
        return ret

    def __len__(self):
        return len(self.image_files)


