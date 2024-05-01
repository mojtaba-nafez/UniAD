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


def build_aptos_dataloader(cfg, training, distributed=True):
    logger.info("building aptos dataset")
    transform = transforms.Compose([
        transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ), ])

    train_path = glob('/kaggle/input/aptos-dataset/APTOS/train/NORMAL/*')
    train_label = [0] * len(train_path)
    test_anomaly_path = glob('/kaggle/input/aptos-dataset/APTOS/test/ABNORMAL/*')
    test_anomaly_label = [1] * len(test_anomaly_path)
    test_normal_path = glob('/kaggle/input/aptos-dataset/APTOS/test/NORMAL/*')
    test_normal_label = [0] * len(test_normal_path)

    test_label = test_anomaly_label + test_normal_label
    test_path = test_anomaly_path + test_normal_path

    df = pd.read_csv('/kaggle/input/ddrdataset/DR_grading.csv')
    label = df["diagnosis"].to_numpy()
    path = df["id_code"].to_numpy()

    normal_path = path[label == 0]
    anomaly_path = path[label != 0]

    shifted_test_path = list(normal_path) + list(anomaly_path)
    shifted_test_label = [0] * len(normal_path) + [1] * len(anomaly_path)

    shifted_test_path = ["/kaggle/input/ddrdataset/DR_grading/DR_grading/" + s for s in shifted_test_path]

    if training:
        dataset = APTOS(image_path=train_path, labels=train_label, transform=transform)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )
        return data_loader
    dataset_main = APTOS(image_path=test_path, labels=test_label, transform=transform)
    dataset_shifted = APTOS(image_path=shifted_test_path, labels=shifted_test_label, transform=transform)
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


class APTOS(Dataset):
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
            'mask': torch.zeros((1, height, width)) if self.labels[index] == 0 else torch.ones((1, height, width))
        }
        return ret

    def __len__(self):
        return len(self.image_files)

