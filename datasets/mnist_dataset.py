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


def build_mnist_dataloader(cfg, training, distributed=True):
    logger.info("building wbc dataset")
    transform = transforms.Compose([
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


class MNIST_Dataset(Dataset):
    def __init__(self, train, test_id=1, transform=None):
        self.transform = transform
        self.train = train
        self.test_id = test_id
        if train:
            with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/train_normal.pkl', 'rb') as f:
                normal_train = pickle.load(f)
            self.images = normal_train['images']
            self.labels = [0]*len(self.images)
        else:
            if test_id == 1:
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_normal_main.pkl', 'rb') as f:
                    normal_test = pickle.load(f)
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_abnormal_main.pkl', 'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0]*len(normal_test['images']) + [1]*len(abnormal_test['images'])
            else:
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_normal_shifted.pkl', 'rb') as f:
                    normal_test = pickle.load(f)
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_abnormal_shifted.pkl', 'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0]*len(normal_test['images']) + [1]*len(abnormal_test['images'])

    def __getitem__(self, index):
        image = torch.tensor(self.images[index])

        if self.transform is not None:
            image = self.transform(image)

        height = image.shape[1]
        width = image.shape[2]
        target = 0 if self.train else self.labels[index]

        ret = {
            'filename': f'{self.train}_{self.test_id}_{index}',
            'image': image,
            'height': height,
            'width': width,
            'label': target,
            'clsname': 'mnist',
            'mask': torch.zeros((1, height, width)) if target == 0 else torch.ones((1, height, width))
        }

        return image, self.labels[index]

    def __len__(self):
        return len(self.images)