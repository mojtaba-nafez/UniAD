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


def build_visa_dataloader(cfg, training, distributed=True, category='candle'):
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



