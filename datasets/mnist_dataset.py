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


def build_wbc_dataloader(cfg, training, distributed=True):
    logger.info("building wbc dataset")
    transform = transforms.Compose([
        transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ), ])
    import pandas as pd
    df1 = pd.read_csv('/kaggle/working/segmentation_WBC/Class Labels of Dataset 1.csv')
    df2 = pd.read_csv('/kaggle/working/segmentation_WBC/Class Labels of Dataset 2.csv')
    test_set1 = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1', '/kaggle/working/segmentation_WBC/Dataset 2',
                           df1, df2, transform=transform, train=False, test_id=1)
    test_set2 = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1', '/kaggle/working/segmentation_WBC/Dataset 2',
                           df1, df2, transform=transform, train=False, test_id=2)
    train_set = WBCDataset('/kaggle/working/segmentation_WBC/Dataset 1', '/kaggle/working/segmentation_WBC/Dataset 2',
                           df1, df2, transform=transform, train=True)

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


def three_digits(a: int):
    x = str(a)
    if len(x) == 1:
        return f'00{a}'
    if len(x) == 2:
        return f'0{a}'
    return x


class WBCDataset(torch.utils.data.Dataset):
    def __init__(self, root1, root2,
                 labels1: pd.DataFrame, labels2: pd.DataFrame, transform=None, train=True, test_id=1, ratio=0.7):
        self.transform = transform
        self.root1 = root1
        self.root2 = root2
        self.labels1 = labels1
        self.labels2 = labels2
        self.train = train
        self.test_id = test_id
        self.targets = []
        labels1 = labels1[labels1['class label'] != 5]
        labels2 = labels2[labels2['class'] != 5]

        normal_df = labels1[labels1['class label'] == 1]
        self.normal_paths = [os.path.join(root1, f'{three_digits(x)}.bmp') for x in list(normal_df['image ID'])]
        random.seed(42)
        random.shuffle(self.normal_paths)
        self.separator = int(ratio * len(self.normal_paths))
        self.train_paths = self.normal_paths[:self.separator]

        if self.train:
            self.image_paths = self.train_paths
            self.targets = [0] * len(self.image_paths)
        else:
            if self.test_id == 1:
                all_images = glob(os.path.join(root1, '*.bmp'))
                self.image_paths = [x for x in all_images if x not in self.train_paths]
                self.image_paths = [x for x in self.image_paths if
                                    int(os.path.basename(x).split('.')[0]) in labels1['image ID'].values]
                ids = [os.path.basename(x).split('.')[0] for x in self.image_paths]
                ids_labels = list(labels1[labels1['image ID'] == int(x)]['class label'] for x in ids)
                self.targets = [0 if x.item() == 1 else 1 for x in ids_labels]
            else:
                self.image_paths = glob(os.path.join(root2, '*.bmp'))
                self.image_paths = [x for x in self.image_paths if int(os.path.basename(x).split('.')[0])
                                    in labels2['image ID'].values]
                self.targets = [
                    0 if labels2[labels2['image ID'] == int(os.path.basename(x).split('.')[0])]['class'].item() == 1
                    else 1 for x in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        height = img.shape[1]
        width = img.shape[2]
        target = 0 if self.train else self.targets[idx]

        ret = {
            'filename': os.path.basename(img_path),
            'image': img,
            'height': height,
            'width': width,
            'label': target,
            'clsname': 'waterbirds',
            'mask': torch.zeros((1, height, width)) if target == 0 else torch.ones((1, height, width))
        }
        return ret
