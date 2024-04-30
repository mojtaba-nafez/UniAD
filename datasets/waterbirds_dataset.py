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

logger = logging.getLogger("global_logger")



def build_waterbirds_dataloader(cfg, training, distributed=True):
    logger.info("building waterbirds dataset")
    transform = transforms.Compose([
                    transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),])
    import pandas as pd
    df = pd.read_csv('/kaggle/input/waterbird/waterbird/metadata.csv')
    if training:
        dataset = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df,
                                       transform=transform, train=True, count_train_landbg=3500,
                                       count_train_waterbg=100)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )
        return data_loader
    dataset_main = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df,
                                       transform=transform, train=False, count_train_landbg=3500,
                                       count_train_waterbg=100, mode='bg_land')
    dataset_shifted = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df,
                                       transform=transform, train=False, count_train_landbg=3500,
                                       count_train_waterbg=100, mode='bg_water')
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



class Waterbird(torch.utils.data.Dataset):
    def __init__(self, root, df, transform, train=True, count_train_landbg=-1, count_train_waterbg=-1, mode='bg_all',
                 count=-1, return_num=2):
        self.transform = transform
        self.train = train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []
        self.return_num = return_num

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])

        # if count != -1:
        #     random.shuffle(self.image_paths)
        #     if count < len(self.image_paths):
        #         self.image_paths = self.image_paths[:count]
        #         if not train:
        #             self.labels = self.labels[:count]
        #     else:
        #         t = len(self.image_paths)
        #         for i in range(count - t):
        #             self.image_paths.append(random.choice(self.image_paths[:t]))
        #             if not train:
        #                 self.labels.append(random.choice(self.labels[:t]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        gt[:, :, 1:3] = 1
        if self.train:
            return img, 0
        else:
            # if self.return_num == 2:
            #     return img, self.labels[idx]
            return img, gt, self.labels[idx], img_path