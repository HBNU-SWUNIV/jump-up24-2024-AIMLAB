# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
# import PIL

import torch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import Dataset
import PIL
from PIL import Image
import pandas as pd


def set_protocols(protocol, is_debug=False):
    train_protocol, test_protocol = protocol.split('_to_')
    train_protocol = train_protocol.replace('_', '')
    
    train_protocol = 'train_' + train_protocol + '.csv'
    test_protocol = 'test_' + test_protocol + '.csv'

    if is_debug:
        train_protocol = f'./csv/for_debug/{train_protocol}'
        test_protocol = f'./csv/for_debug/{test_protocol}'
    else:
        train_protocol = f'./csv/{train_protocol}'
        test_protocol = f'./csv/{test_protocol}'
    return train_protocol, test_protocol

def extract_domain_and_label(row):
        domain = row['image_name'].split('/')[5].split('_')[1]
        label = 'Live' if row['label']==1 else 'Spoof'
        # print(f'domain: {domain}, label: {label}')

        return f"{domain}_{label}"

def set_protocols_for_oversampling(protocol):
    train_protocol, test_protocol = protocol.split('_to_')
    train_protocol = train_protocol.replace('_', '')
    
    train_protocol = 'train_' + train_protocol + '.csv'
    test_protocol = 'test_' + test_protocol + '.csv'

    train_protocol = f'./csv/{train_protocol}'
    test_protocol = f'./csv/{test_protocol}'

    train_df = pd.read_csv(train_protocol)
    train_df['domain_label'] = train_df.apply(extract_domain_and_label, axis=1)
    
    domain_label_dfs_list = []
    # print(f"train df list: {sorted(train_df['domain_label'].unique())}")
    for domain_label in sorted(train_df['domain_label'].unique()):
        domain_label_dfs_list.append(train_df[train_df['domain_label'] == domain_label])
    
    return domain_label_dfs_list, test_protocol

class MyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img_path = img_path.replace('/face_detecting/data_frame/SiW-Mv2/', '/jimin/dataset/FAS/SiW_Mv2_Dataset/')
        img_path = img_path.replace('.jpg', '_crop0.png')

        image_x = Image.open(img_path).convert("RGB")      
        image_x = self.transform(image_x)

        label = self.df.iloc[idx, 1]
        return image_x, label

class MyDataset_2Aug(Dataset):
    def __init__(self, df, transform=None, transform2=None):
        self.df = df
        self.transform = transform
        self.transform2 = transform2
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img_path = img_path.replace('/face_detecting/data_frame/SiW-Mv2/', '/jimin/dataset/FAS/SiW_Mv2_Dataset/')
        img_path = img_path.replace('.jpg', '_crop0.png')

        image_x = Image.open(img_path).convert("RGB")      
        image_x_v1 = self.transform(image_x)
        image_x_v2 = self.transform2(image_x)

        label = self.df.iloc[idx, 1]
        return image_x_v1, image_x_v2, label


class MyDataset_SiW_Detailed_Attack(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img_path = img_path.replace('/face_detecting/data_frame/SiW-Mv2/', '/jimin/dataset/FAS/SiW_Mv2_Dataset/')
        img_path = img_path.replace('.jpg', '_crop0.png')

        if img_path.split('/')[6].startswith('Live'):
            if self.is_train:
                attack_label = 'train_live'
            else:
                attack_label = 'test_live'

        elif img_path.split('/')[6].startswith('Spoof'):
            attack_label = img_path.split('/')[7]
        else:
            raise

        image_x = Image.open(img_path).convert("RGB")      
        image_x = self.transform(image_x)

        label = self.df.iloc[idx, 1]
        return image_x, label, attack_label