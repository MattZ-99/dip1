import os
from torch.utils.data import Dataset
import torch

import pandas as pd
import numpy as np
from PIL import Image
import time
from random import shuffle

class Trainset(Dataset):
    def __init__(self, data_dir='../data/train', label_name="Color", transform=None):
        super().__init__()
        self.label_name = label_name
        self.data_path = os.path.join(data_dir, "data")
        self.label_path = os.path.join(data_dir, "label/sort")
        self.transform = transform

        self.data_list = sorted(os.listdir(self.data_path))
        self.data_list = [os.path.abspath(os.path.join(self.data_path, x)) for x in self.data_list]
        self.label_list = sorted(os.listdir(self.label_path))
        self.label_list = [os.path.abspath(os.path.join(self.label_path, x)) for x in self.label_list]
        self.file_len = len(self.data_list)
        # print(self.file_len)

    def __getitem__(self, index):
        path_label = self.label_list[index]
        pd_data = pd.read_csv(path_label)[self.label_name]
        label = np.array(1 - (pd_data / 15))
        # label = np.array(pd_data)
        label = torch.from_numpy(label)
        # print(label)

        data_dir_path = self.data_list[index]
        path_list = sorted(os.listdir(data_dir_path))
        path_list = [os.path.join(data_dir_path, x) for x in path_list]
        data_init = Image.open(path_list[0])
        if self.transform:
            data_init = self.transform(data_init)
        for i in range(len(path_list) - 1):
            data = Image.open(path_list[i + 1])
            if self.transform:
                data = self.transform(data)
            data_init = torch.cat((data_init, data), 0)
        return data_init, label

    def __len__(self):
        return self.file_len


class Validset(Dataset):
    def __init__(self, data_dir='../data/valid', label_name="Color", transform=None):
        super().__init__()
        self.label_name = label_name
        self.data_path = os.path.join(data_dir, "data")
        self.label_path = os.path.join(data_dir, "label/sort")
        self.transform = transform

        self.data_list = sorted(os.listdir(self.data_path))
        self.data_list = [os.path.abspath(os.path.join(self.data_path, x)) for x in self.data_list]
        self.label_list = sorted(os.listdir(self.label_path))
        self.label_list = [os.path.abspath(os.path.join(self.label_path, x)) for x in self.label_list]
        self.file_len = len(self.data_list)
        # print(self.file_len)

    def __getitem__(self, index):
        path_label = self.label_list[index]
        pd_data = pd.read_csv(path_label)[self.label_name]
        label = np.array(pd_data)
        label = torch.from_numpy(label)

        data_dir_path = self.data_list[index]
        path_list = sorted(os.listdir(data_dir_path))
        path_list = [os.path.join(data_dir_path, x) for x in path_list]
        data_init = Image.open(path_list[0])
        if self.transform:
            data_init = self.transform(data_init)
        for i in range(len(path_list) - 1):
            data = Image.open(path_list[i + 1])
            if self.transform:
                data = self.transform(data)
            data_init = torch.cat((data_init, data), 0)
        return data_init, label

    def __len__(self):
        return self.file_len


class Testset(Dataset):
    def __init__(self, data_dir='../data/test', label_name="Color", transform=None):
        super().__init__()
        self.label_name = label_name
        self.data_path = os.path.join(data_dir, "data")
        self.transform = transform

        self.data_list = sorted(os.listdir(self.data_path))
        self.data_list = [os.path.abspath(os.path.join(self.data_path, x)) for x in self.data_list]

        self.file_len = len(self.data_list)

    def __getitem__(self, index):
        data_dir_path = self.data_list[index]
        path_list = sorted(os.listdir(data_dir_path))
        path_list = [os.path.join(data_dir_path, x) for x in path_list]
        data_init = Image.open(path_list[0])
        if self.transform:
            data_init = self.transform(data_init)
        for i in range(len(path_list) - 1):
            data = Image.open(path_list[i + 1])
            if self.transform:
                data = self.transform(data)
            data_init = torch.cat((data_init, data), 0)

        return data_init, data_dir_path

    def __len__(self):
        return self.file_len
