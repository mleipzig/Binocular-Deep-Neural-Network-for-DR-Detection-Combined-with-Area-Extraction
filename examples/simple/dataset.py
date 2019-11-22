import numpy as np
import torch
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import json
import time

PATH = "/home/xiangyuliu/Downloads/images"
EXCEL = "/home/xiangyuliu/Downloads/label.xlsx"
pathlist = ["/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_0_clahe.npy",
            "/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_1_clahe.npy",
            "/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_2_clahe.npy",
            "/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_3_clahe.npy",
            "/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_4_clahe.npy"]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_list, transform=None, img_size=300):
        self.data = []
        self.len_array = []
        self.img_size = img_size
        if transform == None:
            self.tfms = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        else:
            self.tfms = transform
        for path, i in zip(path_list, range(len(path_list))):
            self.data.append(np.load(path, mmap_mode="r"))
            self.len_array.append(self.data[i].shape[0] - 20)
        for i in range(len(self.len_array)):
            if i == 0:
                continue
            self.len_array[i] = self.len_array[i] + self.len_array[i - 1]
        self.test_batch, self.test_label = self._fetch_test_data()

    def __getitem__(self, item):
        pre_len = 0
        for length, i in zip(self.len_array, range(len(self.len_array))):
            if item <= length - 1:
                label = i
                index = item - pre_len
                break
            pre_len = self.len_array[i]

        img = self.data[label][index]
        img = self.image_transform(img)
        return img, label

    def __len__(self):
        return self.len_array[-1]

    def _fetch_test_data(self):
        image_list = []
        label_list = []
        for label in range(5):
            for index in range(20):
                index = index - 20
                image_list.append(self.image_transform(self.data[label][index]))
                label_list.append(label)
        return torch.stack(image_list, dim=0), torch.tensor(np.array(label_list), dtype=torch.long)

    def image_transform(self, img):
        img = Image.fromarray(np.uint8(img))
        img = transforms.Resize(size=(self.img_size, self.img_size))(img)
        img = transforms.ToTensor()(img)
        return img


if __name__ == '__main__':
    dataset = CustomDataset(pathlist)
    print(dataset.__getitem__(0))
