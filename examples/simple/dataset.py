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


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_list, transform = None):
        self.data = []
        if transform == None:
            self.tfms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        else:
            self.tfms = transform
        for path in path_list:
            self.data.append(np.load(path, mmap_mode="r"))

    def __getitem__(self, item):
        label = item // 2000
        index = item % 2000
        img = self.data[label][index]
        img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return 8000


if __name__ == '__main__':
    x_0 = np.load("/home/xiangyuliu/Downloads/dataset/clahe/x_0_clahe.npy", mmap_mode="r")
