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
    def __init__(self, path_list):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

if __name__ == '__main__':
    x_0 = np.load("/home/xiangyuliu/Downloads/dataset/clahe/x_0_clahe.npy", mmap_mode="r")



