import numpy as np
import torch
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import json

PATH = "/home/xiangyuliu/Downloads/images"
EXCEL = "/home/xiangyuliu/Downloads/label.xlsx"


class DataLoader():
    def __init__(self, image_size, image_base_path, label_path):
        self.image_base_path = image_base_path
        self.label_path = label_path
        self.image_label_dict = json.load(open("data_dict.json", "r"))
        self.total_sample_num = len(self.image_label_dict.values())
        self.tfms = transforms.Compose([transforms.Resize(size=(image_size, image_size)),
                                        transforms.ToTensor(), ])

    def sample_minibatch(self, batch_size, transform=True):
        image_array = []
        label_array = []
        binary_label = []
        raw_image = np.random.choice(list(self.image_label_dict.keys()), size=batch_size + 5)
        for image_path in raw_image:
            image = os.path.splitext(image_path)[0]
            image = os.path.split(image)[1]
            if os.path.exists(os.path.join(self.image_base_path, image)):
                if transform:
                    image = self.tfms(Image.open(os.path.join(self.image_base_path, image))).unsqueeze(0)
                else:
                    image = transforms.ToTensor()(image)
                image_array.append(image)
                label_array.append(self.image_label_dict[image_path])
                binary_label.append(0 if self.image_label_dict[image_path] < 0.5 else 1)
            if len(image_array) == batch_size:
                break
        minibatch = torch.stack([image_array[i][0] for i in range(len(image_array))], dim=0)
        label_tensor = torch.tensor(label_array, dtype=torch.long)
        binary_label_tensor = torch.tensor(binary_label, dtype=torch.long)
        return minibatch, torch.cat((label_tensor.view(1, -1), binary_label_tensor.view(1, -1)), dim=0)

    def test_transform(self):
        image_batch, labels= self.sample_minibatch(50)
        for test_image, label in zip(image_batch, labels[0]):
            print(label)
            transform_image = self.tfms(test_image)

            true_image = transforms.ToPILImage()(test_image[0])
            true_image.show()

            transform_image = transforms.ToPILImage()(transform_image[0])
            transform_image.show()

    def load_label_dict(self):
        worksheet = pd.read_excel(self.label_path, sheet_name="Sheet2")
        return worksheet.set_index("path").to_dict()['level']


if __name__ == '__main__':
    dataset = DataLoader(PATH, EXCEL, 224)
    dataset.test_transform()