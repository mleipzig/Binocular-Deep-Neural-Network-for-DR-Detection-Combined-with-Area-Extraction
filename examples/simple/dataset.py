import numpy as np
import torch
from torchvision import transforms
from PIL import Image

PATH = "/home/xiangyuliu/Downloads/images"
EXCEL = "/home/xiangyuliu/Downloads/label.xlsx"
# pathlist = ["/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_0_clahe.npy",
#             "/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_1_clahe.npy",
#             "/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_2_clahe.npy",
#             "/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_3_clahe.npy",
#             "/home/xiangyuliu/mnt//newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_4_clahe.npy"]

pathlist = ["/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_0_clahe.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_1_clahe.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_2_clahe.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_3_clahe.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_4_clahe.npy"]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_list, transform=None, img_size=300, model_type="five"):
        self.data = []
        self.len_array = []
        self.img_size = img_size
        self.path_list = path_list
        self.kinds = 4 if model_type == "four" else 5
        if self.kinds == 4:
            self.path_list = self.path_list[1:]
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
        for label in range(self.kinds):
            for index in range(20):
                index = index - 20
                image_list.append(self.image_transform(self.data[label][index]))
                label_list.append(label)
        return torch.stack(image_list, dim=0), torch.tensor(np.array(label_list), dtype=torch.long)

    def image_transform(self, img):
        img = Image.fromarray(np.uint8(img))
        img = transforms.Resize(size=(self.img_size, self.img_size))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.341044359, 0.2591041, 0.2165859],
                                   [0.001024714638 * 255, 0.00083609472 * 255, 0.0008257308 * 255])(img)
        return img

    def calculate_mean_std(self):
        for i in range(len(self.data)):
            self.data[i] = np.uint8(self.data[i])
        whole_dataset = np.concatenate(tuple(self.data), axis=0)
        for i in range(3):
            print(np.mean(whole_dataset[:, :, :, i]) / 255, np.std(whole_dataset[:, :, :, i]) / 255)


if __name__ == '__main__':
    dataset = CustomDataset(pathlist)
    dataset.calculate_mean_std()
